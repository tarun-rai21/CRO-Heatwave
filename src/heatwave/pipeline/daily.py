"""One day's forecast: dates and failure policy, scoring, writing (reference.md §1.3, §5.4, §5.5).

Uses the same code as training: main_labels for label-scale Tmax, compute_features for
the as-of features, the development climatology and the frozen model. Nothing is fitted.

As in training, the model runs from as-of date t and its leads count from t + L. On a
late-data day t is earlier, so the published leads are the ones still ahead of the issue
date. Output for an issue date overwrites by key, and equal inputs give equal outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from heatwave.config import Problem
from heatwave.features.build import compute_features
from heatwave.features.climatology import Climatology
from heatwave.features.pipeline import load_climatology
from heatwave.features.targets import _subset, in_window
from heatwave.geo.aggregate import power_district_daily
from heatwave.ingest.raw_store import RawStore
from heatwave.labels.build import main_labels
from heatwave.models.frozen import FrozenModel
from heatwave.validation.walk_forward import COLUMNS

MAX_LATE_DAYS = 2  # beyond this, climatology only (reference.md §5.5)

KEY = ["issue_date", "target", "district_id", "valid_date"]


@dataclass(frozen=True)
class Plan:
    issue_date: pd.Timestamp
    newest_date: pd.Timestamp | None
    asof_date: pd.Timestamp | None  # None: degraded, no model forecast
    status: str  # ok, late_1, late_2, degraded
    reason: str
    model_leads: tuple[int, ...]  # leads as trained, counted from asof_date + L
    valid_dates: tuple[pd.Timestamp, ...]

    @property
    def data_age_days(self) -> int | None:
        return None if self.newest_date is None else (self.issue_date - self.newest_date).days


def make_plan(
    issue_date, newest_date, latency: int, horizons, validation_ok: bool = True, problem: str = ""
) -> Plan:
    """The late-data rule and failure policy for one issue date."""
    s = pd.Timestamp(issue_date).normalize()
    newest = None if newest_date is None else pd.Timestamp(newest_date).normalize()
    hs = tuple(horizons)
    climatology_dates = tuple(s + pd.Timedelta(days=h) for h in hs)

    def degraded(reason: str) -> Plan:
        return Plan(s, newest, None, "degraded", reason, (), climatology_dates)

    if not validation_ok:
        return degraded(f"validation failed: {problem}")
    if newest is None:
        return degraded("no observed data")
    late = (s - newest).days - latency
    if late > MAX_LATE_DAYS:
        return degraded(f"data {late} days late")
    # Fresher data than the trained latency is not used: t stays s - L, as in training.
    t = newest if late > 0 else s - pd.Timedelta(days=latency)
    anchor = t + pd.Timedelta(days=latency)
    leads = tuple(h for h in hs if anchor + pd.Timedelta(days=h) > s)
    return Plan(
        s,
        newest,
        t,
        f"late_{late}" if late > 0 else "ok",
        f"data {late} day{'s' if late > 1 else ''} late" if late > 0 else "",
        leads,
        tuple(anchor + pd.Timedelta(days=h) for h in leads),
    )


def district_statics(root: Path) -> pd.DataFrame:
    districts = pd.read_parquet(root / "reference" / "districts.parquet")
    w = pd.read_parquet(root / "reference" / "district_grid_weights.parquet").merge(
        pd.read_parquet(root / "reference" / "grid_cell_meta.parquet"), on="cell_id"
    )
    districts["elevation_m"] = districts["district_id"].map(
        (w["area_weight"] * w["elevation_m"]).groupby(w["district_id"]).sum()
    )
    return districts


def recent_inputs(
    store: RawStore, root: Path, data_cfg: dict, problem: Problem, districts: pd.DataFrame, years
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """District daily inputs on the label's scale, and labels, for the given years. The
    label's normal is fitted on its normal period, so Tmax alone is read for those years."""
    weights = pd.read_parquet(root / "reference" / "district_grid_weights.parquet")
    share = data_cfg["geo"]["cellmax_min_share"]
    power = power_district_daily(store, weights, data_cfg["power"]["parameters"], share, years)
    first, last = problem.target.normal_period
    normal_years = [y for y in store.years() if first <= y <= last or y in years]
    tmax = power_district_daily(store, weights, ["T2M_MAX"], share, normal_years)
    ref = root / "reference"
    labels = main_labels(
        tmax,
        pd.read_parquet(ref / "label_qmap.parquet"),
        problem.target,
        json.loads((ref / "label_bias_plane.json").read_text()),
        districts,
        json.loads((ref / "label_bias_drift.json").read_text()),
    )
    labels = labels[labels["obs_date"].dt.year.isin(years)].reset_index(drop=True)
    daily = power.merge(
        labels[["district_id", "obs_date", "tmax_used"]], on=["district_id", "obs_date"]
    ).rename(columns={"tmax_used": "tmax"})
    return daily, labels


def newest_complete_date(power: pd.DataFrame, n_districts: int) -> pd.Timestamp | None:
    """Latest date with Tmax for every district."""
    seen = power.dropna(subset=["T2M_MAX"]).groupby("obs_date")["district_id"].nunique()
    full = seen[seen == n_districts]
    return None if full.empty else pd.Timestamp(full.index.max())


def lead_rows(asof_rows: pd.DataFrame, plan: Plan, latency: int, clim: Climatology) -> pd.DataFrame:
    """One row per (district, published lead), as the per-lead training table has them."""
    anchor = plan.asof_date + pd.Timedelta(days=latency)
    parts = []
    for h, valid in zip(plan.model_leads, plan.valid_dates, strict=True):
        f = asof_rows.copy()
        f["issue_date"], f["valid_date"], f["lead"] = anchor, valid, h
        f["clim_hw_freq_valid"] = clim.frequency(f["district_id"], f["valid_date"])
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def _base(plan: Plan, district_ids, label_version, model_version) -> dict:
    return {
        "issue_date": plan.issue_date,
        "asof_date": plan.asof_date if plan.asof_date is not None else pd.NaT,
        "status": plan.status,
        "reason": plan.reason,
        "data_age_days": plan.data_age_days if plan.data_age_days is not None else -1,
        "model_version": model_version,
        "label_version": label_version,
        "district_id": np.asarray(district_ids),
    }


def score(
    plan: Plan,
    districts: list[str],
    features: pd.DataFrame | None,
    clim: Climatology,
    model: FrozenModel,
    problem: Problem,
    label_version: str | None = None,
) -> pd.DataFrame:
    """Forecast rows for one issue date: `target` is lead or window. Tiers are issued on
    the window only, and only when it lies in the season they were evaluated on. Dates
    outside the training window get no probability."""
    L, hs = problem.latency_days, list(problem.horizons)
    season = problem.evaluation_season
    version = model.spec["version"]
    ids = sorted(districts)
    out = []
    if plan.asof_date is None:  # degraded: climatology, never yesterday's forecast
        for valid in plan.valid_dates:
            p = clim.frequency(ids, [valid] * len(ids))
            out.append(
                pd.DataFrame(
                    {
                        **_base(plan, ids, label_version, version),
                        "target": "lead",
                        "model_lead": -1,
                        "valid_date": valid,
                        "window_end": pd.NaT,
                        "p": p,
                        "p_clim": p,
                        "tier": "none",
                    }
                )
            )
        first, last = (
            plan.issue_date + pd.Timedelta(days=hs[0]),
            plan.issue_date + pd.Timedelta(days=hs[-1]),
        )
        q = pd.DataFrame({"district_id": ids, "issue_date": plan.issue_date})
        p = model.m0.predict(q)
        tier = np.full(len(ids), "none", dtype=object)
    else:
        rows = features[features["asof_date"] == plan.asof_date]
        rows = rows.sort_values("district_id").reset_index(drop=True)
        if rows["district_id"].tolist() != ids:
            raise ValueError(f"features for {plan.asof_date.date()} do not cover every district")
        lead = lead_rows(rows, plan, L, clim)
        out.append(
            pd.DataFrame(
                {
                    **_base(plan, lead["district_id"], label_version, version),
                    "target": "lead",
                    "model_lead": lead["lead"].to_numpy(),
                    "valid_date": lead["valid_date"].to_numpy(),
                    "window_end": pd.NaT,
                    "p": model.predict_lead(lead),
                    "p_clim": lead["clim_hw_freq_valid"].to_numpy(dtype=float),
                    "tier": "none",
                }
            )
        )
        anchor = plan.asof_date + pd.Timedelta(days=L)
        q = rows.assign(issue_date=anchor)
        first, last = anchor + pd.Timedelta(days=hs[0]), anchor + pd.Timedelta(days=hs[-1])
        p = model.predict_window(q)
        tier = model.tier(p)
    in_season = bool(in_window([first], season)[0] and in_window([last], season)[0])
    if not in_season:
        tier = np.full(len(ids), "none", dtype=object)
    out.append(
        pd.DataFrame(
            {
                **_base(plan, ids, label_version, version),
                "target": "window",
                "model_lead": 0,
                "valid_date": first,
                "window_end": last,
                "p": p,
                "p_clim": model.m0.predict(q),
                "tier": tier,
            }
        )
    )
    result = pd.concat(out, ignore_index=True)
    is_window = result["target"] == "window"
    result["in_season"] = np.where(is_window, in_season, in_window(result["valid_date"], season))
    # Outside the training window the model and its climatology have no basis (the window
    # climatology would fall back to an overall average): no probability, never a guess.
    trained = problem.training_season
    window_known = bool(in_window([first], trained)[0] and in_window([last], trained)[0])
    known = np.where(is_window, window_known, in_window(result["valid_date"], trained))
    result.loc[~known, ["p", "p_clim"]] = np.nan
    return result.sort_values(KEY).reset_index(drop=True)


def forecast_path(root: Path, issue_date) -> Path:
    s = pd.Timestamp(issue_date)
    return root / "forecasts" / f"issue_year={s.year}" / f"{s.date()}.parquet"


def write(rows: pd.DataFrame, path: Path) -> None:
    """Replace by key: a same-day re-run overwrites the whole file for that issue date."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    rows.to_parquet(tmp, index=False)
    tmp.replace(path)


def run(
    problem: Problem,
    data_cfg: dict,
    feature_cfg: dict,
    model_cfg: dict,
    root: Path,
    model_path: Path,
    issue_date,
    validation_ok: bool = True,
    validation_problem: str = "",
) -> tuple[Plan, pd.DataFrame]:
    """Plan and score one issue date. Reads at most the current and previous calendar years
    of the raw store's variables, which covers every feature's look-back."""
    s = pd.Timestamp(issue_date).normalize()
    frozen = model_cfg["frozen"]
    model = FrozenModel.load(model_path, root / frozen["table_dir"])
    clim = load_climatology(root / "features" / "development_climatology")
    clim.check_held_out((problem.validation.final_test[0], s.year))
    districts = district_statics(root)
    store = RawStore.from_config(data_cfg, root)
    years = [y for y in (s.year - 1, s.year) if y in store.years()]
    daily, labels = recent_inputs(store, root, data_cfg, problem, districts, years)
    daily = daily[daily["obs_date"] <= s]
    labels = labels[labels["obs_date"] <= s]
    newest = newest_complete_date(daily, len(districts))
    plan = make_plan(
        s,
        newest,
        problem.latency_days,
        problem.horizons,
        validation_ok,
        validation_problem,
    )
    features = None
    if plan.asof_date is not None:
        # Nothing dated after the as-of date reaches the features.
        d = daily[daily["obs_date"] <= plan.asof_date]
        lab = labels[labels["obs_date"] <= plan.asof_date]
        features = compute_features(d, lab, clim, districts, feature_cfg)
        features = features[features["asof_date"] == plan.asof_date]
        features = _subset(features, COLUMNS)  # same precision as the training tables
    return plan, score(
        plan,
        districts["district_id"].tolist(),
        features,
        clim,
        model,
        problem,
        labels["label_version"].iloc[0],
    )
