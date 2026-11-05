"""Freeze the production model: M2 on the last 20 development years, Platt calibrators
from out-of-fold predictions, tier thresholds on the 10-day window (decision records 009, 013).

Reads development years only. The feature table also holds the final-test years, so
rows are cut to the development block here and a later row is an error: the final test
must stay unseen until the frozen model is scored on it.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from heatwave.config import Problem
from heatwave.features.pipeline import load_climatology
from heatwave.features.targets import per_lead_table, window_table
from heatwave.models import baselines
from heatwave.models.frozen import (
    TIERS,
    FrozenModel,
    calibrator_to_dict,
    check_matches,
    logistic_to_dict,
    sha256,
)
from heatwave.validation import calibration, metrics
from heatwave.validation.report import _md, _wide
from heatwave.validation.walk_forward import COLUMNS


def target_year(rows: pd.DataFrame, target: str) -> pd.Series:
    return (rows["valid_date"] if target == "lead" else rows["issue_date"]).dt.year


def development_rows(rows: pd.DataFrame, target: str, development: tuple[int, int]) -> pd.DataFrame:
    """Rows of the development years; raises if any later year survives the cut."""
    out = rows[target_year(rows, target).between(*development).to_numpy()].reset_index(drop=True)
    if target_year(out, target).max() > development[1]:
        raise ValueError("final-test rows reached the freeze step")
    return out


def training_tables(problem: Problem, root: Path) -> dict[str, pd.DataFrame]:
    v = problem.validation
    clim = load_climatology(root / "features" / "development_climatology")
    clim.check_held_out(v.final_test)
    labels = pd.read_parquet(
        root / "labels" / "labels_daily.parquet", columns=["district_id", "obs_date", "is_heatwave"]
    )
    labels = labels[
        labels["obs_date"].dt.year <= v.development[1] + 1
    ]  # window targets near 31 Dec
    feats = pd.read_parquet(
        root / "features" / "development.parquet", columns=["district_id", "asof_date", *COLUMNS]
    )
    feats = feats[feats["asof_date"].dt.year <= v.development[1]]
    h, L, season = problem.horizons, problem.latency_days, problem.training_season
    return {
        "lead": development_rows(
            per_lead_table(feats, labels, clim, L, h, season, COLUMNS), "lead", v.development
        ),
        "window": development_rows(
            window_table(feats, labels, L, h, season, COLUMNS), "window", v.development
        ),
    }


def production_cfg(cfg: dict) -> dict:
    """The model settings with the frozen model's own features."""
    out = copy.deepcopy(cfg)
    out["baselines"]["m2"]["features"] = list(cfg["frozen"]["features"])
    return out


def oof_col(cfg: dict) -> str:
    return f"p_{cfg['frozen']['oof_model']}"


def build(problem: Problem, cfg: dict, root: Path, out_dir: Path, version: str, code: str | None):
    """Fit everything, write the window climatology table to out_dir, return the spec
    (the JSON content, written by the caller) and the OOF tables."""
    cfg = production_cfg(cfg)
    col, oof_name = oof_col(cfg), cfg["frozen"]["oof"]
    m2, tiers_cfg = cfg["baselines"]["m2"], cfg["tier_support"]
    rules = {t.name: t.params for t in problem.tiers}
    years = int(m2["sliding_years"])
    last = problem.validation.development[1]
    first = last - years + 1
    tables = training_tables(problem, root)
    oof = {t: pd.read_parquet(root / "oof" / f"{oof_name}_{t}.parquet") for t in ("lead", "window")}

    lead = baselines.SlidingWindow(baselines.make("M2", "lead", cfg), "lead", years).fit(
        tables["lead"]
    )
    window = baselines.SlidingWindow(baselines.make("M2", "window", cfg), "window", years).fit(
        tables["window"]
    )

    # Calibrators from OOF rows only; thresholds on the production calibrator's scale.
    lead_part = {}
    for h, g in oof["lead"].groupby("lead"):
        cal = calibration.fit(g[col].to_numpy(dtype=float), g["y"].to_numpy(dtype=float))
        lead_part[str(int(h))] = {
            **logistic_to_dict(lead.model.models[int(h)]),
            "calibrator": calibrator_to_dict(cal),
        }
    w = oof["window"]
    p, y = w[col].to_numpy(dtype=float), w["y"].to_numpy(dtype=float)
    win_cal = calibration.fit(p, y)
    calibrated = calibration.apply(win_cal, p)
    found = calibration.thresholds(
        calibrated,
        y,
        w["year"].to_numpy(),
        rules,
        tiers_cfg["min_flagged"],
        tiers_cfg["min_seasons"],
    )
    order = [found[t] for t in TIERS]
    if any(np.isnan(order)) or order != sorted(order):
        raise ValueError(f"tier thresholds missing or out of order: {found}")

    out_dir.mkdir(parents=True, exist_ok=True)
    table_name = f"{version}_window_climatology.parquet"
    window.model.m0.table.to_parquet(out_dir / table_name, index=False)
    feats = root / "features" / "development.parquet"
    spec = {
        "version": version,
        "model": "M2 logistic regression without soil moisture, sliding window, Platt calibrated",
        "decision_record": "docs/decisions/013-provisional-soil-moisture.md",
        "code_version": code,
        "fit_years": [first, last],
        "fit_end_date": f"{last}-12-31",
        "calibration": {"method": "platt", "fit_on": "out-of-fold predictions, 2006-2020"},
        "feature_version": pd.read_parquet(feats, columns=["feature_version"]).iloc[0, 0],
        "label_version": pd.read_parquet(
            root / "labels" / "labels_daily.parquet", columns=["label_version"]
        ).iloc[0, 0],
        "inputs_sha256": {
            "features/development.parquet": sha256(feats),
            "labels/labels_daily.parquet": sha256(root / "labels" / "labels_daily.parquet"),
            f"oof/{oof_name}_lead.parquet": sha256(root / "oof" / f"{oof_name}_lead.parquet"),
            f"oof/{oof_name}_window.parquet": sha256(root / "oof" / f"{oof_name}_window.parquet"),
        },
        "features": list(m2["features"]),
        "clim_clip": list(m2["clim_clip"]),
        "m0_smooth_days": int(cfg["baselines"]["m0_window"]["smooth_days"]),
        "m0_overall": window.model.m0.overall,
        "m0_table": table_name,
        "m0_table_sha256": sha256(out_dir / table_name),
        "lead": lead_part,
        "window": {
            **logistic_to_dict(window.model.model),
            "calibrator": calibrator_to_dict(win_cal),
        },
        "tiers": {
            name: {**calibration.scores(calibrated, y, found[name]), "threshold": found[name]}
            for name in TIERS
        },
    }

    # The frozen numbers must reproduce what was fitted, before anything is relied on.
    frozen = FrozenModel.with_table_dir(spec, out_dir)
    rows = tables["lead"][tables["lead"]["lead"] == 1].head(5000)
    check_matches(
        lead.model.lr,
        lead.model.models[1],
        lead_part["1"],
        rows,
        rows["clim_hw_freq_valid"].to_numpy(dtype=float),
        frozen,
    )
    rows = tables["window"].head(5000)
    check_matches(
        window.model.lr,
        window.model.model,
        spec["window"],
        rows,
        window.model.m0.predict(rows),
        frozen,
    )
    return spec, oof


def report(spec: dict, oof: dict, problem: Problem, cfg: dict, root: Path) -> str:
    """reports/model_<version>.md: the tables behind the frozen model, from the OOF files only.
    Decisions on earlier versions keep their own reports (reports/phase7_model.md)."""
    e, t_cfg = cfg["evaluation"], cfg["tier_support"]
    args = (e["bootstrap_resamples"], e["bootstrap_seed"], e["ci"])
    rules = {t.name: t.params for t in problem.tiers}
    col = oof_col(cfg)
    lead, window = oof["lead"].copy(), oof["window"].copy()

    # Scored with other-fold calibrators. S keeps soil moisture; New is the frozen model.
    for name, source in (("p_New", col), ("p_S", "p_M2S")):
        lead[name] = calibration.cross_fit(lead, source, "lead")
        window[name] = calibration.cross_fit(window, source, None)
    pair = [("New", "S")]
    skill = lambda d, by: metrics.bss_table(d, ["S", "New"], "M0", by, *args, pairs=pair)  # noqa: E731
    win_all, win_onset = skill(window, None), skill(window[window["onset"]], None)
    lead_all, lead_onset = skill(lead, "lead"), skill(lead[lead["onset"]], "lead")
    means = {
        g: (lead.groupby(g)[["y", col, "p_New"]].mean() * 100).round(2).reset_index()
        for g in ("lead", "month", "fold")
    }

    # Tiers on the frozen scale: production calibrator for the thresholds, other-fold
    # thresholds for the held_* columns.
    win_cal = spec["window"]["calibrator"]
    w = window.assign(p_Frozen=FrozenModel._calibrate(win_cal, window[col].to_numpy(dtype=float)))
    tiers = calibration.tier_table(
        w, "p_Frozen", None, rules, t_cfg["min_flagged"], t_cfg["min_seasons"]
    ).drop(columns="group")
    cost = calibration.cost_curve(w, "p_Frozen", None, t_cfg["cost_ratios"]).drop(columns="group")
    daily = calibration.tier_table(
        lead, "p_New", "lead", rules, t_cfg["min_flagged"], t_cfg["min_seasons"]
    ).rename(columns={"group": "lead"})

    top = lead.groupby(["district_id", "issue_date"])["p_New"].max().rename("top").reset_index()
    gap = top.merge(window[["district_id", "issue_date", "p_New"]])
    gap = gap["top"] - gap["p_New"]

    coef = pd.DataFrame(
        [
            {
                "lead": h,
                **dict(zip([*spec["features"], "logit_clim"], part["coef"], strict=True)),
                "intercept": part["intercept"],
                "cal_slope": part["calibrator"]["slope"],
                "cal_intercept": part["calibrator"]["intercept"],
            }
            for h, part in [*spec["lead"].items(), ("window", spec["window"])]
        ]
    )
    s = spec
    return (
        f"# Frozen model {s['version']}\n\n"
        "Generated by `heatwave freeze-model`. Reasoning: docs/decisions/013. All scores are "
        "out-of-fold 2006-2020 with paired year-block intervals; "
        "**no later year was read.**\n\n"
        f"- Code: `{s['code_version']}`\n- Fit years: {s['fit_years'][0]}-{s['fit_years'][1]}\n"
        f"- Features: {', '.join(s['features'])}, plus the logit of climatology\n"
        f"- Feature table: `{s['feature_version']}`, labels: `{s['label_version']}`\n\n"
        "## Against the same model with soil moisture (S)\n\n`New-S` below 0 means the frozen "
        "model scores lower, which is the cost of leaving soil moisture out.\n\n"
        "### 10-day window: all days, then onset days\n\n"
        + _md(win_all.drop(columns="group"), 4)
        + "\n"
        + _md(win_onset.drop(columns="group"), 4)
        + "\n### By lead, all days\n\n"
        + _md(_wide(lead_all, "lead"), 4)
        + "\n### By lead, onset days\n\n"
        + _md(_wide(lead_onset, "lead"), 4)
        + "\n## Mean forecast against observed (%), by lead, month and fold\n\n"
        + "\n".join(_md(m, 2) for m in means.values())
        + "\n## Frozen tiers: 10-day window\n\n"
        "`held_*` uses thresholds fitted on the other folds.\n\n"
        + _md(tiers, 3)
        + "\n### Cost curve (miss : false alarm)\n\n"
        + _md(cost, 3)
        + "\n## Why not single-day tiers\n\n"
        + _md(daily, 3)
        + "\n## Window against the highest single-day forecast\n\n"
        f"Window lower in {(gap > 0).mean():.1%} "
        f"of rows, by {gap[gap > 0].mean():.4f} on average and {gap.max():.3f} at most.\n"
        + "\n## Frozen coefficients (standardised inputs) and calibrators\n\n"
        + _md(coef, 3)
    )
