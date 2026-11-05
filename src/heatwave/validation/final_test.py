"""The single final test, gate G4 (reference.md §3.2, §3.7).

The pass bar is computed from development out-of-fold rows only and committed before
the test (`development_bar`). `score` then applies the frozen model, unchanged, to the
requested years. Rows are cut to those years when read, so nothing later is touched.

Skill varies a lot between 5-year periods, so "rough size" means inside the range seen
across 5-year development periods, not close to the 15-year average. Calibrators are
cross-fitted for the bar, because at test time the frozen calibrator is out of sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heatwave.config import Problem
from heatwave.features.pipeline import load_climatology
from heatwave.features.targets import in_window, per_lead_table, window_table
from heatwave.models.frozen import TIERS, FrozenModel
from heatwave.validation import calibration, metrics
from heatwave.validation.report import _md, _wide
from heatwave.validation.walk_forward import COLUMNS

PERIOD_YEARS = 5
SKILLS = ["window_all", "window_onset", "lead1_all", "lead1_onset"]
TIER_SCORES = [f"{t}_{s}" for t in ("watch", "alert") for s in ("pod", "far")]


def bss(d: pd.DataFrame, col: str) -> float:
    return float(1 - ((d[col] - d["y"]) ** 2).sum() / ((d["p_M0"] - d["y"]) ** 2).sum())


def tier_scores(window: pd.DataFrame, col: str, thresholds: dict[str, float]) -> dict[str, float]:
    p, y = window[col].to_numpy(dtype=float), window["y"].to_numpy(dtype=float)
    out = {}
    for name, t in thresholds.items():
        s = calibration.scores(p, y, t)
        out[f"{name}_pod"], out[f"{name}_far"] = s["pod"], s["far"]
    return out


def measures(lead: pd.DataFrame, window: pd.DataFrame, col: str, tier_col: str, thresholds) -> dict:
    one = lead[lead["lead"] == 1]
    return {
        "window_all": bss(window, col),
        "window_onset": bss(window[window["onset"]], col),
        "lead1_all": bss(one, col),
        "lead1_onset": bss(one[one["onset"]], col),
        "forecast_gap": float(abs(window[col].mean() - window["y"].mean())),
        **tier_scores(window, tier_col, thresholds),
    }


def development_bar(spec: dict, lead: pd.DataFrame, window: pd.DataFrame) -> dict:
    """The range of each measure over every 5-year development period."""
    lead, window = lead.copy(), window.copy()
    lead["p_x"] = calibration.cross_fit(lead, "p_M2S", "lead")
    window["p_x"] = calibration.cross_fit(window, "p_M2S", None)
    cal = spec["window"]["calibrator"]
    window["p_f"] = FrozenModel._calibrate(cal, window["p_M2S"].to_numpy(dtype=float))
    thresholds = {t: spec["tiers"][t]["threshold"] for t in TIERS}
    rows = []
    for first in range(int(window["year"].min()), int(window["year"].max()) - PERIOD_YEARS + 2):
        last = first + PERIOD_YEARS - 1
        keep_l, keep_w = lead["year"].between(first, last), window["year"].between(first, last)
        rows.append(
            {
                "period": f"{first}-{last}",
                **measures(lead[keep_l], window[keep_w], "p_x", "p_f", thresholds),
            }
        )
    t = pd.DataFrame(rows)
    return {
        "version": spec["version"],
        "basis": "5-year development periods, 2006-2020; other-fold calibrators for skill",
        "periods": t.to_dict("records"),
        "range": {
            c: {"min": float(t[c].min()), "max": float(t[c].max())}
            for c in [*SKILLS, *TIER_SCORES, "forecast_gap"]
        },
    }


def judge(result: dict[str, float], bar: dict) -> dict:
    """FAIL: a main skill is not positive, or below the lowest development period.
    CONCERN (a pass, reported as such): a skill above the highest period, which could
    mean a leak; a Watch or Alert score, or the forecast bias, outside its range."""
    r = bar["range"]
    fails, concerns = [], []
    for k in SKILLS:
        v = result[k]
        if not v > 0:
            fails.append(f"{k} is {v:.4f}: not above climatology")
        elif v < r[k]["min"]:
            fails.append(f"{k} is {v:.4f}: below the lowest development period, {r[k]['min']:.4f}")
        elif v > r[k]["max"]:
            concerns.append(f"{k} is {v:.4f}: above the highest period, {r[k]['max']:.4f}")
    for k in TIER_SCORES:
        v = result[k]
        if not r[k]["min"] <= v <= r[k]["max"]:
            concerns.append(f"{k} is {v:.3f}: outside {r[k]['min']:.3f} to {r[k]['max']:.3f}")
    if result["forecast_gap"] > r["forecast_gap"]["max"]:
        gap, top = result["forecast_gap"], r["forecast_gap"]["max"]
        concerns.append(f"window forecast bias is {gap:.3f}: above the largest period, {top:.3f}")
    verdict = "FAIL" if fails else "PASS WITH CONCERN" if concerns else "PASS"
    return {"verdict": verdict, "fails": fails, "concerns": concerns}


def load_rows(
    problem: Problem,
    root: Path,
    years: tuple[int, int],
    model: FrozenModel,
    region_map: dict[str, str],
    check_held_out: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scored rows of `years`, evaluation season only. Reads no later row."""
    clim = load_climatology(root / "features" / "development_climatology")
    if check_held_out:
        clim.check_held_out(years)
    start, end = (
        pd.Timestamp(year=years[0], month=1, day=1),
        pd.Timestamp(year=years[1] + 1, month=1, day=1),
    )
    feats = pd.read_parquet(
        root / "features" / "development.parquet",
        columns=["district_id", "asof_date", *COLUMNS],
        filters=[("asof_date", ">=", start), ("asof_date", "<", end)],
    )
    labels = pd.read_parquet(
        root / "labels" / "labels_daily.parquet",
        columns=["district_id", "obs_date", "is_heatwave"],
        filters=[("obs_date", ">=", start), ("obs_date", "<", end)],
    )
    h, L, season = problem.horizons, problem.latency_days, problem.evaluation_season

    lead = per_lead_table(feats, labels, clim, L, h, problem.training_season, COLUMNS)
    lead = lead[in_window(lead["valid_date"], season) & lead["valid_date"].dt.year.between(*years)]
    lead = lead.reset_index(drop=True)
    lead["p_M2F"] = model.predict_lead(lead)
    lead["p_M0"] = lead["clim_hw_freq_valid"].to_numpy(dtype=float)
    lead["year"], lead["month"] = lead["valid_date"].dt.year, lead["valid_date"].dt.month

    w = window_table(feats, labels, L, h, problem.training_season, COLUMNS)
    first, last = (w["issue_date"] + pd.Timedelta(days=d) for d in (h[0], h[-1]))
    w = w[
        in_window(first, season) & in_window(last, season) & w["issue_date"].dt.year.between(*years)
    ]
    w = w.reset_index(drop=True)
    w["p_M2F"] = model.predict_window(w)
    w["p_M0"] = model.m0.predict(w)
    w["year"], w["month"] = w["issue_date"].dt.year, w["issue_date"].dt.month

    for d in (lead, w):
        d["onset"] = d["days_since_hw"] >= problem.onset_lookback_days
        d["region"] = d["district_id"].map(region_map)
    return lead, w


def score(lead: pd.DataFrame, window: pd.DataFrame, spec: dict, bar: dict, cfg: dict) -> dict:
    thresholds = {t: spec["tiers"][t]["threshold"] for t in TIERS}
    result = measures(lead, window, "p_M2F", "p_M2F", thresholds)
    return {"result": result, "thresholds": thresholds, **judge(result, bar)}


def report(lead, window, outcome: dict, bar: dict, spec: dict, cfg: dict, title: str) -> str:
    e = cfg["evaluation"]
    args = (e["bootstrap_resamples"], e["bootstrap_seed"], e["ci"])
    m = ["M2F"]
    skill = lambda d, by: metrics.bss_table(d, m, "M0", by, *args)  # noqa: E731
    by_year = pd.DataFrame(
        [
            {
                "year": y,
                "rows": len(g),
                "observed": g["y"].mean(),
                "forecast": g["p_M2F"].mean(),
                "climatology": g["p_M0"].mean(),
                "bss": bss(g, "p_M2F"),
                **tier_scores(g, "p_M2F", outcome["thresholds"]),
            }
            for y, g in window.groupby("year")
        ]
    )
    r = bar["range"]
    res = outcome["result"]
    main = pd.DataFrame(
        [
            {"measure": k, "result": res[k], "dev_low": r[k]["min"], "dev_high": r[k]["max"]}
            for k in [*SKILLS, *TIER_SCORES, "forecast_gap"]
        ]
    )
    tiers = pd.DataFrame(
        [
            {"tier": t, "threshold": v, **calibration.scores(*_py(window), v)}
            for t, v in outcome["thresholds"].items()
        ]
    )
    lines = [f"- {x}" for x in outcome["fails"] + outcome["concerns"]] or ["- none"]
    n = metrics.discrimination(window, m, None)
    return (
        f"# {title}\n\n**Gate G4: {outcome['verdict']}**\n\nModel `{spec['version']}` unchanged. "
        f"Years {int(window['year'].min())}-{int(window['year'].max())}.\n\n"
        "Findings:\n\n"
        + "\n".join(lines)
        + "\n\n## Against the bar\n\n"
        + _md(main, 4)
        + "\n## Window, all days then onset days\n\n"
        + _md(skill(window, None).drop(columns="group"), 4)
        + "\n"
        + _md(skill(window[window["onset"]], None).drop(columns="group"), 4)
        + "\n## By lead, all days\n\n"
        + _md(_wide(skill(lead, "lead"), "lead"), 4)
        + "\n## By lead, onset days\n\n"
        + _md(_wide(skill(lead[lead["onset"]], "lead"), "lead"), 4)
        + "\n## By region (window)\n\n"
        + _md(_wide(skill(window, "region"), "region"), 4)
        + "\n## By month (window)\n\n"
        + _md(_wide(skill(window, "month"), "month"), 4)
        + "\n## By year (window): one season can dominate\n\n"
        + _md(by_year, 3)
        + "\n## Frozen tiers\n\n"
        + _md(tiers, 3)
        + "\n## Reliability (window)\n\n"
        + _md(metrics.reliability(window, "M2F", 10), 3)
        + "\n## Ranking (window)\n\n"
        + _md(n, 3)
    )


def _py(window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return window["p_M2F"].to_numpy(dtype=float), window["y"].to_numpy(dtype=float)
