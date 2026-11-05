"""Freeze the production model: M2 on the last 20 development years, Platt calibrators
from out-of-fold predictions, tier thresholds on the 10-day window (decision record 009).

Reads development years only. The feature table also holds the final-test years, so
rows are cut to the development block here and a later row is an error: the final test
must stay unseen until the frozen model is scored on it.
"""

from __future__ import annotations

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
from heatwave.validation import calibration, compare, metrics
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


def build(problem: Problem, cfg: dict, root: Path, out_dir: Path, version: str, code: str | None):
    """Fit everything, write the window climatology table to out_dir, return the spec
    (the JSON content, written by the caller) and the OOF tables."""
    m2, tiers_cfg = cfg["baselines"]["m2"], cfg["tier_support"]
    rules = {t.name: t.params for t in problem.tiers}
    years = int(m2["sliding_years"])
    last = problem.validation.development[1]
    first = last - years + 1
    tables = training_tables(problem, root)
    oof = {t: pd.read_parquet(root / "oof" / f"m2_sliding_{t}.parquet") for t in ("lead", "window")}

    lead = baselines.SlidingWindow(baselines.make("M2", "lead", cfg), "lead", years).fit(
        tables["lead"]
    )
    window = baselines.SlidingWindow(baselines.make("M2", "window", cfg), "window", years).fit(
        tables["window"]
    )

    # Calibrators from OOF rows only; thresholds on the production calibrator's scale.
    lead_part = {}
    for h, g in oof["lead"].groupby("lead"):
        cal = calibration.fit(g["p_M2S"].to_numpy(dtype=float), g["y"].to_numpy(dtype=float))
        lead_part[str(int(h))] = {
            **logistic_to_dict(lead.model.models[int(h)]),
            "calibrator": calibrator_to_dict(cal),
        }
    w = oof["window"]
    p, y = w["p_M2S"].to_numpy(dtype=float), w["y"].to_numpy(dtype=float)
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
        "model": "M2 logistic regression, sliding window, Platt calibrated",
        "decision_record": "docs/decisions/009-phase7-model-choice.md",
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
            "oof/m2_sliding_lead.parquet": sha256(root / "oof" / "m2_sliding_lead.parquet"),
            "oof/m2_sliding_window.parquet": sha256(root / "oof" / "m2_sliding_window.parquet"),
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
    """reports/phase7_model.md: everything behind decision record 009, from the OOF files."""
    e, t_cfg = cfg["evaluation"], cfg["tier_support"]
    args = (e["bootstrap_resamples"], e["bootstrap_seed"], e["ci"])
    rules = {t.name: t.params for t in problem.tiers}
    keys = {"lead": ["district_id", "issue_date", "lead"], "window": ["district_id", "issue_date"]}
    lead, window = oof["lead"].copy(), oof["window"].copy()

    # G3: default and tuned LightGBM against the same M2 (expanding, as the gate was run).
    g3 = {}
    tuned = pd.read_parquet(
        root / "oof" / "m3_tuned_lead.parquet", columns=[*keys["lead"], "p_M3T"]
    )
    base = pd.read_parquet(root / "oof" / "m3_default_lead.parquet")
    both = base.merge(tuned, on=keys["lead"], validate="1:1")
    if len(both) != len(base):
        raise ValueError("default and tuned LightGBM rows differ")
    for name in ("M3", "M3T"):
        g3[name] = compare.g3(both, cfg, challenger=name)

    # Sliding against expanding, and calibration, scored with other-fold calibrators.
    sliding = compare.g3(lead, cfg, challenger="M2S", champion="M2")
    lead["p_Cal"] = calibration.cross_fit(lead, "p_M2S", "lead")
    window["p_Cal"] = calibration.cross_fit(window, "p_M2S", None)
    pair = [("Cal", "M2S")]
    cal_all = metrics.bss_table(lead, ["M2S", "Cal"], "M0", "lead", *args, pairs=pair)
    cal_onset = metrics.bss_table(
        lead[lead["onset"]], ["M2S", "Cal"], "M0", "lead", *args, pairs=pair
    )
    win_all = metrics.bss_table(window, ["M2S", "Cal"], "M0", None, *args, pairs=pair)
    win_onset = metrics.bss_table(
        window[window["onset"]], ["M2S", "Cal"], "M0", None, *args, pairs=pair
    )
    means = {
        g: (lead.groupby(g)[["y", "p_M2S", "p_Cal"]].mean() * 100).round(2).reset_index()
        for g in ("lead", "month", "fold")
    }

    # Tiers on the frozen scale: production calibrator for the thresholds, other-fold
    # thresholds for the held_* columns.
    win_cal = spec["window"]["calibrator"]
    w = window.assign(
        p_Frozen=FrozenModel._calibrate(win_cal, window["p_M2S"].to_numpy(dtype=float))
    )
    tiers = calibration.tier_table(
        w, "p_Frozen", None, rules, t_cfg["min_flagged"], t_cfg["min_seasons"]
    ).drop(columns="group")
    cost = calibration.cost_curve(w, "p_Frozen", None, t_cfg["cost_ratios"]).drop(columns="group")
    daily = calibration.tier_table(
        lead, "p_Cal", "lead", rules, t_cfg["min_flagged"], t_cfg["min_seasons"]
    ).rename(columns={"group": "lead"})

    top = lead.groupby(["district_id", "issue_date"])["p_Cal"].max().rename("top").reset_index()
    gap = top.merge(window[["district_id", "issue_date", "p_Cal"]])
    gap = gap["top"] - gap["p_Cal"]

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
        f"# Phase 7: model choice and frozen model ({s['version']})\n\n"
        "Generated by `heatwave freeze-model`. Reasoning: docs/decisions/009. All scores are "
        "out-of-fold 2006-2020 with paired year-block intervals; "
        "**no final-test year was read.**\n\n"
        f"- Code: `{s['code_version']}`\n- Fit years: {s['fit_years'][0]}-{s['fit_years'][1]}\n"
        f"- Features: `{s['feature_version']}`, labels: `{s['label_version']}`\n\n"
        "## Gate G3 (onset subset): LightGBM minus logistic regression\n\n"
        f"Default settings: {'passed' if g3['M3']['passed'] else 'failed'}. "
        f"Tuned, fold-honest settings: {'passed' if g3['M3T']['passed'] else 'failed'}.\n\n"
        + _md(g3["M3T"]["table"].drop(columns="model"), 4)
        + "\n### Default settings\n\n"
        + _md(g3["M3"]["table"].drop(columns="model"), 4)
        + "\n## Sliding 20-year window minus expanding (onset subset)\n\n"
        + _md(sliding["table"].drop(columns="model"), 4)
        + "\n## Calibration, scored with other-fold calibrators\n\n### All days\n\n"
        + _md(_wide(cal_all, "lead"), 4)
        + "\n### Onset days\n\n"
        + _md(_wide(cal_onset, "lead"), 4)
        + "\n### 10-day window: all days, then onset days\n\n"
        + _md(win_all.drop(columns="group"), 4)
        + "\n"
        + _md(win_onset.drop(columns="group"), 4)
        + "\n### Mean forecast against observed (%), by lead, month and fold\n\n"
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
