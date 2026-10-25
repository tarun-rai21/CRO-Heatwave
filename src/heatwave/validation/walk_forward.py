"""Walk-forward runner (reference.md §3.3).

For each fold: assemble the per-lead or window table from that fold's feature
block, train every model on rows whose target year is in the fold's training
years (training window, 15 Feb - 15 Aug), and predict rows whose target year is
in the fold's validation years and whose valid dates are in the evaluation
season (March-July). The 5 folds' predictions are pooled into one
out-of-fold (OOF) table covering 2006-2020.

Fences, checked before anything is fitted: training years end before validation
years start, and the fold's climatology was fitted on training years only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from heatwave.config import Problem
from heatwave.features.pipeline import load_climatology
from heatwave.features.targets import in_window, per_lead_table, window_table
from heatwave.models import baselines

COLUMNS = [
    "hw_today",
    "tmax_anom",
    "tmax_anom_mean7",
    "gwetroot_anom",
    "doy_sin",
    "doy_cos",
    "days_since_hw",
]


def regions(path: Path) -> dict[str, str]:
    r = yaml.safe_load(path.read_text(encoding="utf-8"))["regions"]
    return {d: name for name, ds in r.items() for d in ds}


def fold_tables(
    problem: Problem, fold, root: Path, labels: pd.DataFrame, target: str, columns: list[str]
):
    """(train, validate) rows of one fold for target 'lead' or 'window'."""
    if fold.train[1] >= fold.validate[0]:
        raise ValueError(f"fold {fold.fold}: training years overlap validation years")
    clim = load_climatology(root / "features" / f"fold{fold.fold}_climatology")
    clim.check_held_out(fold.validate)
    feats = pd.read_parquet(root / "features" / f"fold{fold.fold}.parquet")
    horizons, L = problem.horizons, problem.latency_days
    if target == "lead":
        t = per_lead_table(feats, labels, clim, L, horizons, problem.training_season, columns)
        year = t["valid_date"].dt.year
        in_eval = in_window(t["valid_date"], problem.evaluation_season)
    else:
        t = window_table(feats, labels, L, horizons, problem.training_season, columns)
        year = t["issue_date"].dt.year
        first = t["issue_date"] + pd.Timedelta(days=horizons[0])
        last = t["issue_date"] + pd.Timedelta(days=horizons[-1])
        in_eval = in_window(first, problem.evaluation_season) & in_window(
            last, problem.evaluation_season
        )
    train = t[year.between(*fold.train).to_numpy()].reset_index(drop=True)
    validate = t[(year.between(*fold.validate).to_numpy()) & in_eval].reset_index(drop=True)
    return train, validate


def run(
    problem: Problem,
    model_cfg: dict,
    root: Path,
    region_map: dict[str, str],
    target: str,
    models: list[str],
    extra_columns: list[str] = (),
) -> pd.DataFrame:
    labels = pd.read_parquet(
        root / "labels" / "labels_daily.parquet", columns=["district_id", "obs_date", "is_heatwave"]
    )
    columns = [*COLUMNS, *extra_columns]
    parts, checks = [], []
    for fold in problem.validation.folds:
        train, validate = fold_tables(problem, fold, root, labels, target, columns)
        out = validate[["district_id", "issue_date", "y"]].copy()
        if target == "lead":
            out["valid_date"], out["lead"] = validate["valid_date"], validate["lead"]
        for name in models:
            model = baselines.make(name, target, model_cfg).fit(train)
            out[f"p_{name}"] = np.clip(model.predict(validate), 0, 1)
        # Bug check for M0: in its own training rows (evaluation season) its mean
        # forecast must equal the observed rate.
        if target == "lead":
            season = in_window(train["valid_date"], problem.evaluation_season)
        else:
            season = in_window(
                train["issue_date"] + pd.Timedelta(days=1), problem.evaluation_season
            )
        tr = train[season]
        m0 = baselines.make("M0", target, model_cfg).fit(train)
        checks.append(
            {
                "fold": fold.fold,
                "train_rate": float(tr["y"].mean()),
                "m0_train_mean": float(np.mean(m0.predict(tr))),
                "validate_rate": float(validate["y"].mean()),
                "m0_validate_mean": float(out["p_M0"].mean()),
            }
        )
        out["fold"] = fold.fold
        out["days_since_hw"] = validate["days_since_hw"].to_numpy()
        parts.append(out)
        print(
            f"fold {fold.fold} ({target}): train {len(train):,} rows, "
            f"validate {len(validate):,} rows"
        )
    oof = pd.concat(parts, ignore_index=True)
    date = oof["valid_date"] if target == "lead" else oof["issue_date"]
    oof["year"], oof["month"] = date.dt.year, date.dt.month
    oof["region"] = oof["district_id"].map(region_map)
    # Onset case: no heatwave day in the 3 days up to the as-of date t.
    oof["onset"] = oof["days_since_hw"] >= problem.onset_lookback_days
    oof.attrs["fold_checks"] = checks  # list of dicts (parquet stores attrs as JSON)
    return oof
