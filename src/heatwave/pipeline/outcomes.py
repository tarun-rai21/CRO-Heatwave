"""What actually happened, for forecasts whose valid dates have been observed (reference.md §5.4).

Outcomes are recomputed from the current labels each run, because recent values are still
provisional and a label can change when POWER revises them. A window with no heatwave so
far stays unknown until every day of it is observed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heatwave.labels.definition import window_target

KEEP = ["issue_date", "target", "district_id", "valid_date", "window_end", "p", "p_clim", "tier"]


def fill(forecasts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """`forecasts` with column `observed`: 1, 0 or NaN (not known yet)."""
    y = labels.set_index(["district_id", "obs_date"])["is_heatwave"].astype("float")
    y = y[~y.index.duplicated()]
    out = forecasts.copy()
    out["observed"] = np.nan

    is_lead = out["target"] == "lead"
    keys = pd.MultiIndex.from_arrays(
        [out.loc[is_lead, "district_id"], out.loc[is_lead, "valid_date"]]
    )
    out.loc[is_lead, "observed"] = y.reindex(keys).to_numpy()

    for (first, last), g in out[~is_lead].groupby(["valid_date", "window_end"]):
        days = pd.date_range(first, last)
        daily = np.column_stack(
            [
                y.reindex(pd.MultiIndex.from_arrays([g["district_id"], [d] * len(g)])).to_numpy()
                for d in days
            ]
        )
        out.loc[g.index, "observed"] = window_target(daily).to_numpy(dtype=float, na_value=np.nan)
    return out


def run(root: Path, labels: pd.DataFrame) -> list[Path]:
    """Rewrite the outcome table of each issue year covered by `labels`."""
    written = []
    years = sorted(labels["obs_date"].dt.year.unique())
    for year in years:
        files = sorted((root / "forecasts" / f"issue_year={year}").glob("*.parquet"))
        if not files:
            continue
        rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        out = fill(rows, labels)[[*KEEP, "observed"]]
        path = root / "outcomes" / f"issue_year={year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        written.append(path)
    return written
