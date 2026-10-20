"""Targets and training-table assembly (reference.md §1.2-1.3, §4.3).

Dates: as-of t (features use data <= t), issue s = t + L, valid s + h (h = 1..H).

    per-lead table: one row per (district, s, h); features at t = s - L, plus the
                    lead features; target y = heatwave label at s + h
    window table:   one row per (district, s); target y = any heatwave at s+1..s+H

Rows are kept when their valid date (per-lead) or all valid dates (window) fall
in the training window. Rows whose target is unknown are dropped, never filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heatwave.config import DayWindow
from heatwave.features.build import ASOF_FEATURES
from heatwave.features.climatology import Climatology
from heatwave.labels.definition import window_target


def in_window(dates, window: DayWindow) -> np.ndarray:
    allowed = np.array(sorted(m * 100 + dd for m, dd in window.days()))
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return np.isin(d.month.to_numpy() * 100 + d.day.to_numpy(), allowed)


def label_lookup(labels: pd.DataFrame) -> pd.Series:
    return labels.set_index(["district_id", "obs_date"])["is_heatwave"].astype("float")


def per_lead_table(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    clim: Climatology,
    latency: int,
    horizons: range,
    window: DayWindow,
) -> pd.DataFrame:
    """Long table: features at t, lead h, clim frequency at the valid date, target y."""
    y = label_lookup(labels)
    parts = []
    for h in horizons:
        f = features.copy()
        f["issue_date"] = f["asof_date"] + pd.Timedelta(days=latency)
        f["valid_date"] = f["issue_date"] + pd.Timedelta(days=h)
        f = f[in_window(f["valid_date"], window)]
        f["lead"] = h
        f["clim_hw_freq_valid"] = clim.frequency(f["district_id"], f["valid_date"])
        f["y"] = y.reindex(
            pd.MultiIndex.from_arrays([f["district_id"], f["valid_date"]])
        ).to_numpy()
        parts.append(f)
    out = pd.concat(parts, ignore_index=True).dropna(subset=["y"])
    cols = [
        "district_id",
        "asof_date",
        "issue_date",
        "valid_date",
        "lead",
        *ASOF_FEATURES,
        "clim_hw_freq_valid",
        "y",
    ]
    return out[cols].sort_values(["district_id", "issue_date", "lead"]).reset_index(drop=True)


def window_table(
    features: pd.DataFrame, labels: pd.DataFrame, latency: int, horizons: range, window: DayWindow
) -> pd.DataFrame:
    """One row per (district, issue date): features at t, y = any heatwave in s+1..s+H."""
    y = label_lookup(labels)
    f = features.copy()
    f["issue_date"] = f["asof_date"] + pd.Timedelta(days=latency)
    hs = list(horizons)
    first = f["issue_date"] + pd.Timedelta(days=hs[0])
    last = f["issue_date"] + pd.Timedelta(days=hs[-1])
    f = f[in_window(first, window) & in_window(last, window)]
    daily = np.column_stack(
        [
            y.reindex(
                pd.MultiIndex.from_arrays(
                    [f["district_id"], f["issue_date"] + pd.Timedelta(days=h)]
                )
            ).to_numpy()
            for h in hs
        ]
    )
    f["y"] = window_target(daily).astype("float")
    f = f.dropna(subset=["y"])
    cols = ["district_id", "asof_date", "issue_date", *ASOF_FEATURES, "y"]
    return f[cols].sort_values(["district_id", "issue_date"]).reset_index(drop=True)
