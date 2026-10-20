"""Heatwave-day rule (Target A, reference.md §2.4).

    heatwave = (Tmax >= 40.0 and departure >= 4.5)  or  Tmax >= 45.0
    severe   = (Tmax >= 40.0 and departure > 6.4)   or  Tmax >= 47.0

departure = Tmax - normal. Values are rounded to IMD's reporting precision
(0.1 C) before comparison, so float noise cannot flip a label at a threshold.
A missing Tmax or normal gives a missing label, never a negative one.

This is the district-level meteorological condition, not IMD's operational
declaration (>= 2 stations in a subdivision on 2 consecutive days).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from heatwave.config import HeatwaveRule


def classify(tmax, normal, rule: HeatwaveRule) -> pd.DataFrame:
    """Columns tmax_used, normal_tmax, departure (rounded) and nullable-boolean
    is_heatwave, is_severe."""
    nd = rule.round_decimals
    t = np.round(np.asarray(tmax, dtype=float), nd)
    n = np.round(np.asarray(normal, dtype=float), nd)
    dep = np.round(t - n, nd)
    missing = np.isnan(t) | np.isnan(n)
    above_floor = t >= rule.tmax_floor_c
    hw = (above_floor & (dep >= rule.departure_heatwave_c)) | (t >= rule.absolute_heatwave_c)
    sev = (above_floor & (dep > rule.departure_severe_c)) | (t >= rule.absolute_severe_c)

    def nullable(flags):
        return pd.arrays.BooleanArray(flags & ~missing, missing.copy())

    return pd.DataFrame(
        {
            "tmax_used": t,
            "normal_tmax": n,
            "departure": dep,
            "is_heatwave": nullable(hw),
            "is_severe": nullable(sev),
        }
    )


def window_target(daily: np.ndarray) -> pd.arrays.BooleanArray:
    """y_win = any heatwave over the horizons of each row of a (rows, horizons)
    array of 1/0/NaN. A known positive decides it; otherwise any missing day
    leaves it missing, because the window cannot be confirmed negative."""
    daily = np.asarray(daily, dtype=float)
    if daily.ndim != 2:
        raise ValueError("daily must be 2-D: (issue dates, horizons)")
    positive = np.nansum(daily, axis=1) > 0
    missing = np.isnan(daily).any(axis=1)
    return pd.arrays.BooleanArray(positive, ~positive & missing)


def label_version(rule: HeatwaveRule, extra: dict | None = None) -> str:
    """Stable hash of everything that defines the label (reference.md §2.4):
    rule, thresholds, normal settings, source, plus e.g. the bias-map version."""
    payload = json.dumps({"rule": asdict(rule), "extra": extra or {}}, sort_keys=True)
    return "lbl_" + hashlib.sha256(payload.encode()).hexdigest()[:12]
