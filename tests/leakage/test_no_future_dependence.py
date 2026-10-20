"""Leakage tests (reference.md §4.9), on synthetic data so they run in CI.

1. Truncation invariance: features at t from the full table equal features at t
   from a table cut off at t.
2. Future perturbation: scrambling every value dated after t changes no feature at t.

Run at random dates and around a fold boundary. tests/data/test_leakage_real.py
repeats them on the real data when it is available locally.
"""

import numpy as np
import pandas as pd
import pytest

from heatwave.config import read_yaml
from heatwave.features.build import ASOF_FEATURES, compute_features
from heatwave.features.climatology import fit_climatology

from ..unit.test_features import DISTRICTS, synthetic

CFG = read_yaml("features.yaml")
VALUE_COLS = [
    "tmax",
    "T2M_MAX",
    "T2M_MIN",
    "T2MDEW",
    "RH2M",
    "QV2M",
    "PRECTOTCORR",
    "WS10M",
    "U10M",
    "V10M",
    "PS",
    "GWETTOP",
    "GWETROOT",
]

DAILY, LABELS = synthetic()
CLIM = fit_climatology(DAILY, LABELS, (2000, 2004), CFG)
FULL = compute_features(DAILY, LABELS, CLIM, DISTRICTS, CFG)

rng = np.random.default_rng(42)
dates = pd.date_range("2002-03-01", "2006-10-01", freq="D")
DATES = sorted(
    {*rng.choice(dates, 8, replace=False), pd.Timestamp("2004-12-31"), pd.Timestamp("2005-01-01")}
)


def at(features, t):
    return features[features.asof_date == t][["district_id", *ASOF_FEATURES]].reset_index(drop=True)


@pytest.mark.parametrize("t", DATES, ids=lambda d: str(pd.Timestamp(d).date()))
def test_truncation_invariance(t):
    t = pd.Timestamp(t)
    cut = compute_features(
        DAILY[DAILY.obs_date <= t], LABELS[LABELS.obs_date <= t], CLIM, DISTRICTS, CFG
    )
    pd.testing.assert_frame_equal(at(cut, t), at(FULL, t))


@pytest.mark.parametrize("t", DATES, ids=lambda d: str(pd.Timestamp(d).date()))
def test_future_perturbation(t):
    t = pd.Timestamp(t)
    daily, labels = DAILY.copy(), LABELS.copy()
    future = daily.obs_date > t
    noise = np.random.default_rng(1).normal(size=(int(future.sum()), len(VALUE_COLS)))
    daily.loc[future, VALUE_COLS] = daily.loc[future, VALUE_COLS].to_numpy() + 50 * noise
    flip = labels.obs_date > t
    labels.loc[flip, "is_heatwave"] = ~labels.loc[flip, "is_heatwave"]
    pert = compute_features(daily, labels, CLIM, DISTRICTS, CFG)
    pd.testing.assert_frame_equal(at(pert, t), at(FULL, t))
