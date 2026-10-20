"""Leakage tests on the real data (Phase 5 exit criterion: 20 random dates).

Slow (two full feature builds per date): run with `uv run pytest -m slow`.
Skipped when the local data is absent (e.g. in CI).
"""

import json

import numpy as np
import pandas as pd
import pytest

from heatwave.config import REPO_ROOT, load_problem, read_yaml
from heatwave.features.build import ASOF_FEATURES, compute_features
from heatwave.features.pipeline import load_climatology, load_inputs

DATA = REPO_ROOT / "data"
SILVER = REPO_ROOT / "data" / "silver"
CFG = read_yaml("features.yaml")
VALUES = [
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

have = (SILVER / "power_district_daily.parquet").exists() and (
    DATA / "features" / "development_climatology"
).exists()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not have, reason="local data not built")]

rng = np.random.default_rng(2026)
season_days = pd.date_range("1982-01-01", "2020-12-31", freq="D")
season_days = season_days[season_days.month.isin([2, 3, 4, 5, 6, 7, 8])]
DATES = sorted(pd.Timestamp(d) for d in rng.choice(season_days, 20, replace=False))


@pytest.fixture(scope="module")
def setup():
    daily, labels, districts = load_inputs(DATA, SILVER)
    clim = load_climatology(DATA / "features" / "development_climatology")
    full = compute_features(daily, labels, clim, districts, CFG)
    return daily, labels, districts, clim, full


def at(features, t):
    return features[features.asof_date == t][["district_id", *ASOF_FEATURES]].reset_index(drop=True)


@pytest.mark.parametrize("t", DATES, ids=lambda d: str(d.date()))
def test_truncation_invariance_real(setup, t):
    daily, labels, districts, clim, full = setup
    cut = compute_features(
        daily[daily.obs_date <= t], labels[labels.obs_date <= t], clim, districts, CFG
    )
    pd.testing.assert_frame_equal(at(cut, t), at(full, t))


@pytest.mark.parametrize("t", DATES, ids=lambda d: str(d.date()))
def test_future_perturbation_real(setup, t):
    daily, labels, districts, clim, full = setup
    d, lab = daily.copy(), labels.copy()
    future = d.obs_date > t
    noise = np.random.default_rng(7).normal(size=(int(future.sum()), len(VALUES)))
    d.loc[future, VALUES] = d.loc[future, VALUES].to_numpy() + 30 * noise
    flip = lab.obs_date > t
    lab.loc[flip, "is_heatwave"] = ~lab.loc[flip, "is_heatwave"]
    pd.testing.assert_frame_equal(
        at(compute_features(d, lab, clim, districts, CFG), t), at(full, t)
    )


def test_saved_tables_match_a_rebuild(setup):
    """The stored development table equals features recomputed from the saved
    climatology (the saved climatology round-trips)."""
    daily, labels, districts, clim, full = setup
    stored = pd.read_parquet(DATA / "features" / "development.parquet")
    t = stored.asof_date.drop_duplicates().sample(5, random_state=1)
    for d in t:
        pd.testing.assert_frame_equal(at(stored, d), at(full, d))


def test_fold_tables_never_reach_past_their_validation_years():
    m = json.loads((DATA / "features" / "manifest.json").read_text())
    for f in load_problem().validation.folds:
        b = m["blocks"][f"fold{f.fold}"]
        assert pd.Timestamp(b["last_asof"]).year <= f.validate[1]
        assert b["fit_end_date"] < f"{f.validate[0]}-01-01"
