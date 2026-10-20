"""Quality checks on a synthetic 3 x 3 grid where the right answer is known."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from heatwave.config import read_yaml
from heatwave.ingest.raw_store import RawStore
from heatwave.ingest.run import Paths
from heatwave.quality import checks
from heatwave.quality.report import validate_recent

Q = read_yaml("quality.yaml")
DATA = read_yaml("data.yaml")
VARS = list(Q["hard"]["ranges"])
CELLS = pd.DataFrame(
    [
        {"cell_id": f"c{j}{i}", "j": j, "i": i, "lat": 26 + j / 2, "lon": 80 + i * 0.625}
        for j in range(3)
        for i in range(3)
    ]
)


def clean_wide(days=730, start="2020-01-01", seed=0):
    """Smooth seasonal fields shared across cells plus small noise: no flags expected."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    season = np.sin(2 * np.pi * (dates.dayofyear - 100) / 365.25)
    weather = np.cumsum(rng.normal(0, 0.6, days)) * 0.3  # shared synoptic signal
    rows = []
    for c in CELLS.itertuples():
        tmax = 32 + 8 * season + weather + rng.normal(0, 0.3, days)
        tmean = tmax - 6
        rows.append(
            pd.DataFrame(
                {
                    "cell_id": c.cell_id,
                    "obs_date": dates,
                    "T2M_MAX": tmax,
                    "T2M": tmean,
                    "T2M_MIN": tmean - 6,
                    "T2MDEW": tmean - 10,
                    "RH2M": 50.0,
                    "QV2M": 10.0,
                    "PRECTOTCORR": 0.0,
                    "WS10M": 3.0,
                    "U10M": 1.0,
                    "V10M": -1.0,
                    "PS": 98.5,
                    "GWETTOP": 0.4,
                    "GWETROOT": 0.45,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)[["cell_id", "obs_date", *VARS]]


def at(w, cell, d):
    return (w["cell_id"] == cell) & (w["obs_date"] == pd.Timestamp(d))


# --- hard checks -------------------------------------------------------------


def test_clean_data_passes_hard_checks():
    assert checks.hard_failures(clean_wide(), Q["hard"]).empty


@pytest.mark.parametrize(
    ("column", "value", "check"),
    [
        ("RH2M", 104.0, "in_range"),
        ("PRECTOTCORR", -1.0, "in_range"),
        ("GWETROOT", 1.2, "in_range"),
        ("T2M_MIN", 40.0, "tmin_le_tmean"),
        ("T2M_MAX", 0.0, "tmean_le_tmax"),
        ("T2MDEW", 45.0, "dew_le_tmean"),
    ],
)
def test_impossible_values_fail(column, value, check):
    w = clean_wide()
    w.loc[at(w, "c11", "2020-05-01"), column] = value
    fails = checks.hard_failures(w, Q["hard"])
    assert not fails.empty
    assert fails["check"].astype(str).str.contains(check).any()


def test_wind_direction_values_like_the_power_defect_are_impossible():
    # WD10M is not a checked variable any more, but a stray extra column is rejected.
    w = clean_wide().assign(WD10M=-6000.0)
    assert not checks.hard_failures(w, Q["hard"]).empty


def test_duplicate_key_fails():
    w = clean_wide()
    w = pd.concat([w, w.iloc[[5]]], ignore_index=True)
    assert not checks.hard_failures(w, Q["hard"]).empty


def test_missing_values_are_not_hard_failures():
    w = clean_wide()
    w.loc[at(w, "c11", "2020-05-01"), ["T2M_MAX", "GWETTOP"]] = np.nan
    assert checks.hard_failures(w, Q["hard"]).empty


# --- soft checks -------------------------------------------------------------


def test_neighbours_on_a_grid():
    nb = checks.neighbours(CELLS)
    assert len(nb["c11"]) == 8 and len(nb["c00"]) == 3 and len(nb["c01"]) == 5


def test_clean_data_has_no_soft_flags():
    assert checks.soft_flags(clean_wide(), CELLS, Q["soft"]).empty


def test_isolated_jump_is_flagged_but_shared_jump_is_weather():
    w = clean_wide()
    w.loc[at(w, "c11", "2020-05-10"), "T2M_MAX"] -= 14  # one cell only
    for c in CELLS["cell_id"]:  # a front: every cell drops together
        w.loc[at(w, c, "2021-04-10"), "T2M_MAX"] -= 14
    jumps = checks.isolated_jumps(
        w, CELLS, "T2M_MAX", Q["soft"]["jump"]["z"], Q["soft"]["jump"]["neighbour_z"]
    )
    flagged = set(zip(jumps["cell_id"], jumps["obs_date"].dt.date, strict=True))
    assert ("c11", date(2020, 5, 10)) in flagged
    assert not any(d in (date(2021, 4, 10), date(2021, 4, 11)) for _, d in flagged)


def test_spatial_outlier_needs_a_large_absolute_departure():
    w = clean_wide()
    w.loc[at(w, "c11", "2020-06-01"), "T2M_MAX"] += 8.0
    w.loc[at(w, "c11", "2020-07-01"), "T2M_MAX"] += 2.0  # within normal variation
    out = checks.spatial_outliers(w, CELLS, "T2M_MAX", Q["soft"]["spatial"]["abs_dev_c"])
    assert set(zip(out["cell_id"], out["obs_date"].dt.date, strict=True)) == {
        ("c11", date(2020, 6, 1))
    }


def test_reviewed_flags_get_their_reason():
    flags = pd.DataFrame(
        {
            "check": ["spatial", "spatial"],
            "variable": ["T2M_MAX"] * 2,
            "cell_id": ["c11", "c12"],
            "obs_date": pd.to_datetime(["2020-06-01", "2020-06-02"]),
        }
    )
    reviewed = [
        {
            "check": "spatial",
            "variable": "T2M_MAX",
            "cell_id": "c11",
            "obs_date": date(2020, 6, 1),
            "reason": "storm",
        }
    ]
    out = checks.mark_reviewed(flags, reviewed)
    assert out["reason"].iloc[0] == "storm"
    assert pd.isna(out["reason"].iloc[1])


def test_quality_rules_cover_exactly_the_fetched_parameters():
    assert set(VARS) == set(DATA["power"]["parameters"])
    assert "WD10M" not in VARS


# --- daily validate ----------------------------------------------------------


@pytest.fixture
def paths(tmp_path):
    p = Paths.from_config({"storage": {**DATA["storage"], "root": "data"}}, tmp_path)
    p.grid_cells.parent.mkdir(parents=True)
    CELLS.to_parquet(p.grid_cells)
    return p


def store_wide(paths, w):
    long = w.melt(id_vars=["cell_id", "obs_date"], var_name="variable", value_name="value")
    RawStore(paths.raw_power).append(long, "run1", pd.Timestamp("2026-10-20", tz="UTC"), "v")


def test_validate_passes_on_fresh_complete_data(paths):
    store_wide(paths, clean_wide(days=200, start="2026-04-01"))  # ends 2026-10-17
    r = validate_recent(DATA, Q, paths, date(2026, 10, 20))
    assert r["ok"], r
    assert r["latency_days"] == 3


def test_validate_fails_on_stale_data(paths):
    store_wide(paths, clean_wide(days=200, start="2026-04-01"))
    r = validate_recent(DATA, Q, paths, date(2026, 10, 30))
    assert not r["ok"] and "days old" in r["problems"][0]


def test_validate_fails_when_a_cell_is_missing_the_newest_day(paths):
    w = clean_wide(days=200, start="2026-04-01")
    w = w[~at(w, "c22", "2026-10-17")]
    store_wide(paths, w)
    r = validate_recent(DATA, Q, paths, date(2026, 10, 20))
    assert not r["ok"] and r["missing_cells"] == ["c22"]
