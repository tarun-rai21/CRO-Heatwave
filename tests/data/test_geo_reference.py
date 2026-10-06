"""Checks on the real reference tables. Skipped when they have not been built
(e.g. in CI, where data is not available)."""

import pandas as pd
import pytest

from heatwave.config import REPO_ROOT, read_yaml

CFG = read_yaml("data.yaml")
REFERENCE = REPO_ROOT / CFG["outputs"]["reference_dir"]
TABLES = ("districts", "grid_cells", "district_grid_weights", "district_overlap")


@pytest.fixture(scope="module")
def t():
    if not all((REFERENCE / f"{name}.parquet").exists() for name in TABLES):
        pytest.skip("reference tables not built; run `uv run heatwave build-geo`")
    return {name: pd.read_parquet(REFERENCE / f"{name}.parquet") for name in TABLES}


def test_all_districts_present_with_current_names(t):
    d = t["districts"]
    assert len(d) == CFG["boundaries"]["expected_district_count"]
    assert d["district_id"].is_unique
    assert {"prayagraj", "ayodhya", "bhadohi"} <= set(d["district_id"])
    assert not set(CFG["boundaries"]["renames"]) & set(d["name"])


def test_weights_sum_to_one_per_district(t):
    sums = t["district_grid_weights"].groupby("district_id")["area_weight"].sum()
    assert set(sums.index) == set(t["districts"]["district_id"])
    assert (sums - 1).abs().max() < 1e-9


def test_weights_are_valid_shares(t):
    w = t["district_grid_weights"]["area_weight"]
    assert (w > 0).all() and (w <= 1).all()
    assert not t["district_grid_weights"].duplicated(["district_id", "cell_id"]).any()


def test_fetch_list_is_exactly_the_weighted_cells(t):
    assert set(t["grid_cells"]["cell_id"]) == set(t["district_grid_weights"]["cell_id"])
    assert t["grid_cells"]["cell_id"].is_unique


def test_districts_lie_in_uttar_pradesh(t):
    d = t["districts"]
    assert d["centroid_lat"].between(23.8, 30.5).all()
    assert d["centroid_lon"].between(77.0, 84.7).all()
    # UP is ~240,900 km2
    assert d["area_km2"].sum() == pytest.approx(240_900, rel=0.02)


def test_overlap_is_a_share(t):
    ov = t["district_overlap"]["overlap"]
    assert ((ov > 0) & (ov <= 1 + 1e-9)).all()
