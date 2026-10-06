"""Area-weight logic on synthetic districts whose answers are known exactly."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from heatwave.config import read_yaml
from heatwave.geo.grid import Grid
from heatwave.geo.weights import (
    candidate_cells,
    district_grid_weights,
    to_district_id,
    weight_overlap,
)

CFG = read_yaml("data.yaml")
GRID = Grid.from_config(CFG["grid"])
CRS = CFG["geo"]["equal_area_crs"]

# Cell centred at (27.0 N, 80.625 E) spans lon 80.3125..80.9375, lat 26.75..27.25.
CELL = (80.3125, 26.75, 80.9375, 27.25)


def weights_for(geoms: dict, min_weight: float = 0.001) -> pd.DataFrame:
    districts = gpd.GeoDataFrame(
        {"district_id": list(geoms)}, geometry=list(geoms.values()), crs="EPSG:4326"
    )
    cells = candidate_cells(districts, GRID, densify_deg=0.05)
    return district_grid_weights(districts, cells, CRS, min_weight)


def test_district_equal_to_one_cell_gets_weight_one():
    w = weights_for({"a": box(*CELL)})
    assert list(w["cell_id"]) == ["merra2_r234_c417"]
    assert w["area_weight"].iloc[0] == pytest.approx(1.0)


def test_district_split_across_two_cells_by_longitude():
    # Spans the eastern half of one cell and the western half of the next.
    half = 0.625 / 2
    w = weights_for({"a": box(80.9375 - half, 26.75, 80.9375 + half, 27.25)})
    assert len(w) == 2
    assert w["area_weight"].to_list() == pytest.approx([0.5, 0.5], abs=1e-3)


def test_unequal_split_uses_true_area_not_degrees():
    # Two halves of one district, split at a latitude edge: the southern part is
    # slightly larger on the ground (meridians converge northward).
    w = weights_for({"a": box(80.4, 26.5, 80.8, 27.0)})
    south, north = w.sort_values("cell_id")["area_weight"].to_list()
    assert south == pytest.approx(0.5, abs=0.01)
    assert south > north


def test_slivers_are_dropped_and_weights_renormalised():
    # 0.01 deg overhang into the next cell is < 2% of the area.
    w = weights_for({"a": box(80.3125, 26.75, 80.9475, 27.25)}, min_weight=0.05)
    assert len(w) == 1
    assert w["area_weight"].sum() == pytest.approx(1.0)


def test_weights_sum_to_one_for_irregular_districts():
    geoms = {
        "a": box(80.1, 26.3, 81.4, 27.6),
        "b": box(80.3, 26.9, 80.5, 27.1).union(box(81.0, 26.0, 81.2, 26.2)),
    }
    sums = weights_for(geoms).groupby("district_id")["area_weight"].sum()
    assert sums.to_list() == pytest.approx([1.0, 1.0])


def test_overlap_of_identical_and_disjoint_districts():
    w = pd.DataFrame(
        {
            "district_id": ["a", "b", "c", "c"],
            "cell_id": ["x", "x", "y", "z"],
            "area_weight": [1.0, 1.0, 0.5, 0.5],
        }
    )
    ov = weight_overlap(w)
    assert ov.to_dict("records") == [{"district_a": "a", "district_b": "b", "overlap": 1.0}]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Gautam Buddha Nagar", "gautam_buddha_nagar"),
        ("Rae Bareli", "rae_bareli"),
        ("  Kanpur  Dehat ", "kanpur_dehat"),
    ],
)
def test_district_ids(name, expected):
    assert to_district_id(name) == expected
