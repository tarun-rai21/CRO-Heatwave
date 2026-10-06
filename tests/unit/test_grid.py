import numpy as np

from heatwave.config import read_yaml
from heatwave.geo.grid import Grid

GRID = Grid.from_config(read_yaml("data.yaml")["grid"])


def test_known_cell_centre():
    # Lucknow ~ (26.85 N, 80.90 E): nearest MERRA-2 centres are 27.0 N, 80.625 E.
    j, i = GRID.index_of(26.85, 80.90)
    assert tuple(float(x) for x in GRID.center(j, i)) == (27.0, 80.625)
    assert GRID.cell_id(j, i) == "merra2_r234_c417"


def test_every_centre_maps_back_to_its_own_cell():
    js, is_ = np.meshgrid(np.arange(225, 240), np.arange(410, 430), indexing="ij")
    lat, lon = GRID.center(js.ravel(), is_.ravel())
    j2, i2 = GRID.index_of(lat, lon)
    assert (j2 == js.ravel()).all()
    assert (i2 == is_.ravel()).all()


def test_points_on_edges_go_north_and_east():
    j, i = GRID.index_of(26.75, 80.3125)  # exactly on a cell corner
    assert tuple(float(x) for x in GRID.center(j, i)) == (27.0, 80.625)


def test_cell_polygon_has_grid_size_and_is_densified():
    poly = GRID.cell_polygon(*GRID.index_of(26.85, 80.90))
    minx, miny, maxx, maxy = poly.bounds
    assert np.isclose(maxx - minx, GRID.dlon)
    assert np.isclose(maxy - miny, GRID.dlat)
    assert len(poly.exterior.coords) > 5


def test_indices_covering_includes_partial_cells():
    cells = set(GRID.indices_covering(80.0, 26.0, 81.0, 27.0))
    lats = {float(GRID.center(j, 0)[0]) for j, _ in cells}
    assert lats == {26.0, 26.5, 27.0}
