"""District polygons -> grid cells -> area weights (reference.md §2.2).

Writes three reference tables:
    districts              one row per district: name, centroid, area, centroid cell
    grid_cells             every cell overlapping a district (the POWER fetch list)
    district_grid_weights  share of each district's area in each cell (sums to 1)
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import pandas as pd

from heatwave.geo.grid import Grid

WGS84 = "EPSG:4326"


def to_district_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_districts(boundaries_cfg: dict, root: Path) -> gpd.GeoDataFrame:
    """Load the state's districts, apply official renames, and check the count."""
    path = root / boundaries_cfg["path"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see configs/data.yaml boundaries.path")

    gdf = gpd.read_file(path, layer=boundaries_cfg["layer"])
    gdf = gdf[gdf[boundaries_cfg["state_field"]] == boundaries_cfg["state_name"]]
    names = gdf[boundaries_cfg["district_field"]].replace(boundaries_cfg["renames"])

    out = gpd.GeoDataFrame(
        {"district_id": names.map(to_district_id).to_numpy(), "name": names.to_numpy()},
        geometry=gdf.geometry.to_crs(WGS84).to_numpy(),
        crs=WGS84,
    )

    expected = boundaries_cfg["expected_district_count"]
    if len(out) != expected:
        raise ValueError(f"expected {expected} districts, found {len(out)}")
    dupes = out["district_id"][out["district_id"].duplicated()]
    if not dupes.empty:
        raise ValueError(f"duplicate district ids: {sorted(dupes)}")
    unrenamed = set(boundaries_cfg["renames"]) & set(out["name"])
    if unrenamed:
        raise ValueError(f"renames not applied: {sorted(unrenamed)}")
    return out.sort_values("district_id").reset_index(drop=True)


def candidate_cells(
    districts: gpd.GeoDataFrame, grid: Grid, densify_deg: float
) -> gpd.GeoDataFrame:
    """Every grid cell inside the districts' bounding box, as lon/lat polygons."""
    rows = []
    for j, i in grid.indices_covering(*districts.total_bounds):
        lat, lon = grid.center(j, i)
        rows.append(
            {
                "cell_id": grid.cell_id(j, i),
                "j": j,
                "i": i,
                "lat": float(lat),
                "lon": float(lon),
                "geometry": grid.cell_polygon(j, i, densify_deg),
            }
        )
    return gpd.GeoDataFrame(rows, crs=WGS84)


def district_grid_weights(
    districts: gpd.GeoDataFrame, cells: gpd.GeoDataFrame, equal_area_crs: str, min_weight: float
) -> pd.DataFrame:
    """Share of each district's area falling in each cell, in an equal-area CRS.

    Fragments below min_weight (boundary slivers) are dropped and the remaining
    weights renormalised, so each district's weights sum to exactly 1.
    """
    d = districts[["district_id", "geometry"]].to_crs(equal_area_crs)
    c = cells[["cell_id", "geometry"]].to_crs(equal_area_crs)
    d["district_area"] = d.area

    pieces = gpd.overlay(d, c, how="intersection", keep_geom_type=True)
    pieces["area_weight"] = pieces.area / pieces["district_area"]
    pieces = pieces[pieces["area_weight"] >= min_weight].copy()
    pieces["area_weight"] /= pieces.groupby("district_id")["area_weight"].transform("sum")

    return (
        pd.DataFrame(pieces[["district_id", "cell_id", "area_weight"]])
        .sort_values(["district_id", "cell_id"])
        .reset_index(drop=True)
    )


def districts_table(districts: gpd.GeoDataFrame, grid: Grid, equal_area_crs: str) -> pd.DataFrame:
    projected = districts.to_crs(equal_area_crs)
    centroids = projected.geometry.centroid.to_crs(WGS84)
    lat, lon = centroids.y.to_numpy(), centroids.x.to_numpy()
    return pd.DataFrame(
        {
            "district_id": districts["district_id"],
            "name": districts["name"],
            "centroid_lat": lat,
            "centroid_lon": lon,
            "area_km2": projected.area.to_numpy() / 1e6,
            "centroid_cell_id": grid.cell_ids(lat, lon),
        }
    )


def weight_overlap(weights: pd.DataFrame) -> pd.DataFrame:
    """For every district pair, the share of input they have in common:
    sum over cells of min(w_a, w_b). 1 = identical inputs, 0 = no shared cell."""
    wide = weights.pivot(index="district_id", columns="cell_id", values="area_weight").fillna(0)
    rows = []
    for a, b in combinations(wide.index, 2):
        shared = float(wide.loc[a].combine(wide.loc[b], min).sum())
        if shared > 0:
            rows.append({"district_a": a, "district_b": b, "overlap": shared})
    return pd.DataFrame(rows, columns=["district_a", "district_b", "overlap"])


def build_geo(data_cfg: dict, root: Path) -> dict:
    """Build and write the reference tables. Returns a summary for the decision record."""
    grid = Grid.from_config(data_cfg["grid"])
    geo = data_cfg["geo"]
    crs = geo["equal_area_crs"]

    districts = load_districts(data_cfg["boundaries"], root)
    cells = candidate_cells(districts, grid, geo["densify_deg"])
    weights = district_grid_weights(districts, cells, crs, geo["min_weight"])
    dtable = districts_table(districts, grid, crs)

    per_district = weights.groupby("district_id")["area_weight"]
    dtable["n_cells"] = dtable["district_id"].map(per_district.size()).astype(int)
    dtable["max_cell_weight"] = dtable["district_id"].map(per_district.max())
    used = cells[cells["cell_id"].isin(weights["cell_id"])]
    grid_cells = pd.DataFrame(used[["cell_id", "j", "i", "lat", "lon"]]).reset_index(drop=True)
    overlap = weight_overlap(weights)

    out_dir = root / data_cfg["outputs"]["reference_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dtable.to_parquet(out_dir / "districts.parquet", index=False)
    grid_cells.to_parquet(out_dir / "grid_cells.parquet", index=False)
    weights.to_parquet(out_dir / "district_grid_weights.parquet", index=False)
    overlap.to_parquet(out_dir / "district_overlap.parquet", index=False)

    centroid_counts = dtable["centroid_cell_id"].value_counts()
    shared = centroid_counts[centroid_counts > 1]
    return {
        "districts": len(dtable),
        "grid_cells_to_fetch": len(grid_cells),
        "cells_per_district_min_median_max": [
            int(dtable["n_cells"].min()),
            float(dtable["n_cells"].median()),
            int(dtable["n_cells"].max()),
        ],
        "districts_with_one_cell_over_80pct": int((dtable["max_cell_weight"] >= 0.8).sum()),
        "median_district_area_km2": round(float(dtable["area_km2"].median()), 1),
        "centroid_method_distinct_series": int(centroid_counts.size),
        "centroid_method_districts_sharing_a_cell": int(shared.sum()),
        "district_pairs_sharing_any_cell": len(overlap),
        "district_pairs_overlap_over_50pct": int((overlap["overlap"] > 0.5).sum()),
    }
