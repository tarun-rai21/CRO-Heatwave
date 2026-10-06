"""The MERRA-2 grid that NASA POWER meteorology is served on (reference.md §2.2).

Cell centres sit at lat = lat_origin + dlat * j and lon = lon_origin + dlon * i;
each cell spans half a step either side of its centre.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import box


@dataclass(frozen=True)
class Grid:
    name: str
    lat_origin: float
    lon_origin: float
    dlat: float
    dlon: float

    @classmethod
    def from_config(cls, cfg: dict) -> Grid:
        return cls(
            name=cfg["name"],
            lat_origin=float(cfg["lat_origin"]),
            lon_origin=float(cfg["lon_origin"]),
            dlat=float(cfg["dlat"]),
            dlon=float(cfg["dlon"]),
        )

    def index_of(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """(j, i) of the cell containing each point. Points exactly on an edge
        go to the cell on the north / east side."""
        j = np.floor((np.asarray(lat, dtype=float) - self.lat_origin) / self.dlat + 0.5)
        i = np.floor((np.asarray(lon, dtype=float) - self.lon_origin) / self.dlon + 0.5)
        return j.astype(int), i.astype(int)

    def center(self, j, i) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.lat_origin + self.dlat * np.asarray(j),
            self.lon_origin + self.dlon * np.asarray(i),
        )

    def cell_id(self, j: int, i: int) -> str:
        return f"{self.name}_r{int(j):03d}_c{int(i):03d}"

    def cell_ids(self, lat, lon) -> list[str]:
        js, is_ = self.index_of(lat, lon)
        return [
            self.cell_id(j, i) for j, i in zip(np.atleast_1d(js), np.atleast_1d(is_), strict=True)
        ]

    def cell_polygon(self, j: int, i: int, densify_deg: float = 0.05) -> shapely.Polygon:
        """Cell boundary in lon/lat, densified so it stays accurate after projection."""
        lat, lon = (float(x) for x in self.center(j, i))
        half_lat, half_lon = self.dlat / 2, self.dlon / 2
        cell = box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
        return shapely.segmentize(cell, densify_deg)

    def indices_covering(
        self, minx: float, miny: float, maxx: float, maxy: float
    ) -> Iterator[tuple[int, int]]:
        """Every (j, i) whose cell intersects a lon/lat bounding box."""
        j0, i0 = self.index_of(miny, minx)
        j1, i1 = self.index_of(maxy, maxx)
        for j in range(int(j0), int(j1) + 1):
            for i in range(int(i0), int(i1) + 1):
                yield j, i
