"""Bitemporal, append-only raw store (reference.md §2.3, §5.3).

Layout:  <root>/<raw_power_dir>/obs_year=YYYY/<run_id>.parquet

Each run writes, per observation year, only the rows that are new or whose
value changed since the latest known version. Nothing is ever rewritten, so

    best()                 latest known value per (cell, date, variable)
    best(asof=timestamp)   the value as it was known at that time

can both be reconstructed. The second is what measures provisional (GEOS-IT)
vs final (MERRA-2) differences.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEY = ["cell_id", "obs_date", "variable"]
SCHEMA = pa.schema(
    [
        ("cell_id", pa.string()),
        ("obs_date", pa.date32()),
        ("variable", pa.string()),
        ("value", pa.float64()),
        ("ingested_at", pa.timestamp("us", tz="UTC")),
        ("run_id", pa.string()),
        ("api_version", pa.string()),
    ]
)


class RawStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def year_dir(self, year: int) -> Path:
        return self.root / f"obs_year={year}"

    def years(self) -> list[int]:
        return sorted(int(p.name.split("=")[1]) for p in self.root.glob("obs_year=*") if p.is_dir())

    def files(self, years: Iterable[int] | None = None) -> list[Path]:
        years = self.years() if years is None else years
        return sorted(f for y in years for f in self.year_dir(y).glob("*.parquet"))

    def read(self, years: Iterable[int] | None = None) -> pd.DataFrame:
        """Every stored version of every value."""
        files = self.files(years)
        if not files:
            return SCHEMA.empty_table().to_pandas()
        return pa.concat_tables(pq.read_table(f, schema=SCHEMA) for f in files).to_pandas()

    def best_sql(self, years: Iterable[int] | None = None, asof: pd.Timestamp | None = None):
        """SQL (DuckDB) selecting the latest version per key, optionally as known at
        time `asof` (UTC). None if the store is empty. Use for whole-history queries."""
        files = self.files(years)
        if not files:
            return None
        file_list = ", ".join(f"'{f.as_posix()}'" for f in files)
        where = ""
        if asof is not None:
            where = f"WHERE ingested_at <= TIMESTAMPTZ '{pd.Timestamp(asof).isoformat()}'"
        return f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, obs_date, variable ORDER BY ingested_at DESC
                ) AS rn
                FROM read_parquet([{file_list}]) {where}
            ) WHERE rn = 1
        """

    def best(self, years: Iterable[int] | None = None, asof: pd.Timestamp | None = None):
        """Latest version per key as a DataFrame, optionally as known at `asof`."""
        sql = self.best_sql(years, asof)
        if sql is None:
            return SCHEMA.empty_table().to_pandas()
        with duckdb.connect() as con:
            return con.sql(sql + " ORDER BY cell_id, obs_date, variable").df()

    def append(
        self, values: pd.DataFrame, run_id: str, ingested_at: pd.Timestamp, api_version: str
    ) -> int:
        """Store rows of `values` (cell_id, obs_date, variable, value) that are new
        or changed. Returns the number of rows written."""
        if values.empty:
            return 0
        if values.duplicated(KEY).any():
            raise ValueError("values contain duplicate (cell_id, obs_date, variable) keys")

        new = values[KEY + ["value"]].copy()
        new["obs_date"] = pd.to_datetime(new["obs_date"]).dt.date
        years = sorted({d.year for d in new["obs_date"]})

        current = self.best(years)
        if current.empty:
            changed = new
        else:
            current = current[KEY + ["value"]].rename(columns={"value": "old"})
            current["obs_date"] = pd.to_datetime(current["obs_date"]).dt.date
            merged = new.merge(current, on=KEY, how="left", indicator=True)
            is_new = merged["_merge"] == "left_only"
            changed = merged[is_new | ~_same(merged["value"], merged["old"])][KEY + ["value"]]

        if changed.empty:
            return 0
        changed = changed.assign(
            ingested_at=pd.Timestamp(ingested_at).tz_convert("UTC"),
            run_id=run_id,
            api_version=api_version,
        )
        year_of = pd.to_datetime(changed["obs_date"]).dt.year
        for year, part in changed.groupby(year_of):
            out = self.year_dir(int(year)) / f"{run_id}.parquet"
            if out.exists():
                raise FileExistsError(f"{out} already exists; run ids must be unique")
            out.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(
                part.sort_values(KEY).reset_index(drop=True), schema=SCHEMA, preserve_index=False
            )
            pq.write_table(table, out, compression="zstd")
        return len(changed)


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality treating null == null as the same value."""
    a, b = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    both_null = np.isnan(a) & np.isnan(b)
    return both_null | np.isclose(a, b, rtol=0, atol=1e-9)
