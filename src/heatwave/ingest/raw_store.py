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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class KnownDefect:
    """A known upstream defect: versions of `variable` ingested from
    `ingested_from` (until `ingested_to`, exclusive, once the source is fixed) while
    POWER reported any of `api_versions`, for observation dates from `obs_from`
    (to `obs_to`), are quarantined in the best-known views. Raw rows are never changed.

    `refetched`: observation-date ranges that runs after the fix downloaded again.
    The store writes only changed values, so a defective version still latest
    inside these ranges was returned unchanged by the fixed source (e.g. 0 mm
    rain, where x24 changes nothing) and is used again."""

    id: str
    variable: str
    api_versions: tuple[str, ...]
    ingested_from: str
    obs_from: str
    obs_to: str | None = None
    reason: str = ""
    ingested_to: str | None = None
    refetched: tuple[tuple[str, str], ...] = ()

    def sql(self) -> str:
        versions = ", ".join(f"'{v}'" for v in self.api_versions)
        cond = (
            f"(variable = '{self.variable}' AND api_version IN ({versions}) "
            f"AND ingested_at >= TIMESTAMPTZ '{self.ingested_from}' "
            f"AND obs_date >= DATE '{self.obs_from}'"
        )
        if self.ingested_to:
            cond += f" AND ingested_at < TIMESTAMPTZ '{self.ingested_to}'"
        if self.obs_to:
            cond += f" AND obs_date <= DATE '{self.obs_to}'"
        for start, end in self.refetched:
            cond += f" AND NOT obs_date BETWEEN DATE '{start}' AND DATE '{end}'"
        return cond + ")"


def defects_from_config(data_cfg: dict, runs_log: Path | None = None) -> tuple[KnownDefect, ...]:
    """Defects from config; with the ingest run log, closed defects also get the
    observation dates that successful runs after the fix downloaded again."""
    runs = pd.read_parquet(runs_log) if runs_log is not None and runs_log.exists() else None

    def refetched(ingested_to) -> tuple[tuple[str, str], ...]:
        if not ingested_to or runs is None or runs.empty:
            return ()
        after = runs[(runs["status"] == "ok") & (runs["started_at"] >= pd.Timestamp(ingested_to))]
        return tuple(
            (str(pd.Timestamp(a).date()), str(pd.Timestamp(b).date()))
            for a, b in after[["window_start", "window_end"]].itertuples(index=False)
        )

    return tuple(
        KnownDefect(
            id=d["id"],
            variable=d["variable"],
            api_versions=tuple(d["api_versions"]),
            ingested_from=str(d["ingested_from"]),
            obs_from=str(d["obs_from"]),
            obs_to=str(d["obs_to"]) if d.get("obs_to") else None,
            reason=d.get("reason", ""),
            ingested_to=str(d["ingested_to"]) if d.get("ingested_to") else None,
            refetched=refetched(d.get("ingested_to")),
        )
        for d in data_cfg.get("known_defects") or []
    )


class RawStore:
    def __init__(self, root: Path, defects: Iterable[KnownDefect] = ()):
        self.root = Path(root)
        self.defects = tuple(defects)

    @classmethod
    def from_config(cls, data_cfg: dict, data_root: Path) -> RawStore:
        """The store under data_root with the configured known defects applied."""
        s = data_cfg["storage"]
        return cls(
            data_root / s["raw_power_dir"], defects_from_config(data_cfg, data_root / s["runs_log"])
        )

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

    def best_sql(
        self,
        years: Iterable[int] | None = None,
        asof: pd.Timestamp | None = None,
        apply_defects: bool = True,
    ):
        """SQL (DuckDB) selecting the latest version per key, optionally as known at
        time `asof` (UTC). None if the store is empty. Use for whole-history queries.

        With known defects applied, a quarantined version is used only if no good
        version exists, and then its value is NULL (missing, never a wrong number).
        Column `quarantined` names the defect, else NULL."""
        files = self.files(years)
        if not files:
            return None
        file_list = ", ".join(f"'{f.as_posix()}'" for f in files)
        where = ""
        if asof is not None:
            where = f"WHERE ingested_at <= TIMESTAMPTZ '{pd.Timestamp(asof).isoformat()}'"
        defects = self.defects if apply_defects else ()
        if defects:
            flag = "CASE " + " ".join(f"WHEN {d.sql()} THEN '{d.id}'" for d in defects) + " END"
        else:
            flag = "CAST(NULL AS VARCHAR)"
        return f"""
            SELECT * EXCLUDE (rn, value), CASE WHEN quarantined IS NULL THEN value END AS value
            FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY cell_id, obs_date, variable
                    ORDER BY (quarantined IS NOT NULL), ingested_at DESC
                ) AS rn
                FROM (SELECT *, {flag} AS quarantined FROM read_parquet([{file_list}]) {where})
            ) WHERE rn = 1
        """

    def best(
        self,
        years: Iterable[int] | None = None,
        asof: pd.Timestamp | None = None,
        apply_defects: bool = True,
    ):
        """Latest version per key as a DataFrame, optionally as known at `asof`."""
        sql = self.best_sql(years, asof, apply_defects)
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
        year_of = pd.to_datetime(new["obs_date"]).dt.year
        years = sorted(year_of.unique().tolist())

        # Refuse before writing anything, so a reused run id cannot leave a partial run.
        clashes = [self.year_dir(y) / f"{run_id}.parquet" for y in years]
        clashes = [p for p in clashes if p.exists()]
        if clashes:
            raise FileExistsError(f"{clashes[0]} already exists; run ids must be unique")

        written = 0
        # One observation year at a time keeps memory bounded on full-history runs.
        for year, part in new.groupby(year_of):
            changed = self._changed(part, int(year))
            if changed.empty:
                continue
            changed = changed.assign(
                ingested_at=pd.Timestamp(ingested_at).tz_convert("UTC"),
                run_id=run_id,
                api_version=api_version,
            )
            out = self.year_dir(int(year)) / f"{run_id}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(
                changed.sort_values(KEY).reset_index(drop=True), schema=SCHEMA, preserve_index=False
            )
            pq.write_table(table, out, compression="zstd")
            written += len(changed)
        return written

    def _changed(self, new: pd.DataFrame, year: int) -> pd.DataFrame:
        """Rows of `new` (all in `year`) that are absent from, or differ from, the
        latest stored version."""
        # Compare with what was actually stored last, quarantined or not, so a
        # defective version is not re-appended on every run.
        current = self.best([year], apply_defects=False)
        if current.empty:
            return new
        current = current[KEY + ["value"]].rename(columns={"value": "old"})
        current["obs_date"] = pd.to_datetime(current["obs_date"]).dt.date
        merged = new.merge(current, on=KEY, how="left", indicator=True)
        is_new = merged["_merge"] == "left_only"
        return merged[is_new | ~_same(merged["value"], merged["old"])][KEY + ["value"]]


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality treating null == null as the same value."""
    a, b = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    both_null = np.isnan(a) & np.isnan(b)
    return both_null | np.isclose(a, b, rtol=0, atol=1e-9)
