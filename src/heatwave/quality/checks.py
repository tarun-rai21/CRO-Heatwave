"""Data quality checks (reference.md §6.2 Phase 3; rules in configs/quality.yaml).

Hard checks find physically impossible data and fail the pipeline. Soft checks
flag suspicious but possible data for review. Nothing here modifies data.

All checks work on the "wide" best-known table: one row per (cell_id, obs_date),
one column per POWER variable.
"""

from __future__ import annotations

from collections.abc import Iterable

import duckdb
import numpy as np
import pandas as pd
import pandera.pandas as pa

from heatwave.ingest.raw_store import RawStore

KEY = ["cell_id", "obs_date"]


def load_wide(store: RawStore, variables: list[str], years: Iterable[int] | None = None):
    """Best-known values pivoted to one column per variable (only `variables`)."""
    sql = store.best_sql(years)
    if sql is None:
        raise RuntimeError("raw store is empty")
    var_list = ", ".join(f"'{v}'" for v in variables)
    with duckdb.connect() as con:
        wide = con.sql(
            f"""
            PIVOT (SELECT cell_id, obs_date, variable, value FROM ({sql})
                   WHERE variable IN ({var_list}))
            ON variable IN ({var_list}) USING first(value)
            GROUP BY cell_id, obs_date ORDER BY cell_id, obs_date
            """
        ).df()
    wide["obs_date"] = pd.to_datetime(wide["obs_date"])
    return wide


# --- hard checks -------------------------------------------------------------


def power_schema(rules: dict) -> pa.DataFrameSchema:
    ranges, tol = rules["ranges"], rules["consistency_tolerance_c"]
    columns = {
        "cell_id": pa.Column(str, nullable=False),
        "obs_date": pa.Column("datetime64[us]", nullable=False, coerce=True),
        **{
            var: pa.Column(float, pa.Check.in_range(lo, hi), nullable=True, coerce=True)
            for var, (lo, hi) in ranges.items()
        },
    }
    checks = [
        pa.Check(
            lambda d: (d["T2M_MIN"] <= d["T2M"] + tol) | d[["T2M_MIN", "T2M"]].isna().any(axis=1),
            name="tmin_le_tmean",
        ),
        pa.Check(
            lambda d: (d["T2M"] <= d["T2M_MAX"] + tol) | d[["T2M", "T2M_MAX"]].isna().any(axis=1),
            name="tmean_le_tmax",
        ),
        pa.Check(
            lambda d: (d["T2MDEW"] <= d["T2M"] + tol) | d[["T2MDEW", "T2M"]].isna().any(axis=1),
            name="dew_le_tmean",
        ),
    ]
    return pa.DataFrameSchema(columns, checks=checks, unique=KEY, strict=True)


def hard_failures(wide: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Every hard-rule violation (empty DataFrame = all good)."""
    try:
        power_schema(rules).validate(wide, lazy=True)
    except pa.errors.SchemaErrors as e:
        cases = e.failure_cases
        cols = [
            c for c in ("schema_context", "column", "check", "index", "failure_case") if c in cases
        ]
        return cases[cols].reset_index(drop=True)
    return pd.DataFrame(columns=["schema_context", "column", "check", "index", "failure_case"])


# --- soft checks -------------------------------------------------------------


def neighbours(cells: pd.DataFrame) -> dict[str, list[str]]:
    """The up-to-8 adjacent grid cells of each cell (by grid indices j, i)."""
    idx = {(int(r.j), int(r.i)): r.cell_id for r in cells.itertuples(index=False)}
    return {
        cid: [
            idx[(j + dj, i + di)]
            for dj in (-1, 0, 1)
            for di in (-1, 0, 1)
            if (dj or di) and (j + dj, i + di) in idx
        ]
        for (j, i), cid in idx.items()
    }


def _robust_z_by_month(x: pd.DataFrame) -> pd.DataFrame:
    """(x - median) / (1.4826 * MAD), per column and calendar month."""
    month = x.index.month
    med = x.groupby(month).transform("median")
    mad = (x - med).abs().groupby(month).transform("median") * 1.4826
    return (x - med) / mad.replace(0, np.nan)


def isolated_jumps(
    wide: pd.DataFrame, cells: pd.DataFrame, variable: str, z: float, neighbour_z: float
) -> pd.DataFrame:
    """Day-to-day changes extreme for the cell and month, that no neighbour shared."""
    series = wide.pivot(index="obs_date", columns="cell_id", values=variable).sort_index()
    zs = _robust_z_by_month(series.diff())
    nbrs = neighbours(cells)
    rows = []
    for cid in series.columns:
        extreme = zs[cid].abs() > z
        if not extreme.any():
            continue
        shared = (zs[nbrs.get(cid, [])].abs() > neighbour_z).any(axis=1)
        for d in zs.index[extreme & ~shared]:
            rows.append(
                {
                    "check": "jump",
                    "variable": variable,
                    "cell_id": cid,
                    "obs_date": d,
                    "value": series.at[d, cid],
                    "detail": f"robust z {zs.at[d, cid]:+.1f}",
                }
            )
    return pd.DataFrame(
        rows, columns=["check", "variable", "cell_id", "obs_date", "value", "detail"]
    )


def spatial_outliers(
    wide: pd.DataFrame, cells: pd.DataFrame, variable: str, abs_dev_c: float
) -> pd.DataFrame:
    """Cells departing from their neighbours' median by more than abs_dev_c, beyond
    the cell's usual offset for that calendar month."""
    series = wide.pivot(index="obs_date", columns="cell_id", values=variable).sort_index()
    nbrs = neighbours(cells)
    dev = pd.DataFrame(
        {c: series[c] - series[nbrs[c]].median(axis=1) for c in series.columns if nbrs.get(c)}
    )
    anomaly = dev - dev.groupby(dev.index.month).transform("median")
    hits = anomaly.abs() > abs_dev_c
    rows = [
        {
            "check": "spatial",
            "variable": variable,
            "cell_id": c,
            "obs_date": d,
            "value": series.at[d, c],
            "detail": f"{anomaly.at[d, c]:+.1f} C vs neighbours",
        }
        for c in anomaly.columns
        for d in anomaly.index[hits[c]]
    ]
    return pd.DataFrame(
        rows, columns=["check", "variable", "cell_id", "obs_date", "value", "detail"]
    )


def soft_flags(wide: pd.DataFrame, cells: pd.DataFrame, rules: dict) -> pd.DataFrame:
    parts = [
        isolated_jumps(wide, cells, v, rules["jump"]["z"], rules["jump"]["neighbour_z"])
        for v in rules["jump"]["variables"]
    ] + [
        spatial_outliers(wide, cells, v, rules["spatial"]["abs_dev_c"])
        for v in rules["spatial"]["variables"]
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["check", "variable", "cell_id", "obs_date", "value", "detail"])
    return pd.concat(parts, ignore_index=True).sort_values(["obs_date", "cell_id"])


def mark_reviewed(flags: pd.DataFrame, reviewed: list[dict]) -> pd.DataFrame:
    """Attach the review reason to flags that have one; others get reason None."""
    lookup = {
        (r["check"], r["variable"], r["cell_id"], pd.Timestamp(r["obs_date"])): r["reason"]
        for r in reviewed or []
    }
    keys = zip(flags["check"], flags["variable"], flags["cell_id"], flags["obs_date"], strict=True)
    return flags.assign(reason=[lookup.get(k) for k in keys])


def null_rates(wide: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Share of missing values per variable and year."""
    return wide.groupby(wide["obs_date"].dt.year)[variables].apply(lambda g: g.isna().mean())


# --- IMD ---------------------------------------------------------------------


def imd_checks(imd_settings, raw_dir, years: list[int], bbox, tmax_range) -> pd.DataFrame:
    """Per-year IMD Tmax summary over the UP box: land points, gaps, range violations."""
    from heatwave.ingest.imd import read_grd, to_long

    lo, hi = tmax_range
    rows = []
    for year in years:
        data = read_grd(raw_dir / f"{year}.GRD", year, imd_settings)
        box = to_long(data, year, imd_settings, bbox=bbox)
        rows.append(
            {
                "year": year,
                "points": box[["lat", "lon"]].drop_duplicates().shape[0],
                "missing": int(box["tmax"].isna().sum()),
                "out_of_range": int(((box["tmax"] < lo) | (box["tmax"] > hi)).sum()),
                "max_tmax": round(float(box["tmax"].max()), 1),
                "point_days_ge_45": int((box["tmax"] >= 45).sum()),
            }
        )
    return pd.DataFrame(rows)
