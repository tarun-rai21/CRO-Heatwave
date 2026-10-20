"""Grid cells -> districts (reference.md §2.2, §5.3 silver layer).

District value = area-weighted mean of its cells' values. If a cell is missing a
value on a day, the remaining weights are renormalised and the number of missing
cells is recorded, so a gap never silently becomes a smaller-area average.
For Tmax the hottest qualifying cell is also kept (cells covering >= a minimum
share of the district).

Silver tables are derived: they are rebuilt from the raw store, never edited.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from heatwave.ingest.raw_store import RawStore


def power_district_daily(
    store: RawStore, weights: pd.DataFrame, variables: list[str], cellmax_min_share: float
) -> pd.DataFrame:
    """One row per (district_id, obs_date): area-weighted mean of each variable,
    T2M_MAX_cellmax, and n_missing (cell-variable values absent that day)."""
    sql = store.best_sql()
    if sql is None:
        raise RuntimeError("raw store is empty")
    var_list = ", ".join(f"'{v}'" for v in variables)
    with duckdb.connect() as con:
        con.register("weights", weights[["district_id", "cell_id", "area_weight"]])
        long = con.sql(
            f"""
            WITH raw AS (
                SELECT cell_id, obs_date, variable, value FROM ({sql})
                WHERE variable IN ({var_list})
            )
            SELECT w.district_id, r.obs_date, r.variable,
                   sum(r.value * w.area_weight) / sum(CASE WHEN r.value IS NOT NULL
                                                       THEN w.area_weight END) AS value,
                   count(*) - count(r.value) AS n_missing,
                   max(CASE WHEN w.area_weight >= {cellmax_min_share} THEN r.value END)
                       AS cellmax
            FROM raw r JOIN weights w USING (cell_id)
            GROUP BY w.district_id, r.obs_date, r.variable
            """
        ).df()

    wide = long.pivot_table(
        index=["district_id", "obs_date"], columns="variable", values="value", aggfunc="first"
    )
    wide = wide.reindex(columns=variables)
    tmax = long[long["variable"] == "T2M_MAX"].set_index(["district_id", "obs_date"])
    wide["T2M_MAX_cellmax"] = tmax["cellmax"]
    wide["n_missing"] = long.groupby(["district_id", "obs_date"])["n_missing"].sum()
    out = wide.reset_index()
    out.columns.name = None
    out["obs_date"] = pd.to_datetime(out["obs_date"])
    return out.sort_values(["district_id", "obs_date"]).reset_index(drop=True)


def imd_district_daily(
    imd_settings, raw_dir: Path, years: list[int], weights: pd.DataFrame
) -> pd.DataFrame:
    """One row per (district_id, obs_date): area-weighted IMD Tmax (renormalised
    over cells with data) and n_missing."""
    from heatwave.ingest.imd import read_grd

    s = imd_settings
    j = np.rint((weights["lat"].to_numpy() - s.lat_first) / s.step_deg).astype(int)
    i = np.rint((weights["lon"].to_numpy() - s.lon_first) / s.step_deg).astype(int)
    w = weights["area_weight"].to_numpy()
    codes, districts = pd.factorize(weights["district_id"])

    frames = []
    for year in years:
        data = read_grd(raw_dir / f"{year}.GRD", year, s)  # (days, lat, lon)
        values = data[:, j, i]  # (days, n_pairs)
        has = ~np.isnan(values)
        num = np.zeros((len(values), len(districts)))
        den = np.zeros_like(num)
        miss = np.zeros_like(num)
        for k in range(len(districts)):
            cols = codes == k
            num[:, k] = np.nansum(values[:, cols] * w[cols], axis=1)
            den[:, k] = (has[:, cols] * w[cols]).sum(axis=1)
            miss[:, k] = (~has[:, cols]).sum(axis=1)
        with np.errstate(invalid="ignore"):
            tmax = np.where(den > 0, num / den, np.nan)
        dates = pd.date_range(f"{year}-01-01", periods=len(values), freq="D")
        frames.append(
            pd.DataFrame(
                {
                    "district_id": np.tile(np.asarray(districts), len(dates)),
                    "obs_date": np.repeat(dates, len(districts)),
                    "tmax_imd": tmax.ravel(),
                    "n_missing": miss.ravel().astype(int),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["district_id", "obs_date"]).reset_index(drop=True)
