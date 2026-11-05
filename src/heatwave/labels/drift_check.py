"""Does label Tmax keep its level against independent sources? (reference.md §5.6)

For each station that reports in both periods: the label's, POWER's and IMD's Tmax minus
the station's reported maximum, on the same station-days in the evaluation season, and
how that changes between periods. Run yearly when IMD publishes, before re-estimating
the drift term. Stations can change too, so one source is never read alone.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from heatwave.ingest import gsod

SOURCES = {"label": "tmax_used", "power": "tmax_raw", "imd": "tmax_imd"}
TMAX_RANGE = (10, 55)  # reported maxima outside this are station errors


def station_districts(
    stations: pd.DataFrame, districts: gpd.GeoDataFrame, crs: str
) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(
        stations[["station_id", "name"]],
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs=districts.crs,
    )
    found = gpd.sjoin_nearest(pts.to_crs(crs), districts[["district_id", "geometry"]].to_crs(crs))
    return found[["station_id", "name", "district_id"]].drop_duplicates("station_id")


def paired_days(
    days: pd.DataFrame,
    where: pd.DataFrame,
    labels: pd.DataFrame,
    imd: pd.DataFrame,
    months: list[int],
) -> pd.DataFrame:
    """One row per reported station-day, with each source minus the station."""
    d = days.dropna(subset=["tmax_c"]).merge(where, on="station_id")
    d = d.merge(labels, on=["district_id", "obs_date"]).merge(imd, on=["district_id", "obs_date"])
    d = d[d["obs_date"].dt.month.isin(months) & d["tmax_c"].between(*TMAX_RANGE)].copy()
    d["year"] = d["obs_date"].dt.year
    for name, col in SOURCES.items():
        d[name] = d[col] - d["tmax_c"]
    return d


def period_change(
    d: pd.DataFrame,
    before: tuple[int, int],
    after: tuple[int, int],
    min_days: int = 150,
    resamples: int = 5000,
    seed: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Per-station change between periods, stations with min_days in both; and the mean
    change with an interval from resampling years within each period."""
    a = d[d["year"].between(*before)]
    b = d[d["year"].between(*after)]
    na, nb = a.groupby("name").size(), b.groupby("name").size()
    keep = sorted(set(na[na >= min_days].index) & set(nb[nb >= min_days].index))
    names = list(SOURCES)
    by_station = (
        b[b["name"].isin(keep)].groupby("name")[names].mean()
        - a[a["name"].isin(keep)].groupby("name")[names].mean()
    )
    by_station["days_before"], by_station["days_after"] = na[keep], nb[keep]
    # Stations weigh equally in each year's value, so a changing mix of days counts less.
    both = d[d["name"].isin(keep)]
    yearly = both.groupby(["year", "name"])[names].mean().groupby("year").mean()
    rng = np.random.default_rng(seed)
    rows = []
    for k in names:
        x = yearly.loc[before[0] : before[1], k].to_numpy()
        y = yearly.loc[after[0] : after[1], k].to_numpy()
        draws = [
            rng.choice(y, len(y)).mean() - rng.choice(x, len(x)).mean() for _ in range(resamples)
        ]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append(
            {
                "source": k,
                "change": y.mean() - x.mean(),
                "lo": lo,
                "hi": hi,
                "stations_cooler": int((by_station[k] < 0).sum()),
                "stations": len(keep),
            }
        )
    return {
        "summary": pd.DataFrame(rows),
        "by_station": by_station.reset_index(),
        "yearly": yearly.reset_index(),
    }


def load_days(root: Path, g: dict) -> pd.DataFrame:
    frames = [gsod.read_station_days(root / g["raw_dir"], g)]
    if (root / g["check_raw_dir"]).exists():
        frames.append(gsod.read_station_days(root / g["check_raw_dir"], g))
    return pd.concat(frames, ignore_index=True)
