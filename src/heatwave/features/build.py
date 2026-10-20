"""As-of features (reference.md §4.1-4.2).

A feature row for (district, as-of date t) uses only data dated <= t: every
rolling window looks backwards, nothing is centred, and cross-district features
(state mean, upwind) use the same day t. Climatology-based features use a
Climatology fitted on a fold's training years (features.climatology).

Every feature is registered in FEATURES with its group and physical reason;
docs/features.md is generated from this registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from heatwave.features.climatology import Climatology
from heatwave.labels.normals import year_fraction


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    description: str
    uses_climatology: bool = False


GROUP_REASONS = {
    "current heat": "How close the district is to the heatwave thresholds now",
    "persistence": "Heatwaves come from slow, blocked weather patterns that persist",
    "trend": "Whether heat is building up or breaking down",
    "land surface": "Dry soil heats up more (less evaporative cooling): the main slow driver",
    "moisture and wind": (
        "Dry north-westerly flow vs moist easterlies; monsoon arrival ends the season"
    ),
    "season": "The baseline expectation for the time of year",
    "spatial": "Where the district is, and heat arriving from the west / north-west",
    "lead": "Lets one model share strength across lead days",
}

FEATURES = [
    Feature("tmax", "current heat", "Tmax at t (label scale: bias-corrected), C"),
    Feature("tmin", "current heat", "Tmin at t, C"),
    Feature("tmax_anom", "current heat", "Tmax minus fold normal, C", True),
    Feature("tmin_anom", "current heat", "Tmin minus fold normal, C", True),
    Feature("diurnal_range", "current heat", "Raw POWER Tmax minus Tmin at t, C"),
    Feature("tmax_anom_mean3", "persistence", "Mean Tmax anomaly over the 3 days to t", True),
    Feature("tmax_anom_mean7", "persistence", "Mean Tmax anomaly over the 7 days to t", True),
    Feature("tmax_anom_mean14", "persistence", "Mean Tmax anomaly over the 14 days to t", True),
    Feature("tmax_max7", "persistence", "Highest Tmax over the 7 days to t, C"),
    Feature("hot_days7", "persistence", "Days in the 7 to t above the hot-day threshold", True),
    Feature("hot_days14", "persistence", "Days in the 14 to t above the hot-day threshold", True),
    Feature("consec_hot_days", "persistence", "Consecutive hot days ending at t (capped)", True),
    Feature("hw_today", "persistence", "Heatwave day at t (label), 1/0"),
    Feature("days_since_hw", "persistence", "Days since the last heatwave day up to t (capped)"),
    Feature("dtmax1", "trend", "Tmax change over 1 day to t, C"),
    Feature("dtmax3", "trend", "Tmax change over 3 days to t, C"),
    Feature("dtmax7", "trend", "Tmax change over 7 days to t, C"),
    Feature("tmax_slope7", "trend", "Least-squares slope of Tmax over the 7 days to t, C/day"),
    Feature("gwettop", "land surface", "Surface soil wetness at t (0-1)"),
    Feature("gwetroot", "land surface", "Root-zone soil wetness at t (0-1)"),
    Feature("gwettop_anom", "land surface", "Surface soil wetness minus fold normal", True),
    Feature("gwetroot_anom", "land surface", "Root-zone soil wetness minus fold normal", True),
    Feature("rain7", "land surface", "Rainfall over the 7 days to t, mm"),
    Feature("rain30", "land surface", "Rainfall over the 30 days to t, mm"),
    Feature("rain90", "land surface", "Rainfall over the 90 days to t, mm"),
    Feature("days_since_rain", "land surface", "Days since the last day with > 1 mm (capped)"),
    Feature("dewpoint", "moisture and wind", "Dew point at t, C"),
    Feature("rh", "moisture and wind", "Relative humidity at t, %"),
    Feature("qv", "moisture and wind", "Specific humidity at t, g/kg"),
    Feature("wind_speed", "moisture and wind", "Wind speed at 10 m at t, m/s"),
    Feature("wind_u", "moisture and wind", "Eastward wind at 10 m at t, m/s (westerly > 0)"),
    Feature("wind_v", "moisture and wind", "Northward wind at 10 m at t, m/s"),
    Feature("ps_anom", "moisture and wind", "Surface pressure minus fold normal, kPa", True),
    Feature("doy_sin", "season", "sin of the fraction of the year at t"),
    Feature("doy_cos", "season", "cos of the fraction of the year at t"),
    Feature(
        "clim_hw_freq_t",
        "season",
        "Usual heatwave-day frequency for the district on t's calendar day (fold years)",
        True,
    ),
    Feature("lat", "spatial", "District centroid latitude"),
    Feature("lon", "spatial", "District centroid longitude"),
    Feature("elevation_m", "spatial", "Area-weighted elevation of the district's cells, m"),
    Feature("state_tmax_anom", "spatial", "Mean Tmax anomaly over all districts at t", True),
    Feature(
        "upwind_tmax_anom",
        "spatial",
        "Mean Tmax anomaly at t of districts within 250 km to the west / north-west",
        True,
    ),
    Feature("lead", "lead", "Lead day h (1-10); per-lead table only"),
    Feature(
        "clim_hw_freq_valid",
        "lead",
        "Usual heatwave-day frequency on the valid date s+h; per-lead table only",
        True,
    ),
]
ASOF_FEATURES = [f.name for f in FEATURES if f.group != "lead"]
LEAD_FEATURES = [f.name for f in FEATURES if f.group == "lead"]


def upwind_neighbours(districts: pd.DataFrame, max_km: float, bearing: tuple[float, float]):
    """For each district, districts within max_km whose direction from it lies in
    the bearing range (degrees clockwise from north)."""
    lat = np.radians(districts["centroid_lat"].to_numpy())
    lon = np.radians(districts["centroid_lon"].to_numpy())
    ids = districts["district_id"].to_numpy()
    out = {}
    for k, d in enumerate(ids):
        dlon = lon - lon[k]
        a = np.sin((lat - lat[k]) / 2) ** 2 + np.cos(lat[k]) * np.cos(lat) * np.sin(dlon / 2) ** 2
        dist = 2 * 6371 * np.arcsin(np.sqrt(a))
        brg = (
            np.degrees(
                np.arctan2(
                    np.sin(dlon) * np.cos(lat),
                    np.cos(lat[k]) * np.sin(lat) - np.sin(lat[k]) * np.cos(lat) * np.cos(dlon),
                )
            )
            % 360
        )
        sel = (dist > 0) & (dist <= max_km) & (brg >= bearing[0]) & (brg <= bearing[1])
        out[d] = list(ids[sel])
    return out


def _spine(daily: pd.DataFrame) -> pd.DataFrame:
    """Complete daily date spine per district, so day-count windows are calendar days."""
    start, end = daily["obs_date"].min(), daily["obs_date"].max()
    idx = pd.MultiIndex.from_product(
        [sorted(daily["district_id"].unique()), pd.date_range(start, end, freq="D")],
        names=["district_id", "obs_date"],
    )
    return daily.set_index(["district_id", "obs_date"]).reindex(idx).reset_index()


def _rolling(g, col, window, how, min_share):
    r = g[col].rolling(window, min_periods=max(1, math.ceil(window * min_share)))
    return getattr(r, how)().reset_index(level=0, drop=True)


def _days_since(flag: pd.Series, district: pd.Series, cap: int) -> np.ndarray:
    """Days since flag was last True (0 on a True day), per district, capped."""
    out = np.full(len(flag), float(cap))
    f = flag.fillna(False).to_numpy(bool)
    for idx in district.groupby(district).indices.values():
        last = None
        for j, i in enumerate(idx):
            if f[i]:
                last = j
            out[i] = cap if last is None else min(j - last, cap)
    return out


def _run_length(flag: pd.Series, district: pd.Series, cap: int) -> np.ndarray:
    f = flag.fillna(False).to_numpy(bool)
    out = np.zeros(len(f))
    for idx in district.groupby(district).indices.values():
        run = 0
        for i in idx:
            run = run + 1 if f[i] else 0
            out[i] = min(run, cap)
    return out


def compute_features(
    daily: pd.DataFrame, labels: pd.DataFrame, clim: Climatology, districts: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    """One row per (district_id, asof_date) for every day in `daily`.

    daily: district_id, obs_date, tmax (label scale), T2M_MAX (raw), T2M_MIN, T2MDEW,
           RH2M, QV2M, PRECTOTCORR, WS10M, U10M, V10M, PS, GWETTOP, GWETROOT
    labels: district_id, obs_date, is_heatwave
    districts: district_id, centroid_lat, centroid_lon, elevation_m
    """
    w, caps, share = cfg["windows"], cfg["caps"], cfg["windows"]["min_valid_share"]
    d = _spine(daily).merge(
        labels[["district_id", "obs_date", "is_heatwave"]],
        on=["district_id", "obs_date"],
        how="left",
    )
    d = d.sort_values(["district_id", "obs_date"]).reset_index(drop=True)
    g = d.groupby("district_id", sort=False)
    out = d[["district_id", "obs_date"]].rename(columns={"obs_date": "asof_date"}).copy()

    # current heat
    out["tmax"], out["tmin"] = d["tmax"], d["T2M_MIN"]
    anom = d["tmax"] - clim.normal(d, "tmax")
    out["tmax_anom"] = anom
    out["tmin_anom"] = d["T2M_MIN"] - clim.normal(d, "T2M_MIN")
    out["diurnal_range"] = d["T2M_MAX"] - d["T2M_MIN"]

    # persistence
    d["_anom"] = anom
    g = d.groupby("district_id", sort=False)
    for n in w["anomaly_means"]:
        out[f"tmax_anom_mean{n}"] = _rolling(g, "_anom", n, "mean", share)
    out[f"tmax_max{w['tmax_max']}"] = _rolling(g, "tmax", w["tmax_max"], "max", share)
    thr = (
        d[["district_id"]]
        .assign(month=d["obs_date"].dt.month)
        .merge(clim.hot_threshold, on=["district_id", "month"], how="left")["threshold"]
        .to_numpy()
    )
    d["_hot"] = np.where(anom.isna(), np.nan, (anom.to_numpy() > thr).astype(float))
    g = d.groupby("district_id", sort=False)
    for n in w["hot_day_counts"]:
        out[f"hot_days{n}"] = _rolling(g, "_hot", n, "sum", share)
    out["consec_hot_days"] = _run_length(
        d["_hot"] == 1, d["district_id"], caps["consecutive_hot_days"]
    )
    out["hw_today"] = d["is_heatwave"].astype("float")
    out["days_since_hw"] = _days_since(
        d["is_heatwave"].astype("boolean"), d["district_id"], caps["days_since_heatwave"]
    )

    # trend
    for k in w["deltas"]:
        out[f"dtmax{k}"] = d["tmax"] - g["tmax"].shift(k)
    n = w["slope_days"]
    weights = (np.arange(n) - (n - 1) / 2) / np.sum((np.arange(n) - (n - 1) / 2) ** 2)
    # Least-squares slope over the n days to t = weighted sum of the n values
    # (needs all n present: any gap leaves NaN).
    out[f"tmax_slope{n}"] = sum(weights[k] * g["tmax"].shift(n - 1 - k) for k in range(n))

    # land surface
    out["gwettop"], out["gwetroot"] = d["GWETTOP"], d["GWETROOT"]
    out["gwettop_anom"] = d["GWETTOP"] - clim.normal(d, "GWETTOP")
    out["gwetroot_anom"] = d["GWETROOT"] - clim.normal(d, "GWETROOT")
    for n in w["rain_sums"]:
        out[f"rain{n}"] = _rolling(g, "PRECTOTCORR", n, "sum", share)
    out["days_since_rain"] = _days_since(
        d["PRECTOTCORR"] > cfg["rain_day_mm"], d["district_id"], caps["days_since_rain"]
    )

    # moisture and wind
    out["dewpoint"], out["rh"], out["qv"] = d["T2MDEW"], d["RH2M"], d["QV2M"]
    out["wind_speed"], out["wind_u"], out["wind_v"] = d["WS10M"], d["U10M"], d["V10M"]
    out["ps_anom"] = d["PS"] - clim.normal(d, "PS")

    # season
    t = 2 * np.pi * year_fraction(d["obs_date"])
    out["doy_sin"], out["doy_cos"] = np.sin(t), np.cos(t)
    out["clim_hw_freq_t"] = clim.frequency(d["district_id"], d["obs_date"])

    # spatial
    static = districts.set_index("district_id")
    out["lat"] = d["district_id"].map(static["centroid_lat"]).to_numpy()
    out["lon"] = d["district_id"].map(static["centroid_lon"]).to_numpy()
    out["elevation_m"] = d["district_id"].map(static["elevation_m"]).to_numpy()
    by_day = pd.DataFrame({"district_id": d["district_id"], "date": d["obs_date"], "anom": anom})
    out["state_tmax_anom"] = by_day.groupby("date")["anom"].transform("mean").to_numpy()
    wide = by_day.pivot(index="date", columns="district_id", values="anom")
    nbrs = upwind_neighbours(
        districts, cfg["upwind"]["max_distance_km"], tuple(cfg["upwind"]["bearing_deg"])
    )
    upwind = pd.DataFrame(
        {k: wide[v].mean(axis=1) if v else np.nan for k, v in nbrs.items()}, index=wide.index
    )
    out["upwind_tmax_anom"] = (
        upwind.stack(future_stack=True)
        .reindex(pd.MultiIndex.from_arrays([d["obs_date"], d["district_id"]]))
        .to_numpy()
    )

    return out[["district_id", "asof_date", *ASOF_FEATURES]]
