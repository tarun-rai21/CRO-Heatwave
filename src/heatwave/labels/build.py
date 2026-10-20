"""Build the heatwave labels (reference.md §2.4, docs/decisions/005).

Main label:  POWER district Tmax (area-weighted), minus a smooth spatial bias
             plane (from IMD's large-scale pattern), mapped to station levels with
             a monthly quantile map fitted once on station pairs in the bias
             period; then the IMD plains rule against a smoothed 1991-2020 normal
             of the corrected series.
Sensitivity: the same rule on IMD gridded district Tmax (own normal), and a
             percentile label (Tmax above the calendar-day 95th percentile for
             >= 3 consecutive days).
Episodes:    consecutive days with >= N districts in heatwave, state-wide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heatwave.config import HeatwaveRule
from heatwave.labels.bias_correction import (
    apply_quantile_map,
    fit_quantile_map,
    plane_value,
    plane_version,
    qmap_version,
)
from heatwave.labels.definition import classify, label_version
from heatwave.labels.normals import apply_normals, fit_normals


def fit_station_map(
    series: pd.DataFrame, stations: list[str], period, plane: dict | None = None
) -> pd.DataFrame:
    """Quantile map POWER (cell at station, minus the spatial plane at the station
    if given) -> station reported maximum, pooled over the given stations."""
    pairs = series[series["station_id"].isin(stations) & series["station"].notna()].copy()
    if plane is not None:
        pairs["power"] = pairs["power"] - plane_value(plane, pairs["lat"], pairs["lon"])
    return fit_quantile_map(pairs.rename(columns={"power": "source", "station": "target"}), period)


def rule_labels(df: pd.DataFrame, unit: str, value: str, rule: HeatwaveRule) -> pd.DataFrame:
    """Apply the rule with the unit's own normal of `value`."""
    coefs = fit_normals(df, unit, value, rule.normal_period, rule.normal_n_harmonics)
    labels = classify(df[value], apply_normals(df, unit, coefs), rule)
    return pd.concat([df[[unit, "obs_date"]].reset_index(drop=True), labels], axis=1)


def main_labels(
    power: pd.DataFrame,
    qmap: pd.DataFrame,
    rule: HeatwaveRule,
    plane: dict | None = None,
    districts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """power: district_id, obs_date, T2M_MAX. With a plane, `districts` gives each
    district's centroid (centroid_lat, centroid_lon). Returns the labels_daily table."""
    df = power[["district_id", "obs_date", "T2M_MAX"]].copy()
    source = df["T2M_MAX"].to_numpy()
    extra = {"qmap": qmap_version(qmap)}
    if plane is not None:
        pos = df[["district_id"]].merge(districts, on="district_id", how="left")
        source = source - plane_value(plane, pos["centroid_lat"], pos["centroid_lon"])
        extra["plane"] = plane_version(plane)
    df["tmax_corrected"] = apply_quantile_map(source, df["obs_date"].dt.month, qmap)
    out = rule_labels(df, "district_id", "tmax_corrected", rule)
    out.insert(2, "tmax_raw", df["T2M_MAX"].to_numpy())
    out["label_version"] = label_version(rule, extra)
    return out


def percentile_label(
    df: pd.DataFrame, unit: str, value: str, period, q: float, window_days: int, min_run_days: int
) -> pd.Series:
    """True on days in a run of >= min_run_days consecutive days above the
    calendar-day q-th percentile (values within +/- window_days, in `period`)."""
    d = df[[unit, "obs_date", value]].copy()
    doy = d["obs_date"].dt.dayofyear.clip(upper=365)
    ref = d[d["obs_date"].dt.year.between(*period)]
    ref_doy = ref["obs_date"].dt.dayofyear.clip(upper=365)
    thresholds = {}
    for u, g in ref.groupby(unit):
        v, gd = g[value].to_numpy(), ref_doy[g.index].to_numpy()
        thr = np.empty(366)
        for day in range(1, 366):
            dist = np.minimum(np.abs(gd - day), 365 - np.abs(gd - day))
            thr[day] = np.nanquantile(v[dist <= window_days], q)
        thresholds[u] = thr
    above = np.array([thresholds[u][k] for u, k in zip(d[unit], doy, strict=True)])
    hot = d[value].to_numpy() > above

    flag = np.zeros(len(d), dtype=bool)
    for _, idx in d.groupby(unit).indices.items():
        h = hot[idx]
        run_id = np.cumsum(~h)  # hot days in one run share a run id
        counts = pd.Series(h).groupby(run_id).transform("sum").to_numpy()
        flag[idx] = h & (counts >= min_run_days)
    return pd.Series(flag, index=df.index)


def episodes(labels: pd.DataFrame, min_districts: int = 1) -> pd.DataFrame:
    """State-wide episodes: maximal runs of consecutive dates with at least
    min_districts districts in heatwave."""
    daily = (
        labels.assign(hw=labels["is_heatwave"].fillna(False).astype(bool))
        .groupby("obs_date")["hw"]
        .sum()
    )
    active = daily[daily >= min_districts]
    if active.empty:
        return pd.DataFrame(
            columns=[
                "episode",
                "start",
                "end",
                "days",
                "district_days",
                "peak_districts",
                "peak_date",
            ]
        )
    dates = active.index.to_series()
    episode = (dates.diff() != pd.Timedelta(days=1)).cumsum()
    rows = []
    for k, g in active.groupby(episode.to_numpy()):
        rows.append(
            {
                "episode": int(k),
                "start": g.index.min(),
                "end": g.index.max(),
                "days": len(g),
                "district_days": int(g.sum()),
                "peak_districts": int(g.max()),
                "peak_date": g.idxmax(),
            }
        )
    return pd.DataFrame(rows)
