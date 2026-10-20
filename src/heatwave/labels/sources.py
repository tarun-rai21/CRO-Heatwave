"""Which Tmax should the heatwave rule be applied to? (reference.md §2.4)

Candidates are scored on how well their heatwave days reproduce the heatwave
days of weather stations: IMD's criteria are defined on station Tmax. Each
candidate is taken at the grid cell containing the station:

    power        POWER Tmax as served
    imd          IMD 1-degree gridded Tmax
    power_to_imd POWER mapped (monthly quantiles) to IMD at the same place
    power_to_st  POWER mapped to station levels, fitted on the OTHER stations
                 only (leave one station out), as it would be applied to a
                 district with no station
    power_to_st_plane  as power_to_st, after removing the spatial bias plane
                 (IMD's large-scale pattern) at each location (the chosen label)

Each candidate uses its own normal (same period and smoothing as the label),
and only development years are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heatwave.config import HeatwaveRule
from heatwave.labels.bias_correction import apply_quantile_map, fit_quantile_map, plane_value
from heatwave.labels.definition import classify
from heatwave.labels.normals import apply_normals, fit_normals

CANDIDATES = ["power", "imd", "power_to_imd", "power_to_st", "power_to_st_plane"]


def station_series(
    station_days: pd.DataFrame,
    stations: pd.DataFrame,
    power_cells: pd.DataFrame,
    imd_cells: pd.DataFrame,
    years: tuple[int, int],
) -> pd.DataFrame:
    """Full daily series per station: station (reported max or null), power and
    imd values of the cells containing it.
    power_cells: station_id, obs_date, power; imd_cells: station_id, obs_date, imd."""
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31", freq="D")
    grid = pd.MultiIndex.from_product(
        [stations["station_id"], dates], names=["station_id", "obs_date"]
    ).to_frame(index=False)
    out = (
        grid.merge(station_days[["station_id", "obs_date", "tmax_c"]], how="left")
        .rename(columns={"tmax_c": "station"})
        .merge(power_cells, how="left")
        .merge(imd_cells, how="left")
    )
    return out.merge(stations[["station_id", "name"]], how="left")


def qualifying(
    series: pd.DataFrame,
    period: tuple[int, int],
    months=(3, 4, 5, 6),
    min_seasons: int = 10,
    min_coverage: float = 0.7,
) -> list[str]:
    """Stations with >= min_seasons seasons in `period` whose season days have a
    reported maximum at least min_coverage of the time."""
    s = series[
        series["obs_date"].dt.year.between(*period) & series["obs_date"].dt.month.isin(months)
    ]
    cov = s.groupby(["station_id", s["obs_date"].dt.year])["station"].apply(
        lambda x: x.notna().mean()
    )
    good = (cov >= min_coverage).groupby(level=0).sum()
    return sorted(good[good >= min_seasons].index)


def _with_candidates(
    series: pd.DataFrame, period: tuple[int, int], plane: dict, fit_stations: list[str]
) -> pd.DataFrame:
    """Station-based maps for a station are fitted on the OTHER fit stations only."""
    s = series.copy()
    month = s["obs_date"].dt.month
    s["power_to_imd"] = np.nan
    s["power_to_st"] = np.nan
    s["power_to_st_plane"] = np.nan
    s["power_net"] = s["power"] - plane_value(plane, s["lat"], s["lon"])
    for sid, g in s.groupby("station_id"):
        qm_imd = fit_quantile_map(g.rename(columns={"power": "source", "imd": "target"}), period)
        s.loc[g.index, "power_to_imd"] = apply_quantile_map(g["power"], month[g.index], qm_imd)
        others = s[
            (s["station_id"] != sid) & s["station_id"].isin(fit_stations) & s["station"].notna()
        ]
        qm_st = fit_quantile_map(
            others.rename(columns={"power": "source", "station": "target"}), period
        )
        s.loc[g.index, "power_to_st"] = apply_quantile_map(g["power"], month[g.index], qm_st)
        qm_pl = fit_quantile_map(
            others[["obs_date", "power_net", "station"]].rename(
                columns={"power_net": "source", "station": "target"}
            ),
            period,
        )
        s.loc[g.index, "power_to_st_plane"] = apply_quantile_map(
            g["power"] - plane_value(plane, g["lat"], g["lon"]), month[g.index], qm_pl
        )
    return s


def _kappa(pred: np.ndarray, truth: np.ndarray) -> float:
    po = (pred == truth).mean()
    pe = pred.mean() * truth.mean() + (1 - pred.mean()) * (1 - truth.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def compare(
    series: pd.DataFrame,
    rule: HeatwaveRule,
    eval_years: tuple[int, int],
    months: tuple[int, ...],
    plane: dict,
    fit_stations: list[str],
) -> pd.DataFrame:
    """Per station and candidate: heatwave days, POD, FAR, Cohen's kappa against
    the station's own heatwave days, on season days with a reported maximum.
    Stations not in fit_stations are scored as out-of-sample checks (role
    "check"); their own normal may rest on few seasons, so they are indicative."""
    period, k = rule.normal_period, rule.normal_n_harmonics
    s = _with_candidates(series, period, plane, fit_stations)
    for col in ["station", *CANDIDATES]:
        coefs = fit_normals(s, "station_id", col, period, k)
        s[col + "_normal"] = apply_normals(s, "station_id", coefs)

    ev = s[
        s["obs_date"].dt.year.between(*eval_years)
        & s["obs_date"].dt.month.isin(months)
        & s["station"].notna()
    ]
    rows = []
    for sid, g in ev.groupby("station_id"):
        truth = classify(g["station"], g["station_normal"], rule)["is_heatwave"].to_numpy(bool)
        for col in CANDIDATES:
            pred = classify(g[col], g[col + "_normal"], rule)["is_heatwave"].to_numpy(bool)
            tp, fp, fn = (pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()
            rows.append(
                {
                    "station_id": sid,
                    "name": g["name"].iloc[0],
                    "role": "fit" if sid in fit_stations else "check",
                    "seasons": int(g["obs_date"].dt.year.nunique()),
                    "candidate": col,
                    "station_days": int(truth.sum()),
                    "candidate_days": int(pred.sum()),
                    "POD": tp / (tp + fn) if tp + fn else np.nan,
                    "FAR": fp / (tp + fp) if tp + fp else np.nan,
                    "kappa": _kappa(pred, truth),
                    "bias_c": float((g[col] - g["station"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    """Per role (fit / check) and candidate: totals over stations (days pooled)
    and means of per-station scores."""
    keys = ["role", "candidate"]
    tot = scores.groupby(keys)[["station_days", "candidate_days"]].sum()
    tot["ratio"] = tot["candidate_days"] / tot["station_days"]
    means = scores.groupby(keys)[["kappa", "POD", "FAR", "bias_c"]].mean()
    out = tot.join(means).reset_index()
    out["candidate"] = pd.Categorical(out["candidate"], CANDIDATES, ordered=True)
    return out.sort_values(["role", "candidate"], ascending=[False, True]).set_index(keys)
