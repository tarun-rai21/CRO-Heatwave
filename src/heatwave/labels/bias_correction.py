"""Monthly quantile mapping of POWER Tmax to station levels (docs/decisions/005).

POWER Tmax runs 2-3 C warm against UP stations and IMD's 1-degree grid runs
cool on hot days, so neither can feed IMD's absolute thresholds as served.
The mapping is fitted on pairs (POWER value of the cell containing a station,
that station's reported maximum) on the same day, pooled over stations, one
map per calendar month. It is fitted once on a fixed period and is part of
the label definition, like the normal; its parameters enter label_version.

Outside the fitted quantile range the end corrections are carried as constant
shifts, so the most extreme days are corrected like the 1st/99th percentile.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

QUANTILES = np.round(np.linspace(0.01, 0.99, 99), 2)


def fit_quantile_map(pairs: pd.DataFrame, period: tuple[int, int]) -> pd.DataFrame:
    """pairs: obs_date, source, target. Returns month, q, source_q, target_q,
    n_pairs, fit_start, fit_end."""
    years = pd.to_datetime(pairs["obs_date"]).dt.year
    p = pairs[years.between(*period)].dropna(subset=["source", "target"])
    rows = []
    for month, g in p.groupby(pd.to_datetime(p["obs_date"]).dt.month):
        src, tgt = np.quantile(g["source"], QUANTILES), np.quantile(g["target"], QUANTILES)
        rows += [
            {"month": int(month), "q": q, "source_q": s, "target_q": t, "n_pairs": len(g)}
            for q, s, t in zip(QUANTILES, src, tgt, strict=True)
        ]
    out = pd.DataFrame(rows)
    if out["month"].nunique() != 12:
        raise ValueError("quantile map needs pairs in all 12 months")
    out["fit_start"], out["fit_end"] = period
    return out


def apply_quantile_map(values, months, qmap: pd.DataFrame) -> np.ndarray:
    values, months = np.asarray(values, dtype=float), np.asarray(months)
    out = np.full_like(values, np.nan)
    for month, g in qmap.groupby("month"):
        sel = months == month
        src, tgt = g["source_q"].to_numpy(), g["target_q"].to_numpy()
        x = values[sel]
        y = np.interp(x, src, tgt)
        y = np.where(x < src[0], x + (tgt[0] - src[0]), y)
        y = np.where(x > src[-1], x + (tgt[-1] - src[-1]), y)
        out[sel] = y
    return out


def qmap_version(qmap: pd.DataFrame) -> str:
    """Content hash of the fitted map: changes if the data or method change."""
    cols = ["month", "q", "source_q", "target_q", "fit_start", "fit_end"]
    payload = json.dumps(np.round(qmap[cols].to_numpy(dtype=float), 4).tolist())
    return "qm_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


# --- spatial adjustment -----------------------------------------------------------
# POWER's warm bias is not uniform: it is ~1 C smaller over Bundelkhand than over the
# irrigated Ganga plains, both against IMD and against the Jhansi station
# (docs/decisions/005). Station coverage is too thin to map that, so the large-scale
# pattern comes from IMD: a plane in latitude/longitude fitted to the per-district
# mean (POWER - IMD) over the season. A plane keeps the regional gradient and
# discards IMD's local station-density artefacts. POWER minus the plane is then
# mapped to station levels.


def fit_bias_plane(
    power: pd.DataFrame,
    imd: pd.DataFrame,
    districts: pd.DataFrame,
    period: tuple[int, int],
    months: list[int],
) -> dict:
    """Least-squares plane bias = a + b*lat + c*lon over districts, where bias is
    the mean of (POWER Tmax - IMD Tmax) in `months` of `period`."""
    m = power[["district_id", "obs_date", "T2M_MAX"]].merge(
        imd[["district_id", "obs_date", "tmax_imd"]], on=["district_id", "obs_date"]
    )
    sel = m["obs_date"].dt.year.between(*period) & m["obs_date"].dt.month.isin(months)
    bias = (m[sel]["T2M_MAX"] - m[sel]["tmax_imd"]).groupby(m[sel]["district_id"]).mean()
    pos = districts.set_index("district_id").loc[bias.index]
    X = np.column_stack([np.ones(len(bias)), pos["centroid_lat"], pos["centroid_lon"]])
    (a, b, c), *_ = np.linalg.lstsq(X, bias.to_numpy(), rcond=None)
    resid = bias.to_numpy() - X @ np.array([a, b, c])
    return {
        "a": float(a),
        "b_lat": float(b),
        "c_lon": float(c),
        "fit_start": period[0],
        "fit_end": period[1],
        "months": list(months),
        "rmse_c": float(np.sqrt(np.mean(resid**2))),
        "n_districts": len(bias),
    }


def plane_value(plane: dict, lat, lon) -> np.ndarray:
    return (
        plane["a"]
        + plane["b_lat"] * np.asarray(lat, float)
        + plane["c_lon"] * np.asarray(lon, float)
    )


def plane_version(plane: dict) -> str:
    keys = ["a", "b_lat", "c_lon", "fit_start", "fit_end", "months"]
    payload = json.dumps(
        {k: (round(plane[k], 6) if isinstance(plane[k], float) else plane[k]) for k in keys},
        sort_keys=True,
    )
    return "pl_" + hashlib.sha256(payload.encode()).hexdigest()[:12]
