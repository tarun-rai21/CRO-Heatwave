"""Smoothed daily normals (reference.md §2.4).

The normal for a unit (district, cell or station) is a harmonic regression on
the annual cycle, fitted on a fixed period (IMD's 1991-2020). Time is the
fraction of the year elapsed, so leap years line up with ordinary years.
Six harmonics were the best of the methods tried in an out-of-sample
comparison (docs/decisions/005); all were within 0.02 C RMSE of each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def year_fraction(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    days = np.where(d.dt.is_leap_year, 366, 365)
    return (d.dt.dayofyear.to_numpy() - 1) / days


def design(dates: pd.Series, n_harmonics: int) -> np.ndarray:
    t = 2 * np.pi * year_fraction(dates)
    cols = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        cols += [np.sin(k * t), np.cos(k * t)]
    return np.column_stack(cols)


def fit_normals(
    df: pd.DataFrame, unit: str, value: str, period: tuple[int, int], n_harmonics: int
) -> pd.DataFrame:
    """Harmonic coefficients per unit, fitted on non-null values within `period`.
    Returns one row per unit: c0..c{2k}, n_obs, fit_start, fit_end."""
    years = pd.to_datetime(df["obs_date"]).dt.year
    fit = df[years.between(*period) & df[value].notna()]
    rows = []
    for u, g in fit.groupby(unit):
        coef, *_ = np.linalg.lstsq(
            design(g["obs_date"], n_harmonics), g[value].to_numpy(), rcond=None
        )
        rows.append({unit: u, **{f"c{k}": c for k, c in enumerate(coef)}, "n_obs": len(g)})
    out = pd.DataFrame(rows)
    out["fit_start"], out["fit_end"] = period
    return out


def apply_normals(df: pd.DataFrame, unit: str, coefs: pd.DataFrame) -> np.ndarray:
    """Normal for every row of df (obs_date, unit) from fitted coefficients."""
    ccols = sorted(
        (c for c in coefs.columns if c[0] == "c" and c[1:].isdigit()), key=lambda c: int(c[1:])
    )
    n_harmonics = (len(ccols) - 1) // 2
    c = df[[unit]].merge(coefs[[unit, *ccols]], on=unit, how="left")[ccols].to_numpy()
    if np.isnan(c).any():
        missing = sorted(set(df[unit]) - set(coefs[unit]))
        raise ValueError(f"no normal fitted for {missing[:5]}")
    return (design(df["obs_date"], n_harmonics) * c).sum(axis=1)
