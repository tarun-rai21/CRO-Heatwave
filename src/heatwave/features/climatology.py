"""Fold-aware climatologies for features (reference.md §4.1).

Everything a feature compares against (normals, the hot-day threshold, the usual
heatwave frequency for a calendar day) is fitted on one block of years, normally
a fold's training years, and carries those years. Applying it to rows of a
validation or test block whose years overlap the fit years is refused: that is
the leak the walk-forward design exists to prevent.

The label's own normal (1991-2020) is part of the label definition and is
separate from these.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from heatwave.labels.normals import apply_normals, fit_normals


class ClimatologyLeak(ValueError):
    """A climatology was asked to serve rows from its own fit years as held-out data."""


@dataclass
class Climatology:
    fit_years: tuple[int, int]
    normals: dict[str, pd.DataFrame]  # variable -> harmonic coefficients per district
    hot_threshold: pd.DataFrame  # district_id, month, threshold (Tmax anomaly)
    hw_freq: pd.DataFrame  # district_id, doy (1..366), p
    meta: dict = field(default_factory=dict)

    @property
    def fit_end_date(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.fit_years[1], month=12, day=31)

    def check_held_out(self, years: tuple[int, int]) -> None:
        """Raise unless every year in `years` is after the fit years."""
        if years[0] <= self.fit_years[1]:
            raise ClimatologyLeak(
                f"climatology fitted on {self.fit_years} cannot serve held-out years {years}"
            )

    def normal(self, df: pd.DataFrame, variable: str) -> np.ndarray:
        return apply_normals(df, "district_id", self.normals[variable])

    def frequency(self, district_ids, dates) -> np.ndarray:
        """Climatological probability of a heatwave day for each (district, date)."""
        q = pd.DataFrame({"district_id": np.asarray(district_ids), "doy": _doy(dates)})
        return q.merge(self.hw_freq, on=["district_id", "doy"], how="left")["p"].to_numpy()


def _doy(dates) -> np.ndarray:
    """Day of year on a 366-day calendar: non-leap years skip Feb 29 (day 60),
    so the same calendar date always maps to the same index."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    doy = d.dayofyear.to_numpy().copy()
    late = (~d.is_leap_year) & (d.month > 2)
    doy[late] += 1
    return doy


def fit_climatology(
    daily: pd.DataFrame, labels: pd.DataFrame, years: tuple[int, int], cfg: dict
) -> Climatology:
    """daily: district_id, obs_date, tmax and the variables in cfg normal_variables.
    labels: district_id, obs_date, is_heatwave."""
    c = cfg["climatology"]
    in_fit = daily["obs_date"].dt.year.between(*years)
    fit = daily[in_fit]
    normals = {
        v: fit_normals(fit, "district_id", v, years, c["normal_harmonics"])
        for v in c["normal_variables"]
    }

    # Hot-day threshold: per district and calendar month, the q-quantile of the
    # Tmax anomaly in the fit years.
    anom = fit["tmax"].to_numpy() - apply_normals(fit, "district_id", normals["tmax"])
    hot = (
        pd.DataFrame(
            {
                "district_id": fit["district_id"].to_numpy(),
                "month": fit["obs_date"].dt.month.to_numpy(),
                "anom": anom,
            }
        )
        .groupby(["district_id", "month"])["anom"]
        .quantile(c["hot_quantile"])
        .rename("threshold")
        .reset_index()
    )

    # Heatwave frequency per district and calendar day, +/- window, fit years only.
    lab = labels[labels["obs_date"].dt.year.between(*years)].dropna(subset=["is_heatwave"])
    w = int(c["hw_freq_window_days"])
    rows = []
    for d, g in lab.groupby("district_id"):
        counts = np.zeros(367)
        totals = np.zeros(367)
        doy = _doy(g["obs_date"])
        np.add.at(counts, doy, g["is_heatwave"].astype(float).to_numpy())
        np.add.at(totals, doy, 1.0)
        days = np.arange(1, 367)
        circ_c, circ_t = np.zeros(366), np.zeros(366)
        for off in range(-w, w + 1):
            idx = (days - 1 + off) % 366 + 1
            circ_c += counts[idx]
            circ_t += totals[idx]
        rows.append(
            pd.DataFrame(
                {
                    "district_id": d,
                    "doy": days,
                    "p": np.where(circ_t > 0, circ_c / np.maximum(circ_t, 1), 0.0),
                }
            )
        )
    hw_freq = pd.concat(rows, ignore_index=True)
    return Climatology(
        fit_years=tuple(years),
        normals=normals,
        hot_threshold=hot,
        hw_freq=hw_freq,
        meta={"n_rows": int(in_fit.sum())},
    )
