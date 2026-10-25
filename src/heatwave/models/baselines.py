"""Baselines M0-M2 (reference.md §4.4), for the per-lead and the window target.

All share one interface: fit(train_rows) -> self, predict(rows) -> probabilities.
Everything fitted comes from the training rows passed in (one fold's training
years); nothing reads other data.

    M0  climatology: the usual heatwave frequency for the district and date
    M1  persistence lookup: heatwave at t x Tmax-anomaly bin x season band (x lead),
        shrunk toward climatology in sparse cells
    M2  logistic regression on ~6 physically chosen features; one model per lead
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from heatwave.features.climatology import _doy


def _band(months, bands: dict[str, list[int]]) -> np.ndarray:
    lookup = {m: name for name, ms in bands.items() for m in ms}
    return np.array([lookup.get(int(m), "other") for m in np.asarray(months)])


def _logit(p, clip):
    p = np.clip(np.asarray(p, dtype=float), *clip)
    return np.log(p / (1 - p))


# --- M0 -----------------------------------------------------------------------------


class M0Lead:
    """Climatological frequency at the valid date (fold climatology)."""

    name = "M0"

    def fit(self, train: pd.DataFrame):
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["clim_hw_freq_valid"].to_numpy(dtype=float)


class M0Window:
    """Frequency of a heatwave in the next 10 days for the district and the issue
    date's calendar day, from training rows, smoothed +/- smooth_days."""

    name = "M0"

    def __init__(self, cfg: dict):
        self.w = int(cfg["baselines"]["m0_window"]["smooth_days"])

    def fit(self, train: pd.DataFrame):
        rows = []
        for d, g in train.groupby("district_id"):
            doy = _doy(g["issue_date"])
            counts, totals = np.zeros(367), np.zeros(367)
            np.add.at(counts, doy, g["y"].to_numpy(dtype=float))
            np.add.at(totals, doy, 1.0)
            days = np.arange(1, 367)
            c, t = np.zeros(366), np.zeros(366)
            for off in range(-self.w, self.w + 1):
                idx = (days - 1 + off) % 366 + 1
                c += counts[idx]
                t += totals[idx]
            rows.append(
                pd.DataFrame(
                    {
                        "district_id": d,
                        "doy": days,
                        "p": np.where(t > 0, c / np.maximum(t, 1), np.nan),
                    }
                )
            )
        self.table = pd.concat(rows, ignore_index=True)
        self.overall = float(train["y"].mean())
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        q = pd.DataFrame(
            {"district_id": rows["district_id"].to_numpy(), "doy": _doy(rows["issue_date"])}
        )
        p = q.merge(self.table, on=["district_id", "doy"], how="left")["p"].to_numpy()
        return np.where(np.isnan(p), self.overall, p)


# --- M1 -----------------------------------------------------------------------------


class _Lookup:
    """Shrunk cell means: p = (sum y + a * mean prior) / (n + a)."""

    def __init__(self, cfg: dict, keys: list[str], date_col: str):
        c = cfg["baselines"]["m1"]
        self.n_bins, self.bands, self.a = int(c["anomaly_bins"]), c["season_bands"], c["shrinkage"]
        self.keys, self.date_col = keys, date_col

    def _cells(self, rows: pd.DataFrame) -> pd.DataFrame:
        cells = pd.DataFrame(index=rows.index)
        for k in self.keys:
            cells[k] = rows[k].to_numpy()
        cells["hw"] = rows["hw_today"].fillna(0).to_numpy()
        cells["bin"] = np.digitize(rows["tmax_anom"].to_numpy(dtype=float), self.edges)
        cells["band"] = _band(rows[self.date_col].dt.month, self.bands)
        return cells

    def fit_prior(self, train: pd.DataFrame, prior: np.ndarray):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = np.nanquantile(train["tmax_anom"].to_numpy(dtype=float), qs)
        cells = self._cells(train).assign(y=train["y"].to_numpy(), prior=prior)
        cols = [*self.keys, "hw", "bin", "band"]
        g = cells.groupby(cols).agg(s=("y", "sum"), n=("y", "size"), prior=("prior", "mean"))
        g["p"] = (g["s"] + self.a * g["prior"]) / (g["n"] + self.a)
        self.table, self.cols = g["p"].reset_index(), cols
        return self

    def predict_prior(self, rows: pd.DataFrame, prior: np.ndarray) -> np.ndarray:
        cells = self._cells(rows)
        p = cells.merge(self.table, on=self.cols, how="left")["p"].to_numpy()
        return np.where(np.isnan(p), prior, p)


class M1Lead:
    name = "M1"

    def __init__(self, cfg: dict):
        self.lookup = _Lookup(cfg, ["lead"], "valid_date")

    def fit(self, train):
        self.lookup.fit_prior(train, train["clim_hw_freq_valid"].to_numpy(dtype=float))
        return self

    def predict(self, rows):
        return self.lookup.predict_prior(rows, rows["clim_hw_freq_valid"].to_numpy(dtype=float))


class M1Window:
    name = "M1"

    def __init__(self, cfg: dict):
        self.m0 = M0Window(cfg)
        self.lookup = _Lookup(cfg, [], "issue_date")

    def fit(self, train):
        self.m0.fit(train)
        self.lookup.fit_prior(train, self.m0.predict(train))
        return self

    def predict(self, rows):
        return self.lookup.predict_prior(rows, self.m0.predict(rows))


# --- M2 -----------------------------------------------------------------------------


class _Logistic:
    def __init__(self, cfg: dict):
        c = cfg["baselines"]["m2"]
        self.features, self.clip = list(c["features"]), tuple(c["clim_clip"])
        self.C, self.max_iter = float(c["C"]), int(c["max_iter"])

    def _x(self, rows: pd.DataFrame, prior: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [rows[self.features].to_numpy(dtype=float), _logit(prior, self.clip)]
        )

    def fit_one(self, rows, prior):
        x, y = self._x(rows, prior), rows["y"].to_numpy(dtype=float)
        ok = ~np.isnan(x).any(axis=1)
        scaler = StandardScaler().fit(x[ok])
        model = LogisticRegression(C=self.C, max_iter=self.max_iter)
        model.fit(scaler.transform(x[ok]), y[ok])
        return scaler, model

    def predict_one(self, fitted, rows, prior):
        scaler, model = fitted
        x = self._x(rows, prior)
        ok = ~np.isnan(x).any(axis=1)
        p = np.asarray(prior, dtype=float).copy()  # rows with missing inputs: climatology
        if ok.any():
            p[ok] = model.predict_proba(scaler.transform(x[ok]))[:, 1]
        return p


class M2Lead:
    """One logistic model per lead, so each feature's weight can fade with lead."""

    name = "M2"

    def __init__(self, cfg: dict):
        self.lr = _Logistic(cfg)

    def fit(self, train):
        self.models = {
            int(h): self.lr.fit_one(g, g["clim_hw_freq_valid"].to_numpy(dtype=float))
            for h, g in train.groupby("lead")
        }
        return self

    def predict(self, rows):
        out = np.full(len(rows), np.nan)
        for h, idx in rows.groupby("lead").indices.items():
            g = rows.iloc[idx]
            out[idx] = self.lr.predict_one(
                self.models[int(h)], g, g["clim_hw_freq_valid"].to_numpy(dtype=float)
            )
        return out

    def coefficients(self) -> pd.DataFrame:
        names = [*self.lr.features, "logit_clim"]
        return pd.DataFrame(
            [
                dict(
                    zip(
                        ["lead", *names, "intercept"],
                        [h, *m.coef_[0], m.intercept_[0]],
                        strict=True,
                    )
                )
                for h, (_, m) in sorted(self.models.items())
            ]
        )


class M2Window:
    name = "M2"

    def __init__(self, cfg: dict):
        self.m0, self.lr = M0Window(cfg), _Logistic(cfg)

    def fit(self, train):
        self.m0.fit(train)
        self.model = self.lr.fit_one(train, self.m0.predict(train))
        return self

    def predict(self, rows):
        return self.lr.predict_one(self.model, rows, self.m0.predict(rows))


def make(name: str, target: str, cfg: dict):
    table = {
        ("M0", "lead"): lambda: M0Lead(),
        ("M0", "window"): lambda: M0Window(cfg),
        ("M1", "lead"): lambda: M1Lead(cfg),
        ("M1", "window"): lambda: M1Window(cfg),
        ("M2", "lead"): lambda: M2Lead(cfg),
        ("M2", "window"): lambda: M2Window(cfg),
    }
    return table[(name, target)]()
