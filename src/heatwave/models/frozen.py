"""The frozen production model (reference.md §3.6, §4.8, §6.1).

Plain numbers in one JSON file: for the per-lead target and the window target, the
logistic coefficients, scaling and Platt calibrator, plus the window's tier
thresholds. The window climatology (district x day of year) sits next to it as a
Parquet file, checked against a hash. Nothing here depends on a pickled library
object, so a library upgrade cannot silently change a forecast.

Rows need the features named in the file; per-lead rows also need `lead` and
`clim_hw_freq_valid`, window rows `district_id` and `issue_date`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from heatwave.models.baselines import M0Window, _Logistic

CAL_EPS = 1e-6  # as validation.calibration
TIERS = ["watch", "alert", "warning"]  # lowest to highest


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logistic_to_dict(fitted) -> dict:
    scaler, model = fitted
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }


def calibrator_to_dict(model) -> dict:
    return {"slope": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])}


class FrozenModel:
    def __init__(self, spec: dict, m0_table: pd.DataFrame):
        self.spec = spec
        self.features = list(spec["features"])
        self.clip = tuple(spec["clim_clip"])
        self.m0 = M0Window({"baselines": {"m0_window": {"smooth_days": spec["m0_smooth_days"]}}})
        self.m0.table, self.m0.overall = m0_table, float(spec["m0_overall"])

    @classmethod
    def load(cls, path: Path, table_dir: Path | None = None) -> FrozenModel:
        """Read <path>; the climatology table named in it (in `table_dir`, default next
        to it) must match its recorded hash."""
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.with_table_dir(spec, Path(table_dir) if table_dir else Path(path).parent)

    @classmethod
    def with_table_dir(cls, spec: dict, table_dir: Path) -> FrozenModel:
        table = Path(table_dir) / spec["m0_table"]
        if sha256(table) != spec["m0_table_sha256"]:
            raise ValueError(f"{table} does not match the hash frozen with the model")
        return cls(spec, pd.read_parquet(table))

    def _raw(self, part: dict, rows: pd.DataFrame, prior: np.ndarray) -> np.ndarray:
        """Logistic forecast; rows with a missing input get climatology, as in evaluation."""
        p = np.clip(np.asarray(prior, dtype=float), *self.clip)
        x = np.column_stack([rows[self.features].to_numpy(dtype=float), np.log(p / (1 - p))])
        out = np.asarray(prior, dtype=float).copy()
        ok = ~np.isnan(x).any(axis=1)
        z = (x[ok] - np.array(part["mean"])) / np.array(part["scale"])
        out[ok] = _sigmoid(part["intercept"] + z @ np.array(part["coef"]))
        return out

    @staticmethod
    def _calibrate(cal: dict, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, CAL_EPS, 1 - CAL_EPS)
        return _sigmoid(cal["intercept"] + cal["slope"] * np.log(p / (1 - p)))

    def predict_lead(self, rows: pd.DataFrame) -> np.ndarray:
        out = np.full(len(rows), np.nan)
        for h, idx in rows.groupby("lead").indices.items():
            g = rows.iloc[idx]
            part = self.spec["lead"][str(int(h))]
            raw = self._raw(part, g, g["clim_hw_freq_valid"].to_numpy(dtype=float))
            out[idx] = self._calibrate(part["calibrator"], raw)
        return out

    def predict_window(self, rows: pd.DataFrame) -> np.ndarray:
        part = self.spec["window"]
        return self._calibrate(part["calibrator"], self._raw(part, rows, self.m0.predict(rows)))

    def tier(self, window_p: np.ndarray) -> np.ndarray:
        """The highest tier each window probability reaches ('none' below Watch)."""
        p = np.asarray(window_p, dtype=float)
        out = np.full(len(p), "none", dtype=object)
        for name in TIERS:
            out[p >= self.spec["tiers"][name]["threshold"]] = name
        return out


def check_matches(fitted: _Logistic, model_fit, part: dict, rows, prior, frozen: FrozenModel):
    """The JSON numbers must reproduce the fitted scikit-learn model exactly."""
    want = fitted.predict_one(model_fit, rows, prior)
    got = frozen._raw(part, rows, prior)
    if not np.allclose(want, got, rtol=0, atol=1e-10):
        raise ValueError("frozen coefficients do not reproduce the fitted model")
