"""M3: LightGBM (reference.md §4.4-4.6).

Per-lead target: one model for all leads (lead is a feature), on the as-of
features plus the climatological frequency at the valid date. Window target: the
as-of features plus the window climatology (M0Window, fitted on the training rows).

Boosting starts from climatology: the logit of the climatological frequency is
the base margin, so every tree learns how the forecast should DIFFER from
climatology, and a hotter or cooler inner block cannot be fitted by moving the
base rate alone.

Early stopping uses the fold's INNER block: the last `inner_seasons` training
seasons are held out from the first fit to find the number of rounds; the model is
then refitted on all training seasons with that number. The fold's validation
years are never seen. No class weighting or resampling (probabilities are the
product).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from heatwave.features.build import ASOF_FEATURES
from heatwave.models.baselines import M0Window


def default_features(target: str) -> list[str]:
    extra = ["lead", "clim_hw_freq_valid"] if target == "lead" else ["clim_window"]
    return [*ASOF_FEATURES, *extra]


class M3:
    name = "M3"

    def __init__(
        self,
        cfg: dict,
        target: str,
        inner_seasons: int,
        features: list[str] | None = None,
        seed: int | None = None,
        params: dict | None = None,
        use_monotone: bool | None = None,
    ):
        m = cfg["m3"]
        self.cfg, self.target, self.inner = cfg, target, inner_seasons
        self.features = list(features or default_features(target))
        self.params = {**m["params"], **(params or {})}
        self.params["seed"] = m["seeds"][0] if seed is None else seed
        self.max_rounds, self.patience = m["max_rounds"], m["early_stopping_rounds"]
        mono = m["use_monotone"] if use_monotone is None else use_monotone
        if mono:
            inc, dec = set(m["monotone"]["increasing"]), set(m["monotone"]["decreasing"])
            self.params["monotone_constraints"] = [
                1 if f in inc else -1 if f in dec else 0 for f in self.features
            ]
            self.params["monotone_constraints_method"] = "advanced"

    def _year(self, rows):
        return (rows["valid_date"] if self.target == "lead" else rows["issue_date"]).dt.year

    def _x(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Feature matrix (float32, only the model's columns) and base margin
        (logit of climatology). Builds no copy of the full table."""
        clim = (
            self.m0.predict(rows)
            if self.target == "window"
            else rows["clim_hw_freq_valid"].to_numpy(dtype=float)
        )
        cols = []
        for f in self.features:
            cols.append(clim if f == "clim_window" else rows[f].to_numpy(dtype=np.float32))
        p = np.clip(clim, 1e-4, 1 - 1e-4)
        return np.column_stack(cols).astype(np.float32, copy=False), np.log(p / (1 - p))

    def fit(self, train: pd.DataFrame):
        if self.target == "window":
            self.m0 = M0Window(self.cfg).fit(train)
        years = self._year(train)
        last = sorted(years.unique())[-self.inner :]
        inner = np.flatnonzero(years.isin(last).to_numpy())
        outer = np.flatnonzero(~years.isin(last).to_numpy())
        x, margin = self._x(train)
        y = train["y"].to_numpy(dtype=float)

        # Bin once; the early-stopping split uses subsets of the same binned data.
        full = lgb.Dataset(
            x,
            y,
            init_score=margin,
            feature_name=self.features,
            params={"max_bin": self.params["max_bin"], "verbose": -1},
        ).construct()
        del x
        fit_part = full.subset(outer)
        fit_part.set_init_score(margin[outer])
        inner_part = full.subset(inner)
        inner_part.set_init_score(margin[inner])
        first = lgb.train(
            self.params,
            fit_part,
            num_boost_round=self.max_rounds,
            valid_sets=[inner_part],
            callbacks=[lgb.early_stopping(self.patience, verbose=False)],
        )
        self.best_rounds = max(1, int(first.best_iteration))
        self.inner_logloss = float(first.best_score["valid_0"]["binary_logloss"])
        del first, fit_part, inner_part
        self.booster = lgb.train(self.params, full, num_boost_round=self.best_rounds)
        del full
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        x, margin = self._x(rows)
        raw = self.booster.predict(x, raw_score=True, num_threads=self.params["num_threads"])
        return 1 / (1 + np.exp(-(raw + margin)))

    def importance(self) -> pd.DataFrame:
        gain = self.booster.feature_importance("gain")
        return (
            pd.DataFrame({"feature": self.features, "gain": gain / gain.sum()})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
