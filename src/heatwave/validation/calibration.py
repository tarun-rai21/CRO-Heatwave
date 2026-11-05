"""Calibration and alert-tier thresholds (reference.md §3.6, §4.8).

Calibrators are fitted only on out-of-fold predictions. To score them, each fold's
rows are calibrated by a calibrator fitted on the OTHER folds (`cross_fit`). The
production calibrator is then fitted on all OOF rows.

Tier thresholds follow the rules fixed in configs/problem.yaml. A rule may have no
valid threshold (e.g. no long-lead forecast verifies 50 % of the time): that tier is
then not issued at that lead, shown as a missing threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _design(p: np.ndarray, method: str) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    if method == "platt":
        return _logit(p).reshape(-1, 1)
    if method == "beta":
        return np.column_stack([np.log(p), -np.log(1 - p)])
    raise ValueError(f"unknown calibration method {method}")


def fit(p: np.ndarray, y: np.ndarray, method: str = "platt") -> LogisticRegression:
    """Effectively unpenalised logistic regression on the transformed forecast."""
    return LogisticRegression(C=1e10, max_iter=1000).fit(_design(p, method), y)


def apply(model: LogisticRegression, p: np.ndarray, method: str = "platt") -> np.ndarray:
    return model.predict_proba(_design(p, method))[:, 1]


def cross_fit(oof: pd.DataFrame, col: str, by: str | None, method: str = "platt") -> np.ndarray:
    """Calibrated `col`: each fold's rows use a calibrator fitted on the other folds,
    one calibrator per value of `by` (e.g. lead)."""
    out = np.full(len(oof), np.nan)
    groups = oof.groupby(by).indices if by else {None: np.arange(len(oof))}
    fold = oof["fold"].to_numpy()
    p, y = oof[col].to_numpy(dtype=float), oof["y"].to_numpy(dtype=float)
    for idx in groups.values():
        for f in np.unique(fold[idx]):
            held = idx[fold[idx] == f]
            rest = idx[fold[idx] != f]
            out[held] = apply(fit(p[rest], y[rest], method), p[held], method)
    return out


def _curve(p: np.ndarray, y: np.ndarray, year: np.ndarray) -> pd.DataFrame:
    """Scores of flagging p >= threshold, one row per distinct forecast value."""
    order = np.argsort(-p, kind="stable")
    p, y, year = p[order], y[order], year[order]
    first = ~pd.Series(year).duplicated().to_numpy()
    d = pd.DataFrame(
        {
            "threshold": p,
            "flagged": np.arange(1, len(p) + 1),
            "tp": np.cumsum(y),
            "seasons": np.cumsum(first),
        }
    )
    d = d[np.r_[p[1:] != p[:-1], True]].reset_index(drop=True)
    d["fp"] = d["flagged"] - d["tp"]
    d["pod"] = d["tp"] / max(y.sum(), 1)
    d["far"] = d["fp"] / d["flagged"]
    return d


def min_cost_threshold(curve: pd.DataFrame, positives: float, ratio: float) -> float:
    """Threshold minimising ratio x misses + false alarms; NaN if flagging nothing wins."""
    cost = ratio * (positives - curve["tp"]) + curve["fp"]
    i = cost.idxmin()
    return float(curve.loc[i, "threshold"]) if cost[i] < ratio * positives else np.nan


def thresholds(
    p: np.ndarray, y: np.ndarray, year: np.ndarray, rules: dict, min_flagged: int, min_seasons: int
) -> dict[str, float]:
    """Watch, alert and warning thresholds under the rules of reference.md §3.6.
    A warning needs enough flagged rows and seasons to mean anything."""
    c = _curve(p, y, year)
    watch = c[c["pod"] >= rules["watch"]["value"]]
    ok = c[
        (c["far"] <= rules["warning"]["value"])
        & (c["flagged"] >= min_flagged)
        & (c["seasons"] >= min_seasons)
    ]
    return {
        "watch": float(watch["threshold"].iloc[0]) if len(watch) else np.nan,
        "alert": min_cost_threshold(c, y.sum(), rules["alert"]["miss_to_false_alarm"]),
        "warning": float(ok["threshold"].iloc[-1]) if len(ok) else np.nan,
    }


def scores(p: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    """POD, FAR and the share of rows flagged at a threshold (NaN: tier not issued)."""
    if np.isnan(threshold):
        return {"pod": 0.0, "far": np.nan, "share_flagged": 0.0}
    f = p >= threshold
    tp = float(y[f].sum())
    return {
        "pod": tp / max(float(y.sum()), 1.0),
        "far": 1 - tp / f.sum() if f.any() else np.nan,
        "share_flagged": float(f.mean()),
    }


def tier_table(
    oof: pd.DataFrame, col: str, by: str | None, rules: dict, min_flagged: int, min_seasons: int
) -> pd.DataFrame:
    """Per group and tier: the threshold from all rows, then POD and FAR with thresholds
    fitted on the other folds (`held_*`), a fairer view than the same rows' scores."""
    rows = []
    groups = oof.groupby(by) if by else [(None, oof)]
    for key, g in groups:
        p, y = g[col].to_numpy(dtype=float), g["y"].to_numpy(dtype=float)
        yr, fold = g["year"].to_numpy(), g["fold"].to_numpy()
        full = thresholds(p, y, yr, rules, min_flagged, min_seasons)
        held = {tier: np.zeros(len(g), bool) for tier in full}
        for f in np.unique(fold):
            te = fold == f
            t = thresholds(p[~te], y[~te], yr[~te], rules, min_flagged, min_seasons)
            for tier, v in t.items():
                if not np.isnan(v):
                    held[tier][te] = p[te] >= v
        for tier, v in full.items():
            hit = held[tier]
            rows.append(
                {
                    "group": key,
                    "tier": tier,
                    "threshold": v,
                    **scores(p, y, v),
                    "held_pod": y[hit].sum() / max(y.sum(), 1),
                    "held_far": 1 - y[hit].sum() / hit.sum() if hit.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def cost_curve(oof: pd.DataFrame, col: str, by: str | None, ratios: list[float]) -> pd.DataFrame:
    """The alert threshold and its scores over a range of miss : false-alarm ratios."""
    rows = []
    groups = oof.groupby(by) if by else [(None, oof)]
    for key, g in groups:
        p, y = g[col].to_numpy(dtype=float), g["y"].to_numpy(dtype=float)
        c = _curve(p, y, g["year"].to_numpy())
        for r in ratios:
            t = min_cost_threshold(c, y.sum(), r)
            rows.append({"group": key, "ratio": r, "threshold": t, **scores(p, y, t)})
    return pd.DataFrame(rows)
