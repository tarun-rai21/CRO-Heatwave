"""Scores and uncertainty (reference.md §3.1, §3.4-3.5).

Brier score BS = mean((p - y)^2); Brier skill score against the climatology
model M0: BSS = 1 - BS_model / BS_M0.

Uncertainty: a block bootstrap over YEARS. Rows within a season and across
districts are strongly dependent, so whole validation years are resampled with
replacement, all districts together. Every model is scored on the same
resampled years (paired), so differences between models get honest intervals.
The bootstrap works on per-year sums of squared errors, so it is fast and exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def year_sums(oof: pd.DataFrame, models: list[str], group: str | None) -> pd.DataFrame:
    """Per (group, year): row count and sum of squared errors per model."""
    keys = ["year"] if group is None else [group, "year"]
    se = {f"se_{m}": (oof[f"p_{m}"] - oof["y"]) ** 2 for m in models}
    frame = oof[keys].assign(n=1, **se)
    return frame.groupby(keys).sum().reset_index()


def bss_table(
    oof: pd.DataFrame,
    models: list[str],
    ref: str,
    group: str | None,
    n_boot: int,
    seed: int,
    ci: float,
    pairs: list[tuple[str, str]] = (),
) -> pd.DataFrame:
    """BSS point estimate and year-block bootstrap interval per group and model,
    plus paired differences BSS(a) - BSS(b) for each (a, b) in pairs."""
    sums = year_sums(oof, sorted({*models, ref, *[x for p in pairs for x in p]}), group)
    years = np.sort(sums["year"].unique())
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(years), size=(n_boot, len(years)), replace=True)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    rows = []
    groups = [(None, sums)] if group is None else list(sums.groupby(group))
    for g, s in groups:
        s = s.set_index("year").reindex(years, fill_value=0)

        def bss(m, idx=None, s=s):
            se_m = s[f"se_{m}"].to_numpy()
            se_r = s[f"se_{ref}"].to_numpy()
            if idx is None:
                return 1 - se_m.sum() / se_r.sum()
            return 1 - se_m[idx].sum(axis=1) / se_r[idx].sum(axis=1)

        base = {"group": g, "rows": int(s["n"].sum()), "years": int((s["n"] > 0).sum())}
        for m in models:
            boot = bss(m, draws)
            rows.append(
                {
                    **base,
                    "model": m,
                    "bss": bss(m),
                    "lo": np.nanquantile(boot, lo_q),
                    "hi": np.nanquantile(boot, hi_q),
                }
            )
        for a, b in pairs:
            diff = bss(a, draws) - bss(b, draws)
            rows.append(
                {
                    **base,
                    "model": f"{a}-{b}",
                    "bss": bss(a) - bss(b),
                    "lo": np.nanquantile(diff, lo_q),
                    "hi": np.nanquantile(diff, hi_q),
                }
            )
    return pd.DataFrame(rows)


def discrimination(oof: pd.DataFrame, models: list[str], group: str | None) -> pd.DataFrame:
    """ROC-AUC and PR-AUC (average precision) per group and model; base rate beside."""
    rows = []
    groups = [(None, oof)] if group is None else list(oof.groupby(group))
    for g, d in groups:
        y = d["y"].to_numpy()
        if y.min() == y.max():
            continue
        for m in models:
            p = d[f"p_{m}"].to_numpy()
            rows.append(
                {
                    "group": g,
                    "model": m,
                    "base_rate": y.mean(),
                    "roc_auc": roc_auc_score(y, p),
                    "pr_auc": average_precision_score(y, p),
                }
            )
    return pd.DataFrame(rows)


def reliability(oof: pd.DataFrame, model: str, n_bins: int) -> pd.DataFrame:
    """Mean forecast vs observed frequency in equal-count bins of the forecast."""
    p, y = oof[f"p_{model}"].to_numpy(), oof["y"].to_numpy()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    t = (
        pd.DataFrame({"bin": b, "p": p, "y": y})
        .groupby("bin")
        .agg(forecast=("p", "mean"), observed=("y", "mean"), rows=("y", "size"))
    )
    return t.reset_index(drop=True)


def brier_decomposition(oof: pd.DataFrame, model: str, n_bins: int) -> dict:
    """Murphy decomposition BS ~ reliability - resolution + uncertainty (binned)."""
    r = reliability(oof, model, n_bins)
    base = oof["y"].mean()
    w = r["rows"] / r["rows"].sum()
    return {
        "reliability": float((w * (r["forecast"] - r["observed"]) ** 2).sum()),
        "resolution": float((w * (r["observed"] - base) ** 2).sum()),
        "uncertainty": float(base * (1 - base)),
        "brier": float(((oof[f"p_{model}"] - oof["y"]) ** 2).mean()),
    }
