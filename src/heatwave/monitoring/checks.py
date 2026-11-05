"""Daily monitoring checks (reference.md §5.6). Limits come from development years only and
are frozen with the model in configs/frozen/<version>_monitor.json, so a check cannot quietly
adapt to what it is watching.

Levels: ok; skipped (nothing to compare, e.g. out of season); warn (worth a look, reported
only); fail (emailed to the owner).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from heatwave.models.frozen import FrozenModel

ANOMALY_FEATURES = ["tmax_anom", "tmax_anom_mean7", "gwetroot_anom"]
SANITY_MARGIN = 0.05  # on a state-mean probability or a tier share
FEATURE_Z = 3.0  # a 30-day state mean this many year-to-year SDs from its month's history
FEATURE_DAYS = 30
WINDOW_BELOW_LEAD = 0.15  # development worst case was 0.115 (decision record 009)
BASELINE_YEARS = (2001, 2020)  # the training years of m2_v1


def write_baseline(
    spec: dict, window_oof: pd.DataFrame, features: pd.DataFrame, path: Path
) -> dict:
    """Per month: the range of the daily state-mean window forecast and of the Warning share
    in the out-of-fold years, and the year-to-year spread of each anomaly feature."""
    w = window_oof.copy()
    w["p"] = FrozenModel._calibrate(spec["window"]["calibrator"], w["p_M2S"].to_numpy(dtype=float))
    w["warning"] = w["p"] >= spec["tiers"]["warning"]["threshold"]
    daily = w.groupby(["issue_date", "month"]).agg(
        mean_p=("p", "mean"), warning=("warning", "mean")
    )
    sanity = daily.groupby("month").agg(
        mean_p_min=("mean_p", "min"), mean_p_max=("mean_p", "max"), warning_max=("warning", "max")
    )
    f = features[features["asof_date"].dt.year.between(*BASELINE_YEARS)]
    by_year = f.groupby([f["asof_date"].dt.year, f["asof_date"].dt.month])[ANOMALY_FEATURES].mean()
    by_year.index.names = ["year", "month"]
    spread = by_year.groupby("month").agg(["mean", "std"])
    out = {
        "version": spec["version"],
        "basis": f"window OOF 2006-2020; features {BASELINE_YEARS[0]}-{BASELINE_YEARS[1]}",
        "sanity": {str(m): r.to_dict() for m, r in sanity.iterrows()},
        "features": {
            str(m): {
                k: {
                    "mean": float(spread.loc[m, (k, "mean")]),
                    "std": float(spread.loc[m, (k, "std")]),
                }
                for k in ANOMALY_FEATURES
            }
            for m in spread.index
        },
    }
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _check(name: str, level: str, detail: str) -> dict:
    return {"check": name, "level": level, "detail": detail}


def forecast_checks(rows: pd.DataFrame, baseline: dict, n_districts: int = 75) -> list[dict]:
    out = []
    status = rows["status"].iloc[0]
    reason = rows["reason"].iloc[0]
    if status == "ok":
        out.append(
            _check("data_timeliness", "ok", f"data age {rows['data_age_days'].iloc[0]} days")
        )
    elif status == "degraded":
        out.append(_check("data_timeliness", "fail", f"degraded: {reason}"))
    else:
        out.append(_check("data_timeliness", "warn", str(reason)))

    window = rows[rows["target"] == "window"]
    lead = rows[rows["target"] == "lead"]
    if window["district_id"].nunique() != n_districts or len(window) != n_districts:
        out.append(_check("coverage", "fail", f"{len(window)} window rows, expected {n_districts}"))
    else:
        out.append(_check("coverage", "ok", f"{n_districts} districts"))
    p = rows["p"].dropna()
    if not p.between(0, 1).all():
        out.append(_check("probabilities_valid", "fail", "probability outside 0-1"))
    in_season = bool(window["in_season"].iloc[0])
    if in_season and window["p"].isna().any():
        out.append(_check("probabilities_present", "fail", "in season with blank probabilities"))
    if status == "degraded" or not in_season:
        out.append(_check("forecast_sanity", "skipped", "no model forecast in season today"))
        return out

    month = str(pd.Timestamp(rows["issue_date"].iloc[0]).month)
    base = baseline["sanity"].get(month)
    if base is None:
        out.append(_check("forecast_sanity", "skipped", f"no history for month {month}"))
    else:
        mean_p, warning = float(window["p"].mean()), float((window["tier"] == "warning").mean())
        low, high = base["mean_p_min"] - SANITY_MARGIN, base["mean_p_max"] + SANITY_MARGIN
        ok = low <= mean_p <= high
        out.append(
            _check(
                "mean_window_probability",
                "ok" if ok else "warn",
                f"{mean_p:.3f}, history {base['mean_p_min']:.3f} to {base['mean_p_max']:.3f}",
            )
        )
        ok = warning <= base["warning_max"] + SANITY_MARGIN
        out.append(
            _check(
                "warning_share",
                "ok" if ok else "warn",
                f"{warning:.2f}, history up to {base['warning_max']:.2f}",
            )
        )
    top = lead.groupby("district_id")["p"].max()
    gap = float((top - window.set_index("district_id")["p"]).max())
    out.append(
        _check(
            "window_against_single_days",
            "ok" if gap <= WINDOW_BELOW_LEAD else "warn",
            f"largest shortfall {max(gap, 0):.3f}, limit {WINDOW_BELOW_LEAD}",
        )
    )
    return out


def feature_checks(features: pd.DataFrame, asof_date, baseline: dict) -> list[dict]:
    """Anomaly features only: raw temperatures change every season by design."""
    if asof_date is None:
        return [_check("feature_drift", "skipped", "no model forecast today")]
    t = pd.Timestamp(asof_date)
    base = baseline["features"].get(str(t.month))
    if base is None:
        return [_check("feature_drift", "skipped", f"no training history for month {t.month}")]
    recent = features[features["asof_date"] > t - pd.Timedelta(days=FEATURE_DAYS)]
    recent = recent[recent["asof_date"] <= t]
    out = []
    for k in ANOMALY_FEATURES:
        z = (float(recent[k].mean()) - base[k]["mean"]) / base[k]["std"]
        level = "ok" if abs(z) <= FEATURE_Z else "warn"
        out.append(_check(f"feature_drift_{k}", level, f"z = {z:+.1f}, limit {FEATURE_Z}"))
    return out


def log_row(rows: pd.DataFrame, checks: list[dict], started_at) -> dict:
    levels = [c["level"] for c in checks]
    return {
        "issue_date": pd.Timestamp(rows["issue_date"].iloc[0]),
        "checked_at": pd.Timestamp(started_at),
        "status": rows["status"].iloc[0],
        "data_age_days": int(rows["data_age_days"].iloc[0]),
        "fails": levels.count("fail"),
        "warns": levels.count("warn"),
        "checks": json.dumps(checks),
    }


def append_log(path: Path, row: dict) -> pd.DataFrame:
    """One row per issue date; a re-run replaces it. `first_checked_at` keeps the first."""
    new = pd.DataFrame([row])
    new["first_checked_at"] = new["checked_at"]
    if path.exists():
        old = pd.read_parquet(path)
        same = old[old["issue_date"] == row["issue_date"]]
        if not same.empty:
            new["first_checked_at"] = same["first_checked_at"].iloc[0]
        new = pd.concat([old[old["issue_date"] != row["issue_date"]], new], ignore_index=True)
    new = new.sort_values("issue_date").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return new


def problems(checks: list[dict]) -> list[dict]:
    return [c for c in checks if c["level"] == "fail"]


def nan_safe(value):
    return None if isinstance(value, float) and np.isnan(value) else value
