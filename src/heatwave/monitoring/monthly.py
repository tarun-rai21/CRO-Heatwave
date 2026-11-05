"""Monthly checks (reference.md §5.6): provisional against final data, and live label drift.

Skew: for each value POWER has revised, the first version we stored against the latest one.
Live forecasts run on first versions; the model was trained on final ones.

Label drift: label Tmax against reported station maxima this season, compared with the same
months of the period the drift term was last fitted on. IMD's yearly grid is not used here,
because it arrives only after the year ends: `heatwave drift-check` covers that.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from heatwave.ingest.raw_store import RawStore

SKEW_VARIABLES = ["T2M_MAX", "T2M_MIN", "GWETROOT", "GWETTOP"]
DRIFT_LIMIT_C = 0.75  # a change this large against stations is reported
MIN_STATION_DAYS = 30
# Mean first-to-latest change that matters: skill was hit by a shift of 0.02 (decision record 013).
SOIL_GAP_LIMIT = 0.02
SOIL_VARIABLES = ["GWETROOT"]


def skew(store: RawStore, years: list[int], variables=SKEW_VARIABLES) -> pd.DataFrame:
    """Per variable: how many stored values were revised, and latest minus first."""
    d = store.read(years)
    d = d[d["variable"].isin(variables)].sort_values("ingested_at")
    g = d.groupby(["cell_id", "obs_date", "variable"])["value"]
    first, last, n = g.first(), g.last(), g.count()
    t = pd.DataFrame({"first": first, "last": last, "versions": n}).reset_index()
    rows = []
    for v, part in t.groupby("variable"):
        rev = part[part["versions"] > 1]
        diff = rev["last"] - rev["first"]
        rows.append(
            {
                "variable": v,
                "values": len(part),
                "revised": len(rev),
                "mean_change": float(diff.mean()) if len(rev) else None,
                "sd_change": float(diff.std()) if len(rev) > 1 else None,
                "p05": float(diff.quantile(0.05)) if len(rev) else None,
                "p95": float(diff.quantile(0.95)) if len(rev) else None,
            }
        )
    return pd.DataFrame(rows)


def skew_flags(table: pd.DataFrame) -> list[dict]:
    """Soil moisture variables whose revisions exceed the limit."""
    out = []
    for r in table[table["variable"].isin(SOIL_VARIABLES)].itertuples():
        if r.mean_change is not None and abs(r.mean_change) > SOIL_GAP_LIMIT:
            detail = (
                f"provisional {r.variable} differs from final by {r.mean_change:+.3f} on average "
                f"({r.revised} revised values), limit {SOIL_GAP_LIMIT}. See decision record 013."
            )
            out.append({"check": "provisional_soil_moisture", "detail": detail})
    return out


def label_against_stations(
    days: pd.DataFrame,
    where: pd.DataFrame,
    labels: pd.DataFrame,
    season_year: int,
    reference: tuple[int, int],
    months: list[int],
) -> dict:
    """Station-by-station change in label minus station, this season against `reference`, on
    the same calendar months. Stations need MIN_STATION_DAYS reports in both."""
    d = days.dropna(subset=["tmax_c"]).merge(where, on="station_id")
    d = d.merge(labels[["district_id", "obs_date", "tmax_used"]], on=["district_id", "obs_date"])
    d = d[d["obs_date"].dt.month.isin(months) & d["tmax_c"].between(10, 55)].copy()
    d["gap"] = d["tmax_used"] - d["tmax_c"]
    d["year"] = d["obs_date"].dt.year
    now = d[d["year"] == season_year]
    if now.empty:
        return {
            "level": "skipped",
            "detail": f"no season data for {season_year} yet",
            "stations": 0,
        }
    seen = sorted(now["obs_date"].dt.month.unique())
    ref = d[d["year"].between(*reference) & d["obs_date"].dt.month.isin(seen)]
    a, b = (
        now.groupby("name")["gap"].agg(["mean", "size"]),
        ref.groupby("name")["gap"].agg(["mean", "size"]),
    )
    keep = a.index[(a["size"] >= MIN_STATION_DAYS)].intersection(
        b.index[b["size"] >= MIN_STATION_DAYS]
    )
    if not len(keep):
        return {"level": "skipped", "detail": "too few station reports yet", "stations": 0}
    change = a.loc[keep, "mean"] - b.loc[keep, "mean"]
    mean = float(change.mean())
    return {
        "level": "warn" if abs(mean) > DRIFT_LIMIT_C else "ok",
        "detail": f"label minus stations changed {mean:+.2f} C from {reference[0]}-{reference[1]}, "
        f"{len(keep)} stations, months {seen}, limit {DRIFT_LIMIT_C}",
        "stations": len(keep),
        "mean_change": mean,
        "by_station": {k: float(v) for k, v in change.items()},
    }


def read_days(root: Path, g: dict, live_dir: Path | None):
    from heatwave.ingest import gsod
    from heatwave.labels.drift_check import load_days

    days = load_days(root, g)
    if live_dir is not None and live_dir.exists() and list(live_dir.glob("*/*.csv")):
        days = pd.concat([days, gsod.read_station_days(live_dir, g)], ignore_index=True)
        days = days.drop_duplicates(["station_id", "obs_date"], keep="last")
    return days
