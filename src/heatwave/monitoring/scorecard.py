"""The running scorecard (reference.md §5.6): how the forecasts did against what happened.

Event counts are always shown. Within one season a handful of heatwaves can swing these
numbers, so no retraining or threshold change follows from them mid-season. Recent outcomes
rest on provisional data and can still change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_EVENTS = 20  # below this the page says the score is not yet meaningful
TIERS = ["watch", "alert", "warning"]


def scored(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Model forecasts, in season, whose outcome is known. Degraded climatology is excluded."""
    keep = outcomes["observed"].notna() & outcomes["p"].notna() & outcomes["in_season"]
    return outcomes[keep & (outcomes["status"] != "degraded")].copy()


def _bss(d: pd.DataFrame) -> float:
    ref = ((d["p_clim"] - d["observed"]) ** 2).sum()
    return float(1 - ((d["p"] - d["observed"]) ** 2).sum() / ref) if ref > 0 else np.nan


def _summary(d: pd.DataFrame, label: str, name: str) -> dict:
    return {
        name: label,
        "forecasts": len(d),
        "events": int(d["observed"].sum()),
        "mean_forecast": float(d["p"].mean()),
        "observed_rate": float(d["observed"].mean()),
        "skill_vs_climatology": _bss(d),
    }


def tables(outcomes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    o = scored(outcomes)
    if o.empty:
        return {}
    w = o[o["target"] == "window"]
    lead = o[o["target"] == "lead"]
    season = pd.to_datetime(o["issue_date"]).dt.year.max()
    w_now = w[pd.to_datetime(w["issue_date"]).dt.year == season]
    lead_now = lead[pd.to_datetime(lead["issue_date"]).dt.year == season]
    out = {}
    if not w_now.empty:
        out["window"] = pd.DataFrame([_summary(w_now, str(season), "season")])
        rows = []
        for t in TIERS:
            order = TIERS[TIERS.index(t) :]  # a Warning also meets the Alert and Watch thresholds
            f = w_now["tier"].isin(order)
            hits = float(w_now.loc[f, "observed"].sum())
            events = float(w_now["observed"].sum())
            rows.append(
                {
                    "tier_or_higher": t,
                    "issued": int(f.sum()),
                    "came_true": int(hits),
                    "share_true": hits / f.sum() if f.any() else np.nan,
                    "events_caught": hits / events if events else np.nan,
                }
            )
        out["tiers"] = pd.DataFrame(rows)
        recent = w_now[
            pd.to_datetime(w_now["issue_date"])
            >= pd.to_datetime(w_now["issue_date"]).max() - pd.Timedelta(days=6)
        ]
        out["last_7"] = pd.DataFrame([_summary(recent, "last 7 scored issue days", "period")])
    if not lead_now.empty:
        days = (
            pd.to_datetime(lead_now["valid_date"]) - pd.to_datetime(lead_now["issue_date"])
        ).dt.days
        out["by_lead"] = pd.DataFrame(
            [_summary(g, int(h), "days_ahead") for h, g in lead_now.groupby(days)]
        )
    return out


def _html(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, float):
            return "–" if np.isnan(v) else f"{v:.3f}"
        return str(v)

    head = "".join(
        f'<th class="l">{str(c).replace("_", " ").capitalize()}</th>'
        if i == 0
        else f"<th>{str(c).replace('_', ' ').capitalize()}</th>"
        for i, c in enumerate(df.columns)
    )
    rows = "".join(
        "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in r) + "</tr>"
        for r in df.itertuples(index=False)
    )
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
    return f'<div class="scroll">{table}</div>'


def body_html(outcomes: pd.DataFrame | None) -> str:
    intro = (
        "<h1>Scorecard</h1><p>How forecasts issued this season compare with what happened. Only "
        "in-season model forecasts whose outcome is known are counted. Skill above 0 beats the "
        "usual chance for the date. Recent outcomes rest on provisional data and can change, and "
        "one season has few independent heatwaves, so these numbers can swing a lot.</p>"
    )
    t = tables(outcomes) if outcomes is not None and not outcomes.empty else {}
    if not t:
        return intro + '<div class="status">No forecasts have been scored yet this season.</div>'
    html = [intro]
    w = t.get("window")
    if w is not None and w["events"].iloc[0] < MIN_EVENTS:
        html.append(
            f'<div class="status">Only {int(w["events"].iloc[0])} events so far: too few for these '
            "scores to mean much.</div>"
        )
    for key, title in (
        ("window", "Next-10-days forecast"),
        ("last_7", "Most recent week"),
        ("tiers", "Tiers"),
        ("by_lead", "Single-day forecasts, by days ahead"),
    ):
        if key in t:
            html.append(f"<h2>{title}</h2>{_html(t[key])}")
    return "".join(html)
