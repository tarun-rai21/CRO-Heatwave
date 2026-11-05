"""The public static site and its CSV/JSON downloads (reference.md §5.2, §5.4).

Only forecast probabilities and tiers are published: no IMD values and no raw weather data
(docs/decisions/012). Every page says this is not an official forecast.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

TIER_ORDER = {"warning": 0, "alert": 1, "watch": 2, "none": 3}
TIER_TEXT = {"warning": "Warning", "alert": "Alert", "watch": "Watch", "none": ""}

NOTICE = (
    "Experimental research forecast. <strong>This is not an official forecast or warning.</strong> "
    "Official heatwave warnings for India come from the "
    '<a href="https://mausam.imd.gov.in/">India Meteorological Department</a>. '
    "Do not base safety decisions on this page."
)

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#5f6368;--line:#dcdfe3;--panel:#f5f6f7;--link:#0b57d0;
--watch:#fff3c4;--alert:#ffd6a5;--warning:#f8b4b4;--notice:#fff4e5;--notice-line:#c77700}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--fg:#e8eaed;--muted:#a0a4a9;--line:#34373b;
--panel:#1f2226;--link:#8ab4f8;--watch:#5a4a10;--alert:#70431a;--warning:#7a2323;--notice:#33270f;
--notice-line:#c77700}}
*{box-sizing:border-box}
body{margin:0 auto;max-width:1100px;padding:16px;background:var(--bg);color:var(--fg);
font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--link)}h1{font-size:1.5rem;margin:.25rem 0}h2{font-size:1.15rem;margin-top:1.75rem}
.notice{background:var(--notice);border-left:4px solid var(--notice-line);padding:10px 12px;margin:12px 0}
.status{background:var(--panel);padding:10px 12px;margin:12px 0}.muted{color:var(--muted)}
nav a{margin-right:14px}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child,th.l,td.l{text-align:left}th{font-weight:600}
.t-watch{background:var(--watch)}.t-alert{background:var(--alert)}.t-warning{background:var(--warning)}
"""


def _page(title: str, body: str, nav: bool = True) -> str:
    links = (
        '<nav><a href="index.html">Forecast</a><a href="scorecard.html">Scorecard</a><a href="about.html">About</a></nav>'
        if nav
        else ""
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{CSS}</style></head><body>"
        f'{links}<div class="notice">{NOTICE}</div>{body}</body></html>'
    )


def _pct(p) -> str:
    return "–" if pd.isna(p) else f"{100 * float(p):.0f}%"


def status_line(rows: pd.DataFrame) -> str:
    """What today's forecast rests on, in words."""
    r = rows.iloc[0]
    status = r["status"]
    window = rows[rows["target"] == "window"]
    if status == "degraded":
        return (
            f"<strong>Degraded.</strong> No model forecast today: {escape(str(r['reason']))}. "
            "Figures are climatology only (the usual chance for the date). Tiers are not issued."
        )
    text = f"Based on observations through {pd.Timestamp(r['asof_date']).date()}."
    if status != "ok":
        text = f"<strong>{escape(str(r['reason']).capitalize())}.</strong> {text} Fewer days are forecast."
    if window["p"].isna().all():
        text += (
            " <strong>Out of season:</strong> the model has no basis at this time of year. "
            "Probabilities resume in mid-February."
        )
    elif not window["in_season"].any():
        text += " Tiers are issued only when the 10-day window lies within March to July."
    return text


def _table(rows: pd.DataFrame, names: dict, regions: dict) -> str:
    window = rows[rows["target"] == "window"].set_index("district_id")
    lead = rows[rows["target"] == "lead"]
    dates = sorted(lead["valid_date"].unique())
    cols = lead.pivot(index="district_id", columns="valid_date", values="p")
    order = sorted(
        window.index,
        key=lambda d: (
            TIER_ORDER[window.loc[d, "tier"]],
            -(0 if pd.isna(window.loc[d, "p"]) else window.loc[d, "p"]),
            names.get(d, d),
        ),
    )
    head = "".join(f"<th>{pd.Timestamp(d):%d %b}</th>" for d in dates)
    body = []
    for d in order:
        w = window.loc[d]
        tier = w["tier"]
        cls = f' class="l t-{tier}"' if tier != "none" else ' class="l"'
        cells = "".join(f"<td>{_pct(cols.loc[d, x])}</td>" for x in dates)
        body.append(
            f"<tr><td>{escape(names.get(d, d))}</td>"
            f'<td class="l">{escape(regions.get(d, "").capitalize())}</td>'
            f"<td{cls}>{TIER_TEXT[tier]}</td><td>{_pct(w['p'])}</td>{cells}</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>District</th><th class="l">Region</th>'
        f'<th class="l">Tier</th><th>Next 10 days</th>{head}</tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table></div>"
    )


def index_html(rows: pd.DataFrame, names: dict, regions: dict) -> str:
    issue = pd.Timestamp(rows["issue_date"].iloc[0]).date()
    window = rows[rows["target"] == "window"]
    counts = window["tier"].value_counts()
    tiers = ", ".join(f"{TIER_TEXT[t]}: {counts.get(t, 0)}" for t in ("warning", "alert", "watch"))
    first, last = window["valid_date"].iloc[0], window["window_end"].iloc[0]
    return _page(
        "UP heatwave forecast",
        f"<h1>Uttar Pradesh heatwave forecast</h1><p class='muted'>Issued {issue}</p>"
        f'<div class="status">{status_line(rows)}</div>'
        f"<p>Districts by tier: {tiers}.</p>"
        f"<p><strong>Next 10 days</strong> is the chance of at least one heatwave day between "
        f"{pd.Timestamp(first):%d %b} and {pd.Timestamp(last):%d %b}. Dated columns are the chance "
        "of a heatwave day on that date. A dash means no forecast.</p>"
        + _table(rows, names, regions)
        + "<h2>Download</h2><p>"
        '<a href="data/latest.csv">latest.csv</a> · <a href="data/latest.json">latest.json</a> · '
        'past days in <a href="data/index.json">data/index.json</a></p>',
    )


def about_html(spec: dict, repo_url: str) -> str:
    t = spec["tiers"]
    pct = lambda k: f"{100 * t[k]['threshold']:.1f}%"  # noqa: E731
    return _page(
        "About this forecast",
        "<h1>About this forecast</h1>"
        "<p>A daily statistical estimate of the chance of a heatwave day in each of Uttar Pradesh's "
        "75 districts, from recent weather only. It uses no weather-model forecasts, so it knows "
        "what has happened, not what the atmosphere will do next.</p>"
        "<h2>Tiers</h2><p>Issued on the 10-day chance, for windows within March to July:</p>"
        f"<ul><li><strong>Watch:</strong> {pct('watch')} or more</li>"
        f"<li><strong>Alert:</strong> {pct('alert')} or more</li>"
        f"<li><strong>Warning:</strong> {pct('warning')} or more</li></ul>"
        "<p>These are this project's model tiers. They are not IMD warning categories.</p>"
        "<h2>How good is it?</h2>"
        "<p><strong>This version has not had a clean test yet.</strong> An earlier version, which also "
        "used soil moisture, was scored once on 2021–2025, data it had not been built on. It beat the "
        "usual chance for the date, modestly, and its Alerts came true about 35% of the time.</p>"
        "<p>Soil moisture was then removed, because the recent values available each day differ too "
        "much from the final values the model learned from. Temperature corrections were also updated. "
        "Scoring this version on those same years is only a rough guide, because they had already been "
        "looked at:</p><ul>"
        "<li>Alerts came true about 46% of the time overall: more than half the time in big heatwave "
        "years (2022, 2024), and about 1 in 4 or fewer in quiet years (2021, 2023, 2025).</li>"
        "<li>On earlier years, used to build the model, it clearly beat the usual chance only at short "
        "range. Treat single-day figures beyond about 5 days as little better than the usual chance.</li>"
        "<li>It cannot foresee a quiet season, so it over-forecasts in those years.</li>"
        "<li>These scores used final data. Live forecasts use provisional data and may do worse.</li></ul>"
        '<p>The <a href="scorecard.html">scorecard</a> shows how live forecasts are doing.</p>'
        "<h2>Data and credits</h2><ul>"
        "<li>Weather inputs: NASA POWER. These data were obtained from the NASA Langley Research "
        "Center POWER Project, funded through the NASA Earth Science Directorate Applied Science "
        "Program.</li>"
        "<li>The temperature correction was calibrated against the India Meteorological Department "
        "1° gridded temperature data: Srivastava, Rajeevan and Kshirsagar (2009), <em>Atmospheric "
        "Science Letters</em>, doi:10.1002/asl.232. No IMD data is published here, and IMD has not "
        "reviewed or endorsed this forecast.</li>"
        "<li>Station levels: NOAA Global Summary of the Day.</li>"
        "<li>District boundaries: GADM 4.1, interim.</li></ul>"
        f"<p>Model <code>{escape(spec['version'])}</code>. Code and methods: "
        f'<a href="{escape(repo_url)}">{escape(repo_url)}</a>.</p>',
    )


def _table_for_download(rows: pd.DataFrame, names: dict) -> pd.DataFrame:
    d = rows.copy()
    d.insert(d.columns.get_loc("district_id") + 1, "district", d["district_id"].map(names))
    d = d.drop(columns=["label_version"])
    for c in ("issue_date", "asof_date", "valid_date", "window_end"):
        d[c] = pd.to_datetime(d[c]).dt.strftime("%Y-%m-%d")
    return d


def build(
    forecast_dir: Path,
    out: Path,
    names: dict,
    regions: dict,
    spec: dict,
    repo_url: str,
    keep_days: int,
    scorecard_body: str = "",
) -> dict:
    """Write the site from the newest forecast file, plus CSVs of the last keep_days."""
    files = sorted(forecast_dir.glob("issue_year=*/*.parquet"), key=lambda f: f.stem)
    if not files:
        raise SystemExit(f"no forecasts under {forecast_dir}")
    rows = pd.read_parquet(files[-1])
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    (out / ".nojekyll").write_text("")
    (out / "index.html").write_text(index_html(rows, names, regions), encoding="utf-8")
    (out / "about.html").write_text(about_html(spec, repo_url), encoding="utf-8")
    (out / "scorecard.html").write_text(_page("Scorecard", scorecard_body), encoding="utf-8")
    table = _table_for_download(rows, names)
    table.to_csv(data / "latest.csv", index=False)
    records = json.loads(table.replace({np.nan: None}).to_json(orient="records"))
    (data / "latest.json").write_text(json.dumps(records), encoding="utf-8")
    past = []
    for f in files[-keep_days:]:
        _table_for_download(pd.read_parquet(f), names).to_csv(data / f"{f.stem}.csv", index=False)
        past.append(f"{f.stem}.csv")
    (data / "index.json").write_text(json.dumps({"days": past}), encoding="utf-8")
    return {"issue": files[-1].stem, "files": len(past) + 5}
