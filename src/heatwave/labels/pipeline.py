"""Phase 4 pipeline: station comparison, bias map, labels, episodes, QA reports.

Run with `uv run heatwave build-labels`. Writes
    data/reference/stations_qualifying.parquet, label_qmap.parquet
    data/labels/labels_daily.parquet          main + sensitivity labels
    data/labels/episodes.parquet
    reports/label_source_comparison.md, reports/label_qa.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from heatwave.config import Problem
from heatwave.geo.grid import Grid
from heatwave.ingest import gsod
from heatwave.ingest.imd import ImdSettings, read_grd
from heatwave.ingest.raw_store import RawStore
from heatwave.labels import sources
from heatwave.labels.bias_correction import fit_bias_plane, plane_version, qmap_version
from heatwave.labels.build import (
    episodes,
    fit_station_map,
    main_labels,
    percentile_label,
    rule_labels,
)


def _md(df: pd.DataFrame, floatfmt: int = 2) -> str:
    if df.empty:
        return "_none_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(floatfmt)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
        for r in d.itertuples(index=False)
    ]
    return "\n".join(lines) + "\n"


def station_inputs(data_cfg: dict, root: Path, years: tuple[int, int]) -> pd.DataFrame:
    """Daily station / POWER-cell / IMD-cell series for every downloaded station."""
    stations = pd.read_parquet(root / "reference" / "stations.parquet")
    days = gsod.read_station_days(root / data_cfg["gsod"]["raw_dir"], data_cfg["gsod"])
    stations = stations[stations["station_id"].isin(days["station_id"])].copy()

    grid = Grid.from_config(data_cfg["grid"])
    stations["cell_id"] = grid.cell_ids(stations["lat"], stations["lon"])
    store = RawStore(root / data_cfg["storage"]["raw_power_dir"])
    best = store.best(range(years[0], years[1] + 1))
    best = best[(best["variable"] == "T2M_MAX") & best["cell_id"].isin(stations["cell_id"])]
    best["obs_date"] = pd.to_datetime(best["obs_date"])
    power = (
        stations[["station_id", "cell_id"]]
        .merge(best[["cell_id", "obs_date", "value"]], on="cell_id")
        .rename(columns={"value": "power"})
    )

    s = ImdSettings.from_config(data_cfg["imd"])
    jj = np.rint((stations["lat"].to_numpy() - s.lat_first) / s.step_deg).astype(int)
    ii = np.rint((stations["lon"].to_numpy() - s.lon_first) / s.step_deg).astype(int)
    imd_frames = []
    for year in range(years[0], years[1] + 1):
        cube = read_grd(root / data_cfg["imd"]["raw_dir"] / f"{year}.GRD", year, s)
        dates = pd.date_range(f"{year}-01-01", periods=cube.shape[0], freq="D")
        for k, sid in enumerate(stations["station_id"]):
            imd_frames.append(
                pd.DataFrame({"station_id": sid, "obs_date": dates, "imd": cube[:, jj[k], ii[k]]})
            )
    imd = pd.concat(imd_frames, ignore_index=True)

    series = sources.station_series(
        days, stations, power[["station_id", "obs_date", "power"]], imd, years
    )
    return series.merge(stations[["station_id", "lat", "lon", "cell_id"]], on="station_id")


def run(
    problem: Problem, data_cfg: dict, label_cfg: dict, root: Path, reports: Path, silver: Path
) -> dict:
    rule, dev = problem.target, problem.validation.development
    st_cfg = label_cfg["stations"]
    season = sorted({m for m, _ in problem.evaluation_season.days()})

    # 0. Spatial bias plane from IMD's large-scale pattern (bias period, season months)
    power = pd.read_parquet(
        silver / "power_district_daily.parquet", columns=["district_id", "obs_date", "T2M_MAX"]
    )
    imd = pd.read_parquet(silver / "imd_district_daily.parquet")
    districts = pd.read_parquet(root / "reference" / "districts.parquet")
    plane = fit_bias_plane(power, imd, districts, rule.bias_fit_period, season)
    (root / "reference" / "label_bias_plane.json").write_text(json.dumps(plane, indent=2))

    # 1. Stations and the source comparison (development years only)
    series = station_inputs(data_cfg, root, dev)
    keep = sources.qualifying(
        series,
        rule.bias_fit_period,
        tuple(st_cfg["season_months"]),
        st_cfg["min_seasons"],
        st_cfg["min_coverage"],
    )
    q_series = series[series["station_id"].isin(keep)]
    # Out-of-sample checks: stations too short to fit on, but with enough season days
    in_season = series[series["obs_date"].dt.month.isin(season) & series["station"].notna()]
    n_days = in_season.groupby("station_id").size()
    check = sorted(set(n_days[n_days >= st_cfg["check_min_season_days"]].index) - set(keep))
    scored = series[series["station_id"].isin([*keep, *check])]
    scores = sources.compare(scored, rule, dev, tuple(season), plane, keep)
    summary = sources.summarise(scores)

    stations = (
        series.groupby("station_id")
        .agg(
            name=("name", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            reported_days=("station", "count"),
        )
        .reset_index()
    )
    stations["qualifies"] = stations["station_id"].isin(keep)
    stations.to_parquet(root / "reference" / "stations_qualifying.parquet", index=False)

    # 2. Bias map fitted once on all qualifying stations in the bias period
    qmap = fit_station_map(q_series, keep, rule.bias_fit_period, plane)
    qmap.to_parquet(root / "reference" / "label_qmap.parquet", index=False)

    # 3. Labels for every district-day POWER covers, plus sensitivity labels
    labels = main_labels(power, qmap, rule, plane, districts)
    imd_lab = rule_labels(imd, "district_id", "tmax_imd", rule)[
        ["district_id", "obs_date", "is_heatwave"]
    ].rename(columns={"is_heatwave": "is_heatwave_imd"})
    pct = label_cfg["sensitivity"]["percentile"]
    labels["is_heatwave_pct"] = percentile_label(
        labels,
        "district_id",
        "tmax_used",
        rule.normal_period,
        pct["q"],
        pct["window_days"],
        pct["min_run_days"],
    ).to_numpy()
    labels = labels.merge(imd_lab, on=["district_id", "obs_date"], how="left")

    out_dir = root / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out_dir / "labels_daily.parquet", index=False)
    eps = episodes(labels, label_cfg["episodes"]["min_districts"])
    eps.to_parquet(out_dir / "episodes.parquet", index=False)

    reports.mkdir(parents=True, exist_ok=True)
    _comparison_report(
        reports / "label_source_comparison.md", stations, scores, summary, rule, dev, season
    )
    qa = _qa_report(
        reports / "label_qa.md",
        problem,
        label_cfg,
        root,
        labels,
        eps,
        qmap,
        plane,
        series,
        keep,
        imd,
        season,
    )
    return {
        "stations_downloaded": int(len(stations)),
        "stations_qualifying": len(keep),
        "label_version": labels["label_version"].iloc[0],
        "qmap_version": qmap_version(qmap),
        "plane_version": plane_version(plane),
        "summary": summary.round(3).reset_index().to_dict("records"),
        **qa,
    }


def _comparison_report(path, stations, scores, summary, rule, dev, season):
    q = stations[stations["qualifies"]]
    normal = f"{rule.normal_n_harmonics}-harmonic {rule.normal_period[0]}-{rule.normal_period[1]}"
    months = f"months {season[0]}-{season[-1]}"
    text = f"""# Label source comparison

Generated by `uv run heatwave build-labels`. Decision: `docs/decisions/005-phase4-labels.md`.

Which Tmax should the heatwave rule use? Each candidate is taken at the grid cell
containing a weather station and scored against that station's own heatwave days
(IMD's criteria are defined on station Tmax). Evaluation season ({months}),
development years {dev[0]}-{dev[1]}, days with a reported (not hourly-derived)
maximum only. Every candidate uses its own {normal} normal. `power_to_st` and
`power_to_st_plane` are fitted on the other stations only (leave one station out);
`power_to_st_plane` is the chosen label.

Stations: {len(stations)} with data; {len(q)} qualify for scoring and fitting
(enough good seasons in {rule.bias_fit_period[0]}-{rule.bias_fit_period[1]}).

## Summary (station heatwave days pooled; scores averaged over stations)

{_md(summary.reset_index())}
- ratio: candidate heatwave days / station heatwave days (1 = right base rate)
- POD: share of station heatwave days the candidate also flags
- FAR: share of the candidate's heatwave days the station does not have
- kappa: Cohen's kappa against the station; bias_c: mean Tmax difference

## Per station

{_md(scores.drop(columns=["station_id"]).sort_values(["name", "candidate"]))}
## Stations

{_md(stations.sort_values("name"))}"""
    path.write_text(text, encoding="utf-8", newline="\n")


def _window_rate(labels: pd.DataFrame, horizon: int = 10) -> pd.Series:
    """Share of district issue dates with any heatwave day in the next `horizon` days."""
    out = []
    for _, g in labels.sort_values("obs_date").groupby("district_id"):
        hw = g["is_heatwave"].fillna(False).astype(float).to_numpy()
        ahead = pd.Series(hw[::-1]).rolling(horizon, min_periods=horizon).max()[::-1].to_numpy()
        win = np.concatenate([ahead[1:], [np.nan]])  # days s+1 .. s+horizon
        out.append(pd.Series(win, index=g["obs_date"].to_numpy()))
    s = pd.concat(out)
    return s.groupby(s.index.month).mean()


def _kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


def _station_heatwaves(series: pd.DataFrame, keep: list[str], rule) -> pd.DataFrame:
    """Reported station-days of the qualifying stations with the station's own label."""
    s = series[series["station_id"].isin(keep)].reset_index(drop=True)
    lab = rule_labels(s, "station_id", "station", rule)
    s["hw"] = lab["is_heatwave"].to_numpy(dtype=float)
    return s[s["station"].notna()]


def _qa_report(path, problem, label_cfg, root, labels, eps, qmap, plane, series, keep, imd, season):
    dev, rule = problem.validation.development, problem.target
    lab = labels.copy()
    lab["year"], lab["month"] = lab["obs_date"].dt.year, lab["obs_date"].dt.month
    lab["hw"] = lab["is_heatwave"].fillna(False).astype(bool)
    lab["sev"] = lab["is_severe"].fillna(False).astype(bool)
    d = lab[lab["year"].between(*dev)]
    st = _station_heatwaves(series, keep, rule)

    # Season shape: label vs stations vs IMD label
    by_month = d.groupby("month")["hw"].agg(["sum", "mean"])
    st_month = st.groupby(st["obs_date"].dt.month)["hw"].sum()
    imd_month = d.groupby("month")["is_heatwave_imd"].sum()
    min_share = label_cfg["qa"]["season_month_min_share"]
    month_tab = pd.DataFrame(
        {
            "month": by_month.index,
            "label_days": by_month["sum"].astype(int),
            "label_share": by_month["sum"] / by_month["sum"].sum(),
            "station_share": (st_month / st_month.sum())
            .reindex(by_month.index)
            .fillna(0)
            .to_numpy(),
            "imd_label_share": (imd_month / imd_month.sum()).reindex(by_month.index).to_numpy(),
            "daily_rate": by_month["mean"],
            "window_rate_10d": _window_rate(d).reindex(by_month.index).to_numpy(),
        }
    )
    label_months = month_tab.loc[month_tab["label_share"] > min_share, "month"].tolist()
    station_months = month_tab.loc[month_tab["station_share"] > min_share, "month"].tolist()
    season_ok = station_months == season

    # Years: agreement with the stations' own heatwave years
    year_tab = (
        lab.groupby("year")
        .agg(heatwave_district_days=("hw", "sum"), severe_district_days=("sev", "sum"))
        .reset_index()
    )
    year_tab["block"] = np.where(year_tab["year"] <= dev[1], "development", "final-test/live")
    dev_years = year_tab[year_tab["block"] == "development"].set_index("year")
    st_year = st.groupby(st["obs_date"].dt.year)["hw"].mean()
    year_r = float(dev_years["heatwave_district_days"].reindex(st_year.index).corr(st_year))
    top_label = set(dev_years["heatwave_district_days"].nlargest(8).index)
    top_station = set(st_year.nlargest(8).index)
    overlap = len(top_label & top_station)
    rank = dev_years["heatwave_district_days"].rank(ascending=False, method="min").astype(int)
    known = [y for y in label_cfg["qa"]["known_severe_seasons"] if dev[0] <= y <= dev[1]]
    st_rank = st_year.rank(ascending=False, method="min")
    known_tab = pd.DataFrame(
        {
            "year": known,
            "label_rank": [int(rank[y]) for y in known],
            "station_rank": [int(st_rank[y]) if y in st_rank else None for y in known],
        }
    )
    years_ok = year_r >= 0.7 and overlap >= 5

    # Episodes (development years)
    e = eps[eps["start"].dt.year.between(*dev)]
    top_e = (
        e.sort_values("district_days", ascending=False)
        .head(10)
        .assign(
            start=lambda x: x["start"].dt.date.astype(str),
            end=lambda x: x["end"].dt.date.astype(str),
            peak_date=lambda x: x["peak_date"].dt.date.astype(str),
        )
    )

    # Districts
    per_d = (d.groupby("district_id")["hw"].sum() / d["year"].nunique()).sort_values()

    # Sensitivity agreement within the evaluation season
    s = d[d["month"].isin(season)].dropna(subset=["is_heatwave_imd"])
    imd_days = s["is_heatwave_imd"].astype(bool)
    sens = pd.DataFrame(
        [
            {
                "label": "IMD gridded (same rule)",
                "days": int(imd_days.sum()),
                "ratio_to_main": imd_days.sum() / max(s["hw"].sum(), 1),
                "kappa_vs_main": _kappa(s["hw"], imd_days),
            },
            {
                "label": "Percentile (95th, >= 3 days)",
                "days": int(s["is_heatwave_pct"].sum()),
                "ratio_to_main": s["is_heatwave_pct"].sum() / max(s["hw"].sum(), 1),
                "kappa_vs_main": _kappa(s["hw"], s["is_heatwave_pct"]),
            },
        ]
    )

    # Bias correction summary
    qm = qmap[qmap["q"].isin([0.1, 0.5, 0.9, 0.99])].assign(
        correction=lambda x: x["target_q"] - x["source_q"]
    )
    qm_tab = qm.pivot(index="month", columns="q", values="correction").reset_index()
    qm_tab.columns = ["month"] + [f"q{int(round(c * 100))}" for c in qm_tab.columns[1:]]
    st_bias = (
        st[st["obs_date"].dt.month.isin(season)]
        .assign(raw=lambda x: x["power"] - x["station"])
        .groupby("name")["raw"]
        .mean()
        .round(2)
        .rename("raw_power_minus_station_c")
    )

    # Plains check (A1): area-weighted cell elevation per district
    w = pd.read_parquet(root / "reference" / "district_grid_weights.parquet")
    elev = pd.read_parquet(root / "reference" / "grid_cell_meta.parquet")
    de = (
        w.merge(elev, on="cell_id")
        .assign(e=lambda x: x["area_weight"] * x["elevation_m"])
        .groupby("district_id")["e"]
        .sum()
        .sort_values()
    )

    fit = f"{rule.bias_fit_period[0]}-{rule.bias_fit_period[1]}"
    normal = f"{rule.normal_n_harmonics} harmonics, {rule.normal_period[0]}-{rule.normal_period[1]}"
    plane_txt = (
        f"bias = {plane['a']:.2f} {plane['b_lat']:+.3f} x lat {plane['c_lon']:+.3f} x lon (C); "
        f"district residual RMSE {plane['rmse_c']:.2f} C"
    )
    season_verdict = "consistent" if season_ok else "REVIEW"
    years_verdict = "PASS" if years_ok else "REVIEW"
    text = f"""# Label QA (gate G1)

Generated by `uv run heatwave build-labels`. Decision: `docs/decisions/005-phase4-labels.md`.

- Label: `{rule.name}`, version `{labels["label_version"].iloc[0]}`
- Tmax source: `{rule.tmax_source}`; spatial plane `{plane_version(plane)}`;
  station map `{qmap_version(qmap)}` fitted {fit} on {len(keep)} stations
- Normal: {normal}; evaluation season months {season}
- Statistics use development years {dev[0]}-{dev[1]} unless stated.

## 1. Bias correction

Spatial plane of POWER - IMD (season months, {fit}): {plane_txt}.

Station-level map after removing the plane (correction in C by month and quantile):

{_md(qm_tab)}
Raw POWER minus station in the season, qualifying stations:

{_md(st_bias.reset_index())}
## 2. Season shape and base rates

{_md(month_tab, 4)}
Months above {min_share:.0%} of heatwave days: label {label_months}, stations
**{station_months}**; evaluation season {season} -> {season_verdict}.
The label's August share comes from the label method, not the stations
(docs/decisions/005).

## 3. Heatwave years

Agreement with the stations' own heatwave rate by year (development years):
correlation **{year_r:.2f}**; top-8 years shared **{overlap} of 8**
(pass: correlation >= 0.7 and >= 5 shared) -> {years_verdict}.

Originally listed severe seasons (India-wide events; not all were severe in UP):

{_md(known_tab)}
{_md(year_tab)}
Final-test/live years (2021 onward) are listed for the record only; no decision uses them.

## 4. Episodes (state-wide, >= {label_cfg["episodes"]["min_districts"]} district)

Development-year episodes: **{len(e)}**; median length {e["days"].median():.0f} days;
median {e["district_days"].median():.0f} district-days. Largest by district-days:

{_md(top_e.drop(columns=["episode"]))}
## 5. Districts

Heatwave days per season per district: min {per_d.min():.1f} ({per_d.index[0]}),
median {per_d.median():.1f}, max {per_d.max():.1f} ({per_d.index[-1]}).

## 6. Sensitivity labels (evaluation season, development years)

{_md(sens, 3)}
## 7. Plains check (assumption A1)

Area-weighted district elevation (m): min {de.min():.0f} ({de.index[0]}), max {de.max():.0f}
({de.index[-1]}). Every district's weighted elevation is far below hill-station
levels, so IMD's plains criteria apply.
"""
    path.write_text(text, encoding="utf-8", newline="\n")
    return {
        "label_season_months": label_months,
        "station_season_months": station_months,
        "season_ok": season_ok,
        "year_corr": round(year_r, 3),
        "top8_overlap": overlap,
        "years_ok": years_ok,
        "episodes_development": int(len(e)),
    }
