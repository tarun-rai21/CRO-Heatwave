"""Command-line entry point. Every pipeline step becomes a subcommand (reference.md §5.2).

Commands:
    uv run heatwave show-problem       print the frozen problem definition
    uv run heatwave build-geo          district/grid reference tables
    uv run heatwave ingest-backfill    full POWER history for every cell (resumable)
    uv run heatwave ingest-daily       trailing-window re-fetch (--sync: via Hugging Face)
    uv run heatwave check-raw          completeness of the raw store
    uv run heatwave ingest-imd         IMD gridded Tmax, 1981 to last full year (resumable)
    uv run heatwave qa-report          full-history data QA -> reports/data_qa.md
    uv run heatwave validate           daily gate: hard checks, completeness, latency
    uv run heatwave build-district-daily  cells -> district daily tables (silver)
    uv run heatwave ingest-gsod        NOAA station maxima, development years
    uv run heatwave build-labels       labels, episodes, label reports
    uv run heatwave build-features     feature tables per block, docs/features.md
    uv run heatwave hub-push-all       upload reference/raw/runs to Hugging Face
    uv run heatwave hub-pull           download them
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from heatwave.config import REPO_ROOT, load_problem, read_yaml


def cmd_show_problem(args: argparse.Namespace) -> None:
    p = load_problem()
    t, v = p.target, p.validation
    print(f"target       {t.name}  (Tmax source: {t.tmax_source})")
    print(
        f"rule         heatwave: Tmax >= {t.tmax_floor_c}"
        f" and departure >= {t.departure_heatwave_c}, or Tmax >= {t.absolute_heatwave_c}"
    )
    print(
        f"             severe:   Tmax >= {t.tmax_floor_c} and departure > {t.departure_severe_c},"
        f" or Tmax >= {t.absolute_severe_c}"
    )
    print(f"normal       {t.normal_period[0]}-{t.normal_period[1]}, {t.normal_smoothing}")
    print(
        f"dates        issue = as-of + {p.latency_days} days;"
        f" leads {p.horizon_min}..{p.horizon_max}"
    )
    print(
        f"seasons      evaluate {p.evaluation_season.start}..{p.evaluation_season.end}, "
        f"train {p.training_season.start}..{p.training_season.end}"
    )
    print(
        f"blocks       development {v.development}, final test {v.final_test}, "
        f"2026 season {v.season_2026}"
    )
    for f in v.folds:
        print(f"  fold {f.fold}     train {f.train}  validate {f.validate}")
    for tier in p.tiers:
        print(f"tier         {tier.name:8s} {tier.rule} {tier.params}")


def cmd_build_geo(args: argparse.Namespace) -> None:
    from heatwave.geo.weights import build_geo  # geo stack is slow to import

    print(json.dumps(build_geo(read_yaml("data.yaml"), REPO_ROOT), indent=2))


def _ingest_context():
    from heatwave.ingest.run import Paths

    cfg = read_yaml("data.yaml")
    return cfg, Paths.from_config(cfg, REPO_ROOT)


def cmd_ingest_backfill(args: argparse.Namespace) -> None:
    from heatwave.ingest.run import run_ingest

    cfg, paths = _ingest_context()
    start = date.fromisoformat(cfg["power"]["start"])
    end = date.fromisoformat(args.end) if args.end else date.today()
    # Responses are cached, so an interrupted backfill resumes without re-downloading.
    run_ingest(cfg, paths, start, end, kind="backfill", use_cache=True)


def cmd_ingest_daily(args: argparse.Namespace) -> None:
    from heatwave import hub
    from heatwave.ingest.run import daily_window, run_ingest, utc_now

    cfg, paths = _ingest_context()
    start, end = daily_window(utc_now().date(), cfg["power"]["trailing_days"])
    if args.sync:
        years = range(start.year, end.year + 1)
        patterns = ["reference/*", "runs/*", *[f"raw/power/obs_year={y}/*" for y in years]]
        hub.pull(cfg, paths.root, patterns)
    try:
        run_ingest(cfg, paths, start, end, kind="daily")
    finally:
        if args.sync:
            # Upload whatever was stored, even if some cells failed.
            runs = pd.read_parquet(paths.runs_log)
            run_id = runs["run_id"].iloc[-1]
            files = [*hub.files_of_run(paths.raw_power, run_id), paths.runs_log]
            hub.push(cfg, paths.root, files, f"daily ingest {run_id}")


def cmd_check_raw(args: argparse.Namespace) -> None:
    from heatwave.ingest.run import check_spine

    cfg, paths = _ingest_context()
    report = check_spine(
        paths, date.fromisoformat(cfg["power"]["start"]), cfg["power"]["parameters"]
    )
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit("raw data incomplete")


def cmd_hub_push_all(args: argparse.Namespace) -> None:
    from heatwave import hub

    cfg, paths = _ingest_context()
    merged = hub.merge_remote_log(cfg, paths.root, paths.runs_log)
    if merged:
        print(f"merged {merged} run-log rows written remotely since the last pull")
    files = sorted(
        f
        for sub in ("reference", "raw", "runs", "labels")
        for f in (paths.root / sub).rglob("*")
        if f.is_file()
    )
    n = hub.push(cfg, paths.root, files, "sync reference, raw and runs")
    print(f"uploaded {n} files")


def cmd_ingest_gsod(args: argparse.Namespace) -> None:
    import requests

    from heatwave.geo.weights import load_districts
    from heatwave.ingest import gsod

    cfg, paths = _ingest_context()
    g = cfg["gsod"]
    resp = requests.get(g["history_url"], timeout=120)
    resp.raise_for_status()
    up = load_districts(cfg["boundaries"], REPO_ROOT).geometry.union_all()
    stations = gsod.station_list(resp.content, up, g["boundary_buffer_deg"])
    stations.to_parquet(paths.root / "reference" / "stations.parquet", index=False)
    print(f"{len(stations)} stations in or near UP")
    manifest = gsod.download(g, stations, paths.root / g["raw_dir"])
    ok = manifest[manifest["status"] == "ok"]
    print(f"{len(ok)} station-years downloaded from {ok['station_id'].nunique()} stations")


def cmd_build_labels(args: argparse.Namespace) -> None:
    from heatwave.labels.pipeline import run

    cfg, paths = _ingest_context()
    summary = run(
        load_problem(),
        cfg,
        read_yaml("labels.yaml"),
        paths.root,
        REPO_ROOT / "reports",
        REPO_ROOT / cfg["outputs"]["silver_dir"],
    )
    print(json.dumps(summary, indent=2, default=str))


def cmd_build_features(args: argparse.Namespace) -> None:
    from heatwave.features.pipeline import run

    cfg, paths = _ingest_context()
    manifest = run(
        load_problem(),
        read_yaml("features.yaml"),
        paths.root,
        REPO_ROOT / cfg["outputs"]["silver_dir"],
        REPO_ROOT / "docs",
    )
    print(json.dumps(manifest, indent=2))


def cmd_build_district_daily(args: argparse.Namespace) -> None:
    from heatwave.geo.aggregate import imd_district_daily, power_district_daily
    from heatwave.ingest.imd import ImdSettings
    from heatwave.ingest.raw_store import RawStore

    cfg, paths = _ingest_context()
    ref = paths.root / "reference"
    out_dir = REPO_ROOT / cfg["outputs"]["silver_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    power = power_district_daily(
        RawStore(paths.raw_power),
        pd.read_parquet(ref / "district_grid_weights.parquet"),
        cfg["power"]["parameters"],
        cfg["geo"]["cellmax_min_share"],
    )
    power.to_parquet(out_dir / "power_district_daily.parquet", index=False)

    imd_cfg = cfg["imd"]
    imd_dir = paths.root / imd_cfg["raw_dir"]
    years = sorted(int(p.stem) for p in imd_dir.glob("*.GRD"))
    imd = imd_district_daily(
        ImdSettings.from_config(imd_cfg),
        imd_dir,
        years,
        pd.read_parquet(ref / "district_imd_weights.parquet"),
    )
    imd.to_parquet(out_dir / "imd_district_daily.parquet", index=False)
    print(
        json.dumps(
            {
                "power_rows": len(power),
                "power_period": [
                    str(power.obs_date.min().date()),
                    str(power.obs_date.max().date()),
                ],
                "power_missing_cell_values": int(power["n_missing"].sum()),
                "imd_rows": len(imd),
                "imd_period": [str(imd.obs_date.min().date()), str(imd.obs_date.max().date())],
                "imd_missing_cell_values": int(imd["n_missing"].sum()),
                "districts": int(power["district_id"].nunique()),
            },
            indent=2,
        )
    )


def cmd_qa_report(args: argparse.Namespace) -> None:
    from heatwave.quality.report import build_report

    cfg, paths = _ingest_context()
    summary = build_report(
        cfg, read_yaml("quality.yaml"), paths, REPO_ROOT / "reports" / "data_qa.md"
    )
    print(json.dumps(summary, indent=2))
    if not summary["ok"]:
        raise SystemExit("data QA failed: see reports/data_qa.md")


def cmd_validate(args: argparse.Namespace) -> None:
    from heatwave.ingest.run import utc_now
    from heatwave.quality.report import validate_recent

    cfg, paths = _ingest_context()
    result = validate_recent(cfg, read_yaml("quality.yaml"), paths, utc_now().date())
    print(json.dumps(result, indent=2, default=str))
    if not result["ok"]:
        raise SystemExit("validation failed: " + "; ".join(result["problems"]))


def cmd_ingest_imd(args: argparse.Namespace) -> None:
    from heatwave.ingest.imd import ImdSettings, ingest_years

    cfg, paths = _ingest_context()
    imd = cfg["imd"]
    first = imd["start_year"]
    last = int(args.last_year) if args.last_year else utc_now_year() - 1
    manifest = ingest_years(
        ImdSettings.from_config(imd), paths.root / imd["raw_dir"], list(range(first, last + 1))
    )
    print(f"{len(manifest)} years in manifest: {manifest['year'].min()}-{manifest['year'].max()}")


def utc_now_year() -> int:
    from heatwave.ingest.run import utc_now

    return utc_now().year


def cmd_hub_pull(args: argparse.Namespace) -> None:
    from heatwave import hub

    cfg, paths = _ingest_context()
    hub.pull(cfg, paths.root, ["reference/*", "raw/**", "runs/*", "labels/*"])
    print(f"pulled into {paths.root}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="heatwave")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show-problem", help="print the frozen problem definition").set_defaults(
        func=cmd_show_problem
    )
    sub.add_parser(
        "build-geo", help="district polygons -> grid cells -> area weights (data/reference/)"
    ).set_defaults(func=cmd_build_geo)

    backfill = sub.add_parser("ingest-backfill", help="fetch full POWER history for every cell")
    backfill.add_argument("--end", help="last date to request (default: today)")
    backfill.set_defaults(func=cmd_ingest_backfill)

    daily = sub.add_parser("ingest-daily", help="re-fetch the trailing window for every cell")
    daily.add_argument("--sync", action="store_true", help="pull from / push to Hugging Face")
    daily.set_defaults(func=cmd_ingest_daily)

    sub.add_parser("check-raw", help="completeness of the raw store").set_defaults(
        func=cmd_check_raw
    )
    sub.add_parser(
        "build-district-daily", help="cells -> district daily tables for POWER and IMD (silver)"
    ).set_defaults(func=cmd_build_district_daily)
    sub.add_parser(
        "ingest-gsod", help="NOAA station daily maxima for UP, development years (resumable)"
    ).set_defaults(func=cmd_ingest_gsod)
    sub.add_parser(
        "build-labels", help="station comparison, bias map, labels, episodes, QA reports"
    ).set_defaults(func=cmd_build_labels)
    sub.add_parser(
        "build-features", help="feature tables per fold + development, climatologies, dictionary"
    ).set_defaults(func=cmd_build_features)
    sub.add_parser("qa-report", help="full-history data QA -> reports/data_qa.md").set_defaults(
        func=cmd_qa_report
    )
    sub.add_parser("validate", help="daily gate: hard checks, completeness, latency").set_defaults(
        func=cmd_validate
    )
    imd = sub.add_parser("ingest-imd", help="download IMD 1-degree gridded Tmax (resumable)")
    imd.add_argument("--last-year", help="last year to download (default: last full year)")
    imd.set_defaults(func=cmd_ingest_imd)
    sub.add_parser(
        "hub-push-all", help="upload reference, raw, runs and labels to Hugging Face"
    ).set_defaults(func=cmd_hub_push_all)
    sub.add_parser(
        "hub-pull", help="download reference, raw and runs from Hugging Face"
    ).set_defaults(func=cmd_hub_pull)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
