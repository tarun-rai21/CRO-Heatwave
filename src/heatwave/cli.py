"""Command-line entry point. Every pipeline step becomes a subcommand (reference.md §5.2).

Commands:
    uv run heatwave show-problem       print the frozen problem definition
    uv run heatwave build-geo          district/grid reference tables
    uv run heatwave ingest-backfill    full POWER history for every cell (resumable)
    uv run heatwave ingest-daily       trailing-window re-fetch (--sync: via Hugging Face)
    uv run heatwave check-raw          completeness of the raw store
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
    files = sorted(
        f for sub in ("reference", "raw", "runs") for f in (paths.root / sub).rglob("*.parquet")
    )
    n = hub.push(cfg, paths.root, files, "sync reference, raw and runs")
    print(f"uploaded {n} files")


def cmd_hub_pull(args: argparse.Namespace) -> None:
    from heatwave import hub

    cfg, paths = _ingest_context()
    hub.pull(cfg, paths.root, ["reference/*", "raw/**", "runs/*"])
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
        "hub-push-all", help="upload reference, raw and runs to Hugging Face"
    ).set_defaults(func=cmd_hub_push_all)
    sub.add_parser(
        "hub-pull", help="download reference, raw and runs from Hugging Face"
    ).set_defaults(func=cmd_hub_pull)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
