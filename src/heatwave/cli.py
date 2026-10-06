"""Command-line entry point. Every pipeline step becomes a subcommand (reference.md §5.2).

Commands:
    uv run heatwave show-problem
    uv run heatwave build-geo
"""

from __future__ import annotations

import argparse
import json

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="heatwave")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show-problem", help="print the frozen problem definition").set_defaults(
        func=cmd_show_problem
    )
    sub.add_parser(
        "build-geo", help="district polygons -> grid cells -> area weights (data/reference/)"
    ).set_defaults(func=cmd_build_geo)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
