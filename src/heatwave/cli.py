"""Command-line entry point. Every pipeline step becomes a subcommand (reference.md §5.2).

uv run heatwave show-problem
"""

from __future__ import annotations

import argparse

from heatwave.config import load_problem


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="heatwave")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show-problem", help="print the frozen problem definition").set_defaults(
        func=cmd_show_problem
    )
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
