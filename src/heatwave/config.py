"""Typed, validated access to configs/problem.yaml.

The problem definition is loaded once into frozen dataclasses, so the rest of
the code depends on named fields rather than dictionary keys, and an invalid
definition fails at load time instead of producing quietly wrong results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

TMAX_SOURCES = {"undecided", "power", "power_bias_corrected", "imd_gridded"}
NORMAL_SMOOTHING = {"harmonics", "window"}


class ConfigError(ValueError):
    """The problem definition is inconsistent."""


Years = tuple[int, int]


@dataclass(frozen=True)
class DayWindow:
    """A recurring calendar window, inclusive at both ends, as "MM-DD" strings."""

    start: str
    end: str

    def days(self) -> set[tuple[int, int]]:
        """(month, day) pairs inside the window, using a leap year."""
        start, end = _parse_month_day(self.start), _parse_month_day(self.end)
        out = set()
        for ordinal in range(date(2000, 1, 1).toordinal(), date(2000, 12, 31).toordinal() + 1):
            d = date.fromordinal(ordinal)
            md = (d.month, d.day)
            if (start <= md <= end) if start <= end else (md >= start or md <= end):
                out.add(md)
        return out


@dataclass(frozen=True)
class HeatwaveRule:
    name: str
    tmax_floor_c: float
    departure_heatwave_c: float
    departure_severe_c: float
    absolute_heatwave_c: float
    absolute_severe_c: float
    round_decimals: int
    normal_period: Years
    normal_smoothing: str
    normal_n_harmonics: int
    tmax_source: str


@dataclass(frozen=True)
class Fold:
    fold: int
    train: Years
    validate: Years


@dataclass(frozen=True)
class Validation:
    development: Years
    final_test: Years
    season_2026: Years
    folds: tuple[Fold, ...]
    inner_block_seasons: int
    bootstrap_unit: str


@dataclass(frozen=True)
class TierRule:
    name: str
    rule: str
    params: dict[str, float]


@dataclass(frozen=True)
class Problem:
    latency_days: int
    horizon_min: int
    horizon_max: int
    evaluation_season: DayWindow
    training_season: DayWindow
    target: HeatwaveRule
    onset_lookback_days: int
    validation: Validation
    tiers: tuple[TierRule, ...]

    @property
    def horizons(self) -> range:
        return range(self.horizon_min, self.horizon_max + 1)


def read_yaml(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    with open(config_dir / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_problem(config_dir: Path = CONFIG_DIR) -> Problem:
    raw = read_yaml("problem.yaml", config_dir)
    try:
        problem = _parse(raw)
    except (KeyError, TypeError) as e:
        raise ConfigError(f"problem.yaml is malformed: {e!r}") from e
    validate_problem(problem)
    return problem


def _parse_month_day(md: str) -> tuple[int, int]:
    month, day = (int(x) for x in md.split("-"))
    date(2000, month, day)  # raises on impossible dates; 2000 allows 02-29
    return month, day


def _years(pair: list[int]) -> Years:
    start, end = (int(x) for x in pair)
    return start, end


def _parse(raw: dict[str, Any]) -> Problem:
    dates, season, target, val = raw["dates"], raw["season"], raw["target"], raw["validation"]
    normal = target["normal"]
    return Problem(
        latency_days=int(dates["latency_days"]),
        horizon_min=int(dates["horizon_min"]),
        horizon_max=int(dates["horizon_max"]),
        evaluation_season=DayWindow(**season["evaluation"]),
        training_season=DayWindow(**season["training"]),
        target=HeatwaveRule(
            name=target["name"],
            tmax_floor_c=float(target["tmax_floor_c"]),
            departure_heatwave_c=float(target["departure_heatwave_c"]),
            departure_severe_c=float(target["departure_severe_c"]),
            absolute_heatwave_c=float(target["absolute_heatwave_c"]),
            absolute_severe_c=float(target["absolute_severe_c"]),
            round_decimals=int(target["round_decimals"]),
            normal_period=_years(normal["period"]),
            normal_smoothing=normal["smoothing"],
            normal_n_harmonics=int(normal["n_harmonics"]),
            tmax_source=target["tmax_source"],
        ),
        onset_lookback_days=int(raw["onset"]["lookback_days"]),
        validation=Validation(
            development=_years(val["development"]),
            final_test=_years(val["final_test"]),
            season_2026=_years(val["season_2026"]),
            folds=tuple(
                Fold(fold=int(f["fold"]), train=_years(f["train"]), validate=_years(f["validate"]))
                for f in val["folds"]
            ),
            inner_block_seasons=int(val["inner_block_seasons"]),
            bootstrap_unit=val["bootstrap_unit"],
        ),
        tiers=tuple(
            TierRule(
                name=name,
                rule=spec["rule"],
                params={k: float(v) for k, v in spec.items() if k != "rule"},
            )
            for name, spec in raw["tiers"].items()
        ),
    )


def validate_problem(p: Problem) -> None:
    """Reject definitions that would silently break the evaluation design."""
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    # Dates and season
    check(p.latency_days >= 0, "latency_days must be >= 0")
    check(1 <= p.horizon_min <= p.horizon_max, "need 1 <= horizon_min <= horizon_max")
    check(p.onset_lookback_days >= 1, "onset lookback_days must be >= 1")
    try:
        eval_days, train_days = p.evaluation_season.days(), p.training_season.days()
        check(eval_days <= train_days, "training season must cover the evaluation season")
    except ValueError as e:
        errors.append(f"invalid season date: {e}")

    # Label rule
    t = p.target
    check(t.departure_severe_c >= t.departure_heatwave_c, "severe departure below heatwave")
    check(
        t.absolute_severe_c >= t.absolute_heatwave_c >= t.tmax_floor_c,
        "need absolute_severe >= absolute_heatwave >= tmax_floor",
    )
    check(t.round_decimals >= 0, "round_decimals must be >= 0")
    check(t.normal_period[0] <= t.normal_period[1], "normal period is reversed")
    check(t.normal_smoothing in NORMAL_SMOOTHING, f"smoothing must be one of {NORMAL_SMOOTHING}")
    check(t.normal_n_harmonics >= 1, "n_harmonics must be >= 1")
    check(t.tmax_source in TMAX_SOURCES, f"tmax_source must be one of {TMAX_SOURCES}")

    # Blocks: ordered and disjoint
    v = p.validation
    blocks = [
        ("development", v.development),
        ("final_test", v.final_test),
        ("season_2026", v.season_2026),
    ]
    for name, (start, end) in blocks:
        check(start <= end, f"{name} years are reversed")
    for (n1, b1), (n2, b2) in zip(blocks, blocks[1:], strict=False):
        check(b1[1] < b2[0], f"{n1} must end before {n2} starts")

    # Folds: inside development, expanding, contiguous, non-overlapping validation
    check(len(v.folds) >= 1, "at least one fold is required")
    for f in v.folds:
        check(
            f.train[0] == v.development[0],
            f"fold {f.fold}: training must start at development start",
        )
        check(f.train[1] < f.validate[0], f"fold {f.fold}: training must end before validation")
        check(f.validate[0] <= f.validate[1], f"fold {f.fold}: validation years are reversed")
        check(f.validate[1] <= v.development[1], f"fold {f.fold} leaves the development block")
        train_len = f.train[1] - f.train[0] + 1
        check(v.inner_block_seasons < train_len, f"fold {f.fold}: inner block >= training span")
    for prev, cur in zip(v.folds, v.folds[1:], strict=False):
        check(
            cur.validate[0] == prev.validate[1] + 1, f"fold {cur.fold}: validation not contiguous"
        )
        check(cur.train[1] >= prev.train[1], f"fold {cur.fold}: training window shrank")
    if v.folds:
        check(v.folds[-1].validate[1] == v.development[1], "last fold must end at development end")
    check(v.bootstrap_unit == "year", "bootstrap_unit must be 'year' (rows are not independent)")

    # Tiers
    for tier in p.tiers:
        if tier.rule == "min_pod":
            check(
                0 < tier.params.get("value", -1) < 1, f"{tier.name}: POD target must be in (0, 1)"
            )
        elif tier.rule == "max_far":
            check(
                0 < tier.params.get("value", -1) < 1, f"{tier.name}: FAR target must be in (0, 1)"
            )
        elif tier.rule == "min_cost":
            check(
                tier.params.get("miss_to_false_alarm", 0) > 0,
                f"{tier.name}: cost ratio must be > 0",
            )
        else:
            errors.append(f"{tier.name}: unknown tier rule {tier.rule!r}")

    if errors:
        raise ConfigError("invalid problem.yaml:\n  - " + "\n  - ".join(errors))
