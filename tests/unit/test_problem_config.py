"""configs/problem.yaml must load, match reference.md, and reject invalid edits."""

import copy

import pytest
import yaml

from heatwave.config import CONFIG_DIR, ConfigError, DayWindow, load_problem

P = load_problem()
RAW = yaml.safe_load((CONFIG_DIR / "problem.yaml").read_text(encoding="utf-8"))


# --- The config says what reference.md says --------------------------------
# If one of these fails, either the config or reference.md changed without the other.


def test_dates_match_reference_1_3():
    assert P.latency_days == 3
    assert list(P.horizons) == list(range(1, 11))


def test_seasons_match_reference_3_1_and_4_3():
    assert P.evaluation_season == DayWindow("03-01", "06-30")
    assert P.training_season == DayWindow("02-15", "07-15")


def test_label_rule_matches_reference_2_4():
    t = P.target
    assert (t.tmax_floor_c, t.departure_heatwave_c, t.departure_severe_c) == (40.0, 4.5, 6.4)
    assert (t.absolute_heatwave_c, t.absolute_severe_c) == (45.0, 47.0)
    assert t.round_decimals == 1
    assert t.normal_period == (1991, 2020)
    assert t.tmax_source == "undecided"  # decided by measurement in Phase 4


def test_onset_matches_reference_3_5():
    assert P.onset_lookback_days == 3


def test_splits_match_reference_3_2_and_3_3():
    v = P.validation
    assert (v.development, v.final_test, v.season_2026) == (
        (1981, 2020),
        (2021, 2025),
        (2026, 2026),
    )
    assert [(f.train, f.validate) for f in v.folds] == [
        ((1981, 2005), (2006, 2008)),
        ((1981, 2008), (2009, 2011)),
        ((1981, 2011), (2012, 2014)),
        ((1981, 2014), (2015, 2017)),
        ((1981, 2017), (2018, 2020)),
    ]
    assert v.inner_block_seasons == 3
    assert v.bootstrap_unit == "year"


def test_tiers_match_reference_3_6():
    tiers = {t.name: (t.rule, t.params) for t in P.tiers}
    assert tiers == {
        "watch": ("min_pod", {"value": 0.8}),
        "alert": ("min_cost", {"miss_to_false_alarm": 3.0}),
        "warning": ("max_far", {"value": 0.5}),
    }


# --- Invalid edits are rejected ---------------------------------------------


def load_mutated(tmp_path, mutate):
    raw = copy.deepcopy(RAW)
    mutate(raw)
    (tmp_path / "problem.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_problem(tmp_path)


def set_path(*keys_and_value):
    *keys, value = keys_and_value

    def mutate(raw):
        node = raw
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        # training overlaps its own validation block
        (lambda r: r["validation"]["folds"][0].update(train=[1981, 2006]), "fold 1"),
        # a fold reaches into the final-test years
        (lambda r: r["validation"]["folds"][-1].update(validate=[2018, 2021]), "development block"),
        # gap between validation blocks
        (lambda r: r["validation"]["folds"][1].update(validate=[2010, 2011]), "not contiguous"),
        (set_path("validation", "final_test", [2020, 2025]), "development must end before"),
        (set_path("validation", "bootstrap_unit", "row"), "bootstrap_unit"),
        (set_path("target", "departure_severe_c", 4.0), "severe departure"),
        (set_path("target", "absolute_heatwave_c", 39.0), "absolute_severe"),
        (set_path("target", "tmax_source", "station"), "tmax_source"),
        (
            set_path("season", "training", {"start": "03-15", "end": "07-15"}),
            "cover the evaluation",
        ),
        (
            set_path("season", "evaluation", {"start": "02-30", "end": "06-30"}),
            "invalid season date",
        ),
        (set_path("dates", "horizon_min", 0), "horizon_min"),
        (set_path("tiers", "watch", {"rule": "min_pod", "value": 1.5}), "POD target"),
        (set_path("tiers", "watch", {"rule": "top_k", "value": 5}), "unknown tier rule"),
        (lambda r: r.pop("onset"), "malformed"),
    ],
)
def test_invalid_definitions_are_rejected(tmp_path, mutate, message):
    with pytest.raises(ConfigError, match=message):
        load_mutated(tmp_path, mutate)


def test_all_errors_are_reported_together(tmp_path):
    def two_errors(raw):
        raw["dates"]["latency_days"] = -1
        raw["validation"]["bootstrap_unit"] = "row"

    with pytest.raises(ConfigError) as exc:
        load_mutated(tmp_path, two_errors)
    assert "latency_days" in str(exc.value) and "bootstrap_unit" in str(exc.value)


def test_day_window_wrapping_the_year_end():
    winter = DayWindow("12-01", "02-29")
    assert (12, 31) in winter.days() and (1, 15) in winter.days() and (3, 1) not in winter.days()
