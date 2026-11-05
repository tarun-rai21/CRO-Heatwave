"""Gate G4: the bar, the verdict, and the guards on the single run."""

import argparse

import numpy as np
import pandas as pd
import pytest

from heatwave import cli
from heatwave.config import load_problem
from heatwave.validation import final_test as ft

RANGES = {k: {"min": 0.05, "max": 0.2} for k in ft.SKILLS}
RANGES |= {k: {"min": 0.4, "max": 0.8} for k in ft.TIER_SCORES}
RANGES["forecast_gap"] = {"min": 0.0, "max": 0.05}
BAR = {"range": RANGES}


def good():
    return {
        **dict.fromkeys(ft.SKILLS, 0.1),
        **dict.fromkeys(ft.TIER_SCORES, 0.6),
        "forecast_gap": 0.01,
    }


def test_inside_every_range_passes():
    assert ft.judge(good(), BAR)["verdict"] == "PASS"


@pytest.mark.parametrize("value", [-0.01, 0.0, 0.03])
def test_a_skill_not_positive_or_below_the_range_fails(value):
    out = ft.judge({**good(), "lead1_onset": value}, BAR)
    assert out["verdict"] == "FAIL" and "lead1_onset" in out["fails"][0]


def test_a_nan_skill_fails():
    assert ft.judge({**good(), "window_all": float("nan")}, BAR)["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "change",
    [{"window_all": 0.3}, {"alert_pod": 0.2}, {"watch_far": 0.9}, {"forecast_gap": 0.09}],
)
def test_other_departures_are_a_concern_not_a_failure(change):
    assert ft.judge({**good(), **change}, BAR)["verdict"] == "PASS WITH CONCERN"


def test_a_failure_outranks_a_concern():
    assert ft.judge({**good(), "window_all": 0.01, "alert_pod": 0.2}, BAR)["verdict"] == "FAIL"


def test_the_bar_is_the_range_over_five_year_periods():
    rng = np.random.default_rng(0)
    parts = []
    for y in range(2006, 2021):
        n = 3000
        p = rng.uniform(0.01, 0.5, n)
        parts.append(
            pd.DataFrame(
                {
                    "year": y,
                    "fold": (y - 2006) // 3 + 1,
                    "p_M2S": p,
                    "p_M0": 0.2,
                    "y": (rng.random(n) < p).astype(float),
                    "onset": rng.random(n) < 0.8,
                }
            )
        )
    window = pd.concat(parts, ignore_index=True)
    lead = window.assign(lead=1)
    spec = {
        "version": "t",
        "window": {"calibrator": {"slope": 1.0, "intercept": 0.0}},
        "tiers": {
            "watch": {"threshold": 0.1},
            "alert": {"threshold": 0.3},
            "warning": {"threshold": 0.4},
        },
    }
    bar = ft.development_bar(spec, lead, window)
    assert len(bar["periods"]) == 11
    assert (
        bar["periods"][0]["period"] == "2006-2010" and bar["periods"][-1]["period"] == "2016-2020"
    )
    seen = [p["window_all"] for p in bar["periods"]]
    assert bar["range"]["window_all"] == {"min": min(seen), "max": max(seen)}


def args(**kw):
    base = {"write_bar": False, "rehearse": None, "run_once": False, "rescore_info": None}
    return argparse.Namespace(**{**base, **kw})


def test_without_a_flag_nothing_runs(capsys, monkeypatch):
    monkeypatch.setattr(ft, "load_rows", lambda *a: pytest.fail("data was read"))
    cli.cmd_final_test(args())
    assert "--run-once" in capsys.readouterr().out


def test_a_second_real_run_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "FINAL_TEST_REPORT", tmp_path / "final_test.md")
    monkeypatch.setattr(ft, "load_rows", lambda *a: pytest.fail("data was read"))
    (tmp_path / "final_test.md").write_text("done")
    with pytest.raises(SystemExit, match="only once"):
        cli.cmd_final_test(args(run_once=True))


def test_an_interrupted_real_run_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "FINAL_TEST_REPORT", tmp_path / "final_test.md")
    monkeypatch.setattr(ft, "load_rows", lambda *a: pytest.fail("data was read"))
    (tmp_path / "final_test.started").write_text("x")
    with pytest.raises(SystemExit, match="started"):
        cli.cmd_final_test(args(run_once=True))


SEASON_BAR = {
    "range": {
        **{k: {"min": -0.2, "max": 0.3} for k in ft.SKILLS},
        **{k: {"min": 0.4, "max": 0.8} for k in ft.TIER_SCORES},
        "forecast_gap": {"min": 0.0, "max": 0.05},
    },
    "middle_half": {"window_all": {"low": -0.07, "high": 0.11}},
}


def test_season_negative_skill_inside_the_range_is_not_a_failure():
    out = ft.judge_season({**good(), "window_all": -0.1}, SEASON_BAR)
    assert out["verdict"] == "PASS"


@pytest.mark.parametrize("value", [-0.21, float("nan")])
def test_season_skill_below_every_year_or_nan_fails(value):
    out = ft.judge_season({**good(), "lead1_onset": value}, SEASON_BAR)
    assert out["verdict"] == "FAIL" and "lead1_onset" in out["fails"][0]


@pytest.mark.parametrize(
    "change", [{"window_all": 0.4}, {"alert_far": 0.9}, {"forecast_gap": 0.06}]
)
def test_season_other_departures_are_a_concern(change):
    assert ft.judge_season({**good(), **change}, SEASON_BAR)["verdict"] == "PASS WITH CONCERN"


def test_season_note_says_a_pass_does_not_confirm_skill():
    note = ft.season_note("PASS", SEASON_BAR)
    assert "does not confirm skill" in note and "-0.070 to 0.110" in note


def test_season_bar_is_the_range_over_single_years():
    rng = np.random.default_rng(1)
    parts = []
    for y in range(2006, 2021):
        p = rng.uniform(0.01, 0.5, 3000)
        parts.append(
            pd.DataFrame(
                {
                    "year": y,
                    "fold": (y - 2006) // 3 + 1,
                    "p_M2S": p,
                    "p_M0": 0.2,
                    "y": (rng.random(3000) < p).astype(float),
                    "onset": rng.random(3000) < 0.8,
                }
            )
        )
    window = pd.concat(parts, ignore_index=True)
    spec = {
        "version": "t",
        "window": {"calibrator": {"slope": 1.0, "intercept": 0.0}},
        "tiers": {
            "watch": {"threshold": 0.1},
            "alert": {"threshold": 0.3},
            "warning": {"threshold": 0.4},
        },
    }
    bar = ft.season_bar(spec, window.assign(lead=1), window, "p_M2S")
    seen = [y["window_all"] for y in bar["years"]]
    assert len(seen) == 15
    assert bar["range"]["window_all"] == {"min": min(seen), "max": max(seen)}
    assert min(seen) <= bar["middle_half"]["window_all"]["low"] <= max(seen)


def test_a_complete_season_has_the_rows_seen_in_the_final_test():
    problem = load_problem()
    assert ft.expected_rows(problem, 2026) == (114750, 10800)
    assert ft.expected_rows(problem, 2024) == (114750, 10800)
    assert ft.season_end(problem, 2026) == pd.Timestamp("2026-07-31")


def versions(rows):
    return pd.DataFrame(rows, columns=["cell_id", "obs_date", "value"]).assign(
        obs_date=lambda d: pd.to_datetime(d["obs_date"])
    )


REVISED = versions([("a", "2026-09-01", 30.0), ("a", "2026-09-01", 31.0)])
SPEC = {"label_version": "l", "feature_version": "f"}
END = pd.Timestamp("2026-07-31")


def test_season_data_ready(tmp_path):
    (tmp_path / "2026.GRD").write_text("x")
    assert ft.season_data_problems(tmp_path / "2026.GRD", REVISED, END, ["l"], ["f"], SPEC) == []


def test_season_needs_the_imd_grid(tmp_path):
    out = ft.season_data_problems(tmp_path / "2026.GRD", REVISED, END, ["l"], ["f"], SPEC)
    assert len(out) == 1 and "IMD" in out[0]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [("a", "2026-09-01", 30.0)],  # not revised
        [("a", "2026-05-01", 30.0), ("a", "2026-05-01", 31.0)],  # inside the season only
    ],
)
def test_season_needs_a_revision_after_it_ends(tmp_path, rows):
    (tmp_path / "2026.GRD").write_text("x")
    out = ft.season_data_problems(tmp_path / "2026.GRD", versions(rows), END, ["l"], ["f"], SPEC)
    assert len(out) == 1 and "POWER" in out[0]


def test_season_tables_must_match_the_model(tmp_path):
    (tmp_path / "2026.GRD").write_text("x")
    out = ft.season_data_problems(tmp_path / "2026.GRD", REVISED, END, ["x"], ["y"], SPEC)
    assert len(out) == 2


def season_args(**kw):
    return argparse.Namespace(**{"write_bar": False, "rehearse": None, "run_once": False, **kw})


def test_season_check_without_a_flag_reads_nothing(capsys, monkeypatch):
    monkeypatch.setattr(ft, "load_rows", lambda *a: pytest.fail("data was read"))
    cli.cmd_season_check(season_args())
    assert "--run-once" in capsys.readouterr().out


def test_a_second_season_check_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "SEASON_CHECK_REPORT", tmp_path / "season.md")
    monkeypatch.setattr(ft, "load_rows", lambda *a: pytest.fail("data was read"))
    (tmp_path / "season.md").write_text("done")
    with pytest.raises(SystemExit, match="only once"):
        cli.cmd_season_check(season_args(run_once=True))


def test_an_interrupted_season_check_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "SEASON_CHECK_REPORT", tmp_path / "season.md")
    monkeypatch.setattr(ft, "load_rows", lambda *a: pytest.fail("data was read"))
    (tmp_path / "season.started").write_text("x")
    with pytest.raises(SystemExit, match="started"):
        cli.cmd_season_check(season_args(run_once=True))


def test_rehearsal_cannot_touch_final_test_years():
    with pytest.raises(SystemExit, match="development"):
        cli.check_rehearsal_years((2016, 2021), (1981, 2020))
    cli.check_rehearsal_years((2016, 2020), (1981, 2020))
