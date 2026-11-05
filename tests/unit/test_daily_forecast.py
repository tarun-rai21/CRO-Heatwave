"""Dates, the late-data rule, the failure policy, and the shape of a day's forecast."""

import numpy as np
import pandas as pd
import pytest

from heatwave.config import load_problem
from heatwave.pipeline import daily
from tests.unit.test_freeze import FEATURES, frozen_from, rows

P = load_problem()
L, H = P.latency_days, P.horizons
S = pd.Timestamp("2027-05-10")


def plan(newest="2027-05-07", **kw):
    return daily.make_plan(S, newest, L, H, **kw)


def test_on_time_uses_all_leads_counted_from_the_issue_date():
    p = plan()
    assert (p.status, p.asof_date) == ("ok", pd.Timestamp("2027-05-07"))
    assert p.model_leads == tuple(range(1, 11))
    assert p.valid_dates[0] == S + pd.Timedelta(days=1)
    assert p.valid_dates[-1] == S + pd.Timedelta(days=10)


def test_fresher_data_than_trained_is_not_used():
    p = plan("2027-05-09")
    assert (p.status, p.asof_date) == ("ok", pd.Timestamp("2027-05-07"))


def test_one_day_late_keeps_the_model_as_trained_and_publishes_nine_days():
    p = plan("2027-05-06")
    assert (p.status, p.asof_date) == ("late_1", pd.Timestamp("2027-05-06"))
    assert p.model_leads == tuple(range(2, 11))
    assert p.valid_dates[0] == S + pd.Timedelta(days=1)
    assert p.valid_dates[-1] == S + pd.Timedelta(days=9)
    assert p.data_age_days == 4


def test_two_days_late_publishes_eight_days():
    p = plan("2027-05-05")
    assert p.status == "late_2"
    assert p.model_leads == tuple(range(3, 11))
    assert p.valid_dates[-1] == S + pd.Timedelta(days=8)


@pytest.mark.parametrize(
    "newest,kw,why",
    [
        ("2027-05-04", {}, "data 3 days late"),
        (None, {}, "no observed data"),
        ("2027-05-07", {"validation_ok": False, "problem": "bad"}, "validation failed: bad"),
    ],
)
def test_otherwise_degraded(newest, kw, why):
    p = plan(newest, **kw)
    assert (p.status, p.asof_date, p.reason) == ("degraded", None, why)
    assert len(p.valid_dates) == 10


class Clim:
    def frequency(self, ids, dates):
        return np.full(len(ids), 0.05)


def setup(tmp_path, issue=S, newest="2027-05-07"):
    train = rows(8000)
    train["lead"] = train.index % 10 + 1
    frozen, *_ = frozen_from(tmp_path, train)
    ids = ["a", "b"]
    p = daily.make_plan(issue, newest, L, H)
    feats = None
    if p.asof_date is not None:
        feats = pd.DataFrame({f: [0.5, -0.5] for f in FEATURES})
        feats["district_id"], feats["asof_date"] = ids, p.asof_date
    return p, ids, feats, frozen


def test_a_day_has_every_lead_and_a_window_for_every_district(tmp_path):
    p, ids, feats, frozen = setup(tmp_path)
    out = daily.score(p, ids, feats, Clim(), frozen, P, "lbl")
    assert len(out) == 2 * (10 + 1)
    assert not out.duplicated(daily.KEY).any()
    assert out[["p", "p_clim"]].notna().all().all()
    lead = out[out["target"] == "lead"]
    assert (lead["valid_date"] > S).all() and (lead["tier"] == "none").all()
    w = out[out["target"] == "window"]
    assert (w["valid_date"] == S + pd.Timedelta(days=1)).all()
    assert (w["window_end"] == S + pd.Timedelta(days=10)).all()
    assert w["in_season"].all()


def test_tiers_are_not_issued_outside_the_evaluated_season(tmp_path):
    # In the training window, so a probability exists, but before the evaluated season.
    p, ids, feats, frozen = setup(tmp_path, pd.Timestamp("2027-02-20"), "2027-02-17")
    feats["tmax_anom"] = 50.0  # would reach a tier if it were in season
    out = daily.score(p, ids, feats, Clim(), frozen, P, "lbl")
    w = out[out["target"] == "window"]
    assert (w["p"] > 0.4).all()
    assert (w["tier"] == "none").all() and not w["in_season"].any()


@pytest.mark.parametrize("newest", ["2027-01-07", "2026-12-20"])
def test_outside_the_training_window_there_is_no_probability(tmp_path, newest):
    # Second case: degraded as well, where the window climatology would otherwise guess.
    p, ids, feats, frozen = setup(tmp_path, pd.Timestamp("2027-01-10"), newest)
    out = daily.score(p, ids, feats, Clim(), frozen, P, "lbl")
    assert out["p"].isna().all() and out["p_clim"].isna().all()
    assert (out["tier"] == "none").all()
    assert len(out) == 2 * 11


def test_only_the_leads_outside_the_training_window_are_blank(tmp_path):
    p, ids, feats, frozen = setup(tmp_path, pd.Timestamp("2027-08-10"), "2027-08-07")
    lead = daily.score(p, ids, feats, Clim(), frozen, P, "lbl").query("target == 'lead'")
    inside = lead["valid_date"] <= "2027-08-15"
    assert lead.loc[inside, "p"].notna().all() and inside.any()
    assert lead.loc[~inside, "p"].isna().all() and (~inside).any()


def test_tiers_are_issued_in_season(tmp_path):
    p, ids, feats, frozen = setup(tmp_path)
    feats["tmax_anom"] = 50.0
    w = daily.score(p, ids, feats, Clim(), frozen, P, "lbl").query("target == 'window'")
    assert (w["tier"] == "warning").all()


def test_a_window_touching_the_season_end_is_out_of_season(tmp_path):
    p, ids, feats, frozen = setup(tmp_path, pd.Timestamp("2027-07-25"), "2027-07-22")
    feats["tmax_anom"] = 50.0
    w = daily.score(p, ids, feats, Clim(), frozen, P, "lbl").query("target == 'window'")
    assert (w["tier"] == "none").all()


def test_degraded_is_climatology_with_no_tier_and_never_yesterdays_forecast(tmp_path):
    p, ids, _, frozen = setup(tmp_path, newest="2027-05-01")
    out = daily.score(p, ids, None, Clim(), frozen, P, "lbl")
    assert (out["status"] == "degraded").all() and (out["tier"] == "none").all()
    assert (out["p"] == out["p_clim"]).all()
    assert out["asof_date"].isna().all()
    assert len(out) == 2 * 11


def test_a_missing_district_is_refused(tmp_path):
    p, ids, feats, frozen = setup(tmp_path)
    with pytest.raises(ValueError, match="every district"):
        daily.score(p, ids, feats.iloc[:1], Clim(), frozen, P, "lbl")


def test_same_inputs_give_the_same_file_and_a_rerun_overwrites(tmp_path):
    p, ids, feats, frozen = setup(tmp_path)
    a = daily.score(p, ids, feats, Clim(), frozen, P, "lbl")
    b = daily.score(p, ids, feats.iloc[::-1], Clim(), frozen, P, "lbl")
    pd.testing.assert_frame_equal(a, b)
    path = daily.forecast_path(tmp_path, S)
    daily.write(a, path)
    daily.write(a.assign(p=0.9), path)
    assert (pd.read_parquet(path)["p"] == 0.9).all()
    assert not list(path.parent.glob("*.tmp"))


def test_newest_complete_date_needs_every_district():
    d = pd.DataFrame(
        {
            "district_id": ["a", "b", "a"],
            "obs_date": pd.to_datetime(["2027-05-01", "2027-05-01", "2027-05-02"]),
            "T2M_MAX": [40.0, 41.0, 42.0],
        }
    )
    assert daily.newest_complete_date(d, 2) == pd.Timestamp("2027-05-01")
    assert daily.newest_complete_date(d.iloc[:1], 2) is None
