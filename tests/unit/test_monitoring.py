"""Monitoring checks, the scorecard and the monthly checks."""

import numpy as np
import pandas as pd

from heatwave.monitoring import checks, monthly, scorecard

BASE = {
    "sanity": {"5": {"mean_p_min": 0.1, "mean_p_max": 0.4, "warning_max": 0.5}},
    "features": {
        "5": {k: {"mean": 0.0, "std": 0.5} for k in checks.ANOMALY_FEATURES},
    },
}


def forecast(status="ok", reason="", p=0.2, tier="none", n=75, in_season=True, age=3, lead_p=0.05):
    ids = [f"d{i}" for i in range(n)]
    s = pd.Timestamp("2027-05-10")
    window = pd.DataFrame(
        {"district_id": ids, "target": "window", "p": p, "tier": tier, "in_season": in_season}
    )
    lead = pd.DataFrame(
        {"district_id": ids, "target": "lead", "p": lead_p, "tier": "none", "in_season": in_season}
    )
    return pd.concat([window, lead], ignore_index=True).assign(
        issue_date=s, status=status, reason=reason, data_age_days=age
    )


def levels(cs):
    return {c["check"]: c["level"] for c in cs}


def test_a_normal_day_passes():
    got = levels(checks.forecast_checks(forecast(), BASE))
    assert set(got.values()) == {"ok"}
    assert "window_against_single_days" in got


def test_degraded_is_a_failure_and_late_is_a_warning():
    assert (
        levels(checks.forecast_checks(forecast("degraded", "x"), BASE))["data_timeliness"] == "fail"
    )
    assert (
        levels(checks.forecast_checks(forecast("late_1", "x"), BASE))["data_timeliness"] == "warn"
    )


def test_missing_districts_and_blank_season_forecasts_fail():
    assert levels(checks.forecast_checks(forecast(n=70), BASE))["coverage"] == "fail"
    t = forecast(p=np.nan)
    assert levels(checks.forecast_checks(t, BASE))["probabilities_present"] == "fail"


def test_invalid_probability_fails():
    assert levels(checks.forecast_checks(forecast(p=1.4), BASE))["probabilities_valid"] == "fail"


def test_out_of_history_forecasts_warn():
    got = levels(checks.forecast_checks(forecast(p=0.9, tier="warning"), BASE))
    assert got["mean_window_probability"] == "warn" and got["warning_share"] == "warn"


def test_a_window_far_below_a_single_day_warns():
    got = levels(checks.forecast_checks(forecast(p=0.2, lead_p=0.5), BASE))
    assert got["window_against_single_days"] == "warn"


def test_out_of_season_skips_sanity():
    got = levels(checks.forecast_checks(forecast(p=np.nan, in_season=False), BASE))
    assert got["forecast_sanity"] == "skipped" and "probabilities_present" not in got


def feats(value):
    dates = pd.date_range("2027-04-10", "2027-05-07")
    return pd.DataFrame({"asof_date": dates, **dict.fromkeys(checks.ANOMALY_FEATURES, value)})


def test_feature_drift_flags_a_large_shift_only():
    t = pd.Timestamp("2027-05-07")
    assert set(levels(checks.feature_checks(feats(0.5), t, BASE)).values()) == {"ok"}
    assert set(levels(checks.feature_checks(feats(2.0), t, BASE)).values()) == {"warn"}


def test_feature_drift_skips_without_history_or_forecast():
    assert levels(checks.feature_checks(feats(0), None, BASE))["feature_drift"] == "skipped"
    t = pd.Timestamp("2027-11-07")
    assert levels(checks.feature_checks(feats(0), t, BASE))["feature_drift"] == "skipped"


def test_the_log_keeps_one_row_per_day_and_the_first_time(tmp_path):
    path = tmp_path / "m" / "daily.parquet"
    t1, t2 = pd.Timestamp("2027-05-10 11:00", tz="UTC"), pd.Timestamp("2027-05-10 15:00", tz="UTC")
    checks.append_log(path, checks.log_row(forecast(), [], t1))
    log = checks.append_log(path, checks.log_row(forecast("late_1", "x"), [], t2))
    assert len(log) == 1
    assert log["first_checked_at"].iloc[0] == t1 and log["status"].iloc[0] == "late_1"


def test_baseline_gives_ranges_by_month(tmp_path):
    window = pd.DataFrame(
        {
            "issue_date": pd.to_datetime(["2010-05-01", "2010-05-01", "2011-05-02", "2011-05-02"]),
            "month": 5,
            "p_M2S": [0.1, 0.3, 0.5, 0.7],
        }
    )
    dates = pd.to_datetime(["2005-05-01", "2006-05-01", "2007-05-01", "2025-05-01"])
    feat = pd.DataFrame(
        {"asof_date": dates, **dict.fromkeys(checks.ANOMALY_FEATURES, [0.0, 1.0, 2.0, 99.0])}
    )
    spec = {
        "version": "t",
        "window": {"calibrator": {"slope": 1.0, "intercept": 0.0}},
        "tiers": {"warning": {"threshold": 0.6}},
    }
    b = checks.write_baseline(spec, window, feat, tmp_path / "b.json")
    assert abs(b["sanity"]["5"]["mean_p_min"] - 0.2) < 1e-6
    assert abs(b["sanity"]["5"]["mean_p_max"] - 0.6) < 1e-6
    assert b["sanity"]["5"]["warning_max"] == 0.5
    assert b["features"]["5"]["tmax_anom"]["mean"] == 1.0  # 2025 is outside the training years


def outcomes():
    rows = []
    for day in pd.date_range("2027-05-01", "2027-05-12"):
        for d, p, tier, obs in (
            ("a", 0.5, "warning", 1.0),
            ("b", 0.3, "alert", 0.0),
            ("c", 0.05, "none", 0.0),
        ):
            rows.append(
                ("window", d, day, day + pd.Timedelta(days=1), p, 0.2, tier, obs, True, "ok")
            )
            rows.append(
                ("lead", d, day, day + pd.Timedelta(days=2), p, 0.2, "none", obs, True, "ok")
            )
    cols = [
        "target",
        "district_id",
        "issue_date",
        "valid_date",
        "p",
        "p_clim",
        "tier",
        "observed",
        "in_season",
        "status",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_scorecard_counts_only_known_in_season_model_forecasts():
    o = outcomes()
    extra = o.iloc[:4].copy()
    bad = pd.concat(
        [
            extra.assign(observed=np.nan),
            extra.assign(in_season=False),
            extra.assign(status="degraded"),
            extra.assign(p=np.nan),
        ]
    )
    assert len(scorecard.scored(pd.concat([o, bad]))) == len(o)


def test_scorecard_tiers_and_skill():
    t = scorecard.tables(outcomes())
    tiers = t["tiers"].set_index("tier_or_higher")
    assert tiers.loc["warning", "issued"] == 12 and tiers.loc["warning", "share_true"] == 1.0
    assert tiers.loc["alert", "issued"] == 24 and tiers.loc["alert", "share_true"] == 0.5
    assert tiers.loc["alert", "events_caught"] == 1.0
    assert t["window"]["skill_vs_climatology"].iloc[0] > 0
    assert t["by_lead"]["days_ahead"].tolist() == [2]
    assert t["last_7"]["forecasts"].iloc[0] == 21


def test_scorecard_pages_say_when_there_is_nothing_or_little():
    assert "No forecasts have been scored" in scorecard.body_html(None)
    assert "No forecasts have been scored" in scorecard.body_html(outcomes().iloc[:0])
    page = scorecard.body_html(outcomes())
    assert "12 events so far" in page and "nan" not in page.lower()


def test_skew_compares_first_and_latest_stored_versions(tmp_path):
    from datetime import date

    from heatwave.ingest.raw_store import RawStore

    store = RawStore(tmp_path)
    key = {"cell_id": ["c"], "obs_date": [date(2027, 5, 1)], "variable": ["T2M_MAX"]}
    other = {
        "cell_id": ["d"],
        "obs_date": [date(2027, 5, 1)],
        "variable": ["T2M_MAX"],
        "value": [30.0],
    }
    store.append(
        pd.DataFrame({**key, "value": [40.0]}), "r1", pd.Timestamp("2027-05-04", tz="UTC"), "v"
    )
    store.append(pd.DataFrame(other), "r2", pd.Timestamp("2027-05-05", tz="UTC"), "v")
    store.append(
        pd.DataFrame({**key, "value": [41.5]}), "r3", pd.Timestamp("2027-06-04", tz="UTC"), "v"
    )
    t = monthly.skew(store, [2027]).set_index("variable")
    assert t.loc["T2M_MAX", "values"] == 2 and t.loc["T2M_MAX", "revised"] == 1
    assert abs(t.loc["T2M_MAX", "mean_change"] - 1.5) < 1e-9


def station_data(shift):
    rows, labels = [], []
    for sid, d in (("1", "x"), ("2", "y")):
        for y in (2021, 2022, 2027):
            for day in pd.date_range(f"{y}-05-01", periods=40):
                rows.append((sid, day, 38.0))
                labels.append((d, day, 38.0 + (shift if y == 2027 else 0.0)))
    days = pd.DataFrame(rows, columns=["station_id", "obs_date", "tmax_c"])
    where = pd.DataFrame({"station_id": ["1", "2"], "name": ["A", "B"], "district_id": ["x", "y"]})
    return days, where, pd.DataFrame(labels, columns=["district_id", "obs_date", "tmax_used"])


def test_label_moving_against_stations_is_flagged():
    args = station_data(-1.0)
    out = monthly.label_against_stations(*args, 2027, (2021, 2025), [5, 6])
    assert out["level"] == "warn" and abs(out["mean_change"] + 1.0) < 1e-9
    args = station_data(0.2)
    assert monthly.label_against_stations(*args, 2027, (2021, 2025), [5, 6])["level"] == "ok"


def test_no_season_data_is_skipped():
    args = station_data(0.0)
    assert monthly.label_against_stations(*args, 2028, (2021, 2025), [5, 6])["level"] == "skipped"


def test_a_large_soil_moisture_gap_is_flagged_and_a_small_one_is_not():
    t = pd.DataFrame(
        [
            {"variable": "GWETROOT", "revised": 10, "mean_change": 0.13},
            {"variable": "T2M_MAX", "revised": 10, "mean_change": 0.5},
        ]
    )
    flags = monthly.skew_flags(t)
    assert len(flags) == 1 and "GWETROOT" in flags[0]["detail"]
    assert not monthly.skew_flags(t.assign(mean_change=0.01))
