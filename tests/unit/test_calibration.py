"""Calibration and tier thresholds."""

import numpy as np
import pandas as pd
import pytest

from heatwave.validation import calibration as c

RULES = {
    "watch": {"value": 0.8},
    "alert": {"miss_to_false_alarm": 3.0},
    "warning": {"value": 0.5},
}


def data(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.beta(1, 6, n)
    y = (rng.random(n) < true).astype(float)
    return pd.DataFrame(
        {
            "true": true,
            "p": 1 / (1 + (1 - true) / true * 3),  # systematically too low
            "y": y,
            "fold": rng.integers(1, 6, n),
            "year": rng.integers(2006, 2021, n),
            "lead": rng.integers(1, 3, n),
        }
    )


@pytest.mark.parametrize("method", ["platt", "beta"])
def test_calibration_repairs_a_biased_forecast(method):
    d = data()
    fixed = c.apply(c.fit(d.p.to_numpy(), d.y.to_numpy(), method), d.p.to_numpy(), method)
    assert abs(fixed.mean() - d.y.mean()) < 0.001
    assert ((fixed - d.y) ** 2).mean() < ((d.p - d.y) ** 2).mean()


def test_cross_fit_never_uses_the_held_fold():
    d = data()
    base = c.cross_fit(d, "p", "lead")
    changed = d.copy()
    held = changed["fold"] == 1
    changed.loc[held, "y"] = 1 - changed.loc[held, "y"]  # fold 1's outcomes alone change
    again = c.cross_fit(changed, "p", "lead")
    assert np.allclose(base[held], again[held])
    assert not np.allclose(base[~held], again[~held])


def test_thresholds_follow_the_rules():
    d = data()
    p, y, yr = d.true.to_numpy(), d.y.to_numpy(), d.year.to_numpy()
    t = c.thresholds(p, y, yr, RULES, 500, 3)
    assert c.scores(p, y, t["watch"])["pod"] >= 0.8
    assert c.scores(p, y, t["warning"])["far"] <= 0.5
    assert t["watch"] < t["alert"] < t["warning"]
    # A calibrated forecast's cost-minimising threshold is 1 / (1 + ratio).
    assert t["alert"] == pytest.approx(0.25, abs=0.05)


def test_warning_needs_enough_rows_and_seasons():
    d = data(5000)
    p, y, yr = d.true.to_numpy(), d.y.to_numpy(), d.year.to_numpy()
    assert np.isnan(c.thresholds(p, y, yr, RULES, 10**6, 3)["warning"])
    assert np.isnan(c.thresholds(p, y, np.full(len(p), 2010), RULES, 10, 3)["warning"])


def test_no_alert_when_flagging_nothing_is_cheapest():
    p = np.full(1000, 0.01)
    y = np.zeros(1000)
    y[:5] = 1
    t = c.thresholds(p, y, np.arange(1000) % 15, RULES, 10, 3)
    assert np.isnan(t["alert"]) and np.isnan(t["warning"])


def test_tier_table_reports_every_group_and_tier():
    t = c.tier_table(data(), "true", "lead", RULES, 100, 3)
    assert len(t) == 6
    assert set(t["tier"]) == {"watch", "alert", "warning"}
