"""Label rule, normals and bias correction on cases with known answers."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from heatwave.config import load_problem
from heatwave.labels.bias_correction import apply_quantile_map, fit_quantile_map, qmap_version
from heatwave.labels.definition import classify, label_version, window_target
from heatwave.labels.normals import apply_normals, fit_normals, year_fraction

RULE = load_problem().target


def label(tmax, normal):
    out = classify([tmax], [normal], RULE)
    return out["is_heatwave"][0], out["is_severe"][0]


# --- rule ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tmax", "normal", "heatwave", "severe"),
    [
        (39.9, 30.0, False, False),  # below the 40 C floor
        (40.0, 35.5, True, False),  # floor met, departure 4.5
        (40.4, 36.0, False, False),  # departure 4.4
        (40.5, 36.0, True, False),  # departure 4.5
        (42.4, 36.0, True, False),  # departure 6.4: heatwave, not severe
        (42.5, 36.0, True, True),  # departure 6.5: severe (> 6.4)
        (44.9, 43.0, False, False),  # absolute 44.9, small departure
        (45.0, 43.0, True, False),  # absolute 45
        (46.9, 45.0, True, False),
        (47.0, 45.0, True, True),  # absolute 47: severe
    ],
)
def test_threshold_edges(tmax, normal, heatwave, severe):
    assert label(tmax, normal) == (heatwave, severe)


def test_float_noise_cannot_flip_a_threshold():
    assert label(40.1, 35.6) == (True, False)  # 40.1 - 35.6 = 4.4999... in floats


def test_missing_input_gives_missing_label():
    out = classify([np.nan, 41.0], [36.0, np.nan], RULE)
    assert out["is_heatwave"].isna().all() and out["is_severe"].isna().all()


def test_severe_implies_heatwave():
    rng = np.random.default_rng(0)
    out = classify(rng.uniform(30, 50, 5000), rng.uniform(30, 44, 5000), RULE)
    assert not (out["is_severe"] & ~out["is_heatwave"]).any()


def test_window_target():
    w = window_target(np.array([[0, 0, 0], [0, 1, 0], [0, np.nan, 0], [1, np.nan, 0]]))
    assert list(w[:2]) == [False, True] and pd.isna(w[2]) and w[3]


def test_label_version_tracks_rule_and_extras():
    base = label_version(RULE, {"qmap": "qm_a"})
    assert base == label_version(RULE, {"qmap": "qm_a"})
    assert label_version(replace(RULE, departure_heatwave_c=4.6), {"qmap": "qm_a"}) != base
    assert label_version(RULE, {"qmap": "qm_b"}) != base


# --- normals -------------------------------------------------------------------


def synthetic(days="1990-01-01", periods=365 * 32, unit="d1"):
    dates = pd.date_range(days, periods=periods, freq="D")
    t = 2 * np.pi * year_fraction(pd.Series(dates))
    true = 32 + 8 * np.sin(t - 1.6) + 2 * np.cos(2 * t) + 0.7 * np.sin(5 * t)
    return pd.DataFrame({"unit": unit, "obs_date": dates, "tmax": true}), true


def test_normal_recovers_a_known_annual_cycle():
    df, true = synthetic()
    coefs = fit_normals(df, "unit", "tmax", (1991, 2020), n_harmonics=6)
    assert np.abs(apply_normals(df, "unit", coefs) - true).max() < 1e-8
    assert coefs[["fit_start", "fit_end"]].iloc[0].tolist() == [1991, 2020]


def test_normal_ignores_years_outside_the_period_and_nulls():
    df, _ = synthetic()
    shifted = df.copy()
    shifted.loc[shifted.obs_date.dt.year < 1991, "tmax"] += 50  # must not matter
    shifted.loc[::7, "tmax"] = np.nan
    a = fit_normals(df, "unit", "tmax", (1991, 2020), 6)
    b = fit_normals(shifted, "unit", "tmax", (1991, 2020), 6)
    assert np.allclose(a.filter(like="c").to_numpy(), b.filter(like="c").to_numpy())


def test_leap_day_lies_between_its_neighbours():
    df, _ = synthetic()
    coefs = fit_normals(df, "unit", "tmax", (1991, 2020), 6)
    q = pd.DataFrame(
        {"unit": "d1", "obs_date": pd.to_datetime(["2024-02-28", "2024-02-29", "2024-03-01"])}
    )
    n = apply_normals(q, "unit", coefs)
    assert min(n[0], n[2]) <= n[1] <= max(n[0], n[2])


def test_missing_unit_is_an_error():
    df, _ = synthetic()
    coefs = fit_normals(df, "unit", "tmax", (1991, 2020), 6)
    with pytest.raises(ValueError, match="no normal"):
        apply_normals(df.assign(unit="other"), "unit", coefs)


# --- quantile map --------------------------------------------------------------


def pairs(bias, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(rng.choice(pd.date_range("1991-01-01", "2020-12-31"), n))
    target = rng.normal(35, 4, n)
    return pd.DataFrame({"obs_date": dates, "target": target, "source": bias(target)})


def test_constant_bias_is_removed():
    qm = fit_quantile_map(pairs(lambda t: t + 2.5), (1991, 2020))
    x = np.array([30.0, 35.0, 40.0])
    assert np.allclose(apply_quantile_map(x, [5, 5, 5], qm), x - 2.5, atol=0.2)


def test_level_dependent_bias_and_tail_shift():
    qm = fit_quantile_map(pairs(lambda t: t + 0.1 * (t - 35)), (1991, 2020))
    hot = apply_quantile_map([60.0], [5], qm)[0]  # far beyond the fitted range
    top = qm[(qm.month == 5) & (qm.q == 0.99)]
    assert hot == pytest.approx(60.0 + (top.target_q.iloc[0] - top.source_q.iloc[0]))


def test_mapping_is_monotonic():
    qm = fit_quantile_map(pairs(lambda t: t + 2 + 0.05 * (t - 35)), (1991, 2020))
    x = np.linspace(20, 50, 300)
    assert np.all(np.diff(apply_quantile_map(x, np.full(300, 6), qm)) >= -1e-9)


def test_pairs_outside_the_period_are_ignored():
    p = pairs(lambda t: t + 2.0)
    outside = p.assign(obs_date=p.obs_date + pd.DateOffset(years=40), source=p.source + 10)
    a = fit_quantile_map(p, (1991, 2020))
    b = fit_quantile_map(pd.concat([p, outside]), (1991, 2020))
    assert qmap_version(a) == qmap_version(b)
