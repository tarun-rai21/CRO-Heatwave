"""Validation harness: metrics, year-block bootstrap, baselines, fences."""

import numpy as np
import pandas as pd
import pytest

from heatwave.config import Fold, load_problem, read_yaml
from heatwave.models.baselines import M0Lead, M0Window, M1Lead, M2Lead
from heatwave.validation import metrics, walk_forward
from heatwave.validation.report import skill_horizon

CFG = read_yaml("models.yaml")


def oof(n_years=10, per_year=400, seed=0):
    rng = np.random.default_rng(seed)
    years = np.repeat(np.arange(2006, 2006 + n_years), per_year)
    clim = rng.uniform(0.01, 0.1, len(years))
    y = (rng.random(len(years)) < clim * 1.5).astype(float)
    good = np.clip(clim * 1.5 + 0.3 * (y - clim), 0, 1)  # informative forecast
    return pd.DataFrame(
        {
            "year": years,
            "lead": rng.integers(1, 4, len(years)),
            "y": y,
            "p_M0": clim,
            "p_GOOD": good,
            "p_SAME": clim,
        }
    )


def test_bss_point_estimate_matches_the_direct_formula():
    d = oof()
    t = metrics.bss_table(d, ["GOOD"], "M0", None, 200, 1, 0.95)
    direct = 1 - ((d.p_GOOD - d.y) ** 2).sum() / ((d.p_M0 - d.y) ** 2).sum()
    assert t.loc[t.model == "GOOD", "bss"].iloc[0] == pytest.approx(direct)


def test_interval_contains_the_estimate_and_is_reproducible():
    d = oof()
    a = metrics.bss_table(d, ["GOOD"], "M0", "lead", 500, 7, 0.95)
    b = metrics.bss_table(d, ["GOOD"], "M0", "lead", 500, 7, 0.95)
    pd.testing.assert_frame_equal(a, b)
    assert ((a.lo <= a.bss) & (a.bss <= a.hi)).all()


def test_paired_difference_of_a_model_with_itself_is_exactly_zero():
    d = oof()
    t = metrics.bss_table(d, ["GOOD"], "M0", None, 300, 3, 0.95, pairs=[("GOOD", "GOOD")])
    r = t[t.model == "GOOD-GOOD"].iloc[0]
    assert (r.bss, r.lo, r.hi) == (0.0, 0.0, 0.0)


def test_identical_model_has_zero_skill():
    t = metrics.bss_table(oof(), ["SAME"], "M0", None, 100, 1, 0.95)
    assert t.bss.iloc[0] == pytest.approx(0.0)


def test_bootstrap_resamples_whole_years():
    # One wildly different year moves the interval much more than the point estimate.
    d = oof()
    d.loc[d.year == 2006, "p_GOOD"] = 1 - d.loc[d.year == 2006, "y"]  # a disastrous year
    t = metrics.bss_table(d, ["GOOD"], "M0", None, 1000, 2, 0.95)
    assert t.hi.iloc[0] - t.lo.iloc[0] > 0.1


def test_reliability_and_decomposition():
    d = oof()
    r = metrics.reliability(d, "GOOD", 5)
    assert r.rows.sum() == len(d)
    dec = metrics.brier_decomposition(d, "GOOD", 10)
    assert dec["uncertainty"] == pytest.approx(d.y.mean() * (1 - d.y.mean()))


# --- baselines ----------------------------------------------------------------


def lead_rows(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2000-03-01") + pd.to_timedelta(rng.integers(0, 150, n), unit="D")
    anom = rng.normal(0, 2, n)
    clim = np.full(n, 0.03)
    p = 1 / (1 + np.exp(-(-3.5 + 1.2 * anom)))
    return pd.DataFrame(
        {
            "district_id": rng.choice(["a", "b"], n),
            "lead": rng.integers(1, 3, n),
            "issue_date": dates,
            "valid_date": dates + pd.Timedelta(days=1),
            "hw_today": (rng.random(n) < 0.05).astype(float),
            "tmax_anom": anom,
            "tmax_anom_mean7": anom * 0.8 + rng.normal(0, 0.5, n),
            "gwetroot_anom": rng.normal(0, 1, n),
            "doy_sin": np.sin(dates.dayofyear / 58),
            "doy_cos": np.cos(dates.dayofyear / 58),
            "clim_hw_freq_valid": clim,
            "y": (rng.random(n) < p).astype(float),
        }
    )


def test_m0_is_the_climatology_column():
    r = lead_rows()
    assert np.array_equal(M0Lead().fit(r).predict(r), r.clim_hw_freq_valid.to_numpy())


def test_m1_shrinks_toward_climatology():
    r = lead_rows()
    m = M1Lead(CFG).fit(r)
    p = m.predict(r)
    assert ((p >= 0) & (p <= 1)).all()
    hot = r.tmax_anom > np.quantile(r.tmax_anom, 0.9)
    assert p[hot].mean() > p[~hot].mean()  # learned the anomaly effect


def test_m2_learns_the_anomaly_effect_per_lead():
    r = lead_rows()
    m = M2Lead(CFG).fit(r)
    c = m.coefficients()
    assert set(c.lead) == {1, 2} and (c.tmax_anom > 0).all()
    assert 1 - ((m.predict(r) - r.y) ** 2).mean() / ((r.clim_hw_freq_valid - r.y) ** 2).mean() > 0


def test_m0_window_is_a_smoothed_frequency():
    r = lead_rows().drop(columns=["valid_date", "lead"])
    m = M0Window(CFG).fit(r)
    p = m.predict(r)
    assert abs(p.mean() - r.y.mean()) < 0.01


# --- fences and summaries -------------------------------------------------------


def test_overlapping_fold_is_refused_before_anything_is_read(tmp_path):
    bad = Fold(fold=9, train=(1981, 2010), validate=(2010, 2012))
    with pytest.raises(ValueError, match="overlap"):
        walk_forward.fold_tables(load_problem(), bad, tmp_path, pd.DataFrame(), "lead", [])


def test_skill_horizon():
    t = pd.DataFrame({"model": "M2", "group": [1, 2, 3, 4], "lo": [0.02, 0.01, -0.01, 0.01]})
    assert skill_horizon(t, "M2") == 2
    assert skill_horizon(t.assign(lo=-0.1), "M2") is None
    assert skill_horizon(t.assign(lo=0.1), "M2") == 4


def test_regions_cover_every_district_once():
    from heatwave.config import REPO_ROOT

    r = walk_forward.regions(REPO_ROOT / "configs" / "regions.yaml")
    assert len(r) == 75
    assert set(r.values()) == {"bundelkhand", "central", "eastern", "western"}
