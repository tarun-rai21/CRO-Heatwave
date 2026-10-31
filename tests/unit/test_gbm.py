"""LightGBM wrapper: climatology base margin, inner-block early stopping, G3 rule."""

import numpy as np
import pandas as pd

from heatwave.config import read_yaml
from heatwave.models.gbm import M3
from heatwave.validation.compare import g3

CFG = read_yaml("models.yaml")
CFG = {
    **CFG,
    "m3": {
        **CFG["m3"],
        "params": {**CFG["m3"]["params"], "num_threads": 2},
        "max_rounds": 200,
        "early_stopping_rounds": 20,
    },
}


def rows(n_per_year=3000, years=range(1990, 2002), seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for y in years:
        dates = pd.to_datetime(f"{y}-04-01") + pd.to_timedelta(rng.integers(0, 90, n_per_year), "D")
        anom = rng.normal(0, 2, n_per_year)
        clim = rng.uniform(0.01, 0.08, n_per_year)
        logit = np.log(clim / (1 - clim)) + 0.9 * anom
        parts.append(
            pd.DataFrame(
                {
                    "issue_date": dates,
                    "valid_date": dates + pd.Timedelta(days=2),
                    "lead": 1,
                    "tmax_anom": anom,
                    "noise": rng.normal(size=n_per_year),
                    "clim_hw_freq_valid": clim,
                    "y": (rng.random(n_per_year) < 1 / (1 + np.exp(-logit))).astype(float),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def model(**kw):
    return M3(CFG, "lead", 3, features=["tmax_anom", "noise", "lead", "clim_hw_freq_valid"], **kw)


def test_learns_the_signal_and_beats_climatology():
    tr, va = rows(), rows(seed=1, years=range(2002, 2005))
    m = model().fit(tr)
    p = m.predict(va)
    bs = lambda q: ((q - va.y) ** 2).mean()  # noqa: E731
    assert bs(p) < bs(va.clim_hw_freq_valid)
    assert m.importance().iloc[0].feature == "tmax_anom"
    assert 1 <= m.best_rounds < 200


def test_zero_rounds_of_learning_returns_climatology():
    # With a huge learning-rate penalty the trees add nothing: the prediction is the margin.
    tr = rows()
    m = model(params={"learning_rate": 1e-9}).fit(tr)
    assert np.allclose(m.predict(tr), tr.clim_hw_freq_valid, atol=1e-6)


def test_seed_makes_fits_reproducible():
    tr = rows()
    a, b = model(seed=5).fit(tr), model(seed=5).fit(tr)
    assert np.array_equal(a.predict(tr), b.predict(tr))


def test_monotone_constraint_is_respected():
    tr = rows()
    m = M3(
        CFG,
        "lead",
        3,
        features=["tmax_anom", "noise", "lead", "clim_hw_freq_valid"],
        use_monotone=True,
    ).fit(tr)
    grid = tr.iloc[:200].copy()
    for a, b in [(-3, -1), (-1, 1), (1, 3)]:
        lo, hi = grid.assign(tmax_anom=a), grid.assign(tmax_anom=b)
        assert (m.predict(hi) >= m.predict(lo) - 1e-12).all()


def test_g3_rule():
    rng = np.random.default_rng(0)
    n = 6000
    d = pd.DataFrame({"year": np.repeat(np.arange(2006, 2021), n // 15), "lead": 1, "onset": True})
    d["p_M0"] = 0.05
    d["y"] = (rng.random(len(d)) < 0.05 + 0.1 * (rng.random(len(d)) < 0.3)).astype(float)
    d["p_M2"] = np.clip(d.p_M0 + 0.2 * (d.y - 0.05), 0, 1)
    d["p_M3"] = np.clip(d.p_M0 + 0.5 * (d.y - 0.05), 0, 1)  # clearly better
    cfg = {"evaluation": {**CFG["evaluation"], "bootstrap_resamples": 300}}
    assert g3(d, cfg)["passed"]
    assert not g3(d.assign(p_M3=d.p_M2), cfg)["passed"]  # identical models never pass
