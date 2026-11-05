"""Frozen model: reproduces the fitted model, keeps the final test out, assigns tiers."""

import json

import numpy as np
import pandas as pd
import pytest

from heatwave.config import read_yaml
from heatwave.models import baselines
from heatwave.models.frozen import (
    FrozenModel,
    calibrator_to_dict,
    check_matches,
    logistic_to_dict,
    sha256,
)
from heatwave.validation import calibration
from heatwave.validation.freeze import development_rows

CFG = read_yaml("models.yaml")
FEATURES = CFG["baselines"]["m2"]["features"]


def rows(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    issue = pd.to_datetime("2005-03-01") + pd.to_timedelta(rng.integers(0, 150, n), "D")
    clim = rng.uniform(0.01, 0.2, n)
    d = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    d["district_id"] = rng.choice(["a", "b"], n)
    d["issue_date"] = issue
    d["clim_hw_freq_valid"] = clim
    d["lead"] = rng.integers(1, 3, n)
    logit = np.log(clim / (1 - clim)) + 0.8 * d["tmax_anom"]
    d["y"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return d


def frozen_from(tmp_path, train):
    lead = baselines.make("M2", "lead", CFG).fit(train)
    window = baselines.make("M2", "window", CFG).fit(train)
    cal = calibration.fit(lead.predict(train), train["y"].to_numpy())
    window.m0.table.to_parquet(tmp_path / "t.parquet", index=False)
    m2 = CFG["baselines"]["m2"]
    spec = {
        "version": "test",
        "features": list(m2["features"]),
        "clim_clip": list(m2["clim_clip"]),
        "m0_smooth_days": CFG["baselines"]["m0_window"]["smooth_days"],
        "m0_overall": window.m0.overall,
        "m0_table": "t.parquet",
        "m0_table_sha256": sha256(tmp_path / "t.parquet"),
        "lead": {
            str(h): {**logistic_to_dict(f), "calibrator": calibrator_to_dict(cal)}
            for h, f in lead.models.items()
        },
        "window": {**logistic_to_dict(window.model), "calibrator": calibrator_to_dict(cal)},
        "tiers": {
            "watch": {"threshold": 0.15},
            "alert": {"threshold": 0.25},
            "warning": {"threshold": 0.4},
        },
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(spec))
    return FrozenModel.load(path), lead, window, cal


def test_frozen_numbers_reproduce_the_fitted_and_calibrated_model(tmp_path):
    train = rows()
    frozen, lead, window, cal = frozen_from(tmp_path, train)
    test = rows(seed=1)
    test.loc[test.index[:20], "tmax_anom"] = np.nan  # missing inputs fall back to climatology
    want = calibration.apply(cal, lead.predict(test))
    assert np.allclose(frozen.predict_lead(test), want, rtol=0, atol=1e-8)
    want = calibration.apply(cal, window.predict(test))
    assert np.allclose(frozen.predict_window(test), want, rtol=0, atol=1e-8)


def test_check_matches_catches_a_wrong_coefficient(tmp_path):
    train = rows()
    frozen, lead, _, _ = frozen_from(tmp_path, train)
    part = json.loads(json.dumps(frozen.spec["lead"]["1"]))
    part["coef"][0] += 0.1
    prior = train["clim_hw_freq_valid"].to_numpy()
    with pytest.raises(ValueError, match="do not reproduce"):
        check_matches(lead.lr, lead.models[1], part, train, prior, frozen)


def test_a_changed_climatology_table_is_refused(tmp_path):
    frozen_from(tmp_path, rows())
    t = pd.read_parquet(tmp_path / "t.parquet")
    t["p"] = t["p"] + 0.01
    t.to_parquet(tmp_path / "t.parquet", index=False)
    with pytest.raises(ValueError, match="does not match"):
        FrozenModel.load(tmp_path / "m.json")


def test_tiers_are_the_highest_one_reached(tmp_path):
    frozen, *_ = frozen_from(tmp_path, rows())
    got = frozen.tier(np.array([0.1, 0.15, 0.3, 0.4, 0.9]))
    assert got.tolist() == ["none", "watch", "alert", "warning", "warning"]


@pytest.mark.parametrize("target,col", [("lead", "valid_date"), ("window", "issue_date")])
def test_final_test_years_never_reach_the_freeze(target, col):
    d = pd.DataFrame(
        {col: pd.to_datetime(["2019-05-01", "2020-05-01", "2021-05-01", "2025-05-01"])}
    )
    kept = development_rows(d, target, (1981, 2020))
    assert kept[col].dt.year.tolist() == [2019, 2020]
