"""Label build: episodes, percentile runs, main label with a known map."""

import numpy as np
import pandas as pd
import pytest

from heatwave.config import load_problem
from heatwave.labels.bias_correction import QUANTILES
from heatwave.labels.build import episodes, main_labels, percentile_label, rule_labels

RULE = load_problem().target


def identity_map():
    rows = [
        {"month": m, "q": q, "source_q": 20 + 30 * q, "target_q": 20 + 30 * q}
        for m in range(1, 13)
        for q in QUANTILES
    ]
    qm = pd.DataFrame(rows)
    qm["fit_start"], qm["fit_end"] = 1991, 2020
    return qm


def climate(districts=("a", "b"), start="1990-01-01", end="2021-12-31", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    t = 2 * np.pi * (dates.dayofyear - 1) / 365.25
    frames = []
    for k, d in enumerate(districts):
        tmax = 32 + 8 * np.sin(t - 1.6) + rng.normal(0, 1.5, len(dates)) + k
        frames.append(pd.DataFrame({"district_id": d, "obs_date": dates, "T2M_MAX": tmax}))
    return pd.concat(frames, ignore_index=True)


def test_identity_map_gives_the_plain_rule():
    df = climate()
    main = main_labels(df, identity_map(), RULE)
    plain = rule_labels(df, "district_id", "T2M_MAX", RULE)
    assert (main["is_heatwave"].to_numpy() == plain["is_heatwave"].to_numpy()).all()
    assert main["label_version"].nunique() == 1


def test_a_cooling_map_reduces_heatwave_days():
    df = climate()
    df.loc[df.obs_date.dt.month == 5, "T2M_MAX"] += 6  # make May hot enough to label
    qm = identity_map()
    cool = qm.assign(target_q=qm.target_q - 2.0)
    n_ident = main_labels(df, qm, RULE)["is_heatwave"].sum()
    n_cool = main_labels(df, cool, RULE)["is_heatwave"].sum()
    assert n_cool < n_ident


def test_episodes_merge_consecutive_days_across_districts():
    dates = pd.to_datetime(["2020-05-01", "2020-05-02", "2020-05-03", "2020-05-10"])
    labels = pd.DataFrame(
        {
            "district_id": ["a", "b", "a", "b"],
            "obs_date": dates,
            "is_heatwave": pd.array([True, True, True, True], dtype="boolean"),
        }
    )
    ep = episodes(labels)
    assert ep[["days", "district_days"]].values.tolist() == [[3, 3], [1, 1]]


def test_episode_threshold_on_districts():
    labels = pd.DataFrame(
        {
            "district_id": ["a", "b", "a"],
            "obs_date": pd.to_datetime(["2020-05-01", "2020-05-01", "2020-05-02"]),
            "is_heatwave": pd.array([True, True, True], dtype="boolean"),
        }
    )
    ep = episodes(labels, min_districts=2)
    assert len(ep) == 1 and ep["days"].iloc[0] == 1 and ep["peak_districts"].iloc[0] == 2


def test_missing_labels_do_not_create_episodes():
    labels = pd.DataFrame(
        {
            "district_id": ["a"],
            "obs_date": pd.to_datetime(["2020-05-01"]),
            "is_heatwave": pd.array([pd.NA], dtype="boolean"),
        }
    )
    assert episodes(labels).empty


@pytest.mark.parametrize(("hot_days", "flagged"), [(2, 0), (3, 3), (5, 5)])
def test_percentile_label_needs_a_run_of_three(hot_days, flagged):
    df = climate(districts=("a",), seed=1).rename(columns={"T2M_MAX": "tmax"})
    sel = (df.obs_date >= "2021-05-10") & (
        df.obs_date < pd.Timestamp("2021-05-10") + pd.Timedelta(days=hot_days)
    )
    df.loc[sel, "tmax"] += 15
    flags = percentile_label(df, "district_id", "tmax", (1991, 2020), 0.95, 7, 3)
    window = (df.obs_date >= "2021-05-08") & (df.obs_date <= "2021-05-20")
    assert int(flags[window].sum()) == flagged


# --- spatial bias plane -----------------------------------------------------------

from heatwave.labels.bias_correction import fit_bias_plane, plane_value, plane_version  # noqa: E402


def plane_inputs():
    districts = pd.DataFrame(
        {
            "district_id": list("abcd"),
            "centroid_lat": [24.5, 25.0, 28.0, 29.5],
            "centroid_lon": [78.5, 82.0, 80.0, 77.5],
        }
    )
    dates = pd.date_range("1991-03-01", "2020-07-31", freq="D")
    dates = dates[dates.month.isin([3, 4, 5, 6, 7])]
    power, imd = [], []
    for r in districts.itertuples():
        true_bias = -20.0 + 0.8 * r.centroid_lat + 0.05 * r.centroid_lon
        base = 38 + np.sin(np.arange(len(dates)))
        imd.append(
            pd.DataFrame({"district_id": r.district_id, "obs_date": dates, "tmax_imd": base})
        )
        power.append(
            pd.DataFrame(
                {"district_id": r.district_id, "obs_date": dates, "T2M_MAX": base + true_bias}
            )
        )
    return pd.concat(power), pd.concat(imd), districts


def test_bias_plane_recovers_a_known_gradient():
    power, imd, districts = plane_inputs()
    plane = fit_bias_plane(power, imd, districts, (1991, 2020), [3, 4, 5, 6, 7])
    assert (plane["a"], plane["b_lat"], plane["c_lon"]) == pytest.approx((-20.0, 0.8, 0.05))
    assert plane["rmse_c"] < 1e-9


def test_main_label_removes_the_plane_at_each_district():
    power, imd, districts = plane_inputs()
    plane = fit_bias_plane(power, imd, districts, (1991, 2020), [3, 4, 5, 6, 7])
    out = main_labels(power, identity_map(), RULE, plane, districts)
    expected = power["T2M_MAX"].to_numpy() - plane_value(
        plane, *power[["district_id"]].merge(districts)[["centroid_lat", "centroid_lon"]].T.values
    )
    assert np.allclose(out["tmax_used"], np.round(expected, 1))
    with_plane = out["label_version"].iloc[0]
    assert with_plane != main_labels(power, identity_map(), RULE)["label_version"].iloc[0]


def test_plane_version_tracks_coefficients():
    power, imd, districts = plane_inputs()
    p = fit_bias_plane(power, imd, districts, (1991, 2020), [3, 4, 5, 6, 7])
    assert plane_version(p) == plane_version(dict(p))
    assert plane_version({**p, "b_lat": p["b_lat"] + 0.01}) != plane_version(p)


# --- temporal drift ----------------------------------------------------------------

from heatwave.labels.bias_correction import drift_value, fit_bias_drift  # noqa: E402


def test_drift_recovers_a_step_and_holds_it_after_the_fitted_years():
    power, imd, districts = plane_inputs()
    power = power.copy()
    power.loc[power.obs_date.dt.year >= 2005, "T2M_MAX"] -= 1.0  # POWER cools by 1 C
    # The synthetic POWER carries a spatial plane; fit it on the years before the step.
    plane = fit_bias_plane(
        power[power.obs_date.dt.year < 2005], imd, districts, (1991, 2004), [3, 4, 5, 6, 7]
    )
    drift = fit_bias_drift(
        power,
        imd,
        districts,
        plane,
        (1991, 2020),
        [3, 4, 5, 6, 7],
        smooth_years=1,
        hold_years=(2016, 2020),
    )
    assert drift_value(drift, [1995])[0] == pytest.approx(0.0, abs=1e-9)
    assert drift_value(drift, [2010])[0] == pytest.approx(-1.0, abs=1e-9)
    assert drift_value(drift, [2025])[0] == pytest.approx(drift["hold_value"])
    assert drift["hold_value"] == pytest.approx(-1.0, abs=1e-9)
    assert drift_value(drift, [1985])[0] == pytest.approx(drift_value(drift, [1991])[0])


def test_drift_is_removed_before_the_map():
    power, imd, districts = plane_inputs()
    plane = fit_bias_plane(power, imd, districts, (1991, 2020), [3, 4, 5, 6, 7])
    drift = {
        "drift": {y: (0.5 if y >= 2010 else 0.0) for y in range(1991, 2021)},
        "hold_value": 0.5,
        "years": [1991, 2020],
        "smooth_years": 1,
    }
    a = main_labels(power, identity_map(), RULE, plane, districts)
    b = main_labels(power, identity_map(), RULE, plane, districts, drift)
    late = a.obs_date.dt.year >= 2010
    assert np.allclose(a.tmax_used[late] - b.tmax_used[late], 0.5, atol=0.051)
    assert np.allclose(a.tmax_used[~late], b.tmax_used[~late])
    assert a.label_version.iloc[0] != b.label_version.iloc[0]
