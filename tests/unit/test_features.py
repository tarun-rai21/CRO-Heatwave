"""Feature values on hand-built series, climatology fencing, target alignment."""

import numpy as np
import pandas as pd
import pytest

from heatwave.config import DayWindow, read_yaml
from heatwave.features.build import ASOF_FEATURES, FEATURES, compute_features, upwind_neighbours
from heatwave.features.climatology import ClimatologyLeak, fit_climatology
from heatwave.features.targets import per_lead_table, window_table

CFG = read_yaml("features.yaml")
DISTRICTS = pd.DataFrame(
    {
        "district_id": ["west", "mid", "east"],
        "centroid_lat": [27.0, 27.0, 27.0],
        "centroid_lon": [78.0, 79.5, 81.0],  # ~150 km apart along a parallel
        "elevation_m": [180.0, 130.0, 90.0],
    }
)


def synthetic(years=(2000, 2006), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31", freq="D")
    t = 2 * np.pi * (dates.dayofyear - 1) / 365.25
    rows = []
    for k, d in enumerate(DISTRICTS["district_id"]):
        tmax = 32 + 8 * np.sin(t - 1.6) + rng.normal(0, 1.5, len(dates)) + k * 0.3
        rows.append(
            pd.DataFrame(
                {
                    "district_id": d,
                    "obs_date": dates,
                    "tmax": tmax,
                    "T2M_MAX": tmax + 2,
                    "T2M_MIN": tmax - 12,
                    "T2MDEW": tmax - 20,
                    "RH2M": 40.0,
                    "QV2M": 9.0,
                    "PRECTOTCORR": rng.exponential(1.0, len(dates))
                    * (rng.random(len(dates)) < 0.2),
                    "WS10M": 3.0,
                    "U10M": 1.0,
                    "V10M": -0.5,
                    "PS": 99.0 + rng.normal(0, 0.2, len(dates)),
                    "GWETTOP": 0.4 + 0.1 * np.sin(t),
                    "GWETROOT": 0.5 + 0.05 * np.sin(t),
                }
            )
        )
    daily = pd.concat(rows, ignore_index=True)
    labels = daily[["district_id", "obs_date"]].assign(
        is_heatwave=pd.array(daily["tmax"] > 42.5, dtype="boolean")
    )
    return daily, labels


@pytest.fixture(scope="module")
def built():
    daily, labels = synthetic()
    clim = fit_climatology(daily, labels, (2000, 2004), CFG)
    return daily, labels, clim, compute_features(daily, labels, clim, DISTRICTS, CFG)


def row(features, district, date):
    f = features[(features.district_id == district) & (features.asof_date == pd.Timestamp(date))]
    assert len(f) == 1
    return f.iloc[0]


def test_registry_and_budget():
    names = [f.name for f in FEATURES]
    assert len(names) == len(set(names))
    assert len(names) <= 50  # reference.md §4.1 budget
    assert "district_id" not in names  # no district identity feature (§4.2)


def test_output_shape(built):
    daily, _, _, f = built
    assert list(f.columns) == ["district_id", "asof_date", *ASOF_FEATURES]
    assert len(f) == len(daily)


def test_backward_windows_by_hand(built):
    daily, _, clim, f = built
    g = daily[daily.district_id == "mid"].set_index("obs_date")
    t = pd.Timestamp("2005-05-20")
    r = row(f, "mid", t)
    win = g.loc[t - pd.Timedelta(days=6) : t, "tmax"]
    assert r.tmax == pytest.approx(g.loc[t, "tmax"])
    assert r.tmax_max7 == pytest.approx(win.max())
    assert r.dtmax3 == pytest.approx(g.loc[t, "tmax"] - g.loc[t - pd.Timedelta(days=3), "tmax"])
    x = np.arange(7)
    assert r.tmax_slope7 == pytest.approx(np.polyfit(x, win.to_numpy(), 1)[0])
    rain = g.loc[t - pd.Timedelta(days=29) : t, "PRECTOTCORR"].sum()
    assert r.rain30 == pytest.approx(rain)


def test_anomaly_uses_the_fold_normal(built):
    daily, _, clim, f = built
    d = daily[(daily.district_id == "east") & (daily.obs_date == "2006-06-01")]
    r = row(f, "east", "2006-06-01")
    assert r.tmax_anom == pytest.approx(d.tmax.iloc[0] - clim.normal(d, "tmax")[0])


def test_days_since_and_runs():
    daily, labels = synthetic()
    labels["is_heatwave"] = pd.array([False] * len(labels), dtype="boolean")
    hit = (labels.district_id == "west") & (labels.obs_date == "2005-05-10")
    labels.loc[hit, "is_heatwave"] = True
    clim = fit_climatology(daily, labels, (2000, 2004), CFG)
    f = compute_features(daily, labels, clim, DISTRICTS, CFG)
    assert row(f, "west", "2005-05-10").days_since_hw == 0
    assert row(f, "west", "2005-05-15").days_since_hw == 5
    assert row(f, "west", "2005-09-01").days_since_hw == CFG["caps"]["days_since_heatwave"]
    assert row(f, "west", "2005-05-10").hw_today == 1.0


def test_upwind_neighbours_are_west_only():
    nb = upwind_neighbours(DISTRICTS, 250, (225, 337.5))
    assert nb["east"] == ["mid"]  # west is ~300 km away
    assert nb["mid"] == ["west"]
    assert nb["west"] == []


def test_climatology_refuses_its_own_years():
    daily, labels = synthetic()
    clim = fit_climatology(daily, labels, (2000, 2004), CFG)
    clim.check_held_out((2005, 2006))  # fine
    with pytest.raises(ClimatologyLeak):
        clim.check_held_out((2004, 2006))
    assert clim.fit_end_date == pd.Timestamp("2004-12-31")


def test_climatology_ignores_years_outside_the_fit(built):
    daily, labels, clim, _ = built
    shifted = daily.copy()
    shifted.loc[shifted.obs_date.dt.year > 2004, "tmax"] += 20
    other = fit_climatology(shifted, labels, (2000, 2004), CFG)
    assert np.allclose(
        other.normals["tmax"].filter(regex=r"^c\d+$").to_numpy(),
        clim.normals["tmax"].filter(regex=r"^c\d+$").to_numpy(),
    )


def test_per_lead_targets_align_with_issue_and_valid_dates(built):
    _, labels, clim, f = built
    window = DayWindow("03-01", "07-31")
    t = per_lead_table(f, labels, clim, latency=3, horizons=range(1, 11), window=window)
    r = t[(t.district_id == "mid") & (t.asof_date == "2005-05-20") & (t.lead == 4)].iloc[0]
    assert r.issue_date == pd.Timestamp("2005-05-23")
    assert r.valid_date == pd.Timestamp("2005-05-27")
    truth = labels[(labels.district_id == "mid") & (labels.obs_date == "2005-05-27")]
    assert r.y == float(truth.is_heatwave.iloc[0])
    assert t.valid_date.dt.month.between(3, 7).all()


def test_window_rows_need_every_valid_date_in_the_window(built):
    _, labels, _, f = built
    w = window_table(f, labels, 3, range(1, 11), DayWindow("03-01", "07-31"))
    first = w.issue_date + pd.Timedelta(days=1)
    last = w.issue_date + pd.Timedelta(days=10)
    assert first.dt.month.between(3, 7).all() and last.dt.month.between(3, 7).all()
    r = w[(w.district_id == "west") & (w.issue_date == "2005-06-01")].iloc[0]
    days = labels[
        (labels.district_id == "west") & labels.obs_date.between("2005-06-02", "2005-06-11")
    ]
    assert r.y == float(days.is_heatwave.any())
