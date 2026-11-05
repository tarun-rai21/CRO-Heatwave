"""The drift check finds a source that cools against the stations."""

import numpy as np
import pandas as pd

from heatwave.labels import drift_check as dc


def table():
    rng = np.random.default_rng(0)
    rows = []
    for name, sid in (("A", "1"), ("B", "2")):
        for y in range(2016, 2026):
            for day in pd.date_range(f"{y}-05-01", periods=40):
                t = 38 + rng.normal(0, 1)
                rows.append((sid, name, sid, day, t, t + (-0.5 if y >= 2021 else 0), t + 1, t - 1))
    return pd.DataFrame(
        rows,
        columns=[
            "station_id",
            "name",
            "district_id",
            "obs_date",
            "tmax_c",
            "tmax_used",
            "tmax_raw",
            "tmax_imd",
        ],
    )


def change(t, **kw):
    d = dc.paired_days(
        t[["station_id", "obs_date", "tmax_c"]],
        t[["station_id", "name", "district_id"]].drop_duplicates(),
        t[["district_id", "obs_date", "tmax_used", "tmax_raw"]],
        t[["district_id", "obs_date", "tmax_imd"]],
        [5],
    )
    return dc.period_change(d, (2016, 2020), (2021, 2025), resamples=100, **kw)


def test_a_cooling_label_is_found_and_a_steady_source_is_not():
    s = change(table())["summary"].set_index("source")
    assert abs(s.loc["label", "change"] + 0.5) < 1e-6
    assert s.loc["label", "stations_cooler"] == 2
    assert abs(s.loc["imd", "change"]) < 1e-6


def test_a_station_missing_from_one_period_is_left_out():
    t = table()
    t = t[~((t["name"] == "B") & (t["obs_date"].dt.year >= 2021))]
    assert change(t)["by_station"]["name"].tolist() == ["A"]
