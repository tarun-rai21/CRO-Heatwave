"""End-to-end ingest runs against a fake POWER client (no network)."""

from datetime import date, timedelta

import pandas as pd
import pytest

from heatwave.config import read_yaml
from heatwave.ingest.power import PowerClient, PowerError, PowerSettings
from heatwave.ingest.run import IngestIncomplete, Paths, check_spine, daily_window, run_ingest

CFG = read_yaml("data.yaml")
S = PowerSettings.from_config(CFG["power"])
START = date(2026, 5, 1)


class FakeClient(PowerClient):
    """Serves `days` of data per cell; the last `unpublished` days are fill values.
    `values` lets a test change what a cell reports between runs."""

    def __init__(self, days=5, unpublished=2, fail=(), bump=0.0):
        super().__init__(S, session=None, sleep=lambda s: None)
        self.days, self.unpublished, self.fail, self.bump = days, unpublished, set(fail), bump

    def get_json(self, lat, lon, start, end):
        if (lat, lon) in self.fail:
            raise PowerError("simulated outage")
        params = {}
        for k, var in enumerate(S.parameters):
            series = {}
            for d in range(self.days):
                day = start + timedelta(days=d)
                published = d < self.days - self.unpublished
                series[day.strftime("%Y%m%d")] = (
                    round(30 + d + k + lat / 100 + self.bump, 2) if published else S.fill_value
                )
            params[var] = series
        return {
            "geometry": {"coordinates": [lon, lat, 100.0]},
            "properties": {"parameter": params},
            "header": {
                "api": {"version": "vtest"},
                "sources": ["MERRA2"],
                "fill_value": S.fill_value,
                "time_standard": "LST",
            },
        }


@pytest.fixture
def paths(tmp_path):
    p = Paths.from_config({"storage": {**CFG["storage"], "root": "data"}}, tmp_path)
    p.grid_cells.parent.mkdir(parents=True)
    pd.DataFrame({"cell_id": ["a", "b"], "lat": [27.0, 27.5], "lon": [80.625, 80.625]}).to_parquet(
        p.grid_cells
    )
    return p


def ingest(paths, client, kind="daily", now="2026-05-08 03:30"):
    return run_ingest(
        CFG,
        paths,
        START,
        START + timedelta(days=10),
        kind,
        client=client,
        now=pd.Timestamp(now, tz="UTC"),
    )


def test_run_stores_published_days_and_logs_latency(paths):
    s = ingest(paths, FakeClient(days=5, unpublished=2))
    assert s["status"] == "ok"
    assert s["rows_appended"] == 2 * 3 * len(S.parameters)  # 2 cells x 3 published days
    assert s["max_obs_date"] == date(2026, 5, 3)
    assert s["latency_days"] == 5  # run on 8 May, latest data 3 May
    log = pd.read_parquet(paths.runs_log)
    assert list(log["status"]) == ["ok"]


def test_repeat_run_appends_only_new_days(paths):
    ingest(paths, FakeClient(days=5, unpublished=2), now="2026-05-08 03:30")
    s = ingest(paths, FakeClient(days=6, unpublished=2), now="2026-05-09 03:30")
    assert s["rows_appended"] == 2 * len(S.parameters)  # one new day per cell
    assert len(pd.read_parquet(paths.runs_log)) == 2


def test_revised_values_are_captured(paths):
    ingest(paths, FakeClient(), now="2026-05-08 03:30")
    s = ingest(paths, FakeClient(bump=0.5), now="2026-05-09 03:30")
    assert s["rows_appended"] == 2 * 3 * len(S.parameters)


def test_partial_failure_stores_the_rest_logs_and_raises(paths):
    with pytest.raises(IngestIncomplete, match="1 of 2"):
        ingest(paths, FakeClient(fail={(27.5, 80.625)}))
    log = pd.read_parquet(paths.runs_log)
    assert log["status"].iloc[-1] == "partial"
    assert log["cells_ok"].iloc[-1] == 1
    assert log["rows_appended"].iloc[-1] == 3 * len(S.parameters)


def test_backfill_writes_cell_elevation(paths):
    ingest(paths, FakeClient(), kind="backfill")
    meta = pd.read_parquet(paths.cell_meta)
    assert list(meta["cell_id"]) == ["a", "b"] and (meta["elevation_m"] == 100.0).all()


def test_spine_check_passes_on_complete_data(paths):
    ingest(paths, FakeClient(days=5, unpublished=0))
    report = check_spine(paths, START, list(S.parameters))
    assert report["complete"], report
    assert report["cells"] == 2


def test_spine_check_flags_missing_cell(paths):
    with pytest.raises(IngestIncomplete):
        ingest(paths, FakeClient(days=5, unpublished=0, fail={(27.5, 80.625)}))
    report = check_spine(paths, START, list(S.parameters))
    assert not report["complete"]
    assert ["b", "T2M_MAX"] in [list(p) for p in report["missing_cell_variable_pairs"]]


def test_spine_check_flags_late_start(paths):
    ingest(paths, FakeClient(days=5, unpublished=0))
    report = check_spine(paths, START - timedelta(days=1), list(S.parameters))
    assert not report["complete"] and report["late_start"]


def test_daily_window():
    assert daily_window(date(2026, 10, 7), 120) == (date(2026, 6, 9), date(2026, 10, 7))
