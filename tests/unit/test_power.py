"""POWER response parsing and client retry behaviour (no network)."""

from datetime import date

import numpy as np
import pytest
import requests

from heatwave.config import read_yaml
from heatwave.ingest.power import PowerClient, PowerError, PowerSettings, parse_response

CFG = read_yaml("data.yaml")["power"]
S = PowerSettings.from_config(CFG)
FILL = S.fill_value


def payload(series_by_var: dict, time_standard="LST"):
    """Build a POWER-shaped payload; each series is a list of values from 2024-05-01."""
    params = {}
    for var in S.parameters:
        vals = series_by_var.get(var, [30.0, 31.0, 32.0])
        params[var] = {f"202405{d:02d}": v for d, v in enumerate(vals, start=1)}
    return {
        "geometry": {"coordinates": [80.625, 27.0, 127.59]},
        "properties": {"parameter": params},
        "header": {
            "api": {"version": "v2.10.0"},
            "sources": ["MERRA2"],
            "fill_value": FILL,
            "time_standard": time_standard,
        },
    }


def values_of(resp, var):
    v = resp.values[resp.values["variable"] == var].sort_values("obs_date")
    return list(v["obs_date"]), v["value"].to_numpy()


def test_long_format_with_all_parameters():
    r = parse_response(payload({}), "c1", S)
    assert len(r.values) == 3 * len(S.parameters)
    assert set(r.values["variable"]) == set(S.parameters)
    assert r.elevation_m == pytest.approx(127.59)
    assert r.api_version == "v2.10.0"


def test_trailing_fills_are_unpublished_and_dropped():
    r = parse_response(payload({"T2M_MAX": [40.0, 41.0, FILL, FILL]}), "c1", S)
    dates, vals = values_of(r, "T2M_MAX")
    assert dates == [date(2024, 5, 1), date(2024, 5, 2)]
    assert r.last_published["T2M_MAX"] == date(2024, 5, 2)


def test_interior_fill_is_a_genuine_gap_stored_as_null():
    r = parse_response(payload({"T2M_MAX": [40.0, FILL, 42.0]}), "c1", S)
    _, vals = values_of(r, "T2M_MAX")
    assert vals[0] == 40.0 and np.isnan(vals[1]) and vals[2] == 42.0


def test_fill_never_survives_as_a_number():
    r = parse_response(payload({"GWETROOT": [FILL, 0.4, FILL, 0.5, FILL]}), "c1", S)
    assert not np.isclose(r.values["value"], FILL).any()


def test_variable_with_nothing_published():
    r = parse_response(payload({"GWETTOP": [FILL, FILL, FILL]}), "c1", S)
    assert "GWETTOP" not in set(r.values["variable"])
    assert r.last_published["GWETTOP"] is None


def test_wrong_time_standard_is_rejected():
    with pytest.raises(PowerError, match="time standard"):
        parse_response(payload({}, time_standard="UTC"), "c1", S)


def test_missing_parameter_is_rejected():
    p = payload({})
    del p["properties"]["parameter"]["PS"]
    with pytest.raises(PowerError, match="PS"):
        parse_response(p, "c1", S)


def test_more_than_20_parameters_rejected():
    cfg = {**CFG, "parameters": [f"P{i}" for i in range(21)]}
    with pytest.raises(ValueError, match="20"):
        PowerSettings.from_config(cfg)


# --- retries ---------------------------------------------------------------


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = str(body)

    def json(self):
        if self._body is None:
            raise ValueError("truncated")
        return self._body


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def get(self, url, params, timeout):
        self.calls.append(params)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def client(outcomes):
    waits = []
    c = PowerClient(S, session=FakeSession(outcomes), sleep=waits.append)
    return c, waits


def fetch(c):
    return c.get_json(27.0, 80.625, date(2024, 5, 1), date(2024, 5, 3))


def test_retries_429_and_honours_retry_after():
    c, waits = client(
        [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"ok": 1})]
    )
    assert fetch(c) == {"ok": 1}
    assert waits == [7.0]


def test_retries_timeouts_and_server_errors_with_backoff():
    c, waits = client(
        [
            requests.Timeout(),
            FakeResponse(503),
            FakeResponse(200, None),
            FakeResponse(200, {"ok": 1}),
        ]
    )
    assert fetch(c) == {"ok": 1}
    assert len(waits) == 3
    assert waits[0] < waits[1] < waits[2]  # exponential


def test_client_errors_are_not_retried():
    c, waits = client([FakeResponse(400, "bad parameter")])
    with pytest.raises(PowerError, match="HTTP 400"):
        fetch(c)
    assert waits == []


def test_gives_up_after_max_retries():
    c, _ = client([FakeResponse(503)] * (S.max_retries + 1))
    with pytest.raises(PowerError, match="giving up"):
        fetch(c)


def test_request_parameters():
    c, _ = client([FakeResponse(200, {"ok": 1})])
    fetch(c)
    sent = c.session.calls[0]
    assert sent["time-standard"] == "LST"
    assert sent["start"] == "20240501" and sent["end"] == "20240503"
    assert sent["parameters"].split(",") == list(S.parameters)
