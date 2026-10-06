"""NASA POWER Daily API client (reference.md §2.1).

One request per MERRA-2 cell, made at the cell centre, so there is no doubt
which cell the values belong to (POWER returns the containing cell's value).

Fill values (-999) are handled in two ways:
  - trailing fills after a variable's last real value mean "not published yet":
    those days are dropped, not stored, so the daily job does not record
    thousands of fake gaps that later "change" into values;
  - fills between real values are genuine gaps: stored as null.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import requests

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class PowerError(RuntimeError):
    """A request failed permanently, or the response is not what we expect."""


@dataclass(frozen=True)
class PowerSettings:
    base_url: str
    community: str
    time_standard: str
    fill_value: float
    parameters: tuple[str, ...]
    timeout_s: float
    max_retries: int
    backoff_base_s: float
    pause_between_requests_s: float

    @classmethod
    def from_config(cls, cfg: dict) -> PowerSettings:
        req = cfg["request"]
        if len(cfg["parameters"]) > 20:
            raise ValueError("POWER allows at most 20 parameters per point request")
        return cls(
            base_url=cfg["base_url"],
            community=cfg["community"],
            time_standard=cfg["time_standard"],
            fill_value=float(cfg["fill_value"]),
            parameters=tuple(cfg["parameters"]),
            timeout_s=float(req["timeout_s"]),
            max_retries=int(req["max_retries"]),
            backoff_base_s=float(req["backoff_base_s"]),
            pause_between_requests_s=float(req["pause_between_requests_s"]),
        )


@dataclass
class CellResponse:
    """Parsed response for one cell: long-format values plus request metadata."""

    cell_id: str
    values: pd.DataFrame  # cell_id, obs_date, variable, value (null = genuine gap)
    elevation_m: float | None
    api_version: str
    sources: list[str]
    last_published: dict[str, date | None] = field(default_factory=dict)


class PowerClient:
    def __init__(
        self, settings: PowerSettings, session: requests.Session | None = None, sleep=time.sleep
    ):
        self.s = settings
        self.session = session or requests.Session()
        self.sleep = sleep

    def get_json(self, lat: float, lon: float, start: date, end: date) -> dict:
        params = {
            "parameters": ",".join(self.s.parameters),
            "community": self.s.community,
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": self.s.time_standard,
        }
        last_error: Exception | None = None
        for attempt in range(self.s.max_retries + 1):
            if attempt:
                self.sleep(self._wait(attempt, last_error))
            try:
                resp = self.session.get(self.s.base_url, params=params, timeout=self.s.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                continue
            if resp.status_code in RETRYABLE_STATUS:
                last_error = _HttpRetry(resp)
                continue
            if resp.status_code != 200:
                raise PowerError(f"HTTP {resp.status_code} for ({lat}, {lon}): {resp.text[:300]}")
            try:
                return resp.json()
            except ValueError as e:
                last_error = e  # truncated body: retry
        raise PowerError(f"giving up on ({lat}, {lon}) after retries: {last_error!r}")

    def _wait(self, attempt: int, error: Exception | None) -> float:
        if isinstance(error, _HttpRetry) and error.retry_after is not None:
            return error.retry_after
        return self.s.backoff_base_s * 2 ** (attempt - 1) * (1 + 0.25 * random.random())

    def fetch_cell(
        self, cell_id: str, lat: float, lon: float, start: date, end: date
    ) -> CellResponse:
        return parse_response(self.get_json(lat, lon, start, end), cell_id, self.s)


class _HttpRetry(Exception):
    def __init__(self, resp: requests.Response):
        super().__init__(f"HTTP {resp.status_code}")
        try:
            self.retry_after: float | None = float(resp.headers.get("Retry-After"))
        except (TypeError, ValueError):
            self.retry_after = None


def parse_response(payload: dict, cell_id: str, s: PowerSettings) -> CellResponse:
    header = payload.get("header", {})
    if header.get("time_standard", s.time_standard) != s.time_standard:
        raise PowerError(f"{cell_id}: got time standard {header.get('time_standard')}")
    try:
        params = payload["properties"]["parameter"]
    except KeyError as e:
        raise PowerError(f"{cell_id}: unexpected response shape: {payload.get('messages')}") from e
    missing = set(s.parameters) - set(params)
    if missing:
        raise PowerError(f"{cell_id}: response lacks parameters {sorted(missing)}")

    frames, last_published = [], {}
    for var in s.parameters:
        series = params[var]
        dates = pd.to_datetime(list(series.keys()), format="%Y%m%d")
        values = np.asarray(list(series.values()), dtype=float)
        is_fill = np.isclose(values, s.fill_value)
        real = np.flatnonzero(~is_fill)
        if real.size == 0:
            last_published[var] = None
            continue
        keep = slice(0, real[-1] + 1)  # drop the unpublished tail
        last_published[var] = dates[real[-1]].date()
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": cell_id,
                    "obs_date": dates[keep].date,
                    "variable": var,
                    "value": np.where(is_fill[keep], np.nan, values[keep]),
                }
            )
        )

    values_df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["cell_id", "obs_date", "variable", "value"])
    )
    coords = payload.get("geometry", {}).get("coordinates", [])
    return CellResponse(
        cell_id=cell_id,
        values=values_df,
        elevation_m=float(coords[2]) if len(coords) > 2 else None,
        api_version=header.get("api", {}).get("version", "unknown"),
        sources=list(header.get("sources", [])),
        last_published=last_published,
    )
