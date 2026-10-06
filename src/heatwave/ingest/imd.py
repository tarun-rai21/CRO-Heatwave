"""IMD Pune 1-degree gridded daily Tmax (reference.md §2.4).

Used to validate and possibly bias-correct the label in Phase 4. Files are kept
exactly as downloaded (raw/imd/tmax/<year>.GRD) with a manifest recording size,
checksum and download time. A file only counts as downloaded if its size is
exactly days_in_year x n_lat x n_lon x 4 bytes: a failed request can return an
HTML page with status 200, and the IMD server is slow and resets connections.
"""

from __future__ import annotations

import calendar
import hashlib
import random
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) cro-heatwave (research)"


class ImdError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImdSettings:
    tmax_url: str
    form_field: str
    referer: str
    lat_first: float
    lon_first: float
    step_deg: float
    n_lat: int
    n_lon: int
    missing_value: float
    timeout_s: float
    max_retries: int
    backoff_base_s: float
    pause_between_requests_s: float

    @classmethod
    def from_config(cls, cfg: dict) -> ImdSettings:
        req = cfg["request"]
        return cls(
            tmax_url=cfg["tmax_url"],
            form_field=cfg["tmax_form_field"],
            referer=cfg["referer"],
            lat_first=float(cfg["lat_first"]),
            lon_first=float(cfg["lon_first"]),
            step_deg=float(cfg["step_deg"]),
            n_lat=int(cfg["n_lat"]),
            n_lon=int(cfg["n_lon"]),
            missing_value=float(cfg["missing_value"]),
            timeout_s=float(req["timeout_s"]),
            max_retries=int(req["max_retries"]),
            backoff_base_s=float(req["backoff_base_s"]),
            pause_between_requests_s=float(req["pause_between_requests_s"]),
        )

    def expected_bytes(self, year: int) -> int:
        days = 366 if calendar.isleap(year) else 365
        return days * self.n_lat * self.n_lon * 4

    @property
    def lats(self) -> np.ndarray:
        return self.lat_first + self.step_deg * np.arange(self.n_lat)

    @property
    def lons(self) -> np.ndarray:
        return self.lon_first + self.step_deg * np.arange(self.n_lon)


def download_year(
    s: ImdSettings, year: int, session: requests.Session | None = None, sleep=time.sleep
) -> bytes:
    """Download one year's Tmax file, retrying until the size is exactly right."""
    session = session or requests.Session()
    headers = {"User-Agent": USER_AGENT, "Referer": s.referer}
    expected = s.expected_bytes(year)
    last: str = ""
    for attempt in range(s.max_retries + 1):
        if attempt:
            sleep(s.backoff_base_s * 2 ** (attempt - 1) * (1 + 0.25 * random.random()))
        try:
            resp = session.post(
                s.tmax_url, data={s.form_field: year}, headers=headers, timeout=s.timeout_s
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            last = repr(e)
            continue
        if resp.status_code != 200:
            last = f"HTTP {resp.status_code}"
            continue
        if len(resp.content) == expected:
            return resp.content
        # Wrong size: an error page, a truncated body, or the year is not published.
        last = f"got {len(resp.content)} bytes, expected {expected}: {resp.content[:120]!r}"
    raise ImdError(f"IMD Tmax {year}: giving up after retries ({last})")


def read_grd(path: Path, year: int, s: ImdSettings) -> np.ndarray:
    """(days, n_lat, n_lon) float array with missing values as NaN."""
    raw = np.fromfile(path, dtype="<f4")
    days = 366 if calendar.isleap(year) else 365
    if raw.size != days * s.n_lat * s.n_lon:
        raise ImdError(f"{path}: {raw.size} values, expected {days * s.n_lat * s.n_lon}")
    data = raw.reshape(days, s.n_lat, s.n_lon).astype(float)
    data[np.isclose(data, s.missing_value)] = np.nan
    return data


def to_long(data: np.ndarray, year: int, s: ImdSettings, bbox=None) -> pd.DataFrame:
    """Long table (obs_date, lat, lon, tmax) of land cells, optionally within a
    (min_lon, min_lat, max_lon, max_lat) box."""
    dates = pd.date_range(date(year, 1, 1), periods=data.shape[0], freq="D")
    lat_idx, lon_idx = np.meshgrid(np.arange(s.n_lat), np.arange(s.n_lon), indexing="ij")
    keep = ~np.isnan(data).all(axis=0)
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        lat, lon = s.lats[lat_idx], s.lons[lon_idx]
        keep &= (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
    js, is_ = lat_idx[keep], lon_idx[keep]
    values = data[:, js, is_]  # (days, n_kept)
    return pd.DataFrame(
        {
            "obs_date": np.repeat(dates.date, len(js)),
            "lat": np.tile(s.lats[js], len(dates)),
            "lon": np.tile(s.lons[is_], len(dates)),
            "tmax": values.ravel(),
        }
    )


def ingest_years(s: ImdSettings, raw_dir: Path, years: list[int], sleep=time.sleep) -> pd.DataFrame:
    """Download missing years (resumable). Returns the manifest after this run."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else pd.DataFrame()
    failed = {}
    for year in years:
        path = raw_dir / f"{year}.GRD"
        if path.exists() and path.stat().st_size == s.expected_bytes(year):
            print(f"{year}: present")
            continue
        try:
            content = download_year(s, year, sleep=sleep)
        except ImdError as e:
            failed[year] = str(e)
            print(f"{year}: FAILED {e}")
            continue
        path.write_bytes(content)
        row = pd.DataFrame(
            [
                {
                    "year": year,
                    "file": path.name,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "downloaded_at": pd.Timestamp.now(tz="UTC").floor("s"),
                    "url": s.tmax_url,
                }
            ]
        )
        if not manifest.empty:
            manifest = manifest[manifest["year"] != year]
        manifest = pd.concat([manifest, row]).sort_values("year").reset_index(drop=True)
        manifest.to_parquet(manifest_path, index=False)
        print(f"{year}: downloaded {len(content)} bytes")
        sleep(s.pause_between_requests_s)
    if failed:
        raise ImdError(f"{len(failed)} year(s) failed: {sorted(failed)}")
    return manifest
