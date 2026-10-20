"""NOAA Global Summary of the Day station data (docs/decisions/005).

Station daily maxima give the reference level for the label's bias correction:
IMD's heatwave thresholds are defined on station Tmax, while POWER runs warm and
IMD's 1-degree grid runs cool on hot days.

Files are kept exactly as downloaded: raw/stations/gsod/<year>/<station>.csv.
A missing station-year (HTTP 404) is normal and recorded in the manifest.
"""

from __future__ import annotations

import io
import random
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

USECOLS = [
    "STATION",
    "DATE",
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "NAME",
    "TEMP",
    "MAX",
    "MAX_ATTRIBUTES",
]


class GsodError(RuntimeError):
    pass


def station_list(history_csv: bytes, state_polygon, buffer_deg: float) -> pd.DataFrame:
    """Stations from isd-history.csv located inside the polygon (plus buffer)."""
    h = pd.read_csv(io.BytesIO(history_csv), dtype=str)
    h = h[h["CTRY"] == "IN"].dropna(subset=["LAT", "LON"]).copy()
    h[["LAT", "LON"]] = h[["LAT", "LON"]].astype(float)
    pts = gpd.GeoDataFrame(h, geometry=gpd.points_from_xy(h["LON"], h["LAT"]), crs="EPSG:4326")
    inside = pts[pts.within(state_polygon.buffer(buffer_deg))]
    out = pd.DataFrame(
        {
            "station_id": inside["USAF"] + inside["WBAN"],
            "name": inside["STATION NAME"].str.title(),
            "lat": inside["LAT"],
            "lon": inside["LON"],
            "begin": inside["BEGIN"],
            "end": inside["END"],
        }
    )
    return out.sort_values("station_id").drop_duplicates("station_id").reset_index(drop=True)


def _get(session, url: str, cfg: dict, sleep) -> requests.Response | None:
    """GET with retries; returns None for 404 (no data for that station-year)."""
    req = cfg["request"]
    last = ""
    for attempt in range(int(req["max_retries"]) + 1):
        if attempt:
            sleep(float(req["backoff_base_s"]) * 2 ** (attempt - 1) * (1 + 0.25 * random.random()))
        try:
            resp = session.get(url, timeout=float(req["timeout_s"]))
        except (requests.ConnectionError, requests.Timeout) as e:
            last = repr(e)
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code == 200:
            return resp
        last = f"HTTP {resp.status_code}"
    raise GsodError(f"{url}: giving up ({last})")


def download(cfg: dict, stations: pd.DataFrame, raw_dir: Path, session=None, sleep=time.sleep):
    """Download every station-year in cfg['years'] not already on disk (resumable).
    Returns the manifest: one row per station-year with status ok / missing."""
    session = session or requests.Session()
    first, last = cfg["years"]
    manifest_path = raw_dir / "manifest.parquet"
    manifest = (
        pd.read_parquet(manifest_path)
        if manifest_path.exists()
        else pd.DataFrame(columns=["station_id", "year", "status", "bytes"])
    )
    done = set(zip(manifest["station_id"], manifest["year"], strict=True))
    rows = []
    for sid in stations["station_id"]:
        for year in range(first, last + 1):
            if (sid, year) in done:
                continue
            resp = _get(session, cfg["data_url"].format(year=year, station=sid), cfg, sleep)
            if resp is None:
                rows.append({"station_id": sid, "year": year, "status": "missing", "bytes": 0})
            else:
                path = raw_dir / str(year) / f"{sid}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                rows.append(
                    {"station_id": sid, "year": year, "status": "ok", "bytes": len(resp.content)}
                )
            sleep(float(cfg["request"]["pause_between_requests_s"]))
    if rows:
        manifest = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        manifest.sort_values(["station_id", "year"]).to_parquet(manifest_path, index=False)
    return manifest


def read_station_days(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """All downloaded station-days: station_id, obs_date, tmax_c (reported maxima
    only, others null), tmean_c, and whether the maximum was hourly-derived."""
    frames = [
        pd.read_csv(f, usecols=USECOLS, dtype={"STATION": str, "MAX_ATTRIBUTES": str})
        for f in sorted(raw_dir.glob("*/*.csv"))
    ]
    if not frames:
        raise GsodError(f"no GSOD files under {raw_dir}")
    d = pd.concat(frames, ignore_index=True)
    miss = float(cfg["missing_value"])
    derived = d["MAX_ATTRIBUTES"].fillna("").str.strip() == cfg["derived_flag"]
    tmax_f = d["MAX"].where(d["MAX"] < miss - 1)
    tmean_f = d["TEMP"].where(d["TEMP"] < miss - 1)
    return (
        pd.DataFrame(
            {
                "station_id": d["STATION"],
                "obs_date": pd.to_datetime(d["DATE"]),
                "tmax_c": ((tmax_f - 32) * 5 / 9).where(~derived),
                "tmean_c": (tmean_f - 32) * 5 / 9,
                "max_derived": derived,
            }
        )
        .sort_values(["station_id", "obs_date"])
        .reset_index(drop=True)
    )
