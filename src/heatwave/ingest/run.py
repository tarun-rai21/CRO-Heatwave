"""Ingest runs: historical backfill and the daily trailing-window job (reference.md §5.4).

Both fetch every cell in reference/grid_cells, append changed values to the raw
store, and log the run (including observed data latency) to the runs log.
A run that misses any cell still stores what it got, logs status "partial",
and raises, so the scheduler reports a failure.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from heatwave.ingest.power import (
    CellResponse,
    PowerClient,
    PowerError,
    PowerSettings,
    parse_response,
)
from heatwave.ingest.raw_store import RawStore

LATENCY_VARIABLE = "T2M_MAX"


class IngestIncomplete(RuntimeError):
    """Some cells could not be fetched; the rest were stored."""


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_power: Path
    runs_log: Path
    cache: Path
    grid_cells: Path
    cell_meta: Path

    @classmethod
    def from_config(cls, cfg: dict, repo_root: Path) -> Paths:
        s = cfg["storage"]
        root = repo_root / s["root"]
        return cls(
            root=root,
            raw_power=root / s["raw_power_dir"],
            runs_log=root / s["runs_log"],
            cache=root / s["cache_dir"],
            grid_cells=root / "reference" / "grid_cells.parquet",
            cell_meta=root / "reference" / "grid_cell_meta.parquet",
        )


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("s")


def run_ingest(
    cfg: dict,
    paths: Paths,
    start: date,
    end: date,
    kind: str,
    client: PowerClient | None = None,
    use_cache: bool = False,
    now: pd.Timestamp | None = None,
) -> dict:
    settings = PowerSettings.from_config(cfg["power"])
    client = client or PowerClient(settings)
    started = now or utc_now()
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{kind}"

    cells = pd.read_parquet(paths.grid_cells)
    responses: list[CellResponse] = []
    failed: dict[str, str] = {}

    for n, cell in enumerate(cells.itertuples(index=False), 1):
        try:
            payload = _cached_fetch(client, cell, start, end, paths.cache if use_cache else None)
            responses.append(parse_response(payload, cell.cell_id, settings))
            print(f"[{n}/{len(cells)}] {cell.cell_id} ok")
        except PowerError as e:
            failed[cell.cell_id] = str(e)[:300]
            print(f"[{n}/{len(cells)}] {cell.cell_id} FAILED: {e}")

    store = RawStore(paths.raw_power)
    values = pd.concat([r.values for r in responses], ignore_index=True) if responses else None
    api_versions = sorted({r.api_version for r in responses})
    appended = 0
    if values is not None and not values.empty:
        appended = store.append(values, run_id, started, ",".join(api_versions))
    if kind == "backfill" and responses:
        _write_cell_meta(responses, paths.cell_meta)

    summary = _summarise(
        run_id, kind, started, start, end, cells, responses, failed, values, appended
    )
    summary["api_version"] = ",".join(api_versions)
    _append_run_log(paths.runs_log, summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "failed_cells"}, indent=2, default=str)
    )

    if failed:
        raise IngestIncomplete(f"{len(failed)} of {len(cells)} cells failed: {sorted(failed)}")
    return summary


def daily_window(today: date, trailing_days: int) -> tuple[date, date]:
    """Trailing window ending today. POWER clamps the end date and returns its
    unpublished tail as fill values, which parsing drops."""
    return today - timedelta(days=trailing_days), today


def _cached_fetch(client: PowerClient, cell, start: date, end: date, cache: Path | None) -> dict:
    if cache is not None:
        path = cache / f"{cell.cell_id}_{start:%Y%m%d}_{end:%Y%m%d}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
    payload = client.get_json(cell.lat, cell.lon, start, end)
    client.sleep(client.s.pause_between_requests_s)
    if cache is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
    return payload


def _write_cell_meta(responses: list[CellResponse], path: Path) -> None:
    meta = pd.DataFrame(
        {
            "cell_id": [r.cell_id for r in responses],
            "elevation_m": [r.elevation_m for r in responses],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.sort_values("cell_id").to_parquet(path, index=False)


def _summarise(run_id, kind, started, start, end, cells, responses, failed, values, appended):
    last = {
        var: min(
            (r.last_published.get(var) for r in responses if r.last_published.get(var)),
            default=None,
        )
        for var in (responses[0].last_published if responses else {})
    }
    max_obs = last.get(LATENCY_VARIABLE)
    return {
        "run_id": run_id,
        "kind": kind,
        "started_at": started,
        "finished_at": utc_now(),
        "status": "ok" if not failed else ("partial" if responses else "failed"),
        "window_start": start,
        "window_end": end,
        "cells_requested": len(cells),
        "cells_ok": len(responses),
        "failed_cells": json.dumps(failed),
        "rows_fetched": 0 if values is None else len(values),
        "rows_appended": appended,
        # Latest date with published T2M_MAX in EVERY cell, and its age on the run date.
        "max_obs_date": max_obs,
        "latency_days": None if max_obs is None else (started.date() - max_obs).days,
        "last_published_by_variable": json.dumps({k: str(v) for k, v in last.items()}),
        "sources": json.dumps(sorted({s for r in responses for s in r.sources})),
    }


def _append_run_log(path: Path, summary: dict) -> None:
    row = pd.DataFrame([summary])
    for col in ("window_start", "window_end", "max_obs_date"):
        row[col] = pd.to_datetime(row[col])
    log = pd.concat([pd.read_parquet(path), row], ignore_index=True) if path.exists() else row
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_parquet(path, index=False)


def check_spine(paths: Paths, start: date, variables: list[str]) -> dict:
    """Completeness of the best-known raw data: every cell x variable must have a
    row for every date from `start` to that variable's last published date."""
    store = RawStore(paths.raw_power)
    sql = store.best_sql()
    if sql is None:
        raise RuntimeError("raw store is empty; run the backfill first")
    cells = set(pd.read_parquet(paths.grid_cells)["cell_id"])

    with duckdb.connect() as con:
        per = con.sql(
            f"""
            SELECT cell_id, variable, count(*) AS n_rows, count(value) AS n_values,
                   min(obs_date) AS first_date, max(obs_date) AS last_date
            FROM ({sql}) GROUP BY cell_id, variable
            """
        ).df()

    per["first_date"] = pd.to_datetime(per["first_date"]).dt.date
    per["last_date"] = pd.to_datetime(per["last_date"]).dt.date
    per["expected"] = [(last - start).days + 1 for last in per["last_date"]]
    expected_pairs = {(c, v) for c in cells for v in variables}
    present_pairs = set(zip(per["cell_id"], per["variable"], strict=True))

    problems = {
        "missing_cell_variable_pairs": sorted(expected_pairs - present_pairs),
        "unexpected_cells": sorted(set(per["cell_id"]) - cells),
        "late_start": per.loc[per["first_date"] > start, ["cell_id", "variable"]].values.tolist(),
        "date_gaps": per.loc[
            per["n_rows"] != per["expected"], ["cell_id", "variable"]
        ].values.tolist(),
    }
    nulls = per.assign(n_null=per["n_rows"] - per["n_values"]).groupby("variable")["n_null"].sum()
    return {
        "complete": not any(problems.values()),
        "cells": len(set(per["cell_id"])),
        "rows": int(per["n_rows"].sum()),
        "first_date": str(per["first_date"].min()),
        "last_date_by_variable": {
            v: str(d) for v, d in per.groupby("variable")["last_date"].min().items()
        },
        "null_values_by_variable": {k: int(v) for k, v in nulls.items() if v},
        **{k: v[:20] for k, v in problems.items()},
    }
