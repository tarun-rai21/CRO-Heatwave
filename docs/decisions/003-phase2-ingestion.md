# 003: Phase 2 ingestion

**Date:** 2026-10-07
**Phase:** 2
**Affects:** reference.md §1.3 (latency), §2.1, §2.3, §5.3–§5.4; `configs/data.yaml`

## API behaviour confirmed (probe, 2026-10-07)

| Question | Finding | How the code handles it |
|---|---|---|
| Does a point request return the containing cell's value? | **Yes.** Two points inside one cell gave identical series; a point in the neighbouring cell differed. No interpolation | Each cell is requested at its centre; closes the Phase 1 open item |
| What comes back for days not yet published? | The end date is clamped to yesterday, and unpublished days arrive as `-999` | Trailing fills are **dropped** ("not published yet"). Only fills *between* real values are stored, as nulls (genuine gaps) |
| Latency on 2026-10-07 | Last real value **2026-10-03** for all 12 variables: 3–4 days depending on time zone | `L = 3` is borderline. The daily run log measures it; see below |
| Sources in the header | `GEOSIT`, `MERRA2`, `POWER` for a full-history request; `GEOSIT` for recent days | Recorded per run |
| Elevation | Response geometry carries the cell elevation (e.g. 127.6 m) | Saved to `reference/grid_cell_meta` during backfill (for the Phase 4 plains check) |
| Response coordinates | Echo the *requested* point, not the cell centre | Can't be used to verify the cell; hence requests at cell centres |
| Full-history request (1981→today, 12 variables) | ~5–8 s, 3.3 MB JSON | One request per cell; a backfill takes ~15 min |

## Storage design

- **Raw store** `raw/power/obs_year=YYYY/<run_id>.parquet`: append-only. Each run writes only new or changed rows, with `ingested_at`, `run_id`, `api_version`.
  - `best()` = latest version per (cell, date, variable).
  - `best(asof=t)` = what was known at time `t`, used to measure provisional (GEOS-IT) vs final (MERRA-2) values.
  - Partitioning by observation year means the daily job only downloads the current (and, early in the year, previous) year's files.
  - Unchanged re-downloads add nothing, so the 120-day re-fetch costs almost no storage.
- **Run log** `runs/ingest_runs.parquet`: one row per run with status, cells ok / failed, rows fetched / appended, sources, API version, and **observed latency**. Observed latency = run date (UTC) − the latest date that has T2M_MAX in *every* cell.
- **Backfill** caches each cell's response in `cache/power/` (local only), so an interrupted backfill resumes without re-downloading.
- **Partial failure:** a run that misses cells stores what it got, logs `partial`, and exits with an error so GitHub Actions reports it.

## Backfill result (run `20261006T224027Z-backfill`)

| | |
|---|---|
| Cells | 102 / 102 ok |
| Period | 1981-01-01 → 2026-10-03 (last published day on the run date) |
| Values | 20,455,488 (102 cells × 16,712 days × 12 variables) |
| Genuine gaps (nulls) | **0** in every variable |
| `check-raw` | Complete: no missing cell/variable pairs, no late starts, no date gaps |
| Size | 37 MB raw Parquet (zstd), 46 year partitions; uploaded to `tarun-rai/cro-heatwave-data` (private) |
| Observed latency | 3 days |
| Duration | 14 min |

First daily run with sync (`20261006T225641Z-daily`, an hour later): 144,432 values re-fetched, **0 changed**, so no raw files were written; only the run log was pushed. Unchanged re-fetches cost no storage, as designed.

## Daily job

`.github/workflows/daily-ingest.yml`, 03:30 UTC (09:00 IST): pull the needed partitions from Hugging Face → fetch the trailing 120 days for all cells → append changes → push the new files and the run log in one commit.

**Operational risk noted:** GitHub disables scheduled workflows in public repositories after 60 days without repository activity. Commits during development keep it alive. Phase 9 must add a keep-alive before the live season.
