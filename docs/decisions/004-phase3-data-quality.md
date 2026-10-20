# 004: Phase 3 data quality

**Date:** 2026-10-20
**Phase:** 3
**Affects:** reference.md §2.1 (variables); `configs/data.yaml`, `configs/quality.yaml`; daily workflow

## Finding 1: NASA POWER wind direction is invalid from 2012-12-31 onward

`WD10M` holds values from about −5,000 to −98,990 "degrees" for **every cell on every day from 2012-12-31 onward**: 513,876 values, 30 % of the history. The values before that date are normal. The cached raw API responses contain the same numbers, so this is an **upstream POWER defect**, not a parsing error. Example, cell `merra2_r228_c414`: 2012-12-30 → 85.1°, 2012-12-31 → −8,672, 2026-09-01 → −70,150.

**Fix:** stop fetching `WD10M`; fetch the wind components `U10M` (eastward) and `V10M` (northward) instead. Checked over the full record for cell 27.0 °N 80.625 °E:

| Check | Result |
|---|---|
| U/V available 1981 → present, unit m/s | Yes; only the unpublished last day missing |
| Behaviour identical before and after 2013 | Correlation of `hypot(U,V)` with `WS10M`: 0.949 (1981–2012) vs 0.947 (2013–2026) |
| Pre-2013 `WD10M` = direction of mean (U, V) | Median difference 0.1°, 90th percentile 0.2°, so nothing is lost by switching |
| Physically sensible | Westerly (U > 0) in winter and pre-monsoon, easterly (U < 0) in the July monsoon |

U/V are also what reference.md §4.2 asks for as wind features. Old `WD10M` rows stay in the append-only raw store for audit, but no check, feature or completeness test uses them.

Backfill run `20261020T082000Z-backfill`: 3,413,226 values written. That is U/V for 102 cells × 16,726 days, plus the newest day of the other 11 variables. None of the existing 20.5 M values had been revised. Worth reporting to the POWER team (contact on their site); not done yet.

## Finding 2: everything else is physically clean

`uv run heatwave qa-report` → `reports/data_qa.md`, over 1981-01-01 to 2026-10-17 (22,178,676 values, 13 variables):

| Check | Result |
|---|---|
| Complete date spine for every cell × variable | Yes |
| Missing values | None |
| Hard ranges (configs/quality.yaml), Tmin ≤ Tmean ≤ Tmax, dew point ≤ Tmean, unique keys | **0 violations** |
| IMD Tmax over UP, 1981–2025 | 0 missing point-days, 0 out-of-range values |

## Soft checks: how the rules were chosen

Thresholds were set from the data, not guessed:

- **Day-to-day jumps.** A robust z-score per cell and month alone flags 117 jumps with |z| > 10 (the largest is −13 °C in a day). **All 117 are shared with at least one neighbouring cell**, typically across a whole region with rain, e.g. 11 Oct 1992 in western UP. They are weather events (fronts, storms), not glitches. The rule therefore flags only *isolated* jumps (|z| > 10 with no neighbour |z| > 5). Result: **0 flags** in Tmax and Tmin.
- **Spatial outliers.** Reanalysis is so smooth that z-scores of the difference from neighbours explode (|z| up to 38 for a 4 °C difference). An absolute threshold is used instead: more than **5 °C** from the neighbours' median, beyond the cell's usual offset for that month. The 99.99th percentile of that departure is 4.0 °C.

## Review of the 30 spatial flags

All are kept unchanged. Each has a reason recorded in `configs/quality.yaml` `reviewed_flags`, and `qa-report` fails while any flag lacks one.

| Group | Flags | Explanation |
|---|---|---|
| Foothill cell `merra2_r241_c413` (1,404 m, next to plains cells) | 8 | Terrain and winter inversions; only **0.7 % of Saharanpur's area** |
| Rainfall contrast with neighbours, same day or the day after | 13 | Real convective or frontal weather: rain cools one side of a rain-band edge. Includes the Sept 2026 monsoon-withdrawal rain (provisional data) |
| Fog, cloud or dry-air edges, no strong rain contrast | 7 | Mostly single winter days in edge cells with few neighbours |
| Smog (18 Nov 2017, two adjacent western-UP cells, no rain, ~5 °C cooler) | 2 | The severe Delhi–NCR smog episode of November 2017 cut daytime Tmax |

**Heat-season flags (March–June): 3, all in the Sonbhadra southern-edge cell `merra2_r228_c420`, which covers 3.6 % of the district:**
- 14 Jun 1988 and 8 Jun 2006: pre-monsoon thunderstorms over the neighbours (13.0 vs 1.2 mm and 14.3 vs 4.3 mm), so the dry cell stayed 5 °C hotter.
- 8 Jun 1982: no real rain contrast (0.1 vs 0.6 mm), but drier air (RH 52 vs 63 %) and so less cloud.

At 3.6 % weight, a 5–6 °C departure in this cell moves Sonbhadra's area-weighted Tmax by about 0.2 °C. Whether that ever crosses a label threshold is checked once labels exist (Phase 4).

## Consequences for later phases

- **Phase 4 (labels):** the plan keeps the district maximum of Tmax across cells (reference.md §2.2). Edge slivers must not set that maximum: a 0.7 % mountain sliver or a 3.6 % edge cell is not representative of the district. **Compute the cell maximum only over cells with a meaningful area share.** The threshold is to be set in Phase 4 and recorded there.
- **Phase 5 (features):** wind enters as U/V components; there is no direction feature.
- **Daily operation:** the daily workflow now runs `heatwave validate` after ingest. The run fails (GitHub emails the owner) on any hard-check violation in the current year(s), any cell missing the newest day, or data older than 5 days (`L` + 2).
- **Two writers, one run log:** the laptop and GitHub Actions both write `runs/ingest_runs.parquet`. `hub-push-all` now merges with the remote log by `run_id` before uploading, so a laptop push cannot erase a scheduled run's record.
