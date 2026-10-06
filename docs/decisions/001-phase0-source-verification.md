# 001: Phase 0 source verification

**Date:** 2026-10-06 (sources retrieved), recorded 2026-10-07
**Phase:** 0
**Affects:** reference.md §1.3, §2.1, §2.4; `configs/problem.yaml`

Phase 0 requires checking the facts the problem definition rests on against primary sources, not blogs.

## IMD heatwave criteria

**Source:** IMD, *FAQ on Heat Wave*: https://internal.imd.gov.in/section/nhac/dynamic/FAQ_heat_wave.pdf

> Heat wave is considered if maximum temperature of a station reaches at least 40°C or more for Plains and at least 30°C or more for Hilly regions.
> a) Based on Departure from Normal — Heat Wave: Departure from normal is 4.5°C to 6.4°C. Severe Heat Wave: Departure from normal is >6.4°C.
> b) Based on Actual Maximum Temperature — Heat Wave: When actual maximum temperature ≥ 45°C. Severe Heat Wave: When actual maximum temperature ≥ 47°C.
> If above criteria met at least in 2 stations in a Meteorological sub-division for at least two consecutive days and it declared on the second day.

> Based on daily maximum temperature station data, climatology of maximum temperature is prepared for the period 1991-2020 to find out normal maximum temperature of the day for a particular station.

How `problem.yaml` encodes it:

| IMD wording | Config |
|---|---|
| "at least 40°C … for Plains" | `tmax_floor_c: 40.0`, applied to the departure criteria |
| "4.5°C to 6.4°C" (heatwave), ">6.4°C" (severe) | `departure ≥ 4.5` → heatwave; `departure > 6.4` → severe. Heatwave includes all severe days |
| "≥ 45°C", "≥ 47°C" | `absolute_heatwave_c: 45.0`, `absolute_severe_c: 47.0` |
| "1991-2020" | `normal.period: [1991, 2020]` |
| 2 stations × 2 consecutive days | **Not encoded.** This is IMD's declaration rule; we predict the district-level condition (Target A) |

Reported values are to 0.1 °C, hence `round_decimals: 1`. Rounding before comparison stops float noise from flipping a label at a threshold.

**Still assumed:** all 75 UP districts use the *plains* criteria (none qualify as hilly). Check in Phase 4 using district elevation.

## NASA POWER API and data

**Sources:**
- Daily API: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- Data sources: https://power.larc.nasa.gov/docs/methodology/data/sources/

| Fact | Finding | Consequence |
|---|---|---|
| Parameters per request | Max **20** for a single point; regional requests take **1** parameter | Our 12 parameters fit in one point request per cell |
| Time standard | **LST** is the daily default; `time-standard=UTC` or `LST` can be set explicitly | Request `LST` explicitly and record it |
| Date range | 1981-01-01 to near real time | History starts 1981 (reference.md §2.1) |
| Rate limit | No published numeric limit; HTTP 429 when exceeded; repeated requests for the same relative location may be blocked | Low concurrency, backoff, handle 429 (Phase 2) |
| Grid | Meteorology on MERRA-2's **½° lat × ⅝° lon** grid | District ↔ grid mapping (reference.md §2.2) |
| Latency | GEOS-IT meteorology "generally ready within about two days of real-time"; MERRA-2 "updated every several months" and replaces it | `latency_days: 3` (one day margin); bitemporal raw storage |

**Still to confirm in Phase 2:** the observed daily latency (logged by the ingest job), and whether a point request returns the containing cell's value without interpolation. The previous project's pulls suggested it does: neighbouring districts in the same cell got identical series.
