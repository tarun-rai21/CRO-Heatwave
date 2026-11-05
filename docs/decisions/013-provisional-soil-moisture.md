# 013: Provisional soil moisture does not match the data the model was trained on

**Status:** found 6 November 2026 while building the monitoring. **Decided by the owner: remove soil moisture from the live model.** Frozen as `m2_v2` (`configs/frozen/m2_v2.json`, git tag `m2_v2`), which replaces `m2_v1` in the daily run. `m2_v1` is unchanged and stays tagged as the record. The 2026 season check is still to come.

## Finding

The model trains on final POWER data (MERRA-2), and forecasts live on provisional data (GEOS-IT), which MERRA-2 later replaces. For root-zone soil moisture, `GWETROOT`, the two differ a lot. `gwetroot_anom` is the model's strongest weather input.

One revision batch so far, 21 October 2026, covers observation dates 1 September to about 4 October, roughly 3,000 cell-days:

| Variable | Mean (final − provisional) | SD | 5th to 95th percentile |
|---|---|---|---|
| `GWETROOT` | +0.13 | 0.10 | −0.03 to +0.28 |
| `GWETTOP` | +0.10 | 0.10 | −0.03 to +0.31 |
| `T2M_MAX` | −0.05 °C | 1.5 °C | −2.5 to +2.7 °C |
| `T2M_MIN` | −0.07 °C | 0.65 °C | −1.1 to +1.0 °C |

(Values for all cells; see `heatwave monitor-monthly`.)

## Is it real, or the bad rainfall batch?

The same batch carried the ×24 rainfall defect (decision record 007), so this was checked. It looks real:

- Final values run on smoothly from data not touched by that batch: 0.79 on 29 August, 0.82 on 1 September.
- September's final level (0.84) is within the historical range: 0.86 in 2018, 0.84 in 2025.
- The provisional series steps down where it takes over: 0.87 on 28 September (final), 0.72 on 1 October (provisional). On 1 September, the first stored provisional value was 0.66 against 0.82 final.

## How much it matters

The spent 2021–2025 years, re-scored with `gwetroot_anom` shifted down. This is a sensitivity test, not a result:

| Shift | Mean window forecast (observed 16.8%) | Window skill | Share of district-days at Warning | Warnings that were false |
|---|---|---|---|---|
| 0 | 19% | 0.130 | 17% | 53% |
| −0.02 | 24% | 0.084 | 24% | 59% |
| −0.05 | 32% | −0.100 | 40% | 66% |
| −0.08 | 41% | −0.445 | 56% | 72% |
| −0.13 | 57% | −1.326 | 74% | 78% |

If a gap the size of September's holds in season, the forecast would be far worse than climatology, with most of the state at Warning most of the time. A gap of about 0.03–0.04 would already remove the window's skill.

The final test (decision record 010) used final data, so it does not cover this.

## What is not known

- **The gap in March–July.** All evidence is from September, just after the monsoon, with wet soils. It may be much smaller when soils are dry, or not. Provisional versions were not stored before October 2026, and POWER serves only its current best value, so past springs cannot be measured.
- **Whether it is stable.** There is one batch. Later batches will add October onward, but no spring data can arrive before go-live.

## Options

1. **Remove soil moisture from the live model.** Refit M2 without `gwetroot_anom`. Safest against this problem, at some cost in skill that can be measured on development data now. Needs a new freeze (`m2_v2`) and new thresholds.
2. **Correct provisional soil moisture before scoring**, using the gap measured from recent revisions. Keeps `m2_v1`, but rests on a correction that can only be measured months late and has never been seen in spring.
3. **Keep `m2_v1` and watch it in shadow mode.** Probabilities start on 15 February, but MERRA-2 replaces provisional data only after months, so the answer would arrive after the go-live date. Not recommended on its own.

## Now in place

- The public site shows no 10-day probabilities until mid-February, so nothing misleading is published in the meantime.
- `heatwave monitor-monthly` still measures and stores the soil moisture gap each month, for the 2027 retrain, but no longer flags it or emails about it, because `m2_v2` doesn't use soil moisture. It flags the Tmax gap instead (below).
- Daily feature-drift checks watch `tmax_anom` and `tmax_anom_mean7` in season, not `gwetroot_anom`.

## Decision and result

Option 1. The limit was set before the comparison ran: the cost is modest if window skill falls by less than a quarter and stays clearly above 0, on all days and on onset days.

Same folds, 2006–2020, other-fold calibrators. Full tables: `reports/model_m2_v2.md`.

| Window skill | With soil | Without soil | Fall | Difference (95% interval) |
|---|---|---|---|---|
| All days | 0.100 | 0.086 | 14% | −0.014 (−0.028 to −0.001) |
| Onset days | 0.084 | 0.066 | 22% | −0.018 (−0.031 to −0.003) |

Both pass the limit, but the loss is real: both intervals are below 0. Single-day skill at leads 8–10 falls to about 0.001 on all days, effectively none.

**`m2_v2`:** the same logistic model and 2001–2020 training window, on `doy_sin`, `doy_cos`, `tmax_anom`, `tmax_anom_mean7` and the climatology logit, fitted at code `335b579`.

| Tier (10-day window) | Threshold | Caught | False alarms | Held-out folds: caught / false |
|---|---|---|---|---|
| Watch | 14.1% | 80% | 68% | 78% / 69% |
| Alert | 27.4% | 56% | 55% | 55% / 57% |
| Warning | 34.7% | 43% | 50% | 35% / 57% |

## Provisional temperature was checked first

Tmax also differs between provisional and final data. At district level (September–October 2026, 2,325 district-days): mean +0.08 °C, SD 1.35 °C; the 7-day mean has SD 0.89 °C; the state-mean gap persists from day to day (autocorrelation 0.79). A no-soil model fitted the same way, re-scored on 2021–2025 with errors of that size added: window skill 0.159 with none, 0.138 to 0.160 with a steady shift of up to ±0.8 °C, 0.146 with daily noise of SD 1.35 °C. This did not justify stopping. `heatwave monitor-monthly` flags a Tmax gap above 0.5 °C.

## Evidence for `m2_v2`, and its limits

- **There is no clean test.** The G4 result in decision record 010 belongs to `m2_v1`. Re-scoring `m2_v2` on 2021–2025 (`reports/final_test_m2_v2_info.md`) gives window skill 0.159, onset 0.121, Alerts true 46%. This is information only: the years are spent, the label's post-2020 correction was fitted on them, and the choice to remove soil moisture was made after seeing them. That this beats `m2_v1` there (0.130) does not show that removing soil moisture helps; on development years it hurts.
- **The 2026 season check is the only fresh evidence left**, and it is not perfectly unseen (decision record 011). Its bar is proposed in `configs/frozen/m2_v2_g4_bar.json` by the same method as `m2_v1` and is not yet approved.
- **If the spring soil moisture gap turns out small,** this will have given up some skill for nothing. It can be revisited at the 2027 retrain with real spring gaps, which the bitemporal store now records.
