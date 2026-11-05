# 013: Provisional soil moisture does not match the data the model was trained on

**Status:** found 6 November 2026 while building the monitoring. **Open: no fix decided.** It blocks go-live until resolved. Nothing has been changed in `m2_v1`.

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

- The public site shows no probabilities until 15 February, so nothing misleading is published in the meantime.
- `heatwave monitor-monthly` flags a soil moisture gap above 0.02 and emails the owner. Daily feature-drift checks also watch `gwetroot_anom` in season.
