# 011: Label drift term updated through 2025

**Status:** decided by the owner, 5 November 2026. Label `lbl_9105574cbb3a` becomes `lbl_d8b1d4c8200b`; features `ft_e13be4154d21` become `ft_14d3596786eb`. **Frozen model `m2_v1` is unchanged**, and its JSON still names the versions it was fitted on. The G4 verdict is unchanged (decision record 010).

## Why

Decision record 005 assumed POWER's bias stays at its 2016–2020 level after 2020 (−0.37 °C) and required a yearly re-estimate. Decision record 010 found label Tmax about 0.5 °C cooler against stations in 2021–25 than in 2016–20. POWER − IMD beyond the plane averaged −0.79 °C over 2021–25. It moved abruptly around 2020 (−0.13 in 2019, −0.94 in 2020) rather than gradually. The cause is not known.

## What changed

- Drift term extended through 2025 with the same method: yearly POWER − IMD beyond the plane, centred 5-year smoothing (`bias_drift_update_to: 2025`).
- Values through 2020 are kept exactly as fitted. Simply extending the years would have changed 2019–2020, because a centred window would pull later years into them, and so would have changed development labels.
- Years after 2025 use the 2021–25 mean, −0.79 °C (`bias_drift_hold_years`).
- Later station data stays in its own folder, which the label build does not read.
- New command: `heatwave drift-check` (`--download` fetches the station years), writing `reports/label_drift_check.md`. It is meant to run yearly, when IMD publishes.

## Development years are unchanged (verified against a backup)

- Labels through 2020: **0 differences** in any column, across all 1981–2020 rows.
- Plane, station map and drift values through 2020: identical.
- All five fold tables and their climatologies: identical apart from the version column.
- Development features through 2020 and the development climatology: identical.

So `m2_v1`'s coefficients, calibrators and thresholds would come out the same if refitted, and were not refitted.

## After the update

Label minus stations, change from 2016–20 to 2021–25:

| | Before | After | 95% interval after | Stations cooler after |
|---|---|---|---|---|
| Label | −0.54 °C | −0.15 °C | −0.49 to +0.21 | 6 of 9 |

Heatwave share of March–July district-days:

| Year | Old label | New label | IMD-based label |
|---|---|---|---|
| 2021 | 0.61% | 1.14% | 0.86% |
| 2022 | 5.71% | 10.01% | 2.69% |
| 2023 | 0.78% | 1.33% | 0.62% |
| 2024 | 9.79% | 10.98% | 7.73% |
| 2025 | 0.27% | 0.85% | 0.24% |

**2021–2025 re-scored, for information only** (`reports/final_test_label_v2_info.md`). This is not a test: the years are spent, and the drift was fitted on them, so the improvement is partly circular.

| Measure | Old label | New label |
|---|---|---|
| Window skill, all days | 0.103 | 0.130 |
| Window skill, onset days | 0.091 | 0.092 |
| Lead 1 skill, all days | 0.083 | 0.107 |
| Window forecast bias | 5.6 points | 2.3 points |
| Alert caught / false | 65% / 65% | 61% / 59% |
| Warning caught / false | 50% / 59% | 46% / 53% |

## Open problems

- **2022 looks over-corrected.** Smoothing applies −0.84 °C to a year whose own value is −0.52 °C. Its heatwave share nearly doubled and the forecast now falls well short of it (24.6% against 36.7% on the window). The new label is also well above the IMD-based label in 2022, though that label usually runs below this one.
- **A step, not a slope.** If the shift near 2020 is a step, a smoothed term handles the years around it poorly, as 2021 and 2022 show. The cause should be looked for.
- **Quiet years remain.** Window forecasts in 2021, 2023 and 2025 still run 6–11 points too high, and Alert false alarms are 77–91% in those years.
- **Hold value.** −0.79 °C is again an assumption for 2026 onward, and POWER may move again. Live monitoring against IMD and stations stays essential (§5.6).
- **The 2026 season check is not perfectly unseen.** `reports/label_qa.md` has listed its heatwave district-days, and the review for this record printed its yearly rate (3.5% old label, 6.26% new). No decision used them. The year is incomplete and still on provisional data.
