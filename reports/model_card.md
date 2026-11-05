# Model card: `m2_v2`

Written 6 November 2026. Generated figures: [`model_m2_v2.md`](model_m2_v2.md). Reasoning: [decision record 013](../docs/decisions/013-provisional-soil-moisture.md). Data: [`data_card.md`](data_card.md).

## In brief

- Estimates each UP district's daily chance of a heatwave day, and its chance of at least one in the next 10 days.
- **Modest skill, and clearly real only at short range.** It knows recent temperature and the calendar, and nothing about what the atmosphere will do next.
- **It has not had a clean test.** The only clean test (G4) was of `m2_v1`, an earlier version with soil moisture. Fresh evidence for this version can come only from the 2026 season check, not yet run.
- **Not an official forecast or warning.** IMD issues those.

## Use

| | |
|---|---|
| **Built for** | UP government weather officials, as one extra input alongside IMD products |
| **Good for** | A rough view of which districts carry a raised 10-day risk, compared with the usual chance for the date |
| **Not for** | Safety decisions by itself, replacing IMD warnings, single-district claims, single-day planning beyond about 5 days, or use outside UP or 15 Feb–15 Aug |

## What it is

Logistic regression, one model for each lead day 1–10 and one for the 10-day window, each followed by a Platt calibrator.

**Inputs:** `doy_sin`, `doy_cos` (time of year), `tmax_anom` (latest Tmax against normal), `tmax_anom_mean7` (its 7-day mean), and the logit of the climatological heatwave frequency. All are known by the as-of date, 3 days before the issue date.

**Fit:** 2001–2020, 15 Feb–15 Aug rows. Inputs are standardised, L2 with C = 1. Code `335b579`, git tag `m2_v2`. Features `ft_14d3596786eb`, label `lbl_d8b1d4c8200b`.

**Why this model:**

- LightGBM, default and tuned, failed to beat logistic regression under the rule set beforehand (decision record 009).
- A 20-year sliding window was marginally better than using all years.
- Soil moisture was removed because provisional values sit far below the final values the model would have trained on (decision record 013). This cost real skill on development years.

## Output

| | |
|---|---|
| Daily | Chance per district, for 10 single days and the 10-day window, with the usual chance beside it |
| Tiers | On the window only, when all 10 days lie in March–July. Single-day forecasts can't support tiers |
| When blank | Outside the training window, or on a degraded day |

| Tier | Threshold | Rule, fixed before results | Caught | False alarms | Held out: caught / false |
|---|---|---|---|---|---|
| Watch | 14.1% | Catch at least 80% | 80% | 68% | 78% / 69% |
| Alert | 27.4% | Miss : false alarm = 3 : 1 | 56% | 55% | 55% / 57% |
| Warning | 34.7% | At most 50% false alarms | 43% | 50% | 35% / 57% |

Out-of-fold, 2006–2020, on the window target. "Held out" uses thresholds fitted on other folds, so it's the fairer figure: **the Warning tier's 50% was met by construction and is about 57% held out.** The 3 : 1 ratio is a working assumption to review with the officials.

Tiers are this project's own and aren't IMD categories. The target is a modelled condition at district level, not an IMD declaration, which needs at least 2 stations in a subdivision on 2 consecutive days.

## How good it is

Skill is the Brier skill score against the usual chance for the district and date. 0 means no better than that; 0.086 means an 8.6% lower Brier score. Intervals are 95%, from resampling whole years. Onset days are those with no heatwave day in the 3 days before the as-of date, which is the real forecasting task.

### Evidence, strongest first

| Evidence | Period | Limit |
|---|---|---|
| Development, out-of-fold | 2006–2020 | Used to choose the model and thresholds, so somewhat optimistic |
| Re-score | 2021–2025 | **Not a test.** Years spent, label correction fitted on them, and soil moisture was removed after seeing them |
| Clean final test, G4 | 2021–2025 | **Of `m2_v1`, not this version,** on an older label |
| 2026 season check | 2026 | Not run. Bar proposed in `configs/frozen/m2_v2_g4_bar.json`, not yet approved |

All of these use **final** data. Live forecasts use provisional data and may do worse.

### Development, 2006–2020

| Window | Skill | 95% interval | With soil moisture |
|---|---|---|---|
| All days | 0.086 | 0.016 to 0.145 | 0.100 |
| Onset days | 0.066 | 0.019 to 0.103 | 0.084 |

| Lead (days) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| All days | .056 | .034 | .020 | .015 | .014 | .010 | .004 | .001 | .000 | .002 |
| Lower bound | +.024 | +.006 | −.005 | −.010 | −.015 | −.017 | −.020 | −.022 | −.022 | −.020 |
| Onset days | .055 | .041 | .032 | .025 | .021 | .014 | .007 | .003 | .002 | .005 |
| Lower bound | +.038 | +.025 | +.018 | +.012 | +.006 | −.002 | −.009 | −.012 | −.012 | −.010 |

**Skill horizon:** the interval stays above 0 through lead 2 on all days and lead 5 on onset days. After lead 7, single-day forecasts are effectively the usual chance. A lead counts days after the issue date.

### Re-score, 2021–2025: information only

Window skill 0.159 (0.041 to 0.206); onset 0.121 (0.048 to 0.159). Lead 1 skill, 0.106, sits above the top of the development range, which is more likely luck or circularity than a real improvement. **That these beat `m2_v1`'s 0.130 doesn't show that removing soil moisture helps. On development years it hurts.**

| Year | Observed | Mean forecast | Skill | Alert caught | Alert false |
|---|---|---|---|---|---|
| 2021 | 8.0% | 14.2% | 0.177 | 18% | 85% |
| 2022 | 36.7% | 26.1% | 0.193 | 62% | 41% |
| 2023 | 5.8% | 14.5% | 0.018 | 73% | 75% |
| 2024 | 27.3% | 24.9% | 0.235 | 82% | 31% |
| 2025 | 6.1% | 17.0% | −0.022 | 29% | 90% |

**Five seasons, two kinds.** Alerts came true more than half the time in big years and about 1 in 4 or fewer in quiet ones. By region, the window skill runs from 0.125 to 0.190, and Bundelkhand's interval includes 0 (−0.127 to 0.275).

### Clean test of `m2_v1`, for context

PASS WITH CONCERN: window skill 0.103, onset 0.091, inside the development range, with Watch and Alert false alarms just outside theirs (decision record 010). This doesn't transfer to `m2_v2`.

### Calibration

Forecast levels are roughly right on average, but not in every kind of year or month.

| Mean, at lead level | March | April | May | June | July |
|---|---|---|---|---|---|
| Observed | 1.08% | 2.53% | 6.47% | 10.56% | 1.19% |
| Forecast | 1.04% | 1.90% | 5.93% | 8.85% | 3.83% |

- **Too high in July,** when the monsoon's arrival ends the heat, and in quiet years.
- **Too low in April to June,** and in big years such as 2022.
- **Window forecasts on 2021–25 by level:** forecasts of about 10% happened about 4% of the time; about 16%, about 11%; about 37%, about 39%; about 61%, about 56%. So the low to middle range runs high.

## Known limitations

- **It can't see ahead.** There's no weather-model input. A heatwave arriving after a cool spell is largely invisible until it shows up in observed temperature. Adding one is the main V2 work.
- **It can't foresee a quiet season,** so it over-forecasts in those years.
- **Data is 3 days old** on the issue date, so lead 1 is really 4 days after the newest observation.
- **Provisional data.** Tmax errors had little effect in a sensitivity test, but that used September–October gaps. Spring gaps are unknown until 2027.
- **Soil moisture may have been removed for nothing** if the spring gap proves small. This can be revisited at the 2027 retrain.
- **Few independent events.** Rows are many, but districts move together, and a whole season may hold only a few heatwaves. 232 state-wide episodes in development years, measured before the 2025 drift update. Don't read much into a single district or a single season.
- **Window and single days can disagree slightly.** They're separate models. The window was below the highest single-day figure in 6.3% of rows, by 0.003 on average and 0.100 at most.
- **Label limits** carry through: see the [data card](data_card.md).

## Operation

- **Frozen.** It doesn't learn from live data. Coefficients are in `configs/frozen/m2_v2.json`; the climatology table is on Hugging Face and checked by hash.
- **Late data:** 1 or 2 days late gives 9 or 8 valid days, flagged. Older gives climatology only. Fresher data than the trained delay is ignored.
- **Monitoring:** daily data, coverage and range checks; forecast and input-drift checks in season, with limits fixed from development years; monthly provisional-against-final check; public scorecard. See the [runbook](../docs/runbook.md).
- **Changes:** yearly retrain after the season. A challenger is promoted only if the paired interval on skill difference isn't negative and calibration is no worse, with criteria written first. In season, only documented bug fixes.

## Before go-live

`reference.md` §6.1 requires the production model to have passed G4 before going fully live. **`m2_v2` hasn't,** and 2021–2025 can't be reused. Open for the owner: whether the one-time 2026 season check counts as this model's G4, on what bar, and what happens if IMD's 2026 grid isn't out by 1 March.

## Versions

| Version | Date | Change | State |
|---|---|---|---|
| `m2_v1` | 5 Nov 2026 | Logistic with soil moisture. Tiers 15.7%, 27.2%, 36.6% | Tagged, kept as the record. Not to be used live |
| `m2_v2` | 6 Nov 2026 | Soil moisture removed. Tiers 14.1%, 27.4%, 34.7% | Live, in shadow mode |
