# 009: Phase 7 model choice (gate G3), training window, calibration and tiers

**Status:** decided by the owner, 5 November 2026. Frozen as `m2_v1` (`configs/frozen/m2_v1.json`, git tag `m2_v1`). The 2021–2025 final test has not been run.

## Summary

- **Gate G3 failed. Logistic regression (M2) goes live**, per reference.md §3.7. It failed with LightGBM's default settings and again after tuning.
- M2 is trained on a **sliding 20-year window**, not on all years.
- M2 gets **Platt calibration at all leads**, with a recorded exception at lead 10.
- **Tiers are issued on the 10-day window only.** Single-day forecasts are shown as plain probabilities.

All scores below are out-of-fold (OOF), 2006–2020, with paired year-block intervals (2,000 resamples, 95%).

## G3: LightGBM against logistic regression

Rule: LightGBM goes live only if, on the onset subset, its skill minus M2's has an interval entirely above 0 at some lead. Result for tuned LightGBM:

| Lead | Difference | 95% interval |
|---|---|---|
| 1 | +0.0115 | −0.0081 to +0.0445 |
| 2 | +0.0134 | −0.0050 to +0.0438 |
| 3 | +0.0082 | −0.0098 to +0.0342 |
| 4 | +0.0058 | −0.0114 to +0.0285 |
| 5 | −0.0000 | −0.0146 to +0.0151 |
| 6 | −0.0052 | −0.0167 to +0.0045 |
| 7 | −0.0069 | −0.0157 to +0.0010 |
| 8 | −0.0075 | −0.0181 to +0.0011 |
| 9 | −0.0069 | −0.0165 to +0.0015 |
| 10 | −0.0084 | −0.0187 to −0.0004 |

No lead passes, and M2 is clearly better at lead 10. LightGBM with default settings also failed (best lower bound −0.0069, at lead 2).

On all days, tuned LightGBM is clearly better at leads 1–3 (about +0.025) and clearly worse at leads 8–10. That gain comes mostly from heatwaves already under way, which G3 deliberately doesn't count. On the 10-day window the two are within noise (0.1087 against 0.0952).

Results use one seed, not the 3 seeds §4.5 asks for in final results. The closest lower bound is well below zero, so more seeds were not run.

## Tuning

Optuna, 40 trials per target, run on Kaggle on CPU. A GPU check on one fold found it only 1.04× faster, with identical results, against a required 1.5×.

| Target | Defaults (inner log loss) | Best | Gain |
|---|---|---|---|
| Lead | 0.14231 | 0.14040 | 1.3% |
| Window | 0.32698 | 0.31852 | 2.6% |

Against the defaults on OOF, tuning cleared zero only at lead 1 on onset days (+0.0046, lower bound +0.0001). Under §4.6 the gain is not above noise, so **the defaults stay** as LightGBM's settings of record. Per G3, tuning stops here. Group ablation and the monotone test were not run, because both exist to improve LightGBM.

### Correction: tuning had seen the test years

I told the owner tuning never used 2006–2020. That was wrong. A fold's inner block is its last 3 training seasons, so fold 2's inner block (2006–2008) is fold 1's validation block, and so on up to 2015–2017. One best trial chosen over all folds has therefore seen most OOF years.

`tuning.nested_choice` fixes this for evaluation: each fold gets the trial that was best on the inner blocks up to its own, all inside its training years. Fold 5's pick is the overall best trial. A smaller leak remains: Optuna's sampler and pruner learned from later folds when choosing which settings to try. It favours LightGBM, which failed anyway.

## Training window

The sliding 20-year window was compared once with expanding training. The bar, set before seeing results, was the same as G3: an interval above 0 on onset days at some lead, and no lead clearly worse.

It passed, **marginally**: only lead 2 clears (+0.0028, lower bound +0.0001), and a different bootstrap seed could change that. Across all 10 onset leads the gain is +0.0023 to +0.0030, with no lead clearly worse. Climatology inputs still come from each fold's own fit. The live model trains on 2001–2020.

## Calibration

Platt scaling, one calibrator per lead, scored with each fold calibrated on the other folds. Beta calibration was also tried; it was not consistently better, so the simpler Platt stays.

- **Bias fixed:** mean forecast 3.0–3.3% before, about 4.3% after, against 4.34% observed.
- **Skill, all days:** +0.007 to +0.009 at leads 1–3 (intervals include 0), and −0.005 to −0.009 at leads 7–10.
- **Skill, onset days:** no lead clearly worse.
- **By month:** July stays over-forecast (3.46% against 1.19% observed).

**Exception, approved by the owner.** The rule set beforehand was "adopt unless it clearly hurts". Lead 10 clearly hurts on all days, by the thinnest margin (−0.0091, upper bound −0.0001). Following it literally would make the mean forecast jump from 4.3% at lead 9 to 3.0% at lead 10, an artefact officials would see. So all leads are calibrated. This treats a margin as thin as the one that let the sliding window pass differently, which is why it's recorded as an exception rather than presented as following the rule.

## Tiers

Rules are unchanged (§3.6). What changed is where they apply. A warning threshold needs at least 500 flagged rows across at least 3 seasons, set before seeing results.

**Single-day forecasts can't support them:** Watch flags 25–38% of district-days with 86–91% false alarms; Alert usually finds that issuing nothing is cheapest; Warning has no valid threshold at any lead.

**The 10-day window can:**

| Tier | Threshold | Caught | False alarms | Held-out folds: caught / false |
|---|---|---|---|---|
| Watch | 15.7% | 80% | 67% | 78% / 67% |
| Alert | 27.2% | 61% | 57% | 60% / 58% |
| Warning | 36.6% | 43% | 50% | 36% / 56% |

These are the frozen thresholds, on the production calibrator's scale. Provisional figures shown to the owner before the freeze used other-fold calibrators (15%, 25%, 41%) and moved to these values; the rules did not change. Held-out Warning misses the 50% target, so officials should expect about 4.4 in 10 Warnings to verify. The 3:1 cost ratio remains a working assumption: on the window, 5:1 catches 80% with 66% false alarms, and 10:1 catches 91% with 72%.

Window against single-day check (§4.8): the window is below the highest single-day forecast in 8% of rows, typically by 0.2 percentage points, at worst by 11.5.

## Consequences

- Frozen with `heatwave freeze-model` at code `425953c`, fitted on 2001–2020 (features `ft_e13be4154d21`, labels `lbl_9105574cbb3a`). The tables behind this record are generated in `reports/phase7_model.md`. The window climatology table is in the private dataset (`models/frozen/`), checked against the hash in the JSON.
- Next: the single final test on 2021–2025 (gate G4), run with `heatwave final-test --run-once`.

## Gate G4 pass bar (approved by the owner before the test)

Stored in `configs/frozen/m2_v1_g4_bar.json`, computed from the 11 five-year development periods in 2006–2020 only, with other-fold calibrators for skill.

**Pass:** all four main skills are above 0 and at or above the lowest development period: window all days 0.055, window onset 0.048, lead 1 all days 0.021, lead 1 onset 0.027.

**Pass with concern:** a main skill above the highest period (possible leak); a Watch or Alert hit or false-alarm rate outside its range; or a window forecast bias above 6.5 percentage points. Warning is reported only, since it already misses its target in development.

**Fail:** the model does not change and those years are not used again. Investigate before any claim to users (§3.7).

Limits, stated before the result:

- The bar is low: it asks whether skill is no worse than the worst stretch seen, not whether it is strong.
- Skill fell through the 2010s, and 2016–2020 sits close to the floor (0.058 against 0.055), so a chance failure is possible.
- The final-test years were not perfectly unseen. `reports/label_qa.md` lists their heatwave district-days (2021: 70, 2022: 655, 2023: 89, 2024: 1,123, 2025: 31) and `notebooks/explore.ipynb` charts yearly rates. These were not used to set the bar or make any decision. 2022 and 2024 will dominate the result.
- The test uses final MERRA-2 data, while live forecasts use provisional data. Shadow mode measures that gap.
- Other model families (XGBoost, random forest) and NWP inputs are on the To do later list. New candidates should not be judged only by repeated looks at 2006–2020.
