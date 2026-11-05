# 010: Final test result (gate G4)

**Status:** run once on 5 November 2026 with `heatwave final-test --run-once`, on frozen model `m2_v1`, against the bar approved beforehand (decision record 009). **Verdict: PASS WITH CONCERN.** Full tables: `reports/final_test.md`.

The 2021–2025 block is now spent. Nothing may be chosen, tuned or re-tested on it as if it were unseen.

## Result

| Measure | Result | Development range | |
|---|---|---|---|
| Window, all days | 0.103 | 0.055–0.196 | inside |
| Window, onset days | 0.091 | 0.048–0.148 | inside |
| Lead 1, all days | 0.083 | 0.021–0.099 | inside |
| Lead 1, onset days | 0.070 | 0.027–0.083 | inside |
| Watch false alarms | 74.6% | 54.3–73.7% | **just outside** |
| Alert false alarms | 65.2% | 42.9–64.8% | **just outside** |
| Watch caught | 90.5% | 68.1–93.4% | inside |
| Alert caught | 65.5% | 49.3–80.1% | inside |
| Window forecast bias | 5.6 points | up to 6.5 | inside |

Skill sign and size match development, so there is no sign of overfitting. Warning, reported only, catches 50% with 59% false alarms.

## Limits of the result

- **Skill is only clearly real at short range.** Intervals stay above 0 at leads 1–2 on all days and leads 1–3 on onset days. The window's lower bound is +0.0001.
- **Five seasons, two kinds.** 2022 and 2024 had a window heatwave rate of about 26%; 2021, 2023 and 2025 had 2–5%. Alert false alarms were 35–60% in the first group and 85–96% in the second. Skill in 2025 was negative (−0.098).
- **Over-forecast overall:** forecasts averaged about 18% against 12.4% observed, too high at every forecast level.
- **Not perfectly unseen:** yearly label counts for these years were visible earlier (decision record 009). They were not used for any decision.
- **Final data, not provisional:** live forecasts use provisional data. Shadow mode measures that gap.

## Consequences

- `m2_v1` is not changed because of this result.
- Officials should be told plainly that Alerts verify about 35% of the time overall, more than half the time in a big heatwave year and about 1 in 10 or fewer in a quiet one.
- The over-forecasting is being investigated for understanding only.
- Still required before go-live: the daily pipeline, shadow mode, and the one-time 2026 season check.
