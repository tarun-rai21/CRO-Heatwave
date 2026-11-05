# 014: The 2026 season check is the final test (G4) for `m2_v2`

**Status:** decided by the owner, 6 November 2026, including the pass bar and the two rules below. The bar is in `configs/frozen/m2_v2_season_bar.json`, written before the check. The command is `heatwave season-check`. **The check has not been run.**

## Decision

The one-time 2026 season check counts as gate G4 for `m2_v2`. `reference.md` §6.1 requires a passed G4 before going fully live. The 2021–2025 block can't serve, because it was spent on `m2_v1`, and the choice to remove soil moisture was made after seeing it (decision record 013).

## What this test can and can't show

**It's weaker than the test `m2_v1` had:**

- One season instead of five.
- **Not perfectly unseen:** `reports/label_qa.md` lists 2026's heatwave days, and its yearly rate was printed once during review (decision record 011). No decision used them, but anyone setting the bar now knows roughly what kind of season it was.
- It uses final data, so it says nothing about the gap between provisional and final data.

## The first proposed bar doesn't suit one season

`configs/frozen/m2_v2_g4_bar.json`, never approved and not used, has the range across 5-year periods, as `m2_v1` did. Each single development year was scored against it, using out-of-fold predictions and the same rules (2006–2020 only):

| Year | Window heatwave rate | Window, all | Window, onset | Lead 1, all | Lead 1, onset | Verdict |
|---|---|---|---|---|---|---|
| 2006 | 3.4% | −0.265 | −0.227 | −0.062 | −0.014 | FAIL |
| 2007 | 7.8% | −0.125 | −0.040 | −0.010 | +0.048 | FAIL |
| 2008 | 0.0% | −0.190 | −0.190 | +0.522 | +0.522 | FAIL |
| 2009 | 15.2% | +0.083 | +0.090 | +0.101 | +0.083 | Pass with concern |
| 2010 | 42.2% | +0.119 | +0.096 | +0.028 | +0.053 | Pass with concern |
| 2011 | 4.6% | −0.011 | +0.047 | +0.021 | +0.085 | FAIL |
| 2012 | 32.2% | +0.308 | +0.234 | +0.141 | +0.114 | Pass with concern |
| 2013 | 9.8% | +0.113 | +0.140 | +0.116 | +0.086 | Pass with concern |
| 2014 | 26.6% | +0.169 | +0.093 | +0.078 | +0.053 | Pass with concern |
| 2015 | 13.4% | +0.100 | +0.050 | −0.020 | +0.030 | FAIL |
| 2016 | 18.9% | −0.013 | +0.019 | −0.009 | +0.013 | FAIL |
| 2017 | 28.7% | +0.107 | +0.101 | +0.021 | +0.019 | FAIL |
| 2018 | 15.0% | −0.133 | −0.069 | +0.027 | +0.028 | FAIL |
| 2019 | 27.8% | +0.092 | +0.025 | +0.028 | +0.028 | FAIL |
| 2020 | 7.3% | −0.015 | +0.104 | +0.032 | +0.106 | FAIL |

**The model fails this bar in 10 of 15 development years, and no year passes outright.** Window skill is above 0 in only 8 of the 15. So a failure on this bar would say little about the model, and even a pass wouldn't say much. 2008 is an odd case: no window heatwaves at all, so its lead 1 scores rest on almost no events.

## The approved bar

Judge 2026 against the range of **single** development years. It **fails** only if a main skill falls below the lowest year:

| Measure | Fails below |
|---|---|
| Window, all days | −0.265 |
| Window, onset days | −0.227 |
| Lead 1, all days | −0.062 |
| Lead 1, onset days | −0.014 |

The result is reported beside the middle 50% of years, and the Alert false-alarm and bias figures are reported as concerns, as before. The window, all-days middle 50% is −0.070 to +0.110.

**A pass means only "nothing looks broken", not "skill is confirmed".** It's a low bar, deliberately so, because one season can't support a higher one. Reasons to keep it are that it's set from development years only, and doesn't depend on the kind of season 2026 was. Evidence of real skill has to come from the 2006–2020 results and from live seasons.

## Rules

1. **If 2026 fails:** go-live is held and the cause investigated.
2. **If IMD's 2026 grid or final POWER data isn't ready by 1 March:** use the 1 April fallback rather than go live untested. If it's still missing by then, the owner decides again.

## The command

`heatwave season-check --run-once` refuses to run, reading nothing from the season, unless:

- it hasn't run or started before;
- the bar is committed, the code is committed, and the model matches its git tag;
- IMD's 2026 file is present;
- POWER has revised a date after 31 July 2026. POWER doesn't mark values as final, but its replacement batches run forward in date order, so that shows the season is final. This already holds: the 21 October 2026 batch covered September;
- label and feature tables carry the versions the model was fitted on.

If the season's rows are incomplete, it stops without scoring anything, and the run isn't spent.

## Not settled by this

**Nothing in the score uses IMD's 2026 grid.** The label holds its 2021–25 correction for 2026. The grid's value is to check, before scoring, that the label's level still holds for 2026, and no command does that yet: `drift-check` also needs NOAA's 2026 station files, which aren't available. Using the grid to re-estimate the correction would change the label version, and the guard would then refuse the run until this is thought through.
