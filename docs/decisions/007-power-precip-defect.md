# 007: NASA POWER precipitation defect (×24), quarantine rule

**Date:** 2026-10-25
**Type:** operational incident
**Affects:** `configs/data.yaml` `known_defects`; raw store best-known views; daily `validate`

## What happened

From **21 October 2026** the daily job's `validate` step failed every day (5 runs). The ingest step before it kept succeeding, so no data was lost.

Cause: on 21 October NASA POWER revised **precipitation (`PRECTOTCORR`) for every cell, for observation dates 2026-06-23 onward, to about 24× its earlier values**:
- 69 % of the revised values are within 1 % of exactly ×24; September is mixed (median ×11.5).
- POWER was still serving it on 25 October: at one cell, the July mean was 238 mm/day and the maximum 2,700 mm on 25 September. The highest daily value in 46 years of history is 288 mm.
- The first bad batch (21 Oct) carried API version `v2.10.0`; from 22 Oct responses say `v2.10.1`.
- On 21 Oct the other 12 variables were also revised, about 3,000 values each, by modest amounts (e.g. Tmax mean −0.05 °C). That is consistent with the expected **provisional (GEOS-IT) → final (MERRA-2) replacement**, and those revisions are kept.

The system behaved as designed:
- the append-only raw store kept both versions, so the correct rainfall for every date up to 17 October is still there;
- the `validate` hard range check (≤ 600 mm/day) caught the impossible values;
- the job failed loudly rather than passing bad data on.

## Decision (owner)

Quarantine, don't correct. The ratio isn't a clean ×24, so dividing would introduce a different error.

- **Rule** (`configs/data.yaml` `known_defects`): ignore `PRECTOTCORR` versions **ingested from 2026-10-21** while POWER reported **`v2.10.0` or `v2.10.1`**, for **observation dates from 2026-06-23**.
- **Best-known views** use the last good version where one exists. Otherwise the value is **missing (null), never a wrong number**, and the view's `quarantined` column names the rule. Raw rows are never changed.
- **The ingest's change detection ignores the rule**, so bad versions aren't re-appended every day.
- **`validate`** exempts a quarantined variable from the newest-day completeness check and reports `known_defects_active` (408 values on 25 Oct: 18–21 October × 102 cells). A known upstream bug no longer fails the job every day, which would hide new problems.
- **Expiry:** the rule stops applying by itself if POWER ships a fix under a **new** API version. A silent fix under `v2.10.1` needs the rule lifted by hand; the daily `known_defects_active` count shows whether new rainfall is still being quarantined.
- Reporting the bug to NASA POWER: not done (owner's choice).

## Checks after the change

- Cell `merra2_r235_c420`, 2026-09-24: best value 3.96 mm (the original), not 742.08 mm.
- Highest rainfall in the 2026 best view: 265.7 mm. Quarantined (missing) values: 408.
- All 3,148 Tmax revisions from 21 October are kept.
- `heatwave validate` passes; `heatwave qa-report` passes. Its soft flags went from 30 to 26: the four late-September 2026 flags disappeared with the final-data replacement.
- Unit tests: fallback to a good version; missing when none; the rule's narrowness (other variables, earlier dates, a new API version); no re-append of quarantined versions.

## Consequences

- **Development data (1981–2020) is unaffected.** Labels use Tmax only. Rainfall features for June 2026 onward come from good versions or are missing.
- **Before the live season** (decide by mid-December if POWER hasn't fixed it): choose a rainfall fallback for new days, either IMD's real-time rainfall grid or rainfall features left missing. LightGBM uses missing values natively; the logistic model does not use rainfall.
- The 21 October revisions give the first real **provisional-vs-final** pairs for the skew audit (reference.md §5.6).
