# 006: Phase 5 features

**Date:** 2026-10-20
**Phase:** 5
**Affects:** reference.md §4.2–4.3; `configs/features.yaml`
**Generated:** `docs/features.md` (feature dictionary), `data/features/` (local, rebuildable)

## What was built

- **43 features** (41 as-of plus 2 lead features), budget 40–50, in the groups of reference.md §4.2. The list and statistics are generated into `docs/features.md` from the registry in `src/heatwave/features/build.py`.
- **As-of logic:** every feature for (district, t) uses data dated ≤ t. Rolling windows look backwards only, with ≥ 70 % non-missing values required, and cross-district features use the same day t. The issue date is s = t + 3 and the valid date s + h.
- **Fold-aware climatology:** normals (Tmax, Tmin, soil wetness, pressure), the hot-day threshold (90th percentile of the Tmax anomaly per district and month) and the calendar-day heatwave frequency are fitted on each fold's **training years only**. Each carries its fit years. `check_held_out` refuses to build a block whose held-out years overlap them.
- **Blocks:** `fold1`–`fold5` (climatology fitted on the fold's training years) and `development` (fitted on 1981–2020; held out: 2021–2025). Each block's table holds the as-of dates whose issue date has a valid date in the training window (15 Feb – 15 Aug). **A fold's table stops at its last validation year**, so no fold-level work can touch final-test rows. Climatologies are saved per block for reuse in live scoring.
- **Targets** (`features/targets.py`):
  - per-lead table: one row per (district, s, h), y = label at s + h;
  - window table: one row per (district, s), y = any heatwave at s+1..s+10, only when all ten valid dates lie in the training window.
  - Rows with unknown targets are dropped, never filled. Both tables are assembled at training time.
- **Versioning:** `feature_version` hashes the feature list, `features.yaml`, the feature code and the `label_version`. It is stored on every row and in `data/features/manifest.json`.

| Block | Climatology years | Held-out years | Rows | As-of range |
|---|---|---|---|---|
| fold1 | 1981–2005 | 2006–2008 | 401,625 | 1981-02-02 – 2008-08-11 |
| fold2 | 1981–2008 | 2009–2011 | 444,600 | – 2011-08-11 |
| fold3 | 1981–2011 | 2012–2014 | 487,650 | – 2014-08-11 |
| fold4 | 1981–2014 | 2015–2017 | 530,700 | – 2017-08-11 |
| fold5 | 1981–2017 | 2018–2020 | 573,750 | – 2020-08-11 |
| development | 1981–2020 | 2021–2025 | 659,775 | – 2026-08-11 |

## Leakage tests (exit criterion: met)

| Test | Where | Result |
|---|---|---|
| Truncation invariance: features at t identical with the future deleted | Synthetic, 10 dates incl. both sides of a fold boundary (CI) | pass |
| Future perturbation: scramble every value and flip every label after t | Synthetic, 10 dates (CI) | pass |
| Both of the above | **Real data, 20 random dates** 1982–2020 (`pytest -m slow`) | **pass (42/42, incl. the two checks below)** |
| Saved table = rebuild from the saved climatology | Real data | pass |
| Fold tables end at their validation years; climatology fit ends before validation | Real data | pass |
| Climatology refuses its own fit years; ignores years outside the fit | Unit | pass |
| Targets: y at s + h with s = t + 3; window rows need all ten valid dates in the window | Unit | pass |

## Findings and choices

- **The label drift problem surfaced here.** A May anomaly of −0.66 °C against a 1981–2005 climatology led to the drift term (decision record 005 addendum). After it, the development-year mean Tmax anomaly is about 0.006 °C.
- `upwind_tmax_anom` is missing for the westernmost districts (no UP district to their west within 250 km): about 8 % of rows. LightGBM uses missing values natively; the logistic baseline needs an explicit fill (Phase 6). Neighbours outside UP come with the Phase B spatial expansion.
- A separate "Tmax percentile for the calendar day" (§4.2) was not built. The anomaly plus the hot-day counts carry the same information, and the budget favours fewer correlated features.
- Missing values only occur in the first days of the record (rolling warm-up). The POWER data has no gaps.

## Near-duplicate rows

reference.md §4.3 deferred this choice to "when the grid tables exist". After area weighting no two districts share identical inputs; the highest overlap is 0.68. **Decision (owner): no row weighting in V1.** Rely on regularised models (≥ 200 rows per leaf), year-block bootstrap intervals, and region-level claims. Weighting by overlap stays an option if Phase 6 results suggest it.
