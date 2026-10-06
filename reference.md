# UP Heatwave Forecasting — Project Reference

The single guide for this project. Each section records decisions we have agreed on and why. Change a decision only by editing this file and adding an entry to the change log at the bottom.

**Status:** planning complete (7 Oct 2026). Next: Phase 0 (§6.2). V1.0 goes live 1 March 2027.

---

## 1. Purpose and problem definition

### 1.1 What this project is

An **operational heatwave early-warning tool** for Uttar Pradesh: it runs every day by itself, publishes forecasts, and later checks them against what actually happened. It is built to the standard a real user (e.g. a district disaster-management or health office) could rely on. That means automated runs, failure handling, monitoring and honest verification, not only a trained model.

### 1.2 What is predicted

| | Decision |
|---|---|
| Primary target | **Probability that a district has a heatwave day**, for each lead day `h = 1..10`, using an IMD-style heatwave rule (exact rule and data source fixed in the labels section) |
| Headline product | **Probability of at least one heatwave day in the next 10 days**, per district |
| Not the target | Raw Tmax regression. Minimising RMSE says little about extremes, and a warning is a yes/no decision. Tmax may still appear as an internal quantity |

Both targets are probabilities, so they can be checked for calibration (when we say 30 %, it should happen about 30 % of the time).

### 1.3 Forecast horizon and dates

NASA POWER data arrives roughly 2–3 days late, so three dates are kept separate:

| Name | Meaning |
|---|---|
| **As-of date `t`** | Last day with observed data. Every input uses only data up to `t`. |
| **Issue date `s = t + L`** | The day the forecast is produced. `L` = data latency (about 3 days; confirmed from real data). |
| **Valid date `s + h`** | The day being forecast, `h = 1..10`. |

Training applies the same `L` as production. Otherwise a "1-day-ahead" forecast would really be for a day that has already happened.

**Expectation to test, not assume:** without NWP inputs (see §1.5), useful skill probably fades after about 3–5 days of lead. The project measures where it actually fades (the "skill horizon").

### 1.4 Spatial unit

All **75 districts** of Uttar Pradesh. A district's weather is built from the NASA POWER grid cells that cover it, weighted by area, not from one point at its centre. One centre point would give several neighbouring districts identical data.

### 1.5 Version 1 scope

- **V1 uses observations only:** the history of NASA POWER weather variables.
- **NWP (numerical weather prediction)** forecasts are deferred to V2. These are the physics-based weather-model outputs from centres such as ECMWF, NOAA GFS and IMD. V1 measures how far observation-only forecasting goes; that result is the baseline NWP must beat.

### 1.6 Constraints

| Constraint | Value | Consequence for design |
|---|---|---|
| Deadline | None | Phases have no dates; each ends at an exit criterion, not a calendar date |
| Compute | Laptop; Kaggle notebook (T4 GPU) when useful | Data and models must fit on a laptop. Expected models (logistic regression, LightGBM) don't need a GPU |
| Time | 5–6 hours/week | Prefer the simplest tool that works; avoid infrastructure that needs regular maintenance |
| Operational runs | Must not depend on the laptop being on | The daily job will need a free scheduled runner (e.g. GitHub Actions). Decided in the architecture section |

### 1.7 Intended user

A **team of Uttar Pradesh government weather officials.** What follows from that:

- They are domain experts. They will compare our output with IMD bulletins and expect IMD terminology, so labels must say exactly how they differ from an IMD heatwave declaration.
- They need **calibrated probabilities plus alert tiers**, per district and lead day. Tier thresholds will be chosen from out-of-sample results to hit agreed hit-rate / false-alarm trade-offs (validation section).
- They need **evidence of reliability**: published verification (past forecasts vs outcomes), stated skill horizon, and known limitations.
- Output formats (dashboard, map, table, API) are decided in the architecture section.

---

## 2. Data sources and labels

### 2.1 Input weather data

| | Decision |
|---|---|
| Source | **NASA POWER daily** (MERRA-2 reanalysis; recent days from GEOS-IT, later replaced by MERRA-2) |
| History | **1981-01-01 to present** |
| Spatial | MERRA-2 grid (½° lat × ⅝° lon). Fetch **once per grid cell** covering UP, not per district |
| Time standard | Local solar time (POWER's daily default), recorded in metadata |
| Missing values | POWER's fill value (−999) becomes null at ingestion and never reaches a calculation |
| Rejected | ERA5/ERA5-Land (~5-day latency costs lead time); IMD gridded as an input source (coarse, slow updates) |

**Variables (12; POWER allows 20 per request):**

| POWER name | Meaning |
|---|---|
| `T2M_MAX`, `T2M_MIN`, `T2M` | Daily max / min / mean 2 m temperature |
| `T2MDEW` | Dew point |
| `RH2M`, `QV2M` | Relative / specific humidity |
| `PRECTOTCORR` | Precipitation |
| `WS10M`, `WD10M` | Wind speed / direction at 10 m (direction is circular: never average raw degrees) |
| `PS` | Surface pressure |
| `GWETTOP`, `GWETROOT` | Surface / root-zone soil wetness |

Solar radiation is **excluded**: its 5–7 day latency would delay every forecast.

### 2.2 District boundaries and the district ↔ grid mapping

- **Boundaries:** GADM 4.1 level 2 as an interim source, with renames Allahabad → Prayagraj, Faizabad → Ayodhya, Sant Ravi Das Nagar → Bhadohi. Current boundaries are used for all years.
- **Mapping:** intersect each district with the grid cells → area weights summing to 1 per district. Areas are computed in an *equal-area* projection. District value = area-weighted mean of its cells; for Tmax, also keep the maximum across cells.
- Districts that share most of their cells are not independent samples. The number sharing cells is reported.

### 2.3 Raw storage and revisions

- **Raw data is never overwritten.** Every fetched value is stored with its download time (`ingested_at`), so we can reconstruct what was known on any past date.
- The **daily job re-downloads the last ~120 days**, which catches provisional → final replacements.
- **Start the daily download job early,** well before any model exists. Months of provisional-vs-final pairs are needed to measure the gap between training data (final) and live data (provisional).
- Log the observed latency every day (today minus the latest date with data). This confirms or changes `L`.

### 2.4 Heatwave label

**Rule (IMD plains criteria; exact wording verified against IMD's own documentation when implemented):**

```
heatwave day = (Tmax ≥ 40 °C and Tmax − normal ≥ 4.5 °C)  or  Tmax ≥ 45 °C
severe       = (Tmax ≥ 40 °C and Tmax − normal > 6.4 °C)  or  Tmax ≥ 47 °C
```

- This is the **meteorological condition at district level**, not IMD's operational *declaration* (which needs ≥ 2 stations in a subdivision on 2 consecutive days). Every output says so.
- **Normal:** IMD's current period **1991–2020**, as a daily normal smoothed across the calendar (harmonics or ±15-day window), so day-to-day noise cannot create false heatwave days.
- Values are rounded to IMD's reporting precision (0.1 °C) before comparison. A missing Tmax gives a missing label, never a "no".
- Every label row carries a `label_version`: a hash of rule, thresholds, normal period, smoothing, source and any bias correction.

**Which Tmax the rule is applied to is decided by measurement, before any modelling.** IMD's thresholds are meant for station readings; POWER Tmax is a grid-cell average that is likely cooler on extreme days.

1. Download IMD gridded observed Tmax (1°) for 1981 onward.
2. Over the heat season, per district: average bias, bias on the hottest days, correlation of daily anomalies, and agreement of heatwave flags (hits, misses, Cohen's κ).
3. Choose and record the reason:

| Finding | Label source |
|---|---|
| Small bias on hot days, flags agree well | POWER Tmax as-is |
| Clear bias on hot days, but anomalies track well | POWER Tmax **bias-corrected to IMD** (fitted on training years only) |
| Anomalies track poorly | **IMD gridded Tmax** for labels; POWER for inputs only |

A **percentile label** (Tmax above the calendar-day 95th percentile for ≥ 3 days) is also built as a sensitivity check. If conclusions change between the two labels, they are fragile.

**Label QA before modelling:** heatwave days per year, district and month; known severe seasons (e.g. 1998, 2010, 2015, 2019, 2022, 2024) must stand out; contiguous state-wide hot days grouped into **episodes**. The episode count is the real sample size.

---

## 3. Validation and evaluation

### 3.1 Primary metric

**Brier Skill Score against climatology, heat season only:**

```
BS  = mean((p − y)²)
BSS = 1 − BS_model / BS_climatology
```

- **Climatology reference (M0):** the smoothed historical heatwave frequency for that district and calendar day, fitted on training years only.
- Computed **only on valid dates inside the heat season**, per lead day `h` and for the 10-day window target.
- BSS > 0 means information beyond the calendar. The lead at which the BSS interval first includes 0 is the **skill horizon**, reported prominently to users.
- **Heat season:** 1 March – 30 June, provisional. Finalised after the label QA by including any month that holds more than 1% of historical heatwave days.

### 3.2 Data splits (whole calendar years)

| Block | Years | Use |
|---|---|---|
| Development | 1981–2020 | Training, tuning, feature choices, calibration, threshold setting |
| Final test | 2021–2025 | Used **once**, after code, features, parameters, calibrator and thresholds are frozen and git-tagged |
| 2026 season | 2026 | A second one-time check, run only after the final test report is written |
| Live | 2027 onward | Prospective verification |

Calendar-year splits leave a gap of months between training and testing, because the heat season never crosses 1 January. That makes leakage through rolling windows impossible. Peeking at the final test block (even "to check the pipeline runs") burns it; pipeline checks use development years.

### 3.3 Walk-forward folds inside development

| Fold | Train | Validate |
|---|---|---|
| 1 | 1981–2005 | 2006–2008 |
| 2 | 1981–2008 | 2009–2011 |
| 3 | 1981–2011 | 2012–2014 |
| 4 | 1981–2014 | 2015–2017 |
| 5 | 1981–2017 | 2018–2020 |

- 3-year validation blocks, because single years can have almost no positive cases.
- The 15 out-of-fold years (2006–2020) are pooled for headline development metrics and for fitting the calibrator.
- **Everything fitted is refitted per fold on that fold's training years:** climatology, bias correction, scalers, the M0 reference, calibrators.
- Early stopping and tuning use an **inner** block (the last 3 seasons of each fold's training years), never the fold being scored.
- **Also run a sliding 20-year training window** once and compare. Under warming, old years may hurt.

### 3.4 Uncertainty

- All intervals come from a **block bootstrap over years**: resample whole seasons, all districts together. Row-level bootstraps and per-row standard errors are invalid, because rows across districts and days are strongly dependent.
- Model comparisons use the **paired** bootstrap (the same resampled years for both models). A claim that one model beats another needs the interval on the difference to exclude 0.
- The effective sample size is stated in **heatwave episodes**, not rows.

### 3.5 Reported breakdowns

- **Onset subset:** issue dates with no heatwave day in the 3 days up to the as-of date `t`. This is the real forecasting task, reported next to every headline number.
- By lead day, month, year, region, and severity (heatwave vs severe).
- Per district only as a map. Single districts have too few events for separate claims.
- **Supporting metrics:**
  - discrimination: PR-AUC and ROC-AUC, always shown with the climatology and persistence values beside them
  - calibration: reliability diagram, Brier decomposition
  - sharpness: a histogram of forecast probabilities
- **Event-based scores for users:** per episode, whether it was warned and how many days ahead. Also probability of detection (POD), false alarm ratio (FAR), and the critical success index (CSI).

### 3.6 Alert tiers

Rules are fixed **now, before seeing any results**. Probability thresholds are then computed from out-of-fold development predictions and frozen with the model.

| Tier | Threshold rule |
|---|---|
| Watch | Highest threshold that still catches ≥ 80% of heatwave days (POD ≥ 0.8) |
| Alert | Threshold minimising expected cost with **missed heatwave : false alarm = 3 : 1** |
| Warning | Lowest threshold at which ≥ 50% of warnings verify (FAR ≤ 0.5) |

- Users confirmed that **a missed heatwave is worse than a false alarm.** The 3 : 1 ratio is a working assumption; a cost curve over a range of ratios is reported so it can be revised with the officials.
- Thresholds may differ by lead day. Whether they do is decided from the out-of-fold results, not tuned on the test years.

### 3.7 Stop/go checkpoints

| Gate | Condition to pass | If it fails |
|---|---|---|
| G1 — labels | Label passes QA; known severe seasons visible; episode count recorded | Fix the label before any modelling |
| G2 — baselines | Climatology, persistence and logistic models give sensible BSS-by-lead curves | Debug date/latency alignment |
| G3 — challenger (Phase 12) | LightGBM beats logistic regression on the onset subset at some lead, with a paired interval excluding 0 | Keep logistic regression; stop tuning; move effort to NWP (V2) |
| G4 — final test | Skill sign and rough size match development results | Investigate overfitting before any claim to users |

Failing G3 is a legitimate result, not a project failure.

---

## 4. Features and models

### 4.1 Feature principles

1. Every feature is computed only from data dated ≤ the as-of date `t`.
2. **Budget: at most ~40–50 features in V1.** The real sample size is a few hundred episodes; extra correlated features buy variance, not skill.
3. **Prefer anomalies** (departure from the normal for that district and date) to raw values.
4. Every feature group needs a physical reason to exist.
5. Anything climatological (normals, percentiles, historical heatwave frequency) is fitted **per fold on that fold's training years only**.
6. Rolling windows look backwards only, never centred, and use `min_periods` so that gaps give nulls rather than silently shorter windows.

### 4.2 Feature groups

| Group | Features | Physical reason |
|---|---|---|
| Current heat | Tmax, Tmin, their anomalies, Tmax percentile for the calendar day, diurnal range | Distance from the threshold |
| Persistence | Tmax anomaly mean over 3/7/14 d; 7-day max; days above the calendar-day 90th percentile in the last 7/14 d; consecutive hot days; heatwave flag at `t`; days since last heatwave day | Heatwaves come from persistent, blocked patterns |
| Trend | ΔTmax over 1/3/7 d; 7-day Tmax slope | Build-up vs breakdown |
| Land surface | GWETTOP, GWETROOT and their anomalies; rainfall over 7/30/90 d; days since rain > 1 mm | Soil-moisture feedback, the main slow driver |
| Moisture and wind | Dew point, RH, specific humidity; wind speed; wind u/v components; surface-pressure anomaly | Dry north-westerly flow vs moist easterlies; monsoon onset ends the season |
| Season | sin/cos of day-of-year; climatological heatwave frequency for the district and calendar day | The prior |
| Spatial | lat, lon, elevation; state-wide mean Tmax anomaly; upwind (west / north-west) Tmax anomaly | Heat usually advects from the west / north-west |
| Lead | lead day `h`; climatological heatwave frequency at the valid date `s + h` | Lets one model share strength across leads |

**Excluded:** district ID as a category (memorisation), month / week / season dummies alongside the harmonics (redundant), 180/365-day temperature statistics (the anomalies already carry them), solar variables (latency).

Optional, low priority: weekly Niño-3.4 and MJO indices, at most 1–2 features. There are only ~40 seasons, so these have ~40 effective samples. Mind their publication lag.

### 4.3 Training tables

- **Per-lead model:** long format, one row per (district, issue date `s`, lead `h`), with `h` as a feature. One model covers all leads. Ten separate per-lead models are tried later only if needed.
- **Window model:** one row per (district, issue date), as its own model.
- **Rows used:** issue dates whose valid dates fall in **15 Feb – 15 Jul** (season plus shoulders, to see onset and withdrawal).
- **Near-duplicate rows:** neighbouring districts share most grid cells. The share-weighting decision is made when the reference grid tables exist. Either weight rows by cell overlap, or rely on large leaf sizes and claim results at region level only.
- Training tables are versioned by `feature_version` and `label_version`.

### 4.4 Model ladder

| ID | Model | Role |
|---|---|---|
| M0 | **Climatology:** smoothed heatwave frequency by district × calendar day (training years) | Reference for all skill scores |
| M1 | **Persistence lookup:** probability by (heatwave at `t` / current anomaly bin) × lead | The baseline that exposes weak ML |
| M2 | **Logistic regression** on ~6 features: harmonics, Tmax anomaly, 7-d anomaly mean, soil-moisture anomaly, climatological frequency, lead | Interpretable, well calibrated, hard to overfit |
| M3 | **LightGBM**, log-loss objective, full feature set | Challenger, built after the 2027 season (§6.1). Production from 2028 only if it passes G3 |
| M4 (V2) | M3 + NWP features | See To do later |

No neural networks (LSTM/GRU/Transformer) and no random forest in V1. The data volume of independent events doesn't support them, and the ladder above isolates where skill comes from. Everything runs on the laptop; Kaggle is optional.

### 4.5 LightGBM settings

- Regularised by default: `num_leaves ≤ 31`, `min_child_samples ≥ 200`, `feature_fraction ≈ 0.7`, `lambda_l2 > 0`, `learning_rate` 0.03–0.05.
- Early stopping on the **inner** block (last 3 training seasons of the fold), never on the fold being scored.
- **No SMOTE, undersampling, oversampling or class weights.** They distort probabilities, and probabilities are the product.
- Try monotone constraints: probability non-decreasing in the Tmax anomaly and non-increasing in soil moisture.
- Seeds fixed and recorded; final results reported over ≥ 3 seeds.

### 4.6 Tuning

- Start from the defaults above. Optuna, **≤ 40 trials**, objective = in-season log loss on each fold's inner block.
- If the best trial beats the defaults by less than the bootstrap interval width, keep the defaults and record that.

### 4.7 Feature selection

**Group ablation:** drop one feature group, re-run the walk-forward, and record ΔBSS with a paired year-block bootstrap interval. Keep a group only if removing it hurts beyond noise. Importance ranking (e.g. SHAP) is used for explanation, not for selection.

### 4.8 Calibration

- Fitted **only on out-of-fold predictions**.
- **Platt (logistic) scaling**, or beta calibration. Isotonic regression needs more positives than we have and produces step artefacts.
- Calibration checked separately by month and by lead.
- The production calibrator is fitted on all development out-of-fold predictions, then frozen with the model.
- Sanity check: the window probability should be ≥ the largest per-lead probability (approximately). Large violations indicate a bug or bad calibration.

### 4.9 Leakage tests (run automatically on every change)

- **Truncation invariance:** features for date `t` computed from the full table equal features computed from a table cut off at `t`.
- **Future perturbation:** randomising every value dated after `t` changes no feature at `t`.
- Run on random dates and at fold boundaries.
- Every fitted object (climatology, bias correction, calibrator) carries a `fit_end_date` that must be before the start of any period it is applied to.

---

## 5. System architecture

### 5.1 Principles

- **No servers to maintain.** The daily system is a scheduled batch job writing files; everything users see is static.
- **Training and serving are separate.** Training runs on the laptop (or Kaggle) about once a year; the daily job only scores with a frozen model.
- Development and production read **the same stored data**.

### 5.2 Components

| Concern | Choice |
|---|---|
| Daily runner | **GitHub Actions** scheduled workflow (public repo) |
| Persistent data | **Hugging Face dataset repo, private.** Accessed with a token stored as a GitHub Actions secret; the laptop syncs from the same repo. Kept private because IMD data licence terms restrict redistribution |
| Storage format | **Parquet** (partitioned by year where useful) + **DuckDB** for SQL over files. No database server |
| User-facing output | **Static website rebuilt by each daily run, on GitHub Pages**, plus daily CSV/JSON downloads as a read-only API |
| Alerts | Email or Telegram message when any district reaches **warning**; also used for pipeline-failure alerts |
| Code repo | **Public** GitHub repo. No data, tokens or credentials in git |
| Language / env | Python 3.12, **uv** with lockfile |
| Data handling | **pandas**; **pandera** schemas for data checks |
| ML | LightGBM, scikit-learn, Optuna |
| Experiment tracking | **MLflow** with local file store (laptop) |
| Tests / CI | **pytest** + **ruff**, run by GitHub Actions on every push |
| Packaging | uv lockfile pins the daily run's environment; every pipeline step is a CLI command. Dockerfile deferred (To do later) |

Explicitly out of scope: Airflow, Kubernetes, Postgres, Prometheus/Grafana, a feature store, online model serving, Streamlit/FastAPI hosting.

### 5.3 Data layout (on the Hugging Face dataset repo)

| Table | Content |
|---|---|
| `reference/` | districts, grid cells, district–grid weights |
| `raw/power/` | Bitemporal raw values: (cell, date, variable, value, source, `ingested_at`, `run_id`). **Append only, and only values that changed** since the last ingest, so daily re-downloads of the trailing 120 days add little |
| `raw/imd/` | IMD gridded Tmax as downloaded |
| `silver/district_daily/` | Best-known district values (area-weighted), rebuilt from raw |
| `labels/` | Daily labels keyed by `label_version` |
| `features/` | Feature tables keyed by `feature_version` |
| `models/` | Frozen model + calibrator + thresholds per `model_version`, with metadata (git commit, training years, versions, OOF metrics, status) |
| `forecasts/` | One row per (district, issue date, lead 1–10 or 0 = window, model version): probability, raw probability, **climatology probability**, tier, as-of date, data age, run id |
| `outcomes/` | Observed outcome per forecast, written as provisional and again when final data arrives |
| `runs/` | Pipeline run log: start/end, status, rows in/out, latest observed date, error |

Raw POWER history is roughly 20 M values (102 cells × ~16,700 days × 12 variables), likely under 200 MB as Parquet. Check the current Hugging Face storage limits for private datasets during the ingestion phase.

### 5.4 Daily run

1. **ingest**: download the trailing 120 days for every cell; append changed values with `ingested_at`.
2. **validate**: schemas, ranges, completeness, latency. Hard-fail on missing cells or impossible values.
3. **build-district-daily**: best-known, area-weighted district values.
4. **features** at as-of date `t` = latest complete observed date; record data age.
5. **score**: 75 districts × (10 leads + window), frozen calibrator, climatology probability, tier.
6. **publish**: write forecasts; rebuild the static site and CSV/JSON; send alerts.
7. **outcomes**: fill outcomes for forecasts whose valid dates have been observed.
8. **monitor**: checks in §5.6; alert on failure.

### 5.5 Failure policy

| Situation | Behaviour |
|---|---|
| Data age ≤ `L` + 2 days | Forecast with the latest data; show data age |
| Data age > `L` + 2 days | Publish **climatology only**, flagged *degraded*. Never silently reuse yesterday's forecast |
| Validation fails | No model forecast; alert; publish degraded climatology |
| Same day re-run | Overwrites by key; identical inputs must give identical outputs (tested) |

### 5.6 Monitoring and retraining

| Layer | Check |
|---|---|
| Data health (daily) | Cells received, latency, null rate, range violations |
| Feature drift | On **anomaly** features vs the same calendar window in history. Raw-temperature drift tests fire every season change |
| Forecast sanity | Mean probability and tier counts vs the climatological expectation for the date |
| Verification (in-season, weekly) | BSS vs climatology, reliability, hits / false alarms by tier and lead, with event counts shown |
| Train/serve skew (monthly) | Distribution of provisional − final values for Tmax, Tmin and soil moisture, from the bitemporal raw table |

- **End-of-season report every July:** full metrics with bootstrap intervals, an episode-by-episode review, and misses analysed by cause (label, data, genuinely unpredictable).
- **Retraining once a year, after the season.** The challenger is compared with the current model on the same walk-forward plus the newly completed season. It is promoted only if the paired interval on ΔBSS is not negative and calibration is no worse. Criteria are written before the comparison.
- No metric-drop retraining triggers mid-season: with a handful of events a week, changes are noise. Mid-season changes are limited to documented bug fixes.

### 5.7 Repository layout

```
CRO-Heatwave/
├── reference.md              # this guide
├── pyproject.toml  uv.lock  Makefile
├── configs/                  # problem.yaml, data.yaml, features.yaml, model.yaml
├── src/heatwave/
│   ├── ingest/               # POWER, IMD, boundaries
│   ├── geo/                  # grid, district-grid weights
│   ├── quality/              # pandera schemas, checks
│   ├── labels/               # rule, normals, bias correction, episodes
│   ├── features/             # one module per feature group + build
│   ├── models/               # climatology, persistence, logistic, gbm, calibration
│   ├── validation/           # splits, walk-forward, metrics, bootstrap
│   ├── pipeline/             # daily run, outcomes, backfill
│   ├── publish/              # static site, CSV/JSON, alerts
│   ├── monitoring/           # data health, drift, verification
│   └── cli.py                # every pipeline step is a CLI command
├── site/                     # static-site templates
├── tests/                    # unit/, leakage/, data/, e2e/
├── notebooks/                # exploration only; no logic that isn't in src/
├── reports/                  # label QA, baselines, final test, model card, data card
├── docs/decisions/           # one short decision record per significant deviation or choice
└── .github/workflows/        # CI, daily run
```

---

## 6. Phases and schedule

### 6.1 Go-live target

**Fully live on 1 March 2027.** Latest acceptable fallback: **1 April 2027**. March heatwave days are rare in UP; the peak is April–June.

"Fully live" means:
- the forecast is produced every day automatically on GitHub Actions;
- it is published on the website with tiers and alerts;
- outcomes are verified automatically;
- the production model has passed its final test (G4);
- a runbook exists.

**V1.0 production model = M2 (logistic regression)**, calibrated, with frozen tier thresholds. LightGBM (M3) is built after the 2027 season as a **challenger**. It replaces M2 for 2028 only if it passes G3 against M2 at the annual retrain (§5.6). This keeps the 1 March date at 5–6 h/week and respects "no model changes mid-season".

### 6.2 Phase plan

Weeks start on Mondays. Each phase still ends on its exit criterion: if a phase overruns, the slack and then the 1 April fallback absorb it, not skipped checks.

| # | Phase | Weeks | Main work | Exit criterion |
|---|---|---|---|---|
| 0 | Setup and problem freeze | 12–18 Oct 2026 | Repo scaffold (uv, ruff, pytest, CI); `configs/problem.yaml`; verify IMD criteria and POWER API limits from primary sources | CI green; config matches this document |
| 1 | Geography | 19–25 Oct | Boundaries, MERRA-2 grid, district–grid weights, cell-sharing count | Weights sum to 1 per district; tests pass |
| 2 | Ingestion + **daily download job** | 26 Oct – 8 Nov | POWER client (retries, 429, fill values, time standard); 1981→present backfill to Hugging Face; bitemporal raw store; daily GitHub Actions ingest with latency logging | Complete date spine for every cell; daily job ran 7 days unattended |
| 3 | Data quality | 9–15 Nov | pandera schemas; range, continuity and spatial checks; QA report | Every anomaly explained or flagged |
| 4 | Labels (highest risk) | 16 Nov – 6 Dec | IMD gridded download; POWER vs IMD comparison; label-source decision; normals; flags; episodes; percentile label | **G1** passed; label QA report |
| 5 | Features | 7–20 Dec | Feature groups with as-of logic; leakage tests in CI; versioned training tables | Leakage tests green on 20 random dates; feature dictionary |
| 6 | Validation harness + baselines | 21 Dec – 3 Jan 2027 | Splits, walk-forward, metrics, year-block bootstrap; M0, M1, M2 | **G2** passed; baseline report |
| 7a | M2 calibration and tiers | 4–10 Jan | Calibrate M2 on OOF; tier thresholds per §3.6; freeze and git-tag v1.0 | Frozen v1.0 artifact with recorded OOF metrics |
| 8 | Final test | 11–17 Jan | Run 2021–2025 once; then the 2026 season once | **G4** passed; final-test report including what looks bad |
| 9 | Production pipeline | 18 Jan – 7 Feb | Daily scoring; failure policy; outcomes; static site; CSV/JSON; alerts | All daily steps run end-to-end on GitHub Actions |
| 10 | Docs and monitoring | 8–14 Feb | Model card, data card, runbook, monitoring checks, weekly scorecard job | Runbook usable by someone new |
| — | Burn-in (slack) | 15–28 Feb | Pipeline runs unattended; one simulated outage handled per §5.5 | 14 consecutive unattended runs |
| 11 | **Live season** | 1 Mar – 30 Jun 2027 | Weekly scorecards; incident log; documented fixes only | — |
| 12 | Season report + challenger | Jul – Dec 2027 | End-of-season report; **M3 LightGBM** (Phase 7 of the full plan: tuning, ablation, sliding vs expanding window, monotone constraints); annual retrain; champion vs challenger | Promotion decision for 2028, written before the comparison |
| — | V2 | 2028 onward | To do later list, NWP first | — |

### 6.3 Schedule risks

| Risk | Mitigation |
|---|---|
| **Labels (Phase 4)**: IMD data access or a bad POWER–IMD mismatch | Start the IMD download in Phase 2 while the backfill runs. If the comparison is messy, choose the simplest defensible option and record it; refinements go to the 2028 retrain |
| Hugging Face / GitHub Actions friction | Build the daily ingest early (Phase 2), so these problems surface in November, not February |
| Holiday weeks (late Dec – early Jan) | Phases 5–6 are the least risky place to lose time; 2 weeks of slack plus the 1 April fallback |
| 2026 season's final MERRA-2 values not yet available in January | Check the raw table's provisional/final status before Phase 8; if still provisional, run the 2026 check later and record that |

### 6.4 Reports and decision records

- Every gated phase (4, 6, 8) ends with a short report in `reports/`. Phase 12 adds the season report and the challenger report.
- Any change to a decision in this document: edit the section, add a change-log line, and add a short note in `docs/decisions/` if the reason needs more than a line.

---

## To do later (V2 and beyond)

- **Add NWP forecasts as inputs** (statistical post-processing of weather-model output). Must be trained on *archived past forecasts*, not on later observations. Training on observations teaches the model perfect forecasts it will never get in production.
  - Evaluate sources: NOAA GEFS reforecasts, ECMWF open data, Open-Meteo historical forecast archive (archive length, licence, latency).
  - V2 baseline: the raw NWP forecast with the heatwave rule applied. The V2 model must beat that.
- Humid-heat target (heat index / wet-bulb) for eastern UP, which the dry-bulb rule misses.
- Finer-resolution inputs (e.g. ERA5-Land), if their extra latency is worth it.
- Replace GADM boundaries with a current Survey of India–consistent 75-district file.
- Ten separate per-lead models vs the single stacked model, if the stacked model underperforms at specific leads.
- Slow climate indices (Niño-3.4, MJO) as features, if development results justify them.
- **LightGBM challenger (M3)** with Optuna tuning, group ablation, sliding vs expanding window and monotone constraints. Scheduled for Phase 12 (Jul–Dec 2027), promoted for 2028 only via G3.
- Dockerfile for the daily run.

---

## Change log

| Date | Section | Change |
|---|---|---|
| 2026-10-06 | 1 | Purpose, targets, dates, spatial unit, V1 scope and constraints agreed |
| 2026-10-06 | 1.7, 2 | User = UP government weather officials; data sources, variables, boundaries, raw storage and label procedure agreed |
| 2026-10-06 | 3 | Metric, splits, folds, bootstrap, breakdowns, tier rules (miss : false alarm = 3 : 1) and gates agreed |
| 2026-10-06 | 4 | Feature principles and groups, training tables, model ladder (no neural nets in V1), LightGBM settings, tuning, ablation, calibration, leakage tests agreed |
| 2026-10-07 | 5 | Architecture agreed: GitHub Actions daily run, private Hugging Face dataset for data, Parquet + DuckDB, static GitHub Pages site, public code repo |
| 2026-10-07 | 6 | Fully live 1 Mar 2027 (fallback 1 Apr) with M2 as v1.0; LightGBM moved to a post-season challenger; Dockerfile deferred |
