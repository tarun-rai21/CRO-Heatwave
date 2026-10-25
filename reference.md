# UP Heatwave Forecasting — Project Reference

The single guide for this project. Each section records decisions we have agreed on and why. Change a decision only by editing this file and adding an entry to the change log at the bottom.

**Status:** Phases 0–6 complete (25 Oct 2026). Next: Phase 7 (§6.2). Goes live 1 March 2027 after shadow mode from mid-December.

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
| **Issue date `s = t + L`** | The day the forecast is produced. **`L = 3`**: measured latency was 3 days on 11 of the first 13 daily runs and 4 days on 2 (decision record 003). |
| **Valid date `s + h`** | The day being forecast, `h = 1..10`. |

Training applies the same `L` as production. Otherwise a "1-day-ahead" forecast would really be for a day that has already happened.

**Late-data days** (newest data is `s − 4` instead of `s − 3`): the model is still run exactly as trained, from as-of date `t = s − 4`. Its output is therefore anchored one day earlier, and its last lead falls on `s + 9`. The published forecast covers valid dates `s + 1 .. s + 9` (9 days), marked **"data 1 day late"**. No re-training or re-labelling of leads is done for these days. Older data than that follows the failure policy (§5.5).

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
| Label references (not model inputs) | IMD 1° gridded Tmax (1981–2025) and NOAA GSOD station daily maxima for UP (development years 1981–2020 only), used to bias-correct and validate the label (§2.4) |

**Variables (13; POWER allows 20 per request):**

| POWER name | Meaning |
|---|---|
| `T2M_MAX`, `T2M_MIN`, `T2M` | Daily max / min / mean 2 m temperature |
| `T2MDEW` | Dew point |
| `RH2M`, `QV2M` | Relative / specific humidity |
| `PRECTOTCORR` | Precipitation |
| `WS10M` | Wind speed at 10 m |
| `U10M`, `V10M` | Eastward / northward wind at 10 m. Replace `WD10M` (direction), which POWER serves with invalid values for every cell from 2012-12-31 onward (decision record 004). Direction, if needed, is derived from U/V |
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
- **Known upstream defects** are quarantined by rule (`configs/data.yaml` `known_defects`), never corrected in place. Best-known views fall back to the last good version, else return the value as missing; `validate` reports active defects instead of failing on them (decision record 007: POWER rainfall ×24 from 21 Oct 2026).

### 2.4 Heatwave label

**Rule (IMD plains criteria, verified against IMD's *FAQ on Heat Wave*, decision record 001):**

```
heatwave day = (Tmax ≥ 40 °C and Tmax − normal ≥ 4.5 °C)  or  Tmax ≥ 45 °C
severe       = (Tmax ≥ 40 °C and Tmax − normal > 6.4 °C)  or  Tmax ≥ 47 °C
```

- This is the **meteorological condition at district level**, not IMD's operational *declaration* (which needs ≥ 2 stations in a subdivision on 2 consecutive days). Every output says so.
- **Normal:** IMD's current period **1991–2020**, a daily normal from a **6-harmonic** fit of the annual cycle. It was the best of the smoothings tried in an out-of-sample test, though all were within 0.02 °C RMSE of each other.
- Values are rounded to IMD's reporting precision (0.1 °C) before comparison. A missing Tmax gives a missing label, never a "no".
- Every label row carries a `label_version`: a hash of the rule, thresholds, normal, source, and the fitted bias map and spatial plane.
- All 75 districts use the plains criteria (assumption A1 checked: the highest area-weighted district elevation is far below hill-station levels).

**Tmax source: decided by measurement in Phase 4 (decision record 005).** Development years only (1981–2020).

| Finding | Evidence |
|---|---|
| POWER runs warm | +1.8 to +2.7 °C against UP weather stations (NOAA GSOD reported maxima); about 3× too many heatwave days |
| IMD's 1° grid runs cool on hot days | −0.4 to −2.3 °C on average and worse on the hottest days; about 43 % too few heatwave days |
| POWER's warm bias is smaller over Bundelkhand | about 1.1 °C smaller against IMD, and about 1.9 °C smaller at Jhansi station. Consistent with irrigation cooling the Ganga plains in reality but not in the reanalysis |

**Chosen label Tmax:** POWER district Tmax (area-weighted), **minus a smooth spatial bias plane** (latitude/longitude fit of the district mean POWER − IMD over the evaluation season) **and a yearly drift term**, then **quantile-mapped by month to station levels**. The drift term exists because POWER's warm bias shrinks over the record (about +3.3 °C against stations in the 1980s, +2.2 °C after 2005) while IMD against stations stays flat. It is the season mean of POWER − IMD − plane, 5-year smoothed, estimated 1981–2020; later years use the 2016–2020 mean. The map is fitted on the stations with at least 10 good seasons (Lucknow, Prayagraj, Bareilly, Gorakhpur). The plane and map are fitted **once on 1991–2020** and are part of the label definition, like the normal: one fixed label serves every fold and live verification. Leave-one-station-out, it gives 1.02× the station heatwave count with κ 0.48 at the fit stations, and 0.88× with κ 0.53 at six check stations it was not fitted on (Jhansi κ 0.80). IMD as-is gets 0.57–0.60× and κ 0.49–0.52.

**Sensitivity labels**, built alongside: the same rule on IMD gridded Tmax, and a **percentile label** (Tmax above the calendar-day 95th percentile for ≥ 3 consecutive days). If conclusions change between labels, they are fragile.

**Label QA** (`reports/label_qa.md`, gate G1):
- heatwave days by year, month and district;
- agreement with the **stations' own heatwave years** (correlation ≥ 0.7 and ≥ 5 of the top 8 years shared). This replaced an India-wide list of "known" severe seasons, several of which were not severe in UP;
- contiguous state-wide hot days grouped into **episodes**: 265 in development years. The episode count is the real sample size.

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
- **Heat season: 1 March – 31 July** (decided in Phase 4, decision record 005). UP stations put more than 1 % of their heatwave days in each month March–July: 16.5 % fall in July, before a late monsoon onset. August holds only 0.5 % at stations; the label's own August share (2.5 %) comes from the label method, so August is excluded.

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
| G3 — main model (Phase 7) | LightGBM beats logistic regression on the onset subset at some lead, with a paired interval excluding 0 | Logistic regression goes live; stop tuning; move effort to NWP (V2) |
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

Implemented in Phase 5: 43 features (41 as-of + 2 lead). The full list, with statistics, is generated in `docs/features.md`.

| Group | Features | Physical reason |
|---|---|---|
| Current heat | Tmax (label scale), Tmin, their anomalies, diurnal range | Distance from the threshold |
| Persistence | Tmax anomaly mean over 3/7/14 d; 7-day max; days above the hot-day threshold (90th percentile of the anomaly, per district and month) in the last 7/14 d; consecutive hot days; heatwave flag at `t`; days since last heatwave day | Heatwaves come from persistent, blocked patterns |
| Trend | ΔTmax over 1/3/7 d; 7-day Tmax slope | Build-up vs breakdown |
| Land surface | GWETTOP, GWETROOT and their anomalies; rainfall over 7/30/90 d; days since rain > 1 mm | Soil-moisture feedback, the main slow driver |
| Moisture and wind | Dew point, RH, specific humidity; wind speed; wind u/v components; surface-pressure anomaly | Dry north-westerly flow vs moist easterlies; monsoon onset ends the season |
| Season | sin/cos of day-of-year; climatological heatwave frequency for the district and calendar day | The prior |
| Spatial | lat, lon, elevation; state-wide mean Tmax anomaly; upwind Tmax anomaly (districts within 250 km, bearing 225–337.5°; missing for the westernmost districts) | Heat usually advects from the west / north-west |
| Lead | lead day `h`; climatological heatwave frequency at the valid date `s + h` | Lets one model share strength across leads |

**Excluded:** district ID as a category (memorisation), month / week / season dummies alongside the harmonics (redundant), 180/365-day temperature statistics (the anomalies already carry them), solar variables (latency). A separate "Tmax percentile for the calendar day" was dropped in favour of the anomaly and the hot-day counts, which carry the same information.

Optional, low priority: weekly Niño-3.4 and MJO indices, at most 1–2 features. There are only ~40 seasons, so these have ~40 effective samples. Mind their publication lag.

### 4.3 Training tables

- **Per-lead model:** long format, one row per (district, issue date `s`, lead `h`), with `h` as a feature. One model covers all leads. Ten separate per-lead models are tried later only if needed.
- **Window model:** one row per (district, issue date), as its own model.
- **Rows used:** issue dates whose valid dates fall in **15 Feb – 15 Aug** (season plus shoulders, to see onset and withdrawal).
- **Near-duplicate rows:** neighbouring districts share grid cells, but after area weighting none share identical inputs (highest overlap 0.68). **No row weighting in V1** (decision record 006): regularised models, year-block intervals and region-level claims instead.
- Training tables are versioned by `feature_version` and `label_version`.
- **Storage:** one feature table per block (fold 1–5 and `development`) holds one row per (district, as-of date). The per-lead and window tables are assembled from it at training time (`features/targets.py`), since storing the per-lead form would be 10× larger. A fold's table holds only its own training and validation years; only `development` holds 2021 onward.

### 4.4 Model ladder

| ID | Model | Role |
|---|---|---|
| M0 | **Climatology:** smoothed heatwave frequency by district × calendar day (training years) | Reference for all skill scores |
| M1 | **Persistence lookup:** probability by (heatwave at `t` / current anomaly bin) × lead | The baseline that exposes weak ML |
| M2 | **Logistic regression** on ~6 features: harmonics, Tmax anomaly, 7-d anomaly mean, soil-moisture anomaly, climatological frequency, lead | Interpretable, well calibrated, hard to overfit |
| M3 | **LightGBM**, log-loss objective, full feature set | Built before go-live; production model only if it passes G3 against M2 (§6.1) |
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
| Data age = `L` + 1 days (one day late) | Forecast from the latest data as described in §1.3: 9 valid days, flagged "data 1 day late" |
| Data age = `L` + 2 days | Same, 8 valid days, flagged with the data age |
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
| Label drift (monthly in season; yearly at retrain) | Live POWER Tmax against IMD's real-time grid and available stations. The label's drift term holds the 2016–2020 level after 2020; at each annual retrain, once IMD publishes the year, re-estimate it and version the label |

- **End-of-season report every July:** full metrics with bootstrap intervals, an episode-by-episode review, and misses analysed by cause (label, data, genuinely unpredictable).
- **Retraining once a year, after the season.** The challenger is compared with the current model on the same walk-forward plus the newly completed season. It is promoted only if the paired interval on ΔBSS is not negative and calibration is no worse. Criteria are written before the comparison.
- No metric-drop retraining triggers mid-season: with a handful of events a week, changes are noise. Mid-season changes are limited to documented bug fixes.

### 5.7 Repository layout

```
CRO-Heatwave/
├── reference.md              # this guide
├── pyproject.toml  uv.lock
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

**Production model: chosen by gate G3 before go-live** (revised 2026-10-20; the project ran well ahead of schedule). Both M2 (logistic regression) and M3 (LightGBM) are built and evaluated on the same walk-forward folds. **M3 goes live only if it beats M2 on the onset subset at some lead with a paired year-block interval excluding 0**; otherwise M2 goes live. M3 tuning is capped at the limits in §4.6 and is never extended to make it win. The chosen model is calibrated, tier thresholds are frozen, and it is tested once on 2021–2025 before the pipeline goes live.

From completion of the build to 28 February the system runs in **shadow mode**: daily forecasts are produced and published, but not yet presented to users as live.

### 6.2 Phase plan

Weeks start on Mondays. Each phase ends on its exit criterion: if a phase overruns, the slack and then the 1 April fallback absorb it, not skipped checks.

| # | Phase | Weeks | Main work | Exit criterion |
|---|---|---|---|---|
| 0 | Setup and problem freeze | done 7 Oct 2026 | Repo scaffold, `problem.yaml`, CI, source verification | CI green; config matches this document |
| 1 | Geography | done 7 Oct | Boundaries, MERRA-2 grid, district–grid weights | Weights sum to 1 per district |
| 2 | Ingestion + daily download job | done 20 Oct | POWER backfill, bitemporal raw store, daily GitHub Actions ingest, IMD download | Complete spine; 13 unattended daily runs |
| 3 | Data quality | done 20 Oct | Hard/soft checks, QA report, daily validate gate | Every anomaly explained or flagged |
| 4 | Labels | done 20 Oct | Station-referenced label with spatial plane and drift term; episodes; label QA | **G1** passed (recorded exception) |
| 5 | Features | done 20 Oct | 43 as-of features, fold-aware climatology, leakage tests | Leakage tests green on 20 real dates |
| 6 | Validation harness + baselines | 21 Oct – 1 Nov | Fold runner, BSS, year-block bootstrap (paired), onset subset, by-lead/month/region breakdowns; region definitions; M0, M1, M2 | **G2** passed; baseline report |
| 7 | LightGBM, ablation, calibration, tiers | 2–15 Nov | M3 with regularised defaults; ≤ 40-trial tuning on inner blocks; group ablation; sliding vs expanding window; monotone constraints; calibration (Platt/beta on OOF) and tier thresholds for the chosen model | **G3**: production model chosen (M3 or M2) with paired intervals; frozen and git-tagged |
| 8 | Final test | 16–20 Nov | Run 2021–2025 once. The 2026 season check runs once its final (MERRA-2) data and IMD's 2026 grid are available, before go-live | **G4** passed; final-test report including what looks bad |
| 9 | Production pipeline | 21 Nov – 6 Dec | Daily scoring with the late-data rule; failure policy; outcomes; static site; CSV/JSON; alerts; workflow keep-alive | All daily steps run end-to-end on GitHub Actions |
| 10 | Docs and monitoring | 7–13 Dec | Model card, data card, runbook, monitoring (incl. label drift), weekly scorecard job | Runbook usable by someone new |
| — | Shadow mode | 14 Dec – 28 Feb 2027 | Unattended daily runs; one simulated outage; provisional-vs-final skew audit; 2026 season check; officials review the site | 14+ consecutive unattended runs; skew measured |
| 11 | **Live season** | 1 Mar – 31 Jul 2027 | Weekly scorecards; incident log; documented fixes only | — |
| 12 | Season report + annual retrain | Aug – Dec 2027 | End-of-season report; drift term re-estimated with IMD 2026–2027; annual retrain; champion vs challenger | Promotion decision for 2028, written before the comparison |
| — | V2 | 2028 onward | To do later list, NWP first | — |

### 6.3 Schedule risks

| Risk | Mitigation |
|---|---|
| **Labels (Phase 4)**: IMD data access or a bad POWER–IMD mismatch | Start the IMD download in Phase 2 while the backfill runs. If the comparison is messy, choose the simplest defensible option and record it; refinements go to the 2028 retrain |
| Hugging Face / GitHub Actions friction | Build the daily ingest early (Phase 2), so these problems surface in November, not February |
| Holiday weeks (late Dec – early Jan) | Phases 5–6 are the least risky place to lose time; 2 weeks of slack plus the 1 April fallback |
| 2026 season's final MERRA-2 values not yet available in January | Check the raw table's provisional/final status before Phase 8; if still provisional, run the 2026 check later and record that |

### 6.4 Reports and decision records

- Every gated phase (4, 6, 7, 8) ends with a short report in `reports/`. Phase 12 adds the season report and the annual champion-vs-challenger report.
- Any change to a decision in this document: edit the section, add a change-log line, and add a short note in `docs/decisions/` if the reason needs more than a line.

---

## To do later (V2 and beyond)

- **Add NWP forecasts as inputs** (statistical post-processing of weather-model output). Must be trained on *archived past forecasts*, not on later observations. Training on observations teaches the model perfect forecasts it will never get in production.
  - Evaluate sources: NOAA GEFS reforecasts, ECMWF open data, Open-Meteo historical forecast archive (archive length, licence, latency).
  - V2 baseline: the raw NWP forecast with the heatwave rule applied. The V2 model must beat that.
- Humid-heat target (heat index / wet-bulb) for eastern UP, which the dry-bulb rule misses.
- Finer-resolution inputs (e.g. ERA5-Land), if their extra latency is worth it.
- Replace GADM boundaries with a current Survey of India–consistent 75-district file.
- More station data for the label's bias correction, especially Bundelkhand and the east (IMD station records if obtainable). The current map rests on 4 fit stations plus 6 checks (decision record 005).
- Ten separate per-lead models vs the single stacked model, if the stacked model underperforms at specific leads.
- Slow climate indices (Niño-3.4, MJO) as features, if development results justify them.
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
| 2026-10-07 | 5.7 | Makefile dropped from the layout (no `make` on Windows; the `heatwave` CLI covers it) |
| 2026-10-20 | 1.3, 5.5 | `L = 3` kept after 13 days of measured latency (3 d on 11 runs, 4 d on 2); late-data days publish a 9-day forecast flagged "data 1 day late" |
| 2026-10-20 | 2.1 | `WD10M` replaced by `U10M`/`V10M` after finding invalid POWER wind directions from 2012-12-31 onward |
| 2026-10-20 | 2.4 | Label source decided: POWER minus a spatial bias plane (from IMD), mapped to station levels; fitted once on 1991–2020; 6-harmonic normal; QA check uses station heatwave years |
| 2026-10-20 | 3.1, 4.3 | Evaluation season March–July; training rows 15 Feb – 15 Aug |
| 2026-10-20 | 2.1 | IMD gridded and NOAA GSOD stations added as label references |
| 2026-10-20 | 2.4, 5.6 | Label gains a yearly drift term (POWER's bias shrinks over time); drift monitoring and annual re-estimation added |
| 2026-10-20 | 4.2, 4.3 | Features implemented (43; dictionary in docs/features.md); fold tables stop at their validation years; no near-duplicate row weighting in V1 |
| 2026-10-20 | 3.7 | Gate G1 passed with a recorded exception after the drift term; the per-station year check is now primary |
| 2026-10-20 | 6.1, 6.2, 4.4, 3.7 | Ahead of schedule: LightGBM (M3) built before go-live in Phase 7; G3 picks the production model; shadow mode Dec–Feb; plan dates revised; live season March–July |
| 2026-10-25 | 2.3 | Known-defect quarantine rule added after POWER's ×24 rainfall revision (decision record 007) |
| 2026-10-25 | 3.5, 3.7, 4.4 | Phase 6: regions = UP's four economic regions; M1 adds a season band; M2 is one logistic model per lead; G2 passed after its M0 check was redesigned to an in-sample bug check (decision record 008) |
