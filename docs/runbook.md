# Runbook

How to run the UP heatwave forecast, and what to do when something goes wrong. Written for someone new to the project. Why things are built this way is in [`reference.md`](../reference.md) and [`docs/decisions/`](decisions/).

The system runs by itself. In a normal week, nobody needs to do anything.

## What it is

Once a day, GitHub Actions downloads NASA POWER weather data for Uttar Pradesh, scores a frozen model, and publishes a chance of a heatwave day for all 75 districts. Nothing runs on anyone's laptop.

| | |
|---|---|
| Public site | https://tarun-rai21.github.io/CRO-Heatwave/ |
| Code (public) | https://github.com/tarun-rai21/CRO-Heatwave |
| Runs | The repo's Actions tab |
| Data (private) | Hugging Face dataset `tarun-rai/cro-heatwave-data` |
| Live model | `m2_v2`, set by `frozen.version` in `configs/models.yaml` |
| Alerts | Email to the owner only, through a Gmail account |

## The workflows

| Workflow | When | What it does |
|---|---|---|
| `daily-ingest.yml` ("Daily run") | Set for 09:00 IST. GitHub starts it late: 13:26 to 17:00 IST in October and November 2026 | Download, forecast, outcomes, checks, site, email |
| `monthly-checks.yml` | 1st of each month, 11:30 IST or later | Provisional against final data, and label against stations |
| `test-alert-email.yml` | By hand | Sends one test email |
| `ci.yml` | Every push | Tests and lint |

### Steps of the daily run, and what a failure does

| Step | If it fails |
|---|---|
| Ingest | The run carries on from stored data, then is marked failed at the end |
| Pull from Hugging Face | **The run stops. The site keeps showing the last good forecast** |
| Forecast | **The run stops. The site keeps showing the last good forecast** |
| Outcomes, Monitor, Email | The run carries on |
| Build site, Save outputs | **The run stops. The site keeps showing the last good forecast** |
| Validate | Comes after publishing. Marks the run failed, so GitHub emails |

A forecast is made from the data available, as follows:

| Age of newest data on the issue date | What is published |
|---|---|
| 3 days or fewer | Normal forecast from 3 days back |
| 4 days | 9 valid days, flagged "data 1 day late" |
| 5 days | 8 valid days, flagged "data 2 days late" |
| Older, or validation failed | "Degraded": climatology only, no tiers. Yesterday's forecast is never reused |

## The season

| Issue date | What the site shows |
|---|---|
| About 15 Aug to 4 Feb | No probabilities. This is normal |
| About 5 to 13 Feb | Single-day columns start to fill, for dates from 15 Feb. The page still says "Out of season" |
| 14 Feb to 5 Aug | The 10-day figure appears too |
| Last day of Feb to 21 Jul | Tiers are issued. Only now can a Warning email go out |
| About 6 to 14 Aug | Only single-day columns remain, up to 15 Aug |

Checks that compare forecasts with history are skipped when there is no in-season forecast, so a quiet log in winter says little about the model.

## Who gets told what

| You receive | It means | Do this |
|---|---|---|
| GitHub "run failed" email | A step failed, validation failed, or the download failed | [Failed run](#failed-run) |
| "Heatwave pipeline: failed checks" | The forecast is degraded, has missing districts, or has bad or blank probabilities | [Failed checks](#failed-checks) |
| "Heatwave pipeline: monthly checks need a look" | Provisional Tmax is off by more than 0.5 °C on average, or the label moved more than 0.75 °C against stations | [Monthly flag](#monthly-flag) |
| "Heatwave model: Warning tier in N districts" | The system working normally. At most one per issue date | Read it. Remember it is a model tier, not an official warning |

**Warnings from the checks are not emailed.** Levels are `ok`, `skipped`, `warn` and `fail`, and only `fail` sends email. In season, read the Monitor step's log once a week (see [the weekly routine](#calendar)).

**No email is sent if GitHub never starts the run.** There is no failed run, so nothing goes out and the site stays as it was, though the page then shows an out-of-date banner. See [Site is out of date](#site-is-out-of-date).

## What to do when

### Site says "data 1 day late"

Usually nothing. Over 31 days of daily runs from October 2026, POWER's delay was the normal 3 days on 24, and 4 days on 3. The other 4 days were one stall, reaching 8 days. If lateness lasts more than about 3 days, look at [POWER's site](https://power.larc.nasa.gov/) for a notice.

**First check when the run started.** The issue date is the date in India, and a run after midnight IST (18:30 UTC) counts as the next day. It reads as a day late even when POWER is on time. This happened on 6 November 2026 with a manual run at 01:03 IST. The next normal run overwrites that day's forecast.

### Site says "Degraded"

The data is more than 2 days late, or validation failed. The status line gives the reason, and the forecast is doing what it should. Then:

1. Open the day's run in the Actions tab and read the Validate step.
2. If POWER is simply behind, wait. Forecasts return by themselves once data is at most 2 days late. In October 2026, POWER published no new days for about 5 days and then caught up in one day.
3. If values are impossible, see [Bad data from POWER](#bad-data-from-power).

### Site is out of date

The page shows "This forecast may be out of date", or "Issued" at the top is older than today and it is past about 18:00 IST. The banner appears from 20:00 IST on the day after the issue date, in the reader's browser. It covers the forecast page only: `latest.csv` and `latest.json` carry `issue_date` but no warning.

1. In the Actions tab, look for today's Daily run.
2. **No run today and no run yesterday:** the schedule may be off. Open the workflow and click **Enable workflow** if offered, then [run it by hand](#run-the-daily-job-by-hand).
3. **No run today only:** GitHub sometimes delays or drops scheduled runs. Wait until about 20:00 IST, then run it by hand, before midnight IST.
4. **Run failed:** see below.

### Failed run

Open the run and find the first red step.

| First failed step | Likely cause | Fix |
|---|---|---|
| Ingest, or "Report a failed download" | POWER down or slow. A partial download is kept and the run retries tomorrow | Usually wait. If it continues for over a day, check POWER's site |
| Ingest and Pull, with 401 or 403 in the log | Hugging Face token expired or revoked | [Replace a secret](#replace-a-secret) |
| Pull | Hugging Face down | Wait, then run by hand |
| Validate, on "hard-check violations" | POWER served impossible values | [Bad data from POWER](#bad-data-from-power) |
| Validate, on "cells incomplete" or "days old" | POWER is partial or late | Wait a day |
| Forecast or Build site, with a Python error | A code bug, or a missing file under `models/frozen` or `reference` | Reproduce locally, fix, and push. Record it in the [incident log](#incident-log). Bug fixes are the only changes allowed in season |
| Deploy | GitHub Pages problem | Re-run the failed job from the run's page |

A failed Deploy after a failed `daily` job is expected: there was no site to deploy.

### Failed checks

Read the email's list of checks.

| Check | Meaning | Do this |
|---|---|---|
| `data_timeliness` | Degraded | [Site says "Degraded"](#site-says-degraded) |
| `coverage` | Fewer than 75 districts | Treat as a bug. Compare `reference/` with the code |
| `probabilities_valid`, `probabilities_present` | A probability outside 0–1, or blank in season | Treat as a bug. Consider [taking the site down](#take-the-site-down) until fixed |

### Monthly flag

**Provisional Tmax.** A steady gap of up to about ±0.8 °C barely moved skill in a test on spent years (decision record 013), so one flagged month is a reason to watch, not to act. Tell the owner if it reaches about 1 °C or lasts several months.

**Label against stations.** Run `uv run heatwave drift-check` for the full comparison. Don't change the label in season: the correction is re-estimated at the yearly retrain.

### Bad data from POWER

The precedent is the ×24 rainfall bug of October 2026 (decision record 007). Raw data is never edited.

1. Confirm it is POWER, not us: compare a raw API response with the stored value.
2. Add a rule to `known_defects` in `configs/data.yaml`, keyed on variable, API version, ingest date and observation dates. Don't correct values by arithmetic.
3. Write a short decision record.
4. Watch `known_defects_active` in the Validate log. If POWER fixes it quietly under the same version, close the rule with `ingested_to`.

### Model looks wrong in season

Don't retrain, change thresholds or switch model. One season has too few independent heatwaves to judge, and mid-season changes are limited to documented bug fixes (`reference.md` §5.6). Check the data and code for a bug, record what you see, and leave the rest for the end-of-season report.

Reading the numbers: quiet seasons look bad. Alert false alarms of 75–90% in a quiet year are in line with what the model did on 2021, 2023 and 2025.

### IMD objects to publication

Disable the site the same day ([take the site down](#take-the-site-down)), and remove the repo reports that hold summaries computed from IMD data (decision record 012).

## Procedures

### Local setup

Needs [uv](https://docs.astral.sh/uv/), and a Hugging Face token with access to the dataset, set as the `HF_TOKEN` environment variable. Never paste a token into chat, an issue or a file in the repo.

```bash
git clone https://github.com/tarun-rai21/CRO-Heatwave.git
cd CRO-Heatwave
uv sync
uv run pytest                 # fast tests; add -m slow for the real-data checks
uv run heatwave hub-pull --only reference raw/power runs features/development_climatology models/frozen forecasts outcomes alerts monitoring
```

### Run the daily job by hand

```bash
gh workflow run daily-ingest.yml
gh run watch
```

Or use **Run workflow** in the Actions tab. It's safe to repeat on one day: the forecast is overwritten by key, and marker files stop a second Warning email. Run it **before midnight IST**, or it becomes tomorrow's issue date and may show as late.

### Test the alert email

Run after changing an email secret, and before each season:

```bash
gh workflow run test-alert-email.yml
```

A green run shows Gmail accepted the message. Also check that "Heatwave model: test email" arrived.

### Replace a secret

The account details and token are GitHub secrets, never in the repo.

```bash
gh secret set HF_TOKEN              # Hugging Face token with write access
gh secret set ALERT_SMTP_USER       # Gmail address that sends
gh secret set ALERT_SMTP_PASSWORD   # a Gmail app password, not the account password
gh secret set ALERT_TO              # who receives
```

`HF_DATASET_REPO` is a repo variable, not a secret.

### Simulate an outage

The scoring half can be tested without touching live data. Replay a date at least 6 days past the newest data, into a scratch folder:

```bash
uv run heatwave forecast --issue-date YYYY-MM-DD --out some/scratch/folder
```

The summary should say `"status": "degraded"`, with no tiers. Tried on 6 November 2026: "data 15 days late", all 75 districts, no tiers.

**This doesn't test the whole chain.** Shadow mode still needs a drill on GitHub that shows the degraded page goes live and that the failed-run email arrives. It isn't built yet.

### Replay a past date

The same command with a past date. The output goes to the folder you name, and `--out` is required so that a replay can't be taken for a real forecast. A replay uses today's best-known data, not what was known that day, so **never copy one into `forecasts/`**. A missed day stays missed.

### Take the site down

```bash
gh api -X DELETE repos/tarun-rai21/CRO-Heatwave/pages    # or Settings → Pages → Unpublish site
gh workflow disable daily-ingest.yml
```

Disabling the workflow alone leaves the last page up. Not yet tried on this repo. To restore, set Pages' source back to GitHub Actions in Settings, then enable the workflow.

### Change the live model

Not in season, except to fix a bug. A new version is made with `heatwave freeze-model`, reviewed, tagged, and set in `frozen.version`. Its `_monitor.json` limits are written once with `heatwave monitor --write-baseline`, and its table goes to `models/frozen` on Hugging Face. The About page text in `src/heatwave/publish/site.py` is written by hand and must be updated with it.

Don't roll back to `m2_v1` as a quick fix. It's the version with the soil moisture problem (decision record 013).

## Rules

- Data is never committed to git.
- Raw data is never edited. Bad versions are quarantined by rule.
- 2021–2025 is spent. Don't present new scores on it as a test.
- The 2026 season check runs once. Its pass bar is set before it runs.
- Nothing is tuned on live results in season.
- Every page and email says this isn't an official forecast. Keep that.
- Any change to a decision goes in `reference.md` and its change log.

## Calendar

| When | What | Who |
|---|---|---|
| Weekly, March to July | Read the scorecard page and the Monitor step's log for `warn` lines | Operator |
| 1st of each month | Read the Monthly checks run. Keep an eye on the soil moisture gap as spring data arrives | Operator |
| By early February | Replace any token near expiry; send a test email | Operator |
| About 15 February | Check that probabilities have appeared and look sensible | Operator |
| Early 2027 | Run the 2026 season check once, when its final POWER data and IMD's 2026 grid exist. The command isn't built yet | Owner |
| Every July | End-of-season report | Owner |
| After each season | `heatwave ingest-imd` and `heatwave drift-check --download`; re-estimate the drift term; yearly retrain, with promotion criteria written first | Owner |

GitHub stops scheduled workflows after 60 days without repo activity. The daily run re-enables both scheduled workflows each day to prevent this. If the daily run stops for over 60 days, enable them by hand.

## Where things are

| Hugging Face folder | Holds |
|---|---|
| `raw/power/` | Every POWER value ever received, with its download time. Append only |
| `raw/imd/`, `raw/stations/` | IMD grids and station files, as downloaded |
| `reference/` | Districts, grid weights, stations, fitted label corrections |
| `labels/`, `features/` | Built tables |
| `models/frozen/` | Climatology tables for the frozen models. Coefficients are in git, under `configs/frozen/` |
| `forecasts/`, `outcomes/` | Each day's forecast, and what happened |
| `alerts/` | Marker files that stop repeat emails |
| `monitoring/` | `daily.parquet` (one row of checks per issue date), `monthly_YYYY-MM.json` |
| `runs/` | Ingest log, including measured latency |

**Not on Hugging Face:** `data/oof/` (out-of-fold predictions, needed for a new freeze) is only on the laptop. `heatwave evaluate-production` and `evaluate-sliding` rebuild it. `data/silver/` and `data/cache/` can also be rebuilt.

## Known gaps

- **No email if the run never starts.** The page warns readers that it's out of date, but nobody is told unless someone opens it.
- **Publishing is late in the day.** The forecast appears at about 13:30–17:00 IST, not 09:00.
- **Run date is measured two ways.** Forecasts use the date in India, `validate` uses UTC. They differ only after midnight IST.
- **No full outage drill yet** (see above).
- **The monthly label check has no current data yet.** It reports "no season data" because NOAA's station files for 2026 return "not found". If that continues in 2027, this check can't run in season, and the label is checked only at the yearly `drift-check`.
- **Failed-run emails go to one GitHub account:** normally whoever last changed the workflow's schedule or re-enabled it. Check that this is still the right person, and that their GitHub notification settings allow it.

## Incident log

Add a row for every incident from shadow mode onward.

| Date | What happened | Cause | Effect | Fix | Record |
|---|---|---|---|---|---|
| 21–25 Oct 2026 | Validate failed on 5 runs | POWER served rainfall at about ×24 from 21 Oct, re-sent corrected on 27 Oct | No data lost. No forecasts existed yet | Quarantine rule, closed after the fix | 007 |
| 26–29 Oct 2026 | Validate failed on 27, 28 and 29 Oct | POWER published nothing after 21 Oct until 30 Oct. Cause not known | No data lost. In season, those 3 days would have been degraded | None; POWER caught up | This log |
| 6 Nov 2026 | Page showed "data 1 day late" | Manual run after midnight IST, not POWER | None. That day's scheduled run overwrites it | None needed | This log |
| 6 Nov 2026 | First Monthly checks run failed | `reference/station_districts.parquet` was only on the laptop, and GitHub lacks the boundary file to rebuild it | Would have failed on 1 Dec | Table uploaded to Hugging Face | This log |
