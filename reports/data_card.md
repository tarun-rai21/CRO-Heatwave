# Data card

The data behind the UP heatwave forecast: where it comes from, what was done to it, and what's wrong with it. Written 6 November 2026. All data is in a private Hugging Face dataset, never in git.

## In brief

- **Live input:** NASA POWER daily weather, on 102 grid cells covering UP.
- **What is predicted:** a heatwave day under IMD's plains rule, applied to POWER temperature corrected to match stations. This is a modelled condition at district level, not an IMD heatwave declaration.
- **Biggest weaknesses:** the label rests on 4 fitting stations; POWER's bias against stations has shifted over time for reasons not understood; live forecasts run on provisional data, while the model was trained on final data.

## Sources

| Source | Used for | Coverage | Terms |
|---|---|---|---|
| [NASA POWER](https://power.larc.nasa.gov/) daily API. MERRA-2, with recent days from GEOS-IT | Model inputs and label temperature | 1981 to about 3 days ago; ½° × ⅝° | Credit line on the site's About page |
| IMD Pune 1° gridded Tmax (Srivastava et al. 2009, doi:10.1002/asl.232) | Label correction only. Never a model input or published | 1981–2025 | Cited. See decision record 012: no licence found on the page, and IMD's undertaking for data procured through its portal bars redistribution. Raw files stay private |
| NOAA GSOD station daily maxima | Station level for the label, and drift checks | 1981–2020 for the label build; 2021–2025 kept apart, for checks | Treated as public domain. NOAA's conditions on international commercial use were not re-checked; this project is non-commercial |
| GADM 4.1 level 2 boundaries | District shapes | Current boundaries applied to all years | Interim. File not redistributed (not in git). To be replaced by a Survey of India-consistent file |

**Rejected:** ERA5 and ERA5-Land (about 5 days' delay); IMD grid as a live input (coarse, published only after the year ends); POWER solar variables (5–7 days' delay); POWER `WD10M` (invalid from 2012-12-31; `U10M` and `V10M` replace it, decision record 004).

## Variables

13 POWER variables are stored: `T2M_MAX`, `T2M_MIN`, `T2M`, `T2MDEW`, `RH2M`, `QV2M`, `PRECTOTCORR`, `WS10M`, `U10M`, `V10M`, `PS`, `GWETTOP`, `GWETROOT`.

**The live model `m2_v2` uses only `T2M_MAX`,** through the label temperature. The others are still downloaded, checked and stored, so a bad value in any of them can trigger a degraded day. They also keep building the provisional-against-final record for later retrains.

## Geography

- 75 districts. District values are area-weighted means of the cells covering them, computed in an equal-area projection. Fragments under 0.1% of a district are dropped.
- GADM's old names are mapped to current ones: Allahabad → Prayagraj, Faizabad → Ayodhya, Sant Ravi Das Nagar → Bhadohi.
- Neighbouring districts share cells, so they aren't independent samples. All intervals therefore resample whole years.

## The label

A district-day is a heatwave day if, on 0.1 °C-rounded values:

```
(Tmax ≥ 40 °C and Tmax − normal ≥ 4.5 °C)  or  Tmax ≥ 45 °C
```

- **Normal:** 1991–2020, smoothed with 6 harmonics.
- **Tmax:** POWER district Tmax, minus a latitude/longitude plane (POWER − IMD), minus a yearly drift term, then quantile-mapped by month to station levels.
- All 75 districts use the plains criteria. Their elevation is 67–374 m.
- A missing Tmax gives a missing label, never a "no".

Why correct at all, in decision record 005: as served, POWER runs +1.8 to +2.7 °C warm against stations and gives about 3 times too many heatwave days. IMD's grid runs −0.4 to −2.3 °C cool and gives about 43% too few.

| Fitted item | Period | Note |
|---|---|---|
| Plane | March–July, 1991–2020 | Fixed |
| Station map | 1991–2020; Lucknow, Prayagraj, Bareilly, Gorakhpur | Fixed |
| Drift | Through 2025, 5-year centred smoothing | Values through 2020 frozen. Later years hold the 2021–25 mean, −0.79 °C (decision record 011) |

**Versions:** label `lbl_d8b1d4c8200b`, features `ft_14d3596786eb`.

**How well it matches stations** (at the version before the 2025 drift update; development years unchanged since): at the 6 check stations it wasn't fitted on, 0.99 times the station heatwave count, κ 0.57. At the 4 fit stations, 0.93 times, κ 0.46. Station points against districts of about 3,000 km² limit κ, so around 0.5 is close to what a gridded label can reach.

### Heatwave share by year

Share of March–July district-days, current label. Very uneven, so five years can include two very different kinds of season.

| 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| 1.14% | 10.01% | 1.33% | 10.98% | 0.85% |

The historical chance of at least one heatwave day in a 10-day window averages about 13% (training years, 15 Feb–15 Aug).

## Year blocks

| Block | Use | State |
|---|---|---|
| 1981–2020 | Development | Train 2001–2020 for `m2_v2`; out-of-fold scores 2006–2020 |
| 2021–2025 | Final test | **Spent.** Used once on `m2_v1`. The label's drift term was then fitted on these years |
| 2026 | One-time season check | Not yet run. **Not perfectly unseen:** `reports/label_qa.md` lists its heatwave days, and its yearly rate was printed once during review (decision record 011). No decision used them |
| 2027 onward | Live | Prospective |

## Storage

- **Append only:** each POWER value is kept with its download time, and only new or changed values are written. What was known on any past date can be rebuilt.
- **Trailing re-download:** each daily run fetches the last 120 days, to catch revisions.
- **Volume:** about 22 million values at 17 October 2026.
- **Checks:** range limits, Tmin ≤ Tmean ≤ Tmax, every cell present on the newest day, and data age. Full-history QA (`reports/data_qa.md`) found no missing values and no violations in 1981–2026 as of 17 October 2026, before the rainfall defect.

## Known problems

| Problem | Effect | State |
|---|---|---|
| **Provisional is not final.** Live data is GEOS-IT; training data is MERRA-2. One batch so far (September, about 3,000 cell-days): `GWETROOT` +0.13, `T2M_MAX` −0.05 °C with SD 1.5 °C, final minus provisional | Soil moisture removed from the model for this reason. Tmax error at district level: SD 1.35 °C, which cost little skill in a sensitivity test | No spring measurements exist yet. Monthly check runs |
| **POWER's bias has moved twice:** around 2003–2006, and abruptly around 2020 (−0.13 in 2019, −0.94 in 2020) | The drift term handles a step poorly | Cause unknown |
| **2022 looks over-corrected.** Smoothing applies −0.84 °C to a year whose own value is −0.52 °C | Its heatwave share nearly doubled, and the 2022 label sits well above the IMD-based one | Open |
| **Hold value after 2025 is an assumption** | POWER may move again | Monthly check against stations; re-estimated yearly |
| **Thin station base.** 4 fit stations and 6 check stations; Bundelkhand rests largely on Jhansi | Regional error unknown in places | On the To do later list |
| **Gorakhpur under-counts.** 71 label heatwave days against 145 at the station | The north-east may be under-called. Humid regime is one candidate; not explained | Open |
| **Rainfall defect**, POWER ×24 from 21 Oct 2026 | No effect on `m2_v2`. Bad versions kept but ignored; POWER re-sent corrected values on 27 Oct | Closed (decision record 007) |
| **Wind direction**, POWER `WD10M` invalid from 2012-12-31 | Not used | Reported here; not yet sent to POWER |
| **Dry-bulb only** | Humid heat in the east is not captured | On the To do later list |

## What is published

Public, on the site and as CSV/JSON: forecast probabilities and tiers only. No IMD values and no raw weather. A test checks that downloads carry no IMD columns.

The repo's reports do include summaries computed from IMD data, such as heatwave shares and POWER-against-IMD comparisons (decision record 012).

### Fields in `latest.csv` and `latest.json`

| Field | Meaning |
|---|---|
| `issue_date` | Date of the forecast, in India |
| `asof_date` | Newest observed day used; blank if degraded |
| `status`, `reason` | `ok`, `late_1`, `late_2` or `degraded`, and why |
| `data_age_days` | Issue date minus newest data |
| `model_version` | For example `m2_v2` |
| `district_id`, `district` | Stable ID and name |
| `target` | `lead` (a single day) or `window` (at least one heatwave day in the 10 days) |
| `model_lead` | Which of the model's lead days produced this row. On a late day it runs one or two ahead of the days from `issue_date` |
| `valid_date`, `window_end` | The day forecast; for a window, its first and last days |
| `p` | Calibrated chance. Blank outside the training window, or if degraded |
| `p_clim` | The usual chance for that district and date |
| `tier` | `none`, `watch`, `alert` or `warning`. Window rows only |
| `in_season` | For a window, whether all of it lies in March–July; for a day, whether that day does |
