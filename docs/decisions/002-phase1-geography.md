# 002: Phase 1 geography results

**Date:** 2026-10-07
**Phase:** 1
**Affects:** reference.md §2.2; `configs/data.yaml`

## What was built

`uv run heatwave build-geo` turns GADM 4.1 district polygons into four tables in `data/reference/`:

| Table | Content |
|---|---|
| `districts` | 75 districts: id, current name, equal-area centroid, area, centroid cell, cell count |
| `grid_cells` | The 102 MERRA-2 cells overlapping UP: the POWER fetch list for Phase 2 |
| `district_grid_weights` | Share of each district's area in each cell (sums to 1 per district) |
| `district_overlap` | For each district pair, the share of input they have in common, `Σ min(w_a, w_b)` |

Method choices:
- Areas come from an Albers equal-area projection centred on UP; the earlier project's EPSG:7755 is conformal, not equal-area.
- Cell edges are densified before projection.
- Fragments below 0.1 % of a district are dropped as boundary slivers, then the weights are renormalised.
- The three GADM names are mapped to current names (Allahabad → Prayagraj, Faizabad → Ayodhya, Sant Ravi Das Nagar → Bhadohi). The build fails if any old name remains.

## Results

| Quantity | Value |
|---|---|
| Districts | 75 (total area within 2 % of UP's 240,900 km²) |
| Grid cells to fetch | **102** |
| Cells per district (min / median / max) | 2 / 4 / 9 (Ghaziabad 2; Lakhimpur Kheri 9) |
| Districts where one cell covers ≥ 80 % of the area | 5 |
| Median district area | 3,084 km², about one grid cell |
| **Centroid method (earlier project):** distinct series | **62 of 75**; 25 districts share a centroid cell with another district |
| **Area-weighted method:** identical district inputs | **None.** The highest overlap is 0.68 (Gorakhpur / Sant Kabir Nagar) |
| District pairs sharing any cell | 321 |
| District pairs sharing > 50 % of their input | 11 |

## Consequences

- Area weighting removes exact duplicates, but neighbouring districts still share much of their input. Effective sample size stays far below 75 × days. Claims stay at region level (reference.md §3.5), and Phase 5 decides how to treat near-duplicate rows (§4.3), using `district_overlap`.
- Phase 2 fetches 102 cells, not 75 districts.
- **Carried to Phase 2:** confirm that a POWER point request returns the containing cell's value without interpolation. Test: two points inside one cell must give identical series, and two points in neighbouring cells must differ. The earlier project's pulls suggested this, but that data is deleted.

## Not done here (later phases)

- District elevation (needed for the "all districts are plains" check in Phase 4 and the spatial features in Phase 5).
- Region labels (West / Central / East / Bundelkhand) for the by-region breakdowns. The region definition has to be agreed before Phase 6.
