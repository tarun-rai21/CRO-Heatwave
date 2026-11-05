# CRO-Heatwave

Probabilistic heatwave forecasts for all 75 districts of Uttar Pradesh, for each of the next 10 days. Built for UP government weather officials: forecasts are produced daily, published with alert tiers, and verified against what actually happened.

**Status:** running daily in shadow mode at https://tarun-rai21.github.io/CRO-Heatwave/ with model `m2_v2`. V1.0 goes live on 1 March 2027. This is an experimental forecast, not an official one: heatwave warnings for India come from the India Meteorological Department.

Everything about the design (target, data, labels, validation, models, architecture, phases) is in **[`reference.md`](reference.md)**. Read that first.

- **Operating it, or something has gone wrong:** [`docs/runbook.md`](docs/runbook.md)
- **What the model is and how good it is:** [`reports/model_card.md`](reports/model_card.md)
- **What the data is and what is wrong with it:** [`reports/data_card.md`](reports/data_card.md)

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tarun-rai21/CRO-Heatwave.git
cd CRO-Heatwave
uv sync              # creates .venv with Python 3.12 and locked dependencies
uv run pytest        # tests (uv run pytest -m slow: real-data leakage checks)
uv run ruff check .  # lint
uv run heatwave show-problem
```

## Layout

| Path | Contents |
|---|---|
| `reference.md` | The project guide: every agreed decision and why |
| `configs/problem.yaml` | Frozen problem definition (dates, label rule, splits, tiers), validated on load |
| `configs/data.yaml` | Boundaries, grid, NASA POWER and IMD settings |
| `configs/quality.yaml` | Data-quality rules and reviewed flags |
| `configs/labels.yaml` | Label build, sensitivity labels and label QA settings |
| `configs/features.yaml` | Feature windows, caps and climatology settings |
| `configs/models.yaml` | Baseline and evaluation settings |
| `configs/regions.yaml` | UP regions for breakdowns |
| `configs/publish.yaml` | Site address and email server |
| `configs/frozen/` | Frozen models: coefficients, thresholds, monitoring limits |
| `reports/` | Model card, data card, and generated reports for labels, baselines, the model and the final test |
| `src/heatwave/` | Package; every pipeline step is a `heatwave` CLI command |
| `tests/` | `pytest` tests, run in CI on every push (`tests/data/` skip when local data is absent) |
| `docs/decisions/` | Short notes recording decisions and verifications |
| `docs/runbook.md` | How to operate it and what to do when something fails |
| `docs/features.md` | Generated feature dictionary |

Data is never committed. It lives in a private Hugging Face dataset (reference.md §5).
