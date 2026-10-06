# CRO-Heatwave

Probabilistic heatwave forecasts for all 75 districts of Uttar Pradesh, for each of the next 10 days. Built for UP government weather officials: forecasts are produced daily, published with alert tiers, and verified against what actually happened.

**Status:** Phase 2 (ingestion) in progress: backfill complete, daily job running. V1.0 goes live on 1 March 2027.

Everything about the design (target, data, labels, validation, models, architecture, phases) is in **[`reference.md`](reference.md)**. Read that first.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tarun-rai21/CRO-Heatwave.git
cd CRO-Heatwave
uv sync              # creates .venv with Python 3.12 and locked dependencies
uv run pytest        # tests
uv run ruff check .  # lint
uv run heatwave show-problem
```

## Layout

| Path | Contents |
|---|---|
| `reference.md` | The project guide: every agreed decision and why |
| `configs/problem.yaml` | Frozen problem definition (dates, label rule, splits, tiers), validated on load |
| `configs/data.yaml` | Boundaries, grid and projection settings |
| `src/heatwave/` | Package; every pipeline step is a `heatwave` CLI command |
| `tests/` | `pytest` tests, run in CI on every push (`tests/data/` skip when local data is absent) |
| `docs/decisions/` | Short notes recording decisions and verifications |

Data is never committed. It lives in a private Hugging Face dataset (reference.md §5).
