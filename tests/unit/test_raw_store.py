"""Bitemporal append-only store: only changes are written; history is recoverable."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from heatwave.ingest.raw_store import RawStore

T1 = pd.Timestamp("2026-06-01 03:30", tz="UTC")
T2 = pd.Timestamp("2026-06-02 03:30", tz="UTC")
T3 = pd.Timestamp("2026-06-03 03:30", tz="UTC")


def rows(*records):
    return pd.DataFrame(records, columns=["cell_id", "obs_date", "variable", "value"])


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "raw" / "power")


def best_value(store, cell, d, var, asof=None):
    b = store.best(asof=asof)
    b["obs_date"] = pd.to_datetime(b["obs_date"]).dt.date
    hit = b[(b.cell_id == cell) & (b.obs_date == d) & (b.variable == var)]
    return hit["value"].iloc[0]


def test_first_append_writes_everything_partitioned_by_year(store):
    n = store.append(
        rows(
            ("c1", date(2025, 12, 31), "T2M_MAX", 20.0), ("c1", date(2026, 1, 1), "T2M_MAX", 21.0)
        ),
        "run1",
        T1,
        "v1",
    )
    assert n == 2
    assert store.years() == [2025, 2026]
    assert [f.name for f in store.files()] == ["run1.parquet", "run1.parquet"]


def test_unchanged_values_are_not_written_again(store):
    data = rows(
        ("c1", date(2026, 5, 1), "T2M_MAX", 40.0), ("c1", date(2026, 5, 2), "T2M_MAX", np.nan)
    )
    store.append(data, "run1", T1, "v1")
    assert store.append(data, "run2", T2, "v1") == 0
    assert len(store.files()) == 1


def test_revision_is_appended_and_best_takes_latest(store):
    store.append(rows(("c1", date(2026, 5, 1), "T2M_MAX", 40.0)), "run1", T1, "v1")
    n = store.append(
        rows(("c1", date(2026, 5, 1), "T2M_MAX", 41.2), ("c1", date(2026, 5, 2), "T2M_MAX", 39.0)),
        "run2",
        T2,
        "v1",
    )
    assert n == 2  # one revision + one new day
    assert best_value(store, "c1", date(2026, 5, 1), "T2M_MAX") == 41.2
    assert len(store.read()) == 3  # the old version is kept


def test_asof_view_reconstructs_what_was_known(store):
    key = ("c1", date(2026, 5, 1), "T2M_MAX")
    store.append(rows((*key, 40.0)), "run1", T1, "v1")  # provisional
    store.append(rows((*key, 41.2)), "run2", T3, "v1")  # final
    assert best_value(store, *key, asof=T2) == 40.0
    assert best_value(store, *key) == 41.2


def test_gap_filled_later_counts_as_a_change(store):
    key = ("c1", date(2026, 5, 1), "GWETROOT")
    store.append(rows((*key, np.nan)), "run1", T1, "v1")
    assert store.append(rows((*key, 0.42)), "run2", T2, "v1") == 1
    assert best_value(store, *key) == 0.42


def test_duplicate_keys_rejected(store):
    dup = rows(("c1", date(2026, 5, 1), "T2M_MAX", 40.0), ("c1", date(2026, 5, 1), "T2M_MAX", 41.0))
    with pytest.raises(ValueError, match="duplicate"):
        store.append(dup, "run1", T1, "v1")


def test_run_ids_cannot_be_reused(store):
    store.append(rows(("c1", date(2026, 5, 1), "T2M_MAX", 40.0)), "run1", T1, "v1")
    with pytest.raises(FileExistsError):
        store.append(rows(("c1", date(2026, 5, 1), "T2M_MAX", 41.0)), "run1", T2, "v1")


def test_metadata_columns(store):
    store.append(rows(("c1", date(2026, 5, 1), "T2M_MAX", 40.0)), "run1", T1, "v2.10.0")
    r = store.read().iloc[0]
    assert r["run_id"] == "run1" and r["api_version"] == "v2.10.0"
    assert r["ingested_at"] == T1
