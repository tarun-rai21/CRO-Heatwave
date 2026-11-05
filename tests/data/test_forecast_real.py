"""Train/serve match on real data: the daily scorer must reproduce the rows the frozen
model was scored on, from the same stored data. Uses a spent final-test date."""

import numpy as np
import pandas as pd
import pytest

from heatwave.config import REPO_ROOT, load_problem, read_yaml
from heatwave.models.frozen import FrozenModel
from heatwave.pipeline import daily
from heatwave.validation import final_test, walk_forward

ROOT = REPO_ROOT / "data"
VERSION = read_yaml("models.yaml")["frozen"]["version"]
MODEL = REPO_ROOT / "configs" / "frozen" / f"{VERSION}.json"
have = (ROOT / "features" / "development.parquet").exists() and (ROOT / "models/frozen").exists()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not have, reason="local data not available")]


@pytest.mark.parametrize("issue", ["2024-05-10", "2022-04-02", "2024-01-05"])
def test_daily_scorer_matches_the_evaluated_rows(issue):
    problem, cfg = load_problem(), read_yaml("models.yaml")
    _, out = daily.run(
        problem, read_yaml("data.yaml"), read_yaml("features.yaml"), cfg, ROOT, MODEL, issue
    )
    s = pd.Timestamp(issue)
    assert out["status"].eq("ok").all() or s.year == 2024 and s.month == 1
    assert len(out) == 75 * 11
    if s.month < 3:
        return  # out of season: the evaluated tables have no such rows, plumbing only
    model = FrozenModel.load(MODEL, ROOT / cfg["frozen"]["table_dir"])
    regions = walk_forward.regions(REPO_ROOT / "configs" / "regions.yaml")
    lead, window = final_test.load_rows(problem, ROOT, (s.year, s.year), model, regions, True)

    want = lead[lead["issue_date"] == s].sort_values(["district_id", "valid_date"])
    got = out[out["target"] == "lead"].sort_values(["district_id", "valid_date"])
    assert len(got) == 750
    assert (got["valid_date"].to_numpy() == want["valid_date"].to_numpy()).all()
    assert (got["model_lead"].to_numpy() == want["lead"].to_numpy()).all()
    assert np.allclose(got["p"].to_numpy(), want["p_M2F"].to_numpy(), rtol=0, atol=1e-9)
    assert np.allclose(got["p_clim"].to_numpy(), want["p_M0"].to_numpy(), rtol=0, atol=1e-9)

    w = window[window["issue_date"] == s].sort_values("district_id")
    g = out[out["target"] == "window"].sort_values("district_id")
    assert np.allclose(g["p"].to_numpy(), w["p_M2F"].to_numpy(), rtol=0, atol=1e-9)
    assert np.allclose(g["p_clim"].to_numpy(), w["p_M0"].to_numpy(), rtol=0, atol=1e-9)
    assert (g["tier"].to_numpy() == model.tier(w["p_M2F"].to_numpy())).all()
