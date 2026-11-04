"""Tuning: a fold binned, saved and reloaded trains exactly as in memory."""

import numpy as np
import optuna

from heatwave.validation import tuning
from tests.unit.test_gbm import CFG, model, rows


def test_saved_fold_early_stops_like_the_in_memory_one(tmp_path):
    m = model()
    _, fit_part, inner_part = m.datasets(rows())
    expected = m.early_stop(fit_part, inner_part)
    for name, part in (("outer", fit_part), ("inner", inner_part)):
        np.save(tmp_path / f"fold1_{name}_margin.npy", part.get_init_score())
        part.set_init_score(None)
        part.save_binary(str(tmp_path / f"fold1_{name}.bin"))
    got = m.early_stop(*tuning.load_fold(tmp_path, 1, m.params["max_bin"]))
    assert got[0] == expected[0]
    assert np.isclose(got[1], expected[1], rtol=0, atol=1e-12)


def test_sampled_parameters_stay_in_the_configured_space():
    space = CFG["m3"]["tuning"]["space"]
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    for _ in range(20):
        p = tuning.sample(study.ask(), space)
        for k, v in p.items():
            assert space[k][0] <= v <= space[k][1], k
