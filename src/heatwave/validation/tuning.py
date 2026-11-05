"""M3 tuning (reference.md §4.6).

Optuna, at most `m3.tuning.trials` trials. Objective: the mean over the 5 folds of
the in-season log loss on each fold's INNER block (its last training seasons),
with the number of rounds found by early stopping there.

The inner block of a later fold overlaps the validation years of earlier ones
(fold 2's inner block, 2006-2008, is fold 1's validation block), so one best trial
over all folds has seen most of the OOF years. Out-of-fold evaluation therefore uses
`nested_choice`: each fold gets the trial that was best on the inner blocks up to
its own, all inside its training years. Fold 5's choice is the overall best trial.

Trial 0 is the default parameters, so the best trial is never worse than them on
this objective. Whether it is better by more than noise is decided afterwards on
the OOF results (paired year-block interval), as §4.6 requires.

Each fold is binned once and saved (LightGBM binary + base margin); trials load
one fold at a time, so memory stays at one fold. Trials are stored in SQLite, so
an interrupted study resumes, on this machine or another (e.g. Kaggle) once the
study file is synced. Each trial records the device and threads it ran with.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from heatwave.config import Problem
from heatwave.models.gbm import M3
from heatwave.validation import walk_forward

TUNED = ["learning_rate", "num_leaves", "min_child_samples", "feature_fraction", "lambda_l2"]


def _model(model_cfg: dict, problem: Problem, target: str, overrides: dict | None = None) -> M3:
    return M3(
        model_cfg,
        target,
        problem.validation.inner_block_seasons,
        eval_season=problem.evaluation_season,
        params=overrides,
    )


def runtime(device: str = "cpu", threads: int | None = None) -> dict:
    """LightGBM parameters for where the run happens; they change speed, not the
    model (up to floating-point differences, which `bench` measures)."""
    out: dict = {"device_type": device}
    if threads:
        out["num_threads"] = threads
    return out


def prepare(problem: Problem, model_cfg: dict, root: Path, target: str, out_dir: Path) -> None:
    """Bin each fold's outer (fit) and inner (early-stopping) rows once."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = None
    for fold in problem.validation.folds:
        done = out_dir / f"fold{fold.fold}.json"
        if done.exists():
            continue
        if labels is None:
            labels = pd.read_parquet(
                root / "labels" / "labels_daily.parquet",
                columns=["district_id", "obs_date", "is_heatwave"],
            )
        train, _ = walk_forward.fold_tables(problem, fold, root, labels, target, None)
        full, fit_part, inner_part = _model(model_cfg, problem, target).datasets(train)
        del train
        for name, part in (("outer", fit_part), ("inner", inner_part)):
            np.save(out_dir / f"fold{fold.fold}_{name}_margin.npy", part.get_init_score())
            part.set_init_score(None)  # the binary file cannot hold it: saved alongside
            part.save_binary(str(out_dir / f"fold{fold.fold}_{name}.bin"))
        done.write_text(json.dumps({"outer": fit_part.num_data(), "inner": inner_part.num_data()}))
        n_outer, n_inner = fit_part.num_data(), inner_part.num_data()
        print(f"fold {fold.fold} ({target}): outer {n_outer:,}, inner {n_inner:,}")
        del full, fit_part, inner_part
        gc.collect()


def load_fold(out_dir: Path, fold: int, max_bin: int) -> tuple[lgb.Dataset, lgb.Dataset]:
    """(outer, inner) of a saved fold; inner references outer, both built by lgb.train."""
    params = {"max_bin": max_bin, "verbose": -1, "feature_pre_filter": False}
    parts, ref = [], None
    for name in ("outer", "inner"):
        d = lgb.Dataset(str(out_dir / f"fold{fold}_{name}.bin"), reference=ref, params=params)
        d.set_init_score(np.load(out_dir / f"fold{fold}_{name}_margin.npy"))
        parts.append(d)
        ref = d
    return parts[0], parts[1]


def sample(trial: optuna.Trial, space: dict) -> dict:
    lr, leaves, mcs = space["learning_rate"], space["num_leaves"], space["min_child_samples"]
    ff, l2 = space["feature_fraction"], space["lambda_l2"]
    return {
        "learning_rate": trial.suggest_float("learning_rate", *lr, log=True),
        "num_leaves": trial.suggest_int("num_leaves", *leaves),
        "min_child_samples": trial.suggest_int("min_child_samples", *mcs, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", *ff),
        "lambda_l2": trial.suggest_float("lambda_l2", *l2, log=True),
    }


def bench(
    problem: Problem,
    model_cfg: dict,
    root: Path,
    target: str,
    out_dir: Path,
    fold: int,
    setups: dict[str, dict],
) -> list[dict]:
    """Early-stop the default parameters on one saved fold under each runtime setup:
    seconds, rounds and inner log loss, to compare e.g. GPU with CPU."""
    prepare(problem, model_cfg, root, target, out_dir)
    out = []
    for name, overrides in setups.items():
        model = _model(model_cfg, problem, target, overrides)
        fit_part, inner_part = load_fold(out_dir, fold, model.params["max_bin"])
        start = time.perf_counter()
        rounds, loss = model.early_stop(fit_part, inner_part)
        out.append(
            {
                "setup": name,
                "seconds": round(time.perf_counter() - start, 1),
                "rounds": rounds,
                "inner_logloss": loss,
            }
        )
        print(out[-1])
        del fit_part, inner_part
        gc.collect()
    return out


def load_study(study_path: Path, target: str) -> optuna.Study:
    return optuna.load_study(
        study_name=f"m3_{target}", storage=f"sqlite:///{study_path.as_posix()}"
    )


def nested_choice(study: optuna.Study, folds: list[int]) -> dict[int, dict]:
    """For each fold, the trial with the lowest running mean inner log loss up to that
    fold (trials report it after every fold), using no later fold's inner block."""
    S = optuna.trial.TrialState
    out = {}
    for step, fold in enumerate(folds):
        seen = [
            t
            for t in study.trials
            if t.state in (S.COMPLETE, S.PRUNED) and step in t.intermediate_values
        ]
        best = min(seen, key=lambda t: t.intermediate_values[step])
        out[fold] = {
            "trial": best.number,
            "params": best.params,
            "running_inner_logloss": best.intermediate_values[step],
        }
    return out


def tune(
    problem: Problem,
    model_cfg: dict,
    root: Path,
    target: str,
    out_dir: Path,
    study_path: Path,
    overrides: dict | None = None,
    after_trial: Callable[[], None] | None = None,
) -> dict:
    """Run (or resume) the study; returns the best and default trials' results.
    `after_trial` runs after every finished trial (e.g. to upload the study file)."""
    t = model_cfg["m3"]["tuning"]
    prepare(problem, model_cfg, root, target, out_dir)
    model = _model(model_cfg, problem, target, overrides)
    setup = {k: model.params.get(k) for k in ("device_type", "num_threads")}
    folds = [f.fold for f in problem.validation.folds]

    def objective(trial: optuna.Trial) -> float:
        params = sample(trial, t["space"])
        trial.set_user_attr("setup", setup)
        losses, rounds = [], []
        for step, fold in enumerate(folds):
            fit_part, inner_part = load_fold(out_dir, fold, model.params["max_bin"])
            r, loss = model.early_stop(fit_part, inner_part, params)
            del fit_part, inner_part
            losses.append(loss)
            rounds.append(r)
            trial.report(float(np.mean(losses)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_logloss", losses)
        trial.set_user_attr("rounds", rounds)
        return float(np.mean(losses))

    def synced(study, trial):
        if after_trial is not None:
            after_trial()

    study = optuna.create_study(
        study_name=f"m3_{target}",
        storage=f"sqlite:///{study_path.as_posix()}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=t["seed"]),
        # Prune a trial whose running mean over folds is worse than the median of
        # earlier trials at the same fold; defaults and 5 more trials always finish.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1),
    )
    S = optuna.trial.TrialState
    defaults = {k: model_cfg["m3"]["params"][k] for k in TUNED}

    def default_trial():
        return next(
            (tr for tr in study.trials if tr.state == S.COMPLETE and tr.params == defaults), None
        )

    if default_trial() is None and not any(tr.state == S.WAITING for tr in study.trials):
        study.enqueue_trial(defaults)
    # Only finished trials count towards the cap; an interrupted one stays RUNNING
    # in the database and is not counted.
    used = sum(tr.state in (S.COMPLETE, S.PRUNED) for tr in study.trials)
    if used < t["trials"]:
        study.optimize(
            objective, n_trials=t["trials"] - used, gc_after_trial=True, callbacks=[synced]
        )

    default, best = default_trial(), study.best_trial
    view = lambda tr: {  # noqa: E731
        "trial": tr.number,
        "params": tr.params,
        "mean_inner_logloss": tr.value,
        "fold_logloss": tr.user_attrs.get("fold_logloss"),
        "rounds": tr.user_attrs.get("rounds"),
        "setup": tr.user_attrs.get("setup"),
    }
    states = pd.Series([tr.state.name for tr in study.trials]).value_counts().to_dict()
    return {
        "target": target,
        "trials": states,
        "default": view(default) if default else None,
        "best": view(best),
    }
