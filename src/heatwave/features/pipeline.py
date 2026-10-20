"""Phase 5 pipeline: feature tables per block, saved climatologies, dictionary.

Blocks (reference.md §3.2-3.3):
    fold1..fold5  climatology fitted on the fold's training years; its validation
                  years must come after them; rows only up to the fold's last
                  validation year
    development   climatology fitted on 1981-2020; held-out years are the final
                  test (used later for the final model, final test and live scoring)

Each block's table holds every as-of date whose issue date (t + L) has at least
one valid date inside the training window. The per-lead and window training
tables are assembled from it at training time (features.targets).

Writes data/features/<block>.parquet, data/features/<block>_climatology/ and
data/features/manifest.json, plus docs/features.md.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from heatwave.config import Problem
from heatwave.features import build as build_module
from heatwave.features.build import ASOF_FEATURES, FEATURES, GROUP_REASONS, compute_features
from heatwave.features.climatology import Climatology, fit_climatology
from heatwave.features.targets import in_window


def load_inputs(root: Path, silver: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """District daily inputs, labels and district statics (incl. elevation)."""
    power = pd.read_parquet(silver / "power_district_daily.parquet")
    labels = pd.read_parquet(
        root / "labels" / "labels_daily.parquet",
        columns=["district_id", "obs_date", "tmax_used", "is_heatwave", "label_version"],
    )
    daily = power.merge(
        labels[["district_id", "obs_date", "tmax_used"]], on=["district_id", "obs_date"]
    ).rename(columns={"tmax_used": "tmax"})
    districts = pd.read_parquet(root / "reference" / "districts.parquet")
    w = pd.read_parquet(root / "reference" / "district_grid_weights.parquet").merge(
        pd.read_parquet(root / "reference" / "grid_cell_meta.parquet"), on="cell_id"
    )
    districts["elevation_m"] = districts["district_id"].map(
        (w["area_weight"] * w["elevation_m"]).groupby(w["district_id"]).sum()
    )
    return daily, labels, districts


def relevant_asof(dates: pd.Series, problem: Problem) -> np.ndarray:
    """As-of dates whose issue date has at least one valid date in the training window."""
    keep = np.zeros(len(dates), dtype=bool)
    for h in problem.horizons:
        valid = dates + pd.Timedelta(days=problem.latency_days + h)
        keep |= in_window(valid, problem.training_season)
    return keep


def feature_version(cfg: dict, label_version: str) -> str:
    payload = json.dumps(
        {
            "features": ASOF_FEATURES,
            "cfg": cfg,
            "label": label_version,
            "code": hashlib.sha256(inspect.getsource(build_module).encode()).hexdigest(),
        },
        sort_keys=True,
    )
    return "ft_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def blocks(problem: Problem) -> list[dict]:
    v = problem.validation
    out = [
        {"block": f"fold{f.fold}", "fit_years": f.train, "held_out": f.validate} for f in v.folds
    ]
    out.append({"block": "development", "fit_years": v.development, "held_out": v.final_test})
    return out


def save_climatology(clim: Climatology, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.concat([c.assign(variable=v) for v, c in clim.normals.items()]).to_parquet(
        path / "normals.parquet", index=False
    )
    clim.hot_threshold.to_parquet(path / "hot_threshold.parquet", index=False)
    clim.hw_freq.to_parquet(path / "hw_freq.parquet", index=False)
    (path / "meta.json").write_text(json.dumps({"fit_years": list(clim.fit_years), **clim.meta}))


def load_climatology(path: Path) -> Climatology:
    meta = json.loads((path / "meta.json").read_text())
    normals = pd.read_parquet(path / "normals.parquet")
    return Climatology(
        fit_years=tuple(meta["fit_years"]),
        normals={
            v: g.drop(columns="variable").dropna(axis=1, how="all").reset_index(drop=True)
            for v, g in normals.groupby("variable")
        },
        hot_threshold=pd.read_parquet(path / "hot_threshold.parquet"),
        hw_freq=pd.read_parquet(path / "hw_freq.parquet"),
        meta={k: v for k, v in meta.items() if k != "fit_years"},
    )


def run(problem: Problem, cfg: dict, root: Path, silver: Path, docs: Path) -> dict:
    daily, labels, districts = load_inputs(root, silver)
    label_version = labels["label_version"].iloc[0]
    fver = feature_version(cfg, label_version)
    out_dir = root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"feature_version": fver, "label_version": label_version, "blocks": {}}
    dev_table = None
    for b in blocks(problem):
        clim = fit_climatology(daily, labels, b["fit_years"], cfg)
        clim.check_held_out(b["held_out"])  # refuses to build if the fence is broken
        feats = compute_features(daily, labels, clim, districts, cfg)
        keep = relevant_asof(feats["asof_date"], problem)
        if b["block"] != "development":
            # A fold table holds only its own training and validation years, so
            # final-test rows cannot reach fold-level work even by accident.
            keep &= (feats["asof_date"].dt.year <= b["held_out"][1]).to_numpy()
        feats = feats[keep].reset_index(drop=True)
        feats["feature_version"] = fver
        feats.to_parquet(out_dir / f"{b['block']}.parquet", index=False)
        save_climatology(clim, out_dir / f"{b['block']}_climatology")
        manifest["blocks"][b["block"]] = {
            "fit_years": list(b["fit_years"]),
            "held_out_years": list(b["held_out"]),
            "fit_end_date": str(clim.fit_end_date.date()),
            "rows": len(feats),
            "first_asof": str(feats["asof_date"].min().date()),
            "last_asof": str(feats["asof_date"].max().date()),
        }
        if b["block"] == "development":
            dev_table = feats
        print(f"{b['block']}: {len(feats):,} rows, climatology {b['fit_years']}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_dictionary(docs / "features.md", dev_table, problem, cfg, fver)
    return manifest


def write_dictionary(
    path: Path, table: pd.DataFrame, problem: Problem, cfg: dict, fver: str
) -> None:
    dev = problem.validation.development
    d = table[table["asof_date"].dt.year.between(*dev)]
    stats = d[ASOF_FEATURES].agg(["mean", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]).T
    stats.columns = ["mean", "p5", "p95"]
    nulls = d[ASOF_FEATURES].isna().mean()
    lines = [
        "# Feature dictionary",
        "",
        f"Generated by `uv run heatwave build-features` (feature version `{fver}`).",
        "Definitions: `src/heatwave/features/build.py`; settings: `configs/features.yaml`.",
        "",
        f"- {len(FEATURES)} features ({len(ASOF_FEATURES)} as-of features plus "
        f"{len(FEATURES) - len(ASOF_FEATURES)} lead features in the per-lead table); "
        "budget 40-50 (reference.md §4.1).",
        "- Every as-of feature uses data dated <= the as-of date t only. Issue date "
        f"s = t + {problem.latency_days}; valid date s + h.",
        "- Features marked *clim* compare against a climatology fitted on the fold's "
        "training years only.",
        f"- Statistics: development years {dev[0]}-{dev[1]}, as-of dates feeding the "
        "training window, development-block climatology.",
        "",
    ]
    for group, reason in GROUP_REASONS.items():
        members = [f for f in FEATURES if f.group == group]
        if not members:
            continue
        lines += [
            f"## {group.capitalize()}",
            "",
            f"*Why:* {reason}.",
            "",
            "| feature | description | clim | missing | mean | p5 | p95 |",
            "|---|---|---|---|---|---|---|",
        ]
        for f in members:
            if f.name in stats.index:
                s = stats.loc[f.name]
                num = f"{nulls[f.name]:.2%} | {s['mean']:.3g} | {s['p5']:.3g} | {s['p95']:.3g}"
            else:
                num = "per-lead table | | |"
            lines.append(
                f"| `{f.name}` | {f.description} | {'yes' if f.uses_climatology else ''} | {num} |"
            )
        lines.append("")
    lines += ["## Settings", "", "```yaml", yaml.safe_dump(cfg, sort_keys=False).strip(), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
