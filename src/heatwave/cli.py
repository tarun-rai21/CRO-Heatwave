"""Command-line entry point. Every pipeline step becomes a subcommand (reference.md §5.2).

Commands:
    uv run heatwave show-problem       print the frozen problem definition
    uv run heatwave build-geo          district/grid reference tables
    uv run heatwave ingest-backfill    full POWER history for every cell (resumable)
    uv run heatwave ingest-daily       trailing-window re-fetch (--sync: via Hugging Face)
    uv run heatwave check-raw          completeness of the raw store
    uv run heatwave ingest-imd         IMD gridded Tmax, 1981 to last full year (resumable)
    uv run heatwave qa-report          full-history data QA -> reports/data_qa.md
    uv run heatwave validate           daily gate: hard checks, completeness, latency
    uv run heatwave build-district-daily  cells -> district daily tables (silver)
    uv run heatwave ingest-gsod        NOAA station maxima, development years
    uv run heatwave build-labels       labels, episodes, label reports
    uv run heatwave build-features     feature tables per block, docs/features.md
    uv run heatwave evaluate-baselines walk-forward M0-M2 -> reports/baselines.md
    uv run heatwave hub-push-all       upload reference/raw/runs/labels/features/models to HF
    uv run heatwave hub-pull [--only]  download them (or only some top-level folders)
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from heatwave.config import REPO_ROOT, load_problem, read_yaml


def cmd_show_problem(args: argparse.Namespace) -> None:
    p = load_problem()
    t, v = p.target, p.validation
    print(f"target       {t.name}  (Tmax source: {t.tmax_source})")
    print(
        f"rule         heatwave: Tmax >= {t.tmax_floor_c}"
        f" and departure >= {t.departure_heatwave_c}, or Tmax >= {t.absolute_heatwave_c}"
    )
    print(
        f"             severe:   Tmax >= {t.tmax_floor_c} and departure > {t.departure_severe_c},"
        f" or Tmax >= {t.absolute_severe_c}"
    )
    print(f"normal       {t.normal_period[0]}-{t.normal_period[1]}, {t.normal_smoothing}")
    print(
        f"dates        issue = as-of + {p.latency_days} days;"
        f" leads {p.horizon_min}..{p.horizon_max}"
    )
    print(
        f"seasons      evaluate {p.evaluation_season.start}..{p.evaluation_season.end}, "
        f"train {p.training_season.start}..{p.training_season.end}"
    )
    print(
        f"blocks       development {v.development}, final test {v.final_test}, "
        f"2026 season {v.season_2026}"
    )
    for f in v.folds:
        print(f"  fold {f.fold}     train {f.train}  validate {f.validate}")
    for tier in p.tiers:
        print(f"tier         {tier.name:8s} {tier.rule} {tier.params}")


def cmd_build_geo(args: argparse.Namespace) -> None:
    from heatwave.geo.weights import build_geo  # geo stack is slow to import

    print(json.dumps(build_geo(read_yaml("data.yaml"), REPO_ROOT), indent=2))


MODEL_PATHS = ("src", "configs", "pyproject.toml", "uv.lock")  # what can change a model


def _code_version() -> str | None:
    """The git commit of this checkout, recorded with results."""
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *MODEL_PATHS],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("+uncommitted" if dirty else "")


def _ingest_context():
    from heatwave.ingest.run import Paths

    cfg = read_yaml("data.yaml")
    return cfg, Paths.from_config(cfg, REPO_ROOT)


def cmd_ingest_backfill(args: argparse.Namespace) -> None:
    from heatwave.ingest.run import run_ingest

    cfg, paths = _ingest_context()
    start = date.fromisoformat(cfg["power"]["start"])
    end = date.fromisoformat(args.end) if args.end else date.today()
    # Responses are cached, so an interrupted backfill resumes without re-downloading.
    run_ingest(cfg, paths, start, end, kind="backfill", use_cache=True)


def cmd_ingest_daily(args: argparse.Namespace) -> None:
    from heatwave import hub
    from heatwave.ingest.run import daily_window, run_ingest, utc_now

    cfg, paths = _ingest_context()
    start, end = daily_window(utc_now().date(), cfg["power"]["trailing_days"])
    if args.sync:
        years = range(start.year, end.year + 1)
        patterns = ["reference/*", "runs/*", *[f"raw/power/obs_year={y}/*" for y in years]]
        hub.pull(cfg, paths.root, patterns)
    try:
        run_ingest(cfg, paths, start, end, kind="daily")
    finally:
        if args.sync:
            # Upload whatever was stored, even if some cells failed.
            runs = pd.read_parquet(paths.runs_log)
            run_id = runs["run_id"].iloc[-1]
            files = [*hub.files_of_run(paths.raw_power, run_id), paths.runs_log]
            hub.push(cfg, paths.root, files, f"daily ingest {run_id}")


def cmd_check_raw(args: argparse.Namespace) -> None:
    from heatwave.ingest.run import check_spine

    cfg, paths = _ingest_context()
    from heatwave.ingest.raw_store import defects_from_config

    report = check_spine(
        paths,
        date.fromisoformat(cfg["power"]["start"]),
        cfg["power"]["parameters"],
        defects_from_config(cfg, paths.runs_log),
    )
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit("raw data incomplete")


def cmd_hub_push_all(args: argparse.Namespace) -> None:
    from heatwave import hub

    cfg, paths = _ingest_context()
    merged = hub.merge_remote_log(cfg, paths.root, paths.runs_log)
    if merged:
        print(f"merged {merged} run-log rows written remotely since the last pull")
    files = sorted(
        f
        for sub in ("reference", "raw", "runs", "labels", "features", "models")
        for f in (paths.root / sub).rglob("*")
        if f.is_file()
    )
    n = hub.push(cfg, paths.root, files, "sync reference, raw, runs, labels, features, models")
    print(f"uploaded {n} files")


def cmd_ingest_gsod(args: argparse.Namespace) -> None:
    import requests

    from heatwave.geo.weights import load_districts
    from heatwave.ingest import gsod

    cfg, paths = _ingest_context()
    g = cfg["gsod"]
    resp = requests.get(g["history_url"], timeout=120)
    resp.raise_for_status()
    up = load_districts(cfg["boundaries"], REPO_ROOT).geometry.union_all()
    stations = gsod.station_list(resp.content, up, g["boundary_buffer_deg"])
    stations.to_parquet(paths.root / "reference" / "stations.parquet", index=False)
    print(f"{len(stations)} stations in or near UP")
    manifest = gsod.download(g, stations, paths.root / g["raw_dir"])
    ok = manifest[manifest["status"] == "ok"]
    print(f"{len(ok)} station-years downloaded from {ok['station_id'].nunique()} stations")


def cmd_build_labels(args: argparse.Namespace) -> None:
    from heatwave.labels.pipeline import run

    cfg, paths = _ingest_context()
    summary = run(
        load_problem(),
        cfg,
        read_yaml("labels.yaml"),
        paths.root,
        REPO_ROOT / "reports",
        REPO_ROOT / cfg["outputs"]["silver_dir"],
    )
    print(json.dumps(summary, indent=2, default=str))


def cmd_build_features(args: argparse.Namespace) -> None:
    from heatwave.features.pipeline import run

    cfg, paths = _ingest_context()
    manifest = run(
        load_problem(),
        read_yaml("features.yaml"),
        paths.root,
        REPO_ROOT / cfg["outputs"]["silver_dir"],
        REPO_ROOT / "docs",
    )
    print(json.dumps(manifest, indent=2))


def cmd_evaluate_baselines(args: argparse.Namespace) -> None:
    from heatwave.models.baselines import M2Lead
    from heatwave.validation import report, walk_forward

    cfg, paths = _ingest_context()
    problem, model_cfg = load_problem(), read_yaml("models.yaml")
    region_map = walk_forward.regions(REPO_ROOT / "configs" / "regions.yaml")
    oof_dir = paths.root / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for target in ("lead", "window"):
        out[target] = walk_forward.run(
            problem, model_cfg, paths.root, region_map, target, ["M0", "M1", "M2"]
        )
        out[target].to_parquet(oof_dir / f"baselines_{target}.parquet", index=False)
    labels = pd.read_parquet(
        paths.root / "labels" / "labels_daily.parquet",
        columns=["district_id", "obs_date", "is_heatwave"],
    )
    train, _ = walk_forward.fold_tables(
        problem, problem.validation.folds[-1], paths.root, labels, "lead", walk_forward.COLUMNS
    )
    coefs = M2Lead(model_cfg).fit(train).coefficients()
    summary = report.build(
        out["lead"], out["window"], model_cfg, REPO_ROOT / "reports" / "baselines.md", coefs
    )
    print(json.dumps(summary, indent=2, default=str))


def cmd_tune_m3(args: argparse.Namespace) -> None:
    from heatwave.validation import tuning

    cfg, paths = _ingest_context()
    problem, model_cfg = load_problem(), read_yaml("models.yaml")
    bins = paths.root / "cache" / "tuning" / args.target  # rebuilt where needed, not synced
    models = paths.root / "models"
    models.mkdir(parents=True, exist_ok=True)
    study = models / f"m3_tuning_{args.target}_study.db"
    result = models / f"m3_tuning_{args.target}.json"
    push = None
    if args.push:
        from heatwave import hub

        def push():
            hub.push(cfg, paths.root, [study, result], f"M3 tuning ({args.target}): study")

    out = tuning.tune(
        problem,
        model_cfg,
        paths.root,
        args.target,
        bins,
        study,
        tuning.runtime(args.device, args.threads),
        push,
    )
    out["code_version"] = _code_version()
    result.write_text(json.dumps(out, indent=2))
    if push:
        push()
    print(json.dumps(out, indent=2))


def cmd_evaluate_m3_tuned(args: argparse.Namespace) -> None:
    from heatwave.models.gbm import M3
    from heatwave.validation import tuning, walk_forward

    _, paths = _ingest_context()
    problem, model_cfg = load_problem(), read_yaml("models.yaml")
    region_map = walk_forward.regions(REPO_ROOT / "configs" / "regions.yaml")
    folds = [f.fold for f in problem.validation.folds]
    oof_dir = paths.root / "oof"
    for target in ("lead", "window"):
        study = paths.root / "models" / f"m3_tuning_{target}_study.db"
        choice = tuning.nested_choice(tuning.load_study(study, target), folds)
        (paths.root / "models" / f"m3_nested_choice_{target}.json").write_text(
            json.dumps(choice, indent=2)
        )
        print(target, {k: v["trial"] for k, v in choice.items()})
        walk_forward.VARIANTS["M3T"] = lambda t, fold, c=choice: M3(
            model_cfg,
            t,
            problem.validation.inner_block_seasons,
            eval_season=problem.evaluation_season,
            params=c[fold]["params"],
        )
        oof = walk_forward.run(
            problem,
            model_cfg,
            paths.root,
            region_map,
            target,
            ["M0", "M3T"],
            checkpoint=oof_dir / "checkpoints_m3_tuned",
        )
        oof.to_parquet(oof_dir / f"m3_tuned_{target}.parquet", index=False)
        print(target, "done; fits", json.dumps(oof.attrs["fits"]))


def cmd_evaluate_sliding(args: argparse.Namespace) -> None:
    from heatwave.models import baselines
    from heatwave.validation import walk_forward

    _, paths = _ingest_context()
    problem, model_cfg = load_problem(), read_yaml("models.yaml")
    region_map = walk_forward.regions(REPO_ROOT / "configs" / "regions.yaml")
    walk_forward.VARIANTS["M2S"] = lambda t, fold: baselines.SlidingWindow(
        baselines.make("M2", t, model_cfg), t, model_cfg["baselines"]["m2"]["sliding_years"]
    )
    for target in ("lead", "window"):
        oof = walk_forward.run(
            problem, model_cfg, paths.root, region_map, target, ["M0", "M2", "M2S"]
        )
        oof.to_parquet(paths.root / "oof" / f"m2_sliding_{target}.parquet", index=False)


def cmd_freeze_model(args: argparse.Namespace) -> None:
    """Refuses to overwrite a frozen version or to freeze uncommitted code, unless the
    run is a --dry-run, which writes only to the given directory."""
    import subprocess

    from heatwave.validation import freeze

    _, paths = _ingest_context()
    problem, model_cfg = load_problem(), read_yaml("models.yaml")
    frozen = model_cfg["frozen"]
    version = frozen["version"]
    if args.dry_run:
        json_dir = table_dir = report_dir = Path(args.dry_run)
    else:
        json_dir = REPO_ROOT / frozen["dir"]
        table_dir = paths.root / frozen["table_dir"]
        report_dir = REPO_ROOT / "reports"
        if (json_dir / f"{version}.json").exists():
            raise SystemExit(f"{version} exists: a frozen version is never overwritten")
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *MODEL_PATHS],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            raise SystemExit("commit these before freezing:\n" + dirty)
    spec, oof = freeze.build(problem, model_cfg, paths.root, table_dir, version, _code_version())
    json_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / f"{version}.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    text = freeze.report(spec, oof, problem, model_cfg, paths.root)
    report_path = report_dir / f"model_{version}.md"
    report_path.write_text(text, encoding="utf-8")
    keys = ("version", "code_version", "fit_years", "tiers")
    print(json.dumps({k: spec[k] for k in keys}, indent=2))
    print(f"wrote {json_dir / (version + '.json')}, {table_dir}, {report_path}")


FINAL_TEST_REPORT = REPO_ROOT / "reports" / "final_test.md"
REHEARSAL_YEARS = (2016, 2020)  # development years only


def check_rehearsal_years(years: tuple[int, int], development: tuple[int, int]) -> None:
    if years[0] < development[0] or years[1] > development[1]:
        raise SystemExit(f"a rehearsal may use development years only, not {years}")


def _final_test_guards(bar_path: Path, model_path: Path, tag: str) -> None:
    """The real run: once only, on the tagged model, with a committed bar and code."""
    import subprocess

    started = FINAL_TEST_REPORT.with_suffix(".started")
    if FINAL_TEST_REPORT.exists():
        raise SystemExit(f"{FINAL_TEST_REPORT} exists: the final test runs only once")
    if started.exists():
        raise SystemExit(f"{started} exists: a run already started. Read it before doing anything")

    def git(*a):
        return subprocess.run(
            ["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )

    if not bar_path.exists() or git("ls-files", "--error-unmatch", str(bar_path)).returncode:
        raise SystemExit(f"{bar_path} must be written and committed before the test")
    dirty = git("status", "--porcelain", "--", *MODEL_PATHS).stdout.strip()
    if dirty:
        raise SystemExit("commit these before the test:\n" + dirty)
    if git("diff", "--quiet", tag, "--", str(model_path)).returncode:
        raise SystemExit(f"{model_path} differs from tag {tag}")


def cmd_final_test(args: argparse.Namespace) -> None:
    from heatwave.validation import final_test as ft

    if args.rescore_info and not FINAL_TEST_REPORT.exists():
        raise SystemExit("--rescore-info is only for after the final test has run")
    if not (args.write_bar or args.rehearse or args.run_once or args.rescore_info):
        print(
            "Nothing run. Use --write-bar, --rehearse DIR, or --run-once (the real, single test)."
        )
        return
    _, paths = _ingest_context()
    model_cfg = read_yaml("models.yaml")
    frozen = model_cfg["frozen"]
    model_path = REPO_ROOT / frozen["dir"] / f"{frozen['version']}.json"
    bar_path = REPO_ROOT / frozen["dir"] / f"{frozen['version']}_g4_bar.json"
    if args.run_once:
        _final_test_guards(bar_path, model_path, frozen["version"])
    problem = load_problem()

    from heatwave.models.frozen import FrozenModel
    from heatwave.validation import walk_forward

    model = FrozenModel.load(model_path, paths.root / frozen["table_dir"])
    if args.write_bar:
        if bar_path.exists():
            raise SystemExit(f"{bar_path} exists: the bar does not change once written")
        oof = {
            t: pd.read_parquet(paths.root / "oof" / f"{frozen['oof']}_{t}.parquet")
            for t in ("lead", "window")
        }
        col = f"p_{frozen['oof_model']}"
        bar = ft.development_bar(model.spec, oof["lead"], oof["window"], col)
        bar_path.write_text(json.dumps(bar, indent=2), encoding="utf-8")
        print(json.dumps(bar["range"], indent=2))
        return

    bar = json.loads(bar_path.read_text(encoding="utf-8"))
    if args.rescore_info:
        region_map = walk_forward.regions(REPO_ROOT / "configs" / "regions.yaml")
        lead, window = ft.load_rows(
            problem, paths.root, problem.validation.final_test, model, region_map, True
        )
        outcome = ft.score(lead, window, model.spec, bar, model_cfg)
        out = REPO_ROOT / "reports" / args.rescore_info
        title = "Information only: 2021-2025 re-scored after a label update"
        note = (
            "**This is not a test.** These years are spent and were examined before this run. "
            "The G4 verdict stays as in reports/final_test.md."
        )
        out.write_text(
            ft.report(lead, window, outcome, bar, model.spec, model_cfg, title, note),
            encoding="utf-8",
        )
        print(json.dumps(outcome["result"], indent=2))
        print(f"wrote {out}")
        return
    region_map = walk_forward.regions(REPO_ROOT / "configs" / "regions.yaml")
    if args.run_once:
        years, title = problem.validation.final_test, "Final test (gate G4)"
        out = FINAL_TEST_REPORT
        # Before any test row is read: a crash then cannot lead to a quiet second attempt.
        out.with_suffix(".started").write_text(
            f"started with {_code_version()}\n", encoding="utf-8"
        )
    else:
        years, title = REHEARSAL_YEARS, "REHEARSAL on development years: not a result"
        check_rehearsal_years(years, problem.validation.development)
        out = Path(args.rehearse) / "final_test_rehearsal.md"
        out.parent.mkdir(parents=True, exist_ok=True)
    lead, window = ft.load_rows(problem, paths.root, years, model, region_map, bool(args.run_once))
    outcome = ft.score(lead, window, model.spec, bar, model_cfg)
    out.write_text(
        ft.report(lead, window, outcome, bar, model.spec, model_cfg, title), encoding="utf-8"
    )
    if args.run_once:
        out.with_suffix(".started").unlink()
    print(title, "->", outcome["verdict"])
    print(json.dumps(outcome, indent=2))
    print(f"wrote {out}")


def cmd_drift_check(args: argparse.Namespace) -> None:
    from heatwave.geo.weights import load_districts
    from heatwave.ingest import gsod
    from heatwave.labels import drift_check as dc
    from heatwave.validation.report import _md

    cfg, paths = _ingest_context()
    g = cfg["gsod"]
    stations = pd.read_parquet(paths.root / "reference" / "stations.parquet")
    if args.download:
        gsod.download({**g, "years": g["check_years"]}, stations, paths.root / g["check_raw_dir"])
    where = dc.station_districts(
        stations, load_districts(cfg["boundaries"], REPO_ROOT), cfg["geo"]["equal_area_crs"]
    )
    labels = pd.read_parquet(
        paths.root / "labels" / "labels_daily.parquet",
        columns=["district_id", "obs_date", "tmax_raw", "tmax_used", "label_version"],
    )
    imd = pd.read_parquet(
        REPO_ROOT / cfg["outputs"]["silver_dir"] / "imd_district_daily.parquet",
        columns=["district_id", "obs_date", "tmax_imd"],
    )
    months = sorted({m for m, _ in load_problem().evaluation_season.days()})
    d = dc.paired_days(dc.load_days(paths.root, g), where, labels, imd, months)
    out = dc.period_change(d, tuple(args.before), tuple(args.after))
    text = (
        "# Label drift check\n\nGenerated by `heatwave drift-check`. Each source minus reported "
        f"station maxima, label `{labels['label_version'].iloc[0]}`, change from "
        f"{args.before[0]}-{args.before[1]} to {args.after[0]}-{args.after[1]}.\n\n"
        + _md(out["summary"], 2)
        + "\n## By station\n\n"
        + _md(out["by_station"], 2)
        + "\n## Yearly (stations weighted equally)\n\n"
        + _md(out["yearly"], 2)
    )
    (REPO_ROOT / "reports" / "label_drift_check.md").write_text(text, encoding="utf-8")
    print(out["summary"].round(3).to_string(index=False))


def cmd_forecast(args: argparse.Namespace) -> None:
    """Today's forecast (India date), after the daily gate; a failed gate means degraded,
    never a stale forecast. A past --issue-date is a replay: no gate, written to --out,
    using today's best-known data rather than what was known that day."""
    from heatwave.ingest.run import utc_now
    from heatwave.pipeline import daily
    from heatwave.quality.report import validate_recent

    if args.issue_date and not args.out:
        raise SystemExit("a replay with --issue-date needs --out: it is not a real forecast")
    cfg, paths = _ingest_context()
    problem, model_cfg = load_problem(), read_yaml("models.yaml")
    frozen = model_cfg["frozen"]
    issue = args.issue_date or (utc_now() + pd.Timedelta(hours=5, minutes=30)).date().isoformat()
    ok, why = True, ""
    if not args.issue_date:
        gate = validate_recent(cfg, read_yaml("quality.yaml"), paths, utc_now().date())
        ok, why = gate["ok"], "; ".join(gate["problems"])
    plan, rows = daily.run(
        problem,
        cfg,
        read_yaml("features.yaml"),
        model_cfg,
        paths.root,
        REPO_ROOT / frozen["dir"] / f"{frozen['version']}.json",
        issue,
        ok,
        why,
    )
    out = daily.forecast_path(Path(args.out) if args.out else paths.root, issue)
    daily.write(rows, out)
    window = rows[rows["target"] == "window"]
    summary = {
        "issue_date": issue,
        "status": plan.status,
        "reason": plan.reason,
        "asof_date": str(plan.asof_date.date()) if plan.asof_date is not None else None,
        "data_age_days": plan.data_age_days,
        "rows": len(rows),
        "in_season": bool(window["in_season"].iloc[0]),
        "tiers": window["tier"].value_counts().to_dict(),
        "window_mean_p": None if window["p"].isna().all() else round(float(window["p"].mean()), 4),
        "file": str(out),
    }
    print(json.dumps(summary, indent=2))


def _publish_context():
    cfg, paths = _ingest_context()
    model_cfg = read_yaml("models.yaml")
    frozen = model_cfg["frozen"]
    spec = json.loads(
        (REPO_ROOT / frozen["dir"] / f"{frozen['version']}.json").read_text(encoding="utf-8")
    )
    d = pd.read_parquet(paths.root / "reference" / "districts.parquet")
    names = dict(zip(d["district_id"], d["name"], strict=True))
    return cfg, paths, read_yaml("publish.yaml"), spec, names


def cmd_outcomes(args: argparse.Namespace) -> None:
    from heatwave.ingest.raw_store import RawStore
    from heatwave.ingest.run import utc_now
    from heatwave.pipeline import daily, outcomes

    cfg, paths = _ingest_context()
    year = utc_now().year
    store = RawStore.from_config(cfg, paths.root)
    _, labels = daily.recent_inputs(
        store,
        paths.root,
        cfg,
        load_problem(),
        daily.district_statics(paths.root),
        [y for y in (year - 1, year) if y in store.years()],
    )
    for path in outcomes.run(paths.root, labels):
        print(f"wrote {path}")


def cmd_build_site(args: argparse.Namespace) -> None:
    from heatwave.publish import site
    from heatwave.validation import walk_forward

    _, paths, pub, spec, names = _publish_context()
    from heatwave.monitoring import scorecard

    regions = walk_forward.regions(REPO_ROOT / "configs" / "regions.yaml")
    files = sorted((paths.root / "outcomes").glob("*.parquet"))
    outcomes = pd.concat([pd.read_parquet(x) for x in files]) if files else None
    out = site.build(
        paths.root / "forecasts",
        Path(args.out),
        names,
        regions,
        spec,
        pub["repo_url"],
        pub["keep_days"],
        scorecard.body_html(outcomes),
    )
    print(json.dumps(out))


def cmd_send_alerts(args: argparse.Namespace) -> None:
    from heatwave.publish import alerts

    if args.test:
        pub = read_yaml("publish.yaml")
        alerts.send_test(pub, pub["site_url"])
        print("test email sent")
        return
    _, paths, pub, _, names = _publish_context()
    files = sorted((paths.root / "forecasts").glob("issue_year=*/*.parquet"), key=lambda f: f.stem)
    if not files:
        raise SystemExit("no forecasts")
    rows = pd.read_parquet(files[-1])
    print(alerts.send(rows, names, pub, pub["site_url"], paths.root / "alerts"))


def cmd_hub_push_outputs(args: argparse.Namespace) -> None:
    from heatwave import hub

    cfg, paths = _ingest_context()
    recent = sorted((paths.root / "forecasts").glob("issue_year=*/*.parquet"), key=lambda f: f.stem)
    files = [
        *recent[-3:],
        *(paths.root / "outcomes").glob("*.parquet"),
        *(paths.root / "alerts").glob("*.txt"),
        *(paths.root / "monitoring").glob("*.parquet"),
    ]
    print(f"uploaded {hub.push(cfg, paths.root, files, 'daily forecast outputs')} files")


def cmd_monitor(args: argparse.Namespace) -> None:
    """The daily checks on the newest forecast. --write-baseline fixes their limits from
    development data; it never overwrites."""
    from heatwave.features.build import compute_features
    from heatwave.features.pipeline import load_climatology
    from heatwave.ingest.raw_store import RawStore
    from heatwave.ingest.run import utc_now
    from heatwave.monitoring import checks
    from heatwave.pipeline import daily
    from heatwave.publish import alerts

    cfg, paths, pub, spec, _ = _publish_context()
    frozen = read_yaml("models.yaml")["frozen"]
    base_path = REPO_ROOT / frozen["dir"] / f"{frozen['version']}_monitor.json"
    if args.write_baseline:
        if base_path.exists():
            raise SystemExit(f"{base_path} exists: the limits do not change with the model frozen")
        window = pd.read_parquet(paths.root / "oof" / f"{frozen['oof']}_window.parquet")
        features = pd.read_parquet(
            paths.root / "features" / "development.parquet",
            columns=["asof_date", *checks.ANOMALY_FEATURES],
        )
        checks.write_baseline(spec, window, features, base_path, f"p_{frozen['oof_model']}")
        print(f"wrote {base_path}")
        return
    baseline = json.loads(base_path.read_text(encoding="utf-8"))
    files = sorted((paths.root / "forecasts").glob("issue_year=*/*.parquet"), key=lambda f: f.stem)
    if not files:
        raise SystemExit("no forecasts")
    rows = pd.read_parquet(files[-1])
    found = checks.forecast_checks(rows, baseline)
    asof = rows["asof_date"].iloc[0]
    if pd.notna(asof):
        problem = load_problem()
        districts = daily.district_statics(paths.root)
        store = RawStore.from_config(cfg, paths.root)
        t = pd.Timestamp(asof)
        years = [y for y in (t.year - 1, t.year) if y in store.years()]
        d, labels = daily.recent_inputs(store, paths.root, cfg, problem, districts, years)
        d, labels = d[d["obs_date"] <= t], labels[labels["obs_date"] <= t]
        clim = load_climatology(paths.root / "features" / "development_climatology")
        feats = compute_features(d, labels, clim, districts, read_yaml("features.yaml"))
        found += checks.feature_checks(feats, t, baseline)
    else:
        found += checks.feature_checks(pd.DataFrame(), None, baseline)
    checks.append_log(
        paths.root / "monitoring" / "daily.parquet", checks.log_row(rows, found, utc_now())
    )
    for c in found:
        print(f"{c['level']:8s}{c['check']}: {c['detail']}")
    issue = files[-1].stem
    print(
        alerts.send_problems(
            checks.problems(found),
            issue,
            pub,
            pub["site_url"],
            paths.root / "alerts",
            f"Heatwave pipeline: failed checks ({issue})",
        )
    )


def cmd_monitor_monthly(args: argparse.Namespace) -> None:
    from heatwave import hub
    from heatwave.ingest import gsod
    from heatwave.ingest.raw_store import RawStore
    from heatwave.ingest.run import utc_now
    from heatwave.labels import drift_check
    from heatwave.monitoring import monthly
    from heatwave.pipeline import daily
    from heatwave.publish import alerts

    cfg, paths, pub, spec, _ = _publish_context()
    problem, g = load_problem(), cfg["gsod"]
    today = utc_now()
    store = RawStore.from_config(cfg, paths.root)
    years = [y for y in store.years() if y >= today.year - 1]
    skew = monthly.skew(store, years)

    # The current year's stations change daily, so always a fresh copy.
    live = paths.root / "cache" / "gsod_live"
    stations = pd.read_parquet(paths.root / "reference" / "stations.parquet")
    (live / "manifest.parquet").unlink(missing_ok=True)
    gsod.download({**g, "years": [today.year, today.year]}, stations, live)
    days = monthly.read_days(paths.root, g, live)
    where_file = paths.root / "reference" / "station_districts.parquet"
    if where_file.exists():
        where = pd.read_parquet(where_file)
    else:
        from heatwave.geo.weights import load_districts

        where = drift_check.station_districts(
            stations, load_districts(cfg["boundaries"], REPO_ROOT), cfg["geo"]["equal_area_crs"]
        )
        where.to_parquet(where_file, index=False)
    t = problem.target
    reference = t.bias_drift_hold_years
    label_years = sorted({*range(reference[0], reference[1] + 1), today.year})
    _, labels = daily.recent_inputs(
        store,
        paths.root,
        cfg,
        problem,
        daily.district_statics(paths.root),
        [y for y in label_years if y in store.years()],
    )
    months = sorted({m for m, _ in problem.evaluation_season.days()})
    label = monthly.label_against_stations(days, where, labels, today.year, reference, months)

    out = {
        "run_at": str(today),
        "skew": json.loads(skew.to_json(orient="records")),
        "label_against_stations": label,
    }
    month = f"{today:%Y-%m}"
    path = paths.root / "monitoring" / f"monthly_{month}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(skew.round(3).to_string(index=False))
    print(label["level"], label["detail"])
    flagged = monthly.skew_flags(skew, spec["features"])
    if label["level"] == "warn":
        flagged.append({"check": "label_against_stations", "detail": label["detail"]})
    print(
        alerts.send_problems(
            flagged,
            month,
            pub,
            pub["site_url"],
            paths.root / "alerts",
            f"Heatwave pipeline: monthly checks need a look ({month})",
        )
    )
    if args.push:
        n = hub.push(cfg, paths.root, [path, where_file], f"monthly checks {month}")
        print(f"uploaded {n} files")


def cmd_evaluate_production(args: argparse.Namespace) -> None:
    """Walk-forward of the frozen model's feature set, beside the same model with every
    baseline feature, on the same folds. Development years only."""
    from heatwave.models import baselines
    from heatwave.validation import freeze, walk_forward

    _, paths = _ingest_context()
    problem, model_cfg = load_problem(), read_yaml("models.yaml")
    frozen = model_cfg["frozen"]
    years = model_cfg["baselines"]["m2"]["sliding_years"]
    production = freeze.production_cfg(model_cfg)
    region_map = walk_forward.regions(REPO_ROOT / "configs" / "regions.yaml")
    walk_forward.VARIANTS["M2S"] = lambda t, fold: baselines.SlidingWindow(
        baselines.make("M2", t, model_cfg), t, years
    )
    walk_forward.VARIANTS[frozen["oof_model"]] = lambda t, fold: baselines.SlidingWindow(
        baselines.make("M2", t, production), t, years
    )
    for target in ("lead", "window"):
        oof = walk_forward.run(
            problem, model_cfg, paths.root, region_map, target, ["M0", "M2S", frozen["oof_model"]]
        )
        oof.to_parquet(paths.root / "oof" / f"{frozen['oof']}_{target}.parquet", index=False)


def cmd_bench_m3(args: argparse.Namespace) -> None:
    from heatwave.validation import tuning

    _, paths = _ingest_context()
    problem, model_cfg = load_problem(), read_yaml("models.yaml")
    setups = {d: tuning.runtime(d, args.threads) for d in args.devices}
    out = tuning.bench(
        problem,
        model_cfg,
        paths.root,
        args.target,
        paths.root / "cache" / "tuning" / args.target,
        args.fold,
        setups,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cmd_build_district_daily(args: argparse.Namespace) -> None:
    from heatwave.geo.aggregate import imd_district_daily, power_district_daily
    from heatwave.ingest.imd import ImdSettings
    from heatwave.ingest.raw_store import RawStore

    cfg, paths = _ingest_context()
    ref = paths.root / "reference"
    out_dir = REPO_ROOT / cfg["outputs"]["silver_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    power = power_district_daily(
        RawStore.from_config(cfg, paths.root),
        pd.read_parquet(ref / "district_grid_weights.parquet"),
        cfg["power"]["parameters"],
        cfg["geo"]["cellmax_min_share"],
    )
    power.to_parquet(out_dir / "power_district_daily.parquet", index=False)

    imd_cfg = cfg["imd"]
    imd_dir = paths.root / imd_cfg["raw_dir"]
    years = sorted(int(p.stem) for p in imd_dir.glob("*.GRD"))
    imd = imd_district_daily(
        ImdSettings.from_config(imd_cfg),
        imd_dir,
        years,
        pd.read_parquet(ref / "district_imd_weights.parquet"),
    )
    imd.to_parquet(out_dir / "imd_district_daily.parquet", index=False)
    print(
        json.dumps(
            {
                "power_rows": len(power),
                "power_period": [
                    str(power.obs_date.min().date()),
                    str(power.obs_date.max().date()),
                ],
                "power_missing_cell_values": int(power["n_missing"].sum()),
                "imd_rows": len(imd),
                "imd_period": [str(imd.obs_date.min().date()), str(imd.obs_date.max().date())],
                "imd_missing_cell_values": int(imd["n_missing"].sum()),
                "districts": int(power["district_id"].nunique()),
            },
            indent=2,
        )
    )


def cmd_qa_report(args: argparse.Namespace) -> None:
    from heatwave.quality.report import build_report

    cfg, paths = _ingest_context()
    summary = build_report(
        cfg, read_yaml("quality.yaml"), paths, REPO_ROOT / "reports" / "data_qa.md"
    )
    print(json.dumps(summary, indent=2))
    if not summary["ok"]:
        raise SystemExit("data QA failed: see reports/data_qa.md")


def cmd_validate(args: argparse.Namespace) -> None:
    from heatwave.ingest.run import utc_now
    from heatwave.quality.report import validate_recent

    cfg, paths = _ingest_context()
    result = validate_recent(cfg, read_yaml("quality.yaml"), paths, utc_now().date())
    print(json.dumps(result, indent=2, default=str))
    if not result["ok"]:
        raise SystemExit("validation failed: " + "; ".join(result["problems"]))


def cmd_ingest_imd(args: argparse.Namespace) -> None:
    from heatwave.ingest.imd import ImdSettings, ingest_years

    cfg, paths = _ingest_context()
    imd = cfg["imd"]
    first = imd["start_year"]
    last = int(args.last_year) if args.last_year else utc_now_year() - 1
    manifest = ingest_years(
        ImdSettings.from_config(imd), paths.root / imd["raw_dir"], list(range(first, last + 1))
    )
    print(f"{len(manifest)} years in manifest: {manifest['year'].min()}-{manifest['year'].max()}")


def utc_now_year() -> int:
    from heatwave.ingest.run import utc_now

    return utc_now().year


def cmd_hub_pull(args: argparse.Namespace) -> None:
    from heatwave import hub

    cfg, paths = _ingest_context()
    parts = args.only or ["reference", "raw", "runs", "labels", "features", "models"]
    hub.pull(cfg, paths.root, [f"{p}/**" for p in parts])
    print(f"pulled into {paths.root}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="heatwave")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show-problem", help="print the frozen problem definition").set_defaults(
        func=cmd_show_problem
    )
    sub.add_parser(
        "build-geo", help="district polygons -> grid cells -> area weights (data/reference/)"
    ).set_defaults(func=cmd_build_geo)

    backfill = sub.add_parser("ingest-backfill", help="fetch full POWER history for every cell")
    backfill.add_argument("--end", help="last date to request (default: today)")
    backfill.set_defaults(func=cmd_ingest_backfill)

    daily = sub.add_parser("ingest-daily", help="re-fetch the trailing window for every cell")
    daily.add_argument("--sync", action="store_true", help="pull from / push to Hugging Face")
    daily.set_defaults(func=cmd_ingest_daily)

    sub.add_parser("check-raw", help="completeness of the raw store").set_defaults(
        func=cmd_check_raw
    )
    sub.add_parser(
        "build-district-daily", help="cells -> district daily tables for POWER and IMD (silver)"
    ).set_defaults(func=cmd_build_district_daily)
    sub.add_parser(
        "ingest-gsod", help="NOAA station daily maxima for UP, development years (resumable)"
    ).set_defaults(func=cmd_ingest_gsod)
    sub.add_parser(
        "build-labels", help="station comparison, bias map, labels, episodes, QA reports"
    ).set_defaults(func=cmd_build_labels)
    sub.add_parser(
        "build-features", help="feature tables per fold + development, climatologies, dictionary"
    ).set_defaults(func=cmd_build_features)
    sub.add_parser(
        "evaluate-baselines", help="walk-forward M0-M2, OOF predictions, reports/baselines.md"
    ).set_defaults(func=cmd_evaluate_baselines)
    sub.add_parser("qa-report", help="full-history data QA -> reports/data_qa.md").set_defaults(
        func=cmd_qa_report
    )
    sub.add_parser("validate", help="daily gate: hard checks, completeness, latency").set_defaults(
        func=cmd_validate
    )
    tune = sub.add_parser("tune-m3", help="Optuna tuning of LightGBM on inner blocks (resumable)")
    tune.add_argument("--target", choices=["lead", "window"], default="lead")
    tune.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    tune.add_argument("--threads", type=int, default=None, help="default: models.yaml")
    tune.add_argument("--push", action="store_true", help="upload the study to HF after each trial")
    tune.set_defaults(func=cmd_tune_m3)
    sub.add_parser(
        "evaluate-m3-tuned", help="walk-forward of LightGBM with fold-honest tuned parameters"
    ).set_defaults(func=cmd_evaluate_m3_tuned)
    sub.add_parser(
        "evaluate-sliding", help="walk-forward of logistic regression, sliding vs expanding"
    ).set_defaults(func=cmd_evaluate_sliding)
    sub.add_parser(
        "evaluate-production", help="walk-forward of the frozen model's feature set"
    ).set_defaults(func=cmd_evaluate_production)
    fz = sub.add_parser("freeze-model", help="fit and freeze the production model and thresholds")
    fz.add_argument("--dry-run", metavar="DIR", help="write only to DIR; nothing is frozen")
    fz.set_defaults(func=cmd_freeze_model)
    ft = sub.add_parser("final-test", help="gate G4; without a flag nothing runs")
    ft.add_argument(
        "--write-bar", action="store_true", help="write the pass bar (development only)"
    )
    ft.add_argument("--rehearse", metavar="DIR", help="whole path on development years only")
    ft.add_argument(
        "--run-once", action="store_true", help="the real test on 2021-2025: single use"
    )
    ft.add_argument("--rescore-info", metavar="NAME", help="after the test: re-score, not a test")
    ft.set_defaults(func=cmd_final_test)
    fc = sub.add_parser("forecast", help="one day's forecast from the frozen model")
    fc.add_argument("--issue-date", help="replay a past date (needs --out)")
    fc.add_argument("--out", help="root folder for a replay's output")
    fc.set_defaults(func=cmd_forecast)
    sub.add_parser("outcomes", help="what happened, for forecasts already observed").set_defaults(
        func=cmd_outcomes
    )
    bs = sub.add_parser("build-site", help="the public static site and downloads")
    bs.add_argument("--out", default="_site")
    bs.set_defaults(func=cmd_build_site)
    sa = sub.add_parser("send-alerts", help="email if a district reaches Warning")
    sa.add_argument("--test", action="store_true", help="send one test email and stop")
    sa.set_defaults(func=cmd_send_alerts)
    sub.add_parser("hub-push-outputs", help="upload forecasts, outcomes, alert marks").set_defaults(
        func=cmd_hub_push_outputs
    )
    mon = sub.add_parser("monitor", help="daily checks on the newest forecast")
    mon.add_argument("--write-baseline", action="store_true", help="fix the limits, once")
    mon.set_defaults(func=cmd_monitor)
    mm = sub.add_parser("monitor-monthly", help="provisional against final; label against stations")
    mm.add_argument("--push", action="store_true", help="upload the result to the private dataset")
    mm.set_defaults(func=cmd_monitor_monthly)
    dc = sub.add_parser("drift-check", help="label, POWER and IMD Tmax against station maxima")
    dc.add_argument("--download", action="store_true", help="fetch check_years station data first")
    dc.add_argument("--before", nargs=2, type=int, default=[2016, 2020])
    dc.add_argument("--after", nargs=2, type=int, default=[2021, 2025])
    dc.set_defaults(func=cmd_drift_check)
    bench = sub.add_parser("bench-m3", help="time default LightGBM on one fold per device")
    bench.add_argument("--target", choices=["lead", "window"], default="lead")
    bench.add_argument("--fold", type=int, default=5)
    bench.add_argument("--devices", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
    bench.add_argument("--threads", type=int, default=None)
    bench.add_argument("--out", help="also write the results to this JSON file")
    bench.set_defaults(func=cmd_bench_m3)
    imd = sub.add_parser("ingest-imd", help="download IMD 1-degree gridded Tmax (resumable)")
    imd.add_argument("--last-year", help="last year to download (default: last full year)")
    imd.set_defaults(func=cmd_ingest_imd)
    sub.add_parser(
        "hub-push-all", help="upload reference, raw, runs and labels to Hugging Face"
    ).set_defaults(func=cmd_hub_push_all)
    pull = sub.add_parser("hub-pull", help="download the data mirror from Hugging Face")
    pull.add_argument("--only", nargs="+", help="top-level folders, e.g. features labels models")
    pull.set_defaults(func=cmd_hub_pull)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
