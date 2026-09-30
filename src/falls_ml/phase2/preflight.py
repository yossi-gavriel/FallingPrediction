"""``falls_ml meuhedet-phase2 ... --preflight``: every check that can fail before the real run, with NO model fitting and NOTHING written to
``--out`` (the folder is not even created). Row-level work (reading the extract, rebuilding the reference cohort, reproducing the split)
happens in memory and in a temporary folder that is deleted at the end. The last line is exactly ``SAFE TO START FULL RUN`` or
``NOT SAFE TO START FULL RUN - <n> problem(s)``; the exit code is 0 or 2.
"""

from __future__ import annotations

import os
import shutil
import struct
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from falls_ml.phase2 import durable as D
from falls_ml.phase2.config import DEFAULT_CONFIG
from falls_ml.phase2.state import Phase2Stop

SAFE_LINE = "SAFE TO START FULL RUN"
MIN_FREE_GB, WARN_FREE_GB = 2.0, 5.0
KEY_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "pyarrow", "PyYAML", "threadpoolctl", "optuna", "SQLAlchemy",
                "alembic", "xgboost", "xgboost-cpu")


class Report:
    def __init__(self, printer: Callable[[str], None]):
        self.p, self.problems, self.warnings = printer, [], []

    def section(self, title: str) -> None:
        self.p("")
        self.p(f"== {title}")

    def ok(self, what: str, detail: str = "") -> None:
        self.p(f"  [OK]   {what}" + (f": {detail}" if detail else ""))

    def info(self, what: str, detail: str = "") -> None:
        self.p(f"         {what}" + (f": {detail}" if detail else ""))

    def warn(self, what: str, detail: str = "") -> None:
        self.warnings.append(what)
        self.p(f"  [WARN] {what}" + (f": {detail}" if detail else ""))

    def fail(self, what: str, detail: str = "", fix: str = "") -> None:
        self.problems.append(f"{what}" + (f" - FIX: {fix}" if fix else ""))
        self.p(f"  [FAIL] {what}" + (f": {detail}" if detail else ""))
        if fix:
            self.p(f"         FIX: {fix}")


def _lock_pins() -> dict[str, str]:
    from falls_ml.paths import resolve_path

    pins: dict[str, str] = {}
    p = resolve_path("requirements.lock")
    if not p.is_file():
        return pins
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")) or "==" not in line:
            continue
        spec = line.split(";")[0].split("\\")[0].strip()
        name, ver = spec.split("==", 1)
        marker = line.split(";", 1)[1] if ";" in line else ""
        if 'sys_platform == "win32"' in marker and sys.platform != "win32":
            continue
        if 'sys_platform != "win32"' in marker and sys.platform == "win32":
            continue
        pins[name.strip().lower()] = ver.strip()
    return pins


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def _environment(r: Report) -> None:
    from importlib.metadata import PackageNotFoundError, version

    import falls_ml

    r.section("1. Python, packages, dependencies")
    py = sys.version.split()[0]
    bits = struct.calcsize("P") * 8
    (r.ok if py.startswith("3.11.") else r.warn)("Python", f"{py} ({bits}-bit){'' if py.startswith('3.11.') else ' - the work PC runbook expects 3.11.x'}")
    if bits != 64:
        r.fail("64-bit Python required", f"{bits}-bit interpreter", "use the project's .venv (64-bit CPython 3.11)")
    r.ok("falls_ml", falls_ml.__version__)
    pins = _lock_pins()
    for pkg in KEY_PACKAGES:
        try:
            v = version(pkg)
        except PackageNotFoundError:
            if pkg in ("xgboost", "xgboost-cpu"):
                continue
            r.fail(f"package {pkg}", "not installed", "run the one-time pip install from requirements.lock (runbook step 2)")
            continue
        want = pins.get(_norm(pkg))
        if want and want != v:
            r.fail(f"package {pkg}", f"{v} installed, {want} pinned in requirements.lock",
                   "reinstall from requirements.lock: .venv\\Scripts\\python.exe -m pip install --require-hashes --only-binary=:all: -r requirements.lock")
        else:
            r.ok(f"package {pkg}", v + ("" if want else " (not pinned directly)"))
    try:
        from falls_ml.phase2.context import import_xgboost

        xgb = import_xgboost()
        r.ok("XGBoost import (OpenMP runtime)", xgb.__version__)
    except Phase2Stop as exc:
        r.fail("XGBoost import", exc.message, "; ".join(exc.details) or "install the Microsoft Visual C++ Redistributable (x64)")
    try:
        import importlib.util

        import optuna

        if importlib.util.find_spec("sqlite3") is None:   # Optuna's local file storage (no database server; never imported here)
            r.fail("SQLite runtime for the Optuna storage", "the Python sqlite3 module is missing", "reinstall the standard CPython 3.11 (64-bit)")
        else:
            r.ok("Optuna + local SQLite file storage", f"optuna {optuna.__version__}")
    except Exception as exc:  # noqa: BLE001 - report any import failure
        r.fail("Optuna import", f"{type(exc).__name__}: {exc}", "run the one-time pip install from requirements.lock")


def _nearest_existing(p: Path) -> Path:
    q = p.resolve()
    while not q.exists() and q != q.parent:
        q = q.parent
    return q


def _paths(r: Report, src: Path, ref: Path, protected: list[Path], out: Path, allow_synced: bool) -> bool:
    r.section("2. Paths (input, reference, protected folders, output)")
    good = True
    if src.is_file():
        try:
            with open(src, "rb") as fh:
                fh.read(1)
            r.ok("input readable", f"{src.name} ({src.stat().st_size / 2**20:.1f} MB)")
        except OSError as exc:
            good = False
            r.fail("input not readable", str(exc), "close the file in Excel / other programs and retry")
    else:
        good = False
        r.fail("input file not found", str(src), "check the --input path")
    for p in protected:
        role = "reference (read-only)" if p == ref else "protected (read-only)"
        if not p.is_dir():
            good = False
            r.fail(f"{role} folder not found", p.name, "check the --reference / --protect path")
            continue
        r.ok(f"{role} folder", p.name)
        if out.resolve() == p.resolve() or p.resolve() in out.resolve().parents:
            good = False
            r.fail("--out is inside a protected folder", p.name, "choose an output folder outside every earlier result folder")
    if out.exists():
        if (out / "PHASE2_PLAN.json").is_file():
            good = False
            r.fail("--out already holds a Phase 2 run", out.name, "a new run needs a NEW folder; to continue this run use the RESUME command (not the preflight)")
        elif any(out.iterdir()):
            good = False
            r.fail("--out is not empty", out.name, "choose a new, empty (or non-existing) output folder")
        else:
            r.ok("--out", f"{out.name} exists and is empty")
    else:
        r.ok("--out", f"{out.name} does not exist yet (the run creates it; the preflight does not)")
    synced = D.synced_folder(out.parent if not out.exists() else out)
    if synced and not allow_synced:
        good = False
        r.fail("--out is inside a synchronised folder", synced, "use a local folder such as C:\\Users\\<you>\\Downloads\\... outside OneDrive")
    else:
        r.ok("--out is not in OneDrive / a synced folder")
    base = _nearest_existing(out)
    probe = base / f".falls_ml_preflight_write_test_{os.getpid()}"
    try:
        with open(probe, "wb") as fh:
            fh.write(b"ok")
            fh.flush()
            os.fsync(fh.fileno())
        probe.unlink()
        r.ok("write permission", f"can create and delete files in {base.name or base}")
    except OSError as exc:
        good = False
        r.fail("no write permission", f"{base}: {exc}", "choose an output folder you can write to")
    free = shutil.disk_usage(base).free / 2**30
    if free < MIN_FREE_GB:
        good = False
        r.fail("free disk space", f"{free:.1f} GB", f"free at least {MIN_FREE_GB:.0f} GB on that drive (a full run writes ~0.1-0.5 GB)")
    elif free < WARN_FREE_GB:
        r.warn("free disk space", f"{free:.1f} GB (enough for the run; below {WARN_FREE_GB:.0f} GB)")
    else:
        r.ok("free disk space", f"{free:.1f} GB")
    return good


def run_preflight(input_path: str | Path, reference_dir: str | Path, *, out_dir: str | Path, protect: list[str | Path] | None = None,
                  id_pepper_file: str | Path | None = None, config_path: str | Path = DEFAULT_CONFIG, allow_synced_folder: bool = False,
                  allow_unfrozen_config: bool = False, printer: Callable[[str], None] = print) -> int:
    from falls_ml.artifacts import source_tree_sha256
    from falls_ml.d00.dependency import load_d00_config
    from falls_ml.data.dataset import sha256_file
    from falls_ml.data.meuhedet_wide import load_wide_contract, load_wide_mapping, read_wide_extract_report
    from falls_ml.eda.dictionary import load_data_dictionary
    from falls_ml.features.spec import load_feature_spec
    from falls_ml.phase2.accounting import funnel_rows
    from falls_ml.phase2.cohort import build_cohort
    from falls_ml.phase2.config import load_catalogue, load_phase2_config, resolve_threads
    from falls_ml.phase2.engineer import engineer, engineered_registry
    from falls_ml.phase2.featuresets import ALL_SAFE, EXPL_ALL, SAFE_BASE, build_feature_sets, check_category_invariants, design_spec
    from falls_ml.phase2.final_config import FROZEN_PATH
    from falls_ml.phase2.registry import Provenance, column_registry
    from falls_ml.phase2.runner import _plan, _plan_sha, effective_final_config

    t0 = time.perf_counter()
    r = Report(printer)
    src, ref, out = Path(input_path), Path(reference_dir), Path(out_dir)
    protected = [ref, *[Path(p) for p in (protect or [])]]
    printer("falls_ml meuhedet-phase2 PREFLIGHT - no model is fitted; nothing is written to --out; no scientific output is created")
    _environment(r)
    paths_ok = _paths(r, src, ref, protected, out, allow_synced_folder)

    r.section("3. Configuration (frozen FINAL_EXPERIMENT_CONFIG)")
    cfg = cat = contract = mapping = dictionary = d00 = None
    final_sha = frozen = None
    try:
        cfg = load_phase2_config(config_path)
        mapping = load_wide_mapping()
        contract = load_wide_contract(mapping.contract_path)
        dictionary = load_data_dictionary("configs/meuhedet/wide_v1_data_dictionary.yaml", contract)
        d00 = load_d00_config(cfg["d00_config"], contract=contract, dictionary=dictionary)
        cat = load_catalogue(cfg["features"], contract, dictionary, mapping)
        r.ok("analysis settings valid", f"{Path(cfg.path).name} v{cfg.get('version')} sha256 {cfg.sha256}")
        r.ok("feature catalogue valid", f"{Path(cat.path).name} v{cat.version}: {len(cat.features)} engineered features, {len(cat.domains)} domains, sha256 {cat.sha256}")
        _, final_sha, frozen = effective_final_config(cfg, cat, contract=contract, dictionary=dictionary, mapping=mapping, d00=d00,
                                                      allow_unfrozen=allow_unfrozen_config)
        if frozen:
            r.ok(f"effective configuration = frozen {FROZEN_PATH}", f"sha256 {final_sha}")
        else:
            r.warn("NOT the frozen production configuration (development / tests only)", f"sha256 {final_sha}")
    except Phase2Stop as exc:
        r.fail(exc.gate, exc.message, "; ".join(exc.details))
    except Exception as exc:  # noqa: BLE001 - every configuration problem is reported, none is fatal to the report
        r.fail("configuration invalid", f"{type(exc).__name__}: {exc}", "restore the patch files; never edit configs for the real run")
    code_sha = source_tree_sha256()
    r.ok("code", f"falls_ml source sha256 {code_sha}")

    input_sha = None
    if paths_ok and cfg is not None and cat is not None:
        r.section("4. Input (read-only)")
        t = time.perf_counter()
        input_sha = sha256_file(src)
        r.ok("input sha256", f"{input_sha} ({time.perf_counter() - t:.0f} s)")
        r.section("5. Reference cohort, split, TEST exclusion (in memory)")
        scratch = Path(tempfile.mkdtemp(prefix="falls_ml_preflight_"))
        try:
            t = time.perf_counter()
            read = read_wide_extract_report(src, contract)
            espec = load_feature_spec(mapping.exploratory_spec_path)
            res = build_cohort(src, ref, contract=contract, mapping=mapping, spec=espec, pepper_file=id_pepper_file, scratch=scratch, read=read,
                               input_sha=input_sha)
            f = res.facts
            r.ok("input = the reference run's input (sha256)")
            r.ok("reference pepper found and matches", "never created, never printed")
            r.ok("cohort rebuilt = reference D00_CLEAN cohort", f"{f['cohort_rows']:,} rows; predictor values identical")
            r.ok("split reproduced", f"TEST row-set sha256 equals the reference; splits.csv checked: {f['splits_csv_checked']}")
            r.ok("no patient in more than one partition")
            pt = f["partitions"]
            r.ok("TRAIN", f"{pt['train']['n_rows']:,} rows, {pt['train']['n_events']:,} events ({100 * pt['train']['n_events'] / max(1, pt['train']['n_rows']):.2f}%)")
            r.ok("VALIDATION (sealed until the frozen selection)", f"{pt['validation']['n_rows']:,} rows, {pt['validation']['n_events']:,} events")
            r.ok("TEST dropped in memory (count + hash only; outcomes never read)", f"{f['test_rows_dropped']:,} rows")
            la = f.get("label_death_audit") or {}
            if la.get("available"):
                msg = (f"deaths within the 180-day window: {la.get('deaths_within_window')}; with a recorded event: "
                       f"{la.get('events_among_deaths_within_window')}; non-events: {la.get('non_events_among_deaths_within_window')}")
                if la.get("overwrite_suspected"):
                    r.warn("outcome / death semantics (TRAIN+VALIDATION)", msg + " -> the run will stop at LABEL_DEATH_SEMANTICS for your decision")
                else:
                    r.ok("outcome / death semantics (TRAIN+VALIDATION)", msg)
            r.info("cohort / split time", f"{time.perf_counter() - t:.0f} s")

            r.section("6. Provenance and feature plan (exact counts; D-00 rules on TRAIN + VALIDATION record dates)")
            work = res.frame
            prov = Provenance(work, f["index_date"], contract=contract, dictionary=dictionary, d00_config=d00)
            reg = column_registry(res.profile, prov, contract=contract, dictionary=dictionary, mapping=mapping, catalogue=cat,
                                  min_observed_train=int(cfg["eligibility"]["min_observed_train_rows"]))
            values = engineer(work, cat)
            train_mask = (work["partition"] == "train").to_numpy()
            er = engineered_registry(values, train_mask, prov, catalogue=cat, contract=contract, dictionary=dictionary,
                                     min_observed_train=int(cfg["eligibility"]["min_observed_train_rows"]))
            baseline = list(f["baseline_features"])
            bstatus = {m.canonical: prov.feature(tuple(m.source_columns))["status"] for m in mapping.features if m.canonical in baseline}
            from falls_ml.phase2.stages_data import redundancy_universes

            import pandas as pd

            tr = pd.concat([work, values], axis=1).loc[train_mask].reset_index(drop=True)
            order = {x.name: x.order for x in cat.features}
            _, reps, _ = redundancy_universes(tr, er, baseline, bstatus, threshold=float(cfg["screening"]["redundancy_abs_spearman"]), order=order)
            elig = er.loc[er["eligibility"].isin(["ELIGIBLE", "ELIGIBLE_EXPLORATORY"]), "feature"].tolist()
            ds = design_spec(values, train_mask, cat, elig, unanswered_min_share=float(cfg["screening"]["item_unanswered_min_share"]))
            fs = build_feature_sets(cat, er, reps, baseline_features=baseline, baseline_status=bstatus, design=ds)
            for row in funnel_rows(reg, er, fs, bstatus):
                r.info(f"{row['step']}", f"{row['n']}" + (f"  [{row['category']}]" if row["category"] else "") + (f"  {row['note']}" if row["note"] and row["category"] else ""))
            bad = check_category_invariants(fs, er, bstatus, baseline)
            if bad:
                r.fail("category invariant (SAFE_DISCOVERY = SAFE only; exploratory never UNSAFE)", "; ".join(bad[:5]), "report this output; do not start")
            else:
                r.ok("category invariant", "every SAFE_DISCOVERY set holds SAFE features only; UNRESOLVED only in the exploratory sensitivity; UNSAFE nowhere")
            n_safe_new = int(fs.sets[ALL_SAFE]["n_new_features"])
            if n_safe_new == 0:
                r.warn("no SAFE new feature in this extract", "the SAFE discovery can only refit SAFE_BASE; the run is valid but will add little")
            r.info("SAFE_BASE", ", ".join(fs.sets[SAFE_BASE]["baseline"]) or "(none)")
            for w in fs.design["waterfall"]:
                r.info(f"waterfall step {w['step']}: + {w['added']}", f"{w['n_safe_features_added']} SAFE feature(s) added")
            r.info("EXPLORATORY_ALL_REVIEWED", f"{fs.sets[EXPL_ALL]['n_new_features']} new features offered (LASSO only)")
            n_fit = len([s for s in fs.fitted() if fs.sets[s]["kind"] != "loo"])
            r.info("distinct LASSO feature sets to fit", str(n_fit))
        except Phase2Stop as exc:
            r.fail(exc.gate, exc.message, "; ".join(exc.details) or "see the message")
        except Exception as exc:  # noqa: BLE001
            r.fail("cohort / provenance check", f"{type(exc).__name__}: {str(exc)[:500]}", "send this preflight output (it has no patient data)")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    if cfg is not None and cat is not None:
        c = cfg.raw
        x = c["xgb"]
        K = int(c["cv"]["outer_folds"])
        r.section("7. Model families and budgets (frozen)")
        r.info("model families", "LASSO (penalised logistic), elastic net (l1_ratio " + ", ".join(str(a) for a in c["enet"]["l1_ratios"]) + "), XGBoost")
        r.info("CV", f"outer {K} folds (stratified, patient = row) inside TRAIN; inner {c['cv']['inner_folds_linear']} (linear) / {c['cv']['inner_folds_xgb']} (XGBoost)")
        r.info("XGBoost stage 0", f"{len(x['stage0']['sets'])} sets x {K + 1} scopes (defaults)")
        r.info("XGBoost stage 1 (Optuna TPE)", f"{x['stage1']['n_trials']} trials per scope x {K + 1} scopes = {int(x['stage1']['n_trials']) * (K + 1)} trials")
        if x["stage2"].get("enabled", True):
            r.info("XGBoost stage 2 (only if stage 1 >= best SAFE linear + "
                   f"{x['stage2']['condition_min_ap_gain_vs_best_linear']} OOF AP)", f"<= {x['stage2']['n_trials']} trials per scope (<= {int(x['stage2']['n_trials']) * (K + 1)})")
        else:
            r.info("XGBoost stage 2", "disabled in the frozen configuration (Astra F-06): total Optuna budget = the stage-1 trials above")
        r.info("selection", f"SAFE candidates vs SAFE_BASE; principal capacity top {float(c['selection']['principal_capacity']):.0%}; benchmark class vs BASELINE_15")
        r.info("VALIDATION", "opened once after SELECTION_FROZEN.json (VALIDATION_OPENED.json); TEST never loaded")

    r.section("8. Runtime paths of the real run")
    r.info("run state", str(out / "RUN_STATE.json"))
    r.info("stage completion markers", str(out / "STAGE_<nn>_COMPLETE.json") + "  (00 ... 16)")
    r.info("frozen selection", str(out / "SELECTION_FROZEN.json"))
    r.info("validation opened (write-once)", str(out / "VALIDATION_OPENED.json"))
    r.info("Optuna storage (rebuilt from committed trials each attempt)", str(out / "checkpoints" / "xgb_optuna.sqlite"))
    r.info("Optuna trial ledger (append-only)", str(out / "checkpoints" / "xgb_trials.jsonl"))
    r.info("event log (append-only)", str(out / "logs" / "events.jsonl"))
    r.info("timings", str(out / "RUN_TIMINGS.csv"))
    r.info("gate files", str(out / "INVESTIGATION_<GATE>.md") + "  /  " + str(out / "STOPPED.md"))
    r.info("the ONLY folder to send", str(out / "share"))

    r.section("9. Hashes")
    if cfg is not None and cat is not None and input_sha:
        threads = resolve_threads(cfg)
        plan = _plan(cfg, cat, contract=contract, dictionary=dictionary, mapping=mapping, d00=d00, input_sha=input_sha, input_name=src.name, ref_dir=ref,
                     threads=threads, protected=protected, final_sha=final_sha or "", frozen=bool(frozen))
        r.info("config (phase2.yaml) sha256", cfg.sha256)
        r.info("FINAL_EXPERIMENT_CONFIG sha256", str(final_sha))
        r.info("catalogue sha256", cat.sha256)
        r.info("input sha256", input_sha)
        r.info("code sha256", code_sha)
        r.info("threads (frozen at start)", f"BLAS {threads['blas']}, XGBoost {threads['xgboost']}")
        r.info("plan sha256 the run will freeze", _plan_sha(plan))
    r.p("")
    r.p(f"preflight time: {time.perf_counter() - t0:.0f} s; warnings: {len(r.warnings)}")
    if r.problems:
        r.p(f"NOT SAFE TO START FULL RUN - {len(r.problems)} problem(s):")
        for i, pr in enumerate(r.problems, 1):
            r.p(f"  {i}. {pr}")
        return 2
    r.p(SAFE_LINE)
    return 0
