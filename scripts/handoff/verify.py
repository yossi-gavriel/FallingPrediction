"""Verify an installed falls_ml handoff package (9 checks). Standard library only at import time.

Usage: <venv python> scripts/handoff/verify.py [--skip-tests | --full-tests]

 1 Python environment: supported interpreter, installed versions == lock files, pip check
 2 import every falls_ml module (editable install of this folder, version == VERSION)
 3 every feature spec in configs/features loads
 4 every experiment config in configs/experiments loads (ablation configs parsed by their own schema)
 5 every fixture in data/fixtures validates against the feature spec its manifest names
 6 the command-line interface starts and lists its commands
 7 core tests (pytest -m "not slow"; the full suite with --full-tests; skipped with --skip-tests)
 8 synthetic end-to-end scoring (smoke.py)
 9 model bundle save/load and prediction equality (from the smoke result)

Results go to demo_outputs/verify/, pytest temporary files to demo_outputs/pt/ (short, for the Windows path limit), the
matplotlib cache to .venv/mplconfig/. Exit 0 = VERIFICATION SUCCESSFUL, 1 = FAILED.
Checks that need the dependencies run in a subprocess of the venv interpreter (``verify.py --worker <name>``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

WORKER_PREFIX = "WORKER_RESULT "
WORKERS = ("versions", "imports", "feature-specs", "experiment-configs", "fixtures")
ABLATION_KEYS = {"base_experiment_config", "extension_specs", "steps", "name"}
CLI_COMMANDS = ("make-fixture", "validate-dataset", "build-dataset", "train", "evaluate", "reproduce", "compare", "ablation", "predict", "monitor",
                "meuhedet-make-fixture", "meuhedet-audit", "meuhedet-build", "meuhedet-explore", "freeze-baseline")
ROOT = common.PROJECT_ROOT


# ============================================================================ workers (run inside the venv interpreter)
def worker_versions() -> dict[str, Any]:
    import importlib.metadata

    lock_files = common.lock_paths(ROOT)
    missing_locks = [p.name for p in lock_files if not p.is_file()]
    if missing_locks:
        return {"ok": False, "problems": [f"lock file missing: {n}" for n in missing_locks], "checked": 0}
    problems, checked = [], 0
    for req in common.applicable_requirements(lock_files, sys.platform):
        checked += 1
        try:
            installed = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{req.name}=={req.version} is not installed")
            continue
        if installed.lower() != req.version.lower():
            problems.append(f"{req.name} {installed} is installed but the lock pins {req.version}")
    return {"ok": not problems, "problems": problems, "checked": checked, "python": sys.version.split()[0], "executable": sys.executable}


def worker_imports() -> dict[str, Any]:
    import importlib
    import pkgutil
    import traceback

    import falls_ml

    problems = []
    package_dir = Path(falls_ml.__file__).resolve().parent
    expected_dir = (ROOT / "src" / "falls_ml").resolve()
    if package_dir != expected_dir:
        problems.append(f"falls_ml is imported from {package_dir}, not from this folder ({expected_dir}); re-run the setup")
    version_file = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if falls_ml.__version__ != version_file:
        problems.append(f"falls_ml.__version__ {falls_ml.__version__} differs from VERSION {version_file}")
    names = ["falls_ml"] + sorted(m.name for m in pkgutil.walk_packages(falls_ml.__path__, "falls_ml.") if m.name != "falls_ml.__main__")
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            problems.append(f"import {name} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return {"ok": not problems, "problems": problems, "n_modules": len(names), "version": falls_ml.__version__,
            "location": common.display_path(package_dir, ROOT)}


def _is_base_spec(raw: Any) -> bool:
    return isinstance(raw, dict) and "identifiers" in raw


def worker_feature_specs() -> dict[str, Any]:
    import yaml

    from falls_ml.features.spec import load_feature_spec

    problems, loaded = [], []
    paths = sorted((ROOT / "configs" / "features").glob("*.yaml"))
    raws = {p: yaml.safe_load(p.read_text(encoding="utf-8")) for p in paths}
    bases = [p for p in paths if _is_base_spec(raws[p])]
    if not bases:
        problems.append("no base feature spec found in configs/features")
    for path in paths:
        try:
            if path in bases:
                spec = load_feature_spec(path)
                loaded.append(f"{spec.name}: {len(spec.features)} predictors")
            else:
                spec = load_feature_spec(bases[0], [path])
                base = load_feature_spec(bases[0])
                loaded.append(f"{path.stem}: +{len(spec.features) - len(base.features)} extension predictors")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{common.display_path(path, ROOT)}: {type(exc).__name__}: {exc}")
    return {"ok": not problems and bool(paths), "problems": problems, "loaded": loaded, "n_specs": len(paths)}


def _check_ablation(path: Path, raw: dict[str, Any]) -> str:
    from falls_ml.config import load_experiment_config
    from falls_ml.errors import ConfigError
    from falls_ml.features.spec import load_feature_spec
    from falls_ml.paths import resolve_path

    unknown = sorted(set(raw) - ABLATION_KEYS)
    if unknown:
        raise ConfigError(f"unknown ablation keys {unknown}")
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps or not all(isinstance(s, dict) and s.get("name") and isinstance(s.get("groups", []), list) for s in steps):
        raise ConfigError("ablation 'steps' must be a non-empty list of {name, groups}")
    base = load_experiment_config(resolve_path(raw["base_experiment_config"], anchor=path))
    extensions = [str(resolve_path(e, anchor=path)) for e in raw.get("extension_specs", [])]
    spec = load_feature_spec(base.dataset.feature_spec, extensions, anchor=base.source_path)
    unknown_groups = sorted({g for s in steps for g in s.get("groups", [])} - set(spec.groups()))
    if unknown_groups:
        raise ConfigError(f"ablation steps use unknown feature groups {unknown_groups}")
    return f"ablation ({len(steps)} steps on {base.experiment.name})"


def worker_experiment_configs() -> dict[str, Any]:
    import yaml

    import falls_ml.config as config_module
    from falls_ml.features.spec import load_feature_spec

    problems, kinds = [], {}
    paths = sorted((ROOT / "configs" / "experiments").rglob("*.yaml"))
    for path in paths:
        rel = common.display_path(path, ROOT)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "base_experiment_config" in raw:
                kind = _check_ablation(path, raw)
            else:
                cfg = config_module.load_experiment_config(path)
                check = getattr(config_module, "check_feature_declaration", None)
                if check is not None:
                    check(cfg, load_feature_spec(cfg.dataset.feature_spec, cfg.dataset.feature_spec_extensions, anchor=cfg.source_path))
                kind = cfg.experiment.kind
            kinds[rel] = kind
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{rel}: {type(exc).__name__}: {exc}")
    n_ablation = sum(1 for k in kinds.values() if k.startswith("ablation ("))
    return {"ok": not problems and bool(paths), "problems": problems, "n_configs": len(paths), "n_ablation": n_ablation, "kinds": kinds}


def worker_fixtures() -> dict[str, Any]:
    from falls_ml.data.dataset import ModelingDataset
    from falls_ml.features.spec import load_feature_spec

    problems, validated = [], []
    fixtures = sorted(p.parent for p in (ROOT / "data" / "fixtures").glob("*/manifest.json"))
    if not fixtures:
        problems.append("no fixtures found under data/fixtures")
    for directory in fixtures:
        rel = common.display_path(directory, ROOT)
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            parts = str(manifest["feature_spec_name"]).split("+")
            spec_paths = [ROOT / "configs" / "features" / f"{p}.yaml" for p in parts]
            absent = [common.display_path(p, ROOT) for p in spec_paths if not p.is_file()]
            if absent:
                raise FileNotFoundError(f"manifest names feature spec {manifest['feature_spec_name']!r} but {absent} do not exist")
            spec = load_feature_spec(spec_paths[0], spec_paths[1:])
            dataset = ModelingDataset.load(directory, spec, infer_subset=True)
            if manifest.get("scientific_use_allowed"):
                raise ValueError("a fixture must not be marked scientific_use_allowed")
            n_predictors = len(dataset.spec.features)
            validated.append(f"{directory.name}: {len(dataset)} rows, {n_predictors} predictors")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{rel}: {type(exc).__name__}: {exc}")
    return {"ok": not problems, "problems": problems, "validated": validated}


def run_worker(name: str) -> int:
    functions: dict[str, Callable[[], dict[str, Any]]] = {
        "versions": worker_versions, "imports": worker_imports, "feature-specs": worker_feature_specs,
        "experiment-configs": worker_experiment_configs, "fixtures": worker_fixtures}
    try:
        result = functions[name]()
    except Exception as exc:  # noqa: BLE001 - reported to the parent as a failed result, traceback in the log
        import traceback

        traceback.print_exc()
        result = {"ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}
    print(WORKER_PREFIX + json.dumps(result), flush=True)
    return 0 if result.get("ok") else 1


# ============================================================================ checks shared with setup_steps.py
def call_worker(name: str, log_path: Path) -> dict[str, Any]:
    """Run a worker in the venv interpreter; return its result dict (raises StepFailure when it cannot run)."""
    python = common.venv_python(ROOT)
    result = common.run_logged([python, Path(__file__).resolve(), "--worker", name], log_path, env=common.subprocess_env(no_bytecode=True))
    for line in reversed(result.lines):
        if line.startswith(WORKER_PREFIX):
            return json.loads(line[len(WORKER_PREFIX):])
    raise common.StepFailure(f"the '{name}' check could not run (exit code {result.returncode})",
                             f"Run {common.script_command('setup')} again; if it persists send {common.display_path(log_path)} to the maintainer.",
                             log_path, common.tail(result.lines))


def _problems_failure(prefix: str, result: dict[str, Any], action: str, log_path: Path) -> common.StepFailure:
    problems = result.get("problems") or ["unknown problem"]
    more = f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""
    return common.StepFailure(f"{prefix}: {problems[0]}{more}", action, log_path, problems)


def check_python_interpreter(log_path: Path) -> str:
    python = common.venv_python(ROOT)
    if not python.is_file():
        raise common.StepFailure(f"{common.display_path(python)} not found", f"Run {common.script_command('setup')} first.", None)
    result = common.run_logged([python, Path(__file__).resolve().parent / "check_python.py"], log_path)
    line = next((ln for ln in result.lines if ln.startswith("PYTHON|")), "")
    parts = line.split("|")
    if result.returncode != 0 or len(parts) < 6:
        reason = next((ln[len("REASON|"):] for ln in result.lines if ln.startswith("REASON|")), "interpreter check failed")
        raise common.StepFailure(f"the virtual environment uses an unsupported interpreter: {reason}",
                                 f"Install Python 3.11 or 3.13 (64-bit) and run {common.script_command('setup')} --recreate-venv.", log_path,
                                 common.tail(result.lines))
    return f"Python {parts[2]} {parts[3]}-bit {parts[4]}"


def check_locked_versions(log_path: Path) -> str:
    result = call_worker("versions", log_path)
    if not result.get("ok"):
        raise _problems_failure("installed packages differ from the lock files", result,
                                f"Run {common.script_command('setup')} again (it installs the locked versions); if that does not help, "
                                f"run {common.script_command('setup')} --recreate-venv.", log_path)
    return f"{result['checked']} locked distributions match requirements.lock and requirements-build.lock"


def check_pip(log_path: Path) -> str:
    result = common.run_logged([common.venv_python(ROOT), "-m", "pip", "check"], log_path)
    if not result.ok:
        raise common.StepFailure("pip check reports broken or conflicting requirements",
                                 f"Run {common.script_command('setup')} --recreate-venv (packages were changed after installation).",
                                 log_path, common.tail(result.lines))
    return "pip check clean"


def check_imports(log_path: Path) -> str:
    result = call_worker("imports", log_path)
    if not result.get("ok"):
        raise _problems_failure("falls_ml cannot be imported cleanly", result,
                                f"Run {common.script_command('setup')} again from this folder (the editable install must point here).", log_path)
    return f"{result['n_modules']} falls_ml modules import (version {result['version']}, from {result['location']})"


def run_tests(log_path: Path, *, full: bool) -> tuple[str, dict[str, Any]]:
    """Run pytest (fast suite or full); return (message, parsed summary)."""
    basetemp = common.ensure_demo_root(ROOT) / common.PYTEST_BASETEMP
    if basetemp.exists() and any(basetemp.iterdir()) and not common.is_generated(basetemp):
        raise common.StepFailure(f"{common.display_path(basetemp)} exists and was not created by these scripts",
                                 "Move that folder away (pytest would delete it) and run again.", None)
    cmd: list[Any] = [common.venv_python(ROOT), "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", basetemp]
    if not full:
        cmd[3:3] = ["-m", "not slow"]
    try:
        result = common.run_logged(cmd, log_path, env=common.subprocess_env(no_bytecode=True),
                                   echo=lambda line: f"  {line.strip()}" if line.rstrip().endswith("%]") else None)
    finally:
        # pytest empties --basetemp (marker included) when it starts; re-mark it even after Ctrl+C so the next run and
        # clean_demo_outputs recognise the folder as generated
        if basetemp.is_dir() and not basetemp.is_symlink():
            common.mark_generated(basetemp, "pytest temporary files")
    summary = common.parse_pytest_summary(result.output)
    failed_lines = [ln for ln in result.lines if ln.startswith(("FAILED ", "ERROR "))]
    if summary is None:
        raise common.StepFailure(f"pytest did not finish (exit code {result.returncode})",
                                 f"Read {common.display_path(log_path)}; send it to the maintainer if the cause is unclear.",
                                 log_path, common.tail(result.lines))
    if result.returncode != 0 or summary["failed"] or summary["errors"] or not summary["passed"]:
        raise common.StepFailure(f"test suite failed: {common.pytest_counts_text(summary)} (exit code {result.returncode})",
                                 f"Do not use this installation for work. Send {common.display_path(log_path)} and "
                                 f"{common.ENVIRONMENT_REPORT} to the maintainer.", log_path, failed_lines or common.tail(result.lines))
    return f"{common.pytest_counts_text(summary)} in {common.format_seconds(float(summary['seconds']))}", summary


def run_smoke(out: Path, log_path: Path) -> tuple[str, dict[str, Any]]:
    """Run smoke.py into ``out``; return (message, SMOKE_RESULT.json contents)."""
    script = Path(__file__).resolve().parent / "smoke.py"
    result = common.run_logged([common.venv_python(ROOT), script, "--out", out], log_path, env=common.subprocess_env(no_bytecode=True),
                               echo=lambda line: line if line.startswith("  ") else None)
    result_file = out / "SMOKE_RESULT.json"
    smoke = json.loads(result_file.read_text(encoding="utf-8")) if result_file.is_file() else {}
    if result.returncode != 0 or not smoke.get("ok"):
        reason = smoke.get("reason") or f"smoke.py exited with code {result.returncode}"
        raise common.StepFailure(f"synthetic smoke test failed: {reason}",
                                 f"Read {common.display_path(log_path)}; run {common.script_command('verify')} for a full diagnosis.",
                                 log_path, common.tail(result.lines))
    names = ", ".join(r["config"] for r in smoke["runs"])
    return f"{len(smoke['runs'])} synthetic experiments completed ({names}) in {common.format_seconds(smoke['elapsed_seconds'])}", smoke


def check_roundtrip(result_file: Path) -> str:
    """Report bundle save/load and prediction equality from a SMOKE_RESULT.json."""
    if not result_file.is_file():
        raise common.StepFailure(f"{common.display_path(result_file)} not found (the smoke test did not run)",
                                 "Fix the smoke-test failure reported above first.", None)
    smoke = json.loads(result_file.read_text(encoding="utf-8"))
    checks = smoke.get("checks") or {}
    needed = ("artifacts", "bundle_reload_equal", "resave_reload_equal", "cli_predict_equal")
    failed = [c for c in needed if not checks.get(c)]
    if failed or not smoke.get("runs"):
        raise common.StepFailure(f"model round-trip checks failed: {failed or 'no runs recorded'}",
                                 f"Send {common.display_path(result_file)} to the maintainer.", result_file)
    worst = max(max(r["bundle_reload_max_abs_diff"], r["resave_reload_max_abs_diff"], r["cli_predict_max_abs_diff"]) for r in smoke["runs"])
    return (f"bundle reload, re-save + reload and CLI predict reproduce the test predictions of {len(smoke['runs'])} models "
            f"(max abs diff {worst:.1e} <= {common.PREDICTION_TOLERANCE:.0e})")


# ============================================================================ verification run
def verify(args: argparse.Namespace) -> int:
    out = common.ensure_demo_root(ROOT) / "verify"
    common.prepare_generated_dir(out, "installation verification")
    logs = common.mark_generated(out / "logs", "verification logs")
    runner = common.StepRunner(total=9)
    failures: list[common.StepFailure] = []
    state: dict[str, Any] = {}

    def attempt(number: int, title: str, func: Callable[[], str], depends_on: tuple[int, ...] = ()) -> None:
        blocked = [n for n in depends_on if n in state.get("failed", set())]
        if blocked:
            def skipped() -> str:
                raise common.StepSkipped(f"not run because check {blocked[0]} failed")
            func = skipped
        try:
            runner.run(number, title, func)
        except common.StepFailure as failure:
            failures.append(failure)
            state.setdefault("failed", set()).add(number)

    def log(n: int, name: str) -> Path:
        return logs / f"{n:02d}_{name}.log"

    def python_env() -> str:
        lg = common.reset_log(log(1, "python_environment"), "Python environment")
        return "; ".join([check_python_interpreter(lg), check_locked_versions(lg), check_pip(lg)])

    def features() -> str:
        lg = common.reset_log(log(3, "feature_specs"), "feature specs")
        result = call_worker("feature-specs", lg)
        if not result.get("ok"):
            raise _problems_failure("feature spec does not load", result, "Restore configs/features from the original package.", lg)
        return f"{result['n_specs']} feature specs load ({'; '.join(result['loaded'])})"

    def configs() -> str:
        lg = common.reset_log(log(4, "experiment_configs"), "experiment configs")
        result = call_worker("experiment-configs", lg)
        if not result.get("ok"):
            raise _problems_failure("experiment config does not load", result, "Restore configs/experiments from the original package.", lg)
        return f"{result['n_configs']} experiment configs load ({result['n_ablation']} ablation)"

    def fixtures() -> str:
        lg = common.reset_log(log(5, "fixtures"), "fixtures")
        result = call_worker("fixtures", lg)
        if not result.get("ok"):
            raise _problems_failure("synthetic fixture invalid", result, "Restore data/fixtures from the original package.", lg)
        return f"{len(result['validated'])} synthetic fixtures valid ({'; '.join(result['validated'])})"

    def cli() -> str:
        lg = common.reset_log(log(6, "cli"), "command-line interface")
        result = common.run_logged([common.venv_python(ROOT), "-m", "falls_ml", "--help"], lg, env=common.subprocess_env(no_bytecode=True))
        missing = [c for c in CLI_COMMANDS if c not in result.output]
        if not result.ok or missing:
            raise common.StepFailure(f"`python -m falls_ml --help` failed (exit code {result.returncode}; missing commands {missing})",
                                     f"Run {common.script_command('setup')} again.", lg, common.tail(result.lines))
        return f"python -m falls_ml starts; {len(CLI_COMMANDS)} commands available"

    def tests() -> str:
        if args.skip_tests:
            raise common.StepSkipped("tests skipped (--skip-tests); use only for diagnosis")
        message, state["tests"] = run_tests(common.reset_log(log(7, "tests"), "tests"), full=args.full_tests)
        return message

    def smoke() -> str:
        message, state["smoke"] = run_smoke(out / "smoke", common.reset_log(log(8, "smoke"), "smoke test"))
        return message

    attempt(1, "Checking Python environment", python_env)
    attempt(2, "Importing falls_ml modules", lambda: check_imports(common.reset_log(log(2, "imports"), "imports")))
    attempt(3, "Loading feature specifications", features, (2,))
    attempt(4, "Loading experiment configurations", configs, (2,))
    attempt(5, "Validating synthetic fixtures", fixtures, (2,))
    attempt(6, "Starting the command-line interface", cli, (2,))
    attempt(7, "Running core tests" if not args.full_tests else "Running full test suite", tests, (2,))
    attempt(8, "Running synthetic end-to-end scoring", smoke, (2,))
    attempt(9, "Checking model save/load and prediction equality", lambda: check_roundtrip(out / "smoke" / "SMOKE_RESULT.json"), (2, 8))

    if failures:
        first = failures[0]
        common.print_lines(common.failure_block("VERIFICATION FAILED", first.step, first.reason, first.action, first.log_path))
        if len(failures) > 1:
            print("Other failed checks: " + "; ".join(f.step for f in failures[1:]))
        print(f"Verification logs: {logs.resolve()}")
        return 1
    skipped = any(r.status == "skipped" for r in runner.records)
    common.print_lines(common.success_block("VERIFICATION SUCCESSFUL (tests skipped)" if skipped else "VERIFICATION SUCCESSFUL"))
    if skipped:
        print("WARNING: the test suite was skipped; run without --skip-tests before relying on this installation.")
    print(f"Python:         {common.venv_python(ROOT)}")
    if "tests" in state:
        print(f"Tests:          {common.pytest_counts_text(state['tests'])}")
    print(f"Smoke outputs:  {(out / 'smoke').resolve()}  ({common.SYNTHETIC_BANNER})")
    print(f"Logs:           {logs.resolve()}")
    print("Step times:")
    common.print_lines(runner.timing_lines())
    return 0


def main(argv: list[str] | None = None) -> int:
    common.configure_console()
    ap = argparse.ArgumentParser(description="Verify the falls_ml installation (9 checks).")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--skip-tests", action="store_true", help="skip the test suite (diagnosis only)")
    group.add_argument("--full-tests", action="store_true", help="run the full test suite including slow tests")
    ap.add_argument("--worker", choices=WORKERS, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        return run_worker(args.worker)
    try:
        return verify(args)
    except (common.OutputDirError, OSError) as exc:
        common.print_lines(common.failure_block("VERIFICATION FAILED", "Preparing demo_outputs/verify", str(exc),
                                                "Make sure the project folder is writable and demo_outputs/verify was created by these scripts."))
        return 1
    except KeyboardInterrupt:
        common.print_lines(common.failure_block("VERIFICATION FAILED", "interrupted", "stopped by the user (Ctrl+C)",
                                                f"Run {common.script_command('verify')} again."))
        return 1


if __name__ == "__main__":
    sys.exit(main())
