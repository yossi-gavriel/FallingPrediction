"""Installer steps 4-10, called by setup_windows.cmd (and scripts/posix/setup.sh) after the venv exists. Stdlib only.

Usage: .venv\\Scripts\\python.exe scripts\\handoff\\setup_steps.py --first-step 4 --total-steps 10 [--mode auto|online|offline] [--skip-tests]

 4 build tools (pip, setuptools, wheel) from requirements-build.lock
 5 dependencies from requirements.lock (hash-checked, binary wheels only)
 6 falls_ml itself (editable install of this folder, no network)
 7 environment validation: installed versions == locks, pip check, falls_ml imports, environment_report.txt
 8 test suite (pytest -m "not slow", temporary files in demo_outputs/pt; skipped with --skip-tests, for diagnosis only)
 9 synthetic smoke test (demo_outputs/setup_smoke)
10 model save/load and prediction equality (from the smoke result)

Step 4 first checks the interpreter, the lock files and the project folder path length (at most 103 characters on Windows
without long-path support). Mode auto = offline when offline_packages/OFFLINE_MANIFEST.json exists, else online. Online
installs respect the pip configuration (PIP_INDEX_URL, PIP_EXTRA_INDEX_URL, HTTPS_PROXY, pip.ini); PIP_USER, PIP_TARGET and
PIP_PREFIX are cleared for every pip process. Logs: setup_logs/NN_<step>.log. Exit 0 = success, 1 = failure.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import verify  # noqa: E402

ROOT = common.PROJECT_ROOT
N_STEPS = 7
PIP_INSTALL = ["-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
LOCK_FLAGS = ["--require-hashes", "--only-binary=:all:"]


def resolve_mode(requested: str, root: Path = ROOT) -> tuple[str, str]:
    """(mode, explanation) for --mode auto|online|offline."""
    manifest = root / common.OFFLINE_DIR / common.OFFLINE_MANIFEST
    if requested == "online":
        return "online", "online (requested); package index from the pip configuration (PIP_INDEX_URL / pip.ini) or pypi.org"
    if requested == "offline":
        return "offline", f"offline (requested); wheels from {common.OFFLINE_DIR}{'' if manifest.is_file() else ' (manifest missing)'}"
    if manifest.is_file():
        return "offline", f"offline (auto: {common.OFFLINE_DIR}/{common.OFFLINE_MANIFEST} found)"
    note = f"; {common.OFFLINE_DIR} exists but has no {common.OFFLINE_MANIFEST}, so it is ignored" if (root / common.OFFLINE_DIR).exists() else ""
    return "online", f"online (auto: no offline bundle){note}"


def pip_source_flags(mode: str, root: Path = ROOT) -> list[str]:
    return ["--no-index", "--find-links", str(root / common.OFFLINE_DIR)] if mode == "offline" else []


def lock_install_command(python: Path, lock: Path, mode: str, root: Path = ROOT) -> list[str]:
    return [str(python), *PIP_INSTALL, *LOCK_FLAGS, *pip_source_flags(mode, root), "-r", str(lock)]


def project_install_command(python: Path) -> list[str]:
    return [str(python), *PIP_INSTALL, "--no-deps", "--no-build-isolation", "--no-index", "-e", "."]


class Installer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logs = ROOT / common.SETUP_LOGS
        self.python = common.venv_python(ROOT)
        self.mode, self.mode_text = resolve_mode(args.mode)
        self.runner = common.StepRunner(total=args.total_steps,
                                        unexpected_action=f"Run {common.script_command('setup')} again; if it persists send the "
                                                          f"{common.SETUP_LOGS} folder to the maintainer.")
        self.info: dict[str, str] = {}

    def log(self, number: int, name: str, title: str) -> Path:
        return common.reset_log(self.logs / f"{number:02d}_{name}.log", title)

    # ------------------------------------------------------------------ steps
    def preflight(self) -> None:
        if not self.python.is_file():
            raise common.StepFailure(f"{common.display_path(self.python)} not found", f"Run {common.script_command('setup')} (it creates .venv).")
        if Path(sys.prefix).resolve() != common.venv_dir(ROOT).resolve():
            raise common.StepFailure(f"setup_steps.py must run with the .venv interpreter, not {sys.executable}",
                                     f"Start the installation with {common.script_command('setup')}.")
        for lock in common.lock_paths(ROOT):
            if not lock.is_file():
                raise common.StepFailure(f"{lock.name} is missing from the package", "Copy the complete falls_ml_handoff folder again.")
        problem = common.path_length_problem(ROOT, long_paths_enabled=common.windows_long_paths_enabled())
        if problem:
            raise common.StepFailure(problem, "Move the folder to a shorter path such as C:\\Projects\\falls_ml_handoff and run "
                                              f"{common.script_command('setup')} --recreate-venv there (the virtual environment "
                                              "stores its folder path).")

    def check_offline_bundle(self, log_path: Path) -> None:
        requirements_lock, build_lock = common.lock_paths(ROOT)
        problems = common.validate_offline_bundle(ROOT / common.OFFLINE_DIR, requirements_lock, build_lock)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("offline bundle check: " + ("ok" if not problems else "; ".join(p.message for p in problems)) + "\n")
        if problems:
            reason, action = common.offline_problem_action(problems)
            raise common.StepFailure(reason, action, log_path, [p.message for p in problems])

    def pip(self, cmd: list[str], log_path: Path, what: str) -> None:
        result = common.run_logged(cmd, log_path, echo=common.pip_echo)
        if not result.ok:
            _, reason, action = common.classify_pip_failure(result.output, self.mode)
            raise common.StepFailure(f"{what} failed: {reason}", action, log_path, common.tail(result.lines))

    def step_build_tools(self, number: int) -> str:
        log_path = self.log(number, "build_tools", "build tools")
        self.preflight()
        print(f"  Install mode: {self.mode_text}", flush=True)
        if self.mode == "offline":
            self.check_offline_bundle(log_path)
        _, build_lock = common.lock_paths(ROOT)
        try:
            reqs = common.applicable_requirements([build_lock])
        except ValueError as exc:
            raise common.StepFailure(f"{build_lock.name} is damaged: {exc}", "Copy the complete falls_ml_handoff folder again.", log_path)
        self.pip(lock_install_command(self.python, build_lock, self.mode), log_path, "installing build tools")
        return f"{', '.join(f'{r.name} {r.version}' for r in reqs)} ({self.mode})"

    def step_dependencies(self, number: int) -> str:
        log_path = self.log(number, "dependencies", "dependencies")
        requirements_lock, _ = common.lock_paths(ROOT)
        try:
            reqs = common.applicable_requirements([requirements_lock])
        except ValueError as exc:
            raise common.StepFailure(f"{requirements_lock.name} is damaged: {exc}", "Copy the complete falls_ml_handoff folder again.", log_path)
        self.pip(lock_install_command(self.python, requirements_lock, self.mode), log_path, "installing dependencies")
        return f"{len(reqs)} locked packages installed from {requirements_lock.name} (hash-checked wheels, {self.mode})"

    def step_project(self, number: int) -> str:
        log_path = self.log(number, "project", "falls_ml editable install")
        if not (ROOT / "pyproject.toml").is_file() or not (ROOT / "src" / "falls_ml" / "__init__.py").is_file():
            raise common.StepFailure("pyproject.toml or src/falls_ml is missing", "Copy the complete falls_ml_handoff folder again.", log_path)
        self.pip(project_install_command(self.python), log_path, "installing falls_ml (editable)")
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        return f"falls_ml {version} installed in editable mode from {common.display_path(ROOT / 'src')}"

    def step_environment(self, number: int) -> str:
        log_path = self.log(number, "environment", "environment validation")
        parts = [verify.check_locked_versions(log_path), verify.check_pip(log_path), verify.check_imports(log_path)]
        report = ROOT / common.ENVIRONMENT_REPORT
        result = common.run_logged([self.python, Path(__file__).resolve().parent / "environment_report.py", "--mode", self.mode,
                                    "--out", report], log_path)
        if not result.ok or not report.is_file():
            raise common.StepFailure("environment_report.txt could not be written", "Make sure the project folder is writable.",
                                     log_path, common.tail(result.lines))
        self.info["environment_report"] = str(report)
        return "; ".join(parts + [f"{common.ENVIRONMENT_REPORT} written"])

    def step_tests(self, number: int) -> str:
        if self.args.skip_tests:
            self.info["tests"] = "SKIPPED (--skip-tests)"
            raise common.StepSkipped(f"TESTS SKIPPED (--skip-tests). Use this only for diagnosis; run {common.script_command('verify')} before real use.")
        message, summary = verify.run_tests(self.log(number, "tests", "test suite"), full=False)
        self.info["tests"] = common.pytest_counts_text(summary)
        return message

    def step_smoke(self, number: int) -> str:
        out = common.ensure_demo_root(ROOT) / "setup_smoke"
        message, _ = verify.run_smoke(out, self.log(number, "smoke", "synthetic smoke test"))
        self.info["smoke"] = f"passed ({common.display_path(out)})"
        return message

    def step_roundtrip(self, number: int) -> str:
        self.log(number, "model_roundtrip", "model save/load and prediction equality")
        message = verify.check_roundtrip(ROOT / common.DEMO_OUTPUTS / "setup_smoke" / "SMOKE_RESULT.json")
        self.info["roundtrip"] = "passed"
        return message

    # ------------------------------------------------------------------ run
    def run(self) -> int:
        steps = [("Installing build tools", self.step_build_tools), ("Installing dependencies", self.step_dependencies),
                 ("Installing falls_ml", self.step_project), ("Validating environment", self.step_environment),
                 ("Running test suite", self.step_tests), ("Running synthetic smoke test", self.step_smoke),
                 ("Checking model save/load and prediction equality", self.step_roundtrip)]
        if not (self.logs / common.MARKER_FILE).exists():
            common.mark_generated(self.logs, "setup logs")
        start = time.monotonic()
        try:
            for offset, (title, func) in enumerate(steps):
                number = self.args.first_step + offset
                self.runner.run(number, title, lambda f=func, n=number: f(n))
        except common.StepFailure as failure:
            self.write_summary(["INSTALLATION FAILED", failure.step, failure.reason])
            common.print_lines(common.failure_block("INSTALLATION FAILED", failure.step, failure.reason, failure.action,
                                                    failure.log_path or self.logs))
            return 1
        skipped = self.args.skip_tests
        lines = common.success_block("INSTALLATION SUCCESSFUL (tests skipped)" if skipped else "INSTALLATION SUCCESSFUL")
        lines += self.summary_lines(time.monotonic() - start)
        common.print_lines(lines)
        self.write_summary(lines)
        return 0

    def summary_lines(self, seconds: float) -> list[str]:
        windows = common.is_windows()
        activate = ".venv\\Scripts\\activate.bat" if windows else "source .venv/bin/activate"
        baseline = str(Path(common.DEMO_OUTPUTS) / "baseline")
        lines = [f"Install mode:        {self.mode_text}",
                 f"Python:              {sys.version.split()[0]} ({self.python})",
                 f"Virtual environment: {common.venv_dir(ROOT)}",
                 f"Environment report:  {self.info.get('environment_report', ROOT / common.ENVIRONMENT_REPORT)}",
                 f"Tests:               {self.info.get('tests', 'not run')}",
                 f"Smoke test:          {self.info.get('smoke', 'not run')}",
                 f"Model round trip:    {self.info.get('roundtrip', 'not run')}",
                 f"Setup logs:          {self.logs}",
                 f"Elapsed (steps {self.args.first_step}-{self.args.total_steps}): {common.format_seconds(seconds)}"]
        lines += self.runner.timing_lines()
        if self.args.skip_tests:
            lines.append(f"WARNING: TESTS SKIPPED. Run {common.script_command('verify')} before using this installation.")
        lines += ["", "Next commands:",
                  f"  {common.script_command('demo'):<36} baseline synthetic demo ({baseline})",
                  f"  {common.script_command('full_demo'):<36} all algorithms, reduced eFalls, ablation, monitoring",
                  f"  {common.script_command('verify'):<36} re-check the installation at any time",
                  f"  {activate:<36} use falls_ml in this window (python -m falls_ml --help)",
                  "", f"All demo outputs use synthetic data: {common.SYNTHETIC_BANNER}"]
        return lines

    def write_summary(self, lines: list[str]) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        (self.logs / "00_summary.log").write_text("\n".join([common.utc_now(), *lines]) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    common.configure_console()
    ap = argparse.ArgumentParser(description="falls_ml installer steps (called by setup_windows.cmd).")
    ap.add_argument("--first-step", type=int, required=True)
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--mode", choices=["auto", "online", "offline"], default="auto")
    ap.add_argument("--skip-tests", action="store_true", help="skip the test suite (diagnosis only)")
    args = ap.parse_args(argv)
    if args.first_step < 1 or args.total_steps != args.first_step + N_STEPS - 1:
        ap.error(f"--total-steps must equal --first-step + {N_STEPS - 1}")
    try:
        return Installer(args).run()
    except KeyboardInterrupt:
        common.print_lines(common.failure_block("INSTALLATION FAILED", "interrupted", "stopped by the user (Ctrl+C)",
                                                f"Run {common.script_command('setup')} again; completed steps are reused."))
        return 1


if __name__ == "__main__":
    sys.exit(main())
