"""Shared helpers for the handoff installer, verification, smoke-test and demo scripts. Standard library only.

These scripts run before (or without) the project dependencies, so this module never imports numpy, pandas, yaml or
falls_ml. Anything needing them runs in a subprocess of the virtual-environment interpreter.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import sysconfig
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

HANDOFF_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HANDOFF_DIR.parents[1]
if str(HANDOFF_DIR) not in sys.path:
    sys.path.insert(0, str(HANDOFF_DIR))

MARKER_FILE = ".falls_ml_generated"
SYNTHETIC_NOTICE_FILE = "SYNTHETIC_DATA_NOT_SCIENTIFIC_RESULTS.txt"
SYNTHETIC_BANNER = "SYNTHETIC DATA \u2013 NOT SCIENTIFIC RESULTS"
RULE = "=" * 40
DEMO_OUTPUTS = "demo_outputs"
SETUP_LOGS = "setup_logs"
REQUIREMENTS_LOCK = "requirements.lock"
BUILD_LOCK = "requirements-build.lock"
OFFLINE_DIR = "offline_packages"
OFFLINE_MANIFEST = "OFFLINE_MANIFEST.json"
#: offline-bundle target for THIS interpreter on Windows: windows-amd64-cp311 or windows-amd64-cp313 (the locked series)
WINDOWS_TARGET = "windows-amd64-cp%d%d" % sys.version_info[:2]
ENVIRONMENT_REPORT = "environment_report.txt"
PYTEST_BASETEMP = "pt"          # demo_outputs/pt: short, so deep test run folders stay below the Windows path limit
MPLCONFIG_DIR = "mplconfig"     # .venv/mplconfig: matplotlib config and font cache inside the project, not the user profile
#: pip settings that redirect an install away from the virtual environment (or make it fail). They are removed for every
#: child process; index, proxy and pip.ini/pip.conf settings are inherited unchanged.
PIP_TARGET_SETTINGS = ("PIP_USER", "PIP_TARGET", "PIP_PREFIX")
#: names (never values) of PIP_TARGET_SETTINGS that were set, recorded by the CMD/POSIX entry point that cleared them
CLEARED_PIP_SETTINGS_VAR = "FALLS_ML_CLEARED_PIP_SETTINGS"
LOG_TAIL_LINES = 30
WINDOWS_MAX_PATH = 260
# Deepest relative path created below the project folder, measured on the development Mac (2026-09-15; the Windows
# layout differs only in .venv\Lib\site-packages): installed packages .venv\Lib\site-packages\statsmodels\...\
# __init__.cpython-313.pyc 136 characters; full demo demo_outputs\full\runs_reproduced\<run id>\model\adapter\
# unpenalized_refit.csv 139; fast test suite demo_outputs\pt\<test>0\runs\<run id>\predictions_validation.parquet 146;
# full test suite (verify --full-tests) demo_outputs\pt\<test>0\moved\proj\runs_reproduced\<run id>\plots\
# calibration_before_after.png 154. Without long-path support the folder path must leave room for it:
# 260 - 1 (terminating NUL) - 155 - 1 (separator) = at most 103 characters.
DEEPEST_RELATIVE_PATH = 155
PREDICTION_TOLERANCE = 1e-9

_WINDOWS_COMMANDS = {"setup": "setup_windows.cmd", "verify": "verify_installation.cmd", "demo": "run_demo.cmd",
                     "full_demo": "run_full_demo.cmd", "offline": "prepare_offline_package.cmd", "clean": "clean_demo_outputs.cmd"}
_POSIX_COMMANDS = {"setup": "bash scripts/posix/setup.sh", "verify": "bash scripts/posix/verify.sh", "demo": "bash scripts/posix/run_demo.sh",
                   "full_demo": "bash scripts/posix/run_full_demo.sh", "offline": "bash scripts/posix/prepare_offline.sh",
                   "clean": "bash scripts/posix/clean_demo_outputs.sh"}


# ============================================================================ console, paths, environment
def configure_console() -> None:
    """Never crash on characters the console cannot show; flush every line (output is often piped)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_windows(platform: str | None = None) -> bool:
    return (platform or sys.platform) == "win32"


def script_command(name: str, platform: str | None = None) -> str:
    """User-facing command for an entry point on this platform (CMD file on Windows, POSIX mirror elsewhere)."""
    return (_WINDOWS_COMMANDS if is_windows(platform) else _POSIX_COMMANDS)[name]


def venv_dir(root: Path = PROJECT_ROOT) -> Path:
    return root / ".venv"


def venv_python(root: Path = PROJECT_ROOT, platform: str | None = None) -> Path:
    return venv_dir(root) / ("Scripts/python.exe" if is_windows(platform) else "bin/python")


def display_path(path: Path | str, root: Path = PROJECT_ROOT) -> str:
    """Path relative to the project folder when inside it (native separators), else absolute. For console messages only."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)


def posix_path(path: Path | str, root: Path = PROJECT_ROOT) -> str:
    """Path relative to ``root`` when inside it, else absolute, always with '/' separators. For paths stored in JSON, YAML and
    Markdown files, so they are identical on Windows and macOS/Linux (Windows accepts '/' as a separator)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def cleared_pip_settings(environ: dict[str, str] | None = None) -> list[str]:
    """Names of PIP_USER / PIP_TARGET / PIP_PREFIX that were set: still in ``environ`` or recorded by the entry point that cleared
    them (FALLS_ML_CLEARED_PIP_SETTINGS). Names only; values are never read."""
    environ = os.environ if environ is None else environ
    recorded = set(environ.get(CLEARED_PIP_SETTINGS_VAR, "").split())
    return [name for name in PIP_TARGET_SETTINGS if name in environ or name in recorded]


def matplotlib_config_dir(root: Path = PROJECT_ROOT) -> Path | None:
    """<project>/.venv/mplconfig, created when .venv exists (never creates .venv itself: an empty .venv would look like a broken
    virtual environment to the installer). None when there is no .venv or the folder cannot be created."""
    venv = venv_dir(root)
    if not venv.is_dir():
        return None
    target = venv / MPLCONFIG_DIR
    try:
        target.mkdir(exist_ok=True)
    except OSError:
        return None
    return target if target.is_dir() else None


def subprocess_env(*, no_bytecode: bool = False, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for child processes: UTF-8 I/O, no pip self-update check, no leaking PYTHONPATH/PYTHONHOME, no pip
    install-target overrides (PIP_USER/PIP_TARGET/PIP_PREFIX; their names are passed on in FALLS_ML_CLEARED_PIP_SETTINGS for
    environment_report.txt), matplotlib cache in .venv/mplconfig.

    Index and proxy settings (PIP_INDEX_URL, PIP_EXTRA_INDEX_URL, HTTPS_PROXY, pip.ini) are inherited unchanged.
    """
    env = dict(os.environ)
    cleared = cleared_pip_settings(env)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONINSPECT", CLEARED_PIP_SETTINGS_VAR, *PIP_TARGET_SETTINGS):
        env.pop(name, None)
    if cleared:
        env[CLEARED_PIP_SETTINGS_VAR] = " ".join(cleared)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1", MPLBACKEND="Agg")
    mplconfig = matplotlib_config_dir()
    if mplconfig is not None:
        env["MPLCONFIGDIR"] = str(mplconfig)
    if no_bytecode:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra or {})
    return env


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes} min {rest:02d} s"


def format_command(cmd: Sequence[str | Path]) -> str:
    parts = [str(c) for c in cmd]
    return subprocess.list2cmdline(parts) if is_windows() else " ".join(shlex.quote(p) for p in parts)


# ============================================================================ running commands
@dataclass
class CommandResult:
    returncode: int
    lines: list[str]
    seconds: float
    log_path: Path | None = None

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_logged(cmd: Sequence[str | Path], log_path: Path, *, cwd: Path | None = None, env: dict[str, str] | None = None,
               echo: Callable[[str], str | None] | None = None) -> CommandResult:
    """Run ``cmd`` (stdin closed, stderr merged), appending every output line to ``log_path``.

    ``echo`` maps an output line to the text shown on the console (None hides it).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    lines: list[str] = []
    with open(log_path, "a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n$ {format_command(cmd)}\n# cwd: {cwd or PROJECT_ROOT}\n# started: {utc_now()}\n")
        log.flush()
        try:
            proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd or PROJECT_ROOT), env=env if env is not None else subprocess_env(),
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", bufsize=1)
        except OSError as exc:
            message = f"could not start {cmd[0]}: {exc}"
            log.write(f"# {message}\n")
            return CommandResult(127, [message], time.monotonic() - start, log_path)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            lines.append(line)
            log.write(line + "\n")
            log.flush()
            if echo is not None:
                shown = echo(line)
                if shown is not None:
                    print(shown, flush=True)
        returncode = proc.wait()
        seconds = time.monotonic() - start
        log.write(f"# exit code: {returncode} after {seconds:.1f} s\n")
    return CommandResult(returncode, lines, seconds, log_path)


def tail(lines: Sequence[str], n: int = LOG_TAIL_LINES) -> list[str]:
    return [line for line in lines if line.strip()][-n:]


def reset_log(log_path: Path, title: str) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"# {title}\n# {utc_now()}\n", encoding="utf-8")
    return log_path


# ============================================================================ steps and result blocks
class StepFailure(Exception):
    """A step failed; carries the user-facing reason, the recommended action and the log file."""

    def __init__(self, reason: str, action: str, log_path: Path | None = None, details: Sequence[str] = ()):
        super().__init__(reason)
        self.reason, self.action, self.log_path, self.details = reason, action, log_path, list(details)
        self.step = ""


class StepSkipped(Exception):
    """A step was skipped on request; the message says why."""


@dataclass
class StepRecord:
    number: int
    title: str
    status: str
    seconds: float
    message: str


@dataclass
class StepRunner:
    """Prints "[n/T] Title..." then "[OK] ..." / "[SKIPPED] ..." / "[FAILED] ..." and a blank line; records timings."""

    total: int
    records: list[StepRecord] = field(default_factory=list)
    unexpected_action: str = "Run the command again; if the problem persists send the log folder to the maintainer."

    def label(self, number: int, title: str) -> str:
        return f"[{number}/{self.total}] {title}"

    def run(self, number: int, title: str, func: Callable[[], str]) -> str | None:
        print(f"{self.label(number, title)}...", flush=True)
        start = time.monotonic()
        try:
            message = func()
        except StepSkipped as skipped:
            self.records.append(StepRecord(number, title, "skipped", time.monotonic() - start, str(skipped)))
            print(f"[SKIPPED] {skipped}\n", flush=True)
            return None
        except StepFailure as failure:
            failure.step = self.label(number, title)
            self._fail(number, title, start, failure)
            raise
        except Exception as exc:  # noqa: BLE001 - converted into a reported failure, never swallowed
            failure = StepFailure(f"unexpected {type(exc).__name__}: {exc}", self.unexpected_action,
                                  details=traceback.format_exc().splitlines())
            failure.step = self.label(number, title)
            self._fail(number, title, start, failure)
            raise failure from exc
        self.records.append(StepRecord(number, title, "ok", time.monotonic() - start, message))
        print(f"[OK] {message}\n", flush=True)
        return message

    def _fail(self, number: int, title: str, start: float, failure: StepFailure) -> None:
        self.records.append(StepRecord(number, title, "failed", time.monotonic() - start, failure.reason))
        print(f"[FAILED] {failure.reason}", flush=True)
        if failure.details:
            source = f" of {display_path(failure.log_path)}" if failure.log_path else ""
            print(f"  Last {min(len(failure.details), LOG_TAIL_LINES)} lines{source}:")
            for line in failure.details[-LOG_TAIL_LINES:]:
                print(f"    {line}")
        print(flush=True)

    def timing_lines(self) -> list[str]:
        return [f"  [{r.number}/{self.total}] {r.title}: {r.status}, {format_seconds(r.seconds)}" for r in self.records]


def success_block(title: str) -> list[str]:
    return [RULE, title, RULE]


def failure_block(title: str, step: str, reason: str, action: str, log_path: Path | str | None = None) -> list[str]:
    lines = [RULE, title, RULE, f"Step: {step}", f"Reason: {reason}", f"Recommended action: {action}"]
    if log_path:
        lines.append(f"Log file: {Path(log_path).resolve()}")
    return lines


def print_lines(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)
    sys.stdout.flush()


def print_banner() -> None:
    print_lines(["*" * 40, SYNTHETIC_BANNER, "*" * 40])


# ============================================================================ generated output folders
class OutputDirError(RuntimeError):
    """Refusal to replace or delete a folder these scripts did not create."""


def is_generated(directory: Path) -> bool:
    """True for a real directory (not a link) holding the marker file as a regular file."""
    return directory.is_dir() and not directory.is_symlink() and (directory / MARKER_FILE).is_file() and not (directory / MARKER_FILE).is_symlink()


def mark_generated(directory: Path, purpose: str) -> Path:
    """Create ``directory`` if needed and write the marker file and the synthetic-data notice into it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MARKER_FILE).write_text(
        f"Generated by the falls_ml handoff scripts ({purpose}) at {utc_now()}.\n"
        f"Safe to delete with {script_command('clean')}; folders without this file are never deleted by it.\n", encoding="utf-8")
    (directory / SYNTHETIC_NOTICE_FILE).write_text(
        f"{SYNTHETIC_BANNER}\n\nEverything in this folder was produced from deterministic SYNTHETIC fixtures to test the software.\n"
        "No real patient data was used. No number here is scientific evidence, and nothing here may be reported as a\n"
        "result about fall risk, eFalls or Meuhedet.\n", encoding="utf-8")
    return directory


def _make_writable_and_retry(function: Callable, path: str, _exc: BaseException) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    function(path)


def remove_tree(directory: Path) -> None:
    """Delete a directory tree (read-only files included; links are removed, never followed)."""
    if directory.is_symlink():
        directory.unlink()
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(directory, onexc=_make_writable_and_retry)
    else:  # pragma: no cover - the scripts target CPython 3.11/3.13
        shutil.rmtree(directory, onerror=lambda f, p, e: _make_writable_and_retry(f, p, e[1]))


def prepare_generated_dir(directory: Path, purpose: str) -> Path:
    """Return an empty, marked output folder. An existing folder is replaced only if it carries the marker."""
    directory = Path(directory)
    if directory.is_symlink():
        raise OutputDirError(f"{directory} is a link; refusing to write generated outputs through it")
    if directory.exists():
        if not directory.is_dir():
            raise OutputDirError(f"{directory} exists and is not a folder; move it away first")
        if any(directory.iterdir()):
            if not is_generated(directory):
                raise OutputDirError(f"{directory} already exists and was not created by these scripts (no {MARKER_FILE} file). "
                                     "Nothing was changed; move or rename that folder, or choose another output folder.")
            remove_tree(directory)
    return mark_generated(directory, purpose)


def ensure_demo_root(root: Path = PROJECT_ROOT) -> Path:
    """demo_outputs/ itself: marked when created by us or when empty; an existing unmarked folder is left unmarked."""
    demo = root / DEMO_OUTPUTS
    if demo.is_symlink() or (demo.exists() and not demo.is_dir()):
        raise OutputDirError(f"{demo} is not a regular folder")
    if not demo.exists() or not any(demo.iterdir()):
        mark_generated(demo, "demo and verification outputs")
    return demo


# ============================================================================ pytest and pip output
_PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b")
_PYTEST_TIME = re.compile(r"\bin (\d+(?:\.\d+)?)s\b")


def parse_pytest_summary(text: str) -> dict[str, object] | None:
    """Counts from pytest's final summary line (e.g. '978 passed, 12 deselected in 61.2s'), or None if absent."""
    for raw in reversed(text.splitlines()):
        line = raw.strip().strip("=").strip()
        time_match = _PYTEST_TIME.search(line)
        if not time_match or not (_PYTEST_COUNT.search(line) or line.startswith("no tests ran")):
            continue
        counts: dict[str, object] = {k: 0 for k in ("passed", "failed", "errors", "skipped", "deselected", "xfailed", "xpassed", "warnings")}
        for number, kind in _PYTEST_COUNT.findall(line):
            key = {"error": "errors", "warning": "warnings"}.get(kind, kind)
            counts[key] = int(number)
        counts["seconds"] = float(time_match.group(1))
        counts["line"] = line
        return counts
    return None


def pytest_counts_text(summary: dict[str, object]) -> str:
    parts = [f"{summary[k]} {k}" for k in ("passed", "failed", "errors", "skipped", "deselected", "xfailed", "xpassed") if summary.get(k)]
    return ", ".join(parts) or "no tests ran"


_PIP_FAILURE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("proxy", ("ProxyError", "Cannot connect to proxy", "407 Proxy Authentication Required", "Tunnel connection failed")),
    ("ssl", ("SSLError", "CERTIFICATE_VERIFY_FAILED", "certificate verify failed", "SSL: WRONG_VERSION_NUMBER")),
    ("network", ("Failed to establish a new connection", "Temporary failure in name resolution", "getaddrinfo failed",
                 "Name or service not known", "nodename nor servname", "Network is unreachable", "ConnectTimeoutError",
                 "Read timed out", "No route to host", "Connection refused", "connection broken", "RemoteDisconnected")),
    ("user_install", ("Can not perform a '--user' install",)),
    ("long_path", ("[WinError 206]", "filename or extension is too long", "Long Path support")),
    ("disk", ("No space left on device", "[Errno 28]", "There is not enough space on the disk")),
    ("permission", ("[WinError 5]", "Access is denied", "[Errno 13]", "Permission denied", "[WinError 32]")),
    ("hash", ("THESE PACKAGES DO NOT MATCH THE HASHES", "Hashes are required in --require-hashes mode")),
    ("platform", ("is not a supported wheel on this platform",)),
    ("not_found", ("No matching distribution found", "Could not find a version that satisfies")),
)


def classify_pip_failure(output: str, mode: str, platform: str | None = None) -> tuple[str, str, str]:
    """Return (kind, reason, recommended action) for a failed pip command in install ``mode`` (online/offline)."""
    kind = next((k for k, needles in _PIP_FAILURE_PATTERNS if any(n in output for n in needles)), "unknown")
    setup, offline_cmd = script_command("setup", platform), script_command("offline", platform)
    offline_hint = (f"use offline mode: on a computer with internet run {offline_cmd}, copy the offline_packages folder into this "
                    f"package folder and run {setup} --offline")
    online = mode == "online"
    table = {
        "proxy": ("pip could not get through the network proxy",
                  f"Set HTTPS_PROXY (and HTTP_PROXY) for this window as your IT department specifies, or {offline_hint}."),
        "ssl": ("the TLS/SSL certificate of the package index could not be verified (corporate TLS inspection?)",
                f"Ask IT for the corporate root certificate and set PIP_CERT to it, or {offline_hint}."),
        "network": ("the package index could not be reached (no network, firewall or DNS failure)",
                    f"Check the connection and PIP_INDEX_URL, or {offline_hint}."),
        "user_install": ("pip is configured for --user installs (PIP_USER or pip.ini), which cannot target a virtual environment",
                         "Remove 'user = true' from pip.ini / pip.conf for this window, then run the setup again."),
        "long_path": ("a file path exceeded the Windows 260-character limit",
                      "Move the package to a short folder such as C:\\Projects\\falls_ml_handoff and run the setup again."),
        "disk": ("the disk is full", "Free at least 2 GB on this drive and run the setup again."),
        "permission": ("a file could not be written (access denied or file in use)",
                       "Close other programs using .venv (editors, terminals, antivirus scans), make sure the folder is writable, "
                       f"then run {setup} again."),
        "hash": (("a downloaded file does not match the SHA-256 hash in the lock file" if online
                  else "a wheel in offline_packages does not match the SHA-256 hash in the lock file"),
                 (f"The package index or mirror served a different file; check PIP_INDEX_URL, or {offline_hint}." if online
                  else f"Re-run {offline_cmd} --force on a computer with internet and replace the offline_packages folder.")),
        "platform": ("a wheel is not compatible with this Python/platform",
                     (f"Re-run {offline_cmd} --target {WINDOWS_TARGET} --force and replace offline_packages." if not online
                      else "Make sure a 64-bit (x86-64) CPython 3.11 or 3.13 is used; see the log for the incompatible wheel.")),
        "not_found": (("a pinned package version is not available from the configured package index" if online
                       else "offline_packages does not contain a required wheel"),
                      (f"Check PIP_INDEX_URL (the mirror must provide the pinned versions), or {offline_hint}." if online
                       else f"Re-run {offline_cmd} --force with this package's lock files and replace offline_packages.")),
        "unknown": ("pip failed (see the log lines above)", f"Read the log file; if the cause is unclear, {offline_hint}."
                    if online else f"Read the log file; re-run {offline_cmd} --force if offline_packages may be damaged."),
    }
    reason, action = table[kind]
    return kind, reason, action


def pip_echo(line: str) -> str | None:
    """Console filter for pip output: progress milestones and errors only (everything is in the log)."""
    stripped = line.strip()
    if stripped.startswith(("Collecting ", "Processing ", "Installing collected packages", "Successfully installed",
                            "Obtaining ", "ERROR", "WARNING")):
        return "  " + re.sub(r"\s+\(from -r .*\)$", "", stripped)
    return None


# ============================================================================ lock files and the offline bundle
def lock_paths(root: Path = PROJECT_ROOT) -> tuple[Path, Path]:
    return root / REQUIREMENTS_LOCK, root / BUILD_LOCK


def applicable_requirements(lock_files: Iterable[Path], sys_platform: str | None = None) -> list:
    """Requirements (lockfile.Requirement) from ``lock_files`` whose marker applies to ``sys_platform``."""
    import lockfile

    out = []
    for path in lock_files:
        out.extend(r for r in lockfile.parse_lock(path) if lockfile.applies(r, sys_platform or sys.platform))
    return out


@dataclass(frozen=True)
class OfflineProblem:
    kind: str  # missing_manifest | bad_manifest | lock_mismatch | python | platform | missing_file | hash | incomplete
    message: str


_ARCH_ALIASES = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64", "win32": "x86", "i686": "x86", "x86": "x86"}


def platform_family(platform_string: str) -> tuple[str, str]:
    """(os, arch) for a sysconfig platform ('win-amd64', 'macosx-15.0-arm64') or wheel tag ('win_amd64', 'macosx_11_0_arm64')."""
    s = platform_string.lower().replace("-", "_").replace(".", "_")
    if s == "win32":
        return "win", "x86"
    os_name = "win" if s.startswith("win") else "macosx" if s.startswith("macosx") else "linux" if "linux" in s else s.split("_")[0]
    for arch in ("universal2", "x86_64", "amd64", "aarch64", "arm64", "i686", "x86"):
        if s.endswith(arch):
            return os_name, _ARCH_ALIASES.get(arch, arch)
    return os_name, s.rsplit("_", 1)[-1]


def tags_match_host(tags: Iterable[str], host: tuple[str, str]) -> bool:
    for tag in tags:
        os_name, arch = platform_family(str(tag))
        if os_name == host[0] and (arch == host[1] or (arch == "universal2" and host[0] == "macosx")):
            return True
    return False


def validate_offline_bundle(offline_dir: Path, requirements_lock: Path, build_lock: Path, *, sys_platform: str | None = None,
                            host_platform: str | None = None, check_file_hashes: bool = True) -> list[OfflineProblem]:
    """Every reason the offline bundle cannot install the locked environment on this interpreter (empty list = usable)."""
    import lockfile

    sys_platform = sys_platform or sys.platform
    host = platform_family(host_platform or sysconfig.get_platform())
    manifest_path = Path(offline_dir) / OFFLINE_MANIFEST
    if not manifest_path.is_file():
        return [OfflineProblem("missing_manifest", f"{display_path(manifest_path)} not found")]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [OfflineProblem("bad_manifest", f"{OFFLINE_MANIFEST} cannot be read: {exc}")]
    if not isinstance(manifest, dict) or manifest.get("format") != 1 or not isinstance(manifest.get("files"), dict) \
            or not isinstance(manifest.get("target"), dict):
        return [OfflineProblem("bad_manifest", f"{OFFLINE_MANIFEST} has an unsupported format (expected format 1 with target and files)")]
    problems: list[OfflineProblem] = []
    for key, lock in (("requirements_lock_sha256", requirements_lock), ("build_lock_sha256", build_lock)):
        if not lock.is_file():
            problems.append(OfflineProblem("lock_mismatch", f"{lock.name} not found in the package"))
        elif manifest.get(key) != lockfile.sha256_file(lock):
            problems.append(OfflineProblem("lock_mismatch", f"the bundle was prepared for a different {lock.name} (sha256 mismatch)"))
    target = manifest["target"]
    running = "%d.%d" % sys.version_info[:2]
    if (target.get("python_version"), target.get("implementation"), target.get("abi")) != (running, "cp", f"cp{running.replace('.', '')}"):
        problems.append(OfflineProblem("python", f"the bundle targets Python {target.get('python_version')} "
                                                 f"{target.get('implementation')}/{target.get('abi')}, not the running CPython {running}"))
    name, tags = target.get("name"), target.get("platform_tags") or []
    if sys_platform == "win32":
        platform_ok = name == WINDOWS_TARGET or (name == "current" and tags_match_host(tags, host))
        wanted = WINDOWS_TARGET
    else:
        platform_ok = name == "current" and tags_match_host(tags, host)
        wanted = f"current ({host[0]} {host[1]})"
    if not platform_ok:
        problems.append(OfflineProblem("platform", f"the bundle target is {name} {list(tags)}; this computer needs {wanted}"))
    if any(p.kind == "lock_mismatch" for p in problems):
        return problems  # coverage against a different lock would only add noise
    by_name: dict[str, tuple[str, str]] = {}
    for filename, digest in sorted(manifest["files"].items()):
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".whl"):
            problems.append(OfflineProblem("bad_manifest", f"invalid file name in manifest: {filename!r}"))
            continue
        path = Path(offline_dir) / filename
        if not path.is_file():
            problems.append(OfflineProblem("missing_file", f"{filename} is listed in the manifest but missing"))
            continue
        if check_file_hashes and lockfile.sha256_file(path) != digest:
            problems.append(OfflineProblem("hash", f"{filename} is damaged or was replaced (sha256 differs from the manifest)"))
            continue
        try:
            wheel = lockfile.parse_wheel_filename(filename)
        except ValueError as exc:
            problems.append(OfflineProblem("bad_manifest", str(exc)))
            continue
        by_name[wheel.name] = (wheel.version, str(digest))
    try:
        requirements = applicable_requirements((build_lock, requirements_lock), sys_platform)
    except (OSError, ValueError) as exc:
        return problems + [OfflineProblem("lock_mismatch", f"lock file cannot be parsed: {exc}")]
    for req in requirements:
        found = by_name.get(req.name)
        if found is None or found[0].lower() != req.version.lower():
            problems.append(OfflineProblem("incomplete", f"no wheel for {req.name}=={req.version}"))
        elif found[1] not in req.hashes:
            problems.append(OfflineProblem("hash", f"the {req.name} wheel's sha256 is not listed in the lock file"))
    return problems


def offline_problem_action(problems: Sequence[OfflineProblem], platform: str | None = None) -> tuple[str, str]:
    """(reason, recommended action) summarising offline bundle problems for the failure block."""
    kinds = {p.kind for p in problems}
    offline_cmd, setup = script_command("offline", platform), script_command("setup", platform)
    first = problems[0].message if problems else "unknown problem"
    more = f" (+{len(problems) - 1} more problems, see the log)" if len(problems) > 1 else ""
    if kinds == {"missing_manifest"}:
        return (f"offline mode was requested but {first}",
                f"On a computer with internet run {offline_cmd}, copy the offline_packages folder into this package folder, "
                f"then run {setup} --offline (or run {setup} --online if this computer can reach the package index).")
    if "platform" in kinds or "python" in kinds:
        target = f" --target {WINDOWS_TARGET}" if is_windows(platform) else " --target current"
        return f"offline bundle is for another platform: {first}{more}", f"Re-run {offline_cmd}{target} --force and replace the offline_packages folder."
    return (f"offline bundle is not usable: {first}{more}",
            f"Re-run {offline_cmd} --force with the lock files of this package version and replace the offline_packages folder "
            "(copy it completely; do not mix files from different bundles).")


# ============================================================================ Windows path length
def windows_long_paths_enabled() -> bool | None:
    """LongPathsEnabled registry value on Windows (None when not Windows or unreadable)."""
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError:
        return None


def windows_project_path_limit() -> int:
    """Longest project folder path (characters) that leaves room for DEEPEST_RELATIVE_PATH below MAX_PATH (currently 103)."""
    return WINDOWS_MAX_PATH - 1 - DEEPEST_RELATIVE_PATH - 1


def path_length_problem(root: Path, *, platform: str | None = None, long_paths_enabled: bool | None = None) -> str | None:
    """Reason text when the project folder path is too long for Windows without long-path support, else None."""
    if not is_windows(platform) or long_paths_enabled:
        return None
    limit = windows_project_path_limit()
    length = len(str(root))
    if length <= limit:
        return None
    return (f"the project folder path is {length} characters long; without Windows long-path support it must be at most {limit} "
            "characters (installed packages and demo outputs create deep file paths)")
