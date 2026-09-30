"""Windows handoff package: static checks of the CMD entry points and unit tests of the stdlib installer helpers.

The CMD files cannot run on the macOS development machine, so their structure is checked statically (CRLF, ASCII,
goto-based flow, labels, exit codes, pushd/popd, quoting). The Python helpers are tested directly.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

from tests.helpers.subprocesses import utf8_env

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "scripts" / "handoff"
RUNNING_SERIES = "%d.%d" % sys.version_info[:2]  # offline manifests are validated against the running interpreter
RUNNING_ABI = "cp%d%d" % sys.version_info[:2]
POSIX = ROOT / "scripts" / "posix"
sys.path.insert(0, str(HANDOFF))

import check_python  # noqa: E402
import clean_outputs  # noqa: E402
import common  # noqa: E402
import setup_steps  # noqa: E402
import smoke  # noqa: E402

CMD_FILES = ("setup_windows.cmd", "verify_installation.cmd", "run_demo.cmd", "run_full_demo.cmd", "prepare_offline_package.cmd",
             "clean_demo_outputs.cmd")
OWN_HELPERS = ("common.py", "check_python.py", "setup_steps.py", "verify.py", "smoke.py", "demo.py", "clean_outputs.py", "environment_report.py")
UNIX_COMMANDS = ("rm", "cp", "grep", "chmod", "source", "export", "ls", "cat", "sed", "awk", "mv", "touch", "which")
RULE = "=" * 40


# ============================================================================ CMD static checks
def _cmd_bytes(name: str) -> bytes:
    return (ROOT / name).read_bytes()


def _cmd_lines(name: str) -> list[str]:
    return _cmd_bytes(name).decode("ascii").split("\r\n")


def _strip_quoted(line: str) -> str:
    return re.sub(r'"[^"]*"', '""', line)


def _code_lines(name: str) -> list[tuple[int, str]]:
    """(line number, stripped text) of lines that are neither blank, REM comments nor labels."""
    out = []
    for number, raw in enumerate(_cmd_lines(name), start=1):
        line = raw.strip()
        if not line or line.lower().startswith(("rem ", "rem\t")) or line.lower() == "rem" or line.startswith(":"):
            continue
        out.append((number, line))
    return out


def _labels(name: str) -> dict[str, int]:
    return {raw.strip()[1:].split()[0].lower(): n for n, raw in enumerate(_cmd_lines(name), start=1)
            if raw.strip().startswith(":") and not raw.strip().startswith("::") and len(raw.strip()) > 1}


@pytest.mark.parametrize("name", CMD_FILES)
def test_cmd_file_is_crlf_ascii_with_usage_header(name):
    data = _cmd_bytes(name)
    assert data, name
    assert b"\n" not in data.replace(b"\r\n", b""), f"{name}: bare LF line endings"
    assert b"\r" not in data.replace(b"\r\n", b""), f"{name}: bare CR"
    assert all(b < 128 for b in data), f"{name}: non-ASCII bytes"
    assert data.endswith(b"\r\n")
    lines = _cmd_lines(name)
    assert lines[0] == "@echo off"
    assert any(line.startswith("REM Usage:") for line in lines[:12]), f"{name}: missing REM usage header"
    assert any(line.startswith("setlocal EnableExtensions") for line in lines), f"{name}: missing setlocal"


@pytest.mark.parametrize("name", CMD_FILES)
def test_cmd_file_avoids_forbidden_constructs(name):
    text = _cmd_bytes(name).decode("ascii")
    lowered = text.lower()
    assert "enabledelayedexpansion" not in lowered
    assert "powershell" not in lowered and "pwsh" not in lowered
    assert "`" not in text, "no backquote command parsing"
    assert "!" not in "".join(line for _, line in _code_lines(name) if not line.lower().startswith("echo")), "no delayed-expansion syntax"
    for number, line in _code_lines(name):
        code = _strip_quoted(line)
        if code.lower().startswith("echo"):
            continue
        for command in UNIX_COMMANDS:
            assert not re.search(rf"(^|[\s&|(]){command}(\s|$)", code), f"{name}:{number}: Unix command {command!r}: {line}"
        # goto-based control flow: no parenthesised blocks
        assert not re.search(r"\(\s*$", code), f"{name}:{number}: parenthesised block: {line}"
        assert not code.startswith(")"), f"{name}:{number}: parenthesised block: {line}"


@pytest.mark.parametrize("name", CMD_FILES)
def test_cmd_labels_exist_and_exits_have_codes(name):
    labels = _labels(name)
    for number, line in _code_lines(name):
        for target in re.findall(r"\bgoto\s+:?([A-Za-z0-9_]+)", line, flags=re.IGNORECASE):
            assert target.lower() == "eof" or target.lower() in labels, f"{name}:{number}: goto to unknown label {target}"
        for target in re.findall(r"\bcall\s+:([A-Za-z0-9_]+)", line, flags=re.IGNORECASE):
            assert target.lower() in labels, f"{name}:{number}: call to unknown label {target}"
        for match in re.finditer(r"\bexit\s+/b\b(.*)", line, flags=re.IGNORECASE):
            assert re.match(r"\s+(\d+|%[A-Za-z_]+%)\s*$", match.group(1)), f"{name}:{number}: exit /b without explicit code: {line}"
        assert not re.search(r"\bexit\s*$", line, flags=re.IGNORECASE), f"{name}:{number}: bare exit closes the user's window"


@pytest.mark.parametrize("name", CMD_FILES)
def test_cmd_pushd_popd_balanced_on_every_exit_path(name):
    lines = _cmd_lines(name)
    code = _code_lines(name)
    pushd = [n for n, line in code if line.lower().startswith("pushd")]
    popd = [n for n, line in code if line.lower() == "popd"]
    assert len(pushd) == 1 and 'pushd "%~dp0"' in lines[pushd[0] - 1], f"{name}: exactly one pushd \"%~dp0\" expected"
    assert "|| goto pushd_failed" in lines[pushd[0] - 1]
    assert len(popd) == 1, f"{name}: exactly one popd (in :finish) expected"
    labels = _labels(name)
    assert labels["finish"] < popd[0], f"{name}: popd must be inside :finish"
    exits = [n for n, line in code if re.search(r"\bexit\s+/b\b", line, flags=re.IGNORECASE)]
    for n in exits:
        section = max((ln, label) for label, ln in labels.items() if ln < n)[1] if any(ln < n for ln in labels.values()) else None
        if section == "finish":
            assert popd[0] < n, f"{name}:{n}: exit before popd"
        else:
            assert section == "pushd_failed", f"{name}:{n}: exit /b outside :finish skips popd"
    # every non-subroutine section ends by jumping somewhere (never falls into the failure handlers by accident)
    assert lines[labels["finish"] - 2].strip() == "" or lines[labels["finish"] - 2].strip().lower().startswith("goto")


@pytest.mark.parametrize("name", CMD_FILES)
def test_cmd_paths_are_quoted_and_helpers_exist(name):
    text = _cmd_bytes(name).decode("ascii")
    for helper in set(re.findall(r"scripts\\handoff\\([A-Za-z0-9_]+\.py)", text)):
        assert (HANDOFF / helper).is_file(), f"{name}: references missing scripts\\handoff\\{helper}"
    path_var = re.compile(r"%(?:[A-Z_]+_(?:PY|DIR|LOG|OUT|EXE|LINE|REASON)|CD|~dp0)%?")
    for number, line in _code_lines(name):
        if line.lower().startswith("echo"):
            continue
        unquoted = _strip_quoted(line)
        assert not path_var.search(unquoted), f"{name}:{number}: unquoted path variable: {line}"
        assert ".py" not in unquoted and ".venv\\" not in unquoted, f"{name}:{number}: unquoted path: {line}"
    for setting in ('set "PYTHONUTF8=1"', 'set "PYTHONIOENCODING=utf-8"', 'set "PIP_DISABLE_PIP_VERSION_CHECK=1"', 'set "PYTHONHOME="'):
        assert setting in text, f"{name}: missing {setting}"
    assert "%cmdcmdline" in text and "if defined PAUSE_AT_END pause" in text


CMD_PIP_BLOCK = ('if defined PIP_USER set "FALLS_ML_CLEARED_PIP_SETTINGS=%FALLS_ML_CLEARED_PIP_SETTINGS% PIP_USER"',
                 'if defined PIP_TARGET set "FALLS_ML_CLEARED_PIP_SETTINGS=%FALLS_ML_CLEARED_PIP_SETTINGS% PIP_TARGET"',
                 'if defined PIP_PREFIX set "FALLS_ML_CLEARED_PIP_SETTINGS=%FALLS_ML_CLEARED_PIP_SETTINGS% PIP_PREFIX"',
                 'set "PIP_USER="', 'set "PIP_TARGET="', 'set "PIP_PREFIX="')


@pytest.mark.parametrize("name", CMD_FILES)
def test_cmd_no_pause_opt_out_pip_settings_and_matplotlib_cache(name):
    lines = [line.strip() for line in _cmd_lines(name)]
    header_end = [i for i, line in enumerate(lines) if line.startswith("REM =====")][1]
    assert any("FALLS_ML_NO_PAUSE=1" in line for line in lines[:header_end]), f"{name}: opt-out not documented in the REM header"
    detect = next(i for i, line in enumerate(lines) if line.startswith('if /i not "%CMDLINE_CHECK:'))
    pushd = next(i for i, line in enumerate(lines) if line.startswith('pushd "%~dp0"'))
    assert lines[detect + 1] == 'if "%FALLS_ML_NO_PAUSE%"=="1" set "PAUSE_AT_END="' and detect + 1 < pushd, "opt-out must precede every pause"
    start = lines.index(CMD_PIP_BLOCK[0])
    assert tuple(lines[start:start + len(CMD_PIP_BLOCK)]) == CMD_PIP_BLOCK and start > pushd
    first_python = min(i for i, line in enumerate(lines) if re.match(r'^("%VENV_PY%"|call %PY|call :probe_python)', line))
    assert start < first_python, f"{name}: pip settings must be cleared before Python starts"
    mpl = [i for i, line in enumerate(lines) if "MPLCONFIGDIR=" in line and not line.startswith("REM")]
    assert len(mpl) == 1 and pushd < mpl[0], f"{name}: MPLCONFIGDIR is set once, after pushd"
    assert re.fullmatch(r'if exist "(\.venv|%VENV_DIR%)\\mplconfig\\" set "MPLCONFIGDIR=%CD%\\(\.venv|%VENV_DIR%)\\mplconfig"', lines[mpl[0]])
    mkdirs = [line for line in lines if line.lower().startswith(("mkdir", "if")) and "mkdir" in line.lower() and "mplconfig" in line]
    assert len(mkdirs) == 1, f"{name}: exactly one guarded mkdir of the matplotlib cache"
    if name == "setup_windows.cmd":
        assert lines.index(":step3") < mpl[0] < next(i for i, line in enumerate(lines) if line.startswith('"%VENV_PY%" "%STEPS_PY%"')), \
            "setup sets MPLCONFIGDIR only after .venv was created or validated"
    else:
        assert mpl[0] < first_python, f"{name}: MPLCONFIGDIR must be set before Python starts"
        assert mkdirs[0].startswith('if exist ".venv\\" if not exist ".venv\\mplconfig\\" mkdir ".venv\\mplconfig"'), \
            f"{name}: never create .venv itself (an empty .venv looks like a broken virtual environment)"


@pytest.mark.parametrize("name", ("setup.sh", "verify.sh", "run_demo.sh", "run_full_demo.sh", "prepare_offline.sh", "clean_demo_outputs.sh"))
def test_posix_mirrors_clear_pip_settings_and_set_matplotlib_cache(name):
    text = (POSIX / name).read_text(encoding="utf-8")
    assert "for pip_setting in PIP_USER PIP_TARGET PIP_PREFIX; do" in text and 'unset "$pip_setting"' in text
    assert "export FALLS_ML_CLEARED_PIP_SETTINGS" in text
    assert re.search(r'export MPLCONFIGDIR="\$ROOT/(\.venv|\$VENV_DIR)/mplconfig"', text)
    assert not re.search(r"mkdir -p[^\n]*mplconfig", text), "never create .venv itself (mkdir -p would)"
    assert re.search(r'(\[ -d \.venv \] && \[ ! -d \.venv/mplconfig \]|\[ -d "\$VENV_DIR/mplconfig" \] \|\|) *(; then )?mkdir', text)


def test_posix_pip_clearing_records_names_only(tmp_path):
    if sys.platform == "win32":
        pytest.skip("POSIX mirrors are not used on Windows (bash there may be WSL or Git bash)")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    source = (POSIX / "run_demo.sh").read_text(encoding="utf-8")
    block = source[source.index("for pip_setting in"):source.index("export FALLS_ML_CLEARED_PIP_SETTINGS; fi") + len("export FALLS_ML_CLEARED_PIP_SETTINGS; fi")]
    script = tmp_path / "block.sh"
    script.write_text("set -u\n" + block + "\n" + f'"{sys.executable}" -c "import os, json; print(json.dumps([os.environ.get(k) for k in '
                      "('FALLS_ML_CLEARED_PIP_SETTINGS', 'PIP_USER', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_INDEX_URL')]))\"\n", encoding="utf-8")
    env = {k: v for k, v in utf8_env().items() if k not in common.PIP_TARGET_SETTINGS + (common.CLEARED_PIP_SETTINGS_VAR,)}
    env.update(PIP_TARGET="/elsewhere/target", PIP_USER="1", PIP_INDEX_URL="https://mirror.example/simple")
    result = subprocess.run([bash, str(script)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)
    assert result.returncode == 0, result.stderr
    cleared, user, target, prefix, index = json.loads(result.stdout)
    assert cleared.split() == ["PIP_USER", "PIP_TARGET"] and (user, target, prefix) == (None, None, None)
    assert index == "https://mirror.example/simple", "index settings must reach pip"


def test_cmd_entry_points_reference_venv_and_helpers():
    assert '"%VENV_PY%" "%STEPS_PY%" --first-step 4 --total-steps 10 %MODE_ARG% %SKIP_TESTS_ARG%' in _cmd_bytes("setup_windows.cmd").decode()
    for name, helper in (("verify_installation.cmd", "verify.py"), ("run_demo.cmd", "demo.py"), ("run_full_demo.cmd", "demo.py"),
                         ("clean_demo_outputs.cmd", "clean_outputs.py")):
        text = _cmd_bytes(name).decode()
        assert 'set "VENV_PY=.venv\\Scripts\\python.exe"' in text and f"scripts\\handoff\\{helper}" in text
        assert 'if not exist "%VENV_PY%" goto no_venv' in text
        assert "setup_windows.cmd first" in text
    assert 'call :probe_python py -3.13' in _cmd_bytes("prepare_offline_package.cmd").decode()


def test_cmd_setup_detects_interpreters_in_order_and_prints_blocks():
    text = _cmd_bytes("setup_windows.cmd").decode()
    order = [text.index(f"call :probe_python {c}\r\n") for c in ("py -3.13", "python", "python3")]
    assert order == sorted(order)
    for needle in ('"%PROBE_RC%"=="3"', '"%PROBE_RC%"=="0"', "Microsoft Store", "No suitable Python runtime", "<nul",
                   "--write-exe", "rmdir /s /q \"%VENV_DIR%\"", 'call "%VENV_DIR%\\Scripts\\activate.bat"', "--recreate-venv",
                   "[1/10] Checking Python...", "[2/10] Creating virtual environment...", "[3/10] Activating virtual environment...",
                   "echo [OK] Python %PY_VER% (%%~M)"):
        assert needle in text, needle
    block = [f"echo {RULE}", "echo INSTALLATION FAILED", f"echo {RULE}", "echo Step: %%~M", "echo Reason: %%~M", "echo Recommended action: %%~M"]
    lines = [line.strip() for line in text.split("\r\n")]
    start = lines.index("echo INSTALLATION FAILED") - 1
    assert lines[start:start + 3] == block[:3]
    assert [ln.split(" do ")[-1] for ln in lines[start + 3:start + 6]] == block[3:]
    verify_text = _cmd_bytes("verify_installation.cmd").decode()
    assert f"echo {RULE}\r\necho VERIFICATION FAILED\r\necho {RULE}\r\n" in verify_text


def test_python_helpers_contain_verbatim_result_blocks():
    assert common.success_block("INSTALLATION SUCCESSFUL") == [RULE, "INSTALLATION SUCCESSFUL", RULE]
    steps_text = (HANDOFF / "setup_steps.py").read_text(encoding="utf-8")
    verify_text = (HANDOFF / "verify.py").read_text(encoding="utf-8")
    assert '"INSTALLATION SUCCESSFUL"' in steps_text and '"INSTALLATION FAILED"' in steps_text
    assert '"VERIFICATION SUCCESSFUL"' in verify_text and '"VERIFICATION FAILED"' in verify_text
    assert common.SYNTHETIC_BANNER == "SYNTHETIC DATA \u2013 NOT SCIENTIFIC RESULTS"


# ============================================================================ POSIX mirrors
@pytest.mark.parametrize("name", ("setup.sh", "verify.sh", "run_demo.sh", "run_full_demo.sh", "prepare_offline.sh", "clean_demo_outputs.sh"))
def test_posix_mirror_syntax(name):
    path = POSIX / name
    data = path.read_bytes()
    assert data.startswith(b"#!/usr/bin/env bash\n") and b"\r" not in data
    if sys.platform == "win32":
        pytest.skip("POSIX mirrors are not used on Windows (bash there may be WSL or Git bash)")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr


# ============================================================================ helper import boundary
@pytest.mark.parametrize("name", OWN_HELPERS)
def test_helpers_import_only_the_standard_library(name):
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import importlib; importlib.import_module(sys.argv[2]); "
            "bad = sorted({m.split('.')[0] for m in sys.modules} & {'numpy', 'pandas', 'yaml', 'falls_ml', 'scipy', 'sklearn', 'pyarrow', "
            "'matplotlib', 'pytest'}); print(bad); sys.exit(1 if bad else 0)")
    result = subprocess.run([sys.executable, "-c", code, str(HANDOFF), name[:-3]], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=utf8_env(), cwd=str(HANDOFF.parent))
    assert result.returncode == 0, result.stdout + result.stderr


# ============================================================================ check_python.py
def _facts(**overrides):
    facts = {"implementation": "CPython", "version_info": (3, 13, 13), "version": "3.13.13", "bits": 64, "machine": "AMD64",
             "platform": "win32", "executable": r"C:\Python313\python.exe",
             "sysconfig_platform": "win-amd64", "native_machine": "AMD64"}
    facts.update(overrides)
    if "version_info" in overrides:
        facts["version"] = "%d.%d.%d" % overrides["version_info"]
    return facts


@pytest.mark.parametrize("overrides, supported, reason_part", [
    ({}, True, None),
    ({"native_machine": None}, True, None),
    ({"version_info": (3, 12, 4)}, False, "Python 3.12.4 found"),
    ({"version_info": (3, 14, 0)}, False, "3.13.x"),
    ({"version_info": (2, 7, 18)}, False, "2.7.18"),
    ({"bits": 32, "sysconfig_platform": "win32", "machine": "AMD64"}, False, "32-bit"),
    ({"native_machine": "ARM64"}, False, "ARM64"),
    ({"sysconfig_platform": "win-arm64", "machine": "ARM64", "native_machine": "ARM64"}, False, "ARM64"),
    ({"implementation": "PyPy"}, False, "PyPy"),
    ({"platform": "darwin", "machine": "arm64", "sysconfig_platform": "macosx-15.0-arm64", "native_machine": None}, True, None),
    ({"platform": "darwin", "machine": "x86_64", "sysconfig_platform": "macosx-10.13-x86_64", "native_machine": None}, False, "macOS"),
    ({"platform": "linux", "machine": "x86_64", "sysconfig_platform": "linux-x86_64", "native_machine": None}, False, "linux"),
])
def test_check_python_evaluation(overrides, supported, reason_part):
    ok, reason, machine = check_python.evaluate(_facts(**overrides))
    assert ok is supported
    if reason_part:
        assert reason_part in reason
    lines = check_python.format_lines(_facts(**overrides), ok, reason, machine)
    parsed = check_python.parse_line(lines[0])
    assert parsed["status"] == ("ok" if supported else "unsupported")
    assert parsed["executable"] == _facts(**overrides)["executable"]
    assert (len(lines) == 2 and lines[1].startswith("REASON|")) is (not supported)


def test_check_python_parse_line_keeps_paths_with_spaces_and_parentheses():
    line = r"PYTHON|ok|3.13.13|64|AMD64|C:\Projects\Falls Research (test)\falls_ml_handoff\.venv\Scripts\python.exe"
    assert check_python.parse_line(line) == {"status": "ok", "version": "3.13.13", "bits": "64", "machine": "AMD64",
                                             "executable": r"C:\Projects\Falls Research (test)\falls_ml_handoff\.venv\Scripts\python.exe"}
    assert check_python.parse_line("Python was not found; run without arguments to install from the Microsoft Store") is None
    assert check_python.parse_line("PYTHON|maybe|3.13|64|x|y") is None


def test_check_python_script_output_and_write_exe(tmp_path):
    exe_file = tmp_path / "dir with spaces (x)" / "exe.txt"
    exe_file.parent.mkdir()
    result = subprocess.run([sys.executable, str(HANDOFF / "check_python.py"), "--write-exe", str(exe_file)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=utf8_env())
    lines = result.stdout.splitlines()
    parsed = check_python.parse_line(lines[0])
    assert parsed is not None
    expected_ok = check_python.evaluate(check_python.interpreter_facts())[0]
    assert result.returncode == (0 if expected_ok else 3) and (parsed["status"] == "ok") is expected_ok
    assert exe_file.read_text(encoding="utf-8").strip() == sys.executable
    usage = subprocess.run([sys.executable, str(HANDOFF / "check_python.py"), "--bogus"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=utf8_env())
    assert usage.returncode == 2


def test_check_python_is_old_interpreter_syntax_compatible():
    source = (HANDOFF / "check_python.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 7))
    token_types = {tok.type for tok in tokenize.generate_tokens(io.StringIO(source).readline)}
    assert getattr(tokenize, "FSTRING_START", -1) not in token_types, "f-strings are a SyntaxError before Python 3.6"
    assert "from __future__ import annotations" not in source and ":=" not in source
    tree = ast.parse(source)
    assert not any(isinstance(n, (ast.AnnAssign,)) or (isinstance(n, ast.FunctionDef) and (n.returns or any(a.annotation for a in n.args.args)))
                   for n in ast.walk(tree)), "annotations are a SyntaxError on Python 2"


# ============================================================================ result blocks and step runner
def test_failure_block_format(tmp_path):
    log = tmp_path / "setup_logs" / "05_dependencies.log"
    assert common.failure_block("INSTALLATION FAILED", "[5/10] Installing dependencies", "pip failed", "Use offline mode.", log) == [
        RULE, "INSTALLATION FAILED", RULE, "Step: [5/10] Installing dependencies", "Reason: pip failed",
        "Recommended action: Use offline mode.", f"Log file: {log.resolve()}"]
    assert common.failure_block("VERIFICATION FAILED", "s", "r", "a")[-1] == "Recommended action: a"


def test_step_runner_output_and_records(capsys, tmp_path):
    runner = common.StepRunner(total=10)
    assert runner.run(4, "Installing build tools", lambda: "pip 26.2.1") == "pip 26.2.1"

    def skipped():
        raise common.StepSkipped("TESTS SKIPPED")

    assert runner.run(8, "Running test suite", skipped) is None
    log = tmp_path / "x.log"

    def failing():
        raise common.StepFailure("no network", "use offline mode", log, [f"line {i}" for i in range(40)])

    with pytest.raises(common.StepFailure) as info:
        runner.run(5, "Installing dependencies", failing)
    assert info.value.step == "[5/10] Installing dependencies"
    with pytest.raises(common.StepFailure) as unexpected:
        runner.run(6, "Installing falls_ml", lambda: 1 / 0)
    assert "ZeroDivisionError" in unexpected.value.reason and unexpected.value.step == "[6/10] Installing falls_ml"
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == ["[4/10] Installing build tools...", "[OK] pip 26.2.1", ""]
    assert out[3:6] == ["[8/10] Running test suite...", "[SKIPPED] TESTS SKIPPED", ""]
    assert "[FAILED] no network" in out and "    line 39" in out and "    line 9" not in out
    assert [r.status for r in runner.records] == ["ok", "skipped", "failed", "failed"]


@pytest.mark.parametrize("text, expected", [
    ("....\n=== 978 passed, 12 deselected, 3 warnings in 61.23s (0:01:01) ===", {"passed": 978, "deselected": 12, "warnings": 3, "failed": 0}),
    ("FAILED tests/unit/test_x.py::test_y\n2 failed, 976 passed, 12 deselected in 70.00s", {"passed": 976, "failed": 2}),
    ("1 passed, 1 error in 0.50s", {"passed": 1, "errors": 1}),
    ("no tests ran in 0.01s", {"passed": 0}),
])
def test_parse_pytest_summary(text, expected):
    summary = common.parse_pytest_summary(text)
    assert summary is not None
    for key, value in expected.items():
        assert summary[key] == value
    assert common.parse_pytest_summary("collecting ...\nInterrupted") is None


@pytest.mark.parametrize("output, mode, kind, action_part", [
    ("Retrying ... ProxyError('Cannot connect to proxy.')\nERROR: No matching distribution found for numpy==2.5.3", "online", "proxy", "offline"),
    ("SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED]'))", "online", "ssl", "PIP_CERT"),
    ("Failed to establish a new connection: [Errno 11001] getaddrinfo failed", "online", "network", "prepare_offline_package.cmd"),
    ("ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE.", "offline", "hash", "prepare_offline_package.cmd"),
    ("ERROR: No matching distribution found for scipy==1.18.1", "offline", "not_found", "prepare_offline_package.cmd"),
    ("OSError: [WinError 206] The filename or extension is too long", "online", "long_path", "short folder"),
    ("something odd", "online", "unknown", "log file"),
])
def test_classify_pip_failure(output, mode, kind, action_part):
    got_kind, reason, action = common.classify_pip_failure(output, mode, platform="win32")
    assert got_kind == kind and reason and action_part in action


# ============================================================================ marker-protected output folders
def _marked(path: Path) -> Path:
    return common.mark_generated(path, "test")


def test_clean_outputs_never_deletes_unmarked_folders(tmp_path, capsys):
    demo = tmp_path / "demo_outputs"
    _marked(demo)
    marked = _marked(demo / "baseline")
    (marked / "runs" / "run1").mkdir(parents=True)
    (marked / "runs" / "run1" / "metrics.json").write_text("{}", encoding="utf-8")
    unmarked = demo / "my_own_analysis"
    (unmarked / "nested_marked").mkdir(parents=True)
    _marked(unmarked / "nested_marked")
    (demo / "notes.txt").write_text("keep me", encoding="utf-8")
    precious = tmp_path / "data"
    _marked(precious)
    (precious / "fixture.parquet").write_text("x", encoding="utf-8")
    link = demo / "link_to_data"
    try:
        link.symlink_to(precious, target_is_directory=True)
    except OSError:
        link = None
    fake_marker_dir = demo / "fake"
    fake_marker_dir.mkdir()
    (fake_marker_dir / common.MARKER_FILE).mkdir()  # a directory named like the marker is not a marker

    assert clean_outputs.clean(demo, dry_run=True) == 0
    assert marked.exists(), "dry run deletes nothing"
    assert clean_outputs.clean(demo, dry_run=False) == 0
    assert not marked.exists()
    assert unmarked.exists() and (unmarked / "nested_marked").exists() and (demo / "notes.txt").exists()
    assert fake_marker_dir.exists() and precious.exists() and (precious / "fixture.parquet").exists()
    if link is not None:
        assert link.is_symlink()
    assert demo.exists(), "root with unmarked content is kept"
    output = capsys.readouterr().out
    assert "would remove" in output and "kept (not created by these scripts" in output

    for entry in (unmarked, fake_marker_dir):
        shutil.rmtree(entry)
    (demo / "notes.txt").unlink()
    if link is not None:
        link.unlink()
    assert clean_outputs.clean(demo, dry_run=False) == 0
    assert not demo.exists(), "marked root with nothing unmarked left is removed"
    assert precious.exists()
    assert clean_outputs.clean(demo, dry_run=False) == 0  # nothing to clean


def test_unmarked_root_is_never_removed(tmp_path):
    demo = tmp_path / "demo_outputs"
    demo.mkdir()
    _marked(demo / "verify")
    clean_outputs.clean(demo, dry_run=False)
    assert demo.exists() and not (demo / "verify").exists()


def test_prepare_generated_dir_replaces_only_marked_folders(tmp_path):
    target = tmp_path / "baseline"
    common.prepare_generated_dir(target, "demo")
    assert (target / common.MARKER_FILE).is_file() and (target / common.SYNTHETIC_NOTICE_FILE).is_file()
    (target / "old.txt").write_text("old", encoding="utf-8")
    common.prepare_generated_dir(target, "demo")
    assert not (target / "old.txt").exists() and common.is_generated(target)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "results.csv").write_text("important", encoding="utf-8")
    with pytest.raises(common.OutputDirError, match="not created by these scripts"):
        common.prepare_generated_dir(foreign, "demo")
    assert (foreign / "results.csv").read_text(encoding="utf-8") == "important"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert common.is_generated(common.prepare_generated_dir(empty, "demo"))
    notice = (target / common.SYNTHETIC_NOTICE_FILE).read_text(encoding="utf-8")
    assert notice.startswith(common.SYNTHETIC_BANNER)


# ============================================================================ offline bundle validation
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle(tmp_path: Path, *, target: dict | None = None, extra_lock_line: str = "") -> dict[str, Path]:
    """Two fake wheels, a win32-only wheel, matching locks and manifest."""
    offline = tmp_path / "offline_packages"
    offline.mkdir()
    wheels = {"numpy-2.5.3-cp313-cp313-win_amd64.whl": b"numpy wheel", "PyYAML-6.0.3-cp313-cp313-win_amd64.whl": b"yaml wheel",
              "colorama-0.4.6-py2.py3-none-any.whl": b"colorama wheel", "pip-26.2.1-py3-none-any.whl": b"pip wheel"}
    for name, data in wheels.items():
        (offline / name).write_bytes(data)
    lock = tmp_path / "requirements.lock"
    lock.write_text("# test lock\n"
                    f"numpy==2.5.3 --hash=sha256:{_sha(wheels['numpy-2.5.3-cp313-cp313-win_amd64.whl'])}\n"
                    f"pyyaml==6.0.3 --hash=sha256:{_sha(b'other platform')} --hash=sha256:{_sha(wheels['PyYAML-6.0.3-cp313-cp313-win_amd64.whl'])}\n"
                    f"colorama==0.4.6 ; sys_platform == \"win32\" --hash=sha256:{_sha(wheels['colorama-0.4.6-py2.py3-none-any.whl'])}\n"
                    + extra_lock_line, encoding="utf-8")
    build = tmp_path / "requirements-build.lock"
    build.write_text(f"pip==26.2.1 --hash=sha256:{_sha(wheels['pip-26.2.1-py3-none-any.whl'])}\n", encoding="utf-8")
    manifest = {"format": 1, "created_utc": "2026-09-15T00:00:00Z",
                "target": target or {"name": common.WINDOWS_TARGET, "platform_tags": ["win_amd64"], "python_version": RUNNING_SERIES,
                                     "implementation": "cp", "abi": RUNNING_ABI},
                "requirements_lock_sha256": _sha(lock.read_bytes()), "build_lock_sha256": _sha(build.read_bytes()),
                "files": {name: _sha(data) for name, data in wheels.items()}}
    (offline / common.OFFLINE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return {"offline": offline, "lock": lock, "build": build}


def _validate(paths: dict[str, Path], **kwargs):
    kwargs.setdefault("sys_platform", "win32")
    kwargs.setdefault("host_platform", "win-amd64")
    return common.validate_offline_bundle(paths["offline"], paths["lock"], paths["build"], **kwargs)


def _kinds(problems) -> set[str]:
    return {p.kind for p in problems}


def test_offline_bundle_valid_for_windows(tmp_path):
    assert _validate(_bundle(tmp_path)) == []


def test_offline_bundle_current_target_matches_host_platform(tmp_path):
    target = {"name": "current", "platform_tags": ["macosx_15_0_arm64"], "python_version": RUNNING_SERIES, "implementation": "cp", "abi": RUNNING_ABI}
    paths = _bundle(tmp_path, target=target)
    (paths["offline"] / "colorama-0.4.6-py2.py3-none-any.whl").unlink()  # not needed off Windows, but listed -> missing
    problems = _validate(paths, sys_platform="darwin", host_platform="macosx-15.0-arm64")
    assert _kinds(problems) == {"missing_file"}
    assert _kinds(_validate(paths, sys_platform="darwin", host_platform="macosx-15.0-x86_64")) >= {"platform"}
    assert "platform" in _kinds(_validate(paths))  # a macOS bundle on Windows


def test_offline_bundle_windows_bundle_rejected_elsewhere(tmp_path):
    problems = _validate(_bundle(tmp_path), sys_platform="darwin", host_platform="macosx-15.0-arm64")
    assert "platform" in _kinds(problems)
    reason, action = common.offline_problem_action(problems, platform="darwin")
    assert "another platform" in reason and "--target current" in action


def test_offline_bundle_missing_manifest(tmp_path):
    paths = _bundle(tmp_path)
    (paths["offline"] / common.OFFLINE_MANIFEST).unlink()
    problems = _validate(paths)
    assert _kinds(problems) == {"missing_manifest"}
    reason, action = common.offline_problem_action(problems, platform="win32")
    assert "OFFLINE_MANIFEST.json not found" in reason and "prepare_offline_package.cmd" in action and "--online" in action


def test_offline_bundle_lock_mismatch(tmp_path):
    paths = _bundle(tmp_path)
    paths["lock"].write_text(paths["lock"].read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    problems = _validate(paths)
    assert _kinds(problems) == {"lock_mismatch"}
    reason, action = common.offline_problem_action(problems, platform="win32")
    assert "requirements.lock" in reason and "Re-run prepare_offline_package.cmd" in action


def test_offline_bundle_tampered_and_missing_wheels(tmp_path):
    paths = _bundle(tmp_path)
    (paths["offline"] / "numpy-2.5.3-cp313-cp313-win_amd64.whl").write_bytes(b"tampered")
    (paths["offline"] / "PyYAML-6.0.3-cp313-cp313-win_amd64.whl").unlink()
    problems = _validate(paths)
    assert _kinds(problems) == {"hash", "missing_file", "incomplete"}
    assert any("numpy" in p.message for p in problems if p.kind == "hash")
    assert any("pyyaml==6.0.3" in p.message for p in problems if p.kind == "incomplete")


def test_offline_bundle_incomplete_and_wrong_python(tmp_path):
    extra = f"scipy==1.18.1 --hash=sha256:{_sha(b'scipy')}\n"
    paths = _bundle(tmp_path, extra_lock_line=extra)
    problems = _validate(paths)
    assert _kinds(problems) == {"incomplete"} and "scipy==1.18.1" in problems[0].message
    (tmp_path / "second").mkdir()
    paths = _bundle(tmp_path / "second", target={"name": common.WINDOWS_TARGET, "platform_tags": ["win_amd64"], "python_version": "3.12",
                                                   "implementation": "cp", "abi": "cp312"})
    assert "python" in _kinds(_validate(paths))


def test_offline_bundle_requirement_without_wheel(tmp_path):
    paths = _bundle(tmp_path)
    manifest_path = paths["offline"] / common.OFFLINE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths["lock"].write_text(paths["lock"].read_text(encoding="utf-8") + f"scipy==1.18.1 --hash=sha256:{_sha(b'scipy')}\n", encoding="utf-8")
    manifest["requirements_lock_sha256"] = _sha(paths["lock"].read_bytes())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    problems = _validate(paths)
    assert _kinds(problems) == {"incomplete"} and "scipy==1.18.1" in problems[0].message
    # the win32-only colorama requirement is not required on macOS, but a Windows bundle is still the wrong platform
    assert "incomplete" in _kinds(_validate(paths, sys_platform="darwin", host_platform="macosx-15.0-arm64"))


def test_offline_bundle_wheel_hash_not_in_lock(tmp_path):
    paths = _bundle(tmp_path)
    wheel = paths["offline"] / "pip-26.2.1-py3-none-any.whl"
    wheel.write_bytes(b"different pip")
    manifest_path = paths["offline"] / common.OFFLINE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][wheel.name] = _sha(b"different pip")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    problems = _validate(paths)
    assert _kinds(problems) == {"hash"} and "not listed in the lock" in problems[0].message


@pytest.mark.parametrize("value, expected", [
    ("win-amd64", ("win", "x86_64")), ("win_amd64", ("win", "x86_64")), ("win32", ("win", "x86")), ("win-arm64", ("win", "arm64")),
    ("macosx-15.0-arm64", ("macosx", "arm64")), ("macosx_11_0_arm64", ("macosx", "arm64")), ("macosx_10_9_universal2", ("macosx", "universal2")),
    ("linux-x86_64", ("linux", "x86_64")), ("manylinux_2_17_aarch64", ("linux", "arm64")),
])
def test_platform_family(value, expected):
    assert common.platform_family(value) == expected


def test_tags_match_host():
    assert common.tags_match_host(["win_amd64"], ("win", "x86_64"))
    assert common.tags_match_host(["macosx_11_0_universal2"], ("macosx", "arm64"))
    assert not common.tags_match_host(["win_amd64"], ("macosx", "arm64"))
    assert not common.tags_match_host(["macosx_15_0_arm64"], ("macosx", "x86_64"))


# ============================================================================ installer commands and modes
def test_install_commands_follow_the_contract(tmp_path):
    python = Path("C:/p/.venv/Scripts/python.exe")
    lock = Path("requirements.lock")
    online = setup_steps.lock_install_command(python, lock, "online", tmp_path)
    assert online == [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--require-hashes",
                      "--only-binary=:all:", "-r", str(lock)]
    offline = setup_steps.lock_install_command(python, lock, "offline", tmp_path)
    assert offline[:8] == online[:8] and offline[8:11] == ["--no-index", "--find-links", str(tmp_path / "offline_packages")]
    assert setup_steps.project_install_command(python) == [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                                                           "--no-deps", "--no-build-isolation", "--no-index", "-e", "."]


def test_resolve_mode(tmp_path):
    assert setup_steps.resolve_mode("auto", tmp_path)[0] == "online"
    (tmp_path / "offline_packages").mkdir()
    mode, text = setup_steps.resolve_mode("auto", tmp_path)
    assert mode == "online" and "ignored" in text
    (tmp_path / "offline_packages" / "OFFLINE_MANIFEST.json").write_text("{}", encoding="utf-8")
    assert setup_steps.resolve_mode("auto", tmp_path)[0] == "offline"
    assert setup_steps.resolve_mode("online", tmp_path)[0] == "online"
    assert setup_steps.resolve_mode("offline", tmp_path)[0] == "offline"


def test_path_length_problem():
    short = Path("C:/Projects/Falls Research (test)/falls_ml_handoff")
    assert common.path_length_problem(short, platform="win32", long_paths_enabled=False) is None
    deep = Path("C:/" + "x" * 150)
    assert "characters" in common.path_length_problem(deep, platform="win32", long_paths_enabled=False)
    assert common.path_length_problem(deep, platform="win32", long_paths_enabled=True) is None
    assert common.path_length_problem(deep, platform="darwin") is None


def test_path_length_limit_matches_measured_depths_and_docs():
    limit = common.windows_project_path_limit()
    assert limit == common.WINDOWS_MAX_PATH - 1 - common.DEEPEST_RELATIVE_PATH - 1 == 103
    assert common.DEEPEST_RELATIVE_PATH >= 154, "deepest measured path: verify --full-tests run folders under demo_outputs/pt"
    assert common.path_length_problem(Path("C:/" + "x" * (limit - 3)), platform="win32", long_paths_enabled=None) is None
    assert f"at most {limit} characters" in common.path_length_problem(Path("C:/" + "x" * (limit - 2)), platform="win32")
    handoff = (ROOT / "WINDOWS_HANDOFF.md").read_text(encoding="utf-8")
    assert f"At most **{limit} characters**" in handoff and f"it must be at most {limit} characters" in handoff
    assert f"at most {limit} characters on Windows" in (HANDOFF / "setup_steps.py").read_text(encoding="utf-8")


def test_pytest_basetemp_is_short_everywhere():
    assert common.PYTEST_BASETEMP == "pt"
    for rel in ("WINDOWS_HANDOFF.md", "verify_installation.cmd", "scripts/handoff/verify.py", "scripts/handoff/setup_steps.py"):
        text = (ROOT / rel).read_bytes().decode("utf-8")
        assert "pytest_tmp" not in text, rel
        assert "demo_outputs\\pt" in text or "demo_outputs/pt" in text, rel


def test_run_tests_uses_short_basetemp_and_remarks_it_after_interrupt(tmp_path, monkeypatch):
    import verify

    seen: dict[str, Path] = {}

    def interrupted_pytest(cmd, log_path, **kwargs):
        basetemp = Path(cmd[cmd.index("--basetemp") + 1])
        seen["basetemp"] = basetemp
        shutil.rmtree(basetemp, ignore_errors=True)  # pytest empties --basetemp (marker included) when it starts
        (basetemp / "test_something0").mkdir(parents=True)
        raise KeyboardInterrupt

    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(common, "run_logged", interrupted_pytest)
    with pytest.raises(KeyboardInterrupt):
        verify.run_tests(tmp_path / "tests.log", full=False)
    assert seen["basetemp"] == tmp_path / "demo_outputs" / "pt"
    assert common.is_generated(seen["basetemp"]), "a half-finished basetemp must stay recognisable as generated"


def test_subprocess_env_strips_leaking_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    monkeypatch.setenv("PIP_USER", "1")
    monkeypatch.setenv("PIP_TARGET", "/elsewhere/target")
    monkeypatch.delenv("PIP_PREFIX", raising=False)
    monkeypatch.delenv(common.CLEARED_PIP_SETTINGS_VAR, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://extra.example/simple")
    monkeypatch.setattr(common, "matplotlib_config_dir", lambda: tmp_path / "mplconfig")
    env = common.subprocess_env(no_bytecode=True)
    assert "PYTHONPATH" not in env and not {"PIP_USER", "PIP_TARGET", "PIP_PREFIX"} & set(env)
    assert env[common.CLEARED_PIP_SETTINGS_VAR] == "PIP_USER PIP_TARGET", "names only, never values"
    assert env["HTTPS_PROXY"] == "http://proxy.example:8080", "proxy settings must reach pip"
    assert env["PIP_INDEX_URL"] == "https://mirror.example/simple" and env["PIP_EXTRA_INDEX_URL"] == "https://extra.example/simple"
    assert env["PYTHONUTF8"] == "1" and env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1" and env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["MPLCONFIGDIR"] == str(tmp_path / "mplconfig")
    # a grandchild (setup_steps -> environment_report) still knows which settings were cleared upstream
    monkeypatch.delenv("PIP_USER")
    monkeypatch.delenv("PIP_TARGET")
    monkeypatch.setenv(common.CLEARED_PIP_SETTINGS_VAR, " PIP_PREFIX")
    assert common.cleared_pip_settings() == ["PIP_PREFIX"]
    assert common.subprocess_env()[common.CLEARED_PIP_SETTINGS_VAR] == "PIP_PREFIX"
    monkeypatch.delenv(common.CLEARED_PIP_SETTINGS_VAR)
    monkeypatch.setattr(common, "matplotlib_config_dir", lambda: None)
    env = common.subprocess_env()
    assert common.CLEARED_PIP_SETTINGS_VAR not in env
    assert env.get("MPLCONFIGDIR") == os.environ.get("MPLCONFIGDIR"), "without .venv an inherited value (if any) is left alone"


def test_matplotlib_config_dir_never_creates_the_virtual_environment(tmp_path):
    assert common.matplotlib_config_dir(tmp_path) is None
    assert not (tmp_path / ".venv").exists(), "an empty .venv would make setup report a broken virtual environment"
    (tmp_path / ".venv").mkdir()
    assert common.matplotlib_config_dir(tmp_path) == tmp_path / ".venv" / "mplconfig" and (tmp_path / ".venv" / "mplconfig").is_dir()
    assert common.matplotlib_config_dir(tmp_path) == tmp_path / ".venv" / "mplconfig", "idempotent"


def test_json_outputs_record_posix_relative_paths(tmp_path):
    inner = tmp_path / "Falls Research" / "runs" / "2026-09-15_x_0123abcd"
    inner.mkdir(parents=True)
    assert common.posix_path(inner, tmp_path / "Falls Research") == "runs/2026-09-15_x_0123abcd"
    assert common.posix_path(tmp_path / "elsewhere", inner) == (tmp_path / "elsewhere").as_posix()
    tree = ast.parse((HANDOFF / "smoke.py").read_text(encoding="utf-8"))
    stored = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
              and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "common.display_path"]
    assert not stored, "SMOKE_RESULT.json entries must use common.posix_path, not native display paths"
    assert "common.posix_path(run_dir, out)" in ast.unparse(tree) and "common.posix_path(resaved_dir, out)" in ast.unparse(tree)


def test_smoke_checks_require_every_run_within_tolerance():
    good = {"artifacts_ok": True, "bundle_reload_max_abs_diff": 0.0, "resave_reload_max_abs_diff": 1e-16, "cli_predict_max_abs_diff": 5e-16}
    runs = [{"config": name, **good} for name in smoke.CONFIGS]
    assert all(smoke.summarize_checks(runs).values())
    assert not any(smoke.summarize_checks(runs[:1]).values())
    bad = [dict(runs[0]), {**runs[1], "cli_predict_max_abs_diff": 1e-6}]
    checks = smoke.summarize_checks(bad)
    assert checks["bundle_reload_equal"] and not checks["cli_predict_equal"]


def test_environment_report_never_records_variable_values(tmp_path, monkeypatch):
    out = tmp_path / "environment_report.txt"
    for name in (*common.PIP_TARGET_SETTINGS, common.CLEARED_PIP_SETTINGS_VAR):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PIP_TARGET", "C:\\do-not-record-5e1d\\site")  # cleared by subprocess_env, recorded by name only
    env = common.subprocess_env(extra={"PIP_INDEX_URL": "https://do-not-record-7f3a.mirror.example/simple"})
    assert "PIP_TARGET" not in env
    result = subprocess.run([sys.executable, str(HANDOFF / "environment_report.py"), "--out", str(out), "--mode", "online"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(tmp_path), timeout=120)
    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "do-not-record-7f3a" not in text and "mirror.example" not in text and "do-not-record-5e1d" not in text
    assert "PIP_INDEX_URL set: yes" in text and "install mode: online" in text
    assert "PIP_TARGET set: yes" in text and "PIP_USER set: no" in text and "PIP_PREFIX set: no" in text
    assert re.search(r"falls_ml\.__version__: \d+\.\d+\.\d+", text) and "requirements.lock sha256:" in text
    assert "Installed distributions" in text
