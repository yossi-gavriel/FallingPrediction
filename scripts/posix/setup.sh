#!/usr/bin/env bash
# setup.sh - POSIX mirror of setup_windows.cmd (macOS arm64 developers and clean-room tests).
#
# Usage: scripts/posix/setup.sh [--offline | --online] [--recreate-venv] [--skip-tests]
#   --online         install packages from the configured package index (pip.conf, PIP_INDEX_URL, HTTPS_PROXY)
#   --offline        install only from offline_packages/ (see scripts/posix/prepare_offline.sh)
#   (neither)        offline if offline_packages/OFFLINE_MANIFEST.json exists, else online
#   --recreate-venv  delete .venv (and nothing else) and create it again
#   --skip-tests     skip the test suite - for diagnosis only
#
# Same steps, messages and Python helpers as setup_windows.cmd. Synthetic data only.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 1
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)" || exit 1
cd "$ROOT" || exit 1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONHOME PYTHONPATH
# pip settings that would redirect installs away from .venv are cleared; their names (never values) are kept for
# environment_report.txt. PIP_INDEX_URL, PIP_EXTRA_INDEX_URL, proxies and pip.conf stay in effect.
for pip_setting in PIP_USER PIP_TARGET PIP_PREFIX; do
  if [ -n "${!pip_setting+x}" ]; then
    FALLS_ML_CLEARED_PIP_SETTINGS="${FALLS_ML_CLEARED_PIP_SETTINGS:-} $pip_setting"
    unset "$pip_setting"
  fi
done
if [ -n "${FALLS_ML_CLEARED_PIP_SETTINGS:-}" ]; then export FALLS_ML_CLEARED_PIP_SETTINGS; fi

LOG_DIR="setup_logs"
CHECK_PY="scripts/handoff/check_python.py"
STEPS_PY="scripts/handoff/setup_steps.py"
VENV_DIR=".venv"
VENV_PY=".venv/bin/python"
SETUP="bash scripts/posix/setup.sh"
MODE=""
SKIP_TESTS=0
RECREATE_VENV=0

fail() {  # fail <step> <reason> <action> [log file]
  echo
  if [ -n "${4:-}" ] && [ -f "$4" ]; then
    echo "Details from the log file:"
    cat "$4"
    echo
  fi
  echo "========================================"
  echo "INSTALLATION FAILED"
  echo "========================================"
  echo "Step: $1"
  echo "Reason: $2"
  echo "Recommended action: $3"
  if [ -n "${4:-}" ]; then echo "Log file: $ROOT/$4"; fi
  exit 1
}

usage() {
  echo "Usage: $SETUP [--offline | --online] [--recreate-venv] [--skip-tests]"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --offline|--online)
      if [ -n "$MODE" ] && [ "$MODE" != "${1#--}" ]; then echo "ERROR: --offline and --online cannot be combined."; usage; exit 2; fi
      MODE="${1#--}" ;;
    --recreate-venv) RECREATE_VENV=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option $1"; usage; exit 2 ;;
  esac
  shift
done

echo "========================================"
echo "falls_ml handoff - POSIX setup"
echo "========================================"
echo "Project folder: $ROOT"
echo
[ -f "$CHECK_PY" ] && [ -f "$STEPS_PY" ] && [ -f pyproject.toml ] || fail "Checking the package folder" \
  "scripts/handoff/check_python.py, scripts/handoff/setup_steps.py or pyproject.toml is missing - the package is incomplete." \
  "Copy the complete falls_ml_handoff folder again and run $SETUP from inside it."
mkdir -p "$LOG_DIR" 2>/dev/null && touch "$LOG_DIR/write_test.tmp" 2>/dev/null || fail "Checking the package folder" \
  "cannot create or write the setup_logs folder here - the folder is read-only." "Copy the package to a writable folder and run $SETUP there."
rm -f "$LOG_DIR/write_test.tmp"

# ---------------------------------------------------------------------------- [1/10]
echo "[1/10] Checking Python..."
PY_LOG="$LOG_DIR/01_python.log"
PROBE_OUT="$LOG_DIR/01_python_probe.txt"
PROBE_EXE="$LOG_DIR/01_python_exe.txt"
PY_EXE=""
PY_VER=""
PY_REJECTED=""
echo "Python detection log" >"$PY_LOG"

probe_python() {  # probe_python <command...>: sets PY_EXE and PY_VER when the interpreter is supported
  local rc
  rm -f "$PROBE_OUT" "$PROBE_EXE"
  "$@" "$CHECK_PY" --write-exe "$PROBE_EXE" </dev/null >"$PROBE_OUT" 2>&1
  rc=$?
  { echo "---- candidate: $* / exit code $rc"; cat "$PROBE_OUT"; } >>"$PY_LOG" 2>&1
  if [ "$rc" -eq 0 ] && grep -q '^PYTHON|ok|' "$PROBE_OUT" && [ -s "$PROBE_EXE" ]; then
    PY_EXE="$(head -n 1 "$PROBE_EXE")"
    PY_VER="$(grep '^PYTHON|' "$PROBE_OUT" | head -n 1 | cut -d'|' -f3)"
    return 0
  fi
  if [ "$rc" -eq 3 ]; then
    PY_REJECTED="$PY_REJECTED [$*: $(grep '^REASON|' "$PROBE_OUT" | head -n 1 | cut -d'|' -f2-)]"
  fi
  return 1
}

probe_python python3.13 || probe_python python3.11 || probe_python python3 || probe_python python || {
  reason="Python 3.11 or 3.13 (64-bit) was not found. Tried: python3.13, python3.11, python3, python."
  [ -n "$PY_REJECTED" ] && reason="$reason Found but not usable:$PY_REJECTED."
  fail "[1/10] Checking Python" "$reason" \
    "Install Python 3.13 or 3.11 (macOS arm64: python.org installer or 'brew install python@3.13'), open a new terminal and run $SETUP again." "$PY_LOG"
}
echo "[OK] Python $PY_VER ($PY_EXE)"
echo

# ---------------------------------------------------------------------------- [2/10]
echo "[2/10] Creating virtual environment..."
VENV_LOG="$LOG_DIR/02_venv.log"
echo "Virtual environment log" >"$VENV_LOG"
if [ "$RECREATE_VENV" -eq 1 ] && [ -e "$VENV_DIR" ]; then
  echo "  Removing the existing .venv folder because --recreate-venv was given..."
  rm -rf -- "$VENV_DIR" >>"$VENV_LOG" 2>&1
  [ -e "$VENV_DIR" ] && fail "[2/10] Creating virtual environment" "the existing .venv folder could not be deleted." \
    "Close programs that use .venv, then run $SETUP --recreate-venv again." "$VENV_LOG"
fi
VENV_STATE="reused"
if [ ! -e "$VENV_PY" ]; then
  [ -e "$VENV_DIR" ] && fail "[2/10] Creating virtual environment" \
    ".venv exists but has no bin/python - it was created on another computer or damaged." \
    "Run $SETUP --recreate-venv - it deletes only the .venv folder and builds it again." "$VENV_LOG"
  echo "  Creating .venv with Python $PY_VER..."
  "$PY_EXE" -m venv "$VENV_DIR" </dev/null >>"$VENV_LOG" 2>&1 || fail "[2/10] Creating virtual environment" \
    "python -m venv failed." "Read the log file; repair the Python installation, then run $SETUP --recreate-venv." "$VENV_LOG"
  VENV_STATE="created"
fi
"$VENV_PY" "$CHECK_PY" </dev/null >>"$VENV_LOG" 2>&1 || fail "[2/10] Creating virtual environment" \
  ".venv exists but its Python does not start or is not Python 3.11/3.13 64-bit - it was created elsewhere, moved, or Python was reinstalled." \
  "Run $SETUP --recreate-venv - it deletes only the .venv folder and builds it again." "$VENV_LOG"
if [ "$VENV_STATE" = "created" ]; then
  echo "[OK] Created the virtual environment .venv"
else
  echo "[OK] Reusing the existing virtual environment .venv - use --recreate-venv to rebuild it"
fi
echo

# matplotlib cache inside this folder (never in the user profile); .venv exists from here on
[ -d "$VENV_DIR/mplconfig" ] || mkdir "$VENV_DIR/mplconfig" 2>/dev/null
if [ -d "$VENV_DIR/mplconfig" ]; then export MPLCONFIGDIR="$ROOT/$VENV_DIR/mplconfig"; fi

# ---------------------------------------------------------------------------- [3/10]
echo "[3/10] Activating virtual environment..."
ACT_LOG="$LOG_DIR/03_activate.log"
echo "Activation log" >"$ACT_LOG"
[ -f "$VENV_DIR/bin/activate" ] || fail "[3/10] Activating virtual environment" ".venv/bin/activate is missing." "Run $SETUP --recreate-venv." "$ACT_LOG"
set +u
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate" >>"$ACT_LOG" 2>&1
set -u
[ -n "${VIRTUAL_ENV:-}" ] || fail "[3/10] Activating virtual environment" "activating .venv did not set VIRTUAL_ENV." "Run $SETUP --recreate-venv." "$ACT_LOG"
python -c 'import os, sys; a = os.path.realpath(sys.prefix); b = os.path.realpath(".venv"); print("active prefix:", a); print("venv:         ", b); sys.exit(0 if a == b else 1)' \
  </dev/null >>"$ACT_LOG" 2>&1 || fail "[3/10] Activating virtual environment" \
  "after activation the python command is not the .venv interpreter - the folder was probably moved after .venv was created." \
  "Run $SETUP --recreate-venv." "$ACT_LOG"
echo "[OK] Virtual environment active"
echo

# ---------------------------------------------------------------------------- [4/10] .. [10/10]
set -- --first-step 4 --total-steps 10
[ -n "$MODE" ] && set -- "$@" --mode "$MODE"
[ "$SKIP_TESTS" -eq 1 ] && set -- "$@" --skip-tests
"$VENV_PY" "$STEPS_PY" "$@"
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
  fail "[4/10] to [10/10] installer steps" "scripts/handoff/setup_steps.py stopped unexpectedly with exit code $rc" \
    "Run $SETUP again. If it stops the same way, send the setup_logs folder to the maintainer."
fi
exit "$rc"
