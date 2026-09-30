#!/usr/bin/env bash
# prepare_offline.sh - POSIX mirror of prepare_offline_package.cmd.
# Usage: scripts/posix/prepare_offline.sh [--target windows-amd64-cp313 | windows-amd64-cp311 | current] [--force] [--verify-only]
# Run on a computer with internet access; uses .venv/bin/python when present, else python3.13, python3 or python.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)" || exit 1
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
# matplotlib cache inside this folder (never in the user profile); .venv itself is never created here
if [ -d .venv ] && [ ! -d .venv/mplconfig ]; then mkdir .venv/mplconfig 2>/dev/null; fi
if [ -d .venv/mplconfig ]; then export MPLCONFIGDIR="$ROOT/.venv/mplconfig"; fi
CHECK_PY="scripts/handoff/check_python.py"
PY=""
FALLBACK=""
for candidate in ".venv/bin/python" python3.13 python3 python; do
  command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ] || continue
  "$candidate" "$CHECK_PY" </dev/null >/dev/null 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then PY="$candidate"; break; fi
  if [ "$rc" -eq 3 ] && [ -z "$FALLBACK" ]; then FALLBACK="$candidate"; fi
done
if [ -z "$PY" ] && [ -n "$FALLBACK" ]; then
  PY="$FALLBACK"
  echo "NOTE: Python 3.13/3.11 64-bit was not found; using $PY (enough for the windows-amd64-cp313/cp311 targets)."
fi
if [ -z "$PY" ]; then
  echo "========================================"
  echo "OFFLINE PACKAGE FAILED"
  echo "========================================"
  echo "Step: finding Python"
  echo "Reason: no Python was found - tried .venv, python3.13, python3 and python"
  echo "Recommended action: install Python 3.13 and run this command again"
  exit 1
fi
"$PY" "scripts/handoff/prepare_offline.py" "$@"
exit $?
