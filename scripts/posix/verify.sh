#!/usr/bin/env bash
# verify.sh - POSIX mirror of verify_installation.cmd.
# Usage: scripts/posix/verify.sh [--skip-tests | --full-tests]
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
if [ ! -x ".venv/bin/python" ]; then
  echo "========================================"
  echo "VERIFICATION FAILED"
  echo "========================================"
  echo "Step: checking the virtual environment"
  echo "Reason: .venv/bin/python was not found in this folder"
  echo "Recommended action: run bash scripts/posix/setup.sh first"
  exit 1
fi
".venv/bin/python" "scripts/handoff/verify.py" "$@"
exit $?
