#!/usr/bin/env bash
# clean_demo_outputs.sh - POSIX mirror of clean_demo_outputs.cmd.
# Usage: scripts/posix/clean_demo_outputs.sh [--dry-run]
# Deletes only folders under demo_outputs/ that contain the marker file .falls_ml_generated.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)" || exit 1
cd "$ROOT" || exit 1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
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
  echo "ERROR: .venv/bin/python was not found in this folder. Run bash scripts/posix/setup.sh first."
  exit 1
fi
".venv/bin/python" "scripts/handoff/clean_outputs.py" "$@"
exit $?
