@echo off
REM ============================================================================
REM run_demo.cmd - baseline SYNTHETIC demonstration of falls_ml
REM
REM Usage:  run_demo.cmd
REM
REM Validates the synthetic fixture, trains the published eFalls scoring and the retrained
REM eFalls LASSO experiments, writes reports, metrics and model bundles, reloads the bundles,
REM scores a CSV with the command-line interface and compares the two runs.
REM Outputs: demo_outputs\baseline (replaced on every run; see DEMO_SUMMARY.md).
REM SYNTHETIC DATA - NOT SCIENTIFIC RESULTS. Run setup_windows.cmd first.
REM Set FALLS_ML_NO_PAUSE=1 for automation: the window then never waits for a key press at the end
REM (by default it waits only when the file was started by double-click, so the result stays visible).
REM ============================================================================
setlocal EnableExtensions DisableDelayedExpansion
set "RC=1"
set "PAUSE_AT_END="
set "CMDLINE_CHECK=%cmdcmdline:"=%"
if /i not "%CMDLINE_CHECK:run_demo=%"=="%CMDLINE_CHECK%" if /i not "%CMDLINE_CHECK: /c =%"=="%CMDLINE_CHECK%" set "PAUSE_AT_END=1"
if "%FALLS_ML_NO_PAUSE%"=="1" set "PAUSE_AT_END="
pushd "%~dp0" || goto pushd_failed

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONHOME="
set "PYTHONPATH="
REM pip settings that would redirect installs away from .venv are cleared; their names (never values) are kept for
REM environment_report.txt. PIP_INDEX_URL, PIP_EXTRA_INDEX_URL, proxies and pip.ini stay in effect.
if defined PIP_USER set "FALLS_ML_CLEARED_PIP_SETTINGS=%FALLS_ML_CLEARED_PIP_SETTINGS% PIP_USER"
if defined PIP_TARGET set "FALLS_ML_CLEARED_PIP_SETTINGS=%FALLS_ML_CLEARED_PIP_SETTINGS% PIP_TARGET"
if defined PIP_PREFIX set "FALLS_ML_CLEARED_PIP_SETTINGS=%FALLS_ML_CLEARED_PIP_SETTINGS% PIP_PREFIX"
set "PIP_USER="
set "PIP_TARGET="
set "PIP_PREFIX="
REM matplotlib cache inside this folder (never in the user profile); .venv itself is never created here
if exist ".venv\" if not exist ".venv\mplconfig\" mkdir ".venv\mplconfig" >nul 2>&1
if exist ".venv\mplconfig\" set "MPLCONFIGDIR=%CD%\.venv\mplconfig"
set "VENV_PY=.venv\Scripts\python.exe"
set "DEMO_PY=scripts\handoff\demo.py"

if "%~1"=="" goto args_done
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
if "%~1"=="/?" goto usage
for %%M in ("%~1") do echo ERROR: unknown option %%~M
set "RC=2"
goto usage_text

:args_done
if not exist "%VENV_PY%" goto no_venv
if not exist "%DEMO_PY%" goto incomplete_package
"%VENV_PY%" "%DEMO_PY%" baseline
set "RC=%ERRORLEVEL%"
goto finish

:no_venv
echo ========================================
echo DEMO FAILED
echo ========================================
echo Step: checking the virtual environment
echo Reason: .venv\Scripts\python.exe was not found in this folder
echo Recommended action: run setup_windows.cmd first
set "RC=1"
goto finish

:incomplete_package
echo ========================================
echo DEMO FAILED
echo ========================================
echo Step: checking the package folder
echo Reason: scripts\handoff\demo.py is missing - the package is incomplete
echo Recommended action: copy the complete falls_ml_handoff folder again and run setup_windows.cmd
set "RC=1"
goto finish

:usage
set "RC=0"
:usage_text
echo Usage: run_demo.cmd
echo   Runs the baseline synthetic demo into demo_outputs\baseline.
goto finish

:finish
popd
if defined PAUSE_AT_END pause
endlocal & exit /b %RC%

:pushd_failed
echo ERROR: cannot open the folder that contains run_demo.cmd.
if defined PAUSE_AT_END pause
endlocal & exit /b 1
