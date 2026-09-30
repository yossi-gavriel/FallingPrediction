@echo off
REM ============================================================================
REM clean_demo_outputs.cmd - delete generated demo and verification outputs
REM
REM Usage:  clean_demo_outputs.cmd [--dry-run]
REM   --dry-run   only list what would be deleted
REM
REM Deletes only folders under demo_outputs\ that contain the marker file .falls_ml_generated.
REM Never touches data\, configs\, src\, .venv\ or any folder without the marker.
REM Set FALLS_ML_NO_PAUSE=1 for automation: the window then never waits for a key press at the end
REM (by default it waits only when the file was started by double-click, so the result stays visible).
REM ============================================================================
setlocal EnableExtensions DisableDelayedExpansion
set "RC=1"
set "PAUSE_AT_END="
set "CMDLINE_CHECK=%cmdcmdline:"=%"
if /i not "%CMDLINE_CHECK:clean_demo_outputs=%"=="%CMDLINE_CHECK%" if /i not "%CMDLINE_CHECK: /c =%"=="%CMDLINE_CHECK%" set "PAUSE_AT_END=1"
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
set "CLEAN_PY=scripts\handoff\clean_outputs.py"
set "CLEAN_ARGS="

if "%~1"=="" goto args_done
if /i "%~1"=="--dry-run" goto arg_dry_run
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
if "%~1"=="/?" goto usage
goto bad_arg

:arg_dry_run
set "CLEAN_ARGS=--dry-run"
if not "%~2"=="" goto bad_second_arg
goto args_done

:args_done
if not exist "%VENV_PY%" goto no_venv
if not exist "%CLEAN_PY%" goto incomplete_package
"%VENV_PY%" "%CLEAN_PY%" %CLEAN_ARGS%
set "RC=%ERRORLEVEL%"
goto finish

:no_venv
echo ERROR: .venv\Scripts\python.exe was not found in this folder.
echo Run setup_windows.cmd first.
set "RC=1"
goto finish

:incomplete_package
echo ERROR: scripts\handoff\clean_outputs.py is missing - the package is incomplete.
set "RC=1"
goto finish

:bad_second_arg
for %%M in ("%~2") do echo ERROR: unknown option %%~M
set "RC=2"
goto usage_text

:bad_arg
for %%M in ("%~1") do echo ERROR: unknown option %%~M
set "RC=2"
goto usage_text

:usage
set "RC=0"
:usage_text
echo Usage: clean_demo_outputs.cmd [--dry-run]
goto finish

:finish
popd
if defined PAUSE_AT_END pause
endlocal & exit /b %RC%

:pushd_failed
echo ERROR: cannot open the folder that contains clean_demo_outputs.cmd.
if defined PAUSE_AT_END pause
endlocal & exit /b 1
