@echo off
REM ============================================================================
REM verify_installation.cmd - check an installed falls_ml handoff package (9 checks)
REM
REM Usage:  verify_installation.cmd [--skip-tests or --full-tests]
REM   (default)      run the core test suite (tests not marked slow)
REM   --skip-tests   skip the test suite - for diagnosis only
REM   --full-tests   run the full test suite including slow tests
REM
REM Run setup_windows.cmd first. Results go to demo_outputs\verify, temporary test files to
REM demo_outputs\pt and the matplotlib cache to .venv\mplconfig.
REM Synthetic data only.
REM Set FALLS_ML_NO_PAUSE=1 for automation: the window then never waits for a key press at the end
REM (by default it waits only when the file was started by double-click, so the result stays visible).
REM ============================================================================
setlocal EnableExtensions DisableDelayedExpansion
set "RC=1"
set "PAUSE_AT_END="
set "CMDLINE_CHECK=%cmdcmdline:"=%"
if /i not "%CMDLINE_CHECK:verify_installation=%"=="%CMDLINE_CHECK%" if /i not "%CMDLINE_CHECK: /c =%"=="%CMDLINE_CHECK%" set "PAUSE_AT_END=1"
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
set "VERIFY_PY=scripts\handoff\verify.py"
set "VERIFY_ARGS="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--skip-tests" goto arg_skip_tests
if /i "%~1"=="--full-tests" goto arg_full_tests
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
if "%~1"=="/?" goto usage
goto bad_arg

:arg_skip_tests
if "%VERIFY_ARGS%"=="--full-tests" goto conflicting_args
set "VERIFY_ARGS=--skip-tests"
shift
goto parse_args

:arg_full_tests
if "%VERIFY_ARGS%"=="--skip-tests" goto conflicting_args
set "VERIFY_ARGS=--full-tests"
shift
goto parse_args

:args_done
if not exist "%VENV_PY%" goto no_venv
if not exist "%VERIFY_PY%" goto incomplete_package
"%VENV_PY%" "%VERIFY_PY%" %VERIFY_ARGS%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto finish
if "%RC%"=="1" goto finish
echo.
echo ========================================
echo VERIFICATION FAILED
echo ========================================
echo Step: running scripts\handoff\verify.py
echo Reason: verify.py stopped unexpectedly with exit code %RC%
echo Recommended action: run setup_windows.cmd again, then verify_installation.cmd
set "RC=1"
goto finish

:no_venv
echo ========================================
echo VERIFICATION FAILED
echo ========================================
echo Step: checking the virtual environment
echo Reason: .venv\Scripts\python.exe was not found in this folder
echo Recommended action: run setup_windows.cmd first
set "RC=1"
goto finish

:incomplete_package
echo ========================================
echo VERIFICATION FAILED
echo ========================================
echo Step: checking the package folder
echo Reason: scripts\handoff\verify.py is missing - the package is incomplete
echo Recommended action: copy the complete falls_ml_handoff folder again and run setup_windows.cmd
set "RC=1"
goto finish

:conflicting_args
echo ERROR: --skip-tests and --full-tests cannot be combined.
set "RC=2"
goto usage_text

:bad_arg
for %%M in ("%~1") do echo ERROR: unknown option %%~M
set "RC=2"
goto usage_text

:usage
set "RC=0"
:usage_text
echo Usage: verify_installation.cmd [--skip-tests or --full-tests]
goto finish

:finish
popd
if defined PAUSE_AT_END pause
endlocal & exit /b %RC%

:pushd_failed
echo ERROR: cannot open the folder that contains verify_installation.cmd.
if defined PAUSE_AT_END pause
endlocal & exit /b 1
