@echo off
REM ============================================================================
REM prepare_offline_package.cmd - download the locked wheels into offline_packages\
REM
REM Usage:  prepare_offline_package.cmd [--target windows-amd64-cp313 or windows-amd64-cp311 or current] [--force]
REM   --target windows-amd64-cp313   wheels for Windows x64 + Python 3.13 (default; works from any computer)
REM   --target windows-amd64-cp311   wheels for Windows x64 + Python 3.11 (work computers running 3.11)
REM   --target current               wheels for this computer's own platform (needs Python 3.13 64-bit)
REM   --force                        replace a bundle previously created by this command
REM   --verify-only                  only check an existing offline_packages folder
REM
REM Run it on a computer WITH internet access (pip.ini, PIP_INDEX_URL and HTTPS_PROXY are respected).
REM Then copy offline_packages\ next to setup_windows.cmd on the target computer and run
REM setup_windows.cmd --offline there. Uses .venv\Scripts\python.exe when present, otherwise
REM py -3.13, python or python3.
REM Set FALLS_ML_NO_PAUSE=1 for automation: the window then never waits for a key press at the end
REM (by default it waits only when the file was started by double-click, so the result stays visible).
REM ============================================================================
setlocal EnableExtensions DisableDelayedExpansion
set "RC=1"
set "PAUSE_AT_END="
set "CMDLINE_CHECK=%cmdcmdline:"=%"
if /i not "%CMDLINE_CHECK:prepare_offline_package=%"=="%CMDLINE_CHECK%" if /i not "%CMDLINE_CHECK: /c =%"=="%CMDLINE_CHECK%" set "PAUSE_AT_END=1"
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
set "CHECK_PY=scripts\handoff\check_python.py"
set "OFFLINE_PY=scripts\handoff\prepare_offline.py"
set "LOG_DIR=setup_logs"

if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
if "%~1"=="/?" goto usage
if not exist "%OFFLINE_PY%" goto incomplete_package
if not exist "%CHECK_PY%" goto incomplete_package
if not exist "%LOG_DIR%\" mkdir "%LOG_DIR%" >nul 2>&1
if not exist "%LOG_DIR%\" goto cannot_write

echo ========================================
echo falls_ml handoff - prepare offline package
echo ========================================
set "PY_EXE="
set "PY_CMD="
set "PY_FALLBACK="
set "PY_LOG=%LOG_DIR%\offline_python.log"
set "PROBE_OUT=%LOG_DIR%\offline_python_probe.txt"
set "PROBE_EXE=%LOG_DIR%\offline_python_exe.txt"
echo Python detection log>"%PY_LOG%"
if not exist "%VENV_PY%" goto probe_system
call :probe_python "%VENV_PY%"
if defined PY_CMD goto python_found
:probe_system
call :probe_python py -3.13
if defined PY_CMD goto python_found
call :probe_python python
if defined PY_CMD goto python_found
call :probe_python python3
if defined PY_CMD goto python_found
if defined PY_FALLBACK goto python_fallback
goto python_not_found

:python_fallback
set "PY_CMD=%PY_FALLBACK%"
echo NOTE: Python 3.13 64-bit was not found; using another Python 3 installation.
echo       This is enough for the windows-amd64-cp313/cp311 targets; --target current needs Python 3.11 or 3.13.
goto run_prepare

:python_found
for %%M in ("%PY_EXE%") do echo Using Python: %%~M

:run_prepare
echo.
call %PY_CMD% "%OFFLINE_PY%" %*
set "RC=%ERRORLEVEL%"
goto finish

REM ============================================================================ subroutine
:probe_python
REM Runs check_python.py with the command given as arguments. Sets PY_CMD when it is Python 3.13
REM 64-bit, or PY_FALLBACK when it is another Python that at least runs.
set "PROBE_CMD=%*"
if exist "%PROBE_OUT%" del /q "%PROBE_OUT%" >nul 2>&1
if exist "%PROBE_EXE%" del /q "%PROBE_EXE%" >nul 2>&1
call %PROBE_CMD% "%CHECK_PY%" --write-exe "%PROBE_EXE%" <nul >"%PROBE_OUT%" 2>&1
set "PROBE_RC=%ERRORLEVEL%"
>>"%PY_LOG%" echo ---- candidate: %PROBE_CMD% / exit code %PROBE_RC%
if exist "%PROBE_OUT%" type "%PROBE_OUT%" >>"%PY_LOG%" 2>&1
if "%PROBE_RC%"=="3" goto probe_fallback
if not "%PROBE_RC%"=="0" goto :eof
if not exist "%PROBE_EXE%" goto :eof
set /p PY_EXE=<"%PROBE_EXE%"
set "PY_CMD=%PROBE_CMD%"
goto :eof

:probe_fallback
if not defined PY_FALLBACK set "PY_FALLBACK=%PROBE_CMD%"
goto :eof

REM ============================================================================ messages
:python_not_found
echo.
echo ========================================
echo OFFLINE PACKAGE FAILED
echo ========================================
echo Step: finding Python
echo Reason: no Python was found - tried .venv, py -3.13, python and python3
echo Recommended action: install Python 3.13 64-bit from https://www.python.org/downloads/windows/ and run this command again
for %%M in ("%CD%\%PY_LOG%") do echo Log file: %%~M
set "RC=1"
goto finish

:incomplete_package
echo ERROR: scripts\handoff\prepare_offline.py or check_python.py is missing - the package is incomplete.
set "RC=1"
goto finish

:cannot_write
echo ERROR: cannot create the setup_logs folder here - copy the package to a writable folder first.
set "RC=1"
goto finish

:usage
echo Usage: prepare_offline_package.cmd [--target windows-amd64-cp313 or windows-amd64-cp311 or current] [--force] [--verify-only]
set "RC=0"
goto finish

:finish
popd
if defined PAUSE_AT_END pause
endlocal & exit /b %RC%

:pushd_failed
echo ERROR: cannot open the folder that contains prepare_offline_package.cmd.
if defined PAUSE_AT_END pause
endlocal & exit /b 1
