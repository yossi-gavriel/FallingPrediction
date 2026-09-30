@echo off
REM ============================================================================
REM setup_windows.cmd - one-time installer for the falls_ml handoff package
REM
REM Usage:  setup_windows.cmd [--offline or --online] [--recreate-venv] [--skip-tests]
REM   --online         install packages from the configured package index
REM                    (pip.ini, PIP_INDEX_URL, HTTPS_PROXY and HTTP_PROXY are respected)
REM   --offline        install only from offline_packages\ (see prepare_offline_package.cmd)
REM   (neither)        offline if offline_packages\OFFLINE_MANIFEST.json exists, else online
REM   --recreate-venv  delete the .venv folder (and nothing else) and create it again
REM   --skip-tests     skip the test suite - for diagnosis only
REM
REM Needs Windows 10/11 x64 and Python 3.11 or 3.13, 64-bit. No administrator rights.
REM Everything is installed inside this folder (.venv). Logs go to setup_logs\.
REM No database, no real patient data, no credentials: synthetic data only.
REM Set FALLS_ML_NO_PAUSE=1 for automation: the window then never waits for a key press at the end
REM (by default it waits only when the file was started by double-click, so the result stays visible).
REM ============================================================================
setlocal EnableExtensions DisableDelayedExpansion
set "RC=1"
set "PAUSE_AT_END="
set "CMDLINE_CHECK=%cmdcmdline:"=%"
if /i not "%CMDLINE_CHECK:setup_windows=%"=="%CMDLINE_CHECK%" if /i not "%CMDLINE_CHECK: /c =%"=="%CMDLINE_CHECK%" set "PAUSE_AT_END=1"
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
set "VENV_DIR=.venv"
set "VENV_PY=.venv\Scripts\python.exe"
set "LOG_DIR=setup_logs"
set "CHECK_PY=scripts\handoff\check_python.py"
set "STEPS_PY=scripts\handoff\setup_steps.py"
set "MODE_ARG="
set "RECREATE_VENV="
set "SKIP_TESTS_ARG="
set "FAIL_STEP="
set "FAIL_REASON="
set "FAIL_ACTION="
set "FAIL_LOG="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--offline" goto arg_offline
if /i "%~1"=="--online" goto arg_online
if /i "%~1"=="--recreate-venv" goto arg_recreate
if /i "%~1"=="--skip-tests" goto arg_skip_tests
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
if "%~1"=="/?" goto usage
goto bad_arg

:arg_offline
if "%MODE_ARG%"=="--mode online" goto conflicting_modes
set "MODE_ARG=--mode offline"
shift
goto parse_args

:arg_online
if "%MODE_ARG%"=="--mode offline" goto conflicting_modes
set "MODE_ARG=--mode online"
shift
goto parse_args

:arg_recreate
set "RECREATE_VENV=1"
shift
goto parse_args

:arg_skip_tests
set "SKIP_TESTS_ARG=--skip-tests"
shift
goto parse_args

:args_done
echo ========================================
echo falls_ml handoff - Windows setup
echo ========================================
for %%M in ("%CD%") do echo Project folder: %%~M
echo.
if not exist "%CHECK_PY%" goto incomplete_package
if not exist "%STEPS_PY%" goto incomplete_package
if not exist "pyproject.toml" goto incomplete_package
if not exist "%LOG_DIR%\" mkdir "%LOG_DIR%" >nul 2>&1
if not exist "%LOG_DIR%\" goto cannot_write
echo write test>"%LOG_DIR%\write_test.tmp" 2>nul
if not exist "%LOG_DIR%\write_test.tmp" goto cannot_write
del /q "%LOG_DIR%\write_test.tmp" >nul 2>&1

REM ---------------------------------------------------------------------------- [1/10]
echo [1/10] Checking Python...
set "PY_EXE="
set "PY_CMD="
set "PY_VER=3.13"
set "PY_REJECTED="
set "PY_STORE_STUB="
set "PY_LAUNCHER_NO_313="
set "PY_LOG=%LOG_DIR%\01_python.log"
set "PROBE_OUT=%LOG_DIR%\01_python_probe.txt"
set "PROBE_EXE=%LOG_DIR%\01_python_exe.txt"
set "PROBE_LINE=%LOG_DIR%\01_python_line.txt"
set "PROBE_REASON=%LOG_DIR%\01_python_reason.txt"
echo Python detection log>"%PY_LOG%"
call :probe_python py -3.13
if defined PY_EXE goto python_found
call :probe_python py -3.11
if defined PY_EXE goto python_found
call :probe_python python
if defined PY_EXE goto python_found
call :probe_python python3
if defined PY_EXE goto python_found
goto python_not_found

:python_found
set PY_LAUNCH="%PY_EXE%"
if exist "%PY_EXE%" goto python_show
REM the path could not be passed through a file (unusual characters): use the command that worked
set "PY_LAUNCH=%PY_CMD%"
:python_show
for /f "usebackq tokens=3 delims=|" %%A in ("%PROBE_LINE%") do set "PY_VER=%%A"
for %%M in ("%PY_EXE%") do echo [OK] Python %PY_VER% (%%~M)
echo.

REM ---------------------------------------------------------------------------- [2/10]
echo [2/10] Creating virtual environment...
set "VENV_LOG=%LOG_DIR%\02_venv.log"
echo Virtual environment log>"%VENV_LOG%"
set "VENV_STATE=reused"
if not defined RECREATE_VENV goto venv_check
if not exist "%VENV_DIR%\" goto venv_check
echo   Removing the existing .venv folder because --recreate-venv was given...
rmdir /s /q "%VENV_DIR%" >>"%VENV_LOG%" 2>&1
if exist "%VENV_DIR%\" goto venv_remove_failed

:venv_check
if exist "%VENV_PY%" goto venv_validate
if exist "%VENV_DIR%\" goto venv_broken
echo   Creating .venv with Python %PY_VER%...
call %PY_LAUNCH% -m venv "%VENV_DIR%" <nul >>"%VENV_LOG%" 2>&1
set "STEP_RC=%ERRORLEVEL%"
if not "%STEP_RC%"=="0" goto venv_create_failed
if not exist "%VENV_PY%" goto venv_create_failed
set "VENV_STATE=created"

:venv_validate
"%VENV_PY%" "%CHECK_PY%" <nul >>"%VENV_LOG%" 2>&1
set "STEP_RC=%ERRORLEVEL%"
if not "%STEP_RC%"=="0" goto venv_broken
if "%VENV_STATE%"=="created" goto venv_created_message
echo [OK] Reusing the existing virtual environment .venv - use --recreate-venv to rebuild it
echo.
goto step3
:venv_created_message
echo [OK] Created the virtual environment .venv
echo.

REM ---------------------------------------------------------------------------- [3/10]
:step3
REM matplotlib cache inside this folder (never in the user profile); .venv exists from here on
if not exist "%VENV_DIR%\mplconfig\" mkdir "%VENV_DIR%\mplconfig" >nul 2>&1
if exist "%VENV_DIR%\mplconfig\" set "MPLCONFIGDIR=%CD%\%VENV_DIR%\mplconfig"
echo [3/10] Activating virtual environment...
set "ACT_LOG=%LOG_DIR%\03_activate.log"
echo Activation log>"%ACT_LOG%"
if not exist "%VENV_DIR%\Scripts\activate.bat" goto activate_failed
call "%VENV_DIR%\Scripts\activate.bat" >>"%ACT_LOG%" 2>&1
if not defined VIRTUAL_ENV goto activate_failed
python -c "import os, sys; a = os.path.normcase(os.path.realpath(sys.executable)); b = os.path.normcase(os.path.realpath(r'.venv\Scripts\python.exe')); print('active python:', a); print('venv python:  ', b); sys.exit(0 if a == b else 1)" <nul >>"%ACT_LOG%" 2>&1
set "STEP_RC=%ERRORLEVEL%"
if not "%STEP_RC%"=="0" goto activate_mismatch
echo [OK] Virtual environment active
echo.

REM ---------------------------------------------------------------------------- [4/10] .. [10/10]
"%VENV_PY%" "%STEPS_PY%" --first-step 4 --total-steps 10 %MODE_ARG% %SKIP_TESTS_ARG%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto finish
if "%RC%"=="1" goto finish
set "FAIL_STEP=[4/10] to [10/10] installer steps"
set "FAIL_REASON=scripts\handoff\setup_steps.py stopped unexpectedly with exit code %RC%"
set "FAIL_ACTION=Run setup_windows.cmd again. If it stops the same way, send the setup_logs folder to the maintainer."
set "FAIL_LOG=%CD%\%LOG_DIR%"
goto fail_block

REM ============================================================================ subroutine
:probe_python
REM Runs check_python.py with the command given as arguments; sets PY_EXE and PY_CMD when supported.
REM "call" also works when the command is a .bat shim (for example pyenv-win) instead of an .exe.
set "PROBE_CMD=%*"
if exist "%PROBE_OUT%" del /q "%PROBE_OUT%" >nul 2>&1
if exist "%PROBE_EXE%" del /q "%PROBE_EXE%" >nul 2>&1
if exist "%PROBE_LINE%" del /q "%PROBE_LINE%" >nul 2>&1
call %PROBE_CMD% "%CHECK_PY%" --write-exe "%PROBE_EXE%" <nul >"%PROBE_OUT%" 2>&1
set "PROBE_RC=%ERRORLEVEL%"
>>"%PY_LOG%" echo ---- candidate: %PROBE_CMD% / exit code %PROBE_RC%
if exist "%PROBE_OUT%" type "%PROBE_OUT%" >>"%PY_LOG%" 2>&1
if not exist "%PROBE_OUT%" goto :eof
findstr /i /l /c:"Microsoft Store" "%PROBE_OUT%" >nul 2>&1
if not errorlevel 1 set "PY_STORE_STUB=1"
findstr /i /l /c:"No suitable Python runtime" "%PROBE_OUT%" >nul 2>&1
if not errorlevel 1 set "PY_LAUNCHER_NO_313=1"
if "%PROBE_RC%"=="0" goto probe_ok
if "%PROBE_RC%"=="3" goto probe_unsupported
goto :eof

:probe_ok
findstr /b /l /c:"PYTHON|ok|" "%PROBE_OUT%" >"%PROBE_LINE%" 2>nul
if errorlevel 1 goto :eof
if not exist "%PROBE_EXE%" goto :eof
set /p PY_EXE=<"%PROBE_EXE%"
set "PY_CMD=%PROBE_CMD%"
goto :eof

:probe_unsupported
set "REJ_REASON=REASON|unsupported interpreter"
findstr /b /l /c:"REASON|" "%PROBE_OUT%" >"%PROBE_REASON%" 2>nul
set /p REJ_REASON=<"%PROBE_REASON%"
set "REJ_REASON=%REJ_REASON:REASON|=%"
set "PY_REJECTED=%PY_REJECTED% [%PROBE_CMD%: %REJ_REASON%]"
goto :eof

REM ============================================================================ failures
:python_not_found
set "FAIL_STEP=[1/10] Checking Python"
set "FAIL_REASON=Python 3.11 or 3.13 (64-bit) was not found. Tried: py -3.13, py -3.11, python, python3."
if defined PY_REJECTED set "FAIL_REASON=%FAIL_REASON% Found but not usable:%PY_REJECTED%."
if defined PY_LAUNCHER_NO_313 set "FAIL_REASON=%FAIL_REASON% The py launcher has neither 3.13 nor 3.11 registered."
if defined PY_STORE_STUB set "FAIL_REASON=%FAIL_REASON% The python command is only the Microsoft Store placeholder."
set "FAIL_ACTION=Install Python 3.13 or 3.11, 64-bit - Windows installer 64-bit - from https://www.python.org/downloads/windows/ for your user only; no administrator rights are needed. Tick Add python.exe to PATH, open a NEW Command Prompt window and run setup_windows.cmd again."
set "FAIL_LOG=%CD%\%PY_LOG%"
goto fail_block

:venv_remove_failed
set "FAIL_STEP=[2/10] Creating virtual environment"
set "FAIL_REASON=the existing .venv folder could not be deleted - files are in use or read-only."
set "FAIL_ACTION=Close editors, terminals and Python programs that use .venv, then run setup_windows.cmd --recreate-venv again."
set "FAIL_LOG=%CD%\%VENV_LOG%"
goto fail_block

:venv_broken
set "FAIL_STEP=[2/10] Creating virtual environment"
set "FAIL_REASON=.venv exists but its Python does not start or is not Python 3.11/3.13 64-bit - it was created on another computer, moved, or Python was reinstalled."
set "FAIL_ACTION=Run setup_windows.cmd --recreate-venv - it deletes only the .venv folder and builds it again."
set "FAIL_LOG=%CD%\%VENV_LOG%"
goto fail_block

:venv_create_failed
set "FAIL_STEP=[2/10] Creating virtual environment"
set "FAIL_REASON=python -m venv failed with exit code %STEP_RC%."
set "FAIL_ACTION=Read the log file. If it mentions ensurepip, repair the Python installation with the pip feature enabled - Apps and Features, Modify - then run setup_windows.cmd --recreate-venv."
set "FAIL_LOG=%CD%\%VENV_LOG%"
goto fail_block

:activate_failed
set "FAIL_STEP=[3/10] Activating virtual environment"
set "FAIL_REASON=.venv\Scripts\activate.bat is missing or did not activate the environment."
set "FAIL_ACTION=Run setup_windows.cmd --recreate-venv."
set "FAIL_LOG=%CD%\%ACT_LOG%"
goto fail_block

:activate_mismatch
set "FAIL_STEP=[3/10] Activating virtual environment"
set "FAIL_REASON=after activation the python command is not .venv\Scripts\python.exe - the folder was probably moved or renamed after .venv was created."
set "FAIL_ACTION=Run setup_windows.cmd --recreate-venv."
set "FAIL_LOG=%CD%\%ACT_LOG%"
goto fail_block

:incomplete_package
set "FAIL_STEP=Checking the package folder"
set "FAIL_REASON=scripts\handoff\check_python.py, scripts\handoff\setup_steps.py or pyproject.toml is missing - the package is incomplete."
set "FAIL_ACTION=Copy the complete falls_ml_handoff folder again and run setup_windows.cmd from inside it."
goto fail_block

:cannot_write
set "FAIL_STEP=Checking the package folder"
set "FAIL_REASON=cannot create or write the setup_logs folder here - the folder is read-only or protected."
set "FAIL_ACTION=Copy the package to a folder you can write to, for example C:\Projects\falls_ml_handoff - not C:\Program Files - and run setup_windows.cmd there."
goto fail_block

:conflicting_modes
echo ERROR: --offline and --online cannot be combined.
set "RC=2"
goto usage_text

:bad_arg
for %%M in ("%~1") do echo ERROR: unknown option %%~M
set "RC=2"
goto usage_text

:usage
set "RC=0"
:usage_text
echo Usage: setup_windows.cmd [--offline or --online] [--recreate-venv] [--skip-tests]
echo   --online         install from the configured package index - PIP_INDEX_URL and proxies are respected
echo   --offline        install only from offline_packages - created with prepare_offline_package.cmd
echo   --recreate-venv  delete .venv and create it again
echo   --skip-tests     skip the test suite - diagnosis only
goto finish

:fail_block
echo.
if defined FAIL_LOG if exist "%FAIL_LOG%" if not exist "%FAIL_LOG%\" echo Details from the log file:
if defined FAIL_LOG if exist "%FAIL_LOG%" if not exist "%FAIL_LOG%\" type "%FAIL_LOG%"
echo.
echo ========================================
echo INSTALLATION FAILED
echo ========================================
for %%M in ("%FAIL_STEP%") do echo Step: %%~M
for %%M in ("%FAIL_REASON%") do echo Reason: %%~M
for %%M in ("%FAIL_ACTION%") do echo Recommended action: %%~M
if defined FAIL_LOG for %%M in ("%FAIL_LOG%") do echo Log file: %%~M
set "RC=1"
goto finish

:finish
popd
if defined PAUSE_AT_END pause
endlocal & exit /b %RC%

:pushd_failed
echo ERROR: cannot open the folder that contains setup_windows.cmd.
if defined PAUSE_AT_END pause
endlocal & exit /b 1
