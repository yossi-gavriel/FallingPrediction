# Phase 4 on the work PC – temporal validation 2026 (falls_ml 0.10.0, Phase 4 1.0.0)

Phase 4 evaluates the FROZEN Phase 3 models (developed on Index_Date 2025-01-01) on the 2026 snapshot (Index_Date 2026-01-01). It is a SEPARATE
installation: it never writes into the Phase 2 or Phase 3 installations or output folders (the Phase 3 output folder is only read and every file
is verified against its commit record). This package refuses to run Phase 2 or Phase 3 in production by design.

No 2026 outcome is used for any development decision. The four commands run in this order and each has one job:

| Step | Command | Reads outcomes? |
|---|---|---|
| A | `meuhedet-phase4-preflight` – schema / predictor / timing audit | never |
| B | `meuhedet-phase4-score` – blind scoring, everything hashed and frozen | never |
| C | `meuhedet-phase4-evaluate` – verifies the hashes, then opens the outcomes | only after the hash check |
| D | `meuhedet-phase4-status` – progress (read-only) | never |

Prerequisite: the Phase 3 run finished its LASSO stage (the folder contains `STAGE_07_COMPLETE.json`).

All commands are CMD.

## 1. Install (once, about 10-20 minutes)

1. Save `falls_ml_phase4_0.10.0_mailsafe.zip` to `%USERPROFILE%\Downloads` and extract it there. You get the NEW folder
   `%USERPROFILE%\Downloads\falls_ml_phase4_0.10.0`.
2. Restore the mail-safe files and verify every file (sha256):

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase4_0.10.0"
py -3.11 RESTORE_FILES.py.txt
```

   The last line must be `PACKAGE VERIFIED ...`.
3. Install into this folder only (its own `.venv`; the same pinned, hash-checked dependencies as Phases 2-3):

```bat
setup_windows.cmd
```

   Wait for `INSTALLATION SUCCESSFUL` (pip reuses the cache of the Phase 2 / 3 installs - identical lock file). Without internet access: run
   `prepare_offline_package.cmd --target windows-amd64-cp311` in this folder on a connected computer, copy the resulting `offline_packages`
   folder next to `setup_windows.cmd` on the work PC, and run `setup_windows.cmd --offline` (every wheel is hash-checked).
   Check the version (expected `0.10.0 3.2.0 5.0.0`):

```bat
.venv\Scripts\python.exe -c "import falls_ml, xgboost, optuna; print(falls_ml.__version__, xgboost.__version__, optuna.__version__)"
```

## 2. The paths (type these lines in EVERY new CMD window before a command)

Replace only the 2026 file name if it differs:

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase4_0.10.0"
set PYTHONUTF8=1
set INPUT_2026=%USERPROFILE%\Downloads\60k_falling_db_2026.csv
set INPUT_2025=%USERPROFILE%\Downloads\60k_falling_db.csv
set PHASE3_OUT=%USERPROFILE%\Downloads\60k_falling_db_phase3
set OUT4=%USERPROFILE%\Downloads\60k_falling_db_phase4
```

`OUT4` must be a NEW folder (not inside the Phase 2 / Phase 3 folders, not OneDrive).

## A. Preflight (mandatory; a few minutes; outcomes never loaded)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase4-preflight --input-2026 "%INPUT_2026%" --phase3-out "%PHASE3_OUT%" --out "%OUT4%"
```

Check P5: it prints the `Definition_Version` values of the 2026 extract. If the value is the one the V21 documentation states, run the preflight
again WITH that value (the same `--out` is fine until scoring), e.g. for `V21`:

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase4-preflight --input-2026 "%INPUT_2026%" --phase3-out "%PHASE3_OUT%" --out "%OUT4%" --expected-definition-version V21
```

The last line must be exactly `SAFE TO SCORE BLIND`. `STOP - TEMPORAL VALIDATION NOT DEFENSIBLE` means a frozen predictor is absent, unreadable,
semantically changed or no longer available at prediction time (or the IDs are duplicated): do NOT continue; send me `%OUT4%\preflight\PHASE4_PREFLIGHT.md`
(counts only). Nothing is "fixed" automatically.

## B. Blind score (minutes; outcomes never loaded; write-once)

Use exactly the same options as the last preflight, plus `--input-2025`:

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase4-score --input-2026 "%INPUT_2026%" --input-2025 "%INPUT_2025%" --phase3-out "%PHASE3_OUT%" --out "%OUT4%" --expected-definition-version V21
```

It prints `BLIND_SCORE_COMPLETE`, the artifact mode of every model (A = the exact persisted Phase 3 model) and the 2025 / 2026 patient overlap.
The predictions are frozen and hashed (`%OUT4%\sealed\BLIND_SCORE_COMPLETE.txt`); they can never be re-scored in this folder.

## C. Outcome evaluation (10-30 minutes)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase4-evaluate --input-2026 "%INPUT_2026%" --out "%OUT4%"
```

It first verifies every frozen hash (predictions, models, feature list, input file); any change stops it before an outcome is read. Then it checks
the 2026 outcome contract (no index-day event as outcome, window end Index_Date + 180, positives within follow-up, censoring); if that fails it
stops BEFORE any model performance and the share folder contains only the contract result. Running it again is safe (nothing is recomputed).

## D. Status (read-only, any time, any window)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase4-status --out "%OUT4%"
```

## What to send back

Only the folder `%OUT4%\share` (zip it). It exists only after the privacy, path, file-type and row-level scans passed. Never send the extracts,
`%OUT4%\sealed` (row-level predictions), `%OUT4%\frozen` (model objects), `%OUT4%\logs`, or anything from the Phase 3 folder.
