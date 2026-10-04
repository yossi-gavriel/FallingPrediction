# Phase 5 on the work PC – 2026 redevelopment + incremental value of the new V21 predictors (falls_ml 0.11.0, Phase 5 1.0.0)

Phase 5 answers one question on the 2026 extract (Index_Date 2026-01-01): **at >= 70% fall capture, do the new V21 predictors lower the
false-alert burden compared with the Phase 3 feature universe (OLD)?** It is INTERNAL NESTED CROSS-VALIDATION ON THE 2026 SNAPSHOT (development
data), never external validation. Phase 4 (temporal validation) stays frozen and unused; this package does not touch Phase 2 / 3 / 4 installations
or output folders (keep the Phase 4 package `falls_ml_phase4_0.10.0` for any future Phase 4 run).

What the overnight command does, in order (every step is checkpointed; Ctrl+C / a power cut leaves a resumable folder):

| Step | What | Output |
|---|---|---|
| PREFLIGHT | schema, outcome / future columns sealed from X, cohort (one row per patient), V21 definition, the 2026 outcome contract, feature eligibility (timing / provenance / semantics / coverage / leakage), feature sets, the fixed folds | `preflight\` (aggregate), last line `SAFE TO MODEL` or `STOP - 2026 REDEVELOPMENT NOT DEFENSIBLE` |
| PRIMARY | nested CV (5 outer x 5 inner folds), LASSO / elastic net / XGBoost x {OLD, OLD+NEW_SAFE, the 2 sensitivity sets} | an INTERIM `share\` with the main answer as soon as it is done |
| FINAL, DOMAIN, ABLATION, EXPLAIN, STABILITY | final development models, OLD + each new domain, ablations, permutation importance / SHAP, bootstrap stability | – |
| REPORT | management + scientific summaries (Hebrew / English), tables, 12 figures, privacy scan | `share\` (aggregate only) |

All commands are CMD. The real data never leave the work PC; only `share\` (and, after a preflight stop, `preflight\`) may be sent back.

## 1. Restore and set up (once, about 10-20 minutes)

1. Save `falls_ml_phase5_0.11.0_mailsafe.zip` to `%USERPROFILE%\Downloads` and extract it there. You get the NEW folder
   `%USERPROFILE%\Downloads\falls_ml_phase5_0.11.0`.
2. Restore the mail-safe files and verify every file (sha256):

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.11.0"
py -3.11 RESTORE_FILES.py.txt
```

   The last line must be `PACKAGE VERIFIED ...`.
3. Install into this folder only (its own `.venv`; the same pinned, hash-checked dependencies as Phases 2-4):

```bat
setup_windows.cmd
```

   Wait for `INSTALLATION SUCCESSFUL`. Without internet access: run `prepare_offline_package.cmd --target windows-amd64-cp311` in this folder on a
   connected computer, copy the resulting `offline_packages` folder next to `setup_windows.cmd`, and run `setup_windows.cmd --offline`.
   Check the version (expected `0.11.0 3.2.0 5.0.0`):

```bat
.venv\Scripts\python.exe -c "import falls_ml, xgboost, optuna; print(falls_ml.__version__, xgboost.__version__, optuna.__version__)"
```

## 2. The paths (type these lines in EVERY new CMD window before a command)

Replace only the 2026 file name if it differs:

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.11.0"
set PYTHONUTF8=1
set INPUT_2026=%USERPROFILE%\Downloads\60k_falling_db_2026.csv
set OUT5=%USERPROFILE%\Downloads\60k_falling_db_phase5
```

`OUT5` must be a NEW local folder (not a Phase 2 / 3 / 4 folder, not OneDrive, not the folder that holds the CSV).

## 3. Synthetic smoke run (optional, 5-15 minutes; no real data)

Proves the installation end to end on a generated SYNTHETIC extract (results are marked SYNTHETIC and mean nothing):

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5-synthetic --out "%USERPROFILE%\Downloads\phase5_synthetic_smoke"
```

It ends with `"status": "COMPLETE"`.

## 4. Preflight only (mandatory first; a few minutes)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT5%" --preflight-only
```

The last line must be `SAFE TO MODEL`. Then read `%OUT5%\preflight\PHASE5_PREFLIGHT.md` and `%OUT5%\preflight\NEW_FEATURE_CATALOGUE.csv`:
they list which V21 columns became eligible NEW predictors, which were excluded and why, and `V21_UNDECLARED_COLUMNS.csv` lists 2026 columns that
are not in the pre-declared V21 catalogue (names and aggregate profile only - never used in this run). If important V21 columns appear there,
send me the `preflight` folder before the overnight run: the catalogue is extended only BEFORE any model is fitted.
`STOP - 2026 REDEVELOPMENT NOT DEFENSIBLE` (e.g. the outcome contract fails, duplicated patients, wrong Definition_Version, the VIEW's own
leakage flag): do NOT continue; send me `%OUT5%\preflight` (counts only).

## 5. Runtime estimate on the real file (seconds to a minute; no model is fitted on the real data)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT5%" --estimate --mode overnight
```

It reads only the file's shape (and the preflight plan), times the solvers on random matrices of the same size and prints the expected hours per
stage and the hour at which the first (interim) answer appears. `--jobs N` changes the number of parallel workers (default about 60% of the logical
cores; the machine stays usable).

## 6. The overnight run (start in the evening; one command)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT5%" --mode overnight --device auto --resume
```

* Leave the window open; Windows sleep must be off for the night (Settings > System > Power > Sleep: Never, while plugged in).
* `--device auto` uses an NVIDIA GPU for XGBoost only if the installed XGBoost build supports CUDA and a small test training succeeds; otherwise CPU
  (the locked Windows build is `xgboost-cpu`, so CPU is expected). The device used is printed and recorded; a GPU error switches to CPU.
* Progress: `%OUT5%\OVERNIGHT_PROGRESS.log` (one line per finished unit with the ETA), `%OUT5%\RUN_STATUS.json` (heartbeat every 30 s),
  `%OUT5%\RUN_TIMINGS.csv`.
* As soon as the PRIMARY stage is done, `%OUT5%\share` holds an INTERIM report with the main OLD vs OLD+NEW answer; it is replaced at the end.

## 7. Resume after an interruption (Ctrl+C, reboot, power cut, a closed window)

Exactly the same command (it always carries `--resume`):

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT5%" --mode overnight --device auto --resume
```

It verifies that the input file, the settings, the mode and the code are those of the plan and continues with the first unfinished unit; nothing
finished is recomputed (XGBoost searches continue from the last finished trial).

## 8. Status (read-only, any time, from a second CMD window)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --out "%OUT5%" --status
```

Shows the stage, finished / total units per stage, outer folds complete, the current XGBoost trial, elapsed time, the estimated remaining time,
the last checkpoint, failures and retries. "no heartbeat for more than 3 minutes" while RUNNING means the process stopped: run step 7.

## 9. Rebuild the aggregate report only (no fitting)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT5%" --mode overnight --report-only
```

## What you get in the morning – `%OUT5%\share` (aggregate only)

`MANAGEMENT_SUMMARY_HE.md` (first page: patients, falls, OLD vs OLD+NEW at the 70% target, false alerts saved per 10,000 patients, YES / NO /
UNCERTAIN, the domains that helped most, limitations), `SCIENTIFIC_SUMMARY_HE.md`, `SCIENTIFIC_SUMMARY.md`, `PRIMARY_70_SENSITIVITY_COMPARISON.csv`,
`THRESHOLD_TRADEOFF.csv`, `SENSITIVITY_TARGET_TABLE.csv` (50 / 60 / 70 / 75 / 80 / 90%), `MODEL_COMPARISON.csv`, `DOMAIN_INCREMENTAL_VALUE.csv`,
`ABLATION_RESULTS.csv`, `OOF_MODEL_COMPARISON.csv`, `OOF_PREDICTION_SUMMARY.csv`, `CALIBRATION.csv`, `SUBGROUP_SUMMARY.csv`,
`FEATURE_ELIGIBILITY.csv`, `NEW_FEATURE_CATALOGUE.csv`, `FEATURE_STABILITY.csv`, `PERMUTATION_IMPORTANCE.csv`, `SHAP_SUMMARY.csv`,
`COHORT_FACTS_2026.json`, `OUTCOME_CONTRACT_2026.json`, `RUN_MANIFEST.json`, `RUN_TIMINGS.csv`, `ENVIRONMENT.json`, `PRIVACY_SCAN.json`,
`figures\` (12 figures).

Send back only `%OUT5%\share` (zip it). It exists only after the identifier, path, file-type and row-level scans passed. Never send the
extract, `%OUT5%\work` (row-level frame, folds, out-of-fold predictions, model objects) or `%OUT5%\logs`.
