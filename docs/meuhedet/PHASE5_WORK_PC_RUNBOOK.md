# Phase 5 on the work PC – 2026 redevelopment + incremental value of the new V21 information (falls_ml 0.12.0, Phase 5 2.0.0)

Phase 5 answers one question on the 2026 extract (Index_Date 2026-01-01, prediction at the END of the index day): **at approximately the same
>= 70% fall sensitivity, does adding the new V21 information reduce the number and the percentage of false alerts?** The pre-declared primary
comparison is **Elastic Net OLD vs Elastic Net OLD_PLUS_ALL_NEW_ELIGIBLE**; LASSO and XGBoost are secondary, and OLD_PLUS_NEW_SAFE is the stricter
sensitivity analysis. It is INTERNAL NESTED CROSS-VALIDATION ON THE 2026 SNAPSHOT (development data), never external validation. Phase 4 (temporal
validation) stays frozen and unused; this package does not touch Phase 2 / 3 / 4 installations or output folders (keep the Phase 4 package
`falls_ml_phase4_0.10.0` for any future Phase 4 run).

The 2026 schema is the AUTHORITATIVE V21 VIEW definition you supplied (224 columns), embedded in the package
(`configs\meuhedet\phase5_v21_view_definition.txt`, with its column-by-column review in `configs\meuhedet\phase5_v21_schema.yaml`). The preflight
recomputes the exact V1 -> V21 diff from the header of YOUR file and classifies every column; a column the schema does not define stops the run
for review (no clinical column is ignored silently).

| Step | What | Output |
|---|---|---|
| PREFLIGHT (run alone first) | exact V1 -> V21 schema diff, outcome / future / identifier columns sealed from X, cohort (one row per patient), the 2026 outcome contract, feature eligibility (timing / provenance / semantics / coverage / leakage), the three feature sets, the fixed folds | `preflight\` (aggregate); last line `SAFE TO MODEL` or `STOP - REVIEW REQUIRED`; nothing is fitted |
| PRIMARY | nested CV (5 outer x 5 inner folds), Elastic Net (primary) / LASSO / XGBoost x {OLD, OLD_PLUS_ALL_NEW_ELIGIBLE, OLD_PLUS_NEW_SAFE} | an INTERIM `share\` with the main answer as soon as it is done |
| FINAL, DOMAIN, ABLATION, EXPLAIN, STABILITY | final development models, OLD + each new domain (all families), ablations of the primary Elastic Net model, permutation importance / SHAP, coefficient stability | – |
| REPORT | management + scientific summaries (Hebrew / English), tables, 12 figures, privacy scan | `share\` (aggregate only) |

All commands are CMD. The real data never leave the work PC; only `share\` (and, after a preflight, `preflight\`) may be sent back.

## 1. Restore and set up (once, about 10-20 minutes)

1. Save `falls_ml_phase5_0.12.0_mailsafe.zip` to `%USERPROFILE%\Downloads` and extract it there. You get the NEW folder
   `%USERPROFILE%\Downloads\falls_ml_phase5_0.12.0` (the earlier `falls_ml_phase5_0.11.0` folder is not used any more; do not mix them).
2. Restore the mail-safe files and verify every file (sha256):

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.12.0"
py -3.11 RESTORE_FILES.py.txt
```

   The last line must be `PACKAGE VERIFIED ...`.
3. Install into this folder only (its own `.venv`; the same pinned, hash-checked dependencies as Phases 2-4):

```bat
setup_windows.cmd
```

   Wait for `INSTALLATION SUCCESSFUL`. Without internet access: run `prepare_offline_package.cmd --target windows-amd64-cp311` in this folder on a
   connected computer, copy the resulting `offline_packages` folder next to `setup_windows.cmd`, and run `setup_windows.cmd --offline`.
   Check the version (expected `0.12.0 3.2.0 5.0.0`):

```bat
.venv\Scripts\python.exe -c "import falls_ml, xgboost, optuna; print(falls_ml.__version__, xgboost.__version__, optuna.__version__)"
```

## 2. The paths (type these lines in EVERY new CMD window before a command)

Replace only the 2026 file name if it differs:

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.12.0"
set PYTHONUTF8=1
set INPUT_2026=%USERPROFILE%\Downloads\60k_falling_db_2026.csv
set OUT5=%USERPROFILE%\Downloads\60k_falling_db_phase5_v2
```

`OUT5` must be a NEW local folder (not a Phase 2 / 3 / 4 folder, not an earlier Phase 5 folder, not OneDrive, not the folder that holds the CSV).

## 3. Synthetic smoke run (optional, 5-15 minutes; no real data)

Proves the installation end to end on a generated SYNTHETIC extract with the exact V21 header (results are marked SYNTHETIC and mean nothing):

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5-synthetic --out "%USERPROFILE%\Downloads\phase5_synthetic_smoke"
```

It ends with `"status": "COMPLETE"`.

## 4. Preflight ONLY (mandatory first on the real file; a few minutes; fits nothing)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT5%" --preflight-only
```

The overnight command refuses a folder without this step (`PREFLIGHT_REQUIRED`). Read `%OUT5%\preflight\PHASE5_PREFLIGHT.md`: its first table
gives the V1 / V21 column totals, unchanged / removed / new / changed or renamed columns, the eligible new predictors (OLD_PLUS_ALL_NEW_ELIGIBLE /
OLD_PLUS_NEW_SAFE), the excluded new predictors and the unresolved fields. Details: `SCHEMA_DIFF_V1_V21.csv`, `ALL_V21_COLUMN_CLASSIFICATION.csv`
(every column, its class and how it is used or why not), `REMOVED_V1_COLUMNS.csv`, `RENAMED_OR_CHANGED_COLUMNS.csv`, `NEW_FEATURE_CATALOGUE.csv`,
`FEATURE_ELIGIBILITY.csv`, `V21_UNDECLARED_COLUMNS.csv`, `OUTCOME_CONTRACT_2026.json`, `COHORT_FACTS_2026.json`.

* Last line `SAFE TO MODEL`: continue with steps 5-6.
* Last line `STOP - REVIEW REQUIRED`: do NOT continue; send me `%OUT5%\preflight` (aggregate counts and column names only). Typical causes: a
  column of the file that the authoritative V21 schema does not define (`REQUIRES_SEMANTIC_REVIEW`, listed in `V21_UNDECLARED_COLUMNS.csv`), the
  outcome contract fails, duplicated patients at the index date, or the VIEW's own leakage flag is set.

## 5. Runtime estimate on the real file (seconds to a minute; no model is fitted on the real data)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT5%" --estimate --mode overnight
```

It reads the preflight plan (or only the file's shape), times the solvers on random matrices of the same size and prints the expected hours per
stage and the hour at which the first (interim) answer appears. `--jobs N` changes the number of parallel workers (default about 60% of the
logical cores; the machine stays usable).

## 6. The overnight run (start in the evening; one command)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT5%" --mode overnight --device auto --resume
```

* Use the same `--input` and `--out` as the preflight (the run verifies the file's sha256 against the preflight plan).
* Leave the window open; Windows sleep must be off for the night (Settings > System > Power > Sleep: Never, while plugged in).
* `--device auto` uses an NVIDIA GPU for XGBoost only if the installed XGBoost build supports CUDA and a small test training succeeds; otherwise CPU
  (the locked Windows build is `xgboost-cpu`, so CPU is expected). The device used is printed and recorded; a GPU error switches to CPU.
* Progress: `%OUT5%\OVERNIGHT_PROGRESS.log` (one line per finished unit with the ETA), `%OUT5%\RUN_STATUS.json` (heartbeat every 30 s),
  `%OUT5%\RUN_TIMINGS.csv`.
* As soon as the PRIMARY stage is done, `%OUT5%\share` holds an INTERIM report with the main OLD vs OLD_PLUS_ALL_NEW_ELIGIBLE answer; it is
  replaced at the end.

## 7. Resume after an interruption (Ctrl+C, reboot, power cut, a closed window)

Exactly the same command (it always carries `--resume`):

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT5%" --mode overnight --device auto --resume
```

It verifies that the input file, the settings, the V21 schema, the mode and the code are those of the plan and continues with the first
unfinished unit; nothing finished is recomputed (XGBoost searches continue from the last finished trial).

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

`MANAGEMENT_SUMMARY_HE.md` (first page: patients, falls, genuine new V21 predictors found / in ALL_NEW / in NEW_SAFE and what was excluded and
why; OLD and OLD + ALL NEW at ~70% sensitivity - alerts, captured falls, false alerts, false-alert percentage, PPV; the difference - false alerts
and interventions avoided, PPV and false-alert-share change, falls gained / lost; and the answer YES / NO / UNCERTAIN from the pre-declared
Elastic Net rule), `SCIENTIFIC_SUMMARY_HE.md`, `SCIENTIFIC_SUMMARY.md`, `PRIMARY_70_SENSITIVITY_COMPARISON.csv`, `THRESHOLD_TRADEOFF.csv`,
`SENSITIVITY_TARGET_TABLE.csv` (50 / 60 / 70 / 75 / 80 / 90%), `OUTER_FOLD_RESULTS.csv` (each outer fold at the 70% rule), `CAPACITY_CURVE.csv`, `MODEL_COMPARISON.csv`, `DOMAIN_INCREMENTAL_VALUE.csv`,
`ABLATION_RESULTS.csv`, `OOF_MODEL_COMPARISON.csv`, `OOF_PREDICTION_SUMMARY.csv`, `CALIBRATION.csv`, `SUBGROUP_SUMMARY.csv`,
`FEATURE_ELIGIBILITY.csv`, `NEW_FEATURE_CATALOGUE.csv`, `ALL_V21_COLUMN_CLASSIFICATION.csv`, `SCHEMA_DIFF_V1_V21.csv`, `REMOVED_V1_COLUMNS.csv`,
`RENAMED_OR_CHANGED_COLUMNS.csv`, `FEATURE_STABILITY.csv`, `PERMUTATION_IMPORTANCE.csv`, `SHAP_SUMMARY.csv`, `COHORT_FACTS_2026.json`,
`OUTCOME_CONTRACT_2026.json`, `RUN_MANIFEST.json`, `RUN_TIMINGS.csv`, `ENVIRONMENT.json`, `PRIVACY_SCAN.json`, `figures\` (12 figures).

Send back only `%OUT5%\share` (zip it). It exists only after the identifier, path, file-type and row-level scans passed. Never send the
extract, `%OUT5%\work` (row-level frame, folds, out-of-fold predictions, model objects) or `%OUT5%\logs`.
