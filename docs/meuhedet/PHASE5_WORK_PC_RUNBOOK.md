# Phase 5.1 on the work PC – REPAIR-ONLY correction of the completed Phase 5 2.2.0 run (falls_ml 0.13.1, Phase 5 3.0.0)

**Do NOT start the real run before the PI's explicit approval of the implementation evidence.** This package implements Experiment 1 of
`docs/phase6/FINAL_CONSENSUS.md` exactly as approved in `docs/phase6/EXPERIMENT1_IMPLEMENTATION_CONTRACT.md` (registration:
`docs/phase6/REGISTRATION_EXP1.md`). It answers one question: *what happens to the existing Phase 5 result when the methodological defects are
corrected while changing as little else as possible?* It has no success verdict. The completed 2.2.0 folder (**PRE**, for example
`%USERPROFILE%\Downloads\100k_falling_db_phase5_v3`) is verified before anything is fitted, its outer folds are adopted, and it is never written to.

Identical to PRE by construction: the input file (sha256), the usable cohort and labels, the outer folds, the seeds, the ENET family, the tuning
objective, the candidate space (40 lambdas x l1 ratios {0.1, 0.25, 0.5, 0.75, 0.9}) and budgets, the feature-set rules, the capacity arithmetic.
The repair: no outcome-dependent eligibility screen (R-1); a per-fold label-free coverage gate (R-2); the lambda grid anchored inside each inner
training fold (R-3); CCI_Group quarantined (R-4); an explicit `other` level for learned nominal codes (R-5); a local, report-only forensic AUROC (R-6).
Units: ENET x {OLD, OLD_PLUS_ALL_NEW_ELIGIBLE (audit only), OLD_PLUS_NEW_SAFE (= ADMISSIBLE)} x 5 outer folds, FINAL for the three sets, one secondary
diagnostic (ADMISSIBLE minus NEW_REGISTRY), explanation / stability on ENET. No LASSO, no XGBoost, no domain units, no Policy B.

## 0. Prerequisites (confirm before step 4)

- the January 2026 extract on this PC is byte-identical to the file the PRE run used (the preflight checks its sha256 against the PRE plan);
- the PRE folder is complete (`RUN_STATUS.json` COMPLETE, or REPORT_COMPLETE after the approved 0.12.3 `--report-only` regeneration —
  amendment A-1; `share\RUN_MANIFEST.json` FINAL) and holds `share\TOP3_CAPACITY_PRIMARY.csv`
  (written by the 0.12.3 dashboard command; if it is missing, run `meuhedet-phase5-dashboard` from the 0.12.3 package on the PRE folder first);
- `OUT51` is a NEW folder (never the PRE folder, never OneDrive).

## 1. Restore and set up (once)

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.13.1"
py -3.11 RESTORE_FILES.py.txt
setup_windows.cmd
.venv\Scripts\python.exe -c "import falls_ml; from falls_ml.phase5 import PHASE5_VERSION; print(falls_ml.__version__, PHASE5_VERSION)"
```

Expected `PACKAGE VERIFIED`, `INSTALLATION SUCCESSFUL`, then `0.13.1 3.0.0`.

## 2. The paths (every new CMD window)

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.13.1"
set PYTHONUTF8=1
set INPUT_2026=%USERPROFILE%\Downloads\100k_falling_db_2026.csv
set PRE=%USERPROFILE%\Downloads\100k_falling_db_phase5_v3
set OUT51=%USERPROFILE%\Downloads\100k_falling_db_phase51_v1
```

## 3. Synthetic rehearsal (optional, no real data; 10-20 minutes)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5-synthetic --out "%USERPROFILE%\Downloads\phase51_synthetic_smoke" --rows 4000 --negative-controls
```

Ends with `"status": "COMPLETE"`.

## 4. PRE verification + preflight (mandatory first; fits nothing; minutes)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT51%" --pre-run "%PRE%" --preflight-only
```

Read `%OUT51%\preflight\PHASE5_PREFLIGHT.md`: check P10 (PRE verified: input sha256, cohort / labels, fold hash ADOPTED, ENET units, Top-3% table
reproduced). The last line must be `SAFE TO MODEL`. A `STOPPED [PRE_VERIFICATION_FAILED]` names the failed check; nothing was fitted - do not continue.
Without `--pre-run` a real-data preflight stops with `PRE_RUN_REQUIRED`.

## 5. Negative controls (mandatory before the run; quick budget; about 0.5-1.5 h)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT51%" --pre-run "%PRE%" --negative-controls
```

No `--mode` here: the folder keeps the mode of its preflight (overnight by default) and the controls always use their own registered quick budget;
a different `--mode` on the same folder stops with `PLAN_MISMATCH`. Ten frozen-fold label permutations through the complete unit pipeline. Expected last line `NEGATIVE CONTROLS PASSED` (mean AUROC <= 0.55 and mean
Recall@Top3 <= 5%). `STOPPED [NEGATIVE_CONTROL_FAILED]` is a hard stop: nothing is fitted or reported until the cause is found and recorded in
`docs/phase6/AMENDMENTS.md`. The overnight command refuses a folder without passed controls (`NEGATIVE_CONTROLS_REQUIRED`).

## 6. Runtime estimate (optional)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT51%" --estimate --mode overnight
```

## 7. The run (one command; resumable; roughly 2.5-5 h)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT51%" --pre-run "%PRE%" --mode overnight --device cpu --resume
```

`--pre-run` must name the same PRE folder as the preflight (the plan records its name and digest; a changed PRE folder stops with `PRE_RUN_MODIFIED`).
Progress: `OVERNIGHT_PROGRESS.log`, `RUN_STATUS.json`, `RUN_TIMINGS.csv`; status from a second window:

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --out "%OUT51%" --status
```

## 8. Resume after an interruption

Exactly the same command as step 7 (it verifies the input, settings, mode, code and PRE folder and continues with the first unfinished unit).

## 9. Dashboard (reporting only) and what to send back

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5-dashboard --out "%OUT51%"
```

Send back ONLY `%OUT51%\share` (zipped). Its first table, in `TOP3_PRE_POST_HEADLINE.csv` and at the top of `MANAGEMENT_SUMMARY_HE.md` /
`SCIENTIFIC_SUMMARY.md`: for OLD and ADMISSIBLE, PRE 2.2.0 vs POST 5.1 - N, falls, exact selected count at 3%, captured falls, Recall@Top3, PPV@Top3,
false interventions, the PRE -> POST difference in captured falls with its paired 95% CI, the difference in Recall@Top3 and in false interventions
(per 10,000 only in addition). Also `PRE_POST_PAIRED.csv`, `PRE_POST_CONTRAST.csv`, `PRE_POST_CORRECTION.csv`, `PRE_POST_MEMBERSHIP.csv`,
`PRE_POST_HISTORICAL_70_RULE.csv`, `PRECISION_PLANNING.csv`, `NEGATIVE_CONTROLS.csv`, `HISTORICAL_VERDICT_2_2_0.json` (the 2.2.0 70% rule, audit only)
and the 2.2.0 share set. `work\FORENSIC_UNIVARIATE_AUROC_BY_FOLD.csv` stays local unless `--share-forensic` was given. Never send the extract,
`work\` or `logs\`.

---

## Phase 5 2.2.0 runbook (historical; the PRE run) – kept for reference

# Phase 5 on the work PC – 2026 redevelopment + incremental value of the new V21 information (falls_ml 0.12.3, Phase 5 2.2.0)

For an **already completed Phase 5 folder**, use only the dashboard command in step 10. This 0.12.3 patch fixes SQLite ownership in the shared Phase 2/3 tuning helper and audits the unchanged Phase 5 preprocessing. It does not require a new model run. See `docs/phase5/SQLITE_LIFECYCLE_PATCH_0.12.3.md` for the focused existing-output instructions and audit limitations.

Phase 5 answers one question on the 2026 extract (Index_Date 2026-01-01, prediction at the END of the index day): **at approximately the same
>= 70% fall sensitivity, does adding the new V21 information reduce the number and the percentage of false alerts?** The pre-declared primary
comparison is **Elastic Net OLD vs Elastic Net OLD_PLUS_ALL_NEW_ELIGIBLE**; LASSO and XGBoost are secondary, and OLD_PLUS_NEW_SAFE is the stricter
sensitivity analysis. It is INTERNAL NESTED CROSS-VALIDATION ON THE 2026 SNAPSHOT (development data), never external validation. Phase 4 (temporal
validation) stays frozen and unused; this package does not touch Phase 2 / 3 / 4 installations or output folders (keep the Phase 4 package
`falls_ml_phase4_0.10.0` for any future Phase 4 run).

The 2026 schema is the AUTHORITATIVE V21 VIEW definition you supplied (224 columns), embedded in the package
(`configs\meuhedet\phase5_v21_view_definition.txt`, with its column-by-column review in `configs\meuhedet\phase5_v21_schema.yaml`). The preflight
recomputes the exact V1 -> V21 diff from the header of YOUR file and classifies every column; a column the schema does not define stops the run
for review (no clinical column is ignored silently). A rename is never inferred from a column position or name: the three V1 registry flags V21
dropped (hypertension, chronic renal failure, transplant) were compared with the V21 COVID-19 (116 / 118), dialysis (101 / 1) and
immunosuppression (130 / 131) registries; without the V1 SQL / registry IDs and with V21 explicitly contradicting the V1 labels, their lineage is NOT
proven (`OLD_REMOVED_NEW_ADDED`, documented in `SCHEMA_DIFF_V1_V21.csv`): the V1 features leave OLD and the three V21 fields are genuinely new
predictors.

| Step | What | Output |
|---|---|---|
| PREFLIGHT (run alone first) | exact V1 -> V21 schema diff, outcome / future / identifier columns sealed from X, cohort (one row per patient), the 2026 outcome contract, feature eligibility (timing / provenance / semantics / coverage / leakage), the three feature sets, the fixed folds | `preflight\` (aggregate); last line `SAFE TO MODEL` or `STOP - REVIEW REQUIRED`; nothing is fitted |
| PRIMARY | nested CV (5 outer x 5 inner folds), Elastic Net (primary) / LASSO / XGBoost x {OLD, OLD_PLUS_ALL_NEW_ELIGIBLE, OLD_PLUS_NEW_SAFE} | an INTERIM `share\` with the main answer as soon as it is done |
| FINAL, DOMAIN, ABLATION, EXPLAIN, STABILITY | final development models, OLD + each new domain (all families), ablations of the primary Elastic Net model, permutation importance / SHAP, coefficient stability | – |
| REPORT | management + scientific summaries (Hebrew / English), tables, 12 figures, privacy scan | `share\` (aggregate only) |

All commands are CMD. The real data never leave the work PC; only `share\` (and, after a preflight, `preflight\`) may be sent back.

## 1. Restore and set up (once, about 10-20 minutes)

1. Save `falls_ml_phase5_0.12.3_mailsafe.zip` to `%USERPROFILE%\Downloads` and extract it there. You get the NEW folder
   `%USERPROFILE%\Downloads\falls_ml_phase5_0.12.3` (the earlier `falls_ml_phase5_0.11.0` / `0.12.0` / `0.12.1` package folders are not used
   any more; do not mix them. An OUTPUT folder completed by 0.12.1 stays valid: step 10 builds its dashboard with this package).
2. Restore the mail-safe files and verify every file (sha256):

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.12.3"
py -3.11 RESTORE_FILES.py.txt
```

   The last line must be `PACKAGE VERIFIED ...`.
3. Install into this folder only (its own `.venv`; the same pinned, hash-checked dependencies as Phases 2-4):

```bat
setup_windows.cmd
```

   Wait for `INSTALLATION SUCCESSFUL`. Without internet access: run `prepare_offline_package.cmd --target windows-amd64-cp311` in this folder on a
   connected computer, copy the resulting `offline_packages` folder next to `setup_windows.cmd`, and run `setup_windows.cmd --offline`.
   Check the version (expected `0.12.3 3.2.0 5.0.0`):

```bat
.venv\Scripts\python.exe -c "import falls_ml, xgboost, optuna; print(falls_ml.__version__, xgboost.__version__, optuna.__version__)"
```

## 2. The paths (type these lines in EVERY new CMD window before a command)

Replace only the 2026 file name if it differs:

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.12.3"
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
  outcome contract fails (including ANY fall-positive whose event lies after the patient's own `Followup_End_Date` - zero tolerance; only the
  aggregate count and percentage are reported, no label is changed and no patient is excluded), duplicated patients at the index date, or the
  VIEW's own leakage flag is set.

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

## 10. Operating dashboard of a COMPLETED run (minutes; no model is fitted)

"If we can intervene on X% of the population, how many falls do we capture?" - computed from the run's committed outer out-of-fold
predictions (nothing is refitted, tuned or re-validated; the earlier results stay byte-identical):

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5-dashboard --out "%OUT5%"
```

* The run's 2026 extract is found automatically when it lies next to the output folder (same name and sha256 as in the plan); otherwise add
  `--input "<the extract>"` (it is read only for the identifier scan of share\).
* Outputs in `%OUT5%\share` (aggregate only, privacy scan re-run): `PHASE5_OPERATING_DASHBOARD.html` (open in Chrome / Edge; works offline from
  the file system), `CAPACITY_CURVE_FINE.csv` (0.5-20% in 0.1% steps), `CAPACITY_CURVE_EXTENDED.csv`, `CAPACITY_CURVE_POOLED_DESCRIPTIVE.csv`,
  `TOP3_CAPACITY_PRIMARY.csv`, `TOP3_CAPACITY_BY_FOLD.csv`, `TOP3_CAPACITY_COMPARISON.csv`, `TOP3_CAPACITY_BOOTSTRAP.csv` (2,000 paired
  replicates), `TOP3_CAPACITY_SUMMARY_HE.md`, `CAPTURE_TARGET_CAPACITY.csv`, `FEATURE_DRIVERS.csv`; `MANAGEMENT_SUMMARY_HE.md` now opens with the
  3% question (the ~70% sensitivity analysis follows unchanged as the secondary analysis).
* Method: the capacity is allocated across the outer folds (largest remainder) and the highest-risk patients are selected within each fold by
  that fold's model - never a pooled probability threshold. The previous share\ is kept in `%OUT5%\work\dashboard\`.

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
