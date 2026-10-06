# falls_ml 0.12.3 — SQLite lifecycle patch and Phase 5 takeover evidence

Baseline: clean branch `phase5`, commit `d1e611c8d0cf750ec8e439775f4ff902d6387da5`, falls_ml 0.12.2 / Phase 5 2.2.0.
Baseline package: `falls_ml_phase5_0.12.2_mailsafe.zip`, SHA-256 `e22f4efe6c0fcee0eb17739468748841564d2c644a4b3f13bffd73be19f75901`.
All 162 mail-safe restored source files were byte-identical to that baseline checkout. The main checkout was older and was not used.

## Root cause and actual scope

`phase2.xgb_tuning.run_study` passed `TrialStore.url` to Optuna `create_study` and each `load_study`. A string storage URL creates a fresh `RDBStorage` wrapped in `_CachedStorage`, including a SQLAlchemy Engine, connection pool, scoped session registry, and SQLite DBAPI connections. Study and Trial objects reference their storage; object reference cycles and pool references can survive the function. Returning or overwriting a local Study variable does not dispose its engine. The old code never removed the sessions or explicitly disposed those engines. A new attempt calls `TrialStore.fresh`, which immediately archives the main SQLite database and any journal/WAL/SHM sidecars. Windows cannot replace the file while any of those connections hold it open. Unix allows renaming an open file, so a successful Unix rename alone would not prove correct cleanup.

This is a **production resource-lifecycle bug exposed by a restart test**, not only a fixture mistake. Phase 2 `s08_xgb` and Phase 3 `s09_xgb` use the same function. It matters especially for in-process restart or exception recovery and also retains avoidable resources during tuning. A fully terminated Python process releases its handles at OS exit, so ordinary resume in a new process does not necessarily reproduce the observed lock. It does not prove that every real XGB resume failed or that predictions from a completed run changed.

Phase 5 `_tune_xgb` creates in-memory Optuna studies with no storage URL and rebuilds history from trial JSONL/NPY artifacts. It does not call this SQLite function. The dashboard has a separate read-and-aggregate graph and calls neither tuning path. **The reported SQLite failure never affected the dashboard path.** A dedicated dashboard command could safely generate reports from an intact completed 2.x folder even before this patch; a clean patch is supplied first as requested.

Upstream ownership semantics: [Optuna RDBStorage/remove_session](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html), [Optuna storage implementation](https://optuna.readthedocs.io/en/stable/_modules/optuna/storages/_rdb/storage.html), [SQLAlchemy engine disposal](https://docs.sqlalchemy.org/en/21/core/connections.html#engine-disposal). Removing a session releases checked-out connections; engine disposal then closes idle pooled connections. Both steps are necessary. This application's storage calls remain on the invoking thread; its objective does not receive the storage or Study.

## Code changes

- `src/falls_ml/phase2/xgb_tuning.py`: each `run_study` owns one explicit RDBStorage shared by all its create/load calls. A lazy local subclass closes a partially constructed storage if initialization fails; the context manager closes storage on success and all later exceptions. Cleanup removes the current scoped session and then disposes the Engine, including when session removal raises. No skip, delay, retry, ignored PermissionError, or sampler/seed change.
- `TrialStore.fresh` becomes fresh only after archive and ledger repair succeed; a failed archive is propagated rather than silently treated as completed next time.
- `tests/unit/test_xgb_storage_lifecycle.py`: real SQLite journal/WAL regressions, retaining DBAPI handles and traceback references while garbage collection is disabled. No XGBoost training is needed for these objectives.
- Historical Phase 2/3/4 protection manifests are unchanged. Three contract guards use the test-only `protected_source_matches` fixture and `SOURCE_PATCH_EXCEPTION_0.12.3.sha256` to accept only this module's exact reviewed patch hash against its exact historical hash, in version 0.12.3, plus the exact corresponding edit to the protected Phase 4 regression guard (test code only). A regression rejects altered bytes, wrong paths and wrong baseline hashes. All other earlier implementation/configuration files stay frozen.
- New fast dashboard CLI tests and audit documentation; no dashboard implementation changes.
- falls_ml identity becomes 0.12.3. Phase 5 remains 2.2.0. The runbook names the new independent installation; one metadata-only CSV is explicitly admitted by the existing package scanner.

No Phase 5 preprocessing, training, model class, OOF format, or fitting implementation changes. Completed output compatibility is preserved. Install the package in a new folder; do not replace earlier installations or resume/retrain the completed experiment.

## Evidence and limits

Before the first fix: six journal/WAL success/objective/restart tests failed because retained real DBAPI connections still executed SQL, on both Python 3.11 and 3.13. After the first fix: six passed; original resume test also passed once the local OpenMP runtime was available. Review found the constructor exception gap; two newly written initialization regressions failed before that correction and passed afterwards.

Additional post-commit tell/ledger interruption and archive-state tests were run against source extracted from the SHA-verified baseline package: five failed as expected. The patched version closes connections and reconciles durable records on restart. The repeated-restart test compares entire trial results (suggestions, parameters, values, item seeds), checks immediate archives over four attempts, and checks an unduplicated six-trial ledger. Both rollback-journal and WAL modes are exercised.

These tests assert every observed DBAPI connection is actually closed, not merely that a POSIX rename succeeded. Immediate rename/replacement runs in the same tests and will exercise Windows locking when run there. **No Windows or Linux host was available in this session**; macOS Python 3.11 and 3.13 verification does not replace a Windows rerun. No real patient data, completed outputs, or fitted models were accessed.

The first broad development-environment fast run exposed dependency-lock mismatches (latest pip installation versus existing frozen lock), four expected source-freeze failures after the authorized fix, and a pre-existing personal path in tracked `FINAL_RUN_READINESS.md`. Dependencies were aligned to the unchanged locks; the one reviewed freeze exception was added. That old root note is outside the handoff include list and is not changed or shipped. Final delivery validation must run from the restored package, whose scanners examine all shipped files.

### Verified regression totals

- Restored mail-safe package, clean locked Python 3.11 environment: **2436 passed, 1 skipped, 71 deselected**, running `python -m pytest -m "not slow" -q` (178.06 seconds). The skip is an ablation configuration that owns its schema; the warning is an expected undefined calibration slope.
- Restored rebuilt ZIP: **143 passed**, covering lifecycle, original resume, dashboard, capacity, exception policy and handoff scripts; all 389 manifest payloads verified.
- Existing slow Phase 5 dashboard integration: **1 passed** (422.56 seconds), against a fully trained synthetic run with fitting functions blocked during dashboard generation and saved results preserved.
- Focused source checks on Python 3.11: **68 passed**, including lifecycle, original deterministic resume, frozen-source contracts, dependency locks, dashboard and capacity checks.
- Python 3.13: **15 passed**, covering all 13 lifecycle regressions, the original deterministic-resume test and the exception-policy regression.
- An additional run including all slow tests was interrupted after **1194 passed, 1 skipped, 1 failed** (16 minutes). It is not a passing full-suite result. The failure is `test_eda_reproduces_a_completed_reference_run_and_never_writes_into_it`: the legacy Phase 1 synthetic reference model has a constant linear predictor, making logistic recalibration undefined. Isolated fresh runs passed on both releases. Reusing the exact failed synthetic CSV, pseudonym pepper and template reproduces the same `RecalibrationError` on the SHA-verified original 0.12.2 package. This establishes a pre-existing, fixture-dependent legacy issue unrelated to the storage patch; that scientific path is unchanged.

## Preprocessing verdict and retraining

**MATERIAL_ISSUE_RETRAINING_SHOULD_BE_CONSIDERED**

See [exact-source audit](PREPROCESSING_AUDIT_0.12.2.md) and [130-feature table](PREPROCESSING_FEATURE_AUDIT_0.12.2.csv). Inner/outer local median imputation and scaling are fitted correctly. Outcome-dependent eligibility is still calculated over the whole cohort before outer CV; CCI remains continuous despite unvalidated group-code spacing. This is distinct from the SQLite bug. Source inspection cannot establish the real-run exclusion effects or order mandatory retraining. No retraining is needed to fix storage ownership or produce the dashboard; methodological reassessment should precede deciding on any later redevelopment run.

## Windows commands for the already completed folder

Extract `falls_ml_phase5_0.12.3_mailsafe.zip` under Downloads. In CMD:

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.12.3"
py -3.11 RESTORE_FILES.py.txt
setup_windows.cmd
set PYTHONUTF8=1
.venv\Scripts\python.exe -c "import falls_ml; print(falls_ml.__version__)"
.venv\Scripts\python.exe -m pytest tests/unit/test_phase2_core.py::test_study_resumes_with_identical_suggestions tests/unit/test_xgb_storage_lifecycle.py tests/unit/test_phase5_dashboard.py tests/unit/test_phase5_capacity.py -q
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5-dashboard --out "%USERPROFILE%\Downloads\100k_falling_db_phase5_v3"
start "" "%USERPROFILE%\Downloads\100k_falling_db_phase5_v3\share\PHASE5_OPERATING_DASHBOARD.html"
```

The restore step must say PACKAGE VERIFIED, setup INSTALLATION SUCCESSFUL, version 0.12.3, and targeted tests must pass. The dashboard consumes existing committed artifacts and writes aggregate share reports, share backups and a run log. It uses proportional within-outer-fold ranking, with pooled probability ranking confined to descriptive outputs. The original matching extract must remain available for input identity and the read-only identifier privacy scan. No --input flag is needed when its planned file name is next to the output folder; otherwise pass --input with the actual original path. Do not use the training/resume/report-only command to obtain this dashboard; report-only rewrites local OOF artifacts. See [dashboard safety evidence](DASHBOARD_SAFETY_AUDIT.md).

Final test totals and release commit/package SHA are recorded in the delivery evidence after the verified build, rather than embedded here as circular self-identifiers.
