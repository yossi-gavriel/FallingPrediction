# Phase 5 dashboard safety audit

Audit baseline: commit `d1e611c8d0cf750ec8e439775f4ff902d6387da5`,
falls_ml 0.12.2, Phase 5 2.2.0. Delivery package: falls_ml 0.12.3;
the Phase 5 model implementation and scientific version remain 2.2.0.

## Conclusion

`meuhedet-phase5-dashboard` is an analysis and reporting command. Its executed
call graph does not train or refit ENET, LASSO, or XGBoost; run nested CV;
invoke Optuna; construct tuning storage; or open SQLite connections. It reads
the saved outer-OOF predictions and the saved explanation/stability tables.
It does not unpickle or predict with model objects.

The known Windows Optuna/SQLite lifecycle problem belongs to
`phase2/xgb_tuning.py`: the original implementation created and loaded studies
with a storage URL and did not explicitly close the associated engines before
later attempts archived SQLite files. That path, including `TrialStore.fresh`
and `run_study`, is absent from dashboard execution. Phase 5's training search
also uses in-memory Optuna studies in `phase5/engine.py`; the dashboard calls
neither search implementation. The storage lifecycle bug therefore cannot be
triggered by this dashboard command.

The command can extend the existing completed Phase 5 output folder
`%USERPROFILE%\Downloads\100k_falling_db_phase5_v3` without retraining or
replacing its models or OOF predictions. This is a source and synthetic-test
conclusion, conditional on that folder containing an intact completed Phase 5
2.x run and its matching input extract. The audit did not open that folder,
any existing output directory, or any real patient data.

No dashboard implementation change was needed or made for this audit.

## Executed CLI graph

1. `__main__.py` calls `cli.main`; parsing selects
   `_cmd_meuhedet_phase5_dashboard` directly. Its only dashboard arguments are
   `--out`, optional `--input`, and optional integer `--bootstrap` (default
   2000). It does not dispatch `_cmd_meuhedet_phase5` or `run_phase5`.
2. `dashboard.run_dashboard` loads `work/PLAN.json`, verifies the Phase 5 2.x
   version and frozen frame/fold hashes, requires the existing manifest and
   management summary, and locates the exact input using its name, size, and
   SHA-256. It loads the saved frame, labels/folds, metadata and feature sets.
   `DeviceState("cpu")` only constructs a state record; it does not probe or
   train XGBoost.
3. `dashboard.build_operating` calls `capacity.load_primary_oof`.
   `engine.is_complete` verifies saved unit payload hashes against
   `COMPLETE.json`; `engine.load_result` reads JSON and NPZ arrays. The loader
   checks held-out row membership and finite predictions. It does not call
   `engine.load_model`, `run_unit`, or `tune_and_fit`.
4. Capacity curves, fold counts, and the paired bootstrap operate on those
   arrays. `explain.importance_tables` and `explain.stability_table` read
   committed CSV outputs. They do not call `run_fold_explain`, `run_stability`,
   permutation/SHAP recomputation, or any fitting primitive.
5. `write_operating` produces aggregate CSV/Markdown and a self-contained HTML
   document. `report.publish` scans the candidate share before publication.
   The matching source extract is hashed to establish identity and read for
   identifier columns by `phase4.sealed.read_ids`; the saved frame's row keys
   are also checked. These identifier reads are local and read-only.

Importing the engine or explanation modules imports function definitions; it
does not execute the training functions defined in them. The standalone
dashboard does not call `report.build_reports`. That distinction matters:
`meuhedet-phase5 --report-only` calls `build_reports`, which regenerates local
OOF analysis artifacts. Use the dedicated dashboard command for the stated
OOF-preservation requirement.

## Writes and integrity limits

Permitted writes are the aggregate report additions and revised management
summary/manifest/privacy scan in `share/`; staging in `.tmp-share`; the original
management summary and prior share backup in `work/dashboard/`; and the
append-only `logs/dashboard_runs.jsonl` record. The command replaces the share
folder only after the privacy scan passes. Existing aggregate results other
than the management summary, manifest and privacy scan are copied unchanged.

The production before/after `units_digest` hashes the committed unit and
explanation `COMPLETE.json` records. It is not a second direct hash of every
payload and does not cover local OOF or SQLite files. Payload integrity is
also checked while committed units/tasks are loaded. The regression tests
below add direct byte SHA-256 and modification-time checks of all protected
synthetic inputs; they do not rely only on `units_digest` or the manifest's
`no_model_fitted` declaration. This audit does not claim protection against a
separate process concurrently changing an output folder.

## Primary method and privacy

For capacity c, the total number selected is `round(c * N)` with deterministic
half-up arithmetic. The total is allocated proportionally to outer-fold sizes
using largest remainders, with deterministic ties. Each fold selects its own
top k saved OOF risks; counts are summed. Cross-fold probability scales are
never compared for the primary estimate. The pooled OOF view is explicitly
descriptive and the HTML defaults to the fold method. The bootstrap uses the
same patient weights for all models, reallocates capacity to resampled fold
sizes, and reselects within each fold for every replicate.

The HTML contains aggregate capacity counts, bootstrap summaries, and
feature-level explanation summaries. Its CSS and JavaScript are embedded; it
has no downloaded script, stylesheet, font, fetch, socket, or server
requirement. The SVG namespace is a namespace identifier, not a network
request. It opens locally in Chrome or Edge. The shared tables and HTML omit
patient identifiers, row keys, individual risks, and patient dates. Unsafe
curve points are excluded, and rows containing small patient cells are
suppressed. Publication also scans identifiers, local paths, row-level table
columns/size, and forbidden artifact types.

## Synthetic regression evidence

`tests/unit/test_phase5_dashboard.py` constructs a complete artifact layout
from synthetic arrays, without training any model. It includes all three
families and three primary feature sets; stored explanation/stability CSVs;
deliberately non-loadable model canaries; a saved local OOF parquet; a genuine
SQLite file with synthetic contents; and synthetic journal/WAL/SHM canaries.

The actual CLI is run twice with 20 bootstrap replicates. Guards raise on
training/tuning/CV, model loading or prediction, Optuna study/storage APIs,
the Phase 2 study runner, SQLAlchemy engine creation, and SQLite connections.
XGBoost entry points are a trap module, allowing this reporting-only test to
run even without a native OpenMP runtime. No prohibited call occurs. Direct
SHA-256 hashes and modification times of the input, frozen work files, all
unit/model/OOF/explanation payloads, and SQLite canaries remain identical.
The protected directory inventories also remain identical. The old 70%
aggregate comparison remains byte-identical. Output selection is checked
against independent brute-force within-fold ranking on deliberately different
fold probability scales; privacy and offline HTML checks pass. A mismatching
input fails with `INPUT_MISMATCH` before publication or protected-file changes.

Verification, Python 3.11 with Optuna 5.0.0:

```text
PYTHONPATH=src python -m pytest -q tests/unit/test_phase5_dashboard.py tests/unit/test_phase5_capacity.py
7 passed
```

The five existing capacity tests additionally cover exact apportionment,
within-fold versus pooled ranking, bootstrap re-selection, small-cell
suppression, and aggregate/self-contained HTML. This test run used only
temporary synthetic files. Windows execution itself was not performed by this
audit.

## Exact Windows dashboard procedure

Extract the independent `falls_ml_phase5_0.12.3_mailsafe.zip` into Downloads.
Restore its executable source files and verify the manifest, then install the
new package's own environment. These setup commands do not train on the real
run:

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.12.3"
py -3.11 RESTORE_FILES.py.txt
setup_windows.cmd
```

Require `PACKAGE VERIFIED` and `INSTALLATION SUCCESSFUL`. If using the provided
offline dependencies, use `setup_windows.cmd --offline` instead. Leave the
previous installation and completed output folder in place.

In the new package folder, run only the dedicated dashboard command against
the existing completed folder:

```bat
set PYTHONUTF8=1
set "OUT5=%USERPROFILE%\Downloads\100k_falling_db_phase5_v3"
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5-dashboard --out "%OUT5%" --bootstrap 2000
start "" "%OUT5%\share\PHASE5_OPERATING_DASHBOARD.html"
```

The exact original input is located automatically when its planned file name
is next to the output folder (or inside it). If it lives elsewhere, supply the
original extract's actual path; no replacement or re-export is required:

```bat
set "INPUT_2026=<actual path to the original 2026 extract>"
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5-dashboard --out "%OUT5%" --input "%INPUT_2026%" --bootstrap 2000
```

Expected result is exit code 0 and `DASHBOARD_COMPLETE`, with
`privacy_passed: true` and `units_unchanged: true`. Send only the privacy-checked
`share` folder. Do not send the extract, work files, models, local OOF, SQLite,
or logs. The dashboard has no `--resume`, `--mode`, `--device`, `--jobs`, or
`--accept-code-change` flag; no training or preflight command is needed to add
a dashboard to a completed run.

## Mail-safe package build procedure

The source package builder's explicit include list contains source, tests,
configuration, documentation, setup scripts and approved synthetic fixtures;
it excludes environments, real data and generated output. It verifies version
consistency, file line endings, allowed data fixtures, lock format, and content
scans before writing a handoff. After the maintainer updates the patch version
consistently, the existing builders provide the verified mail-safe package:

```bat
python tools\build_handoff.py --dist <clean-build-directory>
python tools\build_phase5_package.py --handoff <clean-build-directory>\falls_ml_handoff --out <clean-build-directory>
```

`--force` is supported by `build_handoff.py` when intentionally replacing a
previous build. The Phase 5 builder renames `.py` and `.cmd` archive entries to
`.py.txt` and `.cmd.txt`, supplies `RESTORE_FILES.py.txt` and a Windows readme,
and writes `SHA256SUMS_phase5_0.12.3.txt`. Restoration verifies every original
package file against `PACKAGE_MANIFEST.txt`. The zip has one top folder
`falls_ml_phase5_0.12.3`; the deliverable is
`falls_ml_phase5_0.12.3_mailsafe.zip`. No model training or real-data access is
part of these build steps.
