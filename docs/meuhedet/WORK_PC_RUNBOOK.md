# Meuhedet Phase 1 — runbook for the work computer

Everything below runs inside the approved Meuhedet environment on one **local extract file**. The pipeline never connects to a
database and never needs credentials. Nothing patient-level leaves the environment: only the aggregate audit
(`wide_extract_audit.md/.json`) and, later, the run report/metrics (aggregates only) are meant to be reviewed outside.

All commands are CMD lines run from the `falls_ml_handoff` folder after `setup_windows.cmd` succeeded. `%PY%` below stands for
`.venv\Scripts\python.exe`; set it once per CMD window:

```bat
set PY=.venv\Scripts\python.exe
set PYTHONUTF8=1
```

## The real-data workflow (v0.5.0): readiness, one snapshot, all snapshots

Three commands, all run from the `falls_ml_handoff` folder on a local extract (CSV, Excel or Parquet; dates in `%Y-%m-%d` or
`%d/%m/%Y`; identifiers as text). Nothing patient-level is written outside the results folder's `datasets/` and `runs/*/predictions_*`
files; every `.md` / `.json` / `.csv` report at the top level and under `audit/` is aggregate and shareable.

**1. Readiness audit (no training; seconds on 60k rows, ~a minute on 100k+):**

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-explore --input "C:\FallsData\60k_falling_db.csv" --all-index-dates --audit-only
```

It reads the file through the contract, writes the aggregate audit, the **D-00 timing diagnostic** (`audit/timing_diagnostic.md`), the
**snapshot audit** (`audit/snapshot_audit.md`: every Index_Date with rows, patients, eligible, censored, events, prevalence, D-00 violations,
180-day observability, USABLE / EXCLUDED with the reason, partition), builds the STRICT and EXTENDED datasets, computes the split
(`split_audit.md`: rows, unique patients, events, index-date ranges and a sha256 per partition), runs the **adequacy gate**
(`adequacy_strict.md`, `adequacy_extended.md`) and ends with `READINESS.md`: **READY TO TRAIN** or **STOPPED — <reason>** (exit code 2).
Use `--index-date YYYY-MM-DD` instead of `--all-index-dates` to check one snapshot.

**2. Single snapshot (the 60k-rows-on-one-date experiment):**

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-explore --input "C:\FallsData\60k_falling_db.csv" --index-date 2025-01-01
```

read → validate → audit → cohort (`Is_Eligible_Cohort = 1`) → labels → point-in-time history (D-00) → adequacy → reproducible
patient-grouped split → STRICT → EXTENDED → evaluate → report. Add `--data-freeze-date YYYY-MM-DD` (last source load; used for the label
maturity check and the snapshot observability cross-check).

**3. All snapshots (the longitudinal experiment):**

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-explore --input "C:\FallsData\60k_falling_db.csv" --all-index-dates --data-freeze-date 2026-06-30
```

Every snapshot is validated; those without a fully observable 180-day outcome (the VIEW's `Has_Full_180D_Label` / `Is_Censored_180D` /
`Fall_Next_180D_Ind` fields decide, the calendar only cross-checks) are EXCLUDED with the reason. The PRIMARY analysis is **temporal**:
earlier usable snapshots train, later ones validate, the latest test, with the D-19 outcome-window embargo (a training row is kept only
if its 180-day window ends before the first validation index date; a validation row only if its window ends before the first test index
date) - with monthly snapshots this removes about six snapshots at each boundary, which the snapshot audit shows as `train (embargoed)` /
`validation (embargoed)`. Boundaries are chosen from the data (test ≥ 20% of usable rows and ≥ 10 events; validation the smallest block
that keeps ≥ 10 events after the embargo; train everything earlier). The same patient may appear in several partitions - this estimates
repeated risk prediction over time, and the split audit counts it. A second, separate estimand is the **patient-disjoint sensitivity
analysis** (`comparison_patient_disjoint.md`, `runs_patient_disjoint/`): all snapshots of a held-out 20% of patients form the test set
(no patient in two partitions). `temporal_test_by_snapshot.md` gives the temporal test performance per Index_Date (`INSUFFICIENT EVENTS`
below 10 events). Options: `--no-patient-disjoint`, `--max-censored-share 0.5`.

**Runtime.** On the development machine a fast STRICT run on 48k rows takes about 2 minutes (37 LASSO CV fits: the final model, 30
stability replicates, 5 optimism replicates); the reading, audits, datasets and split take under 10 seconds. Expect a few minutes per run
on the work PC, four runs in `--all-index-dates` mode, and roughly six times longer with `--full` (stability 200, optimism 25, bootstrap 1000).

### Reproducibility: the pseudonymisation pepper

Research ids are `sha256(pepper | Customer_Full_ID)[:20]`, and the patient-grouped split orders patients by those ids, so the same
pepper is needed for the same split. The first run on a file creates the pepper and stores it next to the input as
`<file>.id_pepper.txt` (e.g. `60k_falling_db.csv.id_pepper.txt`); every later run on that file reuses it automatically. Keep that file
with the extract, never share it, never commit it. Only its sha256 fingerprint appears in the reports (`split_audit.md`,
`readiness.json`), so two runs can be proven to share the pepper without exposing it. `--id-pepper-file` chooses another location,
`--id-pepper` gives the value explicitly (recorded as a warning when it differs from the stored one). Same file + same pepper + same
configs ⇒ same research ids, same partitions, same `test_rows_sha256`, same results.

### What STOPPED means

- **D-00 timing** (`... rows have predictor record dates on/after the index date`): a predictor source has records dated on the index day
  or later. Prediction happens at the *start* of the index day, so those records are unavailable. Read `audit/timing_diagnostic.md`: root
  cause A (every offset is 0 days ⇒ the VIEW's windows use `Event_Date <= Index_Date`), B (records after the index day ⇒ snapshot
  reconstruction / ETL defect or a misread field), the affected predictors per feature set, the label prevalence among affected vs clean
  rows, and the exact DWH predicate (`Event_Date < Index_Date`). Python cannot repair the aggregated counts. Either the DWH corrects the
  windows and every row is kept, or `--index-day-records drop_rows` excludes and counts the rows (the diagnostic tells you whether that
  removal is label-related).
- **INSUFFICIENT EVENTS FOR EXPLORATORY MODEL FIT**: a partition lacks the events the procedure needs (both classes everywhere; train ≥
  max(20, 2 × design columns) events; validation and test ≥ 10 events and ≥ 10 non-events). Nothing is fitted; no AUROC, calibration,
  odds ratio or ranking is written.
- **snapshot EXCLUDED** / **no temporal boundary**: see `audit/snapshot_audit.md`; the message names the arithmetic.
- **several index dates**: choose `--index-date` or `--all-index-dates`.

### What to send back

`READINESS.md`, `readiness.json`, `SUMMARY.md`, `comparison.md`, `comparison_patient_disjoint.md`, `temporal_test_by_snapshot.md`, (sensitivity package: `REAL_DATA_EXPLORATORY_REPORT.md`, `SENSITIVITY_COMPARISON.md`, `feature_provenance.md`, `convergence_audit.md`, `reference_integrity.json`, `split_audit_*.md`, `adequacy_*.md`, `feature_report_*.md/.csv`),
`feature_report_*.md/.csv`, `split_audit.md`, `adequacy_*.md`, everything under `audit/` (`wide_extract_audit.md`, `timing_diagnostic.md`,
`snapshot_audit.md/.csv`, `input_read_report.json`), and from each run directory `metrics.json`, `report.html`, `coefficients.csv`,
`feature_importance.csv`, `feature_stability.csv`, `calibration.csv`, `plots/`. Do **not** send `datasets/`, `id_pepper.txt`,
`*.id_pepper.txt` or `predictions_*.parquet` (row level).

## SAFE-ALL-ROWS sensitivity and ablations next to a completed run (v0.5.1)

When a single-snapshot run finished with `--index-day-records drop_rows` (rows excluded for D-00) and the DWH cannot yet be corrected, the
sensitivity package extracts what the current file still supports - without repairing anything the file does not contain:

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-sensitivity --input "C:\FallsData\60k_falling_db.csv" --reference "C:\FallsData\60k_falling_db_explore" --out "C:\FallsData\60k_falling_db_sensitivity"
```

`--input` must be the same file (sha256 checked), `--reference` the completed results folder (read-only: every file is hashed before and after,
`reference_integrity.json`), `--out` a new folder. The stored pepper next to the input is reused and must match the reference (`--id-pepper-file
<reference>\id_pepper.txt` if the sidecar moved). The run writes:

1. **Feature provenance** (`feature_provenance.md`): every EXTENDED predictor traced through the mapping (source columns, `record_date_column`,
   `same_day_evidence_column`) and the contract to the D-00 evidence measured on this file. A predictor is excluded from SAFE-ALL-ROWS when its
   event source has a last-record date on/after the index day (`falls` ← Last_Fall_Date, `mobility_problems` ← Last_Dx_Date on the first real
   extract); age and sex are derived from immutable attributes; registry predictors carry partial evidence only (`First_Registry_Date`, Q-M-11);
   the medication count is unverifiable from the file (stated, not repaired). The requested Phase-2 columns (`Diagnosis_Count_*`,
   `Prior_Fall_Count_*`, `Days_Since_Last_Fall`, ...) are traced too: none is a modelling predictor of any current analysis.
2. **Ablations on the reference cohort**: EXTENDED without each excluded predictor and without both, on the very same rows and test partition
   (hash-verified; the rebuilt datasets are also compared value by value with the reference dataset) - the *feature-set effect*.
3. **SAFE-ALL-ROWS**: every eligible row with a usable 180-day label, no exclusion because `Last_Dx_Date` / `Last_Fall_Date` equal the index date,
   only the temporally safe predictors (`timing_scope: built_predictors`). A row is excluded only when the file proves an index-day record of a
   source a built predictor reads (`First_Registry_Date` on/after the index date); the build report counts the D-00 rows kept
   (`cohort_guard_rows_retained`). Its split is redrawn on the new population: the comparison with the joint ablation (same 13 predictors) is a
   *population effect* and is **not paired** - the report says so.
4. **Convergence audit** (`convergence_audit.md`): for the reference runs and every new run, which warnings occurred in the final fit versus only in
   the stability / optimism replicates, where lambda* sits on the grid, whether the CV minimum was identified, whether the fold paths converged, and
   whether any solver / grid change is warranted (only when lambda* is on the grid boundary).
5. `SENSITIVITY_COMPARISON.md` (one table: rows, patients, events, prevalence, predictors included / excluded and why, split counts, AUROC, PR-AUC,
   Brier, calibration slope / intercept / CITL, O:E with 95% CIs, non-zero predictors, stability frequencies, convergence verdict, timing-related
   exclusions) with the paired feature-set effects and the unpaired population effect, and `FINAL_DWH_FIXED` documented as NOT RUN.
6. `REAL_DATA_EXPLORATORY_REPORT.md`: the plain-language report (14 sections) with every number taken from the local artifacts.

Runtime: four runs with the reference's own resampling settings (fast: about 2 minutes each on ~60k rows). Send back everything except `datasets/`,
`id_pepper.txt` and `runs/*/predictions_*.parquet`.

## The research workflow and the EDA layer (v0.6.0)

One coherent sequence; each step is a command of this package, run locally, and every shareable output is aggregate:

```
RAW EXTRACT -> CONTRACT CHECK -> FULL EDA -> LEAKAGE / TIMING AUDIT -> COHORT BUILD -> SPLIT -> TRAIN-ONLY SUPERVISED EDA -> ADEQUACY
   (meuhedet-eda covers all of these)
-> MODELLING -> VALIDATION -> FINAL TEST (meuhedet-explore; SAFE-ALL-ROWS / ablations: meuhedet-sensitivity)
-> MODEL LEARNING VISUALS -> INTERPRETABILITY -> MANAGEMENT REPORT -> SCIENTIFIC REPORT (model-report; built automatically at the end of both commands)
```

### Full EDA of the extract

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-eda --input "C:\FallsData\60k_falling_db.csv" --reference "C:\FallsData\60k_falling_db_explore" --out "C:\FallsData\60k_falling_db_eda"
```

`--reference` (recommended once a run exists) makes the EDA reproduce that run's split exactly: the input sha256, the pepper and the test-partition
hash are verified, and the reference folder is only read (hashed before and after). Without it the EDA draws the split a later `meuhedet-explore`
will draw (template + pepper sidecar); a D-00 stop then needs `--index-day-records drop_rows`. About 40 seconds on 66k rows.

What it does: (A) every column of the file is profiled without the outcome (data dictionary, types, missingness classes, quantiles, sentinels,
parse failures such as `00:00.0`, 250 data-quality checks, every date column against Index_Date, provenance source -> column -> feature -> model);
(B) every analysis that looks at the outcome and could inform a modelling decision reads the TRAIN partition only (univariate association,
missingness vs outcome, correlation / redundancy, Table 1 on TRAIN). Tables marked POST-HOC DESCRIPTIVE (outcome by stratum, Table 1 of the whole
modelling population) include test rows and are never used for model choice. Nothing is transformed, imputed or repaired.

Outputs: `REAL_DATA_EDA_REPORT.html` (navigate, filter and sort every table), `REAL_DATA_EDA_SUMMARY.md`, `DATA_QUALITY_REPORT.md`, `eda/*.csv`
(first line = watermark; `pd.read_csv(path, skiprows=1)`), `eda/plots/`, `eda_manifest.json` (hashes, split, privacy scan). The meanings come
from `configs/meuhedet/wide_v1_data_dictionary.yaml`: DOCUMENTED / NAME_ONLY / UNKNOWN - the last two need SME review, nothing is invented.
Send back the whole folder: the canonical dataset used for the split lived in a temporary folder and was deleted; no identifier, pseudonym, pepper
or row is written (the command scans every output and fails if one appears).

### Training dashboards and management reports

`meuhedet-explore` and `meuhedet-sensitivity` now finish by writing `<out>\reports\` (`--model-report full|reduced|off`, default full). For the
completed 0.5.x reference run, build them next to it (the reference is only read):

```bat
.venv\Scripts\python.exe -m falls_ml model-report --results "C:\FallsData\60k_falling_db_explore" --results "C:\FallsData\60k_falling_db_sensitivity" --eda "C:\FallsData\60k_falling_db_eda" --out "C:\FallsData\60k_falling_db_reports"
```

- `MODEL_TRAINING_REPORT_<analysis>.html` per run: cohort funnel and partitions; LASSO CV curve (lambda*, minimum, SE band, flags when the
  minimum is not identified or near the grid end); coefficient paths (recomputed on TRAIN with the saved preprocessor; checked to reproduce the
  saved model); FP shapes of age / medication count with their bootstrap stability; the learning curve (nested TRAIN subsets 10-100%, evaluated
  on VALIDATION; `reduced` fixes lambda at lambda* instead of re-running the CV per subset); ROC, PR (with the prevalence reference), risk
  distributions, calibration (uncalibrated vs recalibrated, slope / CITL / O:E / Brier with CIs); thresholds (no threshold chosen); gains and
  top 1/2/5/10/20% lift with bootstrap CIs; coefficients, stability, permutation importance, feature dictionary; subgroups; error analysis.
- `MODEL_COMPARISON_REPORT.html`: STRICT / EXTENDED / ablations / SAFE-ALL-ROWS; paired bootstrap differences only for identical test rows,
  every other population marked NOT A DIRECT PAIRED MODEL COMPARISON; FINAL_DWH_FIXED listed as not run.
- `MANAGEMENT_MODEL_REPORT_HE.html`, `MANAGEMENT_MODEL_SUMMARY_HE.md`, `EXECUTIVE_ONE_PAGER_HE.html` (printable A4): Hebrew, 8 figures,
  conclusions generated only from the artifacts, descriptive wording ("קשור לתחזית המודל").
- `figures\*.png` (200 dpi) + `*.svg`, `FIGURE_INDEX.md`, `tables\*.csv`, `model_report_manifest.json`.

Test-set rule: coefficient paths, FP shapes and the learning curve use TRAIN / VALIDATION only; held-out-test figures read the predictions the
frozen model produced when the run released the test set once. The reports describe the model; they are not an input for retuning it.

## The D-00 sensitivity framework (v0.7.0)

Question: how much of the reference EXTENDED result depends on prior-fall history and mobility, does useful discrimination remain on ALL
labelled patients with predictors proven safe from the D-00 timing issue, and does a small high-risk group still hold a large share of the falls?
One command, next to the completed reference run (it is only read; every file is hashed before and after):

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-d00 --input "C:\FallsData\60k_falling_db.csv" --reference "C:\FallsData\60k_falling_db_explore" --eda "C:\FallsData\60k_falling_db_eda" --reports "C:\FallsData\60k_falling_db_reports" --out "C:\FallsData\60k_falling_db_d00" --dependency-only
.venv\Scripts\python.exe -m falls_ml meuhedet-d00 --input "C:\FallsData\60k_falling_db.csv" --reference "C:\FallsData\60k_falling_db_explore" --eda "C:\FallsData\60k_falling_db_eda" --reports "C:\FallsData\60k_falling_db_reports" --out "C:\FallsData\60k_falling_db_d00_full"
```

The first line (minutes) writes only the dependency graph and the frozen plan; the second trains the plan. `--resampling reduced` (stability 50,
optimism 10, permutation 5; the confidence intervals keep their 1000 replicates) shortens the training when time is short.

1. **Dependency graph** (`D00_FEATURE_DEPENDENCY.md/.csv/.json`): every modelling feature is traced to the columns the builder itself reads
   (`feature_input_columns`), their data-dictionary source, declared VIEW-level derivations (followed transitively) and the record date that bounds
   the source; its D-00 status per cohort is SAFE (proven), UNSAFE (the extract proves an index-day record) or UNRESOLVED (the extract can neither
   prove nor exclude it: registries have only a first-entry date, medications no date). No outcome value, no test row, no name is used.
2. **Cohorts**: FULL_LABELED = every eligible row with a usable 180-day label (D-00 rows stay; unsafe predictors leave instead);
   D00_CLEAN = the reference cohort, rebuilt and verified row by row, value by value and test-partition hash against the reference.
3. **Feature sets from the graph**: STRICT_SAFE (renamed STRICT_SAFE_SUBSET with a recorded deviation when a STRICT predictor is not proven
   safe), SAFE_EXTENDED, FULL_EXTENDED (reference set; D00_CLEAN only). Primary standard = proven SAFE only; a secondary standard declared in
   `configs/meuhedet/d00_sensitivity.yaml` before any result also admits UNRESOLVED predictors (never UNSAFE).
4. **Matrix + ablations**: STRICT_SAFE / SAFE_EXTENDED on both cohorts, FULL_EXTENDED on D00_CLEAN (the reference run, reused), and EXTENDED
   without falls / without mobility_problems / without both on the reference rows and test partition. Identical specifications share one run.
   FINAL_DWH_FIXED is listed as not run. The plan is frozen in `ANALYSIS_PLAN.json` (sha256) before any fit.
5. **Comparisons**: feature-set effects are paired (identical test keys verified; paired bootstrap for AUROC, PR-AUC, Brier and top-5/10/20%
   capture); population effects are labelled NOT A DIRECT PAIRED MODEL COMPARISON; risk concentration for the top 1/2/5/10/20% of every model.
6. **Learning curves and warnings**: the STRICT / EXTENDED learning curves are recomputed with each refit's LASSO diagnostics (checked against
   the 0.6.0 report tables), with paired 40->60->80->100% increments and a pre-declared plateau rule; `LASSO_WARNINGS_AUDIT.md` separates the
   final fits, the bootstrap replicates and the learning-curve refits, and measures lambda sensitivity on VALIDATION.

Send back ONLY `<out>\share` (aggregate; scanned for member ids, snapshot keys, pseudonyms and the pepper). Never send `datasets\`, `runs\`
(row-level predictions), `configs\`, `id_pepper.txt` or the CSV. `share\management\` is a NEW management report (the 0.6.0 one is untouched)
with the permutation-importance figure ("תרומה לחיזוי המודל") next to the selection-stability figure, and the D-00 comparison.

### Resuming an interrupted D-00 analysis; bootstrap convergence retry (v0.7.1)

If `meuhedet-d00` stops before the end (for example a cell's bootstrap stability exceeded the 10% failed-replicate limit, or the computer
restarted), do NOT start again in a new folder. Run the SAME command with the SAME `--out` plus `--resume`:

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-d00 --input "C:\FallsData\60k_falling_db.csv" --reference "C:\FallsData\60k_falling_db_explore" --eda "C:\FallsData\60k_falling_db_eda" --reports "C:\FallsData\60k_falling_db_reports" --out "C:\FallsData\60k_falling_db_d00" --resume
```

- The plan is re-derived and must have the frozen plan's sha256; `--resampling`, `--learning-curve` and the secondary standard are taken from
  the frozen `ANALYSIS_PLAN.json` (a different explicit value is refused).
- A cell whose run COMPLETED (`RUN_COMPLETE.json` + `metrics.json`, exactly one test release in `runs\test_evaluation_registry.jsonl`, the same
  config, dataset and test partition - all verified) is reused as it is: never refitted, its test set never re-evaluated, its files never touched.
  Its dataset and config are regenerated in a temporary folder and must be identical to the earlier ones.
- A run directory without `RUN_COMPLETE.json` is never read as a result; it stays on disk for audit (listed in `share\RESUME_LOG.json` with
  the reason it stopped) and its cell is refitted - but only if the registry shows it never released its test set (otherwise: stop).
- Every file of the earlier attempt (`runs\`, `datasets\`, `configs\`, `id_pepper.txt`) is hashed before and after; the registry may only be
  appended to (`reference_integrity.json` -> `earlier_attempt`).

**Bootstrap convergence retry (pre-declared policy v1).** When the full-data LASSO solve of a bootstrap stability or optimism replicate does not
converge at the selected lambda (`LassoConvergenceError`), the SAME resample is refitted ONCE with the SAME seed and parameters, only the full-data
path's iteration limit raised 10x (10,000 -> 100,000 passes). The cross-validation folds, the lambda grid, the CV curve, the selected lambda and
the tolerance (1e-7) are unchanged. The retry is accepted only if the refitted model reports convergence at the same lambda under the raised limit;
otherwise the replicate is a failed replicate exactly as before, and more than 10% failed replicates still stop the run. Every initial failure and
its outcome is in `runs\<run>\bootstrap_convergence_retries.csv`, `metrics.json` -> `bootstrap_convergence_retry`, the run log and
`LASSO_WARNINGS_AUDIT.md`. Not applied to the final model, learning-curve refits or internal-external CV. Runs completed before v0.7.1 keep their
results and are marked "policy not in effect".

## Phase 2: feature discovery and model families (v0.8.1, FINAL configuration)

`meuhedet-phase2` runs next to the completed explore run on the SAME extract.

- It builds the column registry (221 columns) and the engineered features (93), with SAFE / UNSAFE / UNRESOLVED provenance (unchanged D-00 rules).
- It screens on TRAIN only and freezes the feature sets in three categories that are never mixed:
  - HISTORICAL_BASELINE_15 (the BASELINE_15 refit);
  - SAFE_DISCOVERY (primary; proven-SAFE features only);
  - EXPLORATORY_UNRESOLVED_SENSITIVITY.
- It compares LASSO, elastic net and XGBoost (Optuna) in nested cross-validation inside TRAIN.
- It freezes the recommendation (`SELECTION_FROZEN.json`) BEFORE VALIDATION is opened once (`VALIDATION_OPENED.json`, write-once).
- It then runs stability, ablation, SHAP and the consensus, and writes reports plus one share folder.
- The TEST partition is dropped right after the split check and never used.

References:
- Plan: `planning/EXPERIMENT_PLAN.md`.
- Frozen effective configuration: `configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json`. The run refuses any other effective configuration.

**One-time install** (unchanged from 0.8.0; `xgboost-cpu 3.2.0`, Optuna 5.0.0) from the hash-pinned lock:

```bat
.venv\Scripts\python.exe -m pip install --disable-pip-version-check --no-input --require-hashes --only-binary=:all: -r requirements.lock
```

**1. Preflight (mandatory, 3–6 min).** It fits no model and writes nothing to `--out` (it does not even create it). It reads the input,
hashes it, rebuilds the reference cohort and split in memory, checks the provenance and the category invariant, and prints the exact
counts, budgets, paths and hashes. The last line must be exactly `SAFE TO START FULL RUN`.

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase2 --preflight --input "C:\FallsData\60k_falling_db.csv" --reference "C:\FallsData\60k_falling_db_explore" --protect "C:\FallsData\60k_falling_db_eda" --protect "C:\FallsData\60k_falling_db_reports" --protect "C:\FallsData\60k_falling_db_d00" --out "C:\FallsData\60k_falling_db_phase2"
```

**2. Run.** Use the same command without `--preflight`, into a NEW local folder (not inside OneDrive). The reference, EDA, report and D-00
folders are protected and hashed at every stage.

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase2 --input "C:\FallsData\60k_falling_db.csv" --reference "C:\FallsData\60k_falling_db_explore" --protect "C:\FallsData\60k_falling_db_eda" --protect "C:\FallsData\60k_falling_db_reports" --protect "C:\FallsData\60k_falling_db_d00" --out "C:\FallsData\60k_falling_db_phase2"
```

**3. Resume** (after a restart, power loss or any stop): the SAME command plus `--resume`. Nothing completed is recomputed or changed;
the interrupted item is moved to `_incomplete\` and redone.

**4. Progress (read-only; safe while the run works):**

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase2-status --out "C:\FallsData\60k_falling_db_phase2"
```

It shows:
- the status and whether the process is alive;
- the current and last completed stage (of 17), with the item;
- times, XGBoost trial counts and the latest small result;
- the frozen selection and whether VALIDATION is opened;
- open gates and whether the share is ready.

**Duration:** roughly 5–10 hours. The full-size synthetic benchmark measured 3.2 h on a fast Mac with 80% of the rows. The final configuration adds two waterfall steps, two exploratory sets and calibration intervals (about +10–15%), and drops XGBoost stage 2, which the benchmark had skipped anyway. Keep the computer plugged in. Sleep only pauses the run; a restart is harmless (resume).

**Stops**
- **HARD stop** (`STOPPED.md`): fix the cause, then resume.
- **Investigation stop** (`INVESTIGATION_<GATE>.md`): nothing was rejected. Read the file and the named aggregate tables; if the evidence
  is acceptable, continue with a recorded reason: `--resume --accept-gate <GATE> --reason "<why>"`. The gates are never accepted
  automatically.
- **Code change during a run** (for example a later patch): resume refuses with CODE_CHANGED. Continue only with
  `--resume --accept-code-change "<reason>"`; the reason is recorded in `RESUME_AUDIT.json` and `logs\gates.jsonl`.

**Send back ONLY `<out>\share`** (zip it). It appears only after the privacy and path scan passed. `stages\`, `work\` and `checkpoints\`
hold row-level or derived files and stay on this computer.

## 0. Export the extract (your SQL client, outside this package)

One row per member for **one** index date, all 221 columns of the VIEW, no eligibility filter (the audit reports the exclusion
reasons and the build applies `Is_Eligible_Cohort = 1` itself):

```sql
SELECT * FROM [Meuhedet_DWH].[dbo].[V_Falls_Prediction_Wide_1] WHERE Index_Date = '2025-01-01';
```

Save it **outside** the package folder (e.g. `D:\falls_ml_data\meuhedet\wide_2025-01-01.csv`) as either
- Parquet (preferred, keeps types), or
- CSV, UTF-8, comma separated, header row, SQL `NULL` written as the literal `NULL` (SSMS "Save results as" writes this; if the
  file is UTF-16 or cp1255 pass `--encoding utf-16` / `--encoding cp1255`; tab separated: `--sep "\t"`).

Column names must match the VIEW exactly; the contract refuses missing or extra columns (a changed VIEW needs a contract update).

### Dates in the extract (v0.4.3, unchanged in v0.5.0)

Date columns are parsed from the contract, not inferred. Text dates are read with the layouts declared in `contract.date_formats` —
`%Y-%m-%d` and `%d/%m/%Y`, tried in that order — so an SSMS export in the Israeli layout works as it is: `01/02/2024` is always
1 February 2024, never 2 January. A time part (`01/02/2024 14:30:00`) is kept; real Excel and Parquet date cells are used as they are.
Anything no declared format fits is counted as `wrong_type` and the offending value is printed next to the column name and the accepted
formats (never an identifier). The audit reports which layout read each column under `datatype_audit.source_date_formats`.
If the extract holds several index dates, the run stops and lists the parsed dates with the `--index-date` to choose from.

Two columns declare the far-future sentinel `2999-12-31`: `Population_Record_To_Date` (open-ended SCD row, kept as a date) and
`Followup_End_Date` (open follow-up with no known end; the canonical `followup_end_date` is NULL for those rows, which is this
column's declared value for an open follow-up, so nothing reads it as a follow-up that ended in 2999; the count appears as
`sentinel_to_null` in the build report). No other date column assumes a meaning for `2999-12-31`: if one holds it, the audit lists
it under `datatype_audit.date_range` as an undeclared far-future value and the run stops if the build uses that column. Send that
section before declaring anything further. The whole SQL date range is kept (`datetime64[us]`), so an open-ended sentinel such as `Population_Record_To_Date = 2999-12-31`
stays a date and is reported in `audit/input_read_report.json` → `date_range` (declared sentinels vs undeclared far-future dates). A far-future
date in a column the build maps into the modelling dataset (index, event, death, follow-up, Last_Fall/Last_Dx dates) stops the run with the
column, the value and the expected type; declare it in `configs/meuhedet/wide_v1_columns.yaml` (`sentinels` / `sentinel_means` /
`sentinel_modeling_rule`) only after confirming its business meaning. Never edit the source extract.

## 1. Aggregate audit (first deliverable — send this for review)

```bat
%PY% -m falls_ml meuhedet-audit --input "D:\falls_ml_data\meuhedet\wide_2025-01-01.csv" --out "D:\falls_ml_data\meuhedet\audit_2025-01-01" --index-date 2025-01-01
```

Outputs `wide_extract_audit.md` and `.json`: row/patient counts, index dates, eligibility and exclusion reasons, outcome counts and
prevalence (30/180 days, label reasons × deaths, window lengths), per-column dtype/NULL%/distribution/quantiles, wrong-type and
invalid-value counts, eFalls coverage with the source distribution of every baseline predictor, the 365-day-outcome verdict and
the leakage checks. Cells below 10 are suppressed (`--min-cell`). No identifiers or rows are written.

Review the audit **before** building: confirm Q-M-01…Q-M-07 in `docs/meuhedet/MAPPING_TABLE.md` (Gender_Code 2 = female,
SMI levels, blood-pressure registry, Siudi codes, fall-event list, substance-count window) and set
`validation.outcome_lag_days` in the experiment config from `Max_Invoice_Lag_365D`.

## 2. Build the canonical dataset (one index date, eligible rows only)

```bat
%PY% -m falls_ml meuhedet-build --input "D:\falls_ml_data\meuhedet\wide_2025-01-01.csv" --out "D:\falls_ml_data\meuhedet\ds_2025-01-01_v1" --index-date 2025-01-01 --dataset-version meuhedet-wide-2025-01-01-v1 --data-freeze-date 2025-09-15 --report "D:\falls_ml_data\meuhedet\ds_2025-01-01_v1_build_report.json"
```

`--data-freeze-date` is the date the VIEW's sources were last loaded (label maturity, D-19). The build stops on: wrong-type values in
any used column, NULL in DDL NOT NULL columns, unknown sex codes, NULL predictor values without a declared eFalls rule, event dates
outside the label window, `Leakage_Check_Ind = 1`, and predictor record dates on/after the index date (add
`--index-day-records drop_rows` to remove and count those rows instead). Every NULL→0 conversion and every dropped row is counted
in the build report and in the dataset manifest (`audit.meuhedet_build`).

For a reportable scientific run add `--scientific-use-allowed --approval-reference <governance id>` (the experiment template
requires it; the fixture variant does not).

## 3. Exploratory baseline run — "Meuhedet 180-day exploratory — NOT eFalls reproduction"

```bat
%PY% -m falls_ml train --config configs\experiments\meuhedet\phase1_180d_exploratory_efalls_reduced.yaml --dataset "D:\falls_ml_data\meuhedet\ds_2025-01-01_v1" --runs-dir "D:\falls_ml_data\meuhedet\runs"
```

Outputs under the run directory: `report.html/.md`, `metrics.json` (AUROC, PR-AUC, Brier, calibration intercept/slope, O:E,
sensitivity/specificity/PPV/NPV/F1 per threshold, decision curve), `coefficients.csv` (LASSO coefficients and odds ratios),
`feature_importance.csv` (permutation importance), `feature_stability.csv` (bootstrap selection frequency), `unpenalized_refit.csv`
(descriptive refit, kept separate), calibration plots, and the model bundle. The report carries the labels
**REDUCED eFalls predictor set – NOT a full eFalls reproduction** and **EXPLORATORY OUTCOME – NOT an eFalls reproduction**.

## 4. Freeze the baseline

```bat
%PY% -m falls_ml freeze-baseline --run "D:\falls_ml_data\meuhedet\runs\<run directory>" --name EFALLS_BASELINE_MEUHEDET_V1 --baselines-dir "D:\falls_ml_data\meuhedet\baselines"
```

`baseline.json` records the dataset hash, index date, feature-spec / mapping-manifest / contract hashes, outcome definition, model
config, selected lambda, coefficients, test metrics and the test-row hash; the directory is immutable.

## Synthetic dry run (no real data; any machine)

```bat
%PY% -m falls_ml meuhedet-make-fixture --out demo_outputs\meuhedet\wide.parquet --n-rows 4000
%PY% -m falls_ml meuhedet-audit --input demo_outputs\meuhedet\wide.parquet --out demo_outputs\meuhedet\audit --index-date 2025-01-01
%PY% -m falls_ml meuhedet-build --input demo_outputs\meuhedet\wide.parquet --out demo_outputs\meuhedet\ds --index-date 2025-01-01 --dataset-version synthetic-wide-0.1.0 --data-freeze-date 2025-09-01
%PY% -m falls_ml train --config configs\experiments\fixture\meuhedet_180d_exploratory_efalls_reduced.yaml --dataset demo_outputs\meuhedet\ds
```

SYNTHETIC DATA – NOT SCIENTIFIC RESULTS.
