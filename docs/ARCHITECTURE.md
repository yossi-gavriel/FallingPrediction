# Architecture and implementation plan — `falls_ml`

Status: Phase 2 deliverable (2026-09-14). This plan is deliberately small. The scientific decisions it implements live in [`EFALLS_REPRODUCTION_SPEC.md`](EFALLS_REPRODUCTION_SPEC.md) (referenced as `D-xx`, `M-xx`, `RT-xx`).

## 1. Principles

1. **Two layers, one boundary.**
   - `falls_ml.dataeng` turns event-level tables into a *versioned, immutable modelling dataset* (Parquet + manifest).
   - Everything else, the ML layer, consumes only that dataset.
   - ML modules never import `dataeng`, never open database connections, and never read raw event tables. A test enforces this.
2. **Configuration over code.**
   - Clinical definitions (feature spec, published coefficients, code-list mappings) and experiment decisions (split, tuning, calibration, thresholds) live in YAML under `configs/`.
   - Python contains algorithms only.
3. **One entry point.** `run_experiment(model_name, dataset, experiment_config)` orchestrates small, independently tested components.
4. **One preprocessing implementation.**
   - The same fitted `Preprocessor` object is used for evaluation and production.
   - It is persisted inside the model bundle and fingerprinted.
   - Inference refuses a bundle whose fingerprint or feature-schema hash does not match.
5. **Timing (spec D-00):** prediction at the start of the index day. Predictors use records strictly before `index_date`; outcomes use `index_date ≤ event ≤ index_date + 1 year − 1 day`; temporal embargo requires a training outcome window to end before the next cohort's index date.
6. **Leakage prevention in the structure itself:**
   - the test split is held by a guard object and released once, after model selection is frozen;
   - preprocessing, tuning and calibration receive only the rows they are allowed to see.
7. **Determinism.**
   - Every random component draws from `numpy.random.Generator(SeedSequence(seed, spawn_key=<component>))`. There is no global RNG state.
   - All seeds and library versions are written to the run directory.
8. **Fail loudly.**
   - Typed exceptions (`DatasetValidationError`, `LeakageError`, `ConfigError`, `BundleIntegrityError`).
   - No silent row or feature dropping; every exclusion is counted and logged.
9. **Established libraries only:** numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, PyYAML, pyarrow, pytest. There are no orchestration frameworks and no notebooks as source of truth.

## 2. Package layout

```text
configs/
  features/efalls_v1.yaml            # machine-readable predictor + outcome spec (layer-tagged)
  models/efalls_published.yaml       # published coefficients, sex-parameterisation variants, recalibration reference
  mappings/meuhedet_v0.yaml          # [MEU] per-feature mapping status (TO_BE_MAPPED; clinically_validated: false)
  meuhedet/wide_v1_columns.yaml      # [MEU] V_Falls_Prediction_Wide_1 column contract: role, SQL/semantic/model type, NULL meaning
  meuhedet/wide_v1_efalls_mapping.yaml  # [MEU] eFalls candidate -> wide-table column mapping (status/quality, ops, include_in_baseline)
  features/efalls_v1__outcome_fall_next_180d_exploratory.yaml  # generated: efalls_v1 predictors + Meuhedet 180-day EXPLORATORY outcome
  experiments/meuhedet/*.yaml        # Phase-1 Meuhedet templates (efalls_retrained_reduced on the exploratory outcome)
  experiments/*.yaml                 # one file per experiment
src/falls_ml/
  __main__.py, cli.py                # python -m falls_ml <command>
  config.py                          # YAML → frozen dataclasses, validation, config hash
  errors.py, logging_utils.py        # typed errors; JSON structured logging
  data/
    schema.py                        # DatasetSchema derived from the feature spec; validate()
    dataset.py                       # ModelingDataset (read-only DataFrame + manifest; hash verification)
    adapters.py                      # ExternalDatasetAdapter: column/recode maps (SAIL/CB/Meuhedet/fixture)
    synthetic.py                     # deterministic SYNTHETIC fixtures (software tests only)
    meuhedet_wide.py                 # MeuhedetWideDatasetAdapter: contract validation, eFalls mapping ops, canonical frame, build
    meuhedet_audit.py                # aggregate, non-identifying extract audit (small-cell suppression)
    meuhedet_synthetic.py            # SYNTHETIC 221-column wide-table extract
  baseline.py                        # freeze-baseline: immutable named record of a completed run
  meuhedet_explore.py                # meuhedet-explore: Excel/CSV/Parquet -> audit -> STRICT/EXTENDED datasets -> two runs -> reports
  dataeng/
    derive.py                        # reference feature/outcome derivation from event tables (D-00, D-03, D-08, D-09)
  features/
    spec.py                          # FeatureSpec / FeatureDefinition dataclasses
    transforms.py                    # pure transforms: BMI category, smoking, alcohol, ln((P+1)/10), FP basis
    fractional_polynomial.py         # Stata-compatible FP scaling + mfp closed-test selection (D-13)
    preprocessing.py                 # Preprocessor: fit(train) / transform / fingerprint / state
  models/
    base.py                          # ModelAdapter ABC
    registry.py                      # name → adapter class
    published_efalls.py              # fixed published equation (LP_A/LP_B/LP_C variants, D-01)
    lasso_cv.py                      # Stata-equivalent LASSO-logit path + 10-fold CV (D-14)
    logistic.py                      # unpenalised logistic; elastic-net logistic
    random_forest.py, hist_gbm.py    # secondary experimental models
  splitting.py                       # temporal / group-holdout / patient-grouped splits; TestSetGuard
  tuning.py                          # bounded grid/random search with configurable composite objective
  evaluation/
    metrics.py                       # AUROC, PR-AUC, Brier, CITL, slope, O/E, grouped/smoothed calibration, thresholds, NB
    bootstrap.py                     # (cluster) bootstrap CIs, paired bootstrap differences
    calibration.py                   # recalibration: intercept-only, intercept+slope (fit on validation only)
    meta_analysis.py                 # REML + HKSJ + prediction interval (D-16)
    importance.py                    # coefficient tables, permutation importance
    stability.py                     # bootstrap refits: selection frequency, coefficient stability, MAPE, instability
    optimism.py                      # Harrell bootstrap optimism (D-11)
    iecv.py                          # internal–external CV by cluster
    sample_size.py                   # Riley 2019/2021 criteria (RT-09, RT-10)
  experiment.py                      # run_experiment orchestrator + ExperimentResult
  artifacts.py                       # run directory layout, environment capture, writers
  bundle.py                          # ModelBundle save/load with SHA-256 integrity
  inference.py                       # predict_risk → PredictionResult
  monitoring.py                      # reference profile, drift report (never retrains)
  reporting/
    plots.py, report.py              # per-run report.md + report.html
    compare.py                       # master comparison across runs
    best_features.py                 # reports/best_features.csv|md
    ablation.py                      # nested feature-group ablations
tests/
  unit/ leakage/ integration/ regression/
```

## 3. Contracts

### 3.1 Modelling dataset (produced by data engineering, consumed by ML)

The dataset is a Parquet file plus `manifest.json`. Required columns (the schema is derived from `configs/features/efalls_v1.yaml`):

| Column | Type | Rule |
|---|---|---|
| `research_id` | string | Irreversible research identifier; **never a predictor** |
| `index_date` | date | |
| `outcome_12m` | int8 ∈ {0, 1} | Non-null |
| `predictor_max_record_date` | date | < `index_date` (D-00) |
| `outcome_first_event_date` | date, nullable | Required in training mode: non-null iff outcome = 1, and `index_date ≤ date ≤ index_date + 1 year − 1 day` |
| predictor columns | per feature spec | Binary int8 {0, 1} non-null; `age_years` float ≥ cohort minimum; `sex` ∈ {female, male}; `polypharmacy_count_120d` int ≥ 0; `bmi_value` float nullable; `smoking_status` ∈ {never, ex, current} nullable; `alcohol_category` ∈ published levels nullable |
| optional metadata | declared in spec | `practice_id`, `deprivation_group`, `site_id`, `death_date`, `followup_end_date` |
| `dataset_version`, `mapping_version`, `source` | string | Constant; must equal the manifest |

**Validation fails if any of the following holds:**
- a required column is missing;
- an undeclared extra column is present;
- a (`research_id`, `index_date`) pair is duplicated;
- an outcome value is illegal;
- age is outside the cohort range;
- a dtype or category value falls outside the declared set;
- a null appears in a non-nullable column;
- `predictor_max_record_date > index_date`;
- an outcome date is inconsistent;
- the manifest data hash, feature-spec version or feature-spec content hash does not match;
- an identifier or metadata column is referenced as a predictor.

### 3.2 Model adapter

```python
class ModelAdapter(ABC):
    name: ClassVar[str]
    representation: ClassVar[str]                  # which Preprocessor representation it consumes
    def __init__(self, params: Mapping[str, Any] | None = None, *, random_state: int = 0): ...
    def fit(self, X: pd.DataFrame, y: np.ndarray, *, groups: np.ndarray | None = None) -> Self: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...          # shape (n,), P(outcome = 1)
    def linear_predictor(self, X: pd.DataFrame) -> np.ndarray | None: ...  # None for non-linear models
    def get_feature_importance(self) -> pd.DataFrame: ...                # native importance/coefficients
    def fit_diagnostics(self) -> dict[str, pd.DataFrame | dict]: ...     # e.g. LASSO CV path
    def save(self, directory: Path) -> None: ...
    @classmethod
    def load(cls, directory: Path) -> Self: ...
    @classmethod
    def default_search_space(cls) -> dict[str, list] | None: ...
```

Adding an algorithm means writing one adapter and registering it. The pipeline does not change.

### 3.3 Preprocessor

- **Constructor:** `Preprocessor(feature_spec, representation, options)`.
- **Representations:**
  - `efalls_published` — deterministic published terms, published reference levels (D-02, D-04).
  - `efalls_fp_all_levels` — FP-selected continuous terms plus all-levels indicators (D-10, D-13).
  - `efalls_reference_coded` — published forms, reference-coded, for standard/elastic-net models.
  - `efalls_raw` — raw age, raw polypharmacy count and indicators, for tree models.
- **Methods:** `fit(train_df)`, `transform(df) → design DataFrame`, `fingerprint() → sha256`, `state() → JSON-serialisable`.
- **Category levels:** fixed by the spec, never learned from data.

### 3.4 Run directory (every run is self-contained)

```text
runs/test_evaluation_registry.jsonl            # append-only log of every test-partition release (D-19 §4)
runs/<YYYY-MM-DD>_<label>_<run8>/                 # label: see below
  config.yaml  dataset_manifest.json (+ dataset_path_used, dataset_path_relative)  run_inputs.json (resolved paths + sha256)
  environment.json  run_log.jsonl  splits.csv
  metrics.json  cv_results.csv  hyperparameter_results.csv  coefficients.csv (_intercept row, omitted flag)
  unpenalized_refit.csv  fp_selection.csv  (when produced)
  predictions_validation.parquet  predictions_test.parquet
  feature_importance.csv  feature_stability.csv  instability.csv  calibration.csv  confusion_matrices.csv
  decision_curve.csv  subgroup_metrics.csv  optimism.csv  iecv_*.csv  heterogeneity_*.csv  (when enabled)
  model/  (bundle.json, preprocessor.pkl, model files, calibrator.json when served, reference_profile.json)
  plots/*.png  report.md  report.html
  RUN_COMPLETE.json                            # written last; readers ignore runs without it
```

**Run id** (`artifacts.create_run_directory`): `<YYYY-MM-DD>_<label>_<run8>`.

- `label` (`artifacts.run_label`) is `<experiment.name>_<model.name>`, or just `<experiment.name>` when the experiment name already contains the model name (model overrides `<name>__<model>`, ablation steps such as `logistic_unpenalized__cognition`, `efalls_published_scoring` with model `efalls_published`). It is capped at 60 characters (`RUN_LABEL_MAX_CHARS`, trailing `_`, `.`, `-` stripped) so run folders stay far below the Windows 260-character path limit.
- `run8` is the first 8 hex characters of SHA-256(config hash | dataset hash | `source_tree_sha256` | UTC timestamp), plus a nanosecond clock reading when no explicit timestamp is given. It keeps truncated ids unique.

**Dataset location.** `dataset_manifest.json` records `dataset_path_used` (the absolute directory as resolved on the machine that ran the experiment) and `dataset_path_relative` (the same directory relative to the project root that contains the run directory, found by walking up to the first folder with `configs/`; `/` separators; `null` when the dataset lies outside that project). `reproduce` without `--dataset` tries `dataset_path_relative` first, resolved from the run directory, so a moved or copied project folder still works; then `dataset_path_used`.

**Code version** (`environment.json` and `metrics.json` `code_version`) always records both fields:

- `source_tree_sha256`: SHA-256 over the relative `/`-separated paths and contents of every `*.py` file of the `falls_ml` package (`src/falls_ml`, `__pycache__` skipped), ordered by case-sensitive path components, with CRLF hashed as LF. It is identical on Windows and macOS/Linux, needs no git, and is the value used in `run8` and in the bundle compatibility check.
- `git_commit`: `HEAD` (suffix `-dirty` when the work tree has changes) only when git is available and `git rev-parse --show-toplevel` is the project folder itself (the parent of `src/`). Otherwise `null`, for example in the handoff folder without git, or when that folder is copied inside an unrelated repository, whose commit must not be recorded. Reports show `git_commit` when present, else `source_tree_sha256`.

### 3.5 Experiment kinds

| Kind | Model / features | Label |
|---|---|---|
| `efalls_published_scoring` | fixed published equation, all 78 predictor columns required (unavailable binaries declared and fixed at 0, M-11) | scientific reproduction (transportability) |
| `efalls_retrained` | published learning process (FP + LASSO, λ-min, 10-fold CV, no tuning, embargo), all 78 candidates, `preprocessing.features` unset | local retraining of eFalls |
| `efalls_retrained_reduced` | same learning process on an explicit STRICT subset of the 78 candidates (§3.6) | local retraining on a reduced eFalls predictor set — never a full eFalls reproduction |
| `alternative_model` | any registered adapter (L4) | experimental algorithm |
| `ablation_member` | generated by `run_ablation` | ablation |

### 3.6 Reduced eFalls predictor set

When the data-availability audit shows that some eFalls candidates cannot be extracted, `configs/experiments/efalls_retrained_reduced.yaml` declares the AVAILABLE predictors explicitly in `preprocessing.features` (the shipped list is illustrative: all Meuhedet mappings are still TO_BE_MAPPED).

- Config rules (`config._validate_semantics`): everything required of `efalls_retrained` (model `lasso_logistic_cv`, representation `efalls_fp_all_levels`, FP `select|fixed_published`, whitelisted LASSO params, λ-min, 10 folds, no tuning, embargo, no extensions, no L3b/L4 layers), plus a non-empty duplicate-free `preprocessing.features` that includes `age_years` and every FP variable.
- Spec rules (`config.check_feature_declaration`, before any data are read or a run directory exists): every name must be an `exact_efalls_baseline` candidate, and the set must be a STRICT subset — all 78 is refused with advice to use `efalls_retrained`.
- Datasets: built with the full spec or with exactly the declared subset spec (`FeatureSpec.subset`, root-based and idempotent); the manifest `feature_spec_sha256` decides. `make-fixture --features-from-config` writes a subset fixture with the same rows as the full fixture (`data/fixtures/synthetic_reduced_v1`); `build-dataset --features-from-config` validates an extract against the subset spec.
- Published scoring and `efalls_retrained` stay strict: a subset dataset fails loudly with both expected hashes listed.
- Comparison category `reduced_local_retraining` with an "eFalls coverage" column; `is_efalls_retrained` is false; best-features never uses a reduced run as local eFalls LASSO evidence.

## 4. `run_experiment` flow

1. Load and validate the config (all semantic and eFalls-purity rules), resolve declared paths (§ `falls_ml.paths`), check the declared feature set against the feature spec (`check_feature_declaration`, §3.6), load and validate the dataset (§3.1; full or declared-subset spec chosen by the manifest `feature_spec_sha256`), and run pre-flight checks: scientific-use approval, configured columns present (only what the config reads: bootstrap/IECV/heterogeneity cluster columns, subgroup sources `sex`→`sex`, `bmi_category`→`bmi_value`, `age_band`→`age_years`, and `sex` for published scoring), label maturity against `data_freeze_date`. Only then open the run directory, attach the run log, and record resolved inputs and the environment.
2. Build the split plan (`temporal` preferred; `group_holdout`; `patient_grouped_random` with a documented limitation). Assert that patients are disjoint where required, that no row key appears in two partitions, and that temporal outcome windows have matured (embargo). Check the test-evaluation registry (a scientific dataset refuses a second evaluation with the same config). Wrap the test rows in a `TestSetGuard`; validation and test row keys are forbidden in every `fit_pipeline` call.
3. Fit the `Preprocessor` on **train only**.
4. Hyperparameter search (if enabled) on train/validation, or patient-grouped CV within train. The objective is configurable (e.g. AUROC, Brier, calibration-slope penalty, net benefit).
5. Fit the final adapter on train. Record diagnostics (e.g. LASSO λ path, fold deviances, λ\*, seed).
6. Validation predictions → calibration assessment → optional recalibration **fitted on validation only**.
7. Feature analysis: native coefficients/importance; permutation importance on validation; bootstrap stability on train (selection frequency, coefficient mean/median/SD/sign stability, MAPE). Resampling refits use the frozen (tuned) parameters of the final model.
8. Optional published-protocol analyses: bootstrap optimism (B = 25 or 200), IECV by cluster.
9. Append the registry entry, then release test data from the guard (once). Evaluate every variant on train (apparent), validation and test with bootstrap CIs, thresholds, decision curves and subgroup metrics; optional cluster heterogeneity on test.
10. Persist predictions, tables, the model bundle (serving `served_variant`), `metrics.json` (atomic), plots, `report.md`/`report.html`, and finally `RUN_COMPLETE.json`. Return an `ExperimentResult`.

**eFalls coverage.** Runs of the three eFalls kinds record `metrics.efalls_coverage` (available/unavailable eFalls candidates, coverage %, published-retained and mandatory predictors; `docs/ARTIFACT_SCHEMAS.md`). Reports show "Available eFalls predictors: X / 78", "Coverage percentage" and the unavailable list; reduced runs carry the warning "REDUCED eFalls predictor set – NOT a full eFalls reproduction".

**Served variant.** `metrics.primary_variant` equals the variant the bundle serves. Eligibility and calibration status are judged on it; the uncalibrated status is reported separately. For `efalls_published_scoring`, step 5 is a no-op (fixed coefficients) and the equation is served uncalibrated by default. The report states *transportability of the published model*. All sex-parameterisation variants are scored, with `lp_c_box_s3_1` primary (D-01).

## 5. CLI

```bash
python -m falls_ml make-fixture --out data/fixtures/synthetic_v1 --n-patients 3000
python -m falls_ml make-fixture --out data/fixtures/synthetic_reduced_v1 --n-patients 3000 \
    --features-from-config configs/experiments/fixture/efalls_retrained_reduced.yaml    # subset fixture, same rows
python -m falls_ml validate-dataset --dataset data/fixtures/synthetic_v1   # full or subset spec selected from the manifest
python -m falls_ml validate-dataset --dataset <dir> --features-from-config <experiment.yaml>
python -m falls_ml build-dataset --input <extract.parquet|.csv> --out <new dir> --dataset-version V --mapping-version V \
    --source TEXT --data-freeze-date YYYY-MM-DD [--feature-spec ... --extension ... | --features-from-config <experiment.yaml>] \
    [--scientific-use-allowed --approval-reference TEXT]   # strict validation of a flat extract file; never opens a database
python -m falls_ml train --config configs/experiments/efalls_retrained_lasso.yaml --dataset <dir>
python -m falls_ml evaluate --run-dir runs/<run>          # re-render report from artifacts
python -m falls_ml reproduce --run-dir runs/<run>         # verify definitions unchanged, re-run into runs_reproduced/, compare
python -m falls_ml ablation --config configs/experiments/fixture/ablation_example.yaml --dataset <dir>   # runs_ablation/, reports/ablation/
python -m falls_ml compare --runs-dir runs --out reports  # reads reports/ablation/ for best-features Q7
python -m falls_ml predict --model runs/<run>/model --input <parquet> --out <parquet>
python -m falls_ml monitor --model runs/<run>/model --input <parquet> [--labels]
python -m falls_ml meuhedet-make-fixture --out <wide.parquet|.csv> [--n-rows N --index-day-falls K]   # SYNTHETIC wide-table extract
python -m falls_ml meuhedet-audit --input <extract> --out <dir> [--index-date D --min-cell 10 --encoding --sep]   # aggregate audit
python -m falls_ml meuhedet-build --input <extract> --out <new dir> --index-date D --dataset-version V --data-freeze-date D \
    [--index-day-records fail|drop_rows] [--scientific-use-allowed --approval-reference TEXT] [--report <json>]
python -m falls_ml freeze-baseline --run runs/<run> --name NAME [--baselines-dir baselines]   # immutable baseline record
python -m falls_ml meuhedet-explore --input <extract.xlsx|.csv|.parquet> [--out DIR --index-date D --sheet S --data-freeze-date D --full]
```

## 6. Test strategy

| Suite | Proves |
|---|---|
| `unit/` | Every transform; missing handling; categorical encoding; polypharmacy window; outcome window boundaries; FP scaling/selection; LASSO objective/KKT/grid/CV rule vs scikit-learn cross-check; metric formulas against hand calculations; meta-analysis; sample size; adapter interface compliance for every registered model; feature ranking; bundle save/load |
| `leakage/` | Post-index events cannot enter predictors; outcome events cannot enter features; preprocessing/tuning/calibration never receive validation-or-test (resp. test) rows or labels; orchestrator spies on `fit_pipeline`, recalibration, stability, optimism and permutation importance; resampling reuses tuned parameters; test-evaluation registry and label-maturity refusal; patient overlap detection; temporal embargo; ML layer imports no `dataeng` module and no DB driver |
| `regression/` | RT-01…RT-13 from the specification (published equation, recalibration, sample size, NB arithmetic, FP scaling) |
| `integration/` | Full `run_experiment` on a tiny deterministic synthetic fixture; bundle → `predict_risk` identical to in-run served predictions; `reproduce` (dataset located from the manifest) recreates metrics, CIs and predictions exactly; ablation; model override; CLI CSV scoring from a foreign working directory |

## 7. Implementation phases

| Phase | Deliverable | Gate |
|---|---|---|
| 3 | Contracts, configs, tests for the critical scientific logic (published equation, transforms, windows, leakage) | Tests written and failing for the right reason |
| 4 | Published eFalls adapter, derivation, preprocessing, FP, LASSO-CV solver | Phase 3 tests green; RT-01…RT-08 green |
| 5 | Metrics, calibration, importance, stability, optimism, IECV, reporting, comparison | Integration run on fixture produces every artifact |
| 6 | Logistic, elastic net, random forest, histogram gradient boosting via the same adapter | Adapter compliance tests green |
| 7 | Bundle, `predict_risk`, monitoring, CLI, reproduce | Bundle round-trip equality; full test suite green |
