# falls_ml — clinical fall-risk prediction (eFalls reproduction)

Research pipeline for predicting 12-month emergency-department attendance or hospital admission with a fall or fracture in adults aged ≥ 65. The published **eFalls** model (Archer et al., *Age Ageing* 2024;53(3):afae057) is the scientific baseline.

> **Status.** The code, tests and reports are complete and verified on deterministic **synthetic fixtures**. **No Meuhedet data has been used.** Every number produced from the fixtures is software-test output, not scientific evidence. Scientific use is blocked on:
> - the Meuhedet modelling dataset and its clinically validated mappings (spec B-01, B-02, B-04);
> - sign-off of the decision register (spec §15);
> - author queries Q-01 and Q-02.

## Documents

| Document | Purpose |
|---|---|
| [`docs/EFALLS_REPRODUCTION_SPEC.md`](docs/EFALLS_REPRODUCTION_SPEC.md) | What eFalls is exactly (published) and what we had to assume. Contains the decision register D-00…D-22, errata, Meuhedet mapping requirements, regression tests, blockers, the unresolved-items register and the sign-off checklist |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Package layout, contracts, `run_experiment` flow |
| [`docs/ARTIFACT_SCHEMAS.md`](docs/ARTIFACT_SCHEMAS.md) | Every file in a run directory and in a model bundle |
| [`docs/research_notes/`](docs/research_notes/) | Evidence archive: sources, quotes, audits, spec reviews |
| [`WINDOWS_HANDOFF.md`](WINDOWS_HANDOFF.md) | Installing and verifying the package on a Windows 10/11 work computer (online or offline) |
| [`data/README.md`](data/README.md) | Modelling-dataset contract: required columns, types, missing values, dates, how to build and validate a real dataset later |
| [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) | Pinned dependencies and lock files |
| [`docs/meuhedet/PHASE1_REPORT.md`](docs/meuhedet/PHASE1_REPORT.md) | Meuhedet wide table, Phase 1: cohort/outcome definitions, eFalls coverage (15/78), datatype contract, leakage audit, blockers; with the generated [mapping table](docs/meuhedet/MAPPING_TABLE.md), [column inventory](docs/meuhedet/COLUMN_INVENTORY.md), [datatype contract](docs/meuhedet/DATATYPE_CONTRACT.md) and the [work-PC runbook](docs/meuhedet/WORK_PC_RUNBOOK.md) |

## Installation

**Windows users:** follow [`WINDOWS_HANDOFF.md`](WINDOWS_HANDOFF.md). `setup_windows.cmd` performs every step below, runs the tests and a synthetic smoke test, and ends with `INSTALLATION SUCCESSFUL`.

**macOS/Linux developers:** `bash scripts/posix/setup.sh` runs the same steps. The manual equivalent is:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check --no-input --require-hashes --only-binary=:all: -r requirements-build.lock
.venv/bin/python -m pip install --disable-pip-version-check --no-input --require-hashes --only-binary=:all: -r requirements.lock
.venv/bin/python -m pip install --disable-pip-version-check --no-input --no-deps --no-build-isolation --no-index -e .
```

- `requirements.lock` pins every runtime and test dependency with sha256 hashes (wheels for Windows x64 CPython 3.11 and 3.13, and macOS arm64 CPython 3.13). `requirements-build.lock` pins pip, setuptools and wheel.
- Offline: `python scripts/handoff/prepare_offline.py` (on any machine with internet) fills `offline_packages/`; then add `--no-index --find-links offline_packages` to the two lock-file commands.
- The editable install (`-e .`) is required: `falls_ml.paths` resolves `configs/` relative to the source tree.
- Supported runtime: CPython 3.11 or 3.13, 64-bit, on Windows 10/11 x64; development on macOS arm64 (3.13). Windows ARM64, 32-bit Python and Python 3.12 are not supported.

Dependencies are limited to numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, PyYAML, pyarrow and pytest (details in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md)). XGBoost, LightGBM and SHAP are intentionally not used.

## Verification and synthetic demos

| Windows | macOS/Linux | What it does | Output |
|---|---|---|---|
| `verify_installation.cmd` | `scripts/posix/verify.sh` | 9 installation checks incl. core tests and bundle save/load | `demo_outputs/verify/` |
| `run_demo.cmd` | `scripts/posix/run_demo.sh` | Published scoring + retrained LASSO on the synthetic fixture, reports, bundles, predictions, comparison | `demo_outputs/baseline/` |
| `run_full_demo.cmd` | `scripts/posix/run_full_demo.sh` | All algorithms, reduced eFalls set, ablation, compare, predict, monitor, reproduce | `demo_outputs/full/` |
| `clean_demo_outputs.cmd` | `scripts/posix/clean_demo_outputs.sh` | Deletes only generated folders under `demo_outputs/` | |

Demos and verification write only to `demo_outputs/` (never to `runs/` or `reports/`; pytest temporary files in `demo_outputs/pt/`, matplotlib cache in `.venv/mplconfig/`); setup logs go to `setup_logs/` and the installation summary to `environment_report.txt`. Every generated folder is marked `SYNTHETIC DATA – NOT SCIENTIFIC RESULTS`.

The handoff zip is built with `python tools/build_handoff.py` (writes `dist/falls_ml_handoff/`, `dist/falls_ml_handoff_<VERSION>.zip` and `dist/SHA256SUMS.txt`; `--check-only` runs the packaging checks without writing).

## The experiments (never confused)

| Kind | Config | Question |
|---|---|---|
| **A. Published eFalls scoring** | `configs/experiments/efalls_published_scoring.yaml` | Does the **original published equation** transport to Meuhedet? Primary sex parameterisation is `lp_c_box_s3_1`, co-reported `lp_a_table_s3_2` (spec D-01) |
| **B. eFalls retrained** | `efalls_retrained_lasso.yaml` (FP re-selection) and `efalls_retrained_lasso_fixed_fp.yaml` | Same predictors and learning process (FP + LASSO, 10-fold CV), coefficients fitted locally |
| **B-reduced. eFalls retrained on a reduced predictor set** | `efalls_retrained_reduced.yaml` | Same learning process on an explicitly declared strict subset of the 78 candidates; reports coverage (X / 78) and is never a full eFalls reproduction |
| Experimental algorithms | `logistic_unpenalized`, `elastic_net_logistic`, `random_forest`, `hist_gradient_boosting` | Alternatives on identical partitions; never replace eFalls on AUROC alone |
| Meuhedet-enhanced ablation | `configs/experiments/fixture/ablation_example.yaml` | Incremental value of feature groups beyond pure eFalls (example uses illustrative placeholders) |

`configs/experiments/*.yaml` are Meuhedet templates. They require a scientifically approved dataset and follow the D-19 validation design. `configs/experiments/fixture/*.yaml` run on the synthetic fixture.

## Commands

Paths in configs are relative to the project root and resolve from any working directory.

```bash
# 1. synthetic fixtures (software tests only)
python -m falls_ml make-fixture --out data/fixtures/synthetic_v1 --n-patients 3000
python -m falls_ml make-fixture --out data/fixtures/synthetic_enhanced_v1 --n-patients 3000 --extension configs/features/meuhedet_enhanced_example.yaml
python -m falls_ml validate-dataset --dataset data/fixtures/synthetic_v1
# real data later (see data/README.md): build-dataset validates an extract and writes an immutable dataset + manifest
# python -m falls_ml build-dataset --input <extract.parquet|.csv> --out <approved storage outside this folder>/meuhedet/<dataset_version> --dataset-version V --mapping-version V --source TEXT --data-freeze-date YYYY-MM-DD

# 2. experiments: run the published equation first so later runs can reference it
python -m falls_ml train --config configs/experiments/fixture/efalls_published_scoring.yaml
python -m falls_ml train --config configs/experiments/fixture/efalls_retrained_lasso.yaml
python -m falls_ml train --config configs/experiments/fixture/random_forest.yaml

# 3. Meuhedet-enhanced ablation (runs go to runs_ablation/, summary to reports/ablation/)
python -m falls_ml ablation --config configs/experiments/fixture/ablation_example.yaml --dataset data/fixtures/synthetic_enhanced_v1

# 4. master comparison and best features (reads reports/ablation/ when present)
python -m falls_ml compare --runs-dir runs --out reports

# 5. production and governance
python -m falls_ml predict --model runs/<run>/model --input new_rows.parquet --out predictions.parquet
python -m falls_ml monitor --model runs/<run>/model --input new_rows.parquet --out drift/
python -m falls_ml reproduce --run-dir runs/<run>          # re-runs into runs_reproduced/
python -m falls_ml evaluate --run-dir runs/<run>           # re-render report only

# 6. Meuhedet wide table, Phase 1 (docs/meuhedet/WORK_PC_RUNBOOK.md): one local extract of V_Falls_Prediction_Wide_1, no database
python -m falls_ml meuhedet-explore --input <falls_extract.xlsx>   # ONE command: audit -> cohort -> STRICT + EXTENDED 180-day exploratory LASSO -> reports
python -m falls_ml meuhedet-make-fixture --out demo_outputs/meuhedet/wide.parquet --n-rows 4000      # SYNTHETIC 221-column extract
python -m falls_ml meuhedet-audit --input <extract> --out <audit dir> --index-date 2025-01-01         # aggregate, non-identifying audit
python -m falls_ml meuhedet-build --input <extract> --out <new dir> --index-date 2025-01-01 --dataset-version V --data-freeze-date YYYY-MM-DD
python -m falls_ml train --config configs/experiments/meuhedet/phase1_180d_exploratory_efalls_reduced.yaml --dataset <dir>
python -m falls_ml freeze-baseline --run runs/<run> --name EFALLS_BASELINE_MEUHEDET_V1               # immutable baseline record
```

Run lifecycle and safeguards:

- A run is complete only when `RUN_COMPLETE.json` exists. Comparison, best-features and eFalls reference lookups ignore incomplete runs.
- Each release of a test partition is recorded in `<runs_dir>/test_evaluation_registry.jsonl`. On a scientific dataset, re-evaluating the same test rows with the same config is refused unless `--allow-test-reevaluation` is given (D-19 section 4). The override is recorded.
- The prediction served by a bundle (`served_variant`) is the one evaluated for eligibility. The published equation is served uncalibrated by default. Its recalibrated variant is reported beside it (`calibration.serve_recalibrated` changes this).
- Scientific datasets must record `data_freeze_date`. A run refuses data whose last outcome window (plus `validation.outcome_lag_days`) is not closed at the freeze date.

Python API:

```python
from falls_ml.experiment import run_experiment
result = run_experiment(model_name=None, dataset="data/fixtures/synthetic_v1",
                        experiment_config="configs/experiments/fixture/efalls_retrained_lasso.yaml")
```

Passing a different `model_name` creates a separately named experiment, `<experiment>__<model>`, using that adapter's representation.

## Tests

```bash
.venv/bin/python -m pytest -m "not slow"      # unit, leakage, regression (fast)
.venv/bin/python -m pytest                    # including the end-to-end integration test
```

| Folder | Covers |
|---|---|
| `tests/regression/` | Published-equation examples (RT-01…RT-08), sample sizes, net benefit, config provenance |
| `tests/leakage/` | Time boundaries, split and patient leakage, train-only preprocessing, calibration and tuning isolation, orchestrator spies (no learning step sees test rows; resampling reuses tuned parameters), test-evaluation registry, label maturity, inference identity, architecture boundaries |
| `tests/integration/` | Whole pipeline on a tiny synthetic fixture: bundle → `predict_risk` equals in-run served predictions; exact reproduction from run artifacts; ablation; model override; CLI CSV scoring from a foreign working directory |

## Data boundary

- The ML layer consumes only immutable modelling datasets (Parquet + hashed manifest) validated against `configs/features/efalls_v1.yaml`.
- `falls_ml.dataeng` is a *reference* derivation from canonical event tables. Meuhedet data engineering implements the mappings in `configs/mappings/`; nothing in the ML layer queries the DWH.
- The Meuhedet research-wide table (`V_Falls_Prediction_Wide_1`, 221 columns) enters only through `falls_ml.data.meuhedet_wide`: a column contract (`configs/meuhedet/wide_v1_columns.yaml`: role, three types, NULL semantics per column) and an eFalls mapping manifest (`configs/meuhedet/wide_v1_efalls_mapping.yaml`: 15 of 78 eFalls predictors mappable, quality-graded). Its 180-day label is an **exploratory** outcome, never called an eFalls reproduction (`docs/meuhedet/PHASE1_REPORT.md`).

## Operational notes

- **Compute:** at eFalls scale (N ≈ 660k), a 10-fold LASSO fit takes tens of minutes, and B = 200 stability replicates multiply that (spec U-32).
- **Pickle safety:** model bundles use pickle for the preprocessor and tree models. Load bundles only from trusted, integrity-checked locations (`load_bundle` verifies file hashes first).
- **Risk categories** stay hidden until thresholds are clinically/business approved (spec M-12).
- **Real-data reports** must not be exported until complementary small-cell disclosure control is designed (spec U-33).
