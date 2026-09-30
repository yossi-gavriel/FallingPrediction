# `data/` — modelling-dataset contract

> **No real patient data and no credentials belong in this folder, in this package or in any zip made from it.**
> Everything shipped here is **synthetic** (`data/fixtures/`) or a column **template** without data rows.
> The pipeline **never connects to a database**: it reads one local, already-extracted file per dataset and nothing else.
> Extraction from the Meuhedet data warehouse is done separately by the data team under their own governance.
> `tools/build_handoff.py` refuses to package any data file outside `data/fixtures/` and any parquet that is not a synthetic fixture.

## 1. What is in this folder

| Path | Content |
|---|---|
| `fixtures/synthetic_v1/` | Synthetic dataset with all 78 eFalls candidate predictors (software tests and demos only) |
| `fixtures/synthetic_enhanced_v1/` | Synthetic dataset with eFalls + illustrative Meuhedet-enhanced features (ablation demo) |
| `fixtures/synthetic_reduced_v1/` | Synthetic dataset restricted to the features declared in `configs/experiments/fixture/efalls_retrained_reduced.yaml` |
| `example_schema.csv` | One row per column of a modelling dataset: role, required for training/scoring, type, allowed values, missing-value behaviour, eFalls term, layer, time window, decisions. **Generated from the code** — do not edit by hand |
| `example_header_only.csv` | Header row of the minimum extract accepted by `build-dataset` (no data rows) |
| `README.md` | This contract |

`example_schema.csv` and `example_header_only.csv` are written by `python tools/generate_data_schema.py` from the feature spec and from the validator in `src/falls_ml/data/schema.py`. `python tools/generate_data_schema.py --check` fails when they (or the sha256 below) are out of date. The files are UTF-8; in Excel open them with *Data → From Text/CSV* so the few non-ASCII symbols in the time-window column display correctly.

Every synthetic fixture has `manifest.json` with `"source": "synthetic_fixture"` and `"scientific_use_allowed": false`. Results computed from them are software-test output: **SYNTHETIC DATA – NOT SCIENTIFIC RESULTS**.

## 2. Where real data goes later

- Recommended location: approved storage OUTSIDE this package folder, for example `D:\falls_ml_data\meuhedet\<dataset_version>` (Windows) — never inside the `falls_ml_handoff` folder. Pass the path with `--dataset` (the Meuhedet experiment templates have `dataset.path: null` on purpose).
- A modelling dataset folder contains exactly `modeling_dataset.parquet` and `manifest.json`, written by `build-dataset`. Datasets are immutable: `build-dataset` refuses a non-empty output folder, and every load verifies the file hash against the manifest.
- Keep raw extracts, modelling datasets, `runs/` and `reports/` from real data on approved storage only. Never copy them into the handoff package, a zip, an e-mail or a shared drive that is not approved for patient data.
- Never write passwords, connection strings or tokens into configs, scripts or notes. The pipeline does not need any.

## 3. Feature specification

| Item | Value |
|---|---|
| File | `configs/features/efalls_v1.yaml` |
| Name | `efalls_v1` |
| Version | `1.0.0` |
| Content sha256 | `9693d523607e76fc4d603aed7fac8fa2b97502a6dc925870d19f9a5ede976109` |
| Predictors | 78 eFalls candidates: 6 non-binary + 72 binary (62 retained in the published model) |

The manifest of every dataset records the spec name, version and content sha256. A dataset built for a different spec content is refused (`Dataset was built for a different feature spec`). The sha256 above is printed by `python tools/generate_data_schema.py`.

A reduced dataset (section 10) records a **subset** spec digest derived from this root sha256 and the declared feature list; `validate-dataset` recognises it from the manifest.

## 4. Columns

The authoritative, complete list is `example_schema.csv`. Summary:

### 4.1 Required for training (`train`, `build-dataset`, `validate-dataset`)

| Column | Role | Type | Rule |
|---|---|---|---|
| `research_id` | identifier | string | Pseudonymised; never a predictor; no nulls; `research_id` + `index_date` unique |
| `index_date` | index date | date | Prediction is made at the start of this day |
| `predictor_max_record_date` | predictor provenance | date | Latest record date used by any predictor of the row; must be **strictly before** `index_date` |
| `outcome_12m` | outcome | integer 0/1 | See section 7 |
| `outcome_first_event_date` | outcome provenance | date, nullable | Date of the first qualifying event; non-null exactly when `outcome_12m = 1` |
| 78 predictors | predictor | see section 5 | Names, types and levels in `example_schema.csv` (`role = predictor`) |
| `dataset_version`, `mapping_version`, `source` | version | string | One constant value each. `build-dataset` writes them from `--dataset-version`, `--mapping-version`, `--source`; an extract may omit them (if present they must equal the arguments) |

`example_header_only.csv` is exactly this list without the three version columns.

### 4.2 Required for scoring (`predict`, `monitor`)

`index_date`, `predictor_max_record_date` and the predictors the model was trained on. `research_id` is optional (echoed in the predictions when present). `outcome_12m` is only needed for `monitor --labels`.

### 4.3 Optional columns

| Column | Type | Use |
|---|---|---|
| `practice_id`, `deprivation_group`, `site_id` | string, nullable | Cluster columns; needed only when an experiment config names them (for example `analysis.iecv.cluster_column`, `evaluation.cluster_column_for_heterogeneity`) |
| `death_date`, `followup_end_date` | date, nullable | Descriptive metadata; never predictors |

**No other column is allowed.** Validation rejects undeclared columns (for example a raw date of birth, names, addresses, free text). Remove them from the extract.

## 5. Data types

| Spec type | Parquet dtype | CSV cell | Nulls |
|---|---|---|---|
| identifier / string | string | text (leading zeros kept) | not allowed for `research_id` |
| date | datetime64, date only | `YYYY-MM-DD` | not allowed (`index_date`, `predictor_max_record_date`) |
| date, nullable | datetime64 | `YYYY-MM-DD` or empty | allowed |
| `binary` | int8/int64 or bool | `0` or `1` | **not allowed**: absence of a record is `0` |
| `count` | int64 | integer ≥ 0 | **not allowed**: no record in the window is `0` |
| `float` | float64 | decimal with `.` | not allowed |
| `float_nullable` | float64 | decimal or empty | allowed |
| `categorical` | string | one declared level | not allowed |
| `categorical_nullable` | string | one declared level or empty | allowed |

In CSV extracts a missing value is an **empty cell**. The texts `NA`, `null`, `nan` or `missing` are not missing values; they are rejected as undeclared levels or non-numeric values.

## 6. Allowed values and missing-value behaviour

| Predictor | Allowed values | Null means | Published scoring (`efalls_published`) | Retrained eFalls (`efalls_fp_all_levels`) | Alternative models (`efalls_reference_coded`, `efalls_raw`) |
|---|---|---|---|---|---|
| `age_years` | 65.0 .. 120.0 (decimal years, D-06) | rejected | linear term | FP selection | linear / raw |
| `sex` | `female`, `male` | rejected | sex term (D-01) | all-levels indicators | indicators |
| `polypharmacy_count_120d` | integer ≥ 0 | rejected (use 0) | ln((P+1)/10) | FP selection | ln((P+1)/10) / raw count |
| `bmi_value` | 10.0 .. 80.0 kg/m² (outside ⇒ set to null before building) | no valid BMI in 5 years | category `missing` (reference `overweight`) | `missing` all-levels indicator | `missing` indicator |
| `smoking_status` | `never`, `ex`, `current` | no smoking record | reference `ex_never` (only `current` has a coefficient, D-04) | merged into `never` | merged into `never` |
| `alcohol_category` | `harmful`, `higher_risk`, `lower_risk`, `previous_higher_risk_or_harmful`, `zero` | no alcohol record in 5 years | level `missing` (reference `lower_risk`, D-05) | `missing` all-levels indicator | `missing` indicator |
| 72 binary predictors | `0`, `1` | rejected (absent = 0) | indicator | indicator | indicator |

- Never write `missing` or `ex_never` into the data: they are derived by preprocessing from nulls.
- BMI categories are derived in preprocessing with WHO cut-points (D-02): underweight < 18.5 ≤ normal < 25 ≤ overweight < 30 ≤ obese.
- Validation never repairs data. It collects every problem and fails with the complete list.

## 7. Outcome requirements

- `outcome_12m = 1` when at least one qualifying event occurs in the window: an emergency-department attendance or non-elective hospital admission with an eFalls fracture code (S22, S32, S42, S52, S72, S82, T08, T10, T12, T14.2, M80) in any diagnosis position or a fall code W00–W19 in any position (D-09). Otherwise `0`.
- **Window:** `index_date ≤ event_date ≤ index_date + 1 calendar year − 1 day`, both ends inclusive (D-00). Example: index 2018-04-01 → last day 2019-03-31. The end is computed as a calendar-year offset, so index 2020-02-29 → last day 2021-02-27.
- `outcome_first_event_date` is the date of the first qualifying event: required when `outcome_12m = 1`, empty when `outcome_12m = 0`, and inside the window. Any violation fails validation (`LEAKAGE/DEFINITION`).
- Deaths are retained as non-events unless a qualifying event occurred first (D-18); disenrolment is not modelled.
- The pre-modelling outcome audit (counts by code block, diagnosis position, encounter type, admission method) belongs in the manifest audit (D-09).

## 8. Dates and time semantics

- **D-00:** the prediction is made at the start of `index_date`. Predictors use only records with `record_date < index_date` **and** available (posted) before `index_date`; if availability is unknown for a source, apply the pre-specified lag buffer (spec D-00, M-03).
- A predictor window of W days covers `index_date − W days ≤ record_date ≤ index_date − 1 day` (for example polypharmacy: 120 days, D-03).
- `predictor_max_record_date` must be `< index_date` on every row; a row on or after the index date fails validation as `LEAKAGE`.
- Use date-only values (no time of day) for all date columns.
- **One row per patient per index date** (D-15). eFalls-labelled experiments use one index-date cohort; pooled index dates of one patient must be at least one year apart.
- Cohort: age ≥ 65 at the index date (`cohort.age_min`).
- **`data_freeze_date`** (manifest, required by `build-dataset`) is the last date of source-data extraction. **Label maturity (D-19):** a run refuses the dataset unless `data_freeze_date ≥ max(index_date) + 1 year − 1 day + validation.outcome_lag_days`, so every outcome window is closed before modelling.

## 9. Building, validating and training with a real dataset later

All commands run from the project folder with the virtual environment active (Windows: `.venv\Scripts\activate.bat`; macOS/Linux: `source .venv/bin/activate`). Scientific use starts only after the Meuhedet mappings are clinically validated and the decision register is signed off (spec §15).

**1. Build** an immutable dataset from a local extract (`.parquet` or `.csv`, one row per `research_id` × `index_date`, columns as in `example_header_only.csv`):

```bat
python -m falls_ml build-dataset --input <extract.parquet|.csv> --out data\meuhedet\<dataset_version> --dataset-version <dataset_version> --mapping-version <mapping_version> --source "<source description>" --data-freeze-date YYYY-MM-DD --scientific-use-allowed --approval-reference "<approval id>"
```

- `build-dataset` reads CSV types from the feature spec, validates strictly, refuses a non-empty `--out`, records the input file hash and the approval reference in the manifest audit, and never connects to a database.
- `--scientific-use-allowed` is accepted only together with `--approval-reference`. Without both, the dataset is marked not approved and the Meuhedet templates refuse it.

**2. Validate** (full or subset spec is selected from the manifest):

```bat
python -m falls_ml validate-dataset --dataset data\meuhedet\<dataset_version>
```

**3. Train** with the Meuhedet templates in `configs\experiments\` (they require `scientific_use_allowed: true`; runs go to `runs\`):

```bat
python -m falls_ml train --config configs\experiments\efalls_published_scoring.yaml --dataset data\meuhedet\<dataset_version>
python -m falls_ml train --config configs\experiments\efalls_retrained_lasso.yaml --dataset data\meuhedet\<dataset_version>
python -m falls_ml compare --runs-dir runs --out reports
```

Run the published scoring first so later runs can reference it. Re-evaluating the same test rows with the same config is refused unless `--allow-test-reevaluation` is given, and the override is recorded (D-19).

## 10. Reduced eFalls extractions

When some of the 78 eFalls predictors cannot be extracted, use `configs/experiments/efalls_retrained_reduced.yaml`:

1. List the predictors that **are** available in `preprocessing.features` of that config (an explicit, strict subset of the 78 candidates that includes `age_years`). The unavailable list shipped in the template is illustrative only.
2. Build and validate the dataset with the declared features only:

```bat
python -m falls_ml build-dataset --input <extract> --out data\meuhedet\<dataset_version>_reduced --dataset-version <dataset_version> --mapping-version <mapping_version> --source "<source description>" --data-freeze-date YYYY-MM-DD --features-from-config configs\experiments\efalls_retrained_reduced.yaml --scientific-use-allowed --approval-reference "<approval id>"
python -m falls_ml validate-dataset --dataset data\meuhedet\<dataset_version>_reduced --features-from-config configs\experiments\efalls_retrained_reduced.yaml
python -m falls_ml train --config configs\experiments\efalls_retrained_reduced.yaml --dataset data\meuhedet\<dataset_version>_reduced
```

A reduced run is **never** a full eFalls reproduction. Its report states `Available eFalls predictors: X / 78`, the coverage percentage and the unavailable predictors, and `metrics.json` records the same under `efalls_coverage`. Declaring all 78 features is refused: use `efalls_retrained_lasso.yaml` instead.

## 11a. Meuhedet wide table (Phase 1)

The research-wide table `V_Falls_Prediction_Wide_1` is not a modelling dataset: it is a 221-column source. `falls_ml meuhedet-build` turns one exported file of it into a modelling dataset through the column contract `configs/meuhedet/wide_v1_columns.yaml` and the eFalls mapping manifest `configs/meuhedet/wide_v1_efalls_mapping.yaml` (see `docs/meuhedet/WORK_PC_RUNBOOK.md`). Its outcome is the 180-day label (`fall_next_180d`, exploratory), not `outcome_12m`; such datasets are validated against `configs/features/efalls_v1__outcome_fall_next_180d_exploratory.yaml` and every run on them is labelled "NOT eFalls reproduction".

## 11. Checklist before any real-data run

- [ ] Extract produced by the data team from approved sources; no database credentials anywhere in this project
- [ ] Only declared columns; no names, addresses, raw dates of birth or free text
- [ ] Binary predictors and counts materialised as 0 when absent; nulls only in nullable columns
- [ ] `predictor_max_record_date < index_date` on every row; outcome dates inside the window
- [ ] `data_freeze_date` recorded and outcome windows mature
- [ ] Mapping version validated clinically; approval reference available
- [ ] Dataset, runs and reports stored on approved storage only, never inside the handoff package
