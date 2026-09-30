# Meuhedet Phase 1 — eFalls-compatible baseline: first deliverable

**Scope:** `[Meuhedet_DWH].[dbo].[V_Falls_Prediction_Wide_1]` (Definition_Version V1.0, 221 columns) → data contract → eFalls mapping →
canonical dataset → reduced eFalls-compatible LASSO baseline → feature analysis → frozen baseline. No row-level Meuhedet data were
used to write this report: every number under "synthetic" comes from the packaged synthetic extract (**SYNTHETIC DATA – NOT SCIENTIFIC
RESULTS**); every real number comes from `falls_ml meuhedet-audit` run on the work computer (section 9).

Feature specification used as the denominator: `configs/features/efalls_v1.yaml`, **efalls_v1 1.0.0**, sha256
`9693d523607e76fc4d603aed7fac8fa2b97502a6dc925870d19f9a5ede976109` — 78 eFalls candidate predictors (72 binary + 6 non-binary;
the count is derived from `exact_efalls_baseline: true`, never hard-coded). Mapping manifest: `configs/meuhedet/wide_v1_efalls_mapping.yaml`
(meuhedet_wide_efalls 1.0.0, **PROPOSED**, not clinically validated). Contract: `configs/meuhedet/wide_v1_columns.yaml` (meuhedet_wide_v1 1.0.0).

## 1. Verdicts

| Layer | Verdict | Reason |
|---|---|---|
| A. Published eFalls scoring | **BLOCKED** | coverage 15/78 (19.2%) < 90% rule (M-11); mandatory `fracture`, `fragility_fracture` unavailable; BMI, smoking, alcohol unavailable; no 12-month ED/admission outcome |
| B. Retrained eFalls, full 78 | **BLOCKED** | 63 candidates have no population-wide pre-index source in the wide table |
| C. Retrained eFalls, reduced | **RUNNABLE as "Meuhedet 180-day exploratory – NOT eFalls reproduction"** | 15 predictors (0 exact, 9 high-confidence, 6 approximate) on the 180-day label; the run, report and metrics carry both labels |
| eFalls 365-day outcome | **BLOCKER** | see section 6 |

## 2. Cohort definition (exact)

- Source rows: `V_Falls_Prediction_Wide_1` exported for one `Index_Date` (primary proposal **2025-01-01**; 49 snapshots exist, one is chosen per build).
- Filter: `Is_Eligible_Cohort = 1` — the only filter. No balancing, no re-weighting, no age/sex/feature restrictions; `Birth_Date_Suspect_Ind = 1`
  rows (default birth dates, ≈10.5% per S2T) stay in the primary cohort and are counted; a sensitivity analysis excludes them.
- Rows whose label is NULL (`Fall_Next_180D_Ind` NULL = censored per S2T) cannot be trained on: they are dropped and counted by
  `Label_Reason_180D` in the build report (deaths inside the window are handled as the VIEW labels them; eFalls keeps deaths as
  non-events, D-18 — the audit cross-tabulates `Label_Reason_180D × Is_Deceased_Ind` so the two conventions can be compared).
- One row per patient (`Customer_Full_ID × Index_Date`), so the patient-grouped random split (validation 20%, test 20%, stratified
  by outcome, seed 42) is exact; a temporal design is not possible with one snapshot (documented `limitation_note`).
- Cohort profile (N, events, prevalence, age/sex distributions, missingness, feature prevalence, source coverage, rare values):
  produced by `meuhedet-audit` — synthetic example in `runs_meuhedet_fixture/` and section 8.

## 3. eFalls coverage (denominator 78, derived from the spec)

| | n | predictors |
|---|---|---|
| Available for the baseline | **15 (19.2%)** | |
| — EXACT | 0 | eFalls uses UK primary-care code groups the wide table does not carry |
| — HIGH_CONFIDENCE | 9 | age_years (Age_At_Index), sex (Gender_Code), dementia, copd, asthma, diabetes_mellitus (T1 ∪ T2), liver_problems, hypertension (blood-pressure registry), housebound (home-confined registry) |
| — APPROXIMATE | 6 | polypharmacy_count_120d (Distinct_Active_Substance_Count: substances vs BNF paragraphs, active exposure vs 120 days), falls (Prior_Fall_Since_Study_Start_Ind: falls+fractures, ≤3-year window), chronic_kidney_disease (CRF registry, no eGFR rule), severe_mental_illness (Registry_SMI_Level ≥ 1), self_harm (suicide-attempt registry), mobility_problems (ICD-9 719.7 since 2022) |
| Unavailable | 63 | |
| — assessment-conditional nurse sources exist (Phase 2 only) | 18 | cognitive_impairment, memory_concerns, visual_impairment, hearing_impairment, weakness, dizziness, hypotension_or_syncope, activity_limitation, dressing_and_grooming_problems, washing_and_bathing, toileting_problems, meal_preparation_problems, shopping_problems, problems_managing_finances, medication_management, requirement_for_care, environment_problems, social_vulnerability |
| — heart-disease registry cannot be split (Q-M-01) | 4 | ischaemic_heart_disease, heart_failure, heart_valve_disease, atrial_fibrillation |
| — no source at all | 41 | bmi_value, smoking_status, alcohol_category, fracture, fragility_fracture, urinary_system_disease, and 35 condition-level candidates (see `MAPPING_TABLE.md`) |
| Mandatory eFalls predictors unavailable | 2 | **fracture, fragility_fracture** (merged into the fall/fracture event counts) |
| Published-model predictors available | 11 of 62 (the 12 binary predictors minus chronic_kidney_disease, which LASSO dropped in the published model) | |

Rule applied: a nurse-assessed field (NULL = not assessed for most members) can never back an eFalls binary, because the eFalls
`absent_is_zero` rule would turn "not assessed" into "absent" and encode "was assessed by a nurse" into every such predictor.
Those fields are Phase-2 groups with explicit missing indicators. Full table: `MAPPING_TABLE.md`, `tables/efalls_meuhedet_mapping.csv`.

## 4. Datatype contract and audit

Three declared types per column (`DATATYPE_CONTRACT.md`): SQL type (DDL) → canonical semantic type (identifier / date / binary /
continuous / count / categorical / ordinal / quality_control / label) → model-matrix type (never / float64 / int8_binary /
onehot_categorical / ordinal_float64). Reading casts to the declared pandas dtype with no inference (`Int64`, `float64`,
`datetime64[us]`, `string`); every value that does not fit is counted as `wrong_type` (date columns also report the offending values,
never identifiers); date text is read only with the source layouts declared in the contract (`date_formats`: `%Y-%m-%d`, `%d/%m/%Y`,
tried in that order, so `01/02/2024` is always 1 February and a set that could read one text two ways is refused at load time), and
dates keep the whole SQL date range, so a business sentinel such as
`2999-12-31` (declared per column: `sentinels` / `sentinel_means`) stays a date and is never converted to NULL without a
column-specific `sentinel_modeling_rule`; only the canonical modelling dates are cast to `datetime64[ns]`, and a far-future value in
one of those columns stops the build naming column, value and expected type; the build stops on any wrong-type value in a
column it uses, on NULL in a DDL NOT NULL column, on values outside `allowed`, on an unknown `Gender_Code`, on
`Definition_Version ≠ V1.0` and on `Leakage_Check_Ind ≠ 0`. Identifiers stay strings and never enter a model; raw dates never enter a
model (the derived `Days_Since_*` columns do).

Synthetic extract: 221/221 columns, wrong-type 0, invalid 0, NOT-NULL violations 0. Real extract: `datatype_audit` section of the audit.

## 5. Missingness rules of the 15 baseline predictors (NULL is never silently 0)

| predictor | source NULL meaning | rule |
|---|---|---|
| age_years, sex | invalid row | `forbid`: the build fails |
| polypharmacy_count_120d | `Distinct_Active_Substance_Count` NULL with `Medication_Missing_Ind = 1` = no medication data | eFalls `absent_is_zero` applied **explicitly** and counted (`null_to_zero`); NULL without the indicator fails |
| falls, mobility_problems | DDL NOT NULL; `Prior_Fall_Missing_Ind = 1` = no fall records at all | copied; source-absent rows counted (`source_absent_rows`) |
| 8 registry binaries | DDL NOT NULL; `Registry_Missing_Ind = 1` = registry source absent | copied; source-absent rows counted |
| severe_mental_illness | `Registry_SMI_Level` NULL = not in the registry | `null_is_absent: true` → 0, counted |

Phase-2 predictors keep NULL as "not assessed" (missing category / missing indicator + training-derived imputation); no rule is
learned from the test split (preprocessing is fitted on train only, unchanged framework).

## 6. Outcome

**A true eFalls-compatible 365-day label does not exist in V1.0 — blocker.** Precisely missing:
1. the 365-day window itself (`Fall_Next_365D_Ind`, `Label_End_365D`, `Has_Full_Followup_365D`, `Is_Censored_365D`, `Next_Fall_Date_365D`);
2. the encounter restriction — eFalls counts ED attendances or hospital admissions; the VIEW's fall events come from Meuhedet sources without a documented encounter setting (M-08);
3. the code list — eFalls uses WHO ICD-10 W00–W19 plus fracture codes (ICD-9-CM candidates in spec §11 M-02); the V1.0 fall/fracture list is undocumented and the legacy list contains 719.7 (difficulty walking), E987 (undetermined intent) and 929.3 (late effect);
4. label maturity for claims-based sources (`Max_Invoice_Lag_365D` is reported by the audit and must set `validation.outcome_lag_days`).

Exploratory outcome used meanwhile: `fall_next_180d` ← `Fall_Next_180D_Ind` (event date `Next_Fall_Date_180D`, window = index day + 180
days as `Label_End_180D`, checked on the data), declared in the generated spec
`configs/features/efalls_v1__outcome_fall_next_180d_exploratory.yaml` (`layer: L3a_meuhedet_exploratory_outcome`). The framework
refuses to tag a day-based window as the published outcome, adds the limitation **EXPLORATORY OUTCOME – NOT an eFalls reproduction**
to every run and records `metrics.outcome.published_efalls_outcome = false`.

## 7. Leakage audit

- Every one of the 221 columns has one role (`COLUMN_INVENTORY.md`): IDENTIFIER 2, COHORT_ELIGIBILITY 21, QA_CONTROL 46,
  EFALLS_BASELINE_FEATURE 16, MEUHEDET_ENHANCED_FEATURE 108, LABEL 16, FORBIDDEN_LEAKAGE 12.
- Forbidden as predictors (never usable): the 16 label columns; `Audit_Only_Adif_Status`, `Audit_Only_Si_Status` (fairness audit only);
  `New_Registry_30D/90D` (S2T: unreliable); `External_Care_Count_30D/90D/180D/365D`, `Last_Ext_Date`, `Days_Since_Last_External_Care`
  (include records billed after the index date — only `External_Care_Count_365D_Visible` is as-of); `Fall_On_Index_Date_Ind`
  (index-day events belong to the outcome window, D-00); `Abroad_Ind` (timing undefined).
- Post-index, cohort-only: `Followup_End_Date/Reason`, `Has_Full_Followup_*`, `Is_Censored_*`, `Is_Deceased_Ind`, `Leave_Join_Ind`,
  `Death_Censor_Date`, `Death_Date_Gap_Days`, `Has_Full_*_Label`.
- Blocked until data engineering confirms as-of semantics (`timing: unknown`): `Hospitalization_Count_180D/365D`, `Total_Hosp_Days_365D`,
  `Last_Hosp_Length/Date`, `Days_Since_Last_Hospitalization` (invoice sources without a visible variant).
- QA/control never used: `Snapshot_Confidence`, `Has_Source_*`, `External_Care_Hidden_By_Billing_Lag_365D`, `Max_Invoice_Lag_365D`, `Leakage_Check_Ind`, …
- D-00 check: `predictor_max_record_date = max(Last_Fall_Date, Last_Dx_Date)` (the event sources behind the baseline predictors);
  any date on/after the index date fails the build (or is dropped and counted with `--index-day-records drop_rows`); the audit also
  reports, for every pre-index date column, how many values lie on/after the index date.
- Synthetic extract: leakage result **PASS** (0 forbidden columns used, `Leakage_Check_Ind` 0, 0 timing violations).

## 8. First run (synthetic dry run — SYNTHETIC DATA – NOT SCIENTIFIC RESULTS)

`falls_ml meuhedet-make-fixture` (4,000 rows) → audit → build (3,000 on 2025-01-01, 2,917 eligible with a label; 220 events, 7.5%) →
`configs/experiments/fixture/meuhedet_180d_exploratory_efalls_reduced.yaml` (FP2 selection + LASSO logistic, 10-fold CV λ-min,
patient-grouped random split) → run `runs_meuhedet_fixture/2026-09-17_fixture_meuhedet_180d_exploratory_efalls_reduced_lasso_logis_a23a5ead`:
test n 583, events 44, AUROC 0.653, PR-AUC 0.138, Brier 0.070, calibration slope 0.58, CITL 0.01, O:E 1.01 (recalibrated variant);
λ* 0.0053, 9 of 20 design columns selected; top features falls, dementia, sex, asthma. Every artifact (`metrics.json`, `report.md`,
`coefficients.csv`, `feature_importance.csv`, `feature_stability.csv`, `unpenalized_refit.csv`, calibration/decision-curve tables and
plots) carries **REDUCED eFalls predictor set – NOT a full eFalls reproduction** and **EXPLORATORY OUTCOME – NOT an eFalls reproduction**,
plus the warning that mandatory `fracture`, `fragility_fracture` are unavailable. The run was frozen with `freeze-baseline` as
`baselines_demo/EFALLS_BASELINE_MEUHEDET_SYNTHETIC_DEMO` to exercise the freeze (dataset hash, index date, spec/mapping/contract
hashes, outcome, model config, λ, coefficients, test metrics, test-row hash). The numbers say nothing about Meuhedet.

## 9. What the work computer produces next (see `WORK_PC_RUNBOOK.md`)

**Shortest path:** export the wide table for one index date to Excel inside Meuhedet and run one command on the work computer —
`python -m falls_ml meuhedet-explore --input "C:\FallsData\falls_extract_2025_01_01.xlsx"` — which audits the file, applies the
cohort rules, builds the STRICT (HIGH_CONFIDENCE) and EXTENDED (HIGH_CONFIDENCE + APPROXIMATE) feature sets from the manifest, runs
both LASSO experiments on identical test rows and writes `SUMMARY.md`, the feature reports and the comparison (all aggregate, ids hashed).

1. `meuhedet-audit` on the real 2025-01-01 export → `wide_extract_audit.md/.json` (aggregates only) → review together, feature by
   feature, and answer Q-M-01…Q-M-07 (`MAPPING_TABLE.md`); set `validation.outcome_lag_days`.
2. `meuhedet-build` → canonical dataset (manifest records every count and hash) → `train` with
   `configs/experiments/meuhedet/phase1_180d_exploratory_efalls_reduced.yaml` → feature report (coefficients, odds ratios, selection,
   bootstrap stability, permutation importance, missing %, prevalence) and performance/calibration report.
3. `freeze-baseline --name EFALLS_BASELINE_MEUHEDET_V1`. Phase 2 (nurse assessment, mobility/ADL, cognition, MEFI, home safety,
   utilisation, medication, registry groups) starts only after that, as grouped ablations on the same test rows.

## 10. What prevents an exact eFalls reproduction

1. No 12-month ED/admission fall-or-fracture outcome (section 6).
2. 63 of 78 candidates have no population-wide pre-index source; two mandatory ones (fracture, fragility fracture) are merged into fall counts.
3. Registry/administrative definitions differ from UK code groups (windows, measurement rules, units — D-03, D-08); no definition is
   clinically validated (B-01/B-02).
4. The only path to an exact reproduction is condition-level data: diagnosis codes with dates, prescriptions with ATC and dates,
   ED/admission encounters with diagnoses — which the existing reference derivation (`falls_ml.dataeng.derive`) already consumes.

## v0.5.0 — the real-data path end to end (2026-09-22)

**Root cause found on the first ~60k extract (D-00).** 4,168 rows carried `Last_Dx_Date = Index_Date` and 22 rows `Last_Fall_Date = Index_Date`.
Prediction happens at the start of the index day (`cohort.prediction_time: START_OF_INDEX_DAY`, `predictor_record_rule: source_event_date <
Index_Date`, both enforced by the mapping loader), so those records are unavailable to the model. `falls_ml.data.meuhedet_timing` now measures,
per pre-index record-date column, the counts before / on / after the index day, the day-offset distribution, the affected eligible rows by
Index_Date, the label prevalence among affected vs clean rows, the consistency with the VIEW's own `Days_Since_*` column, and classifies the
cause: A (every offset 0 ⇒ the VIEW's windows use `Event_Date <= Index_Date`), B (records after the index day ⇒ snapshot reconstruction /
ETL defect, or a misread field). It traces the predictors per source: `Last_Dx_Date` → `mobility_problems` (Gait_Disorder_Since_Study_Start_Ind;
EXTENDED only) and the DIAGNOSIS_BURDEN Phase-2 columns; `Last_Fall_Date` → `falls` (Prior_Fall_Since_Study_Start_Ind; EXTENDED only) and the
PRIOR_FALLS_ENHANCED columns; STRICT reads neither source. Python cannot reconstruct the pre-index state of an affected row (only aggregated
values are exported), so nothing is fabricated: the DWH correction is `Event_Date < Index_Date` for every predictor of those sources
(`Last_*_Date = MAX(Event_Date) WHERE Event_Date < Index_Date`, window counts `>= DATEADD(day, -W, Index_Date) AND < Index_Date`, "since study
start" flags `< Index_Date`, `Days_Since_* >= 1`); until then `--index-day-records drop_rows` excludes and counts the rows and the diagnostic
reports whether that removal is label-related. The guard itself is unchanged.

**Framework additions.** Multi-snapshot builds (`index_dates`), the snapshot audit and embargo-aware temporal design
(`falls_ml.data.meuhedet_snapshots`), the pre-training adequacy gate (`falls_ml.adequacy`: both classes everywhere; train ≥ max(20, 2 × design
columns) events, warning below 5 per column; validation / test ≥ 10 events and non-events; constant and sparse predictors reported), the
pepper sidecar (`<file>.id_pepper.txt`, created once, reused; only its sha256 in reports), `--audit-only` (READINESS.md), `--all-index-dates`
(temporal primary analysis + patient-disjoint sensitivity analysis + temporal test performance by snapshot), the split audit with per-partition
hashes, feature reports with the eFalls concept and feature-set membership, and a synthetic monthly-panel fixture with repeated patients,
freeze censoring and planted same-day / future records.

**Still blocking exact eFalls reproduction.** The 365-day ED/admission outcome does not exist in V1.0; the fall / fracture code lists are
undocumented against eFalls; fracture and fragility_fracture are unavailable; polypharmacy, falls, CKD, SMI, self-harm and mobility are
approximations (retained as APPROXIMATE on purpose). Every report keeps the watermark EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION.

## v0.5.1 — SAFE-ALL-ROWS sensitivity and ablations without DWH access (2026-09-22)

**Three analysis states, never merged.** `CURRENT_D00_DROPPED` (the completed reference run on the D-00-clean cohort; read-only, hashed before and after);
`SAFE_ALL_ROWS` (every labelled eligible row, no exclusion because `Last_Dx_Date` / `Last_Fall_Date` equal the index date, only the predictors the file itself
shows to be temporally safe); `FINAL_DWH_FIXED` (documented only: the full predictor set on the full population once the DWH rebuilds the source windows with
`Event_Date < Index_Date`; never run, never estimated).

**Provenance rule (`falls_ml.meuhedet_sensitivity.feature_provenance`).** Each included predictor is traced through the mapping (`source_columns`, `op`,
`record_date_column`, new `same_day_evidence_column`) and the contract (group, timing) to the D-00 evidence measured on the extract. Classes:
EXCLUDED_PROVEN_SAME_DAY_SOURCE (the event source's last-record date is on/after the index day: `falls`, `mobility_problems` on the first real extract),
EVENT_SOURCE_CLEAN_IN_THIS_EXTRACT, DERIVED_FROM_IMMUTABLE_ATTRIBUTES (age, sex), STATE_AT_INDEX_WITH_PARTIAL_EVIDENCE (the ten registry predictors:
`First_Registry_Date` on/after the index day proves index-day registry entries - sufficient evidence only, Q-M-11), STATE_AT_INDEX_UNVERIFIABLE
(`polypharmacy_count_120d`: no record date in the extract - a stated limitation shared by every analysis). Nothing is imputed, replaced or reconstructed.

**Adapter timing scopes (`MeuhedetWideDatasetAdapter(timing_scope=...)`).** `cohort` (default, unchanged behaviour: the mapping's guard columns, one shared
cohort for STRICT / EXTENDED) and `built_predictors` (guard = the record-date / evidence columns of the predictors actually built; the D-00 rows of unread
sources are kept and counted as `cohort_guard_rows_retained`). The start-of-index-day rule is not weakened in either scope.

**Ablations.** EXTENDED without each excluded predictor and without both, on the reference rows and test partition (hash-verified, values compared column by
column with the reference dataset): the feature-set effect. SAFE-ALL-ROWS vs the joint ablation (same 13 predictors, different rows, redrawn split) is the
population effect and is reported as unpaired.

**Convergence audit.** From `run_log.jsonl` and `metrics.json`: warnings of the final fit vs the stability / optimism replicates, lambda* grid position
(index, distance to the grid end, Stata stop index), CV minimum identified, fold paths converged, unpenalised-refit status (descriptive only), FP form
stability across replicates, perfect-predictor omissions per column and replicate. A solver / grid change is proposed only when lambda* is on the grid
boundary. No solver setting was changed in this version.

**Outputs.** `feature_provenance.md`, `convergence_audit.md`, `SENSITIVITY_COMPARISON.md/.json`, `REAL_DATA_EXPLORATORY_REPORT.md` (plain language, 14
sections), per-analysis split audits, adequacy gates and feature reports; `reference_integrity.json`. CLI: `meuhedet-sensitivity --input <same file>
--reference <explore results> --out <new folder>`.


## v0.6.0 - full EDA layer and model reporting

- `falls_ml meuhedet-eda`: reproducible, aggregate EDA of the whole wide extract with an explicit data dictionary
  (`configs/meuhedet/wide_v1_data_dictionary.yaml`: domain, source, record-date column and meaning per column; NAME_ONLY / UNKNOWN meanings flagged
  for SME review), missingness classes, 250 data-quality checks, a temporal audit of every date column, the provenance chain, the cohort funnel,
  a split diagnostic, and TRAIN-only target-aware analyses. The split is the modelling split (with `--reference`: reproduced and hash-verified).
- `falls_ml model-report` (also at the end of `meuhedet-explore` / `meuhedet-sensitivity`): per-run training dashboards, the model comparison
  (paired only for identical test rows) and the Hebrew management reports (`configs/meuhedet/feature_labels_he.yaml`), figures as PNG + SVG.
- Nothing in the eFalls mapping, the contract, the D-00 guard or the modelling procedure changed.

## v0.7.0 - D-00 sensitivity framework

`falls_ml meuhedet-d00` traces every predictor to its sources and record dates (SAFE / UNSAFE / UNRESOLVED per cohort), derives the safe
feature sets from that graph, and runs the pre-specified matrix (FULL_LABELED vs D00_CLEAN x STRICT_SAFE / SAFE_EXTENDED; FULL_EXTENDED on
D00_CLEAN = the reference run, reused) and the ablations EXTENDED without falls / mobility_problems / both on the reference test rows. On the
wide extract only age and sex can be PROVEN safe (registries carry only a first-entry date, medications no date), so the primary STRICT set is
renamed STRICT_SAFE_SUBSET; a pre-declared secondary standard also admits the unresolved predictors. Real-data results exist only after the
command runs on the work computer; nothing here is a real-data result. The management report now shows permutation importance ("contribution
to the model's prediction", on VALIDATION) separately from bootstrap selection stability.


## v0.7.1 - resume an interrupted D-00 analysis; bootstrap convergence retry

`meuhedet-d00 --resume` continues an interrupted analysis in the same output folder: the frozen plan must re-derive identically; completed runs
(including the three pre-specified ablations) are reused as they are - never refitted, their test sets never re-evaluated - and only an incomplete
cell that never released its test set is refitted; every file of the earlier attempt is verified unchanged. Bootstrap stability and optimism
replicates whose full-data LASSO solve did not converge are refitted once on the same resample with the same seed, the same selected lambda and
the same tolerance at a 10x higher full-data iteration limit, and accepted only if they converged; the 10% failed-replicate stop is unchanged.
No real-data result is contained in this change.
