# Phase 2 reconnaissance: Meuhedet falls prediction (feature discovery and model families)

Status: RECONNAISSANCE ONLY. No code was changed, no model was fitted, and no real data was read to write this document.
Author: lead engineer (Claude). Date: 2026-09-29. Code base: falls_ml 0.7.1 (`src/falls_ml`, 1,892 fast tests + 30 slow tests passing on
Python 3.11 and 3.13).

**Access boundary.** The real extract (`60k_falling_db.csv`, 66,207 rows × 221 columns) and every real run output exist only on the Windows
work PC. The numbers below from the real data were reported by the user from locally generated aggregate outputs. The Phase 2 run itself
will execute on the work PC; this machine only builds and tests the software on synthetic data.

---

## 1. Scientific baseline (inputs to Phase 2, reported by the user)

| Item | Value |
|---|---|
| Extract | `V_Falls_Prediction_Wide_1` (Definition V1.0), one snapshot (Index_Date 2025-01-01), 66,207 rows |
| Labelled cohort (FULL_LABELED) | 65,019 rows |
| Modelling cohort (D00_CLEAN) | 60,851 rows / patients, 1,282 events (2.11%); 4,168 rows removed for D-00 (Last_Dx_Date or Last_Fall_Date on/after the index day) |
| Split (reference explore run) | patient-grouped random, stratified, seed 42, validation 20%, test 20%. Expected ≈ 36.5k / 12.2k / 12.2k rows and ≈ 769 / 256 / 256 events (exact counts are in `split_audit.json` on the work PC) |
| EXTENDED (15 predictors) | AUROC ≈ 0.814, PR-AUC ≈ 0.197, Brier ≈ 0.018 (TEST, recalibrated served variant) |
| D-00 ablations (0.7.x) | prior falls carry much of the STRICT → EXTENDED gain; removing mobility alone changes AUROC very little |
| Test partition | examined repeatedly → **burned for Phase 2**: never loaded after the cohort/split reconstruction step |

## 2. How the current 15 predictors are built (confirmed from code/config)

Source: `configs/meuhedet/wide_v1_efalls_mapping.yaml` (mapping 1.0.1), built by `falls_ml.data.meuhedet_wide` (`MeuhedetWideDatasetAdapter`).

| Feature | Quality | Operation | Source column(s) | NULL rule | Timing evidence |
|---|---|---|---|---|---|
| age_years | HIGH | copy_numeric (FP-selected form) | Age_At_Index | forbid | static attribute |
| sex | HIGH | recode 1→male, 2→female | Gender_Code | forbid | static attribute |
| polypharmacy_count_120d | APPROX | count (FP-selected form; `log((x+1)/10)` family) | Distinct_Active_Substance_Count | 0 only when Medication_Missing_Ind = 1 | no record date (UNRESOLVED) |
| falls | APPROX | any_positive | Prior_Fall_Since_Study_Start_Ind | 0 only when Prior_Fall_Missing_Ind = 1 | Last_Fall_Date (D-00 guard column) |
| mobility_problems | APPROX | any_positive | Gait_Disorder_Since_Study_Start_Ind (ICD-9 719.7) | error on NULL | Last_Dx_Date (D-00 guard column) |
| dementia, copd, asthma, liver_problems, hypertension, housebound | HIGH | any_positive | Registry_{Dementia, COPD, Asthma, Liver, Blood_Pressure, Home_Confined}_Ind | 0 only when Registry_Missing_Ind = 1 | First_Registry_Date (sufficient-only → UNRESOLVED) |
| diabetes_mellitus | HIGH | any_positive of T1 ∪ T2 | Registry_Diabetes_T1_Ind, Registry_Diabetes_T2_Ind | as registries | as registries |
| chronic_kidney_disease | APPROX | any_positive | Registry_Chronic_Renal_Failure_Ind | as registries | as registries |
| severe_mental_illness | APPROX | any_positive of level ≥ 1 | Registry_SMI_Level | NULL = absent (declared) | as registries |
| self_harm | APPROX | any_positive | Registry_Suicide_Attempt_Ind | as registries | as registries |

- **Representation:** `efalls_fp_all_levels`. Fractional polynomials are selected for age and polypharmacy (max degree 2, α 0.05). Binaries are coded 0/1.
- **Model:** `lasso_logistic_cv` (own coordinate-descent solver, 10-fold patient-grouped CV, 100 λ, λ_min). Predictions use the
  **penalised** coefficients at λ*. An unpenalised refit on the selected columns is produced only as an odds-ratio reporting table.
  *(Correction 2026-09-29: the first draft of this file and the Astra brief wrongly said the refit is the served model.)*
- **Calibration:** logistic intercept + slope recalibration fitted on VALIDATION. Metrics are reported for both variants.
- **Test-evaluation registry:** the test set is released once per run, after stability and optimism.
- **eFalls coverage:** 15 of 78 eFalls predictors are mappable (9 HIGH + 6 APPROX). The other 63 are UNAVAILABLE in V1.0; for 19 of them
  a nurse-assessed field exists, which may be used only in Phase 2 because NULL means "not assessed".

## 3. The 221 columns (contract `configs/meuhedet/wide_v1_columns.yaml`, dictionary `configs/meuhedet/wide_v1_data_dictionary.yaml`)

| Contract role | Columns | Phase 2 disposition (proposed) |
|---|---|---|
| EFALLS_BASELINE_FEATURE | 16 (→ 15 features) | BASELINE_15; also re-engineered variants where useful (e.g. fall counts) |
| MEUHEDET_ENHANCED_FEATURE | 108 (90 non-date values + 18 date columns) | candidates; each date column becomes a record-date guard / "days since" source, never a raw predictor |
| FORBIDDEN_LEAKAGE | 12 | FORBIDDEN (billing-lag external care, abroad flag, index-day fall, new-registry counters, fairness-audit statuses) |
| LABEL | 16 | FORBIDDEN (outcome) |
| COHORT_ELIGIBILITY | 21 | NON_PREDICTIVE_METADATA or FORBIDDEN (post-index follow-up/censoring) |
| QA_CONTROL | 46 | NON_PREDICTIVE_METADATA (some are coverage flags; see §7 R-06) |
| IDENTIFIER | 2 | FORBIDDEN, never leave the work PC |

Phase 2 candidate groups declared in the contract (value columns allowed as predictors): nurse fall assessment 23, healthcare utilisation 13
(+6 hospitalisation columns blocked: timing unknown), MEFI frailty 9, home safety 8, registry enhanced 8, prior falls enhanced 6, diagnosis
burden 5, mobility/ADL 5, medication enhanced 6, comorbidity/social 5, cognition 2 (MiniCog score, nurse cognitive impairment; MMSE is empty
per the S2T).

Record dates available for timing proof (data dictionary `sources`): Last_Fall_Date, Last_Dx_Date, Last_Visit_Date, Falls_Risk_Assessment_Date,
Get_Up_And_Go_Date, Mobility_Assessment_Date, MiniCog_Date, Home_Safety_Assessment_Date, Last_Assessment_Date, MEFI_From_Date,
First_Registry_Date (sufficient-only), Last_Ext_Date / Last_Hosp_Date (forbidden sources). **No record date** exists for prescriptions,
labs, medication exposure, comorbidity/social (CCI_Group, Malnutrition_Grade, Siudi_Status, Customer_Risk_Code, Income_Support) or the
source-coverage flags.

**Not available at all in V1.0:** medication classes / ATC groups (no fall-risk-increasing-drug columns; only Narcotic_Drug_Count), BMI,
smoking, fracture separate from falls, condition-level diagnoses beyond the registries and 719.7, a Get-Up-And-Go *score* (only its date), and a
365-day outcome. "Recent medication change" cannot be built (no dispensing dates).

## 4. What already exists and can be reused

| Need | Existing component | Reuse |
|---|---|---|
| Read the extract (types, DD/MM/YYYY dates, sentinels, contract checks) | `data/meuhedet_wide.py` (`read_wide_csv`, contract, `coerce_wide_types`) | as is |
| Rebuild D00_CLEAN exactly as the reference run and reproduce its split | `eda/runner._cohort_and_split` (pepper check, test-row sha256 must equal the reference, every row mapped to train/validation/test); `d00` does the same row by row | extract into a shared function; Phase 2 drops TEST rows immediately after the hash check |
| Per-column D-00 provenance (SAFE / UNSAFE / UNRESOLVED per cohort, per source, transitive derivations) | `d00/dependency.py` (`build_graph`, `_column_status`, evidence kinds from `configs/meuhedet/d00_sensitivity.yaml`) | status of every raw column on TRAIN+VALIDATION rows |
| Column profiling, missingness classes (NOT_MEASURED vs SOURCE_UNAVAILABLE…), co-missingness, TRAIN-only univariate screen (SMD, univariate AUROC, prevalence ratios, sparse/separation flags), Spearman/Cramér's V redundancy clusters | `eda/{profile,missingness,supervised,quality,common}.py`, `TrainPartition` guard | reuse the functions on engineered features (new wrapper) |
| Phase 2 catalogue (readiness per column: BLOCKED / NEEDS SME / CANDIDATE AFTER DWH FIX / assessment-conditional …) | `eda/features.phase2_catalogue` → `phase2_candidate_features.csv` in the user's EDA folder | same rules seed the registry's eligibility; the real CSV stays on the work PC |
| LASSO logistic CV (grouped folds, λ grid, convergence diagnostics, unpenalised refit, retry policy) | `models/lasso_cv.py`, `evaluation/convergence_retry.py` | as is for the LASSO family |
| Elastic net | `models/logistic.ElasticNetLogistic` (sklearn saga, **fixed C, no CV**) | not suitable as is; see §5 |
| Gradient boosting | `models/hist_gbm.py` (sklearn HGB) | not XGBoost; kept only as a fallback idea |
| Metrics: AUROC (DeLong), PR-AUC, Brier, calibration slope/intercept/CITL, O:E, grouped/smoothed calibration, net benefit | `evaluation/metrics.py` | as is |
| Patient-cluster bootstrap CIs and paired differences | `evaluation/bootstrap.py`, `d00/analysis.paired_comparison`, `modelreport/compute.paired_differences` | as is |
| Capacity (top-k%) capture, lift, threshold tables with small-cell suppression, risk deciles | `modelreport/compute.py` (`lift_table`, `threshold_table`), `d00/analysis.risk_concentration` | extend with TP/FP/FN/TN, PPV/NPV, false alerts at 1/3/5/10/20% |
| Stability bootstrap with ≤ 10% failure guard | `evaluation/stability.py` | finalists only, 25 replicates |
| Grouped permutation importance | `evaluation/importance.permutation_importance_grouped` | generalise to any predict function (XGBoost) |
| Figures (ROC, PR, calibration, funnel, gains, h-bars, comparison, RTL Hebrew) and Hebrew management rendering with banned hype words | `modelreport/figures.py`, `modelreport/management.py` | reuse style and helpers; new waterfall figure |
| Atomic JSON write (temp + replace, Windows retry) | `artifacts.write_json`, `replace_file` | add fsync (file and directory) for power-loss safety |
| Code/environment provenance | `artifacts.source_tree_sha256`, `environment_info`, `git_commit` | extend with CPU/RAM/xgboost/optuna |
| Resume with immutability proofs (hash every earlier file before/after, append-only registry, plan sha re-derivation) | `d00/runner.py` (0.7.1) | same pattern, generalised to stage/item checkpoints |
| Privacy scan of shareable text (ids, pseudonyms, pepper, data URIs) | `eda/runner.privacy_scan`, `d00/runner._privacy` | as is on `share/` |
| CLI with STOPPED + resume hint, exit code 2 | `cli.py` | new sub-command |

## 5. What must be added

1. **Phase 2 orchestrator** (`falls_ml.phase2`, CLI `meuhedet-phase2`): stage graph, RUN_STATE.json, item and stage markers, append-only
   `logs/events.jsonl`, RUN_TIMINGS.csv, stop gates, resume with gate overrides that are always recorded.
2. **Column registry builder** (all 221 columns → `01_COLUMN_REGISTRY.csv`): contract, dictionary and mapping facts, plus data facts on
   TRAIN+VALIDATION (missing %, unique count, constant/near-constant, provenance status) and a disposition from pre-declared rules.
3. **Declarative engineered-feature catalogue** (`configs/meuhedet/phase2_features.yaml`) and engine. It covers ≈ 60–90 candidates:
   recency, window counts, latest score, abnormal flags, "assessed" indicators, days-since. The rules: explicit "not measured"; never NULL→0
   unless the contract declares an absence indicator; transforms fitted on TRAIN only.
4. **Generic design-matrix builder** for non-eFalls features. The eFalls Preprocessor is spec-bound; the new builder is fitted on TRAIN, with
   one-hot ordinals/categories, log1p counts, "not assessed" levels and median fill plus an indicator only where declared.
5. **Elastic-net path with CV**: extend the existing CD solver with an L2 term (`l1_ratio`, glmnet parameterisation). With `l1_ratio = 1` the
   code path must stay bit-identical to today's LASSO. Alternative: sklearn `LogisticRegressionCV(saga)` – slower and less consistent.
6. **XGBoost adapter + Optuna tuning** with persistent SQLite storage, a JSONL trial ledger, deterministic per-trial sampler seeding,
   recovery of stale RUNNING trials, and early stopping. New dependencies: `xgboost` (or `xgboost-cpu`), `optuna` (+ SQLAlchemy, alembic,
   Mako, colorlog, tqdm, greenlet). **SHAP is taken from XGBoost's built-in TreeSHAP** (`pred_contribs` / `pred_interactions`), so the
   `shap` package (numba/llvmlite) is not needed.
7. **Feature-set builder** (BASELINE_15, EFALLS_MAPPABLE_ALL, MEUHEDET_NATIVE_SAFE, ALL_REVIEWED_SAFE, 8 domain increments, cumulative
   waterfall), frozen with a sha before any fit.
8. **Consensus, ablation, waterfall, capacity and calibration tables, Hebrew management summary, scientific summary, share builder**.
9. **Synthetic Phase 2 fixture**: the existing panel generator plus planted signal in nurse, MEFI and utilisation fields, planted
   index-day assessment dates, and planted proxy leakage to prove the gates fire.
10. **Rehearsal harness**: kill the process mid-stage (including mid-Optuna trial and mid-bootstrap), resume, and prove identical results
    plus unchanged completed artifacts.

## 6. Proposed compute budget (work PC, CPU only)

Measured here on synthetic data at Phase 2 size (Apple silicon; expect the work PC to be 1.5–3× slower): see §6.1. Funnel:

| Stage | Work | Budget |
|---|---|---|
| Cohort/split, registry, engineering, TRAIN screening | exhaustive over all columns/candidates | < 15 min |
| LASSO per feature set | ≈ 16–18 CV fits (4 main sets + 8 domain increments + ≤ 6 cumulative steps) | ≈ 0.5–1 h |
| Elastic net | 3 l1_ratio values × ≤ 2 promoted sets | ≤ 1 h |
| XGBoost | stage 0 (1 config) + stage 1 (12–15 Optuna trials) + optional stage 2 (≤ 8), each trial = 5-fold grouped CV in TRAIN, early stopping | ≤ 1.5 h |
| Stability | 25 LASSO bootstrap replicates × ≤ 3 finalist sets | ≈ 1–2.5 h |
| Ablation | domain removal (≤ 8) + individual (≤ 15) LASSO refits; XGB with frozen params | ≈ 0.5–1.5 h |
| Explainability | XGB permutation importance (validation) + TreeSHAP on a ≤ 12k sample | < 15 min |
| Bootstrap CIs / reports / share | paired patient-cluster bootstrap on VALIDATION predictions | < 30 min |
| **Total** | | **≈ 4–10 h (one night)**; measured later on a full-size synthetic run: 3.2 h on an Apple M4 → ≈ 5–10 h expected on the work PC (planning/EXPERIMENT_PLAN.md §10) |

### 6.1 Measured cost of one LASSO CV fit (synthetic, 36,500 rows, 10 folds, 100 λ; this Mac)

| Design columns | Events (synthetic) | One full LASSO CV fit (10 folds + full path) |
|---|---|---|
| 15 | 650 | 5.5 s |
| 60 | 647 | 13.4 s |
| 150 | 570 | 39.4 s |

So the LASSO family is cheap: even at 150 design columns and 3× slower hardware, one fit takes ≈ 2 min. The budget is dominated by
stability replicates (75 fits ≈ 1–2.5 h) and by XGBoost CV trials. The multi-day scenario is unlikely, but the run is still designed to
survive a reboot at any point.

## 7. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R-01 | **Timing (D-00) of new sources.** The VIEW used `Event_Date <= Index_Date` for diagnoses and falls. Visits, nurse assessments and MEFI may carry the same defect | Record-date proof per source on TRAIN+VALIDATION; rule pre-declared (Astra Q): exclude the feature (UNSAFE) or define one Phase 2 cohort that excludes rows proven to carry index-day records, fixed before any fit |
| R-02 | **UNRESOLVED sources** (medications, labs, comorbidity index, benefit status: no record date) | Same evidence level as BASELINE_15's registries/polypharmacy; admitted only under the pre-declared SAFE_OR_UNRESOLVED standard, always labelled, with a SAFE-only sensitivity set |
| R-03 | **Assessment-conditional nurse fields** (NULL = not assessed; assessment is a care-process signal: selection by clinical suspicion) | Three-state encoding (positive / negative / not assessed); block-level "assessed" indicators; subgroup reporting assessed vs not; no causal wording; transportability caveat |
| R-04 | **Proxy/target leakage** (e.g. assessments triggered by an index-day fall; fall/fracture counts overlap the outcome definition) | D-00 guard; stop gates on single-feature dominance and implausible jumps; investigation report before acceptance |
| R-05 | **Validation overfitting** (≈ 256 validation events; ≈ 30–40 configurations compared) | All tuning inside TRAIN (grouped CV); validation only for a pre-declared, logged comparison list; paired bootstrap; TRAIN out-of-fold corroboration; final claims deferred to a new temporal holdout |
| R-06 | Coverage flags (`Has_Source_*`, `*_Missing_Ind`) are data-provenance metadata but could act as care-intensity predictors | Treated as NON_PREDICTIVE_METADATA except where a declared "not assessed" indicator is needed (explicit, per block) |
| R-07 | **Dependencies on the work PC** (xgboost, optuna not in the lock; install route online vs offline unknown) | Separate hash-pinned `requirements-phase2.lock` + offline-bundle support; preflight STOP with instructions if missing |
| R-08 | LASSO convergence failures with many correlated features (seen in 0.7.x) | Existing retry policy v1 for bootstrap replicates; redundancy pruning at |ρ| ≥ 0.9 before fitting; failure-fraction gate unchanged |
| R-09 | Power loss / reboot mid-stage (multi-hour run, Windows, antivirus/OneDrive locks) | Temp → fsync → atomic rename; item-level markers; Optuna SQLite + JSONL ledger; resume audit with hash chain |
| R-10 | Immutability of STRICT / EXTENDED / D-00 / EDA / report folders | Hash all protected folders at start and end; Phase 2 writes only below its own `--out`; any change → STOP |
| R-11 | Unknown code semantics (Siudi_Status, Customer_Risk_Code, Malnutrition_Grade, Fall_Self_Report_Value, SMI levels) | Disposition NEEDS_SME, excluded by default. A categorical encoding without semantics is possible but not interpretable; decision in REVIEW_DECISIONS |
| R-12 | Management over-interpretation (SHAP as causal, validation numbers as final) | Fixed wording rules (no "significant", no causal language, "exploratory, validation partition, not a final claim"), Hebrew banned-word list |

## 8. Proposed resume strategy (to be reviewed by AGY)

- **One frozen plan** (`PHASE2_PLAN.json`: config, feature catalogue, feature sets, seeds, compute budget; sha256). On resume the plan must
  re-derive to the same sha, and the input sha256 must match (otherwise STOP).
- **Work item = unit of recovery** (one LASSO fit of one set, one Optuna trial, one bootstrap replicate, one ablation). Each item writes to
  `…/.tmp-<item>/`, fsyncs every file, renames the directory atomically, then writes `ITEM_COMPLETE.json` (output sha256s). On resume,
  complete items are verified by hash and skipped; incomplete temp directories are moved to `_incomplete/<attempt>/`, never read.
- **Stage markers** `STAGE_XX_COMPLETE.json` hold the hashes of the stage outputs and of the upstream markers (a hash chain). Any change
  to a completed artifact is a STOP.
- **RUN_STATE.json**: stage, substage, completed stages, current item, dataset/registry/config/code hashes, seed, started_at, last_checkpoint,
  status, resume_count. Rewritten atomically after every item.
- **Optuna**: SQLite storage in `checkpoints/xgb_optuna.sqlite` plus an append-only `xgb_trials.jsonl` ledger (the source of truth for
  reports). The sampler is re-seeded from (base seed, trial number), so the suggestion for trial k is the same after a restart. A trial left
  RUNNING by a crash is marked FAIL (reason `interrupted`) and its parameters re-enqueued as the next trial.
- **Code changes between attempts** (e.g. a patch that fixes a crash) are refused unless `--accept-code-change "<reason>"` is given; the
  reason, both code hashes and the affected stages are written to RESUME_AUDIT.json. Completed items are never recomputed.

## 9. Delivery order (as requested)

1. RECON (this file) → 2. compact Astra method review → 3. implementation architecture (`planning/ARCHITECTURE_PHASE2.md`) → 4. compact
AGY engineering review → 5. `reviews/REVIEW_DECISIONS.md` → 6. final experiment plan (`planning/EXPERIMENT_PLAN.md` + frozen config) →
7. rehearsal proof on synthetic data (interruptions + resume) → 8. the Windows command for the full run on the work PC.
