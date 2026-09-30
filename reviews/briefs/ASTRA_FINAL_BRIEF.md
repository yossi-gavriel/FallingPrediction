# Final parameter review – Meuhedet Falls Phase 2: the frozen configuration of the only real-data run

You are the statistical / clinical-methods reviewer. You reviewed the Phase 2 *plan* once before, and your recommendations were adopted or
answered (§10). This is the **final** review of the ACTUAL configuration, before one multi-hour real-data run on a Windows work PC. You see
no data and no code. Every value below is the real frozen value (configs/meuhedet/phase2.yaml v1.1.0, phase2_features.yaml v1.1.0, code
0.8.1). Crash/resume engineering was reviewed and rehearsed separately. Do not review it.

## Required output (decision-oriented, ≤ 2,500 words)

1. Q1–Q24 (§11): one short answer each (≤ 5 lines).
2. Recommendations F-01, F-02, … Each one gets:
   - a severity: **MUST FIX BEFORE RUN**, **SHOULD FIX BEFORE RUN**, **ACCEPTABLE AS IS** or **FUTURE IMPROVEMENT**;
   - the exact change: parameter → value, or rule text;
   - a one-line rationale.
3. A one-line verdict: is the configuration scientifically fit for the run once the MUST items are fixed?

The goal is not global hyper-parameter optimality. The goal is:

- a credible answer to whether new SAFE features and/or non-linear boosting add incremental value;
- a management story on the same VALIDATION rows at matched capacity.

Hundreds of fits per setup are not acceptable without a compelling statistical reason.

## 1. Data and scientific context (facts)

**Rows and timing**
- One row per patient, age ≥ 65, one index date (2025-01-01).
- Prediction happens at the START of the index day. The predictor record rule is `source_event_date < Index_Date` and is never weakened.

**Cohort and outcome**
- Cohort: 60,851 patients, 1,282 events (**2.11%**).
- Outcome: `Fall_Next_180D_Ind`, a recorded fall/fracture within 180 days. It is an exploratory label, not the eFalls 365-day outcome. Deaths count as non-events; censored rows are excluded.
- Known limitation: the Phase 1 "D00_CLEAN" cohort removed 4,168 rows whose Last_Dx_Date or Last_Fall_Date was on or after the index day. Inclusion therefore depends on information that is not available at prediction time.

**Split (Phase 1 reference, reproduced exactly and hash-checked)**
- Patient-grouped, stratified, seed 42.
- TRAIN 60% (≈ 36.5k rows, ≈ 769 events); VALIDATION 20% (≈ 12.2k, ≈ 256); TEST 20% (≈ 12.2k, ≈ 256).
- **TEST was examined repeatedly in Phase 1 and is burned.** Phase 2 identifies it in the split audit (count and hash), drops it in memory before any computation, and never loads it again.
- VALIDATION was used in Phase 1 for recalibration, so it counts as reused.

**Historical benchmark BASELINE_15**
- The 15 EXTENDED eFalls-mappable predictors: age, sex, polypharmacy count, asthma, CKD, COPD, dementia, diabetes, prior falls, housebound, hypertension, liver problems, mobility problems, self-harm, SMI.
- Fitted in Phase 1 as a penalised LASSO with fractional-polynomial (FP) age / polypharmacy, then recalibrated on VALIDATION.
- Phase 1 TEST results: AUROC ≈ 0.814, PR-AUC ≈ 0.197, Brier ≈ 0.018.

**Provenance (D-00 rules, unchanged)**
- Each column gets SAFE / UNSAFE / UNRESOLVED from record dates on the TRAIN + VALIDATION rows (never outcomes).
- A feature takes the worst status of its inputs.
- UNSAFE sources are excluded ("candidate after DWH fix"). There is no row removal, no masking and no value replacement.

**Design-time accounting (exact; the data decides the "can be SAFE" ones)**
- **221 raw columns:**

  | Disposition | Columns |
  |---|---|
  | Forbidden (ids, labels / follow-up, contract leakage, post-index or unknown timing) | 51 |
  | Metadata | 52 |
  | Record dates | 12 |
  | Quarantined (undocumented codes) | 8 |
  | BASELINE_15 sources | 16 |
  | Represented by another feature | 3 |
  | From dated sources (SAFE / UNSAFE / UNRESOLVED decided by the data) | 64 |
  | From sources with no record date (UNRESOLVED) | 15 |

- **93 engineered candidates** in 10 domains:
  - 69 can be SAFE if their record dates prove it.
  - 24 cannot be SAFE:
    - 16 have no record date: 10 of the 11 MEDICATION features, 4 UTILISATION, 1 social, 1 comorbidity.
    - 8 are registry-based, where the only date is a first-entry date.
- **BASELINE_15:** at most 4 can be SAFE (age, sex, and prior falls and mobility problems if their dates prove it in this cohort). 11 are structurally UNRESOLVED: 10 registry flags with first-entry dates only, and polypharmacy with no record date.

**New user rule for this final version: three categories, never mixed**
- HISTORICAL_BASELINE_15: the benchmark, reproduced.
- SAFE_DISCOVERY: the PRIMARY category. Only SAFE features, new and baseline.
- EXPLORATORY_UNRESOLVED_SENSITIVITY: separate.
- This replaces the earlier design, whose main track was SAFE + UNRESOLVED.

## 2. Split, CV and seeds (actual)

**Roles**
- TRAIN: all screening, redundancy, encoding decisions, tuning, feature and model selection, stability, ablation and SHAP.
- VALIDATION is sealed until `SELECTION_FROZEN.json` exists. It is then opened ONCE (`VALIDATION_OPENED.json`, write-once) to score the frozen configurations. It is never used to tune, threshold or recalibrate.

**Outer CV:** 5 folds inside TRAIN, outcome-stratified, 1 repeat. One row per patient, so the folds are also patient-grouped.

**Inner CV**
- Linear models: 10 folds, random permutation of patients (grouped, NOT stratified; the Phase 1 LASSO code).
- XGBoost: 5 folds, outcome-stratified.

**Where the steps run**
- Label-blind steps run once on all TRAIN rows:
  - redundancy clusters;
  - which thermometer levels exist;
  - NULL-pattern groups.
- Fitted inside every fold:
  - the eFalls preprocessor, including FP selection for age / polypharmacy;
  - median fills;
  - standardisation;
  - hyper-parameters;
  - tree counts.

**Seeds**
- Master seed 20260929.
- Item seed = sha256(seed | stage | item). It does not depend on execution order or restarts.
- Optuna trial seed = sha256(seed | study | trial index); the sampler is re-created for every trial.
- Bootstrap seed = the item seed.
- Threads = min(4, cores), frozen at start.

## 3. Features, encoding, missingness (actual)

**Candidate features**
- 93 engineered candidates with row-wise, parameter-free operations: copy, flag_lt/ge, any_positive, days_since, present.
- Examples:
  - Mini-Cog score and abnormal (< 3); nurse cognitive impairment;
  - nurse polypharmacy; narcotics; prescriptions written / purchased / not collected;
  - MEFI frailty group, previous group, worsened, days since assessment;
  - ADL worst score, transfer dependency, mobility score, walking aid, balance / shuffling / hesitant gait, muscle weakness;
  - prior-fall counts at 30 / 90 / 180 / 365 days, days since the last fall (bands), nurse-reported falls;
  - nurse falls-risk score 1–3, dizziness, fear of falling, orthostatic hypotension, vision / hearing scores;
  - visit counts at 30–365 days, labs, external care, HMO seniority;
  - home-safety items;
  - no caregiver, income support;
  - diagnosis counts, CCI group, registries.
- 8 undocumented-code columns are quarantined and are never an input.

**"Not measured ≠ negative"**
- Each of the 6 assessment forms has ONE "assessed" indicator, derived from the presence of the form's record date. It is automatically added to every set that uses an item of that form.
- Items are coded 0 when NULL, covered by the form indicator.
- An item gets its own `na__` indicator only if its NULL pattern differs from the form's on more than 1% of TRAIN rows.
- Other NULLs get a fold-median fill plus an `na__` indicator (identical NULL patterns share one indicator).

**Linear encoding**
- Binary: 0/1.
- Ordinal: thermometer dummies (≥ level) for the levels present on TRAIN.
- Counts / days / amounts: log1p.
- Days since the last fall: bands ≤ 90, 91–180, 181–365, > 365, with "no prior fall" as the reference.
- BASELINE_15 block: the unchanged eFalls preprocessor (reference coding, FP terms).
- Design columns are z-scored inside the solver.
- Exact duplicate columns are dropped per fold.

**Trees:** raw engineered values with NULL kept (native missing handling), sex coded female 0/1, form indicators included. No log, no thermometer, no imputation.

**Redundancy:** representative-anchored |Spearman| ≥ 0.90, computed SEPARATELY in:
- the SAFE universe (SAFE features + SAFE baseline);
- the exploratory universe.
The representative is the baseline feature if present, else the one with the lowest missingness, then catalogue order.

**Eligibility:** at least 100 observed TRAIN rows and not constant; otherwise INELIGIBLE_DATA.

## 4. Feature sets (rules; membership decided by data)

**HISTORICAL_BASELINE_15:** BASELINE_15.

**SAFE_DISCOVERY**
- `SAFE_BASE` = the SAFE subset of BASELINE_15.
- `ALL_REVIEWED_SAFE` = SAFE_BASE + every SAFE non-redundant feature.
- `EFALLS_MAPPABLE_SAFE`, `MEUHEDET_NATIVE_SAFE`.
- `SAFE_BASE_PLUS_<domain>` for each of the 10 domains.
- The cumulative waterfall, in the declared order:
  - BASELINE_15 (benchmark)
  - → SAFE_BASE
  - → + COGNITION → + MEDICATION → + FRAILTY → + FUNCTION_MOBILITY → + UTILISATION → + HOME_ENVIRONMENT
  - → + all other domains = ALL_REVIEWED_SAFE ("ALL_SAFE_LINEAR")
  - → ALL_SAFE_ELASTIC_NET → ALL_SAFE_XGBOOST.
- `ALL_REVIEWED_SAFE_NO_ASSESSMENT_FLAGS` (sensitivity).
- `ALL_REVIEWED_SAFE_MINUS_<domain>` (leave one domain out).
- Invariant, enforced as a hard stop before any fit: every SAFE_DISCOVERY set holds SAFE features only.

**EXPLORATORY_UNRESOLVED_SENSITIVITY (LASSO only; never a selection candidate)**
- `EXPLORATORY_B15_PLUS_SAFE` = BASELINE_15 + SAFE new features.
- `EXPLORATORY_ALL_REVIEWED` = BASELINE_15 + SAFE + UNRESOLVED new features.

Identical specifications share one fit.

## 5. Models (actual values)

**LASSO**
- Implementation: Phase 1 Stata-equivalent penalised logistic regression.
  - L1 penalty on standardised coefficients; the intercept is unpenalised.
  - Solver: proximal-Newton (IRLS) with covariance-mode coordinate descent and warm starts.
- λ grid: 100 values, log-spaced from λ_max down to **1e-3 × λ_max**.
  - This was measured: below that, rare binaries are near-separated, and the path does not converge.
  - A λ* on the grid boundary is flagged.
- Selection: λ_min, the minimum mean held-out deviance (= 2 × log loss) over the 10 inner folds.
- Convergence: tol 1e-7, max_iter 10,000.
  - Retry policy: one retry with a 10× full-path iteration limit at the same λ*. Never accepted unconverged.
- No class weighting.
- The served coefficients are the penalised ones (no unpenalised refit).

**Elastic net**
- Same solver, glmnet parameterisation; identical to LASSO at l1_ratio 1.
- l1_ratio ∈ {0.1, 0.5, 0.9}; 100 λ values per ratio.
- (l1_ratio, λ) are chosen jointly by the minimum inner-CV deviance.
- Same tol / max_iter; no class weighting.
- Fitted on at most 2 sets: ALL_REVIEWED_SAFE, plus the best SAFE LASSO set by outer-OOF AP if that is different.

**XGBoost 3.2.0** (native `xgb.train`, CPU)

- **Fixed parameters:**
  - objective `binary:logistic`, eval_metric `logloss`;
  - booster gbtree, tree_method hist, grow_policy depthwise, max_bin 256;
  - learning_rate 0.05;
  - subsample 0.8, sampling_method uniform;
  - colsample_bytree 0.8, colsample_bylevel 1, colsample_bynode 1;
  - gamma 0, reg_alpha 0, max_delta_step 0;
  - **scale_pos_weight 1 (no class weighting)**;
  - base_score: XGBoost's own estimate from the fitting rows' labels.
- **Per fit:** seed = item seed; nthread = the frozen thread count. hist on CPU with fixed threads is bit-deterministic (checked by a self-test).
- **Trees:** at most 3000 (`n_estimators`); early_stopping_rounds 100.
- **Early stopping:** inside each inner fold, on that fold's held-out rows (log loss). A scope's final fit uses the median inner best iteration + 1 trees on all of its training rows. The outer held-out fold is never seen.
- **Tuning scopes:** 5 outer folds + 1 final (all-TRAIN) scope = 6, each tuned independently (nested).

**XGBoost stages (per scope)**

| Stage | Sets | Search | Trials per scope |
|---|---|---|---|
| 0 (sanity) | BASELINE_15 and ALL_REVIEWED_SAFE | fixed: max_depth 3, min_child_weight 10, reg_lambda 5 | – |
| 1 | ALL_REVIEWED_SAFE | Optuna TPE: max_depth int 1–3; min_child_weight 5–30 (log); reg_lambda 1–30 (log) | **15** (6 random start-up) → **90 trials total** |
| 2 (conditional) | ALL_REVIEWED_SAFE | local search around each scope's stage-1 best: depth ± 1; min_child_weight and reg_lambda within best/2 … best×2 (clipped to the stage-1 range); plus subsample 0.6–1.0 and colsample_bytree 0.5–1.0 | **≤ 8** (2 start-up) → ≤ 48 trials total |

- Stage 2 runs only if the stage-1 outer-OOF AP ≥ the best SAFE linear outer-OOF AP + 0.005. The decision is taken once, from the committed results.

**Optuna**
- TPESampler(multivariate=True), direction minimize.
- Objective: **inner-CV mean held-out log loss**. There is no pruner.
- Storage: persistent SQLite, rebuilt from the committed trials at every attempt; plus an append-only ledger.
- Failure handling: a trial is a committed item. An interrupted trial leaves no trace; an error stops the run.
- Promotion uses outer-OOF AP, never VALIDATION.

## 6. Selection, validation, metrics (actual)

**Primary metric:** AP (PR-AUC, step-wise, sklearn-equivalent).

**Secondary metrics**
- Log loss (the tuning objective and a safeguard).
- AUROC.
- Brier and scaled Brier.
- CITL (slope fixed at 1).
- Calibration intercept and slope.
- O:E.

**Capacities**
- Top 1 / 3 / 5 / 10 / 20% of risk: ceil(fraction × n) highest risks, ties in row order.
- Reported at each capacity: falls identified, sensitivity (capture), PPV, false alerts, lift, specificity, NPV.
- Threshold table: 0.01–0.09 in steps of 0.01, then 0.10–0.50 in steps of 0.025.
- Calibration curves: 10 risk groups.

**Advancement rule** (frozen in SELECTION_FROZEN.json before VALIDATION is opened)
- Candidates: SAFE_DISCOVERY main / one-domain / waterfall sets, across the families LASSO, EN and XGB (stage 0 and tuned).
- A candidate qualifies against **LASSO:SAFE_BASE** if ALL of the following hold on TRAIN outer-OOF predictions:
  - ΔAP ≥ **0.010**;
  - Δcapture at the top **10%** (principal capacity) ≥ **+1.0 pp**;
  - log loss ≤ **1.01 ×** the reference's;
  - calibration slope within **0.80–1.25**;
  - |CITL| ≤ **0.20**.
- Recommended model: the simplest (linear before XGB, then fewer features used) within **0.005 AP** of the best qualifying one; else SAFE_BASE.
- The recommended model is then classified against **LASSO:BASELINE_15** (reported, never used to choose):
  - SUPERIOR: meets the same rule;
  - NON_INFERIOR: ΔAP ≥ −0.010 and Δcapture@10% ≥ −1.0 pp;
  - INFERIOR: otherwise.
- Shortlist: SAFE_BASE, BASELINE_15, the recommended model, then the best SAFE candidate of each other family (≤ 3 challengers).

**VALIDATION (one shot)**
- Every frozen configuration is scored on the same rows, for the waterfall and the management story.
- Confirmation (pre-declared): the recommended model vs SAFE_BASE.
  - CONFIRMED_PROMISING if ΔAP ≥ 0.005, Δcapture@10% ≥ 0 pp, slope within 0.70–1.40 and |CITL| ≤ 0.30;
  - otherwise NOT_CONFIRMED.
- Benchmark class vs BASELINE_15: the same superiority margins, and non-inferiority −0.010 AP / −1 pp.
- Paired patient bootstrap: 2000 replicates, percentile 95% intervals of the frozen predictions (Δ AP, AUROC, log loss, Brier, capture@10%, false alerts@10%). The search is acknowledged as unaccounted for.
- Management sentence: at each capacity, the additional recorded falls identified by the recommended SAFE model vs BASELINE_15 on the same VALIDATION rows, and per 10,000 members.
- No recalibration of any model in Phase 2.

## 7. Stability, ablation, explainability, consensus (actual)

**Stability**
- 25 LASSO fits on bootstrap resamples of TRAIN (inner CV repeated), on ≤ 3 finalist sets:
  - the SAFE linear shortlist sets;
  - ALL_REVIEWED_SAFE;
  - BASELINE_15 as the reference.
- Outputs: selection frequency and sign consistency. The run stops if > 10% of replicates fail.

**Ablation**
- Leave one domain out: LASSO, fully nested (retuned), against LASSO:ALL_REVIEWED_SAFE on the same outer folds.
- Individual: the top 15 new features of the recommended model by final-model importance, refitted without the feature using each fold's frozen hyper-parameters (explanatory).

**Explainability** (only the XGB member of the shortlist)
- Permutation importance per outer fold on that fold's held-out rows: 5 repeats, ΔAP and Δlog loss.
- TreeSHAP of the final booster on a class-stratified TRAIN sample of 12,000 rows. Only aggregated quantiles are reported.
- Top-5 interaction pairs.

**Consensus** (evidence classes, not a vote)

| Class | Evidence |
|---|---|
| U (univariate) | \|SMD\| ≥ 0.10 or univariate AUROC outside 0.45–0.55 |
| L (linear) | selected by LASSO or EN on ALL_REVIEWED_SAFE, with stability ≥ 0.60 |
| I (importance) | SHAP rank ≤ 15 or permutation ΔAP > 0 |
| A_dom (domain) | one-domain addition ΔAP vs SAFE_BASE ≥ 0.005 |
| A_feat (feature) | individual-ablation ΔAP ≥ 0.002 |

- STRONG = A_feat + (L or I).
- MODERATE = (L or I) + (U or A_dom), or A_feat alone.
- WEAK = any single class.

## 8. Investigation gates (never auto-accepted; the user continues with a recorded reason)

| Gate | Trigger |
|---|---|
| SINGLE_FEATURE_DOMINANCE | a new feature's univariate AUROC ≥ 0.80 (or ≤ 0.20) on TRAIN |
| IMPLAUSIBLE_GAIN_OOF / _VALIDATION | any SAFE or exploratory configuration vs BASELINE_15: ΔAUROC > 0.05 or AP ratio > 1.5 |
| IMPLAUSIBLE_CALIBRATION_OOF / _VALIDATION | slope outside 0.6–1.5 or \|CITL\| > 0.5 |
| PROXY_DOMINANCE | one new feature holds > 50% of mean \|SHAP\| or of the permutation AP loss |

Hard stops (non-exhaustive): split / cohort / label mismatch, patient overlap, an ineligible feature in a set, a changed input / plan / code / protected folder, fit failures > 10%, and the privacy scan.

## 9. Compute

- Measured on synthetic data at 29k TRAIN rows: 3.2 h on an M4.
- Expected 5–10 h on the work PC.
- Roughly:
  - ≤ 24 LASSO sets × 6 nested fits;
  - EN 2 × 6 fits × 3 ratios;
  - XGB 90 + ≤ 48 trials (5 inner folds each) + 12 stage-0 fits;
  - 75 stability fits;
  - ≤ 10 × 5 + 15 × 5 ablation fits.

## 10. Your earlier recommendations (already adopted; re-open only if now wrong)

- Selection is done on TRAIN, with ONE validation comparison of a frozen shortlist.
- Nested CV for every supervised and fitted step.
- AP with a fixed implementation, plus log loss as a safeguard.
- One principal capacity (10%).
- No class weighting.
- A small, regularised XGB space, with early stopping inside the inner CV.
- Penalised predictions are primary.
- UNRESOLVED is exploratory. UNSAFE sources are excluded, not masked (masking was rejected because it replaces valid source values).
- The waterfall is order-dependent and is reported with one-domain additions and leave-one-domain-out.
- Bootstrap intervals ignore the search.
- Consensus is not a vote.
- The next step is a freeze list before any temporal holdout.

## 11. Questions (answer each)

1. Is the selection design statistically valid?
2. Is nested CV inside TRAIN implemented correctly for this use case?
3. Is opening VALIDATION only after `SELECTION_FROZEN.json` appropriate?
4. Is PR-AUC the correct primary discovery/selection objective for ~2.1% prevalence?
5. Should XGBoost tune directly for PR-AUC or another objective/metric combination?
6. Is the current class-imbalance handling appropriate?
7. Should `scale_pos_weight` be used, fixed, tuned, or avoided?
8. Are the current LASSO regularization/λ settings and CV search sensible?
9. Are the Elastic Net l1-ratio choices sufficient?
10. Are the XGBoost search ranges sensible for ~60k rows and this prevalence? (Note: min_child_weight is in hessian units; at p ≈ 0.02 a row contributes ≈ 0.02, so 5–30 means ≈ 250–1,500 rows per leaf in ≈ 23k inner-training rows.)
11. Is early stopping configured correctly and without leakage?
12. Is the Optuna budget sufficient to find direction without brute force?
13. Are the promotion criteria from Stage 1 → Stage 2 statistically sensible?
14. Is top 10% reasonable as the principal management capacity while still reporting other capacities?
15. Are calibration and Brier being used correctly?
16. Is the planned stability/bootstrapping budget sufficient for finalists?
17. Is the ablation strategy scientifically useful?
18. Are any proposed engineered features likely to create leakage or proxy leakage?
19. Are missingness indicators being handled appropriately?
20. Could assessment presence itself become a healthcare-process proxy, and how should it be interpreted?
21. Should high-missingness but potentially strong nursing features be retained, encoded differently, or analysed only in subcohorts?
22. Is there any parameter or design choice that should be changed BEFORE the real run? (Include the new SAFE_DISCOVERY-primary structure: the advancement rule against SAFE_BASE plus the benchmark classification against BASELINE_15.)
23. What are the top 5 failure modes you would inspect if XGBoost shows a surprisingly large gain?
24. Is anything missing that would materially weaken the scientific validity of Phase 2?
