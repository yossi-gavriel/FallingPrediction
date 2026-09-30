# Phase 2 experiment plan (pre-declared; frozen into `PHASE2_PLAN.json` at the start of every run)

Version **1.1.0 – FINAL** (2026-09-29). It incorporates `reviews/REVIEW_DECISIONS.md` and the final parameter review
(`reviews/ASTRA_FINAL_PARAMETER_REVIEW.md`, resolved in `reviews/ASTRA_FINAL_RESOLUTION.md`).

- Machine-readable twins: `configs/meuhedet/phase2.yaml` (analysis settings) and `configs/meuhedet/phase2_features.yaml` (engineered-feature catalogue, domains, quarantine).
- **Authoritative effective values:** `configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json` and its `.sha256`. The runner and the preflight refuse any other effective configuration (`FROZEN_CONFIG_MISMATCH`).
- Nothing below may be changed after a run has started. A changed config gives a different plan sha, and resume refuses it.

## 1. Question, outcome, population

- **Question:** which additional pre-index information adds genuine predictive value for a recorded fall/fracture within 180 days (index
  day included) beyond BASELINE_15? Which model family and feature set should advance to a new, preferably temporal, holdout?
- **Outcome:** `Fall_Next_180D_Ind`. It is an exploratory label: NOT the eFalls 12-month ED/admission outcome. Censored rows are excluded
  (limitation).
  - Declared death rule: a recorded fall/fracture inside the window stays an event when death follows; death before any recorded event is a non-event.
  - The warehouse label is audited on TRAIN+VALIDATION (`label_death_audit` in COHORT_FACTS.json; printed by the preflight).
  - The investigation gate LABEL_DEATH_SEMANTICS stops the run if members who died inside the window never carry an event (Astra F-05).
- **Scope of every claim:** exploratory prediction of RECORDED events in the D00_CLEAN selected cohort, whose inclusion depended on
  information after the prediction time. VALIDATION was reused in Phase 1 (Astra F-03).
- **Population:** the reference D00_CLEAN cohort (rebuilt row by row and verified against the completed explore run) and the reference
  patient-grouped split. TRAIN + VALIDATION only; **TEST is dropped in memory right after the split hash check and never used.**
- **Prediction time:** start of the index day; a predictor may use only records dated strictly before Index_Date. Nothing is repaired: no
  date moved, no value replaced, no row removed beyond the reference cohort.

## 2. Evidence tracks and eligibility

- **Provenance status** of every raw column is computed with the unchanged Phase 1 D-00 rules (`d00/dependency._column_status`) on the
  TRAIN+VALIDATION rows (record dates only, never labels):
  - SAFE: record date proven before the index day on every row, and the meaning is DOCUMENTED;
  - UNSAFE: an on/after-index record is proven on ≥ 1 row, or the column is forbidden / post-index;
  - UNRESOLVED: anything else.
- An engineered feature takes the **worst** status over its input columns.
- **Discovery eligibility:**
  - ELIGIBLE (SAFE);
  - ELIGIBLE_EXPLORATORY (UNRESOLVED);
  - INELIGIBLE_UNSAFE ("candidate after DWH fix", with row counts);
  - INELIGIBLE_FORBIDDEN (identifier, label, post-index, forbidden, timing unknown);
  - NON_PREDICTIVE_METADATA (QA/cohort bookkeeping; declared absence/not-assessed indicators are used only as encoding helpers);
  - NEEDS_SME (quarantined undocumented codes);
  - INELIGIBLE_DATA (fewer than 100 observed TRAIN rows, or a single value).
- **Three categories, never mixed** (final readiness brief; they replace the earlier EXPLORATORY / PROVEN_SAFE tracks):
  - HISTORICAL_BASELINE_15: the 15 historical predictors (provenance as found), refitted on TRAIN in this protocol as
    `LASSO:BASELINE_15` (the "BASELINE_15 refit"). It is not the Phase 1 model recalibrated on VALIDATION (Astra F-02). Comparison only.
  - SAFE_DISCOVERY (**primary**): only SAFE features, new and baseline. `SAFE_BASE` is the SAFE subset of BASELINE_15.
  - EXPLORATORY_UNRESOLVED_SENSITIVITY: a separately labelled LASSO sensitivity that also offers UNRESOLVED sources (never UNSAFE ones).
    It is never a selection candidate.
- A run STOPS (INELIGIBLE_FEATURE_IN_SET) before any fit if a SAFE_DISCOVERY set holds a non-SAFE feature or a non-SAFE baseline
  predictor, or if any set holds an UNSAFE / ineligible feature.

## 3. Feature engineering (row-wise, parameter-free; catalogue `phase2_features.yaml`)

- **Operations:** copy, threshold flags, any-positive composites, days since a record date, and the "form assessed" indicator (record date
  present).
- **Not measured ≠ negative:** items of assessment-conditional nurse forms and MEFI keep NULL as "not assessed". Each form contributes
  ONE "assessed" indicator, which a feature set includes automatically whenever it includes an item of that form. An item gets its own
  "unanswered" indicator whenever its NULL pattern differs from the form's on ANY TRAIN row (label-blind rule; Astra F-04). Identical
  patterns share one indicator.
- **Quarantine** (NEEDS_SME, not used): Siudi_Status, Customer_Risk_Code, Malnutrition_Grade, Fall_Self_Report_Value,
  On_Medications_Nurse_Ind, Orthostatic_BP_Measured_Ind, Slow_Rising_Ind and Vitamin_D_Ind. Their codes or the meaning of "1" are
  undocumented.
- **Linear design (fitted inside every fold):**
  - BASELINE_15 block: the unchanged eFalls preprocessor (`efalls_fp_all_levels`, FP selection for age and polypharmacy).
  - New counts, days and amounts: `log1p`.
  - Ordinal scores: thermometer dummies (`≥ level`).
  - Days since last fall: fixed bands (≤ 90, 91–180, 181–365, > 365 days; reference = no prior fall).
  - NULL in an observed column: fold-median fill plus the covering indicator.
- **Tree design:** raw engineered values with NULL kept (XGBoost native missing handling), plus the same form indicators.

## 4. TRAIN-only screening (exhaustive, cheap)

- **Per candidate:** N observed, missing %, distribution, outcome rate by level/quintile, prevalence ratio, SMD, univariate AUROC
  (with 95% CI), Spearman with every other candidate and with the BASELINE_15 sources, sparse (< 20 events in a level) and separation
  flags. Small cells (< 10) are suppressed in shared output.
- **Redundancy:** clusters at |Spearman| ≥ 0.90 on pairwise-complete TRAIN rows (label-blind).
  - Clusters are computed SEPARATELY in the SAFE universe (SAFE features + SAFE baseline) and in the exploratory universe, so a SAFE feature is never pruned in favour of an UNRESOLVED one.
  - The representative is a baseline feature if present, else the lowest missingness, then catalogue priority, then name.
  - Non-representatives leave the pooled sets, but stay in their one-domain sets and are reported.
- **No feature is removed for weak univariate association.** Removal only for data quality (INELIGIBLE_DATA) or redundancy.

## 5. Feature sets (frozen in `FEATURE_SETS.json` before any fit)

| Set | Category | Definition |
|---|---|---|
| BASELINE_15 | HISTORICAL | the 15 EXTENDED predictors (unchanged definitions, provenance as found) |
| SAFE_BASE | SAFE | the BASELINE_15 predictors whose D-00 status is SAFE (at most age, sex, prior falls, mobility problems) |
| ALL_REVIEWED_SAFE | SAFE | SAFE_BASE + every SAFE non-redundant feature (= the last waterfall step = ALL_SAFE_LINEAR) |
| EFALLS_MAPPABLE_SAFE / MEUHEDET_NATIVE_SAFE | SAFE | SAFE_BASE + SAFE eFalls-mapped / Meuhedet-native features |
| SAFE_BASE_PLUS_<DOMAIN> | SAFE | SAFE_BASE + every SAFE feature of one domain (skipped if the domain has none) |
| SAFE_WATERFALL_k_<DOMAIN> | SAFE | cumulative, pre-declared: SAFE_BASE → +COGNITION → +MEDICATION → +FRAILTY → +FUNCTION_MOBILITY → +UTILISATION → +HOME_ENVIRONMENT → +all other domains (= ALL_REVIEWED_SAFE) |
| ALL_REVIEWED_SAFE_WITHOUT_EXPLICIT_ASSESSMENT_FLAGS | SAFE | sensitivity: no explicit form flags; other NULL indicators stay (Astra F-10) |
| ALL_REVIEWED_SAFE_MINUS_<DOMAIN> | SAFE | leave-one-domain-out (ablation stage) |
| EXPLORATORY_B15_PLUS_SAFE | EXPLORATORY | BASELINE_15 (incl. its UNRESOLVED predictors) + SAFE new features |
| EXPLORATORY_ALL_REVIEWED | EXPLORATORY | BASELINE_15 + SAFE + UNRESOLVED new features (never UNSAFE) |

The management story (every step on the same VALIDATION rows, matched capacities):

BASELINE_15 refit → SAFE_BASE → the SAFE waterfall → ALL_SAFE_LINEAR → ALL_SAFE_ELASTIC_NET → ALL_SAFE_XGBOOST.

Domains, in catalogue order: COGNITION, MEDICATION, FRAILTY, FUNCTION_MOBILITY, PRIOR_FALLS_DETAIL, NURSE_FALL_RISK, UTILISATION,
HOME_ENVIRONMENT, SOCIAL_SUPPORT, ADDITIONAL_COMORBIDITY. Identical feature lists share one fit.

## 6. Models, tuning and selection (nested grouped CV inside TRAIN)

- **Outer CV:** 5 folds, patient-grouped, outcome-stratified, seed 20260929 (one set of folds for every family and set).
- **LASSO (all sets):** the existing solver (100 λ, λ_min by inner 10-fold grouped CV deviance). Predictions use the **penalised**
  coefficients at λ*, as in Phase 1 (the unpenalised refit is only an odds-ratio table there; REVIEW_DECISIONS A-30).
- **Elastic net:** same solver with an L2 term (glmnet parameterisation). l1_ratio ∈ {0.1, 0.5, 0.9} and λ are chosen jointly by inner
  CV deviance. Applied to ALL_REVIEWED_SAFE and to the best LASSO set if it differs (≤ 2 sets).
- **XGBoost** (`binary:logistic`, `hist`, eta 0.05, subsample 0.8, colsample 0.8, gamma = alpha = 0, ≤ 3,000 trees, early stopping 100 on
  inner-fold log loss, no class weighting, no monotone constraints). The final fit uses the median inner best iteration.
  - Stage 0: defaults (depth 3, min_child_weight 10, lambda 5) on BASELINE_15 and ALL_REVIEWED_SAFE.
  - Stage 1: 15 Optuna TPE trials (6 random start-up trials, multivariate) on ALL_REVIEWED_SAFE; space depth {1, 2, 3},
    min_child_weight 5–30 (log), lambda 1–30 (log); objective = inner 5-fold CV log loss.
  - Stage 2: **disabled** in the final configuration (Astra F-06). Stage 1 answers the question. Deeper tuning is a future experiment,
    run only if stage 1 shows value.
  - Early stopping (Astra F-01):
    - `xgb.train(num_boost_round = 3000, early_stopping_rounds = 100)`;
    - the inner held-out fold is scored with `iteration_range = (0, best_iteration + 1)`;
    - the final scope fit uses the median tree count without early stopping.
  - Inner folds are common to every candidate of a tuning scope (Astra F-07).
  - Hard cap: 25 tuned trials per study.
- **Nesting:** each outer fold runs its own tuning (λ, l1_ratio, FP forms, fills, XGB study and tree count). A final fit on all TRAIN
  repeats the same procedure. Estimates are outer out-of-fold (OOF).
- **Advancement rule (S09, TRAIN outer-OOF only; hierarchy per A-07; FINAL version).** Candidates are SAFE_DISCOVERY configurations only. A
  candidate qualifies if all of the following hold vs **LASSO:SAFE_BASE** (an undefined value never passes):
  - ΔAP ≥ 0.010;
  - Δcapture at the **principal capacity (top 10%)** ≥ +1.0 percentage point (≈ +13 recorded falls per 180 days in a population of
    60,851);
  - log loss ≤ 1.01 × baseline;
  - OOF calibration slope in 0.80–1.25;
  - |CITL| ≤ 0.20.

  - **Choice:** among qualifiers the highest AP wins, unless a simpler qualifier is within 0.005 AP (linear before XGBoost, then fewer non-zero features).
  - **No qualifier:** the outcome is **NO_INCREMENTAL_MODEL_SELECTED** (SAFE_BASE is reported, never self-compared).
  - **Benchmark class:** the recommended model is classified against the BASELINE_15 refit (same margins; non-inferiority −0.010 AP / −1 pp) as MEETS_SUPERIORITY_RULE / WITHIN_NONINFERIORITY_MARGINS / OUTSIDE_MARGINS. This is a point-estimate classification, not a formal test.
  - **Shortlist:** SAFE_BASE, the BASELINE_15 refit, the recommended model, then the best SAFE candidate of each other family (≤ 3 challengers).
  - **Attribution contrasts (frozen):** FEATURE_EXPANSION (LASSO on ALL_REVIEWED_SAFE vs SAFE_BASE) and MODEL_FAMILY_PIPELINE (tuned XGBoost vs the best linear model on the same set).
  - `SELECTION_FROZEN.json` is written and hash-chained **before** VALIDATION can be read.
- **Validation (S13, opened once).** `VALIDATION_OPENED.json` is write-once and lists the frozen finalists and every descriptive
  configuration; re-reading after an interruption is allowed only for the same selection.
  - Exploratory screen (pre-declared): the recommended SAFE model vs SAFE_BASE. It is CONFIRMED_PROMISING if all of these hold:
    - validation ΔAP ≥ 0.005;
    - Δcapture@10% ≥ 0;
    - log loss ≤ 1.01 × SAFE_BASE;
    - calibration slope 0.70–1.40;
    - |CITL| ≤ 0.30.

    Otherwise it is NOT_CONFIRMED. The wording is always "promising, not established". The same rules give the point-estimate class
    against the BASELINE_15 refit.
  - Descriptive: every pre-declared configuration (waterfall, domain additions, families, sensitivity sets), labelled "descriptive – not
    used for selection".
  - Paired patient bootstrap, 2,000 replicates.

## 7. Stability, ablation, explainability, consensus

- **Stability:** 25 patient-bootstrap replicates of the full LASSO procedure on ≤ 3 finalist sets:
  - the SAFE linear shortlist sets;
  - ALL_REVIEWED_SAFE;
  - the BASELINE_15 refit, as the reference.

  Copies of a resampled patient stay in one inner fold. Convergence retry policy v1 applies; more than 10% failed replicates is a STOP.
  Reported: selection count, frequency with Monte Carlo SE, and sign stability (coarse; Astra F-13).
- **Ablation (TRAIN OOF, paired):**
  - Leave-one-domain-out from ALL_REVIEWED_SAFE, fully nested.
  - Individual ablation of the top ≤ 15 new features (ranked by S06/S08 importance, TRAIN-only) in the recommended family and set; each
    outer fold keeps its tuned hyper-parameters fixed (explanatory).
  - Reported: ΔAP, ΔAUROC, Δlog loss, ΔBrier, Δcapture@10%, Δfalse alerts at matched 10% capacity.
- **Explainability (promoted XGBoost only):**
  - Permutation importance on outer-OOF (fold models, 5 repeats, grouped by feature);
  - TreeSHAP (`pred_contribs`) on a ≤ 12,000-row TRAIN sample: mean |SHAP| ranking and aggregated quantile summaries (no individual
    points);
  - interaction values only for the top 5 pairs.
  - No causal interpretation.
- **Consensus (SAFE features only; "exploratory predictive evidence", not causal).** Evidence classes per SAFE feature (final version,
  Astra F-13 / F-14):
  - U, univariate: |SMD| ≥ 0.10 or univariate AUROC outside 0.45–0.55;
  - L, linear: selected by the final LASSO on ALL_REVIEWED_SAFE, with LASSO stability ≥ 0.60 where run (elastic-net-only selection is weak
    support: the stability refits are LASSO only);
  - I, importance: SHAP rank ≤ 15 of the promoted XGBoost (a positive permutation AP loss alone is weak support);
  - A_domain: the feature's one-domain addition (SAFE_BASE + domain) has OOF ΔAP ≥ 0.005 vs SAFE_BASE;
  - A_feature: individual ablation OOF ΔAP ≥ 0.002 (feature-level incremental evidence).

  Ratings (A_domain alone is never enough for STRONG: a domain's gain is not evidence for each of its members):
  - STRONG = A_feature plus (L or I);
  - MODERATE = (L or I) plus (U or A_domain), or A_feature alone;
  - WEAK = any single support (U, L, I, weak permutation, EN-only, A_domain);
  - NONE otherwise.

  UNRESOLVED features are reported separately in EXPLORATORY_UNRESOLVED_FEATURES.csv (univariate + exploratory LASSO only) and never rated
  for a SAFE model.

  Flags: redundancy cluster, assessment-conditional, UNRESOLVED timing, NEEDS_SME interpretation.

## 8. Metrics (fixed implementations)

- **Discrimination:** AP (step-wise average precision, sklearn-equivalent, ties grouped) = "PR-AUC" in tables; AUROC.
- **Probability quality:** log loss, Brier, scaled Brier (vs constant prevalence), CITL (slope fixed at 1), calibration intercept and
  slope (joint), O:E, grouped calibration (10 groups, with event counts).
- **Capacity (1/3/5/10/20%):** N flagged, % population, TP, FP, FN, TN, sensitivity, specificity, PPV, NPV, lift, false alerts (= FP),
  false-alert %. "Capture" = recorded events identified, not falls prevented.
- **Probability thresholds:** 0.01–0.09 by 0.01, then 0.10–0.50 by 0.025.
- **Subgroups** (descriptive; a metric only when ≥ 10 events): sex, age band (65–74, 75–84, 85+), any nurse form assessed vs none,
  MEFI assessed vs not, prior fall vs none.
- Small cells (< 10) are suppressed in `share/`.

## 9. Stop gates

- **Hard stops:**
  - UNSAFE / forbidden / label-derived / post-index input in a fit;
  - UNRESOLVED input outside the exploratory track;
  - patient overlap between partitions;
  - reference split or cohort mismatch;
  - input sha256 change;
  - protected folder changed;
  - `--out` inside a synced folder (unless allowed);
  - > 10% fitting failures;
  - privacy-scan hit in `share/`;
  - completed-artifact hash mismatch.
- **Investigation stops** (pause + `INVESTIGATION_<gate>.md`; continue only with `--accept-gate <GATE> --reason "…"`, recorded):
  - SINGLE_FEATURE_DOMINANCE: a new feature with TRAIN univariate AUROC ≥ 0.80;
  - IMPLAUSIBLE_GAIN: OOF or validation ΔAUROC > 0.05, or AP ratio > 1.5 vs BASELINE_15;
  - IMPLAUSIBLE_CALIBRATION: slope outside 0.6–1.5 or |CITL| > 0.5;
  - PROXY_DOMINANCE: one new feature > 50% of total mean |SHAP| or > 50% of the permutation AP loss.

## 10. Compute budget and checkpointing

- **Plan:** ≈ 25 nested LASSO/EN configurations × 6 fits, XGBoost stage 0 + 15-trial studies (+ ≤ 8 only if stage 1 earns it), 75 stability
  fits, ≈ 50 leave-one-domain-out and ≈ 75 individual-ablation fits.
- **Measured** (full production settings; synthetic 66k-row extract → 28,862 TRAIN rows, ≈ 80% of the real 36.5k; Apple M4; CPU shared
  with up to three other jobs):

  | Stage | Items | Time |
  |---|---|---|
  | S00–S05 (preflight, cohort, registry, engineering, screening, sets) | 6 | < 1 min |
  | S06 LASSO (nested, every set) | 126 | 31 min |
  | S07 elastic net (2 sets × 3 l1_ratios, nested) | 13 | 39 min |
  | S08 XGBoost (stage 0 + 6 × 15 Optuna trials; stage 2 skipped by its rule) | 109 | 5 min |
  | S09 selection (paired bootstrap, 2,000 replicates, every configuration) | 1 | 14 min |
  | S10 stability (3 sets × 25 replicates) | 76 | 49 min |
  | S11 ablation (leave-one-domain-out + 15 individual) | 126 | 42 min |
  | S12–S16 (explain, validation, consensus, report, share) | 10 | 3 min |
  | **Total** | | **≈ 3.2 h** |

- **Expected on the work PC:** × 1.26 for the real row count and × 1.5–3 for a slower CPU → **roughly 5–10 hours** (up to ≈ 15 h on an older
  laptop). Plan it overnight; interruptions cost only the item in progress (resume).
- Every item checkpoints immediately (see `planning/ARCHITECTURE_PHASE2.md` §11 and `planning/REHEARSAL_REPORT.md`). Expansion beyond
  this plan must be logged with a reason.

## 11. Reporting

- **`share/` only:** the user-specified file list and figures.
- **MANAGEMENT_SUMMARY_HE.md:** Hebrew, no hype words, no causal language, the waterfall question "what did each domain buy us?" with
  capture at 5% and 10%, PPV and false alerts at matched capacity.
- **SCIENTIFIC_SUMMARY.md:** baseline, discovery funnel, new signals, family comparison, improvement vs BASELINE_15, eFalls vs
  Meuhedet-native winners, robustness (distributed vs single-feature gains), recommendation, limitations, and the freeze list for the
  temporal holdout (Astra A-35).
- Never "significant", never causal wording; every validation number is labelled descriptive or confirmatory as defined above.
