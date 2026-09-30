# Briefing for a senior clinical-prediction methodologist: Phase 2 of an elderly-falls prediction model (Meuhedet, Israel)

You are asked for a critical methodological review of a proposed analysis plan. You have no data and no code. Please answer the
8 questions at the end concisely (numbered answers, concrete recommendations, explicit "change X to Y" where you disagree). Please do
not repeat the brief back. Aim for at most ~1,800 words.

## 1. Scientific question

Which additional predictors available in the HMO's research data warehouse add genuine, incremental predictive value for a fall within
180 days in adults aged ≥ 65, beyond an existing 15-predictor model? Does a different model family (elastic net, XGBoost) outperform
the LASSO logistic model meaningfully? The output is a recommendation of ONE feature set + model to advance to a new temporal holdout. This
phase is not the final performance claim.

## 2. Cohort, outcome, current baseline

- One wide snapshot table: one row per member at Index_Date 2025-01-01, 221 columns (aggregates of source systems up to the index date).
- Modelling cohort: 60,851 members (age ≥ 65), 1,282 events (2.11%), outcome = any recorded fall/fracture event in the index day + 180
  days (exploratory label; deaths are non-events as labelled by the warehouse; censored rows excluded).
- Existing split (made once, before this phase): patient-grouped random, outcome-stratified, 60/20/20 → TRAIN ≈ 36.5k rows / ≈ 769
  events, VALIDATION ≈ 12.2k / ≈ 256 events, TEST ≈ 12.2k / ≈ 256 events.
- **The TEST partition has been looked at repeatedly in earlier work. It will not be loaded in Phase 2 at all.** A genuinely new claim will
  require a new holdout, preferably temporal (a later snapshot with a complete 180-day label).
- Baseline "EXTENDED-15": age (fractional polynomial), sex, polypharmacy count (FP), prior fall/fracture since 2022 (binary), gait
  disorder diagnosis, and 10 disease-registry flags (dementia, COPD, asthma, diabetes, liver, CKD, hypertension, housebound, severe
  mental illness, suicide attempt). LASSO logistic, 10-fold grouped CV, λ_min, then unpenalised refit on the selected columns (the eFalls
  method); logistic recalibration (intercept + slope) fitted on VALIDATION.
- Baseline TEST performance: AUROC ≈ 0.814, PR-AUC ≈ 0.197, Brier ≈ 0.018. Top 10% of predicted risk captures ≈ 55% of falls.
  Ablations: prior falls carry much of the gain over demographics; removing gait disorder alone changes AUROC very little.

## 3. Prediction-time rule and leakage facts (non-negotiable)

- Prediction happens at the start of the index day: a predictor may use only source records dated strictly before Index_Date.
- The warehouse view was found to aggregate some sources with Event_Date ≤ Index_Date (4,168 rows had a diagnosis dated on the index day,
  22 a fall). Those rows were removed to form the current cohort. The source cannot be fixed now.
- Each source has one of: a complete record-date column (last fall, last diagnosis, last visit, each nurse form's assessment date, MEFI
  frailty assessment date); a first-entry date only (registries: proves an index-day entry but cannot exclude one); or no record date
  (medications, labs, Charlson group, benefit statuses). A feature is SAFE if its record date is proven < Index_Date on every row,
  UNSAFE if proven ≥ Index_Date on any row, UNRESOLVED otherwise. **The current baseline already contains UNRESOLVED features** (registry
  flags; polypharmacy).
- Twelve columns are forbidden: billing-lag external-care counts, hospitalisations without an as-of date, "new registry" counters, an
  index-day-fall flag, fairness-audit fields.

## 4. Candidate information (≈ 90 raw value columns + ≈ 60–90 engineered features)

- **Nurse fall-risk assessment forms** (≈ 23 items: dizziness, fear of falling, balance/shuffling/hesitant gait, walking aid, muscle
  weakness, orthostatic hypotension, vision, sensory aids, "no primary caregiver", nurse-reported prior falls, overall risk score). These are
  **assessment-conditional**: NULL = not assessed (probably the majority), 0 = assessed negative. The decision to assess is clinical
  (selection by suspicion).
- **Function/mobility (nurse)**: ADL worst-domain score, transfer dependency, mobility score, vision and hearing scores; Get-Up-And-Go
  date only (no score).
- **Cognition**: Mini-Cog score and date, nurse "cognitive impairment" item (MMSE empty).
- **Home safety questionnaire**: unsafe environment, footwear, bed/chair height, risk count, fall at home, vitamin D, slow rising.
- **MEFI frailty** (HMO frailty index): group 1–4, assessed flag, days since assessment, previous group, worsened flag, not-assessed flag.
- **Prior falls, finer**: counts in 30/90/180/365 days and since 2022; days since last fall.
- **Utilisation**: visit counts 30/90/180/365 d, days since last visit; diagnosis counts; distinct/chronic diagnoses; lab referrals/performed;
  prescriptions written/purchased/fill ratio; external care already billed at index; HMO seniority.
- **Medication**: polypharmacy ≥ 5 / ≥ 10, narcotic drug count, prescribed-not-collected, bought outside the HMO, total DDD. No drug
  classes (fall-risk-increasing drug groups cannot be built), no dispensing dates.
- **Additional comorbidity/social**: heart-disease, kidney, transplant and stoma registries, registry counts and durations, Charlson group,
  malnutrition grade, income support; Siudi (nursing-care benefit) status and a "customer risk code" have undocumented codes.

## 5. Proposed design (my draft; please critique)

**Partitions.** TRAIN for everything data-driven: screening, engineering parameters, all model fitting and all hyper-parameter tuning
(grouped K-fold CV inside TRAIN). VALIDATION only for comparing a **pre-declared, logged list** of promoted configurations (≈ 30–40
model × feature-set combinations). TEST never loaded. I would also report TRAIN out-of-fold (grouped 10-fold) estimates as a second,
lower-variance view to damp the winner's curse on a 256-event validation set.

**Feature funnel.**
1. Registry of all 221 columns with an explicit disposition (SAFE/UNSAFE/UNRESOLVED; eFalls-mappable vs native; metadata; forbidden).
2. Engineering from eligible columns: recency, window counts, latest score, abnormal flags (e.g. Mini-Cog < 3), assessed indicators.
   "Not measured ≠ negative": binaries become three states (positive / negative / not assessed); one "form assessed" indicator per form.
3. TRAIN-only screening: N observed, missingness, distribution, outcome rate, prevalence ratio, SMD, univariate AUROC, correlation,
   sparse/separation flags. **No feature is removed for weak univariate association**; removal only for quality (constant, near-empty) or
   redundancy (|Spearman| ≥ 0.9 → keep one representative by a pre-declared rule).
4. Feature sets: BASELINE_15; EFALLS_MAPPABLE_ALL; MEUHEDET_NATIVE_SAFE; ALL_REVIEWED_SAFE; BASELINE_15 + one domain at a time
   (cognition, frailty, medication, function/mobility, utilisation, home/environment, social/support, additional comorbidity); and a
   cumulative waterfall in a **pre-declared order** (BASELINE_15 → +cognition → +medication → +frailty → +function → +other native →
   ALL_SAFE → best linear → best XGBoost).
5. Models: LASSO (existing pipeline) per set; elastic net with l1_ratio ∈ {0.1, 0.5, 0.9} on ≤ 2 promoted sets (λ by CV in TRAIN,
   l1_ratio chosen by TRAIN CV deviance); XGBoost: stage 0 defaults + early stopping; stage 1 = 12–15 Optuna TPE trials (max_depth,
   min_child_weight, learning_rate, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda; estimator ceiling 5,000 with early
   stopping), objective = mean grouped 5-fold CV PR-AUC inside TRAIN; stage 2 ≤ 8 local trials only if stage 1 beats the best linear
   model's CV PR-AUC by a pre-declared margin. No class weighting (to keep probabilities interpretable). Then refit on all TRAIN with the
   CV-averaged number of trees and score VALIDATION once.
6. Finalists: 25-replicate bootstrap selection stability (LASSO); individual ablation only for the top 10–15 new features; XGBoost
   permutation importance and TreeSHAP only for the promoted model; a consensus table (univariate, LASSO selection/stability, elastic
   net, SHAP rank, permutation, incremental validation gain, ablation effect).

**Metrics on VALIDATION (no recalibration on validation; raw TRAIN-fitted probabilities).** Primary: PR-AUC (average precision). Secondary:
AUROC. Probability quality: Brier, calibration intercept and slope, calibration plot. Operational: at capacity 1/3/5/10/20% of the
population → N flagged, TP, FP, FN, TN, sensitivity, PPV, NPV, lift, false alerts. Management framing: "if we can intervene on X% of the
population, how many falls do we capture?". Paired patient-cluster bootstrap (2,000 reps) for every Δ vs BASELINE_15.

**Stop gates.** Leakage; unresolved timing for a feature about to be used; split overlap; input hash change; implausible gains (e.g. a
single new feature with TRAIN univariate AUROC ≥ 0.80, or validation AUROC gain > 0.05 over BASELINE_15, or PR-AUC ratio > 1.5); implausible
calibration (validation slope outside 0.6–1.5 or |intercept| > 0.5); a proxy feature dominating importance; > 10% model-fitting failures.

**Compute.** CPU only; LASSO CV fit ≈ 5–40 s per set here (≈ 2 min worst case on the target PC); the whole plan ≈ one night.

## 6. Open methodological decisions (my current leaning in brackets)

a. Validation partition only (≈ 256 events) vs adding TRAIN out-of-fold estimates [report both; decide on validation, require OOF
   agreement in direction].
b. PR-AUC as primary [yes, with paired CIs], or a proper scoring rule (log loss) as primary with PR-AUC secondary?
c. Nurse, MEFI, assessment-conditional data: use three-state encoding and "assessed" indicators [yes], accepting that the model partly
   learns the care process? Should "assessed" indicators be excluded from a "clinical content only" sensitivity set?
d. Features that are UNRESOLVED (no record date; the same evidence level as baseline registries/polypharmacy) [admit under a clearly
   labelled SAFE_OR_UNRESOLVED standard, plus a SAFE-only sensitivity set].
e. A new source proven UNSAFE on some rows (e.g. a nurse assessment dated on the index day): exclude the feature everywhere, or
   pre-declare ONE Phase 2 cohort that removes the proven rows for every source used (and refit BASELINE_15 on it)? [exclude the feature
   if > 1% of rows are affected; otherwise the single fixed cohort].
f. LASSO: keep the baseline's post-selection unpenalised refit for the expanded sets, or use penalised λ_min predictions? [keep the
   baseline method for BASELINE_15 reproducibility, report penalised as a sensitivity].
g. Decision rule for "genuine incremental value" of a domain/feature with ≈ 256 validation events [ΔPR-AUC 95% CI lower bound > 0, or
   ΔPR-AUC ≥ +0.01 with bootstrap P(Δ > 0) ≥ 0.90 and a TRAIN-OOF Δ in the same direction?].
h. Ordering sensitivity of the cumulative waterfall [report the one-domain-at-a-time and leave-one-domain-out tables next to it].

## 7. Questions (please answer each)

1. Is the discovery/validation design statistically sound? What is the single most important change?
2. Is PR-AUC a reasonable primary discovery metric at ≈ 2.1% prevalence and ≈ 256 validation events? If not, what should be primary?
3. How should AUROC, calibration (intercept/slope), Brier and capacity-based capture be used alongside it (decision hierarchy)?
4. What minimal XGBoost tuning is appropriate here (search space ranges, trials, early-stopping design, class weighting, monotone
   constraints?), and how to compare it fairly with the LASSO/elastic-net family?
5. What are the biggest leakage and selection-bias risks in this specific setting (nurse assessments, utilisation, fall/fracture counts,
   care-process signals), and what concrete guard would you add?
6. How should incremental feature value be tested (domain increments, individual ablation, paired bootstrap, decision rule for decisions
   (a)–(h))?
7. What would be required before claiming temporal or generalisable performance (holdout design, prevalence shift, recalibration,
   what must be frozen)?
8. Which parts of the proposed plan should be changed, dropped or added? Please rank the top 5 changes by impact on validity.
