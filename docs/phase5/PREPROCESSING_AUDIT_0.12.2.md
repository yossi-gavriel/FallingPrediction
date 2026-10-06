# Phase 5 preprocessing audit — falls_ml 0.12.2 / Phase 5 2.2.0

**Verdict: MATERIAL_ISSUE_RETRAINING_SHOULD_BE_CONSIDERED**

This is a source audit of baseline commit `d1e611c8d0cf750ec8e439775f4ff902d6387da5`, not an inspection of the completed Windows models. The linear design fits its medians, encoded columns, missing indicators, category discovery, duplicate/constant removal, means and standard deviations on each inner training fold and refits them on each outer training fold. The old concern about median imputation before inner CV does **not** describe this Phase 5 path. However, feature eligibility includes an outcome-dependent single-feature AUROC exclusion on the entire usable cohort before the outer folds exist. This breaches strict isolation of outer holdout outcomes from predictor eligibility. Its material effect on the completed run cannot be established without that run's provenance and exclusion record. CCI group also remains a raw numeric continuous feature despite unvalidated grouping. These findings warrant a methodological review before deciding whether to repeat development; this audit does not order or perform retraining.

No raw clinical data or existing run-output directories were read. Only tracked source/configuration/history and newly generated synthetic fixtures were used. The 130-row companion [feature audit](PREPROCESSING_FEATURE_AUDIT_0.12.2.csv) describes the complete source candidate universe and its per-feature preprocessing. It does not assert that all 130 candidates entered a real model.

## Version identity and chronology

| Commit | Commit date (+03:00) | Source release |
|---|---|---|
| `315fba3` | 2026-10-04 17:46:48 | falls_ml 0.11.0: initial Phase 5 redevelopment |
| `740bf81` | 2026-10-04 21:26:40 | falls_ml 0.12.0 / Phase 5 2.0.0: authoritative V21 schema, exact schema diff, ENET primary |
| `631e95d` | 2026-10-05 07:35:18 | falls_ml 0.12.1 / Phase 5 2.1.0: explicit registry lineage and zero-tolerance follow-up stop |
| `d1e611c` | 2026-10-06 09:08:40 | falls_ml 0.12.2 / Phase 5 2.2.0: capacity dashboard and 3% report; no refit |

`src/falls_ml/__init__.py:12`, `pyproject.toml:7`, and `src/falls_ml/phase5/__init__.py:17` identify the audited release. `git diff 631e95d d1e611c -- src/falls_ml/phase5/design.py src/falls_ml/phase5/data.py src/falls_ml/phase5/models.py src/falls_ml/phase5/engine.py src/falls_ml/phase5/runner.py src/falls_ml/phase5/config.py configs/meuhedet/phase5.yaml configs/meuhedet/phase5_v21_schema.yaml` is empty. The preprocessing/training graph audited here is therefore unchanged between the **source** releases 0.12.1 and 0.12.2. Version text alone does not prove which source/configuration produced a Windows pickle, especially after resume, code changes, or post-processing.

## The actual Phase 5 graph

1. `phase5.runner.run_phase5` loads Phase 5 configuration and `phase3.runner.load_all` definitions. The inherited components supply the fixed feature catalogue, raw-contract parsing, fixed baseline mapping, rowwise engineering, and timing recovery. They do not invoke Phase 2's fitted preprocessing pipeline. Definition loading is at `src/falls_ml/phase3/runner.py:28-46`.
2. `phase5.data.prepare` compares the extract header with the authoritative V21 schema and V1 contract, seals outcome/future fields, resolves which OLD inputs are reproducible, reads predictors and labels separately, checks the cohort/outcome contract, computes engineered values, applies timing/type masks, and fixes eligibility/set-membership. See `src/falls_ml/phase5/data.py:171-439`.
3. `phase5.runner.run_preflight` calls `prepare`, builds sets, applies `x_guard`, and only **then** creates the stratified outer folds. See `src/falls_ml/phase5/runner.py:275-324` (prepare 285; sets 289; outer folds 307).
4. `phase5.engine.run_unit` selects outer training/test rows (`outer < 0` means final development on all usable rows) and calls `tune_and_fit`. It passes holdout predictors, not holdout outcomes, to the tuning/fitting function. See `src/falls_ml/phase5/engine.py:131-174`.
5. For LASSO/ENET, `_tune_linear` fits an outer-training design to construct a shared lambda grid, then `linear_path_task` calls `linear_path` for each inner fold. `linear_path` creates a **new `LinearDesign(...).fit(Xtr)`** before transforming inner training and validation. The selected configuration is refit through `fit_linear`, which creates another design on outer training; prediction uses that stored design's `transform`. See `engine.py:200-241`, `models.py:36-52,65-91`.
6. For XGB, `_tune_xgb` uses `TreeDesign`; its current `fit` creates only a fixed column layout from metadata, with no data-derived statistics. Inner training rows are split again for early stopping; inner validation rows are prediction-only. Final outer-training XGB uses the median inner best-iteration count and no outer holdout early stopping. See `engine.py:256-281,321-325`, `models.py:141-169`.

The reuse of `phase2.enet.enet_logistic_path` is reuse of the penalized logistic **solver**, not evidence that Phase 2's preprocessing executes. Likewise, `phase2.engineer.compute` supplies parameter-free rowwise operations, not fitted medians or polynomial-form selection (`src/falls_ml/phase2/engineer.py:22-37`).

## Linear encoding, imputation and scaling

For every feature in `LinearDesign.features`, source values are converted to numeric arrays; finite values are observed, and NaN/Inf are treated as nonfinite. `_encode` selects behavior using the feature's `linear` and `kind` metadata (`src/falls_ml/phase5/design.py:29-51`).

* Binary: `1` exactly when an observed value equals `1`, otherwise `0`; missing gets `0` in the value column. This is a fixed encoding, not fitted mode/median imputation.
* Ordinal: one cumulative column `I(x >= level)` for each declared level above the lowest. Missing gives zeros. The lowest level is the all-zero reference.
* Fall recency: four columns for `<=90`, `91–180`, `181–365`, `>365` days; no recorded prior fall gives all zeros. The ranges and names are fixed, not outcome-selected.
* Nominal: reference coding, explained below.
* Continuous/count/days: either `v=x` or `v=log(1+max(x,0))`. The code does not winsorize or estimate cutpoints. For transformed numeric columns, a missing value is replaced with the **median of finite transformed training values**. For `log1p`, the median is computed after transformation, not by logging an interpolated raw median. All-missing training uses fallback zero, after which the constant value column is removed.

The feature-level `f__na` indicator is added if **any nonfinite raw feature value occurs in training**, and it encodes nonfiniteness during later transforms. This applies to binary, ordinal, nominal, bands and continuous features. An indicator is not added merely because a validation/test patient is missing. If training has no missingness and validation later does, the numeric feature gets its training median (or the encoding's zeros), with no new indicator. Constant/duplicate missing indicators can be removed. Sources: `design.py:67-95`.

On the **training encoded/imputed matrix**, constant columns are dropped (`max == min`), and exact duplicates are dropped in feature/column order using identical array bytes. This includes disease flags, thermometer columns and missing indicators, not just continuous values. There is no near-zero variance cutoff and no correlation threshold. The retained columns are then centered and scaled:

\[
\mu_j = \frac1n\sum_i v_{ij},\quad
s_j = \sqrt{\frac1n\sum_i(v_{ij}-\mu_j)^2},\quad
z_{ij} = \frac{v_{ij}-\mu_j}{s_j}.
\]

This is NumPy population standard deviation (`ddof=0`), calculated **after encoding and imputation**, over training rows only. The fallback is `s=1` when `s<=0`; ordinary constant columns have already been removed. Prediction uses the stored mean, SD and keep mask, without recomputing any statistic (`design.py:95-117`). Validation extremes do not alter medians or scales.

**Binary and one-hot columns are scaled too.** A binary with training prevalence `p` has mean `p` and SD `sqrt(p(1-p))`; its encoded values are `-p/s` and `(1-p)/s`. Missing indicators and thermometer/band dummies receive the same treatment. There is no binary passthrough branch. Therefore the L1/L2 penalty operates on coefficients per training SD, not per raw unit: if the raw coefficient is `gamma_j`, the standardized coefficient is `s_j*gamma_j`. Its L1 contribution is `lambda*alpha*s_j*|gamma_j|` and L2 is `lambda*(1-alpha)*s_j^2*gamma_j^2/2`. This changes relative penalization compared with leaving rare binaries as raw 0/1, but is the declared implementation, not leakage. The intercept is unpenalized. Sources: `design.py:108-117`; `src/falls_ml/phase2/enet.py:1-6,30-31,77-82`; `phase5.models.py:30-33`.

### Inner versus outer training boundaries

| Learned quantity | Inner-CV fit | Outer fold refit/prediction | Final development |
|---|---|---|---|
| transformed medians | Each inner-training fold | Refitted on outer training; stored values used on holdout | Fitted on all usable development rows |
| raw nominal levels/count threshold | Each inner-training fold | Refitted on outer training; stored levels used on holdout | Fitted on all usable development rows |
| presence of NA indicators | Each inner-training fold | Refitted on outer training; holdout never adds columns | Fitted on all usable development rows |
| constants/exact duplicates | Each inner-training encoded matrix | Refitted on outer training; stored keep mask | Refitted on all usable development rows |
| means/population SD | Each inner-training retained matrix | Refitted on outer training; stored statistics | Refitted on all usable development rows |
| lambda grid | Shared grid constructed on entire outer training, including inner validation labels | No outer holdout labels used | Constructed on all usable development rows |
| candidate eligibility/set membership | Fixed in cohort preflight | Not recomputed per outer fold | Same preflight set |

The shared grid is `lambda_max = max_j |A_j^T(y-mean(y))|/(n*alpha)` followed by a logarithmic grid down to `0.001*lambda_max` (`models.py:23-27`, `engine.py:208-209`, configuration LASSO/ENET). This uses inner validation outcomes to define the candidate grid, but keeps the outer holdout isolated **at this step**. It is not a median/scaler leakage path; inner OOF is a tuning object rather than independent performance evidence. A strict train-only audit must record this boundary instead of claiming every hyperparameter is learned solely within inner training.

## Nominal codes, MEFI and CCI

The actual nominal candidates are `new_registry_smoking_subcode` and `new_registry_obesity_subcode`. Their schema declares `categorical`, `no_event`, `attested`, `UNVALIDATED_CODES`, and no clinical dictionary or declared levels (`configs/meuhedet/phase5_v21_schema.yaml:347-354`). Neither is silently treated as a continuous variable in the linear design.

For raw nominal codes without levels, `LinearDesign.fit` counts finite numeric values **in that training fold**, retains each level with count at least 10, and uses sorted numeric order. The first retained level is reference; remaining levels get `f__eqVALUE` columns. Rare finite levels and unseen finite validation/test codes map to all-zero category columns with NA indicator `0`; missing codes map to all zeros with NA indicator `1` only when that indicator was learned in training. If fewer than two levels survive, no categorical dummy remains; an NA indicator may still survive. Missingness is not the declared category reference, and unseen/rare codes are not assigned an independent “other” parameter. Dropped duplicate/constant columns may make the fitted effective reference behavior more constrained. Sources: `design.py:45-47,81-95`; existing test `tests/unit/test_phase5_contract.py:369-376`.

`TreeDesign` only one-hot encodes nominal features with **declared** levels, using all levels and no omitted reference. The two actual subcodes have no declared levels and remain one raw float32 numeric column apiece. XGB therefore learns ordered code splits; this is **not native categorical XGBoost** and not a clinically validated ordering. Missing stays NaN. An unseen finite numeric code follows ordinary numeric split paths. Sources: `design.py:127-149`, `models.py:150-155,163-166`. This semantic assumption is a note for the ALL exploratory comparison; both subcodes are excluded from NEW_SAFE by provenance (`data.py:450-455`).

MEFI group is OLD, not new. `frail_mefi_group` and `frail_mefi_prev_group` declare ordinal levels `[1,2,3,4]` and thermometer encoding; the three linear thresholds are `>=2`, `>=3`, `>=4`, with conditional NA indicators and scaling. Trees keep raw ordinal numbers. Assessment presence, worsening, duration and counts are distinct catalogue features with their own types. Sources: `configs/meuhedet/phase2_features.yaml:102-108`; `phase5_v21_schema.yaml:360-369`.

**CCI remains continuous.** `com_cci_group` is configured `kind: continuous, linear: none`; the linear path copies the raw group code, median-fills it and z-scores it, then fits a single linear log-odds trend. The tree path keeps the raw code and native missingness. The V21 schema says this is the raw population-record CCI group, with no Charlson computation in the VIEW. Neither ordering nor equal spacing is validated by this audit. `SAFE_ATTESTED` timing eligibility is not proof of numeric semantics. Sources: `configs/meuhedet/phase2_features.yaml:177`; `phase5_v21_schema.yaml:371`; `data.py:460-465`; `design.py:50-51,89-117`. This assumption should be reviewed if CCI materially contributes, without pretending source inspection proves the fitted Windows contribution.

## Dates, missing states, eligibility and selection

**Dates:** `Index_Date`, IDs, outcome dates, follow-up fields, record dates and QA controls are not passed as numeric timestamp features. Date parsing follows declared formats, and rowwise days-since calculations subtract normalized dates from normalized index date (`data.py:135-149`; `phase2.engineer.py:31-36`). Days-since features use the catalogue's log transform or fall bands in linear models and raw numeric days in trees. Row-level dates also determine whether predictor records are known by the end of the index day (`phase3.recovery.py:108-188`; `data.py:552-579`). The same-day prediction contract is explicit. The fact that raw predictors are prepared before CV is not itself leakage when a parameter-free operation only uses a patient's allowed historical records.

**No-record states:** OLD canonical baseline values come from the fixed adapter; binary `any_positive` and count mappings can convert contract-defined source absence to zero (`data.py:468-511`; `data/meuhedet_wide.py:1084-1112`). OLD recovery masks post-index/unreadable records, and NEW type/timing checks likewise mask unknown or unreadable cells. `_no_record_state` gives zero for `present`; NaN for `not_assessed`/`no_event`; otherwise zero for binary/count and NaN for other kinds (`data.py:152-158`). Thus “all NULLs remain NULL” is not a correct blanket description of Phase 5: source-absent zeros and linear encoded zeros are intentional, while conditional NA indicators preserve many missing states. The per-feature table names each missing-semantic declaration.

**Eligibility precedes folds:** OLD recovery is called with `train_mask=np.ones(len(frame))`, so its minimum-known-observed/constant/timing gates are cohort-wide; the minimum is 100 and type-error cutoff is 1%. NEW timing shares and global V3 attestation are cohort-wide too. `prepare` then checks finite coverage and uniqueness across the usable cohort. Sources: `data.py:321-333,367-397,485-488,514-515,549-578`; `phase3.recovery.py:242-262,292-295`. These are not train-only outer preprocessing decisions. Label-blind cohort coverage/timing/type gating is more conservative than a fitted predictor transform, but can still make the assessed development procedure dependent on holdout covariates. Its boundary must be disclosed.

**Outcome-dependent screen is the substantive nesting concern:** `_univariate_auroc` computes AUROC using cohort outcomes and takes `max(AUC,1-AUC)`, treating missingness as a value below the smallest observed value. Any screened predictor with AUROC `>=0.80` becomes `INELIGIBLE_LEAKAGE`; membership and feature sets are then fixed before outer folds. Sources: `data.py:161-168,367-409`; `runner.py:285-307`. Calling the gate “exclusion-only” accurately describes its intent but does not remove the statistical use of outer holdout labels. Excluding a strong predictor can change every fitted model, and even a retained feature's eligibility was decided using the heldout labels. Bias direction/magnitude cannot be inferred solely from the source. A synthetic leaky-proxy fixture confirms this branch executes (AUROC 0.9772; excluded). It is **not** proof that a real Windows feature was excluded, that the real clinical predictors leak, or that the completed result is inflated. Deciding on retraining requires the real run's exclusion counts/provenance and a review of the intended evaluation estimand.

**Redundancy/variance:** Phase 5 does not call Phase 3's Spearman redundancy screening or low-variance feature selection. The only learned numerical removal in the active design is training-only constant and exact-duplicate encoded columns (`design.py:95-105`). Spearman correlation found in `phase5.explain.py:178` compares post-fit fold importance ranks; it is not a predictor selection gate.

**Missing-pattern grouping:** the former Phase 2 fitted pattern-grouping path is not imported or called by Phase 5. Each raw feature can create its own missing indicator inside `LinearDesign.fit`. Equal indicators can subsequently be removed as exact duplicates **within the current training fold**, not grouped by a full-cohort pattern before CV. Duplicate removal can suppress a feature's separate coefficient/attribution even when its raw source exists; do not confuse the 130 candidate features with each model's retained column count.

**Fractional polynomials:** no fractional-polynomial selector or fitted FP form appears in the Phase 5 graph. Log transforms, threshold flags, thermometer levels and fall bands come from fixed metadata. Reusing `phase2.engineer.compute` does not call FP selection. The concern about outcome-selected FP forms before inner CV is therefore inapplicable here. The solver import does not change this finding.

**Feature selection:** LASSO/ENET select nonzero coefficients within fitted paths; the configuration and operating thresholds are chosen from inner OOF predictions (`engine.py:227-240`). Final nonzero coefficients are those of the outer-training refit (`models.py:85-91`). Domains/ablations are declared from membership/domain metadata and do not rank predictors after outer outcomes (`runner.py:189-234`). Explanations and stability are downstream summaries and do not feed feature-set choice; stability bootstraps refit designs with the final development configuration (`phase5.explain.py:107-145`). The upstream AUROC eligibility gate is a separate, outcome-dependent feature-removal operation and must not be hidden behind a statement that all selection is nested.

## XGBoost preprocessing and boundaries

XGB receives engineered raw numeric values as float32, with NaN kept. It has no median imputation, z-scoring, added missing indicators, logarithmic count transform, thermometer expansion or fall-recency banding. Declared categorical metadata would produce fixed one-hot columns, but actual Smoking/Obesity subcodes remain numeric. `TreeDesign.fit(X_tr)` does not read observed categories or estimate any statistic in the current implementation (`design.py:120-149`). Its one outer-training design can safely be sliced for inner folds in that respect; future changes that learn categories/statistics would require revisiting that assumption.

The separate early-stopping split is carved exclusively from inner training, at fraction 0.15; inner validation is not used for early stopping. XGB hist bins and missing split direction are learned by `xgb.train` on the inner fit subset. The selected best iterations are aggregated by median; outer refit uses all outer training for that many trees. No outer holdout outcomes are passed to fitting (`models.py:141-169`, `engine.py:256-270,324-325`). XGB shares the same upstream cohort-wide eligibility issue. Fixed `scale_pos_weight=1` and `hist` settings are declared in `configs/meuhedet/phase5.yaml`.

## Disposition of the old concerns

| Old concern | Classification | Phase 5 evidence and disposition |
|---|---|---|
| Scaling itself nested correctly | NOT_APPLICABLE | This was not a defect to carry forward. Every inner fold creates its own `LinearDesign`, and the outer refit creates a new one; stored means/SDs apply to holdout (`models.py:41-42,85-86`; `design.py:108-117`). |
| Median imputation before inner CV | FIXED_IN_PHASE5 | Medians are fitted inside each inner `LinearDesign.fit`, on transformed training values; no cohort/outer preimputed array is passed to the inner solver (`models.py:41-42`; `design.py:81-95`). |
| Outcome-selected FP forms before inner CV | NOT_APPLICABLE | Phase 5 never selects FP forms; its active fixed encodings and `compute` have no outcome-selected nonlinear-form step (`design.py:29-51`; `phase2.engineer.py:22-37`; complete `phase5.engine`/`models` imports and call path). |
| Some feature eligibility decisions before outer CV | PRESENT_IN_PHASE5 | Coverage/timing/type gates use the cohort; critically, full-cohort AUROC uses labels before sets/folds (`data.py:367-409,485-488`; `runner.py:285-307`). Actual real-run exclusion impact is unknown. |
| Redundancy decisions before outer CV | FIXED_IN_PHASE5 | No inherited Spearman screening executes; exact duplicates/constants are dropped within each fitted linear design (`design.py:95-105`; `models.py:41,85`). |
| CCI group modeled continuous | PRESENT_IN_PHASE5 | `com_cci_group` remains continuous/raw, median-filled and standardized; raw group definition does not validate spacing (`phase2_features.yaml:177`; `phase5_v21_schema.yaml:371`). |
| Missing-pattern grouping too early | NOT_APPLICABLE | The old grouping pipeline does not execute. Indicators are learned within the current design's training rows, with training-only exact-duplicate pruning (`design.py:76-105`; `models.py:41,85`). |

These classifications concern the active Phase 5 **source** path. They do not retrospectively diagnose an inaccessible Windows artifact, and `FIXED_IN_PHASE5` does not imply that 0.12.2 changed the behavior relative to 0.12.1: the audited preprocessing files are identical in those source releases.

## Feature coverage and synthetic verification

Loading the tracked Phase 3 definitions and Phase 5 schema resolves **130 unique candidate names**: 93 catalogue features, 15 baseline features and 22 NEW V21 candidates. Every one has a row in the CSV. All actual NEW categorical candidates and every continuous, count, ordinal, binary and days feature are included, with source-derived raw inputs and encoding metadata.

On a freshly generated **synthetic null fixture** (`make_v21(n_rows=4000, seed=26, scenario='null')`, written to a unique temporary directory, then `prepare`), P0–P7 passed. There were 2,909 usable synthetic rows and 200 synthetic events; 127 candidate value/meta features; 105 OLD, 127 ALL and 124 SAFE features. Three OLD candidates are not reproducible on the authoritative V21 header: `com_registry_transplant`, `chronic_kidney_disease`, `hypertension`. The CSV retains those three explicitly as absent, so the source universe is not silently reduced. These numbers are **software fixture facts**, not clinical results or estimates of the Windows cohort.

A separate deterministic numeric fixture exercised `LinearDesign` and `TreeDesign` without training clinical models. All assertions passed:

* continuous training median `3` for `[1,3,NaN,7]`, log-count training median `log(4)` for `[0,3,NaN,15]`;
* retained design means zero and population SD one, including surviving binary/NA columns;
* raw categories `[1,2]` retained at count >=10, rare category 7 excluded, validation category 9 mapped without growing the schema;
* exact duplicate value and NA columns removed using only training observations;
* extreme validation values and new validation missingness did not mutate medians, means, SDs, categories or retained columns;
* no NA column was added when training had no missingness, even when validation did;
* tree numeric values were unchanged except float32 conversion, with NaN preserved and no scaling;
* the synthetic `leaky_new` trap triggered actual cohort-preflight exclusion of `new_deficit_count_proxy` at AUROC 0.9772.

The fixture used repository dependencies and called only the documented source constructors/functions, never the real training command or an existing artifact loader. The focused repository checks `test_raw_codes_learn_their_levels_on_training_rows_only`, `test_unknown_cells_take_the_no_record_state`, `test_known_new_features_are_new_and_mefi_stays_old`, and `test_identifiers_and_outcome_fields_never_enter_x` passed: **4 passed in 2.35s**. Existing source test `tests/unit/test_phase5_contract.py:369-376` records train-only raw-category discovery behavior. The existing holdout-label perturbation test (`tests/unit/test_phase5_contract.py:485-504`) starts from an already prepared context and flips labels only after eligibility has been fixed. It verifies unit tuning/refitting isolation, not independence of cohort-preflight feature eligibility. It therefore does not refute the upstream screening finding. Numerical fixture assertions are useful evidence of the local transform boundaries; source call-graph evidence establishes where the inner and outer training frames originate.

## What remains unknowable from this source audit

The exact Windows source/hash/configuration identities, resume history, real-run resolved membership, category vocabulary/reference levels, training medians/means/SDs, duplicate/constant columns, nonzero coefficients, CCI importance, convergence and model effects were not inspected. A dynamic real-run audit would need its plan/provenance, feature eligibility record and saved design/model metadata. The source stores such identities and fold/feature information in the frozen plan (`runner.py:314-324`) and embeds the fitted design inside `FittedModel` (`models.py:55-73`), but the existence of those mechanisms is not verification of an inaccessible run.

Accordingly: no scaling/imputation rewrite is justified by the old Phase 2 concerns alone; the cohort-wide outcome screen and CCI numeric semantics deserve explicit review; and no source-only claim of mandatory retraining or of real-model validity is made. No preprocessing or training implementation was changed by this audit.
