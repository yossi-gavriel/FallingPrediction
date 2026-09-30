# Resolution of the Astra final parameter review

**Scope.** The review is `reviews/ASTRA_FINAL_PARAMETER_REVIEW.md` (one Astra call, 47,171 tokens). Every item is resolved here by the ML lead, with:

- the decision;
- the technical rationale;
- the exact configuration or code change;
- the verification.

**No AGY call.** No item raised a new engineering or crash/resume problem. F-01 concerns XGBoost API semantics, which the code already met; it is proven by a new test. The earlier AGY review still covers the unchanged commit and resume protocol, so no AGY tokens were spent.

**Frozen result.** After these changes the frozen configuration is:

- `configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json`
- sha256 in `configs/meuhedet/FINAL_EXPERIMENT_CONFIG.sha256`, also reported in `FINAL_RUN_READINESS.md`.

The runner and the preflight refuse any other effective configuration (`FROZEN_CONFIG_MISMATCH`).

**Verdict (Astra).** "Once the MUST items are resolved, the configuration is scientifically fit for an exploratory incremental-value run and a matched-capacity management comparison within D00_CLEAN." **All 5 MUST items are closed. All 9 SHOULD items are accepted: 7 fully and 2 in part, with the technical reason given.**

## MUST FIX BEFORE RUN

### F-01 – native XGBoost early stopping

| Field | Content |
|---|---|
| Recommendation | Use `num_boost_round = 3000`. Score each inner holdout with `iteration_range=(0, best_iteration+1)`. Final fits use the median tree count without early stopping. Stage-2 depth ⊂ {1,2,3}. |
| Decision | ACCEPTED. The code already met every point. |
| Rationale | `xgb.train(..., num_boost_round=n_max, early_stopping_rounds=100)` returns the LAST model. The inner held-out fold is already predicted with `iteration_range=(0, best+1)`. `xgb_fit` trains exactly the frozen tree count, with no evaluation set. The stage-2 local depth range is clipped to the stage-1 range [1, 3]. |
| Change | Documentation only: the `xgb_inner_cv` docstring. FINAL_EXPERIMENT_CONFIG now names the key `num_boost_round_ceiling` and states the exact rule. |
| Verification | `test_phase2_final.py::test_xgb_native_early_stopping_rules`. The returned booster holds trees after the best iteration; the inner log loss equals the sliced prediction to 1e-9; the final fit has exactly `n_trees` rounds; the local depth range is always within 1–3. |

### F-02 – distinct identities for the benchmark

| Field | Content |
|---|---|
| Recommendation | Separate the Phase 1 historical model (recalibrated on VALIDATION) from the Phase 2 TRAIN-only `LASSO:BASELINE_15`. Do not claim reproduction. |
| Decision | ACCEPTED. |
| Rationale | The Phase 2 comparator is a TRAIN-only refit with no recalibration. Calling it "historical" would hide the Phase 1 model's calibration advantage on these VALIDATION outcomes. |
| Change | Every report labels it "BASELINE_15 refit (15 historical predictors, Phase 2 protocol, TRAIN only; not the recalibrated Phase 1 model)". SELECTION_FROZEN.json carries `benchmark_note`; FINAL_EXPERIMENT_CONFIG carries `selection.benchmark_identity`. The Hebrew summary says the model was refitted on TRAIN. The Phase 1 model is not re-scored: its metrics stay in the reference run's reports. The category name HISTORICAL_BASELINE_15 (the user's term) is kept for the predictor set. |
| Verification | e2e `test_full_run_outputs_and_share` (waterfall row order and labels); synthetic dev run (the summaries' text was read). |

### F-03 – scope of claims and neutral class names

| Field | Content |
|---|---|
| Recommendation | Limit claims to "exploratory recorded-event prediction in the D00_CLEAN selected cohort". Rename the classes MEETS_SUPERIORITY_RULE / WITHIN_NONINFERIORITY_MARGINS / OUTSIDE_MARGINS and label them point estimates. Describe CONFIRMED_PROMISING as an exploratory screen on reused VALIDATION. |
| Decision | ACCEPTED. |
| Rationale | Point estimates on about 256 validation events do not establish non-inferiority. Resealing does not undo earlier reuse. |
| Change | `stages_models.benchmark_class` returns the three new names. A scope sentence heads both summaries. The confirmation detail starts "exploratory screen on reused VALIDATION". `benchmark_class_kind` is recorded in SELECTION_FROZEN.json and FINAL_EXPERIMENT_CONFIG. |
| Verification | `test_phase2_final.py::test_benchmark_class_rule`; e2e run. |

### F-04 – item missingness threshold

| Field | Content |
|---|---|
| Recommendation | Change the 1% threshold for an item-specific missingness indicator to ANY non-identical item/form NULL pattern, unless documented semantics say omission means negative. |
| Decision | ACCEPTED. |
| Rationale | 1% of TRAIN is about 365 patients. Coding their unanswered items as "negative" would distort rare nursing predictors and bias the linear-vs-tree comparison. No form documents omission as negative. |
| Change | `screening.item_unanswered_min_share: 0.01 → 0.0`. An item is "covered" by its form's indicator only when its NULL pattern is identical to the form's; otherwise it gets an `na__` indicator. Identical patterns still share one indicator. |
| Verification | `design_spec` rule `mismatch <= 0.0`; synthetic dev run (design columns built, no duplicate-column failure); frozen config. |

### F-05 – death semantics of the outcome

| Field | Content |
|---|---|
| Recommendation | State that a recorded fall/fracture in the window stays an event when death follows, and that death before any event is a non-event. If the label overwrites events with death, fix the endpoint before the run. |
| Decision | ACCEPTED, as an audit plus a stop. Phase 2 cannot and must not rewrite the warehouse label: the user rule is "no modification of source values". |
| Rationale | The definition must be stated and checked, not assumed. |
| Change | New aggregate audit `cohort.death_label_audit` on TRAIN+VALIDATION rows only. It reports: deaths after the index; deaths inside the window; events and non-events among them; missing labels; `Label_Reason_180D` counts (small cells suppressed). The result goes into COHORT_FACTS.json (shared) and is printed by the preflight. New investigation gate `LABEL_DEATH_SEMANTICS` fires when at least 50 members died inside the window and none carries an event (suspected overwrite): the user must confirm the rule with the warehouse before accepting. The summaries print the measured rule in one sentence (`_death_sentence`). FINAL_EXPERIMENT_CONFIG has `outcome.death_rule`. |
| Verification | `test_phase2_final.py::test_death_label_audit` (suspected / kept / no deaths); e2e run; the preflight prints the audit. |

## SHOULD FIX BEFORE RUN

### F-06 – drop XGBoost stage 2

| Field | Content |
|---|---|
| Recommendation | Set `stage2.enabled → false`; keep the 15 stage-1 trials × 6 scopes. |
| Decision | ACCEPTED. |
| Rationale | The question is whether boosting adds credible value worth deeper investigation (the user's stated goal), not optimality. Stage 2 adds OOF-driven adaptation and up to 240 inner fits. If stage 1 shows value, deeper tuning becomes a declared future experiment. |
| Change | `xgb.stage2.enabled: false`. The `XGB2__decision` item records `stage2_enabled: false` and a note. The compute budget is now 90 stage-1 trials (450 inner booster fits); the stage-2 maximum is 0. |
| Verification | Frozen config; synthetic dev run (no XGB2 study, `XGB_TUNED` = stage 1); preflight output. |

### F-07 – common inner folds; conditional nesting

| Field | Content |
|---|---|
| Recommendation | Freeze common inner folds within each scope across λ / l1_ratio / trials. Move label-blind preprocessing inside the folds where practical; otherwise label OOF results as conditional. |
| Decision | PARTLY ACCEPTED. |
| Rationale | Common folds remove fold noise from within-scope comparisons at no cost. Moving label-blind redundancy, thermometer-level and NULL-pattern decisions inside every fold would make feature-set membership fold-dependent and break the frozen, hashed feature sets that the category invariant and the provenance audit depend on. The decisions use no outcome, so the condition is stated instead. The inner λ search is also conditional on the outer-fold FP choice for age (SAFE_BASE has no polypharmacy); per-inner-fold FP refitting would need a new solver interface. |
| Change | `stages_models.inner_fold_seed(ctx, family, scope)`: one inner-fold assignment per scope (outer<k> / final) for every LASSO / elastic-net set and every XGBoost stage-0 fit and Optuna trial of that scope. Booster randomness keeps the item seed. The summary's "Nesting" paragraph and FINAL_EXPERIMENT_CONFIG `cv_policy.common_inner_folds` / `conditionality` state the conditions. |
| Verification | Code inspection; synthetic e2e (deterministic tables across kills). |

### F-08 – selection fallback and validation safeguard

| Field | Content |
|---|---|
| Recommendation | Choose the simplest model only among qualifiers. Report NO_INCREMENTAL_MODEL_SELECTED when none qualifies. Add VALIDATION log loss ≤ 1.01 × SAFE_BASE to the promising rule. Undefined calibration cannot pass. |
| Decision | ACCEPTED. The first point was already the rule: `near` is computed within the qualifiers only. |
| Rationale | No self-comparison of the fallback, and the probability-quality safeguard also applies on VALIDATION. |
| Change | `selection_outcome` = INCREMENTAL_MODEL_SELECTED / NO_INCREMENTAL_MODEL_SELECTED (SELECTION_FROZEN.json, summaries); the confirmation status is NO_INCREMENTAL_MODEL_SELECTED in that case. `validation_confirmation.max_logloss_ratio: 1.01`. `_meets` rejects any NaN input (OOF and VALIDATION). |
| Verification | Unit tests for the NaN guard (via `benchmark_class`) and for NO_INCREMENTAL naming; e2e run. |

### F-09 – two frozen attribution contrasts

| Field | Content |
|---|---|
| Recommendation | Freeze (a) LASSO:ALL_REVIEWED_SAFE vs LASSO:SAFE_BASE (feature expansion) and (b) XGB on ALL_REVIEWED_SAFE vs the TRAIN-selected best linear model on the same set (pipeline comparison). Report paired intervals for both. |
| Decision | ACCEPTED. |
| Rationale | Separates feature effects from algorithm effects. No extra fits. |
| Change | `stages_models.attribution_contrasts`. The definitions are frozen in SELECTION_FROZEN.json (`attribution_contrasts`). Paired bootstrap intervals are computed on OOF (S09) and VALIDATION (S13, including recorded events identified per capacity). New `ATTRIBUTION_CONTRASTS.csv` (share) and summary section 6b. |
| Verification | e2e run (file present, both contrasts scored). |

### F-10 – value availability and the flag sensitivity

| Field | Content |
|---|---|
| Recommendation | Require provenance to cover value availability, including later updates. Rename the flag sensitivity WITHOUT_EXPLICIT_ASSESSMENT_FLAGS, override automatic flag insertion, and keep the needed item-missingness handling. |
| Decision | PARTLY ACCEPTED. |
| Rationale | The extract has no information about post-record value updates, so this cannot be proven or excluded. Strengthening the D-00 rule on an assumption would drop valid SAFE features without evidence. The risk is therefore disclosed per feature and raised with the warehouse. For the sensitivity: adding item-level NULL indicators where the flag was removed would rebuild the removed flag exactly (collinear), making the sensitivity void. It therefore removes only the explicit flags and keeps every other NULL indicator, and its name says so. |
| Change | Set renamed `ALL_REVIEWED_SAFE_WITHOUT_EXPLICIT_ASSESSMENT_FLAGS` (automatic insertion overridden, other `na__` indicators kept). Consensus caution "status / latest-value field: the record date bounds the record, not later updates of its value (warehouse to confirm)" on status / group / previous / worsened / at-index features. Limitation sentence in the summary. Open question Q-P2-04 for the DWH team (below). |
| Verification | e2e; unit `test_feature_sets_never_mix_categories`. |

### F-11 – calibration definitions, intervals, smooth curve

| Field | Content |
|---|---|
| Recommendation | Define scaled Brier, O:E and CITL. Add calibration-parameter intervals and a smooth curve with uncertainty; keep deciles. |
| Decision | ACCEPTED. The definitions already matched and are now written out. |
| Rationale | About 256 validation events make point calibration indices noisy. |
| Change | `evaluate.calibration_bootstrap`: percentile intervals for slope, intercept, CITL, O:E, Brier and scaled Brier. `evaluate.smooth_calibration`: restricted cubic spline of logit(p) with 4 knots and a 95% band, on a 50-point grid of predicted risks. Both run for the shortlist and the management waterfall (`CALIBRATION_BOOTSTRAP.csv`, `CALIBRATION_SMOOTH.csv`). CALIBRATION_SUMMARY.csv carries the intervals and the definitions. Figure 05 shows smooth curves, bands and decile points (small cells skipped). |
| Verification | Numeric sanity check (slope and CITL recovered on simulated calibrated data; the curve follows the diagonal); e2e run. |

### F-12 – management gains per 10,000 with intervals

| Field | Content |
|---|---|
| Recommendation | Resample the same patients across models and recompute top-⌈qn⌉ in every replicate. Express gains per 10,000 comparable D00_CLEAN patients, with intervals. |
| Decision | ACCEPTED. The first point was already true and is now optimised. |
| Rationale | This matches the declared capacity policy and makes no population extrapolation. |
| Change | `paired_bootstrap(..., capacities=...)` adds `delta_falls_<topN>` intervals at every capacity. It uses one stable sort per model per replicate, which is identical to `top_mask`. The management table and headline show intervals at every capacity, with the wording "per 10,000 comparable D00_CLEAN patients". |
| Verification | Point check: bootstrap Δ equals the `top_mask` difference, including ties; 200 replicates take 0.8 s on 12k rows. |

### F-13 – stability reporting

| Field | Content |
|---|---|
| Recommendation | Report counts, frequencies and Monte Carlo uncertainty. Keep a resampled patient's copies together in the inner CV. Apply stability only to the family assessed; define feature-level selection across encoded columns. |
| Decision | ACCEPTED. The grouping of copies was already done (`groups=idx`). |
| Rationale | 25 refits give a coarse frequency (SE ≈ 0.10 at 0.6). |
| Change | STABILITY.csv adds `selection_count`, `selection_frequency_mc_se`, `family` and a note. Consensus L now requires LASSO selection plus LASSO stability. An elastic-net-only selection is weak support, with a caution. |
| Verification | e2e run. |

### F-14 – frozen representatives, column lists, evidence wording

| Field | Content |
|---|---|
| Recommendation | Freeze representatives and domain membership, list the actual features added or removed, remove all encodings of an ablated feature, and label the consensus as exploratory predictive evidence. A positive single-feature permutation gain is weak support only. |
| Decision | ACCEPTED. Freezing and full removal were already true: the sets are built once from frozen representatives, and ablation removes the feature from the design, so all its encodings go. |
| Change | DOMAIN_INCREMENTAL_GAIN.csv adds `features_added` (waterfall and one-domain rows) and `features_removed_in_loo`. Consensus I = SHAP rank ≤ 15 of the promoted XGBoost. A positive permutation gain alone gives `evidence_I_weak_permutation_only`, which counts toward WEAK only. Every row carries `evidence_label` = "exploratory predictive evidence (not causal; conditional on the other features)". |
| Verification | e2e run. |

## ACCEPTABLE AS IS / FUTURE

| Item | Content | Action |
|---|---|---|
| F-15 | Keep the strict SAFE invariant, the separate UNRESOLVED sensitivity, the λ grid and floor, the elastic-net ratios, unweighted losses, the XGBoost stage-1 ranges, 10% principal capacity, penalised predictions and 2,000 paired bootstraps. | Kept unchanged. |
| F-16 | An untouched temporal cohort with prediction-time eligibility and explicit follow-up/death handling. | Recorded as the next step (SCIENTIFIC_SUMMARY "Next scientific step"; FINAL_RUN_READINESS). |

## Question-level notes that needed action

| Q | Astra note | Action |
|---|---|---|
| Q2 | Nesting is strict only if FP / preprocessing also run inside the inner folds. | = F-07: the conditionality is stated; the outer loop is fully nested, so OOF estimates stay valid. |
| Q4 | Say "average precision (AP)" consistently. | Reports say "AP (average precision = step-wise PR-AUC)"; the user's term PR-AUC is kept in parentheses. |
| Q10 | The hessian leaf-size reading is approximate; exclude depth 0 in stage 2. | Stage 2 is disabled; depth clipping is proven by a test anyway. |
| Q12 | Re-creating TPE is fine if it keeps the completed history. | Already true: the study is rebuilt from committed trials, and the sampler reads them. |
| Q15 | Smooth calibration and uncertainty. | = F-11. |
| Q18 | Status / latest-value fields can leak through later updates. | = F-10: caution plus Q-P2-04. Current registries and purchase / non-collection statuses are UNRESOLVED (structurally non-SAFE) and never enter SAFE sets. |
| Q21 | Keep eligible nursing features in the whole cohort; report assessed vs unassessed performance descriptively. | Already the design (SUBGROUP_SUMMARY: `nurse_form_assessed`, `mefi_assessed`). |
| Q24 | D00 cohort, reused validation, comparator identity, outcome interpretation. | Scope sentence (F-03), comparator identity (F-02), death audit (F-05), limitations. |

**Q-P2-04 (new, for the DWH team; answered by documentation, not by assumption).** For every status / latest-value / "at index" field:

- Can its value be updated after the record date carried in the extract?
- If so, which date bounds the value?

## What was NOT changed, and why

- **The λ grid (1e-3 floor), elastic-net ratios, XGBoost stage-1 ranges, 15 trials, principal capacity 10%, 2,000 bootstraps and 25 stability refits.** Astra judged them adequate (F-15). No brute-force expansion.
- **No class weighting and `scale_pos_weight` = 1.** Astra agreed (Q6, Q7). This is also enforced by the config loader.
- **The strict category structure** (SAFE_DISCOVERY primary, advancement against SAFE_BASE, point-estimate class against the BASELINE_15 refit). Astra: "the correct structure" (Q22).
