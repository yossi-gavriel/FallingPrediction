# Astra final parameter review (Phase 2, frozen configuration before the real-data run)

| Item | Value |
|---|---|
| Reviewer | Astra = OpenAI `gpt-6-astra`, called through the local Codex CLI 0.153.4 (`codex exec`, sandbox read-only, empty working folder, no repository or data access), reasoning effort high |
| Date | 2026-09-29 |
| Input | `reviews/briefs/ASTRA_FINAL_BRIEF.md` (the actual frozen values of configs/meuhedet/phase2.yaml 1.1.0 and phase2_features.yaml 1.1.0, code 0.8.1; sha256 `a22c849599bae6d7376345a3cc4cd5dc068e8f4d5603bab0deac568c088c9ef6`) |
| Tokens used | 47,171 (one call; the second and last Astra consultation of Phase 2 before any real-data result) |
| Answer sha256 | `79ca7366f58d49267fd8502c1df455fcd67b0522bae4adcaae023e8270876721` |
| Resolution | `reviews/ASTRA_FINAL_RESOLUTION.md` (every item: decision, rationale, exact change, verification) |

The text below is Astra's answer, verbatim.

---

**The configuration is broadly suitable for exploratory discovery, with several corrections and clarifications before launch.** This reviews the specification; without code or data, I cannot certify its implementation.

**Q1. Selection validity?** Yes, as a development-and-selection exercise. Selecting across many outer-OOF results introduces selection optimism; those results do not independently validate the winner. Reused VALIDATION provides additional exploratory evidence.

**Q2. Nested CV correct?** The supervised nesting is appropriate if FP selection and preprocessing occur inside **every inner training fold**, too. TRAIN-wide redundancy decisions, OOF-selected EN sets and conditional Stage 2 mean the complete adaptive procedure is not strictly nested.

**Q3. Seal VALIDATION until selection freezes?** Yes. Scoring all prespecified configurations once is acceptable, provided only the declared comparison determines confirmation and no validation winner replaces the recommendation. The seal does not erase previous reuse.

**Q4. AP as primary metric?** Yes, for prioritising recorded events at low prevalence, alongside capacity metrics and probability-quality safeguards. Call it **average precision (AP)** consistently: it differs from trapezoidal PR-AUC. [Definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

**Q5. XGBoost tuning objective?** Retain `binary:logistic`, inner-CV log loss and log-loss early stopping. AP-based outer selection is a defensible compromise; direct AP tuning would be noisy with approximately 123 events per outer-scope inner holdout.

**Q6. Imbalance handling?** Appropriate. Approximately 769 TRAIN events support fitting without resampling or class weighting. Imbalance alone does not require changing the training distribution.

**Q7. `scale_pos_weight`?** Keep **1**, untuned. Probability quality matters, and recalibration is prohibited; weighting would complicate that objective. [XGBoost guidance](https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/param_tuning.html)

**Q8. LASSO settings?** Sensible: retain 100 λ values, the measured `1e-3` floor, penalised predictions and convergence requirements. Boundary selection signals a constrained search, not automatic invalidity. Non-stratified ten-fold CV is acceptable at these event counts.

**Q9. EN ratios?** Sufficient for direction finding. `{0.1, 0.5, 0.9}` plus the separately fitted LASSO covers useful alternatives; adding ridge or a dense ratio grid is unnecessary.

**Q10. XGBoost ranges?** Sensible and conservative for the actual fitting sizes, not 60,851 independent training observations. Hessian-based leaf-size equivalents change during boosting and with subsampling. Stage 2 must explicitly exclude depth zero.

**Q11. Early stopping?** Statistically appropriate: inner holdouts choose iterations, and outer holdouts remain untouched. Native-API round limits and best-iteration scoring need explicit rules; see F-01.

**Q12. Optuna budget?** Fifteen trials per scope is adequate to investigate direction in three dimensions, without guaranteeing an optimum. Recreating TPE is statistically acceptable if it retains completed study history; startup counts depend on that history. [Optuna documentation](https://optuna.readthedocs.io/en/v4.6.0/reference/samplers/generated/optuna.samplers.TPESampler.html)

**Q13. Stage 2 promotion?** Reasonable as a compute-allocation heuristic, not a statistical significance criterion. Its OOF-dependent activation adds adaptation. I would omit Stage 2 for this run; the first stage already answers the principal question.

**Q14. Principal capacity 10%?** Yes, as a declared operational scenario. It is not evidence that the organisation can actually intervene on 10%. At approximately 256 validation events, one capture percentage point represents only about 2.6 events.

**Q15. Calibration and Brier?** Broadly correct. Define scaled Brier and calibration estimators explicitly, add uncertainty, and supplement deciles with a smooth calibration plot. A low raw Brier alone is unsurprising at 2.1% prevalence. [Validation guidance](https://www.bmj.com/content/384/bmj-2023-074820)

**Q16. Stability/bootstrap budget?** Two thousand paired prediction bootstraps are sufficient. Twenty-five model bootstraps provide coarse stability descriptions: a frequency of 0.60 has Monte Carlo SE about 0.10. They cannot establish a precise stability boundary.

**Q17. Ablation usefulness?** Yes. Retuned domain removal estimates incremental predictive contribution conditional on retained domains. Fixed-hyperparameter feature removal is explanatory and selection-conditioned; correlated substitutes can conceal contribution. Neither estimates causality.

**Q18. Leakage-prone features?** Potentially: purchase/noncollection status, current registries, cumulative utilisation, “previous/worsened” assessments and latest-record summaries. A pre-index event date is insufficient if its attached value was updated afterward. Derived flags and missingness require the same provenance.

**Q19. Missingness handling?** The overall structure is sound, but the 1% item-indicator threshold can conflate unanswered items with genuine zero values among assessed patients. Identical-pattern sharing is reasonable; see F-04.

**Q20. Assessment presence as proxy?** Yes. It can encode frailty, clinician concern, access and opportunities to record outcomes. Predictive usefulness is legitimate but process-dependent; SHAP cannot distinguish these mechanisms.

**Q21. High-missingness nursing features?** Retain eligible SAFE features in the whole-cohort analysis with explicit missingness handling. Report coverage and assessed/unassessed performance descriptively. Subcohort-only analyses would change the target population and should remain secondary.

**Q22. Changes before the run?** Yes: the MUST items below. SAFE_DISCOVERY-primary is the correct structure. Keep advancement against SAFE_BASE separate from historical benchmarking; improving an age/sex-dominated SAFE_BASE need not establish competitiveness against BASELINE_15.

**Q23. Five checks after a surprising XGBoost gain?** Inspect: (1) value availability and temporal leakage; (2) cohort, patient and outcome alignment; (3) preprocessing/tuning/early-stopping contamination; (4) assessment, missingness and healthcare-contact proxies; (5) a few influential events, ties or prediction/metric errors driving the gain.

**Q24. Material omissions?** The main threats are the future-conditioned D00 cohort, reused validation, ambiguous historical comparator and outcome interpretation. These constrain scientific claims even with perfect fitting. A later untouched temporal cohort with prediction-time eligibility remains necessary.

**F-01 — MUST FIX BEFORE RUN.**  
**Rule:** For native `xgb.train`, map the ceiling to `num_boost_round=3000`; score each inner holdout using `iteration_range=(0, best_iteration+1)` or an equivalent sliced booster. Final scope fits use the declared median tree count without early stopping. If Stage 2 remains, depth choices are `{best−1,best,best+1} ∩ {1,2,3}`.  
**Rationale:** `n_estimators` is not the native training argument; native early stopping returns the last model, and depth zero means unlimited depth. These are specification checks, not claims that the code is wrong. [API](https://xgboost.readthedocs.io/en/release_3.2.0/python/python_api.html), [parameters](https://xgboost.readthedocs.io/en/release_3.2.0/parameter.html)

**F-02 — MUST FIX BEFORE RUN.**  
**Rule:** Give distinct identities to **Phase-1 historical BASELINE_15**, including its already-fitted validation recalibration, and **Phase-2 TRAIN-only LASSO:BASELINE_15**. Use the latter for OOF and common-protocol validation comparisons. Historical predictions may be displayed separately, labelled as previously calibrated on these validation outcomes. Do not claim a TRAIN-only refit exactly reproduces the historical model.  
**Rationale:** One identifier currently conflates different fitting histories and calibration advantages.

**F-03 — MUST FIX BEFORE RUN.**  
**Rule:** Restrict conclusions to “exploratory recorded-event prediction in the D00_CLEAN selected cohort.” Replace formal-sounding benchmark classes with **MEETS_SUPERIORITY_RULE**, **WITHIN_NONINFERIORITY_MARGINS**, and **OUTSIDE_MARGINS**; identify them as point-estimate classifications. Describe `CONFIRMED_PROMISING` as an exploratory screen on reused validation.  
**Rationale:** Point estimates do not establish non-inferiority, and neither resealing nor bootstrapping repairs prior reuse or future-conditioned inclusion.

**F-04 — MUST FIX BEFORE RUN.**  
**Parameter:** Item-specific missingness threshold `>1%` → **any non-identical item/form NULL pattern**, unless documented form semantics establish that omission means negative. Continue sharing identical indicators.  
**Rationale:** One percent of TRAIN is approximately 365 patients; suppressing their missingness can materially distort rare nursing predictors and the linear-versus-tree comparison.

**F-05 — MUST FIX BEFORE RUN.**  
**Rule:** State explicitly: “A qualifying recorded fall/fracture within the outcome window remains an event even if death follows; death before any qualifying event counts as a non-event.” If the frozen label instead overwrites prior events with death, correct the endpoint before running.  
**Rationale:** “Deaths count as non-events” must not silently erase observed outcomes; otherwise this predicts a different composite endpoint.

**F-06 — SHOULD FIX BEFORE RUN.**  
**Parameter:** `stage2.enabled → false`; retain Stage 1’s 15 trials and six scopes.  
**Rationale:** Stage 1 already requires **450 inner booster fits**; Stage 2 adds up to **240** without being necessary for this exploratory question. The initial nested budget is defensible; more search is not presently justified.

**F-07 — SHOULD FIX BEFORE RUN.**  
**Rule:** Freeze common inner-fold assignments within each scope across λ/ratio/trial comparisons. Move redundancy, level discovery and NULL-pattern grouping inside training partitions where practical; otherwise label OOF results conditional on TRAIN-wide unsupervised preprocessing.  
**Rationale:** This reduces comparison noise and makes the limits of the nesting claim explicit without adding model fits.

**F-08 — SHOULD FIX BEFORE RUN.**  
**Rule:** Choose the simplest model **only among qualifying candidates** within 0.005 AP of the best qualifier. If none qualifies, report **NO_INCREMENTAL_MODEL_SELECTED**. Add validation log loss `≤1.01 × SAFE_BASE log loss` to the promising rule; undefined calibration estimates cannot pass.  
**Rationale:** The fallback should not undergo a self-comparison, and validation should preserve the probability-quality safeguard.

**F-09 — SHOULD FIX BEFORE RUN.**  
**Rule:** Freeze two attribution contrasts: LASSO:ALL_REVIEWED_SAFE versus LASSO:SAFE_BASE for feature expansion, and XGB:ALL_REVIEWED_SAFE versus the TRAIN-selected best linear model on that **same set** for model-family value. Report paired intervals for both.  
**Rationale:** Overall-winner comparisons can mix feature and algorithm effects; these require no additional fits. Call the latter a pipeline comparison because encodings and missing-value handling differ.

**F-10 — SHOULD FIX BEFORE RUN.**  
**Rule:** Require provenance to cover the value’s availability at prediction time, including later updates and derived statuses. Name the flag sensitivity **WITHOUT_EXPLICIT_ASSESSMENT_FLAGS**, override automatic flag insertion for that set, and retain necessary item-missingness handling.  
**Rationale:** An old source date does not prove an attached status was available then; removing explicit flags does not remove all assessment-process information.

**F-11 — SHOULD FIX BEFORE RUN.**  
**Rule:** Define scaled Brier as `1 − Brier/[p_eval(1−p_eval)]`, O:E as `sum(y)/sum(p)`, and CITL through a fixed-slope logit-offset model. Add calibration-parameter intervals and a smooth curve with uncertainty; retain deciles descriptively.  
**Rationale:** These definitions prevent inconsistent reporting, while uncertainty matters with approximately 256 validation events.

**F-12 — SHOULD FIX BEFORE RUN.**  
**Rule:** In each paired patient bootstrap, resample the same patients across models and recompute top-`ceil(qn)` membership. Express management gains as **additional recorded events identified per 10,000 comparable D00_CLEAN patients**, with intervals.  
**Rationale:** This estimates the declared capacity policy; it neither estimates falls prevented nor supports extrapolation to all members.

**F-13 — SHOULD FIX BEFORE RUN.**  
**Rule:** Retain 25 stability resamples; report selection counts, frequencies and Monte Carlo uncertainty. Keep duplicate copies of each bootstrapped patient together in inner CV. Apply stability evidence only to the family/set actually assessed; define feature-level selection across encoded columns.  
**Rationale:** Coarse LASSO stability cannot certify EN-only findings, and duplicated patients must not cross inner partitions.

**F-14 — SHOULD FIX BEFORE RUN.**  
**Rule:** Freeze redundancy representatives and domain membership across additions/removals; list actual columns added or removed. Remove all encodings of an ablated feature. Label consensus classes “exploratory predictive evidence,” with single-feature permutation gains above zero treated as weak support.  
**Rationale:** This preserves interpretable contrasts and avoids overstating post-selection importance or shared-source evidence.

**F-15 — ACCEPTABLE AS IS.**  
**Rule:** Retain the strict SAFE invariant, separate UNRESOLVED sensitivity, λ grid/floor, EN ratios, unweighted losses, XGBoost Stage-1 ranges, principal 10% capacity, penalised predictions and 2,000 paired bootstraps.  
**Rationale:** None needs expansion to provide a credible directional answer.

**F-16 — FUTURE IMPROVEMENT.**  
**Rule:** After this discovery run, freeze the candidate list and evaluate an untouched temporal cohort assembled solely from prediction-time eligibility, with explicit follow-up/death handling and operational subgroup assessment.  
**Rationale:** That addresses generalisability and selection bias that this run cannot resolve.

**Verdict: Yes—once the MUST items are resolved, the configuration is scientifically fit for an exploratory incremental-value run and a matched-capacity management comparison within D00_CLEAN.**