# Astra method review (Phase 2, before implementation)

| Item | Value |
|---|---|
| Reviewer | Astra = OpenAI `gpt-6-astra`, called through the local Codex CLI 0.153.4 (`codex exec`, sandbox read-only, empty working folder, no repository or data access) |
| Date | 2026-09-29 |
| Reasoning effort | high |
| Tokens used (reported by the CLI) | 16,268 |
| Input | `reviews/briefs/ASTRA_BRIEF.md` (1,770 words) plus one line: "Answer from the briefing text only. Do not run any commands or read any files" |
| Tool calls by the reviewer | none |
| Consultations so far | 1 of the planned maximum of 2 (the second is reserved for final interpretation, only if needed) |

How each recommendation was resolved is recorded in `reviews/REVIEW_DECISIONS.md`. The review text below is verbatim.

---

1. **The design is suitable for discovery, but its validation results will remain selection-dependent.** The single most important statistical change is: **change 30–40 validation comparisons to TRAIN-based selection followed by one comparison of a small, frozen shortlist**, ideally baseline plus 2–3 challengers.

   Use nested, grouped CV within TRAIN: outer folds estimate performance; inner folds perform tuning, feature selection, fractional-polynomial selection, imputation, scaling, and any data-dependent redundancy filtering. Ordinary OOF predictions are optimistic if the same folds selected the configuration being evaluated.

   Report outer-OOF and validation results, but do not claim that their agreement removes the winner’s curse or that OOF necessarily has lower variance. VALIDATION has already supported baseline recalibration; treat it as a reused development resource. Refit the baseline under the same evaluation protocol as challengers, with no validation-fitted recalibration. Keeping TEST closed is appropriate.

2. **Average precision is a reasonable primary discovery metric for this capacity-oriented task.** Low prevalence does not invalidate it; approximately 256 validation events make small differences uncertain, especially after model selection.

   Change “PR-AUC” to **“average precision, with the implementation fixed”** throughout: trapezoidal PR area and average precision are not interchangeable. Report absolute differences, paired intervals, and evaluation-set prevalence.

   Add log loss as a proper scoring rule and use probability quality as an advancement safeguard. If the intended product instead requires accurate absolute probabilities across many decision thresholds, log loss would be a stronger primary choice. Given the stated capture-at-capacity objective, retaining average precision is defensible.

   Predeclare one principal operational capacity. A gain in overall average precision can coexist with worse capture at the capacity the HMO can actually serve.

3. **Use an explicit decision hierarchy:**

   - **Eligibility:** prediction-time validity, usable coding, and an acceptable endpoint/cohort definition.
   - **Primary ranking:** average-precision improvement versus the freshly fitted baseline.
   - **Operational relevance:** capture and PPV at the principal capacity, with paired uncertainty; other capacities are secondary.
   - **Probability quality:** log loss, Brier, and calibration, assessed against predefined tolerances.
   - **Supporting discrimination:** AUROC; then prefer the simpler, more reliably available configuration when gains are uncertain.

   Report Brier against a constant-risk reference and the baseline: low prevalence makes a small absolute Brier score insufficient evidence of usefulness.

   Distinguish calibration-in-the-large, estimated with prediction slope fixed at 1, from the intercept in a jointly estimated intercept/slope calibration model. Include uncertainty and a calibration curve with uncertainty or event counts.

   **Change calibration cutoffs from automatic rejection gates to investigation triggers.** Distinguish a calibration shift potentially addressable by a prespecified recalibration procedure from unstable or badly distorted predictions. Capture describes recorded events identified, not falls prevented.

4. **Keep XGBoost deliberately small and strongly regularised.** A reasonable initial CPU search is 15–20 trials:

   | Parameter | Suggested specification |
   |---|---|
   | `max_depth` | 1–3 |
   | `min_child_weight` | 5–30, log-spaced |
   | `reg_lambda` | 1–30, log-spaced |
   | `learning_rate` | Fixed at 0.05 |
   | `subsample` | Fixed at 0.8 |
   | `colsample_bytree` | Fixed at 0.8 |
   | `gamma`, `reg_alpha` | Initially 0 |
   | Trees | Ceiling 3,000–5,000; early stopping after approximately 100 rounds |

   `min_child_weight` measures summed Hessian weight, not a patient or event count. These are starting ranges, not guarantees of adequate regularisation.

   Perform early stopping within inner CV; outer evaluation folds must remain untouched by tuning and stopping. For the final TRAIN fit, freeze a robust summary, such as the median selected tree count, from inner fits.

   No class weighting is a sensible default, but **unweighted fitting does not guarantee calibrated probabilities**. Omit monotone constraints initially unless a feature-specific direction is clinically defensible across its entire range and encoding.

   Compare families using identical eligible feature sets, folds, preprocessing boundaries, and selection rules. A practical choice is inner-CV log loss for tuning all families, followed by outer-OOF average precision for promotion. Tune both elastic-net mixing and penalty parameters inside that structure. Any additional XGBoost search must be part of the prespecified nested procedure. Benchmark runtime before promising overnight completion.

5. **The largest risks are timing uncertainty and selection induced by how this cohort was constructed.**

   - **Undated predictors:** SAFE_OR_UNRESOLVED is an exploratory evidence category, not leakage clearance. Baseline use does not validate the same uncertainty in new features.
   - **Aggregate provenance:** a date field establishes safety only if it covers every source contribution represented by that aggregate. Audit window boundaries, latest-value selection, and score/version timing.
   - **Nurse and frailty assessments:** observed values, missingness, and assessment recency can encode clinician suspicion, recent deterioration, and access to care. These may predict legitimately when available before index, but transport poorly when assessment practice changes.
   - **Utilisation and fall/fracture history:** enforce strict pre-index boundaries, distinguish encounters from distinct episodes, and prevent duplicate coding from inflating counts.
   - **Cohort exclusions:** removing people because they had index-day diagnoses or falls conditions inclusion on information unavailable at prediction time. This cannot be cured by refitting every model on the same filtered cohort.

   **Add a predictor-level provenance contract:** source, contributing-record dates, lookback, availability delay, missingness meaning, and derivation. Quarantine undocumented codes.

   Death as a non-event is coherent for “recorded fall before death within the horizon,” but changes interpretation and makes mortality patterns relevant. Excluding incompletely followed people can introduce additional selection bias. State these limitations explicitly and resolve them for the next cohort.

6. **Use domain-level comparisons as the main evidence for incremental value; individual-feature results should be explanatory.** Refit and retune each increment or ablation within TRAIN. Distinguish added value over baseline from unique contribution after other domains are present. Correlated predictors can have useful joint information and negligible individual ablation effects.

   For decisions **(a)–(h)**:

   - **(a)** Report nested outer-OOF and validation estimates. Change mandatory directional agreement to a consistency assessment with uncertainty; a near-zero sign reversal alone is not decisive.
   - **(b)** Retain average precision as primary, with log loss, calibration, and principal-capacity performance as safeguards.
   - **(c)** Retain assessment status and distinguish unassessed from assessed-negative. Avoid redundant dummy coding with form indicators. A sensitivity analysis removing explicit assessment indicators is useful, but call it that: missingness patterns and tree branches can still reveal assessment. It does not isolate “clinical content.”
   - **(d)** Change unresolved-feature admission to a separately labelled exploratory track. The model advanced as deployable must have prediction-time availability resolved. A SAFE-only baseline comparator will necessarily differ from EXTENDED-15.
   - **(e)** **Drop the 1% exclusion threshold.** Prefer reconstructing valid pre-index values; if impossible, mask the affected values and derived indicators where defensible, or exclude the source. Do not routinely remove people based on index-day information. Any masking rule must itself be implementable at prediction time.
   - **(f)** Change expanded-set LASSO to **penalised predictions as the primary method**. Keep the historical post-selection refit as a benchmark. Unpenalised refitting can amplify selection instability and overfitting.
   - **(g)** Drop the proposed either/or significance rule. Prespecify a minimum worthwhile gain and operational/calibration tolerances. Advance one challenger when the combined evidence supports a worthwhile improvement; otherwise advance baseline. A +0.01 average-precision threshold requires justification from operational consequences. Label uncertain improvement promising, not established.
   - **(h)** Retain one-domain additions and leave-one-domain-out comparisons. Treat the waterfall as descriptive and order-dependent; separate feature-set comparisons from model-family comparisons.

   Paired patient bootstrap is appropriate for differences between frozen predictions on the same people. With one row per member, it is an ordinary paired patient bootstrap. Its intervals do **not** account for searching configurations or training uncertainty; the bootstrap fraction above zero is not a posterior probability.

   Limit validation comparisons to the shortlist. Conduct exploratory ablations, permutation analyses, and importance ranking in TRAIN. Twenty-five stability bootstraps provide only a rough diagnostic; do not turn the consensus table into a vote among correlated importance measures or causal evidence.

7. **A new temporal claim requires a later, untouched cohort constructed to match intended deployment**, complete outcome ascertainment, and source extraction that enforces strict pre-index availability without deleting people based on index-day events.

   Freeze before accessing its outcomes:

   - Cohort, index/horizon boundaries, endpoint, death and incomplete-follow-up handling.
   - Sources, timing rules, feature definitions, missingness processing, and coding.
   - Model family, hyperparameters, fitting/refitting procedure, and training-data cutoff.
   - Any recalibration procedure and the independent data used to fit it.
   - Primary metric, operational capacity, selection rule, and analysis code/version.

   If TRAIN and VALIDATION will be combined for final fitting, prespecify that step before holdout access. A recalibrator cannot be fitted and assessed on the same holdout; use an earlier calibration period or a separate calibration sample.

   Report holdout prevalence and case-mix changes: average precision, PPV, and calibration depend on them. Show original-model performance before any updating. Ensure earlier training labels would actually have been available at the simulated deployment date.

   Repeated members across time may match deployment, but require explicit handling. Temporal validation within Meuhedet supports later-period performance there; broader generalisability requires additional settings or populations.

8. **Top five changes, ranked by impact on validity:**

   1. **Replace SAFE_OR_UNRESOLVED eligibility and outcome-informed row deletion with a deployment-valid timing policy**, retaining unresolved analyses as exploratory.
   2. **Replace broad validation selection with nested TRAIN selection and a small frozen shortlist**, acknowledging previous validation reuse.
   3. **Resolve the endpoint and cohort estimand**, especially index-day exclusions, incomplete follow-up, and death.
   4. **Use penalised predictions for expanded linear models and evaluate all families under one protocol**, including an identically handled baseline.
   5. **Replace binary significance gates and importance consensus with a prespecified advancement rule** integrating worthwhile average-precision gain, operational capture, calibration, uncertainty, and simplicity.

   Also change “implausible gain,” proxy dominance, and calibration thresholds into documented audit triggers. Keep timing violations, split contamination, and prohibited inputs as hard stops.