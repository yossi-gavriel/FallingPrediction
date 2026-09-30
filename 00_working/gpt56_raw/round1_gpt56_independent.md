# 1. Primary outcome definition

The primary endpoint should be named precisely:

> **An acute, medically documented, unintentional fall occurring after the index date and before death, within the prediction horizon.**

This is not equivalent to “all falls.” Many falls never reach medical attention, while documentation depends on injury severity, care-seeking, access, and coding practice. Publications and executive communication must describe the model as predicting **medically documented falls**, unless a prospective fall-calendar or patient-report study later establishes validity for all falls.

Use the Prevention of Falls Network Europe definition: an unexpected event in which a person comes to rest on the ground, floor, or lower level (Lamb et al., 2005, *Journal of the American Geriatrics Society*). Loss of consciousness, environmental causes, and falls from height may count if unintentional; assault, deliberate self-harm, and events of undetermined intent should not automatically count.

## Evidence tiers

Create a versioned fall-episode table from validated sources, using raw codes rather than assumed normalized values.

- **Tier 1: high-specificity acute fall event.** A verified fall/external-cause code in an acute encounter or another source shown by chart adjudication to describe a new event, with a credible encounter/event date. Eligible sources may include emergency, hospitalization, urgent ambulatory care, or other settings, but all source tables and fields are **UNKNOWN — TO DISCOVER**.
- **Tier 2: probable acute fall.** A verified fall code in a less-specific ambulatory or claims context, or a structured statement explicitly referring to a new fall since a known prior contact. It may enter the primary endpoint only if source-specific positive predictive value and date validity are adequate.
- **Tier 3: supporting but insufficient evidence.** Injury codes without a fall mechanism, gait difficulty, fall-risk assessments, historical-condition codes, late effects, sequelae, or retrospective “fell during the last year” responses. These should not independently define the primary event.

The current candidate codes are quarantined until mapping is proven:

- `719.7` appears consistent with difficulty walking and should be a predictor, not a fall outcome.
- `880.9`, `883.9`, `884.9`, `886.9`, `888`, and `929.3` must not be treated as external-cause codes until Meuhedet’s raw representation, code system, prefix stripping, decimal normalization, and dictionary are verified. `929.3` is especially concerning as a possible late-effect code.
- `E987` concerns a fall from height with undetermined intent and should be excluded from the primary unintentional-fall endpoint unless adjudication establishes an acceptable interpretation.
- ICD-10, local codes, and code-system transition dates are **UNKNOWN — TO DISCOVER**.

## Label validation before modeling

Before feature engineering or model fitting:

1. Produce code-by-source-by-year frequency tables, including raw and normalized strings.
2. Identify encounter type, clinical source, diagnosis role, service date, admission date, discharge date, posting date, onset date, history/sequela indicators, and data-availability timestamp. All relevant fields outside the named nursing structures are **UNKNOWN — TO DISCOVER**.
3. Conduct an in-Meuhedet chart-review study, stratified by code, source, era, and acute versus ambulatory setting. Sample candidate positives and a random sample of apparently negative person-time to assess both positive predictive value and missed events.
4. Have two clinicians independently adjudicate event status, intent, setting, injury, and event date, with a third resolving disagreements. Report agreement and reasons for disagreement.
5. Compare sources through overlap analyses. A later nursing report of “a fall in the previous year” may test capture sensitivity of medical-event sources, but it is not an exact gold-standard event date.
6. If feasible, run a small prospective substudy using recommended fall calendars and scheduled follow-up. This is the most defensible way to quantify how prediction of medically documented falls differs from prediction of all falls.

## Date assignment and de-duplication

Use the time when an event actually occurred, or the earliest acute encounter attributable to it—not the warehouse entry date.

Provisional precedence should be:

1. validated explicit occurrence date;
2. emergency-arrival or admission date;
3. acute ambulatory encounter date;
4. validated claim service date.

Posting or ingestion dates determine whether information was available to the model but should not replace the clinical event date. Records with only an imprecise historical interval should be treated as interval-censored supporting evidence or excluded from the primary endpoint.

Combine records into one fall episode using encounter identifiers and source linkage where possible. Otherwise, evaluate 3-, 7-, and 14-day episode windows; use seven days provisionally, retaining the earliest credible date and maximum severity. A second explicitly documented new fall inside that window should remain separate. The rule must be finalized from adjudicated data rather than convention alone.

## Primary and separate outcomes

- **Primary:** time to first qualifying fall after each landmark. Previous fallers remain eligible; “first” means first after that landmark, not first in the patient’s lifetime.
- **Secondary:** any qualifying fall within 90, 180, and 365 days.
- **Separate secondary endpoint:** serious/injurious fall, defined as a validated fall episode accompanied by emergency care, hospitalization, or a validated clinically important injury during the same episode. Injury definitions and source fields are **UNKNOWN — TO DISCOVER**.
- **Exploratory:** recurrent falls, such as two or more distinct episodes within 365 days or fall-event rate. This requires sufficiently reliable episode separation and should use recurrent-event or count methods.
- Inpatient-acquired falls should be separated from community falls if setting can be identified, because prevention pathways differ. Setting fields are **UNKNOWN — TO DISCOVER**.

# 2. Cohort definition

The source roster begins with `[Meuhedet_DWH].[Dims].[Dim_Customer_Details]`, but membership periods, death data, and historical eligibility fields remain **UNKNOWN — TO DISCOVER**.

Include patients who, at a landmark:

- are aged 65 years or older;
- are alive;
- have active Meuhedet eligibility;
- have at least 365 days of observable look-back;
- are not currently in an acute inpatient episode if the intended intervention is community-based. Admission information is **UNKNOWN — TO DISCOVER**.

Use exact age continuously, preferably with restricted cubic splines or another flexible smooth representation. The 65–74, 75–84, and 85+ bands are for reporting and subgroup assessment, not the main age predictor.

A 365-day minimum look-back balances representativeness and history ascertainment. Features may use up to 730 days when available, but membership history must support them. A sensitivity analysis should admit newer members with shorter look-back and explicitly encode observable-history length. Permitted administrative enrollment gaps cannot be assumed; gap rules are **UNKNOWN — TO DISCOVER**.

Do not require future survival or complete 180-day membership when defining the cohort. That would condition on future information and select survivors. Instead:

- death is a competing event;
- disenrollment or end of available data causes censoring;
- loss to follow-up should be examined for informativeness and, when necessary, addressed using inverse-probability-of-censoring methods;
- complete-case binary analyses are secondary sensitivity analyses only.

Include prior fallers. Excluding them would remove the strongest known risk group and produce a less clinically useful population. Report performance separately among patients with and without prior documented falls.

Institutionalized, long-term-care, bed-bound, and wheelchair-bound patients should not automatically be excluded. Transfer falls and care-setting falls remain possible, and exclusion could create inequity. However:

- institutional/long-term-care status is **UNKNOWN — TO DISCOVER**;
- bed/wheelchair status should be represented using pre-index nursing fields, including field 3529 and rehabilitation/mobility fields;
- evaluate these groups separately because exposure opportunity, ascertainment, and feasible interventions differ;
- an implementation trial may restrict intervention eligibility if Meuhedet cannot offer an appropriate pathway, but that is a downstream operational decision, not a retrospective labeling shortcut.

# 3. Index-date / observation-unit strategy

| Strategy | Strength | Principal weakness |
|---|---|---|
| Fixed annual date | Simple, one row per patient-year | Misses changes during the year; poor operational realism |
| Visit-based | Easy to trigger during care | Selects high utilizers and cannot score people who do not visit |
| Assessment-based | Strong nursing information near index | Severe selection bias because assessment is clinically ordered |
| Patient-quarter | Moderate compute and correlation | Up to three months of delay after risk changes |
| Rolling monthly landmarks | Deployable cadence; updates medication, falls, function, and utilization | Repeated correlated records and overlapping outcome windows |
| Daily/continuous landmarks | Maximum responsiveness | Usually unnecessary and computationally burdensome |

The primary design should be **rolling monthly landmarks**: one eligible snapshot per person on a standardized calendar day each month. All predictors must have been available by the end of the preceding day. Same-day information should be excluded unless real-time ordering and availability are unambiguous.

This supports monthly batch scoring and does not depend on a visit or assessment. It also allows a patient to re-enter after a fall with updated prior-fall history, matching real prevention operations.

Repeated snapshots require:

- patient-level grouping in every non-temporal split;
- patient-clustered bootstrap confidence intervals;
- cluster-robust or patient-random-effect sensitivity analyses;
- explicit handling of overlapping 180-day labels;
- no patient identifier or near-identifier as a model feature.

For computational efficiency, model development may use a probability sample of negative landmarks with correct sampling weights, while retaining every positive and every validation/test landmark. Final calibration and evaluation must use the natural, unsampled population.

# 4. Observation windows and prediction horizons

The proposed 180-day horizon is defensible but should not be accepted merely because it was initially suggested. It must correspond to the time needed to contact the patient, assess modifiable risks, start an intervention, and realize benefit.

I recommend:

- **Primary horizon: 180 days**, provisionally retained. Six months is long enough to accumulate events and implement exercise, medication review, home-safety, vision, or orthostatic interventions, while remaining more current than a one-year prediction.
- **Secondary: 90 days**, representing more immediate risk but likely producing fewer events and greater sensitivity to acute illness.
- **Secondary: 365 days**, comparable with annual fall-history and screening practices but more vulnerable to changes after prediction and competing mortality.

The choice should be re-opened if discovery data show insufficient 180-day events, long intervention delays, or materially better clinical utility at another horizon.

Use a common set of recency windows:

- 0–30 days: acute changes, visits, hospitalization, medication starts, and recent measurements;
- 31–90 days: subacute instability;
- 91–365 days: established status and annual history;
- 366–730 days: prior falls, trajectories, and chronic history where continuous observation permits.

Avoid “ever in the database” predictors because their meaning depends on membership duration and system maturity.

The main statistical analysis should be **time to first fall with death as a competing event**, producing a 180-day cumulative incidence. A technically straightforward implementation is a discrete-time competing-risk model with monthly intervals and mutually exclusive transitions to fall, death, or continued follow-up. Cause-specific survival and Fine–Gray models can be sensitivity analyses; Fine and Gray introduced direct subdistribution-hazard modeling in 1999 in *JASA*.

Binary 180-day models remain important operational comparators, but censoring must be handled with survival-aware methods or inverse-probability weights. Treating everyone who dies as an ordinary non-faller changes the estimand and should not be the main approach.

# 5. Feature taxonomy by domain

All features are calculated strictly from information available before the landmark. For diagnoses and claims, availability time may differ from service time; warehouse backfill and posting fields are **UNKNOWN — TO DISCOVER**.

| Domain | Prespecified features and windows |
|---|---|
| Demographics | Continuous age; sex; observable membership duration; validated socioeconomic/deprivation measure if available. District/clinic may be used for validation and recalibration, not automatically as a risk determinant. Other demographic fields are **UNKNOWN — TO DISCOVER** and require fairness review. |
| Fall history | Validated fall counts in 30/90/365/730 days; days since most recent fall; prior serious fall; number of distinct episodes. Retrospective answers such as field 1574 are features known from the assessment date onward, never backdated to an invented event date. |
| Nursing assessments | For each validated field: latest value, worst value, number assessed, days since assessment, change from previous value, and an assessment-available indicator, principally over 365 days and secondarily 730 days. Field meaning, ordering, units, response categories, and historical recoding must be frozen in a dictionary. |
| Fall Risk Assessment 999.52 | Fields 1574–1576, 2426–2428, and 3529, after checking response semantics. Separate patient-reported history from clinician overall-risk judgment. |
| Get Up and Go 999.79 | Known gait, balance, weakness, cognition, sensory, orthostatic, polypharmacy, and concern fields. Every field, any total score, elapsed time, protocol, derived field, and coding change is **UNKNOWN — TO DISCOVER**. Do not calculate an invented total. |
| Home Safety 999.139 | Fields 1574 and 2439–2442 and 2480. Location and device values require harmonization. |
| Rehabilitation/Mobility 999.175 | Vision, cognition, mobility, transfers, cane/crutches, walker, wheelchair, and wheelchair mobility. Preserve clinically meaningful ordering only after confirming category semantics. |
| Cognition | Mini-Cog field 2070; MMSE field 1439 where complete; last score, worst score, change, and days since test. Add validated dementia, cognitive-impairment, and delirium history from diagnosis sources **UNKNOWN — TO DISCOVER**. |
| Mobility/gait/balance | Structured gait and transfer findings, assistive-device use, physical-therapy or rehabilitation contacts, mobility-limiting diagnoses, and walking-difficulty indicators. Relevant treatment/referral sources are **UNKNOWN — TO DISCOVER**. |
| Vision/hearing | Structured impairment and aid-use fields, relevant diagnoses, and recent ophthalmology/audiology utilization if reliable. Do not infer impairment solely from absence of an aid. |
| Orthostatic/dizziness/syncope | Structured dizziness and orthostatic findings; validated syncope, vertigo, hypotension, arrhythmia, and hypoglycemia encounters; last blood pressure and orthostatic change if measured explicitly. |
| Medications | Current medication and fall-risk-increasing-drug counts; starts, stops, dose increases, and class changes within 7/30/90 days; dispensing continuity over 180/365 days. Drug, dispensing, dose, and days-supply fields and mapping are **UNKNOWN — TO DISCOVER**. |
| Diagnoses | Parkinsonism, stroke, neuropathy, dementia, depression, vestibular disease, arthritis, lower-limb impairment, urinary incontinence, syncope, orthostatic hypotension, diabetes/hypoglycemia, cardiovascular disease, anemia, renal disease, and frailty-related conditions. Use clinically curated groups rather than thousands of unreviewed codes. Osteoporosis is more relevant to the serious-injury model than to fall occurrence. |
| Utilization | Counts and recency of primary-care, nursing, emergency, inpatient, home-care, physiotherapy, and rehabilitation contacts in 30/90/365 days; recent discharge; number of distinct care settings. Tables and encounter classifications are **UNKNOWN — TO DISCOVER**. |
| Vitals/labs | Last value, days since measurement, and clinically defensible recent range for BMI, systolic/diastolic pressure, pulse, hemoglobin, sodium, renal function, glucose/HbA1c, and selected other measures only if evidence and coverage justify them. Source fields, units, and reference ranges are **UNKNOWN — TO DISCOVER**. |

Fall-risk-increasing medication classes should be clinically prespecified, including benzodiazepines, Z-drugs, antipsychotics, antidepressants, opioids, antiepileptics, sedating/anticholinergic drugs, selected antihypertensives associated with hypotension, and hypoglycemia-inducing treatments. Local mappings are **UNKNOWN — TO DISCOVER**. Seppala et al.’s STOPPFall consensus (2021, *Age and Ageing*) is a defensible starting framework, but local formulary review is required.

Do not add large, indiscriminate laboratory or diagnosis vocabularies unless temporal validation shows reproducible gain. They increase instability, coding-era dependence, and explainability burden.

# 6. Missingness strategy

Missingness in EHR data is often clinical information: a Mini-Cog test is more likely to exist when someone is suspected of impairment, and frequent blood pressure measurements may indicate illness or high utilization. Conversely, “not assessed” cannot be interpreted as “normal.”

For clinically important variables use a three-part representation:

1. observed value;
2. value-available indicator;
3. days since the value was available.

For repeated assessments, additionally represent count and trajectory. Expire stale measurements using clinically chosen validity windows; do not carry a score forward indefinitely.

Gradient boosting may use native missing-value handling, but explicit availability and recency variables are still useful because they separate “never assessed” from “assessed long ago.” For logistic and penalized regression:

- use deterministic imputation parameters learned only from training data;
- add missingness indicators;
- use category-specific “unknown” levels when appropriate;
- use multiple imputation mainly as a sensitivity analysis, because deployment requires a reproducible single-patient transformation.

No imputation model may use future measurements or the outcome. Training-fold-specific preprocessing must be applied unchanged to validation and test data. Missingness-only models should be evaluated to quantify how much prediction arises from the care process rather than patient state.

# 7. Modeling plan

Build models progressively on identical splits and outcomes:

1. **Minimal clinical baseline:** flexible age, sex, and validated prior-fall history.
2. **Prespecified logistic regression:** a limited clinically selected predictor set, with nonlinear terms where needed.
3. **Elastic-net penalized regression:** all curated features and prespecified interactions.
4. **Gradient-boosted decision trees:** the principal nonlinear challenger.
5. **Optional interpretable challenger:** a generalized additive or explainable boosting model if it can be implemented reproducibly.
6. **Competing-risk survival models:** discrete-time multinomial/pooled models as the main risk-estimation framework, with cause-specific Cox and Fine–Gray sensitivity analyses.

Model selection should be based on temporal validation calibration and clinical utility, not marginal AUROC alone. A simpler calibrated model should win if a complex model does not produce material decision benefit.

Use blocked temporal cross-validation within development data. Tune a small prespecified hyperparameter space with early stopping. Lock preprocessing, feature definitions, and tuning rules before final testing.

Do not balance the validation or test data. Prefer the natural prevalence and proper probabilistic loss. If negative-landmark subsampling is necessary, use known sampling weights and recalibrate. Class weighting or focal loss can be challengers for ranking, but they commonly distort probabilities; van den Goorbergh et al. (2022, *Journal of the American Medical Informatics Association*) describe the harm of routine class-imbalance correction in clinical risk prediction.

Assess calibration-in-the-large and calibration slope. Update only the intercept or baseline hazard when transport requires it. More flexible recalibration—Platt/logistic or isotonic—must be learned on a dedicated calibration set, never on the test set. Van Calster et al. (2019, *BMC Medicine*) emphasize calibration as central to clinical prediction.

Interpretability should include:

- coefficients and confidence intervals for regression;
- clinically constrained effect plots or accumulated-local-effect plots;
- global permutation importance;
- patient-level contribution summaries for the deployed model;
- stability of explanations across resampling and time.

Feature importance and SHAP-like attributions describe associations within a fitted model; **they are not causal effects and do not prove that changing a feature will prevent a fall**.

LLMs should have no role in the primary tabular risk model. A later, separate study could evaluate on-premise extraction of fall mentions from notes for label adjudication or feature extraction, with frozen prompts/models, human validation, and no external data transfer. It must not be allowed to obscure primary reproducibility.

# 8. Ablation plan for nursing assessments

Use the same patients, landmarks, splits, tuning budget, and evaluation code for a nested ablation sequence:

- A: age, sex, and documented prior falls;
- B: routine EHR domains without any nursing assessment variables;
- C: B plus nursing-assessment availability and recency indicators only;
- D: B plus assessment content but excluding explicit historical-fall questions;
- E: B plus explicit historical-fall responses;
- F: B plus all validated nursing fall-risk, mobility, cognition, and home-safety content;
- G: F with each nursing family removed in turn.

This separates the predictive value of being assessed from the value of the answers. It also tests whether apparent gain is almost entirely due to asking about previous falls.

Report paired differences in Brier score, AUROC, AUPRC, calibration, net benefit, and alert-budget capture using patient-clustered bootstrap intervals. Repeat the comparison:

- in the full eligible cohort;
- among patients with any nursing assessment before the landmark;
- among patients with assessments performed within 90 and 365 days;
- among prior fallers and non-fallers;
- by calendar era and district.

Assessment-based subgroup findings must not be generalized to unassessed patients.

# 9. Evaluation metrics and subgroup evaluation

Report metrics with patient-clustered confidence intervals:

- AUROC and time-dependent AUC;
- AUPRC, always accompanied by outcome prevalence;
- C-index for survival models;
- Brier score at 90, 180, and 365 days;
- calibration-in-the-large, observed/expected ratio, calibration slope, and flexible calibration plots;
- cumulative-incidence calibration in the presence of death;
- decision-curve net benefit, following Vickers and Elkin (2006, *Medical Decision Making*).

Decision thresholds must correspond to a plausible intervention, capacity, cost, and harm—not a convenient statistical cut point.

For each monthly cohort, evaluate top **1%, 2%, 5%, 10%, and 20%** risk alert budgets. Report:

- number alerted;
- percentage of subsequent falls captured;
- positive predictive value;
- alerts per captured fall;
- mean predicted versus observed risk among those alerted;
- distribution of alerts across clinics and subgroups;
- turnover and persistence of the alert list from month to month.

Do not call alerts per captured fall a number-needed-to-treat; treatment effectiveness is not yet known.

Mandatory subgroups are age 65–74, 75–84, and 85+, and sex. Additional prespecified groups should include prior fall status, long-term-care/institutional status if discovered, bed/wheelchair status, district, socioeconomic strata if valid, and major language/ethnocultural groups only after governance review. Report discrimination and calibration, not only event rates. Small groups require uncertainty intervals and possibly pooled hierarchical estimates rather than unstable rankings.

# 10. Validation strategy

Use chronological partitions whose dates are chosen after the data-availability audit but before modeling:

1. development/training period;
2. later calibration/validation period;
3. latest untouched test period.

Training landmarks must have fully matured outcome windows before the next partition begins. An outcome episode must not contribute to both training and validation labels through overlapping landmark windows across a boundary; use an embargo or stop generating training landmarks early enough to prevent this.

Within development, all snapshots from a patient must remain in the same fold. No randomly split snapshots.

The final temporal test may appropriately contain continuing Meuhedet members previously present in historical training because this matches deployment, but the model must not use patient identifiers or memorized target encodings. Report a second, stricter analysis among patients absent from model development to test transport to newly observed members.

Perform internal–external validation by leaving out one district at a time. Clinic-level leave-one-site-out validation is desirable only when clinics have enough events; otherwise use clinic-stratified calibration and hierarchical uncertainty. District and clinic identifiers and organizational history are **UNKNOWN — TO DISCOVER**.

After retrospective locking, run silent prospective validation: generate monthly scores without exposing them to clinicians, wait for the full 180-day horizon plus documentation lag, and assess calibration, alert burden, drift, subgroup behavior, and operational data failures. No model update is allowed using the silent-test outcomes before the analysis is signed off.

# 11. Sample-size and power reasoning

Do not use “10 events per variable.” Apply Riley and colleagues’ modern criteria for prediction-model development: control global shrinkage, optimism in explained variation, and precision of the overall risk estimate (Riley et al., 2019, *Statistics in Medicine*; Riley et al., 2020, *BMJ*).

For each candidate model, the BI discovery phase must supply:

1. unique eligible patients;
2. total monthly landmarks and landmarks per patient;
3. distinct patients with a fall;
4. Tier 1 and Tier 1+2 fall counts at 90, 180, and 365 days;
5. cumulative incidence at each horizon;
6. deaths before fall, deaths after fall, disenrollments, and administrative censoring;
7. mean and distribution of follow-up;
8. recurrent-event distribution;
9. the number of candidate predictor parameters after splines, categories, and interactions—not merely the number of source variables;
10. expected Cox–Snell \(R^2\), estimated conservatively from a small prespecified baseline or relevant literature;
11. event and non-event counts by calendar period, district, clinic, age band, sex, prior-fall status, and assessment availability;
12. within-person numbers of snapshots and outcomes, to quantify the loss of effective information from clustering.

Use `pmsampsize`-type calculations for binary and survival models. Because standard formulas assume independent observations, repeat calculations conservatively at the unique-patient level and perform simulation using the observed repeated-landmark structure.

The untouched test set should be sized from desired precision for calibration-in-the-large, calibration slope, AUC, Brier score, subgroup calibration, and alert-budget PPV—not from a universal “100 events and 100 non-events” rule. Tailored external-validation sample-size methods described by Riley et al. (2021, *Statistics in Medicine*) are preferable. If required precision cannot be reached, reduce model complexity, extend accrual, or narrow claims; do not proceed with an underpowered opaque model.

# 12. Leakage tests and hard data-quality gates

## Leakage and negative-control tests

- Verify every feature’s clinical time and database-availability time. Hospital diagnoses posted after discharge must not appear in an earlier snapshot because the final warehouse contains them.
- Remove same-day and post-index information unless availability ordering is provable.
- Search features for fall codes, injury treatments, referrals, or care pathways created after the outcome.
- Confirm zero patient overlap across grouped development folds.
- Confirm no outcome episode crosses split boundaries through overlapping horizon labels.
- Compare grouped versus randomly split snapshots; a large random-split gain is a warning for patient memorization.
- Fit models to permuted patient-level outcomes; performance should return to chance.
- Use deliberately future-shifted variables as a leakage detector in a sandbox audit, never in the real model.
- Repeat analyses excluding the final 7, 14, and 30 pre-index days to test whether prediction depends almost entirely on an already unfolding fall-related episode.
- Fit an assessment-availability-only model to quantify healthcare-process prediction.
- Check whether code-entry batches, clinic identifiers, or warehouse migration dates dominate importance.
- Manually trace a sample of high-risk predictions from raw timestamped records through the feature pipeline.

## Gates that stop training

Training must stop if any of the following remains unresolved:

- external-cause code normalization or code-system eras cannot be established;
- the primary label lacks acceptable adjudicated specificity; I would prespecify overall Tier 1 PPV of at least 0.80 with no major source/era showing clearly unacceptable performance;
- credible event dates cannot be distinguished from follow-up, history, or sequela dates;
- death and membership censoring cannot be measured;
- historical backfill makes it impossible to reconstruct what was known at the landmark;
- retained features show any unresolved post-index leakage;
- patient overlap exists across grouped folds;
- unexplained outcome-rate discontinuities coincide with source or coding transitions;
- unit changes or response-category changes cannot be harmonized;
- the Riley-based minimum sample-size criteria are not met;
- key subgroups or sites have insufficient label quality for claimed generalizability.

If the PPV gate fails but coding is still useful, the project may be reframed explicitly as prediction of a claims-coded fall proxy. It must not silently retain the stronger clinical claim.

# 13. Baselines against existing clinical tools

The available fields do not currently permit defensible reconstruction of a published multicomponent tool.

Defensible baselines are:

- field 2428, the clinician’s overall fall-risk assessment, evaluated exactly as stored and only when available before the landmark;
- individual documented prior-fall, instability, fear-of-falling, gait, and balance questions;
- a simple age/sex/prior-fall model;
- Mini-Cog and MMSE as cognitive predictors, not as fall-risk scores.

A three-question screen resembling STEADI may eventually be possible from prior fall, unsteadiness, and fear-of-falling questions. However, it should be called **STEADI-like**, not STEADI, unless exact question wording, timing, response coding, and scoring match the published process. STEADI was described by Stevens and Phelan (2013, *Frontiers in Public Health*).

The Timed Up and Go test can be reconstructed only if treatment 999.79 contains a true elapsed time, standardized protocol, units, completion status, and historically stable coding—all **UNKNOWN — TO DISCOVER**. Gait-item presence is not a TUG result. Podsiadlo and Richardson introduced the TUG in 1991 in *JAGS*; its stand-alone fall-prediction performance is limited and setting-dependent, as reviewed by Barry et al. (2014, *BMC Geriatrics*).

Do not claim to reconstruct Morse, STRATIFY, Hendrich II, Downton, or another named score from superficially similar fields. Several were developed for inpatient settings, and missing or differently defined components invalidate the score.

# 14. Deployment-study staging

Use five stages:

1. **Retrospective development:** label validation, model development, ablation, and internal validation.
2. **Locked temporal validation:** untouched later-period evaluation and district internal–external validation.
3. **Silent prospective validation:** monthly scoring without clinician display; confirm data latency, calibration, alert capacity, drift, and subgroup safety.
4. **Usability and workflow study:** show clinicians calibrated absolute risk, principal non-causal contributors, recent fall history, and actionable assessment gaps. Measure comprehension, alert fatigue, response time, and missed workflow steps.
5. **Controlled implementation:** preferably cluster-randomized or stepped-wedge by clinic if contamination is likely. Compare usual care with model-triggered, protocolized fall-prevention assessment.

The downstream intervention must be specified before choosing a threshold. It could include confirmatory fall history, medication review, strength/balance exercise referral, gait and assistive-device assessment, orthostatic evaluation, vision review, and home-safety assessment, consistent with the World Falls Guidelines (Montero-Odasso et al., 2022, *Age and Ageing*). Exercise reduces falls in community-dwelling older adults (Sherrington et al., 2019, *Cochrane Database of Systematic Reviews*), but a risk alert by itself has no demonstrated benefit.

The implementation study should measure actual patient-reported falls where possible, injurious falls, documented falls, intervention uptake, adverse effects, workload, equity, and cost. It must test the whole care pathway, not merely whether clinicians open an alert.

# 15. Privacy, governance, and reproducibility

All identifiable processing should remain inside Meuhedet’s controlled environment. Required controls include:

- pseudonymous analytic identifiers;
- least-privilege, role-based access;
- separation of the linkage key from the modeling workspace;
- data minimization and documented purpose limitation;
- encrypted storage and transfer inside approved infrastructure;
- query and access logging;
- suppression of small aggregate cells;
- privacy/security review of exported coefficients, plots, and model artifacts;
- Helsinki/IRB, legal, privacy, and information-security approvals under applicable Israeli requirements;
- explicit governance for sensitive demographic variables and subgroup reporting.

Reproducibility artifacts should include:

- protocol and statistical-analysis plan;
- cohort and outcome diagrams;
- adjudication manual and code lists;
- source-to-concept data dictionary;
- bitemporal timestamp rules;
- executable cohort, label, and feature specifications;
- immutable data-extract identifier and table-version manifest;
- split manifest;
- dependency-locked software environment;
- random seeds and configuration files;
- unit, integration, timestamp, and leakage tests;
- model card with intended use, exclusions, subgroup results, threshold logic, and failure modes;
- calibration and monitoring plan;
- versioned change-control and retirement policy.

Report according to TRIPOD (Collins et al., 2015, *Annals of Internal Medicine*) and assess bias with PROBAST (Wolff et al., 2019, *Annals of Internal Medicine*). Later early-stage clinical deployment should also follow DECIDE-AI principles (Vasey et al., 2022, *Nature Medicine*).

# 16. First discovery data request to the BI team

This should be an aggregate/schema request, not a modeling extract.

1. **Schema and semantics**
   - Metadata for `[Meuhedet_DWH].[Dims].[Dim_Customer_Details]`.
   - Full dictionaries and date semantics for diagnoses, encounters, emergency care, hospitalizations, death, enrollment, medications, dispensing, laboratories, vitals, referrals, rehabilitation, home care, and providers: all table and column names are **UNKNOWN — TO DISCOVER**.
   - Effective dates, loading dates, update cadence, backfill behavior, and historical migrations.

2. **Population flow**
   - Monthly counts of unique active members aged 65+, by exact age/year, candidate age band, sex, district, clinic, observable look-back, and membership duration.
   - Counts excluded for death, insufficient look-back, current hospitalization, and unavailable follow-up.
   - Death and disenrollment counts within 90/180/365 days.

3. **Fall-code inventory**
   - Counts of every raw diagnosis value matching or surrounding `719.7`, `880.9`, `883.9`, `884.9`, `886.9`, `888`, `929.3`, and `E987`, including prefixed/unprefixed, padded, and decimal variants.
   - Code description, code system, diagnosis role, source, encounter type, year/month, event/service date, posting date, and historical/sequela flags.
   - Counts of all other dictionary descriptions containing concepts such as fall, fallen, slip, trip, or accidental fall, without assuming they are valid outcomes.
   - Yearly rates per 1,000 eligible members and change-point summaries.

4. **Episode structure and source overlap**
   - Number of unique persons and records by code/source/year.
   - Distribution of multiple fall-coded records within 0, 1, 3, 7, 14, 30, and 90 days.
   - Pairwise source overlaps on the same day and within 3/7/14 days.
   - Counts with same-episode emergency, hospitalization, fracture/head-injury, imaging, or follow-up records; relevant sources are **UNKNOWN — TO DISCOVER**.

5. **Nursing assessments**
   - All fields, descriptions, units, response categories, first/last use dates, and missingness for treatments 999.52, 999.79, 999.139, 999.175, 999.102, and 999.130.
   - Explicit discovery of every 999.79 field, including possible total, time, hidden, derived, or completion fields.
   - Monthly counts of assessed patients and assessments, value distributions, duplicated records, invalid values, and category changes.
   - Coverage within 30/90/365 days before each candidate landmark.
   - For candidate falls, aggregate percentages with an assessment 30/90/180/365 days before the event.
   - Among later “fell in previous year” responses, the percentage with a qualifying coded event in the preceding year, without assigning an exact retrospective event date.

6. **Medication feasibility**
   - Available medication dictionaries and coding systems, dispensing versus prescription status, dates, quantity, days supplied, dose, discontinuation, and formulary history—all **UNKNOWN — TO DISCOVER**.
   - Annual coverage and counts of patients with active medications, medication changes, polypharmacy, and candidate fall-risk-increasing classes after pharmacist-approved mapping.

7. **Diagnoses, utilization, vitals, and labs**
   - Annual patient coverage, record counts, date availability, coding systems, units, and missingness for the prespecified domains.
   - Distributions of utilization counts and recent hospitalization/discharge.
   - Unit and reference-range changes for candidate vitals/labs.

8. **Sample-size cells**
   - Unique patients, snapshots, distinct fallers, fall episodes, deaths, censoring, and cumulative incidence at 90/180/365 days by period, district, clinic, age band, sex, prior fall, institutional status if available, and nursing-assessment availability.

9. **Chart-review feasibility**
   - Counts available for stratified sampling by candidate code, source, era, and setting.
   - Confirmation that authorized reviewers can inspect source documentation inside Meuhedet and record a minimal adjudication form without exporting identifiable content.

# 17. Top 10 design decisions

| Decision | Confidence | Strongest argument against |
|---|---:|---|
| 1. Name the target “acute medically documented unintentional fall,” not all falls | 98% | It predicts both fall risk and the probability of seeking or receiving documented care. |
| 2. Require source- and era-stratified clinical label validation before training | 99% | Chart adjudication is slow and may still miss unreported falls. |
| 3. Use time to first fall after each landmark, while retaining prior fallers | 95% | Recurrent fallers may represent a distinct process better handled by a separate model. |
| 4. Use standardized monthly rolling landmarks | 90% | Overlapping horizons create substantial correlation and computational burden. |
| 5. Retain 180 days as the provisional primary horizon, with 90 and 365 days secondary | 78% | A 365-day horizon may align better with annual screening and provide more events; 90 days may be more actionable. |
| 6. Make competing-risk cumulative incidence the main estimand | 91% | Discrete-time competing-risk implementation is harder to explain and maintain than ordinary binary classification. |
| 7. Require 365 days of look-back but test shorter-history members separately | 87% | It excludes new members and may reduce transportability to a changing membership population. |
| 8. Include institutionalized, wheelchair-bound, and bed-bound patients, then stratify | 80% | Their exposure and intervention pathways may be so different that one model is clinically incoherent. |
| 9. Compare penalized regression and gradient boosting, selecting on calibration and utility | 95% | A single prespecified regression model would be easier to validate and publish with less researcher flexibility. |
| 10. Do not deploy until a silent prospective phase and an actionable intervention pathway are complete | 98% | The delay may postpone benefit if a simpler high-risk rule could already support prevention. |