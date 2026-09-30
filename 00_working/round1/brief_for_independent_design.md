<role>
You are an independent co-investigator (senior clinical prediction-model methodologist + healthcare ML engineer) on a real clinical research project at Meuhedet, an Israeli health fund (HMO). You have NOT seen any other investigator's proposal. Produce your OWN independent design. Do not run shell commands and do not read files; answer from reasoning and your knowledge of the peer-reviewed literature.
</role>

<task>
Design the strongest scientifically defensible, technically implementable research program for:
PREDICTION OF FUTURE FALLS AMONG ADULTS AGED 65+ USING MEUHEDET ELECTRONIC HEALTH RECORD (EHR) DATA.
Scope is ONLY falls. Do not expand into a generic clinical-AI platform.

The work will later be reviewed by senior clinicians, epidemiologists, biostatisticians, ML researchers, healthcare data scientists, privacy/security teams, management, and an academic peer-review committee. Optimize for: scientific validity; prevention of leakage and bias; reproducibility; clinical usefulness; interpretability; calibration; realistic implementation; privacy/governance; explainability to non-technical executives; eventual publishability.
</task>

<context_facts>
Population source identified: [Meuhedet_DWH].[Dims].[Dim_Customer_Details]. Initial population: adults aged 65+. Age bands 65–74 / 75–84 / 85+ are candidates for SUBGROUP analysis only; do not assume they are the primary age representation.

Candidate (NOT final) primary question: "Given everything known about a Meuhedet patient aged 65+ up to an index date T, what is the probability of a medically documented fall during the following 180 days?" Candidate secondary horizons: 90 and 365 days. You must explicitly challenge this definition and recommend the strongest design.

Diagnosis codes so far identified as "fall-related" in Meuhedet: 719.7, 880.9, 883.9, 884.9, 886.9, 888, 929.3, E987. Concern: several look like ICD-9-CM external-cause codes with the "E" prefix stripped (E880.x, E883.x, E884.x, E886.x, E888.x, E929.3); 719.7 appears to be "difficulty in walking" (a gait symptom, not a fall); E987 is "falling from high place, undetermined intent". It is UNKNOWN how Meuhedet stores/normalizes ICD-9 external-cause codes, whether ICD-10 or local codes coexist, transition dates, diagnosis source/encounter type, and whether a fall diagnosis date is the event date, a follow-up, a historical condition, or a sequela. Nothing about the mapping may be assumed.

Nursing questionnaire data: fact table [Meuhedet_DWH].[Medicine_DWH].[Fact_Nurse_Questionnaire] joined to [Medicine_DWH].[Dim_Treatments] and [Medicine_DWH].[Dim_Nurse_Questionnaire_Fields]. Assessments identified (treatment code — field code — meaning):
- 999.52 Fall Risk Assessment: 1574 fell during last year; 1575 instability/dizziness; 1576 fear of falling; 2426 number of falls last year; 2427 required treatment after fall; 2428 overall fall-risk assessment; 3529 bed/wheelchair bound.
- 999.79 Get Up and Go: 933 balance impairment; 935 shuffling gait; 936 hesitant gait; 938 previous falls; 939 fear of falling; 943 muscle weakness; 944 cognitive impairment; 952 lack of hearing/visual aids; 954 visual impairment; 957 orthostatic hypotension; 958 polypharmacy; 959 concern about falls; 965 muscle weakness; 966 cognitive impairment. No aggregate score identified yet; ALL fields of 999.79 must still be discovered (total score? TUG seconds? categorical gait performance? hidden/derived fields? historical result-coding changes?).
- 999.139 Home Safety: 1574 fall in previous year; 2439 number of falls; 2440 location of fall; 2441 instability/dizziness; 2442 fear of falling; 2480 assistive device.
- 999.175 Rehabilitation/Mobility Device Assessment: 3609 vision; 3615 cognitive state; 3629 mobility; 3630 supine-to-sitting transfer; 3631 sit-to-stand transfer; 3633 cane/crutches; 3634 walker/rollator; 3637 wheelchair; 3639 wheelchair mobility. Observed categorical values, e.g. cognition: oriented/memory preserved, mild impairment, marked impairment, advanced dementia; mobility: walks independently, walks independently with device, walks with partial assistance, wheelchair bound, bed bound; vision: adequate without glasses, adequate with glasses, limited, blindness.
- 999.102 MMSE: 1439 final score (completeness inconsistent).
- 999.130 Mini-Cog: 2070 total score, observed range 0–5 (currently the clearest structured cognitive score).

Other DWH domains believed to exist (table/column names UNKNOWN): demographics, diagnoses, physician visits, hospitalizations, laboratory tests, vital signs, BMI, blood pressure, medications, treatments/performed treatments, referrals, clinical assessment, physical examination, lifestyle, providers, events, hospital departments. Use them ONLY where they may materially improve fall prediction.

Constraints: no identifiable patient data leaves Meuhedet; you work from schema/metadata/aggregates. Do NOT invent Meuhedet table or column names — write "UNKNOWN — TO DISCOVER". Do NOT invent clinical codes or performance numbers. Do not treat retrospective questionnaire answers as exact event dates. Do not use post-outcome information. Do not default to deep learning or LLMs as the tabular risk model.
</context_facts>

<required_output>
Write a complete independent design covering, in this order, with numbered headings:
1. Primary outcome definition (what counts as a fall event; evidence tiers; how to validate the label before modeling; handling of historical/self-reported falls, sequelae, ambiguous codes; date assignment; de-duplication; first fall vs any fall vs recurrent vs serious/injurious fall; and whether to have separate outcomes).
2. Cohort definition (age; membership/eligibility; minimum look-back; deaths; disenrollment; insufficient follow-up; institutionalized/long-term-care, bed-bound, wheelchair-bound patients; prior fallers).
3. Index-date / observation-unit strategy — compare fixed annual index date, rolling monthly landmarks, visit-based, assessment-based, patient-month/quarter; address within-patient correlation, leakage, compute cost, deployment realism; recommend ONE primary design and justify.
4. Observation (look-back) windows and prediction horizon(s) — challenge 180 days; recommend primary and secondary horizons with reasoning; decide whether binary, time-to-event, or both is the main analysis; how to handle the competing risk of death.
5. Feature taxonomy by domain (demographics; fall history; nursing assessments incl. last/worst/count/days-since/trajectory/missing-indicator; mobility/gait/balance; cognition; vision/hearing; orthostatic/dizziness/syncope; medications incl. specific fall-risk-increasing drug classes, counts, recency of change; diagnoses; utilization; vitals/labs) — keep only features with clinical/evidence justification; state lookback windows.
6. Missingness strategy (informative missingness; value/value_available/days_since; native boosting handling vs indicators vs imputation for linear models; leakage from future measurements).
7. Modeling plan — a progressive benchmark from a minimal clinical baseline (age, sex, prior fall) through logistic regression, penalized regression, gradient boosting, optional challenger, and survival/competing-risk models; class-imbalance handling; calibration approach; hyperparameter strategy; interpretability (and the statement that importance is not causality); the role, if any, of LLMs.
8. Ablation plan to quantify the incremental value of the structured nursing fall-risk/mobility/cognition assessments beyond routine EHR data.
9. Evaluation metrics (discrimination, calibration, clinical utility via decision curves, an "alert budget" analysis at top 1/2/5/10/20% risk) and subgroup evaluation (age bands, sex, others).
10. Validation strategy (development/validation/final untouched test; temporal validation; internal–external validation by district/clinic; prevention of patient leakage across snapshots; silent prospective validation).
11. Sample-size/power reasoning using modern prediction-model methodology (not just events-per-variable) and the exact counts needed from Meuhedet to compute it.
12. Leakage/negative-control tests and hard data-quality gates that must STOP training.
13. Baselines against existing clinical tools (which can be reconstructed defensibly from the available assessments, which cannot).
14. Deployment-study staging (retrospective → temporal → silent prospective → usability → controlled implementation) and whether the downstream intervention is actionable.
15. Privacy/governance and reproducibility artifacts.
16. The first discovery data request (counts/queries, not a final dataset) needed from the BI team to validate population, fall coding, source overlap, assessment coverage, assessment-before-fall coverage, medications, diagnoses, utilization, follow-up.
17. Your top 10 design decisions, each with a confidence score (0–100%) and the strongest argument AGAINST your own choice.
</required_output>

<compact_output_contract>
Markdown. Be concrete and specific; prefer decisions over option menus. Where evidence exists, name the source (author, year, journal) — do not invent citations; if unsure of a citation, say "citation to verify". Label every unknown Meuhedet element "UNKNOWN — TO DISCOVER". Target length 3,500–6,000 words.
</compact_output_contract>
