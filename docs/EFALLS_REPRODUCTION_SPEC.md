# eFalls Reproduction Specification

| Field | Value |
|---|---|
| Document | `docs/EFALLS_REPRODUCTION_SPEC.md` |
| Version | 0.4 (revised after adversarial spec review, implementation review and final pipeline review) |
| Date | 2026-09-14 |
| Status | **DRAFT — requires clinical and methodological sign-off** (see §15) |
| Scope | Reproduction of the published eFalls model as the scientific baseline for the Meuhedet fall-risk project |
| Patient data used to write this document | **None.** No Meuhedet, SAIL or Connected Bradford patient-level data were accessed. |

---

## 0. How to read this document

Every statement carries one evidence label.

| Label | Meaning |
|---|---|
| **[PUB]** | Stated in the eFalls publication: Age & Ageing 2024 paper, its supplement, or the NIHR HTA monograph. The source location is given. |
| **[PUB-CONFLICT]** | The publications contradict each other or themselves. The conflict is logged in §9. |
| **[IMPL]** | Stated in the development team's implementation documents (Leeds eFI2/eFI+ implementation instructions) or later author code. These are not peer-reviewed publications. |
| **[INF]** | Inferred by us from published numbers. The reasoning and its confidence are given. |
| **[ASSUME]** | A decision we make because the publications leave it open. Every assumption has an ID (`D-xx`), a sensitivity analysis where meaningful, and an owner for sign-off. |
| **[MEU]** | A Meuhedet-specific element, not part of eFalls. |
| **[NF]** | Searched for and not found in any authoritative public source. |

The implementation must keep four layers strictly separate (§1.2). Configuration files must tag every item with its layer.

Abbreviations:

| Abbreviation | Meaning |
|---|---|
| A&A | Archer et al., Age Ageing 2024;53(3):afae057 |
| S-table/S-box | The supplement to A&A |
| HTA | Archer et al., Health Technol Assess 2026;30(61), doi:10.3310/GJAC1008 |
| LP | Linear predictor |
| EV | External validation |
| IECV | Internal–external cross-validation |
| CITL | Calibration-in-the-large |
| O/E | Observed/expected ratio |
| NB | Net benefit |
| FP | Fractional polynomial |
| CB | Connected Bradford |

---

## 1. Purpose, scope and non-negotiable separations

### 1.1 Purpose

1. Specify, as exactly as public material allows, the published eFalls prediction model and the process used to build and validate it.
2. Record every point where the publications are silent, ambiguous or contradictory, together with the decision we take and its justification.
3. Define what the Meuhedet implementation must do to run two distinct eFalls experiments that must never be confused:
   - **Experiment A — `efalls_published_scoring`.** Apply the published eFalls equation, with fixed published coefficients, to Meuhedet data after mapping Meuhedet data to the eFalls representation. *Question: does the original model transport to Meuhedet?* Nothing is estimated from Meuhedet outcomes in the primary analysis. An optional recalibration on the Meuhedet validation split is reported separately and labelled as such.
   - **Experiment B — `efalls_retrained_lasso`.** Use the same predictor definitions and the same learning process (FP selection, then LASSO logistic regression with 10-fold CV λ selection) to fit new coefficients on the Meuhedet development split. *Question: how much does performance change when the eFalls feature space is fitted locally?*

### 1.2 The four layers

| Layer | Content | May change without re-approval? |
|---|---|---|
| **L1 Exact published eFalls** | Items labelled [PUB] with no conflict: outcome code list, coefficient values, transformations, missing-data strategy, learning algorithm | **Never.** Any change is a new, differently named experiment. |
| **L2 Documented assumptions** | [ASSUME] decisions `D-xx` needed because L1 is silent or contradictory | Only through this document's change log plus sign-off |
| **L3a Meuhedet mapping & mechanics** | [MEU] items unavoidable even for the baseline: code/field mapping, as-of cohort reconstruction, validation design (D-19), availability rules | Versioned mapping/config; signed off before outcome linkage |
| **L3b Meuhedet additional predictors** | [MEU] new predictors (Mini-Cog, nurse assessments, mobility, …) — Enhanced phase only | Any use creates a differently named experiment; never in L1/L2 experiments |
| **L4 Alternative models** | Standard/elastic-net logistic regression, random forest, gradient boosting, … | Free to evolve; always reported separately from L1/L2 |

Rule: an experiment is labelled "eFalls baseline" only if it uses L1 + L2 + L3a. Any L3b predictor, alternative estimator, modified transformation, or incomplete feature coverage beyond the M-11 threshold makes it a *different* experiment with a different name (e.g. `efalls_partial_scoring`).

Every artifact and report header carries: experiment id and kind, layer set, config hash, sex parameterisation (Experiment A), feature-coverage status, and the synthetic-data watermark where applicable. L4 models are evaluated on the identical partitions, cohort and outcome as the eFalls experiments (D-19).

---

## 2. Phase 1 environment findings

| Item | Finding |
|---|---|
| Repository | `<project root>`. **No existing code, data, models or tests.** Only design-brainstorming notes in `00_working/` (a generic falls-prediction design brief and an independent LLM-generated design). Not a git repository. |
| Python | Homebrew Python 3.11.15, 3.13.13 and 3.14.7 are present. System interpreters have almost no scientific packages. A project virtual environment `.venv` was created on **Python 3.13.13**. |
| Installed (project `.venv`) | numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0, matplotlib 3.11.2, PyYAML 6.0.3, pyarrow 25.0.1, pytest 9.1.1 |
| Not available / not added | XGBoost, LightGBM, SHAP. Per the project brief (no new dependency merely because it is fashionable) they are **not** introduced. Gradient boosting uses scikit-learn `HistGradientBoostingClassifier`; SHAP is optional and disabled. |
| Data | **No Meuhedet data are available to this workstream.** No DWH connection exists, and the ML code must never create one (dataset boundary: `docs/ARCHITECTURE.md` §1). |
| Reproducible environment | Pinned dependency lock files `requirements.lock` and `requirements-build.lock` (hash-pinned verified versions) and recreate command in `README.md`. |
| Consequence | Everything beyond published-equation arithmetic is demonstrated on **deterministic synthetic fixtures labelled as software tests**. No scientific result may be derived from them. |

---

## 3. Sources and asset availability

### 3.1 Authoritative sources used

| ID | Source | Licence | Local verbatim copy (research workspace) |
|---|---|---|---|
| S1 | Archer L, Relton SD, Akbari A, et al. *Development and external validation of the eFalls tool…* Age Ageing 2024;53(3):afae057. doi:10.1093/ageing/afae057. PMC10960070 | CC BY 4.0 | JATS XML → `fulltext.md` |
| S2 | S1 supplementary DOCX `aa-23-2211-file002_afae057.docx`: Tables S2.1–S3.5, Boxes S3.1–S3.2, Figures S3.1–S3.18; equations stored as OMML | CC BY 4.0 | `supp.md` plus OMML extraction |
| S3 | Archer L … Riley RD, Clegg A. *Development and evaluation of the electronic frailty index+ (eFI+) tool…* Health Technol Assess 2026;30(61):1–154. doi:10.3310/GJAC1008. PMID 42574037. PMC13478932. NIHR award NIHR127905; NCT04113174 | CC BY 4.0 | journalslibrary HTML and NCBI Bookshelf tables (MathML formulas present only on NCBI Bookshelf) |
| S4 | eFI+ protocol V20 (2019-09-10) and V21 (2021-11-03), NIHR Journals Library | public | text extractions |
| S5 | Leeds eFI2/eFI+ implementation instructions v1.9 / v1.10 / v1.13 and MHRA documentation (GitHub `KateBest/eFI2plus`; instruction files deleted from HEAD 2026-02-26, retrievable from history) | **no licence file** — facts only, not reproduced | text extractions |
| S6 | Chen T … Relton S, Lim S, Clegg A. *Extending eFall risk prediction to working-age adults…* Sci Rep 2026. doi:10.1038/s41598-026-51298-0 (supplement Table S1 = full coefficients) | CC BY 4.0 | text extraction |
| S7 | NIHR ARC Greater Manchester. *Case Finding for Falls Prevention — 'eFalls' pilot study evaluation report* (2026) | public report | text extraction |
| S8 | Best K … Clegg A. eFI2. Age Ageing 2025;54(4):afaf077. PMC11957239; Appendix 1 (79 candidates with time/age rules) | CC BY-NC 4.0 | text |
| S9 | Clegg A, et al. eFI. Age Ageing 2016;45:353–60. PMC4846793 | CC BY-NC 4.0 | text |
| S10 | DynAIRx `Baseline2_Codelist.csv` (third-party copy of the eFI+/eFI2 SNOMED CT/CTV3 list with time/age rules), https://github.com/DynAIRx/GCAF_DynAIRx; also in BMC Med Res Methodol 2025;25:138 supplement | repo BSD-3; **content IP unclear** (University of Leeds) | local CSV (not to be copied into this repository) |
| S11 | StataCorp. *Stata 17 [LASSO] and [R] fp/mfp manuals* (behaviour of `lasso logit`, `fp`, `mfp`); verified numerically against Stata's published example output | © StataCorp — paraphrased | — |
| S12 | Methodological references cited by eFalls: Riley 2016 BMJ; Snell 2018 SMMR; Riley 2019/2021 Stat Med; Van Calster 2019 BMC Med; Vickers 2016/2019; Riley & Collins 2023 Biom J; Röver 2015 (HKSJ); Steyerberg & Harrell 2016; Debray 2013 | various | text |
| S13 | NICE NG249 Evidence Review D (2025, NBK616085) | public | — |
| S14 | CDC/NCHS ICD-9-CM tabular and 2018 GEMs; NHSBSA BNF code structure; WHO ATC/DDD guidelines 2026; WHO BMI classification; UK CMO alcohol guidance; OHID AUDIT-C | public | — |
| S15 | Hershkowitz Sikron F, et al. Meuhedet electronic frailty index (MEFI). Aging 2024. PMC11552639 (Meuhedet locally-adapted ICD-9 and ATC deficit definitions) | CC BY | — |
| S16 | Marom et al. 2026, PMC13224401 (Meuhedet psychiatric extraction: ICD-9 and ICD-10 codes observed; medications from purchase records) — the only public hint of Meuhedet code-system coexistence | CC BY | — |
| S17 | Atkinson MD, et al. Int J Popul Data Sci/SAIL smoking algorithm 2017 (PMC5217540); OpenSAFELY smoking study definitions — UK EHR smoking-status derivation conventions | public | — |

### 3.2 Availability of official eFalls assets

| Asset | Status | Evidence / route |
|---|---|---|
| Raw development data (SAIL, Wales) | **NOT publicly available** | SAIL Information Governance Review Panel application; remote Trusted Research Environment access only |
| Raw external-validation data (Connected Bradford) | **NOT publicly available** | Expression of interest plus board approval; secure virtual environment; fee |
| Individual participant data via registry | Not shared | NCT04113174: IPD sharing = No |
| Model equation (coefficients) | **Public**, "available for research use"; commercial/NHS/supplier use requires a licence | S1 data statement; S3 data-sharing statement |
| Coefficient table | Public: S2 Table S3.2 = S3 Table 15 = S6 Table S1 (numerically identical, 80/80 rows) | verified |
| Worked-example boxes | Public: S2 Box S3.1/S3.2 and S3 Tables 16/34 — **formula and example arithmetic identical; text differs**: S3 Table 16 says "standard BMI cut-offs" (S2 says "obese if BMI ≥ 40"); S3 Table 34 points to "Table 35 for values" (the rejected ×1.21 coefficients) while its MathML gives expit(−0.423 + 1.25·LP); the BNF sub-sub-chapter footnote exists in S2 only | verified |
| Outcome ICD-10 code list | Public: S2 Table S2.3 = S3 Appendix 1 (identical, 31 rows) | verified |
| Predictor code lists (≈7,500 SNOMED / 9,500 CTV3 / 7,500 Read v2 codes for 80 candidates) | **On request only** from the corresponding author (A. Clegg) | S1/S3 data statements. A third-party proxy (S10) is public but unofficial, with IP unclear and known data-quality defects |
| "Technical specification document" (lists code-list sources) | Mentioned once in S3 (l.407); availability not stated [NF] | — |
| Analysis code (Stata 17 development; R validation) | **Not available.** TRIPOD item 18: "Not yet available. To be included at publication" — never released | S2 Table S1.1 |
| CV seed, fold assignment, λ grid | Not reported. λ* = 0.000123 is reported (HTA only) | S3 Ch. 5 |
| Later author code touching eFalls | `sdrelton/PREDICT` (drift/recalibration on CB). Coefficients are loaded from a non-public JSON; its polypharmacy coding differs from the publication (`ln(P+1)/10`, 3 months) | S5-adjacent; **not** a reproduction reference |
| Online calculator / model object | Not found | — |
| Errata / corrigenda | **None** for S1 or S3 (Crossref, PubMed, Europe PMC, OUP, NIHR checked 2026-09-14) | — |

**Conclusion:** eFalls patient-level data are unavailable. We therefore:
- (i) reproduce everything reproducible from the publications: equation arithmetic, worked examples, sample-size calculations, methodology;
- (ii) never fabricate SAIL/CB results;
- (iii) build an external-dataset adapter so the identical evaluation pipeline can run on eFalls-compatible data if access is obtained;
- (iv) use deterministic synthetic fixtures only for software testing.

---

## 4. Published prediction task [PUB]

| Element | Published definition | Source |
|---|---|---|
| Population | Adults aged ≥ 65 years with linked data, registered with a participating general practice on the index date. Implemented inclusion: aged ≥ 65 plus valid recorded sex; **0 exclusions** (SAIL: 3,100,549 registered → 660,417 analysed; 455 practices) | S1 Methods; S3 l.360, Fig. 1 |
| Index (baseline) date | SAIL development: **1 April 2018**. CB external validation: **1 January 2019** (A&A). *HTA text elsewhere gives CB "as of 1 April 2018", n = 88,947 — see E-10* | S1; S3 |
| Look-back | "complete primary care EHR, to first registration, and linked data". No minimum registration length stated [NF] | S3 l.360 |
| Age | Age in years. In SAIL derived from **week of birth** (approximate). No upper age limit stated [NF] — see D-06 | S3 l.556; S2 Fig. S3.1 |
| Outcome | **Any (one or more) ED attendance or hospital admission for a fall or fracture within 12 months** of baseline predictor assessment | S1 Outcomes |
| Outcome data | SAIL: Emergency Department Data Set (EDDS) + Patient Episode Database for Wales (PEDW). CB: "linked secondary care data" | S1; S3 |
| Outcome code list | ICD-10: W00, W01, W02, W03, W04, W05, W06, W07, W08, W09, W10, W11, W12, W13, W14, W15, W16, W17, W18, W19, M80, S22, S32, S42, S52, S72, S82, T08, T10, T12, T14.2 | S2 Table S2.3; S3 App. 1 |
| Codes not in the list (observation only) | e.g. S02 (skull/face), S12 (neck), S62 (wrist/hand), S92 (foot), T02, X59, M84.4, M48.5 — authorial intent not stated | [INF] |
| Follow-up | "Outcomes were assessed over a 12-month period, to 31 March 2019" for index 1 April 2018, i.e. **[index, index + 1 year − 1 day]** (index day included) | S3 l.360 |
| Death during follow-up | **Retained in the risk set for the full 12 months** (a death without prior event = non-event). SAIL deaths 25,148 (3.8%) | S1 Discussion; S3 |
| Deregistration / loss to follow-up | Not modelled; "everyone followed for 12 months or until they died" (S1 Discussion). CB: < 2% had no outcome measures and no further events (S3 l.498) | S1; S3 |
| Uncoded outcome | "where a fall/fracture was not coded within 12 months, it was assumed that no fall/fracture occurred" | S1 Missing data |
| Predictor record sources | "complete primary care EHR, to first registration, and linked data" (S3 l.360) vs predictors built by "organising individual EHR SNOMED-CT codes into groups" (S1). Whether linked secondary-care diagnoses fed predictors is not stated → D-22, Q-10 | S1; S3 |
| Care-home residents | Not excluded from eFalls (0 exclusions; the "impossible outcome" exclusion rule applied to other eFI+ models only) | S3 l.498–499, Fig. 1 [INF] |
| Development events | 32,097 / 660,417 (4.9%) | S1 Table 1 |
| EV events | 2,389 / 81,685 (2.9%) (S1 Table 1), 76 practices (S1 Results). HTA Fig. 1 instead shows 84 practices, n = 88,947 (E-10) | S1; S3 |

**Not reported [NF]:**
- Diagnosis positions searched (primary vs any).
- ED dataset coding system and treatment of free-text or non-ICD ED diagnoses.
- Whether admissions were restricted to emergency admissions.
- Whether the index day is included in the outcome window.
- Code-matching rule (3-character prefix vs exact).
- De-duplication across ED and admission records.

These are decided in D-09.

---

## 5. Published predictors and their representation

### 5.1 Candidate set [PUB, with PUB-CONFLICT on the count]

| Statement | Source |
|---|---|
| Candidates = "the 36 components of the eFI" plus "44 additional variables" (= 80) | S1; S3 |
| Table S3.1 lists **72 binary** candidates; **10** are marked as omitted by LASSO: Anxiety, Chronic kidney disease, Dyspnoea, Environment problems, Heart valve disease, Ischaemic heart disease, Problems managing finances, Shopping problems, Toileting problems, Transient ischaemic attack | S2 Table S3.1 |
| Non-binary candidates: age, polypharmacy, sex, BMI category, smoking, alcohol consumption | S2 Table S3.2 |
| 72 + 6 = **78** listed candidates vs 80 stated → E-05 | [INF] |
| "LASSO regression retained **75** predictors" = 75 non-reference parameters (age 1 + polypharmacy 1 + sex 1 + BMI 4 + smoking 1 + alcohol 5 + 62 binary). HTA Ch. 5 says 74 → E-06 | S1; S3 |
| Under Stata `lasso` all-levels factor coding the candidate parameter count is 72 + 1 + 1 + 2 + 5 + 3 + 6 = 90, *consistent with* the "anticipated 90 predictor parameters" of the sample-size calculation. Not diagnostic: eFI2 used the same planning figure for a different model, 80 candidates with reference coding + FP2 also give 90, and the 13,867 row implies 76 | [INF, low–moderate] |
| The 72 binary names match the eFI+/eFI2 code-list groups one-to-one (S10). For 30/32 predictors shared with eFI2, SAIL counts are identical to eFI2 Table 2, so eFalls and eFI2 appear to share one derived SAIL predictor dataset | [INF, moderate–high] |

### 5.2 Binary clinical predictors [PUB]

- Constructed by grouping SNOMED CT / CTV3 / Read v2 codes into 80 candidate variables. Codes were mapped with TRUD tables in both directions (> 99% matched) and clinically validated [S3 l.400–412].
- **Missing rule [PUB]:** "Where individual diagnoses or prescriptions were not recorded for a patient, they were assumed not to be present." Binary predictors therefore take values {0, 1} only; *no missing category*.
- **Look-back / time constraints [PUB-CONFLICT/NF]:**
  - Global rule: complete EHR look-back [S3].
  - The clinical/PPI panels were asked "to review time constraints" and provided "clinical steering on how the time variability of predictor variables should be incorporated", but the resulting rules are not reported [S3 l.324, l.403].
  - The eFI2 paper (same programme, apparently the same SAIL dataset) publishes per-predictor rules: "all codes within 5 years" (†), "some codes within 5 years" (‡), "age ≥ 18" (*), "age ≥ 55" (α), resolution hierarchies (¥).
  - → Decision D-08.
- The full table of all 72 binary candidates, with coefficients, prevalences, origin and the eFI2 proxy rule, is **Appendix B**.

### 5.3 Age [PUB]

| Aspect | Specification |
|---|---|
| Unit | Years (SAIL: derived from week of birth, approximate; whether continuous or completed years is not stated [NF]) → D-06 |
| Functional form | FP selection up to second order, "in the presence of all predictors"; selected form **linear** (FP power 1) [S2 Fig. S3.1] |
| Model term | `Age (years)` × 0.0415506 (no centring, no scaling in the published equation) |
| Missing | Not applicable; valid age is an inclusion criterion |
| Upper limit | Not stated. Figure S3.1 shows data points only for ages 65–95 → D-06 |

### 5.4 Sex [PUB-CONFLICT — critical]

Categories male/female; valid recorded sex is required. **The published parameterisation is contradictory → D-01 (§9.1).**

### 5.5 Polypharmacy [PUB]

| Aspect | Specification | Source |
|---|---|---|
| Definition | "a count of unique drugs prescribed over the **120 days prior to index date** (excluding non-drug chapters of the BNF e.g., bandages)" | S2 Box S3.1 |
| Counting unit | Footnote: "**Unique BNF sub-sub-chapters.** Combinations of > 1 drug from a sub-sub-chapter only counted once towards the total." (Main Box text says "unique drugs"; footnote is more specific) | S2 Box S3.1 footnote |
| Excluded chapters | Not listed; only "bandages" as an example [NF] | — |
| Repeat vs acute prescriptions | Not stated [NF] | — |
| Missing | "Where no prescription was included on a patient's record, it was assumed that no prescription was written" (S3 l.495) → count 0; never missing | S1; S3 |
| Functional form | FP selection up to FP2; selected **natural logarithm** (FP1, power 0) | S2 Fig. S3.1 |
| Model term | **ln((P + 1) / 10) × 0.3296295** | S2 Table S3.2 |
| Origin of `(P+1)/10` [INF, high] | Exactly Stata's automatic FP pre-scaling: non-positive minimum → shift by counting interval (1), then divide by 10^⌊log10(range)⌋ = 10 for a maximum count in [10, 99]. Figure S3.1 shows maximum ≈ 61 | S11 |
| Observed distribution | SAIL median 4 [IQR 0–9]; CB 4 [1–7] | S1 Table 1 |

Conflicts, all handled in D-03:
- The implementation instructions (S5) say *previous 90 days*, BNF 2017-18 sub-sub-chapters.
- eFI2 uses 90 days, "BNF sub-subchapters (level 3)".
- Later author code (`PREDICT`) uses a 3-month window and `ln(P+1)/10`.

**Polypharmacy must not be replaced by a "≥ 5 drugs" flag.** That is the original eFI deficit, not eFalls.

### 5.6 BMI category [PUB, PUB-CONFLICT on cut-points]

| Aspect | Specification |
|---|---|
| Levels | Underweight, Normal, **Overweight (reference)**, Obese, **Missing** (explicit category) |
| Coefficients | Underweight 0.4896735; Normal 0.2394177; Obese −0.0411134; Missing −0.1451981 |
| Cut-points as printed | Box S3.1 and HTA Table 34: "underweight if BMI < 18.5; normal weight if 18.5 ≤ BMI < 24.9; obese if BMI ≥ 40". Overweight is undefined |
| Cut-points elsewhere | HTA Table 16: "standard BMI cut-offs". S5 instructions: < 18.5 / 18.5–24.9 / 25.0–29.9 / ≥ 30, most recent value. S10: "BMI 30+ = obesity" |
| Source record, window, plausibility filter | Not stated in eFalls [NF]. S10 gives a 5-year window for the BMI observable → D-02 |
| Missing | Explicit "Missing" level (missing-indicator approach) |

### 5.7 Smoking [PUB + INF]

| Aspect | Specification |
|---|---|
| Levels in model | **Ex/never (reference)**, Current (0.0684529) |
| Descriptive levels | Never / Ex / Current. Categories sum exactly to the cohort total → **no missing category** |
| Missing | Merged into the reference (S5: "coded as none … if they have no smoking SNOMED CT codes"; eFI2 reference "none, ex, or missing") [IMPL/INF] |
| Derivation | Not stated in eFalls. S10: "Cannot be current and ex smoker"; S5: an ex code followed by a current code → current (most recent status) → D-04 |
| Data-quality note | CB has 16 ex-smokers (0.0%) — an ex-smoker coding failure; no LP effect because ex is part of the reference |

### 5.8 Alcohol consumption [PUB]

| Aspect | Specification |
|---|---|
| Levels | Harmful drinking 0.4164064; Higher risk drinking 0.1549725; **Lower risk drinking (reference)**; Previous higher risk/harmful drinking 0.0849676; Zero alcohol 0.0070124; **Missing** −0.0679367 |
| Missingness | 97.3% missing in SAIL; 81.0% in CB |
| Definitions | Not stated in eFalls [NF]. S10 (5-year window): units/week 0 = zero; 1–20 = lower; 21–48 = higher; ≥ 49 = harmful; units/day × 7. S5: highest category over 5 years → D-05 |
| HTA label error | HTA tables print "Higher-risk drinking" twice; the second (reference) row carries the lower-risk counts (E-08) |

### 5.9 Missing-data strategy (verbatim principle) [PUB]

> "Missing data were treated the same way in both model development and validation data. Where individual diagnoses or prescriptions were not recorded for a patient, they were assumed not to be present. Similarly, where a fall/fracture was not coded within 12 months, it was assumed that no fall/fracture occurred. For other predictors, this study employs missing indicators, with missing observations allocated to 'missing' groups for categorical variables…" — S1

Implementation consequences:

| Variable type | Rule |
|---|---|
| Binary clinical predictors | Absent ⇒ 0. A null value in the modelling dataset is a **schema violation** (the data-engineering layer must materialise 0/1) |
| Polypharmacy | No prescriptions ⇒ 0. Null is a schema violation |
| Age, sex | Required; null is a schema violation (inclusion criteria) |
| BMI | Null ⇒ `missing` level |
| Alcohol | Null ⇒ `missing` level |
| Smoking | Null ⇒ reference level `ex_never` (D-04) |
| Imputation | **None.** No multiple imputation (the protocol planned it; the published model did not use it) |

### 5.10 Published eFalls equation [PUB, subject to D-01]

p = 1 / (1 + exp(−LP)),
LP = intercept + sex term + 0.0415506·Age + 0.3296295·ln((P+1)/10) + Σ categorical-level coefficients + Σ binary coefficients.

All 75 coefficients are in Appendix A (non-binary) and Appendix B (binary). The machine-readable copy is `configs/models/efalls_published.yaml`.

---

## 6. Published learning process (Experiment B must follow it)

### 6.1 Continuous predictors: fractional polynomials [PUB + INF]

| Aspect | Specification | Label |
|---|---|---|
| Procedure | "second-order fractional polynomials, with functional forms chosen in the presence of all predictors. Transformed age and polypharmacy terms were then used as candidate predictors in the LASSO" | PUB |
| Plot evidence | Fig. S3.1 titles/axes ("Fractional polynomial (…), adjusted for covariates"; "Partial predictor + residual") match Stata `mfp` + `fracplot`; caption "complete model (no variable selection)" matches `mfp` default `select(1)` | INF (high) |
| Powers set | {−2, −1, −0.5, 0, 0.5, 1, 2, 3}; power 0 = ln; repeated powers multiply by ln(x); FP2 max ⇒ 8 FP1 + 36 FP2 = 44 models | Stata default [S11] |
| Selection rule | `mfp` closed-test (function selection procedure): FP2 vs linear (3 df, α); if significant, FP2 vs FP1 (2 df, α). Default α = 0.05; backfitting cycles ≤ 5. The chosen forms (linear, log) are simpler than FP2, consistent with test-based simplification | INF (moderate); α not reported [NF] |
| Pre-scaling | Automatic: if min ≤ 0, shift by (−min + counting interval); divide by 10^(sign(p)·⌊\|p\|⌋), p = log10(range) | Stata [S11] |
| Model | Unpenalised logistic regression with **all** candidate predictors (binary + categorical indicators) | PUB/INF |
| Published result | Age: linear. Polypharmacy: ln((P+1)/10) | PUB |

### 6.2 LASSO logistic regression [PUB + Stata defaults]

| Aspect | Specification | Label |
|---|---|---|
| Model | Logistic regression with L1 (LASSO) penalty; clustering by practice **not** accounted for [PUB]; no interaction terms appear in Table S3.2 [INF] | PUB/INF |
| Software | Stata 17 (development + internal validation) [PUB]. The command is not named; official `lasso logit` is assumed (terminology "cross-validation function"); if community `lassologit` was used, λ is scaled by N and λ\* = 0.000123 means something else [ASSUME, Q-08] | PUB/ASSUME |
| λ selection | "chosen to minimise the cross-validation function on **10-fold** cross-validation" (λ-min, not 1-SE). **λ\* = 0.000123** (HTA only) | PUB |
| Objective (Stata) | Q = (1/N) Σ [−yᵢηᵢ + ln(1 + e^ηᵢ)] + λ Σⱼ \|βⱼ\|; intercept unpenalised; penalty loadings 1 | Stata [S11], verified |
| Standardisation | Every covariate column standardised to mean 0, **SD with divisor N (ddof = 0)** before penalisation; reported coefficients unstandardised. No option to disable | Stata [S11], verified numerically |
| Factor variables | Stata `lasso` includes indicators for **all** levels of factor variables and ignores base levels → candidate parameter count 90 (§5.1) | Stata [S11] + INF |
| λ grid | 100 values, log-spaced from λ_max (smallest λ zeroing all coefficients; λ_max = maxⱼ \|Σᵢ zᵢⱼ(yᵢ − ȳ)\| / N) down to λ_max × 10⁻⁴ (p < N) | Stata [S11], verified |
| CV criterion | Mean held-out deviance (−2 log-likelihood per observation) using penalised fits from the training folds | Stata [S11] |
| CV minimum rule | Minimum "identified" when five smaller λ values have CV deviance larger by relative difference ≥ 10⁻³ (`cvtolerance`); otherwise λ_stop (in-sample deviance change < 10⁻⁵) | Stata [S11] |
| Convergence | Coordinate descent, tolerance 10⁻⁷ on relative coefficient change | Stata [S11] |
| Reported coefficients | "Final penalised model" = penalised, unstandardised coefficients at λ\* | PUB/INF |
| Unpenalised refit | Ordinary logistic regression on LASSO-selected predictors with a reference level per categorical variable; ORs with 95% CI, "presented as a demonstration of effect sizes" (S3 l.948) [PUB]; CIs presumably Wald [INF] | PUB/INF |
| Seed / fold assignment | Not reported; Stata's fold-assignment algorithm is undocumented ⇒ **folds cannot be reproduced bit-for-bit** | NF |
| Python mapping | Stata λ ↔ scikit-learn C: **C = 1/(N·λ)**, with N the rows in the sample being fitted (training folds during CV) and X standardised identically. `liblinear` penalises the intercept (not equivalent). Implementation: own IRLS + coordinate-descent solver for Stata's objective, cross-checked against scikit-learn `saga` | INF, verified algebra |

### 6.3 What cannot be reproduced exactly

- CV fold membership and therefore λ\* (and borderline selected predictors).
- The exact Stata `cvtolerance` bookkeeping.
- Whether standardisation is recomputed inside CV training folds.
- The FP selection α and command.
- The authors' actual indicator coding.

Coefficients re-estimated on SAIL with our implementation would differ slightly even with identical data; this is expected and documented.

---

## 7. Published validation methodology

### 7.1 Performance measures [PUB; formulas S12]

| Measure | Definition to implement |
|---|---|
| C-statistic | Area under ROC curve. CI: DeLong or bootstrap (published SE method not identifiable). Logit(C) for pooling |
| Calibration slope | b in logit P(y=1) = a + b·LP |
| CITL | a in logit P(y=1) = a + 1·LP (LP as offset) |
| O/E | Σ yᵢ / Σ pᵢ |
| Grouped calibration | 20 equal-size groups ("20ths") of predicted risk; observed mean y vs mean p; Wald CI truncated to [0, 1] |
| Smoothed calibration | Loess (R default span 0.75) or lowess (Stata default bwidth 0.8); both labelled |
| Net benefit | NB(pt) = TP/N − FP/N · pt/(1−pt); treat-all = φ − (1−φ)·pt/(1−pt); treat-none = 0; standardised NB = NB/φ |
| Thresholds of clinical interest | **10%–25%**, pre-specified by a clinical user group and PPI (UK context). For Meuhedet, thresholds must be re-approved (M-12) |
| Test accuracy per 1,000 | TP/FP/TN/FN per 1,000, sensitivity, specificity at thresholds 0.10–0.25 in 0.01 steps |

### 7.2 Internal validation [PUB]

| Element | Specification |
|---|---|
| Apparent performance | Whole development cohort |
| Optimism correction | Harrell's enhanced bootstrap: resample with replacement (same N); **repeat the entire development process**; optimism = performance in bootstrap sample − performance of that model in the original data; corrected = apparent − mean optimism. **B = 25** (A&A, "for computational efficiency"; HTA generic text says 50 → E-07) |
| Model stability (Riley & Collins 2023) | Bootstrap models built with the identical process (applied in the original data per S3 l.513 and Riley & Collins; S2 Fig. S3.4 caption says "within that bootstrap sample" → E-21): LP-distribution overlay; prediction instability plot; calibration instability plot (lowess curves per bootstrap model); MAPE (mean over i, b of \|p_bi − p_i\|; eFalls mean 0.00196); classification instability index at thresholds (up to 80% at 0.10) |
| Practice-level heterogeneity | Performance per general practice; random-effects meta-analysis, **REML τ²**, **Hartung–Knapp–Sidik–Jonkman** correction, 95% prediction intervals [PUB]; degrees of freedom t_{k−1} (CI) and t_{k−2} (PI) reverse-engineered from Table 2 [INF]. Practices with < 10 events omitted from visualisations [PUB] |

### 7.3 Internal–external cross-validation [PUB]

- Clusters: **fifths of Welsh Index of Multiple Deprivation 2019 rank, plus a sixth group for missing WIMD** (19.7% of the cohort).
- Each cycle repeats the full development process on 5 groups and validates on the omitted group.
- Results are pooled by random-effects meta-analysis (REML + HKSJ).
- The published IECV prediction-interval arithmetic fits k = 6 with t_{k−2} [INF].

### 7.4 External validation and recalibration [PUB]

| Element | Specification |
|---|---|
| EV | Independent team (Leeds, R 4.2.3) applied the equation "as shown in Supplementary Box S3.1" to CB |
| Subgroups | Sex; BMI category; IMD quintile + missing; eFI frailty group (fit ≤ 0.12, mild ≤ 0.24, moderate ≤ 0.36, severe > 0.36) |
| Recalibration | Logistic recalibration: logit(p_recal) = α + β·LP_eFalls fitted by R `glm` on CB; apparent performance only; **not recommended for use without further validation** |
| Published recalibration | **p = expit(−0.423 + 1.25·LP)** (S2 Box S3.2 OMML; S3 Table 34 MathML). Confirmed by three independent reconstructions (LP distribution before/after, test-accuracy tables, O/E = 1.000). HTA Table 35 (all coefficients × 1.21, constant −7.25089539 ⇒ α = −0.046) is **inconsistent** with all published recalibrated results → rejected (D-07) |

### 7.5 Pooling scales [PUB-CONFLICT]

- HTA text: slope, CITL and O/E pooled on original scales; C on logit scale.
- The cited Snell 2018 guidance recommends log(O/E) and logit(C).
- Reverse-engineering Table 2 prediction intervals shows mixed practice: development O/E on the original scale; IECV/EV O/E on the log scale; IECV C on the original scale.
- → D-16.

### 7.6 Sample size [PUB]

| Calculation | Published | Reproducible? |
|---|---|---|
| Development (Riley 2019), prevalence 4.8%, 90 parameters, Nagelkerke 0.05 | 50,174 (2,409) | **Yes, exactly** (pmsampsize rounding) |
| Development, Nagelkerke 0.049 (headline) | 50,927 (2,445) | **No** — not reproducible by any tested conversion (E-12) |
| Development, Nagelkerke 0.15 | 13,867 (666) | Only with 76 parameters (E-12) |
| HTA Table 3 (108 parameters) | 19,706 / 60,209 | Yes |
| External validation (Riley 2021): O/E 7,625 (366); slope 10,882 (523) with LP mean −3.30, variance 0.690, skew 0.5, kurtosis 3; C 2,027 (98); sNB10% 2,289 (110); sNB25% 519 (25) | overall 10,882 (523) | Slope, C, sNB10 exactly; O/E and sNB25 within ±1 (rounding) |

### 7.7 Published performance (context only — not targets for Meuhedet) [PUB]

| Setting | Slope | CITL | O/E | C |
|---|---|---|---|---|
| Development, apparent, whole cohort | 1.0083 | 0.0000 | 1.0000 | 0.7434 |
| Development, optimism-adjusted | 1.0057 | −0.0003 | 0.9998 | 0.7430 |
| Development, pooled across practices | 0.99 (0.97–1.01) | 0.154 | 1.19 | 0.72 (0.72–0.72) |
| IECV (WIMD groups) | 0.99 (0.75–1.22) | −0.13 (−0.66 to 0.40) | 0.88 (0.53–1.46) | 0.72 (0.68–0.76) |
| EV (CB), whole cohort | 1.248 | −0.931 | 0.432 | 0.825 |
| EV (CB), pooled across practices | 1.203 (1.133–1.273) | −0.874 (−0.964 to −0.783) | 0.431 (0.388–0.479) | 0.816 (0.801–0.830) |
| Recalibrated, apparent, pooled | 0.964 (0.908–1.020) | 0.064 | 1.013 | 0.816 |

Additional published reference values:
- Development LP distribution: mean −3.30, variance 0.690 (SD ≈ 0.83), skewness ≈ 0.5.
- EV LP: mean −2.99, SD 0.905. After recalibration: mean −4.16, SD 1.13.

Published CIs for EV whole-cohort estimates are ≈ 4–5× narrower than analytic SEs and 6/68 subgroup CIs exclude their own point estimates (E-13). **Do not use published CIs as precision benchmarks.**

---

## 8. What we can and cannot reproduce

| Target | Reproducible? | How |
|---|---|---|
| Published equation arithmetic, worked examples (0.203; recalibrated 0.106) | **Yes** | Regression tests (§12) |
| Published outcome/predictor *definitions* | **Partly** | Definitions are specified here; code lists are on request (official) or a third-party proxy only |
| Sample-size calculations | **Mostly** | Unit tests against published numbers where reproducible (§7.6) |
| FP + LASSO + CV + bootstrap + IECV + calibration + DCA methodology | **Yes (method)** | Implemented to Stata-documented behaviour; verified on synthetic data |
| SAIL/CB performance numbers | **No** | No data access. External-dataset adapter prepared; never simulate them |
| Exact SAIL coefficients from re-fitting | **No** | Needs SAIL data plus unreported seed/folds; small differences expected even with access |
| eFalls on Meuhedet (Experiments A & B) | **Yes, once the Meuhedet modelling dataset exists** | Blocked by data mapping (§11, §14) |

---

## 9. Discrepancy and decision register

Decisions are **implementation defaults pending sign-off** (§15). Each item states:
- its evidentiary support separately ([PUB], [IMPL S5], [third-party S10], [INF]);
- that the choice itself is [ASSUME];
- an owner;
- a sensitivity analysis, or "none" with a justification.

Where a decision changes predicted risk, the alternative is a named, configurable variant reported alongside the primary result.

**Pre-registration rule:** Meuhedet outcome data may not be used to change any primary decision in this register. Decisions and code lists are frozen before outcome linkage (D-19). Mapping work may inspect predictor prevalences but never predictor–outcome associations.

### 9.0 D-00 — Global timing convention [ASSUME; supports PUB S3 l.360] — owner: biostatistician + data engineering

**The prediction is made at the start of the index day.**

| Data use | Rule |
|---|---|
| Predictors | Only records with `record_date < index_date` **and** `available_date < index_date` |
| Predictor windows of W days | `index_date − W days ≤ record_date ≤ index_date − 1 day` (W days before the index date) |
| Outcome | Events with `index_date ≤ event_date ≤ index_date + 1 calendar year − 1 day`. Reproduces the published SAIL window "1 April 2018 … to 31 March 2019" exactly and handles leap years |

Consequences:
- No record can contribute to both predictors and outcome, because index-day events belong to the outcome only.
- No record dated on or after the index date may influence any predictor, transformation parameter or model.
- The modelling dataset carries `predictor_max_record_date`. Schema validation fails if `predictor_max_record_date ≥ index_date` or if an outcome event date falls outside the window.

**Availability declaration (mandatory):**
- Every Meuhedet source table must declare its availability semantics (posting/ingestion timestamp column, or documented lag).
- If availability is unknown for a source, a pre-specified **lag buffer** applies: primary `record_date < index_date − 30 days` for that source; sensitivity 90 days. The resulting change in predictor prevalence is reported.
- Registry "first diagnosis date" fields are **forbidden** as record dates unless registry-entry timestamps exist (M-03).

Sensitivity: none for the window boundaries (published-faithful). The lag buffer is reported as above.

### 9.1 D-01 — Sex term and intercept (CRITICAL) — owner: PI + biostatistician

**The conflict [PUB-CONFLICT]:**

| Source | Intercept / constant | Sex term |
|---|---|---|
| S2 Table S3.2 (penalised and unpenalised columns); S3 Table 15 (copy); S5 implementation instructions Table 4; S6 Sci Rep 2026 Table S1 | Constant −5.954459 | Male = reference; **Female −0.303708** |
| S3 Table 35 (recalibrated; rejected, E-09) | Constant −7.25089539 (= −0.046 + 1.21 × −5.954459) | Male = reference; Female −0.36748668 (sex direction as Table S3.2) |
| S2 Box S3.1; S3 Tables 16 and 34; S1 main text: the model was applied "as shown in Supplementary Box S3.1"; TRIPOD checklist item 8d names Box S3.1 as how validation predictions were calculated; item 12b gives "Table S3.2, Box S3.1" as the final model | **−6.258** | "**−0.304 (if male)**" |

The sources agree for women (−5.954459 − 0.303708 = −6.258167) and differ by 0.607 logits for men. The worked example (a woman) cannot discriminate.

The conflict decomposes into two independent factors:

| Reading | Sex direction | Intercept level | Female intercept | Male intercept |
|---|---|---|---|---|
| **LP_C** `lp_c_box_s3_1` (Box literal) | female higher | 0.3037 below printed constant | −6.258167 | −6.561875 |
| **LP_A** `lp_a_table_s3_2` (Table literal) | male higher | printed constant | −6.258167 | −5.954459 |
| LP_B `lp_b_label_swap` | female higher | printed constant | −5.954459 | −6.258167 |
| LP_D2 `lp_d2_numeric_swap` (sex entered as numeric 1/2 with the other coding) | male higher | 0.3037 below | −6.561875 | −6.258167 |

- **LP_B = LP_C + 0.303708** and **LP_D2 = LP_A − 0.303708** for everyone. B and D2 are pure calibration-in-the-large shifts of C and A, with identical C-statistic, calibration slope and within-sex ranking.

**Evidence on the intercept level** [INF — two independent audits; S3 Figs 22 and 40; published predictor distributions]:
- **Development LP mean.** Stata reports −3.30, tied to observed risk 4.86% and apparent O/E 1.000.
  - Rebuilt from published predictor distributions: LP_A −3.00 to −3.11; LP_B −2.99 to −3.09; LP_C −3.29 to −3.39; LP_D2 −3.31 to −3.41.
  - LP_A cannot reach −3.30 under any polypharmacy distribution compatible with the published quartiles.
- **CB as-applied LP mean** −2.99. Rebuilt: LP_A −2.64 to −2.74; LP_C −2.92 to −3.03.
- ⇒ The published predictions were generated with an intercept level about 0.30 below the printed constant, i.e. **C or D2**, not A or B.

**Evidence on the sex direction:**
- **For female higher (C):**
  - Crude risk is higher in women (development F 5.93%, M 3.66%).
  - LASSO with λ\* ≈ 10⁻⁴ reproduces within-sex observed risk, so under the male-higher direction the non-sex covariates would need to raise women's LP by ≈ 0.83 logits (≈ 1 LP SD) versus ≈ 0.22 under the female-higher direction. Mean eFI F 0.14 / M 0.12 comes from Hollinghurst 2019, a different SAIL cohort, not the eFalls cohort. The audit's extreme female-adverse bound is 0.80, close to 0.83, so this argument is strong but not decisive.
  - HTA Ch. 5 narrative: "being female … contributed to a higher predicted fall/fracture risk". The care-home chapter's narrative agrees with its own table.
  - The near-parity of CB sex-specific O/E (F 0.417, M 0.463) is consistent with an applied Box-direction contrast, *conditional on* the covariate-gap argument. Assuming LP_A was applied instead, the same numbers imply the opposite (implementations notes). By itself it is not discriminating.
- **For male higher (A/D2):**
  - The printed table and all its derivatives (not independent).
  - The ARC-GM intermediate-risk list was 55% male (weak; confounded by exclusions).
  - The unpenalised constant OR 0.003: disputed between audits. One shows it fits the numeric-coding mechanism (C/D2 type, constant ≈ −5.9); the other reads it as weakly favouring the printed constant.

**Decision D-01 [ASSUME]:**
1. The published-equation adapter implements all four readings, using full-precision S2 Table S3.2 coefficients for all non-sex terms.
2. **Primary for Experiment A: `lp_c_box_s3_1` ("published as validated").** Justification, in order:
   - (i) the TRIPOD checklist (items 8d/12b) and main text name Box S3.1 as how validation predictions were computed;
   - (ii) it is consistent with the published LP distributions;
   - (iii) the sex-direction evidence.
3. **Mandatory co-reported sensitivity: `lp_a_table_s3_2` ("published as tabulated")**, the version in the implementation instructions, Sci Rep 2026 and probably the GM deployment.
4. `lp_b_label_swap` and `lp_d2_numeric_swap` are reported only as CITL-shift variants in a compact table (O/E, CITL). Their discrimination equals C and A respectively, as stated analytically.
5. Reports always include sex-stratified C, slope, CITL and O/E for C and A. The within-sex C-statistic is identical under all readings. The quantity logit(Y) ~ offset(LP_C) + male is reported **descriptively as "sex-specific CITL of LP_C"**. It is not interpreted as supporting any reading, because in a new population it mixes true sex-specific miscalibration.
6. **Blocker Q-01:** ask the corresponding author whether sex entered the Stata model as numeric `GNDR_CD` (1/2) or `i.sex`, which code is female, and for the exact LP code used by the EV team. Fallback if unanswered at analysis lock: publish with C primary and A co-reported.
7. The config key `published_equation.sex_parameterisation` has no code default and must be stated explicitly.

### 9.2 D-02 — BMI cut-points and derivation [ASSUME; supports S3 Table 16, IMPL S5, S10] — owner: clinical lead

| Aspect | Decision |
|---|---|
| Evidence | Among recorded BMI in SAIL: underweight 2.9%, normal 28.4%, overweight 36.9%, **obese 31.8%** (CB obese 27.8%). A share of 31.8% cannot arise from an "obese ≥ 40" rule. HTA Table 16, S5 (overweight 25.0–29.9, obese ≥ 30) and S10 ("BMI 30+ = obesity") state standard cut-offs |
| Cut-points | **WHO:** underweight < 18.5; normal 18.5 ≤ BMI < 25; overweight (reference) 25 ≤ BMI < 30; obese ≥ 30; missing if no valid BMI |
| Source precedence | (1) recorded BMI value; (2) BMI computed from weight/height recorded within 30 days of each other [IMPL S5 allows computation]; (3) coded BMI category without value mapped only where the code's range is unambiguous — otherwise ignored and counted [ASSUME] |
| Record rule | Most recent valid value before index [IMPL S5]. Same-day multiple values: mean [ASSUME] |
| Window | 5 years before index [third-party S10 for the BMI observable; eFalls NF] [ASSUME]; sensitivity: complete look-back |
| Plausibility / units | Values outside [10, 80] kg/m² invalid ⇒ not recorded; weight in kg, height in m or cm auto-detected only by documented rule (M-06). Requires clinical approval |
| Sensitivity | Printed "obese ≥ 40" (low priority) |

### 9.3 D-03 — Polypharmacy window, unit and exclusions [ASSUME; supports PUB S2 Box S3.1] — owner: clinical pharmacist

| Aspect | Decision |
|---|---|
| Window | **120 days before index:** `index − 120 d ≤ date ≤ index − 1 d` (D-00) [PUB]. Sensitivity: 90 days (S5; eFI2) |
| Unit | **Unique BNF paragraph (6-character code, "level 3", e.g. 040201)** — the most plausible meaning of "sub-sub-chapter" (NHSBSA structure; eFI2 "level 3") [INF]. BNF edition: NHSBSA 2017-18 (S5) [ASSUME]. Sensitivities: 7-character sub-paragraph; chemical substance |
| Excluded chapters | BNF pseudo-chapters 20–23 and non-drug items in chapters 1–15 (e.g. 060106 diagnostic/monitoring agents) [ASSUME]; chapters 18/19 unresolved (Q-03) |
| Counting from dispensing data [MEU] | Primary: an item counts if its **dispensing date** is in the window. Sensitivity: supply overlapping the window. Substances with no ATC→BNF match are **excluded and reported** (count and %). Combination products whose components fall in different paragraphs count **each paragraph once** |
| Transform | Experiment A: ln((P+1)/10) exactly. Experiment B: FP re-selection |
| Naming | `polypharmacy_count_{W}d` (primary `polypharmacy_count_120d`) |
| Must NOT | Replace by a ≥ 5 or ≥ 8 drugs flag (MEFI's ≥ 8 drugs/12 months is a Meuhedet frailty deficit, not eFalls) |

### 9.4 D-04 — Smoking derivation [ASSUME; supports S10, IMPL S5, S17] — owner: clinical lead

- The current-smoker indicator is 1 if the most recent smoking-status record before index (complete look-back) is "current"; otherwise 0, including never, ex and no record.
- A "never" recorded after an earlier smoker/ex code is treated as ex (S17 convention: SAIL Atkinson 2017; OpenSAFELY). An ex code followed by a current code gives current (S5).
- Ex vs never is irrelevant to the published LP (shared reference).
- Experiment B keeps the three-level candidate (never / ex / current; missing → never), per Stata all-levels coding. The published ex/never merge is presumably LASSO zeroing [INF].
- Sensitivity: "ever current".

### 9.5 D-05 — Alcohol derivation (explicit algorithm) [ASSUME; supports IMPL S5, third-party S10] — owner: clinical lead

1. Collect alcohol records with `index − 5 years ≤ date < index`.
2. Map each record to one of six levels:
   - coded concepts via the approved code map;
   - numeric units/week: 0 → zero; 1–20 → lower risk; 21–48 → higher risk; ≥ 49 → harmful [third-party S10 thresholds, provenance unknown];
   - units/day × 7 → units/week.
   - AUDIT-C is **not** mapped in the primary analysis (not in eFalls sources).
3. Patient level = highest in the precedence **harmful > higher_risk > previous_higher_risk_or_harmful > lower_risk > zero** [IMPL S5 "highest category over 5 years"]. A previous-harmful code does not override a later harmful code (already implied by the precedence).
4. Coded category and numeric record on the same day: the coded category wins [ASSUME].
5. No record ⇒ `missing` level (published missing category).

Sensitivity: sex-specific UK CMO thresholds (> 14 units lower-risk limit for both sexes; higher risk > 35 women / > 50 men). Confidence: low–moderate (Q-05).

### 9.6 D-06 — Age definition and upper limit [ASSUME] — owner: PI

| Aspect | Decision |
|---|---|
| Age | **Decimal age = (index_date − date_of_birth) / 365.25** (SAIL derived age from week of birth; flooring to completed years would lower every LP by ≈ 0.021 on average) |
| DOB imputation | Where only year/month is known, impute mid-period (day 15 / 1 July) and flag [ASSUME] |
| Upper limit | **No cap, no exclusion above 95** |
| Evidence | Fig. S3.1 data stop at 95; the original eFI cohort was 65–95 and the eFI+ protocol defines the population "by the existing eFI". No text states a cap; later implementations (ARC-GM) have no cap. The clinical-statistical audit recommended 65–95 as primary; we choose no cap because no eFalls text states one, and restricting the population is the larger departure. Q-06 remains a blocker for the final choice |
| Sensitivity | (a) exclude > 95; (b) cap age at 95 in the LP; (c) completed-year age. Always report the count > 95 |

### 9.7 D-07 — Recalibrated reference model [ASSUME; supports PUB S2 Box S3.2] — owner: biostatistician

- The published Bradford recalibration p = expit(−0.423 + 1.25 × LP_C) is used **only** as a regression-test reference (RT-06/07), never as a Meuhedet model.
- HTA Table 35 is rejected: it is inconsistent with every published recalibrated output (LP distribution, test-accuracy tables, O/E 1.000). HTA Table 34 is internally inconsistent, pointing to Table 35 while its formula gives −0.423/1.25.
- Meuhedet recalibration is re-estimated per D-19 and never on test data.

### 9.8 D-08 — Per-predictor look-back and definition rules [ASSUME; supports S8 eFI2 App. 1, IMPL S5, S10] — owner: clinical lead + biostatistician

| Aspect | Decision |
|---|---|
| Rule source | eFI2 Appendix 1 rules (5-year windows †/‡, age-at-record ≥ 18 / ≥ 55, resolution hierarchies ¥), applied **according to rule evidence**: **(a) count-matched** — the 30 predictors whose SAIL counts in eFI2 equal eFalls exactly (activity limitation, AF, cancer, cognitive impairment, COPD, dementia, dressing/grooming, environment problems, falls, fracture, fragility fracture, heart failure, housebound, liver problems, medication management, memory concerns, mobility problems, MND, palliative care, parkinsonism, peptic ulcer disease, PVD, requirement for care, respiratory disease, seizures, self-harm, skin ulcer, stroke, TIA, weight loss): rules applied, `rule_evidence: count_matched`; **(b) not checkable** (no eFI2 count published): rules applied and flagged `rule_evidence: not_checkable`; **(c) count mismatch** — hypotension/syncope (48,961 vs 37,756): rule unknown, complete look-back, `rule_fidelity: none`; **(d)** shopping problems (not in eFI2 App. 1) and hypertension (‡ without rule text): complete look-back, `rule_fidelity: none` |
| ‡ code-level exceptions | Need the official code list (Q-02); until then complete look-back with `rule_fidelity: partial` |
| Resolution hierarchies | Memory concerns removed if **cognitive impairment or dementia** is recorded (S10); cognitive impairment removed if dementia is recorded. "Removed if … added" is implemented as: the resolving record is dated on or after the resolved record and before index [ASSUME] |
| Non-code (measurement) rules [IMPL S5/S10] | Hypertension: code OR ≥ 3 readings ever with SBP ≥ 140 or DBP ≥ 90. Hypotension: ≥ 3 readings SBP < 90 or DBP < 60. Anaemia: code OR Hb below sex-specific threshold (M < 13.0, F < 11.5 g/dL), resolved by a later normal Hb. Activity limitation: Barthel ≤ 18. Cognitive impairment: 6CIT ≥ 8. Osteoporosis: T-score < −2.5. PVD: ABPI < 0.95. CKD: eGFR < 60 or ACR/protein thresholds. Thyroid: TSH < 0.36 or > 5.5 mU/L. **Primary: codes + measurement rules** (the development team's own implementation specification); **sensitivity: codes only**. Applicability to eFalls SAIL is UNRESOLVED (Q-02). Meuhedet requires lab/vital-sign sources (M-14) |
| Code matching | Specified codes only, not parent/child concepts (S5; S6) [IMPL] |
| BMI, smoking, alcohol | Follow D-02/D-04/D-05; they do **not** inherit eFI2 rules (eFI2 BMI/smoking counts differ from eFalls) |
| Look-back depth | Report membership length and EHR depth at index. Sensitivity restricted to members with ≥ 5 years continuous membership; primary keeps everyone, as eFalls did |
| Global sensitivity | All predictors "ever recorded", no age/resolution/measurement rules |

### 9.9 D-09 — Outcome operationalisation [ASSUME; supports PUB S1/S3 code list and window] — owner: clinical lead + data engineering

| Aspect | Decision |
|---|---|
| Window | `index_date ≤ event_date ≤ index_date + 1 year − 1 day` (D-00; published-faithful) |
| Encounters (primary) | ED attendance **or non-elective (emergency/urgent, including via ED) hospital admission** [ASSUME: avoids elective fixation-removal/rehabilitation admissions]. Sensitivity 1: any admission type |
| Codes (primary) | A **fracture code of the eFalls list (S22, S32, S42, S52, S72, S82, T08, T10, T12, T14.2, M80) in any diagnosis position**, OR **an external-cause fall code W00–W19 in any position**. (External-cause codes are never principal diagnoses, so a "primary position only" rule would delete most falls.) Sensitivity 2: fracture code in principal position OR W00–W19 in any position |
| Episode continuity | Exclude transfers and continuation spells that began before the index date (spell start < index). Sensitivity 3: additionally exclude persons with a qualifying hospital encounter in the 30 days before index |
| ICD variant | Declared **per source** (WHO ICD-10, ICD-10-CM, ICD-9-CM). For ICD-10-CM: keep 7th characters A/B (and C where applicable); exclude D/G/K/P/S (subsequent encounter/sequela); CM analogues required for W02, T08, T10, T12, T14.2 (M-02) |
| Code matching (WHO ICD-10) | 3-character categories match their sub-codes after removing "."; T14.2 matches T14.2 and its sub-codes only |
| Deaths | Retained as non-events (published). Report deaths within the window and outcome counts among decedents; competing-risk analysis is a later, separately named experiment (D-18) |
| Disenrolment | Not modelled (published). Report counts. An exclusion of non-decedents with < 1 year follow-up **conditions on the future** and is sensitivity only |
| Pre-modelling audit | Tabulate outcome events by code block, diagnosis position, encounter type, admission method and source **before** any model fitting; stored in the dataset manifest |
| Binary outcome | 1 if ≥ 1 qualifying event |

### 9.10 D-10 — Categorical coding in the retrained LASSO [ASSUME; supports Stata S11] — owner: biostatistician

- Primary: Stata-faithful all-levels indicators (no base level), standardised.
  - For two-level factors (sex) all-levels and reference coding give an identical penalty after standardisation. D-10 matters only for BMI (5 levels incl. missing), smoking (3) and alcohol (6).
  - Coefficients are re-expressed relative to the published reference levels for reporting; predictions are invariant.
- The FP selection stage (unpenalised) uses reference coding.
- Sensitivity: reference-coded LASSO.

### 9.11 D-11 — Resampling replicates [ASSUME] — owner: biostatistician

| Use | Replicates |
|---|---|
| Published-protocol optimism replication | B = 25 |
| Meuhedet optimism and stability | **B = 200**; report the Monte Carlo SE of each summary |
| Test-cohort CIs and paired A-vs-B differences (ΔC, Δslope, ΔCITL, ΔNB) | ≥ 1,000 patient-level bootstrap replicates (separate from the above) |

Within every bootstrap replicate, CV folds are assigned by **original patient ID** (grouped), so duplicated rows never straddle folds. λ\* is recorded per replicate.

A replicate whose fit is degenerate (single outcome class, or separation created by resampling — `DegenerateFitError`) is recorded as a failure with its reason; the analysis raises if failures exceed 10% of replicates. The primary fit never skips.

At N ≳ 10⁵ optimism is of order 10⁻⁴. Compute is prioritised for temporal testing, heterogeneity and stability.

### 9.12 D-12 — Candidate list [ASSUME] — owner: biostatistician

- Implement exactly the **78 listed candidates**.
- Experiment A uses exactly the 75 published non-reference coefficients plus the intercept.
- Counts 80 vs 78 and 74 vs 75 are logged (E-05, E-06).

### 9.13 D-13 — FP procedure for Experiment B [ASSUME; supports INF S11] — owner: biostatistician

- `mfp`-style closed test (α = 0.05, FP2 maximum, Stata powers set, select = 1, backfitting ≤ 5 cycles) in an unpenalised logistic model with all candidates.
- Categorical variables are reference-coded at this stage.
- **Automatic pre-scaling only when a non-linear power is selected.** A selected linear term enters on the raw scale (so age keeps per-year units, like the published 0.0415506).
- No centring of FP or LASSO columns.
- The xorder statistic is the LR-test p-value for dropping each variable from the all-linear model [ASSUME].
- **Stata `logit` omission semantics** in the unpenalised FP-stage fit (and in the unpenalised alternative logistic model): constant columns are omitted, and an indicator whose value 1 (or 0) occurs with only one outcome class is dropped together with the observations it predicts perfectly; repeated until stable; every omission is logged and stored in the FP selection table / fit diagnostics (`models/separation.py`). Separation that remains (e.g. through combinations of indicators) is a `DegenerateFitError` that stops the primary fit.
- FP selection is repeated inside bootstrap/IECV replicates.
- **Co-reported** Experiment B variants: `fp_mode: select` (primary) and `fp_mode: fixed_published`.

### 9.14 D-14 — λ selection for Experiment B [ASSUME; supports Stata S11] — owner: biostatistician

- Stata-equivalent grid (100 values, ratio 10⁻⁴) and CV mean-deviance minimum with 10 folds.
- Deterministic seeded fold assignment, grouped by patient.
- **Standardisation is recomputed within each CV training fold** (no information from held-out rows); sensitivity: full-sample standardisation.
- If no CV minimum is identified under the 5-point / 10⁻³ rule, λ\* = the grid point with minimum CV deviance, flagged `cv_minimum_identified: false` and reported.
- Record the full CV path, λ\*, standardised and unstandardised coefficients, seed.
- Stata-exact bookkeeping is a verification target (KKT conditions; agreement with an independent solver), not a design requirement: at λ\* ≈ 10⁻⁴ predictions are insensitive to it.

### 9.15 D-15 — Observation units [ASSUME; MEU] — owner: biostatistician

- eFalls-labelled experiments (A, B) use **exactly one row per patient per index-date cohort**, as eFalls did.
- Designs with multiple index dates per patient in one training set are L4 or separately named sensitivities.
- When cohorts are pooled, CV folds, bootstraps and IECV are grouped by patient, and a patient's index dates must be ≥ 1 year apart.

### 9.16 D-16 — Meta-analysis scales and cluster rules [ASSUME; supports S12] — owner: biostatistician

| Aspect | Rule |
|---|---|
| Scales | Pool log(O/E), logit(C), slope (original), CITL (original) (Snell 2018) |
| τ² and CI | REML τ²; **modified** HKSJ CI (max(1, q)) with t_{k−1} |
| Prediction interval | t_{k−2}, conventional REML variance |
| Report | τ, I², number of clusters and members excluded |
| Cluster unit | Clinic with ≥ 10 events (and ≥ 1 non-event, converged slope). If this excludes > 20% of the cohort, aggregate to district |
| Sensitivity | One-stage random-intercept/random-slope logistic calibration model using all clusters |

### 9.17 D-17 — Calibration curves [ASSUME] — owner: biostatistician

- Grouped points: 20 equal-size groups by rank of predicted risk.
- Smoothed curve: local-linear LOWESS (statsmodels, frac = 0.75, it = 0, with `delta` for speed), labelled "LOWESS". It is not R loess and not Stata lowess.
- Summary calibration indices ICI/E50/E90 reported alongside.

### 9.18 D-18 — Death and competing risk [PUB approach] — owner: biostatistician

- Primary: published binary approach.
- Competing-risk analysis: later separately named experiment.

### 9.19 D-19 — Meuhedet validation design (CRITICAL) [ASSUME; L3a; MEU] — owner: biostatistician + PI; signed off before outcome linkage

eFalls developed at one index date (SAIL) and validated at another place and time (CB). It performed **no temporal validation**, so Meuhedet temporal results have no published comparator (stated as a limitation in every report).

**1. Three single-index-date cohorts**, each reconstructed as of its own index date (as-of rules, B-04):

| Cohort | Index date | Role |
|---|---|---|
| **T_dev** | e.g. 1 April 2018, mirroring eFalls | Development (Experiment B fitting; FP; λ) |
| **T_val** | ≥ T_dev + 1 year | Recalibration and hyperparameter selection only |
| **T_test** | ≥ T_val + 1 year; primary windows avoiding the March 2020 – early 2021 lockdown periods, e.g. 1 January 2022 | Locked test cohort |

Windows overlapping October 2023 onward are a drift sensitivity only. Final dates depend on data availability and are signed off.

**2. Embargo.** Anything applied to cohort X (model, recalibration, threshold, hyperparameters) may be estimated only from outcomes whose windows end **before** index(X): `index + 1 year − 1 day < index(X)`.

**3. Data freeze.** Data freeze ≥ index(T_test) + 1 year + documented claims/documentation lag, with a monthly outcome-completeness check.

**4. Locked test.**
- T_test is evaluated once per pre-registered config hash.
- Every evaluation is logged in a run registry.
- The test partition is held by a guard object in code and released only after model selection is frozen.

**5. Common test.** Experiment A (published scoring), recalibrated A, Experiment B and all L4 models report primary results on the **same T_test rows**, with paired patient-level bootstrap CIs for ΔC, Δcalibration and ΔNB.

**6. Recalibration.**
- Experiment A's recalibration (intercept-only and logistic) is fitted on **T_val** (validation data only, as required).
- Sensitivity: recalibration fitted on T_dev, so the comparison is "published structure + 2 local parameters" vs "local refit" learned from the same data.
- A is also reported on T_dev and T_val as secondary results.

**7. Patients across cohorts.** The same patient may appear in several cohorts (deployment-realistic). Sensitivity: T_test restricted to patients absent from T_dev ("new patients").

**8. Sample size.** Before analysis:
- Riley 2019 for Experiment B (90 candidate parameters, Meuhedet T_dev prevalence);
- Riley 2021 (`pmvalsampsize` criteria) for T_test using the LP_C distribution.

If T_dev is too small, pool two annual cohorts (patients grouped, index dates ≥ 1 year apart) and label the deviation.

**9. Secondary validation.** Geographic hold-out by district, and IECV by Meuhedet socio-economic group quintiles + missing (WIMD analogue, M-10).

**10. Single-index-date fallback.** If only one index date is available: patient- and clinic-grouped random partitions (development / validation / test) with a documented limitation statement. eFalls-style bootstrap optimism correction and IECV then become the primary internal validation.

### 9.20 D-20 — Development-support sensitivity [ASSUME] — owner: biostatistician

- Several published coefficients rest on very few SAIL cases, so their penalised estimates are effectively unpenalised:
  - dressing and grooming (< 10 cases; OR CI 0.37–9.3);
  - medication management (< 10; 0.32–31.3);
  - motor neurone disease (251), meal preparation (298), washing and bathing (377).
- If Meuhedet maps these at CB-like prevalence (dressing 15% in CB), they add up to 0.45–0.80 logits to large groups.
- **Sensitivity for Experiment A:** set to 0 every retained predictor with SAIL count < 500. Report the change in C, CITL and slope, and the distribution of these terms' LP contribution.
- Primary keeps the published coefficients (L1).

### 9.21 D-21 — Uncertainty and SE methods [ASSUME; supports S12] — owner: biostatistician

| Measure | Method |
|---|---|
| Point-estimate CIs on a partition | Patient-level percentile bootstrap (≥ 1,000; D-11) |
| Within-cluster SEs (meta-analysis): C | DeLong variance → logit scale by delta method |
| Within-cluster SEs: CITL and slope | GLM Wald SE |
| Within-cluster SEs: log(O/E) | √((1 − φ)/O) |
| Decision-curve grid | Thresholds 0.01–0.50 step 0.01; region of interest 0.10–0.25 (UK); Meuhedet thresholds require approval (M-12) |
| Funnel-plot bounds | ±1.96 × √(τ² + SE²) around the pooled estimate |

### 9.22 D-22 — Permitted predictor source classes [ASSUME; MEU] — owner: clinical lead

| Aspect | Rule |
|---|---|
| Primary | Every Meuhedet coded clinical diagnosis source available before index (community and hospital diagnoses received by Meuhedet), mirroring "complete primary care EHR … and linked data" (S3) |
| Sensitivity | Community-coded diagnoses only |
| Assessment-derived values | Nursing-questionnaire and assessment values (e.g. 999.52 "fell during last year", 999.175 mobility) may populate an **L1 concept only with explicit clinical approval**, recorded with `source_class: assessment` in the mapping. Otherwise they are **L3b additions** and create a differently named experiment |
| Tagging | Every mapped record source carries `source_class` ∈ {community_diagnosis, hospital_diagnosis, prescription, dispensing, measurement, lab, assessment, registry} |

---

## 10. Reporting errata catalogue (non-decision)

These do not change the implementation but must not be used as validation targets.

| ID | Location | Problem |
|---|---|---|
| E-01 | S2 Box S3.1 | "[see table S3.1 for values]" should be Table S3.2 |
| E-02 | S2 | Two different tables both labelled "Table S3.4" |
| E-03 | S2 Box S3.1; S3 Tables 16/34 | Sex term sign/intercept (→ D-01) |
| E-04 | S2 Box S3.1; S3 Table 34 | "obese if BMI ≥ 40"; overweight undefined; "< 24.9" (→ D-02) |
| E-05 | S1 | 80 candidates stated vs 78 listed |
| E-06 | S3 Ch. 5, Fig. 21 | 74 retained vs 75 |
| E-07 | S3 Ch. 2 | 50 bootstrap samples vs 25 in S1 |
| E-08 | S3 Tables 8/14/15/21/35 | "Higher-risk drinking" printed for the lower-risk reference row |
| E-09 | S3 Table 35; S3 Table 34 text | Recalibrated coefficients inconsistent with S2 Box S3.2 / S3 Table 34 formula (→ D-07) |
| E-10 | S3 | CB cohort 88,947 / 3,079 events / 84 practices (Fig. 1, dated 1 January 2019; text l.558 says 1 April 2018) vs S1: 81,685 / 2,389 / 76 practices / 1 January 2019. S1 EV prediction intervals fit k = 76 |
| E-11 | S1 Table 1; S2 Table S3.1 | CB IMD group 3 total 13,337 should be 14,337 (and 16.3% → 17.6%); CB "Never" smoking total 59,295 should be 59,279; S3.1 CB housebound fallers 931 should be 831 |
| E-12 | S2 Table S2.1 | 50,927 (2,445) not reproducible; 13,867 row implies 76 parameters |
| E-13 | S1 Table 2; S2 Table S3.4 | EV whole-cohort CIs ≈ 4–5× too narrow; 6/68 subgroup CIs exclude their point estimates; recalibrated pooled C CI "0.801 to 1.000" |
| E-14 | S2 Table S3.3 | CITL CI "−0.0115 to 0.115" (upper bound 0.0115) |
| E-15 | S1 Table 2 | IECV C τ² reported on C scale; development O/E PI on natural scale; IECV/EV O/E pooled on log scale despite S3 "original scales" |
| E-16 | S3 | R version 4.3.1 vs 4.2.3 |
| E-17 | S5 | eFalls worked example arithmetic wrong (stated L −3.023169; terms sum to −2.978465) — **must not be used as a test** |
| E-18 | S2 figure captions | S3.13 says "development data" for EV curves; S3.7 refers to a "mortality model" |
| E-19 | S1 | SAIL small-cell suppression ineffective (derivable 8 events) — privacy lesson for Meuhedet reporting (M-13) |
| E-20 | S2 only | Polypharmacy described as "unique drugs" (Box text) vs "unique BNF sub-sub-chapters" (footnote; footnote absent from S3) |
| E-21 | S2 Fig. S3.4 vs S3 l.513 | Calibration-instability plots: bootstrap models applied "within that bootstrap sample" (S2) vs "in the original data set" (S3). We follow S3 and Riley & Collins 2023 |
| E-22 | S3 Table 16 vs Box S3.1 | BMI wording differs ("standard BMI cut-offs" vs "obese if BMI ≥ 40") |

---

## 11. Meuhedet mapping requirements [MEU]

Nothing here is part of eFalls. The data-engineering layer, not the ML layer, implements these mappings.

**Mapping states** (`mapping_status`) are:
- `TO_BE_MAPPED`
- `candidate`
- `validated` (clinically_validated true, approver, date)
- `under_recorded` (validated but prevalence ratio vs SAIL/CB outside a pre-set band, e.g. < 0.5)
- `unavailable` (cannot be recorded in Meuhedet)

No feature may be used for a reportable scientific run unless `validated` or `under_recorded`.

| ID | Concept | eFalls source representation | Meuhedet status / candidate | Risk |
|---|---|---|---|---|
| M-01 | Cohort roster, membership, index date(s) | GP registration on index date | `[Meuhedet_DWH].[Dims].[Dim_Customer_Details]` identified (probably a current snapshot). **Membership-history intervals, clinic and SES valid on the index date are required** (B-04) | Survivorship bias and leakage of later attributes if built from a current snapshot |
| M-02 | Outcome encounters and codes | EDDS + PEDW, WHO ICD-10 | ICD version **per source UNKNOWN**. S15 uses locally adapted ICD-9; S16 observed ICD-9 and ICD-10 [INF]. See the code-status list below | E-code prefix collisions: E880–E887 without "E" collide with ICD-9 880–887 (open wounds/amputations); undotted E88x collides with ICD-10 E88.x. The existing Meuhedet list (880.9, 883.9, 884.9, 886.9, 888, 929.3, E987, 719.7) must be re-derived, not reused: 719.7 is difficulty walking, E987 undetermined intent, 929.3 late effect |
| M-03 | Record availability semantics | EHR event dates | Every source table must declare availability/posting semantics (D-00). Unknown ⇒ lag buffer | Back-filled diagnoses/registries leak future information |
| M-04 | Polypharmacy | BNF paragraphs, prescriptions, 120 days | ATC-coded purchase/dispensing records [S16, INF]. Pharmacist-built ATC5 + route → BNF paragraph crosswalk with unmapped-substance log (D-03). No ATC level equals a BNF paragraph; ATC-4 and ATC-3 counts are sensitivity only | Dispensing ≠ prescribing |
| M-05 | 72 binary conditions | SNOMED/CTV3/Read v2 code groups (+ measurement rules, D-08) | ICD-9/ICD-10 (and local) diagnoses per D-22; official eFalls code lists on request (Q-02); MEFI (S15) is a Meuhedet-specific starting point, not eFalls | Functional variables (dressing/grooming, medication management, meal preparation, washing, shopping, toileting, finances) near-zero in SAIL — see D-20 |
| M-06 | BMI | Most recent recorded BMI | Measurement tables UNKNOWN; precedence and units per D-02 | Unit errors |
| M-07 | Smoking / alcohol | UK status codes and units | Lifestyle tables UNKNOWN | Mostly missing alcohol |
| M-08 | Encounter classification | ED vs admission; emergency vs elective; spell start | Hospital invoice / event type / admission method / spell-start fields UNKNOWN | Elective fracture follow-up and continuation spells contaminating the outcome |
| M-09 | Death, disenrolment | ONS mortality; registration | Death date and membership end UNKNOWN | Informative censoring |
| M-10 | Clusters | General practice; WIMD fifths | Clinic / district identifiers valid at index; Meuhedet socio-economic score quintiles + missing | Small clinics (D-16) |
| M-11 | Feature coverage | — | See rule below | Silent bias |
| M-12 | Decision thresholds | 10–25% (UK clinical/PPI panel) | Candidate thresholds are configuration; **no risk category is exposed** until clinically/business approved | — |
| M-13 | Small-cell disclosure | SAIL < 10 suppression | Suppress cells < 10 in exported reports, including complementary disclosure (E-19) | Privacy |
| M-14 | Labs and vital signs for measurement rules | BP, Hb, eGFR/ACR, TSH, T-score, ABPI, Barthel, 6CIT (D-08) | Lab/vitals tables UNKNOWN | Unit and reference-range drift |
| M-15 | Original eFI score (36 deficits) | eFI frailty group subgroup analysis (fit ≤ 0.12, mild ≤ 0.24, moderate ≤ 0.36, severe > 0.36) | Not mappable without the eFI code lists; MEFI (S15) is a labelled Meuhedet-specific alternative for subgroup reporting | Different frailty construct |

**M-11 feature-coverage rule (Experiment A):**
1. **Primary:** unavailable predictors are zero-filled. This is labelled an L3a mechanic, not the published missing-data rule, which covers absent codes in a system that could record them.
2. **Mandatory sensitivities:**
   - (a) unavailable terms filled with SAIL prevalence (preserves the development mean LP);
   - (b) intercept-only recalibration on T_val, to separate CITL change caused by missing features from population differences.
3. **Coverage metric:** the share of development LP variance explained by available terms, approximated as Σ_available β²p(1−p) / Σ_all β²p(1−p), with p the SAIL prevalence. Continuous terms use published distribution summaries.
4. **Hard rule:** if **age, sex, polypharmacy, falls, fracture, fragility fracture or dementia** is unavailable, or coverage < 90%, the run is renamed **`efalls_partial_scoring`** and is not reported as a validation of eFalls.
5. **Publication before outcome linkage:** a per-predictor prevalence table (Meuhedet T_dev vs SAIL vs CB), with mapping states.

**M-02 outcome code status list** (versioned code-list artifact; all candidates, not validated):
- **WHO ICD-10:** the published list.
- **ICD-10-CM:**
  - W00–W19 analogues and fracture categories, 7th character A/B (C where applicable) only.
  - W02, T08, T10, T12 and T14.2 do not exist; CM analogues need clinical review.
- **ICD-9-CM, CANDIDATE:**

  | eFalls code | ICD-9-CM candidate |
  |---|---|
  | W00–W19 | E880–E886, E888 |
  | S22 | 805.2–805.3, 807.0–807.4, 809.0–809.1 |
  | S32 | 805.4–805.7, 808 |
  | S42 | 810–812 |
  | S52 | 813 |
  | S72 | 820–821 |
  | S82 | 822–824 |
  | T08 | 805.8–805.9 |
  | T12 | 827 |
  | T14.2 | 829 |

- **ICD-9-CM, REVIEW:**
  - M80 → 733.1x **and** co-coded 733.0x (733.1 alone is any pathological fracture);
  - 806.x (fracture with cord injury, for S22/S32/T08);
  - T10 → 818.x (GEM maps 818.x to wrist/hand S62.90);
  - E886.9.
- **ICD-9-CM, EXCLUDE:**
  - E887 "fracture, cause unspecified" (WHO analogue X59.0 not in eFalls);
  - intent codes E957, E968.1, E987;
  - late effect E929.3.

---

## 12. Regression and reproducibility tests

### 12.1 Published-equation tests (tolerances documented; all probabilities to 7 dp)

| Test ID | Input | Expected | Tolerance | Source |
|---|---|---|---|---|
| RT-01 | Box S3.1 example with **rounded** printed coefficients (female, 89 y, polypharmacy 8, underweight, previous higher-risk/harmful drinking, dementia, liver problems, osteoporosis) | LP = −1.36777; p = 0.2029805 (v0.2 printed 0.2029811 — erratum); printed "−1.368", "0.203" | ±5×10⁻⁴ vs printed | S2 Box S3.1 |
| RT-02 | Same patient, full precision, **female under `lp_c_box_s3_1` and `lp_a_table_s3_2`** | LP = −1.408453; p = 0.1964781 | LP 1×10⁻⁶; p 5×10⁻⁷ | derived |
| RT-03 | Male twin, `lp_a_table_s3_2` | LP = −1.104745; p = 0.2488518 | same | derived |
| RT-04 | Male twin, `lp_c_box_s3_1` | LP = −1.712161; p = 0.1528836 | same | derived |
| RT-05 | `lp_b_label_swap`: female LP = −1.104745 (p 0.2488518); male LP = −1.408453 (p 0.1964781) | — | same | derived |
| RT-05b | `lp_d2_numeric_swap`: female LP = −1.712161 (p 0.1528836); male LP = −1.408453 (p 0.1964781) | — | same | derived |
| RT-06 | Box S3.2 recalibration of rounded LP −1.368 | p = 0.1059305 (v0.2 printed 0.1059302 — erratum; published 0.106) | ±5×10⁻⁴ vs printed | S2 Box S3.2 |
| RT-07 | Box S3.2 recalibration of full-precision LP −1.408453 | p = 0.1012360 (40-digit arithmetic; v0.2 printed 0.1012433 was an arithmetic error found by the implementation review) | 1×10⁻⁶ | derived |
| RT-08 | Coefficient table integrity | 75 non-reference coefficients + constant; 62 non-zero binary; 10 zero; values equal the printed decimal strings of `tools/source_table_s3_2.csv` (exact decimal match; float equality 1×10⁻¹²) | exact | S2 |
| RT-12 | ln((P+1)/10) at P = 0, 9, 61 | −2.302585, 0.0, 1.824549 | 1×10⁻⁶ | definition |

### 12.2 Methodological tests

| Test ID | Input | Expected | Tolerance | Source |
|---|---|---|---|---|
| RT-09 | Riley 2019 development: prevalence 0.048, 90 parameters, Nagelkerke 0.05, shrinkage 0.9, **pmsampsize convention**: R²_CS = round(0.05 × max R²_CS, 3) = 0.016, n = ceil, events = ceil(n·φ) | 50,174 (2,409). Companion assertion: unrounded R²_CS gives 50,227–50,228 | exact | S2 Table S2.1 |
| RT-10 | Riley 2021 validation C criterion: prevalence 0.048, C 0.743, CI width 0.1, **Newcombe SE**, smallest integer n meeting the width | 2,027 (98) | exact | S2 Table S2.2 |
| RT-11 | Net benefit from the published test-accuracy row (t = 0.10: TP 19.2, FP 174.3 per 1,000) | NB = −0.00017; treat-all NB with φ = 2389/81685 < 0 | 1×10⁻⁵ | S2 Table S3.4 (second table) |
| RT-13 | Stata FP pre-scaling rule on **unit-spaced integer vectors**: 0..61 → (shift 1, scale 10); 21..80 → (0, 10); 0..2380 → (1, 1000); non-unit gaps {0, 5, 10, …, 60} → shift 5 | as stated | exact | S11 |
| RT-14 | FP selection | Closed-test decision function on injected deviances; simulated truths (log / linear / FP2) recovered at large n; deviance equals statsmodels GLM within 1×10⁻⁶ | as stated | S11 |
| RT-15 | LASSO solver | KKT conditions at several λ within 1×10⁻⁵; λ_max zeroes all coefficients; agreement with scikit-learn `saga` at C = 1/(Nλ) within 1×10⁻³; deterministic λ\* for fixed seed | as stated | S11 |
| RT-16 | Random-effects meta-analysis | REML τ², HKSJ CI and PI reproduce the verified Stata [META] example (τ² 0.0754; HK CI −0.1413 to 0.4084; 90% PI −0.414 to 0.681) | 1×10⁻³ | S12 |
| RT-17 | Definitional boundaries | Polypharmacy days index−121/−120/−1/0; outcome days index−1/index/index+1y−1d/index+1y; BMI 18.49/18.5/24.99/25/29.99/30; ICD matching W19 vs W1x, S72.0 vs S62, T14.2 vs T14.1, E88.0 not a fall; schema violations for nulls in binary/count features | as stated | D-00, D-02, D-03, D-09 |

The S5 worked example (E-17) must **not** be used. Published CIs (E-13) must not be used as targets.

---

## 13. External-dataset adapter and synthetic fixtures

### 13.1 External-dataset adapter (eFalls-compatible data)

1. **Responsibility.** Map a source-specific column layout to the canonical modelling-dataset schema (§13.3). This includes the 72 binary indicators, `age_years`, `sex`, `polypharmacy_count_120d`, `bmi_value`, `smoking_status`, `alcohol_category`, outcome, index date, research ID, clusters, and dataset/mapping versions.
2. **Rename and recode only.** It must not compute features from raw events; that is the data engineering of that environment.
3. **Versioning.** Adapters declare their source, column map and recode map, all versioned, and recorded in `dataset_manifest.json`.
4. **Acceptance targets** if SAIL/CB access is ever obtained:
   - Development apparent C within ±0.005 of 0.7434.
   - EV whole-cohort C within ±0.005 of 0.825.
   - O/E within ±0.02 of 0.432 under the parameterisation confirmed by Q-01.
   - Deviations trigger investigation, not tuning.

### 13.2 Synthetic fixtures (software tests only)

| Aspect | Requirement |
|---|---|
| Determinism | Fixed seed (20260914), fixed size, fixed `created_utc`; identical file hash on regeneration |
| Generating mechanism | A **known, arbitrary, non-eFalls** logistic model (coefficients deliberately different from Table S3.2), so that no fixture result can be mistaken for a simulated SAIL/CB result. Predictor prevalences are arbitrary and **must not be tuned to published SAIL/CB marginals** |
| Leakage exercise | Raw event tables contain post-index and back-filled records |
| Watermark | Manifest `source: synthetic_fixture`, `scientific_use_allowed: false`. Every report and comparison generated from it starts with "SYNTHETIC FIXTURE — SOFTWARE TEST OUTPUT, NOT SCIENTIFIC EVIDENCE" (enforced in code). Mixing synthetic and real runs in one comparison is an error |

### 13.3 Modelling-dataset schema

Normative definition: `configs/features/efalls_v1.yaml` plus `docs/ARTIFACT_SCHEMAS.md`, enforced by `falls_ml.data.schema`.

| Column | Type / values |
|---|---|
| `research_id` | string |
| `index_date` | date |
| `predictor_max_record_date` | date (< index_date) |
| `outcome_12m` | int8 {0, 1} |
| `outcome_first_event_date` | date, nullable |
| `age_years` | float ≥ 65 |
| `sex` | {female, male} |
| `polypharmacy_count_120d` | int ≥ 0 |
| `bmi_value` | float, nullable; **the dataset carries the value, the model preprocessing derives the category** |
| `smoking_status` | {never, ex, current}, nullable |
| `alcohol_category` | {harmful, higher_risk, lower_risk, previous_higher_risk_or_harmful, zero}, nullable |
| 72 binaries | int8 {0, 1}, non-null |
| clusters | `practice_id`, `deprivation_group`, `site_id` (strings, nullable) |
| `death_date`, `followup_end_date` | date, nullable |
| `dataset_version`, `mapping_version`, `source` | constants |

### 13.4 Predictor configuration schema

`configs/features/efalls_v1.yaml` is normative and generated from the publication tables by `tools/generate_efalls_configs.py`. Each feature entry has:

| Field | Meaning / allowed values |
|---|---|
| `name` | snake_case canonical feature name |
| `concept` | clinical concept |
| `role` | `predictor` |
| `dtype` | binary, float, float_nullable, count, categorical, categorical_nullable |
| `levels` | categorical levels |
| `layer` | L1_published; later: L2_assumption, L3a_meuhedet_mapping, L3b_meuhedet_predictor, L4_alternative |
| `efalls_term` | label in S2 Table S3.2 |
| `exact_efalls_baseline` | bool |
| `retained_in_published_model` | bool (binaries) |
| `efalls_source` | published source description and code-list availability |
| `time_window` | `window` {complete_history \| lookback_years N \| lookback_days N}, `age_at_record_min`, `resolved_by`, `rule_fidelity` {full, partial, none}, `rule_evidence` {count_matched, not_checkable, count_mismatch}, `proxy_source`, `proxy_rule_text`, `non_code_rule` |
| `transformation` | published and retrained |
| `valid_range` | numeric bounds |
| `missing_rule` | forbid, absent_is_zero, missing_category, reference_level |
| `reference_level` | reference level |
| `decisions` | D-IDs |
| `published_prevalence` | SAIL, CB |
| `meuhedet_source` | mapping file, `mapping_status`, clinically_validated |
| `clinically_validated` | bool |

`configs/mappings/meuhedet_v0.yaml` holds, per feature:
- `source_table`, `source_field`, `code_system`, `code_list` (versioned file), `source_class` (D-22);
- `mapping_status` (§11), `clinically_validated`, `validated_by`, `validation_date`, `notes`.

---

## 14. Blockers and author queries

| ID | Query / blocker | Owner / target | Blocks |
|---|---|---|---|
| **Q-01** | Was sex entered as numeric `GNDR_CD` (1/2) or `i.sex`, and which code is female? Exact LP code applied in CB external validation; which parameterisation GM ICB and later users should use | A. Clegg (corresponding author, Leeds); cc L. Archer (first author) | Interpretation of Experiment A; fallback in D-01 |
| **Q-02** | Official eFalls code lists (SNOMED/CTV3/Read v2), per-predictor time/age/measurement rules, technical specification document; written permission to use them (and/or the S10 proxy) for Meuhedet research | A. Clegg; S. Relton; University of Leeds IP | Clinical validation of the 72 predictor mappings |
| Q-03 | Polypharmacy: BNF level, excluded chapters (18/19?), BNF edition, repeat vs acute, 120 vs 90 days | Authors / K. Best | D-03 confirmation |
| Q-04 | BMI cut-points/window/source precedence; smoking derivation | Authors | D-02/D-04 |
| Q-05 | Alcohol category definitions, thresholds and window | Authors | D-05 |
| Q-06 | Age: continuous vs completed years; cap/exclusion above 95 | Authors | D-06 |
| Q-07 | Outcome extraction: diagnosis positions, ED coding, admission method, index-day handling, spell continuity | Authors / SAIL team | D-09 |
| Q-08 | FP command/α; LASSO command (`lasso logit` vs `lassologit`); seed | Authors | D-13/D-14 |
| Q-09 | Licence for non-UK research use of the equation and code lists | University of Leeds IP / A. Clegg | Publication of Meuhedet results |
| Q-10 | Did linked secondary-care diagnoses feed predictors? | Authors | D-22 |
| Q-11 | Meuhedet: ICD version per data stream; E-code prefix storage; availability/posting timestamps per source; membership-history and clinic/SES history tables | Meuhedet data engineering | M-02, M-03, B-04 |
| **B-01** | **The Meuhedet modelling dataset does not exist yet**: cohort, outcome and all predictor mappings (M-01…M-15) | Meuhedet data engineering + clinical leads | All scientific runs |
| **B-02** | Clinical validation of Meuhedet mappings | Clinical lead | Reportable runs |
| B-03 | Approval of decision thresholds for risk categories | Clinical/business owners | Exposure of `risk_category` |
| **B-04** | **As-of-index reconstruction:** roster from membership-history intervals covering the index date; clinic/SES/address from history valid on that date; availability declaration for every source. If only current snapshots exist, no scientific run is allowed | Meuhedet data engineering | All scientific runs |

### 14a. Unresolved-items register

Status: **U** = unresolved with interim decision; **BL** = blocked.

| ID | Item | Sources in conflict / silent | Affects | Interim | Status |
|---|---|---|---|---|---|
| U-01 | Sex coding/intercept | S2 Table vs Box; S3; S5; S6 | Experiment A | D-01 | U (Q-01) |
| U-02 | Official predictor code lists | on request | 72 binaries | S10 proxy (not for scientific use); D-08 | BL (Q-02) |
| U-03 | Per-predictor windows/age rules | S3 silent; eFI2 App. 1 vs S10 conflicts (falls 5y rule; urinary incontinence ≥ 55 vs > 18; AF ≥ 18; housebound home-visit rule; environment problems content) | 72 binaries | D-08 (eFI2 App. 1 wins) | U (Q-02) |
| U-04 | Measurement rules applicability to eFalls | S5/S10 vs eFalls silent | hypertension, hypotension, anaemia, activity limitation, cognitive impairment, osteoporosis, PVD, CKD, thyroid | D-08 primary codes + measurements | U (Q-02) |
| U-05 | Hypertension ‡ without rule text | eFI2 App. 1 | hypertension | complete look-back | U |
| U-06 | Shopping problems absent from eFI2 App. 1 | eFI2 | shopping problems | complete look-back | U |
| U-07 | Hypotension/syncope SAIL count mismatch (48,961 vs 37,756) | eFI2 vs eFalls | hypotension/syncope | rule_fidelity none | U |
| U-08 | CB bone disease 9.6% vs SAIL 0.8% (S10 group = osteomalacia + Paget's) | S2 vs S10 | bone disease | none | U |
| U-09 | Respiratory disease exclusive of asthma/COPD after eFI split? | S10 tags | respiratory disease | not exclusive (independent groups) [ASSUME] | U |
| U-10 | "Removed if … added" timing semantics | S10 | memory concerns, cognitive impairment | D-08 | U |
| U-11 | Polypharmacy unit/window/chapters | S2 vs S5 vs eFI2 | polypharmacy | D-03 | U (Q-03) |
| U-12 | BMI cut-points/window | S2 vs S3 vs S5 | BMI | D-02 | U (Q-04) |
| U-13 | Smoking derivation | silent | smoking | D-04 | U |
| U-14 | Alcohol definitions/thresholds | silent; S10 provenance unknown | alcohol | D-05 | U (Q-05) |
| U-15 | Age continuous vs completed; cap at 95 | silent; Fig. S3.1 | age | D-06 | U (Q-06) |
| U-16 | Outcome positions/encounters/spells | silent | outcome | D-09 | U (Q-07) |
| U-17 | Linked data in predictors | S3 vs S1 | all binaries | D-22 | U (Q-10) |
| U-18 | FP command/α; LASSO command; seed; fold standardisation | silent | Experiment B | D-13/D-14 | U (Q-08) |
| U-19 | Categorical reference levels a priori vs LASSO zeroing | silent | Experiment B reporting | D-10 | U |
| U-20 | Bootstrap count for MAPE/instability | S1 25 vs S3 50 | stability | D-11 | U |
| U-21 | 80 vs 78 candidates; 74 vs 75 retained | S1/S3 | none (equation unaffected) | D-12 | U |
| U-22 | CB cohort definition (81,685 vs 88,947; 76 vs 84 practices) | S1 vs S3 | acceptance targets | use S1 | U |
| U-23 | Recalibration parameters (Box S3.2 vs Table 35) | S2 vs S3 | RT-06/07 | D-07 | resolved by evidence (Box S3.2) |
| U-24 | SE/CI methods behind published intervals | silent; E-13 | acceptance targets | D-21; do not target CIs | U |
| U-25 | Development sample-size headline 50,927 | S2 | none | RT-09 uses reproducible row | U |
| U-26 | Meuhedet ICD versions, E-prefix, availability timestamps, history tables | UNKNOWN | outcome, all predictors, cohort | D-00, D-09, M-02, M-03 | BL (Q-11, B-04) |
| U-27 | Meuhedet dates for T_dev/T_val/T_test; data freeze; claims lag | UNKNOWN | D-19 | proposal in D-19 | BL (sign-off) |
| U-28 | Licence for Meuhedet use | silent | publication | none | BL (Q-09) |
| U-29 | D-20 scope: `alcohol_category=previous_higher_risk_or_harmful` rests on 90 SAIL cases but D-20 zeroes only binary predictors | implementation review | Experiment A sensitivity | binaries only | U (biostatistician) |
| U-30 | M-11 coverage share uses unmodified published coefficients even when low-support terms are zeroed; covariances between terms ignored | implementation review | coverage label | as implemented, documented | U (PI) |
| U-31 | All-levels LASSO coding: the split of coefficients across levels of a factor is not unique (two-level factors arbitrary); per-level selection frequencies are not meaningful for such factors | implementation review (D-10) | stability report | report re-expressed coefficients; interpret per-level stability with caution | U (biostatistician) |
| U-32 | Computational scale at Meuhedet size: LASSO CV builds an O(N·p²) Gram per IRLS step (tens of minutes per 10-fold fit at N ≈ 660k); FP selection ≈185 logistic fits; B = 200 replicates multiply both | implementation review | D-11 compute plan | reduce B or use fixed λ within replicates if needed (documented deviation) | U |
| U-33 | M-13 complementary-disclosure suppression not implemented (only cells 1–9 shown as "<10") | implementation review | export of real-data reports | block export of real-data reports until designed | BL (privacy) |
| U-34 | Paired bootstrap CIs across experiments (D-19 §5) | implementation review | model comparison | `comparison_paired_differences.csv` vs the published primary variant on common test rows (v0.4); threshold-specific net-benefit differences not bootstrapped | partly resolved |
| U-35 | Drift monitoring: PSI is weak for binary prevalence shifts; label maturity for *monitoring* labels | implementation review | monitoring | prevalence and AUROC-drop rules added against the bundle's validation reference (v0.4); training data are checked against `data_freeze_date`, monitoring labels are not | U |
| U-36 | D-09 sensitivity 3 (exclude persons with a qualifying hospital encounter in the 30 days before index) not implemented; 733.1x+733.0x co-code rule REVIEW only | implementation review | outcome sensitivities | not run | U |
| U-37 | Bootstrap CIs cover AUROC, PR-AUC, Brier, slope, intercept, CITL and O/E only; threshold metrics, net benefit and ICI are point estimates | final review | report uncertainty | labelled point estimates | U |
| U-38 | Locked-test enforcement is per runs directory (`test_evaluation_registry.jsonl`); a new `--runs-dir` starts a new registry, and `TestSetGuard` is an in-process control | final review | D-19 §4 | organisational control: one registered runs directory per locked test cohort | U (PI) |
| U-39 | Resampling analyses (stability, optimism, IECV) refit with the frozen tuned hyperparameters; tuning is not repeated inside replicates, so optimism of tuned L4 models may be understated | final review | L4 internal validation | disclosed in metrics and limitations | U (biostatistician) |
| U-40 | Generic nullable continuous extension features declare `missing_category`; the implementation is training median plus a `__missing` indicator column (BMI keeps the published missing category) | final review | Meuhedet-enhanced features | actual handling written to `metrics.missingness` | U |
| U-41 | Served prediction: published equation served uncalibrated by default; retrained and L4 models serve the validation-fitted recalibration (`calibration.serve_recalibrated`) | final review | production | eligibility judged on the served variant; both variants reported | U (PI sign-off) |

---

## 15. Sign-off checklist

| # | Item | Required role | Status |
|---|---|---|---|
| 1 | D-00 timing convention + M-03 availability declarations / lag buffer | Biostatistician + data engineering | ☐ |
| 2 | D-01 primary `lp_c_box_s3_1`, co-reported `lp_a_table_s3_2`, pending Q-01; fallback rule | PI + biostatistician | ☐ |
| 3 | D-02 WHO BMI cut-offs, precedence, window, plausibility | Clinical lead | ☐ |
| 4 | D-03 polypharmacy window/unit/dispensing rules and ATC crosswalk approach | Clinical pharmacist | ☐ |
| 5 | D-04/D-05 smoking and alcohol algorithms | Clinical lead | ☐ |
| 6 | D-06 decimal age, no cap (sensitivities) | PI | ☐ |
| 7 | D-07 recalibration reference use | Biostatistician | ☐ |
| 8 | D-08 per-predictor rules incl. measurement rules | Clinical lead + biostatistician | ☐ |
| 9 | D-09 outcome operationalisation | Clinical lead + data engineering | ☐ |
| 10 | D-10…D-18, D-21 methodological decisions | Biostatistician | ☐ |
| 11 | **D-19 validation design, cohort dates, embargo, data freeze, test lock** | Biostatistician + PI | ☐ |
| 12 | D-20 development-support sensitivity | Biostatistician | ☐ |
| 13 | D-22 predictor source classes; assessment-derived values rule | Clinical lead | ☐ |
| 14 | M-02 outcome code status list validated per source | Clinical coding lead | ☐ |
| 15 | M-11 coverage rule and thresholds | PI | ☐ |
| 16 | B-04 as-of reconstruction confirmed feasible | Data engineering lead | ☐ |

---

## 16. Change log

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-09-14 | Initial Phase 1 specification from S1–S15 and adversarial audit of publication inconsistencies |
| 0.2 | 2026-09-14 | Revised after four independent adversarial reviews (two fact-checkers, completeness critic, biostatistician). Summary below |

| 0.3 | 2026-09-15 | Implementation-review corrections: RT-07 arithmetic erratum (0.1012360) and 7-dp corrections to RT-01/RT-06; Stata `logit` omission semantics for constant columns and perfect predictors (D-13); degenerate bootstrap replicates recorded, bounded at 10% (D-11); unresolved items U-29…U-36 |
| 0.4 | 2026-09-15 | Final-review enforcement: `data_freeze_date` label-maturity check (D-19 §3); test-evaluation registry (D-19 §4); eFalls purity rules checked at config load (published: primary variant, all variants, no params/extensions/subset; retrained: all 78 candidates, FP select/fixed, whitelisted LASSO params, λ-min, 10 folds; both: no tuning, embargo on); served variant made explicit; U-34/U-35 updated; U-37…U-41 added |

Changes in 0.2:
- **Timing:** D-00 changed to start-of-index-day, matching the published outcome window.
- **Sex term:** D-01 recast as two factors, added LP_D2, TRIPOD 8d/12b justification, pre-registration rule.
- **Definitions:** decimal age (D-06); explicit alcohol algorithm (D-05); BMI precedence (D-02); dispensing counting rules (D-03).
- **Look-back rules:** D-08 applies them by rule evidence and adds measurement rules.
- **Outcome:** D-09 redefined (non-elective admissions, fracture/external-cause any position, ICD-10-CM 7th characters, spell continuity).
- **New decisions:** D-19 validation design, D-20 low-support sensitivity, D-21 SE methods, D-22 source classes.
- **Mapping:** M-11 coverage rule with `efalls_partial_scoring`; M-02 code status list; M-14/M-15.
- **Tests:** corrected RT-02/RT-08/RT-09/RT-10/RT-13; added RT-05b, RT-14…RT-17.
- **Layers:** L3 split into L3a/L3b.
- **Registers:** unresolved-items register (§14a); citation and quotation corrections throughout.

---

## Appendix A — Published non-binary terms and intercept (S2 Table S3.2 = S3 Table 15)

Reproduced from Archer et al., Age Ageing 2024 (CC BY 4.0). "Reference" = level with no coefficient. Constant as printed; see D-01 for sex/intercept variants.

| Group | Term | Penalised coefficient | Unpenalised OR (95% CI) |
|---|---|---:|---|
| — | Age (years) | 0.0415506 | 1.043 (1.041–1.045) |
| Polypharmacy | ln((Polypharmacy+1)/10) | 0.3296295 | 1.392 (1.366–1.418) |
| Gender | Male | Reference | — |
| Gender | Female | -0.303708 | 0.732 (0.712–0.753) |
| BMI category | Underweight | 0.4896735 | 1.632 (1.537–1.732) |
| BMI category | Normal | 0.2394177 | 1.269 (1.228–1.31) |
| BMI category | Overweight | Reference | — |
| BMI category | Obese | -0.0411134 | 0.952 (0.92–0.985) |
| BMI category | Missing | -0.1451981 | 0.858 (0.825–0.892) |
| Smoking | Ex/never | Reference | — |
| Smoking | Current | 0.0684529 | 1.078 (1.039–1.118) |
| Alcohol consumption | Harmful drinking | 0.4164064 | 1.536 (1.343–1.756) |
| Alcohol consumption | Higher risk drinking | 0.1549725 | 1.255 (0.879–1.791) |
| Alcohol consumption | Lower risk drinking | Reference | — |
| Alcohol consumption | Previous higher risk/harmful drinking | 0.0849676 | 1.226 (0.577–2.603) |
| Alcohol consumption | Zero alcohol | 0.0070124 | 1.055 (0.812–1.369) |
| Alcohol consumption | Missing | -0.0679367 | 0.935 (0.858–1.018) |
| — | Constant | -5.954459 | 0.003 (0.002–0.003) |

## Appendix B — All 72 binary candidate predictors

Coefficients and ORs: S2 Table S3.2. Prevalences: S2 Table S3.1 (n (%)). Origin: eFI 2016 Box 1 name or eFI+ additional candidate (from S10 provenance tags; splits INFERRED). Rule column: eFI2 Appendix 1 symbols (S8) — † all codes within 5 years; ‡ some codes within 5 years; * age ≥18; α age ≥55; ¥ resolution hierarchy (memory concerns are removed by cognitive impairment **or** dementia). The rule column is an INFERRED proxy for eFalls applied according to rule evidence (D-08), not an eFalls publication statement.

| # | eFalls predictor (Table S3.1 name) | Retained by LASSO | Penalised coefficient (Table S3.2) | Unpenalised OR (95% CI) | SAIL n (%) | Connected Bradford n (%) | Origin | eFI2 Appendix 1 window/age rule (INFERRED proxy) |
|---:|---|---|---:|---|---:|---:|---|---|
| 1 | Abdominal pain | yes | -0.0641861 | 0.93 (0.902–0.959) | 101210 (15.3) | 15393 (18.8) | eFI+ additional | † all 5y |
| 2 | Activity limitation | yes | 0.0475092 | 1.064 (0.966–1.172) | 5646 (0.9) | 1689 (2.1) | eFI 2016 — Activity limitation | none |
| 3 | Anaemia and haematinic deficiency | yes | 0.1733029 | 1.193 (1.153–1.234) | 57016 (8.6) | 13356 (16.4) | eFI 2016 — Anaemia and haematinic deficiency | ‡ some 5y |
| 4 | Anxiety | no (LASSO zero) | 0 (not selected) | — | 32014 (4.8) | 5838 (7.1) | eFI+ additional | † all 5y |
| 5 | Asthma | yes | 0.0816046 | 1.086 (1.052–1.121) | 135836 (20.6) | 15812 (19.4) | eFI 2016 — Respiratory disease (split; inferred from efi-asthma_copd tag) | * age>=18 |
| 6 | Atrial fibrillation | yes | 0.1519654 | 1.17 (1.131–1.21) | 59098 (8.9) | 11690 (14.3) | eFI 2016 — Atrial fibrillation | * age>=18; lone AF age>=55 |
| 7 | Back pain | yes | 0.0498699 | 1.054 (1.027–1.081) | 181488 (27.5) | 24164 (29.6) | eFI+ additional | ‡ some 5y (exceptions: chronic back pain, spondylosis, disc prolapse) |
| 8 | Bone disease | yes | -0.0267845 | 0.956 (0.867–1.054) | 5574 (0.8) | 7831 (9.6) | eFI+ additional | none |
| 9 | Cancer | yes | 0.0301854 | 1.035 (1.007–1.063) | 134167 (20.3) | 21378 (26.2) | eFI+ additional | none |
| 10 | Chronic kidney disease | no (LASSO zero) | 0 (not selected) | — | 156983 (23.8) | 35713 (43.7) | eFI 2016 — Chronic kidney disease | none |
| 11 | Cognitive impairment | yes | 0.1472251 | 1.168 (1.079–1.264) | 6644 (1) | 26706 (32.7) | eFI 2016 — Memory and cognitive problems (inferred split; no efi- tag) | ¥ resolves if dementia recorded |
| 12 | COPD | yes | 0.039956 | 1.042 (1.007–1.08) | 77849 (11.8) | 10730 (13.1) | eFI 2016 — Respiratory disease (split; inferred from efi-asthma_copd tag) | none |
| 13 | Dementia | yes | 0.1038111 | 1.116 (1.054–1.181) | 18870 (2.9) | 8597 (10.5) | eFI 2016 — Memory and cognitive problems (inferred split; no efi- tag) | ¥ (listed) |
| 14 | Depression | yes | 0.1633415 | 1.182 (1.136–1.229) | 43299 (6.6) | 8321 (10.2) | eFI+ additional | † all 5y |
| 15 | Diabetes mellitus | yes | 0.0373911 | 1.042 (1.01–1.075) | 124173 (18.8) | 21157 (25.9) | eFI 2016 — Diabetes | ‡ 5y only drug-induced/remission/gestational |
| 16 | Dizziness | yes | 0.0198363 | 1.024 (0.993–1.055) | 88020 (13.3) | 11112 (13.6) | eFI 2016 — Dizziness | ‡ 5y except vestibular/vertigo codes |
| 17 | Dressing and grooming problems | yes | 0.4532777 | 1.867 (0.374–9.326) | <10 (0) | 12277 (15.0) | eFI+ additional | none |
| 18 | Dyspnoea | no (LASSO zero) | 0 (not selected) | — | 68751 (10.4) | 15099 (18.5) | eFI 2016 — Dyspnoea | ‡ 5y except MRC grade 4 |
| 19 | Environment problems | no (LASSO zero) | 0 (not selected) | — | 12249 (1.9) | 3054 (3.7) | eFI+ additional | none (footnote: housing & social environment codes) |
| 20 | Faecal incontinence | yes | -0.071791 | 0.914 (0.83–1.007) | 5589 (0.8) | 3824 (4.7) | eFI+ additional | none |
| 21 | Falls | yes | 0.3009161 | 1.351 (1.314–1.389) | 106839 (16.2) | 18182 (22.3) | eFI 2016 — Falls | †α all 5y; age>=55 |
| 22 | Fatigue | yes | -0.0635057 | 0.928 (0.88–0.978) | 23641 (3.6) | 5348 (6.5) | eFI+ additional | † all 5y |
| 23 | Foot problems | yes | 0.0282736 | 1.031 (0.993–1.069) | 51706 (7.8) | 14499 (17.7) | eFI 2016 — Foot problems | ‡ 5y for corns only |
| 24 | Fracture | yes | 0.1957923 | 1.219 (1.187–1.253) | 157004 (23.8) | 19453 (23.8) | eFI+ additional | α age>=55 |
| 25 | Fragility fracture | yes | 0.2031303 | 1.226 (1.186–1.266) | 79033 (12) | 15018 (18.4) | eFI 2016 — Fragility fracture | α age>=55 |
| 26 | General mental health | yes | 0.0991068 | 1.107 (1.06–1.155) | 44450 (6.7) | 18079 (22.1) | eFI+ additional | † all 5y |
| 27 | Headache | yes | -0.0149365 | 0.975 (0.931–1.022) | 34383 (5.2) | 4134 (5.1) | eFI+ additional | † all 5y |
| 28 | Hearing impairment | yes | -0.0168728 | 0.976 (0.949–1.003) | 122237 (18.5) | 18685 (22.9) | eFI 2016 — Hearing impairment | ‡ 5y only hearing loss/deteriorating/impaired |
| 29 | Heart failure | yes | -0.0190399 | 0.971 (0.94–1.004) | 76939 (11.7) | 8915 (10.9) | eFI 2016 — Heart failure | none |
| 30 | Heart valve disease | no (LASSO zero) | 0 (not selected) | — | 24071 (3.6) | 6296 (7.7) | eFI 2016 — Heart valve disease | none |
| 31 | Housebound | yes | 0.2549983 | 1.289 (1.25–1.329) | 74800 (11.3) | 8931 (10.9) | eFI 2016 — Housebound | ‡α 5y and >=55 for home visit (not chronic or acute) |
| 32 | Hypertension | yes | -0.0318888 | 0.959 (0.934–0.986) | 347534 (52.6) | 54726 (67.0) | eFI 2016 — Hypertension | ‡ (no specific rule text given) |
| 33 | Hypotension or syncope | yes | 0.0778591 | 1.084 (1.046–1.122) | 37756 (5.7) | 14582 (17.9) | eFI 2016 — Hypotension/syncope | 5y except PD with orthostatic hypotension, chronic, idiopathic, NOS |
| 34 | Inflammatory arthritis | yes | 0.0586785 | 1.065 (1.034–1.098) | 96213 (14.6) | 17549 (21.5) | eFI 2016 — Arthritis (split; inferred from efi-arthritis tag) | none |
| 35 | Inflammatory bowel disease | yes | 0.0145135 | 1.021 (0.973–1.072) | 30168 (4.6) | 1315 (1.6) | eFI+ additional | none |
| 36 | Ischaemic heart disease | no (LASSO zero) | 0 (not selected) | — | 97668 (14.8) | 22880 (28.0) | eFI 2016 — Ischaemic heart disease | none |
| 37 | Liver problems | yes | 0.3803626 | 1.485 (1.314–1.677) | 3400 (0.5) | 1112 (1.4) | eFI+ additional | none |
| 38 | Meal preparation problems | yes | -0.140024 | 0.834 (0.513–1.357) | 298 (0) | 1106 (1.4) | eFI+ additional | none |
| 39 | Medication management | yes | 0.8030273 | 3.153 (0.318–31.278) | <10 (0) | 480 (0.6) | eFI+ additional | none (listed as "Unable to manage medications") |
| 40 | Memory concerns | yes | 0.2601186 | 1.296 (1.238–1.358) | 25552 (3.9) | 885 (1.1) | eFI 2016 — Memory and cognitive problems (inferred split; no efi- tag) | †¥ all 5y; resolves if cognitive impairment recorded |
| 41 | Mobility problems | yes | -0.1310067 | 0.865 (0.805–0.929) | 8894 (1.3) | 15805 (19.3) | eFI 2016 — Mobility and transfer problems | none |
| 42 | Mono or hemiparesis | yes | 0.1020457 | 1.119 (1.032–1.213) | 8102 (1.2) | 1342 (1.6) | eFI+ additional | none |
| 43 | Motor neurone disease | yes | -0.1209976 | 0.805 (0.464–1.398) | 251 (0) | 91 (0.1) | eFI+ additional | none |
| 44 | Musculoskeletal problems | yes | 0.0419361 | 1.046 (1.019–1.073) | 236264 (35.8) | 29156 (35.7) | eFI+ additional | † all 5y |
| 45 | Osteoarthritis | yes | 0.0634073 | 1.067 (1.041–1.094) | 220727 (33.4) | 33211 (40.7) | eFI 2016 — Arthritis (split; inferred from efi-arthritis tag) | none |
| 46 | Osteoporosis | yes | 0.1276254 | 1.135 (1.1–1.172) | 78236 (11.8) | 14915 (18.3) | eFI 2016 — Osteoporosis | none |
| 47 | Palliative care | yes | -0.2353552 | 0.774 (0.701–0.855) | 5451 (0.8) | 5584 (6.8) | eFI+ additional | none |
| 48 | Parkinsonism and tremor | yes | 0.2312839 | 1.266 (1.205–1.331) | 20670 (3.1) | 2537 (3.1) | eFI 2016 — Parkinsonism and tremor | none |
| 49 | Peptic ulcer disease | yes | 0.0056687 | 1.023 (0.916–1.143) | 4941 (0.7) | 979 (1.2) | eFI 2016 — Peptic ulcer | † all 5y |
| 50 | Peripheral neuropathy | yes | 0.0335789 | 1.038 (0.994–1.084) | 44734 (6.8) | 8797 (10.8) | eFI+ additional | none |
| 51 | Peripheral vascular disease | yes | 0.0173065 | 1.021 (0.981–1.062) | 44496 (6.7) | 10952 (13.4) | eFI 2016 — Peripheral vascular disease | none |
| 52 | Problems managing finances | no (LASSO zero) | 0 (not selected) | — | 227 (0) | 4 (0.0) | eFI+ additional | none |
| 53 | Requirement for care | yes | -0.2177301 | 0.792 (0.75–0.836) | 23427 (3.5) | 6689 (8.2) | eFI 2016 — Requirement for care | none |
| 54 | Respiratory disease | yes | 0.0132757 | 1.018 (0.978–1.059) | 58989 (8.9) | 8783 (10.8) | eFI 2016 — Respiratory disease | ‡ 5y only chronic cough, PE/infarction |
| 55 | Seizures | yes | 0.2571899 | 1.303 (1.227–1.385) | 15238 (2.3) | 2171 (2.7) | eFI+ additional | none |
| 56 | Self-harm | yes | 0.1461241 | 1.182 (0.993–1.406) | 1601 (0.2) | 623 (0.8) | eFI+ additional | † all 5y |
| 57 | Severe mental illness | yes | 0.0676153 | 1.071 (1.026–1.119) | 46665 (7.1) | 15873 (19.4) | eFI+ additional | none |
| 58 | Shopping problems | no (LASSO zero) | 0 (not selected) | — | 302 (0) | 473 (0.6) | eFI+ additional | NOT LISTED in eFI2 App1 |
| 59 | Skin ulcer | yes | 0.0746926 | 1.08 (1.046–1.115) | 75506 (11.4) | 10231 (12.5) | eFI 2016 — Skin ulcer | none |
| 60 | Sleep problems | yes | 0.0056967 | 1.012 (0.961–1.066) | 22554 (3.4) | 3765 (4.6) | eFI 2016 — Sleep disturbance | † all 5y |
| 61 | Social vulnerability | yes | 0.0495075 | 1.061 (1.004–1.122) | 20681 (3.1) | 10331 (12.6) | eFI 2016 — Social vulnerability | ‡ 5y only widowed/bereavement/loneliness |
| 62 | Stress | yes | -0.0200494 | 0.965 (0.9–1.035) | 15374 (2.3) | 3049 (3.7) | eFI+ additional | † all 5y |
| 63 | Stroke | yes | 0.0788542 | 1.086 (1.047–1.126) | 48091 (7.3) | 10733 (13.1) | eFI 2016 — Cerebrovascular disease (inferred; no efi- tag) | none |
| 64 | Thyroid problems | yes | -0.0273864 | 0.965 (0.938–0.993) | 108866 (16.5) | 8765 (10.7) | eFI 2016 — Thyroid disease | none |
| 65 | Toileting problems | no (LASSO zero) | 0 (not selected) | — | 241 (0) | 2401 (2.9) | eFI+ additional | none |
| 66 | Transient ischaemic attack | no (LASSO zero) | 0 (not selected) | — | 29018 (4.4) | 4911 (6.0) | eFI 2016 — Cerebrovascular disease (inferred; no efi- tag) | none |
| 67 | Urinary incontinence | yes | 0.0345173 | 1.039 (0.998–1.081) | 41496 (6.3) | 13310 (16.3) | eFI 2016 — Urinary incontinence | α age>=55 |
| 68 | Urinary system disease | yes | 0.0119309 | 1.016 (0.988–1.044) | 155282 (23.5) | 32769 (40.1) | eFI 2016 — Urinary system disease | ‡ 5y except TURP, catheter, cystitis, detrusor instability, retention, BPH |
| 69 | Visual impairment | yes | 0.02332 | 1.024 (0.996–1.053) | 145462 (22) | 34017 (41.6) | eFI 2016 — Visual impairment | none |
| 70 | Washing and bathing | yes | -0.1118824 | 0.853 (0.555–1.313) | 377 (0.1) | 2796 (3.4) | eFI+ additional | none |
| 71 | Weakness | yes | -0.0589464 | 0.903 (0.699–1.167) | 792 (0.1) | 1120 (1.4) | eFI+ additional | † all 5y |
| 72 | Weight loss | yes | 0.0301414 | 1.033 (0.995–1.072) | 49466 (7.5) | 6114 (7.5) | eFI 2016 — Weight loss and anorexia | ‡ 5y only appetite loss |

## Appendix C — Evidence archive

Detailed research and audit notes (with verbatim short quotes, URLs and line references) are archived in `docs/research_notes/`:

| File | Content |
|---|---|
| `hta_methods.md` | HTA monograph methods, protocols V20/V21, registry |
| `hta_falls_results.md` | HTA eFalls chapter/appendix cross-check vs Age & Ageing; recalibration reconstruction; mean-LP analysis |
| `public_assets.md` | Errata search, code/data availability, licensing, implementation instructions, later literature |
| `implementations.md` | ARC-GM pilot, Sci Rep 2026 validation, Leeds implementation instructions |
| `efi_codelists.md` | eFI 2016, eFI2, public Baseline2 code list, per-predictor rules |
| `stata_methods.md` | Stata 17 lasso/fp/mfp behaviour, numeric verification, Python mapping |
| `validation_methods.md` | Formulas: calibration, C, NB, bootstrap optimism, instability, IECV, REML+HKSJ, sample size |
| `meuhedet_translation.md` | BNF/ATC, ICD-10→ICD-9-CM candidates, alcohol, smoking, BMI definitions |
| `audit_numeric.md`, `audit_clinical_stat.md` | Two independent adversarial audits of the publication inconsistencies |
| `spec_review_v01.md` | Four independent adversarial reviews of specification v0.1 (basis of v0.2) |

Raw downloaded source files (journal XML/DOCX/PDF, manuals) were held in a session workspace and are not redistributed in this repository; every item is re-fetchable from the URLs in §3.1 and in the notes.
