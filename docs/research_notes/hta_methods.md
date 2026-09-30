# eFalls reproduction: what the HTA monograph, protocol, award record and registry add beyond Age & Ageing 2024

Compiled 2026-09-14. Scope: the NIHR HTA monograph GJAC1008 (Chapter 2 Methods in full, plus the eFalls-relevant parts of Chapters 3, 5 and 9, Appendix 1, Appendix 4, and the Data-sharing statement); both posted protocol versions; the NIHR award record; and the ClinicalTrials.gov record NCT04113174.

**Evidence labels**
- **[EXPLICIT]**: stated in the source (quotes are 40 words or fewer).
- **[INFERRED]**: my reasoning from explicit evidence; the reasoning is given.
- **[NOT FOUND]**: searched for in the sources listed and absent.

## 0. Source key (all paths absolute; line numbers refer to these files)

Scratchpad root `S = <research workspace (not distributed)>`

| Key | Source | Local file |
|---|---|---|
| HTA | Archer L, Relton SD, ... Clegg A. *Development and evaluation of the eFI+ tool for older people...* Health Technol Assess 2026;30(61). doi:10.3310/GJAC1008. CC BY 4.0. Web: https://www.journalslibrary.nihr.ac.uk/hta/GJAC1008 | `S/research/raw/gjac1008_struct.txt` (2664 lines); raw HTML `S/research/raw/hta_monograph_GJAC1008.html`; Bookshelf Ch.2 copy `S/research/raw/efiplus_s2.html` |
| HTA-F1 | HTA Figure 1 (patient flow), image | `S/research/raw/hta_fig1.png` |
| AA | Archer L et al. Age Ageing 2024;53(3):afae057, PMC10960070 (CC BY) | `S/efalls/fulltext.md` |
| SUPP | AA supplement | `S/efalls/supp.md`; OMML equations in `S/research/supp_fulltext_with_math.txt` |
| P21 | eFI+ protocol posted on the NIHR award page, doc_date 2021-11-03, doc_ver "V21". Footer still reads "Protocol version 2.0 02/09/2019". URL https://njl-admin.nihr.ac.uk/document/download/2037976 | `S/research/raw/protocol_eFIplus_V21_2021-11-03.txt` (pdftotext of `.pdf`) |
| P20 | Same, doc_date 2019-09-10, doc_ver "V20". URL https://njl-admin.nihr.ac.uk/document/download/2030514 | `S/research/raw/protocol_eFIplus_V20_2019-09-10.txt` |
| FA | NIHR Funding & Awards API record for NIHR127905: https://fundingawards.nihr.ac.uk/api/project?id=NIHR127905 (page https://fundingawards.nihr.ac.uk/award/NIHR127905) | `S/research/raw/fa_api_NIHR127905.json` (single-line JSON; cite by key) |
| NO | NIHR open data record | `S/research/raw/nihr_open.json` (single-line JSON) |
| CTG | ClinicalTrials.gov NCT04113174 (last update 2019-10-23). https://clinicaltrials.gov/study/NCT04113174 | `S/research/raw/ctgov_NCT04113174.json` (single-line JSON; cite by key) |
| RSM1 | HTA Report Supplementary Material 1 = TRIPOD-Cluster checklist (page numbers only, no methods content) | `S/research/raw/nihr_supp1.txt` |

**Note on files that did not help.** `S/research/raw/fa_NIHR127905` and `S/research/raw/nihr_prog_127905.html` are both saved "Page not found" pages; they contain no award content. The award content was recovered from the FA API instead.

---

## 1. Data sources, data quality, eligibility

### 1.1 Data sources [EXPLICIT]
- **SAIL**: "linked primary care, ED attendance, hospital admissions, outpatient data, social care, Welsh Care Homes data set, and Office for National Statistics mortality data" (HTA l.339). SAIL primary care uses Read 2 (HTA l.405; AA l.72).
- **Connected Bradford (CB)**: "linked primary, secondary, community and social care data from 900,000 patients" (HTA l.344). AA says around 800,000 residents and 86 practices, using SNOMED-CT (AA l.74).
- **Data quality.** The only data-quality text is generic. The HTA says SAIL and CB "have been established as sources of data for high quality, reproducible research" and that SAIL was commended in international benchmarking (HTA l.333).
  - No cleaning rules, plausibility ranges or harmonisation checks are described in the HTA. [NOT FOUND]
  - AA's TRIPOD-Cluster item 7a ("data preparation...cleaning, harmonisation, linkage, and quality checks") is left blank (SUPP l.27).
  - RSM1 points item 7a to monograph p.42 (RSM1 l.97-102). That page is Chapter 2 "Code lists"/"Sample size", which contains no cleaning rules. [INFERRED from page mapping]
- **Protocol plan vs what was done.** P21 and P20 planned SAIL, ResearchOne (TPP) and the Leeds Data Model (LDM) (P21 l.318-377). Connected Bradford replaced both ResearchOne and LDM. The HTA does not state why ResearchOne/LDM were dropped. COVID-19 disruption, loss of on-campus data access, and a "wholescale change to the Connected Bradford data model, requiring rebuilding of all data sets and models" are described (HTA l.1723). [EXPLICIT; the reason for the switch is NOT FOUND]
- **Who analysed what.** Model 2 (falls) was developed and internally validated in SAIL by the University of Birmingham team (LA, MH, RR). External validation was in CB by the University of Leeds team (SR, AC) (HTA Table 2, l.368-374).

### 1.2 Index date, look-back, follow-up window
- **Index date and follow-up** [EXPLICIT, HTA l.360]: "Patients ≥ 65 years and registered with a SAIL or Connected Bradford practice on 1 April 2018 are eligible for study... Outcomes were assessed over a 12-month period, to 31 March 2019."
  - This matches protocol P21 l.381-383, word for word apart from tense.
- **Look-back** [EXPLICIT, HTA l.360]: "Lookback period included the complete primary care her [sic, = EHR], to first registration, and linked data."
  - There is **no minimum registration or look-back duration requirement**. [NOT FOUND in HTA, AA, P20, P21 or CTG]
- **External validation index date conflicts** [CONTRADICTION]:
  - HTA l.360 (1 April 2018 for both databases) and HTA l.558 ("As of 1 April 2018, 88,947 met the inclusion criteria") say 1 April 2018.
  - HTA-F1 (CB box: "Registered with Connected Bradford Database on 1 January 2019: 86 GP, 229,572 patients") and AA l.80 ("registered with a Connected Bradford general practice on 1 January 2019") say 1 January 2019.
  - The CB 12-month window end date is not stated for the 1 January 2019 index. [NOT FOUND] The window would be 1 Jan–31 Dec 2019 if the 12-month rule is applied. [INFERRED]
  - The CB economic analysis says "the data set covers the 2018–9 financial year" (HTA l.1629), which matches a 1 April 2018 index for the HTA's general CB cohort. [EXPLICIT]
- **Two different CB cohorts** [EXPLICIT numbers; INFERRED interpretation]:
  - HTA CB cohort: n = 88,947 with 3,079 fall/fracture events, 84 practices (HTA l.459, l.558, Table 8 l.569 and l.676, HTA-F1).
  - eFalls external validation cohort (AA l.100, l.144; HTA Table 14 l.903): n = 81,685 with 2,389 events, 76 practices.
  - For eFalls, use the AA/HTA Table 14 cohort (81,685; 2,389 events) as the external validation target. The HTA l.459 statement about "88,947 participants with 3079 events" belongs to the monograph's generic CB extraction, not the eFalls validation run. [INFERRED]

### 1.3 Inclusion, exclusion, age, linkage
- **Inclusion criteria as implemented** [EXPLICIT, HTA-F1]: "Patients aged ≥ 65 years" and "Valid sex recorded". HTA l.558 adds "having a valid sex recorded within their practice record" for CB.
  - Neither SAIL nor CB shows any exclusions: "Exclusions (0)" in both HTA-F1 boxes.
  - The SAIL flow is 3,100,549 registered patients (455 GPs) → 660,417 eligible → 660,417 analysed. [EXPLICIT, HTA-F1]
- **Linkage requirement**: AA l.80 says "Eligible patients were defined the same way in both populations as those with linked data, aged ≥65 years." [EXPLICIT]
  - The HTA does not state a linkage requirement for SAIL or CB. It only requires linked data for CARE75+ (HTA l.361). [NOT FOUND for SAIL/CB in HTA]
- **Age calculation** [EXPLICIT, HTA l.556, SAIL]: "No data were missing on the continuous predictor of age, although this measurement was based on week of birth to maintain anonymity and thus is approximate."
  - Age reference date: the index date (age at 1 April 2018) is not stated verbatim. [NOT FOUND] It is implied by the eligibility wording. [INFERRED]
  - Integer vs fractional age is not stated. [NOT FOUND] The worked example uses integer years ("aged 89 years"; HTA l.1053).
- **Age cap / top-coding** (e.g. 95): [NOT FOUND] in HTA, AA, SUPP, P20, P21 or CTG. The only "95" is protocol background text about deficit prevalence from age 65 to 95 (P21 l.177), which is not an analysis rule.
- **Care home residents**:
  - General rule [EXPLICIT, HTA l.499]: people who cannot experience an outcome (e.g. "a permanent nursing home resident or a person with an existing home care package" for a new home care package) "were excluded from the development cohort for that specific model".
  - For the falls model, no exclusion applies. HTA-F1 shows 0 exclusions, and the SAIL n is 660,417 for the falls, care-home and mortality models alike (HTA l.555, l.1185). So baseline care-home residents were **included** in eFalls development. [INFERRED]
  - The care-home model also reports n = 660,417 (l.1185), which contradicts the l.499 rule if baseline nursing-home residents had been excluded from that model. [CONTRADICTION, INFERRED]
  - P21 added the rule (P21 l.546-549). It is absent from P20, which appears to have been posted in September 2019 before the rule was added. [EXPLICIT diff]
- **Deaths during follow-up** [EXPLICIT, HTA l.506; AA l.313]: "Individuals that died during the 12-month period were retained in the risk set for the whole 12 months". Death is not an exclusion, and censoring was not modelled.
- **Deregistration / loss to follow-up**: no deregistration rule. [NOT FOUND] The HTA says "Censoring was not accounted for... negligible evidence of dropout during our study period in Connected Bradford (< 2% had none of our outcome measures and no further events in the database)" (HTA l.498). [EXPLICIT]
- **CARE75+ (additional-predictor analysis only)** [EXPLICIT]: "all CARE75+ participants with linked primary care data were considered eligible" (HTA l.361).
  - Numbers conflict: "linked primary care data for 267 participants" (l.551) vs n = 252 analysed (l.1161; Table 8 l.569). [CONTRADICTION/attrition not explained]

---

## 2. Outcome

### 2.1 Definition [EXPLICIT]
- HTA l.381 (identical to AA l.84): "any (one or more) ED attendance or hospital admission for a fall or fracture (as an indicator of an injurious fall) within 12 months of their baseline predictor assessment."
- **SAIL datasets**: HTA l.383 says "SAIL includes existing linkage to the ED data set and Patient Episode Database for Wales". P21 l.436-438 names them "Emergency Department Dataset (EDDS) and Patient Episode Database for Wales (PEDW)".
- **CB**: "linked secondary care data" (HTA l.385; AA l.84). The HTA economic chapter shows CB secondary care is HES-based (HES APC + A&E) at Bradford Royal Infirmary (HTA l.1602-1605). That is for costing, not necessarily the outcome extraction. [EXPLICIT for costing; INFERRED relevance]
- **Missing outcome = no event** [EXPLICIT, HTA l.495]: "where the outcome ... was not coded within the 12-month follow-up, it was assumed that no event took place."
- **Window**: 1 April 2018 to 31 March 2019 for SAIL (HTA l.360). That span is 365 days if the index day is included. Whether events on the index day count is not stated. [NOT FOUND; INFERRED inclusive from "to 31 March 2019"]

### 2.2 Items needed for reproduction but not reported [NOT FOUND in HTA, AA, SUPP, P20, P21, CTG]
- Diagnosis position(s) searched in PEDW (primary only vs any position) and in EDDS.
- Coding system of EDDS/ED diagnoses in 2018–19, and whether ED records were mapped to ICD-10.
- Whether hospital admissions were restricted to emergency admission method.
- Episode vs spell level.
- Whether an ED attendance followed by admission counts once (irrelevant for a binary "any" outcome, but relevant to data processing).
- How CB secondary-care data were coded (ICD-10 in HES APC; ED coding in HES A&E/ECDS).
- Code matching rule (prefix match for 3-character codes).
- Observations [INFERRED]:
  - The list mixes 3-character codes with one 4-character code (T14.2), which implies prefix matching for the 3-character codes.
  - W00–W19 are ICD-10 Chapter XX external-cause codes. These are normally recorded in secondary diagnosis positions of admitted-patient records, so a primary-position-only search would capture few W-code events. An "any position" search is therefore likely but unconfirmed.
- CTG describes the outcome only as "coded evidence of ED attendeance/hospitalisation with fall/fracture in the routine dataset" (CTG protocolSection.outcomesModule.primaryOutcomes[2].description). [EXPLICIT]

### 2.3 Appendix 1 code list vs AA Table S2.3 [EXPLICIT, computed]
- HTA Appendix 1 (HTA l.1884-1920) and SUPP Table S2.3 (SUPP l.86-120) were compared with `diff` on the 31 code rows. They are **textually identical** (codes and descriptions), so there are **no differences**.
- The 31 codes:
  - W00, W01, W02, W03, W04, W05, W06, W07, W08, W09, W10, W11, W12, W13, W14, W15, W16, W17, W18, W19
  - M80
  - S22, S32, S42, S52, S72, S82
  - T08, T10, T12, T14.2
- Observations [INFERRED, not a stated difference]:
  - Not listed: S02 (skull/facial fracture), S12 (neck), S62 (wrist/hand), S92 (foot), T02 (multiple body regions), X59 (unspecified factor), M84.4 (pathological fracture NEC) and M48.5 (collapsed vertebra).
  - Because of these omissions, the outcome is narrower than a generic "any fracture" definition.
  - The HTA title of Appendix 1 says "Fall outcome International Classification of Diseases code list" (l.1884). AA says "ICD10 codes used to define ED attendance or hospital admission for a fall or fracture" (SUPP l.86).

---

## 3. Predictors

### 3.1 Identification [EXPLICIT]
- **Candidates**: the 36 eFI variables plus **44 additional variables**, making 80 candidates. The additional variables came from a systematic review (van der Windt & Riley; NIHR SPCR ESWG) and targeted scoping reviews (HTA l.400-401). They had to be available in routine primary-care EHR data (HTA l.401).
- **Stakeholder review**: the London Frailty Network (LFN) reviewed candidates. It "also provided clinical steering on how the time variability of predictor variables should be incorporated into the modelling" (HTA l.403).
  - PPI input reviewed "time constraints that could be applied to variables that may resolve" (HTA l.324). The resulting constraints are **not reported**. [NOT FOUND]
  - The protocol called this body an "Expert Reference Panel (ERP)" (P21 l.411-419).
- **Not candidates** [EXPLICIT/INFERRED]:
  - Frailty category was "Not considered as a candidate predictor during model development" (AA l.186).
  - Deprivation (WIMD/IMD) is listed among planned demographic candidates in the sample-size text (HTA l.424; P21 l.514). No deprivation term appears in Table 15 (HTA l.953-1038). WIMD was used only for IECV grouping and subgroup analysis. It was either not a candidate or was dropped; this is not stated. [INFERRED]

### 3.2 Code lists [EXPLICIT, HTA l.405-412]
- **Ontologies**: English systems migrated CTV3 → SNOMED; SAIL uses Read 2. The conversion was "from CTV3 into SNOMED and Read 2" (l.405).
- **Two-stage process** (l.406-408):
  1. Collate code lists for all 80 predictors "from project partners and online sources (listed in the technical specification document)".
  2. Convert them with the "NHS technology reference update distribution mapping files" (TRUD; ref. 28 at l.1841).
- **Method** (l.409-410): both mapping directions (e.g. CTV3→SNOMED and SNOMED→CTV3) were used and matches collated. This gave "matches for over 99% of the codes in the CTV3 list"; the remainder were searched by hand using descriptions.
- **Clinical check** (l.411): clinicians (AC/SC/KW) checked for codes under the wrong predictor and for obvious missing codes. The result was "approximately 7500 SNOMED codes, 9500 CTV3 codes and 7500 Read 2 codes, covering the 80 candidate predictor variables."
- **Availability** (l.412): "available on request from the corresponding author".
- **Direction conflict** [CONTRADICTION]: AA l.94 says SNOMED-CT groups were built first "with back transformation to Read version 2 using NHS England Technology Reference Update Distribution lists, with clinical validation of all new predictor variables in both SNOMED-CT and Read version 2." The HTA says CTV3 was the master list, converted to both SNOMED and Read 2.
- The "technical specification document" is not public. [NOT FOUND] See `S/research/efi_codelists.md` §3 for the public DynAIRx Baseline2 SNOMED/CTV3 list that most likely derives from this work (not re-done here).

### 3.3 Look-back windows
- **Global rule** [EXPLICIT]: the complete primary-care EHR to first registration (HTA l.360).
- **Per-predictor time constraints**: none stated in HTA, AA, SUPP, P20, P21 or CTG. [NOT FOUND]
  - The LFN/PPI steering on time variability (l.324, l.403) shows constraints were considered, but the rules are not given. [EXPLICIT that they were considered; the rules are NOT FOUND]
  - The prior note `S/research/efi_codelists.md` §5 shows identical SAIL predictor counts between eFalls and eFI2. This suggests the eFI2 5-year/age rules apply. [INFERRED there]
- **Polypharmacy** [EXPLICIT]:
  - HTA Table 16 (l.1052) and Table 34 (l.2076): "a count of unique drugs prescribed over the 120 prior [sic: days] to index date (excluding non-drug chapters of the BNF e.g. bandages)".
  - SUPP Box S3.1 footnote (SUPP l.322) adds: "Unique BNF sub-sub-chapters. Combinations of >1 drug from a sub-sub-chapter only counted once towards the total." This footnote is absent from the HTA tables.
  - Repeat vs acute prescriptions, issue vs prescription date, and the list of "non-drug chapters" are not specified. [NOT FOUND]
  - Missing prescriptions are assumed not prescribed (HTA l.495).
  - Functional form: ln[(P+1)/10] (HTA l.945).
  - Medicines were deliberately not entered as individual drug-class predictors (HTA l.1726; AA l.315).
- **BMI** [EXPLICIT]:
  - Categories: underweight / normal / overweight / obese / missing (HTA Table 14 l.920-925; Table 15 l.960-965; reference = overweight).
  - Cut-points conflict within the HTA:
    - Table 16 (l.1052): "weight groups are defined by standard BMI cut-offs".
    - Table 34 (l.2076, same as SUPP Box S3.1 l.318): "underweight if BMI < 18.5; normal weight if 18.5 ≤ BMI < 24.9; obese if BMI ≥ 40". Overweight is not defined, and [24.9, 25) is left uncovered.
  - BMI source (recorded value vs computed from height/weight vs BMI codes), look-back window and selection rule (latest before index) are not reported. [NOT FOUND]
- **Smoking** [EXPLICIT]:
  - Descriptive categories: Never / Ex / Current, with **no missing category** reported for SAIL or CB (HTA Table 8 l.592-595; Table 14 l.926-929).
  - Model: "Ex/never" is the reference and "Current" is the only coefficient (HTA Table 15 l.966-968).
  - Missing smoking was apparently folded into never/ex. [INFERRED from the absence of a missing row; NOT FOUND as a rule]
  - Data-quality flag: CB records Ex = 18 (0.0%) vs Never 72.5% (HTA l.594). The CB home-care cohort shows Never 99.5% and Current 0.5% (HTA l.715-717), against Current 27.4% in HTA Table 8 (l.595). Smoking derivation in CB is therefore unstable across CB extractions. [EXPLICIT numbers; INFERRED interpretation]
  - Codes, window and latest-record rule are not stated. [NOT FOUND]
- **Alcohol** [EXPLICIT]:
  - Categories: harmful drinking, higher-risk drinking, lower-risk drinking (reference), previous higher-risk/harmful drinking, zero alcohol, missing (AA Table 1 l.180-185; SUPP Table S3.2 l.242-248).
  - Missing is 97.3% in SAIL (HTA l.602).
  - The HTA misprints "Higher-risk drinking" twice in Tables 8, 9, 14, 15, 20, 21 and 35 (e.g. l.598-599, l.932-933, l.971-972, l.2103-2104). AA Table 1 prints "Lower risk drinking" with the same counts (11,231 SAIL; AA l.182 vs HTA l.599). This confirms the second "Higher-risk drinking" row in the HTA is Lower-risk drinking. [EXPLICIT comparison]
  - Source codes, window and AUDIT/units thresholds are not stated in HTA/AA. [NOT FOUND] See efi_codelists.md §7.1 for Baseline2 unit thresholds [INFERRED there].
- **"Additional predictors" section (HTA l.550-552)** [EXPLICIT]:
  - This is NOT about EHR predictors. It covers CARE75+ clinical measures (gait speed, grip strength) tested on top of eFalls.
  - Method: logistic regression with outcome "falls or fragility fractures recorded in primary care within 12 months" of each CARE75+ assessment (baseline, 6, 12, 24, 48 months). The "eFalls LP as an offset term" is included, with a random intercept per participant. C-statistic and AIC were compared for null vs grip, gait, and grip+gait models.
  - Results: n = 252, 864 assessments, 18 events (l.1161-1172). Null C = 0.621, AIC 174.9; grip C = 0.653, AIC 175.3.
  - Note: this outcome differs from eFalls (it uses primary-care-recorded falls/fragility fractures).

---

## 4. Missing data [EXPLICIT, HTA l.495; AA l.110]
- Method: missing indicators, with missing data allocated to "missing" groups for categorical variables (Sperrin 2020; Sisk 2023 refs at HTA l.1848, l.1850).
- Absent codes: "where deficits were not coded for a patient, the assumption was made that no deficit was present".
  - Absent outcome = no event. No prescription = none written.
  - The same handling was used in development and validation, "reflect[ing] how missing data would be handled at deployment".
- Categorical variables with a missing level in the model: BMI (missing 34.9% in SAIL) and alcohol (97.3%). Smoking has none (see §3.3). Age had no missing values; sex was required valid (eligibility).
- **Protocol deviation** [EXPLICIT]: P21 l.524-526 (and P20, CTG) planned "multiple imputation and Rubin's rules, under a missing at random assumption, including outcome in the imputation model ... accounting for practice clustering". This was not done.

---

## 5. Modelling and validation

### 5.1 Model type and scope [EXPLICIT]
- Logistic regression for 12-month risk. The flexible parametric survival / Fine–Gray competing-risk plan was abandoned after discussion (HTA l.498, l.505-506; P21 l.536-541).
- No censoring handling. Deaths are retained as non-events unless the event occurred first (l.506).
- No interaction terms: "We did not consider interaction terms within our models." (l.504).
- Machine learning was rejected (l.500-501).
- **Clustering**: "Clustering of participants by GP was not accounted for at model development, but predictive performance was assessed by practice." (l.504; AA l.114).
  - Protocol deviation: P21 l.540-541 planned practice random effects.

### 5.2 Fractional polynomials [EXPLICIT]
- HTA l.504: continuous predictors (age, polypharmacy) were modelled "using either standard polynomials up to degree three (models developed in Connected Bradford) or fractional polynomials up to FP2 (models developed within SAIL), with best functional forms in the presence of all model predictors."
- HTA l.945: forms were assessed "in the presence of all candidate predictors (prior to variable selection through use of the LASSO penalty term, λ)".
  - Result: linear age; polypharmacy "best modelled with a natural log transformation"; entered as "age in years, and ln[(polypharmacy + 1)/10]" (HTA Appendix 4 Fig. 20, l.1960-1961).
  - The transformed terms were then LASSO candidates (l.504).
- FP selection details (closed test vs deviance comparison, alpha, scaling/shift) are not stated. [NOT FOUND] See `S/research/stata_methods.md` §4 for Stata `fp`/`mfp` defaults (not re-done).
- Software: Stata 17 for all University of Birmingham analyses (HTA l.502; AA l.106).

### 5.3 LASSO [EXPLICIT unless stated]
- Objective: "addition of a L1 penalty term to the typical binary cross-entropy loss function used for logistic regression" (HTA l.500).
- Lambda: "The strength of the penalty term is set using 10-fold cross-validation to determine the model resulting in the optimal log-likelihood" (l.500). l.504: "lambda, was chosen to minimise the cross-validation function on 10-fold cross-validation".
  - This is λ_min, not λ_1SE. [EXPLICIT for min]
- Reported values: λ = 0.000123; "The LASSO regression retained 74 predictors" (HTA l.947). But the HTA Scientific Summary says 75 (l.2624), and AA l.191 says 75. [CONTRADICTION]
  - Table 15 has exactly 75 non-reference, non-constant coefficients (computed from `S/research/raw/hta_table15_efalls_coefficients.csv`). The correct figure is **75 parameters**. [EXPLICIT computed]
- Omitted candidates: 10 variables were omitted by LASSO (SUPP Table S3.1 footnote, l.210): anxiety, CKD, dyspnoea, environment problems, heart valve disease, IHD, problems managing finances, shopping problems, toileting problems, TIA. That leaves 62 binary terms, consistent with 72 binary candidates. [EXPLICIT + computed]
- Standardisation: the cross-validation plot shows "standardised model coefficients" against λ (HTA l.1963). Whether reported coefficients are on the original scale is not stated; they are presented as usable in the equation. [EXPLICIT caption; INFERRED original scale]
- **Software**: Stata 17 (UoB). The specific command, e.g. `lasso logit`, is not named. [INFERRED; see stata_methods.md]
- **CV fold assignment, seed, λ grid**: [NOT FOUND]
- **Reference categories, a priori or data-driven**: not stated. [NOT FOUND]
  - Evidence that labels may be data-driven: the care-home model Table 21 marks "Ex" smoking, both "Higher-risk drinking" rows and alcohol "Missing" as "Reference" (HTA l.1254-1264), while "Never" gets a coefficient. So "Reference" rows in the HTA LASSO tables may include levels zeroed by LASSO, not just a priori base levels. [INFERRED]
  - For eFalls, "Ex/never" is a combined reference (l.967). This is either an a priori collapse or a zeroed "ex" coefficient. [UNRESOLVED]
- **Post-selection refit**: an unpenalised logistic model with only LASSO-selected variables gives ORs "not intended for use in practice" (HTA l.948). The prediction equation uses the penalised coefficients.

### 5.4 Model stability [EXPLICIT, HTA l.509-514; results l.1057-1067]
- **What was assessed**: variation in the LP distribution, individual predictions, calibration curves, MAPE, and classification to high/low risk.
- **Bootstrap models**: "derived following identical development processes to the original model, applied in a sample of the same size, sampled with replacement from the model development population."
- **Outputs**:
  1. Prediction distribution instability plots: LP density of each bootstrap model, both within its bootstrap sample and in the original data.
  2. Prediction instability plots.
  3. Calibration instability plots.
  4. MAPE: "mean absolute difference between the bootstrap model predictions and the original model prediction for an individual".
- **Number of bootstrap models for stability**: not stated separately. It is presumably the internal-validation bootstraps. [INFERRED]
- **Results (new vs AA)** [EXPLICIT]:
  - Mean MAPE 0.00196 (95% CI 0.00195 to 0.00197); median 0.00117 (LQ–UQ 0.000618 to 0.00236).
  - "Classification instability": at the 0.10 threshold, individuals near 0.1 changed class "in up to 80% of bootstrap models". At 0.25, "up to 60%", with changes for original probabilities between 0.1 and 0.4.
  - AA mentions only probability-distribution and calibration instability plots (AA l.118).

### 5.5 Performance definitions [EXPLICIT, HTA l.515-535]
- **Calibration slope**: logistic regression of outcome on LP as the only covariate; the coefficient is the slope (ideal 1).
- **CITL**: logistic regression with LP as offset; the intercept is CITL (ideal 0).
- **O/E**: "observed risk across the population divided by the mean predicted risk" (ideal 1).
- **Calibration plots**: groups by "20ths of outcome risk" plus loess smooth curves.
- **C-statistic**: concordance / AUROC.
- **Clinical utility**: net benefit via decision curves. Thresholds were fixed a priori without seeing risk distributions (l.525). Model 2 range: 10–25% (Table 7, l.532).
  - Across practices: median difference in standardised net benefit (SNB) between the model and treat-all/treat-none/next-best, for thresholds 0 to 0.5, with 50%, 80% and 90% bands (l.542; Fig. 26 l.1995).
  - The "next best" strategy switches at a threshold of 0.06.
- **Test accuracy tables**: TP/FP/TN/FN per 1000 at thresholds 0.10–0.25 (HTA Tables 32/37).

### 5.6 Internal validation: bootstrap
- **HTA l.539** [EXPLICIT]: "bootstrapping with 50 samples". Optimism = the performance of each bootstrap-developed model in its own bootstrap sample minus its performance in the original data. Estimates are adjusted by the average optimism.
- **AA l.118** [EXPLICIT]: "bootstrapping with 25 samples (chosen for computational efficiency due to the use of big data)". [CONTRADICTION: 50 vs 25]
  - HTA Appendix 4 Table 31 (l.1968-1984) reproduces AA Table S3.3's numbers exactly (e.g. C 0.7434, optimism 0.0004245, adjusted 0.7430).
  - "50 samples" is also stated for the Leeds CB models (home care l.834; mortality l.1501).
  - Conclusion: the HTA Chapter 2 figure is a pooled/generic statement, and the eFalls (Birmingham, SAIL) internal validation used **25** bootstraps as reported in AA. [INFERRED]
- What is repeated in each bootstrap: the full development process (FP selection? LASSO CV?). HTA says "the model developed in each bootstrap sample" (l.539) and stability uses "identical development processes" (l.510). Whether FP forms were re-selected per bootstrap is not stated. [NOT FOUND]
- Optimism-adjusted apparent performance (Table 31): slope 1.0057, CITL −0.0003, O/E 0.9998, C 0.7430.

### 5.7 Apparent performance across practices [EXPLICIT, HTA l.541-543; AA l.122]
- Clusters were defined by GP practice. Estimates were plotted against SEs (funnel plots), with overlaid calibration and decision curves per GP.
- Practices with fewer than 10 events were omitted from visualisations (Fig. 25-27 captions l.1993-1997; AA l.122 says this was "to preserve anonymity").
- **Meta-analysis**: "random-effects meta-analysis with restricted maximum likelihood (REML) estimation and a Hartung-Knapp-Sidik-Jonkman (HKSJ) variance correction" (l.543).
  - Scales: "The calibration slope, CITL and O/E were pooled across practices on their original scales, while the C-statistic was pooled on the logit scale" (l.543). τ² and 95% PIs are reported.
  - Caution [INFERRED]: Snell et al. 2018 (the cited ref. 46) recommends the log O/E scale. The HTA explicitly says the O/E was pooled on the original scale; AA l.122 says only "on appropriate scales".
- **Results** (Table 17 / AA Table 2): pooled slope 0.99, CITL 0.154, O/E 1.19, C 0.72.

### 5.8 Internal–external cross-validation (IECV) [EXPLICIT, HTA l.545-546; AA l.126]
- **SAIL groups**: "fifths of rank by the Welsh Index of Multiple Deprivation (WIMD, 2019 version)" (l.545). The Figure 28 caption adds "WIMD group 6 refers to those missing WIMD information" (l.1999).
  - So there were **6 IECV groups**: 5 quintiles plus missing WIMD (19.7%). [EXPLICIT caption; INFERRED count]
- **Each cycle**: the whole development process was repeated on all-but-one group and applied to the omitted group. Calibration, discrimination and NB were computed, then pooled with random-effects meta-analysis "on appropriate scales" with HKSJ CIs (l.546).
- CB IECV (home care model) used geographical groups. It was described as "fivefold" at l.379 but "4 groups" at l.834 [CONTRADICTION; not eFalls].
- **Results**: slope 0.99 (0.75–1.22), CITL −0.13, O/E 0.88, C 0.72 (0.68–0.76) (Table 17 l.1079-1098). Performance was poorest in the missing-WIMD group (l.1105).

### 5.9 External validation [EXPLICIT, HTA l.548-549; AA l.130-132]
- Performed by a different team at a different institution (Leeds).
- Analysed overall (no clustering), then per GP (same meta-analysis approach), then in subgroups:
  - (W)IMD quintiles "with a sixth group for those missing IMD information"
  - eFI frailty group (fit ≤0.12; mild >0.12–0.24; moderate >0.24–0.36; severe >0.36; HTA l.939)
  - sex
  - BMI category (underweight, normal, overweight, obese, missing)
- **Software**: R 4.3.1 for Leeds analyses (HTA l.502) vs R 4.2.3 for external validation and recalibration (AA l.106; HTA l.1153). [CONTRADICTION, minor]
- **Results** (Table 18): overall slope 1.248, CITL −0.931, O/E 0.432, C 0.825. Pooled across GPs: slope 1.203, CITL −0.874, O/E 0.431, C 0.816.

### 5.10 Recalibration [EXPLICIT, HTA l.1145-1153; SUPP l.121-128]
- **Method**: logistic regression in CB, `ln(P/(1−P)) = α_recal + β_recal·LP_eFalls` ("adjusted only the intercept and slope"). Fitted with glm in R 4.2.3. Only apparent performance was assessed.
- **α, β values**: the HTA HTML leaves the probability formula blank in Tables 16 and 34 (raw HTML `hta_monograph_GJAC1008.html` offsets ~286243 and ~583350: "Probability of falls/fractures within 12 months = ,"). So **the HTA does not print α/β**.
  - Table 34 still reports 0.106 for the worked example (l.2077).
  - AA Box S3.2 OMML gives `exp(-0.423+1.25*LP)/(1+exp(-0.423+1.25*LP))` (`S/research/supp_fulltext_with_math.txt` l.1273, l.1295).
- **Numerical checks** [EXPLICIT computation, python3.13]:
  - Box S3.1 example LP (rounded coefficients) = −1.3678 → p = 0.2030. Recalibrated with (−0.423, 1.25): p = 0.1060. Both match the printed values.
  - The same example with exact Table 15 coefficients: LP = −5.954459 − 0.303708 + 0.0415506·89 + 0.3296295·ln(0.9) + 0.4896735 + 0.0849676 + 0.1038111 + 0.3803626 + 0.1276254 = **−1.4085 → p = 0.1965**.
    - The gap from −1.368 comes almost entirely from rounding the age coefficient to 0.042 (0.00445 × 89 = 0.040).
  - HTA Table 35 coefficients are exactly 1.21 × Table 15 coefficients (73 pairs, ratio range 1.2099999–1.2100000). The constant −7.25089539 = −0.046 + 1.21 × (−5.954459). So Table 35 implies **α = −0.046, β = 1.21** on the Table 15 LP.
    - With Table 35 the worked example gives p = **0.148**, not the 0.106 printed in the adjacent Table 34. [CONTRADICTION]
    - β = 1.25 agrees with the overall EV slope 1.248 (Table 18). β = 1.21 is close to, but not equal to, the pooled-across-GP slope 1.203. [INFERRED, unresolved]
- **Box S3.1 intercept/sex** [EXPLICIT computation]: −5.954459 + (−0.303708) = −6.258167. The Box S3.1 intercept −6.258 is therefore the **female** baseline. So Box S3.1's "– 0.304 (if male)" should read "+0.304 (if male)" to agree with Table S3.2/Table 15 (Male = reference, Female = −0.303708). [INFERRED from arithmetic]

---

## 6. Sample size

### 6.1 Development
- **Protocol / HTA narrative** [EXPLICIT]:
  - Anticipated minimum 8,064 fall/fracture events in SAIL: 7.4% of 72,000 moderate frailty + 11.4% of 24,000 severe frailty (HTA l.416, l.419; P21 l.469-482).
  - Capacity of up to 108 parameters at 20 events per parameter (EPP) (l.422).
  - Riley 2019 with Nagelkerke R² 0.15 gives 7.5 EPP at 3% outcome proportion, 11.5 at 9% (fall/fracture) and 15 at 15% (l.423).
- **HTA Table 3** (l.425-438) [EXPLICIT], falls: estimated incidence 4.8%, 108 parameters.
  - R² 0.15 → n = 19,706 (946 events), EPP 8.76.
  - R² 0.05 ("original eFI") → n = 60,209 (2,891), EPP 26.76.
- **AA Table S2.1** (SUPP l.67-73) [EXPLICIT], 90 parameters:
  - 0.15 → 13,867 (666)
  - 0.05 → 50,174 (2,409)
  - 0.049 → 50,927 (2,445), the one adopted in AA l.98.
  - [CONTRADICTION: 108 vs 90 parameters; different required n]
- **Verification** (`S/research/verify_ss.py`, re-run 2026-09-14) [EXPLICIT computed]:
  - HTA Table 3 falls rows reproduce exactly with Riley criterion 1 (S = 0.9) using max R²_CS rounded to 0.32: 19,705.1 → 19,706 and 60,208.4 → 60,209.
  - AA row 2 (50,174) reproduces with p = 90 and max R² 0.32. AA row 1 (13,867) implies p = 76, not 90. AA row 3 (50,927) does not reproduce (51,207 with max R² 0.32).
- **Available data**: 660,417 participants, 32,097 events (HTA l.555).

### 6.2 External validation
- Original plan: at least 100 events and 100 non-events, ideally 200 (HTA l.440; P21 l.518-522).
- **Riley 2021 criteria** (HTA Table 4 l.444-457 = SUPP Table S2.2 l.74-84) [EXPLICIT], all at prevalence 0.048:

| Criterion | Assumptions | Required n (events) |
|---|---|---|
| O/E | – | 7,625 (366) |
| Calibration slope | LP mean −3.30, variance 0.690, skew 0.5, kurtosis 3 (development values: skew 0.499, kurtosis 2.943; nearest `sknor` pair, most conservative) | 10,882 (523) |
| C-statistic | C 0.743 | 2,027 (98) |
| SNB at 10% | sens 0.34, spec 0.91 | 2,289 (110) |
| SNB at 25% | sens 0.04, spec 0.99 | 519 (25) |
| **Overall** | – | **10,882 (523)** |

- The prior `S/research/verify_valss.py` re-ran these: O/E gave 7,645 vs 7,625 printed; the C and SNB rows match.
- **Available CB data**: HTA l.459 says 88,947 with 3,079 events. The eFalls validation actually had 81,685 with 2,389 events (AA l.100). Both exceed the requirement.

---

## 7. Data sharing, registration, award record

- **HTA Data-sharing statement** (l.1786-1787) [EXPLICIT]:
  - Model equations are available for research use.
  - "Code lists used to define variables are available on reasonable request from the corresponding author. Any unauthorised use or distribution for commercial purposes is forbidden."
  - The models may be licensed to EHR suppliers, risk-stratification vendors and policy/commissioning users. There is a possible "eFI+ revenue share distribution agreement".
- **AA data availability**: code lists "available on reasonable request" (AA l.367). The SUPP version omits "reasonable" (SUPP l.3). HTA l.412: "available on request".
- **Ethics / governance**: CB REC 18/YH/0200 and 22/EM/0127; SAIL projects approved by the IGRP (HTA l.1789).
  - P21 cites University of Leeds SoMREC MREC 19-013 (P21 l.29-30). P20 says separate ethics approval was not required (P20 l.29-31).
- **Award (FA JSON)** [EXPLICIT]:
  - Call 18/50 "Frail older people in primary care"; £545,625.65; University of Leeds; CI Andrew Clegg.
  - Start 2019-10-01, end 2023-02-28 (P21 l.10-11 says end 31/05/2022).
  - Protocol docs: V20 dated 2019-09-10 and V21 dated 2021-11-03.
  - Linked report doi 10.3310/GJAC1008, published 2026-08-06.
  - HTA article history: "The protocol was agreed in October 2019" (HTA l.297).
- **P20 → P21 changes** (diff) [EXPLICIT]:
  - The population restriction to moderate/severe frailty (eFI 0.24–0.36 / >0.36) was removed. P21 l.381 is ≥65 only.
  - The outcome-impossibility / nursing-home transfer paragraph was added (P21 l.546-549).
  - Frailty-subgroup examination was moved to the model-improvement and decision-model sections.
  - Unchanged: missing data by MI; global shrinkage or elastic net; flexible parametric survival; practice random effects; splines/FP; IPD-MA and cross-validation across areas/practices; recalibration "where necessary" (P21 l.524-580).
- **ClinicalTrials.gov NCT04113174** (CTG JSON) [EXPLICIT]:
  - First posted 2019-10-02; last update 2019-10-23; overallStatus UNKNOWN.
  - Eligibility still restricted to moderate/severe frailty and "Registered with a ResearchOne, SAIL or LDM practice on 1st April 2018" (eligibilityModule).
  - Missing data by MI; sample-size text as P20.
  - The registry was never updated to the final design (all ≥65; SAIL + Connected Bradford; LASSO; missing indicators). [EXPLICIT for dates; INFERRED "never updated"]

---

## 8. Contradictions register (eFalls-relevant)

| # | Item | Source A | Source B |
|---|---|---|---|
| C1 | Sex term | Table 15/S3.2: Male ref, Female −0.303708, constant −5.954459 (HTA l.958-959, l.1038) | Box S3.1/HTA Tables 16 & 34: −6.258 intercept and "– 0.304 (if male)" (HTA l.1051, l.2075). The arithmetic shows −6.258 = female baseline, so the male sign is wrong |
| C2 | Recalibration | AA Box S3.2: α −0.423, β 1.25; worked example 0.106 (supp_fulltext_with_math l.1273, l.1295; HTA l.2077) | HTA Table 35 = 1.21 × Table 15 with constant −7.25089539, implying α −0.046, β 1.21; worked example 0.148 |
| C3 | Bootstrap samples (eFalls internal validation) | AA l.118: 25 | HTA l.539: 50 (generic; the Leeds models used 50) |
| C4 | CB external validation index date | HTA l.360, l.558: 1 April 2018 | HTA-F1 and AA l.80: 1 January 2019 |
| C5 | CB validation cohort | HTA l.459, l.558, Fig 1: 88,947 / 3,079 events / 84 practices | AA l.100, l.144 and HTA Table 14: 81,685 / 2,389 events / 76 practices |
| C6 | Predictors retained | HTA l.947: 74 | HTA Sci. Summary l.2624 and AA l.191: 75; Table 15 has 75 parameters |
| C7 | Development sample-size parameters | HTA Table 3: 108 parameters, n 19,706/60,209 | AA S2.1: 90 parameters, n 13,867/50,174/50,927 |
| C8 | BMI cut-offs | HTA Table 16 l.1052: "standard BMI cut-offs" | HTA Table 34 l.2076 / Box S3.1: obese ≥40, normal <24.9, overweight undefined |
| C9 | Alcohol reference label | AA Table 1 / S3.2: "Lower risk drinking" | HTA Tables 8/14/15/35: "Higher-risk drinking" printed twice |
| C10 | Code-list conversion direction | HTA l.405-410: CTV3 master → SNOMED and Read 2 via TRUD (both directions) | AA l.94: SNOMED groups → back-transformed to Read 2 |
| C11 | R version (Leeds / external validation) | HTA l.502: R 4.3.1 | AA l.106, HTA l.1153: R 4.2.3 |
| C12 | SAIL geography | HTA l.555, l.1185, l.1542: "455 GPs from across South Wales" | AA l.144: "455 general practices across Wales" |
| C13 | Missing data plan vs practice | P20/P21/CTG: multiple imputation, MAR, Rubin's rules | HTA l.495 / AA l.110: missing indicators; uncoded = absent |
| C14 | Penalisation plan vs practice | P21 l.551-553: global bootstrap shrinkage, or elastic net if selecting variables | HTA l.500: LASSO (elastic net considered and rejected) |
| C15 | Care-home exclusion rule | HTA l.499: people unable to experience the outcome excluded per model | HTA-F1: 0 exclusions; care-home model n = 660,417 (l.1185) |
| C16 | CARE75+ n | HTA l.551: 267 with linked primary care data | HTA l.1161 / Table 8: 252 |
| C17 | Polypharmacy unit | HTA Tables 16/34: "unique drugs" | SUPP footnote l.322: unique BNF sub-sub-chapters |

---

## 9. Unresolved (not answerable from these sources)

1. **Outcome extraction details.** Diagnosis positions (PEDW/EDDS), ED coding system, admission-method restriction, index-day inclusion and prefix-matching rule. *Resolution*: ask the corresponding author (L. Archer, Birmingham) or the SAIL project team (A. Akbari / J. Hollinghurst) for the outcome specification. For a Meuhedet reproduction, pre-specify and run a sensitivity analysis (primary-only vs any position).
2. **Per-predictor look-back/time constraints and the "technical specification document"**. *Resolution*: request from A. Clegg / S. Relton. Meanwhile use the Baseline2 rules as a documented proxy (see efi_codelists.md).
3. **BMI definitions.** Cut-points, source and window. *Resolution*: author query. Pre-specify WHO standard cut-points (18.5/25/30) as primary and the printed "≥40 obese" as sensitivity.
4. **Smoking.** Handling of missing values (apparently folded into never/ex) and conflict rules. *Resolution*: author query.
5. **Polypharmacy.** Repeat vs acute prescriptions, the list of non-drug BNF chapters, and drug vs sub-sub-chapter counting. *Resolution*: author query; follow the Box S3.1 footnote (sub-sub-chapters).
6. **LASSO implementation.** Stata command, CV fold seed, λ grid, whether FP was re-selected inside bootstraps and IECV, and whether "Reference" rows are a priori or LASSO-zeroed. *Resolution*: author query; the λ value (0.000123) and coefficients are published, so re-derivation is not needed to apply the model.
7. **Recalibration.** Which of (α −0.423, β 1.25) and (α −0.046, β 1.21) is correct. *Resolution*: author/journal query. AA Box S3.2 is consistent with its own worked example and the overall EV slope; Table 35 is internally consistent as a coefficient table but not with Table 34.
8. **Bootstrap number for eFalls** (25 vs 50) and the number of bootstrap models used for stability/MAPE/classification instability.
9. **CB external validation cohort.** Index date (1 Apr 2018 vs 1 Jan 2019) and why 81,685 vs 88,947.
10. **Age.** Top-coding and age reference date: not reported.
11. **Protocol.** V20 and V21 are both labelled "version 2.0 02/09/2019", yet their content differs; the later amendment history is not reported.

---

## 10. Files added in this round
- `S/research/raw/protocol_eFIplus_V21_2021-11-03.pdf` + `.txt`
  - Public PDF from NIHR njl-admin, 228,944 bytes, 21 pages.
  - Retrieved through WebFetch's saved binary. Text extracted with pdftotext; nothing executed.
- `S/research/raw/protocol_eFIplus_V20_2019-09-10.pdf` + `.txt` (174 KB, 21 pages; same retrieval method).
- `S/research/raw/fa_api_NIHR127905.json` (NIHR Funding & Awards public API record).
- `S/research/work/fa_page.html`, `S/research/work/fa_bundle.js`: public site shell and JS, used only to locate the API URL (grep only, not executed).
