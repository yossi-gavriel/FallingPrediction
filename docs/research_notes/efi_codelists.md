# eFI (2016), eFI2 (2025) and the eFI+ code lists as the source of eFalls candidate predictors

Research notes compiled 2026-09-14. Evidence labels used throughout:
- **[EXPLICIT]** = stated verbatim in the cited source (quote given, <= 40 words)
- **[INFERRED]** = my inference from the evidence (reasoning given)
- **[NOT FOUND]** = searched, not located in any authoritative public source

Local copies of everything read are under
`<research workspace (not distributed)>/research/raw/efi/`
(text extractions `*.txt` were produced by parsing JATS XML / DOCX XML; no downloaded file was executed).

---

## 0. Headline findings

1. **The original eFI (Clegg 2016) code lists are NOT publicly available.** Neither the paper nor its supplement (prevalence tables only) lists codes. The 36 deficit names are public; per-deficit definitions, look-back windows and the polypharmacy time window are not. [EXPLICIT for absence in these documents; NOT FOUND elsewhere]
2. **The original eFI had no time constraints: deficits are effectively "ever recorded".** This is stated by the same group in the eFI2 paper, the eFI+ HTA monograph and the SAIL eFI validation. [EXPLICIT, in secondary sources by the eFI authors]
3. **eFI2** = Best K, Shuweihdi F, Bazo Alvarez JC, Relton S, ... Clegg A. Age Ageing 2025;54(4):afaf077, doi:10.1093/ageing/afaf077, PMC11957239. Its supplement Appendix 1 publicly lists **79 candidate predictors with look-back (5-year) and age (>=18 / >=55) rules**. The official code lists are "available from the corresponding author" and are not public. [EXPLICIT]
4. **A public copy of the eFI+/eFI2 SNOMED CT code list exists**: `Baseline2_Codelist.csv` in the DynAIRx GCAF repository (GitHub `DynAIRx/GCAF_DynAIRx`, BSD-3 licence). The same file is in the CC-BY supplement of Aslam A ... Clegg A, Buchan I, Relton SD, BMC Med Res Methodol 2025;25:138. Its README describes it as the SNOMED basis of the eFI2 project (Relton and Clegg), defining "80 long-term conditions" for the four eFI+ outcomes. The file has 7,555 rows (7,285 unique SNOMED concepts) in 77 deficit groups, with columns: SNOMED, CTV3, provenance, time constraint (years), age limit and other instructions. [EXPLICIT]
5. **All 72 binary eFalls variables in Table S3.1 match a Baseline2 deficit name one-to-one.** The 5 remaining Baseline2 groups are the lifestyle variables (Alcohol, Body mass index, Obesity, Smoker current, Smoker ex). [EXPLICIT: I compared the names computationally]
6. **eFalls and eFI2 appear to use the same derived SAIL predictor dataset.** For 30 of 32 shared binary predictors, the SAIL counts in eFI2 Table 2 equal the eFalls Table S3.1 counts exactly (e.g. Housebound 74,800; Memory concerns 25,552; Requirement for care 23,427). The alcohol categories also match exactly. The two mismatches are Hypotension/syncope and Social vulnerability. [EXPLICIT numbers; INFERRED shared derivation] So the eFI2 Appendix 1 rules and the Baseline2 columns are the best public evidence for how the eFalls variables were defined. However, the eFalls papers never state these time windows. [INFERRED]
7. **The polypharmacy definitions differ.** Original eFI: ">=5 prescribed medications, BNF chapters 1-15", with no window stated. eFI2: count of BNF sub-subchapters (level 3) prescribed in the previous **90 days**, grouped 0-4/5-9/10+. eFalls: count of unique BNF sub-sub-chapters over **120 days**, excluding non-drug chapters, entered as a continuous FP term. [EXPLICIT]
8. The HTA monograph (doi:10.3310/GJAC1008) says the eFI+ code list covered **80 candidate predictors with about 7,500 SNOMED, 9,500 CTV3 and 7,500 Read 2 codes**. It is available on request and was described in a "technical specification document" that is not public. The Read 2 version (used in SAIL) was not found publicly. [EXPLICIT / NOT FOUND]

---

## 1. Original eFI (Clegg et al. 2016)

**Citation:** Clegg A, Bates C, Young J, Ryan R, Nichols L, Teale EA, Mohammed MA, Parry J, Marshall T. Development and validation of an electronic frailty index using routine primary care electronic health record data. Age Ageing 2016;45(3):353-60. doi:10.1093/ageing/afw039. PMID 26944937. PMC4846793. Licence CC BY-NC 4.0.
- Full text (JATS): https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4846793/fullTextXML (local: raw/efi/efi2016.xml, efi2016.txt)
- Supplement: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4846793/supplementaryFiles (local: raw/efi/efi2016_supp/supp_afw039_afw039supp.docx; text raw/efi/efi2016_supp.txt)
- Correction (acknowledgement only; no change to definitions): Age Ageing 2017/2018;47(2):319, doi:10.1093/ageing/afx001, PMC6016616.

### 1.1 The 36 deficits (Box 1) [EXPLICIT]
Activity limitation; Anaemia and haematinic deficiency; Arthritis; Atrial fibrillation; Cerebrovascular disease; Chronic kidney disease; Diabetes; Dizziness; Dyspnoea; Falls; Foot problems; Fragility fracture; Hearing impairment; Heart failure; Heart valve disease; Housebound; Hypertension; Hypotension/syncope; Ischaemic heart disease; Memory and cognitive problems; Mobility and transfer problems; Osteoporosis; Parkinsonism and tremor; Peptic ulcer; Peripheral vascular disease; Polypharmacy; Requirement for care; Respiratory disease; Skin ulcer; Sleep disturbance; Social vulnerability; Thyroid disease; Urinary incontinence; Urinary system disease; Visual impairment; Weight loss and anorexia.
- Source: PMC4846793, Box 1 "List of 36 deficits contained in the eFI."
- The supplement "Web table 1" gives the prevalence of each deficit in the ResearchOne development/internal cohorts and the THIN external cohort. It lists names only, with no codes or definitions. [EXPLICIT]

### 1.2 How the deficits were built [EXPLICIT]
- "We ran a series of searches to identify Clinical Terms Version 3 (CTV3) Read codes for inclusion." (PMC4846793, Methods)
- "Cut-points for numeric data were defined by reported laboratory reference ranges and international standard diagnostic criteria." (PMC4846793, Methods). The cut-points themselves are not given. [NOT FOUND]
- Inclusion criteria: "only deficits with a population prevalence >0.5%, a positive regression coefficient and an r2 value of >0.30 were included." (PMC4846793)
- "Thirty-six deficits, constructed using 2,171 CTV3 codes, met our inclusion criteria (Box 1). Polypharmacy was defined on the basis of the presence of ≥5 prescribed medications, using chapters 1–15 of the British National Formulary." (PMC4846793, Results)
- "The CTV3 codes were mapped to 36 deficits containing 1,574 corresponding Read 2 codes (Box 1)." (PMC4846793). Mapping used the HSCIC TRUD standard mapping table.
- IP: "Copyright and database rights for the eFI are held by the University of Leeds." (PMC4846793, Intellectual property)

### 1.3 Code-list availability for the original eFI
| Candidate public source | Result | URL |
|---|---|---|
| 2016 paper and supplement | Names and prevalences only; **no codes** [EXPLICIT absence] | PMC4846793 (above) |
| NHS England eFI page | "It is made up of 36 deficits comprising around 2,000 Read codes." No list and no download [EXPLICIT] | https://www.england.nhs.uk/ourwork/clinical-policy/older-people/frailty/efi/ |
| Vision (Cegedim) help page and user PDF | "The deficits are based on Read codes or Read codes with values." It mentions only the score code 38QI and that Hypertension uses 24-h average BP codes 246V/246W. No list, window or polypharmacy rule [EXPLICIT] | http://help.visionhealth.co.uk/Vision_Consultation_Manager_Help_Centre/Content/ConMgr/76292.htm ; https://help.visionhealth.co.uk/PDFs/Outcome%20Manager/Electronic%20Frailty%20Index%20and%20Stratification%20Tool.pdf |
| NHS Dorset "Appendix D – Read Codes Using the eFI" | Only codes for *recording* the frailty index or category (38QI./XabYS; 2Jd0-2Jd2), not deficit codes [EXPLICIT] | https://nhsdorset.nhs.uk/Downloads/aboutus/finance/11J_0230%20Enhanced%20Frailty/Appendix%20D%20-%20Enhanced%20Frailty%20Read%20Codes.pdf |
| OpenCodelists | Only NHSD refset `EFI_COD` (codes for recording an eFI score) and frailty diagnosis refsets. **No deficit lists** [EXPLICIT from search] | https://www.opencodelists.org/codelist/nhsd-primary-care-domain-refsets/efi_cod/20211221/ |
| HDR UK Phenotype Library (API search "frail", "eFI", "Frailty") | Only PH2517 = EFI_COD score codes. No deficit phenotypes [EXPLICIT from API] | https://phenotypes.healthdatagateway.org/api/v1/phenotypes/?search=eFI |
| Hollinghurst 2019 SAIL eFI validation supplement | Deficit prevalence in SAIL and **care-home Read codes only**; no deficit code lists [EXPLICIT] | PMC6814149, doi:10.1093/ageing/afz110 |
| TPP / EMIS / ClinRisk public docs | [NOT FOUND] (no public deficit code list located) | - |
| Aslam et al. 2025 (DynAIRx) | States the original eFI "contains 1691 SNOMED codes" and exists in SNOMED, CTV2 and CTV3. **The original eFI list itself is not in their repository**; only the eFI2 basis list is (see section 3) [EXPLICIT] | PMC12102889 |

**Conclusion:** the Read v2 (1,574) and CTV3 (2,171) lists for the original eFI are **NOT publicly available** [NOT FOUND].

### 1.4 Look-back windows in the original eFI
- The 2016 paper states no per-deficit window. [NOT FOUND in PMC4846793]
- The eFI authors later described the original eFI as having no time constraints [EXPLICIT]:
  - eFI2 paper (PMC11957239, Abstract): "The original eFI has some limitations such as equal weighting of deficit variables, lack of time constraints on variables known to resolve and definition of frailty category cut-points."
  - HTA monograph (https://www.journalslibrary.nihr.ac.uk/hta/GJAC1008): the eFI has "cumulative adding of deficits that are assumed not to improve or resolve."
  - Hollinghurst 2019 (PMC6814149): "As the eFI uses cumulative deficits, people only transition to a higher frailty status."
  - BJGP editorial 2025 (PMC12117642): "lack of time constraints (constraints were not applied to variables that could resolve over time)".
- **[INFERRED]** In the original eFI every deficit is effectively "ever recorded" before the index date. The polypharmacy window was not stated; see 1.5.

### 1.5 Original eFI polypharmacy definition
- [EXPLICIT] ">=5 prescribed medications, using chapters 1–15 of the British National Formulary" (PMC4846793).
- Time window (e.g. last 12 months or current repeats), counting unit (drug vs BNF section) and repeat vs acute handling: **[NOT FOUND]** in the paper, supplement, NHS England page or Vision documentation.

---

## 2. eFI2 (Best et al. 2025)

**Citation:** Best K, Shuweihdi F, Alvarez JCB, Relton S, Avgerinou C, Nimmons D, Petersen I, Pujades-Rodriguez M, Conroy SP, Walters K, West RM, Clegg A. Development and external validation of the electronic frailty index 2 using routine primary care electronic health record data. Age Ageing 2025;54(4):afaf077. doi:10.1093/ageing/afaf077. PMID 40163740. PMC11957239. CC BY-NC 4.0.
- Full text: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11957239/fullTextXML (local raw/efi/efi2.xml, efi2.txt)
- Supplement DOCX `aa_24_1289_file002_afaf077.docx`: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11957239/supplementaryFiles (local raw/efi/efi2_supp/..., text raw/efi/efi2_supp.txt)
- Related commentary: Nimmons D, Clegg A, Walters K. BJGP 2025;75(755):249-250, doi:10.3399/bjgp25X742473, PMC12117642. Romero-Ortuno R, Keevil VL, editorial, Age Ageing 2025;54(5):afaf111 (https://academic.oup.com/ageing/article/54/5/afaf111/8124755). Editor's view, Age Ageing 2025, doi:10.1093/ageing/afaf127 (not read).
- Programme page: https://arc-yh.nihr.ac.uk/research/projects/development-and-national-implementation-of-efi-2/ (no code lists).

### 2.1 Design facts [EXPLICIT]
- Development cohort: Connected Bradford (78,760); external validation: SAIL Wales (660,417). Both aged >=65 on 1 April 2018. The SAIL N is identical to the eFalls development cohort.
- "A lookback period included the complete primary care EHR from first registration." (PMC11957239, Participants)
- "Candidate predictors included the 36 deficit variables from the original eFI in addition to those identified by a systematic review..." (Predictors)
- "SNOMED CT code lists for each candidate predictor were developed from the original eFI, National Health Service (NHS) Quality and Outcomes Framework incentivisation scheme code lists (2018/19), [21] and from established code lists." (Predictors)
- Mapping from SNOMED CT to Read v2 used TRUD tables; failed mappings were searched manually on athena.ohdsi.org; some SNOMED codes could not be mapped and "were therefore not included". (Predictors)
- "There were 79 candidate predictor variables (Supplementary Materials, Appendix 1)... All predictors were derived based on the presence of a relevant SNOMED CT or READ code."
- "Several predictors had additional inclusion rules, such as occurrence only in the previous 5 years (Supplementary Materials, Appendix 1)."
- Final model: 36 predictors (Cox, positivity constraint). "managing finances" was excluded for prevalence <0.05%.
- Missing data: "For binary predictors, patients that did not have a relevant code recorded in their primary care EHRs were assumed not to have the corresponding condition." "Missing lifestyle data for BMI, smoking status and alcohol intake were represented by a ‘missing’ category"
- Polypharmacy (Table 2 footnote): "Number of medications from different BNF sub-subchapters (level 3) prescribed in previous 90 days." Categories 0–4 (reference) / 5–9 / 10+.
- Categorical reference levels: alcohol "zero intake", BMI "Recommended BMI", smoking "none, ex, or missing", polypharmacy "0–4 medications".
- BMI cut-offs for underweight/recommended/overweight/obese: **[NOT FOUND]** in the eFI2 paper or supplement.
- Frailty cut-points: robust/mild 0.0857, mild/moderate 0.1624, moderate/severe 0.2392 (PMC11957239). BJGP gives them rounded as 0.09/0.16/0.24.
- Data availability: "The eFI2 model equation and associated code lists used to define variables are available from the corresponding author for research use." BJGP adds: "The eFI2 is licensed to suppliers of UK primary care electronic health record systems at no cost..."

### 2.2 eFI2 Appendix 1: 79 candidate predictors and time/age rules [EXPLICIT, verbatim from the supplement DOCX]
Names are as printed. The stray " or " inside some names (e.g. "Chronic or kidney disease", "Requirement or for care") is present in the DOCX XML itself and is probably an editing artefact.

Abdominal pain †; Activity limitation; Alcohol (harmful, high, low, previous harmful/higher, zero, missing) †; Anaemia or haematinic deficiency ‡; Anxiety †; Asthma *; Atrial fibrillation *; Back pain ‡; Body mass Index (underweight, recommended, overweight, obese, missing); Bone disease; Cancer; Chronic or kidney disease; Cognitive impairment ¥; COPD; Dementia ¥; Depression †; Diabetes mellitus ‡; Dizziness ‡; Dressing or grooming problems; Dyspnoea ‡; Environment problems; Faecal incontinence; Falls (history of) †α; Fatigue †; Foot problems ‡; Fracture α; Fragility fracture α; Gardening problems; General or mental health †; Headache †; Hearing impairment ‡; Heart failure; Heart or valve disease; Housebound ‡α; Hypertension ‡; Hypotension or syncope; Inflammatory arthritis; Inflammatory bowel disease; Ischaemic heart disease; Liver problems; Meal preparation problems; Memory concerns †¥; Mobility problems; Mono/hemiparesis; Motor Neurone Disease; Musculoskeletal problems †; Osteoarthritis; Osteoporosis; Palliative care; Parkinsonism or tremor; Peptic or ulcer disease †; Peripheral neuropathy; Peripheral or vascular disease; Polypharmacy (0-4, 5-9, 10+); Problems managing finance; Problems with cleaning and domestic tasks; Requirement or for care; Respiratory disease ‡; Seizures; Self-harm †; Severe mental illness; Skin ulcer; Sleep problems †; Smoking status (current, none/ex/missing); Social vulnerability ‡; Stress †; Stroke; Telephone problems; Thyroid problems; Toileting problems; Transient ischaemic attack; Travelling problems; Unable to manage medications; Urinary incontinence α; Urinary or system disease ‡; Visual impairment; Washing and bathing problems; Weakness †; Weight loss ‡.

Footnotes (verbatim or condensed; each quote <= 40 words):
- "*All codes within deficit are restricted to those occurring after Age 18 years"
- "α All codes within deficit are restricted to those occurring after Age 55 years (with the exception of Atrial Fibrillation where only ‘Lone atrial fibrillation’ has this constraint)"
- "† All codes within deficit are restricted to within 5 years of index date"
- "‡Some codes within deficit are restricted to within 5 years of index date:" The listed exceptions are:
  - Back pain: no 5-year limit for chronic back pain, spondylosis or disc prolapse.
  - Diabetes: 5-year limit only for drug-induced diabetes, diabetes in remission and gestational diabetes.
  - Dizziness: no limit for vestibular disorders or vertigo syndromes.
  - Dyspnoea: no limit for MRC grade 4.
  - Foot problems: 5-year limit for corns only.
  - Hearing: 5-year limit only for hearing loss, deteriorating or impaired hearing.
  - Housebound: "within 5 years and >=55 for home visit (not chronic or acute)".
  - Hypotension: no limit for Parkinsonism with orthostatic hypotension, chronic, idiopathic or NOS hypotension.
  - Respiratory disease: 5-year limit only for chronic cough and PE/infarction.
  - Social vulnerability: 5-year limit only for widowed, bereavement and loneliness.
  - Urinary system disease: 5-year limit except TURP, urinary catheter, cystitis, detrusor instability, retention and BPH.
  - Weight loss: 5-year limit only for appetite loss.
- "Environment problems include codes relating to housing and social environment"
- "Polypharmacy is defined according to the number of medications prescribed from separate BNF sub chapters in the last 90 days"
- "¥Memory concerns code resolves if cognitive impairment is recorded. Cognitive impairment code resolves if dementia is recorded."
- "Supplementary Box 1" (predictor definitions), cited in the eFI2 main text, is **not present** in the published supplement DOCX. [NOT FOUND]

### 2.3 eFI2 final 36 predictors and coefficients (Table 3) [EXPLICIT]
Activity limitation 0.15284; Alcohol harmful 0.23107; Alcohol missing 0.13175; Alcohol previous harmful/higher 1.36434; Atrial fibrillation 0.13025; Cancer 0.2406; Cognitive impairment 0.10985; COPD 0.11683; Dementia 0.41715; Dressing or grooming problems 0.05422; Environment problems 0.11886; Falls (history of) 0.62743; Fracture 0.07353; Fragility fracture 0.17425; Heart failure 0.11086; Housebound 0.33254; Hypotension or syncope 0.18253; Liver problems 0.23787; Medication management problems 0.32125; Memory concerns 0.11915; Mobility problems 0.46836; Motor neuron disease 0.35347; BMI missing 0.25318; BMI underweight 0.4417; Palliative care 0.5145; Parkinsonism or tremor 0.03537; Peptic ulcer disease 0.05427; PVD 0.02672; Polypharmacy 5–9 0.32301; Polypharmacy 10+ 0.50801; Requirement for care 0.21428; Respiratory disease 0.01049; Seizures 0.02885; Self-harm 0.00900; Skin ulcer 0.21935; Smoker current 0.10291; Social vulnerability 0.23585; Stroke 0.10565; TIA 0.02305; Weight loss 0.19256. Sum of possible coefficients = 8.429; 1-year baseline hazard 0.0151 (Appendix 3).

---

## 3. The public eFI+/eFI2 SNOMED code list ("Baseline2")

### 3.1 Provenance [EXPLICIT]
- **Paper:** Aslam A, Walker L, Abaho M, Cant H, O’Connell M, Abuzour AS, Hama L, Schofield P, Mair FS, Ruddle RA, Popoola O, Sperrin M, Tsang JY, Shantsila E, Gabbay M, Clegg A, Woodall AA, Buchan I, Relton SD. An automation framework for clinical codelist development validated with UK data from patients with multiple long-term conditions. BMC Med Res Methodol 2025;25:138. doi:10.1186/s12874-025-02541-1. PMC12102889. CC BY 4.0.
  - "The eFI2 will be released imminently and contains 7556 SNOMED codes. Both of these are available in multiple ontologies (SNOMED, CTV2, and CTV3)"
  - "we begin using two codelists for MLTCs that have been clinical validated previously: eFI2 [44] and SERENDIP [45]"
  - "Both GitHub Repository of framework and Generated Codelists (along with baselines) are publicly available, and attached as supplementary material with manuscript."
- **Repository:** https://github.com/DynAIRx/GCAF_DynAIRx (org DynAIRx; licence BSD-3-Clause; created 2024-09-17). The file https://github.com/DynAIRx/GCAF_DynAIRx/blob/main/Input/Baselines/Baseline2_Codelist.csv (raw: https://raw.githubusercontent.com/DynAIRx/GCAF_DynAIRx/main/Input/Baselines/Baseline2_Codelist.csv) was last changed in commit 9a1df1fb8523622066b4f31db6f2fbb8f55e0b2f (2024-09-18).
  - README (verbatim): "Sam Relton and Andy Clegg are currently finishing off the eFI2 project which expands upon this initial work to build prediction models. The SNOMED codes here are the basis of that work, used to define 80 long-term conditions"
- The same file is in the BMC supplement: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12102889/supplementaryFiles → `12874_2025_2541_MOESM1_ESM.zip` → `Framework GCAF DynAIRx.zip` → `GCAF_DynAIRx-main/Input/Baselines/Baseline2_Codelist.csv`. **The GitHub and supplement copies are byte-identical.** git blob 88cf1234d9e27b2820f9cb236006d735d82875a0; SHA-256 2e242a5efbd631b3de17738dce86c66e7ac1f78dacf8783fed32555cd3c205b4; 607,055 bytes.
- Local copy: raw/efi/Baseline2_Codelist_github.csv

### 3.2 Corroboration that Baseline2 is the eFI+ programme list [EXPLICIT + INFERRED]
- The HTA monograph (https://www.journalslibrary.nihr.ac.uk/hta/GJAC1008, section "Code lists") says: "The result was a list of approximately 7500 SNOMED codes, 9500 CTV3 codes and 7500 Read 2 codes, covering the 80 candidate predictor variables." Baseline2 has 7,555 rows; Aslam says 7,556 SNOMED codes. Both match "approximately 7500 SNOMED" and "80". [EXPLICIT numbers, INFERRED match]
- All 72 binary eFalls variables in Table S3.1 map to Baseline2 deficit names. The only differences are "and" vs "&" and "Medication management" vs "Medication management problems". [EXPLICIT computed]
- 30/32 shared eFI2/eFalls SAIL counts are identical (section 5). [EXPLICIT computed]
- **Caveats [INFERRED]:**
  - This is a Sept-2024 working snapshot hosted by a third party (DynAIRx), not the corresponding author's official release.
  - It is not stated to be the exact version used for eFalls (published March 2024) or eFI2.
  - Some Baseline2 rules disagree with the published eFI2 Appendix 1 (section 6).
  - IP: eFI rights are held by the University of Leeds, and the eFalls and eFI2 code lists are "on request". The BSD-3 licence on the repo does not clearly settle the rights to use the code content. **Obtain written permission from the corresponding author (A. Clegg, a.p.clegg@leeds.ac.uk) before relying on it for Meuhedet.**

### 3.3 Structure and content [EXPLICIT computed from file]
- Columns: `Deficit, SNOMEDCT_CONCEPTID, CTV3, Provenance, Codedescription, TimeConstraintyears, AgeLimit, Otherinstructions`
- 7,555 rows; 7,285 unique SNOMED IDs; 77 deficit groups.
- Provenance prefixes (rows): blank 2,754; `mar-` 2,633; `efi-` 1,620 (1,590 unique SNOMED); `qof-` 202; `ucl-` 117 (alcohol); `clegg-` 99 (falls); `CALIBER` 41. The meaning of `mar-` is not documented [NOT FOUND]. It might be an eFI+ team member's lists; this is speculation.
- `efi-` tags presumably mark codes inherited from the original eFI [INFERRED]. They also appear on groups that are *not* in the 2016 Box 1: abdominal pain, back pain, bathing, environment, faecal, fatigue, grooming, headache, IBD, liver, mono/hemiparesis, MSK, peripheral neuropathy, seizures, toilet, unable_meds, weakness. That is about 17 groups. This fits the HTA statement that "19 potentially relevant deficit variables (e.g. back pain, mood problems and self-care problems) were also identified but not included in the eFI". [INFERRED]
- Original eFI deficits that were split or renamed in the eFI+ list [INFERRED from tags]:
  - Arthritis → Inflammatory arthritis + Osteoarthritis (`efi-arthritis`)
  - Respiratory disease → Respiratory disease + Asthma + COPD (`efi-asthma_copd`, `efi-respiratory`)
  - Mobility and transfer problems → Mobility problems (+9 `efi-mobility` codes placed in Housebound)
  - Cerebrovascular disease → Stroke + TIA (no `efi-` tags; QOF-sourced)
  - Memory and cognitive problems → Memory concerns + Cognitive impairment + Dementia (no `efi-` tags)
  - Renamed: Sleep disturbance → Sleep problems; Thyroid disease → Thyroid problems; Peptic ulcer → Peptic ulcer disease; Weight loss and anorexia → Weight loss; Diabetes → Diabetes mellitus
- Rules in the file:
  - `TimeConstraintyears` is '5' or blank.
  - `AgeLimit` is '55' (Falls, Fracture, Fragility fracture, some Housebound home-visit codes, lone AF) or '18' (Asthma, Urinary incontinence).
  - Numeric rules include: BP "3 readings ... EVER" for hypertension/hypotension; Barthel "Equal to or less than 18" (Activity limitation); 6CIT ">= 8" (Cognitive impairment); T-score < -2.5 (Osteoporosis); ABPI < 0.95 (PVD); eGFR < 60 and urine protein/ACR thresholds (CKD); TSH <0.36 or >5.5 mU/L (Thyroid); Hb below the sex-specific reference range, with anaemia "resolve if there is a normal lab Hb following initial diagnosis".
  - Hierarchies: "Removed if dementia code added" (Cognitive impairment); "Removed if cognitive impairment or dementia code added" (Memory concerns); "Cannot be current and ex smoker".
  - Alcohol: 5-year window. Numeric units/week: 0 = zero; 1-20 = lower risk; 21-48 = higher risk; 49+ = harmful. Units/day are multiplied by 7.
  - Obesity/BMI: "BMI 30+ = obesity, BMI <18.5 Underweight". The BMI observable 60621009 has a 5-year window.
- **Data-quality defects (important for an auditable reproduction)** [EXPLICIT computed]:
  - 14 SNOMED IDs fail the Verhoeff check digit. All are >=16-digit UK-extension IDs, corrupted by spreadsheet float precision (e.g. `10826410000001000` "Alcohol units consumed per week"). Affected groups: CKD 6, Alcohol 2, Thyroid 2, AF 1, COPD 1, Falls 1, Fracture 1.
  - About 85 CTV3 values lost trailing dots or were numerically coerced (e.g. `1955` for `1955.`, `0.392` for `.392.`). Some entries are 7-character Read v2-style codes (e.g. `1B19.00`). 196 CTV3 cells are blank.
  - Self-harm rows carry the code description in the `Provenance` column.
  - There is **no Read v2 column**, so the SAIL Read 2 version remains non-public.
- DynAIRx derived per-condition CSVs (e.g. `Codelists/Housebound.csv`, `Activity limitation.csv`, `Memory concerns.csv` under Others) have origin `['efi']`. For the 22 groups I checked, they contain exactly the same SNOMED sets as Baseline2 but **drop the time/age columns**. (BMC supplement `Codelists.zip`; GitHub `Output/codelists/`.)

### 3.4 Other public lists with similar names (NOT eFI-derived; do not substitute)
- OpenSAFELY "House bound" https://www.opencodelists.org/codelist/opensafely/house-bound/ (and "No longer housebound")
- NHSD refset FF_COD "Fragility fracture codes" https://www.opencodelists.org/codelist/nhsd-primary-care-domain-refsets/ff_cod/
- NHSD refset FALLS_COD "Falls codes" https://www.opencodelists.org/codelist/nhsd-primary-care-domain-refsets/falls_cod/20250912
- No OpenCodelists or HDR UK hits for "activity limitation", "social vulnerability", "requirement for care", "memory concerns" or "polypharmacy" (searched 2026-09-14).

---

## 4. eFalls statements about predictors and code lists (for cross-reference) [EXPLICIT]
Source: Archer L et al. Age Ageing 2024;53(3):afae057, PMC10960070 (local fulltext.md / supp.md).
- "Candidate predictor variables were constructed by organising individual EHR SNOMED-CT codes into groups, with back transformation to Read version 2 using NHS England Technology Reference Update Distribution lists"
- "candidate predictors in the eFalls model included the 36 components of the eFI [15], supplemented with variables available within routinely collected primary care data. These 44 additional variables..."
- "Code lists used to define variables are available on reasonable request from the corresponding author."
- Box S3.1: "polypharmacy is a count of unique drugs* prescribed over the 120 days prior to index date (excluding non-drug chapters of the BNF e.g., bandages)". Footnote: "Unique BNF sub-sub-chapters. Combinations of >1 drug from a sub-sub-chapter only counted once towards the total."
- Box S3.1 BMI: "underweight if BMI < 18.5; normal weight if 18.5 ≤ BMI < 24.9; obese if BMI ≥ 40". See the contradictions section.
- HTA monograph (GJAC1008): "Lookback period included the complete primary care her [sic], to first registration, and linked data." Also: "The panel also provided clinical steering on how the time variability of predictor variables should be incorporated into the modelling." The eFalls publications do **not** state which time constraints were applied. [NOT FOUND]

---

## 5. Numeric cross-check: eFI2 Table 2 vs eFalls Table S3.1 [EXPLICIT computed]
**SAIL (both N = 660,417):** the counts are identical for Activity limitation 5,646; AF 59,098; Cancer 134,167; Cognitive impairment 6,644; COPD 77,849; Dementia 18,870; Dressing/grooming <10; Environment problems 12,249; Falls 106,839; Fracture 157,004; Fragility fracture 79,033; Heart failure 76,939; Housebound 74,800; Liver problems 3,400; Medication management <10; Memory concerns 25,552; Mobility problems 8,894; MND 251; Palliative care 5,451; Parkinsonism 20,670; PUD 4,941; PVD 44,496; Requirement for care 23,427; Respiratory disease 58,989; Seizures 15,238; Self-harm 1,601; Skin ulcer 75,506; Stroke 48,091; TIA 29,018; Weight loss 49,466 (30/32).
- The alcohol categories are also identical (harmful 4,714; higher 686; lower 11,231; previous 90; zero 1,247; missing 642,449).
- **Mismatches:**
  - Hypotension/syncope: eFI2 48,961 vs eFalls 37,756 (unexplained).
  - Social vulnerability: eFI2 186,887 (28.3%) vs eFalls 20,681 (3.1%). The eFI2 value equals its own "current smoker" row, so it is probably a typesetting error.
  - BMI: eFI2 missing 69,392 (10.5%) vs eFalls 230,507 (34.9%). The category counts also differ, so BMI was derived differently.
  - Current smoking: eFI2 186,887 vs eFalls 86,806.
- **Connected Bradford:** all eFI2 counts are lower than eFalls. That fits eFI2's exclusion of 2,925 people with prior home care or care home, so the samples differ. The eFI2 Table 2 header says N = 81,685 (the eFalls validation N), but the text says 78,760.
- **[INFERRED]** eFalls development (SAIL) and eFI2 external validation used the same derived predictor dataset. The eFI2 Appendix 1 time/age rules therefore probably also apply to the eFalls SAIL variables, except where values differ (hypotension/syncope, BMI, smoking). This is not stated anywhere.
- **[INFERRED]** The near-zero SAIL prevalences of the functional variables in eFalls (Dressing <10, Medication management <10, Washing 377, Toileting 241, Meal preparation 298, Shopping 302, Finances 227) contrast with Bradford (e.g. Dressing 15.0%). This fits the Baseline2 lists being dominated by CTV3 `Xa...` concepts with no Read v2 equivalent, which were dropped in Read 2 mapping (eFI2 says unmappable codes "were therefore not included").
- **[INFERRED]** Bone disease: SAIL 0.8% vs Bradford 9.6%. The Baseline2 "Bone disease" group is osteomalacia + Paget's only (25 codes), which is implausible at 9.6%, so the Bradford implementation may differ.

---

## 6. Contradictions / inconsistencies
1. **Polypharmacy window and unit.**
   - eFI 2016: ">=5 ... BNF chapters 1–15", no window.
   - eFI2: BNF sub-subchapters, previous 90 days, 0–4/5–9/10+.
   - eFalls (Box S3.1 and HTA): unique BNF sub-sub-chapters over 120 days excluding non-drug chapters, continuous ln((P+1)/10).
   - Sources: PMC4846793; PMC11957239; PMC10960070 / GJAC1008.
2. **BMI categories.** eFalls Box S3.1 says "obese if BMI ≥ 40" and "normal 18.5 ≤ BMI < 24.9", which leaves a 24.9–25 gap and implies overweight up to <40. Baseline2 says "BMI 30+ = obesity, BMI <18.5 Underweight". The HTA (GJAC1008) says "standard BMI cut-offs" in one box but repeats "obese if BMI ≥ 40" in Appendix 4. eFI2 does not give cut-offs.
3. **eFI2 Appendix 1 vs Baseline2 rules:**
   - Falls: Appendix 1 says all codes 5y and >=55; Baseline2 has 10/111 codes with 5y and 110 with >=55.
   - Urinary incontinence: Appendix 1 says >=55; Baseline2 says >18.
   - Atrial fibrillation: Appendix 1 says >=18 (plus lone AF >=55); Baseline2 has only lone AF >=55 and no >=18.
   - Hypertension: marked ‡ in Appendix 1 without a rule; Baseline2 has no 5y codes.
   - Housebound: Appendix 1 excludes "chronic or acute" home visits from the 5y/55 rule; Baseline2 applies 5y/55 to "Acute home visit".
   - Environment problems: Appendix 1 says "housing and social environment" codes; Baseline2 contains 8 occupational-therapy codes.
   - Shopping problems is in Baseline2 and eFalls but not in the eFI2 79-item list. Appendix 1 has Gardening, Cleaning/domestic, Telephone and Travelling problems, which Baseline2 does not have as groups; equivalent codes sit inside "Activity limitation".
4. **eFI2 Table 2 internal inconsistencies:** Social vulnerability SAIL equals the current-smoker row (186,887); the development N is given as 81,685 in the header vs 78,760 in the text.
5. **Naming:** eFalls says it includes "the 36 components of the eFI", but 10 of the 2016 names do not appear in eFalls. They were split or renamed in the eFI+ list (section 3.3). Polypharmacy became a continuous count.
6. **Code counts:** eFI 2016 has 2,171 CTV3 / 1,574 Read 2 codes (Clegg 2016); NHS England says "around 2,000 Read codes"; Aslam 2025 says "1691 SNOMED codes". These are different terminologies and versions, not necessarily contradictory.

---

## 7. MAIN TABLE: eFalls variable → most likely public definition source → code list → time window

Key:
- **B2** = DynAIRx `Baseline2_Codelist.csv` (public SNOMED CT + CTV3 with time/age columns): https://github.com/DynAIRx/GCAF_DynAIRx/blob/main/Input/Baselines/Baseline2_Codelist.csv (identical copy in the BMC Med Res Methodol 2025;25:138 supplement, https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12102889/supplementaryFiles).
- The **official** eFalls, eFI2 and eFI+ code lists are available only on request (eFalls PMC10960070; eFI2 PMC11957239; HTA GJAC1008).
- Column "Baseline2 codes" = total rows; rows with `TimeConstraintyears`=5; `AgeLimit` counts; `Otherinstructions` (truncated to 70 chars, xN = number of rows).
- Column "eFI2 App1 rule": † = all codes within 5 years of index; ‡ = some codes within 5 years; * = age >=18; α = age >=55; ¥ = resolution hierarchy. These are the published eFI2 rules; **the eFalls papers do not state any per-variable window** (not found).
- "LASSO-retained" follows eFalls Table S3.1 asterisks (omitted = Anxiety, CKD, Dyspnoea, Environment problems, Heart valve disease, IHD, Problems managing finances, Shopping problems, Toileting problems, TIA).
- "eFI 2016" = named in Clegg 2016 Box 1. A split/rename marked "inferred" is based on B2 provenance tags.

| # | eFalls variable (Table S3.1) | LASSO-retained in eFalls? | Most likely definition source | Baseline2 deficit name | Public code list [B2 = DynAIRx Baseline2] | Baseline2 codes (n; n with 5y window; age limit; other rules) | Code provenance tags in Baseline2 | eFI2 App1 time/age rule (published) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Abdominal pain | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Abdominal pain | B2 | 63; 63; -; - | efi:63 | † all 5y |
| 2 | Activity limitation | Yes | eFI 2016: Activity limitation | Activity limitation | B2 | 49; 0; -; "Equal to or less than 18"x1 | blank:45, efi:4 | none |
| 3 | Anaemia and haematinic deficiency | Yes | eFI 2016: Anaemia and haematinic deficiency | Anaemia & haematinic deficiency | B2 | 120; 80; -; "Anaemia deficit (i.e. all codes) resolve if there is a normal lab Hb f"x119; "Result below lower reference range for sex (if not specified in datase"x1 | blank:120 | ‡ some 5y |
| 4 | Anxiety | No (omitted by LASSO) | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Anxiety | B2 | 29; 29; -; - | blank:29 | † all 5y |
| 5 | Asthma | Yes | eFI 2016: Respiratory disease (split; inferred from efi-asthma_copd tag) | Asthma | B2 | 146; 0; age>=18:146; - | efi:132, qof:11, blank:3 | * age>=18 |
| 6 | Atrial fibrillation | Yes | eFI 2016: Atrial fibrillation | Atrial fibrillation | B2 | 24; 0; age>=55:1; "Code if greater than 0"x4 | efi:15, blank:5, mar:4 | * age>=18; lone AF age>=55 |
| 7 | Back pain | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Back pain | B2 | 50; 31; -; - | efi:50 | ‡ some 5y (exceptions: chronic back pain, spondylosis, disc prolapse) |
| 8 | Bone disease | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Bone disease | B2 | 25; 0; -; - | mar:25 | none |
| 9 | Cancer | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Cancer | B2 | 2136; 0; -; - | blank:1248, mar:867, qof:21 | none |
| 10 | Chronic kidney disease | No (omitted by LASSO) | eFI 2016: Chronic kidney disease | Chronic kidney disease | B2 | 111; 2; -; "Any abnormal result (>150mg/24hr if not specified in dataset)"x4; "Any abnormal result (>50mg/mmol if not specified in dataset)"x4; "Any abnormal result (>20mg/24hr if not specified in dataset)"x2; "Any abnormal result (>3mg/mmol if not specified in dataset)"x4; "If less than 60"x1 | blank:78, mar:18, efi:14, qof:1 | none |
| 11 | Cognitive impairment | Yes | eFI 2016: Memory and cognitive problems (inferred split; no efi- tag) | Cognitive impairment | B2 | 11; 0; -; "Removed if dementia code added"x10; "If equal to or greater than 8"x1 | blank:11 | ¥ resolves if dementia recorded |
| 12 | COPD | Yes | eFI 2016: Respiratory disease (split; inferred from efi-asthma_copd tag) | COPD | B2 | 108; 0; -; "Code if >0"x1; "If greater than 0"x1 | mar:52, efi:27, blank:17, qof:12 | none |
| 13 | Dementia | Yes | eFI 2016: Memory and cognitive problems (inferred split; no efi- tag) | Dementia | B2 | 98; 0; -; - | blank:98 | ¥ (listed) |
| 14 | Depression | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Depression | B2 | 101; 101; -; - | blank:101 | † all 5y |
| 15 | Diabetes mellitus | Yes | eFI 2016: Diabetes | Diabetes mellitus | B2 | 278; 13; -; - | mar:198, efi:64, blank:8, qof:8 | ‡ 5y only drug-induced/remission/gestational |
| 16 | Dizziness | Yes | eFI 2016: Dizziness | Dizziness | B2 | 26; 14; -; - | efi:26 | ‡ 5y except vestibular/vertigo codes |
| 17 | Dressing and grooming problems | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Dressing & grooming problems | B2 | 21; 0; -; - | blank:19, efi:2 | none |
| 18 | Dyspnoea | No (omitted by LASSO) | eFI 2016: Dyspnoea | Dyspnoea | B2 | 8; 7; -; - | efi:8 | ‡ 5y except MRC grade 4 |
| 19 | Environment problems | No (omitted by LASSO) | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Environment problems | B2 | 8; 0; -; - | efi:8 | none (footnote: housing & social environment codes) |
| 20 | Faecal incontinence | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Faecal incontinence | B2 | 5; 0; -; - | efi:5 | none |
| 21 | Falls | Yes | eFI 2016: Falls | Falls | B2; also (NOT eFI-derived) NHSD FALLS_COD refset | 111; 10; age>=55:110; "If greater than 0"x1 | clegg:99, efi:11, blank:1 | †α all 5y; age>=55 |
| 22 | Fatigue | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Fatigue | B2 | 7; 7; -; - | efi:7 | † all 5y |
| 23 | Foot problems | Yes | eFI 2016: Foot problems | Foot problems | B2 | 9; 2; -; - | efi:8, mar:1 | ‡ 5y for corns only |
| 24 | Fracture | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Fracture | B2 | 826; 0; age>=55:826; "Exclude all fracture codes if recorded when age<55"x1 | mar:826 | α age>=55 |
| 25 | Fragility fracture | Yes | eFI 2016: Fragility fracture | Fragility fracture | B2; also (NOT eFI-derived) NHSD FF_COD refset | 365; 0; age>=55:365; - | mar:287, efi:78 | α age>=55 |
| 26 | General mental health | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | General mental health | B2 | 274; 274; -; - | blank:274 | † all 5y |
| 27 | Headache | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Headache | B2 | 21; 21; -; - | efi:21 | † all 5y |
| 28 | Hearing impairment | Yes | eFI 2016: Hearing impairment | Hearing impairment | B2 | 35; 3; -; - | efi:35 | ‡ 5y only hearing loss/deteriorating/impaired |
| 29 | Heart failure | Yes | eFI 2016: Heart failure | Heart Failure | B2 | 74; 0; -; - | efi:37, mar:31, qof:4, blank:2 | none |
| 30 | Heart valve disease | No (omitted by LASSO) | eFI 2016: Heart valve disease | Heart valve disease | B2 | 59; 0; -; - | blank:56, efi:3 | none |
| 31 | Housebound | Yes | eFI 2016: Housebound | Housebound | B2; also (NOT eFI-derived) opensafely/house-bound | 23; 10; age>=55:10; - | efi:23 | ‡α 5y and >=55 for home visit (not chronic or acute) |
| 32 | Hypertension | Yes | eFI 2016: Hypertension | Hypertension | B2 | 86; 0; -; "3 reading equal to or greater than 140 systolic OR 90 diastolic EVER"x3; "3 readings equal to or greater than 90 EVER"x4; "3 readings equal to or greater than 140 EVER"x4; "3 readings equal to or greater than 140 systolic OR 90 diastolic EVER"x1; "Equal to or greater than 85"x3; "Equal to or greater than 135"x3; "Equal to or greater than 135 systolic OR 85 diastolic"x1 | mar:32, efi:24, CALIBER:21, qof:6, blank:3 | ‡ (no specific rule text given) |
| 33 | Hypotension or syncope | Yes | eFI 2016: Hypotension/syncope | Hypotension / syncope | B2 | 41; 35; -; "3 readings less than 90 systolic or 60 diastolic EVER"x5; "3 readings less than 60 EVER"x7; "3 readings less than 90 EVER"x7 | efi:20, CALIBER:20, blank:1 | 5y except PD with orthostatic hypotension, chronic, idiopathic, NOS |
| 34 | Inflammatory arthritis | Yes | eFI 2016: Arthritis (split; inferred from efi-arthritis tag) | Inflammatory arthritis | B2 | 61; 0; -; - | efi:33, mar:26, qof:2 | none |
| 35 | Inflammatory bowel disease | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Inflammatory bowel disease | B2 | 35; 0; -; - | mar:28, efi:7 | none |
| 36 | Ischaemic heart disease | No (omitted by LASSO) | eFI 2016: Ischaemic heart disease | Ischaemic heart disease | B2 | 154; 0; -; - | mar:49, qof:46, blank:30, efi:29 | none |
| 37 | Liver problems | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Liver problems | B2 | 35; 0; -; - | efi:35 | none |
| 38 | Meal preparation problems | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Meal preparation problems | B2 | 18; 0; -; - | blank:18 | none |
| 39 | Medication management | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Medication management problems | B2 | 1; 0; -; - | efi:1 | none (listed as "Unable to manage medications") |
| 40 | Memory concerns | Yes | eFI 2016: Memory and cognitive problems (inferred split; no efi- tag) | Memory concerns | B2 | 34; 34; -; "Removed if cognitive impairment or dementia code added"x34 | blank:34 | †¥ all 5y; resolves if cognitive impairment recorded |
| 41 | Mobility problems | Yes | eFI 2016: Mobility and transfer problems | Mobility problems | B2 | 35; 0; -; - | efi:30, blank:5 | none |
| 42 | Mono or hemiparesis | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Mono/hemiparesis | B2 | 12; 0; -; - | efi:12 | none |
| 43 | Motor neurone disease | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Motor neuron disease | B2 | 5; 0; -; - | mar:5 | none |
| 44 | Musculoskeletal problems | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Musculoskeletal problems | B2 | 77; 77; -; - | efi:77 | † all 5y |
| 45 | Osteoarthritis | Yes | eFI 2016: Arthritis (split; inferred from efi-arthritis tag) | Osteoarthritis | B2 | 20; 0; -; - | efi:20 | none |
| 46 | Osteoporosis | Yes | eFI 2016: Osteoporosis | Osteoporosis | B2 | 67; 0; -; "If less than -2.5 (i.e. a higher negative score, for example -3.5)"x1 | efi:36, mar:21, qof:9, blank:1 | none |
| 47 | Palliative care | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Palliative care | B2 | 57; 0; -; - | blank:45, qof:12 | none |
| 48 | Parkinsonism and tremor | Yes | eFI 2016: Parkinsonism and tremor | Parkinsonism & tremor | B2 | 34; 0; -; - | efi:33, mar:1 | none |
| 49 | Peptic ulcer disease | Yes | eFI 2016: Peptic ulcer | Peptic ulcer disease | B2 | 86; 86; -; - | efi:86 | † all 5y |
| 50 | Peripheral neuropathy | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Peripheral neuropathy | B2 | 84; 0; -; - | efi:64, mar:18, qof:2 | none |
| 51 | Peripheral vascular disease | Yes | eFI 2016: Peripheral vascular disease | Peripheral vascular disease | B2 | 76; 0; -; "If less than 0.95"x1 | mar:58, efi:15, blank:2, qof:1 | none |
| 52 | Problems managing finances | No (omitted by LASSO) | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Problems managing finances | B2 | 7; 0; -; - | blank:7 | none |
| 53 | Requirement for care | Yes | eFI 2016: Requirement for care | Requirement for care | B2 | 12; 0; -; - | efi:11, blank:1 | none |
| 54 | Respiratory disease | Yes | eFI 2016: Respiratory disease | Respiratory disease | B2 | 42; 6; -; - | efi:22, blank:17, mar:2, qof:1 | ‡ 5y only chronic cough, PE/infarction |
| 55 | Seizures | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Seizures | B2 | 124; 0; -; - | efi:52, mar:31, qof:21, blank:20 | none |
| 56 | Self-harm | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Self-harm | B2 | 88; 88; -; - | other:88 | † all 5y |
| 57 | Severe mental illness | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Severe mental illness | B2 | 232; 1; -; - | blank:231, other:1 | none |
| 58 | Shopping problems | No (omitted by LASSO) | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Shopping problems | B2 | 2; 0; -; - | blank:2 | NOT LISTED in eFI2 App1 |
| 59 | Skin ulcer | Yes | eFI 2016: Skin ulcer | Skin ulcer | B2 | 95; 0; -; "Any value"x1; "If greater than 0"x1 | efi:84, mar:9, blank:2 | none |
| 60 | Sleep problems | Yes | eFI 2016: Sleep disturbance | Sleep problems | B2 | 41; 41; -; - | blank:36, efi:5 | † all 5y |
| 61 | Social vulnerability | Yes | eFI 2016: Social vulnerability | Social vulnerability | B2 | 23; 7; -; - | efi:19, blank:4 | ‡ 5y only widowed/bereavement/loneliness |
| 62 | Stress | Yes | eFI+ additional candidate (not in eFI 2016 Box 1); also eFI2 candidate | Stress | B2 | 43; 43; -; - | blank:43 | † all 5y |
| 63 | Stroke | Yes | eFI 2016: Cerebrovascular disease (inferred; no efi- tag) | Stroke | B2 | 86; 0; -; - | blank:52, qof:34 | none |
| 64 | Thyroid problems | Yes | eFI 2016: Thyroid disease | Thyroid problems | B2 | 41; 0; -; "Any abnormal result (if not specified in dataset using Oxford Handbook"x2 | efi:39, blank:2 | none |
| 65 | Toileting problems | No (omitted by LASSO) | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Toileting problems | B2 | 9; 0; -; - | blank:6, efi:3 | none |
| 66 | Transient ischaemic attack | No (omitted by LASSO) | eFI 2016: Cerebrovascular disease (inferred; no efi- tag) | Transient ischaemic attack | B2 | 6; 0; -; - | blank:3, qof:3 | none |
| 67 | Urinary incontinence | Yes | eFI 2016: Urinary incontinence | Urinary incontinence | B2 | 30; 0; age>=18:30; "If code recorded age>18"x30 | efi:30 | α age>=55 |
| 68 | Urinary system disease | Yes | eFI 2016: Urinary system disease | Urinary system disease | B2 | 29; 16; -; - | efi:29 | ‡ 5y except TURP, catheter, cystitis, detrusor instability, retention, BPH |
| 69 | Visual impairment | Yes | eFI 2016: Visual impairment | Visual impairment | B2 | 168; 1; -; - | efi:133, mar:35 | none |
| 70 | Washing and bathing | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Washing & bathing problems | B2 | 18; 0; -; - | blank:12, efi:6 | none |
| 71 | Weakness | Yes | eFI+ additional candidate (not in eFI 2016 Box 1; codes carry efi- provenance tag, so likely from original eFI development code pool - inferred); eFI2 candidate | Weakness | B2 | 13; 13; -; - | efi:13 | † all 5y |
| 72 | Weight loss | Yes | eFI 2016: Weight loss and anorexia | Weight loss | B2 | 12; 6; -; "If equal to or greater than 1"x1 | efi:11, blank:1 | ‡ 5y only appetite loss |
### 7.1 Non-binary eFalls predictors
| eFalls predictor | Public definition source | Public code list | Window / rule (source) |
|---|---|---|---|
| Age (years) | eFalls Box S3.1 | n/a | Continuous, linear (PMC10960070) |
| Sex | eFalls Table S3.2 | n/a | Male reference, female -0.303708 (Table S3.2). Box S3.1 prints "– 0.304 (if male)", which is inconsistent (outside this task's scope) |
| Polypharmacy | eFalls Box S3.1 (not eFI 2016 / eFI2 definitions) | **Drug/BNF code list NOT FOUND** (not in B2) | 120 days before index; unique BNF sub-sub-chapters; excludes non-drug BNF chapters (PMC10960070; GJAC1008). eFI2 uses 90 days; eFI 2016 used >=5 meds, BNF ch 1–15, window not stated |
| BMI category (underweight/normal/overweight/obese/missing) | eFalls Box S3.1; eFI2 candidate "Body mass Index" | B2 "Body mass index" (1 code: 60621009, 5y) + B2 "Obesity" (13 codes, rule "BMI 30+ = obesity, BMI <18.5 Underweight") | eFalls cut-offs contradictory (see section 6.2). BMI window in eFalls NOT FOUND; B2 gives 5y for the BMI observable. Missing handled as a category (eFalls, eFI2) |
| Smoking (current vs ex/never) | eFalls Table S3.2; eFI2 "Smoking status (current, none/ex/missing)" | B2 "Smoker (current)" 37 codes, "Smoker (ex)" 16 codes; rule "Cannot be current and ex smoker" | No window in B2 or eFI2 App1. Conflict-resolution rule (e.g. latest record) NOT FOUND. eFalls groups missing with ex/never (Table S3.2 has no missing level; eFI2 reference "none, ex, or missing") |
| Alcohol (harmful / higher / lower / previous higher-harmful / zero / missing) | eFalls Table S3.2; eFI2 Alcohol † | B2 "Alcohol" 128 rows (ucl- 117, mar- 9), each with category in Otherinstructions | 5 years (eFI2 App1 †; B2 all rows 5y). Numeric units/week: 0 zero; 1–20 lower; 21–48 higher; 49+ harmful; units/day ×7 (B2) |

---

## 8. Assets and availability summary
| Asset | Availability | URL / note |
|---|---|---|
| eFI 2016 paper + supplement (36 names, prevalences) | public | PMC4846793; doi:10.1093/ageing/afw039 |
| eFI 2016 CTV3 code list (2,171 codes) | not_available (not found publicly; University of Leeds IP; licensed to EHR suppliers) | - |
| eFI 2016 Read v2 code list (1,574 codes) | not_available (not found) | - |
| eFI 2016 numeric cut-points, polypharmacy window | not_available (not found) | - |
| eFI2 paper + supplement Appendix 1 (79 candidates, 5y/age rules) | public | PMC11957239; doi:10.1093/ageing/afaf077 |
| eFI2 official model + code lists | on_request (A. Clegg) | PMC11957239 data statement; BJGP PMC12117642 |
| eFI2 "Supplementary Box 1" (predictor definitions) | not_available (cited but absent from supplement) | - |
| eFI+ HTA code list (~7,500 SNOMED / 9,500 CTV3 / 7,500 Read 2; 80 predictors) and "technical specification document" | on_request / not public | https://www.journalslibrary.nihr.ac.uk/hta/GJAC1008 |
| eFalls code lists | on_request | PMC10960070 data statement |
| DynAIRx Baseline2_Codelist.csv (eFI2/eFI+ SNOMED basis, with CTV3 and time/age rules) | public (BSD-3 repo; CC-BY supplement), but code-content IP unclear | https://github.com/DynAIRx/GCAF_DynAIRx/blob/main/Input/Baselines/Baseline2_Codelist.csv |
| DynAIRx derived per-condition CSVs (same SNOMED sets, no time columns) | public | https://github.com/DynAIRx/GCAF_DynAIRx (Output/codelists) and BMC supplement |
| NHSD EFI_COD refset (eFI score recording codes) | public | https://www.opencodelists.org/codelist/nhsd-primary-care-domain-refsets/efi_cod/20211221/ |
| SAIL care-home Read codes (Hollinghurst 2019 Table S2) | public | PMC6814149 supplement |

---

## 9. Unresolved items
1. Original eFI per-deficit code lists (CTV3/Read 2), numeric cut-points and the polypharmacy time window: NOT FOUND publicly. Resolution: request from the University of Leeds / A. Clegg, or from EHR supplier documentation under licence.
2. Whether eFalls applied the eFI2 5-year and age rules. Strongly suggested by identical SAIL counts, but not stated. Resolution: ask the eFalls corresponding author (L. Archer / Birmingham) or A. Clegg for the eFalls technical specification.
3. Exact code-list version used for eFalls vs the Baseline2 snapshot (Sept 2024), and why Baseline2 differs from eFI2 Appendix 1 (Falls, Urinary incontinence, AF, Hypertension, Housebound, Environment problems). Resolution: obtain the official list; until then, document the chosen rule set per variable.
4. The Read 2 version of the eFI+ list (used in SAIL eFalls development): not public.
5. Polypharmacy implementation: the drug dictionary (dm+d → BNF sub-sub-chapter map), which chapters count as "non-drug", repeat vs acute handling, and 120 vs 90 days: NOT FOUND in detail.
6. BMI cut-offs (">= 40 obese" vs standard 30) and the BMI window; smoking conflict rule; handling of "3 readings ... EVER" and "resolve if normal Hb": only partially specified (B2 text).
7. Hypotension/syncope SAIL count mismatch (37,756 vs 48,961) and the implausibly high Bradford Bone disease prevalence (9.6%).
8. Meaning of the `mar-` provenance tag and of the "technical specification document": not documented.
9. Licensing/IP of reusing Baseline2 content for Meuhedet: unclear. Resolution: written permission from the University of Leeds / eFI2 team.
10. Corrupted SNOMED IDs (14) and CTV3 formatting defects in Baseline2 need repair against a SNOMED CT release (UK edition) before use; mapping to Meuhedet's coding systems is outside the scope of these sources.

---

## 10. Source URLs consulted
- Clegg 2016: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4846793/fullTextXML ; supplement https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4846793/supplementaryFiles ; https://academic.oup.com/ageing/article/45/3/353/1739750
- Correction: PMC6016616 (https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6016616/fullTextXML)
- eFI2 (Best 2025): https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11957239/fullTextXML ; supplement https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11957239/supplementaryFiles ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11957239/
- BJGP editorial (Nimmons, Clegg, Walters 2025): https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12117642/fullTextXML
- Romero-Ortuno & Keevil editorial: https://academic.oup.com/ageing/article/54/5/afaf111/8124755
- HTA monograph eFI+ (Archer ... Clegg; HTA 2026? Vol 30 No 61): https://www.journalslibrary.nihr.ac.uk/hta/GJAC1008 (local copy raw/nihr_gjac1008.txt fetched by a sibling task; quotes verified in that text); https://fundingawards.nihr.ac.uk/award/NIHR127905
- Hollinghurst 2019: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6814149/fullTextXML and supplementaryFiles
- Aslam 2025 (DynAIRx GCAF): https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12102889/fullTextXML ; supplementaryFiles ; https://github.com/DynAIRx/GCAF_DynAIRx ; https://api.github.com/repos/DynAIRx/GCAF_DynAIRx
- NHS England eFI: https://www.england.nhs.uk/ourwork/clinical-policy/older-people/frailty/efi/
- ARC YH eFI2 project: https://arc-yh.nihr.ac.uk/research/projects/development-and-national-implementation-of-efi-2/
- Vision eFI help: http://help.visionhealth.co.uk/Vision_Consultation_Manager_Help_Centre/Content/ConMgr/76292.htm ; https://help.visionhealth.co.uk/PDFs/Outcome%20Manager/Electronic%20Frailty%20Index%20and%20Stratification%20Tool.pdf
- NHS Dorset Appendix D: https://nhsdorset.nhs.uk/Downloads/aboutus/finance/11J_0230%20Enhanced%20Frailty/Appendix%20D%20-%20Enhanced%20Frailty%20Read%20Codes.pdf
- OpenCodelists searches: https://www.opencodelists.org/?q=frailty (and q=eFI, housebound, activity+limitation, social+vulnerability, requirement+for+care, fragility+fracture, memory+concerns, polypharmacy)
- HDR UK Phenotype Library API: https://phenotypes.healthdatagateway.org/api/v1/phenotypes/?search=eFI (and frail, Frailty, housebound, falls, fragility)
