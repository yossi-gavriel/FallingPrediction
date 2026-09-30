# Translating eFalls definitions to Meuhedet: authoritative definitions and CANDIDATE correspondences

Compiled 2026-09-14. **Everything below that looks like a mapping is a CANDIDATE.** It needs clinical and data-team validation at Meuhedet. It is not a final clinical mapping.

Evidence labels:
- **[EXPLICIT]** = stated in the cited source (short verbatim quote, <= 40 words)
- **[INFERRED]** = my reasoning from the evidence (reasoning given)
- **[NOT FOUND]** = searched, not located in an authoritative public source

Abbreviations used for local paths:
- `SP` = `<research workspace (not distributed)>`
- `TR` = `SP/research/raw/translation`

Line numbers refer to the local text files. Files marked "(single-line)" are one long line, so they are cited by character offset ("@offset").

Prior partial work was found in `TR/` (GEM files, ICD-9-CM tabular, NHSBSA BNF files, WHO ATC pages, UK alcohol pages, smoking papers). It was reused after re-verification. The prior helper scripts (`TR/gem_analysis*.py`) were **not trusted**: one comment in them wrongly says CMS v32 lacks E-codes (it has 1,291 E-code rows). A fresh script was written instead: `SP/research/translation_work/efalls_icd9cm_candidates.py`, which produces `SP/research/translation_work/efalls_outcome_icd9cm_candidates.csv` (400 rows).

New downloads this round:
- WHOCC ATC/DDD Guidelines 2026 (PDF → `TR/whocc_atc_guidelines_2026.txt`)
- UK CMO 2016 guideline PDF (→ `TR/uk_cmo_low_risk_drinking_2016.txt`)
- NI QOF smoking ruleset v30 (→ `TR/ni_smok_ruleset_v30.txt`)
- Europe PMC full texts for three Meuhedet papers (`TR/PMC11552639.txt`, `TR/PMC12151512.txt`, `TR/PMC13224401.txt`)

---

## 0. Headline findings

1. **"BNF sub-sub-chapter" most plausibly means the BNF *paragraph*.** That is hierarchy level 3, a 6-digit code such as `040201`. The sub-paragraph (7-digit, such as `0402010`) is the less likely reading.
   - eFI2, from the same programme and apparently the same SAIL dataset, labels the unit "BNF sub-subchapters (level 3)". [EXPLICIT]
   - NHSBSA levels are chapter (2 digits), section (+2), paragraph (+2) and sub-paragraph (+1). [EXPLICIT]
   - In the current NHSBSA file, only 52 of 503 paragraphs have more than one sub-paragraph. The choice therefore matters only for those 52, which include insulins, antidiabetics, renin-angiotensin drugs and antimigraine drugs. [EXPLICIT, computed]
2. **"Excluding non-drug chapters of the BNF, e.g. bandages" is not enumerated in any eFalls source.** [NOT FOUND]
   - NHSBSA pseudo-chapters 18–23 fall outside chapters 1–15. In NHSBSA data, chapters 20–23 (dressings, appliances, incontinence, stoma) have **no paragraph level**, so they cannot contribute a "sub-sub-chapter". [EXPLICIT]
   - Chapter 19 "Other drugs and preparations" and chapter 18 "Preparations used in diagnosis" are ambiguous.
   - The original eFI used "chapters 1–15". [EXPLICIT]
3. **No WHO ATC level is equivalent to a BNF paragraph.** [INFERRED from examples below]
   - ATC 4th level (chemical/pharmacological subgroup) is usually *finer* than a BNF paragraph, e.g. antipsychotics are 1 BNF paragraph but 11 ATC 4th levels.
   - ATC 3rd level is sometimes comparable, e.g. A10B ≈ BNF 060102.
   - ATC gives combination products their own codes. The WHOCC warns that ATC is "not strictly a therapeutic classification". [EXPLICIT]
   - Meuhedet research already extracts medications by ATC codes from purchase (dispensing) records. [EXPLICIT]
4. **The eFalls outcome list is identical in Age & Ageing Table S2.3 and HTA Appendix 1.** All 31 rows match (`diff` exit 0). [EXPLICIT, computed]
5. **There is no official WHO ICD-10 → ICD-9-CM map.** The CDC/CMS GEMs link ICD-10-**CM** to ICD-9-CM.
   - ICD-10-CM differs from WHO ICD-10 in this block: W02, T08, T10, T12 and T14.2 do not exist in ICD-10-CM; W00, W05, W16 and W18 are redefined; 7th-character encounter codes are added. [EXPLICIT]
   - All 2,595 GEM rows from eFalls-analogous ICD-10-CM initial-encounter codes carry the "approximate" flag. [EXPLICIT, computed]
   - "GEMs are not crosswalks". [EXPLICIT]
6. **Main ICD-9-CM non-equivalences** [EXPLICIT tabular; INFERRED consequences]:
   - 733.1x "Pathologic fracture" covers any cause; WHO M80 is osteoporotic only.
   - ICD-9-CM E-codes are supplementary and coarser. For example, E884.9 absorbs WHO W14 (tree) and W17; E885.9/E888.x absorb W00/W01/W18; E888 "Accidental fall NOS / Fall on same level NOS" spans WHO W18/W19.
   - E887 "Fracture, cause unspecified" sits inside the ICD-9-CM falls block but has no WHO W-code analogue (WHO X59.0).
   - W16 (diving/jumping into water) → E883.0.
   - WHO T10 ("arm NOS") ↔ ICD-9-CM 818 is contradicted by the GEM, which sends 818.x to S62.90 (wrist/hand).
   - If the "E" prefix is dropped, E880–E887 collide with ICD-9-CM 880–887 (open wounds and amputations of the upper limb).
   - In mixed ICD-9/ICD-10 tables, undotted "E880"/"E888" also collide with ICD-10 E88.0/E88.8 (metabolic disorders).
7. **Meuhedet coding context (not a mapping):**
   - Meuhedet's own eFI (MEFI, Aging 2024) lists falls using ICD-9 codes. The list includes both `E880` and a bare `880`, E887 and procedure codes, over a 12-month window.
   - Its polypharmacy is "8+ drugs" in 12 months. Hospitalisations came from hospital invoices.
   - A 2026 Meuhedet paper used **both ICD-9 and ICD-10** diagnosis codes and ATC codes. [EXPLICIT]
8. **Alcohol:**
   - eFalls categories (harmful / higher risk / lower risk / previous higher-risk-or-harmful / zero / missing) do **not** match current UK primary-care categories (lower / increasing / higher risk; AUDIT/AUDIT-C bands). [EXPLICIT]
   - The public eFI2/eFI+ list "Baseline2" defines them from legacy drinker-type codes and units/week cut-offs 0 / 1–20 / 21–48 / 49+. It **omits** the SNOMED "Lower/Increasing/Higher risk alcohol drinking" concepts and AUDIT scores. [EXPLICIT, computed]
   - HTA Table 1 and Table 15 print "Higher-risk drinking" twice. The second row's counts (11,231 / 5,699) equal the "Lower risk drinking" row in Age & Ageing, so it is a labelling error. [EXPLICIT counts; INFERRED error]
9. **Smoking:**
   - eFalls uses current vs ex/never (reference) only. SAIL has no missing category, and the three categories sum exactly to 660,417. [EXPLICIT]
   - Bradford records only 16 ex-smokers (0.0%). [EXPLICIT]
   - UK EHR conventions use the most recent status code, with "never" after an earlier smoker/ex code recoded to ex-smoker (Atkinson 2017 SAIL algorithm; OpenSAFELY). [EXPLICIT]
   - eFalls' own rule is not published. [NOT FOUND]
10. **BMI:**
    - Box S3.1 "obese if BMI ≥ 40" matches the WHO **class III** threshold, not WHO obesity (≥30).
    - The published category distribution (obese = 31.8% of recorded BMI in SAIL) fits ≥30 far better than ≥40. [INFERRED]
    - WHO: <18.5 underweight; 18.5–24.9 normal; 25.0–29.9 pre-obesity; 30.0–34.9 / 35.0–39.9 / ≥40 obesity class I/II/III. [EXPLICIT]

---

## 1. Polypharmacy: BNF structure, eFI definitions, ATC analogue

### 1.1 What eFalls says [EXPLICIT]
- Box S3.1 (`SP/efalls/supp.md` line 318): polypharmacy "is a count of unique drugs* prescribed over the 120 days prior to index date (excluding non-drug chapters of the BNF e.g., bandages)".
- Footnote (supp.md line 322): "Unique BNF sub-sub-chapters. Combinations of >1 drug from a sub-sub-chapter only counted once towards the total."
- Model term: ln((P+1)/10), coefficient 0.3296295 (supp.md lines 224–225).
- HTA Box (`SP/research/raw/gjac1008_struct.txt` line 1052) repeats the text without the footnote and with a typo: "over the 120 prior to index date".
- Missing prescriptions (HTA line 495): "Where no prescription was included on a patient’s record, it was assumed that no prescription was written."
- eFalls does not state:
  - whether repeat vs acute prescriptions are handled differently [NOT FOUND]
  - which BNF edition was used [NOT FOUND]
  - how Read v2 drug codes in SAIL were mapped to BNF [NOT FOUND]
  - which chapters count as "non-drug" [NOT FOUND]

### 1.2 Related definitions in the eFI family [EXPLICIT; from `SP/research/efi_codelists.md` §1.5, §2.1, §6 and re-checked]

| Tool | Definition | Source |
|---|---|---|
| eFI (Clegg 2016) | "Polypharmacy was defined on the basis of the presence of ≥5 prescribed medications, using chapters 1–15 of the British National Formulary." No window stated. | PMC4846793 (`TR/PMC4846793.txt`, single-line @16210) |
| eFI2 (Best 2025), Table 2 footnote | "Number of medications from different BNF sub-subchapters (level 3) prescribed in previous 90 days"; categories 0–4 / 5–9 / 10+ | PMC11957239 (`TR/PMC11957239.txt`, single-line @~23085) |
| eFI2 supplement Appendix 1 | "Polypharmacy is defined according to the number of medications prescribed from separate BNF sub chapters in the last 90 days" (says "sub chapters", which is inconsistent with Table 2) | `SP/research/raw/efi/efi2_supp.txt` line 60 |
| eFalls (Archer 2024) | unique BNF sub-sub-chapters, 120 days, excluding non-drug chapters, continuous FP term | supp.md 318/322 |
| eFalls ARC GM pilot 2026 (implementation) | "number of medications recorded in the 93 days up to the data snapshot" (descriptive only) | `SP/research/raw/arcgm_efalls_report_clean.txt` (single-line @~31041) |
| MEFI (Meuhedet eFI, Hershkowitz Sikron 2024) | "Polypharmacy | 12 months | 8+ drugs" (the counting unit is not defined) | PMC11552639, `TR/PMC11552639.txt` line 201 |

### 1.3 NHSBSA BNF code structure [EXPLICIT]
- **NHSBSA booklet** "BNF Classification and Pseudo Classification Used by the NHS Prescription Services" (April 2017, BNF 72). URL: https://www.nhsbsa.nhs.uk/sites/default/files/2017-04/BNF_Classification_Booklet-2017.pdf ; local `TR/nhsbsa_bnf_booklet_2017.txt`.
  - Lines 40–46: "four levels", described as "A two digit chapter heading", "A two digit section heading", "A two digit paragraph heading", "A single digit sub paragraph heading. Where the BNF does not extend to all these levels, the record is filled with zeros."
  - Lines 49–55: "Pseudo Chapter 19 headed ‘Other Drugs and Preparations’ includes many drugs and preparations …". The listed contents include ‘Alcohol, wines and spirits’, ‘Homeopathic preparations’, ‘Poisoning antidotes’, ‘Disinfectants’, ‘Lubricating jellies’.
  - Lines 332–338: chapter 18 "Preparations used in Diagnosis", "Not used by BNF".
  - Lines 340–387: chapter 19 "Other Drugs and Preparations", "Not used by BNF".
  - Lines 388+: 20 "Dressings".
  - Lines 417+: 21 "Appliances" "DT Part IXA excluding Dressings".
  - Lines 496–498: 22 "Incontinence Appliances DT Part IXB"; 23 "Stoma Appliances DT Part IXC".
  - Lines 165–191: some BNF items were moved into pseudo-chapters. Examples: stoma care 01:08 → 23; peak flow meters/inhaler devices → 21; hypodermic equipment 06:01:01:3 → 21:01:09; contraceptive devices 07:03:04 → 21:04:00; wound management (BNF Appendix 5) → 20/21.
- **NHSBSA PCA "Background Information and Methodology" (June 2025).** URL: https://nhsbsa-opendata.s3.eu-west-2.amazonaws.com/pca/pca_background_info_methodology_june2025_v001.html ; local `TR/pca_methodology_2025.txt` lines 134–150.
  - "The NHSBSA uses and maintains the classification system of the BNF implemented prior to the release of edition 70, including the six pseudo BNF chapters (18 to 23)."
  - "Most of the presentations held in these pseudo chapters are dressings, appliances, and medical devices."
  - Levels: "chapter, section, paragraph, sub-paragraph, chemical substance, product, and individual presentation."
  - "Presentations in chapters 20 to 23 do not have assigned BNF paragraphs, sub-paragraphs, chemical substances, or products."
  - "Every January the NHSBSA updates the classification … This may involve some drugs changing BNF codes."
- **NHSBSA Open Data "BNF Code Information – Current Year"** (https://opendata.nhsbsa.net/dataset/bnf-code-information-current-year ; local `TR/nhsbsa_bnf_pkg.json`):
  - "BNF Drug Code: A 15-digit code, where the first seven digits correspond to the BNF categories, and the last eight digits represent the medicinal product, form, strength, and link to the generic equivalent."
  - "BNF Appliance Code: An 11-digit code used for medical appliances."
- **OpenPrescribing wiki "BNF for dummies"** (https://github.com/ebmdatalab/openprescribing/wiki/BNF-for-dummies, fetched 2026-09-14): "Sections, eg 0601"; "Paragraphs/subsections, eg 060101: Insulin"; "Subparagraphs, eg 0601012"; "Chemicals, eg 0601012W0".
- **Computed from NHSBSA `bnf_code_current_202608_version_90.csv`** (`TR/bnf_code_current_202608_v90.csv`, 55,496 presentations; resource URL in `TR/nhsbsa_bnf_pkg.json`) [EXPLICIT, computed]:
  - 21 chapters: 01–15 and 18–23. There are no chapters 16–17.
  - 222 sections (4 digits); 503 paragraphs (405 are 6-digit; 98 are 4-digit placeholders in chapters 20–23); 615 sub-paragraphs (517 are 7-digit).
  - Chapters 1–15: 111 sections, **382 paragraphs, 493 sub-paragraphs**.
  - For chapters 20–23, BNF_PARAGRAPH_CODE = BNF_SUBPARAGRAPH_CODE = section code (e.g. `2001`), and presentation codes are 11 characters. This confirms that there is no paragraph level.
  - Example: section 0402 has paragraphs 040201 "Antipsychotic drugs" (only sub-paragraph 0402010), 040202 "Antipsychotic depot injections", 040203 "Drugs used for mania and hypomania".
  - Paragraphs with more than one sub-paragraph: 52/503. Examples:
    - 020505 Renin-angiotensin system drugs → 0205051 ACE inhibitors / 0205052 AT-II receptor antagonists / 0205053 renin inhibitors
    - 060101 Insulin → 0601011 short-acting / 0601012 intermediate and long-acting
    - 060102 Antidiabetic drugs → 0601021 sulfonylureas / 0601022 biguanides / 0601023 other
    - 040704 Antimigraine → 0407041 acute / 0407042 prophylaxis
  - Chapter 19 (449 presentations; 11 sections): 1901 Alcohol, wines and spirits; 1902 Selective preparations; 1904 Single substances; 1905 Other preparations; 1906 Acids; 1907 Base/diluent; 1908 Colouring/flavouring; 1909 Disinfectants; 1913 Cordials and soft drinks; 1914 Waters; 1915 Other gases.
  - Chapter 18 (5 presentations): 1803 X-ray contrast media; 1804 Diagnostic agents.
  - Non-drug items **inside chapters 1–15**:
    - 060106 "Diabetic diagnostic and monitoring agents" (160 presentations)
    - 010300 "Test for helicobacter pylori"
    - 130201/130202 emollient/barrier preparations
    - 131301 medicated stockings
    - chapter 9 nutrition: 0903 IV nutrition; 0904 "Oral nutrition (OLD - DO NOT USE)"; 0909 Foods; 0913 Oral nutrition with about 60 paragraphs (ONS, infant formula, gluten-free food…); 0914 Enteral nutrition
  - [INFERRED] The chapter 9 structure (0913/0914 with many paragraphs) is a recent NHSBSA reclassification ("OLD - DO NOT USE" marks the superseded 0904). A 2026 BNF file would therefore not reproduce a 2018 SAIL-era paragraph count for nutrition products. The historic NHSBSA dataset (https://opendata.nhsbsa.net/dataset/bnf-code-information-historic) would be needed.

### 1.4 Most plausible meaning of "BNF sub-sub-chapter"
- [EXPLICIT] eFI2 (same programme, and apparently the same SAIL predictor dataset; see efi_codelists.md §5) calls the unit "BNF sub-subchapters (level 3)".
- [EXPLICIT] NHSBSA's four code levels are chapter, section, paragraph, sub-paragraph.
- [INFERRED] So level 3 = paragraph = the first 6 characters of the 15-character BNF code (`CCSSPP`). "Sub-chapter" = section (`CCSS`) and "sub-sub-chapter" = paragraph. OpenPrescribing's name for this level, "Paragraphs/subsections", is consistent.
- [INFERRED] The alternative reading is the 7-digit "sub-paragraph" (`CCSSPPs`, e.g. `0402010`), sometimes loosely called a "7-digit paragraph". It would count, for example, an ACE inhibitor + ARB, or metformin + gliclazide, as 2 rather than 1. It differs from the paragraph count only for the 52 split paragraphs. I found no eFalls/eFI2 source that uses 7 digits. [NOT FOUND]
- [EXPLICIT] eFI2 Appendix 1 says "separate BNF sub chapters". Read literally that would be the 4-digit section, which contradicts "(level 3)" in eFI2 Table 2. Treat it as an internal inconsistency.
- **Candidate unit definitions for sensitivity analysis** [INFERRED]:
  - (P1) primary: BNF paragraph (6 digits)
  - (P2) BNF sub-paragraph (7 digits)
  - (P3) BNF section (4 digits)

### 1.5 "Non-drug chapters": candidate exclusion sets [INFERRED; none is stated by eFalls]

| Candidate | Chapters counted | Basis |
|---|---|---|
| E1 | 1–15 only (exclude 18–23) | Original eFI "chapters 1–15" [EXPLICIT, PMC4846793]; PCA: pseudo chapters 18–23 are "outside of chapters 1 to 15" [EXPLICIT] |
| E2 | 1–15 + 18–19 (exclude 20–23) | 20–23 have no paragraph level in NHSBSA data [EXPLICIT]; "bandages" (the eFalls example) sit in chapter 20 [EXPLICIT booklet line 390] |
| E3 | E1 minus non-drug paragraphs inside 1–15 (e.g. 060106 diabetic monitoring agents, chapter 9 foods/ONS) | Clinically "unique drugs" [INFERRED only] |

The eFalls example "bandages" only guarantees exclusion of chapter 20. Whether 18, 19, 21–23 and in-chapter devices were excluded is [NOT FOUND].

### 1.6 WHO ATC: structure and closest analogue
**Sources:**
- WHOCC, *Guidelines for ATC classification and DDD assignment 2026* (29th ed.), https://atcddd.fhi.no/filearchive/publications/2026_guidelines_for_atc_classification_and_ddd_assignment.pdf ; local `TR/whocc_atc_guidelines_2026.txt`
- WHOCC web pages "Structure and principles" (https://atcddd.fhi.no/atc/structure_and_principles/ ; local `TR/atc_structure.txt`) and "Purpose" (https://atcddd.fhi.no/atc_ddd_methodology/purpose_of_the_atc_ddd_system/ ; `TR/atc_purpose.txt`)
- ATC index pages, e.g. https://atcddd.fhi.no/atc_ddd_index/?code=A10B (local `TR/atc_*.txt`)

**[EXPLICIT] statements:**
- Five levels (Guidelines lines 446–451, p.14): "the active substances are classified in a hierarchy with five different levels … The 3rd and 4th levels are chemical, pharmacological or therapeutic subgroups and the 5th level is the chemical substance."
- Metformin example (lines 459–477): A (1st) → A10 (2nd) → A10B (3rd, pharmacological subgroup) → A10BA (4th, chemical subgroup) → A10BA02 (5th, chemical substance).
- Lines 525–526: "The ATC system is, however, not strictly a therapeutic classification system." Lines 540–543: "drugs with similar therapeutic use may be classified in different groups."
- Lines 548–553: "only one ATC code for each route of administration" … "allows aggregation of data … without counting a pharmaceutical product more than once."
- Combinations get separate codes (lines ~655–705; web page `TR/atc_structure.txt`): "given a separate 5th level code (50-series)". There are also separate 3rd/4th levels for combinations (e.g. C10B, N02AJ, R03AL).
- Coverage (lines 489–491, 516–517): "The coverage of the system is not comprehensive." "Complementary, homeopathic and herbal traditional medicinal products are in general not included."
- Lines 427–428: "the ATC/DDD system by itself is not suitable for guiding decisions about reimbursement, pricing and therapeutic substitution."
- Group V (lines 8604–8652, 8815–8854, 9067–9069):
  - V04 Diagnostic agents; V06 General nutrients; V07 All other non-therapeutic products (V07AA Plasters; V07AN Incontinence equipment; V07AS Stomi equipment; V07AT Cosmetics; V07AV Technical disinfectants…); V08 Contrast media; V20 Surgical dressings.
  - "A detailed classification of surgical dressings is prepared and maintained by the Ministry of Defence in the UK."
  - "Very few DDDs are assigned in this group."

**Worked comparisons** [EXPLICIT codes from local files; INFERRED comparison]:

| BNF (NHSBSA 2026) | ATC 2nd | ATC 3rd | ATC 4th |
|---|---|---|---|
| 040201 Antipsychotic drugs (1 paragraph; lithium is in 040203) | N05 Psycholeptics | N05A Antipsychotics (includes N05AN Lithium) | 11 subgroups N05AA…N05AX (by chemical structure) (`TR/atc_N05A.txt`) |
| 060102 Antidiabetic drugs (sub-paras 0601021 SU, 0601022 biguanides, 0601023 other) | A10 | A10B Blood glucose lowering drugs, excl. insulins | A10BA biguanides, A10BB sulfonylureas, A10BD **combinations**, A10BH DPP-4, A10BJ GLP-1, A10BK SGLT2, … (10 groups) (`TR/atc_A10B.txt`) |
| 020505 Renin-angiotensin system drugs (sub-paras 0205051 ACEi, 0205052 ARB, 0205053 renin inhibitors) | C09 | C09A ACE inhibitors plain; C09B ACE inhibitors **combinations**; C09C ARBs plain; C09D ARBs combinations; C09X other (`TR/atc_C09.txt`) | C09AA etc. |

**[INFERRED] Conclusions for a Meuhedet analogue (all candidates):**
- **ATC 4th level** is the closest *drug-class* granularity, but it is usually finer than a BNF paragraph: antipsychotics 1 vs up to 11; diabetes drugs 1 vs up to 10. It would inflate counts relative to eFalls.
- **ATC 3rd level** matches some paragraphs well (A10B ≈ 060102) but is coarser or cuts across others (C09A/C09B/C09C/C09D vs one paragraph 020505; N05A includes lithium).
- **ATC 5th level** (substance) = "unique drugs" in the literal sense of Box S3.1's main text. It contradicts the footnote's "only counted once" rule for same sub-sub-chapter drugs.
- Combination products carry their own ATC code (e.g. C09BA02 enalapril+diuretic, A10BD). Counting distinct ATC codes may double-count ingredients already present as plain products, or treat a combination as one unit, depending on level.
- ATC is not therapeutic by indication, whereas the BNF groups by "primary therapeutic indication" (PCA methodology line 136–137). Some drugs therefore sit in different "units" (e.g. low-dose vs high-dose substances, eye drops vs systemic).
- Non-drug analogue of BNF 18–23 in ATC: V04, V06, V07 (esp. V07AA plasters, V07AN incontinence, V07AS stoma), V08, V20. Most medical devices/appliances simply have **no ATC code** in pharmacy data.
- **Recommended candidate strategy (for validation):**
  1. Build a Meuhedet-formulary → BNF-paragraph crosswalk by substance (ATC 5th level + route → BNF chemical substance → first 6 BNF characters), using a dated NHSBSA historic BNF file near 2018. ATC 4th-level counting would be sensitivity analysis S1, and ATC 3rd-level S2.
  2. Keep a log of unmapped substances.
  3. Note the data-type difference: eFalls = GP *prescriptions issued*; Meuhedet = *purchases/dispensing*. See §1.7.

### 1.7 Meuhedet context for medication data [EXPLICIT]
- Marom A et al., BMC Prim Care 2026, doi:10.1186/s12875-026-03314-5, PMC13224401 (`TR/PMC13224401.txt` line 35): "The Anatomical Therapeutic Chemical (ATC) codes that we used for medication extraction were: N06AB, N06AX11, …". Line 13 says the analysis used "purchasing records".
- MEFI (PMC11552639, `TR/PMC11552639.txt` lines 185, 195, 203, 206) defines deficits with ATC codes at mixed levels: ‘N07CA’ (4th), ‘N06D’ (3rd), ‘N05CD09’ (5th), ‘G04BD’ (4th).
- [INFERRED] Purchase data differ from prescribing data. Unfilled prescriptions are missing; 90-day packs may have purchase dates outside a 120-day window; OTC and hospital-dispensed drugs may be absent. The 120-day window therefore needs a supply-overlap rule, which is a clinical/data decision.

---

## 2. Outcome: eFalls ICD-10 list → ICD-9-CM candidates

### 2.1 The list and its twin [EXPLICIT]
- Age & Ageing Table S2.3 (`SP/efalls/supp.md` lines 86–120): W00–W19, M80, S22, S32, S42, S52, S72, S82, T08, T10, T12, T14.2, with WHO ICD-10 titles.
- HTA Appendix 1 "Fall outcome International Classification of Diseases code list" (`gjac1008_struct.txt` lines 1884, 1889–1919): **identical**. `diff` of the 31 code|description rows returned no differences. The HTA also has an empty `[TABLE][/TABLE]` stub at lines 1885–1886.
- Definition (fulltext.md line 84; HTA line 381): "any (one or more) ED attendance or hospital admission for a fall or fracture (as an indicator of an injurious fall) within 12 months".
- Data (fulltext.md line 84; HTA line 383): SAIL via "Emergency Department Dataset and Patient Episode Database for Wales"; Bradford via "linked secondary care data".
- **Not stated** [NOT FOUND]:
  - diagnostic position (primary vs any)
  - whether ED data were ICD-10 coded
  - handling of subsequent/aftercare episodes
  - whether W-codes alone (without an S/T/M nature code) qualified
  - de-duplication across ED and admission
- [EXPLICIT] eFI2 used a different outcome, "Hospitalisation with fall or fragility fracture", "defined using established code lists" (PMC11957239 @~9822). Do not substitute it.

### 2.2 WHO ICD-10 vs ICD-10-CM vs ICD-9-CM: sources
- WHO ICD-10 2019 browser (https://icd.who.int/browse10/2019/en#/W00-W19 ; local `TR/who_icd10_concept_W00-W19.txt`, `…_T14.txt`, `…_M80.txt`, `…_X59.txt`; JSON children `TR/who_icd10_*.json`). Block W00–W19 **Excl.** (`who_icd10_concept_W00-W19.txt` lines 17–27):
  - "assault ( Y01-Y02 )"
  - fall from animal (V80.-), burning building (X00), into fire, into water with drowning, machinery, "repeated falls not resulting from accident ( R29.6 )", transport vehicle (V01-V99)
  - "intentional self-harm ( X80-X81 )"
- WHO inclusion notes used below [EXPLICIT] (W03 lines 43–49; W18 lines ~146–154; W19 line 159 of `who_icd10_concept_W00-W19.txt`; T10 line 66 and T14.2 lines 211–222 of `who_icd10_concept_T14.txt`; M80 lines 13–22 of `who_icd10_concept_M80.txt`; X59.0 line 24 of `who_icd10_concept_X59.txt`):
  - W03 "Incl.: fall due to collision of pedestrian (conveyance) with another pedestrian (conveyance)"
  - W10 incl. escalator, incline, ramp, ice/snow on stairs
  - W13 incl. balcony, bridge, building, floor, railing, roof, …, window
  - W16 incl. striking bottom/wall/diving board/water surface
  - W17 incl. cavity, dock, haystack, hole, pit, quarry, shaft, tank, well…
  - W18 "Incl.: fall: from bumping against object; from or off toilet; on same level NOS"
  - W19 "Incl.: accidental fall NOS"
  - T10 "Incl.: Broken arm NOS; Fracture of arm NOS"
  - T14.2 "Incl.: Fracture: NOS …; Excl.: multiple fractures NOS ( T02.9 )"
  - M80 "Incl.: osteoporotic vertebral collapse and wedging; Excl.: … pathological fracture NOS ( M84.4 )"
  - X59.0 "Exposure to unspecified factor causing fracture"
- ICD-10-CM FY2018 order file (CDC: https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2018/2018-ICD-10-Code-Order-Descriptions.zip ; local `TR/icd10order/icd10cm_order_2018.txt`) [EXPLICIT]:
  - W00 "Fall due to ice and snow" (line 88046)
  - W05 "Fall from non-moving wheelchair, nonmotorized scooter and motorized mobility scooter" (88107)
  - W16 "Fall, jump or diving into water" (88215)
  - W18 "Other slipping, tripping and stumbling and falls" (88465), which includes W18.4x "Slipping, tripping and stumbling without falling" (88510)
  - **No W02, T08, T10, T12** categories. T14 has only T14.8 "Other injury of unspecified body region" (68682) and T14.9x; there is no T14.2.
  - M80 "Osteoporosis with current pathological fracture" (20363), with 7th characters A/D/G/K/P/S.
  - Skates/skis/skateboard falls moved to V00.1x–V00.3x "Fall from …" (lines 82833–82979).
- GEMs 2018 (CDC NCHS: https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2018/2018_I10gem.txt and `2018_I9gem.txt` ; guide `Dxgem_guide_2018.pdf` ; tech doc `GemsTechDoc_2018.pdf` ; local `TR/`) [EXPLICIT]:
  - Guide line 88: "There is no simple “crosswalk from I-9 to I-10” in the GEM files."
  - Line 282: "Please be advised: GEMs are not crosswalks."
  - Line 300: "The correspondence between codes in the source and target systems is approximate in most cases."
  - Tech doc lines 272–273: the approximate flag "identifies entries where the complete meaning of the source system code and that of the target system code are not considered equivalent."
  - Computed: all **2,595** I10→I9 rows for eFalls-analogous ICD-10-CM categories (W00–W19, M80, S22, S32, S42, S52, S72, S82; initial-encounter 7th char A/B/C) have flags `10000` (approximate).
- ICD-9-CM tabular FY2012 (CDC: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD9-CM/2011/Dtab12.zip ; local `TR/dtab/Dtab12.txt`) and CMS v32 descriptions (local `TR/cmsv32/CMS32_DESC_LONG_DX.txt`; undotted, E-codes kept with the "E" prefix, e.g. `E8859`) [EXPLICIT]:
  - Line 31989 (E-code chapter): "it is intended that it shall be used in addition to a code from one of the main chapters of ICD-9-CM, indicating the nature of the condition."
  - ACCIDENTAL FALLS (E880–E888), lines 33694–33793. Block excludes (33695–33702) include transport vehicle, machinery, into water with submersion.
  - E880.0 Escalator; E880.1 sidewalk curb; E880.9 other stairs or steps; E881.0 ladder; E881.1 scaffolding; E882 building or other structure; E883.0 "Accident from diving or jumping into water [swimming pool]"; E883.1 well; E883.2 storm drain or manhole; E883.9 other hole.
  - E884.0 playground; E884.1 cliff; E884.2 chair; E884.3 wheelchair ("Fall from motorized mobility scooter"); E884.4 bed; E884.5 other furniture; E884.6 commode ("Toilet"); E884.9 other fall from one level to another ("embankment, haystack, stationary vehicle, tree").
  - E885.0 nonmotorized scooter; E885.1 roller skates; E885.2 skateboard; E885.3 skis; E885.4 snowboard; E885.9 other slipping, tripping, stumbling.
  - E886.0 in sports; E886.9 other/unspecified ("collision of pedestrian (conveyance) with another pedestrian (conveyance)").
  - **E887 "Fracture, cause unspecified"**.
  - E888 "Other and unspecified fall" ("Accidental fall NOS; Fall on same level NOS"): E888.0 striking sharp object; E888.1 striking other object; E888.8 other; E888.9 unspecified.
  - Related codes outside the block: E917.5–E917.8 "… with subsequent fall" (34328–34333); E929.3 "Late effects of accidental fall" (34772); E957 suicide by jumping (35352); E968.1 assault by pushing from high place; E987 falling, undetermined intent.
  - 733.1 "Pathologic fracture" (21540–21557): "Chronic fracture; Spontaneous fracture; Excludes: stress fracture …; traumatic fractures (800-829)". Subcodes 733.10–733.19 are site-specific.
  - 733.0x Osteoporosis (21527–21539) is a separate code.
  - FRACTURES (800–829) (25350–25375): "Excludes: … pathologic or spontaneous fracture (733.10-733.19)"; "A fracture not indicated as closed or open should be classified as closed."
  - 805 vertebral column without cord injury (25537–25566; 805.8/805.9 "Unspecified"); 806 with spinal cord injury (25567–25661); 807 rib(s), sternum, larynx, trachea (25662–25685); 808 pelvis; 809 ill-defined trunk.
  - 810–812 clavicle/scapula/humerus; 813 radius and ulna; 814–817 wrist/hand; 818 "Ill-defined fractures of upper limb" incl. "arm NOS" (25914–25922); 819 multiple both upper limbs/with ribs (25923).
  - 820–821 femur; 822 patella; 823 tibia and fibula; 824 ankle; 825–826 foot; 827 "Other, multiple, and ill-defined fractures of lower limb" incl. "leg NOS" (26070–26079); 828 multiple lower limbs; 829 "Fracture of unspecified bones" (26086–26088).

### 2.3 Candidate ICD-9-CM correspondence per WHO eFalls category
Full row-level evidence is in `SP/research/translation_work/efalls_outcome_icd9cm_candidates.csv`. Columns: ICD-9-CM dotted/undotted code, title, WHO category, evidence (backward GEM source category or tabular), forward-GEM ICD-10-CM categories, candidate_status, notes. Status counts: 220 CANDIDATE-core, 27 CANDIDATE-forward-GEM-only, 3 CANDIDATE-WHO-inclusion, 38 REVIEW, 102 REVIEW/likely-exclude, 10 EXCLUDE-candidate.

| WHO ICD-10 (eFalls) | ICD-10-CM analogue (FY2018) | ICD-9-CM candidate(s) | Evidence | Non-equivalence notes |
|---|---|---|---|---|
| W00 ice and snow, same level | W00.0–W00.9 (CM adds stairs W00.1 and level-to-level W00.2) | E885.9, E884.9 (+E880.9 if on stairs) | Backward GEM | ICD-9-CM has no ice/snow cause axis. Ice/snow is an activity code (E003) or unrecorded [INFERRED] |
| W01 slip/trip/stumble, same level | W01.x | E885.9; E888.0/E888.1 when striking object | Backward GEM | E888.0/E888.1 do not specify slipping [INFERRED] |
| W02 skates/skis/skateboards | none (CM V00.11x–V00.32x) | E885.1 (roller skates), E885.2 (skateboard), E885.3 (skis), E885.4 (snowboard) | Backward GEM from CM V00 "fall from" codes [INFERRED W02≈V00 set] | Ice-skates fall in CM V00.211 → GEM E885.9; snowboard is not in WHO W02 title |
| W03 collision/pushing by another person | W03 | E886.0 (sports); **E886.9** | E886.0 backward GEM; E886.9 via WHO W03 inclusion note (GEM → CM V00) | Stampede/crowd excluded in both (E917.1/E917.6; WHO W52) |
| W04 carried/supported | W04 | E888.8 | Backward GEM | E888.8 "Other fall" is broader |
| W05 wheelchair | W05 (adds scooters) | E884.3 | Backward GEM | E884.3 includes motorized mobility scooter/wheelchair; E885.0 nonmotorized scooter → REVIEW |
| W06 bed | W06 | E884.4 | Backward GEM | Near-equivalent |
| W07 chair | W07 | E884.2 | Backward GEM | Near-equivalent |
| W08 other furniture | W08 | E884.5 | Backward GEM | Near-equivalent |
| W09 playground equipment | W09 | E884.0 | Backward GEM | Near-equivalent |
| W10 stairs and steps | W10 | E880.0, E880.1, E880.9 | Backward GEM | E880.1 sidewalk curb; WHO W10 incl. incline/ramp |
| W11 ladder | W11 | E881.0 | Backward GEM | Near-equivalent |
| W12 scaffolding | W12 | E881.1 | Backward GEM | Near-equivalent |
| W13 building/structure | W13 | E882 | Backward GEM | Near-equivalent |
| W14 tree | W14 | E884.9 | Backward GEM | **Many-to-one:** E884.9 also = W17 ("tree" is an E884.9 inclusion term) |
| W15 cliff | W15 | E884.1 | Backward GEM | Near-equivalent |
| W16 diving/jumping into water | W16 (broadened) | E883.0 | Backward GEM | Not a fall clinically; kept only for list fidelity [INFERRED] |
| W17 other level-to-level | W17 | E883.1, E883.2, E883.9, E884.9 | Backward GEM | E884.9 shared with W14 |
| W18 other fall on same level | W18 (includes W18.4 "without falling") | E884.6 (toilet), E885.9, E888.0, E888.1, E888.8, E917.8 (+E917.7 via WHO incl.) | Backward GEM + WHO W18 inclusions | E917.x are "striking against" codes with subsequent fall |
| W19 unspecified fall | W19 | E888.9 | Backward GEM | E888 category-level "Accidental fall NOS" matches WHO W19 incl. |
| M80 osteoporosis with pathological fracture | M80.0x/M80.8x (7th A) | 733.10–733.16, 733.19 | Backward GEM (initial encounter) | **733.1x is any-cause pathologic fracture.** Forward GEM sends 733.1x mostly to M84.4x (pathological fracture NEC) and 733.13 also to M48.5/M80.08. Candidate refinement: 733.1x AND 733.0x (osteoporosis) [INFERRED; clinical decision] |
| S22 rib(s), sternum, thoracic spine | S22 | 805.2/805.3; 807.0x–807.4 (incl. 807.10 via forward GEM); 809.0/809.1 | Backward and forward GEM | 807.5/807.6 larynx/trachea → S12.8 (exclude); 806.2x–806.3x (with cord injury) → REVIEW |
| S32 lumbar spine and pelvis | S32 | 805.4–805.7; 808.x | Backward GEM | 806.4–806.7 with cord injury → REVIEW |
| S42 shoulder and upper arm | S42 | 810.x, 811.x, 812.x | Backward GEM | Near-equivalent |
| S52 forearm | S52 | 813.x (incl. radius-with-ulna codes via forward GEM) | Backward + forward GEM | Wrist bones 814 are **not** S52 (→ S62) |
| S72 femur | S72 | 820.x, 821.x | Backward GEM | Near-equivalent; stress fractures 733.96/733.97 excluded in both |
| S82 lower leg incl. ankle | S82 | 822.x, 823.x (incl. fibula-with-tibia via forward GEM), 824.x, 827.x | Backward + forward GEM | Foot 825–826 → S92, not in eFalls |
| T08 spine, level unspecified | none | 805.8, 805.9 (+806.8/806.9 REVIEW) | ICD-9-CM tabular | GEM forward 805.8 → S12.9/S22.009/S32.009/S32.10/S32.2 |
| T10 upper limb, level unspecified | none | 818.0, 818.1 | ICD-9-CM tabular ("arm NOS") | **GEM forward 818.x → S62.90 (wrist/hand)**, a contradiction; REVIEW |
| T12 lower limb, level unspecified | none | 827.0, 827.1 | ICD-9-CM tabular ("leg NOS") | 827 also hit by backward GEM from S82; overlap |
| T14.2 fracture of unspecified body region | CM T14.8 (inclusion "Fracture NOS" per secondary code sites, unverified against CMS tabular) | 829.0, 829.1 | ICD-9-CM tabular; GEM forward 829.x → T14.8XXA [EXPLICIT GEM] | T14.8 is broader (abrasion, contusion…) |

**Codes to review or likely exclude** (from the CSV) [EXPLICIT codes; INFERRED status]:
- **Different WHO block, not in eFalls:**
  - cervical spine 805.0x/805.1x/806.0x/806.1x (WHO S12)
  - wrist/hand 814–817 (S62)
  - foot 825–826 (S92)
  - larynx/trachea 807.5/807.6
  - multiple limbs 819.x/828.x (WHO T02, although the GEM sends them to S22/S42/S52/S72/S82 combinations → REVIEW)
- **Intent not accidental:** E957.x, E968.1, E987.x. WHO W00–W19 excludes assault/self-harm, and X80/Y01/Y30 are not in the eFalls list.
- **Late effect:** E929.3 (GEM → W1x 7th char S "sequela").
- **E887 "Fracture, cause unspecified":** GEM forward → W19XXXA, but the WHO analogue is X59.0, which is not in the eFalls list. The fracture itself would already qualify through its 800–829 code.
- **Subsequent-encounter concepts** (V54.1x/V54.2x aftercare; 733.81/733.82 malunion/nonunion; 905.x late effects) arise only from ICD-10-CM 7th characters D/G/K/P/S. They were excluded from the candidate set. [INFERRED: eFalls counts acute ED/admission events; WHO ICD-10 has no 7th character]

### 2.4 E-code storage without the "E" prefix [EXPLICIT + INFERRED]
- [EXPLICIT] Manitoba MCHP concept "External Cause of Injury Codes and Injury Categories" (http://mchp-appserv.cpe.umanitoba.ca/viewConcept.php?conceptID=1168 ; local `TR/mchp_ecodes.txt` line 23): "External cause of injury codes (E-codes) are not recorded with an "E" prefix in the Medical Services / Physician Claims data". This is an example that real claims systems can drop the prefix.
- [EXPLICIT] ICD-9-CM 880–887 are valid numeric categories: open wound of shoulder/upper arm, elbow/forearm/wrist, hand, finger(s), multiple upper limb; traumatic amputation of thumb, other finger(s), arm and hand (`TR/dtab/Dtab12.txt` lines 26903–26952). There is no 888 category, and 929 has only 929.0/929.9 (lines 27536–27538).
- [INFERRED] If Meuhedet stores E-codes without the prefix:
  - "880"–"887" (and 4–5-character forms like "8859") could be confused with open-wound/amputation codes.
  - "888x" and "9293" would be unambiguous, because 888 and 929.3 do not exist as numeric codes.
- **Mixed ICD-9/ICD-10 collision** [EXPLICIT codes; INFERRED risk]:
  - In ICD-10 (WHO and CM), E88 is "Other and unspecified metabolic disorders". ICD-10-CM order file line 4792: E88.0 "Disorders of plasma-protein metabolism, NEC" (line 4793), E88.01 "Alpha-1-antitrypsin deficiency" (line 4794), E88.8 "Other specified metabolic disorders" (line 4804).
  - An undotted string such as "E880", "E8801" or "E888" is therefore ambiguous between ICD-9-CM falls codes and ICD-10 metabolic codes if a Meuhedet table mixes code systems without a code-system column. Marom 2026 shows Meuhedet uses both ICD-9 and ICD-10 (§2.5).
- [EXPLICIT] MEFI's published falls deficit (`TR/PMC11552639.txt` line 186) lists "430, 733.14, 733.96, 800–801, 803, 835, 852, 880, 81.4, 81.51, 81.59, V43.6, E880, E884.2-E884.9, E885.9, E887, E888, …". It includes **both "880" and "E880"**, plus ICD-9-CM procedure codes (81.x) and V43.6 (hip joint replaced).
  - [INFERRED] The bare "880" could be an intended open-wound code or a prefix-less E880. Verify with the MEFI author ("ML revised and adapted the ICD-9 medical coding to Meuhedet’s data", line 228).

### 2.5 Meuhedet outcome data context [EXPLICIT]
- MEFI (Hershkowitz Sikron F et al., Aging 2024;16:13025–38, doi:10.18632/aging.206141, PMC11552639), line 217: "Hospitalization was identified using invoices submitted to the HMO by the hospitals."
- Marom 2026 (PMC13224401) line 39: diagnoses extracted with "ICD-9 codes: 295, 297, 298 and ICD-10 codes: F20-F29, F33.3". [INFERRED] Meuhedet data may contain both ICD-9-CM-based and ICD-10 codes, so an ICD-10 → ICD-9-CM-only translation may be incomplete. Confirm the code systems per data source (community diagnoses, ED, hospital invoices).
- Whether hospital invoices or ED records carry diagnosis codes, and in which system: [NOT FOUND].

---

## 3. Alcohol categories

### 3.1 eFalls [EXPLICIT]
- Table S3.2 (supp.md lines 242–248): Harmful drinking 0.4164064; Higher risk drinking 0.1549725; **Lower risk drinking = Reference**; Previous higher risk/harmful drinking 0.0849676; Zero alcohol 0.0070124; Missing −0.0679367.
- Table 1 (fulltext.md lines 179–185), SAIL: harmful 4,714; higher 686; lower 11,231; previous 90; zero 1,247; **missing 642,449 (97.3%)**. Bradford missing 66,193 (81.0%).
- HTA Table 1 (gjac1008_struct.txt lines 596–601) and Table 15 (lines 969–974) print "Higher-risk drinking" twice. The second instance has the lower-risk count 11,231 (Table 1) and "Reference" (Table 15). [EXPLICIT; INFERRED as a label error, resolving open issue (iv)]
- No eFalls definition of the categories (units, AUDIT, codes, window). [NOT FOUND]
- eFI2 Appendix 1 gives "Alcohol (harmful, high, low, previous harmful/higher, zero, missing) †" with a 5-year window (efi2_supp.txt line 8; efi_codelists.md §2.2).

### 3.2 UK authoritative definitions [EXPLICIT]
- **UK CMOs’ Low Risk Drinking Guidelines (Aug 2016)**, https://assets.publishing.service.gov.uk/media/5a80b7ed40f0b623026951db/UK_CMOs__report.pdf (landing page https://www.gov.uk/government/publications/alcohol-consumption-advice-on-low-risk-drinking), local `TR/uk_cmo_low_risk_drinking_2016.txt` lines 95–101 (p.4): "To keep health risks from alcohol to a low level it is safest not to drink more than 14 units a week on a regular basis." The guideline applies to men and women and advises spreading drinking over 3+ days.
- **NICE PH24 glossary** (current page), https://www.nice.org.uk/guidance/ph24/chapter/Glossary ; local `TR/nice_ph24_glossary.txt` lines 83–94:
  - Harmful drinking (high-risk drinking): "A pattern of alcohol consumption that is causing mental or physical damage", with "Drinking 35 units a week or more for women. Drinking 50 units a week or more for men."
  - Hazardous drinking (increasing risk drinking): "more than 14 units a week, but less than 35 units a week for women … less than 50 units for men"; "It is not a diagnostic term."
  - Higher-risk drinking: "Regularly consuming over 50 alcohol units per week (adult men) or over 35 units per week (adult women)."
  - Lower-risk drinking: not regularly more than 14 units/week (CMO).
- **GOV.UK "Delivering better oral health – Chapter 12: alcohol"** (updated 10 Sep 2025), https://www.gov.uk/government/publications/delivering-better-oral-health-an-evidence-based-toolkit-for-prevention/chapter-12-alcohol (fetched): "'Low risk' is not regularly exceeding 14 units per week". Increasing risk is >14 up to 35 (women) / 50 (men); higher risk is >35 / >50.
- **OHID AUDIT-C** (https://www.gov.uk/government/publications/alcohol-use-screening-tests ; PDF https://assets.publishing.service.gov.uk/media/6357a7d7e90e0777a45a9caa/Alcohol-use-disorders-identification-test-for-consumption-AUDIT-C_for-print.pdf ; local `TR/gov_auditc.txt`):
  - Lines 38–42: "A total of 5 or more is a positive screen"; "0 to 4 indicates low risk"; "5 to 7 indicates increasing risk"; "8 to10 indicates higher risk"; "11 to 12 indicates possible dependence".
  - Full AUDIT, lines 104–107: 0–7 low risk; 8–15 increasing risk; 16–19 higher risk; 20+ possible dependence.
  - Binge item: "6 or more units if female, or 8 or more if male" (lines 26–28).
- [INFERRED] The eFalls/Baseline2 labels combine NICE "harmful" with "higher risk" as if they were different tiers, and eFalls has no "increasing risk" tier. So eFalls' "higher risk drinking" is **not** the NICE/OHID "higher-risk" tier (>50/>35 units), which NICE equates with harmful.

### 3.3 Code-level concepts [EXPLICIT]
- **Read v2 (SAIL-era)**, ELAStiC Wales protocol supplement S2 (Kennedy et al., IJPDS 2019, doi:10.23889/ijpds.v4i1.581; https://ijpds.org/article/view/581/2028 ; local `TR/ijpds_581_s2.txt`):
  - lines 17–19, 24: `136a.` Increasing risk drinking; `136c.` Higher risk drinking; `136d.` Lower risk drinking
  - line 39: `1361.` Teetotaller; line 40: `136M.` Current non drinker; line 41: `1367.` Stopped drinking alcohol
  - lines 50–52: `136S.` Hazardous alcohol use; `136T.` Harmful alcohol use; `136W.` Alcohol misuse
  - line 54: `136V.` Alcohol units per week
  - lines 56–60: `136A.`–`136E.` Ex-trivial/light/moderate/heavy/very heavy drinker
  - lines 42–48: `1364.` Moderate drinker 3-6u/day; `1365.` Heavy 7-9u/day; `1366.` Very heavy >9u/day; `136R.` Binge drinker
  - ClinicalCodes res61 (Parisi 2017; `TR/cc_res61.txt` lines 44–56, 99–131) also lists `1361.11` "Non drinker alcohol" and `1361.12` "Non-drinker alcohol".
- **SNOMED CT UK, NHS Digital refset ALC_COD** via OpenCodelists (https://www.opencodelists.org/codelist/nhsd-primary-care-domain-refsets/alc_cod/20260630/ ; "Taken from the ALC_COD refset published by NHSD"; SNOMED CT UK Clinical Edition 42.2.0; fetched 2026-09-14; 58 codes). Selected:
  - 777671000000105 Lower risk alcohol drinking; 777631000000108 Increasing risk alcohol drinking; 777651000000101 Higher risk alcohol drinking
  - 198421000000108 Hazardous alcohol use; 198431000000105 Harmful alcohol use
  - 105542008 Current non-drinker of alcohol; 783261004 Lifetime non-drinker of alcohol; 228274009 Lifetime non-drinker; 82581004 Ex-drinker; 160579004 Stopped drinking alcohol; 286857004 Ex-problem drinker
  - 1082641000000106 Alcohol units consumed per week; 1082631000000102 Alcohol units consumed per day
  - 228315001 Binge drinker
  - Related refsets: AUDITC_COD (https://www.opencodelists.org/codelist/nhsd-primary-care-domain-refsets/auditc_cod/20200812/) and AUDIT_COD.
  - [NOTE] The OpenCodelists page text was read through an automated fetch summariser. Spot-checked IDs (160575005, 43783005, 86933000, 228315001) match Baseline2. Re-verify all IDs against a SNOMED CT UK release before use.
- **Baseline2 (DynAIRx; eFI2/eFI+ SNOMED basis)**, local `SP/research/raw/efi/Baseline2_Codelist_github.csv` rows 114–241 (128 rows; `ucl-alcohol` 117, `mar-alcohol` 9, blank 2; all with a 5-year window). Category assignment in `Otherinstructions`:
  - Zero alcohol: 105542008 labelled "Teetotaller" (row 114). ALC_COD displays 105542008 as "Current non-drinker of alcohol".
  - Lower risk: Social drinker, Light drinker, Trivial drinker <1u/day, Light drinker 1-2u/day, "Alcohol intake within recommended sensible limits" (rows 115–120).
  - Higher risk: "Moderate drinker - 3-6u/day", "Moderate drinker" (rows 121–122).
  - Harmful: Heavy/Very heavy drinker, Binge drinker, Alcohol misuse, **"Hazardous alcohol use"** (row 145), "Harmful alcohol use" (146), "Alcohol intake above recommended sensible limits" (144), dependence/alcoholic liver disease/psychosis diagnoses, brief intervention and referral codes (rows 123–227).
  - Previous higher risk/harmful: Ex-moderate/heavy/very heavy drinker, alcoholism in remission, H/O alcoholism, dependence resolved, Alcoholics anonymous (rows 228–237).
  - Numeric (rows 238–241): "0 = code as zero alcohol; 1-20 = code as lower risk drinking; 21-48 = code as higher risk drinking; 49+ = code as harmful drinking". For units per day: "Multiply by 7 to get weekly units."
  - Rows 238/239 carry float-corrupted IDs `10826410000001000` / `10826310000001000`. The correct IDs per ALC_COD are 1082641000000106 / 1082631000000102. [EXPLICIT comparison]
  - **Absent from Baseline2** (grep of the whole file): "Lower/Increasing/Higher risk alcohol drinking" concepts, AUDIT/AUDIT-C score observables, "Lifetime non-drinker", "Ex-drinker" 82581004, "Stopped drinking alcohol", ex-trivial/ex-light drinker. [EXPLICIT, computed]
  - [INFERRED] The 21 and 49/50 cut-offs resemble pre-2016 male weekly thresholds rather than the 2016 CMO 14-unit guideline or the sex-specific 35/50 NICE thresholds. The source of "ucl-alcohol" is not documented. [NOT FOUND]

### 3.4 Candidate translation (for validation) [INFERRED]
- Baseline2 (the only public operationalisation) is a 5-year, code-based category with numeric units/week. Its precedence rule when several categories occur is **not documented** [NOT FOUND]; plausible options are most recent vs most severe.
- Meuhedet: no public information was found on how alcohol intake is recorded (units, AUDIT, free text, diagnoses) [NOT FOUND]. Candidate derivation hierarchy for validation:
  - (a) alcohol-use-disorder diagnoses (ICD-9-CM 291, 303, 305.0, 571.0–571.3; or ICD-10 F10, K70) in 5 years → harmful
  - (b) past-only disorder/remission codes → previous higher-risk/harmful
  - (c) structured intake: 0 → zero; 1–20 units/week → lower; 21–48 → higher; ≥49 → harmful (Baseline2 cut-offs; convert Israeli standard drinks to UK units, 10 ml ethanol/unit per NICE glossary line 108)
  - (d) none → missing
- Given 97.3% missing in SAIL, the model's alcohol terms mostly act through "missing" (−0.068).

---

## 4. Smoking status derivation

### 4.1 eFalls [EXPLICIT]
- Table S3.2 (supp.md lines 238–240): "Ex/never | Reference"; "Current | 0.0684529". Box S3.1: "+ 0.068 (if current smoker)".
- Table 1 (fulltext.md lines 175–178), SAIL: never 302,363 (45.8%); ex 271,248 (41.1%); current 86,806 (13.1%). These sum to 660,417 exactly, with **no missing row**.
- Bradford: never 59,295; **ex 16 (0.0%)**; current 22,390. The Bradford never total does not equal its outcome-stratified components (1,717 + 57,562 = 59,279; diff 16). [EXPLICIT arithmetic]
- [INFERRED] Missing smoking status was folded into never or ex in SAIL, and ex-smoking was effectively not captured in Bradford. The model contrast is therefore "current vs everything else".
- eFI2: "smoking status (reference category = none, ex, or missing)" (PMC11957239 @~12131).
- Baseline2 rows 7026–7078: "Smoker (current)" 37 codes (e.g. 77176002 Current smoker, 65568007 Smoker, 160603005–160606002 light to very heavy smoker, 230059006 Occasional smoker, 266929003 Smoking restarted) and "Smoker (ex)" 16 codes (e.g. 8517006 Former smoker, 281018007 Ex-cigarette smoker, 160617001 Stopped smoking). The instructions say "Cannot be current and ex smoker" and give no time window. **No never-smoker codes.**
- eFalls conflict or recency rule: [NOT FOUND].

### 4.2 UK EHR conventions [EXPLICIT]
- **Atkinson MD et al.** "Development of an algorithm for determining smoking status and behaviour over the life course from UK electronic primary care records". BMC Med Inform Decis Mak 2017;17:2. doi:10.1186/s12911-016-0400-6. PMC5217540. **SAIL Databank, Read v2.** Local `TR/PMC5217540.txt` (single-line):
  - @~16093: "Another problem is the use of ‘never-smoker’ status after previous instances of codes implying smoking"
  - @~17858: EX_SMOKER = "The patient is a self-identified former smoker and those that categorise themselves as never smokers but have a past-coded experience of smoking, are identified as ex-smokers."
  - @~20354: census date after all codes: "since the most recent status is “NEVER_SMOKER” … there is recorded history of “EX_SMOKER” and/or a “SMOKER” status causing the status to be “EX_SMOKER”."
  - @~2677: "Our algorithm assigns ex-smoker status to 34% of never-smokers"
  - @~34597: the parent code 137.. "Tobacco consumption" is used alone or with values, so it is uninformative.
  - Also defines RELAPSED_SMOKER and LIKELY_SMOKER classes (@~17858).
- **OpenSAFELY** (Williamson EJ et al., Nature 2020;584:430–6, PMC7611074; code https://github.com/opensafely/risk-factors-research/blob/master/analysis/study_definition.py, fetched):
  - `smoking_status`: "S": most_recent_smoking_code = 'S'; "E": most_recent_smoking_code = 'E' OR (most_recent_smoking_code = 'N' AND ever_smoked); "N": most recent 'N' AND NOT ever_smoked; "M": DEFAULT
  - most_recent_smoking_code uses `find_last_match_in_period=True`, `on_or_before="2020-02-01"`; ever_smoked filters categories "S","E"
  - Paper (`TR/PMC7611074.txt` @~38324): "those with missing smoking information were assumed to be non-smokers".
- **QOF-style business rules** (DHSSPS Northern Ireland, "Data and Business Rules – Smoking Indicator Set", v30.0NI, 24/10/2014; https://www.health-ni.gov.uk/sites/default/files/publications/dhssps/smok-ruleset-v30.pdf ; local `TR/ni_smok_ruleset_v30.txt`):
  - SMOK_COD selected as "Latest < REF_DAT" (line 243)
  - NSMOK_COD (1371./XE0oh "Code for never smoked") is "Most recent of SMOK_COD" (lines 251–253)
  - Rule notes, line 447: "identify any patient whose most recent smoking status is ‘current smoker’".
  - [INFERRED] QOF rules check recording currency. They are not a research status algorithm.

### 4.3 Candidate rule sets for Meuhedet [INFERRED]
- **S1 (primary):** most recent smoking record on or before index. Current if that record = current. Otherwise not current (ex/never/missing pooled as the eFalls reference).
- **S2:** as S1, but restrict to a look-back window (e.g. 5 years). Baseline2/eFI2 give no window for smoking [EXPLICIT absence].
- **S3:** Atkinson/OpenSAFELY recoding (never after ever → ex). It is **irrelevant to the eFalls LP**, because ex and never share the reference.
- Meuhedet's smoking field structure (structured status vs diagnosis codes such as ICD-9-CM 305.1 tobacco use disorder / V15.82 history of tobacco use): [NOT FOUND].

---

## 5. BMI categories

### 5.1 eFalls statements [EXPLICIT]
- Box S3.1 (supp.md line 318): "underweight if BMI < 18.5; normal weight if 18.5 ≤ BMI < 24.9; obese if BMI ≥ 40". Overweight is not defined.
- Table S3.2 (lines 231–236): Underweight 0.4896735; Normal 0.2394177; **Overweight Reference**; Obese −0.0411134; Missing −0.1451981.
- HTA Box (gjac1008_struct.txt line 1052): "weight groups are defined by standard BMI cut-offs".
- Baseline2 rows 5946–5958 ("Obesity"): "BMI 30+ = obesity, BMI <18.5 Underweight". Row 611 "Body mass index" 60621009 has a 5-year window.
- [INFERRED, plausibility] Table 1 SAIL BMI among recorded values (429,910): underweight 2.9%, normal 28.4%, overweight 36.9%, **obese 31.8%**; Bradford obese 27.8%.
  - An obesity prevalence of about 30% in adults 65+ fits a WHO ≥30 threshold. BMI ≥40 (class III) is usually a small minority.
  - So "≥ 40" in Box S3.1 is most likely a transcription of the WHO class III row, and "< 24.9" a rendering of WHO "18.5–24.9" (i.e. < 25.0).
- eFalls BMI window and the "latest value" rule: [NOT FOUND].

### 5.2 WHO definitions [EXPLICIT]
- WHO/Europe "Nutrition for a healthy life – WHO recommendations", https://www.who.int/europe/news-room/fact-sheets/item/nutrition---maintaining-a-healthy-lifestyle (fetched 2026-09-14), adults over 20: Below 18.5 "Underweight"; 18.5–24.9 "Normal weight"; 25.0–29.9 "Pre-obesity"; 30.0–34.9 "Obesity class I"; 35.0–39.9 "Obesity class II"; Above 40 "Obesity class III".
- WHO fact sheet "Obesity and overweight" (updated 8 Dec 2025), https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight: "overweight is a BMI greater than or equal to 25"; "obesity is a BMI greater than or equal to 30".
- WHO NLIS help page (https://apps.who.int/nutrition/landscape/help.aspx?menu=0&helpid=420) cites WHO Technical Report Series No. 854 (1995) for <18.5 underweight, 18.5–24.9 normal, ≥25 overweight, ≥30 obesity.

### 5.3 Other conventions (for candidate rules) [EXPLICIT]
- OpenSAFELY: "BMI was ascertained from weight measurements within the last 10 years, restricted to those taken when the patient was over 16 years old" (PMC7611074 @~32038). Code: `most_recent_bmi(on_or_after="2010-02-01", minimum_age_at_measurement=16)`.
- Meuhedet (Hershkowitz Sikron F et al., Aging 2025, doi:10.18632/aging.206247, PMC12151512; `TR/PMC12151512.txt` line 246): "The BMI is based on the last height and weight measures recorded in the electronic health record in the HMO. … underweight - less than 18.5, normal 18.5–<25, overweight 25–<30, and obese 30+."
- **Candidate** [INFERRED]: most recent BMI (measured or computed from height and weight) within 5 years before index, age ≥ 18 at measurement. WHO cut-offs <18.5 / 18.5–<25 / 25–<30 / ≥30, with none recorded → missing. Sensitivity analysis: obese ≥40 as literally printed.

---

## 6. Contradictions and inconsistencies (this task)
1. **"Sub-sub-chapter" unit:** eFI2 Table 2 says "(level 3)" but eFI2 Appendix 1 says "sub chapters". eFalls gives no level number. (PMC11957239; efi2_supp.txt line 60; supp.md line 322)
2. **Window:** eFalls 120 days; eFI2 90 days; ARC GM pilot 93 days (descriptive); MEFI 12 months with 8+ drugs.
3. **Box S3.1 main text vs footnote:** "count of unique drugs" vs "Unique BNF sub-sub-chapters … only counted once" (supp.md 318 vs 322).
4. **GEM vs ICD-9-CM tabular for WHO T10:** tabular 818 incl. "arm NOS", while GEM maps 818.x → S62.90 (wrist/hand). (`TR/dtab/Dtab12.txt` 25914–25922; `TR/2018_I9gem.txt`)
5. **E887 "Fracture, cause unspecified":** inside the ICD-9-CM "Accidental falls" block and GEM → W19XXXA "Unspecified fall", but the WHO ICD-10 analogue is X59.0 (not a fall code).
6. **E886.9:** GEM → ICD-10-CM V00 (pedestrian conveyance), but WHO W03 explicitly includes pedestrian collision falls.
7. **M80 vs 733.1x:** backward GEM maps M80 → 733.1x, while forward GEM maps 733.1x → mostly M84.4x (pathological fracture NEC).
8. **HTA alcohol labels:** Table 1 (lines 596–601) and Table 15 (969–974) duplicate "Higher-risk drinking". Counts show the second row is "Lower-risk drinking" (Age & Ageing Table 1 line 182; Table S3.2 line 245).
9. **Baseline2 vs NICE/OHID alcohol tiers:** "hazardous alcohol use" is placed in *harmful*, and "moderate drinker 3–6 u/day" in *higher risk*. Numeric cut-offs 21/49 differ from NICE 14/35/50.
10. **SNOMED 105542008:** labelled "Teetotaller" in Baseline2 but "Current non-drinker of alcohol" in NHSD ALC_COD.
11. **Bradford smoking:** ex = 16; never total 59,295 ≠ 59,279 (sum of strata).
12. **BMI:** Box S3.1 "obese ≥ 40" vs HTA "standard BMI cut-offs" vs Baseline2 "30+" vs WHO (≥30; ≥40 = class III); distributional evidence favours ≥30.
13. **PCA methodology vs 2017 booklet wording:** PCA calls 18–23 "six pseudo BNF chapters", while the booklet marks 18 and 19 as "Not used by BNF". This is consistent in substance, but chapter 19 contains genuine drug-like items (e.g. poisoning antidotes, single substances).

## 7. Unresolved
1. Exact eFalls polypharmacy implementation:
   - BNF level (6 vs 7 digits)
   - chapters excluded
   - BNF edition/year
   - Read v2 drug code → BNF mapping in SAIL (WLGP) and SNOMED/dm+d → BNF in Bradford
   - repeat vs acute handling
   - Resolution: request the eFalls technical specification/code lists from the corresponding author (Age & Ageing data statement, supp.md line 3); obtain the NHSBSA historic BNF code file for 2018.
2. How the Welsh EDDS and PEDW diagnoses were coded, and the diagnostic position used for the outcome. Resolution: eFalls authors; SAIL EDDS documentation.
3. Whether a WHO ICD-10 T14.2 → ICD-10-CM T14.8 inclusion ("Fracture NOS") holds. Only secondary code-lookup sites were seen; verify against the CMS/CDC FY ICD-10-CM tabular.
4. Meuhedet coding facts:
   - which ICD version per data stream (community diagnoses, ED, hospital invoices)
   - whether E-codes are stored with the "E"
   - the meaning of "880" in the MEFI falls list
   - local extension codes
   - Resolution: Meuhedet data team / MEFI authors.
5. The Meuhedet medication dictionary: ATC version, route, combination handling, purchase vs prescription data, pack-supply duration.
6. Meuhedet recording of alcohol intake and smoking (structured fields vs codes); BMI source (measured vs computed).
7. Provenance of Baseline2 alcohol cut-offs (`ucl-` tag) and category precedence; eFalls rules for alcohol/smoking/BMI windows.
8. WHO ICD-10 placement of nonmotorized scooter falls (ICD-9-CM E885.0): not verified.

## 8. Assets
| Asset | Availability | URL / path |
|---|---|---|
| NHSBSA BNF code file (Aug 2026, v90) | public | https://opendata.nhsbsa.net/dataset/bnf-code-information-current-year ; `TR/bnf_code_current_202608_v90.csv` |
| NHSBSA BNF code file, historic (2018-era) | public (not downloaded) | https://opendata.nhsbsa.net/dataset/bnf-code-information-historic |
| NHSBSA BNF classification booklet 2017 | public | https://www.nhsbsa.nhs.uk/sites/default/files/2017-04/BNF_Classification_Booklet-2017.pdf |
| WHOCC ATC/DDD Guidelines 2026 | public (copyright WHOCC; no commercial copying) | https://atcddd.fhi.no/filearchive/publications/2026_guidelines_for_atc_classification_and_ddd_assignment.pdf |
| Full ATC index with DDDs (electronic) | licensed (ordered from WHOCC) | https://atcddd.fhi.no/ (the "Order ATC Index" link on the site) |
| CDC/CMS 2018 GEMs + guides | public domain | https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2018/ |
| ICD-9-CM tabular FY2012 (Dtab12) | public domain | https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD9-CM/2011/ |
| WHO ICD-10 2019 browser | public | https://icd.who.int/browse10/2019/en |
| eFalls outcome ICD-9-CM candidate table (this work) | local | `SP/research/translation_work/efalls_outcome_icd9cm_candidates.csv` (script `…/efalls_icd9cm_candidates.py`) |
| NHSD ALC_COD / AUDITC_COD refsets | public (OGL v3) | https://www.opencodelists.org/codelist/nhsd-primary-care-domain-refsets/alc_cod/20260630/ |
| eFalls official code lists / tech spec | on_request | Age & Ageing supplement data statement (supp.md line 3); HTA line 412 |
| Baseline2 SNOMED list (alcohol/smoking/BMI rules) | public, IP unclear | https://github.com/DynAIRx/GCAF_DynAIRx/blob/main/Input/Baselines/Baseline2_Codelist.csv |
| MEFI ICD-9/ATC deficit definitions (Meuhedet) | public (CC BY) | PMC11552639 |
| Atkinson 2017 SAIL smoking algorithm (SQL flow diagram) | public (CC BY) | PMC5217540 |

## 9. Sources consulted (additional to those in efi_codelists.md)
- NHSBSA: booklet 2017 (URL above); PCA methodology June 2025 https://nhsbsa-opendata.s3.eu-west-2.amazonaws.com/pca/pca_background_info_methodology_june2025_v001.html ; BNF Code Information dataset (URL above).
- OpenPrescribing wiki https://github.com/ebmdatalab/openprescribing/wiki/BNF-for-dummies
- WHOCC: guidelines 2026 PDF; https://atcddd.fhi.no/atc/structure_and_principles/ ; https://atcddd.fhi.no/atc_ddd_methodology/purpose_of_the_atc_ddd_system/ ; ATC index pages for A10B, C09, N05A, V, V04, V06, V07, V07A, V08, V20 (local `TR/atc_*.txt`).
- WHO ICD-10 2019 browser concept pages (W00–W19, T08–T14, M80, X59).
- CDC NCHS: ICD-10-CM 2018 order file, GEMs 2018 + guide + tech doc; ICD-9-CM 2011 tabular. CMS ICD-9-CM v32 descriptions.
- MCHP concept http://mchp-appserv.cpe.umanitoba.ca/viewConcept.php?conceptID=1168
- UK CMO 2016 guideline PDF; NICE PH24 glossary; GOV.UK Delivering better oral health ch.12; OHID alcohol screening tests / AUDIT-C.
- IJPDS ELAStiC Wales S2 (doi:10.23889/ijpds.v4i1.581); ClinicalCodes res61 (Parisi 2017; /medcodes/article/61/codelist/res61-alcohol-consumption/); OpenCodelists ALC_COD; OpenSAFELY hazardous-alcohol-drinking (https://www.opencodelists.org/codelist/opensafely/hazardous-alcohol-drinking/6364474b/).
- Atkinson 2017 (PMC5217540); OpenSAFELY Nature 2020 (PMC7611074) + GitHub study_definition.py; DHSSPS NI smoking ruleset v30.0NI.
- WHO/Europe nutrition fact sheet; WHO obesity fact sheet; WHO NLIS help page.
- Meuhedet: Hershkowitz Sikron 2024 (PMC11552639); Hershkowitz Sikron 2025 (PMC12151512); Marom 2026 (PMC13224401).
- Woodcock 2024 PLOS ONE polypharmacy (PMC11373791) was checked. It describes BNF-chapter counting as a known limitation and was not used for definitions.
