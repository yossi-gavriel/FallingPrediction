# eFalls results in the NIHR HTA eFI+ monograph (GJAC1008), cross-checked against Age & Ageing 2024

Compiled 2026-09-14. Scope: HTA Chapter 5 (falls model), Appendix 4, Chapter 9 Discussion, plus the parts of Chapters 2 and 3 needed to interpret them. Everything is cross-checked against Archer et al., Age Ageing 2024;53(3):afae057 (main text and supplement).

## Sources and abbreviations

| Abbrev | Source | Local file |
|---|---|---|
| HTA | Archer L et al. *Development and evaluation of the eFI+ tool...* Health Technol Assess 2026;30(61). https://doi.org/10.3310/GJAC1008 ; https://www.journalslibrary.nihr.ac.uk/hta/GJAC1008 | `raw/gjac1008_struct.txt` (cited as HTA Lnnn), `raw/hta_monograph_GJAC1008.html` |
| NCBI | NCBI Bookshelf copy of the same monograph: Chapter 5 = https://www.ncbi.nlm.nih.gov/books/NBK623967/ ; Appendix 4 = https://www.ncbi.nlm.nih.gov/books/NBK623977/ ; Methods = NBK623980 ; Discussion = NBK623965 | `raw/efiplus_*.html`; per-table pages fetched this round into `hta_falls_work/NBK6239xx_table_tableNN.html`, parsed into `hta_falls_work/bookshelf_tables.txt` and `hta_falls_work/html_tables.txt` |
| AA | Age & Ageing main text (PMC10960070) | `<research workspace (not distributed)>/efalls/fulltext.md` (AA Lnnn) |
| AAS | Age & Ageing supplement | `<research workspace (not distributed)>/efalls/supp.md` (AAS Lnnn); equations are in `research/supp_fulltext_with_math.txt` (AASM Lnnn) |

Scripts (all in `research/hta_falls_work/`, run with Python 3.13): `compare_coefs.py`, `worked_examples.py`, `compare_perf2.py`, `compare_cohort.py`, `lp_distribution_checks.py`, `recal_from_test_accuracy.py`, `mean_lp_sex_coding.py`.

**Rendering caveat:** the journalslibrary HTML (and so `gjac1008_struct.txt`) drops the probability formulas in Tables 16 and 34; it prints only "Probability ... =". The formulas do appear in the NCBI Bookshelf per-table pages as MathML:
- Table 16: exp(LP)/(1+exp(LP))
- Table 34: exp(-0.423+1.25*LP)/(1+exp(-0.423+1.25*LP))

---

## 0. Executive summary (most important points for reproduction)

1. **The monograph's falls tables match Age & Ageing exactly.**
   - Table 15 = Table S3.2.
   - Tables 17/18/36 = Table 2.
   - Table 31 = S3.3.
   - Table 32 = S3.4 (test accuracy table); Table 33 = S3.4 (subgroup table).
   - Table 37 = S3.5; Table 14 = Table 1 (outcome-stratified columns).
   - Table 16 = Box S3.1; Table 34 = Box S3.2.
   - The only differences are labels, rounding and wording (section 4).
   - The local CSVs `hta_table15_efalls_coefficients.csv` and `hta_table35_recalibrated_coefficients.csv` match the HTA text and NCBI HTML with 0 differences in 80/80 rows.
2. **Sex term (issue i):** the monograph does not resolve it. It reprints both versions:
   - Table 15: Male = Reference, Female = -0.303708, Constant = -5.954459.
   - Tables 16/34: LP = -6.258 ... "- 0.304 (if male)". The worked example (a female) uses -6.258 with no sex term.
   - There is no erratum or clarifying statement.
   - One new HTA sentence (L947) says "being female" contributed to a higher predicted risk. That agrees with the Box direction, not the Table.
   - Independent numerical checks (inferred, section 7) show that the published LP means are inconsistent with the Table coding and consistent with an effective LP about 0.28 to 0.30 lower. The published means are -3.30 in SAIL (Fig 22) and -2.99 in Connected Bradford (Fig 40). This is what the Box literal equation gives.
3. **Recalibration (issue ii):** the monograph contradicts itself.
   - Table 34 (NCBI MathML) gives p = expit(-0.423 + 1.25 x LP), the same as Box S3.2.
   - Table 35 coefficients are exactly 1.21 x every Table 15 coefficient (all 75), with constant = -0.046 + 1.21 x (-5.954459) = -7.25089539. The implied alpha is exactly -0.046 and beta 1.21.
   - Three independent checks show that the reported recalibrated results (Table 36, Table 37, Figure 40) were produced by alpha about -0.42 and beta about 1.25, not by Table 35 (section 5).
   - Table 35 is presented as "Coefficients of the eFalls model when recalibrated for Connected Bradford". Its origin is not explained.
4. **Neither recalibrated version is recommended for use.** Both papers recommend the externally validated original eFalls model. The recalibrated model had apparent validation only and is "not currently recommended in practice without further validation" (HTA L1180-1181; AA L305-307).
5. **Worked example precision:** the box example gives 20.3%, but only because it uses rounded coefficients (age 0.042 x 89). With full-precision Table 15 coefficients the same female patient has LP -1.4085 and p = 0.1965. Recalibrated with -0.423/1.25 this gives 0.1012, not 0.106. With Table 35 it gives 0.148.
6. **Other internal inconsistencies in the monograph:**
   - 74 predictors (Ch 5 L947, Fig 21 caption) vs 75 (HTA abstract L2624; AA L191; Table 15 has 75 non-reference parameters).
   - Bootstrap 50 samples (HTA Methods L539) vs 25 (AA L118).
   - Connected Bradford index date 1 April 2018 (HTA L360, L558) vs 1 January 2019 (HTA Figure 1; AA L80).
   - Connected Bradford cohort 88,947 with 3,079 events (HTA L459, L558, Table 8, Fig 1) vs 81,685 with 2,389 events used for falls EV (Table 14; AA L144).
   - CARE75+ n = 267 (L551) vs 252 (L1161).
   - HTA Table 16 describes BMI groups as "standard BMI cut-offs" (L1052). Table 34 and both AA Boxes say "obese if BMI ≥ 40" (L2076).

---

## 1. Cohorts, events, prevalence, deaths, predictor prevalences

### 1.1 Counts (explicit)

| Item | HTA | AA | Status |
|---|---|---|---|
| Development cohort (SAIL) | 660,417; 455 GPs; 32,097 fall/fracture (4.9%) (L555; Table 14 L903; Table 8 L569, L676) | same (AA L144; Table 1 L152) | agree |
| SAIL non-events | 628,320 (95.1) (L903) | same (AA L152) | agree |
| SAIL deaths within 12 months | 25,148 (3.8%) (L555; Table 8 L678; Fig 1) | "3.8%" (AA L313) | agree |
| SAIL care-home admissions | 5,362 (0.8%) (L555) | n/a | HTA only |
| SAIL age | mean 74.9, SD 7.5 (L556); median 73 (69–80) (Table 8 L571) | median 73 [69 to 80] (AA L154) | HTA adds mean/SD |
| Falls EV cohort (Connected Bradford) | 81,685; 2,389 events (2.9%); non-events 79,296 (97.1) (Table 14 L903; Ch 8 L1654, L1694) | same, "across 76 practices" (AA L144) | agree. The 76-practice count is in AA only |
| Connected Bradford cohort in Methods/Ch 3 | 88,947 aged ≥65 with valid sex, "As of 1 April 2018"; 3,079 fall/fracture (3.5%); 3,856 deaths (4.3%); 898 care-home admissions; 1,113 care packages; 84 GPs included of 86 (L459, L558, Table 8 L569/L676-678, Figure 1 `raw/hta_fig1.png`). Figure 1 says "Registered with Connected Bradford Database on 1 January 2019: 86 GP, 229,572 patients" | EV cohort 81,685 / 2,389 (AA L101, L144). AA index date 1 January 2019 (AA L80). Sample-size text also says "Connected Bradford contained 81,685 participants with 2,389 events" (AA L101) | **DISCREPANCY.** The HTA sample-size paragraph (L459) cites 88,947/3,079 for the falls EV; Chapter 5 uses 81,685/2,389. Nothing in the HTA explains the 7,262-person difference |
| EV deaths | "4.3% in the EV data" (L1725) = 3,856/88,947 (Table 8) | "4.3% in the external validation data" (AA L313) | Deaths in the 81,685 falls-EV cohort specifically: **not_found** (4.3% matches the 88,947 cohort) |
| Development LP summary | Fig 22: mean -3.30, median -3.37, SD 0.83, LQ–UQ -3.92 to -2.77 (`raw/hta_fig22.png`) | Table S2.2: LP mean -3.30, variance 0.690, skew 0.5 (0.499 recorded), kurtosis 3 (2.943) (AAS L78, L84); also HTA Table 4 L450 | agree |
| EV LP summary before/after recalibration | Fig 40: before mean -2.99, SD 0.905, median -3.12, IQR 1.26; after mean -4.16, SD 1.13, median -4.31, IQR 1.58 (`raw/hta_fig40.png`) | Fig S3.11 (image, not in text) | HTA numbers read from the figure image |

### 1.2 Table 14 (HTA) vs Table 1 (AA)

Checked with `compare_cohort.py` plus visual review of the median rows.
- All outcome-stratified values are identical.
- HTA Table 14 omits the "Total" columns.
- **Label error (issue iv):** HTA Table 14 row L933 reads "Higher-risk drinking | 603 (1.9) | 10,628 (1.7) | 168 (7.0) | 5635 (7.1)". AA Table 1 prints exactly these numbers as "**Lower risk drinking**" (AA L182). The same mislabel appears in HTA Table 8 (L599), Table 15 (L972) and Table 35 (L2104), and in the NCBI HTML. It is a monograph typesetting/label error; the reference category is lower-risk drinking. (explicit numbers; inferred cause)
- Small-cell footnote: "< 10" for previous higher-risk drinking among SAIL fallers (L934, L941).

Arithmetic errors found in AA tables (not repeated in HTA, because HTA has no totals column):
- AA Table 1, EV IMD 3 total prints 13,337 (16.3), but 473 + 13,864 = 14,337. Table S3.4 and HTA Table 33 give n = 14,337.
- AA Table 1, EV smoking "Never" total 59,295, but 1,717 + 57,562 = 59,279 (difference 16 = the "Ex" count).
- AAS Table S3.1, EV "Housebound" fall/fracture count prints 931 (34.8), but 34.8% of 2,389 = 831, and 831 + 8,100 = 8,931 = printed total. So 931 is a typo for 831 (AAS L167). (inferred)

### 1.3 Predictor prevalences
- HTA Table 8, SAIL column: all 72 frailty-deficit rows are identical to the AAS Table S3.1 development totals (0 differences; `compare_cohort.py`).
- The HTA does not reproduce the outcome-stratified deficit prevalences (Table S3.1). Its Connected Bradford column in Table 8 is the **88,947** cohort, not the falls-EV cohort.
- **HTA Table 8 internal inconsistency:** the CB frailty-category rows (32,732 (40.0); 24,694 (30.2); 14,496 (17.7); 9,763 (12.0); L582-585) and the eFI median 0.17 (0.08–0.25) (L580) are copied from the 81,685 falls-EV cohort; the four frailty counts sum to 81,685. The column header is n = 88,947. (explicit numbers; inferred copy error)
- The 88,947-cohort deficit prevalences differ greatly from the 81,685 falls-EV cohort (S3.1). Median absolute difference 3.9 percentage points; largest differences:

| Deficit | S3.1 EV (81,685) | HTA Table 8 CB (88,947) |
|---|---|---|
| Urinary system disease | 40.1% | 21.6% |
| Dressing and grooming problems | 15.0% | 0.1% |
| General mental health | 22.1% | 7.5% |
| Mobility problems | 19.3% | 6.1% |
| Musculoskeletal problems | 35.7% | 23.6% |
| Thyroid problems | 10.7% | 22.0% |
| Falls (prior) | 22.3% | 11.5% |
| Skin ulcer | 12.5% | 3.1% |

- The prior-falls count in the "larger" cohort (10,219) is lower than in the 81,685 EV cohort (18,182). So these are not nested extracts with the same definitions. The HTA mentions a "wholescale change to the Connected Bradford data model, requiring rebuilding of all data sets and models" (L1723). (inferred)
- **Observation (inferred, not stated by authors):** in the EV cohort S3.1, several deficits are far more common among fallers than in SAIL:

| Deficit | EV fallers | EV non-fallers | SAIL fallers | SAIL non-fallers |
|---|---|---|---|---|
| Prior "Falls" | 77.6% | 20.6% | 32.8% | 15.3% |
| Mobility problems | 62.0% | 18.1% | 3.0% | 1.3% |
| General mental health | 58.5% | 21.0% | 13.4% | 6.4% |
| Urinary system disease | 69.3% | 39.2% | 32.4% | 23.1% |

  This pattern could reflect look-back/extraction differences, possibly including post-index codes, and may partly explain the higher EV C-statistic (0.825 vs 0.72). Worth checking in any reproduction; no source confirms it.
- AAS S3.1 flags 10 candidate deficits as not selected by LASSO: anxiety, chronic kidney disease, dyspnoea, environment problems, heart valve disease, ischaemic heart disease, problems managing finances, shopping problems, toileting problems, TIA (AAS L140-202, L210). 72 − 10 = 62 binary deficits retained, matching Table 15 (`compare_coefs.py`).

---

## 2. Coefficient table and worked example

### 2.1 Table 15 (HTA L949-1044) = Table S3.2 (AAS L218-314)
- All 80 rows: identical coefficients and unpenalised ORs (`compare_coefs.py`).
- Label differences only:
  - "ln[(Polypharmacy + 1)/10]" vs "ln((Polypharmacy+1)/10)".
  - "Higher-risk drinking (Reference)" (HTA) vs "Lower risk drinking (Reference)" (AA).
  - "Motor neuron" vs "Motor neurone".
  - HTA prints "Reference" in the OR column; AA leaves it blank.
- The CSV `raw/hta_table15_efalls_coefficients.csv` equals the HTA text and the NCBI HTML Table 15 exactly (0 differences). It inherits the duplicated "Higher-risk drinking" label (CSV rows 13-14).
- Parameters: **75 non-reference coefficients** plus the constant:
  - age;
  - ln((P+1)/10);
  - female;
  - 4 BMI (underweight, normal, obese, missing; overweight = reference);
  - 1 smoking (current; ex/never = reference);
  - 5 alcohol (harmful, higher-risk, previous higher-risk/harmful, zero, missing; lower-risk = reference);
  - 62 binary deficits.
- Table footnote (HTA L1041): ORs are from a "bootstrapping logistic model refit with only LASSO selected predictor variables". The AA title says "unpenalised logistic model refit". The HTA text (L948) says the unpenalised model is "not intended for use in practice".
- Values to reproduce exactly: Constant -5.954459; Age 0.0415506; ln((P+1)/10) 0.3296295; Female -0.303708 (OR 0.732, 0.712 to 0.753).

### 2.2 Worked example: HTA Table 16 (L1045-1056; NCBI NBK623967/table/table16) = Box S3.1 (AAS L315-322; AASM L1138-1160)
- Formula (NCBI MathML): Probability = exp(LP)/(1+exp(LP)).
- LP as printed: "−6.258 + 0.042 × (age) + 0.330 × {ln[(polypharmacy + 1)/10]} – 0.304 (if male) + 0.490 (if underweight) + 0.240 (if normal weight) – 0.041 (if obese) – 0.145 (if BMI category missing) + 0.068 (if current smoker) + 0.416 (if harmful drinking) + 0.155 (if higher-risk drinking) + 0.085 (if previous...) + 0.007 (if zero alcohol) – 0.068 (if alcohol consumption missing) – 0.064 (abdominal pain) + 0.048 (activity limitation) + 0.173 (anaemia...) + …"
- **The monograph example uses -6.258, not -5.954, and "-0.304 (if male)"**, the same as AA.
- Example patient X: underweight female, 89 years, dementia, liver problems, osteoporosis, 8 medications, non-smoker, previous higher-risk/harmful drinking.
  - LP = -6.258 + 0.042(89) + 0.330 ln(0.9) + 0.490 + 0.085 + 0.104 + 0.380 + 0.128 = -1.368; p = 0.203.
  - No sex term is added for this female patient.
- Pointer fixes/changes vs AA:
  - HTA Table 16 says "(see Table 15 for values)"; AA Box S3.1 wrongly says "[see table S3.1 for values]".
  - HTA Table 16 describes BMI as "weight groups are defined by standard BMI cut-offs" (L1052; same wording for eHomeCare L830 and eCareHome L1498). HTA Table 34 (L2076) and AA Boxes S3.1/S3.2 (AAS L318, L368) instead say "underweight if BMI < 18.5; normal weight if 18.5 ≤ BMI < 24.9; obese if BMI ≥ 40". Overweight is never defined, and 24.9 to 25 is a gap.
  - "Standard" cut-offs would normally mean obese ≥ 30, so the monograph is ambiguous on issue (iii). No numeric overweight definition appears anywhere in the HTA (grep over `gjac1008_struct.txt`). Methods L549 lists only category names.
  - HTA omits AA's polypharmacy footnote: "Unique BNF sub-sub-chapters. Combinations of >1 drug from a sub-sub-chapter only counted once towards the total." (AAS L322, L372; no "sub-sub" string in HTA).
  - HTA Tables 16/34 also drop the word "days" ("over the 120 prior to index date").
- Recomputation (`worked_examples.py`):

| Calculation | LP | p |
|---|---|---|
| Rounded box coefficients | -1.3678 | 0.2030 (reproduces the box) |
| Full-precision Table 15, female (constant -5.954459 + female -0.303708) | -1.4085 | 0.1965 |
| Same patient as male, Table coding | -1.1047 | 0.2489 |
| Same patient as male, Box literal (-6.258167 - 0.303708) | -1.7122 | 0.1529 |

  - The 0.041 LP gap between rounded and full precision comes mainly from the age coefficient rounding (0.042 vs 0.0415506, times 89).
  - The Box intercept -6.258 equals Table constant + Female coefficient = -5.954459 - 0.303708 = -6.258167.

---

## 3. Functional forms, LASSO, number retained

- **FP:** "Assessing up to second-order fractional polynomials, a linear fit was deemed appropriate for age ... while polypharmacy was best modelled with a natural log transformation" (HTA L945). Forms used in LASSO: age in years and ln[(polypharmacy + 1)/10].
  - FP assessment was done in the presence of all candidate predictors, before LASSO (L945; Methods L504: FP up to FP2 for SAIL-developed models). AA L114 matches.
  - Fig 20 (`raw/hta_fig20.png`) panels: "Fractional polynomial (linear), adjusted for covariates" (age axis labelled "years, from week of birth"; plotted ages about 65-95) and "Fractional polynomial (natural logarithm)" (polypharmacy counts 0 to about 60).
  - SAIL age is "based on week of birth ... approximate" (L556).
- **LASSO:**
  - Logistic LASSO; lambda chosen to minimise the 10-fold cross-validation function (L500, L504). L500 adds that the penalty is set by 10-fold CV "to determine the model resulting in the optimal log-likelihood".
  - Clustering by GP was ignored at development (L504). No interaction terms (L504).
  - **λ = 0.000123** (HTA L947 and Fig 21 caption L1963). The value is not given in AA.
  - Fig 21 (`raw/hta_fig21.png`): top panel "Cross-validation plot with 1 SE bounds"; the CV function falls from about 0.39 and plateaus near 0.353 from λ ≈ 0.001 downward, with λCV marked. Bottom panel shows coefficient paths. Printed note: "λCV = 0.000123 is the cross-validation minimum λ; # coefficients = 74".
  - The top-panel x-axis tick labels read 0.01 / 0.001 / 0.001; the third should read 0.0001 (figure typo).
- **Number retained:** HTA Ch 5 "The LASSO regression retained 74 predictors" (L947) and Fig 21 "# coefficients = 74". The HTA abstract (L2624) says 75; AA says 75 (AA L191). Table 15 has 75 non-reference coefficients.
  - Stata's "# coefficients" count may exclude one term (for example a forced-in variable). Why they differ: **not_found**.
- **Direction statement (HTA only, L947):** "Increased age, polypharmacy, being female, being normal or underweight, current smoking and higher or harmful alcohol consumption (whether current or previous) all contributed to a higher predicted fall/fracture risk."
  - "being female" contradicts the printed Table 15 sign (Female -0.303708). See section 7.
- **Stability:**
  - MAPE mean 0.00196 (95% CI 0.00195 to 0.00197); median 0.00117 (LQ–UQ 0.000618 to 0.00236).
  - About 5% of individuals had high MAPE (L1058, L1064).
  - Classification instability at 0.10: up to 80% of bootstrap models reclassify people near the cut-off. At 0.25: up to 60%, affecting originals between 0.1 and 0.4 (L1066-1067).
  - These MAPE and classification numbers appear in the HTA only.
- **Bootstrap count:** HTA Methods: 50 samples (L539). AA: "bootstrapping with 25 samples (chosen for computational efficiency...)" (AA L118). Unresolved.

---

## 4. Performance numbers: HTA vs AA (all compared cell by cell with `compare_perf2.py`)

| HTA table | AA table | Result |
|---|---|---|
| Table 17 (apparent pooled across GPs; IECV) L1073-1099 | Table 2 cols 1-2 (AA L199-265) | identical (0 cell differences) |
| Table 18 (EV overall; pooled across GPs) L1109-1134 | Table 2 cols 3-4 | identical |
| Table 36 (recalibrated apparent) L2173-2198 | Table 2 cols 5-6 | identical |
| Table 31 (apparent / optimism / optimism-adjusted) L1968-1989 | Table S3.3 (AAS L329-347) | identical |
| Table 33 (EV subgroups) L2034-2061 | Table S3.4 subgroup (AAS L393-421) | identical (226 numbers) |
| Table 32 (TP/FP/TN/FN, before recalibration) L2006-2031 | Table S3.4 [duplicate number] (AAS L451-470) | identical |
| Table 37 (after recalibration) L2207-2230 | Table S3.5 (AAS L471-490) | identical |

Key values:
- **Apparent overall (Table 31):**
  - Slope 1.0083 (0.9953933 to 1.021224); optimism 0.0026419; adjusted 1.0057.
  - CITL 0.0000 (−0.0115 to 0.115); optimism 0.000277; adjusted −0.0003.
  - O/E 1.0000; optimism 0.0002313; adjusted 0.9998.
  - C 0.7434 (0.74077 to 0.74612); optimism 0.0004245; adjusted 0.7430.
  - The CITL upper limit "0.115" is almost certainly a typo for 0.0115 (it appears in both sources).
- **Apparent pooled across GPs:** slope 0.99 (0.97–1.01; PI 0.80–1.18; τ² 0.009); CITL 0.154 (0.095–0.212; PI −0.96–1.27; τ² 0.319); O/E 1.19 (1.12–1.26; PI 0.14–2.24; τ² 0.282); C 0.72 (0.72–0.72; PI 0.68–0.76; τ² 0.010).
- **IECV (WIMD groups):** slope 0.99 (0.75–1.22; PI 0.30–1.67; τ² 0.052); CITL −0.13 (−0.66–0.40; PI −1.65–1.39; τ² 0.256); O/E 0.88 (0.53–1.46; PI 0.21–3.70; τ² 0.228); C 0.72 (0.68–0.76; PI 0.59–0.85; τ² 0.002).
  - Fig 28 IECV LP means in omitted WIMD groups 1-6: −3.11, −3.16, −3.18, −3.19, −3.21, −3.33.
  - Fig 28 panel labels for groups 4-6 wrongly say "Group 1 LP details".
- **EV overall:** slope 1.248 (1.244–1.265); CITL −0.931 (−0.938 to −0.920); O/E 0.432 (0.430–0.437); C 0.825 (0.824–0.828).
- **EV pooled across GPs:** slope 1.203 (1.133–1.273; PI 0.858–1.548; τ² 0.029); CITL −0.874 (−0.964 to −0.783; PI −1.375 to −0.372; τ² 0.061); O/E 0.431 (0.388–0.479; PI 0.194–0.958; τ² 0.157); C 0.816 (0.801–0.830; PI 0.715–0.886; τ² 0.078).
- **Recalibrated apparent:**
  - Overall: slope 1.000 (1.000–1.013); CITL 0.000 (−0.009–0.010); O/E 1.000 (0.992–1.009); C 0.825 (0.825–0.828).
  - Pooled: slope 0.964 (0.908–1.020; PI 0.687–1.240; τ² 0.018); CITL 0.064 (−0.027–0.154; PI −0.442–0.569; τ² 0.062); O/E 1.013 (0.916–1.122; PI 0.475–2.163; τ² 0.142); C 0.816 (95% CI printed "0.801 to 1.000"; PI 0.715–0.886; τ² 0.078).
  - The "1.000" upper limit is almost certainly an error for 0.830 (in both sources).
- **EV subgroups (Table 33 = S3.4):** e.g. severe frailty C 0.643 (0.638–0.646); fit O/E 0.201; female O/E 0.417, CITL −0.986; male O/E 0.463, CITL −0.832; missing BMI O/E 0.797. Subgroup n sums to 81,685 for IMD, frailty, sex and BMI.
  - **Point estimates outside their own 95% CI (errors in both sources):** IMD1 slope 1.123 (1.126–1.158); IMD1 C 0.788 (0.789–0.796); mild frailty slope 1.300 (1.234–1.298); mild frailty C 0.735 (0.722–0.733); male slope 1.368 (1.371–1.393); male C 0.834 (0.835–0.839).
  - On a CI boundary: IMD3 C 0.849 (0.844–0.849); overweight C 0.828 (0.828–0.834).
- **Differences in wording and cross-references (HTA vs AA):**
  - IECV missing-WIMD performance: HTA "less impressive" (L1105) vs AA "notably poor" (AA L274).
  - HTA L1138 cites "Appendix 4, Figure 33" for recalibrated net benefit, but Fig 33 is IMD-subgroup plots (AA cites Figure 1).
  - HTA Figure 4/5/25/31 captions say the horizontal line is "treat-all"; it should be treat-none.
  - HTA Ch 9 L1717 and L1756 say grip/gait "did not appear to have a notable impact", while Ch 5 L1182 says "small improvements".
  - HTA abstract (L2624): "Decision curve analysis indicated that the model had higher clinical utility (than other strategies) at higher-risk thresholds". This contradicts Ch 5 L1138/L1178 (EV utility only at thresholds below 10%; no better than treat-none at 10–25%).
  - HTA abstract also gives EV O/E as "0.43 (95% CI 0.42 to 0.44)" vs Table 18 0.432 (0.430 to 0.437).
  - Software: HTA Methods L502 "R version 4.3.1" for Leeds analyses vs HTA L1153 and AA L106 "R version 4.2.3".
- **Sample-size calculation differences:**
  - HTA Table 3 (L425-437): falls model 108 parameters, R² 0.15 → 19,706 (946); R² 0.05 → 60,209 (2,891).
  - AA Table S2.1 (AAS L68-72): 90 parameters, R² 0.15 → 13,867 (666); 0.05 → 50,174 (2,409); 0.049 → 50,927 (2,445); AA L98 cites 50,927/2,445.
  - The EV sample size is the same in both: 10,882 (523) (HTA Table 4 L444-458; AAS L74-84).
- **Test accuracy internal checks (Tables 32/37):**
  - TP+FN = 29.2–29.3 per 1000 = 2,389/81,685, and rows sum to 1000. Sensitivity/specificity agree with the counts to rounding.
  - Net benefit per 1000 derived from the tables: before recalibration −0.17 (t = 0.10) to −3.95 (t = 0.20), i.e. worse than treat-none across 10–25%. After recalibration +3.14 (0.10) down to +0.16 (0.18), and ≤0 from 0.19. This matches the text "superior to other strategies for threshold probabilities up to 18%" (L1138).

Selected rows (per 1000 assessed):

| Threshold | Before recal: TP / FP / TN / FN / Sens / Spec | After recal: TP / FP / TN / FN / Sens / Spec |
|---|---|---|
| 0.10 | 19.2 / 174.3 / 796.5 / 10.1 / 0.66 / 0.82 | 9.1 / 53.6 / 917.2 / 20.2 / 0.31 / 0.95 |
| 0.15 | 13.2 / 91.5 / 879.3 / 16.0 / 0.45 / 0.91 | 5.0 / 23.9 / 946.9 / 24.3 / 0.17 / 0.98 |
| 0.20 | 8.6 / 50.2 / 920.5 / 20.6 / 0.30 / 0.95 | 2.6 / 11.3 / 959.4 / 26.7 / 0.09 / 0.99 |
| 0.25 | 5.4 / 26.6 / 944.2 / 23.8 / 0.19 / 0.97 | 1.4 / 5.3 / 965.5 / 27.8 / 0.05 / 0.99 |

(Full 0.10–0.25 grids are in HTA L2010-2026 and L2211-2227.)

---

## 5. Recalibration

### 5.1 Method (explicit)
- "a simple method of recalibration which adjusted only the intercept and slope of the model, retaining the relative weighting between coefficients" (HTA L1149). Discrimination was unchanged.
- Model fitted: ln(P_recal/(1−P_recal)) = α_recal + β_recal × LP_eFalls, a logistic regression in the new data with the LP as the only covariate (HTA L1151-1152). "This model was fit using the glm command in R (version 4.2.3)" (L1153).
- AA: same (AAS L121-128; AASM L306-309; AA L134). AA frames it as intercept updating plus "adjustment of all regression coefficients by the same adjustment factor".
- HTA L1153 says "The coefficients for the recalibrated model are shown in Appendix 4, Table 35". AA instead shows the application in Box S3.2.

### 5.2 Values reported

| Source | Content |
|---|---|
| HTA Table 34 (L2070-2080; formula only in NCBI https://www.ncbi.nlm.nih.gov/books/NBK623977/table/table34/) | p = exp(−0.423 + 1.25×LP)/(1 + exp(−0.423 + 1.25×LP)). Same LP as Table 16 (−6.258 ..., −0.304 if male). Example gives 0.106 ("10.6% risk"). **But its pointer says "(see Table 35 for values)".** |
| AA Box S3.2 (AAS L365-372; AASM L1270-1297) | Identical: expit(−0.423 + 1.25*LP); example expit(−0.423 + 1.25*(−1.368)) = 0.106 |
| HTA Table 35 (L2081-2172; CSV `raw/hta_table35_recalibrated_coefficients.csv` identical, 0 differences) | Column header "Final penalised model, coefficient". Constant −7.25089539; Age 0.050276226; ln((P+1)/10) 0.398851695; Female −0.36748668; ... |

### 5.3 Reconciling Table 35 (`compare_coefs.py`, `worked_examples.py`)
- Every one of the 75 non-reference Table 35 coefficients equals 1.21 × the Table 15 coefficient (ratio min/max 1.2099999999999997 / 1.2100000000000002; least-squares slope 1.21).
- Constant: −7.25089539 − 1.21 × (−5.954459) = **−0.04600000** exactly. So Table 35 = logistic recalibration with **α = −0.046, β = 1.21**, applied to the Table 15 parameterisation.
- Box S3.2 in the same coefficient form would be:
  - Table coding: constant −0.423 + 1.25 × (−5.954459) = −7.8661; female −0.3796; age 0.05194; ln poly 0.41204.
  - Box coding: intercept −0.423 + 1.25 × (−6.258167) = −8.2457; male −0.3796.
- Patient X (female) under Table 35: LP −1.7502, p = **0.148**. Table 34, immediately before it, gives 0.106 (0.101 with full precision).

### 5.4 Which α/β produced the published recalibrated results? (inferred from published numbers)
1. **Figure 40 LP summaries** (before: mean −2.99, SD 0.905, median −3.12, IQR 1.26; after: −4.16, 1.13, −4.31, 1.58):
   - SD ratio 1.2486 and IQR ratio 1.254 imply β ≈ 1.25.
   - α = −4.16 − 1.25 × (−2.99) = −0.4225.
   - With α −0.423 and β 1.25: mean −4.160, median −4.323, SD 1.131, IQR 1.575, matching the figure.
   - With α −0.046 and β 1.21: mean −3.664, median −3.821, SD 1.095, IQR 1.525, not matching.
2. **Test-accuracy tables.**
   - Mapping Table 37's high-risk proportions onto Table 32's exceedance curve gives α = −0.436, β = 1.238 from 5 overlapping thresholds.
   - Predicted numbers flagged high per 1000 at t = 0.10–0.14:
     - Box S3.2: 62.3, 53.0, 45.5, 38.9, 33.2.
     - Reported: 62.7, 53.3, 45.6, 39.2, 33.1.
     - Table 35: 111.7, 97.8, 86.0, 76.3, 67.3.
   - (`recal_from_test_accuracy.py`)
3. **Calibration-in-the-large.**
   - Skew-normal LP matched to Fig 40 (mean −2.99, SD 0.905, median −3.12).
   - Mean expit(LP) = 0.0681 vs O/E-implied 0.0677.
   - Mean expit(−0.931 + LP) = 0.0298 vs prevalence 0.0292.
   - Box S3.2 mean predicted = 0.0301 (O/E ≈ 0.97, consistent with the reported 1.000).
   - Table 35 mean predicted = 0.0446 (O/E ≈ 0.66), inconsistent with the reported 1.000.
   - (`lp_distribution_checks.py`)
4. The EV overall calibration slope, 1.248 (Table 18), is the same quantity as β in an intercept-plus-slope glm. So β = 1.25 is internally consistent.
5. Conclusion (inferred, high confidence): the recalibrated performance in Tables 36/37 and Figure 40 corresponds to **α = −0.423, β = 1.25** applied to the LP the Leeds team computed. Table 35 (α −0.046, β 1.21) does not reproduce any reported recalibrated result.
   - Its provenance is **not_found**. Hypotheses, none confirmed:
     - an earlier analysis iteration (HTA L1723 mentions a Connected Bradford data-model rebuild);
     - a different cohort, e.g. the 88,947 cohort with 3.5% prevalence, which would push α upward;
     - a transcription error.
   - Using Table 35 coefficients inside Table 34's formula would apply recalibration twice. Table 34's "(see Table 35 for values)" invites exactly this.

### 5.5 What is presented as final
- The HTA recommends the externally validated original model: "For implementation, we would recommend use of the externally validated eFalls model, although our findings suggest targeted recalibration to local or regional populations could also be beneficial" (L1181; also AA L307).
- The recalibrated model was assessed "only as apparent validation, thus its use is not currently recommended in practice without further validation" (L1180; AA L305).
- HTA-only statement: "This adjusted model has been subsequently tested within Greater Manchester by members of their Integrated Care Board (ICB) and has been found to generalise well to their data" (L1156). Which α/β the Greater Manchester model used is **not_found**. The ARC-GM pilot report does not mention recalibration (grep of `raw/arcgm_efalls_report_clean.txt`).
- Why AA's alpha/beta and Table 35 might differ: see 5.4 hypotheses. Both are presented in the monograph as the recalibrated model without acknowledging any difference.

---

## 6. Additional predictors (CARE75+) and thresholds

- **Methods (L550-552):** CARE75+ "includes linked primary care data for 267 participants". Outcome: falls or fragility fractures recorded in primary care within 12 months of each assessment (baseline, 6, 12, 24, 48 months). Logistic regression with eFalls LP as an offset and a random intercept per participant. C-statistic and AIC compared for null vs grip strength, gait speed, and both.
- **Results (L1161-1173; NCBI Table 19):**
  - n = 252 participants with linkage plus grip and gait data; 864 measurements across five time points; 18 falls/fragility fractures within 1 year.
  - Grip strength median 19 (LQ–UQ 12–26) across all time points; 20 (13–28) at baseline.
  - Gait speed median 5.0 (4.0–6.7) all time points; 4.5 (3.8–6.2) at baseline. Units are **not_found** in text.
  - Table 19: Null C 0.621 (0.482–0.760), AIC 174.9; Grip strength 0.653 (0.518–0.788), 175.3; Gait speed 0.627 (0.489–0.766), 176.1; Grip + gait 0.655 (0.520–0.790), 177.0.
  - "the AIC was lowest for Null model" (L1163). There is no AA counterpart.
  - Table 8 CARE75+ baseline: median age 85 (82–88); 3 fall/fracture (1.2%) (L571, L676).
- **Thresholds:** 10–25% specified a priori (HTA Table 7 L532; AA L120). Test accuracy at 0.10–0.25 in Tables 32/37 (section 4).
- **Development decision curve:** net benefit 0.008 down to 0 across 10–25% (L1103). Utility over treat-all/treat-none up to 28% (L1178).

---

## 7. Sex term (issue i): what the monograph says and what the numbers imply

### 7.1 Explicit statements
- Table 15 (L957-959): Gender, Male Reference, Female −0.303708 (OR 0.732). Constant −5.954459 (L1038).
- Table 16 (L1051) and Table 34 (L2075): "−6.258 ... – 0.304 (if male)". The female example applies no sex term (L1053, L2077).
- HTA L947: "being female" is listed among factors that "contributed to a higher predicted fall/fracture risk". This sentence is absent from AA.
- No erratum, correction note or other clarifying statement was found in the monograph (grep for erratum/correction/sex/male).
- Sci Rep 2026 supplement Table S1 reprints the Table coding (Male Reference, Female −0.303708, Constant −5.954459) (`raw/scirep_efall_supp.txt` L9).

### 7.2 Numerical consistency checks (inferred; `mean_lp_sex_coding.py`)
Four codings compared:
- A = Table as printed.
- B = labels swapped (female reference, male −0.304, constant −5.954).
- C = Box literal (female −6.258, male −6.562).
- D = everyone 0.304 lower than A (e.g. sex entered as 1 = male / 2 = female treated as continuous).

**Development check (SAIL).** Published LP mean −3.30 (Fig 22; Table S2.2). The same LP summary gives a mean predicted risk of 0.0486, equal to the observed 32,097/660,417 = 0.0486, so −3.30 is the calibrated apparent LP.
- Expected mean LP was computed from published means: age 74.9 (L556), deficit prevalences (S3.1), BMI/smoking/alcohol (Table 1), male 47.2%.
- E[ln((P+1)/10)] was bounded from the quartiles 0/4/9:

| Coding | Typical E ≈ −0.83 to −1.03 | Extreme lower bound E = −1.325 (≤50% zero) |
|---|---|---|
| A | −3.01 to −3.07 | −3.17 |
| B | −2.99 to −3.06 | −3.15 |
| C | −3.29 to −3.36 | −3.46 |
| D | −3.31 to −3.38 | −3.47 |

→ A/B cannot reach −3.30 even under the extreme bound; C/D match.

**EV check (Bradford).** Published LP mean −2.99 (Fig 40). Using quartiles 1/4/7, mean age 74.5–76.5 and S3.1/Table 1 EV prevalences:
- A: −2.63 to −2.85.
- C: −2.91 to −3.13.
- Only under the combined extreme (mean age 72.25 and lowest E) does A reach −2.94.
- → C/D fit better. Weaker than the SAIL check.

**Sex-contrast check.** EV sex-subgroup calibration is similar (O/E female 0.417 vs male 0.463; CITL −0.986 vs −0.832).
- If the Leeds LP followed the Box (C), which the mean LP suggests, and the true model had the Table's sex contrast (A or D), males would be under-predicted relative to females by about 0.6 logit. The male/female O/E ratio would then be about 1.8 rather than 1.11.
- This weakly favours C over A/D as the true sex contrast.
- It assumes sex effects transport, so it is suggestive only.

**Plausibility.**
- Crude SAIL risk: females 5.93% vs males 3.66%.
- An adjusted female log-OR of +0.304 (C) is a smaller change from the crude log-OR (+0.50) than a reversal to −0.304 (A). Qualitative only.

**Summary (inferred, moderate confidence).**
- Published LP summaries in both datasets imply an effective intercept about 0.29–0.30 lower than the Table 15 constant/sex coding. This agrees with Box S3.1 (C).
- The female-positive reading is supported by the Box, the example and the HTA L947 sentence.
- A remaining alternative is D (Table direction, but constant effectively −6.258 for males). It fits the mean-LP checks but contradicts the Box example and L947.
- This cannot be settled from published material; confirm with the authors.
- For a reproduction, report sensitivity under A and C at least.

---

## 8. Chapter 9 Discussion items relevant to eFalls (L1714-1756)
- Addition of gait speed/grip strength "did not appear to have a notable impact on prognostic performance" (L1717). This is inconsistent in tone with Ch 5 L1182, "small improvements ... with the addition of grip strength data".
- Deaths kept in the risk set for the full 12 months (no competing-risk model). Mortality 3.8% development, 4.3% EV (L1725).
- Polypharmacy is a single count with FP modelling; fall-risk-increasing drugs were not modelled individually (L1726).
- EV discrimination compared with a 2022 antihypertensive-population model (pooled C 0.866) (L1739).
- Greater Manchester ICB deployment:
  - eFalls was deployed into the Greater Manchester Care Record (2.8M residents).
  - Predictions are produced by the ICB Business Intelligence Unit.
  - Primary care networks contact at-risk patients for FaME-based group falls prevention (L1743).
  - Whether the original or recalibrated equation is used there is not stated.
- Falls model described as "the first such prognostic prediction model to be externally validated in an independent data set and to examine the potential for local recalibration" (L1747).
- EV was restricted to UK databases; international EV is needed (L1724, L1756).
- Data-sharing: "The prediction model equations as published in this manuscript are available for research use. Code lists ... available on reasonable request" (L1786).

---

## 9. Unresolved items and suggested actions
1. Sex coding (A vs C vs D). Ask the corresponding author (Lucinda Archer / Andrew Clegg) for the Stata variable coding and the exact LP used by Leeds. Meanwhile implement both A and C with sensitivity analysis.
2. Table 35 provenance (α −0.046/β 1.21) vs Box S3.2/Table 34 (α −0.423/β 1.25). Ask the authors. Treat Table 35 as unverified; the published recalibrated performance matches −0.423/1.25.
3. BMI "overweight" and obese cut-offs (≥40 vs "standard"). Ask the authors; a Meuhedet reproduction should pre-specify a sensitivity analysis (obese ≥30 vs ≥40).
4. 74 vs 75 predictors, 25 vs 50 bootstraps, index date and 88,947 vs 81,685 Connected Bradford cohorts: internal HTA inconsistencies without explanation.
5. Deaths in the 81,685 falls-EV cohort: not reported.
6. Units of gait speed and grip strength in CARE75+: not reported in the falls chapter.
