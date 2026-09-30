# eFalls: numerical and data-consistency audit (claims C1 to C8)

Auditor role: independent, adversarial, numbers first. I recomputed everything in Python 3.13 (numpy/scipy). The scripts are in `research/audit_numeric/` (listed at the end). I did not trust earlier audit scripts; where one had a useful idea I re-derived it myself.

Path abbreviations used below:
- FT = `efalls/fulltext.md` (Age & Ageing main text)
- SUPP = `efalls/supp.md` (A&A supplement)
- MATH = `research/supp_fulltext_with_math.txt`
- HTA = `research/raw/gjac1008_struct.txt` (NIHR HTA monograph GJAC1008)
- INSTR = `research/raw/efi2plus/instr_v1.13.txt` ("Intended use and implementation instructions for the eFI2 and eFI+", docx creator Kate Best, created 2026-02-26; v1.9 and v1.10 have identical eFalls text; the source URL was not recorded by the earlier round)
- EFI2 = `research/raw/efi/efi2.txt` (Best et al. 2025 eFI2, Age Ageing)

Labels: **[EXPLICIT]** the value or text is printed in the source. **[COMPUTED]** my arithmetic on explicit numbers. **[INFERRED]** my argument, which depends on stated assumptions. **[NOT FOUND]**.

---

## C1: sex term (Table S3.2 / HTA Table 15 versus Box S3.1)

### What the sources print
- Table S3.2: `Male | Reference`, `Female | -0.303708 | 0.732 (0.712 to 0.753)`, `Constant | -5.954459 | 0.003 (0.002 to 0.003)` (SUPP 227-229, 313). HTA Table 15 is identical (HTA 957-959, 1038). **[EXPLICIT]**
- Box S3.1: `LP = -6.258 + 0.042 x (Age) + 0.330 x (ln((Polypharmacy+1)/10)) – 0.304 (if male) + ...` (SUPP 318; MATH 1143). HTA Table 16 is the same (HTA 1051). **[EXPLICIT]**
- HTA text (HTA 947): "Increased age, polypharmacy, being female, being normal or underweight, current smoking and higher or harmful alcohol consumption ... all contributed to a higher predicted fall/fracture risk." **[EXPLICIT]** This agrees with the Box and contradicts the Table's sign.
- Same monograph, eCareHome model: Table 21 prints `Male Reference / Female −0.1205023` (HTA 1245-1247), and the text says "Being male ... contributed to a higher predicted risk of care home admission" (HTA 1234). For that model, table and text agree. **[EXPLICIT]**
- INSTR (developer implementation document) Table 4: `Intercept term -5.954459`, `Sex – Male` blank, `Sex - Female -0.303708` (INSTR 368, 398-401). Worked example for a female aged 70: `L = -5.954459 + 70 * 0.0415506 + 0.3296295 * ln((5+1)/10) -0.303708 + 0.4896735+ 0.0498699`, `L = -3.023169`, `P = 0.0464` (INSTR 358-362). **[EXPLICIT]** My recomputation of that line gives L = -2.978465 and P = 0.0484, so the printed L is wrong by -0.0447. No single coefficient explains the difference. **[COMPUTED]**
- Sci Rep 2026 supplementary Table S1 uses Male Reference / Female -0.303708 / constant -5.954459 (`raw/scirep_efall_supp.txt`). **[EXPLICIT]**

### Arithmetic of the conflict [COMPUTED]
- Female intercept: Table -5.954459 - 0.303708 = -6.258167; Box -6.258. These agree.
- Male intercept: Table -5.954459; Box -6.258167 - 0.303708 = -6.561875. The difference is 0.607416.
- Box intercept -6.258 equals K + b (K = -5.954459, b = -0.303708). The Box looks like a re-expression of the Stata output, not an independent value.

### Test 1: mean linear predictor, which is label-independent [COMPUTED + INFERRED]
The sources print LP summaries:
- Development data, Stata (HTA Fig 22, `raw/hta_fig22.png`): mean -3.30, median -3.37, SD 0.83, LQ-UQ -3.92 to -2.77. Table S2.2 also gives mean -3.30 and variance 0.690 (SUPP 78). **[EXPLICIT]**
- Bradford EV, R (HTA Fig 40 / Fig S3.11, `raw/hta_fig40.png`): mean -2.99, SD 0.905, median -3.12, IQR 1.26. After recalibration: mean -4.16, SD 1.13, median -4.31, IQR 1.58. **[EXPLICIT]**

Check that these LP means match the published calibration. I fitted a skew-normal to the mean, SD and median.
- SAIL: mean expit(LP) = 0.0486, against an observed risk of 32,097/660,417 = 0.0486. Apparent O/E is 1.0000 (SUPP 341). This is expected: the LASSO intercept is unpenalised, so mean predicted equals observed.
- Bradford: mean expit(LP) = 0.0681, against the 0.0677 implied by O/E 0.432. CITL implied by the distribution is -0.953, against the printed -0.931.
- So the LP means are tied to real, published predictions.

The mean LP is linear in covariate means: mean LP = intercept+sex + Σ β·mean(x). I took marginal means from Table 1 and Table S3.1 (exact counts; "<10" cells set to 5, which has no material effect). Contributions from BMI, smoking, alcohol and the 62 binaries:
- SAIL: +0.2700
- Bradford: +0.5737

Mean age:
- SAIL: 74.9 (SD 7.5), same cohort (HTA 560; EFI2 120-121, where N = 660,417 and 311,742 male are identical). **[EXPLICIT]**
- Bradford: median 74 [69-81] (FT 154). The eFI2 Bradford cohort (78,760 = 81,685 - 2,925 exclusions) has mean 75.1 (EFI2 120). I used 75.1 to 75.8.

E[ln((P+1)/10)], from the quartiles:
- SAIL, quartiles 0/4/9. Sharp distribution-free bounds: [-1.324, -0.563]. Zero-inflated negative binomial fits matching the quartiles: [-1.101, -0.821].
- Bradford, quartiles 1/4/7. Sharp bounds: [-1.207, -0.446]. ZINB fits: [-0.978, -0.805].

Mean of intercept+sex under each parameterisation:

| | SAIL | Bradford |
|---|---|---|
| A: Table S3.2 as printed (male ref, const -5.954) | -6.1148 | -6.1194 |
| B: dummy labels swapped (female ref, const -5.954) | -6.0978 | -6.0932 |
| D: Box S3.1 literal (-6.258 female, -6.562 male), equivalently sex entered as a numeric 1/2 code | -6.4015 | -6.3969 |

Published mean LP minus predicted mean LP:
- **SAIL (E[age] 74.9):**
  - A: -0.20 to -0.30 with ZINB-range polypharmacy; -0.13 even at the extreme-low bound
  - D: -0.01 to +0.08
- **Bradford (E[age] 75.1-75.8):**
  - A: -0.24 to -0.33 (ZINB range); -0.17 to -0.20 at the extreme bound
  - D: -0.05 to +0.035

For A to fit, E[ln((P+1)/10)] would need to be about -1.72 (SAIL) or -1.74 (Bradford). That is below the lowest value any distribution with the printed quartiles can reach (-1.32 and -1.21). **So in both the Stata development LP and the R validation LP, the average intercept+sex term behaves like the Box (D), not like Table S3.2 as printed (A or B).**

Independent check with the unpenalised constant. The unpenalised model is also calibrated in-sample, so assume its mean LP ≈ -3.30 and use the ln(OR) coefficients.
- Under A-type coding, the constant would have to be -6.12 to -6.28, i.e. OR 0.0019-0.0022, which would print as 0.002.
- Under D-type coding it would be -5.82 to -5.99, i.e. OR 0.0025-0.0030. The printed value is 0.003 (SUPP 313), which needs OR ≥ 0.0025.
- This agrees with D. **[COMPUTED/INFERRED]** (`c1_unpen.py`)

Most parsimonious mechanism **[INFERRED]**: sex was entered into Stata as a numeric 1/2 code with β = -0.303708 and _cons = -5.954459. The table writer labelled it as a Male-reference dummy. That makes one sex's intercept K+β = -6.258 and the other's K+2β = -6.562, which is exactly the Box's structure. Mean LP cannot tell 1=F/2=M (Box literal, "D1") from 1=M/2=F ("D2", where the Box's sex labels would be swapped). The two differ by 0.017 on the mean.

### Test 2: direction of the sex effect [COMPUTED + INFERRED]
The LASSO KKT conditions with λ = 0.000123 (HTA 947) give within-sex calibration in SAIL to about 0.2%. So mean predicted F/M = observed F/M.
- SAIL observed risk: F 20,674/348,675 = 5.93%, M 11,423/311,742 = 3.66%. RR 1.618; crude log OR 0.505.
- Bradford model-predicted means from Table S3.4 O/E: F 1517/0.417/44,366 = 8.20%, M 872/0.463/37,319 = 5.05%. Ratio 1.625.
- Consistency check: sex-specific expected events add up to O/E 0.4327, matching Table 2's 0.432.

Solving for the non-sex covariate LP gap (female minus male) needed, using within-sex normal LP:

| Scenario | SAIL gap needed | Bradford gap needed |
|---|---|---|
| Females +0.304 (Box direction, D1) | 0.218 | 0.243 |
| Females -0.304 (Table direction, A/D2) | 0.826 | 0.851 |

A gap of 0.83 is about the total LP SD (0.83). It would take women being about 20 years older, or implausibly large condition imbalances.

Evidence that the gap is about 0.2: women have about 0.02 higher mean eFI (≈0.7 extra deficits of 36; `efi/holl2019.txt` 84-85; `efi/efi2016.txt` 73-74) **[EXPLICIT]**. Adding about 1-2 years of age, osteoporosis/fracture imbalance, polypharmacy and BMI gives about 0.2 (assumed sex-specific prevalences, **[INFERRED]**).

External analogue from the same first author: STRATIFY-Falls (UK CPRD), adjusted SHR for women 1.25 (1.23 to 1.27) (`raw/PMC9641577.xml`, line 47 text). **[EXPLICIT]** It supports a positive adjusted female effect.

Bradford counterfactuals (`c1_bradford_oe.py`), assuming the published Table S3.4 values came from applying D1 (the Box, as FT 278 says):

| Model applied instead | Female O/E | Male O/E | M/F | Overall |
|---|---|---|---|---|
| D1 Box (published) | 0.417 | 0.463 | 1.11 | 0.433 |
| A Table S3.2 | 0.417 | 0.271 | 0.65 | 0.348 |
| D2 (numeric, labels swapped) | 0.544 | 0.353 | 0.65 | 0.454 |
| B (swapped dummy) | 0.322 | 0.353 | 1.09 | 0.333 |

If the published values instead came from D2, then D1 gives F 0.322, M 0.612, M/F 1.90.

Published near-parity (0.417 vs 0.463) together with overall O/E 0.432 therefore shows two things. First, the applied model's relative sex effect matched Bradford's data. Second, it had a D-type intercept. With the gap-plausibility argument, that favours D1.

### Evidence tally
- **For Box S3.1 (D1):** mean-LP tests in two independent pipelines; the unpenalised constant; the HTA text "being female"; the covariate-gap plausibility; the external SHR 1.25; FT 278 saying the EV applied Box S3.1; the Bradford sex parity under the applied model.
- **For Table S3.2 (A):** table labels (A&A and HTA; not independent of each other); INSTR Table 4 and example (developer document, but it has its own arithmetic error); Sci Rep 2026 and HTA Table 35, which copy the table and are not independent.
- The GM pilot flagged group was 55% male (`raw/arcgm_efalls_report_clean.txt`, "Of the 1150 people identified by eFalls ... 55% were male and 45% female"). This is consistent with an A-style implementation in GM, but it says nothing about which is correct.

### Verdict and recommendation
**Partially confirmed.** The claim's arithmetic is right: females agree, males differ by 0.607416. It is also right that the evidence conflicts. But the claim understates things in two ways:
- There is quantitative, label-independent evidence (mean LP in both SAIL and Bradford, and the unpenalised constant) that Table S3.2 **as printed** did not generate the published predictions.
- The pro-Table arguments listed in the claim (penalised/unpenalised sign agreement, HTA Table 15, HTA Table 35) are not independent evidence. All of them are consistent with a single labelling or coding error.

Confidence:
- High (≈90%) that Table S3.2's "Male reference, constant -5.954459" is not the parameterisation that produced the published LPs.
- Moderate (≈70-80%) that Box S3.1 literal (female -6.258167, male -6.561875) is correct rather than D2 (male -6.258167, female -6.561875).

**Implementation recommendation:**
- Use Box S3.1 as the primary parameterisation: intercept -6.258167 for females and -6.561875 for males. This is LP = -5.954459 - 0.303708·(1 + male), with all other coefficients at full Table S3.2 precision.
- Carry Table S3.2/INSTR (male intercept -5.954459) as a pre-registered sensitivity model.
- Flag the issue in the protocol and ask the authors for the Stata do-file (INSTR contact: a.p.clegg@leeds.ac.uk).
- In Meuhedet data, test empirically: fit logit(y) = a + b·LP_Box + c·male. c ≈ 0 supports the Box; c ≈ +0.61 supports Table S3.2. Report sex-specific O/E before recalibration.

---

## C2: BMI categories
- **Box S3.1:** "underweight if BMI < 18.5; normal weight if 18.5 ≤ BMI < 24.9; obese if BMI ≥ 40" (SUPP 318; MATH 1154). HTA Table 34 repeats it (HTA 2076). **[EXPLICIT]**
- **HTA Table 16:** "weight groups are defined by standard BMI cut-offs" (HTA 1052). **[EXPLICIT]**
- **INSTR (line ~108):** underweight <18.5, recommended 18.5-24.9, overweight "<25.0-29.9", obese ≥30; also "Body mass index less than 20" SNOMED code as an underweight example; most recent measurement. **[EXPLICIT]**
- **Baseline2 code list rule:** "BMI 30+ = obesity, BMI <18.5 Underweight" (`research/efi_codelists.md` line 173). **[EXPLICIT]**
- **Table 1 shares [COMPUTED]:**
  - SAIL obese 136,646/429,910 non-missing = 31.78%; overweight 36.91%; normal 28.37%; underweight 2.94%.
  - Bradford obese 21,698/77,933 = 27.84%; HTA Table 8 (88,947 extract) 27.68%.
  - These fit WHO ≥30 in an older UK population. BMI ≥40 is typically a few percent (general knowledge, **[INFERRED]**).
- **The 24.9 bound:** "< 24.9" leaves [24.9, 25) unassigned; the WHO table writes 18.5–24.9 meaning <25.
- **Verdict:** confirmed.
- **Recommendation:** use <18.5; 18.5–<25; 25–<30; ≥30; missing if no BMI (most recent value; the eFalls lookback window is **[NOT FOUND]**; Baseline2 uses 5 years). Document the Box misprint. No ≥40 sensitivity analysis is needed.

---

## C3: parameter count
- **Non-reference rows in Table S3.2 [COMPUTED]:** age 1, polypharmacy 1, sex 1, BMI 4, smoking 1, alcohol 5, binary 62. Total **75**, matching "LASSO regression retained 75 predictors" (FT 191).
- **Table S3.1:** 72 rows, 10 asterisked (Anxiety, CKD, Dyspnoea, Environment problems, Heart valve disease, IHD, Problems managing finances, Shopping problems, Toileting problems, TIA). 72 - 10 = 62, and the name sets match Table S3.2 one-to-one.
- **Candidates:** FT 92 says 36 eFI components plus 44 additional = 80. Table S3.1's 72 plus 6 non-binary = 78. The gap of 2 is unexplained **[NOT FOUND]**.
- **Extra discrepancy:** the HTA says "The LASSO regression retained 74 predictors" (HTA 947).
- **Context:** the eFI2 paper has a similar internal count problem ("79 candidate predictor variables ... 70 were binary and four had multiple ordered categories", EFI2 82).
- **Verdict:** confirmed (plus the HTA 74 vs 75 discrepancy).
- **Recommendation:** implement exactly the 75 printed coefficients; treat the candidate count as a reporting inconsistency with no effect on the equation.

---

## C4: worked example [COMPUTED]
- **Rounded Box coefficients:** -6.258 + 0.042·89 + 0.330·ln(0.9) + 0.490 + 0.085 + 0.104 + 0.380 + 0.128 = -1.36777. p = 0.20298 (printed 0.203 ✓).
- **Full precision (female, Table S3.2 = Box for females):** LP = -1.408453, p = 0.19648. Most of the difference is the age rounding: (0.042 - 0.0415506)·89 = 0.0400.
- **Recalibrated:**
  - Rounded: expit(-0.423 + 1.25·(-1.368)) = 0.10593 (printed 0.106 ✓).
  - Full LP: 0.10124.
  - Under HTA Table 35 coefficients: 0.1480 (or 0.1543 with rounded LP).
- **Male twin of Patient X:**
  - Table S3.2: LP -1.1047, p 0.2489
  - Box: LP -1.7122, p 0.1529
- **Verdict:** confirmed.
- **Recommendation:** unit tests should include both paths:
  - rounded-coefficient path reproduces 0.203 and 0.106;
  - full-precision path gives 0.1965 and 0.1012;
  - add a male test case under both sex parameterisations.

---

## C5: smoking
- **SAIL:** Never + Ex + Current = 302,363 + 271,248 + 86,806 = 660,417, exactly. Fall and no-fall columns are also exact. **[COMPUTED]**
- **Bradford:**
  - Outcome columns are exact: 1,717 + 1 + 671 = 2,389; 57,562 + 15 + 21,719 = 79,296.
  - The **Total column is not**: 59,295 + 16 + 22,390 = 81,701 (+16). The Never total 59,295 should be 59,279; 59,295 is the Ex+Never count. **[COMPUTED]** (new detail)
- **No missing row:** INSTR line 110 makes this explicit: "Smoking status should be categorised as Ex, Current, None/missing ... coded as none (smoking)/ missing if they have no smoking SNOMED CT codes recorded." **[EXPLICIT]** Smoking is therefore *not* a missing-indicator variable, contrary to the general statement at FT 110.
- **Bradford ex-smokers:** 16 (0.02%) vs 41.1% in SAIL (HTA Table 8 88,947 extract: 18). This is a coding failure, but it has no LP impact because Ex/never is the reference.
- **Verdict:** confirmed, with the Bradford total-column caveat.
- **Recommendation:** code current (latest smoking code = current) vs not current, with missing = not current.

---

## C6: age top-coding
- **Figure S3.1 / HTA Fig 20** (`raw/hta_fig20.png`): age points run from 65 to 95 only, on an axis to 100, labelled "Age (years, from week of birth)". **[EXPLICIT, visual]**
- A cohort of 660k aged 65+ would normally contain people over 95 **[INFERRED]**. Top-coding at 95, exclusion, or plot truncation are all possible; the source is **[NOT FOUND]**.
- The polypharmacy panel shows values up to about 61, so that axis is not truncated.
- **Verdict:** partially confirmed (weak).
- **Recommendation:** apply the linear age term without a cap in the primary analysis. Run a sensitivity analysis capping age at 95. The impact is at most 0.0415 per year above 95.

---

## C7: reporting inconsistencies
- **(a)** Box S3.1 "[see table S3.1 for values]" (SUPP 318, 368) should say S3.2. HTA Table 16 correctly says "see Table 15" (HTA 1051). **Confirmed.**
- **(b)** Two tables are titled "Table S3.4" (SUPP 393 subgroups; SUPP 451 TP/FP). FT 280 and FT 289 cite them ambiguously. **Confirmed.**
- **(c)** In Table S3.4 (subgroups), 6 of 68 CIs exclude their point estimate. **Confirmed.**

  | Row | Statistic | Point | CI |
  |---|---|---|---|
  | IMD1 (SUPP 398) | slope | 1.123 | 1.126–1.158 |
  | IMD1 (SUPP 398) | C | 0.788 | 0.789–0.796 |
  | Mild frailty (407) | slope | 1.300 | 1.234–1.298 |
  | Mild frailty (407) | C | 0.735 | 0.722–0.733 |
  | Male (413) | slope | 1.368 | 1.371–1.393 |
  | Male (413) | C | 0.834 | 0.835–0.839 |

  HTA Table 33 is identical.
- **(d)** Table 2 (FT 252) and HTA Table 36 (HTA 2195) give the recalibrated pooled C CI as "0.801 to 1.000". Recalibration cannot change C, and the PI and τ² are identical to before recalibration, so it should be 0.830. **Confirmed.**
- **(e)** Table 1 EV IMD3 (FT 159): total 13,337 (16.3%), but 473 + 13,864 = 14,337, which is also the S3.4 n. The IMD column then sums to 81,685 (with 13,337 it is 80,685). Correct: 14,337 (17.6%). **Confirmed.**
- **(f)** Table 2 apparent pooled C 0.72 vs S3.3 overall 0.7434. This is not an arithmetic inconsistency: a within-practice pooled C is expected to be below the overall C. The printed CI 0.72–0.72 and PI 0.68–0.76 check out with k = 455 and τ² 0.010 on the logit scale (implied PI 0.679–0.758). **Refuted as an error.**
- **(g)** Penalised vs unpenalised coefficients (`c7g.py`):
  - All 75 signs agree.
  - 5 rows have |penalised| > |ln OR| by ≤0.0012 (OR rounding).
  - Observed shrinkage tracks λ/(sd_x·w) with λ = 0.000123. Examples: previous higher-risk drinking predicted 0.130 vs observed 0.119; higher-risk 0.079 vs 0.072; zero alcohol 0.054 vs 0.047; motor neurone disease 0.120 vs 0.096.
  - So 0.085 vs OR 1.226 is plausible LASSO shrinkage on a 90-person cell. **Refuted as an error.** This also shows Table S3.2 coefficients are internally coherent, but that is label-independent and silent on C1.
- **(h)** Additional findings:
  1. Table S3.1 EV Housebound: fall count 931 should be 831. The total is 8,931 = 831 + 8,100, and 34.8% × 2,389 = 831 (SUPP 167). **[COMPUTED]**
  2. Table S3.3 / HTA Table 31: CITL CI "-0.0115 to 0.115" should end at 0.0115 (SUPP 338; HTA 1978).
  3. Table 2 EV "overall" CIs are about 4.3–5.3× narrower than analytic SEs:

     | Statistic | Printed CI | Analytic CI |
     |---|---|---|
     | O/E | 0.430–0.437 | 0.415–0.449 |
     | C | 0.824–0.828 | 0.815–0.835 |
     | CITL | half-width 0.009 | half-width 0.041 |
     | Slope | half-width 0.0105 | half-width 0.045 |

     The ratio ≈ 1/√25 suggests the SE of a 25-bootstrap mean was used instead of the bootstrap SD **[INFERRED]**. The same narrowness applies to Table S3.4. Development-data S3.3 CIs are plausible (O/E half-width 0.0103 vs analytic 0.0107).
  4. Recalibrated overall slope "1.000 (1.000 to 1.013)": the point estimate sits on the CI bound.
  5. Development pooled O/E PI 0.14–2.24 is symmetric about 1.19 on the natural scale. It matches τ² 0.282 on the natural O/E scale (half-width 1.046), whereas EV and IECV O/E are pooled on the log scale. That contradicts "on appropriate scales [31]".
  6. IECV C: τ² 0.002 is inconsistent with PI 0.59–0.85 on the logit scale, which needs τ² ≈ 0.055. It is consistent with τ² expressed on the natural C scale.
  7. Polypharmacy window: Box S3.1 says 120 days (SUPP 318); INSTR line 112 says "previous 90 days"; eFI2 also uses 90 days (EFI2 190).
  8. INSTR worked example: L -3.023169 printed vs -2.978465 correct (p 0.0464 vs 0.0484).
  9. HTA says 74 retained predictors vs A&A 75 (HTA 947).
  10. HTA Table 15 note calls the unpenalised model a "bootstrapping logistic model" (HTA 1041).
  11. HTA systematically prints "Higher-risk drinking" where A&A has "Lower risk drinking" (reference): Tables 14 (HTA 933: 603/10,628 = A&A "Lower risk drinking"), 15, 21 and 35. This settles open issue (iv) as a typesetting error.
  12. Other HTA models show similar Box/Table conflicts:
      - Home care: Table 10 "Gender = male 0.003787, OR 1.008757836", where exp(0.003787) = 1.0038 but the OR equals exp(0.00872); the Box says "+0.0087 (if male)" (HTA 739, 829).
      - eCareHome smoking "Never −0.038489" vs OR 1.054 (sign disagreement).
  13. Table 1 SAIL alcohol fall column: the suppressed "<10" cell is derivable as 32,097 - (430+35+603+70+30,951) = 8 (90 - 82 = 8), so the suppression is ineffective (minor).

---

## C8: recalibration discrepancy
- **Table 35 vs Table 15 (HTA CSV matches NCBI Bookshelf HTML):** all 75 coefficient ratios are exactly 1.2100000000 (max |Table35 - 1.21·Table15| = 5.6e-17). Implied α = -7.25089539 - 1.21·(-5.954459) = **-0.04600000** exactly. So Table 35 = -0.046 + 1.21·LP_TableS3.2. **[COMPUTED]**
- **Same recalibration under another LP definition?** No:
  - For females, the Box LP and Table LP are identical, so any re-expression must keep female predictions. Patient X gives 0.148 under Table 35 vs 0.106 under Box S3.2.
  - For males the two LPs differ by a sex-specific shift, which no single (α, β) can absorb.
  - Re-expressing Box S3.2 on the Table S3.2 scale would give constant -0.423 + 1.25·(-5.954459) = -7.866 and female coefficient -0.3796, not -7.2509 and -0.3675.
- **Rounding?** 1.21 is exact to 10 dp. It is not a rounding of 1.248 (→1.25) or of the pooled 1.203 (→1.20).
- **Published outputs reproduce Box S3.2, not Table 35:**

  | Recalibrated LP summary (Fig 40) | Printed | Box S3.2 (-0.423 + 1.25·LP) | Table 35 (-0.046 + 1.21·LP) |
  |---|---|---|---|
  | Mean | -4.16 | -4.1605 | -3.664 |
  | SD | 1.13 | 1.131 | 1.095 |
  | Median | -4.31 | -4.323 | -3.821 |
  | IQR | 1.58 | 1.575 | 1.525 |

  - The SD ratio 1.13/0.905 = 1.2486 (rounding range 1.242–1.255) excludes 1.21.
  - Test-accuracy tables (Table S3.4-second / S3.5 = HTA 32/37), quantile matching at t = 0.10:
    - Box S3.2: threshold LP (logit 0.1 + 0.423)/1.25 = -1.419; interpolated flagged 62.5/1000 vs 62.7 printed.
    - Table 35: threshold -1.778; about 112/1000.
    - A fit across overlapping thresholds gives α ≈ -0.436, β ≈ 1.238.
  - Box β 1.25 matches the Table 2 EV overall slope 1.248, as it should. The pooled slope 1.203 and CITL -0.931/-0.874 are different estimands.
  - On the Fig 40 LP distribution, Table 35 gives mean risk 4.46% (O/E 0.66 for the 81,685 extract; 0.78 for the 88,947 extract with 3,079 events, HTA 459/676). The α giving O/E = 1 with β = 1.21 would be -0.53 or -0.34, not -0.046.
- **HTA text:**
  - "The coefficients for the recalibrated model are shown in Appendix 4, Table 35. This model was fit using the glm command in R" (HTA 1153).
  - "This adjusted model has been subsequently tested within Greater Manchester ... found to generalise well" (HTA 1156).
  - Table 34's example still gives 0.106 (HTA 2077).
  - The HTA is internally inconsistent. The origin of α = -0.046, β = 1.21 is **[NOT FOUND]**.
- **Verdict:** confirmed.
- **Recommendation:**
  - Treat Box S3.2 (α = -0.423, β = 1.25 on the Box S3.1 LP) as the published Bradford recalibration, and do not use HTA Table 35.
  - Neither is appropriate for Meuhedet: re-estimate α and β locally, or use the original model with local recalibration, as the authors recommend.

---

## Scripts (all under `research/audit_numeric/`)
- `parse.py` parses Table S3.2, S3.1 and Table 1 into JSON.
- `c3_c4_c8.py` covers the parameter count, the worked examples, the Table 35 ratios and the Fig 40 check.
- `c1_meanlp.py` computes the mean-LP decomposition by scenario.
- `c1_full.py` covers calibration-consistency of the LP summaries, E[lnP] bounds and ZINB fits, offsets, the sex-gap solve, and α for the other extracts.
- `c1_unpen.py` checks the unpenalised constant.
- `c1_bradford_oe.py` computes the Bradford sex-specific O/E counterfactuals.
- `c7g.py` compares penalised vs unpenalised shrinkage.
- `tables.py` checks Table 1 and S3.1 sums, S3.4 CI containment, Table 2 CI widths and PI recomputation.
- `impl_example.py` recomputes the INSTR example.
