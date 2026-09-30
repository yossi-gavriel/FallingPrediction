# eFalls audit: clinical-epidemiology and statistical plausibility (claims C1 to C8)

Auditor lens: clinical epidemiology and statistical plausibility, working adversarially (trying to refute each claim). Compiled 2026-09-14.
All numbers were recomputed in Python 3.13 (numpy/scipy). The scripts are in
`<research workspace (not distributed)>/research/audit_cs2/`:
`c_basic.py`, `c_tables.py`, `c_ci.py`, `c_lp.py`, `c_delta.py`, `c_level.py`, `c_band.py`, `c_recal_A.py`, `c_shrink.py`, `c_table2.py`, `c_carehome_control.py`, `c_carehome_delta.py`, `c_hetero.py`.

Path abbreviations:
- FT = `scratchpad/efalls/fulltext.md` (Age & Ageing main text)
- SUPP = `scratchpad/efalls/supp.md` (supplement)
- HTA = `scratchpad/research/raw/gjac1008_struct.txt` (NIHR HTA monograph)
- IMG = `scratchpad/efalls/supp_media/` (supplement images; the docx relationship order maps image1/2 to Fig S3.1, image10 to Fig S3.8, image13/14 to Fig S3.11)
- HFIG22 = `scratchpad/research/raw/hta_fig22.png` (HTA Figure 22)
- HOLL = `scratchpad/research/raw/efi/holl2019.txt` (Hollinghurst 2019, eFI in SAIL)
- EFI16 = `scratchpad/research/raw/efi/efi2016.txt` (Clegg 2016)
- PROT = `scratchpad/research/raw/protocol_eFIplus_V21_2021-11-03.txt`
- GM = `scratchpad/research/raw/arcgm_efalls_report_clean.txt` (single-line file)
- SCIREP = `scratchpad/research/raw/scirep_efall_supp.txt`

Evidence labels: **explicit_in_source**, **inferred** (my computation or reasoning), **not_found**.

---

## C1. Sex term: which parameterisation?

### Facts (explicit_in_source)
- Table S3.2 lists Male as Reference, Female as −0.303708 (unpenalised OR 0.732, 0.712 to 0.753) and Constant −5.954459 (OR 0.003, 0.002 to 0.003). Source: SUPP 227-229 and 313. HTA Table 15 (HTA 957-959, 1038) is identical, as is the Sci Rep 2026 supplementary Table S1 (SCIREP), which says the coefficients "were applied without modification".
- Box S3.1 gives "LP = -6.258 + ... – 0.304 (if male)" (SUPP 318). The same wording appears in HTA Table 16 (HTA 1051) and HTA Table 34 (HTA 2075).
- The main text says the model "was applied, as shown in Supplementary Box S3.1" in Connected Bradford (FT 278). The HTA equivalent is "applied, as shown in Table 16" (HTA 1108).
- The HTA narrative says: "Increased age, polypharmacy, being female, being normal or underweight, current smoking ... all contributed to a higher predicted fall/fracture risk" (HTA 947). This contradicts Table 15's sign and agrees with the Box.
- Negative control on the narrative. The care-home model comes from the same team, the same data and the same software. Its table has Female −0.1205 with Male as reference (HTA 1246-1247), and its narrative says "Being male ... contributed to a higher predicted risk" (HTA 1234). In that chapter the narrative agrees with the table, so the falls narrative is not simply a mechanical misreading of Table 15.
- HTA Table 35 Female −0.36748668 is exactly 1.21 × −0.303708. This is derived arithmetic, not independent evidence.

### Arithmetic (inferred, c_basic.py)
- The Box intercept minus (Table constant + Female coefficient) is −6.258 − (−6.258167) = +0.000167.
- For females, Box and Table agree to within 0.0002. For males, the Box baseline is −6.562 against the Table's −5.954459. The difference is −0.6075 on the logit scale, an odds ratio of 0.545.
- Four readings are possible:
  - A: Table as printed (F −6.258167, M −5.954459)
  - B: labels swapped, constant kept (F −5.954459, M −6.258167)
  - C: Box read literally (F −6.258, M −6.562)
  - D: constant −6.258 with Female −0.304 (F −6.562, M −6.258)
- All the "Table" evidence (S3.2 penalised and unpenalised columns, HTA T15, T35, Sci Rep S1) comes from one source copied forward. The columns share a single labelling step, so they are not independent confirmations.

### Stata/LASSO mechanism (inferred, from stata_methods.md §2.7)
- Stata `lasso` puts all levels of an othervars factor into the candidate set, and base levels are ignored.
- For a binary factor, the standardised male and female columns are exact negatives of each other. They tie at λmax and only one can be non-zero; in the Stata example the lower level entered.
- The "Reference" row is therefore just the level whose coefficient was zeroed. The constant belongs to that zeroed level.
- Merged references fit this mechanism: smoking "Ex/never" is Never and Ex both zeroed, and the care-home alcohol table has three "Reference" rows.
- So a sex label error is mechanically easy: the table author has to map `1.gender` or `2.gender` onto Male/Female by hand.
- The same mechanism implies 90 candidate parameters: 72 binary + age + polypharmacy + sex (2) + BMI (5) + smoking (3) + alcohol (6) = 90. That is exactly the "anticipated 90 predictor parameters" in FT 98 and SUPP 70-72 (inferred).

### Test 1: sex-specific calibration in development (inferred; c_lp.py, c_delta.py, c_hetero.py)
- With a tiny λ (0.000123, HTA 947) and an unpenalised intercept, the KKT conditions force observed = expected within each sex, to within Nλ·sd ≈ 40 events. Mean predicted risk in SAIL therefore equals the crude risk: F 20,674/348,675 = 5.93% and M 11,423/311,742 = 3.66% (FT 152-153). Crude OR (F vs M) is 1.66.
- LP by sex was modelled as normal or skew-normal, with total variance 0.690 (SUPP 78). The F − M mean-LP gap needed is 0.52 to 0.53.
- The non-sex gap δ required under each sex contrast:
  - Table direction (Female −0.304; A or D): δ = 0.83.
  - Box direction (Male −0.304; B or C): δ = 0.22.
  - Letting female LP variance exceed male by 0.2 barely changes this: 0.74 versus 0.13.
- Clinical estimate of δ = Σ β_j(p_F − p_M) + age + polypharmacy. Inputs were SAIL prevalences (SUPP 137-208) and assumed F:M prevalence ratios. The ratios are my clinical judgement, not taken from the eFalls sources.
  - Base case: δ = 0.23, with a sensitivity range of 0.18 to 0.29.
  - Largest contributors: age gap of 1.6 years (+0.067), polypharmacy (+0.040), fracture, osteoporosis, fragility fracture, falls (each about +0.02), housebound (+0.011).
  - Anchor: the same ratios imply an eFI gap of 0.022. Hollinghurst reports mean eFI in SAIL of 0.12 for males and 0.14 for females (HOLL 84-85), a gap of 0.02 (explicit).
  - Extreme female-adverse bound: every risk-raising predictor 3 times as common in women, every risk-lowering one 3 times as common in men, and a 3-year age gap. This gives δ = 0.80, still below 0.83.
- Negative control: the same method applied to the care-home table. There it is weakly discriminating: required 0.59 (table) versus 0.34 (reversed), estimate 0.21 to 0.26, and with a variance difference of 0.4 the requirements fall to 0.40 versus 0.16. So the method has limited resolution when the sex coefficient is small (0.12). For eFalls the two hypotheses are separated by 0.61 on the logit scale, and the conclusion holds even if δ is underestimated by 0.2.

### Test 2: LP level in development (inferred; c_level.py)
- Explicit summaries of the development LP distribution:
  - Stata "Full group LP details: Mean = −3.30, Median −3.37, SD 0.83, LQ–UQ −3.92 to −2.77" (HFIG22).
  - Mean −3.30 and variance 0.690 (SUPP 78).
- A skew-normal (skew 0.45) with mean −3.30 and variance 0.690 gives E[expit(LP)] = 4.86%, matching the observed 4.86%. This is consistent with apparent O/E 1.0000 (SUPP 341).
- IECV group LP means are −3.11 to −3.33 (IMG image10.png); their weighted mean is −3.20, which is consistent with IECV O/E 0.88 (FT 235).
- Non-sex LP mean from explicit inputs:
  - age 0.0415506 × 74.9 (HTA 556)
  - binary Σβp = 0.3296 (SUPP 137-208)
  - BMI/smoking/alcohol shares (FT 170-185) = −0.0595
  - E[ln((P+1)/10)] from polypharmacy 4 [0 to 9] (FT 155), −1.13 to −0.82 under zero-inflated negative binomial fits
- Implied mean LP by reading:
  - A: −3.11 to −3.00
  - B: −3.09 to −2.99
  - C: −3.39 to −3.29
  - D: −3.41 to −3.31
- Even an extreme low-polypharmacy distribution leaves A at −3.17. Reproducing −3.30 with Table labels needs a constant of about −6.20, not −5.954459.
- The level test therefore favours C or D. Combined with Test 1 (B or C), it points to C, the Box read literally.
- Counter-evidence (weak): the unpenalised Constant OR is printed as "0.003". Under A or B the unpenalised constant would be about −5.97 to −6.04 (ORs round to 0.002 or 0.003). Under C or D it would be about −6.26 or lower (rounds to 0.002). This one-digit rounding mildly favours the printed constant.

### Test 3: Connected Bradford as applied (inferred; c_lp.py, c_level.py, c_recal_A.py)
- Figure S3.11 (IMG image13.png, explicit): as-applied LP mean −2.99, SD 0.905, median −3.12, IQR 1.26.
- Bradford non-sex mean from SUPP S3.1 EV column and FT Table 1 EV shares (age mean assumed 75.3 to 76.3):
  - A: −2.74 to −2.64
  - B: −2.71 to −2.62
  - C: −3.02 to −2.92
  - D: −3.04 to −2.95
  - So the as-applied LP was C or D.
- Sex-specific as-applied mean predicted risk (Table S3.4 O/E, SUPP 412-413, with FT 152-153 counts):
  - F: 1517/0.417/44,366 = 8.20%
  - M: 872/0.463/37,319 = 5.05%
  - LP gap F − M = 0.545
  - If the EV team applied the Box contrast, δ_Brad = 0.24 (plausible). If they applied the Table contrast, δ_Brad = 0.85 (implausible).
  - Fitted sex CITLs reproduce the reported values: F −0.965 versus −0.986, M −0.822 versus −0.832.
- So the EV team applied C, the Box read literally, which matches FT 278. All published EV metrics (Table 2, S3.4, S3.5) and the recalibration (Box S3.2, Fig S3.11) are defined on LP_C.
- Box S3.2's worked example (LP −1.368 → 0.106) also uses the −6.258 female intercept.
- Counterfactual on the as-applied LP:
  - Switching males to the Table term (+0.6075) gives male O/E 0.270 versus female 0.417 (M/F ratio 0.65) and overall O/E 0.348.
  - Switching the other way gives M/F 1.96.
  - The observed ratio is 1.11. Near 1 is what you would expect when the applied sex structure matches the development fit.
- Could Tests 2 and 3 be jointly wrong? Both use the same Σ method. A hidden shared bias of +0.28 in both datasets would turn "C" into "B". That bias would have to equal the Box offset in two populations by coincidence, and no mechanism for it was found. Even in that world, Test 1 still favours the Box sex direction (B).

### Other signals
- Crude risk is higher in females in both cohorts. SAIL: 5.93% versus 3.66%. Bradford: 3.42% versus 2.34%. This is weak evidence on its own, because confounding can reverse a crude effect; Test 1 quantifies how much reversal would be needed.
- Greater Manchester deployment (GM section 3.2.1, explicit): "Of the 1150 people identified by eFalls ... 55% were male and 45% female". This was the 10–25% intermediate-risk list.
  - My crude simulation (c_band.py, inferred, weak) puts the male share of that band at 17–25% for Box-direction implementations and about 47–48% for Table-direction implementations (A or HTA T35).
  - The observed 55% suggests GM used the Table direction. That says what GM implemented, not what is true.
  - HTA 1156 says the "adjusted model" was tested in Greater Manchester.
  - If the analysis above is right, a Table-direction deployment would over-rank men relative to women.
  - Caveats: GM exclusions, including a prior-falls restriction, and GM's LP distribution are unknown.

### Verdict
**partially_confirmed.**
- Confirmed: the discrepancy itself (female agreement to 0.0002; male difference 0.6075), the non-independence of the "Table" evidence, and the "applied as Box S3.1" statement.
- Not accepted as stated: the claim that EV sex O/E values "are inconsistent with Table S3.2". The O/E values alone only show that the applied sex structure matches the development fit. They discriminate only in combination with a δ estimate.
- Balance of evidence: Box S3.1 read literally (F intercept −6.258167, male −0.303708) fits the development LP level, sex-specific calibration, the Bradford as-applied level and sex O/E, the HTA narrative, and Box S3.2.
- Table S3.2 as printed (reading A) is contradicted by Tests 1 and 2.
- Confidence: about 80% that the sex direction is the Box's; about 70–75% that the female-baseline intercept is −6.258 rather than −5.954.
- The mechanism of the error was not found, and no erratum was found (web searches 2026-09-14; crossref probes in raw/crossref_updates.json were empty).

### Recommendation
- Primary specification: implement the Box S3.1 literal LP_C. Use full-precision Table S3.2 coefficients for all non-sex terms, intercept −6.258167 (female baseline), and −0.303708 if male. This is the LP that was externally validated and recalibrated. Box S3.2 (α −0.423, β 1.25) is only coherent with LP_C.
- Add a configuration flag for readings A and B.
- In Meuhedet data, report O/E by sex and fit `logit(Y) ~ offset(LP_C) + male`:
  - a male coefficient near 0 supports C;
  - near +0.6 supports A;
  - a uniform CITL shift of about +0.3 in both sexes is compatible with B (population differences confound this).
- Ask the authors (Archer/Riley/Clegg) for the Stata e(b) vector.

---

## C2. BMI cut-offs

- Explicit: Box S3.1 defines "underweight if BMI < 18.5; normal weight if 18.5 ≤ BMI < 24.9; obese if BMI ≥ 40" (SUPP 318). HTA Table 34 repeats this (HTA 2076). HTA Table 16 instead says "weight groups are defined by standard BMI cut-offs" (HTA 1052), as do HTA 830 and 1498. HTA methods list the categories as underweight, normal, overweight, obese or missing (HTA 549).
- Recomputed (c_tables.py):
  - Among non-missing BMI, SAIL obese = 136,646/429,910 = 31.8%, overweight 36.9%, normal 28.4%.
  - Bradford: obese 27.8%, overweight 38.0%, normal 31.9%.
  - These match WHO ≥30 and 25–29.9 bands in UK older adults. My external knowledge (inferred, not from local sources) puts BMI ≥30 at roughly 25–35% and BMI ≥40 at about 1–3%.
  - If obese meant ≥40, "overweight" would cover 25–40 and would be about 65% or more of recorded values, not 37%.
- The [24.9, 25.0) gap is real as written. A "normal 18.5–24.9" range is conventionally read as <25.0.
- Alternative explanation: "≥40" may be a slip, for example from an "obese class III" code. No plausible reading makes ≥40 consistent with Table 1.
- **Verdict: confirmed.**
- **Recommendation:** underweight <18.5, normal 18.5 to <25, overweight (reference) 25 to <30, obese ≥30, missing = no valid BMI in the lookback window. Document the choice. As a sensitivity analysis, run ≥40 with 25–<40 as overweight; expect a large calibration shift.

---

## C3. Parameter and predictor counts

- Recomputed (c_basic.py, c_tables.py): non-reference, non-constant coefficients in S3.2 = 75. That is 62 binary + age + polypharmacy (the "63" group in the script) + sex 1 + BMI 4 + smoking 1 + alcohol 5.
- This equals "LASSO regression retained 75 predictors" (FT 191).
- Table S3.1 has 72 binary rows, 10 of them asterisked: Anxiety, CKD, Dyspnoea, Environment problems, Heart valve disease, IHD, Problems managing finances, Shopping problems, Toileting problems, TIA. 72 − 10 = 62.
- The HTA says "retained 74 predictors" (HTA 947). This is an inconsistency.
- Candidates: FT 92 and HTA 401/407 give 36 eFI + 44 additional = 80 ("all 80 predictor variables"). Table S3.1 + age + polypharmacy + sex + BMI + smoking + alcohol = 78.
  - Polypharmacy is itself one of the 36 eFI deficits.
  - The source of the 2 missing candidates was not found.
  - WIMD is unlikely to have been a candidate: its missing level is strongly protective (5.9% of events versus 20.4% of non-events, FT 162), yet it is absent from the model.
- Under all-levels coding, candidate parameters = 90, which matches FT 98 (inferred).
- **Verdict: confirmed** (75 and 62 exactly; the 80 versus 78 gap is real; there is an additional HTA 74 versus 75 discrepancy).
- **Recommendation:** implement exactly the 75 non-zero terms. Log the 80/78/74 discrepancies. Do not add the 10 asterisked variables.

---

## C4. Worked example

Recomputed (c_basic.py):

| Version | LP | Probability |
|---|---|---|
| Rounded Box S3.1 | −1.3678 | 0.2030 |
| Full precision, reading A = C for a female | −1.408453 | 0.1965 |
| Label-swap reading B (female at −5.954459) | −1.1047 | 0.2489 |
| Box S3.2 recalibrated, rounded LP: expit(−0.423 + 1.25 × −1.368) | | 0.1060 |
| Box S3.2 recalibrated, full-precision LP | | 0.1012 |
| HTA Table 35 applied to Patient X | −1.7502 | 0.1480 |

- The main source of the rounded versus full-precision gap is age rounding: 0.042 × 89 − 0.0415506 × 89 = 0.040.
- HTA Table 35 gives 0.1480, which is inconsistent with HTA Table 34's 0.106 for the same patient.
- **Verdict: confirmed.**
- **Recommendation:** use full precision throughout. Unit-test that a female LP of −1.408453 gives p = 0.1965 under LP_C, and that the Box rounded arithmetic gives 0.203 and 0.106.

---

## C5. Smoking

- Explicit (FT 176-178): SAIL Never + Ex + Current = 302,363 + 271,248 + 86,806 = 660,417, and events and non-events also sum exactly. There is no missing category.
- Bradford events (1,717 + 1 + 671 = 2,389) and non-events (57,562 + 15 + 21,719 = 79,296) sum exactly. The printed Bradford "Never" total of 59,295 should be 59,279 (off by 16, the number of ex-smokers), so the total column sums to 81,701.
- Bradford has 16 ex-smokers (0.02%) and 27.4% current smokers among people aged 65 and over. For a "latest status" variable this is implausible. It suggests ex-smoker codes were not captured and that "current" may mean "ever recorded current" (inferred).
- The methods apply missing indicators to "other predictors" (FT 110), yet smoking has none. The likeliest reading is that no current-smoking record was treated as a non-smoker, like diagnoses.
- Because LASSO zeroed both Never and Ex into the "Ex/never" reference, only the current-smoker flag (+0.0685) matters. The 90-parameter count is consistent with smoking having 3 levels.
- **Verdict: confirmed** (with the nuance that this is a definitional choice that does not affect predictions, rather than a hard contradiction).
- **Recommendation:** binary current-smoker flag, with absent or unknown status counted as 0. Define the lookback explicitly (latest status recommended) and run a sensitivity analysis with "ever current".

---

## C6. Age range 65–95

- Explicit: Fig S3.1 (IMG image1.png) shows integer ages 65 to 95 on an axis labelled "Age (years, from week of birth)", with no points above 95.
- The original eFI was developed in "patients aged 65–95" (EFI16 12 and 40). The eFI+ protocol says "Patients ≥65, defined by the existing eFI" (PROT 83). The A&A paper only says "aged ≥65 years" (FT 80).
- In Wales, roughly 0.5% or more of people aged 65 and over are over 95 (inferred), so their absence needs explaining.
- Top-coding and restriction cannot be separated from the plot.
- **Verdict: partially_confirmed.** The 65–95 range is confirmed. The more likely explanation is an unreported upper age cap inherited from the eFI rather than top-coding.
- **Recommendation:** primary population aged 65–95. Handle ages above 95 as a sensitivity analysis: exclude, or cap age at 95 in the LP. Report their count.

---

## C7. Reporting inconsistencies

- **(a) Wrong table reference in Box S3.1 — confirmed.** Box S3.1 and S3.2 say "[see table S3.1 for values]" (SUPP 318, 368), but the coefficients are in S3.2. HTA Table 16 corrects this to Table 15 (HTA 1051), but HTA Table 34 says "see Table 35 for values" (HTA 2075). That would double-apply recalibration if taken literally.
- **(b) Two tables labelled S3.4 — confirmed.** Subgroups (SUPP 393) and TP/FP (SUPP 451). The main text uses S3.4 for subgroups (FT 280) and "S3.4 and S3.5" for sensitivity/specificity (FT 289).
- **(c) Confidence intervals that exclude their estimates — confirmed.** 6 of 68 estimates fall outside their own CI: IMD1 slope 1.123 (1.126 to 1.158) and C 0.788 (0.789 to 0.796); mild frailty slope 1.300 (1.234 to 1.298) and C 0.735 (0.722 to 0.733); male slope 1.368 (1.371 to 1.393) and C 0.834 (0.835 to 0.839). The same values appear in HTA Table 33 (HTA 2037-2059). Separately, EV O/E CIs are about 3 to 6 times narrower than a Poisson approximation: overall O/E 0.430–0.437 reported versus about 0.415–0.450 expected; female 0.413–0.420 versus 0.397–0.439 (c_ci.py).
- **(d) Recalibrated pooled C CI — confirmed.** "0.801 to 1.000" (FT 252; HTA Table 36). The point estimate (0.816), τ² (0.078) and PI (0.715–0.886) are identical to the pre-recalibration column, whose CI is 0.801–0.830, and recalibration cannot change discrimination. This is a typo.
- **(e) Bradford IMD3 row — confirmed.** 473 + 13,864 = 14,337, not the printed 13,337; the 16.3% was computed from the wrong figure (correct 17.6%). The IMD totals column sums to 80,685. S3.4 gives n = 14,337. Also: the Bradford "Never" smoking total is off by +16, and the suppressed SAIL cell for "Previous higher risk/harmful drinking" events can be derived as 90 − 82 = 8 (FT 183), which defeats the small-cell suppression.
- **(f) Apparent pooled C 0.72 vs S3.3 0.7434 — refuted as an inconsistency.** These are different estimands: a random-effects pooled within-practice C versus a whole-population C. Bradford shows the same pattern (0.816 versus 0.825). Table 2's τ² and PI are internally consistent on the logit scale (τ² implied 0.0103 versus 0.010 reported).
- **(g) Penalised vs unpenalised differences — refuted as an inconsistency.** The gaps are predicted by LASSO KKT shrinkage Δ ≈ λ·sd/Var_w(x), with λ = 0.000123 on standardised covariates (c_shrink.py).

  | Predictor | Observed gap | Predicted gap |
  |---|---|---|
  | Previous drinking | 0.119 | 0.130 |
  | Higher-risk drinking | 0.072 | 0.079 |
  | Zero alcohol | 0.047 | 0.054 |
  | Weakness | 0.043 | 0.045 |

  Across rare predictors (n < 2000) the correlation is 0.95. This confirms the penalised column is genuine.
- **(h) Others — confirmed:**
  - HTA says 74 predictors (HTA 947) versus 75 in A&A.
  - HTA says bootstrapping "with 50 samples" (HTA 539) versus 25 (FT 118).
  - HTA gives Connected Bradford as 88,947 participants and 3,079 events indexed 1 April 2018 (HTA 459, 558, 569). A&A gives 81,685 participants and 2,389 events indexed 1 January 2019 (FT 80, 101, 144).
  - R version 4.3.1 (HTA 502) versus 4.2.3 (FT 106).
  - HTA Tables 8, 14, 15, 20 and 35 print "Higher-risk drinking" twice (HTA 598-599, 932-933, 971-972). The second row holds A&A's "Lower risk drinking" numbers (603 / 10,628).
  - Table S3.3 CITL CI "−0.0115 to 0.115" (SUPP 338) should read 0.0115.
  - The IECV C τ² of 0.002 is on the C scale; its PI implies a logit-scale τ² of 0.055, while other columns report logit τ² (c_table2.py).
  - O/E was pooled on the original scale for development-apparent but on the log scale for IECV and EV (PIs are symmetric in log units), whereas HTA 543 says original scale.
  - Box footnote: "unique drugs" versus "Unique BNF sub-sub-chapters".
  - Fig S3.13 caption says "development data" for EV curves; Fig S3.7 caption says "mortality model".
  - Bradford subgroup row labelled "Missing WIMD" (HTA 2046).
  - HTA says "455 GPs from across South Wales" (HTA 555).
  - The sex-term and constant issue (C1) and the recalibration issue (C8) belong here too.
- **Verdict: partially_confirmed** ((a)–(e) and (h) confirmed; (f) and (g) refuted as inconsistencies).
- **Recommendation:** treat (d) and (e) as typos, and use FT/SUPP counts corrected by row and column sums. Do not use the reported EV CIs as benchmarks for our own uncertainty. Implement no "fix" for (g).

---

## C8. Recalibration discrepancy

- Recomputed (c_basic.py):
  - T35/T15 ratio = 1.210000000 for all 75 coefficients.
  - Constant: −7.25089539 = α + 1.21 × (−5.954459), which gives α = −0.04600000 exactly. So Table 35 is an exact transform with (α, β) = (−0.046, 1.21) on LP_A.
- Other (α, β) combinations implied by the T35 constant:
  - β 1.21 on the Box intercept → α = 0.321
  - β 1.25 on the Table constant → α = 0.192
  - β 1.25 on the Box intercept → α = 0.572
  - α −0.423 → β = 1.147 (Table constant) or 1.091 (Box intercept)
- No LP redefinition reconciles T35 with Box S3.2. The slopes differ (1.21 versus 1.25), and 1.21 is not a rounding of 1.248 (overall slope) or 1.203 (pooled slope, which rounds to 1.20).
- Box S3.2 is confirmed by independent explicit data:
  - Fig S3.11 (IMG image13/14): LP mean −2.99 → −4.16 and SD 0.905 → 1.13. That gives β = 1.1300/0.905 = 1.2486 and α = −4.16 + 1.2486 × 2.99 = −0.427.
  - Medians and IQRs agree: −0.423 + 1.25 × (−3.12) = −4.32 versus −4.31; 1.26 × 1.25 = 1.575 versus 1.58.
- The glm(y ~ LP) slope is by definition the overall calibration slope, 1.248 (FT 203). Simulating the Bradford LP distribution from O/E 0.432, CITL −0.931 and slope 1.248 gives α = −0.421 (normal) or −0.412 (skewed) (c_lp.py).
- On the same population, T35's (−0.046, 1.21) gives mean risk 4.4% against observed 2.9% (O/E 0.67), so it cannot be a Bradford recalibration of this LP.
- Recalibrating on LP_A in the simulated population gives (−0.67, 1.23), not (−0.046, 1.21) (c_recal_A.py).
- The HTA text describes fitting glm on LP_eFalls (HTA 1149-1153) and points to Table 35 for coefficients.
- The origin of 1.21 and −0.046 was not found. HTA 1156 mentions testing in Greater Manchester, and the HTA Bradford cohort was 88,947 / 3,079, but neither can be verified as the source.
- HTA Table 34's p = 0.106 matches Box S3.2, not Table 35 (0.148).
- **Verdict: confirmed.**
- **Recommendation:** if recalibration is needed as a reference, use Box S3.2 on LP_C:
  - p = expit(−0.423 + 1.25 × LP_C)
  - Equivalent coefficient form: constant −8.2457 (female baseline), male −0.3796, other terms 1.25 × Table S3.2
  - Do not use HTA Table 35.
  - For Meuhedet, recalibrate locally (intercept and slope on our LP_C), because Bradford's α and β are population-specific and only apparently validated.
