# How eFalls has been implemented outside the development papers (as of 14 Sep 2026)

Scope: what the non-development implementations actually did. The target implementations are the NIHR ARC Greater Manchester (ARC-GM) SWAN pilot and the Sci Rep 2026 working-age validation. A third source turned up while tracing provenance and is directly relevant: the Leeds implementation instructions for eFI2/eFI+ (eFalls), which are what EHR suppliers receive.

Labels: **[EXPLICIT]** = stated in the source. **[INFERRED]** = my reasoning or computation, with the script named. **[NOT FOUND]** = searched for and absent.

Quotes are kept short. Line numbers refer to the derived line-numbered text files listed in section 0. Printed page numbers are also given for the ARC-GM report.

---

## 0. Sources read, provenance and derived files

| # | Source | Local file(s) | Public URL | What it is / status |
|---|---|---|---|---|
| S1 | Money A, Badrock B, ... Clegg A, Todd C. *Case Finding for Falls Prevention – 'eFalls' pilot study evaluation report*. NIHR ARC-GM, March 2026 (59 pp) | `raw/arcgm_efalls_report_clean.txt` (single line with `[PAGE] n` markers); `raw/arcgm_efalls_report.pdf`; **derived:** `work/impl/arcgm_layout.txt` (pdftotext -layout, 2148 lines); rendered pages and extracted figures `work/impl/arcgm_p-*.png`, `work/impl/img-00*.jpg` | Linked as "Full Report" from https://arc-gm.nihr.ac.uk/projects/Case-Falls-Prevention-Pilot-Evaluation | Read in full, including the image-only Box 1, Box 2, Figures 1–4 and Appendix 2 images. |
| S2 | Chen T, Marino LV, Best K, ... Relton S, Lim S, Clegg A. *Extending eFall risk prediction to working-age adults within mental health and learning disability services*. Sci Rep 2026, doi:10.1038/s41598-026-51298-0 (accepted manuscript, published 10 Aug 2026) | `raw/scirep_efall.txt` (landing page: abstract only); `raw/scirep_efall.pdf` (actually HTML, not a PDF); **full text obtained this round:** `work/impl/scirep_51298_reference.pdf` → `work/impl/scirep_ms_layout.txt` (1429 lines) | https://www.nature.com/articles/s41598-026-51298-0 ; manuscript PDF https://www.nature.com/articles/s41598-026-51298-0_reference.pdf | CC-BY 4.0. The local txt had only the abstract. The full manuscript was fetched from nature.com (open access) and read in full. |
| S3 | Sci Rep Supplementary Table S1 | `raw/scirep_efall_supp.txt`, `raw/scirep_efall_supp.pdf` → `work/impl/scirep_supp_layout.txt` (103 lines) | https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-026-51298-0/MediaObjects/41598_2026_51298_MOESM1_ESM.pdf | Read in full. |
| S4 | `raw/arcwx_efalls.html` | → `work/impl/arcwx_efalls.txt` | intended: https://www.arc-wx.nihr.ac.uk/research-areas-list/refinement-of-an-efalls-tool---... | **"Page not found" (404).** The live URL gives the same 404 via curl, and WebFetch failed with a TLS error. Project facts below come only from a search-engine summary (unverified). |
| S5 | `raw/wr_219096.html` | → `work/impl/wr_219096.txt` (119 lines) | https://eprints.whiterose.ac.uk/id/eprint/219096/ | White Rose record for Archer 2024: metadata and abstract only. Deposited 1 Nov 2024. **No correction or erratum is noted** (lines 66–71). Crossref probes `raw/crossref_updates.json` and `raw/crossref_aa_corrig.json` return 0 updates. |
| S6 | Leeds *Intended use and implementation instructions for the eFI2 and eFalls* v1.9 (docx created 13 Jun 2025), and *...eFI2 and eFI+* v1.10 / v1.13 (created 26 Feb 2026); eFI2/eFI+ Change Log; Hazard Log; Risk management plan | `raw/efi2plus/instr_v1.9.txt`, `instr_v1.10.txt`, `instr_v1.13.txt`, `eFI2_+ Change Log.txt`, `Hazard Log.txt`, `Risk management plan.txt` | GitHub **KateBest/eFI2plus** ("MHRA documentation for the eFI2+"): https://github.com/KateBest/eFI2plus | **Provenance verified this round:** `git hash-object` of the local docx files equals the GitHub blob SHAs (v1.9 053eb32…, v1.10 0d82fed…, v1.13 4eff21c…, change log 4cb936d…). All three instruction docs were **deleted** from the repo on 26 Feb 2026 (commits 87084ee, 97d388f, 73df6ce). `efi_codelists.md` did not cover these files. |
| S7 | Web pages on the GM pilot | — | GM ICP news, 15 May 2025: https://gmintegratedcare.org.uk/health-news/world-first-efalls-tech/ ; BGS 2025 abstract/poster 3564: https://www.bgs.org.uk/case-finding-for-falls-prevention-wigan-pilot (poster PDF text at `work/impl/bgs_poster_3564.txt` and render `work/impl/bgs_poster-1.png`); Centre for Ageing Better blog, 19 Sep 2025: https://ageing-better.org.uk/blogs/greater-manchester-leading-way-falls-prevention-through-ageing-better-supported-project ; ARC-GM news 15 May 2025 and project page (above) | Read through WebFetch (model-summarised). The quotes below are short. |

Audit scripts written this round:
- `work/impl/scirep_citl_check.py`: what the Sci Rep "CITL" actually is (section 2.8).
- `work/impl/gm_band_checks.py`: threshold-band mapping, band-size check, sex-coding checks (sections 1.3, 1.4, 4).
- `work/impl/instr_example_check.py`: arithmetic of the worked examples (section 3.6).

---

## 1. ARC-GM SWAN pilot (Wigan, Greater Manchester): the first real-world use

### 1.1 What was done [EXPLICIT]
- **Setting and dates.** South Wigan Ashton North PCN: 7 GP practices on 4 sites, about 34–35k patients (Box 1 image, printed p.8). The news release gives about 37,000.
  - The algorithm was run in the Secure Data Environment (SDE) in **early February 2025** (arcgm_layout.txt L556–557, p.15).
  - The pilot ran Feb–Nov 2025 (L452, p.12).
- **Where it ran.**
  - The eFalls tool was "implemented on the Greater Manchester Shared Care Record (GMCR)" and results sent to PCNs (L353–355, p.9).
  - It "was run in the SDE", and the list was securely transferred to PCN staff and linked to primary care records (L397–401, p.10).
  - Figure 1 (image, p.9) shows: "eFalls algorithm run in Secure Data Environment", then a patient list provided via NHS GM, then a clinical check.
  - GM ICP news: searches "are being run by the NHS Greater Manchester data team". Long-term aim: embed the toolkit in GP IT systems "in just a few clicks" (S7).
  - PCN clinical director (report p.42, L1710–1713): they used "the embedded E-Falls toolkit within an integrated GP IT search framework". He said this used primary care data already held in their systems and avoided external datasets.
  - **The EHR supplier or search product (EMIS / SystmOne / Ardens) is not named anywhere. [NOT FOUND]**
- **Population and threshold.**
  - People ≥65 years at "intermediate risk (10-25%)" of a fall in 12 months resulting in hospital admission (L346–349, p.8).
  - Elsewhere the outcome is worded as "a fall requiring hospital treatment" (L398–399).
  - Figure 1 box: "Age 65 years +" and "10%-25% risk of a fall in the next 12 months".
- **Why 10–25% ("intermediate").**
  - The report frames it through the World Falls Guidelines low/intermediate/high stratification (L280–293, p.7). Intermediate-risk people get targeted strength and balance exercise; high-risk people get multifactorial assessment (L288–293). It then cites Archer 2024 [9] for the risk computation (L348–349).
  - The ARC-GM project page repeats "intermediate fall risk (10-25% risk)".
  - No other rationale is given (e.g. capacity, net benefit, calibration). **[EXPLICIT for what is given; NOT FOUND for any statistical rationale]**
- **Numbers (Figure 3 image, p.19; text L586–594).**
  - Algorithm list: **N=1158**.
  - "System exclusion criteria applied*": **N=273**. The asterisk is never defined. [NOT FOUND]
  - Eligible list sent to PCN: 885.
  - Excluded by GP: 145 (housebound 43, under falls team 35, illness/injury 14, passed away 13, other 14, blind/deaf 12, EOL/frailty 6, care home 5, progressive dementia 3).
  - Final list 740. Declined 354, unable to contact 188, accepted 198.
  - Attended 160; referred 125; declined referral 35 (one reason given was "Age - too old"). Enrolled 54.
- **Cohort profile of the "intermediate" group** (routine data; not a performance evaluation).
  - 1150 in the SDE dataset (8 fewer than 1158 because SDE data are "fluid": footnote 2, L710–719). 55% male (L692–693).
  - Ages 65–69 3%, 70–74 9%, 75–79 22%, 80–84 28%, 85–89 26%, 90+ 11% (L695–697).
  - Figure 4 counts (image p.20), F/M: 65–69 13/26; 70–74 41/65; 75–79 93/161; 80–84 140/185; 85–89 153/144; "90*" 82/47. The asterisk is undefined.
  - Frailty by eFI in the primary care record (N=1128): robust 10%, mild 32%, moderate 30%, severe 28% (L734–744).
  - Living status recorded for 656: 11.6% care/nursing home, 14.5% supported accommodation (L728–730).
  - **Polypharmacy: 95% had ≥5 medications, "recorded in the 93 days up to the data snapshot"** in the primary care dataset (L746–750). This is a descriptive statistic. The report does not say this was the window used inside the eFalls calculation.
  - Secondary care over 23 months (25-02-2024 to 09-01-2026):
    - falls attended by Urgent Community Response 160;
    - falls attended by ambulance 217;
    - fractures in A&E 59;
    - falls-clinic/elderly-medicine referrals 118;
    - A&E-assessed falls reported as "0" (small-number suppression) (Table 5, L862–880);
    - 826 people had 2961 admissions (L884–886).
- **Performance.** The report says the pilot "is not an evaluation of the effectiveness of either the eFalls algorithm or the intervention" (L61–63, and footnote 1 at L406–409). **No discrimination, calibration or observed fall-rate-versus-predicted analysis was reported. [EXPLICIT absence]**
- **Problems the implementers flagged** (L1683–1702, pp.41–42; also L1146–1149, p.29):
  - (a) Some participants reported previous falls, and many falls/fractures were recorded in secondary care. This is called noteworthy because the algorithm, set at intermediate risk, "should be identifying persons without previous fall history" (L1686–1688). They attribute it to lags or non-recording in the EHR and say it "requires clarification by future research … data reliability and validity issues for eFalls implementation".
  - (b) Participants "appeared frailer (58% were moderately or severely frail …) than might have been expected given their eFalls estimated fall risk" (L1696–1699).
  - (c) Referral counts differed between PCN and provider, and access to SDE data was not timely (L1683–1686).
  - (d) Running the algorithm and sharing results "worked well from SDE to PCN" (L1836–1838).
  - **No error or discrepancy in the published equation or coefficients is mentioned. [NOT FOUND]**
- **Recommendations relevant to transport** (L1851–1918, pp.45–46):
  - a standardised GM-wide model for exclusion criteria (rec 14);
  - integrated data flows SDE → PCN → provider (recs 5, 15);
  - better SDE usability (rec 6).

### 1.2 Items the ARC-GM sources do not state [NOT FOUND]
Searched: clean txt, layout text, all images (Box 1, Box 2, Figures 1–4, Appendix 2–3 pages), and the web pages in S7. None of these states:
- which equation or coefficient table was used (Table S3.2 / HTA Table 15 vs Box S3.1 vs a recalibrated model);
- how the sex term was applied;
- BMI cut-points;
- the polypharmacy window or BNF level used in the computation;
- the code lists or where they came from;
- the EHR supplier or search tool;
- the reference date and look-back windows;
- handling of missing BMI/alcohol/smoking;
- any upper age limit;
- the definition of the 273 "system exclusions".

Box 2 (image, p.11) is only a narrative description of eFalls (750,000 records, about 35,000 events, Wales and England datasets) plus "first global implementation … in routine primary care".

### 1.3 Indirect evidence on which coefficients GM used [INFERRED, `work/impl/gm_band_checks.py`]
- **Recalibrated or not.**
  - In Connected Bradford (HTA Tables 32 and 37, `raw/gjac1008_struct.txt` L2005–2030 and L2211–2230), the share of people ≥65 with predicted risk in [10%, 25%) is **16.1%** with the unrecalibrated model and **5.6%** after recalibration.
  - SWAN flagged 1158 before exclusions. The unrecalibrated share implies about 7,200 people aged ≥65 (about 21% of a 34,500 list), which is plausible for an ageing, deprived PCN.
  - The recalibrated share would need about 20,700 people aged ≥65 (60% of the list), which is implausible.
  - So GM most likely applied the **unrecalibrated** published coefficients, not Box S3.2 or HTA Table 35.
  - Caveats: the SWAN 65+ count is not reported, and case-mix differs from Bradford.
- **Sex coding.**
  - Intermediate-group sex split: 54.6% male overall, **63.2% male at 65–79**, 52.9% at 80–89, 36.4% at 90+ (Figure 4).
  - Box 1 gives about 50/50 for all ages. Older populations are normally female-majority, but no age-sex denominators for SWAN are available here.
  - At 65–79 most people sit below 10%, so the sex with the higher linear predictor crosses the threshold first. Men being enriched fits the **Table S3.2 / HTA Table 15 / Leeds instructions coding (Female −0.303708, intercept −5.954459)**. It fits poorly with Box S3.1 "−0.304 (if male)", where women get the higher linear predictor.
  - This is **weak** evidence. Possible confounders: an undefined exclusion of people with prior falls (the implementers expected none), sex differences in comorbidity, and the 25% upper cut removing the highest-risk men.
- **Meaning of the band on other scales.** If the band was on the unrecalibrated scale, 10–25% corresponds to about **4.0–14.2%** on the Bradford-recalibrated scale (Box S3.2: α −0.423, β 1.25) and about 6.3–20.2% on the HTA Table 35 scale. In the Bradford validation, the unrecalibrated model over-predicted (O/E 0.432). This may partly explain why people flagged as "intermediate" looked frailer and had more recorded falls than expected. [INFERRED]

### 1.4 The "intermediate 10–25%" band versus the development team's meaning of 10–25% [EXPLICIT sources; INFERRED contrast]
- In the HTA (`raw/gjac1008_struct.txt` L524–535, Table 7), 10–25% is the pre-specified range of **threshold probabilities, i.e. lower bounds on predicted risk** for calling someone "high risk" and changing treatment. It was set with an Implementation Advisory Group and PPI input. Example text: above 10%, a person "would be considered at high risk of falling" (L524).
- The Leeds instructions (S6, `instr_v1.9.txt` L41) say pre-specified low/medium/high thresholds "are not set". Action thresholds should be agreed by local clinicians and commissioners.
- ARC-GM turned the range into a two-sided **band** (10% ≤ p < 25%, "intermediate"). People ≥25% were not the target of this pilot. This band use is a local choice and not a validated category.
- Also note: on Bradford external validation, the model before recalibration had no net benefit over treat-none in 10–25% (HTA L1138; Age&Ageing fulltext L289).

---

## 2. Sci Rep 2026: working-age external validation (South West Yorkshire Partnership Teaching NHS FT, Wakefield)

### 2.1 Design and data [EXPLICIT, scirep_ms_layout.txt]
- Retrospective external validation in EHRs from the integrated teaching NHS FT at Wakefield, ages **18–65** (L209–212).
- Inclusion: under active Trust care, aged 18–65 as of Feb 2023. Baseline = most recent hospital contact on or before **28 Feb 2023** (L213–214).
- GP records were linked by NHS number where a data-sharing agreement existed. Patients at practices without an agreement, or who had opted out, were excluded (L215–217).
- Predictors came from retrospective data before baseline; the outcome came from 12 months after (L218–222). **The look-back window per predictor is not stated. [NOT FOUND]**
- Outcome: any ED attendance or hospitalisation for falls/fractures within 12 months, from diagnostic codes for fall-related injuries (L239–241). **The code list and data source (ECDS/HES vs Trust records vs GP codes) are not specified. [NOT FOUND]**
- **The EHR system (e.g. SystmOne/RiO) is not named. [NOT FOUND]**
- Ethics: HRA/HCRW 24/PR/1367 (L245–251). Funder: NIHR ARC Wessex (L1388–1390).
- N=32,410; events 670 (Table 2 header, L497) or 671 (Tables 4–5, L935, L1027); fall rate 2.07%.

### 2.2 Equation and coefficients [EXPLICIT + computed check]
- "The full set of published eFalls coefficients was applied unchanged" (L314). Logistic transform P = 1/(1+exp(−LP)), with LP = β0 + Σ βj·Xj (L343–349).
- Supplementary Table S1 (scirep_supp_layout.txt L1–97) lists every coefficient: **Male = Reference; Female = −0.303708; Constant = −5.954459**; BMI Overweight = Reference; Smoking Ex/never = Reference; **Alcohol "Lower risk drinking" = Reference** (L23–29).
- **Computed:** all 80 rows match HTA Table 15 (`raw/hta_table15_efalls_coefficients.csv`) numerically. The only differences are labels: (i) the polypharmacy term is called "Medication count"; (ii) the alcohol reference is labelled "Lower risk drinking", where HTA Table 15 prints "Higher-risk drinking" twice. This supports open issue (iv) being a typo in HTA Table 15. There are 75 non-reference parameters plus the constant, which matches the paper's "75 predictors" (L227). The HTA text says LASSO "retained 74 predictors" (gjac1008 L947), so the counting convention differs.
- **Box S3.1 and the recalibrated equations were not used.** No mention of Box S3.1, the −6.258 intercept or "−0.304 (if male)". [EXPLICIT absence]
- **Sex term:** applied as Female −0.303708 relative to Male (S1 L12–15). Consistency check (gm_band_checks.py §6): mean predicted risk was lower in women (1.15%) than men (1.44%), although observed risk was higher in women (2.37% vs 1.74%). That fits the Table S3.2 coding. [INFERRED]

### 2.3 Predictor derivation [EXPLICIT where quoted]
- **Code lists.** Clinical code mapping "strictly followed the specifications provided in the original eFalls study". CTV3, Read v2 and SNOMED CT were harmonised and validated. **Only the exact codes, not parent or child concepts,** were used. All 75 predictors were available and no proxies were needed (L223–229).
  - **Where the lists came from is not stated.** The official lists are "on request". Three co-authors are the Leeds eFalls/eFI2 team (Best, Relton, Clegg). [NOT FOUND / INFERRED]
- **Polypharmacy.** Called "medication count (referred to as polypharmacy in the original eFalls paper)" (L243–244). Table 1: mean 3.2, median 0 [IQR 0–5]; fallers median 8 (L482–484).
  - **Look-back window (120 vs 90 days), BNF level and data source (GP vs Trust prescribing) are not stated. [NOT FOUND]**
- **BMI.** Categories Underweight/Normal/Overweight/Obese/Missing (Table 1, L427–440). **Cut-points not stated. [NOT FOUND]**
- **Smoking.** Current 40.2%, Ex/Never 59.8%; "fully observed" (L259, L445–446). [INFERRED] Missing smoking was probably folded into Ex/never, as the Leeds instructions specify "None/missing" (S6 L111).
- **Alcohol.** Six categories including Missing, 77.0% missing (L451–464).
- **Learning disability** was not an eFalls predictor; it was used as an external stratifier. The mental health stratifier is built from eFalls predictors (depression, general mental health, self-harm, SMI, stress) (L292–303).

### 2.4 Missing data [EXPLICIT, L253–275]
- BMI 9.8% and alcohol 77.0% missing, both kept as an explicit "Missing" category with no imputation. The authors say this mirrors the original model.
- Age, medication count and smoking were fully observed.
- An absent diagnostic code was taken to mean the condition is absent.

### 2.5 Observed performance [EXPLICIT]
- **Whole cohort, before recalibration (bootstrap 1000; Table 3, L604–619):**
  - calibration slope 1.222 [1.139, 1.312];
  - reported "CITL" 1.357 [1.032, 1.696];
  - O/E 1.604 [1.584, 1.623];
  - C 0.777 [0.757, 0.795].
- **After intercept-and-slope recalibration** (fitted once on the full cohort): slope 1.000, CITL −0.001, O/E 1.000, C unchanged.
- **The recalibration α and β are not reported explicitly. [NOT FOUND]** See 2.8 for the implied values.
- The predicted-risk histogram had very few patients above 10% and a lack of values above 0.30 (L772–777).
- **Sex (Table 4, L931–935):**
  - Male: slope 1.304, CITL 1.335, O/E 1.206, AUC 0.783.
  - Female: slope 1.235, CITL 1.699, O/E 2.060, AUC 0.784.
  - Risk was underestimated more in women (L956–1000).
- **MH/LD (Table 5, L1017–1027):**
  - MH_only: AUC 0.760, O/E 1.236.
  - LD_only: AUC 0.696, O/E 5.720, slope 0.763.
  - MH_and_LD: AUC 0.739, O/E 5.929.
  - No_MH_No_LD: AUC 0.815.
- **DCA:** thresholds 0.01–0.49, with shading at 10–25% "consistent with what original eFall paper specifies" (L334–336, L881–885). Both curves were above treat-all and treat-none in 10–25%; net benefit about 0.002 at 0.10 (L894–898).
- Comparison column in Table 3: "eFalls Original" Connected Bradford values 1.248 / −0.931 / 0.432 / 0.825. These match the Age&Ageing Table 2 **whole-population** external-validation column (fulltext.md L203, L219, L235, L251), not the pooled estimates (1.203 / −0.874).

### 2.6 Lessons the authors drew [EXPLICIT]
- Miscalibration went in the **opposite** direction from Bradford: under-prediction despite a lower event rate (2.07% vs 4.9% in development). They attribute this to predictor–outcome relationships learned in people aged ≥65 not carrying over (L714–730).
- Recalibrate locally before deployment (L745–746, L903–906).
- Recalibration alone is insufficient in LD, which may need LD-specific predictors or interactions (L1241–1250, L1303–1308).
- Consider updating the age coefficient alone ("partial model revision"). This needs the full penalised model specification (L1254–1282).
- Following NICE NG249: use eFalls as system-level triage support within multifactorial pathways, not as a standalone tool (L1202–1218).
- **No error in the published eFalls equation or supplement is reported. [NOT FOUND]**

### 2.7 Data-quality red flags in the Sci Rep predictor prevalences [INFERRED]
Table 2 (L496–597) shows prevalences that are implausible for adults aged 18–65 and far above the development data (SAIL, supp.md L167–169):

| Variable | Sci Rep (18–65) | SAIL (≥65) | Bradford (≥65) |
|---|---|---|---|
| Hypotension/syncope | **40.3%** | 5.7% | 17.9% |
| Hypertension | 39.0% | 52.6% | 67.0% |
| Housebound | **16.3%** | 11.3% | 10.9% |

Musculoskeletal problems 40.3%, requirement for care 10.0% and social vulnerability 18.2% are also high.
- The Leeds instructions define hypotension/hypertension partly from ≥3 blood-pressure readings ever (S6 L104–107). Frequent observations in mental-health or inpatient care would inflate these flags.
- The Sci Rep text mentions only "exact clinical codes". Its numeric rules, or any over-broad codes, cannot be verified.
- Bone disease and motor neurone disease had 0 cases.
- Lesson for transport: numeric-rule deficits depend on the setting, so check prevalences against the development Table S3.1 before scoring.

### 2.8 The Sci Rep "CITL" is not standard CITL [INFERRED, `work/impl/scirep_citl_check.py`]
- For rare outcomes, standard CITL (intercept with LP as offset) ≈ ln(O/E). For the whole cohort ln(1.604) = 0.47, yet "CITL" = 1.358 is reported.
- Method: model the LP as normal. Match each group's prevalence, O/E and AUC, and use the reported slope. The implied **free-slope intercept α** from logit(Y) = α + β·LP then reproduces the reported "CITL" in all five groups tested:

| Group | Implied α | Reported "CITL" |
|---|---|---|
| Whole | 1.362 | 1.358 |
| Male | 1.346 | 1.335 |
| Female | 1.694 | 1.699 |
| LD_only | 0.961 | 0.986 |
| No_MH_No_LD | 2.357 | 2.373 |

- The implied standard CITLs are 0.49, 0.19, 0.75, 1.91 and 0.32.
- **Conclusion:** the Sci Rep "CITL" is the α of its own recalibration model. The Trust recalibration is therefore approximately **logit(p) = 1.358 + 1.223·LP_eFalls** (whole cohort).
- This also explains the authors' puzzle that "CITL" was less extreme than O/E in LD groups (L1039–1051). It is an artefact of the definition.
- Same method applied to Age&Ageing Bradford (O/E 0.432, prev 2.9%, AUC 0.825, slope 1.248):
  - implied standard CITL −0.943 vs reported −0.931, so Age&Ageing used standard CITL;
  - implied free-slope α −0.456 vs **Box S3.2 α = −0.423 with β 1.25**.
  - So Box S3.2 is internally consistent with the Bradford whole-population metrics. This is relevant to open issue (ii): HTA Table 35's multiplier of 1.21 is closer to the pooled slope of 1.203, so it may be a different (pooled) recalibration. That is unverified.

---

## 3. Leeds implementation instructions (KateBest/eFI2plus): what implementers are told

These come from the development team, but they are the operational specification. The Risk management plan says the team liaises with "TPP, EMIS, Cerner UK" (Risk management plan L12). They are the closest public evidence of how eFalls is meant to be built into EHR systems.

### 3.1 Equation [EXPLICIT, instr_v1.9.txt L355–663; the same table is in v1.10/v1.13]
- Logistic model. **Intercept −5.954459; Sex–Female −0.303708 (Sex–Male blank = reference)** (L370–404). BMI Overweight 0 and Alcohol Lower risk 0 are shown as reference.
- The coefficients match Table S3.2 / HTA Table 15. The polypharmacy term is ln((Polypharmacy+1)/10), coefficient 0.3296295.
- **Box S3.1 (−6.258, "−0.304 if male") is not used.** [EXPLICIT absence]
- The worked example is a **female**, age 70 (L358–362), and the female coefficient is applied.
- **Change log** (L11–12, 25 Feb 2024): eFalls equations were "updated … to match final outputs from SAIL (and publication in Age & Ageing)". Earlier versions of the instructions carried different eFalls equations; those are not available locally. [EXPLICIT / NOT FOUND]

### 3.2 BMI [EXPLICIT, L108–109]
- Standard WHO cut-points: underweight <18.5; recommended 18.5–24.9; overweight 25.0–29.9 (printed "<25.0-29.9"); **obese ≥30**.
- Use the **most recent** measurement or code. BMI can be computed from height and weight. Missing if nothing is recorded.
- This contradicts Box S3.1 "obese if BMI ≥ 40" (supp.md L318). It agrees with the HTA box wording "standard BMI cut-offs" (gjac1008 L1052). Relevant to open issue (iii).

### 3.3 Polypharmacy [EXPLICIT, with a contradiction]
- v1.9 L113 and v1.10/v1.13 L112: number of different **BNF 2017-18 sub-sub-chapters** prescribed in the **previous 90 days**, using a separate BNF chapters table of included and excluded chapters. That table is not in the local files. [NOT FOUND]
- Box S3.1 / HTA: unique drugs (BNF sub-sub-chapters) over **120 days**, excluding non-drug chapters (supp.md L318, L322; gjac1008 L1052).
- **GitHub commits 233b23f and 5e0a485 (26 Feb 2026)** say the instructions were "Updated to change the window for calculating polypharmacy to 120 days as opposed to 90 days". **But the uploaded v1.13 and v1.10 files, byte-identical to the local copies, still say 90 days.** All instruction files were then deleted from the repo the same day.
- So anyone who implemented from the instructions between at least June 2025 and Feb 2026 (probably including the GM pilot, run Feb 2025) would have been told **90 days**, not the 120 days used in development. The GM window is not reported (section 1.2). [EXPLICIT facts; INFERRED implication]

### 3.4 Other derivation rules [EXPLICIT]
- **Codes.** 7555 (v1.9) or 7556 (v1.13) unique SNOMED CT codes in 79 deficit variables, also available in CTV3 and Read 2. "We only recommend using the specified codes", not parent codes (L45).
  - "Time constraint" and "Age limit" columns restrict codes to within N years or after age X (L48).
  - The codelist itself is not in the repo. [NOT FOUND]
- **Alcohol** (L101–103).
  - Units per week: 0 = zero; 1–20 = lower; 21–48 = higher; 49+ = harmful. Units per day ×7.
  - Take the **highest category over the last 5 years**.
  - Previous-harmful codes count unless a later harmful code exists.
  - Missing if nothing is recorded.
- **Smoking** (L110–111). Ex / Current / **None/missing** (codes only). An ex-smoker code followed by a current code = current.
- **Hypertension** (L104–106): any code ever, or **≥3 readings ever** of SBP >140 or DBP >90. The three readings cannot mix systolic and diastolic; averaged readings count once.
- **Hypotension** (L107): three readings ever with SBP <90 or DBP <60.
- **Anaemia** (L114–119): code or Hb below threshold (male <13.0 g/dL, female <11.5 g/dL); 5-year limit for most codes; resolves after a later normal Hb.
- **Dementia hierarchy** (L144–145): dementia overrides cognitive impairment, which overrides memory concerns.
- **Activity limitation** (v1.13 L143–146 only): Barthel 0–18 or 21–90 triggers the flag.
- **Target population** (L148–153): falls model "aged 65 years or older". **No upper age limit is stated** in any version. [NOT FOUND]
- **Thresholds** (L41): not pre-specified; set locally. Use clinical judgement. Not for individual diagnosis. The Hazard Log (L26–33) lists "Misclassification of individual patient risk", mitigated by using the tool alongside clinical assessment.

### 3.5 Arithmetic of the worked eFalls example [INFERRED, `work/impl/instr_example_check.py`]
- Example: female, 70, 5 medications, underweight, never smoker, lower-risk drinker, back pain.
- The listed terms sum to **LP = −2.978465, P = 0.0484**. The document states LP = −3.023169, P = 0.0464 (v1.9 L359–362).
- The difference (−0.0447) matches no single omitted term: omitting back pain gives −3.0283, and age 69 gives −3.0200.
- **Do not use this example as a unit test.** The Box S3.1 example is fine (LP −1.3678, p 0.203 with 3-dp coefficients). With full-precision Table S3.2 coefficients, the same patient gives LP −1.4085, p 0.1965. Most of the 0.04 gap comes from rounding the age coefficient to 0.042 (0.00045 × 89 = 0.04).

---

## 4. Cross-cutting evidence on the sex term (open issue i)

| Evidence | Direction | Label |
|---|---|---|
| Leeds implementation instructions Table 4 (v1.9/v1.10/v1.13): Female −0.303708, intercept −5.954459; worked example applies it to a female | Table S3.2 coding | EXPLICIT |
| Sci Rep Table S1: Male Reference, Female −0.303708, constant −5.954459; applied "without modification" | Table S3.2 coding | EXPLICIT |
| Sci Rep sex-specific O/E (M 1.206, F 2.060): predicted mean risk lower in women | consistent with Table S3.2 coding having been applied | INFERRED |
| Age&Ageing Table S3.4 (Bradford EV): O/E Female 0.417, Male 0.463, i.e. similar | Under Box S3.1 coding, male predictions would fall to about 0.56× and male O/E rise to about 0.82 vs female 0.417. So the **validated** model's sex-specific calibration is consistent with Table S3.2 coding | INFERRED (`gm_band_checks.py` §5) |
| ARC-GM intermediate group 63% male at ages 65–79 | weakly consistent with Table S3.2 coding | INFERRED, weak |
| HTA narrative (gjac1008 L947): "being female … contributed to a higher predicted fall/fracture risk"; Box S3.1 "−0.304 (if male)"; crude development data: fallers 35.6% male vs 47.2% overall (fulltext.md L153) | Box S3.1 direction | EXPLICIT (contradicts the tables) |
| Working-age Trust data: CITL gap female − male = +0.36 under Table S3.2 coding. Box S3.1 coding would overshoot by about 0.24 in the other direction | inconclusive; different population | INFERRED |

**Net:** every implementation seen used Female = −0.303708 with intercept −5.954459. No implementer reported the Box S3.1 inconsistency. Applying Box S3.1 would shift male predictions by −0.607 on the logit scale. Bradford sex-specific calibration supports the table coding as the validated model. The HTA narrative sentence and Box S3.1 remain unexplained.

---

## 5. Summary table: how eFalls was actually implemented

| Item | ARC-GM SWAN pilot (2025) | Sci Rep SWYPFT (2026) | Leeds implementation instructions (v1.9 2025 → v1.13 2026) |
|---|---|---|---|
| Equation / coefficients | NOT FOUND. INFERRED: unrecalibrated (band-size argument) | Table S3.2 = HTA Table 15, unchanged; then local intercept+slope recalibration (α, β not reported; implied about 1.358, 1.223) | Table S3.2 values, intercept −5.954459 |
| Sex term | NOT FOUND (weak inference: female −0.304) | Female −0.303708 vs Male ref | Female −0.303708 vs Male ref |
| BMI cut-points | NOT FOUND | NOT FOUND (categories only; Missing 9.8%) | Standard: <18.5 / 18.5–24.9 / 25–29.9 / ≥30; most recent value |
| Polypharmacy | Algorithm window NOT FOUND. Descriptive: ≥5 meds in 93 days, primary care dataset | "Medication count"; window, BNF level and source NOT FOUND | BNF 2017-18 sub-sub-chapters in 90 days (Feb 2026 commit says change to 120 days, file text unchanged) |
| Code lists | NOT FOUND ("embedded E-Falls toolkit") | "Strictly followed original eFalls specification"; CTV3/Read v2/SNOMED harmonised; exact codes only; source NOT FOUND | eFI2 codelist, 7555/7556 SNOMED codes; time and age columns; exact codes only; list not public |
| EHR / search | Run in GM SDE on GM Care Record by NHS GM data team; PCN calls it a toolkit in an "integrated GP IT search framework"; supplier NOT FOUND | Trust EHR + linked GP records (NHS number, data-sharing agreements); system NOT FOUND | Implementation via TPP, EMIS, Cerner UK (risk plan) |
| High-risk threshold | Band 10–25% = "intermediate" (World Falls Guidelines framing); ≥25% not targeted | DCA focus 10–25%, cited as original a priori range | None set; decide locally |
| Missing values | NOT FOUND | Explicit Missing categories for BMI and alcohol; absent code = absent; no imputation | Missing categories for BMI and alcohol; smoking None/missing combined |
| Observed performance | Not evaluated (explicit). Flags: previous falls recorded, frailer than expected | C 0.777; slope 1.22; O/E 1.60; female O/E 2.06; LD AUC 0.70, O/E about 5.7 | n/a |
| Reported equation errors | None | None | Worked example LP arithmetic inconsistent (INFERRED); polypharmacy window conflict |
| Age range | ≥65, no upper limit; 11% aged 90+ | 18–65 (outside intended range) | ≥65; no upper limit |
| Exclusions | 273 undefined "system exclusions"; 145 GP-level exclusions | No GP data-sharing agreement; research opt-out | None for falls model (home care and care home models exclude prior use) |

---

## 6. Lessons for transporting eFalls to Meuhedet [INFERRED unless cited]
1. **Fix the coefficient source.** Every implementer used the Table S3.2 / HTA Table 15 values (intercept −5.954459, Female −0.303708). Do not use Box S3.1's rounded equation. Its "−0.304 (if male)" contradicts all implementations, and 3-dp rounding shifts the LP by about 0.04 at age 89.
2. **Resolve the definitions the papers leave open before scoring.** For polypharmacy: window 120 vs 90 days, BNF sub-sub-chapter mapping for Israeli ATC data, which chapters to exclude. Also BMI cut-points (standard ≥30 for obese per the instructions), alcohol and smoking rules (highest alcohol category in 5 years; missing smoking = reference), and numeric BP/Hb rules. Get the codelist and the "BNF chapters table" from Leeds (a.p.clegg@leeds.ac.uk) and record the version used.
3. **Expect miscalibration in both directions.** Bradford over-predicted (O/E 0.43); working-age mental health under-predicted (O/E 1.60). Plan a local validation and intercept+slope recalibration using standard CITL (offset) and report α and β explicitly. The Sci Rep example shows how a non-standard "CITL" can mislead.
4. **Choose thresholds locally and state which scale they apply to.** A 10–25% band on the unrecalibrated scale corresponds to about 4–14% on the Bradford-recalibrated scale. Classification is unstable near the cut-offs (HTA L1063–1067: up to 80% reclassification near 0.10).
5. **Check predictor prevalences against development Table S3.1** before trusting scores. Setting-dependent numeric rules and code-mapping choices can inflate deficits, as seen in Sci Rep: hypotension 40%, housebound 16%.
6. **Decide exclusions explicitly and document them.** Prior falls are a predictor (+0.301), not an exclusion, in the published model. GM's expectation that "intermediate" patients have no fall history does not follow from the model. GM's 273 system exclusions were never defined. Care-home, housebound and end-of-life exclusions were applied after scoring.
7. **Age.** There is no upper age cap in any implementation or in the instructions; GM included 11% aged 90+. The linear age term was fitted in ≥65s. Use outside that range (e.g. <65) needs recalibration or an age update (Sci Rep §4.6.3).
8. **Operations.** Running centrally in a data environment and sending lists to primary care "worked well" in GM. Attrition, data lags, referral tracking and SDE access were the bottlenecks.

---

## 7. Unresolved
- The exact coefficients, polypharmacy window, BMI rules, codelist version and exclusion rules used in the GM SDE run. Ask NHS GM data team / GMCA (report contact: Bethany Badrock, GMCA) or Leeds for the GM search specification.
- The "System exclusion criteria applied*" definition (N=273) in ARC-GM Figure 3.
- Whether the Feb 2026 change to a 120-day polypharmacy window was ever published; the current instructions were deleted from GitHub. Ask K. Best / A. Clegg for the current version (reportedly ≥1.13) and the BNF chapter table.
- The Sci Rep recalibration α/β (implied here, not reported), the polypharmacy definition, BMI cut-points and outcome code source. Contact T.Chen@hud.ac.uk.
- ARC Wessex project page (moved or 404). Search-summary claims (CIs Marino and Lim; Apr 2025–Mar 2026; data from West Yorkshire and Hampshire & Isle of Wight) are unverified.
- The HTA narrative "being female … higher predicted risk" vs Table S3.2 sign. Only the authors can settle this.
