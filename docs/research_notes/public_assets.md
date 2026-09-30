# eFalls: public assets, corrections, access routes, licensing and later literature

Research notes compiled 2026-09-14 (all web checks run on that date).

Evidence labels:
- **[EXPLICIT]**: stated in the cited source. Quotes are verbatim and 40 words or fewer.
- **[INFERRED]**: my reasoning from the evidence; the reasoning is given.
- **[NOT FOUND]**: searched for and not located; the searches are listed.

Path abbreviations:
- `SP` = `<research workspace (not distributed)>`
- `R` = `SP/research/raw`
- `W` = `SP/research/work/web` (files fetched in this round)

No downloaded file was executed. Python and R scripts were saved as `.txt` and only read. PDFs were converted with `pdftotext` or `pdftoppm`.

---

## 0. Headline findings

1. **No erratum or corrigendum exists** for doi:10.1093/ageing/afae057 or for the HTA monograph doi:10.3310/GJAC1008. Checked: Crossref `updates` filter, Crossref `update-to`/`updated-by`/`relation`, PubMed CommentsCorrections, Europe PMC `commentCorrectionList` and "Published Erratum" type, the OUP article page, the NIHR Journals Library page, and a web search. [NOT FOUND]
2. **PMID 42574037 is the HTA monograph**: Archer L ... Riley RD, Clegg A. *Health Technol Assess* 2026 Aug;30(61):1-154, doi:10.3310/GJAC1008, PMC13478932. [EXPLICIT]
3. **The official implementation instructions were public, but only in GitHub history.** The repo `KateBest/eFI2plus` ("MHRA documentation for the eFI2+") held "Intended use and implementation instructions for the eFI2 and eFI+" (v1.9, v1.10, v1.13).
   - These files were **deleted from HEAD on 2026-02-26**.
   - Git blob hashes prove the local copies in `R/efi2plus/` are byte-identical to those commits.
   - Table 4 gives the eFalls coefficients with **intercept −5.954459, Sex–Male blank (reference), Sex–Female −0.303708**. That matches Table S3.2 and HTA Table 15, not Box S3.1's "−0.304 (if male)".
   - It gives **standard BMI cut-offs (obese ≥30)**, not "obese ≥40".
   - It lists **Alcohol "Lower risk" = 0 (reference)**, which supports Table S3.2 over HTA Table 15's duplicated "Higher-risk drinking".
   - Its worked eFalls example has an arithmetic error that was never corrected: stated L = −3.023169, but the listed terms sum to −2.978465. [EXPLICIT / computed]
4. **No official eFalls analysis code, calculator or model object is public.**
   - The Age & Ageing TRIPOD-Cluster item 18 (analysis code) says "Not yet available. To be included at publication." Nothing was included.
   - The Connected Bradford repo `ConnectedBradford/CB_1551_eFI2` ("…eFI2, eFalls Predictor") contains only template READMEs.
   - No eFalls items were found on Zenodo, figshare, OSF or GitLab. [EXPLICIT / NOT FOUND]
5. **Public author code does touch eFalls.** `sdrelton/PREDICT` (Samuel Relton, BSD-3, NIHR206843) contains `Comparison/efalls.py` and `refit_efalls_model.py`. These apply drift-detection and recalibration to eFalls in Connected Bradford (project CB_2151).
   - The official coefficients are read from a non-public `efalls_coefs_official.json`.
   - The code computes polypharmacy as `np.log(unique_bnf_last_3_months + 1) / 10`, i.e. ln(P+1)/10 over about 3 months. The publication uses ln((P+1)/10) over 120 days.
   - This is research code, not a reference implementation. [EXPLICIT]
6. **Patient-level data are not public.** SAIL requires IGRP approval and remote access through the SAIL Gateway TRE. Connected Bradford/Connected West Yorkshire requires an application and board approval, with access in a secure virtual environment and a fee. ClinicalTrials.gov NCT04113174 states IPD sharing "NO". [EXPLICIT]
7. **Licensing terms:**
   - The equation is "available for research use".
   - Code lists are on "reasonable request" from the corresponding author (Andrew Clegg, a.p.clegg@leeds.ac.uk).
   - EHR suppliers, risk-stratification software and NHS commissioning need "an agreed licence agreement".
   - "Any unauthorised use or distribution for commercial purposes is forbidden."
   - The HTA adds a possible "eFI+ revenue share distribution agreement".
   - Both articles are CC-BY 4.0, which covers the text, while these statements restrict use of the tool. [EXPLICIT; the tension is INFERRED]
8. **Later literature (2024–2026):**
   - One independent-population external validation: Chen et al., *Sci Rep* 2026, working-age mental health and learning disability services (**outside the intended ≥65 population**). The model **under**-predicted there (CITL 1.36, O/E 1.60, slope 1.22) before recalibration.
   - One real-world implementation pilot: Greater Manchester / Wigan SWAN PCN, ARC-GM report March 2026, 10–25% risk band, not an effectiveness evaluation.
   - NICE NG249 Evidence Review D (April 2025) included eFalls and made no recommendation.
   - The HTA monograph itself (Aug 2026) says the recalibrated model was "tested within Greater Manchester" by the ICB.
   - None of these reports a coefficient, sex-term or BMI error. The discrepancies documented here come from comparing the sources. [EXPLICIT / INFERRED]

---

## 1. Errata and corrigenda

### 1.1 Age & Ageing article doi:10.1093/ageing/afae057 (PMID 38520142, PMC10960070)

| Check | Evidence | Result |
|---|---|---|
| Crossref work record (prior round) | `R/crossref_work.json`: `update-to` null, `updated-by` null, `relation` {}, `update-policy` null; indexed 2026-08-17; deposited 2024-03-23 | no update [EXPLICIT] |
| Crossref work record (re-fetched 2026-09-14) | `W/cr_work_afae057_now.json`: same values; is-referenced-by-count 12 | no update [EXPLICIT] |
| Crossref `filter=updates:10.1093/ageing/afae057` | `R/crossref_updates.json` and `W/cr_updates_afae057.json`: total-results 0 | no correction record [EXPLICIT] |
| Crossref bibliographic query "Correction to Development and external validation of the eFalls tool" | `W/cr_q_corr_efalls.json`: top 10 contain only corrections to *other* articles, plus afae057 itself | none [NOT FOUND] |
| Crossref query "eFalls" | `R/crossref_q_efalls.json` and `W/cr_q_efalls_now.json`: 1 result (afae057) | none |
| Crossref Age & Ageing fall/correction search (prior round) | `R/crossref_q_aa.json`: the only "Correction to:" items are for BONE PARK 2, Editor's view, BGS assisted dying, and blood-sugar monitoring | none for eFalls [EXPLICIT] |
| PubMed record (prior round) | `R/pubmed_38520142.xml`: no `CommentsCorrectionsList`; DateRevised 2024-07-01 | no linked erratum [EXPLICIT] |
| PubMed record (re-fetched) | `W/pubmed_now.xml`: DateRevised 2024-07-01; CommentsCorrections None | no linked erratum [EXPLICIT] |
| PubMed esearch `erratum AND Age Ageing[ta] AND eFalls` | count 0 | none |
| Europe PMC core record | `W/epmc_core_38520142.json`: `commentCorrectionList` None; dateOfRevision 2024-07-01 | none [EXPLICIT] |
| Europe PMC `"eFalls" AND PUB_TYPE:"Published Erratum"` | hitCount 0 (`W/epmc_eFallsANDPUB_TYPEPublishedErratum.json`) | none |
| OpenAlex | `R/openalex_work.json`: `is_retracted` false; updated 2026-09-06 | not retracted [EXPLICIT] |
| OUP article page | `R/oup_article.html` is a Cloudflare challenge page with no content. A live WebFetch of https://academic.oup.com/ageing/article/53/3/afae057/7633682 on 2026-09-14 found no correction notice | none [EXPLICIT per fetch summary] |
| Web search `"Correction to" "eFalls tool" Age and Ageing` | returned only the article, the Editor's view afae083 and Yang 2025 | none |
| White Rose repository copy | `R/wr_219096.html`: "Last Modified: 01 Nov 2024", published version, no correction noted | none |

**Conclusion (1.1): NOT FOUND.** No erratum, corrigendum, expression of concern or retraction exists for afae057 as of 2026-09-14.

Consequence [INFERRED]: none of the internal inconsistencies has been formally corrected. These are the Box S3.1 sex term, the BMI "obese ≥40", and the Box S3.2 α/β versus HTA Table 35. The HTA monograph (Aug 2026) repeats the Box S3.1 text verbatim (see 1.3).

### 1.2 HTA monograph doi:10.3310/GJAC1008 (PMID 42574037, PMC13478932)

| Check | Evidence | Result |
|---|---|---|
| Crossref work | `W/cr_work_gjac1008.json`: update-to/updated-by null; relation {}; update-policy https://doi.org/10.3310/crossmarkpolicy (Crossmark enabled); deposited 2026-08-10 | no update |
| Crossref `filter=updates:10.3310/gjac1008` | `W/cr_updates_gjac1008.json`: total 0 | none |
| PubMed | `R/pubmed_42574037.xml` and `W/pubmed_now.xml`: no CommentsCorrections; DateRevised 2026-08-19 | none |
| Europe PMC core | `W/epmc_core_42574037.json`: commentCorrectionList None | none |
| NIHR Journals Library page | `R/gjac1008_struct.txt` lines 119-120 ("Published: August 2026"). Toolkit lines 125-150 list only Scientific Summary, Disclosure of Interest, Publication PDF, Plain Language Summary and JATS XML, with "View responses to this publication (0)". Live curl 2026-09-14 (`W/hta_now.html`, HTTP 200): no "erratum"; the only "correction" strings are "HKSJ variance correction" | none |

**Conclusion (1.2): NOT FOUND.** No erratum exists for the HTA monograph.

### 1.3 Unresolved inconsistencies that a correction would need to address (for context)

- **Box S3.1 LP** says "−6.258 … – 0.304 (if male)" (`SP/efalls/supp.md` line 318). The same text appears in HTA Table 16 (`R/gjac1008_struct.txt` line 1051) and HTA Table 34 (line 2075).
  - Table S3.2 (`SP/efalls/table_s3_2.csv`) and HTA Table 15 (`R/hta_table15_efalls_coefficients.csv`) say Male = Reference, Female = −0.303708, Constant = −5.954459.
  - The **official implementation instructions** (section 4.3) side with Table S3.2.
  - −5.954459 − 0.303708 = −6.258167. So Box S3.1 is the female-baseline intercept; the male term should read "+0.304 (if male)". [INFERRED]
- **BMI definitions:**
  - Box S3.1 and S3.2 (`SP/efalls/supp.md` lines 318 and 368) and HTA Table 34 (`R/gjac1008_struct.txt` line 2076): "obese if BMI ≥ 40".
  - HTA Table 16 (line 1052): "weight groups are defined by standard BMI cut-offs".
  - Implementation instructions v1.13 (`R/efi2plus/instr_v1.13.txt` line 108): "obese (≥30kg/m2)", overweight 25.0–29.9. [EXPLICIT]
- **Alcohol reference category:** HTA Table 15 prints "Higher-risk drinking | Reference" (`R/gjac1008_struct.txt` lines 971-972). The same duplication appears in the HTA descriptive tables (lines 598-599, 720-721, 932-933), where the second "Higher-risk drinking" row has SAIL n = 11,231.
  - 11,231 is the eFI2/eFalls SAIL **lower-risk** count (`SP/research/efi_codelists.md` section 5).
  - Instructions v1.13 line 434-435: "Alcohol - Lower risk | 0".
  - So HTA Table 15 is a typesetting error. [INFERRED, strong]
- **Recalibration:** HTA Table 35 (`R/hta_table35_recalibrated_coefficients.csv`) is exactly 1.21 × the Table S3.2 coefficients, with constant −7.25089539. That implies α ≈ −0.046 and β = 1.21 on the Table S3.2 LP (computed).
  - For Box S3.2's example patient, Table 35 gives p = 0.148.
  - Box S3.2 and HTA Table 34 give 0.106 (α −0.423, β 1.25).
  - Not reconciled in any source. [EXPLICIT numbers; computed]

---

## 2. Code repositories, calculators and scripts

### 2.1 Author and institution repositories found

| Repository | Owner / licence / dates | Content relevant to eFalls | Label |
|---|---|---|---|
| https://github.com/KateBest/eFI2plus | Kate Best (Leeds); **no LICENSE file**; created 2023-04-27; last push 2026-02-26 | MHRA technical-file documents: Compliance with essential requirements, Hazard Log, Risk management plan, eFI2/+ Change Log. Implementation instructions v1.9, v1.10 and v1.13 are **only in commit history** (details in section 4.3) | EXPLICIT |
| https://github.com/ConnectedBradford/CB_1551_eFI2 | Connected Bradford org; MIT; created 2023-09-26; last push 2024-05-10; 2 commits | README title "CB_1551 - Development and national implementation of eFI2, eFalls Predictor" plus news links. `code/README.md` "Contents: <empty>"; `docs/datadict/README.md` empty. **No code** | EXPLICIT (GitHub API tree, 2026-09-14; local `R/cb1551_readme.md`) |
| https://github.com/ConnectedBradford/CB_ResearchInformation | Connected Bradford | README project list; row 1551 links to CB_1551_eFI2, lead "Andy Clegg"; row 2151 "PREDICT: Pragmatic Recalibration and Evaluation of Drift in Clinical Tools", lead Samuel Relton (`R/cb_researchinfo_readme.md` lines 103, 134) | EXPLICIT |
| https://github.com/sdrelton/PREDICT | Samuel Relton; BSD-3-Clause; created 2025-01-14; last push 2026-03-06; Zenodo DOI 10.5281/zenodo.15114705; funded by NIHR206843 (README) | `Comparison/efalls.py` (24,395 B), `Comparison/refit_efalls_model.py` (9,970 B), `Comparison/input_data_drift_analysis.py`, `Comparison/qrisk_efalls_demographics`. Local copies as .txt in `R/predict/` (HEAD sha in `R/predict/HEAD_SHA.txt`) | EXPLICIT |
| https://github.com/ConnectedBradford/CB_2151 | Connected Bradford (description: PREDICT) | Not inspected in detail (companion of PREDICT) | – |
| https://github.com/DynAIRx/GCAF_DynAIRx | DynAIRx; BSD-3 | Public eFI2/eFI+ SNOMED "Baseline2" code list. Already documented in `SP/research/efi_codelists.md` section 3 (not redone) | EXPLICIT (prior notes) |

**PREDICT eFalls code details** [EXPLICIT, `R/predict/efalls.py.txt`, `R/predict/refit_efalls_model.py.txt`]:
- The data source is a Connected Bradford SQL Server database `CB_2151`, table `tbl_final_efalls_deduped` (efalls.py lines 33-47). It is not public.
- Line 54: `df["Polypharmacy"] = np.log(df["unique_bnf_last_3_months"] + 1) / 10`.
  - By Python operator precedence this is **ln(P+1)/10**.
  - The published term is **ln((P+1)/10)** over **120 days** (Box S3.1, `SP/efalls/supp.md` line 318; Sci Rep supplement Table S1 `W/scirep_esm.txt` line 10). [discrepancy EXPLICIT; its effect on their results unknown]
- Predictors include "Female" (so male is the reference) and 75 named predictors (lines 56-67).
- Official coefficients are loaded from `results/efalls/efalls_coefs_official.json` (refit lines 136-143). That file is **not in the repo**.
- Recalibration: `coefs = coefs_official × slope`; `Intercept = α + official_intercept × slope` (refit lines 144-151). This is the same form that reproduces HTA Table 35's "×1.21" structure. [INFERRED link]
- Commit messages (`R/predict/efalls_hist.tsv`, `refit_hist.tsv`):
  - 2026-02-02 "something wrong with the model application (massive CITL value at start of analysis)"
  - 2026-02-03 "some issue with getting fall events leads to underprediction"
- PREDICT newsletters say they aim to "replicate QRISK2 and eFalls models" (Sept 2025; `R/predict/PREDICT_Newsletter_September2025.txt` lines 56-58).

### 2.2 Third-party implementations (not eFalls)

- OHDSI/OmopIndices `R/addElectronicFrailtyIndex2.R` (Apache-2.0) implements **eFI2** from Best et al. 2025 (https://github.com/OHDSI/OmopIndices). It has no eFalls (GitHub code search `efalls repo:OHDSI/OmopIndices` = 0).
- wnl-icb-analytics/dbt-analytics (NHS North West London ICB, public) contains eFI2 scoring models (`int_efi2_scores.sql`). No eFalls (code search = 0).

### 2.3 Searches with no eFalls asset [NOT FOUND]

- **GitHub repository search** for eFI2, efalls, eFall, "electronic frailty index", "frailty index 2": the only relevant hits are those in 2.1.
- **GitHub code search:**
  - `"eFalls" language:Python` found only sdrelton/PREDICT.
  - `"eFalls" language:R` and `language:SQL`: no relevant hits.
  - `"Polypharmacy + 1)/10"`: 0 hits.
  - Numeric coefficient strings: tokeniser noise only.
- **GitHub user checks:** KateBest (2 repos: eFI2plus, opensafely-getting-started); sdrelton (PREDICT only); ZoeHancox (TG-CNN); AshleyAkbari (CCU002_01, healthyr). No accounts found for Lucinda Archer, Miriam Hattle or Joe Hollinghurst.
- **GitLab** (`gitlab.com/api/v4/projects?search=efalls`): 0.
- **Zenodo** (`q=eFalls`): 0. "electronic frailty index" returned only unrelated posters and Trip answers. "eFI2" returned only "Validated Mental Health Readcodes" (Bazo Alvarez, 10.5281/zenodo.7272058; eFI2 co-author, not eFalls).
- **figshare** (`/v2/articles/search "eFalls"`): only fuzzy unrelated matches (EFAL, zeolite). "electronic frailty index 2": 0.
- **OSF / SHARE** (`share.osf.io … q=eFalls`): 0 hits.
- **Age & Ageing supplement:** `SP/efalls/supp.zip` contains one DOCX (`aa-23-2211-file002_afae057.docx`) and figure images. No code or data. TRIPOD-Cluster item 18 (`SP/efalls/supp.md` line 59): "Not yet available. To be included at publication." [EXPLICIT]
- **Software stated:** Stata 17 for development and internal validation; R 4.2.3 for external validation (`SP/efalls/fulltext.md` line 106). HTA: recalibration "fit using the glm command in R (version 4.2.3)" (`R/gjac1008_struct.txt` line 1153). Scripts not released.
- **Calculators:** no public web or online eFalls calculator found. The Leeds news release (9 May 2024, https://www.leeds.ac.uk/news-science/news/article/5570/using-nhs-data-to-predict-patients-risk-of-falling) quotes "Our eFalls calculator…" but gives no public tool. ARC-GM Box 2 has the same quote (`SP/research/work/arcgm_p-12.png`).
- **Generic methods software** used in this area is public but not eFalls-specific: pmsampsize, pmvalsampsize, pmcalplot (`R/pmsampsize.ado` etc.).

### 2.4 What PMID 42574037 is [EXPLICIT]

`R/pubmed_42574037.xml`:
- Archer L, Relton SD, Akbari A, Best K, Bestwick R, Bucknall M, Conroy S, Hattle M, Hollinghurst J, Howdon D, Hulme C, Humphrey S, Lyons RA, Nikolova S, Rodriguez MP, Richards S, Walters K, West R, Van der Windt D, Riley RD, Clegg A.
- "Development and evaluation of the electronic frailty index+ (eFI+) tool for older people: prognostic prediction modelling with integrated decision curve and health economic analysis." *Health Technol Assess* 2026 Aug;30(61):1-154. doi:10.3310/GJAC1008. PMC13478932.
- Status MEDLINE; publication types Journal Article and Observational Study.
- Abstract, falls model on external validation: "calibration slope 1.25 … calibration-in-the-large −0.931 … observed/expected ratio 0.43 … C-statistic 0.83".
  - These are whole-population values, matching A&A Table 2 (`SP/efalls/fulltext.md` lines 203, 219, 235, 251).
  - The A&A abstract's CITL −0.87 is the practice-pooled estimate (line 219: −0.874). Not a contradiction. [INFERRED]
- Registered on ClinicalTrials.gov as NCT04113174. `R/ctgov_NCT04113174.json` lists it as a DERIVED reference.
- Article history (`R/gjac1008_struct.txt` line 297): editorial review began April 2025; accepted November 2025.

---

## 3. Raw data access (patient-level data not public)

| Source | Access route | Evidence |
|---|---|---|
| SAIL Databank (Wales; development cohort) | Two stages: (1) scoping with SAIL, funding, Safe Researcher Training; (2) online **IGRP application**, plus data-owner approvals where needed. About 12 weeks. Analysis only by **remote access to SAIL Gateway (TRE)** | https://saildatabank.com/data/apply-to-work-with-the-data/ (WebFetch 2026-09-14). HTA Table 1: "Data are made available following an application to and independent assessment by the SAIL Information Governance Review Panel" … "Remote secure access" (`R/gjac1008_struct.txt` lines 334-341). A&A: SAIL use "approved by an independent Information Review Governance Panel" (`SP/efalls/fulltext.md` line 138). Small cells <10 suppressed "due to SAIL Databank restrictions" (line 187) |
| Connected Bradford (now presented as Connected West Yorkshire; validation cohort) | Expression of Interest form (CWY_EoI_v1.2), then the BIHR research data access process; Local Health System board meets **bi-monthly**; data in "a Secure Virtual Environment"; "There is a charge for supporting data access"; REC + CAG (s251) approvals; contact cWestYorkshire@bthft.nhs.uk | https://bradfordresearch.nhs.uk/connected-bradford/governance-and-ethics/ (WebFetch 2026-09-14). HTA Table 1: "application to and independent assessment by the Connected Bradford Review Panel", "Remote secure access" (`R/gjac1008_struct.txt` line 345). Ethics: "REC 18/YH/0200 and 22/EM/0127" (line 1789). Project enquiries: "email cbradford@bthft.nhs.uk quoting the Project ID" (`R/cb_researchinfo_readme.md` line 20); eFalls/eFI2 project ID **1551**; PREDICT **2151** |
| CARE75+ (HTA additional predictors only) | CARE75+ Data Request Review Committee | `R/gjac1008_struct.txt` Table 1 |
| Trial registry | NCT04113174 `ipdSharingStatementModule.ipdSharing` = **"NO"** | `R/ctgov_NCT04113174.json` |

**Conclusion:** no patient-level SAIL or Connected Bradford data are public. [EXPLICIT]
- The only public data are aggregate: Table 1/S3.1 counts, coefficients, performance tables, and ICD-10 outcome codes (A&A Table S2.3 at `SP/efalls/supp.md` line 86; HTA Appendix 1 at `R/gjac1008_struct.txt` lines 1884-1920).
- The Sci Rep validation data are "available on reasonable request" (`W/scirep_ref.txt` line 1410).

---

## 4. Licensing, official pages and regulatory documents

### 4.1 Licensing statements [EXPLICIT]

- **Age & Ageing data availability** (`SP/efalls/fulltext.md` lines 366-367; confirmed on the live OUP page):
  - "The eFalls model equation as published in this manuscript is available for research use. Code lists used to define variables are available on reasonable request from the corresponding author."
  - Suppliers and NHS use: "under the terms of an agreed licence agreement."
  - "Any unauthorised use or distribution for commercial purposes is forbidden."
- **HTA data-sharing statement** (`R/gjac1008_struct.txt` lines 1785-1787):
  - Same research-use and code-list wording, and "Any unauthorised use or distribution for commercial purposes is forbidden."
  - Adds: "This could include implementation of an eFI+ revenue share distribution agreement, in the event of future commercialisation."
  - Line 412: "The full list of predictor variables and their corresponding codes is available on request from the corresponding author."
- **Corresponding author:** Andrew Clegg, a.p.clegg@leeds.ac.uk (`SP/efalls/fulltext.xml` line 23; HTA line 108).
- **Article licences:**
  - A&A article CC-BY 4.0 (`R/crossref_work.json` license; `R/wr_219096.html`).
  - HTA CC-BY 4.0 (`R/gjac1008_struct.txt` line 300).
  - [INFERRED] CC-BY covers the text and tables. The authors' statements add use restrictions on the tool for commercial and service deployment. Obtain written confirmation from the University of Leeds and A. Clegg before any Meuhedet service deployment; research reproduction appears to fit "research use".
- **Precedents from the same group** (for interpreting "licence"):
  - Original eFI: "Copyright and database rights for the eFI are held by the University of Leeds", freely licensed to EHR providers on condition of no premium (`R/efi/efi2016.txt` line 225).
  - eFI2: "licensed to suppliers of UK primary care electronic health record systems at no cost, on the basis that a premium charge is not subsequently applied to the end user" (`R/efi/PMC12117642.txt` line 23).
  - eFI2 paper: "Organisations wishing to use eFI2 for commercial purposes should contact the corresponding author directly" (`R/efi/efi2.txt` line 343).
  - **No eFalls-specific licence text, fee schedule or licence portal was found.** [NOT FOUND]

### 4.2 Official and quasi-official eFalls web pages

| Page | Content | Label |
|---|---|---|
| University of Leeds news, 9 May 2024: https://www.leeds.ac.uk/news-science/news/article/5570/using-nhs-data-to-predict-patients-risk-of-falling (also medicinehealth.leeds.ac.uk/.../684/...) | Describes eFalls; hopes for integration into UK primary care EHR systems; no tool, licence or code | EXPLICIT (WebFetch) |
| BBC news article linked from the CB_1551 README: https://www.bbc.co.uk/news/articles/c6py1x3l090o | not fetched | – |
| GM Integrated Care Partnership news: https://gmintegratedcare.org.uk/health-news/world-first-efalls-tech/ | Pilot launched Feb 2025 for 9 months; searches run by the NHS GM data team; "moderate risk"; £100,000 from OHID and Centre for Ageing Better; aim for the toolkit "to be fully embedded in GP IT systems" | EXPLICIT (WebFetch) |
| ARC-GM project page: https://arc-gm.nihr.ac.uk/projects/Case-Falls-Prevention-Pilot-Evaluation; report PDF https://arc-gm.nihr.ac.uk/media/Resources/ARC/Healthy%20Ageing/EFalls/Case%20Finding%20for%20Falls%20Prevention%20eFalls%20pilot%20study%20-%20FINAL%20%20evaluation%20report.pdf | Final report March 2026; 10–25% risk band; feasibility and acceptability evaluation | EXPLICIT |
| ARC Wessex project page: https://www.arc-wx.nihr.ac.uk/research-areas-list/refinement-of-an-efalls-tool---a-multivariable-prediction-model-for-the-risk-of-ed-attendance-or-in-hospital-fall-or-fracture-in-individuals-accessing-mental-health-or-learning-disability-services---efalls | "Refinement of an eFalls tool … mental health or learning disability services". The local `R/arcwx_efalls.html` is a 404 from an old URL; live WebFetch failed (TLS certificate error); title known from a search snippet only | EXPLICIT (title only) |
| ARC YH eFI2 project: https://arc-yh.nihr.ac.uk/research/projects/development-and-national-implementation-of-efi-2/ | eFI2 only (from efi_codelists.md) | – |
| NIHR award: https://fundingawards.nihr.ac.uk/award/NIHR127905 | £545,625.65; 2019-10-01 to 2023-02-28; contracted organisation University of Leeds; status Complete (`R/nihr_open.json`) | EXPLICIT |

No dedicated official eFalls website, calculator, licence page or download page was found. [NOT FOUND]

### 4.3 eFI2/eFI+ medical-device and regulatory documents (KateBest/eFI2plus)

**Provenance** [EXPLICIT; GitHub API, 2026-09-14]:
- Repo https://github.com/KateBest/eFI2plus, description "MHRA documentation for the eFI2+", no licence file.
- The current tree has 4 files: Compliance with essential requirements.docx (blob 0ac5c74b1d49), Hazard Log.docx (64f2a4766e32), Risk management plan.docx (926a45f9c086), eFI2_+ Change Log.docx (4cb936d9462e).
- Implementation-instruction history:
  - v1.9 added in commit 834021d9ab (2025-06-17), blob 053eb325bcf8; removed in commit 87084eeb74 (2026-02-26).
  - v1.13 "Github copy" added in commit 233b23f3e3 (2026-02-26 12:12, "Updated to fix an error in the example calculation for the eFI2 risk predictio[n]"), blob 4eff21ce3803; removed in commit 97d388f97c (12:17).
  - v1.10 "Github copy" added in commit 5e0a4856c4 (12:20, "Updated to fix a mistake in the example calculation of the eFI2 predicted risk"), blob 0d82fed41d39; removed in commit 73df6ce837 (12:58).
- Local files `R/efi2plus/instr_v1.9.docx`, `instr_v1.10.docx`, `instr_v1.13.docx` and the 4 HEAD docs have `git hash-object` values equal to these blobs, so they are byte-identical.
- Example history URL: https://github.com/KateBest/eFI2plus/tree/5e0a4856c476e3c7d9a3f954e85d5eb8a84bc5f0
- DOCX metadata: instructions authored by Kate Best; the v1.10/1.13 files were created 2026-02-26; Compliance and Risk plan last modified 2023-04-27; Hazard Log 2022-12-06; Change Log created by Samuel Relton 2024-03-21 and modified 2025-06-13.
- [INFERRED] The instructions were withdrawn from the public HEAD the same day they were corrected. No replacement location was found (web search for the document title returned nothing). Treat the history copies as public but superseded or withdrawn; their status must be confirmed with the authors.

**What the documents say about eFalls:**
- **Change Log** (`R/efi2plus/eFI2_+ Change Log.txt`):
  - Line 12 (25 Feb 2024): "No changes to codelist. Model equations for eFalls updated in the implementation instructions (version 1.7) to match final outputs from SAIL (and publication in Age & Ageing)."
  - Line 4 (13 June 2025): v1.9 added references to the eFI2 and eFalls papers, and "the alogrithms for eFI+ models for morality, care home admission and new home care were removed…".
  - Line 6 (1 April 2025): a SNOMED code was removed from Severe Mental Illness (codelist SNOMED version 0.27).
  - Lines 13-21 (4 Aug 2023): alcohol units/day × 7; worst category over 5 years; BP rules.
- **Implementation instructions v1.9** (`R/efi2plus/instr_v1.9.txt`):
  - Title "Intended use and implementation instructions for the eFI2 and eFalls" (line 2).
  - "In these instructions, only the eFI+ falls alogorithm (known as eFalls) is provided." (line 42)
  - Table 4 "eFI+ falls model (eFalls) coefficients" (line 366).
- **v1.10 and v1.13** (identical text; `diff` empty): title "…eFI2 and eFI+" (line 2).
  - Intended use of eFI+: to identify people at increased risk "so that they can be identified for consideration for particular interventions"; "Pre-specified thresholds for low, medium and high risk are not set" (line 41).
  - Target population: "Falls model – aged 65 years or older" (line 154).
  - Code list: "7556 unique SNOMED CT codes (or equivalent CTV3 or Read 2 codes) … organised into 79 health deficit variables" (line 44). v1.9 says 7555 (line 45).
  - Polypharmacy: "number of medications from different BNF 2017-18 sub-sub-chapters prescribed in the previous 90 days … Note that the eFI+ used polypharmacy as a continuous numeric variable" (line 112). **This conflicts with eFalls Box S3.1's 120 days.**
  - BMI: "underweight (<18.5kg/m2), recommended weight (18.5-24.9kg/m2), overweight (<25.0-29.9-kg/m2), obese (≥30kg/m2)"; the most recent value is used (line 108).
  - Smoking: "Ex, Current, None/missing" (line 110). Alcohol: highest category over the last 5 years (lines 101-102).
  - Model form: "Each model is a logistic regression … s(x) = 1/(1+exp(-x))" (line 357).
  - **Table 4** (lines 364-661): Intercept −5.954459; Age 0.0415506; ln((Polypharmacy + 1)/10) 0.3296295; **Sex – Male blank; Sex - Female −0.303708**; BMI Underweight 0.4896735, Normal 0.2394177, **Overweight 0**, Obese −0.0411134, Missing −0.1451981; Smoking Never and Ex blank, Current 0.0684529; Alcohol Harmful 0.4164064, Higher risk 0.1549725, **Lower risk 0**, Previous higher/harmful 0.0849676, Zero 0.0070124, Missing −0.0679367; plus the binary predictors.
  - All 77 non-blank values match Table S3.2 exactly (computed; blanks = LASSO-omitted variables and reference levels).
  - **Worked example** (lines 358-362): "female individual aged 70 with 5 medications, who is underweight, never smoked, a lower risk drinker, and has back pain"; L = −5.954459 + 70×0.0415506 + 0.3296295×ln((5+1)/10) −0.303708 + 0.4896735 + 0.0498699; "L = -3.023169"; "P = … = 0.0464".
    - Recomputed, the listed terms sum to **−2.978465 (P = 0.0484)**.
    - The stated value is off by −0.0447, and no single coefficient or alternative transform explains it (brute-force check).
    - The same error is in v1.9, v1.10 and v1.13.
    - The Feb 2026 fixes addressed only the **eFI2** example (1-year baseline hazard 0.0202 → 0.0151; v1.9 lines 168-170 vs v1.13 lines 169-171).
    - [EXPLICIT text; arithmetic computed]
- **Compliance with essential requirements** (`R/efi2plus/Compliance with essential requirements.txt`):
  - Line 3: the document shows the eFI2 "and incorporated eFI+ meet the relevant essential requirements outlined in Part II of the UK MDR 2002, Annex I".
  - Line 194: the eFI2 "predict[s] the possibility of a patient being admitted into hospital with a fall/fracture, requiring increased home care package, requiring nursing home admission or all-cause morality".
  - Instructions "will be made available on the eFI2 GitHub" (lines 13, 248-284).
  - Line 28: "Refer to Risk Management Plan in MHRA Technical Documentation folder".
  - Implementers named: TPP, EMIS, Cerner (lines 48, 153).
  - eFalls is not named separately; it falls under "incorporated eFI+". [EXPLICIT]
- **Hazard Log** (`R/efi2plus/Hazard Log.txt`), 5 hazards:
  1. Misclassification of individual patient risk (30/11/22): the tools "are population risk stratification tools. They are not able to identify individual risk" (line 33).
  2. Missing SNOMED codes.
  3. Incorrect SNOMED codes.
  4. Coding-system updates (Moderate → Low after mitigation).
  5. "User eye strain" (dated 08/07/2022).
  - No falls-specific hazards. The column instruction refers to "an ACMI repository Issue" (line 8), a template carry-over from the ACMI project. [EXPLICIT; carry-over INFERRED]
- **Risk management plan** (`R/efi2plus/Risk management plan.txt` lines 8-18): the issue route is email to a.p.clegg@leeds.ac.uk or a GitHub Issue. The repo has **0 issues** (API, 2026-09-14).
- **Regulatory status:** no MHRA registration, UKCA/CE marking or DCB0129/DCB0160 record for eFI2+ or eFalls was found. Web search found only an EMIS support page for eFI2 (KB5003574), which was not read. The compliance document is a self-assessment against UK MDR 2002 essential requirements (Class I-style). [NOT FOUND / INFERRED]

---

## 5. Subsequent publications and reports, 2024–2026

Sources used to build the list:
- Europe PMC `CITES:38520142_MED` (10 hits), `REF:"afae057"` (7) and `"eFalls"` (6, 3 relevant) (`R/epmc_citations.json`, `W/epmc_*.json`)
- OpenAlex citing works (13; `R/openalex_citing.json`) and fulltext search (`R/openalex_search.json`)
- PubMed esearch
- Crossref reference checks
- Web search

| # | Publication (DOI/URL) | Type | One-line relevance | Coefficient / sex / BMI issues reported? |
|---|---|---|---|---|
| 1 | Archer L … Clegg A. *Health Technol Assess* 2026;30(61). doi:10.3310/GJAC1008; PMID 42574037; PMC13478932 | Companion monograph (Aug 2026) | Full eFI+ report. Falls chapter reproduces the A&A tables. Adds **Table 35 recalibrated coefficients**. States the "adjusted model has been subsequently tested within Greater Manchester by members of their Integrated Care Board (ICB) and has been found to generalise well" (`R/gjac1008_struct.txt` line 1156). eFalls deployed into the GM Care Record (2.8M residents) with predictions calculated by the GM ICB BI Unit (line 1743). Recommends "use of the externally validated eFalls model" and optional local recalibration (line 1181) | None reported. It repeats Box S3.1 "–0.304 (if male)" (lines 1051, 2075), mixes "standard BMI cut-offs" (1052) with "obese if BMI ≥ 40" (2076), and duplicates "Higher-risk drinking" (971-972) [EXPLICIT; issues INFERRED] |
| 2 | Chen T, Marino LV, Best K, Bhatnagar-Knox S, Humble V, Garnham M, Greenbank K, Relton S, Lim S, Clegg A. "Extending eFall risk prediction to working-age adults within mental health and learning disability services: a clinical validation study." *Sci Rep* 2026 (published 10 Aug 2026, accepted-manuscript preview). doi:10.1038/s41598-026-51298-0 | External validation + recalibration | Setting: South West Yorkshire Partnership NHS FT, ages 18–65, n = 32,410, fall rate 2.07%. **C = 0.777; CITL 1.357 (under-prediction); O/E 1.604; slope 1.222**. After recalibration CITL −0.001, slope 1.000. Worse in learning disability (AUC 0.696–0.739). Coefficients "applied unchanged" (`W/scirep_ref.txt` lines 121, 314, 347-348, 607-615). Code mapping "strictly followed the specifications provided in the original eFalls study" (line 222-226; lists presumably obtained from co-authors, INFERRED). Recalibration α/β values not reported (NOT FOUND). Data "available on reasonable request" (line 1410). HRA/HCRW 24/PR/1367. Funding NIHR ARC Wessex | Supplementary Table S1 (`W/scirep_esm.txt`) reproduces Table S3.2 with **Male = Reference, Female −0.303708**, Constant −5.954459, ln((Medication count + 1)/10). No sex, BMI or coefficient problem reported. BMI cut-offs not stated (NOT FOUND). **Used outside the intended ≥65 population** |
| 3 | Money A, Badrock B, Eost-Telling C, Christie R, Vallabh N, Reynolds E, West T, Vardy E, Davies S, Clegg A, Todd C. *Case Finding for Falls Prevention – 'eFalls' pilot study evaluation report.* NIHR ARC Greater Manchester, March 2026. URL in section 4.2; local `R/arcgm_efalls_report.pdf` / `.txt` | Implementation pilot evaluation (grey literature) | Setting: SWAN PCN (Wigan), Feb–Nov 2025. Algorithm run in the GM Secure Data Environment. "intermediate risk (10-25%)" (`R/arcgm_efalls_report.txt` line 840, 973). 1,158 identified → 740 contacted → 160 attended → 54 enrolled in strength and balance classes. Explicitly "not an evaluation of the effectiveness of either the eFalls algorithm or the intervention" (lines 173, 991). Box 2: "first global implementation of eFalls in routine primary care" (`SP/research/work/arcgm_p-12.png`). Participants frailer than expected and some had previous falls (report section 4.1). Funding NIHR200174 (line 100) | Which equation (original or recalibrated) was used is not stated (NOT FOUND; HTA line 1156 implies the recalibrated one, INFERRED). No coefficient issues reported |
| 4 | Badrock B. "Case-Finding for Falls Prevention: Wigan Pilot" poster, Abstract ID 3564, 2025 International Conference on Falls and Postural Stability (BGS). https://www.bgs.org.uk/case-finding-for-falls-prevention-wigan-pilot; PDF https://www.bgs.org.uk/sites/default/files/2025-09/Case-Finding%20for%20Falls%20Prevention-Wigan%20Pilot%20(Poster)_0.pdf | Conference abstract | Early implementation of the GM pilot | none |
| 5 | NICE NG249 Evidence Review D: "Electronic patient records: Falls: assessment and prevention in older people and people 50 and over at higher risk" (April 2025). NBK616085; PMID 40638764. https://www.ncbi.nlm.nih.gov/books/NBK616085/ | Guideline evidence review | Included Archer 2024 as one of 3 externally tested CPMs (c-statistic 0.816). Noted over-prediction on external validation. No paired sensitivity/specificity reached 0.70. "Further research in different cohorts is needed, before this model or similar models can be recommended for use in practice." GRADE low/very low (WebFetch summary) | none |
| 6 | van der Velde N. "Editor's view." *Age Ageing* 2024;53(4):afae083. https://academic.oup.com/ageing/article/53/4/afae083/7659194 | Editorial commentary | Highlights discrimination 0.82 and EHR embedding; "A next step would be international validation of the eFALLs tool" | none |
| 7 | van der Velde N, Seppala LJ, … "Falls prevention in community-dwelling older adults and implementation of world falls guidelines: a call for action across Europe" (EuGMS SIG). *Eur Geriatr Med* 2025. doi:10.1007/s41999-025-01206-y; PMC12378773 (`R/epmc_PMC12378773.xml`) | Position paper | Cites eFalls as an EHR case-finding example; "Currently, the eFalls tool is being tested as a case stratification approach in Greater Manchester" | none |
| 8 | Mitchell A, Ogliari G, Burton JK, Clegg A, Todd O, EuGMS Big Data SIG. *Eur Geriatr Med* 2025. doi:10.1007/s41999-025-01276-y | Position paper | Cites eFalls (Crossref reference match) as a big-data example | not examined in full |
| 9 | Hancox Z, Kingsbury SR, Conaghan PG, Clegg A, Relton SD. *Rheumatology (Oxford)* 2025. doi:10.1093/rheumatology/keaf185; PMC12316357 | Different model (TG-CNN for joint replacement) | Cites eFalls as an analogous "early warning system" (`R/epmc_PMC12316357.xml`) | none |
| 10 | Yang A, … Chan JCM et al. *Age Ageing* 2025;54(10):afaf285. doi:10.1093/ageing/afaf285; PMC12510402 | New ML model (Hong Kong) | Background citation only: "In the United Kingdom (UK), a 1-year fall risk prediction model … C-statistic of 0.82 on external validation [17]". No head-to-head validation of eFalls (`R/epmc_PMC12510402.xml`) | none |
| 11 | Osonuga A et al. "Artificial intelligence in hospital fall prevention…" *Safety Science* 2025. doi:10.1016/j.ssci.2025.107104 | Review | Cites eFalls (Crossref reference match) | not examined |
| 12 | Wang. "From Code to Care and Digital Detection of Frailty: A Scoping Review of electronic Frailty Index (eFI) Applications…" *J Clin Med Res* 2025. doi:10.46889/jcmr.2025.6301 | Scoping review | Cites eFalls | not examined |
| 13 | Relton S et al. PREDICT project (NIHR206843): https://github.com/sdrelton/PREDICT; newsletters in `docs/newsletters` | Methods project with code | Applies drift detection and recalibration methods to eFalls in Connected Bradford (CB_2151). Dec 2025 newsletter: methods applied to a 12-month fall model; no paper yet (NOT FOUND) | Code uses ln(P+1)/10 over 3 months (section 2.1); not reported as an issue |
| 14 | Best K … Clegg A. eFI2. *Age Ageing* 2025;54(4):afaf077; plus BJGP editorial PMC12117642, Romero-Ortuno & Keevil afaf111, Editor's view afaf127 | Sister model | Shares the eFI+ code list and SAIL predictor dataset (see efi_codelists.md); not an eFalls validation | – |
| 15 | Other citing works: Hou XZ et al. *J Affect Disord* 2024 (10.1016/j.jad.2024.08.218); Duy W et al. *Clin Ophthalmol* 2025 (10.2147/opth.s503177); Orlandini L et al. *Aging Clin Exp Res* 2025 (10.1007/s40520-025-03125-1); Xu J et al. *Medicine* 2025 (10.1097/md.0000000000043513); Li X et al. *Geriatr Nurs* 2025 (10.1016/j.gerinurse.2025.103688); Yin M et al. *Front Public Health* 2026 (10.3389/fpubh.2026.1749921) | Unrelated models citing eFalls | Titles indicate background citations only (INFERRED; full texts not examined) | – |

Checked and **not** citing or validating eFalls:
- Dormosh et al., *Age Ageing* 2024 afae131, systematic review of fall models from routine data (Crossref refs: 0 eFalls matches).
- Logan Ellis & Rockwood, *Age Ageing* 2025 afaf309 (0 matches).
- Delbaere, *Age Ageing* 2025 afae291 (0 matches).
- Kitchen L et al., *Future Healthc J* 2026 (PMC13521067; discusses eFI2 only).

**Reported coefficient / sex / BMI issues in the later literature: NOT FOUND.** No later publication, report or guideline review flags the Box S3.1 sex sign, the BMI ≥40 definition, the duplicated alcohol label or the recalibration discrepancy. The Sci Rep supplement and the official implementation instructions both silently use the Table S3.2 parameterisation (Male reference). [EXPLICIT absence in the sources read; web and literature search limited to the sources listed]

---

## 6. Code-list request route and later release

- **Request route** [EXPLICIT]:
  - A&A: "Code lists used to define variables are available on reasonable request from the corresponding author" (`SP/efalls/fulltext.md` line 367). Corresponding author A. Clegg, a.p.clegg@leeds.ac.uk (`SP/efalls/fulltext.xml` line 23).
  - HTA: the same (`R/gjac1008_struct.txt` lines 412, 1786). The HTA also mentions a non-public "technical specification document" (line 407).
  - The implementation instructions list contacts A. Clegg, K. Best (eFI2 lead, k.e.best@leeds.ac.uk) and S. Relton (eFI+ lead, s.d.relton@leeds.ac.uk) (`R/efi2plus/instr_v1.13.txt` lines 3-29).
  - For Connected Bradford project queries: cbradford@bthft.nhs.uk quoting project 1551.
- **Official release of eFalls code lists: NOT FOUND.**
  - Not in the A&A supplement, the HTA toolkit, KateBest/eFI2plus (only 5 example rows in instructions Table 1, lines 48-96) or CB_1551_eFI2 (empty).
  - Not on Zenodo, figshare, OSF, GitLab, OpenCodelists or HDR UK (the latter two per efi_codelists.md).
- **Partial public proxy** (already documented, not redone): DynAIRx `Baseline2_Codelist.csv`, the eFI2/eFI+ SNOMED CT + CTV3 list with time/age columns, 7,555 rows (https://github.com/DynAIRx/GCAF_DynAIRx; BMC Med Res Methodol 2025;25:138). See `SP/research/efi_codelists.md` sections 3 and 7.
  - Instructions v1.13 give 7,556 SNOMED codes and v1.9 gives 7,555, consistent with Baseline2's 7,555 rows [EXPLICIT numbers; match INFERRED].
  - The Read v2 version used in SAIL and the "BNF chapters table" for polypharmacy (instructions line 112) remain non-public.
- **Public outcome code lists:** A&A Table S2.3 ICD-10 codes (`SP/efalls/supp.md` line 86 onward); HTA Appendix 1 (`R/gjac1008_struct.txt` lines 1884-1920). [EXPLICIT]

---

## 7. Contradictions found in this round

1. **Sex term:** Box S3.1 and HTA Tables 16/34 say "−6.258 … −0.304 (if male)". Table S3.2, HTA Table 15, implementation instructions Table 4 and Sci Rep Table S1 say Male = Reference, Female −0.303708, Constant −5.954459.
2. **BMI:** Box S3.1/S3.2 and HTA Table 34 say "obese if BMI ≥ 40". HTA Table 16 says "standard BMI cut-offs". Instructions v1.13 say obese ≥30, overweight 25–29.9.
3. **Polypharmacy window and transform:** Box S3.1/HTA say 120 days. Instructions v1.13 say "previous 90 days" for the shared code list, with eFI+ continuous. PREDICT code uses `unique_bnf_last_3_months` and `np.log(P+1)/10`.
4. **Alcohol reference:** HTA Table 15 prints "Higher-risk drinking | Reference". Table S3.2 and instructions have Lower risk = reference (0).
5. **Recalibrated model:** Box S3.2/HTA Table 34 (α −0.423, β 1.25; example p = 0.106) vs HTA Table 35 (= 1.21 × coefficients, constant −7.2509; example p = 0.148 computed).
6. **Instructions worked example:** stated L −3.023169 / P 0.0464 vs recomputed −2.978465 / 0.0484.
7. **External-validation CITL** (−0.87 in the A&A abstract vs −0.931 in the HTA abstract and Sci Rep Table 3): pooled vs whole-population estimates. **Not a real contradiction** (A&A Table 2, `SP/efalls/fulltext.md` line 219).
8. **Calibration direction:** over-prediction in Connected Bradford ≥65 (O/E 0.43) vs under-prediction in SWYPFT working-age adults (O/E 1.60). A population difference, not an error.

---

## 8. Unresolved

1. **Which equation GM deployed** (original vs Bradford-recalibrated vs GM-recalibrated) and its coefficients: not public. Ask A. Clegg or the GM ICB BI Unit.
2. **Why the implementation instructions were deleted from KateBest/eFI2plus HEAD on 2026-02-26**, and whether a newer official version exists. Ask K. Best or A. Clegg.
3. **Box S3.1 sex sign, BMI ≥40, and the Table 35 vs Box S3.2 discrepancy:** no correction published. Seek written confirmation from the authors. Consider a letter or a comment via the NIHR Journals Library "Submit a response" route (the HTA page has 0 responses).
4. **Licence terms for non-UK research or service use** (e.g. Meuhedet): no licence text is public. Contact the University of Leeds and the corresponding author.
5. **Official eFalls code lists** (SNOMED/CTV3/Read 2), the BNF chapter table and the "technical specification document": on request only.
6. **Sci Rep recalibration α/β and its code-list source:** not reported.
7. **ARC Wessex project page content:** the live fetch failed (TLS error). Only the title is known.
8. **MHRA registration / UKCA status of eFI2+ (including eFalls):** not found in public sources.

---

## 9. Local files created in this round
- `W/`: cr_*.json, pubmed_now.xml, pm_40638764.xml, epmc_*.json, epmc_core_*.json, hta_now.html, scirep_ref.pdf/.txt (accepted manuscript), scirep_esm.pdf/.txt (supplement), osf_search.txt
- `R/predict/`: efalls.py.txt, refit_efalls_model.py.txt, input_data_drift_analysis.py.txt, qrisk_efalls_demographics.txt, README.md.txt, LICENSE.txt, HEAD_SHA.txt, newsletters (.pdf/.txt)
- `SP/research/work/arcgm_p-11.png`, `arcgm_p-12.png` (rendered report pages); `SP/research/work/instr_img/` (images from instructions docx; eFI2 equations)
