# eFalls reproduction — validation, meta-analysis and sample-size methods (implementable formulas)

Compiled 2026-09-14. Scope: calibration (CITL, slope, O/E, grouped 20ths, smoothers), C-statistic SE/pooling, net benefit, bootstrap optimism and instability, IECV, REML+HKSJ+prediction intervals, Riley 2019/2021 sample size (reproducing eFalls Tables S2.1/S2.2), recalibration.

Labels: **[EXPLICIT]** stated in the cited source; **[VERIFIED]** additionally reproduced numerically here; **[INFERRED]** my derivation or reverse-engineering; **[NOT FOUND]** not in any local/checked source.

All paths below are relative to `scratchpad/research/` unless absolute
(`<research workspace (not distributed)>/`).

---

## 0. Sources and verification scripts

| Key | Source | Local file |
|---|---|---|
| AA | Archer et al. Age Ageing 2024;53:afae057 (PMC10960070, CC-BY) main text | `../efalls/fulltext.md` |
| AA-S | same, supplement | `../efalls/supp.md`; equations `supp_fulltext_with_math.txt` |
| HTA | Archer…Clegg, NIHR HTA GJAC1008 (eFI+) | `raw/gjac1008_struct.txt` |
| Riley19 | Riley et al. Stat Med 2019;38:1276 "Minimum sample size … PART II binary and time-to-event" (PMC6519266) | `txt/PMC6519266.txt` |
| Riley21 | Riley et al. Stat Med 2021;40:4230 "Minimum sample size for external validation … binary outcome" (CC-BY, Birmingham repository PDF) | `txt/riley2021_simvalss.txt` |
| Riley24-2 | Riley et al. BMJ 2024;384:e074820 "Evaluation of clinical prediction models (part 2)" + supplement | `txt/PMC10788734.txt`, `txt/rilr074820.supp.txt` |
| Riley24-3 | Riley et al. BMJ 2024 part 3 (external validation sample size) | `txt/PMC11778934.txt` |
| Collins24-1 | Collins et al. BMJ 2024;384:e074819 part 1 (internal / IECV) | `txt/PMC10772854.txt` |
| Snell18 | Snell et al. SMMR 2018;27:3505 (scales for meta-analysis; PMC6193210) | `txt/PMC6193210.txt` |
| Riley16 | Riley et al. BMJ 2016;353:i3140 (big-data external validation; PMC4916924) | `txt/PMC4916924.txt` |
| Rover15 | Röver, Knapp, Friede BMC MRM 2015;15:99 (HKSJ + modification; PMC4647507) | `txt/PMC4647507.txt` |
| IntHout14 | IntHout et al. BMC MRM 2014;14:25 (PMC4015721) | `txt/PMC4015721.txt` |
| Partlett17 | Partlett & Riley Stat Med 2017;36:301 (REML CI/PI coverage; PMC5157768) | `txt/PMC5157768.txt` |
| RC23 | Riley & Collins Biom J 2023;65:e2200302 (stability; PMC10952221) + Stata/R code | `txt/PMC10952221.txt`; code extracted to `work/rc2023/` from `raw/PMC10952221_supp.zip` |
| SH16 | Steyerberg & Harrell J Clin Epidemiol 2016;69:245 (PMC5578404) | `txt/PMC5578404.txt` |
| VC19 | Van Calster et al. BMC Med 2019;17:230 + Additional file 1 (PMC6912996) | `txt/PMC6912996.txt`, `txt/vancalster2019_addfile1.txt` |
| AS14 | Austin & Steyerberg Stat Med 2014;33:517 (loess calibration; PMC4793659) | `txt/PMC4793659.txt` |
| V08 / V16 / V19 | Vickers 2008 BMC MIDM 8:53 (PMC2611975); Vickers 2016 BMJ 352:i6 (PMC4724785); Vickers 2019 Diagn Progn Res 3:18 (PMC6777022) | `txt/PMC2611975.txt`, `txt/PMC4724785.txt`, `txt/PMC6777022.txt` |
| A22 | Archer et al. BMJ 2022;379:e070918 (STRATIFY-Falls, same group; PMC9641577) + supplement | `txt/PMC9641577.txt`, `txt/archer2022_supp.txt` |
| pmsampsize | Stata `pmsampsize` 1.3.2 (Ensor, 4 Dec 2023) | `raw/pmsampsize.ado` |
| pmvalsampsize | Stata `pmvalsampsize` 1.0.1 (Ensor, 10 Nov 2023) | `raw/pmvalsampsize.ado` |
| pmcalplot | Stata `pmcalplot` 2.2.2 (Ensor, 6 Mar 2024) | `raw/pmcalplot.ado` |
| sknor | Stata `sknor` 1.0.1 (Kontopantelis 2008; SSC) — http://fmwww.bc.edu/repec/bocode/s/sknor.ado , .hlp | `raw/web_sknor.ado`, `raw/web_sknor.hlp` (downloaded this round; read, not executed) |
| running | Stata `running` 3.1.1 (Sasieni/Royston/Cox; SSC) — http://fmwww.bc.edu/repec/bocode/r/running.ado , .hlp | `raw/web_running.ado`, `raw/web_running.hlp` |
| Stata meta | Stata [META] meta summarize (Release 19 manual) | `raw/stata_meta_summarize.pdf`, text `txt/stata_meta_summarize.txt` |
| Stata lowess | Stata 17 [R] lowess | `raw/s17_r_full.txt` lines 64971–65275 |
| metafor | metafor 5.1-18 `predict.rma` help | `raw/metafor_predict_rma.html` (text `work/metafor_predict_rma.txt`) |

Verification scripts (numpy/scipy only), directory `validation_verify/`:
- `verify_all.py` → output `verify_all_output.txt` (all sections below).
- `ss_validation.py` — pmvalsampsize criteria (O/E, C, sNB, slope with normal/Fleishman; cross-checks HTA Tables 4–6).
- `slope_sknor.py` — calibration-slope criterion with the exact sknor/Ramberg LP distribution (quadrature + Monte Carlo).
- `meta_reml.py` — REML τ², HKSJ CI, prediction intervals; verified against Stata [META] example.
- `dev_ss_cstat_mc.py` — Monte Carlo of pmsampsize `cstatistic()` route (hypothesis test for 50,927).

---

## 1. What the eFalls publications say about validation methods

- Software: development + internal validation in Stata 17; external validation in R 4.2.3 (AA `fulltext.md` line 106) [EXPLICIT]. HTA says R 4.3.1 for Leeds analyses (HTA line 502) — version discrepancy [EXPLICIT].
- Internal validation: "bootstrapping with 25 samples" with model performance in bootstrap sample and original data to obtain optimism (AA line 118) [EXPLICIT]. HTA: "bootstrapping with 50 samples" (HTA line 539) [EXPLICIT] → contradiction (25 vs 50).
- Stability: "probability distribution and calibration instability plots [27]" (ref 27 = RC23) (AA line 118). HTA adds prediction instability plots and MAPE (HTA lines 510–514) [EXPLICIT].
- Calibration: slope, CITL, O/E; calibration plots "within groups (defined by 20ths of outcome risk)" and "smooth (loess) calibration curves" (AA line 120; HTA line 520) [EXPLICIT]. Supplement Figure S3.4 caption says "lowess smoothed calibration curves" (AA-S line 327) [EXPLICIT].
- HTA definitions (lines 517–519) [EXPLICIT]: slope = coefficient of LP in logistic regression with LP as only covariate; CITL = intercept of logistic regression "using the LP from the model as an offset term (forcing the coefficient to be 1)"; O/E = "observed risk across the population divided by the mean predicted risk".
- Discrimination: C-statistic (AA line 120; HTA line 522). Clinical utility: net benefit + decision curves, thresholds 10–25% a priori (AA line 120; HTA Table 7 line 526–533) [EXPLICIT].
- Clustering: performance per GP practice, plotted against SE; "practices with <10 events omitted from visualisations"; pooled by random-effects MA "estimated using restricted maximum likelihood"; CIs "using the Hartung–Knapp–Sidik–Jonkman variance correction [32]" (ref 32 = Rover15); pooled "on appropriate scales [31]" (ref 31 = Snell18) (AA line 122) [EXPLICIT].
- HTA scale statement: "The calibration slope, CITL and O/E were pooled across practices on their original scales, while the C-statistic was pooled on the logit scale." (HTA line 543) [EXPLICIT]. Snell18 recommends log O/E (see §7.4) → contradiction with Snell18; Table 2 reverse-engineering shows analysis-specific scales (§7.4).
- IECV: across "subgroups by ranked Welsh Index of Multiple Deprivation (WIMD, 2019)"; development process repeated on all-but-one group; pooled by RE-MA "as specified above" (AA line 126). HTA: "fifths of rank by the Welsh Index of Multiple Deprivation" (HTA line 545). AA results mention a group "missing WIMD details" (AA line 274) → k = 6 groups [INFERRED; confirmed by Table 2 PIs, §7.4].
- Recalibration: logistic regression in external data "with the linear predictor value from the eFalls model as the only variable" (AA line 134); supplement equation ln(P_recal/(1−P_recal)) = α_recal + β_recal·LP_eFalls fitted with R `glm` (`supp_fulltext_with_math.txt` lines 307–309) [EXPLICIT].
- Development sample size: 50,927 (2,445 events), 90 parameters (AA line 98); validation 10,882 (523) (AA line 100); details Tables S2.1/S2.2 (AA-S lines 72–84) [EXPLICIT].

---

## 2. Calibration measures (formulas)

Notation: y_i ∈ {0,1}; LP_i = model linear predictor; p_i = expit(LP_i); N individuals; O = Σy_i; E = Σp_i; φ = O/N.

### 2.1 Calibration-in-the-large (CITL) [EXPLICIT]
Fit logit P(y_i=1) = α + 1·LP_i (LP as **offset**); CITL = α̂ (HTA line 518; Riley24-2 supp S3 `rilr074820.supp.txt` lines ~57–66; VC19 add-file line 16; Snell18 eq 3).
- Stata: `logistic y if touse, offset(lp) coef` → `_b[_cons]` (pmcalplot.ado lines 502–505). R: `glm(y ~ offset(lp), family=binomial)` (RC23 code `work/rc2023/Figure_3_code.R` line 61) [EXPLICIT].
- pmcalplot computes the LP as ln(p/(1−p)) from the supplied predicted probabilities (pmcalplot.ado line 494) [EXPLICIT].
- SE: model-based SE(α̂) = 1/√Σ p̂_i(1−p̂_i), p̂_i = expit(α̂+LP_i) (standard GLM information; [INFERRED], VERIFIED numerically in `verify_all.py` §4: glm SE 0.0204 = closed form 0.0204). Wald CI α̂ ± 1.96·SE.
- Solve by Newton: α ← α + Σ(y−p̂)/Σp̂(1−p̂) (score equation Σ y_i = Σ expit(α+LP_i)) [INFERRED].

### 2.2 Calibration slope [EXPLICIT]
Fit logit P(y_i=1) = a + β·LP_i; slope = β̂ (HTA line 517; Riley24-2 supp; pmcalplot.ado lines 496–498 `logistic y lp, coef`). SE from inverse information of the 2-parameter fit; Wald CI [INFERRED standard].
- Property [VERIFIED, synthetic]: after fitting this model in the same data, the recalibrated LP a+β·LP has apparent slope 1.000, CITL 0.000, O/E 1.000 exactly (score equations) — exactly the AA Table 2 "Recalibrated model … Overall performance (apparent)" values 1.000 / 0.000 / 1.000 (AA lines 203, 219, 235).
- Note: at internal validation CITL is irrelevant (average prediction matches event rate) — VC19 line 53 [EXPLICIT].

### 2.3 O/E [EXPLICIT]
O/E = (Σy/N)/(Σp/N) = O/E (HTA line 519; pmcalplot.ado lines 508–513: mean(y)/mean(p)). Riley24-2 supp S3: values <1 over-prediction.
- SE (delta method): SE(ln O/E) ≈ √((1−φ)/O) (Riley21 eq 4, `riley2021_simvalss.txt` lines 363–367); CI on ln scale, back-transformed (Riley21 line 368–369) [EXPLICIT].
- Caution: AA Table S3.3 apparent O/E CI 0.9897182–1.010282 is symmetric on the O/E scale (half-width 0.010282 → SE 0.005246); √((1−φ)/O) with O=32,097, φ=0.0486 gives 0.005444 (half-width 0.01067). Method used for Table S3.3 CIs **NOT FOUND** [INFERRED mismatch].

### 2.4 Grouped calibration by 20ths [EXPLICIT + INFERRED]
- pmcalplot (binary): `xtile binvar = p, n(bin)` (default bin 10; eFalls "20ths" ⇒ `bin(20)`), observed = group mean of y, expected = group mean of p; optional CI = Wald for a proportion, √(ō(1−ō)/n_g), truncated to [0,1] (pmcalplot.ado lines 86, 211–224) [EXPLICIT].
- Stata `xtile` quantile rule (percentile definition, ties) not reproduced from documentation here; `verify_all.py` §7 uses numpy `averaged_inverted_cdf` as an approximation [INFERRED]. R team's grouping implementation NOT FOUND.

### 2.5 Smoothed calibration curves [EXPLICIT/NOT FOUND mix]
- **pmcalplot 2.2.2** (current): `running y p, span(1) ci generate() gense()`; CI = smooth ± 1.96·SE truncated to [0,1] (pmcalplot.ado lines 412–419). Header: smoother switched from lowess to `running` on 12/06/2023 (lines 36, 44) [EXPLICIT].
  - `running` = symmetric-nearest-neighbour running-line least squares (default), knn = (n·span − 1)/2 so span(1) uses ≈ all-n neighbourhoods; default when no span/knn: knn = 0.5·n^0.8 (web_running.hlp lines 83–100; web_running.ado lines 89–92) [EXPLICIT].
- **Earlier pmcalplot (≤ 2023)** used lowess (header line 44 "updated 'lowess' language"), the eFalls supplement says "lowess smoothed" — the bwidth used by the old version is **NOT FOUND** (old .ado not available).
- **Stata lowess** (Stata 17 [R] lowess): running-line least squares with tricube weights, default bwidth(0.8), no robustness iterations; for point i the subset is i±k with k = floor((N·bwidth − 0.5)/2), weights w_j = (1 − (|x_j−x_i|/Δ)³)³ with Δ = 1.0001·max(x_{i+}−x_i, x_i−x_{i−}) (`raw/s17_r_full.txt` lines 65022, 65046–65049, 65240–65252) [EXPLICIT]; implemented in `verify_all.py` §7 `stata_lowess()`.
- **RC23 Stata code** (same Birmingham group): calibration-instability curves use `running day30 pr` with defaults, then plotted with `lowess … bwidth(0.02)`; 95% prediction-instability bands smoothed with `lowess … bwidth(0.2)` (`work/rc2023/Stata Code for examples and figures of Section 4_1.do` lines 99–100, 145, 155, 177) [EXPLICIT].
- **R**: AS14 used R `loess` with default span 0.75 and compared `lowess(Y~P, f=k, iter=0)`, noting Harrell's advice that robustifying iterations be 0 (PMC4793659.txt lines 42, 199) [EXPLICIT]. RC23 R code uses `lowess(pred, y, iter = 0)` (`work/rc2023/Figure_2_code.R` lines 113, 118) [EXPLICIT]. VC19 used `val.prob.ci.2` (CalibrationCurves) loess (add-file line ~20) [EXPLICIT]. R loess default degree (2) and family are not stated in local sources [NOT FOUND locally]. The exact smoother/span used by the Leeds external validation is **NOT FOUND**.

---

## 3. Discrimination: C-statistic, SE, CI, logit scale

- C = proportion of concordant event/non-event pairs (ties ½) = AUROC (HTA line 522) [EXPLICIT]; Mann–Whitney computation [INFERRED standard].
- SE (Newcombe; Riley21 eq 11, lines 723–733; pmvalsampsize.ado line 644) [EXPLICIT]:
  SE(C) ≈ √{ C(1−C)[1 + (N/2 − 1)(1−C)/(2−C) + (N/2 − 1)C/(1+C)] / (N² φ(1−φ)) }.
  Riley21 notes Feng et al. found Wald CIs with this SE have near-nominal coverage (lines 737–740) [EXPLICIT].
- Logit-scale SE (delta method): SE(logit C) = SE(C)/(C(1−C)); A22 used this for pooling ("standard errors of logit C calculated using the delta method", PMC9641577.txt line 116) [EXPLICIT in A22]. Snell18's QRISK2 example used bootstrap SEs (PMC6193210 §4) [EXPLICIT].
- Check vs AA Table S3.3 apparent C 0.7434 (0.74077–0.74612): implied SE 0.001365; Newcombe 0.001403; Hanley–McNeil 0.001611 (N=660,417, O=32,097) → SE method used by Stata team **NOT FOUND** (DeLong-type likely; [INFERRED]) (`verify_all.py` §5).
- Oddity: AA Table S3.4 subgroup CIs often exclude the point estimate (e.g. IMD1 slope 1.123 (1.126 to 1.158)) → not Wald; possibly bootstrap percentile intervals [INFERRED]; AA Table 2 recalibrated C CI "0.801 to 1.000" looks like a typo [INFERRED].

---

## 4. Net benefit, treat-all, treat-none, standardised NB

- NB(p_t) = TP/N − (FP/N)·p_t/(1−p_t) (V08 PMC2611975.txt line 30; Riley24-2 supp S4 eq 3, `rilr074820.supp.txt` ~lines 115–130) [EXPLICIT]; "positive" if p̂ ≥ p_t (V08 steps 1–6, lines 32–37; V16 line 46) [EXPLICIT].
- Equivalent: NB = sens·φ − (1−spec)(1−φ)·p_t/(1−p_t) (Riley21 eq 3 / eq 14) [EXPLICIT].
- Treat none: NB = 0. Treat all: NB_all = φ − (1−φ)·p_t/(1−p_t) (Riley24-2 supp S4, line ~139) [EXPLICIT].
- Standardised NB: sNB = NB/φ (max 1) (Riley24-2 supp line ~136; HTA line 542 "standardised (i.e. to the 0–1 scale) net benefit") [EXPLICIT].
- AA Figure S3.7 / HTA line 542: plotted median (over practices) of sNB(model) − sNB(treat-all / treat-none / "next best" = max of the two), with 50/80/90% bands, thresholds 0–0.5 [EXPLICIT]; practice-level details of median/bands NOT FOUND.
- Bootstrap-corrected NB (V08 lines 66–76): optimism(p_t) = NB_boot-sample − NB_original-data averaged over 200 bootstraps; corrected = apparent − optimism [EXPLICIT].
- SE for sNB (Marsh; Riley21 eq 12): SE(sNB)² = (1/N)[sens(1−sens)/φ + w² spec(1−spec)/(1−φ) + w²(1−spec)²/(φ(1−φ))], w = ((1−φ)/φ)·p_t/(1−p_t) (lines 806–815) [EXPLICIT].
- VERIFIED: from AA Table S3.4 (external validation, per 1000) at p_t=0.10: φ=0.0293, sens 0.66, spec 0.82, NB = −0.00017, NB_all = −0.0786 → model ≈ treat-none at 10%, consistent with AA line 289 ("no better than the treat-none alternative in the pre-specified range") (`verify_all.py` §6).

---

## 5. Internal validation: Harrell bootstrap optimism; instability

### 5.1 Enhanced (Harrell) bootstrap with full re-development [EXPLICIT]
Collins24-1 Box 2 (PMC10772854.txt lines 139–159):
1. Develop model on all data; apparent performance θ_app.
2. For b = 1..B: bootstrap sample (size N, with replacement); **repeat all model-building steps** (for eFalls: FP selection, LASSO with 10-fold CV λ) → model M_b.
3. θ_b,boot = performance of M_b in bootstrap sample; θ_b,orig = performance of M_b in original data.
4. optimism_b = θ_b,boot − θ_b,orig; Ō = mean_b optimism_b.
5. θ_corrected = θ_app − Ō.
Collins24-1 recommends ≥ 500 bootstraps (≥200 per Steyerberg) (line 159) [EXPLICIT]; eFalls used 25 (AA) or 50 (HTA) [EXPLICIT]. Optimism-corrected slope can serve as uniform shrinkage (line 159) [EXPLICIT].
- VERIFIED convention on AA Table S3.3: slope 1.0083 − 0.0026419 = 1.00566 [1.0057]; CITL 0 − 0.000277 [−0.0003]; O/E 1 − 0.0002313 [0.9998]; C 0.7434 − 0.0004245 [0.7430] (`verify_all.py` §5).
- How the "95% CI" of average optimism in Table S3.3 was computed (e.g. mean ± t·SD/√B vs percentiles) — NOT FOUND.
- For O/E and C the same additive optimism is used (no log/logit transform) — [INFERRED from Table S3.3 arithmetic].

### 5.2 Instability (RC23) [EXPLICIT]
Box 1 (PMC10952221.txt lines 140–146): original predictions p̂_i; B bootstrap samples of size N; rebuild model with identical strategy (including tuning by CV and variable selection); predict p̂_bi for every individual in the **original** data; B ≥ 200 recommended.
- Prediction instability plot: scatter p̂_bi vs p̂_i; optional 2.5th/97.5th percentile of p̂_bi per individual smoothed by LOESS (bandwidth 0.2–0.8 suggested) (RC23 §3.2.1). Stata: `egen rowpctile(pr1-prB), p(2.5)` / `p(97.5)`, `lowess … bwidth(0.2)` (`Section 4_1.do` lines 90–100).
- Calibration instability plot: B calibration curves of bootstrap models applied in original data, overlaid with the original curve (RC23 §3.2.2). (AA Fig S3.4 caption instead says bootstrap models "applied within that bootstrap sample" (AA-S line 327) whereas HTA line 513 says "applied in the original data set" → contradiction.)
- Prediction-distribution instability (AA Fig S3.3; HTA line 511): overlaid densities of LP for each bootstrap model in its bootstrap sample vs in original data [EXPLICIT; construction details NOT FOUND].
- MAPE_i = (1/B) Σ_b |p̂_bi − p̂_i|; average MAPE = (1/(BN)) Σ_b Σ_i |p̂_bi − p̂_i| (PMC10952221.txt lines 156, 160); plot MAPE_i vs p̂_i (§3.2.4).
- Classification instability index_i = proportion of bootstrap models giving a different classification (above vs below threshold) than the original model (lines 245–247). Stata code uses ≥ threshold as "high": change = (p̂_bi ≥ t & p̂_i < t) or (p̂_bi < t & p̂_i ≥ t) (`Section 4_1.do` lines 125–133).
- Decision-curve instability: overlay NB curves of bootstrap models in original data (RC23 §5.3).
- C-statistic instability: distribution of C of bootstrap models in original data (`Section 4_1.do` lines 64–79).

---

## 6. Internal–external cross-validation (IECV) [EXPLICIT + INFERRED]
- Definition: leave one cluster out, repeat the entire development process on the remaining clusters, validate in the omitted cluster; repeat for all clusters; final model developed on all data; summarise cluster-level performance by (random-effects) meta-analysis (SH16 PMC5578404.txt line 7; Collins24-1 Box 4 line 188; AA line 126; HTA line 546) [EXPLICIT].
- eFalls clusters: WIMD 2019 fifths plus a "missing WIMD" group → k = 6 [INFERRED from AA line 274 + HTA line 545; supported by PI arithmetic §7.4].

---

## 7. Random-effects meta-analysis: REML, HKSJ, prediction interval, scales

### 7.1 Model and REML [EXPLICIT]
y_j ~ N(μ_j, s_j²), μ_j ~ N(μ, τ²) (Snell18 eq 4). REML log-likelihood = ML log-likelihood − ½ ln Σ_j 1/(s_j²+τ²) + const (Stata meta summarize M&F, `txt/stata_meta_summarize.txt` lines ~1620–1634). Implementation (fixed-point/Fisher scoring) [INFERRED standard]:
w_j = 1/(s_j²+τ²); μ̂ = Σw_j y_j/Σw_j; τ² ← max(0, Σw_j²[(y_j−μ̂)² − s_j²]/Σw_j² + 1/Σw_j); iterate.
Conventional SE(μ̂) = √(1/Σw_j).

### 7.2 HKSJ CI [EXPLICIT]
q = (1/(k−1)) Σ w_j (y_j − μ̂)²; Var_HK = q·(1/Σw_j); CI = μ̂ ± t_{k−1,0.975}·√Var_HK (Snell18 eq 5 line 232; Partlett17 eq 3 line 64; Stata M&F lines ~1797–1829). Modified (truncated) version: q* = max(1, q) (Rover15 eq 11 line 81; Partlett17 eq 4; Stata `se(khartung, truncated)`) — eFalls does not say which; standard HKSJ assumed [INFERRED].

### 7.3 Prediction interval [EXPLICIT]
- Snell18 eq 6 (line 250), Partlett17 eq 8 (line 107), Riley16 (via Higgins), Stata M&F (`stata_meta_summarize.txt` ~lines 1832–1840): μ̂ ± t_{k−2,0.975}·√(τ̂² + Var(μ̂)), **t with k−2 df**, Var(μ̂) = conventional REML variance.
- Partlett17 also studied replacing Var(μ̂) by Var_HK (line 107–110) [EXPLICIT].
- metafor `predict.rma`: with `test="knha"/"hksj"` uses t with **k−p = k−1** df and the model (HK-adjusted) variance; `predtype="Riley"` gives k−2 (metafor help Note) [EXPLICIT].
- VERIFIED implementation (`meta_reml.py`) on Stata [META] meta summarize Examples 1/5/6 (pupil IQ, 10 studies; SEs reconstructed from printed CIs): τ² 0.0754 [0.0754]; θ 0.13353 [0.13353]; SE 0.10616 [0.10616]; HK SE 0.12150 [0.12151]; HK CI −0.14134 to 0.40839 [−0.14134 to 0.40840]; 90% PI with t_{k−2} and conventional SE [−0.414, 0.681] = Stata [−0.414, 0.681] (with HK SE it would be [−0.425, 0.692]).
- Funnel-type bounds in AA Figs 2/3 ("95% prediction intervals for the performance measure across possible standard errors"): presumably μ̂ ± c·√(τ̂² + SE²) as a function of SE (Riley16 Fig 3b concept, `PMC4916924.txt` line 419) — exact multiplier NOT FOUND [INFERRED].

### 7.4 Scales — Snell 2018 vs what eFalls actually did
- Snell18 recommendation: C on **logit** scale; E/O (or O/E) on **log** scale; calibration slope and CITL on original scale (Snell18 abstract, `PMC6193210.txt` line 38; Box 2 in §5) [EXPLICIT]. A22 (same group, 2022): log O/E, logit C (PMC9641577.txt line 116) [EXPLICIT].
- HTA line 543: slope, CITL and **O/E original**, C logit (for practice-level pooling in development data) [EXPLICIT].
- Reverse-engineered from AA Table 2 (`verify_all.py` §3; PI midpoints on each scale and τ²-magnitude back-calculation) [INFERRED, strong]:

| Analysis (software) | slope | CITL | O/E | C | Evidence |
|---|---|---|---|---|---|
| Development, pooled across 455 practices (Stata) | orig | orig | **orig** (PI 0.144–2.236 vs reported 0.14–2.24; log would give 0.418–3.385) | **logit** (τ²=0.010 ⇒ PI 0.679–0.758 vs 0.68–0.76; original would give 0.523–0.917) | matches HTA line 543 |
| IECV, k=6 WIMD groups (Stata) | orig | orig | **log** (0.210–3.693 vs 0.21–3.70) | **orig** (0.589–0.851 vs 0.59–0.85; logit gives 0.667–0.767) | ≠ HTA line 543 for O/E; C not logit |
| External, pooled across practices (R) | orig | orig | **log** (0.194–0.955 vs 0.194–0.958) | **logit** (0.716–0.886 vs 0.715–0.886) | matches Snell18 |
| Recalibrated, external (R) | orig | orig | **log** (0.475–2.160 vs 0.475–2.163) | logit | matches Snell18 |

- IECV df/k: with SE_HK = CI half-width/t_{k−1}, the PI formula μ̂ ± t_{k−2}√(τ² + SE_HK²) reproduces all four IECV PIs with k=6 (sum of absolute errors 0.008 on the analysis scales; k=5 with t_{k−2}: 0.502; k=5 with t_{k−1}: 0.044; k=7: 0.258) [INFERRED]. Conventional-SE variant is indistinguishable (implied q ≈ 1).
- For k ≈ 80–455 the t_{k−1} vs t_{k−2} choice is numerically immaterial; implied multipliers ≈ 1.97–2.02 (> 1.96) consistent with t quantiles and 2–3 dp rounding [INFERRED].

---

## 8. Riley 2019 development sample size (binary) — formulas and eFalls Table S2.1

### 8.1 Formulas [EXPLICIT: Riley19 equation numbers; file line]
- Van Houwelingen shrinkage S_VH = 1 − p/LR (eq 3, line 118); LR = −n ln(1−R²_CS,app) (eq 5, line 126); S_VH = 1 + p/(n ln(1−R²_CS,app)) (eq 7, line 130).
- **Criterion (i)**: n = p / [(S − 1) ln(1 − R²_CS,adj/S)], S = 0.9 (eq 11, line 159). Example p=20, R²=0.1 → 1698 [VERIFIED].
- ln L_null = E ln(E/n) + (n−E) ln(1−E/n) (eq 12); **max R²_CS** = 1 − exp(2 ln L_null/n) = 1 − [φ^φ (1−φ)^(1−φ)]² (eq 23, line 320) — independent of n [INFERRED algebra]; φ=0.5/0.05/0.01 → 0.75/0.33/0.11 [VERIFIED]; φ=0.048 → 0.31966.
- Nagelkerke R² = R²_CS / max R²_CS; conversion R²_CS = R²_Nag·max R²_CS (eq 14–15, lines 225–228).
- **Criterion (ii)** (optimism in Nagelkerke R² ≤ δ=0.05): S_VH ≥ R²_CS,adj / (R²_CS,adj + δ·max R²_CS) (eq 26, line 356); then eq 11 with that S if > 0.9. Example R²=0.1, max 0.33 → 0.858 [VERIFIED].
- **Criterion (iii)** (overall risk): n = (1.96/δ)² φ(1−φ), δ = 0.05 (eq 27, line 376). φ=0.5 → 384.2 [VERIFIED]; φ=0.048 → 70.2 → 71.
- Final n = max of criteria; events = n·φ; EPP = nφ/p (Riley19 §2.4).
- pmsampsize.ado conventions [EXPLICIT]: criterion n's are `ceil` (lines 319, 339, 349); events displayed as `ceil(E_final)` (line 420); with `nagrsquared()` the Cox-Snell R² is computed as nag·max_r2a and **rounded to 3 dp** via `%4.3f` (lines 107–111); max_r2a in criterion 2 uses E1 = n1·φ (lines 326–327, identical value); output table shows Max_Rsq and Nag_Rsq with `%4.3f` (lines 328–330).

### 8.2 eFalls Table S2.1 (p=90, φ=0.048) and HTA Table 3 (p=108) [VERIFIED/INFERRED]
Hypotheses: H1 = pmsampsize nagrsquared() (R²_CS rounded 3 dp); H2 = Nag × max R²_CS rounded to 3 dp (0.320), product unrounded; H3 = exact.

| Row | Reported | H1 | H2 | H3 | Conclusion |
|---|---|---|---|---|---|
| S2.1 Nag 0.15 | 13,867 (666) | 16,421 (789) | 16,421 | 16,439 | **not reproducible with p=90**; exactly 13,867 (666) with p=76 (R²_CS 0.048) |
| S2.1 Nag 0.05 | 50,174 (2,409) | **50,174 (2,409)** | **50,174** | 50,227 | reproduced |
| S2.1 Nag 0.049 | 50,927 (2,445) | 50,174 | 51,207 | 51,262 | **not reproduced**; requires R²_CS = 0.015765 (Nag 0.04932 at φ=.048) |
| HTA falls 0.15 / 0.05 | 19,706 (946) / 60,209 (2,891) | **both** | **both** | no | reproduced |
| HTA home care φ=.013 0.15 / 0.05 | 50,616 (659) / 161,460 (2,099) | **both** | no | no | H1 |
| HTA care home φ=.008 0.15 / 0.05 | 72,268 (579) / 217,887 (1,744) | no | **both** | no | H2 |
| HTA mortality φ=.045 0.15 / 0.05 | 20,563 (926) / 62,781 (2,826) | no | **both** | no | H2 |

- Under every hypothesis criterion (i) dominates for eFalls (criterion ii S ≈ 0.50–0.75 < 0.9; criterion iii 71).
- Hypothesis for 50,927 [INFERRED, unverified]: pmsampsize `cstatistic()` route (cstat2rsq: 1e6 simulated obs, LP ~ N(0,1) non-events / N(√2Φ⁻¹(C),1) events, seed 123456; pmsampsize.ado lines 788–848). Expected C giving exactly 50,927 is 0.6623; with C=0.66 (eFI 12-month emergency hospitalisation C, `raw/efi/efi2016.txt` line 182) the Monte Carlo n = 52,471 ± 872 (range 50,233–54,556; 3% of runs ≤ 50,927) — plausible only as a tail outcome (`dev_ss_cstat_mc.py`). Its Nag_Rsq display would be ≈0.048–0.049.

---

## 9. Riley 2021 external-validation sample size — formulas and eFalls Table S2.2

### 9.1 Formulas [EXPLICIT]
- **O/E**: N = (1−φ)/(φ·SE(ln O/E)²) (Riley21 eq 5, line 373). pmvalsampsize: SE increased in steps of 0.0001 until exp(ln OE + 1.96 SE) − exp(ln OE − 1.96 SE) ≥ 0.2 → SE = 0.0510; n = ceil(...) (pmvalsampsize.ado lines 284–294).
- **Calibration slope**: I = N·[[E(a_i), E(b_i)],[E(b_i), E(c_i)]], a = e^{α+βLP}/(1+e^{α+βLP})², b = LP·a, c = LP²·a; SE(β) = √(I_α / (N(I_α I_β − I_αβ²))); N = I_α / (SE(β)² (I_α I_β − I_αβ²)) with α=0, β=1 (eq 6–7, lines 430–475; Box 1). pmvalsampsize: expectations = means over 1e6 simulated LP; SE = round(0.2/(2·1.96), 0.00001) = 0.05102; n = ceil (lines 532–555).
- **LP distribution `lpskewednormal(mean var skew kurt)`** calls `sknor` (pmvalsampsize.ado line 314). `sknor` = **Ramberg (1979) generalised lambda distribution** from a lookup table of (skew, kurtosis) pairs: X = λ1 + (u^λ3 − (1−u)^λ4)/λ2, u ~ U(0,1); LP = mean + √var·X (mirrored for negative skew); (0.5, 3) → λ = (−0.639, 0.2006, 0.0630, 0.2307); (1, 5) → (−0.533, 0.0340, 0.009695, 0.0285) (web_sknor.ado lines 49–69, 112–118; web_sknor.hlp) [EXPLICIT]. This explains the eFalls note that the "closest available pairs" were used (AA-S line 84). sknor's seed adds the current clock time (web_sknor.ado lines 71–75) → Stata runs are not exactly reproducible [EXPLICIT/INFERRED].
- **C-statistic**: smallest N with 2·1.96·SE_Newcombe(N) ≤ 0.1 (eq 11; ado lines 641–651).
- **sNB**: N = SE⁻²[sens(1−sens)/φ + w² spec(1−spec)/(1−φ) + w²(1−spec)²/(φ(1−φ))] (eq 13, line 824); ado: SE found by 0.0001 steps to width 0.2, then **rounded to 0.001 (=0.051)**, n = ceil (lines 566–582).
- Riley21 worked examples VERIFIED: O/E 384.5, 293.9, 3461; slope DVT 2407.6 [2406.6, paper used rounded I's]; C 1154, 302, 4252, 5125; sNB 545.5.

### 9.2 eFalls Table S2.2 (φ=0.048) [VERIFIED]

| Criterion | Inputs | Reproduced | Reported |
|---|---|---|---|
| O/E | CI width 0.2 → SE 0.051 | raw 7625.27 → ceil **7626** (367) | 7,625 (366) — off by one (floor/round) |
| Slope | sknor(−3.30, 0.690, 0.5, 3), exact quadrature | raw 10,881.0 → **10,882** (523); MC 1e6×20: 10,884 ± 27 | **10,882 (523)** ✓ |
| C | C=0.743, width 0.1 | **2,027** (98) | 2,027 (98) ✓ |
| sNB 10% | sens .34 spec .91 | raw 2288.97 → **2,289** (110) | 2,289 (110) ✓ |
| sNB 25% | sens .04 spec .99 | raw 519.09 → ceil **520** | 519 (25) — off by one |
| Overall | max | 10,882 (523) | 10,882 (523) ✓ |

- A normal LP N(−3.30, 0.690) would give 13,968; Fleishman power-method (skew .5, kurt 3) 10,904; Azzalini skew-normal 10,537 — only the Ramberg/sknor GLD reproduces 10,882 exactly (`ss_validation.py`, `slope_sknor.py`).
- Cross-checks HTA Tables 5–6 (same code): care home O/E 47,674 ✓, C 6,025 ✓, sNB 6,403 ✓ / 1,501 ✓, slope exact 30,255 vs 30,124 (MC range 30,001–30,503 ✓); mortality O/E 8,160 ✓, C 1,672 ✓, sNB 2,525 ✓ / 2,133 ✓, slope exact 6,443 vs 6,455 (MC range 6,401–6,498 ✓). Mortality rows require `ceil` (8159.26→8160; 2132.32→2133) whereas eFalls O/E and sNB25 need floor/round → falls table apparently produced with a different rounding/version [INFERRED].

---

## 10. Model updating: intercept-only and logistic recalibration

- Intercept update ("recalibration-in-the-large"): new intercept = old intercept + CITL (α̂ from offset model); coefficients unchanged (VC19 add-file lines 24–27) [EXPLICIT]. Resulting apparent CITL = 0 exactly; O/E ≈ 1 (not exactly) [VERIFIED synthetic].
- Logistic recalibration (intercept + slope): fit logit p = a + b·LP; updated model LP* = a + b·LP ⇒ every coefficient × b, intercept = a + b·(old intercept) (AA-S equation `supp_fulltext_with_math.txt` line 308; AA line 134; VC19 add-file line 29) [EXPLICIT/INFERRED algebra]. Caution: the VC19 add-file ROMA example multiplies predictor coefficients by 1.014 but reports intercept −11.11 = −12.0 + 0.89 (unscaled), which is inconsistent with a + b·α_old = −11.28 [INFERRED].
- eFalls Box S3.2: p = expit(−0.423 + 1.25·LP), example LP −1.368 → 0.1059 [0.106] ✓; β_recal 1.25 ≈ external overall calibration slope 1.248 (AA Table 2 line 203) [VERIFIED/INFERRED].
- HTA Table 35: all 75 numeric coefficients = 1.2100 × HTA Table 15 (ratio range 1.20999999–1.21000001); constant −7.25089539 = α + 1.21·(−5.954459) ⇒ α = −0.046000 exactly [VERIFIED `verify_all.py` §4]. So HTA recalibration: (α, β) = (−0.046, 1.21) on the Table S3.2/15 LP, vs AA Box S3.2 (−0.423, 1.25) → unresolved contradiction (known issue ii). HTA also reports Connected Bradford n = 88,947 with 3,079 (3.5%) fall/fracture events (HTA lines 186, 459, 676) while AA/HTA falls chapter use 81,685 with 2,389 (HTA lines 903, 1654) [EXPLICIT] — differing external datasets could explain different recalibration parameters [INFERRED, unverified].

---

## 11. Minimal implementation recipe (Python) [INFERRED from the above]
1. Per dataset/cluster: LP → CITL (offset GLM), slope (GLM), O/E = ȳ/p̄, C (Mann–Whitney) with SEs: CITL/slope from GLM information; ln(O/E) SE √((1−φ)/O); C SE Newcombe (or DeLong), logit-C SE = SE(C)/(C(1−C)).
2. Calibration plot: 20 equal-size groups of p̂ (group means of p̂ and y); smooth: Stata-lowess bwidth 0.8 (Stata-side) or R loess span 0.75 / lowess iter=0 (R-side); pmcalplot-2.2.2 emulation = `running` span(1).
3. NB at p_t ∈ [0.10, 0.25] (and 0–0.5 for plots); treat-all; sNB = NB/φ.
4. Bootstrap (B = 25 per AA; 50 per HTA; ≥200 recommended) repeating FP selection + LASSO-CV; optimism = boot − orig; corrected = apparent − mean optimism; store p̂_bi for instability/MAPE/classification index (≥ threshold = high).
5. IECV over 6 WIMD groups (5 fifths + missing), full re-development each cycle.
6. RE-MA: REML τ²; HKSJ CI t_{k−1}; PI t_{k−2}·√(τ² + Var) (Stata convention; for eFalls IECV SE_HK or conventional both fit). Scales as used by eFalls (table §7.4) and alternatively Snell18 (slope/CITL raw, log O/E, logit C).
7. Recalibration: glm(y ~ LP) → (a,b); intercept-only: glm(y ~ offset(LP)).

---

## 12. Unresolved / contradictions (summary)
1. 25 (AA) vs 50 (HTA) bootstrap samples; R 4.2.3 (AA) vs R 4.3.1 (HTA).
2. Pooling scales: HTA says O/E on original scale; AA Table 2 shows original (development practices), log (IECV and external). IECV C pooled on original scale, not logit (contrary to Snell18 and to dev/external).
3. Table S2.1 rows 1 (13,867; matches p=76 not 90) and 3 (50,927; not reproducible from Nag 0.049 under any tested conversion).
4. Table S2.2 O/E 7,625 and sNB25 519 are one below pmvalsampsize 1.0.1 `ceil` results (7,626; 520).
5. Old pmcalplot lowess bandwidth; smoother/span used in R external validation; Stata `xtile` percentile rule details.
6. SE/CI methods behind Table S3.3 (C, O/E) and Table S3.4 (CIs excluding point estimates) not stated.
7. Calibration instability plot: bootstrap models applied in bootstrap sample (AA Fig S3.4 caption) vs original data (HTA line 513; RC23).
8. HTA recalibration (α −0.046, β 1.21) vs AA Box S3.2 (α −0.423, β 1.25).
