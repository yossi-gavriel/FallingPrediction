# Stata 17 defaults to emulate for eFalls: lasso logit + fractional polynomials

Compiled 2026-09-14. Scope: what Stata 17 does by default for (a) "multivariable logistic regression with LASSO penalty, lambda chosen to minimise the cross-validation function on 10-fold cross-validation" and (b) "second-order fractional polynomials with functional forms chosen in the presence of all predictors" (Archer et al., Age Ageing 2024;53(3):afae057, CC-BY), and how to map that to Python.

Evidence labels used below:
- **[EXPLICIT]** stated in the cited source (page numbers are the printed manual page numbers).
- **[VERIFIED]** additionally reproduced numerically here from Stata's own published example data/output (scripts listed in section 0).
- **[INFERRED]** my derivation/interpretation; not stated verbatim anywhere.
- **[NOT FOUND]** searched, not documented in any source I could access.

Copyright note: the Stata manuals are copyrighted (StataCorp). Evidence is given mostly as close paraphrase with page numbers and only very short verbatim fragments in quotation marks. Formulas are restated mathematically.

---

## 0. Sources, local copies, verification scripts

| Source | URL | Local copy (raw/) |
|---|---|---|
| Stata 17 [LASSO] Lasso Reference Manual, Release 17 (full manual, 379 pp.; copyright 2021, PDF built Oct 2022) | https://www.stata.com/manuals17/lasso.pdf | `raw/stata17_lasso.pdf`, text `raw/s17_lasso_full.txt` |
| Stata 19 [LASSO] manual (current; used only for a clarification on weights) | https://www.stata.com/manuals/lasso.pdf | `raw/stata_lasso_current.pdf`, `raw/cur_lasso_full.txt` |
| Stata 17 [R] Base Reference Manual, Release 17 (3077 pp.) — entries fp (pp. 721–744), fp postestimation (745–), mfp (1544–1556), mfp postestimation (1557–), bsample (191–), bootstrap (142–), set rng (2524–) | https://www.stata.com/manuals17/r.pdf | `raw/stata17_r.pdf`, `raw/s17_r_full.txt`, `raw/s17_mfp.txt` |
| (the per-entry URLs https://www.stata.com/manuals17/rfp.pdf, rmfp.pdf, lassolassofitting.pdf etc. return only 6-page previews) | | `raw/s17_*.pdf` |
| Stata 17 [U] ch. 11 (factor variables, base levels §11.4.3.2) | https://www.stata.com/manuals17/u11.pdf | `raw/s17_u11.txt` |
| Stata 19 [R] fp / mfp (current, cross-check only) | https://www.stata.com/manuals/rfp.pdf, https://www.stata.com/manuals/rmfp.pdf | `raw/cur_rfp.txt`, `raw/cur_rmfp.txt` |
| scikit-learn 1.9.1 user guide, logistic regression | https://scikit-learn.org/stable/modules/linear_model.html | (WebFetch) |
| scikit-learn 1.9.1 LogisticRegression API | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html | (WebFetch) |
| scikit-learn 1.9.1 LogisticRegressionCV API | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html | (WebFetch) |
| scikit-learn 1.9.1 StandardScaler API | https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html | (WebFetch) |
| glmnet vignette (Hastie et al., Stanford) — cross-reference only | https://glmnet.stanford.edu/articles/glmnet.html | (WebFetch) |
| lassopack `lassologit` (Ahrens, Hansen, Schaffer; community-contributed, NOT official Stata) | https://statalasso.github.io/docs/lassopack/lassologit/ ; paper https://arxiv.org/pdf/1901.05397 ; Stata Journal https://journals.sagepub.com/doi/10.1177/1536867X20909697 | (WebFetch) |
| Stata example datasets used for numeric verification | https://www.stata-press.com/data/r17/auto.dta, .../fakesurvey_vl.dta, .../smoking.dta, .../brcancer.dta | `raw/*_r17.dta`, `raw/fakesurvey_vl.dta` |
| eFalls paper + supplement (local authoritative copies) | PMC10960070 / doi:10.1093/ageing/afae057 | `../efalls/fulltext.md`, `../efalls/supp.md`, `../efalls/supp_media/image1.png`, `image2.png` |

Verification scripts (pure numpy/scipy; a minimal .dta reader, no Stata needed):
- `research/dta118.py` — reader for .dta formats 117/118/119 (numeric + str# columns).
- `research/verify_lasso_auto.py` — reproduces [LASSO] lasso auto example (λmax, λ grid, standardized + penalized coefficients at the CV-selected λ).
- `research/verify_lasso_logit_fakesurvey.py` — reproduces λmax, covariate count and N of [LASSO] lassoknots Example 2 (`lasso logit`).
- `research/verify_fp_logit_smoking.py` — reproduces [R] fp Example 3 (`fp <cigs>, zero: logit ...`) comparison table and coefficients; checks automatic scaling rule on [R] mfp Example 1.

---

## 1. What the eFalls paper states (CC-BY; local copies)

- Software: model development and internal validation in Stata 17; external validation in R 4.2.3. (fulltext.md, Statistical analysis) [EXPLICIT]
- Model: logistic regression with LASSO penalty; clustering by practice not accounted for at development; "lambda, was chosen to minimise the cross-validation function on 10-fold cross-validation". (fulltext.md, Model development) [EXPLICIT]
- FP: "Continuous predictors (age and polypharmacy) were modelled using second-order fractional polynomials, with functional forms chosen in the presence of all predictors. Transformed age and polypharmacy terms were then used as candidate predictors in the LASSO regression." [EXPLICIT]
- Figure S3.1 caption: best fitting functional forms "after adjusting for other covariates in the complete model (no variable selection)". Plots: age = "Fractional polynomial (linear), adjusted for covariates"; polypharmacy = "Fractional polynomial (natural logarithm), adjusted for covariates"; y-axis "Partial predictor + residual of fall/fracture". (supp.md; image1.png, image2.png) [EXPLICIT]
- Table S3.2 reports two columns: "Final penalised model, Coefficient" and "Unpenalised model, Odds Ratio (95% CI)" (unpenalised logistic refit with only LASSO-selected predictors). Terms: `Age (years)` 0.0415506; `ln((Polypharmacy+1)/10)` 0.3296295; categorical variables shown with a "Reference" level (Male, Overweight, Ex/never smoker, Lower risk drinking); Constant −5.954459 (penalised) / 0.003 (unpenalised). 75 predictors retained. [EXPLICIT]
- Internal validation: 25 bootstrap samples with replacement, full development process repeated. [EXPLICIT]

---

## 2. Stata 17 `lasso logit` — exact documented behaviour

Command presumed: `lasso logit y othervars [, rseed(#)]` (official Stata 17 `lasso`; the phrase "cross-validation function" is the official manual's terminology). The paper does not name the command → see Unresolved. [INFERRED]

### 2.1 Objective function [EXPLICIT + VERIFIED]
Stata 17 [LASSO] lasso, Methods and formulas, eq. (1), p. 158:

  Q_L = (1/N) Σ_i w_i f(y_i, β0 + x_i β') + λ Σ_j κ_j |β_j|

with, for logit (p. 159), f(η) = −y_i η + ln{1 + exp(η)} — i.e. the **per-observation negative log-likelihood (natural log), averaged over N** (not the deviance; no factor 2; no factor 1/2). For linear models f = ½(y − η)², so the linear objective is (1/2N)·RSS + λ‖β‖₁ (Remarks, p. 145).
- Stata 19 manual clarifies the weights: normalized weights w̃_i sum to 1; without weights w̃_i = 1/N (current lasso.pdf, Methods and formulas, "Lasso and elastic-net objective functions"). [EXPLICIT]
- κ_j (penalty loadings / coefficient-level weights) default to 1; only changed via the programmer's option `penaltywt()` or by adaptive lasso (Options p. 145; M&F p. 159). [EXPLICIT]
- β0 (intercept) appears only in the loss, **not in the penalty sum** (j = 1..p covers slopes only) → intercept is unpenalised. [EXPLICIT from formula]
- VERIFIED: the logit λmax of lassoknots Example 2 is reproduced exactly only with this scaling (section 2.11), confirming no factor 2 for logit.

### 2.2 Standardisation and reporting scale [EXPLICIT + VERIFIED]
- Before minimising, the columns of X are standardized to mean 0 and SD 1, so the penalty is not scale-dependent (lasso Remarks "Penalized and postselection coefficients", p. 152; M&F "Coordinate descent", p. 160). There is **no option in Stata 17 `lasso` to turn standardisation off** (none listed in the syntax table p. 138). [EXPLICIT; absence of option = checked syntax table]
- SD denominator: **not stated** in the manual [NOT FOUND in text], but **VERIFIED = population SD (divide by N, ddof=0)**: with ddof=0 the auto example reproduces λmax = 4.691140 (Stata 4.69114) and all 8 standardized/penalized coefficients to 6–7 significant digits; ddof=1 gives λmax 4.657023 and visibly different coefficients. Logit example: ddof=0 gives .0886291 (= Stata), ddof=1 gives .0885806.
- Three coefficient types (p. 152–153): `standardized` (coefficients on standardized X as estimated), `penalized` (same fit with the standardization "unwound", i.e. on original units — this is `e(b)`), `postselection` (ordinary estimator refit on selected variables). Unwinding verified: β_raw,j = β_std,j / sd_j; β0,raw = β0 − Σ β_raw,j·mean_j (auto example _cons 42.62583 reproduced).
- Indicator (factor-level) columns are standardized like any other column (auto example: `0.foreign` std coef 1.49568 ↔ penalized 3.250554). [VERIFIED]
- Stored results (p. 158): `e(b)` penalized unstandardized; `e(b_standardized)`; `e(b_postselection)`. [EXPLICIT]
- `predict` default after lasso = penalized coefficients; for logit/probit/Poisson the manual recommends penalized because there is no theory for postselection predictions (p. 154–155). [EXPLICIT]

**Implication for eFalls Table S3.2:** "Final penalised model, Coefficient" = Stata `e(b)` penalized unstandardized coefficients (original units, e.g. per year of age). [INFERRED, consistent with units in table]

### 2.3 λ grid [EXPLICIT + VERIFIED]
- Default number of grid points #g = **100**; logarithmic grid: ln λ_i = [(i−1)/(n−1)]·ln r + ln λgmax (Options `grid()`, p. 144; M&F p. 161). [EXPLICIT]
- λgmax = the smallest λ for which all (non-alwaysvars) coefficients are zero; cannot be reset by the user (p. 144; lasso fitting p. 210). [EXPLICIT]
- ratio r = λgmin/λgmax default **1e−4 when p < N**, **1e−2 when p ≥ N** (p counts othervars + alwaysvars, excluding constant) (p. 144). eFalls (p ≈ 80–100 columns, N ≈ 660k) → r = 1e−4. [EXPLICIT; application INFERRED]
- Closed form for λgmax is **not printed** in the manual [NOT FOUND], but by KKT for Q_L with standardized X and unpenalised intercept: λgmax = max_j | (1/N) Σ_i z_ij (y_i − ȳ) | (same for linear with centered y). [INFERRED → VERIFIED]: auto linear 4.691140 vs Stata 4.69114; fakesurvey logit .0886291 vs Stata .0886291; λ2 = λ1·10^(−4/99) gives 4.274392 and .0807555, matching Stata; λ42 = .1034458 matches.
- (Tie note: in the logit example `0.q90` and `1.q90` tie at λmax — both levels of a binary factor have identical |score| after standardization; Stata reported `0.q90` entering.) [VERIFIED]

### 2.4 Fitting algorithm, tolerances, stopping [EXPLICIT]
- Coordinate descent (Friedman, Hastie & Tibshirani 2010; Hastie, Tibshirani & Wainwright 2015), with IRLS-type extension for logit; warm starts along the grid from λgmax (M&F p. 160–161; lasso fitting p. 211).
- `tolerance(1e-7)`: convergence when the largest relative change in coefficients < 1e−7 (p. 145). `dtolerance(#)` alternative based on deviance change (not default).
- `stop(1e-5)`: the λ iteration stops when the **relative change in in-sample deviance** between adjacent λ's, (dev_{k−1} − dev_k)/dev_{k−1}, is < 1e−5; that λ is λstop (p. 144; lasso fitting p. 212). Not a CV rule. `stop(0)` disables.
- `cvtolerance(1e-3)`: for **nonlinear models (logit)** a CV minimum is "identified" only when **five** smaller λ's have CV values larger than the nominal minimum by a relative difference ≥ 1e−3 (three for linear) (p. 144–145; lasso fitting p. 213; lasso examples p. 205). The exact bookkeeping (consecutive vs any five; relative to the running minimum) is not given [partially NOT FOUND].
- Default `selection(cv)` evaluates the CV function λ by λ and **stops once a minimum is identified**; `alllambdas` fits the whole path first; the manual states the selected λ* is the same either way (p. 141). [EXPLICIT]
- If no identified minimum: default `stopok` → λ* = λstop if the stop() rule was met, else error; `strict` → error; `gridminok` → λstop or λgmin (p. 141; lasso fitting table p. 214). [EXPLICIT]

### 2.5 Cross-validation [EXPLICIT]
- `selection(cv)` is the default selection method; `folds(10)` is the default number of folds (p. 141). Paper's "10-fold" = default.
- CV algorithm (M&F "How CV is performed", p. 162–163): randomly partition data into K folds; for each fold k estimate the model for each θ (= λ grid value) on observations not in fold k; compute out-of-sample deviance for fold-k observations; CV function = **mean of the out-of-sample deviance**; λ with smallest mean minimises the CV function.
- lassoknots M&F (p. 255): the held-out fit measures use the **penalized** coefficient estimates from the training folds at that λ; observation deviance D_i = −2(ℓ_i − ℓ_saturated), ℓ_saturated = 0 for logit; mean deviance D̄ = (1/N)ΣD_i; deviance ratio D² = 1 − D̄/D̄_null. Glossary (p. 369–370): for nonlinear models the CV function is the **CV mean deviance**. Example output labels the CV column "CV mean deviance" (p. 204, 250).
  - [VERIFIED consistency]: at λgmax (intercept-only) the logit example's CVF = 1.386903 vs in-sample null mean deviance 1.386294 (ȳ≈0.5) → CV function is on the deviance (−2·ℓ/N) scale, not log-loss.
- The CV function is not the deviance ratio; `cvdevratio` is only a display statistic (lassoknots p. 241–). [EXPLICIT]
- `serule`: selects the largest λ whose CV function is within one standard error of the minimum (Hastie, Tibshirani & Wainwright 2015, pp. 13–14) (p. 141). The formula for that standard error is **not given** [NOT FOUND]. Not the default; paper says "minimise" → default min rule.
- Weighting of folds: M&F says mean over observations; Remarks (p. 149, lasso examples p. 199) describe averaging the K fold-level errors. Equivalent when folds are equal-sized; tiny difference otherwise. [EXPLICIT; minor internal inconsistency — also Remarks describe refitting "a linear regression ... using the variables in the model for that λ", which conflicts with M&F/lassoknots M&F (penalized estimates). M&F taken as authoritative.]
- `cluster(clustvar)` would split folds by cluster; eFalls did **not** account for clustering at development → folds on individuals. [EXPLICIT (option) / INFERRED (not used)]
- Whether covariates are **re-standardized within each training fold** (fold-specific means/SDs) vs full-sample standardization: **NOT FOUND**.
- Whether the λ grid used in folds is the full-data grid: implied ("for each value of λ", lassoknots M&F p. 255) [INFERRED: same grid values].

### 2.6 Random numbers / fold assignment [EXPLICIT + NOT FOUND]
- `rseed(#)` is equivalent to `set seed #` before `lasso`; needed to reproduce CV/adaptive results; different seeds can change λ* (p. 144; p. 151). e(rngstate) is stored.
- Stata's default RNG is the 64-bit Mersenne Twister `mt64` ([R] set rng, p. 2524).
- The algorithm mapping random numbers to fold membership (e.g. sort-by-uniform then split, stratified or not) is **NOT FOUND** in the documentation → Stata folds cannot be replicated bit-for-bit in Python; λ* (and the 75-predictor set) may differ slightly.

### 2.7 Factor variables / base levels [EXPLICIT + VERIFIED]
- lasso **does not set a base level** for factor variables among othervars: it creates indicators for **all** levels and adds them as candidates ([LASSO] lasso examples, "Factor variables in lasso", p. 203). If a base level is specified for an othervars factor (e.g. `ib3.group`), **it is ignored**; for alwaysvars (in parentheses) a base level is set/used ([LASSO] Collinear covariates, p. 38–39). If you build indicators yourself you must include all levels (p. 37).
- [VERIFIED] lassoknots Example 2 reports 277 covariates: reproduced exactly as (all levels of i.demographics + i.factors) + 29 continuous, N = 914 complete cases.
- Standard Stata estimation commands (e.g. `logit`): default base = smallest level; `ib#.` uses value # as base; `ibn.` no base; `ib3.group` → `1.group 2.group 3b.group` ([U] 11.4.3.2, u11.pdf). [EXPLICIT]
- Consequence for eFalls: the "Reference" labels in Table S3.2 penalised column cannot come from a Stata base level in `lasso` othervars; either (a) the reference level's indicator was not selected (coefficient 0), (b) the authors created their own indicator set omitting one level, or (c) the categorical variables were alwaysvars (unlikely: some levels have tiny coefficients, e.g. Zero alcohol 0.0070). **Unresolved.** [INFERRED]
- `noconstant` warning: with all levels present lasso can reproduce the constant (p. 140). [EXPLICIT]

### 2.8 Penalized vs postselection coefficients (paper's two columns) [EXPLICIT + INFERRED]
- Postselection = refit of the ordinary estimator (logistic regression for `lasso logit`) using the lasso-selected variables (lasso postestimation, predict options). [EXPLICIT]
- eFalls "Unpenalised model OR (95% CI)" = separate unpenalised logistic refit with reference categories (not necessarily Stata's `e(b_postselection)`, because a refit with all selected levels of a factor could be collinear). Stata logit CIs for ORs are the exponentiated Wald CI endpoints ([R] eform option, p. 499: SEs and CIs are also transformed). [EXPLICIT for eform; mapping INFERRED]

### 2.9 Numeric verification results (reproduced from Stata's published output)
- auto linear example (`lasso linear mpg i.foreign i.rep78 headroom weight turn gear_ratio price trunk length displacement`, N=69, p=15):
  - λmax: 4.691140 (Stata 4.69114); λ2: 4.274392 (4.274392); λ42: .1034458 (.1034458).
  - Coordinate descent on (1/2N)‖y−Zb‖² + λ‖b‖₁, Z standardized with ddof=0, at λ=.1034458: standardized coefs 0.foreign 1.495679 (1.49568), 3.rep78 −.3292315 (−.3292316), 5.rep78 1.293645 (1.293645), weight −.2804675 (−.2804677), turn −.7378128 (−.7378134), gear_ratio 1.378287 (1.378287), price −.2809066 (−.2809065), length −2.942433 (−2.942432); penalized _cons 42.62583 (42.62583). ddof=1 does not match.
- fakesurvey logit example (`lasso logit q106 $idemographics $ifactors $vlcontinuous`): N 914 (914), covariates 277 (277), λmax .0886291 (.0886291), λ2 .0807555 (.0807555); top-|score| covariates 0.q90/1.q90, 2.q134, 0.q142 — exactly the first knot variables Stata reports (0.q90, 2.q134, 0.q142).

### 2.10 Community-contributed alternative: lassopack `lassologit` [EXPLICIT]
- Objective: (1/N)Σ loglik − (λ/N)‖β‖₁ (with optional penalty loadings); intercept not penalised by default; coordinate descent (FHT 2010) + strong rules; `cvlassologit` for K-fold CV by deviance or misclassification (https://statalasso.github.io/docs/lassopack/lassologit/).
- Hence λ_lassologit = N·λ_official for the same data/standardisation [INFERRED]. If the authors had used lassologit, their λ values would be on that scale. Not determinable from the paper.

---

## 3. Mapping to Python

### 3.1 λ (Stata) ↔ C (scikit-learn) [EXPLICIT formulas + INFERRED algebra]
scikit-learn 1.9.1 user guide (logistic regression, binary case) states the optimisation problem as

  min_w (1/S) Σ_i s_i ℓ_logloss(y_i, X_i w + c) + r(w)/(S·C),  S = Σ s_i,  r(w) = ‖w‖₁ for l1_ratio=1

(older docs: min ‖w‖₁ + C Σ_i ℓ_i — identical after dividing by C·N). With unit weights S = N:

  (1/N) Σ ℓ_i + (1/(N·C)) ‖w‖₁   vs   Stata: (1/N) Σ f_i + λ ‖β‖₁,  f_i = ℓ_i (natural-log negative log-likelihood)

→ **λ = 1/(N·C), C = 1/(N·λ)**, where N is the number of observations in the sample being fitted (the training folds during CV, the full sample for the final fit), and X must be standardized exactly as Stata does (mean 0, ddof=0 SD) so that w corresponds to Stata's standardized coefficients. Note the sklearn doc remark that multiplying sample weights by b is equivalent to multiplying C by b. [EXPLICIT doc; mapping INFERRED]
- glmnet (R) uses the same scaling as Stata for binomial: −(1/N)Σ loglik + λ[(1−α)‖β‖²/2 + α‖β‖₁]; α=1 → lasso; default nlambda 100; λmax = smallest λ with all coefficients zero; coefficients returned on original scale; `cv.glmnet` default type.measure "deviance" for binomial, nfolds 10 (https://glmnet.stanford.edu/articles/glmnet.html). So **Stata λ ≈ glmnet λ** (same data/standardisation) [INFERRED], a useful cross-check (glmnet's default lambda.min.ratio should be checked separately — not extracted here).

### 3.2 Solver choice [EXPLICIT]
- `penalty` is deprecated since sklearn 1.8 (removed 1.10): use `l1_ratio=1` (pure L1) and `C` (API page).
- Only `liblinear` and `saga` support L1 (user-guide solver table).
- `liblinear` **penalises the intercept** (table: "Penalize intercept (bad)"); implemented as a synthetic constant feature of value `intercept_scaling` whose weight is regularised like any other (API `intercept_scaling` note). → Not equivalent to Stata; only approximately with a very large `intercept_scaling`.
- `saga`: does not penalise the intercept; fast convergence only guaranteed with features on approximately the same scale (standardise); defaults `tol=1e-4`, `max_iter=100` are far looser than Stata's `tolerance(1e-7)` → need much tighter tol / more iterations; still a stochastic method (random_state).
- `LogisticRegressionCV` defaults are NOT Stata-like: Cs=10 values log-spaced 1e−4..1e4; cv default = StratifiedKFold with 5 folds; scoring default accuracy (to change to 'neg_log_loss' in 1.11); refit=True averages fold scores and refits at best C; a single C grid applies to every fold (API page). Because Stata's λ is per-observation-scaled, a fixed C in folds of size ≈0.9N corresponds to a λ ≈ 1.11× the full-data λ → do the CV loop manually with C_fold = 1/(N_train,fold·λ). [EXPLICIT defaults; consequence INFERRED]
- Recommended for fidelity [INFERRED]: implement glmnet-style IRLS + cyclic coordinate descent with warm starts for Q_L directly (the algorithm Stata's M&F cites), tolerance 1e−7 on relative coefficient change; use sklearn/saga only as a cross-check. The auto-example script shows a plain coordinate-descent implementation reproduces Stata's linear lasso to ~1e−6.

### 3.3 Standardisation in Python [EXPLICIT + VERIFIED]
- `StandardScaler` uses the biased SD, equivalent to `numpy.std(x, ddof=0)`, z = (x − u)/s (sklearn API page) — matches Stata (verified ddof=0).
- Standardise every candidate column, including every one-hot level (drop=None, i.e. all levels, no base) and the FP-transformed columns.
- Report penalized coefficients on original scale: β_raw = β_std/sd; intercept β0_raw = β0 − Σ β_raw·mean.
- Because each FP term is standardized before penalisation, shifting/scaling a single FP column (e.g. ln((P+1)/10) = ln(P+1) − ln 10, or centering) changes only the intercept, not selection or slopes. [INFERRED, follows from standardization]

### 3.4 Emulating `lasso logit, selection(cv)` defaults (pseudo-procedure) [INFERRED from EXPLICIT rules]
1. Build X (all levels of categorical vars; FP terms), drop rows with missing (Stata uses complete cases on model variables; verified N=914).
2. Standardize with ddof=0; λmax = max_j |Z_jᵀ(y−ȳ)|/N; grid λ_i = λmax·(1e−4)^((i−1)/99), i=1..100.
3. Path fit on full data from λ1 with warm starts; after each λ compute in-sample mean deviance; stop when relative change < 1e−5 (λstop).
4. Random 10-fold partition (seeded; cannot match Stata's). For each fold and each λ on the path: fit on training folds at that λ (penalized), compute held-out deviance −2ℓ_i; CV(λ) = mean over all observations.
5. Minimum identified at λ_k when at least five subsequent smaller λ's have CV larger than CV(λ_k) by relative difference ≥ 1e−3; λ* = λ_k. If none and step-3 stop reached → λ* = λstop (stopok).
6. Report penalized (unstandardized) coefficients at λ*; separately refit an unpenalised logistic regression on selected variables with chosen reference levels, Wald 95% CI exponentiated.

### 3.5 What cannot be matched exactly
- Fold assignment / RNG stream (algorithm undocumented; mt64) → λ* and borderline selected variables may differ. [NOT FOUND]
- Exact bookkeeping of the cvtolerance "five smaller λ" rule and of `stopok` edge cases. [partially NOT FOUND]
- Whether standardisation is recomputed inside each CV training fold. [NOT FOUND]
- SE formula used by `serule` (not needed for eFalls). [NOT FOUND]
- Floating-point/convergence differences (Stata tolerance 1e−7) → ~1e−6 relative coefficient differences (observed in verification).
- The authors' actual candidate set coding (indicator construction, reference handling), data, and seed are not published → exact coefficient reproduction is impossible without their code. [INFERRED]

---

## 4. Fractional polynomials in Stata 17

### 4.1 Definition and defaults of `fp` ([R] fp, pp. 721–744) [EXPLICIT + VERIFIED]
- Syntax: `fp <term> [, options]: est_cmd` — `<term>` is replaced wherever it appears in est_cmd by the FP power variables `term_1, term_2, ...`; est_cmd can be almost any command storing `e(ll)` (p. 722). (Note: `lasso` does not store `e(ll)`; it stores `e(ll_sel)` — so FP selection must have been run with an ordinary `logit` or via `mfp`.) [EXPLICIT; implication INFERRED]
- `powers()` default **{−2, −1, −0.5, 0, 0.5, 1, 2, 3}**; `dimension()` default **2** (max degree); with defaults the search covers **44** FP models (8 FP1 + 36 FP2 incl. repeated powers) (p. 723–724). [VERIFIED: 44]
- x^(0) means ln(x). Repeated powers: each repeat multiplies by another ln(x): H1 = x^(p1); Hj = x^(pj) if pj ≠ pj−1, else H_{j−1}·ln(x) (p. 727; M&F p. 742). E.g. FP2 (p,p) = β1 x^p + β2 x^p ln x; (0,0) = β1 ln x + β2 (ln x)². [VERIFIED via smoking example powers (−1,−1)]
- Default: **no scaling and no centering** (Examples p. 734: by default fp does not scale or center); nonpositive values → error unless `scale`, `zero`, or `catzero` (p. 725). `classic` = scale + center + nocompare. [EXPLICIT]

### 4.2 Choice of "best" model and the comparison table [EXPLICIT + VERIFIED]
- Deviance D = −2·(maximised log likelihood) (p. 729; M&F p. 743).
- fp reports as "best" the **lowest-deviance** model, which is always the highest-degree (m = dimension) model; the comparison table lists rows `omitted` (term dropped), `linear` (power 1, untransformed), `m = 1` (best FP1), `m = 2` (best FP2), each with deviance, deviance difference vs the lowest-deviance model, test df and P (p. 729–730). Users may instead choose the lowest-degree model not rejected by the test (p. 730).
- Test df vs FP2 (default dimension 2): omitted 4, linear 3, m=1 2 (df count both coefficients and powers; approximate, generally conservative) (p. 743). For non-normal models (logit) P = Pr(χ²_df > D_k − D_m) (p. 743). For regress an F test is used.
- [VERIFIED] smoking example (`fp <cigs>, zero: logit all10 <cigs> nonsmoker age`): omitted 9990.804, linear 9958.801 (P .003), m=1 power 0 9946.603 (P .388), m=2 powers (−1,−1) 9944.708; coefficients −1.285867, −1.982424, −1.223749, .1194541, −1.591489 — all reproduced exactly.

### 4.3 Scaling, centering, zero/catzero; why ln((Polypharmacy+1)/10) [EXPLICIT + VERIFIED + INFERRED]
- `scale(a b)`: use (term + a)/b. `scale` (automatic): if term has nonpositive values, subtract its minimum and add the **counting interval** (minimum distance between sorted distinct values); then divide by 10^p*, where p = log10{max(term*) − min(term*)}, p* = sign(p)·floor(|p|) (p. 731). Variable itself is not modified; `notes` record a and b for `fp generate, scale(a b)` in new data (p. 725, 732).
- `center(c)` / `center`: report term^(p) − c^(p) (c = mean of scaled term by default when `center` given); changes only the intercept (p. 733).
- `zero`: FP terms set to 0 where scaled term ≤ 0; `catzero`: additionally adds indicator term_0 (p. 725, 741).
- **Polypharmacy**: count with min 0 (Figure S3.1 range 0 to ≈61) → counting interval 1 → term* = P + 1; max(term*) − min(term*) = max(P) = 61 → p = 1.785 → p* = 1 → divide by 10 → **X = (P+1)/10**, FP1 power 0 → **ln((P+1)/10)**, exactly the published term. The divisor is 10 for any max(P) in [10, 99]; 100 if max(P) ≥ 100. [INFERRED from EXPLICIT rule; rule VERIFIED on [R] mfp Example 1: brcancer x1 (21–80) → x1/10, x5 (1–51) → x5/10, x6 (0–2380) → (x6+1)/1000, all as printed by Stata p. 1552]
- **Age** (65–95, positive, no shift): automatic scale would give age/10; the published coefficient 0.0415506 is per year and the Box S3.1 example multiplies by age in years, so the final lasso used age in years (linear FP is scale-invariant apart from coefficient units). [INFERRED]
- Centering: the published penalised Constant −5.954 with uncentered age (years) and uncentered ln((P+1)/10) is consistent with no centering of the terms entered into the lasso (development-data linear predictor mean −3.30, reported in the Appendix S2 sample-size table, supp.md line ~78). [INFERRED]

### 4.4 `mfp` defaults and the closed-test function selection procedure ([R] mfp, pp. 1544–1556) [EXPLICIT]
- Syntax `mfp [, options]: regression_cmd yvar xvarlist`; allowed regression_cmd includes `logit`/`logistic` (p. 1545). Elements in parentheses are tested jointly and not FP-transformed.
- Defaults (options table p. 1545–1547):
  - `powers()` default −2 −1 −0.5 0 0.5 1 2 3 (0 = log).
  - `dfdefault(4)` = FP2 maximum (df = 2 × degree). Default df by number of distinct values: 2–3 → 1 (linear); 4–5 → min(2, dfdefault); ≥6 → dfdefault.
  - `alpha(0.05)`: significance level for testing between FP models of different degree.
  - `select()`: backward-elimination levels; **default 1 for all variables = all forced into the model (no variable selection)**.
  - `cycles(5)` max backfitting cycles; `xorder(+)` = process variables in decreasing significance from the initial all-linear model.
  - `center(mean)` by default (binary: lower value); `zero()`/`catzero()`; otherwise nonpositive predictors get the preliminary linear transformation of fp's `scale` option (p. 1547).
  - Default algorithm: **closed-test procedure** (RA2 / function selection procedure); `sequential` gives the older Royston–Altman sequence.
- Closed test for FP2 (p. 1549): (1) Inclusion: FP2 vs null, 4 df, at `select()` level (skipped when select = 1); (2) Nonlinearity: FP2 vs linear, 3 df, at `alpha()`; if not significant → linear; (3) Simplification: FP2 vs FP1, 2 df, at `alpha()`; significant → FP2, else FP1. Type I error close to nominal; sequential procedure about double (p. 1549).
- Backfitting (p. 1548): start with all terms linear; for each xvar in turn choose its FP function with all other variables in the model (others keep their current functional forms); repeat cycles until functional forms and included variables stop changing (usually 1–4 cycles).
- Iteration log rows `null / Lin. / FP1 / Final` with deviance differences vs FP2 (p. 1551–1553).

### 4.5 What "functional forms chosen in the presence of all predictors" most likely was [INFERRED]
Evidence:
1. Figure S3.1 titles "Fractional polynomial (linear|natural logarithm), adjusted for covariates" and y-axis "Partial predictor + residual of fall/fracture" match the style of **`fracplot`** after **`mfp`** ([R] mfp postestimation p. 1560 example: title "Fractional polynomial (-2 -.5), adjusted for covariates", y-axis "Partial predictor+residual of _t"), not `fp plot` (y-axis "Component+residual of ..." in [R] fp postestimation p. 749–750).
2. Caption "complete model (no variable selection)" matches mfp's default `select(1)` (all variables forced in) with all candidate predictors in the logit model.
3. Chosen forms are linear (age) and FP1 power 0 (polypharmacy) although "second-order" (FP2, dfdefault 4) was the maximum → a test-based simplification (closed test at alpha 0.05 by default) was applied; plain `fp` "best" would always be an FP2.
4. The term ln((P+1)/10) is exactly mfp/fp automatic preliminary scaling.
Likely call (reconstruction, NOT stated by the authors): `mfp, df(4) alpha(0.05) select(1): logit fall age polypharmacy <all other candidate predictors>` (mfp defaults), followed by `fracplot age` / `fracplot polypharmacy`, then generating uncentered terms (e.g. `fp generate`, or by hand) for the lasso. Alternative: `fp <poly>, scale: logit fall <poly> age ...` with manual choice from the comparison table. Neither alpha nor the exact command is reported. 

### 4.6 Python emulation of FP/MFP [INFERRED]
- FP transforms: implement H_j recursion (4.1) on the scaled variable; search 8 FP1 + 36 FP2; fit ordinary logistic regression (IRLS/Newton, converged tightly) with all other predictors; deviance = −2ℓ; χ² tests with df 4/3/2 (verified script).
- MFP closed test with backfitting as in 4.4; default alpha 0.05, select 1, cycles ≤5, xorder by significance of linear terms in the full linear model (Wald/LR p-value — the manual says "multiple linear regression" ordering; for logit the ordering statistic is from the fitted all-linear model [exact statistic NOT FOUND]).
- Automatic preliminary scaling: shift by −min + counting interval if min ≤ 0, then divide by 10^sign(p)floor(|p|), p = log10(range of shifted variable).
- Treat binary/categorical predictors as linear 1-df terms (or joint groups); only age and polypharmacy get FP search.

---

## 5. Paper-internal observations relevant to reproduction [INFERRED]
- Box S3.1 intercept −6.258 = Table S3.2 penalised Constant −5.954459 + Female −0.303708 (= −6.258167): Box uses female as baseline, but then lists "– 0.304 (if male)"; from Table S3.2 (Male reference, Female −0.304) the male term should be **+0.304** relative to a female baseline. Worked example (female) is unaffected.
- Box S3.1 uses rounded coefficients (e.g. 0.042 × age): example LP −1.368 → 0.203; with Table S3.2 full-precision penalised coefficients the same patient gives LP −1.4085 → p = 0.1965 (recalibrated: 0.1012 vs 0.106 in Box S3.2).

---

## 6. Unresolved
1. Which Stata command was used: official `lasso logit` (most likely; terminology) vs lassopack `lassologit`/`cvlassologit` (λ scaled by N). Not stated.
2. Stata fold-assignment algorithm and the authors' seed → CV folds cannot be replicated.
3. Whether Stata re-standardizes within each CV training fold.
4. Exact implementation of the `cvtolerance` "five smaller λ" rule.
5. SE formula for `serule` (not needed if default min rule used).
6. How categorical predictors were entered in lasso (factor-variable all-levels vs own indicators omitting a reference); meaning of "Reference" in the penalised column.
7. Whether FP selection used `mfp` (closed test, alpha?) or `fp` + comparison table; the significance level; whether other predictors entered mfp as 1-df linear terms/groups; whether terms were centered at that stage.
8. Maximum polypharmacy value (divisor 10 requires 10 ≤ max < 100; Figure S3.1 suggests ≈61) and age truncation at 95 (Figure S3.1 x-range 65–95).
9. glmnet default lambda.min.ratio not extracted (only relevant if using glmnet as a cross-check).
