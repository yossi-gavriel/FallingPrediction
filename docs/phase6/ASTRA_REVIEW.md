# Independent adversarial scientific review — Phase 5.1 / Phase 6

**Verdict: MODIFY THE ROADMAP BEFORE IMPLEMENTATION.** The methodological correction is necessary, but Claude's proposed correction is incomplete, its automatic AUROC exclusion should be removed, and its roadmap omits a material management requirement: faller profiling and phenotyping. New longitudinal information is a more credible improvement route than searching more algorithms or repeatedly optimizing Top3.

Reviewer: Astra/Codex, independent scientific reviewer. Date: **2026-10-07** (Asia/Jerusalem). This is a review and proposed protocol, not an implementation or authorization to train.

## 1. Executive verdict

### Provenance and limits of this review

Before reviewing, I inspected the original checkout: `main`, clean, at `add690c`. I fetched `origin` and verified that `origin/research/phase6-strategy` pointed to **`c375aba7c524449ed0e885e0a20e48b600d68913`**. Its immediate parent is **`2398fb9391f449330054f1b6900aceeec1b6a37b`**, the 0.12.3 handoff-package commit; that follows the 0.12.3 SQLite lifecycle fix `4574d8886c1787f451d2c621a20f866cd62be41f`. The parent's `VERSION` is `0.12.3`, and its delivery record identifies Phase 5 2.2.0. The lead commit adds exactly one file, `docs/phase6/CLAUDE_STRATEGY.md`; no source/config/test changes occur between that parent and the lead commit.

The review was written in a separate worktree attached to the strategy branch. The original checkout and other worktrees were preserved. No patient data, Windows run outputs, fitted clinical models, or real-data eligibility tables were read. No model was trained or retrained. Only this review file is added. Statements about achieved performance below are **user-supplied project facts**, not independently reproduced estimates.

The source inspection used the exact strategy revision and its unchanged Phase 5 base, especially:

| Evidence | Verified implication |
|---|---|
| `docs/phase5/PREPROCESSING_AUDIT_0.12.2.md`; companion 130-feature audit | Scaling/imputation are already nested; full-cohort eligibility is not. The catalogue size is not the number of retained design columns. |
| `src/falls_ml/phase5/data.py:161–168, 298–333, 367–409`; `runner.py:285–307` | Labels determine a full-cohort AUROC exclusion before folds and feature-set construction. Only usable 0/1-labelled patients enter the development frame. |
| `src/falls_ml/phase5/engine.py:200–241`; `models.py:23–54, 82–93, 141–169` | Lambda grids depend on outer-training labels, including inner validation. Linear designs are refitted inside inner training; XGB early stopping is inside inner training. |
| `src/falls_ml/phase5/design.py:29–149`; `configs/meuhedet/phase2_features.yaml` | CCI has a linear numeric effect; nominal subcodes can remain numeric in trees; prior-fall windows, MEFI previous/worsened variables, assessment counts and utilisation windows already exist. |
| `src/falls_ml/phase5/capacity.py:35–122, 198–268` | Exact capacity uses within-fold ranking and proportional allocation; its bootstrap explicitly conditions on fitted outer models. |
| `src/falls_ml/phase5/analysis.py:110–143`; `thresholds.py:objective` | The historical verdict concerns roughly 70% sensitivity; tuning also targets that region rather than 3% capacity. |
| `configs/meuhedet/phase5_v21_schema.yaml:17–35, 347–398`; `phase5_v21_view_definition.txt` | Event dates do not establish availability dates; counts may count records rather than episodes; assessment counts can count question rows; CCI semantics are undocumented. |
| `planning/PHASE3_TIME_CONTRACT_AUDIT.md`; `configs/meuhedet/phase3_time_contract.yaml`; `src/falls_ml/phase4/evaluate.py:97–174, 325–327` | Prediction is at end of index day; future outcomes and follow-up are separately audited. Unlabelled/censored patients are excluded from current metrics. |

### Decision in operational units

Use the supplied **97,453 patients**, **2,155 patients with recorded future falls**, and **2,924 available places**. The supplied 1,086 captures imply **50.39% recall**, **37.14% PPV**, and 1,838 flagged patients without a recorded future fall. One percentage point of recall is about **21.55 additional captured fallers**; these are patient-level first/binary outcomes, not counts of recurrent clinical episodes or prevented falls.

At least 55% on this same event denominator requires **1,186 captures**, about **100 more** than 1,086. That is a useful aspiration to translate into staffing and clinical benefit, not an evidence-based attainable ceiling or a stopping rule. Corrected development may move the baseline in either direction. Compare every improvement with the corrected ENET on the same population; do not compare new results only with the historical 50.4% headline.

**ACCEPT** correction before new claims, frozen temporal evaluation, provenance/privacy controls, and parallel data requests. **MODIFY** the correction, tuning/selection policy, validation timing, success rule, and data priorities. **REJECT** automatic high-AUROC exclusion as the primary leakage defense and the claim that a 0.75 triage margin establishes unbiasedness. **DEFER** broad algorithm expansion and complicated survival/ranking architectures until the data justify them. Add a parallel clinical-understanding track now.

## 2. What Claude got right

- **ACCEPT:** The audit correctly identifies full-cohort outcome-dependent eligibility as the main CV defect. Standardization is not the main problem. The existing inner-trained medians, levels, missing indicators, constant/duplicate removal, and scales should be retained.
- **ACCEPT:** Preserve historical outputs and protected Phase 2/3/4 files, use new output folders, hash source/config/input/model manifests, and disclose PRE-CORRECTION results. A SQLite lifecycle fix is not a scientific reason to retrain Phase 5.
- **ACCEPT:** Keep ENET as the principal comparator. The 22 V21 predictors have not established a robust Top3 gain; neither a significant coefficient nor a new column proves incremental information.
- **ACCEPT:** Anchor operational evaluation at a fixed contact capacity. At identical capacity, additional true positives replace false positives one-for-one. Captured is not prevented; intervention benefit needs separate evidence.
- **ACCEPT:** Distinguish internal development CV from later temporal validation; a reused January 2026 cohort is no longer an untouched test set. Same-HMO temporal validation is not independent external-site validation.
- **ACCEPT:** Request medication timing, diagnosis-specific dates, registry creation timestamps, label-source auditing, and schema-definition provenance. These address information and availability rather than cosmetic transformations.
- **ACCEPT:** Treat drift/calibration limits as investigation triggers rather than universal proof of success or failure. Do not fit recalibration to validation labels and then claim the recalibrated model passed that same validation.
- **ACCEPT:** Silent prospective validation and an explicit intervention pathway belong before clinical implementation. Broad neural/LLM expansion and unrestricted stacking are low priorities here.

## 3. Top scientific problems / risks

| Priority | Problem in Claude's proposal | Disposition and required resolution |
|---|---|---|
| P0 | C-01 learns eligibility on the entire outer training fold, then holds that eligibility fixed for inner CV. | **MODIFY:** This removes the direct outer-label path at that step but not inner-validation influence. Remove the screen, or nest it at both levels. |
| P0 | AUROC ≥0.80 is treated as proof of leakage. | **REJECT:** Strong legitimate predictors can exceed it; weak, joint, or availability leakage can evade it. Timing/lineage determines admissibility. |
| P0 | T-01's max-AUROC <0.75 margin rehabilitates historical estimates. | **REJECT:** No universal ±0.02 guarantee exists across missingness patterns, rare features and correlated measurements. Triage is evidence about whether exclusion happened, not a validity certificate. |
| P0 | Retaining the five-criterion historical verdict is treated as sufficient for the new operational question. | **MODIFY:** Preserve it as a historical audit result; preregister a separate 3% improvement rule. ALL with uncertain timing cannot be the confirmatory deployment candidate. |
| P0 | Non-overlapping outcome windows and late extraction are treated as enough for untouched validation. | **MODIFY:** Audit prior access, available July–October outcomes, predictor reconstruction and training-label availability at July scoring time. |
| P0 | Capacity is evaluated after excluding patients based on future label availability. | **MODIFY:** That is a labelled-subcohort estimand. Operational capacity must be allocated among everyone eligible at scoring time, before future follow-up is known. |
| P1 | Longitudinal feature development and an appropriate nonlinear comparator are postponed together to Phase 7. | **MODIFY:** Begin extraction and specifications now; permit a small bounded development plan before a separately sealed temporal test. Keep correction and improvement results distinct. |
| P2 | No faller profiling, faller-only clustering or population phenotyping protocol is specified. | **MODIFY — material omission:** Section 8 supplies three separate questions and stopping criteria. Subgroup calibration and feature importance do not answer them. |
| P0 | Repeated development on January 2026 may be presented as a fresh unbiased estimate after fixing the code. | **MODIFY:** Nested CV isolates a specified procedure, not the accumulated human adaptation to an already studied dataset. Fresh temporal confirmation remains necessary. |

A diagnosis-code outcome also measures detection and healthcare contact. Improving capture of documented falls can improve ascertainment ranking without improving prediction of all actual falls. Audit fracture-only coding, encounter setting, episode continuation and under-recording before strengthening the clinical claim.

## 4. Phase 5.1 correction review

### Direct answers to the correction questions

**1. Is the proposed correction sufficient? No.** C-01 fixes direct use of outer holdout labels by its screen, provided no fold-union exclusion or global membership artifact is reused. It leaves inner contamination, covariate-dependent global eligibility, semantic assumptions and availability leakage unresolved. Nor can it erase previous researcher adaptation. Inner contamination does not automatically bias an honestly isolated outer estimate of that particular training procedure; it makes inner OOF less reliable for choosing that procedure and contradicts a claim of complete nesting.

**2–3. What belongs at each boundary?**

| Operation | Before CV | Each inner-training fit | Outer-training refit |
|---|---|---|---|
| Fixed clinical catalogue, allowed source roles, documented code meanings, prespecified date/window rules | Declare without outcomes; raw forbidden fields remain sealed. | Apply fixed rules. | Apply identical rules. |
| Cohort definition and label quality | Declare cohort/estimand first; audit outcomes for integrity, not model-driven cohort edits. Stratification may use labels to form folds. | No retrospective success-driven exclusions. | Same cohort contract. |
| Rowwise history calculation | May be precomputed when parameter-free and as-of compliant. | No future records or fitted population statistics. | Same calculation. |
| Coverage/rare-level thresholds, constant/duplicate removal, imputation, scales, missing indicators | Summaries/quality-stop checks only; no global data-adaptive model membership. | Fit on inner training, transform validation. | Refit on outer training, transform holdout. |
| Any retained outcome screen or supervised feature selection | Never. | Learn from inner training only. | Relearn from outer training only. |
| Lambda path, knots/bins inferred from data, target encoding, PCA/FAMD, clustering | Prespecify recipe/search bounds only. | Fit using inner training; target encoding must also prevent within-training target leakage. | Fit selected recipe on outer training. |
| Hyperparameters, feature-bundle/family choice | Prespecify finite candidates/budget. | Select using inner OOF only. | Refit selected recipe; outer outcomes never choose it. |
| Early stopping | Prespecify rule. | Use a split within inner training; exclude inner validation from fitting and stopping. | Use inner-selected round count or a declared outer-training split. |
| Calibration / historical 70% threshold | Declare recipe. | Derive from cross-fitted training predictions only. | Fit using outer-training cross-fitted predictions; holdout only evaluates. |
| Top3 allocation | Freeze capacity, rounding, tie rule and denominator. | Rank validation scores to assess candidates. | Rank outer holdout scores without labels. |

Outcome-free dataset-wide QA can stop a run for a corrupted extract. It should not silently learn the model's feature membership from holdout covariates. A bad test-time schema requires a declared failure/missing-value policy, not a predictor substitution selected after inspecting performance.

**4–6. Remove the univariate screen; retain all predeclared admissible predictors with regularization.** The actual Phase 5 screen is an **upper-AUROC safety exclusion**, not a conventional low-AUROC significance filter. I reject both as automatic modelling gates here. Marginal AUROC can miss interactions and conditional effects; it can also exaggerate source/missingness proxies. About 130 candidates and 2,155 events do not require marginal filtering. The crude ratio is 16.6 events per raw candidate, but one-hot/ordinal expansions, rare categories and interactions increase effective complexity; this ratio is not a sample-size assurance. Regularization, constrained nonlinear complexity and honest evaluation are the defensible controls.

All means all **semantically and temporally admissible** predictors, not every available field. Keep outcome/future/ID exclusions and provenance quarantine. A high training-only AUROC can trigger a forensic report; an adjudicated availability violation, not a numeric threshold, establishes leakage. A protocol-changing adjudication requires a logged amendment and a new run, never a fold-union feature veto. A weak-leakage feature cannot be rescued by a reassuring AUROC.

**7. CCI: MODIFY C-02.** Thermometer encoding removes equal-spacing and single-linear-trend assumptions **if ordering is documented**. Merely having a list of codes does not prove that order. Cumulative dummy coefficients are not automatically monotone; monotonicity requires separate constraints. If only nominal meanings are documented, use nominal coding for every family. If meanings are undocumented, quarantine CCI from the primary claim, with raw-code categorical treatment at most exploratory. Raw numeric trees are acceptable for verified ordinal order because thresholds use order, not spacing; arbitrary nominal code thresholds are inappropriate. Define missing, rare and unseen codes explicitly; the existing linear path's pooling of rare/unseen values into the reference is not an adequate general semantic policy. Use a Phase 5 override so protected earlier definitions remain intact. Do not try several encodings and pick the one with the best outer result.

**8. Negative controls: MODIFY C-07.** A single quick ENET permutation and a perfect outcome sentinel are insufficient. The sentinel is especially circular if the AUROC gate removes it before the model sees it.

- Keep outcomes separately sealed; inject deterministic synthetic forbidden/future fields through the *actual input/derivation boundary* and require rejection before fitting. Test event-time-safe but load-time-future examples, a renamed proxy with declared forbidden lineage, and a weak/noisy outcome proxy below 0.80. Unknown fields must fail closed. This tests enforcement of known lineage; it cannot prove that an undocumented real source is clean.
- With **fixed saved fold assignments**, permute labels before every outcome-dependent development operation, refit preprocessing/selection/tuning, and evaluate permuted holdouts. Use a prespecified small set of seeds, for example ten reduced-budget complete-pipeline permutations, as a gross-leakage diagnostic. Under a global exchangeable null, expect AUROC near 0.5 and Recall@Top3 near 3%; do not require exact equality or forbid every chance-positive verdict. Restricted permutations preserve the relevant grouping when snapshots/clinics are involved. This is not a high-resolution significance test or a replacement for the real training graph audit.
- Require holdout-label mutation to leave training membership, selected configuration, model and scores unchanged **after freezing the split**. Re-stratifying after flipping labels would naturally change the split and is a different test. Do the analogous inner-validation boundary test, while allowing validation labels to affect candidate selection as intended.
- Include a synthetic legitimate predictor with AUROC >0.80 that passes provenance checks. It should be retained, demonstrating that strong signal and leakage are different concepts.

Synthetic traps and optional local real-pipeline permutations are **future implementation proposals only**. No controls on real data are run in this review.

**9. PRE/POST remains useful but is not an isolated causal experiment.** Freeze the same file hash, usable-patient IDs, saved folds, seed, budgets, historical tuning objective and historical reporting definitions for the repair comparison. Verify fold hashes rather than trusting the same seed. Paired patient predictions permit a descriptive change calculation. If screen removal, train-local gates and CCI encoding change together, the difference is their combined effect; no attribution to one repair. A switch to log-loss tuning changes another component and must be reported as an improvement branch, not disguised as a minimal repair. OLD versus SAFE must remain paired within either branch. Unknown-timing ALL is a legacy/exploratory audit only.

**10. Phase 5.1 can become post-hoc optimization.** Freeze the repair recipe before reading the aggregate eligibility table; use triage to document what happened, not choose thresholds or encodings. Record every amendment and all candidate results. Do not repeatedly change margins until the old conclusion returns. In the special case where an audit proves the original screen retained exactly the fixed allowed universe, the original predictions may equal those of a no-screen procedure; that can bound the screen's realised effect. The proposed rounded max-AUROC margin does not prove that identity or establish validity of the entire pipeline.

### Concrete corrected architecture

1. Seal the later evaluation labels and register a fixed allowed predictor universe, label/availability contract, folds, tie rule and analysis plan. For the repair use the existing January snapshot and saved patient folds.
2. Delete automatic outcome-AUROC membership decisions. Fit all learned eligibility/preprocessing in inner training. Either use a fixed absolute penalty grid or a fixed dimensionless `lambda/lambda_max` grid, with `lambda_max` calculated **inside each inner training fold**. Align candidates by the dimensionless recipe, not by a full-outer-data-derived penalty. Recompute the selected ratio's path on outer training at refit. This is a stricter inner correction, not evidence that the old grid exposed outer labels.
3. For the historical audit retain the old 70% inner objective and annotate its limitations. For subsequent improvement use the prespecified log-loss/Top3 shortlist policy in section 5. Fit the same candidate procedures in every outer fold; include recipe/family selection inside the inner loop if reporting the selected procedure's outer performance.
4. Refit on outer training and predict its untouched holdout. Allocate exact 3% within each holdout with proportional aggregate allocation and a deterministic outcome-blind tie break. Store immutable paired predictions. Historical 70% metrics remain secondary audit outputs.
5. Report the repaired comparator, finite prespecified alternatives, conditional paired uncertainty and all failed candidates. Do not report the maximum over outer-CV alternatives as an unbiased champion estimate.
6. On all development data, repeat the registered inner selection procedure, fit the final candidate and freeze the full artifact. Select one champion and one corrected ENET comparator **before** opening the later test. Temporal testing is the confirmatory comparison.

Complete nesting estimates a specified algorithm's development performance; it does not prove future transport or reverse prior human model selection. The underlying distinction is supported by [Varma and Simon's nested-CV study](https://link.springer.com/article/10.1186/1471-2105-7-91).

### Disposition of every Phase 5.1 change

| Lead item | Decision | Review disposition |
|---|---|---|
| T-01 | **MODIFY** | Read-only provenance/exclusion inventory; reject validity margin and automatic rehabilitation. |
| C-01 | **REJECT** as proposed | Replace automatic AUROC exclusion with provenance gates; train-local learned preprocessing. If a screen is retained, both inner and outer nesting are required. |
| C-02 | **MODIFY** | Document order/meaning first; categorical or quarantine fallback; fix nominal tree semantics too. |
| C-03 | **ACCEPT** | Promote exact within-fold Top3 outputs; distinguish audit output from new primary success rule. |
| C-04 | **MODIFY** | Report within-fold equal-sensitivity as descriptive, not a deployable threshold. Retain pooled legacy output only for reconciliation; neither is confirmatory. |
| C-05 | **MODIFY** | Make strict inner path isolation part of the repair, rather than optional. |
| C-06 | **ACCEPT** | Foldwise calibration alongside pooled values; suppress unstable/undefined estimates, do not call five folds independent studies. |
| C-07 | **MODIFY** | Boundary traps plus complete-pipeline permutation diagnostics; no deterministic chance-verdict requirement. |
| C-08 | **ACCEPT** | Carry explicit limitations into result summaries. |
| C-09 | **MODIFY** | Immutable PRE and paired saved folds; combined-change interpretation; do not infer repair causality. |
| C-10 | **ACCEPT** | Separate scientific version, manifest, config hash and resume boundary; exact numbering is project convention. |
| S-02 | **MODIFY** | One registered smoother-tuning development arm, shared across serious candidates; no optional search for verdict agreement. |

## 5. Recall@Top3 objective review

### Selection and tuning

**Recall@Top3 should be the primary comparative evaluation endpoint.** At fixed population, event denominator and capacity, maximizing TP, recall and PPV yields identical ordering of models. AUROC improvement elsewhere is insufficient. Define `k = round_half_up(0.03*N)` and a stable pseudonymous-key hash for ties, independent of labels, source row order and model. A numerical risk cutoff estimated in development is not a fixed-capacity rule on a new population; rerank current eligible patients to fill its exact quota.

**Do not broadly tune hundreds of configurations directly on Top3.** With five folds, each outer holdout has about 431 events and 585 selected patients. At current capture it has about 217 captured fallers; one event moves fold recall by about 0.23 percentage points. Each inner validation fold has about 345 events and 468 places. Many close candidates will differ by only a few boundary patients. A crude binomial calculation for all 2,155 events gives a single-model recall SE of 1.08 percentage points and 95% half-width about 2.11 points. This is an illustration, not the CV confidence interval or the SE of a paired improvement; ranking, dependence and training variability matter.

Recommended prespecified improvement policy:

1. Train and tune a small regularized grid with **unweighted inner log loss**; keep Brier/calibration summaries. A proper scoring rule is smooth but can favor probability improvements outside the upper tail, so it is not a sufficient operational selection rule by itself.
2. Define a shortlist of candidates within one prespecified inner log-loss SE of the best, with an explicit reproducible SE calculation and complexity tie rule. Treat the one-SE band as a selection heuristic, not a confidence statement about dependent folds.
3. Among that small shortlist, select by aggregate **within-inner-fold Recall@Top3**, breaking near ties toward simpler models. Freeze the near-tie tolerance in operational units before running. All candidate selection, including feature bundles/families, occurs inside outer training. Alternatively use log loss alone for tuning and reserve Top3 for comparing a few prespecified procedures; record this choice in advance.
4. A bounded upper-tail ranking surrogate, such as a prespecified weighted pairwise loss or partial-AUC region, is **DEFERRED** until this policy demonstrably misses upper-tail structure. Such surrogates are not equivalent to Top3 and need a new protocol. AP is less discontinuous than a single cutoff but is still a ranking statistic, not a proper probability score.

No random-seed search, moving capacity to 2.8%/3.2%, class-weight search, repeated threshold engineering or accumulating optional sensitivities until a gain appears. Plot neighbouring capacities to show fragility, not select a better endpoint.

### Paired uncertainty

For each fixed fitted candidate/comparator on the same people, use a **paired patient bootstrap**, at least the existing 2,000 registered replicates: resample identical patients/weights for both models, recompute capacity and ranking in each replicate, and calculate ΔTP, ΔRecall, ΔPPV and ΔFP. In outer OOF, retain each row's fold/model and reallocate capacity according to resampled fold sizes, as `capacity.py` does. Do not bootstrap each model independently, average fold confidence limits, or use a t-test treating five overlapping-training folds as five independent experiments. Report disagreement counts among fallers selected by only one model and overall selected-set overlap; these explain paired precision better than total events alone.

The existing OOF bootstrap is explicitly **conditional on the fitted outer models**; it does not include retuning, training-sample uncertainty or winner selection. It is useful development evidence, not a fully calibrated confirmatory interval after model shopping. Full-pipeline resampling can assess training variability but is expensive; a small fixed fold-assignment sensitivity, if preregistered, is a stability diagnostic rather than extra independent events. Prefer spending limited compute on an untouched temporal test. On a single later snapshot, paired patient bootstrap estimates conditional test-population uncertainty for frozen models. If patients have repeated landmarks, bootstrap entire patient histories; if inference targets new clinics, use an appropriate clinic-level resampling design with enough clinics.

### Scientifically safe success rule

**MODIFY Claude's section 6.3.** A point improvement ≥MOD with a CI merely above zero supports some benefit but does not establish that the benefit exceeds MOD. Use these categories for one preregistered champion-versus-corrected-ENET comparison:

| Result on the sealed temporal test | Allowed conclusion |
|---|---|
| Lower paired 95% limit of Δcaptured fallers per 10,000 eligible patients exceeds the declared MOD; required data/calibration/safety gates satisfied | **CONFIRMED_MATERIAL_GAIN_AT_3_PERCENT** |
| Lower limit >0, point estimate ≥MOD, but lower limit ≤MOD | **PROMISING_MATERIAL_GAIN_UNCERTAIN** |
| Lower limit >0 and point estimate <MOD | **SMALL_STATISTICAL_GAIN** |
| Interval includes 0 | **INCONCLUSIVE**; cannot claim equivalence or no benefit without an equivalence design. |
| Upper limit <0 | **WORSE_AT_CAPACITY** |

Management should declare MOD from the added implementation burden, programme costs, plausible uptake and intervention effectiveness before candidate results. Prefer additional captured **people** per 10,000 at a fixed 300 contacts; do not equate prediction gain with prevention benefit. An illustrative provisional margin is **5 extra captures per 10,000**: about 49 at this cohort size, or about 2.26 recall points at the supplied prevalence. This is a planning proposal requiring management justification, not an established clinical threshold. A round 50-extra-patient example equals 5.13 per 10,000 and 2.32 recall points. A 1-point gain is about 22 patients and could matter, but cannot be declared meaningful merely because a P value is small.

Plan paired precision/power using development-only score overlap and discordant event selections, with simulation reflecting the exact capacity rule. General validation sample-size guidance for calibration and discrimination does not by itself power a paired Top3 contrast; see [Riley et al.](https://onlinelibrary.wiley.com/doi/10.1002/sim.9025). Claude's PPV=0.10 illustration is poorly matched to supplied facts: current PPV is about 0.371, with a simple binomial 95% half-width about 1.75 points among 2,924 flags. Fifty events per subgroup is a reporting floor, not proof of adequate precision.

Freeze one primary comparison and one test opening. If development uses the outer results to choose a winner, label that winner's CV performance exploratory and use the temporal test for confirmation. Secondary family, SAFE/ALL, subgroup and curve comparisons are descriptive or require a declared multiplicity procedure; they cannot replace a failed primary comparison. If the result is inconclusive, stop this protocol and retain the comparator. A new substantive feature hypothesis needs a new registered development plan and a new untouched test, not repeated access to this one. **55% is a working aspiration only:** a candidate below 55% may deliver a material paired gain, and one above 55% may fail to improve on a corrected comparator.

## 6. Feature engineering review

The lead document gives medication exposure and diagnosis recency sensible priority, but does not provide an adequate trajectory roadmap. Ranking below is a **source-informed hypothesis ranking**, not measured incremental gain. The key distinction is new information versus a useful change of representation.

The OLD catalogue already contains fall counts at 30/90/180/365 days and since study start, fall recency bands, nurse-reported falls; visit counts at those windows; current/previous MEFI group, worsening, assessment count and recency; current mobility/ADL, walking aid, gait observations and GUG date; nurse assessment counts/dates; diagnosis counts and medication-burden proxies. Rebranding these as “trajectory features” does not create data. Raw histories can add ordering, episode identity, timing and change that these summaries discard.

### Top 10 hypotheses

| Rank / family | Concrete hypothesis and expected information gain | Leakage risk and required data | Effort / redundancy with OLD |
|---|---|---|---|
| 1. Time-bounded medication exposure and change | Recent initiation/escalation of selected psychotropics/opioids, concurrent high-risk exposure, burden change over 30/90 days, and interaction with mobility deterioration discriminate among otherwise similar patients. **High potential new information**, contingent on ascertainment. | High if purchase status/dose is updated after index. Need prescribing and dispensing timestamps, ATC, dose/days supplied or a validated exposure rule, cancellations and available-at timestamps. Gaps mean no observed refill, not proven discontinuation/adherence. | High extraction, moderate modelling. OLD substance counts/polypharmacy/narcotic counts are static proxies and some have timing limitations; new episode timing is not redundant. |
| 2. Prior-fall episode trajectory / acceleration / inter-fall intervals | Episode-level last and penultimate dates, intervals, recurrent episodes, shortening intervals, time since first recurrence. **Medium–high new information** if episodes are identifiable; exact recency also improves coarse bands. | High if follow-up visits masquerade as new falls, retrospectively dated records or post-index records enter. Need all historical dated fall/fracture records, encounter/source and episode links; prespecify clinically reviewed episode collapsing. Outcome definition stays fixed for the repair. | Medium–high. Disjoint windows and rate acceleration from existing counts are **representation only**; intervals/episode identities cannot be recovered from nested counts. |
| 3. Mobility / ADL / assistive-device transitions | New aid, increasing transfer dependence, balance/gait change and decline over comparable assessments may mark imminent vulnerability. **High new information** where repeated coverage is adequate. | Medium–high: retrospective form entry and selection of assessments performed after falls. Need dated repeated items, instrument/version, assistance/device records and availability timestamps. | High extraction, moderate modelling. Latest-state OLD flags cannot encode direction or speed; explicitly distinguish stable severe disability from active decline. |
| 4. Diagnosis-specific recency and emergence | First/last recorded dizziness, syncope, gait abnormality, parkinsonism or stroke; recent first-recorded flags versus longstanding records. **Medium–high new information**. | High if “first” comes from truncated history or delayed coding. Need diagnosis/encounter date, creation/load date, code history and observed-history length. First recorded is not clinical onset. | Medium. Static V21 flags and one global Last_Dx_Date do not provide disease-specific timing. |
| 5. Acute-care and utilisation acceleration | Recent ED visit/discharge, repeated acute contacts, hospital-to-home transition, and rising contacts in disjoint windows. **High for newly available acute-care history; medium/low for counts alone**. | High for invoice/discharge updates or a future discharge of a stay ongoing at index. Need encounter type, admission/discharge and arrival/load times; outpatient contact rows must be deduplicated. | High new source effort. OLD visit windows already support simple contrasts; ED/acute setting and safe discharge recency are genuinely additional. |
| 6. Frailty deterioration with elapsed time | Repeated MEFI transitions, time between assessments, persistence or rapid worsening, separating score staleness from current status. **Medium incremental information**. | Medium–high for batch-computed/backdated scores. Need all score records, valid-from/creation times, components/version and tie-break rule. | Medium–high. Current/previous/worsened MEFI are already OLD; an untimed difference is redundant, multi-step trajectory/time-normalized change less so. |
| 7. GUG / TUG trajectory | Repeated standardized performance, slowing or change in balance/rising items. **Potentially high where valid repeated measurements exist**, otherwise no feasible hypothesis. | Medium for selective assessments/late entry; distinguish GUG qualitative instrument from TUG seconds. Need test values, item anchors, instrument, dates, context and timestamps. | High discovery/extraction. Existing GUG date and latest gait items cannot substitute for repeated timed performance. Do not invent a TUG total from an undocumented form. |
| 8. Multimorbidity change | New disease domains, worsening disease burden and recent combinations of neurological/sensory/cardiovascular problems rather than total diagnosis-row counts. **Medium new information**. | High for coding intensity, registry backdating and changing CCI definitions. Need domain-resolved dated diagnoses/registries and availability history. | Medium–high. OLD diagnosis counts and CCI overlap heavily; counts of distinct newly recorded domains can add timing, another static deficit sum likely cannot. |
| 9. Assessment frequency/change | New or increasing *encounter-level* assessment frequency, changes in indication and assessor, and recency coupled with worsening. **Low–medium predictive information; high ascertainment insight**. | High workflow/detection bias. Need form instance IDs, dates, question-to-form mapping and reason for assessment. Never count questions as independent tests. | Medium. OLD counts/recency/repeat indicators already cover much; source audit is essential because Assessment_Count_365D counts question rows. |
| 10. A small set of clinical interactions | Prior falls × mobility impairment; recent sedating exposure × gait impairment; recent syncope × medication change; deterioration × advanced age. **No new raw information**, but may expose conditional effects to ENET. | Low incremental leakage if inputs are safe; high selection risk if many products are searched. Need documented inputs and prespecified interaction list. | Low. XGB can already learn many such interactions; use only a few clinically justified terms with shrinkage, not every pair. |

For count contrasts use the exact declared inclusivity of each window, observed-history duration and non-overlapping bins. For example, `C90-C30` summarizes the older part of the 90-day window; rates require its correct number of days. Avoid unstable ratios with zero denominators, arbitrary numeric differences of ordinal codes and fake slopes from irregularly timed measurements. A verified ordered group can supply transition indicators without assuming equal spacing. Report missing trajectory because never assessed separately from stable values and insufficient history.

Medication associations motivate hypothesis 1; they do not prove incremental Top3 benefit or causal preventability. [Seppala et al.'s psychotropic medication meta-analysis](https://pubmed.ncbi.nlm.nih.gov/29402652/) supports the association rationale. The ranking of development priorities above is my inference from this project's information gaps.

## 7. Model-family review

Changing algorithms cannot recover dates or trajectories that were never extracted. XGB already represents nonlinearities/interactions; a new tree library changes inductive bias and optimization, not the information available. The event count supports a constrained comparison, not unrestricted architecture shopping.

| Family | KEEP / DEFER / DROP | Why it might improve Top3; information beyond existing ENET/XGB | Main risk with 2,155 events; evidence that would convince me |
|---|---|---|---|
| Elastic Net | **KEEP** — primary baseline and serious candidate | Shrinkage for correlated predictors, safe nominal/ordinal representations, small prespecified spline/interaction basis. New histories can add actual information; a basis adds representation. | Rare binaries and expanded bases create unstable tail ranks. Identical nested folds, bounded penalty search, paired gain surviving temporal testing; coefficient selection frequency is supportive, not the endpoint. |
| LASSO | **DROP standalone search; KEEP as ENET endpoint** | Sparsity may simplify the programme; no unique signal beyond ENET with l1_ratio=1. | Unstable correlated-feature selection. Reuse the ENET path/candidate endpoint; prefer it only when operational performance is effectively tied and simplification is valuable. |
| XGBoost | **KEEP** — one existing nonlinear challenger | Thresholds, nonlinearities, interactions and native missing-value routing that plain linear ENET lacks; no new signal without new inputs. | Deep trees, small event leaves and 100-trial tail optimization. Tight depth/leaf/regularization bounds, log-loss tuning, inner-only stopping, reproducible paired Top3 gain and temporal confirmation. |
| CatBoost | **DEFER**, conditional replacement for XGB | Appropriate categorical handling/ordered boosting may improve nominal-code treatment. It does not resolve unknown clinical meanings or source timing, and offers little distinctive information if the catalogue is mostly binary/count/ordinal. | High-cardinality or missingness shortcuts and too many trials. Consider only after documenting substantial useful nominal structure; compare *instead of*, not in addition to, another GBDT with equal budget and same safe inputs. |
| LightGBM | **DEFER** — computational substitute | Efficient tree fitting and a different growth policy; no unique information beyond XGB. | Leafwise local overfit/rare-event leaves. Adopt if measured compute burden requires it; not another simultaneous accuracy lottery. Same bounded candidate count, leaf constraints and temporal comparison. |
| Explainable Boosting Machine / GAM | **KEEP one constrained GAM challenger; DEFER separate EBM sweep** | Smooth nonlinear risk shapes may improve plain ENET without high-order tree interactions; additive bias can regularize better than XGB. This is representation/regularization, not new signal. | Hundreds of bins/interactions and data-chosen shapes can overfit tail patients. Start with low-degree penalized spline logistic GAM on a few named continuous variables plus baseline terms and ≤3 declared interactions; require nested paired gain. Use EBM as an alternative implementation, not a second shopping branch. |
| Random Forest / ExtraTrees | **DROP first-round searches** | Bagged/randomized splits offer variance averaging; no distinct data signal that boosted trees cannot represent. | Many low-event leaves, coarse/poorly calibrated tail scores and added budget. Reconsider only if a prespecified stability analysis establishes a specific deficiency in XGB and a simple fixed forest addresses it. |
| Survival modelling | **DEFER algorithm expansion; KEEP estimand/censoring audit now** | Event times and partial follow-up can use information discarded by complete-label binary models; death-aware 180-day cumulative incidence may alter ranking. That is actual new outcome information. | Informative censoring, false event times and wrong estimand. Convincing evidence requires reliable event/death/loss dates, appropriate competing-risk risk estimates, weighted/time-dependent evaluation at the same 3% policy and temporal confirmation. Start with simple penalized cause-specific/discrete-time models when warranted. |
| Random Survival Forest | **DEFER** | Nonlinear time-dependent/censoring structure beyond a simple survival model, but not a proven advantage over XGB plus a suitable outcome model. | Splitting event information among many leaves/time points and unreliable risk tails. Must beat a simpler competing-risk comparator on 180-day cumulative incidence Top3 with proper censoring evaluation. |
| Boosted survival | **DEFER** | Can combine nonlinear predictors with time-to-event information. | Hazard ranking is not automatically cumulative-incidence ranking when death competes; tuning and calibration add complexity. Requires the same survival data contract and a prespecified fixed-horizon comparison, not a C-index win alone. |
| Learning-to-rank | **DEFER** | Upper-tail losses may allocate model effort to patient ordering; no extra input signal. | Many pairs do not create more than 2,155 event patients; arbitrary query groups, proxy labels and weak probability calibration. Only a fixed limited loss experiment beating the log-loss shortlist on genuinely unseen Top3 would convince me. |
| Mixture-of-experts | **DROP now** | Could model distinct mechanisms, but clinical phenotypes are not proof of separable predictive regimes; ENET interactions/XGB already supply conditional structure. | Splitting events across experts and unstable gating. Require independently reproduced effect heterogeneity and enough events per regime before revisiting; do not use faller-only clusters to build outcome-conditioned gates. |
| Tabular neural networks | **DROP now** | No unique signal in ~130 summary variables beyond nonlinear competitors; sequence models might someday exploit rich histories, a different task. | Architecture/regularization search and tail overfit with 2,155 events. Need a substantially richer sequence dataset and a new bounded, externally confirmed protocol. |

These distinctions match the methods described by the original [XGBoost paper](https://arxiv.org/abs/1603.02754), [CatBoost paper](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), [LightGBM paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), and [InterpretML EBM documentation](https://interpret.ml/docs/ebm.html). None demonstrates a Top3 advantage in this cohort. The KEEP/DEFER/DROP judgments are project-specific inferences.

## 8. Clustering / phenotyping review

**Material omission:** Claude's strategy contains no explicit profiling or clustering protocol. Its importance summaries, prior-faller subgroup and proposed separate model do not answer management's two questions adequately. Add a parallel clinical-understanding track without making its completion a prediction-success gate.

### A. Faller profiling — supervised and descriptive

Use the prespecified future recorded-fall outcome to compare **baseline/pre-index** characteristics of fallers and non-fallers. For each documented feature report prevalence among fallers, prevalence among non-fallers, absolute difference, prevalence enrichment, and missingness/assessment coverage in each group. Give the denominator explicitly: assessed patients and the whole eligible cohort answer different questions. A feature common among fallers may also be common among everyone and provide little discrimination.

Estimate risk ratios and risk differences from `P(fall | feature)` using the full cohort with valid follow-up; the enrichment `P(feature | fall)/P(feature | non-fall)` is a different quantity. Odds ratios can be reported when model-based, but they are not risk ratios. Use prespecified age/sex-adjusted and fuller adjusted associations (including prior falls and an explicitly justified contact/assessment adjustment set), with intervals and regularization when needed. These are conditional descriptive associations, not causal effects; adjusting for care contact can itself condition on a mediator or collider. Report sensitivity to that adjustment rather than asserting causality.

For co-occurrence, use a small clinically declared list of pairs/triples, minimum support, absolute prevalence and lift, followed by clinician review. Do not enumerate all combinations and publish the largest odds ratios. Use exploratory false-discovery control for broad tables and label discoveries as hypotheses for another cohort. Feature importance/SHAP can supplement these tables but cannot replace prevalence, enrichment or adjusted relationships. Suppress small cells under project privacy rules.

This may answer “what features are common among people who fall?” without clustering at all. It is the first deliverable and has value even if every clustering attempt fails.

### B. Faller-only clustering — clinical faller phenotypes

Cluster the **2,155 future fallers** using a concise, prespecified panel of baseline clinical domains; exclude future injury/severity/treatment fields from baseline phenotypes. Remove duplicate measurements/summary scores so one domain does not dominate. Candidate domains: prior-fall episodes, mobility/function, cognition, verified medication exposure, sensory/neurological conditions and frailty. Treat assessment presence carefully and show whether it defines the partition.

This asks whether recorded fallers have clinically distinguishable baseline profiles. It cannot estimate population fall risk, because everyone in the clustering sample falls. Conditioning on future falling can induce associations between causes/detection mechanisms; do not interpret clusters as causal “types” of falls. Data selected for recorded outcomes may primarily represent documentation pathways.

### C. Full-population clustering — population phenotypes with different fall risks

Learn clusters on baseline predictors **without labels**, then describe cluster sizes, clinical profiles and future fall risk with uncertainty. Cluster count/features must not be selected by maximizing observed fall-risk separation. To claim reproducible risk enrichment, lock the partition/assignment rule on development, assign a later population and then estimate risks. A cross-sectional full-cohort description is legitimate exploration, not independent evidence of prediction gain. The 97,453-patient population requires scalable methods.

### Suitable mixed-data methods

| Method | Appropriate role | Main cautions / decision |
|---|---|---|
| Gower + PAM | **KEEP** as faller-only primary mixed-data method; medoids give concrete representative profiles. | Declare nominal/ordinal types and domain weights; distinguish asymmetric presence flags from symmetric binary states. Robust clinical ranges and missing-pair rules matter. Gower with pairwise missing data can make distances poorly comparable. Full population uses a fixed representative subsample/CLARA-style medoids with a declared assignment rule, not a dense full distance matrix. |
| Hierarchical clustering | **DEFER** to one representation sensitivity on fallers if needed; dendrogram can show nested structure. | Average/complete linkage on mixed dissimilarities; do not casually apply Ward's Euclidean criterion to arbitrary Gower distances. Linkage sensitivity and quadratic cost limit population use. |
| Latent class analysis | **KEEP as a possible alternative**, not a parallel default sweep, for a small binary/categorical clinical panel. | Clinically useful class probabilities; local independence can fail for related mobility items and overlapping diagnoses. Check residual dependence, multiple starts, sparse cells, class size and uncertainty. Ordinary categorical LCA is not a model for untreated continuous values. |
| FAMD + clustering | **DEFER/alternative for population scale** when quantitative/categorical structure is substantial. | Training-only imputation, scaling and component selection; rare categories or missingness can drive axes. Explain clusters back in original units. Variance-maximizing dimensions need not capture clinically useful faller structure. |
| HDBSCAN | **DEFER** unless variable-density clusters/noise are substantively expected. | Mixed distance/embedding choices, high-dimensional sparsity, unstable noise labels and min_cluster_size/min_samples dependence. Accept “no stable clusters”; never tune an embedding until coloured outcome plots look separated. |
| k-prototypes / mixed finite mixtures | **DEFER** as scalable alternatives only if Gower/subsampling is inadequate. | k-prototypes requires a defensible numeric/categorical trade-off and initialization checks; general mixtures need appropriate component distributions and identifiable classes. |
| Raw K-means / outcome-selected UMAP clusters | **DROP** | Raw mixed codes violate Euclidean interpretation; attractive 2-D separation is not a clinical validation criterion. K-means in a justified numeric embedding is a possible tool, not the default scientific model. |

Methodological basis: [Gower's original mixed-similarity paper](https://cbio.mines-paristech.fr/~jvert/svn/bibli/local/Gower1971general.pdf), [Linzer and Lewis on categorical LCA](https://www.jstatsoft.org/article/view/v042i10), [FactoMineR's FAMD documentation](https://search.r-project.org/CRAN/refmans/FactoMineR/html/FAMD.html), and [HDBSCAN parameter documentation](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html). These sources inform method suitability, not a claim that genuine clusters exist here.

### Required evidence and separation of clinical/predictive value

Prespecify a small range, e.g. **2–5 clusters**, and include the possibility of no useful partition. For faller-only discovery, a provisional useful-size floor is **max(100 patients, 5% of fallers)**, approximately 108 here. For population phenotypes, start with **≥1% of eligible patients**, approximately 975 here, and require sufficient events for risk precision; rarer clinically important groups may be described separately rather than forced into “stable clusters.” These are planning criteria for clinical usefulness, not universal statistical laws.

Require at least 100 fixed-seed resamples of the full imputation/representation/clustering process, matched-cluster Jaccard/co-assignment stability, membership uncertainty and sensitivity to domain weighting, ordinal coding, outliers, missingness and assessment-heavy variables. A provisional mean matched Jaccard ≥0.75 for each reported cluster is a screening criterion; also report its distribution and instability, not just the mean. Stability can arise from an inflexible method without a meaningful clinical partition, as [Hennig's cluster-stability study](https://www.homepages.ucl.ac.uk/~ucakche/papers/clusta.pdf) explains. Obtain clinician interpretation without showing outcome-risk rankings during phenotype naming, and reproduce profiles/assignment in a later cohort.

For prediction, fit **the whole clustering pipeline inside inner training and then outer training**, and assign heldout patients by a frozen medoid/posterior/assignment rule. An outcome-informed faller-only prototype can only be learned from **training fallers** and must map every heldout patient using predictors alone; it is not an ordinary label-blind population cluster. Adding a full-cohort cluster ID to X would leak transductive structure; adding heldout faller-only membership uses heldout labels directly. Compare clusters/soft memberships against the same original features and simple clinical interactions, with paired Top3 evaluation. Clusters are functions of existing predictors and add no raw information; their benefit would be representation/regularization.

**Expected value:** likely greater for clinical understanding than for prediction, provided patterns are stable and interpretable. Predictive improvement is uncertain and probably small when ENET interactions/XGB already exploit the same variables. Accept a clinically useful phenotype analysis with zero Top3 gain. No prediction-cluster experiment is in the first five-protocol budget.

## 9. Temporal validation review

### Dates and what “untouched” requires

**MODIFY D-3 / section 6.2.** The arithmetic is correct: January 1 +180 days is **June 30, 2026**; July 1 +180 days is **December 28, 2026**. With end-of-index-day scoring, actual outcome windows are **January 2–June 30** and **July 2–December 28**. They do not overlap. A July historical predictor may legitimately include January–June outcomes, if those records were available by July scoring time. Patient overlap and persistent risk remain, so this is same-HMO **internal temporal validation**, not external-site validation.

As of October 7, July's outcome window has already begun and is partly observable. Claude's November–December milestone “before any 2026-07-01 outcome exists” is impossible. Replace it with **before any validation outcomes, partial outcome summaries or outcome-informed findings are accessed by the development team**. An untouched dataset can exist even when events have occurred, if access is genuinely controlled. This review cannot establish that such control has occurred.

Require a written BI/custodian access inventory: who has seen July onward labels, subgroup counts, model performance or records; whether any informed feature choices; and which future extract/version is sealed. An immutable July predictor snapshot is not an untouched July label set by itself. If partial outcomes have influenced development, use July only as exploratory evaluation and select a later truly inaccessible period. Moving the index date alone does not restore prospective timing if training-label lag still reaches past that new index.

There is a further **training-time availability issue**: a January model using labels through June 30 plus billing/documentation lag could not necessarily have been trained before July 1. A model corrected/frozen in October is plainly not a prospectively deployed July model. July evaluation can still be an honest **retrospective temporal test of unseen outcomes**, but must say that and document what training records became available after July 1. For a prospective chronology claim, freeze the complete model and available development labels before the new scoring date. A later silent prospective snapshot is the cleanest design.

### Opening rule and input availability

Freeze capacity/eligibility, one champion and comparator, all preprocessing, feature definitions, death/censoring estimand, tie-break, MOD, metrics and verdict before unblinding. A custodian may verify maturity/contract without returning performance or event enrichment to developers. Score predictors and lock predictions first; a separate evaluation stage then links the mature labels. Predictor availability must be reconstructed **as it was at end of July 1**, using event **and creation/load/version** times; a 2027 query for events dated before July 1 can include backfilled future knowledge. The current V21 study-end of August 1, 2026 cannot certify a December-mature July outcome extract; require an explicit new label observation cutoff and completeness evidence.

Earliest outcome opening is **December 28, 2026 + the prespecified measured documentation/settlement lag**, after all maturity and follow-up checks pass. Mid-February 2027 is a provisional scheduling assumption, not a validated earliest date. Specify source-specific lag distributions, a completeness tolerance and a fixed extraction/freeze rule, including how late records beyond the rule are handled. Do not choose the freeze date after viewing whether performance improves.

Changing/removing an unavailable frozen predictor at test time changes the model. Stop that primary test or evaluate a separately prespecified fallback artifact; do not silently drop its input or deterministically refit after observing outcomes. Mode B refitting, if required, must use only frozen development inputs/labels and reproduce prediction scores/configuration within a prespecified tolerance and reproducible environment. Pickle-byte identity across environments is not the scientific equivalence criterion. No recalibration fitted on test outcomes is part of a frozen-model validation.

### Capacity denominator, death and censoring

**P0:** Phase 5 selects usable 0/1 labels before CV (`data.py:298–333`); Phase 4 excludes unlabelled rows before metrics (`evaluate.py:325–327`). This is acceptable as an explicitly labelled-subcohort historical analysis but can distort an operational quota. At deployment, death/loss during future follow-up is unknown. Allocate 3% among all **baseline eligible** patients before seeing label usability; do not remove censored people and refill contacts with others. Report what proportion of the chosen contacts has an ascertainable outcome and show the difference from the legacy labelled-subcohort estimand.

Audit actual death handling: the lead's shorthand “death as non-event” is not guaranteed by the inspected code, which uses provided binary labels and excludes nonbinary/unlabelled rows. For the future estimand, death before any fall can be a known non-fall by day 180 for real-world cumulative-incidence prediction when event ascertainment before death is complete. Death after a fall does not erase that fall. Non-death loss to follow-up is unknown, not a negative. Do not condition the primary analysis on surviving 180 days; Claude's death-exclusion sensitivity changes the target population and can introduce survivor selection.

If outcome status is essentially complete, a binary 180-day model can estimate the probability of a fall before death by the horizon without survival architecture. If censoring is material or selective, prespecify a justified IPCW/competing-risk evaluation, diagnostics for weight positivity and sensitivity to assumptions; otherwise restrict claims to the evaluable subcohort and report bounds/limitations for the operational population. A death table alone does not resolve this. Resolve the outcome contract before claiming calibrated clinical risk; defer complex survival models, not this estimand decision. Relevant guidance: [van Geloven et al. on competing-risk validation](https://www.bmj.com/content/377/bmj-2021-069249).

### Other design changes

- **MODIFY transport gates:** AUROC loss 0.03, slope 0.80–1.25 and CITL 0.20 are pragmatic triggers requiring context/precision, not natural constants. Comparing temporal recall with a lower internal-CV bound is not a paired noninferiority test. Prespecify an operational noninferiority margin against the contemporaneously scored comparator and report uncertainty. Investigation cannot become permission to search new models on this test.
- **MODIFY direction consistency:** agreement with the development sign is supportive replication evidence. A conflicting development sign calls for explanation, but it should not automatically veto a prespecified material gain established on an independent sealed test. Conversely, two matching positive point estimates do not establish benefit when their uncertainty is large.
- **MODIFY subgroup analysis:** measure selected share and capture among globally selected patients within each subgroup. Independently selecting each subgroup's Top3 answers a different resource-allocation question. Present age, sex, prior falls, assessment coverage, old/new members and sites descriptively with intervals; 50 events does not guarantee adequate precision. Same-patient temporal performance can be clinically relevant; new-member performance tests a different transport question.
- **DEFER 2025 backfill:** useful for chronology/robustness if cheap and as-of verified. January 2026 is already studied; preregistration cannot make it untouched again. Never call this an independent confirmatory test.
- **MODIFY Phase 7 multi-snapshot design:** patient-grouped folds are needed for repeated rows but do not prevent training on future calendar periods. Use forward time splits, group-aware dependence handling and an embargo covering overlapping outcome windows and availability lag. Define whether repeated scoring of existing patients or new-patient transport is the target; patient grouping and temporal separation solve different issues.

## 10. V22 data-request review

Expected information ranking, conditional on usable as-of coverage: **medication exposure/change; repeated mobility/function; episode-level falls; acute-care transitions; diagnosis-specific timing; repeated frailty/GUG; multimorbidity histories**. Metadata/label quality come first as validity gates even when their predictive information gain is small.

| Tier | Request / lead reference | Why it is worth organizational effort; limits | Decision |
|---|---|---|---|
| MUST REQUEST | Source event time **and available-at/creation/load/version time** for every primary predictor source; registry membership history (R-03), view SQL/definition hashes/version and extraction cutoff (R-10) | Establishes deployability and backfill safety. Registry insert time alone is insufficient if historical rows can be overwritten; request update/version history and batch-computation time too. | **MODIFY/expand** |
| MUST REQUEST | Label/source/follow-up audit (R-05): ED/hospital/community, code 888 vs fracture-only, repeat records/episode linkage, event/load lag, death and membership-loss dates | Validity/maturity are not optional model enhancements. Include clinician-adjudicated stratified samples on site with counts/intervals returned. Aggregate source counts alone do not estimate specificity or sensitivity. | **ACCEPT/expand** |
| MUST REQUEST | Medication prescribing/dispensing/start/change dates with ATC, supply/dose when reliable, stop/cancel history (R-02) | Highest-priority genuinely new time-bounded predictor source. Include observation coverage, outside-HMO capture and pharmacist-reviewed classes; prescribe, purchase and intake are different. | **ACCEPT/expand** |
| MUST REQUEST | A custodially sealed later evaluation snapshot/label set with access log, cutoff and identical definitions (R-09) | Needed for claims; July only if access and availability requirements in section 9 can be met. | **MODIFY** |
| HIGH VALUE | Repeated mobility/ADL/gait, assistive-device transitions and standardized functional assessments; physiotherapy/rehab referral and attendance dates | Direct deterioration/transition information missing from latest-state summaries; care-use information also reflects access/indication. | **MODIFY** — add to lead plan |
| HIGH VALUE | Historical fall/fracture records with encounter/episode IDs and availability times | Separates repeated clinical events from repeated documentation and enables inter-fall intervals. | **MODIFY** — add |
| HIGH VALUE | Diagnosis-specific first/last and preferably full dated history (R-01) | Last date alone loses emergence/recurrence; need observed-history duration to interpret first-recorded diagnoses. | **ACCEPT/expand** |
| HIGH VALUE | ED/hospital encounter type, admission/discharge times and arrival times | Current invoice-based hospitalization summaries are not a safe substitute; acute transitions may add substantial signal. | **MODIFY** — add |
| HIGH VALUE | Repeated MEFI components/groups and GUG/TUG values with instrument/version and computation time (R-07) | Move from test-presence/latest category to validated change. First establish that repeated values actually exist and are comparable. | **MODIFY** — raise priority |
| HIGH VALUE | Coarse clinic/district and stable patient linkage (R-06), retained locally | Enables geography/workflow checks, dependence-aware inference and later site transport; identifiers need not become model predictors or leave the work PC. | **ACCEPT** |
| NICE TO HAVE | Restore removed hypertension/CKD/transplant flags with true V1 SQL/lineage (R-04) | Helps historical comparability; likely limited gain if related information already exists. Never replace CKD with dialysis or transplant with immunosuppression by name. | **ACCEPT at lower priority** |
| NICE TO HAVE | Orthostatic measurements, haemoglobin, sodium, eGFR and other clinically nominated repeated vitals/labs (R-08) | Useful if routinely present and timed; selective testing/missingness may dominate. Keep a small preregistered panel. Vitamin D lab/indicator priority is especially uncertain here. | **DEFER until high-value requests scoped** |
| NICE TO HAVE | 2025 V21 backfill (R-09 optional) | Helps development history if inexpensive and creation/version times prove as-of accuracy; no fresh January 2026 confirmation. | **DEFER** |
| NOT WORTH IT | More untimed static registries, overlapping deficit sums, undocumented risk codes, purchases observed after index, raw text/LLM extraction without a specific source question | Likely repackaging, unresolved timing or high governance/validation cost. Do not spend BI effort just to enlarge the feature count. | **REJECT now** |

L-01's proposed overall PPV threshold 0.80 is **MODIFY**, not a universally defensible label gate. Prespecify what clinical outcome is intended, clinically acceptable accuracy and interval precision by source/code family. Chart review of positives estimates label PPV; sampled negatives/linked independent records are needed to assess missed falls. A source-rich cohort with a low-quality label cannot be fixed by adding algorithms. Keep any outcome-definition revision in a separate protocol; do not silently change the Phase 5 repair labels.

## 11. Experiments to DROP

Drop an experiment when it cannot distinguish a useful scientific hypothesis from another already covered:

- **REJECT** parallel “nested AUROC screen”, “different AUROC threshold”, and “screen plus sentinel” model searches. Remove automatic screening and run one provenance/boundary assurance protocol.
- **DROP** standalone LASSO versus an ENET path already including L1, and simultaneous XGB/CatBoost/LightGBM/RF/ExtraTrees tournaments. Keep one existing GBDT and one constrained nonlinear additive alternative.
- **DROP** repeated OLD/ALL/SAFE × every domain × every ablation × every model with full independent tuning. Use the primary safe universe and one source-risk audit contrast; only a genuinely distinct prespecified hypothesis gets another fit. Alias identical feature sets. Timing-uncertain ALL cannot win a deployment contest.
- **DROP** separate experiments for fall count acceleration, ratios, many recency cutpoints and combinations of the same 30/90/180/365-day counts. Use one fixed representation bundle; new event histories are a separate information hypothesis.
- **DROP** repeated CCI categorical/ordinal/continuous outcome comparisons. Determine semantics first and declare one encoding.
- **DROP** pooled versus within-fold 70% thresholding as competing model-selection experiments. They are reporting/estimand sensitivities computed from saved predictions, not reasons to train new models. Likewise calibration plots, paired bootstrap, source profiling and capacity curves mostly reuse outputs.
- **DROP** separate prior-faller experts triggered solely by a subgroup calibration failure; test a few declared interactions before splitting the event sample.
- **DROP** using raw K-means, outcome-coloured embedding search or exhaustive clustering-algorithm sweeps to produce an attractive management graphic. One justified mixed-data method and one bounded sensitivity suffice.
- **DROP** mixture-of-experts, tabular neural networks, LLM features and broad stacking now. **DEFER** survival forests/boosted survival and learning-to-rank until data/estimand needs are established.
- **DEFER** full 2025 backfill/redevelopment for “independent” confirmation on already studied January 2026, and expensive full-pipeline bootstrap repeated for every failed family.

Claude already declines several complex model families; I agree with that restraint. The redundancy is chiefly in repeated sets/families/ablations and optional sensitivities, not an assertion that Claude proposed every dropped method above.

## 12. Revised priority order

1. **P0 assurance:** close phenotype/label/availability semantics, choose the operational denominator, register the success/selection rules and seal the later test. Start the high-value BI request in parallel.
2. **P0 correction:** run one corrected Phase 5.1 protocol after implementation/rehearsal, retaining the historical audit contrast. Establish a corrected ENET comparator and required provenance. No raw-data runs are authorized by this review.
3. **P1 representation:** compare existing XGB with a constrained GAM/spline ENET using the same safe inputs, training boundaries and bounded smoother-tuning policy. Stop if gains are unstable or too small.
4. **P1 information:** develop one prespecified longitudinal bundle from verified histories, led by medication changes, episode-level falls and functional deterioration; assess added value over the same safe comparator. If data are missing, report the blockage, not fabricated trajectories or substitute algorithm shopping.
5. **P2 clinical understanding in parallel:** profiling first, then one faller-only mixed-data phenotype study; population phenotyping can follow. This track is valuable without predictive improvement and need not wait for a new champion.
6. **P0 confirmation:** freeze the selected development procedure and corrected comparator, score an eligible untouched later sample, then open mature outcomes once. Any prospective rollout waits for silent prospective validation and an intervention pathway.

### Lead roadmap recommendations not already resolved above

| Lead recommendation | Decision | Reason |
|---|---|---|
| D-1 two tracks + data track | **MODIFY** | Add a clinical-understanding track and bounded improvement work; separate audit, development and confirmation claims. |
| D-2 fold-specific screen | **REJECT** as preferred remedy | Remove automatic screen; provenance-based gates and complete nesting are stronger. |
| D-3 July 1 validation | **MODIFY** | Conditional on untouched outcomes/as-of data; retrospective timing must be explicit; otherwise later sealed/prospective index. |
| D-4 capacity and MOD declaration | **MODIFY** | Accept the required declaration; use captured people per 10,000 and a lower-CI-over-MOD rule for confirmed material benefit. |
| D-5 V22 priorities | **MODIFY** | Raise repeated function/event histories and acute transitions; broaden available-at metadata. |
| D-6 cheap backfill | **DEFER** | Robustness/development only, never an untouched January confirmation. |
| D-7 protected versioned packages | **ACCEPT** | Needed when eventually implemented; this commit contains only a review. |
| M-1 FRID exposure; M-2 diagnosis recency | **MODIFY** | Accept the data direction; timed changes and ascertainment are more valuable than another static flag. |
| M-3 registry timing | **ACCEPT** | Validity first; remove unresolved availability from primary candidate rather than relying on a sensitivity to rescue it. |
| M-4 direct capacity tuning | **MODIFY** | Smooth tuning plus a small registered Top3 shortlist, not unrestricted TP maximization. |
| M-5 CCI ordinal | **MODIFY** | Verified order required; encoding is not new information. |
| M-6 restored registries | **DEFER** for prediction priority | Useful lineage/comparability, lower likely incremental gain. |
| M-7 separate prior-faller model | **DEFER** | Clinical interactions first; calibration failure does not establish distinct mechanisms. |
| M-8 death/competing risk | **MODIFY** | Audit estimand/censoring now; conditional survival-model development later. |
| M-9 stacking | **ACCEPT** decision not to pursue now | No demonstrated complementary errors worth the added selection layers. |
| M-10 neural/LLM features | **ACCEPT** decision not to pursue now | No compelling information gain or experiment economy. |
| Phase 6 transport triggers, label validation, subgroup rule | **MODIFY** | Precision/clinical margins, not universal constants; no survivor-selection primary analysis. |
| Phase 6 protected scoring/report architecture | **ACCEPT/MODIFY** | Reuse sealing/hashes; frozen fallback and operational capacity/censoring changes must be explicit. |
| Phase 7 multi-snapshot development | **MODIFY** | Forward-time validation plus availability embargo; grouping alone does not prevent temporal leakage. |
| Governance/privacy; silent prospective staging | **ACCEPT** | Amendments/access logs and clinical intervention evaluation remain essential. |

## 13. Five experiments if resources are limited

An “experiment” here is one **bounded scientific protocol**, not one fitted fold/model. Nested CV necessarily refits. Preregister all three development protocols before their outer results are read; reuse fold assignments, cached parameter-free history and saved predictions. The five-protocol budget includes management's clinical-understanding question and confirmation. Champion selection occurs **before experiment 5**, which evaluates the frozen choice.

| # | Protocol, fixed scope and hypothesis | Minimum outputs / stop decision |
|---|---|---|
| **1. Corrected Phase 5.1 baseline** | One repair with no automatic AUROC gate, strict train-local processing/lambda path and documented CCI. Corrected ENET OLD and OLD+admissible NEW are the paired operational comparison; reproduce the legacy ALL arm only for the historical audit if its membership differs from the admissible arm. LASSO is only an ENET endpoint. Reuse historical predictions for PRE/POST, without rerunning the historical model. | Repaired Top3/70% results, integrity/negative-control evidence, conditional paired uncertainty and provenance. Establish the safe comparator regardless of whether its recall falls below 50.4%. |
| **2. One representation comparison** | On the same safe universe compare corrected ENET with existing constrained XGB and one small penalized-spline GAM/ENET representation. Fixed log-loss/Top3 shortlist policy; no CatBoost/LightGBM sweep. Maximum 20 registered XGB configurations per outer-training selection; one declared spline basis and ≤3 interactions. | Does limited nonlinearity add a material/stable tail-ranking signal? Report every arm; no extra search if it fails. Reuse the ENET path from #1 when recipe/objective are identical; otherwise explicitly distinguish the smoother-tuning development arm from the historical repair. |
| **3. One new-information bundle** | Add a fixed, clinically reviewed bundle of verified medication-change, episode-history and functional-change variables to the registered development selection procedure. Include a small no-bundle/bundle choice in each inner loop. Choose backbone only within inner training, not by the highest previously seen outer result. | Added-value estimate of a prespecified selection procedure, coverage/availability and redundancy report. No source exists → defer this experiment; do not replace it with ten algorithms. Use baseline/pre-index data only and no later-test outcomes. |
| **4. Profiling plus one phenotype protocol** | Descriptive faller/non-faller table and limited clinical combinations, then faller-only Gower/PAM on a small domain-balanced panel, 2–5 clusters and 100 resamples. One declared representation sensitivity; full-population clusters deferred under this strict budget. | Management's two questions, stability/size/interpretability report and an explicit “no reproducible phenotypes” outcome if appropriate. No cluster IDs fed into predictive selection in this round. |
| **5. One sealed temporal confirmation** | Score the frozen champion and corrected ENET comparator on the same baseline-eligible later sample. July only if section 9 conditions pass; otherwise a later genuinely sealed/prospective sample. Blind predictions first, mature labels second. | Exact Top3 paired primary contrast and lower-CI-over-MOD success rule; transport/calibration/censoring report. Retain ENET if gain is inconclusive; no retuning on this sample. |

For economy cap the confirmatory candidate library at the corrected baseline procedures, XGB, one GAM representation and one locked longitudinal-bundle procedure; do not take the Cartesian product of all methods, sets, time windows and interactions. The full-development champion is obtained by repeating the registered inner selection. Its temporal result is the primary evidence; selecting the largest outer result and quoting it as a validated estimate is prohibited. If #3 cannot be completed before the model freeze, freeze the best eligible procedure from #1–2 by the registered rule and defer the bundle to a fresh future test.

Experiments 1–4 are proposed first development/clinical protocols, and experiment 5 is the first confirmatory protocol. Only #1 must precede new performance **claims**; extraction, feature specifications and descriptive-analysis design can proceed in parallel. This ordering need not impose an artificial wait until Phase 7 to discover whether new information is feasible.

## 14. Questions Claude must answer before implementation

1. Will automatic AUROC membership exclusion be removed? If retained, why is the fixed 0.80 bound a justified leakage detector, and how will both CV boundaries be respected without union-of-fold exclusions?
2. Which exact Windows source/config/fold/input hashes produced the supplied 1,086/2,155 captures? Does the published 3% estimate use exact within-fold quota rather than pooled score thresholding? This review has not accessed that run.
3. Does 97,453 mean all baseline-eligible people or only usable-labelled people? How many future deaths/losses are omitted, and how many would have occupied the Top3 quota?
4. What is the clinically intended event: any fall, medically attended fall, or recorded fall/fracture proxy? What confirms episode identity and label PPV/missed-event ascertainment by source?
5. What proves event **and availability** timing for prescriptions/purchases, registry membership updates, MEFI computation, diagnoses and hospital discharges? Which primary inputs remain only attested or uncertain?
6. What is the actual CCI code dictionary and order? How will undocumented nominal codes, unseen categories and source absence be handled across linear and tree paths?
7. Will the shared outer-derived lambda grid become a strictly inner-trained ratio/grid recipe? Which data-adaptive coverage/eligibility decisions still occur globally?
8. Which changes are the historical repair and which are improvement hypotheses? Will PRE/POST acknowledge the combined effect instead of attributing it to the screen alone?
9. What finite candidate library, tuning objective, shortlist/near-tie policy, compute cap, multiplicity policy and amendment rule will be committed before the new results?
10. Who declares the contact programme, operational denominator and MOD in absolute units? Is management willing to call a positive but imprecise result inconclusive rather than keep searching for 55%?
11. Has anyone developing the model seen July–October outcomes, partial counts or performance? Who controls the sealed extract? How will the October-frozen model's retrospective July timing be described?
12. How will the July predictor state be reconstructed as-of July 1, with a label freeze after December 28 plus measured lag? What replaces V21's August 1 research-end for mature July labels?
13. What are the death/censoring estimand and exact evaluation procedure? Will allocation precede label-based exclusions, and what occurs when missing follow-up prevents a primary operational claim?
14. Will management receive three distinct outputs: supervised profiling, faller-only phenotypes, and (later) population phenotypes? Who will clinically judge usefulness without seeing predictive outcome rankings?
15. Which repeated function/GUG/medication/fall/acute-care histories actually exist, with what coverage and timestamps? Can a single high-value bundle be specified without fabricating trajectories from current summary counts?
16. Will phenotype and model selection avoid test outcomes, including feature/component/cluster-count decisions? How will no stable clusters or no material model gain be reported?

### Final priorities

**P0 — MUST DO BEFORE ANY NEW MODEL CLAIM**

- Repair outcome-dependent eligibility and strict training boundaries; establish and freeze a corrected ENET comparator. Resolve CCI/nominal semantics and availability of primary inputs.
- Declare the baseline-eligible operational denominator, outcome/death/censoring contract, exact 3% allocation/ties, MOD and bounded selection policy.
- Seal and audit access to a genuinely untouched later label set; verify maturity, training chronology and as-of predictors. A code fix and nested CV alone do not validate an adaptive roadmap.

**P1 — HIGHEST CHANCE OF MATERIAL TOP3 IMPROVEMENT**

- Obtain time-resolved, as-of-verified medication changes, fall episodes and functional deterioration, then test one compact longitudinal bundle with ENET/constrained XGB.
- Compare a small nonlinear additive representation and existing XGB with the repaired ENET under a smoother tuning policy and a bounded Top3 shortlist.

**P2 — SCIENTIFIC / CLINICAL EXPLORATION**

- Faller profiling immediately after a registered analysis, then stable mixed-data faller phenotypes; population phenotyping later. Clinical usefulness need not imply prediction gain.
- Survival approaches when censoring/event-time information warrants them; site validation, justified backfill and multi-snapshot forward-time development under separate protocols.

**DROP — LOW ROI**

- Automatic AUROC feature vetoes, threshold/seed searches for 55%, standalone LASSO duplication, multiple interchangeable boosting libraries and exhaustive domain/model ablations.
- Untimed static feature proliferation, raw mixed-data K-means, phenotype-driven expert splitting, broad neural/ranking/stacking searches without a distinct information hypothesis.

### Explicit answers to the five final questions

1. **Single most likely route above ~50.4%:** add verified longitudinal information about recent medication change and fall/functional deterioration to the existing regularized ranking models. This is a hypothesis about newly available information, not a guarantee; re-expressing existing counts or changing a tree library alone is less likely to deliver a material gain.
2. **Is 55% defensible?** Yes as a working aspiration corresponding to about 100 extra captured fallers on the supplied denominator. No as an empirically justified expectation, tuning target, proof of success or reason to repeat searches. Success is a prespecified material paired improvement over the corrected comparator on an untouched test.
3. **What will clustering improve?** Clinical understanding is the more plausible value. Predictive benefit is uncertain and likely limited; it requires separate nested added-value evidence. Neither benefit is guaranteed, and no stable phenotype is an acceptable result.
4. **Fix/retrain Phase 5.1 first?** Fix and establish the corrected baseline before testing claims of new superiority. Data extraction, feature specifications and profiling/phenotyping design can proceed in parallel. Do not blindly retrain Claude's incomplete screen recipe or wait until a distant phase to prepare high-value data. This review implements and trains nothing.
5. **First five experiments?** (1) Corrected ENET baseline/legacy audit; (2) existing XGB plus one constrained nonlinear additive representation; (3) one verified longitudinal-information bundle; (4) faller profiling plus stable mixed-data faller-only phenotyping; (5) one sealed temporal champion-versus-corrected-ENET confirmation. Freeze the champion before the fifth experiment and stop this protocol if material gain is not established.
