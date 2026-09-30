# Review decisions (Phase 2)

Lead: Claude (lead engineer / data scientist). Inputs:
- `reviews/ASTRA_METHOD_REVIEW.md`: Astra = gpt-6-astra, 16,268 tokens, 1 of ≤ 2 consultations.
- `reviews/AGY_ENGINEERING_REVIEW.md`: AGY = gemini-3.1-pro-high, 21,689 tokens, 1 of ≤ 2 consultations.

**Cross-review:** none was requested. The two reviews cover different ground and no critical choice was contested between them. One
point touched both: Astra's nested CV adds compute, and AGY proposed single-threaded BLAS, which is slower. It is resolved in G-05
without affecting validity.

Decisions: ACCEPT, PARTIAL (accepted with a stated modification) or REJECT. "User rule" = an explicit instruction of the user that
takes precedence.

## Astra (statistical / clinical-prediction method)

| # | Recommendation | Decision | Reason | Implementation impact |
|---|---|---|---|---|
| A-01 | Replace 30–40 validation comparisons by TRAIN-based selection and ONE validation comparison of a small frozen shortlist (baseline + 2–3 challengers) | ACCEPT | Removes the winner's curse on a ≈ 256-event validation set; matches the user rule "feature discovery must use TRAIN" | `SELECTION_FROZEN.json` (S09) is written from TRAIN nested-CV results before VALIDATION can be read; a code-level validation seal plus an append-only validation registry (one unsealing only) |
| A-01b | (same) vs the user's requirement to show every waterfall step "on the same VALIDATION population" | PARTIAL | Both are kept by ordering: selection is frozen first. After unsealing, all pre-declared configurations are scored on VALIDATION as **descriptive** output that cannot change the frozen selection | S13 scores the shortlist (confirmatory) and the full waterfall/domain table (labelled descriptive, "not used for selection") |
| A-02 | Nested grouped CV: inner folds do tuning, feature selection, FP selection, imputation, scaling, redundancy filtering | PARTIAL | Accepted for every **supervised** step and for fitted transforms (imputation medians, scaling, FP forms, λ, l1_ratio, XGB parameters and tree count). Label-blind quality and redundancy filters run once on all TRAIN: they read no outcome, so they cannot leak it into the estimates | Outer 5-fold patient-grouped, outcome-stratified CV in TRAIN; inner 10-fold (linear) / 5-fold (XGB); the design builder is fitted inside each fold |
| A-03 | Treat VALIDATION as reused; refit the baseline under the same protocol; no validation-fitted recalibration | ACCEPT | Fair comparison: the Phase 1 recalibrated model is not the comparator | BASELINE_15 is refitted in the Phase 2 protocol. Fidelity to Phase 1 is checked without VALIDATION: the historical-method refit on full TRAIN is compared with the reference run's TRAIN coefficients (tolerance declared) |
| A-04 | Say "average precision" with a fixed implementation; report absolute Δ, paired intervals and prevalence | ACCEPT | Avoids AP vs trapezoidal ambiguity | Existing `metrics.pr_auc` is step-wise AP (sklearn-equivalent); every table says "AP (step-wise average precision)" and prints the partition prevalence |
| A-05 | Add log loss as a proper scoring rule and safeguard | ACCEPT | Probability quality must not silently degrade | Log loss in every metric bundle; tuning objective for all families (A-14) |
| A-06 | Pre-declare ONE principal operational capacity | ACCEPT | AP gains can coexist with worse capture at the capacity actually served | Principal capacity = **top 10%** (config `principal_capacity: 0.10`, changeable only before the plan is frozen); 1/3/5/20% are secondary |
| A-07 | Decision hierarchy: eligibility → AP gain → principal-capacity capture/PPV → probability quality → AUROC → simplicity | ACCEPT | Makes the advancement rule explicit | Encoded in `phase2/selection.py`, printed in the plan and the report |
| A-08 | Brier against a constant-risk reference and the baseline | ACCEPT | A small absolute Brier score at 2% prevalence says little | Scaled Brier (1 − Brier/Brier_null) reported |
| A-09 | Separate CITL (slope fixed at 1) from the jointly estimated intercept; uncertainty on calibration | ACCEPT | Standard; both are already implemented (`citl`, `calibration_slope_intercept`) | Both are reported with bootstrap intervals, plus the grouped calibration curve with event counts |
| A-10 | Calibration cutoffs as investigation triggers, not automatic rejection; "capture = recorded events identified, not falls prevented" | ACCEPT | Consistent with the user's "investigate before accepting" | Investigation stops (pause + report, continue only with a recorded reason); wording rule in all reports |
| A-11 | Small, strongly regularised XGBoost space: depth 1–3, min_child_weight 5–30 (log), reg_lambda 1–30 (log), learning rate 0.05, subsample 0.8, colsample 0.8, gamma = alpha = 0 | PARTIAL | Stage 1 (15 trials) uses Astra's space: 15 trials can explore 3 dimensions, not 8. The user's further parameters (subsample, colsample_bytree) are opened only in the optional stage 2 local refinement (≤ 8 trials) around the stage-1 optimum. Learning rate stays fixed at 0.05 | `configs/meuhedet/phase2.yaml` `xgb.stage1_space`, `xgb.stage2_space` |
| A-12 | Early stopping inside inner CV; outer folds untouched; final TRAIN fit with the median inner tree count | ACCEPT | Removes early-stopping optimism | Implemented as specified |
| A-13 | No class weighting; no monotone constraints initially | ACCEPT | Keeps probabilities interpretable; no clinically defensible monotone constraint across all encodings | `scale_pos_weight = 1`; no constraints |
| A-14 | Identical sets, folds, preprocessing boundaries and selection rules for all families; inner-CV log loss for tuning; outer-OOF AP for promotion | ACCEPT | A fair family comparison | One nested-CV engine for LASSO, EN and XGB |
| A-15 | Benchmark runtime before promising overnight completion | ACCEPT | Budget honesty | Rehearsal at real size records RUN_TIMINGS; the plan states a measured estimate |
| A-16 | SAFE_OR_UNRESOLVED is exploratory, not leakage clearance; a deployable model needs resolved availability | ACCEPT | Correct; BASELINE_15 itself contains UNRESOLVED registry/medication features | Two tracks: **EXPLORATORY** (SAFE + UNRESOLVED, contains BASELINE_15; main analysis, always labelled) and **PROVEN_SAFE** (sensitivity). Advancement to a temporal holdout is conditional on the new extract resolving timing |
| A-17 | A date proves safety only if it covers every source contribution of the aggregate | PARTIAL | The Phase 1 D-00 evidence kinds (source-level record dates) are kept unchanged, so the D-00 rule is not weakened. The assumption is made explicit per feature | Registry column `timing_assumption`; new open question Q-P2-01 for the warehouse team |
| A-18 | Assessments encode clinician suspicion/access; transport poorly | ACCEPT | Real risk | Subgroup "assessed vs not assessed" in SUBGROUP_SUMMARY; caveat text in both summaries |
| A-19 | Utilisation/fall counts: strict boundaries, episodes vs encounters, duplicates | PARTIAL | Aggregates cannot be rebuilt in Phase 2 (no row-level source) | Q-P2-02 for the warehouse team; limitation text |
| A-20 | The D00_CLEAN row deletion conditions inclusion on unavailable information; refitting on it cannot cure this | PARTIAL | Accepted as a **limitation**. Rejected as a Phase 2 change: the user fixed the modelling cohort, split and baseline on D00_CLEAN, and changing it would break comparability. The next extract must enforce `Event_Date < Index_Date` so no one is removed | Limitation section + next-step freeze list; no cohort change |
| A-21 | Predictor-level provenance contract (source, record dates, look-back, delay, missingness meaning, derivation) | ACCEPT | Needed for review | Columns of `02_ENGINEERED_FEATURE_REGISTRY.csv` |
| A-22 | Quarantine undocumented codes | ACCEPT | Uninterpretable and possibly mis-timed | Declared quarantine list in `configs/meuhedet/phase2_features.yaml` (e.g. Siudi_Status, Customer_Risk_Code, Malnutrition_Grade, Fall_Self_Report_Value and binaries whose "1" meaning is undocumented); disposition NEEDS_SME |
| A-23 | Death as non-event and exclusion of incomplete follow-up are limitations | ACCEPT | The labels come from the warehouse | Limitation text; next-step item |
| A-24 | Domain-level comparisons are the main evidence; refit and retune each increment/ablation within TRAIN; individual ablations are explanatory | PARTIAL | Domain additions and leave-one-domain-out are fully nested (retuned). Individual-feature ablations (≤ 15) keep each outer fold's tuned hyper-parameters fixed; this is labelled explanatory, and it saves ≈ 15 nested refits | S11 |
| A-25 | Separate "added value over baseline" (one-domain additions) from "unique contribution" (leave-one-domain-out) | ACCEPT | Correct distinction | Two columns in DOMAIN_INCREMENTAL_GAIN |
| A-26 | (a) Consistency assessment instead of mandatory directional agreement between OOF and validation | ACCEPT | A near-zero sign flip is not decisive | Reported as a consistency statement with intervals |
| A-27 | (c) Keep assessment status, avoid redundant dummies; call the no-indicator run a sensitivity | ACCEPT | Tree splits and missingness still reveal assessment | One "form assessed" indicator per form (items: positive dummy + item-unanswered dummy only if it differs from the form pattern); sensitivity set `ALL_REVIEWED_NO_ASSESSMENT_FLAGS` |
| A-28 | (d) Unresolved features = separately labelled exploratory track | ACCEPT | = A-16 | = A-16 |
| A-29 | (e) Drop the 1% row-exclusion threshold; do not remove people; reconstruct, mask where defensible, or exclude the source | PARTIAL | Accepted: no row removal, no threshold. **Masking rejected**: it replaces valid source values (against the standing rule that nothing is repaired); the value that would exist at prediction time (the previous assessment) is not in the extract; and a masked state can itself carry index-day information. A source proven UNSAFE is excluded and listed as "candidate after DWH fix" with its row counts | Registry disposition INELIGIBLE_UNSAFE; COLUMN_FUNNEL counts |
| A-30 | (f) Penalised predictions primary for expanded linear models; historical refit as benchmark | ACCEPT (with a correction) | Unpenalised refit amplifies selection instability. **Correction:** the brief said Phase 1 serves the unpenalised refit; the code shows Phase 1 predictions already use the penalised λ* coefficients, and the refit is only an odds-ratio table. Penalised-primary is therefore also the faithful BASELINE_15 method, and the "historical refit benchmark" has no historical counterpart | LASSO = penalised λ* for every set; no refit benchmark model (the odds-ratio table remains available from the final fits) |
| A-31 | (g) No either/or significance rule; pre-specify a minimum worthwhile gain and tolerances; "promising, not established" when uncertain | ACCEPT | Operationally grounded rule | Rule in EXPERIMENT_PLAN §6: ΔAP ≥ 0.010 (≈ 5% relative) **and** Δcapture@10% ≥ +1.0 percentage point (≈ +13 recorded falls per 180 days per 60,851 members), with probability-quality tolerances; uncertain gains are labelled "promising" |
| A-32 | (h) The waterfall is descriptive and order-dependent; separate set comparisons from family comparisons | ACCEPT | Correct | Waterfall labelled; one-domain and leave-one-domain-out tables beside it; family comparison on identical sets |
| A-33 | Paired patient bootstrap intervals ignore the search; P(Δ > 0) is not a posterior | ACCEPT | Correct | Wording rule; no "probability that" language |
| A-34 | Exploratory ablations/permutation/importance in TRAIN; 25 stability replicates are rough; consensus must not be a vote | PARTIAL | Importance, ablation and permutation run on TRAIN outer-OOF. The user asked that "strong" means support from multiple analyses; that is kept, but correlated importance measures count as **one** evidence class, and "strong" requires incremental evidence (domain/ablation gain) plus stability or importance | FEATURE_CONSENSUS rule in EXPERIMENT_PLAN §7 |
| A-35 | Requirements before a temporal claim (the freeze list) | ACCEPT | Needed for the next step | SCIENTIFIC_SUMMARY "Next scientific step" = freeze list |
| A-36 | Implausible gain / proxy dominance / calibration = audit triggers; timing, split contamination, prohibited inputs = hard stops | PARTIAL | User rule: "Stop rather than silently continuing". They stay **stops**, but they are investigation stops (pause + `INVESTIGATION_<gate>.md`; continue only with `--accept-gate` and a recorded reason). They never reject automatically | `phase2/gates.py` |

## AGY (architecture / robustness)

| # | Recommendation | Decision | Reason | Implementation impact |
|---|---|---|---|---|
| G-01 | Commit marker written after the directory rename, outside the item directory; complete iff marker exists and verifies | ACCEPT | A clean commit-record protocol with no ambiguous "tmp dir with a valid marker" state | `items/<item>/` then `items/<item>.COMPLETE.json` (temp → fsync → replace); anything without a verified marker → `_incomplete/` |
| G-02a | Preflight refuses an `--out` inside OneDrive / synced folders | ACCEPT | Sync locks and rewrites break hashes | STOP unless `--allow-synced-folder` (recorded) |
| G-02b | Retries for renames and replaces under AV/OneDrive locks | ACCEPT | Windows sharing violations are transient | `durable.rename_dir` / `replace_file` with back-off |
| G-02c | Read-only attribute on completed artifacts | ACCEPT | Stops accidental Excel saves | Applied to item outputs, stage outputs and `artifacts/` (never to RUN_STATE, logs, ledger) |
| G-03a | JSONL ledger = absolute truth; rebuild SQLite if out of sync | PARTIAL | Each trial is an item: its marker is the per-trial truth. The ledger is the append-only audit log, reconciled with the markers. SQLite is a derived cache, **rebuilt from the completed trials on every resume** (old file moved aside) | `phase2/xgb_tuning.py` |
| G-03b | Record the code hash at every stage and item | ACCEPT | Audit of resumed runs with patched code | Every marker carries `code_sha256`; RESUME_AUDIT maps items to code hashes |
| G-04a | Never open files inside `.tmp-*` / incomplete directories | ACCEPT | Truncated files | Move-only handling |
| G-04b | SQLite `synchronous=FULL`, WAL | ACCEPT | Cheap durability | Set on connect |
| G-04c | `FILE_FLAG_WRITE_THROUGH` | PARTIAL | `os.fsync` (FlushFileBuffers on Windows) before rename, plus the commit-record marker, gives the same guarantee without custom Win32 calls | Documented |
| G-05 | Pin BLAS threads to 1 | PARTIAL | Thread counts are pinned to a **fixed number recorded in the plan** (threadpoolctl; default min(4, cores)), and XGBoost `nthread` likewise. One thread would make the linear stages 2–4× slower; completed items are never recomputed, and the thread count is identical across resumes | `phase2/runtime.py` |
| G-06 | Do not mark stale RUNNING trials FAIL; remove them so the sampler has no memory; re-seed deterministically | ACCEPT | Achieved by rebuilding the study from completed trials on resume; the sampler seed = f(study seed, number of completed trials), so the interrupted trial is re-suggested identically | = G-03a |
| G-07a | Sequential items; library-level multithreading | ACCEPT | Simplest correct design for resume | No process pool |
| G-07b | Push BASELINE_15 through all stages first | PARTIAL | Within every stage BASELINE_15 items run first. Stage order stays, because the selection needs all candidates | Item ordering |
| G-08a | No beeswarm/dependence plots (they draw individual patients) | ACCEPT | Row-level data must not leave the PC | SHAP figures: mean \|SHAP\| bars + aggregated quantile summaries (no individual points, no outlier markers) |
| G-08b | Redact local paths and usernames; build share in `.tmp-share`, scan, rename only if clean | ACCEPT | Prevents leaking user names / paths | Scan for member ids, pseudonyms, pepper, `C:\Users\…`, the OS user name and data URIs; STOP on any hit |
| G-09-1 | Corrupted JSON marker → treat as missing and recompute | ACCEPT | Power-loss safety | JSONDecodeError → item incomplete |
| G-09-2 | Corrupted SQLite → rebuild | ACCEPT | = G-03a | = G-03a |
| G-09-3 | Known-answer numeric self-test at preflight | ACCEPT | Cheap guard against a broken numerical stack | Deterministic tiny LASSO and XGBoost fits compared with stored expected values (tolerance) |
| G-09-4 | Truncate a torn last JSONL line | PARTIAL | The torn tail is **moved to a sidecar** (`*.torn-<attempt>`), not silently dropped, then reconciled from the item markers | `durable.repair_jsonl` |
| G-09-5 | Detect antivirus by writing a dummy executable | REJECT | Deliberately writing executable-like files to provoke antivirus is inappropriate on a managed clinical PC. Retries + hash verification + resume already cover vanished or locked files | Runbook advice only |

## Lead decisions made during implementation (not from a reviewer; recorded for audit)

| # | Decision | Reason | Impact |
|---|---|---|---|
| L-01 | LASSO / EN λ grid ends at 10⁻³ × λ_max (Phase 1 default: 10⁻⁴ when p < n) | Measured on synthetic data at Phase 2 size: below ≈ 10⁻³ × λ_max rare binary features (e.g. zero events in a fold) are near-separated, so 16 of 100 λ values did not converge and one CV fit took 294 s instead of 48 s. λ* was ≈ 0.1 × λ_max with either grid (index 25 vs 33) | `lasso.lambda_min_ratio: 1.0e-3`; a λ* on the last grid point is flagged `lambda_at_grid_boundary` in the tables |
| L-02 | Exact duplicate design columns are dropped inside each fold (first kept, the rest recorded) | Identical columns (e.g. two nurse forms always completed on the same visit) make the penalised solution non-unique and stall coordinate descent | `LinearDesign.duplicates_` reported per fit; label-blind |
| L-03 | Redundancy clusters are representative-anchored (every member ≥ 0.90 with its representative), not connected components | Connected components chained unrelated features (minimum within-cluster \|ρ\| 0.02 on synthetic data) | 04_REDUNDANCY_CLUSTERS reports the minimum and maximum \|ρ\| with the representative |

## Open questions raised for the warehouse team (added to the plan)

- **Q-P2-01:** For each nurse form and for MEFI, does the form's record date bound *every* item value in the row (same assessment
  instance), or can items come from different assessment dates?
- **Q-P2-02:** Do visit/diagnosis/fall counts count encounters, episodes or code rows? Are duplicate codes collapsed?
- **Q-P2-03:** For the next extract, rebuild every source window with `Event_Date < Index_Date`, and add record dates for medications,
  registries (latest entry) and the comorbidity index, so that no cohort rows need to be removed.

## Final version (0.8.1): user rule and final Astra review

The user's final readiness brief and the final Astra parameter review (`reviews/ASTRA_FINAL_PARAMETER_REVIEW.md`) change the decisions below.
Every item is resolved in `reviews/ASTRA_FINAL_RESOLUTION.md`. The authoritative effective values are in
`configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json` and its `.sha256`.

| Earlier decision | Superseded by | Now |
|---|---|---|
| A-16 / A-28: EXPLORATORY track (SAFE + UNRESOLVED) as the main analysis, PROVEN_SAFE as a sensitivity | user rule (three categories, never mixed) | SAFE_DISCOVERY (proven SAFE only, new and baseline) is primary; HISTORICAL_BASELINE_15 is the benchmark (TRAIN-only refit, F-02); EXPLORATORY_UNRESOLVED_SENSITIVITY is a separate LASSO sensitivity. Hard stop if a SAFE set holds a non-SAFE feature |
| A-07 / A-31: advancement rule against LASSO:BASELINE_15 | user rule + Astra Q22 | advancement against LASSO:SAFE_BASE (SAFE incremental value); separate point-estimate class against the BASELINE_15 refit (MEETS_SUPERIORITY_RULE / WITHIN_NONINFERIORITY_MARGINS / OUTSIDE_MARGINS, F-03) |
| A-27: item "unanswered" indicator only above a 1% mismatch | F-04 | any NULL-pattern mismatch with the form gets an indicator |
| A-11: conditional XGBoost stage 2 | F-06 | stage 2 disabled; 15 stage-1 trials × 6 scopes |
| A-01b: validation registry | user §10 | plus the write-once `VALIDATION_OPENED.json` (finalists, configurations, hashes) |
| A-34: consensus classes | F-13 / F-14 | L = LASSO selection with LASSO stability; I = SHAP rank; permutation-only and EN-only are weak support; labelled "exploratory predictive evidence" |
| (new) outcome death semantics | F-05 | aggregate label audit + investigation gate LABEL_DEATH_SEMANTICS |
| (new) calibration uncertainty | F-11 | bootstrap intervals + smooth calibration curve |
| (new) attribution | F-09 | frozen FEATURE_EXPANSION and MODEL_FAMILY_PIPELINE contrasts |
