# Phase 3 design (falls_ml 0.9.0, Phase 3 1.0.0)

Frozen with `configs/meuhedet/PHASE3_FINAL_EXPERIMENT_CONFIG.json`. Isolated from Phase 2: separate Git branch / worktree, separate package folder
and virtual environment on Windows, separate `--out`; the Phase 2 code, configuration and frozen file are unchanged and Phase 2 is never read,
resumed or hashed (the optional `--phase2-out` reads two small status files only).

## 1. Question

Does the full, scientifically defensible Meuhedet clinical feature space (cognition, nurse assessments, frailty, function, medications, utilisation,
home environment, ...) identify more RECORDED falls within 180 days than the 15 historical predictors, at the same 10% intervention capacity, on the
same patients - and which domains carry the information?

## 2. Time contract and eligibility

Prediction at the END of Index_Date (planning/PHASE3_TIME_CONTRACT_AUDIT.md). Per feature: SAFE_VERIFIED (row-level dates), SAFE_VERIFIED_BOUNDED (a few
post-index / undated / unreadable rows, evaluated with bounds), SAFE_ATTESTED (no row-level date; DWH statement + V3), NOT_RECOVERABLE_FUTURE_RECORDS,
UNRESOLVED, FORBIDDEN, INELIGIBLE_DATA. Each feature also carries a declared information-availability risk and assumption.

Standards: PRIMARY_FULL (verified + bounded + attested) - the experiment; SENSITIVITY_VERIFIED_ONLY; SENSITIVITY_LOW_AVAILABILITY_RISK; EXPLORATORY_UNRESOLVED
(never selected).

## 3. Population and partitions

FULL_LABELED (the reference D00_CLEAN cohort rebuilt and verified row by row + the D-00 rows, reconciled exactly). The reference patient-grouped split is
reproduced (TEST sha256 checked) and TEST dropped in memory; D-00 rows get an outcome-blind keyed-hash partition. TRAIN: all tuning, selection and the
primary internal estimate (nested grouped CV, outer out-of-fold). VALIDATION: REUSED (Phase 1, Phase 2) - one-shot descriptive check after the frozen
selection. No partition is an untouched holdout.

## 4. Pipeline

S00 preflight / self-test -> S01 cohort + schema + time contract (V1-V9) + evaluation history -> S02 source timing -> S03 feature eligibility ->
S04 reconstruction + coverage -> S05 feasibility GO / NO_GO -> (GO) S06 frozen feature sets -> S07 LASSO -> S08 elastic net -> S09 XGBoost (Optuna, 15
trials per scope) -> S10 frozen selection -> S11 stability -> S12 ablation -> S13 explainability -> S14 reused-VALIDATION check -> S15 reports ->
S16 share (small cells suppressed; privacy, path, file-type and row-level scans fail closed). NO_GO jumps from S05 to S15 (Outcome B).

## 5. Sets

P3_BASE (the eligible historical predictors), P3_ALL_RECOVERED (everything eligible, redundancy-pruned), P3_BASE_PLUS_<domain>, the cumulative waterfall,
P3_ALL_RECOVERED_MINUS_<domain> (ablation), P3_ALL_RECOVERED_WITHOUT_ASSESSMENT_FLAGS, P3_ALL_NO_FALL_RECENCY (proxy ablation, LASSO + XGBoost),
P3_VERIFIED_BASE / _ALL, P3_LOWRISK_BASE / _ALL, HISTORICAL_B15 (only if some historical predictors are not eligible), the exploratory set (only if
UNRESOLVED features exist).

## 6. Selection (frozen before VALIDATION)

Candidates: PRIMARY_FULL main / one-domain / waterfall configurations (LASSO, elastic net, XGBoost). Qualify vs LASSO:P3_BASE on the ADVERSE bound:
dAP >= 0.010, d capture@10% >= +1.0 pp, log-loss ratio <= 1.01, calibration slope 0.80-1.25 and |CITL| <= 0.20 under both bounds. Highest adverse AP wins
(simplicity margin 0.005). Outcomes: INCREMENTAL_MODEL_SELECTED / INCONCLUSIVE_OVERWRITTEN_HISTORY / NO_INCREMENTAL_MODEL_SELECTED. Robustness verdicts
(ROBUSTNESS.json): attested sources, availability risk, fall recency.

## 7. Gates

Hard: TIME_CONTRACT_CONTRADICTED (V1 / V2), COHORT_RECONCILIATION (V9), SCHEMA_VIOLATION (structure), EXCLUDED_FEATURE_IN_SET, UNKNOWN_ROW_IN_FIT,
BOUNDS_SELFTEST, PRIVACY_SCAN, the Phase 2 protocol stops (input, plan, code, protected folders, frozen configuration). Investigation: BOUNDARY_EPISODE (V6),
EPISODE_CONTINUATION (V7), LABEL_DEATH_SEMANTICS, SINGLE_FEATURE_DOMINANCE, IMPLAUSIBLE_GAIN_OOF, IMPLAUSIBLE_CALIBRATION_OOF, PROXY_DOMINANCE.

## 8. Bounds (the safeguard)

For an UNKNOWN row the pre-index state of a source is the set of states observed among the fitting rows (cumulative upper bounds applied). Linear models:
exact min / max; XGBoost: sound outer bound by interval propagation. Adverse assignment = events lowest, non-events highest risk. Fitting uses KNOWN rows
only; evaluation uses every row. With no UNKNOWN row (the expected real case) the bounds equal the ordinary predictions.
