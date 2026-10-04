# Phase 5 design – 2026 redevelopment + incremental value of the new V21 predictors

falls_ml 0.11.0, Phase 5 1.0.0. Branch `phase5` (from `phase4` da3ea1b). Settings: `configs/meuhedet/phase5.yaml`; pre-declared V21 catalogue:
`configs/meuhedet/phase5_v21_features.yaml`. Code: `src/falls_ml/phase5/`. Runbook: `docs/meuhedet/PHASE5_WORK_PC_RUNBOOK.md`.

## Question and endpoint

Do the V21 / 2026 predictors add information beyond the Phase 3 feature universe, and do they lower the FALSE-ALERT BURDEN AT >= 70%
SENSITIVITY? Design label: INTERNAL NESTED CROSS-VALIDATION ON 2026 SNAPSHOT (development data, never external validation). Phase 4 is frozen and
unused (its files are hash-verified unchanged: `configs/meuhedet/PHASE4_0.10.0_PROTECTED.sha256`).

## Data and sealing (`phase5/data.py`)

* X and the outcome are read separately. X goes through the Phase 4 sealed reader (`usecols`; a sealed column request raises). Sealed from X: the
  brief's outcome columns + label-end / follow-up / censoring columns, 2025-contract LABEL / FORBIDDEN_LEAKAGE / post-index columns, and new
  columns whose name looks like outcome / future / discharge / settlement / billing-lag information.
* Outcome: the Phase 4 outcome reader + contract (O1-O7) on the eligible rows; O1-O6 failures STOP before any model; extra audit items
  (events on / before the index day, beyond the horizon, class balance) are reported.
* Cohort: Index_Date 2026-01-01, Is_Eligible_Cohort = 1, a 0/1 label (censored rows reported, never imputed). Duplicate / NULL
  Customer_Full_ID -> STOP. Definition_Version must be `V21`; a non-zero Leakage_Check_Ind -> STOP (and is never taken as proof of safety).
* OLD = the Phase 3 universe (93 catalogue features + BASELINE_15) rebuilt with the unchanged Phase 1-3 code (adapter, `phase2.engineer`,
  `phase3.recovery.build_recovery` under the END-OF-INDEX-DAY contract, V3 re-checked on 2026 over the Phase 3 predictor record dates).
  Classes map to the brief taxonomy (SAFE_VERIFIED / SAFE_BOUNDED / SAFE_ATTESTED / INELIGIBLE_TIMING / _SEMANTICS / _DATA / _LEAKAGE).
* NEW = a 2026 column outside the 2025 contract that matches the pre-declared V21 catalogue (exact name, alias, or unambiguous name tokens).
  Timing: its record date (post-index / undated records -> UNKNOWN; Phase 3 gates G2 / G3), a day count (negative = post-index), or the
  attestation (V3 passed and no post-index V21 record date in its domain). Medication features are INELIGIBLE_TIMING when a purchase date is
  after the index date. Semantics: declared type / levels (> 1% non-conforming cells -> INELIGIBLE_SEMANTICS). Undeclared new columns are
  catalogued (INELIGIBLE_SEMANTICS) and listed in `V21_UNDECLARED_COLUMNS.csv` for review before the overnight run. Brief entries that are 2025
  columns (the MEFI fields) are OLD.
* Every eligible feature: >= 100 known non-NULL rows and not constant; single-feature AUROC >= 0.80 -> INELIGIBLE_LEAKAGE (exclusion only).
* An UNKNOWN / unreadable cell takes the feature's NO-RECORD state (0 for indicators / counts without a NULL state, NULL otherwise): no
  missingness pattern can carry the post-index record.

## Feature sets (`runner.build_sets`)

OLD; OLD_PLUS_NEW_SAFE (new classes in the Phase 3 primary rule: VERIFIED, BOUNDED, ATTESTED); OLD_PLUS_NEW_VERIFIED_ONLY (new SAFE_VERIFIED only);
OLD_PLUS_NEW_LOW_AVAILABILITY_RISK (new with risk LOW / MEDIUM); OLD + each new domain; OLD_PLUS_NEW_SAFE minus each pre-declared ablation block.
Identical sets share units (aliases). `x_guard` re-checks every set against the registry and the sealed list at the preflight and at every resume
(HARD STOP `X_LEAKAGE`).

## Nested CV (`engine.py`, `models.py`)

* One stratified outer assignment (5 folds; rows ordered by an outcome-blind key) for every family and set; inner 5-fold CV inside each outer
  training set. A unit receives y for the outer-training rows only.
* LASSO / ENET: the Phase 1-3 proximal-Newton coordinate-descent solver (`phase2.enet.enet_logistic_path`), lambda grid from the outer-training
  rows, warm-started paths, fold-fitted design (encodings, training medians, NULL indicators, standardisation). Worker processes (the solver loop
  holds the GIL).
* XGBoost: Optuna TPE with a per-trial deterministic sampler seed (a resumed study proposes the same trials); each trial = inner folds in threads,
  early stopping on a split of the INNER training rows; n_estimators of the refit = median inner best iteration; scale_pos_weight fixed at 1.
* Objective on inner OOF: the highest threshold with sensitivity >= 70%; minimise the share flagged; ties -> false-alert share -> AP -> Brier ->
  simpler. Thresholds for 50 / 60 / 70 / 75 / 80 / 90% come from the same inner OOF and are applied unchanged to the outer holdout.
* Domain / ablation units: linear families re-tuned in full; XGBoost re-uses the reference model's per-fold hyper-parameters (only the feature set
  changes; own inner early stopping and thresholds).
* Commit: `COMPLETE.json` (sha256 of every file) written last; XGBoost trials appended one by one.

## Analysis (`analysis.py`, `metrics.py`)

* NESTED operating results (fold thresholds; counts summed) = the unbiased operational estimate; POOLED OOF re-thresholded at exactly the
  target = descriptive / planning only.
* Paired patient bootstrap (multinomial weights, identical for both models): deltas of false-alert share, share flagged, PPV, FPR, sensitivity,
  false alerts / captured falls per 10,000, AP, AUROC, Brier, the descriptive 70% point.
* Decision rule (pre-declared, `phase5.yaml: decision`): USEFUL = lower false-alert share (nested AND descriptive) + paired CI below 0 AND a lower
  false-alert share in EVERY outer fold (the patient bootstrap holds the fitted models fixed; the synthetic null run showed a noise feature
  perturbing XGBoost / LASSO fits enough for a narrow interval below 0 - fold consistency catches it) + same
  direction without attested / bounded / high-risk features + calibration not materially worse (Phase 3 ranges or no worse than OLD; Brier CI
  not entirely above 0); point gain otherwise = PROMISING BUT NOT ROBUST; else NO ROBUST OPERATIONAL GAIN. Overall YES when >= 2 of 3 families
  are USEFUL; NO when none is USEFUL / PROMISING; else UNCERTAIN. No minimum clinically meaningful delta is invented.

## Outputs, privacy, operations

`share/` behind a fail-closed scan (identifiers and row keys, local paths, file types, row-level tables); counts 1-9 and every rate of such a row
suppressed; curves and the threshold table on a 0.5%-of-population grid. Row-level OOF predictions, the exhaustive threshold table, frames and
models stay in `work/`. `RUN_STATUS.json` (heartbeat, ETA, failures, retries), `RUN_TIMINGS.csv`, `OVERNIGHT_PROGRESS.log`. Ctrl+C -> INTERRUPTED
(exit 130); a failed unit is retried once, then recorded and the night continues. `--jobs` default 60% of the logical cores (max 12); single-threaded
BLAS in every worker; `--device auto` = CUDA only after a successful probe (the locked Windows build `xgboost-cpu` has none -> CPU, recorded).
