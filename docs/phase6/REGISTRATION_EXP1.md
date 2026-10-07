# Registration — Experiment 1 (Phase 5.1 repair-only correction, Phase 5 3.0.0)

| Item | Value |
|---|---|
| Status | REGISTERED SETTINGS of the approved contract (`docs/phase6/EXPERIMENT1_IMPLEMENTATION_CONTRACT.md`, commit `d7ab81d`). Frozen before any real-data preflight. Any later change is a dated entry in `docs/phase6/AMENDMENTS.md` **before** the affected data are read, and a new output folder. |
| Software | falls_ml **0.13.0**, Phase 5 **3.0.0** (`PHASE5_VERSION`), settings `configs/meuhedet/phase5.yaml` version 3 (its sha256 is recorded in every plan; a run refuses any other settings file) |
| Real data | **none read** at registration; nothing trained. The real run waits for the PI's explicit approval after the implementation evidence is reviewed. |

## 1. The question

What happens to the existing Phase 5 2.2.0 result when the methodological defects are corrected while changing as little else as possible? Experiment 1 has **no success verdict**; its deliverables are the corrected ENET comparator and the PRE / POST tables.

## 2. What is identical to PRE (2.2.0) by construction

| Element | Enforcement |
|---|---|
| Input file | sha256 must equal the PRE plan's input sha256 (`PRE_VERIFICATION_FAILED` otherwise) |
| Usable cohort and labels | row-key set and labels asserted equal to PRE (`COHORT_MISMATCH`) |
| Outer folds | **adopted** from the PRE `work/FOLDS.parquet` after its sha256 matches the PRE plan; never regenerated (`folds_source: PRE` in the POST plan) |
| Seeds | plan seed 20261201; `unit_seed(seed, family, outer)`; inner splits `inner_splits(y_tr, 5, seed + 1)` (unchanged code) |
| Family | ENET only (`families: [ENET]`; the loader refuses a list without ENET; LASSO / XGB code retained for Experiment 2, not fitted) |
| Tuning objective | the 2.2.0 objective: highest threshold reaching ≥ 70% inner-OOF sensitivity, then minimise the share flagged; tie resolution 1e-4 (`thresholds.objective`, unchanged) |
| Candidate space | overnight ENET: n_lambda 40, l1 ratios {0.1, 0.25, 0.5, 0.75, 0.9}, lambda_min_ratio 1e-3, tol 1e-7, max_iter 10,000 — **the loader refuses any other overnight ENET budget or solver setting** (`REGISTERED_OVERNIGHT_ENET`, `REGISTERED_ENET_SOLVER`) |
| Feature sets | OLD, OLD_PLUS_ALL_NEW_ELIGIBLE (legacy ALL, audit only), OLD_PLUS_NEW_SAFE (= ADMISSIBLE, the PI-approved comparator); membership rules unchanged (`eligibility.safe_classes`, `all_new_classes`, `safe_provenance`) |
| Capacity arithmetic | T = round-half-up(0.03 · N), largest-remainder allocation across the adopted folds, within-fold top-k, deterministic row-key ties (`capacity.py`, unchanged) |
| Bootstrap, stability, privacy | 2,000 paired replicates; 100 stability refits; min cell 10; the fail-closed share scan |

## 3. The repair (the only differences from 2.2.0)

| # | Change | Where |
|---|---|---|
| R-1 | the cohort-level outcome-dependent single-feature AUROC exclusion is REMOVED; membership is label-free | `phase5/data.py: prepare` |
| R-2 | per-fold label-free coverage gate (≥ 100 known rows, non-constant) on each unit's outer-training rows | `phase5/engine.py: coverage_gate`, `tune_and_fit` |
| R-3 | lambda grid anchored inside each inner training fold (dimensionless ratio grid; outer refit re-anchored on outer training) | `phase5/models.py: linear_path`, `ratio_grid`; `engine.py: _tune_linear` |
| R-4 | CCI_Group **QUARANTINED** (`feature_overrides.com_cci_group.branch: quarantine`): no authoritative DWH dictionary was supplied; no order or spacing inferred | `phase5.yaml`, `phase5/data.py: apply_catalogue_overrides` |
| R-5 | explicit `other` level for nominal codes with learned levels (linear path; legacy ALL only) | `phase5/design.py: _encode` |
| R-6 | training-fold forensic univariate AUROC, local and report-only, WARN at ≥ 0.80, never an exclusion | `phase5/engine.py: univariate_auroc`; `work/FORENSIC_UNIVARIATE_AUROC_BY_FOLD.csv` |

Everything else (Policy B, new l1 ratios, new families, additive bases, clustering, V22 features, temporal validation) is Experiment 2 or later and is not reachable from this version (`tests/unit/test_phase51_repair.py::test_no_experiment2_logic_reachable`).

## 4. Assurance

| Control | Setting |
|---|---|
| PRE verification (R-10) | checks (a) completed 2.x PRE run, (b) input sha256, (c) cohort and labels, (d) fold hash adopted, (e) ENET PRIMARY units complete and hash-verified, (f) `share/TOP3_CAPACITY_PRIMARY.csv` reproduced exactly from the PRE arrays; any failure → `PRE_VERIFICATION_FAILED` before anything is fitted; the PRE folder digest is re-checked at report time (`PRE_RUN_MODIFIED`) |
| Negative controls (R-12) | 10 frozen-fold label permutations (ENET, ADMISSIBLE, quick budget); **registered, fixed thresholds** mean AUROC ≤ 0.55 and mean Recall@Top3 ≤ 0.05 (the loader refuses any other value); a failure → `NEGATIVE_CONTROL_FAILED`: nothing is fitted and no scientific report is published; a real run without passed controls → `NEGATIVE_CONTROLS_REQUIRED` |
| Boundary traps (N-1) | synthetic test suite: `unknown_column`, `future_column`, `forbidden_lineage` → fail closed; `leaky_new` → legacy ALL only by provenance, forensic WARN; `weak_proxy` → retained (no numeric gate exists); `strong_legit` → retained, forensic WARN |
| Label-mutation tests (N-3, N-4) | flipped holdout labels leave the coverage gate, forensic values, designs, grids, selection and holdout scores unchanged; flipped inner-validation labels never reach the inner design or grid |

## 5. Outputs and reporting rules

The first table of both summaries is `TOP3_PRE_POST_HEADLINE.csv`: per arm (OLD, ADMISSIBLE) and version (PRE 2.2.0, POST 5.1, DELTA) — N, falls, exact selected count at 3%, captured falls, Recall@Top3, PPV@Top3, false interventions, Δ captured falls with its paired 95% CI, Δ Recall@Top3, Δ false interventions; per 10,000 only in addition. Then `PRE_POST_PAIRED.csv`, `PRE_POST_CONTRAST.csv` (ADMISSIBLE − OLD in PRE and POST; legacy ALL − OLD audit; NO_NEW_REGISTRY − ADMISSIBLE secondary diagnostic), `PRE_POST_CORRECTION.csv`, `PRE_POST_MEMBERSHIP.csv`, `PRE_POST_HISTORICAL_70_RULE.csv`, `PRECISION_PLANNING.csv`, `NEGATIVE_CONTROLS.csv`, `HISTORICAL_VERDICT_2_2_0.json` (the 2.2.0 five-criterion rule as a historical audit only). The PRE → POST difference is reported as the **combined** effect of the repair.

## 6. Run order on the work PC (CMD)

1. `--preflight-only --pre-run <PRE>` (nothing fitted; PRE verified; folds adopted)
2. `--negative-controls --pre-run <PRE>` (10 permutations with their own registered quick budget; the folder keeps its mode; hard stop on failure)
3. `--mode overnight --resume --pre-run <PRE>` (the run; resumable with the same command)
4. `meuhedet-phase5-dashboard --out <OUT>` (reporting only)
5. send back `<OUT>\share` only

## 7. Decisions recorded (PI, 2026-10-07)

Repair-only scope; OLD and ADMISSIBLE as the primary PRE / POST arms; NO_NEW_REGISTRY secondary diagnostic only; legacy ALL audit only; absolute Top-3% headline; CCI quarantine; PRE provenance required; ADMISSIBLE = `OLD_PLUS_NEW_SAFE`; Policy B deferred to Experiment 2; forensic table local by default (`--share-forensic` only if privacy-safe); negative controls approved with the thresholds above; MOD and death / censoring semantics are not blockers for this paired repair.
