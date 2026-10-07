# Experiment 1 — implementation contract (Phase 5.1 methodological correction)

| Item | Value |
|---|---|
| Status | CONTRACT FOR APPROVAL. No code has been changed. Implementation starts only after the PI approves this document. |
| Scope | Experiment 1 of `docs/phase6/FINAL_CONSENSUS.md` only. Experiments 2–5 are not implemented. |
| Base | falls_ml 0.12.3, Phase 5 2.2.0, commit `c30065d` on `research/phase6-strategy` |
| Target | falls_ml **0.13.0**, Phase 5 **3.0.0** (scientific-graph change → major Phase version) |
| Author | Lead Scientist (Claude), 2026-10-07 |

Terminology: **PRE** = the completed Phase 5 2.2.0 run on the work PC (folder `100k_falling_db_phase5_v3`, read-only). **POST** = the Phase 5.1 run of this contract, in a new output folder. **ADMISSIBLE** = the existing set `OLD_PLUS_NEW_SAFE` (OLD + new predictors with a SAFE timing class and DEFENSIBLE provenance). **Legacy ALL** = the existing set `OLD_PLUS_ALL_NEW_ELIGIBLE` (adds UNCERTAIN_TIMING and non-defensible-provenance predictors). Internal set names are kept; the report shows the aliases.

---

## 1. Exact changes from Phase 5 2.2.0

| # | Change | Where (current code) | Nature |
|---|---|---|---|
| R-1 | **Cohort-level outcome-dependent AUROC exclusion removed.** `prepare` no longer calls `_univariate_auroc`; no feature can receive class `INELIGIBLE_LEAKAGE` from a number. The registry column `univariate_auroc` is kept, empty at preflight, with the note "see FORENSIC_UNIVARIATE_AUROC_BY_FOLD". | `src/falls_ml/phase5/data.py:161–168, 392–396` | Repair |
| R-2 | **Per-fold label-free coverage gate.** Inside every unit, before tuning: a feature with < `min_known_observed_rows` (100) finite values **on that unit's outer-training rows**, or constant on them, is dropped from that unit's X; the drop is recorded in `result.json` (`dropped_coverage`). The cohort-level coverage/constant/unreadable gates stay as label-free QA (identical to PRE). | `src/falls_ml/phase5/engine.py: run_unit` (before `tune_and_fit`) | Repair |
| R-3 | **Lambda grid inside each inner training fold.** `linear_path_task` computes `lambda_max` from its own inner-training design and the dimensionless grid `lambda_max × logspace(0, log10(lambda_min_ratio), n_lambda)`; candidates are aligned by grid index (ratio). The outer refit recomputes the grid on outer-training rows and uses the selected index. `_tune_linear` no longer builds `A_full`/`grids`. Per-fold `lambda_max` recorded in `trials.csv`. | `engine.py:208–209, 232`; `models.py:23–27, 36–54` | Repair |
| R-4 | **CCI_Group encoded by documented semantics or quarantined** (section 4). | `configs/meuhedet/phase5.yaml` (new `feature_overrides`), `phase5/config.py`, `phase5/runner.py` (override applied to the Phase 5 copy of the catalogue) | Repair |
| R-5 | **Explicit `other` level for nominal codes.** Linear path: learned-level one-hot gets an `f__other` column for rare (< 10 training rows) and unseen finite codes; missing stays zeros + NA indicator. Tree path: declared-level one-hot gets an `other` column. (Trees are not fitted in Experiment 1; the change is made once so Experiment 2 inherits it.) | `src/falls_ml/phase5/design.py:45–47, 81–95, 127–149` | Repair |
| R-6 | **Forensic univariate AUROC diagnostic (report-only).** Inside every unit, after R-2: AUROC of each feature in the unit's X against the **outer-training** labels, `max(a, 1−a)`; written to `result.json` (`forensic_auroc`), aggregated by the report into `FORENSIC_UNIVARIATE_AUROC_BY_FOLD.csv`; any value ≥ 0.80 → WARN line in `OVERNIGHT_PROGRESS.log` and a report section. It never changes membership. | `engine.py: run_unit`; `report.py` | Assurance |
| R-7 | **Two selection policies on the same inner path.** The inner ENET path is computed once per inner fold (as now). Policy A (**HISTORICAL_70**, the repair arm): the 2.2.0 objective — highest threshold reaching ≥ 70% inner-OOF sensitivity, minimise share flagged, ties as in 2.2.0 — restricted to the PRE l1 ratios {0.1, 0.25, 0.5, 0.75, 0.9}. Policy B (**LOGLOSS_TOP3**, the improvement arm): unweighted inner-OOF log loss → shortlist within one SE of the best (SE = standard deviation of the per-inner-fold log loss of that configuration / √5) → among the shortlist, within-inner-fold Recall@Top3 (k_f = round-half-up(0.03 · n_f) per inner validation fold, TP summed) → near ties (fewer than one captured patient per inner fold on average, i.e. ΔTP < 5 over the five inner folds) resolved toward fewer non-zero coefficients, then the larger lambda; over all six ratios including the LASSO endpoint 1.0. Each unit refits both selected configurations on outer training and predicts the holdout twice: `p_test_A`, `p_test_B`, two pickles. | `engine.py: _tune_linear`; `thresholds.py` (new `objective_logloss_top3`) | Repair arm = A; improvement arm = B |
| R-8 | **l1 ratio 1.0 added to the inner path** (one extra path per inner fold). Policy A ignores it (exact PRE candidate set); Policy B uses it. LASSO is no longer a separate family. | `phase5.yaml: modes.*.enet.l1_ratios`; `_tune_linear` | Repair (policy B only) |
| R-9 | **Families restricted to ENET; sets unchanged; units reduced.** Config v3 allows `families` to be a non-empty subset of {LASSO, ENET, XGB} containing `primary_family`; the plan records it; `plan_units` iterates the plan's families. Experiment 1: `families: [ENET]`. FINAL units for all three PRIMARY sets (today: OLD and ALL only). | `phase5/config.py:101–102`; `runner.py:436–458` | Scope |
| R-10 | **Ablation base set configurable; single contrast.** `ablations.base_set: OLD_PLUS_NEW_SAFE`, `remove_each_domain: false`, one block `NO_NEW_REGISTRY: {domains: [NEW_REGISTRY]}`. `domain_families: []` (no DOMAIN units). | `runner.py:189–236 (build_sets)`; `phase5.yaml: ablations, domain_families` | Scope |
| R-11 | **PRE/POST machinery.** New CLI option `--pre-run <PRE folder>`: read-only; verifies the PRE plan (phase5_version 2.x, same input sha256), asserts the POST usable cohort equals the PRE cohort (row-key set and labels), adopts the PRE outer folds (`work/FOLDS.parquet`, sha256 recorded in the POST plan as `folds_source: PRE`), reads the committed PRE ENET unit arrays (COMPLETE.json hashes verified) and builds `PRE_POST_CORRECTION.csv` and the paired tables of section 7. The PRE folder's digest is recorded before and after; the POST run never writes into it. | `runner.py` (preflight, report), `report.py`, `cli.py` | Assurance |
| R-12 | **Within-fold Top3 promoted into the main analysis.** The capacity machinery (`capacity.py`, unchanged arithmetic) runs inside the main report for every arm and both policies, with the paired ADMISSIBLE-vs-OLD bootstrap and discordance/overlap tables; the historical 70% tables remain as secondary audit outputs. | `analysis.py`, `report.py` | Reporting |
| R-13 | **Negative controls on real data** (`--negative-controls N`, default 10, quick budget; section 8). | `runner.py` (new stage NEGATIVE_CONTROLS between PREFLIGHT and PRIMARY) | Assurance |
| R-14 | **Versioning and plan guards.** `PHASE5_VERSION = "3.0.0"`, falls_ml 0.13.0, `phase5.yaml: version: 3`; the loader refuses v2 settings; `_verify_plan` and the dashboard refuse a plan of another Phase 5 major version; new keys hashed into `config_sha256`. | `phase5/__init__.py`, `config.py`, `runner.py:549–572`, `dashboard.py` | Governance |
| R-15 | **Disclosure block** in `SCIENTIFIC_SUMMARY.md` / `_HE.md`: boundary table (what is cohort-level label-free QA, what is per-fold, what is inner), the combined-repair statement for PRE/POST, CCI disposition, lost OLD predictors, label caveats. | `report.py` | Reporting |

Phase 2/3/4 source and configuration files are not touched (the protected-manifest tests must keep passing). The CCI override lives in the Phase 5 settings, not in `phase2_features.yaml`.

## 2. What is removed from the current pipeline

- The cohort-level outcome-dependent AUROC exclusion and the class `INELIGIBLE_LEAKAGE` as a numeric outcome (the class survives only for the label-free Phase 3 mapping `NOT_RECOVERABLE_FORBIDDEN`).
- The shared lambda grid computed on the whole outer-training design.
- LASSO as a separately tuned family; XGBoost units and the Optuna path (not run in Experiment 1; code retained for Experiment 2).
- The DOMAIN stage (OLD + each domain), the per-domain ablations other than NO_NEW_REGISTRY, and the NO_FALL_RECENCY / NO_TIMING_UNCERTAIN blocks.
- SHAP tasks (no tree model); permutation importance is kept for the ENET folds (cheap).
- The five-criterion 70% verdict as the headline: it is still computed and printed as "historical audit (2.2.0 rule)"; the management summary opens with the Top3 tables.
- The synthetic trap semantics "`leaky_new` → excluded by AUROC": replaced by "excluded from ADMISSIBLE by provenance/timing, retained in legacy ALL, flagged by the forensic diagnostic".

## 3. Exact replacement for the AUROC eligibility screen

Membership is decided **only** by label-free rules, all of which already exist:

1. **Sealing**: outcome / label / follow-up / censoring / identifier columns and future-looking name patterns never enter X (`x_sealed_map`, unchanged).
2. **Schema classes**: OUTCOME_OR_FUTURE_FORBIDDEN, IDENTIFIER, METADATA_OR_ADMIN never enter X; REQUIRES_SEMANTIC_REVIEW stops the run (unchanged).
3. **Timing classes**: SAFE_VERIFIED / SAFE_BOUNDED / SAFE_ATTESTED admissible; UNCERTAIN_TIMING in legacy ALL only; INELIGIBLE_TIMING / _SEMANTICS / _DATA never (unchanged `_membership`).
4. **Provenance**: DEFENSIBLE admissible; UNVALIDATED_CODES and EXPERIMENTAL_COMPOSITE in legacy ALL only (unchanged).
5. **V3 attestation** on the real file (unchanged).
6. **Label-free QA gates**: cohort-level minimum known rows (100), constant, unreadable share ≤ 1% (unchanged, QA), plus the new per-fold coverage gate R-2.
7. **x_guard** re-checks every set for sealed, future, identifier or ineligible fields (HARD STOP, unchanged).

Assurance that replaces the screen's safety purpose:

- the forensic diagnostic R-6 (training-fold AUROC per feature, report and WARN, never exclusion; a value ≥ 0.80 on the real data requires a **written adjudication of availability** by the DWH before the POST result is used for any claim — an adjudicated violation is a logged amendment and a new run, never a fold-union veto);
- the boundary traps and permutation controls of section 8.

## 4. CCI_Group

Mechanism implemented now (independent of the dictionary):

- `phase5.yaml: feature_overrides` — a Phase 5-only override of catalogue metadata for named features, applied to the in-memory copy of the Phase 3 catalogue after `load_all`, hashed into `config_sha256`, recorded in the plan and in `FEATURE_ELIGIBILITY.csv` (`override: yes`). The protected Phase 2/3 files are not edited.
- Three selectable branches for `com_cci_group`:

| Branch | Condition | Linear path | Tree path (Experiment 2) | Missing / rare / unseen |
|---|---|---|---|---|
| `ordinal` | DWH documents an **ordered** group scale with its levels | `kind: ordinal, levels: [...], linear: thermometer` (as MEFI) | raw numeric code | missing → zeros + NA indicator |
| `nominal` | DWH documents **meanings without order** | `kind: categorical, levels: [...], linear: onehot` with reference level = the documented "no comorbidity / lowest" group | one-hot with `other` | rare (< 10 training rows) and unseen → `other`; missing → NA indicator |
| `quarantine` | **no dictionary at registration** (default) | not in any set; registry class INELIGIBLE_SEMANTICS, reason "CCI_Group dictionary pending (DWH)" | — | — |

- No branch is ever chosen by an outer result; the branch is fixed in the registration before the POST preflight and cannot change on resume (config hash).

**Blocked pending the DWH dictionary**: which branch is active. If the dictionary is not delivered before registration, Experiment 1 runs with `quarantine`; CCI's PRE contribution (continuous) versus its POST absence is then part of the combined repair effect and is stated in the disclosure block.

## 5. What stays identical for a fair PRE/POST comparison

| Element | PRE (2.2.0) | POST (3.0.0) |
|---|---|---|
| Input file | `60k/100k_falling_db_2026.csv`, sha256 in PRE plan | the same file; sha256 must match (STOP otherwise) |
| Cohort, outcome contract, usable patients, labels | as computed by unchanged code | asserted equal to PRE (row-key set and labels), STOP `COHORT_MISMATCH` otherwise |
| Outer folds | `work/FOLDS.parquet`, seed 20261201 | **adopted from PRE**, sha256 recorded; not regenerated |
| Unit seeds, inner splits | `unit_seed(seed, family, outer)`, `inner_splits(y_tr, 5, seed+1)` | identical code and inputs → identical splits |
| ENET budget (overnight) | n_lambda 40, l1 ratios {0.1, 0.25, 0.5, 0.75, 0.9}, lambda_min_ratio 1e-3, tol 1e-7, max_iter 10,000 | identical for policy A; ratio 1.0 added for policy B only |
| Tuning objective of the repair arm | ≥ 70% sensitivity objective, tie resolution 1e-4 | identical (policy A) |
| Set membership rules | safe/all classes, safe provenance | identical |
| Feature sets | OLD, OLD_PLUS_ALL_NEW_ELIGIBLE, OLD_PLUS_NEW_SAFE | identical names and rules (membership may differ only through R-1 and R-4, both reported) |
| Bootstrap, stability, privacy, capacity arithmetic | 2,000; 100; min cell 10; 3% half-up, largest remainder, row-key ties | identical |
| What differs (the combined repair) | — | no AUROC gate; per-fold coverage; inner lambda grid; CCI branch; `other` level for nominal codes (affects legacy ALL only, where the sub-codes live) |

PRE/POST reports the **combined** effect of these changes; no attribution to a single change is made.

## 6. Primary endpoint

**Recall@Top3** on the development frame: N = usable labelled patients (identical to PRE); T = round-half-up(0.03 · N); T allocated across the five outer folds by largest remainder proportional to fold sizes (ties → lower fold); within each fold the k_f highest holdout risks of that fold's model are selected (ties in deterministic pseudonymous row-key order); Recall@Top3 = Σ_f TP_f / Σ_f events_f; PPV@Top3 = Σ_f TP_f / T. Computed per policy (A, B) and per arm (OLD, ADMISSIBLE, legacy ALL, ADMISSIBLE minus NEW_REGISTRY).

**Comparative statistic**: Δ captured fallers per 10,000 patients = 10,000 · (TP_ADMISSIBLE − TP_OLD) / N, paired patient bootstrap with 2,000 replicates and identical multinomial weights for both models, capacity re-allocated from resampled fold sizes and top-k re-selected within fold in every replicate (existing `capacity.py` mechanics); also ΔPPV = ΔTP / T and ΔFP = −ΔTP. Discordance table: fallers captured by only one model; selected-set overlap.

The operational-denominator variant (quota among all baseline-eligible patients including censored) is **deferred** from Experiment 1: it needs censored rows kept in the frame and scored by fold, which touches the sealed-read path; the censored/unlabelled count is reported from preflight check P5.

## 7. PRE vs POST comparison outputs (all aggregate, in `share\`)

| File | Content |
|---|---|
| `PRE_POST_CORRECTION.csv` | per arm × policy A (PRE has only policy A) × {overall, fold 0–4}: Recall@Top3, PPV@Top3, TP, AUROC, AP, Brier, calibration slope / intercept, historical 70%-rule false-alert share; PRE value, POST value, POST − PRE |
| `PRE_POST_PAIRED.csv` | PRE vs POST paired Δ captured per 10,000 with bootstrap interval, per arm (same patients, same folds); PRE vs POST top-3% selected-set overlap and discordant captured fallers |
| `PRE_POST_CONTRAST.csv` | the ADMISSIBLE-vs-OLD paired contrast in PRE and in POST (Δ captured per 10,000, interval, folds improved), plus legacy ALL-vs-OLD |
| `PRE_POST_MEMBERSHIP.csv` | every feature: PRE class and sets, POST class and sets, reason for any difference (R-1 / R-2 per fold / R-4) |
| `FORENSIC_UNIVARIATE_AUROC_BY_FOLD.csv` | feature × outer fold training-AUROC (POST), with the PRE cohort-level value where PRE recorded one |
| `TOP3_*` family (existing dashboard tables) | now produced by the main report for every arm and policy |
| `NEGATIVE_CONTROLS.csv` | section 8 |
| `PRECISION_PLANNING.csv` | from the POST ADMISSIBLE-vs-OLD development predictions: bootstrap SE of Δ captured per 10,000, discordance counts, and the smallest MOD whose lower 95% limit would exceed 0 at this precision — planning information for D-4, not a result |
| `HISTORICAL_VERDICT_2_2_0.json` | the five-criterion 70% verdict recomputed on POST, labelled historical audit |
| `SCIENTIFIC_SUMMARY(.HE).md`, `MANAGEMENT_SUMMARY_HE.md` | open with the POST Top3 tables and the PRE/POST table; disclosure block; historical verdict last |

## 8. Negative controls

| # | Control | Where it runs | Expected | Failure |
|---|---|---|---|---|
| N-1 | **Boundary traps** through the real input boundary on synthetic V21 extracts (test suite + synthetic smoke): `unknown_column` → STOP; `future_column` → sealed; a schema-declared **forbidden-lineage** column present under a predictor-like name → sealed (new trap `forbidden_lineage`); `leaky_new` (AUROC ≈ 0.98, UNCERTAIN_TIMING, EXPERIMENTAL_COMPOSITE) → excluded from ADMISSIBLE by provenance, retained in legacy ALL, forensic WARN; a **weak outcome proxy** (AUROC ≈ 0.65) planted into an admissible column (new trap `weak_proxy`) → retained, forensic value reported below 0.80 (documents that no numeric gate exists); a **strong legitimate predictor** (planted AUROC > 0.80 in an admissible column, new trap `strong_legit`) → retained, forensic WARN, no exclusion | `tests/unit/test_phase5_contract.py`, `test_phase5_e2e.py`, `meuhedet-phase5-synthetic` | as listed | any deviation fails the test suite; the package is not built |
| N-2 | **Full-pipeline label permutations on the real data**: `--negative-controls 10`, quick budget (n_lambda 8, ratios {0.2, 0.8, 1.0}), ENET ADMISSIBLE, both policies; for each of 10 fixed seeds the labels are permuted **within each outer fold** (frozen folds); the complete unit pipeline runs (coverage gate, design, inner grid, selection, refit, holdout prediction); outer-OOF AUROC and within-fold Recall@Top3 recorded | work PC, after preflight, before PRIMARY | mean AUROC ≈ 0.50, mean Recall@Top3 ≈ 3% | mean AUROC > 0.55 **or** mean Recall@Top3 > 5% → STOP `NEGATIVE_CONTROL_FAILED`; investigate before PRIMARY |
| N-3 | **Holdout-label mutation with frozen split** (test): flipping every holdout label of a unit leaves its coverage drops, forensic values (training labels only), fitted design, inner grids, selected configurations (A and B), coefficients and holdout scores byte-identical | test suite (extends `test_unit_choices_do_not_depend_on_outer_labels`) | identical | test failure |
| N-4 | **Inner-validation boundary** (test): flipping the validation labels of one inner fold leaves that fold's fitted design, lambda grid and validation predictions unchanged; only the selection may change | test suite (new) | as stated | test failure |
| N-5 | **PRE folder immutability**: digest of the PRE folder before and after the POST run | work PC | identical | STOP `PRE_RUN_MODIFIED` |

## 9. Stopping and failure conditions

Unchanged hard stops: schema diff with undeclared columns; duplicate or NULL patient IDs; Leakage_Check_Ind non-zero; outcome contract O1–O7 incl. zero tolerance; x_guard; privacy scan; plan/config/schema/input/code mismatch on resume.

New: `COHORT_MISMATCH` (POST usable cohort or labels differ from PRE); `FOLDS_MISMATCH` (PRE folds file hash differs from the PRE plan); `NEGATIVE_CONTROL_FAILED` (N-2); `PRE_RUN_MODIFIED` (N-5); `PHASE5_VERSION_MISMATCH` (plan of another major version).

Warnings that require written adjudication before any claim (not stops): any forensic AUROC ≥ 0.80 on real data; more than 5% non-converged path fits in any unit; policy B selecting a configuration on the grid edge in ≥ 3 folds; the per-fold coverage gate dropping a feature in some folds but not others (reported per fold).

Protocol stop: the run is executed **once** per registration. Any change after the POST preflight (branch, budget, universe) is a dated amendment and a new output folder; no re-selection of thresholds, seeds, capacities or encodings after reading results.

## 10. Expected runtime on the work PC

Reference: the 2.2.0 overnight run on the same machine (its `RUN_TIMINGS.csv` gives the measured seconds per ENET unit; the runbook sized the full three-family plan at about 2–4 h for a 48k synthetic cohort on six cores). Experiment 1 removes all LASSO and XGB units, the DOMAIN stage and four of five ablation blocks, and adds one l1 ratio (+20% per inner path), one extra outer refit per unit and the negative controls.

| Stage | Units | Estimate (≈ 100k patients, default `--jobs`) |
|---|---|---|
| PREFLIGHT + PRE verification | — | minutes |
| NEGATIVE_CONTROLS (N-2) | 10 seeds × 5 outer units, quick budget (3 ratios × 8 lambdas × 5 inner) | 0.5–1.5 h |
| PRIMARY | 15 ENET units (3 sets × 5 folds), 6 ratios × 40 lambdas × 5 inner paths + 2 refits each | 1–2.5 h |
| FINAL | 3 ENET units on all rows | 10–30 min |
| ABLATION | 5 ENET units (NO_NEW_REGISTRY on ADMISSIBLE) | 20–50 min |
| EXPLAIN + STABILITY | ENET permutation importance (10 repeats) + 100 stability refits | 30–60 min |
| REPORT + PRE/POST + dashboard + privacy scan | — | minutes |
| **Total** | | **≈ 3–6 h, one night**; confirmed by `--estimate` (updated for the v3 unit plan) before the real run |

Work-PC sequence: `--preflight-only --pre-run <PRE>` → `--negative-controls 10 --mode quick --pre-run <PRE>` → `--mode overnight --resume --pre-run <PRE>` → `meuhedet-phase5-dashboard` → send `share\` only.

## 11. Tests and deliverables

Tests added or changed: R-1 (no numeric exclusion; registry column empty), R-2 (per-fold gate, label-free, FINAL equals cohort gate), R-3 (grid from inner training only; outer refit grid from outer training; index alignment), R-4 (three CCI branches; quarantine default; override hashed; Phase 2/3 files byte-identical), R-5 (`other` level in both paths), R-6 (forensic diagnostic never changes membership), R-7 (policy A reproduces the 2.2.0 choice on a 2.2.0-layout synthetic unit; policy B shortlist and near-tie rule), R-9/R-10 (plan with `families: [ENET]`, FINAL for three sets, single contrast), R-11 (PRE/POST from a synthetic 2.2.0-layout folder, read-only, digest identical), R-13 (permutation stage, thresholds), R-14 (version guards), N-1/N-3/N-4; existing planted-gain and null worlds still return the expected verdicts under both policies; protected-manifest tests unchanged.

Deliverables: package `falls_ml_phase5_0.13.0_mailsafe.zip` with checksums; updated `docs/meuhedet/PHASE5_WORK_PC_RUNBOOK.md` (Experiment 1 sequence); `docs/phase6/REGISTRATION_EXP1.md` (frozen settings, CCI branch, hashes) committed **before** the real preflight; `docs/phase6/AMENDMENTS.md` created empty.

---

## 12. Decisions and inputs required from the PI, management and the DWH

| # | Needed | Blocks | Default if absent |
|---|---|---|---|
| B-1 | **CCI_Group dictionary** with level meanings and whether they are ordered | choice of the CCI branch (section 4) | `quarantine` |
| B-2 | **Death / censoring semantics** of `Fall_Next_180D_Ind`: which of death before a fall, disenrolment and administrative censoring yield NULL, 0 or 1 | interpretation of the usable/censored counts and the Experiment 5 estimand; **not** Experiment 1's execution (labels are used unchanged) | reported as "semantics pending", counts shown |
| B-3 | **MOD** (minimal operational difference) in captured fallers per 10,000 at 3% | interpretation of `PRECISION_PLANNING.csv` and Experiment 5; **not** Experiment 1's execution | planning placeholder 5 per 10,000, labelled as such |
| B-4 | **PRE provenance**: confirmation that `100k_falling_db_phase5_v3` is the completed 2.2.0 run, intact, available read-only on the work PC next to the identical input file, and that the supplied 1,086 / 2,155 / 2,924 come from `TOP3_CAPACITY_PRIMARY.csv` | PRE/POST (R-11); the run proceeds without PRE/POST if the folder is unavailable, with the comparison deferred | POST runs standalone; PRE/POST deferred |
| B-5 | **Admissible universe sign-off**: ADMISSIBLE = `OLD_PLUS_NEW_SAFE` (attested registries included under the Phase 3 standard), NO_NEW_REGISTRY as the single source-risk contrast, legacy ALL as audit only | R-9/R-10 | as stated |
| B-6 | **Approval of policy B details**: one-SE shortlist, near-tie tolerance (< 1 captured patient per inner fold), l1 ratio 1.0 added for policy B only | R-7/R-8 | as stated |
| B-7 | **Approval of the forensic diagnostic as a shareable aggregate** (feature name × fold × training AUROC) | R-6 output in `share\` | kept local in `work\` only |
| B-8 | **Negative-control thresholds** (mean AUROC > 0.55 or mean Recall@Top3 > 5% → stop) and N = 10 seeds | R-13 | as stated |
| B-9 | **Clinical lead** confirms Hebrew wording for the new tables (Top3, PRE/POST, forensic, negative controls) | report text | English labels with existing Hebrew glossary terms |
| B-10 | **Data custodian** confirms the January 2026 extract on the work PC is byte-identical to the PRE input (sha256) and has not been re-exported | STOP otherwise | — |

None of B-2, B-3 and B-9 blocks the implementation or the run; B-1, B-4, B-5, B-6, B-7, B-8 and B-10 must be settled before `REGISTRATION_EXP1.md` is frozen and the real preflight starts.
