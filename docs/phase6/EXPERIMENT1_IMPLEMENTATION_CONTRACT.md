# Experiment 1 — implementation contract (Phase 5.1 methodological correction, REPAIR-ONLY)

| Item | Value |
|---|---|
| Status | REVISION 2, incorporating the PI's three required changes and decisions of 2026-10-07. No code has been changed. Implementation starts only after the PI approves this revision. |
| Scope | Experiment 1 of `docs/phase6/FINAL_CONSENSUS.md` only, **repair-only**. Every new selection or optimisation strategy is deferred to Experiment 2. |
| Base | falls_ml 0.12.3, Phase 5 2.2.0, commit `d150204` on `research/phase6-strategy` |
| Target | falls_ml **0.13.0**, Phase 5 **3.0.0** (scientific-graph change → major Phase version) |
| Author | Lead Scientist (Claude) |

**The one question Experiment 1 answers:** *what happens to the existing Phase 5 result when the methodological defects are corrected while changing as little else as possible?* Experiment 1 has **no success verdict**. Its deliverables are the corrected ENET comparator and the PRE/POST tables of section 7.

Terminology: **PRE** = the completed Phase 5 2.2.0 run on the work PC (folder `100k_falling_db_phase5_v3`, read-only). **POST** = the Phase 5.1 run of this contract, in a new output folder. **ADMISSIBLE** = the existing set `OLD_PLUS_NEW_SAFE` (OLD + new predictors with a SAFE timing class and DEFENSIBLE provenance), approved by the PI. **Legacy ALL** = the existing set `OLD_PLUS_ALL_NEW_ELIGIBLE`, audit/exploratory only. Internal set names are kept; the report shows the aliases.

---

## 1. Exact changes from Phase 5 2.2.0

| # | Change | Where (current code) | Nature |
|---|---|---|---|
| R-1 | **Cohort-level outcome-dependent AUROC exclusion removed.** `prepare` no longer calls `_univariate_auroc`; no feature can receive class `INELIGIBLE_LEAKAGE` from a number. The registry column `univariate_auroc` is kept, empty at preflight, with the note "see the forensic table in work\". | `src/falls_ml/phase5/data.py:161–168, 392–396` | Repair |
| R-2 | **Per-fold label-free coverage gate.** Inside every unit, before tuning: a feature with < `min_known_observed_rows` (100) finite values **on that unit's outer-training rows**, or constant on them, is dropped from that unit's X; recorded in `result.json` (`dropped_coverage`). The cohort-level coverage/constant/unreadable gates stay as label-free QA (identical to PRE). | `src/falls_ml/phase5/engine.py: run_unit` (before `tune_and_fit`) | Repair |
| R-3 | **Lambda grid inside each inner training fold.** `linear_path_task` computes `lambda_max` from its own inner-training design and the dimensionless grid `lambda_max × logspace(0, log10(lambda_min_ratio), n_lambda)`; candidates are aligned by grid index (ratio). The outer refit recomputes the grid on outer-training rows and uses the selected index. `_tune_linear` no longer builds `A_full`/`grids`. Per-fold `lambda_max` recorded in `trials.csv`. The hyper-parameter space (5 l1 ratios × 40 ratio steps, `lambda_min_ratio` 1e-3) is unchanged. | `engine.py:208–209, 232`; `models.py:23–27, 36–54` | Repair |
| R-4 | **CCI_Group quarantined unless an authoritative DWH dictionary exists before the run** (section 4). No ordinal spacing is inferred. | `configs/meuhedet/phase5.yaml` (new `feature_overrides`), `phase5/config.py`, `phase5/runner.py` | Repair |
| R-5 | **Explicit `other` level for nominal codes with learned levels.** Linear path: rare (< 10 training rows) and unseen finite codes → `f__other` column; missing stays zeros + NA indicator. Affects only features with learned levels: the smoking/obesity sub-codes (legacy ALL only) and CCI if the `nominal` branch is ever activated. **The OLD and ADMISSIBLE arms are unaffected.** Tree path untouched in Experiment 1. | `src/falls_ml/phase5/design.py:45–47, 81–95` | Repair (legacy ALL only) |
| R-6 | **Forensic univariate AUROC diagnostic (local, report-only).** Inside every unit, after R-2: AUROC of each feature in the unit's X against the **outer-training** labels, `max(a, 1−a)`; written to `result.json` (`forensic_auroc`) and aggregated into `work\FORENSIC_UNIVARIATE_AUROC_BY_FOLD.csv`. Any value ≥ 0.80 → WARN line in `OVERNIGHT_PROGRESS.log` and a one-line count in the scientific summary (number of features and folds flagged, no names). Kept in `work\` by default; published to `share\` only with `--share-forensic` after the privacy scan. It never changes membership. | `engine.py: run_unit`; `report.py` | Assurance |
| R-7 | **Selection policy unchanged.** The 2.2.0 inner objective (highest threshold reaching ≥ 70% inner-OOF sensitivity, then minimise share flagged; ties → lower false-alert share → lower share flagged → higher AP → lower Brier → simpler; tie resolution 1e-4), the same candidate set (l1 ratios {0.1, 0.25, 0.5, 0.75, 0.9} × 40 grid steps), one refit per unit, one holdout prediction per unit. No shortlist, no Top3-based selection, no near-tie rule, no added l1 ratio. | `engine.py: _tune_linear`, `thresholds.py: objective` | Unchanged |
| R-8 | **Families restricted to ENET; sets unchanged; units reduced.** Config v3 allows `families` to be a non-empty subset of {LASSO, ENET, XGB} containing `primary_family`; the plan records it; `plan_units` iterates the plan's families. Experiment 1: `families: [ENET]`. FINAL units for all three PRIMARY sets (today: OLD and ALL only). | `phase5/config.py:101–102`; `runner.py:436–458` | Scope |
| R-9 | **Single secondary contrast.** `ablations.base_set: OLD_PLUS_NEW_SAFE`, `remove_each_domain: false`, one block `NO_NEW_REGISTRY: {domains: [NEW_REGISTRY]}`, labelled SECONDARY DIAGNOSTIC in every table. `domain_families: []` (no DOMAIN units). It never enters the PRE/POST headline or any verdict. | `runner.py:189–236 (build_sets)`; `phase5.yaml: ablations, domain_families` | Scope |
| R-10 | **PRE verification and PRE/POST machinery (mandatory).** New CLI option `--pre-run <PRE folder>` (required for a real-data run; the preflight stops without it). Read-only. Before any fitting the preflight verifies: (a) the PRE plan is Phase 5 2.x and complete; (b) the PRE input sha256 equals the POST input sha256; (c) the POST usable cohort equals the PRE cohort (row-key set and labels); (d) `work\FOLDS.parquet` sha256 equals `folds_sha256` in the PRE plan, and the POST run **adopts** these folds (`folds_source: PRE` in the POST plan); (e) every PRE ENET PRIMARY unit of OLD / ALL / SAFE verifies against its `COMPLETE.json`; (f) `share\TOP3_CAPACITY_PRIMARY.csv` exists and is **reproduced exactly** (T, TP, PPV per model) from the PRE unit arrays by the POST capacity code. Any failure → STOP `PRE_VERIFICATION_FAILED`. The PRE folder digest is recorded before and after the POST run; POST never writes into it. | `runner.py` (preflight, report), `report.py`, `cli.py` | Assurance |
| R-11 | **Top 3% promoted into the main analysis, absolute numbers first.** The capacity machinery (`capacity.py`, arithmetic unchanged) runs inside the main report for every arm, with the paired bootstrap and discordance/overlap tables; the headline table is section 7's `TOP3_PRE_POST_HEADLINE.csv`. The historical 70% tables remain secondary audit outputs. | `analysis.py`, `report.py` | Reporting |
| R-12 | **Negative controls on real data** (`--negative-controls 10`, quick budget; section 8). | `runner.py` (new stage NEGATIVE_CONTROLS between PREFLIGHT and PRIMARY) | Assurance |
| R-13 | **Versioning and plan guards.** `PHASE5_VERSION = "3.0.0"`, falls_ml 0.13.0, `phase5.yaml: version: 3`; the loader refuses v2 settings; `_verify_plan` and the dashboard refuse a plan of another Phase 5 major version; new keys hashed into `config_sha256`. | `phase5/__init__.py`, `config.py`, `runner.py:549–572`, `dashboard.py` | Governance |
| R-14 | **Disclosure block** in `SCIENTIFIC_SUMMARY.md` / `_HE.md`: boundary table (what is cohort-level label-free QA, what is per-fold, what is inner), the combined-repair statement for PRE/POST, CCI disposition, lost OLD predictors, label caveats. | `report.py` | Reporting |

Phase 2/3/4 source and configuration files are not touched (the protected-manifest tests must keep passing). The CCI override lives in the Phase 5 settings, not in `phase2_features.yaml`.

**Deferred to Experiment 2 (not implemented here):** the log-loss one-SE shortlist, within-inner-fold Recall@Top3 selection, the near-tie rule, l1 ratio 1.0, any other selection or optimisation change, XGBoost's registered library, the additive basis, the tree-path `other` level. Experiment 2 will recompute the inner path on the same folds and recipe; Experiment 1 caches nothing for it.

## 2. What is removed from the current pipeline

- The cohort-level outcome-dependent AUROC exclusion and the class `INELIGIBLE_LEAKAGE` as a numeric outcome (the class survives only for the label-free Phase 3 mapping `NOT_RECOVERABLE_FORBIDDEN`).
- The shared lambda grid computed on the whole outer-training design.
- LASSO as a separately tuned family; XGBoost units and the Optuna path (not run in Experiment 1; code retained for Experiment 2).
- The DOMAIN stage (OLD + each domain), the per-domain ablations other than NO_NEW_REGISTRY, and the NO_FALL_RECENCY / NO_TIMING_UNCERTAIN blocks.
- SHAP tasks (no tree model); permutation importance is kept for the ENET folds (cheap).
- The five-criterion 70% verdict as the headline: it is still computed and printed as "historical audit (2.2.0 rule)"; the management summary opens with the Top 3% headline table.
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

- the forensic diagnostic R-6 (training-fold AUROC per feature, local, WARN, never exclusion; a value ≥ 0.80 on the real data requires a **written adjudication of availability** by the DWH before the POST result is used for any claim — an adjudicated violation is a logged amendment and a new run, never a fold-union veto);
- the boundary traps and permutation controls of section 8.

## 4. CCI_Group

PI decision: **quarantine unless an authoritative DWH dictionary is available before the run; never infer ordinal spacing.**

Mechanism implemented now: `phase5.yaml: feature_overrides` — a Phase 5-only override of catalogue metadata for named features, applied to the in-memory copy of the Phase 3 catalogue after `load_all`, hashed into `config_sha256`, recorded in the plan and in `FEATURE_ELIGIBILITY.csv` (`override: yes`). The protected Phase 2/3 files are not edited.

| Branch | Condition | Linear path | Missing / rare / unseen |
|---|---|---|---|
| `quarantine` | **default; no authoritative dictionary at registration** | not in any set; registry class INELIGIBLE_SEMANTICS, reason "CCI_Group dictionary pending (DWH)" | — |
| `ordinal` | DWH documents an **ordered** group scale with its levels | `kind: ordinal, levels: [...], linear: thermometer` (as MEFI; order only, no spacing assumed) | missing → zeros + NA indicator |
| `nominal` | DWH documents **meanings without order** | `kind: categorical, levels: [...], linear: onehot`, reference = the documented lowest/"no comorbidity" group | rare and unseen → `other`; missing → NA indicator |

The branch is fixed in `REGISTRATION_EXP1.md` before the POST preflight and cannot change on resume (config hash). No branch is ever chosen by an outer result. Under `quarantine`, CCI's PRE contribution (continuous) versus its POST absence is part of the combined repair effect and is stated in the disclosure block.

## 5. What stays identical for a fair PRE/POST comparison

| Element | PRE (2.2.0) | POST (3.0.0) |
|---|---|---|
| Input file | sha256 in the PRE plan | the same file; sha256 must match (STOP otherwise) |
| Cohort, outcome contract, usable patients, labels | as computed by unchanged code | asserted equal to PRE (row-key set and labels), STOP otherwise |
| Outer folds | `work\FOLDS.parquet`, seed 20261201 | **adopted from PRE**, sha256 verified against the PRE plan and recorded; not regenerated |
| Unit seeds, inner splits | `unit_seed(seed, family, outer)`, `inner_splits(y_tr, 5, seed+1)` | identical code and inputs → identical splits |
| Family | ENET (primary); LASSO and XGB also fitted | ENET only |
| Tuning objective and tie resolution | ≥ 70% sensitivity objective, 1e-4 | identical |
| Hyper-parameter space and budget (overnight) | l1 ratios {0.1, 0.25, 0.5, 0.75, 0.9}, n_lambda 40, lambda_min_ratio 1e-3, tol 1e-7, max_iter 10,000 | identical (the grid is now anchored per inner fold, R-3) |
| Set membership rules and names | safe/all classes, safe provenance; OLD, OLD_PLUS_ALL_NEW_ELIGIBLE, OLD_PLUS_NEW_SAFE | identical (membership may differ only through R-1, R-2 per fold and R-4, all reported in `PRE_POST_MEMBERSHIP.csv`) |
| Bootstrap, stability, privacy, capacity arithmetic | 2,000; 100; min cell 10; 3% half-up, largest remainder, row-key ties | identical |
| What differs (the combined repair) | — | no AUROC gate; per-fold coverage; inner lambda grid; CCI quarantined (or declared); `other` level for learned-level codes (legacy ALL only) |

PRE/POST reports the **combined** effect of these changes; no attribution to a single change is made.

## 6. Primary endpoint and primary comparison

**Primary comparison:** OLD and ADMISSIBLE (`OLD_PLUS_NEW_SAFE`), each under PRE 2.2.0 versus corrected Phase 5.1 (POST). Legacy ALL is audit/exploratory; NO_NEW_REGISTRY is a secondary diagnostic. Neither influences the headline or any verdict.

**Recall@Top3** on the development frame: N = usable labelled patients (identical to PRE); T = round-half-up(0.03 · N) (= 2,924 on the supplied cohort size); T allocated across the five outer folds by largest remainder proportional to fold sizes (ties → lower fold); within each fold the k_f highest holdout risks of that fold's model are selected (ties in deterministic pseudonymous row-key order); TP = Σ_f TP_f; Recall@Top3 = TP / Σ_f events_f; PPV@Top3 = TP / T; false interventions = T − TP.

**Headline statistic (absolute):** for OLD and for ADMISSIBLE, Δ captured falls = TP_POST − TP_PRE at the same T, on the same patients and folds, with a paired patient bootstrap 95% interval (2,000 replicates, identical multinomial weights for PRE and POST, capacity re-allocated from resampled fold sizes and top-k re-selected within fold in every replicate — existing `capacity.py` mechanics). Captured per 10,000 patients is reported **in addition**, never instead.

The management question answered first: *with the same ≈ 2,924 interventions, how many falls did the corrected model capture compared with PRE?*

The operational-denominator variant (quota among all baseline-eligible patients including censored) is **deferred** from Experiment 1; the censored/unlabelled count is reported from preflight check P5.

## 7. PRE vs POST comparison outputs (all aggregate, in `share\`)

| File | Content |
|---|---|
| `TOP3_PRE_POST_HEADLINE.csv` | **first table of both summaries.** Rows: OLD-PRE, OLD-POST, ADMISSIBLE-PRE, ADMISSIBLE-POST. Columns: patients N, falls (events), exact number selected T, falls captured TP, Recall@Top3, PPV@Top3, false interventions T − TP; then per arm: Δ captured falls POST − PRE, paired 95% CI, Δ Recall@Top3, Δ false interventions (= −Δ captured), Δ captured per 10,000 (additional) |
| `PRE_POST_CORRECTION.csv` | per arm × {overall, fold 0–4}: T_f, TP, Recall@Top3, PPV@Top3, AUROC, AP, Brier, calibration slope / intercept, historical 70%-rule false-alert share; PRE, POST, POST − PRE |
| `PRE_POST_PAIRED.csv` | PRE vs POST paired Δ captured falls with interval per arm; PRE vs POST top-3% selected-set overlap; discordant captured fallers (captured by PRE only / POST only) |
| `PRE_POST_CONTRAST.csv` | the ADMISSIBLE-vs-OLD paired contrast in PRE and in POST (Δ captured falls, interval, per 10,000, folds improved); legacy ALL-vs-OLD (audit); NO_NEW_REGISTRY-vs-ADMISSIBLE in POST (secondary diagnostic) |
| `PRE_POST_MEMBERSHIP.csv` | every feature: PRE class and sets, POST class and sets, reason for any difference (R-1 / R-2 per fold / R-4) |
| `TOP3_*` family (existing dashboard tables) | produced by the main report for every arm |
| `NEGATIVE_CONTROLS.csv` | section 8 |
| `PRECISION_PLANNING.csv` | from the POST ADMISSIBLE-vs-OLD development predictions: bootstrap SE of Δ captured falls, discordance counts, and the smallest MOD whose lower 95% limit would exceed 0 at this precision — planning information for the later MOD decision, not a result |
| `HISTORICAL_VERDICT_2_2_0.json` | the five-criterion 70% verdict recomputed on POST, labelled historical audit |
| `SCIENTIFIC_SUMMARY(.HE).md`, `MANAGEMENT_SUMMARY_HE.md` | open with `TOP3_PRE_POST_HEADLINE`; then the ADMISSIBLE-vs-OLD contrast; disclosure block; secondary diagnostic, audit arm and historical verdict last |
| `work\FORENSIC_UNIVARIATE_AUROC_BY_FOLD.csv` | local by default (R-6); the summaries carry only the flagged count |

## 8. Negative controls

| # | Control | Where it runs | Expected | Failure |
|---|---|---|---|---|
| N-1 | **Boundary traps** through the real input boundary on synthetic V21 extracts (test suite + synthetic smoke): `unknown_column` → STOP; `future_column` → sealed; a schema-declared **forbidden-lineage** column under a predictor-like name → sealed (new trap `forbidden_lineage`); `leaky_new` (AUROC ≈ 0.98, UNCERTAIN_TIMING, EXPERIMENTAL_COMPOSITE) → excluded from ADMISSIBLE by provenance, retained in legacy ALL, forensic WARN; a **weak outcome proxy** (AUROC ≈ 0.65) planted into an admissible column (new trap `weak_proxy`) → retained, forensic value below 0.80 (documents that no numeric gate exists); a **strong legitimate predictor** (planted AUROC > 0.80 in an admissible column, new trap `strong_legit`) → retained, forensic WARN, no exclusion | `tests/unit/test_phase5_contract.py`, `test_phase5_e2e.py`, `meuhedet-phase5-synthetic` | as listed | any deviation fails the test suite; the package is not built |
| N-2 | **Full-pipeline label permutations on the real data** (approved): `--negative-controls 10`, quick budget (n_lambda 8, ratios {0.2, 0.8}), ENET ADMISSIBLE, the 2.2.0 objective; for each of 10 fixed seeds the labels are permuted **within each outer fold** (frozen PRE folds); the complete unit pipeline runs (coverage gate, design, inner grid, selection, refit, holdout prediction); outer-OOF AUROC and within-fold Recall@Top3 recorded | work PC, after preflight, before PRIMARY | mean AUROC ≈ 0.50, mean Recall@Top3 ≈ 3% | mean AUROC > 0.55 **or** mean Recall@Top3 > 5% → STOP `NEGATIVE_CONTROL_FAILED`; investigate before PRIMARY |
| N-3 | **Holdout-label mutation with frozen split** (test): flipping every holdout label of a unit leaves its coverage drops, forensic values (training labels only), fitted design, inner grids, selected configuration, coefficients and holdout scores byte-identical | test suite (extends `test_unit_choices_do_not_depend_on_outer_labels`) | identical | test failure |
| N-4 | **Inner-validation boundary** (test): flipping the validation labels of one inner fold leaves that fold's fitted design, lambda grid and validation predictions unchanged; only the selection may change | test suite (new) | as stated | test failure |
| N-5 | **PRE folder immutability**: digest of the PRE folder before and after the POST run | work PC | identical | STOP `PRE_RUN_MODIFIED` |

## 9. Stopping and failure conditions

Unchanged hard stops: schema diff with undeclared columns; duplicate or NULL patient IDs; Leakage_Check_Ind non-zero; outcome contract O1–O7 incl. zero tolerance; x_guard; privacy scan; plan/config/schema/input/code mismatch on resume.

New: `PRE_VERIFICATION_FAILED` (any of R-10 a–f, including a PRE capacity table that the POST code cannot reproduce exactly); `NEGATIVE_CONTROL_FAILED` (N-2); `PRE_RUN_MODIFIED` (N-5); `PHASE5_VERSION_MISMATCH` (plan of another major version); a real-data run started without `--pre-run`.

Warnings that require written adjudication before any claim (not stops): any forensic AUROC ≥ 0.80 on real data; more than 5% non-converged path fits in any unit; the selected lambda on the grid edge in ≥ 3 folds (`lambda_on_grid_edge`, as in 2.2.0); the per-fold coverage gate dropping a feature in some folds but not others.

Protocol stop: the run is executed **once** per registration. Any change after the POST preflight (branch, budget, universe) is a dated amendment and a new output folder; no re-selection of thresholds, seeds, capacities or encodings after reading results.

## 10. Expected runtime on the work PC

Reference: the 2.2.0 overnight run on the same machine (its `RUN_TIMINGS.csv` gives the measured seconds per ENET unit; the runbook sized the full three-family plan at about 2–4 h for a 48k synthetic cohort on six cores). Experiment 1 removes all LASSO and XGB units, the DOMAIN stage and four of five ablation blocks; the ENET work per unit is unchanged.

| Stage | Units | Estimate (≈ 100k patients, default `--jobs`) |
|---|---|---|
| PREFLIGHT + PRE verification (R-10) | — | minutes |
| NEGATIVE_CONTROLS (N-2) | 10 seeds × 5 outer units, quick budget (2 ratios × 8 lambdas × 5 inner) | 0.5–1 h |
| PRIMARY | 15 ENET units (3 sets × 5 folds), 5 ratios × 40 lambdas × 5 inner paths + 1 refit each | 1–2 h |
| FINAL | 3 ENET units on all rows | 10–30 min |
| ABLATION (secondary diagnostic) | 5 ENET units (NO_NEW_REGISTRY on ADMISSIBLE) | 20–50 min |
| EXPLAIN + STABILITY | ENET permutation importance (10 repeats) + 100 stability refits | 30–60 min |
| REPORT + PRE/POST + dashboard + privacy scan | — | minutes |
| **Total** | | **≈ 2.5–5 h, one night**; confirmed by `--estimate` (updated for the v3 unit plan) before the real run |

Work-PC sequence: `--preflight-only --pre-run <PRE>` → `--negative-controls 10 --mode quick --pre-run <PRE>` → `--mode overnight --resume --pre-run <PRE>` → `meuhedet-phase5-dashboard` → send `share\` only.

## 11. Tests and deliverables

Tests added or changed: R-1 (no numeric exclusion; registry column empty), R-2 (per-fold gate, label-free, FINAL equals cohort gate), R-3 (grid from inner training only; outer refit grid from outer training; index alignment), R-4 (quarantine default; `ordinal`/`nominal` branches only via override; override hashed; Phase 2/3 files byte-identical), R-5 (`other` level, linear path), R-6 (forensic diagnostic never changes membership; local by default), R-7 (the selection on a 2.2.0-layout synthetic unit reproduces the 2.2.0 choice when the inner grid is held equal), R-8/R-9 (plan with `families: [ENET]`, FINAL for three sets, single secondary contrast never in the headline), R-10 (PRE verification a–f on a synthetic 2.2.0-layout folder incl. exact reproduction of its capacity table; read-only; digest identical; stop on each failure), R-12 (permutation stage, thresholds), R-13 (version guards), N-1/N-3/N-4; existing planted-gain and null worlds still return the expected historical verdicts; protected-manifest tests unchanged.

Deliverables: package `falls_ml_phase5_0.13.0_mailsafe.zip` with checksums; updated `docs/meuhedet/PHASE5_WORK_PC_RUNBOOK.md` (Experiment 1 sequence); `docs/phase6/REGISTRATION_EXP1.md` (frozen settings, CCI branch, hashes) committed **before** the real preflight; `docs/phase6/AMENDMENTS.md` created empty.

---

## 12. Decisions recorded and inputs still open

### Recorded (PI, 2026-10-07)

| Topic | Decision |
|---|---|
| Scope | Experiment 1 is repair-only; Policy B and every new selection strategy deferred to Experiment 2 |
| Primary comparison | OLD and ADMISSIBLE under PRE 2.2.0 vs corrected 5.1; NO_NEW_REGISTRY secondary diagnostic only; legacy ALL audit only |
| Headline | absolute Top 3% numbers first (selected, captured, Recall, PPV, false interventions, Δ captured with paired 95% CI); per 10,000 additional |
| CCI_Group | quarantine without an authoritative DWH dictionary; no inferred spacing |
| PRE provenance | required; input SHA, completed PRE folder, fold hashes and primary capacity table verified before fitting |
| ADMISSIBLE | the existing `OLD_PLUS_NEW_SAFE` |
| Forensic table | local in `work\` by default; shared only if privacy-safe (`--share-forensic`) |
| Negative controls | 10 frozen-fold permutations; stop at mean AUROC > 0.55 or mean Recall@Top3 > 5% |
| MOD | not a blocker; frozen before improvement/champion selection |
| Death/censoring semantics | required before the temporal validation; not a blocker for the paired repair |

### Still open (none blocks implementation; B-1 and B-2 must be settled before the real preflight)

| # | Input | Needed for | Default if absent |
|---|---|---|---|
| B-1 | **Data custodian**: the January 2026 extract on the work PC is byte-identical to the PRE input (sha256) and was not re-exported | R-10 (b); STOP otherwise | — |
| B-2 | **PRE folder**: `100k_falling_db_phase5_v3` intact, complete, read-only, available next to the input | R-10; STOP otherwise | — |
| B-3 | **CCI_Group dictionary** (optional): if an authoritative ordered or nominal dictionary arrives before registration, the corresponding branch is recorded in `REGISTRATION_EXP1.md` | R-4 | `quarantine` |
| B-4 | **Clinical lead**: Hebrew wording for the new tables (Top 3% headline, PRE/POST, negative controls) | report text | English labels with existing Hebrew glossary terms |
