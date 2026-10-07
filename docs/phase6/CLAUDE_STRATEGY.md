# Phase 5.1 / Phase 6 improvement strategy — Lead Scientist proposal

| Item | Value |
|---|---|
| Status | PROPOSAL FOR INDEPENDENT REVIEW. Nothing in this document has been implemented. No modelling code was changed, no model was trained or re-trained, no real data were read. |
| Author role | Lead Scientist (Claude) |
| Base commit | `2398fb9391f449330054f1b6900aceeec1b6a37b` on `main` — "chore: include verified 0.12.3 handoff package and delivery evidence" |
| Software state | falls_ml **0.12.3**, Phase 5 **2.2.0** (`VERSION`, `src/falls_ml/phase5/__init__.py:17`) |
| Branch | `research/phase6-strategy` (strategy documents only) |
| Date | 2026-10-07 |
| Intended readers | the PI; the independent Astra/Codex reviewer; later the clinical, biostatistical and management reviewers |
| Decision owner | the PI (section 10). The reviewer's recommendations are resolved ACCEPT / PARTIAL / REJECT with reasons, as in `reviews/REVIEW_DECISIONS.md`. |

**Reading guide for the reviewer.** The repository facts this proposal rests on are, in reading order: `planning/PHASE5_DESIGN.md`, `docs/phase5/PREPROCESSING_AUDIT_0.12.2.md`, `configs/meuhedet/phase5.yaml` (the pre-declared rules), `src/falls_ml/phase5/data.py` (`prepare`, lines 171–439) and `src/falls_ml/phase5/analysis.py` (`decide`, lines 110–143). Appendix A maps every claim below to a file and line.

---

## 0. Summary

Phase 5 (2.2.0) answers one pre-declared question on the 2026-01-01 snapshot with internal nested cross-validation: *at about 70% fall sensitivity, does the new V21 information lower the false-alert burden?* A completed real-data run exists on the work PC; its results are not in the repository and were not read for this document.

The 0.12.3 source audit found one **material methodological defect** — the single-feature AUROC leakage screen uses the outcomes of the whole usable cohort *before* the outer folds exist (`data.py:367–409`, `runner.py:285–307`) — and one **modelling weakness** (the Charlson group entered as a continuous number). The audit could not say whether either changed the real result.

I propose two gated tracks and one parallel data track:

- **Track A — Phase 5.1 (weeks).** A *correction, not a redesign*: the same question, the same snapshot, the same folds, the same decision rule. First a read-only triage of the completed run's eligibility table (minutes, no training) decides whether the defect was binding. Then a nested re-run with the screen moved inside the outer folds, the Charlson group encoded label-blind as ordinal, negative controls on the real data, and a pre/post comparison table so the correction's effect is visible rather than hidden. Output: a defensible Phase 5 estimate and frozen candidate models.
- **Track B — Phase 6 (months).** The scientific step no re-run on the 2026-01-01 snapshot can supply: a **temporal validation of the frozen Phase 5.1 models on a later snapshot (Index_Date 2026-07-01)**, built on the Phase 4 frozen-model machinery, with the estimand re-anchored at the **operational capacity (3%)** rather than at 70% sensitivity, a management-declared minimal operational difference, and a pre-registered decision rule. Earliest extract: after the 180-day labels mature plus documentation lag, i.e. about February 2027.
- **Track C — data and governance (parallel).** A prioritised V22 data request (bounded medication exposure, per-flag diagnosis dates, registry record-creation timestamps, restoration of the three removed V1 registries, label-source audit), the capacity and minimal-difference declarations by management, and pre-registration of Phase 6 before its data exist.

What this proposal deliberately does **not** do: change the label definition inside Phase 5.1; add algorithm families, class weighting, deep learning or LLM-derived features; reuse the 2026-01-01 outcomes as "validation" (they have now served as Phase 4 validation and Phase 5 development); or retrain anything before the triage is read and the reviewer round is closed.

---

## 1. Verified project state (what exists at the base commit)

| Area | Fact | Where |
|---|---|---|
| Repository | `https://github.com/yossi-gavriel/FallingPrediction`, `main` at `2398fb9` | `git log` |
| Versions | falls_ml 0.12.3; Phase 5 2.2.0; design label "INTERNAL NESTED CROSS-VALIDATION ON 2026 SNAPSHOT" | `VERSION`; `src/falls_ml/phase5/__init__.py:17–18` |
| Phase 5 code | 20 modules, 6,014 lines under `src/falls_ml/phase5/` | `wc -l` |
| Phase 5 tests | 30 contract, 14 end-to-end, 5 capacity, 2 dashboard (synthetic only) | `tests/unit/test_phase5_*.py` |
| Settings | `configs/meuhedet/phase5.yaml` (version 2, seed 20261201, 5×5 nested CV, families LASSO/ENET/XGB, primary ENET, sets OLD / OLD_PLUS_ALL_NEW_ELIGIBLE / OLD_PLUS_NEW_SAFE, five-criterion decision rule) | `phase5.yaml` |
| Schema | authoritative V21 (224 columns): 201 shared, 20 removed, 23 new (22 candidate predictors + 1 metadata column); study window 2022-01-01 … 2026-08-01; label window 180 days | `configs/meuhedet/phase5_v21_schema.yaml`; `planning/PHASE5_DESIGN.md` |
| Removed V1 predictors | Registry_Blood_Pressure_Ind, Registry_Chronic_Renal_Failure_Ind, Registry_Transplant_Ind, Siudi_Status; lineage to the new COVID / dialysis / immunosuppression registries NOT proven → the three OLD features leave the OLD set | `PHASE5_DESIGN.md` (registry lineage); `phase5_v21_schema.yaml: lineage` |
| New predictor domains | NEW_DIAGNOSIS (dizziness, gait abnormality, syncope, tremor, osteoporosis, parkinsonism, stroke), NEW_VISION_HEARING (cataract, hearing loss, vision impairment), NEW_REGISTRY (dialysis, COVID-19, immunosuppression, smoking, obesity, oncology, IBD, opiate, severe function; smoking / obesity sub-codes UNVALIDATED_CODES), NEW_FRAILTY_OR_RISK_PROXY (Deficit_Count_Proxy, EXPERIMENTAL_COMPOSITE, UNCERTAIN_TIMING) | `phase5_v21_schema.yaml: columns` |
| Audits | Preprocessing audit verdict **MATERIAL_ISSUE_RETRAINING_SHOULD_BE_CONSIDERED**; dashboard safety audit (reporting only, no fitting); SQLite lifecycle patch (shared Phase 2/3 helper; Phase 5 unaffected) | `docs/phase5/*.md` |
| Real-data state | A completed Phase 5 2.x run exists on the work PC (`100k_falling_db_phase5_v3`); the 0.12.3 dashboard command was prepared for it. No real result is in the repository. | `dist/DELIVERY_0.12.3.md` |
| Earlier phases | Phase 3 (0.9.0) developed on the 2025-01-01 snapshot; Phase 4 (0.10.0) validated the frozen Phase 3 models on the 2026-01-01 snapshot; Phase 5 then *developed* on that same 2026-01-01 snapshot | `planning/PHASE3_DESIGN.md`, `PHASE4_DESIGN.md`, `PHASE5_DESIGN.md` |
| Snapshot availability | the DWH holds monthly snapshots of the wide view (49 at Phase 1) | `docs/meuhedet/PHASE1_REPORT.md:24` |

Planning magnitudes used below (to be replaced by the real values in `COHORT_FACTS_2026.json` of the completed run; they are *not* results): about 100,000 eligible patients per snapshot (folder name), 180-day recorded-fall prevalence of the order of 2% (Phase 2 planning text: +1 percentage point of capture at 10% ≈ 13 falls per 60,851 members).

---

## 2. What Phase 5 can and cannot establish

**It can establish**, within the 2026-01-01 snapshot and the pre-declared rule: a fair paired comparison of OLD versus OLD_PLUS_ALL_NEW_ELIGIBLE on identical patients, folds, preprocessing rules and tuning budgets; the operating read-off at 3% capacity (dashboard, within-fold ranking); which new columns are eligible and why.

**It cannot establish**:

1. *Transportability in time.* There is no later snapshot. Everything is internal cross-validation on development data, which the design label states.
2. *An absolute performance claim.* The 2026-01-01 snapshot shares patients with 2025 (Phases 2–4) and has itself been used twice: as the Phase 4 temporal-validation set of the Phase 3 models and as the Phase 5 development set. Any further use of its outcomes is development, never validation.
3. *The deployable operating point.* The primary endpoint is anchored at ≥ 70% sensitivity; the management question is anchored at 3% capacity. At a prevalence of the order of 2%, these two points lie far apart on the risk ranking, and a gain at one does not imply a gain at the other. The 0.12.2 dashboard already opens the management summary with the 3% question; the *verdict* is still the 70% one.
4. *Label validity.* The outcome is a recorded-diagnosis proxy (code 888 or fracture prefixes 820/821/813/812/808; repeated records may document one event; no encounter setting). Death within the window is a non-event; positives after the personal Followup_End_Date stop the run (zero tolerance).
5. *Freedom from the preflight defect.* The outcome-dependent eligibility screen ran on all labels before the folds existed. Its practical effect is unknown until the completed run's eligibility table is read (section 5.1).

---

## 3. Lead Scientist's review of Phase 5 2.2.0 — findings and disposition

Severity: **M** material to validity, **S** strategic (changes what the result means for the decision), **D** disclosure or minor. Disposition: 5.1 (Track A), 6 (Track B), V22 (Track C), DISCLOSE.

| # | Finding | Sev. | Evidence | Disposition |
|---|---|---|---|---|
| F-01 | The single-feature AUROC ≥ 0.80 leakage screen is computed on the outcomes of the whole usable cohort, and feature-set membership is fixed, *before* `outer_folds` is called. Outer-holdout labels therefore influence which predictors every outer model may use. Exclusion-only intent does not remove the statistical dependence. | M | `data.py:161–168` (`_univariate_auroc`), `data.py:367–409`; `runner.py:285–307` (prepare → sets → folds at 307) | 5.1: triage T-01, change C-01 |
| F-02 | Primary estimand (false-alert share at the inner-selected ≥ 70% rule) versus the operational estimand (falls captured at 3% capacity) can differ in sign. The verdict rule reads only the 70% point. | S | `phase5.yaml: decision`; `capacity.py` docstring; `dashboard.py` | 6: estimand re-anchored; 5.1: 3% paired comparison promoted into the main outputs (C-03), verdict unchanged |
| F-03 | Criterion (2)'s equal-sensitivity interval re-derives one global 70% threshold on **pooled** out-of-fold predictions of five different outer models, i.e. it mixes probability scales — exactly what the capacity dashboard deliberately avoids by ranking within folds. | M/D | `analysis.py:119–124`; `metrics.py:101–110` (`desc_fas_0.70` on pooled `p`); `capacity.py` docstring | 5.1: within-fold companion (C-04), reported beside the pre-declared one; 6: within-fold only |
| F-04 | `com_cci_group` is `kind: continuous, linear: none`: a raw group code is median-filled, z-scored and fitted as one linear log-odds trend; neither ordering nor spacing is documented ("grouping not documented"). | M (modelling) | `configs/meuhedet/phase2_features.yaml:177`; `phase5_v21_schema.yaml` (CCI_Group: raw group code, no Charlson computation); audit | 5.1: C-02 |
| F-05 | OLD lost hypertension, chronic kidney disease and transplant (V1 inputs removed; lineage not proven). The OLD vs NEW comparison stays internally fair (same OLD on both sides), but OLD is no longer the Phase 3 universe, so comparisons with Phase 3/4 performance are confounded. | S | `PHASE5_DESIGN.md` (lineage); `SCHEMA_DIFF_V1_V21.csv` of a preflight | V22: R-04; DISCLOSE |
| F-06 | NEW_REGISTRY timing is SAFE_ATTESTED only; the schema rates registry availability risk HIGH (backdated memberships cannot be detected by the V3 check). The composite proxy inherits this plus the unbounded medication purchase status. | S | `phase5_v21_schema.yaml: availability`, `global_caveats` | V22: R-02, R-03; 6: NO_NEW_REGISTRY sensitivity (ablation exists) |
| F-07 | The new diagnosis flags are "any record from 2022-01-01 up to the index day" with one global `Last_Dx_Date`; no per-flag recency exists, although recent dizziness, syncope or gait abnormality carry more short-horizon information than a lifetime flag. | S | `phase5_v21_schema.yaml: columns` (definitions) | V22: R-01 |
| F-08 | Label: recorded-diagnosis proxy; repeated records; no encounter setting; death as non-event; labels not bounded by personal follow-up (zero-tolerance stop). Label specificity has never been audited, even in aggregate. | S | `phase5_v21_schema.yaml: global_caveats`; `phase5.yaml: outcome_contract` | Track C: L-01 (aggregate audit, R-05); 6: death table and censoring sensitivity |
| F-09 | The LASSO/ENET lambda grid is built on the whole outer-training design (inner validation labels included). This is the standard `cv.glmnet` convention and does not touch outer-holdout labels; the inner OOF is a tuning object, not evidence. | D | `engine.py:208–209`; `models.py:23–27` | DISCLOSE (C-08); optional C-05 |
| F-10 | The tuning objective is an operating-point statistic (share flagged at the highest ≥ 70%-sensitivity threshold). It is business-aligned but noisier than a proper scoring rule; Phase 2's reviewer (A-14) recommended inner log loss for tuning. | D | `thresholds.py: objective`; `reviews/REVIEW_DECISIONS.md` A-14 | 5.1: optional sensitivity S-02 (linear families only) |
| F-11 | Criterion (3) "lower in EVERY outer fold" is an all-or-nothing rule over five folds; fine as pre-declared, brittle in expectation. | D | `analysis.py:123–124` | DISCLOSE; keep; report fold-level intervals |
| F-12 | Calibration slope / intercept are computed on pooled OOF only. | D | `metrics.py:24–37` | 5.1: C-06 (per-fold values added) |
| F-13 | No negative control has run on the real pipeline on the real data (permuted labels, future-shifted sentinel); only synthetic traps exist. | D | tests only | 5.1: C-07 |
| F-14 | "No minimum clinically meaningful effect size is assumed." A statistically supported but operationally trivial gain can be called USEFUL. | S | `phase5.yaml: decision.rule` | 6: management-declared minimal operational difference (MOD) before unblinding; 5.1 unchanged for comparability, effect reported in absolute operational units |
| F-15 | The 2026-01-01 outcomes have now been used for Phase 4 validation and Phase 5 development. | S | `PHASE4_DESIGN.md`, `PHASE5_DESIGN.md` | 6 requires a new snapshot |
| F-16 | Timing of the Phase 4 → Phase 5 snapshot reuse means the eFalls-style "frozen then validated" chain is broken for the 2026-developed models: nothing validates them yet. | S | — | 6 |

Things I checked and found **sound** (so the reviewer need not re-derive them): one row per patient with a hard stop on duplicates, so row-stratified outer folds cannot leak a patient (`engine.py:86–93`); X and y are read separately through the sealed reader; every inner fold and every outer refit fits its own `LinearDesign` (medians, categories, missing indicators, constants/duplicates, means, SDs) on training rows only; XGBoost early stopping uses an inner-training split only; the outer-holdout outcome is never passed to a unit; the inner threshold, not the holdout, sets the operating rule; the 3% capacity selection ranks within folds and allocates by largest remainder; share outputs are aggregate behind a fail-closed scan.

---

## 4. Strategy: why two tracks, in this order

**Why Phase 5.1 before Phase 6.** The models Phase 6 will freeze and validate must come from a development whose predictor eligibility did not see held-out labels. Correcting that is cheap (one overnight run), the folds and seed are unchanged, so the pre/post comparison is paired at the fold level and the correction's effect becomes a reportable number instead of an unknown.

**Why Phase 6 is a temporal validation and not another redevelopment.** Three snapshots have now been analysed (2025-01-01 development; 2026-01-01 validation, then development). No model developed on 2026 has been tested on anything unseen. Until one is, no management decision can rest on more than internal CV. A later snapshot (2026-07-01) has no outcome-window overlap with the development snapshot (2026-01-01 + 180 days = 2026-06-30), which is the condition for a clean forward validation.

**Why the estimand changes in Phase 6 and not in 5.1.** Changing the operating anchor in 5.1 would make the pre/post comparison uninterpretable. Phase 6 is a new pre-registration anyway, and its primary question should be the one management will act on (capacity-anchored).

**Why not the larger redesign now** (monthly rolling landmarks, patient-grouped multi-snapshot folds, competing-risk cumulative incidence, label adjudication — the independent round-1 design in `00_working/gpt56_raw/round1_gpt56_independent.md` argues for all of these). They are scientifically right and feasible later because the DWH keeps monthly snapshots, but they need new extracts, a new label contract and a new fold design. I place them in **Phase 7** (section 6.13) so that Phases 5.1 and 6 remain small, auditable steps on data that exist or will exist within months.

---

## 5. Phase 5.1 — specification (pre-declared; Track A)

### 5.1 Triage T-01 (read-only; minutes; no training)

From the completed run's `share\FEATURE_ELIGIBILITY.csv` (aggregate; it already carries `class`, `univariate_auroc`, `n_known_observed` per feature):

1. `n_excluded` = number of rows with class `INELIGIBLE_LEAKAGE`;
2. `auc_max` = the maximum `univariate_auroc` among screened features;
3. the list of features with `univariate_auroc ≥ 0.75` (feature names only).

Pre-declared reading:

- **Non-binding** (`n_excluded = 0` and `auc_max < 0.75`): the screen removed nothing and no feature was close enough to the 0.80 bound for fold-level resampling to have moved it across (fold-to-fold variation of a univariate AUROC with ≥ 1,000 events is of the order of ±0.02; the 0.05 margin is conservative). The Phase 5 2.2.0 estimates then stand with a disclosure note; the 5.1 re-run is still performed, for the record and for the other corrections, with the pre-declared expectation "no material change".
- **Potentially binding** (any exclusion, or `auc_max ≥ 0.75`): the 2.2.0 results are labelled PRE-CORRECTION in every later document, and no management use is made of them until the 5.1 run reports.

In both cases nothing in the completed folder is deleted or overwritten; the 5.1 run writes to a new `--out`.

### 5.2 Changes C-01 … C-10 (each small, nested, label-blind where it must be)

| # | Change | Where it lands | Rationale and design choice |
|---|---|---|---|
| C-01 | **Move the leakage screen inside the outer folds.** Preflight keeps only the label-free gates (coverage ≥ 100 known rows, constant, unreadable share ≤ 1%, timing class, provenance). Inside each unit, the univariate AUROC of every screened feature is computed on that outer fold's **training rows only**; a hit (≥ 0.80) excludes the feature from that fold's X for every family and set of that fold; the FINAL unit (no holdout) screens on all rows as today. A per-fold eligibility table (`FEATURE_ELIGIBILITY_BY_FOLD.csv`, aggregate) and a run-level WARN on any hit are reported. | `data.py: prepare` (remove the screen block at 367–391, keep registry columns); `engine.py: run_unit` (screen before `tune_and_fit`); `runner.py: run_preflight` (no label-dependent membership) | Fold-specific exclusion is the only option that uses no holdout label anywhere. The alternative "excluded in any fold → excluded everywhere" uses every label through the union and is rejected. The alternative "drop the screen entirely" loses the safety purpose; the reviewer is asked (Q1) whether the sentinel test (C-07) makes the screen redundant. |
| C-02 | **Charlson group label-blind ordinal.** If the data dictionary documents the group levels, declare `kind: ordinal, levels: [...], linear: thermometer` (exactly the MEFI treatment, `phase2_features.yaml:103`). If the levels are undocumented, declare `kind: categorical` with no levels: the existing raw-code path learns levels on training rows (count ≥ 10, reference coding). Trees unchanged (raw code, native NULL). | `configs/meuhedet/phase2_features.yaml:177` (or a Phase 5 override so Phase 2/3 protected files stay byte-identical — preferred) | Removes an unvalidated linearity assumption without using any outcome. The Phase 3 protected manifests must not change: implement as a Phase 5 feature override, hashed into the plan. |
| C-03 | **Promote the 3% capacity paired comparison** (already computed by `capacity.py`) into the main run's analysis outputs and the PRE/POST table. The verdict rule is unchanged. | `analysis.py`, `report.py` | Makes the operational read-off a first-class output of the run, not an add-on. |
| C-04 | **Within-fold equal-sensitivity companion** to criterion (2): in each outer fold, re-derive the 70% threshold on that fold's holdout for both models, sum the counts, paired bootstrap re-derives within fold (the dashboard's mechanics at a sensitivity target instead of a capacity). Reported beside the pooled version; the verdict still uses the pre-declared pooled version in 5.1, with a flag when the two disagree in sign or interval. | `analysis.py: compare`, `metrics.py` | Consistency with the within-fold principle already adopted for capacity. |
| C-05 | *(optional, low)* lambda grid per inner training fold. | `engine.py:208–209` | Standard practice already; only if the reviewer wants strict inner isolation of the grid. |
| C-06 | Per-fold calibration slope / intercept / CITL added to `CALIBRATION.csv` next to the pooled values. | `analysis.py: calibration_rows` | Pooled-OOF calibration is an approximation. |
| C-07 | **Negative controls on the real data**, quick mode, ENET only, before the main run: (i) outcomes permuted within outer folds → expected AUROC ≈ 0.5, verdict NO_ROBUST_OPERATIONAL_GAIN; (ii) a sentinel column derived from the sealed outcome, local only, never shared → must be excluded in every fold by C-01 (AUROC ≈ 1). Results are aggregate (`NEGATIVE_CONTROLS.csv`). The sentinel never enters any shared artefact or any non-control unit. | `runner.py` (a `--negative-controls` stage gated to quick mode) | Proves on the real pipeline and data that the chain returns chance under the null and catches outcome-derived information. |
| C-08 | **Disclosure block** in `SCIENTIFIC_SUMMARY.md`: the eligibility boundary table (label-free cohort-wide gates vs nested screen), the lambda-grid note, the pooled vs within-fold notes, the lost OLD predictors, the label caveats — lifted from the 0.12.2 audit. | `report.py` | The audit's findings must reach the reader of the result. |
| C-09 | **Protected PRE run and PRE/POST table.** The 2.2.0 output folder is hashed (its `share\` CSVs and the unit `COMPLETE.json` digests) and the 5.1 run refuses to write into it. The 5.1 report contains `PRE_POST_CORRECTION.csv`: verdict, criteria 1–5, Δ false-alert share at the nested rule, Δ at 70% (pooled and within-fold), Δ falls captured at 3%, per fold, PRE vs POST. | `runner.py`, `report.py` | The correction's effect is a reported number, paired by fold (same seed, same assignment). |
| C-10 | **Versioning.** falls_ml 0.13.0; `PHASE5_VERSION` 3.0.0 (the preprocessing/eligibility graph changes the results); `phase5.yaml` version 3 with the new keys; resume refuses a plan of another version. | `__init__.py`, `config.py`, `phase5.yaml` | Project convention: a scientific-graph change is a major Phase version. |

**Unchanged in 5.1** (so PRE and POST are comparable): the question, the snapshot and file hash, cohort and outcome contract, the three sets and their membership rules, the three families and budgets, the inner objective, the five-criterion rule, the seed and outer-fold assignment, the privacy rules, Phase 2/3/4 protected code.

**Optional sensitivity S-02** (linear families only, if compute allows): tune by inner log loss (A-14) instead of the operating-point objective and report whether the verdict agrees. Agreement is evidence of robustness to the tuning criterion; disagreement is reported, not resolved, in 5.1.

### 5.3 Acceptance before the real run

- All existing Phase 5 tests pass; new tests: (i) flipping any **holdout** label leaves that fold's eligibility and X columns unchanged (extends `test_unit_choices_do_not_depend_on_outer_labels`, `tests/unit/test_phase5_contract.py:485–504`, upstream to eligibility); (ii) the sentinel is excluded in every outer fold and never appears in any unit's X or in `share\`; (iii) the permuted-label control yields no USEFUL/PROMISING verdict on the planted synthetic world; (iv) the planted-gain world still returns USEFUL; (v) the PRE/POST table builds from a 2.2.0-layout synthetic folder read-only (its files byte-identical afterwards).
- A synthetic overnight rehearsal and `--estimate` on the real file; the frozen `phase5.yaml` v3 sha256 recorded before the real preflight.
- Work-PC sequence: `--preflight-only` → negative controls (quick) → `--mode overnight --resume` → `meuhedet-phase5-dashboard`. Only `share\` returns.

### 5.4 Outputs of Phase 5.1

The 2.2.0 share set plus `FEATURE_ELIGIBILITY_BY_FOLD.csv`, `NEGATIVE_CONTROLS.csv`, `PRE_POST_CORRECTION.csv`, the within-fold equal-sensitivity rows, per-fold calibration rows, the disclosure block; and the **frozen candidates for Phase 6**: the FINAL development models of ENET × {OLD, OLD_PLUS_ALL_NEW_ELIGIBLE, OLD_PLUS_NEW_SAFE} (primary) and LASSO / XGB (secondary), with their inner-selected 70% thresholds and their 3% capacity cut-offs, hashed.

---

## 6. Phase 6 — specification (Track B; to be frozen before any 2026-07-01 outcome exists)

### 6.1 Question and estimand

*Do the Phase 5.1 development models, frozen, transport to a later 2026 snapshot — and at the operational capacity, does the V21 information still convert false interventions into captured falls?*

**Primary estimand** (operational): on the validation snapshot, at capacity *c* = 3% of the eligible population (management to confirm; section 10), for ENET OLD_PLUS_ALL_NEW_ELIGIBLE versus ENET OLD: falls captured (TP), PPV and false interventions, and their paired difference. At fixed capacity ΔFP = −ΔTP, so one number carries the comparison.

**Secondary estimands**: (a) the same comparison at the frozen inner-selected 70% thresholds (realised sensitivity and false-alert share on the new snapshot — threshold transport); (b) at the re-derived 70% point within fold-free scoring (one model, one snapshot: no pooling issue); (c) discrimination and probability quality of each frozen model — AUROC, AP, Brier and scaled Brier, log loss, CITL, slope, decile calibration; (d) the capacity curve 0.5–20% (0.1% steps) and the capture-target read-off; (e) OLD_PLUS_NEW_SAFE as the stricter sensitivity analysis; (f) secondary families.

### 6.2 Design

- **Type**: temporal validation of frozen models on a later snapshot of the same population (shares patients with development → "temporal validation of a later snapshot", never "external validation"; Phase 4 wording).
- **Validation snapshot**: Index_Date **2026-07-01**, prediction at the end of the index day, V21 definitions identical to the development extract (the exact schema diff machinery stops the run on any undeclared column; a definition-hash column is requested in V22, R-10).
- **No outcome-window overlap**: development window ends 2026-06-30 < 2026-07-01.
- **Label maturity**: window ends 2026-12-28; the data freeze must also close the documentation lag (billing / discharge caveat in the schema). The lag is to be measured by the DWH (R-05); until then plan the extract for **mid-February 2027** at the earliest, and apply the project's freeze-date rule (a run refuses data whose window plus lag is not closed).
- **Frozen models**: Phase 4 mode A (persisted Phase 5.1 FINAL fits, loaded only after their `COMPLETE.json` hashes verify and re-serialise to the committed coefficients); mode B deterministic refit as fallback, with a test that B reproduces A byte-for-byte.
- **Sealing and blind scoring**: the Phase 4 pattern unchanged — sealed reader, predictors-only preflight P0–P12 (frozen inputs present and not sealed, cohort, uniqueness, leakage flag, unreadable ≤ 1%, V3 record dates, timing classes kept, semantic-change checks, new columns catalogued only), write-once blind predictions with pseudonymous keys, then the outcome contract, then metrics.
- **Outcome contract**: Phase 4 O1–O7 with Phase 5's zero tolerance for positives after the personal Followup_End_Date, label/date consistency ≤ 0.1%, ≥ 100 usable events, plus a death-within-window table by Followup_End_Reason (descriptive).
- **Subgroups** (pre-declared, descriptive, ≥ 50 events and ≥ 50 non-events): A = patients in the Phase 5.1 development cohort, B = new in the validation snapshot; age 65–74 / 75–84 / 85+; sex; prior fallers vs not; nurse-assessed vs not; MEFI groups when ≥ 20% covered.
- **Uncertainty**: 2,000-replicate patient bootstrap; paired for every model comparison; capacity re-selected in every replicate.
- **Drift**: PSI > 0.25 and missingness shift > 20 percentage points per frozen predictor are warnings (Phase 4 P10 rules), reported in `DRIFT.csv`.

### 6.3 Pre-declared decision rule (draft for the reviewer; frozen only after the review)

For ENET OLD_PLUS_ALL_NEW_ELIGIBLE (primary) and ENET OLD (fallback), on the validation snapshot:

1. **Transport check** (investigation triggers, not automatic rejection — A-10): AUROC not below the Phase 5.1 outer-OOF point estimate by more than 0.03; calibration slope 0.80–1.25 and |CITL| ≤ 0.20; capacity-curve value at 3% not below the lower 95% bound of the Phase 5.1 within-fold estimate. A trigger pauses interpretation for a written investigation (`INVESTIGATION_<trigger>.md`), as in Phases 2–3.
2. **Incremental value at capacity**: paired 95% interval of Δ falls captured at 3% entirely above 0 **and** point estimate ≥ MOD → `CONFIRMED_AT_CAPACITY`; interval above 0 but point estimate < MOD → `STATISTICAL_NOT_OPERATIONAL`; otherwise `NOT_CONFIRMED`. The MOD (minimal operational difference, in falls captured per 10,000 members at 3%, or false interventions avoided per captured fall) is declared by management in writing **before** the extract exists (section 10, D-4); if none is declared, the verdict is reported without the operational class and says so.
3. **Direction consistency**: the sign of the Phase 5.1 3% difference and the Phase 6 difference agree (reported; disagreement downgrades to `NOT_CONFIRMED` with the reason).
4. **Deployment candidate**: recommended only if (1) passes or its investigation is closed, and (2) is `CONFIRMED_AT_CAPACITY` (OLD_PLUS_ALL_NEW_ELIGIBLE) or, failing that, OLD passes (1) on its own. A candidate may receive an **intercept-only recalibration** documented as such, but never fitted on the Phase 6 outcomes before sign-off: a recalibrated model needs the silent prospective phase (6.12) or a third snapshot for its own check.

### 6.4 Precision planning (not power for a hypothesis test)

Use the Riley et al. (2021, *Statistics in Medicine*) validation sample-size criteria for calibration-in-the-large, slope, AUROC and the operating-point PPV; the required counts come from `COHORT_FACTS_2026.json` and the validation preflight. Planning illustration (placeholders, to be recomputed): with N ≈ 100,000 and c = 3%, T ≈ 3,000 flagged; a single-model PPV of 0.10 has a 95% half-width of about ±1.1 percentage points (1.96·√(0.1·0.9/3000)); the paired difference between two models that flag largely the same patients is narrower. With prevalence ≈ 2% there are ≈ 2,000 events, ample for calibration curves and the subgroups above. If the real counts fall short for a subgroup, that subgroup is reported descriptively or not at all; the primary analysis is not changed.

### 6.5 Model-improvement candidates (ranked by expected value per unit cost; none is adopted after seeing Phase 6 results)

| # | Candidate | Expected value | Cost | Enters where |
|---|---|---|---|---|
| M-1 | **Time-bounded fall-risk-increasing-drug (FRID) exposure** (psychotropics, opioids, antihypertensives, hypoglycaemics; counts, recency of change, polypharmacy) | High — FRIDs are among the strongest modifiable predictors in the geriatric literature (Seppala et al. 2018; de Vries et al. 2018, *JAMDA* — citations to verify); Phase 2 found only one medication feature can be SAFE under current timing | Data (R-02); no new code family | Phase 7 development |
| M-2 | **Recency of the new diagnosis flags** (days since last dizziness / syncope / gait abnormality / stroke record) | Medium–high for a 180-day horizon | Data (R-01) | Phase 7 |
| M-3 | **Verified registry timing** (record-creation timestamps) | Removes the HIGH availability risk; converts SAFE_ATTESTED → SAFE_VERIFIED; little new information | Data (R-03) | Phase 7; Phase 6 sensitivity without NEW_REGISTRY meanwhile |
| M-4 | **Capacity-anchored tuning objective** (maximise true positives in the top *c*% on inner OOF, log loss as guard) | Medium; aligns selection with the estimand; risk of noisier selection | Modelling; synthetic test first | Phase 7 (not 5.1, not the frozen Phase 6 models) |
| M-5 | **Charlson group ordinal** | Small, removes an assumption | Config | 5.1 (C-02) |
| M-6 | Restoration of hypertension / CKD / transplant with proven lineage | Small for prediction; large for comparability with Phase 3/4 | Data (R-04) | Phase 7 |
| M-7 | Separate prior-faller model or interaction | Unclear; only if subgroup calibration fails in Phase 6 | Modelling | Phase 7 at most |
| M-8 | Competing risk of death (discrete-time or Fine–Gray cumulative incidence) | Low for a 180-day fixed-horizon ranking if deaths within the window are a small share (to verify from the Phase 6 death table); higher if not | Modelling + label contract | Phase 7 if the death share warrants it |
| M-9 | Ensembling / stacking across families | Marginal; harms explainability | — | Not proposed |
| M-10 | Deep learning or LLM-derived features | Out of scope by project rule | — | Not proposed |

### 6.6 V22 data request (Track C; prioritised; names not in the view are "UNKNOWN — TO DISCOVER")

| # | Request | Why | Priority |
|---|---|---|---|
| R-01 | Per-flag last (and first) record date for every NEW diagnosis flag, ≤ Index_Date | recency features (M-2); per-flag timing verification instead of one global Last_Dx_Date | P1 |
| R-02 | Medication dispensing/purchase dates ≤ Index_Date by ATC group (N05A, N05BA, N05CF, N06A, N02A, N03A, C02/C03/C07/C08/C09, A10 — list to be confirmed with pharmacy), counts in 90/180/365 days, recent start/stop | time-bounded FRID exposure (M-1); currently "exposure starts at prescription, no purchase bound" | P1 |
| R-03 | Registry membership record-creation / ETL insert timestamp per registry | bounds backdating (F-06); SAFE_VERIFIED instead of SAFE_ATTESTED | P1 |
| R-04 | Hypertension, chronic renal failure and transplant flags with the V1 SQL, registry IDs and source table (lineage proof) | OLD universe complete (F-05) | P2 |
| R-05 | **Label-source audit (aggregate only)**: label-defining records by source type (ED / hospital / community), by code (888 vs each fracture prefix), repeat records within 30 days of a first record, deaths within the window, and the documentation lag distribution (record date vs load date) | label specificity and the freeze lag (F-08; 6.2) | P1 |
| R-06 | Coarse district / clinic identifiers | internal–external validation and fairness (if permitted by governance) | P2 |
| R-07 | Nursing assessment totals where they exist (Get-up-and-go total, TUG seconds) with assessment dates | the clearest functional predictors, if present | P2 |
| R-08 | Orthostatic blood pressure, recent vitals, selected labs (haemoglobin, sodium, eGFR, vitamin D) | exploratory; only after P1/P2 | P3 |
| R-09 | The 2026-07-01 snapshot under the identical V21 definitions (Phase 6); optionally the 2025-01-01 backfill of the V21 columns (robustness track, section 6.9) | Phase 6 data | P1 (2026-07-01), P3 (backfill) |
| R-10 | A definition-hash / version column in the view (Definition_Version was dropped in V21) | schema identity across snapshots | P1 |

### 6.7 Label validation L-01

Aggregate only (R-05). The question is the specificity of the recorded-fall proxy: what share of label-defining records come from ED or hospital settings, how often a single episode generates repeated records, and how many "falls" are fracture-only codes without a fall record. If a clinician at Meuhedet can adjudicate a small sample on site, only counts (PPV by source and code family, with a pre-declared minimum of 0.80 overall as the independent round-1 design proposes) return to us. A failure of that gate reframes the claim as "prediction of a recorded fall/fracture proxy"; it never silently keeps the clinical claim.

### 6.8 Negative controls and hard gates in Phase 6

Hard stops: Phase 4 P0–P9 (frozen run verified; frozen inputs present and unsealed; cohort ≥ 1,000; unique patients; leakage flag; unreadable ≤ 1%; adapter rebuild; V3 record dates; timing classes), the Phase 5 x-sealing and name patterns, the outcome contract O1–O7 with zero tolerance. Investigation triggers: P10 semantic change, drift, the transport triggers of 6.3. Negative controls: the Phase 5.1 permuted-label and sentinel controls already establish the development pipeline; Phase 6 adds a **future-shifted sentinel** in the synthetic rehearsal only (never on real data) to prove the sealed reader refuses it.

### 6.9 Optional robustness track: backfill to 2025

Phase 4 recommended backfilling the new columns to Index_Date 2025-01-01, developing on 2025 and validating on 2026-01-01. It remains attractive as a second, independent development, with two caveats that make it secondary: the 2026-01-01 outcomes have already been analysed (so that validation is "on previously analysed data" unless the 2025 development is pre-registered to ignore every Phase 5 result — feature choice must follow the label-blind rules, never the Phase 5 verdict), and registry flags backfilled to 2025 cannot be protected from backdating without R-03. Decision D-6.

### 6.10 Fairness and subgroups

Calibration (CITL, slope) and capture at 3% per subgroup, descriptively, with the ≥ 50 / ≥ 50 rule; no subgroup-specific thresholds. District/clinic only if R-06 is granted.

### 6.11 Operational translation

The Phase 6 report leads with the 3% table (patients flagged, falls captured, false interventions, PPV, lift) for OLD and OLD_PLUS_ALL_NEW_ELIGIBLE on the validation snapshot, the capture-target read-off ("to capture 50% of falls you must contact X%"), and the realised sensitivity at 3% so that management sees what a 3% programme can and cannot reach. Wording rule unchanged: "falls captured" means recorded events identified, not falls prevented.

### 6.12 Deployment staging (roadmap, not in scope of this proposal)

Retrospective (Phases 2–5.1) → temporal validation (Phase 6) → **silent prospective validation** (monthly scoring from the DWH, no exposure to clinicians, evaluated after 180 days plus lag; no model update on those outcomes before sign-off) → intervention-pathway definition with management (what happens to a flagged patient; who acts; within what time) → controlled implementation. A deployable model needs resolved availability of every input at scoring time (A-16); R-01–R-03 are therefore also deployment prerequisites.

### 6.13 Phase 7 horizon (recorded so it is not lost)

Multi-snapshot development (2026-01-01 + 2026-07-01, patient-grouped folds, embargo between outcome windows), V22 predictors (M-1 … M-3), capacity-anchored tuning (M-4), a label contract with encounter setting and death handling, competing-risk cumulative incidence as a sensitivity, and validation on the 2027-01-01 snapshot (matures August 2027 plus lag).

### 6.14 Package and versioning for Phase 6

`src/falls_ml/phase6/` modelled on `phase4/` (frozen models, sealed reading, blind scoring, outcome contract) reusing `phase5.capacity` for the 3% analysis and `phase5.metrics` for the paired bootstrap; `configs/meuhedet/phase6.yaml` frozen (sha256 in the plan) before the extract; Phase 5.1 code protected by a manifest (`PHASE5_0.13.0_PROTECTED.sha256`) verified by the Phase 6 tests. Version: falls_ml 0.14.0, Phase 6 1.0.0.

---

## 7. Timeline and milestones (from 2026-10-07)

| When | Milestone | Gate |
|---|---|---|
| Week 1 (Oct 7–14) | Independent review of this proposal; triage T-01 read from the completed run's `share\FEATURE_ELIGIBILITY.csv` (no training); decisions D-1 … D-7 | review round closed |
| Weeks 2–4 | Phase 5.1 implementation + tests + synthetic rehearsal; `phase5.yaml` v3 frozen; package 0.13.0 | acceptance 5.3 |
| Week 5 | Phase 5.1 on the work PC: preflight → negative controls → overnight → dashboard; PRE/POST report | `share\` returned |
| Weeks 1–6 (parallel) | V22 request R-01 … R-10 to BI; label-source audit L-01; management declares capacity *c* and MOD in writing | written declarations |
| Nov–Dec 2026 | Phase 6 package built and rehearsed on synthetic data; `phase6.yaml` frozen; pre-registration document committed | frozen sha256 before any 2026-07-01 outcome exists |
| ≈ Feb 2027 (after maturity + lag) | 2026-07-01 extract: `--preflight-only` → blind scoring → outcome contract → report | Phase 6 verdict |
| 2027 H1 | Phase 7 pre-registration on V22 (section 6.13) | — |

---

## 8. Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| K-1 | Triage finds the screen binding and the 5.1 verdict differs from the 2.2.0 verdict already seen by management | low–medium | PRE/POST table pre-declared; communication prepared before the run; the 2.2.0 result labelled PRE-CORRECTION |
| K-2 | The DWH cannot reproduce identical V21 definitions for 2026-07-01 | medium | R-10 definition hash; preflight-only first; the schema diff stops on any change; a changed definition makes the affected frozen predictor ineligible (Phase 4 P9/P10) rather than silently drifting |
| K-3 | Documentation lag unknown → labels immature at extraction | medium | R-05 lag distribution; freeze-date rule; extract no earlier than mid-Feb 2027 unless the lag is shown shorter |
| K-4 | Compute: 5.1 ≈ 2.2.0 overnight plus ≈ 1 h of quick negative controls; Phase 6 is scoring only | low | `--estimate` before every real run |
| K-5 | Reviewer disagrees with fold-specific exclusion (C-01) | medium | alternatives listed in C-01; the decision is recorded with reasons; either alternative is label-blind at preflight |
| K-6 | Registry backdating undetectable until R-03 | high | NO_NEW_REGISTRY ablation in 5.1; Phase 6 sensitivity; registries stay out of OLD_PLUS_NEW_SAFE's provenance class only if R-03 fails |
| K-7 | At 3% capacity the realised sensitivity is low and management expectations are anchored at 70% | high | the capacity curve and the capture-target read-off are the first page of the Phase 6 report; the MOD is declared in capacity units |
| K-8 | New outputs leak row-level information | low | every new table is aggregate; per-fold eligibility tables carry feature names and counts only; the fail-closed scan is unchanged |
| K-9 | The 2026-07-01 snapshot shares most patients with development, overstating transport | certain | reported as "temporal validation of a later snapshot"; subgroup B (new members) reported; external validation remains future work |

---

## 9. Governance

- **Pre-registration**: this document, the frozen `phase5.yaml` v3 and `phase6.yaml` (sha256 recorded in the run plans), and the protected-code manifests. Changes after freezing require a recorded amendment before the affected data are read.
- **Review loop**: the independent Astra/Codex reviewer reads this branch; every recommendation is resolved in `reviews/REVIEW_DECISIONS_PHASE6.md` (ACCEPT / PARTIAL / REJECT with reasons and verification), the format of `reviews/REVIEW_DECISIONS.md`.
- **Sequencing rule**: no training, no re-run and no Phase 6 scoring before the review round is closed and the triage is read.
- **Protection of the past**: the Phase 5 2.2.0 folder, the Phase 2/3/4 outputs and the protected source manifests are never written to; 5.1 and 6 use new output folders; all historical artefacts are hashed at start and re-verified at every stage.
- **Privacy**: real data never leave the work PC; only `share\` returns, behind the existing identifier / path / file-type / row-level scans; counts 1–9 suppressed.
- **Claims discipline**: internal CV results are "development"; Phase 6 results are "temporal validation of a later snapshot"; "captured" means recorded events identified; no causal language for importance.

---

## 10. Decision points for the PI

| # | Decision | Recommendation |
|---|---|---|
| D-1 | Adopt the two-track plan (5.1 correction, then 6 temporal validation) with Track C in parallel | Yes |
| D-2 | C-01 design: fold-specific nested exclusion (recommended) vs label-free preflight only with the sentinel test as the sole safety net | Fold-specific exclusion, pending Q1 |
| D-3 | Phase 6 validation snapshot: 2026-07-01 (recommended; no outcome-window overlap) vs an earlier snapshot with overlap (not recommended) | 2026-07-01 |
| D-4 | Management declares, in writing and before the Phase 6 extract: the capacity *c* (3% assumed) and the minimal operational difference (MOD) in falls captured per 10,000 members at *c* or false interventions avoided per captured fall | Required; without it Phase 6 reports no operational class |
| D-5 | V22 request priorities R-01 … R-10 and timing (P1 items before Phase 7 pre-registration) | As listed |
| D-6 | Request the optional 2025 backfill robustness track | Only if cheap for BI; secondary |
| D-7 | Versioning: 0.13.0 / Phase 5 3.0.0 for 5.1; 0.14.0 / Phase 6 1.0.0 | Yes |

---

## 11. Questions for the independent reviewer

1. **C-01.** Is fold-specific exclusion inside the outer folds the right remedy for the outcome-dependent screen, or would you remove the univariate screen entirely and rely on label-free gates plus the sentinel negative control? What is the strongest argument against the option you prefer?
2. **Triage margin.** Is the pre-declared reading of T-01 acceptable (non-binding if no exclusion and max univariate AUROC < 0.75 → the 2.2.0 estimates stand with disclosure; otherwise PRE-CORRECTION)? Would you set the margin differently, and on what basis?
3. **Estimand.** Do you agree that Phase 6 should be anchored at capacity (3%) rather than at 70% sensitivity, and should Phase 7's tuning objective follow (precision in the top *c*% with a log-loss guard) or stay a proper scoring rule with the capacity read off afterwards?
4. **Pooled vs within-fold.** Should the within-fold equal-sensitivity comparison (C-04) replace, rather than accompany, the pooled version in 5.1 — accepting the loss of strict comparability with the 2.2.0 rule?
5. **Temporal design.** 2026-07-01 forward validation of frozen models versus the 2025 backfill route: do you see a cleaner option given that the 2026-01-01 outcomes have been used twice?
6. **Death.** For a 180-day fixed-horizon operational ranking, is a descriptive death table (plus a sensitivity excluding patients who die within the window) sufficient, or should Phase 6 already estimate a competing-risk cumulative incidence?
7. **MOD.** How should the minimal operational difference be elicited from management so that it is defensible and cannot be chosen after the result is known?
8. **V22.** From the fall-risk literature, is anything in R-01 … R-08 mis-prioritised or missing (for example anticholinergic burden, vitamin D, orthostatic measurements), and is any item unlikely to be prediction-time available?
9. **Transport triggers.** Are the draft Phase 6 triggers (AUROC within 0.03 of the Phase 5.1 OOF estimate; slope 0.80–1.25; |CITL| ≤ 0.20; 3% capture not below the Phase 5.1 lower bound) sensible as investigation triggers, and should any be a hard stop?
10. **Residual leakage.** Do you see a leakage path this proposal does not cover — for instance `Diagnosis_Source_Absent_Ind` (metadata), billing/discharge lag in utilisation counters, the composite proxy's medication component, or the V3 attestation's blind spot for backdated registries?

---

## Appendix A — Evidence pointers

| Claim | File:lines |
|---|---|
| Eligibility screen uses cohort-wide outcomes before folds | `src/falls_ml/phase5/data.py:161–168` (`_univariate_auroc`), `:367–391` (screen loop), `:392–409` (membership); `src/falls_ml/phase5/runner.py:285` (`prepare`), `:289` (`build_sets`), `:307` (`outer_folds`) |
| Every inner fold fits its own design; outer refit fits another | `src/falls_ml/phase5/models.py:36–54` (`linear_path`), `:82–93` (`fit_linear`); `design.py:67–119` (`LinearDesign._raw`, `fit`, `transform`) |
| Lambda grid on outer-training rows incl. inner validation labels | `src/falls_ml/phase5/engine.py:208–209`; `models.py:23–27` |
| Inner objective = share flagged at the highest ≥ 70%-sensitivity threshold | `src/falls_ml/phase5/thresholds.py: objective` |
| Five-criterion rule; criterion (2) pooled OOF | `src/falls_ml/phase5/analysis.py:110–143`; `metrics.py:101–110` |
| Row-stratified outer folds, one row per patient | `src/falls_ml/phase5/engine.py:86–93`; `phase5.yaml: cohort.stop_if_duplicate_ids` |
| Capacity ranked within folds, largest-remainder allocation | `src/falls_ml/phase5/capacity.py` (docstring); `docs/phase5/DASHBOARD_SAFETY_AUDIT.md` |
| CCI continuous | `configs/meuhedet/phase2_features.yaml:177`; MEFI thermometer at `:103` |
| Registry lineage not proven; three OLD features lost | `planning/PHASE5_DESIGN.md` (registry LINEAGE); `configs/meuhedet/phase5_v21_schema.yaml: lineage` |
| Availability risk HIGH for registries and the composite | `configs/meuhedet/phase5_v21_schema.yaml: availability`, `global_caveats` |
| Phase 4 frozen-model validation pattern (modes A/B, sealing, P0–P12, O1–O7) | `planning/PHASE4_DESIGN.md` |
| Phase 4's own recommendation to backfill | `planning/PHASE4_DESIGN.md:73` |
| Phase 2 reviewer positions reused here (A-10, A-14, A-16, A-31, A-36) | `reviews/REVIEW_DECISIONS.md` |
| Monthly snapshots exist in the DWH | `docs/meuhedet/PHASE1_REPORT.md:24` |
| Completed real run exists; not read | `dist/DELIVERY_0.12.3.md` |

## Appendix B — Glossary

- **OLD**: the Phase 3 feature universe reproducible on V21 with a SAFE timing class. **OLD_PLUS_ALL_NEW_ELIGIBLE**: OLD + every new predictor with a SAFE class or UNCERTAIN_TIMING. **OLD_PLUS_NEW_SAFE**: OLD + new SAFE-class predictors with DEFENSIBLE provenance.
- **Timing classes**: SAFE_VERIFIED (row-level dates), SAFE_BOUNDED, SAFE_ATTESTED (DWH statement + V3 check), UNCERTAIN_TIMING, INELIGIBLE_TIMING / _SEMANTICS / _DATA / _LEAKAGE.
- **Nested rule**: the operating threshold chosen on inner out-of-fold predictions and applied to the outer holdout. **Descriptive 70% point**: the threshold re-derived on the holdout itself.
- **Capacity *c***: the share of the population that can be contacted; at fixed *c* both models flag the same number, so ΔFP = −ΔTP.
- **MOD**: minimal operational difference, declared by management before unblinding.
- **PRE / POST**: Phase 5 2.2.0 (pre-correction) versus Phase 5.1 (post-correction) on identical folds.
