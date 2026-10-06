# Phase 5 design – 2026 redevelopment + incremental value of the new V21 information

falls_ml 0.12.2, Phase 5 2.2.0 (0.12.0 / 2.0.0 after the authoritative V21 schema was supplied; 0.12.1: registry lineage proven-or-not, zero-tolerance follow-up stop; 0.12.2: operating-capacity dashboard). Branch `phase5`. Settings:
`configs/meuhedet/phase5.yaml`; authoritative V21 schema: `configs/meuhedet/phase5_v21_view_definition.txt` (the VIEW definition with its Hebrew
business definitions, sha256-pinned) + `configs/meuhedet/phase5_v21_schema.yaml` (the column-by-column review). Code: `src/falls_ml/phase5/`.
Runbook: `docs/meuhedet/PHASE5_WORK_PC_RUNBOOK.md`.

## Question and endpoint (final)

At approximately the same >= 70% fall sensitivity, does adding the new V21 information reduce the number and percentage of false alerts?
PRIMARY comparison (pre-declared): ENET OLD vs ENET OLD_PLUS_ALL_NEW_ELIGIBLE; LASSO and XGBoost secondary; OLD_PLUS_NEW_SAFE the stricter
sensitivity analysis. Design label: INTERNAL NESTED CROSS-VALIDATION ON 2026 SNAPSHOT. Phase 4 is frozen and unused (hash-verified).

## Exact V1 -> V21 schema diff (`phase5/schema.py`)

Computed at every preflight from three column lists: the V1 contract (221), the authoritative V21 header (224) and the header of the file read.
The reviewed schema supplies only semantics: for each shared column the V1 -> V21 equivalence evidence (DOCUMENTED_MATCH / CLARIFIED_NO_CONFLICT
/ CHANGED, with notes), for each new column its role, definition, domain, kind, timing basis (record_date / attested / uncertain) and provenance
(DEFENSIBLE / UNVALIDATED_CODES / EXPERIMENTAL_COMPOSITE). Classes (exactly one per column): OLD_UNCHANGED,
OLD_CHANGED_DEFINITION, RENAMED_OR_REPLACED, NEW_CANDIDATE_PREDICTOR, METADATA_OR_ADMIN, OUTCOME_OR_FUTURE_FORBIDDEN, IDENTIFIER,
REQUIRES_SEMANTIC_REVIEW (any column the schema does not define -> STOP before training). Shared-column classes follow the V1 role (IDENTIFIER;
LABEL / FORBIDDEN_LEAKAGE / post-index -> forbidden; QA / cohort -> metadata; predictor roles -> OLD_*).

Findings on the authoritative header: 201 shared, 20 removed, 23 new. Removed V1 predictors: Registry_Blood_Pressure_Ind (eFalls hypertension),
Registry_Chronic_Renal_Failure_Ind (CKD), Registry_Transplant_Ind, Siudi_Status. Registry LINEAGE (0.12.1): a rename is never inferred from a
column position or a name replacement. The schema's `lineage` section compares, for each removed V1 registry flag and the V21 column in its place,
the V1 / V21 SQL expression, registry IDs, source table and logic. The V1 contract has no SQL, registry ID or source table (only a DDL / S2T label;
Q-M-04 never answered), while V21 defines Registry_Corona_Ind (116 / 118, "not hypertension"), Registry_Dialysis_Ind (101 / 1, "not all kidney
failure") and Registry_Immunosuppressant_Ind (130 / 131, "not a transplant flag"): lineage NOT proven -> OLD_REMOVED_NEW_ADDED for all three (the
V1 features leave OLD; the V21 fields are genuinely new NEW_CANDIDATE_PREDICTORs, domain NEW_REGISTRY, DEFENSIBLE). The loader refuses a
same-lineage class (TRUE_RENAME_SAME_SEMANTICS / CORRECTED_LABEL_SAME_SOURCE / MATERIAL_DEFINITION_CHANGE) without lineage_proven and the V1 SQL
evidence, and refuses `replaces` without a proven lineage; a proven same-source pair stops for review (OLD inputs are never aliased).
Changed definitions: Last_Hosp_Length (max, not last stay), Fall_Self_Report_Value (binary,
not unconfirmed codes); neither is a Phase 3 input. Prior_Fall_Missing_Ind (the removed VALIDATION-only input of eFalls 'falls') is bridged as 0
(V21 COUNT gives 0 without a record), so 'falls' is reproduced exactly. Definition_Version no longer exists: the extract is identified by its
header. MEFI is not new.

## Data, sealing, eligibility (`phase5/data.py`)

* X and the outcome are read separately (Phase 4 sealed reader; a sealed column request raises). Sealed: the brief's outcome / follow-up columns,
  V1 LABEL / FORBIDDEN_LEAKAGE / post-index columns, every OUTCOME_OR_FUTURE_FORBIDDEN / IDENTIFIER column of the diff, unknown future-looking names.
* Outcome: Phase 4 reader + contract O1-O7 on the eligible rows (strictly after the index day, within 180 days, positives after the personal
  Followup_End_Date: ZERO tolerance (any one -> STOP - REVIEW REQUIRED, aggregate count / % only; the config refuses any other limit), label /
  date consistency, >= 100 usable events; episode audit descriptive); labels never rewritten, patients never excluded by O4.
* Cohort: Index_Date 2026-01-01, Is_Eligible_Cohort = 1; duplicate / NULL Customer_Full_ID -> HARD STOP; Leakage_Check_Ind non-zero -> STOP.
* OLD = the Phase 3 universe rebuilt with the unchanged Phase 1-3 code; features whose V1 VALUE input was removed are INELIGIBLE_DATA (hypertension,
  chronic_kidney_disease, com_registry_transplant); OLD_CHANGED_DEFINITION inputs would be kept with the V21 definition in every set.
* NEW = every NEW_CANDIDATE_PREDICTOR / RENAMED_OR_REPLACED (proven lineage only; none in V21) column: diagnosis flags verified row by row against Last_Dx_Date (post-index source
  records -> UNKNOWN -> no-record state; Phase 3 gates G2 / G3), registry flags SAFE_ATTESTED (V3), Deficit_Count_Proxy UNCERTAIN_TIMING
  (composite incl. medication exposure whose purchase status is not bounded by Index_Date). Coverage (>= 100 known rows) and the single-feature
  AUROC >= 0.80 leakage safety screen (exclusion only).
* Sets: OLD (SAFE classes); OLD_PLUS_ALL_NEW_ELIGIBLE (+ every new SAFE or UNCERTAIN_TIMING predictor); OLD_PLUS_NEW_SAFE (+ new SAFE predictors
  with DEFENSIBLE provenance: no raw sub-codes, no composite). Every inclusion / exclusion carries its reason
  (FEATURE_ELIGIBILITY.csv, NEW_FEATURE_CATALOGUE.csv, ALL_V21_COLUMN_CLASSIFICATION.csv x_use). `x_guard` re-checks the sets (HARD STOP X_LEAKAGE).
* Domains: NEW_DIAGNOSIS, NEW_VISION_HEARING, NEW_REGISTRY, NEW_FRAILTY_OR_RISK_PROXY (from the schema). Ablations (ENET):
  NO_<each domain>, NO_FALL_RECENCY, NO_TIMING_UNCERTAIN.

## Nested CV (`engine.py`, `models.py`, `design.py`)

Unchanged from 1.0.0: one stratified outer assignment (5 folds) for every family and set; inner 5-fold CV; a unit receives y for outer-training rows
only; LASSO / ENET with the Phase 1-3 solver, XGBoost Optuna (deterministic resume); objective = highest threshold with sensitivity >= 70% on
inner OOF, then minimise flagged share (ties: false-alert share, AP, Brier, simpler). New: raw codes without declared levels (registry sub-codes)
learn their one-hot levels on the training rows only. Budgets (overnight): XGB 100 trials, LASSO 60 / ENET 40 penalties x 5 l1 ratios,
stability 100 / 30, bootstrap 2000.

## Decision rule (pre-declared, `phase5.yaml: decision`)

For each family, OLD_PLUS_ALL_NEW_ELIGIBLE vs OLD: (1) lower false-alert share nested AND descriptive; (2) paired CIs entirely below 0 at the
nested rule AND at exactly 70% sensitivity (threshold re-derived in each bootstrap replicate: the synthetic null showed that noise features can
move the inner-selected threshold enough for a narrow nested interval below 0 - the equal-sensitivity interval does not confirm it); (3) lower in
EVERY outer fold; (4) OLD_PLUS_NEW_SAFE also lower (the gain does not rest on timing / provenance-questionable predictors); (5) calibration not
materially worse. USEFUL = all five; PROMISING = (1) and ((2) or (3)); otherwise NO ROBUST GAIN. The answer is the ENET verdict (USEFUL -> YES,
PROMISING -> UNCERTAIN, otherwise NO); LASSO / XGBoost secondary; the best family is exploratory only. No minimum effect size is invented.

Synthetic lesson: the composite proxy can give a small, "significant" gain that is not new information (a sum of OLD flags). Criterion (4), the
NEW_FRAILTY_OR_RISK_PROXY domain test and the NO_<domain> / NO_TIMING_UNCERTAIN ablations make that visible.

## Operations, outputs, privacy

The first real-data action must be `--preflight-only` (fits nothing; final line SAFE TO MODEL or STOP - REVIEW REQUIRED); a modelling call on a
folder without a SAFE preflight plan stops with PREFLIGHT_REQUIRED (the synthetic smoke may do both). Resumable units (COMPLETE.json last),
RUN_STATUS.json heartbeat / ETA, RUN_TIMINGS.csv, OVERNIGHT_PROGRESS.log, interim report after PRIMARY, Ctrl+C -> exit 130, one retry per unit.
`share/` behind the fail-closed scan (identifiers, row keys, local paths, file types, row-level tables; counts 1-9 suppressed; 0.5% grid).

## Operating-capacity dashboard (`capacity.py`, `dashboard.py`, `dashboard_html.py`) - 0.12.2

Question: "if we can intervene on X% of the population, how many falls do we capture?" (3% = the operational target). Analysis / reporting
only: the committed outer-OOF predictions of the PRIMARY units are read (COMPLETE.json hashes checked; holdout rows must equal the frozen
fold); nothing is fitted, tuned or re-validated (an e2e test replaces every fitting entry point with a failing stub). PRIMARY curve = outer-fold
capacity: T = round(c x N) (half up, integer arithmetic), allocated across the outer folds by largest remainder (ties -> lower fold), top-k_f by
that fold's own model, counts summed - probability scales of different outer models are never mixed; POOLED OOF (global ranking) is a
separately labelled descriptive view. Grid 0.5-20% in 0.1% steps (coarsened so neighbouring points differ by >= 10 patients; 3.0% always
exact) + 20.5-100% in 0.5% steps for the secondary "target fall capture" read-off. At 3%: a paired patient bootstrap (2000 replicates,
identical multinomial weights for every model; the capacity re-allocated from the resampled fold sizes and the top-k re-selected within each
fold in every replicate) for delta falls captured / sensitivity / PPV / false interventions (at fixed capacity delta FP = -delta TP).
"What drives the model": ENET / LASSO |median standardised coefficient| of the stability refits, XGB mean |SHAP| (permutation importance as
fallback), OLD vs NEW, SAFE vs ALL_NEW_ONLY, stability - descriptive. Privacy: a capacity point is shared only if every model's four cells are
0 or >= 10; a row with a small cell suppresses all its counts, rates and differences; the HTML embeds only per-point counts (no patient, score,
date or key), uses no external resource and passes the same identifier / path scan. The management summary opens with the 3% question; the
~70% sensitivity analysis follows unchanged as the secondary analysis. `meuhedet-phase5-dashboard --out <completed folder>` adds all this to
a folder completed by 0.12.1 (earlier share files byte-identical; the previous share kept in work/dashboard/); the full report of a new run
builds it too.
