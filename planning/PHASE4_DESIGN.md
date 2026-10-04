# Phase 4 – temporal validation 2026 of the frozen Phase 3 models (design, pre-declared)

Status: frozen before any 2026 value was examined (falls_ml 0.10.0, Phase 4 1.0.0; settings `configs/meuhedet/phase4.yaml`).

## 1. Question and scope

How do the models frozen in Phase 3 (developed on the 2025-01-01 snapshot) perform on the 2026-01-01 snapshot, with nothing re-fitted, re-tuned,
re-calibrated, re-thresholded or re-selected? The 2026 cohort shares patients with 2025: this is a **temporal validation of a later snapshot**,
not an independent external validation.

No 2026 outcome is used for training, tuning, feature selection, thresholds, calibration, model choice, imputation rules or schema adaptation.

## 2. Frozen models

- Primary `LASSO:P3_BASE`; pre-declared secondary `LASSO:P3_VERIFIED_ALL`, `LASSO:P3_ALL_RECOVERED` (all existed before the 2026 data).
- **Mode A** (preferred): the persisted Phase 3 final fit `stages/S07_lasso/items/LASSO__<set>__final/model.pkl`, loaded only after its Phase 3
  commit record (sha256 of every file) is verified; the loaded object must re-serialise to the byte-identical committed `coefficients.csv`.
- **Mode B** (only if A is absent): a deterministic re-fit with the unchanged Phase 3 code (`linear_item(..., "final")`, same item and inner-fold
  seeds, the frozen `FEATURE_SETS.json`) on a private copy of the committed 2025 products; the Phase 3 settings / catalogue / mapping / contract in
  the package must have the sha256 recorded in the Phase 3 plan. A test proves B reproduces A byte-for-byte.
- The mode of every model is recorded (FROZEN_MODEL_MANIFEST.json, RUN_MANIFEST.json, summaries).
- The Phase 3 code is protected: `configs/meuhedet/PHASE3_0.9.0_PROTECTED.sha256` (144 files) is verified by the test suite.

## 3. Sealing

Sealed columns are never requested from the file (`usecols`) before `meuhedet-phase4-evaluate` verified the frozen hashes: the brief's 13 outcome
columns, every contract column with role LABEL / FORBIDDEN_LEAKAGE or timing post_index (follow-up end and reason, censoring, death, Label_End,
Fall_On_Index_Date_Ind, ...), and every new 2026 column whose name looks like outcome / future information. Every read is logged by name.

## 4. Preflight (predictors only) – pre-declared stop rules

P0 the Phase 3 run (frozen production configuration, files verified; models available in mode A or B) · P1 every input of a frozen predictor present
and not sealed (STOP) · P2 no frozen input is label / forbidden / post-index / identifier / a non-contract column (STOP) · P3 index date and
eligible cohort (STOP below 1000 rows) · P4 Customer_Full_ID / Snapshot_Key unique among eligible rows (STOP) · P5 Definition_Version (STOP if it
differs from `--expected-definition-version`), Leakage_Check_Ind (STOP if non-zero; not a complete guarantee) · P6 unreadable / not-allowed cells
of a frozen input (> 1% STOP; otherwise those cells are UNKNOWN and bounded, never imputed) · P7 the unchanged adapter rebuilds the baseline
(undeclared codes / forbidden NULLs STOP) · P8 predictor record dates after 2026-01-01 (Phase 3 check V3; the attestation holds only without
them) · P9 every frozen predictor keeps an eligible Phase 3 timing class on 2026 (primary STOP; a secondary model is excluded, recorded) ·
P10 semantic change: absent in 2026 but observed in >= 1% of 2025 TRAIN, binary outside {0,1}, constant in 2026 only, a median / p95 ratio beyond
x10, a new category level (primary STOP, secondary excluded); missingness change > 20 pp or PSI > 0.25 are warnings · P11 known V21 concerns
(medication purchase timing, external-care billing lag, episode duplication, follow-up) listed with the affected frozen predictors · P12 new
columns catalogued only.

## 5. Blind scoring and freezing

Order: the preflight is recomputed and must reproduce its stored fingerprint → models frozen (copied, hashed) → FROZEN_MODEL_MANIFEST.json (models,
feature lists, frozen absolute 2025 cut-offs) and PHASE3_INTERNAL_ESTIMATE.json (Phase 3 outer out-of-fold estimate on 2025 TRAIN, same metric
code) → 2026 predictions (UNKNOWN cells → exact intervals over 2025 pre-index states, Phase 3 bounds) → overlap (member ids in memory only) →
BLIND_PREDICTIONS.parquet (pseudonymous keys: sha256 of the 2026 file hash + member id), BLIND_PREDICTIONS_MANIFEST.json (sha256 of predictions,
model manifest, feature list, input schema, Phase 3 digest; time; code) → BLIND_SCORE_COMPLETE.txt. Everything read-only; scoring is write-once.

## 6. Outcome contract 2026 (before any metric)

O1 no positive with an event on / before 2026-01-01 · O2 Label_End_180D = Index_Date + 180 · O3 no positive after Index_Date + 180 · O4 positives
after a real Followup_End_Date <= 0.5% of positives · O5 label / event-date inconsistencies <= 0.1% · O6 censored patients reported (by reason),
usable events >= 100 · O7 episode audit (early events after a recent / index-day fall; descriptive). O1-O6 failures stop before any performance.

## 7. Metrics (identical usable 2026 patients for every model)

AUROC, average precision, Brier, Brier skill vs the development prevalence (and vs the observed one), log loss, calibration intercept / slope, CITL,
O:E, decile calibration; capture / PPV / lift / patients selected / falls captured at 1%, 5%, 10% (top ceil(q·n) of the evaluated patients – the
Phase 3 rule) and at the frozen absolute 2025 cut-offs. 95% percentile intervals from 2000 patient bootstrap replicates (one row per patient).
Patients with UNKNOWN cells: ADVERSE bound headline, FAVOURABLE reported (identical when no UNKNOWN cell exists).

2025 vs 2026: Phase 3 internal estimate vs 2026 for AUROC, AP, Brier, Capture/PPV/Lift@10%, prevalence, N, events; absolute change; a change is
called CI-SUPPORTED only when the two 95% intervals do not overlap (conservative because patients overlap; no formal test claimed).

Subgroups (pre-declared): A = 2026 patients in the 2025 Phase 3 population (FULL_LABELED), B = new in 2026; same model and the same flags as the
full population; reported only with >= 50 events and >= 50 non-events. Secondary models: paired patient-bootstrap differences vs the primary.

## 8. New 2026 predictors

Catalogued only (NEW_2026_FEATURES_CATALOGUE.csv). Recommended later design: backfill them for Index_Date 2025-01-01, develop on 2025, validate once
on a still-unseen later snapshot.

## 9. Share and privacy

Aggregate only (counts 1-9 and every rate that would give them back suppressed); fail-closed scan for member ids, snapshot keys, row keys, local
paths, row-level tables and model / data file types.
