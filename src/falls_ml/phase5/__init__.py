"""Phase 5 of the Meuhedet falls project: 2026 REDEVELOPMENT and the INCREMENTAL VALUE of the new V21 information (Index_Date 2026-01-01).

Phase 5 3.0.0 = **Phase 5.1, the REPAIR-ONLY methodological correction (Experiment 1 of docs/phase6/FINAL_CONSENSUS.md)** of the Phase 5 2.2.0 run:
the same question, snapshot, cohort, labels, outer folds, seeds, feature-set rules, ENET family, tuning objective, candidate space and budgets, with
the methodological defects of 2.2.0 corrected and nothing else changed (docs/phase6/EXPERIMENT1_IMPLEMENTATION_CONTRACT.md):

    R-1  the cohort-level outcome-dependent single-feature AUROC exclusion is REMOVED: predictor membership is decided by label-free rules only
         (sealing, schema classes, timing classes, provenance, the V3 attestation, label-free coverage / constant / unreadable QA gates)
    R-2  a per-fold, label-free coverage gate inside every unit (outer-training rows only)
    R-3  the LASSO / ENET lambda grid is anchored INSIDE each inner training fold (dimensionless ratio grid; the outer refit re-anchors on the
         outer-training rows); the candidate space itself is unchanged
    R-4  CCI_Group is QUARANTINED unless an authoritative DWH dictionary is registered (feature_overrides); no ordinal spacing is inferred
    R-5  an explicit ``other`` level for nominal codes with learned levels (linear path)
    R-6  a training-fold forensic univariate AUROC per feature - LOCAL and REPORT-ONLY, it never changes membership
    R-10 the completed 2.2.0 run (PRE) is verified before anything is fitted (input sha256, cohort / labels, fold hashes, units, the Top-3%
         capacity table reproduced exactly) and its outer folds are ADOPTED, never regenerated
    R-12 negative controls on the real data (frozen-fold label permutations) before PRIMARY; a failure is a hard stop
    the headline is the ABSOLUTE Top-3% result (selected, captured, Recall@Top3, PPV@Top3, false interventions, PRE vs POST delta with a
    paired 95% CI); the 2.2.0 five-criterion 70%-sensitivity verdict is kept as a HISTORICAL AUDIT output only.

One resumable command, after a mandatory ``--preflight-only --pre-run <completed 2.2.0 folder>`` and ``--negative-controls`` on the real file:

    meuhedet-phase5 --input <2026 extract> --out <NEW folder> --pre-run <PRE folder> --mode overnight --resume

This is INTERNAL NESTED CROSS-VALIDATION ON THE 2026 SNAPSHOT (development data), never external validation. Phase 4 stays frozen and unused.
Policy B (log-loss shortlist, Top-3% selection), new l1 ratios, new families, the additive basis and every other improvement belong to Experiment 2
and are NOT reachable from this version.
"""

PHASE5_VERSION = "3.0.0"
PHASE5_MAJOR = PHASE5_VERSION.split(".")[0]
PRE_MAJOR = "2"                                     # the Phase 5 generation whose completed run is verified and compared as PRE
DESIGN_LABEL = "INTERNAL NESTED CROSS-VALIDATION ON 2026 SNAPSHOT"
EXPERIMENT_LABEL = "PHASE 5.1 REPAIR-ONLY CORRECTION (EXPERIMENT 1): PRE 2.2.0 vs POST 3.0.0 ON IDENTICAL PATIENTS AND FOLDS"
WATERMARK = ("PHASE 5.1 – REPAIR-ONLY METHODOLOGICAL CORRECTION OF THE 2026 REDEVELOPMENT – INTERNAL NESTED CROSS-VALIDATION ON THE 2026 SNAPSHOT "
             "(DEVELOPMENT DATA, NOT EXTERNAL VALIDATION) – EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION")
WATERMARK_HE = ("שלב 5.1 – תיקון מתודולוגי בלבד של הפיתוח מחדש על נתוני 2026 – תיקוף צולב מקונן פנימי על תמונת מצב 2026 (נתוני פיתוח, לא תיקוף חיצוני) – "
                "תוצא חקרני: נפילה תוך 180 יום")
SYNTHETIC_WATERMARK = "SYNTHETIC DATA – SOFTWARE TEST ONLY – NOT A SCIENTIFIC RESULT"
