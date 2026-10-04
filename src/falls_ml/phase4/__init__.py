"""Phase 4 of the Meuhedet falls project: TEMPORAL VALIDATION of the frozen Phase 3 (Index_Date 2025-01-01) models on the 2026 snapshot
(Index_Date 2026-01-01).

Four separate commands, each with one job (planning/PHASE4_DESIGN.md):

    meuhedet-phase4-preflight   read-only predictor / schema / timing audit of the 2026 extract (outcome columns are never loaded)
    meuhedet-phase4-score       blind scoring with the frozen models; predictions, models, feature list and input schema are hashed and frozen
    meuhedet-phase4-evaluate    verifies every frozen hash FIRST, only then opens the 2026 outcomes, checks the 2026 outcome contract and reports
    meuhedet-phase4-status      read-only progress

No 2026 outcome is used for any development decision. The 2026 cohort shares patients with 2025: this is a TEMPORAL validation of a future
snapshot, not an independent external population. Phase 4 never writes into the Phase 2 or Phase 3 folders.
"""

PHASE4_VERSION = "1.0.0"
WATERMARK = ("TEMPORAL VALIDATION 2026 OF THE FROZEN PHASE 3 (2025) MODELS – EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION – "
             "OVERLAPPING PATIENTS, NOT AN INDEPENDENT EXTERNAL POPULATION")
WATERMARK_HE = "תיקוף זמני 2026 של מודלי שלב 3 (2025) המוקפאים – תוצא חקרני: נפילה תוך 180 יום – לא שחזור eFalls – חלק מהמטופלים משותפים לשתי השנים"
