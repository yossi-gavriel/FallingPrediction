"""Phase 3 of the Meuhedet falls project (``falls_ml meuhedet-phase3``): row-level historical recovery of predictors whose source was declared
UNSAFE for EVERY row in Phase 2 because SOME rows carry a record dated on/after the index day, followed - only when the pre-declared scientific
feasibility gates say GO - by the extended modelling experiment (nested grouped CV of LASSO / elastic net / XGBoost inside TRAIN) evaluated with
partial-identification bounds for every patient whose pre-index history was overwritten in the extract.

Two acceptable outcomes: A (recovered predictors -> approved models trained and compared) or B (nothing defensible to recover -> the run stops
before any model is fitted and writes the DWH remediation requirements). See planning/PHASE3_DESIGN.md.

Phase 3 never modifies Phase 2: it imports the unchanged Phase 2 components (commit protocol, fitting, tuning, metrics) and writes only below
its own ``--out``.
"""

PHASE3_VERSION = "1.0.0"
WATERMARK = ("EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION – PHASE 3 FULL CLINICAL FEATURE SPACE, CORRECTED TIME CONTRACT (INTERNAL DEVELOPMENT ESTIMATES; "
             "TEST NOT USED; NO INDEPENDENT VALIDATION)")
WATERMARK_HE = "תוצאה חקרנית: נפילה תוך 180 יום – לא שחזור של eFalls – שלב 3 (חוזה זמן מתוקן; אומדני פיתוח פנימיים; ללא סט הבדיקה)"
