"""Phase 2 of the Meuhedet falls project (``falls_ml meuhedet-phase2``): column registry, engineered-feature catalogue, TRAIN-only screening,
frozen feature sets, nested grouped CV of LASSO / elastic net / XGBoost inside TRAIN, a pre-declared advancement rule frozen before the
VALIDATION partition is opened once, stability / ablation / explainability, feature consensus and a non-identifying share package.

Crash-safe by construction: every unit of work is an item committed with a marker written last; a run resumes from the first incomplete
item and never rewrites a completed artifact. See planning/EXPERIMENT_PLAN.md and planning/ARCHITECTURE_PHASE2.md.
"""

WATERMARK = "EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION – PHASE 2 DISCOVERY (TRAIN/VALIDATION ONLY; TEST NOT USED)"
WATERMARK_HE = "תוצאה חקרנית: נפילה תוך 180 יום – לא שחזור של eFalls – שלב 2 (גילוי; ללא סט הבדיקה)"
