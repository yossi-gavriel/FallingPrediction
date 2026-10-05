"""Phase 5 of the Meuhedet falls project: 2026 REDEVELOPMENT and the INCREMENTAL VALUE of the new V21 information (Index_Date 2026-01-01).

One resumable command (planning/PHASE5_DESIGN.md), after a mandatory ``--preflight-only`` on the real file:

    meuhedet-phase5 --input <2026 extract> --out <folder> --mode overnight --device auto --resume

The preflight computes the EXACT V1 -> V21 schema diff of the file against the authoritative V21 schema (configs/meuhedet/phase5_v21_schema.yaml +
the embedded VIEW definition), seals outcome / future / identifier columns, applies the timing / provenance / leakage gates and the 2026 outcome
contract. The run then compares, on identical nested folds, OLD (the Phase 3 feature universe reproducible on V21) with OLD_PLUS_ALL_NEW_ELIGIBLE
(primary business comparison) and OLD_PLUS_NEW_SAFE (stricter), for the pre-declared PRIMARY family elastic net and the secondary LASSO and XGBoost,
plus domain additions, ablations, explanation / stability and the aggregate-only management and scientific reports. The primary endpoint is the
FALSE-ALERT BURDEN AT >= 70% SENSITIVITY, with every threshold chosen on inner out-of-fold predictions only.

This is INTERNAL NESTED CROSS-VALIDATION ON THE 2026 SNAPSHOT (development data), never external validation. Phase 4 stays frozen and unused.
"""

PHASE5_VERSION = "2.1.0"
DESIGN_LABEL = "INTERNAL NESTED CROSS-VALIDATION ON 2026 SNAPSHOT"
WATERMARK = ("PHASE 5 – 2026 REDEVELOPMENT – INTERNAL NESTED CROSS-VALIDATION ON THE 2026 SNAPSHOT (DEVELOPMENT DATA, NOT EXTERNAL VALIDATION) – "
             "EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION")
WATERMARK_HE = "שלב 5 – פיתוח מחדש על נתוני 2026 – תיקוף צולב מקונן פנימי על תמונת מצב 2026 (נתוני פיתוח, לא תיקוף חיצוני) – תוצא חקרני: נפילה תוך 180 יום"
SYNTHETIC_WATERMARK = "SYNTHETIC DATA – SOFTWARE TEST ONLY – NOT A SCIENTIFIC RESULT"
