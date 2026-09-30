<!-- produced by tools/phase2_rehearsal.py on the final 0.8.0 code, macOS arm64, Python 3.13, 6,000-row synthetic extract; the run is repeatable with: python tools/phase2_rehearsal.py --work <empty folder> -->

# Phase 2 crash / resume rehearsal (SYNTHETIC DATA – software test only)

**Overall: PASSED**

| # | Kill point (stage, item, point) | Exit | Committed files after | Earlier committed files changed |
|---|---|---|---|---|
| 1 | S01_cohort|cohort|item_start | 137 | 4 | 0 |
| 2 | S06_lasso|LASSO__BASELINE_15__outer1|before_rename | 137 | 21 | 0 |
| 3 | S06_lasso|LASSO__ALL_REVIEWED_SAFE__final|before_record | 137 | 34 | 0 |
| 4 | S08_xgb|XGB1_outer1__t02|item_start | 137 | 249 | 0 |
| 5 | S10_stability|STAB__ALL_REVIEWED_SAFE__r01|before_record | 137 | 298 | 0 |
| 6 | S13_validation|validation|item_start | 137 | 388 | 0 |
| 7 | S16_share|share|before_rename | 137 | 405 | 0 |
| 8 | none (final attempt) | 0 | 405 | 0 |

- Resumed after every kill: **True**
- Completed artifacts never changed: **True**
- Final tables byte-identical to the uninterrupted run (20 files): **True**
- Reference folder unchanged: **True**
- VALIDATION opened for one frozen selection only: **True** (1 registry record(s))
- Share privacy scan passed: **True**
- Investigation stops met and accepted with a recorded reason (planted synthetic signal): IMPLAUSIBLE_GAIN_OOF, IMPLAUSIBLE_CALIBRATION_VALIDATION
- Uninterrupted run: 1999.8 s; whole rehearsal: 3883.2 s
