GO FOR REAL-DATA PHASE 2 RUN

# Phase 2 final run readiness (falls_ml 0.8.1)

**Package:** `falls_ml_patch_0.8.1_phase2_final_mailsafe.zip`, cumulative from the 0.4.0 handoff. Its sha256 is given with the handoff message and in `SHA256SUMS_0.8.1.txt`.

**Hashes**
- Frozen scientific configuration: `configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json`, sha256 `a1112a9a9d2bc74f4503be4577492a8c1c51bc802a69e61554583992d3fcce13`
- Code (falls_ml source tree, OS-independent): sha256 `63c7ae2ae9fd66b2e8ab9bf9a6963b0be128370e02cb495fd19d08ae0313525b`

**Evidence basis.** Everything below was verified on SYNTHETIC data only. No real-data result exists until the user returns `<out>\share`.

## Checklist

| Item | Status | Evidence |
|---|---|---|
| Patch verified | ✅ | Clean-room restore of the mail-safe zip onto the untouched 0.4.0 package: every patched file byte-identical to the source project, no other file changed, `falls_ml` 0.8.1 imports (see the handoff message) |
| Astra final parameter review complete | ✅ | `reviews/ASTRA_FINAL_PARAMETER_REVIEW.md` (one call, 47,171 tokens, the actual frozen values) |
| All MUST FIX items closed | ✅ | F-01 … F-05, `reviews/ASTRA_FINAL_RESOLUTION.md` (each with change + verification); all 9 SHOULD items accepted (7 fully, 2 partly, with reasons) |
| Final config frozen | ✅ | sha256 `a1112a9a9d2bc74f4503be4577492a8c1c51bc802a69e61554583992d3fcce13`. The runner and the preflight refuse any other effective configuration (`FROZEN_CONFIG_MISMATCH`, unit-tested) |
| Preflight works | ✅ | Synthetic, frozen configuration, no override: last line `SAFE TO START FULL RUN`; `--out` not created; no model fitted (e2e `test_preflight_fits_nothing_and_writes_nothing`) |
| Python 3.11 environment supported | ✅ | Slow e2e on CPython 3.11: 9 / 9 passed (CPython 3.11.15, locked package versions). Fast suite on 3.11: 2,058 passed, 0 failed (3.13: 2,058 passed, 0 failed; slow e2e on 3.13: 9 / 9) |
| Resume tested | ✅ | Kills (os._exit, like a power loss) inside an Optuna trial, inside the one-shot validation and before the share rename, each resumed: final tables byte-identical to an uninterrupted run; no committed file changed; the frozen configuration unchanged (e2e 3.11 + 3.13). The full 7-kill rehearsal (`planning/REHEARSAL_REPORT.md`) is not repeated: the commit / resume code (`state.py`, `durable.py`, `xgb_tuning.py`) is unchanged |
| Optuna persistence tested | ✅ | The killed trial is moved to `_incomplete`; the earlier trial record is unchanged (not rerun); each (study, trial) appears once in the ledger; SQLite is rebuilt from committed trials |
| Privacy / share pipeline tested | ✅ | Share built only after the identifier / pepper / path scan; small cells (< 10) suppressed in tables, summaries and figures; no `research_id` column in any shared CSV |
| Historical artifacts protected | ✅ | `--reference` + three `--protect` folders are hashed at S00 and re-verified at every model stage (PROTECTED_CHANGED = hard stop) |
| TEST protected | ✅ | Identified in the split audit (count + sha256 only) and dropped in memory; only `train` / `validation` rows reach any stage (e2e); `test_outcomes_read: false` |
| Windows commands ready | ✅ | Below, and in `PATCH_README.txt` |

## Scientific integrity checks (§21)

| Check | How it is guaranteed | Evidence |
|---|---|---|
| No target leakage | Labels, identifiers, follow-up and forbidden-leakage columns are INELIGIBLE_FORBIDDEN by contract role; the outcome is dropped from the raw columns after mapping | column registry; category invariant |
| No future-data leakage | Unchanged D-00 rules, `source_event_date < Index_Date`; UNSAFE sources are excluded; UNRESOLVED ones never enter a SAFE set | `INELIGIBLE_FEATURE_IN_SET` hard stop; e2e planted index-day records → INELIGIBLE_UNSAFE, never fitted |
| No patient overlap | PATIENT_OVERLAP hard stop in S01; one row per patient | cohort facts |
| No TEST-based selection | TEST dropped before any computation | e2e (`partition` ∈ {train, validation}) |
| VALIDATION opened once | `VALIDATION_OPENED.json` write-once; re-read only for the same selection and configuration (else VALIDATION_REOPEN_REFUSED); the registry holds one selection sha | unit + e2e |
| SAFE model contains SAFE features only | `check_category_invariants` before any fit | unit + e2e |
| UNRESOLVED sources stay labelled | EXPLORATORY_UNRESOLVED_SENSITIVITY category; separate table `EXPLORATORY_UNRESOLVED_FEATURES.csv` | e2e |
| Missingness treatment recorded | FEATURE_SETS.json `design` (form indicators, `na__` groups, fills); item indicator for any mismatch (F-04) | frozen config `encoding_policy` |
| Feature engineering reproducible | Row-wise, parameter-free catalogue (sha256 in the plan); fitted transforms inside folds | plan hash |
| XGBoost ingests no metadata / IDs | The tree matrix holds only the set's catalogue features + baseline predictors | e2e check of every XGBoost fit's columns |
| Categorical encodings appropriate | Baseline: eFalls reference coding + FP; ordinals: thermometer; binaries 0/1; counts log1p; fall recency bands | frozen config |
| Linear scaling appropriate | z-score of every design column on the fitting rows inside the solver | LASSO / EN implementation |
| XGBoost preprocessing is its own | Raw values, native NULL routing, no log / thermometer / imputation | frozen config |
| Class weighting does not distort calibration | None (`scale_pos_weight` 1, enforced by the config loader) | config loader test |
| Every probability output is evaluated for calibration | CITL, slope, intercept, O:E, Brier, scaled Brier for every configuration; intervals + smooth curves for the shortlist and the waterfall | CALIBRATION_SUMMARY / BOOTSTRAP / SMOOTH |
| Identical populations where claimed | Every waterfall step and every paired comparison uses the same VALIDATION rows | e2e (`val_n` unique) |
| Matched capacity | ceil(q × n) highest risks per model, identical rule; the bootstrap recomputes top-N per replicate | unit (bootstrap = top_mask) |
| Seeds and hashes recorded | Master seed, item seeds, scope fold seeds, plan / config / catalogue / input / code / feature-set sha256 | PHASE2_PLAN.json, RUN_MANIFEST.json |

## Real-run facts known only after the preflight

- The exact SAFE / UNSAFE / UNRESOLVED counts of the real extract.
- The SAFE_BASE composition. The design allows at most 4: age, sex, prior falls, mobility problems.
- The number of SAFE new features.
- The label / death audit.

Design-time limits are in `planning/PROVENANCE_ACCOUNTING.md`: of the 93 features, 69 can be SAFE and 24 never can. MEDICATION has only 1 feature that can be SAFE.

## Limitations that no run can remove

- D00_CLEAN selected cohort.
- VALIDATION reused in Phase 1.
- Exploratory 180-day recorded-event label.
- Point-estimate classifications only.
- A new temporal holdout is needed before any claim.

## Windows commands (CMD, work PC)

```bat
cd /d "C:\Users\Yosef.g9\Downloads\falls_ml_handoff_0.4.0_mailsafe\falls_ml_handoff"
set PYTHONUTF8=1
.venv\Scripts\python.exe -m falls_ml meuhedet-phase2 --preflight --input "C:\Users\Yosef.g9\Downloads\60k_falling_db.csv" --reference "C:\Users\Yosef.g9\Downloads\60k_falling_db_explore" --protect "C:\Users\Yosef.g9\Downloads\60k_falling_db_eda" --protect "C:\Users\Yosef.g9\Downloads\60k_falling_db_reports" --protect "C:\Users\Yosef.g9\Downloads\60k_falling_db_d00" --out "C:\Users\Yosef.g9\Downloads\60k_falling_db_phase2"
```

**Full run:** the same command without `--preflight`.

**Resume:** the full-run command plus `--resume`.

**Status (read-only):**

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase2-status --out "C:\Users\Yosef.g9\Downloads\60k_falling_db_phase2"
```

**Send back ONLY** `C:\Users\Yosef.g9\Downloads\60k_falling_db_phase2\share` (zipped).
