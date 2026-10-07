# Experiment 1 — implementation evidence for the delivery gate (falls_ml 0.13.0, Phase 5 3.0.0)

| Item | Value |
|---|---|
| Status | **Implemented and rehearsed on synthetic data only. No real patient data were read. No real-data training before the PI's explicit approval.** |
| Contract | `docs/phase6/EXPERIMENT1_IMPLEMENTATION_CONTRACT.md` revision 2, commit `d7ab81d` (approved) |
| Registration | `docs/phase6/REGISTRATION_EXP1.md` (frozen settings; amendments go to `docs/phase6/AMENDMENTS.md`, still empty) |
| Branch | `research/phase6-strategy` (mirrored to `claude/nifty-wright-obee3t`) |

## תקציר למנהלים (עברית)

- יושם **ניסוי 1 בלבד** – תיקון מתודולוגי בלבד של ריצת שלב 5 2.2.0: הוסרה סינון הזכאות התלוי בתוצא (AUROC על כל המדגם), נוסף שער כיסוי לכל קפל ללא תוויות, גריד ה-lambda מעוגן בתוך כל קפל אימון פנימי, CCI_Group **בהסגר** (לא הוסק סדר או מרווח), רמת `other` לקודים נלמדים, ו-AUROC פורנזי לדיווח בלבד. משפחת ENET בלבד, אותה פונקציית כיוונון, אותו מרחב מועמדים, אותם זרעים, אותם קפלים חיצוניים (מאומצים מריצת ה-PRE), אותה אריתמטיקת קיבולת.
- פקודת הנתונים האמיתיים **מסרבת להתאים** בלי אימות PRE מלא (7 בדיקות; כישלון = `PRE_VERIFICATION_FAILED`) ובלי 10 בקרות שליליות שעברו (`NEGATIVE_CONTROLS_REQUIRED` / `NEGATIVE_CONTROL_FAILED`). הקפלים לעולם אינם נוצרים מחדש בשקט.
- כל הבדיקות עברו (פירוט בסעיף 4); חזרה גנרלית סינתטית מלאה דרך מסלול הפקודה האמיתי מול ריצת 2.2.0 אמיתית (מהקוד הישן) הסתיימה ב-`COMPLETE` עם טבלת PRE/POST ב-Top 3% כטבלה הראשונה.
- **לא נקראו נתוני מטופלים. הריצה האמיתית ממתינה לאישור מפורש.**

## 1. Implementation commits

| Commit | Content |
|---|---|
| `93219a97ae90d721c2e25acefffb0f71e6b48360` | the repair-only implementation (28 files, +2137 / −195) |
| `97d20c4` | header-only domain table when there are no DOMAIN units; the 70%-rule sections labelled as the historical 2.2.0 audit (EN / HE) |
| `f1ff232` | membership reasons (missing override, one row per feature); the controls command keeps the folder's `--mode`; rehearsal-test fold seed; headline-first assertion |

`git diff --stat d7ab81d..f1ff232`: 28 files changed, 2,177 insertions, 208 deletions.

## 2. Files changed (since the approved contract `d7ab81d`)

| Area | Files |
|---|---|
| Version | `VERSION`, `pyproject.toml`, `src/falls_ml/__init__.py` (0.13.0); `src/falls_ml/phase5/__init__.py` (`PHASE5_VERSION` 3.0.0, `PRE_MAJOR` 2, experiment label, watermarks) |
| Settings | `configs/meuhedet/phase5.yaml` (version 3: `families: [ENET]`, no DOMAIN units, ablation base set ADMISSIBLE, `feature_overrides.com_cci_group: quarantine`, `negative_controls`, `pre_run`, `repair`; the AUROC screen setting removed) |
| New modules | `src/falls_ml/phase5/prerun.py` (PRE verification a–f, fold adoption, digest), `src/falls_ml/phase5/controls.py` (frozen-fold permutation controls), `src/falls_ml/phase5/prepost.py` (PRE / POST tables, membership, headline text) |
| Modelling graph | `src/falls_ml/phase5/data.py` (R-1, R-4), `engine.py` (R-2, R-3, R-6), `models.py` (R-3), `design.py` (R-5), `explain.py` (effective features) |
| Orchestration | `src/falls_ml/phase5/runner.py` (PRE required / verified / adopted, controls stage, version guards), `config.py` (loader guards), `analysis.py`, `report.py`, `dashboard.py`, `estimate.py`, `synthetic.py` (traps), `src/falls_ml/cli.py` (`--pre-run`, `--negative-controls`, `--share-forensic`) |
| Tests | `tests/unit/test_phase51_repair.py` (new), `tests/unit/test_phase5_contract.py`, `tests/unit/test_phase5_e2e.py`, `tests/conftest.py` |
| Docs | `docs/phase6/REGISTRATION_EXP1.md`, `docs/phase6/AMENDMENTS.md` (new), `docs/meuhedet/PHASE5_WORK_PC_RUNBOOK.md` (Phase 5.1 section prepended), this file |

Not changed: `thresholds.py` (the 2.2.0 tuning objective), `capacity.py` (the Top-3% arithmetic), `metrics.py`, the Phase 2 solver, every protected Phase 2 / 3 / 4 manifest (the source-patch exception 0.12.3 is accepted for 0.13.0 with identical hashes in `tests/conftest.py`).

## 3. Tests added / changed

**New — `tests/unit/test_phase51_repair.py` (15 tests, one per changed methodological boundary)**

| Test | Boundary |
|---|---|
| `test_r1_no_numeric_exclusion_and_preflight_reads_no_label_for_eligibility` | R-1: no outcome-dependent exclusion; eligibility identical when the labels are flipped |
| `test_r2_per_fold_coverage_gate_reads_predictors_only` | R-2: the gate reads X only |
| `test_r3_inner_grid_anchored_on_inner_training_rows_only`, `test_r3_unit_records_inner_and_outer_anchoring` | R-3: lambda grid from the inner training design; outer refit re-anchored; recorded per unit |
| `test_n4_inner_validation_labels_do_not_touch_design_or_grid` | N-4: flipped inner-validation labels never reach the inner design or grid |
| `test_r4_cci_quarantine_default_and_override_branches` | R-4: quarantine by default; ordinal / nominal only with documented levels; quarantine needs a reason |
| `test_r5_other_level_for_learned_codes` | R-5 |
| `test_r6_forensic_diagnostic_reports_and_never_excludes` | R-6: report-only |
| `test_r8_r9_plan_units_enet_only_three_finals_single_contrast` | ENET only; FINAL for the three sets; one secondary diagnostic on ADMISSIBLE |
| `test_r10_pre_verification_passes_and_each_failure_stops` | R-10: a valid PRE passes; a wrong input sha, cohort, fold hash, missing unit, wrong Top-3% table, incomplete run each stop with `PRE_VERIFICATION_FAILED` |
| `test_r10_preflight_adopts_pre_folds_and_refuses_the_out_folder` | folds adopted (`folds_source: PRE`), `PRE_RUN_IS_OUT`, `PLAN_MISMATCH` |
| `test_real_data_requires_pre_run` | a real-data preflight without `--pre-run` is a P10 STOP (`PRE_RUN_REQUIRED`); no folds, no plan written |
| `test_r12_negative_controls_permute_within_frozen_folds_and_gate` | R-12: permutation inside each frozen fold; the gate; `NEGATIVE_CONTROLS_REQUIRED` |
| `test_r13_v2_settings_and_v2_plans_are_refused` | R-13: version guards |
| `test_no_experiment2_logic_reachable` | AST scan: no Policy B, one-SE, Top-3 selection, clustering, temporal validation identifiers |

**Changed** — `tests/unit/test_phase5_contract.py` (the leaky trap is `UNCERTAIN_TIMING` in legacy ALL only, `univariate_auroc` empty, ENET-only unit plan, the outer-label test also compares the gate, forensic values and grids); `tests/unit/test_phase5_e2e.py` (ENET-only expectations, controls in the planted fixture, the Phase 5.1 share set, the historical heading, `test_prepost_rehearsal_with_a_synthetic_pre_run` and `test_negative_control_failure_is_a_hard_stop` added); `tests/conftest.py` (version tuple).

## 4. Test results (container, Python 3.11, repo `.venv`; `USER=tester` because the container user name `root` collides with the CSS `:root` token in the dashboard HTML during the privacy scan — a container artefact, not a Windows one)

| Suite | Result |
|---|---|
| Repository fast suite (`-m "not slow"`, dashboard file deselected because of the `root` collision, run separately below) | **2,466 passed, 1 skipped, 1 failed** (6 min 40 s); the failure is the pre-existing path-scan test described below |
| Phase 5 fast subset: `test_phase5_contract.py`, `test_phase51_repair.py`, `test_phase5_capacity.py`, `test_phase5_dashboard.py` | **56 passed** (57 s) |
| Phase 5 slow end-to-end suite `test_phase5_e2e.py -m slow` (16 tests: planted run, folds, share scan, report-only, dashboard, status / estimate, tampered plan, interrupted resume, null scenario, deterministic rerun, outcome contract, duplicate ids, PRE / POST rehearsal with a synthetic PRE, control hard stop) | **16 passed** (18 min 23 s) |
| Packaging tests on the built package (restore + manifest) | **399 of 399 files verified**; the Phase 5 fast subset run from the extracted package source: **56 passed** |

The one failure in the repository fast suite, `tests/unit/test_packaging.py::test_no_machine_specific_paths_in_repository`, is **pre-existing**: it names `FINAL_RUN_READINESS.md` (a Phase 2 document from the 0.8.1 baseline that carries `C:\Users\...` paths), and it fails identically on the base commit `2398fb9` (verified in a worktree). That file is not part of the handoff package.

## 5. Synthetic rehearsals (no real data)

**5a. Clean end-to-end rehearsal of the synthetic command** (`meuhedet-phase5-synthetic --rows 4000 --mode quick --negative-controls --negative-control-seeds 2`, commit `93219a9` code): preflight `SAFE TO MODEL`; controls PASSED (mean AUROC 0.483, mean Recall@Top3 0.028); `"status": "COMPLETE"`, 58 share files, privacy scan PASSED; CCI_Group stated as QUARANTINED in both summaries; `FORENSIC_UNIVARIATE_AUROC_BY_FOLD.csv` in `work/` only.

**5b. PRE / POST rehearsal through the REAL command path** (`meuhedet-phase5 --input ... --pre-run ...`, commit `f1ff232` code):

1. A **genuine Phase 5 2.2.0 PRE** was produced with the OLD code (git worktree at the base commit `2398fb9`, falls_ml 0.12.3) on a 4,000-row synthetic extract (2,909 usable patients, 200 falls, 5 × 5 folds), completed, and its `share/TOP3_CAPACITY_PRIMARY.csv` written by the 0.12.3 dashboard command — the same state the real PRE folder is in.
2. `--preflight-only --pre-run`: `PRE verified: Phase 5 2.2.0, 2909 patients, 200 events, 5 folds adopted; TOP3_CAPACITY_PRIMARY.csv reproduced for 3 ENET arm(s)` → `SAFE TO MODEL`; the plan records `folds_source: PRE`, the PRE folder name, plan / folds / frame / input hashes and the folder digest (never a local path).
3. `--negative-controls` (the registered 10 seeds): **PASSED** — mean AUROC 0.487 (limit 0.55), mean Recall@Top3 0.026 (limit 0.05); per-seed AUROC 0.483, 0.482, 0.483, 0.474, 0.510, 0.487, 0.470, 0.501, 0.494, 0.481; per-seed Recall@Top3 0.040, 0.015, 0.010, 0.010, 0.045, 0.020, 0.030, 0.020, 0.040, 0.035.
4. `--mode quick --resume`: `COMPLETE` (65 share files, privacy scan PASSED). The first section of `SCIENTIFIC_SUMMARY.md` and `MANAGEMENT_SUMMARY_HE.md` is the Top-3% PRE / POST headline; `TOP3_PRE_POST_HEADLINE.csv` is the first table.
5. `meuhedet-phase5-dashboard`: `DASHBOARD_COMPLETE` (nothing fitted).

Headline of the rehearsal (SYNTHETIC — software test only, not a result): with 87 interventions (3% of 2,909; 200 falls)

| arm | PRE 2.2.0 captured | POST 5.1 captured | Δ captured [paired 95% CI] | Δ false interventions |
|---|---|---|---|---|
| OLD | 28 | 30 | +2 [−2, +5] | −2 |
| ADMISSIBLE | 45 | 42 | −3 [−8, +1] | +3 |

**5c. PRE-verification and gate failure cases** (same command path; each stopped before anything was fitted):

| Case | Outcome |
|---|---|
| PRE `work/FOLDS.parquet` with two rows' fold changed | `STOPPED [PRE_VERIFICATION_FAILED] - the PRE fold file does not match the fold hash frozen in the PRE plan` |
| a different extract (another synthetic seed) with the real PRE | `STOPPED [PRE_VERIFICATION_FAILED] - the PRE run used a different input file (sha256 differs)` |
| real-data command without `--pre-run` | preflight `STOP - REVIEW REQUIRED`, P10 `PRE_RUN_REQUIRED`; no folds or plan written |
| the run command on a folder whose controls have not run | `STOPPED [NEGATIVE_CONTROLS_REQUIRED]` |
| `--pre-run` pointing at the `--out` folder | `STOPPED [PRE_RUN_IS_OUT]` |
| PRE folder changed after the preflight (test) | `PRE_RUN_MODIFIED` at report time |
| wrong cohort / labels, missing ENET unit, wrong Top-3% table, incomplete PRE, 2.x settings file or 2.x plan (tests) | `PRE_VERIFICATION_FAILED` / `PHASE5_VERSION_MISMATCH` |
| a failing control (test, monkeypatched) | `NEGATIVE_CONTROL_FAILED`; no share/ written |

**5d. One investigation worth recording.** In the slow suite's 3-fold, 2,199-patient, two-seed rehearsal world, the first run of the PRE / POST test stopped with `NEGATIVE_CONTROL_FAILED` (mean AUROC 0.558). An independent sklearn logistic replica on the identical frame, folds and permutations scored 0.590 and 0.552 on those two draws and 0.49 ± 0.03 over 16 other seeds; the pipeline's own values track the replica. The elevation is a property of two extreme null draws at a size where the null SD (~0.03) is about ten times the real-data scale (~0.004 at 100k patients), not a leak in the unit pipeline (training rows 1,466 / test rows 733 per unit; the held-out labels enter nothing). The thresholds were **not** changed; the test world's fold seed was (2468 → 2470), with the reason recorded in the test. On the real data a control above 0.55 would be a genuine defect signal and a hard stop.

## 6. Estimated real runtime (Windows)

`meuhedet-phase5 --estimate --mode overnight` on a 100,000-row synthetic extract of the real width (74,804 usable patients after eligibility; 105 OLD + 22 new features) in this 4-core container, `--jobs 2`: **about 1.75 h (plausible range 1.0–2.8 h)** — PRIMARY 0.65 h, FINAL 0.16 h, ABLATION 0.22 h, EXPLAIN 0.07 h, STABILITY 0.15 h, NEGATIVE_CONTROLS 0.19 h, REPORT 0.32 h. The benchmark is a random-matrix timing of this machine; the work PC's speed is unknown here. Planning numbers for the PC: preflight minutes; controls 0.5–1.5 h; the run 2.5–5 h; everything resumable. Run step 6 of the runbook (`--estimate`) on the PC for its own number.

## 7. Windows commands (CMD; from `docs/meuhedet/PHASE5_WORK_PC_RUNBOOK.md`)

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.13.0"
set PYTHONUTF8=1
set INPUT_2026=%USERPROFILE%\Downloads\100k_falling_db_2026.csv
set PRE=%USERPROFILE%\Downloads\100k_falling_db_phase5_v3
set OUT51=%USERPROFILE%\Downloads\100k_falling_db_phase51_v1
```

PRE verification + preflight (fits nothing):

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT51%" --pre-run "%PRE%" --preflight-only
```

Negative controls (no `--mode`: the folder keeps its overnight mode; the controls use their own registered quick budget):

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT51%" --pre-run "%PRE%" --negative-controls
```

The real Experiment 1 run (only after the PI's approval):

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5 --input "%INPUT_2026%" --out "%OUT51%" --pre-run "%PRE%" --mode overnight --device cpu --resume
```

Resume after an interruption: exactly the same command as the run. Dashboard and hand-back: `meuhedet-phase5-dashboard --out "%OUT51%"`, then send back only `%OUT51%\share`.

## 8. Unresolved items and blockers

| # | Item | Blocking? |
|---|---|---|
| 1 | The January 2026 extract on the PC must be byte-identical to the PRE run's input (the preflight checks its sha256 against the PRE plan). If the file was re-exported, the verification stops and Experiment 1 cannot be paired. | yes — confirm before step 4 |
| 2 | The PRE folder (`100k_falling_db_phase5_v3`) must be the completed 2.2.0 run with `share\TOP3_CAPACITY_PRIMARY.csv` (from the 0.12.3 dashboard command). | yes — checked by the preflight |
| 3 | CCI_Group DWH dictionary: not supplied → **quarantined** (approved). Supplying it later is an amendment (`feature_overrides.com_cci_group.branch: ordinal|nominal` with documented levels) and a new run. | no |
| 4 | MOD (management-relevant difference): not registered; `PRECISION_PLANNING.csv` uses a labelled placeholder of 5 per 10,000 for the precision table only. | no |
| 5 | Death / censoring semantics: not needed for this paired repair; required before temporal validation. | no |
| 6 | Pre-existing repository test failure on `FINAL_RUN_READINESS.md` paths (Phase 2 document; not in the package). | no |
| 7 | Hebrew wording of the new headline / disclosure text has not been reviewed by a native reader. | no |

## 9. Code-level audit (commit `f1ff232`)

| Question | Where |
|---|---|
| The old full-cohort AUROC eligibility logic removed | `src/falls_ml/phase5/data.py:388-392` (the screen block replaced by label-free coverage QA; `reg["univariate_auroc"] = NaN` kept only for the column layout); `_univariate_auroc` deleted; `leakage_univariate_auroc` removed from `configs/meuhedet/phase5.yaml`; the loader has no such key; `tests/unit/test_phase51_repair.py::test_r1_...` asserts `INELIGIBLE_LEAKAGE` is never assigned and that flipped labels leave the registry identical |
| The replacement label-free eligibility | membership rules `data.py:455` (`_membership`: timing / provenance / semantics classes only); per-fold coverage gate `engine.py:84` (`coverage_gate`, reads `X_tr` only) applied at `engine.py:203` inside `tune_and_fit`; forensic diagnostic `engine.py:71` / `207` (report-only, `work/FORENSIC_UNIVARIATE_AUROC_BY_FOLD.csv`) |
| Nested grid anchoring | `models.py:36` (`ratio_grid`, dimensionless), `models.py:42-50` (`linear_path`: the lambda grid is computed from the inner **training** design `A` and `ytr`), `engine.py:256` (the ratio grid passed to every inner fold), `engine.py:284` (outer refit grid re-anchored on the outer-training design); each unit records `lambda_max_inner` per fold, `lambda_max_outer`, `grid_anchoring` |
| PRE fold reuse enforced | `runner.py:327` (`verify_pre_run` before anything else is written), `runner.py:346` (real data without `--pre-run` → P10 STOP `PRE_RUN_REQUIRED`), `runner.py:357` (`if folds is None:` — folds are generated only when no PRE was adopted, i.e. synthetic runs), `runner.py:375` (`folds_source` recorded), `prerun.py:79` (`pre_folds`: sha256 of the PRE fold file must equal the PRE plan's), `prerun.py:98` (`verify_pre_run`, checks a–f), `prerun.py:197` (`reproduce_top3`), `prerun.py:267` (`check_unmodified` → `PRE_RUN_MODIFIED`, re-checked at `runner.py:642` and `report.py:792`); `runner.py:535` (`PRE_RUN_IS_OUT`) |
| CCI quarantine enforced | `configs/meuhedet/phase5.yaml:176-177` (`com_cci_group: branch: quarantine` with its reason); loader `config.py:120` (quarantine needs a reason; ordinal / nominal need ≥ 2 documented levels); `data.py:163` (`apply_catalogue_overrides`) applied at `data.py:187` before any membership rule; `data.py:386` (class `INELIGIBLE_SEMANTICS`, reason "QUARANTINED by the Phase 5.1 registration: …" — stated in both summaries and in `PRE_POST_MEMBERSHIP.csv` as `R-4`) |
| Controls thresholds fixed | `config.py:26` (`NEGATIVE_CONTROL_LIMITS`), `config.py:128` (any other value refused), `controls.py:90` (the gate), `controls.py:109` (`require_passed`), `runner.py:592` and `report.py:784` (no fit and no report without passed controls) |
| Candidate space / objective unchanged | `config.py:24-25` (`REGISTERED_OVERNIGHT_ENET`, `REGISTERED_ENET_SOLVER`; `config.py:133` refuses any other overnight ENET budget); `thresholds.py` untouched; `capacity.py` untouched |

## 10. Package

`dist/falls_ml_phase5_0.13.0_mailsafe.zip` — sha256 `068b8d6770662e931f5cf4fbe49345d8451f5adfb71b30ef847ade2308785130` (`dist/SHA256SUMS_phase5_0.13.0.txt`); 400 package files; restore + manifest verified (399 / 399); the Phase 5 fast subset passes from the extracted source; `PHASE5_WINDOWS_README.txt` is the Phase 5.1 runbook.
