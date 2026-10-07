# Amendment A-1 — evidence (falls_ml 0.13.1, Phase 5 3.0.0)

| Item | Value |
|---|---|
| Kind | **COMPATIBILITY amendment, not methodological** (entry A-1 in `docs/phase6/AMENDMENTS.md`) |
| Patch commit | `a19132ef66eb63fe17b362b429ad2b669316b687` (code, test, versions, amendment, registration, runbook) |
| Package commit | `625f245d4ffb089c54332eda5973b502d9ed68cf` (the patch + one reworded sentence the package path scan flags; no code) |
| Package | `dist/falls_ml_phase5_0.13.1_mailsafe.zip`, sha256 `964e27545c6665fd7a72ff37e8086935e98f89a895f6b4ed19bd87d51ff41270` |
| Real data | none read in this environment; nothing fitted on real data |

## The defect

The real PRE is a completed 2.2.0 run whose FINAL share was regenerated with the approved 0.12.3 `meuhedet-phase5 --report-only` command and
then extended by the 0.12.3 dashboard. That command leaves `RUN_STATUS.json` status `REPORT_COMPLETE` (stage `REPORT`) with a FINAL
`share/RUN_MANIFEST.json`. The 0.13.0 PRE check (a) accepted only `COMPLETE*` and stopped the preflight:
`PRE_VERIFICATION_FAILED: the PRE run is not complete (RUN_STATUS.json status = 'REPORT_COMPLETE')`. The stop came before any plan,
fold file or model was written.

## The change (`src/falls_ml/phase5/prerun.py`, `pre_summary`)

`REPORT_COMPLETE` is accepted in addition to the unchanged `COMPLETE*` rule. Nothing else in checks (a)–(f) changed, and all of them are
still required for that state: a FINAL manifest, the input sha256, cohort and labels, the frozen fold hash with the folds adopted, every ENET
PRIMARY unit against its `COMPLETE.json`, the exact reproduction of `TOP3_CAPACITY_PRIMARY.csv`. The 0.12.3 report-only branch always writes a
FINAL manifest, so the manifest alone proves nothing about the units; checks (e) and (f) carry the PRE result, as before. The accepted PRE status
is recorded in the POST plan and manifest as `pre_run.run_status`.

Package comparison 0.13.0 → 0.13.1 (both restored): the only source file that differs is `src/falls_ml/phase5/prerun.py`, apart from the
version strings (`VERSION`, `pyproject.toml`, `src/falls_ml/__init__.py`). Every modelling, tuning, capacity, control and reporting module and
`configs/meuhedet/phase5.yaml` are byte-identical. The lock files are identical, so an existing offline bundle remains valid.

## Evidence

| Check | Result |
|---|---|
| Reproduction with the real 0.12.3 code (git worktree at `2398fb9`) on a completed synthetic 2.2.0 PRE: report-only branch → `REPORT_COMPLETE`, stage `REPORT`, FINAL manifest; dashboard → `DASHBOARD_COMPLETE`, privacy passed, units unchanged | identical state to the Windows PRE |
| 0.13.0 preflight against it | `STOPPED [PRE_VERIFICATION_FAILED] - the PRE run is not complete (RUN_STATUS.json status = 'REPORT_COMPLETE')`; output folder: no plan, no folds |
| New regression test `test_a1_report_only_pre_state_is_accepted_only_with_every_other_check` | passes; **fails on the 0.13.0 condition with the exact Windows message** |
| — with `REPORT_COMPLETE`: non-FINAL manifest, missing manifest, missing unit, unit failing its hashes, wrong Top-3% table, missing Top-3% table, another extract, another fold hash, other labels | each stops with `PRE_VERIFICATION_FAILED` |
| — other states `STOPPED`, `RUNNING`, `INTERRUPTED`, `PREFLIGHT_COMPLETE`, `STOPPED_PREFLIGHT`, `PAUSED_TEST_LIMIT`, `report_complete`, `REPORT_COMPLETE_PARTIAL`, empty | each still stops |
| — `COMPLETE`, `COMPLETE_WITH_FAILURES` | still accepted (unchanged) |
| Phase 5 fast tests (contract, repair incl. A-1, capacity, dashboard) | **57 passed** (repository and extracted package) |
| Repository fast suite | **2,467 passed, 1 skipped, 1 failed**: the pre-existing path-scan finding in the Phase 2 document `FINAL_RUN_READINESS.md` (fails identically on `2398fb9`) |
| Package | restore + manifest **400 / 400**; Phase 5 fast tests from the extracted source **57 passed**; the extracted package's preflight against the report-only PRE → `SAFE TO MODEL`, folds adopted, no unit fitted |

## PRE / POST rehearsal through the real command path (synthetic, quick budget)

Continued in the **same output folder** in which the 0.13.0 preflight had stopped, nothing deleted:

1. preflight `--pre-run`: `PRE verified: Phase 5 2.2.0, 2909 patients, 200 events, 5 folds adopted; TOP3_CAPACITY_PRIMARY.csv reproduced for 3
   ENET arm(s)` → `SAFE TO MODEL`; plan `folds_source: PRE`, `pre_run.run_status: REPORT_COMPLETE`, the six checks listed;
2. negative controls, the registered 10 seeds: PASSED, mean AUROC 0.487, mean Recall@Top3 0.026;
3. run with `--resume`: `COMPLETE`, 65 share files, privacy scan PASSED;
4. dashboard: `DASHBOARD_COMPLETE`, privacy passed, units unchanged;
5. PRE folder digest identical before and after (never written to);
6. `REPORT_COMPLETE` PRE copies with an INTERIM manifest, a missing ENET unit, or a wrong Top-3% table: each `STOPPED [PRE_VERIFICATION_FAILED]`.

The POST headline (synthetic, software test only) is identical to the 0.13.0 rehearsal against the same PRE in `COMPLETE` state: OLD 28 → 30
captured falls of 87 selected, Δ +2 [−2, +5]; ADMISSIBLE 45 → 42, Δ −3 [−8, +1].
