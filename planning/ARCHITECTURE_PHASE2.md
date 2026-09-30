# Phase 2 implementation architecture (draft for AGY review)

Status: **DRAFT 2 (2026-09-29, as implemented)**. Draft 1 went to AGY; the accepted changes (reviews/REVIEW_DECISIONS.md G-01…G-09) and
the lead decisions L-01…L-03 are folded in and summarised in §11. The code lives in `src/falls_ml/phase2/`.

## 1. Command and scope

```
python -m falls_ml meuhedet-phase2 --input <extract.csv> --reference <completed explore dir> --out <new phase2 dir>
                                   [--eda <eda dir>] [--id-pepper-file <file>] [--resume]
                                   [--accept-gate GATE_ID --reason "..."] [--accept-code-change "..."] [--rehearsal]
```

- **One command, autonomous run.** It stops only at a stop gate or on an error; `--resume` continues from the last completed work item.
- **Writes only below `--out`.** The reference explore / EDA / model-report / D-00 folders are read-only and hashed (see §6).
- Python 3.11 / 3.13, CPU only, on Windows. New dependencies: `xgboost-cpu==3.2.0` (Windows; 2 MB wheel), `xgboost==3.2.0` (macOS dev),
  `optuna==5.0.0` (+ SQLAlchemy, alembic, Mako, MarkupSafe, colorlog, tqdm, greenlet), all in a separate hash-pinned
  `requirements-phase2.lock`. The preflight stage STOPs with install instructions if they are missing. If `vcomp140.dll` cannot be found,
  it retries the import after adding scikit-learn's bundled `.libs` folder as a DLL directory.

## 2. Data flow and partitions

```
extract.csv ──contract reader──▶ D00_CLEAN cohort rebuilt (row hash = reference) ──▶ split reproduced (test sha256 = reference)
     └─▶ TEST rows dropped in memory immediately (only their count + sha256 kept) ──▶ work/trainval.parquet (pseudonymised, local only)
TRAIN  (≈ 36.5k rows, ≈ 769 events): registry facts, engineering checks, screening, all fitting, all tuning (nested grouped CV)
VALIDATION (≈ 12.2k, ≈ 256 events): SEALED until SELECTION_FROZEN.json exists; then scored once (validation registry, append-only)
TEST: never loaded after the hash check
```

- **Nested grouped CV in TRAIN.** Outer 5 folds (patient-grouped, outcome-stratified, fixed seed) give out-of-fold (OOF) estimates. Inner
  folds do every supervised step: LASSO/EN λ (inner 10-fold), EN l1_ratio, fractional-polynomial selection, imputation/scaling fits, and
  XGBoost tuning and early stopping. Label-blind steps (redundancy pruning by |Spearman|, quality filters) run once on all TRAIN; they read
  no outcome.
- **Validation seal.** Code that scores VALIDATION rows needs a `ValidationSeal` token. The token is issued only when a valid
  `SELECTION_FROZEN.json` (hash-chained to the TRAIN results) exists. Unsealing is appended once to `validation_evaluation_registry.jsonl`;
  a second unsealing is refused. After unsealing, the frozen shortlist is scored for the confirmatory comparison. Every pre-declared
  configuration (waterfall, domain increments, families) is also scored, as descriptive output only.

## 3. Stages (each with items; an item is the unit of recovery)

| Stage | Items | Main outputs |
|---|---|---|
| S00 PREFLIGHT | 1 | environment (Python, packages, CPU, RAM), input sha256, reference checks, protected-folder hashes, `PHASE2_PLAN.json` (frozen, sha) |
| S01 COHORT | 1 | cohort/split reproduction, TEST dropped, overlap checks, `work/trainval.parquet` |
| S02 REGISTRY | 1 | `01_COLUMN_REGISTRY.csv` (221 rows; D-00 provenance computed on TRAIN+VALIDATION record dates, never labels) |
| S03 ENGINEER | 1 | `02_ENGINEERED_FEATURE_REGISTRY.csv`, `work/features.parquet` (row-wise, parameter-free transforms only) |
| S04 SCREEN | 1 | `03_UNIVARIATE_SCREEN.csv`, `04_REDUNDANCY_CLUSTERS.csv`, `05_FEATURE_QUALITY.csv` (TRAIN only) |
| S05 FEATURE_SETS | 1 | `FEATURE_SETS.json` (frozen, sha; the sets are defined by pre-declared rules applied to S02–S04) |
| S06 LINEAR | (set × outer fold) + (set × final) | penalised LASSO for every set; post-selection refit benchmark for BASELINE_15 and the finalists; OOF predictions |
| S07 ENET | (set × l1_ratio × outer fold) + finals | ≤ 2 sets × {0.1, 0.5, 0.9}; l1_ratio and λ chosen by inner CV |
| S08 XGB | stage 0 / 1 / 2 trials per study (5 outer studies + 1 final) | Optuna SQLite + JSONL ledger; best params and tree count; final boosters of pre-declared configs only |
| S09 SELECT | 1 | pre-declared advancement rule on outer-OOF → `SELECTION_FROZEN.json` (shortlist = BASELINE_15 + ≤ 3 challengers) |
| S10 STABILITY | (finalist set × replicate), 25 each | LASSO selection frequencies (TRAIN bootstrap; retry policy v1; ≤ 10% failure gate) |
| S11 ABLATION | domain LOO (nested) + individual (≤ 15, per-fold hyper-parameters frozen) | `ABLATION_RESULTS.csv` (TRAIN OOF, paired) |
| S12 EXPLAIN | 1–3 | XGB permutation importance on outer-OOF; TreeSHAP (`pred_contribs`) on a ≤ 12k TRAIN sample; selected interactions |
| S13 VALIDATION | 1 | unseal once; confirmatory comparison + descriptive scoring; paired patient bootstrap (2,000) |
| S14 CONSENSUS | 1 | `FEATURE_CONSENSUS.csv` |
| S15 REPORT | 1 | tables, figures, `MANAGEMENT_SUMMARY_HE.md`, `SCIENTIFIC_SUMMARY.md` |
| S16 SHARE | 1 | `share/` + privacy scan + `RUN_STATE_FINAL.json` + `RESUME_AUDIT.json` |

## 4. On-disk layout (`--out`)

```
RUN_STATE.json               progress summary (rewritten atomically after every item; NOT the source of truth)
PHASE2_PLAN.json             frozen plan (config + catalogue + seeds + budgets + code/input hashes); sha256 recorded in every marker
STAGE_XX_COMPLETE.json       stage marker: output sha256s, upstream marker sha256s (hash chain), plan sha, timings
stages/SXX_name/items/<item>/ITEM_COMPLETE.json   item outputs + marker (source of truth)
stages/SXX_name/_incomplete/<attempt-n>/<item>/  partial items moved aside on resume (never read, never deleted)
checkpoints/xgb_optuna.sqlite, checkpoints/xgb_trials.jsonl
logs/events.jsonl            append-only, fsync per event;   RUN_TIMINGS.csv
artifacts/                   01–05 registries, consensus, model comparison, … (aggregate only)
work/                        row-level files (trainval, features, OOF and validation predictions): LOCAL ONLY, never shared
share/                       the only folder to zip for review
```

## 5. Durability and resume protocol

1. **Durable write:** write `x.tmp` → flush → `os.fsync(file)` → close → `os.replace(x.tmp, x)` → fsync the parent directory (POSIX; on
   Windows a directory fsync is not available, so `FlushFileBuffers` on the file plus `MoveFileEx`/`os.replace` is used). Windows
   sharing violations are retried (existing `replace_file`).
2. **Item protocol:** build `items/.tmp-<item>` → durable-write every output → write `ITEM_COMPLETE.json` (list of output sha256s, seed,
   elapsed, plan sha) inside the temp dir → rename the directory to `items/<item>`. An item exists iff its directory has a valid marker
   whose hashes verify. A crash leaves at most one `.tmp-*` directory; on resume it is moved to `_incomplete/attempt-n/`.
3. **Stage protocol:** when every item of a stage is complete, `finalize()` merges item outputs into stage outputs (durably), then writes
   `STAGE_XX_COMPLETE.json`. A stage whose marker exists is verified by hash and never recomputed.
4. **Resume:** re-derive the plan and compare its sha; compare input sha256 (**STOP** if different); compare the code hash (**STOP** unless
   `--accept-code-change` is given, which is recorded); verify every marker's hashes (**STOP** on any mismatch = tampering or disk
   corruption); re-hash protected folders; then continue with the first incomplete item. `resume_count` is incremented and every attempt is
   appended to `RESUME_AUDIT.json`.
5. **Determinism:** every item's RNG seed = `sha256(plan_seed | stage | item)`, independent of execution order and of interruptions. XGBoost
   `nthread` is fixed in the plan. Sorting is always stable, and the columns of every design matrix are in a fixed order.

## 6. Immutability of earlier work

- At S00 every file under the protected folders (explore, EDA, model reports, D-00) is hashed (sha256 + size). The hashes are re-checked
  at the end of each stage and at S16; any change is a STOP.
- Phase 2 never opens a protected file for writing. It reads only `split_audit.json`, `readiness.json`, the reference dataset manifest and
  config, and (for the historical benchmark) the reference run's `predictions_validation.parquet`, read-only. It never reads a
  `predictions_test.parquet`.

## 7. XGBoost / Optuna persistence

- One SQLite storage (`checkpoints/xgb_optuna.sqlite`); one study per (stage 0/1/2 × outer fold k or `final`).
- **Ask/tell loop; each trial is an item.** `trial = study.ask()` → params are suggested with a sampler re-created with
  `seed = f(study seed, number of COMPLETE trials)` → inner 5-fold CV with early stopping (≤ 3,000 trees, patience 100) → item written
  → `study.tell(trial, inner-CV log loss)`.
- **Ledger:** `xgb_trials.jsonl` (append + fsync) records RUNNING / COMPLETE / FAIL with trial number, study, feature set, params, seed,
  inner-CV log loss and average precision, best_iteration per fold, elapsed, status/error, feature/config hash.
- **Reconciliation on resume:**
  - An item that is complete but still RUNNING in SQLite → `tell` the stored value.
  - A trial RUNNING in SQLite with no complete item → marked FAIL (`interrupted`) and its parameters re-enqueued.
  - The target counts COMPLETE trials only, and TPE ignores FAIL trials, so suggestions after a crash equal those of an uninterrupted
    run.
  - If the SQLite file is unreadable, the study is rebuilt from the ledger (`add_trial` of the COMPLETE trials).
- **Model files:** no binary model per trial. The final booster is kept (JSON) only for pre-declared configurations that will be scored on
  VALIDATION (≤ 4), plus per-outer-fold boosters of the promoted XGB configuration (needed for OOF permutation importance).

## 8. Stop gates

- **Hard stops** (resume refuses to continue until the cause is fixed):
  - forbidden, post-index or label-derived input in a feature set;
  - an UNSAFE feature about to be fitted;
  - an UNRESOLVED feature outside the labelled exploratory track;
  - patient overlap between partitions;
  - reference-split mismatch;
  - input hash change;
  - a protected file changed or about to be written;
  - more than 10% model-fitting failures in a resampling stage.
- **Investigation stops** (the run pauses with `INVESTIGATION_<gate>.md`; continuing needs `--accept-gate GATE --reason`, recorded):
  - a single feature with TRAIN univariate AUROC ≥ 0.80;
  - OOF or validation ΔAUROC > 0.05 or AP ratio > 1.5 vs BASELINE_15;
  - calibration slope outside 0.6–1.5 or |CITL| > 0.5 on OOF or validation;
  - one new feature holding > 50% of total |SHAP| or of the permutation loss.

## 9. Share package

Built only in S16 from `artifacts/`, `reviews/` and aggregate tables; small cells (< 10) suppressed; every text file scanned for member ids,
snapshot keys, research-id pseudonyms and the pepper (existing scanner). Row-level files stay in `work/`. The file list follows the user's
specification (README, RUN_MANIFEST, RUN_STATE_FINAL, COLUMN_FUNNEL, FEATURE_REGISTRY, FEATURE_QUALITY, FEATURE_SHORTLIST,
FEATURE_CONSENSUS, MODEL_COMPARISON, DOMAIN_INCREMENTAL_GAIN, ABLATION_RESULTS, OPERATIONAL_CAPACITY, THRESHOLD_RESULTS,
CALIBRATION_SUMMARY, SUBGROUP_SUMMARY, MANAGEMENT_SUMMARY_HE, SCIENTIFIC_SUMMARY, RESUME_AUDIT, RUN_TIMINGS, reviews/, figures/).

## 10. Rehearsal (before the real run)

A synthetic extract with the real schema (existing panel generator + planted Phase 2 signals, planted index-day assessment dates and a
planted proxy). The rehearsal kills the process at chosen points: mid-item, mid-Optuna trial, between rename and marker, mid-bootstrap and
mid-share. It then resumes and proves three things: every completed artifact is byte-identical; the final tables equal those of an
uninterrupted run; and the gates fire on the planted proxy and the planted timing violation.

## 11. Changes after the AGY review (as implemented)

| Area | Draft 1 | Implemented |
|---|---|---|
| Commit record (G-01) | marker inside the temp directory, then rename | directory renamed first; commit record `items/<item>.COMPLETE.json` written LAST (temp → fsync → replace); complete iff the record parses and every listed file verifies |
| Incomplete work (G-04a, G-09-1) | temp dirs moved aside | temp dirs, directories without a record and torn records are moved to `_incomplete/attempt-<n>/`; never opened |
| Tampering | not specified | a committed file whose sha256 changed, or a changed upstream stage record, is a HARD stop |
| Synced folders (G-02a) | not checked | `--out` inside OneDrive / Dropbox / … is refused unless `--allow-synced-folder` |
| Read-only artifacts (G-02c) | – | committed item files and stage outputs get the read-only attribute |
| Optuna state (G-03a, G-06) | SQLite + ledger, stale RUNNING → FAIL + re-enqueue | each trial is a committed item (per-trial truth); the JSONL ledger is the audit log (torn tail moved to a sidecar, reconciled from the records); SQLite is rebuilt from the committed trials at the first use in every attempt (old file moved to `checkpoints/_previous/`), so an interrupted trial leaves no trace; the sampler is re-seeded per trial index → identical suggestions after a crash (unit-tested) |
| Code hash (G-03b) | once per run | every item record and stage record carries `code_sha256`; RESUME_AUDIT.json lists every attempt |
| Threads (G-05) | – | BLAS (threadpoolctl) and XGBoost `nthread` pinned to the value frozen in the plan (default min(4, cores)) |
| Scheduling (G-07) | – | strictly sequential items; BASELINE_15 items first within a stage |
| Share (G-08) | privacy scan | built in `.tmp-share-a<n>/`, scanned for ids / pseudonyms / pepper / local paths / the OS user name, renamed to `share/` only when clean; SHAP figures aggregated (no individual points) |
| Self-test (G-09-3) | – | S00 checks LASSO invariants and bit-determinism, XGBoost determinism and learning on a known-answer problem |
| Validation re-open | once | the registry accepts re-reading VALIDATION after a crash only for the SAME frozen selection (never for a changed one) |
| λ grid (L-01), duplicate columns (L-02), clusters (L-03) | – | grid to 10⁻³ λ_max; exact duplicate design columns dropped per fold; representative-anchored redundancy clusters |
| Rehearsal hooks | – | `FALLS_ML_PHASE2_CRASH_AT="<stage>|<item>|<point>"` kills the process with `os._exit` at item_start / before_rename / before_record (and inside the share build); inert unless set |

## 12. Final version 0.8.1 (after the user's final readiness brief and the final Astra review)

| Module / file | Purpose |
|---|---|
| `phase2/final_config.py` + `tools/freeze_phase2_config.py` | builds the effective configuration (settings + catalogue + result-relevant code constants) as `FINAL_EXPERIMENT_CONFIG.json`; the repository copy and its `.sha256` are the freeze. The runner and the preflight refuse any other effective configuration (`FROZEN_CONFIG_MISMATCH`) unless `--allow-unfrozen-config` (tests only; flagged in every output). The run keeps a read-only copy in `<out>` and verifies it on every resume. |
| `phase2/featuresets.py` | three categories that are never mixed (HISTORICAL_BASELINE_15 / SAFE_DISCOVERY / EXPLORATORY_UNRESOLVED_SENSITIVITY); aliases only within a category; `check_category_invariants` = hard stop INELIGIBLE_FEATURE_IN_SET before any fit |
| `phase2/stages_data.py` | redundancy computed per universe (SAFE / exploratory); `baseline_status`; LABEL_DEATH_SEMANTICS gate after S01 |
| `phase2/cohort.py` | `death_label_audit` (TRAIN + VALIDATION, aggregate) |
| `phase2/stages_models.py` | scope-common inner folds (`inner_fold_seed`); SAFE advancement vs SAFE_BASE; point-estimate benchmark class; attribution contrasts; `VALIDATION_OPENED.json` (write-once, before any VALIDATION row is read; re-read only for the same selection and configuration, else VALIDATION_REOPEN_REFUSED); the frozen configuration list is checked before scoring (VALIDATION_CONFIG_MISMATCH) |
| `phase2/evaluate.py` | per-capacity paired deltas (one sort per model per replicate), calibration bootstrap, smooth calibration curve |
| `phase2/accounting.py` | exact funnel / provenance counts (shared by the preflight and COLUMN_FUNNEL.csv) |
| `phase2/preflight.py` | `--preflight`: environment and lock pins, paths, write permission, disk, frozen configuration, input hash, reference cohort / split / TEST exclusion in memory, provenance counts, category invariant, budgets, runtime paths, hashes; writes nothing to `--out`; last line `SAFE TO START FULL RUN` |
| `phase2/status.py` | `meuhedet-phase2-status`: read-only progress (never uses the Stage recovery code; retries a file being replaced) |
| CLI | `--accept-gate` and `--accept-code-change` are refused without `--resume`; an accepted code change is logged in `logs/gates.jsonl` and the event log |

The commit / resume protocol (`state.py`, `durable.py`, `xgb_tuning.py`) is unchanged. The new write-once and verification steps are covered by
the targeted e2e kills: inside an Optuna trial, inside the one-shot validation and before the share rename; plus the code-change test.
