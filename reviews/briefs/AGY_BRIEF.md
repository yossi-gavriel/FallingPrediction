# Briefing for an engineering reviewer: crash-safe, resumable ML experiment runner (Phase 2 of a clinical falls-prediction project)

You are reviewing an architecture DRAFT, not code. There is no repository to read: answer from this text only, without tools. Be
concrete and concise (≤ ~1,500 words): numbered answers to the 9 questions at the end; for each problem give the failure scenario and
the specific fix. Do not restate the design.

Context: Python 3.11.4 on a Windows 10/11 work PC (CPU only, possibly 4–8 cores, antivirus and possibly OneDrive active; the machine may
sleep, reboot or lose power during a multi-hour run). ~49k rows × ~150 design columns; a LASSO CV fit takes ~5–40 s here and maybe
3× more there; ~700 XGBoost fits in total. Clinical data: nothing row-level may leave the PC; only the `share/` folder is uploaded for
review. Earlier result folders must remain byte-identical (they were already reviewed). Existing code base: numpy/pandas/scikit-learn,
an in-house LASSO solver, an append-only test-evaluation registry, an atomic JSON writer (temp + os.replace with a retry loop for Windows
sharing violations, but no fsync), and a resume mechanism for an earlier multi-run command (hash every earlier file before/after,
append-only registries, frozen plan sha re-derived on resume).

---

# Phase 2 implementation architecture (draft for AGY review)

Status: DRAFT 1 (2026-09-29). It builds in the Astra recommendations I intend to accept (see `reviews/REVIEW_DECISIONS.md`). It is written
before any Phase 2 code exists.

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

---

## Questions (answer each)

1. Checkpoint/resume architecture: is the item/stage/marker protocol correct? Which crash windows are still unsafe (between which two
   operations), and what is the fix?
2. Artifact immutability: is hashing protected folders at start/stage-ends enough? What else could modify or corrupt completed
   artifacts (Windows specifics, antivirus, OneDrive, Excel opening a CSV), and how to detect or prevent it?
3. Experiment registry: are the plan/marker/ledger/registry files the right sources of truth? Any redundancy or missing record (e.g.
   for audit by an external reviewer)?
4. Crash recovery: directory rename semantics on Windows/NTFS (atomicity of renaming a directory, fsync of directories, FlushFileBuffers),
   partial writes of parquet/CSV/SQLite, and what `_incomplete/` handling should do.
5. Deterministic/reproducible execution: seeds per item, XGBoost `nthread` and float determinism on CPU `hist`, BLAS thread counts in
   numpy (OpenBLAS/MKL) and their effect on bit-reproducibility across resume; what to pin and what to record.
6. Optuna/XGBoost persistent state: is the ask/tell + ledger + re-seeded sampler + stale-RUNNING reconciliation scheme sound with
   Optuna 5 and SQLite on Windows? Pitfalls (locking, WAL, `enqueue_trial`, TPESampler state, heartbeats)?
7. Efficient compute scheduling: sequential items with multi-threaded numerics vs process-level parallelism (joblib) given
   checkpointing and determinism; ordering of stages to get the most valuable results earliest.
8. Share-package design: what could leak row-level data or identifiers into `share/` (figures, metadata, error messages, paths,
   usernames), and what checks to add.
9. Can the implementation resume safely after an abrupt machine shutdown? List the top 5 residual risks, ranked, with mitigations.
