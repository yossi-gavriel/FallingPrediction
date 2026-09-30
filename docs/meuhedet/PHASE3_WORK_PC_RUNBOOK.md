# Phase 3 on the work PC (falls_ml 0.9.0, Phase 3 1.0.0)

Phase 3 is a SEPARATE installation. It never touches the Phase 2 installation
(`...\falls_ml_handoff_0.4.0_mailsafe\falls_ml_handoff`), its `.venv`, or its output folder. Do not resume or change Phase 2 from this package; this
package refuses to run Phase 2 in production by design.

All commands are CMD. Replace nothing except, if different on your PC, the four data paths (input, explore, EDA, reports, D-00 folders).

## 1. Install (once, about 10-20 minutes)

1. Save `falls_ml_phase3_0.9.0_mailsafe.zip` to `%USERPROFILE%\Downloads` and extract it there. You get the NEW folder
   `%USERPROFILE%\Downloads\falls_ml_phase3_0.9.0`.
2. Restore the mail-safe files and verify every file (sha256):

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase3_0.9.0"
py -3.11 RESTORE_FILES.py.txt
```

   The last line must be `PACKAGE VERIFIED ...`.
3. Install into this folder only (its own `.venv`; same pinned, hash-checked dependencies as Phase 2 - pip reuses its cache):

```bat
setup_windows.cmd
```

   Wait for `INSTALLATION SUCCESSFUL`. Then check the version:

```bat
.venv\Scripts\python.exe -c "import falls_ml, xgboost, optuna; print(falls_ml.__version__, xgboost.__version__, optuna.__version__)"
```

   Expected: `0.9.0 3.2.0 5.0.0`.

## 2. Preflight (mandatory; a few minutes; fits nothing; writes nothing)

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase3_0.9.0"
set PYTHONUTF8=1
.venv\Scripts\python.exe -m falls_ml meuhedet-phase3 --preflight --input "%USERPROFILE%\Downloads\60k_falling_db.csv" --reference "%USERPROFILE%\Downloads\60k_falling_db_explore" --protect "%USERPROFILE%\Downloads\60k_falling_db_eda" --protect "%USERPROFILE%\Downloads\60k_falling_db_reports" --protect "%USERPROFILE%\Downloads\60k_falling_db_d00" --phase2-out "%USERPROFILE%\Downloads\60k_falling_db_phase2" --out "%USERPROFILE%\Downloads\60k_falling_db_phase3"
```

Read section 4b (TIME CONTRACT) first:

| Line | Must be | If not |
|---|---|---|
| `V1 outcome excludes Index_Date` | `[OK]` (0 positive labels with an event on/before 2025-01-01) | `[FAIL]`: the DWH explanation is contradicted - do NOT run; send the preflight printout (it holds counts only) |
| `V2 window end = Index_Date + 180` | `[OK]` | as V1 |
| `V9 cohort reconciliation` | `[OK]` (FULL_LABELED = 60,851 + 4,168 expected) | `[FAIL]`: send the printout |
| `V3 no predictor record dated after Index_Date` | `[OK]` (attestation HOLDS) | `[WARN]`: the run still works; undated sources become exploratory only |
| `V7 fall-recency proxy / episode audit` | `[OK]` | `[WARN]`: the run will pause at EPISODE_CONTINUATION for your review |

The last line must be exactly `SAFE TO START FULL RUN`. The line `SCIENTIFIC FORECAST: GO` / `NO_GO` tells you whether the full modelling will run
(Outcome A) or the run will stop after the eligibility analysis with the DWH remediation report (Outcome B).

`--phase2-out` only reads two small status files of the Phase 2 folder (whether it opened VALIDATION); nothing there is written, moved or hashed.
You may omit it.

## 3. Full run (same command without `--preflight`; plan several hours - keep the PC plugged in)

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase3 --input "%USERPROFILE%\Downloads\60k_falling_db.csv" --reference "%USERPROFILE%\Downloads\60k_falling_db_explore" --protect "%USERPROFILE%\Downloads\60k_falling_db_eda" --protect "%USERPROFILE%\Downloads\60k_falling_db_reports" --protect "%USERPROFILE%\Downloads\60k_falling_db_d00" --phase2-out "%USERPROFILE%\Downloads\60k_falling_db_phase2" --out "%USERPROFILE%\Downloads\60k_falling_db_phase3"
```

## 4. Progress (read-only, safe at any time, from a second CMD window)

```bat
cd /d "%USERPROFILE%\Downloads\falls_ml_phase3_0.9.0"
.venv\Scripts\python.exe -m falls_ml meuhedet-phase3-status --out "%USERPROFILE%\Downloads\60k_falling_db_phase3"
```

## 5. Resume (after a restart, power loss or any stop): the full-run command + `--resume`

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-phase3 --input "%USERPROFILE%\Downloads\60k_falling_db.csv" --reference "%USERPROFILE%\Downloads\60k_falling_db_explore" --protect "%USERPROFILE%\Downloads\60k_falling_db_eda" --protect "%USERPROFILE%\Downloads\60k_falling_db_reports" --protect "%USERPROFILE%\Downloads\60k_falling_db_d00" --phase2-out "%USERPROFILE%\Downloads\60k_falling_db_phase2" --out "%USERPROFILE%\Downloads\60k_falling_db_phase3" --resume
```

Nothing completed is recomputed. An INVESTIGATION stop writes `INVESTIGATION_<GATE>.md` in the output folder: read it; to continue with a recorded
reason add `--accept-gate <GATE> --reason "<why>"` to the resume command. A HARD stop writes `STOPPED.md`.

## 6. What to send back

Only the folder `%USERPROFILE%\Downloads\60k_falling_db_phase3\share` (zip it). It exists only after the privacy, path, file-type and row-level
scans passed. Never send: the CSV, `<out>\work`, `<out>\stages`, `<out>\checkpoints`, `<out>\logs`, any `id_pepper.txt`, identifiers or pseudonyms.
