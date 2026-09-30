1. Copy this folder to the work computer.
2. Open CMD in this directory.
3. Run:
   setup_windows.cmd
4. Wait for:
   INSTALLATION SUCCESSFUL
5. Run:
   run_demo.cmd

---

# falls_ml on Windows — handoff guide

This folder is a complete, self-contained copy of the `falls_ml` fall-risk prediction pipeline (eFalls reproduction for Meuhedet).

- It contains **no real patient data, no credentials and no database connection**. The only datasets are synthetic fixtures in `data\fixtures\`.
- Everything the demos produce is software-test output: **SYNTHETIC DATA – NOT SCIENTIFIC RESULTS**.
- Installation needs no administrator rights. Python packages go into `.venv\` inside this folder.

**Tip — open CMD in this folder:** in File Explorer, click the address bar, type `cmd` and press Enter. Use CMD, not PowerShell.

## Requirements

| Item | Requirement |
|---|---|
| Operating system | Windows 10 or Windows 11, 64-bit (x64). Windows on ARM is not supported |
| Python | CPython **3.11.x or 3.13.x, 64-bit** (x86-64). The lock files carry wheels for those two series only; 32-bit Python, ARM64 and other versions (including 3.12) are rejected with a clear message |
| Rights | Normal user account; no administrator rights |
| Network | Online mode: access to the Python package index or an internal mirror. Offline mode: none (see *Offline installation*) |
| Disk space | About **1.5 GB** free: virtual environment 0.6–0.8 GB, offline wheels about 0.2 GB (offline mode only), demo outputs up to 0.2 GB |
| Folder path | At most **103 characters** (for example `C:\Projects\Falls Research\falls_ml_handoff`) unless Windows long-path support is enabled; see *Troubleshooting* |
| Time | Measured on the development Mac (Apple silicon, pip download cache already filled): setup steps 4–10 about **2 min 20 s**, of which the test suite about 1 min 15 s; `run_demo.cmd` about **2 min 45 s**; `run_full_demo.cmd` about **9 min**. Windows has not been timed: expect it to be slower, especially the first online installation (downloads) and on computers where antivirus software scans every file written into `.venv\` and `demo_outputs\` |

### If Python 3.11 or 3.13 is already installed

Nothing to do: run `setup_windows.cmd`. Many managed computers already have CPython 3.11 — that is supported.

### Installing Python without administrator rights

1. On python.org, open *Downloads → Windows* and choose the **Windows installer (64-bit)** for the latest Python 3.13.x release (3.11.x also works).
2. On the first installer screen, **untick** "Use admin privileges when installing py.exe". Ticking "Add python.exe to PATH" is optional.
3. Click **Install Now**. Python installs for the current user under `%LOCALAPPDATA%\Programs\Python\Python313`, together with the `py` launcher.
4. Open a **new** CMD window and check: `py -3.13 --version` prints `Python 3.13.x` (or `py -3.11 --version` for 3.11).

If your organisation provides Python through a software portal or the Python install manager, any CPython 3.11 or 3.13 64-bit works, as long as `py -3.13`, `py -3.11`, `python` or `python3` starts it.

## First installation

Run `setup_windows.cmd` from CMD in this folder. Double-clicking it also works: the window then waits for a key press at the end so the result stays visible. Each step prints `[n/10] ...` followed by `[OK] ...` (or `[SKIPPED] ...`), or by a failure block. Setup stops at the first failed step and exits with code 1 (code 2 for an unknown option); it never continues after a failure.

| Step | What happens |
|---|---|
| `[1/10] Checking Python...` | Tries `py -3.13`, `py -3.11`, `python`, `python3` in that order (3.13 is preferred; 3.11 is equally supported). Skips the Microsoft Store placeholder, reports versions that were found but are unsupported, and remembers the full path of the interpreter it uses |
| `[2/10]` Creating virtual environment | Creates `.venv\` with that interpreter. A valid existing `.venv` is reused. A broken one stops setup with advice to rerun with `--recreate-venv` |
| `[3/10]` Activating virtual environment | Activates `.venv` and checks that the active Python is the one inside `.venv` |
| `[4/10]` Installing build tools | First checks that it runs with the `.venv` Python, that both lock files are present and that the folder path is short enough (see *Troubleshooting*); in offline mode it also checks `offline_packages\` against the lock files. Then installs pip, setuptools, wheel and packaging from `requirements-build.lock` (exact versions, sha256-checked, binary wheels only) |
| `[5/10]` Installing dependencies | Installs numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, PyYAML, pyarrow, pytest and their dependencies from `requirements.lock` (exact versions, sha256-checked, binary wheels only) |
| `[6/10]` Installing falls_ml | Installs `falls_ml` from this folder in editable mode, without network access |
| `[7/10]` Validating environment | Installed versions equal both lock files, `pip check` is clean, every `falls_ml` module imports; writes `environment_report.txt` |
| `[8/10]` Running test suite | Runs the fast test suite (`pytest -m "not slow"`); temporary files go to `demo_outputs\pt\` |
| `[9/10]` Running synthetic smoke test | Runs two small synthetic experiments (published eFalls scoring and the retrained LASSO with fixed FP terms) in `demo_outputs\setup_smoke\` |
| `[10/10]` Checking model save/load and prediction equality | Confirms that the saved model bundles, a re-saved copy and `python -m falls_ml predict` reproduce the in-run test predictions (maximum absolute difference at most 1e-9) |

Steps 4 and 5 install offline when `offline_packages\OFFLINE_MANIFEST.json` exists, otherwise online (see the options below).

Success ends with:

```
========================================
INSTALLATION SUCCESSFUL
========================================
```

followed by the install mode, the test counts, the location of the logs, the time of each step and the next commands. A failure ends with a block naming the failed step, the reason, the recommended action and the log file:

```
========================================
INSTALLATION FAILED
========================================
Step: ...
Reason: ...
Recommended action: ...
Log file: ...
```

Options (combine as needed):

| Option | Effect |
|---|---|
| `--online` | Force download from the package index even if `offline_packages\` exists |
| `--offline` | Force installation from `offline_packages\` only |
| `--recreate-venv` | Delete `.venv\` (only that folder) and create it again |
| `--skip-tests` | Skip the test suite in step 8. For diagnosis only: the result then reads `INSTALLATION SUCCESSFUL (tests skipped)`; run `verify_installation.cmd` before real use |

Logs are written to `setup_logs\`, one file per step (for example `setup_logs\05_dependencies.log`), plus `setup_logs\00_summary.log`. Running `setup_windows.cmd` again is safe: it reuses the virtual environment, pip keeps packages that already match the lock files, and the tests and smoke test run again.

**What every command sets up (only inside that command, not in your CMD window).** Python runs in UTF-8 mode; `PYTHONHOME` and `PYTHONPATH` are cleared; the pip settings `PIP_USER`, `PIP_TARGET` and `PIP_PREFIX` are cleared because they would install packages outside `.venv` (`environment_report.txt` records only whether each was set, yes or no); `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, `PIP_CERT`, the proxy variables and `pip.ini` stay in effect; once `.venv\` exists, matplotlib keeps its settings and font cache in `.venv\mplconfig\` instead of your user profile.

**Automation.** When a command is started through `cmd /c` (double-click, a shortcut, a scheduled task), it waits for a key press at the end. Set `FALLS_ML_NO_PAUSE=1` to switch that off, for example `set FALLS_ML_NO_PAUSE=1` in the calling script. Exit codes: 0 success, non-zero failure (1 for a reported failure, 2 for an unknown option).

## Offline installation

Use this when the work computer cannot download Python packages.

1. **On a computer with internet access** (Windows, macOS or Linux), take a copy of the *same* package version and run one of:
   - Windows: `prepare_offline_package.cmd` (default target `windows-amd64-cp313`)
   - macOS/Linux: `bash scripts/posix/prepare_offline.sh` (same options)
   - any OS: `python scripts\handoff\prepare_offline.py` (macOS/Linux: `python3 scripts/handoff/prepare_offline.py`)

   This downloads exactly the wheels pinned in the lock files for Windows x64 into `offline_packages\` (on Windows the default target matches the Python series running the command, e.g. `windows-amd64-cp311` with Python 3.11; from a Mac/Linux machine pass `--target windows-amd64-cp311` or `--target windows-amd64-cp313` to match the work computer), checks every sha256 and writes `offline_packages\OFFLINE_MANIFEST.json` last. Any Python 3.9 or newer with pip is enough for the Windows target. `--target current` prepares wheels for the preparing computer's own platform instead (it needs 64-bit Python 3.11 or 3.13 on Windows x64 or macOS arm64) and is **not** usable on a Windows work computer when prepared on a Mac.
2. Copy the whole `offline_packages\` folder into the package folder on the work computer, next to `setup_windows.cmd`.
3. Run `setup_windows.cmd`. It detects `offline_packages\OFFLINE_MANIFEST.json` and installs without network access. To be explicit, run `setup_windows.cmd --offline`; to ignore the bundle, run `setup_windows.cmd --online`.

**Running it again.** The command refuses a non-empty `offline_packages\`. Add `--force` to replace a bundle it created earlier (for example after the lock files changed): it deletes only the files listed in the old `OFFLINE_MANIFEST.json` and stops without deleting anything if the folder holds other files. `--verify-only` checks an existing bundle against the lock files and the platform of the computer it runs on (so a Windows bundle only passes on Windows).

The manifest records the sha256 of both lock files. Setup refuses a bundle that was prepared for different lock files (another package version), for another Python or for another platform. If setup reports *offline bundle is for another platform*, the bundle was made with `--target current` on a non-Windows computer: on the computer with internet run `prepare_offline_package.cmd --target windows-amd64-cp313 --force` (or `bash scripts/posix/prepare_offline.sh --target windows-amd64-cp313 --force`) and copy the folder again.

## Verify installation

```bat
verify_installation.cmd
```

It runs 9 checks and ends with `VERIFICATION SUCCESSFUL` or `VERIFICATION FAILED` (exit code 1). A check that depends on a failed check is reported as skipped; every failed check is listed.

1. Python environment: supported interpreter, installed versions equal `requirements.lock` and `requirements-build.lock`, `pip check`
2. Every `falls_ml` module imports from this folder and the version equals `VERSION`
3. Feature specifications load
4. All experiment configurations load (ablation configurations included)
5. All synthetic fixtures in `data\fixtures\` validate against the feature specification named in their manifest
6. The command-line interface starts and lists its commands
7. Core tests (`pytest -m "not slow"`)
8. Synthetic end-to-end scoring (`demo_outputs\verify\smoke\`)
9. Model bundle save/load with identical predictions

| Option | Effect |
|---|---|
| `--skip-tests` | Skip check 7 (faster; the other 8 checks still run; the result reads `VERIFICATION SUCCESSFUL (tests skipped)`) |
| `--full-tests` | Run the complete test suite, including slow tests (on the development Mac the 13 slow tests alone took about 4 min 15 s on top of the core tests) |

Verification results go to `demo_outputs\verify\` (replaced on every run) and temporary test files to `demo_outputs\pt\`.

## Run the synthetic demo

```bat
run_demo.cmd
```

Baseline demo (about 2 min 45 s on the development Mac): validates the synthetic fixture, runs the **published eFalls equation** and trains the **retrained eFalls LASSO**, writes metrics, reports and model bundles, reloads both bundles and checks their predictions, scores a CSV with both models through the command-line interface and compares the two experiments. Output: `demo_outputs\baseline\` (replaced on every run), summary in `DEMO_SUMMARY.md`.

```bat
run_full_demo.cmd
```

Full demo (about 9 minutes on the development Mac): validates the three synthetic fixtures; runs all 7 fixture experiments (published scoring, retrained LASSO, LASSO with fixed FP terms, unpenalised logistic regression, elastic net, random forest, histogram gradient boosting); trains the **reduced eFalls predictor set** (`efalls_retrained_reduced` on `data\fixtures\synthetic_reduced_v1`); runs the Meuhedet-enhanced ablation (illustrative feature groups, `data\fixtures\synthetic_enhanced_v1`), the comparison of all runs, CLI scoring, drift monitoring on unchanged and deliberately shifted rows, and an exact reproduction of one run. Output: `demo_outputs\full\` (replaced on every run).

Each run folder contains `metrics.json`, `report.md`, `report.html`, `plots\` and the model bundle (`model\`). The console output and the generated folders carry the label **SYNTHETIC DATA – NOT SCIENTIFIC RESULTS** (file `SYNTHETIC_DATA_NOT_SCIENTIFIC_RESULTS.txt` in `demo_outputs\baseline\`, `demo_outputs\full\` and their output folders such as `runs\` and `comparison\`). The numbers show that the software works; they say nothing about fall risk in any population.

## Where results are created

| Path | Created by | Content |
|---|---|---|
| `.venv\` | setup | Python virtual environment (delete only with `setup_windows.cmd --recreate-venv`) |
| `.venv\mplconfig\` | every command, once `.venv\` exists | matplotlib settings and font cache |
| `setup_logs\` | setup | One log per installation step (`NN_<step>.log`) and `00_summary.log` |
| `environment_report.txt` | setup (step 7) | OS, Python, pip, installed package versions, project version, lock-file hashes, install mode; yes/no only for network settings (`PIP_INDEX_URL`, proxies, `PIP_CERT`, ...) and for `PIP_USER` / `PIP_TARGET` / `PIP_PREFIX`. No environment variable values |
| `demo_outputs\setup_smoke\` | setup (step 9) | Smoke-test runs |
| `demo_outputs\verify\`, `demo_outputs\pt\` | verification, test suite | Verification runs and temporary test files |
| `demo_outputs\baseline\` | `run_demo.cmd` | Baseline demo |
| `demo_outputs\full\` | `run_full_demo.cmd` | Full demo |
| `offline_packages\` | `prepare_offline_package.cmd` | Offline wheels and `OFFLINE_MANIFEST.json` |
| `runs\`, `reports\` | later real experiments (`train`, `compare`) | Not used by the demos. Real-data results stay on approved storage |

To remove demo outputs: `clean_demo_outputs.cmd` (add `--dry-run` to only list what would be deleted). It deletes only folders under `demo_outputs\` that carry the `.falls_ml_generated` marker, and `demo_outputs\` itself when the scripts created it and nothing unmarked is left in it. Unmarked folders, loose files, links and everything outside `demo_outputs\` (`.venv\`, `setup_logs\`, `offline_packages\`, `data\`, `runs\`, `reports\`) are never touched.

## How to add a real dataset later

Real data is **not** part of this package. When an approved Meuhedet extract exists:

1. Read `data\README.md`: required columns, types, allowed values, missing values, outcome window and date rules. `data\example_schema.csv` lists every column; `data\example_header_only.csv` is the minimum header of an extract.
2. Activate the environment in CMD: `.venv\Scripts\activate.bat`
3. Build an immutable modelling dataset from the local extract file (`build-dataset` reads one `.parquet` or `.csv` file, validates it strictly and never connects to a database; it refuses a non-empty output folder):

   ```bat
   python -m falls_ml build-dataset --input <extract.parquet|.csv> --out data\meuhedet\<dataset_version> --dataset-version <dataset_version> --mapping-version <mapping_version> --source "<source description>" --data-freeze-date YYYY-MM-DD --scientific-use-allowed --approval-reference "<approval id>"
   ```

4. Validate it: `python -m falls_ml validate-dataset --dataset data\meuhedet\<dataset_version>` (the full or reduced feature specification is selected from the dataset manifest)
5. Train with the Meuhedet templates in `configs\experiments\` (published scoring first):

   ```bat
   python -m falls_ml train --config configs\experiments\efalls_published_scoring.yaml --dataset data\meuhedet\<dataset_version>
   python -m falls_ml train --config configs\experiments\efalls_retrained_lasso.yaml --dataset data\meuhedet\<dataset_version>
   ```

The templates accept only datasets built with `--scientific-use-allowed` and an approval reference. **Scientific use starts only after the Meuhedet mappings are clinically validated and the decision register is signed off** (`docs\EFALLS_REPRODUCTION_SPEC.md`, section 15). Keep extracts, datasets, `runs\` and `reports\` from real data on approved storage and never add them to this package.

## Meuhedet wide table (Phase 1)

`docs\meuhedet\WORK_PC_RUNBOOK.md` has the exact CMD lines for the research-wide table `V_Falls_Prediction_Wide_1` (one exported file, no database). The first real-data experiment is one command on a local Excel export:

```bat
.venv\Scripts\python.exe -m falls_ml meuhedet-explore --input "C:\FallsData\falls_extract_2025_01_01.xlsx"
.venv\Scripts\python.exe -m falls_ml meuhedet-audit --input <extract.csv|.parquet> --out <audit folder> --index-date 2025-01-01
.venv\Scripts\python.exe -m falls_ml meuhedet-build --input <extract> --out <new dataset folder> --index-date 2025-01-01 --dataset-version <V> --data-freeze-date YYYY-MM-DD
.venv\Scripts\python.exe -m falls_ml train --config configs\experiments\meuhedet\phase1_180d_exploratory_efalls_reduced.yaml --dataset <dataset folder>
```

The audit output contains aggregates only (no identifiers, cells below 10 suppressed) and is the first thing to send for review. The
mapping (`docs\meuhedet\MAPPING_TABLE.md`), the column inventory and the datatype contract are generated from `configs\meuhedet\*.yaml`.
The run is labelled "Meuhedet 180-day exploratory - NOT eFalls reproduction" (see `docs\meuhedet\PHASE1_REPORT.md`).

## How to run a reduced eFalls feature set

If the data-availability audit shows that some of the 78 eFalls predictors cannot be extracted:

1. Edit `configs\experiments\efalls_retrained_reduced.yaml`: under `preprocessing.features` list **exactly the available predictors**: a strict subset of the 78 that includes `age_years` and every fractional-polynomial variable. The 12 predictors left out in the shipped template (66 of 78 listed) are **only an illustration**: every Meuhedet mapping is still TO_BE_MAPPED, so the list must be replaced by the result of the audit.
2. Build the dataset with only those columns by adding `--features-from-config configs\experiments\efalls_retrained_reduced.yaml` to the `build-dataset` command (instead of `--feature-spec`/`--extension`), then validate it with the same option:

   ```bat
   python -m falls_ml validate-dataset --dataset data\meuhedet\<dataset_version>_reduced --features-from-config configs\experiments\efalls_retrained_reduced.yaml
   ```

3. Train: `python -m falls_ml train --config configs\experiments\efalls_retrained_reduced.yaml --dataset data\meuhedet\<dataset_version>_reduced`

A reduced run is **never a full eFalls reproduction**. Its report states *Available eFalls predictors: X / 78*, the coverage percentage and the list of unavailable predictors under the warning *REDUCED eFalls predictor set – NOT a full eFalls reproduction*. If all 78 predictors are available, use `efalls_retrained_lasso.yaml` instead (the reduced config refuses all 78). The published scoring and full retraining experiments stay strict: they refuse a reduced dataset.

**Synthetic rehearsal.** `run_full_demo.cmd` shows this on `data\fixtures\synthetic_reduced_v1` with `configs\experiments\fixture\efalls_retrained_reduced.yaml`. That fixture has the same patients and rows as `synthetic_v1` but only the declared predictor columns; it was generated with

```bat
python -m falls_ml make-fixture --out data\fixtures\synthetic_reduced_v1 --n-patients 3000 --features-from-config configs\experiments\fixture\efalls_retrained_reduced.yaml
```

To try your own declared subset on synthetic data, run `make-fixture` with your config and a new `--out` folder outside `data\fixtures\`, then `validate-dataset --dataset <that folder> --features-from-config <your config>`.

## Troubleshooting

First look at the failure block: it names the step, the reason and the log file in `setup_logs\`. The logs and `environment_report.txt` contain no patient data and no environment variable values, so they can be shared with IT support.

**"Python was not found; run without arguments to install from the Microsoft Store" / Python opens the Store.** The Windows *App execution alias* is a placeholder, not Python. Install Python 3.13 (or 3.11) as described above. Optionally turn the placeholders off: *Settings → Apps → Advanced app settings → App execution aliases* → switch off `python.exe` and `python3.exe`. Then open a new CMD window.

**"unsupported" Python version, 32-bit or ARM64.** Setup lists the interpreters it found. Python 3.12 and older-than-3.11 versions are refused because the lock files hold wheels for 3.11 and 3.13 only. Install a **64-bit** Python 3.13 (or 3.11) installer. `py -0p` lists every Python the launcher knows, with its path.

**`py` is not recognized.** The launcher was not installed. Setup then tries `python` and `python3` on `PATH`. Either rerun the Python installer with the launcher selected, or tick "Add python.exe to PATH".

**Downloads fail: proxy, SSL certificate or internal mirror.** Set the variables in the same CMD window, then rerun setup:

```bat
set HTTPS_PROXY=http://proxy.example.org:8080
set PIP_INDEX_URL=https://pypi-mirror.example.org/simple
set PIP_CERT=C:\Certificates\corporate-root-ca.pem
setup_windows.cmd
```

Use the proxy, mirror and certificate names your IT department gives you. Recent pip versions use the Windows certificate store, so `PIP_CERT` is often unnecessary. Do not write user names or passwords into files or scripts. Setup clears `PIP_USER`, `PIP_TARGET` and `PIP_PREFIX` itself, but not settings in `pip.ini`: if `pip.ini` sets `user = true` (or `target`/`prefix`), remove that setting, because such installs cannot target a virtual environment. If downloads remain blocked, use offline mode.

**Offline mode: "No matching distribution found", a bundle/lock mismatch or "offline bundle is for another platform".** `offline_packages\` was prepared for another target or another package version. On the computer with internet, run `prepare_offline_package.cmd --target windows-amd64-cp313 --force` from this package version and copy the new `offline_packages\` folder completely (do not mix files from different bundles).

**"THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE".** The mirror or a proxy served different files than the locked ones, or a download was corrupted. Prepare the offline bundle again (with `--force`), or ask IT whether the mirror modifies packages. Never remove the hash checking.

**"Access is denied", "WinError 5" or "WinError 32" (file in use).** Close editors and other CMD windows that use `.venv`, wait for the antivirus scan to finish, and rerun setup. If it persists, ask IT to allow this folder for the antivirus scanner.

**Scripts are blocked or Windows warns about files from the internet.** The zip carries the "Mark of the Web". Before extracting: right-click the zip → *Properties* → tick **Unblock** → OK, then extract.

**"the project folder path is N characters long; without Windows long-path support it must be at most 103 characters".** Windows limits a full file path to 260 characters unless long-path support is enabled. The deepest files this package creates lie up to 155 characters below the project folder (measured on the development Mac, Windows layout: installed packages in `.venv\Lib\site-packages\` 136, full-demo runs 139, fast test suite under `demo_outputs\pt\` 146, `verify_installation.cmd --full-tests` 154), so step 4 stops when the folder path is longer than 103 characters and the `LongPathsEnabled` registry setting is off or unreadable. Move the folder to a shorter path such as `C:\Projects\falls_ml_handoff` and run `setup_windows.cmd --recreate-venv` (the virtual environment stores its absolute path). Spaces in the path are supported (`C:\Projects\Falls Research\falls_ml_handoff`). Enabling long-path support needs administrator rights; ask IT if a longer path is unavoidable. "WinError 206" or "filename or extension is too long" during installation has the same cause.

**The folder was moved or renamed.** A virtual environment stores absolute paths. Run `setup_windows.cmd --recreate-venv`.

**Running setup again.** Always safe. It reuses a valid `.venv` and reinstalls only what is needed.

**Demo outputs use too much space.** Run `clean_demo_outputs.cmd --dry-run` to see what would be removed, then `clean_demo_outputs.cmd`.

**Check that the zip was not modified.** `SHA256SUMS.txt` is delivered next to the zip (it cannot be inside the zip it hashes). In CMD, in the folder that contains the zip:

```bat
certutil -hashfile falls_ml_handoff_<VERSION>.zip SHA256
```

Compare the printed hash with the line for the zip in `SHA256SUMS.txt`. Inside the extracted folder, `PACKAGE_MANIFEST.txt` lists the sha256 of every file.

## File map

| Path | Purpose |
|---|---|
| `setup_windows.cmd` | One-time installation and verification |
| `verify_installation.cmd` | Re-check the installation at any time |
| `run_demo.cmd`, `run_full_demo.cmd` | Synthetic demos |
| `prepare_offline_package.cmd` | Download the offline wheel bundle (on a computer with internet) |
| `clean_demo_outputs.cmd` | Remove generated demo outputs |
| `requirements.lock`, `requirements-build.lock` | Exact, hash-pinned dependency versions |
| `README.md` | Project overview and commands |
| `.gitattributes` | Keeps LF line endings for text files and CRLF for `.cmd` files if this folder is later put under git |
| `data\README.md`, `data\example_schema.csv` | Dataset contract for real data later |
| `configs\` | Feature specification, published model, experiment templates (`configs\experiments\fixture\` = synthetic) |
| `configs\meuhedet\` | Meuhedet wide-table column contract (221 columns) and eFalls mapping manifest (Phase 1) |
| `docs\meuhedet\` | Phase-1 report, mapping table, column inventory, datatype contract, work-PC runbook |
| `src\falls_ml\` | Pipeline source code |
| `tests\` | Automated tests |
| `scripts\handoff\`, `scripts\posix\` | Installer helpers (Windows and macOS/Linux) |
| `docs\` | Specification, architecture, artifact schemas, dependencies, research notes |
| `tools\` | Maintenance tools (config and schema generators, package builder) |
| `PACKAGE_MANIFEST.txt`, `VERSION` | File checksums and package version |
