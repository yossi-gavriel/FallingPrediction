# Dependencies — `falls_ml`

Every third-party package is pinned to an exact version and to the sha256 of the wheel file(s) that were verified. The
installer (`setup_windows.cmd`, `scripts/posix/setup.sh`) installs only these files. Nothing is compiled from source.

## Supported Python

| Item | Value |
|---|---|
| Interpreter | CPython **3.13.x**, 64-bit (tested with 3.13.13) |
| Supported runtime | Windows 10/11 x64 (`win_amd64`, lock target `windows-amd64-cp313`) |
| Development / verification | macOS arm64 (lock target `macos-arm64-cp313`) |
| Unsupported | Windows ARM64, 32-bit Python, Python 3.12 or older than 3.11, Linux (no Linux wheel hashes in the lock; regenerate with a Linux target first) |

## Lock files

| File | Content | Installed with |
|---|---|---|
| `requirements-build.lock` | pip, setuptools, wheel (+ `packaging`, required by wheel) | `<venv-python> -m pip install --disable-pip-version-check --no-input --require-hashes --only-binary=:all: -r requirements-build.lock` |
| `requirements.lock` | runtime dependencies (`[project].dependencies`) and the test extra (`pytest`), with their full transitive closure | `<venv-python> -m pip install --disable-pip-version-check --no-input --require-hashes --only-binary=:all: -r requirements.lock` |
| project | `falls_ml` itself, editable (configs are resolved relative to the source tree) | `<venv-python> -m pip install --disable-pip-version-check --no-input --no-deps --no-build-isolation --no-index -e .` |

Offline mode adds `--no-index --find-links offline_packages` to the first two commands.

Format rules (enforced by `scripts/handoff/lockfile.py` and `tests/unit/test_lockfile.py`): one requirement per logical
line, `name==version`, at least one `--hash=sha256:<hex>`, canonical lower-case names, and environment markers only in the
two forms `; sys_platform == "win32"` and `; sys_platform != "win32"`. No index URLs, extras, source distributions or
other options appear in the files.

## Locked packages

"Both" means the package is installed on Windows and macOS; one hash per distinct wheel file is listed (pure-Python
wheels are shared, so they carry one hash; compiled packages carry a `win_amd64` and a `macosx_*_arm64` hash).

### Runtime and test (`requirements.lock`, 32 packages)

| Package | Version | Why | Platform |
|---|---|---|---|
| numpy | 2.4.6 | direct: arrays, random generators | both |
| pandas | 3.0.5 | direct: modelling dataset frames | both |
| scipy | 1.17.1 | direct: fractional polynomials, optimisation, statistics | both |
| scikit-learn | 1.9.1 | direct: LASSO/logistic models, random forest, gradient boosting, ROC/calibration helpers | both |
| statsmodels | 0.15.0 | direct: logistic models, calibration statistics | both |
| matplotlib | 3.11.2 | direct: report plots | both |
| pyyaml | 6.0.3 | direct: configuration and feature specs | both |
| openpyxl | 3.1.5 | direct: Excel (.xlsx) input for the Meuhedet wide-table extract (`meuhedet-explore`, `meuhedet-audit`, `meuhedet-build`) | both |
| pyarrow | 25.0.1 | direct: Parquet datasets and predictions | both |
| pytest | 9.1.1 | direct (test extra): test suite used by the installer and verification | both |
| cloudpickle | 3.1.2 | transitive: joblib | both |
| et-xmlfile | 2.0.0 | transitive: openpyxl | both |
| colorama | 0.4.6 | transitive: pytest (console colours) | **Windows only** (`sys_platform == "win32"`) |
| contourpy | 1.3.3 | transitive: matplotlib | both |
| cycler | 0.12.1 | transitive: matplotlib | both |
| fonttools | 4.65.0 | transitive: matplotlib | both |
| formulaic | 1.2.2 | transitive: statsmodels | both |
| iniconfig | 2.3.0 | transitive: pytest | both |
| interface-meta | 2.0.1 | transitive: formulaic | both |
| joblib | 1.6.0 | transitive: scikit-learn | both |
| kiwisolver | 1.5.1 | transitive: matplotlib | both |
| narwhals | 2.26.0 | transitive: formulaic, scikit-learn | both |
| packaging | 26.3 | transitive: matplotlib, patsy, pytest, statsmodels (also in the build lock, same pin) | both |
| patsy | 1.0.3 | transitive: statsmodels | both |
| pillow | 12.3.0 | transitive: matplotlib | both |
| pluggy | 1.6.0 | transitive: pytest | both |
| pygments | 2.21.0 | transitive: pytest | both |
| pyparsing | 3.3.2 | transitive: matplotlib | both |
| python-dateutil | 2.9.0.post0 | transitive: matplotlib, pandas | both |
| six | 1.17.0 | transitive: python-dateutil | both |
| threadpoolctl | 3.6.0 | transitive: scikit-learn | both |
| typing-extensions | 4.16.0 | transitive: formulaic | both |
| tzdata | 2026.4 | transitive: pandas (IANA time-zone database; Windows has no system copy) | **Windows only** (`sys_platform == "win32"`) |
| wrapt | 2.4.1 | transitive: formulaic | both |

No package is macOS-only, so the `!=` marker is currently unused.

### Build tools (`requirements-build.lock`, 4 packages)

| Package | Version | Why | Platform |
|---|---|---|---|
| pip | 26.2.1 | installer; same version as the verified development environment | both |
| setuptools | 84.0.0 | build backend of `pyproject.toml`; PEP 660 editable install (≥ 69 required) | both |
| wheel | 0.48.0 | listed in `[build-system].requires` | both |
| packaging | 26.3 | transitive: wheel (identical pin and hashes in `requirements.lock`) | both |

XGBoost, LightGBM and SHAP are intentionally not used.

## How the hashes are verified

1. **Generation.** `tools/lock/generate_lock.py` downloads each pinned wheel for both targets with
   `pip download --only-binary=:all: --no-deps` and computes the sha256 of the downloaded files itself.
   The lock is only written after it re-parses with the installer's own parser.
2. **Online install.** `--require-hashes` makes pip refuse any requirement without a hash, any unpinned transitive
   dependency and any file whose sha256 is not listed. `--only-binary=:all:` forbids source builds.
3. **Offline bundle.** `scripts/handoff/prepare_offline.py` downloads with `--require-hashes`, then independently checks
   every file against the lock (name, version, sha256, Windows tags), requires exactly one wheel per applicable
   requirement, and records the sha256 of both lock files and of every wheel in `offline_packages/OFFLINE_MANIFEST.json`.
   `prepare_offline.py --verify-only` repeats these checks on the destination computer.
4. **Offline install.** pip hashes each local file again before installing it (`--require-hashes --no-index --find-links`).
5. **After installation.** The environment validation step (`pip check`, installed versions equal to the lock) and
   `tests/unit/test_lockfile.py` confirm the result.

## Offline mode

For computers without access to a package index:

1. On any computer with internet access (Windows, macOS or Linux, any CPython 3.9+ with pip) run, in the project folder,
   `prepare_offline_package.cmd` on Windows, or `python scripts/handoff/prepare_offline.py --target windows-amd64-cp313`.
   Because pip evaluates environment markers against the computer it runs on, the script first writes a copy of each lock
   evaluated for `sys_platform=win32` and downloads with
   `--platform win_amd64 --python-version 3.13 --implementation cp --abi cp313` (or `--python-version 3.11 --abi cp311` for `--target windows-amd64-cp311`). The result is `offline_packages/`
   (35 wheels, about 133 MB, for the Windows target). Use `--target current` to prepare a bundle for the computer the
   script runs on (CPython 3.11 or 3.13 on Windows x64 or macOS arm64 only).
2. Copy the whole project folder including `offline_packages/` to the target computer.
3. Run `setup_windows.cmd --offline` (or just `setup_windows.cmd`: mode `auto` uses offline mode when
   `offline_packages\OFFLINE_MANIFEST.json` exists).

The bundle is tied to the exact lock files: if either lock changes, its sha256 no longer matches the manifest and the
bundle must be prepared again. `--force` replaces an earlier bundle but deletes only the files listed in its manifest.

In online mode pip uses its normal configuration: `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, `HTTPS_PROXY`, `pip.ini`
(Windows) or `pip.conf`. The entry points and helpers clear only `PIP_USER`, `PIP_TARGET` and `PIP_PREFIX` for their pip processes (these
would install outside `.venv`); `environment_report.txt` records whether each was set (yes/no). An internal mirror must serve files byte-identical to PyPI, otherwise the hash check fails
(this is intended).

## Regenerating the locks

Only needed when a dependency version changes. On the macOS arm64 development machine with internet access:

```bash
# 1. change versions in the development venv (and pyproject.toml minimums if needed), run the full test suite
.venv/bin/python -m pytest -q
# 2. regenerate (downloads wheels for windows-amd64-cp313 and macos-arm64-cp313 into the cache directory)
.venv/bin/python tools/lock/generate_lock.py --cache-dir /tmp/falls_ml_lock_wheel_cache
# 3. confirm idempotency and the format tests
.venv/bin/python tools/lock/generate_lock.py --check --cache-dir /tmp/falls_ml_lock_wheel_cache
.venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_lockfile.py
# 4. prove a clean install in a throwaway venv (commands in tools/lock/README.md), then rebuild any offline bundle
```

Rules applied by the generator:

- versions installed in the development venv are kept unchanged;
- packages not installed there (Windows-only packages, setuptools, wheel) keep their existing lock pin while it satisfies
  every specifier; otherwise, or with `--upgrade`, the newest version with a compatible wheel is chosen;
- Windows requirements are discovered from each wheel's `Requires-Dist`, evaluated for Windows x64 CPython 3.13 and 3.11
  (several patch and OS-release variants, which must agree), recursively and including requested extras;
- a package needed on only one target gets a `sys_platform` marker; shared packages have identical pins in both locks;
- the `Generated:` date changes only when the pinned content changes.
