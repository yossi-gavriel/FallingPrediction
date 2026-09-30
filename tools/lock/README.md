# tools/lock — dependency lock generator

`generate_lock.py` writes `requirements.lock` and `requirements-build.lock` at the project root. It is a developer tool
for the macOS arm64 development machine (CPython 3.13, internet access); the Windows installation never runs it.
Background, package list and offline mode: [`docs/DEPENDENCIES.md`](../../docs/DEPENDENCIES.md).

## Usage

```bash
.venv/bin/python tools/lock/generate_lock.py --cache-dir <wheel-cache-dir>            # write locks if changed
.venv/bin/python tools/lock/generate_lock.py --cache-dir <wheel-cache-dir> --check    # exit 1 if out of date
```

| Option | Meaning |
|---|---|
| `--cache-dir DIR` | wheel download cache, one sub-folder per target (default: system temp `falls_ml_lock_wheel_cache`) |
| `--pins-python PY` | interpreter whose installed versions are kept (default: `.venv` interpreter) |
| `--upgrade` | ignore existing lock pins for packages not installed in `--pins-python` (Windows-only packages, setuptools, wheel) |
| `--check` | compare only; nothing is written |
| `--date YYYY-MM-DD` | `Generated:` date for files whose content changed (default: today, UTC) |

The generator prints a table of every package with its version, marker, the packages that require it on each target and
the number of hashes. It fails (exit 1) when a pinned version has no `win_amd64`/cp313-compatible wheel, when a pin
violates a dependency specifier, or when a marker is ambiguous for the target.

## Verifying a regenerated lock

```bash
BASE=$(sed -n 's/^home = //p' .venv/pyvenv.cfg)/python3.13
$BASE -m venv /tmp/lockcheck_venv
/tmp/lockcheck_venv/bin/python -m pip install --disable-pip-version-check --no-input --require-hashes --only-binary=:all: -r requirements-build.lock
/tmp/lockcheck_venv/bin/python -m pip install --disable-pip-version-check --no-input --require-hashes --only-binary=:all: -r requirements.lock
/tmp/lockcheck_venv/bin/python -m pip check
/tmp/lockcheck_venv/bin/python -c "import numpy, pandas, scipy, sklearn, statsmodels, matplotlib, yaml, pyarrow, pytest"
```

Offline variant: `python scripts/handoff/prepare_offline.py --target current --dest /tmp/offline_mac`, then the same
install commands with `--no-index --find-links /tmp/offline_mac` in a fresh venv.
