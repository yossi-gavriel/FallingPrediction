"""Frozen baselines: an immutable, named record of a completed run (spec D-19; Phase 1 'freeze the eFalls baseline').

``freeze_baseline`` copies nothing patient-level: it records the dataset manifest (hashes, index dates, outcome prevalence),
feature-spec / mapping-manifest / contract hashes, the outcome definition, the model config, the selected lambda, the
coefficient table, the test-set metrics and the test-row hash, and refuses to overwrite an existing baseline of the same name.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from falls_ml.errors import ConfigError, DatasetValidationError

COMPLETE_MARKER = "RUN_COMPLETE.json"
COPIED_FILES = ("config.yaml", "metrics.json", "coefficients.csv", "dataset_manifest.json", "feature_importance.csv", "feature_stability.csv",
                "fp_selection.csv", "cv_results.csv", "unpenalized_refit.csv", "splits.csv", "report.md")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_baseline(run_dir: str | Path, name: str, *, baselines_dir: str | Path = "baselines", note: str = "") -> dict[str, Any]:
    run = Path(run_dir)
    if not (run / COMPLETE_MARKER).exists():
        raise DatasetValidationError(f"{run} is not a completed run ({COMPLETE_MARKER} missing)")
    if not name or any(ch in name for ch in "/\\ "):
        raise ConfigError(f"baseline name {name!r} must be a single token without spaces or path separators")
    out = Path(baselines_dir) / name
    if out.exists():
        raise DatasetValidationError(f"baseline {name} already exists at {out}; baselines are immutable (choose a new name)")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    ds_manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    coef = pd.read_csv(run / "coefficients.csv") if (run / "coefficients.csv").exists() else pd.DataFrame()
    build = (ds_manifest.get("audit") or {}).get("meuhedet_build") or {}
    outcome = {"name": metrics.get("outcome", {}).get("name") or config["dataset"].get("outcome") or "see feature spec",
               **(metrics.get("outcome") or {})}
    record: dict[str, Any] = {
        "baseline_name": name, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note,
        "run_id": metrics["run_id"], "run_dir": str(run), "experiment": metrics["experiment"],
        "dataset": {k: ds_manifest.get(k) for k in ("dataset_version", "mapping_version", "source", "data_sha256", "n_rows", "n_patients",
                                                   "index_date_min", "index_date_max", "outcome_prevalence", "data_freeze_date",
                                                   "scientific_use_allowed", "feature_spec_sha256")},
        "index_date": build.get("index_date") or ds_manifest.get("index_date_min"),
        "feature_spec": metrics.get("feature_set"), "mapping_manifest": build.get("mapping"), "wide_contract": build.get("contract"),
        "outcome": outcome, "model_config": {"model": config["model"], "preprocessing": config["preprocessing"], "validation": config["validation"],
                                             "calibration": config.get("calibration")},
        "lasso": metrics.get("lasso"), "selected_lambda": (metrics.get("lasso") or {}).get("lambda_star"),
        "coefficients": coef.to_dict(orient="records"), "metrics_test": (metrics.get("performance") or {}).get("test"),
        "calibration": metrics.get("calibration"), "split": metrics.get("split"), "efalls_coverage": metrics.get("efalls_coverage"),
        "limitations": metrics.get("limitations"), "warnings": metrics.get("warnings"), "code_version": metrics.get("code_version"),
    }
    out.mkdir(parents=True, exist_ok=False)
    files: dict[str, str] = {}
    for fname in COPIED_FILES:
        src = run / fname
        if src.exists():
            shutil.copyfile(src, out / fname)
            files[fname] = _sha256(out / fname)
    record["files"] = files
    (out / "baseline.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8", newline="\n")
    record["baseline_dir"] = str(out)
    return record
