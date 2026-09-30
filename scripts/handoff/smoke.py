"""Synthetic end-to-end smoke test of an installed falls_ml (needs the project dependencies; run with the venv python).

Usage: <venv python> scripts/handoff/smoke.py --out <dir> [--quick]

Runs the fixture configs efalls_published_scoring and efalls_retrained_lasso_fixed_fp on data/fixtures/synthetic_v1
with reduced settings (bootstrap n 50, or 20 with --quick; stability disabled; permutation repeats 1; reporting on),
then for each run: checks metrics.json, report.md and the model bundle exist; loads the bundle and verifies that
predict_risk on the test rows reproduces predictions_test.parquet risk_served (max abs diff <= 1e-9); re-saves the
loaded pipeline with save_bundle, reloads it and verifies again; scores a CSV of the test rows with
`python -m falls_ml predict` and verifies again. Writes <out>/SMOKE_RESULT.json (paths relative to <out>, "/" separators).
Exit 0 = passed, 1 = failed.

Imports only the standard library at module import time; falls_ml, pandas and numpy are imported inside functions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FIXTURE = "data/fixtures/synthetic_v1"
CONFIGS = ("efalls_published_scoring", "efalls_retrained_lasso_fixed_fp")
RESULT_FILE = "SMOKE_RESULT.json"
REQUIRED_ARTIFACTS = ("metrics.json", "report.md", "predictions_test.parquet", "splits.csv", "RUN_COMPLETE.json", "model/bundle.json")


class SmokeFailure(RuntimeError):
    """A smoke check failed; the message is the user-facing reason."""


def reduced_config(name: str, out: Path, *, bootstrap_n: int) -> Path:
    """Write the fixture config ``name`` with smoke-test settings to <out>/smoke_configs/<name>.yaml."""
    import yaml

    source = common.PROJECT_ROOT / "configs" / "experiments" / "fixture" / f"{name}.yaml"
    if not source.is_file():
        raise SmokeFailure(f"fixture config not found: {common.display_path(source)} (package incomplete?)")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["evaluation"]["bootstrap"]["n"] = bootstrap_n
    raw["evaluation"]["permutation_importance_repeats"] = 1
    raw["analysis"]["stability"]["enabled"] = False
    raw["reporting"]["enabled"] = True
    raw["output"]["runs_dir"] = (out / "runs").resolve().as_posix()  # "/" separators: identical config text on every OS
    # never name this folder "configs": the project root is found by walking up to a folder containing configs/
    target = common.mark_generated(out / "smoke_configs", "smoke-test configs") / f"{name}.yaml"
    header = (f"# {common.SYNTHETIC_BANNER}\n# Smoke-test copy of configs/experiments/fixture/{name}.yaml with reduced settings "
              f"(bootstrap n {bootstrap_n}, stability disabled, permutation repeats 1). Generated; do not edit.\n")
    target.write_text(header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return target


def _keys(df: Any, id_col: str, index_col: str) -> Any:
    import pandas as pd

    return df[id_col].astype(str) + "|" + pd.to_datetime(df[index_col]).dt.strftime("%Y-%m-%d")


def test_rows(run_dir: Path, frame: Any, spec: Any) -> tuple[Any, Any]:
    """(input rows of the test partition, served risks) in the order of predictions_test.parquet."""
    import pandas as pd

    id_col, index_col = spec.identifier_columns[0], spec.index_column
    preds = pd.read_parquet(run_dir / "predictions_test.parquet")
    splits = pd.read_csv(run_dir / "splits.csv", dtype=str)
    test_keys = set(_keys(splits[splits["split"] == "test"], "research_id", "index_date"))
    pred_keys = _keys(preds, "research_id", "index_date")
    if set(pred_keys) != test_keys:
        raise SmokeFailure(f"{run_dir.name}: predictions_test.parquet rows differ from the test rows in splits.csv")
    by_key = frame.assign(_key=_keys(frame, id_col, index_col)).set_index("_key")
    if not by_key.index.is_unique:
        raise SmokeFailure("synthetic fixture has duplicate (research_id, index_date) rows")
    rows = by_key.loc[pred_keys.to_numpy()].reset_index(drop=True)
    return rows, preds["risk_served"].to_numpy(dtype="float64")


def max_abs_diff(a: Any, b: Any) -> float:
    import numpy as np

    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    if a.shape != b.shape:
        raise SmokeFailure(f"prediction count differs: {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _require_equal(label: str, diff: float) -> None:
    if not diff <= common.PREDICTION_TOLERANCE:
        raise SmokeFailure(f"{label}: predictions differ from the in-run test predictions (max abs diff {diff:.3e} > "
                           f"{common.PREDICTION_TOLERANCE:.0e})")


def cli_predict(model_dir: Path, rows: Any, work: Path, log_path: Path) -> Any:
    import pandas as pd

    work.mkdir(parents=True, exist_ok=True)
    csv_in, csv_out = work / "test_rows.csv", work / "predictions.csv"
    rows.to_csv(csv_in, index=False)
    result = common.run_logged([sys.executable, "-m", "falls_ml", "predict", "--model", model_dir, "--input", csv_in, "--out", csv_out],
                               log_path, env=common.subprocess_env())
    if not result.ok:
        raise SmokeFailure(f"`python -m falls_ml predict` failed with exit code {result.returncode}: "
                           + " | ".join(common.tail(result.lines, 5)))
    out = pd.read_csv(csv_out, dtype={"research_id": str})
    if out["research_id"].tolist() != rows["research_id"].astype(str).tolist():
        raise SmokeFailure("CLI predictions are not in input order or research_id values changed")
    return out["risk_12m"].to_numpy(dtype="float64")


def check_run(name: str, run_dir: Path, frame: Any, spec: Any, out: Path, entry: dict[str, Any]) -> None:
    """Run every check for one run, recording each result in ``entry`` as soon as it is known."""
    from falls_ml.bundle import load_bundle, save_bundle
    from falls_ml.config import experiment_config_from_dict
    from falls_ml.inference import predict_risk

    entry["run_dir"] = common.posix_path(run_dir, out)  # JSON paths use "/" on every OS
    missing = [a for a in REQUIRED_ARTIFACTS if not (run_dir / a).exists()]
    if missing:
        raise SmokeFailure(f"{name}: run artifacts missing: {missing}")
    entry["artifacts_ok"] = True
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    served_variant = metrics["served_variant"]
    entry["served_variant"] = served_variant
    entry["test_auroc_synthetic"] = (metrics["performance"]["test"].get(served_variant) or {}).get("auroc", {}).get("estimate")
    rows, served = test_rows(run_dir, frame, spec)
    entry["n_test_rows"] = int(len(rows))

    bundle = load_bundle(run_dir / "model")
    entry["bundle_reload_max_abs_diff"] = max_abs_diff([r.risk_12m for r in predict_risk(rows, bundle)], served)
    _require_equal(f"{name}: reloaded bundle", entry["bundle_reload_max_abs_diff"])

    resaved_dir = common.mark_generated(out / "resaved_bundles", "re-saved model bundles") / name
    meta = bundle.metadata
    save_bundle(resaved_dir, bundle.pipeline, experiment_config=experiment_config_from_dict(meta["config"], name=f"{name} bundle"),
                dataset_manifest=meta["training_dataset"], reference_profile=bundle.reference_profile, model_version=bundle.model_version,
                created_utc=meta["created_utc"], extra_metadata=meta.get("extra_metadata"))
    entry["resaved_bundle"] = common.posix_path(resaved_dir, out)
    resaved = load_bundle(resaved_dir)
    entry["resave_reload_max_abs_diff"] = max_abs_diff([r.risk_12m for r in predict_risk(rows, resaved)], served)
    _require_equal(f"{name}: re-saved bundle", entry["resave_reload_max_abs_diff"])

    cli = cli_predict(run_dir / "model", rows, common.mark_generated(out / "cli_predict", "CLI predictions") / name,
                      common.mark_generated(out / "logs", "smoke-test logs") / f"cli_predict_{name}.log")
    entry["cli_predict_max_abs_diff"] = max_abs_diff(cli, served)
    _require_equal(f"{name}: CLI predict", entry["cli_predict_max_abs_diff"])


CHECK_KEYS = {"artifacts": "artifacts_ok", "bundle_reload_equal": "bundle_reload_max_abs_diff",
              "resave_reload_equal": "resave_reload_max_abs_diff", "cli_predict_equal": "cli_predict_max_abs_diff"}


def summarize_checks(runs: list[dict[str, Any]]) -> dict[str, bool]:
    """A check passes only when every configured run recorded it within tolerance."""
    def passed(entry: dict[str, Any], key: str) -> bool:
        value = entry.get(key)
        return value is True if key == "artifacts_ok" else isinstance(value, float) and value <= common.PREDICTION_TOLERANCE

    complete = [e for e in runs if e.get("config") in CONFIGS]
    return {check: len(complete) == len(CONFIGS) and all(passed(e, key) for e in complete) for check, key in CHECK_KEYS.items()}


def run_smoke(out: Path, result: dict[str, Any], *, quick: bool) -> None:
    import falls_ml
    from falls_ml.data.dataset import ModelingDataset
    from falls_ml.experiment import run_experiment
    from falls_ml.features.spec import load_feature_spec

    result["falls_ml_version"] = falls_ml.__version__
    result["dataset"] = FIXTURE
    fixture = common.PROJECT_ROOT / FIXTURE
    if not (fixture / "manifest.json").is_file():
        raise SmokeFailure(f"synthetic fixture not found: {FIXTURE} (package incomplete?)")
    spec = load_feature_spec(common.PROJECT_ROOT / "configs" / "features" / "efalls_v1.yaml")
    dataset = ModelingDataset.load(fixture, spec)
    bootstrap_n = 20 if quick else 50
    result["settings"] = {"bootstrap_n": bootstrap_n, "stability_enabled": False, "permutation_importance_repeats": 1, "reporting_enabled": True}
    for name in CONFIGS:
        start = time.monotonic()
        entry: dict[str, Any] = {"config": name}
        result["runs"].append(entry)
        print(f"  running {name} (synthetic fixture, bootstrap n {bootstrap_n})...", flush=True)
        config_path = reduced_config(name, out, bootstrap_n=bootstrap_n)
        run = run_experiment(None, dataset, config_path, runs_dir=common.mark_generated(out / "runs", "smoke-test runs"))
        check_run(name, run.run_dir, dataset.frame, spec, out, entry)
        entry["seconds"] = round(time.monotonic() - start, 1)
        worst = max(entry["bundle_reload_max_abs_diff"], entry["resave_reload_max_abs_diff"], entry["cli_predict_max_abs_diff"])
        print(f"  {name}: bundle reload, re-save and CLI predictions identical (max abs diff {worst:.1e})", flush=True)


def main(argv: list[str] | None = None) -> int:
    common.configure_console()
    ap = argparse.ArgumentParser(description="Synthetic end-to-end smoke test (software check only).")
    ap.add_argument("--out", required=True, help="output folder (replaced only if it was created by these scripts)")
    ap.add_argument("--quick", action="store_true", help="smaller bootstrap for a faster check")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = common.PROJECT_ROOT / out
    start = time.monotonic()
    common.print_banner()
    try:
        if common.PROJECT_ROOT / common.DEMO_OUTPUTS in out.resolve().parents:
            common.ensure_demo_root()
        common.prepare_generated_dir(out, "synthetic smoke test")
    except (common.OutputDirError, OSError) as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1
    result: dict[str, Any] = {"ok": False, "reason": None, "synthetic": True, "banner": common.SYNTHETIC_BANNER, "created_utc": common.utc_now(),
                              "tolerance": common.PREDICTION_TOLERANCE, "runs": []}
    try:
        run_smoke(out, result, quick=args.quick)
        result["ok"] = all(summarize_checks(result["runs"]).values())
        if not result["ok"]:
            raise SmokeFailure("not every smoke check was recorded as passed")
    except SmokeFailure as exc:
        result["reason"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - recorded in SMOKE_RESULT.json and re-reported with the traceback
        result["reason"] = f"unexpected {type(exc).__name__}: {exc}"
        traceback.print_exc()
    result["elapsed_seconds"] = round(time.monotonic() - start, 1)
    result["checks"] = summarize_checks(result["runs"])
    (out / RESULT_FILE).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["ok"]:
        print(f"SMOKE TEST PASSED in {common.format_seconds(result['elapsed_seconds'])}: {common.display_path(out / RESULT_FILE)}")
    else:
        print(f"SMOKE TEST FAILED: {result['reason']}\nDetails: {common.display_path(out / RESULT_FILE)}")
    common.print_banner()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
