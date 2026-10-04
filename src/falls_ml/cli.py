"""Command-line interface: ``python -m falls_ml <command>`` (architecture §5)."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from falls_ml.logging_utils import configure_logging

if TYPE_CHECKING:
    from falls_ml.features.spec import FeatureSpec

DEFAULT_FEATURE_SPEC = "configs/features/efalls_v1.yaml"


@dataclass(frozen=True)
class _SpecChoice:
    """Feature spec selection of the dataset commands: --feature-spec/--extension, or --features-from-config."""

    spec_path: str
    extensions: tuple[str, ...]
    full: FeatureSpec
    used: FeatureSpec            # full spec, or the config's declared feature subset
    config_path: str | None


def _choose_spec(a: argparse.Namespace) -> _SpecChoice:
    """Resolve the spec from --feature-spec/--extension or from --features-from-config (the experiment config's
    dataset.feature_spec, extensions and preprocessing.features; efalls_retrained_reduced declarations are checked)."""
    from falls_ml.config import check_feature_declaration, load_experiment_config
    from falls_ml.features.spec import load_feature_spec
    from falls_ml.paths import resolve_path

    if a.features_from_config is None:
        spec_path, extensions = a.feature_spec or DEFAULT_FEATURE_SPEC, tuple(a.extension or ())
        full = load_feature_spec(spec_path, extensions)
        return _SpecChoice(spec_path, extensions, full, full, None)
    if a.feature_spec is not None or a.extension:
        raise SystemExit("--features-from-config takes the feature spec, extensions and features from the experiment config; "
                         "do not combine it with --feature-spec/--extension")
    cfg = load_experiment_config(resolve_path(a.features_from_config))
    anchor = cfg.source_path
    spec_path = str(resolve_path(cfg.dataset.feature_spec, anchor=anchor))
    extensions = tuple(str(resolve_path(e, anchor=anchor)) for e in cfg.dataset.feature_spec_extensions)
    full = load_feature_spec(spec_path, extensions)
    check_feature_declaration(cfg, full)
    used = full if cfg.preprocessing.features is None else full.subset(cfg.preprocessing.features)
    return _SpecChoice(spec_path, extensions, full, used, anchor)


def _spec_summary(full: FeatureSpec, used: FeatureSpec) -> dict[str, Any]:
    return {"name": used.name, "version": used.version, "sha256": used.content_sha256, "is_subset": used.is_subset,
            "root_sha256": full.content_sha256, "n_predictors": len(used.features), "n_root_predictors": len(full.features),
            "absent_predictors": [n for n in full.predictor_names() if n not in set(used.predictor_names())]}


def _cmd_make_fixture(a: argparse.Namespace) -> int:
    from falls_ml.data.synthetic import generate_synthetic_modeling_dataset

    choice = _choose_spec(a)
    manifest = generate_synthetic_modeling_dataset(a.out, n_patients=a.n_patients, seed=a.seed, feature_spec_path=choice.spec_path,
                                                   extension_spec_paths=choice.extensions,
                                                   features=choice.used.predictor_names() if choice.used.is_subset else None)
    print(json.dumps(manifest.to_dict(), indent=2))
    return 0


def _cmd_validate_dataset(a: argparse.Namespace) -> int:
    from falls_ml.data.dataset import ModelingDataset

    choice = _choose_spec(a)
    if choice.config_path is None:
        ds = ModelingDataset.load(a.dataset, choice.full, infer_subset=True)
    else:
        ds = ModelingDataset.load(a.dataset, choice.full, features=choice.used.predictor_names() if choice.used.is_subset else None)
    print(json.dumps({"valid": True, "n_rows": len(ds), "warnings": list(ds.validation_report.warnings),
                      "feature_spec": _spec_summary(choice.full, ds.spec), "features_from_config": choice.config_path,
                      "manifest": ds.manifest.to_dict()}, indent=2))
    return 0


_DATE_ARG = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _iso_date(value: str, flag: str) -> str:
    try:
        if not _DATE_ARG.match(value):
            raise ValueError("expected YYYY-MM-DD")
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise SystemExit(f"{flag} must be a calendar date YYYY-MM-DD, got {value!r} ({exc})") from exc


def _cmd_build_dataset(a: argparse.Namespace) -> int:
    """Validate a flat extract (one row per research_id x index_date) and write an immutable modelling dataset. Reads one
    local file; never connects to a database."""
    from falls_ml.data.dataset import feature_subset_record, sha256_file, write_modeling_dataset

    if a.scientific_use_allowed and not (a.approval_reference or "").strip():
        raise SystemExit("--scientific-use-allowed requires --approval-reference (the governance/ethics approval identifier)")
    if a.approval_reference is not None and not a.scientific_use_allowed:
        raise SystemExit("--approval-reference is only recorded together with --scientific-use-allowed")
    if a.scientific_use_allowed and a.source == "synthetic_fixture":
        raise SystemExit("a synthetic_fixture source can never be allowed for scientific use")
    freeze = _iso_date(a.data_freeze_date, "--data-freeze-date")
    out, source_file = Path(a.out), Path(a.input)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"--out {out} exists and is not an empty directory; datasets are immutable, choose a new directory")
    if not source_file.is_file():
        raise SystemExit(f"--input {source_file} does not exist")
    choice = _choose_spec(a)
    full, spec, config_path = choice.full, choice.used, choice.config_path
    df = _read_table(str(source_file), spec)
    audit: dict[str, Any] = {"build_dataset": {
        "input_file_name": source_file.name, "input_sha256": sha256_file(source_file), "input_rows": len(df),
        "feature_spec_sha256": spec.content_sha256, "features_from_config": config_path,
        "scientific_use_allowed": bool(a.scientific_use_allowed),
        "scientific_use_approval_reference": a.approval_reference.strip() if a.scientific_use_allowed else None,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "database_connections": "none"}}
    if spec.is_subset:
        audit["feature_spec_subset"] = feature_subset_record(full, spec)
    manifest = write_modeling_dataset(df, out, spec, dataset_version=a.dataset_version, mapping_version=a.mapping_version, source=a.source,
                                      generator="falls_ml build-dataset", scientific_use_allowed=bool(a.scientific_use_allowed),
                                      notes=a.notes or "", audit=audit, data_freeze_date=freeze)
    print(json.dumps({"dataset": str(out), "feature_spec": _spec_summary(full, spec), "manifest": manifest.to_dict()}, indent=2, default=str))
    return 0


def _cmd_train(a: argparse.Namespace) -> int:
    from falls_ml.experiment import run_experiment

    result = run_experiment(a.model, a.dataset, a.config, runs_dir=a.runs_dir, allow_test_reevaluation=a.allow_test_reevaluation)
    report = result.run_dir / "report.html"
    print(json.dumps({"run_id": result.run_id, "run_dir": str(result.run_dir),
                      "report": str(report) if report.exists() else None}, indent=2))
    return 0


def _cmd_evaluate(a: argparse.Namespace) -> int:
    from falls_ml.reporting.report import render_run_report

    md, html = render_run_report(Path(a.run_dir))
    print(json.dumps({"report_md": str(md), "report_html": str(html)}, indent=2))
    return 0


def _cmd_reproduce(a: argparse.Namespace) -> int:
    from falls_ml.experiment import reproduce

    report = reproduce(a.run_dir, dataset=a.dataset, runs_dir=a.runs_dir)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["identical"] else 2


def _cmd_compare(a: argparse.Namespace) -> int:
    from falls_ml.reporting.best_features import build_best_features
    from falls_ml.reporting.compare import build_comparison

    table = build_comparison(Path(a.runs_dir), Path(a.out), n_bootstrap=a.n_bootstrap)
    build_best_features(Path(a.runs_dir), Path(a.out), ablation_dir=Path(a.ablation_dir) if a.ablation_dir else None)
    print(table.to_string(max_rows=50))
    return 0


def _cmd_ablation(a: argparse.Namespace) -> int:
    from falls_ml.experiment import run_ablation

    table = run_ablation(a.config, a.dataset, runs_dir=a.runs_dir, out_dir=a.out)
    print(table.to_string())
    return 0


_DATE_COLUMNS = ("predictor_max_record_date", "outcome_first_event_date")


def spec_csv_dtypes(spec: FeatureSpec, header: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """(string dtypes, date columns) for the CSV columns in ``header``, from the feature spec: identifiers, categorical
    predictors and string-typed provenance/metadata columns stay strings; index, provenance and metadata dates are parsed."""
    cols = set(header)
    declared = {**spec.provenance_columns, **spec.metadata_columns}
    dtypes = {c: "string" for c in spec.identifier_columns if c in cols}
    dtypes.update({f.name: "string" for f in spec.features if f.is_categorical and f.name in cols})
    dtypes.update({c: "string" for c, d in declared.items() if c in cols and str((d or {}).get("dtype", "")).startswith("string")})
    dates = [c for c in (spec.index_column, *_DATE_COLUMNS) if c in cols]
    dates += [c for c, d in declared.items() if c in cols and c not in dates and str((d or {}).get("dtype", "")).startswith("date")]
    return dtypes, dates


def _read_table(path: str, spec: FeatureSpec) -> pd.DataFrame:
    """Read a .parquet or .csv table (extension case-insensitive, e.g. EXTRACT.CSV). CSV must be UTF-8 (a BOM is accepted);
    other encodings fail with guidance instead of being guessed. CSV types come from the feature spec (:func:`spec_csv_dtypes`):
    identifiers and categorical predictors stay strings (no lost leading zeros, empty cells are missing, never the text 'nan');
    date columns are parsed."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix != ".csv":
        raise SystemExit(f"unsupported input format: {p.suffix} (use .parquet or .csv)")
    try:
        header = pd.read_csv(p, nrows=0, encoding="utf-8").columns
        dtypes, dates = spec_csv_dtypes(spec, list(header))
        return pd.read_csv(p, dtype=dtypes, parse_dates=dates, keep_default_na=False, na_values=[""], encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"{p} is not UTF-8 encoded ({exc.reason} at byte {exc.start}); save it as 'CSV UTF-8 (Comma delimited)' "
                         "in Excel or with UTF-8 encoding in SQL Server Management Studio, then retry") from exc


def _check_out_suffix(out: str) -> None:
    if Path(out).suffix.lower() not in {".csv", ".parquet"}:
        raise SystemExit(f"--out must end in .csv or .parquet, got {out!r}")


def _cmd_predict(a: argparse.Namespace) -> int:
    from dataclasses import asdict

    from falls_ml.bundle import load_bundle
    from falls_ml.inference import predict_risk

    _check_out_suffix(a.out)
    bundle = load_bundle(Path(a.model))
    results = predict_risk(_read_table(a.input, bundle.feature_spec), bundle, include_research_id=not a.no_ids)
    out = pd.DataFrame([{**asdict(r), "top_contributing_features": json.dumps(list(r.top_contributing_features)),
                         "data_quality_flags": ";".join(r.data_quality_flags)} for r in results])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.out.lower().endswith(".parquet"):
        out.to_parquet(a.out, index=False)
    else:
        out.to_csv(a.out, index=False, encoding="utf-8", lineterminator="\n")
    print(json.dumps({"n_predictions": len(out), "out": a.out}, indent=2))
    return 0


def _cmd_monitor(a: argparse.Namespace) -> int:
    from falls_ml.bundle import load_bundle
    from falls_ml.monitoring import drift_report, write_drift_report

    bundle = load_bundle(Path(a.model))
    df = _read_table(a.input, bundle.feature_spec)
    outcome = bundle.feature_spec.outcome.name
    if a.labels and outcome not in df.columns:
        raise SystemExit(f"--labels given but the input has no outcome column {outcome!r}")
    labels = df[outcome].to_numpy() if a.labels else None
    report = drift_report(bundle.reference_profile, df, bundle.feature_spec, pipeline=bundle.pipeline, labels=labels)
    paths = write_drift_report(report, Path(a.out))
    print(json.dumps({"overall_status": report.overall_status, "recommendation": report.recommendation,
                      "files": [str(p) for p in paths]}, indent=2))
    return 0


# ---------------------------------------------------------------------------- Meuhedet wide table (Phase 1)
def _cmd_meuhedet_make_fixture(a: argparse.Namespace) -> int:
    from falls_ml.data.meuhedet_synthetic import write_synthetic_wide_extract

    path = write_synthetic_wide_extract(a.out, n_rows=a.n_rows, seed=a.seed, index_date=a.index_date, n_index_day_falls=a.index_day_falls)
    print(json.dumps({"extract": str(path), "n_rows": a.n_rows, "seed": a.seed, "index_date": a.index_date,
                      "label": "SYNTHETIC wide-table extract - NOT real Meuhedet data"}, indent=2))
    return 0


def _cmd_meuhedet_audit(a: argparse.Namespace) -> int:
    """Aggregate, non-identifying audit of a wide-table extract (run inside the approved environment; no database access)."""
    from falls_ml.data.meuhedet_audit import audit_wide_extract, write_audit
    from falls_ml.data.meuhedet_wide import load_wide_contract, load_wide_mapping, read_wide_extract_report
    from falls_ml.features.spec import load_feature_spec

    mapping = load_wide_mapping(a.mapping)
    contract = load_wide_contract(a.contract or mapping.contract_path)
    spec = load_feature_spec(mapping.exploratory_spec_path)
    index_date = _iso_date(a.index_date, "--index-date") if a.index_date else None
    read = read_wide_extract_report(a.input, contract, encoding=a.encoding, sep=a.sep, sheet=a.sheet)
    rep = audit_wide_extract(read.frame, contract, mapping, spec, index_date=index_date, min_cell=a.min_cell, wrong_type=read.wrong_type,
                             wrong_type_examples=read.wrong_type_examples, date_formats=read.date_formats)
    rep["input"] = read.to_dict()
    j, m = write_audit(rep, a.out)
    cov, lk, oc = rep["efalls_coverage"], rep["leakage_audit"], rep["outcomes"]["efalls_365d_outcome"]
    print(json.dumps({"audit_json": str(j), "audit_md": str(m), "rows": rep["overview"]["row_count"], "patients": rep["overview"]["unique_patient_count"],
                      "eligible_on_index_date": rep["overview"].get("eligible_on_selected_index_date"),
                      "wrong_type_total": rep["datatype_audit"]["wrong_type_total"], "leakage_result": lk["result"],
                      "efalls_coverage": f"{cov['n_available']}/{cov['n_expected']} ({cov['coverage_pct']}%)",
                      "efalls_365d_outcome": oc["verdict"], "min_cell_suppression": a.min_cell}, indent=2, default=str))
    return 0


def _cmd_meuhedet_build(a: argparse.Namespace) -> int:
    """Wide-table extract -> canonical eFalls-mapped modelling dataset (immutable; validated; no database access)."""
    from falls_ml.data.meuhedet_wide import build_meuhedet_dataset

    if a.scientific_use_allowed and not (a.approval_reference or "").strip():
        raise SystemExit("--scientific-use-allowed requires --approval-reference (the governance/ethics approval identifier)")
    if a.approval_reference is not None and not a.scientific_use_allowed:
        raise SystemExit("--approval-reference is only recorded together with --scientific-use-allowed")
    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"--out {out} exists and is not an empty directory; datasets are immutable, choose a new directory")
    manifest, report = build_meuhedet_dataset(a.input, out, index_date=_iso_date(a.index_date, "--index-date"), dataset_version=a.dataset_version,
                                              data_freeze_date=_iso_date(a.data_freeze_date, "--data-freeze-date"), mapping_path=a.mapping,
                                              contract_path=a.contract, scientific_use_allowed=bool(a.scientific_use_allowed),
                                              approval_reference=a.approval_reference, notes=a.notes or "", index_day_records=a.index_day_records,
                                              encoding=a.encoding, sep=a.sep, sheet=a.sheet, feature_set=a.feature_set, id_pepper=a.id_pepper)
    summary = {"dataset": str(out), "manifest": manifest.to_dict(), "build_report": report.to_dict(),
               "label": "Meuhedet 180-day exploratory outcome - NOT eFalls reproduction; " + report.coverage.get("label", "")}
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _cmd_meuhedet_explore(a: argparse.Namespace) -> int:
    """One command: local Excel/CSV/Parquet wide extract -> audits -> readiness (READY TO TRAIN / STOPPED) -> STRICT + EXTENDED
    exploratory LASSO runs (one snapshot, or every usable snapshot with a temporal design) -> reports."""
    from falls_ml.errors import ConfigError, DatasetValidationError
    from falls_ml.meuhedet_explore import explore

    try:
        res = explore(a.input, out_dir=a.out, index_date=_iso_date(a.index_date, "--index-date") if a.index_date else None,
                      all_index_dates=bool(a.all_index_dates), audit_only=bool(a.audit_only), sheet=a.sheet,
                      data_freeze_date=_iso_date(a.data_freeze_date, "--data-freeze-date") if a.data_freeze_date else None,
                      dataset_version=a.dataset_version, id_pepper=a.id_pepper, id_pepper_file=a.id_pepper_file, fast=not a.full,
                      encoding=a.encoding, sep=a.sep, mapping_path=a.mapping, index_day_records=a.index_day_records, min_cell=a.min_cell,
                      max_censored_share=a.max_censored_share, patient_disjoint=not a.no_patient_disjoint, model_report=a.model_report)
    except (DatasetValidationError, ConfigError) as exc:
        print("STOPPED - the input did not pass the checks; nothing was trained.", file=sys.stderr)
        print("  " + str(exc).replace("\n", "\n  "), file=sys.stderr)
        return 2
    print(res.summary_text)
    print(json.dumps({"readiness": res.readiness.get("verdict"), "out_dir": str(res.out_dir), "readiness_report": str(res.out_dir / "READINESS.md"),
                      "summary": str(res.out_dir / "SUMMARY.md") if res.runs else None, "runs": {k: str(v) for k, v in res.runs.items()},
                      "patient_disjoint_runs": {k: str(v) for k, v in res.sensitivity_runs.items()},
                      "model_reports": res.model_report or None}, indent=2, ensure_ascii=False))
    return 0


def _cmd_meuhedet_sensitivity(a: argparse.Namespace) -> int:
    """SAFE-ALL-ROWS sensitivity + targeted ablations next to a completed meuhedet-explore run (the reference is never written)."""
    from falls_ml.errors import ConfigError, DatasetValidationError
    from falls_ml.meuhedet_sensitivity import sensitivity

    try:
        res = sensitivity(a.input, a.reference, out_dir=a.out, id_pepper=a.id_pepper, id_pepper_file=a.id_pepper_file, sheet=a.sheet, encoding=a.encoding,
                          sep=a.sep, mapping_path=a.mapping, data_freeze_date=_iso_date(a.data_freeze_date, "--data-freeze-date") if a.data_freeze_date else None,
                          min_cell=a.min_cell, run_ablations=not a.no_ablations, run_safe=not a.no_safe_all_rows, model_report=a.model_report)
    except (DatasetValidationError, ConfigError) as exc:
        print("STOPPED - the sensitivity package could not run; nothing was trained.", file=sys.stderr)
        print("  " + str(exc).replace("\n", "\n  "), file=sys.stderr)
        return 2
    print((res.out_dir / "SENSITIVITY_README.md").read_text(encoding="utf-8"))
    print(json.dumps({"out_dir": str(res.out_dir), "reference_unchanged": res.integrity.get("unchanged"), "report": str(res.out_dir / "REAL_DATA_EXPLORATORY_REPORT.md"),
                      "comparison": str(res.out_dir / "SENSITIVITY_COMPARISON.md"), "runs": {k: str(v) for k, v in res.runs.items()}, "skipped": res.skipped,
                      "model_reports": res.model_report or None}, indent=2, ensure_ascii=False))
    return 0


def _cmd_meuhedet_d00(a: argparse.Namespace) -> int:
    """D-00 sensitivity framework next to a completed meuhedet-explore run: dependency graph, provenance-derived feature sets, the frozen
    sensitivity matrix + pre-specified ablations, risk concentration, learning-curve interpretation, LASSO warnings audit, reports."""
    from falls_ml.d00.runner import run_d00
    from falls_ml.errors import FallsMLError

    try:
        res = run_d00(a.input, a.reference, out_dir=a.out, eda_dir=a.eda, reports_dir=a.reports, id_pepper_file=a.id_pepper_file, config_path=a.config,
                      mapping_path=a.mapping, dependency_only=a.dependency_only, learning_curve=a.learning_curve, resampling=a.resampling,
                      secondary=False if a.no_secondary_standard else None, min_cell=a.min_cell, n_boot=a.n_boot,
                      management_report=not a.no_management_report, sheet=a.sheet, encoding=a.encoding, sep=a.sep, resume=a.resume)
    except FallsMLError as exc:   # every typed stop (validation, leakage, config, >10% failed bootstrap replicates): no traceback needed
        print("STOPPED - the D-00 analysis could not complete; do not share the output folder.", file=sys.stderr)
        print(f"  {type(exc).__name__}: " + str(exc).replace("\n", "\n  "), file=sys.stderr)
        if not a.dependency_only and (Path(a.out) / "share" / "ANALYSIS_PLAN.json").is_file():
            print("  The completed runs in --out are kept. After the cause is fixed, continue with the same command plus --resume "
                  "(completed runs are reused, never refitted).", file=sys.stderr)
        return 2
    print((res.share_dir / "README_D00.md").read_text(encoding="utf-8"))
    files = ({"dependency": str(res.share_dir / "D00_FEATURE_DEPENDENCY.md"), "plan": str(res.share_dir / "ANALYSIS_PLAN.json")} if a.dependency_only else
             {"report": str(res.share_dir / "D00_SENSITIVITY_REPORT.html"), "summary_he": str(res.share_dir / "D00_SENSITIVITY_SUMMARY_HE.md"),
              "management_report": res.management or None})
    print(json.dumps({"out_dir": str(res.out_dir), "SEND_BACK_ONLY": str(res.share_dir), "plan_sha256": res.plan.get("plan_sha256"),
                      "protected_folders_unchanged": res.integrity.get("all_unchanged"), "privacy_scan_passed": res.privacy.get("passed"),
                      **files, "seconds": res.seconds}, indent=2, ensure_ascii=False))
    return 0


def _cmd_meuhedet_phase2_status(a: argparse.Namespace) -> int:
    from falls_ml.phase2.status import phase2_status

    return phase2_status(a.out)


def _cmd_meuhedet_phase2(a: argparse.Namespace) -> int:
    """Phase 2: column registry, engineered features, TRAIN-only screening, nested grouped CV of LASSO / elastic net / XGBoost, frozen
    selection, one-time VALIDATION scoring, stability / ablation / SHAP, consensus, reports and the share package. Crash-safe: --resume."""
    from falls_ml.errors import FallsMLError
    from falls_ml.phase2.runner import run_phase2
    from falls_ml.phase2.state import STOP_HARD, Phase2Stop

    if a.preflight:
        from falls_ml.phase2.preflight import run_preflight

        if a.resume or a.accept_gate or a.accept_code_change:
            print("--preflight checks a NEW run; it cannot be combined with --resume / --accept-gate / --accept-code-change", file=sys.stderr)
            return 2
        return run_preflight(a.input, a.reference, out_dir=a.out, protect=a.protect or [], id_pepper_file=a.id_pepper_file, config_path=a.config,
                             allow_synced_folder=a.allow_synced_folder, allow_unfrozen_config=a.allow_unfrozen_config)
    gates: dict[str, str] = {}
    for g in a.accept_gate or []:
        if not a.reason:
            print("--accept-gate needs --reason \"<why the evidence was judged acceptable>\" (it is recorded)", file=sys.stderr)
            return 2
        if not a.resume:
            print("--accept-gate is only valid with --resume (it continues a run that stopped at that gate)", file=sys.stderr)
            return 2
        gates[g] = a.reason
    if a.accept_code_change and not a.resume:
        print("--accept-code-change is only valid with --resume", file=sys.stderr)
        return 2
    try:
        res = run_phase2(a.input, a.reference, out_dir=a.out, protect=a.protect or [], id_pepper_file=a.id_pepper_file, resume=a.resume,
                         accept_gates=gates, accept_code_change=a.accept_code_change, allow_synced_folder=a.allow_synced_folder, config_path=a.config,
                         allow_unfrozen_config=a.allow_unfrozen_config)
    except Phase2Stop as exc:
        kind = "INVESTIGATION STOP" if exc.kind != STOP_HARD else "STOPPED"
        print(f"{kind} [{exc.gate}] - {exc.message}", file=sys.stderr)
        for d in exc.details:
            print(f"  - {d}", file=sys.stderr)
        if exc.kind != STOP_HARD:
            print(f"  Review {Path(a.out) / ('INVESTIGATION_' + exc.gate + '.md')}; to continue with a recorded justification add: "
                  f"--resume --accept-gate {exc.gate} --reason \"...\"", file=sys.stderr)
        elif (Path(a.out) / "PHASE2_PLAN.json").is_file():
            print("  Completed work is kept. After fixing the cause, run the same command with --resume (nothing completed is recomputed).", file=sys.stderr)
        print("  Do not share the output folder until the run completes (share\\ appears only after the privacy scan passes).", file=sys.stderr)
        return 2
    except FallsMLError as exc:
        print(f"STOPPED - {type(exc).__name__}: " + str(exc).replace("\n", "\n  "), file=sys.stderr)
        print("  Completed work is kept; fix the cause and run the same command with --resume.", file=sys.stderr)
        return 2
    print(json.dumps({"status": res["status"], "out_dir": res["out"], "SEND_BACK_ONLY": str(Path(res["out"]) / "share"), "attempt": res["attempt"],
                      "plan_sha256": res["plan_sha256"]}, indent=2, ensure_ascii=False))
    return 0


def _cmd_meuhedet_phase3_status(a: argparse.Namespace) -> int:
    from falls_ml.phase3.status import phase3_status

    return phase3_status(a.out)


def _cmd_meuhedet_phase3(a: argparse.Namespace) -> int:
    """Phase 3: row-level historical recovery -> scientific GO / NO-GO -> (GO) extended modelling with bounds -> reports -> share."""
    from falls_ml.errors import FallsMLError
    from falls_ml.phase2.state import STOP_HARD, Phase2Stop

    if a.preflight:
        from falls_ml.phase3.preflight import run_preflight

        if a.resume or a.accept_gate or a.accept_code_change:
            print("--preflight checks a NEW run; it cannot be combined with --resume / --accept-gate / --accept-code-change", file=sys.stderr)
            return 2
        return run_preflight(a.input, a.reference, out_dir=a.out, protect=a.protect or [], id_pepper_file=a.id_pepper_file, phase2_out=a.phase2_out,
                             config_path=a.config, allow_synced_folder=a.allow_synced_folder, allow_unfrozen_config=a.allow_unfrozen_config)
    from falls_ml.phase3.runner import run_phase3

    gates: dict[str, str] = {}
    for g in a.accept_gate or []:
        if not a.reason or not a.resume:
            print("--accept-gate needs --reason \"<why>\" and is only valid with --resume (it is recorded)", file=sys.stderr)
            return 2
        gates[g] = a.reason
    if a.accept_code_change and not a.resume:
        print("--accept-code-change is only valid with --resume", file=sys.stderr)
        return 2
    try:
        res = run_phase3(a.input, a.reference, out_dir=a.out, protect=a.protect or [], id_pepper_file=a.id_pepper_file, phase2_out=a.phase2_out,
                         resume=a.resume, accept_gates=gates, accept_code_change=a.accept_code_change, allow_synced_folder=a.allow_synced_folder,
                         config_path=a.config, allow_unfrozen_config=a.allow_unfrozen_config)
    except Phase2Stop as exc:
        kind = "INVESTIGATION STOP" if exc.kind != STOP_HARD else "STOPPED"
        print(f"{kind} [{exc.gate}] - {exc.message}", file=sys.stderr)
        for d in exc.details:
            print(f"  - {d}", file=sys.stderr)
        if exc.kind != STOP_HARD:
            print(f"  Review {Path(a.out) / ('INVESTIGATION_' + exc.gate + '.md')}; to continue with a recorded justification add: "
                  f"--resume --accept-gate {exc.gate} --reason \"...\"", file=sys.stderr)
        elif (Path(a.out) / "PHASE3_PLAN.json").is_file():
            print("  Completed work is kept. After fixing the cause, run the same command with --resume (nothing completed is recomputed).", file=sys.stderr)
        print("  Do not share the output folder until the run completes (share\\ appears only after the privacy scan passes).", file=sys.stderr)
        return 2
    except FallsMLError as exc:
        print(f"STOPPED - {type(exc).__name__}: " + str(exc).replace("\n", "\n  "), file=sys.stderr)
        print("  Completed work is kept; fix the cause and run the same command with --resume.", file=sys.stderr)
        return 2
    print(json.dumps({"status": res["status"], "decision": res.get("decision"), "out_dir": res["out"], "SEND_BACK_ONLY": str(Path(res["out"]) / "share"),
                      "attempt": res["attempt"], "plan_sha256": res["plan_sha256"]}, indent=2, ensure_ascii=False))
    return 0


def _phase4_stop(exc: Exception) -> int:
    from falls_ml.phase2.state import Phase2Stop

    if isinstance(exc, Phase2Stop):
        print(f"STOPPED [{exc.gate}] - {exc.message}", file=sys.stderr)
        for d in exc.details:
            print(f"  - {d}", file=sys.stderr)
    else:
        print(f"STOPPED - {type(exc).__name__}: " + str(exc).replace("\n", "\n  "), file=sys.stderr)
    print("  The outcomes stay sealed unless meuhedet-phase4-evaluate verified the frozen predictions. Send back only <out>\\share.", file=sys.stderr)
    return 2


def _cmd_meuhedet_phase4_preflight(a: argparse.Namespace) -> int:
    from falls_ml.phase4.preflight import run_preflight

    return run_preflight(a.input_2026, a.phase3_out, out_dir=a.out, phase3_config=a.phase3_config, config_path=a.config,
                         allow_unfrozen_phase3=a.allow_unfrozen_phase3, expected_definition_version=a.expected_definition_version,
                         allow_synced_folder=a.allow_synced_folder)


def _cmd_meuhedet_phase4_score(a: argparse.Namespace) -> int:
    from falls_ml.errors import FallsMLError
    from falls_ml.phase4.score import run_score

    try:
        res = run_score(a.input_2026, a.input_2025, a.phase3_out, out_dir=a.out, phase3_config=a.phase3_config, config_path=a.config,
                        allow_unfrozen_phase3=a.allow_unfrozen_phase3, expected_definition_version=a.expected_definition_version)
    except FallsMLError as exc:
        return _phase4_stop(exc)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def _cmd_meuhedet_phase4_evaluate(a: argparse.Namespace) -> int:
    from falls_ml.errors import FallsMLError
    from falls_ml.phase4.evaluate import run_evaluate

    try:
        res = run_evaluate(a.input_2026, out_dir=a.out, config_path=a.config, phase3_config=a.phase3_config, accept_code_change=a.accept_code_change)
    except FallsMLError as exc:
        return _phase4_stop(exc)
    print(json.dumps({**res, "SEND_BACK_ONLY": str(Path(a.out) / "share")}, indent=2, ensure_ascii=False))
    return 0


def _cmd_meuhedet_phase4_status(a: argparse.Namespace) -> int:
    from falls_ml.phase4.status import phase4_status

    return phase4_status(a.out)


def _phase4_common(p: argparse.ArgumentParser, *, score: bool = False) -> None:
    p.add_argument("--input-2026", required=True, help="the 2026 extract (Index_Date 2026-01-01; .csv with NULL literals or .parquet)")
    p.add_argument("--out", required=True, help="NEW local folder for Phase 4 (separate from every Phase 2 / Phase 3 folder; not OneDrive)")
    p.add_argument("--config", default="configs/meuhedet/phase4.yaml", help="Phase 4 settings (pre-declared; hashed into every manifest)")
    p.add_argument("--phase3-config", default="configs/meuhedet/phase3.yaml", help="the frozen Phase 3 settings (verified against the Phase 3 run)")
    if score:
        p.add_argument("--phase3-out", required=True, help="the FINISHED Phase 3 output folder (read-only: every file verified, nothing written)")
        p.add_argument("--allow-unfrozen-phase3", action="store_true", help="tests only: accept a Phase 3 run without the frozen production configuration")
        p.add_argument("--expected-definition-version", help="the Definition_Version value the 2026 V21 extract must carry (mismatch = STOP)")


def _add_phase4_parsers(sub: Any) -> None:
    p = sub.add_parser("meuhedet-phase4-preflight", help="Phase 4 (temporal validation 2026): read-only predictor / schema / timing audit of the 2026 "
                                                         "extract against the frozen Phase 3 models; outcomes are never loaded. Last line: SAFE TO SCORE BLIND "
                                                         "or STOP - TEMPORAL VALIDATION NOT DEFENSIBLE")
    _phase4_common(p, score=True)
    p.add_argument("--allow-synced-folder", action="store_true", help="allow --out inside OneDrive / a synced folder (not recommended)")
    p.set_defaults(func=_cmd_meuhedet_phase4_preflight)
    p = sub.add_parser("meuhedet-phase4-score", help="Phase 4: BLIND scoring with the frozen 2025 models; predictions, models, feature list and input "
                                                     "schema are hashed and frozen before any outcome is opened (write-once)")
    _phase4_common(p, score=True)
    p.add_argument("--input-2025", required=True, help="the 2025 extract Phase 3 used (member ids + 2025 eligibility only, for the patient overlap)")
    p.set_defaults(func=_cmd_meuhedet_phase4_score)
    p = sub.add_parser("meuhedet-phase4-evaluate", help="Phase 4: verifies every frozen hash, then opens the 2026 outcomes, checks the 2026 outcome "
                                                        "contract (stops before any metric if it fails) and reports the temporal validation")
    _phase4_common(p)
    p.add_argument("--accept-code-change", help="evaluate although the falls_ml code differs from the scoring code (reason recorded)")
    p.set_defaults(func=_cmd_meuhedet_phase4_evaluate)
    p = sub.add_parser("meuhedet-phase4-status", help="READ-ONLY progress of a Phase 4 folder")
    p.add_argument("--out", required=True, help="the Phase 4 output folder")
    p.set_defaults(func=_cmd_meuhedet_phase4_status)


def _phase5_stop(exc: Exception) -> int:
    from falls_ml.phase2.state import Phase2Stop

    if isinstance(exc, Phase2Stop):
        print(f"STOPPED [{exc.gate}] - {exc.message}", file=sys.stderr)
        for d in exc.details:
            print(f"  - {d}", file=sys.stderr)
    else:
        print(f"STOPPED - {type(exc).__name__}: " + str(exc).replace("\n", "\n  "), file=sys.stderr)
    print("  Every finished unit is kept. Fix the cause, then run the same command with --resume. Send back only <out>\\share "
          "(or <out>\\preflight for a preflight stop).", file=sys.stderr)
    return 2


def _cmd_meuhedet_phase5(a: argparse.Namespace) -> int:
    from falls_ml.errors import FallsMLError

    if a.status:
        from falls_ml.phase5.status import phase5_status

        r = phase5_status(a.out)
        print(r["text"])
        return 0
    if a.estimate:
        from falls_ml.phase5.config import load_phase5_config
        from falls_ml.phase5.estimate import estimate, estimate_text
        from falls_ml.phase5.resources import default_jobs, limit_threads

        limit_threads()
        cfg = load_phase5_config(a.config, mode=a.mode, v21_catalogue=a.v21_catalogue)
        try:
            r = estimate(Path(a.input) if a.input else None, Path(a.out), cfg, int(a.jobs) if a.jobs else default_jobs(cfg))
        except (FallsMLError, ValueError) as exc:
            return _phase5_stop(exc)
        print(estimate_text(r))
        return 0
    from falls_ml.phase5.runner import run_phase5

    try:
        r = run_phase5(a.input, a.out, mode=a.mode, device=a.device, jobs=a.jobs, resume=a.resume, preflight_only=a.preflight_only,
                       report_only=a.report_only, accept_code_change=a.accept_code_change, allow_synced_folder=a.allow_synced_folder,
                       config_path=a.config, v21_catalogue=a.v21_catalogue)
    except FallsMLError as exc:
        return _phase5_stop(exc)
    print(json.dumps({k: v for k, v in r.items() if k != "report"}, indent=2, ensure_ascii=False))
    if r["status"].startswith(("COMPLETE", "REPORT_COMPLETE")):
        print(f"SEND BACK ONLY: {Path(a.out) / 'share'}")
    elif r["status"] == "PREFLIGHT_COMPLETE":
        print(f"preflight aggregate outputs: {Path(a.out) / 'preflight'}")
    return int(r["exit_code"])


def _cmd_meuhedet_phase5_synthetic(a: argparse.Namespace) -> int:
    from falls_ml.errors import FallsMLError
    from falls_ml.phase5.runner import run_phase5
    from falls_ml.phase5.synthetic import make_v21, write_v21_csv

    out = Path(a.out)
    src_dir = out.parent / (out.name + "_synthetic_input")
    src_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / f"synthetic_v21_{a.scenario}.csv"
    if not src.is_file():
        df, _ = make_v21(int(a.rows), scenario=a.scenario, seed=int(a.seed))
        write_v21_csv(df, src)
    ov = {"eligibility": {"min_known_observed_rows": 20}}
    try:
        r = run_phase5(src, out, mode=a.mode, device=a.device, jobs=a.jobs, resume=True, overrides=ov, synthetic=True,
                       accept_code_change=a.accept_code_change)
    except FallsMLError as exc:
        return _phase5_stop(exc)
    print(json.dumps({k: v for k, v in r.items() if k != "report"}, indent=2, ensure_ascii=False))
    print(f"SYNTHETIC smoke run (software test only): {out / 'share'}")
    return int(r["exit_code"])


def _add_phase5_parsers(sub: Any) -> None:
    p = sub.add_parser("meuhedet-phase5", help="Phase 5 (2026 redevelopment + incremental value of the new V21 predictors): ONE resumable command - "
                                               "preflight, nested CV of LASSO / elastic net / XGBoost on identical folds for OLD vs OLD+NEW_SAFE at >= 70% "
                                               "sensitivity, domains, ablations, explanation, stability, aggregate-only share/. Also --preflight-only, "
                                               "--estimate, --status, --report-only")
    p.add_argument("--input", help="the 2026 V21 extract (Index_Date 2026-01-01; .csv with NULL literals or .parquet)")
    p.add_argument("--out", required=True, help="NEW local folder for Phase 5 (never a Phase 2 / 3 / 4 folder; not OneDrive)")
    p.add_argument("--mode", choices=["quick", "overnight"], default="overnight", help="tuning budget (quick = daytime check; overnight = the real run)")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="XGBoost device; auto = GPU only if it passes a smoke test, else CPU")
    p.add_argument("--jobs", type=int, help="parallel workers (default ~60%% of the logical cores, never all)")
    p.add_argument("--resume", action="store_true", help="continue this folder's run (verifies input / settings / mode / code; nothing finished is redone)")
    p.add_argument("--preflight-only", action="store_true", help="only the preflight (ends with SAFE TO MODEL or STOP)")
    p.add_argument("--estimate", action="store_true", help="runtime estimate from the file's shape only (no model fitted on the real data)")
    p.add_argument("--status", action="store_true", help="READ-ONLY progress of the folder")
    p.add_argument("--report-only", action="store_true", help="re-build share/ from the finished units only (no fitting)")
    p.add_argument("--accept-code-change", help="continue a run although the falls_ml code changed (reason recorded)")
    p.add_argument("--allow-synced-folder", action="store_true", help="allow --out inside OneDrive / a synced folder (not recommended)")
    p.add_argument("--config", default="configs/meuhedet/phase5.yaml", help="Phase 5 settings (pre-declared; hashed into the plan)")
    p.add_argument("--v21-catalogue", help="the pre-declared V21 catalogue (default: the one named in the settings)")
    p.set_defaults(func=_cmd_meuhedet_phase5)
    p = sub.add_parser("meuhedet-phase5-synthetic", help="Phase 5 SMOKE RUN on a generated SYNTHETIC V21 extract (software test only, no real data)")
    p.add_argument("--out", required=True, help="output folder for the synthetic run (the synthetic input is written next to it)")
    p.add_argument("--scenario", choices=["planted", "null"], default="planted", help="planted = one new feature carries signal; null = none does")
    p.add_argument("--rows", type=int, default=4000)
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--mode", choices=["quick", "overnight"], default="quick")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--jobs", type=int)
    p.add_argument("--accept-code-change", help=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_meuhedet_phase5_synthetic)


def _cmd_meuhedet_eda(a: argparse.Namespace) -> int:
    """Full, aggregate EDA of the wide extract: data dictionary, profiles, missingness, data quality, timing / leakage audit, cohort funnel,
    split diagnostic and TRAIN-only supervised EDA -> REAL_DATA_EDA_REPORT.html, REAL_DATA_EDA_SUMMARY.md, DATA_QUALITY_REPORT.md, eda/*.csv."""
    from falls_ml.eda.runner import run_eda
    from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError

    try:
        res = run_eda(a.input, out_dir=a.out, index_date=_iso_date(a.index_date, "--index-date") if a.index_date else None, reference_dir=a.reference,
                      id_pepper=a.id_pepper, id_pepper_file=a.id_pepper_file, sheet=a.sheet, encoding=a.encoding, sep=a.sep, mapping_path=a.mapping,
                      dictionary_path=a.dictionary, index_day_records=a.index_day_records, min_cell=a.min_cell, make_plots=not a.no_plots)
    except (DatasetValidationError, ConfigError, LeakageError) as exc:
        print("STOPPED - the EDA could not run; nothing was written to be shared.", file=sys.stderr)
        print("  " + str(exc).replace("\n", "\n  "), file=sys.stderr)
        return 2
    print((res.out_dir / "REAL_DATA_EDA_SUMMARY.md").read_text(encoding="utf-8"))
    print(json.dumps({"out_dir": str(res.out_dir), "report": str(res.out_dir / "REAL_DATA_EDA_REPORT.html"),
                      "summary": str(res.out_dir / "REAL_DATA_EDA_SUMMARY.md"), "data_quality": str(res.out_dir / "DATA_QUALITY_REPORT.md"),
                      "stages": {k: v["status"] for k, v in res.stages.items()}, "privacy_scan_passed": res.manifest["privacy"]["identifier_scan"]["passed"],
                      "seconds": res.timings.get("total")}, indent=2))
    return 0


def _cmd_model_report(a: argparse.Namespace) -> int:
    """Training dashboards, model comparison and Hebrew management reports for completed runs (reads artifacts; never writes into a run)."""
    from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError
    from falls_ml.modelreport.runner import build_model_reports

    try:
        rs = build_model_reports(a.results, out_dir=a.out, eda_dir=a.eda, learning_curve=a.learning_curve, min_cell=a.min_cell, n_boot=a.n_boot,
                                 mapping_path=a.mapping)
    except (DatasetValidationError, ConfigError, LeakageError, ValueError) as exc:
        print("STOPPED - the model reports could not be built.", file=sys.stderr)
        print("  " + str(exc).replace("\n", "\n  "), file=sys.stderr)
        return 2
    print(json.dumps({"out_dir": str(rs.out_dir), "primary_analysis": rs.manifest.get("primary_analysis"),
                      "reports": sorted(p.name for p in rs.out_dir.glob("*.html")), "figures": len(list((rs.out_dir / "figures").glob("*.png"))),
                      "conclusions_he": rs.conclusions_he, "problems": {r.run.key: r.problems for r in rs.runs if r.problems},
                      "privacy_scan_passed": rs.manifest["privacy"]["identifier_scan"]["passed"], "seconds": rs.manifest.get("seconds")}, indent=2, ensure_ascii=False))
    return 0


def _cmd_freeze_baseline(a: argparse.Namespace) -> int:
    """Freeze a completed run as a named, immutable baseline record (dataset/spec/mapping hashes, outcome, lambda, coefficients, metrics)."""
    from falls_ml.baseline import freeze_baseline

    record = freeze_baseline(a.run, a.name, baselines_dir=a.baselines_dir, note=a.note or "")
    print(json.dumps({"baseline": record["baseline_dir"], "name": a.name, "run_id": record["run_id"], "test_rows_sha256": record["split"]["test_rows_sha256"],
                      "data_sha256": record["dataset"]["data_sha256"], "outcome": record["outcome"], "coverage": record.get("efalls_coverage", {}).get("label")},
                     indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="falls_ml", description="Clinical fall-risk prediction pipeline (eFalls reproduction)")
    ap.add_argument("--verbose", action="store_true", help="debug-level structured logs on stderr")
    sub = ap.add_subparsers(dest="command", required=True)

    spec_help = f"feature spec YAML (default {DEFAULT_FEATURE_SPEC})"
    config_help = ("experiment config whose dataset.feature_spec, extensions and preprocessing.features define the spec "
                   "(a declared feature subset uses the subset spec); not combinable with --feature-spec/--extension")

    p = sub.add_parser("make-fixture", help="generate a deterministic SYNTHETIC dataset (software tests only)")
    p.add_argument("--out", required=True)
    p.add_argument("--n-patients", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--feature-spec", help=spec_help)
    p.add_argument("--extension", action="append")
    p.add_argument("--features-from-config", help=config_help + "; the fixture keeps only the declared predictor columns")
    p.set_defaults(func=_cmd_make_fixture)

    p = sub.add_parser("validate-dataset", help="validate a modelling dataset against its schema and manifest "
                                                "(full or subset feature spec selected by the manifest)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--feature-spec", help=spec_help)
    p.add_argument("--extension", action="append")
    p.add_argument("--features-from-config", help=config_help)
    p.set_defaults(func=_cmd_validate_dataset)

    p = sub.add_parser("build-dataset", help="validate a flat extract file (.parquet/.csv) and write an immutable modelling dataset "
                                             "(no database access)")
    p.add_argument("--input", required=True, help="extract file: one row per research_id x index_date (.parquet or .csv)")
    p.add_argument("--out", required=True, help="new or empty output directory")
    p.add_argument("--dataset-version", required=True)
    p.add_argument("--mapping-version", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--data-freeze-date", required=True, help="last date of source data extraction, YYYY-MM-DD (D-19 label maturity)")
    p.add_argument("--feature-spec", help=spec_help)
    p.add_argument("--extension", action="append")
    p.add_argument("--features-from-config", help=config_help)
    p.add_argument("--scientific-use-allowed", action="store_true", help="mark the dataset as approved for scientific use (requires "
                                                                          "--approval-reference; recorded in the manifest audit)")
    p.add_argument("--approval-reference", help="governance/ethics approval identifier for scientific use")
    p.add_argument("--notes", help="free-text manifest notes (no patient data)")
    p.set_defaults(func=_cmd_build_dataset)

    p = sub.add_parser("train", help="run one experiment (fit, evaluate, report, bundle)")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset")
    p.add_argument("--model", help="override the model name in the config")
    p.add_argument("--runs-dir")
    p.add_argument("--allow-test-reevaluation", action="store_true",
                   help="re-evaluate test rows already evaluated with this exact config (recorded; D-19 section 4)")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("evaluate", help="re-render the report of an existing run from its artifacts")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("reproduce", help="re-run an experiment from its saved artifacts and verify it is identical")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dataset")
    p.add_argument("--runs-dir", help="default <runs root>_reproduced")
    p.set_defaults(func=_cmd_reproduce)

    p = sub.add_parser("compare", help="build the master comparison and best-features reports")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="reports")
    p.add_argument("--ablation-dir", help="ablation summary directory (default <out>/ablation; a missing summary is reported as a warning)")
    p.add_argument("--n-bootstrap", type=int, default=1000, help="replicates for paired differences vs published eFalls")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("ablation", help="run nested feature-group ablation experiments")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset")
    p.add_argument("--runs-dir", help="default <output.runs_dir>_ablation (kept apart from the master comparison)")
    p.add_argument("--out", default="reports/ablation")
    p.set_defaults(func=_cmd_ablation)

    p = sub.add_parser("predict", help="score new rows with a saved model bundle")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-ids", action="store_true", help="omit research identifiers from the output")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("monitor", help="drift report for new data against a bundle's training profile (never retrains)")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--labels", action="store_true", help="input contains observed outcomes")
    p.set_defaults(func=_cmd_monitor)

    # ---- Meuhedet wide table (Phase 1): one local extract file, no database, no patient data in any output
    meu_mapping_help = "eFalls mapping manifest (default configs/meuhedet/wide_v1_efalls_mapping.yaml)"
    p = sub.add_parser("meuhedet-make-fixture", help="write a SYNTHETIC 221-column wide-table extract (software tests only)")
    p.add_argument("--out", required=True, help=".parquet or .csv path")
    p.add_argument("--n-rows", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--index-date", default="2025-01-01")
    p.add_argument("--index-day-falls", type=int, default=0, help="plant N index-day fall records (tests the D-00 timing guard)")
    p.set_defaults(func=_cmd_meuhedet_make_fixture)

    p = sub.add_parser("meuhedet-audit", help="aggregate, non-identifying audit of a wide-table extract (row counts, missingness, "
                                              "distributions, eFalls coverage, leakage checks); safe to take out of the data environment")
    p.add_argument("--input", required=True, help="extract file (.xlsx, .parquet or .csv with NULL literals)")
    p.add_argument("--sheet", help="Excel sheet name or 0-based index (default: first sheet)")
    p.add_argument("--out", required=True, help="output directory for wide_extract_audit.json / .md")
    p.add_argument("--index-date", help="cohort index date YYYY-MM-DD for the cohort profile (default: all rows)")
    p.add_argument("--mapping", default="configs/meuhedet/wide_v1_efalls_mapping.yaml", help=meu_mapping_help)
    p.add_argument("--contract", help="column contract YAML (default: the mapping's contract)")
    p.add_argument("--min-cell", type=int, default=10, help="suppress category/cross-tab cells below this count (M-13)")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding (SSMS exports: utf-16 or cp1255)")
    p.add_argument("--sep", default=",", help="CSV separator")
    p.set_defaults(func=_cmd_meuhedet_audit)

    p = sub.add_parser("meuhedet-build", help="wide-table extract -> canonical eFalls-mapped modelling dataset "
                                              "(Is_Eligible_Cohort = 1, one index date, 180-day EXPLORATORY outcome)")
    p.add_argument("--input", required=True, help="extract file (.xlsx, .parquet or .csv with NULL literals)")
    p.add_argument("--sheet", help="Excel sheet name or 0-based index (default: first sheet)")
    p.add_argument("--feature-set", choices=["strict", "extended"], help="manifest-derived exploratory feature set (default: every included mapping)")
    p.add_argument("--id-pepper", help="pseudonymise research ids with this pepper (sha256(pepper | Customer_Full_ID)[:20])")
    p.add_argument("--out", required=True, help="new or empty dataset directory")
    p.add_argument("--index-date", required=True, help="the single Index_Date to keep, YYYY-MM-DD")
    p.add_argument("--dataset-version", required=True)
    p.add_argument("--data-freeze-date", required=True, help="last date of source data extraction, YYYY-MM-DD (D-19 label maturity)")
    p.add_argument("--mapping", default="configs/meuhedet/wide_v1_efalls_mapping.yaml", help=meu_mapping_help)
    p.add_argument("--contract", help="column contract YAML (default: the mapping's contract)")
    p.add_argument("--index-day-records", choices=["fail", "drop_rows"], help="predictor record dates on/after the index date: "
                                                                                "fail (default, from the mapping) or drop_rows (counted)")
    p.add_argument("--scientific-use-allowed", action="store_true", help="mark the dataset as approved for scientific use (requires --approval-reference)")
    p.add_argument("--approval-reference", help="governance/ethics approval identifier for scientific use")
    p.add_argument("--notes", help="free-text manifest notes (no patient data)")
    p.add_argument("--report", help="also write the build summary JSON to this path (outside the dataset directory)")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--sep", default=",")
    p.set_defaults(func=_cmd_meuhedet_build)

    p = sub.add_parser("meuhedet-explore", help="ONE command for the first exploratory experiment: local .xlsx/.csv/.parquet wide extract -> "
                                                "audit -> STRICT and EXTENDED 180-day exploratory LASSO runs -> feature reports, comparison, SUMMARY")
    p.add_argument("--input", required=True, help="wide-table extract on this computer (.xlsx first sheet by default, .csv or .parquet)")
    p.add_argument("--out", help="new/empty results directory (default: <input name>_explore next to the input)")
    p.add_argument("--index-date", help="YYYY-MM-DD: model this one snapshot (required when the file holds several Index_Date values "
                                        "and --all-index-dates is not given)")
    p.add_argument("--all-index-dates", action="store_true", help="model every usable snapshot: temporal train / validation / test over "
                                                                   "Index_Date (D-19 embargo) plus a patient-disjoint sensitivity analysis")
    p.add_argument("--audit-only", action="store_true", help="everything except training: audits, D-00 diagnostic, snapshot audit, cohort, "
                                                              "split, adequacy -> READINESS.md says READY TO TRAIN or STOPPED - <reason>")
    p.add_argument("--sheet", help="Excel sheet name or 0-based index (default: first sheet)")
    p.add_argument("--data-freeze-date", help="YYYY-MM-DD last source load date (recorded; label maturity and snapshot observability cross-check)")
    p.add_argument("--dataset-version", help="dataset version label (default derived from the index date and file name)")
    p.add_argument("--id-pepper", help="pseudonymisation pepper value (default: the pepper stored next to the input, created on the first run)")
    p.add_argument("--id-pepper-file", help="where the pepper is stored (default: <input file>.id_pepper.txt next to the input; keep it local)")
    p.add_argument("--full", action="store_true", help="use the template's full resampling sizes (default: fast settings, see SUMMARY)")
    p.add_argument("--index-day-records", choices=["fail", "drop_rows"], help="predictor record dates on/after the index date (D-00): fail "
                                                                              "(default) stops with the timing diagnostic; drop_rows excludes and counts them")
    p.add_argument("--max-censored-share", type=float, default=0.5, help="a snapshot whose eligible rows are censored beyond this share is EXCLUDED "
                                                                          "(partially observable 180-day outcome)")
    p.add_argument("--no-patient-disjoint", action="store_true", help="skip the patient-disjoint sensitivity runs in --all-index-dates mode")
    p.add_argument("--model-report", choices=["full", "reduced", "off"], default="full", help="after training, build the training dashboards and the "
                   "Hebrew management reports in <out>/reports (full / reduced learning curve, or off)")
    p.add_argument("--min-cell", type=int, default=10, help="small-cell suppression in the audit (M-13)")
    p.add_argument("--mapping", default="configs/meuhedet/wide_v1_efalls_mapping.yaml", help=meu_mapping_help)
    p.add_argument("--encoding", default="utf-8", help="CSV encoding")
    p.add_argument("--sep", default=",", help="CSV separator")
    p.set_defaults(func=_cmd_meuhedet_explore)

    p = sub.add_parser("meuhedet-eda", help="full aggregate EDA of the wide extract (every column): data dictionary, profiles, missingness, data quality, "
                                            "timing / leakage audit, cohort funnel, split diagnostic, TRAIN-only supervised EDA, Table 1, Phase-2 catalogue")
    p.add_argument("--input", required=True, help="wide-table extract on this computer (.csv, .xlsx or .parquet)")
    p.add_argument("--out", help="new/empty results directory (default: <input name>_eda next to the input)")
    p.add_argument("--reference", help="completed meuhedet-explore results folder: its split is reproduced exactly (input sha256, pepper and "
                                       "test-partition hash verified); read-only")
    p.add_argument("--index-date", help="YYYY-MM-DD snapshot for the cohort / split part (default: the reference's, or the file's only index date)")
    p.add_argument("--index-day-records", choices=["fail", "drop_rows"], help="D-00 policy for the cohort build (default: the reference run's, else the mapping's)")
    p.add_argument("--sheet", help="Excel sheet name or 0-based index (default: first sheet)")
    p.add_argument("--id-pepper", help="pseudonymisation pepper value (default: the pepper stored next to the input)")
    p.add_argument("--id-pepper-file", help="where the pepper is stored (default: <input file>.id_pepper.txt; the reference folder's id_pepper.txt also works)")
    p.add_argument("--min-cell", type=int, default=10, help="small-cell suppression: counts 1..min_cell-1 are never shown")
    p.add_argument("--no-plots", action="store_true", help="skip the figures (tables and reports only)")
    p.add_argument("--dictionary", default="configs/meuhedet/wide_v1_data_dictionary.yaml", help="data dictionary YAML (domains, sources, meanings)")
    p.add_argument("--mapping", default="configs/meuhedet/wide_v1_efalls_mapping.yaml", help=meu_mapping_help)
    p.add_argument("--encoding", default="utf-8", help="CSV encoding")
    p.add_argument("--sep", default=",", help="CSV separator")
    p.set_defaults(func=_cmd_meuhedet_eda)

    p = sub.add_parser("model-report", help="training dashboards (CV curve, coefficient paths, learning curve, ROC/PR, calibration, thresholds, lift, "
                                            "features, subgroups, error analysis), model comparison and Hebrew management reports for completed runs")
    p.add_argument("--results", required=True, action="append", help="a meuhedet-explore or meuhedet-sensitivity results folder (repeatable); read-only")
    p.add_argument("--out", help="new/empty folder (default: <first results folder>_reports next to it)")
    p.add_argument("--eda", help="the meuhedet-eda results folder of the same file (linked in the comparison report)")
    p.add_argument("--learning-curve", choices=["full", "reduced", "off"], default="full", help="full = the whole fitting procedure per TRAIN subset; "
                   "reduced = lambda fixed at the run's lambda* (no CV per subset), documented in the report")
    p.add_argument("--min-cell", type=int, default=10, help="small-cell suppression in every shareable table and histogram")
    p.add_argument("--n-boot", type=int, default=1000, help="bootstrap replicates for lift intervals and paired differences")
    p.add_argument("--mapping", default="configs/meuhedet/wide_v1_efalls_mapping.yaml", help=meu_mapping_help)
    p.set_defaults(func=_cmd_model_report)

    p = sub.add_parser("meuhedet-sensitivity", help="SAFE-ALL-ROWS sensitivity analysis + targeted ablations next to a completed meuhedet-explore run: "
                                                    "feature provenance, all labelled rows with temporally safe predictors, EXTENDED without the implicated "
                                                    "predictors on identical rows, convergence audit, comparison, plain-language report")
    p.add_argument("--input", required=True, help="the SAME extract file the reference run used (verified by sha256)")
    p.add_argument("--reference", required=True, help="the completed meuhedet-explore results directory (single snapshot); read-only, hashed before and after")
    p.add_argument("--out", help="new/empty results directory (default: <reference>_sensitivity next to it)")
    p.add_argument("--sheet", help="Excel sheet name or 0-based index (default: first sheet)")
    p.add_argument("--data-freeze-date", help="YYYY-MM-DD last source load date (recorded)")
    p.add_argument("--id-pepper", help="pseudonymisation pepper value (default: the pepper stored next to the input; must match the reference run)")
    p.add_argument("--id-pepper-file", help="where the pepper is stored (default: <input file>.id_pepper.txt; the reference folder's id_pepper.txt also works)")
    p.add_argument("--no-ablations", action="store_true", help="skip the EXTENDED-minus-implicated-predictor runs on the reference cohort")
    p.add_argument("--no-safe-all-rows", action="store_true", help="skip the SAFE-ALL-ROWS run")
    p.add_argument("--model-report", choices=["full", "reduced", "off"], default="full", help="after training, build the training dashboards and the "
                   "Hebrew management reports in <out>/reports (full / reduced learning curve, or off)")
    p.add_argument("--min-cell", type=int, default=10, help="small-cell suppression in the timing diagnostic (M-13)")
    p.add_argument("--mapping", default="configs/meuhedet/wide_v1_efalls_mapping.yaml", help=meu_mapping_help)
    p.add_argument("--encoding", default="utf-8", help="CSV encoding")
    p.add_argument("--sep", default=",", help="CSV separator")
    p.set_defaults(func=_cmd_meuhedet_sensitivity)

    p = sub.add_parser("meuhedet-d00", help="D-00 sensitivity framework next to a completed meuhedet-explore run: feature-dependency graph, "
                                             "provenance-derived feature sets, FULL_LABELED vs D00_CLEAN matrix, pre-specified ablations (falls, "
                                             "mobility), risk concentration, learning-curve interpretation, LASSO warnings audit, reports")
    p.add_argument("--input", required=True, help="the SAME extract file the reference run used (verified by sha256)")
    p.add_argument("--reference", required=True, help="the completed meuhedet-explore results folder (read-only, hashed before and after)")
    p.add_argument("--out", required=True, help="NEW folder; everything shareable goes to <out>\\share")
    p.add_argument("--eda", help="the existing meuhedet-eda folder (read-only, hashed before and after)")
    p.add_argument("--reports", help="the existing model-report folder (read-only; its learning-curve tables are cross-checked)")
    p.add_argument("--id-pepper-file", help="the reference's pepper (default: <reference>\\id_pepper.txt; its fingerprint must match the reference run)")
    p.add_argument("--config", default="configs/meuhedet/d00_sensitivity.yaml", help="D-00 evidence standard and pre-specified plan")
    p.add_argument("--dependency-only", action="store_true", help="write the dependency graph and the frozen plan only (no training; minutes)")
    p.add_argument("--learning-curve", choices=["full", "reduced", "reuse", "off"], default=None,
                   help="recompute the STRICT / EXTENDED learning curves with diagnostics (full, the default / reduced), reuse the --reports tables, or off")
    p.add_argument("--resampling", choices=["reference", "reduced"], default=None,
                   help="stability / optimism / permutation repeats of the new runs: as the reference (default) or reduced (faster; CIs unchanged)")
    p.add_argument("--no-secondary-standard", action="store_true", help="run only the primary 'proven safe' evidence standard")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted analysis in the SAME --out: the frozen plan must re-derive identically (its resampling / learning-curve "
                        "settings are used); completed runs are reused as they are (never refitted, test sets never re-evaluated); an incomplete run "
                        "is never read as a result and its cell is refitted only if it never released its test set")
    p.add_argument("--no-management-report", action="store_true", help="skip the new management-report folder")
    p.add_argument("--min-cell", type=int, default=10, help="small-cell suppression in every shareable table")
    p.add_argument("--n-boot", type=int, default=1000, help="bootstrap replicates for paired differences and risk-concentration intervals")
    p.add_argument("--sheet", help="Excel sheet name or 0-based index (default: first sheet)")
    p.add_argument("--mapping", default="configs/meuhedet/wide_v1_efalls_mapping.yaml", help=meu_mapping_help)
    p.add_argument("--encoding", default="utf-8", help="CSV encoding")
    p.add_argument("--sep", default=",", help="CSV separator")
    p.set_defaults(func=_cmd_meuhedet_d00)

    p = sub.add_parser("meuhedet-phase2", help="Phase 2 discovery next to a completed meuhedet-explore run: column registry, engineered features, "
                                               "TRAIN-only screening, nested CV of LASSO / elastic net / XGBoost (Optuna), frozen selection, one-time "
                                               "VALIDATION scoring, stability, ablation, SHAP, consensus, reports, share package (crash-safe, --resume)")
    p.add_argument("--input", required=True, help="the SAME extract file the reference run used (verified by sha256)")
    p.add_argument("--reference", required=True, help="the completed meuhedet-explore results folder (read-only, hashed at every stage)")
    p.add_argument("--out", required=True, help="NEW local folder (not OneDrive); only <out>\\share is for review")
    p.add_argument("--protect", action="append", help="another earlier result folder that must stay byte-identical (repeatable: EDA, reports, D-00)")
    p.add_argument("--id-pepper-file", help="the reference's pepper (default: <reference>\\id_pepper.txt); never created by Phase 2")
    p.add_argument("--config", default="configs/meuhedet/phase2.yaml", help="Phase 2 analysis settings (frozen into the plan)")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run in the SAME --out from the first incomplete item")
    p.add_argument("--accept-gate", action="append", help="continue past this investigation stop (needs --reason; recorded)")
    p.add_argument("--reason", help="why the investigation stop was judged acceptable (recorded in RESUME_AUDIT.json and logs/gates.jsonl)")
    p.add_argument("--accept-code-change", help="resume although the falls_ml code changed (reason recorded; completed items are never recomputed)")
    p.add_argument("--allow-synced-folder", action="store_true", help="allow --out inside OneDrive / a synced folder (not recommended; recorded)")
    p.add_argument("--preflight", action="store_true", help="check everything for a NEW run without fitting any model or writing to --out; "
                                                            "the last line is SAFE TO START FULL RUN or NOT SAFE TO START FULL RUN")
    p.add_argument("--allow-unfrozen-config", action="store_true", help="development / tests only: allow a configuration that differs from "
                                                                        "configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json (flagged in every output)")
    p.set_defaults(func=_cmd_meuhedet_phase2)

    p = sub.add_parser("meuhedet-phase2-status", help="READ-ONLY progress of a Phase 2 run (stage, item, XGBoost trials, times, gates); "
                                                      "never writes, moves or locks anything")
    p.add_argument("--out", required=True, help="the Phase 2 output folder")
    p.set_defaults(func=_cmd_meuhedet_phase2_status)

    p = sub.add_parser("meuhedet-phase3", help="Phase 3: row-level historical recovery of predictors Phase 2 had to exclude, a pre-declared scientific "
                                               "GO / NO-GO, and (GO only) the extended LASSO / elastic net / XGBoost experiment evaluated with bounds for "
                                               "overwritten history; NO-GO writes the DWH remediation requirements (crash-safe, --resume)")
    p.add_argument("--input", required=True, help="the SAME extract file the reference run used (verified by sha256)")
    p.add_argument("--reference", required=True, help="the completed meuhedet-explore results folder (read-only, hashed at every stage)")
    p.add_argument("--out", required=True, help="NEW local folder, separate from every Phase 2 folder (not OneDrive); only <out>\\share is for review")
    p.add_argument("--protect", action="append", help="another earlier, FINISHED result folder that must stay byte-identical (EDA, reports, D-00); "
                                                      "never the running Phase 2 folder")
    p.add_argument("--phase2-out", help="optional: the Phase 2 output folder, only to record whether it opened VALIDATION (two small files read; "
                                        "never written, never hashed)")
    p.add_argument("--id-pepper-file", help="the reference's pepper (default: <reference>\\id_pepper.txt or <input>.id_pepper.txt); never created")
    p.add_argument("--config", default="configs/meuhedet/phase3.yaml", help="Phase 3 analysis settings (frozen into the plan)")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run in the SAME --out from the first incomplete item")
    p.add_argument("--accept-gate", action="append", help="continue past this investigation stop (needs --reason and --resume; recorded)")
    p.add_argument("--reason", help="why the investigation stop was judged acceptable (recorded)")
    p.add_argument("--accept-code-change", help="resume although the falls_ml code changed (reason recorded; completed items are never recomputed)")
    p.add_argument("--allow-synced-folder", action="store_true", help="allow --out inside OneDrive / a synced folder (not recommended; recorded)")
    p.add_argument("--preflight", action="store_true", help="check everything for a NEW run without fitting any model or writing to --out; prints the "
                                                            "scientific GO / NO-GO forecast and last line SAFE TO START FULL RUN or NOT SAFE TO START FULL RUN")
    p.add_argument("--allow-unfrozen-config", action="store_true", help="development / tests only: allow a configuration that differs from "
                                                                        "configs/meuhedet/PHASE3_FINAL_EXPERIMENT_CONFIG.json (flagged in every output)")
    p.set_defaults(func=_cmd_meuhedet_phase3)

    p = sub.add_parser("meuhedet-phase3-status", help="READ-ONLY progress of a Phase 3 run; never writes, moves or locks anything")
    p.add_argument("--out", required=True, help="the Phase 3 output folder")
    p.set_defaults(func=_cmd_meuhedet_phase3_status)

    _add_phase4_parsers(sub)
    _add_phase5_parsers(sub)

    p = sub.add_parser("freeze-baseline", help="freeze a completed run as a named immutable baseline (e.g. EFALLS_BASELINE_MEUHEDET_V1)")
    p.add_argument("--run", required=True, help="completed run directory (RUN_COMPLETE.json present)")
    p.add_argument("--name", required=True, help="baseline name, e.g. EFALLS_BASELINE_MEUHEDET_V1")
    p.add_argument("--baselines-dir", default="baselines")
    p.add_argument("--note", help="free text (no patient data)")
    p.set_defaults(func=_cmd_freeze_baseline)
    return ap


def _safe_console_streams() -> None:
    """Never crash on non-ASCII output: a redirected stdout/stderr on Windows uses the ANSI code page (cp1252/cp1255) with strict
    errors, and tables printed by compare/ablation hold user-controlled text. Non-UTF-8 streams get ``errors="backslashreplace"``."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if encoding not in {"utf8", "utf8sig"} and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def main(argv: list[str] | None = None) -> int:
    _safe_console_streams()
    args = build_parser().parse_args(argv)
    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
