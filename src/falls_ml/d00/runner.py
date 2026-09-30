"""``falls_ml meuhedet-d00``: the full D-00 sensitivity framework next to a completed ``meuhedet-explore`` reference run.

Stages (each one aggregate-only in its shareable output):
 0. reference, EDA and report folders located, fingerprinted (every file hashed; re-checked at the end - they are only read)
 1. the extract is read; it must be the reference's file (sha256) and use the reference's pepper (fingerprint)
 2. cohorts FULL_LABELED / D00_CLEAN (label availability only) and the D-00 feature-dependency graph (record dates only)
 3. the analysis plan (matrix + ablations + future cell) is derived from the graph and FROZEN (ANALYSIS_PLAN.json, sha256) before any fit
 4. datasets per cell; D00_CLEAN rebuilt and verified row by row against the reference (rows, values, test partition); FULL_LABELED built
    with the D-00 guard of the built predictors in 'fail' mode and every unsafe column forbidden - it can never drop a row silently
 5. runs (reference cells reused, identical cells share one run), with the reference run's own config (same model, seed, proportions)
 6. comparisons: feature-set effects (paired, identical test rows verified), population effects (NOT paired), pre-specified ablations,
    risk concentration, learning-curve increments, LASSO warnings audit
 7. reports (D00_SENSITIVITY_REPORT.html, D00_SENSITIVITY_SUMMARY_HE.md, dependency files, warnings audit, tables, figures) and a NEW
    management-report folder; identifier scan of everything shareable; integrity of the reference / EDA / report folders

Everything shareable is written under ``<out>/share``; ``<out>/datasets``, ``<out>/runs`` (row-level predictions) and ``<out>/id_pepper.txt``
stay on the work computer.

``resume=True`` (``--resume``, v0.7.1) continues an interrupted attempt in the SAME ``--out`` folder: the plan is re-derived and must have
the frozen plan's sha256 (the plan's settings - resampling, learning curve, secondary standard - are taken from it); every cell whose run
COMPLETED earlier (RUN_COMPLETE.json + metrics.json, test release recorded once in the registry, same config / dataset / test partition)
is reused as it is - never refitted, its test set never re-evaluated; an incomplete run directory is never read as a result - it is kept
for audit and its cell is refitted, but only if the registry shows that it never released its test set. Every file of the earlier attempt
is hashed before and after: nothing may change (the registry may only be appended to).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from falls_ml.d00 import analysis as A
from falls_ml.d00.dependency import DEFAULT_D00_CONFIG, Graph, build_graph, cohort_masks, load_d00_config
from falls_ml.d00.plan import FINAL, REF_EXTENDED, REF_STRICT, Cell, build_plan, plan_record
from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError
from falls_ml.logging_utils import get_logger

log = get_logger(__name__)

WATERMARK = "EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION"
STATUS_LINES = ["one Index Date only (single snapshot)", "internal split only (patient-grouped random split of that snapshot)",
                "no temporal validation and no external validation", "approximate mappings exist (APPROXIMATE eFalls concepts; mappings not clinically validated)",
                "the D-00 timing issue still requires the DWH source correction (Event_Date < Index_Date)", "not ready for clinical deployment"]
REQUESTED = ("falls", "mobility_problems", "Prior_Fall_Since_Study_Start_Ind", "Prior_Fall_Count_Since_Study_Start", "Prior_Fall_Count_30D",
             "Prior_Fall_Count_90D", "Prior_Fall_Count_180D", "Prior_Fall_Count_365D", "Days_Since_Last_Fall", "Gait_Disorder_Since_Study_Start_Ind",
             "Diagnosis_Count_180D", "Diagnosis_Count_365D", "Distinct_Diagnosis_Codes_365D", "Chronic_Diagnosis_Count_365D", "Days_Since_Last_Diagnosis",
             "Fall_On_Index_Date_Ind", "Prior_Fall_Missing_Ind", "CCI_Group")
RESAMPLING_REDUCED = {"stability_n": 50, "optimism_n": 10, "permutation_repeats": 5}
STATUS_NEW = "RUN (new)"
STATUS_REUSED_EARLIER = "REUSED (completed in an earlier attempt of this D-00 run; not refitted, test set not re-evaluated)"


@dataclass
class D00Result:
    out_dir: Path
    share_dir: Path
    reference_dir: Path
    index_date: str = ""
    graph: Graph | None = None
    cells: list[Cell] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    integrity: dict[str, Any] = field(default_factory=dict)
    privacy: dict[str, Any] = field(default_factory=dict)
    management: dict[str, Any] = field(default_factory=dict)
    synthetic: bool = False
    seconds: dict[str, float] = field(default_factory=dict)
    resume: dict[str, Any] = field(default_factory=lambda: {"resumed": False})


def _clean(obj: Any) -> Any:
    """Strict JSON: NaN / inf -> null, numpy scalars -> Python, DataFrames never reach here."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return obj.name
    return obj


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8", newline="\n")


def _write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def _csv(df: pd.DataFrame, p: Path, basis: str) -> None:
    from falls_ml.eda.common import write_csv

    write_csv(df, p, watermark=WATERMARK, basis=basis)


def _perf(metrics: dict[str, Any], variant: str | None = None) -> dict[str, Any]:
    test = (metrics.get("performance") or {}).get("test") or {}
    return test.get(variant or metrics.get("served_variant") or metrics.get("primary_variant") or "uncalibrated") or {}


def _est(d: Any) -> tuple[float | None, float | None, float | None]:
    d = d or {}
    return d.get("estimate"), d.get("ci_low"), d.get("ci_high")


def run_d00(input_path: str | Path, reference_dir: str | Path, *, out_dir: str | Path, eda_dir: str | Path | None = None,
            reports_dir: str | Path | None = None, id_pepper_file: str | Path | None = None, config_path: str | Path = DEFAULT_D00_CONFIG,
            mapping_path: str | Path | None = None, dictionary_path: str | Path | None = None, dependency_only: bool = False,
            learning_curve: str | None = None, resampling: str | None = None, secondary: bool | None = None, min_cell: int = 10, n_boot: int = 1000,
            management_report: bool = True, sheet: str | None = None, encoding: str = "utf-8", sep: str = ",",
            template_path: str | Path | None = None, resume: bool = False) -> D00Result:
    """``learning_curve`` full|reduced|reuse|off (default full), ``resampling`` reference|reduced (default reference), ``secondary`` (default
    True). With ``resume=True`` these three come from the frozen plan of the interrupted attempt in ``out_dir`` (a differing value is refused)."""
    from falls_ml.data.meuhedet_wide import DEFAULT_MAPPING
    from falls_ml.eda.dictionary import DEFAULT_DICTIONARY

    src, ref_dir, out = Path(input_path), Path(reference_dir), Path(out_dir)
    if not src.is_file():
        raise DatasetValidationError(f"Input file not found: {src}")
    stored_plan: dict[str, Any] | None = None
    if resume:
        if dependency_only:
            raise ConfigError("--resume continues an interrupted full analysis; it cannot be combined with --dependency-only")
        plan_file = out / "share" / "ANALYSIS_PLAN.json"
        if not plan_file.is_file():
            raise DatasetValidationError(f"nothing to resume in {out}: share/ANALYSIS_PLAN.json (the frozen plan of an earlier attempt) is missing")
        stored_plan = json.loads(plan_file.read_text(encoding="utf-8"))
        for name, given, key in (("learning_curve", learning_curve, "learning_curve"), ("resampling", resampling, "resampling"),
                                 ("secondary", secondary, "secondary_standard")):
            if given is not None and given != stored_plan.get(key):
                raise DatasetValidationError(f"--resume: {name}={given!r} differs from the frozen plan of the interrupted attempt "
                                             f"({stored_plan.get(key)!r}); omit it (the frozen value is used) or pass the original value")
        learning_curve, resampling, secondary = stored_plan["learning_curve"], stored_plan["resampling"], bool(stored_plan["secondary_standard"])
    else:
        learning_curve = learning_curve or "full"
        resampling = resampling or "reference"
        secondary = True if secondary is None else bool(secondary)
    if learning_curve not in ("full", "reduced", "reuse", "off") or resampling not in ("reference", "reduced"):
        raise ValueError("learning_curve must be full|reduced|reuse|off and resampling reference|reduced")
    protected = [p for p in (ref_dir, Path(eda_dir) if eda_dir else None, Path(reports_dir) if reports_dir else None) if p is not None]
    for p in protected:
        if not p.is_dir():
            raise DatasetValidationError(f"folder not found: {p}")
        if out.resolve() == p.resolve() or p.resolve() in out.resolve().parents or out.resolve() in p.resolve().parents:
            raise DatasetValidationError(f"--out {out} must be a NEW folder outside {p} (completed results are immutable)")
    if not resume and out.exists() and any(out.iterdir()):
        hint = (" (it holds an interrupted D-00 analysis: continue it with --resume, which reuses the completed runs)"
                if (out / "share" / "ANALYSIS_PLAN.json").is_file() else "")
        raise DatasetValidationError(f"Output directory {out} is not empty; results are immutable - choose a new --out{hint}")
    share = out / "share"
    for d in (share, share / "tables", share / "figures"):
        d.mkdir(parents=True, exist_ok=True)
    res = D00Result(out_dir=out, share_dir=share, reference_dir=ref_dir)
    t0 = time.perf_counter()
    before = {str(p): _digest(p) for p in protected}
    earlier = _earlier_files(out) if resume else None
    prior_attempts = _prior_resume_attempts(share) if resume else []
    if resume:
        res.resume["earlier_resume_attempts"] = [
            {"falls_ml_version": a.get("falls_ml_version_now"), "completed": a.get("completed", False),
             "refitted": [c["cell"] for c in a.get("cells", []) if c.get("action") != "REUSED_COMPLETED_RUN"],
             "refitted_cells": [{"cell": c["cell"], "bootstrap_convergence_retry": c.get("bootstrap_convergence_retry") or {}}
                                for c in a.get("cells", []) if c.get("action") != "REUSED_COMPLETED_RUN"]} for a in prior_attempts]
    try:
        _stages(res, src, out, share, protected=protected, eda_dir=eda_dir, reports_dir=reports_dir, id_pepper_file=id_pepper_file,
                config_path=config_path, mapping_path=mapping_path or DEFAULT_MAPPING, dictionary_path=dictionary_path or DEFAULT_DICTIONARY,
                dependency_only=dependency_only, learning_curve=learning_curve, resampling=resampling, secondary=secondary, min_cell=min_cell,
                n_boot=n_boot, management_report=management_report, sheet=sheet, encoding=encoding, sep=sep, template_path=template_path,
                stored_plan=stored_plan, earlier=earlier, prior_attempts=prior_attempts)
    except BaseException as exc:
        if res.resume.get("resumed") and not res.resume.get("completed"):   # a stopped resume is logged too (not a shareable result)
            res.resume["stopped"] = " ".join(f"{type(exc).__name__}: {exc}".split())[:600]
            _write_resume_log(share, prior_attempts, res.resume)
        raise
    finally:
        after = {str(p): _digest(p) for p in protected}
        res.integrity = {"folders": {k: {"n_files": before[k]["n_files"], "combined_sha256_before": before[k]["combined_sha256"],
                                         "combined_sha256_after": after[k]["combined_sha256"], "unchanged": before[k] == after[k],
                                         "changed_files": sorted(f for f in set(before[k]["files"]) | set(after[k]["files"])
                                                                 if before[k]["files"].get(f) != after[k]["files"].get(f))}
                                     for k in before},
                         "rule": "every file of the reference run, the EDA folder and the report folder hashed before and after this command (they are only read)"}
        res.integrity["all_unchanged"] = all(v["unchanged"] for v in res.integrity["folders"].values())
        if earlier is not None:
            res.integrity["earlier_attempt"] = _earlier_check(out, earlier)
            res.integrity["all_unchanged"] = res.integrity["all_unchanged"] and res.integrity["earlier_attempt"]["unchanged"]
        _write_json(share / "reference_integrity.json", {**res.integrity, "folders": {Path(k).name: v for k, v in res.integrity["folders"].items()}})
    if not res.integrity["all_unchanged"]:
        problems = [f"{Path(k).name}: {v['changed_files'][:5]}" for k, v in res.integrity["folders"].items() if not v["unchanged"]]
        if earlier is not None and not res.integrity["earlier_attempt"]["unchanged"]:
            problems.append(f"earlier attempt in {out.name}: {res.integrity['earlier_attempt']['changed_files'][:5]}")
        raise DatasetValidationError("a protected folder (reference / EDA / reports / the earlier attempt's runs) changed during the D-00 analysis", problems)
    res.seconds["total"] = round(time.perf_counter() - t0, 1)
    return res


def _digest(p: Path) -> dict[str, Any]:
    from falls_ml.meuhedet_sensitivity import directory_digest

    return directory_digest(p)


# ============================================================================ resume: inventory and integrity of the earlier attempt
REGISTRY = "test_evaluation_registry.jsonl"
EARLIER_PARTS = ("runs", "datasets", "configs")


def _earlier_files(out: Path) -> dict[str, Any]:
    """sha256 of every file the interrupted attempt left in runs/, datasets/, configs/ and id_pepper.txt (the registry: its bytes)."""
    import hashlib

    files: dict[str, str] = {}
    for part in EARLIER_PARTS:
        root = out / part
        if root.is_dir():
            for f in sorted(p for p in root.rglob("*") if p.is_file()):
                rel = f.relative_to(out).as_posix()
                if rel != f"runs/{REGISTRY}":
                    files[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    if (out / "id_pepper.txt").is_file():
        files["id_pepper.txt"] = hashlib.sha256((out / "id_pepper.txt").read_bytes()).hexdigest()
    reg = out / "runs" / REGISTRY
    return {"files": files, "registry": reg.read_bytes() if reg.is_file() else b""}


def _prior_resume_attempts(share: Path) -> list[dict[str, Any]]:
    p = share / "RESUME_LOG.json"
    if not p.is_file():
        return []
    try:
        return list(json.loads(p.read_text(encoding="utf-8")).get("attempts") or [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return []


def _write_resume_log(share: Path, prior: list[dict[str, Any]], current: dict[str, Any]) -> None:
    """RESUME_LOG.json keeps every resume attempt (oldest first) and repeats the latest one."""
    entry = {k: v for k, v in current.items() if k != "earlier_resume_attempts"}
    _write_json(share / "RESUME_LOG.json", {"watermark": WATERMARK, "n_resume_attempts": len(prior) + 1, "latest": entry, "attempts": [*prior, entry]})


def _earlier_check(out: Path, before: dict[str, Any]) -> dict[str, Any]:
    now = _earlier_files(out)
    changed = sorted(k for k, v in before["files"].items() if now["files"].get(k) != v)
    reg_ok = now["registry"].startswith(before["registry"])
    new_lines = now["registry"][len(before["registry"]):].decode("utf-8").splitlines() if reg_ok else []
    return {"n_files_checked": len(before["files"]), "changed_files": changed, "registry_only_appended": reg_ok,
            "registry_entries_added": len([ln for ln in new_lines if ln.strip()]), "unchanged": not changed and reg_ok,
            "rule": "every file of the interrupted attempt (runs/ - completed and incomplete - datasets/, configs/, id_pepper.txt) hashed before "
                    "and after the resumed analysis; the test-evaluation registry may only be appended to"}


def _run_inventory(out: Path) -> dict[str, Any]:
    """Run directories of the earlier attempt, by experiment name, with completion and test-release status (from the registry)."""
    from falls_ml.experiment import COMPLETE_MARKER

    runs_dir = out / "runs"
    released: dict[str, list[dict[str, Any]]] = {}
    reg = runs_dir / REGISTRY
    if reg.is_file():
        for ln in reg.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                e = json.loads(ln)
                released.setdefault(str(e.get("run_id")), []).append(e)
    by_label: dict[str, list[dict[str, Any]]] = {}
    unidentified: list[dict[str, Any]] = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []:
        complete = (d / COMPLETE_MARKER).is_file() and (d / "metrics.json").is_file()
        name = None
        try:
            name = (yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) or {}).get("experiment", {}).get("name")
        except (OSError, yaml.YAMLError):
            name = None
        entry = {"dir": d, "run_id": d.name, "complete": complete, "releases": released.get(d.name, []), "failure": None if complete else _failure_of(d)}
        if name is None:
            if complete:
                raise DatasetValidationError(f"--resume: completed run {d.name} has no readable config.yaml; it cannot be matched to a cell")
            unidentified.append(entry)
        else:
            by_label.setdefault(str(name), []).append(entry)
    return {"by_label": by_label, "unidentified": unidentified, "released": released}


def _failure_of(run_dir: Path) -> dict[str, Any]:
    """Why an incomplete run stopped (last run_failed event of its run_log; aggregate text only) and what its log shows about replicates."""
    info: dict[str, Any] = {"error": None, "bootstrap_replicate_failed_events": 0, "bootstrap_convergence_retry_events": 0}
    log_file = run_dir / "run_log.jsonl"
    if not log_file.is_file():
        info["error"] = "no run log (stopped before logging)"
        return info
    for ln in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        ev = e.get("event")
        if ev == "bootstrap_replicate_failed":
            info["bootstrap_replicate_failed_events"] += 1
        elif ev == "bootstrap_convergence_retry":
            info["bootstrap_convergence_retry_events"] += 1
        elif ev == "run_failed":
            lines = [x for x in str(e.get("exception") or "").strip().splitlines() if x.strip()]
            info["error"] = " ".join(lines[-1].split())[:500] if lines else "run_failed (no exception text)"
    if info["error"] is None:
        info["error"] = "no run_failed event (the process was interrupted)"
    return info


# ============================================================================ stages
def _stages(res: D00Result, src: Path, out: Path, share: Path, *, protected: list[Path], eda_dir: Any, reports_dir: Any, id_pepper_file: Any,
            config_path: Any, mapping_path: Any, dictionary_path: Any, dependency_only: bool, learning_curve: str, resampling: str, secondary: bool,
            min_cell: int, n_boot: int, management_report: bool, sheet: Any, encoding: str, sep: str, template_path: Any,
            stored_plan: dict[str, Any] | None = None, earlier: dict[str, Any] | None = None, prior_attempts: list[dict[str, Any]] | None = None) -> None:
    from falls_ml.data.dataset import sha256_file
    from falls_ml.data.meuhedet_wide import load_wide_contract, load_wide_mapping, read_wide_extract_report
    from falls_ml.eda.dictionary import load_data_dictionary
    from falls_ml.features.spec import load_feature_spec
    from falls_ml.meuhedet_explore import resolve_id_pepper
    from falls_ml.meuhedet_sensitivity import _Stop, load_reference

    t = time.perf_counter()
    try:
        ref = load_reference(res.reference_dir)
    except _Stop as stop:
        raise DatasetValidationError(f"STOPPED - {stop.reason}", stop.details) from None
    res.index_date = str(ref["index_date"])
    mapping = load_wide_mapping(mapping_path)
    contract = load_wide_contract(mapping.contract_path)
    dictionary = load_data_dictionary(dictionary_path, contract)
    config = load_d00_config(config_path, contract=contract, dictionary=dictionary)
    spec = load_feature_spec(mapping.exploratory_spec_path)
    read = read_wide_extract_report(src, contract, encoding=encoding, sep=sep, sheet=sheet)
    input_sha = sha256_file(src)
    ref_input_sha = ref["build"].get("input_sha256")
    if ref_input_sha and ref_input_sha != input_sha:
        raise DatasetValidationError("the input file is not the file the reference run used (sha256 differs); rows would not be comparable",
                                     [f"reference input sha256 {ref_input_sha}", f"this file {input_sha}"])
    if id_pepper_file is None and (res.reference_dir / "id_pepper.txt").is_file():
        id_pepper_file = res.reference_dir / "id_pepper.txt"   # the reference's own pepper: never create a new one by accident
    if stored_plan is None:
        pepper, pinfo = resolve_id_pepper(src, out, pepper_file=id_pepper_file)
    else:   # resume: the earlier attempt's id_pepper.txt is never rewritten; the pepper must be the frozen plan's
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pepper, pinfo = resolve_id_pepper(src, Path(tmp), pepper_file=id_pepper_file)
        if pinfo["sha256"] != stored_plan.get("pepper_sha256"):
            raise DatasetValidationError("--resume: the pseudonymisation pepper differs from the interrupted attempt's (research ids would not match)",
                                         [f"frozen plan pepper sha256 {stored_plan.get('pepper_sha256')}", f"this run {pinfo['sha256']}"])
    if ref["pepper_sha256"] and pinfo["sha256"] != ref["pepper_sha256"]:
        raise DatasetValidationError("the pseudonymisation pepper differs from the reference run's (research ids and split would not match)",
                                     [f"reference pepper sha256 {ref['pepper_sha256']}", f"this run {pinfo['sha256']}",
                                      "pass --id-pepper-file <reference results folder>\\id_pepper.txt"])
    res.synthetic = bool(ref["build"].get("synthetic"))
    res.seconds["read"] = round(time.perf_counter() - t, 1)

    # ---- 2. cohorts + dependency graph (no outcome values, no test partition)
    cohorts = cohort_masks(read.frame, mapping, res.index_date)
    coef = pd.read_csv(ref["runs"]["extended"] / "coefficients.csv") if (ref["runs"]["extended"] / "coefficients.csv").exists() else pd.DataFrame()
    design = ({f: sorted(set(coef.loc[(coef["feature"] == f) & (coef["design_column"] != "_intercept"), "design_column"].astype(str)))
               for f in coef["feature"].dropna().unique()} if len(coef) and "feature" in coef else {})
    graph = build_graph(read.frame, cohorts, mapping=mapping, contract=contract, dictionary=dictionary, config=config, design_columns=design,
                        requested=REQUESTED)
    res.graph = graph
    from falls_ml.d00 import report as R

    # ---- 3. the plan, frozen before any fit (resume: re-derived and required to be identical to the frozen one)
    cells = build_plan(graph, config, mapping, ref["features"], secondary=secondary)
    res.cells = cells
    facts = {"index_date": res.index_date, "input_sha256": input_sha, "pepper_sha256": pinfo["sha256"], "mapping_sha256": mapping.content_sha256,
             "dictionary_sha256": dictionary.sha256, "reference_test_rows_sha256": ref["split_audit"]["test_rows_sha256"],
             "reference_runs": {k: v.name for k, v in ref["runs"].items()}, "resampling": resampling, "learning_curve": learning_curve,
             "secondary_standard": secondary, "risk_fractions": list(config.plan.get("risk_concentration_fractions") or [0.01, 0.02, 0.05, 0.1, 0.2])}
    res.plan = plan_record(cells, graph=graph, config=config, facts=facts)
    if stored_plan is not None and stored_plan.get("plan_sha256") != res.plan["plan_sha256"]:
        differ = sorted(k for k in set(res.plan) | set(stored_plan) if k not in ("plan_sha256", "watermark") and
                        json.dumps(_clean(res.plan.get(k)), sort_keys=True, default=str) != json.dumps(stored_plan.get(k), sort_keys=True, default=str))
        raise DatasetValidationError("--resume refused: the re-derived analysis plan differs from the frozen plan of the interrupted attempt "
                                     "(same input file, reference, pepper, definitions and falls_ml plan code are required)",
                                     [f"frozen plan sha256 {stored_plan.get('plan_sha256')}", f"re-derived {res.plan['plan_sha256']}", f"differing parts: {differ}"])
    _write_json(share / "D00_FEATURE_DEPENDENCY.json", {"watermark": WATERMARK, "synthetic_data": res.synthetic, **graph.to_dict()})
    _csv(R.dependency_table(graph), share / "D00_FEATURE_DEPENDENCY.csv", "DEFINITIONS + record-date evidence (no outcome values)")
    _csv(R.column_table(graph), share / "tables" / "d00_column_dependency.csv", "DEFINITIONS + record-date evidence (no outcome values)")
    _write_text(share / "D00_FEATURE_DEPENDENCY.md", R.dependency_markdown(graph, WATERMARK, STATUS_LINES, synthetic=res.synthetic))
    if stored_plan is None:
        _write_json(share / "ANALYSIS_PLAN.json", {"watermark": WATERMARK, **res.plan})
        log.info("d00_plan_frozen", extra_fields={"plan_sha256": res.plan["plan_sha256"], "cells": [c.cell for c in cells]})
    else:
        log.info("d00_plan_resumed", extra_fields={"plan_sha256": res.plan["plan_sha256"], "cells": [c.cell for c in cells]})
    if dependency_only:
        res.summary = {"mode": "dependency_only"}
        _write_text(share / "README_D00.md", R.readme(res, WATERMARK, STATUS_LINES, dependency_only=True))
        res.privacy = _privacy(res, share, read.frame, contract, pepper, ref, out)
        _write_json(share / "d00_manifest.json", _manifest(res, config, mapping, dictionary, input_sha, pinfo, min_cell))
        _raise_on_privacy(res)
        return

    # ---- 4-5. datasets, verification, runs
    t = time.perf_counter()
    template = Path(template_path) if template_path else ref["configs"]["extended"]
    runs = _build_and_run(res, cells, graph, config, ref, read, spec, mapping, contract, pepper, pinfo, input_sha, src, out, share,
                          template=template, resampling=resampling, cohorts=cohorts, resume=stored_plan is not None)
    res.seconds["runs"] = round(time.perf_counter() - t, 1)

    # ---- 6. records and comparisons (test predictions read only now, after every run of the frozen plan completed)
    t = time.perf_counter()
    fractions = tuple(float(x) for x in facts["risk_fractions"])
    preds = {k: _load_predictions(v.get("run_dir")) for k, v in runs.items()}
    for c in cells:
        res.records[c.cell] = _record(c, runs, preds)
    ref_keys = set(runs[REF_EXTENDED]["keys"])
    comparisons = _comparisons(res, cells, runs, preds, ref_keys, n_boot=n_boot, min_cell=min_cell)
    risk = []
    for c in cells:
        if c.source in ("run", "reference:extended", "reference:strict") and preds.get(c.cell):
            p = preds[c.cell]
            risk.append(A.risk_concentration(p["y"], p["p"], fractions=fractions, min_cell=min_cell, n_boot=n_boot, label=c.cell).assign(
                cohort=c.cohort, served_variant=p["variant"], alias_cells=", ".join(x.cell for x in cells if x.alias_of == c.cell)))
    risk_df = pd.concat(risk, ignore_index=True) if risk else pd.DataFrame()
    res.seconds["comparisons"] = round(time.perf_counter() - t, 1)

    # ---- learning curves + warnings audit (TRAIN / VALIDATION only)
    t = time.perf_counter()
    lc = _learning_curves(res, ref, config, mode=learning_curve, reports_dir=Path(reports_dir) if reports_dir else None, n_boot=n_boot)
    audit = _warnings_audit(res, ref, runs, lc)
    res.seconds["learning_and_audit"] = round(time.perf_counter() - t, 1)

    # ---- 7. outputs
    res.summary = {"cohort_counts": graph.cohort_counts, "comparisons": comparisons,
                   "learning_curve": {k: {kk: vv for kk, vv in v.items() if kk not in ("table", "frames")} for k, v in lc.items()},
                   "warnings_audit": {k: v for k, v in audit.items() if k != "_raw"}, "risk_concentration": risk_df.to_dict(orient="records"),
                   "population": _population_facts(res, runs, cohorts, read.frame, mapping, min_cell)}
    for name, df, basis in [("sensitivity_matrix", R.matrix_table(res), "HELD-OUT TEST of each frozen model"),
                            ("ablation_paired", pd.DataFrame(comparisons["ablations"]), "HELD-OUT TEST, identical rows (paired)"),
                            ("feature_set_effects_paired", pd.DataFrame(comparisons["feature_set_effects"]), "HELD-OUT TEST, identical rows (paired)"),
                            ("population_effects_unpaired", pd.DataFrame(comparisons["population_effects"]), "HELD-OUT TEST of each model - NOT PAIRED"),
                            ("population_d00_decomposition", pd.DataFrame(comparisons["d00_decomposition"]), "HELD-OUT TEST subsets of FULL_LABELED models"),
                            ("risk_concentration", risk_df, "HELD-OUT TEST (exploratory operational characterisation; no threshold chosen)")]:
        _csv(df, share / "tables" / f"{name}.csv", basis)
    for k, v in lc.items():
        if v.get("table") is not None:
            _csv(v["table"], share / "tables" / f"learning_curve_{k}.csv", "TRAIN subsets / VALIDATION (test never read)")
        if v.get("increments"):
            _csv(pd.DataFrame(v["increments"]), share / "tables" / f"learning_curve_increments_{k}.csv", "VALIDATION (paired bootstrap)")
    from falls_ml.d00 import warnings_audit as W

    _write_json(share / "LASSO_WARNINGS_AUDIT.json", {"watermark": WATERMARK, **{k: v for k, v in audit.items() if k != "_raw"}})
    _write_text(share / "LASSO_WARNINGS_AUDIT.md", W.render_markdown(audit, WATERMARK, STATUS_LINES))
    figs = R.figures(res, share / "figures", risk_df, lc)
    d00_json = R.summary_json(res, WATERMARK, STATUS_LINES, risk_df, figs)
    _write_json(share / "d00_sensitivity.json", d00_json)
    _write_text(share / "D00_SENSITIVITY_REPORT.html", R.render_html(res, WATERMARK, STATUS_LINES, risk_df, lc, audit, figs))
    _write_text(share / "D00_SENSITIVITY_SUMMARY_HE.md", R.render_summary_he(res, d00_json))
    _write_text(share / "README_D00.md", R.readme(res, WATERMARK, STATUS_LINES, dependency_only=False))
    if management_report:
        from falls_ml.modelreport.runner import build_model_reports

        try:
            rs = build_model_reports([res.reference_dir, out], out_dir=share / "management", learning_curve="off", own_results=out, n_boot=n_boot,
                                     min_cell=min_cell, d00=d00_json)
            res.management = {"status": "built", "dir": str(share / "management"), "primary_analysis": rs.manifest.get("primary_analysis")}
        except Exception as exc:  # noqa: BLE001 - the analysis stays valid; the failure is recorded next to it
            _write_text(share / "management" / "REPORT_FAILED.txt", f"management report failed: {type(exc).__name__}: {exc}\n")
            res.management = {"status": f"FAILED: {type(exc).__name__}: {exc}"}
    if earlier is not None:    # resume: provenance + integrity of the earlier attempt, written before the identifier scan so it is scanned too
        res.resume["earlier_attempt_integrity"] = _earlier_check(out, earlier)
        res.resume["earlier_attempt_files_unchanged"] = res.resume["earlier_attempt_integrity"]["unchanged"]
        res.resume["completed"] = True
        _write_resume_log(share, prior_attempts or [], res.resume)
    res.privacy = _privacy(res, share, read.frame, contract, pepper, ref, out)
    _write_json(share / "d00_manifest.json", _manifest(res, config, mapping, dictionary, input_sha, pinfo, min_cell))
    _raise_on_privacy(res)


def _raise_on_privacy(res: D00Result) -> None:
    if not res.privacy.get("passed", False):
        raise LeakageError("identifier values found in the shareable D-00 outputs; do NOT share the folder", res.privacy.get("hits"))


# ============================================================================ datasets and runs
def _config_for(template: Path, out: Path, cell: Cell, spec: Any, description: str, resampling: str, *, path: Path | None = None) -> Path:
    from falls_ml.meuhedet_explore import _write_config

    p = _write_config(template, path or (out / "configs" / f"{cell.label}.yaml"), name=str(cell.label), description=description, features=cell.features,
                      fast=False, spec=spec)
    if resampling == "reduced":
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        cfg["analysis"]["stability"]["n_bootstrap"] = RESAMPLING_REDUCED["stability_n"]
        cfg["analysis"]["optimism"]["n_bootstrap"] = RESAMPLING_REDUCED["optimism_n"]
        cfg["evaluation"]["permutation_importance_repeats"] = RESAMPLING_REDUCED["permutation_repeats"]
        cfg["experiment"]["description"] += " RESAMPLING REDUCED (stability 50, optimism 10, permutation 5; CIs unchanged)."
        p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8", newline="\n")
    return p


def _build_and_run(res: D00Result, cells: list[Cell], graph: Graph, config: Any, ref: dict[str, Any], read: Any, spec: Any, mapping: Any, contract: Any,
                   pepper: str, pinfo: dict[str, Any], input_sha: str, src: Path, out: Path, share: Path, *, template: Path, resampling: str,
                   cohorts: Any, resume: bool = False) -> dict[str, dict[str, Any]]:
    from falls_ml.adequacy import assess_adequacy, write_adequacy
    from falls_ml.config import load_experiment_config
    from falls_ml.data.dataset import ModelingDataset
    from falls_ml.data.meuhedet_wide import build_meuhedet_dataset_from_frame
    from falls_ml.experiment import run_experiment
    from falls_ml.meuhedet_explore import render_split_audit, split_audit
    from falls_ml.meuhedet_sensitivity import _differing_columns, _keys
    from falls_ml.splitting import make_split

    wm = WATERMARK
    runs: dict[str, dict[str, Any]] = {}
    ref_frame = pd.read_parquet(ref["dir"] / "datasets" / "extended" / "modeling_dataset.parquet")
    ref_keys = set(_keys(ref_frame, spec))
    ref_test = ref["split_audit"]["test_rows_sha256"]
    for key, cell in (("extended", REF_EXTENDED), ("strict", REF_STRICT)):
        runs[cell] = {"run_dir": ref["runs"][key], "metrics": ref["metrics"][key], "keys": sorted(ref_keys), "test_rows_sha256": ref["metrics"][key]["split"]["test_rows_sha256"],
                      "n_rows": ref["build"]["n_rows_final"], "n_patients": ref["build"]["n_patients_final"], "n_events": ref["build"]["n_events"],
                      "prevalence": ref["build"]["outcome_prevalence"], "status": "REUSED (completed reference run, not retrained)", "build": ref["build"]}
    full_keys: set[str] | None = None
    full_test: str | None = None
    version_base = f"meuhedet-wide-{res.index_date}-{src.stem}-d00"
    inventory = _run_inventory(out) if resume else None
    if inventory is not None:
        from falls_ml import __version__
        from falls_ml.artifacts import source_tree_sha256

        label_to_cell = {str(c.label): c.cell for c in cells if c.source == "run"}
        res.resume = {**res.resume, "resumed": True, "plan_sha256": res.plan["plan_sha256"], "falls_ml_version_now": __version__, "source_tree_sha256_now": source_tree_sha256(),
                      "cells": [],
                      "incomplete_attempts": [{"run_id": e["run_id"], "cell": label_to_cell.get(lbl, lbl), "test_set_released": bool(e["releases"]),
                                               "failure": e["failure"], "handling": "kept on disk for audit; never read as a result"}
                                              for lbl, es in inventory["by_label"].items() for e in es if not e["complete"]]
                                             + [{"run_id": e["run_id"], "cell": None, "test_set_released": bool(e["releases"]), "failure": e["failure"],
                                                 "handling": "kept on disk for audit; never read as a result"} for e in inventory["unidentified"]],
                      "completed_runs_not_in_plan": sorted(e["run_id"] for lbl, es in inventory["by_label"].items() if lbl not in label_to_cell
                                                          for e in es if e["complete"]),
                      "rule": "completed runs of the interrupted attempt are reused as they are (not refitted; test set not re-evaluated); an incomplete "
                              "run is never read as a result and its cell is refitted only if it never released its test set"}
    for cell in cells:
        if cell.source != "run":
            continue
        standard = cell.standard or config.primary_standard
        if cell.cohort == "D00_CLEAN":
            scope, policy, forbidden = "cohort", "drop_rows", None
        else:
            scope, policy, forbidden = "built_predictors", "fail", graph.forbidden_columns[standard]["FULL_LABELED"]
        ds_dir = out / "datasets" / cell.cell
        reuse_ds = resume and ds_dir.is_dir() and any(ds_dir.iterdir())
        check_root = Path(tempfile.mkdtemp(prefix=".resume_check_", dir=out)) if reuse_ds else None
        try:
            manifest, report = build_meuhedet_dataset_from_frame(
                read.frame, (check_root / cell.cell) if check_root is not None else ds_dir, index_date=res.index_date, dataset_version=f"{version_base}-{cell.cell}", data_freeze_date=None, mapping=mapping,
                contract=contract, spec=spec, input_name=src.name, input_sha256=input_sha, wrong_type=read.wrong_type,
                wrong_type_examples=read.wrong_type_examples, date_formats=read.date_formats, features=cell.features, feature_set_label=cell.cell,
                id_pepper=pepper, index_day_records=policy, read_report=read.to_dict(), notes=f"{wm}; D-00 sensitivity cell {cell.cell}: {cell.text}",
                timing_scope=scope, forbidden_columns=forbidden)
        except DatasetValidationError as exc:
            raise DatasetValidationError(f"{cell.cell}: the dataset cannot be built (the D-00 plan and the extract disagree)", exc.problems) from None
        finally:
            if check_root is not None:
                shutil.rmtree(check_root, ignore_errors=True)
        if reuse_ds:   # resume: the earlier attempt's dataset is used as it is, after the rebuild proved it identical
            earlier_sha = ModelingDataset.load(ds_dir, spec, features=cell.features).manifest.data_sha256
            if earlier_sha != manifest.data_sha256:
                raise DatasetValidationError(f"--resume: {cell.cell}: the rebuilt dataset differs from the earlier attempt's (data sha256); nothing was reused",
                                             [f"earlier {earlier_sha}", f"rebuilt {manifest.data_sha256}"])
        frame = ModelingDataset.load(ds_dir, spec, features=cell.features).frame
        keys = set(_keys(frame, spec))
        checks: dict[str, Any] = {}
        if cell.cohort == "D00_CLEAN":
            if keys != ref_keys:
                raise DatasetValidationError(f"{cell.cell}: the rebuilt D00_CLEAN cohort differs from the reference cohort",
                                             [f"reference {len(ref_keys)}", f"rebuilt {len(keys)}", f"only in reference {len(ref_keys - keys)}", f"only rebuilt {len(keys - ref_keys)}"])
            differing = _differing_columns(ref_frame, frame, spec, cell.features)
            if differing:
                raise DatasetValidationError(f"{cell.cell}: rebuilt predictor values differ from the reference dataset", differing)
            checks = {"rows_equal_reference": True, "predictor_values_equal_reference": True}
        else:
            if report.n_rows_timing_violation:
                raise LeakageError(f"{cell.cell}: FULL_LABELED dropped rows for D-00 - it must not ({report.timing_violations_by_column})")
            if len(keys) != cohorts.counts["FULL_LABELED"]:
                raise DatasetValidationError(f"{cell.cell}: FULL_LABELED has {len(keys)} rows, the cohort definition gives {cohorts.counts['FULL_LABELED']}")
            if not ref_keys <= keys or len(keys - ref_keys) != cohorts.counts["d00_rows_in_full_labeled"]:
                raise DatasetValidationError(f"{cell.cell}: FULL_LABELED must contain every reference row plus exactly the D-00 rows",
                                             [f"missing reference rows {len(ref_keys - keys)}", f"added {len(keys - ref_keys)}", f"D-00 rows {cohorts.counts['d00_rows_in_full_labeled']}"])
            if full_keys is not None and keys != full_keys:
                raise DatasetValidationError(f"{cell.cell}: the FULL_LABELED cells do not share one set of rows")
            full_keys = keys
            checks = {"rows_equal_cohort_definition": True, "d00_rows_kept": int(len(keys - ref_keys)), "rows_dropped_for_d00": 0,
                      "cohort_guard_rows_retained": report.cohort_guard_rows_retained, "forbidden_columns_enforced": len(forbidden or [])}
        description = f"{wm}. D-00 sensitivity cell {cell.cell} ({cell.cohort}, {cell.feature_set}): {len(cell.features)} predictors. {cell.text}"
        cfg_path = out / "configs" / f"{cell.label}.yaml"
        if resume and cfg_path.is_file():   # the earlier attempt's config is used as it is, after regeneration proved it byte-identical
            with tempfile.TemporaryDirectory(prefix=".resume_check_", dir=out) as tmp:
                again = _config_for(template, out, cell, spec, description, resampling, path=Path(tmp) / cfg_path.name).read_bytes()
            if again != cfg_path.read_bytes():
                raise DatasetValidationError(f"--resume: {cell.cell}: the regenerated experiment config differs from the earlier attempt's "
                                             f"({cfg_path.name}); nothing was reused")
        else:
            cfg_path = _config_for(template, out, cell, spec, description, resampling)
        cfg = load_experiment_config(cfg_path)
        sub = spec.subset(cell.features)
        plan = make_split(frame, sub, cfg.validation)
        sa = split_audit(frame, sub, plan, strategy_note=f"D-00 cell {cell.cell}",
                         reproducibility={"input_sha256": input_sha, "pepper_sha256": pinfo["sha256"], "mapping_sha256": mapping.content_sha256,
                                          "config_sha256": cfg.sha256(), "dataset_sha256": manifest.data_sha256, "reference_test_rows_sha256": ref_test,
                                          "split_seed": cfg.validation.seed, "template": template.name, "plan_sha256": res.plan["plan_sha256"]})
        _write_json(share / "split" / f"split_audit_{cell.cell}.json", sa)
        _write_text(share / "split" / f"split_audit_{cell.cell}.md", render_split_audit(sa))
        if cell.cohort == "D00_CLEAN" and sa["test_rows_sha256"] != ref_test:
            raise LeakageError(f"{cell.cell}: the split does not reproduce the reference test partition ({sa['test_rows_sha256']} vs {ref_test})")
        if cell.cohort == "FULL_LABELED":
            if full_test is not None and sa["test_rows_sha256"] != full_test:
                raise LeakageError(f"{cell.cell}: FULL_LABELED cells must share one test partition")
            full_test = sa["test_rows_sha256"]
        adequacy = assess_adequacy(frame, sub, plan, cfg)
        write_adequacy(adequacy, share / "adequacy", stem=f"adequacy_{cell.cell}")
        entry = {"keys": sorted(keys), "test_rows_sha256": sa["test_rows_sha256"], "n_rows": report.n_rows_final, "n_patients": report.n_patients_final,
                 "n_events": report.n_events, "prevalence": report.outcome_prevalence, "checks": checks, "build": report.to_dict()}
        if not adequacy.ready:
            runs[cell.cell] = {**entry, "status": "NOT RUN - INSUFFICIENT EVENTS FOR EXPLORATORY MODEL FIT: " + "; ".join(adequacy.reasons)}
            continue
        incomplete: list[dict[str, Any]] = []
        if inventory is not None:
            found = inventory["by_label"].get(str(cell.label), [])
            complete = [e for e in found if e["complete"]]
            incomplete = [e for e in found if not e["complete"]]
            if len(complete) > 1:
                raise DatasetValidationError(f"--resume: {cell.cell} has {len(complete)} completed runs ({[e['run_id'] for e in complete]}); "
                                             "which one is the result is ambiguous - nothing was reused")
            if complete:
                e = complete[0]
                metrics = json.loads((e["dir"] / "metrics.json").read_text(encoding="utf-8"))
                problems = _completed_run_problems(e, metrics, label=str(cell.label), config_sha=cfg.sha256(), data_sha=manifest.data_sha256,
                                                   test_sha=sa["test_rows_sha256"])
                if problems:
                    raise DatasetValidationError(f"--resume: the completed run {e['run_id']} of {cell.cell} does not match the frozen specification; "
                                                 "nothing was reused", problems)
                runs[cell.cell] = {**entry, "run_dir": e["dir"], "metrics": metrics, "status": STATUS_REUSED_EARLIER}
                res.resume["cells"].append({"cell": cell.cell, "action": "REUSED_COMPLETED_RUN", "run_id": e["run_id"],
                                            "incomplete_runs_ignored": [x["run_id"] for x in incomplete],
                                            "source_tree_sha256_of_run": (metrics.get("code_version") or {}).get("source_tree_sha256"),
                                            "bootstrap_convergence_retry_policy_in_effect": "bootstrap_convergence_retry" in metrics})
                log.info("d00_cell_reused", extra_fields={"cell": cell.cell, "run_id": e["run_id"]})
                continue
            released = [x["run_id"] for x in incomplete if x["releases"]]
            if released:
                raise LeakageError(f"--resume: {cell.cell}: the incomplete earlier run(s) {released} had already released the test set (registry "
                                   "entry); refitting would evaluate those test rows again - stopped, nothing was refitted. Decide explicitly how to proceed.")
        result = run_experiment(None, ds_dir, cfg_path, runs_dir=out / "runs")
        if result.metrics["split"]["test_rows_sha256"] != sa["test_rows_sha256"]:
            raise LeakageError(f"{cell.cell}: the run's test rows differ from the split audit computed before training")
        runs[cell.cell] = {**entry, "run_dir": Path(result.run_dir), "metrics": result.metrics, "status": STATUS_NEW}
        if inventory is not None:
            res.resume["cells"].append({"cell": cell.cell, "action": ("REFITTED (the earlier attempt's run was incomplete and never released its test set)"
                                                                      if incomplete else "RUN (not reached by the earlier attempt)"),
                                        "run_id": result.run_id, "incomplete_runs_ignored": [x["run_id"] for x in incomplete],
                                        "earlier_failures": [x["failure"] for x in incomplete],
                                        "bootstrap_convergence_retry_policy_in_effect": "bootstrap_convergence_retry" in result.metrics,
                                        "bootstrap_convergence_retry": {k: v for k, v in (result.metrics.get("bootstrap_convergence_retry") or {}).items()
                                                                        if k != "records"}})
        log.info("d00_cell_done", extra_fields={"cell": cell.cell, "run_id": result.run_id})
    return runs


def _completed_run_problems(e: dict[str, Any], metrics: dict[str, Any], *, label: str, config_sha: str, data_sha: str, test_sha: str) -> list[str]:
    """Why a completed run of the earlier attempt may NOT be reused for this cell (empty = it is the frozen cell's result)."""
    problems = []
    if (metrics.get("experiment") or {}).get("name") != label:
        problems.append(f"experiment name {(metrics.get('experiment') or {}).get('name')!r} is not {label!r}")
    if (metrics.get("split") or {}).get("test_rows_sha256") != test_sha:
        problems.append("its test partition differs from the frozen split")
    if (metrics.get("dataset") or {}).get("data_sha256") != data_sha:
        problems.append("its dataset differs from the rebuilt dataset")
    if len(e["releases"]) != 1:
        problems.append(f"the test-evaluation registry records {len(e['releases'])} test releases for it (exactly 1 expected)")
    else:
        r = e["releases"][0]
        if r.get("config_sha256") != config_sha:
            problems.append("its registered config sha256 differs from the regenerated config")
        if r.get("test_rows_sha256") != test_sha or r.get("data_sha256") != data_sha:
            problems.append("its registered test rows / data differ from the frozen cell")
    return problems


def _load_predictions(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    p = run_dir / "predictions_test.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    key = df["research_id"].astype(str) + "|" + pd.to_datetime(df["index_date"]).dt.strftime("%Y-%m-%d")
    df = df.assign(_k=key).sort_values("_k").reset_index(drop=True)
    variant = "risk_served"
    return {"keys": df["_k"].to_numpy(), "y": df["outcome"].to_numpy(dtype=np.int64), "p": df[variant].to_numpy(dtype=float),
            "p_uncal": df["risk_uncalibrated"].to_numpy(dtype=float),
            "p_recal": df["risk_recalibrated"].to_numpy(dtype=float) if "risk_recalibrated" in df and df["risk_recalibrated"].notna().all() else None,
            "variant": "served"}


def _record(c: Cell, runs: dict[str, dict[str, Any]], preds: dict[str, Any]) -> dict[str, Any]:
    base = {k: v for k, v in asdict(c).items()}
    serving = c.cell if c.source in ("run", "reference:extended", "reference:strict") else (c.alias_of if c.source == "alias" else None)
    if serving is None or serving not in runs:
        return {**base, "status": "NOT RUN - " + (c.not_run_reason or "not in the plan's run list"), "serving_cell": None}
    r = runs[serving]
    rec = {**base, "serving_cell": serving, "status": r["status"] if serving == c.cell else f"SAME RUN AS {serving} (identical cohort and predictors)",
           "n_rows": r.get("n_rows"), "n_patients": r.get("n_patients"), "n_events": r.get("n_events"), "prevalence": r.get("prevalence"),
           "test_rows_sha256": r.get("test_rows_sha256"), "checks": r.get("checks", {})}
    m = r.get("metrics")
    if not m:
        return rec
    served = m.get("served_variant") or m.get("primary_variant") or "uncalibrated"
    rec.update({"run_id": Path(r["run_dir"]).name, "served_variant": served, "n_predictors": len(c.features),
                "selected_raw_features": _selected(Path(r["run_dir"])), "lambda_star": (m.get("lasso") or {}).get("lambda_star")})
    for variant, suffix in ((served, ""), ("uncalibrated", "_uncalibrated"), ("recalibrated", "_recalibrated")):
        perf = _perf(m, variant)
        for metric in ("auroc", "pr_auc", "brier", "calibration_slope", "calibration_intercept", "citl", "oe_ratio"):
            e, lo, hi = _est(perf.get(metric))
            rec[f"{metric}{suffix}"], rec[f"{metric}{suffix}_ci"] = e, (None if lo is None else [lo, hi])
        if not suffix:
            rec["test_n"], rec["test_events"] = perf.get("n"), perf.get("n_events")
    return rec


def _selected(run_dir: Path) -> list[str]:
    from falls_ml.meuhedet_explore import _selected_raw_features

    try:
        return list(_selected_raw_features(run_dir))
    except Exception:  # noqa: BLE001
        return []


# ============================================================================ comparisons
def _paired_ok(pa: dict[str, Any], pb: dict[str, Any]) -> bool:
    return bool(pa) and bool(pb) and len(pa["keys"]) == len(pb["keys"]) and bool((pa["keys"] == pb["keys"]).all()) and bool((pa["y"] == pb["y"]).all())


def _variant_pair(pa: dict[str, Any], pb: dict[str, Any], ra: dict[str, Any], rb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    """Probabilities of the same variant for both models (the served one if both serve the same variant, else recalibrated, else uncalibrated)."""
    if ra.get("served_variant") == rb.get("served_variant"):
        return pa["p"], pb["p"], str(ra.get("served_variant"))
    if pa.get("p_recal") is not None and pb.get("p_recal") is not None:
        return pa["p_recal"], pb["p_recal"], "recalibrated"
    return pa["p_uncal"], pb["p_uncal"], "uncalibrated"


def _paired_row(kind: str, a: str, b: str, res: D00Result, preds: dict[str, Any], n_boot: int, min_cell: int, question: str | None = None,
                text: str = "") -> dict[str, Any]:
    ra, rb = res.records[a], res.records[b]
    pa, pb = preds.get(ra["serving_cell"]), preds.get(rb["serving_cell"])
    row: dict[str, Any] = {"comparison_type": kind, "question": question, "question_text": text, "reference": a, "compared": b,
                           "cohort": ra.get("cohort"), "predictors_removed": sorted(set(ra.get("features") or []) - set(rb.get("features") or [])),
                           "predictors_added": sorted(set(rb.get("features") or []) - set(ra.get("features") or []))}
    if not (pa and pb) or not _paired_ok(pa, pb):
        return {**row, "paired": False, "note": "NOT A DIRECT PAIRED MODEL COMPARISON: the test rows are not identical (verified by keys)"}
    x, y_, variant = _variant_pair(pa, pb, ra, rb)
    d = A.paired_comparison(pa["y"], x, y_, n_boot=n_boot, label=f"{a}->{b}")
    row.update({"paired": True, "note": "paired: identical held-out test rows (keys and outcomes verified)", "variant_for_brier": variant, "n_test_rows": d["n_rows"]})
    for side, r in (("reference", ra), ("compared", rb)):
        for m in ("auroc", "pr_auc", "brier", "calibration_slope", "citl", "oe_ratio"):
            row[f"{side}_{m}"], row[f"{side}_{m}_ci"] = r.get(m), r.get(f"{m}_ci")
        row[f"{side}_calibration_slope_uncalibrated"], row[f"{side}_citl_uncalibrated"], row[f"{side}_oe_uncalibrated"] = (
            r.get("calibration_slope_uncalibrated"), r.get("citl_uncalibrated"), r.get("oe_ratio_uncalibrated"))
    for m in ("auroc", "pr_auc", "brier", "capture_top5", "capture_top10", "capture_top20"):
        row[f"delta_{m}"], row[f"delta_{m}_ci_low"], row[f"delta_{m}_ci_high"] = d[m]["estimate"], d[m]["ci_low"], d[m]["ci_high"]
    for side, p in (("reference", x), ("compared", y_)):
        rc = A.risk_concentration(pa["y"], p, fractions=(0.05, 0.10, 0.20), min_cell=min_cell, n_boot=0)
        for _, r in rc.iterrows():
            tp = int(round(float(r["top_pct"])))
            row[f"{side}_top{tp}_capture_pct"], row[f"{side}_top{tp}_lift"] = r["pct_of_all_falls_captured"], r["lift"]
    return row


def _comparisons(res: D00Result, cells: list[Cell], runs: dict[str, dict[str, Any]], preds: dict[str, Any], ref_keys: set[str], *, n_boot: int,
                 min_cell: int) -> dict[str, Any]:
    by = {c.cell: c for c in cells}
    served = [c for c in cells if res.records[c.cell].get("serving_cell") and res.records[c.cell].get("auroc") is not None]
    ablations = [_paired_row("FEATURE-SET EFFECT (pre-specified ablation; paired, identical test rows)", REF_EXTENDED, c.cell, res, preds, n_boot, min_cell,
                             c.question, c.text) for c in cells if c.kind == "ablation" and c.cell in {s.cell for s in served}]
    fse: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for c in served:   # every other D00_CLEAN cell against the reference, on the reference test rows
        if c.cohort == "D00_CLEAN" and c.kind != "ablation" and c.cell != REF_EXTENDED:
            s = (res.records[REF_EXTENDED]["serving_cell"], res.records[c.cell]["serving_cell"])
            if s[0] != s[1] and s not in seen:
                seen.add(s)
                fse.append(_paired_row("FEATURE-SET EFFECT (paired, identical test rows)", REF_EXTENDED, c.cell, res, preds, n_boot, min_cell))
    full = [c for c in served if c.cohort == "FULL_LABELED" and c.source == "run"]
    for i, a in enumerate(full):
        for b in full[i + 1:]:
            lo, hi = (a, b) if len(a.features) >= len(b.features) else (b, a)
            fse.append(_paired_row("FEATURE-SET EFFECT (paired, identical FULL_LABELED test rows)", lo.cell, hi.cell, res, preds, n_boot, min_cell))
    pop: list[dict[str, Any]] = []
    pop_seen: set[tuple[str, str]] = set()
    for c in served:   # same predictors, other population: never paired
        if c.cohort != "FULL_LABELED":
            continue
        partner = next((x for x in served if x.cohort == "D00_CLEAN" and x.feature_set == c.feature_set and x.standard == c.standard
                        and sorted(x.features) == sorted(c.features)), None)
        if partner is None:
            partner = next((x for x in served if x.cohort == "D00_CLEAN" and sorted(x.features) == sorted(c.features)), None)
        if partner is None:
            continue
        ra, rb = res.records[partner.cell], res.records[c.cell]
        pair = (ra["serving_cell"], rb["serving_cell"])
        if pair in pop_seen:
            continue
        pop_seen.add(pair)
        pop.append({"comparison_type": "POPULATION EFFECT - NOT A DIRECT PAIRED MODEL COMPARISON", "predictors": c.feature_set,
                    "n_predictors": len(c.features), "d00_clean_cell": partner.cell, "full_labeled_cell": c.cell,
                    "d00_clean_rows": ra.get("n_rows"), "full_labeled_rows": rb.get("n_rows"), "d00_clean_prevalence": ra.get("prevalence"),
                    "full_labeled_prevalence": rb.get("prevalence"), "d00_clean_auroc": ra.get("auroc"), "d00_clean_auroc_ci": ra.get("auroc_ci"),
                    "full_labeled_auroc": rb.get("auroc"), "full_labeled_auroc_ci": rb.get("auroc_ci"),
                    "difference_of_estimates": (rb["auroc"] - ra["auroc"]) if (ra.get("auroc") is not None and rb.get("auroc") is not None) else None,
                    "d00_clean_brier": ra.get("brier"), "full_labeled_brier": rb.get("brier"), "d00_clean_oe": ra.get("oe_ratio"), "full_labeled_oe": rb.get("oe_ratio"),
                    "note": "same predictors on two different populations: different rows, a redrawn split and different test partitions - the "
                            "difference is NOT a paired estimate and its uncertainty is larger than either interval suggests"})
    dec: list[dict[str, Any]] = []
    for c in full:
        p = preds.get(c.cell)
        if not p:
            continue
        in_clean = np.array([k in ref_keys for k in p["keys"]])
        for part, mask in (("test rows that belong to the D00_CLEAN cohort", in_clean), ("test rows that are D-00 rows (excluded from the reference)", ~in_clean)):
            dec.append({"model": c.cell, "subset": part, **A.subset_performance(p["y"], p["p"], mask, min_cell=min_cell),
                        "note": "one FULL_LABELED model evaluated on two subsets of ITS OWN test partition (descriptive; post-hoc; not used for any decision)"})
    share: dict[str, Any] = {}
    pe, ps = preds.get(res.records[REF_EXTENDED]["serving_cell"]), preds.get(res.records[REF_STRICT]["serving_cell"])
    for c in cells:
        if c.kind == "ablation" and res.records[c.cell].get("serving_cell"):
            pa = preds.get(res.records[c.cell]["serving_cell"])
            if pe and ps and pa and _paired_ok(pe, ps) and _paired_ok(pe, pa):
                share[c.cell] = A.share_of_gain(pe["y"], ps["p"], pe["p"], pa["p"], n_boot=n_boot, label=c.cell)
    return {"ablations": ablations, "feature_set_effects": fse, "population_effects": pop, "d00_decomposition": dec, "share_of_strict_to_extended_gain": share,
            "rule": "paired only when the held-out test keys and outcomes are identical (verified); otherwise labelled NOT A DIRECT PAIRED MODEL COMPARISON",
            "cells": list(by)}


def _population_facts(res: D00Result, runs: dict[str, dict[str, Any]], cohorts: Any, frame: pd.DataFrame, mapping: Any, min_cell: int) -> dict[str, Any]:
    """Counts and outcome prevalence of FULL_LABELED, D00_CLEAN and the D-00 rows (descriptive, post-hoc: computed after the plan was frozen,
    never used to define a cohort, a feature set or a model)."""
    y = frame[mapping.outcome["label_column"]]
    out: dict[str, Any] = {}
    full, clean = cohorts["FULL_LABELED"], cohorts["D00_CLEAN"]
    raw: dict[str, tuple[int, int]] = {}
    for name, m in (("FULL_LABELED", full), ("D00_CLEAN", clean), ("D00_ROWS", full & ~clean)):
        n = int(m.sum())
        e = int(pd.to_numeric(y[m], errors="coerce").fillna(0).sum())
        raw[name] = (e, n)
        out[name] = {"n_rows": n, "n_events": A.cell(e, min_cell), "prevalence": None if (0 < e < min_cell) else A._r(e / n if n else None, 5)}
    (e1, n1), (e2, n2) = raw["D00_ROWS"], raw["D00_CLEAN"]
    out["d00_rows_minus_clean"] = A.two_proportions(e1, n1, e2, n2) if (e1 == 0 or e1 >= min_cell) else {"difference": None, "ci": [None, None], "differs": None,
                                                                                                        "note": "suppressed: fewer than min_cell events"}
    out["note"] = "descriptive, post-hoc (after the plan was frozen); the D-00 rows' prevalence is reported because their exclusion may bias the reference cohort"
    return out


# ============================================================================ learning curves + warnings audit
def _learning_curves(res: D00Result, ref: dict[str, Any], config: Any, *, mode: str, reports_dir: Path | None, n_boot: int) -> dict[str, dict[str, Any]]:
    from falls_ml.modelreport import compute as C
    from falls_ml.modelreport.artifacts import discover_runs

    intervals = [tuple(float(x) for x in pair) for pair in (config.plan.get("learning_curve_intervals") or [[0.4, 0.6], [0.6, 0.8], [0.8, 1.0]])]
    max_gain = float(config.plan.get("plateau_max_gain", 0.01))
    out: dict[str, dict[str, Any]] = {}
    runs = {r.key: r for r in discover_runs([ref["dir"]]) if r.key in ("STRICT", "EXTENDED")}
    for key in ("STRICT", "EXTENDED"):
        entry: dict[str, Any] = {"model": key, "mode": mode}
        existing = reports_dir / "tables" / f"{key}_learning_curve.csv" if reports_dir else None
        old = pd.read_csv(existing) if existing is not None and existing.exists() else None
        if mode in ("full", "reduced") and key in runs:
            try:
                frames = C.load_frames(runs[key], [ref["dir"]])
                preds: dict[float, np.ndarray] = {}
                lc, info = C.learning_curve(runs[key], frames, mode=mode, n_boot=200, predictions_out=preds)
                yv = frames.validation[frames.spec.outcome.name].to_numpy(dtype=np.int64)
                entry.update({"table": lc, "info": {k: v for k, v in info.items()}, "frames": frames})
                entry["increments"] = A.learning_increments(yv, preds, intervals, n_boot=n_boot, label=key)
                if old is not None and "validation_auroc" in old:
                    m = old.merge(lc, on="fraction", suffixes=("_existing", "_now"))
                    diff = (pd.to_numeric(m["validation_auroc_existing"], errors="coerce") - pd.to_numeric(m["validation_auroc_now"], errors="coerce")).abs()
                    entry["matches_existing_report_table"] = bool(len(m) and diff.max() < 1e-3 and str(old["mode"].iloc[0]) == mode)
                    entry["existing_report_mode"] = str(old["mode"].iloc[0]) if "mode" in old else None
            except Exception as exc:  # noqa: BLE001 - recorded, never silent
                entry["error"] = f"{type(exc).__name__}: {exc}"
        elif mode == "reuse" and old is not None:
            entry.update({"table": old, "increments": _increments_from_table(old, intervals),
                          "note": "reused the existing 0.6.0 report table: increments are differences of point estimates (no paired interval) "
                                  "and the refits' warnings were not recorded"})
        else:
            entry["note"] = "learning curve not computed (mode off or no table found)"
        if entry.get("increments"):
            entry["plateau"] = A.plateau_statement(entry["increments"], max_gain=max_gain, name=key)
        t = entry.get("table")
        if t is not None and len(t):
            entry["by_fraction"] = t[[c for c in ("fraction", "n_train", "n_train_events", "train_auroc_apparent", "validation_auroc", "validation_auroc_ci_low",
                                                  "validation_auroc_ci_high", "train_brier_apparent", "validation_brier", "status") if c in t.columns]].to_dict(orient="records")
        out[key] = entry
    return out


def _increments_from_table(t: pd.DataFrame, intervals: list[tuple[float, ...]]) -> list[dict[str, Any]]:
    v = dict(zip(t["fraction"].round(2), pd.to_numeric(t["validation_auroc"], errors="coerce")))
    return [{"from_fraction": a, "to_fraction": b, "delta_validation_auroc": A._r(v.get(round(b, 2)) - v.get(round(a, 2)), 4)
             if v.get(round(a, 2)) is not None and v.get(round(b, 2)) is not None else None, "ci_low": None, "ci_high": None, "paired": False}
            for a, b in intervals]


def _warnings_audit(res: D00Result, ref: dict[str, Any], runs: dict[str, dict[str, Any]], lc: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from falls_ml.d00 import warnings_audit as W

    final, reps, sens, lcs, mat = {}, {}, {}, {}, {}
    for key, fs in (("EXTENDED", "extended"), ("STRICT", "strict")):
        f = W.final_fit_summary(ref["runs"][fs], ref["metrics"][fs])
        final[key] = {k: v for k, v in f.items() if k != "_audit"}
        reps[key] = W.replicate_summary(f["_audit"])
        frames = lc.get(key, {}).get("frames")
        if frames is None:
            try:
                from falls_ml.modelreport import compute as C
                from falls_ml.modelreport.artifacts import discover_runs

                run = next(r for r in discover_runs([ref["dir"]]) if r.key == key)
                frames = C.load_frames(run, [ref["dir"]])
            except Exception as exc:  # noqa: BLE001
                sens[key] = {"error": f"{type(exc).__name__}: {exc}"}
        if frames is not None:
            try:
                s = W.lambda_sensitivity(ref["runs"][fs], frames)
                sens[key] = s
            except Exception as exc:  # noqa: BLE001
                sens[key] = {"error": f"{type(exc).__name__}: {exc}"}
        lcs[key] = W.learning_curve_summary(lc.get(key, {}).get("table") if lc.get(key, {}).get("info") is not None else None)
        mat[key] = W.materiality(final[key], sens.get(key) if not sens.get(key, {}).get("error") else None, lcs[key], reps[key])
    new, new_reps = {}, {}
    for cell, r in runs.items():
        if r.get("status") in (STATUS_NEW, STATUS_REUSED_EARLIER):
            f = W.final_fit_summary(Path(r["run_dir"]), r["metrics"])
            new[cell] = {**{k: v for k, v in f.items() if k != "_audit"},
                         "origin": "fitted in this attempt" if r["status"] == STATUS_NEW else "completed in an earlier attempt (reused, not refitted)"}
            new_reps[cell] = W.replicate_summary(f["_audit"])
    return {"final_fits": final, "replicates": reps, "lambda_sensitivity": sens, "learning_curve": lcs, "new_runs": new, "new_run_replicates": new_reps,
            "materiality": mat}


# ============================================================================ privacy + manifest
def _privacy(res: D00Result, share: Path, frame: pd.DataFrame, contract: Any, pepper: str, ref: dict[str, Any], out: Path) -> dict[str, Any]:
    import hashlib

    from falls_ml.eda.runner import privacy_scan

    ids: set[str] = set()
    for col in ("Customer_Full_ID", "Snapshot_Key"):
        if col in frame.columns:
            ids |= {str(v) for v in frame[col].dropna().astype(str) if len(str(v)) >= 6}
    raw = frame["Customer_Full_ID"].dropna().astype(str) if "Customer_Full_ID" in frame.columns else pd.Series([], dtype=str)
    ids |= {hashlib.sha256(f"{pepper}|{v}".encode("utf-8")).hexdigest()[:20] for v in raw}
    scan = privacy_scan(share, ids, pepper)
    return {**scan, "rule": "every text file under share/ (csv, md, json, html, txt) checked for member ids, snapshot keys, research-id pseudonyms and the pepper"}


def _manifest(res: D00Result, config: Any, mapping: Any, dictionary: Any, input_sha: str, pinfo: dict[str, Any], min_cell: int) -> dict[str, Any]:
    from falls_ml import __version__

    return {"watermark": WATERMARK, "status": STATUS_LINES, "synthetic_data": res.synthetic, "falls_ml_version": __version__, "command": "meuhedet-d00",
            "index_date": res.index_date, "input_sha256": input_sha, "pepper_sha256": pinfo["sha256"],
            "definitions": {"d00_config": {"name": config.name, "version": config.version, "sha256": config.sha256},
                            "mapping": {"name": mapping.name, "version": mapping.version, "sha256": mapping.content_sha256},
                            "data_dictionary": {"name": dictionary.name, "version": dictionary.version, "sha256": dictionary.sha256}},
            "plan_sha256": res.plan.get("plan_sha256"), "graph_sha256": res.graph.sha256 if res.graph else None,
            "cells": {c.cell: res.records.get(c.cell, {}).get("status", c.source) for c in res.cells},
            "integrity": {Path(k).name: v["unchanged"] for k, v in (res.integrity.get("folders") or {}).items()} if res.integrity else "checked at exit",
            "privacy": {"min_cell": min_cell, "identifier_scan": {k: v for k, v in res.privacy.items() if k != "hits"}},
            "management_report": res.management, "seconds": res.seconds,
            "resume": {k: v for k, v in res.resume.items() if k not in ("cells", "incomplete_attempts", "earlier_attempt_integrity")} | (
                {"cells": {c["cell"]: c["action"] for c in res.resume.get("cells", [])}, "see": "RESUME_LOG.json"} if res.resume.get("resumed") else {}),
            "share_rule": "send back ONLY the share/ folder; never datasets/, runs/ (row-level predictions), configs/ with local paths, id_pepper.txt or the CSV"}
