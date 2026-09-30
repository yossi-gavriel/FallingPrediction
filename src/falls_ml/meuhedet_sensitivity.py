"""SAFE-ALL-ROWS sensitivity analysis and feature ablations on top of a completed ``meuhedet-explore`` run (Phase 1, single snapshot).

Three analysis states are distinguished (never merged):

1. ``CURRENT_D00_DROPPED`` - the completed reference run in ``--reference`` (STRICT + EXTENDED on the D-00-clean cohort). It is read,
   hashed before and after, and never written to.
2. ``SAFE_ALL_ROWS`` - every eligible row with a usable 180-day label, no exclusion because ``Last_Dx_Date`` / ``Last_Fall_Date`` equal
   the index date, and only the predictors that the extract itself shows to be temporally safe: a predictor is excluded when it reads a
   source whose last-record date is on/after the index day (its aggregated value may hold index-day records and cannot be repaired), and
   a row is excluded only when the file proves an index-day record of a source a *built* predictor reads (the registry evidence column).
   Nothing is imputed, replaced or reconstructed.
3. ``FINAL_DWH_FIXED`` - documented only: all eligible rows with the complete predictor set once the DWH rebuilds the source windows with
   ``Event_Date < Index_Date``. It is not run and no number is invented for it.

Targeted ablations on the reference cohort (identical rows and test partition, hash-verified) remove the predictors that SAFE-ALL-ROWS
must drop - individually and jointly - so that the feature-set effect (same rows, different predictors) is separated from the population
effect (same predictors, different rows: the joint ablation vs SAFE-ALL-ROWS, which is NOT a paired comparison).

Every run uses the reference run's own experiment config as its template (same model, seed, proportions, resampling), the same stored
pseudonymisation pepper (verified by fingerprint), the same outcome and the same adequacy gate. Outputs are aggregate only.
Every report carries EXPLORATORY 180-DAY OUTCOME - NOT EFALLS REPRODUCTION.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from falls_ml.adequacy import READY, assess_adequacy, write_adequacy
from falls_ml.data.dataset import ModelingDataset, sha256_file
from falls_ml.data.meuhedet_timing import timing_diagnostic, write_timing_diagnostic
from falls_ml.data.meuhedet_wide import (DEFAULT_MAPPING, PREDICTOR_ROLES, BuildReport, WideContract, WideMapping,
                                         build_meuhedet_dataset_from_frame, load_wide_contract, load_wide_mapping, read_wide_extract_report)
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.logging_utils import get_logger
from falls_ml.meuhedet_explore import (MAPPING_CAVEATS, RUN_LABELS, SCIENTIFIC_STATUS, _ci, _est, _fmt, _md_table, _selected_raw_features,
                                       _write_config, feature_report, render_split_audit, resolve_id_pepper, split_audit, watermark)

log = get_logger(__name__)

REFERENCE_STATE = "CURRENT_D00_DROPPED"
SAFE_KEY, SAFE_STATE = "safe_all_rows", "SAFE_ALL_ROWS"
SAFE_LABEL = "MEUHEDET_EFALLS_SAFE_ALL_ROWS_180D_SENSITIVITY"
FINAL_KEY, FINAL_STATE = "final_dwh_fixed", "FINAL_DWH_FIXED"
NOT_RUN = "NOT RUN - requires the DWH correction (Event_Date < Index_Date); no result is fabricated"
INSUFFICIENT = "INSUFFICIENT EVENTS FOR EXPLORATORY MODEL FIT"
#: columns the researcher asked to be traced explicitly (any other predictor column of the same source groups is traced as well)
REQUESTED_TRACE = ("mobility_problems", "falls", "Gait_Disorder_Since_Study_Start_Ind", "Prior_Fall_Since_Study_Start_Ind", "Diagnosis_Count_180D",
                   "Diagnosis_Count_365D", "Distinct_Diagnosis_Codes_365D", "Chronic_Diagnosis_Count_365D", "Prior_Fall_Count_30D", "Prior_Fall_Count_90D",
                   "Prior_Fall_Count_180D", "Prior_Fall_Count_365D", "Days_Since_Last_Fall")
CLASS_TEXT = {
    "EXCLUDED_PROVEN_SAME_DAY_SOURCE": "reads an event source whose last-record date is on/after the index day in this extract: the aggregated "
                                       "value may contain index-day (or later) records and the pre-index value cannot be reconstructed",
    "EVENT_SOURCE_CLEAN_IN_THIS_EXTRACT": "reads an event source with a last-record date; no record on/after the index day in this extract",
    "DERIVED_FROM_IMMUTABLE_ATTRIBUTES": "computed from the birth date / sex and the index date; no event record can enter it",
    "STATE_AT_INDEX_WITH_PARTIAL_EVIDENCE": "registry membership at the index date; the extract carries no per-registry record date, only "
                                            "First_Registry_Date (sufficient-only evidence): rows whose first registry entry is on/after the index "
                                            "day are excluded in SAFE-ALL-ROWS; a later same-day entry of a member with an older registry is "
                                            "undetectable from the wide table",
    "STATE_AT_INDEX_UNVERIFIABLE": "declared pre-index / at-index by the contract; the extract carries no record date for the source, so an "
                                   "index-day record can neither be proven nor excluded (a limitation shared by every analysis, including the "
                                   "reference run)",
}
WARNING_TEXT = {
    "lasso_cv_minimum_not_identified": "the cross-validated deviance curve is flat around its minimum (fewer than 5 later grid points exceed it by "
                                       "0.1%); the argmin lambda is used (D-14). Neighbouring lambdas give nearly equivalent models; not an error",
    "lasso_path_not_converged": "some lambdas of a coordinate-descent path used every allowed pass without meeting the tolerance (typically the "
                                "smallest lambdas, where the fit approaches saturation); the framework refuses to report the model if the "
                                "SELECTED lambda itself did not converge, so a completed run always has a converged final model",
    "lasso_cv_fold_paths_not_converged": "at least one cross-validation fold path had unconverged lambdas; the CV deviance at those lambdas is "
                                         "approximate. If lambda* lies well inside the converged part of the grid the selection is unaffected",
    "lasso_unpenalized_refit_warnings": "the descriptive unpenalised logistic refit on the selected columns (odds ratios with Wald CIs) raised "
                                        "a warning or did not converge; it does not feed the served model, its coefficients or the metrics",
    "fp_selection_stata_omissions": "during fractional-polynomial form selection a column was constant or predicted one outcome class perfectly "
                                    "in that (re)sample and was omitted as Stata logit does; in bootstrap replicates this is expected for rare "
                                    "binary predictors",
    "fp_selection_not_converged": "the fractional-polynomial closed-test cycles did not stabilise within max_cycles; the last forms are used",
    "bootstrap_replicate_failed": "a bootstrap replicate could not be fitted (single outcome class or a degenerate fit) and was recorded, not "
                                  "silently skipped; the run stops if more than 10% fail",
    "bootstrap_convergence_retry": "a bootstrap replicate's full-data LASSO solve did not converge at the configured iteration limit; the same "
                                   "resample was refitted once (same seed and lambda, tolerance unchanged, higher full-data iteration limit) and "
                                   "accepted only if it converged - otherwise it counts as a failed replicate (pre-declared policy, v0.7.1)",
    "lasso_zero_variance_columns": "a design column is constant in that (re)sample; it gets a zero coefficient",
    "lasso_line_search_failed": "a proximal-Newton step found no descent direction at one lambda; that lambda is reported as unconverged",
    "stability_coefficients_at_numerical_zero": "coefficients below 1e-12 were counted as selected in a replicate",
    "optimism_metric_undefined_in_replicates": "a metric was undefined in some optimism replicates (single class or constant predictions)",
}
FINAL_FIT_EVENTS = ("lasso_cv_fitted",)


# ---------------------------------------------------------------------------- helpers
def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8", newline="\n")


def _write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def _keys(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    return frame[spec.identifier_columns[0]].astype(str) + "|" + pd.to_datetime(frame[spec.index_column]).dt.strftime("%Y-%m-%d")


def directory_digest(root: Path) -> dict[str, Any]:
    """sha256 of every file under ``root`` (relative path -> digest) and one combined digest; used to prove the reference is untouched."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    per_file = {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in files}
    combined = hashlib.sha256("\n".join(f"{k}  {v}" for k, v in per_file.items()).encode("utf-8")).hexdigest()
    return {"n_files": len(per_file), "combined_sha256": combined, "files": per_file}


def _feature_label(name: str) -> str:
    return name.upper()


# ---------------------------------------------------------------------------- 1. feature provenance
def feature_provenance(mapping: WideMapping, contract: WideContract, timing: dict[str, Any]) -> dict[str, Any]:
    """Trace every included eFalls predictor (and the Phase-2 columns of the implicated source groups) to its source columns, source
    groups, record-date / same-day-evidence columns and the D-00 evidence measured on this extract; decide, predictor by predictor,
    whether it can be defended as temporally safe from the information in the file. Aggregate only."""
    cols = timing["columns"]
    sets = mapping.feature_sets()
    guard = list(timing["d00_guard_columns"])
    violating = {c for c in guard if c in cols and cols[c]["root_cause"] != "CLEAN"}
    entries: list[dict[str, Any]] = []
    for f in mapping.features:
        if not f.include_in_baseline:
            continue
        groups = sorted({str(contract.get(c).group) for c in f.source_columns if contract.get(c).group})
        timings = sorted({contract.get(c).timing for c in f.source_columns})
        evidence: dict[str, Any] | None = None
        if f.record_date_column and f.record_date_column in cols:
            e = cols[f.record_date_column]
            evidence = {"column": f.record_date_column, "kind": "last_record_date", "n_on_index": e["n_on_index"], "n_after_index": e["n_after_index"],
                        "n_affected_eligible": e["n_affected_eligible"], "root_cause": e["root_cause"]}
        elif f.same_day_evidence_column and f.same_day_evidence_column in cols:
            e = cols[f.same_day_evidence_column]
            evidence = {"column": f.same_day_evidence_column, "kind": "same_day_evidence_sufficient_only", "n_on_index": e["n_on_index"],
                        "n_after_index": e["n_after_index"], "n_affected_eligible": e["n_affected_eligible"], "root_cause": e["root_cause"]}
        if f.record_date_column:
            klass = "EXCLUDED_PROVEN_SAME_DAY_SOURCE" if f.record_date_column in violating else "EVENT_SOURCE_CLEAN_IN_THIS_EXTRACT"
        elif f.op in ("copy_numeric", "recode") and set(timings) == {"at_index"} and not groups:
            klass = "DERIVED_FROM_IMMUTABLE_ATTRIBUTES"
        elif f.same_day_evidence_column:
            klass = "STATE_AT_INDEX_WITH_PARTIAL_EVIDENCE"
        else:
            klass = "STATE_AT_INDEX_UNVERIFIABLE"
        included = klass != "EXCLUDED_PROVEN_SAME_DAY_SOURCE"
        if klass == "EXCLUDED_PROVEN_SAME_DAY_SOURCE":
            reason = (f"{f.record_date_column} is on/after the index day on {evidence['n_affected_eligible'] if evidence else '?'} eligible rows "
                      f"(root cause {evidence['root_cause'] if evidence else '?'}); the source column(s) {list(f.source_columns)} aggregate that source")
        elif klass == "STATE_AT_INDEX_WITH_PARTIAL_EVIDENCE":
            reason = (f"kept; rows with {f.same_day_evidence_column} on/after the index day are excluded in SAFE-ALL-ROWS "
                      f"({evidence['n_affected_eligible'] if evidence else 0} eligible rows in this extract)")
        elif klass == "EVENT_SOURCE_CLEAN_IN_THIS_EXTRACT":
            reason = f"kept; {f.record_date_column} is before the index day on every eligible row of this extract"
        else:
            reason = "kept; " + CLASS_TEXT[klass]
        entries.append({"feature": f.canonical, "efalls_concept": f.efalls.get("concept"), "mapping_quality": f.quality,
                        "feature_sets": [k for k, v in sets.items() if f.canonical in v], "op": f.op, "source_columns": list(f.source_columns),
                        "source_groups": groups, "contract_timing": timings, "record_date_column": f.record_date_column,
                        "same_day_evidence_column": f.same_day_evidence_column, "d00_evidence": evidence, "provenance_class": klass,
                        "class_text": CLASS_TEXT[klass], "safe_all_rows": "included" if included else "excluded", "reason": reason})
    safe = [e["feature"] for e in entries if e["safe_all_rows"] == "included" and "extended" in e["feature_sets"]]
    excluded = [e["feature"] for e in entries if e["safe_all_rows"] == "excluded"]
    # Phase-2 columns of the implicated source groups (never modelling predictors of any current analysis) + the requested names
    groups_of_guard = {c: contract.get(c).group for c in guard if c in contract.names}
    phase2: list[dict[str, Any]] = []
    seen: set[str] = set()
    for guard_col, group in groups_of_guard.items():
        cause = cols.get(guard_col, {}).get("root_cause", "not measured")
        for c in contract.columns:
            if group and c.group == group and c.role in PREDICTOR_ROLES and c.name != guard_col:
                seen.add(c.name)
                phase2.append({"column": c.name, "source_group": group, "guard_column": guard_col, "d00_root_cause_of_source": cause, "timing": c.timing,
                               "status": "not a modelling predictor of any current analysis (Phase-2 column); same source as the guard column; "
                                         + ("would be excluded from a SAFE Phase-2 set" if cause != "CLEAN" else "clean in this extract")})
    requested: list[dict[str, Any]] = []
    feature_names = {e["feature"] for e in entries}
    for name in REQUESTED_TRACE:
        if name in feature_names:
            e = next(x for x in entries if x["feature"] == name)
            requested.append({"name": name, "kind": "eFalls predictor", "resolution": f"{e['provenance_class']}: SAFE-ALL-ROWS {e['safe_all_rows']} - {e['reason']}"})
        elif name in contract.names:
            c = contract.get(name)
            src = next((e for e in entries if name in e["source_columns"]), None)
            cause = cols.get(name, {}).get("root_cause") if name in cols else None
            if src is not None:
                requested.append({"name": name, "kind": "source column", "resolution": f"source of eFalls predictor {src['feature']} ({src['provenance_class']}; SAFE-ALL-ROWS {src['safe_all_rows']})"})
            elif name in seen:
                p2 = next(x for x in phase2 if x["column"] == name)
                requested.append({"name": name, "kind": "Phase-2 column", "resolution": p2["status"] + f" (source D-00 root cause {p2['d00_root_cause_of_source']})"})
            else:
                requested.append({"name": name, "kind": "contract column", "resolution": f"role {c.role}, group {c.group}, timing {c.timing}; not used by any current analysis"
                                  + (f"; D-00 root cause {cause}" if cause else "")})
        else:
            requested.append({"name": name, "kind": "unknown", "resolution": "not an eFalls predictor and not a contract column"})
    row_rule = ("SAFE-ALL-ROWS rows: every eligible row with a non-null Fall_Next_180D_Ind. No row is excluded because Last_Dx_Date or Last_Fall_Date "
                "equals the index date (no built predictor reads those sources). A row IS excluded when the file proves an index-day record of a "
                "source a built predictor reads: " + ", ".join(sorted({e['same_day_evidence_column'] for e in entries
                                                                       if e['safe_all_rows'] == 'included' and e['same_day_evidence_column']}) or ["(none declared)"])
                + " on/after Index_Date (counted in the build report as n_rows_timing_violation under timing scope built_predictors).")
    return {"prediction_time": timing["prediction_time"], "predictor_record_rule": timing["predictor_record_rule"], "d00_guard_columns": guard,
            "violating_guard_columns": sorted(violating), "features": entries, "safe_all_rows_feature_set": safe, "excluded_from_safe_all_rows": excluded,
            "safe_row_rule": row_rule, "phase2_columns_of_implicated_sources": phase2, "requested_trace": requested,
            "note": "Provenance is traced through the mapping manifest (source_columns, op, record_date_column, same_day_evidence_column) and the "
                    "contract (group, timing), and the D-00 evidence is measured on this extract by the timing diagnostic. 'Unverifiable' sources "
                    "are a stated limitation, not evidence of a problem."}


def render_provenance_markdown(prov: dict[str, Any], wm: str) -> str:
    lines = ["# Feature provenance and the SAFE-ALL-ROWS predictor rule", "", f"**{wm}**", "",
             f"Prediction time {prov['prediction_time']}; predictors may use `{prov['predictor_record_rule']}`. D-00 guard columns of the cohort: "
             f"{prov['d00_guard_columns']}; with records on/after the index day in this extract: {prov['violating_guard_columns'] or 'none'}.", "",
             f"**SAFE-ALL-ROWS predictors ({len(prov['safe_all_rows_feature_set'])}):** {', '.join(prov['safe_all_rows_feature_set'])}", "",
             f"**Excluded from SAFE-ALL-ROWS ({len(prov['excluded_from_safe_all_rows'])}):** {', '.join(prov['excluded_from_safe_all_rows']) or 'none'}", "",
             prov["safe_row_rule"], "", "## Every eFalls predictor of the EXTENDED set", "",
             "| predictor | sets | sources | groups | timing | record-date / evidence column | D-00 evidence (on / after index, eligible affected) | class | SAFE-ALL-ROWS | reason |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e in prov["features"]:
        ev = e["d00_evidence"]
        ev_txt = "" if ev is None else f"{ev['column']}: {ev['n_on_index']} / {ev['n_after_index']} / {ev['n_affected_eligible']} ({ev['root_cause']})"
        lines.append(f"| {e['feature']} | {'; '.join(e['feature_sets'])} | {'; '.join(e['source_columns'])} | {'; '.join(e['source_groups']) or '-'} | "
                     f"{'; '.join(e['contract_timing'])} | {e['record_date_column'] or e['same_day_evidence_column'] or '-'} | {ev_txt} | {e['provenance_class']} | "
                     f"{e['safe_all_rows']} | {e['reason'].replace('|', '/')} |")
    lines += ["", "## Provenance classes", ""] + [f"- **{k}**: {v}" for k, v in CLASS_TEXT.items()]
    lines += ["", "## Requested columns", "", "| name | kind | resolution |", "|---|---|---|"]
    for r in prov["requested_trace"]:
        lines.append(f"| {r['name']} | {r['kind']} | {r['resolution'].replace('|', '/')} |")
    lines += ["", "## Phase-2 columns of the implicated source groups (not modelled anywhere yet)", "", "| column | group | guard column | source D-00 cause | status |", "|---|---|---|---|---|"]
    for p2 in prov["phase2_columns_of_implicated_sources"]:
        lines.append(f"| {p2['column']} | {p2['source_group']} | {p2['guard_column']} | {p2['d00_root_cause_of_source']} | {p2['status']} |")
    lines += ["", prov["note"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- 2. convergence / warnings audit
def _grid_index(lasso: dict[str, Any]) -> int | None:
    try:
        lam, lmax, ratio, n = float(lasso["lambda_star"]), float(lasso["lambda_max"]), float(lasso["lambda_ratio"]), int(lasso["n_lambda"])
    except (KeyError, TypeError, ValueError):
        return None
    if n <= 1 or lmax <= 0 or lam <= 0 or ratio <= 0 or ratio >= 1:
        return None
    return int(round(math.log(lam / lmax) / math.log(ratio) * (n - 1)))


def convergence_audit(run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    """Read ``run_log.jsonl`` and ``metrics.json`` of one run: which warnings occurred in the final fit versus the stability / optimism
    replicates, whether the selected lambda converged and where it sits on the grid, and whether any warning touches the reported metrics."""
    lines = (run_dir / "run_log.jsonl").read_text(encoding="utf-8").splitlines() if (run_dir / "run_log.jsonl").exists() else []
    events = [json.loads(ln) for ln in lines if ln.strip()]
    phase = "final_fit"
    per_phase: dict[str, Counter] = {"final_fit": Counter(), "stability": Counter(), "optimism": Counter(), "post_fit": Counter()}
    fp_forms: dict[str, Counter] = {}
    final_fp: dict[str, str] | None = None
    final_diag: dict[str, Any] | None = None
    path_nc: dict[str, list[int]] = {"final_fit": [], "stability": [], "optimism": [], "post_fit": []}
    refit_final: dict[str, Any] | None = None
    perfect: Counter = Counter()
    failures: Counter = Counter()
    retries: dict[str, Counter] = {}
    seen_final = False
    has_stab, has_opt = any(e.get("event") == "bootstrap_stability_done" for e in events), any(e.get("event") == "harrell_optimism_done" for e in events)
    for e in events:
        ev = e.get("event", "")
        if ev == "fp_selection_done":
            forms = {v: str(d.get("powers")) for v, d in (e.get("forms") or {}).items()}
            if not seen_final:
                final_fp = forms
            else:
                for v, f in forms.items():
                    fp_forms.setdefault(v, Counter())[f] += 1
        if ev == "lasso_cv_fitted" and not seen_final:
            final_diag = e
            seen_final = True
            phase = "stability" if has_stab else ("optimism" if has_opt else "post_fit")
            continue
        if e.get("level") == "WARNING":
            per_phase[phase][ev] += 1
            if ev == "lasso_path_not_converged":
                path_nc[phase].append(int(e.get("n_not_converged") or 0))
            if ev == "lasso_unpenalized_refit_warnings" and phase == "final_fit":
                refit_final = {"warnings": e.get("warnings"), "converged": e.get("converged")}
            if ev == "fp_selection_stata_omissions":
                for p in e.get("perfect_predictors") or []:
                    perfect[str(p.get("column"))] += 1
            if ev == "bootstrap_replicate_failed":
                failures[str(e.get("reason", ""))[:60]] += 1
            if ev == "bootstrap_convergence_retry":
                retries.setdefault(str(e.get("component") or phase), Counter())[str(e.get("outcome"))] += 1
        if ev == "bootstrap_stability_done":
            phase = "optimism" if has_opt else "post_fit"
        elif ev == "harrell_optimism_done":
            phase = "post_fit"
    lasso = metrics.get("lasso") or {}
    resampling: dict[str, Any] = {}
    if (run_dir / "config.yaml").exists():
        try:
            cfg_raw = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
            an = cfg_raw.get("analysis") or {}
            resampling = {"stability_n": (an.get("stability") or {}).get("n_bootstrap"), "stability_enabled": (an.get("stability") or {}).get("enabled"),
                          "optimism_n": (an.get("optimism") or {}).get("n_bootstrap"), "optimism_enabled": (an.get("optimism") or {}).get("enabled"),
                          "bootstrap_ci_n": ((cfg_raw.get("evaluation") or {}).get("bootstrap") or {}).get("n")}
        except (OSError, yaml.YAMLError):
            resampling = {}
    n_lambda = int(lasso.get("n_lambda") or 0)
    k = final_diag.get("lambda_star_index") if final_diag else None
    if k is None:
        k = _grid_index(lasso)
    stop = final_diag.get("lambda_stop_index") if final_diag else None
    grid = {"lambda_star": lasso.get("lambda_star"), "lambda_max": lasso.get("lambda_max"), "lambda_ratio": lasso.get("lambda_ratio"), "n_lambda": n_lambda,
            "lambda_star_index": k, "grid_points_to_end": (n_lambda - 1 - k) if (k is not None and n_lambda) else None,
            "at_grid_end": bool(k is not None and n_lambda and k >= n_lambda - 1), "near_grid_end": bool(k is not None and n_lambda and k >= n_lambda - 5),
            "lambda_stop_index": stop, "beyond_stata_stop": bool(stop is not None and k is not None and k >= stop),
            "cv_minimum_identified": lasso.get("cv_minimum_identified"), "cv_paths_converged": lasso.get("cv_paths_converged"),
            "path_converged": final_diag.get("path_converged") if final_diag else None, "n_selected": lasso.get("n_selected")}
    final_warnings = dict(per_phase["final_fit"])
    qualifications: list[str] = []
    change: list[str] = []
    if grid["at_grid_end"]:
        qualifications.append("lambda* is the smallest grid value: the CV minimum may lie beyond the grid; extend the grid (lambda_min_ratio) in a "
                              "separately named run and check that the selection moves")
        change.append("extend the lambda grid (model.params.lambda_min_ratio below the current ratio) - justified because the minimum is on the boundary")
    elif grid["near_grid_end"]:
        qualifications.append("lambda* is within 5 grid points of the smallest lambda (weak penalty); the selection is inside the grid but close to its end")
    if grid["cv_minimum_identified"] is False:
        qualifications.append("CV minimum not identified: the deviance curve is flat around lambda*; nearby lambdas give nearly equivalent models (D-14 argmin used)")
    if grid["cv_paths_converged"] is False:
        m = max(path_nc["final_fit"] or [0])
        margin = grid["grid_points_to_end"]
        qualifications.append(f"some CV fold paths had unconverged lambdas (up to {m} of {n_lambda} per path; the log does not name them - they are "
                              f"typically the smallest lambdas). lambda* is {margin} grid points from the end" + (": if the unconverged lambdas are the smallest ones, "
                              "the selection is unaffected" if margin is not None and m <= margin else ": the selection may sit in the unconverged region - compare with a 1se rule"))
    if refit_final is not None:
        qualifications.append(f"the unpenalised refit (descriptive odds ratios only) warned: {refit_final['warnings']}, converged {refit_final['converged']}; "
                              "the served coefficients and every metric come from the LASSO solution, not from the refit")
    for w in metrics.get("warnings") or []:
        if "Stata-style omissions in the final fit" in w or "FP selection (Stata logit omissions)" in w:
            qualifications.append(f"final fit: {w}")
    if grid["path_converged"] is False and not grid["at_grid_end"]:
        qualifications.append("the full-data path had unconverged lambdas other than lambda* (the selected lambda converged, else the run would have refused to report)")
    if final_diag is None and not lasso:
        verdict = "NOT ASSESSABLE (no LASSO diagnostics found)"
    elif grid["at_grid_end"]:
        verdict = "QUESTIONABLE - lambda* on the grid boundary"
    elif qualifications:
        verdict = "VALID WITH QUALIFICATIONS"
    else:
        verdict = "VALID"
    fp_instab = {}
    for v, cnt in fp_forms.items():
        total = sum(cnt.values())
        same = cnt.get(final_fp.get(v, ""), 0) if final_fp else 0
        fp_instab[v] = {"final_form": final_fp.get(v) if final_fp else None, "replicates": total, "share_with_final_form": round(same / total, 3) if total else None,
                        "forms": dict(cnt.most_common(5))}
    return {"run": run_dir.name, "experiment": (metrics.get("experiment") or {}).get("name"), "final_fit": {"converged_at_lambda_star": True if final_diag or lasso else None,
            "rule": "the run raises LassoConvergenceError and writes no metrics if the full-data solve at lambda* does not converge",
            "grid": grid, "warnings": final_warnings, "unpenalised_refit": refit_final},
            "replicates": {"stability_warnings": dict(per_phase["stability"]), "optimism_warnings": dict(per_phase["optimism"]), "post_fit_warnings": dict(per_phase["post_fit"]),
                           "path_not_converged_lambdas_per_event": {p: sorted(v)[-5:] for p, v in path_nc.items() if v},
                           "perfect_predictor_omissions_by_column": dict(perfect), "replicate_failures": dict(failures), "fp_form_instability": fp_instab,
                           "resampling": resampling,
                           "convergence_retry_policy_in_effect": "bootstrap_convergence_retry" in metrics,
                           "convergence_retries": {c: dict(v) for c, v in retries.items()},
                           "run_warnings": list(metrics.get("warnings") or [])},
            "verdict": verdict, "qualifications": qualifications, "solver_or_grid_change_warranted": change,
            "warning_meaning": {k: WARNING_TEXT.get(k, "") for k in set().union(*[set(c) for c in per_phase.values()])}}


def render_convergence_markdown(audits: dict[str, dict[str, Any]], wm: str) -> str:
    lines = ["# Convergence and warnings audit", "", f"**{wm}**", "",
             "For every run: did the final fit converge at the selected lambda, where does lambda* sit on the grid, which warnings occurred in the "
             "final fit and which only in the bootstrap stability / optimism replicates, and does anything qualify the reported metrics. Changing the "
             "solver or the grid is proposed only where the audit finds a boundary condition.", "",
             "| analysis | run | verdict | lambda* index / n | grid points to end | CV minimum identified | CV fold paths converged | final-fit warnings | stability warnings | optimism warnings |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, a in audits.items():
        g = a["final_fit"]["grid"]
        lines.append(f"| {key} | {a['run']} | {a['verdict']} | {g['lambda_star_index']} / {g['n_lambda']} | {g['grid_points_to_end']} | {g['cv_minimum_identified']} | "
                     f"{g['cv_paths_converged']} | {a['final_fit']['warnings'] or '-'} | {a['replicates']['stability_warnings'] or '-'} | {a['replicates']['optimism_warnings'] or '-'} |")
    for key, a in audits.items():
        lines += ["", f"## {key} ({a['run']})", "", f"Verdict: **{a['verdict']}**"]
        lines += [f"- {q}" for q in a["qualifications"]] or ["- no qualification: the final fit converged at lambda*, inside the grid, with an identified CV minimum"]
        if a["solver_or_grid_change_warranted"]:
            lines += ["", "Solver / grid change warranted:"] + [f"- {c}" for c in a["solver_or_grid_change_warranted"]]
        else:
            lines += ["", "Solver / grid change warranted: **no** (no boundary condition; warnings are informative or confined to replicates)"]
        r = a["replicates"]
        if r["perfect_predictor_omissions_by_column"]:
            lines.append(f"- perfect-predictor omissions in replicates (column: replicates): {r['perfect_predictor_omissions_by_column']} - a rare binary predictor "
                         "with no events (or no non-events) in a resample is dropped for that replicate as Stata logit does; the final fit on the full training "
                         "data is unaffected unless listed under final-fit warnings")
        if r["replicate_failures"]:
            lines.append(f"- replicate failures (recorded): {r['replicate_failures']}")
        if r["fp_form_instability"]:
            lines.append("- fractional-polynomial form stability across stability replicates (variable: share of replicates choosing the final form): "
                         + "; ".join(f"{v} {d['final_form']}: {d['share_with_final_form']}" for v, d in r["fp_form_instability"].items())
                         + " - a form change in a resample is expected when the non-linearity test is close to alpha; it affects the replicate, not the final model")
        if r["run_warnings"]:
            lines += ["- run warnings (metrics.json):"] + [f"    - {w}" for w in r["run_warnings"]]
    lines += ["", "## What the warnings mean", ""] + [f"- `{k}`: {v}" for k, v in WARNING_TEXT.items()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- 3. analysis records and comparison
def _stability_top(run_dir: Path, n: int = 8) -> list[str]:
    p = run_dir / "feature_stability.csv"
    if not p.exists():
        return []
    s = pd.read_csv(p)
    if not len(s) or "raw_feature" not in s.columns:
        return []
    col = "raw_feature_selection_frequency" if "raw_feature_selection_frequency" in s.columns else "selection_frequency"
    t = s.groupby("raw_feature")[col].max().sort_values(ascending=False).dropna()
    return [f"{k} ({v:.2f})" for k, v in t.head(n).items()]


def analysis_record(key: str, state: str, metrics: dict[str, Any], run_dir: Path, *, features: list[str], excluded: dict[str, str],
                    timing_exclusion: dict[str, Any], population: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    test = metrics["performance"]["test"]
    primary = metrics.get("primary_variant") or "uncalibrated"
    p, u, r = test.get(primary) or {}, test.get("uncalibrated") or {}, test.get("recalibrated") or {}
    by_split = (metrics.get("cohort") or {}).get("by_split") or {}
    split = {k: {"n_rows": v.get("n_rows") or v.get("n"), "n_patients": v.get("n_patients"), "n_events": v.get("n_events")} for k, v in by_split.items()}
    return {"key": key, "state": state, "label": metrics["experiment"]["name"], "run_dir": str(run_dir), "run_id": run_dir.name,
            "n_rows": population.get("n_rows"), "n_patients": population.get("n_patients"), "n_events": population.get("n_events"),
            "event_prevalence": population.get("prevalence"), "n_predictors": len(features), "predictors": list(features), "predictors_excluded": excluded,
            "split": split, "test_rows_sha256": metrics["split"]["test_rows_sha256"], "primary_variant": primary,
            "auroc": _est(p.get("auroc")), "auroc_ci": _ci(p.get("auroc")), "pr_auc": _est(p.get("pr_auc")), "pr_auc_ci": _ci(p.get("pr_auc")),
            "brier": _est(p.get("brier")), "brier_ci": _ci(p.get("brier")), "brier_uncalibrated": _est(u.get("brier")),
            "calibration_slope_uncalibrated": _est(u.get("calibration_slope")), "calibration_slope_uncalibrated_ci": _ci(u.get("calibration_slope")),
            "calibration_intercept_uncalibrated": _est(u.get("calibration_intercept")), "calibration_intercept_uncalibrated_ci": _ci(u.get("calibration_intercept")),
            "citl_uncalibrated": _est(u.get("citl")), "citl_uncalibrated_ci": _ci(u.get("citl")), "oe_ratio_uncalibrated": _est(u.get("oe_ratio")), "oe_ratio_uncalibrated_ci": _ci(u.get("oe_ratio")),
            "calibration_slope_recalibrated": _est(r.get("calibration_slope")), "calibration_intercept_recalibrated": _est(r.get("calibration_intercept")),
            "citl_recalibrated": _est(r.get("citl")), "oe_ratio_recalibrated": _est(r.get("oe_ratio")),
            "lambda_star": (metrics.get("lasso") or {}).get("lambda_star"), "selected_design_columns": (metrics.get("lasso") or {}).get("n_selected"),
            "selected_raw_features": _selected_raw_features(run_dir), "stability_top": _stability_top(run_dir),
            "convergence_verdict": audit.get("verdict"), "final_fit_warnings": audit.get("final_fit", {}).get("warnings"),
            "replicate_warnings": {"stability": audit.get("replicates", {}).get("stability_warnings"), "optimism": audit.get("replicates", {}).get("optimism_warnings")},
            "timing_related_exclusion": timing_exclusion, "run_warnings": list(metrics.get("warnings") or [])}


def render_comparison_table(records: dict[str, dict[str, Any]], wm: str, *, feature_effect: list[dict[str, Any]], population_effect: dict[str, Any],
                            final_state: dict[str, Any]) -> str:
    rows = [("Analysis state", "state"), ("Run label", "label"), ("Rows (modelling)", "n_rows"), ("Patients", "n_patients"), ("Events", "n_events"),
            ("Event prevalence", "event_prevalence"), ("Predictors (n)", "n_predictors"), ("AUROC (test)", "auroc"), ("AUROC 95% CI", "auroc_ci"),
            ("PR-AUC", "pr_auc"), ("PR-AUC 95% CI", "pr_auc_ci"), ("Brier (served variant)", "brier"), ("Brier 95% CI", "brier_ci"),
            ("Calibration slope (uncalibrated)", "calibration_slope_uncalibrated"), ("slope 95% CI", "calibration_slope_uncalibrated_ci"),
            ("Calibration intercept (uncalibrated)", "calibration_intercept_uncalibrated"), ("CITL (uncalibrated)", "citl_uncalibrated"), ("CITL 95% CI", "citl_uncalibrated_ci"),
            ("O:E (uncalibrated)", "oe_ratio_uncalibrated"), ("O:E 95% CI", "oe_ratio_uncalibrated_ci"),
            ("Calibration slope (after recalibration on validation)", "calibration_slope_recalibrated"), ("CITL (after recalibration)", "citl_recalibrated"),
            ("Non-zero LASSO design columns", "selected_design_columns"), ("Lambda*", "lambda_star"), ("Convergence verdict", "convergence_verdict"),
            ("Test rows sha256 (first 12)", "test_rows_sha256")]
    keys = list(records)
    table = []
    for title, k in rows:
        row = {"metric": title}
        for key in keys:
            v = records[key].get(k)
            if k == "test_rows_sha256" and isinstance(v, str):
                v = v[:12] + "…"
            row[key] = v if isinstance(v, str) else _fmt(v) if isinstance(v, float) else ("" if v is None else str(v))
        table.append(row)
    lines = ["# Sensitivity comparison - reference, ablations, SAFE-ALL-ROWS", "", f"**{wm}**", "",
             "Every analysis uses the same outcome (Fall_Next_180D_Ind), the same LASSO learning process, the same seed and split proportions, the "
             "same resampling settings and the same stored pepper. Two kinds of difference are reported separately and must not be read as one:",
             "- **feature-set effect**: same rows and the same test partition (hash-verified), different predictor sets;",
             "- **population effect**: the same predictor set on a different set of rows (NOT a paired comparison; the split is redrawn on the new population).", "",
             _md_table(pd.DataFrame(table)), "## Per analysis", ""]
    for key in keys:
        r = records[key]
        te = r["timing_related_exclusion"]
        lines += [f"### {key} - {r['state']}", "", f"- run: `{r['run_id']}` ({r['label']})",
                  f"- predictors ({r['n_predictors']}): {', '.join(r['predictors'])}",
                  f"- predictors excluded and why: " + ("; ".join(f"{k}: {v}" for k, v in r["predictors_excluded"].items()) if r["predictors_excluded"] else "none"),
                  f"- split (rows / patients / events): " + ", ".join(f"{k} {v['n_rows']} / {v['n_patients']} / {v['n_events']}" for k, v in r["split"].items()),
                  f"- LASSO-selected predictors (non-zero at lambda*): {', '.join(r['selected_raw_features']) or 'none'}",
                  f"- stability (bootstrap selection frequency = share of refits with a non-zero coefficient): {', '.join(r['stability_top']) or 'n/a'}",
                  f"- convergence: {r['convergence_verdict']}; final-fit warnings {r['final_fit_warnings'] or 'none'}; replicate warnings {r['replicate_warnings']}",
                  f"- known timing-related exclusion: {'YES' if te.get('n_rows_excluded') else 'NO'} - {te.get('text')}", ""]
    lines += ["## Feature-set effect (same rows, same test partition)", "", "| comparison | test rows identical | AUROC A | AUROC B | difference (B - A) |", "|---|---|---|---|---|"]
    for fe in feature_effect:
        lines.append(f"| {fe['a']} -> {fe['b']} | {fe['same_test_rows']} | {_fmt(fe['auroc_a'])} | {_fmt(fe['auroc_b'])} | {_fmt(fe['delta'])} |")
    pe = population_effect
    lines += ["", "## Population effect (same predictors, different rows - unpaired)", "",
              f"{pe.get('text', '')}", ""]
    if pe.get("rows"):
        lines += ["| quantity | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in pe["rows"].items()]
    lines += ["", f"## {FINAL_STATE} (documented future state)", "", f"- status: {final_state['status']}", f"- population: {final_state['population']}",
              f"- predictors: {final_state['predictors']}", f"- requirement: {final_state['requirement']}", f"- what it will answer: {final_state['answers']}", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- 4. the driver
@dataclass
class SensitivityResult:
    out_dir: Path
    reference_dir: Path
    input_file: str
    index_date: str = ""
    reference: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    datasets: dict[str, Path] = field(default_factory=dict)
    build_reports: dict[str, BuildReport] = field(default_factory=dict)
    runs: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    audits: dict[str, dict[str, Any]] = field(default_factory=dict)
    adequacy: dict[str, Any] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    population: dict[str, Any] = field(default_factory=dict)
    integrity: dict[str, Any] = field(default_factory=dict)
    pepper_sha256: str = ""
    summary_text: str = ""
    model_report: dict[str, Any] = field(default_factory=dict)


class _Stop(Exception):
    def __init__(self, reason: str, details: list[str] | None = None):
        super().__init__(reason)
        self.reason, self.details = reason, list(details or [])


def load_reference(reference_dir: Path) -> dict[str, Any]:
    """Locate and read the completed reference run (readiness, split audit, comparison, run metrics, dataset manifests, configs)."""
    ref = Path(reference_dir)
    need = ["readiness.json", "split_audit.json", "comparison.json", "configs", "datasets/extended/manifest.json", "audit/timing_diagnostic.json"]
    missing = [n for n in need if not (ref / n).exists()]
    if missing:
        raise _Stop(f"{ref} is not a completed meuhedet-explore results directory", [f"missing {m}" for m in missing])
    readiness = _read_json(ref / "readiness.json")
    if readiness.get("mode") != "single":
        raise _Stop("the reference run must be a single-snapshot run (--index-date); the sensitivity analysis for --all-index-dates is not implemented",
                    [f"reference mode: {readiness.get('mode')}"])
    runs: dict[str, Path] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for d in sorted((ref / "runs").iterdir()) if (ref / "runs").exists() else []:
        if (d / "RUN_COMPLETE.json").exists() and (d / "metrics.json").exists():
            m = _read_json(d / "metrics.json")
            for fs, label in RUN_LABELS.items():
                if m.get("experiment", {}).get("name") == label:
                    runs[fs], metrics[fs] = d, m
    if set(runs) != set(RUN_LABELS):
        raise _Stop("the reference directory does not hold completed STRICT and EXTENDED runs", [f"found {sorted(runs)}"])
    manifests = {fs: _read_json(ref / "datasets" / fs / "manifest.json") for fs in RUN_LABELS if (ref / "datasets" / fs / "manifest.json").exists()}
    build = manifests["extended"]["audit"]["meuhedet_build"]
    configs = {fs: ref / "configs" / f"{label}.yaml" for fs, label in RUN_LABELS.items()}
    if not configs["extended"].exists():
        raise _Stop("the reference EXTENDED experiment config is missing", [str(configs["extended"])])
    return {"dir": ref, "readiness": readiness, "split_audit": _read_json(ref / "split_audit.json"), "comparison": _read_json(ref / "comparison.json"),
            "runs": runs, "metrics": metrics, "manifests": manifests, "build": build, "configs": configs, "timing": _read_json(ref / "audit" / "timing_diagnostic.json"),
            "index_date": readiness.get("index_date"), "pepper_sha256": readiness.get("pepper_sha256"),
            "features": {fs: list(metrics[fs].get("feature_set", {}).get("names") or manifests[fs]["audit"]["meuhedet_build"]["feature_set"]["features"]) for fs in RUN_LABELS}}


def sensitivity(input_path: str | Path, reference_dir: str | Path, *, out_dir: str | Path | None = None, id_pepper: str | None = None,
                id_pepper_file: str | Path | None = None, sheet: str | None = None, encoding: str = "utf-8", sep: str = ",",
                mapping_path: str | Path = DEFAULT_MAPPING, data_freeze_date: str | None = None, min_cell: int = 10,
                run_ablations: bool = True, run_safe: bool = True, template_path: str | Path | None = None,
                model_report: str = "full") -> SensitivityResult:
    """Run the sensitivity package next to a completed reference run. Raises ``DatasetValidationError`` (after writing SENSITIVITY_README.md)
    when the inputs do not allow it."""
    src, ref_dir = Path(input_path), Path(reference_dir)
    if not src.is_file():
        raise DatasetValidationError(f"Input file not found: {src}")
    out = Path(out_dir) if out_dir is not None else ref_dir.with_name(ref_dir.name + "_sensitivity")
    if out.resolve() == ref_dir.resolve() or ref_dir.resolve() in out.resolve().parents:
        raise DatasetValidationError(f"--out {out} must not be the reference directory or inside it (the reference is immutable)")
    if out.exists() and any(out.iterdir()):
        raise DatasetValidationError(f"Output directory {out} is not empty; results are immutable - choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    res = SensitivityResult(out_dir=out, reference_dir=ref_dir, input_file=src.name)
    try:
        _stages(res, src, out, id_pepper=id_pepper, id_pepper_file=id_pepper_file, sheet=sheet, encoding=encoding, sep=sep, mapping_path=mapping_path,
                data_freeze_date=data_freeze_date, min_cell=min_cell, run_ablations=run_ablations, run_safe=run_safe, template_path=template_path)
    except _Stop as stop:
        _write_text(out / "SENSITIVITY_README.md", f"# STOPPED - {stop.reason}\n\n" + "\n".join(f"- {d}" for d in stop.details) + "\n")
        raise DatasetValidationError(f"STOPPED - {stop.reason}", stop.details) from None
    if res.runs and model_report != "off":
        from falls_ml.modelreport.runner import attach_reports

        res.model_report = attach_reports([ref_dir, out], out / "reports", own_results=out, learning_curve=model_report)
        again = directory_digest(ref_dir)
        if again["combined_sha256"] != res.integrity.get("combined_sha256_after"):
            raise DatasetValidationError("the reference directory changed while the model reports were built (it is only read)", [])
    return res


def _stages(res: SensitivityResult, src: Path, out: Path, *, id_pepper: str | None, id_pepper_file: str | Path | None, sheet: str | None, encoding: str,
            sep: str, mapping_path: str | Path, data_freeze_date: str | None, min_cell: int, run_ablations: bool, run_safe: bool,
            template_path: str | Path | None) -> None:
    from falls_ml.config import load_experiment_config
    from falls_ml.experiment import run_experiment
    from falls_ml.splitting import make_split

    # 0. the reference: read, fingerprint, never written
    ref = load_reference(res.reference_dir)
    res.reference, res.index_date = ref, str(ref["index_date"])
    before = directory_digest(ref["dir"])
    mapping = load_wide_mapping(mapping_path)
    contract = load_wide_contract(mapping.contract_path)
    spec = load_feature_spec(mapping.exploratory_spec_path)
    wm = watermark(mapping)
    sets = mapping.feature_sets()
    ref_features = ref["features"]
    if sorted(ref_features["extended"]) != sorted(sets["extended"]):
        raise _Stop("the reference EXTENDED feature set differs from the current mapping's EXTENDED set", [f"reference {ref_features['extended']}", f"mapping {sets['extended']}"])
    template = Path(template_path) if template_path else ref["configs"]["extended"]
    # 1. read the extract; D-00 diagnostic on THIS file; provenance
    read = read_wide_extract_report(src, contract, encoding=encoding, sep=sep, sheet=sheet)
    input_sha = sha256_file(src)
    res.timing = timing_diagnostic(read.frame, contract, mapping, min_cell=min_cell)
    write_timing_diagnostic(res.timing, out / "audit")
    res.provenance = feature_provenance(mapping, contract, res.timing)
    _write_json(out / "feature_provenance.json", res.provenance)
    _write_text(out / "feature_provenance.md", render_provenance_markdown(res.provenance, wm))
    ref_input_sha = (ref["manifests"]["extended"]["audit"]["meuhedet_build"].get("input_sha256"))
    if ref_input_sha and ref_input_sha != input_sha:
        raise _Stop("the input file is not the file the reference run used (sha256 differs); the rows would not be comparable",
                    [f"reference input sha256 {ref_input_sha}", f"this file {input_sha}"])
    # 2. the pepper must be the reference's (same research ids => same split on the same rows)
    pepper, pinfo = resolve_id_pepper(src, out, explicit=id_pepper, pepper_file=id_pepper_file)
    res.pepper_sha256 = pinfo["sha256"]
    if ref["pepper_sha256"] and pinfo["sha256"] != ref["pepper_sha256"]:
        raise _Stop("the pseudonymisation pepper differs from the reference run's; the research ids and the split would not match",
                    [f"reference pepper sha256 {ref['pepper_sha256']}", f"this run {pinfo['sha256']} ({pinfo['source']})",
                     "pass --id-pepper-file <the reference results folder>\\id_pepper.txt"])
    safe_set = list(res.provenance["safe_all_rows_feature_set"])
    excluded = list(res.provenance["excluded_from_safe_all_rows"])
    by_feature = {e["feature"]: e for e in res.provenance["features"]}
    # 3. analyses: ablations on the reference cohort (identical rows), then SAFE-ALL-ROWS
    analyses: dict[str, dict[str, Any]] = {}
    if run_ablations and excluded:
        singles = [(f"ablation_no_{f}", (f,)) for f in excluded]
        joint = [("ablation_no_" + "_no_".join(excluded), tuple(excluded))] if len(excluded) > 1 else []
        for key, removed in singles + joint:
            feats = [f for f in sets["extended"] if f not in removed]
            analyses[key] = {"state": f"{REFERENCE_STATE}_EXTENDED_ABLATION", "label": "MEUHEDET_EFALLS_EXTENDED_180D_ABLATION_NO_" + "_NO_".join(_feature_label(f) for f in removed),
                             "features": feats, "timing_scope": "cohort", "index_day_records": "drop_rows", "expect_reference_rows": True,
                             "excluded": {f: f"ablation: removed on purpose (reference cohort kept); provenance {by_feature[f]['provenance_class']}" for f in removed},
                             "title": f"EXTENDED without {', '.join(removed)} on the reference (D-00-dropped) cohort"}
    elif run_ablations:
        res.skipped["ablations"] = "no predictor is excluded by the SAFE-ALL-ROWS provenance rule on this extract, so there is nothing to ablate"
    if run_safe:
        if not safe_set:
            res.skipped[SAFE_KEY] = "the SAFE-ALL-ROWS predictor set is empty"
        else:
            analyses[SAFE_KEY] = {"state": SAFE_STATE, "label": SAFE_LABEL, "features": safe_set, "timing_scope": "built_predictors", "index_day_records": "drop_rows",
                                  "expect_reference_rows": False, "excluded": {f: by_feature[f]["reason"] for f in excluded}, "title": "SAFE-ALL-ROWS: all labelled rows, temporally safe predictors only"}
    if not analyses:
        raise _Stop("nothing to run", [f"skipped: {res.skipped}"])
    ref_frame = pd.read_parquet(ref["dir"] / "datasets" / "extended" / "modeling_dataset.parquet")
    ref_keys = set(_keys(ref_frame, spec))
    version_base = f"meuhedet-wide-{res.index_date}-{src.stem}"
    ref_test_sha = ref["split_audit"]["test_rows_sha256"]
    # datasets + split checks + adequacy (before any fit)
    plans: dict[str, Any] = {}
    configs: dict[str, Path] = {}
    for key, a in analyses.items():
        ds_dir = out / "datasets" / key
        try:
            manifest, report = build_meuhedet_dataset_from_frame(
                read.frame, ds_dir, index_date=res.index_date, dataset_version=f"{version_base}-{key}", data_freeze_date=data_freeze_date, mapping=mapping,
                contract=contract, spec=spec, input_name=src.name, input_sha256=input_sha, wrong_type=read.wrong_type, wrong_type_examples=read.wrong_type_examples,
                date_formats=read.date_formats, features=a["features"], feature_set_label=key, id_pepper=pepper, index_day_records=a["index_day_records"],
                read_report=read.to_dict(), notes=f"{wm}; {a['title']}", timing_scope=a["timing_scope"])
        except DatasetValidationError as exc:
            raise _Stop(f"the {key} dataset cannot be built: {exc}", exc.problems) from None
        res.datasets[key], res.build_reports[key] = ds_dir, report
        ds = ModelingDataset.load(ds_dir, spec, features=a["features"])
        frame, sub = ds.frame, ds.spec
        keys = set(_keys(frame, spec))
        if a["expect_reference_rows"]:
            if keys != ref_keys:
                raise _Stop(f"{key}: the rebuilt cohort differs from the reference cohort", [f"reference rows {len(ref_keys)}", f"rebuilt rows {len(keys)}",
                            f"only in reference {len(ref_keys - keys)}", f"only in rebuilt {len(keys - ref_keys)}"])
            differing = _differing_columns(ref_frame, frame, spec, a["features"])
            if differing:
                raise _Stop(f"{key}: the rebuilt predictor values differ from the reference dataset (mapping definitions changed?)", differing)
        description = f"{wm}. {a['title']}: {len(a['features'])} predictors ({', '.join(a['features'])}); reference run {ref['runs']['extended'].name}."
        cfg_path = _write_config(template, out / "configs" / f"{a['label']}.yaml", name=a["label"], description=description, features=a["features"], fast=False, spec=spec)
        configs[key] = cfg_path
        cfg = load_experiment_config(cfg_path)
        plan = make_split(frame, sub, cfg.validation)
        plans[key] = plan
        sa = split_audit(frame, sub, plan, strategy_note=a["title"],
                         reproducibility={"input_sha256": input_sha, "pepper_sha256": res.pepper_sha256, "mapping_sha256": mapping.content_sha256,
                                          "config_sha256": cfg.sha256(), "dataset_sha256": manifest.data_sha256, "reference_test_rows_sha256": ref_test_sha,
                                          "split_seed": cfg.validation.seed, "template": str(template)})
        _write_json(out / f"split_audit_{key}.json", sa)
        _write_text(out / f"split_audit_{key}.md", render_split_audit(sa))
        if a["expect_reference_rows"] and sa["test_rows_sha256"] != ref_test_sha:
            raise _Stop(f"{key}: the split does not reproduce the reference test partition", [f"reference {ref_test_sha}", f"this build {sa['test_rows_sha256']}"])
        a["split_audit"] = sa
        adequacy = assess_adequacy(frame, sub, plan, cfg)
        write_adequacy(adequacy, out, stem=f"adequacy_{key}")
        res.adequacy[key] = adequacy.to_dict()
        if not adequacy.ready:
            res.skipped[key] = f"{INSUFFICIENT}: {'; '.join(adequacy.reasons)}"
    # SAFE population vs reference population (counts only)
    if SAFE_KEY in res.datasets:
        safe_frame = ModelingDataset.load(res.datasets[SAFE_KEY], spec, features=analyses[SAFE_KEY]["features"]).frame
        res.population = population_comparison(ref_frame, safe_frame, spec, res.build_reports[SAFE_KEY], ref["build"])
    # 4. runs
    for key, a in analyses.items():
        if key in res.skipped:
            continue
        result = run_experiment(None, res.datasets[key], configs[key], runs_dir=out / "runs")
        res.runs[key], res.metrics[key] = Path(result.run_dir), result.metrics
        if result.metrics["split"]["test_rows_sha256"] != a["split_audit"]["test_rows_sha256"]:
            raise DatasetValidationError(f"{key}: the run's test rows differ from the split audit computed before training", [])
        fr = feature_report(a["features"], mapping, spec, res.datasets[key], Path(result.run_dir), result.metrics)
        fr.to_csv(out / f"feature_report_{key}.csv", index=False, lineterminator="\n")
        _write_text(out / f"feature_report_{key}.md", f"# Feature report - {a['label']}\n\n**{wm}**\n\n{a['title']}; {len(a['features'])} predictors; run `{Path(result.run_dir).name}`.\n"
                    "Ordered by permutation importance, then |coefficient| (descriptive ranking on this label - not causal, not clinical importance).\n\n" + _md_table(fr))
        log.info("meuhedet_sensitivity_run_done", extra_fields={"analysis": key, "run_id": result.run_id, "n_features": len(a["features"])})
    # 5. convergence audits (reference runs + new runs) and analysis records
    for fs in ("strict", "extended"):
        res.audits[f"reference_{fs}"] = convergence_audit(ref["runs"][fs], ref["metrics"][fs])
    for key in res.runs:
        res.audits[key] = convergence_audit(res.runs[key], res.metrics[key])
    _write_json(out / "convergence_audit.json", res.audits)
    _write_text(out / "convergence_audit.md", render_convergence_markdown(res.audits, wm))
    ref_pop = {"n_rows": ref["build"]["n_rows_final"], "n_patients": ref["build"]["n_patients_final"], "n_events": ref["build"]["n_events"], "prevalence": ref["build"]["outcome_prevalence"]}
    ref_excl = {"n_rows_excluded": ref["build"]["n_rows_timing_violation"], "columns": ref["build"].get("timing_violations_by_column"),
                "text": f"{ref['build']['n_rows_timing_violation']} eligible labelled rows excluded before training because a D-00 guard column was on/after the index date "
                        f"({ref['build'].get('timing_violations_by_column')}); excluded rows never entered training, validation or test"}
    for fs in ("strict", "extended"):
        res.records[f"reference_{fs}"] = analysis_record(f"reference_{fs}", f"{REFERENCE_STATE}_{fs.upper()}", ref["metrics"][fs], ref["runs"][fs], features=ref_features[fs],
                                                          excluded=({f: "not in the STRICT set (APPROXIMATE mapping)" for f in sets["extended"] if f not in sets["strict"]} if fs == "strict" else {}),
                                                          timing_exclusion=ref_excl, population=ref_pop, audit=res.audits[f"reference_{fs}"])
    for key in res.runs:
        a, br = analyses[key], res.build_reports[key]
        pop = {"n_rows": br.n_rows_final, "n_patients": br.n_patients_final, "n_events": br.n_events, "prevalence": br.outcome_prevalence}
        if a["timing_scope"] == "cohort":
            te = ref_excl
        else:
            te = {"n_rows_excluded": br.n_rows_timing_violation, "columns": br.timing_violations_by_column, "cohort_guard_rows_retained": br.cohort_guard_rows_retained,
                  "text": f"{br.n_rows_timing_violation} rows excluded by the same-day evidence of the built predictors ({br.timing_violations_by_column or 'none'}); "
                          f"{br.cohort_guard_rows_retained} rows KEPT although Last_Dx_Date / Last_Fall_Date are on/after the index date ({br.cohort_guard_rows_retained_by_column}) "
                          "because no built predictor reads those sources"}
        res.records[key] = analysis_record(key, a["state"], res.metrics[key], res.runs[key], features=a["features"], excluded=a["excluded"], timing_exclusion=te,
                                           population=pop, audit=res.audits[key])
    # 6. comparison: feature-set effect (paired) and population effect (unpaired)
    feature_effect = []
    ref_ext = res.records["reference_extended"]
    for key, r in res.records.items():
        if key == "reference_extended":
            continue
        if r["test_rows_sha256"] == ref_ext["test_rows_sha256"]:
            feature_effect.append({"a": "reference_extended", "b": key, "same_test_rows": True, "auroc_a": ref_ext["auroc"], "auroc_b": r["auroc"],
                                   "delta": (r["auroc"] - ref_ext["auroc"]) if (r["auroc"] is not None and ref_ext["auroc"] is not None) else None})
    population_effect = _population_effect(res, analyses, excluded)
    final_state = {"status": NOT_RUN, "population": "all eligible rows with a usable 180-day label (no D-00 exclusion caused by same-day inclusion; the registry evidence "
                   "exclusion also disappears once the DWH windows are correct)", "predictors": "the complete EXTENDED set (and STRICT) with every source rebuilt as of the start of the index day",
                   "requirement": "the DWH rebuilds every diagnosis- and fall-derived column with Event_Date < Index_Date (see audit/timing_diagnostic.md) and re-exports; "
                                  "then meuhedet-explore runs again with index_day_records=fail and reports 0 violations",
                   "answers": "the performance of the full predictor set on the full population - the quantity the current analyses can only bracket"}
    comparison = {"watermark": wm, "reference_dir": str(ref["dir"]), "reference_test_rows_sha256": ref_test_sha, "pepper_sha256": res.pepper_sha256,
                  "mapping": {"reference_dataset_mapping_version": ref["manifests"]["extended"].get("mapping_version"), "current": f"{mapping.name}-{mapping.version}",
                              "current_sha256": mapping.content_sha256,
                              "note": "the rebuilt reference-cohort datasets were verified row by row and column by column against the reference dataset"},
                  "records": res.records, "skipped": res.skipped, "feature_set_effect": feature_effect, "population_effect": population_effect,
                  FINAL_KEY: final_state, "provenance": {"safe_all_rows_feature_set": safe_set, "excluded_from_safe_all_rows": excluded}}
    _write_json(out / "sensitivity_comparison.json", comparison)
    _write_text(out / "SENSITIVITY_COMPARISON.md", render_comparison_table(res.records, wm, feature_effect=feature_effect, population_effect=population_effect, final_state=final_state))
    # 7. reference integrity (nothing written there) and the executive report
    after = directory_digest(ref["dir"])
    res.integrity = {"reference_dir": str(ref["dir"]), "n_files": before["n_files"], "combined_sha256_before": before["combined_sha256"],
                     "combined_sha256_after": after["combined_sha256"], "unchanged": before == after,
                     "changed_files": sorted(set(k for k in before["files"] if before["files"].get(k) != after["files"].get(k)) | set(after["files"]) - set(before["files"]))}
    _write_json(out / "reference_integrity.json", res.integrity)
    if not res.integrity["unchanged"]:
        raise DatasetValidationError("the reference directory changed during the sensitivity run", res.integrity["changed_files"])
    res.summary_text = render_executive_report(res, mapping, wm)
    _write_text(out / "REAL_DATA_EXPLORATORY_REPORT.md", res.summary_text)
    _write_text(out / "SENSITIVITY_README.md", render_readme(res, wm))


def _differing_columns(ref_frame: pd.DataFrame, frame: pd.DataFrame, spec: FeatureSpec, features: list[str]) -> list[str]:
    """Names of the shared columns (outcome + built predictors) whose values differ between the reference dataset and a rebuilt one on the same keys."""
    a = ref_frame.assign(_k=_keys(ref_frame, spec)).set_index("_k").sort_index()
    b = frame.assign(_k=_keys(frame, spec)).set_index("_k").sort_index()
    out = []
    for col in [spec.outcome.name, *features]:
        if col not in a.columns or col not in b.columns:
            out.append(f"{col}: missing in one dataset")
            continue
        x, y = a[col].astype("string").fillna("<NA>"), b[col].astype("string").fillna("<NA>")
        n = int((x.to_numpy() != y.to_numpy()).sum())
        if n:
            out.append(f"{col}: {n} rows differ")
    return out


def population_comparison(ref_frame: pd.DataFrame, safe_frame: pd.DataFrame, spec: FeatureSpec, safe_build: BuildReport, ref_build: dict[str, Any]) -> dict[str, Any]:
    """Counts only: how the SAFE-ALL-ROWS rows relate to the reference rows (rows added back, rows removed by the registry evidence, prevalence of each)."""
    y = spec.outcome.name
    rk, sk = _keys(ref_frame, spec), _keys(safe_frame, spec)
    ref_set, safe_set = set(rk), set(sk)
    added = safe_frame.loc[~sk.isin(ref_set)]
    removed = ref_frame.loc[~rk.isin(safe_set)]
    common = safe_frame.loc[sk.isin(ref_set)]
    def block(df: pd.DataFrame) -> dict[str, Any]:
        n = int(len(df))
        ev = int(df[y].sum()) if n else 0
        return {"n_rows": n, "n_patients": int(df[spec.identifier_columns[0]].nunique()) if n else 0, "n_events": ev, "prevalence": (ev / n) if n else None}
    return {"reference": block(ref_frame), "safe_all_rows": block(safe_frame), "rows_in_both": block(common), "rows_only_in_safe": block(added),
            "rows_only_in_reference": block(removed), "paired": ref_set == safe_set,
            "safe_build": {"n_rows_timing_violation": safe_build.n_rows_timing_violation, "timing_violations_by_column": safe_build.timing_violations_by_column,
                           "cohort_guard_rows_retained": safe_build.cohort_guard_rows_retained, "cohort_guard_rows_retained_by_column": safe_build.cohort_guard_rows_retained_by_column,
                           "n_rows_label_null": safe_build.n_rows_label_null},
            "reference_build": {"n_rows_timing_violation": ref_build.get("n_rows_timing_violation"), "timing_violations_by_column": ref_build.get("timing_violations_by_column")},
            "note": "rows_only_in_safe are the rows the reference excluded for D-00 (minus any row the registry evidence excludes); their outcome prevalence versus "
                    "rows_in_both shows whether the D-00 exclusion was label-related. No row-level information is reported."}


def _population_effect(res: SensitivityResult, analyses: dict[str, dict[str, Any]], excluded: list[str]) -> dict[str, Any]:
    joint = "ablation_no_" + "_no_".join(excluded) if len(excluded) > 1 else (f"ablation_no_{excluded[0]}" if excluded else None)
    safe = res.records.get(SAFE_KEY)
    pop = res.population
    if safe is None or not pop:
        return {"text": "SAFE-ALL-ROWS was not run (see skipped), so no population effect is reported.", "rows": {}}
    partner = res.records.get(joint) if joint else None
    same_predictors = partner is not None and sorted(partner["predictors"]) == sorted(safe["predictors"])
    rows = {"reference rows / patients / events": f"{pop['reference']['n_rows']} / {pop['reference']['n_patients']} / {pop['reference']['n_events']} (prevalence {_fmt(pop['reference']['prevalence'], 4)})",
            "SAFE-ALL-ROWS rows / patients / events": f"{pop['safe_all_rows']['n_rows']} / {pop['safe_all_rows']['n_patients']} / {pop['safe_all_rows']['n_events']} (prevalence {_fmt(pop['safe_all_rows']['prevalence'], 4)})",
            "rows in both": f"{pop['rows_in_both']['n_rows']} (events {pop['rows_in_both']['n_events']}, prevalence {_fmt(pop['rows_in_both']['prevalence'], 4)})",
            "rows only in SAFE-ALL-ROWS (D-00 rows added back)": f"{pop['rows_only_in_safe']['n_rows']} (events {pop['rows_only_in_safe']['n_events']}, prevalence {_fmt(pop['rows_only_in_safe']['prevalence'], 4)})",
            "rows only in the reference (excluded by the registry same-day evidence)": f"{pop['rows_only_in_reference']['n_rows']} (events {pop['rows_only_in_reference']['n_events']})",
            "rows kept although Last_Dx_Date / Last_Fall_Date on/after index": f"{pop['safe_build']['cohort_guard_rows_retained']} {pop['safe_build']['cohort_guard_rows_retained_by_column']}",
            "paired with the reference": str(pop["paired"])}
    if partner is not None:
        rows[f"AUROC {joint} (reference cohort)"] = f"{_fmt(partner['auroc'])} ({partner['auroc_ci']})"
        rows["AUROC SAFE-ALL-ROWS (all rows)"] = f"{_fmt(safe['auroc'])} ({safe['auroc_ci']})"
        rows["difference (SAFE - ablation)"] = _fmt((safe["auroc"] - partner["auroc"]) if (safe["auroc"] is not None and partner["auroc"] is not None) else None)
    if pop["paired"]:
        text = ("SAFE-ALL-ROWS has exactly the reference rows (no D-00 row was excluded in the reference and no registry evidence row exists), so the "
                "comparison with the joint ablation is paired and the difference is a feature-set effect of zero by construction.")
    elif same_predictors:
        text = (f"SAFE-ALL-ROWS and {joint} use the SAME {len(safe['predictors'])} predictors, so their difference is a population effect: the rows added back "
                "(and any row removed by the registry evidence) change the case mix, and the split is redrawn on the new population. It is NOT a paired "
                "comparison: the test partitions differ, the confidence intervals overlap for that reason as well, and no part of the difference can be "
                "attributed to a predictor.")
    else:
        text = ("SAFE-ALL-ROWS and the ablations use different predictor sets, so the population effect cannot be isolated; the table reports both populations.")
    return {"text": text, "rows": rows, "partner": joint, "same_predictors": same_predictors, "paired": pop["paired"]}


# ---------------------------------------------------------------------------- 5. plain-language executive report
def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.2f}%"


def render_executive_report(res: SensitivityResult, mapping: WideMapping, wm: str) -> str:
    ref, prov, pop = res.reference, res.provenance, res.population
    build, tv = ref["build"], res.timing.get("verdict", {})
    rs, re_ = res.records.get("reference_strict", {}), res.records.get("reference_extended", {})
    safe = res.records.get(SAFE_KEY)
    synthetic = bool(build.get("synthetic")) or any(m.get("synthetic_fixture") for m in ref["metrics"].values())
    head = "SYNTHETIC DEMO - SYNTHETIC DATA - NOT SCIENTIFIC RESULTS" if synthetic else "REAL-DATA EXPLORATORY REPORT"
    n_in, n_other, n_inel, n_null, n_d00, n_final = (build.get("n_rows_input"), build.get("n_rows_other_index_dates"), build.get("n_rows_ineligible"),
                                                      build.get("n_rows_label_null"), build.get("n_rows_timing_violation"), build.get("n_rows_final"))
    cause = tv.get("root_cause", "not measured")
    by_col = build.get("timing_violations_by_column") or {}
    cols_t = res.timing.get("columns", {})
    per_col = [(c, cols_t[c]["label_prevalence"]) for c in res.timing.get("d00_guard_columns", [])
               if cols_t.get(c, {}).get("label_prevalence") and cols_t[c]["root_cause"] != "CLEAN"]
    if pop.get("rows_only_in_safe", {}).get("n_rows"):
        lp = {"affected": pop["rows_only_in_safe"]["prevalence"], "clean": pop["rows_in_both"]["prevalence"]}
        excl_prev_text = (f"Outcome prevalence among the excluded rows: {_pct(lp['affected'])} ({pop['rows_only_in_safe']['n_events']} falls in "
                          f"{pop['rows_only_in_safe']['n_rows']} rows) versus {_pct(lp['clean'])} among the rows the reference kept - "
                          f"{'the exclusion is related to the outcome' if _differs(lp) else 'no material difference in outcome rate was measured'}; per source column: "
                          + "; ".join(f"{c} {_pct(v.get('affected'))} vs {_pct(v.get('clean'))}" for c, v in per_col) + " (audit/timing_diagnostic.md).")
    else:
        excl_prev_text = ("Outcome prevalence among affected vs clean rows per source column: " + ("; ".join(f"{c} {_pct(v.get('affected'))} vs {_pct(v.get('clean'))}" for c, v in per_col) or "n/a")
                          + " (audit/timing_diagnostic.md).")
    excluded = prov.get("excluded_from_safe_all_rows", [])
    safe_feats = prov.get("safe_all_rows_feature_set", [])
    fr_ext = pd.read_csv(res.reference_dir / "feature_report_extended.csv") if (res.reference_dir / "feature_report_extended.csv").exists() else pd.DataFrame()
    top_ext = []
    if len(fr_ext):
        t = fr_ext.dropna(subset=["permutation_importance"]).sort_values("permutation_importance", ascending=False).head(6)
        top_ext = [f"{r.canonical_efalls_feature} (permutation importance {r.permutation_importance:.4f}; selection frequency {'' if pd.isna(r.selection_frequency) else f'{r.selection_frequency:.2f}'})"
                   for r in t.itertuples()]
    added_to_ext = [f for f in re_.get("predictors", []) if f not in set(rs.get("predictors", []))]
    contributing = [f for f in added_to_ext if f in set(re_.get("selected_raw_features", []))]
    auroc_gain = (re_.get("auroc") - rs.get("auroc")) if (re_.get("auroc") is not None and rs.get("auroc") is not None) else None
    ablation_lines = []
    for key, r in res.records.items():
        if key.startswith("ablation_") and re_.get("auroc") is not None and r.get("auroc") is not None:
            ablation_lines.append(f"- On identical rows and the same test partition, removing {', '.join(r['predictors_excluded'])} changed the EXTENDED AUROC from "
                                  f"{_fmt(re_['auroc'])} to {_fmt(r['auroc'])} (difference {'+' if r['auroc'] - re_['auroc'] >= 0 else ''}{_fmt(r['auroc'] - re_['auroc'])}; "
                                  f"CIs {re_['auroc_ci']} vs {r['auroc_ci']}).")
    lines = [f"# {head}: falls within 180 days on the Meuhedet wide table", "", f"**{wm}**", "",
             f"Reference run: `{res.reference_dir}` (untouched; every file hashed before and after this analysis: {'unchanged' if res.integrity.get('unchanged') else 'CHANGED'}). "
             f"Sensitivity outputs: `{res.out_dir}`. Written for a reader who is not a statistician; every number comes from the local artifacts.", "",
             "## 1. What population we used", "",
             f"The extract holds {n_in} rows. Rows on other snapshot dates ({n_other}) and rows the VIEW marks as not eligible ({n_inel}) were set aside. "
             f"Of the eligible rows on {res.index_date}, {n_null} have no usable 180-day outcome (censored: the member left, died or the data ends before the "
             f"window closes; reasons: {build.get('label_null_by_reason')}). The reference run then excluded {n_d00} rows for the timing problem explained in "
             f"section 4 and modelled **{n_final} rows = {build.get('n_patients_final')} members** with {build.get('n_events')} falls "
             f"({_pct(build.get('outcome_prevalence'))}).", "",
             "## 2. What the prediction date was", "",
             f"Every member is looked at on **{res.index_date}**, at the **start** of that day. Predictors may only use information recorded before that day; "
             "the outcome is what happens on that day and the following 180 days.", "",
             "## 3. What outcome we predicted", "",
             "`Fall_Next_180D_Ind`: whether the VIEW records a fall/fracture event within the index day plus 180 days. This is the Meuhedet exploratory outcome, "
             "not the eFalls outcome (eFalls uses emergency attendance or admission for a fall or fracture within 12 months), which is why every file says "
             "EXPLORATORY 180-DAY OUTCOME - NOT EFALLS REPRODUCTION.", "",
             "## 4. What D-00 means", "",
             "D-00 is the rule that a predictor may only be built from records dated strictly before the index day. The wide table stores, for each source, "
             "the date of the last record it used. When that date equals (or follows) the index date, the source's aggregated predictors (counts, 'since study start' "
             f"flags, days-since values) were built with a record the model is not allowed to see. On this extract: {tv.get('n_eligible_rows_violating_d00')} "
             f"eligible rows ({tv.get('pct_of_eligible')}%), by column {by_col}; root cause **{cause}** - {tv.get('root_cause_text')}.", "",
             "What we know with certainty: rows whose last-record dates are all before the index day contain no index-day record of those sources. Rows with a "
             "last-record date on the index day contain at least one. What the extract does not tell us: what the aggregated values would have been without "
             "that record - the underlying events are not exported, so the correct pre-index value cannot be rebuilt in Python, and inventing it (0, NULL or a "
             "guessed earlier date) would be fabrication.", "",
             f"## 5. Why {n_d00} rows were excluded from the first run", "",
             f"The reference run applied the guard with `--index-day-records drop_rows`: the {n_d00} rows with a diagnosis or fall record on/after the index day "
             "were removed before any modelling, and counted. STRICT does not read those sources, but it shares the same rows so that STRICT and EXTENDED are "
             f"compared on identical members. {excl_prev_text}", "",
             "## 6. Did this cause leakage in the model?", "",
             "No. The affected rows were removed at the dataset-build stage, before the split, before feature selection and before any fit; they are in no "
             "training, validation or test partition of the reference run. In the rows that were kept, the last-record dates are before the index day, so the "
             "predictors built from those sources contain only pre-index records. The price is a smaller and outcome-related subset (section 13), not leakage.", "",
             "## 7. STRICT result (reference)", "",
             f"{len(rs.get('predictors', []))} predictors from registries and demographics: {', '.join(rs.get('predictors', []))}. Test AUROC {_fmt(rs.get('auroc'))} "
             f"({rs.get('auroc_ci')}), PR-AUC {_fmt(rs.get('pr_auc'))}, Brier {_fmt(rs.get('brier'))}, calibration slope {_fmt(rs.get('calibration_slope_uncalibrated'))}, "
             f"O:E {_fmt(rs.get('oe_ratio_uncalibrated'))}. Non-zero coefficients: {', '.join(rs.get('selected_raw_features', [])) or 'none'}.", "",
             "## 8. EXTENDED result (reference)", "",
             f"{len(re_.get('predictors', []))} predictors (STRICT + {', '.join(added_to_ext)}). Test AUROC {_fmt(re_.get('auroc'))} ({re_.get('auroc_ci')}), PR-AUC "
             f"{_fmt(re_.get('pr_auc'))}, Brier {_fmt(re_.get('brier'))}, calibration slope {_fmt(re_.get('calibration_slope_uncalibrated'))}, O:E {_fmt(re_.get('oe_ratio_uncalibrated'))}. "
             f"Non-zero coefficients: {', '.join(re_.get('selected_raw_features', [])) or 'none'}.", "",
             f"Reading the AUROC of {_fmt(re_.get('auroc'))} correctly: it describes how well the EXTENDED model ranked members of the **D-00-clean test subset** on "
             f"{res.index_date}. It is not a measurement on the full Meuhedet target population, because {n_d00} rows with a different outcome rate were excluded; "
             "how far the two differ is what the SAFE-ALL-ROWS analysis brackets (section 9).", "",
             f"The change from STRICT to EXTENDED ({'+' if (auroc_gain or 0) >= 0 else ''}{_fmt(auroc_gain)} AUROC on the same test rows) comes with the added predictors "
             f"{', '.join(added_to_ext)}; of these, {', '.join(contributing) or 'none'} kept a non-zero coefficient. This is a statement about predictive contribution on "
             "this label and this subset, not about causes and not a validation of the clinical definitions.", "",
             "## 9. SAFE-ALL-ROWS result", ""]
    if safe is not None:
        po = pop.get("rows_only_in_safe", {})
        lines += [f"All {safe['n_rows']} labelled rows ({safe['n_patients']} members, {safe['n_events']} falls, {_pct(safe['event_prevalence'])}) with the "
                  f"{safe['n_predictors']} predictors the file shows to be temporally safe: {', '.join(safe_feats)}. Excluded on purpose: "
                  f"{', '.join(f'{f} ({prov_reason(prov, f)})' for f in excluded) or 'none'}. Rows added back relative to the reference: {po.get('n_rows')} "
                  f"(prevalence {_pct(po.get('prevalence'))}); rows removed by the registry same-day evidence: {pop.get('rows_only_in_reference', {}).get('n_rows')}.",
                  "", f"Test AUROC {_fmt(safe['auroc'])} ({safe['auroc_ci']}), PR-AUC {_fmt(safe['pr_auc'])}, Brier {_fmt(safe['brier'])}, calibration slope "
                  f"{_fmt(safe['calibration_slope_uncalibrated'])}, O:E {_fmt(safe['oe_ratio_uncalibrated'])}. Non-zero coefficients: {', '.join(safe['selected_raw_features']) or 'none'}.", "",
                  "This result is on a different set of rows than the reference, with a redrawn split: it is not paired with the reference, and any difference "
                  "mixes the predictor set (two features removed) with the population (rows added back). The ablations below separate the two.", ""]
        for key, r in res.records.items():
            if key.startswith("ablation_"):
                lines.append(f"- {key} on the reference rows (same test partition): AUROC {_fmt(r['auroc'])} ({r['auroc_ci']}); non-zero: {', '.join(r['selected_raw_features']) or 'none'}")
        pe = res.population and _population_effect(res, {}, excluded)
        if pe and pe.get("text"):
            lines += ["", pe["text"]]
    else:
        lines.append(f"SAFE-ALL-ROWS was not run: {res.skipped.get(SAFE_KEY, 'see SENSITIVITY_README.md')}")
    lines += ["", "## 10. Which features matter most predictively", "",
              "Ranking by permutation importance in the reference EXTENDED run (how much the VALIDATION AUROC drops when the predictor is shuffled) with the bootstrap "
              "selection frequency (share of refits keeping a non-zero coefficient). Descriptive, on this label and subset; not causal, not clinical importance."]
    lines += [f"- {t}" for t in top_ext] or ["- n/a"]
    lines += ["", "## 11. What warnings occurred", ""]
    for key, a in res.audits.items():
        lines.append(f"- {key}: {a['verdict']}; final-fit warnings {a['final_fit']['warnings'] or 'none'}; stability replicates {a['replicates']['stability_warnings'] or 'none'}; "
                     f"optimism replicates {a['replicates']['optimism_warnings'] or 'none'}; lambda* at grid position {a['final_fit']['grid']['lambda_star_index']} of {a['final_fit']['grid']['n_lambda']}")
    lines += ["", "See convergence_audit.md for what each warning means and whether a solver or grid change is warranted (it is proposed only when the selected "
              "lambda sits on the grid boundary).", "",
              "## 12. What conclusions are currently justified", "",
              f"- On the D-00-clean subset of {res.index_date}, the EXTENDED model separated members who fell within 180 days from those who did not with test AUROC "
              f"{_fmt(re_.get('auroc'))} ({re_.get('auroc_ci')}); STRICT reached {_fmt(rs.get('auroc'))} on the same test rows.",
              "- The excluded rows were removed before training; the reported metrics are free of index-day leakage.",
              f"- The added APPROXIMATE predictors carry predictive contribution on this label (non-zero coefficients: {', '.join(contributing) or 'none'}).",
              *ablation_lines,
              (f"- With every labelled row and without {', '.join(excluded) or 'the implicated predictors'}, the SAFE-ALL-ROWS model reached AUROC {_fmt(safe['auroc'])} "
               f"({safe['auroc_ci']}) on its own test partition; compared with the same predictor set on the reference rows this is a population effect and is not paired." if safe else
               "- SAFE-ALL-ROWS did not run; no statement about the full population is possible yet."),
              "- Everything above is exploratory: unvalidated mappings, a 180-day Meuhedet outcome, one snapshot.", "",
              "## 13. What conclusions are NOT yet justified", "",
              "- That the model performs this way on the full Meuhedet target population: the reference excluded an outcome-related subset, and the full-population "
              "model with the full predictor set (FINAL_DWH_FIXED) has not been run.",
              "- That any predictor causes falls, or that the approximate Meuhedet definitions match the eFalls definitions.",
              "- Anything about eFalls reproduction: the 365-day ED/admission outcome, fracture and fragility_fracture, and the fall/fracture code lists are missing.",
              "- Performance at other index dates, or over time (single snapshot).", "",
              "## 14. What should be done when DWH access is available", "",
              "1. Rebuild every diagnosis- and fall-derived column with `Event_Date < Index_Date` (exact predicates in audit/timing_diagnostic.md) and re-export.",
              "2. Run `meuhedet-explore` again with the default `--index-day-records fail`; it must report 0 D-00 violations - that run is FINAL_DWH_FIXED.",
              "3. Confirm First_Registry_Date semantics (Q-M-11) and the fall/fracture event list (Q-M-06).",
              "4. Compare FINAL_DWH_FIXED with the reference and SAFE-ALL-ROWS in this same table; only then is a statement about the full population possible.",
              "5. Run `--all-index-dates` for the temporal design once several snapshots carry a fully observed 180-day outcome.", "",
              "## Known mapping limitations (unchanged)", ""] + [f"- {c}" for c in MAPPING_CAVEATS] + ["", f"Scientific status: {SCIENTIFIC_STATUS}. {wm}."]
    return "\n".join(lines) + "\n"


def _differs(lp: dict[str, Any]) -> bool:
    a, c = lp.get("affected"), lp.get("clean")
    return a is not None and c is not None and abs(float(a) - float(c)) >= 0.01


def prov_reason(prov: dict[str, Any], feature: str) -> str:
    for e in prov.get("features", []):
        if e["feature"] == feature:
            return e["provenance_class"]
    return "excluded"


def render_readme(res: SensitivityResult, wm: str) -> str:
    lines = [f"# Sensitivity package - {SAFE_STATE} and ablations", "", f"**{wm}**", "", f"Reference (read-only, verified unchanged: {res.integrity.get('unchanged')}): `{res.reference_dir}`",
             f"Input: {res.input_file}; index date {res.index_date}; pepper fingerprint {res.pepper_sha256}", "", "## Analyses", ""]
    for key, r in res.records.items():
        lines.append(f"- {key} [{r['state']}]: {r['n_predictors']} predictors, {r['n_rows']} rows, AUROC {_fmt(r['auroc'])} ({r['auroc_ci']}); run `{r['run_id']}`")
    for key, why in res.skipped.items():
        lines.append(f"- {key}: SKIPPED - {why}")
    lines += [f"- {FINAL_KEY} [{FINAL_STATE}]: {NOT_RUN}", "", "## Files", "",
              "REAL_DATA_EXPLORATORY_REPORT.md (plain language), SENSITIVITY_COMPARISON.md/.json, feature_provenance.md/.json, convergence_audit.md/.json, "
              "split_audit_<analysis>.md/.json, adequacy_<analysis>.md/.json, feature_report_<analysis>.md/.csv, audit/timing_diagnostic.md/.json, "
              "reference_integrity.json, runs/<run>/ (metrics.json, report.html, coefficients.csv, feature_stability.csv, calibration.csv, plots/).", "",
              "Never share: datasets/, id_pepper.txt, runs/*/predictions_*.parquet."]
    return "\n".join(lines) + "\n"
