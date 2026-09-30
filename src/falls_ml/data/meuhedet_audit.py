"""Non-identifying aggregate audit of a Meuhedet wide-table extract (run inside the approved environment; spec M-13).

Output is JSON + Markdown with counts, distributions and quantiles only: no identifiers, no row-level values, no free-text
identifiers (event ids), and category / cross-tab cells below ``min_cell`` are suppressed. The report can therefore leave the
clinical data environment for review while the extract stays inside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_wide import DATE_DTYPE, MODELING_DATE_DTYPE, PREDICTOR_ROLES, WideContract, WideMapping, validate_wide_contract
from falls_ml.features.spec import FeatureSpec

SUPPRESSED = "<min_cell"
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _cell(n: int, min_cell: int) -> int | str:
    n = int(n)
    return n if n == 0 or n >= min_cell else SUPPRESSED


def _counts(s: pd.Series, min_cell: int, *, top: int = 25, dropna: bool = False) -> dict[str, Any]:
    vc = s.value_counts(dropna=dropna)
    out = {("NULL" if pd.isna(k) else str(k)): _cell(v, min_cell) for k, v in vc.head(top).items()}
    if len(vc) > top:
        out["_other_levels"] = int(len(vc) - top)
    return out


def _numeric_summary(s: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(s, errors="coerce").astype("float64").dropna()
    if v.empty:
        return {"n": 0}
    q = v.quantile(list(QUANTILES))
    return {"n": int(len(v)), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "median": float(v.median()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, **{f"p{int(p * 100)}": float(q[p]) for p in QUANTILES}}


def _crosstab(a: pd.Series, b: pd.Series, min_cell: int) -> dict[str, dict[str, Any]]:
    tab = pd.crosstab(a.astype("string").fillna("NULL"), b.astype("string").fillna("NULL"))
    return {str(r): {str(c): _cell(tab.loc[r, c], min_cell) for c in tab.columns} for r in tab.index}


def audit_wide_extract(df: pd.DataFrame, contract: WideContract, mapping: WideMapping, spec: FeatureSpec, *, index_date: str | None = None,
                       min_cell: int = 10, wrong_type: dict[str, int] | None = None,
                       wrong_type_examples: dict[str, list[str]] | None = None,
                       date_formats: dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
    """Aggregate audit of an extract already cast to the contract types (``read_wide_extract``)."""
    id_col, idx_col = mapping.identity["research_id"], mapping.identity["index_date"]
    rep: dict[str, Any] = {"audit": "meuhedet_wide_extract", "min_cell_suppression": min_cell, "contract": {"name": contract.name, "version": contract.version,
                           "sha256": contract.content_sha256}, "mapping": {"name": mapping.name, "version": mapping.version, "sha256": mapping.content_sha256},
                           "feature_spec": {"name": spec.name, "version": spec.version, "sha256": spec.content_sha256}}
    contract_report = validate_wide_contract(df, contract, wrong_type=wrong_type, wrong_type_examples=wrong_type_examples,
                                             date_formats=date_formats, strict=False)
    # ---- overview
    idx = df[idx_col].dt.normalize() if idx_col in df.columns else pd.Series(pd.NaT, index=df.index)
    ov: dict[str, Any] = {"row_count": int(len(df)), "unique_patient_count": int(df[id_col].nunique()) if id_col in df.columns else None,
                          "index_dates": {(str(k.date()) if pd.notna(k) else "NULL"): int(v) for k, v in idx.value_counts(dropna=False).items()},
                          "duplicate_patient_index_rows": int(df.duplicated(subset=[id_col, idx_col]).sum()) if id_col in df.columns else None}
    for col in ("Definition_Version", "Snapshot_Confidence", "Index_Date_Status", "Exclusion_Reason", "Followup_End_Reason", "Death_Identification_Status"):
        if col in df.columns:
            ov[col] = _counts(df[col], min_cell)
    if "Is_Eligible_Cohort" in df.columns:
        elig = df["Is_Eligible_Cohort"].fillna(0) == 1
        ov["eligible_cohort_count"] = int(elig.sum())
        if index_date is not None:
            sel = elig & (idx == pd.Timestamp(index_date))
            ov["eligible_on_selected_index_date"] = {"index_date": str(index_date), "n": int(sel.sum())}
    if "Leakage_Check_Ind" in df.columns:
        ov["leakage_check_ind_positive_rows"] = int((df["Leakage_Check_Ind"].fillna(0) != 0).sum())
    for col in ("Birth_Date_Suspect_Ind", "Multiple_Valid_Population_Rows_Ind", "Fall_On_Index_Date_Ind", "Is_Deceased_Ind", "Unsettled_Visit_Counter_Ind"):
        if col in df.columns:
            ov[f"{col}_rows"] = _cell((df[col].fillna(0) == 1).sum(), min_cell)
    if "Gender_Code" in df.columns and "Gender_Desc" in df.columns:
        ov["gender_code_x_desc"] = _crosstab(df["Gender_Code"], df["Gender_Desc"], min_cell)
    rep["overview"] = ov
    # ---- cohort profile on the selected index date (eligible rows)
    base = df
    if index_date is not None and "Is_Eligible_Cohort" in df.columns:
        base = df.loc[(df["Is_Eligible_Cohort"].fillna(0) == 1) & (idx == pd.Timestamp(index_date))]
    prof: dict[str, Any] = {"n_rows": int(len(base)), "n_patients": int(base[id_col].nunique()) if id_col in base.columns else None}
    if "Age_At_Index" in base.columns:
        prof["age"] = _numeric_summary(base["Age_At_Index"])
        prof["age_bands"] = _counts(pd.cut(base["Age_At_Index"].astype("float64"), [0, 70, 75, 80, 85, 90, 200], right=False,
                                           labels=["65-69", "70-74", "75-79", "80-84", "85-89", "90+"]).astype("string"), min_cell)
    if "Gender_Code" in base.columns:
        prof["sex"] = _counts(base["Gender_Code"], min_cell)
    if "Snapshot_Confidence" in base.columns:
        prof["snapshot_confidence"] = _counts(base["Snapshot_Confidence"], min_cell)
    src_cols = [c.name for c in contract.columns if c.name.startswith("Has_Source_")]
    prof["source_coverage"] = {c: _cell((base[c].fillna(0) == 1).sum(), min_cell) for c in src_cols if c in base.columns}
    rep["cohort_profile"] = prof
    # ---- outcomes
    outcomes: dict[str, Any] = {}
    for h in ("30D", "180D"):
        lab, reason, ev, end, dtn = (f"Fall_Next_{h}_Ind", f"Label_Reason_{h}", f"Next_Fall_Date_{h}", f"Label_End_{h}", f"Days_To_Next_Fall_{h}")
        if lab not in base.columns:
            continue
        y = base[lab]
        o: dict[str, Any] = {"values": _counts(y, min_cell), "n_labelled": int(y.notna().sum()), "n_events": _cell(int((y == 1).sum()), min_cell),
                             "prevalence_among_labelled": (float((y == 1).sum() / y.notna().sum()) if y.notna().any() else None)}
        if reason in base.columns:
            o["label_reason"] = _counts(base[reason], min_cell)
            if "Is_Deceased_Ind" in base.columns:
                o["label_reason_x_deceased"] = _crosstab(base[reason], base["Is_Deceased_Ind"], min_cell)
        if end in base.columns and idx_col in base.columns:
            o["window_days_observed"] = _counts((base[end].dt.normalize() - base[idx_col].dt.normalize()).dt.days, min_cell)
        if ev in base.columns and idx_col in base.columns:
            evd = base[ev].dt.normalize()
            o["events_before_index"] = int((evd < base[idx_col].dt.normalize()).sum())
            o["events_on_index_day"] = _cell(int((evd == base[idx_col].dt.normalize()).sum()), min_cell)
        if dtn in base.columns:
            o["days_to_event"] = _numeric_summary(base[dtn])
        hc = f"Fall_Next_{h}_HighConf_Ind"
        if hc in base.columns:
            o["high_confidence_values"] = _counts(base[hc], min_cell)
        if "Fall_On_Index_Date_Ind" in base.columns:
            o["fall_on_index_date_x_label"] = _crosstab(base["Fall_On_Index_Date_Ind"], y, min_cell)
        outcomes[f"fall_next_{h.lower()}"] = o
    if "Fall_On_Index_Date_Ind" in base.columns and "Prior_Fall_Count_30D" in base.columns:
        outcomes["fall_on_index_date_x_prior_fall_30d_positive"] = _crosstab(base["Fall_On_Index_Date_Ind"], (base["Prior_Fall_Count_30D"].fillna(0) > 0).astype(int), min_cell)
    needed_365 = ["Fall_Next_365D_Ind", "Label_End_365D", "Has_Full_Followup_365D", "Is_Censored_365D", "Next_Fall_Date_365D"]
    outcomes["efalls_365d_outcome"] = {"exists": all(c in df.columns for c in needed_365), "missing_columns": [c for c in needed_365 if c not in df.columns],
                                       "blockers": list(mapping.outcome.get("efalls_outcome_blockers", [])),
                                       "verdict": "BLOCKED - no eFalls-compatible 365-day ED/admission fall-or-fracture label in this VIEW"
                                       if not all(c in df.columns for c in needed_365) else "columns present - definition still to be verified (M-02, M-08)"}
    if "Max_Invoice_Lag_365D" in base.columns:
        outcomes["max_invoice_lag_365d"] = _numeric_summary(base["Max_Invoice_Lag_365D"])
    rep["outcomes"] = outcomes
    # ---- per-column datatype / missingness / distributions (cohort rows)
    columns: dict[str, Any] = {}
    for c in contract.columns:
        if c.name not in base.columns:
            columns[c.name] = {"present": False}
            continue
        s = base[c.name]
        entry: dict[str, Any] = {"present": True, "sql_type": c.sql, "role": c.role, "semantic_type": c.semantic, "expected_dtype": c.pandas_dtype,
                                 "observed_dtype": str(s.dtype), "n_null": int(s.isna().sum()), "pct_null": round(100.0 * s.isna().mean(), 2) if len(s) else None,
                                 "wrong_type": int((wrong_type or {}).get(c.name, 0)), "invalid_values": int(contract_report.invalid_values.get(c.name, 0)),
                                 "null_means": c.null_means, "timing": c.timing}
        if c.semantic == "identifier":
            entry["n_unique"] = int(s.nunique())
        elif c.is_date:
            d = s.dt.normalize().dropna()
            entry["min"] = str(d.min().date()) if len(d) else None
            entry["max"] = str(d.max().date()) if len(d) else None
            if idx_col in base.columns and c.timing in ("pre_index", "unknown") and c.role != "LABEL":
                day = s.dt.normalize()
                is_sentinel = day.isin([pd.Timestamp(v) for v in c.sentinels]) if c.sentinels else pd.Series(False, index=s.index)
                entry["n_on_or_after_index"] = int(((day >= base[idx_col].dt.normalize()) & ~is_sentinel).sum())  # declared sentinels are not record dates
                if c.sentinels:
                    entry["n_declared_sentinel"] = int(is_sentinel.sum())
            if c.name in contract_report.date_range:
                entry["date_range"] = dict(contract_report.date_range[c.name])
            if c.name in contract_report.date_formats:
                entry["source_date_formats"] = dict(contract_report.date_formats[c.name])
        elif c.semantic in ("binary", "ordinal", "categorical", "quality_control", "label") and (c.allowed is not None or c.is_text or s.nunique() <= 25):
            entry["n_unique"] = int(s.nunique())
            entry["distribution"] = _counts(s, min_cell, dropna=False)
            if c.allowed is not None:
                allowed = {str(int(v)) if c.is_integer else str(v) for v in c.allowed}
                seen = {str(v) for v in s.dropna().unique()}
                entry["unexpected_values"] = sorted(seen - allowed)[:10]
        else:
            entry["n_unique"] = int(s.nunique())
            entry.update(_numeric_summary(s))
        columns[c.name] = entry
    rep["columns"] = columns
    rep["datatype_audit"] = {"wrong_type_total": contract_report.wrong_type_total, "wrong_type": dict(contract_report.wrong_type),
                             "non_null_violations": dict(contract_report.non_null_violations), "invalid_values": dict(contract_report.invalid_values),
                             "wrong_type_examples": {k: list(v) for k, v in contract_report.wrong_type_examples.items()},
                             "date_range": {k: dict(v) for k, v in contract_report.date_range.items()},
                             "date_dtype": {"contract_layer": DATE_DTYPE, "modelling_layer": MODELING_DATE_DTYPE},
                             "declared_date_formats": list(contract.date_formats),
                             "source_date_formats": {k: dict(v) for k, v in contract_report.date_formats.items()},
                             "contract_problems": list(contract_report.problems), "contract_warnings": list(contract_report.warnings),
                             "conversions": {c.name: {"from_sql": c.sql, "to_pandas": c.pandas_dtype, "model": c.model_dtype} for c in contract.columns}}
    # ---- eFalls coverage and per-predictor source audit
    cov = mapping.coverage(spec)
    per: dict[str, Any] = {}
    for f in mapping.features:
        if not f.include_in_baseline:
            continue
        e: dict[str, Any] = {"quality": f.quality, "source_columns": list(f.source_columns), "op": f.op}
        for src in f.source_columns:
            if src in base.columns:
                s = base[src]
                e[src] = {"pct_null": round(100.0 * s.isna().mean(), 2) if len(s) else None}
                if f.op in ("any_positive", "recode"):
                    e[src]["distribution"] = _counts(s, min_cell)
                else:
                    e[src].update(_numeric_summary(s))
        if f.absent_indicator and f.absent_indicator in base.columns:
            e["source_absent_rows"] = _cell((base[f.absent_indicator].fillna(0) == 1).sum(), min_cell)
            if f.op == "count" and f.source_columns[0] in base.columns:
                s = base[f.source_columns[0]]
                e["null_to_zero_rows_by_efalls_rule"] = _cell((s.isna() & (base[f.absent_indicator].fillna(0) == 1)).sum(), min_cell)
                e["unexplained_null_rows"] = int((s.isna() & (base[f.absent_indicator].fillna(0) != 1)).sum())
        per[f.canonical] = e
    rep["efalls_coverage"] = {**cov, "per_predictor": per,
                              "mapping_status_by_feature": {f.canonical: {"status": f.status, "quality": f.quality} for f in mapping.features}}
    # ---- data quality (cohort rows): constant / near-empty columns, duplicate keys
    constant_all = [c.name for c in contract.columns if c.name in base.columns and base[c.name].notna().any() and base[c.name].nunique(dropna=True) <= 1]
    constant = [c for c in constant_all if contract.get(c).role in ("EFALLS_BASELINE_FEATURE", "MEUHEDET_ENHANCED_FEATURE")]
    near_empty = [c.name for c in contract.columns if c.name in base.columns and len(base) and base[c.name].isna().mean() >= 0.99]
    rep["data_quality"] = {"constant_columns": constant, "constant_columns_all_roles": constant_all, "near_empty_columns": near_empty,
                           "duplicate_snapshot_key": int(base["Snapshot_Key"].dropna().duplicated().sum()) if "Snapshot_Key" in base.columns else None,
                           "duplicate_patient_index": int(base.duplicated(subset=[id_col, idx_col]).sum()) if id_col in base.columns else None,
                           "rule": "constant and near-empty columns are warnings (reported, never modelled blindly); duplicate keys stop the build"}
    # ---- leakage audit
    forbidden = [c.name for c in contract.by_role("FORBIDDEN_LEAKAGE") if c.name in df.columns]
    labels = [c.name for c in contract.by_role("LABEL") if c.name in df.columns]
    unknown_timing = [c.name for c in contract.columns if c.role in ("MEUHEDET_ENHANCED_FEATURE",) and c.timing == "unknown" and c.name in df.columns]
    post_index = [c.name for c in contract.columns if c.timing == "post_index" and c.name in df.columns]
    date_viol = {c.name: columns[c.name].get("n_on_or_after_index", 0) for c in contract.columns
                 if c.role in PREDICTOR_ROLES and columns.get(c.name, {}).get("n_on_or_after_index")}  # only columns that could be predictors
    other_dates = {c.name: columns[c.name].get("n_on_or_after_index", 0) for c in contract.columns
                   if c.role not in PREDICTOR_ROLES and columns.get(c.name, {}).get("n_on_or_after_index")}
    leakage = {"forbidden_columns_present": forbidden, "label_columns_present": labels, "post_index_columns_present": post_index,
               "retrospective_or_unknown_timing_enhanced_columns": unknown_timing,
               "predictor_record_dates_on_or_after_index": date_viol,
               "non_predictor_dates_on_or_after_index": other_dates,  # QA / forbidden columns (never predictors): informative only
               "leakage_check_ind_positive_rows": ov.get("leakage_check_ind_positive_rows"),
               "baseline_uses_only_pre_or_at_index_columns": all(contract.get(c).timing in ("pre_index", "at_index")
                                                                   for f in mapping.features if f.include_in_baseline for c in f.source_columns),
               "rule": "forbidden/label/post-index columns are never predictors; unknown-timing columns are blocked until confirmed"}
    leakage["result"] = "PASS" if (not date_viol and not ov.get("leakage_check_ind_positive_rows") and leakage["baseline_uses_only_pre_or_at_index_columns"]) else "FAIL"
    rep["leakage_audit"] = leakage
    return rep


def _md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = "".join("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |\n" for r in rows)
    return head + body


def render_audit_markdown(rep: dict[str, Any]) -> str:
    ov, prof, oc, cov, lk, dt = rep["overview"], rep["cohort_profile"], rep["outcomes"], rep["efalls_coverage"], rep["leakage_audit"], rep["datatype_audit"]
    lines = ["# Meuhedet wide-table extract audit (aggregate, non-identifying)", "",
             f"Contract {rep['contract']['name']} {rep['contract']['version']} · mapping {rep['mapping']['name']} {rep['mapping']['version']} · "
             f"cells below {rep['min_cell_suppression']} suppressed as `{SUPPRESSED}`.", "",
             "## Overview", "", f"- rows: {ov['row_count']}; unique patients: {ov['unique_patient_count']}; duplicate patient×index rows: {ov['duplicate_patient_index_rows']}",
             f"- index dates: {ov['index_dates']}", f"- eligible cohort rows: {ov.get('eligible_cohort_count')}; selected: {ov.get('eligible_on_selected_index_date')}",
             f"- Leakage_Check_Ind positive rows: {ov.get('leakage_check_ind_positive_rows')}", f"- Gender_Code × Gender_Desc: {ov.get('gender_code_x_desc')}", "",
             "## Cohort profile (eligible rows on the selected index date)", "", f"- N rows {prof['n_rows']}, N patients {prof['n_patients']}",
             f"- age: {prof.get('age')}", f"- age bands: {prof.get('age_bands')}", f"- sex (Gender_Code): {prof.get('sex')}", f"- source coverage: {prof.get('source_coverage')}", "",
             "## Outcomes", ""]
    for name in ("fall_next_30d", "fall_next_180d"):
        o = oc.get(name)
        if o:
            lines += [f"### {name}", "", f"- values: {o['values']}; labelled: {o['n_labelled']}; events: {o['n_events']}; prevalence: {o['prevalence_among_labelled']}",
                      f"- label reason: {o.get('label_reason')}", f"- label reason × deceased: {o.get('label_reason_x_deceased')}",
                      f"- window days observed (Label_End − Index): {o.get('window_days_observed')}", f"- events before index: {o.get('events_before_index')}; on index day: {o.get('events_on_index_day')}",
                      f"- days to event: {o.get('days_to_event')}", ""]
    e365 = oc["efalls_365d_outcome"]
    lines += ["### eFalls 365-day outcome", "", f"- exists: {e365['exists']}; missing columns: {e365['missing_columns']}", f"- verdict: **{e365['verdict']}**"]
    lines += [f"  - {b}" for b in e365["blockers"]] + [f"- Max_Invoice_Lag_365D: {oc.get('max_invoice_lag_365d')}", "",
              "## eFalls coverage (baseline proposal)", "",
              f"- expected {cov['n_expected']} ({cov['feature_spec']} {cov['feature_spec_version']}); available {cov['n_available']} ({cov['coverage_pct']}%): "
              f"exact {cov['n_exact']}, high-confidence {cov['n_high_confidence']}, approximate {cov['n_approximate']}; unavailable {cov['n_unavailable']}",
              f"- mandatory unavailable: {cov['mandatory_unavailable']}", f"- **{cov['label']}**", "", "| predictor | quality | source | pct_null / distribution |", "|---|---|---|---|"]
    for k, v in cov["per_predictor"].items():
        srcs = "; ".join(f"{s}: {json.dumps(v[s], default=str)}" for s in v["source_columns"] if s in v)
        lines.append(f"| {k} | {v['quality']} | {', '.join(v['source_columns'])} | {srcs} |")
    lines += ["", "## Datatype audit", "", f"- wrong-type total: {dt['wrong_type_total']}; wrong-type by column: {dt['wrong_type']}",
              f"- wrong-type offending values (date columns; kept as NULL): {dt.get('wrong_type_examples')}",
              f"- dates beyond the nanosecond range ({dt.get('date_dtype')}): {dt.get('date_range')}",
              f"- declared source date formats (contract, trial order): {dt.get('declared_date_formats')}",
              f"- source date format actually read per column: {dt.get('source_date_formats')}",
              f"- DDL NOT NULL violations: {dt['non_null_violations']}", f"- invalid values (outside allowed): {dt['invalid_values']}",
              f"- contract problems: {dt['contract_problems']}", f"- contract warnings: {dt['contract_warnings'][:20]}", "",
              "## Leakage audit", "", f"- result: **{lk['result']}**", f"- forbidden columns present (never predictors): {lk['forbidden_columns_present']}",
              f"- label columns present: {lk['label_columns_present']}", f"- post-index columns present: {lk['post_index_columns_present']}",
              f"- enhanced columns with unknown/retrospective timing (blocked): {lk['retrospective_or_unknown_timing_enhanced_columns']}",
              f"- predictor record dates on/after index: {lk['predictor_record_dates_on_or_after_index']}", "",
              "## Missingness and distributions by column (cohort rows)", ""]
    rows = []
    for name, c in rep["columns"].items():
        if not c.get("present"):
            rows.append({"column": name, "role": "MISSING FROM EXTRACT"})
            continue
        summary = c.get("distribution") or {k: c[k] for k in ("min", "median", "mean", "max") if k in c}
        rows.append({"column": name, "role": c["role"], "semantic": c["semantic_type"], "expected": c["expected_dtype"], "observed": c["observed_dtype"],
                     "pct_null": c["pct_null"], "wrong_type": c["wrong_type"], "invalid": c["invalid_values"], "summary": json.dumps(summary, default=str, ensure_ascii=False)[:160]})
    lines.append(_md_table(rows, ["column", "role", "semantic", "expected", "observed", "pct_null", "wrong_type", "invalid", "summary"]))
    return "\n".join(lines) + "\n"


def write_audit(rep: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j = out / "wide_extract_audit.json"
    m = out / "wide_extract_audit.md"
    j.write_text(json.dumps(rep, indent=2, default=str, ensure_ascii=False), encoding="utf-8", newline="\n")
    m.write_text(render_audit_markdown(rep), encoding="utf-8", newline="\n")
    return j, m
