"""D-00 timing diagnostic for a Meuhedet wide-table extract (aggregate only; runs inside the data environment).

Prediction happens at the START of the index day (mapping ``cohort.prediction_time``), so a predictor may only use source records
dated strictly before it: ``source_event_date < Index_Date``. The wide table carries, per event source, the date of the last
record the VIEW used (``Last_Dx_Date``, ``Last_Fall_Date``, ...). A last-record date on or after the index day therefore proves
that the aggregated predictors of that source (window counts, "since study start" flags, ``Days_Since_*``) were built with
records the model may not see. This module measures that per column and classifies the cause:

- ``A_SAME_DAY_INCLUSION``  every offending record is dated exactly on the index day -> the VIEW's windows use ``<= Index_Date``;
- ``B_FUTURE_RECORDS``      records dated after the index day -> the snapshot was not reconstructed as of the index date (an ETL /
                            snapshot defect, C) or the column does not mean "last pre-index record" (D): the DWH must say which;
- ``A_AND_B``               both patterns.

Nothing is repaired here and nothing can be: the extract holds only the aggregated values, so the pre-index state of an affected row
is not reconstructible without event-level data (no prior date or count is ever fabricated). The output is counts, percentages, day
offsets and per-snapshot totals only - no identifiers, no row-level values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.meuhedet_wide import PREDICTION_TIME, PREDICTOR_RECORD_RULE, PREDICTOR_ROLES, WideContract, WideMapping

#: "last record" date column -> the VIEW's derived day count (consistency evidence for the root cause)
DAYS_SINCE_COMPANION = {
    "Last_Dx_Date": "Days_Since_Last_Diagnosis", "Last_Fall_Date": "Days_Since_Last_Fall", "Last_Visit_Date": "Days_Since_Last_Visit",
    "Last_Assessment_Date": "Days_Since_Last_Assessment", "Last_Ext_Date": "Days_Since_Last_External_Care",
    "Last_Hosp_Date": "Days_Since_Last_Hospitalization", "MEFI_From_Date": "Days_Since_MEFI_Assessment",
    "First_Registry_Date": "Days_Since_First_Registry",
}
OFFSET_BUCKETS = ((0, 0, "0 (index day)"), (1, 7, "1-7"), (8, 30, "8-30"), (31, 90, "31-90"), (91, 365, "91-365"), (366, None, ">365"))
CAUSE_TEXT = {
    "CLEAN": "no record on or after the index day",
    "A_SAME_DAY_INCLUSION": "every offending record is dated exactly on the index day (offset 0): the VIEW's predictor windows include the "
                            "index day (Event_Date <= Index_Date); start-of-index-day prediction requires Event_Date < Index_Date",
    "B_FUTURE_RECORDS": "records dated after the index day: either the snapshot was not reconstructed as of Index_Date (ETL / snapshot "
                        "reconstruction defect, C) or the column does not hold the last pre-index record (semantics, D); the DWH must say which",
    "A_AND_B": "both same-day records (window rule <= Index_Date) and records after the index day (snapshot / semantics defect)",
}
SUPPRESSED = "<min_cell"


def _cell(n: int, min_cell: int) -> int | str:
    n = int(n)
    return n if n == 0 or n >= min_cell else SUPPRESSED


def _by_date(idx: pd.Series, mask: pd.Series, min_cell: int) -> dict[str, int | str]:
    counts = idx[mask].value_counts().sort_index()
    return {str(k.date()): _cell(v, min_cell) for k, v in counts.items()}


def _offsets(off: pd.Series) -> dict[str, Any]:
    v = off.dropna().to_numpy(dtype="float64")
    out: dict[str, Any] = {"n": int(v.size), "min": int(v.min()), "max": int(v.max()), "median": float(np.median(v)),
                           "p90": float(np.quantile(v, 0.9)), "p99": float(np.quantile(v, 0.99)), "buckets": {}}
    for lo, hi, label in OFFSET_BUCKETS:
        out["buckets"][label] = int(((v >= lo) & ((v <= hi) if hi is not None else True)).sum())
    return out


def _related(col: str, contract: WideContract, mapping: WideMapping) -> dict[str, Any]:
    sets = mapping.feature_sets()
    feats = [{"feature": f.canonical, "source_columns": list(f.source_columns), "mapping_quality": f.quality,
              "feature_sets": [name for name, members in sets.items() if f.canonical in members]}
             for f in mapping.features if f.include_in_baseline and f.record_date_column == col]
    group = contract.get(col).group
    phase2 = [c.name for c in contract.columns if group and c.group == group and c.name != col and c.role in PREDICTOR_ROLES]
    return {"efalls_features": feats, "phase2_columns": phase2, "group": group,
            "affected_feature_sets": {name: [f["feature"] for f in feats if name in f["feature_sets"]] for name in sets}}


def _dwh_correction(col: str, related: dict[str, Any], cause: str) -> list[str]:
    source = {"Last_Dx_Date": "diagnoses", "Last_Fall_Date": "fall / fracture events", "Last_Visit_Date": "visits"}.get(col, col)
    derived = [f["source_columns"] for f in related["efalls_features"]]
    flat = sorted({c for cols in derived for c in cols} | set(related["phase2_columns"]))
    lines = [f"Source {source} ({col}): every predictor derived from it must use records with Event_Date < Index_Date "
             f"(equivalently Event_Date <= DATEADD(day, -1, Index_Date)); derived columns: {flat or ['(none mapped)']}",
             f"  {col} = MAX(Event_Date) WHERE Event_Date < Index_Date; {DAYS_SINCE_COMPANION.get(col, 'Days_Since_*')} = DATEDIFF(day, {col}, Index_Date) (>= 1 by construction)",
             "  window counts: Event_Date >= DATEADD(day, -W, Index_Date) AND Event_Date < Index_Date; "
             "'since study start' flags: Event_Date >= '2022-01-01' AND Event_Date < Index_Date"]
    if cause in ("B_FUTURE_RECORDS", "A_AND_B"):
        lines.append(f"  records dated AFTER the index day exist in {col}: the snapshot must be reconstructed as of Index_Date (no source row with "
                     "Event_Date > Index_Date may reach any snapshot column); if the field is documented as something other than the last "
                     "pre-index record, its contract timing must change and it leaves the predictor set")
    return lines


def timing_diagnostic(frame: pd.DataFrame, contract: WideContract, mapping: WideMapping, *, min_cell: int = 10) -> dict[str, Any]:
    """Per pre-index record-date column: counts before / on / after the index day, day-offset distribution, affected eligible rows,
    affected rows by Index_Date, label prevalence among affected vs clean rows (selection-bias evidence should rows be dropped),
    consistency with the VIEW's own ``Days_Since_*`` column, the root-cause class, the predictors traced to the source (per feature
    set and Phase-2 group) and the exact DWH predicate required. Aggregate only."""
    idx_col, label_col = mapping.identity["index_date"], mapping.outcome["label_column"]
    idx = frame[idx_col].dt.normalize()
    eligible = frame["Is_Eligible_Cohort"].fillna(0) == 1 if "Is_Eligible_Cohort" in frame.columns else pd.Series(True, index=frame.index)
    y = frame[label_col] if label_col in frame.columns else pd.Series(pd.NA, index=frame.index, dtype="Int64")
    labelled = eligible & y.notna()
    guard = list(mapping.cohort["predictor_max_record_date"]["columns"])
    others = [c.name for c in contract.columns if c.is_date and c.timing in ("pre_index", "at_index") and c.role in PREDICTOR_ROLES
              and c.name not in guard and c.name in frame.columns]
    evidence = sorted({f.same_day_evidence_column for f in mapping.features if f.include_in_baseline and f.same_day_evidence_column})
    rep: dict[str, Any] = {"diagnostic": "d00_timing", "prediction_time": PREDICTION_TIME, "predictor_record_rule": PREDICTOR_RECORD_RULE,
                           "d00_guard_columns": guard, "min_cell_suppression": min_cell, "n_rows": int(len(frame)),
                           "n_eligible_rows": int(eligible.sum()), "columns": {}}
    any_guard = pd.Series(False, index=frame.index)
    for col in [*guard, *others]:
        if col not in frame.columns:
            continue
        c = contract.get(col)
        d = frame[col].dt.normalize()
        off = (d - idx).dt.days
        obs = d.notna() & idx.notna()
        before, on, after = obs & (off < 0), obs & (off == 0), obs & (off > 0)
        affected = on | after
        if col in guard:
            any_guard |= affected
        cause = "CLEAN" if not affected.any() else ("A_SAME_DAY_INCLUSION" if not after.any() else ("B_FUTURE_RECORDS" if not on.any() else "A_AND_B"))
        entry: dict[str, Any] = {
            "role": c.role, "group": c.group, "timing": c.timing, "in_d00_guard": col in guard, "same_day_evidence_for": [
                f.canonical for f in mapping.features if f.include_in_baseline and f.same_day_evidence_column == col],
            "n_observed": int(obs.sum()), "n_before_index": int(before.sum()),
            "n_on_index": int(on.sum()), "n_after_index": int(after.sum()),
            "pct_on_or_after_of_observed": round(100.0 * affected.sum() / max(int(obs.sum()), 1), 3),
            "pct_on_or_after_of_rows": round(100.0 * affected.sum() / max(len(frame), 1), 3),
            "n_affected_eligible": int((affected & eligible).sum()), "n_affected_eligible_labelled": int((affected & labelled).sum()),
            "positive_offset_days": _offsets(off[affected]) if affected.any() else None,
            "affected_rows_by_index_date": _by_date(idx, affected, min_cell), "affected_eligible_rows_by_index_date": _by_date(idx, affected & eligible, min_cell),
            "root_cause": cause, "root_cause_text": CAUSE_TEXT[cause],
        }
        if labelled.any():
            ya = y[labelled & affected].astype("float64")
            yu = y[labelled & ~affected].astype("float64")
            entry["label_prevalence"] = {"affected": (float(ya.mean()) if len(ya) else None), "clean": (float(yu.mean()) if len(yu) else None),
                                         "n_affected_labelled": _cell(len(ya), min_cell), "n_clean_labelled": int(len(yu)),
                                         "note": "prevalence among eligible labelled rows; a difference means that dropping affected rows is label-related "
                                                 "(selection bias) - the DWH correction keeps every row"}
        comp = DAYS_SINCE_COMPANION.get(col)
        if comp and comp in frame.columns:
            ds = pd.to_numeric(frame[comp], errors="coerce").astype("Float64")
            both = obs & ds.notna()
            consistent = both & (ds == (idx - d).dt.days)
            on_ds, after_ds = ds[on], ds[after]
            entry["days_since_companion"] = {
                "column": comp, "n_both_observed": int(both.sum()),
                "share_equal_to_index_minus_date": round(float(consistent.sum() / max(int(both.sum()), 1)), 4),
                "on_index_rows": {"days_since_0": int((on_ds == 0).sum()), "days_since_null": int(on_ds.isna().sum()),
                                  "days_since_positive": int((on_ds > 0).sum()), "days_since_negative": int((on_ds < 0).sum())},
                "after_index_rows": {"days_since_negative": int((after_ds < 0).sum()), "days_since_null": int(after_ds.isna().sum()),
                                     "days_since_non_negative": int((after_ds >= 0).sum())},
            }
            if on.any() and int((on_ds == 0).sum()) == int(on.sum()):
                entry["root_cause_text"] += f"; {comp} = 0 on every index-day row, so the VIEW's derived day count applies the same inclusive rule"
            if after.any() and int((after_ds < 0).sum()) == int(after.sum()):
                entry["root_cause_text"] += f"; {comp} is negative on every after-index row: the ETL computed it from a future record (C, not D)"
        entry["related_predictors"] = _related(col, contract, mapping)
        entry["dwh_correction"] = _dwh_correction(col, entry["related_predictors"], cause) if cause != "CLEAN" else []
        rep["columns"][col] = entry
    n_any = int((any_guard & eligible).sum())
    causes = {col: rep["columns"][col]["root_cause"] for col in guard if col in rep["columns"]}
    overall = "CLEAN" if all(v == "CLEAN" for v in causes.values()) else ("A_SAME_DAY_INCLUSION" if set(causes.values()) <= {"CLEAN", "A_SAME_DAY_INCLUSION"}
                                                                          else ("B_FUTURE_RECORDS" if set(causes.values()) <= {"CLEAN", "B_FUTURE_RECORDS"} else "A_AND_B"))
    sets = mapping.feature_sets()
    affected_sets = {name: sorted({f for col in guard if col in rep["columns"] and rep["columns"][col]["root_cause"] != "CLEAN"
                                   for f in rep["columns"][col]["related_predictors"]["affected_feature_sets"][name]}) for name in sets}
    rep["same_day_evidence_columns"] = {col: {"features": rep["columns"][col]["same_day_evidence_for"], "n_on_or_after_index_eligible": rep["columns"][col]["n_affected_eligible"]}
                                        for col in evidence if col in rep["columns"]}
    rep["verdict"] = {
        "n_eligible_rows_violating_d00": n_any, "pct_of_eligible": round(100.0 * n_any / max(int(eligible.sum()), 1), 3),
        "violating_eligible_rows_by_index_date": _by_date(idx, any_guard & eligible, min_cell), "root_cause": overall,
        "root_cause_text": CAUSE_TEXT[overall], "root_cause_by_column": causes, "affected_efalls_features_by_feature_set": affected_sets,
        "python_repair_possible": False,
        "python_repair_note": ("The extract holds only the aggregated window values and the last record date of each source: for a row whose last "
                               "record is on/after the index day the pre-index counts, flags and Days_Since_* are not reconstructible without "
                               "event-level data, so nothing is fabricated. A row whose last record date is before the index day has no index-day "
                               "record for that source and is clean. Feature sets that read no affected source (STRICT today) are unaffected, but "
                               "STRICT and EXTENDED share one cohort for a fair comparison, so the rows are excluded (--index-day-records drop_rows, "
                               "counted) or the DWH corrects the windows and every row is kept."),
        "dwh_correction": [line for col in guard if col in rep["columns"] for line in rep["columns"][col]["dwh_correction"]],
        "policy": "index_day_records=fail stops the run; drop_rows removes and counts the violating rows (build report n_rows_timing_violation)",
    }
    if n_any == 0:
        rep["verdict"]["dwh_correction"] = []
    return rep


def render_timing_markdown(rep: dict[str, Any]) -> str:
    v = rep["verdict"]
    lines = ["# D-00 timing diagnostic (aggregate, non-identifying)", "",
             f"Prediction time: **{rep['prediction_time']}** - predictors may use `{rep['predictor_record_rule']}`. Guard columns: {rep['d00_guard_columns']}. "
             f"Cells below {rep['min_cell_suppression']} suppressed as `{SUPPRESSED}`.", "",
             f"**Verdict:** {v['n_eligible_rows_violating_d00']} of {rep['n_eligible_rows']} eligible rows ({v['pct_of_eligible']}%) have a guard "
             f"record date on/after the index day. Root cause: **{v['root_cause']}** - {v['root_cause_text']}.",
             f"Affected eFalls features by feature set: {v['affected_efalls_features_by_feature_set']}.", f"Violating eligible rows by Index_Date: {v['violating_eligible_rows_by_index_date']}", "",
             f"Python repair possible: **{v['python_repair_possible']}**. {v['python_repair_note']}", ""]
    if v["dwh_correction"]:
        lines += ["## DWH / source correction required", ""] + [f"- {l.strip()}" for l in v["dwh_correction"]] + [""]
    lines += ["## Per column", "", "| column | guard | observed | before | on index | after | % on/after (obs) | affected eligible | cause | offsets (min/median/max) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for col, e in rep["columns"].items():
        o = e["positive_offset_days"] or {}
        offsets = "" if not o else "%s/%s/%s" % (o["min"], o["median"], o["max"])
        guard_flag = "yes" if e["in_d00_guard"] else ""
        lines.append(f"| {col} | {guard_flag} | {e['n_observed']} | {e['n_before_index']} | {e['n_on_index']} | {e['n_after_index']} | "
                     f"{e['pct_on_or_after_of_observed']} | {e['n_affected_eligible']} | {e['root_cause']} | {offsets} |")
    for col, e in rep["columns"].items():
        if e["root_cause"] == "CLEAN":
            continue
        lines += ["", f"### {col}", "", f"- cause: {e['root_cause_text']}", f"- offset buckets (days after index): {(e['positive_offset_days'] or {}).get('buckets')}",
                  f"- affected eligible rows by Index_Date: {e['affected_eligible_rows_by_index_date']}", f"- label prevalence affected vs clean: {e.get('label_prevalence')}",
                  f"- Days_Since companion: {e.get('days_since_companion')}",
                  f"- eFalls features reading this source: {e['related_predictors']['efalls_features']}",
                  f"- Phase-2 columns of the same source group ({e['related_predictors']['group']}): {e['related_predictors']['phase2_columns']}"]
    return "\n".join(lines) + "\n"


def write_timing_diagnostic(rep: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / "timing_diagnostic.json", out / "timing_diagnostic.md"
    j.write_text(json.dumps(rep, indent=2, default=str, ensure_ascii=False), encoding="utf-8", newline="\n")
    m.write_text(render_timing_markdown(rep), encoding="utf-8", newline="\n")
    return j, m
