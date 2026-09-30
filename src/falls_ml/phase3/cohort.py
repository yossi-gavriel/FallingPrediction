"""S01 Phase 3 cohort: FULL_LABELED (no row removed because of a record date) = the verified reference D00_CLEAN cohort + the rows the reference
build removed for on/after-index diagnosis / fall dates (D-00 rows).

1. The unchanged Phase 2 reconstruction (``falls_ml.phase2.cohort.build_cohort``) rebuilds D00_CLEAN, proves it identical to the reference run
   (rows and predictor values), reproduces the reference split (TEST sha256 must match) and drops TEST in memory.
2. FULL_LABELED is taken with the unchanged D-00 cohort definition (``falls_ml.d00.dependency.cohort_masks``: eligible, on the index date, a
   non-NULL label - label availability only, never its value). Its D00_CLEAN part must equal the reference cohort row count.
3. The D-00 rows get a partition from a pre-declared, outcome-blind keyed hash of the pseudonymised id (train / validation / test shares from
   the frozen settings); TEST-assigned rows are dropped in memory with the reference TEST rows.
4. Their canonical BASELINE_15 values are built by the unchanged adapter feature code (``MeuhedetWideDatasetAdapter._feature``), whose output is
   verified to equal the reference canonical values on every D00_CLEAN row first.
The pepper is read, never created. Nothing is written outside the run folder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2.cohort import IDENTIFIER_COLUMNS, build_cohort, death_label_audit, read_reference_pepper
from falls_ml.phase2.state import Phase2Stop


def partition_of(research_id: str, salt: str, train: float, validation: float) -> str:
    """Outcome-blind, deterministic partition of an extra (D-00) row from its pseudonymised id."""
    u = int(hashlib.sha256(f"{salt}|{research_id}".encode("utf-8")).hexdigest()[:13], 16) / float(16**13)
    return "train" if u < train else "validation" if u < train + validation else "test"


def canonical_baseline(adapter: Any, mapping: Any, df: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Canonical BASELINE_15 values built by the unchanged adapter feature code (no new parser, no new rule)."""
    from falls_ml.data.meuhedet_wide import BuildReport

    rep = BuildReport(index_date="phase3", n_rows_input=len(df))
    problems: list[str] = []
    out = pd.DataFrame(index=df.index)
    for f in mapping.features:
        if f.canonical in names:
            out[f.canonical] = adapter._feature(f, df, rep, problems)
    return out[names], problems


def _same(a: pd.Series, b: pd.Series) -> bool:
    if a.dtype == object or pd.api.types.is_string_dtype(a) or pd.api.types.is_string_dtype(b):
        return bool((a.astype(str).to_numpy() == b.astype(str).to_numpy()).all())
    return bool(np.allclose(pd.to_numeric(a).to_numpy(dtype=float), pd.to_numeric(b).to_numpy(dtype=float), equal_nan=True))


STRUCTURAL = ("missing contract columns", "columns not in the contract", "duplicate columns", "Leakage_Check_Ind", "Definition_Version")
UNREADABLE_PREFIX = "__unreadable__"


def verify_schema(read: Any, contract: Any, strict_columns: list[str]) -> dict[str, Any]:
    """Exact source-schema verification with the validated contract reader: 221 columns, contract types, declared date layouts (never inferred),
    sentinels, NOT NULL, allowed values, VIEW constants. A STRUCTURAL problem (column set, duplicates, VIEW constants, the VIEW's own leakage flag)
    is a HARD stop. A cell-level problem in a column Phase 3 reads (a value that cannot be read as its contract type - e.g. '00:00.0' in a date -
    or a value outside the allowed set) never becomes a silent NULL / 'not assessed': the cell is marked UNREADABLE and the row is UNKNOWN for that
    source (see :func:`cell_problem_masks`). Only counts are kept (no cell value)."""
    from falls_ml.data.meuhedet_wide import validate_wide_contract

    rep = validate_wide_contract(read.frame, contract, wrong_type=read.wrong_type, wrong_type_examples=read.wrong_type_examples,
                                 date_formats=read.date_formats, strict_columns=strict_columns, strict=False)
    structural = [p for p in rep.problems if p.startswith(STRUCTURAL)]
    out = {"n_rows": rep.n_rows, "n_columns": rep.n_columns, "n_contract_columns": len(contract.names), "source_format": read.source_format,
           "declared_date_formats": list(read.declared_date_formats), "date_formats_used": {k: dict(v) for k, v in rep.date_formats.items()},
           "wrong_type_counts": dict(rep.wrong_type), "invalid_value_counts": dict(rep.invalid_values), "non_null_violations": dict(rep.non_null_violations),
           "beyond_ns_range_dates": {k: {kk: vv for kk, vv in v.items() if kk not in ("min", "max")} for k, v in rep.date_range.items()},
           "n_structural_problems": len(structural), "n_cell_problems_in_read_columns": len(rep.problems) - len(structural),
           "n_warnings_other_columns": len(rep.warnings), "problems": [p.split("; offending values")[0] for p in rep.problems],
           "warnings": [w.split("; offending values")[0] for w in rep.warnings][:50], "strict_columns": len(strict_columns),
           "cell_problem_rule": "a cell of a read column that cannot be read as its contract type or is outside the allowed values is UNREADABLE: its row is "
                                "UNKNOWN for that source (never NULL-as-absent)"}
    if structural:
        raise Phase2Stop("SCHEMA_VIOLATION", "the extract's structure differs from the wide-table contract", structural[:15])
    return out


def _raw_columns(src: Path, cols: list[str], encoding: str = "utf-8", sep: str = ",", sheet: str | None = None) -> pd.DataFrame:
    """The raw cells of ``cols`` exactly as the validated reader reads them (same NULL literals, no inference)."""
    from falls_ml.data.meuhedet_wide import EXCEL_SUFFIXES, NA_VALUES

    suffix = src.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(src, columns=cols)
    if suffix in EXCEL_SUFFIXES:
        target: int | str = 0 if sheet is None else (int(sheet) if str(sheet).isdigit() else str(sheet))
        return pd.read_excel(src, sheet_name=target, dtype=object, na_values=list(NA_VALUES), keep_default_na=False, engine="openpyxl", usecols=cols)
    return pd.read_csv(src, dtype="string", na_values=list(NA_VALUES), keep_default_na=False, encoding=encoding, sep=sep, usecols=cols)


def cell_problem_masks(src: Path, read: Any, contract: Any, columns: list[str]) -> pd.DataFrame:
    """Per-cell UNREADABLE masks (rows of ``read.frame``) for the read columns with a cell-level problem: the raw cell is present but the
    validated coercion (``coerce_wide_types_report``) could not represent it, or the typed value is outside the contract's allowed values."""
    from falls_ml.data.meuhedet_wide import coerce_wide_types_report

    frame = read.frame
    out = pd.DataFrame(index=frame.index)
    bad_type = [c for c in columns if int(read.wrong_type.get(c, 0))]
    if bad_type:
        raw = _raw_columns(src, bad_type)
        raw.index = frame.index
        coerced, _ = coerce_wide_types_report(raw, contract)
        for c in bad_type:
            out[c] = (raw[c].notna() & coerced[c].isna()).to_numpy(dtype=bool)
    for c in columns:
        cc = contract.get(c)
        if cc.allowed is None or c not in frame.columns:
            continue
        s = frame[c]
        if cc.is_integer:
            ok = s.isna() | s.astype("Float64").isin([float(v) for v in cc.allowed])
        elif cc.pandas_dtype == "float64":
            ok = s.isna() | s.astype("float64").isin([float(v) for v in cc.allowed])
        else:
            ok = s.isna() | s.astype("string").isin([str(v) for v in cc.allowed])
        bad = ~ok.fillna(False).to_numpy(dtype=bool)
        if bad.any():
            out[c] = (out[c].to_numpy(dtype=bool) | bad) if c in out.columns else bad
    return out


def build_phase3_cohort(src: Path, ref_dir: Path, *, contract: Any, mapping: Any, spec: Any, pepper_file: str | Path | None, scratch: Path,
                        input_sha: str, split_cfg: dict[str, Any], schema_columns: list[str] | None = None, time_contract: Any = None,
                        dictionary: Any = None, d00: Any = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    from falls_ml.d00.dependency import cohort_masks
    from falls_ml.data.meuhedet_wide import MeuhedetWideDatasetAdapter, read_wide_extract_report

    read = read_wide_extract_report(src, contract)
    schema = verify_schema(read, contract, list(schema_columns or []))
    ref_res = build_cohort(src, ref_dir, contract=contract, mapping=mapping, spec=spec, pepper_file=pepper_file, scratch=scratch, read=read,
                           input_sha=input_sha)
    work, facts = ref_res.frame, dict(ref_res.facts)
    index_date = facts["index_date"]
    frame = read.frame
    masks = cohort_masks(frame, mapping, index_date)
    if int(masks.counts["D00_CLEAN"]) != int(facts["cohort_rows"]):
        raise Phase2Stop("COHORT_MISMATCH", "the D-00 cohort definition does not give the reference D00_CLEAN row count",
                         [f"cohort_masks D00_CLEAN {masks.counts['D00_CLEAN']}", f"reference {facts['cohort_rows']}"])
    # ---- V9 exact cohort reconciliation against the reference build (hard stop if it does not add up)
    rb = ref_res.ref.get("build") or {}
    lab_col = mapping.outcome["label_column"]
    elig_on = (frame[mapping.identity["index_date"]].dt.normalize() == pd.Timestamp(index_date)).fillna(False) & (frame["Is_Eligible_Cohort"] == 1).fillna(False)
    null_lab = elig_on & frame[lab_col].isna()
    reasons = frame.loc[null_lab, "Label_Reason_180D"].astype("string").fillna("NULL").value_counts().to_dict() if "Label_Reason_180D" in frame else {}
    n_full, n_clean = int(masks.counts["FULL_LABELED"]), int(masks.counts["D00_CLEAN"])
    n_d00 = n_full - n_clean
    ref_removed = rb.get("n_rows_timing_violation")
    recon = {"extract_rows": int(len(frame)), "rows_on_index_date": int(masks.counts["rows_on_index_date"]), "eligible_on_index_date": int(masks.counts["eligible"]),
             "label_null_eligible": int(null_lab.sum()), "label_null_by_reason": {str(k): int(v) for k, v in reasons.items()},
             "full_labeled": n_full, "d00_clean": n_clean, "d00_rows": n_d00, "reference_cohort_rows": int(facts["cohort_rows"]),
             "reference_rows_removed_for_d00": ref_removed, "reference_label_null_rows": rb.get("n_rows_label_null"),
             "reference_ineligible_rows": rb.get("n_rows_ineligible")}
    checks = {"full_equals_clean_plus_d00": n_full == n_clean + n_d00, "clean_equals_reference": n_clean == int(facts["cohort_rows"]),
              "d00_equals_reference_removed": ref_removed is None or int(ref_removed) == n_d00,
              "label_null_equals_reference": rb.get("n_rows_label_null") is None or int(rb["n_rows_label_null"]) == int(null_lab.sum())}
    recon.update({"checks": checks, "passed": all(checks.values()),
                  "residual_selection_note": ("FULL_LABELED still requires a usable 180-day label: rows censored within the window (death, leaving the HMO) "
                                              "are excluded by the warehouse label - a post-index criterion that is reported as a limitation, not removed")})
    if not recon["passed"]:
        raise Phase2Stop("COHORT_RECONCILIATION", "the Phase 3 cohort counts do not reconcile with the reference build",
                         [f"{k}: {v}" for k, v in checks.items() if not v] + [f"counts: {json.dumps({k: v for k, v in recon.items() if k != 'label_null_by_reason'}, default=str)}"])
    pepper, _ = read_reference_pepper(src, ref_dir, pepper_file, facts.get("pepper_sha256"))
    baseline = list(facts["baseline_features"])
    adapter = MeuhedetWideDatasetAdapter(mapping, contract, spec, features=baseline, id_pepper=pepper)

    # ---- the adapter feature code reproduces the reference canonical values on every reference (TRAIN + VALIDATION) row
    rebuilt, problems = canonical_baseline(adapter, mapping, work, baseline)
    if problems:
        raise Phase2Stop("BASELINE_REBUILD", "the canonical baseline could not be rebuilt on the reference rows", problems[:10])
    diff = [f for f in baseline if not _same(rebuilt[f], work[f])]
    if diff:
        raise Phase2Stop("BASELINE_REBUILD", "rebuilt canonical baseline values differ from the reference dataset", diff)

    # ---- the D-00 rows of FULL_LABELED
    extra_m = (masks["FULL_LABELED"] & ~masks["D00_CLEAN"]).to_numpy()
    ex = frame.loc[extra_m].copy()
    ids = ex[mapping.identity["research_id"]].astype("string")
    rid = ids.map(lambda v: hashlib.sha256(f"{pepper}|{v}".encode("utf-8")).hexdigest()[:20] if pd.notna(v) else None)
    if rid.isna().any() or rid.duplicated().any():
        raise Phase2Stop("ROW_MAPPING", "D-00 rows without a unique member id")
    salt, trs, vas = str(split_cfg["salt"]), float(split_cfg["train"]), float(split_cfg["validation"])
    part = rid.map(lambda r: partition_of(str(r), salt, trs, vas))
    overlap = set(rid) & set(work["research_id"].astype(str))
    if overlap:
        raise Phase2Stop("PATIENT_OVERLAP", f"{len(overlap)} D-00 patients also appear in the reference cohort")
    n_test = int((part == "test").sum())
    test_sha = hashlib.sha256("\n".join(sorted(rid[part == "test"].astype(str))).encode("utf-8")).hexdigest()
    keep = (part != "test").to_numpy()
    ex, rid, part = ex.loc[keep], rid[keep], part[keep]            # TEST-assigned D-00 rows dropped here, never read further
    label_col = mapping.outcome["label_column"]
    death_audit_all = death_label_audit(ex, label_col, mapping.identity["index_date"]) if len(ex) else {}
    canon, problems = canonical_baseline(adapter, mapping, ex, baseline)
    if problems:
        raise Phase2Stop("BASELINE_REBUILD", "the canonical baseline could not be built on the D-00 rows", problems[:10])
    y = pd.to_numeric(ex[label_col], errors="coerce")
    if y.isna().any() or not set(y.unique()) <= {0, 1}:
        raise Phase2Stop("LABEL_MISMATCH", "D-00 rows with an unusable label inside FULL_LABELED")
    label_cols = [c.name for c in contract.columns if c.role == "LABEL"]
    raw = ex.drop(columns=[c for c in (*IDENTIFIER_COLUMNS, *label_cols) if c in ex.columns])
    extra = pd.DataFrame({"research_id": rid.astype(str).to_numpy(), "index_date": pd.to_datetime(ex[mapping.identity["index_date"]]).dt.normalize()
                          .astype(work["index_date"].dtype).to_numpy(), "partition": part.to_numpy(), "y": y.astype(int).to_numpy()})
    for f in baseline:
        extra[f] = canon[f].to_numpy()
        try:
            extra[f] = extra[f].astype(work[f].dtype)
        except (TypeError, ValueError):
            pass
    extra = pd.concat([extra.reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
    ref_part = work.drop(columns=["row_id"]).assign(d00_row=False)
    extra = extra.assign(d00_row=True)[ref_part.columns]
    allw = pd.concat([ref_part, extra], ignore_index=True)
    # the index-date eligible extract rows, keyed by the pseudonymised id (to link row-level checks to the Phase 3 rows)
    on = (frame[mapping.identity["index_date"]].dt.normalize() == pd.Timestamp(index_date)).fillna(False).to_numpy() & \
        (frame["Is_Eligible_Cohort"] == 1).fillna(False).to_numpy()
    rid_all = frame.loc[on, mapping.identity["research_id"]].astype("string").map(
        lambda v: hashlib.sha256(f"{pepper}|{v}".encode("utf-8")).hexdigest()[:20] if pd.notna(v) else None)
    part_of = dict(zip(allw["research_id"].astype(str), allw["partition"].astype(str)))
    tv_part = rid_all.astype(str).map(part_of)
    tv_rows = tv_part.notna().to_numpy()                 # TRAIN + VALIDATION only: TEST rows (and their labels) are never passed on
    if time_contract is not None:
        from falls_ml.phase3.timecontract import verify

        tcres = verify(frame.loc[on].loc[tv_rows], tv_part[tv_rows], dictionary=dictionary, d00=d00, tc=time_contract)
    else:
        tcres = None
    # per-cell UNREADABLE masks of the read columns (linked by the pseudonymised id of the index-date rows)
    cellm = cell_problem_masks(src, read, contract, list(schema_columns or []))
    if len(cellm.columns):
        for c in cellm.columns:
            m = dict(zip(rid_all.astype(str), cellm.loc[on, c].to_numpy(dtype=bool)))
            allw[UNREADABLE_PREFIX + c] = allw["research_id"].astype(str).map(m).fillna(False).astype(bool)
        schema["unreadable_cells_in_population"] = {c: int(allw[UNREADABLE_PREFIX + c].sum()) for c in cellm.columns}
    order = np.lexsort((allw["research_id"].astype(str).to_numpy(), (allw["partition"] != "train").to_numpy()))
    allw = allw.iloc[order].reset_index(drop=True)
    allw.insert(0, "row_id", np.arange(len(allw), dtype=np.int64))
    if allw["research_id"].duplicated().any():
        raise Phase2Stop("PATIENT_OVERLAP", "a patient appears twice in the Phase 3 population")

    counts = {}
    for p in ("train", "validation"):
        m = allw["partition"] == p
        counts[p] = {"n_rows": int(m.sum()), "n_events": int(allw.loc[m, "y"].sum()), "n_d00_rows": int((m & allw["d00_row"]).sum()),
                     "n_d00_events": int(allw.loc[m & allw["d00_row"], "y"].sum()), "n_reference_rows": int((m & ~allw["d00_row"]).sum())}
    facts.update({"population": "FULL_LABELED", "full_labeled_rows": int(masks.counts["FULL_LABELED"]), "d00_rows_in_full_labeled": int(extra_m.sum()),
                  "d00_rows_by_guard_column": {k: v for k, v in masks.counts.items() if k.startswith("d00_rows_by_")},
                  "extra_rows_split_rule": {"salt_sha256": hashlib.sha256(salt.encode()).hexdigest(), "train": trs, "validation": vas,
                                            "test": round(1 - trs - vas, 6), "outcome_blind": True},
                  "extra_test_rows_dropped": n_test, "extra_test_rows_sha256": test_sha, "phase3_partitions": counts,
                  "baseline_rebuilt_equal_reference": True, "label_death_audit_d00_rows": death_audit_all,
                  "test_outcomes_read": False, "schema": schema, "time_contract": tcres, "cohort_reconciliation": recon})
    return allw, facts


# ============================================================================ evaluation history (which partitions were already examined)
def _jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def evaluation_history(ref_dir: Path, protected: list[Path], phase2_out: Path | None, facts: dict[str, Any]) -> dict[str, Any]:
    """Read-only audit of the recorded evaluation history of the reference split. Only small bookkeeping files are opened; no prediction file."""
    test_sha = facts.get("test_rows_sha256")
    entries = []
    for d in [ref_dir, *protected]:
        reg = _jsonl(d / "runs" / "test_evaluation_registry.jsonl")
        same = [e for e in reg if e.get("test_rows_sha256") == test_sha]
        entries.append({"folder": d.name, "test_releases_recorded": len(reg), "test_releases_same_test_rows": len(same),
                        "purposes": sorted({str(e.get("purpose")) for e in reg})})
    p2: dict[str, Any] = {"folder_given": phase2_out is not None}
    if phase2_out is not None:
        vo = phase2_out / "VALIDATION_OPENED.json"
        st = phase2_out / "RUN_STATE.json"
        p2.update({"validation_opened_file_present": vo.is_file(),
                   "run_status": (json.loads(st.read_text(encoding="utf-8")).get("status") if st.is_file() else None)})
    total = sum(e["test_releases_same_test_rows"] for e in entries)
    return {
        "TEST": {"status": "EXAMINED - NOT INDEPENDENT", "recorded_releases_of_the_reference_test_rows": total, "by_folder": entries,
                 "phase3_use": "never loaded: dropped in memory right after the split hash check (and the TEST-assigned D-00 rows likewise)"},
        "VALIDATION": {"status": "REUSED - NOT INDEPENDENT",
                       "known_uses": ["Phase 1 meuhedet-explore: recalibration of the served models and model reporting on VALIDATION",
                                      "Phase 1 D-00 / model reports: VALIDATION permutation importance",
                                      "Phase 2: one-shot VALIDATION evaluation of its frozen selection (running independently on the work PC)"],
                       "phase2": p2,
                       "phase3_use": "secondary one-shot descriptive check of the frozen Phase 3 selection only; never used to select, tune or recalibrate"},
        "TRAIN": {"status": "DEVELOPMENT", "phase3_use": "every recovery gate, screening, tuning, selection, stability, ablation and the primary "
                                                         "internal estimate (nested grouped cross-validation, outer out-of-fold)"},
        "conclusion": ("No partition of the current retrospective extract is an untouched holdout. Phase 3 reports INTERNAL development estimates "
                       "(nested cross-validation inside TRAIN) and a descriptive check on the reused VALIDATION partition; independent evidence needs "
                       "a new temporal extract (index dates after the current label windows, rebuilt with Event_Date < Index_Date) or an external one."),
    }
