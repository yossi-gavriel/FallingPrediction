"""``falls_ml meuhedet-eda``: reproducible, aggregate EDA of the real wide extract, run locally inside the approved environment.

    RAW EXTRACT -> CONTRACT CHECK -> FULL EDA (every column, unsupervised) -> LEAKAGE / TIMING AUDIT -> COHORT BUILD -> SPLIT
    -> TRAIN-ONLY SUPERVISED EDA -> ADEQUACY          (then: MODELLING -> VALIDATION -> FINAL TEST -> REPORTING = meuhedet-explore)

Scientific separation: (A) data-quality / unsupervised analyses may read the whole file; (B) every target-aware analysis that could
inform a modelling decision reads the TRAIN partition only, through :class:`falls_ml.eda.supervised.TrainPartition`. The split is the
modelling split itself: with ``--reference`` the completed meuhedet-explore run's configuration, pepper and input file are verified and
its test-partition hash must be reproduced exactly; without it the explore template and the pepper sidecar give the split a later
explore run will draw. Outcome-stratified tables on the full cohort are labelled POST-HOC DESCRIPTIVE and never feed a decision.

Privacy: the canonical dataset needed for the split is written to a temporary directory that is deleted before the command returns;
the pepper is never written into the results; every output is aggregate with counts 1..min_cell-1 suppressed; before returning, every
text output is scanned for member ids, snapshot keys, event ids, research ids and the pepper, and the command fails if one is found.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.data.dataset import ModelingDataset, sha256_file
from falls_ml.data.meuhedet_timing import timing_diagnostic
from falls_ml.data.meuhedet_wide import (DEFAULT_MAPPING, PREDICTOR_ROLES, MeuhedetWideDatasetAdapter, build_meuhedet_dataset_from_frame,
                                         load_wide_contract, load_wide_mapping, read_wide_extract_report, validate_wide_contract)
from falls_ml.eda import cohort as cohort_mod
from falls_ml.eda import features as feat_mod
from falls_ml.eda import missingness as miss_mod
from falls_ml.eda import plots as P
from falls_ml.eda import profile as prof_mod
from falls_ml.eda import quality as dq_mod
from falls_ml.eda import split as split_mod
from falls_ml.eda import supervised as sup_mod
from falls_ml.eda import temporal as temp_mod
from falls_ml.eda.common import (BASIS_TEXT, DEFINITIONS, ELIGIBLE_LABELLED, FULL_EXTRACT, MODELLING_COHORT, POSTHOC_DESCRIPTIVE, SPLIT_DIAGNOSTIC,
                                 TRAIN_ONLY, UNIVARIATE_BANNER, write_csv, write_json)
from falls_ml.eda.dictionary import DEFAULT_DICTIONARY, dictionary_table, load_data_dictionary
from falls_ml.errors import ConfigError, DatasetValidationError, LeakageError
from falls_ml.features.spec import load_feature_spec
from falls_ml.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_TEMPLATE = "configs/experiments/meuhedet/explore_180d_template.yaml"
STAGES = ("RAW EXTRACT", "CONTRACT CHECK", "FULL EDA", "LEAKAGE / TIMING AUDIT", "COHORT BUILD", "SPLIT", "TRAIN-ONLY SUPERVISED EDA", "ADEQUACY",
          "MODELLING", "VALIDATION", "FINAL TEST", "REPORTING")
NEXT_STAGES = {"MODELLING": "meuhedet-explore (STRICT / EXTENDED) and meuhedet-sensitivity; not run by meuhedet-eda",
               "VALIDATION": "inside the meuhedet-explore runs (validation partition: recalibration, model checks)",
               "FINAL TEST": "inside the meuhedet-explore runs: the locked test partition is released once (test-evaluation registry)",
               "REPORTING": "run reports, comparison, REAL_DATA_EXPLORATORY_REPORT.md (meuhedet-explore / meuhedet-sensitivity)"}
FILE_BASIS = {  # output file -> (basis, what it holds)
    "data_dictionary.csv": (DEFINITIONS, "explicit data dictionary: every column's type, domain, meaning (status), role, timing, availability, STRICT/EXTENDED use"),
    "column_profile.csv": (FULL_EXTRACT, "per-column profile: counts, missingness, uniqueness, constant status, top values, quantiles, dates, sentinels, parse failures"),
    "numeric_profile.csv": (f"{FULL_EXTRACT} + {MODELLING_COHORT}", "numeric distributions, zero inflation, skewness, extremes, transformations to investigate"),
    "categorical_profile.csv": (f"{FULL_EXTRACT} + {MODELLING_COHORT}", "level counts (NULL as a level), sparse and out-of-contract levels"),
    "missingness.csv": (f"{FULL_EXTRACT} (+ per-basis %)", "missing counts, declared NULL meaning, missingness classes, bands"),
    "missingness_by_domain.csv": (FULL_EXTRACT, "missingness summary per domain"),
    "co_missingness_pairs.csv": (MODELLING_COHORT, "column pairs whose missing indicators move together (phi >= 0.8)"),
    "missingness_patterns.csv": (MODELLING_COHORT, "row-level source-block missingness patterns (small patterns merged)"),
    "missingness_outcome_train.csv": (TRAIN_ONLY, "outcome prevalence among missing vs observed rows per column"),
    "data_quality_findings.csv": (FULL_EXTRACT, "every check with findings (severity, rows, recommendation)"),
    "data_quality_checks.csv": (FULL_EXTRACT, "every check performed, including PASS and NOT_APPLICABLE"),
    "cohort_flow.csv": (f"{FULL_EXTRACT} -> {MODELLING_COHORT}", "cohort funnel with rows, patients, events per step (+ SAFE-ALL-ROWS branch)"),
    "population_profile.csv": (f"{FULL_EXTRACT}, {ELIGIBLE_LABELLED}, {MODELLING_COHORT}", "age, sex, seniority, history, exclusions, labels, censoring, follow-up"),
    "outcome_prevalence.csv": (f"{ELIGIBLE_LABELLED}, {MODELLING_COHORT}, {POSTHOC_DESCRIPTIVE}", "180-day prevalence with Wilson CIs overall and by stratum"),
    "numeric_outcome_train.csv": (TRAIN_ONLY, "numeric columns by outcome: medians, SMD, univariate AUROC"),
    "categorical_outcome_train.csv": (TRAIN_ONLY, "categorical levels: N, events, prevalence, prevalence ratio, sparse / separation flags"),
    "correlation_pairs.csv": (TRAIN_ONLY, "correlated pairs (Spearman, Pearson/phi, Cramér's V) and redundancy level"),
    "redundancy_clusters.csv": (TRAIN_ONLY, "clusters of redundant columns (|Spearman| >= 0.8)"),
    "univariate_association_train.csv": (TRAIN_ONLY, UNIVARIATE_BANNER),
    "table1.csv": (POSTHOC_DESCRIPTIVE, "descriptive Table 1 of the modelling population by future fall (never used for model choice)"),
    "table1_train.csv": (TRAIN_ONLY, "the same Table 1 on the training partition"),
    "temporal_audit.csv": (FULL_EXTRACT, "every date column vs Index_Date + sources without record dates; leakage-risk class"),
    "feature_provenance.csv": (DEFINITIONS, "source -> aggregate column -> canonical feature -> model(s), with the source timing status"),
    "current_15_feature_dictionary.csv": (f"{DEFINITIONS} + {MODELLING_COHORT}", "the EXTENDED predictors: mapping, meaning, timing, D-00, distribution, reference coefficients"),
    "phase2_candidate_features.csv": (f"{MODELLING_COHORT} + {TRAIN_ONLY}", "Meuhedet-native candidate predictors for Phase 2 (NOT trained)"),
    "split_balance.csv": (SPLIT_DIAGNOSTIC, "train / validation / test comparison (SMD vs train)"),
}


@dataclass
class EDAResult:
    out_dir: Path
    eda_dir: Path
    input_file: str
    watermark: str
    synthetic: bool = False
    index_date: str = ""
    policy: str = ""
    stages: dict[str, dict[str, str]] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    plots: dict[str, list[str]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def stage(self, name: str, status: str, detail: str = "") -> None:
        self.stages[name] = {"status": status, "detail": detail}


class _Stop(Exception):
    pass


def _load_reference(ref_dir: Path) -> dict[str, Any]:
    """Read (never write) a completed single-snapshot meuhedet-explore results directory."""
    from falls_ml.meuhedet_explore import RUN_LABELS

    need = ["readiness.json", "split_audit.json", "datasets/extended/manifest.json", f"configs/{RUN_LABELS['extended']}.yaml"]
    missing = [n for n in need if not (ref_dir / n).exists()]
    if missing:
        raise DatasetValidationError(f"{ref_dir} is not a completed meuhedet-explore results directory", [f"missing {m}" for m in missing])
    readiness = json.loads((ref_dir / "readiness.json").read_text(encoding="utf-8"))
    if readiness.get("mode") != "single":
        raise DatasetValidationError("the reference must be a single-snapshot meuhedet-explore run (--index-date)", [f"mode: {readiness.get('mode')}"])
    manifest = json.loads((ref_dir / "datasets" / "extended" / "manifest.json").read_text(encoding="utf-8"))
    build = manifest["audit"]["meuhedet_build"]

    def rep(name: str) -> pd.DataFrame | None:
        p = ref_dir / name
        return pd.read_csv(p) if p.exists() else None

    return {"dir": ref_dir, "readiness": readiness, "split_audit": json.loads((ref_dir / "split_audit.json").read_text(encoding="utf-8")),
            "build": build, "config": ref_dir / "configs" / f"{RUN_LABELS['extended']}.yaml", "input_sha256": build.get("input_sha256"),
            "pepper_sha256": readiness.get("pepper_sha256"), "index_date": str(readiness.get("index_date") or ""),
            "policy": "drop_rows" if int(build.get("n_rows_timing_violation") or 0) else None,
            "features": list(build["feature_set"]["features"]),
            "feature_report_extended": rep("feature_report_extended.csv"), "feature_report_strict": rep("feature_report_strict.csv")}


def run_eda(input_path: str | Path, *, out_dir: str | Path | None = None, index_date: str | None = None, reference_dir: str | Path | None = None,
            id_pepper: str | None = None, id_pepper_file: str | Path | None = None, sheet: str | None = None, encoding: str = "utf-8", sep: str = ",",
            mapping_path: str | Path = DEFAULT_MAPPING, dictionary_path: str | Path = DEFAULT_DICTIONARY, template_path: str | Path = DEFAULT_TEMPLATE,
            index_day_records: str | None = None, min_cell: int = 10, make_plots: bool = True) -> EDAResult:
    """Run the whole EDA. Raises ``DatasetValidationError`` / ``ConfigError`` / ``LeakageError`` when the inputs are inconsistent."""
    from falls_ml.meuhedet_explore import watermark
    from falls_ml.meuhedet_sensitivity import directory_digest

    t0 = time.perf_counter()
    src = Path(input_path)
    if not src.is_file():
        raise DatasetValidationError(f"Input file not found: {src}")
    if min_cell < 1:
        raise ConfigError("--min-cell must be >= 1")
    ref_dir = Path(reference_dir) if reference_dir else None
    out = Path(out_dir) if out_dir is not None else src.with_name(f"{src.stem}_eda")
    if ref_dir is not None and (out.resolve() == ref_dir.resolve() or ref_dir.resolve() in out.resolve().parents):
        raise DatasetValidationError(f"--out {out} must not be the reference directory or inside it (the reference is read-only)")
    if out.exists() and any(out.iterdir()):
        raise DatasetValidationError(f"Output directory {out} is not empty; results are immutable - choose a new --out")
    if index_day_records not in (None, "fail", "drop_rows"):
        raise ConfigError("--index-day-records must be fail or drop_rows")
    mapping = load_wide_mapping(mapping_path)
    contract = load_wide_contract(mapping.contract_path)
    spec = load_feature_spec(mapping.exploratory_spec_path)
    dictionary = load_data_dictionary(dictionary_path, contract)
    ref = _load_reference(ref_dir) if ref_dir is not None else None
    ref_digest = directory_digest(ref_dir) if ref_dir is not None else None
    wm = watermark(mapping)
    read = read_wide_extract_report(src, contract, encoding=encoding, sep=sep, sheet=sheet)
    input_sha = sha256_file(src)
    if ref is not None and ref["input_sha256"] and ref["input_sha256"] != input_sha:
        raise DatasetValidationError("--input is not the file the reference run used (sha256 differs); the EDA split would not be the modelling split",
                                     [f"reference {ref['input_sha256'][:16]}…", f"input {input_sha[:16]}…"])
    if ref is not None and index_day_records and ref["policy"] and index_day_records != ref["policy"]:
        raise ConfigError(f"--index-day-records {index_day_records} differs from the reference run's policy {ref['policy']}")
    eda = out / "eda"
    plots_dir = eda / "plots"
    eda.mkdir(parents=True, exist_ok=True)
    res = EDAResult(out_dir=out, eda_dir=eda, input_file=src.name, watermark=wm)
    res.timings["read"] = round(time.perf_counter() - t0, 1)
    frame = read.frame
    res.synthetic = bool("Snapshot_Key" in frame.columns and len(frame) and frame["Snapshot_Key"].astype("string").str.startswith("SYN_").fillna(False).all())
    res.stage("RAW EXTRACT", "DONE", f"{read.source_format}, {len(frame)} rows x {frame.shape[1]} columns; sha256 {input_sha[:12]}…; "
                                      f"{sum(1 for v in read.wrong_type.values() if v)} columns with unparseable values")
    contract_report = validate_wide_contract(frame, contract, wrong_type=read.wrong_type, wrong_type_examples=read.wrong_type_examples,
                                             date_formats=read.date_formats, strict=False)
    res.stage("CONTRACT CHECK", "DONE", f"{len(contract_report.problems)} contract problems, {len(contract_report.warnings)} warnings, "
                                         f"{sum(contract_report.invalid_values.values())} out-of-contract values")
    res.facts["contract"] = {"problems": list(contract_report.problems), "warnings": list(contract_report.warnings),
                             "missing_columns": [c for c in contract.names if c not in frame.columns],
                             "extra_columns": [c for c in frame.columns if c not in set(contract.names)]}
    idx_col, id_col = mapping.identity["index_date"], mapping.identity["research_id"]
    label_col = mapping.outcome["label_column"]
    index = frame[idx_col].dt.normalize()
    eligible_any = (frame["Is_Eligible_Cohort"] == 1).fillna(False).astype(bool) if "Is_Eligible_Cohort" in frame.columns else pd.Series(True, index=frame.index)

    # ---- index date of the cohort
    chosen: str | None = None
    dates = index.dropna().unique()
    if index_date:
        chosen = str(pd.Timestamp(index_date).date())
    elif ref is not None and ref["index_date"]:
        chosen = str(pd.Timestamp(ref["index_date"]).date())
    elif len(dates) == 1:
        chosen = str(pd.Timestamp(dates[0]).date())
    if ref is not None and ref["index_date"] and chosen != str(pd.Timestamp(ref["index_date"]).date()):
        raise ConfigError(f"--index-date {chosen} differs from the reference run's index date {ref['index_date']}")
    res.index_date = chosen or ""
    on_date = (index == pd.Timestamp(chosen)).fillna(False) if chosen else pd.Series(False, index=frame.index)
    eligible = on_date & eligible_any
    labelled = eligible & frame[label_col].notna()
    guard = [c for c in mapping.cohort["predictor_max_record_date"]["columns"] if c in frame.columns]
    evidence_cols = sorted({f.same_day_evidence_column for f in mapping.features if f.include_in_baseline and f.same_day_evidence_column} & set(frame.columns))

    def on_after(cols: list[str]) -> pd.Series:
        m = pd.Series(False, index=frame.index)
        for c in cols:
            m |= (frame[c].dt.normalize() >= index).fillna(False)
        return m

    d00 = labelled & on_after(guard)
    evidence = labelled & on_after(evidence_cols)

    # ---- (A) timing audit + dictionary (definitions + D-00 evidence)
    timing = timing_diagnostic(frame, contract, mapping, min_cell=min_cell)
    from falls_ml.meuhedet_sensitivity import feature_provenance

    provenance = feature_provenance(mapping, contract, timing)
    safe_features = list(provenance["safe_all_rows_feature_set"])
    on_after_counts = {}
    for c in contract.columns:
        if c.is_date and c.name in frame.columns:
            sent = frame[c.name].isin([pd.Timestamp(v) for v in c.sentinels]) if c.sentinels else pd.Series(False, index=frame.index)
            on_after_counts[c.name] = int(((frame[c.name].dt.normalize() >= index) & eligible_any & ~sent).fillna(False).sum())
    dict_table = dictionary_table(contract, dictionary, mapping, on_after_counts)
    availability = dict(zip(dict_table["column"], dict_table["available_at_prediction_time"]))
    temporal = temp_mod.temporal_audit(frame, contract, dictionary, mapping, index=index, eligible=eligible_any, labelled=labelled,
                                       safe_features=safe_features, min_cell=min_cell)
    prov_table = temp_mod.provenance_table(contract, dictionary, mapping, temporal, provenance, safe_features)
    n_d00 = int(d00.sum())
    res.stage("LEAKAGE / TIMING AUDIT", "DONE", f"{len(temporal)} date columns / sources audited; D-00 guard violations among eligible labelled rows: {n_d00} "
                                                f"(root cause {timing['verdict']['root_cause']})")
    patterns, failure_masks = prof_mod.failure_patterns(src, read.source_format, frame, read.wrong_type, encoding=encoding, sep=sep, min_cell=min_cell)
    res.timings["audit"] = round(time.perf_counter() - t0, 1)

    # ---- cohort build + split (canonical dataset in a temporary directory, deleted afterwards)
    policy = index_day_records or (ref["policy"] if ref is not None else None) or mapping.cohort.get("index_day_records", "fail")
    res.policy = policy
    partition = pd.Series(pd.NA, index=frame.index, dtype="string")
    canonical = None
    build = safe_build = None
    plan = cfg = ds = None
    research_ids: set[str] = set()
    pepper_value = ""
    split_facts: dict[str, Any] = {}
    sets = mapping.feature_sets()
    if ref is not None and list(ref["features"]) != list(sets["extended"]):
        raise ConfigError("the reference EXTENDED feature set differs from the current mapping's", [f"reference {ref['features']}", f"mapping {sets['extended']}"])
    if not chosen:
        res.stage("COHORT BUILD", "NOT RUN", f"the file holds {len(dates)} index dates; choose one with --index-date (the unsupervised EDA above covers every row)")
    else:
        # ignore_cleanup_errors: on Windows a file still held open cannot be deleted at exit; the leftover is removed again below and reported
        with tempfile.TemporaryDirectory(prefix="falls_ml_eda_", ignore_cleanup_errors=True) as tmp:
            tmpd = Path(tmp)
            temp_dir_used = tmpd
            try:
                _cohort_and_split(res, frame, src, tmpd, mapping=mapping, contract=contract, spec=spec, ref=ref, read=read, input_sha=input_sha, chosen=chosen,
                                  policy=policy, id_pepper=id_pepper, id_pepper_file=id_pepper_file, template_path=template_path, sets=sets,
                                  safe_features=safe_features, partition=partition, split_facts=split_facts)
                canonical = split_facts.pop("_canonical", None)
                build, safe_build = split_facts.pop("_build", None), split_facts.pop("_safe_build", None)
                plan, cfg, ds = split_facts.pop("_plan", None), split_facts.pop("_cfg", None), split_facts.pop("_ds", None)
                research_ids = split_facts.pop("_research_ids", set())
                pepper_value = split_facts.pop("_pepper", "")
            except _Stop as stop:
                for k in [k for k in split_facts if k.startswith("_")]:
                    if k == "_pepper":
                        pepper_value = split_facts[k]
                    split_facts.pop(k)
                if "COHORT BUILD" not in res.stages:
                    res.stage("COHORT BUILD", "NOT RUN", str(stop))
                elif "SPLIT" not in res.stages:
                    res.stage("SPLIT", "NOT RUN", str(stop))
    if chosen:
        import gc
        import shutil

        gc.collect()
        if temp_dir_used.exists():
            shutil.rmtree(temp_dir_used, ignore_errors=True)
        split_facts["temporary_dataset_deleted"] = not temp_dir_used.exists()
        if temp_dir_used.exists():
            log.warning("eda_temp_dataset_not_deleted", extra_fields={"note": "delete the falls_ml_eda_* folder in the local temp directory"})
    cohort = partition.notna()
    safe_mask = labelled & ~evidence
    masks = cohort_mod.CohortMasks(on_date=on_date, eligible=eligible, labelled=labelled, d00=d00, evidence=evidence, cohort=cohort, safe=safe_mask,
                                   partition=partition)
    res.facts["split"] = split_facts
    res.timings["cohort_split"] = round(time.perf_counter() - t0, 1)

    # ---- (A) unsupervised profiles on the whole file
    bases = {FULL_EXTRACT: pd.Series(True, index=frame.index), ELIGIBLE_LABELLED: labelled}
    if cohort.any():
        bases[MODELLING_COHORT] = cohort
    col_prof = prof_mod.column_profile(frame, contract, dictionary, index=index, wrong_type=read.wrong_type, patterns=patterns,
                                       invalid_values=dict(contract_report.invalid_values), bases=bases, min_cell=min_cell, dict_table=dict_table)
    num_cols = prof_mod.numeric_columns(contract, frame)
    cat_cols = prof_mod.categorical_columns(contract, frame)
    prof_bases = {FULL_EXTRACT: bases[FULL_EXTRACT], **({MODELLING_COHORT: cohort} if cohort.any() else {})}
    num_prof = prof_mod.numeric_profile(frame, contract, dictionary, num_cols, prof_bases, min_cell)
    cat_prof = prof_mod.categorical_profile(frame, contract, dictionary, cat_cols, prof_bases, min_cell)
    miss = miss_mod.missingness_table(frame, contract, dictionary, bases, failure_masks, read.wrong_type, min_cell)
    miss_dom = miss_mod.by_domain(miss, frame)
    pred_cols = miss_mod.predictor_columns_for_missingness(contract, frame)
    focus = cohort if cohort.any() else (labelled if labelled.any() else bases[FULL_EXTRACT])
    focus_name = MODELLING_COHORT if cohort.any() else (ELIGIBLE_LABELLED if labelled.any() else FULL_EXTRACT)
    comat, _order = miss_mod.co_missingness(frame.loc[focus], pred_cols)
    co_pairs = miss_mod.co_missing_pairs(frame.loc[focus], comat, dictionary, min_cell) if len(comat) else pd.DataFrame()
    patterns_df, pattern_sources = miss_mod.source_block_patterns(frame, contract, dictionary, focus, min_cell)
    row_miss = miss_mod.row_missingness(frame, pred_cols, focus)
    res.stage("FULL EDA", "DONE", f"{len(col_prof)} columns profiled on {len(frame)} rows; {len(num_cols)} numeric and {len(cat_cols)} categorical columns; "
                                  f"missingness classified")
    res.timings["profiles"] = round(time.perf_counter() - t0, 1)

    # ---- (B) target-aware analyses on TRAIN only
    tp = None
    uni = num_out = cat_out = miss_out = corr_pairs = clusters = spear = t1_train = None
    cand_cols = [c.name for c in contract.columns if c.name in frame.columns and not c.is_date and not c.is_text
                 and (c.role in PREDICTOR_ROLES or c.role == "FORBIDDEN_LEAKAGE")]
    if cohort.any() and (partition == "train").any():
        tp = sup_mod.TrainPartition.from_assignment(frame, partition, label_col)
        num_sup = [c for c in num_cols if c in cand_cols]
        cat_sup = [c for c in cat_cols if c in cand_cols]
        num_out = sup_mod.numeric_outcome(tp, num_sup, contract, dictionary, min_cell)
        cat_out = sup_mod.categorical_outcome(tp, cat_sup, contract, dictionary, min_cell)
        miss_out = miss_mod.missingness_outcome(tp, pred_cols, contract, dictionary, min_cell)
        corr_cols = [c for c in cand_cols if contract.get(c).role in PREDICTOR_ROLES]
        spear = sup_mod.correlation_matrix(tp, corr_cols, "spearman")
        pear = sup_mod.correlation_matrix(tp, corr_cols, "pearson")
        ext_sources = {c for f in sets["extended"] for c in mapping.get(f).source_columns}
        nominal = [c for c in corr_cols if contract.get(c).semantic == "categorical"]
        partners = [c for c in corr_cols if contract.get(c).semantic in ("binary", "categorical", "ordinal")]
        corr_pairs = sup_mod.correlation_pairs(tp, spear, pear, contract, dictionary, ext_sources, nominal=nominal, nominal_partners=partners)
        clusters = sup_mod.redundancy_clusters(spear, dictionary, ext_sources)
        uni = sup_mod.univariate_association(tp, cand_cols, contract, dictionary, min_cell, availability)
        t1_train = sup_mod.table1(tp.frame, tp.y, basis=TRAIN_ONLY, min_cell=min_cell)
        res.stage("TRAIN-ONLY SUPERVISED EDA", "DONE", f"{tp.n} training rows, {tp.n_events} events; {len(cand_cols)} candidate columns; "
                                                       "validation and test rows never read")
    else:
        res.stage("TRAIN-ONLY SUPERVISED EDA", "NOT RUN", "no split (see COHORT BUILD / SPLIT); no target-aware analysis was produced")
    t1 = sup_mod.table1(frame.loc[cohort], frame.loc[cohort, label_col].astype("int64").to_numpy(), basis=POSTHOC_DESCRIPTIVE, min_cell=min_cell) if cohort.any() else None
    res.timings["supervised"] = round(time.perf_counter() - t0, 1)

    # ---- cohort / outcome / quality / features / split tables
    flow = cohort_mod.cohort_flow(frame, masks, id_col=id_col, label_col=label_col, index_date=chosen or "?", policy=policy, guard_cols=guard,
                                  evidence_cols=evidence_cols, build=build, safe_build=safe_build, min_cell=min_cell) if chosen else pd.DataFrame()
    pop = cohort_mod.population_profile(frame, masks, label_col=label_col, min_cell=min_cell)
    prev = cohort_mod.outcome_prevalence(frame, masks, label_col=label_col, min_cell=min_cell) if labelled.any() else pd.DataFrame()
    dq = dq_mod.data_quality(frame, contract, mapping, index=index, wrong_type=read.wrong_type, patterns=patterns, contract_report=contract_report,
                             min_cell=min_cell, eligible=eligible_any, corr_pairs=corr_pairs,
                             window_days=int(mapping.outcome["window_days_including_index"]) - 1)
    if flow is not None and len(flow):
        for chk in flow.attrs.get("build_crosscheck", []):
            if not chk["agree"]:
                dq = pd.concat([dq, pd.DataFrame([{"check_id": "H01", "category": "build", "severity": "CRITICAL", "status": "FINDING",
                                                    "title": f"EDA funnel disagrees with the canonical build ({chk['what']})", "columns": "",
                                                    "n_rows": abs(chk["eda_rows"] - chk["build_rows"]), "pct_rows": None, "basis": FULL_EXTRACT,
                                                    "detail": f"EDA {chk['eda_rows']} vs build {chk['build_rows']}", "recommendation": "report to the maintainers"}])],
                               ignore_index=True)
    cohort_raw = frame.loc[cohort] if cohort.any() else frame.loc[labelled]
    cur = feat_mod.current_feature_dictionary(mapping, contract, dictionary, cohort_raw=cohort_raw, canonical=canonical, build=build, timing=timing,
                                              provenance=provenance, reference_report=(ref or {}).get("feature_report_extended"),
                                              reference_strict_report=(ref or {}).get("feature_report_strict"))
    p2 = feat_mod.phase2_catalogue(mapping, contract, dictionary, cohort_raw=cohort_raw, temporal=temporal, univariate=uni, spearman=spear,
                                   clusters=clusters, availability=availability, missing_classes=miss)
    bal = None
    if cohort.any():
        sources = {f: list(mapping.get(f).source_columns) for f in sets["extended"]}
        bal = split_mod.split_balance(frame.loc[cohort], partition[cohort], label_col=label_col, canonical_names=sets["extended"], sources=sources,
                                      min_cell=min_cell)
    if plan is not None and cfg is not None and ds is not None:
        from falls_ml.adequacy import assess_adequacy

        adq = assess_adequacy(ds.frame, ds.spec, plan, cfg)
        res.facts["adequacy"] = adq.to_dict()
        res.stage("ADEQUACY", "DONE", f"EXTENDED: {adq.verdict}" + (f" ({'; '.join(adq.reasons)})" if adq.reasons else ""))
    else:
        res.stage("ADEQUACY", "NOT RUN", "needs the split")
    for s, text in NEXT_STAGES.items():
        res.stage(s, "NEXT", text)

    tables = {"data_dictionary": dict_table, "column_profile": col_prof, "numeric_profile": num_prof, "categorical_profile": cat_prof,
              "missingness": miss, "missingness_by_domain": miss_dom, "co_missingness_pairs": co_pairs, "missingness_patterns": patterns_df,
              "data_quality_checks": dq, "data_quality_findings": dq[dq["status"] == "FINDING"].reset_index(drop=True), "cohort_flow": flow,
              "population_profile": pop, "outcome_prevalence": prev, "temporal_audit": temporal, "feature_provenance": prov_table,
              "current_15_feature_dictionary": cur, "phase2_candidate_features": p2}
    optional = {"missingness_outcome_train": miss_out, "numeric_outcome_train": num_out, "categorical_outcome_train": cat_out,
                "correlation_pairs": corr_pairs, "redundancy_clusters": clusters, "univariate_association_train": uni, "table1": t1,
                "table1_train": t1_train, "split_balance": bal}
    tables.update({k: v for k, v in optional.items() if v is not None})
    res.tables = tables
    res.facts.update(_facts(res, frame, masks, timing, provenance, read, contract_report, mapping, row_miss, focus_name, ref, input_sha, dictionary,
                            dates, id_col, label_col, pattern_sources))
    res.timings["tables"] = round(time.perf_counter() - t0, 1)

    # ---- write CSV / JSON
    for name, df in tables.items():
        basis = FILE_BASIS.get(f"{name}.csv", (FULL_EXTRACT, ""))[0]
        write_csv(df if df is not None else pd.DataFrame(), eda / f"{name}.csv", watermark=wm, basis=basis)
    write_json({"watermark": wm, **res.facts["timing_summary"], "diagnostic": timing}, eda / "timing_diagnostic.json")
    write_json({"watermark": wm, **provenance}, eda / "feature_provenance.json")

    # ---- plots
    if make_plots:
        _plots(res, frame, masks, contract, dictionary, mapping, tables, num_cols, cat_cols, cand_cols, tp, comat, patterns_df, pattern_sources, row_miss,
               plots_dir, min_cell)
    res.timings["plots"] = round(time.perf_counter() - t0, 1)

    # ---- reports + manifest + privacy scan
    from falls_ml.eda.report import render_dq_report, render_html, render_summary

    integrity = None
    if ref_dir is not None and ref_digest is not None:
        after = directory_digest(ref_dir)
        integrity = {"reference_dir": ref_dir.name, "files": after["n_files"], "unchanged": after["combined_sha256"] == ref_digest["combined_sha256"]}
        if not integrity["unchanged"]:
            raise DatasetValidationError("the reference directory changed while the EDA ran (it is only read); investigate", [])
    ids = _identifier_values(frame, contract, research_ids)
    pending = {"passed": None, "files_scanned": None, "identifier_values_checked": len(ids), "pepper_checked": bool(pepper_value),
               "note": "the scan runs after every file is written; its result is recorded in eda_manifest.json"}
    res.manifest = _manifest(res, mapping, contract, dictionary, spec, input_sha, read, ref, integrity, pending, min_cell)
    (out / "REAL_DATA_EDA_SUMMARY.md").write_text(render_summary(res), encoding="utf-8", newline="\n")
    (out / "DATA_QUALITY_REPORT.md").write_text(render_dq_report(res), encoding="utf-8", newline="\n")
    (out / "REAL_DATA_EDA_REPORT.html").write_text(render_html(res), encoding="utf-8", newline="\n")
    scan = privacy_scan(out, ids, pepper_value)
    res.timings["total"] = round(time.perf_counter() - t0, 1)
    res.manifest = _manifest(res, mapping, contract, dictionary, spec, input_sha, read, ref, integrity, scan, min_cell)
    write_json(res.manifest, out / "eda_manifest.json")
    scan2 = privacy_scan(out, ids, pepper_value)
    if not scan2["passed"]:
        raise LeakageError("identifier values found in the EDA outputs; the results must not be shared", scan2["hits"])
    res.manifest["privacy"]["identifier_scan"] = {k: v for k, v in scan2.items() if k != "hits"}
    write_json(res.manifest, out / "eda_manifest.json")
    log.info("meuhedet_eda_done", extra_fields={"out": str(out), "rows": len(frame), "seconds": res.timings["total"]})
    return res


def _cohort_and_split(res: EDAResult, frame: pd.DataFrame, src: Path, tmpd: Path, *, mapping: Any, contract: Any, spec: Any, ref: dict[str, Any] | None,
                      read: Any, input_sha: str, chosen: str, policy: str, id_pepper: str | None, id_pepper_file: str | Path | None,
                      template_path: str | Path, sets: dict[str, list[str]], safe_features: list[str], partition: pd.Series,
                      split_facts: dict[str, Any]) -> None:
    from falls_ml.config import load_experiment_config
    from falls_ml.meuhedet_explore import RUN_LABELS, _write_config, resolve_id_pepper
    from falls_ml.splitting import make_split

    pepper, pinfo = resolve_id_pepper(src, tmpd, explicit=id_pepper, pepper_file=id_pepper_file)
    split_facts["_pepper"] = pepper
    split_facts["pepper_sha256"], split_facts["pepper_source"] = pinfo["sha256"], pinfo["source"]
    if ref is not None and ref["pepper_sha256"] and ref["pepper_sha256"] != pinfo["sha256"]:
        raise DatasetValidationError("the pseudonymisation pepper differs from the reference run's: the research ids and the split would differ",
                                     [f"pepper source: {pinfo['source']}", "pass --id-pepper-file <reference folder>\\id_pepper.txt"])
    ds_dir = tmpd / "dataset_extended"
    try:
        _manifest, build = build_meuhedet_dataset_from_frame(
            frame, ds_dir, index_dates=[chosen], dataset_version=f"eda-{chosen}", data_freeze_date=None, mapping=mapping, contract=contract, spec=spec,
            input_name=src.name, input_sha256=input_sha, wrong_type=read.wrong_type, wrong_type_examples=read.wrong_type_examples,
            date_formats=read.date_formats, features=sets["extended"], feature_set_label="extended", id_pepper=pepper, index_day_records=policy,
            read_report=read.to_dict(), notes="EDA split reconstruction (temporary)")
    except DatasetValidationError as exc:
        d00 = any("D-00 timing" in p for p in exc.problems)
        detail = ("D-00: rows with a predictor record date on/after the index day and policy 'fail'. Rerun with --reference <completed explore run> "
                  "(its policy is reused) or --index-day-records drop_rows (rows excluded and counted, as in the reference run)") if d00 else f"{exc}: {exc.problems[:3]}"
        res.stage("COHORT BUILD", "STOPPED", detail)
        raise _Stop(detail) from None
    split_facts["_build"] = build
    res.stage("COHORT BUILD", "DONE", f"policy {policy}: {build.n_rows_final} rows, {build.n_patients_final} patients, {build.n_events} events; "
                                      f"D-00 excluded {build.n_rows_timing_violation}")
    try:
        adapter = MeuhedetWideDatasetAdapter(mapping, contract, spec, features=safe_features, id_pepper=None, timing_scope="built_predictors")
        _f, safe_build = adapter.apply(frame, index_date=chosen, index_day_records="drop_rows", wrong_type=read.wrong_type,
                                       wrong_type_examples=read.wrong_type_examples, date_formats=read.date_formats)
        split_facts["_safe_build"] = safe_build
    except (DatasetValidationError, ConfigError) as exc:
        split_facts["safe_all_rows_error"] = str(exc)
    if ref is not None:
        cfg_path = ref["config"]
    else:
        cfg_path = _write_config(template_path, tmpd / "config.yaml", name=RUN_LABELS["extended"], description="EDA split reconstruction",
                                 features=sets["extended"], fast=True, spec=spec)
    cfg = load_experiment_config(cfg_path)
    ds = ModelingDataset.load(ds_dir, spec, features=sets["extended"])
    try:
        plan = make_split(ds.frame, ds.spec, cfg.validation)
    except (ConfigError, DatasetValidationError) as exc:
        res.stage("SPLIT", "STOPPED", str(exc))
        raise _Stop(str(exc)) from None
    test_sha = plan.test_rows_sha256(ds.frame, ds.spec)
    summary = plan.summary(ds.frame, ds.spec)
    split_facts.update({"strategy": plan.strategy, "seed": cfg.validation.seed, "test_rows_sha256": test_sha, "partitions": summary,
                        "config_source": "reference run config" if ref is not None else f"template {template_path}",
                        "reference_test_rows_sha256": ref["split_audit"].get("test_rows_sha256") if ref is not None else None})
    if ref is not None:
        if ref["split_audit"].get("test_rows_sha256") != test_sha:
            raise LeakageError("the EDA split does not reproduce the reference run's test partition (hash differs): target-aware EDA would not be "
                               "restricted to the modelling TRAIN rows", [f"reference {ref['split_audit'].get('test_rows_sha256')}", f"eda {test_sha}"])
        split_facts["matches_reference"] = True
    assign = plan.assignments(ds.frame, ds.spec)
    key = assign["research_id"].astype(str) + "|" + pd.to_datetime(assign["index_date"]).dt.strftime("%Y-%m-%d")
    lookup = dict(zip(key, assign["split"].astype(str)))
    ids = frame[mapping.identity["research_id"]].astype("string")
    pseudo = ids.map(lambda v: hashlib.sha256(f"{pepper}|{v}".encode("utf-8")).hexdigest()[:20] if pd.notna(v) else None)
    raw_key = pseudo.astype("string") + "|" + frame[mapping.identity["index_date"]].dt.strftime("%Y-%m-%d")
    on = (frame[mapping.identity["index_date"]].dt.normalize() == pd.Timestamp(chosen)).fillna(False) & (frame["Is_Eligible_Cohort"] == 1).fillna(False)
    mapped = raw_key.where(on).map(lookup)
    partition.loc[:] = mapped.astype("string")
    if int(partition.notna().sum()) != len(ds.frame):
        raise DatasetValidationError("could not map every modelling row back to exactly one extract row", [f"mapped {int(partition.notna().sum())}", f"cohort {len(ds.frame)}"])
    split_facts.update({"_canonical": ds.frame.copy(), "_plan": plan, "_cfg": cfg, "_ds": ds, "_research_ids": set(ds.frame[ds.spec.identifier_columns[0]].astype(str))})
    res.stage("SPLIT", "DONE", f"{plan.strategy}, seed {cfg.validation.seed}: " + ", ".join(f"{k} {v['n_rows']} rows / {v['n_events']} events" for k, v in summary.items())
              + f"; test sha256 {test_sha[:12]}…" + (" = reference run" if ref is not None else ""))


def _facts(res: EDAResult, frame: pd.DataFrame, masks: Any, timing: dict[str, Any], provenance: dict[str, Any], read: Any, contract_report: Any, mapping: Any,
           row_miss: dict[str, Any], focus_name: str, ref: dict[str, Any] | None, input_sha: str, dictionary: Any, dates: np.ndarray, id_col: str,
           label_col: str, pattern_sources: list[str]) -> dict[str, Any]:
    y = pd.to_numeric(frame[label_col], errors="coerce")
    cohort = masks.cohort
    dq = res.tables["data_quality_checks"]
    fnd = dq[dq["status"] == "FINDING"]
    return {
        "n_rows": int(len(frame)), "n_columns_in_file": int(frame.shape[1]), "n_patients": int(frame[id_col].nunique()), "n_snapshots": int(len(dates)),
        "index_dates": sorted(str(pd.Timestamp(d).date()) for d in dates)[:24], "n_eligible_on_index_date": int(masks.eligible.sum()),
        "n_labelled": int(masks.labelled.sum()), "n_unlabelled_eligible": int((masks.eligible & ~masks.labelled).sum()),
        "n_d00_excluded": int(masks.d00.sum()) if res.policy == "drop_rows" else 0, "n_d00_violating": int(masks.d00.sum()),
        "n_evidence_rows": int(masks.evidence.sum()), "n_safe_rows": int(masks.safe.sum()),
        "n_cohort": int(cohort.sum()), "n_cohort_patients": int(frame.loc[cohort, id_col].nunique()) if cohort.any() else 0,
        "n_cohort_events": int((y[cohort] == 1).sum()) if cohort.any() else 0,
        "cohort_prevalence": float((y[cohort] == 1).mean()) if cohort.any() else None,
        "labelled_prevalence": float((y[masks.labelled] == 1).mean()) if masks.labelled.any() else None,
        "d00_prevalence": float((y[masks.d00] == 1).mean()) if masks.d00.any() else None,
        "clean_prevalence": float((y[masks.labelled & ~masks.d00] == 1).mean()) if (masks.labelled & ~masks.d00).any() else None,
        "n_findings": {s: int((fnd["severity"] == s).sum()) for s in dq_mod.SEVERITIES}, "n_checks": int(len(dq)),
        "timing_summary": {"root_cause": timing["verdict"]["root_cause"], "n_eligible_rows_violating_d00": timing["verdict"]["n_eligible_rows_violating_d00"],
                           "affected_efalls_features_by_feature_set": timing["verdict"]["affected_efalls_features_by_feature_set"]},
        "safe_all_rows_feature_set": list(provenance["safe_all_rows_feature_set"]), "excluded_from_safe_all_rows": list(provenance["excluded_from_safe_all_rows"]),
        "row_missingness": {k: v for k, v in row_miss.items() if k != "share_values"}, "row_missingness_basis": focus_name,
        "meaning_status": dict(pd.Series([v["status"] for v in dictionary.columns.values()]).value_counts()),
        "unknown_meaning_columns": [c for c, v in dictionary.columns.items() if v["status"] == "UNKNOWN"],
        "wrong_type_columns": {c: int(v) for c, v in read.wrong_type.items() if v}, "pattern_sources": pattern_sources,
        "reference": None if ref is None else {"dir": ref["dir"].name, "index_date": ref["index_date"], "policy": ref["policy"]},
        "input_sha256": input_sha,
    }


def _plots(res: EDAResult, frame: pd.DataFrame, masks: Any, contract: Any, dictionary: Any, mapping: Any, tables: dict[str, pd.DataFrame],
           num_cols: list[str], cat_cols: list[str], cand_cols: list[str], tp: Any, comat: pd.DataFrame, patterns_df: pd.DataFrame,
           pattern_sources: list[str], row_miss: dict[str, Any], plots_dir: Path, min_cell: int) -> None:
    wm = res.watermark
    out: dict[str, list[str]] = {}

    def add(group: str, p: Path | None) -> None:
        if p is not None:
            out.setdefault(group, []).append(p.relative_to(res.out_dir).as_posix())

    if len(tables["cohort_flow"]):
        add("cohort", P.cohort_funnel(plots_dir / "cohort_funnel.png", tables["cohort_flow"], watermark=wm))
    pop = tables["population_profile"]
    add("cohort", P.grouped_distribution(plots_dir / "population_age.png", pop, "age", "age band", "Age bands by population basis", watermark=wm))
    add("cohort", P.grouped_distribution(plots_dir / "population_sex.png", pop, "sex", "Gender_Code", "Sex (Gender_Code) by population basis", watermark=wm))
    add("cohort", P.grouped_distribution(plots_dir / "population_seniority.png", pop, "HMO seniority", "seniority band", "HMO seniority by population basis", watermark=wm))
    prev = tables["outcome_prevalence"]
    if len(prev):
        overall = res.facts.get("cohort_prevalence")
        post = prev[prev["basis"] == POSTHOC_DESCRIPTIVE]
        add("outcome", P.prevalence_forest(plots_dir / "outcome_prevalence_by_stratum.png", post, overall, watermark=wm,
                                           title="POST-HOC DESCRIPTIVE (full modelling cohort incl. test rows; not for model choice): 180-day prevalence by stratum",
                                           basis=POSTHOC_DESCRIPTIVE))
        head = prev[prev["basis"] != POSTHOC_DESCRIPTIVE]
        add("outcome", P.prevalence_forest(plots_dir / "outcome_prevalence_overall.png", head, overall, watermark=wm,
                                           title="180-day prevalence: overall, D-00 status, SAFE-ALL-ROWS, partitions", basis=f"{ELIGIBLE_LABELLED} / {MODELLING_COHORT}"))
    add("missingness", P.missing_by_column(plots_dir / "missingness_by_column.png", tables["missingness"], miss_mod.CLASSES, watermark=wm))
    md = tables["missingness_by_domain"]
    if len(md):
        md = md.sort_values("mean_missing_pct", ascending=False)
        add("missingness", P.bars(plots_dir / "missingness_by_domain.png", list(md["domain"]), [float(v or 0) for v in md["mean_missing_pct"]],
                                  title="Mean % missing per domain (full extract)", xlabel="mean % missing across the domain's columns", watermark=wm, basis=FULL_EXTRACT))
    add("missingness", P.heatmap(plots_dir / "co_missingness_heatmap.png", comat, title="Co-missingness: phi correlation of missing indicators (clustered)",
                                 watermark=wm, basis=res.facts.get("row_missingness_basis", ""), label="phi"))
    if len(patterns_df):
        add("missingness", P.pattern_matrix(plots_dir / "missingness_patterns.png", patterns_df, pattern_sources, watermark=wm))
    add("missingness", P.histogram_share(plots_dir / "row_missingness.png", row_miss["share_values"], title="Share of predictor columns missing per row",
                                         xlabel="% of predictor columns missing on the row", watermark=wm, basis=res.facts.get("row_missingness_basis", ""), min_cell=min_cell))
    tr = tp.frame if tp is not None else None
    for col in num_cols:
        full = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype=float)
        coh = pd.to_numeric(frame.loc[masks.cohort, col], errors="coerce").dropna().to_numpy(dtype=float) if masks.cohort.any() else np.array([])
        f1 = f0 = None
        if tr is not None and col in cand_cols:
            v = pd.to_numeric(tr[col], errors="coerce").to_numpy(dtype=float)
            f1, f0 = v[(tp.y == 1) & np.isfinite(v)], v[(tp.y == 0) & np.isfinite(v)]
        if full.size < min_cell or contract.get(col).role not in (*PREDICTOR_ROLES, "FORBIDDEN_LEAKAGE"):
            continue   # per-column panels for candidate / forbidden columns; QA / cohort / label columns are in the tables
        add(f"numeric::{dictionary.domain_label(col)}", P.numeric_panel(plots_dir / "numeric" / f"{P.safe_name(col)}.png", col, full, coh, f1, f0, watermark=wm,
                                                                          min_cell=min_cell, subtitle=f"{dictionary.domain_label(col)}; {contract.get(col).role}"))
    cp = tables["categorical_profile"]
    co = tables.get("categorical_outcome_train")
    for col in cat_cols:
        c = contract.get(col)
        if c.role not in PREDICTOR_ROLES or c.semantic == "binary":
            continue
        basis = MODELLING_COHORT if masks.cohort.any() else FULL_EXTRACT
        lv = cp[(cp["column"] == col) & (cp["basis"] == basis) & cp["n"].notna()]
        if lv.empty:
            continue
        train = co[co["column"] == col] if co is not None and len(co) else None
        add(f"categorical::{dictionary.domain_label(col)}", P.categorical_panel(plots_dir / "categorical" / f"{P.safe_name(col)}.png", col, lv, train, watermark=wm))
    uni = tables.get("univariate_association_train")
    if uni is not None and len(uni):
        u = uni[uni["available_at_prediction_time"].isin(["YES_RECORD_DATE_CLEAN", "YES_BUT_INDEX_DAY_RECORDS_PROVEN", "YES_PARTIAL_EVIDENCE",
                                                         "YES_DECLARED_UNVERIFIABLE"])].copy()
        u["abs"] = pd.to_numeric(u["smd_fall_vs_no_fall"], errors="coerce").abs()
        u = u.dropna(subset=["abs"]).sort_values("abs", ascending=False).head(40)
        u["label"] = u["column"] + " [" + u["domain"] + "]"
        add("association", P.smd_dots(plots_dir / "univariate_association_top40.png", u, value_cols=[("smd_fall_vs_no_fall", "SMD")], label_col="label",
                                      title=f"TRAIN ONLY - {UNIVARIATE_BANNER} (top 40 by |SMD|; columns available at prediction time)",
                                      watermark=wm, basis=TRAIN_ONLY))
        b = uni[(uni["kind"] == "binary") & uni["available_at_prediction_time"].str.startswith("YES")].copy()
        if "prevalence_ratio_1_vs_0" in b.columns and len(b):
            b["label"] = b["column"] + " [" + b["domain"] + "]"
            b = b.sort_values(["domain", "column"])
            add("association", P.forest_ratio(plots_dir / "binary_prevalence_ratio_train.png", b, label_col="label",
                                              title=f"TRAIN ONLY - binary flags: prevalence ratio of the outcome, flag 1 vs 0 ({UNIVARIATE_BANNER})",
                                              watermark=wm, basis=TRAIN_ONLY))
    cps = tables.get("correlation_pairs")
    if tp is not None:
        corr_cols = [c for c in cand_cols if contract.get(c).role in PREDICTOR_ROLES]
        spear = sup_mod.correlation_matrix(tp, corr_cols, "spearman")
        if len(spear) >= 2:
            order = list(spear.columns)
            try:
                from scipy.cluster.hierarchy import leaves_list, linkage
                from scipy.spatial.distance import squareform

                dist = np.clip(1 - np.abs(np.nan_to_num(spear.to_numpy(), nan=0.0)), 0, 1)
                np.fill_diagonal(dist, 0)
                order = [order[i] for i in leaves_list(linkage(squareform(dist, checks=False), method="average"))]
            except ValueError:
                pass
            add("association", P.heatmap(plots_dir / "correlation_heatmap_train.png", spear.loc[order, order],
                                         title="TRAIN ONLY: Spearman correlation between candidate predictor columns (clustered)", watermark=wm,
                                         basis=TRAIN_ONLY, label="Spearman rho"))
    add("temporal", P.temporal_bars(plots_dir / "temporal_audit.png", tables["temporal_audit"], watermark=wm))
    add("temporal", P.provenance_graph(plots_dir / "provenance_graph.png", tables["feature_provenance"], watermark=wm))
    add("quality", P.dq_summary(plots_dir / "data_quality_summary.png", tables["data_quality_checks"], watermark=wm))
    bal = tables.get("split_balance")
    if bal is not None and len(bal):
        d = bal[bal["kind"].isin(["binary", "continuous"])]
        add("split", P.smd_dots(plots_dir / "split_balance.png", d, value_cols=[("smd_validation_vs_train", "validation vs train"), ("smd_test_vs_train", "test vs train")],
                                label_col="variable", title="Split diagnostic: SMD of each variable against TRAIN (grey band = |SMD| <= 0.1)",
                                watermark=wm, basis=SPLIT_DIAGNOSTIC))
    _ = cps
    res.plots = out


def _identifier_values(frame: pd.DataFrame, contract: Any, research_ids: set[str]) -> set[str]:
    vals: set[str] = set(research_ids)
    for c in contract.columns:
        if c.name in frame.columns and (c.role == "IDENTIFIER" or c.semantic == "identifier"):
            vals |= {str(v).strip() for v in frame[c.name].dropna().astype(str) if len(str(v).strip()) >= 4}
    return vals


_DATA_URI = re.compile(r"data:image/png;base64,[A-Za-z0-9+/=]+")
_TOKEN = re.compile(r"[^\W_]+(?:[_-][^\W_]+)*", re.UNICODE)


def privacy_scan(out: Path, identifiers: set[str], pepper: str) -> dict[str, Any]:
    """Every text output of the EDA must be free of identifier values and of the pepper (images are aggregate by construction)."""
    hits: list[str] = []
    n_files = 0
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".csv", ".md", ".json", ".html", ".txt"):
            continue
        n_files += 1
        text = _DATA_URI.sub(" ", p.read_text(encoding="utf-8", errors="ignore"))
        if pepper and pepper in text:
            hits.append(f"{p.relative_to(out).as_posix()}: pseudonymisation pepper")
        tokens = set(_TOKEN.findall(text)) | set(re.findall(r"[A-Za-z0-9]+", text))
        found = tokens & identifiers
        if found:
            hits.append(f"{p.relative_to(out).as_posix()}: {len(found)} identifier value(s)")
    return {"passed": not hits, "files_scanned": n_files, "identifier_values_checked": len(identifiers), "pepper_checked": bool(pepper), "hits": hits}


def _manifest(res: EDAResult, mapping: Any, contract: Any, dictionary: Any, spec: Any, input_sha: str, read: Any, ref: dict[str, Any] | None,
              integrity: dict[str, Any] | None, scan: dict[str, Any], min_cell: int) -> dict[str, Any]:
    from falls_ml import __version__

    files = {}
    for p in sorted(res.eda_dir.glob("*.csv")):
        basis, what = FILE_BASIS.get(p.name, (FULL_EXTRACT, ""))
        files[f"eda/{p.name}"] = {"basis": basis, "what": what, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return {"watermark": res.watermark, "synthetic_data": res.synthetic, "falls_ml_version": __version__, "command": "meuhedet-eda",
            "input": {"file_name": res.input_file, "sha256": input_sha, "format": read.source_format, "rows": res.facts["n_rows"]},
            "definitions": {"contract": {"name": contract.name, "version": contract.version, "sha256": contract.content_sha256},
                            "mapping": {"name": mapping.name, "version": mapping.version, "sha256": mapping.content_sha256},
                            "data_dictionary": {"name": dictionary.name, "version": dictionary.version, "sha256": dictionary.sha256},
                            "feature_spec": {"name": spec.name, "version": spec.version, "sha256": spec.content_sha256}},
            "index_date": res.index_date, "index_day_records_policy": res.policy, "stages": res.stages,
            "split": {k: v for k, v in res.facts.get("split", {}).items() if not k.startswith("_")},
            "reference": None if ref is None else {"dir": ref["dir"].name, "input_sha256_matches": True, "test_partition_matches": res.facts.get("split", {}).get("matches_reference", False),
                                                   "integrity": integrity},
            "bases": BASIS_TEXT, "files": files, "plots": res.plots,
            "privacy": {"min_cell": min_cell, "identifier_scan": {k: v for k, v in scan.items() if k != "hits"},
                        "rule": "aggregate only; no identifiers, pseudonyms, raw rows or row-level predictions; counts 1..min_cell-1 suppressed; "
                                "the canonical dataset used for the split lived in a temporary directory that was deleted; the pepper is never written here"},
            "timings_seconds": res.timings}
