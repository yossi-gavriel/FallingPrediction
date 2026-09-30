"""One command for the real-data exploratory experiment on the Meuhedet wide table (Phase 1, ``falls_ml meuhedet-explore``).

    local Excel / CSV / Parquet export of V_Falls_Prediction_Wide_1
        -> contract-driven reading (declared date formats, SQL date range, declared sentinels; identifier corruption fails loudly)
        -> aggregate audit, D-00 timing diagnostic, snapshot audit (180-day outcome observability per Index_Date)
        -> one snapshot (--index-date) or every usable snapshot (--all-index-dates, temporal train / validation / test)
        -> STRICT (HIGH_CONFIDENCE mappings) and EXTENDED (+ APPROXIMATE) canonical datasets on identical rows, pseudonymised ids
           with a pepper that is created once next to the input and reused on every later run (same file + same config => same split)
        -> split audit (row counts, unique patients, partition hashes), pre-training adequacy gate
        -> READINESS: "READY TO TRAIN" or "STOPPED - <reason>" (--audit-only ends here)
        -> two runs of the existing reduced-eFalls LASSO experiment (same test rows, verified by hash); in the multi-snapshot mode
           also the patient-disjoint sensitivity runs and the temporal test performance by snapshot
        -> feature reports, STRICT-vs-EXTENDED comparison, one plain-language SUMMARY

Every output is aggregate; patient identifiers never leave the dataset directory (research ids are salted hashes).
Every report is watermarked "EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION".
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from falls_ml.adequacy import AdequacyReport, assess_adequacy, write_adequacy
from falls_ml.data.dataset import ModelingDataset, sha256_file
from falls_ml.data.meuhedet_audit import audit_wide_extract, write_audit
from falls_ml.data.meuhedet_snapshots import USABLE, SnapshotRow, TemporalDesign, choose_temporal_design, snapshot_audit, write_snapshot_audit
from falls_ml.data.meuhedet_timing import timing_diagnostic, write_timing_diagnostic
from falls_ml.data.meuhedet_wide import (DEFAULT_MAPPING, BuildReport, WideMapping, build_meuhedet_dataset_from_frame,
                                         load_wide_contract, load_wide_mapping, read_wide_extract_report)
from falls_ml.errors import ConfigError, DatasetValidationError
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_TEMPLATE = "configs/experiments/meuhedet/explore_180d_template.yaml"
FEATURE_SETS = ("strict", "extended")
RUN_LABELS = {"strict": "MEUHEDET_EFALLS_STRICT_180D_EXPLORATORY", "extended": "MEUHEDET_EFALLS_EXTENDED_180D_EXPLORATORY"}
SENSITIVITY_LABELS = {"strict": "MEUHEDET_EFALLS_STRICT_180D_PATIENT_DISJOINT", "extended": "MEUHEDET_EFALLS_EXTENDED_180D_PATIENT_DISJOINT"}
SET_TITLES = {"strict": "STRICT (HIGH_CONFIDENCE mappings only)", "extended": "EXTENDED (HIGH_CONFIDENCE + APPROXIMATE mappings)"}
FAST_SETTINGS = {"bootstrap_n": 200, "permutation_repeats": 3, "stability_n": 30, "optimism_n": 5}
SCIENTIFIC_STATUS = "EXPLORATORY — NOT EFALLS REPRODUCTION"
READY_TEXT = "READY TO TRAIN"
STOPPED_TEXT = "STOPPED"
INSUFFICIENT_EVENTS = "INSUFFICIENT EVENTS"
MIN_METRIC_EVENTS = 10   # the framework's floor for reporting a metric on a subset (subgroups, clusters, snapshots)
MAPPING_CAVEATS = [
    "polypharmacy_count_120d is approximated from Distinct_Active_Substance_Count (active exposure at index); not proven equivalent to the "
    "published 120-day unique prescribed-drug count (Q-M-07)",
    "falls is approximated from Prior_Fall_Since_Study_Start_Ind: fall AND fracture events since 2022-01-01 (<= 3 years) vs the eFalls 5-year "
    "fall history; index-day inclusion under review (Q-M-06, D-00 diagnostic)",
    "chronic_kidney_disease uses the chronic-renal-failure registry; the published eGFR / CKD-stage detail is not reproduced",
    "severe_mental_illness (Registry_SMI_Level >= 1) and self_harm (suicide-attempt registry) are registry approximations of the eFalls code groups (Q-M-03)",
    "fracture and fragility_fracture (mandatory eFalls predictors) are unavailable: the VIEW does not separate fractures from falls",
    "mobility_problems is the single ICD-9 code 719.7 since 2022-01-01 vs the eFalls mobility / transfer code group",
    "the exact published fall / fracture code lists and the ED-attendance / admission outcome are not available in V1.0: exact eFalls "
    "reproduction stays blocked (365-day outcome absent)",
]


def watermark(mapping: WideMapping) -> str:
    days = int(mapping.outcome["window_days_including_index"]) - 1
    return f"EXPLORATORY {days}-DAY OUTCOME – NOT EFALLS REPRODUCTION"


@dataclass
class ExploreResult:
    out_dir: Path
    input_file: str
    index_date: str
    mode: str = "single"                                  # single | all_snapshots
    datasets: dict[str, Path] = field(default_factory=dict)
    runs: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    build_reports: dict[str, BuildReport] = field(default_factory=dict)
    feature_reports: dict[str, pd.DataFrame] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""
    audit: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    snapshots: list[SnapshotRow] = field(default_factory=list)
    snapshot_totals: dict[str, Any] = field(default_factory=dict)
    design: TemporalDesign | None = None
    split_audit: dict[str, Any] = field(default_factory=dict)
    adequacy: dict[str, AdequacyReport] = field(default_factory=dict)
    readiness: dict[str, Any] = field(default_factory=dict)
    sensitivity_runs: dict[str, Path] = field(default_factory=dict)
    sensitivity_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    sensitivity_comparison: dict[str, Any] = field(default_factory=dict)
    by_snapshot: dict[str, pd.DataFrame] = field(default_factory=dict)
    pepper_sha256: str = ""
    pepper_source: str = ""
    model_report: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.readiness.get("verdict") == READY_TEXT


# ---------------------------------------------------------------------------- helpers
def infer_index_date(frame: pd.DataFrame, column: str = "Index_Date") -> str:
    """The single index date of the extract, or a listing of the parsed index dates and how to choose (one snapshot per run, or
    every usable snapshot with --all-index-dates)."""
    dates = frame[column].dt.normalize().dropna().unique()
    if len(dates) == 1:
        return str(pd.Timestamp(dates[0]).date())
    counts = frame[column].dt.normalize().value_counts(dropna=False).sort_index()
    listed = [f"{(str(k.date()) if pd.notna(k) else 'NULL')}: {int(v)} rows" for k, v in counts.items()]
    first = next((str(k.date()) for k in counts.index if pd.notna(k)), "YYYY-MM-DD")
    raise DatasetValidationError(f"The extract holds {len(dates)} index dates ({column} parsed with the contract's declared date "
                                 "formats); one run models one snapshot, so choose one with --index-date, or model every usable "
                                 "snapshot with --all-index-dates (temporal train / validation / test)",
                                 [*listed, f"for example: --index-date {first}", "or: --all-index-dates"])


def fp_variables(spec: FeatureSpec, features: list[str]) -> list[str]:
    """Predictors of ``features`` whose retrained eFalls transformation is fractional-polynomial selection (from the spec, D-13)."""
    out = []
    for name in features:
        tr = (spec.get(name).raw.get("transformation") or {}).get("retrained", "")
        if str(tr).startswith("fractional_polynomial"):
            out.append(name)
    return out


def resolve_id_pepper(src: Path, out: Path, *, explicit: str | None = None, pepper_file: str | Path | None = None) -> tuple[str, dict[str, Any]]:
    """The pseudonymisation pepper of this input file. Precedence: ``explicit`` (--id-pepper) > the sidecar file next to the input
    (``<file>.id_pepper.txt``, or --id-pepper-file) > a new random pepper written to that sidecar so every later run on the same file
    reuses it without anyone having to remember or type a value. Reports only the sha256 fingerprint of the pepper; the value itself stays local."""
    path = Path(pepper_file) if pepper_file else src.with_name(src.name + ".id_pepper.txt")
    stored = None
    if path.is_file():
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        stored = lines[0] if lines else None
    warnings: list[str] = []
    if explicit:
        pepper, source = explicit, "--id-pepper (given on the command line)"
        if stored is not None and stored != explicit:
            warnings.append(f"--id-pepper differs from the pepper stored in {path.name}; the research ids and the split differ from runs that used the file")
    elif stored is not None:
        pepper, source = stored, f"reused from {path}"
    else:
        pepper = secrets.token_hex(16)
        try:
            path.write_text("# falls_ml pseudonymisation pepper for " + src.name + " - keep local, never share, never commit\n" + pepper + "\n",
                            encoding="utf-8", newline="\n")
            source = f"created and stored in {path}"
        except OSError as exc:
            source = f"created (could not write {path}: {exc}); pass --id-pepper-file <this run's id_pepper.txt> to reproduce the split later"
            warnings.append(source)
    (out / "id_pepper.txt").write_text(pepper + "\n", encoding="utf-8", newline="\n")
    fp = hashlib.sha256(pepper.encode("utf-8")).hexdigest()
    return pepper, {"source": source, "file": str(path), "sha256": fp, "warnings": warnings,
                    "rule": "research_id = sha256(pepper | Customer_Full_ID)[:20]; the same pepper on the same file gives the same ids and the same split"}


def _write_config(template_path: str | Path, out_path: Path, *, name: str, description: str, features: list[str], fast: bool,
                  spec: FeatureSpec, validation: dict[str, Any] | None = None, analysis: dict[str, Any] | None = None) -> Path:
    from falls_ml.paths import resolve_path

    cfg = yaml.safe_load(resolve_path(template_path).read_text(encoding="utf-8"))
    cfg["experiment"]["name"] = name
    cfg["experiment"]["description"] = description
    cfg["preprocessing"]["features"] = list(features)
    cfg["preprocessing"]["fractional_polynomial"]["variables"] = fp_variables(spec, features)
    if validation is not None:
        cfg["validation"] = {**{k: v for k, v in cfg["validation"].items() if k in ("seed", "cv_folds", "outcome_lag_days")}, **validation}
    if fast:
        cfg["evaluation"]["bootstrap"]["n"] = FAST_SETTINGS["bootstrap_n"]
        cfg["evaluation"]["permutation_importance_repeats"] = FAST_SETTINGS["permutation_repeats"]
        cfg["analysis"]["stability"]["n_bootstrap"] = FAST_SETTINGS["stability_n"]
        cfg["analysis"]["optimism"]["n_bootstrap"] = FAST_SETTINGS["optimism_n"]
    for section, values in (analysis or {}).items():
        cfg["analysis"][section] = {**cfg["analysis"][section], **values}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8", newline="\n")
    return out_path


def _est(v: Any) -> float | None:
    if isinstance(v, dict):
        return v.get("estimate")
    return v if isinstance(v, (int, float)) else None


def _ci(v: Any) -> str:
    if isinstance(v, dict) and v.get("ci_low") is not None and v.get("ci_high") is not None:
        return f"{v['ci_low']:.3f}–{v['ci_high']:.3f}"
    return ""


def _fmt(v: Any, digits: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def _distribution(s: pd.Series, dtype: str) -> str:
    v = s.dropna()
    if v.empty:
        return "all missing"
    if dtype == "binary":
        return f"prevalence {100.0 * float(v.astype('float64').mean()):.1f}%"
    if dtype.startswith("categorical"):
        shares = v.astype(str).value_counts(normalize=True)
        return "; ".join(f"{k} {100.0 * p:.1f}%" for k, p in shares.items())
    x = v.astype("float64")
    return f"median {x.median():.1f} (IQR {x.quantile(0.25):.1f}–{x.quantile(0.75):.1f}); mean {x.mean():.2f}; range {x.min():.0f}–{x.max():.0f}"


def _keys_sha256(part: pd.DataFrame, spec: FeatureSpec) -> str:
    keys = part[spec.identifier_columns[0]].astype("str") + "|" + pd.to_datetime(part[spec.index_column]).dt.strftime("%Y-%m-%d")
    return hashlib.sha256("\n".join(sorted(keys.tolist())).encode("utf-8")).hexdigest()


def split_audit(frame: pd.DataFrame, spec: FeatureSpec, plan: Any, *, strategy_note: str, reproducibility: dict[str, Any]) -> dict[str, Any]:
    """Row counts, unique patients, events, index-date ranges and key hashes per partition; patients appearing in several partitions;
    the reproducibility metadata (input / pepper / definition hashes). No identifiers."""
    id_col, idx_col, y_col = spec.identifier_columns[0], spec.index_column, spec.outcome.name
    parts, patients = {}, {}
    for name, idx in plan.indices().items():
        part = frame.iloc[idx]
        y = part[y_col].to_numpy(dtype=np.int64)
        patients[name] = set(part[id_col].astype(str))
        parts[name] = {"n_rows": int(len(part)), "n_patients": len(patients[name]), "n_events": int(y.sum()), "prevalence": float(y.mean()) if len(y) else None,
                       "index_date_min": str(part[idx_col].min().date()) if len(part) else None, "index_date_max": str(part[idx_col].max().date()) if len(part) else None,
                       "n_snapshots": int(part[idx_col].nunique()), "rows_sha256": _keys_sha256(part, spec)}
    overlap = {f"{a}&{b}": len(patients[a] & patients[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    all_ids = frame[id_col].astype(str)
    return {"strategy": plan.strategy, "description": plan.description, "limitation_note": plan.limitation_note, "note": strategy_note,
            "metadata": dict(plan.metadata), "partitions": parts, "test_rows_sha256": parts["test"]["rows_sha256"],
            "patients_in_several_partitions": overlap, "patients_in_any_two_partitions": len((patients["train"] & patients["validation"]) |
                                                                                             (patients["train"] & patients["test"]) | (patients["validation"] & patients["test"])),
            "n_rows_total": int(len(frame)), "n_patients_total": int(all_ids.nunique()), "mean_snapshots_per_patient": float(all_ids.value_counts().mean()),
            "rows_used_by_split": int(sum(p["n_rows"] for p in parts.values())), "rows_excluded_by_split": int(len(frame) - sum(p["n_rows"] for p in parts.values())),
            "reproducibility": reproducibility}


def render_split_audit(sa: dict[str, Any]) -> str:
    lines = ["# Split audit (aggregate, non-identifying)", "", f"Strategy: **{sa['strategy']}** - {sa['description']}", "", sa["note"], "",
             f"Rows in the dataset {sa['n_rows_total']}; unique patients {sa['n_patients_total']}; mean snapshots per patient {sa['mean_snapshots_per_patient']:.2f}; "
             f"rows used by the split {sa['rows_used_by_split']}; rows excluded by the split (embargo / patient overlap) {sa['rows_excluded_by_split']} ({sa['metadata']}).", "",
             "| partition | rows | patients | events | prevalence | snapshots | index dates | rows sha256 |", "|---|---|---|---|---|---|---|---|"]
    for name, p in sa["partitions"].items():
        lines.append(f"| {name} | {p['n_rows']} | {p['n_patients']} | {p['n_events']} | {(p['prevalence'] or 0):.4f} | {p['n_snapshots']} | "
                     f"{p['index_date_min']}..{p['index_date_max']} | `{p['rows_sha256'][:16]}…` |")
    lines += ["", f"Patients present in several partitions: {sa['patients_in_several_partitions']} (any two: {sa['patients_in_any_two_partitions']}).",
              f"Test rows sha256: `{sa['test_rows_sha256']}`", "", "## Reproducibility", ""] + [f"- {k}: {v}" for k, v in sa["reproducibility"].items()]
    if sa.get("limitation_note"):
        lines += ["", f"Limitation: {sa['limitation_note']}"]
    return "\n".join(lines) + "\n"


def feature_report(features: list[str], mapping: WideMapping, spec: FeatureSpec, dataset_dir: Path, run_dir: Path,
                   metrics: dict[str, Any]) -> pd.DataFrame:
    """One row per predictor of a run: eFalls concept, mapping, feature-set membership, datatype, missing %, distribution, LASSO
    coefficient / odds ratio / non-zero status, bootstrap selection frequency and permutation importance. Ordered by permutation
    importance (a descriptive ranking on this label - not causal, not clinical importance)."""
    frame = ModelingDataset.load(dataset_dir, spec, features=features).frame
    coef = pd.read_csv(run_dir / "coefficients.csv") if (run_dir / "coefficients.csv").exists() else pd.DataFrame()
    if len(coef) and "relative_to_reference" in coef.columns:
        coef = coef[~coef["relative_to_reference"].astype(bool)]   # keep the fitted all-levels coefficients (D-10), not their re-expression
    stab = pd.read_csv(run_dir / "feature_stability.csv") if (run_dir / "feature_stability.csv").exists() else pd.DataFrame()
    imp = pd.read_csv(run_dir / "feature_importance.csv") if (run_dir / "feature_importance.csv").exists() else pd.DataFrame()
    missing = metrics.get("missingness") or {}
    sets = mapping.feature_sets()
    rows = []
    for name in features:
        fm = mapping.get(name)
        fd = spec.get(name)
        c = coef[coef["feature"] == name] if len(coef) else pd.DataFrame()
        c = c[c["design_column"] != "_intercept"] if len(c) else c
        if len(c) == 1:
            coef_txt, or_txt = _fmt(float(c["coefficient"].iloc[0])), _fmt(float(c["odds_ratio"].iloc[0]))
        elif len(c) > 1:
            coef_txt = "; ".join(f"{d}: {float(v):.3f}" for d, v in zip(c["design_column"], c["coefficient"]))
            or_txt = "; ".join(f"{d}: {float(v):.3f}" for d, v in zip(c["design_column"], c["odds_ratio"]))
        else:
            coef_txt = or_txt = "n/a"
        selected = bool(c["selected"].any()) if len(c) else None
        sel_freq = None
        if len(stab) and "raw_feature" in stab.columns:
            s = stab[stab["raw_feature"] == name]
            if len(s):
                col = "raw_feature_selection_frequency" if "raw_feature_selection_frequency" in s.columns else "selection_frequency"
                sel_freq = float(s[col].max())
        perm = None
        if len(imp):
            i = imp[imp["feature"] == name]
            if len(i) and pd.notna(i["permutation_importance_mean"].iloc[0]):
                perm = float(i["permutation_importance_mean"].iloc[0])
        max_abs = float(c["coefficient"].abs().max()) if len(c) else 0.0
        rows.append({"canonical_efalls_feature": name, "efalls_concept": fm.efalls.get("concept"), "meuhedet_source_column": "; ".join(fm.source_columns),
                     "mapping_quality": fm.quality, "feature_sets": "; ".join(k for k, v in sets.items() if name in v),
                     "semantic_dtype": fm.canonical_dtype or fd.dtype, "model_dtype": fm.model_dtype,
                     "missing_pct": round(100.0 * float((missing.get(name) or {}).get("missing_rate", 0.0)), 2),
                     "distribution": _distribution(frame[name], fd.dtype), "lasso_coefficient": coef_txt, "odds_ratio": or_txt,
                     "non_zero_coefficient": ("yes" if selected else "no") if selected is not None else "n/a",
                     "selection_frequency": None if sel_freq is None else round(sel_freq, 3),
                     "permutation_importance": None if perm is None else round(perm, 5), "_sort": (perm if perm is not None else -1.0, max_abs)})
    df = pd.DataFrame(rows)
    df = df.sort_values("_sort", ascending=False, key=lambda s: s.map(lambda t: t[0] * 1e6 + t[1])).drop(columns="_sort").reset_index(drop=True)
    return df


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)).replace("|", "/") for v in r.tolist()) + " |")
    return "\n".join(lines) + "\n"


def comparison(metrics: dict[str, dict[str, Any]], features: dict[str, list[str]]) -> dict[str, Any]:
    """STRICT vs EXTENDED on the same test rows (hash-verified)."""
    shas = {k: m["split"]["test_rows_sha256"] for k, m in metrics.items()}
    if len(set(shas.values())) != 1:
        raise DatasetValidationError("STRICT and EXTENDED runs do not share the same test rows", [f"{k}: {v}" for k, v in shas.items()])
    out: dict[str, Any] = {"same_test_rows": True, "test_rows_sha256": next(iter(shas.values())), "runs": {}}
    for k, m in metrics.items():
        test = m["performance"]["test"]
        primary = m.get("primary_variant") or "uncalibrated"
        p, u = test.get(primary) or {}, test.get("uncalibrated") or {}
        r = test.get("recalibrated") or {}
        coef = m.get("lasso") or {}
        out["runs"][k] = {
            "label": m["experiment"]["name"], "n_test": p.get("n"), "n_events_test": p.get("n_events"),
            "event_prevalence_test": (p.get("n_events") / p.get("n")) if p.get("n") else None,
            "n_predictors": len(features[k]), "predictors": list(features[k]),
            "auroc": _est(p.get("auroc")), "auroc_ci": _ci(p.get("auroc")), "pr_auc": _est(p.get("pr_auc")), "pr_auc_ci": _ci(p.get("pr_auc")),
            "brier": _est(p.get("brier")), "brier_ci": _ci(p.get("brier")),
            "calibration_intercept_uncalibrated": _est(u.get("calibration_intercept")), "calibration_slope_uncalibrated": _est(u.get("calibration_slope")),
            "calibration_intercept_recalibrated": _est(r.get("calibration_intercept")), "calibration_slope_recalibrated": _est(r.get("calibration_slope")),
            "oe_ratio_uncalibrated": _est(u.get("oe_ratio")), "primary_variant": primary,
            "lambda_star": coef.get("lambda_star"), "selected_design_columns": coef.get("n_selected"),
            "selected_raw_features": m.get("features", {}).get("n_selected_raw"),
            "split": {"strategy": m["split"].get("strategy"), "test_rows_sha256": m["split"].get("test_rows_sha256")},
        }
    return out


def _selected_raw_features(run_dir: Path) -> list[str]:
    p = run_dir / "coefficients.csv"
    if not p.exists():
        return []
    c = pd.read_csv(p)
    c = c[c["design_column"] != "_intercept"]
    if "relative_to_reference" in c.columns:
        c = c[~c["relative_to_reference"].astype(bool)]
    return sorted(set(c.loc[c["selected"].astype(bool), "feature"]))


def render_comparison(comp: dict[str, Any], wm: str, title: str = "STRICT vs EXTENDED comparison", note: str = "") -> str:
    rows = [("Run label", "label"), ("Test rows (N)", "n_test"), ("Test events", "n_events_test"), ("Event prevalence (test)", "event_prevalence_test"),
            ("Number of predictors", "n_predictors"), ("AUROC", "auroc"), ("AUROC 95% CI", "auroc_ci"), ("PR-AUC", "pr_auc"), ("PR-AUC 95% CI", "pr_auc_ci"),
            ("Brier score", "brier"), ("Calibration intercept (uncalibrated)", "calibration_intercept_uncalibrated"),
            ("Calibration slope (uncalibrated)", "calibration_slope_uncalibrated"), ("Calibration intercept (after recalibration on validation)", "calibration_intercept_recalibrated"),
            ("Calibration slope (after recalibration on validation)", "calibration_slope_recalibrated"), ("O:E ratio (uncalibrated)", "oe_ratio_uncalibrated"),
            ("Design columns with a non-zero coefficient (LASSO)", "selected_design_columns"), ("Raw predictors with a non-zero coefficient (LASSO)", "selected_raw_features"),
            ("Lambda*", "lambda_star")]
    keys = list(comp["runs"])
    df = pd.DataFrame([{"metric": title_, **{k: _fmt(comp["runs"][k].get(key)) if not isinstance(comp["runs"][k].get(key), str) else comp["runs"][k].get(key) for k in keys}}
                       for title_, key in rows])
    text = (f"# {title}\n\n**{wm}**\n\nSame test rows in both runs (sha256 `{comp['test_rows_sha256']}`). "
            "A better EXTENDED score means the APPROXIMATE mappings carry predictive contribution on this label; it is not evidence that their "
            f"clinical definitions match eFalls, and no coefficient here is a causal or clinical-importance claim.{(' ' + note) if note else ''}\n\n" + _md_table(df))
    for k in keys:
        text += f"\n{k.upper()} predictors ({comp['runs'][k]['n_predictors']}): {', '.join(comp['runs'][k]['predictors'])}\n"
    return text


def performance_by_snapshot(run_dir: Path, served_variant: str) -> pd.DataFrame:
    """Test performance per Index_Date from the run's own test predictions (a slice of the one released test evaluation - no new
    evaluation). Snapshots with fewer than MIN_METRIC_EVENTS events or non-events print INSUFFICIENT EVENTS."""
    from falls_ml.evaluation import metrics as M

    pred = pd.read_parquet(run_dir / "predictions_test.parquet")
    rows = []
    for date, g in pred.groupby(pd.to_datetime(pred["index_date"]).dt.strftime("%Y-%m-%d")):
        y = g["outcome"].to_numpy(dtype=np.int64)
        p = g["risk_served"].to_numpy(dtype="float64")
        lp = g["linear_predictor"].to_numpy(dtype="float64")
        n, ev = int(len(y)), int(y.sum())
        row: dict[str, Any] = {"index_date": date, "n": n, "n_events": ev, "prevalence": ev / n if n else None, "status": "ok",
                               "auroc": None, "brier": None, "oe_ratio": None, "calibration_slope": None, "citl": None}
        if ev < MIN_METRIC_EVENTS or n - ev < MIN_METRIC_EVENTS:
            row["status"] = INSUFFICIENT_EVENTS
        else:
            lp_eff = lp if np.isfinite(lp).all() else M.logit(p)
            row.update({"auroc": M.auroc(y, p), "brier": M.brier(y, p), "oe_ratio": M.oe_ratio(y, p),
                        "calibration_slope": M.calibration_slope_intercept(y, lp_eff)[1], "citl": M.citl(y, lp_eff)[0]})
        rows.append(row)
    return pd.DataFrame(rows).assign(served_variant=served_variant)


def render_by_snapshot(tables: dict[str, pd.DataFrame], wm: str) -> str:
    text = f"# Temporal test performance by snapshot\n\n**{wm}**\n\nEach table slices the single released test evaluation of a run by Index_Date; " \
           f"snapshots with fewer than {MIN_METRIC_EVENTS} events or non-events print `{INSUFFICIENT_EVENTS}` instead of a metric.\n"
    for fs, df in tables.items():
        show = df.copy()
        for c in ("prevalence", "auroc", "brier", "oe_ratio", "calibration_slope", "citl"):
            show[c] = show[c].map(lambda v: _fmt(float(v)) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "")
        text += f"\n## {fs.upper()}\n\n" + _md_table(show)
    return text


# ---------------------------------------------------------------------------- summary
def render_summary(res: ExploreResult, mapping: WideMapping, wm: str, fast: bool) -> str:
    br = next(iter(res.build_reports.values()))
    comp = res.comparison.get("runs", {})
    m_ext = res.metrics.get("extended") or (next(iter(res.metrics.values())) if res.metrics else {})
    fr_ext = res.feature_reports.get("extended")
    stable, eliminated = [], {}
    for k, fr in res.feature_reports.items():
        eliminated[k] = list(fr.loc[fr["non_zero_coefficient"] == "no", "canonical_efalls_feature"])
    if fr_ext is not None and fr_ext["selection_frequency"].notna().any():
        top = fr_ext.dropna(subset=["selection_frequency"]).sort_values("selection_frequency", ascending=False)
        stable = [f"{r.canonical_efalls_feature} ({r.selection_frequency:.2f})" for r in top.head(6).itertuples()]
    audit, dq, dt = res.audit, res.audit.get("data_quality", {}), res.audit["datatype_audit"]
    findings = []
    if dt["wrong_type_total"]:
        findings.append(f"{dt['wrong_type_total']} values could not be represented as their contract type: {dt['wrong_type']}")
    if dt.get("wrong_type_examples"):
        findings.append(f"offending date values (now NULL, counted above): {dt['wrong_type_examples']}; declared source date formats: {dt.get('declared_date_formats')}")
    for col, dr in (dt.get("date_range") or {}).items():
        if dr.get("undeclared_beyond_ns_range_values"):
            findings.append(f"{col}: far-future dates that are not declared sentinels {dr['undeclared_beyond_ns_range_values']} (kept as dates; declare them in the contract if they are business sentinels)")
        elif dr.get("declared_sentinels"):
            findings.append(f"{col}: declared sentinel dates {dr['declared_sentinels']} = {dr.get('sentinel_means')} (rule {dr.get('sentinel_modeling_rule')})")
    if dt["non_null_violations"]:
        findings.append(f"NULL in DDL NOT NULL columns: {dt['non_null_violations']}")
    if dt["invalid_values"]:
        findings.append(f"values outside the allowed sets: {dt['invalid_values']}")
    if dq.get("constant_columns"):
        findings.append(f"constant columns among cohort rows: {dq['constant_columns'][:15]}")
    if dq.get("near_empty_columns"):
        findings.append(f"near-empty columns (>= 99% NULL): {dq['near_empty_columns'][:15]}")
    if dq.get("duplicate_snapshot_key"):
        findings.append(f"duplicate Snapshot_Key rows: {dq['duplicate_snapshot_key']}")
    if br.n_rows_label_null:
        findings.append(f"{br.n_rows_label_null} eligible rows without a usable 180-day label were excluded: {br.label_null_by_reason}")
    tv = res.timing.get("verdict", {})
    if tv:
        findings.append(f"D-00 timing: {tv['n_eligible_rows_violating_d00']} eligible rows ({tv['pct_of_eligible']}%) with a predictor record on/after the index day; "
                        f"root cause {tv['root_cause']}; affected eFalls features {tv['affected_efalls_features_by_feature_set']}; "
                        f"{'rows dropped and counted (--index-day-records drop_rows)' if br.n_rows_timing_violation else 'no row dropped'} - see audit/timing_diagnostic.md")
    if br.sentinel_to_null:
        findings.append(f"declared sentinel dates set to NULL in the modelling layer (column rule): {br.sentinel_to_null}")
    if br.birth_date_suspect_rows:
        findings.append(f"{br.birth_date_suspect_rows} modelling rows have a suspect (default) birth date (Birth_Date_Suspect_Ind = 1) - age unreliable there")
    if br.null_to_zero:
        findings.append(f"NULL -> 0 by the declared eFalls absent_is_zero rule (counted): {br.null_to_zero}")
    if br.source_absent_rows:
        findings.append(f"rows whose source was absent (predictor 0 by construction): {br.source_absent_rows}")
    if audit["leakage_audit"]["result"] != "PASS":
        findings.append(f"leakage audit of the raw extract: {audit['leakage_audit']['result']} - predictor record dates on/after the index day "
                        f"{audit['leakage_audit']['predictor_record_dates_on_or_after_index']}"
                        + ("; those rows were excluded from the modelling datasets (counted above)" if br.n_rows_timing_violation else ""))
    for fs, a in res.adequacy.items():
        for w in a.warnings:
            findings.append(f"adequacy ({fs}): {w}")
    findings.append(f"365-day eFalls outcome: {audit['outcomes']['efalls_365d_outcome']['verdict']}")
    synthetic = bool(m_ext.get("synthetic_fixture")) or (m_ext.get("dataset", {}).get("source") == "synthetic_fixture") or br.synthetic
    st = res.snapshot_totals
    lines = ["SYNTHETIC DEMO EXPERIMENT — SYNTHETIC DATA – NOT SCIENTIFIC RESULTS" if synthetic else "REAL-DATA EXPLORATORY EXPERIMENT",
             "", f"Readiness: {res.readiness.get('verdict')}", "", "Input:", f"  {res.input_file}", "",
             "Mode / index date(s):", f"  {res.mode}: {res.index_date}", "",
             "Row funnel (rows supplied -> other / excluded snapshots -> ineligible -> no usable label -> D-00 timing dropped -> used):",
             f"  {br.n_rows_input} -> {br.n_rows_other_index_dates} -> {br.n_rows_ineligible} -> {br.n_rows_label_null} -> {br.n_rows_timing_violation} -> {br.n_rows_final}",
             f"  exclusion reasons: {br.ineligible_by_reason or 'none'}", f"  excluded snapshots: {st.get('excluded_snapshots') or 'none'}", "",
             "Patients / snapshot rows used for modelling:", f"  {br.n_patients_final} patients, {br.n_rows_final} rows"
             + (f" ({len(br.index_dates)} snapshots; mean {br.n_rows_final / max(br.n_patients_final, 1):.2f} rows per patient)" if len(br.index_dates) > 1 else ""), "",
             f"{watermark(mapping).split(' OUTCOME')[0].replace('EXPLORATORY ', '')} falls (positive / negative / prevalence):",
             f"  {br.n_events} / {br.n_rows_final - br.n_events} / {100.0 * (br.outcome_prevalence or 0):.2f}%", ""]
    sa = res.split_audit
    if sa:
        lines += [f"Split ({sa['strategy']}; identical rows in STRICT and EXTENDED, verified by hash):"]
        for part in ("train", "validation", "test"):
            s = sa["partitions"].get(part) or {}
            lines.append(f"  {part:<10} N {s.get('n_rows')}  patients {s.get('n_patients')}  events {s.get('n_events')}  prevalence {100.0 * (s.get('prevalence') or 0):.2f}%"
                         f"  index dates {s.get('index_date_min')}..{s.get('index_date_max')}  sha256 {str(s.get('rows_sha256'))[:12]}…")
        lines += [f"  patients in several partitions: {sa['patients_in_several_partitions']}; rows excluded by the split: {sa['rows_excluded_by_split']} {sa['metadata']}",
                  f"  test rows sha256: {sa['test_rows_sha256']}", f"  pepper fingerprint (sha256 of the local pepper, value never shared): {res.pepper_sha256}", ""]
    if res.design is not None:
        lines += ["Temporal design (primary, repeated-risk):", f"  train_end {res.design.train_end}; validation_end {res.design.validation_end}; {res.design.rule}",
                  f"  embargo removed: {res.design.embargo_removed}", ""]
    for fs, a in res.adequacy.items():
        p = a.partitions
        lines.append(f"Adequacy ({fs}): {a.verdict}; train {p.get('train', {}).get('n_events')} events / validation {p.get('validation', {}).get('n_events')} / "
                     f"test {p.get('test', {}).get('n_events')}; constant predictors {a.predictors.get('constant_in_train')}; sparse {a.predictors.get('extremely_sparse_binary_in_train')}")
    lines.append("")
    for k in res.metrics:
        c = comp[k]
        lines += [k.upper(), f"  Label: {c['label']}", f"  Features: {c['n_predictors']}  ({', '.join(c['predictors'])})",
                  f"  AUROC: {_fmt(c['auroc'])} ({c['auroc_ci']})", f"  PR-AUC: {_fmt(c['pr_auc'])} ({c['pr_auc_ci']})", f"  Brier: {_fmt(c['brier'])}",
                  f"  Calibration (uncalibrated model on test): intercept {_fmt(c['calibration_intercept_uncalibrated'])}, slope {_fmt(c['calibration_slope_uncalibrated'])}, O:E {_fmt(c['oe_ratio_uncalibrated'])}",
                  f"  Calibration (after recalibration on validation): intercept {_fmt(c['calibration_intercept_recalibrated'])}, slope {_fmt(c['calibration_slope_recalibrated'])}",
                  f"  Non-zero LASSO coefficients: {c['selected_raw_features']} raw predictors ({c['selected_design_columns']} design columns); zero: {', '.join(eliminated.get(k, [])) or 'none'}",
                  f"  Run directory: {res.runs[k]}", ""]
    if res.by_snapshot:
        lines += ["Temporal test performance by snapshot (AUROC; INSUFFICIENT EVENTS below 10 events):"]
        for fs, df in res.by_snapshot.items():
            cells = [f"{r.index_date}: {_fmt(float(r.auroc)) if r.status == 'ok' else r.status} (n {r.n}, events {r.n_events})" for r in df.itertuples()]
            lines.append(f"  {fs.upper()}: " + "; ".join(cells))
        lines.append("")
    if res.sensitivity_comparison:
        sc = res.sensitivity_comparison["runs"]
        lines += ["Patient-disjoint sensitivity analysis (all snapshots of held-out patients; separate estimand: unseen patients):"]
        for k, c in sc.items():
            lines.append(f"  {k.upper()}: AUROC {_fmt(c['auroc'])} ({c['auroc_ci']}); PR-AUC {_fmt(c['pr_auc'])}; Brier {_fmt(c['brier'])}; "
                         f"slope {_fmt(c['calibration_slope_uncalibrated'])}; test N {c['n_test']} / events {c['n_events_test']}; run {res.sensitivity_runs[k]}")
        lines.append(f"  test rows sha256 {res.sensitivity_comparison['test_rows_sha256']}; no patient of the test partition appears in train or validation (verified)")
        lines.append("")
    lines += ["Most stable predictors (EXTENDED bootstrap selection frequency = share of bootstrap refits with a non-zero coefficient; 'stable' means repeatedly selected):", f"  {', '.join(stable) or 'n/a'}", "",
              "Predictors with a zero LASSO coefficient (EXTENDED):", f"  {', '.join(eliminated.get('extended', [])) or 'none'}", "",
              "Important data-quality findings:"] + [f"  - {f}" for f in findings] + ["", "Known mapping limitations (retained on purpose):"] + [f"  - {c}" for c in MAPPING_CAVEATS] + ["",
              f"Resampling: {'fast settings (' + ', '.join(f'{k} {v}' for k, v in FAST_SETTINGS.items()) + '); pass --full for the template values' if fast else 'template (full) settings'}",
              f"Mapping: {mapping.name} {mapping.version} ({mapping.status}); eFalls coverage {res.comparison.get('coverage_label', '')}", "",
              "Scientific status:", f"  {SCIENTIFIC_STATUS}", f"  {wm}", "",
              "Files: READINESS.md, audit/wide_extract_audit.md, audit/timing_diagnostic.md, audit/snapshot_audit.md, split_audit.md, adequacy_*.md, "
              "feature_report_strict.md, feature_report_extended.md, comparison.md" + (", temporal_test_by_snapshot.md, comparison_patient_disjoint.md" if res.mode == "all_snapshots" else "")
              + ", runs/<run>/report.html (plots: coefficients.png, feature_importance.png, selection_stability.png, calibration.png). "
              "id_pepper.txt links research ids to members: keep it local."]
    return "\n".join(lines) + "\n"


def render_readiness(res: ExploreResult, wm: str) -> str:
    r = res.readiness
    lines = [f"# {r['verdict']}", "", f"**{wm}**", "", f"Input: {res.input_file}; mode: {res.mode}; index date(s): {res.index_date}", ""]
    if r.get("reason"):
        lines += [f"Reason: {r['reason']}", ""] + [f"- {d}" for d in r.get("details", [])] + [""]
    lines += ["## Stages", ""] + [f"- {k}: {v}" for k, v in r.get("stages", {}).items()] + [""]
    if r.get("next"):
        lines += ["## Next", ""] + [f"- {n}" for n in r["next"]] + [""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- orchestration
class _Stop(Exception):
    def __init__(self, reason: str, details: list[str] | None = None, nxt: list[str] | None = None):
        super().__init__(reason)
        self.reason, self.details, self.next = reason, list(details or []), list(nxt or [])


def explore(input_path: str | Path, *, out_dir: str | Path | None = None, index_date: str | None = None, all_index_dates: bool = False,
            audit_only: bool = False, sheet: str | None = None, data_freeze_date: str | None = None, dataset_version: str | None = None,
            id_pepper: str | None = None, id_pepper_file: str | Path | None = None, fast: bool = True, encoding: str = "utf-8", sep: str = ",",
            mapping_path: str | Path = DEFAULT_MAPPING, template_path: str | Path = DEFAULT_TEMPLATE, index_day_records: str | None = None,
            feature_sets: tuple[str, ...] = FEATURE_SETS, min_cell: int = 10, max_censored_share: float = 0.5,
            patient_disjoint: bool = True, model_report: str = "full") -> ExploreResult:
    """Run the whole real-data path. Raises ``DatasetValidationError`` (after writing READINESS.md) when the data cannot be trained on."""
    src = Path(input_path)
    if not src.is_file():
        raise DatasetValidationError(f"Input file not found: {src}")
    if index_date is not None and all_index_dates:
        raise ConfigError("give either --index-date (one snapshot) or --all-index-dates (every usable snapshot), not both")
    out = Path(out_dir) if out_dir is not None else src.with_name(f"{src.stem}_explore")
    if out.exists() and any(out.iterdir()):
        raise DatasetValidationError(f"Output directory {out} is not empty; results are immutable - choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    for fs in feature_sets:
        if fs not in FEATURE_SETS:
            raise ConfigError(f"unknown feature set {fs!r}; choose from {FEATURE_SETS}")
    mapping = load_wide_mapping(mapping_path)
    contract = load_wide_contract(mapping.contract_path)
    spec = load_feature_spec(mapping.exploratory_spec_path)
    wm = watermark(mapping)
    sets = mapping.feature_sets()
    policy = index_day_records or mapping.cohort.get("index_day_records", "fail")
    res = ExploreResult(out_dir=out, input_file=src.name, index_date=index_date or "", mode="all_snapshots" if all_index_dates else "single")
    stages: dict[str, str] = {}
    res.readiness = {"verdict": STOPPED_TEXT, "stages": stages}
    try:
        _explore_stages(res, src, out, mapping=mapping, contract=contract, spec=spec, wm=wm, sets=sets, policy=policy, stages=stages,
                        index_date=index_date, all_index_dates=all_index_dates, audit_only=audit_only, sheet=sheet, data_freeze_date=data_freeze_date,
                        dataset_version=dataset_version, id_pepper=id_pepper, id_pepper_file=id_pepper_file, fast=fast, encoding=encoding, sep=sep,
                        template_path=template_path, feature_sets=feature_sets, min_cell=min_cell, max_censored_share=max_censored_share,
                        patient_disjoint=patient_disjoint)
    except _Stop as stop:
        res.readiness.update({"verdict": f"{STOPPED_TEXT} — {stop.reason}", "reason": stop.reason, "details": stop.details, "next": stop.next})
        _write_readiness(res, wm)
        raise DatasetValidationError(f"{STOPPED_TEXT} — {stop.reason}", [*stop.details, *(f"next: {n}" for n in stop.next),
                                                                          f"readiness report: {out / 'READINESS.md'}"]) from None
    if res.runs and model_report != "off":
        from falls_ml.modelreport.runner import attach_reports

        res.model_report = attach_reports([out], out / "reports", own_results=out, learning_curve=model_report)
        stages["model_report"] = res.model_report["status"]
        _write_readiness(res, wm)
    return res


def _write_readiness(res: ExploreResult, wm: str) -> None:
    (res.out_dir / "READINESS.md").write_text(render_readiness(res, wm), encoding="utf-8", newline="\n")
    payload = {**res.readiness, "mode": res.mode, "index_date": res.index_date, "input_file": res.input_file, "pepper_sha256": res.pepper_sha256,
               "split_audit": res.split_audit or None, "adequacy": {k: v.to_dict() for k, v in res.adequacy.items()},
               "snapshot_totals": res.snapshot_totals or None, "timing_verdict": res.timing.get("verdict")}
    (res.out_dir / "readiness.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8", newline="\n")


def _explore_stages(res: ExploreResult, src: Path, out: Path, *, mapping: WideMapping, contract: Any, spec: FeatureSpec, wm: str,
                    sets: dict[str, list[str]], policy: str, stages: dict[str, str], index_date: str | None, all_index_dates: bool, audit_only: bool,
                    sheet: str | None, data_freeze_date: str | None, dataset_version: str | None, id_pepper: str | None, id_pepper_file: str | Path | None,
                    fast: bool, encoding: str, sep: str, template_path: str | Path, feature_sets: tuple[str, ...], min_cell: int,
                    max_censored_share: float, patient_disjoint: bool) -> None:
    from falls_ml.config import load_experiment_config
    from falls_ml.experiment import run_experiment
    from falls_ml.splitting import make_split

    # 1. read + contract + audit (aggregate only)
    read = read_wide_extract_report(src, contract, encoding=encoding, sep=sep, sheet=sheet)
    input_sha = sha256_file(src)
    stages["input"] = f"{read.source_format}, {len(read.frame)} rows, wrong-type values {sum(read.wrong_type.values())}, declared date formats {list(read.declared_date_formats)}"
    audit = audit_wide_extract(read.frame, contract, mapping, spec, index_date=None if all_index_dates else index_date, min_cell=min_cell,
                               wrong_type=read.wrong_type, wrong_type_examples=read.wrong_type_examples, date_formats=read.date_formats)
    audit["input"] = {"file_name": src.name, "sha256": input_sha, **read.to_dict()}
    write_audit(audit, out / "audit")
    (out / "audit" / "input_read_report.json").write_text(json.dumps(audit["input"], indent=2, default=str), encoding="utf-8", newline="\n")
    res.audit = audit
    problems = audit["datatype_audit"]["contract_problems"]
    stages["contract"] = f"{len(problems)} problems, {len(audit['datatype_audit']['contract_warnings'])} warnings; sentinels {list((audit['datatype_audit'].get('date_range') or {}).keys())}"
    # 2. D-00 timing diagnostic
    res.timing = timing_diagnostic(read.frame, contract, mapping, min_cell=min_cell)
    write_timing_diagnostic(res.timing, out / "audit")
    tv = res.timing["verdict"]
    stages["d00_timing"] = f"{tv['n_eligible_rows_violating_d00']} eligible rows violate; root cause {tv['root_cause']}; policy {policy}"
    # 3. snapshot audit and mode
    rows, totals = snapshot_audit(read.frame, contract, mapping, data_freeze_date=data_freeze_date, max_censored_share=max_censored_share, index_day_records=policy)
    res.snapshots, res.snapshot_totals = rows, totals
    design: TemporalDesign | None = None
    if all_index_dates:
        usable = [r.index_date for r in rows if r.status == USABLE]
        stages["snapshots"] = f"{totals['n_snapshots']} snapshots, {len(usable)} usable, excluded {totals['excluded_snapshots']}"
        try:
            design = choose_temporal_design(rows, horizon_days=totals["horizon_days"])
        except DatasetValidationError as exc:
            write_snapshot_audit(rows, totals, None, out / "audit")
            raise _Stop(str(exc), exc.problems, ["run one snapshot with --index-date, or supply more usable snapshots"]) from None
        res.design = design
        index_dates = usable
        res.index_date = f"{usable[0]}..{usable[-1]} ({len(usable)} usable snapshots)"
    else:
        try:
            index_date = index_date or infer_index_date(read.frame, mapping.identity["index_date"])
        except DatasetValidationError as exc:
            write_snapshot_audit(rows, totals, None, out / "audit")
            raise _Stop(str(exc), exc.problems) from None
        row = next((r for r in rows if r.index_date == str(pd.Timestamp(index_date).date())), None)
        if row is None:
            write_snapshot_audit(rows, totals, None, out / "audit")
            raise _Stop(f"no rows with Index_Date = {index_date}", [f"snapshots present: {[r.index_date for r in rows]}"])
        if row.status != USABLE:
            write_snapshot_audit(rows, totals, None, out / "audit")
            raise _Stop(f"snapshot {index_date} is EXCLUDED: {row.reason}", [], ["choose a snapshot marked USABLE in audit/snapshot_audit.md"])
        index_dates = [row.index_date]
        res.index_date = row.index_date
        stages["snapshots"] = f"{totals['n_snapshots']} snapshots in the file; modelling {row.index_date} (USABLE)"
    write_snapshot_audit(rows, totals, design, out / "audit")
    # 4. pseudonymisation pepper: created once next to the input, reused afterwards
    pepper, pinfo = resolve_id_pepper(src, out, explicit=id_pepper, pepper_file=id_pepper_file)
    res.pepper_sha256, res.pepper_source = pinfo["sha256"], pinfo["source"]
    stages["pepper"] = pinfo["source"] + (f"; warnings {pinfo['warnings']}" if pinfo["warnings"] else "")
    # 5. canonical datasets (identical rows for every feature set by construction)
    version = dataset_version or f"meuhedet-wide-{res.index_date.split(' ')[0]}-{src.stem}"
    for fs in feature_sets:
        ds_dir = out / "datasets" / fs
        try:
            manifest, report = build_meuhedet_dataset_from_frame(
                read.frame, ds_dir, index_dates=index_dates, dataset_version=f"{version}-{fs}", data_freeze_date=data_freeze_date, mapping=mapping,
                contract=contract, spec=spec, input_name=src.name, input_sha256=input_sha, wrong_type=read.wrong_type,
                wrong_type_examples=read.wrong_type_examples, date_formats=read.date_formats, features=sets[fs], feature_set_label=fs, id_pepper=pepper,
                index_day_records=policy, read_report=read.to_dict(), notes=f"{wm}; feature set {fs}")
        except DatasetValidationError as exc:
            nxt = []
            if any("D-00 timing" in p for p in exc.problems):
                nxt = [f"read audit/timing_diagnostic.md (root cause {tv['root_cause']}; affected eFalls features {tv['affected_efalls_features_by_feature_set']})",
                       "correct the VIEW's predictor windows to Event_Date < Index_Date (see the DWH correction in the diagnostic) and re-export, or",
                       "re-run with --index-day-records drop_rows to exclude and count the violating rows (the diagnostic reports their label prevalence vs the clean rows)"]
            raise _Stop(f"the {fs} dataset cannot be built: {exc}", exc.problems, nxt) from None
        res.datasets[fs], res.build_reports[fs] = ds_dir, report
    br = res.build_reports[feature_sets[0]]
    stages["cohort"] = (f"{br.n_rows_input} rows -> other/excluded snapshots {br.n_rows_other_index_dates} -> ineligible {br.n_rows_ineligible} -> "
                        f"no usable label {br.n_rows_label_null} -> D-00 dropped {br.n_rows_timing_violation} -> {br.n_rows_final} rows, {br.n_patients_final} patients, {br.n_events} events")
    # 6. experiment configs and the split (computed once; every run recomputes the identical plan - verified by hash after training)
    if design is not None:
        validation = {"strategy": "temporal", "patients_disjoint": False, "temporal": {"train_end": design.train_end, "validation_end": design.validation_end,
                      "embargo_outcome_windows": True}, "limitation_note": design.limitation_note}
        split_note = "PRIMARY temporal repeated-risk design: earlier usable snapshots train, later ones validate, the latest test (D-19 embargo on)."
    else:
        validation = None
        split_note = ("Single index-date cohort (one row per patient): patient-grouped random split stratified by outcome (D-19 §10); the research ids "
                      "come from the stored pepper, so the split is identical on every rerun of this file.")
    configs: dict[str, Path] = {}
    for fs in feature_sets:
        description = (f"{wm}. {SET_TITLES[fs]}: {len(sets[fs])} eFalls-compatible predictors mapped from V_Falls_Prediction_Wide_1 "
                       f"({', '.join(sets[fs])}); outcome Fall_Next_180D_Ind; published learning process (FP selection + LASSO logistic, "
                       f"10-fold CV lambda-min) on unvalidated mappings ({mapping.name} {mapping.version}).")
        configs[fs] = _write_config(template_path, out / "configs" / f"{RUN_LABELS[fs]}.yaml", name=RUN_LABELS[fs], description=description,
                                    features=sets[fs], fast=fast, spec=spec, validation=validation)
    ext_fs = feature_sets[-1]
    cfg_obj = load_experiment_config(configs[ext_fs])
    ds = ModelingDataset.load(res.datasets[ext_fs], spec, features=sets[ext_fs])
    try:
        plan = make_split(ds.frame, ds.spec, cfg_obj.validation)
    except (ConfigError, DatasetValidationError) as exc:
        raise _Stop(f"the split cannot be built: {exc}") from None
    repro = {"input_sha256": input_sha, "pepper_sha256": res.pepper_sha256, "pepper_source": res.pepper_source, "contract_sha256": contract.content_sha256,
             "mapping_sha256": mapping.content_sha256, "feature_spec_sha256": spec.content_sha256, "split_seed": cfg_obj.validation.seed,
             "config_sha256": {fs: load_experiment_config(configs[fs]).sha256() for fs in feature_sets},
             "dataset_sha256": {fs: json.loads((res.datasets[fs] / "manifest.json").read_text(encoding="utf-8"))["data_sha256"] for fs in feature_sets},
             "rule": "same input file + same pepper + same configs => same research ids, same split, same partition hashes"}
    res.split_audit = split_audit(ds.frame, ds.spec, plan, strategy_note=split_note, reproducibility=repro)
    (out / "split_audit.json").write_text(json.dumps(res.split_audit, indent=2, default=str), encoding="utf-8", newline="\n")
    (out / "split_audit.md").write_text(render_split_audit(res.split_audit), encoding="utf-8", newline="\n")
    sa = res.split_audit["partitions"]
    stages["split"] = ", ".join(f"{k} {v['n_rows']} rows / {v['n_patients']} patients / {v['n_events']} events" for k, v in sa.items()) + f"; test sha256 {res.split_audit['test_rows_sha256'][:12]}…"
    # 7. adequacy gate (before any preprocessing or fitting)
    for fs in feature_sets:
        cfg_fs = load_experiment_config(configs[fs])
        ds_fs = ModelingDataset.load(res.datasets[fs], spec, features=sets[fs])
        res.adequacy[fs] = assess_adequacy(ds_fs.frame, ds_fs.spec, plan, cfg_fs)
        write_adequacy(res.adequacy[fs], out, stem=f"adequacy_{fs}")
    stages["adequacy"] = "; ".join(f"{fs}: {a.verdict}" for fs, a in res.adequacy.items())
    failing = {fs: a for fs, a in res.adequacy.items() if not a.ready}
    if failing:
        details = [f"{fs}: {r}" for fs, a in failing.items() for r in a.reasons]
        raise _Stop("INSUFFICIENT EVENTS FOR EXPLORATORY MODEL FIT", details,
                    ["no model was fitted; no AUROC, calibration, odds ratio or feature ranking was produced",
                     "a larger extract (more eligible rows on a usable snapshot, or --all-index-dates over more snapshots) is needed"])
    res.readiness.update({"verdict": READY_TEXT, "reason": "", "details": [], "next": [] if not audit_only else
                          ["run the same command without --audit-only to train STRICT and EXTENDED"]})
    _write_readiness(res, wm)
    if audit_only:
        res.summary_text = render_readiness(res, wm)
        return
    # 8. primary runs
    features_used = {fs: sets[fs] for fs in feature_sets}
    for fs in feature_sets:
        result = run_experiment(None, res.datasets[fs], configs[fs], runs_dir=out / "runs")
        res.runs[fs], res.metrics[fs] = Path(result.run_dir), result.metrics
        if result.metrics["split"]["test_rows_sha256"] != res.split_audit["test_rows_sha256"]:
            raise DatasetValidationError("the run's test rows differ from the split audit computed before training (non-deterministic split?)",
                                         [f"audit {res.split_audit['test_rows_sha256']}", f"run {result.metrics['split']['test_rows_sha256']}"])
        fr = feature_report(sets[fs], mapping, spec, res.datasets[fs], Path(result.run_dir), result.metrics)
        res.feature_reports[fs] = fr
        fr.to_csv(out / f"feature_report_{fs}.csv", index=False, lineterminator="\n")
        text = (f"# Feature report — {RUN_LABELS[fs]}\n\n**{wm}**\n\n{SET_TITLES[fs]}; {len(sets[fs])} predictors; run `{Path(result.run_dir).name}`.\n"
                "Ordered by permutation importance, then |coefficient| (a descriptive ranking on this label - not causal, not clinical importance, "
                "not a validation of the clinical definitions; 'non-zero coefficient' = retained by the LASSO at lambda*, 'selection frequency' = "
                "share of bootstrap refits that retained it).\n\n" + _md_table(fr))
        (out / f"feature_report_{fs}.md").write_text(text, encoding="utf-8", newline="\n")
        log.info("meuhedet_explore_run_done", extra_fields={"feature_set": fs, "run_id": result.run_id, "n_features": len(sets[fs])})
    for fs in feature_sets:
        res.metrics[fs].setdefault("features", {})["n_selected_raw"] = len(_selected_raw_features(res.runs[fs]))
    res.comparison = comparison(res.metrics, features_used)
    cov = br.coverage
    res.comparison["coverage_label"] = f"{cov.get('n_available')}/{cov.get('n_expected')} mappable ({cov.get('coverage_pct')}%); {cov.get('label')}"
    (out / "comparison.json").write_text(json.dumps(res.comparison, indent=2, default=str), encoding="utf-8", newline="\n")
    (out / "comparison.md").write_text(render_comparison(res.comparison, wm), encoding="utf-8", newline="\n")
    # 9. multi-snapshot extras: temporal test performance by snapshot; patient-disjoint sensitivity runs
    if design is not None:
        for fs in feature_sets:
            res.by_snapshot[fs] = performance_by_snapshot(res.runs[fs], res.metrics[fs].get("served_variant") or "uncalibrated")
            res.by_snapshot[fs].to_csv(out / f"temporal_test_by_snapshot_{fs}.csv", index=False, lineterminator="\n")
        (out / "temporal_test_by_snapshot.md").write_text(render_by_snapshot(res.by_snapshot, wm), encoding="utf-8", newline="\n")
        if patient_disjoint:
            _sensitivity_runs(res, out, mapping=mapping, spec=spec, wm=wm, sets=sets, feature_sets=feature_sets, fast=fast, template_path=template_path)
    res.summary_text = render_summary(res, mapping, wm, fast)
    (out / "SUMMARY.md").write_text(res.summary_text, encoding="utf-8", newline="\n")
    stages["training"] = "STRICT and EXTENDED fitted" + ("; patient-disjoint sensitivity fitted" if res.sensitivity_runs else "")
    _write_readiness(res, wm)


def _sensitivity_runs(res: ExploreResult, out: Path, *, mapping: WideMapping, spec: FeatureSpec, wm: str, sets: dict[str, list[str]],
                      feature_sets: tuple[str, ...], fast: bool, template_path: str | Path) -> None:
    """PATIENT-DISJOINT TEST: every snapshot of a held-out patient set forms the test partition (patient-grouped hold-out stratified by
    the patient's maximum outcome); no patient of the test partition appears in train or validation. A separate estimand from the
    temporal design (generalisation to unseen patients), fitted without the bootstrap stability / optimism analyses."""
    from falls_ml.experiment import run_experiment

    note = ("PATIENT-DISJOINT sensitivity analysis on the same rows as the temporal design: all snapshots of a random 20% of patients form the "
            "test partition, 20% the validation partition (stratified by the patient's maximum outcome); no patient is in two partitions. It "
            "estimates generalisation to unseen patients, ignores time ordering, and is reported separately from the primary temporal analysis.")
    validation = {"strategy": "patient_grouped_random", "patients_disjoint": True,
                  "patient_grouped_random": {"validation_fraction": 0.2, "test_fraction": 0.2, "stratify_outcome": True}, "limitation_note": note}
    analysis = {"stability": {"enabled": False}, "optimism": {"enabled": False}}
    metrics: dict[str, dict[str, Any]] = {}
    for fs in feature_sets:
        cfg = _write_config(template_path, out / "configs" / f"{SENSITIVITY_LABELS[fs]}.yaml", name=SENSITIVITY_LABELS[fs],
                            description=f"{wm}. {SET_TITLES[fs]}. {note}", features=sets[fs], fast=fast, spec=spec, validation=validation, analysis=analysis)
        result = run_experiment(None, res.datasets[fs], cfg, runs_dir=out / "runs_patient_disjoint")
        res.sensitivity_runs[fs] = Path(result.run_dir)
        m = result.metrics
        m.setdefault("features", {})["n_selected_raw"] = len(_selected_raw_features(Path(result.run_dir)))
        metrics[fs] = m
        splits = pd.read_csv(Path(result.run_dir) / "splits.csv")
        by = {k: set(v) for k, v in splits.groupby("split")["research_id"]}
        if (by.get("test", set()) & (by.get("train", set()) | by.get("validation", set()))):
            raise DatasetValidationError("patient-disjoint sensitivity run has patients in two partitions", [])
    res.sensitivity_metrics = metrics
    res.sensitivity_comparison = comparison(metrics, {fs: sets[fs] for fs in feature_sets})
    (out / "comparison_patient_disjoint.json").write_text(json.dumps(res.sensitivity_comparison, indent=2, default=str), encoding="utf-8", newline="\n")
    (out / "comparison_patient_disjoint.md").write_text(render_comparison(res.sensitivity_comparison, wm, title="STRICT vs EXTENDED — patient-disjoint sensitivity analysis",
                                                                           note=note), encoding="utf-8", newline="\n")
