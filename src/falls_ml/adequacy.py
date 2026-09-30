"""Pre-training adequacy gate: can the configured fitting procedure be run on this split at all?

The published learning process (FP selection by unpenalised logistic regression, LASSO logistic with 10-fold CV, logistic
recalibration on the validation rows, bootstrap CIs and permutation importance) breaks - or returns numbers that mean nothing - when a
partition has too few events. On a real ~118-row / 1-event extract it failed inside the recalibration GLM; that failure was correct,
but late. This gate runs before any preprocessing or fitting and stops with ``INSUFFICIENT EVENTS FOR EXPLORATORY MODEL FIT`` and the
exact counts.

The thresholds are **pipeline safeguards derived from the procedure**, not universal medical-statistics laws:

- every partition must hold both outcome classes (nothing can be fitted, recalibrated or scored otherwise);
- training events (and non-events) >= max(MIN_TRAIN_EVENTS, EPP_STOP x the largest possible number of design columns): below this the
  unpenalised FP-selection logit is quasi-separated and the 10 CV folds cannot all hold both classes - the fit is expected to fail;
  between EPP_STOP and EPP_WARN events per design column the fit runs but the estimates are unstable - a warning, not a stop;
- validation events and non-events >= MIN_PARTITION_EVENTS: the recalibration GLM ``y ~ LP`` and the AUROC-based permutation
  importance need a handful of both classes;
- test events and non-events >= MIN_PARTITION_EVENTS: the same floor the framework uses before it reports a subgroup or cluster
  metric (``INSUFFICIENT EVENTS`` otherwise);
- constant predictors and extremely sparse binary predictors are reported as warnings (the LASSO omits constants Stata-style; a
  predictor with fewer than SPARSE_MIN_POSITIVES training positives cannot be estimated stably) - they stop the run only when every
  predictor is constant.

For reference only, the Riley et al. (2019) minimum development sample size is reported at the observed prevalence with the
conservative assumption R2_CS = 0.15 x max R2_CS (Riley et al. 2020, BMJ); it is information for the report, not a gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.config import ExperimentConfig
from falls_ml.features.spec import FeatureSpec
from falls_ml.splitting import SplitPlan

MIN_PARTITION_EVENTS = 10      # events AND non-events required in validation and in test
MIN_TRAIN_EVENTS = 20          # absolute floor for training events
EPP_STOP = 2                   # training events per (maximum) design column below which the fit is expected to fail (hard stop)
EPP_WARN = 5                   # ... below which the estimates are unstable (warning)
SPARSE_MIN_POSITIVES = 5       # binary predictor with fewer training positives -> warning
SPARSE_PREVALENCE = 0.005      # binary predictor below this training prevalence -> warning
INSUFFICIENT = "INSUFFICIENT EVENTS FOR EXPLORATORY MODEL FIT"
READY = "READY"


@dataclass
class AdequacyReport:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    partitions: dict[str, dict[str, Any]] = field(default_factory=dict)
    predictors: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    riley_reference: dict[str, Any] | None = None
    total: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.verdict == READY

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "reasons": list(self.reasons), "warnings": list(self.warnings), "partitions": dict(self.partitions),
                "predictors": dict(self.predictors), "thresholds": dict(self.thresholds), "riley_reference": self.riley_reference, "total": dict(self.total)}


def max_design_columns(spec: FeatureSpec, config: ExperimentConfig) -> int:
    """Upper bound on the model-matrix columns: binary 1; continuous 1, or 2 when fractional-polynomial selection may pick FP2;
    categorical one column per level (all-levels coding)."""
    fp_vars = set(config.preprocessing.fractional_polynomial.variables or ())
    fp_max = 2 if config.preprocessing.fractional_polynomial.max_degree >= 2 else 1
    n = 0
    for f in spec.features:
        if f.dtype == "binary":
            n += 1
        elif f.is_categorical:
            n += max(len(f.levels), 1) + (1 if f.nullable else 0)
        elif f.name in fp_vars:
            n += fp_max
        else:
            n += 1 + (1 if f.dtype == "float_nullable" else 0)
    return n


def assess_adequacy(frame: pd.DataFrame, spec: FeatureSpec, plan: SplitPlan, config: ExperimentConfig) -> AdequacyReport:
    """Count rows / events / classes per partition and predictor support on the training rows; decide READY or INSUFFICIENT."""
    y_col, id_col = spec.outcome.name, spec.identifier_columns[0]
    n_params = max_design_columns(spec, config)
    train_floor = max(MIN_TRAIN_EVENTS, EPP_STOP * n_params)
    train_warn = EPP_WARN * n_params
    rep = AdequacyReport(verdict=READY, thresholds={
        "min_partition_events": MIN_PARTITION_EVENTS, "min_train_events": MIN_TRAIN_EVENTS, "epp_stop": EPP_STOP, "epp_warn": EPP_WARN,
        "max_design_columns": n_params, "train_events_required": train_floor, "train_events_unstable_below": train_warn, "cv_folds": config.validation.cv_folds,
        "sparse_min_positives": SPARSE_MIN_POSITIVES, "sparse_prevalence": SPARSE_PREVALENCE,
        "rule": "pipeline safeguards derived from the fitting procedure (FP-selection logit, 10-fold CV LASSO, validation recalibration, "
                "bootstrap CIs); not universal medical-statistics laws"})
    y_all = frame[y_col].to_numpy(dtype=np.int64)
    rep.total = {"n_rows": int(len(frame)), "n_patients": int(frame[id_col].nunique()), "n_events": int(y_all.sum()),
                 "prevalence": float(y_all.mean()) if len(y_all) else None, "n_predictors": len(spec.features)}
    for name, idx in plan.indices().items():
        part = frame.iloc[idx]
        y = part[y_col].to_numpy(dtype=np.int64)
        n, ev = int(len(y)), int(y.sum())
        classes = sorted(int(v) for v in np.unique(y)) if n else []
        entry = {"n_rows": n, "n_patients": int(part[id_col].nunique()), "n_events": ev, "n_non_events": n - ev,
                 "prevalence": (ev / n) if n else None, "classes_present": classes}
        rep.partitions[name] = entry
        if len(classes) < 2:
            rep.reasons.append(f"{name}: only outcome class(es) {classes} present ({n} rows, {ev} events)")
            continue
        if name == "train":
            if ev < train_floor:
                rep.reasons.append(f"train: {ev} events < {train_floor} required (max({MIN_TRAIN_EVENTS}, {EPP_STOP} x {n_params} design columns))")
            elif ev < train_warn:
                rep.warnings.append(f"train: {ev} events for up to {n_params} design columns ({ev / n_params:.1f} events per column, below {EPP_WARN}): "
                                    "the fit runs but coefficient estimates and selection are unstable - exploratory only")
            if n - ev < train_floor:
                rep.reasons.append(f"train: {n - ev} non-events < {train_floor} required")
            if ev < 2 * config.validation.cv_folds:
                rep.reasons.append(f"train: {ev} events cannot populate {config.validation.cv_folds} CV folds with both classes reliably")
        else:
            if ev < MIN_PARTITION_EVENTS:
                rep.reasons.append(f"{name}: {ev} events < {MIN_PARTITION_EVENTS}")
            if n - ev < MIN_PARTITION_EVENTS:
                rep.reasons.append(f"{name}: {n - ev} non-events < {MIN_PARTITION_EVENTS}")
    # predictor support on the training rows
    train = frame.iloc[plan.train_idx]
    constant, sparse, support = [], [], {}
    for f in spec.features:
        s = train[f.name]
        n_unique = int(s.dropna().nunique())
        entry: dict[str, Any] = {"dtype": f.dtype, "n_unique_train": n_unique, "missing_rate_train": float(s.isna().mean()) if len(s) else None}
        if n_unique <= 1:
            constant.append(f.name)
            entry["constant"] = True
        if f.dtype == "binary" and len(s):
            pos = int((s.fillna(0).astype("int64") == 1).sum())
            entry["n_positive_train"] = pos
            entry["prevalence_train"] = pos / len(s)
            if n_unique > 1 and (pos < SPARSE_MIN_POSITIVES or pos / len(s) < SPARSE_PREVALENCE or (len(s) - pos) < SPARSE_MIN_POSITIVES):
                sparse.append(f.name)
                entry["sparse"] = True
        support[f.name] = entry
    rep.predictors = {"n_predictors": len(spec.features), "constant_in_train": constant, "extremely_sparse_binary_in_train": sparse, "support": support}
    if constant:
        rep.warnings.append(f"constant predictors in the training rows (omitted Stata-style by the fit, coefficient 0): {constant}")
    if sparse:
        rep.warnings.append(f"extremely sparse binary predictors in the training rows (< {SPARSE_MIN_POSITIVES} positives or < {SPARSE_PREVALENCE:.1%}): "
                            f"{sparse} - estimates for them are unstable; interpret with care")
    if spec.features and len(constant) == len(spec.features):
        rep.reasons.append("every predictor is constant in the training rows")
    # Riley 2019 reference (information, not a gate)
    prev = rep.total["prevalence"]
    if prev and 0 < prev < 1:
        try:
            from falls_ml.evaluation.sample_size import max_r2_cox_snell, riley2019_binary_development
            r2 = 0.15 * max_r2_cox_snell(prev)
            r = riley2019_binary_development(prev, n_params, r2_cox_snell=r2)
            rep.riley_reference = {"assumption": "R2_CS = 0.15 x max R2_CS (Riley 2020 conservative default); parameters = max design columns",
                                   "n_parameters": n_params, "prevalence": prev, "n_required": r["n_required"], "events_required": r["events_required"],
                                   "epp_at_n_required": r["epp"], "train_rows": rep.partitions.get("train", {}).get("n_rows"),
                                   "train_events": rep.partitions.get("train", {}).get("n_events"),
                                   "train_meets_reference": (rep.partitions.get("train", {}).get("n_rows", 0) >= r["n_required"]),
                                   "note": "reference only - reported, not enforced (exploratory run)"}
        except ValueError as exc:   # degenerate prevalence / parameters
            rep.riley_reference = {"note": f"not computed: {exc}"}
    if rep.reasons:
        rep.verdict = INSUFFICIENT
    return rep


def render_adequacy_markdown(rep: AdequacyReport, title: str = "Pre-training adequacy gate") -> str:
    lines = [f"# {title}", "", f"**Verdict: {rep.verdict}**", ""]
    if rep.reasons:
        lines += ["Reasons:"] + [f"- {r}" for r in rep.reasons] + [""]
    t = rep.total
    lines += [f"Total usable rows {t.get('n_rows')}; patients {t.get('n_patients')}; events {t.get('n_events')}; prevalence "
              f"{(t.get('prevalence') or 0):.4f}; predictors {t.get('n_predictors')}.", "",
              "| partition | rows | patients | events | non-events | prevalence | classes |", "|---|---|---|---|---|---|---|"]
    for name, e in rep.partitions.items():
        lines.append(f"| {name} | {e['n_rows']} | {e['n_patients']} | {e['n_events']} | {e['n_non_events']} | {(e['prevalence'] or 0):.4f} | {e['classes_present']} |")
    p = rep.predictors
    lines += ["", f"Constant predictors (train): {p.get('constant_in_train')}", f"Extremely sparse binary predictors (train): {p.get('extremely_sparse_binary_in_train')}",
              "", f"Thresholds (pipeline safeguards): {rep.thresholds}", ""]
    if rep.riley_reference:
        lines += [f"Riley 2019 reference: {rep.riley_reference}", ""]
    if rep.warnings:
        lines += ["Warnings:"] + [f"- {w}" for w in rep.warnings] + [""]
    return "\n".join(lines) + "\n"


def write_adequacy(rep: AdequacyReport, out_dir: str | Path, stem: str = "adequacy") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / f"{stem}.json", out / f"{stem}.md"
    j.write_text(json.dumps(rep.to_dict(), indent=2, default=str), encoding="utf-8", newline="\n")
    m.write_text(render_adequacy_markdown(rep), encoding="utf-8", newline="\n")
    return j, m
