"""``falls_ml model-report``: training visualisation + management reporting for completed runs (reads run artifacts; never writes into a run).

Outputs (one folder, default ``<first results folder>_reports``):
    MODEL_TRAINING_REPORT_<analysis>.html   one scientific dashboard per completed run
    MODEL_COMPARISON_REPORT.html            STRICT / EXTENDED / SAFE-ALL-ROWS / ablations side by side (paired only where the test rows are identical)
    MANAGEMENT_MODEL_REPORT_HE.html, MANAGEMENT_MODEL_SUMMARY_HE.md, EXECUTIVE_ONE_PAGER_HE.html   Hebrew, non-technical
    figures/*.png + *.svg, FIGURE_INDEX.md, tables/*.csv, model_report_manifest.json
Everything is aggregate; the canonical datasets and predictions are read locally and never copied into the outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.errors import DatasetValidationError, LeakageError
from falls_ml.logging_utils import get_logger
from falls_ml.modelreport import compute as C
from falls_ml.modelreport import figures as F
from falls_ml.modelreport.artifacts import Run, discover_runs, feature_label, load_labels

log = get_logger(__name__)
WATERMARK = "EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION"
PRIMARY_ORDER = ("EXTENDED", "STRICT", "SAFE_ALL_ROWS")


@dataclass
class RunReport:
    run: Run
    figures: dict[str, dict[str, str]] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


@dataclass
class ReportSet:
    out_dir: Path
    runs: list[RunReport]
    comparison: pd.DataFrame
    paired: list[dict[str, Any]]
    primary: RunReport | None
    figures: dict[str, dict[str, str]] = field(default_factory=dict)
    conclusions_he: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    eda: dict[str, Any] = field(default_factory=dict)
    synthetic: bool = False
    d00: dict[str, Any] = field(default_factory=dict)


def _fmt_ci(d: dict[str, Any] | None, digits: int = 3) -> str:
    if not d or d.get("estimate") is None:
        return "n/a"
    est = f"{d['estimate']:.{digits}f}"
    return est + (f" ({d['ci_low']:.{digits}f}–{d['ci_high']:.{digits}f})" if d.get("ci_low") is not None else "")


def build_model_reports(results_dirs: list[str | Path], *, out_dir: str | Path | None = None, eda_dir: str | Path | None = None,
                        learning_curve: str = "full", min_cell: int = 10, labels_path: str | Path | None = None, n_boot: int = 1000,
                        mapping_path: str | Path | None = None, own_results: str | Path | None = None, d00: dict[str, Any] | None = None) -> ReportSet:
    """Build every report. ``learning_curve``: full | reduced | off. ``own_results``: the results folder of the command that is building the
    reports right now (meuhedet-explore / -sensitivity / -d00 write them into a sub-folder of it); every other results folder is read-only.
    ``d00``: the d00_sensitivity.json summary of a meuhedet-d00 run; the management report then adds the D-00 comparison (reference, safe
    full-population cells, ablations, FINAL_DWH_FIXED as pending) - read only, never recomputed."""
    t0 = time.perf_counter()
    roots = [Path(p) for p in results_dirs]
    out = Path(out_dir) if out_dir else roots[0].with_name(roots[0].name + "_reports")
    own = Path(own_results).resolve() if own_results else None
    for r in roots:
        if own is not None and r.resolve() == own and own in out.resolve().parents:
            continue
        if out.resolve() == r.resolve() or r.resolve() in out.resolve().parents:
            raise DatasetValidationError(f"--out {out} must not be inside a results folder (completed runs are immutable)")
    if out.exists() and any(out.iterdir()):
        raise DatasetValidationError(f"Output directory {out} is not empty; choose a new --out")
    if learning_curve not in ("full", "reduced", "off"):
        raise ValueError("learning_curve must be full, reduced or off")
    runs = discover_runs(roots)
    labels = load_labels(labels_path) if labels_path else load_labels()
    from falls_ml.data.meuhedet_wide import DEFAULT_MAPPING, load_wide_mapping

    mapping = load_wide_mapping(mapping_path or DEFAULT_MAPPING)
    figs, tabs = out / "figures", out / "tables"
    for d in (out, figs, tabs):
        d.mkdir(parents=True, exist_ok=True)
    reports: list[RunReport] = []
    for run in runs:
        if run.key.endswith("PATIENT_DISJOINT"):
            continue
        rr = RunReport(run=run)
        _one_run(rr, roots, figs, tabs, mapping=mapping, labels=labels, learning_curve=learning_curve, min_cell=min_cell, n_boot=n_boot)
        reports.append(rr)
    primary = next((r for k in PRIMARY_ORDER for r in reports if r.run.key == k), reports[0] if reports else None)
    comp, paired = _comparison(reports, primary, n_boot=n_boot)
    rs = ReportSet(out_dir=out, runs=reports, comparison=comp, paired=paired, primary=primary, labels=labels,
                   synthetic=any(r.run.synthetic for r in reports), d00=dict(d00 or {}))
    comp.to_csv(tabs / "model_comparison.csv", index=False, lineterminator="\n")
    pd.DataFrame(paired).to_csv(tabs / "paired_differences.csv", index=False, lineterminator="\n")
    if len(comp) > 1:
        rs.figures["comparison"] = F.save(F.comparison(comp, lang="en"), figs, "comparison_metrics", WATERMARK)
    same = [r for r in reports if primary is not None and r.run.test_sha == primary.run.test_sha]
    if len(same) > 1:
        curves = [(f"{r.run.display}: AUROC {_fmt_ci(r.run.perf('test').get('auroc'))}", r.tables["roc"], F.SERIES[i]) for i, r in enumerate(same) if "roc" in r.tables]
        rs.figures["roc_paired"] = F.save(F.roc(curves, lang="en", title="ROC on the same held-out test rows (paired comparison)"), figs, "comparison_roc_paired", WATERMARK)
        prev = primary.run.perf("test").get("observed_rate") or 0.0
        pcurves = [(f"{r.run.display}: PR-AUC {_fmt_ci(r.run.perf('test').get('pr_auc'))}", r.tables["pr"], F.SERIES[i]) for i, r in enumerate(same) if "pr" in r.tables]
        rs.figures["pr_paired"] = F.save(F.pr(pcurves, prev, title="Precision-recall on the same held-out test rows"), figs, "comparison_pr_paired", WATERMARK)
    if eda_dir:
        rs.eda = _eda_links(Path(eda_dir))
    from falls_ml.modelreport import management as M
    from falls_ml.modelreport import render as R

    M.management_figures(rs, figs)
    rs.conclusions_he = M.conclusions_he(rs)
    for rr in reports:
        (out / f"MODEL_TRAINING_REPORT_{rr.run.key}.html").write_text(R.render_run(rs, rr), encoding="utf-8", newline="\n")
    (out / "MODEL_COMPARISON_REPORT.html").write_text(R.render_comparison(rs), encoding="utf-8", newline="\n")
    (out / "MANAGEMENT_MODEL_REPORT_HE.html").write_text(M.render_management_html(rs), encoding="utf-8", newline="\n")
    (out / "MANAGEMENT_MODEL_SUMMARY_HE.md").write_text(M.render_management_md(rs), encoding="utf-8", newline="\n")
    (out / "EXECUTIVE_ONE_PAGER_HE.html").write_text(M.render_one_pager(rs), encoding="utf-8", newline="\n")
    (out / "FIGURE_INDEX.md").write_text(R.figure_index(rs), encoding="utf-8", newline="\n")
    ids: set[str] = set()
    for rr in reports:
        s = rr.run.csv("splits.csv")
        if len(s):
            ids |= set(s["research_id"].astype(str))
    scan = privacy_scan(out, ids)
    rs.manifest = {"watermark": WATERMARK, "synthetic_data": rs.synthetic, "results_dirs": [r.name for r in roots],
                   "runs": [{"analysis": rr.run.key, "run_id": rr.run.metrics.get("run_id"), "test_rows_sha256": rr.run.test_sha, "problems": rr.problems,
                             "learning_curve": rr.info.get("learning_curve", {}).get("mode"), "coefficient_path_reproduces_model": rr.info.get("path", {}).get("reproduces_saved_model")}
                            for rr in reports],
                   "primary_analysis": primary.run.key if primary else None,
                   "primary_rule": "pre-declared order EXTENDED > STRICT > SAFE-ALL-ROWS; never chosen by test performance",
                   "test_set_rule": "TRAIN/VALIDATION only for coefficient paths, learning curves and the primary error analysis; held-out test figures read the "
                                    "predictions the frozen model produced when the run released the test set once; nothing here refits on test",
                   "privacy": {"min_cell": min_cell, "identifier_scan": scan}, "seconds": round(time.perf_counter() - t0, 1)}
    (out / "model_report_manifest.json").write_text(json.dumps(rs.manifest, indent=2, default=str), encoding="utf-8", newline="\n")
    if not scan["passed"]:
        raise LeakageError("identifier values found in the report outputs; they must not be shared", scan["hits"])
    return rs


def attach_reports(results_dirs: list[Path], out_dir: Path, *, own_results: Path, learning_curve: str = "full") -> dict[str, Any]:
    """Build the reports at the end of a training command. The training results are complete before this runs, so a reporting failure is
    recorded (REPORT_FAILED.txt, returned status) instead of invalidating them."""
    if learning_curve == "none":
        return {"status": "not built (--model-report off)"}
    try:
        rs = build_model_reports(results_dirs, out_dir=out_dir, learning_curve=learning_curve, own_results=own_results)
        return {"status": "built", "dir": str(out_dir), "primary_analysis": rs.manifest.get("primary_analysis"), "conclusions_he": rs.conclusions_he,
                "problems": {r.run.key: r.problems for r in rs.runs if r.problems}}
    except Exception as exc:  # noqa: BLE001 - the trained runs stay valid; the failure is written next to them and printed
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "REPORT_FAILED.txt").write_text(f"model-report failed: {type(exc).__name__}: {exc}\nrerun: falls_ml model-report --results <folder>\n",
                                                   encoding="utf-8", newline="\n")
        log.warning("model_report_failed", extra_fields={"error": f"{type(exc).__name__}: {exc}"})
        return {"status": f"FAILED: {type(exc).__name__}: {exc}", "dir": str(out_dir)}


def _eda_links(eda: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"dir": eda.name}
    m = eda / "eda_manifest.json"
    if m.exists():
        man = json.loads(m.read_text(encoding="utf-8"))
        out["stages"] = man.get("stages")
        out["input_sha256"] = (man.get("input") or {}).get("sha256")
        out["test_rows_sha256"] = (man.get("split") or {}).get("test_rows_sha256")
    return out


def _d00_note(run: Run, mapping: Any, feature: str) -> str:
    try:
        f = mapping.get(feature)
    except Exception:  # noqa: BLE001 - a non-Meuhedet feature
        return "n/a"
    viol = run.build.get("timing_violations_by_column") or {}
    if f.record_date_column:
        n = viol.get(f.record_date_column)
        return f"yes: source {f.record_date_column} has index-day records" + (f" ({n} rows excluded in this cohort)" if n else "")
    if f.same_day_evidence_column:
        n = viol.get(f.same_day_evidence_column)
        return f"partial evidence only ({f.same_day_evidence_column})" + (f"; {n} rows excluded" if n else "")
    return "not verifiable (no record date)"


def _one_run(rr: RunReport, roots: list[Path], figs: Path, tabs: Path, *, mapping: Any, labels: dict[str, Any], learning_curve: str,
             min_cell: int, n_boot: int) -> None:
    from falls_ml.meuhedet_sensitivity import convergence_audit

    run, key = rr.run, rr.run.key
    lab = {c: feature_label(labels, c, "en") for c in run.features}
    b = run.build
    steps = [("rows in the extract", int(b.get("n_rows_input") or 0), "")]
    n = steps[0][1] - int(b.get("n_rows_other_index_dates") or 0)
    steps.append(("rows on the index date", n, ""))
    n -= int(b.get("n_rows_ineligible") or 0)
    steps.append(("eligible (Is_Eligible_Cohort = 1)", n, ""))
    n -= int(b.get("n_rows_label_null") or 0)
    steps.append(("usable 180-day label", n, ""))
    n -= int(b.get("n_rows_timing_violation") or 0)
    steps.append((f"after D-00 exclusion (scope {b.get('timing_scope', 'cohort')})", n, f"-{int(b.get('n_rows_timing_violation') or 0)} rows"))
    steps.append(("modelling cohort", int(b.get("n_rows_final") or 0), f"{int(b.get('n_events') or 0)} events"))
    rr.info["funnel"] = steps
    rr.figures["funnel"] = F.save(F.funnel(steps, title=f"{run.display}: cohort", lang="en"), figs, f"{key}_cohort_funnel", WATERMARK)
    parts = (run.metrics.get("cohort") or {}).get("by_split") or {}
    if parts:
        rr.figures["partitions"] = F.save(F.partitions(parts, lang="en"), figs, f"{key}_partitions", WATERMARK)
    audit = convergence_audit(run.run_dir, run.metrics)
    rr.info["audit"] = audit
    cv = run.csv("cv_results.csv")
    if len(cv) and cv["cv_mean_deviance"].notna().any():
        rr.figures["cv"] = F.save(F.cv_curve(cv, audit["final_fit"]["grid"]), figs, f"{key}_lasso_cv_curve", WATERMARK)
    frames = None
    try:
        frames = C.load_frames(run, roots)
    except (DatasetValidationError, LeakageError, FileNotFoundError) as exc:
        rr.problems.append(f"dataset not available locally: {exc} - coefficient path, learning curve, subgroups by predictor and error analysis skipped")
    if frames is not None and len(cv):
        try:
            path, pinfo = C.coefficient_path(run, frames)
            rr.info["path"] = pinfo
            path.to_csv(tabs / f"{key}_coefficient_path.csv", index=False, lineterminator="\n")
            dlab = {c: feature_label(labels, c, "en") for c in path["design_column"].unique()}
            rr.figures["path"] = F.save(F.coef_paths(path, pinfo["lambda_star"], dlab), figs, f"{key}_lasso_coefficient_paths", WATERMARK)
        except Exception as exc:  # noqa: BLE001 - reported in the report, never silent
            rr.problems.append(f"coefficient path not computed: {type(exc).__name__}: {exc}")
        fp = run.csv("fp_selection.csv")
        effects = {}
        forms = dict(audit["replicates"]["fp_form_instability"])
        for _, r in fp.iterrows() if len(fp) else []:
            v = str(r["variable"])
            forms.setdefault(v, {})["powers"] = r.get("powers")
            forms[v].setdefault("final_form", r.get("powers"))
            try:
                eff = C.fp_effect(run, frames, v)
                if eff is not None:
                    effects[v] = eff
            except Exception as exc:  # noqa: BLE001
                rr.problems.append(f"FP shape of {v} not computed: {exc}")
        rr.info["fp_forms"] = forms
        fig = F.fp_panel(effects, forms, lab)
        if fig is not None:
            rr.figures["fp"] = F.save(fig, figs, f"{key}_fp_shapes", WATERMARK)
        if learning_curve != "off":
            lc, linfo = C.learning_curve(run, frames, mode=learning_curve)
            rr.tables["learning_curve"], rr.info["learning_curve"] = lc, linfo
            lc.to_csv(tabs / f"{key}_learning_curve.csv", index=False, lineterminator="\n")
            if (lc["status"] == "fitted").sum() >= 2:
                rr.figures["learning"] = F.save(F.learning(lc, linfo), figs, f"{key}_learning_curve", WATERMARK)
    # ---- held-out test (predictions of the frozen model)
    pt = run.predictions("test")
    served_col = "risk_served" if "risk_served" in pt.columns else "risk_uncalibrated"
    if len(pt):
        y, p = pt["outcome"].to_numpy(dtype=np.int64), pt[served_col].to_numpy(dtype=float)
        rr.tables["roc"], rr.tables["pr"] = C.roc_points(y, p), C.pr_points(y, p)
        perf = run.perf("test")
        rr.figures["roc"] = F.save(F.roc([(f"{run.display}: AUROC {_fmt_ci(perf.get('auroc'))}", rr.tables["roc"], F.BLUE)], lang="en",
                                         title=f"ROC, held-out test (served: {run.served})"), figs, f"{key}_roc", WATERMARK)
        rr.figures["pr"] = F.save(F.pr([(f"{run.display}: PR-AUC {_fmt_ci(perf.get('pr_auc'))}", rr.tables["pr"], F.ORANGE)], float(y.mean()),
                                       title="Precision-recall, held-out test"), figs, f"{key}_precision_recall", WATERMARK)
        hist = C.risk_histogram(y, p, min_cell)
        rr.figures["risk"] = F.save(F.risk_distribution(hist), figs, f"{key}_risk_distributions", WATERMARK)
        cal = run.csv("calibration.csv")
        cal = cal[cal["split"] == "test"] if len(cal) else cal
        stats = {v: (f"slope {_fmt_ci(run.perf('test', v).get('calibration_slope'), 2)}, CITL {_fmt_ci(run.perf('test', v).get('citl'), 2)}, "
                     f"O:E {_fmt_ci(run.perf('test', v).get('oe_ratio'), 2)}, Brier {_fmt_ci(run.perf('test', v).get('brier'), 4)}")
                 for v in ("uncalibrated", "recalibrated") if run.perf("test", v)}
        rr.info["calibration_stats"] = stats
        if len(cal):
            rr.figures["calibration"] = F.save(F.calibration(cal, stats, hist), figs, f"{key}_calibration", WATERMARK)
        tt = C.threshold_table(y, p, min_cell)
        rr.tables["thresholds"] = tt
        tt.to_csv(tabs / f"{key}_thresholds_test.csv", index=False, lineterminator="\n")
        rr.figures["thresholds"] = F.save(F.thresholds_plot(tt), figs, f"{key}_thresholds", WATERMARK)
        lift = C.lift_table(y, p, min_cell=min_cell, n_boot=n_boot)
        gains = C.gains_curve(y, p)
        rr.tables["lift"], rr.tables["gains"] = lift, gains
        lift.to_csv(tabs / f"{key}_lift_test.csv", index=False, lineterminator="\n")
        rr.figures["gains"] = F.save(F.gains_lift(gains, lift, lang="en"), figs, f"{key}_gains_lift", WATERMARK)
        dec = C.risk_deciles(y, p, min_cell)
        rr.tables["deciles"] = dec
        dec.to_csv(tabs / f"{key}_risk_deciles_test.csv", index=False, lineterminator="\n")
        rr.figures["deciles"] = F.save(F.deciles(dec, lang="en"), figs, f"{key}_risk_deciles", WATERMARK)
        if frames is not None:
            test_rows = _test_rows(run, roots, frames, pt)
            if test_rows is not None:
                groups = _subgroups(test_rows)
                sg = C.subgroup_performance(test_rows, y, p, groups, min_cell)
                rr.tables["subgroups"] = sg
                sg.to_csv(tabs / f"{key}_subgroups_test.csv", index=False, lineterminator="\n")
                fig = F.subgroup_plot(sg)
                if fig is not None:
                    rr.figures["subgroups"] = F.save(fig, figs, f"{key}_subgroups", WATERMARK)
                ea_t = C.error_analysis(test_rows, y, p, run.features, frames.spec, min_cell)
                rr.tables["error_test"] = ea_t
    pv = run.predictions("validation")
    if len(pv) and frames is not None:
        v = frames.validation.copy()
        key_v = C._keys(v, frames.spec)
        keyp = pv["research_id"].astype(str) + "|" + pd.to_datetime(pv["index_date"]).dt.strftime("%Y-%m-%d")
        vv = v.set_index(key_v).loc[keyp].reset_index(drop=True)
        yv = pv["outcome"].to_numpy(dtype=np.int64)
        pvv = pv["risk_uncalibrated"].to_numpy(dtype=float)
        rr.tables["error_validation"] = C.error_analysis(vv, yv, pvv, run.features, frames.spec, min_cell)
    # ---- features
    coef = run.csv("coefficients.csv")
    if len(coef):
        if "relative_to_reference" in coef.columns:
            coef = coef[~coef["relative_to_reference"].astype(bool)]
        sel = coef[coef["selected"].astype(bool) & (coef["design_column"] != "_intercept")].copy()
        sel = sel.reindex(sel["standardized_coefficient"].abs().sort_values(ascending=False).index)
        if len(sel):
            names = [feature_label(labels, c, "en") + f"  [{c}]" for c in sel["design_column"]]
            vals = sel["standardized_coefficient"].astype(float).tolist()
            rr.figures["coefficients"] = F.save(F.hbars(names, vals, title="LASSO coefficients at lambda* (standardised: per 1 SD of the design column)",
                                                        xlabel="coefficient on the log-odds scale (orange: higher predicted risk; blue: lower)", lang="en",
                                                        colors=[F.ORANGE if v > 0 else F.BLUE for v in vals], fmt="{:+.3f}"), figs, f"{key}_coefficients", WATERMARK)
    stab = run.csv("feature_stability.csv")
    if len(stab):
        s = stab.groupby("raw_feature", as_index=False)["raw_feature_selection_frequency"].max().sort_values("raw_feature_selection_frequency", ascending=False)
        rr.tables["stability"] = s
        rr.figures["stability"] = F.save(F.hbars([feature_label(labels, f, "en") for f in s["raw_feature"]], (100 * s["raw_feature_selection_frequency"]).tolist(),
                                                 title=f"Bootstrap selection stability ({int(stab['n_bootstrap'].max())} refits)",
                                                 xlabel="% of bootstrap refits that kept the predictor", lang="en", ref=80.0, fmt="{:.0f}%"), figs,
                                         f"{key}_selection_stability", WATERMARK)
    imp = run.csv("feature_importance.csv")
    if len(imp) and imp["permutation_importance_mean"].notna().any():
        d = imp.dropna(subset=["permutation_importance_mean"]).sort_values("permutation_importance_mean", ascending=False)
        rr.figures["permutation"] = F.save(F.hbars([feature_label(labels, f, "en") for f in d["feature"]], d["permutation_importance_mean"].astype(float).tolist(),
                                                   title="Permutation importance on VALIDATION (drop in AUROC when the predictor is shuffled; not causal)",
                                                   xlabel="mean decrease in AUROC (error bar: SD over repeats)", lang="en",
                                                   err=d["permutation_importance_std"].fillna(0).astype(float).tolist(), fmt="{:.4f}"), figs, f"{key}_permutation_importance", WATERMARK)
    rr.tables["feature_dictionary"] = _feature_dictionary(run, mapping, labels, coef, stab, imp)
    rr.tables["feature_dictionary"].to_csv(tabs / f"{key}_feature_dictionary.csv", index=False, lineterminator="\n")


def _test_rows(run: Run, roots: list[Path], frames: C.Frames, pt: pd.DataFrame) -> pd.DataFrame | None:
    """Canonical features of the test rows, aligned to the stored test predictions (read only after the run released the test set)."""
    from falls_ml.data.dataset import ModelingDataset

    ds_dir = run.dataset_dir(roots)
    if ds_dir is None:
        return None
    frame = ModelingDataset.load(ds_dir, _full_spec(run), features=frames.frame_features or None).frame
    k = C._keys(frame, frames.spec)
    kp = pt["research_id"].astype(str) + "|" + pd.to_datetime(pt["index_date"]).dt.strftime("%Y-%m-%d")
    idx = pd.Index(k)
    if not kp.isin(idx).all():
        return None
    return frame.set_index(k).loc[kp].reset_index(drop=True)


def _full_spec(run: Run) -> Any:
    from falls_ml.config import load_experiment_config
    from falls_ml.features.spec import load_feature_spec

    cfg = load_experiment_config(run.run_dir / "config.yaml")
    return load_feature_spec(cfg.dataset.feature_spec, cfg.dataset.feature_spec_extensions, anchor=run.run_dir)


def _subgroups(df: pd.DataFrame) -> dict[str, pd.Series]:
    g: dict[str, pd.Series] = {}
    if "age_years" in df.columns:
        g["age group"] = pd.cut(pd.to_numeric(df["age_years"], errors="coerce"), [-np.inf, 75, 85, np.inf], right=False, labels=["<75", "75-84", "85+"]).astype("string")
    if "sex" in df.columns:
        g["sex"] = df["sex"].astype("string")
    for f, name in (("falls", "prior fall since 2022"), ("dementia", "dementia"), ("housebound", "housebound"), ("diabetes_mellitus", "diabetes"),
                    ("chronic_kidney_disease", "chronic renal failure"), ("mobility_problems", "difficulty walking")):
        if f in df.columns:
            g[name] = pd.to_numeric(df[f], errors="coerce").map({1: "yes", 0: "no"}).astype("string")
    if "polypharmacy_count_120d" in df.columns:
        g["medications"] = pd.cut(pd.to_numeric(df["polypharmacy_count_120d"], errors="coerce"), [-np.inf, 5, 10, np.inf], right=False,
                                  labels=["0-4", "5-9", "10+"]).astype("string")
    return g


def _feature_dictionary(run: Run, mapping: Any, labels: dict[str, Any], coef: pd.DataFrame, stab: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
    sets = mapping.feature_sets()
    rows = []
    for f in run.features:
        try:
            fm = mapping.get(f)
            src, qual = "; ".join(fm.source_columns), fm.quality
        except Exception:  # noqa: BLE001
            src, qual = "", ""
        c = coef[coef["feature"] == f] if len(coef) else pd.DataFrame()
        c = c[c["design_column"] != "_intercept"] if len(c) else c
        s = stab[stab["raw_feature"] == f] if len(stab) else pd.DataFrame()
        i = imp[imp["feature"] == f] if len(imp) else pd.DataFrame()
        rows.append({"canonical_feature": f, "meuhedet_source_field": src, "name_he": feature_label(labels, f, "he"), "name_en": feature_label(labels, f, "en"),
                     "domain_he": (labels.get("features") or {}).get(f, {}).get("domain_he", ""), "strict": f in sets["strict"], "extended": f in sets["extended"],
                     "mapping_confidence": qual, "d00_sensitivity": _d00_note(run, mapping, f),
                     "standardized_coefficient": "; ".join(f"{float(v):+.3f}" for v in c["standardized_coefficient"]) if len(c) else "n/a",
                     "selected_at_lambda_star": bool(c["selected"].astype(bool).any()) if len(c) else False,
                     "bootstrap_selection_pct": round(100 * float(s["raw_feature_selection_frequency"].max()), 1) if len(s) else None,
                     "permutation_importance": round(float(i["permutation_importance_mean"].iloc[0]), 5) if len(i) and pd.notna(i["permutation_importance_mean"].iloc[0]) else None})
    return pd.DataFrame(rows)


def _comparison(reports: list[RunReport], primary: RunReport | None, *, n_boot: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows, paired = [], []
    for rr in reports:
        r = rr.run
        perf = r.perf("test")
        coh = r.metrics.get("cohort") or {}
        row: dict[str, Any] = {"label": r.display, "analysis": r.key, "rows": coh.get("n_rows"), "events": coh.get("n_events"),
                               "prevalence": coh.get("prevalence"), "n_predictors": len(r.features), "test_rows": perf.get("n"), "test_events": perf.get("n_events"),
                               "served_variant": r.served, "test_rows_sha256": r.test_sha[:16],
                               "paired_with_reference": bool(primary is not None and r.test_sha == primary.run.test_sha)}
        for m in ("auroc", "pr_auc", "brier", "calibration_slope", "oe_ratio", "citl"):
            d = perf.get(m) or {}
            row[m], row[f"{m}_ci_low"], row[f"{m}_ci_high"] = d.get("estimate"), d.get("ci_low"), d.get("ci_high")
        row["comparison_note"] = ("reference" if primary is not None and rr is primary else
                                  ("paired: identical test rows" if row["paired_with_reference"] else "NOT A DIRECT PAIRED MODEL COMPARISON: different population / test rows"))
        rows.append(row)
        if primary is not None and rr is not primary and row["paired_with_reference"]:
            a, b = primary.run.predictions("test"), r.predictions("test")
            if len(a) and len(b):
                ka = a["research_id"].astype(str) + "|" + a["index_date"].astype(str)
                kb = b["research_id"].astype(str) + "|" + b["index_date"].astype(str)
                m = pd.DataFrame({"k": ka, "y": a["outcome"], "pa": a["risk_served"]}).merge(pd.DataFrame({"k": kb, "pb": b["risk_served"]}), on="k")
                d = C.paired_differences(m["y"].to_numpy(dtype=np.int64), m["pa"].to_numpy(dtype=float), m["pb"].to_numpy(dtype=float), n_boot=n_boot)
                paired.append({"reference": primary.run.display, "compared": r.display, "difference": f"{r.display} minus {primary.run.display}",
                               **{f"{k}_{s}": v[s] for k, v in d.items() if isinstance(v, dict) for s in ("estimate", "ci_low", "ci_high")},
                               "n_rows": d["n_rows"], "n_boot": d["n_boot"]})
    return pd.DataFrame(rows), paired


_DATA_URI = re.compile(r"data:image/(?:png|svg\+xml);base64,[A-Za-z0-9+/=]+")


def privacy_scan(out: Path, identifiers: set[str]) -> dict[str, Any]:
    hits, n = [], 0
    ids = {i for i in identifiers if len(i) >= 6}
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".csv", ".md", ".json", ".html", ".svg", ".txt"):
            continue
        n += 1
        text = _DATA_URI.sub(" ", p.read_text(encoding="utf-8", errors="ignore"))
        found = set(re.findall(r"[A-Za-z0-9_]+", text)) & ids
        if found:
            hits.append(f"{p.relative_to(out).as_posix()}: {len(found)} identifier value(s)")
    return {"passed": not hits, "files_scanned": n, "identifier_values_checked": len(ids), "hits": hits}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_ = math
