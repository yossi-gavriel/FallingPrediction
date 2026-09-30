"""Technical (English) HTML: one MODEL_TRAINING_REPORT per run, the MODEL_COMPARISON_REPORT and FIGURE_INDEX.md. Self-contained pages
(inline CSS, embedded PNGs); canonical names are always shown."""

from __future__ import annotations

import base64
import html
import math
from typing import Any

import pandas as pd

SYNTHETIC = "SYNTHETIC DATA — SOFTWARE TEST OUTPUT, NOT SCIENTIFIC RESULTS"
CSS = """
body{margin:0;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:#fcfcfb;color:#0b0b0b;line-height:1.5;font-size:14px}
.banner{background:#b00020;color:#fff;font-weight:700;text-align:center;padding:10px;font-size:15px;position:sticky;top:0;z-index:3}
.synthetic{background:#6b21a8;color:#fff;font-weight:700;text-align:center;padding:6px}
main{max-width:1250px;margin:0 auto;padding:18px 28px 80px}
h1{font-size:22px}h2{font-size:18px;border-bottom:2px solid #e4e3df;padding-bottom:4px;margin-top:38px}h3{font-size:15px}
.basis{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;margin-left:8px;vertical-align:middle;background:#e8f0fb;color:#104281}
.basis.test{background:#fde8e1;color:#8a2c0b}.basis.train{background:#e6f5ee;color:#0b5d3b}
.note{background:#f3f2ee;border-left:4px solid #b9b8b3;padding:8px 12px;margin:10px 0;font-size:13px}
.warn{background:#fff3cd;border-left:4px solid #d39e00;padding:8px 12px;margin:10px 0;font-weight:600;font-size:13px}
.tw{overflow:auto;max-height:480px;border:1px solid #e4e3df;border-radius:4px;margin:6px 0 14px;background:#fff}
table{border-collapse:collapse;font-size:12px;width:max-content;min-width:100%}th,td{border-bottom:1px solid #e4e3df;padding:3px 8px;text-align:left;vertical-align:top;max-width:460px}
th{position:sticky;top:0;background:#f0efec}
figure{margin:10px 0 20px}figure img{max-width:100%;border:1px solid #e4e3df;border-radius:4px;background:#fff}figcaption{font-size:12px;color:#52514e}
nav.toc a{margin-right:12px;font-size:13px;color:#2a78d6}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin:12px 0}
.card{background:#fff;border:1px solid #e4e3df;border-radius:6px;padding:10px}.card .v{font-size:18px;font-weight:700}.card .k{font-size:12px;color:#52514e}
"""


def esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.4g}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def table(df: pd.DataFrame | None, cols: list[str] | None = None) -> str:
    if df is None or not len(df):
        return "<p class='note'>Not available for this run.</p>"
    d = df[[c for c in (cols or list(df.columns)) if c in df.columns]]
    head = "".join(f"<th>{esc(c)}</th>" for c in d.columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(fmt(v))}</td>" for v in r) + "</tr>" for r in d.itertuples(index=False, name=None))
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def img(rs: Any, fig: dict[str, str] | None, caption: str) -> str:
    if not fig:
        return ""
    p = rs.out_dir / "figures" / fig["png"]
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return (f"<figure><img alt='{esc(caption)}' src='data:image/png;base64,{data}'><figcaption>{esc(caption)} — figures/{esc(fig['png'])} "
            f"(+ .svg)</figcaption></figure>")


def badge(kind: str) -> str:
    cls = {"TRAIN": "train", "TRAIN / VALIDATION": "train", "HELD-OUT TEST (model frozen)": "test"}.get(kind, "")
    return f"<span class='basis {cls}'>{esc(kind)}</span>"


def page(title: str, body: str, synthetic: bool, watermark: str) -> str:
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><div class='banner'>{esc(watermark)}</div>"
            + (f"<div class='synthetic'>{SYNTHETIC}</div>" if synthetic else "") + f"<main>{body}</main></body></html>")


def _ci(d: dict[str, Any] | None, digits: int = 3) -> str:
    if not d or d.get("estimate") is None:
        return "n/a"
    return f"{d['estimate']:.{digits}f}" + (f" (95% CI {d['ci_low']:.{digits}f}–{d['ci_high']:.{digits}f})" if d.get("ci_low") is not None else "")


def render_run(rs: Any, rr: Any) -> str:
    from falls_ml.modelreport.runner import WATERMARK

    run, info, t, f = rr.run, rr.info, rr.tables, rr.figures
    perf = run.perf("test")
    grid = info.get("audit", {}).get("final_fit", {}).get("grid", {})
    b: list[str] = [f"<h1>Model training report — {esc(run.display)}</h1>",
                    f"<p class='note'>Run <code>{esc(run.run_dir.name)}</code>; experiment {esc(run.name)}; served variant {esc(run.served)}; "
                    f"test-partition sha256 {esc(run.test_sha[:16])}…; {len(run.features)} predictors: {esc(', '.join(run.features))}.</p>",
                    "<nav class='toc'>" + "".join(f"<a href='#{a}'>{esc(x)}</a>" for a, x in (("cohort", "Cohort"), ("lasso", "LASSO fitting"), ("learning", "Learning curve"),
                    ("disc", "Discrimination"), ("cal", "Calibration"), ("use", "Operational usefulness"), ("feat", "Features"), ("sub", "Subgroups"),
                    ("err", "Error analysis"), ("unc", "Uncertainty"), ("lim", "Limitations"))) + "</nav>"]
    b.append("<div class='cards'>" + "".join(f"<div class='card'><div class='v'>{esc(v)}</div><div class='k'>{esc(k)}</div></div>" for k, v in (
        ("AUROC (test)", _ci(perf.get("auroc"))), ("PR-AUC (test)", _ci(perf.get("pr_auc"))), ("Brier (test)", _ci(perf.get("brier"), 4)),
        ("calibration slope (test)", _ci(perf.get("calibration_slope"), 2)), ("O:E (test)", _ci(perf.get("oe_ratio"), 2)),
        ("test rows / events", f"{perf.get('n')} / {perf.get('n_events')}"))) + "</div>")
    b.append("<div class='note'>Test-set rule: coefficient paths, FP shapes and the learning curve use TRAIN (and VALIDATION) only. Every HELD-OUT TEST figure "
             "reads the predictions the frozen model produced when the run released the test set once; nothing here refits or retunes on test.</div>")
    # A cohort
    b.append(f"<h2 id='cohort'>A. Cohort and split</h2>")
    b.append(img(rs, f.get("funnel"), "Cohort funnel from the build report (rows removed at each step)"))
    b.append(img(rs, f.get("partitions"), "Train / validation / test sizes, events and prevalence"))
    b.append(table(pd.DataFrame(info.get("funnel", []), columns=["step", "rows", "note"])))
    # B lasso
    au = info.get("audit", {})
    b.append(f"<h2 id='lasso'>B. How the LASSO was fitted {badge('TRAIN')}</h2>")
    b.append(f"<p>lambda* = {esc(fmt(grid.get('lambda_star')))} (grid index {esc(grid.get('lambda_star_index'))} of {esc(grid.get('n_lambda'))}; "
             f"{esc(grid.get('grid_points_to_end'))} points from the weak-penalty end); CV minimum identified: {esc(grid.get('cv_minimum_identified'))}; "
             f"non-zero coefficients: {esc(grid.get('n_selected'))}. Convergence verdict: <b>{esc(au.get('verdict'))}</b>.</p>")
    if au.get("qualifications"):
        b.append("<ul>" + "".join(f"<li>{esc(q)}</li>" for q in au["qualifications"]) + "</ul>")
    b.append(img(rs, f.get("cv"), "10-fold CV deviance against lambda; the band is ±1 SE across folds; orange = selected lambda*"))
    pinfo = info.get("path")
    if pinfo:
        b.append(f"<p class='note'>Coefficient path recomputed on the run's TRAIN design matrix with the saved preprocessor and the run's own lambda grid; "
                 f"it reproduces the saved model at lambda*: <b>{esc(pinfo['reproduces_saved_model'])}</b> (max |difference| {pinfo['max_abs_difference_at_lambda_star']:.2e}).</p>")
        b.append(img(rs, f.get("path"), "Coefficient paths: predictors enter from left (strong penalty) to right; dashed line = lambda*"))
        b.append(table(pd.DataFrame(pinfo.get("entry_order", [])), None))
    b.append(img(rs, f.get("fp"), "Fractional-polynomial shape of continuous predictors (log-odds relative to the median; association, not cause)"))
    forms = info.get("fp_forms") or {}
    if forms:
        b.append(table(pd.DataFrame([{"variable": k, **{kk: (str(vv) if isinstance(vv, dict) else vv) for kk, vv in v.items()}} for k, v in forms.items()])))
    # C learning
    b.append(f"<h2 id='learning'>C. Learning curve {badge('TRAIN / VALIDATION')}</h2>")
    lc_info = info.get("learning_curve")
    if lc_info:
        if lc_info.get("reduced_note"):
            b.append(f"<div class='warn'>{esc(lc_info['reduced_note'])}</div>")
        b.append(f"<p class='note'>{esc(lc_info['rule'])}. Validation: {lc_info['n_validation']} rows, {lc_info['n_validation_events']} events. "
                 "Reading: if the validation AUROC still rises at the right end, more patients would probably still help; a flat right end suggests a plateau "
                 "for this predictor set (richer predictors, not more rows, would then be the lever).</p>")
        b.append(img(rs, f.get("learning"), "Learning curve"))
        b.append(table(t.get("learning_curve")))
    else:
        b.append("<p class='note'>Not computed (learning curve off, or the dataset was not available locally).</p>")
    # D discrimination
    b.append(f"<h2 id='disc'>D. Discrimination {badge('HELD-OUT TEST (model frozen)')}</h2>")
    b.append(img(rs, f.get("roc"), "ROC curve with AUROC and its bootstrap 95% CI"))
    b.append("<div class='note'><b>Why precision-recall matters here.</b> Falls within 180 days are uncommon (the dashed line in the PR plot is the event "
             "prevalence). AUROC counts correct orderings of one faller against one non-faller and is insensitive to how rare the event is; with thousands of "
             "non-fallers, a model can have a respectable AUROC while most flagged patients do not fall. Precision (PPV) answers the operational question "
             "'of those we flag, how many fall?', and PR-AUC summarises it across thresholds. A model is only useful above the no-skill line, and the gap "
             "to that line - not the absolute PR-AUC - is what to read.</div>")
    b.append(img(rs, f.get("pr"), "Precision-recall curve; dashed = no-skill reference (event prevalence)"))
    b.append(img(rs, f.get("risk"), "Predicted-risk distributions of future fallers and non-fallers"))
    # E calibration
    b.append(f"<h2 id='cal'>E. Calibration {badge('HELD-OUT TEST (model frozen)')}</h2>")
    b.append(img(rs, f.get("calibration"), "Observed vs predicted risk by group, uncalibrated and recalibrated (recalibration fitted on validation)"))
    b.append("<ul>" + "".join(f"<li>{esc(k)}: {esc(v)}</li>" for k, v in (info.get("calibration_stats") or {}).items()) + "</ul>")
    b.append("<p class='note'>Slope 1 and CITL 0 / O:E 1 mean predicted risks match observed rates on average and across the range; a slope below 1 means "
             "predictions are too extreme (over-fitting), above 1 too moderate.</p>")
    # F operational
    b.append(f"<h2 id='use'>F. Operational usefulness (exploratory) {badge('HELD-OUT TEST (model frozen)')}</h2>")
    b.append("<div class='warn'>No clinical threshold is chosen here. The tables show what would happen at each threshold / each top-x% selection on the held-out "
             "test partition; they are exploratory and describe the frozen model, they are not a basis for retuning it.</div>")
    lift = t.get("lift")
    if lift is not None and len(lift):
        r5 = lift[lift["top_pct"] == 5.0]
        if len(r5) and r5["pct_of_all_falls_captured"].notna().all():
            r = r5.iloc[0]
            b.append(f"<p><b>If Meuhedet intervened with only the highest-risk 5% of patients</b>, that group would have held {fmt(r['pct_of_all_falls_captured'])}% "
                     f"(95% CI {fmt(r['capture_ci_low'])}–{fmt(r['capture_ci_high'])}%) of the falls observed in the following 180 days in the held-out test "
                     f"partition; its fall rate was {100 * float(r['observed_prevalence']):.1f}%, {fmt(r['lift'])} times the population rate "
                     f"({100 * float(r['population_prevalence']):.1f}%).</p>")
    b.append(img(rs, f.get("gains"), "Cumulative gains and top-x% risk concentration (lift)"))
    b.append(table(lift))
    b.append(img(rs, f.get("deciles"), "Observed fall rate by tenth of predicted risk"))
    b.append(img(rs, f.get("thresholds"), "Sensitivity, specificity, PPV, NPV, % flagged and % of falls captured across thresholds"))
    b.append(table(t.get("thresholds")))
    # G features
    b.append(f"<h2 id='feat'>G. Feature interpretation {badge('TRAIN')}</h2>")
    b.append("<p class='note'>Coefficients are on the model's log-odds scale per 1 SD of each design column (standardised), so their sizes are comparable; "
             "positive = associated with a higher predicted risk, negative = lower. They describe the fitted model on this exploratory label - not causes, "
             "not clinical importance. Stability = share of bootstrap refits of the whole procedure that kept the predictor. Permutation importance = drop "
             "in validation AUROC when the predictor is shuffled.</p>")
    b.append(img(rs, f.get("coefficients"), "LASSO coefficients at lambda*"))
    b.append(img(rs, f.get("stability"), "Bootstrap selection stability"))
    b.append(img(rs, f.get("permutation"), "Permutation importance"))
    b.append(table(t.get("feature_dictionary")))
    # H subgroups
    b.append(f"<h2 id='sub'>H. Subgroup performance (exploratory) {badge('HELD-OUT TEST (model frozen)')}</h2>")
    b.append("<div class='warn'>A metric computed in a subgroup does not establish subgroup validity or fairness. Groups with fewer than 10 events or non-events "
             "are listed without metrics.</div>")
    b.append(img(rs, f.get("subgroups"), "AUROC by subgroup"))
    b.append(table(t.get("subgroups")))
    # I error analysis
    b.append(f"<h2 id='err'>I. Error analysis {badge('TRAIN / VALIDATION')}</h2>")
    ea = t.get("error_validation")
    if ea is not None:
        cp = ea.attrs.get("cut_points", {})
        b.append(f"<p class='note'>Primary: VALIDATION (uncalibrated model). Descriptive risk cut-points of this partition: 'higher risk' = top 20% "
                 f"(risk ≥ {fmt(cp.get('higher_risk_from'))}), 'lower risk' = bottom 50% (risk ≤ {fmt(cp.get('lower_risk_up_to'))}); not clinical thresholds. "
                 "Cells: % with the flag, or median [IQR]. Groups below the small-cell threshold are not profiled.</p>")
        b.append(table(ea))
    et = t.get("error_test")
    if et is not None:
        b.append(f"<h3>Held-out test (after the model was frozen) {badge('HELD-OUT TEST (model frozen)')}</h3>")
        b.append(table(et))
    # J uncertainty
    b.append("<h2 id='unc'>J. Uncertainty of the headline metrics</h2>")
    rows = []
    for split in ("validation", "test"):
        for variant in ("uncalibrated", "recalibrated"):
            p = run.perf(split, variant)
            if p:
                rows.append({"split": split, "variant": variant, "n": p.get("n"), "events": p.get("n_events"), "AUROC": _ci(p.get("auroc")), "PR-AUC": _ci(p.get("pr_auc")),
                             "Brier": _ci(p.get("brier"), 4), "slope": _ci(p.get("calibration_slope"), 2), "CITL": _ci(p.get("citl"), 2), "O:E": _ci(p.get("oe_ratio"), 2)})
    b.append(table(pd.DataFrame(rows)))
    b.append(table(run.csv("optimism.csv")))
    b.append(f"<h2 id='lim'>K. Limitations and problems of this report</h2><ul>" + "".join(f"<li>{esc(x)}</li>" for x in [
        *rr.problems, "Exploratory 180-day Meuhedet outcome - not the eFalls 12-month ED/admission outcome; not an eFalls reproduction.",
        "Single snapshot, internal random split: no temporal or external validation yet.",
        *(run.metrics.get("warnings") or [])[:8]]) + "</ul>")
    return page(f"Model training — {run.display}", "".join(b), run.synthetic, WATERMARK)


def render_comparison(rs: Any) -> str:
    from falls_ml.modelreport.runner import WATERMARK

    b = ["<h1>Model comparison</h1>",
         "<p class='note'>Each analysis is evaluated on its own held-out test partition. Rows with the same test rows as the reference analysis are PAIRED: their "
         "difference is a feature-set effect and comes with a paired bootstrap interval. Rows with a different population are marked NOT A DIRECT PAIRED MODEL "
         "COMPARISON: they differ in patients (and case mix) as well as in predictors, so a difference in a metric cannot be attributed to the predictors.</p>",
         table(rs.comparison, ["label", "analysis", "comparison_note", "rows", "events", "prevalence", "n_predictors", "test_rows", "test_events", "served_variant",
                               "auroc", "auroc_ci_low", "auroc_ci_high", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "brier", "calibration_slope",
                               "calibration_slope_ci_low", "calibration_slope_ci_high", "oe_ratio", "oe_ratio_ci_low", "oe_ratio_ci_high", "test_rows_sha256"]),
         img(rs, rs.figures.get("comparison"), "Metrics side by side (blue = paired with the reference; orange = NOT A DIRECT PAIRED MODEL COMPARISON)"),
         img(rs, rs.figures.get("roc_paired"), "ROC curves of the analyses sharing the reference's test rows"),
         img(rs, rs.figures.get("pr_paired"), "Precision-recall curves of the analyses sharing the reference's test rows"),
         "<h2>Paired differences (same test rows)</h2>", table(pd.DataFrame(rs.paired)),
         "<h2>Reports</h2><ul>" + "".join(f"<li><a href='MODEL_TRAINING_REPORT_{esc(r.run.key)}.html'>{esc(r.run.display)}</a></li>" for r in rs.runs) + "</ul>",
         "<p class='note'>FINAL_DWH_FIXED: not run - it requires the DWH timing correction (Event_Date &lt; Index_Date); no result is fabricated.</p>"]
    if rs.eda:
        b.append(f"<h2>Research-system context</h2><p class='note'>EDA folder {esc(rs.eda.get('dir'))}; EDA split test sha256 "
                 f"{esc(str(rs.eda.get('test_rows_sha256'))[:16])}… (compare with the reference run's). Stages: {esc(rs.eda.get('stages'))}</p>")
    return page("Model comparison", "".join(b), rs.synthetic, WATERMARK)


FIGURE_TEXT = {
    "cohort_funnel": "How many rows remain after each cohort step (file, index date, eligibility, usable label, D-00 exclusion).",
    "partitions": "Train / validation / test sizes with events and non-events.",
    "lasso_cv_curve": "Cross-validated deviance for every penalty strength; the orange line is the chosen penalty (lambda*). A flat bottom means nearby penalties are equivalent.",
    "lasso_coefficient_paths": "How each coefficient grows as the penalty weakens; predictors that enter early and stay large are the ones the model relies on most.",
    "fp_shapes": "The fitted shape of age and medication count (fractional polynomials): change in predicted log-odds relative to the median. Association, not cause.",
    "learning_curve": "Validation performance when the model is trained on 10%...100% of the training patients: still rising = more patients from the same snapshot still add information; flattening = they add little (this cannot show that other data would not help).",
    "roc": "ROC curve: trade-off between catching future fallers (sensitivity) and flagging non-fallers; AUROC with its 95% CI.",
    "precision_recall": "Precision-recall curve: of those flagged, how many fell, against the share of fallers caught; dashed = no-skill (prevalence).",
    "risk_distributions": "Predicted risks of future fallers and non-fallers; the overlap is what the model cannot separate.",
    "calibration": "Predicted vs observed fall rates; the diagonal is perfect agreement. Includes slope, CITL, O:E and Brier.",
    "thresholds": "What each possible risk threshold would mean: sensitivity, specificity, PPV, NPV, workload and falls captured. No threshold is chosen.",
    "gains_lift": "If the highest-risk x% are selected, what share of all falls do they hold, and how many times the average rate.",
    "risk_deciles": "Fall rate in each tenth of predicted risk, lowest to highest.",
    "coefficients": "Model coefficients (standardised): orange = associated with higher predicted risk, blue = lower.",
    "selection_stability": "How often each predictor was kept when the whole model was refitted on bootstrap samples.",
    "permutation_importance": "How much validation AUROC drops when a predictor is shuffled (not causal).",
    "subgroups": "Exploratory AUROC in subgroups; small groups are not shown.",
    "comparison_metrics": "All analyses side by side; orange rows are NOT a direct paired comparison.",
    "comparison_roc_paired": "ROC curves of analyses evaluated on the same test rows.",
    "comparison_pr_paired": "Precision-recall curves of analyses evaluated on the same test rows.",
    "he_funnel": "Management: population funnel (Hebrew).",
    "he_deciles": "Management: fall rate by tenth of predicted risk (Hebrew).",
    "he_roc": "Management: ROC curve with a plain-language legend (Hebrew).",
    "he_calibration": "Management: predicted vs observed risk by risk group (Hebrew).",
    "he_features": "Management: SELECTION STABILITY - how consistently LASSO keeps each predictor across bootstrap refits (Hebrew). Not predictive importance.",
    "he_permutation": "Management: CONTRIBUTION TO THE MODEL'S PREDICTION - drop in validation AUROC when a predictor's values are shuffled (Hebrew). Model-specific, not causal.",
    "he_d00_matrix": "Management: D-00 sensitivity - current reference, safe full-population models, ablations; FINAL_DWH_FIXED shown as pending (Hebrew).",
    "he_d00_top10": "Management: share of falls in the top 10% predicted risk for the reference, the ablations and the safe full-population models (Hebrew).",
    "d00_": "D-00 sensitivity figure (see D00_SENSITIVITY_REPORT.html).",
    "he_lift": "Management: top 5 / 10 / 20% risk concentration (Hebrew).",
    "he_strict_vs_extended": "Management: STRICT vs EXTENDED discrimination with CIs (Hebrew).",
    "he_roadmap": "Management: roadmap from the exploratory model to any clinical use (Hebrew).",
}


def figure_index(rs: Any) -> str:
    from falls_ml.modelreport.runner import WATERMARK

    lines = ["# Figure index", "", f"**{WATERMARK}**", ""]
    if rs.synthetic:
        lines += [f"**{SYNTHETIC}**", ""]
    lines += ["Every figure exists as PNG (200 dpi, for PowerPoint / e-mail) and SVG (vector, for papers) in `figures/`. Technical figures use English and "
              "canonical names; figures starting with `he_` are the Hebrew management versions. Held-out-test figures describe the frozen model.", "",
              "| file | what it shows |", "|---|---|"]
    for p in sorted((rs.out_dir / "figures").glob("*.png")):
        stem = p.stem
        key = next((k for k in sorted(FIGURE_TEXT, key=len, reverse=True) if stem.endswith(k) or stem.startswith(k)), None)
        lines.append(f"| {p.name} / {p.stem}.svg | {FIGURE_TEXT.get(key, '') if key else ''} |")
    return "\n".join(lines) + "\n"
