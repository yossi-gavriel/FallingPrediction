"""Outputs of ``meuhedet-d00``: dependency files, the sensitivity tables, figures (PNG + SVG), D00_SENSITIVITY_REPORT.html (technical,
English), D00_SENSITIVITY_SUMMARY_HE.md (Hebrew, for a reader who is not a statistician), the machine-readable d00_sensitivity.json and
the README. Every sentence with a number is generated from the artifacts; statements are conditional on the intervals; no causal wording,
no hype, no 'best model'."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from falls_ml.d00.dependency import COHORTS, KIND_TEXT, STATUS_TEXT, Graph
from falls_ml.d00.plan import FINAL, REF_EXTENDED, REF_STRICT
from falls_ml.modelreport import figures as F

STATUS_FILL = {"SAFE": "#d7efe2", "UNRESOLVED": "#fbefc9", "UNSAFE": "#f8d4d3"}
STATUS_MARK = {"SAFE": "✓ SAFE", "UNRESOLVED": "? UNRESOLVED", "UNSAFE": "✗ UNSAFE"}
HE_STATUS = {"SAFE": "בטוח (מוכח)", "UNRESOLVED": "לא מוכרע", "UNSAFE": "לא בטוח (מוכח)"}


def _f(v: Any, d: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:.{d}f}" if np.isfinite(x) else "n/a"


def _ci(v: Any, d: int = 3) -> str:
    if not v or v[0] is None:
        return ""
    return f"{_f(v[0], d)}–{_f(v[1], d)}"


def _sgn(v: Any, d: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:+.{d}f}"


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


# ============================================================================ dependency files
def dependency_table(g: Graph) -> pd.DataFrame:
    rows = []
    for f, e in g.features.items():
        rows.append({"canonical_feature": f, "strict": e["strict"], "extended": e["extended"], "mapping_quality": e["mapping_quality"],
                     "transformation": f"{e['op']}: {e['transformation']}", "exact_source_columns": "; ".join(e["value_columns"]),
                     "validation_columns": "; ".join(e["validation_columns"]) or "-", "source_domain": "; ".join(e["domains"]),
                     "sources": "; ".join(e["sources"]), "intermediate_aggregate": e["intermediate_aggregate"],
                     "timing_date_fields": "; ".join(e["record_date_fields"]) or "none in the extract",
                     "same_day_data_may_enter": e["same_day_data_may_enter"], "d00_status_full_labeled": e["FULL_LABELED"]["status"],
                     "d00_status_d00_clean": e["D00_CLEAN"]["status"], "explanation": e["FULL_LABELED"]["explanation"],
                     "design_columns": "; ".join(e["design_columns"]) or "-", "mapping_vs_dictionary": e["mapping_vs_dictionary"],
                     "evidence_code_config": e["evidence_code"]})
    return pd.DataFrame(rows)


def column_table(g: Graph) -> pd.DataFrame:
    rows = []
    for c, e in g.columns.items():
        ev = e.get("evidence") or {}
        rows.append({"column": c, "source": e["source"], "domain": e["domain"], "contract_role": e["contract_role"], "contract_timing": e["contract_timing"],
                     "evidence_kind": e["evidence_kind"], "record_date_column": e["record_date_column"] or "-",
                     "derived_from": "; ".join(e["derived_from"]) or "-", "used_by_features": "; ".join(e["used_by_features"]) or "-",
                     "status_full_labeled": e["FULL_LABELED"]["status"], "status_d00_clean": e["D00_CLEAN"]["status"],
                     "n_on_index_full": ev.get("n_on_index"), "n_after_index_full": ev.get("n_after_index"), "reason_full_labeled": e["FULL_LABELED"]["reason"]})
    return pd.DataFrame(rows)


def dependency_markdown(g: Graph, wm: str, status: list[str], *, synthetic: bool) -> str:
    L = ["# D-00 feature-dependency graph", "", f"**{wm}**", ""]
    if synthetic:
        L += ["**SYNTHETIC DATA - software test only, not scientific results**", ""]
    L += [*[f"- {s}" for s in status], "",
          f"Index date {g.index_date}; prediction at the START of the index day: a predictor may use only records dated strictly before it. "
          f"Cohorts: FULL_LABELED {g.cohort_counts.get('FULL_LABELED')} rows, D00_CLEAN {g.cohort_counts.get('D00_CLEAN')} rows "
          f"({g.cohort_counts.get('d00_rows_in_full_labeled')} D-00 rows: " + ", ".join(f"{k.replace('d00_rows_by_', '')} {v}" for k, v in g.cohort_counts.items()
                                                                                          if k.startswith("d00_rows_by_")) + ").", "",
          "How a feature is traced (never from its name): the builder's own input list (`feature_input_columns`, used by the adapter) -> the "
          "data-dictionary source of every input column -> declared VIEW-level derivations (followed transitively) -> the record-date column that "
          "bounds each source, measured on the rows of each cohort. No outcome value and no test row is used.", "",
          "## Status definitions", ""] + [f"- **{k}**: {v}" for k, v in STATUS_TEXT.items()] + ["", "## Every modelling feature", "",
          "| feature | STRICT | EXTENDED | source columns | domain | source | timing / date field | same-day data may enter | FULL_LABELED | D00_CLEAN | explanation |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f, e in g.features.items():
        L.append(f"| {f} | {'yes' if e['strict'] else '-'} | {'yes' if e['extended'] else '-'} | {', '.join(e['value_columns'])} | {', '.join(e['domains'])} | "
                 f"{', '.join(e['sources'])} | {', '.join(e['record_date_fields']) or 'none'} | {e['same_day_data_may_enter']} | **{e['FULL_LABELED']['status']}** | "
                 f"{e['D00_CLEAN']['status']} | {e['FULL_LABELED']['explanation'].replace('|', '/')} |")
    L += ["", "Per feature, the intermediate VIEW aggregate, the transformation, the design columns of the reference model and the code/config evidence "
          "are in D00_FEATURE_DEPENDENCY.csv / .json.", "", "## Feature sets derived from the graph", ""]
    for std, s in g.feature_sets.items():
        ss, se = s["STRICT_SAFE"], s["SAFE_EXTENDED"]
        L += [f"### Evidence standard {std} ({s['role']}; admits {', '.join(s['admit'])})", "",
              f"- **{ss['name']}** ({len(ss['features'])}): {', '.join(ss['features']) or '(empty)'}" + ("" if ss["identical_to_strict"] else "  -  **DEVIATION from STRICT** (renamed, not called STRICT):")]
        L += [f"    - {f}: {why}" for f, why in ss["deviation"].items()]
        L += [f"- **{se['name']}** ({len(se['features'])}): {', '.join(se['features']) or '(empty)'}; EXTENDED-only predictors added: {', '.join(se['added_to_strict_safe']) or 'none'}"]
        L += [f"    - excluded {f}: {why}" for f, why in se["excluded_extended_predictors"].items()]
        L.append("")
    L += ["- **FULL_EXTENDED**: the reference EXTENDED set - used on D00_CLEAN only until the DWH correction.", "",
          "## Requested columns", "", "| name | kind | FULL_LABELED | D00_CLEAN | resolution |", "|---|---|---|---|---|"]
    for r in g.requested:
        L.append(f"| {r['name']} | {r['kind']} | {r['FULL_LABELED']} | {r['D00_CLEAN']} | {str(r['resolution']).replace('|', '/')} |")
    L += ["", "## Every predictor-role column of the extract (Phase-2 candidates included)", "",
          "| column | source | evidence kind | date field | derived from | used by | FULL_LABELED | D00_CLEAN |", "|---|---|---|---|---|---|---|---|"]
    for c, e in g.columns.items():
        L.append(f"| {c} | {e['source']} | {e['evidence_kind']} | {e['record_date_column'] or '-'} | {', '.join(e['derived_from']) or '-'} | "
                 f"{', '.join(e['used_by_features']) or '-'} | {e['FULL_LABELED']['status']} | {e['D00_CLEAN']['status']} |")
    L += ["", "## Evidence kinds of the sources", ""] + [f"- **{k}**: {v}" for k, v in KIND_TEXT.items()]
    L += ["", "## Open provenance questions (answered only by the DWH / SME, never by assumption)", ""] + [f"- **{k}**: {v}" for k, v in g.config.open_questions.items()]
    L += ["", f"Graph sha256 {g.sha256}; D-00 configuration {g.config.name} {g.config.version} (sha256 {g.config.sha256[:16]}…)."]
    return "\n".join(L) + "\n"


# ============================================================================ tables
def matrix_table(res: Any) -> pd.DataFrame:
    ref_sha = res.records.get(REF_EXTENDED, {}).get("test_rows_sha256")
    rows = []
    for c in res.cells:
        r = res.records[c.cell]
        rows.append({"cell": c.cell, "planned_as": c.planned, "kind": c.kind, "cohort": c.cohort, "feature_set": c.feature_set, "evidence_standard": c.standard or "-",
                     "n_predictors": len(c.features), "predictors": ", ".join(c.features), "status": r.get("status"), "served_by_run_of": r.get("serving_cell") or "-",
                     "rows": r.get("n_rows"), "events": r.get("n_events"), "prevalence": r.get("prevalence"), "test_rows": r.get("test_n"), "test_events": r.get("test_events"),
                     "auroc": r.get("auroc"), "auroc_ci": _ci(r.get("auroc_ci")), "pr_auc": r.get("pr_auc"), "pr_auc_ci": _ci(r.get("pr_auc_ci")),
                     "brier": r.get("brier"), "calibration_slope": r.get("calibration_slope"), "citl": r.get("citl"), "oe_ratio": r.get("oe_ratio"),
                     "served_variant": r.get("served_variant"), "test_rows_sha256_12": (r.get("test_rows_sha256") or "")[:12],
                     "paired_with_reference": bool(ref_sha and r.get("test_rows_sha256") == ref_sha)})
    return pd.DataFrame(rows)


# ============================================================================ figures
def _forest(ax: Any, labels: list[str], est: list[float], lo: list[float], hi: list[float], colors: list[str], ref: float | None = None) -> None:
    y = np.arange(len(labels))
    for i in range(len(labels)):
        if est[i] is None:
            continue
        ax.errorbar([est[i]], [y[i]], xerr=[[est[i] - lo[i]], [hi[i] - est[i]]] if lo[i] is not None else None, fmt="o", color=colors[i], ms=7, capsize=3, lw=2)
        ax.text(hi[i] if hi[i] is not None else est[i], y[i], f"  {est[i]:.3f}", va="center", fontsize=8, color=F.INK)
    if ref is not None:
        ax.axvline(ref, color=F.MUTED, ls="--", lw=1)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()


def figures(res: Any, out: Path, risk: pd.DataFrame, lc: dict[str, Any]) -> dict[str, dict[str, str]]:
    wm = "EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION"
    figs: dict[str, dict[str, str]] = {}
    g = res.graph
    # dependency status heat-table (status colour + text label, never colour alone)
    feats = list(g.features)
    fig = F._fig(7.2, 0.34 * len(feats) + 1.4)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    for i, f in enumerate(feats):
        for j, coh in enumerate(COHORTS):
            st = g.features[f][coh]["status"]
            ax.add_patch(Rectangle((j, i), 0.96, 0.9, color=STATUS_FILL[st]))
            ax.text(j + 0.48, i + 0.45, STATUS_MARK[st], ha="center", va="center", fontsize=8, color=F.INK)
        ax.text(-0.05, i + 0.45, f + ("  (STRICT)" if g.features[f]["strict"] else ""), ha="right", va="center", fontsize=8, color=F.INK)
    for j, coh in enumerate(COHORTS):
        ax.text(j + 0.48, -0.35, coh, ha="center", va="center", fontsize=9, color=F.INK2, fontweight="bold")
    ax.set_xlim(-2.2, 2)
    ax.set_ylim(len(feats) + 0.1, -0.8)
    ax.set_title("D-00 status of every modelling feature (from provenance + record dates; no outcome, no test rows)", fontsize=10, loc="left")
    figs["dependency_status"] = F.save(fig, out, "d00_dependency_status", wm)
    # sensitivity matrix forest
    cells = [c for c in res.cells if res.records[c.cell].get("auroc") is not None]
    if cells:
        fig = F._fig(9, 0.42 * len(cells) + 1.6)
        ax = fig.add_subplot(111)
        F._style(ax)
        labs = [f"{c.cell}  [{c.cohort}, {len(c.features)} predictors]" + (f"  = run of {c.alias_of}" if c.alias_of else "") for c in cells]
        est = [res.records[c.cell]["auroc"] for c in cells]
        ci = [res.records[c.cell].get("auroc_ci") or [None, None] for c in cells]
        cols = [F.BLUE if c.cohort == "D00_CLEAN" else F.ORANGE for c in cells]
        _forest(ax, labs, est, [x[0] for x in ci], [x[1] for x in ci], cols, ref=0.5)
        ax.set_xlabel("held-out test AUROC (95% CI) - blue: D00_CLEAN cohort, orange: FULL_LABELED cohort (different test rows: not paired across colours)")
        ax.set_title("D-00 sensitivity matrix, ablations and reference", fontsize=11, loc="left")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        figs["matrix"] = F.save(fig, out, "d00_matrix_auroc", wm)
    comp = res.summary.get("comparisons", {})
    paired = [r for r in comp.get("ablations", []) + comp.get("feature_set_effects", []) if r.get("paired")]
    if paired:
        fig = F._fig(9, 0.45 * len(paired) + 1.6)
        ax = fig.add_subplot(111)
        F._style(ax)
        labs = [f"{r['compared']}  vs  {r['reference']}" for r in paired]
        _forest(ax, labs, [r["delta_auroc"] for r in paired], [r["delta_auroc_ci_low"] for r in paired], [r["delta_auroc_ci_high"] for r in paired],
                [F.ORANGE if r["comparison_type"].startswith("FEATURE-SET EFFECT (pre-specified") else F.BLUE for r in paired], ref=0.0)
        ax.set_xlabel("paired difference in held-out test AUROC (compared minus reference), 95% CI - identical test rows")
        ax.set_title("Feature-set effects (orange: pre-specified ablations A-C)", fontsize=11, loc="left")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        figs["paired"] = F.save(fig, out, "d00_paired_feature_set_effects", wm)
    if len(risk):
        t = risk[risk["top_pct"] == 10.0].dropna(subset=["pct_of_all_falls_captured"])
        if len(t):
            fig = F._fig(9, 0.42 * len(t) + 1.6)
            ax = fig.add_subplot(111)
            F._style(ax)
            labs = [f"{m}  [{c}]" for m, c in zip(t["model"], t["cohort"])]
            _forest(ax, labs, t["pct_of_all_falls_captured"].astype(float).tolist(), t["capture_ci_low"].tolist(), t["capture_ci_high"].tolist(),
                    [F.BLUE if c == "D00_CLEAN" else F.ORANGE for c in t["cohort"]], ref=10.0)
            ax.set_xlabel("% of all observed falls in the 10% of patients with the highest predicted risk (95% CI); dashed line = 10% (no concentration)")
            ax.set_title("Risk concentration: top 10% (held-out test; exploratory, no threshold chosen)", fontsize=11, loc="left")
            fig.tight_layout(rect=(0, 0.03, 1, 1))
            figs["top10"] = F.save(fig, out, "d00_top10_capture", wm)
    have = [(k, v) for k, v in lc.items() if v.get("table") is not None and len(v["table"])]
    if have:
        fig = F._fig(10, 3.8)
        for i, (k, v) in enumerate(have):
            ax = fig.add_subplot(1, len(have), i + 1)
            F._style(ax)
            t = v["table"]
            t = t[t["status"] == "fitted"] if "status" in t else t
            x = 100 * t["fraction"].astype(float)
            ax.plot(x, t["validation_auroc"], "o-", color=F.BLUE, lw=2, label="validation AUROC")
            if "validation_auroc_ci_low" in t:
                ax.fill_between(x, t["validation_auroc_ci_low"].astype(float), t["validation_auroc_ci_high"].astype(float), color=F.BLUE, alpha=0.15, lw=0)
            if "train_auroc_apparent" in t:
                ax.plot(x, t["train_auroc_apparent"], "s--", color=F.ORANGE, lw=1.5, label="training AUROC (apparent)")
            ax.set_title(f"{k} ({v.get('mode')})", fontsize=10, loc="left")
            ax.set_xlabel("% of the training patients used")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        figs["learning"] = F.save(fig, out, "d00_learning_curves", wm)
    return figs


# ============================================================================ machine-readable summary
def summary_json(res: Any, wm: str, status: list[str], risk: pd.DataFrame, figs: dict[str, Any]) -> dict[str, Any]:
    g = res.graph
    return {"watermark": wm, "status": status, "synthetic_data": res.synthetic, "index_date": res.index_date, "plan_sha256": res.plan.get("plan_sha256"),
            "cohort_counts": g.cohort_counts, "population": res.summary.get("population"),
            "features": {f: {"FULL_LABELED": e["FULL_LABELED"]["status"], "D00_CLEAN": e["D00_CLEAN"]["status"], "strict": e["strict"], "extended": e["extended"],
                             "sources": e["sources"], "explanation": e["FULL_LABELED"]["explanation"]} for f, e in g.features.items()},
            "feature_sets": g.feature_sets, "cells": res.records, "comparisons": res.summary.get("comparisons"),
            "risk_concentration": risk.to_dict(orient="records") if len(risk) else [], "learning_curve": res.summary.get("learning_curve"),
            "warnings_materiality": (res.summary.get("warnings_audit") or {}).get("materiality"), "figures": figs,
            "final_dwh_fixed": {"cell": FINAL, "status": "NOT RUN - requires the DWH correction (Event_Date < Index_Date); no result is fabricated"}}


# ============================================================================ HTML (technical, English)
CSS = """
body{margin:0;font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;background:#fcfcfb;color:#0b0b0b;line-height:1.55;font-size:15px}
.banner{background:#b00020;color:#fff;font-weight:700;text-align:center;padding:9px}.synthetic{background:#6b21a8;color:#fff;font-weight:700;text-align:center;padding:6px}
main{max-width:1180px;margin:0 auto;padding:18px 26px 70px}h1{font-size:25px}h2{font-size:19px;border-bottom:2px solid #e4e3df;padding-bottom:4px;margin-top:34px}
.status{background:#fff3cd;border-left:5px solid #d39e00;padding:8px 14px}.note{background:#f3f2ee;border-radius:6px;padding:8px 12px;font-size:14px}
table{border-collapse:collapse;font-size:12.5px;margin:8px 0;display:block;overflow-x:auto}th,td{border:1px solid #e4e3df;padding:4px 7px;text-align:left;vertical-align:top}
th{background:#f3f2ee}figure{margin:10px 0}figure img{max-width:100%;border:1px solid #e4e3df;border-radius:4px;background:#fff}figcaption{font-size:13px;color:#52514e}
.SAFE{background:#d7efe2}.UNRESOLVED{background:#fbefc9}.UNSAFE{background:#f8d4d3}.paired{color:#1c6e2d;font-weight:600}.unpaired{color:#b00020;font-weight:600}
"""


def _img(share: Path, fig: dict[str, str] | None, caption: str) -> str:
    if not fig:
        return ""
    p = share / "figures" / fig["png"]
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"<figure><img alt='{_esc(caption)}' src='data:image/png;base64,{data}'><figcaption>{_esc(caption)} (figures/{_esc(fig['png'])}, .svg)</figcaption></figure>"


def _table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 200) -> str:
    if df is None or not len(df):
        return "<p class='note'>none</p>"
    d = df[[c for c in (cols or list(df.columns)) if c in df.columns]].head(max_rows)
    head = "".join(f"<th>{_esc(c)}</th>" for c in d.columns)
    body = []
    for _, r in d.iterrows():
        tds = []
        for c in d.columns:
            v = r[c]
            cls = f" class='{v}'" if isinstance(v, str) and v in STATUS_FILL else ""
            if isinstance(v, float):
                v = _f(v, 4)
            tds.append(f"<td{cls}>{_esc(v)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def render_html(res: Any, wm: str, status: list[str], risk: pd.DataFrame, lc: dict[str, Any], audit: dict[str, Any], figs: dict[str, Any]) -> str:
    g, share = res.graph, res.share_dir
    comp = res.summary.get("comparisons", {})
    pop = res.summary.get("population") or {}
    prim = g.feature_sets[g.config.primary_standard]
    b = [f"<h1>D-00 sensitivity analysis - technical report</h1>",
         "<div class='status'><b>Status of every result in this report:</b><ul>" + "".join(f"<li>{_esc(s)}</li>" for s in status) + "</ul></div>",
         f"<p class='note'>Index date {res.index_date}. Plan frozen before any fit: sha256 <code>{_esc(res.plan.get('plan_sha256'))}</code> (ANALYSIS_PLAN.json). "
         "Test-set discipline: feature statuses, feature sets and ablations come from provenance and record dates only; the held-out test rows were "
         "read only after every run of the frozen plan had finished; no result was used to add, drop or re-specify an analysis, and no model is called best. "
         f"Reference folder <code>{_esc(res.reference_dir.name)}</code> verified unchanged at exit (reference_integrity.json).</p>",
         *(["<div class='note'><b>Resumed analysis.</b><ul>" + "".join(f"<li>{_esc(x)}</li>" for x in resume_lines(res)) + "</ul></div>"]
           if resume_lines(res) else []),
         "<h2>1. The D-00 problem</h2>",
         f"<p>Prediction happens at the start of {res.index_date}; a predictor may only use records dated strictly before that day. In this extract "
         f"{g.cohort_counts.get('d00_rows_in_full_labeled')} of the {g.cohort_counts.get('FULL_LABELED')} labelled rows carry a D-00 guard date on/after the "
         "index day (" + ", ".join(f"{k.replace('d00_rows_by_', '')}: {v}" for k, v in g.cohort_counts.items() if k.startswith("d00_rows_by_")) +
         "). The date is known, the time of the record is not; the pre-index value of an aggregate that includes such a record cannot be rebuilt from the "
         "wide table, so nothing is repaired: the reference run removed those rows before training (D00_CLEAN); this analysis instead keeps them (FULL_LABELED) "
         "and removes every predictor whose provenance is not proven safe.</p>",
         f"<p class='note'>Post-hoc description (after the plan was frozen; never used for a decision): FULL_LABELED prevalence {_f((pop.get('FULL_LABELED') or {}).get('prevalence'), 4)}, "
         f"D00_CLEAN {_f((pop.get('D00_CLEAN') or {}).get('prevalence'), 4)}, D-00 rows {_f((pop.get('D00_ROWS') or {}).get('prevalence'), 4)} "
         f"({(pop.get('D00_ROWS') or {}).get('n_rows')} rows, {(pop.get('D00_ROWS') or {}).get('n_events')} events).</p>",
         "<h2>2. Feature-dependency graph</h2>", _img(share, figs.get("dependency_status"), "D-00 status per feature and cohort"),
         _table(dependency_table(g), ["canonical_feature", "strict", "extended", "exact_source_columns", "source_domain", "sources", "timing_date_fields",
                                      "same_day_data_may_enter", "d00_status_full_labeled", "d00_status_d00_clean", "explanation"]),
         "<p class='note'>Full detail: D00_FEATURE_DEPENDENCY.md / .csv / .json and tables/d00_column_dependency.csv (every predictor-role column, "
         "including the Phase-2 fall and diagnosis columns).</p>",
         "<h2>3. Feature sets derived from the graph</h2>"]
    for std, s in g.feature_sets.items():
        ss, se = s["STRICT_SAFE"], s["SAFE_EXTENDED"]
        b.append(f"<h3>{_esc(std)} ({_esc(s['role'])}: admits {', '.join(s['admit'])})</h3><ul>"
                 f"<li><b>{_esc(ss['name'])}</b>: {_esc(', '.join(ss['features']) or '(empty)')}"
                 + ("" if ss["identical_to_strict"] else " - <b>DEVIATION from STRICT</b> (renamed): " + _esc("; ".join(f"{k}: {v}" for k, v in ss["deviation"].items())))
                 + f"</li><li><b>{_esc(se['name'])}</b>: {_esc(', '.join(se['features']) or '(empty)')}; excluded EXTENDED predictors: "
                 + _esc("; ".join(f"{k} ({v.split(' - ')[0]})" for k, v in se["excluded_extended_predictors"].items()) or "none") + "</li></ul>")
    b.append(f"<p class='note'>Primary standard = the rule 'only features proven safe'. The secondary standard (pre-declared in the configuration before any "
             "result) also admits UNRESOLVED features - never an UNSAFE one - so that the full-population question can be answered for predictors the extract "
             "does not implicate but cannot prove.</p>")
    b += ["<h2>4. Sensitivity matrix</h2>", _img(share, figs.get("matrix"), "Held-out test AUROC of every cell"),
          _table(matrix_table(res), ["cell", "kind", "cohort", "evidence_standard", "n_predictors", "status", "served_by_run_of", "rows", "events", "test_rows",
                                     "auroc", "auroc_ci", "pr_auc", "pr_auc_ci", "brier", "calibration_slope", "citl", "oe_ratio", "paired_with_reference"]),
          f"<p class='note'>{_esc(FINAL)}: NOT RUN - all FULL_LABELED rows with the complete EXTENDED set, after the DWH rebuilds every source window with "
          "Event_Date &lt; Index_Date. No number is shown for it anywhere.</p>",
          "<h2>5. Feature-set effects (paired: identical held-out test rows)</h2>",
          "<p><span class='paired'>PAIRED</span>: same rows, same test partition (keys and outcomes verified), different predictors. Differences are "
          "'compared minus reference' with one shared bootstrap.</p>", _img(share, figs.get("paired"), "Paired AUROC differences"),
          "<h3>Pre-specified ablations (questions A-C)</h3>",
          _table(pd.DataFrame(comp.get("ablations", [])), ["question", "question_text", "compared", "reference_auroc", "compared_auroc", "delta_auroc", "delta_auroc_ci_low",
                                                          "delta_auroc_ci_high", "delta_pr_auc", "delta_pr_auc_ci_low", "delta_pr_auc_ci_high", "delta_brier",
                                                          "delta_brier_ci_low", "delta_brier_ci_high", "reference_calibration_slope", "compared_calibration_slope",
                                                          "reference_citl", "compared_citl", "reference_oe_ratio", "compared_oe_ratio", "reference_top5_capture_pct",
                                                          "compared_top5_capture_pct", "reference_top10_capture_pct", "compared_top10_capture_pct", "delta_capture_top10",
                                                          "delta_capture_top10_ci_low", "delta_capture_top10_ci_high", "reference_top20_capture_pct",
                                                          "compared_top20_capture_pct", "reference_top10_lift", "compared_top10_lift", "variant_for_brier"]),
          "<h3>Other feature-set effects</h3>",
          _table(pd.DataFrame(comp.get("feature_set_effects", [])), ["comparison_type", "reference", "compared", "predictors_removed", "predictors_added", "paired",
                                                                    "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high", "delta_pr_auc", "delta_brier",
                                                                    "delta_capture_top10", "delta_capture_top10_ci_low", "delta_capture_top10_ci_high", "note"]),
          "<h2>6. Population effects (NOT paired)</h2>",
          "<p><span class='unpaired'>NOT A DIRECT PAIRED MODEL COMPARISON</span>: the same predictors on two populations; the split is redrawn, the test "
          "partitions differ, and the difference of the two estimates carries more uncertainty than either interval shows. The FULL_LABELED split "
          "is drawn anew (same seed and proportions) on the larger population, so rows of the reference test partition can fall into a FULL_LABELED "
          "training partition: that is legitimate for a separate analysis, it is one more reason the two are never paired, and nothing from the "
          "FULL_LABELED models feeds back into the reference model or its frozen test partition.</p>",
          _table(pd.DataFrame(comp.get("population_effects", []))),
          "<h3>Where the FULL_LABELED models' test rows come from (descriptive, post-hoc)</h3>", _table(pd.DataFrame(comp.get("d00_decomposition", []))),
          "<h2>7. Risk concentration (held-out test; exploratory, no threshold chosen)</h2>", _img(share, figs.get("top10"), "Top-10% capture"),
          _table(risk, ["model", "cohort", "top_pct", "n_selected", "pct_of_population", "falls_in_group", "pct_of_all_falls_captured", "capture_ci_low", "capture_ci_high",
                        "observed_prevalence", "prevalence_ci_low", "prevalence_ci_high", "lift", "lift_ci_low", "lift_ci_high", "population_prevalence"]),
          "<h2>8. Learning curves (TRAIN subsets, VALIDATION evaluation; the test set is never read)</h2>", _img(share, figs.get("learning"), "Learning curves")]
    for k, v in lc.items():
        if v.get("error"):
            b.append(f"<p class='note'>{_esc(k)}: not computed ({_esc(v['error'])})</p>")
            continue
        if v.get("by_fraction"):
            b.append(f"<h3>{_esc(k)}</h3>" + _table(pd.DataFrame(v["by_fraction"])))
        if v.get("increments"):
            b.append(_table(pd.DataFrame(v["increments"])))
        if v.get("plateau"):
            b.append(f"<p class='note'>{_esc(v['plateau']['statement'])}</p>")
        if "matches_existing_report_table" in v:
            b.append(f"<p class='note'>Recomputed points match the 0.6.0 report table (same seeds): {v['matches_existing_report_table']} "
                     f"(existing table mode {v.get('existing_report_mode')}).</p>")
    b.append("<h2>9. LASSO warnings (final fits vs replicates vs learning-curve refits)</h2>")
    for k, items in (audit.get("materiality") or {}).items():
        b.append(f"<h3>{_esc(k)}</h3><ul>" + "".join(f"<li>{_esc(x)}</li>" for x in items) + "</ul>")
    b.append("<p class='note'>Full audit: LASSO_WARNINGS_AUDIT.md / .json (including the sparse-predictor note: a predictor omitted in some resamples is "
             "never read as protective).</p>")
    b += ["<h2>10. What the DWH correction must deliver</h2><ul>",
          "<li>Every diagnosis- and fall-derived column rebuilt with Event_Date &lt; Index_Date (Last_Dx_Date, Last_Fall_Date and every count, flag and "
          "days-since column of those sources), then a re-export and a new meuhedet-explore run with --index-day-records fail reporting 0 violations - "
          f"that run is {FINAL}.</li>"] + [f"<li>{_esc(k)}: {_esc(v)}</li>" for k, v in g.config.open_questions.items()] + ["</ul>"]
    banner = f"<div class='banner'>{_esc(wm)}</div>" + ("<div class='synthetic'>SYNTHETIC DATA - software test only, not scientific results</div>" if res.synthetic else "")
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>D-00 sensitivity report</title><style>{CSS}</style></head><body>{banner}<main>{''.join(b)}</main></body></html>")


# ============================================================================ Hebrew summary
def _he_name(f: str) -> str:
    from falls_ml.modelreport.artifacts import feature_label, load_labels

    try:
        return feature_label(load_labels(), f, "he")
    except Exception:  # noqa: BLE001
        return f


def _auc_he(rec: dict[str, Any] | None) -> str:
    if not rec or rec.get("auroc") is None:
        return "לא חושב"
    ci = rec.get("auroc_ci")
    return f"{rec['auroc']:.3f}" + (f" (רווח סמך 95%: {ci[0]:.3f}–{ci[1]:.3f})" if ci and ci[0] is not None else "")


def _delta_he(row: dict[str, Any] | None, key: str = "auroc", pct: bool = False) -> str:
    if not row or row.get(f"delta_{key}") is None:
        return "לא חושב"
    u = "נקודות אחוז" if pct else ""
    lo, hi = row.get(f"delta_{key}_ci_low"), row.get(f"delta_{key}_ci_high")
    return (f"{row[f'delta_{key}']:+.3f}{(' ' + u) if u else ''}" + (f" (רווח סמך 95%: {lo:+.3f} עד {hi:+.3f})" if lo is not None else ""))


def _direction_he(row: dict[str, Any] | None) -> str:
    if not row or row.get("delta_auroc_ci_high") is None:
        return ""
    if row["delta_auroc_ci_high"] < 0:
        return "ההבחנה ירדה, ורווח הסמך כולו מתחת לאפס – הירידה ברורה."
    if row["delta_auroc_ci_low"] > 0:
        return "ההבחנה עלתה, ורווח הסמך כולו מעל לאפס."
    return "רווח הסמך כולל את האפס – לא הוכח הבדל ברור."


def render_summary_he(res: Any, d: dict[str, Any]) -> str:
    g = res.graph
    rec = res.records
    comp = d.get("comparisons") or {}
    ab = {r["compared"]: r for r in comp.get("ablations", [])}
    risk = pd.DataFrame(d.get("risk_concentration") or [])
    pop = d.get("population") or {}
    prim, sec = g.feature_sets.get(g.config.primary_standard), next((v for k, v in g.feature_sets.items() if k != g.config.primary_standard), None)
    unsafe = [f for f, e in g.features.items() if e["FULL_LABELED"]["status"] == "UNSAFE"]
    unres = [f for f, e in g.features.items() if e["FULL_LABELED"]["status"] == "UNRESOLVED"]
    safe = [f for f, e in g.features.items() if e["FULL_LABELED"]["status"] == "SAFE"]
    ext, strict = rec.get(REF_EXTENDED), rec.get(REF_STRICT)
    n_d00 = g.cohort_counts.get("d00_rows_in_full_labeled")

    def cap(cell: str, top: float = 10.0) -> str:
        if not len(risk):
            return "לא חושב"
        serving = (rec.get(cell) or {}).get("serving_cell")
        r = risk[(risk["model"] == serving) & (risk["top_pct"] == top)]
        if not len(r) or pd.isna(r["pct_of_all_falls_captured"].iloc[0]):
            return "לא דווח (תא קטן מדי או לא הורץ)"
        r = r.iloc[0]
        return (f"{float(r['pct_of_all_falls_captured']):.0f}% מהנפילות (רווח סמך 95%: {float(r['capture_ci_low']):.0f}%–{float(r['capture_ci_high']):.0f}%), "
                f"פי {float(r['lift']):.1f} מהשיעור הממוצע")

    L = ["<div dir=\"rtl\">", "", "# ניתוח רגישות לבעיית התזמון D-00 – סיכום בשפה פשוטה", "", "**תוצאה חקרנית: נפילה תוך 180 יום – לא שחזור של מודל eFalls**",
         "**EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION**", ""]
    if res.synthetic:
        L += ["**נתונים סינתטיים – בדיקת תוכנה בלבד, לא תוצאות מדעיות (SYNTHETIC DATA)**", ""]
    L += ["מצב התוצאות: נקודת זמן אחת בלבד; חלוקה פנימית בלבד; אין תיקוף לאורך זמן ואין תיקוף חיצוני; חלק מההתאמות להגדרות eFalls מקורבות; "
          "בעיית התזמון עדיין דורשת תיקון במקור (DWH); המודל אינו מוכן לשימוש קליני.", "",
          "תוכנית הניתוח (אילו מודלים, אילו משתנים, אילו ניסויי הסרה) נקבעה והוקפאה לפני שאומן מודל כלשהו ולפני שנקראה קבוצת הבדיקה. "
          "שום תוצאה לא שימשה לשינוי התוכנית, ואף מודל לא הוכרז \"הטוב ביותר\".", ""]
    L += resume_lines_he(res)
    # 1
    L += ["## 1. מה הייתה בעיית D-00?", "",
          f"המודל אמור לחזות בתחילת היום {res.index_date}, ולכן מותר לו להשתמש רק במידע שנרשם לפני אותו יום. בקובץ נמצאו {n_d00:,} שורות "
          f"(מתוך {g.cohort_counts.get('FULL_LABELED'):,} שורות עם תוצא שמיש) שבהן תאריך האבחנה האחרונה או הנפילה האחרונה הוא יום התחזית עצמו "
          "(" + ", ".join(f"{k.replace('d00_rows_by_', '')}: {v:,}" for k, v in g.cohort_counts.items() if k.startswith("d00_rows_by_")) + "). "
          "אנחנו יודעים את התאריך אבל לא את השעה, ולכן אי אפשר לדעת אם הרשומה הייתה זמינה ברגע התחזית. אי אפשר גם \"לתקן\" את הערך בפייתון: "
          "הקובץ מכיל רק סיכומים, לא את האירועים עצמם. לכן לא הזזנו תאריכים, לא איפסנו ולא המצאנו ערכים.", ""]
    if pop.get("D00_ROWS") and pop["D00_ROWS"].get("prevalence") is not None:
        diff = pop.get("d00_rows_minus_clean") or {}
        verdict = ("השיעורים שונים (רווח הסמך של ההפרש אינו כולל אפס), ולכן החרגת השורות עלולה להטות את התמונה"
                   if diff.get("differs") else "לא נמדד הבדל ברור בין השיעורים (רווח הסמך של ההפרש כולל אפס)")
        ci = diff.get("ci") or [None, None]
        L += [f"לשם תיאור בלבד (חושב אחרי הקפאת התוכנית ולא שימש לשום החלטה): שיעור הנפילות בשורות D-00 הוא {100 * pop['D00_ROWS']['prevalence']:.2f}%, לעומת "
              f"{100 * ((pop.get('D00_CLEAN') or {}).get('prevalence') or 0):.2f}% באוכלוסייה הנקייה"
              + (f" (הפרש {100 * diff['difference']:+.2f} נקודות אחוז, רווח סמך 95%: {100 * ci[0]:+.2f} עד {100 * ci[1]:+.2f})" if diff.get("difference") is not None else "")
              + f". {verdict}. בכל מקרה, תוצאת המודל הנוכחי מתארת את האוכלוסייה הנקייה בלבד.", ""]
    # 2-3
    L += ["## 2. אילו משתנים של המודל הושפעו?", "",
          "כל משתנה נבדק לפי מקור הנתונים שלו (לא לפי שמו): מאילו עמודות הוא נבנה, מאיזה מקור הן מגיעות ואיזה תאריך תוחם את המקור בזמן.", "",
          f"- **לא בטוחים (הוכח שנכנס מידע מיום התחזית):** {', '.join(_he_name(f) for f in unsafe) or 'אין'}.",
          f"- **לא מוכרעים (הקובץ לא מאפשר להוכיח או לשלול):** {', '.join(_he_name(f) for f in unres) or 'אין'}. "
          "לרשמי המחלות יש רק תאריך הכניסה הראשונה לרשם, ולתרופות אין תאריך כלל; זו מגבלה שקיימת גם במודל הנוכחי.", "",
          "## 3. אילו משתנים הוכחו כבטוחים?", "", f"{', '.join(_he_name(f) for f in safe) or 'אין'}: ערכים קבועים או מחושבים מתאריך התחזית בלבד.", ""]
    if prim:
        ss = prim["STRICT_SAFE"]
        if not ss["identical_to_strict"]:
            L += [f"לכן סט המשתנים \"STRICT הבטוח\" אינו זהה ל-STRICT. הוא נקרא בשם אחר ({ss['name']}) כדי שלא יתבלבל עם STRICT, והסטייה מתועדת משתנה אחר משתנה.", ""]
    # 4
    L += ["## 4. מה קרה כשכללנו את כל המטופלים עם תוצא, רק עם משתנים בטוחים?", ""]
    for std_sets, tag in ((prim, "לפי הכלל המחמיר (רק משתנים שהוכחו בטוחים)"), (sec, "לפי הכלל המשני שנקבע מראש (בטוחים + לא מוכרעים, אף פעם לא משתנה שהוכח לא בטוח)")):
        if not std_sets:
            continue
        name = f"{std_sets['SAFE_EXTENDED']['name']}_FULL"
        r = rec.get(name)
        if r is None:
            continue
        L.append(f"- {tag}: {len(r.get('features') or [])} משתנים, {r.get('n_rows') or 0:,} מטופלים; יכולת הבחנה AUROC {_auc_he(r)}; "
                 f"10% העליונים: {cap(name)}.")
    L += ["", f"לשם השוואה, המודל הנוכחי (EXTENDED על האוכלוסייה הנקייה): AUROC {_auc_he(ext)}. חשוב: זו אינה השוואה ישירה – "
          "אלה אוכלוסיות שונות וקבוצות בדיקה שונות (NOT A DIRECT PAIRED MODEL COMPARISON). כדי להפריד בין השפעת האוכלוסייה להשפעת המשתנים, "
          "אותם סטים הורצו גם על האוכלוסייה הנקייה (סעיף 6 בדוח הטכני).", ""]
    for pe in comp.get("population_effects", []):
        L.append(f"- אותם משתנים ({pe['predictors']}): אוכלוסייה נקייה AUROC {_f(pe['d00_clean_auroc'])}, כל המטופלים AUROC {_f(pe['full_labeled_auroc'])} "
                 "(השוואה לא-מזווגת).")
    L.append("")
    # 5-7
    L += ["## 5. מה קרה כשהסרנו את ההיסטוריה של נפילות קודמות?", "",
          f"על אותם מטופלים ואותה קבוצת בדיקה: AUROC השתנה מ-{_f((ext or {}).get('auroc'))} ל-{_f((rec.get('EXTENDED_NO_FALLS') or {}).get('auroc'))}; "
          f"שינוי {_delta_he(ab.get('EXTENDED_NO_FALLS'))}. {_direction_he(ab.get('EXTENDED_NO_FALLS'))}", "",
          "## 6. מה קרה כשהסרנו את משתנה הניידות (קושי בהליכה)?", "",
          f"שינוי AUROC {_delta_he(ab.get('EXTENDED_NO_MOBILITY'))}. {_direction_he(ab.get('EXTENDED_NO_MOBILITY'))}", "",
          "## 7. כמה יכולת נשארה בלי שניהם?", "",
          f"AUROC {_auc_he(rec.get('EXTENDED_NO_FALLS_NO_MOBILITY'))}; שינוי לעומת המודל המלא {_delta_he(ab.get('EXTENDED_NO_FALLS_NO_MOBILITY'))}. "
          f"{_direction_he(ab.get('EXTENDED_NO_FALLS_NO_MOBILITY'))} לשם השוואה, המודל הבסיסי STRICT: AUROC {_auc_he(strict)}.", "",
          "## 8. האם 10% בסיכון הגבוה ביותר עדיין מרכזים חלק גדול מהנפילות?", ""]
    for cell, he in ((REF_EXTENDED, "המודל הנוכחי"), ("EXTENDED_NO_FALLS", "בלי נפילות קודמות"), ("EXTENDED_NO_MOBILITY", "בלי ניידות"),
                     ("EXTENDED_NO_FALLS_NO_MOBILITY", "בלי שניהם"), (f"{prim['SAFE_EXTENDED']['name']}_FULL" if prim else "", "כל המטופלים, כלל מחמיר"),
                     (f"{sec['SAFE_EXTENDED']['name']}_FULL" if sec else "", "כל המטופלים, כלל משני")):
        if cell and cell in rec:
            L.append(f"- {he}: {cap(cell)}")
    L += ["", "(אם לא הייתה שום התאמה, 10% מהמטופלים היו מכילים בערך 10% מהנפילות. זהו תיאור חקרני – לא נבחר סף להתערבות.)", ""]
    # 9
    L += ["## 9. כמה מהביצועים של EXTENDED תלויים במידע על נפילות קודמות?", ""]
    nf = ab.get("EXTENDED_NO_FALLS")
    sg = (comp.get("share_of_strict_to_extended_gain") or {}).get("EXTENDED_NO_FALLS")
    if nf and nf.get("delta_auroc") is not None and sg:
        gci = sg.get("gain_ci") or [None, None]
        L.append(f"השיפור של EXTENDED על STRICT על אותם מטופלים: {sg['gain']:+.3f} ב-AUROC" + (f" (רווח סמך 95%: {gci[0]:+.3f} עד {gci[1]:+.3f})" if gci[0] is not None else "")
                 + f". השינוי ב-AUROC כשמסירים רק את הנפילות הקודמות: {_delta_he(nf)}.")
        if sg.get("defined") and sg.get("share") is not None:
            sci = sg.get("share_ci") or [None, None]
            L.append(f"כלומר בערך {100 * sg['share']:.0f}% מהשיפור של EXTENDED על STRICT אובד כשמוציאים את משתנה הנפילות"
                     + (f" (רווח סמך 95%: {100 * sci[0]:.0f}%–{100 * sci[1]:.0f}%)" if sci[0] is not None else "") + ".")
            if sci[1] is not None and sci[1] > 1.0:
                L.append("ערך מעל 100% אפשרי: הוא אומר שבלי משתנה הנפילות, שאר המשתנים שנוספו ב-EXTENDED כמעט אינם מוסיפים על STRICT.")
        else:
            L.append("השיפור של EXTENDED על STRICT עצמו אינו ברור (רווח הסמך שלו כולל אפס), ולכן לא מחושב אחוז – מספר כזה היה מטעה.")
        L.append("זהו אומדן גס: המשתנים קשורים זה לזה, ולכן תרומותיהם אינן מצטברות בפשטות. זו תרומה לחיזוי המודל – לא סיבה לנפילה.")
    else:
        L.append("לא חושב (ניסוי ההסרה לא הורץ או לא הושלם).")
    L.append("")
    # 10-12
    L += ["## 10. אילו מסקנות מוצדקות עכשיו?", ""]
    concl = []
    if ext and ext.get("auroc_ci") and ext["auroc_ci"][0] is not None and ext["auroc_ci"][0] > 0.5:
        concl.append(f"על האוכלוסייה הנקייה, המודל הנוכחי מבדיל בין מי שנפל למי שלא נפל (AUROC {_auc_he(ext)}).")
    for cell, he in (("EXTENDED_NO_FALLS", "בלי נפילות קודמות"), ("EXTENDED_NO_FALLS_NO_MOBILITY", "בלי נפילות קודמות ובלי ניידות")):
        r = rec.get(cell)
        if r and r.get("auroc_ci") and r["auroc_ci"][0] is not None:
            concl.append(f"{he} נשארת יכולת הבחנה של AUROC {_auc_he(r)}" + (" – מעל ניחוש מקרי." if r["auroc_ci"][0] > 0.5 else " – לא הוכח שהיא מעל ניחוש מקרי."))
    for cell in ("EXTENDED_NO_FALLS", "EXTENDED_NO_MOBILITY", "EXTENDED_NO_FALLS_NO_MOBILITY"):
        r = ab.get(cell)
        if r and r.get("delta_auroc_ci_high") is not None and r["delta_auroc_ci_high"] < 0:
            concl.append(f"הסרת {' וגם '.join(_he_name(f) for f in r.get('predictors_removed') or [])} מורידה את ההבחנה בבירור על אותם מטופלים ({_delta_he(r)}).")
    for std_sets, how in ((prim, "רק עם משתנים שהוכחו בטוחים"), (sec, "בלי אף משתנה שהוכח לא בטוח (כולל משתנים לא מוכרעים)")):
        r = rec.get(f"{std_sets['SAFE_EXTENDED']['name']}_FULL") if std_sets else None
        if r and r.get("auroc_ci") and r["auroc_ci"][0] is not None and r["auroc_ci"][0] > 0.5:
            concl.append(f"גם על כל המטופלים עם תוצא, {how} ({len(r.get('features') or [])} משתנים), נשארת הבחנה מעל ניחוש מקרי: AUROC {_auc_he(r)}.")
    concl.append("שורות D-00 הוסרו לפני האימון במודל הנוכחי, ולכן אין כאן עדות לכך שמידע עתידי הוזן אליו ביודעין.")
    concl.append("ההשוואות בין סטי משתנים על אותם מטופלים הן מזווגות ואמינות יותר מהשוואות בין אוכלוסיות, שאינן מזווגות.")
    L += [f"- {c}" for c in concl] + ["", "## 11. מה עדיין לא ודאי?", "",
          "- איך המודל המלא היה מתפקד על כל האוכלוסייה אחרי תיקון התזמון (FINAL_DWH_FIXED) – טרם הורץ, ואין לו מספר.",
          "- האם רשמי המחלות ומספר התרופות כוללים רשומות מיום התחזית (לא ניתן לבדוק מהקובץ).",
          "- ביצועים בתאריכים אחרים, לאורך זמן ובאוכלוסיות אחרות (נקודת זמן אחת, חלוקה פנימית).",
          "- ההתאמה של ההגדרות להגדרות eFalls (חלקן מקורבות), והתוצא הוא 180 יום ולא התוצא של eFalls.", "",
          "## 12. מה נדרש מתיקון ה-DWH?", "",
          "- לבנות מחדש את כל העמודות שמבוססות על אבחנות ונפילות עם התנאי Event_Date < Index_Date (כולל Last_Dx_Date, Last_Fall_Date, ספירות, דגלים וימים-מאז).",
          "- לאשר את כלל הזמן של רשמי המחלות (כניסה לרשם ביום התחזית), של חשיפה לתרופות ושל קבוצת Charlson.",
          "- לייצא מחדש ולהריץ meuhedet-explore עם --index-day-records fail; ההרצה חייבת לדווח 0 הפרות. זה יהיה FINAL_DWH_FIXED, והוא יוצג באותה טבלה.", "",
          "קבצים: D00_SENSITIVITY_REPORT.html (דוח טכני), D00_FEATURE_DEPENDENCY.md/.csv/.json, LASSO_WARNINGS_AUDIT.md, tables/, figures/, management/ (דוח הנהלה חדש).",
          "", "</div>", ""]
    return "\n".join(L)


def _retry_counts(cells: list[dict[str, Any]]) -> tuple[int, int, int]:
    n = ok = bad = 0
    for c in cells:
        r = c.get("bootstrap_convergence_retry") or {}
        n += int(r.get("n_initial_convergence_failures") or 0)
        ok += int(r.get("n_converged_on_retry") or 0)
        bad += int(r.get("n_still_failed") or 0)
    return n, ok, bad


def _resume_parts(rs: dict[str, Any]) -> dict[str, Any]:
    reused = [c for c in rs.get("cells", []) if c["action"] == "REUSED_COMPLETED_RUN"]
    refit = [c for c in rs.get("cells", []) if c["action"] != "REUSED_COMPLETED_RUN"]
    earlier = rs.get("earlier_resume_attempts") or []
    refit_all = [*[x for a in earlier for x in a.get("refitted_cells", [])], *refit]     # every cell refitted after the interruption
    return {"reused": reused, "refit": refit, "earlier": earlier, "refit_all": refit_all, "counts": _retry_counts(refit_all),
            "policy_yes": [c["cell"] for c in rs.get("cells", []) if c.get("bootstrap_convergence_retry_policy_in_effect")],
            "policy_no": [c["cell"] for c in rs.get("cells", []) if not c.get("bootstrap_convergence_retry_policy_in_effect")]}


def resume_lines(res: Any) -> list[str]:
    """Plain-text provenance of a resumed analysis (empty when the analysis ran in one attempt)."""
    rs = getattr(res, "resume", None) or {}
    if not rs.get("resumed"):
        return []
    q = _resume_parts(rs)
    n, ok, bad = q["counts"]
    out = [f"This analysis was RESUMED after an interrupted attempt (same frozen plan, sha256 {rs.get('plan_sha256')}).",
           f"Reused as they were - not refitted, test set not re-evaluated ({len(q['reused'])}): " + (", ".join(c["cell"] for c in q["reused"]) or "none") + ".",
           f"Fitted in this attempt ({len(q['refit'])}): " + (", ".join(f"{c['cell']} [{c['action']}]" for c in q["refit"]) or "none") + "."]
    if q["earlier"]:
        refitted = sorted({c for a in q["earlier"] for c in a.get("refitted", [])})
        out.append(f"Earlier resume attempts: {len(q['earlier'])}; cells refitted in them: {', '.join(refitted) or 'none'} (RESUME_LOG.json keeps every attempt).")
    out += [f"Bootstrap convergence retry (policy v0.7.1: same resample, seed and lambda; tolerance unchanged; higher full-data iteration limit; "
            f"accepted only if converged) in the cell(s) refitted after the interruption ({', '.join(c['cell'] for c in q['refit_all']) or 'none'}): "
            f"{n} initial convergence failure(s), {ok} converged on retry, {bad} still failed and were counted as failed replicates "
            "(the >10% abort rule is unchanged).",
            f"Retry policy in effect for: {', '.join(q['policy_yes']) or 'none'}. Fitted before the policy existed (results kept exactly as they were): "
            f"{', '.join(q['policy_no']) or 'none'}.",
            "Incomplete run directories of the earlier attempt (kept on disk for audit, never read as results): "
            + (", ".join(f"{a['run_id']} ({a.get('cell')}; test set released: {a['test_set_released']})" for a in rs.get("incomplete_attempts", [])) or "none") + ".",
            "Every file of the earlier attempt (runs, datasets, configs, pepper copy) is hashed before and after; the test-evaluation registry "
            "may only be appended to (RESUME_LOG.json, reference_integrity.json)."]
    return out


def resume_lines_he(res: Any) -> list[str]:
    rs = getattr(res, "resume", None) or {}
    if not rs.get("resumed"):
        return []
    q = _resume_parts(rs)
    n, ok, bad = q["counts"]
    return ["**הניתוח הושלם בהמשך להרצה שנקטעה.** התוכנית המוקפאת לא השתנתה.", "",
            f"- מודלים שהושלמו קודם ונוצלו כפי שהם (לא אומנו מחדש, וקבוצת הבדיקה שלהם לא נבדקה שוב): {len(q['reused'])}",
            f"- מודלים שאומנו בהרצה הנוכחית: {len(q['refit'])} ({', '.join(c['cell'] for c in q['refit']) or '—'})",
            *([f"- בהרצות המשך קודמות אומנו: {', '.join(sorted({c for a in q['earlier'] for c in a.get('refitted', [])})) or '—'}"] if q["earlier"] else []),
            f"- במודלים שאומנו אחרי ההפסקה ({', '.join(c['cell'] for c in q['refit_all']) or '—'}): דגימות bootstrap שלא התכנסו בניסיון הראשון: {n}; "
            f"התכנסו בניסיון חוזר אחד שנקבע מראש (אותה דגימה, אותו seed ואותו lambda, אותה סבילות; רק מגבלת האיטרציות הוגדלה): {ok}; "
            f"נכשלו גם בניסיון החוזר ונספרו ככישלון: {bad}. כלל העצירה (מעל 10% כישלונות) לא השתנה.",
            f"- מודלים שאומנו לפני שמדיניות הניסיון החוזר נוספה (התוצאות נשמרו בדיוק כפי שהיו): {', '.join(q['policy_no']) or '—'}", ""]


def readme(res: Any, wm: str, status: list[str], *, dependency_only: bool) -> str:
    L = ["# D-00 sensitivity package (falls_ml meuhedet-d00)", "", f"**{wm}**", "", *[f"- {s}" for s in status], "",
         f"Reference (read-only, hashed before and after): `{res.reference_dir.name}`; index date {res.index_date}; plan sha256 {res.plan.get('plan_sha256')}.", ""]
    if dependency_only:
        L += ["Mode: --dependency-only (the graph and the frozen plan; no model was trained).", ""]
    if resume_lines(res):
        L += ["## Resumed analysis", "", *[f"- {x}" for x in resume_lines(res)], ""]
    L += ["## Cells", ""]
    for c in res.cells:
        L.append(f"- {c.cell} [{c.kind}, {c.cohort}, {len(c.features)} predictors]: {res.records.get(c.cell, {}).get('status', c.source + (' of ' + c.alias_of if c.alias_of else ''))}")
    L += ["", "## Files in share/ (aggregate only - this is the folder to send back)", "",
          "D00_SENSITIVITY_REPORT.html, D00_SENSITIVITY_SUMMARY_HE.md, D00_FEATURE_DEPENDENCY.md/.csv/.json, ANALYSIS_PLAN.json, LASSO_WARNINGS_AUDIT.md/.json, "
          "d00_sensitivity.json, tables/*.csv, figures/*.png|svg, split/, adequacy/, management/ (new Hebrew management report), reference_integrity.json, d00_manifest.json"
          + (", RESUME_LOG.json." if (getattr(res, "resume", None) or {}).get("resumed") else "."),
          "", "Never share: datasets/, runs/ (row-level predictions), configs/, id_pepper.txt, the CSV."]
    return "\n".join(L) + "\n"
