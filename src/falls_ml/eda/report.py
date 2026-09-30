"""Rendering of the EDA results: ``REAL_DATA_EDA_REPORT.html`` (self-contained: inline CSS / JS, embedded PNGs, no external resources),
``REAL_DATA_EDA_SUMMARY.md`` (plain language) and ``DATA_QUALITY_REPORT.md``. Everything rendered here is aggregate."""

from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import Any

import pandas as pd

from falls_ml.eda.common import BASIS_TEXT, POSTHOC_DESCRIPTIVE, TRAIN_ONLY, UNIVARIATE_BANNER

SYNTHETIC = "SYNTHETIC DATA — SOFTWARE TEST OUTPUT, NOT SCIENTIFIC RESULTS"
SECTIONS = (
    ("overview", "Overview and workflow"), ("dictionary", "1. Data dictionary and column profile"), ("cohort", "2. Population and cohort"),
    ("outcome", "3. Outcome"), ("numeric", "4. Numeric features"), ("categorical", "5. Binary and categorical features"),
    ("missingness", "6. Missingness"), ("quality", "7. Data quality"), ("correlation", "8. Correlation, redundancy, univariate association"),
    ("temporal", "9. Temporal audit and provenance"), ("table1", "10. Table 1: fallers vs non-fallers"), ("current", "11. Current EXTENDED predictors"),
    ("phase2", "12. Phase-2 candidate catalogue"), ("split", "13. Split diagnostic"), ("outputs", "14. Outputs, privacy, reproducibility"),
)
CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--surface:#fcfcfb;--panel:#ffffff;--accent:#2a78d6;--banner:#b00020}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:var(--surface);color:var(--ink);line-height:1.45;font-size:14px}
.banner{background:var(--banner);color:#fff;font-weight:700;text-align:center;padding:10px 16px;font-size:15px;position:sticky;top:0;z-index:5}
.synthetic{background:#6b21a8;color:#fff;font-weight:700;text-align:center;padding:6px}
.layout{display:flex;align-items:flex-start}
nav{position:sticky;top:42px;width:250px;flex:0 0 250px;max-height:calc(100vh - 42px);overflow:auto;padding:16px 12px;border-right:1px solid var(--line);background:var(--panel)}
nav a{display:block;color:var(--ink2);text-decoration:none;padding:4px 6px;border-radius:4px;font-size:13px}nav a:hover{background:#eef3fb;color:var(--ink)}
main{flex:1;min-width:0;padding:16px 32px 80px;max-width:1400px}
h1{font-size:22px;margin:8px 0}h2{font-size:18px;border-bottom:2px solid var(--line);padding-bottom:4px;margin-top:40px}h3{font-size:15px;margin-top:22px}
.meta{color:var(--ink2);font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin:12px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px}.card .v{font-size:20px;font-weight:700}.card .k{font-size:12px;color:var(--ink2)}
.note{background:#f3f2ee;border-left:4px solid #b9b8b3;padding:8px 12px;margin:10px 0;font-size:13px}
.warn{background:#fff3cd;border-left:4px solid #d39e00;padding:8px 12px;margin:10px 0;font-weight:600;font-size:13px}
.basis{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;background:#e8f0fb;color:#104281;margin-left:6px;vertical-align:middle}
.basis.train{background:#e6f5ee;color:#0b5d3b}.basis.posthoc{background:#fde8e1;color:#8a2c0b}
.tw{overflow:auto;max-height:520px;border:1px solid var(--line);border-radius:4px;margin:6px 0 14px;background:var(--panel)}
table{border-collapse:collapse;font-size:12px;width:max-content;min-width:100%}th,td{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;vertical-align:top;max-width:420px}
th{position:sticky;top:0;background:#f0efec;cursor:pointer;white-space:nowrap}tr:hover td{background:#f7f9fc}
input.filter{width:360px;max-width:100%;padding:5px 8px;border:1px solid #c9c8c2;border-radius:4px;font-size:13px;margin:4px 0}
figure{margin:10px 0 18px}figure img{max-width:100%;border:1px solid var(--line);border-radius:4px;background:#fff}figcaption{font-size:12px;color:var(--ink2)}
details{margin:6px 0;border:1px solid var(--line);border-radius:4px;background:var(--panel)}summary{cursor:pointer;padding:6px 10px;font-weight:600}
details .inner{padding:4px 12px}
.st-DONE{color:#0b7a0b;font-weight:700}.st-NEXT{color:#52514e;font-weight:700}.st-NOT{color:#b25e00;font-weight:700}.st-STOPPED{color:#b00020;font-weight:700}
ul.tight li{margin:2px 0}
"""
JS = """
document.querySelectorAll('input[data-filter]').forEach(function(inp){inp.addEventListener('input',function(){
 var t=document.getElementById(inp.dataset.filter);var q=inp.value.toLowerCase();
 t.querySelectorAll('tbody tr').forEach(function(tr){tr.style.display=tr.textContent.toLowerCase().indexOf(q)>=0?'':'none';});});});
document.querySelectorAll('table.sortable').forEach(function(t){t.querySelectorAll('th').forEach(function(th,i){th.addEventListener('click',function(){
 var tb=t.tBodies[0];var rows=Array.prototype.slice.call(tb.rows);var asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
 rows.sort(function(a,b){var x=a.cells[i].textContent.trim(),y=b.cells[i].textContent.trim();var nx=parseFloat(x),ny=parseFloat(y);
  var c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c;});rows.forEach(function(r){tb.appendChild(r);});});});});
"""


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.4g}" if abs(v) < 1e6 else f"{v:,.0f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


class Html:
    def __init__(self, res: Any):
        self.res = res
        self.parts: list[str] = []
        self.n_tables = 0

    def add(self, s: str) -> None:
        self.parts.append(s)

    def h(self, level: int, text: str, anchor: str | None = None, basis: str | None = None) -> None:
        a = f' id="{anchor}"' if anchor else ""
        self.add(f"<h{level}{a}>{_esc(text)}{self.badge(basis) if basis else ''}</h{level}>")

    @staticmethod
    def badge(basis: str) -> str:
        cls = "train" if basis == TRAIN_ONLY else ("posthoc" if basis == POSTHOC_DESCRIPTIVE else "")
        return f'<span class="basis {cls}" title="{_esc(BASIS_TEXT.get(basis, basis))}">{_esc(basis)}</span>'

    def p(self, text: str, cls: str = "") -> None:
        self.add(f'<p class="{cls}">{_esc(text)}</p>' if cls else f"<p>{_esc(text)}</p>")

    def div(self, text: str, cls: str) -> None:
        self.add(f'<div class="{cls}">{_esc(text)}</div>')

    def ul(self, items: list[str]) -> None:
        self.add('<ul class="tight">' + "".join(f"<li>{_esc(i)}</li>" for i in items) + "</ul>")

    def table(self, df: pd.DataFrame | None, cols: list[str] | None = None, *, max_rows: int | None = None, filterable: bool = False,
              caption: str = "") -> None:
        if df is None or not len(df):
            self.p("Not available for this run (no rows).", "note")
            return
        d = df[[c for c in (cols or list(df.columns)) if c in df.columns]]
        if max_rows is not None and len(d) > max_rows:
            caption = (caption + " " if caption else "") + f"(first {max_rows} of {len(d)} rows; the CSV holds all)"
            d = d.head(max_rows)
        self.n_tables += 1
        tid = f"t{self.n_tables}"
        if filterable:
            self.add(f'<input class="filter" data-filter="{tid}" placeholder="filter rows (type any text)…">')
        head = "".join(f"<th>{_esc(c)}</th>" for c in d.columns)
        body = "".join("<tr>" + "".join(f"<td>{_esc(_fmt(v))}</td>" for v in row) + "</tr>" for row in d.itertuples(index=False, name=None))
        if caption:
            self.add(f'<div class="meta">{_esc(caption)}</div>')
        self.add(f'<div class="tw"><table id="{tid}" class="sortable"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')

    def img(self, rel: str, caption: str = "") -> None:
        p = self.res.out_dir / rel
        if not p.exists():
            return
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        self.add(f'<figure><img loading="lazy" alt="{_esc(caption or rel)}" src="data:image/png;base64,{data}"><figcaption>{_esc(caption or rel)}</figcaption></figure>')

    def imgs(self, group: str, captions: bool = True) -> None:
        for rel in self.res.plots.get(group, []):
            self.img(rel, Path(rel).stem.replace("_", " ") if captions else "")


def _stage_table(res: Any) -> str:
    rows = []
    for name in ("RAW EXTRACT", "CONTRACT CHECK", "FULL EDA", "LEAKAGE / TIMING AUDIT", "COHORT BUILD", "SPLIT", "TRAIN-ONLY SUPERVISED EDA", "ADEQUACY",
                 "MODELLING", "VALIDATION", "FINAL TEST", "REPORTING"):
        s = res.stages.get(name, {"status": "NOT RUN", "detail": ""})
        cls = "st-" + s["status"].split(" ")[0]
        rows.append(f"<tr><td>{_esc(name)}</td><td class='{cls}'>{_esc(s['status'])}</td><td>{_esc(s['detail'])}</td></tr>")
    return ('<div class="tw"><table><thead><tr><th>stage</th><th>status</th><th>detail</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{100.0 * float(v):.2f}%"


def _top_available(res: Any, n: int = 10) -> pd.DataFrame:
    uni = res.tables.get("univariate_association_train")
    if uni is None or not len(uni):
        return pd.DataFrame()
    u = uni[uni["available_at_prediction_time"].astype(str).str.startswith("YES")].copy()
    u["abs"] = pd.to_numeric(u["smd_fall_vs_no_fall"], errors="coerce").abs()
    return u.dropna(subset=["abs"]).sort_values("abs", ascending=False).head(n)


def _blocked_signal(res: Any, n: int = 6) -> pd.DataFrame:
    uni = res.tables.get("univariate_association_train")
    if uni is None or not len(uni):
        return pd.DataFrame()
    u = uni[~uni["available_at_prediction_time"].astype(str).str.startswith("YES")].copy()
    u["abs"] = pd.to_numeric(u["smd_fall_vs_no_fall"], errors="coerce").abs()
    return u[u["abs"] >= 0.2].sort_values("abs", ascending=False).head(n)


def render_html(res: Any) -> str:
    f, t = res.facts, res.tables
    H = Html(res)
    H.add('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
          f"<title>Real data EDA</title><style>{CSS}</style></head><body>")
    H.add(f'<div class="banner">{_esc(res.watermark)}</div>')
    if res.synthetic:
        H.add(f'<div class="synthetic">{SYNTHETIC}</div>')
    H.add('<div class="layout"><nav><div class="meta" style="padding:0 6px 8px">Sections</div>'
          + "".join(f'<a href="#{a}">{_esc(t_)}</a>' for a, t_ in SECTIONS) + "</nav><main>")
    # ---------------- overview
    H.h(1, "Real-data exploratory data analysis — Meuhedet wide extract", "overview")
    H.add(f'<div class="meta">input {_esc(res.input_file)} (sha256 {_esc(f["input_sha256"][:16])}…); index date {_esc(res.index_date or "n/a")}; '
          f'D-00 policy {_esc(res.policy)}; falls_ml {_esc(res.manifest.get("falls_ml_version", ""))}</div>')
    cards = [("rows in the file", f"{f['n_rows']:,}"), ("distinct patients", f"{f['n_patients']:,}"), ("snapshots (index dates)", str(f["n_snapshots"])),
             ("eligible with a usable label", f"{f['n_labelled']:,}"), ("D-00 rows (guard date >= index day)", f"{f['n_d00_violating']:,}"),
             ("modelling cohort rows", f"{f['n_cohort']:,}"), ("events in the cohort", f"{f['n_cohort_events']:,}"),
             ("180-day prevalence (cohort)", _pct(f["cohort_prevalence"])), ("columns profiled", str(len(t["column_profile"]))),
             ("data-quality findings", f"{f['n_findings']['CRITICAL']} critical / {f['n_findings']['WARNING']} warning / {f['n_findings']['INFO']} info")]
    H.add('<div class="cards">' + "".join(f'<div class="card"><div class="v">{_esc(v)}</div><div class="k">{_esc(k)}</div></div>' for k, v in cards) + "</div>")
    H.h(3, "Workflow: where this command sits")
    H.add(_stage_table(res))
    H.h(3, "How to read the badges (analysis basis)")
    H.ul([f"{k}: {v}" for k, v in BASIS_TEXT.items()])
    H.div("Scientific separation: data-quality and unsupervised analyses read the whole file. Every analysis that looks at the outcome and could inform a "
          "modelling decision reads the TRAIN partition only (the modelling split itself). Tables marked POST-HOC DESCRIPTIVE include test rows and describe "
          "the population; they are never used to choose a model or a feature.", "note")
    H.div("Nothing in this report is a p-value, and no association here is causal or a measure of multivariate importance.", "note")
    # ---------------- 1 dictionary
    H.h(2, SECTIONS[1][1], "dictionary", "DEFINITIONS")
    dd = t["data_dictionary"]
    dom = dd.groupby("domain").agg(columns=("column", "count"), predictor_role=("predictor_role", "sum"), used_by_extended=("used_by_extended", "sum")).reset_index()
    H.table(dom, caption="columns per domain")
    ms = dd["meaning_status"].value_counts().to_dict()
    H.p(f"Meaning status: {ms.get('DOCUMENTED', 0)} documented, {ms.get('NAME_ONLY', 0)} name only (business definition NEEDS SME REVIEW), "
        f"{ms.get('UNKNOWN', 0)} UNKNOWN / NEEDS SME REVIEW. Meanings are never invented; the dictionary file is configs/meuhedet/wide_v1_data_dictionary.yaml.")
    H.table(dd, ["column", "domain", "declared_type", "sql_type", "contract_role", "contract_timing", "meaning", "meaning_status", "available_at_prediction_time",
                 "used_by_strict", "used_by_extended", "efalls_features", "mapping_confidence", "record_date_column", "null_means"], filterable=True,
            caption="data dictionary (all columns; full detail in eda/data_dictionary.csv)")
    H.h(3, "Column profile (full extract)", basis="FULL_EXTRACT")
    H.table(t["column_profile"], ["column", "domain", "declared_type", "inferred_pandas_dtype", "row_count", "non_null_count", "missing_pct", "unique_count",
                                  "status", "top_values", "min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max", "mean", "sd", "iqr", "date_min",
                                  "date_max", "declared_sentinel_count", "undeclared_far_future_count", "on_or_after_index_count", "parse_failures",
                                  "parse_failure_patterns", "out_of_contract_values", "out_of_contract_levels", "available_at_prediction_time"],
            filterable=True, caption="identifier values are never shown; counts below the small-cell threshold read '<min_cell'")
    # ---------------- 2 cohort
    H.h(2, SECTIONS[2][1], "cohort")
    H.imgs("cohort")
    H.table(t["cohort_flow"], caption="cohort funnel (eda/cohort_flow.csv)")
    xc = t["cohort_flow"].attrs.get("build_crosscheck", []) if len(t["cohort_flow"]) else []
    if xc:
        H.p("Cross-check against the actual canonical build: " + "; ".join(f"{c['what']}: EDA {c['eda_rows']} vs build {c['build_rows']} ({'agree' if c['agree'] else 'DISAGREE'})" for c in xc))
    H.table(t["population_profile"], filterable=True, caption="population profile: age, sex, suspect birth dates, HMO seniority, history, eligibility, exclusions, labels, censoring, follow-up")
    # ---------------- 3 outcome
    H.h(2, SECTIONS[3][1], "outcome")
    H.p(f"Fall_Next_180D_Ind: {f['n_cohort_events']:,} events among {f['n_cohort']:,} modelling-cohort rows ({_pct(f['cohort_prevalence'])}); among all "
        f"eligible rows with a usable label {_pct(f['labelled_prevalence'])}. D-00 rows: {_pct(f['d00_prevalence'])} vs clean rows {_pct(f['clean_prevalence'])} "
        "(a difference means the D-00 exclusion is outcome-related).")
    H.div("POST-HOC DESCRIPTIVE: the stratified prevalences below use the whole modelling cohort (test rows included). They describe the population "
          "and are never used for feature selection or model choice. This is the Meuhedet 180-day exploratory outcome, not the eFalls outcome.", "warn")
    H.imgs("outcome")
    H.table(t["outcome_prevalence"], filterable=True, caption="prevalence with Wilson 95% intervals (eda/outcome_prevalence.csv)")
    # ---------------- 4 numeric
    H.h(2, SECTIONS[4][1], "numeric")
    np_ = t["numeric_profile"]
    H.table(np_[np_["basis"] == ("MODELLING_COHORT" if (np_["basis"] == "MODELLING_COHORT").any() else "FULL_EXTRACT")] if len(np_) else np_,
            ["column", "basis", "domain", "declared_type", "n_observed", "missing_pct", "min", "p1", "p25", "median", "p75", "p99", "max", "mean", "sd",
             "zero_pct", "skewness", "n_above_q3_plus_3iqr", "n_above_p99", "n_negative", "transformations_to_investigate"], filterable=True,
            caption="distribution shape (unsupervised). Transformations are suggestions to investigate on TRAIN; nothing was transformed.")
    H.h(3, "By outcome (TRAIN only)", basis=TRAIN_ONLY)
    H.table(t.get("numeric_outcome_train"), filterable=True)
    for group in sorted(g for g in res.plots if g.startswith("numeric::")):
        H.add(f"<details><summary>Numeric plots — {_esc(group.split('::', 1)[1])} ({len(res.plots[group])})</summary><div class='inner'>")
        H.imgs(group)
        H.add("</div></details>")
    # ---------------- 5 categorical
    H.h(2, SECTIONS[5][1], "categorical")
    H.h(3, "Level distribution (unsupervised)", basis="MODELLING_COHORT")
    cp = t["categorical_profile"]
    H.table(cp[cp["basis"] == ("MODELLING_COHORT" if (cp["basis"] == "MODELLING_COHORT").any() else "FULL_EXTRACT")] if len(cp) else cp, filterable=True, max_rows=1500)
    H.h(3, "Outcome by level (TRAIN only)", basis=TRAIN_ONLY)
    H.div(UNIVARIATE_BANNER + ". Prevalence ratios are versus the most frequent level; 'separation' marks levels with no events or only events.", "warn")
    H.table(t.get("categorical_outcome_train"), filterable=True, max_rows=1500)
    for group in sorted(g for g in res.plots if g.startswith("categorical::")):
        H.add(f"<details><summary>Categorical plots — {_esc(group.split('::', 1)[1])} ({len(res.plots[group])})</summary><div class='inner'>")
        H.imgs(group)
        H.add("</div></details>")
    # ---------------- 6 missingness
    H.h(2, SECTIONS[6][1], "missingness")
    mi = t["missingness"]
    pct = pd.to_numeric(mi["missing_pct"], errors="coerce").fillna(0.0)
    H.ul([f"{lab}: {int((pct > thr).sum())} columns — " + ", ".join(mi.loc[pct > thr, "column"].head(25)) + (" …" if (pct > thr).sum() > 25 else "")
          for lab, thr in ((">99% missing", 99), (">90% missing", 90), (">50% missing", 50), (">25% missing", 25))])
    H.div("NULL is never converted to zero here. Each NULL is classified: true clinical absence, not measured, source unavailable (row-level source-absent "
          "flag), not applicable, technical failure (unparseable value or NULL in a NOT NULL column), sentinel date, or unknown (no declared meaning).", "note")
    H.imgs("missingness")
    H.table(t["missingness_by_domain"], caption="by domain")
    H.table(mi, ["column", "domain", "source", "declared_null_meaning", "declared_null_class", "missing_count", "missing_pct", "missing_or_sentinel_pct",
                 *[c for c in mi.columns if c.startswith("missing_pct_")], "n_true_clinical_absence", "n_not_measured", "n_source_unavailable",
                 "n_not_applicable", "n_technical_failure", "n_sentinel", "n_unknown", "dominant_class", "band"], filterable=True)
    H.h(3, "Co-missingness pairs (phi >= 0.8)")
    H.table(t.get("co_missingness_pairs"), max_rows=300, filterable=True)
    H.h(3, "Source-level patterns")
    H.table(t.get("missingness_patterns"))
    rm = f["row_missingness"]
    H.p(f"Rows with unusually high missingness ({f['row_missingness_basis']}): median share of predictor columns missing {rm.get('median_share')}, p99 {rm.get('p99_share')}; "
        f"{rm.get('n_rows_above_50pct')} rows miss more than 50% and {rm.get('n_rows_above_75pct')} more than 75% of the {rm.get('n_columns')} predictor columns.")
    H.h(3, "Is missingness associated with the outcome?", basis=TRAIN_ONLY)
    H.table(t.get("missingness_outcome_train"), filterable=True)
    # ---------------- 7 quality
    H.h(2, SECTIONS[7][1], "quality", "FULL_EXTRACT")
    H.imgs("quality")
    H.p("Consistency rules are expected relations read from the column names and S2T notes: a finding is a question for the DWH / SME, not a proven "
        "error. Nothing was repaired. The full list is in DATA_QUALITY_REPORT.md.")
    H.table(t["data_quality_findings"], ["check_id", "severity", "category", "title", "columns", "n_rows", "pct_rows", "basis", "detail", "recommendation"], filterable=True)
    dqc = t["data_quality_checks"]
    H.p(f"{len(dqc)} checks run: {int((dqc['status'] == 'FINDING').sum())} with findings, {int((dqc['status'] == 'PASS').sum())} passed, "
        f"{int((dqc['status'] == 'NOT_APPLICABLE').sum())} not applicable.")
    # ---------------- 8 correlation
    H.h(2, SECTIONS[8][1], "correlation", TRAIN_ONLY)
    H.imgs("association")
    H.div(f"{UNIVARIATE_BANNER}. Effect sizes on the TRAIN partition only; no p-values; a large univariate difference is not a reason to select a feature.", "warn")
    H.table(t.get("univariate_association_train"), filterable=True, caption="eda/univariate_association_train.csv")
    H.h(3, "Correlated pairs")
    H.table(t.get("correlation_pairs"), filterable=True, max_rows=400)
    H.h(3, "Redundancy clusters (|Spearman| >= 0.8)")
    H.table(t.get("redundancy_clusters"))
    # ---------------- 9 temporal
    H.h(2, SECTIONS[9][1], "temporal", "FULL_EXTRACT")
    H.p(f"Prediction at the START of the index day: a predictor may use records dated strictly before Index_Date. D-00 root cause in this file: "
        f"{f['timing_summary']['root_cause']}; eligible rows violating the cohort guard: {f['timing_summary']['n_eligible_rows_violating_d00']}. "
        f"SAFE-ALL-ROWS predictor set: {', '.join(f['safe_all_rows_feature_set'])} (excluded: {', '.join(f['excluded_from_safe_all_rows']) or 'none'}).")
    H.imgs("temporal")
    H.table(t["temporal_audit"], filterable=True, caption="every date column + every source without a record date (eda/temporal_audit.csv)")
    H.h(3, "Provenance: source -> aggregate column -> canonical feature -> model(s)", basis="DEFINITIONS")
    H.table(t["feature_provenance"], filterable=True)
    # ---------------- 10 table 1
    H.h(2, SECTIONS[10][1], "table1", POSTHOC_DESCRIPTIVE)
    H.div("POST-HOC DESCRIPTIVE Table 1 of the whole modelling population (test rows included): describes who fell; never used for model or feature "
          "choice. Standardised mean differences describe group differences; they are not p-values and not importance.", "warn")
    H.table(t.get("table1"))
    H.h(3, "The same table on TRAIN only", basis=TRAIN_ONLY)
    H.table(t.get("table1_train"))
    # ---------------- 11 current
    H.h(2, SECTIONS[11][1], "current")
    H.p("Every field is derived from the mapping YAML, the contract, the data dictionary and the D-00 evidence of this file; the coefficient, bootstrap "
        "selection frequency and permutation importance come from the completed reference run when --reference is given.")
    H.table(t["current_15_feature_dictionary"])
    # ---------------- 12 phase 2
    H.h(2, SECTIONS[12][1], "phase2")
    H.div("Catalogue only: none of these columns was added to the eFalls baseline or trained on. Univariate figures are TRAIN-only and descriptive.", "warn")
    p2 = t["phase2_candidate_features"]
    if len(p2):
        H.table(p2["readiness"].str.split(" - ").str[0].value_counts().rename_axis("readiness").reset_index(name="columns"))
    H.table(p2, filterable=True)
    # ---------------- 13 split
    H.h(2, SECTIONS[13][1], "split", "SPLIT_DIAGNOSTIC")
    sp = f.get("split", {})
    if sp:
        H.p(f"Strategy {sp.get('strategy')}, seed {sp.get('seed')}, configuration from the {sp.get('config_source')}; test-partition sha256 "
            f"{str(sp.get('test_rows_sha256', ''))[:16]}…" + (" — identical to the reference run" if sp.get("matches_reference") else ""))
    H.p("A diagnostic of the deterministic split: variables are compared without outcome stratification; |SMD| > 0.1 is flagged. It is never a reason "
        "to re-split until the partitions look alike.")
    H.imgs("split")
    H.table(t.get("split_balance"))
    # ---------------- 14 outputs
    H.h(2, SECTIONS[14][1], "outputs")
    rows = [{"file": k, "basis": v["basis"], "what": v["what"]} for k, v in res.manifest.get("files", {}).items()]
    H.table(pd.DataFrame(rows))
    priv = res.manifest.get("privacy", {})
    H.p(f"Privacy: {priv.get('rule', '')}. Identifier scan: {priv.get('identifier_scan', {})}.")
    H.p(f"Definitions: {res.manifest.get('definitions', {})}")
    H.p(f"Run time (seconds, cumulative): {res.timings}")
    H.add(f"</main></div><script>{JS}</script></body></html>")
    return "\n".join(H.parts)


# ============================================================================ markdown
def _md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 40) -> list[str]:
    if df is None or not len(df):
        return ["(none)"]
    d = df[[c for c in cols if c in df.columns]].head(max_rows)
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    for row in d.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_fmt(v).replace("|", "/").replace("\n", " ") for v in row) + " |")
    return lines


def render_summary(res: Any) -> str:
    f, t = res.facts, res.tables
    wm = res.watermark
    dq = t["data_quality_findings"]
    mi = t["missingness"]
    pct = pd.to_numeric(mi["missing_pct"], errors="coerce").fillna(0.0)
    L = [f"# Real-data EDA summary", "", f"**{wm}**", ""]
    if res.synthetic:
        L += [f"**{SYNTHETIC}**", ""]
    L += ["Every number below comes from the local EDA outputs of this file. Nothing here is a p-value, a causal statement or a model result.", "",
          "## What the dataset contains", "",
          f"- {f['n_rows']:,} rows and {f['n_columns_in_file']} columns for {f['n_patients']:,} patients on {f['n_snapshots']} index date(s) "
          f"({', '.join(f['index_dates'][:6])}{' …' if len(f['index_dates']) > 6 else ''}).",
          f"- {f['n_eligible_on_index_date']:,} rows are eligible on the index date {res.index_date or 'n/a'}; {f['n_labelled']:,} of them have a usable "
          f"180-day label and {f['n_unlabelled_eligible']:,} do not (censored or otherwise unusable).",
          f"- The modelling cohort holds {f['n_cohort']:,} rows ({f['n_cohort_patients']:,} patients) with {f['n_cohort_events']:,} falls within 180 days "
          f"({_pct(f['cohort_prevalence'])})." if f["n_cohort"] else "- No modelling cohort was built (see the workflow table in the HTML report).",
          f"- Columns by domain: " + "; ".join(f"{r['domain']} {r['columns']}" for _, r in t["data_dictionary"].groupby("domain").size().rename("columns").reset_index().iterrows()) + ".",
          f"- Meanings: {f['meaning_status'].get('DOCUMENTED', 0)} documented, {f['meaning_status'].get('NAME_ONLY', 0)} name only, "
          f"{f['meaning_status'].get('UNKNOWN', 0)} unknown (both need SME review).", ""]
    checks = t["data_quality_checks"]

    def passed(cid: str) -> bool:
        s = checks[checks["check_id"] == cid]["status"]
        return len(s) > 0 and (s == "PASS").all()

    good = []
    if not res.facts["contract"]["missing_columns"]:
        good.append("every contract column is present in the file")
    if passed("G01"):
        good.append("no duplicated patient x index-date rows")
    if passed("G03"):
        good.append("no duplicated snapshot keys")
    if passed("B03"):
        good.append("no out-of-contract codes")
    if passed("B01"):
        good.append("every value parsed into its declared type")
    if passed("F01") and passed("F02"):
        good.append("labels and event dates agree")
    complete = [c for c, p_ in zip(mi["column"], pct) if p_ == 0]
    good.append(f"{len(complete)} of {len(mi)} columns have no missing value")
    L += ["## What is good", ""] + [f"- {g}" for g in good] + [""]
    L += ["## What is poor quality or needs an explanation", ""]
    crit = dq[dq["severity"].isin(["CRITICAL", "WARNING"])]
    for _, r in crit.head(15).iterrows():
        L.append(f"- [{r['severity']}] {r['title']} — {r['columns']}: {r['n_rows']} rows. {r['recommendation']}")
    if len(crit) > 15:
        L.append(f"- … {len(crit) - 15} more critical/warning findings in DATA_QUALITY_REPORT.md")
    if f["n_d00_violating"]:
        L.append(f"- D-00: {f['n_d00_violating']:,} eligible labelled rows have a Last_Dx_Date / Last_Fall_Date on or after the index day "
                 f"(root cause {f['timing_summary']['root_cause']}); their prevalence {_pct(f['d00_prevalence'])} vs {_pct(f['clean_prevalence'])} on clean rows.")
    if f["wrong_type_columns"]:
        L.append(f"- Unparseable values (kept as NULL, never guessed): {f['wrong_type_columns']}.")
    L.append(f"- {int((pct > 90).sum())} columns are more than 90% missing and {int((pct > 50).sum())} more than 50% missing.")
    L += ["", "## What appears predictive (TRAIN only, univariate, descriptive)", "", f"**{UNIVARIATE_BANNER}.**", ""]
    top = _top_available(res)
    if len(top):
        L += ["Largest standardised differences between fallers and non-fallers among columns available at the prediction time:", ""]
        L += _md_table(top, ["column", "domain", "smd_fall_vs_no_fall", "univariate_auroc", "missing_pct", "available_at_prediction_time"])
        L += ["", "These describe single columns one at a time on the training partition. Correlated columns repeat the same signal; the multivariate "
              "model decides what adds information. None of these columns is selected because of this table."]
        blk = _blocked_signal(res)
        if len(blk):
            L += ["", "Columns that must never be predictors but show a strong association (a sign they carry post-index or retrospective information):", ""]
            L += _md_table(blk, ["column", "smd_fall_vs_no_fall", "available_at_prediction_time"])
    else:
        L.append("Not computed (no split): run with --reference <completed meuhedet-explore folder>.")
    L += ["", "## What is missing", ""]
    md = t["missingness_by_domain"].sort_values("mean_missing_pct", ascending=False)
    L.append("- Domains with the most missing values: " + "; ".join(f"{r['domain']} {r['mean_missing_pct']}%" for _, r in md.head(6).iterrows()) + ".")
    nm = mi[mi["dominant_class"] == "NOT_MEASURED"]
    L.append(f"- {len(nm)} columns are mostly 'not measured' (assessment-conditional nurse / frailty fields): NULL means not assessed, never 'no'.")
    L.append("- The eFalls 12-month ED/admission outcome and several mandatory eFalls predictors (fracture, fragility fracture) are not in this VIEW.")
    p2 = t["phase2_candidate_features"]
    L += ["", "## What to investigate next", ""]
    nxt = []
    if f["n_d00_violating"]:
        nxt.append("DWH: rebuild the predictor windows with Event_Date < Index_Date (D-00), then rerun this EDA and the models on every row.")
    if f["unknown_meaning_columns"]:
        nxt.append(f"SME: define the {len(f['unknown_meaning_columns'])} UNKNOWN columns ({', '.join(f['unknown_meaning_columns'][:8])}…) and confirm the NAME_ONLY meanings.")
    if f["wrong_type_columns"]:
        nxt.append("DWH: re-export the columns with unparseable values as ISO dates / plain numbers.")
    if len(p2):
        ready = p2[p2["readiness"].str.startswith("CANDIDATE")]
        nxt.append(f"Phase 2: {len(ready)} Meuhedet-native columns are candidates (eda/phase2_candidate_features.csv); none is trained yet.")
    red = t.get("redundancy_clusters")
    if red is not None and len(red):
        nxt.append(f"Redundancy: {len(red)} clusters of near-duplicate columns (eda/redundancy_clusters.csv) - choose one representative per cluster on TRAIN.")
    bal = t.get("split_balance")
    if bal is not None and len(bal) and bal["imbalance_flag"].fillna(False).any():
        nxt.append("Split: some variables differ by |SMD| > 0.1 between partitions - report it; do not re-split.")
    nxt.append("Blocked sources: confirm the as-of timing of hospitalisation / external-care fields before any use.")
    L += [f"- {x}" for x in nxt]
    L += ["", "Full report: REAL_DATA_EDA_REPORT.html. Data quality: DATA_QUALITY_REPORT.md. Tables: eda/*.csv (first line = watermark; read with skiprows=1).", ""]
    return "\n".join(L)


def render_dq_report(res: Any) -> str:
    t = res.tables
    dq = t["data_quality_checks"]
    L = ["# DATA QUALITY REPORT", "", f"**{res.watermark}**", ""]
    if res.synthetic:
        L += [f"**{SYNTHETIC}**", ""]
    L += [f"Input {res.input_file}; {res.facts['n_rows']:,} rows; basis FULL_EXTRACT unless stated. Nothing was repaired. Consistency rules are expected "
          "relations read from the column names and the S2T notes: a finding is a question for the DWH / SME, not a proven error. Counts below the "
          "small-cell threshold read '<min_cell'.", "",
          f"{len(dq)} checks: {int((dq['status'] == 'FINDING').sum())} with findings, {int((dq['status'] == 'PASS').sum())} passed, "
          f"{int((dq['status'] == 'NOT_APPLICABLE').sum())} not applicable.", ""]
    for sev in ("CRITICAL", "WARNING", "INFO"):
        d = dq[(dq["status"] == "FINDING") & (dq["severity"] == sev)]
        L += [f"## {sev} ({len(d)})", ""] + _md_table(d, ["check_id", "category", "title", "columns", "n_rows", "pct_rows", "detail", "recommendation"], max_rows=500) + [""]
    L += ["## Checks passed", ""] + [f"- {r['check_id']} {r['title']} ({r['columns']})" for _, r in dq[dq["status"] == "PASS"].iterrows()] + [""]
    na = dq[dq["status"] == "NOT_APPLICABLE"]
    L += ["## Not applicable", ""] + ([f"- {r['check_id']} {r['title']}: {r['detail']}" for _, r in na.iterrows()] or ["(none)"]) + [""]
    L += ["## D-00 and temporal findings", "", "See temporal_audit.csv: every date column against Index_Date with its leakage-risk class.", ""]
    ta = t["temporal_audit"]
    L += _md_table(ta[ta["leakage_risk"] == "PROVEN_INDEX_DAY_OR_FUTURE_RECORDS"], ["column", "n_equal_index", "n_after_index", "eligible_on_or_after_index",
                                                                                     "derived_aggregate_columns", "canonical_features", "models"])
    return "\n".join(L) + "\n"
