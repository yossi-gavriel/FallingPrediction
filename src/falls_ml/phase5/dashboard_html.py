"""The standalone Phase 5 operating dashboard: ONE self-contained HTML file (no server, no internet, no external script or font, no Python after
generation) that opens from the local file system in Chrome / Edge. It embeds AGGREGATE values only (counts per capacity point and model,
bootstrap summaries, feature-level importance) - never a patient, a pseudonym, a risk score or a date of a patient."""

from __future__ import annotations

import json
from typing import Any


def render(data: dict[str, Any]) -> str:
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", blob)


TEMPLATE = r"""<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 5 operating dashboard</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3f0; --surface-1: #fcfcfb; --surface-2: #f0efec; --line: #e2e1dc; --line-strong: #c9c8c2;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --accent: #2a78d6; --accent-wash: #e8f1fc; --target: #0b0b0b;
  --s-old: #2a78d6; --s-all: #eb6834; --s-safe: #1baf7a;
  --good: #1f7a3a; --bad: #b3261e; --warn-bg: #fff4d6; --warn-line: #e3b33d;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --surface-2: #232321; --line: #333330; --line-strong: #4a4a46;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #96958c;
    --accent: #3987e5; --accent-wash: #1d2a3b; --target: #ffffff;
    --s-old: #3987e5; --s-all: #d95926; --s-safe: #199e70;
    --good: #6fcf8a; --bad: #ff8a80; --warn-bg: #3a3018; --warn-line: #a8862c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --surface-2: #232321; --line: #333330; --line-strong: #4a4a46;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #96958c;
  --accent: #3987e5; --accent-wash: #1d2a3b; --target: #ffffff;
  --s-old: #3987e5; --s-all: #d95926; --s-safe: #199e70;
  --good: #6fcf8a; --bad: #ff8a80; --warn-bg: #3a3018; --warn-line: #a8862c;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
       font: 14px/1.45 "Segoe UI", system-ui, -apple-system, Arial, sans-serif; }
.wrap { max-width: 1280px; margin: 0 auto; padding: 16px; }
header h1 { font-size: 22px; margin: 0 0 4px; font-weight: 650; }
header .sub { color: var(--text-secondary); margin: 0; }
.banner { border: 1px solid var(--warn-line); background: var(--warn-bg); border-radius: 8px; padding: 8px 12px; margin: 10px 0; }
.card { background: var(--surface-1); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
.controls { display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: center; }
.ctl-label { font-size: 12px; color: var(--text-secondary); display: block; margin-bottom: 4px; }
.cap-value { font-size: 44px; font-weight: 650; line-height: 1; font-variant-numeric: tabular-nums; }
.cap-value small { font-size: 16px; color: var(--text-secondary); font-weight: 400; }
input[type=range] { width: min(520px, 90vw); accent-color: var(--accent); direction: ltr; }
.presets, .seg { display: flex; flex-wrap: wrap; gap: 6px; }
button { font: inherit; cursor: pointer; border: 1px solid var(--line-strong); background: var(--surface-1); color: var(--text-primary);
         border-radius: 999px; padding: 4px 12px; }
button:hover { background: var(--surface-2); }
button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
button.target { border-width: 2px; border-color: var(--text-primary); font-weight: 650; }
button.target::after { content: " ◆ יעד"; font-size: 11px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 10px; }
.kpi { background: var(--surface-1); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.kpi .lab { font-size: 12px; color: var(--text-secondary); }
.kpi .big { font-size: 28px; font-weight: 650; font-variant-numeric: tabular-nums; }
.kpi .rows { display: grid; grid-template-columns: 1fr auto; gap: 2px 10px; margin-top: 4px; font-variant-numeric: tabular-nums; }
.kpi .rows b { white-space: nowrap; direction: ltr; unicode-bidi: isolate; text-align: end; }
.kpi .subv { font-size: 12px; color: var(--text-secondary); }
.key { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-inline-end: 6px; }
.k-OLD { background: var(--s-old); } .k-ALL { background: var(--s-all); } .k-SAFE { background: var(--s-safe); }
.delta-pos { color: var(--good); font-weight: 650; } .delta-neg { color: var(--bad); font-weight: 650; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--line); text-align: start; vertical-align: top; }
th { font-size: 12px; color: var(--text-secondary); font-weight: 600; }
td.num, th.num { text-align: end; direction: ltr; unicode-bidi: plaintext; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(520px, 100%), 1fr)); gap: 12px; }
.stack > div + div { margin-top: 14px; }
.chart h3 { font-size: 15px; margin: 0 0 2px; font-weight: 600; }
.chart .note { font-size: 12px; color: var(--text-muted); margin: 0 0 6px; }
.chart svg { width: 100%; height: auto; display: block; overflow: visible; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--text-secondary); margin: 4px 0; }
.tip { position: fixed; pointer-events: none; background: var(--surface-1); border: 1px solid var(--line-strong); border-radius: 8px;
       padding: 6px 10px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.12); display: none; z-index: 10; min-width: 150px; }
.tip b { font-variant-numeric: tabular-nums; }
.badge { display: inline-block; white-space: nowrap; font-size: 11px; border-radius: 999px; padding: 1px 8px; border: 1px solid var(--line-strong); color: var(--text-secondary); }
.badge.new { border-color: var(--s-all); } .badge.safe { border-color: var(--s-safe); } .badge.stable { border-color: var(--text-primary); color: var(--text-primary); }
.bar { height: 8px; border-radius: 0 4px 4px 0; background: var(--accent); min-width: 2px; }
details summary { cursor: pointer; color: var(--text-secondary); }
.muted { color: var(--text-muted); font-size: 12px; }
.tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
h2 { font-size: 17px; margin: 0 0 8px; font-weight: 650; }
.headline { font-size: 16px; }
.hidden { display: none; }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>אם ניתן להתערב רק אצל X% מהמטופלים – כמה מהנפילות נתפוס?</h1>
  <p class="sub" id="subtitle"></p>
</header>
<div class="banner hidden" id="synthBanner"></div>
<div class="banner hidden" id="pooledBanner">תצוגת OOF מאוחד – <b>לתכנון / תיאור בלבד</b>: הסיכונים של מודלים חיצוניים שונים דורגו יחד. האומדן הראשי הוא חישוב הקיבולת לפי קפלים חיצוניים.</div>

<section class="card">
  <div class="controls">
    <div>
      <span class="ctl-label">קיבולת התערבות (INTERVENTION CAPACITY)</span>
      <div class="cap-value"><span id="capValue">3.0%</span> <small id="capSel"></small></div>
    </div>
    <div>
      <label class="ctl-label" for="cap">גרור לבחירת אחוז האוכלוסייה שניתן להתערב אצלו</label>
      <input type="range" id="cap" aria-label="intervention capacity">
      <div class="presets" id="presets"></div>
    </div>
  </div>
  <div class="controls" style="margin-top:12px">
    <div>
      <span class="ctl-label">משפחת מודל</span>
      <div class="seg" id="famSeg"></div>
    </div>
    <div>
      <span class="ctl-label">שיטת חישוב</span>
      <div class="seg" id="methodSeg"></div>
    </div>
  </div>
</section>

<section>
  <div class="kpis" id="kpis"></div>
</section>

<section class="card">
  <h2>השוואה בקיבולת הנבחרת</h2>
  <p class="muted" id="fixedNote"></p>
  <div style="overflow-x:auto"><table id="metricTable"></table></div>
  <div id="top3Box" class="muted" style="margin-top:8px"></div>
</section>

<section class="grid2">
  <div class="card chart" id="chA"></div>
  <div class="card chart" id="chB"></div>
  <div class="card chart" id="chC"></div>
  <div class="card chart" id="chD"></div>
</section>

<section class="card">
  <h2>מצב משני: יעד תפיסת נפילות</h2>
  <p class="muted">כמה קיבולת התערבות נדרשת כדי לתפוס לפחות X% מהנפילות? (נקרא מעקומת הקיבולת לפי קפלים; משני בלבד – הבקר הראשי הוא קיבולת ההתערבות)</p>
  <div class="seg" id="captureSeg"></div>
  <div style="overflow-x:auto;margin-top:8px"><table id="captureTable"></table></div>
</section>

<section class="card">
  <h2>מה מניע את המודל?</h2>
  <p class="muted" id="drvNote"></p>
  <div class="stack">
    <div><h3 style="font-size:15px;margin:4px 0">פיצ'רים ישנים (OLD) מובילים</h3><div style="overflow-x:auto"><table id="drvOld"></table></div></div>
    <div><h3 style="font-size:15px;margin:4px 0">פיצ'רים חדשים של V21</h3><div style="overflow-x:auto"><table id="drvNew"></table></div></div>
  </div>
</section>

<section class="card">
  <details>
    <summary>שיטה, פרטיות והערות</summary>
    <ul id="methodList" class="muted"></ul>
  </details>
  <details>
    <summary>טבלת העקומה המלאה (כל נקודות הקיבולת)</summary>
    <div style="overflow-x:auto"><table id="fullTable"></table></div>
  </details>
</section>
<div class="tip" id="tip" role="status"></div>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const SETS = D.sets;                         // [OLD, ALL, SAFE] raw names
const SK = {}; SETS.forEach((s, i) => { SK[s] = ["OLD", "ALL", "SAFE"][i]; });
const SLAB = {OLD: "OLD (ישן)", ALL: "OLD + כל החדשים", SAFE: "OLD + חדשים בטוחים"};
const KLAB = {OLD: "OLD", ALL: "כל החדשים", SAFE: "חדשים בטוחים"};
const FAMLAB = {ENET: "Elastic Net", LASSO: "LASSO", XGB: "XGBoost"};
const N = D.n, E = D.events;
const state = {pm: D.default_permille, fam: D.primary_family, method: "fold", capture: 0.7};
const $ = (id) => document.getElementById(id);
const el = (tag, attrs, text) => { const e = document.createElement(tag); if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]); if (text !== undefined) e.textContent = text; return e; };
const svgEl = (tag, attrs) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmtInt = (v) => (v === null || !isFinite(v)) ? "—" : Math.round(v).toLocaleString("en-US");
const fmtPct = (v, d) => (v === null || !isFinite(v)) ? "—" : (100 * v).toFixed(d === undefined ? 1 : d) + "%";
const fmtNum = (v, d) => (v === null || !isFinite(v)) ? "—" : v.toFixed(d);
const sgn = (v, f) => (v === null || !isFinite(v)) ? "—" : (v > 0 ? "+" : (v < 0 ? "−" : "±")) + f(Math.abs(v));
const T = (pm) => Math.floor((2 * pm * N + 1000) / 2000);

function series(fam, setKey) {
  const s = SETS[["OLD", "ALL", "SAFE"].indexOf(setKey)];
  const src = state.method === "pooled" ? D.tp_pooled : D.tp;
  return (src[fam] || {})[s] || null;
}
function grid() { return state.method === "pooled" ? D.pooled_grid : D.grid; }
function idxOf(pm) { const g = grid(); let best = 0; for (let i = 0; i < g.length; i++) if (Math.abs(g[i] - pm) < Math.abs(g[best] - pm)) best = i; return best; }
function metrics(tp, t) {
  if (tp === null || tp === undefined) return null;
  const fp = t - tp, fn = E - tp, tn = N - E - fp;
  return {sel: t, tp: tp, fp: fp, fn: fn, tn: tn, sens: tp / E, ppv: t ? tp / t : NaN, fas: t ? fp / t : NaN, spec: tn / (N - E), fpr: fp / (N - E),
          lift: t ? (tp / t) / (E / N) : NaN, nni: tp ? t / tp : NaN, cap1000: t ? 1000 * tp / t : NaN, false1000: t ? 1000 * fp / t : NaN};
}
function setsAvail(fam) { return ["OLD", "ALL", "SAFE"].filter((k) => series(fam, k)); }

// ---------------------------------------------------------------- controls
function buildControls() {
  const r = $("cap");
  r.min = D.grid[0]; r.max = D.grid[D.grid.length - 1]; r.step = D.grid_step; r.value = state.pm;
  r.addEventListener("input", () => { state.pm = D.grid[idxOfFine(+r.value)]; render(); });
  const pr = $("presets");
  D.presets.forEach((pm) => {
    if (D.grid.indexOf(pm) < 0) return;
    const b = el("button", {type: "button", "data-pm": pm}, (pm / 10).toString() + "%");
    if (pm === D.target_permille) b.classList.add("target");
    b.addEventListener("click", () => { state.pm = pm; r.value = pm; render(); });
    pr.appendChild(b);
  });
  D.families.forEach((f) => {
    const b = el("button", {type: "button", "data-fam": f}, FAMLAB[f] + (f === D.primary_family ? " · מודל ראשי" : ""));
    b.addEventListener("click", () => { state.fam = f; render(); });
    $("famSeg").appendChild(b);
  });
  [["fold", "לפי קפלים חיצוניים (ראשי)"], ["pooled", "OOF מאוחד (תיאורי בלבד)"]].forEach(([m, lab]) => {
    const b = el("button", {type: "button", "data-m": m}, lab);
    b.addEventListener("click", () => { state.method = m; render(); });
    $("methodSeg").appendChild(b);
  });
  D.capture_targets.forEach((t) => {
    const b = el("button", {type: "button", "data-c": t}, "לתפוס " + Math.round(100 * t) + "% מהנפילות");
    b.addEventListener("click", () => { state.capture = t; render(); });
    $("captureSeg").appendChild(b);
  });
}
function idxOfFine(pm) { let best = 0; for (let i = 0; i < D.grid.length; i++) if (Math.abs(D.grid[i] - pm) < Math.abs(D.grid[best] - pm)) best = i; return best; }

// ---------------------------------------------------------------- KPI cards and the metric table
function kpis() {
  const box = $("kpis"); box.textContent = "";
  const i = idxOf(state.pm), pm = grid()[i], t = T(pm), avail = setsAvail(state.fam);
  const m = {}; avail.forEach((k) => { m[k] = metrics(series(state.fam, k)[i], t); });
  const card = (lab, big, rows, sub) => {
    const c = el("div", {class: "kpi"}); c.appendChild(el("div", {class: "lab"}, lab));
    if (big) c.appendChild(el("div", {class: "big"}, big));
    if (sub) c.appendChild(el("div", {class: "subv"}, sub));
    if (rows) { const g = el("div", {class: "rows"}); rows.forEach(([k, v]) => { const a = el("span"); a.appendChild(el("span", {class: "key k-" + k})); a.appendChild(document.createTextNode(KLAB[k])); g.appendChild(a); g.appendChild(el("b", null, v)); }); c.appendChild(g); }
    box.appendChild(c);
  };
  card("קיבולת התערבות", (pm / 10).toFixed(1) + "%", null, "אחוז התערבות בפועל " + fmtPct(t / N, 2));
  card("מטופלים שנבחרו להתערבות", fmtInt(t), null, "מתוך " + fmtInt(N) + " מטופלים");
  card("נפילות שנתפסו (TP)", null, avail.map((k) => [k, fmtInt(m[k].tp) + " / " + fmtInt(E)]), "מתוך " + fmtInt(E) + " נפילות");
  card("שיעור תפיסה (Recall)", null, avail.map((k) => [k, fmtPct(m[k].sens)]));
  card("PPV (דיוק)", null, avail.map((k) => [k, fmtPct(m[k].ppv)]));
  card("התערבויות מיותרות (FP)", null, avail.map((k) => [k, fmtInt(m[k].fp)]));
  return {i, pm, t, m, avail};
}
function metricTable(ctx) {
  const tb = $("metricTable"); tb.textContent = "";
  const {m, avail, t, pm} = ctx;
  const head = el("tr"); head.appendChild(el("th", null, "מדד"));
  avail.forEach((k) => { const th = el("th", {class: "num"}); th.appendChild(el("span", {class: "key k-" + k})); th.appendChild(document.createTextNode(SLAB[k])); head.appendChild(th); });
  const dk = avail.filter((k) => k !== "OLD" && m.OLD);
  dk.forEach((k) => head.appendChild(el("th", {class: "num"}, "Δ " + (k === "ALL" ? "כל החדשים" : "חדשים בטוחים") + " − OLD")));
  tb.appendChild(head);
  const rows = [
    ["אוכלוסייה", () => N, fmtInt, null], ["נבחרו להתערבות", (x) => x.sel, fmtInt, null], ["אחוז התערבות בפועל", (x) => x.sel / N, (v) => fmtPct(v, 2), null],
    ["סה\"כ נפילות", () => E, fmtInt, null], ["נפילות שנתפסו (TP)", (x) => x.tp, fmtInt, 1], ["נפילות שהוחמצו (FN)", (x) => x.fn, fmtInt, -1],
    ["רגישות / Recall", (x) => x.sens, fmtPct, 1], ["PPV / Precision", (x) => x.ppv, fmtPct, 1],
    ["התערבויות מיותרות (FP)", (x) => x.fp, fmtInt, -1], ["שיעור התראות שווא", (x) => x.fas, fmtPct, -1], ["סגוליות (Specificity)", (x) => x.spec, (v) => fmtPct(v, 2), 1],
    ["שיעור חיוביים כוזבים (FPR)", (x) => x.fpr, (v) => fmtPct(v, 2), -1], ["Lift", (x) => x.lift, (v) => fmtNum(v, 2), 1],
    ["מספר התערבויות לכל נפילה שנתפסה (NNI)", (x) => x.nni, (v) => fmtNum(v, 1), -1],
    ["נפילות שנתפסו לכל 1,000 התערבויות", (x) => x.cap1000, (v) => fmtNum(v, 1), 1], ["התערבויות מיותרות לכל 1,000 התערבויות", (x) => x.false1000, (v) => fmtNum(v, 1), -1]];
  rows.forEach(([lab, f, fmt, good]) => {
    const tr = el("tr"); tr.appendChild(el("td", null, lab));
    avail.forEach((k) => tr.appendChild(el("td", {class: "num"}, fmt(f(m[k])))));
    dk.forEach((k) => {
      const td = el("td", {class: "num"});
      if (good !== null) {
        const d = f(m[k]) - f(m.OLD);
        const isPct = fmt === fmtPct || /%/.test(fmt(0.5));
        td.textContent = isPct ? sgn(d, (v) => (100 * v).toFixed(2) + " נ\"א") : sgn(d, (v) => (fmt === fmtInt ? fmtInt(v) : v.toFixed(2)));
        if (d * good > 0) td.className = "num delta-pos"; else if (d * good < 0) td.className = "num delta-neg";
      } else td.textContent = "";
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  $("fixedNote").textContent = "בקיבולת קבועה מספר ההתערבויות זהה לכל המודלים (" + fmtInt(t) + "), ולכן כל נפילה נוספת שנתפסת היא בדיוק התערבות מיותרת אחת פחות: יותר TP = פחות FP.";
  const b = $("top3Box"); b.textContent = "";
  const t3 = (D.top3[state.fam] || {});
  if (pm === D.target_permille && state.method === "fold" && Object.keys(t3).length) {
    b.appendChild(el("div", null, "ב-3% (חישוב מדויק) – רווח סמך 95% מזווג (" + D.n_boot.toLocaleString("en-US") + " דגימות bootstrap; הבחירה חושבה מחדש בכל קפל):"));
    const tb = el("table"); const h = el("tr");
    ["השוואה", "נפילות נוספות שנתפסו", "רווח סמך 95%", "קפלים עם יותר נפילות"].forEach((x, i) => h.appendChild(el("th", {class: i ? "num" : ""}, x))); tb.appendChild(h);
    Object.keys(t3).forEach((s) => { const r = t3[s]; const tr = el("tr");
      tr.appendChild(el("td", null, (SK[s] === "ALL" ? "OLD + כל החדשים" : "OLD + חדשים בטוחים") + " מול OLD"));
      tr.appendChild(el("td", {class: "num"}, sgn(r.d, fmtInt)));
      tr.appendChild(el("td", {class: "num"}, "[" + sgn(r.lo, fmtInt) + ", " + sgn(r.hi, fmtInt) + "]"));
      tr.appendChild(el("td", {class: "num"}, r.folds_better + " / " + D.folds)); tb.appendChild(tr); });
    b.appendChild(tb);
  } else if (state.method === "fold") {
    b.textContent = "רווחי סמך מזווגים מחושבים לנקודת היעד 3% (ראו TOP3_CAPACITY_COMPARISON.csv).";
  }
}

// ---------------------------------------------------------------- charts
const tip = $("tip");
function lineChart(boxId, title, note, lines, yfmt, opts) {
  const box = $(boxId); box.textContent = "";
  box.appendChild(el("h3", null, title)); box.appendChild(el("p", {class: "note"}, note));
  const lg = el("div", {class: "legend"});
  lines.forEach((ln) => { const s = el("span"); s.appendChild(el("span", {class: "key", style: "background:" + ln.color})); s.appendChild(document.createTextNode(ln.name)); lg.appendChild(s); });
  box.appendChild(lg);
  const W = 560, H = 300, ml = 62, mr = 96, mt = 12, mb = 40;
  const g = grid(); const xs = g.map((v) => v / 10);
  const vals = lines.flatMap((ln) => ln.values.filter((v) => v !== null && isFinite(v)));
  let y0 = Math.min(0, ...vals), y1 = Math.max(...vals, 0);
  if (opts && opts.yMin !== undefined) y0 = opts.yMin;
  if (y1 === y0) y1 = y0 + 1;
  const step = niceStep((y1 - y0) / 5); y0 = Math.floor(y0 / step) * step; y1 = Math.ceil(y1 / step) * step;
  const X = (v) => ml + (v - xs[0]) / (xs[xs.length - 1] - xs[0]) * (W - ml - mr);
  const Y = (v) => mt + (1 - (v - y0) / (y1 - y0)) * (H - mt - mb);
  const svg = svgEl("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title, direction: "ltr"});
  const gridC = css("--line"), txt = css("--text-secondary");
  for (let v = y0; v <= y1 + 1e-9; v += step) {
    svg.appendChild(svgEl("line", {x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), stroke: Math.abs(v) < 1e-12 ? css("--line-strong") : gridC, "stroke-width": 1}));
    const t = svgEl("text", {x: ml - 6, y: Y(v) + 4, "text-anchor": "end", "font-size": 13, fill: txt}); t.textContent = yfmt(v); svg.appendChild(t);
  }
  [0.5, 3, 5, 10, 15, 20].filter((v) => v >= xs[0] && v <= xs[xs.length - 1]).forEach((v) => {
    const t = svgEl("text", {x: X(v), y: H - mb + 16, "text-anchor": "middle", "font-size": 13, fill: txt}); t.textContent = v + "%"; svg.appendChild(t);
  });
  const xl = svgEl("text", {x: (ml + W - mr) / 2, y: H - 4, "text-anchor": "middle", "font-size": 13, fill: txt}); xl.textContent = "capacity – % of population selected"; svg.appendChild(xl);
  if (D.target_permille / 10 >= xs[0]) {
    const xt = X(D.target_permille / 10);
    svg.appendChild(svgEl("line", {x1: xt, x2: xt, y1: mt, y2: H - mb, stroke: css("--target"), "stroke-width": 1.5}));
    const tt = svgEl("text", {x: xt + 4, y: mt + 10, "font-size": 13, fill: css("--text-primary"), "font-weight": 600}); tt.textContent = "3% target"; svg.appendChild(tt);
  }
  const xc = X(g[idxOf(state.pm)] / 10);
  svg.appendChild(svgEl("line", {x1: xc, x2: xc, y1: mt, y2: H - mb, stroke: css("--accent"), "stroke-width": 1, opacity: 0.8}));
  const ends = [];
  lines.forEach((ln) => {
    let d = ""; ln.values.forEach((v, i) => { if (v === null || !isFinite(v)) return; d += (d ? "L" : "M") + X(xs[i]).toFixed(1) + "," + Y(v).toFixed(1); });
    svg.appendChild(svgEl("path", {d, fill: "none", stroke: ln.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"}));
    const ci = idxOf(state.pm), v = ln.values[ci];
    if (v !== null && isFinite(v)) svg.appendChild(svgEl("circle", {cx: X(xs[ci]), cy: Y(v), r: 4.5, fill: ln.color, stroke: css("--surface-1"), "stroke-width": 2}));
    const li = ln.values.length - 1; if (ln.values[li] !== null) ends.push({y: Y(ln.values[li]), name: ln.short || ln.name});
  });
  ends.sort((a, b) => a.y - b.y); for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
  ends.forEach((e) => { const t = svgEl("text", {x: W - mr + 6, y: e.y + 4, "font-size": 13, fill: css("--text-primary")}); t.textContent = e.name; svg.appendChild(t); });
  const hair = svgEl("line", {x1: 0, x2: 0, y1: mt, y2: H - mb, stroke: css("--text-muted"), "stroke-width": 1, visibility: "hidden"}); svg.appendChild(hair);
  const hit = svgEl("rect", {x: ml, y: mt, width: W - ml - mr, height: H - mt - mb, fill: "transparent", style: "cursor:crosshair"});
  const near = (evt) => { const r = svg.getBoundingClientRect(); const px = (evt.clientX - r.left) / r.width * W; let b = 0; xs.forEach((x, i) => { if (Math.abs(X(x) - px) < Math.abs(X(xs[b]) - px)) b = i; }); return b; };
  hit.addEventListener("pointermove", (evt) => {
    const i = near(evt); hair.setAttribute("x1", X(xs[i])); hair.setAttribute("x2", X(xs[i])); hair.setAttribute("visibility", "visible");
    tip.textContent = ""; tip.appendChild(el("div", {class: "muted"}, xs[i].toFixed(1) + "% · " + fmtInt(T(g[i])) + " מטופלים"));
    lines.forEach((ln) => { const row = el("div"); row.appendChild(el("span", {class: "key", style: "background:" + ln.color})); row.appendChild(el("b", null, yfmt(ln.values[i], true))); row.appendChild(document.createTextNode(" " + ln.name)); tip.appendChild(row); });
    tip.style.display = "block"; tip.style.left = Math.min(evt.clientX + 14, window.innerWidth - 230) + "px"; tip.style.top = (evt.clientY + 14) + "px";
  });
  hit.addEventListener("pointerleave", () => { hair.setAttribute("visibility", "hidden"); tip.style.display = "none"; });
  hit.addEventListener("click", (evt) => { const pm = g[near(evt)]; if (D.grid.indexOf(pm) >= 0) { state.pm = pm; $("cap").value = pm; render(); } });
  svg.appendChild(hit);
  box.appendChild(svg);
}
function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9)))); const f = raw / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; }

function charts() {
  const g = grid(), fam = state.fam, avail = setsAvail(fam);
  const col = {OLD: css("--s-old"), ALL: css("--s-all"), SAFE: css("--s-safe")};
  const short = {OLD: "OLD", ALL: "ALL NEW", SAFE: "NEW SAFE"};
  const mk = (f) => avail.map((k) => ({name: SLAB[k], short: short[k], color: col[k], values: series(fam, k).map((tp, i) => { const x = metrics(tp, T(g[i])); return x ? f(x) : null; })}));
  const pctF = (v, tip) => (v === null || !isFinite(v)) ? "—" : (100 * v).toFixed(tip ? 1 : 0) + "%";
  const intF = (v) => fmtInt(v);
  const tag = state.method === "pooled" ? " (OOF מאוחד – תיאורי)" : "";
  lineChart("chA", "A. קיבולת מול % הנפילות שנתפסו" + tag, "Recall: חלק הנפילות שנתפסו בקרב המטופלים שנבחרו", mk((x) => x.sens), pctF);
  lineChart("chB", "B. קיבולת מול PPV" + tag, "PPV: חלק המטופלים שנבחרו שאכן נפלו", mk((x) => x.ppv), pctF);
  lineChart("chC", "C. קיבולת מול התערבויות מיותרות (FP)" + tag, "מטופלים שנבחרו ולא נפלו", mk((x) => x.fp), intF);
  const dl = avail.filter((k) => k !== "OLD").map((k) => ({name: (k === "ALL" ? "כל החדשים" : "חדשים בטוחים") + " − OLD", short: short[k] + " − OLD", color: col[k],
    values: series(fam, k).map((tp, i) => { const o = series(fam, "OLD")[i]; return (tp === null || o === null) ? null : tp - o; })}));
  lineChart("chD", "D + E. תועלת מוספת של V21 מול OLD" + tag, "נפילות נוספות שנתפסו = התערבויות מיותרות שנחסכו (בקיבולת קבועה הם שווים בדיוק); ערך שלילי = OLD טוב יותר",
            dl, (v) => (v === null || !isFinite(v)) ? "—" : (v > 0 ? "+" : "") + Math.round(v).toLocaleString("en-US"), {});
}

// ---------------------------------------------------------------- secondary: target fall capture
function captureTable() {
  const tb = $("captureTable"); tb.textContent = "";
  const fam = state.fam, avail = ["OLD", "ALL", "SAFE"].filter((k) => (D.tp[fam] || {})[SETS[["OLD", "ALL", "SAFE"].indexOf(k)]]);
  const head = el("tr"); ["", "קיבולת נדרשת", "מטופלים להתערבות", "נפילות שנתפסו", "PPV", "התערבויות מיותרות"].forEach((h, i) => head.appendChild(el("th", {class: i ? "num" : ""}, h))); tb.appendChild(head);
  avail.forEach((k) => {
    const s = SETS[["OLD", "ALL", "SAFE"].indexOf(k)];
    const gg = D.grid.concat(D.ext_grid), tps = D.tp[fam][s].concat((D.tp_ext[fam] || {})[s] || []);
    let hit = -1; for (let i = 0; i < gg.length; i++) if (tps[i] !== null && tps[i] / E >= state.capture - 1e-12) { hit = i; break; }
    const tr = el("tr"); const c0 = el("td"); c0.appendChild(el("span", {class: "key k-" + k})); c0.appendChild(document.createTextNode(SLAB[k])); tr.appendChild(c0);
    if (hit < 0) { tr.appendChild(el("td", {class: "num", colspan: 5}, "לא הושג בטווח המחושב")); tb.appendChild(tr); return; }
    const x = metrics(tps[hit], T(gg[hit]));
    [(gg[hit] / 10).toFixed(1) + "%", fmtInt(x.sel), fmtInt(x.tp) + " (" + fmtPct(x.sens) + ")", fmtPct(x.ppv), fmtInt(x.fp)].forEach((v) => tr.appendChild(el("td", {class: "num"}, v)));
    tb.appendChild(tr);
  });
  document.querySelectorAll("#captureSeg button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.c === state.capture)));
}

// ---------------------------------------------------------------- what drives the model
function drivers() {
  const d = D.drivers[state.fam];
  $("drvNote").textContent = d ? ("מדד הדירוג: " + d.measure + ". מבוסס על המודל OLD + כל החדשים של " + FAMLAB[state.fam] + ". אסוציאציה עם תחזית המודל – לא סיבתיות.") : "אין תוצרי הסבר זמינים למשפחה זו.";
  const fill = (id, rows, isNew) => {
    const tb = $(id); tb.textContent = ""; if (!d) return;
    const head = el("tr"); const hs = isNew ? ["דירוג כללי", "פיצ'ר", "תחום", "מעמד", "חשיבות", "יציבות"] : ["דירוג כללי", "פיצ'ר", "חשיבות", "יציבות"];
    hs.forEach((h) => head.appendChild(el("th", null, h))); tb.appendChild(head);
    const mx = Math.max(...d.old.concat(d.new).map((r) => r.importance || 0), 1e-12);
    rows.forEach((r) => {
      const tr = el("tr");
      tr.appendChild(el("td", {class: "num"}, String(r.rank)));
      const c = el("td"); c.appendChild(document.createTextNode(r.label + " ")); c.appendChild(el("span", {class: "badge" + (isNew ? " new" : "")}, isNew ? "NEW FEATURE" : "OLD FEATURE"));
      if (r.direction) c.appendChild(el("div", {class: "muted"}, r.direction === "higher risk" ? "↑ קשור לסיכון גבוה יותר בתחזית" : "↓ קשור לסיכון נמוך יותר בתחזית"));
      tr.appendChild(c);
      if (isNew) { tr.appendChild(el("td", null, r.domain || "")); const st = el("td"); st.appendChild(el("span", {class: "badge " + (r.status === "SAFE" ? "safe" : "")}, r.status === "SAFE" ? "SAFE" : "ALL_NEW_ONLY")); tr.appendChild(st); }
      const imp = el("td"); const bar = el("div", {class: "bar", style: "width:" + Math.max(2, 120 * (r.importance || 0) / mx).toFixed(0) + "px"}); imp.appendChild(bar); imp.appendChild(el("div", {class: "muted"}, r.importance === null ? "—" : r.importance.toPrecision(3))); tr.appendChild(imp);
      const sb = el("td"); sb.appendChild(document.createTextNode(r.stability || "—")); if (r.stable) { sb.appendChild(document.createTextNode(" ")); sb.appendChild(el("span", {class: "badge stable"}, "יציב")); } tr.appendChild(sb);
      tb.appendChild(tr);
    });
  };
  fill("drvOld", d ? d.old : [], false); fill("drvNew", d ? d.new : [], true);
}

function fullTable() {
  const tb = $("fullTable"); tb.textContent = "";
  const fam = state.fam, avail = setsAvail(fam), g = grid();
  const head = el("tr"); head.appendChild(el("th", {class: "num"}, "קיבולת")); head.appendChild(el("th", {class: "num"}, "נבחרו"));
  avail.forEach((k) => { ["TP", "Recall", "PPV", "FP"].forEach((h) => head.appendChild(el("th", {class: "num"}, short(k) + " " + h))); }); tb.appendChild(head);
  g.forEach((pm, i) => { const tr = el("tr"); tr.appendChild(el("td", {class: "num"}, (pm / 10).toFixed(1) + "%")); tr.appendChild(el("td", {class: "num"}, fmtInt(T(pm))));
    avail.forEach((k) => { const x = metrics(series(fam, k)[i], T(pm)); [fmtInt(x.tp), fmtPct(x.sens), fmtPct(x.ppv), fmtInt(x.fp)].forEach((v) => tr.appendChild(el("td", {class: "num"}, v))); }); tb.appendChild(tr); });
  function short(k) { return {OLD: "OLD", ALL: "ALL NEW", SAFE: "NEW SAFE"}[k]; }
}

function render() {
  document.querySelectorAll("#presets button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.pm === state.pm)));
  document.querySelectorAll("#famSeg button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.fam === state.fam)));
  document.querySelectorAll("#methodSeg button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.m === state.method)));
  $("pooledBanner").classList.toggle("hidden", state.method !== "pooled");
  const pm = grid()[idxOf(state.pm)];
  $("capValue").textContent = (pm / 10).toFixed(1) + "%";
  $("capSel").textContent = fmtInt(T(pm)) + " מטופלים מתוך " + fmtInt(N);
  const ctx = kpis(); metricTable(ctx); charts(); captureTable(); drivers(); fullTable();
}

// ---------------------------------------------------------------- static text
$("subtitle").textContent = D.subtitle;
if (D.synthetic) { $("synthBanner").textContent = D.synthetic_text; $("synthBanner").classList.remove("hidden"); }
D.method_notes.forEach((t) => $("methodList").appendChild(el("li", null, t)));
buildControls();
render();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
})();
</script>
</body>
</html>
"""
