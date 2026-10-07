"""PRE vs POST (Phase 5.1 R-10 / R-11 / section 7 of docs/phase6/EXPERIMENT1_IMPLEMENTATION_CONTRACT.md): the completed Phase 5 2.2.0 run (PRE)
against the repair-only correction (POST) on IDENTICAL patients, labels and outer folds - analysis / reporting only, nothing is fitted.

The headline is ABSOLUTE and at the 3% operating capacity: for OLD and for ADMISSIBLE (OLD_PLUS_NEW_SAFE), the exact number selected, the falls
captured, Recall@Top3, PPV@Top3, the false interventions, and the PRE -> POST difference in captured falls with a PAIRED 95% patient-bootstrap
interval (identical multinomial weights for PRE and POST; the capacity re-allocated across the adopted folds from the resampled fold sizes and the
top-k re-selected within each fold in every replicate - the ``capacity.py`` mechanics). Captured per 10,000 patients is reported IN ADDITION, never
instead. PRE / POST reports the COMBINED effect of the repair; no attribution to a single change is made.

Secondary: the ADMISSIBLE-vs-OLD contrast in PRE and in POST; legacy ALL (OLD_PLUS_ALL_NEW_ELIGIBLE) as AUDIT only; NO_NEW_REGISTRY as a SECONDARY
DIAGNOSTIC in POST only; the 2.2.0 70%-rule metrics as a historical audit; the membership differences; precision planning for the later MOD decision.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase5.analysis import ModelRun
from falls_ml.phase5.capacity import Ranked, allocate, capacity_bootstrap, fold_capacity, n_selected, rank_model
from falls_ml.phase5.config import PRIMARY_FAMILY, SET_ALL, SET_OLD, SET_SAFE
from falls_ml.phase5.metrics import model_summary, paired_bootstrap

PERMILLE = 30
MOD_PLACEHOLDER_PER_10000 = 5.0        # planning placeholder only (docs/phase6/FINAL_CONSENSUS.md 4.6); management declares the MOD before Experiment 5
ARMS = ((SET_OLD, "OLD"), (SET_SAFE, "ADMISSIBLE"), (SET_ALL, "LEGACY_ALL_AUDIT"))
PRIMARY_ARMS = (SET_OLD, SET_SAFE)


def pre_model_runs(pre: Path, plan: dict[str, Any], *, keys_post: np.ndarray, outer_post: np.ndarray, targets: list[float]) -> dict[str, ModelRun]:
    """The PRE ENET runs (outer-OOF risks, the PRE inner-selected 70% flags, fold rows) aligned to the POST row order."""
    from falls_ml.phase5.prerun import pre_oof

    pre_plan = __import__("json").loads((Path(pre) / "work" / "PLAN.json").read_text(encoding="utf-8"))
    raw = pre_oof(Path(pre), pre_plan, keys_post=keys_post, outer_post=outer_post, families=(PRIMARY_FAMILY,), targets=tuple(targets))
    out = {}
    for (fam, s), r in raw.items():
        out[s] = ModelRun(family=fam, setname=s, canonical=r["canonical"], p=r["p"], flags=r["flags"],
                          folds=[{"outer": f["outer"], "test_idx": f["test_idx"], "thresholds": f["thresholds"], "config": f["config"]} for f in r["folds"]])
    return out


def _cap(y: np.ndarray, outer: np.ndarray, r: Ranked, T: int) -> dict[str, Any]:
    c = fold_capacity(y, r, T)
    return {"selected_total": int(c["flagged"]), "captured_falls": int(c["tp"]), "recall_top3": c["sensitivity"], "ppv_top3": c["ppv"],
            "false_interventions": int(c["fp"]), "missed_falls": int(c["fn"]), "per_fold_selected": c["per_fold_selected"],
            "ties_split_at_boundary": c["ties_split_at_boundary"]}


def _selected_sets(r: Ranked, T: int) -> list[set[int]]:
    k = allocate(T, [len(f["idx"]) for f in r.folds])
    return [set(int(i) for i in f["idx"][: int(kf)]) for f, kf in zip(r.folds, k)]


def _overlap(y: np.ndarray, ra: Ranked, rb: Ranked, T: int) -> dict[str, Any]:
    """Within-fold selected-set overlap and the discordant captured fallers (captured by A only / by B only)."""
    sa, sb = _selected_sets(ra, T), _selected_sets(rb, T)
    inter = sum(len(a & b) for a, b in zip(sa, sb))
    fa = {i for s in sa for i in s if y[i] == 1}
    fb = {i for s in sb for i in s if y[i] == 1}
    return {"selected_overlap": int(inter), "selected_overlap_share": inter / T if T else math.nan, "fallers_captured_by_both": len(fa & fb),
            "fallers_captured_only_by_a": len(fa - fb), "fallers_captured_only_by_b": len(fb - fa)}


def _ci(col: np.ndarray) -> tuple[float, float]:
    col = col[np.isfinite(col)]
    return (float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))) if col.size else (math.nan, math.nan)


def analyse_prepost(y: np.ndarray, outer: np.ndarray, post: dict[str, ModelRun], pre: dict[str, ModelRun], *, n_boot: int, seed: int,
                    ablation_runs: dict[str, ModelRun] | None = None, target: float = 0.70, log: Any = print) -> dict[str, Any]:
    """All PRE / POST tables from the aligned outer-OOF predictions (POST = this run's units; PRE = the verified 2.2.0 units)."""
    y = np.asarray(y).astype(np.int64)
    outer = np.asarray(outer).astype(int)
    N, E = len(y), int(y.sum())
    T = n_selected(PERMILLE, N)
    ranked: dict[tuple[str, str], Ranked] = {}
    for s, _ in ARMS:
        if s in post:
            ranked[("POST", s)] = rank_model("POST", s, y, outer, post[s].p)
        if s in pre:
            ranked[("PRE", s)] = rank_model("PRE", s, y, outer, pre[s].p)
    for name, run in (ablation_runs or {}).items():
        ranked[("POST", name)] = rank_model("POST", name, y, outer, run.p)
    log(f"PRE / POST: paired bootstrap at {PERMILLE / 10:.1f}% ({n_boot} replicates, {len(ranked)} ranked arms, T = {T})")
    boot = capacity_bootstrap(y, outer, ranked, PERMILLE, n_boot=n_boot, seed=seed)
    tp = boot["tp"]
    point = {k: _cap(y, outer, r, T) for k, r in ranked.items()}

    def delta_rows(a: tuple[str, str], b: tuple[str, str], label: str, role: str) -> dict[str, Any]:
        pa, pb = point[a], point[b]
        d = tp[b] - tp[a]
        lo, hi = _ci(d)
        ov = _overlap(y, ranked[a], ranked[b], T)
        return {"comparison": label, "role": role, "population": N, "falls_total": E, "selected_total": T,
                "captured_a": pa["captured_falls"], "captured_b": pb["captured_falls"],
                "delta_captured_falls": pb["captured_falls"] - pa["captured_falls"], "delta_captured_falls_ci_low": lo, "delta_captured_falls_ci_high": hi,
                "delta_recall_top3": (pb["captured_falls"] - pa["captured_falls"]) / E if E else math.nan,
                "delta_recall_top3_ci_low": lo / E if E else math.nan, "delta_recall_top3_ci_high": hi / E if E else math.nan,
                "delta_ppv_top3": (pb["captured_falls"] - pa["captured_falls"]) / T if T else math.nan,
                "delta_false_interventions": -(pb["captured_falls"] - pa["captured_falls"]),
                "delta_captured_per_10000": 1e4 * (pb["captured_falls"] - pa["captured_falls"]) / N if N else math.nan,
                "delta_captured_per_10000_ci_low": 1e4 * lo / N if N else math.nan, "delta_captured_per_10000_ci_high": 1e4 * hi / N if N else math.nan,
                "bootstrap_sd_delta_captured": float(np.std(d[np.isfinite(d)], ddof=1)) if np.isfinite(d).sum() > 1 else math.nan,
                "share_replicates_b_better": float((d > 0).mean()), "share_replicates_equal": float((d == 0).mean()), **ov,
                "n_boot": int(n_boot), "seed": int(seed), "method": "paired patient bootstrap (identical multinomial weights); capacity re-allocated "
                "across the adopted outer folds and the top-k re-selected within each fold in every replicate; percentile 95% interval; conditional "
                "on the fitted models"}

    # ---- the headline: OLD and ADMISSIBLE, PRE vs POST (absolute numbers first)
    head, paired = [], []
    for s, lab in ARMS:
        role = "PRIMARY" if s in PRIMARY_ARMS else "AUDIT_ONLY"
        for ver in ("PRE", "POST"):
            if (ver, s) in point:
                c = point[(ver, s)]
                head.append({"arm": lab, "feature_set": s, "version": "PRE 2.2.0" if ver == "PRE" else "POST 5.1", "role": role, "population": N,
                             "falls_total": E, "selected_total": c["selected_total"], "captured_falls": c["captured_falls"], "recall_top3": c["recall_top3"],
                             "ppv_top3": c["ppv_top3"], "false_interventions": c["false_interventions"], "missed_falls": c["missed_falls"],
                             "captured_per_10000": 1e4 * c["captured_falls"] / N if N else math.nan, "ties_split_at_boundary": c["ties_split_at_boundary"]})
        if ("PRE", s) in point and ("POST", s) in point:
            d = delta_rows(("PRE", s), ("POST", s), f"{lab}: POST 5.1 minus PRE 2.2.0", role)
            head.append({"arm": lab, "feature_set": s, "version": "DELTA POST minus PRE", "role": role, "population": N, "falls_total": E,
                         "selected_total": T, "captured_falls": d["delta_captured_falls"], "recall_top3": d["delta_recall_top3"], "ppv_top3": d["delta_ppv_top3"],
                         "false_interventions": d["delta_false_interventions"], "missed_falls": -d["delta_captured_falls"],
                         "captured_per_10000": d["delta_captured_per_10000"], "delta_captured_falls_ci_low": d["delta_captured_falls_ci_low"],
                         "delta_captured_falls_ci_high": d["delta_captured_falls_ci_high"], "delta_recall_top3_ci_low": d["delta_recall_top3_ci_low"],
                         "delta_recall_top3_ci_high": d["delta_recall_top3_ci_high"], "ties_split_at_boundary": ""})
            paired.append({"arm": lab, "feature_set": s, **d})
    # ---- the contrasts: ADMISSIBLE vs OLD (PRE, POST), legacy ALL vs OLD (audit), NO_NEW_REGISTRY vs ADMISSIBLE (POST, secondary diagnostic)
    contrast = []
    for ver in ("PRE", "POST"):
        if (ver, SET_OLD) in point and (ver, SET_SAFE) in point:
            contrast.append({"version": "PRE 2.2.0" if ver == "PRE" else "POST 5.1", **delta_rows((ver, SET_OLD), (ver, SET_SAFE), "ADMISSIBLE minus OLD", "PRIMARY_CONTRAST")})
        if (ver, SET_OLD) in point and (ver, SET_ALL) in point:
            contrast.append({"version": "PRE 2.2.0" if ver == "PRE" else "POST 5.1", **delta_rows((ver, SET_OLD), (ver, SET_ALL), "LEGACY_ALL minus OLD", "AUDIT_ONLY")})
    for name in (ablation_runs or {}):
        if ("POST", SET_SAFE) in point:
            contrast.append({"version": "POST 5.1", **delta_rows(("POST", SET_SAFE), ("POST", name), f"{name} minus ADMISSIBLE", "SECONDARY_DIAGNOSTIC")})
    # ---- per-fold and pooled 70%-rule / discrimination metrics (historical audit) for every PRE / POST arm
    corr = []
    for s, lab in ARMS:
        for ver, runs in (("PRE", pre), ("POST", post)):
            if s not in runs:
                continue
            run = runs[s]
            key = f"{target:.2f}"
            k_f = allocate(T, [int((outer == f).sum()) for f in range(int(outer.max()) + 1)])
            for f in range(int(outer.max()) + 1):
                m = outer == f
                rk = rank_model(ver, s, y[m], np.zeros(int(m.sum()), dtype=int), run.p[m])
                c = fold_capacity(y[m], rk, int(k_f[f]))
                sm = model_summary(y[m].astype(float), run.p[m], run.flags[key][m] if key in run.flags else None, targets=(target,))
                corr.append({"arm": lab, "feature_set": s, "version": "PRE 2.2.0" if ver == "PRE" else "POST 5.1", "scope": f"fold {f}", "n": int(m.sum()),
                             "events": int(y[m].sum()), "selected_total": int(k_f[f]), "captured_falls": int(c["tp"]), "recall_top3": c["sensitivity"],
                             "ppv_top3": c["ppv"], "auroc": sm["auroc"], "ap": sm["ap"], "brier": sm["brier"], "calibration_slope": sm["calibration_slope"],
                             "calibration_intercept": sm["calibration_intercept"], "hist70_false_alert_share": sm.get("op_false_alert_share"),
                             "hist70_flagged_share": sm.get("op_flagged_share"), "hist70_sensitivity": sm.get("op_sensitivity")})
            c = point[(ver, s)]
            sm = model_summary(y.astype(float), run.p, run.flags[key] if key in run.flags else None, targets=(target,))
            corr.append({"arm": lab, "feature_set": s, "version": "PRE 2.2.0" if ver == "PRE" else "POST 5.1", "scope": "overall", "n": N, "events": E,
                         "selected_total": T, "captured_falls": c["captured_falls"], "recall_top3": c["recall_top3"], "ppv_top3": c["ppv_top3"],
                         "auroc": sm["auroc"], "ap": sm["ap"], "brier": sm["brier"], "calibration_slope": sm["calibration_slope"],
                         "calibration_intercept": sm["calibration_intercept"], "hist70_false_alert_share": sm.get("op_false_alert_share"),
                         "hist70_flagged_share": sm.get("op_flagged_share"), "hist70_sensitivity": sm.get("op_sensitivity")})
    # ---- paired PRE vs POST differences of the pooled discrimination / 70%-rule metrics (historical audit)
    hist = []
    for s, lab in ARMS:
        if s in pre and s in post:
            key = f"{target:.2f}"
            pb = paired_bootstrap(y.astype(float), pre[s].p, post[s].p, pre[s].flags.get(key), post[s].flags.get(key), n_boot=min(n_boot, 500),
                                  seed=seed + 7, targets=(target,))
            hist.append({"arm": lab, "feature_set": s, "delta_auroc": pb["delta_auroc"], "delta_auroc_ci_low": pb["delta_auroc_ci_low"],
                         "delta_auroc_ci_high": pb["delta_auroc_ci_high"], "delta_ap": pb["delta_ap"], "delta_brier": pb["delta_brier"],
                         "delta_hist70_false_alert_share": pb.get("delta_op_false_alert_share"), "delta_hist70_flagged_share": pb.get("delta_op_flagged_share"),
                         "note": "historical audit (2.2.0 70%-sensitivity rule); the headline is the Top-3% table"})
    # ---- precision planning (for the later MOD decision; not a result)
    prec = []
    pc = [c for c in contrast if c["comparison"] == "ADMISSIBLE minus OLD" and c["version"] == "POST 5.1"]
    if pc:
        c = pc[0]
        sd = c["bootstrap_sd_delta_captured"]
        hw = 1.96 * sd if math.isfinite(sd) else math.nan
        prec.append({"comparison": "ADMISSIBLE minus OLD (POST 5.1, development outer-OOF)", "population": N, "falls_total": E, "selected_total": T,
                     "delta_captured_falls": c["delta_captured_falls"], "bootstrap_sd_delta_captured": sd, "half_width_95_captured_falls": hw,
                     "half_width_95_per_10000": 1e4 * hw / N if N else math.nan, "selected_overlap_share": c["selected_overlap_share"],
                     "fallers_captured_only_by_old": c["fallers_captured_only_by_a"], "fallers_captured_only_by_admissible": c["fallers_captured_only_by_b"],
                     "mod_placeholder_per_10000": MOD_PLACEHOLDER_PER_10000, "mod_placeholder_captured_falls": MOD_PLACEHOLDER_PER_10000 * N / 1e4,
                     "point_estimate_needed_for_lower_limit_above_mod_placeholder_captured": MOD_PLACEHOLDER_PER_10000 * N / 1e4 + hw,
                     "note": "planning information for the MOD decision (docs/phase6/FINAL_CONSENSUS.md 4.6); the placeholder is NOT a declared MOD"})
    return {"headline": pd.DataFrame(head), "paired": pd.DataFrame(paired), "contrast": pd.DataFrame(contrast), "correction": pd.DataFrame(corr),
            "historical": pd.DataFrame(hist), "precision": pd.DataFrame(prec), "selected_total": T, "n": N, "events": E, "n_boot": int(n_boot)}


def membership_table(pre: Path, post_registry: pd.DataFrame, post_units: list[dict[str, Any]]) -> pd.DataFrame:
    """Every feature: PRE class / sets vs POST class / sets, and the reason for any difference (R-1 / R-2 per fold / R-4)."""
    pf = Path(pre) / "share" / "FEATURE_ELIGIBILITY.csv"
    pre_t = pd.read_csv(pf) if pf.is_file() else pd.DataFrame(columns=["feature", "class", "in_OLD", "in_OLD_PLUS_ALL_NEW_ELIGIBLE", "in_OLD_PLUS_NEW_SAFE"])
    pre_t = pre_t.set_index("feature") if "feature" in pre_t.columns else pre_t
    tf = lambda v: str(v).strip().lower() in ("true", "1")  # noqa: E731
    dropped: dict[str, list[str]] = {}
    for u in post_units:
        for f, why in (u.get("dropped_coverage") or {}).items():
            dropped.setdefault(f, []).append(f"{u.get('uid', '?')}: {why}")
    rows = []
    for _, r in post_registry.iterrows():
        f = str(r["feature"])
        pre_cls = str(pre_t.at[f, "class"]) if f in pre_t.index else "not in PRE registry"
        pre_sets = [s for s in (SET_OLD, SET_ALL, SET_SAFE) if f in pre_t.index and f"in_{s}" in pre_t.columns and tf(pre_t.at[f, f"in_{s}"])]
        post_sets = [s for s in (SET_OLD, SET_ALL, SET_SAFE) if tf(r.get(f"in_{s}", False))]
        why = []
        if pre_cls == "INELIGIBLE_LEAKAGE" and str(r["class"]) != "INELIGIBLE_LEAKAGE":
            why.append("R-1: the 2.2.0 outcome-dependent AUROC exclusion no longer applies (label-free gates only)")
        if str(r.get("override", "")):
            why.append(f"R-4: registered override '{r.get('override')}'")
        if f in dropped:
            why.append(f"R-2: dropped by the per-fold coverage gate in {len(dropped[f])} unit(s)")
        if pre_cls != str(r["class"]) and not why:
            why.append("class differs (see reason)")
        rows.append({"feature": f, "origin": r.get("origin", ""), "domain": r.get("domain", ""), "pre_class": pre_cls, "post_class": r["class"],
                     "pre_sets": "; ".join(pre_sets), "post_sets": "; ".join(post_sets), "changed": bool(pre_cls != str(r["class"]) or pre_sets != post_sets or f in dropped),
                     "reason": "; ".join(why) if why else "", "post_reason": r.get("reason", ""),
                     "per_fold_coverage_drops": "; ".join(dropped.get(f, []))[:500]})
    return pd.DataFrame(rows)


# ============================================================================ the headline text (first in both summaries)
def _n(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if not math.isfinite(x) else (f"{x:+,.0f}" if isinstance(v, (int, np.integer)) and False else f"{x:,.0f}")


def _pct(v: Any, d: int = 1) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(x) else f"{100 * x:.{d}f}%"


def _pp(v: Any, d: int = 1) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(x) else f"{100 * x:+.{d}f} pp"


def headline_lines(PP: dict[str, Any], *, he: bool, synthetic: bool) -> list[str]:
    H, P = PP["headline"], PP["paired"]
    N, E, T = PP["n"], PP["events"], PP["selected_total"]
    if he:
        L = ["## תוצאה ראשית – Top 3%: PRE 2.2.0 מול POST 5.1 (אותם מטופלים, אותן תוויות, אותם קפלים)", "",
             f"**השאלה:** עם אותן {T:,} התערבויות (3% מתוך {N:,} מטופלים, {E:,} נופלים), כמה נפילות תופס המודל המתוקן לעומת PRE?", "",
             "| זרוע | גרסה | נבחרו | נפילות שנתפסו | Recall@Top3 | PPV@Top3 | התערבויות שווא | Δ נפילות שנתפסו (רווח סמך 95%) | Δ Recall | Δ התערבויות שווא | לכל 10,000 (נוסף) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    else:
        L = ["## Headline – Top 3%: PRE 2.2.0 vs POST 5.1 (identical patients, labels and outer folds)", "",
             f"**Question:** with the same {T:,} interventions (3% of {N:,} patients, {E:,} fallers), how many falls does the corrected model capture compared with PRE?", "",
             "| arm | version | selected | captured falls | Recall@Top3 | PPV@Top3 | false interventions | Δ captured (95% CI) | Δ Recall | Δ false interventions | per 10,000 (additional) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    if synthetic:
        L.insert(2, "> SYNTHETIC DATA – SOFTWARE TEST ONLY – NOT A SCIENTIFIC RESULT")
        L.insert(3, "")
    for r in H.itertuples():
        if r.role != "PRIMARY":
            continue
        if str(r.version).startswith("DELTA"):
            L.append(f"| {r.arm} | Δ POST − PRE | {T:,} | {float(r.captured_falls):+,.0f} | {_pp(r.recall_top3)} | {_pp(r.ppv_top3)} | {float(r.false_interventions):+,.0f} | "
                     f"{float(r.captured_falls):+,.0f} [{float(r.delta_captured_falls_ci_low):+,.0f}, {float(r.delta_captured_falls_ci_high):+,.0f}] | "
                     f"{_pp(r.recall_top3)} | {float(r.false_interventions):+,.0f} | {float(r.captured_per_10000):+.2f} |")
        else:
            L.append(f"| {r.arm} | {r.version} | {int(r.selected_total):,} | {int(r.captured_falls):,} | {_pct(r.recall_top3)} | {_pct(r.ppv_top3)} | "
                     f"{int(r.false_interventions):,} | | | | {float(r.captured_per_10000):.1f} |")
    L.append("")
    if he:
        L += ["- הטבלה מדווחת מספרים מוחלטים תחילה; 'לכל 10,000' הוא תוספת בלבד. ההפרש PRE→POST הוא ההשפעה המשולבת של כל התיקונים (R-1 ... R-6) – לא מיוחס לתיקון אחד.",
              "- OLD ו-ADMISSIBLE (OLD_PLUS_NEW_SAFE) הן זרועות ההשוואה הראשיות; ALL_NEW הישן הוא ביקורת בלבד; NO_NEW_REGISTRY הוא אבחון משני בלבד. לניסוי 1 אין 'הצלחה/כישלון'.", ""]
    else:
        L += ["- Absolute numbers first; per 10,000 is additional only. The PRE -> POST difference is the COMBINED effect of the repairs (R-1 ... R-6), "
              "attributed to no single change.",
              "- OLD and ADMISSIBLE (OLD_PLUS_NEW_SAFE) are the primary PRE / POST arms; legacy ALL is audit only; NO_NEW_REGISTRY is a secondary "
              "diagnostic only. Experiment 1 has no success verdict.", ""]
    return L
