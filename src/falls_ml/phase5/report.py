"""Phase 5 reports: aggregate tables, the Hebrew management summary, the Hebrew / English scientific summaries, figures, and the share/
folder behind a fail-closed privacy scan (identifier values, row keys, local paths, file types, row-level tables). Row-level OOF predictions and
the exhaustive threshold table stay local in work/analysis/."""

from __future__ import annotations

import getpass
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase5 import DESIGN_LABEL, PHASE5_VERSION, SYNTHETIC_WATERMARK, WATERMARK, WATERMARK_HE
from falls_ml.phase5.analysis import (PRIMARY_SETS, ModelRun, calibration_rows, capacity_rows, collect, compare, decide, model_row, nested_by_target,
                                      outer_fold_rows, overall, prediction_summary, subgroup_rows, threshold_tables)
from falls_ml.phase5.config import FAMILIES, PRIMARY_FAMILY, SET_ALL, SET_OLD, SET_SAFE

MIN_CELL = 10
ROW_LEVEL_COLUMNS = {"row_key", "Customer_Full_ID", "Snapshot_Key", "member_id", "research_id", "key"}
COUNT_COLS = re.compile(r"(^n$|_n$|^events$|_events$|^non_events$|(^|_)tp$|(^|_)fp$|(^|_)fn$|(^|_)tn$|(^|_)flagged$|^n_test$|^n_train$|captured_falls$|"
                        r"(^|_)false_alerts$|missed_falls$)")      # patient counts (never feature / fold / configuration counts)
HISTORICAL = {"source": "Phase 2/3 historical reference (2025 snapshot, VALIDATION partition) - context only, never evidence of incremental value",
              "threshold": 0.02, "sensitivity": 0.547, "events_captured": 140, "events": 256, "flagged": 1224, "ppv": 0.114, "false_positives": 1084,
              "false_alert_share": 0.886, "specificity": 0.909}
VERDICT_HE = {"NEW_FEATURES_OPERATIONALLY_USEFUL": "שימושי תפעולית", "PROMISING_BUT_NOT_ROBUST": "מבטיח אך לא יציב", "NO_ROBUST_OPERATIONAL_GAIN":
              "אין רווח תפעולי יציב", "NO_ELIGIBLE_NEW_FEATURES": "אין פיצ'רים חדשים כשירים", "INCOMPLETE": "לא הושלם"}
OVERALL_HE = {"YES": "כן", "NO": "לא", "UNCERTAIN": "לא ודאי"}
FAM_HE = {"LASSO": "LASSO", "ENET": "Elastic Net", "XGB": "XGBoost"}


def pct(x: Any, d: int = 1) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(v) else f"{100 * v:.{d}f}%"


def pp(x: Any, d: int = 1) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(v) else f"{100 * v:+.{d}f} pp"


def num(x: Any, d: int = 0) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "—" if not math.isfinite(v) else (f"{v:,.{d}f}")


def ci(lo: Any, hi: Any, f: Any = pct) -> str:
    return f"[{f(lo)}, {f(hi)}]"


RATE_COLS = re.compile(r"(sensitivity|specificity|ppv|npv|false_alert|fpr|flagged|lift|per_10000|per_capture|per_captured|captured_falls|false_alerts|"
                       r"missed_falls)")


def _small_cell(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v)) and 0 < float(v) < MIN_CELL


def suppress(df: pd.DataFrame) -> pd.DataFrame:
    """Counts 1-9 become '<10'; in a row with such a count every operating rate is 'suppressed' too (a rate would give the cell back).
    Differences between two models (delta_ columns) are not counts of patients and are kept."""
    out = df.copy().astype(object)
    hit = pd.Series(False, index=out.index)
    for c in out.columns:
        if COUNT_COLS.search(str(c)) and not str(c).startswith("delta"):
            small = out[c].map(_small_cell)
            hit |= small.astype(bool)
            out.loc[small.astype(bool), c] = "<10"
    if hit.any():
        for c in out.columns:
            if RATE_COLS.search(str(c)) and not str(c).startswith("delta"):
                out.loc[hit, c] = out.loc[hit, c].map(lambda v: v if isinstance(v, str) else "suppressed")
    return out


def grid_ok(t: pd.DataFrame) -> pd.Series:
    """Rows whose four confusion cells are all 0 or >= 10 (a rate of a smaller cell would give the cell back)."""
    ok = pd.Series(True, index=t.index)
    for c in ("tp", "fp", "fn", "tn"):
        if c in t:
            v = pd.to_numeric(t[c], errors="coerce")
            ok &= ~((v > 0) & (v < MIN_CELL))
    return ok


def suppress_grid(t: pd.DataFrame) -> pd.DataFrame:
    out = suppress(t)
    bad = ~grid_ok(t)
    for c in [c for c in out.columns if c not in ("family", "feature_set", "threshold", "tp", "fp", "fn", "tn", "n", "events", "flagged")]:
        out.loc[bad, c] = out.loc[bad, c].map(lambda v: v if isinstance(v, str) else "suppressed")
    return out


def suppress_subgroups(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy().astype(object)
    for i, r in out.iterrows():
        n, ev = r["n"], r["events"]
        if (0 < ev < MIN_CELL) or (0 < n - ev < MIN_CELL) or (0 < n < MIN_CELL):
            out.at[i, "n"] = "<10" if 0 < n < MIN_CELL else n
            out.at[i, "events"] = "<10" if 0 < ev < MIN_CELL else ("suppressed" if 0 < n - ev < MIN_CELL else ev)
            for c in ("auroc", "ap", "sensitivity", "ppv", "false_alert_share", "flagged_share"):
                out.at[i, c] = "suppressed"
    return out


# ============================================================================ the analysis
def analyse(ctx: Any, plan: dict[str, Any], cfg: Any, *, interim: bool) -> dict[str, Any]:
    y = ctx.y.astype(float)
    runs = collect(ctx, plan, cfg)
    T = cfg.primary_sensitivity
    nb = int(cfg.budget["bootstrap_n"])
    seed = int(plan["seed"])
    alias = plan["alias"]
    A: dict[str, Any] = {"runs": runs, "interim": interim, "primary_family": PRIMARY_FAMILY}
    rows = []
    for (fam, s), run in runs.items():
        prim = s in PRIMARY_SETS
        r = model_row(y, run, target=T, n_boot=nb, seed=seed + 3, with_ci=prim)
        r["n_features"] = len(plan["sets"][s])
        r["kind"] = plan["kinds"].get(alias[s], "")
        r["role"] = ("PRIMARY" if fam == PRIMARY_FAMILY and s in (SET_OLD, SET_ALL) else "SECONDARY") if prim else plan["kinds"].get(alias[s], "")
        r["identical_to"] = alias[s] if alias[s] != s else ""
        r["inner_selected_configs"] = json.dumps([f["config"] for f in run.folds], default=float)[:2000]
        r["not_converged_fits"] = int(sum(int(f.get("not_converged_fits") or 0) for f in run.folds))
        r["lambda_on_grid_edge_folds"] = int(sum(bool(f.get("edge")) for f in run.folds))
        rows.append(r)
    A["models"] = pd.DataFrame(rows)
    A["targets"] = pd.concat([nested_by_target(y, run, cfg.targets) for (fam, s), run in runs.items() if s in PRIMARY_SETS],
                             ignore_index=True) if runs else pd.DataFrame()
    no_new = alias[SET_ALL] == alias[SET_OLD]
    safe_same = alias[SET_SAFE] == alias[SET_ALL]
    safe_none = alias[SET_SAFE] == alias[SET_OLD]
    cmps, dec = {}, {}
    for fam in FAMILIES:
        o, al, sf = runs.get((fam, SET_OLD)), runs.get((fam, SET_ALL)), runs.get((fam, SET_SAFE))
        c_all = compare(y, o, al, target=T, n_boot=nb, seed=seed + 11) if o and al and not no_new else None
        c_safe = c_all if safe_same else (compare(y, o, sf, target=T, n_boot=nb, seed=seed + 12) if o and sf and not safe_none else None)
        cmps[fam] = {"all": c_all, "safe": c_safe}
        dec[fam] = decide(c_all, c_safe, no_new=no_new, safe_same=safe_same, cfg=cfg)
    A["comparisons"], A["decisions"] = cmps, dec
    A["overall"] = overall({f: d["verdict"] for f, d in dec.items()})
    A["safe_same"], A["safe_none"], A["no_new"] = safe_same, safe_none, no_new
    # exploratory only: the family with the lowest nested false-alert share of OLD_PLUS_ALL_NEW_ELIGIBLE (never the basis of the answer)
    cand = [(runs[(f, SET_ALL)], f) for f in FAMILIES if (f, SET_ALL) in runs]
    A["best_family_exploratory"] = min(cand, key=lambda t: (np.nan_to_num(_op(y, t[0], T)["false_alert_share"], nan=9.0), FAMILIES.index(t[1])))[1] if cand else None
    A["lead_family"] = PRIMARY_FAMILY
    # domains (OLD + domain vs OLD) and ablations (ALL_NEW minus a block vs ALL_NEW), paired
    drows, arows = [], []
    for fam in [f for f in FAMILIES if f in plan.get("domain_families", FAMILIES)]:
        o = runs.get((fam, SET_OLD))
        for dom, info in plan["domains"].items():
            s = info["name"]
            if not info["added"]:
                drows.append({"family": fam, "domain": dom, "n_new_features": 0, "status": "NO_ELIGIBLE_FEATURES_IN_DOMAIN"})
                continue
            if o is None or (fam, s) not in runs:
                drows.append({"family": fam, "domain": dom, "n_new_features": len(info["added"]), "status": "NOT_COMPLETE"})
                continue
            c = compare(y, o, runs[(fam, s)], target=T, n_boot=nb, seed=seed + 21)
            drows.append({"family": fam, "role": "PRIMARY" if fam == PRIMARY_FAMILY else "SECONDARY", "domain": dom, "n_new_features": len(info["added"]),
                          "new_features": "; ".join(info["added"]), "status": "COMPLETE", **_delta_cols(c), "delta_false_alerts": c["delta_op_fp"],
                          "folds_improved": c["folds_improved"], "n_folds": len(c["fold_delta_false_alert_share"])})
    for fam in [f for f in FAMILIES if f in plan.get("ablation_families", [PRIMARY_FAMILY])]:
        al = runs.get((fam, SET_ALL))
        for ab, info in plan["ablations"].items():
            s = info["name"]
            if not info["removed"]:
                arows.append({"family": fam, "ablation": ab, "n_removed": 0, "status": "NOTHING_TO_REMOVE"})
                continue
            if al is None or (fam, s) not in runs:
                arows.append({"family": fam, "ablation": ab, "n_removed": len(info["removed"]), "status": "NOT_COMPLETE"})
                continue
            c = compare(y, al, runs[(fam, s)], target=T, n_boot=nb, seed=seed + 31)
            b = c["b"]
            arows.append({"family": fam, "ablation": ab, "n_removed": len(info["removed"]), "removed": "; ".join(info["removed"]), "status": "COMPLETE",
                          "ap": b["ap"], "auroc": b["auroc"], "brier": b["brier"], "ppv_at_70": b["op_ppv"], "false_alert_share_at_70": b["op_false_alert_share"],
                          "flagged_share_at_70": b["op_flagged_share"], **_delta_cols(c), "delta_false_alerts": c["delta_op_fp"],
                          "folds_worse_without_block": c["folds_improved"], "note": "delta = (all new minus the block) - (all new): > 0 means the block helped"})
    A["domains"], A["ablations"] = pd.DataFrame(drows), pd.DataFrame(arows)
    A["subgroups"] = subgroup_rows(ctx, runs, cfg)
    A["calibration"] = calibration_rows(y, runs, 10)
    A["pred_summary"] = prediction_summary(y, runs)
    A["outer_folds"] = outer_fold_rows(y, runs, T)
    A["capacity"] = capacity_rows(y, runs, [float(c) for c in cfg["operating"].get("capacities", [0.1, 0.2, 0.3])])
    A["grid"], A["full_thresholds"] = threshold_tables(y, runs, float(cfg["privacy"]["threshold_grid_step"]))
    return A


def _op(y: np.ndarray, run: ModelRun, T: float) -> dict[str, Any]:
    from falls_ml.phase5.thresholds import counts

    return counts(y, run.flags[f"{T:.2f}"])


def _delta_cols(c: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, lab in (("op_false_alert_share", "delta_false_alert_share"), ("op_flagged_share", "delta_flagged_share"), ("op_ppv", "delta_ppv"),
                   ("op_sensitivity", "delta_sensitivity"), ("op_false_alerts_per_10000", "delta_false_alerts_per_10000"), ("ap", "delta_ap"),
                   ("auroc", "delta_auroc"), ("brier", "delta_brier")):
        out[lab] = c[f"delta_{k}"]
        out[f"{lab}_ci_low"] = c[f"delta_{k}_ci_low"]
        out[f"{lab}_ci_high"] = c[f"delta_{k}_ci_high"]
    return out


# ============================================================================ tables
def _model_cols(m: dict[str, Any]) -> dict[str, Any]:
    return {"achieved_sensitivity": m["op_sensitivity"], "ppv": m["op_ppv"], "false_alert_share": m["op_false_alert_share"], "flagged_share": m["op_flagged_share"],
            "fpr": m["op_fpr"], "specificity": m["op_specificity"], "captured_falls": m["op_tp"], "false_alerts": m["op_fp"], "missed_falls": m["op_fn"],
            "true_negatives": m["op_tn"], "flagged": m["op_flagged"], "flagged_per_10000": m["op_flagged_per_10000"],
            "false_alerts_per_10000": m["op_false_alerts_per_10000"], "flagged_per_captured_fall": m["op_flagged_per_capture"], "lift": m["op_lift"],
            "ap": m["ap"], "auroc": m["auroc"], "brier": m["brier"], "brier_skill": m.get("brier_skill"), "logloss": m.get("logloss"),
            "calibration_intercept": m["calibration_intercept"], "calibration_slope": m["calibration_slope"],
            "descriptive_pooled_false_alert_share_at_70": m["desc_fas_0.70"], "descriptive_pooled_flagged_share_at_70": m["desc_flagged_0.70"]}


def _delta_row(c: dict[str, Any]) -> dict[str, Any]:
    return {"achieved_sensitivity": c["delta_op_sensitivity"], **{k: c.get(f"delta_op_{k}") for k in ("ppv", "false_alert_share", "flagged_share", "fpr", "specificity")},
            "captured_falls": c["delta_op_tp"], "false_alerts": c["delta_op_fp"], "missed_falls": c.get("delta_op_fn"), "flagged": c.get("delta_op_flagged"),
            "flagged_per_10000": c["delta_op_flagged_per_10000"], "false_alerts_per_10000": c["delta_op_false_alerts_per_10000"],
            "false_alerts_avoided_per_10000": -c["delta_op_false_alerts_per_10000"],
            "false_alerts_avoided_per_10000_ci_low": -c["delta_op_false_alerts_per_10000_ci_high"],
            "false_alerts_avoided_per_10000_ci_high": -c["delta_op_false_alerts_per_10000_ci_low"],
            **{f"delta_{k}_ci_{e}": c[f"delta_op_{k}_ci_{e}"] for k in ("false_alert_share", "flagged_share", "ppv", "false_alerts_per_10000", "fp", "tp", "flagged")
               for e in ("low", "high") if f"delta_op_{k}_ci_{e}" in c},
            "delta_captured_per_10000": c["delta_op_captured_per_10000"],
            **{f"delta_{k}{suf}": c[f"delta_{k}{suf}"] for k in ("ap", "auroc", "brier") for suf in ("", "_ci_low", "_ci_high")},
            "delta_descriptive_false_alert_share_at_70": c.get("delta_desc_fas_0.70"), "folds_improved": c["folds_improved"],
            "fold_delta_false_alert_share": "; ".join(f"{100 * v:+.2f} pp" for v in c["fold_delta_false_alert_share"])}


def primary_table(A: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for fam in (PRIMARY_FAMILY, *[f for f in FAMILIES if f != PRIMARY_FAMILY]):
        role = "PRIMARY" if fam == PRIMARY_FAMILY else "SECONDARY"
        c, cs, d = A["comparisons"][fam]["all"], A["comparisons"][fam]["safe"], A["decisions"][fam]
        if c is None:
            rows.append({"family": fam, "role": role, "comparison": f"{SET_OLD} vs {SET_ALL}", "row": "VERDICT", "verdict": d["verdict"], "reason": d["reason"]})
            continue
        rows.append({"family": fam, "role": role, "comparison": f"{SET_OLD} vs {SET_ALL}", "row": SET_OLD, **_model_cols(c["a"])})
        rows.append({"family": fam, "role": role, "comparison": f"{SET_OLD} vs {SET_ALL}", "row": SET_ALL, **_model_cols(c["b"])})
        rows.append({"family": fam, "role": role, "comparison": f"{SET_OLD} vs {SET_ALL}", "row": "DELTA_NEW_MINUS_OLD", **_delta_row(c)})
        if cs is not None and not A.get("safe_same"):
            rows.append({"family": fam, "role": "SECONDARY", "comparison": f"{SET_OLD} vs {SET_SAFE}", "row": SET_SAFE, **_model_cols(cs["b"])})
            rows.append({"family": fam, "role": "SECONDARY", "comparison": f"{SET_OLD} vs {SET_SAFE}", "row": "DELTA_NEW_MINUS_OLD", **_delta_row(cs)})
        rows.append({"family": fam, "role": role, "comparison": f"{SET_OLD} vs {SET_ALL}", "row": "VERDICT", "verdict": d["verdict"], "reason": d["reason"],
                     **{f"criterion_{k}": v for k, v in d["criteria"].items()}})
    return pd.DataFrame(rows)


# ============================================================================ figures
def figures(fig_dir: Path, A: dict[str, Any], y: np.ndarray, perm: pd.DataFrame, stab: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    runs, grid = A["runs"], A["grid"]
    grid = grid[grid_ok(grid)] if len(grid) else grid          # the figures show only rows the shared table shows in full
    made = []
    col = {SET_OLD: "#6b7280", SET_ALL: "#2563eb", SET_SAFE: "#16a34a"}
    lab = {SET_OLD: "OLD", SET_ALL: "OLD + ALL NEW", SET_SAFE: "OLD + NEW SAFE"}
    shown = (SET_OLD, SET_ALL) if A.get("safe_same") else (SET_OLD, SET_ALL, SET_SAFE)

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=130)
        plt.close(fig)
        made.append(name)

    fams = [f for f in (PRIMARY_FAMILY, *[f for f in FAMILIES if f != PRIMARY_FAMILY]) if (f, SET_OLD) in runs]
    ttl = lambda fam: f"{fam}{' (PRIMARY)' if fam == PRIMARY_FAMILY else ''}"  # noqa: E731
    if len(grid) and fams:
        for name, ycol, ylab in (("01_sensitivity_vs_false_alert_share.png", "false_alert_share", "false-alert share (FP / alerts)"),
                                 ("02_sensitivity_vs_population_flagged.png", "flagged_share", "population flagged"),
                                 ("03_sensitivity_vs_ppv.png", "ppv", "PPV (precision)")):
            fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4), squeeze=False)
            for ax, fam in zip(axes[0], fams):
                for s in shown:
                    g = grid[(grid["family"] == fam) & (grid["feature_set"] == s)]
                    ax.plot(g["sensitivity"], g[ycol], label=lab[s], color=col[s])
                ax.axvline(0.70, ls="--", color="#b91c1c", lw=1)
                ax.set_title(f"{ttl(fam)} - pooled OOF, descriptive")
                ax.set_xlabel("sensitivity (falls captured)")
                ax.set_ylabel(ylab)
                ax.legend(fontsize=8)
            save(fig, name)
        for name, kind in (("04_precision_recall.png", "pr"), ("05_roc.png", "roc")):
            fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4), squeeze=False)
            for ax, fam in zip(axes[0], fams):
                for s in shown:
                    g = grid[(grid["family"] == fam) & (grid["feature_set"] == s)]
                    if not len(g):
                        continue
                    if kind == "pr":
                        ax.plot(g["sensitivity"], g["ppv"], color=col[s], label=lab[s])
                        ax.set_xlabel("recall (sensitivity)")
                        ax.set_ylabel("precision (PPV)")
                    else:
                        ax.plot(np.r_[0.0, g["fpr"].to_numpy(dtype=float)], np.r_[0.0, g["sensitivity"].to_numpy(dtype=float)], color=col[s], label=lab[s])
                        ax.plot([0, 1], [0, 1], ":", color="#9ca3af")
                        ax.set_xlabel("false-positive rate")
                        ax.set_ylabel("sensitivity")
                ax.set_title(f"{ttl(fam)} (pooled OOF, 0.5% grid)")
                ax.legend(fontsize=8)
            save(fig, name)
    cal = A["calibration"]
    if len(cal) and fams:
        fig, axes = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4), squeeze=False)
        for ax, fam in zip(axes[0], fams):
            for s in shown:
                c = cal[(cal["family"] == fam) & (cal["feature_set"] == s)]
                ax.plot(c["mean_predicted"], c["observed_rate"], "o-", color=col[s], label=lab[s])
            mx = float(cal["mean_predicted"].max() * 1.1) if len(cal) else 1
            ax.plot([0, mx], [0, mx], ":", color="#9ca3af")
            ax.set_title(f"{ttl(fam)} calibration (deciles)")
            ax.set_xlabel("mean predicted risk")
            ax.set_ylabel("observed fall rate")
            ax.legend(fontsize=8)
        save(fig, "06_calibration.png")
    m = A["models"]
    if len(m):
        mm = m[m["feature_set"].isin(list(shown))]
        if len(mm):
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = [f"{r.family}\n{lab[r.feature_set]}" for r in mm.itertuples()]
            err = np.array([[r.brier - r.brier_ci_low if pd.notna(getattr(r, "brier_ci_low", np.nan)) else 0 for r in mm.itertuples()],
                            [r.brier_ci_high - r.brier if pd.notna(getattr(r, "brier_ci_high", np.nan)) else 0 for r in mm.itertuples()]])
            ax.bar(range(len(mm)), mm["brier"], yerr=err, color=[col[s] for s in mm["feature_set"]])
            ax.set_xticks(range(len(mm)), labels, fontsize=7)
            ax.set_ylabel("Brier score (lower = better)")
            ax.set_title("Brier comparison (pooled OOF, 95% CI)")
            save(fig, "07_brier_comparison.png")
    dm = A["domains"]
    if len(dm) and "delta_false_alert_share" in dm:
        dd = dm[dm["status"] == "COMPLETE"]
        if len(dd):
            fig, ax = plt.subplots(figsize=(8, 0.5 * len(dd) + 1.5))
            yl = [f"{r.family}: {r.domain}" for r in dd.itertuples()]
            v = dd["delta_false_alert_share"].to_numpy(dtype=float) * 100
            lo = dd["delta_false_alert_share_ci_low"].to_numpy(dtype=float) * 100
            hi = dd["delta_false_alert_share_ci_high"].to_numpy(dtype=float) * 100
            ax.errorbar(v, range(len(dd)), xerr=[v - lo, hi - v], fmt="o", color="#2563eb")
            ax.axvline(0, color="#6b7280", lw=1)
            ax.set_yticks(range(len(dd)), yl, fontsize=8)
            ax.set_xlabel("change in false-alert share at the 70% target (percentage points; < 0 = fewer false alerts)")
            ax.set_title("Domain incremental value: OLD + domain vs OLD")
            save(fig, "08_domain_incremental_value.png")
    st_ = stab[(stab["family"] == PRIMARY_FAMILY)] if len(stab) and "selection_frequency" in stab else pd.DataFrame()
    px = perm[(perm["family"] == "XGB") & (perm["feature_set"] == SET_ALL)] if len(perm) else pd.DataFrame()
    if len(st_) or len(px):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if len(st_):
            t = st_.sort_values(["selection_frequency", "coefficient_median"], ascending=[False, False]).head(20).iloc[::-1]
            axes[0].barh(t["feature"], t["selection_frequency"], color=["#2563eb" if str(f).startswith("new_") else "#6b7280" for f in t["feature"]])
            axes[0].set_xlim(0, 1)
            axes[0].set_title(f"{PRIMARY_FAMILY} (primary): bootstrap selection frequency\n(blue = new V21 predictor; not causal)")
            axes[0].tick_params(axis="y", labelsize=7)
        if len(px):
            t = px.head(15).iloc[::-1]
            axes[1].barh(t["feature"], t["mean_drop_ap"], color=["#2563eb" if str(f).startswith("new_") else "#6b7280" for f in t["feature"]])
            axes[1].set_title("XGBoost (secondary): permutation importance (AP drop)")
            axes[1].tick_params(axis="y", labelsize=7)
        save(fig, "09_feature_importance_coefficient_stability.png")
    if len(grid):
        g = grid[(grid["family"] == PRIMARY_FAMILY) & (grid["feature_set"] == SET_ALL)]
        if len(g):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(g["threshold"], g["sensitivity"], label="sensitivity")
            ax.plot(g["threshold"], g["ppv"], label="PPV")
            ax.plot(g["threshold"], g["flagged_share"], label="population flagged")
            ax.plot(g["threshold"], g["false_alert_share"], label="false-alert share")
            ax.set_xscale("log")
            ax.set_xlabel("risk threshold")
            ax.set_title(f"Threshold trade-off ({PRIMARY_FAMILY}, {SET_ALL}, pooled OOF)")
            ax.legend(fontsize=8)
            save(fig, "10_threshold_tradeoff.png")
    c = A["comparisons"].get(PRIMARY_FAMILY, {}).get("all")
    if c is not None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, (lb, mm_) in zip(axes, (("OLD", c["a"]), ("OLD + ALL NEW", c["b"]))):
            n = mm_["op_tp"] + mm_["op_fp"] + mm_["op_fn"] + mm_["op_tn"]
            vals = np.array([[mm_["op_tp"], mm_["op_fn"]], [mm_["op_fp"], mm_["op_tn"]]]) * 1e4 / n
            ax.imshow(vals, cmap="Blues")
            for (i, j), v in np.ndenumerate(vals):
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=11, color="white" if v > vals.max() * 0.5 else "black")
            ax.set_xticks([0, 1], ["alerted", "not alerted"])
            ax.set_yticks([0, 1], ["fell", "did not fall"])
            ax.set_title(f"{lb} ({PRIMARY_FAMILY}) per 10,000 patients")
        save(fig, "11_confusion_per_10000_at_70.png")
    rows = [(f, A["comparisons"][f]["all"]) for f in fams if A["comparisons"].get(f, {}).get("all")]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(rows))
        a = [r["a"]["op_false_alerts_per_10000"] for _, r in rows]
        b = [r["b"]["op_false_alerts_per_10000"] for _, r in rows]
        ax.bar(x - 0.2, a, 0.4, label="OLD", color=col[SET_OLD])
        ax.bar(x + 0.2, b, 0.4, label="OLD + ALL NEW", color=col[SET_ALL])
        ax.set_xticks(x, [ttl(f) for f, _ in rows])
        ax.set_ylabel("false alerts per 10,000 patients")
        ax.set_title("False alerts at the nested 70%-sensitivity rule")
        ax.legend()
        save(fig, "12_false_alerts_per_10000.png")
    return made


# ============================================================================ summaries
REASON_HE = (("UNVALIDATED_CODES", "תת-קוד גולמי ללא מילון מאומת"), ("EXPERIMENTAL_COMPOSITE", "מדד מורכב ניסויי"),
             ("timing uncertain", "תזמון לא ודאי ביום המדד (כולל חשיפה תרופתית שאינה מוגבלת בתאריך רכישה)"),
             ("INELIGIBLE_LEAKAGE", "חשד לדליפת מידע מהעתיד (חזק מדי לבדו)"), ("INELIGIBLE_TIMING", "רשומות אחרי תאריך המדד"),
             ("INELIGIBLE_DATA", "מעט מדי נתונים / ערך קבוע"), ("INELIGIBLE_SEMANTICS", "הערכים אינם תואמים את ההגדרה"))


def _reason_he(text: str) -> str:
    for key, he in REASON_HE:
        if key in text:
            return he
    return text


def _excluded_lines(plan: dict[str, Any]) -> list[str]:
    np_ = plan.get("new_predictors") or {}
    out = [f"  - {r['column']}: לא נכלל באף מערך – {_reason_he(r['class'] + ' ' + r['set_reason'])}" for r in np_.get("excluded_from_all_new", [])]
    out += [f"  - {r['column']}: נכלל ב-ALL_NEW בלבד (לא ב-NEW_SAFE) – {_reason_he(r['set_reason'])}" for r in np_.get("all_new_only", [])]
    out += [f"  - {r['column']}: עמודה חדשה שאינה מנבא – {'דגל היעדר מקור / מטא-דאטה' if r['class'] == 'METADATA_OR_ADMIN' else r['class']}"
            for r in np_.get("new_non_predictor_columns", [])]
    return out


def management_he(A: dict[str, Any], plan: dict[str, Any], synthetic: bool, interim: bool) -> str:
    n, ev = plan["n"], plan["events"]
    fam = PRIMARY_FAMILY
    sc = plan.get("schema_counts") or {}
    np_ = plan.get("new_predictors") or {}
    L = ["# סיכום למנהלים – שלב 5: פיתוח מחדש על נתוני 2026 והערך המוסף של מידע V21", "", f"**{WATERMARK_HE}**", ""]
    if synthetic:
        L += [f"> **{SYNTHETIC_WATERMARK}** – המספרים להלן מנתונים סינתטיים (בדיקת תוכנה) ואינם תוצאה מדעית.", ""]
    if interim:
        L += ["> **דו\"ח ביניים** – ההשוואה הראשית הושלמה; ניתוחי התחומים, האבלציות והיציבות עדיין רצים.", ""]
    L += ["## עמוד ראשון", "", "**השאלה:** בערך באותה רגישות של 70% לפחות (תפיסת נפילות), האם הוספת המידע החדש של V21 מפחיתה את מספר ואחוז התראות השווא?", "",
          "### הנתונים והפיצ'רים",
          f"- **מטופלים שנותחו:** {num(n)} (כל הזכאים עם תוצא ידוע, תאריך מדד 01/01/2026, תחזית בסוף יום המדד).",
          f"- **נפילות:** {num(ev)} ב-180 הימים שאחרי יום המדד ({pct(ev / n if n else float('nan'))} מהמטופלים).",
          f"- **השוואה מדויקת V1 → V21:** {sc.get('v1_columns', '—')} עמודות ב-V1, {sc.get('extract_columns', '—')} ב-V21; "
          f"{sc.get('new_columns', '—')} עמודות חדשות, {sc.get('removed_v1_columns', '—')} הוסרו, {sc.get('renamed_or_replaced', '—')} הוחלפו בשושלת מוכחת, "
          f"{sc.get('changed_definition', '—')} שינו הגדרה. שדות MEFI אינם חדשים (קיימים ב-V1).",
          f"- **פיצ'רים חדשים אמיתיים שנמצאו ב-V21:** {np_.get('genuine_new_predictors', '—')}.",
          f"- **נכנסו ל-OLD + כל החדשים הכשירים (ההשוואה העסקית):** {np_.get('all_new_eligible', '—')}.",
          f"- **נכנסו ל-OLD + חדשים בטוחים בלבד (בדיקה מחמירה):** {np_.get('new_safe', '—')}."]
    ex = _excluded_lines(plan)
    L += ["- **מה הוצא ולמה:**" if ex else "- **מה הוצא:** אף פיצ'ר חדש לא הוצא.", *ex, ""]
    c = A["comparisons"].get(fam, {}).get("all")
    if c is not None:
        a, b = c["a"], c["b"]
        L += [f"### המודל הישן (OLD, {FAM_HE[fam]}) בכ-70% רגישות",
              f"- מטופלים שקיבלו התראה: {num(a['op_flagged'])} ({pct(a['op_flagged_share'])} מהאוכלוסייה)",
              f"- נפילות שנתפסו: {num(a['op_tp'])} מתוך {num(ev)} ({pct(a['op_sensitivity'])})",
              f"- התראות שווא: {num(a['op_fp'])} ({pct(a['op_false_alert_share'])} מההתראות)",
              f"- PPV (שיעור ההתראות שבהן אכן הייתה נפילה): {pct(a['op_ppv'])}", "",
              f"### OLD + כל הפיצ'רים החדשים ({FAM_HE[fam]})",
              f"- מטופלים שקיבלו התראה: {num(b['op_flagged'])} ({pct(b['op_flagged_share'])} מהאוכלוסייה)",
              f"- נפילות שנתפסו: {num(b['op_tp'])} מתוך {num(ev)} ({pct(b['op_sensitivity'])})",
              f"- התראות שווא: {num(b['op_fp'])} ({pct(b['op_false_alert_share'])} מההתראות)",
              f"- PPV: {pct(b['op_ppv'])}", "",
              "### ההבדל (חדש פחות ישן, אותם מטופלים)",
              f"- **התראות שווא שנחסכו:** {num(a['op_fp'] - b['op_fp'])} בקוהורט; {num(-c['delta_op_false_alerts_per_10000'])} לכל 10,000 מטופלים "
              f"(רווח סמך 95%: {num(-c['delta_op_false_alerts_per_10000_ci_high'])} עד {num(-c['delta_op_false_alerts_per_10000_ci_low'])}; ערך שלילי = יותר התראות שווא)",
              f"- **התערבויות שנחסכו:** {num(a['op_flagged'] - b['op_flagged'])} מטופלים פחות קיבלו התראה ({num(-c['delta_op_flagged_per_10000'])} לכל 10,000)",
              f"- **שינוי ב-PPV:** {pp(c['delta_op_ppv'])} (רווח סמך {pp(c['delta_op_ppv_ci_low'])} עד {pp(c['delta_op_ppv_ci_high'])})",
              f"- **שינוי בשיעור התראות השווא:** {pp(c['delta_op_false_alert_share'])} (רווח סמך {pp(c['delta_op_false_alert_share_ci_low'])} עד "
              f"{pp(c['delta_op_false_alert_share_ci_high'])}); שיפור ב-{c['folds_improved']} מתוך {len(c['fold_delta_false_alert_share'])} הקפלים",
              f"- **נפילות שנוספו / אבדו:** {num(b['op_tp'] - a['op_tp'])} (חיובי = יותר נפילות נתפסו)", ""]
    elif A.get("no_new"):
        L += ["### אין השוואה", f"אף פיצ'ר חדש של V21 לא עבר את בדיקות התזמון / המשמעות / הכיסוי / הדליפה, ולכן OLD+ALL_NEW זהה ל-OLD.", ""]
    else:
        L += ["### ההשוואה הראשית עוד לא הושלמה", ""]
    d = A["decisions"].get(fam, {})
    ans = OVERALL_HE.get(A["overall"], A["overall"])
    crit = d.get("criteria") or {}
    why = {"1_lower_false_alert_share_nested_and_descriptive": "פחות התראות שווא בכלל ה-70% המקונן ובנקודת ה-70% התיאורית",
           "2_paired_ci_below_zero_nested_and_at_equal_sensitivity": "רווח הסמך המזווג של ההפרש כולו מתחת ל-0 (בכלל המקונן ובאותה רגישות בדיוק)",
           "3_every_outer_fold_improves": f"שיפור בכל {plan['cv']['outer_folds']} הקפלים החיצוניים",
           "4_not_dependent_on_timing_or_provenance_questionable_predictors": "השיפור נשמר גם בלי פיצ'רים עם תזמון / מקור מפוקפק (NEW_SAFE)",
           "5_calibration_not_materially_worse": "הכיול לא נפגע מהותית"}
    L += ["### האם הפיצ'רים הנוספים של V21 הוסיפו מידע שימושי?", "", f"## **{ans}**", "",
          f"התשובה נקבעת מראש לפי Elastic Net בלבד (OLD מול OLD + כל החדשים הכשירים), לפי כלל החלטה שנקבע לפני שנראו תוצאות: "
          f"{VERDICT_HE.get(d.get('verdict', ''), d.get('verdict', ''))}."]
    if crit:
        L += [f"- {'✔' if v else '✘'} {why.get(k, k)}" for k, v in crit.items()]
    sec = [f"{FAM_HE[f]} – {VERDICT_HE.get(A['decisions'][f]['verdict'], A['decisions'][f]['verdict'])}" for f in FAMILIES if f != fam]
    L += ["", f"ניתוחים משניים (לא קובעים את התשובה): {'; '.join(sec)}. "
          + (f"המשפחה עם הכי מעט התראות שווא (חקרני בלבד): {FAM_HE.get(A.get('best_family_exploratory'), A.get('best_family_exploratory'))}." if A.get("best_family_exploratory") else "")]
    cs = A["comparisons"].get(fam, {}).get("safe")
    if cs is not None and not A.get("safe_same"):
        L += [f"בדיקה מחמירה (OLD + חדשים בטוחים בלבד, {FAM_HE[fam]}): שינוי בשיעור התראות השווא {pp(cs['delta_op_false_alert_share'])} "
              f"(רווח סמך {pp(cs['delta_op_false_alert_share_ci_low'])} עד {pp(cs['delta_op_false_alert_share_ci_high'])}); "
              f"{num(-cs['delta_op_false_alerts_per_10000'])} התראות שווא פחות לכל 10,000."]
    dm = A["domains"]
    if len(dm) and "delta_false_alert_share" in dm and ((dm["status"] == "COMPLETE") & (dm["family"] == fam)).any():
        dd = dm[(dm["status"] == "COMPLETE") & (dm["family"] == fam)].sort_values("delta_false_alert_share")
        L += ["", f"**תרומת תחומים ({FAM_HE[fam]}, OLD + תחום מול OLD):** " + "; ".join(
            f"{r.domain}: {pp(r.delta_false_alert_share)} (רווח סמך {pp(r.delta_false_alert_share_ci_low)} עד {pp(r.delta_false_alert_share_ci_high)}; "
            f"שיפור ב-{int(r.folds_improved)}/{int(r.n_folds)} קפלים)" for r in dd.itertuples())]
    L += ["", "### מגבלות",
          "- תיקוף צולב מקונן פנימי על תמונת מצב אחת (2026) – לא תיקוף חיצוני; נדרשת תמונת מצב עתידית בלתי תלויה לפני הטמעה.",
          "- התראות השווא נמדדות מול נפילות מתועדות בלבד (רשומות אבחנה); רשומות חוזרות עשויות לתעד מעקב אחר אותו אירוע.",
          "- פיצ'רים 'מאושרים' (SAFE_ATTESTED, למשל רשמים) מסתמכים על הצהרת ה-DWH שאין מידע אחרי תאריך המדד; חשיפה תרופתית אינה מוגבלת בתאריך רכישה (הסתייגות V21).",
          "- חשיבות פיצ'רים אינה סיבתית.", "",
          "## הערות לקריאה",
          "- 'כלל 70% מקונן': הסף נבחר בכל קפל רק על נתוני האימון של אותו קפל (תחזיות פנימיות) והוחל על מטופלים שלא נראו – זו ההערכה התפעולית הנאמנה; "
          "הרגישות בפועל משתנה סביב 70%.",
          "- 'OOF מאוחד': אותן תחזיות מכל הקפלים יחד, עם סף שנקבע ישירות על התוצאות – לתכנון בלבד, לא תיקוף.",
          f"- נתון היסטורי (2025, הקשר בלבד, לא ראיה לערך הפיצ'רים החדשים): בסף 0.02 נתפסו {pct(HISTORICAL['sensitivity'])} מהנפילות, PPV {pct(HISTORICAL['ppv'])}.", "",
          "קבצים: PRIMARY_70_SENSITIVITY_COMPARISON.csv, THRESHOLD_TRADEOFF.csv, DOMAIN_INCREMENTAL_VALUE.csv, ALL_V21_COLUMN_CLASSIFICATION.csv, figures/."]
    return "\n".join(L) + "\n"


def scientific(A: dict[str, Any], plan: dict[str, Any], cfg: Any, synthetic: bool, interim: bool, he: bool) -> str:
    n, ev = plan["n"], plan["events"]
    t = A["models"]
    sc = plan.get("schema_counts") or {}
    np_ = plan.get("new_predictors") or {}
    L = [("# סיכום מדעי – שלב 5" if he else "# Phase 5 scientific summary"), "", f"**{WATERMARK_HE if he else WATERMARK}**", ""]
    if synthetic:
        L += [f"> **{SYNTHETIC_WATERMARK}**", ""]
    if interim:
        L += ["> INTERIM: primary comparison complete; domain / ablation / explanation / stability analyses still running.", ""]
    L += [("## שיטה" if he else "## Design"), "",
          (f"- {DESIGN_LABEL}: {n} מטופלים, {ev} נפילות; {plan['cv']['outer_folds']} קפלים חיצוניים × {plan['cv']['inner_folds']} פנימיים, שיוך קפלים קבוע אחד לכל המודלים והקבוצות." if he else
           f"- {DESIGN_LABEL}: {n} patients, {ev} events (Fall_Next_180D_Ind, strictly after the index day); {plan['cv']['outer_folds']} outer x "
           f"{plan['cv']['inner_folds']} inner stratified folds, one fixed assignment shared by every family and feature set."),
          ("- שאלה ראשית: ב-~70% רגישות, האם OLD_PLUS_ALL_NEW_ELIGIBLE מפחית התראות שווא לעומת OLD? משפחה ראשית מוצהרת מראש: Elastic Net; LASSO ו-XGBoost משניים." if he else
           f"- Primary question: at ~70% sensitivity, does {SET_ALL} reduce false alerts versus {SET_OLD}? Pre-declared primary family: ENET (elastic net); "
           "LASSO and XGBoost are secondary analyses of the same sets."),
          ("- כיוונון, early stopping, עיבוד מקדים, בחירת קונפיגורציה וספים – רק על תחזיות OOF פנימיות של נתוני האימון החיצוניים." if he else
           "- Preprocessing (medians, scaling, learned code levels), tuning, early stopping, configuration choice and every threshold use INNER out-of-fold "
           "predictions of the outer-training patients only."),
          f"- Schema: exact V1 -> V21 diff: V1 {sc.get('v1_columns')} columns, V21 {sc.get('extract_columns')}; unchanged {sc.get('unchanged_columns')}, removed "
          f"{sc.get('removed_v1_columns')}, new {sc.get('new_columns')}, renamed / replaced {sc.get('renamed_or_replaced')}, changed definition "
          f"{sc.get('changed_definition')}, unresolved {sc.get('unresolved_requires_semantic_review')}.",
          f"- OLD = {plan['n_old']} Phase 3 universe features reproducible on V21; {SET_ALL} adds {plan['n_all_new']} of the "
          f"{np_.get('genuine_new_predictors', '—')} genuine new V21 predictors; {SET_SAFE} adds {plan['n_new_safe']} (SAFE timing and DEFENSIBLE provenance).", ""]
    L += [("## תוצאה ראשית (כלל 70% מקונן)" if he else "## Primary result (nested 70% rule)"), "",
          "| family | role | set | sensitivity | PPV | false-alert share | flagged | AP | AUROC | Brier | BSS | cal. slope | cal. intercept |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t[t["feature_set"].isin(list(PRIMARY_SETS))].itertuples():
        L.append(f"| {r.family} | {r.role} | {r.feature_set} | {pct(r.op_sensitivity)} | {pct(r.op_ppv)} | {pct(r.op_false_alert_share)} | {pct(r.op_flagged_share)} | "
                 f"{r.ap:.3f} | {r.auroc:.3f} | {r.brier:.4f} | {r.brier_skill:.3f} | {r.calibration_slope:.2f} | {r.calibration_intercept:+.2f} |")
    L += ["", ("## השוואה מזווגת" if he else f"## Paired comparisons ({SET_OLD} vs {SET_ALL}; secondary {SET_OLD} vs {SET_SAFE})"), ""]
    for fam in (PRIMARY_FAMILY, *[f for f in FAMILIES if f != PRIMARY_FAMILY]):
        c, d = A["comparisons"][fam]["all"], A["decisions"][fam]
        tag = "PRIMARY" if fam == PRIMARY_FAMILY else "secondary"
        if c is None:
            L.append(f"- {fam} ({tag}): {d['verdict']} - {d['reason']}")
            continue
        L.append(f"- {fam} ({tag}), ALL_NEW: Δ false-alert share {pp(c['delta_op_false_alert_share'])} {ci(c['delta_op_false_alert_share_ci_low'], c['delta_op_false_alert_share_ci_high'], pp)}; "
                 f"Δ flagged {pp(c['delta_op_flagged_share'])}; Δ PPV {pp(c['delta_op_ppv'])} {ci(c['delta_op_ppv_ci_low'], c['delta_op_ppv_ci_high'], pp)}; "
                 f"Δ FP {num(c['delta_op_fp'])}; false alerts avoided / 10,000 {num(-c['delta_op_false_alerts_per_10000'])} "
                 f"{ci(-c['delta_op_false_alerts_per_10000_ci_high'], -c['delta_op_false_alerts_per_10000_ci_low'], num)}; Δ captured falls {num(c['delta_op_tp'])}; "
                 f"Δ AP {c['delta_ap']:+.4f} [{c['delta_ap_ci_low']:+.4f}, {c['delta_ap_ci_high']:+.4f}]; Δ AUROC {c['delta_auroc']:+.4f} "
                 f"[{c['delta_auroc_ci_low']:+.4f}, {c['delta_auroc_ci_high']:+.4f}]; Δ Brier {c['delta_brier']:+.5f}; descriptive pooled Δ false-alert share at 70% "
                 f"{pp(c.get('delta_desc_fas_0.70'))}; folds improved {c['folds_improved']}/{len(c['fold_delta_false_alert_share'])} -> **{d['verdict']}** ({d['reason']})")
        cs = A["comparisons"][fam]["safe"]
        if cs is not None and not A.get("safe_same"):
            L.append(f"  - {fam} NEW_SAFE: Δ false-alert share {pp(cs['delta_op_false_alert_share'])} {ci(cs['delta_op_false_alert_share_ci_low'], cs['delta_op_false_alert_share_ci_high'], pp)}; "
                     f"false alerts avoided / 10,000 {num(-cs['delta_op_false_alerts_per_10000'])}; folds improved {cs['folds_improved']}/{len(cs['fold_delta_false_alert_share'])}")
    L += ["", f"**Answer (pre-declared: the {PRIMARY_FAMILY} verdict): {A['overall']}**"
          + (f"; best family (exploratory only): {A['best_family_exploratory']}" if A.get("best_family_exploratory") else ""), "",
          f"Decision rule: {cfg['decision']['rule']}", ""]
    if len(A["domains"]) and "delta_false_alert_share" in A["domains"]:
        L += ["## Domain additions (OLD + domain vs OLD)", ""]
        for r in A["domains"].itertuples():
            if getattr(r, "status", "") == "COMPLETE":
                L.append(f"- {r.family} {r.domain} (+{r.n_new_features}): Δ false-alert share {pp(r.delta_false_alert_share)} "
                         f"{ci(r.delta_false_alert_share_ci_low, r.delta_false_alert_share_ci_high, pp)}, Δ false alerts {num(r.delta_false_alerts)}, Δ flagged "
                         f"{pp(r.delta_flagged_share)}, Δ PPV {pp(r.delta_ppv)}, Δ AP {r.delta_ap:+.4f}, Δ Brier {r.delta_brier:+.5f}, folds improved {int(r.folds_improved)}/{int(r.n_folds)}")
            else:
                L.append(f"- {r.family} {r.domain}: {r.status}")
        L.append("")
    if len(A["ablations"]) and "delta_false_alert_share" in A["ablations"]:
        L += [f"## Ablations ({SET_ALL} minus a block; {PRIMARY_FAMILY})", ""]
        for r in A["ablations"].itertuples():
            if getattr(r, "status", "") == "COMPLETE":
                L.append(f"- {r.family} {r.ablation} (-{r.n_removed}): Δ false-alert share {pp(r.delta_false_alert_share)} "
                         f"{ci(r.delta_false_alert_share_ci_low, r.delta_false_alert_share_ci_high, pp)}, Δ AP {r.delta_ap:+.4f} (> 0 pp = the block helped)")
        L.append("")
    L += ["## Limitations", "",
          "- Internal nested cross-validation on one 2026 snapshot (development data): not external validation; a later independent snapshot is required.",
          "- The nested operating results use thresholds chosen on inner OOF predictions; achieved outer sensitivity varies around 70%. The pooled-OOF "
          "threshold results are descriptive / planning-only.",
          "- Paired bootstrap intervals are conditional on the fitted models and fold thresholds (patient resampling of the OOF predictions); the "
          "every-fold criterion guards against training-to-training noise.",
          "- SAFE_ATTESTED features rely on the DWH statement that no post-index information enters the extract (re-checked by V3 on record dates); "
          "information-availability lag (registry backdating, coding / billing lag) is not verifiable from the extract.",
          "- V21 documents timing caveats for OLD inputs too (medication purchase status not bounded by Index_Date; visit counters may settle after it); "
          "these inputs are identical in every feature set, so they cannot create the OLD vs NEW difference, but they are not proven pre-index.",
          "- Fall outcomes are diagnosis records (falls and fractures); a positive after the personal follow-up end stops the run (zero tolerance, audited "
          "before any fit).",
          "- Three removed V1 registry flags (hypertension, chronic renal failure, transplant) were compared with the V21 COVID-19 (116 / 118), dialysis "
          "(101 / 1) and immunosuppression (130 / 131) registries: the V1 SQL / registry IDs are not available and V21 contradicts the V1 labels, so lineage "
          "is NOT proven (OLD_REMOVED_NEW_ADDED): the V1 features are removed from OLD and the V21 fields are genuinely new predictors.",
          "- The leakage screen (single-feature AUROC >= 0.80) uses every label but can only exclude, never select, a predictor.",
          "- Importance, coefficients and SHAP are descriptive, never causal.",
          f"- Historical 2025 reference (context only): threshold 0.02, sensitivity {pct(HISTORICAL['sensitivity'])}, PPV {pct(HISTORICAL['ppv'])}, "
          f"false-alert share {pct(HISTORICAL['false_alert_share'])}; never used as evidence of incremental value.", ""]
    return "\n".join(L) + "\n"


# ============================================================================ share + privacy
def _identifiers(src: Path, frame: pd.DataFrame) -> set[str]:
    from falls_ml.phase4.sealed import read_ids

    ids: set[str] = set()
    raw = read_ids(src, {}, extra=())
    for c in raw.columns:
        ids |= {str(v) for v in raw[c].dropna().astype(str) if len(str(v)) >= 6 and str(v).upper() != "NULL"}
    ids |= set(frame["row_key"].astype(str))
    return ids


def publish(out: Path, tmp: Path, src: Path, frame: pd.DataFrame) -> dict[str, Any]:
    from falls_ml.eda.runner import privacy_scan
    from falls_ml.phase2.stages_report import _path_hits

    scan = privacy_scan(tmp, _identifiers(src, frame), "")
    needles = [str(out.resolve()), str(src.resolve()), str(src.resolve().parent), os.environ.get("USERNAME", ""), getpass.getuser()]
    needles = [x for x in needles if x and len(x) >= 3]
    path_hits, row_level = [], []
    for p in sorted(tmp.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".csv", ".md", ".json", ".txt", ".svg"):
            h = _path_hits(p.read_text(encoding="utf-8", errors="ignore"), needles)
            if h:
                path_hits.append(f"{p.relative_to(tmp).as_posix()}: {sorted(set(h))}")
        if p.is_file() and p.suffix.lower() == ".csv":
            try:
                t = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                continue
            if set(t.columns) & ROW_LEVEL_COLUMNS or len(t) > 2000:
                row_level.append(p.relative_to(tmp).as_posix())
    forbidden = [p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file() and p.suffix.lower() in (".parquet", ".pkl", ".npz", ".npy", ".sqlite",
                                                                                                          ".jsonl", ".joblib", ".xlsx", ".feather")]
    if not scan["passed"] or path_hits or forbidden or row_level:
        shutil.rmtree(tmp)
        raise Phase2Stop("PRIVACY_SCAN", "the share package failed the privacy / path / file-type / row-level scan; nothing was published to share/",
                         [*scan["hits"][:10], *path_hits[:10], *[f"forbidden file type: {f}" for f in forbidden], *[f"row-level table: {f}" for f in row_level]])
    res = {**scan, "path_scan": "passed", "file_type_scan": "passed", "row_level_scan": "passed", "hits": [], "min_cell": MIN_CELL,
           "rule": "aggregate only: no identifiers, row keys, row-level predictions, raw data, model objects or local paths; counts 1-9 suppressed; "
                   "curves and threshold tables on a >= 0.5%-of-population grid"}
    D.write_json(tmp / "PRIVACY_SCAN.json", res)
    share = out / "share"
    if share.exists():
        shutil.rmtree(share)
    D.rename_dir(tmp, share)
    return res


def build_reports(ctx: Any, plan: dict[str, Any], cfg: Any, *, src: Path | None, interim: bool, mon: Any = None) -> dict[str, Any]:
    from falls_ml.phase5.explain import importance_tables, stability_table

    if src is None:
        raise Phase2Stop("PRIVACY_SCAN_NEEDS_INPUT", "the share folder is published only after the identifier scan against the input file: pass --input")
    out = ctx.out
    y = ctx.y.astype(float)
    A = analyse(ctx, plan, cfg, interim=interim)
    aw = out / "work" / "analysis"
    aw.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame({"row_key": ctx.frame["row_key"], "y": ctx.y, "outer_fold": ctx.outer})
    for (fam, s), run in A["runs"].items():
        if s in PRIMARY_SETS:
            oof[f"p__{fam}__{s}"] = run.p
            oof[f"flag70__{fam}__{s}"] = run.flags[f"{cfg.primary_sensitivity:.2f}"]
    D.write_parquet(aw / "OOF_PREDICTIONS_LOCAL.parquet", oof)
    if len(A["full_thresholds"]):
        D.write_csv(aw / "THRESHOLD_TABLE_EXHAUSTIVE_LOCAL.csv", A["full_thresholds"])
    perm, shap = importance_tables(ctx, list(FAMILIES), list(dict.fromkeys([plan["alias"][SET_OLD], plan["alias"][SET_ALL]])))
    stab = stability_table(ctx, list(FAMILIES), plan["alias"][SET_ALL], perm)
    tmp = out / ".tmp-share"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "figures").mkdir(parents=True)
    synthetic = bool(plan.get("synthetic"))
    D.write_str(tmp / "MANAGEMENT_SUMMARY_HE.md", management_he(A, plan, synthetic, interim))
    D.write_str(tmp / "SCIENTIFIC_SUMMARY_HE.md", scientific(A, plan, cfg, synthetic, interim, he=True))
    D.write_str(tmp / "SCIENTIFIC_SUMMARY.md", scientific(A, plan, cfg, synthetic, interim, he=False))
    pt = primary_table(A)
    if len(pt):
        keep = pt["row"].isin(["DELTA_NEW_MINUS_OLD", "VERDICT"])          # net differences between two models on the same patients: not cells
        pt = pd.concat([suppress(pt[~keep]), pt[keep].astype(object)]).sort_index()
    D.write_csv(tmp / "PRIMARY_70_SENSITIVITY_COMPARISON.csv", pt)
    D.write_csv(tmp / "THRESHOLD_TRADEOFF.csv", suppress_grid(A["grid"]))
    D.write_csv(tmp / "SENSITIVITY_TARGET_TABLE.csv", suppress(A["targets"]))
    mcols = [c for c in A["models"].columns if not c.startswith("desc_")] + [c for c in A["models"].columns if c.startswith("desc_")]
    D.write_csv(tmp / "MODEL_COMPARISON.csv", suppress(A["models"][mcols]))
    oofc = [c for c in ["family", "feature_set", "kind", "n", "events", "prevalence", "mean_predicted", "auroc", "auroc_ci_low", "auroc_ci_high", "ap", "ap_ci_low",
                        "ap_ci_high", "brier", "brier_ci_low", "brier_ci_high", "brier_skill", "logloss", "calibration_intercept", "calibration_slope", "oe_ratio",
                        "desc_fas_0.70", "desc_flagged_0.70"] if c in A["models"].columns]
    D.write_csv(tmp / "OOF_MODEL_COMPARISON.csv", suppress(A["models"][oofc]))
    D.write_csv(tmp / "OOF_PREDICTION_SUMMARY.csv", suppress(A["pred_summary"]))
    D.write_csv(tmp / "CALIBRATION.csv", suppress(A["calibration"]))
    D.write_csv(tmp / "CAPACITY_CURVE.csv", suppress(A["capacity"]))
    D.write_csv(tmp / "OUTER_FOLD_RESULTS.csv", suppress(A["outer_folds"]))
    D.write_csv(tmp / "DOMAIN_INCREMENTAL_VALUE.csv", A["domains"])
    D.write_csv(tmp / "ABLATION_RESULTS.csv", A["ablations"])
    D.write_csv(tmp / "SUBGROUP_SUMMARY.csv", suppress_subgroups(A["subgroups"]) if len(A["subgroups"]) else A["subgroups"])
    D.write_csv(tmp / "PERMUTATION_IMPORTANCE.csv", perm)
    D.write_csv(tmp / "FEATURE_STABILITY.csv", stab)
    if len(shap):
        D.write_csv(tmp / "SHAP_SUMMARY.csv", shap)
    pf = out / "preflight"
    for name in ("FEATURE_ELIGIBILITY.csv", "NEW_FEATURE_CATALOGUE.csv", "V21_UNDECLARED_COLUMNS.csv", "COHORT_FACTS_2026.json", "OUTCOME_CONTRACT_2026.json",
                 "PHASE5_PREFLIGHT.md", "FEATURE_SETS.json", "SCHEMA_DIFF_V1_V21.csv", "ALL_V21_COLUMN_CLASSIFICATION.csv", "REMOVED_V1_COLUMNS.csv",
                 "RENAMED_OR_CHANGED_COLUMNS.csv"):
        if (pf / name).is_file():
            shutil.copyfile(pf / name, tmp / name)
    for name in ("RUN_TIMINGS.csv", "ENVIRONMENT.json"):
        if (out / name).is_file():
            shutil.copyfile(out / name, tmp / name)
    D.write_json(tmp / "HISTORICAL_BENCHMARK_2025.json", HISTORICAL)
    figs = figures(tmp / "figures", A, y, perm, stab)
    status = json.loads((out / "RUN_STATUS.json").read_text(encoding="utf-8")) if (out / "RUN_STATUS.json").is_file() else {}
    D.write_json(tmp / "RUN_MANIFEST.json", {
        "watermark": WATERMARK, "synthetic": synthetic, "design": DESIGN_LABEL, "phase5_version": PHASE5_VERSION, "falls_ml_version": __import__("falls_ml").__version__,
        "report_kind": "INTERIM" if interim else "FINAL", "created_at": utc_now(), "mode": plan["mode"], "input": plan["input"], "config_sha256": plan["config_sha256"],
        "v21_schema_sha256": plan["v21_schema_sha256"], "v21_definition_sha256": plan["v21_definition_sha256"], "code_sha256_plan": plan["code_sha256"],
        "frame_sha256": plan["frame_sha256"], "schema_counts": plan.get("schema_counts"), "primary_family": PRIMARY_FAMILY,
        "primary_comparison": [SET_OLD, SET_ALL], "secondary_comparison": [SET_OLD, SET_SAFE],
        "folds_sha256": plan["folds_sha256"], "seed": plan["seed"], "cv": plan["cv"], "n": plan["n"], "events": plan["events"],
        "fold_sizes": plan["fold_sizes"], "fold_events": {k: ("<10" if 0 < int(v) < MIN_CELL else v) for k, v in plan["fold_events"].items()},
        "feature_sets": {k: {"n_features": len(v), "kind": plan["kinds"].get(plan["alias"][k]), "identical_to": plan["alias"][k] if plan["alias"][k] != k else None}
                         for k, v in plan["sets"].items()},
        "decisions": {f: {"verdict": d["verdict"], "criteria": d["criteria"], "role": "PRIMARY" if f == PRIMARY_FAMILY else "SECONDARY"}
                      for f, d in A["decisions"].items()}, "overall_answer": A["overall"], "overall_answer_basis": f"the {PRIMARY_FAMILY} verdict (pre-declared)",
        "best_family_exploratory": A["best_family_exploratory"], "jobs": status.get("jobs"), "device": status.get("device"), "sessions": status.get("sessions"),
        "failures": [{"item": f.get("item"), "error": f.get("error")} for f in status.get("failures", [])], "figures": figs,
        "row_level_outputs_kept_locally": ["work/analysis/OOF_PREDICTIONS_LOCAL.parquet", "work/analysis/THRESHOLD_TABLE_EXHAUSTIVE_LOCAL.csv", "work/units/*"]})
    res = publish(out, tmp, src, ctx.frame)
    n_files = sum(1 for p in (out / "share").rglob("*") if p.is_file())
    return {"files": n_files, "privacy_passed": bool(res["passed"]), "overall": A["overall"], "interim": interim,
            "decisions": {f: d["verdict"] for f, d in A["decisions"].items()}, "primary_family": PRIMARY_FAMILY}
