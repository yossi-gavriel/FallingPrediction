"""Aggregate analysis of the committed units (no fitting): pooled out-of-fold predictions, the nested operating results, the paired OLD vs NEW
comparisons, the pre-declared decision rule, domain additions, ablations, subgroups and the threshold tables.

Two operating views are always kept apart:
    NESTED (unbiased operational estimate)   each outer fold flags its holdout with the threshold chosen on ITS inner out-of-fold predictions;
                                             counts are summed over the folds; the achieved sensitivity is whatever it turned out to be
    POOLED OOF (descriptive / planning only) the concatenated outer-fold predictions re-thresholded at exactly the target sensitivity using the
                                             outcomes themselves - never called validation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase5.config import FAMILIES, SET_LOWRISK, SET_NEW, SET_OLD, SET_VERIFIED
from falls_ml.phase5.engine import Ctx, UnitSpec, is_complete, load_result
from falls_ml.phase5.metrics import calibration_table, model_summary, paired_bootstrap, single_bootstrap
from falls_ml.phase5.thresholds import counts, operating_point, threshold_table

STAGE_OF_KIND = {"PRIMARY": "PRIMARY", "SENSITIVITY": "PRIMARY", "DOMAIN": "DOMAIN", "ABLATION": "ABLATION"}


@dataclass
class ModelRun:
    family: str
    setname: str
    canonical: str
    p: np.ndarray
    flags: dict[str, np.ndarray]
    folds: list[dict[str, Any]] = field(default_factory=list)


def collect(ctx: Ctx, plan: dict[str, Any], cfg: Any) -> dict[tuple[str, str], ModelRun]:
    K = int(plan["cv"]["outer_folds"])
    out: dict[tuple[str, str], ModelRun] = {}
    n = len(ctx.y)
    for s in plan["sets"]:
        c = plan["alias"][s]
        stage = STAGE_OF_KIND[plan["kinds"][c]]
        for fam in FAMILIES:
            specs = [UnitSpec(uid=f"{stage}|{fam}|{c}|outer{k}", stage=stage, family=fam, setname=c, outer=k) for k in range(K)]
            if not all(is_complete(ctx, u) for u in specs):
                continue
            p = np.full(n, np.nan)
            flags = {f"{t:.2f}": np.zeros(n, dtype=bool) for t in cfg.targets}
            folds = []
            for u in specs:
                r = load_result(ctx, u)
                te, pt = r["arrays"]["test_idx"], r["arrays"]["p_test"]
                p[te] = pt
                for t, thr in r["thresholds"].items():
                    flags[t][te] = pt >= float(thr) if thr is not None and math.isfinite(float(thr)) else False
                folds.append({"outer": u.outer, "thresholds": r["thresholds"], "config": r["config"], "inner_op": r["inner_op"], "seconds": r["seconds"],
                              "n_test": r["n_test"], "test_idx": te, "not_converged_fits": r.get("not_converged_fits", 0),
                              "edge": r.get("lambda_on_grid_edge", False)})
            out[(fam, s)] = ModelRun(family=fam, setname=s, canonical=c, p=p, flags=flags, folds=folds)
    return out


def nested_by_target(y: np.ndarray, run: ModelRun, targets: list[float]) -> pd.DataFrame:
    rows = []
    for t in targets:
        key = f"{t:.2f}"
        c = counts(y, run.flags[key])
        thr = [float(f["thresholds"][key]) for f in run.folds if f["thresholds"].get(key) is not None]
        desc = operating_point(y, run.p, t)
        rows.append({"family": run.family, "feature_set": run.setname, "target_sensitivity": t,
                     "nested_threshold_min": min(thr) if thr else math.nan, "nested_threshold_median": float(np.median(thr)) if thr else math.nan,
                     "nested_threshold_max": max(thr) if thr else math.nan, **{f"nested_{k}": v for k, v in c.items()},
                     "descriptive_threshold": desc["threshold"], **{f"descriptive_{k}": desc[k] for k in ("sensitivity", "ppv", "false_alert_share",
                                                                                                         "flagged_share", "fp", "tp", "flagged",
                                                                                                         "false_alerts_per_10000", "flagged_per_10000")}})
    return pd.DataFrame(rows)


def model_row(y: np.ndarray, run: ModelRun, *, target: float, n_boot: int, seed: int, with_ci: bool) -> dict[str, Any]:
    key = f"{target:.2f}"
    s = model_summary(y, run.p, run.flags[key], targets=(target,))
    row = {"family": run.family, "feature_set": run.setname, "n_features": None, **s}
    if with_ci:
        row.update(single_bootstrap(y, run.p, run.flags[key], n_boot=n_boot, seed=seed, targets=(target,)))
    return row


def fold_deltas(y: np.ndarray, a: ModelRun, b: ModelRun, target: float) -> list[float]:
    key = f"{target:.2f}"
    out = []
    for fa in a.folds:
        te = fa["test_idx"]
        ca, cb = counts(y[te], a.flags[key][te]), counts(y[te], b.flags[key][te])
        out.append(cb["false_alert_share"] - ca["false_alert_share"])
    return out


def compare(y: np.ndarray, a: ModelRun, b: ModelRun, *, target: float, n_boot: int, seed: int) -> dict[str, Any]:
    key = f"{target:.2f}"
    r = paired_bootstrap(y, a.p, b.p, a.flags[key], b.flags[key], n_boot=n_boot, seed=seed, targets=(target,))
    r["fold_delta_false_alert_share"] = fold_deltas(y, a, b, target)
    r["folds_improved"] = int(sum(1 for v in r["fold_delta_false_alert_share"] if v < 0))
    return r


# ============================================================================ the pre-declared decision rule
def decide(cmp_new: dict[str, Any] | None, cmp_ver: dict[str, Any] | None, cmp_low: dict[str, Any] | None, *, no_new: bool, ver_same: bool, low_same: bool,
           cfg: Any) -> dict[str, Any]:
    if no_new:
        return {"verdict": "NO_ELIGIBLE_NEW_FEATURES", "criteria": {}, "reason": "no new V21 predictor passed the gates: OLD_PLUS_NEW_SAFE = OLD"}
    if cmp_new is None:
        return {"verdict": "INCOMPLETE", "criteria": {}, "reason": "the OLD / OLD_PLUS_NEW_SAFE units of this family are not all complete"}
    dc = cfg["decision"]
    op_d = cmp_new["delta_op_false_alert_share"]
    desc_d = cmp_new.get("delta_desc_fas_0.70")
    c1 = bool(op_d < 0 and desc_d is not None and desc_d < 0)
    folds = cmp_new.get("fold_delta_false_alert_share") or []
    c2 = bool(cmp_new["delta_op_false_alert_share_ci_high"] < 0 and folds and all(v < 0 for v in folds))

    def same_dir(c: dict[str, Any] | None, same: bool) -> bool:
        if same:
            return c1
        return bool(c is not None and c["delta_op_false_alert_share"] < 0)

    c3 = same_dir(cmp_ver, ver_same) and same_dir(cmp_low, low_same)
    a, b = cmp_new["a"], cmp_new["b"]
    lo, hi = dc["calibration_slope_range"]
    mx = float(dc["max_abs_calibration_intercept"])

    def ok_range(m: dict[str, Any]) -> bool:
        return bool(lo <= m["calibration_slope"] <= hi and abs(m["calibration_intercept"]) <= mx)

    no_worse = bool(abs(b["calibration_slope"] - 1) <= abs(a["calibration_slope"] - 1) + 1e-12 and abs(b["calibration_intercept"]) <= abs(a["calibration_intercept"]) + 1e-12)
    c4 = bool((ok_range(b) or no_worse) and not (cmp_new["delta_brier_ci_low"] > 0))
    crit = {"1_lower_false_alert_share_nested_and_descriptive": c1, "2_paired_ci_below_zero_and_every_outer_fold_improves": c2, "3_not_dependent_on_attested_bounded_or_high_risk": c3,
            "4_calibration_not_materially_worse": c4}
    if c1 and c2 and c3 and c4:
        v = "NEW_FEATURES_OPERATIONALLY_USEFUL"
    elif op_d < 0:
        v = "PROMISING_BUT_NOT_ROBUST"
    else:
        v = "NO_ROBUST_OPERATIONAL_GAIN"
    return {"verdict": v, "criteria": crit, "reason": "; ".join(k for k, x in crit.items() if not x) or "all four criteria met"}


def overall(verdicts: dict[str, str]) -> str:
    vs = list(verdicts.values())
    if vs and all(v == "NO_ELIGIBLE_NEW_FEATURES" for v in vs):
        return "NO"
    if sum(v == "NEW_FEATURES_OPERATIONALLY_USEFUL" for v in vs) >= 2:
        return "YES"
    if not any(v in ("NEW_FEATURES_OPERATIONALLY_USEFUL", "PROMISING_BUT_NOT_ROBUST") for v in vs):
        return "NO" if all(v != "INCOMPLETE" for v in vs) else "UNCERTAIN"
    return "UNCERTAIN"


# ============================================================================ subgroups
SUBGROUPS = (("age", "sg_age"), ("sex", "sg_sex"), ("prior_fall", "sg_prior_fall"), ("mefi_group", "sg_mefi_group"),
             ("frailty_assessment", "sg_frailty_assessed"), ("data_coverage", "sg_data_coverage"))


def subgroup_rows(ctx: Ctx, runs: dict[tuple[str, str], ModelRun], cfg: Any) -> pd.DataFrame:
    from falls_ml.phase5.metrics import point_metrics

    sc = cfg["subgroups"]
    y = ctx.y.astype(float)
    key = f"{cfg.primary_sensitivity:.2f}"
    rows = []
    mefi_cov = float((ctx.frame["sg_mefi_group"] != "not assessed").mean()) if "sg_mefi_group" in ctx.frame else 0.0
    for (fam, s), run in runs.items():
        if s not in (SET_OLD, SET_NEW):
            continue
        for name, col in SUBGROUPS:
            if col not in ctx.frame:
                continue
            if name == "mefi_group" and mefi_cov < float(sc["mefi_min_coverage"]):
                continue
            for lev in sorted(ctx.frame[col].astype(str).unique()):
                m = (ctx.frame[col].astype(str) == lev).to_numpy()
                n, ev = int(m.sum()), int(y[m].sum())
                c = counts(y[m], run.flags[key][m])
                powered = ev >= int(sc["min_events"]) and (n - ev) >= int(sc["min_nonevents"])
                pm = point_metrics(y[m], run.p[m]) if powered else {}
                rows.append({"family": fam, "feature_set": s, "subgroup": name, "level": lev, "n": n, "events": ev, "adequately_powered": powered,
                             "auroc": pm.get("auroc") if powered else None, "ap": pm.get("ap") if powered else None,
                             "sensitivity": c["sensitivity"] if powered else None, "ppv": c["ppv"] if powered else None,
                             "false_alert_share": c["false_alert_share"] if powered else None, "flagged_share": c["flagged_share"] if powered else None,
                             "note": "descriptive; the operating threshold is the overall nested one (no subgroup thresholds)"})
    return pd.DataFrame(rows)


def threshold_tables(y: np.ndarray, runs: dict[tuple[str, str], ModelRun], step: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(shareable grid table for OLD / OLD_PLUS_NEW_SAFE, exhaustive local table)."""
    grid, full = [], []
    for (fam, s), run in runs.items():
        if s not in (SET_OLD, SET_NEW):
            continue
        g = threshold_table(y, run.p, grid_step=step)
        g.insert(0, "feature_set", s)
        g.insert(0, "family", fam)
        grid.append(g)
        f = threshold_table(y, run.p)
        f.insert(0, "feature_set", s)
        f.insert(0, "family", fam)
        full.append(f)
    return (pd.concat(grid, ignore_index=True) if grid else pd.DataFrame(), pd.concat(full, ignore_index=True) if full else pd.DataFrame())


def calibration_rows(y: np.ndarray, runs: dict[tuple[str, str], ModelRun], groups: int = 10) -> pd.DataFrame:
    out = []
    for (fam, s), run in runs.items():
        if s not in (SET_OLD, SET_NEW, SET_VERIFIED, SET_LOWRISK):
            continue
        t = calibration_table(y, run.p, groups)
        t.insert(0, "feature_set", s)
        t.insert(0, "family", fam)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def prediction_summary(y: np.ndarray, runs: dict[tuple[str, str], ModelRun]) -> pd.DataFrame:
    out = []
    qs = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    for (fam, s), run in runs.items():
        for lab, m in (("all", np.ones(len(y), dtype=bool)), ("fallers", y == 1), ("non_fallers", y == 0)):
            v = run.p[m]
            out.append({"family": fam, "feature_set": s, "group": lab, "n": int(m.sum()), "mean": float(v.mean()) if len(v) else math.nan,
                        **{f"p{int(q * 100):02d}": float(np.quantile(v, q)) if len(v) else math.nan for q in qs}})
    return pd.DataFrame(out)
