"""Operating CAPACITY analysis of the completed Phase 5 run - analysis / reporting only: no model is fitted, tuned or re-validated, and the
committed outer out-of-fold (OOF) predictions are read, never changed.

The question: "if we can intervene on X% of the population, how many falls do we capture?"

PRIMARY (outer-fold capacity)   for a requested capacity c the total number of interventions is T = round(c x N) (half up, integer arithmetic);
                                T is allocated across the outer folds in proportion to their sizes by the deterministic largest-remainder rule
                                (ties -> lower fold number), so the folds' allocations sum to T exactly; inside each fold the k_f patients with
                                the highest OOF risk of THAT fold's model are selected (equal risks keep the frozen row order, which is the
                                pseudonymous row-key order - a reported, deterministic tie rule); the confusion counts are summed over the folds.
                                Probabilities of different outer models are therefore never compared with each other.
POOLED OOF (descriptive only)   the concatenated OOF risks ranked globally and the top T selected - planning / descriptive, never the estimate.

At a fixed capacity both models flag the same number of patients, so every additional fall captured is exactly one false intervention
avoided (delta FP = - delta TP; delta PPV = delta TP / T).

Uncertainty at the operating point: a paired patient bootstrap (multinomial weights identical for every model); in every replicate the
capacity is re-allocated across the folds from the resampled fold sizes and the top-k selection is recomputed within each fold.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase5.config import FAMILIES, PRIMARY_FAMILY, SET_ALL, SET_OLD, SET_SAFE
from falls_ml.phase5.thresholds import derive

PRIMARY_SETS = (SET_OLD, SET_ALL, SET_SAFE)
COMPARISONS = ((SET_OLD, SET_ALL, "PRIMARY"), (SET_OLD, SET_SAFE, "SECONDARY"))
GRID_MIN, GRID_MAX = 5, 200                      # capacity grid in per-mille of the population: 0.5% ... 20.0%
EXT_STEP, EXT_MAX = 5, 1000                      # extended grid (20.5% ... 100%, 0.5% steps) - only for the "target fall capture" mode
TARGET = 30                                      # the operational target: 3.0%
PRESETS = (10, 20, 30, 50, 75, 100, 150, 200)
CAPTURE_TARGETS = (0.50, 0.60, 0.70, 0.80)
MIN_CELL = 10
STEP_CHOICES = (1, 2, 5, 10)                     # every choice divides 30, so 3.0% is always an exact grid point


def n_selected(permille: int, n: int) -> int:
    """round(permille / 1000 x n), half up, in exact integer arithmetic (no floating-point drift at 3.00%)."""
    return (2 * int(permille) * int(n) + 1000) // 2000


def allocate(total: int, sizes: list[int] | np.ndarray) -> np.ndarray:
    """Largest-remainder allocation of ``total`` across groups proportional to ``sizes`` (exact integers; ties -> lower group index)."""
    s = np.asarray(sizes, dtype=np.int64)
    N = int(s.sum())
    if N <= 0:
        return np.zeros(len(s), dtype=np.int64)
    num = int(total) * s
    base = num // N
    rem = num % N
    left = int(total) - int(base.sum())
    order = sorted(range(len(s)), key=lambda f: (-int(rem[f]), f))
    base[order[:left]] += 1
    return np.minimum(base, s)


@dataclass
class Ranked:
    """One model's outer-OOF risks ranked WITHIN each outer fold (stable: equal risks keep the frozen row order)."""
    family: str
    setname: str
    folds: list[dict[str, np.ndarray]]           # per fold: idx (row indices, descending risk), ys, ps, cum_tp
    pooled_idx: np.ndarray


def rank_model(family: str, setname: str, y: np.ndarray, outer: np.ndarray, p: np.ndarray) -> Ranked:
    folds = []
    for f in range(int(outer.max()) + 1):
        m = np.flatnonzero(outer == f)
        o = m[np.argsort(-p[m], kind="mergesort")]
        ys = y[o].astype(np.int64)
        folds.append({"idx": o, "ys": ys, "ps": p[o], "cum_tp": np.r_[0, np.cumsum(ys)]})
    return Ranked(family=family, setname=setname, folds=folds, pooled_idx=np.argsort(-p, kind="mergesort"))


def _ties_split(ps: np.ndarray, k: int) -> bool:
    return bool(0 < k < len(ps) and ps[k - 1] == ps[k])


def fold_capacity(y: np.ndarray, r: Ranked, total: int) -> dict[str, Any]:
    """PRIMARY: T allocated across the outer folds, top-k_f within each fold, counts summed."""
    sizes = [len(f["idx"]) for f in r.folds]
    k = allocate(total, sizes)
    tp = int(sum(int(f["cum_tp"][kf]) for f, kf in zip(r.folds, k)))
    E, N = int(y.sum()), len(y)
    fp = int(k.sum()) - tp
    out = derive(tp, fp, E - tp, N - E - fp)
    out.update({"per_fold_selected": [int(v) for v in k], "ties_split_at_boundary": any(_ties_split(f["ps"], int(kf)) for f, kf in zip(r.folds, k))})
    return out


def fold_rows(y: np.ndarray, r: Ranked, total: int) -> list[dict[str, Any]]:
    sizes = [len(f["idx"]) for f in r.folds]
    k = allocate(total, sizes)
    rows = []
    for fi, (f, kf) in enumerate(zip(r.folds, k)):
        n_f, E_f = len(f["idx"]), int(f["ys"].sum())
        tp = int(f["cum_tp"][kf])
        fp = int(kf) - tp
        rows.append({"family": r.family, "feature_set": r.setname, "outer_fold": fi, "fold_patients": n_f, "selected": int(kf),
                     **_metrics(derive(tp, fp, E_f - tp, n_f - E_f - fp)), "ties_split_at_boundary": _ties_split(f["ps"], int(kf))})
    return rows


def pooled_capacity(y: np.ndarray, r: Ranked, total: int) -> dict[str, Any]:
    """DESCRIPTIVE ONLY: the pooled OOF risks ranked globally (different outer models' probability scales mixed)."""
    E, N = int(y.sum()), len(y)
    tp = int(y[r.pooled_idx[:total]].sum())
    fp = int(total) - tp
    return derive(tp, fp, E - tp, N - E - fp)


def _metrics(c: dict[str, Any]) -> dict[str, Any]:
    flagged = c["flagged"]
    return {"selected_total": flagged, "falls_total": c["events"], "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
            "sensitivity": c["sensitivity"], "ppv": c["ppv"], "false_alert_share": c["false_alert_share"], "specificity": c["specificity"],
            "fpr": c["fpr"], "lift": c["lift"], "nni_per_captured_fall": c["flagged_per_capture"],
            "captured_per_1000_interventions": 1000 * c["tp"] / flagged if flagged else math.nan,
            "false_per_1000_interventions": 1000 * c["fp"] / flagged if flagged else math.nan}


# ============================================================================ grids and privacy
def grid_step(n: int) -> int:
    """The smallest capacity step (per-mille) whose neighbouring grid points differ by >= MIN_CELL patients (0.1% when N >= 10,000)."""
    for s in STEP_CHOICES:
        if s * n / 1000 >= MIN_CELL:
            return s
    return STEP_CHOICES[-1]


def fine_grid(n: int) -> list[int]:
    s = grid_step(n)
    return sorted({k for k in range(s, GRID_MAX + 1, s) if k >= GRID_MIN} | {TARGET})


def extended_grid() -> list[int]:
    return list(range(GRID_MAX + EXT_STEP, EXT_MAX + 1, EXT_STEP))


def cells_safe(c: dict[str, Any]) -> bool:
    return all(not (0 < int(c[k]) < MIN_CELL) for k in ("tp", "fp", "fn", "tn"))


def curve(y: np.ndarray, ranked: dict[tuple[str, str], Ranked], grid: list[int], *, pooled: bool = False) -> pd.DataFrame:
    N = len(y)
    rows = []
    for (fam, s), r in ranked.items():
        for pm in grid:
            T = n_selected(pm, N)
            c = pooled_capacity(y, r, T) if pooled else fold_capacity(y, r, T)
            rows.append({"family": fam, "feature_set": s, "capacity_permille": pm, "capacity_pct": pm / 10, "population": N, **_metrics(c),
                         "actual_intervention_pct": 100 * T / N if N else math.nan, "cells_safe": cells_safe(c),
                         **({} if pooled else {"ties_split_at_boundary": c["ties_split_at_boundary"]}),
                         "method": "POOLED_OOF_DESCRIPTIVE_ONLY" if pooled else "OUTER_FOLD_CAPACITY_PRIMARY"})
    return pd.DataFrame(rows)


def safe_points(cv: pd.DataFrame) -> tuple[list[int], list[int]]:
    """(capacity points whose four confusion cells are 0 or >= 10 for EVERY model, the unsafe points)."""
    ok = cv.groupby("capacity_permille")["cells_safe"].all()
    return [int(k) for k, v in ok.items() if v], [int(k) for k, v in ok.items() if not v]


# ============================================================================ the paired bootstrap at one capacity
def weighted_tp(r: Ranked, w: np.ndarray, k: np.ndarray, prefix: list[int] | None = None) -> int:
    """Captured falls when fold f selects its k[f] highest-risk resampled copies (patient i appears w[i] times; copies are interchangeable)."""
    t = 0
    for i, f in enumerate(r.folds):
        kf = int(k[i])
        if kf == 0:
            continue
        m = prefix[i] if prefix is not None else len(f["idx"])
        ws = w[f["idx"][:m]]
        cs = np.cumsum(ws)
        if cs[-1] < kf and m < len(f["idx"]):                         # the prefix holds too few resampled copies: use the whole fold
            m = len(f["idx"])
            ws = w[f["idx"]]
            cs = np.cumsum(ws)
        sel = np.clip(kf - (cs - ws), 0, ws)
        t += int(sel @ f["ys"][:m])
    return t


def capacity_bootstrap(y: np.ndarray, outer: np.ndarray, ranked: dict[tuple[str, str], Ranked], permille: int, *, n_boot: int, seed: int,
                       progress: Any = None) -> dict[str, Any]:
    """Paired patient bootstrap at a fixed capacity: identical multinomial weights for every model; the capacity is re-allocated across the
    folds from the resampled fold sizes and the top-k selection recomputed within each fold in every replicate (resampled copies of one patient
    are interchangeable, so the boundary patient may contribute part of its copies)."""
    y = np.asarray(y, dtype=np.int64)
    N = len(y)
    T = n_selected(permille, N)
    K = int(outer.max()) + 1
    members = [np.flatnonzero(outer == f) for f in range(K)]
    k0 = allocate(T, [len(m) for m in members])
    pref = {key: [min(len(f["idx"]), 4 * int(k0[i]) + 200) for i, f in enumerate(r.folds)] for key, r in ranked.items()}
    rng = np.random.default_rng(seed)
    tp = {key: np.zeros(n_boot) for key in ranked}
    E = np.zeros(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(np.int64)
        k = allocate(T, [int(w[m].sum()) for m in members])
        E[b] = float((w * y).sum())
        for key, r in ranked.items():
            tp[key][b] = weighted_tp(r, w, k, pref[key])
        if progress is not None and (b + 1) % 200 == 0:
            progress(b + 1, n_boot)
    return {"tp": tp, "events": E, "selected": T, "n": N, "n_boot": int(n_boot), "seed": int(seed), "permille": int(permille)}


def bootstrap_tables(y: np.ndarray, ranked: dict[tuple[str, str], Ranked], boot: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(TOP3_CAPACITY_COMPARISON rows: point delta + percentile 95% CI; TOP3_CAPACITY_BOOTSTRAP rows: the replicate distribution summary)."""
    N, T = boot["n"], boot["selected"]
    E0 = int(y.sum())
    cmp_rows, dist_rows = [], []
    for fam in FAMILIES:
        for a, b, role in COMPARISONS:
            if (fam, a) not in ranked or (fam, b) not in ranked:
                continue
            ca, cb = fold_capacity(y, ranked[(fam, a)], T), fold_capacity(y, ranked[(fam, b)], T)
            d_tp = boot["tp"][(fam, b)] - boot["tp"][(fam, a)]
            with np.errstate(divide="ignore", invalid="ignore"):
                reps = {"delta_falls_captured": d_tp, "delta_sensitivity": d_tp / boot["events"], "delta_ppv": d_tp / T if T else d_tp * np.nan,
                        "delta_false_interventions": -d_tp}
            point = {"delta_falls_captured": cb["tp"] - ca["tp"], "delta_sensitivity": (cb["tp"] - ca["tp"]) / E0 if E0 else math.nan,
                     "delta_ppv": (cb["tp"] - ca["tp"]) / T if T else math.nan, "delta_false_interventions": cb["fp"] - ca["fp"]}
            row = {"family": fam, "role": "PRIMARY" if (fam == PRIMARY_FAMILY and role == "PRIMARY") else
                   ("SECONDARY_SAFE" if fam == PRIMARY_FAMILY else "SECONDARY_FAMILY"),
                   "comparison": f"{b} minus {a}", "capacity_pct": boot["permille"] / 10, "population": N, "selected_total": T,
                   "falls_total": E0, "tp_a": ca["tp"], "tp_b": cb["tp"], "fp_a": ca["fp"], "fp_b": cb["fp"]}
            for name, v in point.items():
                col = reps[name][np.isfinite(reps[name])]
                lo, hi = (float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))) if col.size else (math.nan, math.nan)
                row[name] = v
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = lo, hi
                dist_rows.append({"family": fam, "comparison": f"{b} minus {a}", "metric": name, "capacity_pct": boot["permille"] / 10, "point": v,
                                  "bootstrap_mean": float(col.mean()) if col.size else math.nan,
                                  "bootstrap_sd": float(col.std(ddof=1)) if col.size > 1 else math.nan, "ci_low_2_5": lo, "ci_high_97_5": hi,
                                  "share_of_replicates_new_better": float((col > 0).mean() if name != "delta_false_interventions" else (col < 0).mean())
                                  if col.size else math.nan,
                                  "share_of_replicates_equal": float((col == 0).mean()) if col.size else math.nan,
                                  "n_boot": boot["n_boot"], "seed": boot["seed"],
                                  "method": "paired patient bootstrap (identical multinomial weights for both models); capacity re-allocated across the outer "
                                            "folds and the top-k re-selected within each fold in every replicate; percentile 95% interval; conditional on the "
                                            "fitted outer-fold models"})
            row["ci_excludes_zero"] = bool(row["delta_falls_captured_ci_low"] > 0 or row["delta_falls_captured_ci_high"] < 0)
            cmp_rows.append(row)
    return pd.DataFrame(cmp_rows), pd.DataFrame(dist_rows)


# ============================================================================ loading the committed outer-OOF predictions (read-only)
def load_primary_oof(ctx: Any, plan: dict[str, Any]) -> dict[tuple[str, str], np.ndarray]:
    """{(family, primary set): outer-OOF risk of every patient} from the committed PRIMARY units (integrity-checked, never recomputed)."""
    from falls_ml.phase5.engine import UnitSpec, is_complete, load_result

    K = int(plan["cv"]["outer_folds"])
    n = len(ctx.y)
    out: dict[tuple[str, str], np.ndarray] = {}
    for s in PRIMARY_SETS:
        if s not in plan["alias"]:
            continue
        c = plan["alias"][s]
        for fam in FAMILIES:
            specs = [UnitSpec(uid=f"PRIMARY|{fam}|{c}|outer{k}", stage="PRIMARY", family=fam, setname=c, outer=k) for k in range(K)]
            if not all(is_complete(ctx, u) for u in specs):
                continue
            p = np.full(n, np.nan)
            for u in specs:
                r = load_result(ctx, u)
                te, pt = np.asarray(r["arrays"]["test_idx"]), np.asarray(r["arrays"]["p_test"], dtype=float)
                if not np.array_equal(np.sort(te), np.flatnonzero(ctx.outer == u.outer)):
                    raise Phase2Stop("OOF_INTEGRITY", f"unit {u.uid}: its holdout rows differ from outer fold {u.outer} of the frozen folds")
                p[te] = pt
            if not np.isfinite(p).all():
                raise Phase2Stop("OOF_INTEGRITY", f"{fam} {s}: some patients have no finite outer-OOF prediction")
            out[(fam, s)] = p
    return out


# ============================================================================ what drives each model (descriptive, never causal)
def _labels() -> dict[str, str]:
    import yaml

    from falls_ml.paths import resolve_path

    try:
        raw = yaml.safe_load(resolve_path("configs/meuhedet/feature_labels_he.yaml").read_text(encoding="utf-8")) or {}
        return {k: str((v or {}).get("he", k)) for k, v in (raw.get("features") or {}).items()}
    except Exception:  # noqa: BLE001 - labels are cosmetic
        return {}


def feature_drivers(perm: pd.DataFrame, shap: pd.DataFrame, stab: pd.DataFrame, registry: pd.DataFrame, setname: str, schema: Any = None) -> pd.DataFrame:
    """One row per (family, feature) of the OLD_PLUS_ALL_NEW_ELIGIBLE models: the importance used for ranking (ENET / LASSO: |median standardised
    bootstrap coefficient| of the final model's stability refits; XGBoost: mean |SHAP| over the outer-fold models; permutation importance when the
    primary measure is not available), the permutation rank, stability and the OLD / NEW / SAFE status. Descriptive, never causal."""
    lab = _labels()
    reg = registry.set_index("feature") if len(registry) else pd.DataFrame()
    he_new = {f"new_{c.lower()}": p.text_he for c, p in (schema.predictors.items() if schema is not None else [])}
    rows = []
    for fam in FAMILIES:
        pm = perm[(perm["family"] == fam) & (perm["feature_set"] == setname)] if len(perm) else pd.DataFrame()
        sh = shap[(shap["family"] == fam) & (shap["feature_set"] == setname)] if len(shap) else pd.DataFrame()
        st = stab[(stab["family"] == fam) & (stab["feature_set"] == setname)] if len(stab) else pd.DataFrame()
        feats = list(dict.fromkeys([*pm.get("feature", pd.Series(dtype=str)), *sh.get("feature", pd.Series(dtype=str)), *st.get("feature", pd.Series(dtype=str))]))
        if not feats:
            continue
        pmi, shi, sti = (t.set_index("feature") if len(t) else pd.DataFrame() for t in (pm, sh, st))
        for f in feats:
            r = {"family": fam, "feature": f}
            if f in reg.index:
                origin = str(reg.at[f, "origin"])
                r["origin"] = "NEW" if origin == "NEW_V21" else "OLD"
                r["domain"] = str(reg.at[f, "domain"])
                safe = str(reg.at[f, "in_OLD_PLUS_NEW_SAFE"]).lower() in ("true", "1")
                r["new_status"] = ("SAFE" if safe else "ALL_NEW_ONLY") if r["origin"] == "NEW" else "OLD"
            else:
                r.update({"origin": "NEW" if f.startswith("new_") else "OLD", "domain": "", "new_status": ""})
            r["label_he"] = he_new.get(f) or lab.get(f) or f
            if f in pmi.index:
                r.update({"permutation_rank": int(pmi.at[f, "rank"]), "permutation_mean_drop_ap": float(pmi.at[f, "mean_drop_ap"]),
                          "permutation_folds_in_top10": int(pmi.at[f, "folds_in_top10"]), "permutation_n_folds": int(pmi.at[f, "n_folds"])})
            if f in shi.index:
                r.update({"shap_rank": int(shi.at[f, "rank"]), "mean_abs_shap": float(shi.at[f, "mean_abs_shap"]),
                          "shap_folds_in_top10": int(shi.at[f, "folds_in_top10"])})
            if f in sti.index:
                for c in ("selection_frequency", "sign_consistency", "coefficient_median", "top10_frequency", "rank_median", "replicates"):
                    if c in sti.columns and pd.notna(sti.at[f, c]):
                        r[c] = float(sti.at[f, c])
                r["stability_flag"] = str(sti.at[f, "flag"]) if "flag" in sti.columns and pd.notna(sti.at[f, "flag"]) else ""
            rows.append(r)
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    for c in ("coefficient_median", "mean_abs_shap", "permutation_mean_drop_ap", "selection_frequency", "sign_consistency", "top10_frequency"):
        if c not in t:
            t[c] = np.nan
    out = []
    for fam, g in t.groupby("family", sort=False):
        g = g.copy()
        if fam in ("LASSO", "ENET") and g["coefficient_median"].notna().any():
            g["importance"], g["importance_measure"] = g["coefficient_median"].abs(), "|median standardised coefficient| (bootstrap refits of the final model)"
        elif fam == "XGB" and g["mean_abs_shap"].notna().any():
            g["importance"], g["importance_measure"] = g["mean_abs_shap"], "mean |SHAP| (outer-fold models, holdout samples)"
        else:
            g["importance"], g["importance_measure"] = g["permutation_mean_drop_ap"], "permutation importance (mean drop in AP, outer-fold holdouts)"
        g = g.sort_values(["importance", "feature"], ascending=[False, True], na_position="last").reset_index(drop=True)
        g["rank_overall"] = np.arange(1, len(g) + 1)
        g["rank_within_origin"] = g.groupby("origin").cumcount() + 1
        if fam in ("LASSO", "ENET"):
            g["stable"] = (g["selection_frequency"] >= 0.6) & (g["sign_consistency"] >= 0.9) & (g["coefficient_median"].fillna(0) != 0)
            g["direction"] = np.where(g["coefficient_median"] > 0, "higher risk", np.where(g["coefficient_median"] < 0, "lower risk", ""))
        else:
            g["stable"] = g["top10_frequency"] >= 0.5
            g["direction"] = ""
        out.append(g)
    t = pd.concat(out, ignore_index=True)
    t["note"] = "descriptive association with the model's prediction, never causal"
    return t


def read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))
