"""Partial-identification bounds for patients whose pre-index history is UNKNOWN (planning/PHASE3_DESIGN.md §3).

For an UNKNOWN row the true pre-index state of a source is not in the extract. It is represented - never imputed - by the SET of pre-index states
observed among the model's fitting rows for that source (the source's features jointly, "profiles"), restricted by the declared cumulative upper
bounds (a 'since 2022' value cannot exceed the observed one). A fitted model then gives every row an interval [lo, hi] of predicted risk:

- linear models (LASSO / elastic net): EXACT min / max over the allowed profiles - the linear predictor is additive over design columns and every
  design column belongs to one source, so each source contributes its own min / max (sources vary independently);
- XGBoost: a SOUND OUTER bound by interval propagation through every tree (a split on an UNKNOWN feature may follow every branch that an allowed
  value of that feature can reach); it can only make XGBoost look worse, never better.

Rows without an UNKNOWN used feature get lo = hi = the model's ordinary prediction (verified by a built-in self-check). Metrics are computed under two
assignments: ADVERSE (events get lo, non-events get hi: the worst case for the model on every rank metric, log loss and Brier) and FAVOURABLE
(the reverse). For a comparison new vs reference, the adverse difference is new-adverse minus reference-favourable (the most conservative).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from falls_ml.phase2.state import Phase2Stop

MAX_PROFILES = 20000
SENTINEL = -9.87654321e300          # NaN stand-in for de-duplicating profiles


@dataclass
class Intervals:
    lo: np.ndarray
    hi: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def exact(self) -> bool:
        return bool(np.array_equal(self.lo, self.hi))


def _profiles(fit_frame: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Distinct joint pre-index states of ``feats`` among the fitting rows (NaN kept as a state)."""
    num = fit_frame[feats].apply(lambda s: pd.to_numeric(s, errors="coerce") if s.dtype != object else s)
    key = num.copy()
    for c in feats:
        if pd.api.types.is_numeric_dtype(key[c]):
            key[c] = key[c].astype(float).fillna(SENTINEL)
    first = ~key.duplicated()
    prof = fit_frame.loc[first.to_numpy(), feats].reset_index(drop=True)
    if len(prof) > MAX_PROFILES:
        raise Phase2Stop("BOUNDS_PROFILE_LIMIT", f"{len(prof)} distinct pre-index profiles for {feats} (> {MAX_PROFILES}); the bound engine refuses "
                                                 "to approximate")
    return prof


def _allowed(prof: pd.DataFrame, rows_ub: pd.DataFrame) -> np.ndarray:
    """allowed[r, k]: profile k is a possible pre-index state of row r given the cumulative upper bounds (observed values) of the row."""
    allowed = np.ones((len(rows_ub), len(prof)), dtype=bool)
    for f in rows_ub.columns:
        pv = pd.to_numeric(prof[f], errors="coerce").to_numpy(dtype=float)
        u = pd.to_numeric(rows_ub[f], errors="coerce").to_numpy(dtype=float)
        ok_u = np.isfinite(u)[:, None]
        with np.errstate(invalid="ignore"):
            le = (pv[None, :] <= u[:, None]) & np.isfinite(pv)[None, :]
        allowed &= np.where(ok_u, le, True)
    return allowed


@dataclass
class UnknownPlan:
    """Which rows are UNKNOWN for which source, for the features one model uses."""

    sources: dict[str, list[str]]                  # source -> used features of that source with an UNKNOWN cell somewhere
    rows: dict[str, np.ndarray]                    # source -> bool mask over the evaluated rows
    ub: dict[str, pd.DataFrame]                    # source -> upper-bound columns (observed values) for the monotone features of that source

    @property
    def any_rows(self) -> np.ndarray | None:
        masks = list(self.rows.values())
        if not masks:
            return None
        out = np.zeros(len(masks[0]), dtype=bool)
        for m in masks:
            out |= m
        return out


def plan_unknown(features: list[str], unknown: pd.DataFrame, upper_bound: pd.DataFrame, feature_source: dict[str, str]) -> UnknownPlan:
    """``unknown`` / ``upper_bound``: aligned with the evaluated rows."""
    by_src: dict[str, list[str]] = {}
    for f in features:
        if f in unknown.columns and bool(unknown[f].any()):
            by_src.setdefault(feature_source[f], [])
    # every used feature of a source with an UNKNOWN row varies jointly with that source (a row UNKNOWN for one feature of a source is UNKNOWN for
    # the source); monotone features are then restricted by their observed upper bound (an observed 0 stays 0)
    rows, ubs, srcs = {}, {}, {}
    for s in by_src:
        fs = [f for f in features if feature_source.get(f) == s]
        m = np.zeros(len(unknown), dtype=bool)
        for f in fs:
            if f in unknown.columns:
                m |= unknown[f].to_numpy(dtype=bool)
        srcs[s] = fs
        rows[s] = m
        ubs[s] = upper_bound[[f for f in fs if f in upper_bound.columns]] if len(upper_bound.columns) else pd.DataFrame(index=unknown.index)
    return UnknownPlan(sources=srcs, rows=rows, ub=ubs)


# ============================================================================ linear models
def _col_feature(design: Any, col: str) -> str:
    f = design.feature_of_.get(col, col)
    if f in design.na_groups:
        return design.na_groups[f][0]
    return f


def linear_intervals(fit: Any, frame: pd.DataFrame, fit_frame: pd.DataFrame, plan: UnknownPlan) -> Intervals:
    """Exact prediction intervals of a fitted LASSO / elastic net (``falls_ml.phase2.fitting.LinearFit``) on ``frame``."""
    p = fit.predict(_fill_unknown(frame, fit_frame, plan))
    if not plan.sources:
        return Intervals(lo=p.copy(), hi=p.copy(), info={"n_unknown_rows": 0})
    design = fit.design
    coef = np.asarray(fit.model.coef_, dtype=float)
    b0 = float(fit.model.intercept_)
    filled = _fill_unknown(frame, fit_frame, plan)
    X = design.transform(filled).to_numpy(dtype=float)
    cols = list(design.columns_)
    col_src = {c: None for c in cols}
    eta = b0 + X @ coef
    lo_eta, hi_eta = eta.copy(), eta.copy()
    n_fallback = 0
    for s, fs in plan.sources.items():
        idx_cols = [j for j, c in enumerate(cols) if _col_feature(design, c) in set(fs)]
        for j in idx_cols:
            col_src[cols[j]] = s
        rows = np.flatnonzero(plan.rows[s])
        if not idx_cols or not len(rows):
            continue
        prof = _profiles(fit_frame, fs)
        P = pd.concat([fit_frame.iloc[[0]]] * len(prof), ignore_index=True)
        for f in fs:
            P[f] = prof[f].to_numpy()
        XP = design.transform(P).to_numpy(dtype=float)
        cs = XP[:, idx_cols] @ coef[idx_cols]
        cur = X[rows][:, idx_cols] @ coef[idx_cols]
        ubf = plan.ub[s]
        allowed = _allowed(prof, ubf.iloc[rows]) if len(ubf.columns) else np.ones((len(rows), len(prof)), dtype=bool)
        empty = ~allowed.any(axis=1)
        if empty.any():                         # no observed state satisfies the bound: fall back to every observed state (wider, still sound
            allowed[empty] = True               # within the observed support) and count it
            n_fallback += int(empty.sum())
        big = np.where(allowed, cs[None, :], np.inf)
        small = np.where(allowed, cs[None, :], -np.inf)
        lo_eta[rows] += big.min(axis=1) - cur
        hi_eta[rows] += small.max(axis=1) - cur
    lo, hi = expit(lo_eta), expit(hi_eta)
    known = ~plan.any_rows
    lo[known], hi[known] = p[known], p[known]
    return Intervals(lo=lo, hi=hi, info={"n_unknown_rows": int((~known).sum()), "n_bound_fallback": n_fallback, "sources": sorted(plan.sources)})


def _fill_unknown(frame: pd.DataFrame, fit_frame: pd.DataFrame, plan: UnknownPlan) -> pd.DataFrame:
    """UNKNOWN cells get a placeholder state (the first fitting row's) so the design can be computed; their contribution is replaced by the
    bounds. The placeholder never reaches a reported prediction: every UNKNOWN row's prediction is an interval."""
    out = frame.copy()
    for s, fs in plan.sources.items():
        rows = plan.rows[s]
        if rows.any():
            for f in fs:
                col = out[f]
                val = fit_frame[f].iloc[0]
                if col.dtype == object or pd.api.types.is_string_dtype(col):
                    out.loc[rows, f] = val
                else:
                    arr = col.to_numpy(dtype=float, na_value=np.nan).copy()
                    arr[rows] = float(val) if pd.notna(val) else np.nan
                    out[f] = arr.astype(col.dtype) if not pd.api.types.is_float_dtype(col) and np.isfinite(arr).all() else arr
    return out


# ============================================================================ XGBoost
@dataclass
class TreeArrays:
    feature: np.ndarray        # int, -1 for leaves
    threshold: np.ndarray      # float32
    yes: np.ndarray
    no: np.ndarray
    missing: np.ndarray
    leaf: np.ndarray           # float64 (leaf value; NaN for internal nodes)
    order: np.ndarray          # node ids in parent-before-child order


def parse_booster(booster: Any, columns: list[str]) -> list[TreeArrays]:
    pos = {c: i for i, c in enumerate(columns)}
    trees = []
    for js in booster.get_dump(dump_format="json"):
        root = json.loads(js)
        nodes: dict[int, dict[str, Any]] = {}
        stack, order = [root], []
        while stack:
            nd = stack.pop(0)
            nodes[int(nd["nodeid"])] = nd
            order.append(int(nd["nodeid"]))
            stack.extend(nd.get("children", []))
        m = max(nodes) + 1
        feat = np.full(m, -1, dtype=np.int64)
        thr = np.zeros(m, dtype=np.float32)
        yes, no, mis = (np.full(m, -1, dtype=np.int64) for _ in range(3))
        leaf = np.full(m, np.nan)
        for i, nd in nodes.items():
            if "leaf" in nd:
                leaf[i] = float(nd["leaf"])
            else:
                feat[i] = pos[str(nd["split"])]
                thr[i] = np.float32(nd["split_condition"])
                yes[i], no[i], mis[i] = int(nd["yes"]), int(nd["no"]), int(nd["missing"])
        trees.append(TreeArrays(feature=feat, threshold=thr, yes=yes, no=no, missing=mis, leaf=leaf, order=np.array(order)))
    return trees


def tree_margin_bounds(trees: list[TreeArrays], X: np.ndarray, unk: np.ndarray, amin: np.ndarray, amax: np.ndarray, anan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum over trees of the min / max leaf value reachable per row. Known cells follow the booster's rule (x < threshold -> yes; NaN -> missing);
    an UNKNOWN cell may follow 'yes' if an allowed value is < threshold, 'no' if one is >= threshold, 'missing' if NaN is allowed."""
    n = X.shape[0]
    Xf = X.astype(np.float32)
    lo = np.zeros(n)
    hi = np.zeros(n)
    for t in trees:
        reach = {int(t.order[0]): np.ones(n, dtype=bool)}
        tlo = np.full(n, np.inf)
        thi = np.full(n, -np.inf)
        for node in t.order:
            r = reach.pop(int(node), None)
            if r is None or not r.any():
                continue
            if t.feature[node] < 0:
                v = t.leaf[node]
                tlo = np.where(r, np.minimum(tlo, v), tlo)
                thi = np.where(r, np.maximum(thi, v), thi)
                continue
            j, th = int(t.feature[node]), t.threshold[node]
            x = Xf[:, j]
            u = unk[:, j]
            isn = np.isnan(x)
            with np.errstate(invalid="ignore"):
                go_yes = ~u & ~isn & (x < th)
                go_no = ~u & ~isn & (x >= th)
                can_yes = u & (amin[:, j] < th)
                can_no = u & (amax[:, j] >= th)
            miss_known = ~u & isn
            miss_unk = u & anan[:, j]
            to_yes = r & (go_yes | can_yes | ((miss_known | miss_unk) & (t.missing[node] == t.yes[node])))
            to_no = r & (go_no | can_no | ((miss_known | miss_unk) & (t.missing[node] == t.no[node])))
            for child, mask in ((int(t.yes[node]), to_yes), (int(t.no[node]), to_no)):
                reach[child] = reach[child] | mask if child in reach else mask
        lo += tlo
        hi += thi
    return lo, hi


def xgb_intervals(booster: Any, X: pd.DataFrame, fit_frame_tree: pd.DataFrame, plan: UnknownPlan, tree_col_of: dict[str, str]) -> Intervals:
    """Prediction intervals of a fitted booster on the tree matrix ``X``. ``fit_frame_tree``: the fitting rows' tree matrix (the observed states);
    ``tree_col_of``: feature name -> tree-matrix column."""
    import xgboost as xgb

    cols = list(X.columns)
    dm = xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=cols, missing=np.nan)
    p = booster.predict(dm).astype(np.float64)
    if not plan.sources:
        return Intervals(lo=p.copy(), hi=p.copy(), info={"n_unknown_rows": 0})
    trees = parse_booster(booster, cols)
    any_rows = plan.any_rows
    rows = np.flatnonzero(any_rows)
    n, m = len(rows), len(cols)
    unk = np.zeros((n, m), dtype=bool)
    amin = np.full((n, m), np.nan)
    amax = np.full((n, m), np.nan)
    anan = np.zeros((n, m), dtype=bool)
    pos = {c: i for i, c in enumerate(cols)}
    for s, fs in plan.sources.items():
        tf = [f for f in fs if tree_col_of.get(f) in pos]
        if not tf:
            continue
        in_s = plan.rows[s][rows]
        prof = _profiles(fit_frame_tree.rename(columns={tree_col_of[f]: f for f in tf}), tf)
        ubf = plan.ub[s]
        allowed = _allowed(prof, ubf.iloc[rows]) if len(ubf.columns) else np.ones((n, len(prof)), dtype=bool)
        allowed[~allowed.any(axis=1)] = True
        for f in tf:
            j = pos[tree_col_of[f]]
            v = pd.to_numeric(prof[f], errors="coerce").to_numpy(dtype=float)
            fin = np.isfinite(v)
            a_f = allowed & fin[None, :]
            with np.errstate(invalid="ignore"):
                mn = np.where(a_f, v[None, :], np.inf).min(axis=1)
                mx = np.where(a_f, v[None, :], -np.inf).max(axis=1)
            mn[~np.isfinite(mn)] = np.nan
            mx[~np.isfinite(mx)] = np.nan
            has_nan = (allowed & ~fin[None, :]).any(axis=1)
            unk[in_s, j] = True
            amin[in_s, j], amax[in_s, j], anan[in_s, j] = mn[in_s], mx[in_s], has_nan[in_s]
    Xa = X.to_numpy(dtype=float)
    # base margin: the booster's margin minus the sum of leaves reached, identical for every row (checked on up to 50 known rows)
    known_rows = np.flatnonzero(~any_rows)[:50]
    if len(known_rows):
        mk = booster.predict(xgb.DMatrix(Xa[known_rows].astype(np.float32), feature_names=cols, missing=np.nan), output_margin=True).astype(float)
        z = np.zeros((len(known_rows), m), dtype=bool)
        lk, hk = tree_margin_bounds(trees, Xa[known_rows], z, np.full((len(known_rows), m), np.nan), np.full((len(known_rows), m), np.nan), z)
        if not np.allclose(lk, hk) or np.ptp(mk - lk) > 1e-4:
            raise Phase2Stop("BOUNDS_SELFTEST", "the XGBoost tree interpreter does not reproduce the booster's margins on known rows")
        base = float(np.median(mk - lk))
    else:
        base = float(booster.predict(xgb.DMatrix(Xa[:1].astype(np.float32), feature_names=cols, missing=np.nan), output_margin=True)[0])
        z = np.zeros((1, m), dtype=bool)
        l1, _ = tree_margin_bounds(trees, Xa[:1], z, np.full((1, m), np.nan), np.full((1, m), np.nan), z)
        base -= float(l1[0])
    lo_m, hi_m = tree_margin_bounds(trees, Xa[rows], unk, amin, amax, anan)
    lo, hi = p.copy(), p.copy()
    lo[rows], hi[rows] = expit(base + lo_m), expit(base + hi_m)
    return Intervals(lo=lo, hi=hi, info={"n_unknown_rows": n, "sources": sorted(plan.sources), "bound": "outer (interval propagation)"})


# ============================================================================ bounded metrics
def assign(y: np.ndarray, iv: Intervals, mode: str) -> np.ndarray:
    """ADVERSE: events get the lowest possible risk and non-events the highest; FAVOURABLE: the reverse."""
    y = np.asarray(y).astype(int)
    if mode == "adverse":
        return np.where(y == 1, iv.lo, iv.hi)
    if mode == "favourable":
        return np.where(y == 1, iv.hi, iv.lo)
    raise ValueError(mode)


def bounded_bundle(y: np.ndarray, iv: Intervals, *, capacities: tuple[float, ...], principal: float) -> dict[str, Any]:
    from falls_ml.phase2.evaluate import bundle

    out: dict[str, Any] = {"n_unknown_rows": int(np.sum(iv.lo != iv.hi))}
    for mode in ("adverse", "favourable"):
        b = bundle(y, assign(y, iv, mode), capacities=capacities, principal=principal)
        out.update({f"{mode}_{k}": v for k, v in b.items()})
    return out


def bounded_delta(y: np.ndarray, ref: Intervals, new: Intervals, *, n_boot: int, seed: int, principal: float,
                  capacities: tuple[float, ...] = ()) -> dict[str, Any]:
    """Paired patient bootstrap of new - reference: ADVERSE = new adverse vs reference favourable (the conservative difference: worst case for the
    new model, best case for the reference); FAVOURABLE = the reverse. Identical rows and resamples for both ends."""
    from falls_ml.phase2.evaluate import paired_bootstrap

    out: dict[str, Any] = {}
    for mode, (r, n_) in {"adverse": ("favourable", "adverse"), "favourable": ("adverse", "favourable")}.items():
        d = paired_bootstrap(y, assign(y, ref, r), assign(y, new, n_), n_boot=n_boot, seed=seed, principal=principal, capacities=capacities)
        out.update({f"{mode}_{k}": v for k, v in d.items() if k.startswith("delta")})
    return out
