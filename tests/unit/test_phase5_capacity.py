"""Phase 5 operating-capacity analysis (fast, synthetic arrays only): exact capacity arithmetic, largest-remainder allocation across the outer
folds, top-k WITHIN each fold (never a pooled probability threshold), the paired bootstrap's per-replicate re-selection, small-cell suppression
and the aggregate-only dashboard payload."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest


def _toy(n: int = 3000, k: int = 5, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    outer = np.arange(n) % k
    p = 1 / (1 + np.exp(-(rng.normal(size=n) + 1.2 * y - 2.5)))
    return y, outer, p


def test_capacity_arithmetic_is_exact() -> None:
    from falls_ml.phase5.capacity import allocate, fine_grid, grid_step, n_selected

    assert n_selected(30, 97467) == 2924 and n_selected(30, 100000) == 3000 and n_selected(5, 1000) == 5 and n_selected(5, 100) == 1 and n_selected(15, 100) == 2   # half up
    for total, sizes in ((2924, [19494, 19494, 19493, 19493, 19493]), (7, [3, 3, 3]), (0, [5, 5]), (10, [1, 1, 98])):
        k = allocate(total, sizes)
        assert int(k.sum()) == min(total, sum(sizes)) and (k <= np.asarray(sizes)).all()
        assert all(abs(int(kf) - total * s / sum(sizes)) < 1 for kf, s in zip(k, sizes))           # never more than one from the exact quota
    assert list(allocate(7, [3, 3, 3])) == [3, 2, 2]                                               # equal remainders -> lower fold first
    assert grid_step(100000) == 1 and grid_step(5000) == 2 and grid_step(2000) == 5
    g = fine_grid(100000)
    assert g[0] == 5 and g[-1] == 200 and len(g) == 196 and 30 in g
    assert 30 in fine_grid(1500) and 30 in fine_grid(2500)


def test_selection_is_within_each_outer_fold_not_a_pooled_threshold() -> None:
    from falls_ml.phase5.capacity import allocate, fold_capacity, n_selected, pooled_capacity, rank_model

    y, outer, p = _toy()
    p = np.where(outer == 0, np.minimum(p * 10, 0.99), p)          # fold 0's model is on a different probability scale
    r = rank_model("ENET", "OLD", y, outer, p)
    T = n_selected(30, len(y))
    c = fold_capacity(y, r, T)
    k = allocate(T, [int((outer == f).sum()) for f in range(5)])
    assert c["per_fold_selected"] == [int(v) for v in k] and c["flagged"] == T
    tp = 0
    for f in range(5):                                             # brute force: the k_f highest risks of each fold
        m = np.flatnonzero(outer == f)
        top = m[np.argsort(-p[m], kind="mergesort")[: k[f]]]
        tp += int(y[top].sum())
    assert c["tp"] == tp and c["fp"] == T - tp and c["fn"] == int(y.sum()) - tp
    pc = pooled_capacity(y, r, T)
    top = np.argsort(-p, kind="mergesort")[:T]
    assert pc["tp"] == int(y[top].sum()) and (outer[top] == 0).mean() > 0.9                  # the pooled view is dominated by fold 0
    assert c["per_fold_selected"][0] < 0.25 * T                                                 # the primary one is not


def test_bootstrap_reselects_within_folds_and_identical_models_give_zero() -> None:
    from falls_ml.phase5.capacity import allocate, bootstrap_tables, capacity_bootstrap, n_selected, rank_model, weighted_tp

    y, outer, p = _toy(1200, 3, 5)
    r = rank_model("ENET", "OLD", y, outer, p)
    rng = np.random.default_rng(1)
    for _ in range(20):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))
        k = allocate(n_selected(30, len(y)), [int(w[outer == f].sum()) for f in range(3)])
        brute = 0
        for f in range(3):                                         # expand every patient into its resampled copies, rank, take the top k_f copies
            m = np.flatnonzero(outer == f)
            o = m[np.argsort(-p[m], kind="mergesort")]
            copies = np.repeat(o, w[o])
            brute += int(y[copies[: k[f]]].sum())
        assert weighted_tp(r, w, k) == brute == weighted_tp(r, w, k, [min(len(f["idx"]), 4 * int(k[i]) + 200) for i, f in enumerate(r.folds)])
    ranked = {("ENET", "OLD"): r, ("ENET", "OLD_PLUS_ALL_NEW_ELIGIBLE"): rank_model("ENET", "OLD_PLUS_ALL_NEW_ELIGIBLE", y, outer, p.copy())}
    boot = capacity_bootstrap(y, outer, ranked, 30, n_boot=60, seed=4)
    cmp, dist = bootstrap_tables(y, ranked, boot)
    row = cmp.iloc[0]
    assert row["delta_falls_captured"] == 0 and row["delta_falls_captured_ci_low"] == 0 and row["delta_falls_captured_ci_high"] == 0
    assert set(dist["metric"]) == {"delta_falls_captured", "delta_sensitivity", "delta_ppv", "delta_false_interventions"}
    # a better model captures more falls at the same capacity, and FP falls by exactly the same number
    better = rank_model("ENET", "OLD_PLUS_ALL_NEW_ELIGIBLE", y, outer, p + 0.5 * y)
    ranked[("ENET", "OLD_PLUS_ALL_NEW_ELIGIBLE")] = better
    cmp2, _ = bootstrap_tables(y, ranked, capacity_bootstrap(y, outer, ranked, 30, n_boot=200, seed=4))
    r2 = cmp2.iloc[0]
    assert r2["delta_falls_captured"] > 0 and r2["delta_falls_captured_ci_low"] > 0 and r2["delta_false_interventions"] == -r2["delta_falls_captured"]


def test_small_cells_never_appear_alone() -> None:
    from falls_ml.phase5.dashboard import _suppress

    t = pd.DataFrame([{"family": "ENET", "feature_set": "OLD", "capacity_pct": 0.5, "selected_total": 15, "tp": 4, "fp": 11, "fn": 300, "tn": 2000,
                       "sensitivity": 4 / 304, "ppv": 4 / 15, "captured_per_1000_interventions": 266.7},
                      {"family": "ENET", "feature_set": "OLD", "capacity_pct": 3.0, "selected_total": 90, "tp": 30, "fp": 60, "fn": 274, "tn": 1951,
                       "sensitivity": 30 / 304, "ppv": 30 / 90, "captured_per_1000_interventions": 333.3}])
    s = _suppress(t)
    assert s.at[0, "tp"] == "<10" and s.at[0, "fp"] == "suppressed" and s.at[0, "ppv"] == "suppressed" and s.at[0, "captured_per_1000_interventions"] == "suppressed"
    assert s.at[0, "selected_total"] == 15 and s.at[1, "tp"] == 30


def test_dashboard_payload_is_aggregate_and_self_contained() -> None:
    from falls_ml.phase5.capacity import curve, extended_grid, fine_grid, rank_model, safe_points
    from falls_ml.phase5.dashboard import dashboard_data
    from falls_ml.phase5.dashboard_html import render

    y, outer, p = _toy(12000, 5, 9)
    ranked = {("ENET", s): rank_model("ENET", s, y, outer, p + d * y) for s, d in (("OLD", 0.0), ("OLD_PLUS_ALL_NEW_ELIGIBLE", 0.1), ("OLD_PLUS_NEW_SAFE", 0.05))}
    fine, ext = curve(y, ranked, fine_grid(len(y))), curve(y, ranked, extended_grid())
    pooled = curve(y, ranked, fine_grid(len(y)), pooled=True)
    safe, unsafe = safe_points(fine)
    A = {"N": len(y), "E": int(y.sum()), "safe": safe, "unsafe": unsafe, "grid_step": 1, "ext_safe": safe_points(ext)[0], "pooled_safe": safe_points(pooled)[0],
         "fine": fine, "ext": ext, "pooled": pooled, "models": sorted(ranked), "three_pct_safe": 30 in safe, "cmp3": pd.DataFrame(), "boot_n": 10, "folds": 5,
         "drivers": pd.DataFrame()}
    d = dashboard_data(A, {}, synthetic=True)
    assert set(d) >= {"n", "events", "grid", "tp", "tp_pooled", "tp_ext", "drivers", "top3"} and d["default_permille"] == 30
    assert len(d["tp"]["ENET"]["OLD"]) == len(d["grid"]) and all(isinstance(v, int) for v in d["tp"]["ENET"]["OLD"])
    html = render(d)
    assert html.startswith("<!doctype html>") and "__DATA__" not in html
    assert not re.search(r"<script[^>]+src=|<link[^>]+href=|@import|url\(", html)              # no external script / stylesheet / font
    assert re.findall(r"https?://[^\s\"']+", html) == ["http://www.w3.org/2000/svg"]          # the only URL is the SVG namespace (never fetched)
    blob = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1).replace("<\\/", "</"))
    assert blob["n"] == len(y) and "row_key" not in html and "Customer_Full_ID" not in html
    assert max(len(v) for v in blob["tp"]["ENET"].values()) <= 196                               # one count per capacity point, never per patient
