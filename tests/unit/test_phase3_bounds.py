"""Phase 3 partial-identification engine (falls_ml.phase3.bounds): known rows are predicted exactly; for an UNKNOWN row the interval contains the
prediction of EVERY allowed pre-index state (in particular its true state, when that state was observed); linear bounds are exact (attained);
the XGBoost interval propagation reproduces the booster on known rows and is a sound outer bound; cumulative upper bounds are respected; the
adverse / favourable assignments bracket every possible truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from falls_ml.phase3 import bounds as B


def _world(n: int = 1500, seed: int = 3) -> dict:
    from falls_ml.data.meuhedet_wide import load_wide_mapping
    from falls_ml.features.spec import load_feature_spec

    rng = np.random.default_rng(seed)
    age = rng.integers(65, 95, n).astype(float)
    sex = np.where(rng.random(n) < 0.55, "female", "male")
    falls = (rng.random(n) < 0.2).astype(np.int8)
    assessed = rng.random(n) < 0.3
    item = np.where(assessed, (rng.random(n) < 0.4).astype(float), np.nan)
    score = np.where(assessed, rng.integers(0, 6, n).astype(float), np.nan)
    cnt = rng.poisson(2.0, n).astype(float)
    lp = -4 + 0.04 * (age - 75) + 0.9 * falls + 0.8 * np.nan_to_num(item) - 0.2 * np.nan_to_num(score, nan=5) + 0.15 * cnt
    y = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    fr = pd.DataFrame({"age_years": age, "sex": sex, "falls": falls, "form": assessed.astype(float), "item": item, "score": score, "cnt": cnt})
    design = {"features": {
        "form": {"kind": "binary", "linear": "none", "missing": "no_event", "form": None, "na_group": None, "levels_used": None, "fill": "zero"},
        "item": {"kind": "binary", "linear": "none", "missing": "not_assessed", "form": "F", "na_group": None, "levels_used": None, "fill": "zero"},
        "score": {"kind": "ordinal", "linear": "thermometer", "missing": "not_assessed", "form": "F", "na_group": None, "levels_used": [1.0, 2.0, 3.0, 4.0, 5.0],
                  "fill": "zero"},
        "cnt": {"kind": "count", "linear": "log1p", "missing": "unexpected", "form": None, "na_group": None, "levels_used": None, "fill": "median"}},
        "na_groups": {}}
    mapping = load_wide_mapping()
    espec = load_feature_spec(mapping.exploratory_spec_path)
    cfg = {"lasso": {"n_lambda": 20, "lambda_min_ratio": 1e-3, "selection": "min", "tol": 1e-7, "max_iter": 10000}, "cv": {"inner_folds_linear": 3}}
    return {"frame": fr, "y": y, "design": design, "espec": espec, "cfg": cfg, "rng": rng}


def _fit(w: dict, rows: np.ndarray):
    from falls_ml.phase2.fitting import LASSO, fit_linear

    return fit_linear(LASSO, w["frame"].loc[rows].reset_index(drop=True), w["y"][rows], new=["form", "item", "score", "cnt"],
                      baseline=["age_years", "sex", "falls"], design=w["design"], efalls_spec=w["espec"], cfg=w["cfg"], seed=1)


SRC = {"form": "NURSE", "item": "NURSE", "score": "NURSE", "cnt": "VISITS", "falls": "FALLS", "age_years": "STATIC", "sex": "STATIC"}


def test_linear_intervals_exact_on_known_rows_and_contain_every_allowed_state() -> None:
    w = _world()
    n = len(w["y"])
    fit_rows = np.arange(n) < 1000
    fit = _fit(w, fit_rows)
    ev = w["frame"].loc[~fit_rows].reset_index(drop=True)
    unk = pd.DataFrame(False, index=ev.index, columns=["form", "item", "score", "cnt", "falls"])
    hidden = np.arange(len(ev)) % 5 == 0
    for f in ("form", "item", "score"):
        unk.loc[hidden, f] = True
    unk.loc[np.arange(len(ev)) % 7 == 0, "falls"] = True
    ub = pd.DataFrame({"falls": ev["falls"].to_numpy(dtype=float)})
    plan = B.plan_unknown(["form", "item", "score", "cnt", "falls", "age_years", "sex"], unk, ub, SRC)
    iv = B.linear_intervals(fit, ev, w["frame"].loc[fit_rows].reset_index(drop=True), plan)
    p = fit.predict(ev)
    known = ~unk.any(axis=1).to_numpy()
    assert np.array_equal(iv.lo[known], p[known]) and np.array_equal(iv.hi[known], p[known])
    assert np.all(iv.lo <= iv.hi + 1e-15)
    # every observed state of the fitting rows is a possible pre-index state: its prediction lies inside the interval
    fitfr = w["frame"].loc[fit_rows].reset_index(drop=True)
    r = int(np.flatnonzero(hidden & ~known)[0]) if (hidden & ~known).any() else int(np.flatnonzero(hidden)[0])
    for k in range(0, 200, 7):
        row = ev.iloc[[r]].copy()
        for f in ("form", "item", "score"):
            row[f] = fitfr[f].iloc[k]
        q = fit.predict(row)[0]
        assert iv.lo[r] - 1e-12 <= q <= iv.hi[r] + 1e-12
    # the bounds are attained (exact): some allowed state reaches lo and some reaches hi
    preds = []
    for k in range(len(fitfr)):
        row = ev.iloc[[r]].copy()
        for f in ("form", "item", "score"):
            row[f] = fitfr[f].iloc[k]
        preds.append(fit.predict(row)[0])
    assert np.isclose(min(preds), iv.lo[r]) and np.isclose(max(preds), iv.hi[r])


def test_cumulative_upper_bound_makes_an_observed_zero_exact() -> None:
    w = _world()
    n = len(w["y"])
    fit_rows = np.arange(n) < 1000
    fit = _fit(w, fit_rows)
    ev = w["frame"].loc[~fit_rows].reset_index(drop=True)
    unk = pd.DataFrame({"falls": np.ones(len(ev), dtype=bool)})
    ub = pd.DataFrame({"falls": ev["falls"].to_numpy(dtype=float)})
    plan = B.plan_unknown(["falls", "age_years", "sex", "form", "item", "score", "cnt"], unk, ub, SRC)
    iv = B.linear_intervals(fit, ev, w["frame"].loc[fit_rows].reset_index(drop=True), plan)
    p = fit.predict(ev)
    zero = ev["falls"].to_numpy() == 0
    assert np.allclose(iv.lo[zero], p[zero]) and np.allclose(iv.hi[zero], p[zero])
    one = ~zero
    assert np.all(iv.hi[one] >= p[one] - 1e-12) and np.all(iv.lo[one] <= p[one] + 1e-12)


def test_xgb_interval_propagation_reproduces_known_rows_and_bounds_every_state() -> None:
    pytest.importorskip("xgboost")
    from falls_ml.phase2.fitting import xgb_fit

    w = _world(2000, seed=5)
    fr = w["frame"].assign(sex_female=(w["frame"]["sex"] == "female").astype(float)).drop(columns=["sex"])
    cols = ["age_years", "sex_female", "falls", "form", "item", "score", "cnt"]
    X = fr[cols].astype(float)
    fit_rows = np.arange(len(X)) < 1400
    params = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "min_child_weight": 2.0, "seed": 1, "nthread": 1, "verbosity": 0}
    bst = xgb_fit(X.loc[fit_rows], w["y"][fit_rows], params, 60)
    ev = X.loc[~fit_rows].reset_index(drop=True)
    unk = pd.DataFrame(False, index=ev.index, columns=["form", "item", "score", "cnt"])
    hidden = np.arange(len(ev)) % 4 == 0
    for f in ("form", "item", "score"):
        unk.loc[hidden, f] = True
    plan = B.plan_unknown(["form", "item", "score", "cnt"], unk, pd.DataFrame(index=ev.index), SRC)
    iv = B.xgb_intervals(bst, ev, X.loc[fit_rows].reset_index(drop=True), plan, {c: c for c in cols})
    import xgboost as xgb

    p = bst.predict(xgb.DMatrix(ev.to_numpy(dtype=np.float32), feature_names=cols, missing=np.nan))
    assert np.allclose(iv.lo[~hidden], p[~hidden]) and np.allclose(iv.hi[~hidden], p[~hidden])
    fitX = X.loc[fit_rows].reset_index(drop=True)
    for r in np.flatnonzero(hidden)[:20]:
        rows = ev.iloc[[r] * 60].copy().reset_index(drop=True)
        for f in ("form", "item", "score"):
            rows[f] = fitX[f].iloc[:60].to_numpy()
        q = bst.predict(xgb.DMatrix(rows.to_numpy(dtype=np.float32), feature_names=cols, missing=np.nan))
        assert np.all(q >= iv.lo[r] - 1e-6) and np.all(q <= iv.hi[r] + 1e-6)


def test_adverse_and_favourable_bracket_every_possible_truth() -> None:
    from falls_ml.phase2.evaluate import bundle

    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.05).astype(int)
    base = rng.random(n) * 0.2 + 0.1 * y
    lo, hi = base.copy(), base.copy()
    u = rng.random(n) < 0.03
    hi[u] = np.minimum(1, base[u] + rng.random(u.sum()) * 0.3)
    iv = B.Intervals(lo=lo, hi=hi)
    adv = bundle(y, B.assign(y, iv, "adverse"), capacities=(0.1,), principal=0.1)
    fav = bundle(y, B.assign(y, iv, "favourable"), capacities=(0.1,), principal=0.1)
    for _ in range(20):
        truth = np.where(u, lo + rng.random(n) * (hi - lo), base)
        b = bundle(y, truth, capacities=(0.1,), principal=0.1)
        for k in ("ap", "auroc", "capture@0.1"):
            assert adv[k] - 1e-12 <= b[k] <= fav[k] + 1e-12, k
        assert fav["logloss"] - 1e-12 <= b["logloss"] <= adv["logloss"] + 1e-12
