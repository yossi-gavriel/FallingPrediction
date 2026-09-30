"""Stages S06-S13: nested grouped CV of LASSO / elastic net / XGBoost inside TRAIN, the pre-declared advancement rule (frozen before
VALIDATION is opened), stability, ablation, explainability and the one-time VALIDATION scoring (planning/EXPERIMENT_PLAN.md §6-§9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.errors import DegenerateFitError
from falls_ml.models.lasso_cv import LassoConvergenceError, LassoError
from falls_ml.phase2 import durable as D
from falls_ml.phase2.context import Ctx, import_xgboost
from falls_ml.phase2.design import tree_feature_of, tree_matrix
from falls_ml.phase2.evaluate import (bundle, calibration_bootstrap, calibration_table, capacity_table, paired_bootstrap, smooth_calibration,
                                      threshold_table, top_mask)
from falls_ml.phase2.featuresets import EXPLORATORY, SAFE_DISCOVERY
from falls_ml.phase2.fitting import ENET, LASSO, fit_linear, xgb_fit, xgb_gain_importance, xgb_inner_cv, xgb_params, xgb_predict
from falls_ml.phase2.state import Phase2Stop, Stage
from falls_ml.phase2.xgb_tuning import TrialStore, best_trial, run_study

BENCHMARK = "LASSO:BASELINE_15"            # HISTORICAL_BASELINE_15: the benchmark every model is compared with (never a SAFE candidate)
SAFE_REF = "LASSO:SAFE_BASE"               # SAFE_DISCOVERY reference of the advancement rule
REFERENCE = BENCHMARK                      # the plausibility gates compare with the best known model
KIND_ORDER = {"historical": 0, "main": 1, "domain_add": 2, "waterfall": 3, "sensitivity": 4, "exploratory": 5, "loo": 6}
CANDIDATE_KINDS = ("main", "domain_add", "waterfall")
FIT_ERRORS = (DegenerateFitError, LassoError)


def _sets_in_order(ctx: Ctx, kinds: tuple[str, ...]) -> list[str]:
    fs = ctx.fs()
    names = [s for s in fs.fitted() if fs.sets[s]["kind"] in kinds]
    return sorted(names, key=lambda s: (s != "BASELINE_15", s != "SAFE_BASE", KIND_ORDER[fs.sets[s]["kind"]], s))


def inner_fold_seed(ctx: Ctx, family: str, scope: str) -> int:
    """Seed of the inner-fold assignment of one tuning scope ('outer<k>' or 'final'): the same folds for every feature set, lambda,
    l1_ratio and Optuna trial of that scope (Astra F-07). Model randomness (XGBoost subsampling) keeps the item seed."""
    return ctx.run.item_seed(f"inner_folds_{family}", scope)


def lasso_config(ctx: Ctx, setname: str) -> str:
    """The LASSO configuration that holds the fit of ``setname`` (identical specifications share one fit)."""
    return f"{LASSO}:{ctx.fs().fitted_name(setname)}"


def is_safe_candidate(ctx: Ctx, setname: str) -> bool:
    s = ctx.fs().sets[setname]
    return s["category"] == SAFE_DISCOVERY and s["kind"] in CANDIDATE_KINDS


def config_set(config: str) -> str:
    return config.split(":", 1)[1]


def config_family(config: str) -> str:
    return config.split(":", 1)[0]


# ============================================================================ linear (S06 LASSO, S07 ENET, S11 LOO / individual ablation)
def _fit_with_retry(ctx: Ctx, family: str, frame: pd.DataFrame, y: np.ndarray, s: dict[str, Any], seed: int, *, fixed: dict[str, float] | None = None,
                    new: list[str] | None = None, groups: np.ndarray | None = None) -> tuple[Any, dict[str, Any]]:
    """LASSO fits follow convergence-retry policy v1 (one retry, full-path iteration limit x10, lambda* unchanged; never accepted unconverged)."""
    kw = dict(new=list(s["features"] if new is None else new), baseline=s["baseline"], design=ctx.fs().design, efalls_spec=ctx.espec, cfg=ctx.cfg, seed=seed,
              fixed=fixed, groups=groups)
    try:
        return fit_linear(family, frame, y, **kw), {"retry": None}
    except LassoConvergenceError as exc:
        if family != LASSO or exc.max_iter is None:
            raise
        fit = fit_linear(family, frame, y, **kw, model_param_overrides={"full_path_max_iter": int(exc.max_iter) * 10})
        d = fit.model.fit_diagnostics()
        if not d.get("lambda_star_converged") or d.get("lambda_star_index") != exc.lambda_star_index:
            raise LassoConvergenceError(f"convergence retry rejected: {exc}") from exc
        return fit, {"retry": "CONVERGED_ON_RETRY", "initial_max_iter": exc.max_iter}


def _feature_importance_linear(fit: Any) -> dict[str, float]:
    t = fit.coefficients()
    t = t[~t["feature"].str.startswith("na__")]
    return {str(k): float(v) for k, v in t.assign(a=t["standardized_coefficient"].abs()).groupby("feature")["a"].sum().items()}


def linear_outer_item(ctx: Ctx, family: str, setname: str, k: int, *, new: list[str] | None = None, fixed: dict[str, float] | None = None,
                      config: str | None = None) -> Any:
    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        s = ctx.fs().sets[setname]
        tr = ctx.train_frame()
        y = ctx.y_train()
        folds = ctx.outer_folds()
        te = folds == k
        try:   # inner folds common to every set / family of this outer fold (Astra F-07): comparisons are not blurred by fold noise
            fit, retry = _fit_with_retry(ctx, family, tr.loc[~te], y[~te], s, inner_fold_seed(ctx, "linear", f"outer{k}"), fixed=fixed, new=new)
        except FIT_ERRORS as exc:
            return {"config": config or f"{family}:{setname}", "fold": k, "failed": True, "error": f"{type(exc).__name__}: {exc}"[:500]}
        p = fit.predict(tr.loc[te])
        D.write_npz(tmp / "oof.npz", rows=np.flatnonzero(te), p=p)
        h = fit.hyper()
        diag = fit.diagnostics
        return {"config": config or f"{family}:{setname}", "fold": k, "failed": False, "lambda": h["lambda"], "l1_ratio": h["l1_ratio"],
                "lambda_star_index": diag.get("lambda_star_index"), "lambda_at_grid_boundary": bool(diag.get("lambda_star_index") == (diag.get("n_lambda") or 0) - 1),
                "cv_minimum_identified": diag.get("cv_minimum_identified"), "n_design_columns": len(fit.design.columns_),
                "duplicate_columns_dropped": fit.design.duplicates_, "n_selected": len(fit.selected_features()), "selected": fit.selected_features(),
                "retry": retry["retry"]}
    return fn


def linear_final_item(ctx: Ctx, family: str, setname: str) -> Any:
    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        s = ctx.fs().sets[setname]
        tr = ctx.train_frame()
        y = ctx.y_train()
        try:
            fit, retry = _fit_with_retry(ctx, family, tr, y, s, inner_fold_seed(ctx, "linear", "final"))
        except FIT_ERRORS as exc:
            return {"config": f"{family}:{setname}", "failed": True, "error": f"{type(exc).__name__}: {exc}"[:500]}
        D.write_pickle(tmp / "model.pkl", fit)
        D.write_csv(tmp / "coefficients.csv", fit.coefficients())
        h = fit.hyper()
        return {"config": f"{family}:{setname}", "failed": False, "lambda": h["lambda"], "l1_ratio": h["l1_ratio"], "n_design_columns": len(fit.design.columns_),
                "duplicate_columns_dropped": fit.design.duplicates_, "selected": fit.selected_features(), "n_selected": len(fit.selected_features()),
                "importance": _feature_importance_linear(fit), "retry": retry["retry"],
                "cv_minimum_identified": fit.diagnostics.get("cv_minimum_identified")}
    return fn


def _nested_linear(ctx: Ctx, st: Stage, family: str, sets: list[str], *, final: bool = True) -> None:
    K = int(ctx.cfg["cv"]["outer_folds"])
    for s in sets:
        for k in range(K):
            st.item(f"{family}__{s}__outer{k}", linear_outer_item(ctx, family, s, k))
        if final:
            st.item(f"{family}__{s}__final", linear_final_item(ctx, family, s))


def oof(ctx: Ctx, st: Stage, prefix: str) -> np.ndarray | None:
    """Assemble outer-fold predictions of one configuration (item prefix) into a TRAIN-length vector; None if any fold failed."""
    K = int(ctx.cfg["cv"]["outer_folds"])
    p = np.full(len(ctx.y_train()), np.nan)
    for k in range(K):
        r = st.result(f"{prefix}__outer{k}")
        if r.get("failed"):
            return None
        z = np.load(st.path(f"{prefix}__outer{k}") / "oof.npz")
        p[z["rows"]] = z["p"]
    if np.isnan(p).any():
        raise Phase2Stop("OOF_INCOMPLETE", f"{prefix}: out-of-fold predictions do not cover every TRAIN row")
    return p


def _failure_gate(ctx: Ctx, st: Stage) -> dict[str, Any]:
    recs = [r["result"] for r in st.records().values() if isinstance(r.get("result"), dict) and "failed" in r["result"]]
    n_fail = sum(1 for r in recs if r["failed"])
    frac = n_fail / len(recs) if recs else 0.0
    if frac > float(ctx.cfg["gates"]["max_fit_failure_fraction"]):
        raise Phase2Stop("FIT_FAILURES", f"{n_fail} of {len(recs)} fits failed in {st.label} (> {ctx.cfg['gates']['max_fit_failure_fraction']:.0%})",
                         [r.get("error", "") for r in recs if r["failed"]][:10])
    return {"n_fits": len(recs), "n_failed": n_fail}


def _fits_table(st: Stage) -> pd.DataFrame:
    rows = []
    for rec in st.records().values():
        r = rec["result"]
        if not isinstance(r, dict) or "config" not in r:
            continue
        rows.append({"item": rec["item"], "config": r.get("config"), "fold": r.get("fold", "final"), "failed": r.get("failed"), "error": r.get("error"),
                     "lambda": r.get("lambda"), "l1_ratio": r.get("l1_ratio"), "lambda_at_grid_boundary": r.get("lambda_at_grid_boundary"),
                     "cv_minimum_identified": r.get("cv_minimum_identified"), "n_design_columns": r.get("n_design_columns"), "n_selected": r.get("n_selected"),
                     "duplicate_columns_dropped": json.dumps(r.get("duplicate_columns_dropped") or {}), "retry": r.get("retry"),
                     "elapsed_s": rec.get("elapsed_s"), "attempt": rec.get("attempt")})
    return pd.DataFrame(rows)


def s06_linear(ctx: Ctx) -> None:
    from falls_ml.phase2.stages_data import verify_protected

    st = ctx.run.stage("06", "lasso", ("05",))
    if st.complete_record():
        return
    _nested_linear(ctx, st, LASSO, _sets_in_order(ctx, ("historical", "main", "domain_add", "waterfall", "sensitivity", "exploratory")))
    summ = _failure_gate(ctx, st)
    out = D.write_csv(ctx.run.artifacts / "FITS_LASSO.csv", _fits_table(st))
    st.finalize([out], summ)
    verify_protected(ctx)


def oof_ap(ctx: Ctx, st: Stage, prefix: str) -> float | None:
    from falls_ml.evaluation.metrics import pr_auc

    p = oof(ctx, st, prefix)
    return None if p is None else float(pr_auc(ctx.y_train(), p))


def enet_sets(ctx: Ctx) -> list[str]:
    """ALL_REVIEWED_SAFE + the best SAFE_DISCOVERY LASSO set with new features by outer-OOF AP, if different (<= max_sets)."""
    st6 = ctx.run.stage("06", "lasso")
    cands = [s for s in _sets_in_order(ctx, CANDIDATE_KINDS) if is_safe_candidate(ctx, s) and ctx.fs().sets[s]["n_new_features"] > 0]
    scores = {s: oof_ap(ctx, st6, f"LASSO__{s}") for s in cands}
    scores = {s: v for s, v in scores.items() if v is not None}
    out = ["ALL_REVIEWED_SAFE"]
    if scores:
        best = max(sorted(scores), key=lambda s: scores[s])
        if best not in out:
            out.append(best)
    return out[: int(ctx.cfg["enet"]["max_sets"])]


def s07_enet(ctx: Ctx) -> None:
    from falls_ml.phase2.stages_data import verify_protected

    st = ctx.run.stage("07", "enet", ("06",))
    if st.complete_record():
        return
    sets = st.item("ENET__plan", lambda tmp, seed: {"sets": enet_sets(ctx), "rule": "ALL_REVIEWED_SAFE + best SAFE LASSO set by outer-OOF AP"})["sets"]
    _nested_linear(ctx, st, ENET, sets)
    summ = _failure_gate(ctx, st)
    out = D.write_csv(ctx.run.artifacts / "FITS_ENET.csv", _fits_table(st))
    st.finalize([out], {**summ, "sets": sets})
    verify_protected(ctx)


# ============================================================================ XGBoost (S08)
def _xgb_cfg(ctx: Ctx) -> dict[str, Any]:
    return ctx.cfg["xgb"]


def _tree_X(ctx: Ctx, frame: pd.DataFrame, setname: str) -> pd.DataFrame:
    s = ctx.fs().sets[setname]
    return tree_matrix(frame, s["features"], s["baseline"])


def _rows(ctx: Ctx, fold: str) -> tuple[np.ndarray, np.ndarray | None]:
    """(training-row mask, held-out mask or None) of a study scope: 'outer<k>' or 'final'."""
    n = len(ctx.y_train())
    if fold == "final":
        return np.ones(n, dtype=bool), None
    k = int(fold.replace("outer", ""))
    te = ctx.outer_folds() == k
    return ~te, te


def _xgb_objective(ctx: Ctx, setname: str, fold: str) -> Any:
    x = _xgb_cfg(ctx)

    def objective(params: dict[str, Any], seed: int, tmp: Path) -> dict[str, Any]:
        tr_mask, _ = _rows(ctx, fold)
        X = _tree_X(ctx, ctx.train_frame().loc[tr_mask], setname)
        y = ctx.y_train()[tr_mask]
        cv = xgb_inner_cv(X, y, xgb_params(ctx.cfg, params, seed=seed, nthread=ctx.threads["xgboost"]), n_folds=int(ctx.cfg["cv"]["inner_folds_xgb"]),
                          seed=inner_fold_seed(ctx, "xgb", fold), n_max=int(x["n_estimators_max"]), early_stopping=int(x["early_stopping_rounds"]))
        return {"value": cv["mean_logloss"], "ap": cv["mean_ap"], "n_trees": cv["n_trees"], "best_iterations": [f["best_iteration"] for f in cv["folds"]]}
    return objective


def xgb_refit_item(ctx: Ctx, setname: str, fold: str, params: dict[str, Any], n_trees: int, config: str, *, keep_booster: bool) -> Any:
    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        tr_mask, te = _rows(ctx, fold)
        frame = ctx.train_frame()
        X = _tree_X(ctx, frame.loc[tr_mask], setname)
        bst = xgb_fit(X, ctx.y_train()[tr_mask], xgb_params(ctx.cfg, params, seed=seed, nthread=ctx.threads["xgboost"]), n_trees)
        if keep_booster:
            D.write_bytes(tmp / "booster.json", bytes(bst.save_raw("json")))
        out = {"config": config, "fold": fold, "failed": False, "params": params, "n_trees": int(n_trees), "importance": xgb_gain_importance(bst),
               "columns": list(X.columns)}
        if te is not None:
            D.write_npz(tmp / "oof.npz", rows=np.flatnonzero(te), p=xgb_predict(bst, _tree_X(ctx, frame.loc[te], setname)))
        return out
    return fn


def _stage0_item(ctx: Ctx, setname: str, fold: str, config: str) -> Any:
    x = _xgb_cfg(ctx)
    params = {k: x["stage0"][k] for k in ("max_depth", "min_child_weight", "reg_lambda")}

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        tr_mask, te = _rows(ctx, fold)
        frame = ctx.train_frame()
        X = _tree_X(ctx, frame.loc[tr_mask], setname)
        y = ctx.y_train()[tr_mask]
        p = xgb_params(ctx.cfg, params, seed=seed, nthread=ctx.threads["xgboost"])
        cv = xgb_inner_cv(X, y, p, n_folds=int(ctx.cfg["cv"]["inner_folds_xgb"]), seed=inner_fold_seed(ctx, "xgb", fold), n_max=int(x["n_estimators_max"]),
                          early_stopping=int(x["early_stopping_rounds"]))
        bst = xgb_fit(X, y, p, cv["n_trees"])
        D.write_bytes(tmp / "booster.json", bytes(bst.save_raw("json")))
        out = {"config": config, "fold": fold, "failed": False, "params": params, "n_trees": cv["n_trees"], "inner_cv_logloss": cv["mean_logloss"],
               "inner_cv_ap": cv["mean_ap"], "importance": xgb_gain_importance(bst), "columns": list(X.columns)}
        if te is not None:
            D.write_npz(tmp / "oof.npz", rows=np.flatnonzero(te), p=xgb_predict(bst, _tree_X(ctx, frame.loc[te], setname)))
        return out
    return fn


def _local_space(ctx: Ctx, best: dict[str, Any]) -> dict[str, dict[str, Any]]:
    x = _xgb_cfg(ctx)
    s1 = x["stage1"]["space"]
    f = float(x["stage2"]["local_factor"])
    b = best["params"]
    space = {"max_depth": {"type": "int", "low": max(int(s1["max_depth"]["low"]), int(b["max_depth"]) - 1),
                           "high": min(int(s1["max_depth"]["high"]), int(b["max_depth"]) + 1)}}
    for k in ("min_child_weight", "reg_lambda"):
        space[k] = {"type": "float", "low": max(float(s1[k]["low"]), float(b[k]) / f), "high": min(float(s1[k]["high"]), float(b[k]) * f), "log": True}
    space.update({k: dict(v) for k, v in x["stage2"]["space"].items()})
    return space


def s08_xgb(ctx: Ctx) -> None:
    from falls_ml.phase2.stages_data import verify_protected

    import_xgboost()
    st = ctx.run.stage("08", "xgb", ("07",))
    if st.complete_record():
        return
    x = _xgb_cfg(ctx)
    K = int(ctx.cfg["cv"]["outer_folds"])
    scopes = [f"outer{k}" for k in range(K)] + ["final"]
    # stage 0: defaults, inner-CV tree count, on the declared sets
    for s in x["stage0"]["sets"]:
        for fold in scopes:
            st.item(f"XGB0__{s}__{fold}", _stage0_item(ctx, s, fold, f"XGB_DEFAULT:{s}"))
    # stage 1: persistent Optuna TPE per scope
    store = TrialStore(ctx.run)
    tset = x["stage1"]["set"]
    meta = {"feature_set": tset, "feature_sets_sha256": ctx.fs().sha256, "plan_sha256": ctx.run.plan_sha}
    best1: dict[str, dict[str, Any]] = {}
    for fold in scopes:
        res = run_study(st, store, name=f"XGB1_{fold}", space=x["stage1"]["space"], n_trials=int(x["stage1"]["n_trials"]),
                        n_startup=int(x["stage1"]["n_startup_trials"]), seed_base=ctx.run.seed, objective=_xgb_objective(ctx, tset, fold), meta=meta)
        best1[fold] = best_trial(res)
        st.item(f"XGB1__{fold}__refit", xgb_refit_item(ctx, tset, fold, best1[fold]["params"], best1[fold]["n_trees"], f"XGB_TUNED_S1:{tset}",
                                                      keep_booster=True))
    # stage 2 (conditional, decided once from committed results)
    from falls_ml.evaluation.metrics import pr_auc

    p1 = np.full(len(ctx.y_train()), np.nan)
    for k in range(K):
        z = np.load(st.path(f"XGB1__outer{k}__refit") / "oof.npz")
        p1[z["rows"]] = z["p"]
    ap1 = float(pr_auc(ctx.y_train(), p1))
    lin = []
    for sid, name, fam in (("06", "lasso", LASSO), ("07", "enet", ENET)):
        stl = ctx.run.stage(sid, name)
        for rec in stl.records().values():
            if rec["item"].endswith("__final") and not rec["result"].get("failed"):
                v = oof_ap(ctx, stl, rec["item"][: -len("__final")])
                s = config_set(rec["result"]["config"])
                if v is not None and is_safe_candidate(ctx, s):
                    lin.append(v)
    best_lin = max(lin) if lin else float("nan")
    margin = float(x["stage2"]["condition_min_ap_gain_vs_best_linear"])
    enabled = bool(x["stage2"].get("enabled", True))
    decision = st.item("XGB2__decision", lambda tmp, seed: {"ap_stage1_oof": ap1, "best_linear_oof_ap": best_lin, "margin": margin, "stage2_enabled": enabled,
                                                            "run_stage2": bool(enabled and np.isfinite(best_lin) and ap1 >= best_lin + margin),
                                                            "note": None if enabled else "stage 2 disabled by the frozen configuration (Astra F-06)"})
    final_config = f"XGB_TUNED_S1:{tset}"
    if decision["run_stage2"]:
        for fold in scopes:
            res2 = run_study(st, store, name=f"XGB2_{fold}", space=_local_space(ctx, best1[fold]), n_trials=int(x["stage2"]["n_trials"]),
                             n_startup=max(2, int(x["stage2"]["n_trials"]) // 3), seed_base=ctx.run.seed, objective=_xgb_objective(ctx, tset, fold),
                             meta={**meta, "stage": 2})
            b = best_trial([best1[fold], *res2])
            st.item(f"XGB2__{fold}__refit", xgb_refit_item(ctx, tset, fold, b["params"], b["n_trees"], f"XGB_TUNED_S2:{tset}", keep_booster=True))
        final_config = f"XGB_TUNED_S2:{tset}"
    trials, _ = D.read_jsonl(store.ledger)
    out1 = D.write_csv(ctx.run.artifacts / "XGB_TRIALS.csv", pd.json_normalize(trials) if trials else pd.DataFrame())
    out2 = D.write_csv(ctx.run.artifacts / "FITS_XGB.csv", _fits_table(st))
    st.finalize([out1, out2], {"stage2_run": decision["run_stage2"], "tuned_config": final_config, "n_trials_ledger": len(trials)})
    verify_protected(ctx)


# ============================================================================ configurations and OOF predictions
def configurations(ctx: Ctx) -> dict[str, dict[str, Any]]:
    """Every fitted configuration -> {stage, prefix (outer items), final item, family, set}."""
    out: dict[str, dict[str, Any]] = {}
    for sid, name in (("06", "lasso"), ("07", "enet")):
        st = ctx.run.stage(sid, name)
        for rec in st.records().values():
            if rec["item"].endswith("__final"):
                fam, s, _ = rec["item"].split("__")
                out[f"{fam}:{s}"] = {"stage": (sid, name), "prefix": f"{fam}__{s}", "final": rec["item"], "family": fam, "set": s,
                                     "failed": bool(rec["result"].get("failed")), "category": ctx.fs().sets[s]["category"]}
    st8 = ctx.run.stage("08", "xgb")
    x = _xgb_cfg(ctx)
    for s in x["stage0"]["sets"]:
        out[f"XGB_DEFAULT:{s}"] = {"stage": ("08", "xgb"), "prefix": f"XGB0__{s}", "final": f"XGB0__{s}__final", "family": "XGB_DEFAULT", "set": s,
                                   "failed": False, "oof_items": [f"XGB0__{s}__outer{k}" for k in range(int(ctx.cfg["cv"]["outer_folds"]))],
                                   "category": ctx.fs().sets[s]["category"]}
    dec = st8.result("XGB2__decision") if st8.done("XGB2__decision") else {"run_stage2": False}
    tag = "XGB2" if dec["run_stage2"] else "XGB1"
    tset = x["stage1"]["set"]
    out[f"XGB_TUNED:{tset}"] = {"stage": ("08", "xgb"), "prefix": tag, "final": f"{tag}__final__refit", "family": "XGB_TUNED", "set": tset, "failed": False,
                                "oof_items": [f"{tag}__outer{k}__refit" for k in range(int(ctx.cfg["cv"]["outer_folds"]))],
                                "category": ctx.fs().sets[tset]["category"]}
    return out


def config_oof(ctx: Ctx, cfgs: dict[str, dict[str, Any]], name: str) -> np.ndarray | None:
    c = cfgs[name]
    st = ctx.run.stage(*c["stage"])
    if "oof_items" not in c:
        return oof(ctx, st, c["prefix"])
    p = np.full(len(ctx.y_train()), np.nan)
    for item in c["oof_items"]:
        z = np.load(st.path(item) / "oof.npz")
        p[z["rows"]] = z["p"]
    return p


def config_final_result(ctx: Ctx, cfgs: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    c = cfgs[name]
    return ctx.run.stage(*c["stage"]).result(c["final"])


def complexity(ctx: Ctx, cfgs: dict[str, dict[str, Any]], name: str) -> tuple[int, int]:
    """(family rank, number of non-zero / used features): linear before XGBoost, then fewer features (simplicity rule, plan §6)."""
    r = config_final_result(ctx, cfgs, name)
    if cfgs[name]["family"] in (LASSO, ENET):
        return 0, int(r.get("n_selected") or 0)
    return 1, len([k for k, v in (r.get("importance") or {}).items() if v > 0])


# ============================================================================ S09 selection (frozen before VALIDATION)
MEETS_SUPERIORITY_RULE, WITHIN_NONINFERIORITY_MARGINS, OUTSIDE_MARGINS = "MEETS_SUPERIORITY_RULE", "WITHIN_NONINFERIORITY_MARGINS", "OUTSIDE_MARGINS"
NO_INCREMENTAL_MODEL_SELECTED = "NO_INCREMENTAL_MODEL_SELECTED"


def benchmark_class(d_ap: float, d_cap_pp: float, *, superior: bool, max_ap_loss: float, max_cap_loss_pp: float) -> str:
    """Pre-declared POINT-ESTIMATE classification of a SAFE model against the BASELINE_15 refit (Astra F-03: not a formal superiority or
    non-inferiority test; never mixed with the discovery rule)."""
    if superior:
        return MEETS_SUPERIORITY_RULE
    if np.isfinite(d_ap) and np.isfinite(d_cap_pp) and d_ap >= -float(max_ap_loss) and d_cap_pp >= -float(max_cap_loss_pp):
        return WITHIN_NONINFERIORITY_MARGINS
    return OUTSIDE_MARGINS


def _meets(row: Any, *, d_ap: str, d_cap: str, ll_ratio: float, sel: dict[str, Any], slope_range: Any, max_citl: float, prefix: str = "oof") -> bool:
    """The pre-declared rule; an undefined (NaN) quantity never passes (Astra F-08)."""
    lo, hi = slope_range
    vals = [row[d_ap], row[d_cap], row[f"{prefix}_logloss"], row[f"{prefix}_cal_slope"], row[f"{prefix}_citl"], ll_ratio]
    if not all(np.isfinite(float(v)) for v in vals):
        return False
    return bool(row[d_ap] >= float(sel["min_delta_ap"]) and 100.0 * row[d_cap] >= float(sel["min_delta_capture_principal_pp"])
                and row[f"{prefix}_logloss"] <= float(sel["max_logloss_ratio"]) * ll_ratio and float(lo) <= row[f"{prefix}_cal_slope"] <= float(hi)
                and abs(row[f"{prefix}_citl"]) <= float(max_citl))


def attribution_contrasts(ctx: Ctx, cfgs: dict[str, dict[str, Any]], oof_ap: dict[str, float]) -> list[dict[str, str]]:
    """The two frozen attribution contrasts (Astra F-09): FEATURE_EXPANSION = LASSO on ALL_REVIEWED_SAFE vs LASSO:SAFE_BASE (same family,
    more SAFE features); MODEL_FAMILY_PIPELINE = tuned XGBoost vs the TRAIN-selected (OOF AP) best linear model on the SAME set (a pipeline
    comparison: encodings and missing-value handling differ by design)."""
    out = []
    full = lasso_config(ctx, "ALL_REVIEWED_SAFE")
    if full != SAFE_REF and full in cfgs and not cfgs[full]["failed"]:
        out.append({"contrast": "FEATURE_EXPANSION", "new": full, "reference": SAFE_REF,
                    "question": "what do the SAFE new features add to the SAFE baseline predictors (same model family)?"})
    lin = [c for c in (full, "ENET:ALL_REVIEWED_SAFE") if c in cfgs and not cfgs[c]["failed"] and np.isfinite(oof_ap.get(c, np.nan))]
    xgb = "XGB_TUNED:ALL_REVIEWED_SAFE"
    if lin and xgb in cfgs:
        best = max(sorted(lin), key=lambda c: oof_ap[c])
        out.append({"contrast": "MODEL_FAMILY_PIPELINE", "new": xgb, "reference": best,
                    "question": "does gradient boosting add to the best linear model on the same SAFE feature set (pipeline comparison)?"})
    return out


def s09_select(ctx: Ctx) -> None:
    st = ctx.run.stage("09", "select", ("08",))
    if st.complete_record():
        return
    sel = ctx.cfg["selection"]
    g = ctx.cfg["gates"]
    principal = float(sel["principal_capacity"])
    caps = tuple(float(c) for c in ctx.cfg["metrics"]["capacities"])
    y = ctx.y_train()
    fs = ctx.fs()

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        cfgs = configurations(ctx)
        bench, sref = config_oof(ctx, cfgs, BENCHMARK), config_oof(ctx, cfgs, SAFE_REF)
        if bench is None or sref is None:
            raise Phase2Stop("REFERENCE_FAILED", "the BASELINE_15 benchmark or the SAFE_BASE reference could not be fitted in every outer fold")
        bb = bundle(y, bench, capacities=caps, principal=principal)
        sb = bundle(y, sref, capacities=caps, principal=principal)
        nb = int(ctx.cfg["metrics"]["bootstrap_n"])
        rows = []
        for name in sorted(cfgs):
            p = config_oof(ctx, cfgs, name)
            s = cfgs[name]["set"]
            meta = fs.sets[s]
            row = {"config": name, "family": cfgs[name]["family"], "feature_set": s, "category": meta["category"], "set_kind": meta["kind"],
                   "n_new_features_offered": meta["n_new_features"], "n_baseline_features": meta["n_baseline_features"], "failed": p is None}
            if p is not None:
                b = bundle(y, p, capacities=caps, principal=principal)
                row.update({f"oof_{k}": v for k, v in b.items()})
                if name != BENCHMARK:   # every configuration vs the historical benchmark (paired patient bootstrap)
                    row.update({f"oof_{k}": v for k, v in paired_bootstrap(y, bench, p, n_boot=nb, seed=seed, principal=principal).items()
                                if k.startswith("delta")})
                if meta["category"] == SAFE_DISCOVERY:   # SAFE configurations vs SAFE_BASE (point estimates; intervals for the shortlist below)
                    row.update({"oof_delta_ap_vs_safe_base": b["ap"] - sb["ap"], "oof_delta_auroc_vs_safe_base": b["auroc"] - sb["auroc"],
                                "oof_delta_capture_principal_vs_safe_base": b[f"capture@{principal:g}"] - sb[f"capture@{principal:g}"],
                                "oof_logloss_ratio_vs_safe_base": b["logloss"] / sb["logloss"]})
                row["complexity_rank"], row["n_features_used"] = complexity(ctx, cfgs, name)
            rows.append(row)
        t = pd.DataFrame(rows)
        # ---- plausibility gate vs the historical benchmark (investigation stop): SAFE and exploratory configurations
        others = t[t["category"].isin([SAFE_DISCOVERY, EXPLORATORY]) & ~t["failed"]]
        jumps = others[(others["oof_auroc"] - bb["auroc"] > float(g["max_delta_auroc"])) | (others["oof_ap"] / bb["ap"] > float(g["max_ap_ratio"]))]
        # ---- advancement rule: SAFE_DISCOVERY candidates vs SAFE_BASE
        cand = t[(t["category"] == SAFE_DISCOVERY) & t["set_kind"].isin(list(CANDIDATE_KINDS)) & ~t["failed"] & (t["config"] != SAFE_REF)]
        ok = cand.apply(lambda r: _meets(r, d_ap="oof_delta_ap_vs_safe_base", d_cap="oof_delta_capture_principal_vs_safe_base", ll_ratio=sb["logloss"],
                                         sel=sel, slope_range=sel["calibration_slope_range"], max_citl=sel["max_abs_citl"]), axis=1) if len(cand) else pd.Series(dtype=bool)
        q = cand[ok.astype(bool)] if len(cand) else cand
        recommended, why = SAFE_REF, ("NO_INCREMENTAL_MODEL_SELECTED: no SAFE_DISCOVERY configuration met every pre-declared advancement criterion "
                                      "over SAFE_BASE; SAFE_BASE is reported, not selected")
        if len(q):
            best_ap = float(q["oof_ap"].max())
            near = q[q["oof_ap"] >= best_ap - float(sel["simplicity_margin_ap"])].sort_values(["complexity_rank", "n_features_used", "oof_ap", "config"],
                                                                                           ascending=[True, True, False, True])
            recommended = str(near.iloc[0]["config"])
            why = (f"highest OOF AP among {len(q)} qualifying SAFE configuration(s) ({best_ap:.4f}); simplest within {sel['simplicity_margin_ap']} AP: "
                   f"{recommended}")
        t["qualifies_vs_safe_base"] = t["config"].isin(q["config"])
        # ---- benchmark class of the recommended SAFE model vs BASELINE_15 (pre-declared margins; reported, never used to choose)
        rrow = t.set_index("config").loc[recommended]
        ni = sel["benchmark_noninferiority"]
        sup = _meets(rrow, d_ap="oof_delta_ap", d_cap="oof_delta_capture_principal", ll_ratio=bb["logloss"], sel=sel,
                     slope_range=sel["calibration_slope_range"], max_citl=sel["max_abs_citl"])
        bclass = benchmark_class(float(rrow["oof_delta_ap"]), 100.0 * float(rrow["oof_delta_capture_principal"]), superior=sup,
                                 max_ap_loss=ni["max_ap_loss"], max_cap_loss_pp=ni["max_capture_loss_pp"])
        # ---- shortlist: the two references, the recommended model, then the best SAFE candidate of each other family (<= max_challengers)
        shortlist = [SAFE_REF, BENCHMARK] + ([recommended] if recommended != SAFE_REF else [])
        for fam in ("XGB_TUNED", "ENET", "LASSO", "XGB_DEFAULT"):
            if len(shortlist) - 2 >= int(sel["max_challengers"]):
                break
            if any(config_family(c) == fam or (fam.startswith("XGB") and config_family(c).startswith("XGB")) for c in shortlist[2:]):
                continue
            pool = cand[cand["family"] == fam]
            if len(pool):
                shortlist.append(str(pool.sort_values(["oof_ap", "config"], ascending=[False, True]).iloc[0]["config"]))
        # paired intervals vs SAFE_BASE for the SAFE members of the shortlist
        vs_safe = {}
        for c in shortlist:
            if c not in (SAFE_REF, BENCHMARK) and cfgs[c]["category"] == SAFE_DISCOVERY:
                vs_safe[c] = paired_bootstrap(y, sref, config_oof(ctx, cfgs, c), n_boot=nb, seed=seed, principal=principal)
        for c, d in vs_safe.items():
            for k, v in d.items():
                if k.startswith("delta") and ("_ci_" in k or k.endswith("share_gt0")):
                    t.loc[t["config"] == c, f"oof_{k}_vs_safe_base"] = v
        oap = {r["config"]: float(r.get("oof_ap", np.nan)) for r in rows}
        contr = attribution_contrasts(ctx, cfgs, oap)
        crow = []
        for c in contr:
            d = paired_bootstrap(y, config_oof(ctx, cfgs, c["reference"]), config_oof(ctx, cfgs, c["new"]), n_boot=nb, seed=seed, principal=principal)
            crow.append({**c, **{f"oof_{k}": v for k, v in d.items() if k.startswith("delta")}})
        D.write_csv(tmp / "ATTRIBUTION_CONTRASTS_OOF.csv", pd.DataFrame(crow))
        D.write_csv(tmp / "OOF_MODEL_COMPARISON.csv", t)
        D.write_npz(tmp / "oof_reference.npz", p=bench, p_safe_base=sref)
        return {"recommended": recommended, "reason": why, "shortlist": shortlist, "benchmark": BENCHMARK, "discovery_reference": SAFE_REF,
                "selection_outcome": "INCREMENTAL_MODEL_SELECTED" if recommended != SAFE_REF else NO_INCREMENTAL_MODEL_SELECTED,
                "attribution_contrasts": contr,
                "benchmark_oof": bb, "safe_base_oof": sb, "benchmark_class_oof": bclass, "n_qualifying": int(len(q)), "qualifying": q["config"].tolist(),
                "implausible_gain": jumps["config"].tolist(), "recommended_cal_slope": float(rrow["oof_cal_slope"]),
                "recommended_citl": float(rrow["oof_citl"]), "descriptive_configs": sorted(c for c in cfgs if not cfgs[c]["failed"])}

    res = st.item("selection", fn)
    ctx.run.gate("IMPLAUSIBLE_GAIN_OOF", bool(res["implausible_gain"]),
                 f"out-of-fold gain vs the BASELINE_15 benchmark above the plausibility limits (dAUROC > {g['max_delta_auroc']} or AP ratio > {g['max_ap_ratio']})",
                 [f"config: {c}" for c in res["implausible_gain"]])
    slo, shi = g["calibration_slope_range"]
    bad_cal = res["recommended"] != SAFE_REF and not (float(slo) <= res["recommended_cal_slope"] <= float(shi) and abs(res["recommended_citl"]) <= float(g["max_abs_citl"]))
    ctx.run.gate("IMPLAUSIBLE_CALIBRATION_OOF", bad_cal, "the recommended SAFE model's out-of-fold calibration is implausible",
                 [f"slope {res['recommended_cal_slope']:.3f}, CITL {res['recommended_citl']:.3f}"])
    frozen = {"frozen_at": utc_now(), "rule": "planning/EXPERIMENT_PLAN.md §6 advancement rule on TRAIN outer-OOF (VALIDATION not read)",
              "candidates": "SAFE_DISCOVERY configurations (main / one-domain / waterfall sets; LASSO, elastic net, XGBoost)",
              "selection_outcome": res["selection_outcome"], "recommended": res["recommended"], "reason": res["reason"], "shortlist": res["shortlist"],
              "discovery_reference": SAFE_REF, "benchmark": BENCHMARK,
              "benchmark_note": "LASSO:BASELINE_15 = the 15 historical predictors REFITTED on TRAIN in the Phase 2 protocol (no recalibration); not the "
                                "Phase 1 model, which was recalibrated on these VALIDATION outcomes (Astra F-02)",
              "benchmark_class_oof": res["benchmark_class_oof"], "benchmark_class_kind": "point-estimate classification (Astra F-03)",
              "attribution_contrasts": res["attribution_contrasts"], "qualifying": res["qualifying"],
              "descriptive_configs": res["descriptive_configs"], "selection_item_files_sha256": st.records()["selection"]["files"],
              "plan_sha256": ctx.run.plan_sha, "final_experiment_config_sha256": ctx.run.plan.get("final_experiment_config_sha256"),
              "feature_sets_sha256": ctx.fs().sha256}
    sel_path = D.write_json(ctx.run.out / "SELECTION_FROZEN.json", frozen)
    out = D.write_bytes(ctx.run.artifacts / "OOF_MODEL_COMPARISON.csv", (st.path("selection") / "OOF_MODEL_COMPARISON.csv").read_bytes())
    out2 = D.write_bytes(ctx.run.artifacts / "ATTRIBUTION_CONTRASTS_OOF.csv", (st.path("selection") / "ATTRIBUTION_CONTRASTS_OOF.csv").read_bytes())
    st.finalize([sel_path, out, out2], {"recommended": res["recommended"], "shortlist": res["shortlist"], "benchmark_class_oof": res["benchmark_class_oof"],
                                        "selection_outcome": res["selection_outcome"]})


def selection(ctx: Ctx) -> dict[str, Any]:
    p = ctx.run.out / "SELECTION_FROZEN.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["sha256"] = D.sha256_file(p)
    return rec


# ============================================================================ S10 stability
def stability_sets(ctx: Ctx, sel: dict[str, Any]) -> list[str]:
    """Finalists only: the feature sets of the SAFE linear shortlist members (recommended first), ALL_REVIEWED_SAFE (for the consensus
    table), then the historical BASELINE_15 as the reference of selection stability, up to max_sets."""
    cfgs = configurations(ctx)
    lin = [c for c in sel["shortlist"] if cfgs[c]["family"] in (LASSO, ENET) and cfgs[c]["category"] == SAFE_DISCOVERY and c != SAFE_REF]
    sets: list[str] = []
    for s in [*(config_set(c) for c in lin), "ALL_REVIEWED_SAFE", "BASELINE_15"]:
        if s not in sets:
            sets.append(s)
    return sets[: int(ctx.cfg["stability"]["max_sets"])]


def s10_stability(ctx: Ctx) -> None:
    st = ctx.run.stage("10", "stability", ("09",))
    if st.complete_record():
        return
    sel = selection(ctx)
    sets = st.item("STAB__plan", lambda tmp, seed: {"sets": stability_sets(ctx, sel)})["sets"]
    n_rep = int(ctx.cfg["stability"]["n_replicates"])
    tr = ctx.train_frame()
    y = ctx.y_train()
    for s in sets:
        for r in range(n_rep):
            def fn(tmp: Path, seed: int, s: str = s) -> dict[str, Any]:
                rng = np.random.default_rng(seed)
                idx = rng.integers(0, len(y), len(y))
                frame = tr.iloc[idx].reset_index(drop=True)
                try:
                    fit, retry = _fit_with_retry(ctx, LASSO, frame, y[idx], ctx.fs().sets[s], seed, groups=idx)
                except FIT_ERRORS as exc:
                    return {"set": s, "failed": True, "error": f"{type(exc).__name__}: {exc}"[:300]}
                coef = fit.coefficients()
                sign = {f: float(np.sign(g["coefficient"].sum())) for f, g in coef[coef["selected"]].groupby("feature")}
                return {"set": s, "failed": False, "selected": fit.selected_features(), "sign": sign, "retry": retry["retry"]}
            st.item(f"STAB__{s}__r{r:02d}", fn)
    rows, gate = [], []
    for s in sets:
        res = [st.result(f"STAB__{s}__r{r:02d}") for r in range(n_rep)]
        ok = [r for r in res if not r["failed"]]
        frac = 1 - len(ok) / len(res)
        if frac > float(ctx.cfg["stability"]["max_failure_fraction"]):
            gate.append(f"{s}: {len(res) - len(ok)} of {len(res)} replicates failed")
        feats = sorted({f for r in ok for f in r["selected"]} | set(ctx.fs().sets[s]["features"]) | set(ctx.fs().sets[s]["baseline"]))
        for f in feats:   # a feature is "selected" when any of its encoded design columns is non-zero (Astra F-13)
            sel_n = sum(1 for r in ok if f in r["selected"])
            signs = [r["sign"].get(f) for r in ok if f in r["sign"]]
            pos = sum(1 for v in signs if v and v > 0)
            freq = sel_n / len(ok) if ok else None
            rows.append({"feature_set": s, "family": "LASSO", "feature": f, "n_replicates_ok": len(ok), "selection_count": sel_n,
                         "selection_frequency": freq, "selection_frequency_mc_se": float(np.sqrt(freq * (1 - freq) / len(ok))) if ok else None,
                         "sign_positive_share": pos / len(signs) if signs else None, "n_failed_replicates": len(res) - len(ok),
                         "retries": sum(1 for r in ok if r.get("retry")),
                         "note": "coarse: 25 bootstrap refits of LASSO only (never certifies elastic-net-only or XGBoost findings)"})
    if gate:
        raise Phase2Stop("FIT_FAILURES", "stability replicates failed above the pre-declared limit", gate)
    out = D.write_csv(ctx.run.artifacts / "STABILITY.csv", pd.DataFrame(rows))
    st.finalize([out], {"sets": sets, "n_replicates": n_rep})


# ============================================================================ S11 ablation
def ablation_target(ctx: Ctx, sel: dict[str, Any]) -> str | None:
    """The recommended SAFE model; if that is SAFE_BASE, the best SAFE candidate with new features by OOF AP (explanatory only)."""
    if sel["recommended"] != SAFE_REF:
        return sel["recommended"]
    t = pd.read_csv(ctx.run.artifacts / "OOF_MODEL_COMPARISON.csv")
    t = t[(t["category"] == SAFE_DISCOVERY) & t["set_kind"].isin(list(CANDIDATE_KINDS)) & ~t["failed"].astype(bool) & (t["n_new_features_offered"] > 0)]
    return None if t.empty else str(t.sort_values(["oof_ap", "config"], ascending=[False, True]).iloc[0]["config"])


def s11_ablation(ctx: Ctx) -> None:
    from falls_ml.phase2.stages_data import verify_protected

    st = ctx.run.stage("11", "ablation", ("10",))
    if st.complete_record():
        return
    sel = selection(ctx)
    cfgs = configurations(ctx)
    K = int(ctx.cfg["cv"]["outer_folds"])
    # (a) leave-one-domain-out, fully nested (LASSO)
    loo = _sets_in_order(ctx, ("loo",))
    for s in loo:
        for k in range(K):
            st.item(f"LASSO__{s}__outer{k}", linear_outer_item(ctx, LASSO, s, k))
    # (b) individual ablation of the top new features in the target configuration, per-fold hyper-parameters frozen
    target = st.item("IND__plan", lambda tmp, seed: _individual_plan(ctx, sel, cfgs))
    for f in target.get("features", []):
        for k in range(K):
            st.item(f"IND__{f}__outer{k}", _individual_item(ctx, cfgs, target["config"], f, k))
    y = ctx.y_train()
    principal = float(ctx.cfg["selection"]["principal_capacity"])
    rows = []
    full_cfg = lasso_config(ctx, "ALL_REVIEWED_SAFE")
    full = config_oof(ctx, cfgs, full_cfg)
    for s in loo:
        p = oof(ctx, st, f"LASSO__{s}")
        if p is None or full is None:
            continue
        d = paired_bootstrap(y, p, full, n_boot=int(ctx.cfg["metrics"]["bootstrap_n"]), seed=ctx.run.item_seed("S11", s), principal=principal)
        rows.append({"analysis": "leave_one_domain_out", "model": full_cfg, "removed": ctx.fs().sets[s]["domain"], "set": s,
                     **{f"{k}_full_minus_ablated": v for k, v in d.items() if k.startswith("delta")}})
    if target.get("config"):
        base = config_oof(ctx, cfgs, target["config"])
        for f in target.get("features", []):
            p = np.full(len(y), np.nan)
            failed = False
            for k in range(K):
                r = st.result(f"IND__{f}__outer{k}")
                if r.get("failed"):
                    failed = True
                    break
                z = np.load(st.path(f"IND__{f}__outer{k}") / "oof.npz")
                p[z["rows"]] = z["p"]
            if failed:
                rows.append({"analysis": "individual", "model": target["config"], "removed": f, "failed": True})
                continue
            d = paired_bootstrap(y, p, base, n_boot=int(ctx.cfg["metrics"]["bootstrap_n"]), seed=ctx.run.item_seed("S11", f), principal=principal)
            rows.append({"analysis": "individual", "model": target["config"], "removed": f, "importance_rank": target["features"].index(f) + 1,
                         **{f"{k}_full_minus_ablated": v for k, v in d.items() if k.startswith("delta")}})
    out = D.write_csv(ctx.run.artifacts / "ABLATION_RESULTS.csv", pd.DataFrame(rows))
    st.finalize([out], {"loo_sets": len(loo), "individual_target": target.get("config"), "n_individual": len(target.get("features", []))})
    verify_protected(ctx)


def _individual_plan(ctx: Ctx, sel: dict[str, Any], cfgs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    target = ablation_target(ctx, sel)
    if target is None:
        return {"config": None, "features": []}
    imp = config_final_result(ctx, cfgs, target).get("importance") or {}
    s = ctx.fs().sets[cfgs[target]["set"]]
    new = [f for f in s["features"]]
    ranked = sorted((f for f in new if imp.get(f, 0) > 0), key=lambda f: (-imp[f], f))
    return {"config": target, "features": ranked[: int(ctx.cfg["ablation"]["individual_top_n"])],
            "ranking": "final-model importance on TRAIN (linear: sum |standardised coefficient|; XGBoost: total gain)"}


def _individual_item(ctx: Ctx, cfgs: dict[str, dict[str, Any]], config: str, feature: str, k: int) -> Any:
    c = cfgs[config]
    s = ctx.fs().sets[c["set"]]
    new = [f for f in s["features"] if f != feature]
    base = [f for f in s["baseline"] if f != feature]
    if c["family"] in (LASSO, ENET):
        fold_res = ctx.run.stage(*c["stage"]).result(f"{c['prefix']}__outer{k}")
        fixed = {"lambda": fold_res["lambda"], "l1_ratio": fold_res["l1_ratio"]}

        def fn(tmp: Path, seed: int) -> dict[str, Any]:
            tr, y, te = ctx.train_frame(), ctx.y_train(), ctx.outer_folds() == k
            s2 = {**s, "baseline": base}
            try:
                fit, _ = _fit_with_retry(ctx, c["family"], tr.loc[~te], y[~te], s2, seed, fixed=fixed, new=new)
            except FIT_ERRORS as exc:
                return {"failed": True, "error": str(exc)[:300]}
            D.write_npz(tmp / "oof.npz", rows=np.flatnonzero(te), p=fit.predict(tr.loc[te]))
            return {"failed": False, "fixed": fixed}
        return fn
    item = c["oof_items"][k]
    fold_res = ctx.run.stage(*c["stage"]).result(item)

    def fnx(tmp: Path, seed: int) -> dict[str, Any]:
        tr, y, te = ctx.train_frame(), ctx.y_train(), ctx.outer_folds() == k
        X = tree_matrix(tr, new, base)
        bst = xgb_fit(X.loc[~te], y[~te], xgb_params(ctx.cfg, fold_res["params"], seed=seed, nthread=ctx.threads["xgboost"]), fold_res["n_trees"])
        D.write_npz(tmp / "oof.npz", rows=np.flatnonzero(te), p=xgb_predict(bst, X.loc[te]))
        return {"failed": False, "params": fold_res["params"], "n_trees": fold_res["n_trees"]}
    return fnx


# ============================================================================ S12 explainability (promoted XGBoost only)
def explain_target(ctx: Ctx, sel: dict[str, Any]) -> str | None:
    xs = [c for c in sel["shortlist"] if config_family(c).startswith("XGB")]
    return xs[0] if xs else None


def s12_explain(ctx: Ctx) -> None:
    st = ctx.run.stage("12", "explain", ("11",))
    if st.complete_record():
        return
    import_xgboost()
    import xgboost as xgb

    sel = selection(ctx)
    cfgs = configurations(ctx)
    target = explain_target(ctx, sel)
    ex = ctx.cfg["explain"]
    outs = []
    if target is not None:
        c = cfgs[target]
        s = ctx.fs().sets[c["set"]]
        tr, y = ctx.train_frame(), ctx.y_train()
        X = tree_matrix(tr, s["features"], s["baseline"])
        K = int(ctx.cfg["cv"]["outer_folds"])
        items = c["oof_items"]
        stx = ctx.run.stage(*c["stage"])
        for k in range(K):
            def fn(tmp: Path, seed: int, k: int = k) -> dict[str, Any]:
                from falls_ml.evaluation.metrics import pr_auc
                from falls_ml.phase2.evaluate import logloss

                te = ctx.outer_folds() == k
                bst = xgb.Booster()
                bst.load_model(bytearray((stx.path(items[k]) / "booster.json").read_bytes()))
                Xt, yt = X.loc[te].reset_index(drop=True), y[te]
                dm = lambda M: xgb.DMatrix(M.to_numpy(dtype=np.float32), feature_names=list(M.columns), missing=np.nan)  # noqa: E731
                p0 = bst.predict(dm(Xt))
                ap0, ll0 = float(pr_auc(yt, p0)), logloss(yt, p0)
                rng = np.random.default_rng(seed)
                rows = []
                for col in Xt.columns:
                    dap, dll = [], []
                    for _ in range(int(ex["permutation_repeats"])):
                        Xp = Xt.copy()
                        Xp[col] = Xp[col].to_numpy()[rng.permutation(len(Xp))]
                        p = bst.predict(dm(Xp))
                        dap.append(ap0 - float(pr_auc(yt, p)))
                        dll.append(logloss(yt, p) - ll0)
                    rows.append({"feature": tree_feature_of(col), "fold": k, "delta_ap_mean": float(np.mean(dap)), "delta_logloss_mean": float(np.mean(dll)),
                                 "n_rows": int(len(yt))})
                D.write_csv(tmp / "permutation.csv", pd.DataFrame(rows))
                return {"fold": k, "ap_baseline": ap0}
            st.item(f"PERM__outer{k}", fn)

        def shap_fn(tmp: Path, seed: int) -> dict[str, Any]:
            bst = xgb.Booster()
            bst.load_model(bytearray((stx.path(c["final"]) / "booster.json").read_bytes()))
            rng = np.random.default_rng(seed)
            n = min(int(ex["shap_sample_rows"]), len(y))
            idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == v), size=max(1, round(n * (y == v).mean())), replace=False) for v in (0, 1)]))
            Xs = X.iloc[idx]
            dm = xgb.DMatrix(Xs.to_numpy(dtype=np.float32), feature_names=list(Xs.columns), missing=np.nan)
            contrib = bst.predict(dm, pred_contribs=True)[:, :-1]
            q = np.percentile(contrib, [5, 25, 50, 75, 95], axis=0)
            summ = pd.DataFrame({"feature": [tree_feature_of(cn) for cn in Xs.columns], "mean_abs_shap": np.abs(contrib).mean(axis=0),
                                 "mean_shap": contrib.mean(axis=0), "p05": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4]})
            summ = summ.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
            summ["rank"] = np.arange(1, len(summ) + 1)
            summ["share_of_total"] = summ["mean_abs_shap"] / summ["mean_abs_shap"].sum()
            m = min(2000, len(idx))
            dmi = xgb.DMatrix(Xs.iloc[:m].to_numpy(dtype=np.float32), feature_names=list(Xs.columns), missing=np.nan)
            inter = np.abs(bst.predict(dmi, pred_interactions=True)[:, :-1, :-1]).mean(axis=0)
            np.fill_diagonal(inter, 0.0)
            iu = np.triu_indices_from(inter, 1)
            order = np.argsort(-inter[iu], kind="mergesort")[: int(ex["interaction_pairs"])]
            pairs = pd.DataFrame({"feature_a": [tree_feature_of(Xs.columns[iu[0][o]]) for o in order], "feature_b": [tree_feature_of(Xs.columns[iu[1][o]]) for o in order],
                                  "mean_abs_interaction": [float(2 * inter[iu[0][o], iu[1][o]]) for o in order]})
            D.write_csv(tmp / "SHAP_SUMMARY.csv", summ)
            D.write_csv(tmp / "SHAP_INTERACTIONS.csv", pairs)
            return {"n_rows_sample": int(len(idx)), "n_rows_interactions": int(m)}
        st.item("SHAP", shap_fn)
        perm = pd.concat([pd.read_csv(st.path(f"PERM__outer{k}") / "permutation.csv") for k in range(K)], ignore_index=True)
        prow = []
        for feat, g_ in perm.groupby("feature", sort=True):
            prow.append({"feature": feat, "delta_ap_mean": float(np.average(g_["delta_ap_mean"], weights=g_["n_rows"])),
                         "delta_ap_sd_over_folds": float(g_["delta_ap_mean"].std(ddof=1)) if len(g_) > 1 else 0.0,
                         "delta_logloss_mean": float(np.average(g_["delta_logloss_mean"], weights=g_["n_rows"]))})
        pagg = pd.DataFrame(prow)
        pagg = pagg.sort_values(["delta_ap_mean", "feature"], ascending=[False, True]).reset_index(drop=True)
        pagg["rank"] = np.arange(1, len(pagg) + 1)
        pagg.insert(0, "model", target)
        shap = pd.read_csv(st.path("SHAP") / "SHAP_SUMMARY.csv")
        shap.insert(0, "model", target)
        inter = pd.read_csv(st.path("SHAP") / "SHAP_INTERACTIONS.csv")
        outs += [D.write_csv(ctx.run.artifacts / "PERMUTATION_IMPORTANCE.csv", pagg), D.write_csv(ctx.run.artifacts / "SHAP_SUMMARY.csv", shap),
                 D.write_csv(ctx.run.artifacts / "SHAP_INTERACTIONS.csv", inter)]
        newf = set(s["features"])
        sh_new = shap[shap["feature"].isin(newf)]
        pos = pagg[pagg["delta_ap_mean"] > 0]
        dom = []
        lim = float(ctx.cfg["gates"]["max_single_feature_share"])
        if len(sh_new) and float(sh_new["share_of_total"].max()) > lim:
            dom.append(f"SHAP: {sh_new.iloc[0]['feature']} holds {float(sh_new['share_of_total'].max()):.0%} of mean |SHAP|")
        if len(pos) and pos["delta_ap_mean"].sum() > 0:
            share = pos.set_index("feature")["delta_ap_mean"] / pos["delta_ap_mean"].sum()
            share = share[share.index.isin(newf)]
            if len(share) and float(share.max()) > lim:
                dom.append(f"permutation: {share.idxmax()} holds {float(share.max()):.0%} of the AP loss")
        ctx.run.gate("PROXY_DOMINANCE", bool(dom), "one new feature dominates the promoted XGBoost model (possible proxy / leak)", dom)
    st.finalize(outs, {"target": target})


# ============================================================================ S13 VALIDATION (opened once, after the frozen selection)
def _predict_final(ctx: Ctx, cfgs: dict[str, dict[str, Any]], name: str, frame: pd.DataFrame) -> np.ndarray:
    c = cfgs[name]
    st = ctx.run.stage(*c["stage"])
    if c["family"] in (LASSO, ENET):
        fit = D.read_pickle(st.path(c["final"]) / "model.pkl")
        return fit.predict(frame)
    import xgboost as xgb

    bst = xgb.Booster()
    bst.load_model(bytearray((st.path(c["final"]) / "booster.json").read_bytes()))
    s = ctx.fs().sets[c["set"]]
    return xgb_predict(bst, tree_matrix(frame, s["features"], s["baseline"]))


def subgroup_masks(frame: pd.DataFrame, cat: Any) -> dict[str, dict[str, np.ndarray]]:
    age = pd.to_numeric(frame["age_years"], errors="coerce")
    forms = [cat.forms[f]["indicator"] for f in cat.forms if cat.forms[f]["indicator"] != "form_mefi_assessed" and cat.forms[f]["indicator"] in frame]
    any_form = frame[forms].fillna(0).sum(axis=1) > 0 if forms else pd.Series(False, index=frame.index)
    out = {"sex": {"female": (frame["sex"].astype(str) == "female").to_numpy(), "male": (frame["sex"].astype(str) == "male").to_numpy()},
           "age_band": {"65-74": (age < 75).to_numpy(), "75-84": ((age >= 75) & (age < 85)).to_numpy(), "85+": (age >= 85).to_numpy()},
           "nurse_form_assessed": {"any": any_form.to_numpy(), "none": (~any_form).to_numpy()},
           "prior_fall": {"yes": (pd.to_numeric(frame["falls"]) == 1).to_numpy(), "no": (pd.to_numeric(frame["falls"]) == 0).to_numpy()}}
    if "form_mefi_assessed" in frame:
        out["mefi_assessed"] = {"yes": (frame["form_mefi_assessed"] == 1).to_numpy(), "no": (frame["form_mefi_assessed"] != 1).to_numpy()}
    return out


def management_configs(ctx: Ctx, sel: dict[str, Any]) -> list[str]:
    """The configurations of the management waterfall, in order: BASELINE_15 refit, SAFE_BASE, each cumulative SAFE step (the last one =
    ALL_REVIEWED_SAFE = ALL_SAFE_LINEAR), ALL_SAFE_ELASTIC_NET, ALL_SAFE_XGBOOST, then the recommended model if not already listed."""
    cfgs = configurations(ctx)
    out = [BENCHMARK, SAFE_REF, *[lasso_config(ctx, w["set"]) for w in ctx.fs().design.get("waterfall", [])],
           "ENET:ALL_REVIEWED_SAFE", "XGB_TUNED:ALL_REVIEWED_SAFE", sel["recommended"]]
    return [c for c in dict.fromkeys(out) if c in cfgs]


def role_of(config: str, sel: dict[str, Any]) -> str:
    if config == BENCHMARK:
        return "benchmark"
    if config == SAFE_REF:
        return "discovery_reference"
    if config == sel["recommended"]:
        return "recommended"
    return "shortlist" if config in sel["shortlist"] else "descriptive"


def open_validation(ctx: Ctx, sel: dict[str, Any]) -> dict[str, Any]:
    """Write VALIDATION_OPENED.json ONCE, before any VALIDATION row is read. A later attempt (after an interruption) may re-read VALIDATION
    only for the identical frozen selection and final configuration; anything else is a hard stop (VALIDATION is never a tuning set)."""
    path = ctx.run.out / "VALIDATION_OPENED.json"
    fin_sha = ctx.run.plan.get("final_experiment_config_sha256")
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("frozen_selection_sha256") != sel["sha256"] or old.get("final_experiment_config_sha256") != fin_sha:
            raise Phase2Stop("VALIDATION_REOPEN_REFUSED", "VALIDATION was already opened for a different frozen selection or configuration",
                             [f"opened for selection {str(old.get('frozen_selection_sha256'))[:16]}…", f"now {sel['sha256'][:16]}…"])
        ctx.run.events("S13_validation", "validation_reread_same_selection", "CONTINUE", opened_at=old.get("opened_at"))
        return old
    rec = {"opened_at": utc_now(), "attempt": ctx.run.attempt, "frozen_selection_sha256": sel["sha256"], "final_experiment_config_sha256": fin_sha,
           "plan_sha256": ctx.run.plan_sha, "code_sha256": ctx.run.code_sha, "feature_sets_sha256": ctx.fs().sha256,
           "confirmatory_finalists": [{"config": c, "role": role_of(c, sel)} for c in sel["shortlist"]],
           "descriptive_configs": list(sel["descriptive_configs"]), "n_configs_to_score": len(sel["descriptive_configs"]),
           "rule": ("one-shot evaluation of pre-frozen models: every configuration was fitted on TRAIN and frozen in SELECTION_FROZEN.json before "
                    "this file was written; no model, feature set, hyper-parameter or threshold is changed after VALIDATION is read; a future "
                    "improvement suggested by VALIDATION is recorded as a future experiment, never tuned against these rows")}
    D.write_json(path, rec)
    D.mark_readonly(path)
    return rec


def s13_validation(ctx: Ctx) -> None:
    from falls_ml.phase2.stages_data import verify_protected

    st = ctx.run.stage("13", "validation", ("12",))
    if st.complete_record():
        return
    sel = selection(ctx)
    import_xgboost()
    cfgs = configurations(ctx)
    frozen_cfgs = sorted(sel["descriptive_configs"])
    now_cfgs = sorted(c for c in cfgs if not cfgs[c]["failed"])
    if frozen_cfgs != now_cfgs:
        raise Phase2Stop("VALIDATION_CONFIG_MISMATCH", "the configurations to score differ from the list frozen in SELECTION_FROZEN.json",
                         [f"frozen only: {sorted(set(frozen_cfgs) - set(now_cfgs))}", f"now only: {sorted(set(now_cfgs) - set(frozen_cfgs))}"])
    opened = open_validation(ctx, sel)
    registry = ctx.run.out / "validation_evaluation_registry.jsonl"
    records, _ = D.read_jsonl(registry)
    D.append_jsonl(registry, {"ts": utc_now(), "attempt": ctx.run.attempt, "selection_sha256": sel["sha256"], "recommended": sel["recommended"],
                              "shortlist": sel["shortlist"], "validation_opened_at": opened["opened_at"],
                              "purpose": ("one confirmatory comparison of the frozen shortlist + descriptive scoring" if not records
                                          else "re-read after an interruption (same frozen selection; nothing re-selected)")})
    val = ctx.validation_frame(sel)
    caps = tuple(float(c) for c in ctx.cfg["metrics"]["capacities"])
    principal = float(ctx.cfg["selection"]["principal_capacity"])
    vc = ctx.cfg["validation_confirmation"]
    g = ctx.cfg["gates"]
    nb = int(ctx.cfg["metrics"]["bootstrap_n"])

    mgmt = management_configs(ctx, sel)
    sc = ctx.cfg["metrics"].get("smooth_calibration") or {"knots": 4, "grid_points": 50}

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        y = val["y"].to_numpy(dtype=int)
        preds = {}
        for name in frozen_cfgs:
            if not config_final_result(ctx, cfgs, name).get("failed"):
                preds[name] = _predict_final(ctx, cfgs, name, val)
        D.write_npz(tmp / "validation_predictions.npz", **{n.replace(":", "__"): p for n, p in preds.items()})
        bench, sref = preds[BENCHMARK], preds[SAFE_REF]
        rows, caprows, thr, cal, sub, calb, smooth = [], [], [], [], [], [], []
        masks = subgroup_masks(val, ctx.cat)
        for name, p in preds.items():
            role = role_of(name, sel)
            b = bundle(y, p, capacities=caps, principal=principal)
            row = {"config": name, "role": role, "family": cfgs[name]["family"], "feature_set": cfgs[name]["set"], "category": cfgs[name]["category"],
                   "in_management_waterfall": name in mgmt, **{f"val_{k}": v for k, v in b.items()}}
            if name != BENCHMARK:   # same resampled patients for both models; top-ceil(q x n) recomputed per replicate (Astra F-12)
                row.update({f"val_{k}": v for k, v in paired_bootstrap(y, bench, p, n_boot=nb, seed=seed, principal=principal, capacities=caps).items()
                            if k.startswith("delta")})
            if cfgs[name]["category"] == SAFE_DISCOVERY and name != SAFE_REF:
                row.update({f"val_{k}_vs_safe_base": v for k, v in paired_bootstrap(y, sref, p, n_boot=nb, seed=seed, principal=principal).items()
                            if k.startswith("delta")})
            rows.append(row)
            caprows.append(capacity_table(y, p, caps).assign(config=name, role=role, category=cfgs[name]["category"]))
            if role != "descriptive" or name in mgmt:   # calibration uncertainty + a smooth curve (Astra F-11) for the shortlist and the waterfall
                cb = calibration_bootstrap(y, p, n_boot=nb, seed=seed)
                calb.append({"config": name, "role": role, "category": cfgs[name]["category"], **{k: v for k, v in cb.items() if k != "n_boot"},
                             "n_boot": nb})
                smooth.append(smooth_calibration(y, p, n_knots=int(sc["knots"]), n_grid=int(sc["grid_points"])).assign(config=name, role=role))
            if role != "descriptive":
                thr.append(threshold_table(y, p).assign(config=name, role=role))
                cal.append(calibration_table(y, p, int(ctx.cfg["metrics"]["calibration_groups"])).assign(config=name, role=role))
                for dim, groups in masks.items():
                    for lvl, m in groups.items():
                        ev = int(y[m].sum())
                        ok = ev >= int(ctx.cfg["metrics"]["subgroup_min_events"]) and int(m.sum()) - ev >= int(ctx.cfg["metrics"]["subgroup_min_events"])
                        bb = bundle(y[m], p[m]) if ok else {}
                        sub.append({"config": name, "role": role, "subgroup": dim, "level": lvl, "n": int(m.sum()), "events": ev,
                                    "ap": bb.get("ap"), "auroc": bb.get("auroc"), "citl": bb.get("citl"), "cal_slope": bb.get("cal_slope"),
                                    "capture@principal": float(y[m & top_mask(p, principal)].sum() / ev) if ok and ev else None,
                                    "reported": ok})
        t = pd.DataFrame(rows)
        crow = []
        for c in sel.get("attribution_contrasts") or []:
            if c["new"] in preds and c["reference"] in preds:
                d = paired_bootstrap(y, preds[c["reference"]], preds[c["new"]], n_boot=nb, seed=seed, principal=principal, capacities=caps)
                crow.append({**c, **{f"val_{k}": v for k, v in d.items() if k.startswith("delta")}})
        D.write_csv(tmp / "ATTRIBUTION_CONTRASTS_VALIDATION.csv", pd.DataFrame(crow))
        D.write_csv(tmp / "CALIBRATION_BOOTSTRAP.csv", pd.DataFrame(calb))
        D.write_csv(tmp / "CALIBRATION_SMOOTH.csv", pd.concat(smooth, ignore_index=True) if smooth else pd.DataFrame())
        D.write_csv(tmp / "VALIDATION_MODEL_COMPARISON.csv", t)
        D.write_csv(tmp / "OPERATIONAL_CAPACITY.csv", pd.concat(caprows, ignore_index=True))
        D.write_csv(tmp / "THRESHOLD_RESULTS.csv", pd.concat(thr, ignore_index=True))
        D.write_csv(tmp / "CALIBRATION_CURVES.csv", pd.concat(cal, ignore_index=True))
        D.write_csv(tmp / "SUBGROUP_SUMMARY.csv", pd.DataFrame(sub))
        ti = t.set_index("config")
        # ---- confirmatory decision (pre-declared): the recommended SAFE model vs SAFE_BASE
        rec = sel["recommended"]
        status, detail, bclass = NO_INCREMENTAL_MODEL_SELECTED, "no SAFE configuration qualified on TRAIN; nothing to confirm", None
        if rec != SAFE_REF and rec in preds:
            r = ti.loc[rec]
            ok = _meets(r, d_ap="val_delta_ap_vs_safe_base", d_cap="val_delta_capture_principal_vs_safe_base", ll_ratio=float(ti.at[SAFE_REF, "val_logloss"]),
                        sel=vc, slope_range=vc["calibration_slope_range"], max_citl=vc["max_abs_citl"], prefix="val")
            status = "CONFIRMED_PROMISING" if ok else "NOT_CONFIRMED"
            detail = (f"exploratory screen on reused VALIDATION, vs SAFE_BASE: dAP {r['val_delta_ap_vs_safe_base']:+.4f} [{r['val_delta_ap_ci_low_vs_safe_base']:+.4f}, "
                      f"{r['val_delta_ap_ci_high_vs_safe_base']:+.4f}], d capture@{principal:.0%} {100 * r['val_delta_capture_principal_vs_safe_base']:+.1f} pp, "
                      f"log loss ratio {r['val_logloss'] / ti.at[SAFE_REF, 'val_logloss']:.3f}, slope {r['val_cal_slope']:.2f}, CITL {r['val_citl']:+.2f}")
        if rec in preds and rec != BENCHMARK:
            r = ti.loc[rec]
            sup = _meets(r, d_ap="val_delta_ap", d_cap="val_delta_capture_principal", ll_ratio=float(ti.at[BENCHMARK, "val_logloss"]), sel=vc,
                         slope_range=vc["calibration_slope_range"], max_citl=vc["max_abs_citl"], prefix="val")
            ni = vc["benchmark_noninferiority"]
            bclass = benchmark_class(float(r["val_delta_ap"]), 100.0 * float(r["val_delta_capture_principal"]), superior=sup,
                                     max_ap_loss=ni["max_ap_loss"], max_cap_loss_pp=ni["max_capture_loss_pp"])
        rb = ti.loc[BENCHMARK]
        others = t[t["category"].isin([SAFE_DISCOVERY, EXPLORATORY])]
        jumps = others[(others["val_auroc"] - rb["val_auroc"] > float(g["max_delta_auroc"])) | (others["val_ap"] / rb["val_ap"] > float(g["max_ap_ratio"]))]["config"].tolist()
        slo, shi = g["calibration_slope_range"]
        badcal = []
        for c in sel["shortlist"]:
            if c in preds:
                r = ti.loc[c]
                if not (float(slo) <= r["val_cal_slope"] <= float(shi) and abs(r["val_citl"]) <= float(g["max_abs_citl"])):
                    badcal.append(f"{c}: slope {r['val_cal_slope']:.2f}, CITL {r['val_citl']:+.2f}")
        return {"confirmation": status, "detail": detail, "recommended": rec, "benchmark_class_validation": bclass,
                "benchmark_class_oof": sel.get("benchmark_class_oof"), "implausible_gain": jumps, "implausible_calibration": badcal,
                "n_configs_scored": len(preds), "validation_rows": int(len(y)), "validation_events": int(y.sum())}

    res = st.item("validation", fn)
    ctx.run.gate("IMPLAUSIBLE_GAIN_VALIDATION", bool(res["implausible_gain"]), "validation gain vs the BASELINE_15 benchmark above the plausibility limits",
                 [f"config: {c}" for c in res["implausible_gain"]])
    ctx.run.gate("IMPLAUSIBLE_CALIBRATION_VALIDATION", bool(res["implausible_calibration"]), "implausible validation calibration in the shortlist",
                 res["implausible_calibration"])
    outs = [D.write_bytes(ctx.run.artifacts / n, (st.path("validation") / n).read_bytes())
            for n in ("VALIDATION_MODEL_COMPARISON.csv", "OPERATIONAL_CAPACITY.csv", "THRESHOLD_RESULTS.csv", "CALIBRATION_CURVES.csv", "SUBGROUP_SUMMARY.csv",
                      "ATTRIBUTION_CONTRASTS_VALIDATION.csv", "CALIBRATION_BOOTSTRAP.csv", "CALIBRATION_SMOOTH.csv")]
    outs.append(D.write_json(ctx.run.artifacts / "VALIDATION_CONFIRMATION.json", {k: res[k] for k in (
        "confirmation", "detail", "recommended", "benchmark_class_oof", "benchmark_class_validation", "validation_rows", "validation_events")}))
    st.finalize(outs, {"confirmation": res["confirmation"], "benchmark_class_validation": res["benchmark_class_validation"]})
    verify_protected(ctx)
