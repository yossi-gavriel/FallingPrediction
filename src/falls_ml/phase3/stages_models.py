"""Phase 3 stages S07-S14 (run only after the feasibility decision GO): nested grouped CV of LASSO / elastic net / XGBoost inside TRAIN, fitted on
rows whose used features are all KNOWN and evaluated on EVERY row with adverse / favourable bounds; the pre-declared selection rule on the ADVERSE
bound, frozen before VALIDATION is read; stability; ablation; explainability; the one-shot descriptive check on the reused VALIDATION partition.

The unchanged Phase 2 components do the fitting (``_fit_with_retry`` = LASSO convergence-retry policy v1, ``fit_linear``, ``xgb_inner_cv``,
``xgb_fit``, the persistent Optuna study) and the metrics (``bundle``, ``paired_bootstrap``, ``capacity_table``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.errors import DegenerateFitError
from falls_ml.models.lasso_cv import LassoError
from falls_ml.phase2 import durable as D
from falls_ml.phase2.context import import_xgboost
from falls_ml.phase2.design import tree_feature_of, tree_matrix
from falls_ml.phase2.evaluate import bundle, calibration_table, capacity_table, top_mask
from falls_ml.phase2.fitting import ENET, LASSO, xgb_fit, xgb_gain_importance, xgb_inner_cv, xgb_params
from falls_ml.phase2.stages_data import verify_protected
from falls_ml.phase2.stages_models import _fit_with_retry, inner_fold_seed
from falls_ml.phase2.state import Phase2Stop, Stage
from falls_ml.phase2.xgb_tuning import TrialStore, best_trial, run_study
from falls_ml.phase3.bounds import Intervals, assign, bounded_bundle, bounded_delta, linear_intervals, plan_unknown, xgb_intervals
from falls_ml.phase3.config import P3_ALL, P3_BASE
from falls_ml.phase3.context import P3Ctx
from falls_ml.phase3.stages_data import EXPLORATORY, HIST_SET, HISTORICAL, PRIMARY

REF = f"{LASSO}:{P3_BASE}"
HIST = f"{LASSO}:{HIST_SET}"
KIND_ORDER = {"main": 0, "historical": 1, "domain_add": 2, "waterfall": 3, "proxy_ablation": 4, "sensitivity": 5, "exploratory": 6, "loo": 7}
CANDIDATE_KINDS = ("main", "domain_add", "waterfall")
FIT_ERRORS = (DegenerateFitError, LassoError)
INCREMENTAL, INCONCLUSIVE, NONE = "INCREMENTAL_MODEL_SELECTED", "INCONCLUSIVE_OVERWRITTEN_HISTORY", "NO_INCREMENTAL_MODEL_SELECTED"


def sets_in_order(ctx: P3Ctx, kinds: tuple[str, ...]) -> list[str]:
    fs = ctx.fs()
    names = [s for s in fs.fitted() if fs.sets[s]["kind"] in kinds]
    return sorted(names, key=lambda s: (s != P3_BASE, s != P3_ALL, KIND_ORDER[fs.sets[s]["kind"]], s))


def feats_of(s: dict[str, Any]) -> list[str]:
    return [*s["features"], *s["baseline"]]


def tree_col_of(features: list[str]) -> dict[str, str]:
    return {f: ("sex_female" if f == "sex" else f) for f in features}


# ============================================================================ items (fit on KNOWN rows, evaluate every held-out row)
def _rows(ctx: P3Ctx, fold: str) -> tuple[np.ndarray, np.ndarray | None]:
    fm = ctx.fit_mask_train()
    if fold == "final":
        return fm.copy(), None
    te = ctx.outer_folds() == int(fold.replace("outer", ""))
    return ~te & fm, te


def _plan(ctx: P3Ctx, feats: list[str], te: np.ndarray) -> Any:
    return plan_unknown(feats, ctx.unknown_train().loc[te].reset_index(drop=True), ctx.ub_train().loc[te].reset_index(drop=True), ctx.feature_source())


def linear_item(ctx: P3Ctx, family: str, setname: str, fold: str, *, new: list[str] | None = None, base: list[str] | None = None,
                fixed: dict[str, float] | None = None, config: str | None = None) -> Any:
    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        s = dict(ctx.fs().sets[setname])
        if base is not None:
            s["baseline"] = list(base)
        feats = [*(s["features"] if new is None else new), *s["baseline"]]
        tr, y = ctx.train_frame(), ctx.y_train()
        fit_rows, te = _rows(ctx, fold)
        if np.asarray(ctx.unknown_train().loc[fit_rows, [f for f in feats if f in ctx.unknown_train().columns]].to_numpy()).any():
            raise Phase2Stop("UNKNOWN_ROW_IN_FIT", f"{setname}: a fitting row carries an UNKNOWN (overwritten) value")
        try:
            fit, retry = _fit_with_retry(ctx, family, tr.loc[fit_rows], y[fit_rows], s, inner_fold_seed(ctx, "linear", fold if fold != "final" else "final"),
                                         fixed=fixed, new=new)
        except FIT_ERRORS as exc:
            return {"config": config or f"{family}:{setname}", "fold": fold, "failed": True, "error": f"{type(exc).__name__}: {exc}"[:500]}
        out = {"config": config or f"{family}:{setname}", "fold": fold, "failed": False, "n_fit_rows": int(fit_rows.sum()), **fit.hyper(),
               "n_design_columns": len(fit.design.columns_), "n_selected": len(fit.selected_features()), "selected": fit.selected_features(),
               "retry": retry["retry"], "cv_minimum_identified": fit.diagnostics.get("cv_minimum_identified")}
        if te is not None:
            ev = tr.loc[te].reset_index(drop=True)
            iv = linear_intervals(fit, ev, tr.loc[fit_rows].reset_index(drop=True), _plan(ctx, feats, te))
            D.write_npz(tmp / "oof.npz", rows=np.flatnonzero(te), lo=iv.lo, hi=iv.hi)
            out["n_unknown_heldout"] = iv.info.get("n_unknown_rows", 0)
        else:
            D.write_pickle(tmp / "model.pkl", fit)
            D.write_csv(tmp / "coefficients.csv", fit.coefficients())
            t = fit.coefficients()
            t = t[~t["feature"].str.startswith("na__")]
            out["importance"] = {str(k): float(v) for k, v in t.assign(a=t["standardized_coefficient"].abs()).groupby("feature")["a"].sum().items()}
        return out
    return fn


def nested_linear(ctx: P3Ctx, st: Stage, family: str, sets: list[str]) -> None:
    K = int(ctx.cfg["cv"]["outer_folds"])
    for s in sets:
        for k in range(K):
            st.item(f"{family}__{s}__outer{k}", linear_item(ctx, family, s, f"outer{k}"))
        st.item(f"{family}__{s}__final", linear_item(ctx, family, s, "final"))


def oof_intervals(ctx: P3Ctx, st: Stage, items: list[str]) -> Intervals | None:
    n = len(ctx.y_train())
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    for it in items:
        if st.result(it).get("failed"):
            return None
        z = np.load(st.path(it) / "oof.npz")
        lo[z["rows"]], hi[z["rows"]] = z["lo"], z["hi"]
    if np.isnan(lo).any():
        raise Phase2Stop("OOF_INCOMPLETE", f"{items[0]}: out-of-fold predictions do not cover every TRAIN row")
    return Intervals(lo=lo, hi=hi)


def failure_gate(ctx: P3Ctx, st: Stage) -> dict[str, Any]:
    recs = [r["result"] for r in st.records().values() if isinstance(r.get("result"), dict) and "failed" in r["result"]]
    n_fail = sum(1 for r in recs if r["failed"])
    if recs and n_fail / len(recs) > float(ctx.cfg["gates"]["max_fit_failure_fraction"]):
        raise Phase2Stop("FIT_FAILURES", f"{n_fail} of {len(recs)} fits failed in {st.label}", [r.get("error", "") for r in recs if r["failed"]][:10])
    return {"n_fits": len(recs), "n_failed": n_fail}


def fits_table(st: Stage) -> pd.DataFrame:
    rows = []
    for rec in st.records().values():
        r = rec["result"]
        if isinstance(r, dict) and "config" in r:
            rows.append({"item": rec["item"], "config": r.get("config"), "fold": r.get("fold"), "failed": r.get("failed"), "error": r.get("error"),
                         "n_fit_rows": r.get("n_fit_rows"), "lambda": r.get("lambda"), "l1_ratio": r.get("l1_ratio"), "n_selected": r.get("n_selected"),
                         "n_unknown_heldout": r.get("n_unknown_heldout"), "retry": r.get("retry"), "n_trees": r.get("n_trees"),
                         "elapsed_s": rec.get("elapsed_s"), "attempt": rec.get("attempt")})
    return pd.DataFrame(rows)


# ============================================================================ S07 LASSO, S08 elastic net
def s07_lasso(ctx: P3Ctx) -> None:
    st = ctx.run.stage("07", "lasso", ("06",))
    if st.complete_record():
        return
    nested_linear(ctx, st, LASSO, sets_in_order(ctx, ("main", "historical", "domain_add", "waterfall", "proxy_ablation", "sensitivity", "exploratory")))
    summ = failure_gate(ctx, st)
    st.finalize([D.write_csv(ctx.run.artifacts / "FITS_LASSO.csv", fits_table(st))], summ)
    verify_protected(ctx)


def _adverse_ap(ctx: P3Ctx, iv: Intervals | None) -> float | None:
    from falls_ml.evaluation.metrics import pr_auc

    return None if iv is None else float(pr_auc(ctx.y_train(), assign(ctx.y_train(), iv, "adverse")))


def s08_enet(ctx: P3Ctx) -> None:
    st = ctx.run.stage("08", "enet", ("07",))
    if st.complete_record():
        return
    K = int(ctx.cfg["cv"]["outer_folds"])
    st7 = ctx.run.stage("07", "lasso")

    def plan(tmp: Path, seed: int) -> dict[str, Any]:
        cands = [s for s in sets_in_order(ctx, CANDIDATE_KINDS) if ctx.fs().sets[s]["category"] == PRIMARY and ctx.fs().sets[s]["n_new_features"] > 0]
        scores = {s: _adverse_ap(ctx, oof_intervals(ctx, st7, [f"LASSO__{s}__outer{k}" for k in range(K)])) for s in cands}
        scores = {s: v for s, v in scores.items() if v is not None}
        out = [P3_ALL]
        if scores:
            best = max(sorted(scores), key=lambda s: scores[s])
            if best not in out:
                out.append(best)
        return {"sets": out[: int(ctx.cfg["enet"]["max_sets"])], "rule": "P3_ALL_RECOVERED + best primary LASSO set by adverse outer-OOF AP"}

    sets = st.item("ENET__plan", plan)["sets"]
    nested_linear(ctx, st, ENET, sets)
    summ = failure_gate(ctx, st)
    st.finalize([D.write_csv(ctx.run.artifacts / "FITS_ENET.csv", fits_table(st))], {**summ, "sets": sets})
    verify_protected(ctx)


# ============================================================================ S09 XGBoost
def _tree(ctx: P3Ctx, frame: pd.DataFrame, setname: str) -> pd.DataFrame:
    s = ctx.fs().sets[setname]
    return tree_matrix(frame, s["features"], s["baseline"])


def _xgb_eval(ctx: P3Ctx, bst: Any, setname: str, fit_rows: np.ndarray, te: np.ndarray) -> Intervals:
    s = ctx.fs().sets[setname]
    tr = ctx.train_frame()
    Xe = _tree(ctx, tr.loc[te].reset_index(drop=True), setname)
    Xf = _tree(ctx, tr.loc[fit_rows].reset_index(drop=True), setname)
    return xgb_intervals(bst, Xe, Xf, _plan(ctx, feats_of(s), te), tree_col_of(feats_of(s)))


def xgb_item(ctx: P3Ctx, setname: str, fold: str, config: str, *, params: dict[str, Any] | None = None, n_trees: int | None = None) -> Any:
    x = ctx.cfg["xgb"]

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        fit_rows, te = _rows(ctx, fold)
        tr, y = ctx.train_frame(), ctx.y_train()
        X = _tree(ctx, tr.loc[fit_rows], setname)
        p = xgb_params(ctx.cfg, params or {k: x["stage0"][k] for k in ("max_depth", "min_child_weight", "reg_lambda")}, seed=seed, nthread=ctx.threads["xgboost"])
        out: dict[str, Any] = {"config": config, "fold": fold, "failed": False, "params": params or {k: x["stage0"][k] for k in ("max_depth", "min_child_weight", "reg_lambda")},
                               "n_fit_rows": int(fit_rows.sum())}
        if n_trees is None:
            cv = xgb_inner_cv(X, y[fit_rows], p, n_folds=int(ctx.cfg["cv"]["inner_folds_xgb"]), seed=inner_fold_seed(ctx, "xgb", fold),
                              n_max=int(x["n_estimators_max"]), early_stopping=int(x["early_stopping_rounds"]))
            nt = cv["n_trees"]
            out.update({"inner_cv_logloss": cv["mean_logloss"], "inner_cv_ap": cv["mean_ap"]})
        else:
            nt = int(n_trees)
        bst = xgb_fit(X, y[fit_rows], p, nt)
        D.write_bytes(tmp / "booster.json", bytes(bst.save_raw("json")))
        out.update({"n_trees": int(nt), "importance": xgb_gain_importance(bst), "columns": list(X.columns)})
        if te is not None:
            iv = _xgb_eval(ctx, bst, setname, fit_rows, te)
            D.write_npz(tmp / "oof.npz", rows=np.flatnonzero(te), lo=iv.lo, hi=iv.hi)
            out["n_unknown_heldout"] = iv.info.get("n_unknown_rows", 0)
        return out
    return fn


def s09_xgb(ctx: P3Ctx) -> None:
    import_xgboost()
    st = ctx.run.stage("09", "xgb", ("08",))
    if st.complete_record():
        return
    x = ctx.cfg["xgb"]
    K = int(ctx.cfg["cv"]["outer_folds"])
    scopes = [f"outer{k}" for k in range(K)] + ["final"]
    for s in x["stage0"]["sets"]:
        for fold in scopes:
            st.item(f"XGB0__{s}__{fold}", xgb_item(ctx, s, fold, f"XGB_DEFAULT:{s}"))
    store = TrialStore(ctx.run)
    tset = x["stage1"]["set"]

    def objective(fold: str) -> Any:
        def obj(params: dict[str, Any], seed: int, tmp: Path) -> dict[str, Any]:
            fit_rows, _ = _rows(ctx, fold)
            X = _tree(ctx, ctx.train_frame().loc[fit_rows], tset)
            cv = xgb_inner_cv(X, ctx.y_train()[fit_rows], xgb_params(ctx.cfg, params, seed=seed, nthread=ctx.threads["xgboost"]),
                              n_folds=int(ctx.cfg["cv"]["inner_folds_xgb"]), seed=inner_fold_seed(ctx, "xgb", fold), n_max=int(x["n_estimators_max"]),
                              early_stopping=int(x["early_stopping_rounds"]))
            return {"value": cv["mean_logloss"], "ap": cv["mean_ap"], "n_trees": cv["n_trees"], "best_iterations": [f["best_iteration"] for f in cv["folds"]]}
        return obj

    meta = {"feature_set": tset, "feature_sets_sha256": ctx.fs().sha256, "plan_sha256": ctx.run.plan_sha}
    for fold in scopes:
        res = run_study(st, store, name=f"XGB1_{fold}", space=x["stage1"]["space"], n_trials=int(x["stage1"]["n_trials"]),
                        n_startup=int(x["stage1"]["n_startup_trials"]), seed_base=ctx.run.seed, objective=objective(fold), meta=meta)
        b = best_trial(res)
        st.item(f"XGB1__{fold}__refit", xgb_item(ctx, tset, fold, f"XGB_TUNED:{tset}", params=b["params"], n_trees=b["n_trees"]))
    trials, _ = D.read_jsonl(store.ledger)
    outs = [D.write_csv(ctx.run.artifacts / "XGB_TRIALS.csv", pd.json_normalize(trials) if trials else pd.DataFrame()),
            D.write_csv(ctx.run.artifacts / "FITS_XGB.csv", fits_table(st))]
    st.finalize(outs, {"n_trials_ledger": len(trials)})
    verify_protected(ctx)


# ============================================================================ configurations
def configurations(ctx: P3Ctx) -> dict[str, dict[str, Any]]:
    K = int(ctx.cfg["cv"]["outer_folds"])
    out: dict[str, dict[str, Any]] = {}
    for sid, name in (("07", "lasso"), ("08", "enet")):
        st = ctx.run.stage(sid, name)
        for rec in st.records().values():
            if rec["item"].endswith("__final"):
                fam, s, _ = rec["item"].split("__")
                out[f"{fam}:{s}"] = {"stage": (sid, name), "items": [f"{fam}__{s}__outer{k}" for k in range(K)], "final": rec["item"], "family": fam,
                                     "set": s, "failed": bool(rec["result"].get("failed")), "category": ctx.fs().sets[s]["category"],
                                     "kind": ctx.fs().sets[s]["kind"]}
    x = ctx.cfg["xgb"]
    for s in x["stage0"]["sets"]:
        out[f"XGB_DEFAULT:{s}"] = {"stage": ("09", "xgb"), "items": [f"XGB0__{s}__outer{k}" for k in range(K)], "final": f"XGB0__{s}__final",
                                   "family": "XGB_DEFAULT", "set": s, "failed": False, "category": ctx.fs().sets[s]["category"], "kind": ctx.fs().sets[s]["kind"]}
    t = x["stage1"]["set"]
    out[f"XGB_TUNED:{t}"] = {"stage": ("09", "xgb"), "items": [f"XGB1__outer{k}__refit" for k in range(K)], "final": "XGB1__final__refit",
                             "family": "XGB_TUNED", "set": t, "failed": False, "category": ctx.fs().sets[t]["category"], "kind": ctx.fs().sets[t]["kind"]}
    return out


def config_oof(ctx: P3Ctx, cfgs: dict[str, dict[str, Any]], name: str) -> Intervals | None:
    c = cfgs[name]
    return None if c["failed"] else oof_intervals(ctx, ctx.run.stage(*c["stage"]), c["items"])


def final_result(ctx: P3Ctx, cfgs: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    c = cfgs[name]
    return ctx.run.stage(*c["stage"]).result(c["final"])


def complexity(ctx: P3Ctx, cfgs: dict[str, dict[str, Any]], name: str) -> tuple[int, int]:
    r = final_result(ctx, cfgs, name)
    if cfgs[name]["family"] in (LASSO, ENET):
        return 0, int(r.get("n_selected") or 0)
    return 1, len([k for k, v in (r.get("importance") or {}).items() if v > 0])


# ============================================================================ S10 selection (ADVERSE bound; frozen before VALIDATION)
def _qualifies(row: dict[str, Any], mode: str, sel: dict[str, Any]) -> bool:
    """The pre-declared rule under one end of the bounds; an undefined value never passes."""
    lo, hi = sel["calibration_slope_range"]
    keys = [f"{mode}_delta_ap", f"{mode}_delta_capture_principal", f"{mode}_logloss_ratio", "adverse_cal_slope", "favourable_cal_slope",
            "adverse_citl", "favourable_citl"]
    vals = [row.get(k) for k in keys]
    if not all(v is not None and np.isfinite(float(v)) for v in vals):
        return False
    return bool(row[f"{mode}_delta_ap"] >= float(sel["min_delta_ap"]) and 100.0 * row[f"{mode}_delta_capture_principal"] >= float(sel["min_delta_capture_principal_pp"])
                and row[f"{mode}_logloss_ratio"] <= float(sel["max_logloss_ratio"])
                and all(float(lo) <= row[f"{m}_cal_slope"] <= float(hi) and abs(row[f"{m}_citl"]) <= float(sel["max_abs_citl"]) for m in ("adverse", "favourable")))


def compare_row(y: np.ndarray, iv: Intervals, ref: Intervals, caps: tuple[float, ...], principal: float) -> dict[str, Any]:
    """Bounded metrics of one configuration and its point differences vs the reference (ADVERSE: new adverse - reference favourable)."""
    b = bounded_bundle(y, iv, capacities=caps, principal=principal)
    r = bounded_bundle(y, ref, capacities=caps, principal=principal)
    cap = f"capture@{principal:g}"
    out = dict(b)
    for mode, other in (("adverse", "favourable"), ("favourable", "adverse")):
        out[f"{mode}_delta_ap"] = b[f"{mode}_ap"] - r[f"{other}_ap"]
        out[f"{mode}_delta_auroc"] = b[f"{mode}_auroc"] - r[f"{other}_auroc"]
        out[f"{mode}_delta_capture_principal"] = b[f"{mode}_{cap}"] - r[f"{other}_{cap}"]
        out[f"{mode}_logloss_ratio"] = b[f"{mode}_logloss"] / r[f"{other}_logloss"]
        n_ev = int(np.sum(y))
        out[f"{mode}_additional_falls_principal"] = round((b[f"{mode}_{cap}"] - r[f"{other}_{cap}"]) * n_ev)
    return out


def s10_select(ctx: P3Ctx) -> None:
    st = ctx.run.stage("10", "select", ("09",))
    if st.complete_record():
        return
    sel = ctx.cfg["selection"]
    g = ctx.cfg["gates"]
    principal = float(sel["principal_capacity"])
    caps = tuple(float(c) for c in ctx.cfg["metrics"]["capacities"])
    y = ctx.y_train()
    nb = int(ctx.cfg["metrics"]["bootstrap_n"])
    bridge = ctx.bridge_train()

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        cfgs = configurations(ctx)
        # the plausibility comparator: the historical 15-predictor refit; when all 15 are eligible it IS P3_BASE (no separate set is fitted)
        ref = config_oof(ctx, cfgs, REF)
        hist = config_oof(ctx, cfgs, HIST if HIST in cfgs else REF)
        if ref is None:
            raise Phase2Stop("REFERENCE_FAILED", "LASSO:P3_BASE could not be fitted in every outer fold")
        rows, ivs = [], {}
        for name in sorted(cfgs):
            iv = config_oof(ctx, cfgs, name)
            c = cfgs[name]
            row: dict[str, Any] = {"config": name, "family": c["family"], "feature_set": c["set"], "category": c["category"], "set_kind": c["kind"],
                                   "n_new_features_offered": ctx.fs().sets[c["set"]]["n_new_features"], "failed": iv is None}
            if iv is not None:
                ivs[name] = iv
                row.update(compare_row(y, iv, ref, caps, principal))
                if hist is not None:
                    hb = bounded_bundle(y, hist, capacities=caps, principal=principal)
                    row["favourable_delta_auroc_vs_hist"] = row["favourable_auroc"] - hb["adverse_auroc"]
                    row["favourable_ap_ratio_vs_hist"] = row["favourable_ap"] / hb["adverse_ap"] if hb["adverse_ap"] else np.nan
                bb = bounded_bundle(y[bridge], Intervals(iv.lo[bridge], iv.hi[bridge]), capacities=caps, principal=principal)
                row.update({f"bridge_d00clean_{k}": v for k, v in bb.items() if k.split("_", 1)[1] in ("ap", "auroc", f"capture@{principal:g}", "n", "events")})
                row["complexity_rank"], row["n_features_used"] = complexity(ctx, cfgs, name)
            rows.append(row)
        t = pd.DataFrame(rows)
        ok = ~t["failed"].astype(bool)
        cand = t[ok & (t["category"] == PRIMARY) & t["set_kind"].isin(list(CANDIDATE_KINDS)) & (t["config"] != REF)]
        adv = cand[cand.apply(lambda r: _qualifies(r.to_dict(), "adverse", sel), axis=1)] if len(cand) else cand
        fav = cand[cand.apply(lambda r: _qualifies(r.to_dict(), "favourable", sel), axis=1)] if len(cand) else cand
        t["qualifies_adverse"] = t["config"].isin(adv["config"])
        t["qualifies_favourable_only"] = t["config"].isin(fav["config"]) & ~t["qualifies_adverse"]
        recommended, outcome = REF, NONE
        why = "no recovered configuration met every pre-declared criterion over P3_BASE even under the favourable bound; P3_BASE is reported"
        if len(adv):
            best = float(adv["adverse_ap"].max())
            near = adv[adv["adverse_ap"] >= best - float(sel["simplicity_margin_ap"])].sort_values(
                ["complexity_rank", "n_features_used", "adverse_ap", "config"], ascending=[True, True, False, True])
            recommended, outcome = str(near.iloc[0]["config"]), INCREMENTAL
            why = f"highest ADVERSE-bound OOF AP among {len(adv)} qualifying configuration(s) ({best:.4f}); simplest within {sel['simplicity_margin_ap']}: {recommended}"
        elif len(fav):
            outcome = INCONCLUSIVE
            why = (f"{len(fav)} configuration(s) meet the rule only under the FAVOURABLE bound: the overwritten pre-index history prevents a conclusion "
                   "(the DWH correction would resolve it)")
        # shortlist: reference, recommended, the historical refit (context), the best candidate of each other family (<= max_challengers)
        shortlist = [REF] + ([recommended] if recommended != REF else [])
        for fam in ("XGB_TUNED", "ENET", "LASSO", "XGB_DEFAULT"):
            if len(shortlist) - 1 >= int(sel["max_challengers"]):
                break
            if any(cfgs[c]["family"] == fam or (fam.startswith("XGB") and cfgs[c]["family"].startswith("XGB")) for c in shortlist[1:]):
                continue
            pool = cand[cand["family"] == fam]
            if len(pool):
                shortlist.append(str(pool.sort_values(["adverse_ap", "config"], ascending=[False, True]).iloc[0]["config"]))
        if HIST in cfgs and not cfgs[HIST]["failed"]:
            shortlist.append(HIST)
        # paired bootstrap intervals vs P3_BASE: shortlist + main + one-domain additions (the domain question)
        boot = [c for c in dict.fromkeys([*shortlist, *t.loc[ok & t["set_kind"].isin(["main", "domain_add"]) & (t["category"] == PRIMARY), "config"]]) if c != REF and c in ivs]
        brows = []
        for c in boot:
            d = bounded_delta(y, ref, ivs[c], n_boot=nb, seed=seed, principal=principal, capacities=caps)
            brows.append({"config": c, "reference": REF, **d})
        D.write_csv(tmp / "OOF_BOOTSTRAP_VS_P3_BASE.csv", pd.DataFrame(brows))
        prow = []
        for fam in ("LASSO", "XGB_DEFAULT"):
            full_c, abl_c = f"{fam}:{P3_ALL}", f"{fam}:P3_ALL_NO_FALL_RECENCY"
            if full_c in ivs and abl_c in ivs:
                d = bounded_delta(y, ivs[abl_c], ivs[full_c], n_boot=nb, seed=seed, principal=principal, capacities=caps)
                prow.append({"family": fam, "full": full_c, "without_fall_recency": abl_c, "same_rows": True,
                             **{f"{k}_full_minus_ablated": v for k, v in d.items()}})
        D.write_csv(tmp / "PROXY_ABLATION_OOF.csv", pd.DataFrame(prow))
        D.write_csv(tmp / "OOF_MODEL_COMPARISON.csv", t)
        # capacity tables (both ends) for the shortlist
        caprows = []
        for c in shortlist:
            if c in ivs:
                for mode in ("adverse", "favourable"):
                    caprows.append(capacity_table(y, assign(y, ivs[c], mode), caps).assign(config=c, bound=mode, partition="train_oof"))
        D.write_csv(tmp / "OOF_OPERATIONAL_CAPACITY.csv", pd.concat(caprows, ignore_index=True) if caprows else pd.DataFrame())
        others = t[ok & t["category"].isin([PRIMARY, EXPLORATORY])]
        jumps = []
        if "favourable_delta_auroc_vs_hist" in others:
            jumps = others[(others["favourable_delta_auroc_vs_hist"] > float(g["max_delta_auroc"])) | (others["favourable_ap_ratio_vs_hist"] > float(g["max_ap_ratio"]))]["config"].tolist()
        rrow = t.set_index("config").loc[recommended]
        return {"recommended": recommended, "selection_outcome": outcome, "reason": why, "shortlist": shortlist, "n_qualifying_adverse": int(len(adv)),
                "qualifying_adverse": adv["config"].tolist(), "qualifying_favourable_only": [c for c in fav["config"] if c not in set(adv["config"])],
                "implausible_gain": jumps, "recommended_slopes": [float(rrow["adverse_cal_slope"]), float(rrow["favourable_cal_slope"])],
                "recommended_citl": [float(rrow["adverse_citl"]), float(rrow["favourable_citl"])], "descriptive_configs": sorted(ivs)}

    res = st.item("selection", fn)
    ctx.run.gate("IMPLAUSIBLE_GAIN_OOF", bool(res["implausible_gain"]),
                 f"favourable-bound gain vs the historical 15-predictor refit above the plausibility limits (dAUROC > {g['max_delta_auroc']} or AP ratio > {g['max_ap_ratio']})",
                 [f"config: {c}" for c in res["implausible_gain"]])
    slo, shi = g["calibration_slope_range"]
    bad = res["recommended"] != REF and not all(float(slo) <= s <= float(shi) for s in res["recommended_slopes"]) or \
        res["recommended"] != REF and not all(abs(c) <= float(g["max_abs_citl"]) for c in res["recommended_citl"])
    ctx.run.gate("IMPLAUSIBLE_CALIBRATION_OOF", bool(bad), "the recommended model's out-of-fold calibration is implausible",
                 [f"slopes {res['recommended_slopes']}, CITL {res['recommended_citl']}"])
    frozen = {"frozen_at": utc_now(), "rule": "planning/PHASE3_DESIGN.md §6: the pre-declared rule on TRAIN outer-OOF under the ADVERSE bound (VALIDATION not read)",
              "selection_outcome": res["selection_outcome"], "recommended": res["recommended"], "reason": res["reason"], "shortlist": res["shortlist"],
              "reference": REF, "qualifying_adverse": res["qualifying_adverse"], "qualifying_favourable_only": res["qualifying_favourable_only"],
              "descriptive_configs": res["descriptive_configs"], "selection_item_files_sha256": st.records()["selection"]["files"],
              "plan_sha256": ctx.run.plan_sha, "final_experiment_config_sha256": ctx.run.plan.get("final_experiment_config_sha256"),
              "feature_sets_sha256": ctx.fs().sha256}
    outs = [D.write_json(ctx.run.out / "SELECTION_FROZEN.json", frozen)]
    outs += [D.write_bytes(ctx.run.artifacts / n, (st.path("selection") / n).read_bytes())
             for n in ("OOF_MODEL_COMPARISON.csv", "OOF_BOOTSTRAP_VS_P3_BASE.csv", "OOF_OPERATIONAL_CAPACITY.csv", "PROXY_ABLATION_OOF.csv")]
    st.finalize(outs, {"recommended": res["recommended"], "selection_outcome": res["selection_outcome"]})


def selection(ctx: P3Ctx) -> dict[str, Any]:
    p = ctx.run.out / "SELECTION_FROZEN.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["sha256"] = D.sha256_file(p)
    return rec


# ============================================================================ S11 stability
def s11_stability(ctx: P3Ctx) -> None:
    st = ctx.run.stage("11", "stability", ("10",))
    if st.complete_record():
        return
    sel = selection(ctx)
    cfgs = configurations(ctx)

    def plan(tmp: Path, seed: int) -> dict[str, Any]:
        sets = []
        for c in [sel["recommended"], f"{LASSO}:{P3_ALL}"]:
            if c in cfgs and cfgs[c]["family"] in (LASSO, ENET) and cfgs[c]["set"] not in sets and cfgs[c]["set"] != P3_BASE:
                sets.append(cfgs[c]["set"])
        return {"sets": sets[: int(ctx.cfg["stability"]["max_sets"])]}

    sets = st.item("STAB__plan", plan)["sets"]
    n_rep = int(ctx.cfg["stability"]["n_replicates"])
    fm = ctx.fit_mask_train()
    tr = ctx.train_frame().loc[fm].reset_index(drop=True)
    y = ctx.y_train()[fm]
    for s in sets:
        for r in range(n_rep):
            def fn(tmp: Path, seed: int, s: str = s) -> dict[str, Any]:
                idx = np.random.default_rng(seed).integers(0, len(y), len(y))
                try:
                    fit, retry = _fit_with_retry(ctx, LASSO, tr.iloc[idx].reset_index(drop=True), y[idx], ctx.fs().sets[s], seed, groups=idx)
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
        if 1 - len(ok) / len(res) > float(ctx.cfg["stability"]["max_failure_fraction"]):
            gate.append(f"{s}: {len(res) - len(ok)} of {len(res)} replicates failed")
        for f in sorted({f for r in ok for f in r["selected"]} | set(feats_of(ctx.fs().sets[s]))):
            n_sel = sum(1 for r in ok if f in r["selected"])
            signs = [r["sign"].get(f) for r in ok if f in r["sign"]]
            freq = n_sel / len(ok) if ok else None
            rows.append({"feature_set": s, "feature": f, "n_replicates_ok": len(ok), "selection_count": n_sel, "selection_frequency": freq,
                         "selection_frequency_mc_se": float(np.sqrt(freq * (1 - freq) / len(ok))) if ok else None,
                         "sign_positive_share": sum(1 for v in signs if v and v > 0) / len(signs) if signs else None,
                         "note": "coarse: bootstrap refits of LASSO on the KNOWN fitting rows"})
    if gate:
        raise Phase2Stop("FIT_FAILURES", "stability replicates failed above the pre-declared limit", gate)
    st.finalize([D.write_csv(ctx.run.artifacts / "STABILITY.csv", pd.DataFrame(rows))], {"sets": sets, "n_replicates": n_rep})


# ============================================================================ S12 ablation
def s12_ablation(ctx: P3Ctx) -> None:
    st = ctx.run.stage("12", "ablation", ("11",))
    if st.complete_record():
        return
    sel = selection(ctx)
    cfgs = configurations(ctx)
    K = int(ctx.cfg["cv"]["outer_folds"])
    y = ctx.y_train()
    principal = float(ctx.cfg["selection"]["principal_capacity"])
    caps = tuple(float(c) for c in ctx.cfg["metrics"]["capacities"])
    nb = int(ctx.cfg["metrics"]["bootstrap_n"])
    loo = sets_in_order(ctx, ("loo",))
    for s in loo:
        for k in range(K):
            st.item(f"LASSO__{s}__outer{k}", linear_item(ctx, LASSO, s, f"outer{k}"))

    def plan(tmp: Path, seed: int) -> dict[str, Any]:
        target = sel["recommended"] if sel["recommended"] != REF else None
        if target is None:
            t = pd.read_csv(ctx.run.artifacts / "OOF_MODEL_COMPARISON.csv")
            t = t[(t["category"] == PRIMARY) & t["set_kind"].isin(list(CANDIDATE_KINDS)) & ~t["failed"].astype(bool) & (t["n_new_features_offered"] > 0)]
            target = None if t.empty else str(t.sort_values(["adverse_ap", "config"], ascending=[False, True]).iloc[0]["config"])
        if target is None:
            return {"config": None, "features": []}
        imp = final_result(ctx, cfgs, target).get("importance") or {}
        new = list(ctx.fs().sets[cfgs[target]["set"]]["features"])
        ranked = sorted((f for f in new if imp.get(f, 0) > 0), key=lambda f: (-imp[f], f))
        return {"config": target, "features": ranked[: int(ctx.cfg["ablation"]["individual_top_n"])],
                "ranking": "final-model importance on the TRAIN fitting rows (linear: sum |standardised coefficient|; XGBoost: total gain)"}

    target = st.item("IND__plan", plan)
    if target.get("config"):
        c = cfgs[target["config"]]
        s = ctx.fs().sets[c["set"]]
        for f in target["features"]:
            new = [x for x in s["features"] if x != f]
            for k in range(K):
                fold_res = ctx.run.stage(*c["stage"]).result(c["items"][k])
                if c["family"] in (LASSO, ENET):
                    st.item(f"IND__{f}__outer{k}", linear_item(ctx, c["family"], c["set"], f"outer{k}", new=new,
                                                               fixed={"lambda": fold_res["lambda"], "l1_ratio": fold_res["l1_ratio"]}, config=f"IND:{f}"))
                else:
                    st.item(f"IND__{f}__outer{k}", _xgb_ablate_item(ctx, c["set"], new, k, fold_res))
    rows = []
    full = config_oof(ctx, cfgs, f"{LASSO}:{P3_ALL}")
    for s in loo:
        iv = oof_intervals(ctx, st, [f"LASSO__{s}__outer{k}" for k in range(K)])
        if iv is None or full is None:
            continue
        d = bounded_delta(y, iv, full, n_boot=nb, seed=ctx.run.item_seed("S12", s), principal=principal, capacities=caps)
        rows.append({"analysis": "leave_one_domain_out", "model": f"{LASSO}:{P3_ALL}", "removed": ctx.fs().sets[s]["domain"], "set": s,
                     **{f"{k}_full_minus_ablated": v for k, v in d.items()}})
    if target.get("config"):
        base = config_oof(ctx, cfgs, target["config"])
        for f in target["features"]:
            iv = oof_intervals(ctx, st, [f"IND__{f}__outer{k}" for k in range(K)])
            if iv is None or base is None:
                rows.append({"analysis": "individual", "model": target["config"], "removed": f, "failed": True})
                continue
            d = bounded_delta(y, iv, base, n_boot=nb, seed=ctx.run.item_seed("S12", f), principal=principal, capacities=caps)
            rows.append({"analysis": "individual", "model": target["config"], "removed": f, "importance_rank": target["features"].index(f) + 1,
                         **{f"{k}_full_minus_ablated": v for k, v in d.items()}})
    st.finalize([D.write_csv(ctx.run.artifacts / "ABLATION_RESULTS.csv", pd.DataFrame(rows))], {"loo_sets": len(loo), "individual_target": target.get("config")})
    verify_protected(ctx)


def _xgb_ablate_item(ctx: P3Ctx, setname: str, new: list[str], k: int, fold_res: dict[str, Any]) -> Any:
    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        s = ctx.fs().sets[setname]
        fit_rows, te = _rows(ctx, f"outer{k}")
        tr, y = ctx.train_frame(), ctx.y_train()
        Xf = tree_matrix(tr.loc[fit_rows], new, s["baseline"])
        bst = xgb_fit(Xf, y[fit_rows], xgb_params(ctx.cfg, fold_res["params"], seed=seed, nthread=ctx.threads["xgboost"]), fold_res["n_trees"])
        feats = [*new, *s["baseline"]]
        iv = xgb_intervals(bst, tree_matrix(tr.loc[te].reset_index(drop=True), new, s["baseline"]), tree_matrix(tr.loc[fit_rows].reset_index(drop=True), new, s["baseline"]),
                           _plan(ctx, feats, te), tree_col_of(feats))
        D.write_npz(tmp / "oof.npz", rows=np.flatnonzero(te), lo=iv.lo, hi=iv.hi)
        return {"failed": False, "params": fold_res["params"], "n_trees": fold_res["n_trees"]}
    return fn


# ============================================================================ S13 explainability (tuned XGBoost on P3_ALL_RECOVERED)
def s13_explain(ctx: P3Ctx) -> None:
    st = ctx.run.stage("13", "explain", ("12",))
    if st.complete_record():
        return
    import_xgboost()
    import xgboost as xgb

    from falls_ml.evaluation.metrics import pr_auc
    from falls_ml.phase2.evaluate import logloss

    cfgs = configurations(ctx)
    target = f"XGB_TUNED:{ctx.cfg['xgb']['stage1']['set']}"
    c = cfgs[target]
    s = ctx.fs().sets[c["set"]]
    tr, y = ctx.train_frame(), ctx.y_train()
    X = tree_matrix(tr, s["features"], s["baseline"])
    unk = ctx.unknown_train()
    known_row = ~unk[[f for f in feats_of(s) if f in unk.columns]].to_numpy(dtype=bool).any(axis=1) if len(unk.columns) else np.ones(len(y), dtype=bool)
    ex = ctx.cfg["explain"]
    K = int(ctx.cfg["cv"]["outer_folds"])
    stx = ctx.run.stage(*c["stage"])
    dm = lambda M: xgb.DMatrix(M.to_numpy(dtype=np.float32), feature_names=list(M.columns), missing=np.nan)  # noqa: E731
    for k in range(K):
        def fn(tmp: Path, seed: int, k: int = k) -> dict[str, Any]:
            te = (ctx.outer_folds() == k) & known_row      # permutation needs real values: held-out rows whose used features are all KNOWN
            bst = xgb.Booster()
            bst.load_model(bytearray((stx.path(c["items"][k]) / "booster.json").read_bytes()))
            Xt, yt = X.loc[te].reset_index(drop=True), y[te]
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
            return {"fold": k, "ap_baseline": ap0, "n_rows_known": int(te.sum())}
        st.item(f"PERM__outer{k}", fn)

    def shap_fn(tmp: Path, seed: int) -> dict[str, Any]:
        bst = xgb.Booster()
        bst.load_model(bytearray((stx.path(c["final"]) / "booster.json").read_bytes()))
        fm = ctx.fit_mask_train()
        pool = np.flatnonzero(fm)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(pool, size=min(int(ex["shap_sample_rows"]), len(pool)), replace=False))
        Xs = X.iloc[idx]
        contrib = bst.predict(dm(Xs), pred_contribs=True)[:, :-1]
        q = np.percentile(contrib, [5, 25, 50, 75, 95], axis=0)
        summ = pd.DataFrame({"feature": [tree_feature_of(cn) for cn in Xs.columns], "mean_abs_shap": np.abs(contrib).mean(axis=0), "mean_shap": contrib.mean(axis=0),
                             "p05": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4]})
        summ = summ.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
        summ["rank"] = np.arange(1, len(summ) + 1)
        summ["share_of_total"] = summ["mean_abs_shap"] / summ["mean_abs_shap"].sum()
        D.write_csv(tmp / "SHAP_SUMMARY.csv", summ)
        return {"n_rows_sample": int(len(idx))}

    st.item("SHAP", shap_fn)
    perm = pd.concat([pd.read_csv(st.path(f"PERM__outer{k}") / "permutation.csv") for k in range(K)], ignore_index=True)
    pagg = pd.DataFrame([{"feature": f, "delta_ap_mean": float(np.average(g_["delta_ap_mean"], weights=g_["n_rows"])),
                          "delta_ap_sd_over_folds": float(g_["delta_ap_mean"].std(ddof=1)) if len(g_) > 1 else 0.0,
                          "delta_logloss_mean": float(np.average(g_["delta_logloss_mean"], weights=g_["n_rows"]))} for f, g_ in perm.groupby("feature", sort=True)])
    pagg = pagg.sort_values(["delta_ap_mean", "feature"], ascending=[False, True]).reset_index(drop=True)
    pagg["rank"] = np.arange(1, len(pagg) + 1)
    pagg.insert(0, "model", target)
    shap = pd.read_csv(st.path("SHAP") / "SHAP_SUMMARY.csv")
    shap.insert(0, "model", target)
    outs = [D.write_csv(ctx.run.artifacts / "PERMUTATION_IMPORTANCE.csv", pagg), D.write_csv(ctx.run.artifacts / "SHAP_SUMMARY.csv", shap)]
    newf = set(s["features"])
    lim = float(ctx.cfg["gates"]["max_single_feature_share"])
    dom = []
    sn = shap[shap["feature"].isin(newf)]
    if len(sn) and float(sn["share_of_total"].max()) > lim:
        dom.append(f"SHAP: {sn.iloc[0]['feature']} holds {float(sn['share_of_total'].max()):.0%} of mean |SHAP|")
    pos = pagg[pagg["delta_ap_mean"] > 0]
    if len(pos):
        share = (pos.set_index("feature")["delta_ap_mean"] / pos["delta_ap_mean"].sum())
        share = share[share.index.isin(newf)]
        if len(share) and float(share.max()) > lim:
            dom.append(f"permutation: {share.idxmax()} holds {float(share.max()):.0%} of the AP loss")
    pre = set(ctx.tc.header.get("proxy_dominance_preinvestigated") or [])
    v7 = (ctx.facts().get("time_contract") or {}).get("V7_proxy_episode_audit") or {}
    stop = [d for d in dom if not any(d.split(": ", 1)[1].startswith(f + " ") for f in pre) or v7.get("investigation")]
    for d in dom:
        if d not in stop:
            ctx.run.events("S13_explain", "proxy_dominance_preinvestigated", "CONTINUE", detail=d,
                           reason="pre-declared: investigated by the V7 episode audit (not flagged) and the P3_ALL_NO_FALL_RECENCY ablation")
    D.write_json(ctx.run.artifacts / "PROXY_DOMINANCE_CHECK.json", {"dominance": dom, "stopped_on": stop, "preinvestigated": sorted(pre),
                                                                    "v7_investigation": bool(v7.get("investigation"))})
    ctx.run.gate("PROXY_DOMINANCE", bool(stop), "one feature dominates the tuned XGBoost model (possible proxy of the outcome)", stop)
    st.finalize(outs, {"target": target})


# ============================================================================ S14 reused VALIDATION (one-shot, descriptive)
def _final_intervals(ctx: P3Ctx, cfgs: dict[str, dict[str, Any]], name: str, val: pd.DataFrame, vunk: pd.DataFrame, vub: pd.DataFrame) -> Intervals:
    import xgboost as xgb

    c = cfgs[name]
    s = ctx.fs().sets[c["set"]]
    st = ctx.run.stage(*c["stage"])
    fm = ctx.fit_mask_train()
    fit_frame = ctx.train_frame().loc[fm].reset_index(drop=True)
    plan = plan_unknown(feats_of(s), vunk, vub, ctx.feature_source())
    if c["family"] in (LASSO, ENET):
        return linear_intervals(D.read_pickle(st.path(c["final"]) / "model.pkl"), val, fit_frame, plan)
    bst = xgb.Booster()
    bst.load_model(bytearray((st.path(c["final"]) / "booster.json").read_bytes()))
    return xgb_intervals(bst, tree_matrix(val, s["features"], s["baseline"]), tree_matrix(fit_frame, s["features"], s["baseline"]), plan, tree_col_of(feats_of(s)))


def open_validation(ctx: P3Ctx, sel: dict[str, Any]) -> dict[str, Any]:
    path = ctx.run.out / "VALIDATION_OPENED.json"
    fin = ctx.run.plan.get("final_experiment_config_sha256")
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("frozen_selection_sha256") != sel["sha256"] or old.get("final_experiment_config_sha256") != fin:
            raise Phase2Stop("VALIDATION_REOPEN_REFUSED", "VALIDATION was already opened for a different frozen selection or configuration")
        return old
    rec = {"opened_at": utc_now(), "attempt": ctx.run.attempt, "frozen_selection_sha256": sel["sha256"], "final_experiment_config_sha256": fin,
           "plan_sha256": ctx.run.plan_sha, "feature_sets_sha256": ctx.fs().sha256, "configs_to_score": list(sel["descriptive_configs"]),
           "status_of_this_partition": "REUSED (Phase 1 recalibration / reporting; Phase 2 one-shot evaluation): NOT an independent validation",
           "rule": "every configuration was fitted on TRAIN and frozen before this file was written; nothing is changed after VALIDATION is read"}
    D.write_json(path, rec)
    D.mark_readonly(path)
    return rec


def s14_validation(ctx: P3Ctx) -> None:
    st = ctx.run.stage("14", "validation", ("13",))
    if st.complete_record():
        return
    import_xgboost()
    sel = selection(ctx)
    cfgs = configurations(ctx)
    now = sorted(c for c in cfgs if not cfgs[c]["failed"])
    if sorted(sel["descriptive_configs"]) != now:
        raise Phase2Stop("VALIDATION_CONFIG_MISMATCH", "the configurations to score differ from the list frozen in SELECTION_FROZEN.json")
    opened = open_validation(ctx, sel)
    D.append_jsonl(ctx.run.out / "validation_evaluation_registry.jsonl", {"ts": utc_now(), "attempt": ctx.run.attempt, "selection_sha256": sel["sha256"],
                                                                           "recommended": sel["recommended"], "validation_opened_at": opened["opened_at"]})
    val, vunk, vub = ctx.validation_frame(sel)
    caps = tuple(float(c) for c in ctx.cfg["metrics"]["capacities"])
    principal = float(ctx.cfg["selection"]["principal_capacity"])
    vc = ctx.cfg["validation_confirmation"]
    nb = int(ctx.cfg["metrics"]["bootstrap_n"])

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        y = val["y"].to_numpy(dtype=int)
        ivs = {n: _final_intervals(ctx, cfgs, n, val, vunk, vub) for n in sel["descriptive_configs"]}
        D.write_npz(tmp / "validation_predictions.npz", **{f"{n.replace(':', '__')}__{e}": getattr(iv, e) for n, iv in ivs.items() for e in ("lo", "hi")})
        ref = ivs[REF]
        rows, caprows, cal, sub, brows = [], [], [], [], []
        for n, iv in ivs.items():
            row = {"config": n, "family": cfgs[n]["family"], "feature_set": cfgs[n]["set"], "category": cfgs[n]["category"], "set_kind": cfgs[n]["kind"],
                   "role": "reference" if n == REF else "recommended" if n == sel["recommended"] else "shortlist" if n in sel["shortlist"] else "descriptive",
                   **compare_row(y, iv, ref, caps, principal)}
            rows.append({k: v for k, v in row.items()})
            if n in sel["shortlist"] or cfgs[n]["kind"] in ("main", "domain_add"):
                if n != REF:
                    brows.append({"config": n, "reference": REF, **bounded_delta(y, ref, iv, n_boot=nb, seed=seed, principal=principal, capacities=caps)})
                for mode in ("adverse", "favourable"):
                    caprows.append(capacity_table(y, assign(y, iv, mode), caps).assign(config=n, bound=mode, partition="validation"))
            if n in sel["shortlist"]:
                cal.append(calibration_table(y, assign(y, iv, "adverse"), int(ctx.cfg["metrics"]["calibration_groups"])).assign(config=n, bound="adverse"))
                age = pd.to_numeric(val["age_years"], errors="coerce")
                groups = {"sex": {"female": (val["sex"].astype(str) == "female").to_numpy(), "male": (val["sex"].astype(str) == "male").to_numpy()},
                          "age_band": {"65-74": (age < 75).to_numpy(), "75-84": ((age >= 75) & (age < 85)).to_numpy(), "85+": (age >= 85).to_numpy()},
                          "reference_cohort": {"d00_clean": ~val["d00_row"].to_numpy(dtype=bool), "d00_rows": val["d00_row"].to_numpy(dtype=bool)}}
                for dim, gg in groups.items():
                    for lvl, m in gg.items():
                        ev = int(y[m].sum())
                        ok = ev >= int(ctx.cfg["metrics"]["subgroup_min_events"]) and int(m.sum()) - ev >= int(ctx.cfg["metrics"]["subgroup_min_events"])
                        b = bundle(y[m], assign(y[m], Intervals(iv.lo[m], iv.hi[m]), "adverse")) if ok else {}
                        sub.append({"config": n, "subgroup": dim, "level": lvl, "n": int(m.sum()), "events": ev, "adverse_ap": b.get("ap"),
                                    "adverse_auroc": b.get("auroc"), "adverse_citl": b.get("citl"), "reported": ok})
        t = pd.DataFrame(rows)
        D.write_csv(tmp / "VALIDATION_MODEL_COMPARISON.csv", t)
        D.write_csv(tmp / "VALIDATION_BOOTSTRAP_VS_P3_BASE.csv", pd.DataFrame(brows))
        D.write_csv(tmp / "VALIDATION_OPERATIONAL_CAPACITY.csv", pd.concat(caprows, ignore_index=True) if caprows else pd.DataFrame())
        D.write_csv(tmp / "CALIBRATION_CURVES.csv", pd.concat(cal, ignore_index=True) if cal else pd.DataFrame())
        D.write_csv(tmp / "SUBGROUP_SUMMARY.csv", pd.DataFrame(sub))
        rec = sel["recommended"]
        status, detail = NONE, "no configuration was selected on TRAIN; nothing to check"
        if rec != REF:
            r = t.set_index("config").loc[rec].to_dict()
            ok = _qualifies(r, "adverse", vc)
            status = "CONSISTENT_ON_REUSED_VALIDATION" if ok else "NOT_CONSISTENT_ON_REUSED_VALIDATION"
            detail = (f"reused VALIDATION, adverse bound vs P3_BASE: dAP {r['adverse_delta_ap']:+.4f}, d capture@{principal:.0%} "
                      f"{100 * r['adverse_delta_capture_principal']:+.1f} pp, log-loss ratio {r['adverse_logloss_ratio']:.3f}")
        return {"check": status, "detail": detail, "recommended": rec, "validation_rows": int(len(y)), "validation_events": int(y.sum()),
                "n_unknown_rows_validation": int(max((int(np.sum(iv.lo != iv.hi)) for iv in ivs.values()), default=0))}

    res = st.item("validation", fn)
    outs = [D.write_bytes(ctx.run.artifacts / n, (st.path("validation") / n).read_bytes())
            for n in ("VALIDATION_MODEL_COMPARISON.csv", "VALIDATION_BOOTSTRAP_VS_P3_BASE.csv", "VALIDATION_OPERATIONAL_CAPACITY.csv", "CALIBRATION_CURVES.csv",
                      "SUBGROUP_SUMMARY.csv")]
    outs.append(D.write_json(ctx.run.artifacts / "VALIDATION_CHECK.json", res))
    st.finalize(outs, {"check": res["check"]})
    verify_protected(ctx)


# the model stages, in order (run only after GO)
MODEL_STAGES = (s07_lasso, s08_enet, s09_xgb, s10_select, s11_stability, s12_ablation, s13_explain, s14_validation)
__all__ = ["MODEL_STAGES", "configurations", "selection", "top_mask", "HISTORICAL"]
