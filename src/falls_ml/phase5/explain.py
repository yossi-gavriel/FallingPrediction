"""Explanation and stability (descriptive; importance is never causal).

permutation   each outer-fold model on ITS OWN holdout: drop in AP / AUROC when one raw predictor is permuted (n repeats); aggregated over folds
              with the number of folds in which the feature ranks in the top 10
SHAP          XGBoost TreeSHAP contributions (``pred_contribs``) of each outer-fold model on a sample of its holdout; mean |SHAP| per raw
              predictor and the rank agreement between folds
stability     the FINAL development process (configuration tuned on every patient) refitted on bootstrap resamples: LASSO / ENET selection
              frequency, sign consistency and the coefficient distribution; XGBoost permutation importance on the out-of-bag patients
Every task is checkpointed (``work/explain/<task>/result.json`` + COMPLETE.json); a finished task is never recomputed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2 import durable as D
from falls_ml.phase5.engine import COMPLETE, Ctx, UnitSpec, _commit, load_model, load_result
from falls_ml.phase5.metrics import fast_ap


def task_dir(ctx: Ctx, name: str) -> Path:
    return ctx.out / "work" / "explain" / name.replace("|", "__")


def task_done(ctx: Ctx, name: str) -> bool:
    d = task_dir(ctx, name)
    if not (d / COMPLETE).is_file():
        return False
    r = json.loads((d / COMPLETE).read_text(encoding="utf-8"))
    return all((d / n).is_file() and D.sha256_file(d / n) == h for n, h in r["files"].items())


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    from falls_ml.evaluation.metrics import _auroc as a

    return float(a(np.asarray(y, dtype=float), np.asarray(p, dtype=float)))


def permutation_importance(model: Any, X: pd.DataFrame, y: np.ndarray, features: list[str], *, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = model.predict(X)
    ap0, au0 = fast_ap(y, base), _auroc(y, base)
    rows = []
    for f in features:
        dap, dau = [], []
        col = X[f].to_numpy().copy()
        for _ in range(int(repeats)):
            Xp = X.copy()
            Xp[f] = col[rng.permutation(len(col))]
            p = model.predict(Xp)
            dap.append(ap0 - fast_ap(y, p))
            dau.append(au0 - _auroc(y, p))
        rows.append({"feature": f, "drop_ap": float(np.mean(dap)), "drop_auroc": float(np.mean(dau))})
    return pd.DataFrame(rows)


def shap_importance(model: Any, X: pd.DataFrame, *, rows: int, seed: int) -> pd.DataFrame:
    import xgboost as xgb

    rng = np.random.default_rng(seed)
    ix = np.sort(rng.choice(len(X), size=min(int(rows), len(X)), replace=False)) if len(X) > rows else np.arange(len(X))
    A = model.design.transform(X.iloc[ix][model.features])
    contrib = model.model.predict(xgb.DMatrix(A, missing=np.nan), pred_contribs=True)[:, :-1]
    per_col = np.abs(contrib).mean(axis=0)
    df = pd.DataFrame({"column": model.design.columns_, "mean_abs_shap": per_col})
    df["feature"] = df["column"].map(model.design.feature_of_)
    return df.groupby("feature", as_index=False)["mean_abs_shap"].sum()


def run_fold_explain(ctx: Ctx, family: str, setname: str, folds: int, *, repeats: int, shap_rows: int) -> None:
    """Permutation importance (+ SHAP for XGBoost) of every outer-fold model of (family, set) on its own holdout."""
    name = f"FOLDS|{family}|{setname}"
    if task_done(ctx, name):
        return
    d = task_dir(ctx, name)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feats = ctx.sets[setname]
    perm, shap = [], []
    for k in range(folds):
        spec = UnitSpec(uid=f"PRIMARY|{family}|{setname}|outer{k}", stage="PRIMARY", family=family, setname=setname, outer=k)
        res = load_result(ctx, spec)
        model = load_model(ctx, spec)
        te = res["arrays"]["test_idx"]
        X = ctx.frame.iloc[te][feats].reset_index(drop=True)
        y = ctx.y[te].astype(float)
        pi = permutation_importance(model, X, y, feats, repeats=repeats, seed=ctx.seed + 17 * k)
        pi["outer"] = k
        perm.append(pi)
        if family == "XGB":
            sh = shap_importance(model, X, rows=shap_rows, seed=ctx.seed + 31 * k)
            sh["outer"] = k
            shap.append(sh)
    D.write_csv(d / "permutation_folds.csv", pd.concat(perm, ignore_index=True))
    if shap:
        D.write_csv(d / "shap_folds.csv", pd.concat(shap, ignore_index=True))
    D.write_json(d / "result.json", {"task": name, "seconds": round(time.time() - t0, 2), "repeats": repeats, "shap_rows": shap_rows})
    _commit(d)


def run_stability(ctx: Ctx, family: str, setname: str, *, replicates: int) -> None:
    """Bootstrap refits of the FINAL configuration (tuned on every patient) - the development process applied to resamples."""
    from falls_ml.phase5.models import fit_linear, fit_xgb, lambda_grid

    name = f"STABILITY|{family}|{setname}"
    if task_done(ctx, name):
        return
    d = task_dir(ctx, name)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feats = ctx.sets[setname]
    final = load_result(ctx, f"FINAL__{family}__{setname}__all")
    cfgm = final["config"]
    rng = np.random.default_rng(ctx.seed + 101)
    n = len(ctx.y)
    X = ctx.frame[feats]
    rows = []
    for b in range(int(replicates)):
        ix = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), ix)
        Xb, yb = X.iloc[ix].reset_index(drop=True), ctx.y[ix].astype(float)
        if family in ("LASSO", "ENET"):
            from falls_ml.phase5.design import LinearDesign

            spec = ctx.cfg["lasso" if family == "LASSO" else "enet"]
            A = LinearDesign(feats, ctx.meta).fit(Xb).transform(Xb)
            lmax_b, lam = float(lambda_grid(A, yb, float(cfgm["l1_ratio"]), 1, 1.0)[0]), float(cfgm["lambda"])
            grid = np.geomspace(lmax_b, lam, max(2, int(cfgm["lambda_index"]) + 2)) if lam < lmax_b else np.array([lam])   # warm-started path
            m = fit_linear(Xb, yb, feats, ctx.meta, l1_ratio=float(cfgm["l1_ratio"]), lambdas=grid, k=len(grid) - 1, spec=spec, family=family)
            co = m.coefficients()
            agg = co.groupby("feature")["coefficient_standardised"].agg(lambda s: s.iloc[int(np.argmax(np.abs(s.to_numpy())))])
            for f in feats:
                v = float(agg.get(f, 0.0))
                rows.append({"replicate": b, "feature": f, "coefficient": v, "selected": v != 0.0})
        else:
            params = {k: cfgm[k] for k in ctx.cfg["xgb"]["space"]}
            m = fit_xgb(Xb, yb.astype(int), feats, ctx.meta, params, int(cfgm["n_estimators"]), ctx.cfg, seed=ctx.seed + b, nthread=max(1, ctx.jobs),
                        state=ctx.state)
            Xo, yo = X.iloc[oob].reset_index(drop=True), ctx.y[oob].astype(float)
            pi = permutation_importance(m, Xo, yo, feats, repeats=1, seed=ctx.seed + 7 * b)
            pi["rank"] = pi["drop_ap"].rank(ascending=False, method="min")
            for _, r in pi.iterrows():
                rows.append({"replicate": b, "feature": r["feature"], "drop_ap": r["drop_ap"], "rank": r["rank"]})
        ctx.progress(trial=b + 1, n_trials=int(replicates))
    D.write_csv(d / "replicates.csv", pd.DataFrame(rows))
    D.write_json(d / "result.json", {"task": name, "seconds": round(time.time() - t0, 2), "replicates": int(replicates), "final_config": cfgm})
    _commit(d)


# ============================================================================ aggregation (aggregate tables only)
def importance_tables(ctx: Ctx, families: list[str], sets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    perm_rows, shap_rows = [], []
    for fam in families:
        for s in sets:
            d = task_dir(ctx, f"FOLDS|{fam}|{s}")
            if not task_done(ctx, f"FOLDS|{fam}|{s}"):
                continue
            pf = pd.read_csv(d / "permutation_folds.csv")
            pf["rank"] = pf.groupby("outer")["drop_ap"].rank(ascending=False, method="min")
            g = pf.groupby("feature")
            t = pd.DataFrame({"mean_drop_ap": g["drop_ap"].mean(), "sd_drop_ap": g["drop_ap"].std(ddof=0), "mean_drop_auroc": g["drop_auroc"].mean(),
                              "folds_in_top10": g["rank"].apply(lambda r: int((r <= 10).sum())), "n_folds": g["outer"].nunique()}).reset_index()
            t = t.sort_values("mean_drop_ap", ascending=False).reset_index(drop=True)
            t.insert(0, "rank", np.arange(1, len(t) + 1))
            t.insert(0, "feature_set", s)
            t.insert(0, "family", fam)
            perm_rows.append(t)
            if (d / "shap_folds.csv").is_file():
                sf = pd.read_csv(d / "shap_folds.csv")
                sf["rank"] = sf.groupby("outer")["mean_abs_shap"].rank(ascending=False, method="min")
                piv = sf.pivot_table(index="feature", columns="outer", values="rank")
                rho = piv.corr(method="spearman").to_numpy()
                stab = float(np.nanmean(rho[np.triu_indices_from(rho, 1)])) if rho.shape[0] > 1 else float("nan")
                g = sf.groupby("feature")
                st = pd.DataFrame({"mean_abs_shap": g["mean_abs_shap"].mean(), "sd_abs_shap": g["mean_abs_shap"].std(ddof=0),
                                   "folds_in_top10": g["rank"].apply(lambda r: int((r <= 10).sum()))}).reset_index()
                st = st.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
                st.insert(0, "rank", np.arange(1, len(st) + 1))
                st["fold_rank_agreement_spearman"] = stab
                st.insert(0, "feature_set", s)
                st.insert(0, "family", fam)
                shap_rows.append(st)
    return (pd.concat(perm_rows, ignore_index=True) if perm_rows else pd.DataFrame(),
            pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame())


def stability_table(ctx: Ctx, families: list[str], setname: str, perm: pd.DataFrame) -> pd.DataFrame:
    out = []
    for fam in families:
        name = f"STABILITY|{fam}|{setname}"
        if not task_done(ctx, name):
            continue
        r = pd.read_csv(task_dir(ctx, name) / "replicates.csv")
        nrep = int(r["replicate"].nunique())
        g = r.groupby("feature")
        if fam in ("LASSO", "ENET"):
            def sign_cons(s: pd.Series) -> float:
                nz = s[s != 0]
                return float(max((nz > 0).mean(), (nz < 0).mean())) if len(nz) else float("nan")

            t = pd.DataFrame({"selection_frequency": g["selected"].mean(), "sign_consistency": g["coefficient"].apply(sign_cons),
                              "coefficient_median": g["coefficient"].median(), "coefficient_q25": g["coefficient"].quantile(0.25),
                              "coefficient_q75": g["coefficient"].quantile(0.75)}).reset_index()
            unstable = (t["selection_frequency"] < 0.6) | (t["sign_consistency"] < 0.9)
        else:
            t = pd.DataFrame({"top10_frequency": g["rank"].apply(lambda s: float((s <= 10).mean())), "rank_median": g["rank"].median(),
                              "rank_q25": g["rank"].quantile(0.25), "rank_q75": g["rank"].quantile(0.75), "mean_drop_ap_oob": g["drop_ap"].mean()}).reset_index()
            unstable = t["top10_frequency"] < 0.5
        imp = perm[(perm["family"] == fam) & (perm["feature_set"] == setname)] if len(perm) else pd.DataFrame()
        top = set(imp.loc[imp["rank"] <= 10, "feature"]) if len(imp) else set()
        t["important_top10_permutation"] = t["feature"].isin(top)
        t["flag"] = np.where(t["important_top10_permutation"] & unstable, "IMPORTANT_BUT_UNSTABLE", "")
        t.insert(0, "replicates", nrep)
        t.insert(0, "feature_set", setname)
        t.insert(0, "family", fam)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
