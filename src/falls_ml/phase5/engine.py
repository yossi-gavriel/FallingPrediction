"""One nested-cross-validation UNIT = (model family, feature set, outer fold): tune on the outer-TRAINING patients only, choose the configuration
and the operating thresholds on INNER out-of-fold predictions, refit on the outer-training patients, predict the outer holdout.

The outcome of the outer holdout is never passed to a unit (``run_unit`` receives y for the training rows only): hyper-parameters, early stopping,
the configuration choice and every threshold are therefore functions of the outer-training data alone. A FINAL unit (outer = -1) tunes the same
way on every patient (the development model for coefficients, importance and stability; it has no holdout).
Units are written to their own folder and committed by ``COMPLETE.json`` (sha256 of every file) written LAST: an interrupted unit is redone, a
committed unit is never recomputed. XGBoost trials are appended one by one (``trials.jsonl``) and resume mid-unit.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from falls_ml.phase2 import durable as D
from falls_ml.phase5.models import DeviceState, fit_linear, fit_xgb, lambda_grid, linear_path_task, xgb_inner_fold
from falls_ml.phase5.thresholds import objective, operating_point, thresholds_for

COMPLETE = "COMPLETE.json"


@dataclass
class UnitSpec:
    uid: str
    stage: str
    family: str
    setname: str
    outer: int
    tuning: str = "full"
    reference: str | None = None

    @property
    def folder(self) -> str:
        return self.uid.replace("|", "__")


@dataclass
class Ctx:
    out: Path
    frame: pd.DataFrame
    y: np.ndarray
    meta: dict[str, dict[str, Any]]
    sets: dict[str, list[str]]
    outer: np.ndarray
    cfg: Any
    seed: int
    jobs: int
    state: DeviceState
    log: Callable[[str], None] = print
    progress: Callable[..., None] = lambda **kw: None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def units_dir(self) -> Path:
        return self.out / "work" / "units"


def unit_seed(plan_seed: int, family: str, outer: int) -> int:
    """Identical for every feature set of a family and fold: the same tuning randomness for OLD and NEW (apples to apples)."""
    return int(hashlib.sha256(f"{plan_seed}|{family}|{outer}".encode()).hexdigest()[:8], 16)


def trial_seed(seed: int, k: int) -> int:
    return int(hashlib.sha256(f"{seed}|trial|{k}".encode()).hexdigest()[:8], 16)


def inner_splits(y_tr: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed) % (2**31 - 1))
    return [(a, b) for a, b in skf.split(np.zeros(len(y_tr)), y_tr)]


def outer_folds(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold of every patient (rows are ordered by the outcome-blind row key, so the assignment does not depend on the file order)."""
    from sklearn.model_selection import StratifiedKFold

    out = np.full(len(y), -1, dtype=int)
    for f, (_, te) in enumerate(StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed) % (2**31 - 1)).split(np.zeros(len(y)), y)):
        out[te] = f
    return out


# ============================================================================ commit protocol
def unit_dir(ctx: Ctx, spec: UnitSpec) -> Path:
    return ctx.units_dir / spec.folder


def is_complete(ctx: Ctx, spec: UnitSpec) -> bool:
    d = unit_dir(ctx, spec)
    rec = d / COMPLETE
    if not rec.is_file():
        return False
    try:
        r = json.loads(rec.read_text(encoding="utf-8"))
        return all((d / n).is_file() and D.sha256_file(d / n) == h for n, h in r["files"].items())
    except Exception:  # noqa: BLE001
        return False


def load_result(ctx: Ctx, spec_or_folder: UnitSpec | str) -> dict[str, Any]:
    d = ctx.units_dir / (spec_or_folder.folder if isinstance(spec_or_folder, UnitSpec) else spec_or_folder)
    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
    arr = np.load(d / "arrays.npz")
    res["arrays"] = {k: arr[k] for k in arr.files}
    return res


def load_model(ctx: Ctx, spec: UnitSpec) -> Any:
    return D.read_pickle(unit_dir(ctx, spec) / "model.pkl")


def _commit(d: Path, extra_keep: tuple[str, ...] = ()) -> None:
    files = {p.name: D.sha256_file(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != COMPLETE and not p.name.startswith(".")}
    D.write_json(d / COMPLETE, {"files": files, "committed": time.strftime("%Y-%m-%dT%H:%M:%S")})


# ============================================================================ the unit
def run_unit(ctx: Ctx, spec: UnitSpec) -> dict[str, Any]:
    d = unit_dir(ctx, spec)
    if is_complete(ctx, spec):
        return json.loads((d / "result.json").read_text(encoding="utf-8"))
    keep_trials = spec.family == "XGB" and spec.tuning == "full"
    if d.exists():
        for p in d.iterdir():                                   # an interrupted unit: everything except the XGBoost trial log is redone
            if keep_trials and p.name in ("trials.jsonl", "trials"):
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feats = ctx.sets[spec.setname]
    rows = np.arange(len(ctx.y))
    tr = rows if spec.outer < 0 else rows[ctx.outer != spec.outer]
    te = np.array([], dtype=int) if spec.outer < 0 else rows[ctx.outer == spec.outer]
    X_tr = ctx.frame.iloc[tr][feats].reset_index(drop=True)
    y_tr = ctx.y[tr].astype(float)
    X_te = ctx.frame.iloc[te][feats].reset_index(drop=True)
    seed = unit_seed(ctx.seed, spec.family, spec.outer)
    res = tune_and_fit(ctx, spec, X_tr, y_tr, X_te, feats, seed, d)
    res.update({"uid": spec.uid, "stage": spec.stage, "family": spec.family, "set": spec.setname, "outer": spec.outer, "tuning": spec.tuning,
                "reference": spec.reference, "n_train": int(len(tr)), "n_test": int(len(te)), "n_features": len(feats), "seconds": round(time.time() - t0, 2),
                "device": ctx.state.device if spec.family == "XGB" else "cpu", "seed": seed})
    model = res.pop("_model")
    oof = res.pop("_oof")
    p_te = res.pop("_p_test")
    D.write_npz(d / "arrays.npz", train_idx=tr, test_idx=te, inner_oof=np.asarray(oof, dtype=np.float64), p_test=np.asarray(p_te, dtype=np.float64))
    D.write_pickle(d / "model.pkl", model)
    D.write_json(d / "result.json", res)
    _commit(d)
    return res


def tune_and_fit(ctx: Ctx, spec: UnitSpec, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, feats: list[str], seed: int, d: Path) -> dict[str, Any]:
    """Everything below sees the outer-TRAINING rows (X_tr, y_tr) and the holdout PREDICTORS (X_te) only."""
    cfg = ctx.cfg
    k = cfg.inner_folds
    splits = inner_splits(y_tr.astype(int), k, seed + 1)
    target = cfg.primary_sensitivity
    tie = float(cfg["operating"]["tie_resolution"])
    if spec.family in ("LASSO", "ENET"):
        return _tune_linear(ctx, spec, X_tr, y_tr, X_te, feats, seed, splits, target, tie, d)
    return _tune_xgb(ctx, spec, X_tr, y_tr, X_te, feats, seed, splits, target, tie, d)


def _parallel(n_jobs: int, tasks: list[Callable[[], Any]]) -> list[Any]:
    """Threads (XGBoost releases the GIL while it trains)."""
    from joblib import Parallel, delayed

    if n_jobs <= 1 or len(tasks) <= 1:
        return [t() for t in tasks]
    return Parallel(n_jobs=min(n_jobs, len(tasks)), backend="threading")(delayed(t)() for t in tasks)


def _processes(n_jobs: int, fn: Callable[..., Any], arg_list: list[tuple[Any, ...]]) -> list[Any]:
    """Worker processes (loky; large arrays memory-mapped, single-threaded BLAS inherited from the environment)."""
    from joblib import Parallel, delayed

    if n_jobs <= 1 or len(arg_list) <= 1:
        return [fn(*a) for a in arg_list]
    return Parallel(n_jobs=min(n_jobs, len(arg_list)), backend="loky", max_nbytes="1M")(delayed(fn)(*a) for a in arg_list)


def _choose(rows: list[dict[str, Any]]) -> int:
    best = min(range(len(rows)), key=lambda i: tuple(rows[i]["key"]))
    return best


def _tune_linear(ctx: Ctx, spec: UnitSpec, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, feats: list[str], seed: int,
                 splits: list[tuple[np.ndarray, np.ndarray]], target: float, tie: float, d: Path) -> dict[str, Any]:
    from falls_ml.phase5.design import LinearDesign

    cfg, fam = ctx.cfg, spec.family
    b = cfg.budget["lasso" if fam == "LASSO" else "enet"]
    fspec = cfg["lasso" if fam == "LASSO" else "enet"]
    l1s = [1.0] if fam == "LASSO" else [float(r) for r in b["l1_ratios"]]
    A_full = LinearDesign(feats, ctx.meta).fit(X_tr).transform(X_tr)
    grids = {r: lambda_grid(A_full, y_tr, r, int(b["n_lambda"]), float(fspec["lambda_min_ratio"])) for r in l1s}   # outer-training rows only
    del A_full
    Xarr = X_tr[feats].to_numpy(dtype=np.float64)
    keys = [(j, r) for j in range(len(splits)) for r in l1s]
    res_list = _processes(ctx.jobs, linear_path_task, [(Xarr, feats, splits[j][0], splits[j][1], y_tr, ctx.meta, r, grids[r], fspec) for j, r in keys])
    out = [(j, r, res) for (j, r), res in zip(keys, res_list)]
    n = len(y_tr)
    nl = int(b["n_lambda"])
    oof = {(r, c): np.full(n, np.nan) for r in l1s for c in range(nl)}
    nnz: dict[tuple[float, int], list[float]] = {(r, c): [] for r in l1s for c in range(nl)}
    nconv = 0
    for j, r, res in out:
        v = splits[j][1]
        for c in range(nl):
            oof[(r, c)][v] = res["preds"][:, c]
            nnz[(r, c)].append(res["nnz"][c])
        nconv += res["not_converged"]
    rows = []
    for (r, c), p in oof.items():
        o = objective(y_tr, p, target=target, complexity=float(np.mean(nnz[(r, c)])), tie=tie)
        rows.append({"l1_ratio": r, "lambda": float(grids[r][c]), "lambda_index": c, **o})
    best = _choose(rows)
    br = rows[best]
    p_best = oof[(br["l1_ratio"], br["lambda_index"])]
    model = fit_linear(X_tr, y_tr, feats, ctx.meta, l1_ratio=br["l1_ratio"], lambdas=grids[br["l1_ratio"]], k=int(br["lambda_index"]), spec=fspec, family=fam)
    p_te = model.predict(X_te) if len(X_te) else np.array([])
    trials = pd.DataFrame([{k: v for k, v in r_.items() if k != "key"} for r_ in rows])
    D.write_csv(d / "trials.csv", trials)
    return {"config": {"lambda": br["lambda"], "l1_ratio": br["l1_ratio"], "lambda_index": int(br["lambda_index"])}, "complexity": br["complexity"],
            "n_configs": len(rows), "lambda_on_grid_edge": int(br["lambda_index"]) in (0, nl - 1),
            "inner_objective": {k: v for k, v in br.items() if k not in ("key",)}, "thresholds": thresholds_for(y_tr, p_best, ctx.cfg.targets),
            "inner_op": operating_point(y_tr, p_best, target), "not_converged_fits": int(nconv) + int(model.info["not_converged"]),
            "nnz_final": model.info["nnz"], "_model": model, "_oof": p_best, "_p_test": p_te}


def _distributions(cfg: Any) -> dict[str, Any]:
    from optuna.distributions import FloatDistribution, IntDistribution

    out = {}
    for name, s in cfg["xgb"]["space"].items():
        if s["type"] == "int":
            out[name] = IntDistribution(int(s["low"]), int(s["high"]))
        else:
            out[name] = FloatDistribution(float(s["low"]), float(s["high"]), log=bool(s.get("log", False)))
    return out


def _xgb_trial(ctx: Ctx, A: np.ndarray, y_tr: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]], params: dict[str, Any], seed: int,
               target: float, tie: float) -> dict[str, Any]:
    npar = max(1, min(ctx.jobs, len(splits)))
    nthread = max(1, ctx.jobs // npar)
    tasks = [lambda a=a, v=v, j=j: (j, xgb_inner_fold(A[a], y_tr[a], A[v], params, ctx.cfg, seed=seed + j, nthread=nthread, state=ctx.state))
             for j, (a, v) in enumerate(splits)]
    out = _parallel(npar, tasks)
    oof = np.full(len(y_tr), np.nan)
    iters = []
    for j, r in out:
        oof[splits[j][1]] = r["pred"]
        iters.append(r["best_iter"])
    n_est = int(statistics.median(iters))
    o = objective(y_tr, oof, target=target, complexity=float(params["max_depth"] * n_est), tie=tie)
    return {"oof": oof, "best_iters": iters, "n_estimators": n_est, **o}


def _tune_xgb(ctx: Ctx, spec: UnitSpec, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, feats: list[str], seed: int,
              splits: list[tuple[np.ndarray, np.ndarray]], target: float, tie: float, d: Path) -> dict[str, Any]:
    import optuna
    from optuna.samplers import TPESampler

    from falls_ml.phase5.design import TreeDesign

    cfg = ctx.cfg
    A = TreeDesign(feats, ctx.meta).fit(X_tr).transform(X_tr)
    log_path, tdir = d / "trials.jsonl", d / "trials"
    tdir.mkdir(exist_ok=True)
    done, _ = D.read_jsonl(log_path) if log_path.exists() else ([], None)
    done = [t for t in done if (tdir / f"t{int(t['number']):03d}.npy").is_file()]
    rows: list[dict[str, Any]] = []
    if spec.tuning == "reuse_reference":
        ref = json.loads((ctx.units_dir / spec.reference.replace("|", "__") / "result.json").read_text(encoding="utf-8"))   # type: ignore[union-attr]
        params = {k: ref["config"][k] for k in cfg["xgb"]["space"]}
        r = _xgb_trial(ctx, A, y_tr, splits, params, seed, target, tie)
        rows.append({"number": 0, "params": params, **{k: v for k, v in r.items() if k != "oof"}})
        oofs = {0: r["oof"]}
    else:
        b = cfg.budget["xgb"]
        n_trials, n_start = int(b["n_trials"]), int(b["n_startup_trials"])
        dists = _distributions(cfg)
        oofs = {}
        for t in done[:n_trials]:
            rows.append(t)
            oofs[int(t["number"])] = np.load(tdir / f"t{int(t['number']):03d}.npy").astype(float)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for k in range(len(rows), n_trials):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=trial_seed(seed, k) % (2**31 - 1), n_startup_trials=n_start,
                                                                                     multivariate=True))
                for t in rows:
                    study.add_trial(optuna.trial.create_trial(params=t["params"], distributions=dists, value=float(t["value"])))
                trial = study.ask(dists)
            params = dict(trial.params)
            r = _xgb_trial(ctx, A, y_tr, splits, params, seed, target, tie)
            val = float(r["flagged_share"]) if r["feasible"] else 1.0
            row = {"number": k, "params": params, "value": val, **{kk: vv for kk, vv in r.items() if kk not in ("oof",)}}
            np.save(tdir / f"t{k:03d}.npy", r["oof"].astype(np.float32))
            D.append_jsonl(log_path, json.loads(json.dumps(row, default=float)))
            rows.append(row)
            oofs[k] = r["oof"].astype(np.float32).astype(float)
            ctx.progress(trial=k + 1, n_trials=n_trials)
    for r_ in rows:
        r_["key"] = [9.0e9 if v is None else float(v) for v in r_["key"]]
    best = _choose(rows)
    br = rows[best]
    params = dict(br["params"])
    model = fit_xgb(X_tr, y_tr.astype(int), feats, ctx.meta, params, int(br["n_estimators"]), cfg, seed=seed, nthread=max(1, ctx.jobs), state=ctx.state)
    p_te = model.predict(X_te) if len(X_te) else np.array([])
    p_best = oofs[int(br["number"])]
    trials = pd.DataFrame([{"number": r_["number"], **r_["params"], **{k: r_.get(k) for k in ("flagged_share", "false_alert_share", "ppv", "sensitivity",
                                                                                               "threshold", "ap", "brier", "n_estimators", "feasible")}}
                           for r_ in rows])
    D.write_csv(d / "trials.csv", trials)
    return {"config": {**params, "n_estimators": int(br["n_estimators"])}, "complexity": br["complexity"], "n_configs": len(rows),
            "inner_objective": {k: v for k, v in br.items() if k not in ("key", "params", "best_iters")},
            "thresholds": thresholds_for(y_tr, p_best, ctx.cfg.targets), "inner_op": operating_point(y_tr, p_best, target),
            "not_converged_fits": 0, "device_fallbacks": list(ctx.state.fallbacks), "_model": model, "_oof": p_best, "_p_test": p_te}
