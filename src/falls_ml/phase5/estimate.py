"""``meuhedet-phase5 --estimate``: runtime estimate WITHOUT fitting any model on the real data.

The real file contributes only its SHAPE: the header (which declared V21 columns exist) and the number of eligible index-date rows (two
columns: Index_Date, Is_Eligible_Cohort - never an outcome or a predictor value). If the preflight already ran, its plan gives the exact cohort,
feature-set and domain / ablation counts. The cost of one solver step is then measured on RANDOM synthetic matrices of the same size (a LASSO
and an elastic-net path, an XGBoost fit with early stopping) and multiplied by the work the plan implies for the chosen --mode and --jobs.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2 import durable as D

TREES_PER_FIT = 400            # typical early-stopped size observed on a 48k-patient synthetic run (lr 0.02-0.15)


def _shape(src: Path | None, out: Path, cfg: Any) -> dict[str, Any]:
    plan_p = out / "work" / "PLAN.json"
    if plan_p.is_file():
        plan = json.loads(plan_p.read_text(encoding="utf-8"))
        kinds = plan["kinds"]
        canon = [s for s in plan["sets"] if plan["alias"][s] == s]
        return {"source": "preflight plan", "n": int(plan["n"]), "prevalence": plan["events"] / max(1, plan["n"]),
                "p_old": len(plan["sets"]["OLD"]), "p_new": len(plan["sets"]["OLD_PLUS_NEW_SAFE"]) - len(plan["sets"]["OLD"]),
                "n_primary_sets": sum(1 for s in canon if kinds.get(s) in ("PRIMARY", "SENSITIVITY")),
                "n_domain_sets": sum(1 for s in canon if kinds.get(s) == "DOMAIN"), "n_ablation_sets": sum(1 for s in canon if kinds.get(s) == "ABLATION")}
    if src is None:
        raise ValueError("--estimate needs --input (or a folder whose preflight already ran)")
    from falls_ml.data.meuhedet_wide import NA_VALUES
    from falls_ml.phase4.sealed import read_header
    from falls_ml.phase5.data import tokens

    header = read_header(src)
    raw = (pd.read_parquet(src, columns=["Index_Date", "Is_Eligible_Cohort"]).astype("string") if src.suffix.lower() == ".parquet" else
           pd.read_csv(src, usecols=["Index_Date", "Is_Eligible_Cohort"], dtype="string", na_values=list(NA_VALUES), keep_default_na=False, encoding="utf-8"))
    d = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[us]")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        got = pd.to_datetime(raw["Index_Date"].str.strip().str[:10], format=fmt, errors="coerce")
        d = d.fillna(got)
    n = int(((d == pd.Timestamp(cfg["index_date"])) & (pd.to_numeric(raw["Is_Eligible_Cohort"], errors="coerce") == 1)).sum())
    from falls_ml.data.meuhedet_wide import load_wide_contract, load_wide_mapping

    names25 = set(load_wide_contract(load_wide_mapping().contract_path).names)
    htok = {tokens(c) for c in header if c not in names25}            # 2025 contract columns are OLD, never new
    p_new = sum(1 for f in cfg.v21.features if any(tokens(x) in htok for x in (f.column, *f.aliases)))
    doms = {f.domain for f in cfg.v21.features if any(tokens(x) in htok for x in (f.column, *f.aliases))}
    return {"source": "header + eligibility columns (no outcome, no predictor value read)", "n": n, "prevalence": 0.02, "p_old": 108, "p_new": p_new,
            "n_primary_sets": 4 if p_new else 1, "n_domain_sets": len(doms), "n_ablation_sets": len(cfg["ablations"])}


def _bench(n: int, p: int, prev: float, cfg: Any, jobs: int, seed: int = 0) -> dict[str, float]:
    import xgboost as xgb

    from falls_ml.phase2.enet import enet_logistic_path

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, : p // 2] = (X[:, : p // 2] > 1.0).astype(float)
    beta = np.zeros(p)
    beta[: min(15, p)] = rng.normal(0, 0.5, min(15, p))
    lp = math.log(prev / (1 - prev)) + X @ beta - (X @ beta).mean()
    y = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float)
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    b = cfg.budget
    out: dict[str, float] = {}
    for fam, key in (("LASSO", "lasso"), ("ENET", "enet")):
        a = 1.0 if fam == "LASSO" else 0.5
        nl = int(b[key]["n_lambda"])
        lmax = float(np.max(np.abs(Z.T @ (y - y.mean()))) / n) / a
        lams = lmax * np.logspace(0, math.log10(float(cfg[key]["lambda_min_ratio"])), nl)
        t = time.perf_counter()
        enet_logistic_path(Z, y, lams, a, tol=float(cfg[key]["tol"]), max_iter=int(cfg[key]["max_iter"]))
        out[f"{fam}_path"] = time.perf_counter() - t
    k = int(cfg["cv"]["inner_folds"])
    npar = max(1, min(jobs, k))
    nth = max(1, jobs // npar)
    # a fixed, realistic number of trees: random matrices carry no signal, so early stopping would stop almost at once and under-estimate
    Xf = X.astype(np.float32)
    t = time.perf_counter()
    xgb.train({"objective": "binary:logistic", "tree_method": "hist", "max_depth": 4, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
               "nthread": nth, "verbosity": 0}, xgb.DMatrix(Xf, label=y), num_boost_round=TREES_PER_FIT)
    out["XGB_fit"] = (time.perf_counter() - t) * 1.1            # + the early-stopping evaluations
    out["XGB_trees"] = float(TREES_PER_FIT)
    t = time.perf_counter()
    for _ in range(5):
        xgb.DMatrix(X[: max(1, n // 4)].astype(np.float32))
    out["predict"] = (time.perf_counter() - t) / 5 + 0.002
    return out


def estimate(src: Path | None, out: Path, cfg: Any, jobs: int) -> dict[str, Any]:
    sh = _shape(src, out, cfg)
    K, k = int(cfg["cv"]["outer_folds"]), int(cfg["cv"]["inner_folds"])
    n_tr = int(sh["n"] * (K - 1) / K)
    n_in = int(n_tr * (k - 1) / k)
    p = int(1.3 * (sh["p_old"] + sh["p_new"]))           # design columns (encodings + NULL indicators)
    bm = _bench(max(500, n_in), max(5, p), max(0.005, float(sh["prevalence"])), cfg, jobs)
    b = cfg.budget
    l1 = len(b["enet"]["l1_ratios"])
    npar = max(1, min(jobs, k))
    unit = {"LASSO": math.ceil(k / jobs) * bm["LASSO_path"] * 1.2 + bm["LASSO_path"] * 1.25,
            "ENET": math.ceil(k * l1 / jobs) * bm["ENET_path"] * 1.2 + bm["ENET_path"] * 1.25,
            "XGB": int(b["xgb"]["n_trials"]) * math.ceil(k / npar) * bm["XGB_fit"] * 1.15 + bm["XGB_fit"] * 1.5}
    reuse = {"LASSO": unit["LASSO"], "ENET": unit["ENET"], "XGB": math.ceil(k / npar) * bm["XGB_fit"] * 1.15 + bm["XGB_fit"] * 1.5}
    derived = {f: (unit[f] if cfg["derived_tuning"][f] == "full" else reuse[f]) for f in unit}
    pfeat = sh["p_old"] + sh["p_new"]
    per_item = {}
    stages = {}
    for f in ("LASSO", "ENET", "XGB"):
        per_item[f"PRIMARY|{f}"] = unit[f]
        per_item[f"FINAL|{f}"] = unit[f] * K / max(1, K - 1)
        per_item[f"DOMAIN|{f}"] = derived[f]
        per_item[f"ABLATION|{f}"] = derived[f]
        per_item[f"EXPLAIN|{f}"] = K * pfeat * int(b["permutation_repeats"]) * bm["predict"] * (3.0 if f == "XGB" else 1.0)
        per_item[f"STABILITY|{f}"] = (int(b["stability_linear"]) * bm[f"{'LASSO' if f == 'LASSO' else 'ENET'}_path"] * 0.6 if f != "XGB" else
                                      int(b["stability_xgb"]) * (bm["XGB_fit"] * 1.5 + pfeat * bm["predict"] * 3))
    stages["PRIMARY"] = sum(per_item[f"PRIMARY|{f}"] for f in unit) * K * sh["n_primary_sets"]
    stages["FINAL"] = sum(per_item[f"FINAL|{f}"] for f in unit) * (2 if sh["p_new"] else 1)
    stages["DOMAIN"] = sum(per_item[f"DOMAIN|{f}"] for f in unit) * K * sh["n_domain_sets"]
    stages["ABLATION"] = sum(per_item[f"ABLATION|{f}"] for f in unit) * K * sh["n_ablation_sets"]
    stages["EXPLAIN"] = sum(per_item[f"EXPLAIN|{f}"] for f in unit) * (2 if sh["p_new"] else 1)
    stages["STABILITY"] = sum(per_item[f"STABILITY|{f}"] for f in unit)
    stages["REPORT"] = 60.0 + int(b["bootstrap_n"]) * sh["n"] * 2.5e-7 * (8 + sh["n_domain_sets"] * 3 + sh["n_ablation_sets"] * 3)
    total = sum(stages.values())
    res = {"mode": cfg.mode, "jobs": jobs, "shape": sh, "inner_training_rows": n_in, "design_columns_assumed": p, "benchmark_seconds": bm,
           "per_item_seconds": per_item, "stage_hours": {k_: round(v / 3600, 2) for k_, v in stages.items()}, "total_hours": round(total / 3600, 2),
           "range_hours": [round(0.6 * total / 3600, 1), round(1.6 * total / 3600, 1)],
           "note": "measured on random matrices of the real size (no model was fitted on the real data); real data converge differently, so the "
                   "range is wide. The run is resumable: if it does not finish overnight, the same command with --resume continues it.",
           "first_answer_after_hours": round(stages["PRIMARY"] / 3600, 2)}
    out.mkdir(parents=True, exist_ok=True)
    D.write_json(out / "ESTIMATE.json", res)
    return res


def estimate_text(r: dict[str, Any]) -> str:
    sh = r["shape"]
    lines = [f"Phase 5 runtime estimate (--mode {r['mode']}, --jobs {r['jobs']}) - no model was fitted on the real data",
             f"  shape from {sh['source']}: {sh['n']:,} patients; {sh['p_old']} OLD + {sh['p_new']} new features; "
             f"{sh['n_primary_sets']} primary/sensitivity sets, {sh['n_domain_sets']} domain sets, {sh['n_ablation_sets']} ablations",
             "  stage hours: " + ", ".join(f"{k} {v}" for k, v in r["stage_hours"].items()),
             f"  TOTAL about {r['total_hours']} h (plausible range {r['range_hours'][0]}-{r['range_hours'][1]} h)",
             f"  the primary OLD vs OLD+NEW answer (interim share/) is expected after about {r['first_answer_after_hours']} h",
             f"  {r['note']}"]
    return "\n".join(lines)
