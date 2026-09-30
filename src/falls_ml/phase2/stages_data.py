"""Stages S00-S05: preflight, cohort/split, column registry, feature engineering, TRAIN-only screening, frozen feature sets."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2 import durable as D
from falls_ml.phase2.context import Ctx, import_xgboost
from falls_ml.phase2.engineer import engineer, engineered_registry
from falls_ml.phase2.featuresets import EXPLORATORY, SAFE_DISCOVERY, build_feature_sets, check_category_invariants, design_spec
from falls_ml.phase2.registry import Provenance, column_registry
from falls_ml.phase2.screen import rank_spearman, redundancy_clusters, univariate_screen
from falls_ml.phase2.state import Phase2Stop

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "pyarrow", "PyYAML", "xgboost", "optuna", "SQLAlchemy",
            "threadpoolctl")


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    if out.get("xgboost") is None:
        try:
            out["xgboost-cpu"] = version("xgboost-cpu")
        except PackageNotFoundError:
            pass
    return out


def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.ullTotalPhys / 2**30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (OSError, ValueError, AttributeError):
        return None


def digest_protected(paths: list[Path]) -> dict[str, Any]:
    from falls_ml.meuhedet_sensitivity import directory_digest

    return {p.name: directory_digest(p) for p in paths}


def verify_protected(ctx: Ctx) -> None:
    """HARD stop if any protected (earlier, already reviewed) folder changed since S00 (REVIEW_DECISIONS G-02, plan §9)."""
    before = json.loads((ctx.run.stage("00", "preflight").path("preflight") / "protected_digest.json").read_text(encoding="utf-8"))
    now = digest_protected(ctx.protected)
    changed = [name for name in before if before[name]["combined_sha256"] != now.get(name, {}).get("combined_sha256")]
    if changed:
        raise Phase2Stop("PROTECTED_CHANGED", "an earlier, protected result folder changed during the Phase 2 run", [f"changed: {c}" for c in changed])


def _selftest(ctx: Ctx) -> dict[str, Any]:
    """Known-answer numerical checks (REVIEW_DECISIONS G-09-3): invariants and bit-determinism of the LASSO solver and XGBoost."""
    from falls_ml.models.lasso_cv import lambda_max, lasso_logistic_path, standardize
    from falls_ml.phase2.fitting import xgb_fit, xgb_params, xgb_predict

    rng = np.random.default_rng(20260929)
    X = rng.normal(size=(3000, 6))
    y = (rng.random(3000) < 1 / (1 + np.exp(-(-2.0 + 1.2 * X[:, 0] - 0.8 * X[:, 1])))).astype(float)
    Z, _, _ = standardize(X)
    lm = lambda_max(Z, y)
    grid = lm * np.array([1.0, 0.3, 0.05, 0.01])
    a, b = lasso_logistic_path(Z, y, grid), lasso_logistic_path(Z, y, grid)
    checks = {"lasso_zero_at_lambda_max": bool(np.all(a.beta[0] == 0.0)), "lasso_deterministic": bool(np.array_equal(a.beta, b.beta)),
              "lasso_converged": bool(a.converged.all()), "lasso_signal_signs": bool(a.beta[-1][0] > 0 > a.beta[-1][1])}
    import_xgboost()
    Xd = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    params = xgb_params(ctx.cfg, {"max_depth": 2, "min_child_weight": 5.0, "reg_lambda": 1.0}, seed=7, nthread=ctx.threads["xgboost"])
    p1 = xgb_predict(xgb_fit(Xd, y, params, 50), Xd)
    p2 = xgb_predict(xgb_fit(Xd, y, params, 50), Xd)
    from falls_ml.evaluation.metrics import auroc

    checks.update({"xgboost_deterministic": bool(np.array_equal(p1, p2)), "xgboost_learns_signal": bool(auroc(y, p1) > 0.75)})
    if not all(checks.values()):
        raise Phase2Stop("SELFTEST_FAILED", "the numerical self-test failed on this computer", [k for k, v in checks.items() if not v])
    return checks


def s00_preflight(ctx: Ctx) -> None:
    st = ctx.run.stage("00", "preflight")
    if st.complete_record():
        return

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        env = {"python": sys.version.split()[0], "implementation": platform.python_implementation(), "platform": platform.platform(),
               "machine": platform.machine(), "processor": platform.processor() or None, "cpu_count": os.cpu_count(), "ram_gb": _ram_gb(),
               "gpu": "not used (CPU only)", "packages": _versions(), "threads": ctx.threads,
               "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
               "disk_free_gb": round(shutil.disk_usage(ctx.run.out).free / 2**30, 1)}
        D.write_json(tmp / "environment.json", env)
        D.write_json(tmp / "protected_digest.json", digest_protected(ctx.protected))
        checks = _selftest(ctx)
        D.write_json(tmp / "selftest.json", checks)
        return {"python": env["python"], "cpu_count": env["cpu_count"], "ram_gb": env["ram_gb"], "selftest": "PASSED",
                "protected_folders": len(ctx.protected), "xgboost": env["packages"].get("xgboost") or env["packages"].get("xgboost-cpu"),
                "optuna": env["packages"].get("optuna")}

    st.item("preflight", fn)
    env = json.loads((st.path("preflight") / "environment.json").read_text(encoding="utf-8"))
    out = D.write_json(ctx.run.artifacts / "ENVIRONMENT.json", {k: v for k, v in env.items() if k != "disk_free_gb"})
    st.finalize([out], {"selftest": "PASSED"})


def s01_cohort(ctx: Ctx) -> None:
    from falls_ml.phase2.cohort import build_cohort

    st = ctx.run.stage("01", "cohort", ("00",))
    if st.complete_record():
        return
    scratch = ctx.run.out / "work"
    scratch.mkdir(parents=True, exist_ok=True)

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        r = build_cohort(ctx.src, ctx.ref_dir, contract=ctx.contract, mapping=ctx.mapping, spec=ctx.espec, pepper_file=ctx.pepper_file, scratch=scratch,
                         input_sha=ctx.input_sha)
        D.write_parquet(tmp / "trainval.parquet", r.frame)
        D.write_csv(tmp / "profile.csv", r.profile)
        return r.facts

    facts = st.item("cohort", fn)
    audit = facts.get("label_death_audit") or {}
    ctx.run.gate("LABEL_DEATH_SEMANTICS", bool(audit.get("overwrite_suspected")),
                 "no member who died within the 180-day window carries a recorded fall/fracture: the warehouse label may overwrite earlier events "
                 "with death (a composite endpoint), contrary to the declared definition (a recorded event stays an event when death follows)",
                 [f"deaths within the window: {audit.get('deaths_within_window')}", f"events among them: {audit.get('events_among_deaths_within_window')}",
                  "confirm the label rule with the warehouse team; accept only if the rule is confirmed"])
    out = D.write_json(ctx.run.artifacts / "COHORT_FACTS.json", {k: v for k, v in facts.items() if k != "pepper_source"})
    st.finalize([out], {"train_rows": facts["partitions"]["train"]["n_rows"], "validation_rows": facts["partitions"]["validation"]["n_rows"],
                        "test_rows_dropped": facts["test_rows_dropped"]})
    verify_protected(ctx)


def _provenance(ctx: Ctx) -> Provenance:
    if "prov" not in ctx.cache:
        ctx.cache["prov"] = Provenance(ctx.work(), ctx.facts()["index_date"], contract=ctx.contract, dictionary=ctx.dictionary, d00_config=ctx.d00)
    return ctx.cache["prov"]


def baseline_status(ctx: Ctx) -> dict[str, str]:
    """D-00 status of every BASELINE_15 feature on the Phase 2 rows (worst over its source columns; label-blind)."""
    if "bstatus" not in ctx.cache:
        prov = _provenance(ctx)
        ctx.cache["bstatus"] = {f.canonical: prov.feature(tuple(f.source_columns))["status"] for f in ctx.mapping.features if f.canonical in ctx.baseline}
    return ctx.cache["bstatus"]


def redundancy_universes(tr: pd.DataFrame, er: pd.DataFrame, baseline: list[str], bstatus: dict[str, str], *, threshold: float,
                         order: dict[str, int]) -> tuple[pd.DataFrame, dict[str, dict[str, str]], pd.DataFrame]:
    """Label-blind redundancy clusters computed separately in the SAFE universe (SAFE features + SAFE baseline features) and in the
    exploratory universe (SAFE + UNRESOLVED features + all BASELINE_15 features), so a SAFE feature is never pruned in favour of an
    UNRESOLVED representative. Returns (clusters with a universe column, representatives per universe, Spearman matrix)."""
    from falls_ml.phase2.featuresets import CATEGORY_ELIGIBILITY

    elig = er.loc[er["eligibility"].isin(CATEGORY_ELIGIBILITY[EXPLORATORY]), "feature"].tolist()
    safe = er.loc[er["eligibility"].isin(CATEGORY_ELIGIBILITY[SAFE_DISCOVERY]), "feature"].tolist()
    base = tr[baseline].copy()
    base["sex"] = (base["sex"].astype(str) == "female").astype(float)
    X = pd.concat([tr[elig], base.astype(float)], axis=1)
    corr, _ = rank_spearman(X)
    miss = dict(zip(er["feature"], er["missing_pct_train"]))
    safe_base = [f for f in baseline if bstatus.get(f) == "SAFE"]
    tables, reps = [], {}
    for universe, feats, bl in ((SAFE_DISCOVERY, safe, safe_base), (EXPLORATORY, elig, list(baseline))):
        cols = [c for c in [*feats, *bl] if c in corr.columns]
        cl, rep = redundancy_clusters(corr.loc[cols, cols], threshold=threshold, missing_pct=miss, order=order, baseline=set(bl))
        tables.append(cl.assign(universe=universe) if len(cl) else cl)
        reps[universe] = rep
    clusters = pd.concat([t for t in tables if len(t)], ignore_index=True) if any(len(t) for t in tables) else pd.DataFrame(
        columns=["cluster", "n_members", "representative", "members", "universe"])
    return clusters, reps, corr


def s02_registry(ctx: Ctx) -> None:
    st = ctx.run.stage("02", "registry", ("01",))
    if st.complete_record():
        return
    profile = pd.read_csv(ctx.item_file("01", "cohort", "cohort", "profile.csv"))

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        reg = column_registry(profile, _provenance(ctx), contract=ctx.contract, dictionary=ctx.dictionary, mapping=ctx.mapping, catalogue=ctx.cat,
                              min_observed_train=int(ctx.cfg["eligibility"]["min_observed_train_rows"]))
        D.write_csv(tmp / "01_COLUMN_REGISTRY.csv", reg)
        return {"n_columns": int(len(reg)), "dispositions": reg["final_discovery_eligibility"].value_counts().to_dict(),
                "provenance": reg["provenance_status"].value_counts().to_dict()}

    res = st.item("registry", fn)
    out = ctx.run.artifacts / "01_COLUMN_REGISTRY.csv"
    D.write_bytes(out, (st.path("registry") / "01_COLUMN_REGISTRY.csv").read_bytes())
    st.finalize([out], {"n_columns": res["n_columns"]})


def s03_engineer(ctx: Ctx) -> None:
    st = ctx.run.stage("03", "engineer", ("02",))
    if st.complete_record():
        return
    w = ctx.work()
    train_mask = (w["partition"] == "train").to_numpy()

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        values = engineer(w, ctx.cat)
        er = engineered_registry(values, train_mask, _provenance(ctx), catalogue=ctx.cat, contract=ctx.contract, dictionary=ctx.dictionary,
                                 min_observed_train=int(ctx.cfg["eligibility"]["min_observed_train_rows"]))
        D.write_parquet(tmp / "values.parquet", values.reset_index(drop=True))
        D.write_csv(tmp / "02_ENGINEERED_FEATURE_REGISTRY.csv", er)
        return {"n_features": int(len(er)), "eligibility": er["eligibility"].value_counts().to_dict()}

    res = st.item("engineer", fn)
    out = ctx.run.artifacts / "02_ENGINEERED_FEATURE_REGISTRY.csv"
    D.write_bytes(out, (st.path("engineer") / "02_ENGINEERED_FEATURE_REGISTRY.csv").read_bytes())
    st.finalize([out], {"n_features": res["n_features"]})


def eng_registry(ctx: Ctx) -> pd.DataFrame:
    if "er" not in ctx.cache:
        ctx.cache["er"] = pd.read_csv(ctx.item_file("03", "engineer", "engineer", "02_ENGINEERED_FEATURE_REGISTRY.csv"))
    return ctx.cache["er"]


def s04_screen(ctx: Ctx) -> None:
    st = ctx.run.stage("04", "screen", ("03",))
    if st.complete_record():
        return
    er = eng_registry(ctx)
    tr = ctx.train_frame()
    y = tr["y"].to_numpy(dtype=int)
    sc = ctx.cfg["screening"]

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        uni = univariate_screen(tr[ctx.cat.names], y, er, min_cell=int(sc["min_cell"]), sparse_min_events=int(sc["sparse_min_events"]),
                                baseline=tr[ctx.baseline])
        elig = er.loc[er["eligibility"].isin(["ELIGIBLE", "ELIGIBLE_EXPLORATORY"]), "feature"].tolist()
        order = {f.name: f.order for f in ctx.cat.features}
        clusters, reps, corr = redundancy_universes(tr, er, ctx.baseline, baseline_status(ctx), threshold=float(sc["redundancy_abs_spearman"]), order=order)
        base_cols = [c for c in ctx.baseline if c in corr.columns]
        q = er[["feature", "domain", "eligibility", "provenance_status", "n_observed_train", "missing_pct_train", "n_unique_train", "n_unexpected_null",
                "assessment_form", "mapping_class"]].copy()
        u = uni.set_index("feature")
        q["sparse_flag"] = q["feature"].map(lambda f: bool(u.at[f, "sparse_flag"]) if f in u.index and pd.notna(u.at[f, "sparse_flag"]) else False)
        q["separation_flag"] = q["feature"].map(lambda f: bool(u.at[f, "separation_flag"]) if f in u.index and pd.notna(u.at[f, "separation_flag"]) else False)
        safe_set = set(er.loc[er["eligibility"] == "ELIGIBLE", "feature"])
        for uni_name, tag in ((SAFE_DISCOVERY, "safe"), (EXPLORATORY, "exploratory")):
            cl_u = clusters[clusters["universe"] == uni_name] if len(clusters) else clusters
            members = {m: r["cluster"] for _, r in cl_u.iterrows() for m in str(r["members"]).split("; ")}
            in_uni = safe_set if uni_name == SAFE_DISCOVERY else set(elig)
            rep_u = reps[uni_name]
            q[f"redundancy_cluster_{tag}"] = q["feature"].map(lambda f, members=members: members.get(f, ""))
            q[f"representative_{tag}"] = q["feature"].map(lambda f, rep_u=rep_u, in_uni=in_uni: rep_u.get(f, f) if f in in_uni else "")
            q[f"kept_in_{tag}_pooled_sets"] = q["feature"].map(lambda f, rep_u=rep_u, in_uni=in_uni: f in in_uni and rep_u.get(f, f) == f)
        q["max_abs_spearman_with_baseline"] = q["feature"].map(
            lambda f: round(float(np.nanmax(np.abs(corr.loc[f, base_cols].to_numpy()))), 3) if f in corr.index and base_cols else None)
        q["most_correlated_baseline_feature"] = q["feature"].map(
            lambda f: str(corr.loc[f, base_cols].abs().idxmax()) if f in corr.index and base_cols and corr.loc[f, base_cols].notna().any() else "")
        D.write_csv(tmp / "03_UNIVARIATE_SCREEN.csv", uni)
        D.write_csv(tmp / "04_REDUNDANCY_CLUSTERS.csv", clusters)
        D.write_csv(tmp / "05_FEATURE_QUALITY.csv", q)
        D.write_json(tmp / "representatives.json", reps)
        new = uni[(uni["role"] == "candidate") & uni["eligibility"].isin(["ELIGIBLE", "ELIGIBLE_EXPLORATORY"])]
        dom = new.loc[new["abs_auroc_deviation"].fillna(0) >= float(ctx.cfg["gates"]["single_feature_univariate_auroc"]) - 0.5, "feature"].tolist()
        return {"n_screened": int(len(uni)), "n_clusters": int(len(clusters)), "dominant_features": dom}

    res = st.item("screen", fn)
    ctx.run.gate("SINGLE_FEATURE_DOMINANCE", bool(res["dominant_features"]),
                 f"new feature(s) with TRAIN univariate AUROC >= {ctx.cfg['gates']['single_feature_univariate_auroc']} (or <= its mirror): a single "
                 "feature rivalling the whole baseline model must be investigated as a possible proxy/leak before it is used",
                 [f"feature: {f}" for f in res["dominant_features"]])
    outs = []
    for name in ("03_UNIVARIATE_SCREEN.csv", "04_REDUNDANCY_CLUSTERS.csv", "05_FEATURE_QUALITY.csv"):
        outs.append(D.write_bytes(ctx.run.artifacts / name, (st.path("screen") / name).read_bytes()))
    st.finalize(outs, {"n_clusters": res["n_clusters"]})


def s05_feature_sets(ctx: Ctx) -> None:
    st = ctx.run.stage("05", "feature_sets", ("04",))
    if st.complete_record():
        return
    er = eng_registry(ctx)
    rep = json.loads(ctx.item_file("04", "screen", "screen", "representatives.json").read_text(encoding="utf-8"))
    w, v = ctx.work(), ctx.values()
    train_mask = (w["partition"] == "train").to_numpy()
    bstatus = baseline_status(ctx)

    def fn(tmp: Path, seed: int) -> dict[str, Any]:
        elig = er.loc[er["eligibility"].isin(["ELIGIBLE", "ELIGIBLE_EXPLORATORY"]), "feature"].tolist()
        ds = design_spec(v, train_mask, ctx.cat, elig, unanswered_min_share=float(ctx.cfg["screening"]["item_unanswered_min_share"]))
        fs = build_feature_sets(ctx.cat, er, rep, baseline_features=ctx.baseline, baseline_status=bstatus, design=ds)
        fs.design["baseline_provenance"] = bstatus
        D.write_json(tmp / "FEATURE_SETS.json", fs.to_dict())
        # the category invariants (SAFE_DISCOVERY = proven SAFE only; exploratory never UNSAFE): hard stop, before anything is fitted
        bad = check_category_invariants(fs, er, bstatus, ctx.baseline)
        if bad:
            raise Phase2Stop("INELIGIBLE_FEATURE_IN_SET", "a feature set contains a feature that is not eligible for its category (timing / leakage rule)", bad[:20])
        return {"n_sets": len(fs.sets), "n_fitted": len(fs.fitted()), "sha256": fs.sha256, "aliases": fs.aliases}

    res = st.item("feature_sets", fn)
    out = D.write_bytes(ctx.run.artifacts / "FEATURE_SETS.json", (st.path("feature_sets") / "FEATURE_SETS.json").read_bytes())
    ctx.run.checkpoint(feature_registry_sha256=res["sha256"])
    st.finalize([out], {"n_sets": res["n_sets"], "sha256": res["sha256"]})
