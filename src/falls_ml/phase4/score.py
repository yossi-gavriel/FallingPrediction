"""``falls_ml meuhedet-phase4-score``: BLIND scoring of the 2026 snapshot with the frozen 2025 models. No outcome is read.

Order (each step before the next):
1. the preflight is recomputed and must reproduce the stored SAFE fingerprint (same input, Phase 3 files, settings, code);
2. the frozen models are loaded (A: the persisted Phase 3 fit, commit record verified; B: deterministic re-fit from the committed 2025 products)
   and copied into <out>/frozen; their coefficients must equal the committed Phase 3 coefficients;
3. FROZEN BEFORE ANY 2026 PREDICTION: FROZEN_MODEL_MANIFEST.json (models, feature lists, the frozen 2025 risk cut-offs) and
   PHASE3_INTERNAL_ESTIMATE.json (the Phase 3 nested-CV estimate on 2025 TRAIN, computed with the Phase 4 metric code - 2025 labels only);
4. the 2026 predictors are built (sealed reader) and scored; the 2025 / 2026 patient overlap is counted (identifiers in memory only);
5. BLIND_PREDICTIONS.parquet (pseudonymous row keys, intervals, overlap flag), its manifest (sha256 of predictions, models, feature list,
   input schema; time; code) and BLIND_SCORE_COMPLETE.txt are written once and made read-only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.artifacts import utc_now
from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop
from falls_ml.phase4 import PHASE4_VERSION, WATERMARK
from falls_ml.phase4.common import MODEL_MANIFEST, PLAN, PRED_FILE, PRED_MANIFEST, SCORE_DONE, code_sha, dirs, event, input_identity, read_json, sha_json
from falls_ml.phase4.features import eligible_mask, predict_intervals
from falls_ml.phase4.metrics import metrics_with_ci
from falls_ml.phase4.preflight import audit


def row_key(input_sha: str, member_id: str) -> str:
    """Pseudonymous row key (no pepper): reproducible only with the identical 2026 input file (its sha256 is part of the key)."""
    return hashlib.sha256(f"phase4-row|{input_sha}|{member_id}".encode("utf-8")).hexdigest()[:24]


def frozen_cutoffs(p: np.ndarray, caps: list[float]) -> dict[str, float]:
    """Absolute risk at the top ceil(q x n) of the 2025 TRAIN fitting rows (final model) - the frozen 2025 operating points."""
    s = np.sort(np.asarray(p, dtype=float))[::-1]
    return {str(q): float(s[max(1, int(np.ceil(q * len(s)))) - 1]) for q in caps}


def feature_list(specs: dict[str, Any]) -> dict[str, Any]:
    return {c: {"set": s.setname, "new": s.new, "baseline": s.baseline} for c, s in specs.items()}


def schema_fingerprint(a: Any) -> dict[str, Any]:
    return {"header": a.header, "sealed": a.sealed, "read_columns": list(a.read.frame.columns),
            "dtypes": {c: str(t) for c, t in a.read.frame.dtypes.items()}}


def cohort_2025(input_2025: Path, index_date: str, label_col: str) -> dict[str, set[str]]:
    """2025 cohort membership by member id (in memory only): eligible on the 2025 index date, and the Phase 3 analysis population
    FULL_LABELED (eligible + a usable 2025 label - its availability, a 2025 development fact; no 2026 information)."""
    from falls_ml.data.meuhedet_wide import NA_VALUES
    from falls_ml.phase4.sealed import READ_LOG, read_header

    hdr = read_header(input_2025)
    cols = [c for c in ("Customer_Full_ID", "Index_Date", "Is_Eligible_Cohort", label_col) if c in hdr]
    if len(cols) < 4:
        raise Phase2Stop("INPUT_2025_COLUMNS", f"{input_2025.name}: needs Customer_Full_ID, Index_Date, Is_Eligible_Cohort and {label_col}")
    READ_LOG.append((input_2025.name, tuple(cols)))
    raw = pd.read_csv(input_2025, usecols=cols, dtype="string", na_values=list(NA_VALUES), keep_default_na=False, encoding="utf-8") \
        if input_2025.suffix.lower() != ".parquet" else pd.read_parquet(input_2025, columns=cols).astype("string")
    from falls_ml.data.meuhedet_wide import _coerce_date_column

    dates, _b, _e, _u = _coerce_date_column(raw["Index_Date"], ("%Y-%m-%d", "%d/%m/%Y"))
    on = (dates.dt.normalize() == pd.Timestamp(index_date)).fillna(False).to_numpy()
    el = on & (pd.to_numeric(raw["Is_Eligible_Cohort"], errors="coerce") == 1).fillna(False).to_numpy()
    lab = el & raw[label_col].notna().to_numpy()
    ids = raw["Customer_Full_ID"].astype("string")
    return {"eligible": set(ids[el].dropna().astype(str)), "full_labeled": set(ids[lab].dropna().astype(str))}


def overlap_counts(ids26: pd.Series, c25: dict[str, set[str]]) -> tuple[dict[str, Any], np.ndarray]:
    s26 = set(ids26.astype(str))
    a25, f25 = c25["eligible"], c25["full_labeled"]
    both = s26 & f25
    seen = ids26.astype(str).isin(f25).to_numpy()
    out = {"n_2025_cohort_full_labeled": len(f25), "n_2025_eligible_on_index_date": len(a25), "n_2026_cohort_eligible": len(s26),
           "n_in_both": len(both), "pct_2026_previously_in_2025": round(100.0 * len(both) / len(s26), 3) if s26 else None,
           "n_new_in_2026": len(s26 - f25), "n_2025_not_in_2026": len(f25 - s26),
           "n_in_both_using_2025_eligible": len(s26 & a25),
           "definitions": {"2025_cohort": "the Phase 3 analysis population FULL_LABELED on 2025-01-01 (eligible, usable 2025 label; all partitions)",
                           "2026_cohort": "eligible on 2026-01-01 (Is_Eligible_Cohort = 1); every one is scored",
                           "subgroup_A": "2026 patients also in the 2025 cohort", "subgroup_B": "2026 patients not in the 2025 cohort",
                           "linkage": "Customer_Full_ID, in memory only; never written"},
           "computed_before_outcomes": True}
    return out, seen


def run_score(input_2026: str | Path, input_2025: str | Path, phase3_out: str | Path, *, out_dir: str | Path, phase3_config: str | Path,
              config_path: str | Path, allow_unfrozen_phase3: bool = False, expected_definition_version: str | None = None) -> dict[str, Any]:
    from falls_ml.phase4.common import load_definitions

    src, src25, p3dir, out = Path(input_2026), Path(input_2025), Path(phase3_out), Path(out_dir)
    d = dirs(out)
    if (d["sealed"] / SCORE_DONE).exists():
        raise Phase2Stop("ALREADY_SCORED", "this folder's blind predictions are already frozen (immutable); evaluate them or use a new --out")
    if not (out / PLAN).is_file() or not (d["preflight"] / "PREFLIGHT_RESULT.json").is_file():
        raise Phase2Stop("NO_PREFLIGHT", "run meuhedet-phase4-preflight on this --out first (its last line must be SAFE TO SCORE BLIND)")
    pre = read_json(d["preflight"] / "PREFLIGHT_RESULT.json")
    if not pre.get("safe"):
        raise Phase2Stop("PREFLIGHT_NOT_SAFE", "the stored preflight did not end with SAFE TO SCORE BLIND: scoring is not allowed")
    L = load_definitions(phase3_config, config_path)
    cfg = L["cfg4"]
    # ---- 1. the preflight must reproduce exactly (same input, Phase 3 files, settings, code)
    a = audit(src, p3dir, L=L, allow_unfrozen_phase3=allow_unfrozen_phase3, expected_definition_version=expected_definition_version)
    code = code_sha()
    fp = a.fingerprint(cfg.sha256, code)
    if not a.safe or fp != pre["fingerprint"]:
        raise Phase2Stop("PREFLIGHT_CHANGED", "the preflight no longer reproduces the stored SAFE result (input, Phase 3 files, settings or code changed)",
                         [f"stored {pre['fingerprint'][:16]}…, now {fp[:16]}…", "re-run meuhedet-phase4-preflight (the same --out is fine: nothing is scored yet)"])
    for k in ("frozen", "sealed"):                     # an interrupted earlier attempt never scored anything: keep it aside, start clean
        if d[k].exists():
            aside = out / f"_incomplete_score_{utc_now().replace(':', '').replace('-', '')[:15]}" / k
            aside.parent.mkdir(parents=True, exist_ok=True)
            D.rename_dir(d[k], aside)
            event(out, "score", "incomplete_attempt_moved_aside", "RECOVERED", folder=k)
    event(out, "score", "start", "RUNNING", fingerprint=fp)
    p3 = a.p3
    specs = {c: s for c, s in a.specs.items() if c not in a.excluded_models}
    d["frozen"].mkdir(parents=True, exist_ok=True)
    # ---- 2. frozen models
    fit_frame = p3.fit_frame()
    fsrc = p3.feature_source()
    models, fits = {}, {}
    for c, s in specs.items():
        m = p3.persisted_model(s.setname, cfg["phase3_stage_lasso"])
        if m is None:
            m = p3.refit(s.setname, config_path=phase3_config, workdir=d["frozen"] / "refit_work")
        dst = d["frozen"] / "models" / f"{m['item']}.pkl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m["path"], dst)
        if D.sha256_file(dst) != m["sha256"]:
            raise Phase2Stop("FROZEN_MODEL_COPY", f"{c}: the copied model object differs from its verified source")
        D.mark_readonly(dst)
        fit = D.read_pickle(dst)
        coefs = fit.coefficients()
        # the loaded object must serialise to exactly the committed coefficient file (same writer as Phase 3): byte-identical model
        D.write_csv(d["frozen"] / "models" / f"{m['item']}.coefficients.csv", coefs)
        if D.sha256_file(d["frozen"] / "models" / f"{m['item']}.coefficients.csv") != m["coefficients_sha256"]:
            raise Phase2Stop("FROZEN_MODEL_MISMATCH", f"{c}: the loaded model does not reproduce the committed Phase 3 coefficients")
        used = sorted({fit.design.feature_of_.get(col, col) for col in fit.design.columns_})
        foreign = [u for u in used if u not in set(s.features) and not u.startswith("na__") and u not in fit.design.na_groups]
        if foreign:
            raise Phase2Stop("FROZEN_MODEL_FEATURES", f"{c}: design columns outside the frozen feature list: {foreign[:10]}")
        p25 = fit.predict(fit_frame)
        cut = frozen_cutoffs(p25, [float(q) for q in cfg["metrics"]["frozen_cutoff_capacities"]])
        fits[c] = fit
        models[c] = {"config": c, "set": s.setname, "fitted_set": m["fitted_set"], "phase3_item": m["item"], "mode": m["mode"],
                     "mode_text": "A - exact persisted Phase 3 fitted artifact" if m["mode"] == "A" else "B - deterministic frozen re-fit from the 2025 Phase 3 products only",
                     "artifact": f"frozen/models/{dst.name}", "artifact_sha256": m["sha256"], "coefficients_sha256": m["coefficients_sha256"],
                     "phase3_commit_record_sha256": m["record_sha256"], "phase3": m["result"], "n_design_columns": len(fit.design.columns_),
                     "selected_features": fit.selected_features(), "features_new": s.new, "features_baseline": s.baseline,
                     "frozen_2025_cutoffs": cut, "development_prevalence_fit_rows": float(np.mean(p3.y_train()[p3.fit_mask_train()])),
                     "is_primary": c == cfg.primary}
    flist = feature_list(specs)
    manifest = {"watermark": WATERMARK, "phase4_version": PHASE4_VERSION, "falls_ml_version": __import__("falls_ml").__version__, "created_at": utc_now(),
                "frozen_before_any_2026_prediction": True, "primary": cfg.primary, "models": models, "excluded_models": a.excluded_models,
                "feature_list_sha256": sha_json(flist), "phase3": a.facts["phase3"], "phase3_digest": p3.digest(), "config_sha256": cfg.sha256,
                "code_sha256": code, "capacities": cfg["metrics"]["capacities"], "development_index_date": cfg["development_index_date"]}
    D.write_json(d["frozen"] / "FEATURE_LIST.json", flist)
    D.write_json(d["frozen"] / MODEL_MANIFEST, manifest)
    # ---- Phase 3 internal estimate (2025 TRAIN nested CV, outer out-of-fold) with the SAME metric code (2025 labels only)
    from falls_ml.phase3.bounds import Intervals, assign

    caps = tuple(float(q) for q in cfg["metrics"]["capacities"])
    est: dict[str, Any] = {"definition": "Phase 3 internal estimate: nested grouped cross-validation inside 2025 TRAIN, outer out-of-fold predictions, "
                                         "ADVERSE bound for patients with overwritten history (identical metric code as the 2026 evaluation)", "models": {}}
    y25 = p3.y_train()
    for c, s in specs.items():
        o = p3.oof(s.setname, cfg["phase3_stage_lasso"])
        if o is None:
            est["models"][c] = {"available": False, "reason": "the Phase 3 outer out-of-fold predictions are not committed"}
            continue
        iv = Intervals(lo=o[0], hi=o[1])
        dev = models[c]["development_prevalence_fit_rows"]
        r = metrics_with_ci(y25, assign(y25, iv, "adverse"), caps=caps, dev_prev=dev, n_boot=int(cfg["metrics"]["bootstrap_n"]),
                            seed=int(cfg["seed"]) + 1)
        fav = metrics_with_ci(y25, assign(y25, iv, "favourable"), caps=caps, dev_prev=dev, n_boot=0, seed=int(cfg["seed"]) + 2) if not iv.exact else None
        est["models"][c] = {"available": True, "exact": bool(iv.exact), "n_unknown_rows": int(np.sum(o[0] != o[1])), "adverse": r,
                            "favourable_point": {k: v for k, v in (fav or r).items() if not k.endswith(("_ci_low", "_ci_high"))}}
    D.write_json(d["frozen"] / "PHASE3_INTERNAL_ESTIMATE.json", est)
    model_manifest_sha = D.sha256_file(d["frozen"] / MODEL_MANIFEST)
    for p in (d["frozen"] / MODEL_MANIFEST, d["frozen"] / "FEATURE_LIST.json", d["frozen"] / "PHASE3_INTERNAL_ESTIMATE.json"):
        D.mark_readonly(p)
    event(out, "score", "models_frozen", "DONE", frozen_model_manifest_sha256=model_manifest_sha)
    # ---- 4. blind 2026 predictions
    b = a.built
    ids = b.frame["Customer_Full_ID"].astype("string")
    keys = ids.map(lambda v: row_key(a.input["sha256"], str(v)))
    if keys.duplicated().any():
        raise Phase2Stop("DUPLICATE_PATIENT", "a 2026 patient appears twice among the scored rows")
    c25 = cohort_2025(src25, cfg["development_index_date"], cfg["outcome_contract"]["label_column"])
    ov, seen = overlap_counts(ids, c25)
    pred = pd.DataFrame({"key": keys.to_numpy(), "seen_in_2025_cohort": seen})
    info = {}
    for c, s in specs.items():
        iv = predict_intervals(fits[c], s, b, fit_frame, fsrc)
        if np.any(iv.lo > iv.hi + 1e-15) or not (np.isfinite(iv.lo).all() and np.isfinite(iv.hi).all()):
            raise Phase2Stop("PREDICTION_INVALID", f"{c}: invalid prediction interval")
        pred[f"lo__{s.setname}"] = iv.lo
        pred[f"hi__{s.setname}"] = iv.hi
        info[c] = {"n_scored": int(len(iv.lo)), "n_unknown_rows": int(np.sum(iv.lo != iv.hi)), "mean_predicted": float(np.mean((iv.lo + iv.hi) / 2))}
    d["sealed"].mkdir(parents=True, exist_ok=True)
    D.write_parquet(d["sealed"] / PRED_FILE, pred)
    D.write_json(d["sealed"] / "COHORT_OVERLAP.json", ov)
    pred_sha = D.sha256_file(d["sealed"] / PRED_FILE)
    man = {"watermark": WATERMARK, "phase4_version": PHASE4_VERSION, "created_at": utc_now(), "code_sha256": code, "config_sha256": cfg.sha256,
           "input_2026": a.input, "input_2025": input_identity(src25), "input_schema_sha256": sha_json(schema_fingerprint(a)),
           "feature_list_sha256": sha_json(flist), "frozen_model_manifest_sha256": model_manifest_sha,
           "phase3_internal_estimate_sha256": D.sha256_file(d["frozen"] / "PHASE3_INTERNAL_ESTIMATE.json"),
           "predictions_file": PRED_FILE, "predictions_sha256": pred_sha, "n_scored": int(len(pred)), "models": info,
           "cohort_overlap_sha256": D.sha256_file(d["sealed"] / "COHORT_OVERLAP.json"), "phase3_digest": p3.digest(), "preflight_fingerprint": fp,
           "sealed_columns_never_read": sorted(a.sealed), "outcomes_read": False,
           "row_key": "sha256('phase4-row|' + sha256(2026 input) + '|' + member id)[:24] - reproducible only with the identical 2026 file"}
    D.write_json(d["sealed"] / PRED_MANIFEST, man)
    man_sha = D.sha256_file(d["sealed"] / PRED_MANIFEST)
    D.write_str(d["sealed"] / SCORE_DONE, "\n".join(["BLIND SCORE COMPLETE", f"created_at {man['created_at']}", f"manifest_sha256 {man_sha}",
                                                     f"predictions_sha256 {pred_sha}", f"frozen_model_manifest_sha256 {model_manifest_sha}",
                                                     "outcomes_read false", ""]))
    for p in (d["sealed"] / PRED_FILE, d["sealed"] / PRED_MANIFEST, d["sealed"] / SCORE_DONE, d["sealed"] / "COHORT_OVERLAP.json"):
        D.mark_readonly(p)
    event(out, "score", "complete", "DONE", manifest_sha256=man_sha, predictions_sha256=pred_sha)
    return {"status": "BLIND_SCORE_COMPLETE", "n_scored": int(len(pred)), "models": list(info), "excluded_models": a.excluded_models,
            "modes": {c: m["mode"] for c, m in models.items()}, "manifest_sha256": man_sha, "predictions_sha256": pred_sha,
            "overlap": {k: v for k, v in ov.items() if k.startswith(("n_", "pct_"))}}
