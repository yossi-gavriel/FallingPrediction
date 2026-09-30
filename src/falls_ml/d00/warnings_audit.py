"""LASSO warnings audit across three contexts that must not be confused:

A. the ORIGINAL final fitted reference models (the served models behind the headline metrics);
B. the bootstrap stability / optimism REPLICATES (they feed stability tables and optimism, never the served model or the test metrics);
C. the LEARNING-CURVE refits (they feed the learning-curve points only).

For A it also measures, on TRAIN / VALIDATION only, how sensitive the model is to lambda around lambda* (validation AUROC along the run's
own lambda grid): the test partition is never read. No solver or grid setting is changed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.evaluation.metrics import auroc
from falls_ml.meuhedet_sensitivity import WARNING_TEXT, convergence_audit

CKD_TEXT = ("A sparse binary predictor (few members in the registry) can have no fall, or no non-fall, among its positives in a bootstrap "
            "sample; the Stata-logit rule then omits it from that replicate's FP-selection fit. This is a sparse-data artefact of the resample. It "
            "is NOT evidence that the condition is protective, and a zero or negative coefficient on a sparse predictor must not be read that way.")


def lambda_sensitivity(run_dir: Path, frames: Any, *, window: int = 5) -> dict[str, Any]:
    """Validation AUROC along the run's own lambda grid (path refitted on the run's TRAIN design matrix with the saved preprocessor), at
    lambda* and within +/- ``window`` grid points; plus the CV deviance difference over the same window in units of its standard error."""
    from falls_ml.bundle import load_bundle
    from falls_ml.models.lasso_cv import lasso_logistic_path, standardize

    bundle = load_bundle(run_dir / "model")
    model, pre = bundle.pipeline.model, bundle.pipeline.preprocessor
    names = list(model.feature_names_)
    Xa = pre.transform(frames.train)[names].to_numpy(dtype=np.float64)
    y = frames.train[frames.spec.outcome.name].to_numpy(dtype=np.float64)
    Z, mean, sd = standardize(Xa)
    cv = pd.read_csv(run_dir / "cv_results.csv")
    grid = cv["lambda"].to_numpy(dtype=np.float64)
    p = model.params
    path = lasso_logistic_path(Z, y, grid, tol=float(p["tol"]), max_iter=int(p["max_iter"]), stop=float(p["stop"]))
    k = int(np.flatnonzero(cv["selected"].astype(bool).to_numpy())[0])
    Zv = (pre.transform(frames.validation)[names].to_numpy(dtype=np.float64) - mean) / sd
    yv = frames.validation[frames.spec.outcome.name].to_numpy(dtype=np.int64)
    lo, hi = max(0, k - window), min(grid.size - 1, k + window)
    rows = []
    for i in range(lo, hi + 1):
        lp = path.intercept[i] + Zv @ path.beta[i]
        prob = 1.0 / (1.0 + np.exp(-np.clip(lp, -35.0, 35.0)))   # monotone in lp: the AUROC is that of the linear predictor
        rows.append({"grid_index": i, "offset_from_lambda_star": i - k, "lambda": float(grid[i]), "n_nonzero": int((path.beta[i] != 0).sum()),
                     "validation_auroc": round(float(auroc(yv, prob)), 5), "path_converged": bool(path.converged[i]),
                     "cv_mean_deviance": float(cv["cv_mean_deviance"].iloc[i]) if "cv_mean_deviance" in cv else None})
    t = pd.DataFrame(rows)
    at = float(t.loc[t["offset_from_lambda_star"] == 0, "validation_auroc"].iloc[0])
    se = float(cv["cv_se"].iloc[k]) if "cv_se" in cv and pd.notna(cv["cv_se"].iloc[k]) else None
    dev_span = float(t["cv_mean_deviance"].max() - t["cv_mean_deviance"].min()) if t["cv_mean_deviance"].notna().all() else None
    return {"window_grid_points": window, "validation_auroc_at_lambda_star": round(at, 5),
            "validation_auroc_min_in_window": round(float(t["validation_auroc"].min()), 5), "validation_auroc_max_in_window": round(float(t["validation_auroc"].max()), 5),
            "max_abs_change_in_window": round(float((t["validation_auroc"] - at).abs().max()), 5),
            "cv_deviance_span_in_window": dev_span, "cv_se_at_lambda_star": se,
            "cv_deviance_span_in_se_units": round(dev_span / se, 3) if (dev_span is not None and se) else None,
            "n_nonzero_range": [int(t["n_nonzero"].min()), int(t["n_nonzero"].max())], "table": t.to_dict(orient="records"),
            "rule": "TRAIN path with the saved preprocessor and the run's own lambda grid, evaluated on VALIDATION (uncalibrated linear predictor); "
                    "the test partition is not read"}


def final_fit_summary(run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    a = convergence_audit(run_dir, metrics)
    g = a["final_fit"]["grid"]
    w = a["final_fit"]["warnings"]
    return {"run": run_dir.name, "experiment": a["experiment"], "verdict": a["verdict"], "qualifications": a["qualifications"],
            "preprocessing_fp_selection_converged": "fp_selection_not_converged" not in w,
            "final_lasso_path_converged_all_lambdas": g.get("path_converged"), "selected_lambda_converged": True,
            "cv_fold_paths_converged": g.get("cv_paths_converged"), "cv_minimum_identified": g.get("cv_minimum_identified"),
            "lambda_star_index": g.get("lambda_star_index"), "n_lambda": g.get("n_lambda"), "grid_points_to_end": g.get("grid_points_to_end"),
            "lambda_star_near_grid_end": g.get("near_grid_end"), "lambda_star_at_grid_end": g.get("at_grid_end"),
            "final_fit_warnings": w, "unpenalised_refit": a["final_fit"].get("unpenalised_refit"),
            "solver_or_grid_change_warranted": a["solver_or_grid_change_warranted"], "_audit": a}


def replicate_summary(a: dict[str, Any]) -> dict[str, Any]:
    r = a["replicates"]
    res = r.get("resampling") or {}
    n_stab, n_opt = res.get("stability_n"), res.get("optimism_n")

    def rate(counts: dict[str, int], n: Any) -> dict[str, str]:
        return {k: (f"{v} events in {n} replicates" if n else f"{v} events") for k, v in counts.items()}

    ckd = {k: v for k, v in (r.get("perfect_predictor_omissions_by_column") or {}).items()}
    return {"stability_replicates": n_stab, "optimism_replicates": n_opt, "stability_warnings": rate(r.get("stability_warnings") or {}, n_stab),
            "optimism_warnings": rate(r.get("optimism_warnings") or {}, n_opt), "perfect_predictor_omissions_by_column": ckd,
            "replicate_failures": r.get("replicate_failures") or {},
            "convergence_retry_policy_in_effect": bool(r.get("convergence_retry_policy_in_effect")),
            "convergence_retries": r.get("convergence_retries") or {},
            "fp_form_share_with_final_form": {v: d.get("share_with_final_form") for v, d in (r.get("fp_form_instability") or {}).items()},
            "sparse_predictor_note": CKD_TEXT if ckd else ""}


def _policy_text(r: dict[str, Any]) -> str:
    if not r.get("convergence_retry_policy_in_effect"):
        return "no (run fitted before the v0.7.1 retry policy existed; its replicates were not refitted)"
    return f"yes; initial convergence failures -> outcome: {r.get('convergence_retries') or 'none'}"


def learning_curve_summary(lc: pd.DataFrame | None) -> dict[str, Any]:
    if lc is None or not len(lc):
        return {"available": False, "note": "learning curve not recomputed (use --learning-curve full or reduced)"}
    fitted = lc[lc["status"] == "fitted"]
    col = lambda c: fitted[c] if c in fitted.columns else pd.Series(dtype=object)  # noqa: E731
    not_identified = int((col("cv_minimum_identified") == False).sum())  # noqa: E712
    if str(lc["mode"].iloc[0]) == "reduced":
        na = "not applicable (reduced mode: lambda fixed at lambda*, no CV per refit)"
        return {"available": True, "mode": "reduced", "n_refits": int(len(lc)), "n_fitted": int(len(fitted)), "cv_minimum_not_identified": na,
                "lambda_star_near_grid_end": na, "path_not_converged": f"{int((col('path_converged') == False).sum())} of {len(fitted)} refits",  # noqa: E712
                "cv_fold_paths_not_converged": na,
                "per_fraction": lc[[c for c in ("fraction", "n_train", "n_train_events", "status", "path_converged", "warnings") if c in lc.columns]].to_dict(orient="records")}
    return {"available": True, "mode": str(lc["mode"].iloc[0]), "n_refits": int(len(lc)), "n_fitted": int(len(fitted)),
            "cv_minimum_not_identified": f"{not_identified} of {len(fitted)} refits",
            "lambda_star_near_grid_end": f"{int((col('lambda_star_near_grid_end') == True).sum())} of {len(fitted)} refits",  # noqa: E712
            "path_not_converged": f"{int((col('path_converged') == False).sum())} of {len(fitted)} refits",  # noqa: E712
            "cv_fold_paths_not_converged": f"{int((col('cv_paths_converged') == False).sum())} of {len(fitted)} refits",  # noqa: E712
            "per_fraction": lc[[c for c in ("fraction", "n_train", "n_train_events", "status", "cv_minimum_identified", "lambda_star_index", "n_lambda",
                                            "lambda_star_near_grid_end", "path_converged", "cv_paths_converged", "warnings") if c in lc.columns]].to_dict(orient="records")}


def materiality(final: dict[str, Any], sens: dict[str, Any] | None, lc: dict[str, Any], rep: dict[str, Any]) -> list[str]:
    """Rule-based statements on whether the warnings threaten the headline (held-out test) estimates."""
    out = []
    if final.get("lambda_star_at_grid_end"):
        out.append("MAY THREATEN: lambda* is the last grid value, so the CV minimum may lie beyond the grid. A separately named run with an extended "
                   "grid would be justified (the original result stays as it is).")
    elif final.get("cv_minimum_identified") is False:
        if sens and sens.get("max_abs_change_in_window") is not None:
            ch = sens["max_abs_change_in_window"]
            verdict = "does not materially threaten" if ch <= 0.005 else ("is unlikely to materially threaten" if ch <= 0.01 else "may affect")
            out.append(f"cv_minimum_not_identified (flat CV curve) {verdict} the headline AUROC: across +/-{sens['window_grid_points']} grid points around "
                       f"lambda*, VALIDATION AUROC ranged {sens['validation_auroc_min_in_window']:.4f}-{sens['validation_auroc_max_in_window']:.4f} "
                       f"(max change {ch:.4f}); the CV deviance varied by {sens.get('cv_deviance_span_in_se_units')} standard errors.")
        else:
            out.append("cv_minimum_not_identified: the CV curve is flat around lambda*; neighbouring lambdas give nearly equivalent models (lambda "
                       "sensitivity not computed - dataset not available locally).")
    else:
        out.append("Final fit: the CV minimum is identified and lambda* lies inside the grid.")
    if final.get("cv_fold_paths_converged") is False:
        out.append(f"Some CV fold paths had unconverged lambdas (typically the smallest ones); lambda* is {final.get('grid_points_to_end')} grid points from "
                   "the end of the grid and the selected lambda itself converged (a completed run guarantees it).")
    if final.get("final_lasso_path_converged_all_lambdas") is False:
        out.append("The full-data path had unconverged lambdas other than lambda*; the served model at lambda* converged.")
    if rep.get("stability_warnings") or rep.get("optimism_warnings"):
        out.append("Replicate warnings (stability / optimism) affect the selection-stability and optimism summaries only - not the served model and "
                   "not the held-out test metrics.")
    if lc.get("available"):
        out.append(f"Learning-curve refits: cv_minimum_not_identified in {lc['cv_minimum_not_identified']}; these refits produce only the "
                   "learning-curve points and never the headline metrics.")
    if final.get("solver_or_grid_change_warranted"):
        out.append("Solver / grid change warranted by the audit rule: " + "; ".join(final["solver_or_grid_change_warranted"]) + " (not applied here).")
    else:
        out.append("No solver or grid change is warranted: no boundary condition; settings were not changed to silence warnings.")
    return out


def render_markdown(audit: dict[str, Any], wm: str, status_lines: list[str]) -> str:
    L = ["# LASSO warnings audit", "", f"**{wm}**", "", *[f"- {s}" for s in status_lines], "",
         "Three contexts are kept apart: (A) the original final fitted reference models - the served models behind every headline number; "
         "(B) the bootstrap stability / optimism replicates; (C) the learning-curve refits. Nothing was re-tuned; no solver or grid setting was "
         "changed. The lambda-sensitivity check uses TRAIN and VALIDATION only.", ""]
    L += ["## A. Final fitted models", "", "| model | verdict | FP selection converged | path converged (all lambdas) | CV fold paths converged | CV minimum identified | lambda* index / n | near grid end | final-fit warnings |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, f in audit["final_fits"].items():
        L.append(f"| {k} | {f['verdict']} | {f['preprocessing_fp_selection_converged']} | {f['final_lasso_path_converged_all_lambdas']} | {f['cv_fold_paths_converged']} | "
                 f"{f['cv_minimum_identified']} | {f['lambda_star_index']} / {f['n_lambda']} | {f['lambda_star_near_grid_end']} | {f['final_fit_warnings'] or 'none'} |")
    for k, s in (audit.get("lambda_sensitivity") or {}).items():
        if s.get("error"):
            L += ["", f"Lambda sensitivity {k}: not computed ({s['error']})"]
            continue
        L += ["", f"**Lambda sensitivity, {k}** (VALIDATION): AUROC at lambda* {s['validation_auroc_at_lambda_star']:.4f}; within +/-{s['window_grid_points']} grid points "
              f"{s['validation_auroc_min_in_window']:.4f}-{s['validation_auroc_max_in_window']:.4f} (max change {s['max_abs_change_in_window']:.4f}); non-zero columns "
              f"{s['n_nonzero_range'][0]}-{s['n_nonzero_range'][1]}; CV deviance span {s.get('cv_deviance_span_in_se_units')} SE."]
    L += ["", "## B. Bootstrap stability / optimism replicates", ""]
    for k, r in audit["replicates"].items():
        L += [f"### {k}", "", f"- stability replicates: {r['stability_replicates']}; warnings: {r['stability_warnings'] or 'none'}",
              f"- optimism replicates: {r['optimism_replicates']}; warnings: {r['optimism_warnings'] or 'none'}",
              f"- perfect-predictor omissions by column (replicates): {r['perfect_predictor_omissions_by_column'] or 'none'}",
              f"- replicate failures (recorded, never skipped silently): {r['replicate_failures'] or 'none'}",
              f"- convergence-retry policy in effect: {_policy_text(r)}",
              f"- FP form: share of replicates choosing the final form: {r['fp_form_share_with_final_form'] or 'n/a'}"]
        if r["sparse_predictor_note"]:
            L.append(f"- **Sparse predictor note:** {r['sparse_predictor_note']}")
        L.append("")
    L += ["## C. Learning-curve refits", ""]
    for k, lc in audit["learning_curve"].items():
        if not lc.get("available"):
            L += [f"- {k}: {lc['note']}"]
            continue
        L += [f"### {k} ({lc['mode']} mode, {lc['n_fitted']} of {lc['n_refits']} refits fitted)", "",
              f"- cv_minimum_not_identified: {lc['cv_minimum_not_identified']}; lambda* near the grid end: {lc['lambda_star_near_grid_end']}; "
              f"path not converged: {lc['path_not_converged']}; CV fold paths not converged: {lc['cv_fold_paths_not_converged']}", "",
              "| fraction | n train | status | CV min identified | lambda* index / n | near end | warnings |", "|---|---|---|---|---|---|---|"]
        na = lambda v: "n/a" if v is None or (isinstance(v, float) and v != v) else v  # noqa: E731
        for r in lc["per_fraction"]:
            L.append(f"| {r.get('fraction')} | {r.get('n_train')} | {str(r.get('status'))[:90]} | {na(r.get('cv_minimum_identified'))} | "
                     f"{na(r.get('lambda_star_index'))} / {na(r.get('n_lambda'))} | {na(r.get('lambda_star_near_grid_end'))} | {r.get('warnings')} |")
        L.append("")
    L += ["## Sensitivity runs of this analysis (final fits)", "", "| cell | origin | verdict | CV minimum identified | lambda* index / n | near grid end | final-fit warnings |",
          "|---|---|---|---|---|---|---|"]
    for k, f in (audit.get("new_runs") or {}).items():
        L.append(f"| {k} | {f.get('origin', 'fitted in this attempt')} | {f['verdict']} | {f['cv_minimum_identified']} | {f['lambda_star_index']} / {f['n_lambda']} | "
                 f"{f['lambda_star_near_grid_end']} | {f['final_fit_warnings'] or 'none'} |")
    reps = audit.get("new_run_replicates") or {}
    if reps:
        L += ["", "### Bootstrap replicates of the sensitivity runs", "",
              "The convergence-retry policy (v0.7.1) refits a replicate whose full-data LASSO solve did not converge ONCE on the same resample, with the "
              "same seed, the same selected lambda and the same tolerance, at a higher full-data iteration limit; it is accepted only if it converged. "
              "A replicate that still fails is counted as failed; more than 10% failed replicates abort the run (threshold unchanged). Runs fitted "
              "before the policy existed are shown as such and were not refitted.", "",
              "| cell | stability / optimism replicates | retry policy in effect | initial convergence failures -> outcome | replicate failures (counted) |",
              "|---|---|---|---|---|"]
        for k, r in reps.items():
            L.append(f"| {k} | {r['stability_replicates']} / {r['optimism_replicates']} | {'yes' if r['convergence_retry_policy_in_effect'] else 'no (fitted before v0.7.1)'} | "
                     f"{r['convergence_retries'] or 'none'} | {r['replicate_failures'] or 'none'} |")
    L += ["", "## Do the warnings threaten the headline estimates?", ""]
    for k, items in audit["materiality"].items():
        L += [f"**{k}**", ""] + [f"- {x}" for x in items] + [""]
    L += ["## What the warnings mean", ""] + [f"- `{k}`: {v}" for k, v in WARNING_TEXT.items()]
    return "\n".join(L) + "\n"
