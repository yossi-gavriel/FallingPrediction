"""Deterministic convergence retry for LASSO bootstrap-replicate fits (bootstrap stability and Harrell optimism only).

Pre-declared policy (version :data:`POLICY_VERSION`, identical for every replicate and every run):

- only a :class:`~falls_ml.models.lasso_cv.LassoConvergenceError` triggers it (the full-data solve at the CV-selected lambda did
  not converge); every other degenerate fit is recorded exactly as before;
- the SAME resample is refitted ONCE with the SAME fit seed and the SAME model parameters, except that the iteration limit of the
  full-data path is raised to ``MAX_ITER_FACTOR`` x the limit that failed (LASSO parameter ``full_path_max_iter``). The CV fold
  paths keep their limit, so the folds, the lambda grid, the CV curve and therefore the selected lambda are those of the failed
  attempt; the tolerance is unchanged;
- the retry is accepted ONLY when the fitted model itself reports convergence at the selected lambda, under the raised limit, at the
  same selected lambda (grid index and value). Anything else is a failure - unconverged coefficients are never accepted;
- a replicate that still fails is recorded and counted exactly as before: the failure threshold
  (:data:`~falls_ml.evaluation.stability.MAX_FAILURE_FRACTION`) is unchanged, so more than 10% failed replicates still abort;
- every initial failure and its retry outcome is recorded (``retry_log`` rows; run-log events ``bootstrap_convergence_retry``).

Not applied to the final model, the apparent-performance refit of the optimism correction, learning-curve refits or internal-external CV.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.errors import DegenerateFitError
from falls_ml.logging_utils import get_logger
from falls_ml.models.lasso_cv import LassoConvergenceError

log = get_logger(__name__)

POLICY_VERSION = 1
MAX_ITER_FACTOR = 10
OVERRIDE_KEYWORD = "model_param_overrides"
OUTCOMES = ("CONVERGED_ON_RETRY", "FAILED_ON_RETRY", "REJECTED_ON_RETRY", "NOT_RETRIED")
RETRY_COLUMNS = ["component", "replicate", "fit_seed", "policy_version", "outcome", "accepted", "tol", "initial_max_iter", "retry_max_iter",
                 "lambda_star", "lambda_star_index", "retry_lambda_star", "retry_lambda_star_index", "retry_converged_at_lambda_star",
                 "initial_error", "retry_error"]


def policy() -> dict[str, Any]:
    """The policy as recorded next to every run that used it."""
    return {"name": "bootstrap_lasso_convergence_retry", "version": POLICY_VERSION, "max_iter_factor": MAX_ITER_FACTOR,
            "applies_to": ["bootstrap stability replicates", "Harrell optimism replicates"],
            "not_applied_to": ["final model", "optimism apparent-performance refit", "learning-curve refits", "internal-external CV"],
            "trigger": "LassoConvergenceError only (full-data solve at the CV-selected lambda did not converge)",
            "retry": f"same resample, same fit seed, same parameters, one retry; full-data path iteration limit x{MAX_ITER_FACTOR}; "
                     "CV fold paths, lambda grid, lambda selection and tolerance unchanged",
            "acceptance": "the refitted model reports convergence at the selected lambda, under the raised limit, at the same lambda "
                          "(grid index and value); otherwise the replicate is a failure",
            "failure_threshold": "unchanged: a replicate that still fails is counted as before; > 10% failed replicates abort the run"}


def supports_overrides(fit_fn: Callable[..., Any]) -> bool:
    try:
        return OVERRIDE_KEYWORD in inspect.signature(fit_fn).parameters
    except (TypeError, ValueError):
        return False


def _short(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:400]


def fit_replicate(fit_fn: Callable[..., Any], sample: pd.DataFrame, fit_seed: int, *, component: str, replicate: int,
                  retry_log: list[dict[str, Any]] | None) -> Any:
    """``fit_fn(sample, fit_seed)`` with the convergence-retry policy. Returns the fitted pipeline or raises a DegenerateFitError
    (the caller records it as a failed replicate, exactly as before)."""
    try:
        return fit_fn(sample, fit_seed)
    except LassoConvergenceError as exc:
        first = exc
    rec: dict[str, Any] = {"component": component, "replicate": int(replicate), "fit_seed": int(fit_seed), "policy_version": POLICY_VERSION,
                           "tol": first.tol, "initial_max_iter": first.max_iter, "lambda_star": first.lambda_star,
                           "lambda_star_index": first.lambda_star_index, "initial_error": _short(first), "retry_max_iter": None,
                           "retry_lambda_star": None, "retry_lambda_star_index": None, "retry_converged_at_lambda_star": None, "retry_error": None}

    def finish(outcome: str, error: BaseException | None = None) -> None:
        rec.update({"outcome": outcome, "accepted": outcome == "CONVERGED_ON_RETRY"})
        if error is not None:
            rec["retry_error"] = _short(error)
        if retry_log is not None:
            retry_log.append(rec)
        log.warning("bootstrap_convergence_retry", extra_fields={k: rec[k] for k in ("component", "replicate", "outcome", "initial_max_iter",
                                                                                     "retry_max_iter", "lambda_star", "lambda_star_index", "tol")})

    if first.max_iter is None or first.lambda_star_index is None or not supports_overrides(fit_fn):
        finish("NOT_RETRIED", ValueError("the failed fit did not report its iteration limit / selected lambda, or the fit function "
                                         "takes no parameter overrides"))
        raise first
    rec["retry_max_iter"] = retry_max_iter = int(first.max_iter) * MAX_ITER_FACTOR
    try:
        model = fit_fn(sample, fit_seed, **{OVERRIDE_KEYWORD: {"full_path_max_iter": retry_max_iter}})
    except DegenerateFitError as exc:
        finish("FAILED_ON_RETRY", exc)
        raise LassoConvergenceError(f"{_short(first)} [convergence retry at full_path_max_iter={retry_max_iter}: FAILED_ON_RETRY - {_short(exc)}]",
                                    lambda_star=first.lambda_star, lambda_star_index=first.lambda_star_index, tol=first.tol,
                                    max_iter=retry_max_iter) from exc
    diag = model.model.fit_diagnostics() if getattr(model, "model", None) is not None else {}
    rec.update({"retry_lambda_star": diag.get("lambda_star"), "retry_lambda_star_index": diag.get("lambda_star_index"),
                "retry_converged_at_lambda_star": diag.get("lambda_star_converged")})
    problems = []
    if diag.get("lambda_star_converged") is not True:
        problems.append("the refitted model does not report convergence at the selected lambda")
    if diag.get("full_path_max_iter") != retry_max_iter:
        problems.append(f"the refit did not use the raised limit (reported {diag.get('full_path_max_iter')})")
    if diag.get("lambda_star_index") != first.lambda_star_index or diag.get("lambda_star") is None or first.lambda_star is None \
            or not np.isclose(float(diag["lambda_star"]), float(first.lambda_star), rtol=1e-12, atol=0.0):
        problems.append(f"the selected lambda changed ({first.lambda_star} [{first.lambda_star_index}] -> "
                        f"{diag.get('lambda_star')} [{diag.get('lambda_star_index')}])")
    if problems:
        reason = LassoConvergenceError("; ".join(problems))
        finish("REJECTED_ON_RETRY", reason)
        raise LassoConvergenceError(f"{_short(first)} [convergence retry at full_path_max_iter={retry_max_iter}: REJECTED_ON_RETRY - "
                                    f"{'; '.join(problems)}]", lambda_star=first.lambda_star, lambda_star_index=first.lambda_star_index,
                                    tol=first.tol, max_iter=retry_max_iter)
    finish("CONVERGED_ON_RETRY")
    return model


def retry_table(retry_log: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(retry_log, columns=RETRY_COLUMNS)


def retry_summary(retry_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per component and outcome (aggregate; no row-level data involved)."""
    by: dict[str, dict[str, int]] = {}
    for r in retry_log:
        c = by.setdefault(str(r["component"]), {"initial_convergence_failures": 0, **{o: 0 for o in OUTCOMES}})
        c["initial_convergence_failures"] += 1
        c[str(r["outcome"])] += 1
    return {"policy": policy(), "n_initial_convergence_failures": len(retry_log),
            "n_converged_on_retry": sum(1 for r in retry_log if r["outcome"] == "CONVERGED_ON_RETRY"),
            "n_still_failed": sum(1 for r in retry_log if r["outcome"] != "CONVERGED_ON_RETRY"), "by_component": by,
            "records": [{k: r.get(k) for k in RETRY_COLUMNS} for r in retry_log]}
