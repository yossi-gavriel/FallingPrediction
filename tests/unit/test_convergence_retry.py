"""Bootstrap convergence-retry policy for LASSO replicate fits (v0.7.1). Synthetic data only.

Proves: the real solver's retry converges at a higher full-data iteration limit with the SAME lambda (the CV is untouched); a retry is
accepted only after actual convergence; a retry that still fails is a failed replicate exactly as before; more than 10% failed replicates
still abort; other degenerate fits are never retried; the retry refits the exact same resample with the same seed; results are deterministic."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from falls_ml.errors import DegenerateFitError
from falls_ml.evaluation import convergence_retry as CR
from falls_ml.evaluation.optimism import harrell_optimism
from falls_ml.evaluation.stability import ResamplingError, bootstrap_stability
from falls_ml.models.base import ModelAdapter, coefficient_importance
from falls_ml.models.lasso_cv import LassoConvergenceError, LassoLogisticCV
from falls_ml.models.separation import SeparationError
from falls_ml.pipeline import FittedPipeline
from falls_ml.seeding import int_seed_for, rng_for

FEATURES = ["x_strong", "x_noise1", "x_noise2"]


# ---------------------------------------------------------------------- the real solver
def collinear_frame(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Two almost collinear columns: coordinate descent needs many passes, so a low iteration limit fails at lambda* and 10x converges."""
    rng = np.random.default_rng(seed)
    n, p = 400, 6
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] + 0.05 * rng.normal(size=n)
    y = (rng.random(n) < expit(-1 + 1.5 * X[:, 0] + 0.8 * X[:, 2])).astype(int)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(p)]), y


def lasso(max_iter: int, **extra: Any) -> LassoLogisticCV:
    return LassoLogisticCV({"max_iter": max_iter, "cv_folds": 5, "n_lambda": 30, "refit_unpenalized": False, **extra}, random_state=1)


def test_real_solver_fails_at_the_limit_and_converges_at_10x_with_the_same_lambda_and_cv() -> None:
    X, y = collinear_frame()
    with pytest.raises(LassoConvergenceError) as info:
        lasso(300).fit(X, y)
    err = info.value
    assert err.max_iter == 300 and err.tol == 1e-7 and err.lambda_star_index is not None and err.lambda_star > 0
    m = lasso(300, full_path_max_iter=3000).fit(X, y)
    d = m.diagnostics_
    assert d["lambda_star_converged"] is True and d["full_path_max_iter"] == 3000 and d["tol"] == 1e-7
    assert d["lambda_star_index"] == err.lambda_star_index and d["lambda_star"] == err.lambda_star
    again = lasso(300, full_path_max_iter=3000).fit(X, y)     # deterministic
    assert np.array_equal(again.coef_, m.coef_) and again.cv_results_.equals(m.cv_results_)


def test_cv_curve_does_not_depend_on_the_full_path_limit() -> None:
    X, y = collinear_frame()
    a = lasso(300, full_path_max_iter=3000).fit(X, y).cv_results_
    b = lasso(300, full_path_max_iter=6000).fit(X, y).cv_results_
    pd.testing.assert_series_equal(a["cv_mean_deviance"], b["cv_mean_deviance"])
    pd.testing.assert_series_equal(a["selected"], b["selected"])


def test_default_full_path_limit_is_max_iter_and_is_validated() -> None:
    X, y = collinear_frame(seed=3)
    m = LassoLogisticCV({"cv_folds": 5, "n_lambda": 20}, random_state=1).fit(X, y)
    assert m.diagnostics_["full_path_max_iter"] == 10000 and m.diagnostics_["lambda_star_converged"] is True
    from falls_ml.errors import ConfigError
    with pytest.raises(ConfigError, match="full_path_max_iter"):
        LassoLogisticCV({"full_path_max_iter": 0})


def lasso_fit_fn(max_iter: int, calls: list[dict[str, Any]]):
    X, y = collinear_frame()

    def fit_fn(df: pd.DataFrame, seed: int, *, model_param_overrides: dict[str, Any] | None = None) -> Any:
        calls.append({"seed": seed, "overrides": model_param_overrides, "rows": hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()})
        return SimpleNamespace(model=lasso(max_iter, **(model_param_overrides or {})).fit(df[X.columns], df["y"].to_numpy()))
    return fit_fn, X.assign(y=y)


def test_fit_replicate_accepts_a_real_retry_only_after_convergence_on_the_same_resample_and_seed() -> None:
    calls: list[dict[str, Any]] = []
    fit_fn, df = lasso_fit_fn(300, calls)
    log: list[dict[str, Any]] = []
    model = CR.fit_replicate(fit_fn, df, 17, component="stability", replicate=4, retry_log=log)
    assert model.model.diagnostics_["lambda_star_converged"] is True and model.model.diagnostics_["full_path_max_iter"] == 3000
    assert [c["overrides"] for c in calls] == [None, {"full_path_max_iter": 3000}]
    assert calls[0]["seed"] == calls[1]["seed"] == 17 and calls[0]["rows"] == calls[1]["rows"]     # same replicate, data and seed
    (rec,) = log
    assert rec["outcome"] == "CONVERGED_ON_RETRY" and rec["accepted"] and rec["initial_max_iter"] == 300 and rec["retry_max_iter"] == 3000
    assert rec["retry_converged_at_lambda_star"] is True and rec["retry_lambda_star_index"] == rec["lambda_star_index"]
    assert rec["component"] == "stability" and rec["replicate"] == 4 and "did not converge" in rec["initial_error"]


def test_a_retry_that_still_does_not_converge_is_a_failure() -> None:
    calls: list[dict[str, Any]] = []
    fit_fn, df = lasso_fit_fn(20, calls)          # 20 -> 200 passes: still not enough for this design
    log: list[dict[str, Any]] = []
    with pytest.raises(LassoConvergenceError, match="FAILED_ON_RETRY") as info:
        CR.fit_replicate(fit_fn, df, 17, component="optimism", replicate=0, retry_log=log)
    assert isinstance(info.value, DegenerateFitError)          # counted by the caller exactly like any failed replicate
    assert len(calls) == 2 and log[0]["outcome"] == "FAILED_ON_RETRY" and not log[0]["accepted"] and log[0]["retry_max_iter"] == 200


class FakeModel:
    def __init__(self, diag: dict[str, Any]):
        self.diag = diag

    def fit_diagnostics(self) -> dict[str, Any]:
        return dict(self.diag)


def scripted_fit_fn(retry_diag: dict[str, Any] | None, calls: list[Any], first: BaseException | None = None):
    def fit_fn(df, seed, *, model_param_overrides=None):
        calls.append(model_param_overrides)
        if model_param_overrides is None:
            raise first or LassoConvergenceError("full-data solve did not converge", lambda_star=0.01, lambda_star_index=7, tol=1e-7, max_iter=10000)
        return SimpleNamespace(model=FakeModel(retry_diag))
    return fit_fn


@pytest.mark.parametrize("diag, why", [
    ({"lambda_star_converged": False, "full_path_max_iter": 100000, "lambda_star_index": 7, "lambda_star": 0.01}, "does not report convergence"),
    ({"lambda_star_converged": None, "full_path_max_iter": 100000, "lambda_star_index": 7, "lambda_star": 0.01}, "does not report convergence"),
    ({"lambda_star_converged": True, "full_path_max_iter": 10000, "lambda_star_index": 7, "lambda_star": 0.01}, "did not use the raised limit"),
    ({"lambda_star_converged": True, "full_path_max_iter": 100000, "lambda_star_index": 8, "lambda_star": 0.009}, "selected lambda changed"),
    ({"lambda_star_converged": True, "full_path_max_iter": 100000, "lambda_star_index": 7, "lambda_star": 0.0100001}, "selected lambda changed"),
])
def test_unconverged_or_different_retries_are_rejected(diag, why) -> None:
    calls: list[Any] = []
    log: list[dict[str, Any]] = []
    with pytest.raises(LassoConvergenceError, match="REJECTED_ON_RETRY") as info:
        CR.fit_replicate(scripted_fit_fn(diag, calls), pd.DataFrame(), 1, component="stability", replicate=2, retry_log=log)
    assert why in str(info.value) and log[0]["outcome"] == "REJECTED_ON_RETRY" and calls == [None, {"full_path_max_iter": 100000}]


def test_other_degenerate_fits_are_never_retried_and_old_style_fit_functions_are_not_retried() -> None:
    calls: list[Any] = []
    log: list[dict[str, Any]] = []
    with pytest.raises(SeparationError):
        CR.fit_replicate(scripted_fit_fn({}, calls, first=SeparationError("separation")), pd.DataFrame(), 1, component="stability",
                         replicate=0, retry_log=log)
    assert calls == [None] and log == []

    def old_style(df, seed):
        raise LassoConvergenceError("x", lambda_star=0.1, lambda_star_index=3, tol=1e-7, max_iter=10)
    with pytest.raises(LassoConvergenceError, match="^x$"):
        CR.fit_replicate(old_style, pd.DataFrame(), 1, component="stability", replicate=0, retry_log=log)
    assert log[0]["outcome"] == "NOT_RETRIED" and not log[0]["accepted"]


# ---------------------------------------------------------------------- inside the bootstrap loops
class StubSpec:
    outcome = SimpleNamespace(name="y")
    identifier_columns = ("research_id",)

    def predictor_names(self) -> list[str]:
        return list(FEATURES)


class Identity:
    def transform(self, df):
        return df[FEATURES].astype(float)

    def raw_feature_of(self, column):
        return column

    def design_columns(self):
        return list(FEATURES)


class ConvergenceStub(ModelAdapter):
    """Least-squares-on-logit stand-in with LASSO-like convergence behaviour, scripted per fit seed: seeds in ``hard`` fail at the base
    limit and converge at the raised one; seeds in ``hopeless`` fail at both."""

    name, representation, is_linear = "convergence_stub", "stub", True

    def fit(self, X, y, *, groups=None):
        seed, limit = self.random_state, self.params.get("full_path_max_iter")
        if seed in self.params["hopeless"] or (seed in self.params["hard"] and limit is None):
            raise LassoConvergenceError("scripted non-convergence", lambda_star=0.05, lambda_star_index=11, tol=1e-7, max_iter=limit or 10000)
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        self.coef = np.linalg.lstsq(A, (np.asarray(y) - 0.5) * 4.0, rcond=None)[0]
        self.feature_names_, self.limit = list(X.columns), limit or 10000
        return self

    def linear_predictor(self, X):
        return self.coef[0] + X.to_numpy() @ self.coef[1:]

    def predict_proba(self, X):
        return expit(self.linear_predictor(X))

    def get_feature_importance(self):
        return coefficient_importance(self.feature_names_, self.coef[1:], standardized_scale=False)

    def fit_diagnostics(self):
        return {"lambda_star": 0.05, "lambda_star_index": 11, "lambda_star_converged": True, "full_path_max_iter": self.limit}

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        raise NotImplementedError


def stub_fit_fn(hard: set[int], hopeless: set[int], calls: list[dict[str, Any]], *, overrides: bool = True):
    def fit(df, seed, model_param_overrides=None):
        calls.append({"seed": seed, "overrides": model_param_overrides, "rows": pd.util.hash_pandas_object(df, index=False).sum()})
        model = ConvergenceStub({"hard": hard, "hopeless": hopeless, **(model_param_overrides or {})}, random_state=seed)
        return FittedPipeline(spec=StubSpec(), preprocessor=Identity(), model=model.fit(Identity().transform(df), df["y"].to_numpy()))
    if overrides:
        return lambda df, seed, *, model_param_overrides=None: fit(df, seed, model_param_overrides)
    return lambda df, seed: fit(df, seed)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    n = 300
    rng = rng_for(20260923, "test_convergence_retry")
    df = pd.DataFrame({"research_id": [f"P{i:04d}" for i in range(n)], **{c: rng.normal(size=n) for c in FEATURES}})
    df["y"] = rng.binomial(1, expit(-0.3 + 1.2 * df["x_strong"].to_numpy())).astype("int8")
    return df


SEED, N = 5, 20


def fit_seeds(component: str, replicates: list[int]) -> set[int]:
    return {int_seed_for(SEED, f"{component}:fit:{b}") for b in replicates}


def stability(frame, fit_fn, log=None):
    reference = stub_fit_fn(set(), set(), [])(frame, SEED)
    return bootstrap_stability(frame, fit_fn, n_bootstrap=N, seed=SEED, cluster_column="research_id", thresholds=(0.3,),
                               reference_pipeline=reference, spec=StubSpec(), retry_log=log)


def test_rescued_replicates_are_used_and_recorded_and_without_the_policy_the_same_run_aborts(frame) -> None:
    hard = fit_seeds("stability", [1, 6, 13])                       # 3 of 20 = 15% fail at the base limit
    calls: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    res = stability(frame, stub_fit_fn(hard, set(), calls), log)
    assert res.n_failed == 0 and res.n_bootstrap == N and len(res.convergence_retries) == 3 and log == res.convergence_retries
    assert [r["replicate"] for r in log] == [1, 6, 13] and all(r["outcome"] == "CONVERGED_ON_RETRY" for r in log)
    for b in (1, 6, 13):      # the retry refits the exact same resample with the same seed
        first, retry = [c for c in calls if c["seed"] == int_seed_for(SEED, f"stability:fit:{b}")]
        assert first["overrides"] is None and retry["overrides"] == {"full_path_max_iter": 100000} and first["rows"] == retry["rows"]
    with pytest.raises(ResamplingError, match="3 of 20"):          # the old behaviour (no retry possible): 15% > 10% aborts
        stability(frame, stub_fit_fn(hard, set(), [], overrides=False))


def test_replicates_that_still_fail_count_as_before_and_more_than_ten_percent_abort(frame) -> None:
    two = fit_seeds("stability", [2, 9])                             # 2 of 20 = 10%: allowed, recorded
    res = stability(frame, stub_fit_fn(set(), two, []))
    assert res.n_failed == 2 and res.n_bootstrap == 18 and [f["replicate"] for f in res.failures] == [2, 9]
    assert all("FAILED_ON_RETRY" in f["reason"] for f in res.failures) and all(r["outcome"] == "FAILED_ON_RETRY" for r in res.convergence_retries)
    three = fit_seeds("stability", [2, 9, 15])                       # 3 of 20 = 15% after the retry: the hard stop stays
    with pytest.raises(ResamplingError, match="3 of 20 bootstrap replicates failed"):
        stability(frame, stub_fit_fn(set(), three, []))


def test_results_are_deterministic_and_equal_to_a_run_without_failures(frame) -> None:
    hard = fit_seeds("stability", [0, 4])
    a, b = stability(frame, stub_fit_fn(hard, set(), [])), stability(frame, stub_fit_fn(hard, set(), []))
    clean = stability(frame, stub_fit_fn(set(), set(), []))
    pd.testing.assert_frame_equal(a.feature_stability, b.feature_stability)
    pd.testing.assert_frame_equal(a.feature_stability, clean.feature_stability)   # the stub's converged solution does not depend on the limit
    assert a.convergence_retries == b.convergence_retries


def test_optimism_replicates_use_the_same_policy(frame) -> None:
    log: list[dict[str, Any]] = []
    hard = fit_seeds("optimism", [0, 3])
    out = harrell_optimism(frame, stub_fit_fn(hard, set(), []), n_bootstrap=10, seed=SEED, cluster_column="research_id", spec=StubSpec(),
                           metrics=("auroc",), retry_log=log)
    assert int(out["n_bootstrap"].iloc[0]) == 10 and [(r["component"], r["replicate"], r["outcome"]) for r in log] == [
        ("optimism", 0, "CONVERGED_ON_RETRY"), ("optimism", 3, "CONVERGED_ON_RETRY")]
    with pytest.raises(ResamplingError):
        harrell_optimism(frame, stub_fit_fn(set(), fit_seeds("optimism", [1, 2]), []), n_bootstrap=10, seed=SEED, cluster_column="research_id",
                         spec=StubSpec(), metrics=("auroc",))


def test_policy_record_and_summary() -> None:
    p = CR.policy()
    assert p["max_iter_factor"] == CR.MAX_ITER_FACTOR == 10 and "final model" in p["not_applied_to"] and "> 10%" in p["failure_threshold"]
    s = CR.retry_summary([{"component": "stability", "outcome": "CONVERGED_ON_RETRY", "accepted": True},
                          {"component": "stability", "outcome": "FAILED_ON_RETRY", "accepted": False}])
    assert s["n_initial_convergence_failures"] == 2 and s["n_converged_on_retry"] == 1 and s["n_still_failed"] == 1
    assert s["by_component"]["stability"]["FAILED_ON_RETRY"] == 1 and list(CR.retry_table([]).columns) == CR.RETRY_COLUMNS
