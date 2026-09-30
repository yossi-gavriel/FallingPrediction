"""Linear alternative adapters on the ``efalls_reference_coded`` design (L4_alternative; architecture §3.2).

- :class:`UnpenalizedLogistic`: maximum-likelihood logistic regression (statsmodels ``Logit``, Newton).
- :class:`ElasticNetLogistic`: elastic-net logistic regression (scikit-learn ``saga``) on internally
  standardised columns.

Both are experimental comparators and never eFalls-labelled models (spec §1.2). Neither silently repairs a
design: singular designs, separation and non-convergence raise :class:`ModelFitError`. The small input
helpers :func:`design_array` and :func:`binary_outcome` are shared with the tree adapters.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.linear_model import LogisticRegression
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from falls_ml.errors import BundleIntegrityError, ConfigError, DatasetValidationError, FallsMLError, PreprocessingMismatchError, DegenerateFitError
from falls_ml.logging_utils import get_logger
from falls_ml.models.base import ModelAdapter, coefficient_importance
from falls_ml.models.separation import SeparationError, stata_logit_omissions

log = get_logger(__name__)


class ModelFitError(DegenerateFitError):
    """A model could not be estimated reliably (singular design, separation, non-convergence)."""


# ---------------------------------------------------------------------- shared input validation
def design_array(model_name: str, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Float64 matrix and column names of a design DataFrame; non-numeric or non-finite values are refused."""
    if not isinstance(X, pd.DataFrame):
        raise PreprocessingMismatchError(f"{model_name}: expected a design DataFrame, got {type(X).__name__}")
    names = [str(c) for c in X.columns]
    if not names or len(set(names)) != len(names):
        raise PreprocessingMismatchError(f"{model_name}: design columns must be non-empty and unique")
    bad = [c for c in names if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    if bad:
        raise PreprocessingMismatchError(f"{model_name}: non-numeric design columns {bad[:5]}")
    Xa = X.to_numpy(dtype=np.float64, na_value=np.nan)
    n_bad = int((~np.isfinite(Xa)).sum())
    if n_bad:
        raise PreprocessingMismatchError(f"{model_name}: design matrix contains {n_bad} missing or non-finite values")
    return Xa, names


def binary_outcome(model_name: str, y: Any, n_rows: int) -> np.ndarray:
    """Outcome as int64 0/1 of shape (n_rows,) with both classes present."""
    yv = np.asarray(y)
    if yv.shape != (n_rows,):
        raise DatasetValidationError(f"{model_name}: y must have shape ({n_rows},), got {yv.shape}")
    if n_rows == 0 or not np.isin(yv, (0, 1)).all() or np.unique(yv).size < 2:
        raise DatasetValidationError(f"{model_name}: y must be binary 0/1 with both classes present")
    return yv.astype(np.int64)


def _merge_params(model_name: str, defaults: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        raise ConfigError(f"{model_name}: unknown params {unknown}; allowed {sorted(defaults)}")
    return {**defaults, **params}


def _real(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
        return None
    return float(value)


def _array(meta: Mapping[str, Any], key: str, size: int, directory: Path, model_name: str) -> np.ndarray:
    arr = np.asarray(meta[key], dtype=np.float64)
    if arr.shape != (size,):
        raise BundleIntegrityError(f"{model_name}: {key} in {directory} has shape {arr.shape}, expected ({size},)")
    return arr


def _solve_intercept(y: np.ndarray, offset: np.ndarray, start: float, model_name: str) -> float:
    """Unpenalised intercept b solving sum(y - expit(b + offset)) = 0 (damped Newton; unique since both classes occur)."""
    b = start
    for _ in range(100):
        p = expit(b + offset)
        step = float(np.clip(np.sum(y - p) / max(float(np.sum(p * (1.0 - p))), 1e-300), -1.0, 1.0))
        b += step
        if abs(step) < 1e-12:
            return b
    raise ModelFitError(f"{model_name}: intercept score equation did not converge")


# ---------------------------------------------------------------------- unpenalised logistic regression
class UnpenalizedLogistic(ModelAdapter):
    """Maximum-likelihood logistic regression (statsmodels ``Logit``, Newton, ``maxiter`` iterations).

    ``native_importance`` is |coefficient x column SD (ddof = 0)|, the coefficient of the standardised column.
    ``fit_diagnostics()['coefficients']`` holds the Wald table (SE, z, p, 95% CI, odds ratios).
    """

    name = "logistic_unpenalized"
    representation = "efalls_reference_coded"
    is_linear = True
    DEFAULT_PARAMS = {"maxiter": 100}

    def __init__(self, params: Mapping[str, Any] | None = None, *, random_state: int = 0):
        super().__init__(params, random_state=random_state)
        self.params = _merge_params(self.name, self.DEFAULT_PARAMS, self.params)
        maxiter = self.params["maxiter"]
        if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
            raise ConfigError(f"{self.name}: maxiter must be an integer >= 1, got {maxiter!r}")
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coefficient_table_: pd.DataFrame | None = None
        self.diagnostics_: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, *, groups: np.ndarray | None = None) -> Self:
        Xa_full, names = design_array(self.name, X)
        yv_full = binary_outcome(self.name, y, Xa_full.shape[0])
        # Stata ``logit`` semantics: omit constant columns and perfect predictors (with the rows they predict);
        # omitted terms get coefficient 0 and are recorded in fit_diagnostics (never silent).
        try:
            omissions = stata_logit_omissions(Xa_full, yv_full, names)
        except SeparationError as exc:
            raise ModelFitError(f"{self.name}: {exc}") from exc
        if omissions.any:
            log.warning("logistic_unpenalized_stata_omissions", extra_fields=omissions.to_dict())
        kept = list(omissions.kept_columns)
        Xa, yv = Xa_full[np.ix_(omissions.kept_rows, kept)], yv_full[omissions.kept_rows]
        kept_names = [names[j] for j in kept]
        A = np.column_stack([np.ones(Xa.shape[0]), Xa])
        rank = int(np.linalg.matrix_rank(A))
        if rank < A.shape[1]:
            constant = [kept_names[j] for j in np.flatnonzero(np.ptp(Xa, axis=0) == 0)]
            raise ModelFitError(f"{self.name}: design with intercept is singular (rank {rank} < {A.shape[1]} columns; "
                                f"constant columns {constant[:10]}); remove collinear columns explicitly")
        fit_warnings = (ConvergenceWarning, PerfectSeparationWarning, HessianInversionWarning)
        with warnings.catch_warnings():
            for category in fit_warnings:
                warnings.simplefilter("error", category)
            try:
                result = sm.Logit(yv, A).fit(method="newton", maxiter=int(self.params["maxiter"]), disp=False)
            except (*fit_warnings, PerfectSeparationError, np.linalg.LinAlgError) as exc:
                raise ModelFitError(f"{self.name}: maximum-likelihood fit failed: {exc}") from exc
        params = np.asarray(result.params, dtype=np.float64)
        se = np.asarray(result.bse, dtype=np.float64)
        if not result.mle_retvals.get("converged", False) or not (np.isfinite(params).all() and np.isfinite(se).all()):
            raise ModelFitError(f"{self.name}: Newton did not converge to finite estimates within {self.params['maxiter']} iterations")
        ci = np.asarray(result.conf_int(alpha=0.05), dtype=np.float64)
        table = pd.DataFrame({
            "term": ["intercept", *kept_names], "coefficient": params, "se": se, "z": params / se,
            "p_value": np.asarray(result.pvalues, dtype=np.float64), "ci_low": ci[:, 0], "ci_high": ci[:, 1],
            "odds_ratio": np.exp(params), "or_ci_low": np.exp(ci[:, 0]), "or_ci_high": np.exp(ci[:, 1]),
            "omitted": False,
        })
        omitted_names = [n for n in names if n not in set(kept_names)]
        if omitted_names:
            table = pd.concat([table, pd.DataFrame({"term": omitted_names, "coefficient": 0.0, "omitted": True})], ignore_index=True)
        self.coefficient_table_ = table
        coef = np.zeros(len(names))
        coef[kept] = params[1:]
        self.feature_names_ = names
        self.intercept_, self.coef_ = float(params[0]), coef
        self.scale_ = Xa_full.std(axis=0, ddof=0)
        self.diagnostics_ = {"method": "statsmodels Logit newton", "converged": True,
                             "n_iterations": int(result.mle_retvals.get("iterations", -1)),
                             "log_likelihood": float(result.llf), "n_obs": int(Xa.shape[0]), "n_events": int(yv.sum()),
                             "stata_logit_omissions": omissions.to_dict()}
        log.info("logistic_unpenalized_fitted", extra_fields=self.diagnostics_)
        return self

    def linear_predictor(self, X: pd.DataFrame) -> np.ndarray:
        self._check_columns(X)
        return self.intercept_ + design_array(self.name, X)[0] @ self.coef_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return expit(self.linear_predictor(X))

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        table = coefficient_importance(self.feature_names_, self.coef_, standardized_scale=False)
        table["native_importance"] = np.abs(self.coef_ * self.scale_)
        return table

    def fit_diagnostics(self) -> dict[str, Any]:
        self._check_fitted()
        return {**self.diagnostics_, "intercept": self.intercept_, "coefficients": self.coefficient_table_.copy()}

    def save(self, directory: Path) -> None:
        self._check_fitted()
        self._write_meta(Path(directory), extra={
            "intercept": self.intercept_, "coef": self.coef_, "scale": self.scale_, "diagnostics": self.diagnostics_,
            "coefficient_table": self.coefficient_table_.to_dict(orient="list")})

    @classmethod
    def load(cls, directory: Path) -> Self:
        directory = Path(directory)
        try:
            meta = cls._read_meta(directory)
            if not isinstance(meta, dict) or meta.get("adapter") != cls.name:
                raise BundleIntegrityError(f"{directory} does not hold a {cls.name!r} adapter")
            model = cls(meta["params"], random_state=meta["random_state"])
            names = list(meta["feature_names"])
            model.coef_ = _array(meta, "coef", len(names), directory, cls.name)
            model.scale_ = _array(meta, "scale", len(names), directory, cls.name)
            model.intercept_ = float(meta["intercept"])
            model.coefficient_table_ = pd.DataFrame(meta["coefficient_table"])
            model.diagnostics_ = dict(meta["diagnostics"])
        except (OSError, ValueError, KeyError, TypeError, ConfigError) as exc:
            raise BundleIntegrityError(f"{cls.name}: cannot load adapter from {directory}: {exc}") from exc
        model.feature_names_ = names
        return model


# ---------------------------------------------------------------------- elastic-net logistic regression
class ElasticNetLogistic(ModelAdapter):
    """Elastic-net logistic regression: scikit-learn ``LogisticRegression(solver='saga', C, l1_ratio)``.

    Columns are standardised internally (mean, SD with ddof = 0; constant columns get coefficient 0) and the
    coefficients are back-transformed to the original design scale. ``native_importance`` is
    |standardised coefficient|. Non-convergence within ``MAX_ITER`` epochs raises :class:`ModelFitError`.

    The unpenalised intercept is re-solved exactly given the saga coefficients (score equation
    ``sum(y - p) = 0``): saga's stopping rule ignores the intercept and stops after one epoch when every
    coefficient stays at zero, which would otherwise leave a strongly penalised model miscalibrated. For a
    converged fit the shift is of the order of ``TOL`` (recorded as ``intercept_refit_shift``).
    """

    name = "elastic_net_logistic"
    representation = "efalls_reference_coded"
    is_linear = True
    DEFAULT_PARAMS = {"C": 1.0, "l1_ratio": 0.5}
    TOL = 1e-6
    MAX_ITER = 20000

    def __init__(self, params: Mapping[str, Any] | None = None, *, random_state: int = 0):
        super().__init__(params, random_state=random_state)
        self.params = _merge_params(self.name, self.DEFAULT_PARAMS, self.params)
        C, l1_ratio = _real(self.params["C"]), _real(self.params["l1_ratio"])
        if C is None or C <= 0 or l1_ratio is None or not 0.0 <= l1_ratio <= 1.0:
            raise ConfigError(f"{self.name}: C must be a finite number > 0 and l1_ratio in [0, 1], got {self.params}")
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_standardized_: float | None = None
        self.coef_standardized_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, *, groups: np.ndarray | None = None) -> Self:
        Xa, names = design_array(self.name, X)
        yv = binary_outcome(self.name, y, Xa.shape[0])
        constant = np.ptp(Xa, axis=0) == 0
        mean = Xa.mean(axis=0)
        scale = np.where(constant, 1.0, Xa.std(axis=0, ddof=0))
        Z = (Xa - mean) / scale
        Z[:, constant] = 0.0
        if constant.any():
            log.warning("elastic_net_constant_columns", extra_fields={"columns": [names[j] for j in np.flatnonzero(constant)]})
        clf = LogisticRegression(C=float(self.params["C"]), l1_ratio=float(self.params["l1_ratio"]), solver="saga",
                                 tol=self.TOL, max_iter=self.MAX_ITER, random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SklearnConvergenceWarning)
            try:
                clf.fit(Z, yv)
            except SklearnConvergenceWarning as exc:
                raise ModelFitError(f"{self.name}: saga did not converge (tol={self.TOL}, max_iter={self.MAX_ITER}, "
                                    f"params={self.params}): {exc}") from exc
        coef_std = clf.coef_[0].astype(np.float64)
        coef_std[constant] = 0.0
        coef = coef_std / scale
        saga_intercept = float(clf.intercept_[0])
        self.feature_names_ = names
        self.mean_, self.scale_ = mean, scale
        self.coef_standardized_ = coef_std
        self.intercept_standardized_ = _solve_intercept(yv, Z @ coef_std, saga_intercept, self.name)
        self.coef_ = coef
        self.intercept_ = float(self.intercept_standardized_ - np.sum(coef * mean))
        self.diagnostics_ = {"solver": "saga", "C": float(self.params["C"]), "l1_ratio": float(self.params["l1_ratio"]),
                             "tol": self.TOL, "max_iter": self.MAX_ITER, "n_iter": int(np.max(clf.n_iter_)),
                             "intercept_refit_shift": self.intercept_standardized_ - saga_intercept,
                             "n_selected": int(np.count_nonzero(coef)), "seed": self.random_state,
                             "constant_columns": [names[j] for j in np.flatnonzero(constant)]}
        log.info("elastic_net_fitted", extra_fields={k: v for k, v in self.diagnostics_.items() if k != "constant_columns"})
        return self

    def linear_predictor(self, X: pd.DataFrame) -> np.ndarray:
        self._check_columns(X)
        return self.intercept_ + design_array(self.name, X)[0] @ self.coef_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return expit(self.linear_predictor(X))

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        table = coefficient_importance(self.feature_names_, self.coef_, standardized_scale=False)
        table["native_importance"] = np.abs(self.coef_standardized_)
        return table

    def fit_diagnostics(self) -> dict[str, Any]:
        self._check_fitted()
        return {**self.diagnostics_, "intercept": self.intercept_, "standardized_intercept": self.intercept_standardized_,
                "standardized_coefficients": pd.Series(self.coef_standardized_, index=self.feature_names_,
                                                       name="standardized_coefficient")}

    @classmethod
    def default_search_space(cls) -> dict[str, list[Any]]:
        return {"C": [0.001, 0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.9]}

    def save(self, directory: Path) -> None:
        self._check_fitted()
        self._write_meta(Path(directory), extra={
            "intercept": self.intercept_, "coef": self.coef_, "intercept_standardized": self.intercept_standardized_,
            "coef_standardized": self.coef_standardized_, "mean": self.mean_, "scale": self.scale_,
            "diagnostics": self.diagnostics_})

    @classmethod
    def load(cls, directory: Path) -> Self:
        directory = Path(directory)
        try:
            meta = cls._read_meta(directory)
            if not isinstance(meta, dict) or meta.get("adapter") != cls.name:
                raise BundleIntegrityError(f"{directory} does not hold a {cls.name!r} adapter")
            model = cls(meta["params"], random_state=meta["random_state"])
            names = list(meta["feature_names"])
            for key in ("coef", "coef_standardized", "mean", "scale"):
                setattr(model, f"{key}_", _array(meta, key, len(names), directory, cls.name))
            model.intercept_ = float(meta["intercept"])
            model.intercept_standardized_ = float(meta["intercept_standardized"])
            model.diagnostics_ = dict(meta["diagnostics"])
        except (OSError, ValueError, KeyError, TypeError, ConfigError) as exc:
            raise BundleIntegrityError(f"{cls.name}: cannot load adapter from {directory}: {exc}") from exc
        model.feature_names_ = names
        return model
