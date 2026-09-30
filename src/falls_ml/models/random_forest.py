"""Random-forest adapter and the shared scikit-learn tree-ensemble base (L4_alternative; architecture §3.2).

Tree ensembles consume the ``efalls_raw`` design (raw age, raw polypharmacy count, all-levels indicators).
They are non-linear: ``linear_predictor`` returns ``None`` and no coefficients are reported.

Persistence pickles the fitted estimator next to ``adapter.json``. Pickles execute code when loaded, so only load
adapter directories from a model bundle whose SHA-256 manifest has been verified; loading also refuses a pickle
written by a different scikit-learn version.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils._param_validation import InvalidParameterError

from falls_ml.errors import BundleIntegrityError, ConfigError
from falls_ml.logging_utils import get_logger
from falls_ml.models.base import ModelAdapter, empty_importance
from falls_ml.models.logistic import binary_outcome, design_array

log = get_logger(__name__)

ESTIMATOR_FILE = "estimator.pkl"


class SklearnEnsembleAdapter(ModelAdapter):
    """Shared fit / predict / persistence for scikit-learn tree-ensemble classifiers (not registered itself).

    Subclasses set ``name``, ``estimator_class``, ``DEFAULT_PARAMS`` (tunable estimator arguments) and
    ``FIXED_PARAMS`` (non-tunable arguments); ``random_state`` is always passed through.
    """

    representation = "efalls_raw"
    is_linear = False
    estimator_class: ClassVar[type[ClassifierMixin]]
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {}
    FIXED_PARAMS: ClassVar[dict[str, Any]] = {}

    def __init__(self, params: Mapping[str, Any] | None = None, *, random_state: int = 0):
        super().__init__(params, random_state=random_state)
        unknown = sorted(set(self.params) - set(self.DEFAULT_PARAMS))
        if unknown:
            raise ConfigError(f"{self.name}: unknown params {unknown}; allowed {sorted(self.DEFAULT_PARAMS)}")
        self.params = {**self.DEFAULT_PARAMS, **self.params}
        self.estimator_: Any = None

    def _native_importance(self) -> np.ndarray:
        return np.full(len(self.feature_names_ or []), np.nan)

    def fit(self, X: pd.DataFrame, y: np.ndarray, *, groups: np.ndarray | None = None) -> Self:
        Xa, names = design_array(self.name, X)
        yv = binary_outcome(self.name, y, Xa.shape[0])
        estimator = self.estimator_class(**self.params, **self.FIXED_PARAMS, random_state=self.random_state)
        try:
            estimator.fit(Xa, yv)
        except InvalidParameterError as exc:
            raise ConfigError(f"{self.name}: invalid params {self.params}: {exc}") from exc
        self.estimator_ = estimator
        self.feature_names_ = names
        log.info(f"{self.name}_fitted", extra_fields={"params": self.params, "n_rows": int(Xa.shape[0]),
                                                      "n_features": len(names), "seed": self.random_state})
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_columns(X)
        return self.estimator_.predict_proba(design_array(self.name, X)[0])[:, 1]  # classes_ == [0, 1] (binary_outcome)

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        table = empty_importance(list(self.feature_names_))
        table["native_importance"] = self._native_importance()
        return table

    def fit_diagnostics(self) -> dict[str, Any]:
        self._check_fitted()
        return {"estimator": self.estimator_class.__name__, "params": dict(self.params), "fixed_params": dict(self.FIXED_PARAMS),
                "n_features": len(self.feature_names_), "seed": self.random_state}

    def save(self, directory: Path) -> None:
        self._check_fitted()
        directory = Path(directory)
        self._write_meta(directory, extra={"estimator_file": ESTIMATOR_FILE, "sklearn_version": sklearn.__version__})
        with (directory / ESTIMATOR_FILE).open("wb") as handle:
            pickle.dump(self.estimator_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, directory: Path) -> Self:
        directory = Path(directory)
        try:
            meta = cls._read_meta(directory)
        except (OSError, ValueError) as exc:
            raise BundleIntegrityError(f"{cls.name}: cannot read adapter.json in {directory}: {exc}") from exc
        if not isinstance(meta, dict) or meta.get("adapter") != cls.name:
            raise BundleIntegrityError(f"{directory} does not hold a {cls.name!r} adapter")
        if meta.get("sklearn_version") != sklearn.__version__:
            raise BundleIntegrityError(f"{cls.name}: estimator pickled with scikit-learn {meta.get('sklearn_version')}, "
                                       f"running {sklearn.__version__}; refit or use the matching environment")
        try:
            model = cls(meta["params"], random_state=meta["random_state"])
            names = list(meta["feature_names"])
            with (directory / ESTIMATOR_FILE).open("rb") as handle:
                estimator = pickle.load(handle)
        except (OSError, KeyError, TypeError, ValueError, IndexError, AttributeError, ImportError, EOFError,
                pickle.UnpicklingError, ConfigError) as exc:  # corrupt or foreign pickles fail in many ways
            raise BundleIntegrityError(f"{cls.name}: cannot load adapter from {directory}: {exc}") from exc
        if not isinstance(estimator, cls.estimator_class) or getattr(estimator, "n_features_in_", None) != len(names):
            raise BundleIntegrityError(f"{cls.name}: {ESTIMATOR_FILE} is not a fitted {cls.estimator_class.__name__} "
                                       f"with {len(names)} features")
        model.estimator_ = estimator
        model.feature_names_ = names
        return model


class RandomForestModel(SklearnEnsembleAdapter):
    """``RandomForestClassifier`` (single-threaded for determinism).

    ``native_importance`` is the impurity-based (mean decrease in impurity) importance computed on training data.
    It is biased towards continuous and high-cardinality columns; prefer validation permutation importance.
    """

    name = "random_forest"
    estimator_class = RandomForestClassifier
    DEFAULT_PARAMS = {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 50, "max_features": "sqrt"}
    FIXED_PARAMS = {"n_jobs": 1}

    def _native_importance(self) -> np.ndarray:
        return np.asarray(self.estimator_.feature_importances_, dtype=np.float64)

    @classmethod
    def default_search_space(cls) -> dict[str, list[Any]]:
        return {"n_estimators": [300], "max_depth": [4, 8, None], "min_samples_leaf": [20, 100], "max_features": ["sqrt", 0.33]}
