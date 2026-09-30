"""Histogram gradient-boosting adapter (L4_alternative; architecture §3.2).

Uses scikit-learn ``HistGradientBoostingClassifier`` on the ``efalls_raw`` design. Early stopping is fixed off:
it would carve an internal validation set out of the training rows by random row sampling, which is not
patient-grouped (D-15); the number of boosting iterations is tuned on the validation partition instead.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier

from falls_ml.models.random_forest import SklearnEnsembleAdapter


class HistGradientBoostingModel(SklearnEnsembleAdapter):
    """``HistGradientBoostingClassifier`` with log-loss. No native importance (``native_importance`` is NaN)."""

    name = "hist_gradient_boosting"
    estimator_class = HistGradientBoostingClassifier
    DEFAULT_PARAMS = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "max_depth": None,
                      "min_samples_leaf": 100, "l2_regularization": 0.0, "max_features": 1.0}
    FIXED_PARAMS = {"early_stopping": False}

    def fit_diagnostics(self) -> dict[str, Any]:
        return {**super().fit_diagnostics(), "n_iter": int(self.estimator_.n_iter_)}

    @classmethod
    def default_search_space(cls) -> dict[str, list[Any]]:
        return {"learning_rate": [0.03, 0.1], "max_iter": [200, 500], "max_leaf_nodes": [15, 31],
                "min_samples_leaf": [50, 200], "l2_regularization": [0.0, 1.0], "max_features": [0.5, 1.0]}
