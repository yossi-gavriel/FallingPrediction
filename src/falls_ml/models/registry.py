"""Name → adapter registry. Adapters are imported lazily so optional heavy code loads only when used."""

from __future__ import annotations

import importlib

from falls_ml.errors import ConfigError
from falls_ml.models.base import ModelAdapter

_ADAPTERS: dict[str, str] = {
    "efalls_published": "falls_ml.models.published_efalls:PublishedEfallsModel",
    "lasso_logistic_cv": "falls_ml.models.lasso_cv:LassoLogisticCV",
    "logistic_unpenalized": "falls_ml.models.logistic:UnpenalizedLogistic",
    "elastic_net_logistic": "falls_ml.models.logistic:ElasticNetLogistic",
    "random_forest": "falls_ml.models.random_forest:RandomForestModel",
    "hist_gradient_boosting": "falls_ml.models.hist_gbm:HistGradientBoostingModel",
}


def available_models() -> list[str]:
    return sorted(_ADAPTERS)


def get_adapter_class(name: str) -> type[ModelAdapter]:
    try:
        target = _ADAPTERS[name]
    except KeyError as exc:
        raise ConfigError(f"Unknown model {name!r}; available: {available_models()}") from exc
    module_name, cls_name = target.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    if not issubclass(cls, ModelAdapter) or cls.name != name:
        raise ConfigError(f"Registry entry {name!r} points to an invalid adapter {target}")
    return cls
