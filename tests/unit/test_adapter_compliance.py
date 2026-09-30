"""Adapter interface compliance for every registered model (architecture §3.2, §6; spec §5, D-01, D-14).

Each adapter is fitted on the design its declared representation produces from a synthetic modelling frame built
per ``configs/features/efalls_v1.yaml`` (software test data only), then checked for: probability output,
determinism, expit(linear predictor) == risk for linear models, the importance-table contract, save/load
identity and refusal of a design whose columns were reordered.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from falls_ml.config import FractionalPolynomialSection
from falls_ml.errors import NotFittedError, PreprocessingMismatchError
from falls_ml.features.preprocessing import REPRESENTATIONS, Preprocessor
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.models.base import IMPORTANCE_COLUMNS, ModelAdapter
from falls_ml.models.registry import available_models, get_adapter_class
from falls_ml.pipeline import FittedPipeline
from falls_ml.seeding import rng_for
from falls_ml.tuning import expand_search_space

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"
MODELS = available_models()
PARAMS: dict[str, dict[str, Any]] = {
    "efalls_published": {"config": "configs/models/efalls_published.yaml", "sex_parameterisation": "lp_c_box_s3_1"},
    "lasso_logistic_cv": {"n_lambda": 20, "cv_folds": 3},
}
RANDOM_STATE = 20260915


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return load_feature_spec(SPEC_PATH)


def modelling_frame(spec: FeatureSpec, n: int, component: str) -> pd.DataFrame:
    """Deterministic synthetic modelling-dataset rows (identifiers, dates, outcome and all 78 eFalls predictors)."""
    rng = rng_for(RANDOM_STATE, component)
    alcohol = list(spec.get("alcohol_category").levels)
    data: dict[str, Any] = {
        "research_id": pd.Series([f"{component}-{i:05d}" for i in range(n)], dtype="str"),
        "index_date": pd.Timestamp("2018-04-01"),
        "predictor_max_record_date": pd.Timestamp("2018-03-31"),
        "age_years": rng.uniform(65.0, 100.0, n),
        "sex": pd.Series(rng.choice(["female", "male"], n), dtype="str"),
        "polypharmacy_count_120d": rng.poisson(6.0, n).astype("int64"),
        "bmi_value": np.where(rng.random(n) < 0.15, np.nan, rng.uniform(15.0, 45.0, n)),
        "smoking_status": pd.Series(rng.choice(np.array(["never", "ex", "current", None], dtype=object), n), dtype="str"),
        "alcohol_category": pd.Series(rng.choice(np.array([*alcohol, None], dtype=object), n), dtype="str"),
    }
    for name, prevalence in zip(spec.binary_names(), rng.uniform(0.05, 0.3, len(spec.binary_names()))):
        data[name] = rng.binomial(1, prevalence, n).astype("int8")
    df = pd.DataFrame(data)
    eta = (-2.0 + 0.04 * (df["age_years"] - 80.0) + 0.3 * np.log((df["polypharmacy_count_120d"] + 1) / 10)
           + 0.7 * df["falls"] + 0.5 * df["fracture"] + 0.4 * df["dementia"])
    df["outcome_12m"] = rng.binomial(1, expit(eta.to_numpy())).astype("int8")
    return df


@pytest.fixture(scope="module")
def frames(spec: FeatureSpec) -> dict[str, pd.DataFrame]:
    return {"train": modelling_frame(spec, 1500, "compliance_train"), "new": modelling_frame(spec, 400, "compliance_new")}


def fit_adapter(name: str, spec: FeatureSpec, train: pd.DataFrame) -> tuple[Preprocessor, ModelAdapter]:
    cls = get_adapter_class(name)
    y = train["outcome_12m"].to_numpy(dtype=np.int64)
    pre = Preprocessor(spec, cls.representation, fp=FractionalPolynomialSection(mode="fixed_published")).fit(train, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        model = cls(PARAMS.get(name), random_state=RANDOM_STATE).fit(pre.transform(train), y, groups=train["research_id"].to_numpy())
    return pre, model


@pytest.fixture(scope="module", params=MODELS)
def fitted(request: pytest.FixtureRequest, spec: FeatureSpec, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    name = request.param
    pre, model = fit_adapter(name, spec, frames["train"])
    return {"name": name, "pre": pre, "model": model, "X": pre.transform(frames["new"])}


def test_registry_entries_are_consistent():
    assert {"efalls_published", "lasso_logistic_cv", "logistic_unpenalized", "elastic_net_logistic", "random_forest",
            "hist_gradient_boosting"} <= set(MODELS)
    for name in MODELS:
        cls = get_adapter_class(name)
        assert cls.name == name and cls.representation in REPRESENTATIONS
        space = cls.default_search_space()
        if space is not None:
            assert expand_search_space(space, "grid") and all(isinstance(v, list) and v for v in space.values())


@pytest.mark.parametrize("name", [m for m in MODELS if get_adapter_class(m).requires_fit])
def test_unfitted_adapter_refuses_to_predict(name, frames, spec):
    cls = get_adapter_class(name)
    pre = Preprocessor(spec, cls.representation, fp=FractionalPolynomialSection(mode="fixed_published")).fit(frames["train"])
    with pytest.raises(NotFittedError):
        cls(PARAMS.get(name), random_state=0).predict_proba(pre.transform(frames["new"]))


def test_predict_proba_shape_and_range(fitted):
    p = fitted["model"].predict_proba(fitted["X"])
    assert isinstance(p, np.ndarray) and p.shape == (len(fitted["X"]),)
    assert np.isfinite(p).all() and ((p >= 0.0) & (p <= 1.0)).all()
    assert np.unique(p).size > 1


def test_fit_is_deterministic(fitted, spec, frames):
    _, again = fit_adapter(fitted["name"], spec, frames["train"])
    np.testing.assert_array_equal(again.predict_proba(fitted["X"]), fitted["model"].predict_proba(fitted["X"]))


def test_linear_predictor_contract(fitted):
    model, X = fitted["model"], fitted["X"]
    lp = model.linear_predictor(X)
    if model.is_linear:
        assert lp.shape == (len(X),)
        np.testing.assert_allclose(expit(lp), model.predict_proba(X), rtol=0, atol=1e-12)
    else:
        assert lp is None


def test_feature_importance_columns(fitted):
    model = fitted["model"]
    table = model.get_feature_importance()
    assert list(table.columns) == IMPORTANCE_COLUMNS
    assert table["feature"].tolist() == list(fitted["X"].columns)
    assert isinstance(model.fit_diagnostics(), dict)


def test_save_load_identity(fitted, tmp_path):
    model, X = fitted["model"], fitted["X"]
    model.save(tmp_path / "adapter")
    loaded = type(model).load(tmp_path / "adapter")
    assert loaded.name == model.name and loaded.feature_names_ == model.feature_names_
    assert loaded.get_params() == model.get_params()
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
    if model.is_linear:
        np.testing.assert_array_equal(loaded.linear_predictor(X), model.linear_predictor(X))
    pd.testing.assert_frame_equal(loaded.get_feature_importance(), model.get_feature_importance())


def test_reordered_or_missing_columns_are_refused(fitted):
    model, X = fitted["model"], fitted["X"]
    with pytest.raises(PreprocessingMismatchError):
        model.predict_proba(X[list(reversed(X.columns))])
    with pytest.raises(PreprocessingMismatchError):
        model.predict_proba(X.drop(columns=X.columns[-1]))


def test_fitted_pipeline_uses_the_same_preprocessing(fitted, spec, frames):
    pipeline = FittedPipeline(spec=spec, preprocessor=fitted["pre"], model=fitted["model"])
    np.testing.assert_array_equal(pipeline.predict_proba(frames["new"]), fitted["model"].predict_proba(fitted["X"]))
