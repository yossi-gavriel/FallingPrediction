"""Unit tests for coefficient tables and grouped permutation importance (architecture §4 step 7; ARTIFACT_SCHEMAS
coefficients.csv / feature_importance.csv; spec D-10, D-14). Synthetic data only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from falls_ml.config import experiment_config_from_dict
from falls_ml.errors import ConfigError
from falls_ml.evaluation import importance as importance_module
from falls_ml.evaluation import metrics as M
from falls_ml.evaluation.importance import (
    COEFFICIENT_COLUMNS,
    FEATURE_IMPORTANCE_COLUMNS,
    PERMUTATION_COLUMNS,
    ImportanceError,
    build_feature_importance_table,
    coefficient_table,
    permutation_importance_grouped,
)
from falls_ml.features.spec import load_feature_spec
from falls_ml.models.base import ModelAdapter, coefficient_importance, empty_importance
from falls_ml.pipeline import FittedPipeline, fit_pipeline
from falls_ml.seeding import rng_for

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"
SEED = 20260915


# ---------------------------------------------------------------------- stubs
class StubSpec:
    outcome = SimpleNamespace(name="y")
    identifier_columns = ("research_id",)

    def __init__(self, predictors: list[str]):
        self._predictors = list(predictors)

    def predictor_names(self) -> list[str]:
        return list(self._predictors)


class StubPreprocessor:
    """Design column 'f' copies raw 'f'; design column 'f=level' is the indicator of raw 'f' == level."""

    def __init__(self, design_columns: list[str]):
        self.columns = list(design_columns)

    def raw_feature_of(self, column: str) -> str:
        return column.split("=")[0]

    def design_columns(self) -> list[str]:
        return list(self.columns)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.columns:
            raw, _, level = c.partition("=")
            out[c] = (df[raw].astype(object) == level).astype(float).to_numpy() if level else df[raw].to_numpy(dtype=float)
        return pd.DataFrame(out, index=df.index)


class FixedLinear(ModelAdapter):
    name = "fixed_linear_stub"
    representation = "stub"
    is_linear = True

    def __init__(self, coefs: dict[str, float], intercept: float = -0.5, standardized: Any = None):
        super().__init__()
        self.coef = pd.Series(coefs, dtype=float)
        self.intercept = intercept
        self.standardized = standardized
        self.feature_names_ = list(coefs)

    def fit(self, X, y, *, groups=None):
        return self

    def linear_predictor(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + X[self.feature_names_].to_numpy() @ self.coef.to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return expit(self.linear_predictor(X))

    def get_feature_importance(self) -> pd.DataFrame:
        return coefficient_importance(self.feature_names_, self.coef.to_numpy(), standardized_scale=False)

    def fit_diagnostics(self) -> dict[str, Any]:
        return {} if self.standardized is None else {"standardized_coefficients": self.standardized}

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        raise NotImplementedError


class FixedNonLinear(FixedLinear):
    name = "fixed_nonlinear_stub"
    is_linear = False

    def linear_predictor(self, X):
        return None

    def predict_proba(self, X):
        return expit(self.intercept + X[self.feature_names_].to_numpy() @ self.coef.to_numpy())

    def get_feature_importance(self):
        return empty_importance(self.feature_names_)


class SquareCalibrator:
    def transform(self, p, lp=None):
        return np.asarray(p) ** 2


DESIGN = ["x_strong", "x_noise", "cat=b", "cat=c"]
PREDICTORS = ["x_strong", "x_noise", "cat"]


def make_pipeline(coefs: dict[str, float] | None = None, *, model_cls=FixedLinear, standardized=None, design=DESIGN) -> FittedPipeline:
    coefs = coefs or {"x_strong": 1.5, "x_noise": 0.0, "cat=b": 0.8, "cat=c": 0.0}
    model = model_cls(coefs, standardized=standardized)
    return FittedPipeline(spec=StubSpec(sorted({c.split("=")[0] for c in design})), preprocessor=StubPreprocessor(design), model=model)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    n = 2000
    rng = rng_for(SEED, "test_importance_frame")
    df = pd.DataFrame({"research_id": pd.Series([f"P{i:05d}" for i in range(n)], dtype="str"),
                       "x_strong": rng.normal(size=n), "x_noise": rng.normal(size=n),
                       "cat": pd.Series(rng.choice(["a", "b", "c"], n), dtype="str")})
    eta = -0.5 + 1.5 * df["x_strong"].to_numpy() + 0.8 * (df["cat"] == "b").to_numpy()
    df["y"] = rng.binomial(1, expit(eta)).astype("int8")
    return df


# ---------------------------------------------------------------------- permutation importance
class TestPermutationImportance:
    def test_noise_feature_is_zero_and_strong_feature_dominates(self, frame):
        pipe = make_pipeline()
        y = frame["y"].to_numpy()
        for metric in ("auroc", "brier"):
            table = permutation_importance_grouped(pipe, frame, y, features=PREDICTORS, n_repeats=5, seed=1, metric=metric)
            assert list(table.columns) == PERMUTATION_COLUMNS
            imp = table.set_index("feature")["permutation_importance_mean"]
            assert imp["x_noise"] == 0.0 and table.set_index("feature").loc["x_noise", "permutation_importance_std"] == 0.0
            assert imp["x_strong"] > imp["cat"] > 0.0
            assert (table["metric"] == metric).all()
        baseline = table["baseline_metric"].iloc[0]
        assert baseline == pytest.approx(M.brier(y, pipe.predict_proba(frame)))

    def test_small_noise_coefficient_gives_near_zero_importance(self, frame):
        pipe = make_pipeline({"x_strong": 1.5, "x_noise": 0.05, "cat=b": 0.8, "cat=c": 0.0})
        table = permutation_importance_grouped(pipe, frame, frame["y"].to_numpy(), features=["x_strong", "x_noise"],
                                               n_repeats=10, seed=2).set_index("feature")
        assert abs(table.loc["x_noise", "permutation_importance_mean"]) < 0.01
        assert table.loc["x_strong", "permutation_importance_mean"] > 20 * abs(table.loc["x_noise", "permutation_importance_mean"])

    def test_raw_feature_permutation_reruns_full_pipeline_with_documented_rng(self, frame):
        pipe = make_pipeline()
        y = frame["y"].to_numpy()
        table = permutation_importance_grouped(pipe, frame, y, features=["cat"], n_repeats=2, seed=7)
        baseline = M.auroc(y, pipe.predict_proba(frame))
        drops = []
        for r in range(2):
            permuted = frame.copy()
            permuted["cat"] = frame["cat"].to_numpy()[rng_for(7, f"perm:cat:{r}").permutation(len(frame))]
            X = pipe.design(permuted)
            assert ((X["cat=b"] + X["cat=c"]) <= 1).all()  # indicators of one raw feature move together
            drops.append(baseline - M.auroc(y, pipe.predict_proba(permuted)))
        assert table.loc[0, "permutation_importance_mean"] == pytest.approx(np.mean(drops), abs=1e-15)
        assert table.loc[0, "permutation_importance_std"] == pytest.approx(np.std(drops), abs=1e-15)

    def test_deterministic_and_input_frame_unchanged(self, frame):
        pipe = make_pipeline()
        before = frame.copy()
        kwargs = dict(features=PREDICTORS, n_repeats=3, metric="auroc")
        a = permutation_importance_grouped(pipe, frame, frame["y"].to_numpy(), seed=11, **kwargs)
        b = permutation_importance_grouped(pipe, frame, frame["y"].to_numpy(), seed=11, **kwargs)
        c = permutation_importance_grouped(pipe, frame, frame["y"].to_numpy(), seed=12, **kwargs)
        pd.testing.assert_frame_equal(a, b)
        assert not np.allclose(a["permutation_importance_mean"].iloc[:1], c["permutation_importance_mean"].iloc[:1])
        pd.testing.assert_frame_equal(frame, before)

    def test_calibrated_predictions_are_used_when_requested(self, frame):
        pipe = make_pipeline()
        pipe.calibrator = SquareCalibrator()
        y = frame["y"].to_numpy()
        table = permutation_importance_grouped(pipe, frame, y, features=["x_strong"], n_repeats=1, seed=1, metric="brier",
                                               calibrated=True)
        assert table.loc[0, "baseline_metric"] == pytest.approx(M.brier(y, pipe.predict_proba(frame) ** 2))

    @pytest.mark.parametrize("kwargs, error", [
        ({"metric": "pr_auc"}, ConfigError),
        ({"n_repeats": 0}, ConfigError),
        ({"features": ["x_strong", "x_strong"]}, ConfigError),
        ({"features": ["research_id"]}, ConfigError),
    ])
    def test_invalid_arguments(self, frame, kwargs, error):
        args = {"features": ["x_strong"], "n_repeats": 1, "seed": 1, **kwargs}
        with pytest.raises(error):
            permutation_importance_grouped(make_pipeline(), frame, frame["y"].to_numpy(), **args)

    def test_single_class_auroc_is_refused(self, frame):
        with pytest.raises(ImportanceError):
            permutation_importance_grouped(make_pipeline(), frame, np.zeros(len(frame), dtype=int), features=["x_strong"],
                                           n_repeats=1, seed=1)


# ---------------------------------------------------------------------- coefficient table
class TestCoefficientTable:
    def test_raw_coefficients_are_not_ranked_without_a_standardized_scale(self):
        # x_noise has the larger raw coefficient; without a common scale no rank may be implied
        pipe = make_pipeline({"x_strong": 0.001, "x_noise": 2.0, "cat=b": 0.8, "cat=c": 0.0})
        table = coefficient_table(pipe)
        assert list(table.columns) == COEFFICIENT_COLUMNS
        assert table["rank"].isna().all() and table["standardized_coefficient"].isna().all()
        assert table["design_column"].tolist() == DESIGN
        assert table["feature"].tolist() == ["x_strong", "x_noise", "cat", "cat"]
        assert not table["relative_to_reference"].any()
        np.testing.assert_allclose(table["odds_ratio"], np.exp(table["coefficient"]))
        assert table["selected"].tolist() == [True, True, True, False]
        assert table["direction"].tolist() == ["increases_risk", "increases_risk", "increases_risk", "none"]

    def test_rank_follows_absolute_standardized_coefficient(self):
        standardized = pd.Series({"cat=c": 0.0, "x_noise": 0.1, "cat=b": -0.4, "x_strong": 2.0})  # order differs from design
        pipe = make_pipeline({"x_strong": 0.002, "x_noise": 5.0, "cat=b": -0.8, "cat=c": 0.0}, standardized=standardized)
        table = coefficient_table(pipe).set_index("design_column")
        assert table["rank"].to_dict() == {"x_strong": 1, "cat=b": 2, "x_noise": 3, "cat=c": 4}
        assert table.loc["x_strong", "standardized_coefficient"] == 2.0

    def test_standardized_coefficients_reported_as_mapping(self):
        standardized = {"x_strong": -1.0, "x_noise": 0.5, "cat=b": 2.0, "cat=c": 0.0}
        table = coefficient_table(make_pipeline(standardized=standardized)).set_index("design_column")
        assert table["rank"].to_dict() == {"cat=b": 1, "x_strong": 2, "x_noise": 3, "cat=c": 4}

    def test_floating_point_residue_is_reported_as_selected_and_flagged(self, monkeypatch):
        warnings_logged: list[tuple[str, dict]] = []
        monkeypatch.setattr(importance_module, "log", SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda event, extra_fields=None: warnings_logged.append((event, extra_fields))))
        coefs = {"x_strong": 1.5, "x_noise": 0.0, "cat=b": -0.8, "cat=c": 3e-16}  # 'cat=c': collinear all-levels residue
        table = coefficient_table(make_pipeline(coefs, standardized=pd.Series(coefs))).set_index("design_column")
        assert bool(table.loc["cat=c", "selected"])  # contract (models.base): selected = coefficient != 0
        assert dict(warnings_logged)["coefficients_at_numerical_zero"]["design_columns"] == ["cat=c"]

    def test_mismatched_standardized_coefficients_fail(self):
        pipe = make_pipeline(standardized=pd.Series({"x_strong": 1.0}))
        with pytest.raises(ImportanceError):
            coefficient_table(pipe)

    def test_non_linear_model_gives_header_only_table(self):
        table = coefficient_table(make_pipeline(model_cls=FixedNonLinear))
        assert table.empty and list(table.columns) == COEFFICIENT_COLUMNS


# ---------------------------------------------------------------------- feature importance table
class TestFeatureImportanceTable:
    PERM = pd.DataFrame({"feature": ["x_noise", "cat", "x_strong"], "permutation_importance_mean": [0.0, 0.05, 0.2],
                         "permutation_importance_std": [0.0, 0.01, 0.02], "baseline_metric": 0.8, "metric": "auroc"})
    STABILITY = pd.DataFrame({"feature": DESIGN, "raw_feature": ["x_strong", "x_noise", "cat", "cat"],
                              "selection_frequency": [1.0, 0.3, 0.6, 0.9]})

    def test_linear_model_rows_per_raw_feature(self):
        table = build_feature_importance_table(make_pipeline(), self.PERM, self.STABILITY, model_name="stub")
        assert list(table.columns) == FEATURE_IMPORTANCE_COLUMNS
        assert table["feature"].tolist() == ["x_strong", "cat", "x_noise"]
        assert table["importance_rank"].tolist() == [1, 2, 3]
        rows = table.set_index("feature")
        assert rows.loc["x_strong", "coefficient"] == 1.5 and rows.loc["x_strong", "odds_ratio"] == pytest.approx(np.exp(1.5))
        assert np.isnan(rows.loc["cat", "coefficient"]) and np.isnan(rows.loc["cat", "odds_ratio"])  # two design columns
        assert rows.loc["cat", "selection_frequency"] == 0.9  # max over its design columns
        assert table["shap_mean_abs"].isna().all() and (table["model"] == "stub").all()

    def test_non_linear_model_and_no_stability(self):
        table = build_feature_importance_table(make_pipeline(model_cls=FixedNonLinear), self.PERM, None, model_name="rf")
        assert table["coefficient"].isna().all() and table["selection_frequency"].isna().all()
        assert table["feature"].tolist() == ["x_strong", "cat", "x_noise"]

    def test_duplicated_permutation_features_fail(self):
        with pytest.raises(ConfigError):
            build_feature_importance_table(make_pipeline(), pd.concat([self.PERM, self.PERM]), model_name="stub")


# ---------------------------------------------------------------------- real pipeline
def test_real_fit_pipeline_permutation_and_tables():
    spec = load_feature_spec(SPEC_PATH).subset(["age_years", "sex", "polypharmacy_count_120d", "bmi_value", "falls", "dementia"])
    n = 1200
    rng = rng_for(SEED, "test_importance_real")
    df = pd.DataFrame({
        "research_id": pd.Series([f"R{i:05d}" for i in range(n)], dtype="str"),
        "index_date": pd.Timestamp("2018-04-01"),
        "age_years": rng.uniform(65.0, 100.0, n),
        "sex": pd.Series(rng.choice(["female", "male"], n), dtype="str"),
        "polypharmacy_count_120d": rng.poisson(6.0, n).astype("int64"),
        "bmi_value": np.where(rng.random(n) < 0.15, np.nan, rng.uniform(15.0, 45.0, n)),
        "falls": rng.binomial(1, 0.25, n).astype("int8"),
        "dementia": rng.binomial(1, 0.2, n).astype("int8"),
    })
    eta = -1.5 + 0.04 * (df["age_years"] - 80.0) + 1.5 * df["falls"]
    df["outcome_12m"] = rng.binomial(1, expit(eta.to_numpy())).astype("int8")
    config = experiment_config_from_dict({
        "experiment": {"name": "unit", "kind": "alternative_model", "description": "unit test", "layers": ["L4_alternative"]},
        "dataset": {"feature_spec": str(SPEC_PATH)}, "model": {"name": "logistic_unpenalized"},
        "preprocessing": {"representation": "efalls_reference_coded"},
        "validation": {"strategy": "patient_grouped_random", "seed": 1, "limitation_note": "unit test"}})
    pipe = fit_pipeline(df, spec, config, random_state=3)
    perm = permutation_importance_grouped(pipe, df, df["outcome_12m"].to_numpy(), features=spec.predictor_names(), n_repeats=3, seed=5)
    imp = perm.set_index("feature")["permutation_importance_mean"]
    assert imp.idxmax() == "falls" and imp["falls"] > 0.05 > abs(imp["dementia"])

    coefs = coefficient_table(pipe)
    assert set(coefs["feature"]) == set(spec.predictor_names()) and coefs["rank"].isna().all()
    fi = build_feature_importance_table(pipe, perm, model_name="logistic_unpenalized").set_index("feature")
    assert fi.loc["falls", "importance_rank"] == 1
    assert np.isfinite(fi.loc["falls", "coefficient"]) and np.isnan(fi.loc["bmi_value", "coefficient"])
