"""Leakage tests: the Preprocessor learns from training rows only and transform reads predictors only.

Architecture §1 principle 6 and §4 step 3: preprocessing is fitted on train only; transforming
validation/test rows must not change any fitted quantity (FP forms, medians, fingerprint) and must
never touch outcome, identifier or date columns.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import expit

from falls_ml.errors import DatasetValidationError
from falls_ml.features.preprocessing import Preprocessor
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.seeding import rng_for

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"
ALCOHOL = ["harmful", "higher_risk", "lower_risk", "previous_higher_risk_or_harmful", "zero"]


@pytest.fixture(scope="module")
def spec(tmp_path_factory: pytest.TempPathFactory) -> FeatureSpec:
    ext = {"feature_set": {"name": "leakage_ext", "version": "0.0.1", "layer": "L3b_meuhedet_predictor"},
           "features": [{"name": "egfr", "concept": "eGFR", "dtype": "float_nullable", "missing_rule": "missing_category",
                         "layer": "L3b_meuhedet_predictor", "exact_efalls_baseline": False,
                         "clinically_validated": False, "group": "labs"}]}
    path = tmp_path_factory.mktemp("ext") / "ext.yaml"
    path.write_text(yaml.safe_dump(ext), encoding="utf-8")
    return load_feature_spec(SPEC_PATH, extensions=[path])


def make_frame(spec: FeatureSpec, n: int, component: str, *, age: tuple[float, float] = (65, 95), poly_max: int = 40,
               bmi_missing: float = 0.2, egfr_mean: float = 60.0, egfr_missing: float = 0.3) -> pd.DataFrame:
    rng = rng_for(20260914, component)
    data: dict[str, Any] = {
        "research_id": pd.Series([f"{component}-{i}" for i in range(n)], dtype="str"),
        "index_date": pd.Timestamp("2021-04-01"),
        "age_years": rng.uniform(*age, n),
        "sex": pd.Series(rng.choice(["female", "male"], n), dtype="str"),
        "polypharmacy_count_120d": rng.integers(0, poly_max + 1, n).astype("int64"),
        "bmi_value": np.where(rng.random(n) < bmi_missing, np.nan, rng.uniform(15, 45, n)),
        "smoking_status": pd.Series(rng.choice(np.array(["never", "ex", "current", None], dtype=object), n), dtype="str"),
        "alcohol_category": pd.Series(rng.choice(np.array([*ALCOHOL, None], dtype=object), n), dtype="str"),
        "egfr": np.where(rng.random(n) < egfr_missing, np.nan, rng.normal(egfr_mean, 10, n)),
    }
    for name in spec.binary_names():
        data[name] = rng.binomial(1, 0.15, n).astype("int8")
    df = pd.DataFrame(data)
    eta = -7.0 + 0.06 * df["age_years"] + 0.5 * np.log((df["polypharmacy_count_120d"] + 1) / 10) + 0.4 * df["falls"]
    df["outcome_12m"] = rng.binomial(1, expit(eta.to_numpy())).astype("int8")
    return df


class ColumnAccessSpy:
    """Minimal DataFrame stand-in that records which columns are read."""

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.accessed: list[str] = []

    @property
    def columns(self) -> pd.Index:
        return self._df.columns

    @property
    def index(self) -> pd.Index:
        return self._df.index

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, key: str) -> pd.Series:
        self.accessed.append(key)
        return self._df[key]


@pytest.fixture(scope="module")
def train(spec: FeatureSpec) -> pd.DataFrame:
    return make_frame(spec, 3000, "leakage_train")


@pytest.fixture(scope="module")
def validation(spec: FeatureSpec) -> pd.DataFrame:
    # deliberately very different: older, heavier polypharmacy, no BMI, shifted/mostly-missing eGFR
    return make_frame(spec, 800, "leakage_validation", age=(95, 110), poly_max=150, bmi_missing=1.0,
                      egfr_mean=20.0, egfr_missing=0.8)


@pytest.mark.parametrize("representation", ["efalls_fp_all_levels", "efalls_published", "efalls_reference_coded", "efalls_raw"])
def test_transforming_validation_rows_changes_no_fitted_state(spec: FeatureSpec, train: pd.DataFrame,
                                                              validation: pd.DataFrame, representation: str) -> None:
    pre = Preprocessor(spec, representation).fit(train, train["outcome_12m"].to_numpy())
    before = copy.deepcopy(pre.state())
    X_val = pre.transform(validation)
    assert pre.state() == before and pre.fingerprint() == before["fingerprint"]
    train_median = float(np.nanmedian(train["egfr"]))
    assert pre.medians_ == {"egfr": train_median}
    missing = validation["egfr"].isna().to_numpy()
    np.testing.assert_allclose(X_val.loc[missing, "egfr"], train_median)  # training median, not validation's
    # sensitivity check: fitting on train + validation would change the fitted state
    pooled = pd.concat([train, validation], ignore_index=True)
    assert Preprocessor(spec, representation).fit(pooled, pooled["outcome_12m"].to_numpy()).fingerprint() != before["fingerprint"]


def test_fp_forms_depend_on_training_rows_only(spec: FeatureSpec, train: pd.DataFrame, validation: pd.DataFrame) -> None:
    y = train["outcome_12m"].to_numpy()
    pre = Preprocessor(spec, "efalls_fp_all_levels").fit(train, y)
    forms = {v: f.to_dict() for v, f in pre.fp_forms_.items()}
    pre.transform(validation)
    assert {v: f.to_dict() for v, f in pre.fp_forms_.items()} == forms
    assert pre.n_train_rows_ == len(train)
    refit = Preprocessor(spec, "efalls_fp_all_levels").fit(train, y)
    assert {v: f.to_dict() for v, f in refit.fp_forms_.items()} == forms


def test_transform_is_row_wise(spec: FeatureSpec, train: pd.DataFrame, validation: pd.DataFrame) -> None:
    pre = Preprocessor(spec, "efalls_fp_all_levels").fit(train, train["outcome_12m"].to_numpy())
    X_batch = pre.transform(validation)
    for i in (0, 17, 799):
        pd.testing.assert_frame_equal(pre.transform(validation.iloc[[i]]), X_batch.iloc[[i]])


def test_fit_and_transform_never_read_outcome_or_identifiers(spec: FeatureSpec, train: pd.DataFrame,
                                                             validation: pd.DataFrame) -> None:
    y = train["outcome_12m"].to_numpy()
    spy_train = ColumnAccessSpy(train)
    pre = Preprocessor(spec, "efalls_fp_all_levels").fit(spy_train, y)  # type: ignore[arg-type]
    spy_val = ColumnAccessSpy(validation)
    pre.transform(spy_val)  # type: ignore[arg-type]
    forbidden = {"outcome_12m", "research_id", "index_date"}
    predictors = set(spec.predictor_names())
    for spy in (spy_train, spy_val):
        assert not forbidden & set(spy.accessed)
        assert set(spy.accessed) <= predictors
    corrupted = validation.assign(outcome_12m=pd.Series(["not-an-outcome"] * len(validation), dtype="str"))
    pd.testing.assert_frame_equal(pre.transform(corrupted), pre.transform(validation.drop(columns=list(forbidden))))


def test_fp_domain_violation_in_new_rows_fails_loudly(spec: FeatureSpec, train: pd.DataFrame) -> None:
    # A form fitted without shift cannot silently extrapolate to values <= 0 at transform time.
    positive = train.assign(polypharmacy_count_120d=train["polypharmacy_count_120d"] + 1)
    pre = Preprocessor(spec, "efalls_fp_all_levels").fit(positive, train["outcome_12m"].to_numpy())
    form = pre.fp_forms_["polypharmacy_count_120d"]
    assert not form.is_linear and form.shift == 0.0
    with pytest.raises(DatasetValidationError, match="<= 0"):
        pre.transform(train.assign(polypharmacy_count_120d=0))
