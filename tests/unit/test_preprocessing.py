"""Unit tests for the Preprocessor (architecture §3.3; spec §5.3–5.9, D-02, D-04, D-10, D-13, RT-17)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import expit

from falls_ml.config import FractionalPolynomialSection
from falls_ml.errors import ConfigError, DatasetValidationError, NotFittedError, PreprocessingMismatchError
from falls_ml.features.preprocessing import Preprocessor
from falls_ml.features.spec import FeatureSpec, load_feature_spec
from falls_ml.seeding import rng_for

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "configs" / "features" / "efalls_v1.yaml"
ALCOHOL = ["harmful", "higher_risk", "lower_risk", "previous_higher_risk_or_harmful", "zero"]
ALCOHOL_NON_REF = ["harmful", "higher_risk", "previous_higher_risk_or_harmful", "zero", "missing"]


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return load_feature_spec(SPEC_PATH)


@pytest.fixture(scope="module")
def ext_spec(tmp_path_factory: pytest.TempPathFactory) -> FeatureSpec:
    ext = {
        "feature_set": {"name": "test_extension", "version": "0.0.1", "layer": "L3b_meuhedet_predictor"},
        "features": [
            {"name": "egfr", "concept": "eGFR", "dtype": "float_nullable", "missing_rule": "missing_category"},
            {"name": "frailty_band", "concept": "band", "dtype": "categorical_nullable", "missing_rule": "missing_category",
             "levels": ["low", "mid", "high"]},
            {"name": "lab_count", "concept": "labs", "dtype": "count", "missing_rule": "absent_is_zero"},
            {"name": "home_care", "concept": "home care", "dtype": "binary", "missing_rule": "absent_is_zero"},
            {"name": "region", "concept": "region", "dtype": "categorical", "missing_rule": "forbid", "levels": ["north", "south"]},
        ],
    }
    for f in ext["features"]:
        f.update({"layer": "L3b_meuhedet_predictor", "exact_efalls_baseline": False, "clinically_validated": False,
                  "group": "test_group"})
    path = tmp_path_factory.mktemp("ext") / "ext.yaml"
    path.write_text(yaml.safe_dump(ext), encoding="utf-8")
    return load_feature_spec(SPEC_PATH, extensions=[path])


def make_frame(spec: FeatureSpec, n: int, seed: int = 20260914) -> pd.DataFrame:
    """Deterministic synthetic predictor frame (software test data only)."""
    rng = rng_for(seed, "test_preprocessing_frame")
    data: dict[str, Any] = {
        "age_years": rng.uniform(65, 95, n),
        "sex": pd.Series(rng.choice(["female", "male"], n), dtype="str"),
        "polypharmacy_count_120d": rng.integers(0, 41, n).astype("int64"),
        "bmi_value": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(15, 45, n)),
        "smoking_status": pd.Series(rng.choice(np.array(["never", "ex", "current", None], dtype=object), n), dtype="str"),
        "alcohol_category": pd.Series(rng.choice(np.array([*ALCOHOL, None], dtype=object), n), dtype="str"),
        "egfr": np.where(rng.random(n) < 0.3, np.nan, rng.normal(60, 15, n)),
        "frailty_band": pd.Series(rng.choice(np.array(["low", "mid", "high", None], dtype=object), n), dtype="str"),
        "lab_count": rng.integers(0, 10, n).astype("int64"),
        "region": pd.Series(rng.choice(["north", "south"], n), dtype="str"),
    }
    for name in spec.binary_names():
        data[name] = rng.binomial(1, 0.15, n).astype("int8")
    return pd.DataFrame({f.name: data[f.name] for f in spec.features})


def rows_frame(spec: FeatureSpec, rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Explicit rows; binaries default to 0."""
    base = {name: 0 for name in spec.binary_names()}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df = df.astype({name: "int8" for name in spec.binary_names()} | {"bmi_value": "float64", "age_years": "float64",
                                                                     "polypharmacy_count_120d": "int64"})
    return df.astype({c: "str" for c in ("sex", "smoking_status", "alcohol_category")})[[f.name for f in spec.features]]


def outcome(df: pd.DataFrame, seed: int = 1) -> np.ndarray:
    rng = rng_for(seed, "test_preprocessing_outcome")
    eta = -7.0 + 0.06 * df["age_years"] + 0.5 * np.log((df["polypharmacy_count_120d"] + 1) / 10) + 0.4 * df["falls"]
    return rng.binomial(1, expit(eta.to_numpy()))


# ---------------------------------------------------------------------- column lists
def test_published_columns_exact(spec: FeatureSpec) -> None:
    pre = Preprocessor(spec, "efalls_published").fit(make_frame(spec, 50))
    expected = ["age_years", "polypharmacy_log_p1_div10", "sex=male",
                "bmi_category=underweight", "bmi_category=normal", "bmi_category=obese", "bmi_category=missing",
                "smoking=current", *[f"alcohol_category={lv}" for lv in ALCOHOL_NON_REF], *spec.binary_names()]
    assert pre.design_columns() == expected and len(expected) == 85


def test_fp_all_levels_columns_exact(spec: FeatureSpec) -> None:
    fp = FractionalPolynomialSection(mode="fixed_published")
    pre = Preprocessor(spec, "efalls_fp_all_levels", fp=fp).fit(make_frame(spec, 50))
    expected = ["age_years__fp_p1", "polypharmacy_count_120d__fp_p0", "sex=female", "sex=male",
                *[f"bmi_category={lv}" for lv in ["underweight", "normal", "overweight", "obese", "missing"]],
                "smoking_status=never", "smoking_status=ex", "smoking_status=current",
                *[f"alcohol_category={lv}" for lv in [*ALCOHOL, "missing"]], *spec.binary_names()]
    assert pre.design_columns() == expected


def test_reference_coded_and_raw_columns_exact(spec: FeatureSpec) -> None:
    df = make_frame(spec, 50)
    ref = Preprocessor(spec, "efalls_reference_coded").fit(df)
    assert ref.design_columns() == [
        "age_years", "polypharmacy_log_p1_div10", "sex=male",
        "bmi_category=underweight", "bmi_category=normal", "bmi_category=obese", "bmi_category=missing",
        "smoking_status=ex", "smoking_status=current", *[f"alcohol_category={lv}" for lv in ALCOHOL_NON_REF],
        *spec.binary_names()]
    raw = Preprocessor(spec, "efalls_raw").fit(df)
    fp_cols = Preprocessor(spec, "efalls_fp_all_levels", fp=FractionalPolynomialSection(mode="linear")).fit(df).design_columns()
    assert raw.design_columns() == ["age_years", "polypharmacy_count_120d", *fp_cols[2:]]


def test_extension_features_generic_encoding(ext_spec: FeatureSpec) -> None:
    df = make_frame(ext_spec, 400)
    pub = Preprocessor(ext_spec, "efalls_published").fit(df)
    tail = ["egfr", "egfr__missing", "frailty_band=low", "frailty_band=mid", "frailty_band=high", "frailty_band=missing",
            "lab_count", "home_care", "region=north", "region=south"]
    assert pub.design_columns()[-10:] == tail
    ref = Preprocessor(ext_spec, "efalls_reference_coded").fit(df)
    assert ref.design_columns()[-8:] == ["egfr", "egfr__missing", "frailty_band=mid", "frailty_band=high",
                                         "frailty_band=missing", "lab_count", "home_care", "region=south"]
    X = pub.transform(df)
    median = float(np.nanmedian(df["egfr"]))
    assert pub.medians_ == {"egfr": median}
    missing = df["egfr"].isna().to_numpy()
    np.testing.assert_array_equal(X["egfr__missing"].to_numpy(), missing.astype(float))
    np.testing.assert_allclose(X.loc[missing, "egfr"], median)
    np.testing.assert_allclose(X.loc[~missing, "egfr"], df.loc[~missing, "egfr"])
    band = X[["frailty_band=low", "frailty_band=mid", "frailty_band=high", "frailty_band=missing"]]
    assert (band.sum(axis=1) == 1).all()
    np.testing.assert_array_equal(X["frailty_band=missing"].to_numpy(), df["frailty_band"].isna().to_numpy().astype(float))
    assert pub.raw_feature_of("egfr__missing") == "egfr" and pub.raw_feature_of("frailty_band=missing") == "frailty_band"
    assert pub.reference_levels()["region"] == "north"


# ---------------------------------------------------------------------- values
def test_published_hand_computed_rows(spec: FeatureSpec) -> None:
    rows = [
        {"age_years": 89.0, "sex": "female", "polypharmacy_count_120d": 8, "bmi_value": 17.0, "smoking_status": None,
         "alcohol_category": "previous_higher_risk_or_harmful", "dementia": 1, "liver_problems": 1, "osteoporosis": 1},
        {"age_years": 70.5, "sex": "male", "polypharmacy_count_120d": 0, "bmi_value": np.nan, "smoking_status": "current",
         "alcohol_category": None},
        {"age_years": 65.0, "sex": "male", "polypharmacy_count_120d": 61, "bmi_value": 30.0, "smoking_status": "ex",
         "alcohol_category": "lower_risk", "falls": 1},
    ]
    df = rows_frame(spec, rows)
    df.index = [101, 102, 103]
    X = Preprocessor(spec, "efalls_published").fit(df).transform(df)
    assert list(X.index) == [101, 102, 103] and set(X.dtypes) == {np.dtype("float64")}
    expected = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    expected["age_years"] = [89.0, 70.5, 65.0]
    expected["polypharmacy_log_p1_div10"] = [np.log(0.9), np.log(0.1), np.log(6.2)]
    expected["sex=male"] = [0.0, 1.0, 1.0]
    expected.loc[101, ["bmi_category=underweight", "alcohol_category=previous_higher_risk_or_harmful",
                       "dementia", "liver_problems", "osteoporosis"]] = 1.0
    expected.loc[102, ["bmi_category=missing", "smoking=current", "alcohol_category=missing"]] = 1.0
    expected.loc[103, ["bmi_category=obese", "falls"]] = 1.0
    pd.testing.assert_frame_equal(X, expected, check_exact=False, rtol=0, atol=1e-12)


def test_bmi_boundaries_through_preprocessor_rt17(spec: FeatureSpec) -> None:
    bmis = [18.49, 18.5, 24.99, 25.0, 29.99, 30.0, 35.0, 40.0]
    rows = [{"age_years": 70.0, "sex": "female", "polypharmacy_count_120d": 1, "bmi_value": b,
             "smoking_status": "never", "alcohol_category": "zero"} for b in bmis]
    df = rows_frame(spec, rows)
    cols = ["bmi_category=underweight", "bmi_category=normal", "bmi_category=obese", "bmi_category=missing"]
    X = Preprocessor(spec, "efalls_published").fit(df).transform(df)[cols]
    assert X.to_numpy().tolist() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                                     [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
    X40 = Preprocessor(spec, "efalls_published", bmi_obese_cutpoint=40.0).fit(df).transform(df)
    assert X40["bmi_category=obese"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]


def test_smoking_and_alcohol_missing_per_representation(spec: FeatureSpec) -> None:
    rows = [{"age_years": 70.0, "sex": "male", "polypharmacy_count_120d": 3, "bmi_value": 26.0,
             "smoking_status": s, "alcohol_category": a} for s, a in [(None, None), ("ex", "harmful"), ("never", "lower_risk")]]
    df = rows_frame(spec, rows)
    pub = Preprocessor(spec, "efalls_published").fit(df).transform(df)
    assert pub["smoking=current"].tolist() == [0, 0, 0]
    assert pub["alcohol_category=missing"].tolist() == [1, 0, 0]
    ref = Preprocessor(spec, "efalls_reference_coded").fit(df).transform(df)
    assert ref[["smoking_status=ex", "smoking_status=current"]].to_numpy().tolist() == [[0, 0], [1, 0], [0, 0]]
    assert ref[[f"alcohol_category={lv}" for lv in ALCOHOL_NON_REF]].sum(axis=1).tolist() == [1, 1, 0]
    raw = Preprocessor(spec, "efalls_raw").fit(df).transform(df)
    assert raw[["smoking_status=never", "smoking_status=ex", "smoking_status=current"]].to_numpy().tolist() == \
        [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
    assert raw["alcohol_category=missing"].tolist() == [1, 0, 0] and raw["alcohol_category=lower_risk"].tolist() == [0, 0, 1]
    assert raw["polypharmacy_count_120d"].tolist() == [3.0, 3.0, 3.0]


@pytest.mark.parametrize("representation", ["efalls_published", "efalls_fp_all_levels", "efalls_reference_coded", "efalls_raw"])
def test_nullable_string_dtype_with_pandas_na(spec: FeatureSpec, representation: str) -> None:
    """Categoricals read from Parquet may use pandas' nullable ``string`` dtype (pd.NA instead of NaN)."""
    df = make_frame(spec, 60)
    pre = Preprocessor(spec, representation, fp=FractionalPolynomialSection(mode="fixed_published")).fit(df)
    na = df.astype({c: "string" for c in ("sex", "smoking_status", "alcohol_category")})
    assert na["smoking_status"].isna().any() and na["alcohol_category"].isna().any()
    pd.testing.assert_frame_equal(pre.transform(na), pre.transform(df))


def test_fixed_published_fp_values(spec: FeatureSpec) -> None:
    df = make_frame(spec, 30)
    pre = Preprocessor(spec, "efalls_fp_all_levels", fp=FractionalPolynomialSection(mode="fixed_published")).fit(df)
    X = pre.transform(df)
    np.testing.assert_allclose(X["age_years__fp_p1"], df["age_years"])
    np.testing.assert_allclose(X["polypharmacy_count_120d__fp_p0"], np.log((df["polypharmacy_count_120d"] + 1) / 10))
    assert pre.raw_feature_of("polypharmacy_count_120d__fp_p0") == "polypharmacy_count_120d"


def test_fp_select_mode_uses_reference_coded_other_design(spec: FeatureSpec) -> None:
    df = make_frame(spec, 3000)
    y = outcome(df)
    with pytest.raises(ConfigError, match="requires the training outcome"):
        Preprocessor(spec, "efalls_fp_all_levels").fit(df)
    pre = Preprocessor(spec, "efalls_fp_all_levels").fit(df, y)
    assert set(pre.fp_forms_) == {"age_years", "polypharmacy_count_120d"}
    table = pre.fp_forms_["age_years"].selection_table
    # D-10: FP stage sees reference-coded categoricals (1 + 4 + 2 + 5) + 72 binaries
    assert table["n_other_columns"] + table["n_zero_variance_dropped"] == 84
    fp_columns = [c for c in pre.design_columns() if "__fp_p" in c]
    assert fp_columns == [*pre.fp_forms_["age_years"].column_names(), *pre.fp_forms_["polypharmacy_count_120d"].column_names()]
    X = pre.transform(df)
    assert X.shape == (3000, len(pre.design_columns()))
    for name, form in pre.fp_forms_.items():  # design values are exactly the fitted forms applied to raw values
        np.testing.assert_array_equal(X[form.column_names()].to_numpy(), form.transform(df[name].to_numpy()))
    assert pre.fp_forms_["age_years"].is_linear or pre.fp_forms_["age_years"].scale == 10.0


# ---------------------------------------------------------------------- errors
def test_validation_errors(spec: FeatureSpec) -> None:
    df = make_frame(spec, 20)
    pre = Preprocessor(spec, "efalls_published")
    with pytest.raises(NotFittedError):
        pre.transform(df)
    for method in (pre.design_columns, pre.fingerprint, pre.state):
        with pytest.raises(NotFittedError):
            method()
    pre.fit(df)
    with pytest.raises(DatasetValidationError, match="missing predictor columns"):
        pre.transform(df.drop(columns=["falls"]))
    bad = df.copy()
    bad.loc[0, "sex"] = "unknown"
    bad.loc[1, "alcohol_category"] = "moderate"
    bad["falls"] = bad["falls"].astype("float64")
    bad.loc[2, "falls"] = np.nan
    bad.loc[3, "age_years"] = np.nan
    bad.loc[4, "polypharmacy_count_120d"] = -1
    with pytest.raises(DatasetValidationError) as info:
        pre.transform(bad)
    assert len(info.value.problems) == 5
    assert any("undeclared levels ['unknown']" in p for p in info.value.problems)
    with pytest.raises(DatasetValidationError):
        pre.fit(bad)
    with pytest.raises(NotFittedError):  # a failed refit never leaves a half-updated preprocessor
        pre.transform(df)


def test_configuration_errors(spec: FeatureSpec) -> None:
    with pytest.raises(ConfigError):
        Preprocessor(spec, "efalls_onehot")
    with pytest.raises(ConfigError):
        Preprocessor(spec, "efalls_published", bmi_obese_cutpoint=25.0)
    subset = spec.subset(["age_years", "sex", "falls"])
    with pytest.raises(ConfigError, match="full eFalls feature set"):
        Preprocessor(subset, "efalls_published")
    with pytest.raises(ConfigError, match="FP variable"):
        Preprocessor(subset, "efalls_fp_all_levels")
    pre = Preprocessor(subset, "efalls_raw").fit(make_frame(spec, 10))
    assert pre.design_columns() == ["age_years", "sex=female", "sex=male", "falls"]
    with pytest.raises(ConfigError):
        pre.raw_feature_of("not_a_column")


# ---------------------------------------------------------------------- identity and persistence
@pytest.mark.parametrize("representation", ["efalls_published", "efalls_fp_all_levels", "efalls_reference_coded", "efalls_raw"])
def test_state_round_trip(ext_spec: FeatureSpec, spec: FeatureSpec, representation: str) -> None:
    use_spec = spec if representation == "efalls_fp_all_levels" else ext_spec
    df = make_frame(use_spec, 3000)
    pre = Preprocessor(use_spec, representation, random_state=7).fit(df, outcome(df))
    state = json.loads(json.dumps(pre.state()))
    restored = Preprocessor.from_state(use_spec, state)
    assert restored.fingerprint() == pre.fingerprint() == state["fingerprint"]
    pd.testing.assert_frame_equal(restored.transform(df), pre.transform(df))
    assert restored.n_train_rows_ == 3000 and restored.random_state == 7
    assert {v: f.to_dict() for v, f in restored.fp_forms_.items()} == {v: f.to_dict() for v, f in pre.fp_forms_.items()}
    unpickled = pickle.loads(pickle.dumps(pre))  # bundles persist preprocessor.pkl
    assert unpickled.fingerprint() == pre.fingerprint()
    pd.testing.assert_frame_equal(unpickled.transform(df), pre.transform(df))


def test_from_state_refuses_mismatch(spec: FeatureSpec, ext_spec: FeatureSpec) -> None:
    df = make_frame(ext_spec, 200)
    state = Preprocessor(ext_spec, "efalls_raw").fit(df).state()
    with pytest.raises(PreprocessingMismatchError, match="different feature spec"):
        Preprocessor.from_state(spec, state)
    tampered = json.loads(json.dumps(state))
    tampered["medians"]["egfr"] += 1.0
    with pytest.raises(PreprocessingMismatchError):
        Preprocessor.from_state(ext_spec, tampered)


def test_fingerprint_identifies_transformation(spec: FeatureSpec) -> None:
    df = make_frame(spec, 100)
    a = Preprocessor(spec, "efalls_published").fit(df).fingerprint()
    assert a == Preprocessor(spec, "efalls_published").fit(make_frame(spec, 100, seed=5)).fingerprint()
    assert a != Preprocessor(spec, "efalls_published", bmi_obese_cutpoint=40.0).fit(df).fingerprint()
    assert a != Preprocessor(spec, "efalls_reference_coded").fit(df).fingerprint()
    assert len(a) == 64


def test_raw_feature_mapping_and_reference_levels(spec: FeatureSpec) -> None:
    pre = Preprocessor(spec, "efalls_published").fit(make_frame(spec, 20))
    assert pre.raw_feature_of("polypharmacy_log_p1_div10") == "polypharmacy_count_120d"
    assert pre.raw_feature_of("bmi_category=missing") == "bmi_value"
    assert pre.raw_feature_of("smoking=current") == "smoking_status"
    assert pre.raw_feature_of("sex=male") == "sex" and pre.raw_feature_of("falls") == "falls"
    assert pre.reference_levels() == {"sex": "male", "bmi_category": "overweight", "smoking_status": "never",
                                      "alcohol_category": "lower_risk"}


def test_transform_needs_only_predictor_columns(spec: FeatureSpec) -> None:
    df = make_frame(spec, 40)
    pre = Preprocessor(spec, "efalls_raw").fit(df)
    extra = df.assign(research_id="p", outcome_12m=pd.Series(["garbage"] * 40, dtype="str"),
                      index_date=pd.Timestamp("2020-01-01"))
    pd.testing.assert_frame_equal(pre.transform(extra), pre.transform(df))
