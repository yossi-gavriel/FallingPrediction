"""Published-equation regression tests RT-01…RT-08 and RT-12 (spec §12.1, D-01, D-07).

Patient (Box S3.1): age 89, polypharmacy 8, underweight, previous higher-risk/harmful drinking, dementia,
liver problems, osteoporosis; every other predictor 0 / reference level. Designs are built directly as
1-row DataFrames (no Preprocessor) so these tests isolate the equation.
"""

from __future__ import annotations

import csv
import json
import math
import re
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import expit

from falls_ml.features.transforms import log_polypharmacy
from falls_ml.models.published_efalls import PublishedEfallsModel, published_linear_predictor

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/models/efalls_published.yaml"
SOURCE_TABLE = REPO_ROOT / "tools/source_table_s3_2.csv"
SOURCE_BINARY = REPO_ROOT / "tools/source_table_s3_1_binary_predictors.json"

# ---------------------------------------------------------------------- TOLERANCES (spec §12.1, all probabilities to 7 dp)
TOLERANCES = {
    "RT-01": 5e-4,         # rounded Box S3.1 coefficients: LP and p vs the printed "-1.368" / "0.203" (and spec LP/p)
    "RT-02_lp": 1e-6,      # full precision LP (RT-02 … RT-05b)
    "RT-02_p": 5e-7,       # full precision probability (RT-02 … RT-05b)
    "RT-06": 5e-4,         # Box S3.2 recalibration of the rounded LP vs printed "0.106" (and spec p)
    "RT-07": 1e-6,         # Box S3.2 recalibration of the full-precision LP
    "RT-08_float": 1e-12,  # coefficient table: float equality (decimal strings must match exactly)
    "RT-12": 1e-6,         # ln((P + 1) / 10)
}

# Spec §12.1 RT-07 prints p = 0.1012433, but expit(-0.423 + 1.25 * -1.408453) = expit(-2.18356625) = 0.10123598…
# (40-digit decimal arithmetic; the printed value implies LP ≈ -1.408389). SPEC ERRATUM reported to the spec owner:
# the test asserts the value of the stated formula at the stated tolerance. Likewise RT-01 p (spec 0.2029811; exact
# 0.2029805) and RT-06 p (spec 0.1059302; exact 0.1059305) differ in the 7th decimal but lie inside their 5e-4 tolerance.
RT07_EXPECTED_P = 0.1012360
RT07_SPEC_PRINTED_P = 0.1012433

PATIENT_BINARIES = ("dementia", "liver_problems", "osteoporosis")
POLYPHARMACY_COUNT = 8


@pytest.fixture(scope="module")
def raw_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def models() -> dict[str, PublishedEfallsModel]:
    return {v: PublishedEfallsModel({"config": str(CONFIG), "sex_parameterisation": v})
            for v in ("lp_c_box_s3_1", "lp_a_table_s3_2", "lp_b_label_swap", "lp_d2_numeric_swap")}


def box_patient_design(columns: list[str], *, male: bool) -> pd.DataFrame:
    row = dict.fromkeys(columns, 0.0)
    row.update({"age_years": 89.0, "polypharmacy_log_p1_div10": math.log((POLYPHARMACY_COUNT + 1) / 10),
                "sex=male": float(male), "bmi_category=underweight": 1.0,
                "alcohol_category=previous_higher_risk_or_harmful": 1.0, **dict.fromkeys(PATIENT_BINARIES, 1.0)})
    return pd.DataFrame([row], columns=columns)


# ---------------------------------------------------------------------- RT-01
def test_rt01_rounded_box_s3_1_example(raw_config):
    ex = raw_config["box_s3_1_rounded_example"]
    assert ex["patient"] == {"sex": "female", "age_years": 89, "polypharmacy_count_120d": POLYPHARMACY_COUNT}
    coefs = {"age_years": ex["age"], "polypharmacy_log_p1_div10": ex["log_polypharmacy"],
             "bmi_category=underweight": ex["underweight"],
             "alcohol_category=previous_higher_risk_or_harmful": ex["previous_higher_risk_or_harmful"],
             **{name: ex[name] for name in PATIENT_BINARIES}}
    X = pd.DataFrame([{"age_years": 89.0, "polypharmacy_log_p1_div10": math.log(0.9), "bmi_category=underweight": 1.0,
                       "alcohol_category=previous_higher_risk_or_harmful": 1.0, **dict.fromkeys(PATIENT_BINARIES, 1.0)}])
    lp = float(published_linear_predictor(X, ex["intercept"], coefs)[0])
    p = float(expit(lp))
    tol = TOLERANCES["RT-01"]
    assert abs(lp - -1.36777) <= tol and abs(lp - ex["expected_lp_printed"]) <= tol
    assert abs(p - 0.2029811) <= tol and abs(p - ex["expected_probability_printed"]) <= tol


# ---------------------------------------------------------------------- RT-02 … RT-05b
@pytest.mark.parametrize(("test_id", "variant", "male", "lp", "p"), [
    ("RT-02", "lp_c_box_s3_1", False, -1.408453, 0.1964781),
    ("RT-02", "lp_a_table_s3_2", False, -1.408453, 0.1964781),
    ("RT-03", "lp_a_table_s3_2", True, -1.104745, 0.2488518),
    ("RT-04", "lp_c_box_s3_1", True, -1.712161, 0.1528836),
    ("RT-05", "lp_b_label_swap", False, -1.104745, 0.2488518),
    ("RT-05", "lp_b_label_swap", True, -1.408453, 0.1964781),
    ("RT-05b", "lp_d2_numeric_swap", False, -1.712161, 0.1528836),
    ("RT-05b", "lp_d2_numeric_swap", True, -1.408453, 0.1964781),
])
def test_rt02_to_rt05b_full_precision(models, test_id, variant, male, lp, p):
    model = models[variant]
    X = box_patient_design(model.design_columns, male=male)
    assert abs(float(model.linear_predictor(X)[0]) - lp) <= TOLERANCES["RT-02_lp"], test_id
    assert abs(float(model.predict_proba(X)[0]) - p) <= TOLERANCES["RT-02_p"], test_id
    scores = models["lp_c_box_s3_1"].score_all_variants(X)  # every adapter scores every variant identically
    assert abs(float(scores[f"lp_{variant}"].iloc[0]) - lp) <= TOLERANCES["RT-02_lp"], test_id
    assert abs(float(scores[f"risk_{variant}"].iloc[0]) - p) <= TOLERANCES["RT-02_p"], test_id


# ---------------------------------------------------------------------- RT-06, RT-07 (D-07)
def test_rt06_recalibration_of_rounded_lp(models, raw_config):
    ref = models["lp_c_box_s3_1"].recalibration_reference()
    assert ref["applies_to"] == "lp_c_box_s3_1"
    p = float(expit(ref["alpha"] + ref["beta"] * raw_config["box_s3_1_rounded_example"]["expected_lp_printed"]))
    tol = TOLERANCES["RT-06"]
    assert abs(p - 0.1059302) <= tol
    assert abs(p - raw_config["box_s3_1_rounded_example"]["expected_recalibrated_probability_printed"]) <= tol


def test_rt07_recalibration_of_full_precision_lp(models):
    model = models["lp_c_box_s3_1"]
    ref = model.recalibration_reference()
    # Evidence for the documented erratum: 40-digit decimal value of the stated formula.
    with localcontext() as ctx:
        ctx.prec = 40
        exact = 1 / (1 + (-(Decimal("-0.423") + Decimal("1.25") * Decimal("-1.408453"))).exp())
    assert abs(float(exact) - RT07_EXPECTED_P) <= 5e-8
    assert abs(float(exact) - RT07_SPEC_PRINTED_P) > TOLERANCES["RT-07"]  # the printed spec value is not reachable
    assert abs(float(expit(ref["alpha"] + ref["beta"] * -1.408453)) - RT07_EXPECTED_P) <= TOLERANCES["RT-07"]
    lp_c = float(model.linear_predictor(box_patient_design(model.design_columns, male=False))[0])
    assert abs(float(expit(ref["alpha"] + ref["beta"] * lp_c)) - RT07_EXPECTED_P) <= TOLERANCES["RT-07"]


# ---------------------------------------------------------------------- RT-08
NON_BINARY_LOCATIONS = {  # (group, term) in Table S3.2 → path in the published YAML
    ("", "Age (years)"): ("terms", "age_years", "coefficient"),
    ("Polypharmacy", "ln((Polypharmacy+1)/10)"): ("terms", "polypharmacy_count_120d", "coefficient"),
    ("Gender", "Female"): ("published_female_coefficient_as_printed",),
    ("BMI category", "Underweight"): ("terms", "bmi_category", "levels", "underweight"),
    ("BMI category", "Normal"): ("terms", "bmi_category", "levels", "normal"),
    ("BMI category", "Obese"): ("terms", "bmi_category", "levels", "obese"),
    ("BMI category", "Missing"): ("terms", "bmi_category", "levels", "missing"),
    ("Smoking", "Current"): ("terms", "smoking", "levels", "current"),
    ("Alcohol consumption", "Harmful drinking"): ("terms", "alcohol_category", "levels", "harmful"),
    ("Alcohol consumption", "Higher risk drinking"): ("terms", "alcohol_category", "levels", "higher_risk"),
    ("Alcohol consumption", "Previous higher risk/harmful drinking"):
        ("terms", "alcohol_category", "levels", "previous_higher_risk_or_harmful"),
    ("Alcohol consumption", "Zero alcohol"): ("terms", "alcohol_category", "levels", "zero"),
    ("Alcohol consumption", "Missing"): ("terms", "alcohol_category", "levels", "missing"),
    ("", "Constant"): ("published_constant_as_printed",),
}
DESIGN_COLUMN_FOR = {  # non-sex Table S3.2 terms → adapter design column
    ("", "Age (years)"): "age_years", ("Polypharmacy", "ln((Polypharmacy+1)/10)"): "polypharmacy_log_p1_div10",
    ("BMI category", "Underweight"): "bmi_category=underweight", ("BMI category", "Normal"): "bmi_category=normal",
    ("BMI category", "Obese"): "bmi_category=obese", ("BMI category", "Missing"): "bmi_category=missing",
    ("Smoking", "Current"): "smoking=current",
    ("Alcohol consumption", "Harmful drinking"): "alcohol_category=harmful",
    ("Alcohol consumption", "Higher risk drinking"): "alcohol_category=higher_risk",
    ("Alcohol consumption", "Previous higher risk/harmful drinking"): "alcohol_category=previous_higher_risk_or_harmful",
    ("Alcohol consumption", "Zero alcohol"): "alcohol_category=zero",
    ("Alcohol consumption", "Missing"): "alcohol_category=missing",
}


def _slug(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


def _lookup(tree: dict, path: tuple[str, ...]):
    for key in path:
        tree = tree[key]
    return tree


def test_rt08_coefficient_table_integrity(models, raw_config):
    with SOURCE_TABLE.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["coef"] != "Reference"]
    printed = {(r["group"], r["term"]): r["coef"] for r in rows}
    as_strings = yaml.load(CONFIG.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)  # scalars as written

    non_reference = {k: v for k, v in printed.items() if k != ("", "Constant")}
    binary_printed = {_slug(term): value for (group, term), value in non_reference.items() if (group, term) not in NON_BINARY_LOCATIONS}
    assert len(non_reference) == 75 and len(binary_printed) == 62 and ("", "Constant") in printed

    yaml_binary = raw_config["terms"]["binary"]
    source_binary = json.loads(SOURCE_BINARY.read_text(encoding="utf-8"))
    assert len(yaml_binary) == 72
    assert sorted(k for k, v in yaml_binary.items() if v != 0.0) == sorted(binary_printed)
    zero = sorted(k for k, v in yaml_binary.items() if v == 0.0)
    assert len(zero) == 10 and zero == sorted(_slug(r["name"]) for r in source_binary if r["retained"] != "yes")

    for key, path in NON_BINARY_LOCATIONS.items():
        assert Decimal(_lookup(as_strings, path)) == Decimal(printed[key]), key
        assert abs(_lookup(raw_config, path) - float(printed[key])) <= TOLERANCES["RT-08_float"], key
    for name, value in binary_printed.items():
        assert Decimal(as_strings["terms"]["binary"][name]) == Decimal(value), name
        assert abs(yaml_binary[name] - float(value)) <= TOLERANCES["RT-08_float"], name

    # The adapter consumes exactly these values (variant A = Table S3.2 as printed: male reference, female -0.303708).
    const, female = float(printed[("", "Constant")]), float(printed[("Gender", "Female")])
    intercept_a, coefs_a = models["lp_a_table_s3_2"].coefficients.design_coefficients("lp_a_table_s3_2")
    for key, column in DESIGN_COLUMN_FOR.items():
        assert abs(coefs_a[column] - float(printed[key])) <= TOLERANCES["RT-08_float"], key
    for name, value in binary_printed.items():
        assert abs(coefs_a[name] - float(value)) <= TOLERANCES["RT-08_float"], name
    assert sum(1 for c, v in coefs_a.items() if v != 0.0) == 75  # 74 non-sex terms + sex
    assert abs(intercept_a + coefs_a["sex=male"] - const) <= TOLERANCES["RT-08_float"]
    assert abs(coefs_a["sex=male"] + female) <= TOLERANCES["RT-08_float"]
    expected_intercepts = {  # D-01 decomposition of the printed constant and female coefficient
        "lp_c_box_s3_1": (const + female, const + 2 * female), "lp_a_table_s3_2": (const + female, const),
        "lp_b_label_swap": (const, const + female), "lp_d2_numeric_swap": (const + 2 * female, const + female)}
    for variant, (female_intercept, male_intercept) in expected_intercepts.items():
        sp = raw_config["sex_parameterisations"][variant]
        assert abs(sp["intercept_female"] - female_intercept) <= TOLERANCES["RT-08_float"], variant
        assert abs(sp["intercept_male"] - male_intercept) <= TOLERANCES["RT-08_float"], variant


# ---------------------------------------------------------------------- RT-12
def test_rt12_log_polypharmacy_term(raw_config):
    assert raw_config["terms"]["polypharmacy_count_120d"]["transform"] == "log((P + 1) / 10)"
    got = log_polypharmacy(np.array([0, 9, 61]))
    np.testing.assert_allclose(got, [-2.302585, 0.0, 1.824549], rtol=0, atol=TOLERANCES["RT-12"])
    # the Box patient designs above use the same term as the preprocessing transform
    assert abs(float(log_polypharmacy(np.array([POLYPHARMACY_COUNT]))[0]) - math.log(0.9)) <= 1e-15
