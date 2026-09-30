"""Unit tests for internal–external cross-validation (spec §7.3, D-15, D-16, D-21). Synthetic data and stub pipelines."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from falls_ml.errors import ConfigError, FallsMLError, LeakageError
from falls_ml.evaluation import metrics as M
from falls_ml.evaluation.iecv import PER_CLUSTER_COLUMNS, POOLED_COLUMNS, internal_external_cv
from falls_ml.evaluation.meta_analysis import pool_measure
from falls_ml.models.base import ModelAdapter, coefficient_importance
from falls_ml.pipeline import FittedPipeline
from falls_ml.seeding import rng_for

SPEC = SimpleNamespace(outcome=SimpleNamespace(name="y"), identifier_columns=("research_id",), predictor_names=lambda: ["x"])


class XPreprocessor:
    def transform(self, df):
        return df[["x"]].astype(float)

    def raw_feature_of(self, column):
        return column

    def design_columns(self):
        return ["x"]


class Logistic(ModelAdapter):
    name = "logistic_stub"
    representation = "stub"
    is_linear = True

    def fit(self, X, y, *, groups=None):
        self.estimator = LogisticRegression(C=1e4, max_iter=5000).fit(X.to_numpy(), y)
        self.feature_names_ = ["x"]
        return self

    def linear_predictor(self, X):
        return self.estimator.decision_function(X.to_numpy())

    def predict_proba(self, X):
        return expit(self.linear_predictor(X))

    def get_feature_importance(self):
        return coefficient_importance(["x"], self.estimator.coef_[0], standardized_scale=False)

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        raise NotImplementedError


def fit_fn_for(calls=None):
    def fit_fn(df, seed):
        if calls is not None:
            calls.append({"clusters": set(df["site"].astype(object).where(df["site"].notna(), "<null>")), "seed": seed})
        model = Logistic(random_state=seed).fit(df[["x"]], df["y"].to_numpy())
        return FittedPipeline(spec=SPEC, preprocessor=XPreprocessor(), model=model)
    return fit_fn


def cluster_frame(site, n, rng, *, y=None, x=None):
    x = rng.normal(size=n) if x is None else np.asarray(x, dtype=float)
    y = rng.binomial(1, expit(-0.7 + 1.2 * x)) if y is None else np.asarray(y)
    return pd.DataFrame({"site": site, "x": x, "y": y.astype("int8")})


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = rng_for(20260915, "test_iecv_frame")
    few = np.zeros(100, dtype=int)
    few[:5] = 1
    separated_x = np.r_[np.linspace(-2.0, -0.1, 12), np.linspace(0.1, 2.0, 12)]
    parts = [
        cluster_frame("A", 300, rng),
        cluster_frame("B", 100, rng, y=few),                                   # 5 events
        cluster_frame("C", 150, rng),
        cluster_frame("D", 15, rng, y=np.ones(15, dtype=int)),                 # events only
        cluster_frame("E", 24, rng, x=separated_x, y=(separated_x > 0)),       # LP separates outcomes
        cluster_frame(None, 200, rng),                                         # missing cluster value
    ]
    df = pd.concat(parts, ignore_index=True)
    df["site"] = df["site"].astype("str")  # pandas 3 string dtype with NaN for missing
    df.insert(0, "research_id", pd.Series([f"P{i:04d}" for i in range(len(df))], dtype="str"))
    return df


def test_per_cluster_counts_and_exclusion_reasons(frame):
    calls: list[dict] = []
    per, pooled = internal_external_cv(frame, fit_fn_for(calls), cluster_column="site", spec=SPEC, seed=3)
    assert list(per.columns) == PER_CLUSTER_COLUMNS and list(pooled.columns) == POOLED_COLUMNS
    assert per["cluster"].tolist() == ["A", "B", "C", "D", "E", "missing"]
    labels = frame["site"].astype(object).where(frame["site"].notna(), "missing")
    expected_n = labels.value_counts()
    expected_events = frame.groupby(labels)["y"].sum()
    rows = per.set_index("cluster")
    for cluster in per["cluster"]:
        assert rows.loc[cluster, "n"] == expected_n[cluster] and rows.loc[cluster, "n_events"] == expected_events[cluster]
    assert rows["included"].to_dict() == {"A": True, "B": False, "C": True, "D": False, "E": False, "missing": True}
    assert rows.loc["B", "exclusion_reason"] == "fewer than 10 events (5)"
    assert rows.loc["D", "exclusion_reason"] == "no non-events; calibration slope did not converge"
    assert rows.loc["E", "exclusion_reason"] == "calibration slope did not converge"
    assert (rows.loc[["A", "C", "missing"], "exclusion_reason"] == "").all()
    assert per["n"].sum() == len(frame)
    # every cycle trains on all other clusters only (the missing group counts as a cluster)
    all_clusters = {"A", "B", "C", "D", "E", "<null>"}
    held_out = [{"A"}, {"B"}, {"C"}, {"D"}, {"E"}, {"<null>"}]
    assert [c["clusters"] for c in calls] == [all_clusters - h for h in held_out]
    assert len({c["seed"] for c in calls}) == 6
    assert (pooled["measure"] != "note").all()  # excluded clusters hold 139/789 rows (< 20%)


def test_pooled_estimates_match_meta_analysis_of_held_out_predictions(frame):
    per, pooled = internal_external_cv(frame, fit_fn_for(), cluster_column="site", spec=SPEC, seed=3)
    labels = frame["site"].astype(object).where(frame["site"].notna(), "missing")
    values: dict[str, list[float]] = {m: [] for m in ("auroc", "calibration_slope", "citl", "oe_ratio")}
    variances: dict[str, list[float]] = {m: [] for m in values}
    for cluster in ("A", "C", "missing"):
        held = (labels == cluster).to_numpy()
        pipe = fit_fn_for()(frame.loc[~held], 0)
        test = frame.loc[held]
        y, p, lp = test["y"].to_numpy(), pipe.predict_proba(test), pipe.linear_predictor(test)
        row = per.set_index("cluster").loc[cluster]
        assert row["auroc_se_logit"] == pytest.approx(M.auroc_logit_se(y, p), rel=1e-12)
        values["auroc"].append(M.auroc(y, p))
        variances["auroc"].append(M.auroc_delong_variance(y, p))
        _, slope, _, slope_se = M.calibration_slope_intercept(y, lp)
        values["calibration_slope"].append(slope)
        variances["calibration_slope"].append(slope_se**2)
        citl, citl_se = M.citl(y, lp)
        values["citl"].append(citl)
        variances["citl"].append(citl_se**2)
        values["oe_ratio"].append(M.oe_ratio(y, p))
        variances["oe_ratio"].append(M.log_oe_se(y, p) ** 2)
    rows = pooled.set_index("measure")
    assert rows.index.tolist() == ["auroc", "calibration_slope", "citl", "oe_ratio"]
    assert rows["scale"].to_dict() == {"auroc": "logit", "calibration_slope": "identity", "citl": "identity", "oe_ratio": "log"}
    for measure in values:
        expected = pool_measure(values[measure], variances[measure], measure)
        got = rows.loc[measure]
        assert got["k"] == 3
        for column in ("mu", "ci_low", "ci_high", "pi_low", "pi_high", "tau2", "i2"):
            assert got[column] == pytest.approx(getattr(expected, column), rel=1e-9, abs=1e-12)


def test_note_row_when_excluded_clusters_hold_more_than_20_percent(frame):
    per, pooled = internal_external_cv(frame, fit_fn_for(), cluster_column="site", spec=SPEC, min_events=70, seed=3)
    rows = per.set_index("cluster")
    assert rows.loc["C", "exclusion_reason"] == "fewer than 70 events (65)"
    note = pooled[pooled["measure"] == "note"]
    assert len(note) == 1 and "of rows (> 20%)" in note["scale"].iloc[0]
    assert note["k"].iloc[0] == (~per["included"]).sum()
    assert pooled.loc[pooled["measure"] != "note", "k"].eq(per["included"].sum()).all()


def test_fewer_than_two_included_clusters_are_not_pooled(frame):
    subset = frame[frame["site"].isin(["A", "B"])]
    per, pooled = internal_external_cv(subset, fit_fn_for(), cluster_column="site", spec=SPEC, seed=3)
    assert per["cluster"].tolist() == ["A", "B"] and per["included"].tolist() == [True, False]
    measures = pooled[pooled["measure"] != "note"]
    assert (measures["k"] == 1).all() and measures["mu"].isna().all()


def test_numeric_clusters_are_ordered_by_value_and_label_collisions_are_refused():
    rng = rng_for(20260915, "test_iecv_numeric")
    df = pd.concat([cluster_frame(site, 150, rng) for site in (10, 2, 1, None)], ignore_index=True)
    df["site"] = pd.array([10] * 150 + [2] * 150 + [1] * 150 + [None] * 150, dtype="Int64")
    per, _ = internal_external_cv(df, fit_fn_for(), cluster_column="site", spec=SPEC, seed=3)
    assert per["cluster"].tolist() == ["1", "2", "10", "missing"] and (per["n"] == 150).all()
    mixed = df.assign(site=pd.Series([1] * 300 + ["1"] * 300, dtype=object))  # distinct values, same label: never merged
    with pytest.raises(ConfigError):
        internal_external_cv(mixed, fit_fn_for(), cluster_column="site", spec=SPEC, seed=3)


def test_patients_in_more_than_one_cluster_are_refused(frame):
    leaky = frame.copy()
    leaky.loc[leaky.index[-1], "research_id"] = leaky.loc[0, "research_id"]  # same patient in cluster A and 'missing'
    with pytest.raises(LeakageError):
        internal_external_cv(leaky, fit_fn_for(), cluster_column="site", spec=SPEC, seed=3)


def test_invalid_cluster_definitions(frame):
    collision = frame.copy()
    collision["site"] = collision["site"].replace({"B": "missing"})
    with pytest.raises(ConfigError):
        internal_external_cv(collision, fit_fn_for(), cluster_column="site", spec=SPEC, seed=3)
    with pytest.raises(ConfigError):
        internal_external_cv(frame, fit_fn_for(), cluster_column="district", spec=SPEC, seed=3)
    with pytest.raises(FallsMLError):
        internal_external_cv(frame[frame["site"] == "A"], fit_fn_for(), cluster_column="site", spec=SPEC, seed=3)
    with pytest.raises(ConfigError):
        internal_external_cv(frame, fit_fn_for(), cluster_column="site", spec=SPEC, min_events=0, seed=3)
