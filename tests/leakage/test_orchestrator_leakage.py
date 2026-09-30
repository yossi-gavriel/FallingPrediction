"""Orchestrator-level leakage tests: spy on every learning step of ``run_experiment`` (spec §13.1, D-19).

Every component that fits, tunes, recalibrates or ranks features must only ever receive training or validation
rows; the test partition is touched only by final evaluation. Resampling analyses must reuse the frozen (tuned)
hyperparameters of the final model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import falls_ml.evaluation.calibration as calibration_mod
import falls_ml.evaluation.importance as importance_mod
import falls_ml.evaluation.optimism as optimism_mod
import falls_ml.evaluation.stability as stability_mod
import falls_ml.experiment as experiment_mod
from falls_ml.config import experiment_config_from_dict
from falls_ml.data.dataset import ModelingDataset, write_modeling_dataset
from falls_ml.data.synthetic import generate_synthetic_modeling_dataset
from falls_ml.errors import DatasetValidationError, LeakageError
from falls_ml.features.spec import load_feature_spec

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs" / "features" / "efalls_v1.yaml"



def _config(name: str, **overrides):
    raw = yaml.safe_load((ROOT / "configs" / "experiments" / "fixture" / f"{name}.yaml").read_text(encoding="utf-8"))
    raw["evaluation"]["bootstrap"]["n"] = 5
    raw["evaluation"]["permutation_importance_repeats"] = 1
    raw["analysis"]["stability"].update(enabled=True, n_bootstrap=2)
    raw["analysis"]["optimism"].update(enabled=True, n_bootstrap=2)
    raw["reporting"]["enabled"] = False
    for section, values in overrides.items():
        raw[section].update(values)
    return experiment_config_from_dict(raw, name=name)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    data = tmp_path_factory.mktemp("leak") / "fixture"
    generate_synthetic_modeling_dataset(data, n_patients=700, index_dates=("2018-04-01", "2019-04-01", "2021-04-01"), feature_spec_path=SPEC)
    return ModelingDataset.load(data, load_feature_spec(SPEC))


def _keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["research_id"].astype(str), pd.to_datetime(df["index_date"]).dt.strftime("%Y-%m-%d")))


class _Spy:
    def __init__(self):
        self.frames: dict[str, list[set]] = {}
        self.fit_params: list[dict] = []
        self.recalibration_sizes: list[int] = []

    def record(self, component: str, df: pd.DataFrame) -> None:
        self.frames.setdefault(component, []).append(_keys(df))


@pytest.fixture()
def spy(monkeypatch):
    s = _Spy()
    real_fit = experiment_mod.fit_pipeline

    def fit_pipeline(df, spec, config, **kw):
        s.record("fit_pipeline", df)
        s.fit_params.append(dict(kw.get("model_params") or {}))
        return real_fit(df, spec, config, **kw)
    monkeypatch.setattr(experiment_mod, "fit_pipeline", fit_pipeline)

    for module, name, df_position in ((stability_mod, "bootstrap_stability", 0), (optimism_mod, "harrell_optimism", 0),
                                      (importance_mod, "permutation_importance_grouped", 1)):
        real = getattr(module, name)

        def wrapper(*args, _real=real, _name=name, _pos=df_position, **kw):
            s.record(_name, args[_pos])
            return _real(*args, **kw)
        monkeypatch.setattr(module, name, wrapper)

    real_make = calibration_mod.make_recalibrator

    def make_recalibrator(method):
        rec = real_make(method)
        real_rec_fit = rec.fit

        def fit(p, y, lp=None):
            s.recalibration_sizes.append(len(p))
            return real_rec_fit(p, y, lp=lp)
        rec.fit = fit
        return rec
    monkeypatch.setattr(calibration_mod, "make_recalibrator", make_recalibrator)
    return s


def _partition_keys(run_dir: Path) -> dict[str, set]:
    splits = pd.read_csv(run_dir / "splits.csv", dtype={"research_id": str})
    return {name: _keys(g) for name, g in splits.groupby("split")}


def test_no_learning_step_sees_test_rows(dataset, spy, tmp_path):
    result = experiment_mod.run_experiment(None, dataset, _config("elastic_net_logistic", tuning={"enabled": False}), runs_dir=tmp_path / "runs")
    parts = _partition_keys(result.run_dir)
    assert parts["test"] and not (parts["test"] & (parts["train"] | parts["validation"]))
    for component in ("fit_pipeline", "bootstrap_stability", "harrell_optimism", "permutation_importance_grouped"):
        assert spy.frames.get(component), f"{component} was never called"
        for keys in spy.frames[component]:
            assert not keys & parts["test"], f"{component} received test rows"
    # the final fit and the resampling analyses see training rows only; permutation importance uses validation
    for keys in spy.frames["fit_pipeline"] + spy.frames["bootstrap_stability"] + spy.frames["harrell_optimism"]:
        assert keys <= parts["train"]
    assert all(keys <= parts["validation"] for keys in spy.frames["permutation_importance_grouped"])
    assert spy.recalibration_sizes == [len(parts["validation"])], "recalibration must be fitted once, on the validation split"
    registry = [json.loads(line) for line in (tmp_path / "runs" / "test_evaluation_registry.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["run_id"] for e in registry] == [result.run_id]


def test_resampling_reuses_tuned_hyperparameters(dataset, spy, tmp_path):
    config = _config("elastic_net_logistic", tuning={"search_space": {"C": [0.01, 1.0], "l1_ratio": [0.5]}},
                     analysis={"optimism": {"enabled": False, "n_bootstrap": 2}})
    result = experiment_mod.run_experiment(None, dataset, config, runs_dir=tmp_path / "runs")
    best = result.metrics["model"]["best_params"]
    assert best and result.metrics["model"]["resampling_params"] == best
    n_trials = result.metrics["hyperparameter_search"]["n_trials"]
    after_tuning = spy.fit_params[n_trials:]
    assert after_tuning and all(p == best for p in after_tuning), "final fit and bootstrap refits must use the tuned parameters"


def test_same_config_cannot_reevaluate_scientific_test_set(dataset, tmp_path):
    data = tmp_path / "scientific"
    frame = dataset.frame.drop(columns=["dataset_version", "mapping_version", "source"], errors="ignore")
    write_modeling_dataset(frame, data, dataset.spec, dataset_version="v-sci", mapping_version="m", source="unit_test_only",
                           generator="test", scientific_use_allowed=True, created_utc="2026-01-01T00:00:00+00:00",
                           data_freeze_date="2023-12-31", audit={"note": "test double of a scientific dataset"})
    ds = ModelingDataset.load(data, dataset.spec)
    config = _config("elastic_net_logistic", tuning={"enabled": False}, analysis={"stability": {"enabled": False, "n_bootstrap": 2},
                                                                                  "optimism": {"enabled": False, "n_bootstrap": 2}})
    runs = tmp_path / "runs"
    experiment_mod.run_experiment(None, ds, config, runs_dir=runs)
    with pytest.raises(LeakageError, match="already evaluated"):
        experiment_mod.run_experiment(None, ds, config, runs_dir=runs)
    again = experiment_mod.run_experiment(None, ds, config, runs_dir=runs, allow_test_reevaluation=True)
    assert again.metrics["test_evaluation_registry"]["prior_evaluations_same_test_rows"] == 1
    assert any("already evaluated" in w for w in again.metrics["warnings"])


def test_immature_outcome_windows_rejected(dataset, tmp_path):
    data = tmp_path / "immature"
    frame = dataset.frame.drop(columns=["dataset_version", "mapping_version", "source"], errors="ignore")
    write_modeling_dataset(frame, data, dataset.spec, dataset_version="v-imm", mapping_version="m", source="unit_test_only",
                           generator="test", scientific_use_allowed=True, created_utc="2026-01-01T00:00:00+00:00",
                           data_freeze_date="2021-12-31")
    ds = ModelingDataset.load(data, dataset.spec)
    with pytest.raises(DatasetValidationError, match="not mature"):
        experiment_mod.run_experiment(None, ds, _config("elastic_net_logistic", tuning={"enabled": False}), runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir()), "no run directory for a rejected dataset"
