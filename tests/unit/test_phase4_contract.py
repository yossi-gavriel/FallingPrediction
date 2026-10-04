"""Phase 4 (temporal validation 2026) - fast contract tests: isolation from Phase 3, settings guards, sealing, the outcome contract, overlap
counting, suppression and the pre-declared comparison rule. The end-to-end proofs are in test_phase4_e2e.py (slow)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def L() -> dict[str, Any]:
    from falls_ml.phase4.common import load_definitions

    return load_definitions("configs/meuhedet/phase3.yaml", "configs/meuhedet/phase4.yaml")


def test_phase3_files_are_byte_identical_to_the_delivered_0_9_0_package() -> None:
    """Phase 4 must never change what the delivered Phase 3 uses (its code also defines the classes of the persisted Phase 3 models)."""
    lines = (ROOT / "configs/meuhedet/PHASE3_0.9.0_PROTECTED.sha256").read_text(encoding="utf-8").splitlines()
    bad = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split("  ", 1)
        data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != digest:
            bad.append(rel)
    assert not bad, f"Phase 3 files changed: {bad}"
    assert len([x for x in lines if x.strip() and not x.startswith("#")]) >= 100


def test_phase4_settings_load_and_guard_the_frozen_design(tmp_path: Path) -> None:
    from falls_ml.errors import ConfigError
    from falls_ml.phase4.config import BRIEF_SEALED, load_phase4_config

    cfg = load_phase4_config()
    assert cfg.primary == "LASSO:P3_BASE" and cfg.models == ["LASSO:P3_BASE", "LASSO:P3_VERIFIED_ALL", "LASSO:P3_ALL_RECOVERED"]
    assert set(BRIEF_SEALED) <= set(cfg["sealed_columns"]) and cfg["index_date"] == "2026-01-01"
    base = yaml.safe_load((ROOT / "configs/meuhedet/phase4.yaml").read_text(encoding="utf-8"))
    for mutate in (lambda r: r["models"].update({"primary": "LASSO:P3_ALL_RECOVERED"}),
                   lambda r: r["models"].update({"secondary": ["XGB:P3_ALL_RECOVERED"]}),
                   lambda r: r["sealed_columns"].remove("Fall_Next_180D_Ind"),
                   lambda r: r["sealed_contract_roles"].remove("LABEL"),
                   lambda r: r["metrics"].update({"principal_capacity": 0.2}),
                   lambda r: r["outcome_contract"].update({"window_days": 365})):
        r = json.loads(json.dumps(base))
        mutate(r["phase4"])
        q = tmp_path / "p4.yaml"
        q.write_text(yaml.safe_dump(r, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_phase4_config(q)


def test_sealing_covers_outcome_label_forbidden_post_index_and_outcome_like_new_columns(L: dict[str, Any]) -> None:
    from falls_ml.phase4.config import BRIEF_SEALED
    from falls_ml.phase4.sealed import sealed_map

    contract = L["contract"]
    header = [*contract.names, "Dizziness_Vertigo_Ind", "Cataract_Dx_Date", "Fall_Next_365D_Ind", "Death_Within_90D_Ind", "Days_To_Admission"]
    sm = sealed_map(header, contract, L["cfg4"])
    assert set(BRIEF_SEALED) <= set(sm)
    for c in contract.columns:
        if c.role in ("LABEL", "FORBIDDEN_LEAKAGE") or c.timing == "post_index":
            assert c.name in sm, c.name
    assert {"Followup_End_Date", "Is_Deceased_Ind", "Label_End_180D", "Fall_On_Index_Date_Ind"} <= set(sm)
    assert {"Fall_Next_365D_Ind", "Death_Within_90D_Ind", "Days_To_Admission"} <= set(sm)
    assert "Dizziness_Vertigo_Ind" not in sm and "Cataract_Dx_Date" not in sm
    assert "Index_Date" not in sm and "Is_Eligible_Cohort" not in sm and "Customer_Full_ID" not in sm


def test_the_sealed_reader_refuses_a_sealed_column(tmp_path: Path, L: dict[str, Any]) -> None:
    from falls_ml.phase2.state import Phase2Stop
    from falls_ml.phase4.sealed import read_columns

    p = tmp_path / "x.csv"
    pd.DataFrame({"Index_Date": ["2026-01-01"], "Fall_Next_180D_Ind": ["1"]}).to_csv(p, index=False)
    with pytest.raises(Phase2Stop) as e:
        read_columns(p, ["Index_Date", "Fall_Next_180D_Ind"], {"Fall_Next_180D_Ind": "BRIEF_OUTCOME_COLUMN"}, L["contract"])
    assert e.value.gate == "SEALED_COLUMN_REQUESTED"


def _outcomes(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.Timestamp("2026-01-01")
    y = (rng.random(n) < 0.3).astype(int)
    days = rng.integers(1, 181, n)
    ev = pd.Series([idx + pd.Timedelta(days=int(d)) if v else pd.NaT for v, d in zip(y, days)], dtype="datetime64[us]")
    return pd.DataFrame({"Index_Date": pd.Series([idx] * n, dtype="datetime64[us]"), "Is_Eligible_Cohort": 1,
                         "Fall_Next_180D_Ind": pd.Series(y, dtype="Int64"), "Next_Fall_Date_180D": ev,
                         "Days_To_Next_Fall_180D": pd.Series(np.where(y == 1, days, np.nan)).astype("Float64"),
                         "Label_End_180D": pd.Series([idx + pd.Timedelta(days=180)] * n, dtype="datetime64[us]"),
                         "Label_Reason_180D": np.where(y == 1, "POSITIVE", "NEGATIVE"),
                         "Followup_End_Date": pd.Series([pd.Timestamp("2999-12-31")] * n, dtype="datetime64[us]")})


def test_outcome_contract_passes_a_clean_window_and_stops_each_violation(L: dict[str, Any]) -> None:
    from falls_ml.phase4.evaluate import outcome_contract

    cfg = L["cfg4"]
    ok = outcome_contract(_outcomes(), cfg)
    assert ok["passed"] and ok["checks"]["O1_outcome_after_index_day"]["min_days_to_event_positives"] >= 1
    pos = np.flatnonzero(_outcomes()["Fall_Next_180D_Ind"].to_numpy() == 1)
    o = _outcomes()                                                    # O1: an index-day event counted as outcome
    o.loc[pos[:3], "Next_Fall_Date_180D"] = pd.Timestamp("2026-01-01")
    o.loc[pos[:3], "Days_To_Next_Fall_180D"] = 0
    r = outcome_contract(o, cfg)
    assert not r["passed"] and any(h.startswith("O1") for h in r["hard_failures"])
    o = _outcomes()                                                    # O2: window end not Index + 180
    o["Label_End_180D"] = pd.Timestamp("2026-07-01")
    assert any(h.startswith("O2") for h in outcome_contract(o, cfg)["hard_failures"])
    o = _outcomes()                                                    # O3: event after the window
    o.loc[pos[:2], "Next_Fall_Date_180D"] = pd.Timestamp("2026-08-30")
    o.loc[pos[:2], "Days_To_Next_Fall_180D"] = 241
    assert any(h.startswith("O3") for h in outcome_contract(o, cfg)["hard_failures"])
    o = _outcomes()                                                    # O4: positives after the end of follow-up
    o.loc[pos[:10], "Followup_End_Date"] = pd.Timestamp("2026-01-02")
    assert any(h.startswith("O4") for h in outcome_contract(o, cfg)["hard_failures"])
    o = _outcomes()                                                    # O5: positive without an event date
    o.loc[pos[:5], "Next_Fall_Date_180D"] = pd.NaT
    assert any(h.startswith("O5") for h in outcome_contract(o, cfg)["hard_failures"])
    o = _outcomes(n=200)                                               # O6: too few usable events
    o.loc[:, "Fall_Next_180D_Ind"] = pd.Series([pd.NA] * 200, dtype="Int64")
    assert any(h.startswith("O6") for h in outcome_contract(o, cfg)["hard_failures"])


def test_overlap_counts_each_patient_once() -> None:
    from falls_ml.phase4.score import overlap_counts

    ids26 = pd.Series(["a", "b", "c", "d", "e"], dtype="string")
    c25 = {"eligible": {"a", "b", "x", "y"}, "full_labeled": {"a", "x", "y"}}
    ov, seen = overlap_counts(ids26, c25)
    assert (ov["n_2025_cohort_full_labeled"], ov["n_2026_cohort_eligible"], ov["n_in_both"], ov["n_new_in_2026"], ov["n_2025_not_in_2026"]) == (3, 5, 1, 4, 2)
    assert ov["n_in_both_using_2025_eligible"] == 2 and seen.tolist() == [True, False, False, False, False] and ov["pct_2026_previously_in_2025"] == 20.0


def test_suppression_hides_small_cells_and_their_rates() -> None:
    from falls_ml.phase4.report import suppress_capacity, suppress_obj

    t = suppress_capacity(pd.DataFrame([{"tp": 4, "fp": 30, "fn": 50, "tn": 900, "n_selected": 34, "falls_captured": 4, "capture": 0.07, "ppv": 0.12, "lift": 2.0},
                                        {"tp": 40, "fp": 300, "fn": 50, "tn": 900, "n_selected": 340, "falls_captured": 40, "capture": 0.4, "ppv": 0.12, "lift": 2.0}]))
    assert t.loc[0, "tp"] == "<10" and t.loc[0, "ppv"] == "suppressed" and t.loc[1, "tp"] == 40
    o = suppress_obj({"n_in_both": 7, "n_new_in_2026": 700, "checks": {"x": {"positives_without_event_date": 3, "min_days_to_event_positives": 1}},
                      "censored_by_reason": {"DEATH": 4, "LEFT": 40}})
    assert o["n_in_both"] == "<10" and o["n_new_in_2026"] == 700 and o["checks"]["x"]["positives_without_event_date"] == "<10"
    assert o["checks"]["x"]["min_days_to_event_positives"] == 1 and o["censored_by_reason"] == {"DEATH": "<10", "LEFT": 40}


def test_the_pre_declared_comparison_rule_needs_non_overlapping_intervals() -> None:
    from falls_ml.phase4.metrics import interpret, pct

    assert interpret((0.70, 0.66, 0.74), (0.68, 0.64, 0.72)) == "WITHIN UNCERTAINTY"
    assert interpret((0.80, 0.78, 0.82), (0.70, 0.67, 0.73)) == "CI-SUPPORTED CHANGE"
    assert pct(0.835) == "83.5%"


def test_metric_intervals_are_reproducible_and_cover_the_point() -> None:
    from falls_ml.phase4.metrics import metrics_with_ci

    rng = np.random.default_rng(3)
    y = (rng.random(1500) < 0.1).astype(int)
    p = np.clip(0.05 + 0.2 * y + rng.normal(0, 0.05, 1500), 0.001, 0.999)
    a = metrics_with_ci(y, p, caps=(0.01, 0.05, 0.1), dev_prev=0.1, n_boot=200, seed=5)
    b = metrics_with_ci(y, p, caps=(0.01, 0.05, 0.1), dev_prev=0.1, n_boot=200, seed=5)
    assert a == b
    for k in ("auroc", "ap", "brier", "capture_top10", "ppv_top10", "lift_top10"):
        assert a[f"{k}_ci_low"] <= a[k] <= a[f"{k}_ci_high"], k


def test_cli_offers_the_four_separate_phase4_commands() -> None:
    out = subprocess.run([sys.executable, "-m", "falls_ml", "--help"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}).stdout
    for c in ("meuhedet-phase4-preflight", "meuhedet-phase4-score", "meuhedet-phase4-evaluate", "meuhedet-phase4-status"):
        assert c in out
