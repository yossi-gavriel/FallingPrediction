"""Phase 3 time contract and feature eligibility (fast): the corrected END-OF-INDEX-DAY contract is verified, never assumed; index-day records are
history; post-index records, undated values and unreadable cells are UNKNOWN; undated sources are SAFE_ATTESTED only while V3 holds (and carry an
availability assumption); the configuration guards refuse weakened rules; Phase 2 files are untouched."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
IDX = pd.Timestamp("2025-01-01")


@pytest.fixture(scope="module")
def L() -> dict[str, Any]:
    from falls_ml.phase3.runner import load_all

    return load_all()


def _frame(n: int = 400, seed: int = 1) -> pd.DataFrame:
    from falls_ml.data.meuhedet_synthetic import generate_synthetic_wide_extract

    df = generate_synthetic_wide_extract(n, seed=seed, index_date="2025-01-01")
    df = df[(df["Index_Date"] == IDX) & (df["Is_Eligible_Cohort"] == 1) & df["Fall_Next_180D_Ind"].notna()].reset_index(drop=True)
    for h in ("180D", "30D"):   # the corrected contract: first events from day 1
        d0 = (df[f"Next_Fall_Date_{h}"].dt.normalize() == IDX).fillna(False)
        df.loc[d0, f"Next_Fall_Date_{h}"] = df.loc[d0, f"Next_Fall_Date_{h}"] + pd.Timedelta(days=1)
        df.loc[d0, f"Days_To_Next_Fall_{h}"] = pd.to_numeric(df.loc[d0, f"Days_To_Next_Fall_{h}"]) + 1
    return df


def _verify(L: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    from falls_ml.phase3.timecontract import verify

    part = pd.Series(np.where(np.arange(len(df)) % 5 == 0, "validation", "train"), index=df.index)
    return verify(df, part, dictionary=L["dictionary"], d00=L["d00"], tc=L["tc"])


def test_contract_passes_on_a_consistent_extract_and_same_day_records_are_history(L: dict[str, Any]) -> None:
    df = _frame()
    k = np.flatnonzero(df["MiniCog_Date"].notna().to_numpy())[:5]
    df.loc[k, "MiniCog_Date"] = np.datetime64(IDX, "us")               # ON the index day: history under the corrected contract
    r = _verify(L, df)
    assert r["hard_passed"] and r["attestation_holds"]
    assert r["V3_no_post_index_records"]["by_column"]["MiniCog_Date"]["n_on_index"] == 5
    assert r["test_rows_read"] is False


def test_v1_fails_when_the_outcome_contains_index_day_events(L: dict[str, Any]) -> None:
    df = _frame()
    pos = np.flatnonzero((df["Fall_Next_180D_Ind"] == 1).to_numpy())[:3]
    df.loc[pos, "Next_Fall_Date_180D"] = np.datetime64(IDX, "us")
    df.loc[pos, "Days_To_Next_Fall_180D"] = 0
    r = _verify(L, df)
    assert not r["V1_label_excludes_index_day"]["passed"] and not r["hard_passed"]
    assert r["V1_label_excludes_index_day"]["positives_event_on_index_day"] == 3


def test_v2_fails_on_a_wrong_window_and_v3_withdraws_the_attestation_on_post_index_records(L: dict[str, Any]) -> None:
    df = _frame()
    df.loc[:3, "Label_End_180D"] = df.loc[:3, "Label_End_180D"] + pd.Timedelta(days=1)
    k = np.flatnonzero(df["Falls_Risk_Assessment_Date"].notna().to_numpy())[:2]
    df.loc[k, "Falls_Risk_Assessment_Date"] = np.datetime64(IDX + pd.Timedelta(days=10), "us")
    r = _verify(L, df)
    assert not r["V2_window_end"]["passed"] and not r["hard_passed"]
    assert not r["attestation_holds"] and r["V3_no_post_index_records"]["by_column"]["Falls_Risk_Assessment_Date"]["n_after_index"] == 2


def test_v7_flags_one_episode_counted_as_history_and_outcome(L: dict[str, Any]) -> None:
    from falls_ml.phase3.synthetic import plant_phase3_history

    base = _frame(1500, seed=4)
    ok = _verify(L, base)["V7_proxy_episode_audit"]
    assert not ok["flag_short_gap"]
    df, _ = plant_phase3_history(base.assign(Is_Eligible_Cohort=1), episode_duplicates=40, d00_dx_share=0.0)
    r = _verify(L, df)["V7_proxy_episode_audit"]
    assert r["investigation"] and r["flag_short_gap"]
    assert r["days_since_inconsistent"] == 0


def _recovery(L: dict[str, Any], df: pd.DataFrame, attest: bool) -> pd.DataFrame:
    from falls_ml.phase2.engineer import engineer
    from falls_ml.phase3.recovery import build_recovery

    base = [f.canonical for f in L["mapping"].features if f.include_in_baseline]
    adapter_like = df.copy()
    from falls_ml.phase3.cohort import canonical_baseline
    from falls_ml.data.meuhedet_wide import MeuhedetWideDatasetAdapter

    ad = MeuhedetWideDatasetAdapter(L["mapping"], L["contract"], L["espec"], features=base, id_pepper="p")
    canon, _ = canonical_baseline(ad, L["mapping"], adapter_like, base)
    w = pd.concat([df, canon], axis=1)
    values = engineer(w, L["cat"])
    res = build_recovery(w, values, catalogue=L["cat"], mapping=L["mapping"], baseline=base, contract=L["contract"], dictionary=L["dictionary"],
                         d00=L["d00"], rules=L["rules"], train_mask=np.ones(len(w), dtype=bool), min_observed_train=5, tc=L["tc"], attestation_holds=attest)
    return res.registry.set_index("feature"), res


def test_eligibility_classes_under_the_corrected_contract(L: dict[str, Any]) -> None:
    df = _frame(1200, seed=2)
    k = np.flatnonzero(df["MiniCog_Date"].notna().to_numpy())
    df.loc[k[:6], "MiniCog_Date"] = np.datetime64(IDX, "us")                                  # same day -> still KNOWN
    df.loc[k[6:8], "MiniCog_Date"] = np.datetime64(IDX + pd.Timedelta(days=3), "us")          # after -> UNKNOWN
    reg, res = _recovery(L, df, attest=True)
    assert reg.at["cog_minicog_score", "phase3_class"] == "SAFE_VERIFIED_BOUNDED" and reg.at["cog_minicog_score", "n_unknown"] == 2
    assert res.unknown["cog_minicog_score"].sum() == 2
    assert reg.at["med_polypharmacy_5plus", "phase3_class"] == "SAFE_ATTESTED" and not reg.at["med_polypharmacy_5plus", "verified"]
    assert reg.at["dementia", "phase3_class"] == "SAFE_ATTESTED" and reg.at["dementia", "availability_risk"] == "HIGH"
    assert reg.at["age_years", "phase3_class"] == "SAFE_VERIFIED" and reg.at["falls", "phase3_class"].startswith("SAFE_VERIFIED")
    assert "availability_assumption" in reg.columns and reg["availability_assumption"].str.len().min() > 0
    reg2, _ = _recovery(L, df, attest=False)                                                 # V3 failed -> the attestation is withdrawn
    assert reg2.at["med_polypharmacy_5plus", "phase3_class"] == "UNRESOLVED" and reg2.at["dementia", "phase3_class"] == "UNRESOLVED"
    assert reg2.at["cog_minicog_score", "phase3_class"] == "SAFE_VERIFIED_BOUNDED"


def test_undated_values_and_unreadable_cells_are_unknown_never_absent(L: dict[str, Any]) -> None:
    df = _frame(1200, seed=3)
    k = np.flatnonzero((df["Get_Up_And_Go_Date"].notna() & df["Uses_Walking_Aid_Ind"].notna()).to_numpy())
    df.loc[k[:3], "Get_Up_And_Go_Date"] = pd.NaT                  # a recorded item without its record date
    df["__unreadable__Get_Up_And_Go_Date"] = False
    df.loc[k[3:5], "__unreadable__Get_Up_And_Go_Date"] = True     # e.g. '00:00.0' the validated reader could not read
    reg, res = _recovery(L, df, attest=True)
    assert int(res.unknown["func_walking_aid"].sum()) == 5


def test_cumulative_zero_is_exact_and_forbidden_columns_never_enter(L: dict[str, Any]) -> None:
    df = _frame(800, seed=5)
    rows = np.flatnonzero(df["Last_Dx_Date"].notna().to_numpy())[:30]
    df.loc[rows, "Last_Dx_Date"] = np.datetime64(IDX + pd.Timedelta(days=2), "us")
    reg, res = _recovery(L, df, attest=False)
    gait0 = (df["Gait_Disorder_Since_Study_Start_Ind"] == 0).to_numpy()
    assert not res.unknown["mobility_problems"].to_numpy()[gait0].any()          # an observed 0 of a 'since 2022' flag stays exact
    assert reg.at["util_external_care_visible_365d", "phase3_class"] in ("UNRESOLVED", "SAFE_ATTESTED", "INELIGIBLE_DATA")
    tokens = {c for s in reg["inputs"] for c in str(s).split("; ")}
    assert not tokens & {"External_Care_Count_365D", "Fall_On_Index_Date_Ind", "Last_Ext_Date", "Next_Fall_Date_180D"}


def test_config_guards_refuse_a_weakened_contract(tmp_path: Path) -> None:
    from falls_ml.errors import ConfigError
    from falls_ml.phase3.config import load_phase3_config
    from falls_ml.phase3.timecontract import load_time_contract

    raw = yaml.safe_load((ROOT / "configs/meuhedet/phase3_time_contract.yaml").read_text(encoding="utf-8"))
    raw["phase3_time_contract"]["prediction_time"] = "START_OF_INDEX_DAY"
    p = tmp_path / "tc.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_time_contract(p)
    base = yaml.safe_load((ROOT / "configs/meuhedet/phase3.yaml").read_text(encoding="utf-8"))
    for mutate in (lambda r: r["standards"]["PRIMARY_FULL"].append("UNRESOLVED"),
                   lambda r: r["standards"]["SENSITIVITY_VERIFIED_ONLY"].append("SAFE_ATTESTED"),
                   lambda r: r["standards"]["PRIMARY_FULL"].append("NOT_RECOVERABLE_FUTURE_RECORDS"),
                   lambda r: r["population"].update({"primary": "D00_CLEAN"}),
                   lambda r: r["xgb"]["fixed"].update({"scale_pos_weight": 40.0}),
                   lambda r: r["selection"].update({"discovery_reference": "LASSO:HISTORICAL_B15"})):
        r = json.loads(json.dumps(base))
        mutate(r["phase3"])
        q = tmp_path / "p3.yaml"
        q.write_text(yaml.safe_dump(r, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_phase3_config(q)


def test_phase3_frozen_configuration_is_current(L: dict[str, Any]) -> None:
    from falls_ml.phase3.final_config import FROZEN_PATH, build_final_config, sha256_of, sha_line, to_bytes

    fc = build_final_config(L["cfg"], L["rules"], L["cat"], contract=L["contract"], dictionary=L["dictionary"], mapping=L["mapping"], d00=L["d00"], tc=L["tc"])
    assert (ROOT / FROZEN_PATH).read_bytes() == to_bytes(fc), "run tools/freeze_phase3_config.py after a reviewed change"
    assert (ROOT / FROZEN_PATH).with_suffix(".sha256").read_text(encoding="utf-8") == sha_line(sha256_of(fc))
    assert fc["outcome"]["prediction_time"] == "END_OF_INDEX_DAY" and "UNRESOLVED" not in fc["standards"]["PRIMARY_FULL"]


def test_phase2_files_are_byte_identical_to_the_running_0_8_1_package() -> None:
    """Phase 3 must never change what the running Phase 2 experiment uses (configs/meuhedet/PHASE2_0.8.1_PROTECTED.sha256)."""
    lines = (ROOT / "configs/meuhedet/PHASE2_0.8.1_PROTECTED.sha256").read_text(encoding="utf-8").splitlines()
    bad = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split("  ", 1)
        data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != digest:
            bad.append(rel)
    assert not bad, f"Phase 2 files changed: {bad}"
    assert len([x for x in lines if x.strip() and not x.startswith("#")]) >= 30
