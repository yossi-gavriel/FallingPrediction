"""Model-training dashboards and Hebrew management reports (falls_ml model-report). Synthetic fixtures only - no real data."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from falls_ml.cli import main as cli_main
from falls_ml.data.meuhedet_synthetic import write_synthetic_wide_extract
from falls_ml.data.meuhedet_wide import DEFAULT_MAPPING, load_wide_mapping
from falls_ml.errors import LeakageError
from falls_ml.modelreport import compute as C
from falls_ml.modelreport import figures as F
from falls_ml.modelreport.artifacts import analysis_of, discover_runs, feature_label, load_labels
from falls_ml.modelreport.management import BANNED
from falls_ml.modelreport.runner import build_model_reports

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "experiments" / "meuhedet" / "explore_180d_template.yaml"
WM = "EXPLORATORY 180-DAY OUTCOME – NOT EFALLS REPRODUCTION"
ID_RE = re.compile(r"\bS\d{10}\b")
HEB = re.compile(r"[֐-׿]")


# ------------------------------------------------------------------ units (fast)
def test_analysis_names_and_labels_cover_every_extended_predictor() -> None:
    assert analysis_of("MEUHEDET_EFALLS_EXTENDED_180D_EXPLORATORY")[0] == "EXTENDED"
    assert analysis_of("MEUHEDET_EFALLS_STRICT_180D_EXPLORATORY")[0] == "STRICT"
    assert analysis_of("MEUHEDET_EFALLS_SAFE_ALL_ROWS_180D_SENSITIVITY")[0] == "SAFE_ALL_ROWS"
    key, disp = analysis_of("MEUHEDET_EFALLS_EXTENDED_180D_ABLATION_NO_FALLS_NO_MOBILITY_PROBLEMS")
    assert key.startswith("ABLATION_") and "falls" in disp and "mobility_problems" in disp
    assert analysis_of("MEUHEDET_EFALLS_STRICT_180D_PATIENT_DISJOINT")[0] == "STRICT_PATIENT_DISJOINT"
    labels = load_labels()
    for f in load_wide_mapping(DEFAULT_MAPPING).feature_sets()["extended"]:
        assert HEB.search(feature_label(labels, f, "he")) and feature_label(labels, f, "en") != f, f
    assert feature_label(labels, "sex=female", "he") == "מין: אישה" and feature_label(labels, "age_years__fp_p1", "en") == "Age (years)"


def test_rtl_manual_reordering_when_matplotlib_does_not_reorder(monkeypatch) -> None:
    monkeypatch.setattr(F, "native_bidi", lambda: False)
    assert F.rtl("גיל (שנים)") == "(םינש) ליג"
    assert F.rtl("10% העליונים") == "םינוילעה 10%"
    assert F.rtl("AUROC 0.67") == "AUROC 0.67"
    monkeypatch.setattr(F, "native_bidi", lambda: True)
    assert F.rtl("גיל (שנים)") == "גיל (שנים)"


def test_lift_threshold_and_decile_tables() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p * 0.3).astype(int)
    lift = C.lift_table(y, p, min_cell=5, n_boot=200)
    top10 = lift[lift["top_pct"] == 10.0].iloc[0]
    order = np.argsort(-p)[:200]
    assert int(top10["falls_in_group"]) == int(y[order].sum()) and float(top10["lift"]) > 1.0
    assert float(top10["capture_ci_low"]) <= float(top10["pct_of_all_falls_captured"]) <= float(top10["capture_ci_high"])
    tt = C.threshold_table(y, p, min_cell=5)
    assert (pd.to_numeric(tt["pct_flagged"], errors="coerce").dropna().diff().dropna() <= 1e-9).all()   # fewer flagged as the threshold rises
    dec = C.risk_deciles(y, p, min_cell=5)
    rates = pd.to_numeric(dec["observed_rate"], errors="coerce")
    assert len(dec) == 10 and dec["events"].iloc[0] == "<min_cell" and pd.isna(rates.iloc[0])   # small cell suppressed
    assert rates.dropna().iloc[-1] > rates.dropna().iloc[0]
    small = C.lift_table(y[:40], p[:40], min_cell=10, n_boot=50)
    assert small["pct_of_all_falls_captured"].isna().any()   # tiny groups are suppressed, never shown


# ------------------------------------------------------------------ end to end (slow)
def _small_template(tmp: Path) -> Path:
    tpl = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    tpl["evaluation"]["bootstrap"]["n"] = 20
    tpl["evaluation"]["permutation_importance_repeats"] = 1
    tpl["analysis"]["stability"]["n_bootstrap"] = 6
    tpl["analysis"]["optimism"]["n_bootstrap"] = 2
    p = tmp / "template.yaml"
    p.write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def explored(tmp_path_factory):
    from falls_ml.meuhedet_explore import explore

    tmp = tmp_path_factory.mktemp("mr")
    csv = write_synthetic_wide_extract(tmp / "wide.csv", n_rows=3000, seed=9, other_index_date_share=0.0, csv_date_format="%d/%m/%Y", n_index_day_falls=4)
    res = explore(csv, out_dir=tmp / "explore", index_day_records="drop_rows", template_path=_small_template(tmp), model_report="reduced", fast=False)
    return {"res": res, "out": tmp / "explore", "tmp": tmp, "csv": csv}


@pytest.mark.slow
def test_explore_builds_the_reports_and_they_follow_the_rules(explored) -> None:
    res, out = explored["res"], explored["out"]
    assert res.model_report["status"] == "built", res.model_report
    rep = out / "reports"
    for name in ("MODEL_TRAINING_REPORT_STRICT.html", "MODEL_TRAINING_REPORT_EXTENDED.html", "MODEL_COMPARISON_REPORT.html", "MANAGEMENT_MODEL_REPORT_HE.html",
                 "MANAGEMENT_MODEL_SUMMARY_HE.md", "EXECUTIVE_ONE_PAGER_HE.html", "FIGURE_INDEX.md", "model_report_manifest.json"):
        assert (rep / name).exists(), name
    pngs = sorted((rep / "figures").glob("*.png"))
    assert len(pngs) > 25 and all(p.with_suffix(".svg").exists() for p in pngs)
    assert {"he_funnel", "he_deciles", "he_roc", "he_calibration", "he_features", "he_permutation", "he_lift", "he_strict_vs_extended",
            "he_roadmap"} <= {p.stem for p in pngs}
    mgmt = (rep / "MANAGEMENT_MODEL_REPORT_HE.html").read_text(encoding="utf-8")
    assert "יציבות הבחירה" in mgmt and "תרומה לחיזוי המודל" in mgmt and "אינה סיבתיות" in mgmt   # stability and importance kept apart
    assert "he_permutation" in (rep / "FIGURE_INDEX.md").read_text(encoding="utf-8")
    man = json.loads((rep / "model_report_manifest.json").read_text(encoding="utf-8"))
    assert man["primary_analysis"] == "EXTENDED" and man["privacy"]["identifier_scan"]["passed"] is True
    assert all(r["coefficient_path_reproduces_model"] is True for r in man["runs"]), man["runs"]
    lc = pd.read_csv(rep / "tables" / "EXTENDED_learning_curve.csv")
    assert list(lc["fraction"]) == [0.1, 0.2, 0.4, 0.6, 0.8, 1.0] and set(lc["mode"]) == {"reduced"} and (lc["status"] == "fitted").sum() >= 4
    assert (lc["n_train"].diff().dropna() > 0).all()
    comp = pd.read_csv(rep / "tables" / "model_comparison.csv").set_index("analysis")
    assert comp.at["STRICT", "comparison_note"].startswith("paired") and comp.at["EXTENDED", "comparison_note"] == "reference"
    paired = pd.read_csv(rep / "tables" / "paired_differences.csv")
    assert len(paired) == 1 and paired["auroc_ci_low"].iloc[0] <= paired["auroc_estimate"].iloc[0] <= paired["auroc_ci_high"].iloc[0]
    for name in ("MANAGEMENT_MODEL_REPORT_HE.html", "MANAGEMENT_MODEL_SUMMARY_HE.md", "EXECUTIVE_ONE_PAGER_HE.html"):
        text = (rep / name).read_text(encoding="utf-8")
        assert HEB.search(text) and "dir='rtl'" in text or 'dir="rtl"' in text
        low = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", text).lower()
        assert not [w for w in BANNED if w in low], name
    assert len(res.model_report["conclusions_he"]) >= 3
    for p in rep.glob("*.html"):
        body = p.read_text(encoding="utf-8")
        assert WM in body and "SYNTHETIC" in body
        assert not ID_RE.search(re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", body))
    one = (rep / "EXECUTIVE_ONE_PAGER_HE.html").read_text(encoding="utf-8")
    assert "@page{size:A4" in one


@pytest.mark.slow
def test_learning_curve_refuses_validation_rows_in_a_training_subset(explored) -> None:
    runs = discover_runs([explored["out"]])
    ext = next(r for r in runs if r.key == "EXTENDED")
    frames = C.load_frames(ext, [explored["out"]])
    assert frames.test_keys and not (set(zip(frames.train["research_id"].astype(str), frames.train["index_date"].dt.strftime("%Y-%m-%d"))) & frames.test_keys)
    leaky = C.Frames(spec=frames.spec, config=frames.config, train=pd.concat([frames.train, frames.validation.head(5)], ignore_index=True),
                     validation=frames.validation, test_keys=frames.test_keys, frame_features=frames.frame_features)
    lc, _info = C.learning_curve(ext, leaky, mode="reduced", fractions=(1.0,), n_boot=10)
    assert "LeakageError" in lc["status"].iloc[0]


@pytest.mark.slow
def test_standalone_command_refuses_inside_results_and_marks_unpaired_populations(explored, tmp_path) -> None:
    out = explored["out"]
    assert cli_main(["model-report", "--results", str(out), "--out", str(out / "reports2")]) == 2
    other = tmp_path / "other_results"
    run = next(r for r in discover_runs([out]) if r.key == "EXTENDED").run_dir
    shutil.copytree(run, other / "runs" / run.name)
    m = json.loads((other / "runs" / run.name / "metrics.json").read_text(encoding="utf-8"))
    m["experiment"]["name"] = "MEUHEDET_EFALLS_SAFE_ALL_ROWS_180D_SENSITIVITY"
    m["run_id"] = "copy-for-test"
    m["split"]["test_rows_sha256"] = "0" * 64
    (other / "runs" / run.name / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    rs = build_model_reports([out, other], out_dir=tmp_path / "reports", learning_curve="off", n_boot=50)
    comp = rs.comparison.set_index("analysis")
    assert comp.at["SAFE_ALL_ROWS", "comparison_note"].startswith("NOT A DIRECT PAIRED MODEL COMPARISON")
    assert "NOT A DIRECT PAIRED MODEL COMPARISON" in (tmp_path / "reports" / "MODEL_COMPARISON_REPORT.html").read_text(encoding="utf-8")
    safe = next(r for r in rs.runs if r.run.key == "SAFE_ALL_ROWS")
    assert any("dataset not available" in p or "test partition" in p for p in safe.problems)
