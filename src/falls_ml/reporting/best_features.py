"""Best-features evidence table across runs (``best_features.csv`` / ``best_features.md``).

One row per raw feature (feature-spec name). Evidence sources, never mixed up:
- the ORIGINAL published eFalls LASSO (fixed coefficients, S2 Table S3.2; sex term shown for ``lp_c_box_s3_1``, D-01),
  read from the coefficient file named in the published run's ``config.yaml`` (``model.published_equation.config``);
- the local retraining (``efalls_retrained``, full 78-predictor set only): coefficients and bootstrap stability (per raw
  feature: any design column, e.g. any fractional-polynomial form, selected). A run on a REDUCED eFalls predictor set
  (``efalls_retrained_reduced``) is never used as local eFalls LASSO evidence; it contributes permutation importance
  as a local model only;
- permutation importance of every model run on local data (published scoring, retraining, experimental algorithms);
  the "strongest" ranking averages only locally fitted models;
- the feature-group ablation (``ablation_dir/ablation_results.csv``) for Meuhedet-specific (L3b) additions.

Run selection: only completed runs; ablation members are excluded; for each model name one run is used — the first
by (kind: published < retrained < experimental < reduced retraining, experiment name) and, within that experiment, the
latest run id.
Used and ignored runs are listed in the Markdown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from falls_ml.logging_utils import get_logger
from falls_ml.models.published_efalls import CO_REPORTED_VARIANT, NON_BINARY_TERMS, PRIMARY_VARIANT, PublishedEfallsCoefficients
from falls_ml.paths import resolve_path
from falls_ml.reporting.compare import check_not_mixed, load_run_metrics
from falls_ml.reporting.report import (
    SYNTHETIC_BANNER, Block, Heading, Paragraph, as_bool, bullets, dig, fmt, read_optional_csv, table,
    write_document,
)

log = get_logger(__name__)

TOP_K = 10
NOT_AVAILABLE = "Not available"
KIND_PRIORITY = {"efalls_published_scoring": 0, "efalls_retrained": 1, "alternative_model": 2, "efalls_retrained_reduced": 3}
REDUCED_KIND = "efalls_retrained_reduced"
PUBLISHED_MODEL = "efalls_published"


def _select_runs(runs: list[tuple[Path, dict[str, Any]]]) -> tuple[dict[str, tuple[Path, dict[str, Any]]], list[str]]:
    eligible = [(p, m) for p, m in runs if dig(m, "experiment", "kind") in KIND_PRIORITY]
    ignored = [p.name for p, m in runs if dig(m, "experiment", "kind") not in KIND_PRIORITY]
    chosen: dict[str, tuple[Path, dict[str, Any]]] = {}
    for model in sorted({dig(m, "model", "name") for _, m in eligible}):
        group = [(p, m) for p, m in eligible if dig(m, "model", "name") == model]
        first = min(group, key=lambda pm: (KIND_PRIORITY[pm[1]["experiment"]["kind"]], pm[1]["experiment"]["name"]))
        same = [pm for pm in group if pm[1]["experiment"] == first[1]["experiment"]]
        chosen[model] = max(same, key=lambda pm: str(pm[1].get("run_id") or pm[0].name))
        ignored += [p.name for p, _ in group if p != chosen[model][0]]
    return chosen, sorted(ignored)


def _published_config_path(chosen: dict[str, tuple[Path, dict[str, Any]]], fallback: str | Path) -> tuple[Path, str]:
    """(coefficient file, provenance): the published run's ``model.published_equation.config``, else ``fallback``."""
    if PUBLISHED_MODEL in chosen:
        run_dir = chosen[PUBLISHED_MODEL][0]
        config = run_dir / "config.yaml"
        raw = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}) if config.is_file() else {}
        declared = dig(raw, "model", "published_equation", "config")
        if declared:
            return resolve_path(declared, anchor=run_dir), f"model.published_equation.config of run {run_dir.name}"
        log.warning("published_coefficients_from_fallback", extra_fields={"run_dir": str(run_dir), "fallback": str(fallback)})
    return resolve_path(fallback), "published_config argument (no published run config declares one)"


def _published(config: str | Path) -> pd.DataFrame:
    coefficients = PublishedEfallsCoefficients.from_yaml(config)
    _, beta = coefficients.design_coefficients(PRIMARY_VARIANT)
    _, beta_a = coefficients.design_coefficients(CO_REPORTED_VARIANT)
    rows = []
    for feature, cols in {**NON_BINARY_TERMS, **{b: (b,) for b in coefficients.binary}}.items():
        values = [beta[c] for c in cols]
        if feature == "sex":
            terms = f"sex=male: {beta['sex=male']:+.7g} ({PRIMARY_VARIANT}) vs {beta_a['sex=male']:+.7g} ({CO_REPORTED_VARIANT}); conflict D-01"
        else:
            terms = "; ".join(f"{c}={beta[c]:+.7g}" for c in cols) if len(cols) > 1 else ""
        rows.append({"feature": feature, "published_efalls_coefficient": values[0] if len(cols) == 1 else np.nan,
                     "published_efalls_terms": terms, "selected_in_published_lasso": any(v != 0.0 for v in values)})
    return pd.DataFrame(rows).set_index("feature")


def _retrained(run: Path) -> pd.DataFrame:
    out = pd.DataFrame(columns=["retrained_lasso_coefficient", "retrained_selected", "selection_frequency", "sign_stability", "robust"])
    coef = read_optional_csv(run / "coefficients.csv")
    if coef is not None:
        coef = coef.loc[coef["design_column"].astype(str) != "_intercept"]
        if "relative_to_reference" in coef.columns:
            coef = coef.loc[~as_bool(coef["relative_to_reference"])]
        grouped = coef.assign(nonzero=pd.to_numeric(coef["coefficient"]) != 0.0).groupby("feature", sort=False)
        summary = grouped.agg(n=("coefficient", "size"), first=("coefficient", "first"), selected=("nonzero", "any"))
        out = out.reindex(summary.index)
        out["retrained_lasso_coefficient"] = summary["first"].where(summary["n"] == 1)
        out["retrained_selected"] = summary["selected"]
    stability = read_optional_csv(run / "feature_stability.csv")
    if stability is not None:
        stability = stability.assign(robust=as_bool(stability["robust"]))
        best = stability.sort_values("selection_frequency", ascending=False, kind="stable").groupby("raw_feature", sort=False).first()
        out = out.reindex(out.index.union(best.index, sort=False))
        # any design column of the raw feature selected (e.g. any FP form); older runs only have per-column frequencies
        column = "raw_feature_selection_frequency" if "raw_feature_selection_frequency" in stability.columns else "selection_frequency"
        out["selection_frequency"] = stability.groupby("raw_feature")[column].max()
        out["sign_stability"] = best["sign_stability"]
        out["robust"] = stability.groupby("raw_feature")["robust"].any()
    return out


def _importance(model: str, run: Path) -> pd.DataFrame | None:
    fi = read_optional_csv(run / "feature_importance.csv")
    if fi is None:
        return None
    fi = fi.set_index("feature")
    rank = pd.to_numeric(fi["importance_rank"], errors="coerce") if "importance_rank" in fi.columns else pd.Series(np.nan, index=fi.index)
    if rank.isna().all():
        rank = fi["permutation_importance_mean"].rank(ascending=False, method="first")
    return pd.DataFrame({f"perm_{model}_mean": fi["permutation_importance_mean"], f"perm_{model}_std": fi["permutation_importance_std"],
                         f"importance_rank_{model}": rank})


def _ablation_text(ablation_dir: Path) -> tuple[pd.Series, pd.DataFrame | None]:
    results = read_optional_csv(ablation_dir / "ablation_results.csv")
    values: dict[str, str] = {}
    if results is None:
        log.warning("ablation_results_missing", extra_fields={"ablation_dir": str(ablation_dir),
                                                              "consequence": "question 7 (Meuhedet incremental value) is not answered"})
        return pd.Series(values, dtype="string"), None
    for _, r in results.iloc[1:].iterrows():
        added = r.get("added_features")
        if not isinstance(added, str) or added == "not available":
            continue
        text = (f"{r['step']}: AUROC {_signed(r['auroc_change_vs_previous_step'])} vs previous step; "
                f"{_signed_ci(r, 'auroc')} vs baseline")
        values.update({f: text for f in added.split(";") if f})
    return pd.Series(values, dtype="string"), results


def _signed(x: Any, dp: int = 3) -> str:
    return "not available" if x is None or pd.isna(x) else f"{float(x):+.{dp}f}"


def _signed_ci(r: pd.Series, metric: str) -> str:
    low, high = r[f"{metric}_delta_ci_low"], r[f"{metric}_delta_ci_high"]
    ci = "CI not available" if pd.isna(low) or pd.isna(high) else f"{low:.3f}–{high:.3f}"
    return f"{_signed(r[f'{metric}_delta'])} ({ci})"


def build_best_features(runs_dir: str | Path, out_dir: str | Path, *,
                        published_config: str | Path = "configs/models/efalls_published.yaml",
                        ablation_dir: str | Path | None = None) -> pd.DataFrame:
    """Write ``best_features.csv`` and ``best_features.md`` into ``out_dir``; return the per-feature table.

    Published coefficients come from the published run's ``model.published_equation.config`` (resolved against the
    run directory); ``published_config`` is used only without such a run. ``ablation_dir`` (default
    ``out_dir/ablation``) holds ``ablation_results.csv``; its absence is logged and shown as a warning.
    """
    runs, skipped = load_run_metrics(runs_dir)
    synthetic = check_not_mixed(runs)
    chosen, ignored = _select_runs(runs)
    out_dir = Path(out_dir)
    ablation_dir = out_dir / "ablation" if ablation_dir is None else Path(ablation_dir)
    coefficient_file, coefficient_source = _published_config_path(chosen, published_config)
    published_df = _published(coefficient_file)
    table_df = published_df
    retrained_run = next((p for p, m in chosen.values() if m["experiment"]["kind"] == "efalls_retrained"), None)
    parts = [_retrained(retrained_run)] if retrained_run is not None else []
    nonlinear = [model for model, (_, m) in chosen.items() if dig(m, "model", "is_linear") is False]
    importance = {model: _importance(model, p) for model, (p, _) in chosen.items()}
    parts += [imp for imp in importance.values() if imp is not None]
    for part in parts:
        table_df = table_df.join(part, how="outer", sort=False)
    incremental, ablation = _ablation_text(ablation_dir)
    extra = (set(table_df.index) | set(incremental.index)) - set(published_df.index)
    table_df = table_df.reindex(list(published_df.index) + sorted(extra))
    for col in ("retrained_lasso_coefficient", "retrained_selected", "selection_frequency", "sign_stability", "robust"):
        if col not in table_df.columns:
            table_df[col] = np.nan
    table_df["is_published_efalls_feature"] = table_df["published_efalls_terms"].notna()
    table_df["selected_in_published_lasso"] = table_df["selected_in_published_lasso"].astype("boolean").fillna(False)
    table_df["important_in_nonlinear"] = _important_in_nonlinear(table_df, nonlinear)
    table_df["little_or_no_value"] = _little_or_no_value(table_df, [m for m, imp in importance.items() if imp is not None])
    table_df["meuhedet_incremental_value"] = incremental.reindex(table_df.index)
    perm_cols = [c for m in sorted(chosen) for c in (f"perm_{m}_mean", f"perm_{m}_std", f"importance_rank_{m}") if c in table_df.columns]
    ordered = ["published_efalls_coefficient", "published_efalls_terms", "selected_in_published_lasso", "is_published_efalls_feature",
               "retrained_lasso_coefficient", "retrained_selected", "selection_frequency", "sign_stability", "robust", *perm_cols,
               "important_in_nonlinear", "little_or_no_value", "meuhedet_incremental_value"]
    result = table_df[ordered].reset_index(names="feature")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "best_features.csv", index=False, encoding="utf-8", lineterminator="\n")
    blocks = _document(result, chosen, ignored, skipped, retrained_run, nonlinear, ablation, ablation_dir,
                       f"{coefficient_file} ({coefficient_source})", synthetic)
    write_document(blocks, out_dir / "best_features.md", None, title="Best features")
    log.info("best_features_written", extra_fields={"out_dir": str(out_dir), "n_features": len(result), "models": sorted(chosen)})
    return result


def _important_in_nonlinear(df: pd.DataFrame, nonlinear: list[str]) -> pd.Series:
    """True when a nonlinear model ranks the feature in its top ``TOP_K`` with importance above one SD; NA without evidence."""
    models = [m for m in nonlinear if f"perm_{m}_mean" in df.columns]
    if not models:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    hits = pd.concat([(df[f"importance_rank_{m}"] <= TOP_K) & (df[f"perm_{m}_mean"] > df[f"perm_{m}_std"].fillna(0.0))
                      for m in models], axis=1).any(axis=1)
    has_evidence = df[[f"perm_{m}_mean" for m in models]].notna().any(axis=1)
    return pd.Series(np.where(has_evidence, hits, pd.NA), index=df.index, dtype="boolean")


def _little_or_no_value(df: pd.DataFrame, models: list[str]) -> pd.Series:
    """True when every local permutation importance is within one SD of zero and the local LASSO shows no support."""
    if not models:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    means = df[[f"perm_{m}_mean" for m in models]]
    stds = df[[f"perm_{m}_std" for m in models]].fillna(0.0).to_numpy()
    has_evidence = means.notna().any(axis=1)
    within_noise = ((means.to_numpy() <= stds) | means.isna().to_numpy()).all(axis=1)
    no_lasso_support = ~(df["retrained_selected"].astype("boolean").fillna(False) | df["robust"].astype("boolean").fillna(False))
    return pd.Series(np.where(has_evidence, within_noise & no_lasso_support, pd.NA), index=df.index, dtype="boolean")


# ============================================================================ document
def _cell(column: str, value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return "not available"
    if "rank" in column:
        return f"{float(value):.1f}"
    return fmt(value.item() if isinstance(value, np.generic) else value, 4)


def _rows(df: pd.DataFrame, cols: list[str]) -> list[tuple[str, ...]]:
    return [tuple(_cell(c, v) for c, v in zip(cols, r)) for r in df[cols].itertuples(index=False)]


def _document(df: pd.DataFrame, chosen: dict[str, tuple[Path, dict[str, Any]]], ignored: list[str], skipped: list[str],
              retrained_run: Path | None, nonlinear: list[str], ablation: pd.DataFrame | None, ablation_dir: Path,
              coefficient_source: str, synthetic: bool) -> list[Block]:
    models = [m for m in chosen if f"perm_{m}_mean" in df.columns]
    local = [m for m in models if m != PUBLISHED_MODEL]  # the published equation was not fitted locally
    rank_cols = [f"importance_rank_{m}" for m in local]
    blocks: list[Block] = [Paragraph(SYNTHETIC_BANNER, "banner")] if synthetic else []
    blocks += [Heading("Best features", 1), Paragraph(
        "Evidence per raw feature from: the ORIGINAL published eFalls LASSO (fixed coefficients); the local eFalls retraining "
        "(coefficients and bootstrap stability); permutation importance of each local model (drop in validation AUROC); and "
        "the feature-group ablation. Importance is predictive, not causal."),
        bullets([f"{model}: run {m.get('run_id', p.name)} ({m['experiment']['kind']}; dataset {dig(m, 'dataset', 'dataset_version')}, "
                 f"test rows {dig(m, 'split', 'test_rows_sha256')})" for model, (p, m) in chosen.items()]
                + [f"ignored (duplicate model or ablation member): {', '.join(ignored)}" if ignored else "ignored runs: none"]
                + ([f"skipped directories: {', '.join(skipped)}"] if skipped else [])
                + [f"published coefficients: {coefficient_source}"])]
    reduced = [(model, p, m) for model, (p, m) in chosen.items() if m["experiment"]["kind"] == REDUCED_KIND]
    for model, p, m in reduced:
        cov = m.get("efalls_coverage") or {}
        blocks.append(Paragraph(
            f"WARNING: run {m.get('run_id', p.name)} ({model}) uses a REDUCED eFalls predictor set – NOT a full eFalls reproduction "
            f"(available eFalls predictors: {cov.get('n_available', 'not available')} / {cov.get('n_total', 'not available')}). It is "
            "used as a local model for permutation importance only, never as the local eFalls LASSO evidence (questions 2 and 4); "
            "its unavailable predictors have no importance.", "warning"))
    if ablation is None:
        blocks.append(Paragraph(f"WARNING: no ablation_results.csv in {ablation_dir}; question 7 (Meuhedet-specific incremental value) "
                                "cannot be answered. Run the ablation first or pass the ablation output directory.", "warning"))

    versions = {(dig(m, "dataset", "dataset_version"), dig(m, "split", "test_rows_sha256")) for _, m in chosen.values()}
    if len(versions) > 1:
        blocks.append(Paragraph("WARNING: the evidence comes from runs on different dataset versions or test rows; permutation "
                                "importances are not directly comparable across these runs.", "warning"))
    blocks.append(Heading("1. Which features are strongest?", 2))
    if rank_cols:
        strong = df.assign(mean_rank=df[rank_cols].mean(axis=1)).dropna(subset=["mean_rank"]).sort_values("mean_rank", kind="stable").head(TOP_K)
        blocks += [Paragraph(f"Ranked by mean permutation-importance rank across {len(local)} locally fitted model(s) "
                             f"({', '.join(local)}; 1 = most important). The published eFalls equation has fixed coefficients and is "
                             "not part of this mean."),
                   table(("Feature", "Mean rank", *[f"Importance {m}" for m in local], "Retrained coefficient", "Published coefficient"),
                         _rows(strong, ["feature", "mean_rank", *[f"perm_{m}_mean" for m in local], "retrained_lasso_coefficient",
                                        "published_efalls_coefficient"]))]
    else:
        blocks.append(Paragraph(NOT_AVAILABLE, "note"))

    blocks.append(Heading("2. Which features are most stable?", 2))
    robust = df.loc[df["robust"].astype("boolean").fillna(False)]
    robust = robust.sort_values(["selection_frequency", "sign_stability"], ascending=False, kind="stable")
    if not robust.empty:
        blocks.append(Paragraph("Selection frequency per raw feature: share of bootstrap refits in which any of its design columns "
                                "(for example any fractional-polynomial form) was selected."))
        blocks.append(table(("Feature", "Selection frequency", "Sign stability"), _rows(robust, ["feature", "selection_frequency", "sign_stability"])))
    elif df["selection_frequency"].notna().any():
        blocks.append(Paragraph("No feature met the robustness criteria.", "note"))
    else:
        blocks.append(Paragraph(f"{NOT_AVAILABLE} (no bootstrap stability from a local eFalls retraining)", "note"))

    blocks.append(Heading("3. Which features were selected by the published eFalls LASSO?", 2))
    published = df.loc[df["is_published_efalls_feature"]]
    selected = published.loc[published["selected_in_published_lasso"]]
    not_selected = published.loc[~published["selected_in_published_lasso"]]
    top = selected.iloc[np.argsort(-selected["published_efalls_coefficient"].abs().fillna(0.0).to_numpy(), kind="stable")].head(15)
    blocks += [Paragraph(f"{len(selected)} of {len(published)} candidate predictors have a non-zero published coefficient. Largest single-term "
                         "coefficients (log-odds; multi-level terms listed separately):"),
               table(("Feature", "Published coefficient", "Terms"), _rows(top, ["feature", "published_efalls_coefficient", "published_efalls_terms"])),
               Paragraph("Not selected (coefficient 0): " + (", ".join(not_selected["feature"]) or "none"))]

    blocks.append(Heading("4. Which features were important in the local LASSO?", 2))
    lasso_rank = "importance_rank_lasso_logistic_cv"
    if retrained_run is not None and lasso_rank in df.columns:
        local = df.loc[df["retrained_selected"].astype("boolean").fillna(False) & (df[lasso_rank] <= TOP_K)].sort_values(lasso_rank, kind="stable")
        blocks.append(table(("Feature", "Rank", "Retrained coefficient", "Selection frequency", "Published coefficient"),
                            _rows(local, ["feature", lasso_rank, "retrained_lasso_coefficient", "selection_frequency", "published_efalls_coefficient"]))
                      if not local.empty else Paragraph("No selected feature ranked in the top 10.", "note"))
    else:
        blocks.append(Paragraph(f"{NOT_AVAILABLE} (no local eFalls retraining run)", "note"))

    blocks.append(Heading("5. Which features were important in nonlinear models?", 2))
    models_nl = [m for m in nonlinear if f"perm_{m}_mean" in df.columns]
    if models_nl:
        nl = df.loc[df["important_in_nonlinear"].fillna(False)]
        nl = nl.assign(best_rank=nl[[f"importance_rank_{m}" for m in models_nl]].min(axis=1)).sort_values("best_rank", kind="stable")
        cols = [c for m in models_nl for c in (f"importance_rank_{m}", f"perm_{m}_mean")]
        headers = [h for m in models_nl for h in (f"Rank {m}", f"Importance {m}")]
        blocks.append(table(("Feature", *headers), _rows(nl, ["feature", *cols])) if not nl.empty
                      else Paragraph(f"No feature ranked in the top {TOP_K} with importance above one SD.", "note"))
    else:
        blocks.append(Paragraph(f"{NOT_AVAILABLE} (no nonlinear model run)", "note"))

    blocks.append(Heading("6. Which features had little or no value?", 2))
    little = df.loc[df["little_or_no_value"].fillna(False)]
    blocks.append(Paragraph("Permutation importance within one SD of zero in every local model and no support in the local LASSO: "
                            + (", ".join(little["feature"]) if not little.empty else "none") + "."
                            if df["little_or_no_value"].notna().any() else NOT_AVAILABLE))

    blocks.append(Heading("7. Which Meuhedet-specific features improve beyond pure eFalls?", 2))
    if ablation is None:
        blocks.append(Paragraph(f"{NOT_AVAILABLE}: no ablation_results.csv in {ablation_dir}", "warning"))
    else:
        base = ablation.iloc[0]
        model = base["model"] if "model" in ablation.columns and isinstance(base["model"], str) else "model not recorded"
        blocks.append(Paragraph(f"Read from {ablation_dir / 'ablation_results.csv'}. Baseline: {model}, run {base['run_id']} (the ablation's "
                                "own fitted model on its test rows; not the published eFalls equation). Each step adds feature groups to "
                                "the previous step; changes are step minus baseline with paired bootstrap CIs."))
        steps = ablation.iloc[1:]
        blocks.append(table(("Step", "Added features", "AUROC change vs baseline (95% CI)", "Brier change", "Calibration slope change", "Verdict"),
                            [(r["step"], r["added_features"] if isinstance(r["added_features"], str) else "-",
                              _signed_ci(r, "auroc"),
                              _signed(r["brier_delta"], 4), _signed(r["calibration_slope_delta"]),
                              "improves AUROC beyond pure eFalls (CI excludes 0)" if r["auroc_delta_ci_low"] > 0 else "no clear improvement")
                             for _, r in steps.iterrows()]) if not steps.empty else Paragraph("No ablation steps.", "note"))
    return blocks
