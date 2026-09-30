"""Per-run report: ``report.md`` and a self-contained ``report.html`` (architecture §3.4, §4 step 10).

The report is built once as a small intermediate document (headings, paragraphs, bullets, tables, images) and
rendered twice: Markdown with relative image paths, and HTML with inline CSS, base64-embedded PNGs, escaped
text and no external resources. Input contract: ``docs/ARTIFACT_SCHEMAS.md``.

Reading rules: ``metrics.json`` is required (``FileNotFoundError`` otherwise); every other artifact is optional
and its absence is shown as "Not available for this run". The three result types are never confused (spec
§1.1/§1.2): the published equation (transportability, D-01, M-11), local retraining, and experimental algorithms.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from falls_ml.errors import FallsMLError
from falls_ml.logging_utils import get_logger

log = get_logger(__name__)

NOT_AVAILABLE = "Not available for this run"
SYNTHETIC_BANNER = "SYNTHETIC FIXTURE — SOFTWARE TEST OUTPUT, NOT SCIENTIFIC EVIDENCE"
PARTIAL_LABEL = "efalls_partial_scoring"
EXECUTIVE_SUMMARY = "Executive Summary"
SECTION_TITLES = (
    "1. Dataset", "2. Cohort", "3. Feature definitions", "4. Missingness", "5. Split methodology", "6. Model",
    "7. Hyperparameter search", "8. Cross-validation", "9. Test performance", "10. Calibration",
    "11. Feature importance", "12. Feature stability", "13. Error analysis", "14. Comparison to eFalls",
    "15. Comparison to alternative algorithms", "16. Limitations", "17. Production readiness",
)
KIND_TITLES = {
    "efalls_published_scoring": "Scientific reproduction: ORIGINAL published eFalls equation (transportability)",
    "efalls_retrained": "Local retraining: eFalls predictors and learning process, coefficients fitted locally",
    "efalls_retrained_reduced": "Local retraining on a REDUCED eFalls predictor set (NOT a full eFalls reproduction)",
    "alternative_model": "Experimental algorithm (not eFalls)",
    "ablation_member": "Ablation step: experimental algorithm with added feature groups (not eFalls)",
}
KIND_STATEMENTS = {
    "efalls_published_scoring": (
        "This report measures the TRANSPORTABILITY of the ORIGINAL published eFalls equation: the fixed published "
        "coefficients (Archer et al., Age Ageing 2024) were applied to this data. No coefficient was estimated from "
        "these outcomes. Any recalibration was fitted on the validation split only and is reported separately."),
    "efalls_retrained": (
        "This report describes a LOCAL RETRAINING of eFalls: the eFalls predictor definitions and published learning "
        "process were used, but the coefficients were fitted locally on this dataset's training split. It is not the "
        "published eFalls equation."),
    "efalls_retrained_reduced": (
        "This report describes a LOCAL RETRAINING on a REDUCED eFalls predictor set: the published eFalls learning process was "
        "used, but only a declared subset of the eFalls candidate predictors was available in this dataset, and the coefficients "
        "were fitted locally on its training split. It is NOT a full eFalls reproduction and NOT the published eFalls equation; "
        "its results must not be reported as eFalls performance."),
    "alternative_model": (
        "This report describes an EXPERIMENTAL ALGORITHM evaluated on the same partitions as the eFalls experiments. "
        "It is neither the published eFalls equation nor a retrained eFalls model."),
    "ablation_member": (
        "This report describes one step of a feature-group ABLATION (experimental algorithm). It is neither the "
        "published eFalls equation nor a retrained eFalls model."),
}
MODEL_LABELS = {
    "efalls_published": "Published eFalls equation (fixed coefficients)",
    "lasso_logistic_cv": "LASSO logistic regression (lambda chosen by cross-validation)",
    "logistic_unpenalized": "Logistic regression (unpenalised)",
    "elastic_net_logistic": "Elastic-net logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
}
REDUCED_KIND = "efalls_retrained_reduced"
RETRAINED_KINDS = frozenset({"efalls_retrained", REDUCED_KIND})
EFALLS_KINDS = frozenset({"efalls_published_scoring", *RETRAINED_KINDS})
REDUCED_WARNING = "REDUCED eFalls predictor set – NOT a full eFalls reproduction"
PARTIAL_WARNING = (
    "EFFECTIVE EXPERIMENT LABEL: efalls_partial_scoring. A mandatory eFalls predictor is unavailable or the "
    "feature coverage is below the M-11 threshold. This run is NOT a validation of eFalls.")
SERVED_EXPLANATION = (
    "Served variant: the prediction the saved model bundle returns by default. The executive summary, eligibility check, "
    "candidate thresholds, plots and the master comparison use it; other variants are reported alongside.")
TRANSPORTABILITY_EXPLANATION = (
    " Transportability variant: the unmodified published eFalls equation, which measures how well the original equation "
    "transports to these data. Unless a recalibration is served, the served prediction IS the published equation.")
CALIBRATION_STATUS_TEXT = {
    "adequate": "adequate", "miscalibrated": "miscalibrated",
    "not_assessed": "not assessed (calibration slope or calibration-in-the-large could not be estimated)"}
COUNT_COLUMNS = {"n", "n_events", "n_rows", "n_patients", "tp", "fp", "tn", "fn"}
PAIRED_CI_NOTE = ("point estimate only; paired patient-level bootstrap CIs are produced by the master comparison "
                  "(falls_ml compare) in reports/comparison_paired_differences.csv, not in this report")
STANDARD_LIMITATIONS = (
    "eFalls performed no temporal validation, so temporal results have no published comparator (D-19).",
    "The eFalls publications conflict with each other (sex term and intercept, D-01; BMI cut-points, D-02; "
    "candidate count), and several definitions are documented assumptions (spec §9).",
)


class ArtifactError(FallsMLError, ValueError):
    """Run artifacts are inconsistent or violate ``docs/ARTIFACT_SCHEMAS.md``."""


# ============================================================================ document model
@dataclass(frozen=True)
class Heading:
    text: str
    level: int = 2


@dataclass(frozen=True)
class Paragraph:
    text: str
    style: Literal["normal", "banner", "warning", "note"] = "normal"


@dataclass(frozen=True)
class Bullets:
    items: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Image:
    path: str  # relative to the document directory, POSIX separators
    caption: str


Block = Heading | Paragraph | Bullets | Table | Image


def bullets(items: Iterable[Any]) -> Bullets:
    return Bullets(tuple(str(i) for i in items))


def table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> Table:
    rows_t = tuple(tuple(str(c) for c in r) for r in rows)
    if any(len(r) != len(columns) for r in rows_t):
        raise ArtifactError(f"table rows must have {len(columns)} cells")
    return Table(tuple(columns), rows_t)


# ============================================================================ rendering
def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(blocks: Sequence[Block]) -> str:
    """Markdown with relative image links."""
    parts: list[str] = []
    for b in blocks:
        if isinstance(b, Heading):
            parts.append(f"{'#' * b.level} {b.text}")
        elif isinstance(b, Paragraph):
            parts.append({"banner": f"{b.text}\n{'=' * len(b.text)}", "warning": f"> **{b.text}**", "note": f"*{b.text}*"}.get(b.style, b.text))
        elif isinstance(b, Bullets):
            parts.append("\n".join(f"- {_md_cell(i)}" for i in b.items) if b.items else NOT_AVAILABLE)
        elif isinstance(b, Table):
            lines = ["| " + " | ".join(_md_cell(c) for c in b.columns) + " |", "|" + "---|" * len(b.columns)]
            lines += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in b.rows]
            parts.append("\n".join(lines))
        elif isinstance(b, Image):
            parts.append(f"![{_md_cell(b.caption)}]({b.path})")
        else:
            raise ArtifactError(f"unknown document block {type(b).__name__}")
    return "\n\n".join(parts) + "\n"


_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#f7f7f5;color:#1d1d1d;line-height:1.45}
main{max-width:1100px;margin:0 auto;padding:24px 32px 64px;background:#fff}
h1{font-size:1.7em;margin-top:.6em}h2{border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:2em}h3{margin-top:1.4em}
.banner{background:#b00020;color:#fff;font-weight:700;padding:12px 16px;font-size:1.15em;text-align:center;margin:0 0 12px}
.warning{background:#fff3cd;border-left:6px solid #d39e00;padding:10px 14px;font-weight:700}
.note{color:#555;font-style:italic}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;margin:8px 0 16px;font-size:.9em}
th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f0f0f0}
figure{margin:12px 0}img{max-width:100%;height:auto;border:1px solid #eee}figcaption{font-size:.85em;color:#555}
"""


def _image_data_uri(base_dir: Path, rel: str) -> str:
    rel_path = PurePosixPath(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts or rel_path.suffix.lower() != ".png":
        raise ArtifactError(f"image path must be a relative .png inside the document directory: {rel!r}")
    data = (base_dir / Path(*rel_path.parts)).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def render_html(blocks: Sequence[Block], *, title: str, base_dir: Path) -> str:
    """Self-contained HTML: inline CSS, embedded PNGs, escaped text, no external resources."""
    esc = html.escape
    body: list[str] = []
    for b in blocks:
        if isinstance(b, Heading):
            body.append(f"<h{b.level}>{esc(b.text)}</h{b.level}>")
        elif isinstance(b, Paragraph):
            tag = "div" if b.style == "banner" else "p"
            body.append(f'<{tag} class="{b.style}">{esc(b.text)}</{tag}>')
        elif isinstance(b, Bullets):
            body.append("<ul>" + "".join(f"<li>{esc(i)}</li>" for i in b.items) + "</ul>" if b.items else f"<p>{NOT_AVAILABLE}</p>")
        elif isinstance(b, Table):
            head = "".join(f"<th>{esc(c)}</th>" for c in b.columns)
            rows = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in b.rows)
            body.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')
        elif isinstance(b, Image):
            body.append(f'<figure><img alt="{esc(b.caption)}" src="{_image_data_uri(base_dir, b.path)}">'
                        f"<figcaption>{esc(b.caption)}</figcaption></figure>")
        else:
            raise ArtifactError(f"unknown document block {type(b).__name__}")
    return ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f"<title>{esc(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
            + "\n".join(body) + "\n</main>\n</body>\n</html>\n")


def write_document(blocks: Sequence[Block], md_path: Path, html_path: Path | None, *, title: str) -> tuple[Path, Path | None]:
    """Write Markdown (and optionally HTML); image paths are resolved relative to ``md_path.parent``."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(blocks), encoding="utf-8", newline="\n")
    if html_path is not None:
        html_path.write_text(render_html(blocks, title=title, base_dir=md_path.parent), encoding="utf-8", newline="\n")
    return md_path, html_path


# ============================================================================ formatting and artifact helpers
def fmt(x: Any, dp: int = 3) -> str:
    """Numbers to ``dp`` decimals; ints verbatim; None/NaN -> 'not available'; bools -> yes/no."""
    if x is None:
        return "not available"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{dp}f}" if math.isfinite(x) else "not available"
    if isinstance(x, (dict, list, tuple)):
        return json.dumps(x, sort_keys=True, default=str) if x else "none"
    return str(x)


def fmt_ci(entry: Any) -> str:
    """``{estimate, ci_low, ci_high}`` -> 'x (lo–hi)' to 3 dp."""
    if not isinstance(entry, Mapping):
        return fmt(entry)
    est, low, high = entry.get("estimate"), entry.get("ci_low"), entry.get("ci_high")
    if est is None:
        return "not available"
    if low is None or high is None:
        return f"{est:.3f} (CI not available)"
    return f"{est:.3f} ({low:.3f}–{high:.3f})"


def fmt_pct(x: Any) -> str:
    return "not available" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * float(x):.1f}%"


def estimate(entry: Any) -> float | None:
    """Point estimate of a metrics.json measure (dict or scalar)."""
    value = entry.get("estimate") if isinstance(entry, Mapping) else entry
    return None if value is None or (isinstance(value, float) and not math.isfinite(value)) else float(value)


def dig(d: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, Mapping) or d.get(k) is None:
            return default
        d = d[k]
    return d


def as_bool(series: pd.Series) -> pd.Series:
    """Booleans from bool columns or CSV round-tripped strings ('True'/'False'); missing -> False."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "1.0"]).fillna(False).astype(bool)


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as dicts with missing cells as None (safe for column names that are not identifiers)."""
    return [{k: (None if _is_missing(v) else v) for k, v in row.items()} for row in df.to_dict("records")]


def _is_missing(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def read_metrics(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"metrics.json not found in run directory {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    """CSV artifact or None when missing or header-only."""
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def config_sha256(run_dir: Path) -> str | None:
    """SHA-256 of the saved config (identical to ``ExperimentConfig.sha256`` of the run's config)."""
    path = Path(run_dir) / "config.yaml"
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ============================================================================ run report
@dataclass(frozen=True)
class _Run:
    path: Path
    m: dict[str, Any]

    @property
    def kind(self) -> str:
        kind = dig(self.m, "experiment", "kind")
        if kind not in KIND_TITLES:
            raise ArtifactError(f"{self.path}: unknown experiment kind {kind!r}")
        return kind

    @property
    def published(self) -> bool:
        return self.kind == "efalls_published_scoring"

    @property
    def synthetic(self) -> bool:
        return bool(self.m.get("synthetic_fixture"))

    @property
    def primary(self) -> str:
        return str(self.m.get("primary_variant") or "uncalibrated")

    @property
    def served(self) -> str:
        """Variant the bundle serves (``served_variant``; equals ``primary_variant`` in current runs)."""
        return str(self.m.get("served_variant") or self.primary)

    @property
    def ps(self) -> dict[str, Any]:
        return self.m.get("published_scoring") or {}

    @property
    def reduced(self) -> bool:
        return self.kind == REDUCED_KIND

    @property
    def coverage(self) -> dict[str, Any] | None:
        """``metrics.efalls_coverage`` (eFalls kinds; None for other kinds and for runs written before it existed)."""
        return self.m.get("efalls_coverage") or None

    def perf(self, split: str, variant: str | None = None) -> dict[str, Any] | None:
        return dig(self.m, "performance", split, variant or self.primary)

    def csv(self, name: str) -> pd.DataFrame | None:
        return read_optional_csv(self.path / name)

    def config(self) -> dict[str, Any]:
        """Saved ``config.yaml`` as a raw mapping ({} when absent)."""
        path = self.path / "config.yaml"
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}

    def unavailable_predictors(self) -> tuple[list[dict[str, Any]], list[str]]:
        """(per-predictor M-11 rows, declared names); names fall back to the model params when rows were not recorded."""
        rows = list(self.ps.get("unavailable_predictors") or [])
        names = [str(r.get("feature")) for r in rows] or [str(n) for n in dig(self.m, "model", "params", "unavailable_predictors", default=[]) or []]
        return rows, names

    def plot(self, name: str, caption: str) -> list[Block]:
        if (self.path / "plots" / name).is_file():
            return [Image(f"plots/{name}", caption)]
        return [Paragraph(f"{caption}: {NOT_AVAILABLE}", "note")]

    def count(self, x: Any) -> str:
        """Counts with M-13 small-cell suppression (1–9 shown as '<10') unless the data are synthetic."""
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "not available"
        n = int(x)
        return "<10" if not self.synthetic and 0 < n < 10 else str(n)


def _na() -> list[Block]:
    return [Paragraph(NOT_AVAILABLE, "note")]


def _kv(rows: Iterable[tuple[str, Any]]) -> Table:
    return table(("Item", "Value"), [(k, v if isinstance(v, str) else fmt(v)) for k, v in rows])


def _variant_role(run: _Run, variant: str) -> str:
    ps = run.ps
    base = variant.split("+")[0]
    role = ("primary: published as validated (Box S3.1)" if base == ps.get("primary") else
            "co-reported: published as tabulated (Table S3.2)" if base == ps.get("co_reported") else
            "CITL-shift variant (D-01)" if run.published else "primary" if variant == run.primary else "secondary")
    return (role + ("; recalibrated on validation split" if variant.endswith("recalibrated") else "")
            + ("; served by the model bundle" if variant == run.served else ""))


def _status_text(status: Any) -> str:
    return "not available" if status is None else CALIBRATION_STATUS_TEXT.get(str(status), str(status))


def _served_text(run: _Run) -> str:
    served = run.served
    if run.published:
        if "recalibrated" in served:
            return (f"{served}: a LOCAL RECALIBRATION of the published equation (fitted on the validation split), not the published "
                    f"equation itself; the unmodified equation ({fmt(run.m.get('transportability_variant'))}) is the transportability result")
        return f"{served}: the published eFalls equation itself (no recalibration is served)"
    return f"{served}: " + ("model output recalibrated on the validation split" if "recalibrated" in served else
                            "uncalibrated model output (no recalibration is served)")


def _g(x: Any) -> str:
    return "not available" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{float(x):.4g}"


def _has_content(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_content(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_content(v) for v in value)
    return bool(value)


def _omissions_text(value: Any) -> str:
    if value is None:
        return "not available"
    return json.dumps(value, sort_keys=True, default=str) if _has_content(value) else "none"


def _metrics_table(run: _Run, split: str, variants: Sequence[str]) -> Table:
    rows = []
    for v in variants:
        e = run.perf(split, v)
        if e is None:
            continue
        rows.append((v, _variant_role(run, v), run.count(e.get("n")), run.count(e.get("n_events")), fmt_ci(e.get("auroc")),
                     fmt_ci(e.get("pr_auc")), fmt_ci(e.get("brier")), fmt_ci(e.get("calibration_slope")),
                     fmt_ci(e.get("citl")), fmt_ci(e.get("oe_ratio")), fmt(dig(e, "calibration_indices", "ici"))))
    return table(("Variant", "Role", "n", "Events", "AUROC", "PR-AUC", "Brier", "Calibration slope", "CITL", "O/E", "ICI"), rows)


def _coverage_counts(cov: Mapping[str, Any]) -> str:
    return f"{fmt(cov.get('n_available'))} / {fmt(cov.get('n_total'))}"


def _coverage_pct(cov: Mapping[str, Any]) -> str:
    pct = cov.get("coverage_pct")
    return "not available" if pct is None else f"{float(pct):.1f}%"


def _unavailable_text(cov: Mapping[str, Any]) -> str:
    return ", ".join(map(str, cov.get("unavailable") or [])) or "none"


def _reduced_warning(run: _Run) -> list[Block]:
    """Prominent warning for reduced eFalls predictor sets (and for missing mandatory eFalls predictors)."""
    cov = run.coverage or {}
    blocks: list[Block] = []
    if run.reduced:
        blocks.append(Paragraph(f"{REDUCED_WARNING}. Available eFalls predictors: {_coverage_counts(cov)} "
                                f"({_coverage_pct(cov)}). Unavailable eFalls predictors: {_unavailable_text(cov)}. The coefficients "
                                "were fitted locally on this reduced predictor set; do not report these results as eFalls.", "warning"))
    if cov.get("mandatory_unavailable"):
        blocks.append(Paragraph(f"WARNING: mandatory eFalls predictors are unavailable (cohort.mandatory_for_efalls_label): "
                                f"{', '.join(map(str, cov['mandatory_unavailable']))}. No eFalls label applies to this run.", "warning"))
    return blocks


def _coverage_rows(run: _Run) -> list[tuple[str, str]]:
    cov = run.coverage
    if cov is None:
        return []
    return [("Available eFalls predictors", _coverage_counts(cov)), ("Coverage percentage", _coverage_pct(cov)),
            ("Unavailable eFalls predictors", _unavailable_text(cov))]


def exploratory_outcome_banner(metrics: dict[str, Any]) -> str | None:
    """Watermark for runs whose outcome is not the published eFalls outcome (metrics.outcome.published_efalls_outcome false)."""
    outcome = metrics.get("outcome") or {}
    if not outcome or outcome.get("published_efalls_outcome", True):
        return None
    horizon = str(outcome.get("horizon", ""))
    days = f"{int(horizon[:-1]) - 1}-DAY " if horizon.endswith("d") and horizon[:-1].isdigit() else ""
    return f"EXPLORATORY {days}OUTCOME ({outcome.get('name')}) – NOT EFALLS REPRODUCTION"


def _header(run: _Run) -> list[Block]:
    m = run.m
    blocks: list[Block] = [Paragraph(SYNTHETIC_BANNER, "banner")] if run.synthetic else []
    watermark = exploratory_outcome_banner(m)
    if watermark:
        blocks.append(Paragraph(watermark, "banner"))
    blocks.append(Heading(f"Run report: {dig(m, 'experiment', 'name', default='unnamed experiment')}", 1))
    if run.ps.get("effective_experiment_label") == PARTIAL_LABEL:
        blocks.append(Paragraph(PARTIAL_WARNING, "warning"))
    blocks += _reduced_warning(run)
    if m.get("warnings"):
        blocks += [Paragraph(f"RUN WARNINGS ({len(m['warnings'])}): the pipeline recorded the following warnings; read them before "
                             "using any result.", "warning"), bullets(m["warnings"])]
    rows = [("Run id", fmt(m.get("run_id"))), ("Created (UTC)", fmt(m.get("created_utc"))),
            ("Experiment id and kind", f"{dig(m, 'experiment', 'name')} ({run.kind})"), ("Result type", KIND_TITLES[run.kind]),
            ("Layers", ", ".join(dig(m, "experiment", "layers", default=[])) or "not available"),
            ("Config SHA-256", config_sha256(run.path) or NOT_AVAILABLE),
            ("Synthetic data watermark", "SYNTHETIC FIXTURE" if run.synthetic else "none (dataset source: "
             f"{dig(m, 'dataset', 'source', default='not available')})"),
            ("Outcome watermark", watermark or f"published eFalls outcome ({dig(m, 'outcome', 'name', default='outcome_12m')})")]
    if run.published:
        rows += [("Sex parameterisation", f"primary {run.ps.get('primary')}; co-reported {run.ps.get('co_reported')} (D-01)"),
                 ("Effective experiment label", fmt(run.ps.get("effective_experiment_label"))),
                 ("eFalls feature coverage", fmt(run.ps.get("efalls_feature_coverage")))]
    else:
        rows.append(("Feature set", f"{dig(m, 'feature_set', 'name')} {dig(m, 'feature_set', 'version')} "
                                    f"(pure eFalls: {fmt(dig(m, 'feature_set', 'pure_efalls'))})"))
    if run.coverage is not None:
        rows.append(("eFalls predictor coverage", f"{_coverage_counts(run.coverage)} ({_coverage_pct(run.coverage)}); "
                                                  f"{fmt(run.coverage.get('label'))}"))
    return blocks + [_kv(rows)]


def _best_hyperparameters(run: _Run) -> str:
    if run.published:
        return "Not applicable: fixed published coefficients (nothing was tuned)"
    parts = []
    lasso = run.m.get("lasso")
    if lasso and lasso.get("lambda_star") is not None:
        parts.append(f"lambda* = {lasso['lambda_star']:.4g} chosen by {lasso.get('cv_folds')}-fold cross-validation")
    best = dig(run.m, "model", "best_params")
    if best:
        parts.append(", ".join(f"{k}={v}" for k, v in sorted(best.items())))
    return "; ".join(parts) or "None: no hyperparameter search (configured parameters used)"


def _improvement(run: _Run) -> str:
    if run.published:
        if run.ps.get("effective_experiment_label") == PARTIAL_LABEL:
            return "Not applicable: this run scores the published equation with incomplete coverage (efalls_partial_scoring)"
        return "Not applicable: this run is the published eFalls baseline"
    comp = run.m.get("comparison_to_efalls") or {}
    delta = comp.get("delta_auroc_test")
    if delta is None:
        return "Unknown: no published-eFalls run on identical test rows was available"
    verdict = "Yes" if delta > 0 else "No"
    return (f"{verdict}: AUROC {delta:+.3f} vs published eFalls ({comp.get('reference_variant') or 'primary variant'}, run "
            f"{comp.get('reference_run_id')}) on identical test rows ({PAIRED_CI_NOTE}); calibration status of this run: "
            f"{_status_text(run.m.get('calibration_status'))}. Calibration of both models is compared in section 14")


def _executive_summary(run: _Run) -> list[Block]:
    m, test = run.m, run.perf("test") or {}
    cohort = m.get("cohort") or {}
    blocks: list[Block] = [Heading(EXECUTIVE_SUMMARY), Paragraph(KIND_STATEMENTS[run.kind]),
                           Paragraph(SERVED_EXPLANATION + (TRANSPORTABILITY_EXPLANATION if run.published else ""), "note")]
    if run.ps.get("effective_experiment_label") == PARTIAL_LABEL:
        blocks.append(Paragraph(PARTIAL_WARNING, "warning"))
    blocks += _reduced_warning(run)
    model = dig(m, "model", "name", default="not available")
    elig = m.get("eligibility") or {}
    eligible = elig.get("eligible_for_further_validation")
    top = (dig(m, "features", "top_features", default=[]) or [])[:10]
    rows = [
        ("Experiment name", fmt(dig(m, "experiment", "name"))),
        ("Type of result", KIND_TITLES[run.kind] + ("; labelled efalls_partial_scoring: NOT a validation of eFalls (M-11)"
                                                     if run.ps.get("effective_experiment_label") == PARTIAL_LABEL else "")),
        ("Algorithm", f"{MODEL_LABELS.get(model, model)} [{model}]"),
        ("Data version", f"{dig(m, 'dataset', 'dataset_version', default='not available')} "
                         f"(mapping {dig(m, 'dataset', 'mapping_version', default='not available')})"),
        *_coverage_rows(run),
        ("Number of patients", f"{run.count(cohort.get('n_patients'))} ({run.count(cohort.get('n_rows'))} patient-index rows)"),
        ("Outcomes (fall/fracture within 12 months)", f"{run.count(cohort.get('n_events'))} ({fmt_pct(cohort.get('prevalence'))})"),
        ("Results below are on", f"test split, {run.count(test.get('n'))} rows, {run.count(test.get('n_events'))} outcomes, "
                                 f"served prediction variant {run.primary}"),
        ("Served prediction (what the model bundle returns)", _served_text(run)),
        ("Main AUROC (0.5 = chance, 1 = perfect)", fmt_ci(test.get("auroc"))),
        ("PR-AUC", fmt_ci(test.get("pr_auc"))),
        ("Brier score (lower is better)", fmt_ci(test.get("brier"))),
        ("Calibration status", f"{_status_text(m.get('calibration_status'))} (slope {fmt(estimate(test.get('calibration_slope')))}, "
                               f"calibration-in-the-large {fmt(estimate(test.get('citl')))})"),
        ("Best hyperparameters", _best_hyperparameters(run)),
        ("Top 10 features", ", ".join(top) if top else "not available"),
        ("Improved over the eFalls baseline?", _improvement(run)),
        ("Eligible for further validation?", "not available" if eligible is None else
         "Yes" if eligible else "No: " + "; ".join(elig.get("reasons") or ["no reason recorded"])),
    ]
    if run.published:
        co = run.perf("test", run.ps.get("co_reported"))
        rows.insert(2, ("Effective experiment label", fmt(run.ps.get("effective_experiment_label"))))
        after_auroc = next(i for i, (k, _) in enumerate(rows) if k.startswith("Main AUROC")) + 1
        rows.insert(after_auroc, ("Co-reported AUROC (" + str(run.ps.get("co_reported")) + ")", fmt_ci((co or {}).get("auroc"))))
        transport = run.m.get("transportability_variant")
        if transport and transport != run.primary:
            t_entry = run.perf("test", transport) or {}
            rows.insert(after_auroc, (f"Transportability of the unmodified published equation ({transport})",
                             f"AUROC {fmt_ci(t_entry.get('auroc'))}; calibration slope {fmt(estimate(t_entry.get('calibration_slope')))}; "
                             f"CITL {fmt(estimate(t_entry.get('citl')))}; status {_status_text(m.get('calibration_status_uncalibrated'))}"))
    return blocks + [_kv(rows)]


def _dataset(run: _Run) -> list[Block]:
    ds = run.m.get("dataset")
    if not ds:
        return _na()
    fs = run.m.get("feature_set") or {}
    code = run.m.get("code_version") or {}
    rows = [(k, fmt(v)) for k, v in ds.items() if k != "audit"]
    rows += [("feature set", f"{fs.get('name')} {fs.get('version')} (sha256 {fs.get('sha256')})"),
             ("code version", fmt(code.get("git_commit") or code.get("source_tree_sha256")))]
    blocks: list[Block] = [_kv(rows)]
    audit = ds.get("audit") or {}
    exclusions = audit.get("derivation_exclusions")
    if exclusions:
        blocks.append(Paragraph("Cohort derivation exclusions (counted by reason; no row is dropped silently):"))
        blocks.append(table(("Reason", "Rows excluded"), [(k, run.count(v)) for k, v in sorted(exclusions.items())]))
    outcome_audit = audit.get("outcome_audit_d09")
    if outcome_audit:
        summary = pd.DataFrame(outcome_audit).groupby(["kind", "status"], as_index=False)["n_records"].sum()
        blocks.append(Paragraph("Pre-modelling outcome audit (spec D-09): candidate fall/fracture code records by outcome status."))
        blocks.append(table(("Code kind", "Status", "Records"),
                            [(r.kind, r.status, run.count(int(r.n_records))) for r in summary.itertuples(index=False)]))
    return blocks


def _cohort(run: _Run) -> list[Block]:
    c = run.m.get("cohort")
    if not c:
        return _na()
    blocks: list[Block] = [_kv([("Rows (patient-index dates)", run.count(c.get("n_rows"))), ("Patients", run.count(c.get("n_patients"))),
                                ("Outcomes", run.count(c.get("n_events"))), ("Outcome rate", fmt_pct(c.get("prevalence")))])]
    by_split = c.get("by_split") or {}
    if by_split:
        blocks.append(table(("Split", "Rows", "Patients", "Outcomes", "Outcome rate", "First index date", "Last index date"),
                            [(s, run.count(v.get("n_rows")), run.count(v.get("n_patients")), run.count(v.get("n_events")),
                              fmt_pct(v.get("prevalence")), fmt(v.get("index_date_min")), fmt(v.get("index_date_max")))
                             for s, v in by_split.items()]))
    return blocks


def _feature_definitions(run: _Run) -> list[Block]:
    fs = run.m.get("feature_set")
    if not fs:
        return _na()
    raw_cfg = run.config()
    feats = run.m.get("features") or {}
    blocks: list[Block] = [_kv([
        ("Feature specification", f"{fs.get('name')} {fs.get('version')}"), ("Specification SHA-256", fmt(fs.get("sha256"))),
        ("Number of predictors", fmt(fs.get("n_features"))), ("Pure eFalls feature set", fmt(fs.get("pure_efalls"))),
        ("Preprocessing representation", fmt(dig(raw_cfg, "preprocessing", "representation"))),
        ("Design-matrix columns", fmt(feats.get("n_design_columns"))), ("Non-zero coefficients", fmt(feats.get("n_selected")))]),
        Paragraph(f"Clinical definitions, time windows and missing-data rules are in configs/features/{fs.get('name')}.yaml; "
                  "decisions are documented in docs/EFALLS_REPRODUCTION_SPEC.md (§5, §9). Predictors use only records "
                  "dated before the index date (D-00).")]
    blocks += _coverage_section(run)
    if run.published:
        cov = run.ps.get("coverage") or {}
        blocks += [Heading("Published-predictor coverage (M-11)", 3), _kv([
            ("Effective experiment label", fmt(run.ps.get("effective_experiment_label"))),
            ("Share of development LP variance available", fmt(cov.get("lp_variance_share_available"))),
            ("Coverage threshold", fmt(cov.get("threshold"))),
            ("Mandatory predictors unavailable", ", ".join(cov.get("mandatory_unavailable") or []) or "none"),
            ("Unavailable-predictor fill", fmt(run.ps.get("unavailable_fill"))),
            ("Low-support predictors zeroed (D-20)", fmt(run.ps.get("low_support_zeroed")))])]
        unavailable, names = run.unavailable_predictors()
        if unavailable:
            blocks.append(table(("Unavailable predictor", "Published coefficient", "SAIL prevalence", "Expected mean LP shift"),
                                [(u.get("feature"), fmt(u.get("coefficient")), fmt(u.get("sail_prevalence")),
                                  fmt(u.get("expected_mean_lp_shift"))) for u in unavailable]))
        elif names:
            blocks.append(Paragraph(f"Published predictors declared unavailable: {', '.join(names)} (per-predictor coefficient table "
                                    "not recorded in metrics.json).", "warning"))
        else:
            blocks.append(Paragraph("No published predictor was declared unavailable."))
    return blocks


def _coverage_section(run: _Run) -> list[Block]:
    """eFalls predictor coverage table (section 3) for eFalls-labelled runs."""
    cov = run.coverage
    if cov is None:
        return [] if run.kind not in EFALLS_KINDS else [Heading("eFalls predictor coverage", 3), Paragraph(NOT_AVAILABLE, "note")]
    blocks: list[Block] = [Heading("eFalls predictor coverage", 3)]
    if run.reduced:
        blocks.append(Paragraph(f"{REDUCED_WARNING}: the model uses only the available eFalls predictors listed in the experiment "
                                "config (preprocessing.features); the unavailable predictors below were declared unavailable and not used.", "warning"))
    if cov.get("mandatory_unavailable"):
        blocks.append(Paragraph("WARNING: mandatory eFalls predictors unavailable (cohort.mandatory_for_efalls_label): "
                                f"{', '.join(map(str, cov['mandatory_unavailable']))}.", "warning"))
    blocks += [
        bullets([f"Available eFalls predictors: {_coverage_counts(cov)}", f"Coverage percentage: {_coverage_pct(cov)}",
                 f"Unavailable eFalls predictors: {_unavailable_text(cov)}"]),
        _kv([("Available eFalls predictors", _coverage_counts(cov)), ("Coverage percentage", _coverage_pct(cov)),
             ("Published-retained predictors available (non-zero published coefficient)",
              f"{fmt(cov.get('n_published_retained_available'))} / {fmt(cov.get('n_published_retained_total'))}"),
             ("Mandatory eFalls predictors unavailable", ", ".join(map(str, cov.get("mandatory_unavailable") or [])) or "none"),
             ("Full eFalls predictor set", fmt(cov.get("is_full_efalls_feature_set"))), ("Label", fmt(cov.get("label")))])]
    unavailable = list(cov.get("unavailable") or [])
    if unavailable:
        mandatory = set(cov.get("mandatory_unavailable") or [])
        blocks.append(table(("Unavailable eFalls predictor", "Mandatory for the eFalls label"),
                            [(name, "YES" if name in mandatory else "no") for name in unavailable]))
    return blocks


def _missingness(run: _Run) -> list[Block]:
    miss = run.m.get("missingness")
    if not miss:
        return _na()
    rows = sorted(miss.items(), key=lambda kv: -(kv[1].get("missing_rate") if kv[1].get("missing_rate") is not None else 2.0))
    splits = [s for s in ("train", "validation", "test") if any(s in (v.get("missing_rate_by_split") or {}) for _, v in rows)]
    handling = any(v.get("handling") for _, v in rows)
    nonzero = [(f, fmt_pct(v.get("missing_rate")), *[fmt_pct((v.get("missing_rate_by_split") or {}).get(s)) for s in splits],
                fmt(v.get("rule")), *([fmt(v.get("handling"))] if handling else []))
               for f, v in rows if v.get("missing_rate") != 0 or any((v.get("missing_rate_by_split") or {}).get(s) for s in splits)]
    blocks: list[Block] = [Paragraph(f"{len(rows) - len(nonzero)} of {len(rows)} predictors have no missing values. Binary "
                                     "predictors treat an absent code as 0 (published missing-data principle, spec §5.9)."
                                     + (" 'Handling' is what the fitted preprocessing actually did with missing values." if handling else ""))]
    if nonzero:
        blocks.append(table(("Predictor", "Missing (all rows)", *[f"Missing ({s})" for s in splits], "Missing-data rule",
                             *(["Handling in the fitted preprocessing"] if handling else [])), nonzero))
    return blocks


def _split(run: _Run) -> list[Block]:
    s = run.m.get("split")
    if not s:
        return _na()
    strategy = s.get("strategy")
    blocks: list[Block] = [_kv([("Strategy", fmt(strategy)), ("Description", fmt(s.get("description"))), ("Seed", fmt(s.get("seed"))),
                                ("Test rows SHA-256 (identical rows are required for comparisons)", fmt(s.get("test_rows_sha256"))),
                                ("Rows removed by the temporal embargo", fmt(s.get("embargo_removed"))),
                                ("Rows removed for patient overlap", fmt(s.get("patient_overlap_removed"))),
                                ("Limitation note", s.get("limitation_note") or "none recorded"),
                                ("Prior evaluations of these test rows (test-evaluation registry, D-19 §4)", _registry_text(run))])]
    prior = dig(run.m, "test_evaluation_registry", "prior_evaluations_same_test_rows")
    if prior:
        blocks.append(Paragraph(f"The locked test rows were evaluated {prior} time(s) before this run. Every additional look at the test "
                                "set weakens it as an independent check; interpret test results with this in mind (D-19 §4).", "warning"))
    if strategy != "temporal":
        return blocks + [Paragraph("Non-temporal split: results describe internal validity only and are not a temporal or external "
                                   "validation (D-19 §10).")]
    embargo = dig(run.config(), "validation", "temporal", "embargo_outcome_windows")
    text = "Temporal validation: development, validation and test cohorts are separated in time. eFalls itself performed no temporal validation."
    if embargo is True:
        return blocks + [Paragraph(text + " Outcome-window embargo on: a training outcome window must end before the next cohort's "
                                          "index date (D-19).")]
    return blocks + [Paragraph(text + (" WARNING: the outcome-window embargo (D-19) was DISABLED in the configuration; training "
                                       "outcomes may overlap later cohorts." if embargo is False else
                                       f" Outcome-window embargo setting: {NOT_AVAILABLE}."), "warning")]


def _registry_text(run: _Run) -> str:
    reg = run.m.get("test_evaluation_registry")
    if not reg:
        return NOT_AVAILABLE
    configs = reg.get("prior_config_sha256") or []
    return (f"{fmt(reg.get('prior_evaluations_same_test_rows'))} (purpose of this run: {fmt(reg.get('purpose'))}; prior config hashes: "
            f"{', '.join(map(str, configs)) or 'none'}; registry {fmt(reg.get('registry_file'))})")


def _model(run: _Run) -> list[Block]:
    m = run.m
    model = m.get("model")
    if not model:
        return _na()
    cal = m.get("calibration") or {}
    rows = [("Algorithm", MODEL_LABELS.get(model.get("name"), model.get("name"))), ("Registry name", fmt(model.get("name"))),
            ("Linear model", fmt(model.get("is_linear"))), ("Configured parameters", fmt(model.get("params"))),
            ("Effective parameters (configured plus adapter defaults)", fmt(model.get("effective_params"))),
            ("Parameters of resampling refits (stability, optimism, IECV)", fmt(model.get("resampling_params"))),
            ("Served prediction", _served_text(run)),
            ("Recalibration method", fmt(cal.get("method"))), ("Recalibration fitted on", fmt(cal.get("fitted_on"))),
            ("Recalibration parameters", _calibration_params(cal.get("params")))]
    blocks: list[Block] = [Paragraph(KIND_STATEMENTS[run.kind]), _kv(rows)]
    notes = model.get("fit_notes")
    blocks += [Heading("Fit notes (omissions are recorded, never silent; D-13)", 3),
               _kv([("Stata-style omissions in the final fit (constant columns, perfect predictors and the rows they predict)",
                     _omissions_text((notes or {}).get("stata_logit_omissions") if notes is not None else None)),
                    ("Omissions during fractional-polynomial selection",
                     _omissions_text((notes or {}).get("fp_selection_omissions") if notes is not None else None))])]
    lasso = m.get("lasso")
    if lasso:
        blocks += [Heading("LASSO fit (D-14)", 3), _kv([
            ("Intercept", fmt(lasso.get("intercept"), 6)), ("Selection rule", fmt(lasso.get("selection_rule"))),
            ("lambda* (selected)", _g(lasso.get("lambda_star"))), ("lambda_max", _g(lasso.get("lambda_max"))),
            ("Non-zero coefficients (n_selected)", fmt(lasso.get("n_selected"))), ("Converged", fmt(lasso.get("converged")))])]
    if run.published:
        blocks += [Heading("Sex parameterisations (D-01)", 3),
                   table(("Variant", "Role"), [(v, _variant_role(run, v)) for v in run.ps.get("sex_parameterisations") or []])]
    coef = run.csv("coefficients.csv")
    if coef is not None:
        if "relative_to_reference" in coef.columns:
            coef = coef.loc[~as_bool(coef["relative_to_reference"])]
        is_intercept = coef["design_column"].astype(str) == "_intercept"
        if is_intercept.any():
            blocks.append(_kv([("Model intercept (coefficients.csv)", fmt(pd.to_numeric(coef.loc[is_intercept, "coefficient"]).iloc[0], 6))]))
        omitted = as_bool(coef["omitted"]) if "omitted" in coef.columns else pd.Series(False, index=coef.index)
        if omitted.any():
            blocks.append(Paragraph("Omitted design columns (coefficient forced to 0, Stata-style): "
                                    + ", ".join(coef.loc[omitted, "design_column"].astype(str)), "warning"))
        coef = coef.loc[~is_intercept & ~omitted & (pd.to_numeric(coef["coefficient"], errors="coerce").fillna(0.0) != 0.0)]
        coef = coef.iloc[np.argsort(-pd.to_numeric(coef["coefficient"]).abs().to_numpy(), kind="stable")].head(25)
        unstandardised = "standardized_coefficient" not in coef.columns or pd.to_numeric(coef["standardized_coefficient"], errors="coerce").isna().any()
        blocks += [Heading("Largest non-zero coefficients", 3),
                   Paragraph(("Fixed published coefficients (S2 Table S3.2)." if run.published else
                              "Coefficients fitted locally on the training split." if run.kind == "efalls_retrained" else
                              "Coefficients fitted locally on the training split using a REDUCED eFalls predictor set." if run.reduced else
                              "Coefficients of the experimental algorithm fitted on the training split.")
                             + (" Ranked by |coefficient| on the original (unstandardised) scale: magnitudes are not comparable "
                                "across predictors with different units." if unstandardised else "")),
                   table(("Design column", "Feature", "Coefficient", "Odds ratio", "Standardised coefficient", "Direction"),
                         [(r.get("design_column"), r.get("feature"), fmt(r.get("coefficient")), fmt(r.get("odds_ratio")),
                           fmt(r.get("standardized_coefficient")), fmt(r.get("direction"))) for r in records(coef)])]
    refit = run.csv("unpenalized_refit.csv")
    if refit is not None:
        blocks += [Heading("Unpenalised refit of the LASSO-selected columns: DEMONSTRATION ONLY (spec §6.2)", 3),
                   Paragraph("An ordinary logistic regression refitted on the columns the LASSO selected, as eFalls presented, to show "
                             "odds ratios with Wald 95% CIs. It is a demonstration refit: predictions, metrics and the model bundle use "
                             "the penalised LASSO coefficients, and these CIs ignore the selection step.", "note"),
                   table(("Term", "Coefficient", "SE", "z", "p", "Odds ratio (95% CI)", "Status"),
                         [(r.get("term"), fmt(r.get("coefficient")), fmt(r.get("se")), fmt(r.get("z"), 2), fmt(r.get("p_value"), 4),
                           f"{fmt(r.get('odds_ratio'))} ({fmt(r.get('or_ci_low'))}–{fmt(r.get('or_ci_high'))})", fmt(r.get("status")))
                          for r in records(refit)])]
    fp = run.csv("fp_selection.csv")
    if fp is not None:
        blocks += [Heading("Fractional-polynomial selection (D-13)", 3),
                   table(("Variable", "Decision", "Selected powers", "Shift", "Scale", "Deviance linear / FP1 / FP2", "p non-linear",
                          "p FP2 vs FP1", "Cycles", "Converged", "Omissions"),
                         [(r.get("variable"), fmt(r.get("decision")), fmt(r.get("powers")), fmt(r.get("shift")), fmt(r.get("scale")),
                           " / ".join(fmt(r.get(k), 2) for k in ("deviance_linear", "deviance_fp1", "deviance_fp2")),
                           fmt(r.get("p_nonlinear"), 4), fmt(r.get("p_fp2_vs_fp1"), 4), fmt(r.get("cycles_run")), fmt(r.get("converged")),
                           r.get("omissions_json") if _has_content(_json_or_text(r.get("omissions_json"))) else "none")
                          for r in records(fp)])]
    return blocks + run.plot("coefficients.png", "Coefficients as odds ratios")


def _json_or_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _calibration_params(params: Any) -> str:
    """``{alpha, beta}`` or ``{variant: {alpha, beta}}`` as text with 3 dp."""
    if not isinstance(params, Mapping) or not params:
        return "none"
    flat = lambda p: ", ".join(f"{k} {fmt(v)}" for k, v in p.items())  # noqa: E731
    if all(isinstance(v, Mapping) for v in params.values()):
        return "; ".join(f"recalibration of {variant}: {flat(p)}" for variant, p in params.items())
    return flat(params)


def _hyperparameter_search(run: _Run) -> list[Block]:
    hs = run.m.get("hyperparameter_search") or {}
    if not hs.get("enabled"):
        reason = ("Tuning is not allowed for eFalls-labelled experiments (published equation or published learning process)."
                  if run.kind in EFALLS_KINDS else "Hyperparameter search was disabled in the configuration.")
        lasso = " The LASSO penalty is chosen by cross-validation (section 8)." if run.m.get("lasso") else ""
        return [Paragraph(f"No hyperparameter search was run. {reason}{lasso}")]
    blocks: list[Block] = [_kv([("Method", fmt(hs.get("method"))), ("Trials", fmt(hs.get("n_trials"))),
                                ("Objective weights (validation split)", fmt(hs.get("objective"))),
                                ("Selected parameters", _best_hyperparameters(run))])]
    res = run.csv("hyperparameter_results.csv")
    if res is None:
        return blocks + _na()
    top = res.sort_values("objective", ascending=False, kind="stable").head(10)
    blocks.append(table(("Trial", "Parameters", "Objective", "AUROC", "Brier", "Calibration slope", "Selected"),
                        [(fmt(r.get("trial")), fmt(r.get("params_json")), fmt(r.get("objective")), fmt(r.get("auroc")), fmt(r.get("brier")),
                          fmt(r.get("calibration_slope")), fmt(r.get("selected"))) for r in records(top)]))
    return blocks + run.plot("hyperparameter_search.png", "Hyperparameter search results")


def _cross_validation(run: _Run) -> list[Block]:
    blocks: list[Block] = []
    lasso = run.m.get("lasso")
    cv = run.csv("cv_results.csv")
    if lasso:
        blocks += [Heading("LASSO penalty selection (D-14)", 3), _kv([(k, fmt(v, 6) if isinstance(v, float) else fmt(v)) for k, v in lasso.items()])]
        if cv is not None:
            deviance = pd.to_numeric(cv["cv_mean_deviance"], errors="coerce")
            shown = [("selected lambda*", r) for r in records(cv.loc[as_bool(cv["selected"])])] if "selected" in cv.columns else []
            if deviance.notna().any():
                shown.append(("minimum mean deviance", records(cv.loc[[deviance.idxmin()]])[0]))
            blocks.append(table(("Row", "lambda", "CV mean deviance", "CV SE", "Non-zero coefficients"),
                                [(label, f"{r['lambda']:.6g}", fmt(r.get("cv_mean_deviance"), 4), fmt(r.get("cv_se"), 4), fmt(r.get("n_nonzero")))
                                 for label, r in shown]))
        blocks += run.plot("lasso_cv_path.png", "LASSO cross-validation path")
    internal = run.m.get("internal_validation") or {}
    for key, title in (("optimism", "Bootstrap optimism correction (D-11)"), ("iecv", "Internal–external cross-validation")):
        rows = internal.get(key)
        if rows:
            df = pd.DataFrame(rows)
            blocks += [Heading(title, 3), table(tuple(df.columns), [[fmt(v) for v in r.values()] for r in records(df)])]
    if not blocks and run.published:
        return [Paragraph("Not applicable: the published equation has fixed coefficients; nothing was cross-validated.")]
    return blocks or _na()


def _test_performance(run: _Run) -> list[Block]:
    test = dig(run.m, "performance", "test")
    if not test:
        return _na()
    blocks: list[Block] = []
    if run.published:
        ps = run.ps
        full = [v for v in (ps.get("primary"), ps.get("co_reported")) if v in test]
        shift = [v for v in ps.get("sex_parameterisations") or [] if v in test and v not in full]
        recal = [f"{v}+recalibrated" for v in full if f"{v}+recalibrated" in test]
        not_scored = [str(v) for v in (ps.get("primary"), ps.get("co_reported"), *(ps.get("sex_parameterisations") or []))
                      if v not in test]
        blocks.append(Paragraph(KIND_STATEMENTS[run.kind]))
        if not_scored:
            blocks.append(Paragraph(f"Sex parameterisations not scored on the test split: {', '.join(dict.fromkeys(not_scored))}. "
                                    "D-01 requires the primary and co-reported variants.", "warning"))
        blocks += [Heading("Primary and co-reported sex parameterisations (full metrics)", 3),
                   _metrics_table(run, "test", full), Heading("CITL-shift variants (O/E and CITL only)", 3),
                   Paragraph("lp_b_label_swap and lp_d2_numeric_swap are pure calibration-in-the-large shifts of lp_c_box_s3_1 "
                             "and lp_a_table_s3_2 (D-01): their discrimination and calibration slope equal those variants."),
                   table(("Variant", "O/E", "CITL"), [(v, fmt_ci(test[v].get("oe_ratio")), fmt_ci(test[v].get("citl"))) for v in shift])
                   if shift else Paragraph(NOT_AVAILABLE, "note")]
        blocks += _sex_stratified(run, [v for v in (ps.get("primary"), ps.get("co_reported")) if v])
        if recal:
            blocks += [Heading("Recalibrated on the validation split (reported separately; not the published equation)", 3),
                       _metrics_table(run, "test", recal)]
        if dig(run.m, "performance", "full_cohort"):
            blocks += [Heading("Full cohort (secondary)", 3), _metrics_table(run, "full_cohort", full)]
    else:
        blocks.append(_metrics_table(run, "test", list(test)))
    if dig(run.m, "performance", "validation"):
        blocks += [Heading("Validation split (used for recalibration and model selection only)", 3),
                   _metrics_table(run, "validation", [run.primary])]
    if dig(run.m, "performance", "train"):
        blocks += [Heading("Training split — apparent (in-sample) performance, optimistic by construction; not evidence of validity", 3),
                   _metrics_table(run, "train", [run.primary])]
    thresholds = (run.perf("test") or {}).get("thresholds") or []
    approved = dig(run.m, "production_readiness", "thresholds_approved") is True
    blocks += [Heading(f"Candidate thresholds (served variant {run.primary}; {'clinically approved' if approved else 'not clinically approved'}, M-12)", 3),
               table(("Threshold", "Sensitivity", "Specificity", "PPV", "NPV", "Net benefit"),
                     [(fmt(t.get("threshold"), 2), fmt(t.get("sensitivity")), fmt(t.get("specificity")), fmt(t.get("ppv")),
                       fmt(t.get("npv")), fmt(t.get("net_benefit"), 4)) for t in thresholds]) if thresholds else Paragraph(NOT_AVAILABLE, "note")]
    for name, caption in (("roc.png", "ROC curve"), ("precision_recall.png", "Precision–recall curve"),
                          ("risk_distribution.png", "Predicted-risk distribution"), ("decision_curve.png", "Decision curve")):
        blocks += run.plot(name, caption)
    return blocks


def _sex_stratified(run: _Run, variants: Sequence[str]) -> list[Block]:
    """Sex-stratified C, slope, CITL and O/E of the primary and co-reported variants on the test split (D-01 item 5)."""
    blocks: list[Block] = [Heading("Sex-stratified performance of the primary and co-reported variants (D-01)", 3),
                           Paragraph("Within each sex the parameterisations differ only by a constant, so the within-sex C-statistic "
                                     "and calibration slope are identical under all readings; CITL and O/E differ (D-01). Metrics are "
                                     "suppressed when a cell has fewer than 10 events.")]
    sub = run.csv("subgroup_metrics.csv")
    if sub is not None:
        sub = sub.loc[(sub["split"] == "test") & (sub["subgroup_variable"] == "sex") & sub["variant"].isin(variants)]
    if sub is None or sub.empty:
        return blocks + [Paragraph(NOT_AVAILABLE, "note")]
    order = {v: i for i, v in enumerate(variants)}
    sub = sub.assign(_o=sub["variant"].map(order)).sort_values(["_o", "subgroup_level"], kind="stable")
    return blocks + [table(("Variant", "Sex", "n", "Events", "AUROC", "Calibration slope", "CITL", "O/E"),
                           [(r.get("variant"), r.get("subgroup_level"), run.count(r.get("n")), run.count(r.get("n_events")),
                             fmt(r.get("auroc")), fmt(r.get("calibration_slope")), fmt(r.get("citl")), fmt(r.get("oe_ratio")))
                            for r in records(sub)])]


def _calibration_variants(run: _Run) -> list[str]:
    """Uncalibrated and recalibrated versions of the served prediction that were scored on the test split."""
    if run.published:
        base = str(run.m.get("transportability_variant") or run.served.split("+")[0])
        pair = [base, f"{base}+recalibrated"]
    else:
        pair = ["uncalibrated", "recalibrated"]
    test = dig(run.m, "performance", "test", default={})
    return [v for v in pair if v in test] or [run.primary]


def _variant_status(run: _Run, variant: str) -> str:
    if variant == run.served:
        return _status_text(run.m.get("calibration_status"))
    if "recalibrated" not in variant:
        return _status_text(run.m.get("calibration_status_uncalibrated"))
    return "not assessed separately (not served)"


def _calibration(run: _Run) -> list[Block]:
    e = run.perf("test")
    if not e:
        return _na()
    groups = dig(run.config(), "evaluation", "calibration_groups")
    variants = _calibration_variants(run)
    entries = [run.perf("test", v) or {} for v in variants]
    measure = lambda label, get: (label, *[get(x) for x in entries])  # noqa: E731
    cal = run.m.get("calibration") or {}
    blocks: list[Block] = [
        Paragraph("Calibration compares predicted with observed risk. Ideal: slope 1, calibration-in-the-large (CITL) 0, "
                  f"observed/expected (O/E) 1. Curves use {groups if groups is not None else 'equal-size'} risk groups and LOWESS (D-17)."
                  + (" Both the uncalibrated and the recalibrated predictions are shown; the recalibration was fitted on the validation "
                     "split only." if len(variants) > 1 else "")),
        table(("Measure (test split)", *[f"{v} ({'served' if v == run.served else 'not served'})" for v in variants]), [
            ("Calibration status", *[_variant_status(run, v) for v in variants]),
            measure("Calibration slope", lambda x: fmt_ci(x.get("calibration_slope"))),
            measure("Calibration intercept", lambda x: fmt_ci(x.get("calibration_intercept"))),
            measure("CITL", lambda x: fmt_ci(x.get("citl"))), measure("O/E ratio", lambda x: fmt_ci(x.get("oe_ratio"))),
            measure("Mean predicted risk", lambda x: fmt(x.get("mean_predicted"))), measure("Observed rate", lambda x: fmt(x.get("observed_rate"))),
            measure("ICI / E50 / E90", lambda x: " / ".join(fmt(dig(x, "calibration_indices", k)) for k in ("ici", "e50", "e90")))]),
        _kv([("Recalibration parameters", _calibration_params(cal.get("params"))), ("Recalibration fitted on", fmt(cal.get("fitted_on")))])]
    if run.published:
        sc = run.ps.get("sex_specific_citl_lp_c")
        citl = fmt_ci({"estimate": sc.get("male_minus_female"), "ci_low": sc.get("ci_low"), "ci_high": sc.get("ci_high")}) if sc else "not available"
        blocks += [Heading("Sex-specific calibration (descriptive only)", 3),
                   Paragraph(f"Sex-specific CITL of LP_C (male minus female): {citl}. DESCRIPTIVE ONLY: in a new population it mixes "
                             "true sex-specific miscalibration and does not support any sex parameterisation (D-01).")]
        oe = run.ps.get("sex_specific_oe") or {}
        if oe:
            blocks.append(table(("Variant", "O/E female", "O/E male"), [(v, fmt(d.get("female")), fmt(d.get("male"))) for v, d in oe.items()]))
    blocks += run.plot("calibration.png", f"Calibration plot (served variant {run.served})")
    if len(variants) > 1 or (run.path / "plots" / "calibration_before_after.png").is_file():
        blocks += run.plot("calibration_before_after.png", "Calibration before and after recalibration (grouped points and LOWESS)")
    return blocks


def _feature_importance(run: _Run) -> list[Block]:
    fi = run.csv("feature_importance.csv")
    if fi is None:
        return _na() + run.plot("feature_importance.png", "Feature importance")
    top = fi.sort_values("permutation_importance_mean", ascending=False, kind="stable").head(15)
    return [Paragraph("Permutation importance: the drop in validation AUROC when a feature's values are shuffled. Larger means the "
                      "model relies more on the feature; it does not imply causation."),
            table(("Feature", "Rank", "Permutation importance (mean ± SD)", "Coefficient", "Odds ratio", "Selection frequency"),
                  [(r.get("feature"), fmt(r.get("importance_rank")),
                    f"{fmt(r.get('permutation_importance_mean'), 4)} ± {fmt(r.get('permutation_importance_std'), 4)}",
                    fmt(r.get("coefficient")), fmt(r.get("odds_ratio")), fmt(r.get("selection_frequency"))) for r in records(top)])] \
        + run.plot("feature_importance.png", "Feature importance")


def _feature_stability(run: _Run) -> list[Block]:
    fs = run.csv("feature_stability.csv")
    if fs is None:
        if run.published:
            return [Paragraph("Not applicable: the published equation has fixed coefficients and no fitting process to resample.")]
        return _na()
    robust = as_bool(fs["robust"]) if "robust" in fs.columns else pd.Series(False, index=fs.index)
    top = fs.assign(robust=robust).sort_values("selection_frequency", ascending=False, kind="stable").head(30)
    blocks: list[Block] = [
        Paragraph(f"Bootstrap refits on the training split ({fmt(fs['n_bootstrap'].iloc[0]) if 'n_bootstrap' in fs else 'not available'} "
                  f"replicates): {int(robust.sum())} of {len(fs)} design columns are robust (frequently selected with a stable sign)."),
        table(("Design column", "Raw feature", "Selection frequency", "Sign stability", "Coefficient mean", "Coefficient SD", "Robust"),
              [(r.get("feature"), r.get("raw_feature"), fmt(r.get("selection_frequency")), fmt(r.get("sign_stability")),
                fmt(r.get("coef_mean")), fmt(r.get("coef_sd")), fmt(r.get("robust"))) for r in records(top)])]
    inst = run.csv("instability.csv")
    if inst is not None:
        blocks += [Heading("Prediction instability", 3),
                   table(("Threshold", "Mean classification instability", "Max classification instability", "MAPE mean", "MAPE median"),
                         [(fmt(r.get("threshold"), 2), fmt(r.get("mean_classification_instability")),
                           fmt(r.get("max_classification_instability")), fmt(r.get("mape_mean"), 4), fmt(r.get("mape_median"), 4))
                          for r in records(inst)])]
    return blocks + run.plot("selection_stability.png", "Feature-selection stability")


def _error_analysis(run: _Run) -> list[Block]:
    e = run.perf("test")
    if not e:
        return _na()
    blocks: list[Block] = [Paragraph("Errors at candidate thresholds: false negatives are missed falls/fractures; false positives are "
                                     "patients flagged who did not fall. Thresholds are candidates only; no risk category is exposed "
                                     "until clinically approved (M-12).")]
    if not run.synthetic:
        blocks.append(Paragraph("Counts 1–9 are shown as <10 (M-13). Check complementary disclosure before exporting.", "note"))
    rows = []
    for t in e.get("thresholds") or []:
        events = (t.get("tp") or 0) + (t.get("fn") or 0)
        rows.append((fmt(t.get("threshold"), 2), run.count(t.get("tp")), run.count(t.get("fp")), run.count(t.get("tn")), run.count(t.get("fn")),
                     fmt_pct(t["fn"] / events) if events else "not available",
                     fmt(t["fp"] / t["tp"], 2) if t.get("tp") else "not available"))
    blocks.append(table(("Threshold", "TP", "FP", "TN", "FN", "Share of outcomes missed", "False alarms per true positive"), rows)
                  if rows else Paragraph(NOT_AVAILABLE, "note"))
    sub = run.csv("subgroup_metrics.csv")
    if sub is not None:
        sub = sub.loc[(sub["split"] == "test") & (sub["variant"] == run.primary)]
    if sub is None or sub.empty:
        blocks += [Heading("Subgroups", 3), Paragraph(NOT_AVAILABLE, "note")]
    else:
        blocks += [Heading("Subgroups (metrics suppressed when fewer than 10 events)", 3),
                   table(("Variable", "Level", "n", "Events", "AUROC", "Brier", "Calibration slope", "CITL", "O/E"),
                         [(r.get("subgroup_variable"), r.get("subgroup_level"), run.count(r.get("n")), run.count(r.get("n_events")),
                           fmt(r.get("auroc")), fmt(r.get("brier")), fmt(r.get("calibration_slope")), fmt(r.get("citl")),
                           fmt(r.get("oe_ratio"))) for r in records(sub)])]
    for name, title in (("heterogeneity_clusters.csv", "Heterogeneity: test metrics per cluster (D-16)"),
                        ("heterogeneity_pooled.csv", "Heterogeneity: random-effects pooled summary (D-16)")):
        het = run.csv(name)
        if het is not None:
            blocks += [Heading(title, 3), table(tuple(het.columns), [[run.count(v) if c in COUNT_COLUMNS else fmt(v) for c, v in r.items()]
                                                                     for r in records(het)])]
    return blocks


def _summary_row(label: str, entry: Mapping[str, Any] | None) -> tuple[str, ...]:
    e = entry or {}
    return (label, fmt_ci(e.get("auroc")), fmt_ci(e.get("brier")), fmt_ci(e.get("calibration_slope")), fmt_ci(e.get("citl")), fmt_ci(e.get("oe_ratio")))


SUMMARY_COLUMNS = ("Model", "AUROC", "Brier", "Calibration slope", "CITL", "O/E")


def _comparison_to_efalls(run: _Run) -> list[Block]:
    blocks: list[Block] = [Paragraph(STANDARD_LIMITATIONS[0] + " Published eFalls performance figures (spec §7.7) are context only, "
                                     "not targets for this data.")]
    if run.published:
        rows = [_summary_row(f"{v} ({_variant_role(run, v)})", run.perf("test", v)) for v in (run.ps.get("primary"), run.ps.get("co_reported"))]
        return blocks + [Paragraph("This run IS the published eFalls scoring (transportability of the original equation)."),
                         table(SUMMARY_COLUMNS, rows)]
    comp = run.m.get("comparison_to_efalls") or {}
    blocks.append(_kv([("Reference published-eFalls run", fmt(comp.get("reference_run_id"))),
                       ("Reference variant", fmt(comp.get("reference_variant"))),
                       ("AUROC difference on identical test rows (this run's served variant minus published)", fmt(comp.get("delta_auroc_test"))),
                       ("Uncertainty", PAIRED_CI_NOTE), ("Note", fmt(comp.get("note")))]))
    ref_path = run.path.parent / str(comp.get("reference_run_id")) / "metrics.json"
    if comp.get("reference_run_id") and ref_path.is_file():
        ref = _Run(ref_path.parent, read_metrics(ref_path.parent))
        reference = [v for v in (ref.ps.get("primary"), ref.ps.get("co_reported")) if v] or [ref.primary]
        blocks.append(table(SUMMARY_COLUMNS, [_summary_row(f"This run ({dig(run.m, 'model', 'name')}, {run.primary})", run.perf("test")),
                                              *[_summary_row(f"Published eFalls ({v}; {_variant_role(ref, v)})", ref.perf("test", v))
                                                for v in reference]]))
    return blocks


def _comparison_to_alternatives(run: _Run) -> list[Block]:
    from falls_ml.reporting.compare import load_run_metrics  # local import: compare renders with this module

    note = ("This run is an experimental algorithm. " if run.kind in {"alternative_model", "ablation_member"} else "") + \
        ("All completed runs are compared on identical test rows in the master comparison (reports/comparison.md), which "
         "requires discrimination, calibration and net-benefit criteria, prefers the operationally simplest qualifying model, "
         "and reports stability and interpretability descriptively.")
    try:
        runs, _ = load_run_metrics(run.path.parent)
    except ArtifactError as exc:
        return [Paragraph(note), Paragraph(f"Other runs could not be read: {exc}", "warning")]
    sha, data = dig(run.m, "split", "test_rows_sha256"), dig(run.m, "dataset", "data_sha256")
    rows = []
    for path, other in runs:
        if path.resolve() == run.path.resolve() or dig(other, "split", "test_rows_sha256") != sha \
                or dig(other, "dataset", "data_sha256") != data or bool(other.get("synthetic_fixture")) != run.synthetic:
            continue
        o = _Run(path, other)
        rows.append(_summary_row(f"{dig(other, 'experiment', 'name')} [{KIND_TITLES[o.kind].split(':')[0]}; {o.primary}]", o.perf("test")))
    blocks: list[Block] = [Paragraph(note)]
    if not rows:
        return blocks + [Paragraph(f"Other runs on identical test rows: {NOT_AVAILABLE}", "note")]
    return blocks + [table(SUMMARY_COLUMNS, [_summary_row(f"THIS RUN: {dig(run.m, 'experiment', 'name')} [{run.primary}]", run.perf("test")), *rows])]


def _limitations(run: _Run) -> list[Block]:
    m = run.m
    pr = m.get("production_readiness") or {}
    validated = pr.get("mappings_clinically_validated")
    unavailable = run.unavailable_predictors()[1]
    reduced_unavailable = list((run.coverage or {}).get("unavailable") or []) if not run.published else []
    declared = pr.get("unavailable_predictors")
    declared_text = ", ".join(map(str, declared)) if isinstance(declared, list) else f"{declared} predictor(s)"
    split = m.get("split") or {}
    standard = [
        *STANDARD_LIMITATIONS,
        "Clinical code mappings: " + ("recorded as clinically validated for this dataset." if validated else
                                      "NOT clinically validated (spec §11, B-02); results depend on unvalidated mappings."
                                      if validated is False else "validation status not available."),
        "Unavailable predictors: " + (", ".join(unavailable) + " (handled per M-11)." if unavailable else
                                      ", ".join(reduced_unavailable) + " (declared unavailable and excluded from the model: "
                                      f"{REDUCED_WARNING})." if reduced_unavailable else
                                      f"{declared_text} declared unavailable." if declared else
                                      "none declared for this run." if "unavailable_predictors" in pr or run.published else "not available."),
        f"Split: {split.get('strategy', 'not available')}. " + (split.get("limitation_note") or
                                                                ("A single temporal test cohort; results may not hold for other periods or sites."
                                                                 if split.get("strategy") == "temporal" else
                                                                 "Non-temporal partitions estimate internal validity only (D-19 §10).")),
    ]
    blocks: list[Block] = [Heading("Standard limitations (always reported)", 3), bullets(standard),
                           Heading("Run-specific limitations", 3)]
    blocks.append(bullets(m["limitations"]) if m.get("limitations") else Paragraph(NOT_AVAILABLE, "note"))
    if m.get("warnings"):
        blocks += [Heading("Pipeline warnings", 3), bullets(m["warnings"])]
    return blocks


def _production_readiness(run: _Run) -> list[Block]:
    pr = run.m.get("production_readiness")
    elig = dig(run.m, "eligibility", "eligible_for_further_validation")
    if pr is None:
        return [Paragraph(f"Production-readiness checks: {NOT_AVAILABLE}"),
                _kv([("Eligible for further validation", fmt(elig))])]
    checks = {"Model bundle saved": pr.get("bundle_saved"), "Decision thresholds clinically approved (M-12)": pr.get("thresholds_approved"),
              "Clinical mappings validated": pr.get("mappings_clinically_validated"),
              "No unavailable predictors": None if pr.get("unavailable_predictors") is None else not pr.get("unavailable_predictors"),
              "Eligible for further validation": elig, "Real (non-synthetic) data": not run.synthetic}
    ready = all(v is True for v in checks.values())
    return [Paragraph("Production readiness: " + ("all recorded checks pass; deployment still requires governance sign-off." if ready else
                                                  "NOT READY for production use; see the failed or unknown checks below."),
                      "normal" if ready else "warning"),
            _kv([(k, "not available" if v is None else "pass" if v else "FAIL") for k, v in checks.items()] +
                [("Model bundle location", fmt(dig(run.m, "artifacts", "model_bundle")))])]


_SECTION_BUILDERS = (_dataset, _cohort, _feature_definitions, _missingness, _split, _model, _hyperparameter_search, _cross_validation,
                     _test_performance, _calibration, _feature_importance, _feature_stability, _error_analysis,
                     _comparison_to_efalls, _comparison_to_alternatives, _limitations, _production_readiness)


def build_run_document(run_dir: Path) -> list[Block]:
    """Executive summary followed by the 17 technical sections in fixed order."""
    run = _Run(Path(run_dir), read_metrics(Path(run_dir)))
    blocks = _header(run) + _executive_summary(run)
    for title, builder in zip(SECTION_TITLES, _SECTION_BUILDERS, strict=True):
        blocks.append(Heading(title))
        blocks.extend(builder(run))
    return blocks


def render_run_report(run_dir: str | Path) -> tuple[Path, Path]:
    """Render ``report.md`` and ``report.html`` into ``run_dir``; raises ``FileNotFoundError`` without metrics.json."""
    run_dir = Path(run_dir)
    blocks = build_run_document(run_dir)
    title = next(b.text for b in blocks if isinstance(b, Heading))
    md_path, html_path = run_dir / "report.md", run_dir / "report.html"
    write_document(blocks, md_path, html_path, title=title)
    log.info("run_report_rendered", extra_fields={"run_dir": str(run_dir), "n_blocks": len(blocks)})
    return md_path, html_path
