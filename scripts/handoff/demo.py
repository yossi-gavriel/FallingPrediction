"""Synthetic demonstrations of falls_ml (run with the venv python). Nothing is written to runs/ or reports/.

Usage: <venv python> scripts/handoff/demo.py baseline|full [--out demo_outputs/<name>]

baseline: validate data/fixtures/synthetic_v1; train efalls_published_scoring and efalls_retrained_lasso (the real
          fixture configs); reports, metrics and model bundles; reload the bundles; score a CSV with the CLI; compare
          the two runs; write DEMO_SUMMARY.md.
full:     validate all three fixtures; train all 7 fixture algorithms plus efalls_retrained_reduced (on
          data/fixtures/synthetic_reduced_v1); Meuhedet-enhanced ablation (synthetic_enhanced_v1); comparison;
          CLI scoring; drift monitoring on unchanged and shifted rows; exact reproduction of one run; DEMO_SUMMARY.md.

Every output folder carries ".falls_ml_generated" and SYNTHETIC_DATA_NOT_SCIENTIFIC_RESULTS.txt. An existing output
folder is replaced only if it carries the marker. Uses only public commands (python -m falls_ml ...) and the public API.
Exit 0 = DEMO SUCCESSFUL, 1 = DEMO FAILED.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import smoke  # noqa: E402

ROOT = common.PROJECT_ROOT
FIXTURE_CONFIGS = "configs/experiments/fixture"
SYNTHETIC = "data/fixtures/synthetic_v1"
ENHANCED = "data/fixtures/synthetic_enhanced_v1"
REDUCED = "data/fixtures/synthetic_reduced_v1"
REDUCED_CONFIG = f"{FIXTURE_CONFIGS}/efalls_retrained_reduced.yaml"
ABLATION_CONFIG = f"{FIXTURE_CONFIGS}/ablation_example.yaml"
EXTENSION_SPEC = "configs/features/meuhedet_enhanced_example.yaml"
BASELINE_EXPERIMENTS = ("efalls_published_scoring", "efalls_retrained_lasso")
FULL_EXPERIMENTS = ("efalls_published_scoring", "efalls_retrained_lasso", "efalls_retrained_lasso_fixed_fp", "logistic_unpenalized",
                    "elastic_net_logistic", "random_forest", "hist_gradient_boosting")
COMPARE_BOOTSTRAP = 200
SCORING_MODEL = "efalls_retrained_lasso"
REPRODUCE_EXPERIMENT = "efalls_retrained_lasso_fixed_fp"


def rel(path: Path | str, base: Path = ROOT) -> str:
    """Forward-slash path relative to ``base`` (for Markdown and portable logs)."""
    return common.posix_path(path, base)


class Demo:
    def __init__(self, kind: str, out: Path):
        self.kind, self.out = kind, out
        self.logs = out / "logs"
        self.commands: list[str] = []
        self.runs: dict[str, Path] = {}
        self.notes: dict[str, Any] = {}
        self.runner = common.StepRunner(total=0)
        self._synthetic: tuple[Any, Any] | None = None

    # ------------------------------------------------------------------ helpers
    def folder(self, name: str, purpose: str) -> Path:
        return common.mark_generated(self.out / name, purpose)

    def cli(self, number: int, args: list[Any], *, expect_json: bool = True, ok_codes: tuple[int, ...] = (0,)) -> Any:
        """Run `python -m falls_ml <args>` from the project folder; stdout is returned (parsed JSON by default)."""
        args = [common.display_path(a) if isinstance(a, Path) else str(a) for a in args]
        display = "python -m falls_ml " + " ".join(f'"{a}"' if " " in a else a for a in args)
        self.commands.append(display)
        log_path = self.logs / f"{number:02d}_{args[0]}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {display}", flush=True)
        start = time.monotonic()
        proc = subprocess.run([sys.executable, "-m", "falls_ml", *args], cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", env=common.subprocess_env())
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"$ {display}\n# exit code {proc.returncode} after {time.monotonic() - start:.1f} s\n# --- stdout\n{proc.stdout}\n"
                      f"# --- stderr\n{proc.stderr}\n")
        if proc.returncode not in ok_codes:
            lines = (proc.stderr + "\n" + proc.stdout).splitlines()
            last = next((ln for ln in reversed(lines) if ln.strip()), "no output")
            raise common.StepFailure(f"`{display}` failed with exit code {proc.returncode}: {last.strip()[:300]}",
                                     f"Read {log_path.resolve()}; run {common.script_command('verify')} to check the installation.",
                                     log_path, common.tail(lines))
        if not expect_json:
            return proc.stdout
        try:
            return json.loads(proc.stdout)
        except ValueError as exc:
            raise common.StepFailure(f"`{display}` did not print JSON ({exc})", "Report this to the maintainer.", log_path) from exc

    def train(self, number: int, name: str, dataset: str | None = None) -> str:
        args: list[Any] = ["train", "--config", f"{FIXTURE_CONFIGS}/{name}.yaml", "--runs-dir", self.out / "runs"]
        if dataset:
            args += ["--dataset", dataset]
        result = self.cli(number, args)
        run_dir = Path(result["run_dir"])
        run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
        self.runs[name] = run_dir
        metrics = self.metrics(name)
        head = headline(metrics)
        return f"{name}: run {run_dir.name}; synthetic test AUROC {fmt(head['auroc'])} ({head['variant']})"

    def metrics(self, name: str) -> dict[str, Any]:
        return json.loads((self.runs[name] / "metrics.json").read_text(encoding="utf-8"))

    def scoring_rows(self, run_name: str) -> tuple[Any, Any]:
        """(synthetic_v1 test rows of a run, its served test risks)."""
        from falls_ml.data.dataset import ModelingDataset
        from falls_ml.features.spec import load_feature_spec

        if self._synthetic is None:
            spec = load_feature_spec(ROOT / "configs" / "features" / "efalls_v1.yaml")
            self._synthetic = (ModelingDataset.load(ROOT / SYNTHETIC, spec).frame, spec)
        frame, spec = self._synthetic
        return smoke.test_rows(self.runs[run_name], frame, spec)

    # ------------------------------------------------------------------ shared steps
    def step_validate(self, number: int, fixtures: list[tuple[str, list[str]]]) -> str:
        found = []
        for fixture, extra in fixtures:
            result = self.cli(number, ["validate-dataset", "--dataset", fixture, *extra])
            found.append(f"{Path(fixture).name} ({result.get('n_rows')} rows)")
        self.notes["fixtures"] = found
        return "valid: " + ", ".join(found)

    def step_reload_bundles(self, names: tuple[str, ...]) -> str:
        from falls_ml.bundle import load_bundle
        from falls_ml.inference import predict_risk

        worst = 0.0
        reloaded = []
        for name in names:
            bundle = load_bundle(self.runs[name] / "model")
            rows, served = self.scoring_rows(name)
            diff = smoke.max_abs_diff([r.risk_12m for r in predict_risk(rows, bundle)], served)
            if diff > common.PREDICTION_TOLERANCE:
                raise common.StepFailure(f"{name}: reloaded bundle predictions differ from the run (max abs diff {diff:.2e})",
                                         f"Run {common.script_command('verify')}.", self.runs[name] / "model")
            worst = max(worst, diff)
            reloaded.append(f"{name} ({bundle.served_variant})")
        self.notes["bundle_reload_max_abs_diff"] = worst
        return f"{len(names)} bundles reloaded and verified ({', '.join(reloaded)}); max abs diff {worst:.1e}"

    def step_predict(self, number: int, names: tuple[str, ...]) -> str:
        import pandas as pd

        folder = self.folder("predictions", "CLI predictions")
        rows, _ = self.scoring_rows(names[0])
        csv_in = folder / "scoring_input_synthetic.csv"
        rows.to_csv(csv_in, index=False)
        outputs = []
        for name in names:
            csv_out = folder / f"{name}_predictions.csv"
            self.cli(number, ["predict", "--model", self.runs[name] / "model", "--input", csv_in, "--out", csv_out])
            _, served = self.scoring_rows(name)
            predicted = pd.read_csv(csv_out, dtype={"research_id": str})
            diff = smoke.max_abs_diff(predicted["risk_12m"].to_numpy(), served)
            if diff > common.PREDICTION_TOLERANCE:
                raise common.StepFailure(f"{name}: CLI predictions differ from the run (max abs diff {diff:.2e})",
                                         f"Run {common.script_command('verify')}.", csv_out)
            outputs.append({"model": name, "file": rel(csv_out, self.out), "rows": int(len(predicted)), "max_abs_diff": diff})
        self.notes["predictions"] = outputs
        self.notes["scoring_input"] = rel(csv_in, self.out)
        return f"{len(outputs)} prediction files ({len(rows)} synthetic rows each) identical to the in-run test predictions"

    def step_compare(self, number: int, ablation_dir: Path | None) -> str:
        folder = self.folder("comparison", "comparison reports")
        args: list[Any] = ["compare", "--runs-dir", self.out / "runs", "--out", folder, "--n-bootstrap", COMPARE_BOOTSTRAP]
        if ablation_dir is not None:
            args += ["--ablation-dir", ablation_dir]
        self.cli(number, args, expect_json=False)
        files = sorted(rel(p, self.out) for p in folder.rglob("*") if p.is_file() and p.name not in (common.MARKER_FILE, common.SYNTHETIC_NOTICE_FILE))
        if not files:
            raise common.StepFailure("the comparison wrote no files", "Read the compare log.", self.logs)
        self.notes["comparison_files"] = files
        return f"comparison of {len(self.runs)} runs written ({len(files)} files, paired bootstrap n {COMPARE_BOOTSTRAP})"

    # ------------------------------------------------------------------ plans
    def plan(self) -> list[tuple[str, Any]]:
        if self.kind == "baseline":
            return [
                ("Validating the synthetic fixture", lambda n: self.step_validate(n, [(SYNTHETIC, [])])),
                ("Training efalls_published_scoring", lambda n: self.train(n, "efalls_published_scoring")),
                ("Training efalls_retrained_lasso", lambda n: self.train(n, "efalls_retrained_lasso")),
                ("Reloading model bundles", lambda n: self.step_reload_bundles(BASELINE_EXPERIMENTS)),
                ("Scoring a CSV with the command-line interface", lambda n: self.step_predict(n, BASELINE_EXPERIMENTS)),
                ("Comparing the two experiments", lambda n: self.step_compare(n, None)),
            ]
        steps: list[tuple[str, Any]] = [
            ("Checking full-demo inputs", lambda n: self.step_full_inputs()),
            ("Validating the synthetic fixtures", lambda n: self.step_validate(n, [
                (SYNTHETIC, []), (ENHANCED, ["--extension", EXTENSION_SPEC]), (REDUCED, ["--features-from-config", REDUCED_CONFIG])])),
        ]
        steps += [(f"Training {name}", lambda n, name=name: self.train(n, name)) for name in FULL_EXPERIMENTS]
        steps += [
            ("Training efalls_retrained_reduced (REDUCED eFalls predictor set)", lambda n: self.train(n, "efalls_retrained_reduced", REDUCED)),
            ("Running the Meuhedet-enhanced ablation (illustrative groups)", self.step_ablation),
            ("Comparing all experiments", lambda n: self.step_compare(n, self.out / "ablation")),
            ("Scoring a CSV with the command-line interface", lambda n: self.step_predict(n, (SCORING_MODEL,))),
            ("Monitoring drift on unchanged and shifted rows", self.step_monitor),
            (f"Reproducing {REPRODUCE_EXPERIMENT} from its artifacts", self.step_reproduce),
        ]
        return steps

    def step_full_inputs(self) -> str:
        needed = [f"{FIXTURE_CONFIGS}/{n}.yaml" for n in FULL_EXPERIMENTS] + [REDUCED_CONFIG, ABLATION_CONFIG, EXTENSION_SPEC]
        needed += [f"{d}/manifest.json" for d in (SYNTHETIC, ENHANCED, REDUCED)]
        missing = [p for p in needed if not (ROOT / p).is_file()]
        if missing:
            raise common.StepFailure(f"the full demo needs files that are missing: {', '.join(missing)}",
                                     "Copy the complete package again (the reduced eFalls config and the synthetic_reduced_v1 fixture are "
                                     "part of version 0.2.0); if you built the package yourself, rebuild it with tools/build_handoff.py.")
        return f"{len(needed)} configs, specs and fixtures present"

    def step_ablation(self, number: int) -> str:
        folder = self.folder("ablation", "ablation summary")
        self.folder("runs_ablation", "ablation runs")
        self.cli(number, ["ablation", "--config", ABLATION_CONFIG, "--dataset", ENHANCED, "--runs-dir", self.out / "runs_ablation",
                          "--out", folder], expect_json=False)
        files = sorted(rel(p, self.out) for p in folder.iterdir() if p.is_file() and p.name not in (common.MARKER_FILE, common.SYNTHETIC_NOTICE_FILE))
        self.notes["ablation_files"] = files
        return f"ablation summary written ({', '.join(files)})"

    def step_monitor(self, number: int) -> str:
        folder = self.folder("monitoring", "drift monitoring")
        rows, _ = self.scoring_rows(SCORING_MODEL)
        shifted = rows.copy()
        shifted["polypharmacy_count_120d"] = shifted["polypharmacy_count_120d"] + 15
        shifted["age_years"] = (shifted["age_years"] + 5).clip(upper=120)
        statuses = {}
        for label, frame in (("unchanged", rows), ("shifted", shifted)):
            csv_in = folder / f"{label}_rows_synthetic.csv"
            frame.to_csv(csv_in, index=False)
            result = self.cli(number, ["monitor", "--model", self.runs[SCORING_MODEL] / "model", "--input", csv_in, "--out", folder / label])
            statuses[label] = result["overall_status"]
        self.notes["monitoring"] = {k: {"status": v, "folder": f"monitoring/{k}"} for k, v in statuses.items()}
        if statuses["shifted"] == "ok":
            raise common.StepFailure("monitoring did not flag deliberately shifted rows (status ok)", "Report this to the maintainer.", folder)
        return f"drift status: unchanged rows = {statuses['unchanged']}, shifted rows = {statuses['shifted']}"

    def step_reproduce(self, number: int) -> str:
        self.folder("runs_reproduced", "reproduced runs")
        report = self.cli(number, ["reproduce", "--run-dir", self.runs[REPRODUCE_EXPERIMENT], "--runs-dir", self.out / "runs_reproduced"],
                          ok_codes=(0, 2))
        if not report.get("identical"):
            raise common.StepFailure(f"reproduction of {REPRODUCE_EXPERIMENT} is not identical ({len(report.get('differences', []))} differences)",
                                     "Send the reproduce log to the maintainer.", self.logs)
        self.notes["reproduction"] = {"original": report["original_run_id"], "reproduced": report["reproduced_run_id"], "identical": True}
        return f"reproduced run {report['reproduced_run_id']} is identical to {report['original_run_id']}"

    # ------------------------------------------------------------------ run and summary
    def run(self) -> int:
        plan = self.plan()
        self.runner = common.StepRunner(total=len(plan))
        start = time.monotonic()
        common.print_banner()
        try:
            self.folder("runs", "experiment runs")
            self.folder("logs", "demo logs")
            for number, (title, func) in enumerate(plan, start=1):
                self.runner.run(number, title, lambda f=func, n=number: f(n))
        except common.StepFailure as failure:
            common.print_lines(common.failure_block("DEMO FAILED", failure.step, failure.reason, failure.action, failure.log_path))
            common.print_banner()
            return 1
        summary = self.write_summary(time.monotonic() - start)
        common.print_lines(common.success_block("DEMO SUCCESSFUL"))
        print(f"Outputs:  {self.out.resolve()}")
        print(f"Summary:  {summary.resolve()}")
        for name, run_dir in self.runs.items():
            print(f"Report:   {(run_dir / 'report.html').resolve()}  ({name})")
        print(f"Elapsed:  {common.format_seconds(time.monotonic() - start)}")
        common.print_banner()
        return 0

    def write_summary(self, seconds: float) -> Path:
        import falls_ml

        out = self.out
        lines = [f"# falls_ml {self.kind} demo — {common.SYNTHETIC_BANNER}", "",
                 f"> **{common.SYNTHETIC_BANNER}.** Every file listed here was produced from deterministic synthetic fixtures to show that "
                 "the software works end to end. No real patient data was used. The metrics are software-test output, not evidence "
                 "about fall risk, eFalls or Meuhedet, and must not be reported as results.", "",
                 f"- Generated (UTC): {common.utc_now()}", f"- falls_ml {falls_ml.__version__}, Python {platform.python_version()}, "
                 f"{platform.system()} {platform.machine()}", f"- Elapsed: {common.format_seconds(seconds)}",
                 f"- Fixtures validated: {', '.join(self.notes.get('fixtures', []))}", "",
                 "## Experiment runs (headline metrics on SYNTHETIC test rows)", "",
                 "| Experiment | Kind | Served variant | AUROC (synthetic) | Calibration slope | CITL | Brier | eFalls predictors | Report | Model bundle |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for name, run_dir in self.runs.items():
            m = self.metrics(name)
            h = headline(m)
            coverage = m.get("efalls_coverage") or {}
            cov = f"{coverage['n_available']} / {coverage['n_total']}" if coverage else "n/a"
            if coverage and not coverage.get("is_full_efalls_feature_set", True):
                cov += " (REDUCED – not a full eFalls reproduction)"
            lines.append(f"| {name} | {m['experiment']['kind']} | {h['variant']} | {fmt(h['auroc'])} | {fmt(h['calibration_slope'])} | "
                         f"{fmt(h['citl'])} | {fmt(h['brier'])} | {cov} | [{rel(run_dir / 'report.md', out)}]({rel(run_dir / 'report.md', out)}) | "
                         f"`{rel(run_dir / 'model', out)}` |")
        lines += ["", "Run folders also hold report.html, metrics.json, predictions_test.parquet and plots/.", ""]
        if self.notes.get("predictions"):
            lines += ["## Predictions (command-line scoring of synthetic rows)", "", f"Input: `{self.notes['scoring_input']}`", ""]
            lines += [f"- `{p['file']}`: {p['rows']} rows from `{p['model']}`; max abs difference to the in-run test predictions "
                      f"{p['max_abs_diff']:.1e}" for p in self.notes["predictions"]]
            lines.append("")
        if "bundle_reload_max_abs_diff" in self.notes:
            lines += [f"Model bundles were reloaded with `load_bundle`; `predict_risk` reproduced the in-run test predictions "
                      f"(max abs difference {self.notes['bundle_reload_max_abs_diff']:.1e}).", ""]
        if self.notes.get("comparison_files"):
            lines += ["## Comparison", "", *[f"- `{f}`" for f in self.notes["comparison_files"]], ""]
        if self.notes.get("ablation_files"):
            lines += ["## Meuhedet-enhanced ablation (ILLUSTRATIVE placeholder feature groups)", "",
                      *[f"- `{f}`" for f in self.notes["ablation_files"]], ""]
        if self.notes.get("monitoring"):
            lines += ["## Drift monitoring", "", *[f"- {k} rows: status **{v['status']}** (`{v['folder']}`)" for k, v in self.notes["monitoring"].items()], ""]
        if self.notes.get("reproduction"):
            r = self.notes["reproduction"]
            lines += ["## Reproduction", "", f"- `{r['reproduced']}` reproduces `{r['original']}` exactly (`runs_reproduced/`).", ""]
        lines += ["## Commands run (from the project folder)", "", "```", *self.commands, "```", "", f"_{common.SYNTHETIC_BANNER}_", ""]
        path = out / "DEMO_SUMMARY.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def headline(metrics: dict[str, Any]) -> dict[str, Any]:
    variant = metrics.get("served_variant")
    test = (metrics.get("performance") or {}).get("test", {}).get(variant) or {}
    return {"variant": variant, **{k: (test.get(k) or {}).get("estimate") for k in ("auroc", "calibration_slope", "citl", "brier")}}


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    common.configure_console()
    ap = argparse.ArgumentParser(description="falls_ml synthetic demos (software demonstration only).")
    ap.add_argument("kind", choices=["baseline", "full"])
    ap.add_argument("--out", help="output folder (default demo_outputs/<kind>)")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else ROOT / common.DEMO_OUTPUTS / args.kind
    out = out if out.is_absolute() else ROOT / out
    try:
        if ROOT / common.DEMO_OUTPUTS in out.resolve().parents:
            common.ensure_demo_root(ROOT)
        common.prepare_generated_dir(out, f"{args.kind} demo")
    except (common.OutputDirError, OSError) as exc:
        common.print_lines(common.failure_block("DEMO FAILED", "Preparing the output folder", str(exc),
                                                "Choose another --out folder or move the existing folder away."))
        return 1
    try:
        return Demo(args.kind, out).run()
    except KeyboardInterrupt:
        common.print_lines(common.failure_block("DEMO FAILED", "interrupted", "stopped by the user (Ctrl+C)", "Run the demo again."))
        return 1


if __name__ == "__main__":
    sys.exit(main())
