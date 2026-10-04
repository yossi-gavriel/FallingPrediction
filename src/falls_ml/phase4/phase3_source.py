"""READ-ONLY access to a finished Phase 3 output folder (the frozen 2025 development run).

Nothing is ever written, moved or locked in the Phase 3 folder. Every file Phase 4 uses is checked against the Phase 3 commit records (the sha256
of every committed item file, written last at commit time) before it is read; a changed file is a HARD stop. The digest of all verified files is
recorded in the Phase 4 plan and re-checked by every later command.

The frozen models:
    A  the persisted final fit of Phase 3 (``S07_lasso/items/LASSO__<set>__final/model.pkl``, fitted on the 2025 TRAIN rows whose used predictors
       are all KNOWN), loaded after its commit record was verified;
    B  only when A does not exist: a deterministic re-fit with the UNCHANGED Phase 3 code (``phase3.stages_models.linear_item``, fold 'final',
       the same item / inner-fold seeds, the frozen FEATURE_SETS.json, the Phase 3 configuration verified by sha256) on a private copy of the
       committed 2025 products - no 2026 information exists at that point.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from falls_ml.phase2 import durable as D
from falls_ml.phase2.state import Phase2Stop, safe_item_id

S01, S03, S04, S06 = ("01", "cohort", "cohort"), ("03", "recovery", "recovery"), ("04", "reconstruct", "reconstruct"), ("06", "feature_sets", "feature_sets")


@dataclass
class Phase3Source:
    out: Path
    allow_unfrozen: bool = False
    verified: dict[str, str] = field(default_factory=dict)          # relative path -> sha256 (verified against a commit record)
    cache: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- plan / state
    def plan(self) -> dict[str, Any]:
        if "plan" not in self.cache:
            p = self.out / "PHASE3_PLAN.json"
            if not p.is_file():
                raise Phase2Stop("PHASE3_NOT_FOUND", f"{self.out.name}: no PHASE3_PLAN.json - pass the Phase 3 output folder (--phase3-out)")
            self.cache["plan"] = json.loads(p.read_text(encoding="utf-8"))
            self.verified["PHASE3_PLAN.json"] = D.sha256_file(p)
        return self.cache["plan"]

    def run_state(self) -> dict[str, Any]:
        p = self.out / "RUN_STATE.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def check_frozen(self) -> dict[str, Any]:
        plan = self.plan()["plan"]
        frozen = bool(plan.get("frozen_production_config"))
        if not frozen and not self.allow_unfrozen:
            raise Phase2Stop("PHASE3_NOT_FROZEN_PRODUCTION", "the Phase 3 run did not use the frozen production configuration (PHASE3_FINAL_EXPERIMENT_CONFIG.json)",
                             ["temporal validation is only defined for the frozen Phase 3 models"])
        return {"frozen_production_config": frozen, "phase3_version": plan.get("phase3_version"), "plan_sha256": self.plan()["plan_sha256"],
                "config_sha256": plan.get("config_sha256"), "final_experiment_config_sha256": plan.get("final_experiment_config_sha256"),
                "input_sha256_2025": plan.get("input_sha256"), "seed": plan.get("seed"), "run_status": self.run_state().get("status"),
                "decision": self.run_state().get("decision")}

    # ---------------------------------------------------------------- verified access
    def _item_dir(self, sid: str, name: str, item: str) -> tuple[Path, dict[str, Any]]:
        """Verify one committed item (record + every file sha256). Never writes."""
        sdir = self.out / "stages" / f"S{sid}_{name}" / "items"
        iid = safe_item_id(item)
        rec_path = sdir / f"{iid}.COMPLETE.json"
        if not rec_path.is_file():
            raise Phase2Stop("PHASE3_ITEM_MISSING", f"Phase 3 item S{sid}_{name}/{item} is not committed")
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise Phase2Stop("PHASE3_ITEM_MISSING", f"Phase 3 commit record S{sid}_{name}/{item} is unreadable ({exc})") from exc
        if rec.get("plan_sha256") != self.plan()["plan_sha256"]:
            raise Phase2Stop("PHASE3_ARTIFACT_CHANGED", f"S{sid}_{name}/{item}: committed under a different Phase 3 plan")
        d = sdir / iid
        problems = []
        for rel, digest in rec["files"].items():
            p = d / rel
            if not p.is_file():
                problems.append(f"{rel}: missing")
            elif D.sha256_file(p) != digest:
                problems.append(f"{rel}: content changed")
            else:
                self.verified[p.relative_to(self.out).as_posix()] = digest
        if problems:
            raise Phase2Stop("PHASE3_ARTIFACT_CHANGED", f"Phase 3 item S{sid}_{name}/{item} no longer matches its commit record", problems)
        self.verified[rec_path.relative_to(self.out).as_posix()] = D.sha256_file(rec_path)
        return d, rec

    def has_item(self, sid: str, name: str, item: str) -> bool:
        return (self.out / "stages" / f"S{sid}_{name}" / "items" / f"{safe_item_id(item)}.COMPLETE.json").is_file()

    def stage_complete(self, sid: str) -> bool:
        return (self.out / f"STAGE_{sid}_COMPLETE.json").is_file()

    def file(self, sid: str, name: str, item: str, rel: str) -> Path:
        d, _ = self._item_dir(sid, name, item)
        p = d / rel
        if not p.is_file():
            raise Phase2Stop("PHASE3_ITEM_MISSING", f"S{sid}_{name}/{item}/{rel} missing")
        return p

    def digest(self) -> str:
        return D.sha256_text(json.dumps(sorted(self.verified.items())))

    # ---------------------------------------------------------------- the committed 2025 products
    def feature_sets(self) -> Any:
        from falls_ml.phase2.featuresets import FeatureSets

        if "fs" not in self.cache:
            if not self.stage_complete("06"):
                raise Phase2Stop("PHASE3_INCOMPLETE", "Phase 3 never froze its feature sets (S06): the run stopped before modelling (Outcome B or an "
                                                      "earlier stop) - there is no frozen model to validate")
            self.cache["fs"] = FeatureSets.from_dict(json.loads(self.file(*S06, "FEATURE_SETS.json").read_text(encoding="utf-8")))
        return self.cache["fs"]

    def work(self) -> pd.DataFrame:
        if "work" not in self.cache:
            self.cache["work"] = pd.read_parquet(self.file(*S01, "trainval.parquet"))
        return self.cache["work"]

    def values(self) -> pd.DataFrame:
        if "values" not in self.cache:
            self.cache["values"] = pd.read_parquet(self.file(*S04, "values.parquet"))
        return self.cache["values"]

    def unknown(self) -> pd.DataFrame:
        if "unknown" not in self.cache:
            self.cache["unknown"] = pd.read_parquet(self.file(*S04, "unknown.parquet"))
        return self.cache["unknown"]

    def registry(self) -> pd.DataFrame:
        if "reg" not in self.cache:
            self.cache["reg"] = pd.read_csv(self.file(*S03, "FEATURE_RECOVERY.csv"))
        return self.cache["reg"]

    def feature_source(self) -> dict[str, str]:
        if "fsrc" not in self.cache:
            self.cache["fsrc"] = json.loads(self.file(*S03, "feature_source.json").read_text(encoding="utf-8"))
        return self.cache["fsrc"]

    def _train_mask(self) -> np.ndarray:
        return (self.work()["partition"] == "train").to_numpy()

    def train_frame(self) -> pd.DataFrame:
        """Exactly ``P3Ctx.train_frame``: committed cohort rows + reconstructed values, TRAIN partition."""
        if "train" not in self.cache:
            w, v = self.work(), self.values()
            if len(w) != len(v):
                raise Phase2Stop("PHASE3_ARTIFACT_CHANGED", "Phase 3 reconstructed values are not aligned with its cohort rows")
            dup = [c for c in v.columns if c in w.columns]
            both = pd.concat([w.drop(columns=dup), v], axis=1)
            self.cache["train"] = both.loc[self._train_mask()].reset_index(drop=True)
        return self.cache["train"]

    def y_train(self) -> np.ndarray:
        return self.train_frame()["y"].to_numpy(dtype=int)

    def unknown_train(self) -> pd.DataFrame:
        return self.unknown().loc[self._train_mask()].reset_index(drop=True)

    def fit_mask_train(self) -> np.ndarray:
        """Exactly ``P3Ctx.fit_mask_train``: TRAIN rows whose every scope feature (of ANY fitted set) is KNOWN."""
        fs = self.feature_sets()
        scope: set[str] = set()
        for s in fs.sets.values():
            scope |= set(s["features"]) | set(s["baseline"])
        u = self.unknown_train()
        cols = [c for c in sorted(scope) if c in u.columns]
        return ~u[cols].to_numpy(dtype=bool).any(axis=1) if cols else np.ones(len(u), dtype=bool)

    def fit_frame(self) -> pd.DataFrame:
        return self.train_frame().loc[self.fit_mask_train()].reset_index(drop=True)

    # ---------------------------------------------------------------- frozen models
    def model_item(self, setname: str) -> tuple[str, str]:
        fitted = self.feature_sets().fitted_name(setname)
        return fitted, f"LASSO__{fitted}__final"

    def persisted_model(self, setname: str, stage: str) -> dict[str, Any] | None:
        """Mode A: the committed final fit, or None when Phase 3 never committed it."""
        sid, name = stage.split("_", 1)
        fitted, item = self.model_item(setname)
        if not self.has_item(sid, name, item):
            return None
        d, rec = self._item_dir(sid, name, item)
        if rec["result"].get("failed"):
            raise Phase2Stop("PHASE3_MODEL_FAILED", f"Phase 3 recorded the final fit of {setname} as failed: {rec['result'].get('error')}")
        pkl = d / "model.pkl"
        return {"mode": "A", "fitted_set": fitted, "item": item, "path": pkl, "sha256": rec["files"]["model.pkl"],
                "coefficients_sha256": rec["files"].get("coefficients.csv"), "record_sha256": D.sha256_file(d.parent / f"{safe_item_id(item)}.COMPLETE.json"),
                "result": {k: v for k, v in rec["result"].items() if k in ("lambda", "l1_ratio", "n_fit_rows", "n_selected", "n_design_columns", "retry")},
                "code_sha256_phase3": rec.get("code_sha256")}

    def oof(self, setname: str, stage: str) -> tuple[np.ndarray, np.ndarray] | None:
        """The committed Phase 3 outer out-of-fold prediction intervals of a set on 2025 TRAIN (or None)."""
        sid, name = stage.split("_", 1)
        fitted, _ = self.model_item(setname)
        K = int(self.plan()["plan"]["config"]["cv"]["outer_folds"])
        n = len(self.y_train())
        lo, hi = np.full(n, np.nan), np.full(n, np.nan)
        for k in range(K):
            item = f"LASSO__{fitted}__outer{k}"
            if not self.has_item(sid, name, item):
                return None
            d, rec = self._item_dir(sid, name, item)
            if rec["result"].get("failed"):
                return None
            z = np.load(d / "oof.npz")
            lo[z["rows"]], hi[z["rows"]] = z["lo"], z["hi"]
        if np.isnan(lo).any():
            return None
        return lo, hi

    def refit(self, setname: str, *, config_path: str | Path, workdir: Path) -> dict[str, Any]:
        """Mode B (only when A is absent): deterministic re-fit with the unchanged Phase 3 code on a PRIVATE COPY of the committed 2025 products."""
        from falls_ml.artifacts import source_tree_sha256
        from falls_ml.phase2.fitting import LASSO
        from falls_ml.phase2.state import EventLog, Run
        from falls_ml.phase3.context import P3Ctx
        from falls_ml.phase3.runner import load_all
        from falls_ml.phase3.stages_models import linear_item

        plan = self.plan()
        L = load_all(config_path)
        checks = {"config": (L["cfg"].sha256, plan["plan"]["config_sha256"]), "catalogue": (L["cat"].sha256, plan["plan"]["catalogue_sha256"]),
                  "mapping": (L["mapping"].content_sha256, plan["plan"]["mapping_sha256"]), "contract": (L["contract"].content_sha256, plan["plan"]["contract_sha256"])}
        bad = [f"{k}: package {a[:12]}… vs Phase 3 run {str(b)[:12]}…" for k, (a, b) in checks.items() if a != b]
        if bad:
            raise Phase2Stop("REFIT_CONFIG_MISMATCH", "the Phase 3 settings in this package differ from those of the 2025 run: a re-fit would not be the frozen model", bad)
        fs = self.feature_sets()
        for sid, name, item in (S01, S03, S04, S06):           # verify, then copy the committed products (never touch the Phase 3 folder)
            d, _ = self._item_dir(sid, name, item)
            dst = workdir / "phase3_copy" / "stages" / f"S{sid}_{name}" / "items"
            dst.mkdir(parents=True, exist_ok=True)
            if not (dst / d.name).exists():
                shutil.copytree(d, dst / d.name)
                shutil.copyfile(d.parent / f"{d.name}.COMPLETE.json", dst / f"{d.name}.COMPLETE.json")
        clone = workdir / "phase3_copy"
        run = Run(out=clone, plan=plan["plan"], plan_sha=plan["plan_sha256"], code_sha=source_tree_sha256(), attempt=1, seed=int(plan["plan"]["seed"]),
                  events=EventLog(clone / "logs" / "events.jsonl", 1))
        ctx = P3Ctx(run=run, src=Path("."), ref_dir=Path("."), pepper_file=None, cfg=L["cfg"], rules=L["rules"], tc=L["tc"], cat=L["cat"],
                    contract=L["contract"], mapping=L["mapping"], dictionary=L["dictionary"], d00=L["d00"], espec=L["espec"],
                    threads=plan["plan"]["threads"], protected=[], input_sha=plan["plan"]["input_sha256"])
        ctx.cache["fs"] = fs
        fitted, item = self.model_item(setname)
        tmp = workdir / "refit" / safe_item_id(item)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        res = linear_item(ctx, LASSO, fitted, "final")(tmp, run.item_seed("S07_lasso", item))
        if res.get("failed"):
            raise Phase2Stop("REFIT_FAILED", f"the deterministic 2025 re-fit of {setname} failed: {res.get('error')}")
        return {"mode": "B", "fitted_set": fitted, "item": item, "path": tmp / "model.pkl", "sha256": D.sha256_file(tmp / "model.pkl"),
                "coefficients_sha256": D.sha256_file(tmp / "coefficients.csv"), "record_sha256": None,
                "result": {k: v for k, v in res.items() if k in ("lambda", "l1_ratio", "n_fit_rows", "n_selected", "n_design_columns", "retry")},
                "code_sha256_phase3": None}
