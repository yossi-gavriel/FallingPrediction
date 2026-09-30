"""Freeze the effective Phase 2 configuration: writes configs/meuhedet/FINAL_EXPERIMENT_CONFIG.json and .sha256 from
configs/meuhedet/phase2.yaml + the feature catalogue + the result-relevant code constants (falls_ml.phase2.final_config).

Run ONLY after a reviewed change (reviews/ASTRA_FINAL_RESOLUTION.md). The runner and the preflight refuse a production run whose effective
configuration differs from the frozen file; tests/unit/test_phase2_final.py fails when the repository copy is stale.

    python tools/freeze_phase2_config.py            # write
    python tools/freeze_phase2_config.py --check    # exit 1 if the frozen copy is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    from falls_ml.d00.dependency import load_d00_config
    from falls_ml.data.meuhedet_wide import load_wide_contract, load_wide_mapping
    from falls_ml.eda.dictionary import load_data_dictionary
    from falls_ml.phase2.config import load_catalogue, load_phase2_config
    from falls_ml.phase2.final_config import FROZEN_PATH, build_final_config, sha256_of, sha_line, to_bytes

    cfg = load_phase2_config()
    mapping = load_wide_mapping()
    contract = load_wide_contract(mapping.contract_path)
    dictionary = load_data_dictionary("configs/meuhedet/wide_v1_data_dictionary.yaml", contract)
    d00 = load_d00_config(cfg["d00_config"], contract=contract, dictionary=dictionary)
    cat = load_catalogue(cfg["features"], contract, dictionary, mapping)
    fc = build_final_config(cfg, cat, contract=contract, dictionary=dictionary, mapping=mapping, d00=d00)
    data, sha = to_bytes(fc), sha256_of(fc)
    target = ROOT / FROZEN_PATH
    shafile = target.with_suffix(".sha256")
    if a.check:
        ok = target.is_file() and target.read_bytes() == data and shafile.is_file() and shafile.read_text(encoding="utf-8") == sha_line(sha)
        print(("FROZEN copy is current: " if ok else "STALE frozen copy; effective sha256 ") + sha)
        return 0 if ok else 1
    target.write_bytes(data)
    shafile.write_bytes(sha_line(sha).encode("utf-8"))
    print(f"wrote {FROZEN_PATH} ({len(data)} bytes) sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
