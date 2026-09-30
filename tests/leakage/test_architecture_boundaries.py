"""Dataset boundary (architecture §1, user requirement 4): ML code never touches the clinical DWH or raw event tables.

Static proof over the source tree:
- no module outside ``falls_ml.dataeng`` / ``falls_ml.data.synthetic`` imports ``falls_ml.dataeng``;
- no module imports a database driver or ORM;
- patient identifiers never appear in a model design matrix (checked on the published representation).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "falls_ml"
DB_MODULES = {"pyodbc", "sqlalchemy", "psycopg2", "psycopg", "pymssql", "cx_Oracle", "oracledb", "sqlite3", "mysql", "pymysql", "teradatasql"}
ALLOWED_DATAENG_IMPORTERS = {"dataeng", "data/synthetic.py"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.relative_to(SRC).as_posix())
def test_ml_layer_does_not_import_data_engineering(path: Path) -> None:
    rel = path.relative_to(SRC).as_posix()  # POSIX separators: the allow-list must match on Windows too
    if rel.startswith("dataeng") or rel in ALLOWED_DATAENG_IMPORTERS:
        return
    offending = {m for m in _imports(path) if m == "falls_ml.dataeng" or m.startswith("falls_ml.dataeng.")}
    assert not offending, f"{rel} imports the data-engineering layer: {offending}"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.relative_to(SRC).as_posix())
def test_no_database_drivers(path: Path) -> None:
    offending = {m for m in _imports(path) if m.split(".")[0] in DB_MODULES}
    assert not offending, f"{path.relative_to(SRC).as_posix()} imports database drivers: {offending}"


def test_identifiers_never_predictors() -> None:
    from falls_ml.features.spec import load_feature_spec

    spec = load_feature_spec(Path(__file__).resolve().parents[2] / "configs" / "features" / "efalls_v1.yaml")
    forbidden = set(spec.identifier_columns) | {spec.index_column, spec.outcome.name} | set(spec.metadata_columns) | set(spec.provenance_columns)
    assert not forbidden & set(spec.predictor_names())
