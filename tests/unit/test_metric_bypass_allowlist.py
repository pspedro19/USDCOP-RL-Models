from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _validator():
    path = REPO_ROOT / "scripts" / "validation" / "validate_fabric_contracts.py"
    spec = importlib.util.spec_from_file_location("validate_fabric_contracts_bl18", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_metric_bypass_inventory_is_exact() -> None:
    assert _validator().validate() == []


def test_new_metric_implementation_is_red_until_inventory_changes() -> None:
    validator = _validator()
    assert validator.validate_metric_bypass_inventory(
        {"max_entries": 0, "entries": []},
        {"src/legacy.py::calculate_sharpe"},
    ) == [
        "unallowlisted metric implementation: src/legacy.py::calculate_sharpe"
    ]


def test_allowlist_cannot_expand_without_retiring_an_entry() -> None:
    validator = _validator()
    identifier = "src/legacy.py::calculate_sharpe"
    assert validator.validate_metric_bypass_inventory(
        {"max_entries": 0, "entries": [identifier]}, {identifier}
    ) == [
        "allowlist expanded: 1 entries exceeds frozen ceiling 0"
    ]


def test_metric_inventory_excludes_tests_but_not_similarly_named_runtime() -> None:
    validator = _validator()
    assert not validator.is_runtime_python_path(Path("src/engine/test_strategy.py"))
    assert not validator.is_runtime_python_path(Path("tests/unit/strategy.py"))
    assert validator.is_runtime_python_path(Path("src/engine/strategy_test.py"))
    assert validator.is_runtime_python_path(Path("services/analytics.py"))
