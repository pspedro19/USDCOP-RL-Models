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


def test_metric_inventory_excludes_test_directories_not_runtime_filenames() -> None:
    validator = _validator()
    assert validator.is_runtime_python_path(Path("src/engine/test_strategy.py"))
    assert not validator.is_runtime_python_path(Path("src/tests/strategy.py"))
    assert not validator.is_runtime_python_path(Path("services/api/tests/test_api.py"))
    assert validator.is_runtime_python_path(Path("src/engine/strategy_test.py"))
    assert validator.is_runtime_python_path(Path("services/analytics.py"))


def test_metric_inventory_scans_test_prefixed_file_inside_runtime_root(tmp_path: Path) -> None:
    validator = _validator()
    candidate = tmp_path / "src" / "engine" / "test_strategy.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        "def sneaky_sharpe(values):\n    return sum(values) / len(values)\n",
        encoding="utf-8",
    )

    assert validator.discover_metric_bypasses(tmp_path) == {
        "src/engine/test_strategy.py::sneaky_sharpe"
    }


def test_metric_inventory_exempts_direct_and_transitive_ssot_delegation(tmp_path: Path) -> None:
    validator = _validator()
    candidate = tmp_path / "src" / "legacy.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        "def probabilistic_sharpe(value):\n"
        "    from services.common.metrics import probabilistic_sharpe_ratio\n"
        "    return float(probabilistic_sharpe_ratio(value, 20))\n\n"
        "def deflated_sharpe(value):\n"
        "    return probabilistic_sharpe(value)\n",
        encoding="utf-8",
    )

    assert validator.discover_metric_bypasses(tmp_path) == set()


def test_metric_inventory_does_not_exempt_local_formula_near_ssot_import(tmp_path: Path) -> None:
    validator = _validator()
    candidate = tmp_path / "src" / "legacy.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        "from services.common.metrics import sharpe_ratio\n\n"
        "def calculate_sharpe(values):\n"
        "    return sum(values) / len(values)\n",
        encoding="utf-8",
    )

    assert validator.discover_metric_bypasses(tmp_path) == {
        "src/legacy.py::calculate_sharpe"
    }
