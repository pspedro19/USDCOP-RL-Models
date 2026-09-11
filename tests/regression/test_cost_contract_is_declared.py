"""Cost assumptions are declared, unitful and auditable."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.research.cost_contract import COST_CONTRACT, CONTRACT_PATH, load_cost_contract


def test_cost_contract_declares_unit_source_and_three_scenarios() -> None:
    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert raw["unit"] == "cop_per_usd"
    assert raw["market_tick_cop"] == 0.01
    assert raw["observability"] == "implicit_bounds"
    assert set(raw["scenarios"]) == {"low", "central", "high"}
    assert all(len(values) == 3 for values in raw["scenarios"].values())
    assert "TwelveData" in raw["source"]


def test_cost_contract_loader_fails_closed_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    try:
        load_cost_contract(missing)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing cost contract was accepted")


def test_runtime_uses_declared_central_scenario() -> None:
    assert COST_CONTRACT.scenarios["central"] == (2.0, 3.0, 6.0)
