from __future__ import annotations

import ast
from pathlib import Path

import pytest

from services.demo_mode.config import DEMO_MODEL_ID, load_demo_config
from src.governance.synthetic_isolation import SyntheticIsolationError


class _Connection:
    def __init__(self, row):
        self.row = row
        self.calls: list[tuple[str, str]] = []

    async def fetchrow(self, query: str, model_id: str):
        self.calls.append((query, model_id))
        return self.row


def _demo_row(**overrides):
    row = {
        "model_id": DEMO_MODEL_ID,
        "display_name": "[DEMO - SYNTHETIC] Investor Demo",
        "algorithm": "SYNTHETIC",
        "environment": "demo",
        "surface": "synthetic",
        "execution_eligible": False,
        "display_metadata": {},
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_load_demo_config_reads_isolated_display_view() -> None:
    connection = _Connection(_demo_row())

    config = await load_demo_config(connection)

    query, model_id = connection.calls[0]
    assert "FROM demo.synthetic_model_display" in query
    assert "config.models" not in query
    assert model_id == DEMO_MODEL_ID
    assert config.model_id == DEMO_MODEL_ID
    assert config.model_name.startswith("[DEMO - SYNTHETIC]")


@pytest.mark.asyncio
async def test_load_demo_config_fails_closed_when_registration_is_missing() -> None:
    with pytest.raises(SyntheticIsolationError, match="not registered"):
        await load_demo_config(_Connection(None))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("algorithm", "PPO"),
        ("environment", "production"),
        ("surface", "action"),
        ("execution_eligible", True),
    ],
)
async def test_load_demo_config_revalidates_database_boundary(field, value) -> None:
    with pytest.raises(SyntheticIsolationError):
        await load_demo_config(_Connection(_demo_row(**{field: value})))


def test_demo_backtest_consumes_governed_registration() -> None:
    path = Path("services/inference_api/routers/backtest.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_demo_backtest"
    )
    loads = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "load_demo_config"
    ]
    assert len(loads) == 1
    generators = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DemoTradeGenerator"
    ]
    assert len(generators) == 1
    config_kw = next(kw for kw in generators[0].keywords if kw.arg == "config")
    assert isinstance(config_kw.value, ast.Name)
    assert config_kw.value.id == "demo_config"
