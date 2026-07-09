"""
Regression: freshness gate is symbol-parameterized (audit A1-09).

The pre-training freshness gate used to hardcode WHERE symbol='USD/COP', so a 2nd
asset silently reused COP's freshness. Guard that the symbol is now a parameter and
is safely escaped into the WHERE clause.
"""
import inspect
import os
import sys
from pathlib import Path

import pytest

# utils/__init__ -> dag_common reads POSTGRES_PASSWORD at import time. The test
# never opens a real connection (check_table_freshness/get_db_connection are
# monkeypatched), so a dummy value just satisfies the import.
os.environ.setdefault("POSTGRES_PASSWORD", "test_dummy")

_DAGS = str(Path(__file__).resolve().parents[2] / "airflow" / "dags")
if _DAGS not in sys.path:
    sys.path.insert(0, _DAGS)


@pytest.fixture(scope="module")
def data_quality():
    return __import__("utils.data_quality", fromlist=["*"])


def test_freshness_gate_accepts_symbol_param(data_quality):
    sig = inspect.signature(data_quality.validate_training_data_freshness)
    assert "symbol" in sig.parameters, "freshness gate must accept a `symbol` param (A1-09)"
    assert sig.parameters["symbol"].default == "USD/COP", "default must preserve COP behavior"


def test_freshness_gate_symbol_is_escaped(data_quality, monkeypatch):
    """The symbol must flow into the WHERE clause with single-quote escaping."""
    captured = {}

    def fake_check(conn, table, col, max_age, label, where_clause=None):
        # Capture per-table so the macro check (no WHERE) doesn't clobber the OHLCV one.
        captured[table] = where_clause
        return "2026-07-01"

    # Avoid a real DB connection.
    monkeypatch.setattr(data_quality, "check_table_freshness", fake_check, raising=True)

    import types
    fake_common = types.ModuleType("utils.dag_common")
    fake_common.get_db_connection = lambda: types.SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, "utils.dag_common", fake_common)

    data_quality.validate_training_data_freshness(symbol="XAU/USD")
    assert captured["usdcop_m5_ohlcv"] == "WHERE symbol = 'XAU/USD'"

    data_quality.validate_training_data_freshness(symbol="A'B")
    assert captured["usdcop_m5_ohlcv"] == "WHERE symbol = 'A''B'", "single quotes must be escaped"
