from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRODUCER = ROOT / "airflow" / "dags" / "forecast_h5_l6_weekly_monitor.py"
CONSUMER = ROOT / "airflow" / "dags" / "control_system_health.py"


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def test_h5_producer_computes_and_persists_governed_sharpe() -> None:
    body = _function_source(PRODUCER, "persist_governed_metric_events")

    assert "MetricEngine.from_asset_registry" in body
    assert 'metric="strategy.sharpe"' in body
    assert "persist_metric_event_dbapi" in body
    assert "conn.commit()" in body
    assert "conn.rollback()" in body


def test_health_consumer_reads_metric_event_without_legacy_sharpe_fallback() -> None:
    body = _function_source(CONSUMER, "evaluate_pnl_clock")

    assert "FROM control.metric_event" in body
    assert "INSUFFICIENT_SAMPLE" in body
    assert "SELECT running_sharpe FROM forecast_h5_paper_trading" not in body


def test_metric_event_task_is_causally_between_evaluation_and_consumers() -> None:
    source = PRODUCER.read_text(encoding="utf-8")

    assert "t_persist >> t_metric_event >> t_alert" in source
    assert "t_metric_event >> t_paper_ledger >> t_verify_anchor" in source
