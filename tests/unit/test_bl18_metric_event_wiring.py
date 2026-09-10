from __future__ import annotations

import ast
from pathlib import Path

from tests.unit.test_bl16_declaration_gate import _dependency_edges, _task_var_by


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


def _reachable(edges: set[tuple[str, str]], start: str, target: str) -> bool:
    visited, frontier = {start}, [start]
    while frontier:
        current = frontier.pop()
        for upstream, downstream in edges:
            if upstream == current and downstream not in visited:
                visited.add(downstream)
                frontier.append(downstream)
    return target in visited


def _executed_sql(source: str, function_name: str) -> list[str]:
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    queries: list[str] = []
    for call in ast.walk(function):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "execute"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            queries.append(call.args[0].value)
    return queries


def _consumer_uses_governed_ledger(source: str) -> bool:
    queries = _executed_sql(source, "evaluate_pnl_clock")
    return (
        any("FROM control.metric_event" in query for query in queries)
        and any("status <> 'INSUFFICIENT_SAMPLE'" in query for query in queries)
        and not any(
            "running_sharpe" in query and "forecast_h5_paper_trading" in query
            for query in queries
        )
    )


def test_h5_producer_computes_and_persists_governed_sharpe() -> None:
    body = _function_source(PRODUCER, "persist_governed_metric_events")

    assert "MetricEngine.from_asset_registry" in body
    assert 'metric="strategy.sharpe"' in body
    assert "persist_metric_event_dbapi" in body
    assert "conn.commit()" in body
    assert "conn.rollback()" in body


def test_health_consumer_reads_metric_event_without_legacy_sharpe_fallback() -> None:
    assert _consumer_uses_governed_ledger(CONSUMER.read_text(encoding="utf-8"))


def test_consumer_source_lock_rejects_a_legacy_query_even_if_comments_claim_governance() -> None:
    source = CONSUMER.read_text(encoding="utf-8")
    mutated = source.replace(
        "SELECT metric_value\n                FROM control.metric_event",
        "SELECT running_sharpe\n                FROM forecast_h5_paper_trading",
        1,
    )
    mutated = "# FROM control.metric_event status <> 'INSUFFICIENT_SAMPLE'\n" + mutated

    assert mutated != source
    assert not _consumer_uses_governed_ledger(mutated)


def test_metric_event_task_is_causally_between_evaluation_and_consumers() -> None:
    tree = ast.parse(PRODUCER.read_text(encoding="utf-8"))
    edges = _dependency_edges(tree)
    metric = _task_var_by(
        tree,
        lambda kwargs: isinstance(kwargs.get("python_callable"), ast.Name)
        and kwargs["python_callable"].id == "persist_governed_metric_events",
    )
    persist = _task_var_by(
        tree,
        lambda kwargs: isinstance(kwargs.get("task_id"), ast.Constant)
        and kwargs["task_id"].value == "persist_evaluation",
    )
    alert = _task_var_by(
        tree,
        lambda kwargs: isinstance(kwargs.get("task_id"), ast.Constant)
        and kwargs["task_id"].value == "alert_summary",
    )
    ledger = _task_var_by(
        tree,
        lambda kwargs: isinstance(kwargs.get("task_id"), ast.Constant)
        and kwargs["task_id"].value == "paper_ledger_2026",
    )

    assert None not in (metric, persist, alert, ledger)
    assert _reachable(edges, persist, metric)
    assert _reachable(edges, metric, alert)
    assert _reachable(edges, metric, ledger)


def test_causal_lock_rejects_an_orphan_even_if_a_comment_contains_the_old_chain() -> None:
    source = PRODUCER.read_text(encoding="utf-8")
    mutated = source.replace(
        "t_load >> t_metrics >> t_gates >> t_persist >> t_metric_event >> t_alert",
        "# t_persist >> t_metric_event >> t_alert\n"
        "    t_load >> t_metrics >> t_gates >> t_persist >> t_alert",
        1,
    ).replace(
        "t_metric_event >> t_paper_ledger >> t_verify_anchor",
        "t_persist >> t_paper_ledger >> t_verify_anchor",
        1,
    )
    tree = ast.parse(mutated)
    edges = _dependency_edges(tree)
    metric = _task_var_by(
        tree,
        lambda kwargs: isinstance(kwargs.get("python_callable"), ast.Name)
        and kwargs["python_callable"].id == "persist_governed_metric_events",
    )
    persist = _task_var_by(
        tree,
        lambda kwargs: isinstance(kwargs.get("task_id"), ast.Constant)
        and kwargs["task_id"].value == "persist_evaluation",
    )

    assert mutated != source
    assert metric is not None and persist is not None
    assert not _reachable(edges, persist, metric)
