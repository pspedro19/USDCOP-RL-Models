"""The H5 performance view must preserve the composite strategy identity."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIGRATION_083 = ROOT / "database" / "migrations" / "083_h5_strategy_performance_view.sql"


def _normalized_sql() -> str:
    return re.sub(r"\s+", " ", MIGRATION_083.read_text(encoding="utf-8")).lower()


def test_migration_083_replaces_the_existing_view_without_mutating_history() -> None:
    sql = _normalized_sql()

    assert "create or replace view v_h5_performance_summary as" in sql
    assert "alter table" not in sql
    assert "drop table" not in sql


def test_performance_view_projects_and_joins_the_composite_identity() -> None:
    sql = _normalized_sql()

    select_list = sql.split(" from forecast_h5_executions e", 1)[0]
    assert "e.strategy_id" in select_list
    assert re.search(
        r"left join forecast_h5_signals s on "
        r"e\.signal_date\s*=\s*s\.signal_date and "
        r"e\.strategy_id\s*=\s*s\.strategy_id",
        sql,
    )


def test_performance_view_orders_with_strategy_as_a_stable_tiebreaker() -> None:
    sql = _normalized_sql()

    assert re.search(
        r"order by e\.signal_date desc, e\.strategy_id(?: asc)?\s*;",
        sql,
    )
