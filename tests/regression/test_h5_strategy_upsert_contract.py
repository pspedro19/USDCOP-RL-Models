"""H5 writers must target the composite identity installed by migration 064.

Migration 064 intentionally replaced uniqueness on ``signal_date`` with
``(signal_date, strategy_id)`` so v11 and a challenger can coexist. PostgreSQL
cannot infer that composite constraint from ``ON CONFLICT (signal_date)`` and
raises 42P10 before any row is written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "database" / "migrations" / "064_h5_strategy_id.sql"

ACTIVE_WRITERS = (
    ROOT / "airflow" / "dags" / "forecast_h5_l5_weekly_signal.py",
    ROOT / "airflow" / "dags" / "forecast_h5_l6_weekly_monitor.py",
    ROOT / "airflow" / "dags" / "forecast_h5_l7_multiday_executor.py",
    ROOT / "scripts" / "pipeline" / "train_and_export_smart_simple.py",
)

EXPECTED_WRITER_INVENTORY = {
    "airflow/dags/forecast_h5_l5_weekly_signal.py": ["forecast_h5_signals"],
    "airflow/dags/forecast_h5_l6_weekly_monitor.py": [
        "forecast_h5_paper_trading"
    ],
    "airflow/dags/forecast_h5_l7_multiday_executor.py": [
        "forecast_h5_executions"
    ],
    "scripts/pipeline/train_and_export_smart_simple.py": [
        "forecast_h5_executions",
        "forecast_h5_executions",
        "forecast_h5_paper_trading",
        "forecast_h5_signals",
    ],
}

UPSERT_RE = re.compile(
    r"""
    INSERT\s+INTO\s+
    (?P<table>
        forecast_h5_signals
        |forecast_h5_executions
        |forecast_h5_paper_trading
    )
    \s*\((?P<columns>[^)]*)\)
    \s*VALUES\b.*?
    ON\s+CONFLICT\s*\((?P<target>[^)]*)\)
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def _upserts() -> list[tuple[str, str, int, tuple[str, ...], tuple[str, ...]]]:
    rows: list[tuple[str, str, int, tuple[str, ...], tuple[str, ...]]] = []
    for path in ACTIVE_WRITERS:
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        per_table: dict[str, int] = {}
        for match in UPSERT_RE.finditer(text):
            table = match.group("table").lower()
            per_table[table] = per_table.get(table, 0) + 1
            columns = tuple(
                token.strip().lower()
                for token in match.group("columns").split(",")
                if token.strip()
            )
            target = tuple(
                token.strip().lower()
                for token in match.group("target").split(",")
                if token.strip()
            )
            rows.append((relative, table, per_table[table], columns, target))
    return rows


UPSERTS = _upserts()


def _case_id(case: tuple[str, str, int, tuple[str, ...], tuple[str, ...]]) -> str:
    path, table, occurrence, _, _ = case
    return f"{Path(path).stem}-{table}-{occurrence}"


def test_writer_inventory_is_complete_and_non_vacuous() -> None:
    observed: dict[str, list[str]] = {}
    for path, table, _, _, _ in UPSERTS:
        observed.setdefault(path, []).append(table)
    observed = {path: sorted(tables) for path, tables in observed.items()}
    expected = {
        path: sorted(tables) for path, tables in EXPECTED_WRITER_INVENTORY.items()
    }

    assert observed == expected


def test_migration_064_declares_the_composite_conflict_identity() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    for table in (
        "forecast_h5_signals",
        "forecast_h5_executions",
        "forecast_h5_paper_trading",
    ):
        pattern = re.compile(
            rf"ALTER\s+TABLE\s+{table}.*?"
            r"UNIQUE\s*\(\s*signal_date\s*,\s*strategy_id\s*\)",
            re.IGNORECASE | re.DOTALL,
        )
        assert pattern.search(sql), f"064 lacks composite identity for {table}"


@pytest.mark.parametrize("case", UPSERTS, ids=_case_id)
def test_each_h5_upsert_uses_the_migration_064_identity(
    case: tuple[str, str, int, tuple[str, ...], tuple[str, ...]],
) -> None:
    path, table, occurrence, columns, target = case

    assert "strategy_id" in columns, (
        f"{path}: INSERT #{occurrence} into {table} relies on a default "
        "strategy_id, so its identity is not explicit or reviewable"
    )
    assert target == ("signal_date", "strategy_id"), (
        f"{path}: INSERT #{occurrence} into {table} targets {target}; "
        "PostgreSQL migration 064 only provides (signal_date, strategy_id)"
    )
