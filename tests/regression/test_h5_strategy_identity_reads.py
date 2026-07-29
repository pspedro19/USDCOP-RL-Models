"""The other half of migration 064: identity on WRITES *and* on READS.

Migration 064 replaced uniqueness on ``signal_date`` with
``(signal_date, strategy_id)`` on three H5 tables so v11 and a challenger can
coexist. Two failure modes follow, and they fail very differently:

* A writer with ``ON CONFLICT (signal_date)`` raises 42P10 and writes **nothing**.
  That is loud once you look — it froze ``forecast_h5_*`` from 2026-07-07 — and
  ``test_h5_strategy_upsert_contract.py`` (Codex) is the guard for it.
* A reader with no ``strategy_id`` filter keeps returning rows and is **silent**.
  The day a second strategy writes, v11 gets sized off v12's losing streak, v11's
  live PnL gets joined to v12's paper PnL, and the bus republishes v12's signal as
  production. Nothing errors. This file is the guard for that half.

Both guards derive their file list by SCANNING the tree, not from a hand-written
inventory: a new reader or a new writer added tomorrow is covered on arrival.

Contract: CTR-STRAT-REGISTRY-001 (extends)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.contracts.h5_strategy_identity import (
    H5_CONFLICT_TARGET,
    H5_PRODUCTION_STRATEGY_ID,
    H5_STRATEGY_SCOPED_TABLES,
)

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "database" / "migrations" / "064_h5_strategy_id.sql"

#: Where production code lives. `tmp/` holds throwaway worktrees, `archive/`
#: holds superseded scripts, and neither reaches a database.
SCAN_ROOTS = (
    ROOT / "airflow",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "services",
    ROOT / "usdcop-trading-dashboard" / "app",
    ROOT / "usdcop-trading-dashboard" / "lib",
)
SCAN_SUFFIXES = (".py", ".ts", ".tsx")
EXCLUDED_PARTS = ("node_modules", "__pycache__", "archive", ".next", "tmp")

TABLE_ALTERNATION = "|".join(H5_STRATEGY_SCOPED_TABLES)


def _source_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in SCAN_SUFFIXES or not path.is_file():
                continue
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            files.append(path)
    return sorted(files)


# ---------------------------------------------------------------------------
# String-literal extraction
# ---------------------------------------------------------------------------
# A SQL statement lives inside a string literal. Extract literals (Python triple
# and single quoted, TS template and quoted) WITH their offsets, then merge runs
# that are separated only by whitespace -- Python implicit concatenation splits
# one statement across several literals, and checking them separately would
# report a false failure on a query that is in fact filtered.
_LITERAL_RE = re.compile(
    r'"""(?P<td>.*?)"""'
    r"|'''(?P<ts>.*?)'''"
    r"|`(?P<bt>[^`]*)`"
    r'|"(?P<dq>[^"\n]*)"'
    r"|'(?P<sq>[^'\n]*)'",
    re.DOTALL,
)


def _sql_statements(text: str) -> list[str]:
    """Return string literals, with whitespace-adjacent runs merged."""
    chunks: list[tuple[int, int, str]] = []
    for m in _LITERAL_RE.finditer(text):
        body = next(g for g in m.groups() if g is not None)
        chunks.append((m.start(), m.end(), body))

    merged: list[str] = []
    current = ""
    prev_end: int | None = None
    for start, end, body in chunks:
        if prev_end is not None and text[prev_end:start].strip() == "":
            current += body
        else:
            if current:
                merged.append(current)
            current = body
        prev_end = end
    if current:
        merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# Occurrence model
# ---------------------------------------------------------------------------

_READ_RE = re.compile(rf"\b(?:FROM|JOIN)\s+(?P<table>{TABLE_ALTERNATION})\b", re.IGNORECASE)
#: Prose mentions these tables constantly ("Load this week's execution from
#: forecast_h5_executions"), and `from` reads the same as SQL's FROM. Require an
#: uppercase SELECT in the literal to tell a query from a docstring. SQL in this
#: repo is uniformly uppercase-keyword, so this discriminates without blinding us.
_IS_SQL_RE = re.compile(r"\bSELECT\b")
#: A filter, not a mention. `SELECT strategy_id FROM ...` names the column but
#: still reads every strategy; only a comparison actually narrows the read.
_FILTER_RE = re.compile(r"\bstrategy_id\s*(?:=|\bIN\b)", re.IGNORECASE)
_WRITE_RE = re.compile(
    rf"INSERT\s+INTO\s+(?P<table>{TABLE_ALTERNATION})"
    r"\s*\((?P<columns>[^)]*)\)"
    r"\s*VALUES\s*\((?P<values>.*?)\)\s*"
    r"ON\s+CONFLICT\s*\((?P<target>[^)]*)\)",
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not nested inside parentheses or quotes."""
    items, depth, buf, quote = [], 0, "", None
    for ch in text:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote, buf = ch, buf + ch
        elif ch == "(":
            depth, buf = depth + 1, buf + ch
        elif ch == ")":
            depth, buf = depth - 1, buf + ch
        elif ch == "," and depth == 0:
            items.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        items.append(buf.strip())
    return items


def _collect() -> tuple[list[tuple], list[tuple]]:
    reads: list[tuple[str, str, int, str]] = []
    writes: list[tuple[str, str, int, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(TABLE_ALTERNATION, text):
            continue
        rel = path.relative_to(ROOT).as_posix()

        read_seen: dict[str, int] = {}
        for stmt in _sql_statements(text):
            if not _IS_SQL_RE.search(stmt):
                continue
            for m in _READ_RE.finditer(stmt):
                table = m.group("table").lower()
                read_seen[table] = read_seen.get(table, 0) + 1
                reads.append((rel, table, read_seen[table], stmt))

        write_seen: dict[str, int] = {}
        for m in _WRITE_RE.finditer(text):
            table = m.group("table").lower()
            write_seen[table] = write_seen.get(table, 0) + 1
            cols = tuple(t.strip().lower() for t in m.group("columns").split(",") if t.strip())
            target = tuple(t.strip().lower() for t in m.group("target").split(",") if t.strip())
            values = tuple(_split_top_level(m.group("values")))
            writes.append((rel, table, write_seen[table], cols, target, values))
    return reads, writes


READS, WRITES = _collect()


# ---------------------------------------------------------------------------
# Exemptions -- deliberate, reasoned, and themselves checked for rot
# ---------------------------------------------------------------------------
#: (path, table, occurrence) -> why this read is legitimately strategy-agnostic.
#: Anything not listed here MUST filter. Keep this list short and argued.
READ_EXEMPTIONS: dict[tuple[str, str, int], str] = {
    (
        "airflow/dags/core_watchdog.py",
        "forecast_h5_signals",
        2,
    ): "liveness probe: COUNT(*) answers 'did the table get wiped', which is a "
       "question about the table, not about one strategy",
    (
        "airflow/dags/core_watchdog.py",
        "forecast_h5_executions",
        1,
    ): "same liveness probe, executions counterpart",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_the_scan_actually_found_something() -> None:
    """A scanning test that silently matches nothing proves nothing."""
    assert len(READS) >= 10, f"reader scan collapsed: only {len(READS)} reads found"
    assert len(WRITES) >= 7, f"writer scan collapsed: only {len(WRITES)} upserts found"
    files_with_reads = {r[0] for r in READS}
    assert len(files_with_reads) >= 5, f"reader scan too narrow: {sorted(files_with_reads)}"


def test_canonical_identity_matches_migration_064_default() -> None:
    """The constant is not a free-floating literal: 064's DEFAULT is what the
    entire historical backfill carries, so drifting from it orphans every row."""
    sql = MIGRATION.read_text(encoding="utf-8")
    defaults = set(
        re.findall(r"strategy_id\s+TEXT\s+NOT\s+NULL\s+DEFAULT\s+'([^']+)'", sql, re.IGNORECASE)
    )
    assert defaults == {H5_PRODUCTION_STRATEGY_ID}, (
        f"migration 064 defaults {sorted(defaults)} but the contract says "
        f"{H5_PRODUCTION_STRATEGY_ID}"
    )


def test_canonical_identity_is_known_to_the_strategy_registry() -> None:
    import yaml

    registry = yaml.safe_load(
        (ROOT / "config" / "strategy_registry.yaml").read_text(encoding="utf-8")
    )
    assert H5_PRODUCTION_STRATEGY_ID in registry["strategies"]


@pytest.mark.parametrize(
    "path",
    [
        "services/kafka_bridge/producer.py",
        "usdcop-trading-dashboard/app/api/production/live/route.ts",
        "usdcop-trading-dashboard/app/api/trading/signals/route.ts",
    ],
)
def test_copies_that_cannot_import_the_contract_still_agree_with_it(path: str) -> None:
    """The kafka bridge builds from ./services/kafka_bridge and the dashboard is
    TypeScript -- neither can import the Python contract, so each carries its own
    literal. That is acceptable only while a test pins them together."""
    text = (ROOT / path).read_text(encoding="utf-8")
    literals = set(re.findall(r"""["']((?:smart_simple|forecast_vt|rl)_\w+)["']""", text))
    assert literals == {H5_PRODUCTION_STRATEGY_ID}, (
        f"{path} declares strategy literals {sorted(literals)}; expected only "
        f"{H5_PRODUCTION_STRATEGY_ID}"
    )


@pytest.mark.parametrize(
    "case",
    WRITES,
    ids=lambda c: f"{Path(c[0]).stem}-{c[1]}-{c[2]}",
)
def test_every_h5_upsert_targets_the_composite_identity(case: tuple) -> None:
    """Derived from the tree, so a NEW writer file is covered the day it lands.

    ``ON CONFLICT (signal_date)`` is not a style issue: PostgreSQL raises 42P10
    and the statement writes nothing.
    """
    path, table, occurrence, columns, target, _values = case
    assert "strategy_id" in columns, (
        f"{path}: INSERT #{occurrence} into {table} leaves strategy_id to the "
        "column DEFAULT, so a challenger run would silently stamp itself as production"
    )
    assert target == H5_CONFLICT_TARGET, (
        f"{path}: INSERT #{occurrence} into {table} uses ON CONFLICT {target}; "
        f"migration 064 only provides {H5_CONFLICT_TARGET} -- this raises 42P10 at runtime"
    )


@pytest.mark.parametrize(
    "case",
    READS,
    ids=lambda c: f"{Path(c[0]).stem}-{c[1]}-read{c[2]}",
)
def test_every_h5_read_filters_by_strategy(case: tuple) -> None:
    """The silent half. A read without a strategy filter mixes strategies the
    moment a challenger writes, and nothing raises."""
    path, table, occurrence, stmt = case
    key = (path, table, occurrence)
    if key in READ_EXEMPTIONS:
        pytest.skip(f"exempt: {READ_EXEMPTIONS[key]}")
    assert _FILTER_RE.search(stmt), (
        f"{path}: read #{occurrence} of {table} has no `strategy_id =` / `IN` filter, "
        f"so it will mix strategies once a challenger writes.\nSQL: {' '.join(stmt.split())[:400]}"
    )


@pytest.mark.parametrize(
    "case",
    WRITES,
    ids=lambda c: f"{Path(c[0]).stem}-{c[1]}-{c[2]}",
)
def test_upsert_column_and_value_arity_agree(case: tuple) -> None:
    """Adding `strategy_id` to an INSERT means adding a slot to VALUES and a
    param to the tuple. Miss one and psycopg2 raises at runtime, in a weekly DAG
    nobody watches. Counting is cheap; discovering it on a Monday is not."""
    path, table, occurrence, columns, _target, values = case
    assert len(columns) == len(values), (
        f"{path}: INSERT #{occurrence} into {table} lists {len(columns)} columns "
        f"but {len(values)} VALUES slots.\ncolumns={columns}\nvalues={values}"
    )


def test_read_exemptions_still_point_at_real_reads() -> None:
    """An exemption whose target moved away becomes a silent hole. Fail instead."""
    live = {(r[0], r[1], r[2]) for r in READS}
    stale = sorted(set(READ_EXEMPTIONS) - live)
    assert not stale, f"exemptions no longer match any read (remove them): {stale}"


def test_exempted_reads_are_narrow_by_construction() -> None:
    """An exemption must stay a counting probe. If one grows into a query that
    returns strategy-specific values, it stops being exempt."""
    by_key = {(r[0], r[1], r[2]): r[3] for r in READS}
    for key in READ_EXEMPTIONS:
        stmt = " ".join(by_key[key].split())
        assert re.search(r"SELECT\s+COUNT\s*\(", stmt, re.IGNORECASE), (
            f"{key} is exempt as a counting probe but is no longer a COUNT(): {stmt[:200]}"
        )
