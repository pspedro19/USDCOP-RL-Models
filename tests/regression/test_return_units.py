"""Return-unit convention: every ``*_pct`` field means PERCENTAGE POINTS, never decimals.

Backlog: BL-42 (.claude/specs/planes/backlog/BL-42-unidades-decimales-signal-normalizada.md)

The bug this guards: forecast_h5_predictions.predicted_return_pct=1.6481 (pct points) vs
forecast_h5_signals.ensemble_return=0.01606 (decimal) are the SAME 1.606% — a ``_pct``
suffix over a decimal is a communication bug waiting for capital (BL-42 nota constitución).

Two layers, test-first (data migration is BL-42 phase 2):

1. PUBLISHED JSONs (always run, DB-agnostic): the dashboard bundles under
   ``public/data/production/`` already follow the pct-points convention — pin it so no
   exporter regresses to decimals. Heuristic: |value| <= 100, plus a coherence check
   equity_final/initial - 1 ~= total_return_pct/100 whenever both fields are present.

2. DB tables ``forecast_h5_*`` (xfail until BL-42 phase 2): the future convention demands
   no ``_pct`` column whose |median| < 0.5 — a return/stop column with a sub-0.5 median is
   a decimal in disguise (verified live 2026-07-27: week_pnl_pct median_abs=0.0045,
   hard_stop_pct=0.027, take_profit_pct=0.013). Skips cleanly when postgres is down.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "usdcop-trading-dashboard" / "public" / "data" / "production"

SUMMARY_FILES = sorted(PROD.glob("summary*.json"))
LEDGER_FILE = PROD / "paper" / "candidates_ledger_2026.json"

DEFAULT_INITIAL_CAPITAL = 10_000.0
# 1% relative-to-capital tolerance on the decimal scale (rounding of published pct).
EQUITY_PCT_TOLERANCE = 0.01


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_pct_fields(node, path=""):
    """Yield (json_path, value) for every scalar under a key ending in ``_pct``.

    Recurses into dicts and lists; list-valued ``*_pct`` keys (e.g. monthly.pnl_pct)
    yield one entry per element.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            if key.endswith("_pct"):
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        yield f"{child}[{i}]", item
                elif not isinstance(value, dict):
                    yield child, value
                else:
                    yield from _iter_pct_fields(value, child)
            else:
                yield from _iter_pct_fields(value, child)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _iter_pct_fields(item, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# Layer 1 — published JSONs (always run)
# ---------------------------------------------------------------------------

def test_published_json_files_exist():
    """A fresh clone must carry the bundles (git-tracking policy 2026-07-09)."""
    assert SUMMARY_FILES, f"no summary*.json under {PROD}"
    assert LEDGER_FILE.is_file(), f"missing {LEDGER_FILE}"


@pytest.mark.parametrize(
    "path", SUMMARY_FILES + [LEDGER_FILE], ids=lambda p: p.name)
def test_pct_fields_are_percentage_points(path: Path):
    """|*_pct| <= 100: a 3.36% return published as 336 (or 0.0336 pretending to be
    basis-agnostic) would slip through no other gate before the dashboard renders it."""
    offenders = []
    for field, value in _iter_pct_fields(_load(path)):
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue  # null is legal (strategy-contract: safe_json_dump)
        if not math.isfinite(value):
            offenders.append(f"{field}={value!r} (non-finite — strategy-contract violation)")
        elif abs(value) > 100:
            offenders.append(f"{field}={value}")
    assert not offenders, (
        f"{path.name}: *_pct fields out of percentage-point range (|v|<=100): {offenders}")


@pytest.mark.parametrize("path", SUMMARY_FILES, ids=lambda p: p.name)
def test_summary_equity_coheres_with_total_return_pct(path: Path):
    """final_equity/initial - 1 must equal total_return_pct/100 (tolerance 1%).

    This is the check that distinguishes pct-points from decimals outright: a decimal
    0.0336 stored under total_return_pct would miss the equity implied return by ~97%.
    """
    data = _load(path)
    initial = float(data.get("initial_capital") or DEFAULT_INITIAL_CAPITAL)
    checked = 0
    for sid, strat in (data.get("strategies") or {}).items():
        if not isinstance(strat, dict):
            continue
        equity, pct = strat.get("final_equity"), strat.get("total_return_pct")
        if equity is None or pct is None:
            continue  # e.g. gold buy_and_hold publishes pct only
        implied = float(equity) / initial - 1.0
        declared = float(pct) / 100.0
        assert abs(implied - declared) <= EQUITY_PCT_TOLERANCE, (
            f"{path.name} strategies[{sid}]: final_equity={equity} implies "
            f"{implied:+.4%} but total_return_pct={pct} declares {declared:+.4%} — "
            "either the equity is wrong or *_pct is not in percentage points")
        checked += 1
    if checked == 0:
        pytest.skip(f"{path.name}: no strategy publishes both final_equity and "
                    "total_return_pct")


def test_ledger_trade_equity_coheres_with_ytd_pct():
    """candidates_ledger: the last trade's running equity must reproduce ret_2026_ytd_pct
    (same $10K base as the summaries). Guards the paper ledger the operator reads for
    the v12/v14 judge decision."""
    data = _load(LEDGER_FILE)
    checked = 0
    for sid, strat in (data.get("strategies") or {}).items():
        trades = strat.get("trades") or []
        ytd_pct = strat.get("ret_2026_ytd_pct")
        if not trades or ytd_pct is None:
            continue
        last_equity = trades[-1].get("equity")
        if last_equity is None:
            continue
        implied = float(last_equity) / DEFAULT_INITIAL_CAPITAL - 1.0
        declared = float(ytd_pct) / 100.0
        assert abs(implied - declared) <= EQUITY_PCT_TOLERANCE, (
            f"candidates_ledger strategies[{sid}]: last trade equity={last_equity} "
            f"implies {implied:+.4%} but ret_2026_ytd_pct={ytd_pct} declares "
            f"{declared:+.4%}")
        checked += 1
    assert checked > 0, "ledger has no strategy with trades+equity+ret_2026_ytd_pct"


# ---------------------------------------------------------------------------
# Layer 2 — DB convention (xfail until BL-42 phase 2 migrates the data)
# ---------------------------------------------------------------------------

MIN_ROWS_FOR_MEDIAN = 3  # below this a median says nothing (constitution §6 spirit)


def _conn():
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        return psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
            user=os.environ.get("POSTGRES_USER", "admin"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            connect_timeout=3)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"postgres unreachable ({e.__class__.__name__}) — DB unit-convention "
                    "check runs only where the stack is up")


@pytest.mark.xfail(strict=False,
                   reason="BL-42 fase 2: DB usa mezcla decimal/pct (week_pnl_pct, "
                          "hard_stop_pct, take_profit_pct almacenan decimales)")
def test_forecast_h5_pct_columns_hold_percentage_points():
    """Future convention (BL-42): no ``_pct`` column in forecast_h5_* may have
    |median| < 0.5 over its non-null non-zero rows — a return/stop stored as 0.0269
    under hard_stop_pct is a decimal wearing a pct suffix.

    Once phase 2 renames to *_decimal (DB stores decimals, frontend formats pct),
    this test flips: either the columns die or their data becomes true pct points.
    """
    conn = _conn()
    try:
        from psycopg2 import sql

        cur = conn.cursor()
        cur.execute("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name LIKE 'forecast\\_h5\\_%'
              AND column_name LIKE '%\\_pct'
              AND data_type IN ('numeric', 'double precision', 'real',
                                'integer', 'bigint', 'smallint')
            ORDER BY table_name, column_name""")
        columns = cur.fetchall()
        if not columns:
            pytest.skip("no numeric *_pct columns in forecast_h5_* (already migrated?)")

        offenders = []
        for table, column in columns:
            cur.execute(sql.SQL(
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY abs({col})), "
                "count(*) FROM {tbl} WHERE {col} IS NOT NULL AND {col} <> 0"
            ).format(col=sql.Identifier(column), tbl=sql.Identifier(table)))
            median_abs, n = cur.fetchone()
            if n < MIN_ROWS_FOR_MEDIAN or median_abs is None:
                continue
            if median_abs < 0.5:
                offenders.append(f"{table}.{column} median_abs={median_abs:.4g} (n={n})")

        assert not offenders, (
            "decimal-in-disguise *_pct columns (|median|<0.5): " + "; ".join(offenders)
            + " — BL-42 phase 2 must migrate these to *_decimal or true pct points")
    finally:
        conn.close()
