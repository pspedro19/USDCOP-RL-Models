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

2. DB tables ``forecast_h5_*`` (xfail STRICT until BL-42 phase 2): the future convention
   demands no ``_pct`` column whose |median| < 0.5 — a return/stop column with a sub-0.5
   median is a decimal in disguise (verified live 2026-07-27: week_pnl_pct
   median_abs=0.0045, hard_stop_pct=0.027, take_profit_pct=0.013). When phase 2 lands the
   XPASS turns red (strict) and the marker must be deleted in the same commit. Postgres
   down ⇒ skip, unless ``BL42_REQUIRE_DB=1`` (stack environments) makes that a failure —
   a silent skip must never read as green where the DB is supposed to exist.

Remediation CXD-012: xfail strict + BL42_REQUIRE_DB, decimal-disguise family detector on
the published JSONs (0.01606 under a ``_pct`` family now fails), and strategy_signal
coverage: the signal table's suffixless return columns are pinned to DECIMAL scale so
``ensemble_return`` can't silently flip to pct points while ``*_pct`` migrates.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
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


def _pct_families(data) -> dict:
    """Group finite non-zero ``*_pct`` values by their terminal key name.

    week_pnl_pct appearing 30 times across trades forms one family — unit bugs are
    per-exporter-column, so the family (not the lone value) is the honest unit witness.
    """
    families: dict[str, list[float]] = {}
    for field, value in _iter_pct_fields(data):
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value) or value == 0:
            continue
        key = field.rsplit(".", 1)[-1].split("[")[0]
        families.setdefault(key, []).append(float(value))
    return families


def _decimal_disguise_offenders(families: dict) -> list[str]:
    """A ``*_pct`` family whose |median| < 0.5 AND |max| < 1.0 is a decimal in disguise.

    Same 0.5 prior as the DB layer (declared ex-ante in BL-42, not tuned on outcomes);
    the |max| < 1.0 guard keeps genuinely-small pct families (one +1.6% week among
    noise) out: a real pct return column crosses 1 somewhere, a decimal one does not.
    """
    offenders = []
    for key, values in families.items():
        if len(values) < MIN_ROWS_FOR_MEDIAN:
            continue
        ordered = sorted(abs(v) for v in values)
        median_abs = ordered[len(ordered) // 2]
        if median_abs < 0.5 and ordered[-1] < 1.0:
            offenders.append(
                f"{key}: median_abs={median_abs:.4g} max_abs={ordered[-1]:.4g} "
                f"(n={len(values)})")
    return offenders


@pytest.mark.parametrize(
    "path", SUMMARY_FILES + [LEDGER_FILE], ids=lambda p: p.name)
def test_pct_families_are_not_decimals_in_disguise(path: Path):
    """CXD-012: |v| <= 100 alone accepts 0.01606 under ``_pct``. The family-median
    detector closes that hole for every published bundle."""
    offenders = _decimal_disguise_offenders(_pct_families(_load(path)))
    assert not offenders, (
        f"{path.name}: *_pct families that are decimals in disguise: {offenders}")


def test_decimal_disguise_detector_catches_live_db_signature():
    """Fail-first witness: the exact values Codex caught live (ensemble_return=0.01606,
    week_pnl_pct median 0.0045) MUST trip the detector; an honest pct family must not."""
    red = {"trades": [{"week_pnl_pct": v} for v in (0.01606, 0.0045, -0.027, 0.013)]}
    offenders = _decimal_disguise_offenders(_pct_families(red))
    assert offenders and offenders[0].startswith("week_pnl_pct"), (
        "detector failed to flag the live decimal-in-disguise signature — the BL-42 "
        f"candado is vacuous: {offenders}")

    green = {"trades": [{"week_pnl_pct": v} for v in (1.606, 0.45, -2.7, 1.3)]}
    assert not _decimal_disguise_offenders(_pct_families(green)), (
        "detector flags genuine percentage points — false positive")


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
# Layer 3 — the PRODUCER, not the artifact it already wrote
# ---------------------------------------------------------------------------
#
# Layer 1 reads JSONs that are already committed, so it guards the DATA and leaves the
# EXPORTER unguarded. Verified by mutation: turning
#
#     "total_return_pct": round(total_return, 2)
#   → "total_return_pct": round(total_return / 100.0, 6)
#
# in scripts/pipeline/train_and_export_smart_simple.py::_compute_result_metrics —
# literally the bug BL-42 exists to forbid — left the whole suite green, because no
# test ever CALLED the function that computes the number. These do: a synthetic ledger
# of known compounded return goes in, and the units of what comes out (in memory, before
# any JSON exists) are pinned.

from scripts.pipeline.train_and_export_smart_simple import (  # noqa: E402
    _compute_result_metrics,
)

# 24 trades (>= MIN_TRADES_FOR_STATS = 20, quant-constitution §6 — below that neither
# Sharpe nor p-value may be reported). The expected return is closed-form and derived
# OUTSIDE the code under test: prod(1 + p/100) over _LEDGER_PNL_PCT = 1.14461621…
_LEDGER_PNL_PCT = (
    [1.8, -1.2, 2.4, 0.9, -0.6, 1.5, -2.1, 3.0, 0.6, -0.9, 1.2, 2.7]
    + [1.8, -1.2, 2.4, 0.9, -0.6, 1.5, -2.1, 3.0, 0.6, -0.9, 1.2, -2.04]
)
EXPECTED_FINAL_EQUITY = 11446.16          # 10_000 · prod(1 + p/100)
EXPECTED_TOTAL_RETURN_PCT = 14.46         # PERCENTAGE POINTS — never 0.1446


def _synthetic_ledger(scale: float = 1.0):
    """Trade ledger + the equity it compounds to. Scale stretches the same shape."""
    equity = DEFAULT_INITIAL_CAPITAL
    trades = []
    for i, base in enumerate(_LEDGER_PNL_PCT):
        pnl_pct = round(base * scale, 6)
        start, equity = equity, equity * (1.0 + pnl_pct / 100.0)
        trades.append({
            "pnl_pct": pnl_pct,
            "pnl_usd": round(equity - start, 6),
            "side": "LONG" if i % 2 == 0 else "SHORT",
            "exit_reason": "take_profit" if pnl_pct > 0 else "hard_stop",
        })
    return trades, equity


def _synthetic_prices(year: int = 2025):
    """Weekly closes +5% over the year — feeds the buy&hold leg of the metrics."""
    return pd.DataFrame({
        "date": pd.date_range(f"{year}-01-06", periods=52, freq="W-MON"),
        "close": np.linspace(4000.0, 4200.0, 52),
    })


def test_producer_emits_percentage_points_not_a_decimal():
    # rojo con: `round(total_return, 2)` -> `round(total_return / 100.0, 6)` en
    # scripts/pipeline/train_and_export_smart_simple.py:1052
    trades, equity = _synthetic_ledger()
    metrics = _compute_result_metrics(trades, equity, _synthetic_prices(), 2025)["metrics"]

    assert metrics["final_equity"] == pytest.approx(EXPECTED_FINAL_EQUITY, abs=0.01)
    assert metrics["total_return_pct"] == pytest.approx(
        EXPECTED_TOTAL_RETURN_PCT, abs=0.01), (
        f"total_return_pct={metrics['total_return_pct']} for a ledger that compounds "
        f"10_000 -> {EXPECTED_FINAL_EQUITY}: expected {EXPECTED_TOTAL_RETURN_PCT} "
        "PERCENTAGE POINTS (a 0.1446 here is the BL-42 bug at the source)")

    # Same equity<->pct identity layer 1 applies to the written JSON, applied to the
    # in-memory dict: a decimal disguised as pct misses the implied return by ~99%.
    assert metrics["final_equity"] / DEFAULT_INITIAL_CAPITAL - 1.0 == pytest.approx(
        metrics["total_return_pct"] / 100.0, abs=EQUITY_PCT_TOLERANCE)


def test_producer_output_survives_the_decimal_disguise_detector():
    # rojo con: la MISMA mutación /100.0 — la familia total_return_pct cae a
    # median_abs<0.5 y max_abs<1.0, la firma exacta que el detector persigue
    runs = [
        _compute_result_metrics(*_synthetic_ledger(scale), _synthetic_prices(), 2025)["metrics"]
        for scale in (1.0, 0.25, -0.6, 1.4)
    ]
    assert [r["n_trades"] for r in runs] == [len(_LEDGER_PNL_PCT)] * 4

    offenders = _decimal_disguise_offenders(_pct_families({"runs": runs}))
    assert not offenders, (
        f"the exporter's own output is a decimal in disguise: {offenders}")

    # …and the detector is NOT vacuous on this surface: the same output with
    # total_return_pct divided by 100 (the mutation) MUST be caught.
    disguised = [dict(m, total_return_pct=m["total_return_pct"] / 100.0) for m in runs]
    caught = _decimal_disguise_offenders(_pct_families({"runs": disguised}))
    assert any(o.startswith("total_return_pct") for o in caught), (
        "detector blind to the producer-side decimal disguise — this gate would be "
        f"vacuous: {caught}")


# ---------------------------------------------------------------------------
# Layer 2 — DB convention (xfail until BL-42 phase 2 migrates the data)
# ---------------------------------------------------------------------------

MIN_ROWS_FOR_MEDIAN = 3  # below this a median says nothing (constitution §6 spirit)

# CXD-012: a silent skip must never read as green where the DB is supposed to exist.
# Stack/CI environments with postgres set BL42_REQUIRE_DB=1 to turn unavailability red.
REQUIRE_DB = os.environ.get("BL42_REQUIRE_DB") == "1"


def _db_unavailable(msg: str):
    if REQUIRE_DB:
        pytest.fail(f"BL42_REQUIRE_DB=1 but {msg}")
    pytest.skip(f"{msg} — DB unit-convention check runs only where the stack is up")


def _conn():
    try:
        import psycopg2
    except ImportError:
        _db_unavailable("psycopg2 not installed")
    try:
        return psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
            user=os.environ.get("POSTGRES_USER", "admin"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            connect_timeout=3)
    except Exception as e:  # noqa: BLE001
        _db_unavailable(f"postgres unreachable ({e.__class__.__name__})")


def test_db_available_when_required():
    """BL42_REQUIRE_DB=1 canary. The strict-xfail test below swallows ANY failure
    (including our _db_unavailable fail) as 'expected' — so unavailability must turn
    red HERE, outside the xfail, or the requirement is vacuous."""
    if not REQUIRE_DB:
        pytest.skip("BL42_REQUIRE_DB not set — advisory mode, DB tests may skip")
    _conn().close()


@pytest.mark.xfail(strict=True,
                   reason="BL-42 fase 2: DB usa mezcla decimal/pct (week_pnl_pct, "
                          "hard_stop_pct, take_profit_pct almacenan decimales). "
                          "strict: cuando fase 2 migre, el XPASS rompe y este marker "
                          "se borra en el mismo commit")
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


# Suffixless return/stop columns on the SIGNAL surface (BL-42: strategy_signal). These
# store decimals TODAY (ensemble_return=0.01606 = 1.606%) and the target convention is
# also decimal — so this passes now and pins the scale: an exporter flipping
# ensemble_return to pct points would corrupt sizing silently while all *_pct gates look.
SIGNAL_RETURN_COLUMN_HINTS = ("return", "ret_", "pnl", "stop", "profit", "drawdown")


def test_signal_suffixless_return_columns_stay_decimal():
    """strategy_signal coverage (CXD-012): numeric return-like columns WITHOUT ``_pct``
    in forecast_h5_signals must hold decimal scale (|median| < 0.5 over non-null
    non-zero rows) — the mirror invariant of the ``*_pct`` check."""
    conn = _conn()
    try:
        from psycopg2 import sql

        cur = conn.cursor()
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'forecast_h5_signals'
              AND column_name NOT LIKE '%\\_pct'
              AND data_type IN ('numeric', 'double precision', 'real')
            ORDER BY column_name""")
        candidates = [
            c for (c,) in cur.fetchall()
            if any(h in c for h in SIGNAL_RETURN_COLUMN_HINTS)]
        if not candidates:
            pytest.skip("forecast_h5_signals has no suffixless return-like numeric "
                        "columns (renamed to *_decimal already?)")

        offenders = []
        for column in candidates:
            cur.execute(sql.SQL(
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY abs({col})), "
                "count(*) FROM forecast_h5_signals "
                "WHERE {col} IS NOT NULL AND {col} <> 0"
            ).format(col=sql.Identifier(column)))
            median_abs, n = cur.fetchone()
            if n < MIN_ROWS_FOR_MEDIAN or median_abs is None:
                continue
            if median_abs >= 0.5:
                offenders.append(
                    f"forecast_h5_signals.{column} median_abs={median_abs:.4g} (n={n}) "
                    "— pct points in a suffixless column")
        assert not offenders, (
            "signal surface broke the decimal convention: " + "; ".join(offenders))
    finally:
        conn.close()
