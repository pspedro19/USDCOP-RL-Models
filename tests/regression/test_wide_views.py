"""Wide-view contract: closed != missing, PIT-safe macro lags, no corrupted series.

Contract: CTR-MKT-CANON-001 (migrations 060/061/062, plan 2026-07-21)

The row that motivated the whole layer: 2026-07-20 (Colombian Independence Day) must read
usdcop='closed' while btcusdt='ok' — a holiday must never masquerade as a data incident.
The daily-anchoring test pins governance invariant #1 (a 00:00-UTC daily stamp dated in
the closing tz shifts every bar a day back — caught live: SPX 07-20 read as 07-19).

DB-backed tests skip when postgres is unreachable (CI has no DB); the SQL-text tests always
run, so the contract's structural guarantees are enforced everywhere.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MIG = ROOT / "database" / "migrations"

SQL_060 = (MIG / "060_market_canonical_views.sql").read_text(encoding="utf-8")
SQL_061 = (MIG / "061_market_wide_views.sql").read_text(encoding="utf-8")
SQL_062 = (MIG / "062_macro_monthly_views.sql").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Structural (always run)
# ---------------------------------------------------------------------------

def test_repaired_fx_series_are_served_with_t1_form():
    """USDMXN/USDCLP were excluded while corrupted x10^4; reinstated 2026-07-22 after the
    twelvedata repair (111 rows fixed + 147 gaps filled; investing.com es-locale was the
    corruptor and now 403s — SSOT primary flipped to twelvedata).

    The daily wide must serve BOTH forms (as-of + _t1); the scale guard that keeps them
    honest is test_macro_clean_fx_scale, now a HARD test (xfail lifted).
    """
    code = "\n".join(line.split("--")[0] for line in SQL_061.splitlines()).lower()
    for col in ("macro_usdmxn", "macro_usdmxn_t1", "macro_usdclp", "macro_usdclp_t1"):
        assert col in code, f"061: {col} missing from daily wide"


def test_daily_bars_are_anchored_in_utc():
    """Governance invariant #1: date a daily bar in UTC, never in the closing tz."""
    m = re.search(r"CREATE OR REPLACE VIEW market_ohlcv_daily AS.*?FROM asset_daily_ohlcv",
                  SQL_060, re.S)
    assert m, "060 lost the market_ohlcv_daily view"
    assert "AT TIME ZONE 'UTC')::date" in m.group(0), (
        "daily session_date_local must be UTC-anchored; converting a 00:00-UTC stamp to "
        "the session tz shifts every bar one day back (verified live on SPX 2026-07-20)")
    assert "AT TIME ZONE da.session_tz)::date" not in m.group(0)


def test_staleness_is_computed_never_materialized():
    """A stored staleness lies by construction; wide views must compute it from now()."""
    assert re.search(r"EXTRACT\(epoch FROM now\(\)", SQL_061)
    for stmt in re.findall(r"CREATE MATERIALIZED VIEW.*?;", SQL_060, re.S):
        assert "staleness" not in stmt.lower(), "materialized staleness is frozen at refresh"


def test_monthly_view_exposes_conservative_availability():
    """Monthly is where PIT bites hardest (June CPI unknowable on June 30).

    The model-side join key must be COALESCE(publication_date, conservative bound) —
    never the reference month itself.
    """
    for view in ("market_macro_monthly_wide", "market_macro_quarterly_wide"):
        body = SQL_062[SQL_062.index(view):]
        assert "COALESCE(m.publication_date" in body, f"{view}: no conservative bound"
        assert "interval '3 months'" in body


def test_wide_views_are_views_not_tables():
    """Wide = VIEW (plan decision): closed-market NULLs are never stored, a new asset is
    an ALTER VIEW, and staleness stays live. Only the 1h/4h AGGREGATES may materialize."""
    assert "CREATE TABLE" not in SQL_061 and "CREATE TABLE" not in SQL_062
    for bad in re.findall(r"CREATE MATERIALIZED VIEW \S+", SQL_061 + SQL_062):
        pytest.fail(f"wide layer must not materialize: {bad}")


# ---------------------------------------------------------------------------
# DB-backed (skip without postgres)
# ---------------------------------------------------------------------------

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
        pytest.skip(f"postgres unreachable ({e.__class__.__name__}) — DB-backed wide-view "
                    "checks run only where the stack is up")


@pytest.fixture(scope="module")
def db():
    conn = _conn()
    yield conn
    conn.close()


def test_july_20_reads_closed_not_missing(db):
    """The founding case: the holiday that cost a morning of diagnosis."""
    cur = db.cursor()
    cur.execute("""
        SELECT status_usdcop, status_btcusdt, status_spx500, status_xauusd
        FROM market_ohlcv_daily_wide WHERE session_date_cot = '2026-07-20'""")
    row = cur.fetchone()
    assert row is not None, "2026-07-20 row absent from daily wide view"
    usdcop, btc, spx, xau = row
    assert usdcop == "closed", f"Colombian holiday must read closed, got {usdcop!r}"
    assert btc == "ok", f"BTC trades 24/7, got {btc!r}"
    assert spx == "ok" and xau == "ok", f"NYSE/metals were open: spx={spx!r} xau={xau!r}"


def test_weekend_is_closed_and_today_is_pending_never_missing(db):
    cur = db.cursor()
    cur.execute("""
        SELECT session_date_cot, status_usdcop, status_spx500, status_btcusdt
        FROM market_ohlcv_daily_wide
        WHERE extract(dow FROM session_date_cot) IN (0, 6)
          AND session_date_cot >= '2026-01-01'""")
    for d, cop, spx, _btc in cur.fetchall():
        assert cop == "closed" and spx == "closed", f"weekend {d}: cop={cop} spx={spx}"
    cur.execute("""
        SELECT count(*) FROM market_ohlcv_daily_wide
        WHERE session_date_cot >= (now() AT TIME ZONE 'UTC')::date
          AND 'missing' IN (status_usdcop, status_xauusd, status_btcusdt, status_spx500)""")
    assert cur.fetchone()[0] == 0, (
        "a bar for today-or-future can never be 'missing' — it is 'pending': the session "
        "has not produced a closed daily bar yet")


def test_monthly_t1_is_really_last_months_value(db):
    """t1 of month M == as-of of month M-1, for every consecutive pair (PIT lag is real)."""
    cur = db.cursor()
    cur.execute("""
        WITH x AS (
          SELECT month_start, macro_cpi_usa, macro_cpi_usa_t1,
                 LAG(macro_cpi_usa)   OVER (ORDER BY month_start) AS prev_val,
                 LAG(month_start)     OVER (ORDER BY month_start) AS prev_month
          FROM market_macro_monthly_wide)
        SELECT count(*) FROM x
        WHERE prev_month = (month_start - interval '1 month')::date
          AND macro_cpi_usa_t1 IS DISTINCT FROM prev_val""")
    assert cur.fetchone()[0] == 0, "macro_cpi_usa_t1 != previous month's as-of value"


def test_monthly_rows_are_month_start_unified(db):
    """The MASTER mixed anchors (US month-start, COL month-end) splitting each month in
    two rows; the seeder must have unified them."""
    cur = db.cursor()
    cur.execute("""
        SELECT count(*) FROM macro_indicators_monthly
        WHERE fecha <> date_trunc('month', fecha)::date""")
    assert cur.fetchone()[0] == 0, "month-end anchored strays present — reseed"
    cur.execute("""
        SELECT count(*) FROM market_macro_monthly_wide
        WHERE macro_cpi_usa IS NOT NULL AND macro_ipc_col IS NOT NULL""")
    assert cur.fetchone()[0] > 50, "US and Colombian series never share a row — anchors split"
