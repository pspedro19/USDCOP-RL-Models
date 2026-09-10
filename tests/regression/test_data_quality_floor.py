"""Data-quality FLOOR: the 2026-07-22 remediation becomes a ratchet, not a snapshot.

Contract: CTR-MKT-CANON-001 / CTR-DQ-MACRO-001 (quality plan Fases 1-6)

Each floor below was MEASURED after the remediation (column_audit + scorecard). A future
regression — a series going dark again, coherence breaking, a wide view inventing rows —
must fail CI instead of waiting for a human to rerun the audit. Floors are set slightly
under the measured values so normal jitter passes but real degradation does not.

DB-backed; skips cleanly where the stack is down (structural cousins live in
test_wide_views.py). The monthly-macro freshness floor is calendar-aware: series may lag
their declared publication schedules, never more.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.usefixtures()


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
        pytest.skip(f"postgres unreachable ({e.__class__.__name__})")


@pytest.fixture(scope="module")
def db():
    conn = _conn()
    conn.cursor().execute("SET max_parallel_workers_per_gather = 0")
    yield conn
    conn.close()


def test_ohlc_coherence_is_absolute(db):
    """0 incoherent bars across every physical price table — measured 0/2.55M."""
    cur = db.cursor()
    for table in ("usdcop_m5_ohlcv", "asset_native_ohlcv", "asset_daily_ohlcv"):
        cur.execute(f"""SELECT count(*) FROM {table}
            WHERE NOT (high >= GREATEST(open, close) AND low <= LEAST(open, close)
                       AND high >= low)""")
        bad = cur.fetchone()[0]
        assert bad == 0, f"{table}: {bad} incoherent OHLC bars (was 0 at remediation)"


def test_monthly_macro_no_dark_tail(db):
    """The Oct-2025 blackout must not repeat: every live monthly series must have a value
    within its publication lag of the newest month in the table.

    Slow publishers (trade ~7wk, EOF) get 3 months of slack; the rest 2. TOT is excluded:
    its only source is Selenium-SUAMECA, documented pending (the one open item)."""
    cur = db.cursor()
    slack = {"ftrd_exports_total_col_m_expusd": 3, "ftrd_imports_total_col_m_impusd": 3,
             "infl_exp_eof_col_m_infexp": 3, "crsk_sentiment_ici_col_m_ici": 3}
    series = ["polr_fed_funds_usa_m_fedfunds", "infl_cpi_all_usa_m_cpiaucsl",
              "infl_cpi_core_usa_m_cpilfesl", "infl_pce_usa_m_pcepi",
              "labr_unemployment_usa_m_unrate", "prod_industrial_usa_m_indpro",
              "mnys_m2_supply_usa_m_m2sl", "sent_consumer_usa_m_umcsent",
              "infl_cpi_total_col_m_ipccol", "fxrt_reer_bilateral_col_m_itcr",
              "fxrt_reer_bilateral_usa_col_m_itcr_usa",
              "rsbp_reserves_international_col_m_resint",
              "crsk_sentiment_cci_col_m_cci", "crsk_sentiment_ici_col_m_ici",
              "ftrd_exports_total_col_m_expusd", "ftrd_imports_total_col_m_impusd",
              "infl_exp_eof_col_m_infexp"]
    cur.execute("SELECT max(fecha) FROM macro_indicators_monthly")
    newest = cur.fetchone()[0]
    dark = []
    for s in series:
        cur.execute(f"SELECT max(fecha) FROM macro_indicators_monthly WHERE {s} IS NOT NULL")
        last = cur.fetchone()[0]
        allowed = slack.get(s, 2)
        months_behind = ((newest.year - last.year) * 12 + newest.month - last.month) if last else 99
        if months_behind > allowed:
            dark.append(f"{s} ({months_behind}m behind)")
    assert not dark, f"monthly series going dark again: {dark}"


def test_fx_repair_holds_in_db(db):
    """MXN/CLP x10^4 corruption stays dead in the DB (parquet twin: test_macro_clean_fx_scale)."""
    cur = db.cursor()
    cur.execute("""SELECT count(*) FROM macro_indicators_daily
        WHERE fxrt_spot_usdmxn_mex_d_usdmxn > 1000 OR fxrt_spot_usdclp_chl_d_usdclp > 20000""")
    assert cur.fetchone()[0] == 0, "scale-corrupted MXN/CLP rows are back"


# ---------------------------------------------------------------------------
# RESUELTO 2026-08-25 — se conserva el registro porque el sintoma era desconcertante.
#
# Estos cuatro tests fallaban sin que hubiera NADA mal en los datos: la primera consulta
# mataba al servidor Postgres y las tres siguientes caian en cascada, porque la fixture
# `db` es de modulo y la conexion ya estaba cerrada.
#
#     psycopg2.OperationalError: server closed the connection unexpectedly
#     psycopg2.InterfaceError: connection already closed   (x3)
#
#     docker logs usdcop-postgres-timescale:
#       LOG: server process (PID ...) was terminated by signal 9: Killed
#
# Causa raiz: `asset_daily_ohlcv` tenia **2.430 chunks de 7 dias** (mas uno de 943) para
# 60.542 filas — 25 filas por chunk. El hypertable se creo con el `chunk_time_interval`
# por defecto de TimescaleDB y el backfill de historia completa (1979-2026) escribio
# decadas con el; cambiar el intervalo despues no reescribe los chunks existentes. Un
# GROUP BY que los recorria agotaba la memoria del contenedor y el OOM killer se llevaba
# el backend.
#
# Arreglado con `scripts/ops/fix_daily_hypertable_chunks.py`: **2.431 chunks -> 6**, con
# respaldo CSV verificado de las 60.542 filas y las 3 vistas dependientes recreadas desde
# sus definiciones capturadas. La consulta de abajo pasa de matar al servidor a tardar
# 0,7 s.
#
# La leccion, por si vuelve a aparecer en otra tabla: **un modo de fallo que parece de
# conectividad puede ser de layout de chunks**. `SELECT (range_end - range_start), count(*)
# FROM timescaledb_information.chunks GROUP BY 1` lo diagnostica en un segundo.
# ---------------------------------------------------------------------------


def test_no_duplicate_daily_dates(db):
    """One date = one daily bar per symbol, whatever hour each feed stamps.

    Feeds stamp inconsistently (seed 21:00/22:00 UTC, backfill 00:00) so the (time,symbol)
    PK cannot enforce this; caught live when the catch-up duplicated XAU 2026-07-21 and the
    doubled join broke wide-no-invention. Writers now dedupe by UTC date; this is the teeth.
    """
    cur = db.cursor()
    cur.execute("""SELECT symbol, count(*) FROM (
        SELECT symbol, (time AT TIME ZONE 'UTC')::date d FROM asset_daily_ohlcv
        GROUP BY 1, 2 HAVING count(*) > 1) x GROUP BY 1""")
    dups = cur.fetchall()
    assert not dups, f"duplicate daily dates per symbol: {dups}"


def test_wide_views_do_not_invent_data(db):
    """A wide view may only re-shape: every non-NULL close it serves must exist in the base
    long table — counts must match exactly per asset."""
    cur = db.cursor()
    for wide_col, asset in (("close_usdcop", "usdcop"), ("close_xauusd", "xauusd"),
                            ("close_btcusdt", "btcusdt"), ("close_spx500", "spx500")):
        # ONE statement -> one snapshot: the hourly catch-up DAG ingests live, and two
        # separate counts can straddle an insert (caught as a phantom +1 on 2026-07-22).
        cur.execute(f"""SELECT
            (SELECT count({wide_col}) FROM market_ohlcv_daily_wide),
            (SELECT count(*) FROM market_ohlcv_daily WHERE asset_id=%s)""", (asset,))
        served, base = cur.fetchone()
        assert served == base, (
            f"daily wide serves {served} {wide_col} bars but the base has {base} — "
            "a wide view must never invent (or drop) data")


def test_m5_session_completeness_floor(db):
    """COP in-session M5 completeness >= 96% (measured 96.8% pre gap-fill; the residual
    missing list is provider-permanent, documented)."""
    cur = db.cursor()
    cur.execute("""
        WITH sesion AS (
          SELECT count(*) * 60 AS esperado FROM market_session_calendar
          WHERE asset_id='usdcop' AND is_trading_day
            AND session_date BETWEEN '2020-01-02' AND current_date - 1),
        real AS (
          SELECT count(*) AS n FROM usdcop_m5_ohlcv m
          JOIN market_session_calendar c ON c.asset_id='usdcop'
            AND c.session_date = (m.time AT TIME ZONE 'America/Bogota')::date
            AND c.is_trading_day
          WHERE m.symbol='USD/COP'
            AND m.time >= c.session_open_utc AND m.time <= c.session_close_utc)
        SELECT real.n::float / sesion.esperado FROM sesion, real""")
    ratio = cur.fetchone()[0]
    assert ratio >= 0.96, f"COP M5 in-session completeness {ratio:.1%} < 96% floor"


def test_pit_stamping_is_universal_for_new_rows(db):
    """Every row ingested after the PIT wiring (2026-07-21) must carry available_at."""
    cur = db.cursor()
    for table, ts_col in (("usdcop_m5_ohlcv", "created_at"),
                          ("asset_native_ohlcv", "created_at"),
                          ("asset_daily_ohlcv", "updated_at")):  # daily has no created_at
        cur.execute(f"""SELECT count(*) FROM {table}
            WHERE {ts_col} >= '2026-07-21 16:00+00' AND available_at IS NULL""")
        bad = cur.fetchone()[0]
        assert bad == 0, f"{table}: {bad} post-wiring rows missing available_at"
