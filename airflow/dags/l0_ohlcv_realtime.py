"""
DAG: core_l0_02_ohlcv_realtime
==============================
USD/COP Trading System - L0 Consolidated Realtime OHLCV Acquisition

Purpose:
    Acquires 5-minute OHLCV bars for USD/COP, USD/MXN, and USD/BRL from
    TwelveData API. All timestamps stored in America/Bogota (COT).
    Stores in usdcop_m5_ohlcv table (multi-symbol via symbol column).

    Consolidates:
    - l0_ohlcv_realtime.py (COP only, with TradingCalendar + CircuitBreaker)
    - l0_ohlcv_realtime_multi.py (3 pairs, BRL tz handling)

Schedule:
    */5 13-17 * * 1-5 (Every 5 minutes, 8:00-12:55 COT, Mon-Fri)
    Note: Airflow is UTC-based, 8:00-12:55 COT = 13:00-17:55 UTC

Timezone handling per pair:
    - USD/COP: TwelveData timezone=America/Bogota -> COT natively
    - USD/MXN: TwelveData timezone=America/Bogota -> COT natively
    - USD/BRL: TwelveData timezone=UTC -> convert to COT before DB insert
      (BRL returns incomplete data with timezone=America/Bogota)

Features:
    - TradingCalendar holiday validation (skip holidays)
    - CircuitBreaker per-symbol API protection
    - Parallel tasks per symbol (failure isolation)
    - API key rotation across symbols
    - UPSERT on (time, symbol) for idempotent inserts

Author: Pedro @ Lean Tech Solutions
Version: 4.0.0
Created: 2026-02-12
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import pandas as pd
import requests
import pytz
from psycopg2.extras import execute_values
import os
import logging

# =============================================================================
# CONFIGURATION
# =============================================================================

from utils.dag_common import get_db_connection, load_feature_config
from utils.trading_calendar import TradingCalendar
from utils.circuit_breaker import get_circuit_breaker, CircuitOpenError
from contracts.dag_registry import CORE_L0_OHLCV_REALTIME

CONFIG = load_feature_config(raise_on_error=False)
TRADING_CONFIG = CONFIG.get('trading', {})
OHLCV_CONFIG = CONFIG.get('sources', {}).get('ohlcv', {})
MARKET_HOURS = TRADING_CONFIG.get('market_hours', {})

DAG_ID = CORE_L0_OHLCV_REALTIME

# Timezone
TIMEZONE_STR = MARKET_HOURS.get('timezone', 'America/Bogota')
COT_TZ = pytz.timezone(TIMEZONE_STR)
UTC_TZ = pytz.UTC

# Trading hours
_local_start = MARKET_HOURS.get('local_start', '08:00')
_local_end = MARKET_HOURS.get('local_end', '12:55')
MARKET_START_COT = int(_local_start.split(':')[0])
MARKET_END_COT = int(_local_end.split(':')[0]) + 1  # 13 for "12:55"
MARKET_DAYS = TRADING_CONFIG.get('trading_days', [0, 1, 2, 3, 4])

# Symbol configuration (api_tz, needs_tz_convert)
SYMBOL_CONFIG = {
    'USD/COP': {'api_tz': 'America/Bogota', 'needs_tz_convert': False},
    'USD/MXN': {'api_tz': 'America/Bogota', 'needs_tz_convert': False},
    'USD/BRL': {'api_tz': 'UTC', 'needs_tz_convert': True},
}

# TwelveData API keys (rotation pool)
TWELVEDATA_API_KEYS = [
    os.environ.get(f'TWELVEDATA_API_KEY_{i}') for i in range(1, 9)
] + [os.environ.get('TWELVEDATA_API_KEY')]
TWELVEDATA_API_KEYS = [k for k in TWELVEDATA_API_KEYS if k]

_api_key_index = 0

# Trading calendar for holiday checks
trading_cal = TradingCalendar()


# =============================================================================
# HELPERS
# =============================================================================

def get_next_api_key() -> str:
    """Rotate through API keys."""
    global _api_key_index
    if not TWELVEDATA_API_KEYS:
        return None
    key = TWELVEDATA_API_KEYS[_api_key_index % len(TWELVEDATA_API_KEYS)]
    _api_key_index += 1
    return key


def is_market_hours_now() -> bool:
    """Check if current COT time is within market hours."""
    now_cot = datetime.now(COT_TZ)
    return (
        now_cot.weekday() in MARKET_DAYS
        and MARKET_START_COT <= now_cot.hour < MARKET_END_COT
    )


def is_bar_in_session(bar_time: datetime) -> bool:
    """Check if a bar timestamp (in COT) is within 8:00-12:55 Mon-Fri."""
    if bar_time.weekday() >= 5:
        return False
    hour = bar_time.hour
    minute = bar_time.minute
    if hour < MARKET_START_COT or hour > (MARKET_END_COT - 1):
        return False
    if hour == (MARKET_END_COT - 1) and minute > 55:
        return False
    return True


def should_run_today() -> bool:
    """Check if today is a valid trading day (not weekend/holiday)."""
    today = datetime.now(COT_TZ)
    if not trading_cal.is_trading_day(today):
        reason = trading_cal.get_violation_reason(today)
        logging.info(f"Skipping - {today.date()}: {reason}")
        return False
    return True


# =============================================================================
# MAIN TASK: Fetch + store for one symbol
# =============================================================================

def fetch_and_store_symbol(symbol: str, **context):
    """
    Fetch OHLCV for a single symbol and store in DB.

    Uses CircuitBreaker for API resilience.
    Handles timezone conversion for BRL (UTC -> COT).
    """
    cfg = SYMBOL_CONFIG[symbol]
    api_key = get_next_api_key()
    if not api_key:
        raise ValueError(f"No TWELVEDATA_API_KEY configured for {symbol}")

    # Skip if outside market hours
    if not is_market_hours_now():
        logging.info(f"[{symbol}] Outside market hours, skipping")
        return {'status': 'skipped', 'reason': 'outside_market_hours'}

    # Circuit breaker protection
    cb = get_circuit_breaker(f'twelvedata_realtime_{symbol.replace("/", "_").lower()}')
    if not cb.can_execute():
        logging.warning(f"[{symbol}] Circuit breaker OPEN - skipping")
        raise CircuitOpenError(f"Circuit breaker OPEN for {symbol}")

    # Fetch from TwelveData
    url = 'https://api.twelvedata.com/time_series'
    lookback = OHLCV_CONFIG.get('lookback_bars_needed', 60)
    params = {
        'symbol': symbol,
        'interval': '5min',
        'format': 'JSON',
        'timezone': cfg['api_tz'],
        'apikey': api_key,
        'outputsize': lookback,
    }

    try:
        logging.info(f"[{symbol}] Fetching {lookback} bars (tz={cfg['api_tz']})")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if 'values' not in data:
            logging.warning(f"[{symbol}] No data: {data.get('message', 'unknown')}")
            cb.record_failure(ValueError("No data returned"))
            return {'status': 'no_data', 'symbol': symbol}

        # Parse bars
        records = []
        for bar in data['values']:
            bar_time = datetime.fromisoformat(bar['datetime'])

            # BRL: convert UTC -> COT
            if cfg['needs_tz_convert']:
                if bar_time.tzinfo is None:
                    bar_time = UTC_TZ.localize(bar_time)
                bar_time = bar_time.astimezone(COT_TZ)
            else:
                if bar_time.tzinfo is None:
                    bar_time = COT_TZ.localize(bar_time)

            records.append({
                'time': bar_time,
                'symbol': symbol,
                'open': float(bar['open']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'close': float(bar['close']),
                'volume': float(bar.get('volume', 0)),
                'source': 'twelvedata_multi',
            })

        df = pd.DataFrame(records)
        logging.info(f"[{symbol}] Fetched {len(df)} bars")

        # Filter to market hours only
        df_filtered = df[df['time'].apply(
            lambda t: is_bar_in_session(t if t.tzinfo else COT_TZ.localize(t))
        )]
        n_dropped = len(df) - len(df_filtered)
        if n_dropped > 0:
            logging.info(f"[{symbol}] Filtered {n_dropped} off-session bars")

        if df_filtered.empty:
            cb.record_success()
            return {'status': 'empty_after_filter', 'symbol': symbol}

        # Insert to DB
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            values = [
                (row['time'], row['symbol'], row['open'], row['high'],
                 row['low'], row['close'], row['volume'], row['source'])
                for _, row in df_filtered.iterrows()
            ]

            execute_values(
                cur,
                """
                INSERT INTO usdcop_m5_ohlcv
                    (time, symbol, open, high, low, close, volume, source)
                VALUES %s
                ON CONFLICT (time, symbol) DO UPDATE SET
                    volume = EXCLUDED.volume,
                    source = EXCLUDED.source,
                    updated_at = NOW()
                """,
                values,
            )
            conn.commit()
            cur.close()
            logging.info(f"[{symbol}] Inserted/updated {len(values)} bars")

            cb.record_success()
            return {'status': 'success', 'symbol': symbol, 'rows': len(values)}

        except Exception as e:
            conn.rollback()
            logging.error(f"[{symbol}] DB insert failed: {e}")
            cb.record_failure(e)
            raise
        finally:
            conn.close()

    except CircuitOpenError:
        raise
    except Exception as e:
        logging.error(f"[{symbol}] Fetch error: {e}")
        cb.record_failure(e)
        raise


def validate_data(**context):
    """Validate that latest OHLCV bars have been inserted for all symbols."""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        for symbol in SYMBOL_CONFIG:
            cur.execute("""
                SELECT COUNT(*), MAX(time)
                FROM usdcop_m5_ohlcv WHERE symbol = %s
            """, (symbol,))
            count, latest = cur.fetchone()
            logging.info(f"[{symbol}] {count} bars, latest: {latest}")

        # Check for holiday data in last 7 days
        cur.execute("""
            SELECT DISTINCT DATE(time AT TIME ZONE 'America/Bogota') as trade_date
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP' AND time >= NOW() - INTERVAL '7 days'
            ORDER BY trade_date DESC
        """)
        dates_with_data = [row[0] for row in cur.fetchall()]
        invalid_dates = []
        for trade_date in dates_with_data:
            if not trading_cal.is_trading_day(trade_date):
                reason = trading_cal.get_violation_reason(trade_date)
                invalid_dates.append({'date': str(trade_date), 'reason': reason})

        if invalid_dates:
            logging.warning(f"Data on {len(invalid_dates)} non-trading days: {invalid_dates}")

        return {'status': 'valid', 'invalid_dates_found': len(invalid_dates)}

    finally:
        cur.close()
        conn.close()


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='L0: Multi-pair realtime OHLCV (COP/MXN/BRL) with holiday + circuit breaker protection',
    schedule_interval='*/5 13-17 * * 1-5',
    catchup=False,
    max_active_runs=1,
    tags=['core', 'l0', 'ohlcv', 'realtime', 'multi-pair'],
)

with dag:

    # Check if today is a trading day (holidays/weekends)
    def check_trading_day(**context):
        if should_run_today():
            return 'start_fetch'
        return 'skip_processing'

    task_check = BranchPythonOperator(
        task_id='check_trading_day',
        python_callable=check_trading_day,
    )

    task_skip = EmptyOperator(task_id='skip_processing')

    task_start = EmptyOperator(task_id='start_fetch')

    # One task per symbol, running in parallel
    symbol_tasks = []
    for symbol in SYMBOL_CONFIG:
        task_id = f"fetch_{symbol.replace('/', '_').lower()}"
        task = PythonOperator(
            task_id=task_id,
            python_callable=fetch_and_store_symbol,
            op_kwargs={'symbol': symbol},
            pool='api_requests',
        )
        symbol_tasks.append(task)

    task_validate = PythonOperator(
        task_id='validate_ohlcv',
        python_callable=validate_data,
        trigger_rule='none_failed_min_one_success',
    )

    task_end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success',
    )

    # Flow: check_day -> [start -> [fetch_cop, fetch_mxn, fetch_brl] -> validate -> end | skip -> end]
    task_check >> [task_start, task_skip]
    task_start >> symbol_tasks >> task_validate >> task_end
    task_skip >> task_end
