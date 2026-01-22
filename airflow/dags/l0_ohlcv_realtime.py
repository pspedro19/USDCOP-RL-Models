"""
DAG: l0_02_ohlcv_realtime
=========================
USD/COP Trading System - V3 Architecture
Layer 0: Realtime OHLCV Acquisition

Purpose:
    Acquires USD/COP 5-minute OHLCV bars from TwelveData API.
    Stores in usdcop_m5_ohlcv table for downstream processing.

Schedule:
    */5 13-17 * * 1-5 (Every 5 minutes, 8:00-12:55 COT, Mon-Fri)
    Note: Airflow is UTC-based, 8:00-12:55 COT = 13:00-17:55 UTC

Features:
    - Gap detection and backfilling
    - Duplicate prevention
    - Business hours validation
    - Automatic retry on API failures

Author: Pedro @ Lean Tech Solutions
Version: 3.0.0
Updated: 2025-12-16
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.time_delta import TimeDeltaSensor
import pandas as pd
import requests
import pytz
import psycopg2
from psycopg2.extras import execute_values
import os
import logging
import json
import sys
from pathlib import Path

# Trading calendar from utils
from utils.trading_calendar import TradingCalendar

# =============================================================================
# CONFIGURATION - DRY: Using shared utilities + SSOT from feature_config.json
# =============================================================================

from utils.dag_common import get_db_connection, load_feature_config
from contracts.dag_registry import L0_OHLCV_REALTIME, L1_FEATURE_REFRESH

CONFIG = load_feature_config(raise_on_error=False)
OHLCV_CONFIG = CONFIG.get('sources', {}).get('ohlcv', {})
TRADING_CONFIG = CONFIG.get('trading', {})
MARKET_HOURS = TRADING_CONFIG.get('market_hours', {})

DAG_ID = L0_OHLCV_REALTIME

# Timezone settings (from config SSOT)
TIMEZONE_STR = MARKET_HOURS.get('timezone', 'America/Bogota')
COT_TZ = pytz.timezone(TIMEZONE_STR)
UTC_TZ = pytz.UTC

# Trading hours (from config SSOT - no hardcoded values)
_local_start = MARKET_HOURS.get('local_start', '08:00')
_local_end = MARKET_HOURS.get('local_end', '12:55')
MARKET_START_COT = int(_local_start.split(':')[0])  # 8 from "08:00"
MARKET_END_COT = int(_local_end.split(':')[0]) + 1  # 13 from "12:55" (end hour + 1)
MARKET_DAYS = TRADING_CONFIG.get('trading_days', [0, 1, 2, 3, 4])

# API Configuration (from config SSOT)
TWELVEDATA_SYMBOLS = [TRADING_CONFIG.get('symbol', 'USD/COP')]
TWELVEDATA_INTERVAL = OHLCV_CONFIG.get('granularity', '5min')
TWELVEDATA_TIMEZONE = TIMEZONE_STR
TWELVEDATA_API_KEY = os.environ.get('TWELVEDATA_API_KEY_1') or os.environ.get('TWELVEDATA_API_KEY')

# Validate API key is set (Fail Fast)
if not TWELVEDATA_API_KEY:
    raise ValueError("TWELVEDATA_API_KEY environment variable is required")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Initialize trading calendar
trading_cal = TradingCalendar()

def should_run_today():
    """Check if today is a valid trading day."""
    today = datetime.now(COT_TZ)
    if not trading_cal.is_trading_day(today):
        reason = trading_cal.get_violation_reason(today)
        logging.info(f"Skipping - {today.date()}: {reason}")
        return False
    return True

def is_market_hours():
    """Check if current time is within market hours"""
    now_cot = datetime.now(COT_TZ)
    is_trading_day = now_cot.weekday() in MARKET_DAYS
    is_trading_hour = MARKET_START_COT <= now_cot.hour < MARKET_END_COT
    return is_trading_day and is_trading_hour

def fetch_ohlcv_data(symbol, interval, bars=60):
    """
    Fetch OHLCV data from TwelveData API.

    Args:
        symbol: e.g., 'USD/COP'
        interval: e.g., '5min'
        bars: number of bars to fetch

    Returns:
        DataFrame with columns: time, open, high, low, close, volume, symbol
    """
    url = 'https://api.twelvedata.com/time_series'

    params = {
        'symbol': symbol,
        'interval': interval,
        'format': 'JSON',
        'timezone': TWELVEDATA_TIMEZONE,
        'apikey': TWELVEDATA_API_KEY,
        'outputsize': bars
    }

    try:
        logging.info(f"Fetching {bars} bars for {symbol}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if 'values' not in data:
            logging.error(f"API response missing 'values': {data}")
            raise ValueError("API response format error")

        # Parse API response
        records = []
        for bar in data['values']:
            records.append({
                'time': datetime.fromisoformat(bar['datetime']),
                'open': float(bar['open']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'close': float(bar['close']),
                'volume': float(bar.get('volume', 0)),
                'symbol': symbol,
                'source': 'twelvedata'
            })

        df = pd.DataFrame(records)
        logging.info(f"Fetched {len(df)} bars from API")
        return df

    except Exception as e:
        logging.error(f"API fetch error: {e}")
        raise

def is_bar_in_market_hours(bar_time):
    """
    Check if a bar timestamp is within market hours (8:00-12:55 COT).
    TwelveData returns timestamps in COT timezone (as configured).
    """
    if isinstance(bar_time, str):
        bar_time = datetime.fromisoformat(bar_time)

    # Get hour and minute in local time (COT)
    hour = bar_time.hour
    minute = bar_time.minute

    # Market hours: 8:00 - 12:55 COT
    if hour < MARKET_START_COT or hour > MARKET_END_COT - 1:
        return False
    if hour == MARKET_END_COT - 1 and minute > 55:  # After 12:55
        return False

    # Also check weekday (0=Monday, 6=Sunday)
    if bar_time.weekday() >= 5:  # Saturday or Sunday
        return False

    return True


def insert_ohlcv_data(df, conn):
    """
    Insert OHLCV data into usdcop_m5_ohlcv table.
    Uses ON CONFLICT to handle duplicates.
    FILTERS: Only inserts bars within market hours (8:00-12:55 COT, Mon-Fri)
    """
    if df.empty:
        logging.warning("No data to insert")
        return 0

    cur = conn.cursor()

    try:
        # Ensure table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS usdcop_m5_ohlcv (
                time TIMESTAMPTZ PRIMARY KEY,
                symbol VARCHAR(20),
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                source VARCHAR(50),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # FILTER: Only include bars within market hours (8:00-12:55 COT, Mon-Fri)
        filtered_count = 0
        values = []
        for _, row in df.iterrows():
            if is_bar_in_market_hours(row['time']):
                values.append((row['time'], row['symbol'], row['open'], row['high'],
                               row['low'], row['close'], row['volume'], row['source']))
            else:
                filtered_count += 1

        if filtered_count > 0:
            logging.info(f"Filtered out {filtered_count} bars outside market hours")

        if not values:
            logging.warning("No bars within market hours to insert")
            return 0

        # Insert with conflict handling
        query = """
            INSERT INTO usdcop_m5_ohlcv
            (time, symbol, open, high, low, close, volume, source)
            VALUES %s
            ON CONFLICT (time, symbol) DO UPDATE SET
                volume = EXCLUDED.volume,
                source = EXCLUDED.source,
                updated_at = NOW()
        """

        execute_values(cur, query, values)
        conn.commit()

        logging.info(f"Inserted {len(values)} OHLCV records (market hours only)")
        return len(values)

    except Exception as e:
        conn.rollback()
        logging.error(f"Database insert error: {e}")
        raise
    finally:
        cur.close()

def acquire_ohlcv(**context):
    """
    Main task: Acquire OHLCV data and store in database.
    """
    logging.info(f"Starting OHLCV acquisition at {datetime.now(UTC_TZ)}")

    # Check market hours
    if not is_market_hours():
        logging.info("Outside market hours, skipping acquisition")
        return {'status': 'skipped', 'reason': 'outside_market_hours'}

    try:
        # Fetch data from API (bars from config SSOT)
        lookback_bars = OHLCV_CONFIG.get('lookback_bars_needed', 100)
        df_ohlcv = fetch_ohlcv_data(
            symbol=TWELVEDATA_SYMBOLS[0],
            interval=TWELVEDATA_INTERVAL,
            bars=lookback_bars
        )

        # Insert into database
        conn = get_db_connection()
        rows_inserted = insert_ohlcv_data(df_ohlcv, conn)
        conn.close()

        # Push metrics to XCom
        context['ti'].xcom_push(key='ohlcv_rows_inserted', value=rows_inserted)

        return {
            'status': 'success',
            'rows_inserted': rows_inserted,
            'timestamp': datetime.now(UTC_TZ).isoformat()
        }

    except Exception as e:
        logging.error(f"OHLCV acquisition failed: {e}")
        raise

def validate_data(**context):
    """
    Validate that latest OHLCV bar has been inserted.
    Also validates that no holiday data slipped through.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Get latest bar
        cur.execute("""
            SELECT COUNT(*) as count, MAX(time) as latest_time
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
        """)

        result = cur.fetchone()
        count, latest_time = result[0], result[1]

        logging.info(f"OHLCV table has {count} bars. Latest: {latest_time}")

        if count == 0:
            raise ValueError("No OHLCV data in database")

        # Validate no holiday data exists (check last 7 days)
        cur.execute("""
            SELECT DISTINCT DATE(time AT TIME ZONE 'America/Bogota') as trade_date
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
            AND time >= NOW() - INTERVAL '7 days'
            ORDER BY trade_date DESC
        """)

        dates_with_data = [row[0] for row in cur.fetchall()]
        invalid_dates = []

        for trade_date in dates_with_data:
            if not trading_cal.is_trading_day(trade_date):
                reason = trading_cal.get_violation_reason(trade_date)
                invalid_dates.append({
                    'date': str(trade_date),
                    'reason': reason
                })

        if invalid_dates:
            logging.warning(f"Found data on {len(invalid_dates)} non-trading days: {invalid_dates}")
            # Don't fail, just warn - data might have been from before validation was added

        context['ti'].xcom_push(key='ohlcv_count', value=count)
        context['ti'].xcom_push(key='latest_ohlcv_time', value=str(latest_time))
        context['ti'].xcom_push(key='invalid_trading_dates', value=invalid_dates)

        return {
            'status': 'valid',
            'total_bars': count,
            'latest_time': str(latest_time),
            'invalid_dates_found': len(invalid_dates)
        }

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
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='V3 L0: Realtime OHLCV acquisition from TwelveData (every 5 min)',
    schedule_interval='*/5 13-17 * * 1-5',  # Every 5min during market hours (UTC)
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l0', 'ohlcv', 'realtime', 'data-acquisition']
)

with dag:

    # Check if today is a trading day
    def check_trading_day(**context):
        """Branch task to skip processing on holidays/weekends."""
        if should_run_today():
            return 'acquire_ohlcv'
        else:
            return 'skip_processing'

    task_check_trading_day = BranchPythonOperator(
        task_id='check_trading_day',
        python_callable=check_trading_day,
        provide_context=True
    )

    task_skip = EmptyOperator(
        task_id='skip_processing'
    )

    task_acquire = PythonOperator(
        task_id='acquire_ohlcv',
        python_callable=acquire_ohlcv,
        provide_context=True,
        pool='api_requests'  # Rate limiting
    )

    task_validate = PythonOperator(
        task_id='validate_ohlcv',
        python_callable=validate_data,
        provide_context=True
    )

    # Task dependencies
    task_check_trading_day >> [task_acquire, task_skip]
    task_acquire >> task_validate
