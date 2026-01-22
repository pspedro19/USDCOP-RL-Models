"""
DAG: l0_03_ohlcv_backfill
=========================
USD/COP Trading System - V3 Architecture
Layer 0: OHLCV COMPREHENSIVE Gap Detection and Automatic Backfill

Purpose:
    Detects ALL gaps in OHLCV data from MIN to MAX date and backfills from TwelveData API.
    Runs on system startup when backup data is outdated, or can be triggered manually.

UPDATED (2026-01-08): Now detects INTERNAL gaps, not just last timestamp to NOW.
    - Scans from MIN date to MAX date in database
    - Identifies ALL missing trading days
    - Backfills each gap range individually
    - Excludes weekends and Colombian holidays

Schedule:
    @once (manual trigger or startup)

Features:
    - COMPREHENSIVE gap detection from MIN to MAX date
    - Internal gap detection (finds holes in the middle of data)
    - Smart backfill only during market hours (8:00-12:55 COT, Mon-Fri)
    - Holiday awareness (uses holidays from feature_config.json)
    - Batch processing to respect API rate limits
    - Detailed gap reporting

Author: Pedro @ Lean Tech Solutions
Version: 3.1.0 (Updated for internal gap detection)
Created: 2025-12-17
Updated: 2026-01-08
"""

from datetime import datetime, timedelta, date
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import pandas as pd
import requests
import pytz
import psycopg2
from psycopg2.extras import execute_values
import os
import logging
import time
from typing import List, Dict, Tuple

# =============================================================================
# CONFIGURATION - DRY: Using shared utilities + SSOT from feature_config.json
# =============================================================================

from utils.dag_common import get_db_connection, load_feature_config
from contracts.dag_registry import L0_OHLCV_BACKFILL

CONFIG = load_feature_config(raise_on_error=False)
OHLCV_CONFIG = CONFIG.get('sources', {}).get('ohlcv', {})
TRADING_CONFIG = CONFIG.get('trading', {})
MARKET_HOURS = TRADING_CONFIG.get('market_hours', {})

DAG_ID = L0_OHLCV_BACKFILL

# Timezone settings (from config SSOT)
TIMEZONE_STR = MARKET_HOURS.get('timezone', 'America/Bogota')
COT_TZ = pytz.timezone(TIMEZONE_STR)
UTC_TZ = pytz.UTC

# Trading hours (from config SSOT - no hardcoded values)
_local_start = MARKET_HOURS.get('local_start', '08:00')
_local_end = MARKET_HOURS.get('local_end', '12:55')
MARKET_START_HOUR = int(_local_start.split(':')[0])  # 8
MARKET_START_MINUTE = int(_local_start.split(':')[1])  # 0
MARKET_END_HOUR = int(_local_end.split(':')[0])  # 12
MARKET_END_MINUTE = int(_local_end.split(':')[1])  # 55
MARKET_DAYS = TRADING_CONFIG.get('trading_days', [0, 1, 2, 3, 4])
BARS_PER_SESSION = TRADING_CONFIG.get('bars_per_session', 60)

# Holidays from config (SSOT)
HOLIDAYS_STR = CONFIG.get('holidays_2025_colombia', [])
HOLIDAYS = {datetime.strptime(d, '%Y-%m-%d').date() for d in HOLIDAYS_STR}

# Comprehensive holidays list - Colombia + US market holidays (TwelveData has no data on these)
# This ensures gap detection doesn't flag these days as missing
ADDITIONAL_HOLIDAYS = [
    # 2020 - US Market holidays + Colombia
    '2020-01-01', '2020-01-20', '2020-02-17', '2020-04-10', '2020-05-25',
    '2020-07-03', '2020-09-07', '2020-11-26', '2020-12-25',
    # 2021 - US Market holidays + Colombia
    '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31',
    '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24', '2021-12-31',
    # 2022 - US Market holidays + Colombia
    '2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30', '2022-06-20',
    '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-26',
    # 2023 - US Market holidays + Colombia
    '2023-01-02', '2023-01-16', '2023-02-20', '2023-04-07', '2023-05-29',
    '2023-06-19', '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25',
    # 2024 - US Market holidays + Colombia
    '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27',
    '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-24', '2024-12-25', '2024-12-31',
    # 2025 - Colombian holidays (add any missing from config)
    '2025-01-01', '2025-01-06', '2025-03-24', '2025-04-17', '2025-04-18',
    '2025-05-01', '2025-06-02', '2025-06-23', '2025-06-30', '2025-07-20',
    '2025-08-07', '2025-08-18', '2025-10-13', '2025-11-03', '2025-11-17',
    '2025-12-08', '2025-12-24', '2025-12-25', '2025-12-31',
    # 2026 - Colombian holidays + US Market
    '2026-01-01', '2026-01-12', '2026-01-19', '2026-02-16', '2026-03-23',
    '2026-04-02', '2026-04-03', '2026-05-01', '2026-05-18', '2026-05-25',
    '2026-06-08', '2026-06-15', '2026-06-19', '2026-06-29', '2026-07-03',
    '2026-07-20', '2026-08-07', '2026-08-17', '2026-09-07', '2026-10-12',
    '2026-11-02', '2026-11-16', '2026-11-26', '2026-12-08', '2026-12-24', '2026-12-25',
]
for d in ADDITIONAL_HOLIDAYS:
    try:
        HOLIDAYS.add(datetime.strptime(d, '%Y-%m-%d').date())
    except:
        pass

# API Configuration (from config SSOT)
TWELVEDATA_SYMBOL = TRADING_CONFIG.get('symbol', 'USD/COP')
TWELVEDATA_INTERVAL = OHLCV_CONFIG.get('granularity', '5min')
TWELVEDATA_TIMEZONE = TIMEZONE_STR

# TwelveData API keys - rotate through them to avoid rate limiting
TWELVEDATA_API_KEYS = [
    os.environ.get('TWELVEDATA_API_KEY_1'),
    os.environ.get('TWELVEDATA_API_KEY_2'),
    os.environ.get('TWELVEDATA_API_KEY_3'),
    os.environ.get('TWELVEDATA_API_KEY_4'),
    os.environ.get('TWELVEDATA_API_KEY_5'),
    os.environ.get('TWELVEDATA_API_KEY_6'),
    os.environ.get('TWELVEDATA_API_KEY_7'),
    os.environ.get('TWELVEDATA_API_KEY_8'),
    os.environ.get('TWELVEDATA_API_KEY'),
]
TWELVEDATA_API_KEYS = [k for k in TWELVEDATA_API_KEYS if k]  # Filter None values
_api_key_index = 0

def get_next_api_key() -> str:
    """Rotate through API keys to avoid rate limiting."""
    global _api_key_index
    if not TWELVEDATA_API_KEYS:
        return None
    key = TWELVEDATA_API_KEYS[_api_key_index % len(TWELVEDATA_API_KEYS)]
    _api_key_index += 1
    return key

# For backward compatibility
TWELVEDATA_API_KEY = TWELVEDATA_API_KEYS[0] if TWELVEDATA_API_KEYS else None

# Backfill settings
MAX_BARS_PER_REQUEST = 5000  # TwelveData API limit
API_RATE_DELAY_SECONDS = 1  # Reduced delay since we rotate keys
MAX_BACKFILL_DAYS = 365  # Maximum days to backfill (safety limit)

# Gap detection settings
# Note: TwelveData has incomplete data for some historical dates (especially 2021)
# Use a lower threshold to avoid flagging days with limited but existing data
MIN_BARS_THRESHOLD = 10  # Minimum bars per day to NOT be considered a gap

# Validate API keys
if not TWELVEDATA_API_KEYS:
    logging.warning("No TWELVEDATA_API_KEY environment variables are set - backfill will fail")
else:
    logging.info(f"Loaded {len(TWELVEDATA_API_KEYS)} TwelveData API keys for rotation")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_trading_day(d: date) -> bool:
    """Check if a date is a trading day (Mon-Fri, not a holiday)."""
    return d.weekday() in MARKET_DAYS and d not in HOLIDAYS


def get_all_trading_days(start_date: date, end_date: date) -> List[date]:
    """Get all trading days between two dates (inclusive)."""
    trading_days = []
    current = start_date
    while current <= end_date:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)
    return trading_days


def get_market_hours_for_date(d: date):
    """Get market open/close times for a specific date in COT timezone."""
    open_time = COT_TZ.localize(datetime(d.year, d.month, d.day, MARKET_START_HOUR, MARKET_START_MINUTE))
    close_time = COT_TZ.localize(datetime(d.year, d.month, d.day, MARKET_END_HOUR, MARKET_END_MINUTE))
    return open_time, close_time


def get_data_date_range(conn) -> Tuple[date, date]:
    """Get MIN and MAX dates from the OHLCV table."""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT DATE(MIN(time)), DATE(MAX(time))
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
        """)
        result = cur.fetchone()
        if result and result[0] and result[1]:
            return result[0], result[1]
        return None, None
    finally:
        cur.close()


def get_bars_per_day(conn) -> Dict[date, int]:
    """Get count of bars per trading day (within market hours only)."""
    cur = conn.cursor()
    try:
        # Handle both timestamp formats (UTC and COT-as-UTC)
        # OLD FORMAT: 13:00-17:55 UTC = 8:00-12:55 COT
        # NEW FORMAT: 08:00-12:55 stored with +00 offset
        cur.execute("""
            SELECT DATE(time) as trading_date, COUNT(*) as bar_count
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
              AND EXTRACT(DOW FROM time) BETWEEN 1 AND 5
              AND (
                  (time::time >= '13:00:00'::time AND time::time <= '17:55:00'::time)
                  OR
                  (time::time >= '08:00:00'::time AND time::time <= '12:55:00'::time)
              )
            GROUP BY DATE(time)
            ORDER BY DATE(time)
        """)

        return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        cur.close()


def get_last_ohlcv_timestamp(conn) -> datetime:
    """Get the most recent timestamp from usdcop_m5_ohlcv table."""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT MAX(time) FROM usdcop_m5_ohlcv WHERE symbol = 'USD/COP'
        """)
        result = cur.fetchone()[0]
        if result:
            # Ensure timezone-aware
            if result.tzinfo is None:
                result = COT_TZ.localize(result)
            return result
        return None
    finally:
        cur.close()


def get_current_market_time() -> datetime:
    """Get current time in market timezone."""
    return datetime.now(COT_TZ)


def detect_all_gaps(trading_days: List[date], bars_per_day: Dict[date, int]) -> List[Dict]:
    """
    Detect ALL gaps in the data from MIN to MAX date.

    A gap is:
    - A trading day with 0 bars
    - A trading day with significantly fewer bars than expected (< MIN_BARS_THRESHOLD)

    Returns list of gap dictionaries
    """
    gaps = []
    expected_bars = BARS_PER_SESSION  # 60 bars per session

    for day in trading_days:
        actual_bars = bars_per_day.get(day, 0)

        if actual_bars == 0:
            gaps.append({
                'date': day,
                'expected_bars': expected_bars,
                'actual_bars': 0,
                'missing_bars': expected_bars,
                'gap_type': 'FULL_DAY_MISSING'
            })
        elif actual_bars < MIN_BARS_THRESHOLD:
            gaps.append({
                'date': day,
                'expected_bars': expected_bars,
                'actual_bars': actual_bars,
                'missing_bars': expected_bars - actual_bars,
                'gap_type': 'PARTIAL_DAY'
            })

    return gaps


def group_consecutive_gaps(gaps: List[Dict]) -> List[Dict]:
    """
    Group consecutive gap days into ranges for more efficient backfill.
    """
    if not gaps:
        return []

    # Sort by date
    gaps = sorted(gaps, key=lambda x: x['date'])

    ranges = []
    current_range = {
        'start_date': gaps[0]['date'],
        'end_date': gaps[0]['date'],
        'days_missing': 1,
        'total_bars_missing': gaps[0]['missing_bars'],
        'gap_days': [gaps[0]]
    }

    for gap in gaps[1:]:
        prev_date = current_range['end_date']
        current_date = gap['date']

        # Check if consecutive (allowing for weekends/holidays in between)
        days_diff = (current_date - prev_date).days

        # Find number of trading days between prev and current
        trading_days_between = len(get_all_trading_days(prev_date + timedelta(days=1), current_date - timedelta(days=1)))

        # If no trading days between (consecutive gap) or within 3 days
        if trading_days_between == 0 or days_diff <= 3:
            current_range['end_date'] = current_date
            current_range['days_missing'] += 1
            current_range['total_bars_missing'] += gap['missing_bars']
            current_range['gap_days'].append(gap)
        else:
            # Start new range
            ranges.append(current_range)
            current_range = {
                'start_date': gap['date'],
                'end_date': gap['date'],
                'days_missing': 1,
                'total_bars_missing': gap['missing_bars'],
                'gap_days': [gap]
            }

    ranges.append(current_range)
    return ranges


def calculate_expected_bars(start_dt: datetime, end_dt: datetime) -> int:
    """
    Calculate expected number of 5-min bars between two timestamps,
    considering only market hours on trading days.
    """
    total_bars = 0
    current_date = start_dt.date()
    end_date = end_dt.date()

    while current_date <= end_date:
        if is_trading_day(current_date):
            market_open, market_close = get_market_hours_for_date(current_date)

            # Calculate overlap with our time range
            period_start = max(start_dt.astimezone(COT_TZ), market_open)
            period_end = min(end_dt.astimezone(COT_TZ), market_close)

            if period_start < period_end:
                # Count 5-minute bars in this period
                minutes = (period_end - period_start).total_seconds() / 60
                total_bars += int(minutes / 5)

        current_date += timedelta(days=1)

    return total_bars


def fetch_ohlcv_data(symbol: str, interval: str, start_date: str, end_date: str, bars: int = None) -> pd.DataFrame:
    """
    Fetch OHLCV data from TwelveData API for a specific date range.

    Args:
        symbol: e.g., 'USD/COP'
        interval: e.g., '5min'
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format
        bars: Optional max number of bars

    Returns:
        DataFrame with columns: time, open, high, low, close, volume, symbol, source

    NOTE: TwelveData requires end_date to be at least 1 day after start_date.
    If they are the same, we automatically add 1 day to end_date.
    """
    api_key = get_next_api_key()
    if not api_key:
        logging.error("No TWELVEDATA_API_KEY configured")
        return pd.DataFrame()

    # TwelveData API quirk: end_date must be > start_date
    # If same day, add 1 day to end_date to get data for that day
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    if end_dt <= start_dt:
        end_dt = start_dt + timedelta(days=1)
        end_date = end_dt.strftime('%Y-%m-%d')
        logging.info(f"Adjusted end_date to {end_date} (TwelveData requires end > start)")

    url = 'https://api.twelvedata.com/time_series'

    params = {
        'symbol': symbol,
        'interval': interval,
        'format': 'JSON',
        'timezone': TWELVEDATA_TIMEZONE,
        'apikey': api_key,
        'start_date': start_date,
        'end_date': end_date,
    }

    if bars:
        params['outputsize'] = min(bars, MAX_BARS_PER_REQUEST)

    try:
        logging.info(f"Fetching OHLCV: {start_date} to {end_date}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if 'code' in data and data['code'] != 200:
            logging.warning(f"API warning: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()

        if 'values' not in data:
            logging.warning(f"No data returned for {start_date} to {end_date}")
            return pd.DataFrame()

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
                'source': 'twelvedata_backfill'
            })

        df = pd.DataFrame(records)
        logging.info(f"Fetched {len(df)} bars")
        return df

    except Exception as e:
        logging.error(f"API fetch error: {e}")
        raise


def insert_ohlcv_batch(conn, df: pd.DataFrame) -> int:
    """Insert OHLCV batch with conflict handling."""
    if df.empty:
        return 0

    cur = conn.cursor()

    try:
        values = [
            (row['time'], row['symbol'], row['open'], row['high'],
             row['low'], row['close'], row['volume'], row['source'])
            for _, row in df.iterrows()
        ]

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
        return len(values)

    except Exception as e:
        conn.rollback()
        logging.error(f"Insert error: {e}")
        raise
    finally:
        cur.close()


# =============================================================================
# DAG TASKS
# =============================================================================

def detect_gap(**context) -> str:
    """
    UPDATED: Comprehensive gap detection from MIN to MAX date.

    Now scans ALL data from MIN to MAX date to find INTERNAL gaps,
    not just from last timestamp to NOW.

    Returns:
        'backfill_needed' if gaps exist
        'no_backfill' if data is complete
    """
    logging.info("=" * 70)
    logging.info("COMPREHENSIVE GAP DETECTION - Scanning ALL data from MIN to MAX")
    logging.info("=" * 70)

    conn = get_db_connection()

    try:
        # Step 1: Get data date range
        min_date, max_date = get_data_date_range(conn)
        current_time = get_current_market_time()

        if min_date is None:
            logging.warning("No existing OHLCV data found - full backfill needed")
            # Push default start date (1 year ago max)
            start_date = current_time - timedelta(days=MAX_BACKFILL_DAYS)
            context['ti'].xcom_push(key='mode', value='initial_backfill')
            context['ti'].xcom_push(key='last_timestamp', value=start_date.isoformat())
            context['ti'].xcom_push(key='current_time', value=current_time.isoformat())
            context['ti'].xcom_push(key='gap_bars', value=MAX_BACKFILL_DAYS * BARS_PER_SESSION)
            return 'backfill_needed'

        # Also check from max_date to current time (edge case)
        today = current_time.date()
        if max_date < today:
            # Extend max_date to today for complete analysis
            max_date = today

        logging.info(f"Data range in database: {min_date} to {max_date}")
        logging.info(f"Current time: {current_time}")

        # Step 2: Get all trading days in range
        all_trading_days = get_all_trading_days(min_date, max_date)
        logging.info(f"Total trading days in range: {len(all_trading_days)}")

        # Step 3: Get bars per day from database
        bars_per_day = get_bars_per_day(conn)
        logging.info(f"Days with data: {len(bars_per_day)}")

        # Step 4: Detect ALL gaps (including internal ones)
        gaps = detect_all_gaps(all_trading_days, bars_per_day)
        logging.info(f"Individual gap days found: {len(gaps)}")

        # Step 5: Group consecutive gaps into ranges
        gap_ranges = group_consecutive_gaps(gaps)
        logging.info(f"Gap ranges found: {len(gap_ranges)}")

        # Step 6: Calculate totals
        total_missing_bars = sum(g['total_bars_missing'] for g in gap_ranges)
        total_missing_days = sum(g['days_missing'] for g in gap_ranges)

        # Step 7: Report gaps
        logging.info("")
        logging.info("=" * 70)
        logging.info("GAP DETECTION SUMMARY")
        logging.info("=" * 70)
        logging.info(f"Data range scanned: {min_date} to {max_date}")
        logging.info(f"Total trading days: {len(all_trading_days)}")
        logging.info(f"Days with sufficient data: {len(bars_per_day)}")
        logging.info(f"Gap ranges found: {len(gap_ranges)}")
        logging.info(f"Total missing days: {total_missing_days}")
        logging.info(f"Total missing bars: {total_missing_bars}")
        logging.info("")

        # Detail each gap range
        for i, gap_range in enumerate(gap_ranges, 1):
            logging.info(f"GAP #{i}:")
            logging.info(f"  Period: {gap_range['start_date']} to {gap_range['end_date']}")
            logging.info(f"  Missing days: {gap_range['days_missing']}")
            logging.info(f"  Missing bars: {gap_range['total_bars_missing']}")

        logging.info("=" * 70)

        # Serialize gap ranges for XCom
        gap_ranges_serialized = []
        for gr in gap_ranges:
            gap_ranges_serialized.append({
                'start_date': gr['start_date'].isoformat(),
                'end_date': gr['end_date'].isoformat(),
                'days_missing': gr['days_missing'],
                'total_bars_missing': gr['total_bars_missing']
            })

        # Push to XCom
        context['ti'].xcom_push(key='mode', value='comprehensive_backfill')
        context['ti'].xcom_push(key='gap_ranges', value=gap_ranges_serialized)
        context['ti'].xcom_push(key='gap_bars', value=total_missing_bars)
        context['ti'].xcom_push(key='gaps_found', value=len(gap_ranges))
        context['ti'].xcom_push(key='total_missing_days', value=total_missing_days)
        context['ti'].xcom_push(key='min_date', value=min_date.isoformat())
        context['ti'].xcom_push(key='max_date', value=max_date.isoformat())

        # If gaps exist, we need backfill
        if len(gap_ranges) > 0:
            logging.info(f"BACKFILL NEEDED: {len(gap_ranges)} gap ranges, {total_missing_bars} bars missing")
            return 'backfill_needed'
        else:
            logging.info("Data is complete - no gaps found")
            return 'no_backfill'

    finally:
        conn.close()


def execute_backfill(**context):
    """
    UPDATED: Execute backfill for ALL detected gap ranges.

    Processes each gap range individually, fetching data from TwelveData API.
    """
    ti = context['ti']

    mode = ti.xcom_pull(key='mode', task_ids='detect_gap')
    gap_ranges = ti.xcom_pull(key='gap_ranges', task_ids='detect_gap')

    # Handle initial backfill mode (no existing data)
    if mode == 'initial_backfill':
        last_timestamp_str = ti.xcom_pull(key='last_timestamp', task_ids='detect_gap')
        current_time_str = ti.xcom_pull(key='current_time', task_ids='detect_gap')

        last_timestamp = datetime.fromisoformat(last_timestamp_str)
        current_time = datetime.fromisoformat(current_time_str)

        if last_timestamp.tzinfo is None:
            last_timestamp = COT_TZ.localize(last_timestamp)
        if current_time.tzinfo is None:
            current_time = COT_TZ.localize(current_time)

        logging.info(f"Initial backfill mode: {last_timestamp} to {current_time}")

        conn = get_db_connection()
        total_inserted = 0

        try:
            # Process in monthly chunks
            current_start = last_timestamp

            while current_start < current_time:
                chunk_end = min(current_start + timedelta(days=30), current_time)

                start_date = current_start.strftime('%Y-%m-%d')
                end_date = chunk_end.strftime('%Y-%m-%d')

                try:
                    df = fetch_ohlcv_data(
                        symbol=TWELVEDATA_SYMBOL,
                        interval=TWELVEDATA_INTERVAL,
                        start_date=start_date,
                        end_date=end_date
                    )

                    if not df.empty:
                        # Filter to only market hours
                        # TwelveData returns naive datetimes in COT timezone (as requested)
                        df['time_dt'] = pd.to_datetime(df['time'])
                        logging.info(f"Before filter: {len(df)} bars")
                        df = df[
                            (df['time_dt'].dt.hour >= MARKET_START_HOUR) &
                            ((df['time_dt'].dt.hour < MARKET_END_HOUR) |
                             ((df['time_dt'].dt.hour == MARKET_END_HOUR) & (df['time_dt'].dt.minute <= MARKET_END_MINUTE))) &
                            (df['time_dt'].dt.dayofweek.isin(MARKET_DAYS))
                        ]
                        logging.info(f"After market hours filter: {len(df)} bars")

                        # Filter out holidays
                        df = df[~df['time_dt'].dt.date.isin(HOLIDAYS)]
                        logging.info(f"After holiday filter: {len(df)} bars")
                        df = df.drop(columns=['time_dt'])

                        if not df.empty:
                            inserted = insert_ohlcv_batch(conn, df)
                            total_inserted += inserted
                            logging.info(f"Inserted {inserted} bars for {start_date} to {end_date}")
                        else:
                            logging.warning(f"All bars filtered out for {start_date} to {end_date}")

                    time.sleep(API_RATE_DELAY_SECONDS)

                except Exception as e:
                    logging.error(f"Error processing {start_date} to {end_date}: {e}")

                current_start = chunk_end

            context['ti'].xcom_push(key='total_bars_inserted', value=total_inserted)
            return {'status': 'success', 'bars_inserted': total_inserted, 'mode': 'initial'}

        finally:
            conn.close()

    # Comprehensive backfill mode (fill gap ranges)
    if not gap_ranges:
        logging.info("No gap ranges to backfill")
        context['ti'].xcom_push(key='total_bars_inserted', value=0)
        return {'status': 'skipped', 'message': 'No gaps found'}

    logging.info(f"Starting comprehensive backfill for {len(gap_ranges)} gap ranges")

    conn = get_db_connection()
    total_inserted = 0

    try:
        for i, gap_range in enumerate(gap_ranges, 1):
            start_date = gap_range['start_date']
            end_date = gap_range['end_date']

            logging.info(f"Backfilling gap #{i}/{len(gap_ranges)}: {start_date} to {end_date}")

            try:
                df = fetch_ohlcv_data(
                    symbol=TWELVEDATA_SYMBOL,
                    interval=TWELVEDATA_INTERVAL,
                    start_date=start_date,
                    end_date=end_date
                )

                if not df.empty:
                    # Filter to only market hours
                    # TwelveData returns naive datetimes in COT timezone (as requested)
                    df['time_dt'] = pd.to_datetime(df['time'])
                    logging.info(f"  Gap #{i}: Before filter: {len(df)} bars")
                    df = df[
                        (df['time_dt'].dt.hour >= MARKET_START_HOUR) &
                        ((df['time_dt'].dt.hour < MARKET_END_HOUR) |
                         ((df['time_dt'].dt.hour == MARKET_END_HOUR) & (df['time_dt'].dt.minute <= MARKET_END_MINUTE))) &
                        (df['time_dt'].dt.dayofweek.isin(MARKET_DAYS))
                    ]
                    logging.info(f"  Gap #{i}: After market hours filter: {len(df)} bars")

                    # Filter out holidays
                    df = df[~df['time_dt'].dt.date.isin(HOLIDAYS)]
                    logging.info(f"  Gap #{i}: After holiday filter: {len(df)} bars")
                    df = df.drop(columns=['time_dt'])

                    if not df.empty:
                        inserted = insert_ohlcv_batch(conn, df)
                        total_inserted += inserted
                        logging.info(f"  Inserted {inserted} bars for gap #{i}")
                    else:
                        logging.warning(f"  All bars filtered out for gap #{i}")
                else:
                    logging.warning(f"  No data returned for gap #{i}")

                time.sleep(API_RATE_DELAY_SECONDS)

            except Exception as e:
                logging.error(f"Error backfilling gap #{i}: {e}")
                continue

        logging.info(f"Comprehensive backfill complete. Total bars inserted: {total_inserted}")
        context['ti'].xcom_push(key='total_bars_inserted', value=total_inserted)

        return {
            'status': 'success',
            'bars_inserted': total_inserted,
            'gaps_processed': len(gap_ranges),
            'mode': 'comprehensive'
        }

    finally:
        conn.close()


def validate_backfill(**context):
    """Validate backfill results and log comprehensive summary."""
    ti = context['ti']

    total_inserted = ti.xcom_pull(key='total_bars_inserted', task_ids='execute_backfill') or 0
    expected_bars = ti.xcom_pull(key='gap_bars', task_ids='detect_gap') or 0
    gaps_found = ti.xcom_pull(key='gaps_found', task_ids='detect_gap') or 0
    total_missing_days = ti.xcom_pull(key='total_missing_days', task_ids='detect_gap') or 0
    min_date = ti.xcom_pull(key='min_date', task_ids='detect_gap')
    max_date = ti.xcom_pull(key='max_date', task_ids='detect_gap')

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Get current data summary
        cur.execute("""
            SELECT
                COUNT(*) as total_bars,
                MIN(time) as earliest,
                MAX(time) as latest
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
        """)

        result = cur.fetchone()
        total_bars, earliest, latest = result

        logging.info("")
        logging.info("=" * 70)
        logging.info("COMPREHENSIVE BACKFILL - FINAL VALIDATION REPORT")
        logging.info("=" * 70)
        logging.info(f"Data range scanned: {min_date} to {max_date}")
        logging.info(f"Gap ranges detected: {gaps_found}")
        logging.info(f"Total missing days: {total_missing_days}")
        logging.info(f"Expected bars to fill: {expected_bars}")
        logging.info(f"Bars inserted this run: {total_inserted}")
        logging.info(f"Fill rate: {round(total_inserted / max(expected_bars, 1) * 100, 1)}%")
        logging.info("-" * 70)
        logging.info(f"Current database state:")
        logging.info(f"  Total bars in database: {total_bars}")
        logging.info(f"  Data range: {earliest} to {latest}")
        logging.info("=" * 70)

        return {
            'status': 'validated',
            'gaps_found': gaps_found,
            'total_missing_days': total_missing_days,
            'expected_bars': expected_bars,
            'bars_inserted': total_inserted,
            'fill_rate_pct': round(total_inserted / max(expected_bars, 1) * 100, 1),
            'total_bars_in_db': total_bars,
            'earliest': str(earliest),
            'latest': str(latest)
        }

    finally:
        cur.close()
        conn.close()


def skip_backfill(**context):
    """Log that backfill was skipped."""
    logging.info("=" * 70)
    logging.info("BACKFILL SKIPPED - Data is complete, no gaps found")
    logging.info("=" * 70)
    return {'status': 'skipped', 'reason': 'data_complete'}


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='V3 L0: COMPREHENSIVE OHLCV gap detection from MIN to MAX date and automatic backfill',
    schedule_interval=None,  # Manual trigger or startup only
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l0', 'ohlcv', 'backfill', 'startup', 'gap-detection', 'comprehensive']
)

with dag:

    # Task 1: Comprehensive gap detection (MIN to MAX date)
    task_detect = BranchPythonOperator(
        task_id='detect_gap',
        python_callable=detect_gap,
        provide_context=True
    )

    # Task 2a: Execute backfill (if gaps detected)
    task_backfill = PythonOperator(
        task_id='backfill_needed',
        python_callable=execute_backfill,
        provide_context=True,
        pool='api_requests'  # Rate limiting
    )

    # Task 2b: Skip (if no gaps)
    task_skip = PythonOperator(
        task_id='no_backfill',
        python_callable=skip_backfill,
        provide_context=True
    )

    # Task 3: Validate results
    task_validate = PythonOperator(
        task_id='validate_backfill',
        python_callable=validate_backfill,
        provide_context=True,
        trigger_rule='none_failed_min_one_success'  # Run after either branch
    )

    # Task dependencies with branching
    task_detect >> [task_backfill, task_skip]
    task_backfill >> task_validate
    task_skip >> task_validate
