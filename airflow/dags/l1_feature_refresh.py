"""
DAG: v3.l1_feature_refresh
===========================
V3 Architecture - Layer 1: Feature Calculation with Macro FFILL

Purpose:
    Calculates ALL 13 features for each OHLCV bar using Python.
    Macro data is forward-filled (ffill) from daily values to each 5-min bar.

Schedule:
    Event-driven: Uses NewOHLCVBarSensor to wait for new data from L0.
    Fallback schedule: */5 13-17 * * 1-5 (8:00-12:55 COT)

    Best Practice: Instead of running blindly every 5 minutes, the sensor
    waits for actual new OHLCV data before processing. This prevents:
    - Schedule drift
    - Overlapping jobs
    - Processing stale data

Feature Calculation Strategy:
    1. Load OHLCV bars (last 100 for indicator warmup)
    2. Load macro_indicators_daily and FFILL to each bar's date
    3. Calculate returns: log_ret_5m, log_ret_1h, log_ret_4h
    4. Calculate indicators: rsi_9, atr_pct, adx_14
    5. Calculate macro z-scores: dxy_z, vix_z (using historical mean/std)
    6. Calculate macro changes: dxy_change_1d, brent_change_1d
    7. Calculate derived: rate_spread, usdmxn_change_1d
    8. Store ALL features in inference_features_5m table

Data Flow:
    ┌─────────────────┐     ┌────────────────┐
    │usdcop_m5_ohlcv  │────►│                │
    │ time, OHLCV     │     │                │
    └─────────────────┘     │  PYTHON        │     ┌────────────────────┐
                            │  FEATURE       │────►│inference_features  │
    ┌─────────────────┐     │  CALCULATION   │     │_5m (13 features)   │
    │macro_indicators │────►│  + FFILL       │     └────────────────────┘
    │_daily           │     │                │
    │ dxy, vix, etc   │     └────────────────┘
    └─────────────────┘

Author: Pedro @ Lean Tech Solutions
Version: 3.2.0 (with sensor-driven execution)
Updated: 2025-01-12
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import pandas as pd
import numpy as np
import logging
import psycopg2
from psycopg2.extras import execute_values
import os

# Trading calendar from DAG utils
from utils.trading_calendar import TradingCalendar
from utils.dag_common import get_db_connection, load_feature_config

# Event-driven sensor (Best Practice: wait for actual data instead of fixed schedule)
from sensors.new_bar_sensor import NewOHLCVBarSensor

DAG_ID = 'v3.l1_feature_refresh'
CONFIG = load_feature_config(raise_on_error=False)

# Feature calculation parameters (from config or defaults)
MACRO_ZSCORE_STATS = CONFIG.get('macro_zscore_stats', {
    'dxy': {'mean': 103.5, 'std': 2.5},
    'vix': {'mean': 18.0, 'std': 5.0},
    'embi': {'mean': 400.0, 'std': 50.0}  # Not used currently
})

# Initialize trading calendar
trading_cal = TradingCalendar()


def should_run_today():
    """Check if today is a valid trading day."""
    today = datetime.now()
    if not trading_cal.is_trading_day(today):
        reason = trading_cal.get_violation_reason(today)
        logging.info(f"Skipping - {today.date()}: {reason}")
        return False
    return True


# =============================================================================
# FEATURE CALCULATION FUNCTIONS
# =============================================================================

def calc_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns for different periods."""
    df = df.copy()
    df['log_ret_5m'] = np.log(df['close'] / df['close'].shift(1))
    df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(12))   # 12 bars = 1 hour
    df['log_ret_4h'] = np.log(df['close'] / df['close'].shift(48))   # 48 bars = 4 hours
    return df


def calc_rsi(series: pd.Series, period: int = 9) -> pd.Series:
    """Calculate RSI indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Calculate ATR (Average True Range)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX indicator."""
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff().abs() * -1

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()

    tr = calc_atr(df, 1) * 1  # True range (period 1)

    plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx


def calc_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators."""
    df = df.copy()
    df['rsi_9'] = calc_rsi(df['close'], period=9)
    df['atr_pct'] = (calc_atr(df, period=10) / df['close']) * 100
    df['adx_14'] = calc_adx(df, period=14)
    return df


def calc_macro_features(df: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate macro features with FFILL from daily to 5-min bars.

    Args:
        df: OHLCV DataFrame with 'time' column
        df_macro: macro_indicators_daily DataFrame with 'date', 'dxy', 'vix', etc.

    Returns:
        DataFrame with macro features ffilled to each bar
    """
    df = df.copy()

    # Extract date from OHLCV time
    df['date'] = pd.to_datetime(df['time']).dt.date

    # Ensure macro has date column
    if 'date' in df_macro.columns:
        df_macro = df_macro.copy()
        df_macro['date'] = pd.to_datetime(df_macro['date']).dt.date
    else:
        logging.warning("Macro data missing 'date' column")
        return df

    # Sort macro by date for proper merge_asof
    df_macro = df_macro.sort_values('date')
    df = df.sort_values('time')

    # Convert date to datetime for merge_asof
    df['date_dt'] = pd.to_datetime(df['date'])
    df_macro['date_dt'] = pd.to_datetime(df_macro['date'])

    # FFILL: merge_asof finds the most recent macro data for each bar
    df = pd.merge_asof(
        df,
        df_macro[['date_dt', 'dxy', 'vix', 'brent', 'treasury_2y', 'treasury_10y', 'usdmxn']],
        left_on='date_dt',
        right_on='date_dt',
        direction='backward'  # Use most recent available data
    )

    # Calculate Z-scores using config stats
    dxy_stats = MACRO_ZSCORE_STATS.get('dxy', {'mean': 103.5, 'std': 2.5})
    vix_stats = MACRO_ZSCORE_STATS.get('vix', {'mean': 18.0, 'std': 5.0})

    df['dxy_z'] = (df['dxy'] - dxy_stats['mean']) / dxy_stats['std']
    df['vix_z'] = (df['vix'] - vix_stats['mean']) / vix_stats['std']
    df['embi_z'] = 0.0  # EMBI not available, set to 0

    # Calculate daily changes (requires previous day data)
    df['dxy_change_1d'] = df.groupby('date')['dxy'].transform(
        lambda x: (x / x.iloc[0] - 1) if len(x) > 0 else 0
    )
    df['brent_change_1d'] = df.groupby('date')['brent'].transform(
        lambda x: (x / x.iloc[0] - 1) if len(x) > 0 else 0
    )

    # Rate spread
    df['rate_spread'] = df['treasury_10y'] - df['treasury_2y']

    # USDMXN daily change
    df['usdmxn_change_1d'] = np.log(df['usdmxn'] / df['usdmxn'].shift(12))

    # Clean up temp columns
    df = df.drop(columns=['date', 'date_dt'], errors='ignore')

    return df


# =============================================================================
# MAIN TASK FUNCTIONS
# =============================================================================

def calculate_all_features(**context):
    """
    Main feature calculation task with macro FFILL.

    This replaces both refresh_sql_features and calculate_python_features
    with a unified Python implementation that properly handles ffill.
    """
    logging.info("=" * 60)
    logging.info("STARTING FEATURE CALCULATION WITH MACRO FFILL")
    logging.info("=" * 60)

    conn = get_db_connection()

    try:
        # =================================================================
        # STEP 1: Load OHLCV data (last 100 bars for indicator warmup)
        # =================================================================
        query_ohlcv = """
            SELECT time, open, high, low, close, volume
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
            ORDER BY time DESC
            LIMIT 100
        """
        df = pd.read_sql(query_ohlcv, conn)
        df = df.sort_values('time').reset_index(drop=True)
        logging.info(f"Loaded {len(df)} OHLCV bars")

        if df.empty:
            logging.error("No OHLCV data available!")
            return {'status': 'error', 'reason': 'no_ohlcv_data'}

        # =================================================================
        # STEP 2: Load macro data (last 30 days for ffill history)
        # =================================================================
        query_macro = """
            SELECT
                fecha as date,
                fxrt_index_dxy_usa_d_dxy as dxy,
                volt_vix_usa_d_vix as vix,
                comm_oil_brent_glb_d_brent as brent,
                finc_bond_yield2y_usa_d_dgs2 as treasury_2y,
                finc_bond_yield10y_usa_d_ust10y as treasury_10y,
                fxrt_spot_usdmxn_mex_d_usdmxn as usdmxn
            FROM macro_indicators_daily
            WHERE fecha >= CURRENT_DATE - INTERVAL '30 days'
            ORDER BY fecha
        """
        df_macro = pd.read_sql(query_macro, conn)
        logging.info(f"Loaded {len(df_macro)} macro records for ffill")

        if df_macro.empty:
            logging.warning("No macro data available - using defaults")
            # Create default macro row for today
            df_macro = pd.DataFrame([{
                'date': datetime.now().date(),
                'dxy': MACRO_ZSCORE_STATS['dxy']['mean'],
                'vix': MACRO_ZSCORE_STATS['vix']['mean'],
                'brent': 75.0,
                'treasury_2y': 4.5,
                'treasury_10y': 4.0,
                'usdmxn': 17.5
            }])

        # =================================================================
        # STEP 3: Calculate log returns
        # =================================================================
        df = calc_log_returns(df)
        logging.info("Calculated log returns")

        # =================================================================
        # STEP 4: Calculate technical indicators
        # =================================================================
        df = calc_technical_indicators(df)
        logging.info("Calculated technical indicators (RSI, ATR, ADX)")

        # =================================================================
        # STEP 5: Calculate macro features with FFILL
        # =================================================================
        df = calc_macro_features(df, df_macro)
        logging.info("Calculated macro features with FFILL")

        # =================================================================
        # STEP 6: Create/update inference_features_5m table
        # =================================================================
        cur = conn.cursor()

        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS inference_features_5m (
                time TIMESTAMPTZ PRIMARY KEY,
                -- Returns
                log_ret_5m DOUBLE PRECISION,
                log_ret_1h DOUBLE PRECISION,
                log_ret_4h DOUBLE PRECISION,
                -- Macro Z-scores
                dxy_z DOUBLE PRECISION,
                vix_z DOUBLE PRECISION,
                embi_z DOUBLE PRECISION,
                -- Macro changes
                dxy_change_1d DOUBLE PRECISION,
                brent_change_1d DOUBLE PRECISION,
                -- Derived
                rate_spread DOUBLE PRECISION,
                -- Technical indicators
                rsi_9 DOUBLE PRECISION,
                atr_pct DOUBLE PRECISION,
                adx_14 DOUBLE PRECISION,
                -- USDMXN correlation feature
                usdmxn_change_1d DOUBLE PRECISION,
                -- Metadata
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Prepare data for insert (only recent bars to update)
        # Skip first 50 bars (warmup period for indicators)
        df_to_insert = df.iloc[50:].copy()

        if df_to_insert.empty:
            logging.warning("Not enough bars after warmup period")
            return {'status': 'warning', 'reason': 'insufficient_warmup'}

        # Insert/update features
        insert_sql = """
            INSERT INTO inference_features_5m (
                time, log_ret_5m, log_ret_1h, log_ret_4h,
                dxy_z, vix_z, embi_z,
                dxy_change_1d, brent_change_1d, rate_spread,
                rsi_9, atr_pct, adx_14, usdmxn_change_1d
            ) VALUES %s
            ON CONFLICT (time) DO UPDATE SET
                log_ret_5m = EXCLUDED.log_ret_5m,
                log_ret_1h = EXCLUDED.log_ret_1h,
                log_ret_4h = EXCLUDED.log_ret_4h,
                dxy_z = EXCLUDED.dxy_z,
                vix_z = EXCLUDED.vix_z,
                embi_z = EXCLUDED.embi_z,
                dxy_change_1d = EXCLUDED.dxy_change_1d,
                brent_change_1d = EXCLUDED.brent_change_1d,
                rate_spread = EXCLUDED.rate_spread,
                rsi_9 = EXCLUDED.rsi_9,
                atr_pct = EXCLUDED.atr_pct,
                adx_14 = EXCLUDED.adx_14,
                usdmxn_change_1d = EXCLUDED.usdmxn_change_1d,
                updated_at = NOW()
        """

        # Prepare values (handle NaN)
        values = []
        for _, row in df_to_insert.iterrows():
            values.append((
                row['time'],
                None if pd.isna(row.get('log_ret_5m')) else float(row['log_ret_5m']),
                None if pd.isna(row.get('log_ret_1h')) else float(row['log_ret_1h']),
                None if pd.isna(row.get('log_ret_4h')) else float(row['log_ret_4h']),
                None if pd.isna(row.get('dxy_z')) else float(row['dxy_z']),
                None if pd.isna(row.get('vix_z')) else float(row['vix_z']),
                None if pd.isna(row.get('embi_z')) else float(row['embi_z']),
                None if pd.isna(row.get('dxy_change_1d')) else float(row['dxy_change_1d']),
                None if pd.isna(row.get('brent_change_1d')) else float(row['brent_change_1d']),
                None if pd.isna(row.get('rate_spread')) else float(row['rate_spread']),
                None if pd.isna(row.get('rsi_9')) else float(row['rsi_9']),
                None if pd.isna(row.get('atr_pct')) else float(row['atr_pct']),
                None if pd.isna(row.get('adx_14')) else float(row['adx_14']),
                None if pd.isna(row.get('usdmxn_change_1d')) else float(row['usdmxn_change_1d'])
            ))

        execute_values(cur, insert_sql, values)
        conn.commit()

        rows_inserted = len(values)
        logging.info(f"Inserted/updated {rows_inserted} feature rows")

        # Push metrics
        context['ti'].xcom_push(key='features_count', value=rows_inserted)
        context['ti'].xcom_push(key='macro_rows_used', value=len(df_macro))

        # Log sample of latest features
        if len(df_to_insert) > 0:
            latest = df_to_insert.iloc[-1]
            logging.info("Latest feature values:")
            logging.info(f"  time: {latest['time']}")
            logging.info(f"  log_ret_5m: {latest.get('log_ret_5m', 'N/A'):.6f}")
            logging.info(f"  dxy_z: {latest.get('dxy_z', 'N/A'):.4f}")
            logging.info(f"  vix_z: {latest.get('vix_z', 'N/A'):.4f}")
            logging.info(f"  rsi_9: {latest.get('rsi_9', 'N/A'):.2f}")
            logging.info(f"  rate_spread: {latest.get('rate_spread', 'N/A'):.4f}")

        return {
            'status': 'success',
            'rows_inserted': rows_inserted,
            'ohlcv_bars': len(df),
            'macro_records': len(df_macro)
        }

    except Exception as e:
        logging.error(f"Error calculating features: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()


def validate_features(**context):
    """
    Validate that features are available for the latest bar.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Get latest OHLCV time
        cur.execute("""
            SELECT MAX(time) FROM usdcop_m5_ohlcv WHERE symbol = 'USD/COP'
        """)
        latest_ohlcv = cur.fetchone()[0]

        # Get latest feature time
        cur.execute("""
            SELECT MAX(time), COUNT(*)
            FROM inference_features_5m
            WHERE time >= NOW() - INTERVAL '1 hour'
        """)
        result = cur.fetchone()
        latest_feature, recent_count = result

        # Check feature completeness for latest bar
        cur.execute("""
            SELECT
                time,
                CASE WHEN log_ret_5m IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN log_ret_1h IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN dxy_z IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN vix_z IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN rsi_9 IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN rate_spread IS NOT NULL THEN 1 ELSE 0 END as non_null_count
            FROM inference_features_5m
            ORDER BY time DESC
            LIMIT 1
        """)
        completeness = cur.fetchone()

        logging.info("=" * 60)
        logging.info("FEATURE VALIDATION SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Latest OHLCV bar: {latest_ohlcv}")
        logging.info(f"Latest feature time: {latest_feature}")
        logging.info(f"Features in last hour: {recent_count}")
        if completeness:
            logging.info(f"Feature completeness (latest): {completeness[1]}/6 key features")
        logging.info("=" * 60)

        # Validate no holiday data
        cur.execute("""
            SELECT DISTINCT DATE(time AT TIME ZONE 'America/Bogota') as trade_date
            FROM inference_features_5m
            WHERE time >= NOW() - INTERVAL '7 days'
            ORDER BY trade_date DESC
        """)

        dates_with_data = [row[0] for row in cur.fetchall()]
        invalid_dates = []

        for trade_date in dates_with_data:
            if not trading_cal.is_trading_day(trade_date):
                reason = trading_cal.get_violation_reason(trade_date)
                invalid_dates.append({'date': str(trade_date), 'reason': reason})

        if invalid_dates:
            logging.warning(f"Found features on {len(invalid_dates)} non-trading days")

        context['ti'].xcom_push(key='invalid_trading_dates', value=invalid_dates)

        is_valid = (recent_count > 0) and (completeness and completeness[1] >= 4)

        return {
            'status': 'valid' if is_valid else 'incomplete',
            'latest_ohlcv': str(latest_ohlcv),
            'latest_feature': str(latest_feature),
            'recent_features': recent_count,
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
    'retries': 2,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='V3 L1 Feature Refresh - Event-driven with NewOHLCVBarSensor',
    schedule_interval='*/5 13-17 * * 1-5',  # Trigger schedule (sensor waits for actual data)
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l1', 'features', 'realtime', 'ffill', 'sensor-driven']
)

with dag:

    def check_trading_day(**context):
        """Branch task to skip processing on holidays/weekends."""
        if should_run_today():
            return 'wait_for_ohlcv'  # Changed: go to sensor first
        else:
            return 'skip_processing'

    task_check = BranchPythonOperator(
        task_id='check_trading_day',
        python_callable=check_trading_day,
        provide_context=True
    )

    task_skip = EmptyOperator(
        task_id='skip_processing'
    )

    # EVENT-DRIVEN SENSOR: Wait for new OHLCV data instead of running blindly
    # This prevents schedule drift and ensures L0 has completed before L1 runs
    task_wait_ohlcv = NewOHLCVBarSensor(
        task_id='wait_for_ohlcv',
        table_name='usdcop_m5_ohlcv',
        symbol='USD/COP',
        max_staleness_minutes=10,  # Data must be < 10 minutes old
        poke_interval=30,          # Check every 30 seconds
        timeout=300,               # Max 5 minutes wait (matches schedule)
        mode='poke',               # Keep worker while waiting
    )

    task_calculate = PythonOperator(
        task_id='calculate_all_features',
        python_callable=calculate_all_features,
        provide_context=True
    )

    task_validate = PythonOperator(
        task_id='validate_features',
        python_callable=validate_features,
        provide_context=True
    )

    def mark_processed(**context):
        """Mark OHLCV time as processed for next sensor check."""
        ti = context['ti']
        detected_time = ti.xcom_pull(key='detected_ohlcv_time', task_ids='wait_for_ohlcv')
        if detected_time:
            ti.xcom_push(key='last_processed_ohlcv_time', value=detected_time)
            logging.info(f"Marked OHLCV time as processed: {detected_time}")

    task_mark_processed = PythonOperator(
        task_id='mark_processed',
        python_callable=mark_processed,
        provide_context=True,
        trigger_rule='all_success'
    )

    # Task dependencies with sensor
    # Trading day check -> (sensor wait OR skip) -> calculate -> validate -> mark
    task_check >> [task_wait_ohlcv, task_skip]
    task_wait_ohlcv >> task_calculate >> task_validate >> task_mark_processed
