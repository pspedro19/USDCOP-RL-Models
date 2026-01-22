"""
L1 Feature Refresh DAG - SSOT Edition
======================================
P0-1: Uses CanonicalFeatureBuilder as Single Source of Truth

Contract: CTR-FEAT-001 - 15 features in canonical order

This DAG replaces all duplicated feature calculations with the
CanonicalFeatureBuilder from src.feature_store.builders, ensuring:
1. Same calculations as training (Wilder's EMA for RSI/ATR/ADX)
2. Same feature order (FEATURE_ORDER from core.py)
3. Same normalization stats
4. Builder version tracked for audit

Schedule:
    Event-driven: Uses NewOHLCVBarSensor to wait for new data from L0.
    Fallback schedule: */5 13-17 * * 1-5 (8:00-12:55 COT)

Data Flow:
    +-----------------+     +----------------+
    |usdcop_m5_ohlcv  |---->|                |
    | time, OHLCV     |     |                |
    +-----------------+     |  CANONICAL     |     +--------------------+
                            |  FEATURE       |---->|inference_features  |
    +-----------------+     |  BUILDER       |     |_5m (15 features)   |
    |macro_indicators |---->|  (SSOT)        |     +--------------------+
    |_daily           |     |  Wilder's EMA  |
    | dxy, vix, etc   |     +----------------+
    +-----------------+

Author: Pedro @ Lean Tech Solutions / Trading Team
Version: 4.0.0 (SSOT CanonicalFeatureBuilder integration)
Updated: 2026-01-17
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import pandas as pd
import numpy as np
import logging
import psycopg2
from psycopg2.extras import execute_values
import json

# Trading calendar from DAG utils
from utils.trading_calendar import TradingCalendar
from utils.dag_common import get_db_connection, load_feature_config

# Event-driven sensor (Best Practice: wait for actual data instead of fixed schedule)
from sensors.new_bar_sensor import NewOHLCVBarSensor

# =============================================================================
# SSOT IMPORTS - Single Source of Truth for Feature Calculations
# =============================================================================
# CRITICAL: Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import SSOT components (REQUIRED - no fallback)
from src.core.contracts import FEATURE_ORDER, OBSERVATION_DIM, FEATURE_CONTRACT
from src.feature_store.builders import CanonicalFeatureBuilder

SSOT_AVAILABLE = True
logging.info(
    f"SSOT CanonicalFeatureBuilder loaded successfully. "
    f"Builder version: {CanonicalFeatureBuilder.VERSION}, "
    f"Feature count: {OBSERVATION_DIM}"
)

from contracts.dag_registry import L1_FEATURE_REFRESH

logger = logging.getLogger(__name__)

DAG_ID = L1_FEATURE_REFRESH
CONFIG = load_feature_config(raise_on_error=False)

# Feature calculation parameters (from config or defaults)
MACRO_ZSCORE_STATS = CONFIG.get('macro_zscore_stats', {
    'dxy': {'mean': 103.5, 'std': 2.5},
    'vix': {'mean': 18.0, 'std': 5.0},
    'embi': {'mean': 400.0, 'std': 50.0}
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
# MAIN TASK FUNCTIONS - USING SSOT CANONICAL BUILDER
# =============================================================================

def compute_features_ssot(**context) -> dict:
    """
    Compute features using the Single Source of Truth builder.

    This ensures:
    1. Same calculations as training (Wilder's EMA for RSI/ATR/ADX)
    2. Same feature order (CTR-FEAT-001)
    3. Same normalization stats
    4. Builder version tracked for audit
    """
    logging.info("=" * 60)
    logging.info("STARTING FEATURE CALCULATION WITH SSOT CANONICAL BUILDER")
    logging.info("=" * 60)

    if not SSOT_AVAILABLE:
        raise RuntimeError(
            "SSOT CanonicalFeatureBuilder not available. "
            "Cannot compute features without SSOT to ensure training/inference parity."
        )

    # Initialize SSOT builder
    builder = CanonicalFeatureBuilder()
    logging.info(f"Using CanonicalFeatureBuilder v{builder.VERSION}")

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
            raise ValueError("No OHLCV data found for feature computation")

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
                fxrt_spot_usdmxn_mex_d_usdmxn as usdmxn,
                0.0 as embi  -- EMBI not available, default to 0
            FROM macro_indicators_daily
            WHERE fecha >= CURRENT_DATE - INTERVAL '30 days'
            ORDER BY fecha
        """
        df_macro = pd.read_sql(query_macro, conn)
        logging.info(f"Loaded {len(df_macro)} macro records for ffill")

        # =================================================================
        # STEP 3: Merge macro data with OHLCV using FFILL
        # =================================================================
        if not df_macro.empty:
            df = _merge_macro_ffill(df, df_macro)
            logging.info("Merged macro data with FFILL")
        else:
            logging.warning("No macro data available - using defaults")
            # Add default macro columns
            df['dxy'] = MACRO_ZSCORE_STATS['dxy']['mean']
            df['vix'] = MACRO_ZSCORE_STATS['vix']['mean']
            df['brent'] = 75.0
            df['treasury_2y'] = 4.5
            df['treasury_10y'] = 4.0
            df['usdmxn'] = 17.5
            df['embi'] = 0.0

        # =================================================================
        # STEP 4: Compute features using SSOT CanonicalFeatureBuilder
        # =================================================================
        # Use the canonical builder - guaranteed contract compliance
        features_df = builder.compute_features(df, include_state=False)
        logging.info(f"Computed {len(features_df.columns)} features using SSOT builder")

        # Get latest features for validation
        latest_features = builder.get_latest_features_dict(df, position=0.0)

        # =================================================================
        # STEP 5: Validate feature count matches contract
        # =================================================================
        if len(latest_features) != len(FEATURE_ORDER):
            raise ValueError(
                f"Feature count mismatch: got {len(latest_features)}, "
                f"expected {len(FEATURE_ORDER)} (CTR-FEAT-001)"
            )

        # Validate features against contract
        is_valid, errors = builder.validate_features(latest_features)
        if not is_valid:
            for error in errors:
                logging.warning(f"Feature validation warning: {error}")

        # Build feature vector in canonical order
        feature_vector = [latest_features[f] for f in FEATURE_ORDER]

        # =================================================================
        # STEP 6: Store in database with builder version for audit
        # =================================================================
        cur = conn.cursor()

        # Create/update inference_features_5m table with all 15 features
        cur.execute("""
            CREATE TABLE IF NOT EXISTS inference_features_5m (
                time TIMESTAMPTZ PRIMARY KEY,
                -- Returns
                log_ret_5m DOUBLE PRECISION,
                log_ret_1h DOUBLE PRECISION,
                log_ret_4h DOUBLE PRECISION,
                -- Technical indicators (Wilder's EMA)
                rsi_9 DOUBLE PRECISION,
                atr_pct DOUBLE PRECISION,
                adx_14 DOUBLE PRECISION,
                -- Macro Z-scores
                dxy_z DOUBLE PRECISION,
                dxy_change_1d DOUBLE PRECISION,
                vix_z DOUBLE PRECISION,
                embi_z DOUBLE PRECISION,
                -- Macro changes
                brent_change_1d DOUBLE PRECISION,
                rate_spread DOUBLE PRECISION,
                usdmxn_change_1d DOUBLE PRECISION,
                -- State features
                position DOUBLE PRECISION DEFAULT 0.0,
                time_normalized DOUBLE PRECISION,
                -- Metadata
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Create feature_cache table for JSON storage and version tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feature_cache (
                timestamp TIMESTAMPTZ PRIMARY KEY,
                features_json JSONB NOT NULL,
                feature_vector DOUBLE PRECISION[] NOT NULL,
                builder_version VARCHAR(20) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Skip first 50 bars (warmup period for indicators)
        df_to_insert = features_df.iloc[50:].copy()
        df_to_insert['time'] = df.iloc[50:]['time'].values

        if df_to_insert.empty:
            logging.warning("Not enough bars after warmup period")
            return {'status': 'warning', 'reason': 'insufficient_warmup'}

        # Insert/update inference_features_5m
        insert_sql = """
            INSERT INTO inference_features_5m (
                time, log_ret_5m, log_ret_1h, log_ret_4h,
                rsi_9, atr_pct, adx_14,
                dxy_z, dxy_change_1d, vix_z, embi_z,
                brent_change_1d, rate_spread, usdmxn_change_1d,
                position, time_normalized
            ) VALUES %s
            ON CONFLICT (time) DO UPDATE SET
                log_ret_5m = EXCLUDED.log_ret_5m,
                log_ret_1h = EXCLUDED.log_ret_1h,
                log_ret_4h = EXCLUDED.log_ret_4h,
                rsi_9 = EXCLUDED.rsi_9,
                atr_pct = EXCLUDED.atr_pct,
                adx_14 = EXCLUDED.adx_14,
                dxy_z = EXCLUDED.dxy_z,
                dxy_change_1d = EXCLUDED.dxy_change_1d,
                vix_z = EXCLUDED.vix_z,
                embi_z = EXCLUDED.embi_z,
                brent_change_1d = EXCLUDED.brent_change_1d,
                rate_spread = EXCLUDED.rate_spread,
                usdmxn_change_1d = EXCLUDED.usdmxn_change_1d,
                position = EXCLUDED.position,
                time_normalized = EXCLUDED.time_normalized,
                updated_at = NOW()
        """

        # Prepare values (handle NaN)
        values = []
        for idx, row in df_to_insert.iterrows():
            values.append((
                row['time'],
                _safe_float(row.get('log_ret_5m')),
                _safe_float(row.get('log_ret_1h')),
                _safe_float(row.get('log_ret_4h')),
                _safe_float(row.get('rsi_9')),
                _safe_float(row.get('atr_pct')),
                _safe_float(row.get('adx_14')),
                _safe_float(row.get('dxy_z')),
                _safe_float(row.get('dxy_change_1d')),
                _safe_float(row.get('vix_z')),
                _safe_float(row.get('embi_z')),
                _safe_float(row.get('brent_change_1d')),
                _safe_float(row.get('rate_spread')),
                _safe_float(row.get('usdmxn_change_1d')),
                _safe_float(row.get('position', 0.0)),
                _safe_float(row.get('time_normalized', 0.5)),
            ))

        execute_values(cur, insert_sql, values)

        # Store in feature_cache with hash for validation
        latest_time = df['time'].iloc[-1]
        cache_insert = """
            INSERT INTO feature_cache (timestamp, features_json, feature_vector, builder_version)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (timestamp) DO UPDATE SET
                features_json = EXCLUDED.features_json,
                feature_vector = EXCLUDED.feature_vector,
                builder_version = EXCLUDED.builder_version,
                updated_at = NOW()
        """
        cur.execute(cache_insert, (
            latest_time.isoformat() if hasattr(latest_time, 'isoformat') else str(latest_time),
            json.dumps(latest_features),
            feature_vector,
            builder.VERSION,
        ))

        conn.commit()

        rows_inserted = len(values)
        logging.info(f"Inserted/updated {rows_inserted} feature rows")
        logging.info(
            f"SSOT features computed: {len(feature_vector)} features, "
            f"builder v{builder.VERSION}"
        )

        # Push metrics to XCom
        context['ti'].xcom_push(key='features_count', value=rows_inserted)
        context['ti'].xcom_push(key='builder_version', value=builder.VERSION)
        context['ti'].xcom_push(key='feature_count', value=len(feature_vector))

        # Log sample of latest features
        logging.info("Latest feature values (SSOT computed):")
        for name in FEATURE_ORDER[:6]:  # First 6 features
            logging.info(f"  {name}: {latest_features.get(name, 'N/A'):.6f}")

        return {
            'status': 'success',
            'timestamp': str(latest_time),
            'feature_count': len(feature_vector),
            'builder_version': builder.VERSION,
            'rows_inserted': rows_inserted,
        }

    except Exception as e:
        logging.error(f"Error calculating features: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()


def validate_feature_consistency(**context) -> bool:
    """
    Validate features match expected schema and ranges.

    Part of CTR-FEAT-001 contract enforcement.
    """
    if not SSOT_AVAILABLE:
        logging.error("SSOT not available for validation")
        return False

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Get latest computed features from cache
        query = """
            SELECT features_json, builder_version, timestamp
            FROM feature_cache
            ORDER BY timestamp DESC
            LIMIT 1
        """
        cur.execute(query)
        result = cur.fetchone()

        if not result:
            raise ValueError("No features found for validation")

        features = json.loads(result[0])
        builder_version = result[1]
        timestamp = result[2]

        logging.info("=" * 60)
        logging.info("FEATURE VALIDATION (CTR-FEAT-001)")
        logging.info("=" * 60)
        logging.info(f"Timestamp: {timestamp}")
        logging.info(f"Builder Version: {builder_version}")

        # Validate all expected features present
        missing = set(FEATURE_ORDER) - set(features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        logging.info(f"Feature count: {len(features)} (expected {OBSERVATION_DIM})")

        # Validate 15 features exactly
        if len(features) != OBSERVATION_DIM:
            raise ValueError(
                f"Feature count mismatch: got {len(features)}, expected {OBSERVATION_DIM}"
            )

        # Validate ranges
        validations = {
            'rsi_9': (0, 100),
            'atr_pct': (0, 0.1),  # 0-10%
            'adx_14': (0, 100),
            'position': (-1, 1),
            'time_normalized': (0, 1),
        }

        warnings = []
        for feature, (min_val, max_val) in validations.items():
            if feature in features:
                val = features[feature]
                if not (min_val <= val <= max_val):
                    msg = f"Feature {feature}={val} outside expected range [{min_val}, {max_val}]"
                    logging.warning(msg)
                    warnings.append(msg)

        # Check feature completeness for recent data
        cur.execute("""
            SELECT COUNT(*) as cnt
            FROM inference_features_5m
            WHERE time >= NOW() - INTERVAL '1 hour'
        """)
        recent_count = cur.fetchone()[0]

        logging.info(f"Features in last hour: {recent_count}")
        logging.info(f"Validation warnings: {len(warnings)}")
        logging.info(f"Feature validation passed: {len(features)} features, builder v{builder_version}")
        logging.info("=" * 60)

        # Check for data on non-trading days
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

        context['ti'].xcom_push(key='validation_status', value='passed')
        context['ti'].xcom_push(key='validation_warnings', value=len(warnings))
        context['ti'].xcom_push(key='invalid_trading_dates', value=invalid_dates)

        return True

    except Exception as e:
        logging.error(f"Feature validation failed: {e}")
        context['ti'].xcom_push(key='validation_status', value='failed')
        context['ti'].xcom_push(key='validation_error', value=str(e))
        raise

    finally:
        cur.close()
        conn.close()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _merge_macro_ffill(df: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro data with OHLCV using forward fill.

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

    # Sort macro by date for proper merge_asof
    df_macro = df_macro.sort_values('date')
    df = df.sort_values('time')

    # Convert date to datetime for merge_asof
    df['date_dt'] = pd.to_datetime(df['date'])
    df_macro['date_dt'] = pd.to_datetime(df_macro['date'])

    # FFILL: merge_asof finds the most recent macro data for each bar
    macro_cols = ['date_dt', 'dxy', 'vix', 'brent', 'treasury_2y', 'treasury_10y', 'usdmxn']
    if 'embi' in df_macro.columns:
        macro_cols.append('embi')

    df = pd.merge_asof(
        df,
        df_macro[[c for c in macro_cols if c in df_macro.columns]],
        left_on='date_dt',
        right_on='date_dt',
        direction='backward'  # Use most recent available data
    )

    # Clean up temp columns
    df = df.drop(columns=['date', 'date_dt'], errors='ignore')

    # Ensure embi column exists
    if 'embi' not in df.columns:
        df['embi'] = 0.0

    return df


def _safe_float(value) -> float:
    """Safely convert value to float, handling NaN and None."""
    if value is None or pd.isna(value):
        return None
    return float(value)


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['trading@company.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Feature refresh using SSOT CanonicalFeatureBuilder',
    schedule_interval='*/5 13-17 * * 1-5',  # Every 5 min during trading hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l1', 'features', 'ssot', 'production'],
)

with dag:

    def check_trading_day(**context):
        """Branch task to skip processing on holidays/weekends."""
        if should_run_today():
            return 'wait_for_ohlcv'
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
    task_wait_ohlcv = NewOHLCVBarSensor(
        task_id='wait_for_ohlcv',
        table_name='usdcop_m5_ohlcv',
        symbol='USD/COP',
        max_staleness_minutes=10,
        poke_interval=30,
        timeout=300,
        mode='poke',
    )

    # SSOT Feature Computation
    task_compute = PythonOperator(
        task_id='compute_features_ssot',
        python_callable=compute_features_ssot,
        provide_context=True,
    )

    # Feature Validation (CTR-FEAT-001)
    task_validate = PythonOperator(
        task_id='validate_feature_consistency',
        python_callable=validate_feature_consistency,
        provide_context=True,
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
    # Trading day check -> (sensor wait OR skip) -> compute -> validate -> mark
    task_check >> [task_wait_ohlcv, task_skip]
    task_wait_ohlcv >> task_compute >> task_validate >> task_mark_processed
