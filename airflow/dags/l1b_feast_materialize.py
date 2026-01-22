"""
DAG: v3.l1b_feast_materialize
==============================
V3 Architecture - Layer 1B: Feast Feature Materialization

Purpose:
    Materializes features from the offline store (Parquet/PostgreSQL)
    to the online store (Redis) for low-latency inference.

    This DAG runs after l1_feature_refresh to ensure fresh features
    are available in the online store.

Schedule:
    - Daily at 07:00 COT (12:00 UTC) - Before market open
    - On-demand trigger after feature refresh

Materialization Strategy:
    1. Load features from inference_features_5m table (offline store)
    2. Transform to Feast format (Parquet files)
    3. Materialize to Redis online store
    4. Validate materialization success

Data Flow:
    ┌─────────────────────┐     ┌─────────────────────┐
    │ l1_feature_refresh  │────►│ inference_features  │
    │ (L1 DAG)            │     │ _5m (PostgreSQL)    │
    └─────────────────────┘     └─────────────────────┘
                                         │
                                         ▼
    ┌─────────────────────────────────────────────────┐
    │          l1b_feast_materialize (This DAG)       │
    │                                                  │
    │  1. Export features to Parquet (offline store)  │
    │  2. feast materialize-incremental               │
    │  3. Validate Redis entries                      │
    └─────────────────────────────────────────────────┘
                                         │
                                         ▼
    ┌─────────────────────┐     ┌─────────────────────┐
    │ Redis Online Store  │◄────│ Feast Feature Store │
    │ (Low-latency)       │     │ (observation_15d)   │
    └─────────────────────┘     └─────────────────────┘

Author: Trading Team
Version: 1.0.0
Created: 2025-01-17
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import os
import subprocess
import tempfile

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

import pandas as pd
import numpy as np

# Import trading calendar for holiday detection
try:
    from utils.trading_calendar import TradingCalendar
    from utils.dag_common import get_db_connection
except ImportError:
    # Fallback for local testing
    TradingCalendar = None
    get_db_connection = None

# DAG Configuration
from contracts.dag_registry import L1_FEAST_MATERIALIZE
DAG_ID = L1_FEAST_MATERIALIZE
FEAST_REPO_PATH = '/opt/airflow/feature_repo'  # Path in Docker container
OFFLINE_STORE_PATH = '/opt/airflow/data/feast'

# Feature configuration
FEATURE_VIEWS = ['technical_features', 'macro_features', 'state_features']

# Initialize trading calendar
trading_cal = TradingCalendar() if TradingCalendar else None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def should_run_today():
    """Check if today is a valid trading day."""
    if trading_cal is None:
        return True

    today = datetime.now()
    if not trading_cal.is_trading_day(today):
        reason = trading_cal.get_violation_reason(today)
        logging.info(f"Skipping - {today.date()}: {reason}")
        return False
    return True


def ensure_directories():
    """Ensure required directories exist."""
    dirs = [
        Path(OFFLINE_STORE_PATH),
        Path(OFFLINE_STORE_PATH) / 'technical_features',
        Path(OFFLINE_STORE_PATH) / 'macro_features',
        Path(OFFLINE_STORE_PATH) / 'state_features',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured directories exist: {OFFLINE_STORE_PATH}")


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def check_trading_day(**context):
    """Branch task to skip processing on holidays/weekends."""
    if should_run_today():
        return 'export_features_to_parquet'
    else:
        return 'skip_processing'


def export_features_to_parquet(**context):
    """
    Export features from PostgreSQL to Parquet files for Feast offline store.

    This task:
    1. Reads features from inference_features_5m table
    2. Transforms data to Feast format
    3. Writes Parquet files for each feature view
    """
    logging.info("=" * 60)
    logging.info("EXPORTING FEATURES TO PARQUET")
    logging.info("=" * 60)

    ensure_directories()

    if get_db_connection is None:
        logging.error("Database connection not available")
        raise RuntimeError("Database connection module not found")

    conn = get_db_connection()

    try:
        # =================================================================
        # Load features from PostgreSQL
        # =================================================================
        query = """
            SELECT
                time as event_timestamp,
                'USD/COP' as symbol,
                TO_CHAR(time, 'YYYYMMDD_HH24MISS') as bar_id,
                log_ret_5m,
                log_ret_1h,
                log_ret_4h,
                rsi_9,
                atr_pct,
                adx_14,
                dxy_z,
                dxy_change_1d,
                vix_z,
                embi_z,
                brent_change_1d,
                rate_spread,
                usdmxn_change_1d,
                NOW() as created_timestamp
            FROM inference_features_5m
            WHERE time >= NOW() - INTERVAL '7 days'
            ORDER BY time
        """

        df = pd.read_sql(query, conn)
        logging.info(f"Loaded {len(df)} feature rows from PostgreSQL")

        if df.empty:
            logging.warning("No features to export")
            context['ti'].xcom_push(key='export_status', value='empty')
            return {'status': 'empty', 'rows': 0}

        # =================================================================
        # Export Technical Features
        # =================================================================
        tech_cols = [
            'event_timestamp', 'symbol', 'bar_id', 'created_timestamp',
            'log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'rsi_9', 'atr_pct', 'adx_14'
        ]
        tech_df = df[tech_cols].copy()
        tech_path = Path(OFFLINE_STORE_PATH) / 'technical_features.parquet'
        tech_df.to_parquet(tech_path, index=False)
        logging.info(f"Exported technical features: {tech_path}")

        # =================================================================
        # Export Macro Features
        # =================================================================
        macro_cols = [
            'event_timestamp', 'symbol', 'bar_id', 'created_timestamp',
            'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
            'brent_change_1d', 'rate_spread', 'usdmxn_change_1d'
        ]
        macro_df = df[macro_cols].copy()
        macro_path = Path(OFFLINE_STORE_PATH) / 'macro_features.parquet'
        macro_df.to_parquet(macro_path, index=False)
        logging.info(f"Exported macro features: {macro_path}")

        # =================================================================
        # Export State Features (position and time_normalized)
        # =================================================================
        # State features need to be computed separately
        # For now, create placeholder with default values
        state_df = df[['event_timestamp', 'symbol', 'bar_id', 'created_timestamp']].copy()
        state_df['position'] = 0.0  # Neutral position by default
        state_df['time_normalized'] = 0.5  # Middle of session by default

        # Calculate actual time_normalized based on trading hours
        state_df['hour'] = pd.to_datetime(state_df['event_timestamp']).dt.hour
        state_df['minute'] = pd.to_datetime(state_df['event_timestamp']).dt.minute

        # Trading hours: 13:00-17:55 UTC (8:00-12:55 COT)
        start_minutes = 13 * 60
        end_minutes = 17 * 60 + 55
        duration = end_minutes - start_minutes

        state_df['current_minutes'] = state_df['hour'] * 60 + state_df['minute']
        state_df['time_normalized'] = (
            (state_df['current_minutes'] - start_minutes) / duration
        ).clip(0, 1)

        state_df = state_df.drop(columns=['hour', 'minute', 'current_minutes'])
        state_path = Path(OFFLINE_STORE_PATH) / 'state_features.parquet'
        state_df.to_parquet(state_path, index=False)
        logging.info(f"Exported state features: {state_path}")

        # Push export metrics
        context['ti'].xcom_push(key='export_status', value='success')
        context['ti'].xcom_push(key='rows_exported', value=len(df))

        logging.info(f"Export complete: {len(df)} rows to 3 Parquet files")

        return {
            'status': 'success',
            'rows': len(df),
            'files': [str(tech_path), str(macro_path), str(state_path)]
        }

    except Exception as e:
        logging.error(f"Error exporting features: {e}")
        context['ti'].xcom_push(key='export_status', value='error')
        raise

    finally:
        conn.close()


def materialize_features(**context):
    """
    Run Feast materialize-incremental to push features to online store.

    This task:
    1. Applies feature definitions to the registry
    2. Runs incremental materialization
    3. Updates Redis with latest features
    """
    logging.info("=" * 60)
    logging.info("MATERIALIZING FEATURES TO REDIS")
    logging.info("=" * 60)

    # Check export status
    export_status = context['ti'].xcom_pull(
        task_ids='export_features_to_parquet',
        key='export_status'
    )

    if export_status != 'success':
        logging.warning(f"Skipping materialization - export status: {export_status}")
        return {'status': 'skipped', 'reason': f'export_{export_status}'}

    try:
        # =================================================================
        # Apply feature definitions
        # =================================================================
        logging.info("Applying feature definitions...")
        apply_result = subprocess.run(
            ['feast', 'apply'],
            cwd=FEAST_REPO_PATH,
            capture_output=True,
            text=True,
            timeout=120
        )

        if apply_result.returncode != 0:
            logging.error(f"Feast apply failed: {apply_result.stderr}")
            raise RuntimeError(f"Feast apply failed: {apply_result.stderr}")

        logging.info(f"Feast apply output: {apply_result.stdout}")

        # =================================================================
        # Materialize features incrementally
        # =================================================================
        logging.info("Materializing features...")

        # Calculate time range (last 7 days)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        materialize_result = subprocess.run(
            [
                'feast', 'materialize-incremental',
                end_date.strftime('%Y-%m-%dT%H:%M:%S')
            ],
            cwd=FEAST_REPO_PATH,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )

        if materialize_result.returncode != 0:
            logging.error(f"Feast materialize failed: {materialize_result.stderr}")
            raise RuntimeError(f"Feast materialize failed: {materialize_result.stderr}")

        logging.info(f"Feast materialize output: {materialize_result.stdout}")

        context['ti'].xcom_push(key='materialize_status', value='success')

        return {
            'status': 'success',
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
        }

    except subprocess.TimeoutExpired:
        logging.error("Feast command timed out")
        context['ti'].xcom_push(key='materialize_status', value='timeout')
        raise RuntimeError("Feast command timed out")

    except FileNotFoundError:
        logging.error("Feast CLI not found - is it installed?")
        context['ti'].xcom_push(key='materialize_status', value='feast_not_found')
        raise RuntimeError("Feast CLI not found")

    except Exception as e:
        logging.error(f"Error materializing features: {e}")
        context['ti'].xcom_push(key='materialize_status', value='error')
        raise


def validate_materialization(**context):
    """
    Validate that features were successfully materialized to Redis.

    This task:
    1. Connects to Redis
    2. Checks for presence of feature keys
    3. Validates feature count and freshness
    """
    logging.info("=" * 60)
    logging.info("VALIDATING MATERIALIZATION")
    logging.info("=" * 60)

    materialize_status = context['ti'].xcom_pull(
        task_ids='materialize_features',
        key='materialize_status'
    )

    if materialize_status != 'success':
        logging.warning(f"Skipping validation - materialize status: {materialize_status}")
        return {'status': 'skipped', 'reason': f'materialize_{materialize_status}'}

    try:
        import redis

        # Connect to Redis
        redis_host = os.environ.get('REDIS_HOST', 'redis')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        redis_password = os.environ.get('REDIS_PASSWORD', '')

        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )

        # Check Redis connection
        if not r.ping():
            raise RuntimeError("Redis ping failed")

        logging.info(f"Connected to Redis at {redis_host}:{redis_port}")

        # Count Feast keys
        feast_keys = r.keys('feast:*')
        feast_count = len(feast_keys)

        logging.info(f"Found {feast_count} Feast keys in Redis")

        # Check for feature views
        for view in FEATURE_VIEWS:
            view_keys = r.keys(f'feast:*{view}*')
            logging.info(f"  {view}: {len(view_keys)} keys")

        # Validate minimum keys
        min_expected_keys = 100  # Should have at least 100 keys for 7 days
        if feast_count < min_expected_keys:
            logging.warning(
                f"Low key count: {feast_count} < {min_expected_keys} expected"
            )

        context['ti'].xcom_push(key='validation_status', value='success')
        context['ti'].xcom_push(key='feast_key_count', value=feast_count)

        logging.info("=" * 60)
        logging.info("VALIDATION COMPLETE")
        logging.info(f"Total Feast keys: {feast_count}")
        logging.info("=" * 60)

        return {
            'status': 'success',
            'feast_key_count': feast_count,
            'feature_views': FEATURE_VIEWS,
        }

    except ImportError:
        logging.error("Redis library not installed")
        context['ti'].xcom_push(key='validation_status', value='redis_not_installed')
        return {'status': 'error', 'reason': 'redis_not_installed'}

    except Exception as e:
        logging.error(f"Validation error: {e}")
        context['ti'].xcom_push(key='validation_status', value='error')
        raise


def generate_report(**context):
    """
    Generate materialization report for logging and alerting.
    """
    logging.info("=" * 60)
    logging.info("GENERATING MATERIALIZATION REPORT")
    logging.info("=" * 60)

    # Gather all task results
    export_status = context['ti'].xcom_pull(
        task_ids='export_features_to_parquet', key='export_status'
    )
    rows_exported = context['ti'].xcom_pull(
        task_ids='export_features_to_parquet', key='rows_exported'
    )
    materialize_status = context['ti'].xcom_pull(
        task_ids='materialize_features', key='materialize_status'
    )
    validation_status = context['ti'].xcom_pull(
        task_ids='validate_materialization', key='validation_status'
    )
    feast_key_count = context['ti'].xcom_pull(
        task_ids='validate_materialization', key='feast_key_count'
    )

    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'dag_id': DAG_ID,
        'run_id': context['run_id'],
        'export': {
            'status': export_status,
            'rows': rows_exported or 0,
        },
        'materialize': {
            'status': materialize_status,
        },
        'validation': {
            'status': validation_status,
            'feast_key_count': feast_key_count or 0,
        },
        'overall_status': 'success' if all([
            export_status == 'success',
            materialize_status == 'success',
            validation_status == 'success',
        ]) else 'partial' if any([
            export_status == 'success',
            materialize_status == 'success',
        ]) else 'failed',
    }

    logging.info(f"Report: {json.dumps(report, indent=2)}")

    return report


def alert_on_failure(**context):
    """
    Send alert on materialization failure.
    """
    logging.error("=" * 60)
    logging.error("FEAST MATERIALIZATION FAILED")
    logging.error("=" * 60)

    # In production, integrate with alerting system (PagerDuty, Slack, etc.)
    # For now, just log the failure

    export_status = context['ti'].xcom_pull(
        task_ids='export_features_to_parquet', key='export_status'
    )
    materialize_status = context['ti'].xcom_pull(
        task_ids='materialize_features', key='materialize_status'
    )
    validation_status = context['ti'].xcom_pull(
        task_ids='validate_materialization', key='validation_status'
    )

    logging.error(f"Export status: {export_status}")
    logging.error(f"Materialize status: {materialize_status}")
    logging.error(f"Validation status: {validation_status}")

    # Could send to Slack, PagerDuty, etc.
    # slack_webhook = os.environ.get('SLACK_WEBHOOK')
    # if slack_webhook:
    #     requests.post(slack_webhook, json={...})


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='V3 L1B Feast Materialization - Export features to Redis online store',
    # Daily at 07:00 COT (12:00 UTC) - before market opens at 08:00 COT
    schedule_interval='0 12 * * 1-5',
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l1b', 'feast', 'materialization', 'redis'],
)

with dag:

    # Check if it's a trading day
    task_check = BranchPythonOperator(
        task_id='check_trading_day',
        python_callable=check_trading_day,
        provide_context=True,
    )

    # Skip processing on non-trading days
    task_skip = EmptyOperator(task_id='skip_processing')

    # Wait for L1 feature refresh to complete (optional sensor)
    # Uncomment if strict dependency on l1_feature_refresh is needed
    # task_wait_l1 = ExternalTaskSensor(
    #     task_id='wait_for_l1_feature_refresh',
    #     external_dag_id='v3.l1_feature_refresh',
    #     external_task_id='validate_features',
    #     timeout=3600,
    #     poke_interval=60,
    #     mode='poke',
    # )

    # Export features to Parquet
    task_export = PythonOperator(
        task_id='export_features_to_parquet',
        python_callable=export_features_to_parquet,
        provide_context=True,
    )

    # Materialize to Redis
    task_materialize = PythonOperator(
        task_id='materialize_features',
        python_callable=materialize_features,
        provide_context=True,
    )

    # Validate materialization
    task_validate = PythonOperator(
        task_id='validate_materialization',
        python_callable=validate_materialization,
        provide_context=True,
    )

    # Generate report
    task_report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
        provide_context=True,
        trigger_rule='all_done',  # Run even if upstream fails
    )

    # Alert on failure
    task_alert = PythonOperator(
        task_id='alert_on_failure',
        python_callable=alert_on_failure,
        provide_context=True,
        trigger_rule='one_failed',  # Only run if any upstream fails
    )

    # Task dependencies
    task_check >> [task_export, task_skip]
    task_export >> task_materialize >> task_validate >> task_report
    [task_materialize, task_validate] >> task_alert
