"""
Forecast L1: Daily Features Engineering
=======================================

Builds forecasting features from daily USDCOP data.
Creates the bi.v_forecasting_features view with all features and targets.

Schedule: Daily at 8:00 AM Colombia time (after L0 completes)
Depends: forecast_l0_01_daily_data
Triggers: forecast_l2_01_dataset_build

Data Lineage:
    bi.dim_daily_usdcop → bi.v_forecasting_features

Features Generated:
    - Price: open, high, low, close
    - Returns: 1d, 5d, 10d, 20d
    - Volatility: 5d, 10d, 20d
    - Technical: RSI-14, MA ratios
    - Calendar: day_of_week, month, is_month_end
    - Targets: 1d, 5d, 10d, 15d, 20d, 25d, 30d ahead

Author: Trading Team
Date: 2026-01-22
Contract: CTR-FORECAST-DATA-001
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sensors.external_task import ExternalTaskSensor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import from SSOT
from airflow.dags.contracts.dag_registry import (
    FORECAST_L0_DAILY_DATA,
    FORECAST_L1_DAILY_FEATURES,
    get_dag_tags,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE VIEW SQL (SSOT)
# =============================================================================

CREATE_FEATURES_VIEW_SQL = """
-- Forecasting features view (computed from daily OHLCV)
-- SSOT for forecasting feature engineering
-- Contract: CTR-FORECAST-DATA-001

DROP VIEW IF EXISTS bi.v_forecasting_features CASCADE;

CREATE VIEW bi.v_forecasting_features AS
WITH price_data AS (
    SELECT
        date,
        open,
        high,
        low,
        close,
        volume,
        -- Returns
        (close - LAG(close, 1) OVER (ORDER BY date)) /
            NULLIF(LAG(close, 1) OVER (ORDER BY date), 0) AS return_1d,
        (close - LAG(close, 5) OVER (ORDER BY date)) /
            NULLIF(LAG(close, 5) OVER (ORDER BY date), 0) AS return_5d,
        (close - LAG(close, 10) OVER (ORDER BY date)) /
            NULLIF(LAG(close, 10) OVER (ORDER BY date), 0) AS return_10d,
        (close - LAG(close, 20) OVER (ORDER BY date)) /
            NULLIF(LAG(close, 20) OVER (ORDER BY date), 0) AS return_20d,
        -- Moving averages
        AVG(close) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma_20d,
        AVG(close) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma_50d,
        -- Calendar
        EXTRACT(DOW FROM date)::INT AS day_of_week,
        EXTRACT(MONTH FROM date)::INT AS month,
        CASE WHEN date = (date_trunc('month', date) + INTERVAL '1 month' - INTERVAL '1 day')::date
             THEN 1 ELSE 0 END AS is_month_end
    FROM bi.dim_daily_usdcop
    WHERE close IS NOT NULL
),
with_volatility AS (
    SELECT
        p.*,
        STDDEV(p.return_1d) OVER (ORDER BY p.date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS volatility_5d,
        STDDEV(p.return_1d) OVER (ORDER BY p.date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS volatility_10d,
        STDDEV(p.return_1d) OVER (ORDER BY p.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS volatility_20d
    FROM price_data p
),
with_rsi AS (
    SELECT
        v.*,
        -- RSI calculation (14-day)
        100 - (100 / (1 + NULLIF(
            AVG(CASE WHEN v.return_1d > 0 THEN v.return_1d ELSE 0 END)
                OVER (ORDER BY v.date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) /
            NULLIF(
                AVG(CASE WHEN v.return_1d < 0 THEN ABS(v.return_1d) ELSE 0.0001 END)
                    OVER (ORDER BY v.date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW),
                0
            ),
            0
        ))) AS rsi_14d
    FROM with_volatility v
)
SELECT
    r.date,
    -- Price features
    r.close,
    r.open,
    r.high,
    r.low,
    -- Returns
    r.return_1d,
    r.return_5d,
    r.return_10d,
    r.return_20d,
    -- Volatility
    r.volatility_5d,
    r.volatility_10d,
    r.volatility_20d,
    -- Technical
    COALESCE(r.rsi_14d, 50) AS rsi_14d,
    r.close / NULLIF(r.ma_20d, 0) AS ma_ratio_20d,
    r.close / NULLIF(r.ma_50d, 0) AS ma_ratio_50d,
    -- Calendar
    r.day_of_week,
    r.month,
    r.is_month_end,
    -- Target columns (future prices for each horizon)
    LEAD(r.close, 1) OVER (ORDER BY r.date) AS target_1d,
    LEAD(r.close, 5) OVER (ORDER BY r.date) AS target_5d,
    LEAD(r.close, 10) OVER (ORDER BY r.date) AS target_10d,
    LEAD(r.close, 15) OVER (ORDER BY r.date) AS target_15d,
    LEAD(r.close, 20) OVER (ORDER BY r.date) AS target_20d,
    LEAD(r.close, 25) OVER (ORDER BY r.date) AS target_25d,
    LEAD(r.close, 30) OVER (ORDER BY r.date) AS target_30d,
    -- Target returns (what models actually predict)
    (LEAD(r.close, 1) OVER (ORDER BY r.date) - r.close) / NULLIF(r.close, 0) AS target_return_1d,
    (LEAD(r.close, 5) OVER (ORDER BY r.date) - r.close) / NULLIF(r.close, 0) AS target_return_5d,
    (LEAD(r.close, 10) OVER (ORDER BY r.date) - r.close) / NULLIF(r.close, 0) AS target_return_10d,
    (LEAD(r.close, 15) OVER (ORDER BY r.date) - r.close) / NULLIF(r.close, 0) AS target_return_15d,
    (LEAD(r.close, 20) OVER (ORDER BY r.date) - r.close) / NULLIF(r.close, 0) AS target_return_20d,
    (LEAD(r.close, 25) OVER (ORDER BY r.date) - r.close) / NULLIF(r.close, 0) AS target_return_25d,
    (LEAD(r.close, 30) OVER (ORDER BY r.date) - r.close) / NULLIF(r.close, 0) AS target_return_30d
FROM with_rsi r
WHERE r.date >= '2015-01-01'
  AND r.close IS NOT NULL
ORDER BY r.date;

-- Add comment
COMMENT ON VIEW bi.v_forecasting_features IS
    'Forecasting features for ML models. Contract: CTR-FORECAST-DATA-001 v1.0.0';
"""

# Feature statistics materialized table
CREATE_FEATURE_STATS_SQL = """
-- Feature statistics for normalization
DROP TABLE IF EXISTS bi.forecast_feature_stats CASCADE;

CREATE TABLE bi.forecast_feature_stats (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    mean DECIMAL(20,10),
    std DECIMAL(20,10),
    min_val DECIMAL(20,10),
    max_val DECIMAL(20,10),
    q25 DECIMAL(20,10),
    q50 DECIMAL(20,10),
    q75 DECIMAL(20,10),
    null_count INT DEFAULT 0,
    total_count INT DEFAULT 0,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(feature_name)
);

-- Insert statistics for each feature
INSERT INTO bi.forecast_feature_stats
(feature_name, mean, std, min_val, max_val, null_count, total_count)
SELECT
    'close' as feature_name,
    AVG(close) as mean,
    STDDEV(close) as std,
    MIN(close) as min_val,
    MAX(close) as max_val,
    SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_count,
    COUNT(*) as total_count
FROM bi.v_forecasting_features
ON CONFLICT (feature_name) DO UPDATE SET
    mean = EXCLUDED.mean,
    std = EXCLUDED.std,
    min_val = EXCLUDED.min_val,
    max_val = EXCLUDED.max_val,
    null_count = EXCLUDED.null_count,
    total_count = EXCLUDED.total_count,
    computed_at = NOW();

-- Add more feature statistics
INSERT INTO bi.forecast_feature_stats
(feature_name, mean, std, min_val, max_val, null_count, total_count)
SELECT
    'return_1d' as feature_name,
    AVG(return_1d) as mean,
    STDDEV(return_1d) as std,
    MIN(return_1d) as min_val,
    MAX(return_1d) as max_val,
    SUM(CASE WHEN return_1d IS NULL THEN 1 ELSE 0 END) as null_count,
    COUNT(*) as total_count
FROM bi.v_forecasting_features
ON CONFLICT (feature_name) DO UPDATE SET
    mean = EXCLUDED.mean,
    std = EXCLUDED.std,
    min_val = EXCLUDED.min_val,
    max_val = EXCLUDED.max_val,
    null_count = EXCLUDED.null_count,
    total_count = EXCLUDED.total_count,
    computed_at = NOW();

INSERT INTO bi.forecast_feature_stats
(feature_name, mean, std, min_val, max_val, null_count, total_count)
SELECT
    'volatility_20d' as feature_name,
    AVG(volatility_20d) as mean,
    STDDEV(volatility_20d) as std,
    MIN(volatility_20d) as min_val,
    MAX(volatility_20d) as max_val,
    SUM(CASE WHEN volatility_20d IS NULL THEN 1 ELSE 0 END) as null_count,
    COUNT(*) as total_count
FROM bi.v_forecasting_features
ON CONFLICT (feature_name) DO UPDATE SET
    mean = EXCLUDED.mean,
    std = EXCLUDED.std,
    min_val = EXCLUDED.min_val,
    max_val = EXCLUDED.max_val,
    null_count = EXCLUDED.null_count,
    total_count = EXCLUDED.total_count,
    computed_at = NOW();

INSERT INTO bi.forecast_feature_stats
(feature_name, mean, std, min_val, max_val, null_count, total_count)
SELECT
    'rsi_14d' as feature_name,
    AVG(rsi_14d) as mean,
    STDDEV(rsi_14d) as std,
    MIN(rsi_14d) as min_val,
    MAX(rsi_14d) as max_val,
    SUM(CASE WHEN rsi_14d IS NULL THEN 1 ELSE 0 END) as null_count,
    COUNT(*) as total_count
FROM bi.v_forecasting_features
ON CONFLICT (feature_name) DO UPDATE SET
    mean = EXCLUDED.mean,
    std = EXCLUDED.std,
    min_val = EXCLUDED.min_val,
    max_val = EXCLUDED.max_val,
    null_count = EXCLUDED.null_count,
    total_count = EXCLUDED.total_count,
    computed_at = NOW();
"""


# =============================================================================
# AIRFLOW TASKS
# =============================================================================

def task_create_features_view(**context) -> Dict[str, Any]:
    """Create or replace the forecasting features view."""
    hook = PostgresHook(postgres_conn_id="postgres_default")

    logger.info("Creating forecasting features view...")
    hook.run(CREATE_FEATURES_VIEW_SQL)

    # Verify view was created
    verify_query = """
    SELECT COUNT(*) FROM bi.v_forecasting_features
    """
    result = hook.get_first(verify_query)
    row_count = result[0] if result else 0

    logger.info(f"Features view created with {row_count} rows")
    return {"view_created": True, "row_count": row_count}


def task_compute_feature_stats(**context) -> Dict[str, Any]:
    """Compute feature statistics for normalization."""
    hook = PostgresHook(postgres_conn_id="postgres_default")

    logger.info("Computing feature statistics...")
    hook.run(CREATE_FEATURE_STATS_SQL)

    # Get stats count
    count_query = "SELECT COUNT(*) FROM bi.forecast_feature_stats"
    result = hook.get_first(count_query)
    stats_count = result[0] if result else 0

    logger.info(f"Computed statistics for {stats_count} features")
    return {"stats_computed": True, "feature_count": stats_count}


def task_validate_features(**context) -> Dict[str, Any]:
    """Validate feature quality and completeness."""
    hook = PostgresHook(postgres_conn_id="postgres_default")

    # Check for nulls in critical features
    null_check_query = """
    SELECT
        COUNT(*) as total_rows,
        SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
        SUM(CASE WHEN return_1d IS NULL THEN 1 ELSE 0 END) as null_return_1d,
        SUM(CASE WHEN target_1d IS NULL THEN 1 ELSE 0 END) as null_target_1d,
        MIN(date) as first_date,
        MAX(date) as last_date
    FROM bi.v_forecasting_features
    """

    result = hook.get_first(null_check_query)

    validation_result = {
        "total_rows": result[0] if result else 0,
        "null_close": result[1] if result else 0,
        "null_return_1d": result[2] if result else 0,
        "null_target_1d": result[3] if result else 0,
        "first_date": str(result[4]) if result and result[4] else None,
        "last_date": str(result[5]) if result and result[5] else None,
    }

    # Calculate data quality score
    total = validation_result["total_rows"]
    if total > 0:
        null_pct = (
            validation_result["null_close"] +
            validation_result["null_return_1d"]
        ) / (total * 2) * 100
        validation_result["quality_score"] = 100 - null_pct
    else:
        validation_result["quality_score"] = 0

    validation_result["validation_passed"] = validation_result["quality_score"] >= 95

    logger.info(f"Feature validation: {validation_result}")
    return validation_result


def task_export_feature_snapshot(**context) -> Dict[str, Any]:
    """Export latest features to CSV for backup/audit."""
    hook = PostgresHook(postgres_conn_id="postgres_default")

    # Get last 30 days for snapshot
    export_query = """
    SELECT *
    FROM bi.v_forecasting_features
    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    ORDER BY date DESC
    """

    df = hook.get_pandas_df(export_query)

    # Save to file
    snapshot_path = PROJECT_ROOT / "data" / "forecasting" / "feature_snapshots"
    snapshot_path.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    filename = snapshot_path / f"features_snapshot_{today}.csv"

    df.to_csv(filename, index=False)
    logger.info(f"Exported {len(df)} rows to {filename}")

    return {
        "exported": True,
        "rows": len(df),
        "path": str(filename),
    }


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    "owner": "forecast-pipeline",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=FORECAST_L1_DAILY_FEATURES,
    default_args=default_args,
    description="Build forecasting features from daily USDCOP data",
    schedule_interval="0 8 * * 1-5",  # 8 AM on weekdays (after L0)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=get_dag_tags(FORECAST_L1_DAILY_FEATURES),
    doc_md=__doc__,
) as dag:

    # Wait for L0 to complete
    wait_for_data = ExternalTaskSensor(
        task_id="wait_for_daily_data",
        external_dag_id=FORECAST_L0_DAILY_DATA,
        external_task_id="validate_data",
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        poke_interval=60,
        timeout=3600,
        mode="reschedule",
    )

    create_view = PythonOperator(
        task_id="create_features_view",
        python_callable=task_create_features_view,
    )

    compute_stats = PythonOperator(
        task_id="compute_feature_stats",
        python_callable=task_compute_feature_stats,
    )

    validate = PythonOperator(
        task_id="validate_features",
        python_callable=task_validate_features,
    )

    export = PythonOperator(
        task_id="export_feature_snapshot",
        python_callable=task_export_feature_snapshot,
    )

    # Task dependencies
    wait_for_data >> create_view >> compute_stats >> validate >> export
