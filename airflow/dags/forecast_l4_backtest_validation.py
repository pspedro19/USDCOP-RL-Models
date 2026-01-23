"""
Forecast L4: Walk-Forward Backtest Validation
=============================================

Validates forecasting models using walk-forward backtesting.
Compares actual vs predicted prices for historical periods.

Schedule: Weekly (after L5 inference, on Mondays 2:00 AM)
Depends: forecast_l5_01_weekly_inference (previous week predictions)
Triggers: None (validation endpoint)

Metrics Computed:
    - Direction Accuracy (DA): % of correct UP/DOWN predictions
    - RMSE: Root Mean Squared Error of returns
    - MAE: Mean Absolute Error of returns
    - Sharpe Ratio: Risk-adjusted return
    - Profit Factor: Gross profit / Gross loss

Author: Trading Team
Date: 2026-01-22
Contract: CTR-FORECAST-VALIDATION-001
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import from SSOT
from airflow.dags.contracts.dag_registry import (
    FORECAST_L4_BACKTEST_VALIDATION,
    FORECAST_L5_WEEKLY_INFERENCE,
    get_dag_tags,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

BACKTEST_CONFIG = {
    "lookback_weeks": 12,  # Number of weeks to backtest
    "min_samples_per_horizon": 5,  # Minimum samples for valid metrics
    "direction_threshold": 0.0001,  # Min return to count as directional
}


# =============================================================================
# METRIC CALCULATION FUNCTIONS
# =============================================================================

def calculate_direction_accuracy(
    actual_returns: pd.Series,
    predicted_returns: pd.Series,
    threshold: float = 0.0001,
) -> float:
    """Calculate direction accuracy (% correct UP/DOWN predictions)."""
    actual_dir = (actual_returns > threshold).astype(int) - (actual_returns < -threshold).astype(int)
    pred_dir = (predicted_returns > threshold).astype(int) - (predicted_returns < -threshold).astype(int)

    # Only count non-zero predictions
    mask = (pred_dir != 0) & (actual_dir != 0)
    if mask.sum() == 0:
        return 0.5

    return (actual_dir[mask] == pred_dir[mask]).mean()


def calculate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate Root Mean Squared Error."""
    return ((actual - predicted) ** 2).mean() ** 0.5


def calculate_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate Mean Absolute Error."""
    return (actual - predicted).abs().mean()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe Ratio (annualized for daily data)."""
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0.0
    return (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)


def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate Profit Factor (gross profit / gross loss)."""
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 1.0

    return gross_profit / gross_loss


# =============================================================================
# AIRFLOW TASKS
# =============================================================================

def task_load_historical_predictions(**context) -> Dict[str, Any]:
    """Load historical predictions for backtesting."""
    hook = PostgresHook(postgres_conn_id="postgres_default")
    config = BACKTEST_CONFIG

    lookback_date = (datetime.now() - timedelta(weeks=config["lookback_weeks"])).strftime("%Y-%m-%d")

    query = """
    SELECT
        f.inference_date,
        f.model_id,
        f.horizon,
        f.base_price,
        f.predicted_return_pct,
        f.direction,
        f.signal
    FROM bi.fact_forecasts f
    WHERE f.inference_date >= %s
    ORDER BY f.inference_date, f.model_id, f.horizon
    """

    df = hook.get_pandas_df(query, parameters=(lookback_date,))
    logger.info(f"Loaded {len(df)} historical predictions")

    context["ti"].xcom_push(key="predictions", value=df.to_json(orient="records", date_format="iso"))
    return {"predictions_loaded": len(df), "date_range": [lookback_date, datetime.now().strftime("%Y-%m-%d")]}


def task_load_actual_prices(**context) -> Dict[str, Any]:
    """Load actual prices for comparison."""
    hook = PostgresHook(postgres_conn_id="postgres_default")
    config = BACKTEST_CONFIG

    lookback_date = (datetime.now() - timedelta(weeks=config["lookback_weeks"])).strftime("%Y-%m-%d")

    query = """
    SELECT
        date,
        close
    FROM bi.dim_daily_usdcop
    WHERE date >= %s
    ORDER BY date
    """

    df = hook.get_pandas_df(query, parameters=(lookback_date,))
    logger.info(f"Loaded {len(df)} actual prices")

    context["ti"].xcom_push(key="actual_prices", value=df.to_json(orient="records", date_format="iso"))
    return {"prices_loaded": len(df)}


def task_calculate_metrics(**context) -> Dict[str, Any]:
    """Calculate backtest metrics for each model/horizon combination."""
    ti = context["ti"]
    config = BACKTEST_CONFIG

    predictions_json = ti.xcom_pull(task_ids="load_historical_predictions", key="predictions")
    prices_json = ti.xcom_pull(task_ids="load_actual_prices", key="actual_prices")

    predictions = pd.read_json(predictions_json, orient="records")
    prices = pd.read_json(prices_json, orient="records")

    if len(predictions) == 0 or len(prices) == 0:
        logger.warning("No data available for backtesting")
        return {"status": "no_data"}

    predictions["inference_date"] = pd.to_datetime(predictions["inference_date"]).dt.date
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.set_index("date")

    all_metrics = []

    # Group by model and horizon
    for (model_id, horizon), group in predictions.groupby(["model_id", "horizon"]):
        if len(group) < config["min_samples_per_horizon"]:
            continue

        # Calculate actual returns for each prediction
        actual_returns = []
        predicted_returns = []

        for _, row in group.iterrows():
            pred_date = row["inference_date"]
            target_date = pred_date + timedelta(days=int(horizon))

            if target_date in prices.index and pred_date in prices.index:
                actual_return = (prices.loc[target_date, "close"] - prices.loc[pred_date, "close"]) / prices.loc[pred_date, "close"]
                actual_returns.append(actual_return)
                predicted_returns.append(row["predicted_return_pct"])

        if len(actual_returns) < config["min_samples_per_horizon"]:
            continue

        actual_returns = pd.Series(actual_returns)
        predicted_returns = pd.Series(predicted_returns)

        # Calculate metrics
        metrics = {
            "model_id": model_id,
            "horizon": horizon,
            "sample_count": len(actual_returns),
            "direction_accuracy": calculate_direction_accuracy(
                actual_returns, predicted_returns, config["direction_threshold"]
            ),
            "rmse": calculate_rmse(actual_returns, predicted_returns),
            "mae": calculate_mae(actual_returns, predicted_returns),
            "sharpe_ratio": calculate_sharpe_ratio(actual_returns),
            "profit_factor": calculate_profit_factor(actual_returns),
        }

        all_metrics.append(metrics)
        logger.info(f"Metrics for {model_id} H{horizon}: DA={metrics['direction_accuracy']:.2%}")

    ti.xcom_push(key="metrics", value=all_metrics)
    return {"metrics_computed": len(all_metrics)}


def task_persist_metrics(**context) -> Dict[str, Any]:
    """Persist backtest metrics to PostgreSQL."""
    hook = PostgresHook(postgres_conn_id="postgres_default")
    ti = context["ti"]

    metrics = ti.xcom_pull(task_ids="calculate_metrics", key="metrics")
    if not metrics:
        return {"status": "no_metrics"}

    # Ensure table exists
    hook.run("""
        CREATE TABLE IF NOT EXISTS bi.fact_forecast_backtest (
            id SERIAL PRIMARY KEY,
            model_id VARCHAR(100) NOT NULL,
            horizon INT NOT NULL,
            backtest_date DATE NOT NULL,
            sample_count INT,
            direction_accuracy DECIMAL(5,4),
            rmse DECIMAL(10,6),
            mae DECIMAL(10,6),
            sharpe_ratio DECIMAL(10,6),
            profit_factor DECIMAL(10,6),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(model_id, horizon, backtest_date)
        );

        CREATE INDEX IF NOT EXISTS idx_forecast_backtest_date
            ON bi.fact_forecast_backtest (backtest_date DESC);
    """)

    backtest_date = datetime.now().strftime("%Y-%m-%d")
    inserted = 0

    for m in metrics:
        hook.run("""
            INSERT INTO bi.fact_forecast_backtest
            (model_id, horizon, backtest_date, sample_count, direction_accuracy, rmse, mae, sharpe_ratio, profit_factor)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_id, horizon, backtest_date) DO UPDATE SET
                sample_count = EXCLUDED.sample_count,
                direction_accuracy = EXCLUDED.direction_accuracy,
                rmse = EXCLUDED.rmse,
                mae = EXCLUDED.mae,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                profit_factor = EXCLUDED.profit_factor,
                created_at = NOW()
        """, parameters=(
            m["model_id"],
            m["horizon"],
            backtest_date,
            m["sample_count"],
            m["direction_accuracy"],
            m["rmse"],
            m["mae"],
            m["sharpe_ratio"],
            min(m["profit_factor"], 999999),  # Cap infinite values
        ))
        inserted += 1

    logger.info(f"Persisted {inserted} backtest metrics")
    return {"status": "success", "metrics_persisted": inserted}


def task_generate_report(**context) -> Dict[str, Any]:
    """Generate backtest summary report."""
    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="calculate_metrics", key="metrics")

    if not metrics:
        logger.warning("No metrics for report")
        return {"status": "no_metrics"}

    df = pd.DataFrame(metrics)

    logger.info("=" * 60)
    logger.info("FORECASTING BACKTEST VALIDATION REPORT")
    logger.info("=" * 60)

    # Best models per horizon
    for horizon in sorted(df["horizon"].unique()):
        h_df = df[df["horizon"] == horizon]
        best = h_df.loc[h_df["direction_accuracy"].idxmax()]
        logger.info(f"H{horizon}: Best = {best['model_id']} (DA: {best['direction_accuracy']:.1%})")

    # Overall stats
    logger.info("-" * 60)
    logger.info(f"Average Direction Accuracy: {df['direction_accuracy'].mean():.1%}")
    logger.info(f"Average RMSE: {df['rmse'].mean():.6f}")
    logger.info(f"Average Sharpe: {df['sharpe_ratio'].mean():.2f}")
    logger.info("=" * 60)

    return {"report_generated": True, "avg_direction_accuracy": df["direction_accuracy"].mean()}


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    "owner": "forecast-pipeline",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=FORECAST_L4_BACKTEST_VALIDATION,
    default_args=default_args,
    description="Walk-forward backtest validation for forecasting models",
    schedule_interval="0 2 * * 1",  # Mondays 2:00 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=get_dag_tags(FORECAST_L4_BACKTEST_VALIDATION),
    doc_md=__doc__,
) as dag:

    load_predictions = PythonOperator(
        task_id="load_historical_predictions",
        python_callable=task_load_historical_predictions,
    )

    load_prices = PythonOperator(
        task_id="load_actual_prices",
        python_callable=task_load_actual_prices,
    )

    calculate = PythonOperator(
        task_id="calculate_metrics",
        python_callable=task_calculate_metrics,
    )

    persist = PythonOperator(
        task_id="persist_metrics",
        python_callable=task_persist_metrics,
    )

    report = PythonOperator(
        task_id="generate_report",
        python_callable=task_generate_report,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Task dependencies
    [load_predictions, load_prices] >> calculate >> persist >> report
