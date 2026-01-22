"""
Automated Drift-Triggered Retraining DAG
========================================

Monitors feature drift and model performance degradation, automatically
triggering model retraining when thresholds are exceeded.

P1: Automated Retraining Trigger

Triggers:
- Drift severity >= HIGH for 24 hours
- Agreement rate < 0.7 for 7 days
- Sharpe ratio decrease > 20% from baseline

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
from airflow.utils.state import State

logger = logging.getLogger(__name__)

from contracts.dag_registry import L2_DRIFT_RETRAIN, L3_MODEL_TRAINING

# =============================================================================
# Configuration
# =============================================================================

# Retraining trigger thresholds
DRIFT_SEVERITY_THRESHOLD = "HIGH"  # Minimum drift severity to trigger
DRIFT_DURATION_HOURS = 24  # How long drift must persist
AGREEMENT_RATE_THRESHOLD = 0.70  # Minimum model agreement rate
AGREEMENT_DURATION_DAYS = 7  # How long low agreement must persist
SHARPE_DECREASE_THRESHOLD = 0.20  # 20% decrease from baseline
COOLDOWN_HOURS = 24  # Minimum time between retraining runs

# DAG configuration
default_args = {
    "owner": "trading-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# =============================================================================
# Task Functions
# =============================================================================

def check_drift_status(**context) -> Dict[str, Any]:
    """
    Check current drift status from monitoring tables.

    Returns:
        Dictionary with drift metrics
    """
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")

    query = """
    SELECT
        feature_name,
        ks_statistic,
        p_value,
        drift_severity,
        checked_at
    FROM monitoring.feature_drift_results
    WHERE checked_at >= NOW() - INTERVAL '%s hours'
    ORDER BY checked_at DESC
    """

    results = pg_hook.get_records(query, parameters=(DRIFT_DURATION_HOURS,))

    # Analyze drift patterns
    high_drift_features = []
    severe_drift_duration = 0

    for row in results:
        feature_name, ks_stat, p_value, severity, checked_at = row
        if severity in ["HIGH", "CRITICAL"]:
            high_drift_features.append(feature_name)

    # Check if drift has persisted
    drift_query = """
    SELECT
        COUNT(DISTINCT date_trunc('hour', checked_at)) as drift_hours
    FROM monitoring.feature_drift_results
    WHERE drift_severity IN ('HIGH', 'CRITICAL')
    AND checked_at >= NOW() - INTERVAL '%s hours'
    """

    drift_hours = pg_hook.get_first(drift_query, parameters=(DRIFT_DURATION_HOURS,))[0]

    metrics = {
        "high_drift_features": list(set(high_drift_features)),
        "high_drift_feature_count": len(set(high_drift_features)),
        "drift_hours": drift_hours or 0,
        "drift_threshold_exceeded": (drift_hours or 0) >= DRIFT_DURATION_HOURS * 0.8,
        "timestamp": datetime.utcnow().isoformat(),
    }

    context["ti"].xcom_push(key="drift_metrics", value=metrics)
    logger.info(f"Drift metrics: {metrics}")

    return metrics


def check_model_agreement(**context) -> Dict[str, Any]:
    """
    Check model agreement rate from shadow trading comparison.

    Returns:
        Dictionary with agreement metrics
    """
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")

    query = """
    SELECT
        AVG(CASE WHEN primary_action = shadow_action THEN 1.0 ELSE 0.0 END) as agreement_rate,
        COUNT(*) as total_comparisons,
        COUNT(DISTINCT DATE(created_at)) as days_compared
    FROM monitoring.shadow_trading_comparison
    WHERE created_at >= NOW() - INTERVAL '%s days'
    """

    result = pg_hook.get_first(query, parameters=(AGREEMENT_DURATION_DAYS,))

    agreement_rate, total_comparisons, days_compared = result or (1.0, 0, 0)

    metrics = {
        "agreement_rate": float(agreement_rate) if agreement_rate else 1.0,
        "total_comparisons": total_comparisons or 0,
        "days_compared": days_compared or 0,
        "threshold_exceeded": (agreement_rate or 1.0) < AGREEMENT_RATE_THRESHOLD,
        "timestamp": datetime.utcnow().isoformat(),
    }

    context["ti"].xcom_push(key="agreement_metrics", value=metrics)
    logger.info(f"Agreement metrics: {metrics}")

    return metrics


def check_performance_degradation(**context) -> Dict[str, Any]:
    """
    Check if model performance (Sharpe ratio) has degraded significantly.

    Returns:
        Dictionary with performance metrics
    """
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")

    # Get baseline Sharpe ratio (from model registration or historical best)
    baseline_query = """
    SELECT sharpe_ratio
    FROM mlops.model_registry
    WHERE status = 'production'
    ORDER BY registered_at DESC
    LIMIT 1
    """

    baseline_result = pg_hook.get_first(baseline_query)
    baseline_sharpe = float(baseline_result[0]) if baseline_result else 1.5

    # Get current rolling Sharpe ratio
    current_query = """
    SELECT
        AVG(daily_sharpe) as avg_sharpe,
        STDDEV(daily_sharpe) as sharpe_std,
        COUNT(*) as days
    FROM (
        SELECT
            DATE(timestamp) as trade_date,
            (AVG(return_pct) / NULLIF(STDDEV(return_pct), 0)) * SQRT(252) as daily_sharpe
        FROM trading.paper_trades
        WHERE timestamp >= NOW() - INTERVAL '30 days'
        GROUP BY DATE(timestamp)
    ) daily_stats
    """

    current_result = pg_hook.get_first(current_query)
    current_sharpe = float(current_result[0]) if current_result and current_result[0] else baseline_sharpe

    # Calculate degradation
    degradation = (baseline_sharpe - current_sharpe) / baseline_sharpe if baseline_sharpe != 0 else 0

    metrics = {
        "baseline_sharpe": baseline_sharpe,
        "current_sharpe": current_sharpe,
        "degradation_pct": degradation * 100,
        "threshold_exceeded": degradation > SHARPE_DECREASE_THRESHOLD,
        "timestamp": datetime.utcnow().isoformat(),
    }

    context["ti"].xcom_push(key="performance_metrics", value=metrics)
    logger.info(f"Performance metrics: {metrics}")

    return metrics


def check_cooldown(**context) -> bool:
    """
    Check if we're within the retraining cooldown period.

    Returns:
        True if cooldown has passed, False if still in cooldown
    """
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")

    query = """
    SELECT MAX(started_at)
    FROM mlops.training_runs
    WHERE status IN ('completed', 'running')
    """

    result = pg_hook.get_first(query)
    last_training = result[0] if result else None

    if last_training is None:
        return True

    hours_since_training = (datetime.utcnow() - last_training).total_seconds() / 3600
    cooldown_passed = hours_since_training >= COOLDOWN_HOURS

    context["ti"].xcom_push(key="cooldown_passed", value=cooldown_passed)
    context["ti"].xcom_push(key="hours_since_training", value=hours_since_training)

    logger.info(f"Cooldown check: {hours_since_training:.1f}h since last training, "
                f"cooldown {'passed' if cooldown_passed else 'active'}")

    return cooldown_passed


def evaluate_retraining_need(**context) -> str:
    """
    Evaluate all metrics and decide whether to trigger retraining.

    Returns:
        Task ID to branch to: 'trigger_retraining' or 'skip_retraining'
    """
    ti = context["ti"]

    # Get all metrics
    drift_metrics = ti.xcom_pull(task_ids="check_drift_status", key="drift_metrics")
    agreement_metrics = ti.xcom_pull(task_ids="check_model_agreement", key="agreement_metrics")
    performance_metrics = ti.xcom_pull(task_ids="check_performance_degradation", key="performance_metrics")
    cooldown_passed = ti.xcom_pull(task_ids="check_cooldown", key="cooldown_passed")

    # Check if any threshold is exceeded
    reasons = []

    if drift_metrics and drift_metrics.get("drift_threshold_exceeded"):
        reasons.append(f"High drift for {drift_metrics['drift_hours']}+ hours")

    if agreement_metrics and agreement_metrics.get("threshold_exceeded"):
        reasons.append(f"Agreement rate {agreement_metrics['agreement_rate']:.2%} < {AGREEMENT_RATE_THRESHOLD:.0%}")

    if performance_metrics and performance_metrics.get("threshold_exceeded"):
        reasons.append(f"Sharpe degradation {performance_metrics['degradation_pct']:.1f}%")

    # Store decision context
    decision = {
        "should_retrain": len(reasons) > 0 and cooldown_passed,
        "reasons": reasons,
        "cooldown_passed": cooldown_passed,
        "evaluated_at": datetime.utcnow().isoformat(),
    }

    ti.xcom_push(key="retraining_decision", value=decision)

    if decision["should_retrain"]:
        logger.warning(f"Retraining triggered! Reasons: {', '.join(reasons)}")
        return "trigger_retraining"
    else:
        if not cooldown_passed:
            logger.info("Retraining needed but still in cooldown period")
        else:
            logger.info("No retraining needed - all metrics within thresholds")
        return "skip_retraining"


def log_retraining_trigger(**context) -> None:
    """Log the retraining trigger decision to database."""
    ti = context["ti"]
    decision = ti.xcom_pull(task_ids="evaluate_retraining_need", key="retraining_decision")

    pg_hook = PostgresHook(postgres_conn_id="postgres_default")

    query = """
    INSERT INTO mlops.retraining_triggers (
        triggered_at,
        reasons,
        drift_metrics,
        agreement_metrics,
        performance_metrics,
        decision
    ) VALUES (
        NOW(),
        %s,
        %s,
        %s,
        %s,
        %s
    )
    """

    drift_metrics = ti.xcom_pull(task_ids="check_drift_status", key="drift_metrics")
    agreement_metrics = ti.xcom_pull(task_ids="check_model_agreement", key="agreement_metrics")
    performance_metrics = ti.xcom_pull(task_ids="check_performance_degradation", key="performance_metrics")

    pg_hook.run(
        query,
        parameters=(
            json.dumps(decision.get("reasons", [])),
            json.dumps(drift_metrics or {}),
            json.dumps(agreement_metrics or {}),
            json.dumps(performance_metrics or {}),
            "RETRAIN" if decision.get("should_retrain") else "SKIP"
        )
    )

    logger.info("Retraining trigger logged to database")


def send_retraining_notification(**context) -> None:
    """Send notification about retraining trigger."""
    ti = context["ti"]
    decision = ti.xcom_pull(task_ids="evaluate_retraining_need", key="retraining_decision")

    if not decision:
        return

    message = f"""
🔄 **Model Retraining Triggered**

**Reasons:**
{chr(10).join('- ' + r for r in decision.get('reasons', ['No reasons provided']))}

**Timestamp:** {decision.get('evaluated_at', 'Unknown')}

The L3 training pipeline will be triggered automatically.
    """

    # In production, send to Slack/PagerDuty
    logger.warning(message)

    # Try to send Slack notification if configured
    try:
        slack_webhook = Variable.get("slack_webhook_url", default_var=None)
        if slack_webhook:
            import requests
            requests.post(slack_webhook, json={"text": message}, timeout=10)
    except Exception as e:
        logger.warning(f"Could not send Slack notification: {e}")


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    dag_id=L2_DRIFT_RETRAIN,
    default_args=default_args,
    description="Automated drift-triggered model retraining",
    schedule_interval="0 */4 * * *",  # Every 4 hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "drift", "retraining", "automated"],
) as dag:

    # Check all metrics in parallel
    check_drift = PythonOperator(
        task_id="check_drift_status",
        python_callable=check_drift_status,
    )

    check_agreement = PythonOperator(
        task_id="check_model_agreement",
        python_callable=check_model_agreement,
    )

    check_performance = PythonOperator(
        task_id="check_performance_degradation",
        python_callable=check_performance_degradation,
    )

    check_cooldown_task = PythonOperator(
        task_id="check_cooldown",
        python_callable=check_cooldown,
    )

    # Evaluate and branch
    evaluate = BranchPythonOperator(
        task_id="evaluate_retraining_need",
        python_callable=evaluate_retraining_need,
    )

    # Trigger retraining path
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id=L3_MODEL_TRAINING,
        conf={"triggered_by": "drift_monitoring", "automated": True},
        wait_for_completion=False,
    )

    log_trigger = PythonOperator(
        task_id="log_retraining_trigger",
        python_callable=log_retraining_trigger,
        trigger_rule="none_failed_min_one_success",
    )

    send_notification = PythonOperator(
        task_id="send_retraining_notification",
        python_callable=send_retraining_notification,
    )

    # Skip retraining path
    skip_retraining = EmptyOperator(
        task_id="skip_retraining",
    )

    # End
    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    # Define dependencies
    [check_drift, check_agreement, check_performance, check_cooldown_task] >> evaluate
    evaluate >> [trigger_retraining, skip_retraining]
    trigger_retraining >> [log_trigger, send_notification] >> end
    skip_retraining >> end
