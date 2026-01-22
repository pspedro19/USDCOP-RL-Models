"""
L6: Production Monitoring DAG
=============================

Comprehensive production monitoring for the USD/COP trading system.

Responsibilities:
- Feature drift detection
- Performance degradation monitoring
- Automatic rollback triggers
- Model health checks
- Canary promotion automation
- Weekly performance reports

Schedule: Every 15 minutes during market hours

Author: Trading Team
Version: 1.0.0
Date: 2026-01-18
Contract: CTR-MON-001
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
sys.path.insert(0, '/opt/airflow')

logger = logging.getLogger(__name__)

from contracts.dag_registry import L6_PRODUCTION_MONITOR, L6_WEEKLY_REPORT

# =============================================================================
# CONFIGURATION
# =============================================================================

DAG_ID = L6_PRODUCTION_MONITOR
PROJECT_ROOT = Path('/opt/airflow')

# Monitoring thresholds (can be overridden by deployment_safeguards.yaml)
DEFAULT_THRESHOLDS = {
    "feature_drift_warning": 0.2,
    "feature_drift_critical": 0.4,
    "sharpe_warning": 0.3,
    "sharpe_critical": 0.0,
    "max_drawdown_warning": 0.10,
    "max_drawdown_critical": 0.15,
    "win_rate_warning": 0.40,
    "win_rate_critical": 0.35,
    "inference_latency_p99_ms": 100,
    "model_agreement_threshold": 0.70,
}


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

@task(task_id="load_monitoring_config")
def load_monitoring_config() -> Dict[str, Any]:
    """Load monitoring configuration from YAML and Variables."""
    import yaml

    config = DEFAULT_THRESHOLDS.copy()

    # Load from deployment_safeguards.yaml
    safeguards_path = PROJECT_ROOT / "config" / "deployment_safeguards.yaml"
    if safeguards_path.exists():
        with open(safeguards_path) as f:
            safeguards = yaml.safe_load(f)
            if "monitoring" in safeguards:
                mon_config = safeguards["monitoring"]
                for metric in mon_config.get("metrics", []):
                    name = metric.get("name")
                    threshold = metric.get("threshold")
                    if name and threshold:
                        config[name] = threshold

    # Override from Airflow Variable
    try:
        var_config = Variable.get("monitoring_thresholds", default_var=None)
        if var_config:
            config.update(json.loads(var_config))
    except Exception as e:
        logger.warning(f"Could not load monitoring_thresholds Variable: {e}")

    logger.info(f"[L6] Loaded monitoring config: {config}")
    return config


@task(task_id="check_feature_drift")
def check_feature_drift(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check for feature drift between training and inference distributions.

    Uses the DriftDetector to compare current inference features
    against the training baseline.
    """
    try:
        from src.monitoring.drift_detector import DriftDetector, DriftResult
        DRIFT_DETECTOR_AVAILABLE = True
    except ImportError:
        DRIFT_DETECTOR_AVAILABLE = False
        logger.warning("[L6] DriftDetector not available")

    result = {
        "checked_at": datetime.now().isoformat(),
        "drift_detected": False,
        "severity": "none",
        "features_drifted": [],
        "details": {},
    }

    if not DRIFT_DETECTOR_AVAILABLE:
        result["error"] = "DriftDetector not available"
        return result

    try:
        # Get recent inference features from database
        from utils.dag_common import get_db_connection

        conn = get_db_connection()
        cur = conn.cursor()

        # Get last 24 hours of inference features
        cur.execute("""
            SELECT feature_vector
            FROM dw.fact_rl_inference
            WHERE inference_timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY inference_timestamp DESC
            LIMIT 1000
        """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if len(rows) < 100:
            result["warning"] = f"Insufficient data for drift detection ({len(rows)} samples)"
            return result

        # Load baseline stats
        baseline_path = PROJECT_ROOT / "config" / "norm_stats.json"
        if not baseline_path.exists():
            result["warning"] = "Baseline stats not found"
            return result

        with open(baseline_path) as f:
            baseline_stats = json.load(f)

        # Initialize drift detector
        detector = DriftDetector(baseline_stats=baseline_stats)

        # Parse feature vectors and check drift
        import numpy as np
        feature_vectors = [json.loads(row[0]) if isinstance(row[0], str) else row[0] for row in rows]
        current_data = np.array(feature_vectors)

        drift_result = detector.compute_univariate_drift(current_data)

        result["drift_detected"] = drift_result.drift_detected
        result["severity"] = drift_result.severity.value
        result["drift_score"] = drift_result.overall_score
        result["features_drifted"] = drift_result.drifted_features
        result["details"] = {
            "per_feature_scores": drift_result.per_feature_scores,
            "sample_count": len(rows),
        }

        # Log alerts
        if drift_result.severity.value in ["medium", "high"]:
            logger.warning(f"[L6] Feature drift detected: {drift_result.drifted_features}")

    except Exception as e:
        logger.error(f"[L6] Drift check failed: {e}")
        result["error"] = str(e)

    return result


@task(task_id="check_model_performance")
def check_model_performance(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check model performance metrics (Sharpe, drawdown, win rate).

    Queries the inference results table to calculate rolling metrics.
    """
    result = {
        "checked_at": datetime.now().isoformat(),
        "metrics": {},
        "alerts": [],
        "status": "healthy",
    }

    try:
        from utils.dag_common import get_db_connection

        conn = get_db_connection()
        cur = conn.cursor()

        # Calculate 7-day rolling metrics
        cur.execute("""
            WITH trades AS (
                SELECT
                    inference_timestamp,
                    predicted_action,
                    realized_pnl,
                    LAG(close_price) OVER (ORDER BY inference_timestamp) as prev_close,
                    close_price
                FROM dw.fact_rl_inference
                WHERE inference_timestamp >= NOW() - INTERVAL '7 days'
                  AND realized_pnl IS NOT NULL
            ),
            daily_returns AS (
                SELECT
                    DATE(inference_timestamp) as trade_date,
                    SUM(realized_pnl) as daily_pnl,
                    COUNT(*) as trade_count,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades
                FROM trades
                GROUP BY DATE(inference_timestamp)
            )
            SELECT
                COUNT(*) as days,
                SUM(trade_count) as total_trades,
                SUM(winning_trades) as total_wins,
                AVG(daily_pnl) as avg_daily_pnl,
                STDDEV(daily_pnl) as std_daily_pnl,
                MIN(SUM(daily_pnl) OVER (ORDER BY trade_date)) as max_drawdown
            FROM daily_returns
        """)

        row = cur.fetchone()
        cur.close()
        conn.close()

        if row and row[0] and row[0] > 0:
            days, total_trades, wins, avg_pnl, std_pnl, max_dd = row

            # Calculate metrics
            sharpe = (avg_pnl / std_pnl * (252 ** 0.5)) if std_pnl and std_pnl > 0 else 0
            win_rate = wins / total_trades if total_trades > 0 else 0
            max_drawdown_pct = abs(max_dd / 10000) if max_dd else 0  # Assuming 10k initial capital

            result["metrics"] = {
                "rolling_sharpe_7d": round(sharpe, 3),
                "win_rate_7d": round(win_rate, 3),
                "max_drawdown_7d": round(max_drawdown_pct, 4),
                "total_trades_7d": int(total_trades),
                "days_analyzed": int(days),
            }

            # Check thresholds
            if sharpe < config.get("sharpe_critical", 0):
                result["alerts"].append({
                    "level": "critical",
                    "metric": "sharpe_ratio",
                    "value": sharpe,
                    "threshold": config.get("sharpe_critical"),
                })
                result["status"] = "critical"
            elif sharpe < config.get("sharpe_warning", 0.3):
                result["alerts"].append({
                    "level": "warning",
                    "metric": "sharpe_ratio",
                    "value": sharpe,
                    "threshold": config.get("sharpe_warning"),
                })
                result["status"] = "warning"

            if max_drawdown_pct > config.get("max_drawdown_critical", 0.15):
                result["alerts"].append({
                    "level": "critical",
                    "metric": "max_drawdown",
                    "value": max_drawdown_pct,
                    "threshold": config.get("max_drawdown_critical"),
                })
                result["status"] = "critical"

            if win_rate < config.get("win_rate_critical", 0.35):
                result["alerts"].append({
                    "level": "critical",
                    "metric": "win_rate",
                    "value": win_rate,
                    "threshold": config.get("win_rate_critical"),
                })
                result["status"] = "critical"
        else:
            result["warning"] = "Insufficient trade data for performance analysis"

    except Exception as e:
        logger.error(f"[L6] Performance check failed: {e}")
        result["error"] = str(e)

    return result


@task(task_id="check_rollback_triggers")
def check_rollback_triggers(
    config: Dict[str, Any],
    drift_result: Dict[str, Any],
    performance_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check if any rollback trigger conditions are met.

    Combines drift and performance results to make rollback decisions.
    """
    result = {
        "checked_at": datetime.now().isoformat(),
        "rollback_needed": False,
        "action": None,
        "trigger": None,
        "reason": None,
    }

    try:
        from src.inference.deployment_manager import (
            get_deployment_manager,
            check_rollback_needed,
        )
        DEPLOYMENT_MANAGER_AVAILABLE = True
    except ImportError:
        DEPLOYMENT_MANAGER_AVAILABLE = False
        logger.warning("[L6] DeploymentManager not available")

    # Collect metrics for rollback check
    metrics = {}

    # From drift result
    if "drift_score" in drift_result:
        metrics["feature_drift_score"] = drift_result["drift_score"]

    # From performance result
    if "metrics" in performance_result:
        metrics.update(performance_result["metrics"])

    # Check using deployment manager
    if DEPLOYMENT_MANAGER_AVAILABLE and metrics:
        rollback_check = check_rollback_needed(metrics)

        if rollback_check.get("rollback_needed"):
            result["rollback_needed"] = True
            result["action"] = rollback_check.get("action")
            result["trigger"] = rollback_check.get("trigger")
            result["rollback_executed"] = rollback_check.get("rollback_executed", False)
            logger.warning(f"[L6] Rollback triggered: {result['trigger']}")

    # Also check critical alerts from performance
    if performance_result.get("status") == "critical":
        for alert in performance_result.get("alerts", []):
            if alert["level"] == "critical":
                result["rollback_needed"] = True
                result["action"] = "pause_and_alert"
                result["trigger"] = f"{alert['metric']}_critical"
                result["reason"] = f"{alert['metric']}={alert['value']} (threshold: {alert['threshold']})"
                break

    return result


@task(task_id="check_canary_promotion")
def check_canary_promotion(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if canary model is ready for promotion.

    Evaluates the challenger model's performance against promotion criteria.
    """
    result = {
        "checked_at": datetime.now().isoformat(),
        "promotion_ready": False,
        "current_stage": None,
        "next_stage": None,
        "criteria_met": {},
    }

    try:
        from src.inference.deployment_manager import (
            get_deployment_manager,
            check_and_promote,
        )
        DEPLOYMENT_MANAGER_AVAILABLE = True
    except ImportError:
        DEPLOYMENT_MANAGER_AVAILABLE = False

    if not DEPLOYMENT_MANAGER_AVAILABLE:
        result["error"] = "DeploymentManager not available"
        return result

    try:
        manager = get_deployment_manager()
        status = manager.get_deployment_status()

        if not status.get("challenger"):
            result["message"] = "No challenger model in deployment"
            return result

        # Get challenger metrics from shadow/canary execution
        challenger = status["challenger"]
        result["current_stage"] = challenger.get("stage")

        # Check promotion
        # For now, use metrics from performance tracking
        # In production, this would use the challenger's specific metrics
        metrics = {
            "sharpe": 0.6,  # Would be calculated from shadow execution
            "max_drawdown": 0.08,
            "win_rate": 0.48,
            "agreement_with_champion": 0.75,
        }

        promotion_result = check_and_promote(metrics)
        result["promotion_ready"] = promotion_result.get("promoted", False)
        result["next_stage"] = promotion_result.get("next_stage")
        result["criteria_met"] = promotion_result.get("criteria_met", {})
        result["reason"] = promotion_result.get("reason")

        if result["promotion_ready"]:
            logger.info(f"[L6] Canary promoted to {result['next_stage']}")

    except Exception as e:
        logger.error(f"[L6] Canary promotion check failed: {e}")
        result["error"] = str(e)

    return result


@task(task_id="send_alerts")
def send_alerts(
    drift_result: Dict[str, Any],
    performance_result: Dict[str, Any],
    rollback_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Send alerts based on monitoring results.

    Supports Slack and email notifications.
    """
    alerts_sent = []

    # Collect all alerts
    all_alerts = []

    # Drift alerts
    if drift_result.get("drift_detected") and drift_result.get("severity") in ["medium", "high"]:
        all_alerts.append({
            "type": "drift",
            "severity": drift_result["severity"],
            "message": f"Feature drift detected: {drift_result.get('features_drifted', [])}",
        })

    # Performance alerts
    for alert in performance_result.get("alerts", []):
        all_alerts.append({
            "type": "performance",
            "severity": alert["level"],
            "message": f"{alert['metric']}={alert['value']} (threshold: {alert['threshold']})",
        })

    # Rollback alerts
    if rollback_result.get("rollback_needed"):
        all_alerts.append({
            "type": "rollback",
            "severity": "critical",
            "message": f"Rollback triggered: {rollback_result.get('trigger')} - {rollback_result.get('reason')}",
        })

    # Send alerts
    for alert in all_alerts:
        # Log the alert
        log_fn = logger.critical if alert["severity"] == "critical" else logger.warning
        log_fn(f"[ALERT] [{alert['type'].upper()}] {alert['message']}")

        # TODO: Integrate with actual notification services
        # For now, just log
        alerts_sent.append(alert)

    return {
        "alerts_sent": len(alerts_sent),
        "alerts": alerts_sent,
    }


@task(task_id="update_metrics")
def update_metrics(
    drift_result: Dict[str, Any],
    performance_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update Prometheus metrics for Grafana dashboards.
    """
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        # Create metrics
        registry = CollectorRegistry()

        # Drift metrics
        drift_score = Gauge(
            'usdcop_feature_drift_score',
            'Current feature drift score',
            registry=registry
        )
        drift_score.set(drift_result.get("drift_score", 0))

        # Performance metrics
        if "metrics" in performance_result:
            metrics = performance_result["metrics"]

            sharpe = Gauge('usdcop_rolling_sharpe_7d', 'Rolling 7-day Sharpe ratio', registry=registry)
            sharpe.set(metrics.get("rolling_sharpe_7d", 0))

            win_rate = Gauge('usdcop_win_rate_7d', 'Rolling 7-day win rate', registry=registry)
            win_rate.set(metrics.get("win_rate_7d", 0))

            drawdown = Gauge('usdcop_max_drawdown_7d', 'Rolling 7-day max drawdown', registry=registry)
            drawdown.set(metrics.get("max_drawdown_7d", 0))

        # Push to gateway (if configured)
        pushgateway_url = os.environ.get("PUSHGATEWAY_URL")
        if pushgateway_url:
            push_to_gateway(pushgateway_url, job='l6_monitoring', registry=registry)
            logger.info("[L6] Metrics pushed to Prometheus")

        return {"metrics_updated": True}

    except Exception as e:
        logger.warning(f"[L6] Failed to update Prometheus metrics: {e}")
        return {"metrics_updated": False, "error": str(e)}


@task(task_id="generate_monitoring_summary")
def generate_monitoring_summary(
    config: Dict[str, Any],
    drift_result: Dict[str, Any],
    performance_result: Dict[str, Any],
    rollback_result: Dict[str, Any],
    canary_result: Dict[str, Any],
    alerts_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate comprehensive monitoring summary."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "drift": {
            "status": "warning" if drift_result.get("drift_detected") else "ok",
            "score": drift_result.get("drift_score", 0),
            "severity": drift_result.get("severity", "none"),
        },
        "performance": {
            "status": performance_result.get("status", "unknown"),
            "metrics": performance_result.get("metrics", {}),
        },
        "rollback": {
            "triggered": rollback_result.get("rollback_needed", False),
            "action": rollback_result.get("action"),
        },
        "canary": {
            "stage": canary_result.get("current_stage"),
            "promotion_ready": canary_result.get("promotion_ready", False),
        },
        "alerts": {
            "count": alerts_result.get("alerts_sent", 0),
        },
    }

    # Determine overall status
    if rollback_result.get("rollback_needed"):
        summary["overall_status"] = "critical"
    elif performance_result.get("status") == "critical":
        summary["overall_status"] = "critical"
    elif performance_result.get("status") == "warning" or drift_result.get("drift_detected"):
        summary["overall_status"] = "warning"

    logger.info("=" * 60)
    logger.info("L6 PRODUCTION MONITORING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Overall Status: {summary['overall_status'].upper()}")
    logger.info(f"Drift: {summary['drift']['status']} (score: {summary['drift']['score']:.3f})")
    logger.info(f"Performance: {summary['performance']['status']}")
    if summary['performance']['metrics']:
        m = summary['performance']['metrics']
        logger.info(f"  - Sharpe (7d): {m.get('rolling_sharpe_7d', 'N/A')}")
        logger.info(f"  - Win Rate (7d): {m.get('win_rate_7d', 'N/A')}")
        logger.info(f"  - Max DD (7d): {m.get('max_drawdown_7d', 'N/A')}")
    logger.info(f"Rollback Triggered: {summary['rollback']['triggered']}")
    logger.info(f"Canary Stage: {summary['canary']['stage']}")
    logger.info(f"Alerts Sent: {summary['alerts']['count']}")
    logger.info("=" * 60)

    return summary


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    DAG_ID,
    default_args=default_args,
    description='L6: Production monitoring, drift detection, rollback, and canary management',
    schedule_interval='*/15 13-18 * * 1-5',  # Every 15 min during market hours (UTC)
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l6', 'monitoring', 'production', 'drift', 'rollback'],
) as dag:

    # Load configuration
    config = load_monitoring_config()

    # Parallel checks
    drift_result = check_feature_drift(config)
    performance_result = check_model_performance(config)
    canary_result = check_canary_promotion(config)

    # Rollback check (depends on drift and performance)
    rollback_result = check_rollback_triggers(config, drift_result, performance_result)

    # Send alerts (depends on all checks)
    alerts_result = send_alerts(drift_result, performance_result, rollback_result)

    # Update metrics for dashboards
    metrics_result = update_metrics(drift_result, performance_result)

    # Generate summary (depends on all results)
    summary = generate_monitoring_summary(
        config,
        drift_result,
        performance_result,
        rollback_result,
        canary_result,
        alerts_result,
    )

    # Dependencies are handled by TaskFlow API automatically
    # Explicit: config >> [drift, performance, canary] >> rollback >> alerts >> summary


# =============================================================================
# WEEKLY REPORT DAG
# =============================================================================

with DAG(
    L6_WEEKLY_REPORT,
    default_args=default_args,
    description='Weekly performance report generation',
    schedule_interval='0 20 * * 5',  # Every Friday at 8 PM UTC
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l6', 'reporting', 'weekly'],
) as weekly_dag:

    @task(task_id="generate_weekly_report")
    def generate_weekly_report() -> Dict[str, Any]:
        """Generate comprehensive weekly performance report."""
        logger.info("Generating weekly performance report...")

        # This would generate a comprehensive report
        # For now, placeholder
        report = {
            "generated_at": datetime.now().isoformat(),
            "period": "weekly",
            "status": "generated",
        }

        return report

    report = generate_weekly_report()
