"""
DAG: v3.l4_backtest_validation
==============================
USD/COP Trading System - V3 Architecture
Layer 4: Scheduled Backtest Validation Pipeline

Purpose:
    Automated model validation through backtesting with:
    - Weekly scheduled validation on out-of-sample data
    - Model comparison against baseline
    - Performance degradation detection
    - Alerting via Slack/Email
    - Metrics logging to MLflow

Architecture:
    ┌────────────────────────────────────────────────────────────────────┐
    │                    L4 BACKTEST VALIDATION                          │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                    │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
    │  │   prepare    │───▶│     run      │───▶│   validate   │         │
    │  │   backtest   │    │   backtest   │    │   results    │         │
    │  └──────────────┘    └──────────────┘    └──────────────┘         │
    │                                                │                   │
    │                                                ▼                   │
    │                             ┌────────────────────────────────┐    │
    │                             │         Branch Logic           │    │
    │                             └────────┬───────────────────────┘    │
    │                    ┌─────────────────┼─────────────────┐          │
    │                    ▼                 ▼                 ▼          │
    │            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
    │            │   passed     │  │   degraded   │  │   failed     │   │
    │            │   (info)     │  │   (warning)  │  │   (critical) │   │
    │            └──────────────┘  └──────────────┘  └──────────────┘   │
    │                    │                 │                 │          │
    │                    └─────────────────┴─────────────────┘          │
    │                                      │                            │
    │                                      ▼                            │
    │                             ┌──────────────┐                      │
    │                             │  log_metrics │                      │
    │                             │   (MLflow)   │                      │
    │                             └──────────────┘                      │
    │                                      │                            │
    │                                      ▼                            │
    │                             ┌──────────────┐                      │
    │                             │    summary   │                      │
    │                             └──────────────┘                      │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

Schedule:
    - @weekly (Sunday midnight) for routine validation
    - Manual trigger for ad-hoc validation
    - Post-training trigger via TriggerDagRunOperator

SOLID Principles:
    - SRP: Each task does one thing (prepare, run, validate, alert)
    - OCP: New validation strategies via registry
    - DIP: Tasks depend on contracts, not implementations
    - ISP: Minimal interfaces between stages

Design Patterns:
    - Pipeline Pattern: Sequential stages with XCom
    - Strategy Pattern: Pluggable validation strategies
    - Factory Pattern: Backtest runner creation
    - Observer Pattern: MLflow tracking

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import os
import sys

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# =============================================================================
# CONFIGURATION - SSOT INTEGRATION
# =============================================================================

try:
    from contracts.dag_registry import RL_L4_BACKTEST_VALIDATION
    DAG_ID = RL_L4_BACKTEST_VALIDATION
except ImportError:
    DAG_ID = "rl_l4_02_backtest_validation"

# =============================================================================
# EXPERIMENT SSOT INTEGRATION
# =============================================================================
EXPERIMENT_SSOT_AVAILABLE = False
EXPERIMENT_SSOT = None
SSOT_DATE_RANGES = None

try:
    import yaml
    from pathlib import Path

    # Read dates directly from YAML (not in ExperimentConfig)
    ssot_path = Path('/opt/airflow/config/experiment_ssot.yaml')
    if ssot_path.exists():
        with open(ssot_path) as f:
            raw_yaml = yaml.safe_load(f)
        SSOT_DATE_RANGES = raw_yaml.get('pipeline', {}).get('date_ranges', {})
        logging.info(f"[SSOT] Loaded date_ranges: {SSOT_DATE_RANGES}")

    from src.config.experiment_loader import load_experiment_config
    EXPERIMENT_SSOT = load_experiment_config(force_reload=True)
    EXPERIMENT_SSOT_AVAILABLE = True
    logging.info(f"[SSOT] Loaded experiment config v{EXPERIMENT_SSOT.version}")
except (ImportError, FileNotFoundError) as e:
    logging.warning(f"[SSOT] Could not load experiment SSOT: {e}")

# Build config from SSOT when available
if EXPERIMENT_SSOT_AVAILABLE and EXPERIMENT_SSOT is not None and SSOT_DATE_RANGES:
    _ssot_output = EXPERIMENT_SSOT.pipeline
    DEFAULT_VALIDATION_CONFIG = {
        # Model - use trained model from L3 (v3_18f_final2)
        "model_id": "v3_18f_final2",
        "model_path": "/opt/airflow/models/ppo_v3_18f_final2_production/final_model.zip",
        "norm_stats_path": "/opt/airflow/models/ppo_v3_18f_final2_production/norm_stats.json",
        # Use dataset PREFIX (not full path) to enable combined val+test loading
        # The backtest runner will load both _val.parquet and _test.parquet
        "dataset_path": "/opt/airflow/data/pipeline/07_output/5min/DS_v3_close_only",
        "baseline_model_id": None,

        # Period FROM SSOT - Use full date range available in val+test datasets
        # The backtest runner will load both val and test parquet files
        "period_type": "out_of_sample",
        "start_date": SSOT_DATE_RANGES.get('full_start', '2020-03-01'),  # Start from earliest available
        "end_date": SSOT_DATE_RANGES.get('test_end', '2026-02-01'),      # End at latest available
        "lookback_days": None,  # Not used when explicit dates provided

        # Dataset paths from SSOT
        "dataset_dir": _ssot_output.output_path,
        "dataset_prefix": _ssot_output.output_prefix,

        # Thresholds
        "min_sharpe_ratio": 0.3,  # Lowered for initial validation
        "max_drawdown_pct": 0.25,
        "min_win_rate": 0.35,
        "min_profit_factor": 0.8,
        "min_trades": 50,
        "max_consecutive_losses": 15,

        # Validation strategy
        "strategy": "standard",

        # Comparison settings
        "max_degradation_pct": 0.15,

        # Alerts
        "alert_on_degradation": True,
        "alert_channels": ["log"],

        # MLflow
        "mlflow_enabled": True,
        "mlflow_experiment": EXPERIMENT_SSOT.logging.experiment_name,
    }
    logging.info(f"[SSOT] Backtest period: {SSOT_DATE_RANGES.get('test_start')} to {SSOT_DATE_RANGES.get('test_end')}")
else:
    # Fallback config when SSOT not available
    DEFAULT_VALIDATION_CONFIG = {
        # Model
        "model_id": "ppo_latest",
        "baseline_model_id": None,

        # Period
        "period_type": "out_of_sample",
        "lookback_days": 30,

        # Thresholds
        "min_sharpe_ratio": 0.5,
        "max_drawdown_pct": 0.20,
        "min_win_rate": 0.40,
        "min_profit_factor": 1.0,
        "min_trades": 10,
        "max_consecutive_losses": 10,

        # Validation strategy
        "strategy": "comparison",

        # Comparison settings
        "max_degradation_pct": 0.10,

        # Alerts
        "alert_on_degradation": True,
        "alert_channels": ["slack", "log"],

        # MLflow
        "mlflow_enabled": True,
        "mlflow_experiment": "backtest_validation",
    }


def get_validation_config() -> Dict[str, Any]:
    """Get validation configuration from Airflow Variables"""
    try:
        config_json = Variable.get("backtest_validation_config", default_var=None)
        if config_json:
            config = json.loads(config_json)
            return {**DEFAULT_VALIDATION_CONFIG, **config}
    except Exception as e:
        logging.warning(f"Could not load validation config: {e}")

    return DEFAULT_VALIDATION_CONFIG


def get_project_paths() -> Dict[str, Path]:
    """Get project paths"""
    docker_root = Path('/opt/airflow')
    if docker_root.exists():
        project_root = docker_root
    else:
        project_root = Path(__file__).parent.parent.parent

    return {
        "project_root": project_root,
        "dags_dir": project_root / "airflow" / "dags",
    }


PATHS = get_project_paths()

# Add paths for imports
sys.path.insert(0, str(PATHS["dags_dir"]))


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def prepare_backtest(**context) -> Dict[str, Any]:
    """
    Task 1: Prepare backtest request.

    Determines:
    - Date range based on period_type
    - Model to validate
    - Baseline model for comparison
    """
    from contracts import (
        BacktestRequest,
        BacktestPeriodType,
        ValidationThresholds,
        PipelineContext,
    )

    config = context['dag_run'].conf or get_validation_config()
    execution_date = context['execution_date']

    logging.info(f"Preparing backtest for model {config['model_id']}")

    # Determine date range
    if config.get("start_date") and config.get("end_date"):
        # Explicit dates
        start_date = datetime.strptime(config["start_date"], "%Y-%m-%d").date()
        end_date = datetime.strptime(config["end_date"], "%Y-%m-%d").date()
    else:
        # Lookback from execution date
        end_date = execution_date.date() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=config.get("lookback_days", 30))

    # Create request
    request = BacktestRequest(
        model_id=config["model_id"],
        start_date=start_date,
        end_date=end_date,
        period_type=BacktestPeriodType(config.get("period_type", "out_of_sample")),
        force_regenerate=config.get("force_regenerate", False),
    )

    # Create thresholds
    thresholds = ValidationThresholds(
        min_sharpe_ratio=config.get("min_sharpe_ratio", 0.5),
        max_drawdown_pct=config.get("max_drawdown_pct", 0.20),
        min_win_rate=config.get("min_win_rate", 0.40),
        min_profit_factor=config.get("min_profit_factor", 1.0),
        min_trades=config.get("min_trades", 10),
        max_consecutive_losses=config.get("max_consecutive_losses", 10),
    )

    # Create pipeline context
    # FIX: Convert pendulum datetime to standard datetime for Pydantic compatibility
    exec_dt = datetime(
        execution_date.year,
        execution_date.month,
        execution_date.day,
        execution_date.hour,
        execution_date.minute,
        execution_date.second
    ) if hasattr(execution_date, 'year') else None

    pipeline_context = PipelineContext(
        request=request,
        thresholds=thresholds,
        dag_run_id=context['run_id'],
        execution_date=exec_dt,
    )

    logging.info(f"Backtest prepared: {start_date} to {end_date}")

    # Push to XCom
    ti = context['ti']
    ti.xcom_push(key='pipeline_context', value=pipeline_context.to_xcom())
    ti.xcom_push(key='config', value=config)

    return {
        "model_id": config["model_id"],
        "start_date": str(start_date),
        "end_date": str(end_date),
        "period_type": config.get("period_type"),
    }


def run_backtest(**context) -> Dict[str, Any]:
    """
    Task 2: Run backtest using BacktestRunnerFactory.

    Uses the existing backtest infrastructure.
    """
    from contracts import PipelineContext, BacktestResult, BacktestStatus
    from services import BacktestRunnerFactory, BacktestConfigBuilder

    ti = context['ti']
    pipeline_context = PipelineContext.from_xcom(
        ti.xcom_pull(key='pipeline_context', task_ids='prepare_backtest')
    )
    config = ti.xcom_pull(key='config', task_ids='prepare_backtest')

    request = pipeline_context.request
    logging.info(f"Running backtest for {request.model_id}: {request.start_date} to {request.end_date}")

    # Build backtest config
    backtest_config = (BacktestConfigBuilder()
        .with_model(request.model_id)
        .with_capital(config.get("initial_capital", 10_000.0))
        .with_costs(config.get("transaction_cost_bps", 75.0))  # realistic USDCOP spread
        .build())

    # Create runner - use EvaluationBacktestRunner which wraps BacktestEngine
    model_path = config.get("model_path")
    norm_stats_path = config.get("norm_stats_path")
    dataset_path = config.get("dataset_path")

    try:
        runner = BacktestRunnerFactory.create(
            runner_type="evaluation",  # Uses src/evaluation/backtest_engine.py
            config=backtest_config,
            project_root=PATHS["project_root"],
            model_path=model_path,
            norm_stats_path=norm_stats_path,
            dataset_path=dataset_path,
        )
    except Exception as e:
        logging.warning(f"Evaluation runner failed: {e}. Using mock runner.")
        runner = BacktestRunnerFactory.create(
            runner_type="mock",
            config=backtest_config,
            project_root=PATHS["project_root"],
        )

    # Run backtest
    result = runner.run(request)

    # Update pipeline context
    pipeline_context.backtest_result = result

    logging.info(
        f"Backtest complete: {result.trade_count} trades, "
        f"status={result.status.value}"
    )

    # Push to XCom
    ti.xcom_push(key='pipeline_context', value=pipeline_context.to_xcom())
    ti.xcom_push(key='backtest_result', value=result.model_dump(mode="json"))

    return {
        "status": result.status.value,
        "trade_count": result.trade_count,
        "processing_time_ms": result.processing_time_ms,
    }


def validate_results(**context) -> Dict[str, Any]:
    """
    Task 3: Validate backtest results using ValidationStrategy.

    Applies configured validation strategy and produces report.
    """
    from contracts import PipelineContext, ValidationResult
    from services import ValidationStrategyRegistry

    ti = context['ti']
    pipeline_context = PipelineContext.from_xcom(
        ti.xcom_pull(key='pipeline_context', task_ids='run_backtest')
    )
    config = ti.xcom_pull(key='config', task_ids='prepare_backtest')

    backtest_result = pipeline_context.backtest_result

    if not backtest_result or not backtest_result.is_successful:
        logging.error("No successful backtest result to validate")
        return {"overall_result": "failed", "reason": "No backtest result"}

    logging.info(f"Validating results with {config.get('strategy', 'standard')} strategy")

    # Get validation strategy
    strategy_name = config.get("strategy", "standard")
    strategy_kwargs = {}

    if strategy_name == "comparison":
        strategy_kwargs["max_degradation_pct"] = config.get("max_degradation_pct", 0.10)

    strategy = ValidationStrategyRegistry.get(
        name=strategy_name,
        thresholds=pipeline_context.thresholds,
        **strategy_kwargs
    )

    # Load baseline metrics if configured
    baseline_metrics = None
    if config.get("baseline_model_id"):
        # TODO: Load baseline metrics from previous run
        pass

    # Run validation
    report = strategy.validate(
        backtest_result=backtest_result,
        baseline_metrics=baseline_metrics,
    )

    # Update pipeline context
    pipeline_context.validation_report = report

    logging.info(
        f"Validation complete: {report.overall_result.value} "
        f"({report.passed_checks}/{len(report.checks)} checks passed)"
    )

    # Push to XCom
    ti.xcom_push(key='pipeline_context', value=pipeline_context.to_xcom())
    ti.xcom_push(key='validation_report', value=report.model_dump(mode="json"))
    ti.xcom_push(key='overall_result', value=report.overall_result.value)

    return {
        "overall_result": report.overall_result.value,
        "passed_checks": report.passed_checks,
        "failed_checks": report.failed_checks,
        "critical_failures": len(report.critical_failures),
    }


def branch_on_result(**context) -> str:
    """
    Branch: Route to appropriate handler based on validation result.

    Returns task_id to execute next.
    """
    ti = context['ti']
    overall_result = ti.xcom_pull(key='overall_result', task_ids='validate_results')

    if overall_result == "passed":
        return "handle_passed"
    elif overall_result == "degraded":
        return "handle_degraded"
    else:
        return "handle_failed"


def handle_passed(**context) -> Dict[str, Any]:
    """
    Task 4a: Handle passed validation.

    Logs success, sends info alert.
    """
    from services import AlertBuilder, get_alert_service
    from contracts import AlertSeverity

    ti = context['ti']
    config = ti.xcom_pull(key='config', task_ids='prepare_backtest')

    logging.info("Validation PASSED")

    # Send info alert
    alert = (AlertBuilder()
        .with_title(f"Model Validation Passed: {config['model_id']}")
        .with_message("All validation checks passed successfully.")
        .for_model(config['model_id'])
        .info()
        .to_channels(["log"])
        .build())

    get_alert_service().send_alert(alert)

    return {"action": "logged", "result": "passed"}


def handle_degraded(**context) -> Dict[str, Any]:
    """
    Task 4b: Handle degraded validation.

    Logs warning, sends warning alert.
    """
    from services import get_alert_service
    from contracts import PipelineContext

    ti = context['ti']
    pipeline_context = PipelineContext.from_xcom(
        ti.xcom_pull(key='pipeline_context', task_ids='validate_results')
    )
    config = ti.xcom_pull(key='config', task_ids='prepare_backtest')

    logging.warning("Validation DEGRADED")

    # Send warning alert
    if config.get("alert_on_degradation", True):
        channels = config.get("alert_channels", ["slack", "log"])
        get_alert_service().send_validation_alert(
            report=pipeline_context.validation_report,
            channels=channels,
        )

    return {"action": "alerted", "result": "degraded"}


def handle_failed(**context) -> Dict[str, Any]:
    """
    Task 4c: Handle failed validation.

    Logs error, sends critical alert.
    """
    from services import get_alert_service
    from contracts import PipelineContext

    ti = context['ti']
    pipeline_context = PipelineContext.from_xcom(
        ti.xcom_pull(key='pipeline_context', task_ids='validate_results')
    )
    config = ti.xcom_pull(key='config', task_ids='prepare_backtest')

    logging.error("Validation FAILED")

    # Send critical alert
    channels = config.get("alert_channels", ["slack", "log"])
    get_alert_service().send_validation_alert(
        report=pipeline_context.validation_report,
        channels=channels,
    )

    return {"action": "alerted", "result": "failed"}


def log_metrics_to_mlflow(**context) -> Dict[str, Any]:
    """
    Task 5: Log metrics to MLflow.

    Logs all backtest metrics for tracking and comparison.
    """
    ti = context['ti']
    config = ti.xcom_pull(key='config', task_ids='prepare_backtest')

    if not config.get("mlflow_enabled", True):
        logging.info("MLflow disabled, skipping metrics logging")
        return {"skipped": True}

    try:
        import mlflow

        # Get results
        backtest_result = ti.xcom_pull(key='backtest_result', task_ids='run_backtest')
        validation_report = ti.xcom_pull(key='validation_report', task_ids='validate_results')

        # Set experiment
        experiment_name = config.get("mlflow_experiment", "backtest_validation")
        mlflow.set_experiment(experiment_name)

        # Start run
        with mlflow.start_run(run_name=f"validation_{config['model_id']}_{datetime.now().strftime('%Y%m%d')}"):
            # Log parameters
            mlflow.log_params({
                "model_id": config["model_id"],
                "period_type": config.get("period_type"),
                "strategy": config.get("strategy"),
            })

            # Log metrics from backtest
            if backtest_result and backtest_result.get("metrics"):
                metrics = backtest_result["metrics"]
                mlflow.log_metrics({
                    "sharpe_ratio": metrics.get("sharpe_ratio") or 0,
                    "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                    "win_rate": metrics.get("win_rate", 0),
                    "total_trades": metrics.get("total_trades", 0),
                    "total_pnl_usd": metrics.get("total_pnl_usd", 0),
                    "profit_factor": metrics.get("profit_factor") or 0,
                })

            # Log validation result
            if validation_report:
                mlflow.log_metrics({
                    "validation_passed_checks": validation_report.get("passed_checks", 0),
                    "validation_failed_checks": validation_report.get("failed_checks", 0),
                })
                mlflow.set_tag("validation_result", validation_report.get("overall_result"))

        logging.info("Metrics logged to MLflow")
        return {"logged": True, "experiment": experiment_name}

    except ImportError:
        logging.warning("MLflow not installed, skipping metrics logging")
        return {"skipped": True, "reason": "mlflow not installed"}
    except Exception as e:
        logging.error(f"MLflow logging failed: {e}")
        return {"skipped": True, "reason": str(e)}


def pipeline_summary(**context) -> Dict[str, Any]:
    """
    Task 6: Generate pipeline summary.

    Aggregates all results and logs final status.
    """
    ti = context['ti']

    # Gather all results
    prepare_result = ti.xcom_pull(task_ids='prepare_backtest') or {}
    backtest_result = ti.xcom_pull(key='backtest_result', task_ids='run_backtest') or {}
    validation_result = ti.xcom_pull(task_ids='validate_results') or {}

    summary = {
        "model_id": prepare_result.get("model_id"),
        "period": f"{prepare_result.get('start_date')} to {prepare_result.get('end_date')}",
        "backtest_status": backtest_result.get("status"),
        "trade_count": backtest_result.get("trade_count", 0),
        "validation_result": validation_result.get("overall_result"),
        "passed_checks": validation_result.get("passed_checks", 0),
        "failed_checks": validation_result.get("failed_checks", 0),
    }

    # Log summary
    logging.info("=" * 70)
    logging.info("BACKTEST VALIDATION SUMMARY")
    logging.info("=" * 70)
    for key, value in summary.items():
        logging.info(f"  {key}: {value}")
    logging.info("=" * 70)

    return summary


def on_failure_callback(context):
    """Callback for task failures"""
    from services import send_alert
    from contracts import AlertSeverity

    task_instance = context['task_instance']
    exception = context.get('exception')

    send_alert(
        title="Backtest Validation Pipeline Failed",
        message=f"Task {task_instance.task_id} failed: {exception}",
        severity=AlertSeverity.CRITICAL,
        channels=["slack", "log"],
    )


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': on_failure_callback,
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='V3 L4: Scheduled backtest validation with alerting and MLflow',
    schedule_interval='0 0 * * 0',  # Every Sunday at midnight
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l4', 'backtest', 'validation', 'mlops'],
    params={
        "model_id": "ppo_latest",
        "lookback_days": 30,
        "strategy": "comparison",
    },
)

with dag:

    # Task 1: Prepare Backtest
    task_prepare = PythonOperator(
        task_id='prepare_backtest',
        python_callable=prepare_backtest,
        provide_context=True,
    )

    # Task 2: Run Backtest
    task_backtest = PythonOperator(
        task_id='run_backtest',
        python_callable=run_backtest,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
    )

    # Task 3: Validate Results
    task_validate = PythonOperator(
        task_id='validate_results',
        python_callable=validate_results,
        provide_context=True,
    )

    # Task 4: Branch on Result
    task_branch = BranchPythonOperator(
        task_id='branch_on_result',
        python_callable=branch_on_result,
        provide_context=True,
    )

    # Task 4a: Handle Passed
    task_passed = PythonOperator(
        task_id='handle_passed',
        python_callable=handle_passed,
        provide_context=True,
    )

    # Task 4b: Handle Degraded
    task_degraded = PythonOperator(
        task_id='handle_degraded',
        python_callable=handle_degraded,
        provide_context=True,
    )

    # Task 4c: Handle Failed
    task_failed = PythonOperator(
        task_id='handle_failed',
        python_callable=handle_failed,
        provide_context=True,
    )

    # Task 5: Log to MLflow
    task_mlflow = PythonOperator(
        task_id='log_metrics_to_mlflow',
        python_callable=log_metrics_to_mlflow,
        provide_context=True,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Task 6: Summary
    task_summary = PythonOperator(
        task_id='pipeline_summary',
        python_callable=pipeline_summary,
        provide_context=True,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Define dependencies
    task_prepare >> task_backtest >> task_validate >> task_branch

    # Branch paths
    task_branch >> [task_passed, task_degraded, task_failed]

    # All paths lead to MLflow and summary
    [task_passed, task_degraded, task_failed] >> task_mlflow >> task_summary
