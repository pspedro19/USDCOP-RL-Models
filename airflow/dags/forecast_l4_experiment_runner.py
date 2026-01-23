"""
Forecasting L4: A/B Experiment Runner DAG
==========================================

Orchestrates A/B testing experiments for forecasting models:
1. Load experiment configuration from YAML
2. Train treatment models
3. Run walk-forward backtest
4. Compare with baseline using statistical tests
5. Generate recommendation
6. Persist results to database

Design Patterns:
    - Command Pattern: Experiment operations as discrete tasks
    - Factory Pattern: ExperimentManager creation from config
    - Repository Pattern: Database persistence
    - SSOT: All IDs and configs from contracts

Schedule:
    - Triggered manually or via experiment API
    - Can also run on schedule for automated A/B testing

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
Contract: CTR-FORECAST-L4-EXPERIMENT-001
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
import json
import logging
import os

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule

# =============================================================================
# SSOT IMPORTS
# =============================================================================

from contracts.dag_registry import (
    FORECAST_L4_EXPERIMENT_RUNNER,
    DAG_TAGS,
    DagLayer,
    DagPipeline,
)

from contracts.xcom_contracts import (
    validate_xcom_output,
    XComSchema,
)

# =============================================================================
# CONSTANTS (SSOT)
# =============================================================================

DAG_ID = FORECAST_L4_EXPERIMENT_RUNNER
SCHEDULE_INTERVAL = None  # Manual trigger or API-triggered
CONFIG_DIR = Path("/opt/airflow/config/forecast_experiments")
DEFAULT_DATASET_PATH = "/opt/airflow/data/forecasting/features.parquet"

# Experiment status values (SSOT)
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"

# =============================================================================
# DEFAULT ARGS
# =============================================================================

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=6),  # Experiments can be long
}


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

@task(task_id="load_experiment_config")
def load_experiment_config(**context) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML or Airflow Variable.

    Supports two modes:
    1. Config file path in dag_run.conf
    2. Full config dict in dag_run.conf

    Returns:
        Dict with experiment configuration
    """
    import yaml

    dag_run_conf = context.get("dag_run", {})
    if hasattr(dag_run_conf, "conf"):
        conf = dag_run_conf.conf or {}
    else:
        conf = {}

    # Mode 1: Config file path provided
    config_path = conf.get("config_path")
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            # Try looking in default config directory
            config_file = CONFIG_DIR / config_path

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        logging.info(f"Loaded config from: {config_file}")
        return config

    # Mode 2: Full config dict in dag_run.conf
    if "experiment" in conf:
        logging.info("Using config from dag_run.conf")
        return conf

    # Mode 3: Try default experiment variable
    try:
        default_config = Variable.get("forecast_experiment_default", deserialize_json=True)
        logging.info("Using default experiment config from Variable")
        return default_config
    except Exception:
        pass

    # No config found, use baseline defaults for testing
    logging.warning("No experiment config found, using defaults")
    return {
        "experiment": {
            "name": "baseline_v1",
            "version": "1.0.0",
            "description": "Baseline forecasting experiment",
            "baseline_experiment": None,
        },
        "models": {"include": None},  # All models
        "horizons": {"include": None},  # All horizons
        "training": {
            "walk_forward_windows": 5,
            "min_train_pct": 0.4,
            "gap_days": 30,
        },
        "evaluation": {
            "primary_metric": "direction_accuracy",
            "secondary_metrics": ["rmse", "mae"],
            "significance_level": 0.05,
            "bonferroni_correction": True,
        },
    }


@task(task_id="validate_config")
def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate experiment configuration.

    Checks:
    - Required fields present
    - Values within acceptable ranges
    - Baseline experiment exists (if specified)

    Returns:
        Validated config (unchanged if valid)

    Raises:
        ValueError: If config is invalid
    """
    required_sections = ["experiment"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    experiment = config["experiment"]
    if "name" not in experiment:
        raise ValueError("Experiment must have a name")

    # Validate significance level
    evaluation = config.get("evaluation", {})
    alpha = evaluation.get("significance_level", 0.05)
    if not 0 < alpha < 1:
        raise ValueError(f"Invalid significance_level: {alpha}")

    # Validate walk_forward_windows
    training = config.get("training", {})
    windows = training.get("walk_forward_windows", 5)
    if windows < 1 or windows > 20:
        raise ValueError(f"walk_forward_windows must be 1-20, got: {windows}")

    logging.info(f"Config validated: {experiment['name']}")
    return config


@task(task_id="initialize_experiment")
def initialize_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize experiment run using ExperimentManager.

    Creates:
    - Experiment run record
    - MLflow experiment (if enabled)
    - Tracking metadata

    Returns:
        Dict with run info and manager reference
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from src.forecasting.experiment_manager import (
        ExperimentConfig,
        ForecastExperimentManager,
    )

    # Build ExperimentConfig from YAML structure
    experiment = config.get("experiment", {})
    models_config = config.get("models", {})
    horizons_config = config.get("horizons", {})
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})

    exp_config = ExperimentConfig(
        name=experiment.get("name", "unnamed"),
        version=experiment.get("version", "1.0.0"),
        description=experiment.get("description", ""),
        hypothesis=experiment.get("hypothesis", ""),
        baseline_experiment=experiment.get("baseline_experiment"),
        models=models_config.get("include"),
        horizons=horizons_config.get("include"),
        walk_forward_windows=training_config.get("walk_forward_windows", 5),
        min_train_pct=training_config.get("min_train_pct", 0.4),
        gap_days=training_config.get("gap_days", 30),
        primary_metric=evaluation_config.get("primary_metric", "direction_accuracy"),
        secondary_metrics=evaluation_config.get("secondary_metrics", ["rmse"]),
        significance_level=evaluation_config.get("significance_level", 0.05),
        bonferroni_correction=evaluation_config.get("bonferroni_correction", True),
    )

    # Initialize manager
    manager = ForecastExperimentManager(config=exp_config)
    run = manager.create_run()

    logging.info(f"Initialized experiment run: {run.run_id}")
    logging.info(f"  Experiment: {run.experiment_name}")
    logging.info(f"  Models: {exp_config.models}")
    logging.info(f"  Horizons: {exp_config.horizons}")

    return {
        "run_id": run.run_id,
        "experiment_name": run.experiment_name,
        "experiment_version": run.experiment_version,
        "config_hash": run.config_hash,
        "models": exp_config.models,
        "horizons": exp_config.horizons,
        "baseline_experiment": exp_config.baseline_experiment,
        "config_dict": config,
    }


@task(task_id="train_models")
def train_models(run_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train treatment models for the experiment.

    Trains all specified models for all horizons using
    walk-forward cross-validation.

    Returns:
        Dict with training results and metrics
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from src.forecasting.experiment_manager import (
        ExperimentConfig,
        ForecastExperimentManager,
        ExperimentRun,
    )

    # Reconstruct config and manager
    config = ExperimentConfig(**_extract_experiment_config(run_info["config_dict"]))
    manager = ForecastExperimentManager(config=config)

    # Reconstruct run
    run = ExperimentRun(
        run_id=run_info["run_id"],
        experiment_name=run_info["experiment_name"],
        experiment_version=run_info["experiment_version"],
        config=config,
        config_hash=run_info["config_hash"],
        status=STATUS_PENDING,
    )

    # Train models
    dataset_path = Variable.get(
        "forecast_dataset_path",
        default_var=DEFAULT_DATASET_PATH
    )

    logging.info(f"Training models with dataset: {dataset_path}")
    run = manager.train(run, dataset_path=dataset_path)
    manager.save(run)

    if run.status == STATUS_FAILED:
        logging.error(f"Training failed: {run.error_message}")
        return {
            **run_info,
            "status": STATUS_FAILED,
            "error_message": run.error_message,
            "training_metrics": {},
        }

    logging.info(f"Training completed. Status: {run.status}")
    logging.info(f"Aggregate metrics: {run.aggregate_metrics}")

    return {
        **run_info,
        "status": run.status,
        "training_metrics": run.training_metrics,
        "aggregate_metrics": run.aggregate_metrics,
        "model_artifacts_path": run.model_artifacts_path,
        "mlflow_run_ids": run.mlflow_run_ids,
    }


@task(task_id="run_backtest")
def run_backtest(training_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run walk-forward backtest on trained models.

    Evaluates model performance on held-out data
    using direction accuracy and RMSE metrics.

    Returns:
        Dict with backtest results by model and horizon
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from src.forecasting.experiment_manager import (
        ExperimentConfig,
        ForecastExperimentManager,
        ExperimentRun,
    )

    if training_result.get("status") == STATUS_FAILED:
        logging.warning("Skipping backtest due to training failure")
        return training_result

    # Reconstruct config, manager, and run
    config = ExperimentConfig(**_extract_experiment_config(training_result["config_dict"]))
    manager = ForecastExperimentManager(config=config)

    run = ExperimentRun(
        run_id=training_result["run_id"],
        experiment_name=training_result["experiment_name"],
        experiment_version=training_result["experiment_version"],
        config=config,
        config_hash=training_result["config_hash"],
        status=training_result["status"],
        training_metrics=training_result.get("training_metrics", {}),
        aggregate_metrics=training_result.get("aggregate_metrics", {}),
        model_artifacts_path=training_result.get("model_artifacts_path"),
    )

    # Run backtest
    logging.info("Running walk-forward backtest")
    run = manager.backtest(run)
    manager.save(run)

    logging.info(f"Backtest completed. Metrics count: {len(run.backtest_metrics)}")

    return {
        **training_result,
        "backtest_metrics": run.backtest_metrics,
        "aggregate_metrics": run.aggregate_metrics,
    }


def check_baseline_exists(**context) -> bool:
    """
    Check if baseline experiment exists for A/B comparison.

    ShortCircuit operator - skips A/B comparison if no baseline.
    """
    ti = context["ti"]
    run_info = ti.xcom_pull(task_ids="initialize_experiment")

    baseline = run_info.get("baseline_experiment")
    if not baseline:
        logging.info("No baseline experiment specified, skipping A/B comparison")
        return False

    # Check if baseline exists in database
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from src.forecasting.experiment_manager import ExperimentRepository

    repo = ExperimentRepository()
    baseline_run = repo.get_latest_run(baseline)
    repo.close()

    if baseline_run is None:
        logging.warning(f"Baseline experiment '{baseline}' not found in database")
        return False

    logging.info(f"Baseline experiment found: {baseline} (run: {baseline_run.run_id})")
    return True


@task(task_id="compare_with_baseline")
def compare_with_baseline(backtest_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run A/B statistical comparison with baseline.

    Compares treatment vs baseline using:
    - McNemar test for direction accuracy
    - Paired t-test for RMSE
    - Fisher's combined test for aggregate

    Returns:
        Dict with comparison results and recommendation
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from src.forecasting.experiment_manager import (
        ExperimentConfig,
        ForecastExperimentManager,
        ExperimentRun,
    )

    if backtest_result.get("status") == STATUS_FAILED:
        logging.warning("Skipping comparison due to previous failure")
        return backtest_result

    # Reconstruct everything
    config = ExperimentConfig(**_extract_experiment_config(backtest_result["config_dict"]))
    manager = ForecastExperimentManager(config=config)

    run = ExperimentRun(
        run_id=backtest_result["run_id"],
        experiment_name=backtest_result["experiment_name"],
        experiment_version=backtest_result["experiment_version"],
        config=config,
        config_hash=backtest_result["config_hash"],
        status=backtest_result["status"],
        training_metrics=backtest_result.get("training_metrics", {}),
        backtest_metrics=backtest_result.get("backtest_metrics", {}),
        aggregate_metrics=backtest_result.get("aggregate_metrics", {}),
    )

    # Run A/B comparison
    baseline_name = backtest_result.get("baseline_experiment")
    logging.info(f"Comparing with baseline: {baseline_name}")

    comparison = manager.compare_with_baseline(run, baseline_name)

    if comparison is None:
        logging.error("A/B comparison failed")
        return {
            **backtest_result,
            "comparison_result": None,
            "recommendation": "comparison_failed",
        }

    logging.info(f"A/B Comparison complete:")
    logging.info(f"  Recommendation: {comparison.recommendation.value}")
    logging.info(f"  Confidence: {comparison.confidence_score:.2%}")
    logging.info(f"  Treatment wins: {comparison.summary.get('treatment_wins', 0)}")
    logging.info(f"  Baseline wins: {comparison.summary.get('baseline_wins', 0)}")

    return {
        **backtest_result,
        "comparison_result": comparison.to_dict(),
        "recommendation": comparison.recommendation.value,
        "confidence_score": comparison.confidence_score,
    }


@task(task_id="generate_report")
def generate_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate experiment report and summary.

    Creates:
    - JSON summary for dashboards
    - Markdown report for documentation
    - Alerts if recommendation is significant

    Returns:
        Dict with report paths and summary
    """
    from datetime import datetime
    import json

    experiment_name = results.get("experiment_name", "unknown")
    run_id = results.get("run_id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build summary
    summary = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "timestamp": timestamp,
        "status": results.get("status", "unknown"),
        "training_completed": bool(results.get("training_metrics")),
        "backtest_completed": bool(results.get("backtest_metrics")),
        "comparison_completed": bool(results.get("comparison_result")),
        "recommendation": results.get("recommendation", "not_compared"),
        "confidence_score": results.get("confidence_score", 0.0),
        "aggregate_metrics": results.get("aggregate_metrics", {}),
    }

    # Log report
    logging.info("=" * 60)
    logging.info("EXPERIMENT REPORT")
    logging.info("=" * 60)
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Run ID: {run_id}")
    logging.info(f"Status: {summary['status']}")

    if summary["comparison_completed"]:
        logging.info(f"Recommendation: {summary['recommendation']}")
        logging.info(f"Confidence: {summary['confidence_score']:.2%}")

    if summary["aggregate_metrics"]:
        logging.info(f"Avg Direction Accuracy: {summary['aggregate_metrics'].get('avg_direction_accuracy', 'N/A'):.2%}")

    logging.info("=" * 60)

    # Save JSON report
    report_dir = Path("/opt/airflow/data/forecasting/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f"{experiment_name}_{run_id}_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logging.info(f"Report saved: {report_path}")

    return {
        "report_path": str(report_path),
        "summary": summary,
    }


@task(task_id="notify_results", trigger_rule=TriggerRule.ALL_DONE)
def notify_results(report: Dict[str, Any]) -> None:
    """
    Send notifications based on experiment results.

    Notifications sent when:
    - Treatment significantly beats baseline → Alert for deployment
    - Experiment fails → Alert ML team
    - Low confidence → Suggest more data
    """
    summary = report.get("summary", {})
    recommendation = summary.get("recommendation", "")
    confidence = summary.get("confidence_score", 0.0)

    # For now, just log the notification
    # In production, integrate with Slack/Email/PagerDuty
    if recommendation == "deploy_treatment" and confidence >= 0.8:
        logging.info("🚀 ALERT: Treatment model ready for deployment!")
        logging.info(f"   Confidence: {confidence:.2%}")
        # send_slack_notification(...)

    elif recommendation == "keep_baseline":
        logging.info("📊 INFO: Baseline model remains best choice")

    elif summary.get("status") == STATUS_FAILED:
        logging.error("❌ ALERT: Experiment failed - check logs")
        # send_pagerduty_alert(...)

    else:
        logging.info(f"📈 Experiment completed: {recommendation}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_experiment_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ExperimentConfig fields from YAML-style config dict."""
    experiment = config_dict.get("experiment", {})
    models_config = config_dict.get("models", {})
    horizons_config = config_dict.get("horizons", {})
    training_config = config_dict.get("training", {})
    evaluation_config = config_dict.get("evaluation", {})

    return {
        "name": experiment.get("name", "unnamed"),
        "version": experiment.get("version", "1.0.0"),
        "description": experiment.get("description", ""),
        "hypothesis": experiment.get("hypothesis", ""),
        "baseline_experiment": experiment.get("baseline_experiment"),
        "models": models_config.get("include"),
        "horizons": horizons_config.get("include"),
        "walk_forward_windows": training_config.get("walk_forward_windows", 5),
        "min_train_pct": training_config.get("min_train_pct", 0.4),
        "gap_days": training_config.get("gap_days", 30),
        "primary_metric": evaluation_config.get("primary_metric", "direction_accuracy"),
        "secondary_metrics": evaluation_config.get("secondary_metrics", ["rmse"]),
        "significance_level": evaluation_config.get("significance_level", 0.05),
        "bonferroni_correction": evaluation_config.get("bonferroni_correction", True),
    }


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Forecasting A/B Experiment Runner - Compare treatment vs baseline",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,  # Only one experiment at a time
    tags=DAG_TAGS.get(FORECAST_L4_EXPERIMENT_RUNNER, ["forecast", "l4", "experiment"]),
    doc_md=__doc__,
    params={
        "config_path": {
            "description": "Path to experiment YAML config file",
            "type": "string",
            "default": "",
        },
    },
) as dag:

    # ==========================================================================
    # TASK FLOW
    # ==========================================================================

    # Phase 1: Configuration
    config = load_experiment_config()
    validated_config = validate_config(config)

    # Phase 2: Initialize
    run_info = initialize_experiment(validated_config)

    # Phase 3: Training
    training_result = train_models(run_info)

    # Phase 4: Backtest
    backtest_result = run_backtest(training_result)

    # Phase 5: A/B Comparison (conditional)
    baseline_check = ShortCircuitOperator(
        task_id="check_baseline_exists",
        python_callable=check_baseline_exists,
        provide_context=True,
    )

    comparison_result = compare_with_baseline(backtest_result)

    # Phase 6: Reporting
    report = generate_report(comparison_result)
    notify = notify_results(report)

    # ==========================================================================
    # DEPENDENCIES
    # ==========================================================================

    # Main flow
    config >> validated_config >> run_info >> training_result >> backtest_result

    # Conditional A/B comparison
    backtest_result >> baseline_check >> comparison_result >> report >> notify


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "dag",
    "DAG_ID",
]
