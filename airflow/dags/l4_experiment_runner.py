"""
L4: Experiment Runner DAG - Orchestrator
=========================================

Airflow DAG for orchestrating A/B experiments via L2→L3 pipeline.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  L4 Experiment Runner (this file)                       │
    │  - Orchestrates L2 and L3                               │
    │  - Passes experiment config                             │
    │  - Collects results via XCom                            │
    └─────────────────────────────────────────────────────────┘
              │                               │
              │ triggers                      │ triggers
              ▼                               ▼
    ┌──────────────────────┐    ┌──────────────────────┐
    │  L2 Preprocessing    │───▶│  L3 Training         │
    │  - Builds dataset    │    │  - Trains model      │
    │  - Pushes to XCom    │    │  - Pushes to XCom    │
    └──────────────────────┘    └──────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │ XCom contracts
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │  L4 Experiment Runner (results collection)              │
    │  - Pulls L3 results                                     │
    │  - Registers experiment                                 │
    │  - Compares with baseline                               │
    │  - Sends notification                                   │
    └─────────────────────────────────────────────────────────┘

Supports:
- Running experiments from YAML configuration
- Automatic MLflow logging via L3
- Experiment comparison after completion
- Notification on completion

Trigger:
- Manual via Airflow UI
- API trigger with experiment_name parameter

Usage:
    1. Create experiment YAML in config/experiments/
    2. Trigger DAG with experiment_name parameter
    3. Monitor progress in Airflow UI (L2 → L3 → L4)
    4. View results in MLflow

Contract: CTR-TRAIN-001

Author: Trading Team
Version: 2.0.0 (Refactored as orchestrator with XCom contracts)
Date: 2026-01-18
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
sys.path.insert(0, '/opt/airflow')

# Configure logging
logger = logging.getLogger(__name__)

# Import DAG IDs from registry
from contracts.dag_registry import L4_EXPERIMENT_RUNNER, L4_SCHEDULED_RETRAINING

# =============================================================================
# IMPORTS FROM SSOT
# =============================================================================

# XCom Contracts - SSOT for inter-DAG communication
try:
    from airflow.dags.contracts.xcom_contracts import (
        L2XComKeysEnum,
        L3XComKeysEnum,
        L4XComKeysEnum,
        L2_DAG_ID,
        L3_DAG_ID,
        L2Output,
        L3Output,
        L4Output,
        pull_l2_output,
        pull_l3_output,
        compute_config_hash,
    )
    XCOM_CONTRACTS_AVAILABLE = True
    logger.info("[SSOT] XCom contracts loaded successfully")
except ImportError as e:
    XCOM_CONTRACTS_AVAILABLE = False
    logger.warning(f"[SSOT] XCom contracts not available: {e}")
    # Fallback DAG IDs - import from dag_registry
    from contracts.dag_registry import L2_DATASET_BUILD, L3_MODEL_TRAINING
    L2_DAG_ID = L2_DATASET_BUILD
    L3_DAG_ID = L3_MODEL_TRAINING

# Experiment configuration support
try:
    from src.experiments import load_experiment_config, validate_experiment_config
    EXPERIMENT_CONFIG_AVAILABLE = True
except ImportError:
    EXPERIMENT_CONFIG_AVAILABLE = False
    logger.warning("[SSOT] Experiment config loader not available")

# Default DAG arguments
default_args = {
    "owner": "trading-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG parameters
dag_params = {
    "experiment_name": Param(
        default="baseline_ppo_v1",
        type="string",
        description="Name of experiment config file (without .yaml)",
    ),
    "dry_run": Param(
        default=False,
        type="boolean",
        description="Validate config only without training",
    ),
    "compare_with": Param(
        default="",
        type="string",
        description="Optional: experiment name to compare with after completion",
    ),
    "notify_on_complete": Param(
        default=True,
        type="boolean",
        description="Send notification when experiment completes",
    ),
}


# =============================================================================
# Task Functions
# =============================================================================

@task(task_id="validate_config")
def validate_config(experiment_name: str) -> Dict[str, Any]:
    """
    Validate experiment configuration.

    Args:
        experiment_name: Name of experiment config file

    Returns:
        Config metadata for downstream tasks

    Raises:
        AirflowSkipException: If experiment not found
        ValueError: If config is invalid
    """
    PROJECT_ROOT = Path("/opt/airflow")
    config_path = PROJECT_ROOT / "config" / "experiments" / f"{experiment_name}.yaml"

    if not config_path.exists():
        raise AirflowSkipException(f"Experiment config not found: {config_path}")

    # Compute config hash for lineage
    config_hash = None
    if XCOM_CONTRACTS_AVAILABLE:
        config_hash = compute_config_hash(config_path)

    # Validate if loader available
    if EXPERIMENT_CONFIG_AVAILABLE:
        errors = validate_experiment_config(config_path)
        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Configuration validation failed:\n{error_msg}")

        # Load and return metadata
        config = load_experiment_config(config_path)

        return {
            "experiment_name": config.experiment.name,
            "experiment_version": config.experiment.version,
            "algorithm": config.model.algorithm.value if hasattr(config.model.algorithm, 'value') else str(config.model.algorithm),
            "total_timesteps": config.training.total_timesteps,
            "config_path": str(config_path),
            "config_hash": config_hash,
            "validated_at": datetime.now().isoformat(),
        }
    else:
        # Fallback: basic YAML validation
        import yaml
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        return {
            "experiment_name": config_data.get("experiment", {}).get("name", experiment_name),
            "experiment_version": config_data.get("experiment", {}).get("version", "1.0.0"),
            "algorithm": config_data.get("model", {}).get("algorithm", "PPO"),
            "total_timesteps": config_data.get("training", {}).get("total_timesteps", 500000),
            "config_path": str(config_path),
            "config_hash": config_hash,
            "validated_at": datetime.now().isoformat(),
        }


@task(task_id="trigger_l2_preprocessing")
def trigger_l2_preprocessing(
    config_metadata: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Trigger L2 preprocessing DAG to generate experiment-specific dataset.

    This task triggers L2 with the experiment config path, which causes L2
    to use the CanonicalFeatureBuilder (SSOT) and push dataset info to XCom.

    Args:
        config_metadata: Metadata from validate_config
        dry_run: If True, skip actual trigger

    Returns:
        Trigger information
    """
    from airflow.api.common.trigger_dag import trigger_dag

    if dry_run:
        logger.info("[L4] Dry run mode - skipping L2 trigger")
        return {
            "status": "dry_run",
            "triggered": False,
            "experiment_name": config_metadata["experiment_name"],
        }

    experiment_name = config_metadata["experiment_name"]
    config_path = config_metadata["config_path"]

    logger.info(f"[L4] Triggering L2 preprocessing for experiment: {experiment_name}")

    # Trigger L2 with experiment configuration
    l2_conf = {
        "experiment_name": experiment_name,
        "experiment_config_path": config_path,
        "triggered_by": "l4_experiment_runner",
    }

    run_id = trigger_dag(
        dag_id=L2_DAG_ID,
        conf=l2_conf,
        execution_date=None,
        replace_microseconds=False,
    )

    logger.info(f"[L4] L2 triggered with run_id: {run_id}")

    return {
        "status": "triggered",
        "triggered": True,
        "dag_id": L2_DAG_ID,
        "run_id": str(run_id) if run_id else None,
        "experiment_name": experiment_name,
        "config_path": config_path,
    }


@task(task_id="trigger_l3_training")
def trigger_l3_training(
    config_metadata: Dict[str, Any],
    l2_trigger_info: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Trigger L3 training DAG with experiment configuration.

    L3 will pull the dataset from L2 via XCom contracts.

    Args:
        config_metadata: Metadata from validate_config
        l2_trigger_info: Info from L2 trigger
        dry_run: If True, skip actual trigger

    Returns:
        Trigger information
    """
    from airflow.api.common.trigger_dag import trigger_dag

    if dry_run or l2_trigger_info.get("status") == "dry_run":
        logger.info("[L4] Dry run mode - skipping L3 trigger")
        return {
            "status": "dry_run",
            "triggered": False,
            "experiment_name": config_metadata["experiment_name"],
        }

    experiment_name = config_metadata["experiment_name"]
    config_path = config_metadata["config_path"]

    logger.info(f"[L4] Triggering L3 training for experiment: {experiment_name}")

    # Trigger L3 with experiment configuration
    # L3 will pull dataset info from L2 XCom automatically
    l3_conf = {
        "experiment_name": experiment_name,
        "experiment_config_path": config_path,
        "triggered_by": "l4_experiment_runner",
        # L3 will pull dataset from L2 XCom, but we can provide fallback
        "dataset_name": f"RL_DS_{experiment_name}.csv",
    }

    run_id = trigger_dag(
        dag_id=L3_DAG_ID,
        conf=l3_conf,
        execution_date=None,
        replace_microseconds=False,
    )

    logger.info(f"[L4] L3 triggered with run_id: {run_id}")

    return {
        "status": "triggered",
        "triggered": True,
        "dag_id": L3_DAG_ID,
        "run_id": str(run_id) if run_id else None,
        "experiment_name": experiment_name,
        "config_path": config_path,
    }


@task(task_id="collect_experiment_results")
def collect_experiment_results(
    config_metadata: Dict[str, Any],
    l3_trigger_info: Dict[str, Any],
    **context,
) -> Dict[str, Any]:
    """
    Collect experiment results from L3 via XCom contracts.

    This task pulls the training results from L3 XCom.

    Args:
        config_metadata: Metadata from validate_config
        l3_trigger_info: Info from L3 trigger
        context: Airflow context with TaskInstance

    Returns:
        Experiment results
    """
    ti = context['ti']

    if l3_trigger_info.get("status") == "dry_run":
        logger.info("[L4] Dry run mode - no results to collect")
        return {
            "status": "dry_run",
            "experiment_name": config_metadata["experiment_name"],
            "config_path": config_metadata["config_path"],
        }

    experiment_name = config_metadata["experiment_name"]
    logger.info(f"[L4] Collecting results for experiment: {experiment_name}")

    # Pull L3 output from XCom using contracts
    l3_output = None
    if XCOM_CONTRACTS_AVAILABLE:
        l3_output = pull_l3_output(ti)

    if l3_output:
        logger.info(f"[L4] Retrieved L3 output via contracts")
        logger.info(f"[L4] Model path: {l3_output.model_path}")
        logger.info(f"[L4] Model hash: {l3_output.model_hash}")
        logger.info(f"[L4] Best reward: {l3_output.best_reward}")
        logger.info(f"[L4] Training duration: {l3_output.training_duration/60:.1f} min")

        return {
            "status": "success",
            "experiment_name": experiment_name,
            "experiment_version": config_metadata.get("experiment_version", "1.0.0"),
            "run_id": l3_output.mlflow_run_id,
            "duration_seconds": l3_output.training_duration,
            "metrics": {
                "best_reward": l3_output.best_reward,
            },
            "model_path": l3_output.model_path,
            "model_hash": l3_output.model_hash,
            "mlflow_run_id": l3_output.mlflow_run_id,
            "norm_stats_hash": l3_output.norm_stats_hash,
            "dataset_hash": l3_output.dataset_hash,
            "config_hash": l3_output.config_hash,
            "error": None,
        }
    else:
        # Fallback: try legacy XCom keys
        logger.warning("[L4] XCom contracts not available, trying legacy keys")

        training_result = ti.xcom_pull(
            key='training_result',
            dag_id=L3_DAG_ID,
            include_prior_dates=True,
        )

        if training_result:
            return {
                "status": training_result.get("status", "success"),
                "experiment_name": experiment_name,
                "experiment_version": config_metadata.get("experiment_version", "1.0.0"),
                "run_id": training_result.get("model_id"),
                "duration_seconds": training_result.get("training_duration_seconds", 0),
                "metrics": {
                    "best_reward": training_result.get("best_mean_reward", 0),
                },
                "model_path": training_result.get("model_path"),
                "mlflow_run_id": training_result.get("mlflow_run_id"),
                "error": training_result.get("error"),
            }
        else:
            logger.error("[L4] Could not retrieve L3 results from XCom")
            return {
                "status": "error",
                "experiment_name": experiment_name,
                "error": "Could not retrieve L3 results from XCom",
            }


@task(task_id="register_results")
def register_results(
    experiment_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Register experiment results in database.

    Args:
        experiment_result: Results from run_experiment

    Returns:
        Registration status
    """
    import sys
    sys.path.insert(0, "/opt/airflow/project")

    if experiment_result.get("status") == "dry_run":
        return {"registered": False, "reason": "dry_run"}

    if experiment_result.get("status") != "success":
        return {"registered": False, "reason": experiment_result.get("error")}

    try:
        import asyncio
        from src.experiments.experiment_registry import ExperimentRegistry
        from src.experiments.experiment_runner import ExperimentResult

        # Reconstruct result
        result = ExperimentResult(
            experiment_name=experiment_result["experiment_name"],
            experiment_version=experiment_result["experiment_version"],
            run_id=experiment_result["run_id"],
            status=experiment_result["status"],
            started_at=datetime.now(),  # Simplified
            completed_at=datetime.now(),
            duration_seconds=experiment_result["duration_seconds"],
            backtest_metrics=experiment_result.get("metrics", {}),
            model_path=experiment_result.get("model_path"),
            mlflow_run_id=experiment_result.get("mlflow_run_id"),
        )

        # Register (file-based fallback if no DB)
        from src.experiments.experiment_registry import FileBasedRegistry
        registry = FileBasedRegistry(Path("/opt/airflow/project/experiments"))
        path = registry.register_run(result)

        return {"registered": True, "path": path}

    except Exception as e:
        logger.error(f"Failed to register results: {e}")
        return {"registered": False, "reason": str(e)}


@task(task_id="compare_experiments")
def compare_experiments_task(
    experiment_result: Dict[str, Any],
    compare_with: str,
) -> Optional[Dict[str, Any]]:
    """
    Compare completed experiment with baseline.

    Args:
        experiment_result: Results from run_experiment
        compare_with: Baseline experiment name

    Returns:
        Comparison results or None
    """
    import sys
    sys.path.insert(0, "/opt/airflow/project")

    if not compare_with:
        logger.info("No comparison experiment specified")
        return None

    if experiment_result.get("status") != "success":
        logger.info("Experiment did not complete successfully, skipping comparison")
        return None

    try:
        from src.experiments import compare_experiments

        comparison = compare_experiments(
            baseline_name=compare_with,
            treatment_name=experiment_result["experiment_name"],
            primary_metric="sharpe_ratio",
        )

        return {
            "baseline": comparison.baseline_name,
            "treatment": comparison.treatment_name,
            "recommendation": comparison.recommendation,
            "primary_metric_comparison": comparison.metric_comparisons.get(
                "sharpe_ratio"
            ).to_dict() if "sharpe_ratio" in comparison.metric_comparisons else None,
        }

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return {"error": str(e)}


@task(task_id="send_notification")
def send_notification(
    experiment_result: Dict[str, Any],
    comparison_result: Optional[Dict[str, Any]],
    notify: bool,
) -> None:
    """
    Send notification about experiment completion.

    Args:
        experiment_result: Results from run_experiment
        comparison_result: Optional comparison results
        notify: Whether to send notification
    """
    if not notify:
        logger.info("Notifications disabled")
        return

    # Format message
    status = experiment_result.get("status", "unknown")
    exp_name = experiment_result.get("experiment_name", "unknown")

    if status == "success":
        metrics = experiment_result.get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", "N/A")
        msg = f"Experiment '{exp_name}' completed successfully. Sharpe: {sharpe}"
    elif status == "dry_run":
        msg = f"Experiment '{exp_name}' dry run completed (no training)"
    else:
        error = experiment_result.get("error", "Unknown error")
        msg = f"Experiment '{exp_name}' failed: {error}"

    if comparison_result and "recommendation" in comparison_result:
        msg += f"\n\nComparison: {comparison_result['recommendation']}"

    logger.info(f"Notification: {msg}")

    # TODO: Integrate with actual notification service (Slack, email, etc.)
    # For now, just log
    print(f"[NOTIFICATION] {msg}")


# =============================================================================
# DAG Definition - Orchestrator Pattern
# =============================================================================

with DAG(
    dag_id=L4_EXPERIMENT_RUNNER,
    default_args=default_args,
    description="Orchestrate A/B experiments via L2→L3 pipeline",
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["training", "experiments", "ml", "a/b-testing", "orchestrator"],
    params=dag_params,
    doc_md=__doc__,
    max_active_runs=2,  # Limit concurrent experiments
) as dag:

    # ==========================================================================
    # Phase 1: Validate configuration
    # ==========================================================================
    config_metadata = validate_config(
        experiment_name="{{ params.experiment_name }}",
    )

    # ==========================================================================
    # Phase 2: Trigger L2 preprocessing (generates experiment-specific dataset)
    # ==========================================================================
    l2_trigger = trigger_l2_preprocessing(
        config_metadata=config_metadata,
        dry_run="{{ params.dry_run }}",
    )

    # Wait for L2 to complete
    wait_for_l2 = ExternalTaskSensor(
        task_id="wait_for_l2",
        external_dag_id=L2_DAG_ID,
        external_task_id="build_rl_dataset",  # Updated to use SSOT task
        timeout=7200,  # 2 hours
        poke_interval=30,
        mode="reschedule",  # Free up worker while waiting
        execution_delta=timedelta(seconds=0),  # Same execution time
        allowed_states=["success"],
        failed_states=["failed", "upstream_failed"],
    )

    # ==========================================================================
    # Phase 3: Trigger L3 training (uses dataset from L2 XCom)
    # ==========================================================================
    l3_trigger = trigger_l3_training(
        config_metadata=config_metadata,
        l2_trigger_info=l2_trigger,
        dry_run="{{ params.dry_run }}",
    )

    # Wait for L3 to complete
    wait_for_l3 = ExternalTaskSensor(
        task_id="wait_for_l3",
        external_dag_id=L3_DAG_ID,
        external_task_id="train_model",
        timeout=14400,  # 4 hours for training
        poke_interval=60,
        mode="reschedule",
        execution_delta=timedelta(seconds=0),
        allowed_states=["success"],
        failed_states=["failed", "upstream_failed"],
    )

    # ==========================================================================
    # Phase 4: Collect results from L3 via XCom
    # ==========================================================================
    experiment_result = collect_experiment_results(
        config_metadata=config_metadata,
        l3_trigger_info=l3_trigger,
    )

    # ==========================================================================
    # Phase 5: Register, compare, and notify
    # ==========================================================================
    registration = register_results(
        experiment_result=experiment_result,
    )

    comparison_result = compare_experiments_task(
        experiment_result=experiment_result,
        compare_with="{{ params.compare_with }}",
    )

    notification = send_notification(
        experiment_result=experiment_result,
        comparison_result=comparison_result,
        notify="{{ params.notify_on_complete }}",
    )

    # ==========================================================================
    # Dependencies: L4 orchestrates L2 → L3 → Results
    # ==========================================================================
    config_metadata >> l2_trigger >> wait_for_l2 >> l3_trigger >> wait_for_l3
    wait_for_l3 >> experiment_result >> [registration, comparison_result]
    [registration, comparison_result] >> notification


# =============================================================================
# Additional DAGs for scheduled training
# =============================================================================

with DAG(
    dag_id=L4_SCHEDULED_RETRAINING,
    default_args=default_args,
    description="Scheduled model retraining based on drift detection",
    schedule_interval="0 2 * * 0",  # Weekly at 2 AM Sunday
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["training", "scheduled", "ml"],
    max_active_runs=1,
) as retraining_dag:

    @task(task_id="check_drift_status")
    def check_drift_status() -> Dict[str, Any]:
        """Check if drift warrants retraining."""
        import sys
        sys.path.insert(0, "/opt/airflow/project")

        try:
            import asyncio
            from services.inference_api.services.drift_persistence_service import (
                DriftPersistenceService,
            )

            # This would connect to actual drift database
            # For now, return placeholder
            return {
                "high_drift_detected": False,
                "drift_score": 0.15,
                "recommendation": "no_retrain_needed",
            }
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            return {"error": str(e)}

    @task(task_id="trigger_retraining")
    def trigger_retraining(drift_status: Dict[str, Any]) -> None:
        """Trigger retraining if needed."""
        from airflow.operators.trigger_dagrun import TriggerDagRunOperator

        if drift_status.get("high_drift_detected"):
            logger.info("High drift detected - triggering retraining")
            # Would trigger l4_experiment_runner with production config
        else:
            logger.info("No retraining needed")

    drift_status = check_drift_status()
    trigger_retraining(drift_status)
