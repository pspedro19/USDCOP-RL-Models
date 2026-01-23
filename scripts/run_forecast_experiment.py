#!/usr/bin/env python3
"""
Forecasting A/B Experiment Runner CLI
=====================================

Command-line interface for running and managing forecasting experiments.

Usage:
    # Run experiment locally
    python run_forecast_experiment.py --config baseline_v1.yaml

    # Trigger via Airflow API
    python run_forecast_experiment.py --config feature_oil_vix_v1.yaml --airflow

    # Compare two experiments
    python run_forecast_experiment.py --compare baseline_v1 feature_oil_vix_v1

    # List available experiments
    python run_forecast_experiment.py --list

Design Patterns:
    - Command Pattern: CLI operations as commands
    - Factory Pattern: Manager creation from config
    - SSOT: All configs from contracts

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
Contract: CTR-FORECAST-CLI-001
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS (SSOT)
# =============================================================================

CONFIG_DIR = PROJECT_ROOT / "config" / "forecast_experiments"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "forecasting" / "features.parquet"

# Airflow API settings
AIRFLOW_API_URL = os.environ.get("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "airflow")
DAG_ID = "forecast_l4_02_experiment_runner"


# =============================================================================
# CLI COMMANDS
# =============================================================================

def run_experiment_local(config_path: str, dataset_path: Optional[str] = None) -> int:
    """
    Run experiment locally (not via Airflow).

    Args:
        config_path: Path to experiment YAML config
        dataset_path: Optional path to dataset (defaults to standard location)

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    from src.forecasting.experiment_manager import (
        ForecastExperimentManager,
        ExperimentConfig,
    )

    # Load config
    config_file = resolve_config_path(config_path)
    logger.info(f"Loading config: {config_file}")

    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    # Build ExperimentConfig
    exp_config = build_experiment_config(config_dict)
    logger.info(f"Experiment: {exp_config.name} v{exp_config.version}")

    # Initialize manager
    manager = ForecastExperimentManager(
        config=exp_config,
        project_root=PROJECT_ROOT,
    )

    # Create run
    run = manager.create_run()
    logger.info(f"Created run: {run.run_id}")

    try:
        # Train
        logger.info("Starting training...")
        dataset = dataset_path or str(DEFAULT_DATASET_PATH)
        run = manager.train(run, dataset_path=dataset)

        if run.status == "failed":
            logger.error(f"Training failed: {run.error_message}")
            manager.save(run)
            return 1

        logger.info(f"Training completed. Status: {run.status}")

        # Backtest
        logger.info("Running backtest...")
        run = manager.backtest(run)
        logger.info(f"Backtest completed. Metrics: {len(run.backtest_metrics)} models")

        # Compare with baseline if specified
        if exp_config.baseline_experiment:
            logger.info(f"Comparing with baseline: {exp_config.baseline_experiment}")
            comparison = manager.compare_with_baseline(run)

            if comparison:
                logger.info("=" * 60)
                logger.info("A/B COMPARISON RESULTS")
                logger.info("=" * 60)
                logger.info(f"Recommendation: {comparison.recommendation.value}")
                logger.info(f"Confidence: {comparison.confidence_score:.2%}")
                logger.info(f"Treatment wins: {comparison.summary.get('treatment_wins', 0)}")
                logger.info(f"Baseline wins: {comparison.summary.get('baseline_wins', 0)}")
                logger.info("=" * 60)
            else:
                logger.warning("Comparison failed or baseline not found")

        # Save final state
        manager.save(run)

        # Print summary
        print_experiment_summary(run)

        return 0

    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        run.status = "failed"
        run.error_message = str(e)
        manager.save(run)
        return 1


def trigger_airflow_dag(config_path: str) -> int:
    """
    Trigger experiment via Airflow REST API.

    Args:
        config_path: Path to experiment YAML config

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    import requests
    from requests.auth import HTTPBasicAuth

    config_file = resolve_config_path(config_path)
    logger.info(f"Triggering Airflow DAG with config: {config_file}")

    # Load config for dag_run.conf
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Build API request
    url = f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dagRuns"
    auth = HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)

    payload = {
        "conf": config,
        "logical_date": datetime.now().isoformat() + "Z",
    }

    try:
        response = requests.post(
            url,
            json=payload,
            auth=auth,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code in (200, 201):
            result = response.json()
            dag_run_id = result.get("dag_run_id", "unknown")
            logger.info(f"DAG triggered successfully!")
            logger.info(f"  DAG Run ID: {dag_run_id}")
            logger.info(f"  State: {result.get('state', 'unknown')}")
            logger.info(f"  URL: {AIRFLOW_API_URL.replace('/api/v1', '')}/dags/{DAG_ID}/grid")
            return 0
        else:
            logger.error(f"Failed to trigger DAG: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return 1

    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        return 1


def compare_experiments(baseline_name: str, treatment_name: str) -> int:
    """
    Compare two existing experiments.

    Args:
        baseline_name: Name of baseline experiment
        treatment_name: Name of treatment experiment

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    from src.forecasting.experiment_manager import ExperimentRepository
    from src.forecasting.ab_statistics import ForecastABStatistics

    repo = ExperimentRepository()

    # Load experiments
    baseline_run = repo.get_latest_run(baseline_name)
    treatment_run = repo.get_latest_run(treatment_name)

    if baseline_run is None:
        logger.error(f"Baseline experiment not found: {baseline_name}")
        return 1

    if treatment_run is None:
        logger.error(f"Treatment experiment not found: {treatment_name}")
        return 1

    logger.info(f"Comparing experiments:")
    logger.info(f"  Baseline: {baseline_name} (run: {baseline_run.run_id})")
    logger.info(f"  Treatment: {treatment_name} (run: {treatment_run.run_id})")

    # Initialize A/B statistics
    ab_stats = ForecastABStatistics(
        alpha=0.05,
        bonferroni_correction=True,
    )

    # Prepare results for comparison
    baseline_results = {
        h: v for h, v in baseline_run.backtest_metrics.items()
        if isinstance(h, int)
    }
    treatment_results = {
        h: v for h, v in treatment_run.backtest_metrics.items()
        if isinstance(h, int)
    }

    # Load actual prices
    actual_prices = load_actual_prices()

    if actual_prices is None or actual_prices.empty:
        logger.error("Could not load actual prices for comparison")
        return 1

    # Note: Full comparison requires prediction-level data
    # This is a simplified comparison using aggregate metrics
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    for horizon in sorted(set(baseline_results.keys()) & set(treatment_results.keys())):
        b_da = baseline_run.aggregate_metrics.get("avg_direction_accuracy", 0)
        t_da = treatment_run.aggregate_metrics.get("avg_direction_accuracy", 0)
        diff = t_da - b_da

        logger.info(f"Horizon {horizon:2d}d: Baseline DA={b_da:.2%}, Treatment DA={t_da:.2%}, Diff={diff:+.2%}")

    logger.info("=" * 60)

    repo.close()
    return 0


def list_experiments() -> int:
    """
    List available experiments from database.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    from src.forecasting.experiment_manager import ExperimentRepository

    repo = ExperimentRepository()
    conn = repo._get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            experiment_name,
            experiment_version,
            run_id,
            status,
            completed_at,
            aggregate_metrics->>'avg_direction_accuracy' as avg_da
        FROM bi.forecast_experiment_runs
        ORDER BY completed_at DESC
        LIMIT 20
    """)

    rows = cur.fetchall()
    cur.close()
    repo.close()

    if not rows:
        logger.info("No experiments found in database")
        return 0

    print("\n" + "=" * 80)
    print("RECENT FORECAST EXPERIMENTS")
    print("=" * 80)
    print(f"{'Name':<25} {'Version':<10} {'Run ID':<20} {'Status':<10} {'Avg DA':<10}")
    print("-" * 80)

    for row in rows:
        name, version, run_id, status, completed, avg_da = row
        avg_da_str = f"{float(avg_da):.2%}" if avg_da else "N/A"
        print(f"{name:<25} {version:<10} {run_id[:18]:<20} {status:<10} {avg_da_str:<10}")

    print("=" * 80 + "\n")

    return 0


def list_configs() -> int:
    """
    List available experiment configs.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    if not CONFIG_DIR.exists():
        logger.info(f"Config directory not found: {CONFIG_DIR}")
        return 1

    yaml_files = list(CONFIG_DIR.glob("*.yaml")) + list(CONFIG_DIR.glob("*.yml"))

    if not yaml_files:
        logger.info("No config files found")
        return 0

    print("\n" + "=" * 70)
    print("AVAILABLE EXPERIMENT CONFIGS")
    print("=" * 70)
    print(f"{'File':<30} {'Name':<25} {'Baseline':<15}")
    print("-" * 70)

    for config_file in sorted(yaml_files):
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            exp = config.get("experiment", {})
            name = exp.get("name", "unknown")
            baseline = exp.get("baseline_experiment") or "(is baseline)"
            print(f"{config_file.name:<30} {name:<25} {baseline:<15}")
        except Exception as e:
            print(f"{config_file.name:<30} (error: {e})")

    print("=" * 70)
    print(f"\nDirectory: {CONFIG_DIR}\n")

    return 0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def resolve_config_path(config_path: str) -> Path:
    """Resolve config path to absolute path."""
    path = Path(config_path)

    # If absolute, use as-is
    if path.is_absolute() and path.exists():
        return path

    # Try relative to config directory
    config_file = CONFIG_DIR / config_path
    if config_file.exists():
        return config_file

    # Try adding .yaml extension
    config_file = CONFIG_DIR / f"{config_path}.yaml"
    if config_file.exists():
        return config_file

    # Try relative to current directory
    config_file = Path(config_path)
    if config_file.exists():
        return config_file.resolve()

    raise FileNotFoundError(f"Config file not found: {config_path}")


def build_experiment_config(config_dict: Dict[str, Any]):
    """Build ExperimentConfig from YAML dict."""
    from src.forecasting.experiment_manager import ExperimentConfig

    experiment = config_dict.get("experiment", {})
    models_config = config_dict.get("models", {})
    horizons_config = config_dict.get("horizons", {})
    training_config = config_dict.get("training", {})
    evaluation_config = config_dict.get("evaluation", {})

    return ExperimentConfig(
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


def load_actual_prices():
    """Load actual daily USDCOP prices."""
    import pandas as pd
    import psycopg2

    try:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            logger.warning("DATABASE_URL not set")
            return None

        conn = psycopg2.connect(db_url)
        df = pd.read_sql("""
            SELECT date, close
            FROM bi.dim_daily_usdcop
            ORDER BY date
        """, conn)
        conn.close()

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    except Exception as e:
        logger.warning(f"Could not load actual prices: {e}")
        return None


def print_experiment_summary(run):
    """Print experiment summary."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Experiment: {run.experiment_name}")
    print(f"Run ID: {run.run_id}")
    print(f"Status: {run.status}")

    if run.aggregate_metrics:
        avg_da = run.aggregate_metrics.get("avg_direction_accuracy", 0)
        print(f"Avg Direction Accuracy: {avg_da:.2%}")

        if "best_model_per_horizon" in run.aggregate_metrics:
            print("\nBest Model per Horizon:")
            for h, model in run.aggregate_metrics["best_model_per_horizon"].items():
                print(f"  {h}d: {model}")

    if run.backtest_metrics:
        print(f"\nModels Evaluated: {len(run.backtest_metrics)}")

    print("=" * 60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Forecasting A/B Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment locally
  %(prog)s --config baseline_v1.yaml

  # Trigger via Airflow
  %(prog)s --config feature_oil_vix_v1.yaml --airflow

  # Compare two experiments
  %(prog)s --compare baseline_v1 feature_oil_vix_v1

  # List experiments
  %(prog)s --list

  # List available configs
  %(prog)s --list-configs
        """,
    )

    parser.add_argument(
        "--config", "-c",
        help="Path to experiment YAML config",
    )

    parser.add_argument(
        "--airflow", "-a",
        action="store_true",
        help="Trigger experiment via Airflow API instead of local run",
    )

    parser.add_argument(
        "--dataset", "-d",
        help="Path to dataset (for local runs)",
    )

    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "TREATMENT"),
        help="Compare two experiments",
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List experiments from database",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available experiment configs",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Execute command
    if args.list:
        return list_experiments()

    if args.list_configs:
        return list_configs()

    if args.compare:
        return compare_experiments(args.compare[0], args.compare[1])

    if args.config:
        if args.airflow:
            return trigger_airflow_dag(args.config)
        else:
            return run_experiment_local(args.config, args.dataset)

    # No command specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
