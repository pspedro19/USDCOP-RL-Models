"""
DAG: v3.l3_model_training
=========================
USD/COP Trading System - V3 Architecture
Layer 3: Model Training Pipeline with MLflow Integration

Purpose:
    End-to-end RL model training pipeline with:
    - Dataset validation
    - Normalization statistics generation
    - Feature contract creation
    - PPO model training
    - Model registration
    - Backtest validation (optional)

Architecture:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  AIRFLOW    │────▶│   MLFLOW    │────▶│   MODEL     │
    │ Orchestrator│     │  Tracking   │     │  REGISTRY   │
    └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                    │
          ▼                   ▼                    ▼
    ┌─────────────────────────────────────────────────────┐
    │              TRAINING PIPELINE (Python)              │
    │  • EnvironmentFactory                                │
    │  • PPOTrainer                                        │
    │  • ContractFactory                                   │
    └─────────────────────────────────────────────────────┘

Schedule:
    Manual trigger (training is expensive)
    Can be triggered by:
    - Manual UI trigger with config
    - After L2 preprocessing pipeline
    - Scheduled weekly/monthly retraining

SOLID Principles:
    - SRP: Each task does one thing
    - OCP: Extensible via operator inheritance
    - DIP: Depends on abstractions (XCom, TrainingConfig)

Design Patterns:
    - Pipeline Pattern: Sequential stages with XCom
    - Observer Pattern: MLflow tracks all stages
    - Factory Pattern: Environment and contract creation
    - Strategy Pattern: Pluggable reward strategies

Features:
    - MLflow experiment tracking
    - XCom artifact passing
    - Hash verification for integrity
    - Alerting on failure
    - Configurable hyperparameters

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging
import os
import sys
import time

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
import pandas as pd

# Local imports
from utils.dag_common import get_db_connection, load_feature_config

# =============================================================================
# AUTO-INCREMENT VERSION
# =============================================================================


def get_next_model_version() -> str:
    """
    Get the next model version by auto-incrementing from the database.

    Queries model_registry for the highest version number and returns v(N+1).

    Returns:
        str: Next version string (e.g., "v2" if latest is "v1")
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Find the maximum version number from model_registry
        cur.execute("""
            SELECT MAX(
                CAST(
                    REGEXP_REPLACE(model_version, '[^0-9]', '', 'g')
                    AS INTEGER
                )
            ) as max_version
            FROM model_registry
            WHERE model_version ~ '^v?[0-9]+$'
        """)

        result = cur.fetchone()
        cur.close()
        conn.close()

        if result and result[0] is not None:
            next_version = result[0] + 1
            logging.info(f"Auto-increment: Found max version v{result[0]}, next is v{next_version}")
            return f"v{next_version}"
        else:
            logging.info("Auto-increment: No versions found in DB, starting at v1")
            return "v1"

    except Exception as e:
        logging.warning(f"Auto-increment failed: {e}. Using default v1")
        return "v1"


# =============================================================================
# CONFIGURATION
# =============================================================================

DAG_ID = 'v3.l3_model_training'

# Default training configuration (can be overridden via Airflow Variables)
# NOTE: version="auto" enables auto-increment from database
DEFAULT_TRAINING_CONFIG = {
    "version": "auto",  # "auto" = auto-increment, or explicit like "v1"
    "experiment_name": "ppo_usdcop",

    # Dataset
    "dataset_name": "RL_DS3_MACRO_CORE.csv",
    "dataset_dir": "5min",  # "5min" or "daily"

    # Features (13 market + 2 state = 15 observation dim)
    "feature_columns": [
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
        "position", "time_normalized"
    ],
    "state_features": ["position", "time_normalized"],

    # Technical indicators
    "rsi_period": 9,
    "atr_period": 10,
    "adx_period": 14,

    # Trading hours (Colombia)
    "trading_hours_start": "13:00",  # UTC (8:00 Bogota)
    "trading_hours_end": "17:55",    # UTC (12:55 Bogota)

    # Training hyperparameters
    "total_timesteps": 500_000,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.90,  # shorter-term focus for noisy 5-min data
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.05,  # more exploration

    # Environment - from config/trading_config.yaml SSOT
    "initial_capital": 10_000.0,
    "transaction_cost_bps": 75.0,  # From SSOT: costs.transaction_cost_bps
    "slippage_bps": 15.0,  # From SSOT: costs.slippage_bps

    # Action Thresholds - from config/trading_config.yaml SSOT
    "threshold_long": 0.33,  # From SSOT: thresholds.long
    "threshold_short": -0.33,  # From SSOT: thresholds.short

    # Train/Val/Test split
    "train_ratio": 0.70,
    "val_ratio": 0.15,

    # Options
    "auto_register": True,
    "run_backtest_validation": False,
    "backtest_start_date": None,
    "backtest_end_date": None,

    # MLflow
    "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "mlflow_enabled": True,
}


def get_training_config() -> Dict[str, Any]:
    """
    Get training configuration from Airflow Variables or defaults.

    Priority:
    1. Airflow Variable: training_config (JSON)
    2. DAG run conf
    3. Default config

    Special handling:
    - version="auto": Auto-increments from model_registry database
    """
    config = DEFAULT_TRAINING_CONFIG.copy()

    try:
        config_json = Variable.get("training_config", default_var=None)
        if config_json:
            user_config = json.loads(config_json)
            # Merge with defaults
            config = {**config, **user_config}
    except Exception as e:
        logging.warning(f"Could not load training_config Variable: {e}")

    # Resolve "auto" version to actual next version
    if config.get("version") == "auto":
        config["version"] = get_next_model_version()
        logging.info(f"Auto-increment resolved version to: {config['version']}")

    return config


def get_project_paths() -> Dict[str, Path]:
    """Get project paths based on environment"""
    # Docker paths
    docker_root = Path('/opt/airflow')
    if docker_root.exists():
        project_root = docker_root
    else:
        # Local development
        project_root = Path(__file__).parent.parent.parent

    return {
        "project_root": project_root,
        "data_dir": project_root / "data" / "pipeline" / "07_output",
        "config_dir": project_root / "config",
        "models_dir": project_root / "models",
        "contracts_dir": project_root / "config" / "contracts",
    }


PATHS = get_project_paths()


# =============================================================================
# MLFLOW UTILITIES
# =============================================================================

def init_mlflow(experiment_name: str, tracking_uri: Optional[str] = None):
    """Initialize MLflow for tracking"""
    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        logging.info(f"MLflow initialized: experiment={experiment_name}")
        return mlflow
    except ImportError:
        logging.warning("MLflow not installed. Experiment tracking disabled.")
        return None
    except Exception as e:
        logging.warning(f"MLflow initialization failed: {e}. Tracking disabled.")
        return None


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def validate_dataset(**context) -> Dict[str, Any]:
    """
    Task 1: Validate training dataset exists and has required columns.

    Outputs (XCom):
    - dataset_path: Path to validated dataset
    - dataset_info: Row count, column count, date range
    - dataset_hash: SHA256 hash for reproducibility
    """
    config = context['dag_run'].conf or get_training_config()

    logging.info(f"Validating dataset for version {config['version']}...")

    # Build dataset path
    dataset_dir = PATHS["data_dir"] / f"datasets_{config['dataset_dir']}"
    dataset_path = dataset_dir / config["dataset_name"]

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Run L2 preprocessing pipeline first."
        )

    # Load and validate
    df = pd.read_csv(dataset_path)

    # Check required columns
    missing = [c for c in config["feature_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Compute hash
    with open(dataset_path, 'rb') as f:
        dataset_hash = hashlib.sha256(f.read()).hexdigest()

    # Get date range if 'time' column exists
    date_range = {}
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        date_range = {
            "start": str(df['time'].min()),
            "end": str(df['time'].max()),
        }

    result = {
        "dataset_path": str(dataset_path),
        "dataset_hash": dataset_hash,
        "row_count": len(df),
        "column_count": len(df.columns),
        "feature_count": len(config["feature_columns"]),
        "date_range": date_range,
    }

    logging.info(f"✓ Dataset validated: {len(df):,} rows, {len(df.columns)} columns")

    # Push to XCom
    ti = context['ti']
    ti.xcom_push(key='dataset_path', value=str(dataset_path))
    ti.xcom_push(key='dataset_hash', value=dataset_hash)
    ti.xcom_push(key='dataset_info', value=result)

    return result


def generate_norm_stats(**context) -> Dict[str, Any]:
    """
    Task 2: Calculate and save normalization statistics.

    Inputs (XCom):
    - dataset_path: From validate_dataset

    Outputs (XCom):
    - norm_stats_path: Path to generated JSON
    - norm_stats_hash: SHA256 hash
    """
    config = context['dag_run'].conf or get_training_config()
    ti = context['ti']

    dataset_path = ti.xcom_pull(key='dataset_path', task_ids='validate_dataset')

    logging.info(f"Generating norm stats for {config['version']}...")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Calculate statistics for each feature
    norm_stats = {}
    feature_columns = [c for c in config["feature_columns"] if c not in config["state_features"]]

    for col in feature_columns:
        if col in df.columns:
            values = df[col].dropna()
            norm_stats[col] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    # Add metadata
    norm_stats["_metadata"] = {
        "version": config["version"],
        "created_at": datetime.utcnow().isoformat(),
        "dataset_hash": ti.xcom_pull(key='dataset_hash', task_ids='validate_dataset'),
        "feature_count": len(feature_columns),
        "sample_count": len(df),
    }

    # Save
    output_path = PATHS["config_dir"] / f"{config['version']}_norm_stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)

    # Compute hash
    normalized = json.dumps(norm_stats, sort_keys=True, separators=(',', ':'))
    norm_stats_hash = hashlib.sha256(normalized.encode()).hexdigest()

    logging.info(f"✓ Norm stats generated: {len(feature_columns)} features -> {output_path}")

    # Push to XCom
    ti.xcom_push(key='norm_stats_path', value=str(output_path))
    ti.xcom_push(key='norm_stats_hash', value=norm_stats_hash)
    ti.xcom_push(key='norm_stats', value=norm_stats)

    return {
        "norm_stats_path": str(output_path),
        "norm_stats_hash": norm_stats_hash,
        "feature_count": len(feature_columns),
    }


def create_contract(**context) -> Dict[str, Any]:
    """
    Task 3: Create feature contract for model.

    Inputs (XCom):
    - norm_stats_path: From generate_norm_stats
    - dataset_info: From validate_dataset

    Outputs (XCom):
    - contract_path: Path to contract JSON
    - contract_hash: SHA256 hash
    """
    config = context['dag_run'].conf or get_training_config()
    ti = context['ti']

    norm_stats_path = ti.xcom_pull(key='norm_stats_path', task_ids='generate_norm_stats')
    norm_stats_hash = ti.xcom_pull(key='norm_stats_hash', task_ids='generate_norm_stats')
    dataset_info = ti.xcom_pull(key='dataset_info', task_ids='validate_dataset')

    logging.info(f"Creating contract for {config['version']}...")

    # Build contract
    contract = {
        "version": config["version"],
        "observation_dim": len(config["feature_columns"]),
        "feature_order": config["feature_columns"],
        "norm_stats_path": str(Path(norm_stats_path).relative_to(PATHS["project_root"])),
        "model_path": f"models/ppo_{config['version']}_production/final_model.zip",

        # Technical indicators config
        "rsi_period": config["rsi_period"],
        "atr_period": config["atr_period"],
        "adx_period": config["adx_period"],
        "warmup_bars": max(config["rsi_period"], config["atr_period"], config["adx_period"]),

        # Trading hours
        "trading_hours_start": config["trading_hours_start"],
        "trading_hours_end": config["trading_hours_end"],

        # Metadata
        "created_at": datetime.utcnow().isoformat(),
        "created_from_dataset": dataset_info["dataset_path"],
        "dataset_hash": dataset_info["dataset_hash"],
        "norm_stats_hash": norm_stats_hash,
        "sample_count": dataset_info["row_count"],
    }

    # Compute contract hash
    normalized = json.dumps(contract, sort_keys=True, separators=(',', ':'))
    contract_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    contract["contract_hash"] = contract_hash

    # Save
    contracts_dir = PATHS["contracts_dir"]
    contracts_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contracts_dir / f"{config['version']}_contract.json"

    with open(contract_path, 'w') as f:
        json.dump(contract, f, indent=2)

    logging.info(f"✓ Contract created: {contract_path}")

    # Push to XCom
    ti.xcom_push(key='contract_path', value=str(contract_path))
    ti.xcom_push(key='contract_hash', value=contract_hash)
    ti.xcom_push(key='contract', value=contract)

    return {
        "contract_path": str(contract_path),
        "contract_hash": contract_hash,
        "observation_dim": contract["observation_dim"],
    }


def train_model(**context) -> Dict[str, Any]:
    """
    Task 4: Train PPO model using professional training infrastructure.

    This is the main training task that:
    1. Creates training/eval environments via EnvironmentFactory
    2. Configures PPO via PPOConfig
    3. Trains via PPOTrainer
    4. Logs to MLflow

    Inputs (XCom):
    - dataset_path: From validate_dataset
    - norm_stats_path: From generate_norm_stats
    - contract: From create_contract

    Outputs (XCom):
    - model_path: Path to trained model
    - model_hash: SHA256 hash
    - training_result: Metrics and duration
    """
    config = context['dag_run'].conf or get_training_config()
    ti = context['ti']

    dataset_path = Path(ti.xcom_pull(key='dataset_path', task_ids='validate_dataset'))
    norm_stats_path = Path(ti.xcom_pull(key='norm_stats_path', task_ids='generate_norm_stats'))
    contract = ti.xcom_pull(key='contract', task_ids='create_contract')

    logging.info(f"Training model {config['version']}...")
    logging.info(f"  Dataset: {dataset_path}")
    logging.info(f"  Timesteps: {config['total_timesteps']:,}")

    # Initialize MLflow
    mlflow = init_mlflow(
        experiment_name=config["experiment_name"],
        tracking_uri=config.get("mlflow_tracking_uri"),
    )

    # Add src to path for imports
    src_path = str(PATHS["project_root"] / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Import training infrastructure
    try:
        from training import (
            EnvironmentFactory,
            TradingEnvConfig,
            PPOTrainer,
            PPOConfig,
        )
    except ImportError as e:
        logging.error(f"Failed to import training infrastructure: {e}")
        raise

    start_time = time.time()

    # Create model output directory
    model_dir = PATHS["models_dir"] / f"ppo_{config['version']}_production"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Start MLflow run
    run_id = None
    if mlflow:
        run = mlflow.start_run(run_name=f"train_{config['version']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_params({
            "version": config["version"],
            "total_timesteps": config["total_timesteps"],
            "learning_rate": config["learning_rate"],
            "n_steps": config["n_steps"],
            "batch_size": config["batch_size"],
            "gamma": config["gamma"],
            "clip_range": config["clip_range"],
            "ent_coef": config["ent_coef"],
            "observation_dim": contract["observation_dim"],
        })

    try:
        # Create environment factory
        env_factory = EnvironmentFactory(project_root=PATHS["project_root"])

        # Configure environment
        env_config = TradingEnvConfig(
            observation_dim=contract["observation_dim"],
            initial_capital=config["initial_capital"],
            transaction_cost_bps=config["transaction_cost_bps"],
            random_episode_start=True,
            max_episode_steps=2000,
        )

        # Create train/eval environments
        env_dict = env_factory.create_train_eval_envs(
            dataset_path=dataset_path,
            norm_stats_path=norm_stats_path,
            config=env_config,
            train_ratio=config["train_ratio"],
            val_ratio=config["val_ratio"],
            n_train_envs=1,
            n_eval_envs=1,
        )

        train_env = env_dict["train"]
        eval_env = env_dict["val"]

        logging.info(
            f"Environments created: "
            f"train={env_dict['splits']['train_size']} bars, "
            f"val={env_dict['splits']['val_size']} bars"
        )

        # Configure PPO
        ppo_config = PPOConfig(
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            total_timesteps=config["total_timesteps"],
            eval_freq=max(config["total_timesteps"] // 20, 10000),
            n_eval_episodes=5,
            checkpoint_freq=max(config["total_timesteps"] // 10, 25000),
            tensorboard_log=True,
            verbose=1,
        )

        # Create trainer
        trainer = PPOTrainer(
            train_env=train_env,
            eval_env=eval_env,
            config=ppo_config,
            output_dir=model_dir,
            experiment_name=f"ppo_{config['version']}",
        )

        # Train
        result = trainer.train()

        # Cleanup
        train_env.close()
        eval_env.close()

        training_duration = time.time() - start_time

        # Log to MLflow
        if mlflow:
            mlflow.log_metrics({
                "training_duration_seconds": training_duration,
                "best_mean_reward": result.best_mean_reward,
                "final_mean_reward": result.final_mean_reward,
                "total_timesteps": result.total_timesteps,
            })

            # Log model artifact
            if result.model_path and result.model_path.exists():
                mlflow.log_artifact(str(result.model_path))

            mlflow.set_tag("status", "SUCCESS" if result.success else "FAILED")
            mlflow.end_run()

        if not result.success:
            raise RuntimeError(f"Training failed: {result.error_message}")

        logging.info(
            f"✓ Model trained in {training_duration/60:.1f} min, "
            f"best_reward={result.best_mean_reward:.2f}"
        )

        # Push to XCom
        ti.xcom_push(key='model_path', value=str(result.model_path))
        ti.xcom_push(key='model_hash', value=result.model_hash)
        ti.xcom_push(key='mlflow_run_id', value=run_id)
        ti.xcom_push(key='training_result', value=result.to_dict())

        return {
            "model_path": str(result.model_path),
            "model_hash": result.model_hash,
            "training_duration_seconds": training_duration,
            "best_mean_reward": result.best_mean_reward,
            "mlflow_run_id": run_id,
        }

    except Exception as e:
        if mlflow:
            mlflow.log_metric("success", 0)
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(e)[:250])
            mlflow.end_run()
        raise


def register_model(**context) -> Dict[str, Any]:
    """
    Task 5: Register model in database and MLflow Model Registry.

    Inputs (XCom):
    - model_path, model_hash: From train_model
    - contract: From create_contract
    - training_result: From train_model

    Outputs (XCom):
    - model_id: Registered model ID
    """
    config = context['dag_run'].conf or get_training_config()
    ti = context['ti']

    if not config.get("auto_register", True):
        logging.info("Auto-registration disabled, skipping")
        return {"skipped": True}

    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')
    model_hash = ti.xcom_pull(key='model_hash', task_ids='train_model')
    contract = ti.xcom_pull(key='contract', task_ids='create_contract')
    training_result = ti.xcom_pull(key='training_result', task_ids='train_model')
    norm_stats_hash = ti.xcom_pull(key='norm_stats_hash', task_ids='generate_norm_stats')

    logging.info(f"Registering model {config['version']}...")

    # Generate model ID
    model_id = f"ppo_{config['version']}_{model_hash[:8]}"

    # Register in database
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Use existing model_registry schema
        cur.execute("""
            INSERT INTO model_registry (
                model_id, model_version, model_path, model_hash,
                norm_stats_hash, config_hash, observation_dim,
                action_space, feature_order, validation_metrics,
                status, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'registered', NOW()
            )
            ON CONFLICT (model_id) DO UPDATE SET
                model_path = EXCLUDED.model_path,
                validation_metrics = EXCLUDED.validation_metrics,
                status = 'registered'
            RETURNING id
        """, (
            model_id,
            config["version"],
            model_path,
            model_hash,
            norm_stats_hash,
            contract["contract_hash"],
            contract["observation_dim"],
            3,  # action_space: LONG, SHORT, HOLD
            json.dumps(contract["feature_order"]),
            json.dumps(training_result or {}),
        ))

        db_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        logging.info(f"✓ Model registered in DB: {model_id} (id={db_id})")

    except Exception as e:
        logging.warning(f"DB registration failed: {e}. Continuing...")

    # Register in MLflow Model Registry
    try:
        import mlflow

        mlflow_run_id = ti.xcom_pull(key='mlflow_run_id', task_ids='train_model')
        if mlflow_run_id and model_path:
            # Register model
            model_uri = f"runs:/{mlflow_run_id}/model"

            result = mlflow.register_model(
                model_uri=model_uri,
                name=f"ppo_{config['version']}"
            )

            logging.info(f"✓ Model registered in MLflow: {result.name} v{result.version}")

    except Exception as e:
        logging.warning(f"MLflow registration failed: {e}. Continuing...")

    # Push to XCom
    ti.xcom_push(key='model_id', value=model_id)

    return {
        "model_id": model_id,
        "model_hash": model_hash,
    }


def decide_backtest(**context) -> str:
    """
    Branch: Decide whether to run backtest validation.

    Returns:
        Task ID to execute next
    """
    config = context['dag_run'].conf or get_training_config()

    if config.get("run_backtest_validation", False):
        if config.get("backtest_start_date") and config.get("backtest_end_date"):
            return "run_backtest_validation"

    return "skip_backtest"


def run_backtest_validation(**context) -> Dict[str, Any]:
    """
    Task 6a: Run backtest to validate model performance.

    Inputs (XCom):
    - model_id: From register_model

    Outputs (XCom):
    - backtest_result: Sharpe, max drawdown, win rate
    """
    config = context['dag_run'].conf or get_training_config()
    ti = context['ti']

    model_id = ti.xcom_pull(key='model_id', task_ids='register_model')

    logging.info(
        f"Running backtest validation: "
        f"{config['backtest_start_date']} to {config['backtest_end_date']}"
    )

    # TODO: Implement backtest via inference API
    # This would call the backtest endpoint with the new model

    logging.warning("Backtest validation not yet implemented. Skipping.")

    return {
        "status": "skipped",
        "reason": "Not implemented",
    }


def pipeline_summary(**context) -> Dict[str, Any]:
    """
    Task 7: Generate pipeline execution summary.

    Aggregates results from all stages and logs final status.
    """
    config = context['dag_run'].conf or get_training_config()
    ti = context['ti']

    # Pull all results
    dataset_info = ti.xcom_pull(key='dataset_info', task_ids='validate_dataset') or {}
    norm_stats_path = ti.xcom_pull(key='norm_stats_path', task_ids='generate_norm_stats')
    contract_path = ti.xcom_pull(key='contract_path', task_ids='create_contract')
    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')
    model_id = ti.xcom_pull(key='model_id', task_ids='register_model')
    training_result = ti.xcom_pull(key='training_result', task_ids='train_model') or {}

    # Build summary
    summary = {
        "status": "SUCCESS",
        "version": config["version"],
        "model_id": model_id,
        "model_path": model_path,
        "norm_stats_path": norm_stats_path,
        "contract_path": contract_path,
        "dataset_rows": dataset_info.get("row_count", 0),
        "training_duration_seconds": training_result.get("training_duration_seconds", 0),
        "best_mean_reward": training_result.get("best_mean_reward", 0),
        "total_timesteps": training_result.get("total_timesteps", 0),
    }

    # Log summary
    logging.info("=" * 70)
    logging.info("MODEL TRAINING PIPELINE SUMMARY")
    logging.info("=" * 70)
    logging.info(f"  Version: {summary['version']}")
    logging.info(f"  Model ID: {summary['model_id']}")
    logging.info(f"  Model Path: {summary['model_path']}")
    logging.info(f"  Dataset Rows: {summary['dataset_rows']:,}")
    logging.info(f"  Training Duration: {summary['training_duration_seconds']/60:.1f} min")
    logging.info(f"  Best Mean Reward: {summary['best_mean_reward']:.2f}")
    logging.info(f"  Total Timesteps: {summary['total_timesteps']:,}")
    logging.info("=" * 70)

    return summary


def on_failure_callback(context):
    """
    Callback for task failures.
    Sends alert and logs error details.
    """
    task_instance = context['task_instance']
    exception = context.get('exception')

    error_msg = (
        f"TRAINING PIPELINE FAILED\n"
        f"Task: {task_instance.task_id}\n"
        f"DAG: {task_instance.dag_id}\n"
        f"Error: {exception}\n"
        f"Execution Date: {context['execution_date']}"
    )

    logging.error(error_msg)

    # TODO: Send alert via Slack/Email
    # slack_alert(error_msg)


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 0,  # Training is expensive - no auto-retry
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': on_failure_callback,
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='V3 L3: End-to-end RL model training with MLflow tracking',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    max_active_runs=1,  # Only one training at a time
    tags=['v3', 'l3', 'training', 'rl', 'ppo', 'mlflow'],
    params={
        "version": "auto",
        "total_timesteps": 500_000,
        "run_backtest_validation": False,
    },
)

with dag:

    # Task 1: Validate Dataset
    task_validate = PythonOperator(
        task_id='validate_dataset',
        python_callable=validate_dataset,
        provide_context=True,
    )

    # Task 2: Generate Norm Stats
    task_norm_stats = PythonOperator(
        task_id='generate_norm_stats',
        python_callable=generate_norm_stats,
        provide_context=True,
    )

    # Task 3: Create Contract
    task_contract = PythonOperator(
        task_id='create_contract',
        python_callable=create_contract,
        provide_context=True,
    )

    # Task 4: Train Model (main training task)
    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
        execution_timeout=timedelta(hours=4),  # 4 hour timeout
    )

    # Task 5: Register Model
    task_register = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
        provide_context=True,
    )

    # Task 6: Branch - Decide backtest
    task_branch = BranchPythonOperator(
        task_id='decide_backtest',
        python_callable=decide_backtest,
        provide_context=True,
    )

    # Task 6a: Run Backtest (optional)
    task_backtest = PythonOperator(
        task_id='run_backtest_validation',
        python_callable=run_backtest_validation,
        provide_context=True,
    )

    # Task 6b: Skip Backtest
    task_skip_backtest = EmptyOperator(
        task_id='skip_backtest',
    )

    # Task 7: Pipeline Summary
    task_summary = PythonOperator(
        task_id='pipeline_summary',
        python_callable=pipeline_summary,
        provide_context=True,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Define dependencies
    # Main pipeline
    task_validate >> task_norm_stats >> task_contract >> task_train >> task_register

    # Backtest branch
    task_register >> task_branch
    task_branch >> [task_backtest, task_skip_backtest]

    # Summary (after backtest decision)
    [task_backtest, task_skip_backtest] >> task_summary
