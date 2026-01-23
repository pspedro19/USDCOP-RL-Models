"""
DAG: l3_02_forecasting_training
===============================
USD/COP Trading System - Forecasting Model Training Pipeline

This is a THIN WRAPPER around ForecastingEngine.
All training logic is in src/forecasting/engine.py (DRY principle).

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  L1 Feature Refresh (upstream)                          │
    │  - Generates features from OHLCV + Macro                │
    │  - Must complete before training                        │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │  Airflow DAG (this file)                                │
    │  - Orchestration only                                   │
    │  - XCom passing (using contracts)                       │
    │  - Alerting                                             │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │  ForecastingEngine (src/forecasting/engine.py)          │
    │  - ALL training logic                                   │
    │  - 9 models × 7 horizons = 63 combinations              │
    │  - Walk-forward validation                              │
    │  - MLflow logging                                       │
    │  - MinIO artifact storage                               │
    └─────────────────────────────────────────────────────────┘
                              ↓ PostgreSQL + MinIO
    ┌─────────────────────────────────────────────────────────┐
    │  L5b Forecasting Inference (downstream)                 │
    │  - Loads models from MinIO/Registry                     │
    │  - Weekly forecasts                                     │
    └─────────────────────────────────────────────────────────┘

Schedule: Monthly (1st day, 2:00 AM COT)
Models: Ridge, Bayesian Ridge, ARD, XGBoost, LightGBM, CatBoost, Hybrids
Horizons: 1, 5, 10, 15, 20, 25, 30 days

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
Contract: CTR-FORECASTING-001
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import os
import sys

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
sys.path.insert(0, '/opt/airflow')

# =============================================================================
# IMPORTS FROM SSOT
# =============================================================================

# DAG Registry - SSOT for DAG IDs and Tags
from contracts.dag_registry import (
    FORECAST_L3_MODEL_TRAINING,
    get_dag_tags,
)

DAG_ID = FORECAST_L3_MODEL_TRAINING
DAG_TAGS = get_dag_tags(DAG_ID)
PROJECT_ROOT = Path('/opt/airflow')

# Forecasting Contracts - SSOT for forecasting pipeline
try:
    from src.forecasting.contracts import (
        ForecastingTrainingRequest,
        ForecastingTrainingResult,
        HORIZONS,
        MODEL_IDS,
        FORECASTING_CONTRACT_VERSION,
        FORECASTING_CONTRACT_HASH,
    )
    FORECASTING_CONTRACTS_AVAILABLE = True
    logging.info(f"[SSOT] Forecasting contracts loaded: v{FORECASTING_CONTRACT_VERSION}")
except ImportError as e:
    FORECASTING_CONTRACTS_AVAILABLE = False
    logging.error(f"[SSOT] Forecasting contracts not available: {e}")

# Forecasting Engine - Core training logic
try:
    from src.forecasting.engine import ForecastingEngine
    ENGINE_AVAILABLE = True
    logging.info("[SSOT] ForecastingEngine loaded successfully")
except ImportError as e:
    ENGINE_AVAILABLE = False
    logging.error(f"[SSOT] ForecastingEngine not available: {e}")

# MLflow Integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("[SSOT] MLflow not available")

# MinIO Integration
try:
    from src.mlops.minio_client import MinIOClient
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logging.warning("[SSOT] MinIO client not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "version": "auto",
    "experiment_name": "forecasting_usdcop",
    "dataset_path": None,  # Auto-detect from L1 features
    "use_db": False,  # v2.0: If True, use SSOT (PostgreSQL/Parquet) instead of dataset file
    "models": None,  # None = all 9 models
    "horizons": None,  # None = all 7 horizons
    "mlflow_enabled": True,
    "mlflow_tracking_uri": None,
    "minio_enabled": True,
    "walk_forward_windows": 5,
    # MinIO storage config
    "minio_bucket": "forecasting-models",
}


def get_training_config(**context) -> Dict[str, Any]:
    """Get training configuration from Airflow Variables or DAG run conf."""
    config = DEFAULT_CONFIG.copy()

    # Override from Variable
    try:
        var_config = Variable.get("forecasting_training_config", default_var=None)
        if var_config:
            config.update(json.loads(var_config))
    except Exception as e:
        logging.warning(f"Could not load forecasting_training_config Variable: {e}")

    # Override from dag_run.conf
    if context.get('dag_run') and context['dag_run'].conf:
        config.update(context['dag_run'].conf)

    # Auto-increment version if needed
    if config.get("version") == "auto":
        config["version"] = _get_next_version()

    return config


def _get_next_version() -> str:
    """Get next forecasting model version from database."""
    try:
        from utils.dag_common import get_db_connection
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT MAX(CAST(REGEXP_REPLACE(version, '[^0-9]', '', 'g') AS INTEGER))
            FROM bi.dim_models
            WHERE model_type = 'forecasting' AND version ~ '^v?[0-9]+$'
        """)
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result and result[0]:
            return f"v{result[0] + 1}"
        return "v1"
    except Exception as e:
        logging.warning(f"Auto-version failed: {e}")
        return f"v{datetime.now().strftime('%Y%m%d')}"


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def validate_environment(**context) -> Dict[str, Any]:
    """
    Task 1: Validate that all required components are available.

    Checks:
    - ForecastingEngine available
    - Forecasting contracts available
    - Database connection working
    - MLflow (optional)
    - MinIO (optional)
    """
    logging.info("=" * 60)
    logging.info("Validating forecasting training environment")
    logging.info("=" * 60)

    validation_results = {
        "engine_available": ENGINE_AVAILABLE,
        "contracts_available": FORECASTING_CONTRACTS_AVAILABLE,
        "mlflow_available": MLFLOW_AVAILABLE,
        "minio_available": MINIO_AVAILABLE,
        "db_connection": False,
    }

    # Check database connection
    try:
        from utils.dag_common import get_db_connection
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        validation_results["db_connection"] = True
    except Exception as e:
        logging.error(f"Database connection failed: {e}")

    # Critical checks
    if not ENGINE_AVAILABLE:
        raise RuntimeError("ForecastingEngine not available. Check imports.")

    if not FORECASTING_CONTRACTS_AVAILABLE:
        raise RuntimeError("Forecasting contracts not available. Check imports.")

    if not validation_results["db_connection"]:
        raise RuntimeError("Database connection failed.")

    # Log contract info
    logging.info(f"[SSOT] Contract version: {FORECASTING_CONTRACT_VERSION}")
    logging.info(f"[SSOT] Contract hash: {FORECASTING_CONTRACT_HASH}")
    logging.info(f"[SSOT] Models: {len(MODEL_IDS)} ({', '.join(MODEL_IDS[:3])}...)")
    logging.info(f"[SSOT] Horizons: {HORIZONS}")

    context['ti'].xcom_push(key='validation_results', value=validation_results)
    return validation_results


def prepare_dataset(**context) -> Dict[str, Any]:
    """
    Task 2: Prepare dataset for training.

    Dataset resolution priority:
    1. dag_run.conf['dataset_path'] - Explicit path
    2. Latest features from PostgreSQL
    3. Fallback to default dataset location
    """
    logging.info("Preparing dataset for forecasting training...")

    config = get_training_config(**context)
    ti = context['ti']

    dataset_path = None
    dataset_source = None

    # Priority 1: Explicit path from config
    if config.get('dataset_path'):
        dataset_path = Path(config['dataset_path'])
        dataset_source = "dag_run.conf"
        logging.info(f"[DATASET] Using explicit path: {dataset_path}")

    # Priority 2: Build from PostgreSQL features
    if dataset_path is None:
        try:
            from utils.dag_common import get_db_connection
            import pandas as pd

            conn = get_db_connection()

            # Query features for training
            query = """
                SELECT *
                FROM bi.v_forecasting_features
                WHERE date >= NOW() - INTERVAL '5 years'
                ORDER BY date ASC
            """
            df = pd.read_sql(query, conn)
            conn.close()

            if len(df) > 0:
                # Save to temporary location
                dataset_dir = PROJECT_ROOT / "data" / "cache" / "forecasting"
                dataset_dir.mkdir(parents=True, exist_ok=True)

                dataset_path = dataset_dir / f"forecasting_features_{datetime.now().strftime('%Y%m%d')}.parquet"
                df.to_parquet(dataset_path, index=False)

                dataset_source = "postgresql"
                logging.info(f"[DATASET] Loaded {len(df)} rows from PostgreSQL")
            else:
                logging.warning("[DATASET] No data found in PostgreSQL")

        except Exception as e:
            logging.warning(f"[DATASET] PostgreSQL query failed: {e}")

    # Priority 3: Fallback to default location
    if dataset_path is None or not dataset_path.exists():
        fallback_paths = [
            PROJECT_ROOT / "data" / "pipeline" / "forecasting" / "features.parquet",
            PROJECT_ROOT / "data" / "forecasting" / "features_daily.csv",
            PROJECT_ROOT / "NewFeature" / "consolidated_backend" / "data" / "usdcop_daily_features.csv",
        ]

        for path in fallback_paths:
            if path.exists():
                dataset_path = path
                dataset_source = "fallback"
                logging.info(f"[DATASET] Using fallback path: {dataset_path}")
                break

    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError("No dataset found for forecasting training")

    # Compute dataset hash for lineage
    import hashlib
    with open(dataset_path, 'rb') as f:
        dataset_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    result = {
        "dataset_path": str(dataset_path),
        "dataset_source": dataset_source,
        "dataset_hash": dataset_hash,
    }

    ti.xcom_push(key='dataset_info', value=result)
    logging.info(f"[DATASET] Prepared: {dataset_path} (hash: {dataset_hash})")

    return result


def run_training(**context) -> Dict[str, Any]:
    """
    Task 3: Run forecasting model training.

    This delegates ALL training logic to ForecastingEngine.
    The DAG only handles orchestration.

    Trains 9 models × 7 horizons = 63 model/horizon combinations
    with walk-forward validation.
    """
    if not ENGINE_AVAILABLE:
        raise RuntimeError("ForecastingEngine not available")

    config = get_training_config(**context)
    ti = context['ti']

    # Get dataset info from previous task
    dataset_info = ti.xcom_pull(task_ids='prepare_dataset', key='dataset_info')
    if not dataset_info:
        raise ValueError("No dataset info from prepare_dataset task")

    logging.info("=" * 60)
    logging.info(f"Starting forecasting training v{config['version']}")
    logging.info("=" * 60)

    # Build training request (SSOT integration v2.0)
    use_db = config.get('use_db', False)
    dataset_path = None if use_db else dataset_info.get('dataset_path')

    request = ForecastingTrainingRequest(
        dataset_path=dataset_path,
        use_db=use_db,  # v2.0: Use PostgreSQL/Parquet as SSOT
        version=config['version'],
        experiment_name=config.get('experiment_name', 'forecasting_usdcop'),
        models=config.get('models'),  # None = all models
        horizons=config.get('horizons'),  # None = all horizons
        mlflow_enabled=config.get('mlflow_enabled', True),
        mlflow_tracking_uri=config.get('mlflow_tracking_uri') or os.environ.get("MLFLOW_TRACKING_URI"),
        minio_enabled=config.get('minio_enabled', True),
        db_connection_string=os.environ.get("DATABASE_URL"),
        walk_forward_windows=config.get('walk_forward_windows', 5),
    )

    logging.info(f"[TRAINING] Request: {len(request.models)} models × {len(request.horizons)} horizons")
    logging.info(f"[TRAINING] Walk-forward windows: {request.walk_forward_windows}")

    # Initialize and run engine
    engine = ForecastingEngine(project_root=PROJECT_ROOT)
    result = engine.train(request)

    if not result.success:
        raise RuntimeError(f"Forecasting training failed: {result.errors}")

    # Push results to XCom
    result_dict = result.to_dict()
    ti.xcom_push(key='training_result', value=result_dict)
    ti.xcom_push(key='models_trained', value=result.models_trained)
    ti.xcom_push(key='best_models', value=result.best_model_per_horizon)
    ti.xcom_push(key='mlflow_run_ids', value=result.mlflow_run_ids)
    ti.xcom_push(key='model_artifacts_path', value=result.model_artifacts_path)

    logging.info("=" * 60)
    logging.info(f"Training complete: {result.models_trained} models trained")
    logging.info(f"Duration: {result.training_duration_seconds/60:.1f} min")
    logging.info(f"Best models per horizon: {result.best_model_per_horizon}")
    if result.minio_artifacts_uri:
        logging.info(f"MinIO artifacts: {result.minio_artifacts_uri}")
    logging.info("=" * 60)

    return result_dict


def persist_metrics(**context) -> Dict[str, Any]:
    """
    Task 4: Persist training metrics to PostgreSQL.

    Saves:
    - bi.dim_models: Model catalog
    - bi.fact_model_metrics: Walk-forward metrics
    """
    logging.info("Persisting forecasting metrics to PostgreSQL...")

    ti = context['ti']
    training_result = ti.xcom_pull(task_ids='run_training', key='training_result')

    if not training_result:
        return {"status": "no_results"}

    try:
        from utils.dag_common import get_db_connection
        conn = get_db_connection()
        cur = conn.cursor()

        version = training_result.get('version', 'unknown')
        metrics_summary = training_result.get('metrics_summary', {})
        best_models = training_result.get('best_model_per_horizon', {})

        # Insert model entries
        for model_id in MODEL_IDS:
            cur.execute("""
                INSERT INTO bi.dim_models (model_id, model_name, model_type, version, created_at)
                VALUES (%s, %s, 'forecasting', %s, NOW())
                ON CONFLICT (model_id) DO UPDATE SET
                    version = EXCLUDED.version,
                    updated_at = NOW()
            """, (
                f"forecasting_{model_id}_{version}",
                model_id,
                version,
            ))

        # Insert metrics per model/horizon
        for model_id, horizon_metrics in metrics_summary.items():
            for horizon, da in horizon_metrics.items():
                cur.execute("""
                    INSERT INTO bi.fact_model_metrics
                    (model_id, horizon, direction_accuracy, version, metric_date)
                    VALUES (%s, %s, %s, %s, CURRENT_DATE)
                    ON CONFLICT DO NOTHING
                """, (
                    f"forecasting_{model_id}_{version}",
                    horizon,
                    da,
                    version,
                ))

        conn.commit()
        cur.close()
        conn.close()

        logging.info(f"[DB] Persisted metrics for {len(metrics_summary)} models")
        return {"status": "success", "models_persisted": len(metrics_summary)}

    except Exception as e:
        logging.error(f"Error persisting metrics: {e}")
        return {"status": "error", "error": str(e)}


def training_summary(**context) -> Dict[str, Any]:
    """
    Task 5: Generate training summary.
    """
    ti = context['ti']
    training_result = ti.xcom_pull(task_ids='run_training', key='training_result') or {}
    validation = ti.xcom_pull(task_ids='validate_environment', key='validation_results') or {}

    logging.info("=" * 60)
    logging.info("FORECASTING TRAINING SUMMARY")
    logging.info("=" * 60)
    logging.info(f"  Version: {training_result.get('version', 'N/A')}")
    logging.info(f"  Models Trained: {training_result.get('models_trained', 0)}")
    logging.info(f"  Total Combinations: {training_result.get('total_combinations', 0)}")
    logging.info(f"  Duration: {training_result.get('training_duration_seconds', 0)/60:.1f} min")
    logging.info(f"  MLflow: {'Enabled' if validation.get('mlflow_available') else 'Disabled'}")
    logging.info(f"  MinIO: {'Enabled' if validation.get('minio_available') else 'Disabled'}")

    best_models = training_result.get('best_model_per_horizon', {})
    if best_models:
        logging.info("  Best Models per Horizon:")
        for h, model in best_models.items():
            logging.info(f"    H{h}: {model}")

    logging.info("=" * 60)

    return training_result


def on_failure_callback(context):
    """Handle task failures."""
    task = context['task_instance']
    error = context.get('exception')

    logging.error(
        f"FORECASTING TRAINING FAILED\n"
        f"Task: {task.task_id}\n"
        f"Error: {error}"
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
    'retry_delay': timedelta(minutes=10),
    'on_failure_callback': on_failure_callback,
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Forecasting model training pipeline (9 models × 7 horizons)',
    schedule_interval='0 2 1 * *',  # First day of month, 2:00 AM
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS,
    params={
        "version": "auto",
        "models": None,
        "horizons": None,
    },
)

with dag:
    # Task 1: Validate environment
    validate_task = PythonOperator(
        task_id='validate_environment',
        python_callable=validate_environment,
        provide_context=True,
    )

    # Task 2: Prepare dataset
    prepare_task = PythonOperator(
        task_id='prepare_dataset',
        python_callable=prepare_dataset,
        provide_context=True,
    )

    # Task 3: Run training (main task)
    train_task = PythonOperator(
        task_id='run_training',
        python_callable=run_training,
        provide_context=True,
        execution_timeout=timedelta(hours=4),
    )

    # Task 4: Persist metrics to DB
    persist_task = PythonOperator(
        task_id='persist_metrics',
        python_callable=persist_metrics,
        provide_context=True,
    )

    # Task 5: Summary
    summary_task = PythonOperator(
        task_id='training_summary',
        python_callable=training_summary,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Task chain
    validate_task >> prepare_task >> train_task >> persist_task >> summary_task
