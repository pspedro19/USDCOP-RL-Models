"""
DAG: l5_02_forecasting_inference
================================
USD/COP Trading System - Weekly Forecasting Inference Pipeline

This is a THIN WRAPPER around ForecastingEngine.predict().
All inference logic is in src/forecasting/engine.py (DRY principle).

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  L3b Forecasting Training (upstream)                    │
    │  - Trained 9 models × 7 horizons = 63 models           │
    │  - Stored in MinIO/Model Registry                       │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │  Airflow DAG (this file)                                │
    │  - Orchestration only                                   │
    │  - Loads latest features                                │
    │  - Triggers inference on all models                     │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │  ForecastingEngine.predict()                            │
    │  - Load models from MinIO/Registry                      │
    │  - Generate predictions for 7 horizons                  │
    │  - Create 3 ensemble strategies                         │
    │  - Upload forecast images to MinIO                      │
    └─────────────────────────────────────────────────────────┘
                              ↓ PostgreSQL
    ┌─────────────────────────────────────────────────────────┐
    │  Output Tables                                          │
    │  - bi.fact_forecasts: Individual predictions            │
    │  - bi.fact_consensus: Ensemble consensus                │
    │  - MinIO: Visualization images                          │
    └─────────────────────────────────────────────────────────┘

Schedule: Weekly (Sundays 6:00 AM COT)
Output: 63 predictions + 3 ensembles + consensus per horizon

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
    FORECAST_L5_WEEKLY_INFERENCE,
    FORECAST_L3_MODEL_TRAINING,
    get_dag_tags,
)

DAG_ID = FORECAST_L5_WEEKLY_INFERENCE
DAG_TAGS = get_dag_tags(DAG_ID)
PROJECT_ROOT = Path('/opt/airflow')

# Forecasting Contracts - SSOT for forecasting pipeline
try:
    from src.forecasting.contracts import (
        ForecastingInferenceRequest,
        ForecastingInferenceResult,
        ForecastPrediction,
        HORIZONS,
        MODEL_IDS,
        HORIZON_LABELS,
        EnsembleType,
        FORECASTING_CONTRACT_VERSION,
        FORECASTING_CONTRACT_HASH,
    )
    FORECASTING_CONTRACTS_AVAILABLE = True
    logging.info(f"[SSOT] Forecasting contracts loaded: v{FORECASTING_CONTRACT_VERSION}")
except ImportError as e:
    FORECASTING_CONTRACTS_AVAILABLE = False
    logging.error(f"[SSOT] Forecasting contracts not available: {e}")

# Forecasting Engine - Core inference logic
try:
    from src.forecasting.engine import ForecastingEngine
    ENGINE_AVAILABLE = True
    logging.info("[SSOT] ForecastingEngine loaded successfully")
except ImportError as e:
    ENGINE_AVAILABLE = False
    logging.error(f"[SSOT] ForecastingEngine not available: {e}")

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
    "inference_date": None,  # Auto = today
    "models": None,  # None = all 9 models
    "horizons": None,  # None = all 7 horizons
    "generate_ensembles": True,
    "upload_images": True,
    "persist_to_db": True,
    # Model loading config
    "models_bucket": "forecasting-models",
    "models_version": "latest",
}


def get_inference_config(**context) -> Dict[str, Any]:
    """Get inference configuration from Airflow Variables or DAG run conf."""
    config = DEFAULT_CONFIG.copy()

    # Override from Variable
    try:
        var_config = Variable.get("forecasting_inference_config", default_var=None)
        if var_config:
            config.update(json.loads(var_config))
    except Exception as e:
        logging.warning(f"Could not load forecasting_inference_config Variable: {e}")

    # Override from dag_run.conf
    if context.get('dag_run') and context['dag_run'].conf:
        config.update(context['dag_run'].conf)

    # Set inference date if not provided
    if config.get("inference_date") is None:
        config["inference_date"] = datetime.now().strftime("%Y-%m-%d")

    return config


def _get_models_path(version: str = "latest") -> Optional[Path]:
    """Get path to trained models."""
    # Priority 1: MinIO
    if MINIO_AVAILABLE:
        try:
            client = MinIOClient()
            # Check for latest version in MinIO
            objects = list(client.client.list_objects(
                "forecasting-models",
                prefix=f"{version}/" if version != "latest" else "",
                recursive=False
            ))
            if objects:
                # Return MinIO URI
                return Path(f"s3://forecasting-models/{version}")
        except Exception as e:
            logging.warning(f"MinIO models lookup failed: {e}")

    # Priority 2: Local paths
    local_paths = [
        PROJECT_ROOT / "models" / "forecasting" / version,
        PROJECT_ROOT / "models" / "forecasting" / "latest",
        PROJECT_ROOT / "NewFeature" / "consolidated_backend" / "outputs" / "models",
    ]

    for path in local_paths:
        if path.exists():
            return path

    return None


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def validate_environment(**context) -> Dict[str, Any]:
    """
    Task 1: Validate that all required components are available.
    """
    logging.info("=" * 60)
    logging.info("Validating forecasting inference environment")
    logging.info("=" * 60)

    validation_results = {
        "engine_available": ENGINE_AVAILABLE,
        "contracts_available": FORECASTING_CONTRACTS_AVAILABLE,
        "minio_available": MINIO_AVAILABLE,
        "db_connection": False,
        "models_found": False,
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

    # Check models exist
    config = get_inference_config(**context)
    models_path = _get_models_path(config.get("models_version", "latest"))
    if models_path:
        validation_results["models_found"] = True
        validation_results["models_path"] = str(models_path)
        logging.info(f"[MODELS] Found at: {models_path}")
    else:
        logging.warning("[MODELS] No trained models found")

    # Critical checks
    if not ENGINE_AVAILABLE:
        raise RuntimeError("ForecastingEngine not available. Check imports.")

    if not FORECASTING_CONTRACTS_AVAILABLE:
        raise RuntimeError("Forecasting contracts not available. Check imports.")

    if not validation_results["models_found"]:
        raise RuntimeError("No trained forecasting models found. Run L3b training first.")

    # Log contract info
    logging.info(f"[SSOT] Contract version: {FORECASTING_CONTRACT_VERSION}")
    logging.info(f"[SSOT] Models: {len(MODEL_IDS)}")
    logging.info(f"[SSOT] Horizons: {HORIZONS}")

    context['ti'].xcom_push(key='validation_results', value=validation_results)
    return validation_results


def load_latest_features(**context) -> Dict[str, Any]:
    """
    Task 2: Load latest features for inference.

    Queries the most recent feature row from PostgreSQL
    to use as input for all model predictions.
    """
    logging.info("Loading latest features for inference...")

    ti = context['ti']
    config = get_inference_config(**context)
    inference_date = config['inference_date']

    try:
        from utils.dag_common import get_db_connection
        import pandas as pd

        conn = get_db_connection()

        # Query latest features
        query = """
            SELECT *
            FROM bi.v_forecasting_features
            WHERE date <= %s::date
            ORDER BY date DESC
            LIMIT 1
        """
        df = pd.read_sql(query, conn, params=(inference_date,))
        conn.close()

        if len(df) == 0:
            raise ValueError(f"No features found for date {inference_date}")

        # Get current price from OHLCV
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT close FROM usdcop_m5_ohlcv
            ORDER BY time DESC LIMIT 1
        """)
        price_result = cur.fetchone()
        cur.close()
        conn.close()

        current_price = float(price_result[0]) if price_result else None

        features_dict = df.iloc[0].to_dict()
        feature_date = str(features_dict.get('date', inference_date))

        result = {
            "features": features_dict,
            "feature_date": feature_date,
            "current_price": current_price,
            "feature_count": len(features_dict),
        }

        ti.xcom_push(key='feature_data', value=result)
        logging.info(f"[FEATURES] Loaded {len(features_dict)} features for date {feature_date}")
        logging.info(f"[FEATURES] Current price: {current_price}")

        return result

    except Exception as e:
        logging.error(f"Error loading features: {e}")
        raise


def run_inference(**context) -> Dict[str, Any]:
    """
    Task 3: Run forecasting inference on all models.

    This delegates ALL inference logic to ForecastingEngine.
    Generates predictions for 9 models × 7 horizons + 3 ensembles.
    """
    if not ENGINE_AVAILABLE:
        raise RuntimeError("ForecastingEngine not available")

    config = get_inference_config(**context)
    ti = context['ti']

    # Get feature data from previous task
    feature_data = ti.xcom_pull(task_ids='load_latest_features', key='feature_data')
    validation = ti.xcom_pull(task_ids='validate_environment', key='validation_results')

    if not feature_data:
        raise ValueError("No feature data from load_latest_features task")

    logging.info("=" * 60)
    logging.info(f"Running forecasting inference for {config['inference_date']}")
    logging.info("=" * 60)

    # Build inference request
    request = ForecastingInferenceRequest(
        inference_date=config['inference_date'],
        models=config.get('models'),  # None = all models
        horizons=config.get('horizons'),  # None = all horizons
        generate_ensembles=config.get('generate_ensembles', True),
        upload_images=config.get('upload_images', True),
        persist_to_db=config.get('persist_to_db', True),
    )

    logging.info(f"[INFERENCE] Models: {len(request.models)}")
    logging.info(f"[INFERENCE] Horizons: {request.horizons}")
    logging.info(f"[INFERENCE] Ensembles: {request.generate_ensembles}")

    # Get models path
    models_path = validation.get('models_path')

    # Initialize and run engine
    import pandas as pd
    features_df = pd.DataFrame([feature_data['features']])

    engine = ForecastingEngine(project_root=PROJECT_ROOT)
    result = engine.predict(
        request=request,
        models_path=models_path,
        features=features_df,
        current_price=feature_data.get('current_price'),
    )

    if not result.success:
        raise RuntimeError(f"Forecasting inference failed: {result.errors}")

    # Convert result to serializable dict
    result_dict = {
        "success": result.success,
        "inference_date": result.inference_date,
        "inference_week": result.inference_week,
        "inference_year": result.inference_year,
        "predictions_count": len(result.predictions),
        "ensembles_count": len(result.ensembles),
        "consensus": result.consensus_by_horizon,
        "minio_week_path": result.minio_week_path,
        "images_uploaded": result.images_uploaded,
        "forecasts_persisted": result.forecasts_persisted,
        "errors": result.errors,
    }

    # Push results to XCom
    ti.xcom_push(key='inference_result', value=result_dict)

    # Push predictions for downstream tasks
    predictions_summary = []
    for pred in result.predictions:
        predictions_summary.append({
            "model_id": pred.model_id,
            "horizon": pred.horizon,
            "direction": pred.direction.value,
            "predicted_return_pct": pred.predicted_return_pct,
            "signal": pred.signal,
        })
    ti.xcom_push(key='predictions', value=predictions_summary)

    logging.info("=" * 60)
    logging.info(f"Inference complete: {len(result.predictions)} predictions")
    logging.info(f"Ensembles: {list(result.ensembles.keys())}")
    logging.info(f"Images uploaded: {result.images_uploaded}")
    logging.info("=" * 60)

    return result_dict


def persist_forecasts(**context) -> Dict[str, Any]:
    """
    Task 4: Persist forecasts to PostgreSQL.

    Saves:
    - bi.fact_forecasts: Individual model predictions
    - bi.fact_consensus: Consensus by horizon
    """
    logging.info("Persisting forecasts to PostgreSQL...")

    ti = context['ti']
    inference_result = ti.xcom_pull(task_ids='run_inference', key='inference_result')
    predictions = ti.xcom_pull(task_ids='run_inference', key='predictions')
    feature_data = ti.xcom_pull(task_ids='load_latest_features', key='feature_data')

    if not inference_result:
        return {"status": "no_results"}

    try:
        from utils.dag_common import get_db_connection
        conn = get_db_connection()
        cur = conn.cursor()

        inference_date = inference_result['inference_date']
        inference_week = inference_result['inference_week']
        inference_year = inference_result['inference_year']
        current_price = feature_data.get('current_price', 0)

        # Create tables if not exist
        cur.execute("""
            CREATE SCHEMA IF NOT EXISTS bi;

            CREATE TABLE IF NOT EXISTS bi.fact_forecasts (
                id SERIAL PRIMARY KEY,
                inference_date DATE NOT NULL,
                inference_week INT,
                inference_year INT,
                model_id VARCHAR(100) NOT NULL,
                horizon INT NOT NULL,
                base_price DECIMAL(12,4),
                predicted_price DECIMAL(12,4),
                predicted_return_pct DECIMAL(10,6),
                direction VARCHAR(10),
                signal INT,
                confidence DECIMAL(5,4),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(inference_date, model_id, horizon)
            );

            CREATE TABLE IF NOT EXISTS bi.fact_consensus (
                id SERIAL PRIMARY KEY,
                inference_date DATE NOT NULL,
                horizon INT NOT NULL,
                bullish_count INT,
                bearish_count INT,
                consensus_direction VARCHAR(10),
                consensus_strength DECIMAL(5,4),
                avg_predicted_return DECIMAL(10,6),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(inference_date, horizon)
            );

            CREATE INDEX IF NOT EXISTS idx_fact_forecasts_date
                ON bi.fact_forecasts (inference_date DESC);
            CREATE INDEX IF NOT EXISTS idx_fact_consensus_date
                ON bi.fact_consensus (inference_date DESC);
        """)

        # Insert predictions
        inserted_count = 0
        for pred in predictions or []:
            cur.execute("""
                INSERT INTO bi.fact_forecasts
                (inference_date, inference_week, inference_year, model_id, horizon,
                 base_price, predicted_return_pct, direction, signal)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (inference_date, model_id, horizon) DO UPDATE SET
                    predicted_return_pct = EXCLUDED.predicted_return_pct,
                    direction = EXCLUDED.direction,
                    signal = EXCLUDED.signal,
                    created_at = NOW()
            """, (
                inference_date,
                inference_week,
                inference_year,
                pred['model_id'],
                pred['horizon'],
                current_price,
                pred['predicted_return_pct'],
                pred['direction'],
                pred['signal'],
            ))
            inserted_count += 1

        # Insert/update consensus
        consensus = inference_result.get('consensus', {})
        for horizon, cons in consensus.items():
            cur.execute("""
                INSERT INTO bi.fact_consensus
                (inference_date, horizon, bullish_count, bearish_count,
                 consensus_direction, consensus_strength, avg_predicted_return)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (inference_date, horizon) DO UPDATE SET
                    bullish_count = EXCLUDED.bullish_count,
                    bearish_count = EXCLUDED.bearish_count,
                    consensus_direction = EXCLUDED.consensus_direction,
                    consensus_strength = EXCLUDED.consensus_strength,
                    avg_predicted_return = EXCLUDED.avg_predicted_return,
                    created_at = NOW()
            """, (
                inference_date,
                int(horizon),
                cons.get('bullish_count', 0),
                cons.get('bearish_count', 0),
                cons.get('direction', 'NEUTRAL'),
                cons.get('strength', 0.5),
                cons.get('avg_return', 0),
            ))

        conn.commit()
        cur.close()
        conn.close()

        logging.info(f"[DB] Persisted {inserted_count} forecasts and {len(consensus)} consensus records")
        return {"status": "success", "forecasts_persisted": inserted_count}

    except Exception as e:
        logging.error(f"Error persisting forecasts: {e}")
        return {"status": "error", "error": str(e)}


def generate_csv_backup(**context) -> Dict[str, Any]:
    """
    Task 5: Generate CSV backup for frontend fallback.

    Creates bi_dashboard_unified.csv for cases where
    the frontend cannot reach the database.
    """
    logging.info("Generating CSV backup for frontend...")

    ti = context['ti']
    inference_result = ti.xcom_pull(task_ids='run_inference', key='inference_result')
    predictions = ti.xcom_pull(task_ids='run_inference', key='predictions')

    if not predictions:
        return {"status": "no_predictions"}

    try:
        import pandas as pd

        # Build DataFrame
        df = pd.DataFrame(predictions)

        # Add metadata columns
        df['inference_date'] = inference_result['inference_date']
        df['inference_week'] = inference_result['inference_week']
        df['inference_year'] = inference_result['inference_year']

        # Save to CSV
        csv_path = PROJECT_ROOT / "data" / "forecasting" / "bi_dashboard_unified.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

        # Also upload to MinIO if available
        if MINIO_AVAILABLE:
            try:
                client = MinIOClient()
                minio_path = f"forecasting/dashboards/{inference_result['inference_year']}/week_{inference_result['inference_week']}/bi_dashboard_unified.csv"
                client.upload_file(str(csv_path), "forecasting", minio_path)
                logging.info(f"[MinIO] CSV backup uploaded: {minio_path}")
            except Exception as e:
                logging.warning(f"MinIO CSV upload failed: {e}")

        logging.info(f"[CSV] Backup saved to: {csv_path}")
        return {"status": "success", "csv_path": str(csv_path)}

    except Exception as e:
        logging.error(f"Error generating CSV backup: {e}")
        return {"status": "error", "error": str(e)}


def inference_summary(**context) -> Dict[str, Any]:
    """
    Task 6: Generate inference summary.
    """
    ti = context['ti']
    inference_result = ti.xcom_pull(task_ids='run_inference', key='inference_result') or {}
    predictions = ti.xcom_pull(task_ids='run_inference', key='predictions') or []

    logging.info("=" * 60)
    logging.info("FORECASTING INFERENCE SUMMARY")
    logging.info("=" * 60)
    logging.info(f"  Date: {inference_result.get('inference_date', 'N/A')}")
    logging.info(f"  Week: {inference_result.get('inference_week', 'N/A')}/{inference_result.get('inference_year', 'N/A')}")
    logging.info(f"  Predictions: {inference_result.get('predictions_count', 0)}")
    logging.info(f"  Ensembles: {inference_result.get('ensembles_count', 0)}")
    logging.info(f"  Images Uploaded: {inference_result.get('images_uploaded', 0)}")
    logging.info(f"  Forecasts Persisted: {inference_result.get('forecasts_persisted', 0)}")

    # Consensus summary
    consensus = inference_result.get('consensus', {})
    if consensus:
        logging.info("  Consensus by Horizon:")
        for h, cons in sorted(consensus.items(), key=lambda x: int(x[0])):
            dir_str = cons.get('direction', 'N/A')
            strength = cons.get('strength', 0)
            logging.info(f"    H{h}: {dir_str} ({strength:.1%})")

    logging.info("=" * 60)

    return inference_result


def on_failure_callback(context):
    """Handle task failures."""
    task = context['task_instance']
    error = context.get('exception')

    logging.error(
        f"FORECASTING INFERENCE FAILED\n"
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
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': on_failure_callback,
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Weekly forecasting inference (9 models × 7 horizons + ensembles)',
    schedule_interval='0 6 * * 0',  # Sundays 6:00 AM
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS,
    params={
        "inference_date": None,
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

    # Task 2: Load latest features
    features_task = PythonOperator(
        task_id='load_latest_features',
        python_callable=load_latest_features,
        provide_context=True,
    )

    # Task 3: Run inference (main task)
    inference_task = PythonOperator(
        task_id='run_inference',
        python_callable=run_inference,
        provide_context=True,
        execution_timeout=timedelta(hours=1),
    )

    # Task 4: Persist to PostgreSQL
    persist_task = PythonOperator(
        task_id='persist_forecasts',
        python_callable=persist_forecasts,
        provide_context=True,
    )

    # Task 5: Generate CSV backup
    csv_task = PythonOperator(
        task_id='generate_csv_backup',
        python_callable=generate_csv_backup,
        provide_context=True,
    )

    # Task 6: Summary
    summary_task = PythonOperator(
        task_id='inference_summary',
        python_callable=inference_summary,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Task chain
    validate_task >> features_task >> inference_task >> [persist_task, csv_task] >> summary_task
