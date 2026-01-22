"""
DAG: v3.l2_preprocessing_pipeline
==================================
USD/COP Trading System - V3 Architecture
Layer 2: Data Preprocessing Pipeline

Purpose:
    Orchestrates the complete data preprocessing pipeline to generate
    RL-ready training datasets from raw OHLCV and Macro data.

    Pipeline Steps:
    1. Export database data to source files
    2. Run fusion (03_fusion)
    3. Run cleaning (04_cleaning)
    4. Run resampling (05_resampling)
    5. Build RL datasets using CanonicalFeatureBuilder (SSOT)
    6. Validate output datasets
    7. Push results to XCom for L3/L4 consumption

Schedule:
    Manual trigger or after data updates

Features:
    - Uses CanonicalFeatureBuilder (SSOT for features)
    - Reads date_ranges.yaml (SSOT for dates)
    - Supports experiment-specific datasets
    - XCom contracts for inter-DAG communication
    - Progress logging and metrics

Author: Pipeline Automatizado
Version: 2.0.0
Created: 2025-12-26
Updated: 2026-01-18 (SSOT refactoring)
"""

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import os
import logging
import json
import hashlib
from io import StringIO
from typing import Dict, Any, Optional, List

import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import pandas as pd
import psycopg2
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

from utils.dag_common import get_db_connection

# XCom Contracts - SSOT for inter-DAG communication
try:
    from airflow.dags.contracts.xcom_contracts import (
        L2XComKeysEnum,
        L2Output,
        compute_file_hash,
    )
    XCOM_CONTRACTS_AVAILABLE = True
except ImportError:
    XCOM_CONTRACTS_AVAILABLE = False
    logging.warning("[SSOT] XCom contracts not available")

# Feature Contract - SSOT for feature order
try:
    from src.core.contracts.feature_contract import (
        FEATURE_ORDER,
        FEATURE_ORDER_HASH,
    )
    FEATURE_CONTRACT_AVAILABLE = True
except ImportError:
    FEATURE_CONTRACT_AVAILABLE = False
    logging.warning("[SSOT] Feature contract not available")

# CanonicalFeatureBuilder - SSOT for feature calculation
try:
    from src.feature_store.builders import CanonicalFeatureBuilder
    CANONICAL_BUILDER_AVAILABLE = True
except ImportError:
    CANONICAL_BUILDER_AVAILABLE = False
    logging.warning("[SSOT] CanonicalFeatureBuilder not available")

# MinIO-First Architecture: ExperimentManager for storage
try:
    from src.ml_workflow.experiment_manager import ExperimentManager
    EXPERIMENT_MANAGER_AVAILABLE = True
except ImportError:
    EXPERIMENT_MANAGER_AVAILABLE = False
    logging.warning("[SSOT] ExperimentManager not available - using local storage")

from contracts.dag_registry import L2_DATASET_BUILD

DAG_ID = L2_DATASET_BUILD

# Pipeline paths
def get_pipeline_paths():
    """Get pipeline paths based on environment."""
    docker_path = Path('/app/data/pipeline')
    if docker_path.exists():
        return docker_path

    # Local development
    project_root = Path('/opt/airflow') if Path('/opt/airflow').exists() else Path(__file__).parent.parent.parent.parent
    return project_root / 'data' / 'pipeline'


PIPELINE_DIR = get_pipeline_paths()
SOURCES_DIR = PIPELINE_DIR / '01_sources'
OUTPUT_DIR = PIPELINE_DIR / '07_output'

# Script execution timeout (30 minutes)
SCRIPT_TIMEOUT = 1800


# =============================================================================
# SSOT HELPER FUNCTIONS
# =============================================================================


def load_date_ranges_ssot() -> Dict[str, Any]:
    """
    Load date ranges from SSOT config (config/date_ranges.yaml).

    This is the authoritative source for all date ranges in the system.

    Returns:
        Dict with training, validation, test date ranges

    Raises:
        FileNotFoundError: If SSOT config not found
    """
    # Docker path
    config_path = Path('/opt/airflow/config/date_ranges.yaml')
    if not config_path.exists():
        # Local development fallback
        config_path = Path(__file__).parent.parent.parent / 'config' / 'date_ranges.yaml'

    if not config_path.exists():
        raise FileNotFoundError(f"SSOT date_ranges.yaml not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_experiment_config(experiment_name: str) -> Optional[Dict[str, Any]]:
    """
    Load experiment-specific configuration from YAML.

    Args:
        experiment_name: Name of the experiment (e.g., 'baseline_full_macro')

    Returns:
        Experiment config dict or None if not found
    """
    # Docker path
    config_path = Path(f'/opt/airflow/config/experiments/{experiment_name}.yaml')
    if not config_path.exists():
        # Local development fallback
        config_path = Path(__file__).parent.parent.parent / 'config' / 'experiments' / f'{experiment_name}.yaml'

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            logging.info(f"Loaded experiment config: {experiment_name}")
            return config

    logging.warning(f"Experiment config not found: {experiment_name}")
    return None


def get_market_features() -> List[str]:
    """
    Get list of market features from SSOT (feature contract).

    Returns first 13 features (market features, excluding state features).
    """
    if FEATURE_CONTRACT_AVAILABLE:
        # First 13 features are market features
        return list(FEATURE_ORDER[:13])
    else:
        # Fallback if contract not available
        return [
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d"
        ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_pipeline_script(script_path: Path, timeout: int = SCRIPT_TIMEOUT) -> dict:
    """
    Execute a pipeline Python script.

    Args:
        script_path: Path to the script
        timeout: Execution timeout in seconds

    Returns:
        Dict with execution result
    """
    if not script_path.exists():
        return {
            'success': False,
            'error': f'Script not found: {script_path}',
            'stdout': '',
            'stderr': ''
        }

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(script_path.parent)
        )

        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout[-2000:] if result.stdout else '',  # Last 2000 chars
            'stderr': result.stderr[-2000:] if result.stderr else '',
            'error': None if result.returncode == 0 else f'Exit code: {result.returncode}'
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': f'Script timeout after {timeout}s',
            'stdout': '',
            'stderr': ''
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'stdout': '',
            'stderr': ''
        }


def count_output_datasets() -> dict:
    """Count generated datasets in output directory."""
    results = {
        '5min': [],
        'daily': [],
        '15min': []
    }

    for timeframe in ['5min', 'daily', '15min']:
        ds_dir = OUTPUT_DIR / f'datasets_{timeframe}'
        if ds_dir.exists():
            datasets = list(ds_dir.glob('RL_DS*.csv'))
            results[timeframe] = [
                {
                    'name': d.name,
                    'size_mb': round(d.stat().st_size / (1024 * 1024), 2)
                }
                for d in datasets
            ]

    return results


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def export_db_to_sources(**context):
    """
    Export database tables to source CSV files for preprocessing.
    This step ensures preprocessing uses the latest data from DB.
    """
    logging.info("Exporting database data to source files...")

    conn = get_db_connection()

    results = {
        'ohlcv_exported': 0,
        'macro_exported': 0
    }

    try:
        # Create source directories if needed
        ohlcv_dir = SOURCES_DIR / '16_usdcop_historical'
        ohlcv_dir.mkdir(parents=True, exist_ok=True)

        # Export OHLCV
        cur = conn.cursor()
        buffer = StringIO()

        cur.copy_expert(
            """COPY (
                SELECT time, open, high, low, close, volume
                FROM usdcop_m5_ohlcv
                WHERE symbol = 'USD/COP'
                ORDER BY time
            ) TO STDOUT WITH CSV HEADER""",
            buffer
        )

        buffer.seek(0)
        ohlcv_file = ohlcv_dir / 'USDCOP_M5_HISTORICAL.csv'
        with open(ohlcv_file, 'w') as f:
            f.write(buffer.read())

        # Count exported rows
        cur.execute("SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE symbol = 'USD/COP'")
        results['ohlcv_exported'] = cur.fetchone()[0]
        cur.close()

        logging.info(f"Exported {results['ohlcv_exported']:,} OHLCV rows to {ohlcv_file.name}")

        # Note: Macro data is typically in 05_resampling/output from previous runs
        # We just verify it exists
        macro_file = PIPELINE_DIR / '05_resampling' / 'output' / 'MACRO_DAILY_CONSOLIDATED.csv'
        if macro_file.exists():
            df_macro = pd.read_csv(macro_file)
            results['macro_exported'] = len(df_macro)
            logging.info(f"Macro data available: {results['macro_exported']:,} rows")
        else:
            logging.warning("MACRO_DAILY_CONSOLIDATED.csv not found - will use existing source files")

        context['ti'].xcom_push(key='export_results', value=results)
        return results

    finally:
        conn.close()


def run_fusion(**context):
    """Run the fusion step (03_fusion)."""
    logging.info("Running data fusion...")

    script = PIPELINE_DIR / '03_fusion' / 'run_fusion.py'
    result = run_pipeline_script(script)

    if result['success']:
        logging.info("Fusion completed successfully")
        if result['stdout']:
            logging.info(f"Output: {result['stdout']}")
    else:
        logging.error(f"Fusion failed: {result['error']}")
        if result['stderr']:
            logging.error(f"Stderr: {result['stderr']}")
        raise Exception(f"Fusion step failed: {result['error']}")

    context['ti'].xcom_push(key='fusion_result', value=result)
    return result


def run_cleaning(**context):
    """Run the cleaning step (04_cleaning)."""
    logging.info("Running data cleaning...")

    script = PIPELINE_DIR / '04_cleaning' / 'run_clean.py'
    result = run_pipeline_script(script)

    if result['success']:
        logging.info("Cleaning completed successfully")
    else:
        logging.error(f"Cleaning failed: {result['error']}")
        raise Exception(f"Cleaning step failed: {result['error']}")

    context['ti'].xcom_push(key='cleaning_result', value=result)
    return result


def run_resampling(**context):
    """Run the resampling step (05_resampling)."""
    logging.info("Running data resampling...")

    script = PIPELINE_DIR / '05_resampling' / 'run_resample.py'
    result = run_pipeline_script(script)

    if result['success']:
        logging.info("Resampling completed successfully")
    else:
        logging.error(f"Resampling failed: {result['error']}")
        raise Exception(f"Resampling step failed: {result['error']}")

    context['ti'].xcom_push(key='resampling_result', value=result)
    return result


def run_rl_dataset_builder(**context):
    """Run the RL dataset builder step (06_rl_dataset_builder) - LEGACY subprocess method."""
    logging.info("Building RL datasets (legacy subprocess method)...")

    # Run multiple scripts in sequence
    scripts = [
        PIPELINE_DIR / '06_rl_dataset_builder' / '01_build_5min_datasets.py',
        PIPELINE_DIR / '06_rl_dataset_builder' / '02_build_daily_datasets.py',
    ]

    all_results = []

    for script in scripts:
        if script.exists():
            logging.info(f"Running: {script.name}")
            result = run_pipeline_script(script)

            if not result['success']:
                logging.error(f"Dataset builder failed at {script.name}: {result['error']}")
                raise Exception(f"Dataset builder failed: {result['error']}")

            all_results.append({
                'script': script.name,
                'success': result['success']
            })
        else:
            logging.warning(f"Script not found: {script}")

    logging.info("RL dataset building completed")
    context['ti'].xcom_push(key='dataset_builder_result', value=all_results)
    return all_results


def build_rl_dataset_ssot(**context) -> Dict[str, Any]:
    """
    Build RL dataset using CanonicalFeatureBuilder (SSOT).

    This task replaces the legacy subprocess-based approach.
    Uses SSOT for:
    - Date ranges (config/date_ranges.yaml)
    - Feature calculation (CanonicalFeatureBuilder)
    - Feature order (FEATURE_ORDER contract)

    Supports two modes:
    1. Experiment mode: Generates dataset for specific experiment
    2. Default mode: Generates generic dataset with all market features

    XCom Output (via L2Output contract):
    - dataset_path: Path to generated dataset
    - dataset_hash: SHA256 hash for lineage
    - feature_columns: List of features in dataset
    - row_count: Number of rows
    - experiment_name: Name of experiment (if applicable)

    Args:
        **context: Airflow context with dag_run.conf

    Returns:
        Dict with dataset metadata (also pushed to XCom)
    """
    ti = context['ti']

    logging.info("=" * 60)
    logging.info("Building RL Dataset with SSOT (CanonicalFeatureBuilder)")
    logging.info("=" * 60)

    # ========================================================================
    # 1. LOAD SSOT DATE RANGES
    # ========================================================================
    try:
        date_ranges = load_date_ranges_ssot()
        logging.info(f"Loaded SSOT date ranges from config/date_ranges.yaml")
    except FileNotFoundError as e:
        logging.error(f"Failed to load date ranges: {e}")
        raise

    # ========================================================================
    # 2. CHECK FOR EXPERIMENT-SPECIFIC CONFIG
    # ========================================================================
    experiment_name = None
    if context.get('dag_run') and context['dag_run'].conf:
        experiment_name = context['dag_run'].conf.get('experiment_name')

    experiment_config = None

    if experiment_name:
        logging.info(f"Building dataset for experiment: {experiment_name}")
        experiment_config = load_experiment_config(experiment_name)

        if experiment_config:
            # Use experiment-specific settings
            data_config = experiment_config.get('data', {})
            train_start = data_config.get('train_start', date_ranges['experiment_training']['start'])
            train_end = data_config.get('train_end', date_ranges['experiment_training']['end'])
            feature_columns = data_config.get('feature_columns', get_market_features())
            logging.info(f"Using experiment config: {len(feature_columns)} features")
        else:
            # Experiment specified but no config found - use defaults
            logging.warning(f"No config found for {experiment_name}, using defaults")
            train_start = date_ranges['experiment_training']['start']
            train_end = date_ranges['experiment_training']['end']
            feature_columns = get_market_features()
    else:
        # Default mode - no specific experiment
        logging.info("Building default dataset (no experiment specified)")
        train_start = date_ranges['training']['start']
        train_end = date_ranges['training']['end']
        feature_columns = get_market_features()

    logging.info(f"Date range: {train_start} to {train_end}")
    logging.info(f"Features ({len(feature_columns)}): {feature_columns[:5]}...")

    # ========================================================================
    # 3. LOAD SOURCE DATA
    # ========================================================================
    # Use pre-processed files from previous pipeline steps
    ohlcv_path = PIPELINE_DIR / '03_fusion' / 'output' / 'USDCOP_M5_FUSED.csv'
    macro_path = PIPELINE_DIR / '05_resampling' / 'output' / 'MACRO_DAILY_CONSOLIDATED.csv'

    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV fused data not found: {ohlcv_path}")

    logging.info(f"Loading OHLCV from: {ohlcv_path}")
    ohlcv_df = pd.read_csv(ohlcv_path, parse_dates=['time'])
    logging.info(f"Loaded {len(ohlcv_df)} OHLCV rows")

    macro_df = None
    if macro_path.exists():
        logging.info(f"Loading macro from: {macro_path}")
        macro_df = pd.read_csv(macro_path, parse_dates=['date'])
        logging.info(f"Loaded {len(macro_df)} macro rows")
    else:
        logging.warning("Macro data not found - macro features will be zeros")

    # ========================================================================
    # 4. USE CANONICAL FEATURE BUILDER (SSOT - NO SUBPROCESS!)
    # ========================================================================
    if CANONICAL_BUILDER_AVAILABLE:
        logging.info("Using CanonicalFeatureBuilder for SSOT feature calculation")
        builder = CanonicalFeatureBuilder()

        # Filter OHLCV by date range
        ohlcv_filtered = ohlcv_df[
            (ohlcv_df['time'] >= train_start) &
            (ohlcv_df['time'] <= train_end)
        ].copy()

        logging.info(f"Filtered to {len(ohlcv_filtered)} rows in date range")

        # Merge macro data if available
        if macro_df is not None:
            ohlcv_filtered['date'] = ohlcv_filtered['time'].dt.date
            macro_df['date'] = pd.to_datetime(macro_df['date']).dt.date
            ohlcv_filtered = ohlcv_filtered.merge(macro_df, on='date', how='left')

            # Forward fill macro data
            macro_cols = ['dxy', 'vix', 'embi', 'brent', 'treasury_10y', 'treasury_2y', 'usdmxn']
            for col in macro_cols:
                if col in ohlcv_filtered.columns:
                    ohlcv_filtered[col] = ohlcv_filtered[col].ffill()

        # Compute features using canonical builder
        features_df = builder.compute_features(ohlcv_filtered, include_state=False)

        # Prepare final dataset
        df = pd.DataFrame()
        df['time'] = ohlcv_filtered['time'].values[:len(features_df)]

        # Add requested features
        for col in feature_columns:
            if col in features_df.columns:
                df[col] = features_df[col].values
            else:
                logging.warning(f"Feature {col} not computed, using zeros")
                df[col] = 0.0

        # Drop NaN rows
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)
        if dropped > 0:
            logging.info(f"Dropped {dropped} rows with NaN values")

    else:
        # Fallback: Load from existing CSV if builder not available
        logging.warning("CanonicalFeatureBuilder not available, falling back to CSV")
        fallback_path = OUTPUT_DIR / 'datasets_5min' / 'RL_DS3_MACRO_CORE.csv'
        if fallback_path.exists():
            df = pd.read_csv(fallback_path, parse_dates=['time'])
        else:
            raise RuntimeError("Neither CanonicalFeatureBuilder nor fallback CSV available")

    logging.info(f"Built dataset: {len(df)} rows, {len(df.columns)} columns")

    # ========================================================================
    # 5. SAVE DATASET - MINIO-FIRST ARCHITECTURE
    # ========================================================================
    # MinIO-First: Use ExperimentManager to save to object storage
    # Falls back to local storage if ExperimentManager not available

    dataset_snapshot = None
    dataset_path = None
    norm_stats_path = None
    manifest_path = None
    dataset_hash = None
    norm_stats_hash = None
    feature_order_hash_val = FEATURE_ORDER_HASH if FEATURE_CONTRACT_AVAILABLE else "unknown"

    # Generate version from timestamp
    version = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_name and EXPERIMENT_MANAGER_AVAILABLE:
        # ====================================================================
        # MINIO-FIRST PATH: Use ExperimentManager for S3 storage
        # ====================================================================
        logging.info(f"[MinIO-First] Saving dataset to MinIO for experiment: {experiment_name}")

        try:
            manager = ExperimentManager(experiment_name)

            # Prepare metadata for dataset
            metadata = {
                "date_range_start": train_start,
                "date_range_end": train_end,
                "canonical_builder_version": CanonicalFeatureBuilder.VERSION if CANONICAL_BUILDER_AVAILABLE else 'fallback',
                "dag_run_id": context.get('dag_run').run_id if context.get('dag_run') else None,
            }

            # Save dataset to MinIO - returns DatasetSnapshot
            dataset_snapshot = manager.save_dataset(
                data=df,
                version=version,
                metadata=metadata,
            )

            logging.info(f"[MinIO-First] Dataset saved to: {dataset_snapshot.storage_uri}")
            logging.info(f"[MinIO-First] Data hash: {dataset_snapshot.data_hash}")
            logging.info(f"[MinIO-First] Norm stats: {dataset_snapshot.norm_stats_uri}")

            # Extract values from snapshot
            dataset_hash = dataset_snapshot.data_hash
            norm_stats_hash = dataset_snapshot.schema_hash  # Use schema hash as norm stats proxy

            # Also save locally for backward compatibility and validation
            output_dir = OUTPUT_DIR / 'experiments' / experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = output_dir / 'train.parquet'
            df.to_parquet(dataset_path, index=False)
            logging.info(f"[MinIO-First] Also saved locally for validation: {dataset_path}")

        except Exception as e:
            logging.error(f"[MinIO-First] Failed to save to MinIO: {e}")
            logging.warning("[MinIO-First] Falling back to local storage")
            dataset_snapshot = None

    # Fallback or non-experiment mode: use local storage
    if dataset_snapshot is None:
        if experiment_name:
            # Experiment-specific directory structure
            output_dir = OUTPUT_DIR / 'experiments' / experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save as parquet for efficiency
            dataset_path = output_dir / 'train.parquet'
            df.to_parquet(dataset_path, index=False)

            # Also save as CSV for compatibility
            csv_path = output_dir / 'train.csv'
            df.to_csv(csv_path, index=False)

            # Compute and save norm stats
            norm_stats = {}
            for col in feature_columns:
                if col in df.columns:
                    norm_stats[col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                    }
            norm_stats_path = output_dir / 'norm_stats.json'
            with open(norm_stats_path, 'w') as f:
                json.dump(norm_stats, f, indent=2)

            manifest_path = output_dir / 'manifest.json'
        else:
            # Default path for backward compatibility
            output_dir = OUTPUT_DIR / 'datasets_5min'
            output_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = output_dir / 'RL_DS3_MACRO_CORE.parquet'
            df.to_parquet(dataset_path, index=False)

            # Also save as CSV for compatibility
            csv_path = output_dir / 'RL_DS3_MACRO_CORE.csv'
            df.to_csv(csv_path, index=False)

            norm_stats_path = None
            manifest_path = None

        logging.info(f"Saved dataset locally to: {dataset_path}")

        # Compute hashes for lineage
        if XCOM_CONTRACTS_AVAILABLE:
            dataset_hash = compute_file_hash(str(dataset_path))
        else:
            with open(dataset_path, 'rb') as f:
                dataset_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        if norm_stats_path and norm_stats_path.exists():
            with open(norm_stats_path, 'rb') as f:
                norm_stats_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    logging.info(f"Dataset hash: {dataset_hash}")
    logging.info(f"Feature order hash: {feature_order_hash_val}")

    # ========================================================================
    # 6. CREATE MANIFEST (for experiment mode, local storage)
    # ========================================================================
    if manifest_path and dataset_snapshot is None:
        manifest = {
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'dataset_hash': dataset_hash,
            'norm_stats_hash': norm_stats_hash,
            'feature_order_hash': feature_order_hash_val,
            'feature_columns': feature_columns,
            'date_range': {
                'start': train_start,
                'end': train_end,
            },
            'row_count': len(df),
            'column_count': len(df.columns),
            'canonical_builder_version': CanonicalFeatureBuilder.VERSION if CANONICAL_BUILDER_AVAILABLE else 'fallback',
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logging.info(f"Saved manifest to: {manifest_path}")

    # ========================================================================
    # 7. PUSH TO XCOM USING CONTRACTS - MinIO-First Aware
    # ========================================================================
    if XCOM_CONTRACTS_AVAILABLE:
        if dataset_snapshot is not None:
            # MinIO-First: Use S3 URIs from snapshot
            output = L2Output(
                # S3 URIs (MinIO-First)
                dataset_uri=dataset_snapshot.storage_uri,
                norm_stats_uri=dataset_snapshot.norm_stats_uri,
                manifest_uri=dataset_snapshot.manifest_uri,
                experiment_id=experiment_name,
                version=version,
                # Legacy fields (for backward compatibility)
                dataset_path=str(dataset_path) if dataset_path else None,
                dataset_hash=dataset_hash,
                date_range_start=train_start,
                date_range_end=train_end,
                feature_order_hash=feature_order_hash_val,
                feature_columns=feature_columns,
                row_count=len(df),
                experiment_name=experiment_name,
                norm_stats_path=str(norm_stats_path) if norm_stats_path else None,
                manifest_path=str(manifest_path) if manifest_path else None,
            )
            logging.info(f"[MinIO-First] Pushing S3 URIs to XCom")
        else:
            # Local storage: use file paths
            output = L2Output(
                dataset_path=str(dataset_path),
                dataset_hash=dataset_hash,
                date_range_start=train_start,
                date_range_end=train_end,
                feature_order_hash=feature_order_hash_val,
                feature_columns=feature_columns,
                row_count=len(df),
                experiment_name=experiment_name,
                norm_stats_path=str(norm_stats_path) if norm_stats_path else None,
                manifest_path=str(manifest_path) if manifest_path else None,
            )

        # Push using contract method
        output.push_to_xcom(ti)

        logging.info(f"Pushed to XCom: {L2XComKeysEnum.DATASET_PATH.value}")
        result = output.to_dict()
    else:
        # Fallback XCom push
        result = {
            'dataset_path': str(dataset_path) if dataset_path else None,
            'dataset_hash': dataset_hash,
            'date_range_start': train_start,
            'date_range_end': train_end,
            'feature_order_hash': feature_order_hash_val,
            'feature_columns': feature_columns,
            'row_count': len(df),
            'experiment_name': experiment_name,
        }
        # Add MinIO URIs if available
        if dataset_snapshot:
            result['dataset_uri'] = dataset_snapshot.storage_uri
            result['norm_stats_uri'] = dataset_snapshot.norm_stats_uri
            result['version'] = version
            result['experiment_id'] = experiment_name

        ti.xcom_push(key='dataset_path', value=str(dataset_path) if dataset_path else None)
        ti.xcom_push(key='dataset_hash', value=dataset_hash)
        ti.xcom_push(key='row_count', value=len(df))
        if dataset_snapshot:
            ti.xcom_push(key='dataset_uri', value=dataset_snapshot.storage_uri)
            ti.xcom_push(key='experiment_id', value=experiment_name)
            ti.xcom_push(key='version', value=version)

    logging.info(f"Build complete for experiment: {experiment_name or 'default'}")
    if dataset_snapshot:
        logging.info(f"[MinIO-First] S3 URI: {dataset_snapshot.storage_uri}")
    logging.info("=" * 60)

    return result


def validate_data_quality_gate(**context):
    """
    Data Quality Gate - Validates dataset before passing to L3.

    This is the critical validation step between L2 (preprocessing)
    and L3 (training). It ensures data quality meets training requirements.

    Contract: CTR-DQ-001
    """
    ti = context['ti']

    # Get dataset path from previous task
    dataset_path = ti.xcom_pull(key='dataset_path', task_ids='build_rl_dataset')

    if not dataset_path:
        # Fallback to main dataset
        dataset_path = str(OUTPUT_DIR / 'datasets_5min' / 'RL_DS3_MACRO_CORE.csv')

    logging.info("=" * 60)
    logging.info("DATA QUALITY GATE - L2 → L3 VALIDATION")
    logging.info("=" * 60)
    logging.info(f"Dataset: {dataset_path}")

    # Import the data quality gate
    try:
        from src.validation.data_quality_gate import (
            DataQualityGate,
            DataQualityError,
            validate_dataset_for_training,
        )
        QUALITY_GATE_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"[DQ] DataQualityGate not available: {e}")
        QUALITY_GATE_AVAILABLE = False

    if QUALITY_GATE_AVAILABLE and Path(dataset_path).exists():
        try:
            # Run comprehensive validation
            report = validate_dataset_for_training(
                dataset_path=dataset_path,
                strict=True,  # Raise exception on failure
            )

            # Push results to XCom
            ti.xcom_push(key='data_quality_report', value=report)
            ti.xcom_push(key='data_quality_passed', value=report.get('passed', False))

            logging.info(f"[DQ] Validation PASSED")
            logging.info(f"[DQ] Rows: {report.get('summary', {}).get('total_rows', 'N/A')}")
            logging.info(f"[DQ] NaN%: {report.get('summary', {}).get('nan_percentage', 'N/A'):.2f}%")

            return report

        except DataQualityError as e:
            logging.error(f"[DQ] Validation FAILED: {e}")
            ti.xcom_push(key='data_quality_passed', value=False)
            ti.xcom_push(key='data_quality_report', value=e.report.to_dict())
            raise  # Re-raise to fail the task

        except Exception as e:
            logging.error(f"[DQ] Unexpected error: {e}")
            ti.xcom_push(key='data_quality_passed', value=False)
            raise

    else:
        # Fallback: basic validation
        logging.warning("[DQ] Using fallback validation (DataQualityGate not available)")

        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)

        # Basic checks
        validations = {
            'row_count_ok': len(df) >= 100000,
            'nan_pct_ok': df.isna().sum().sum() / df.size < 0.05,
            'feature_count_ok': len(df.columns) >= 13,
        }

        passed = all(validations.values())

        report = {
            'passed': passed,
            'validations': validations,
            'summary': {
                'total_rows': len(df),
                'nan_percentage': df.isna().sum().sum() / df.size * 100,
            }
        }

        ti.xcom_push(key='data_quality_passed', value=passed)
        ti.xcom_push(key='data_quality_report', value=report)

        if not passed:
            raise ValueError(f"Data quality validation failed: {validations}")

        logging.info(f"[DQ] Fallback validation PASSED")
        return report


def validate_datasets(**context):
    """Validate generated datasets."""
    logging.info("Validating output datasets...")

    datasets = count_output_datasets()

    validation = {
        'valid': True,
        'issues': [],
        'datasets': datasets
    }

    # Check 5-min datasets
    if not datasets['5min']:
        validation['issues'].append("No 5-min datasets found")
        validation['valid'] = False
    else:
        logging.info(f"Found {len(datasets['5min'])} 5-min datasets")
        for ds in datasets['5min']:
            logging.info(f"  - {ds['name']}: {ds['size_mb']} MB")

    # Check daily datasets
    if not datasets['daily']:
        validation['issues'].append("No daily datasets found")
        validation['valid'] = False
    else:
        logging.info(f"Found {len(datasets['daily'])} daily datasets")

    # Check for main training dataset
    main_dataset = OUTPUT_DIR / 'datasets_5min' / 'RL_DS3_MACRO_CORE.csv'
    if main_dataset.exists():
        df = pd.read_csv(main_dataset)
        validation['main_dataset'] = {
            'name': 'RL_DS3_MACRO_CORE.csv',
            'rows': len(df),
            'columns': len(df.columns),
            'size_mb': round(main_dataset.stat().st_size / (1024 * 1024), 2)
        }
        logging.info(f"Main training dataset: {len(df):,} rows, {len(df.columns)} columns")

        # Validate minimum rows (at least 10K for training)
        if len(df) < 10000:
            validation['issues'].append(f"Main dataset has only {len(df)} rows (minimum 10000 expected)")
            validation['valid'] = False
    else:
        validation['issues'].append("Main training dataset (RL_DS3_MACRO_CORE.csv) not found")
        validation['valid'] = False

    if validation['valid']:
        logging.info("Dataset validation passed!")
    else:
        logging.warning(f"Dataset validation issues: {validation['issues']}")

    context['ti'].xcom_push(key='validation_result', value=validation)
    return validation


def pipeline_summary(**context):
    """Generate pipeline execution summary."""
    ti = context['ti']

    export_results = ti.xcom_pull(key='export_results', task_ids='export_db_to_sources') or {}
    validation = ti.xcom_pull(key='validation_result', task_ids='validate_datasets') or {}

    logging.info("=" * 70)
    logging.info("PREPROCESSING PIPELINE SUMMARY")
    logging.info("=" * 70)

    # Export stats
    logging.info(f"Data Exported:")
    logging.info(f"  - OHLCV: {export_results.get('ohlcv_exported', 0):,} rows")
    logging.info(f"  - Macro: {export_results.get('macro_exported', 0):,} rows")

    # Dataset stats
    if validation.get('main_dataset'):
        ds = validation['main_dataset']
        logging.info(f"\nMain Training Dataset:")
        logging.info(f"  - Name: {ds['name']}")
        logging.info(f"  - Rows: {ds['rows']:,}")
        logging.info(f"  - Columns: {ds['columns']}")
        logging.info(f"  - Size: {ds['size_mb']} MB")

    # Validation status
    if validation.get('valid'):
        logging.info(f"\nValidation: PASSED")
    else:
        logging.info(f"\nValidation: FAILED")
        for issue in validation.get('issues', []):
            logging.info(f"  - {issue}")

    logging.info("=" * 70)

    return {
        'status': 'success' if validation.get('valid', False) else 'failed',
        'export': export_results,
        'validation': validation
    }


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,  # P1 Remediation: Enable failure notifications
    'email': ['trading-alerts@example.com'],  # Configure in Airflow Variables
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='V3 L2: Run complete preprocessing pipeline to generate RL datasets',
    schedule_interval=None,  # Manual trigger
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l2', 'preprocessing', 'datasets', 'rl']
)

with dag:

    # Task 1: Export database to source files
    task_export = PythonOperator(
        task_id='export_db_to_sources',
        python_callable=export_db_to_sources,
        provide_context=True
    )

    # Task 2: Run fusion
    task_fusion = PythonOperator(
        task_id='run_fusion',
        python_callable=run_fusion,
        provide_context=True
    )

    # Task 3: Run cleaning
    task_cleaning = PythonOperator(
        task_id='run_cleaning',
        python_callable=run_cleaning,
        provide_context=True
    )

    # Task 4: Run resampling
    task_resampling = PythonOperator(
        task_id='run_resampling',
        python_callable=run_resampling,
        provide_context=True
    )

    # Task 5: Build RL datasets using SSOT (CanonicalFeatureBuilder)
    # This is the preferred method - uses SSOT contracts
    task_build_datasets_ssot = PythonOperator(
        task_id='build_rl_dataset',
        python_callable=build_rl_dataset_ssot,
        provide_context=True,
    )

    # Task 5b: Legacy RL dataset builder (kept for backward compatibility)
    # Use this only if SSOT method fails
    task_build_datasets_legacy = PythonOperator(
        task_id='run_rl_dataset_builder_legacy',
        python_callable=run_rl_dataset_builder,
        provide_context=True,
    )

    # Task 6: Data Quality Gate (L2 → L3 boundary validation)
    # This is the critical gate that ensures data quality before training
    task_quality_gate = PythonOperator(
        task_id='data_quality_gate',
        python_callable=validate_data_quality_gate,
        provide_context=True,
    )

    # Task 7: Validate datasets (file-level checks)
    task_validate = PythonOperator(
        task_id='validate_datasets',
        python_callable=validate_datasets,
        provide_context=True,
    )

    # Task 8: Pipeline summary
    task_summary = PythonOperator(
        task_id='pipeline_summary',
        python_callable=pipeline_summary,
        provide_context=True,
    )

    # Sequential pipeline execution with Data Quality Gate
    # Main path: export → fusion → cleaning → resampling → build → QUALITY GATE → validate → summary
    task_export >> task_fusion >> task_cleaning >> task_resampling >> task_build_datasets_ssot >> task_quality_gate >> task_validate >> task_summary

    # Legacy path is separate (can be triggered manually if needed)
    # task_resampling >> task_build_datasets_legacy >> task_quality_gate >> task_validate
