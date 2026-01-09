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
    5. Build RL datasets (06_rl_dataset_builder)
    6. Validate output datasets
    7. Optionally upload to MinIO

Schedule:
    Manual trigger or after data updates

Features:
    - Calls existing Python preprocessing scripts
    - Progress logging and metrics
    - Dataset validation
    - Optional MinIO upload

Author: Pipeline Automatizado
Version: 1.0.0
Created: 2025-12-26
"""

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import os
import logging
from io import StringIO

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import pandas as pd
import psycopg2

# =============================================================================
# CONFIGURATION
# =============================================================================

from utils.dag_common import get_db_connection

DAG_ID = 'v3.l2_preprocessing_pipeline'

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
    """Run the RL dataset builder step (06_rl_dataset_builder)."""
    logging.info("Building RL datasets...")

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
    'email_on_failure': False,
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

    # Task 5: Build RL datasets
    task_build_datasets = PythonOperator(
        task_id='run_rl_dataset_builder',
        python_callable=run_rl_dataset_builder,
        provide_context=True
    )

    # Task 6: Validate datasets
    task_validate = PythonOperator(
        task_id='validate_datasets',
        python_callable=validate_datasets,
        provide_context=True
    )

    # Task 7: Pipeline summary
    task_summary = PythonOperator(
        task_id='pipeline_summary',
        python_callable=pipeline_summary,
        provide_context=True
    )

    # Sequential pipeline execution
    task_export >> task_fusion >> task_cleaning >> task_resampling >> task_build_datasets >> task_validate >> task_summary
