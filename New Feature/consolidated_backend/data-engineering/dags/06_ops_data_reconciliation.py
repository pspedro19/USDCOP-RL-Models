"""
DAG: data_reconciliation_pipeline
=================================
Pipeline de Reconciliacion de Datos - USD/COP Forecasting

Ejecuta diariamente a las 9:00 AM (despues del scraping de 8 AM):
1. Cargar datos historicos (COMBINED_V2.csv)
2. Cargar datos recientes de PostgreSQL (core.features_ml)
3. Validar alineacion de columnas y valores
4. Merge si la validacion pasa (agregar filas nuevas)
5. Regenerar RL_COMBINED_ML_FEATURES_FIXED.csv
6. Backup de versiones anteriores
7. Notificar completado o alertar si hay desalineacion

Schedule: 0 9 * * * (9:00 AM diario, hora local)

Alertas:
- Si hay desalineacion de columnas
- Si hay diferencias significativas en valores de overlap
- Si el merge falla
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from airflow.exceptions import AirflowException
import pandas as pd
import numpy as np
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Agregar path del backend
sys.path.insert(0, '/opt/airflow/backend')

DAG_ID = "06_ops_data_reconciliation"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - configurable via environment variables
DATA_DIR = Path(os.getenv('DATA_DIR', '/opt/airflow/data'))
HISTORICAL_CSV = DATA_DIR / 'raw' / 'COMBINED_V2.csv'
ML_FEATURES_OUTPUT = DATA_DIR / 'processed' / 'RL_COMBINED_ML_FEATURES_FIXED.csv'
BACKUP_DIR = DATA_DIR / 'backups'

# Reconciliation settings
OVERLAP_DAYS = 30  # Days to check for overlap
MAX_PRICE_DIFF_THRESHOLD = 5.0  # Max acceptable price difference in COP
MAX_VALUE_MISMATCH_PCT = 0.10  # Max 10% of overlap rows can have mismatches

# Horizons for target regeneration
HORIZONS = [1, 5, 10, 15, 20, 25, 30]

# =============================================================================
# DEFAULT ARGS
# =============================================================================

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def load_historical_csv(**context) -> Dict[str, Any]:
    """
    Task 1: Cargar datos historicos desde COMBINED_V2.csv
    """
    logging.info("=" * 60)
    logging.info("TASK: Loading historical CSV data")
    logging.info("=" * 60)

    from src.data.reconciler import DataReconciler

    reconciler = DataReconciler()

    # Check if file exists
    historical_path = Path(HISTORICAL_CSV)
    if not historical_path.exists():
        # Try alternative paths
        alt_paths = [
            DATA_DIR / 'COMBINED_V2.csv',
            DATA_DIR / 'raw' / 'RL_COMBINED_V2.csv',
            Path('/opt/airflow/data/COMBINED_V2.csv'),
        ]
        for alt in alt_paths:
            if alt.exists():
                historical_path = alt
                break
        else:
            raise AirflowException(f"Historical CSV not found. Tried: {[str(HISTORICAL_CSV)] + [str(p) for p in alt_paths]}")

    # Load historical data
    try:
        df_historical = reconciler.load_historical_csv(historical_path)
    except Exception as e:
        raise AirflowException(f"Failed to load historical CSV: {e}")

    # Store metadata
    metadata = {
        'path': str(historical_path),
        'n_rows': len(df_historical),
        'n_cols': len(df_historical.columns),
        'date_min': str(df_historical.index.min().date()),
        'date_max': str(df_historical.index.max().date()),
        'columns': list(df_historical.columns[:20]),  # First 20 columns for XCom limit
    }

    # Save to temp file for next task
    temp_path = '/tmp/historical_data.pkl'
    df_historical.to_pickle(temp_path)

    context['ti'].xcom_push(key='historical_path', value=str(historical_path))
    context['ti'].xcom_push(key='historical_temp', value=temp_path)
    context['ti'].xcom_push(key='historical_metadata', value=metadata)

    logging.info(f"Loaded historical data: {metadata['n_rows']} rows")
    logging.info(f"Date range: {metadata['date_min']} to {metadata['date_max']}")

    return metadata


def load_recent_from_db(**context) -> Dict[str, Any]:
    """
    Task 2: Cargar datos recientes desde PostgreSQL (core.features_ml)
    """
    logging.info("=" * 60)
    logging.info("TASK: Loading recent data from database")
    logging.info("=" * 60)

    from src.data.reconciler import DataReconciler
    from utils.dag_common import get_db_connection

    reconciler = DataReconciler()

    try:
        conn = get_db_connection()
        df_recent = reconciler.load_recent_from_db(
            connection=conn,
            days=OVERLAP_DAYS,
            table='core.features_ml'
        )
        conn.close()
    except Exception as e:
        logging.warning(f"Database connection failed: {e}")
        # Try to continue with empty recent data
        df_recent = pd.DataFrame()

    if df_recent.empty:
        logging.warning("No recent data loaded from database")
        metadata = {
            'n_rows': 0,
            'n_cols': 0,
            'date_min': None,
            'date_max': None,
            'columns': [],
            'is_empty': True
        }
        context['ti'].xcom_push(key='recent_metadata', value=metadata)
        context['ti'].xcom_push(key='recent_is_empty', value=True)
        return metadata

    # Store metadata
    metadata = {
        'n_rows': len(df_recent),
        'n_cols': len(df_recent.columns),
        'date_min': str(df_recent.index.min().date()),
        'date_max': str(df_recent.index.max().date()),
        'columns': list(df_recent.columns[:20]),
        'is_empty': False
    }

    # Save to temp file
    temp_path = '/tmp/recent_data.pkl'
    df_recent.to_pickle(temp_path)

    context['ti'].xcom_push(key='recent_temp', value=temp_path)
    context['ti'].xcom_push(key='recent_metadata', value=metadata)
    context['ti'].xcom_push(key='recent_is_empty', value=False)

    logging.info(f"Loaded recent data: {metadata['n_rows']} rows")
    logging.info(f"Date range: {metadata['date_min']} to {metadata['date_max']}")

    return metadata


def validate_alignment(**context) -> str:
    """
    Task 3: Validar alineacion de columnas y valores en overlap.

    Returns branch to take:
    - 'merge_if_valid' if validation passes
    - 'alert_misalignment' if validation fails
    """
    logging.info("=" * 60)
    logging.info("TASK: Validating data alignment")
    logging.info("=" * 60)

    from src.data.reconciler import DataReconciler
    from src.data.alignment_validator import AlignmentValidator

    # Check if recent data is empty
    recent_is_empty = context['ti'].xcom_pull(key='recent_is_empty', task_ids='load_recent_from_db')
    if recent_is_empty:
        logging.info("Recent data is empty - skipping validation, proceeding with historical only")
        context['ti'].xcom_push(key='validation_passed', value=True)
        context['ti'].xcom_push(key='skip_merge', value=True)
        return 'merge_if_valid'

    # Load data from temp files
    historical_temp = context['ti'].xcom_pull(key='historical_temp', task_ids='load_historical_csv')
    recent_temp = context['ti'].xcom_pull(key='recent_temp', task_ids='load_recent_from_db')

    df_historical = pd.read_pickle(historical_temp)
    df_recent = pd.read_pickle(recent_temp)

    reconciler = DataReconciler()
    validator = AlignmentValidator()

    # Find overlap dates
    overlap_dates = reconciler.find_overlap_dates(df_historical, df_recent)
    logging.info(f"Found {len(overlap_dates)} overlapping dates")

    if not overlap_dates:
        logging.info("No overlapping dates - new data only, proceeding to merge")
        context['ti'].xcom_push(key='validation_passed', value=True)
        context['ti'].xcom_push(key='overlap_dates', value=[])
        return 'merge_if_valid'

    # Validate overlap values
    validation_report = reconciler.validate_overlap_values(
        df_historical,
        df_recent,
        overlap_dates
    )

    # Also run alignment validator
    column_report = validator.validate_columns_match(df_historical, df_recent)
    value_report = validator.validate_values_in_overlap(
        df_historical,
        df_recent,
        'date'  # Will use index
    )

    # Log validation summary
    logging.info("\n" + validation_report.summary())

    # Store validation results
    validation_data = {
        'n_overlap_dates': len(overlap_dates),
        'columns_matched': validation_report.columns_matched,
        'value_mismatches': validation_report.value_mismatches,
        'avg_price_diff': validation_report.avg_price_difference,
        'max_price_diff': validation_report.max_price_difference,
        'n_issues': validation_report.n_issues,
        'validation_passed': validation_report.validation_passed,
    }

    context['ti'].xcom_push(key='validation_data', value=validation_data)
    context['ti'].xcom_push(key='overlap_dates', value=[str(d) for d in overlap_dates])

    # Determine if we should proceed or alert
    if validation_report.validation_passed:
        logging.info("Validation PASSED - proceeding to merge")
        context['ti'].xcom_push(key='validation_passed', value=True)
        return 'merge_if_valid'
    else:
        logging.warning(f"Validation FAILED - {validation_report.error_message}")
        context['ti'].xcom_push(key='validation_passed', value=False)
        context['ti'].xcom_push(key='validation_error', value=validation_report.error_message)
        return 'alert_misalignment'


def merge_if_valid(**context) -> Dict[str, Any]:
    """
    Task 4: Merge datasets si la validacion paso.
    Agrega filas nuevas de recent a historical.
    """
    logging.info("=" * 60)
    logging.info("TASK: Merging datasets")
    logging.info("=" * 60)

    from src.data.reconciler import DataReconciler

    # Check if we should skip merge
    skip_merge = context['ti'].xcom_pull(key='skip_merge', task_ids='validate_alignment')
    if skip_merge:
        logging.info("Skipping merge - using historical data only")
        historical_temp = context['ti'].xcom_pull(key='historical_temp', task_ids='load_historical_csv')
        df_merged = pd.read_pickle(historical_temp)
    else:
        # Load data from temp files
        historical_temp = context['ti'].xcom_pull(key='historical_temp', task_ids='load_historical_csv')
        recent_temp = context['ti'].xcom_pull(key='recent_temp', task_ids='load_recent_from_db')

        df_historical = pd.read_pickle(historical_temp)

        try:
            df_recent = pd.read_pickle(recent_temp)
        except:
            df_recent = pd.DataFrame()

        reconciler = DataReconciler()

        # Merge datasets (prefer recent values in overlap)
        df_merged = reconciler.merge_datasets(
            df_historical,
            df_recent,
            prefer_recent=True,
            fill_missing=True
        )

    # Calculate new rows added
    historical_metadata = context['ti'].xcom_pull(key='historical_metadata', task_ids='load_historical_csv')
    original_rows = historical_metadata['n_rows']
    new_rows = len(df_merged) - original_rows

    logging.info(f"Merged dataset: {len(df_merged)} rows ({new_rows} new)")

    # Save merged data
    merged_temp = '/tmp/merged_data.pkl'
    df_merged.to_pickle(merged_temp)

    context['ti'].xcom_push(key='merged_temp', value=merged_temp)
    context['ti'].xcom_push(key='merged_rows', value=len(df_merged))
    context['ti'].xcom_push(key='new_rows', value=new_rows)

    return {
        'total_rows': len(df_merged),
        'new_rows': new_rows,
        'date_min': str(df_merged.index.min().date()),
        'date_max': str(df_merged.index.max().date()),
    }


def update_ml_features(**context) -> Dict[str, Any]:
    """
    Task 5: Regenerar RL_COMBINED_ML_FEATURES_FIXED.csv con targets actualizados.
    """
    logging.info("=" * 60)
    logging.info("TASK: Updating ML features file")
    logging.info("=" * 60)

    from src.data.reconciler import DataReconciler

    reconciler = DataReconciler()

    # Load merged data
    merged_temp = context['ti'].xcom_pull(key='merged_temp', task_ids='merge_if_valid')
    df_merged = pd.read_pickle(merged_temp)

    # Ensure output directory exists
    output_path = Path(ML_FEATURES_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Regenerate ML features (forward returns as targets)
    df_with_targets = reconciler.regenerate_ml_features(
        df_merged,
        output_path,
        horizons=HORIZONS
    )

    logging.info(f"Saved ML features to: {output_path}")

    # Verify targets were created
    target_cols = [c for c in df_with_targets.columns if c.startswith('target_')]
    logging.info(f"Target columns: {target_cols}")

    context['ti'].xcom_push(key='ml_features_path', value=str(output_path))
    context['ti'].xcom_push(key='target_columns', value=target_cols)

    return {
        'output_path': str(output_path),
        'n_rows': len(df_with_targets),
        'target_columns': target_cols,
    }


def backup_old_versions(**context) -> Dict[str, Any]:
    """
    Task 6: Guardar backup de versiones anteriores.
    """
    logging.info("=" * 60)
    logging.info("TASK: Backing up old versions")
    logging.info("=" * 60)

    from src.data.reconciler import DataReconciler

    reconciler = DataReconciler()

    # Create backup directory
    backup_dir = Path(BACKUP_DIR)
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Get original historical path
    historical_path = context['ti'].xcom_pull(key='historical_path', task_ids='load_historical_csv')
    historical_path = Path(historical_path)

    # Save reconciled version as new COMBINED
    merged_temp = context['ti'].xcom_pull(key='merged_temp', task_ids='merge_if_valid')
    df_merged = pd.read_pickle(merged_temp)

    # Backup and save to original historical location
    saved_path = reconciler.save_reconciled(
        df_merged,
        historical_path,
        backup=True,
        backup_dir=backup_dir
    )

    # List recent backups
    backup_files = sorted(backup_dir.glob('COMBINED*.csv'), reverse=True)[:5]

    logging.info(f"Saved reconciled data to: {saved_path}")
    logging.info(f"Recent backups: {[f.name for f in backup_files]}")

    context['ti'].xcom_push(key='saved_path', value=saved_path)
    context['ti'].xcom_push(key='backup_count', value=len(list(backup_dir.glob('*.csv'))))

    return {
        'saved_path': saved_path,
        'backup_dir': str(backup_dir),
        'recent_backups': [f.name for f in backup_files],
    }


def notify_completion(**context) -> Dict[str, Any]:
    """
    Task 7: Notificar completado exitoso.
    """
    logging.info("=" * 60)
    logging.info("DATA RECONCILIATION PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("=" * 60)

    # Gather all results
    historical_meta = context['ti'].xcom_pull(key='historical_metadata', task_ids='load_historical_csv')
    merged_rows = context['ti'].xcom_pull(key='merged_rows', task_ids='merge_if_valid')
    new_rows = context['ti'].xcom_pull(key='new_rows', task_ids='merge_if_valid')
    ml_features_path = context['ti'].xcom_pull(key='ml_features_path', task_ids='update_ml_features')
    validation_data = context['ti'].xcom_pull(key='validation_data', task_ids='validate_alignment') or {}

    summary = {
        'status': 'SUCCESS',
        'execution_date': datetime.now().isoformat(),
        'original_rows': historical_meta['n_rows'],
        'final_rows': merged_rows,
        'new_rows_added': new_rows,
        'validation_passed': validation_data.get('validation_passed', True),
        'ml_features_updated': ml_features_path,
    }

    logging.info("Summary:")
    for key, value in summary.items():
        logging.info(f"  {key}: {value}")

    # Here you could add notifications:
    # - Send Slack message
    # - Send email
    # - Update status in database

    return summary


def alert_misalignment(**context) -> None:
    """
    Task (branch): Alertar si hay desalineacion en los datos.
    """
    logging.error("=" * 60)
    logging.error("ALERT: DATA MISALIGNMENT DETECTED")
    logging.error("=" * 60)

    validation_error = context['ti'].xcom_pull(key='validation_error', task_ids='validate_alignment')
    validation_data = context['ti'].xcom_pull(key='validation_data', task_ids='validate_alignment') or {}

    logging.error(f"Validation failed: {validation_error}")
    logging.error(f"Mismatches: {validation_data.get('value_mismatches', 'N/A')}")
    logging.error(f"Max price diff: ${validation_data.get('max_price_diff', 'N/A'):.2f}")

    # Here you could add alerts:
    # - Send urgent Slack notification
    # - Send PagerDuty alert
    # - Create Jira ticket

    # Raise exception to mark task as failed
    raise AirflowException(
        f"Data reconciliation failed due to misalignment: {validation_error}. "
        f"Manual review required before proceeding."
    )


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Pipeline de reconciliacion de datos historicos y recientes',
    schedule_interval='0 9 * * *',  # 9:00 AM diario
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['reconciliation', 'data-engineering', 'daily', 'production'],
    max_active_runs=1,
    doc_md=__doc__,
) as dag:

    # Task 1: Load historical CSV
    task_load_historical = PythonOperator(
        task_id='load_historical_csv',
        python_callable=load_historical_csv,
        provide_context=True,
    )

    # Task 2: Load recent from DB
    task_load_recent = PythonOperator(
        task_id='load_recent_from_db',
        python_callable=load_recent_from_db,
        provide_context=True,
    )

    # Task 3: Validate alignment (branching)
    task_validate = BranchPythonOperator(
        task_id='validate_alignment',
        python_callable=validate_alignment,
        provide_context=True,
    )

    # Task 4: Merge if valid
    task_merge = PythonOperator(
        task_id='merge_if_valid',
        python_callable=merge_if_valid,
        provide_context=True,
    )

    # Task 5: Update ML features
    task_update_features = PythonOperator(
        task_id='update_ml_features',
        python_callable=update_ml_features,
        provide_context=True,
    )

    # Task 6: Backup old versions
    task_backup = PythonOperator(
        task_id='backup_old_versions',
        python_callable=backup_old_versions,
        provide_context=True,
    )

    # Task 7: Notify completion
    task_notify = PythonOperator(
        task_id='notify_completion',
        python_callable=notify_completion,
        provide_context=True,
    )

    # Alert branch: Misalignment detected
    task_alert = PythonOperator(
        task_id='alert_misalignment',
        python_callable=alert_misalignment,
        provide_context=True,
    )

    # Join point after branch
    task_join = EmptyOperator(
        task_id='join',
        trigger_rule='none_failed_min_one_success',
    )

    # =============================================================================
    # DEPENDENCIES
    # =============================================================================
    #
    # Flow:
    #   load_historical_csv ----\
    #                            --> validate_alignment --+--> merge_if_valid --> update_ml_features --> backup --> notify
    #   load_recent_from_db ----/                         |
    #                                                     +--> alert_misalignment
    #

    [task_load_historical, task_load_recent] >> task_validate

    task_validate >> task_merge >> task_update_features >> task_backup >> task_notify >> task_join
    task_validate >> task_alert >> task_join
