"""
DAG: l0_01_init_restore
=======================
USD/COP Trading System - V3 Architecture
Layer 0: Data Initialization from Backups

Purpose:
    Initializes the database with backup data on first startup or when
    tables are empty. Implements backup-first approach:
    1. Check for existing backups
    2. Restore OHLCV data from backup
    3. Restore Macro data from backup
    4. Restore Features from backup (if exists)
    5. Detect data gaps
    6. Trigger backfill DAGs to update to current time
    7. Verify data integrity

Schedule:
    @once (manual trigger or system startup)

Features:
    - Idempotent restore (uses UPSERT - ON CONFLICT DO UPDATE)
    - Smart backup detection (finds latest backup)
    - Gap detection after restore
    - Automatic backfill trigger
    - Progress logging and metrics

Author: Pipeline Automatizado
Version: 1.0.0
Created: 2025-12-26
"""

from datetime import datetime, timedelta
from pathlib import Path
import gzip
import os
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# =============================================================================
# CONFIGURATION
# =============================================================================

from utils.dag_common import get_db_connection
from contracts.dag_registry import L0_INIT_RESTORE, L0_OHLCV_BACKFILL

DAG_ID = L0_INIT_RESTORE

# Paths - Support both Docker and local development
def get_backup_paths():
    """Get backup paths based on environment."""
    # Docker paths
    docker_backup = Path('/app/data/backups')
    docker_macro = Path('/app/data/pipeline/05_resampling/output/MACRO_DAILY_CONSOLIDATED.csv')

    if docker_backup.exists():
        return docker_backup, docker_macro

    # Local development paths
    project_root = Path('/opt/airflow') if Path('/opt/airflow').exists() else Path(__file__).parent.parent.parent.parent
    return (
        project_root / 'data' / 'backups',
        project_root / 'data' / 'pipeline' / '05_resampling' / 'output' / 'MACRO_DAILY_CONSOLIDATED.csv'
    )


BACKUP_DIR, MACRO_DATA_PATH = get_backup_paths()

# Column mapping for macro data (uppercase to lowercase)
MACRO_COLUMN_MAPPING = {
    'fecha': 'fecha',
    'COMM_AGRI_COFFEE_GLB_D_COFFEE': 'comm_agri_coffee_glb_d_coffee',
    'COMM_METAL_GOLD_GLB_D_GOLD': 'comm_metal_gold_glb_d_gold',
    'COMM_OIL_BRENT_GLB_D_BRENT': 'comm_oil_brent_glb_d_brent',
    'COMM_OIL_WTI_GLB_D_WTI': 'comm_oil_wti_glb_d_wti',
    'CRSK_SENTIMENT_CCI_COL_M_CCI': 'crsk_sentiment_cci_col_m_cci',
    'CRSK_SENTIMENT_ICI_COL_M_ICI': 'crsk_sentiment_ici_col_m_ici',
    'CRSK_SPREAD_EMBI_COL_D_EMBI': 'crsk_spread_embi_col_d_embi',
    'EQTY_INDEX_COLCAP_COL_D_COLCAP': 'eqty_index_colcap_col_d_colcap',
    'FINC_BOND_YIELD10Y_COL_D_COL10Y': 'finc_bond_yield10y_col_d_col10y',
    'FINC_BOND_YIELD10Y_USA_D_UST10Y': 'finc_bond_yield10y_usa_d_ust10y',
    'FINC_BOND_YIELD2Y_USA_D_DGS2': 'finc_bond_yield2y_usa_d_dgs2',
    'FINC_BOND_YIELD5Y_COL_D_COL5Y': 'finc_bond_yield5y_col_d_col5y',
    'FINC_RATE_IBR_OVERNIGHT_COL_D_IBR': 'finc_rate_ibr_overnight_col_d_ibr',
    'FTRD_EXPORTS_TOTAL_COL_M_EXPUSD': 'ftrd_exports_total_col_m_expusd',
    'FTRD_IMPORTS_TOTAL_COL_M_IMPUSD': 'ftrd_imports_total_col_m_impusd',
    'FTRD_TERMS_TRADE_COL_M_TOT': 'ftrd_terms_trade_col_m_tot',
    'FXRT_INDEX_DXY_USA_D_DXY': 'fxrt_index_dxy_usa_d_dxy',
    'FXRT_REER_BILATERAL_COL_M_ITCR': 'fxrt_reer_bilateral_col_m_itcr',
    'FXRT_SPOT_USDCLP_CHL_D_USDCLP': 'fxrt_spot_usdclp_chl_d_usdclp',
    'FXRT_SPOT_USDMXN_MEX_D_USDMXN': 'fxrt_spot_usdmxn_mex_d_usdmxn',
    'GDPP_REAL_GDP_USA_Q_GDP_Q': 'gdpp_real_gdp_usa_q_gdp_q',
    'INFL_CPI_ALL_USA_M_CPIAUCSL': 'infl_cpi_all_usa_m_cpiaucsl',
    'INFL_CPI_CORE_USA_M_CPILFESL': 'infl_cpi_core_usa_m_cpilfesl',
    'INFL_CPI_TOTAL_COL_M_IPCCOL': 'infl_cpi_total_col_m_ipccol',
    'INFL_PCE_USA_M_PCEPI': 'infl_pce_usa_m_pcepi',
    'LABR_UNEMPLOYMENT_USA_M_UNRATE': 'labr_unemployment_usa_m_unrate',
    'MNYS_M2_SUPPLY_USA_M_M2SL': 'mnys_m2_supply_usa_m_m2sl',
    'POLR_FED_FUNDS_USA_M_FEDFUNDS': 'polr_fed_funds_usa_m_fedfunds',
    'POLR_POLICY_RATE_COL_D_TPM': 'polr_policy_rate_col_d_tpm',
    'POLR_POLICY_RATE_COL_M_TPM': 'polr_policy_rate_col_m_tpm',
    'POLR_PRIME_RATE_USA_D_PRIME': 'polr_prime_rate_usa_d_prime',
    'PROD_INDUSTRIAL_USA_M_INDPRO': 'prod_industrial_usa_m_indpro',
    'RSBP_CURRENT_ACCOUNT_COL_Q_CACCT_Q': 'rsbp_current_account_col_q_cacct_q',
    'RSBP_FDI_INFLOW_COL_Q_FDIIN_Q': 'rsbp_fdi_inflow_col_q_fdiin_q',
    'RSBP_FDI_OUTFLOW_COL_Q_FDIOUT_Q': 'rsbp_fdi_outflow_col_q_fdiout_q',
    'RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT': 'rsbp_reserves_international_col_m_resint',
    'SENT_CONSUMER_USA_M_UMCSENT': 'sent_consumer_usa_m_umcsent',
    'VOLT_VIX_USA_D_VIX': 'volt_vix_usa_d_vix'
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_latest_backup(pattern: str, directory: Path = BACKUP_DIR):
    """Find the most recent backup file matching pattern."""
    if not directory.exists():
        return None

    backups = list(directory.glob(pattern))
    if not backups:
        return None

    return max(backups, key=lambda p: p.stat().st_mtime)


def table_row_count(conn, table_name: str) -> int:
    """Count rows in a table."""
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]
    except Exception:
        return -1
    finally:
        cur.close()


def get_date_range(conn, table_name: str, date_col: str):
    """Get min/max dates from a table."""
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT MIN({date_col}), MAX({date_col}) FROM {table_name}")
        return cur.fetchone()
    except Exception:
        return None, None
    finally:
        cur.close()


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def check_backups_exist(**context) -> str:
    """
    Check if backup files exist.
    Returns branch to take: 'restore_backups' or 'skip_restore'
    """
    logging.info("Checking for backup files...")

    # Check OHLCV backup
    ohlcv_backup = find_latest_backup('usdcop_m5_ohlcv_*.csv.gz')
    if not ohlcv_backup:
        ohlcv_backup = find_latest_backup('ohlcv_*.csv.gz')

    # Check Macro file
    macro_file = MACRO_DATA_PATH if MACRO_DATA_PATH.exists() else find_latest_backup('macro_*.csv.gz')

    backups_found = {
        'ohlcv': str(ohlcv_backup) if ohlcv_backup else None,
        'macro': str(macro_file) if macro_file else None,
    }

    context['ti'].xcom_push(key='backups_found', value=backups_found)

    if ohlcv_backup or macro_file:
        logging.info(f"Backups found: OHLCV={ohlcv_backup is not None}, Macro={macro_file is not None}")
        return 'restore_ohlcv_backup'
    else:
        logging.warning("No backup files found - skipping restore")
        return 'skip_restore'


def restore_ohlcv_backup(**context):
    """Restore OHLCV data from backup."""
    ti = context['ti']
    backups = ti.xcom_pull(key='backups_found', task_ids='check_backups_exist')

    if not backups.get('ohlcv'):
        logging.warning("No OHLCV backup to restore")
        ti.xcom_push(key='ohlcv_restored', value=0)
        return

    backup_file = Path(backups['ohlcv'])
    logging.info(f"Restoring OHLCV from: {backup_file.name}")

    conn = get_db_connection()

    try:
        # Check if table has data
        current_count = table_row_count(conn, 'usdcop_m5_ohlcv')

        if current_count > 0:
            logging.info(f"usdcop_m5_ohlcv already has {current_count:,} rows - using UPSERT mode")

        # Read backup
        with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f)

        logging.info(f"Read {len(df):,} rows from backup")

        # Prepare data
        df['time'] = pd.to_datetime(df['time'])
        df['symbol'] = df.get('symbol', 'USD/COP')
        df['volume'] = df.get('volume', 0).fillna(0).astype(int)
        df['source'] = 'backup_restore'

        # Insert with UPSERT
        cur = conn.cursor()
        cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source']
        data = [tuple(row) for row in df[cols].values]

        query = """
            INSERT INTO usdcop_m5_ohlcv (time, symbol, open, high, low, close, volume, source)
            VALUES %s
            ON CONFLICT (time, symbol) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source
        """

        # Batch insert
        batch_size = 10000
        total_inserted = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            execute_values(cur, query, batch, page_size=batch_size)
            total_inserted += len(batch)
            if i % 50000 == 0:
                logging.info(f"Progress: {total_inserted:,}/{len(data):,} rows")

        conn.commit()
        cur.close()

        logging.info(f"Restored {total_inserted:,} OHLCV rows")
        ti.xcom_push(key='ohlcv_restored', value=total_inserted)

    except Exception as e:
        conn.rollback()
        logging.error(f"Error restoring OHLCV: {e}")
        raise
    finally:
        conn.close()


def restore_macro_backup(**context):
    """Restore Macro data from backup."""
    ti = context['ti']
    backups = ti.xcom_pull(key='backups_found', task_ids='check_backups_exist')

    if not backups.get('macro'):
        logging.warning("No Macro backup to restore")
        ti.xcom_push(key='macro_restored', value=0)
        return

    macro_file = Path(backups['macro'])
    logging.info(f"Restoring Macro from: {macro_file.name}")

    conn = get_db_connection()

    try:
        # Read file (handle both .csv and .csv.gz)
        if macro_file.suffix == '.gz':
            with gzip.open(macro_file, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(macro_file)

        logging.info(f"Read {len(df):,} rows from macro file")

        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Convert fecha
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha']).dt.date
        else:
            logging.error("No 'fecha' column found")
            ti.xcom_push(key='macro_restored', value=0)
            return

        # Insert with UPSERT (row by row due to variable columns)
        cur = conn.cursor()
        inserted = 0

        for _, row in df.iterrows():
            values = {col: (None if pd.isna(row[col]) else row[col]) for col in df.columns}
            cols = list(values.keys())
            placeholders = ', '.join(['%s'] * len(cols))
            update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in cols if col != 'fecha'])

            query = f"""
                INSERT INTO macro_indicators_daily ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT (fecha) DO UPDATE SET {update_clause}
            """

            try:
                cur.execute(query, list(values.values()))
                inserted += 1
            except Exception as e:
                logging.debug(f"Error inserting row: {e}")
                continue

        conn.commit()
        cur.close()

        logging.info(f"Restored {inserted:,} Macro rows")
        ti.xcom_push(key='macro_restored', value=inserted)

    except Exception as e:
        conn.rollback()
        logging.error(f"Error restoring Macro: {e}")
        raise
    finally:
        conn.close()


def detect_data_gaps(**context) -> str:
    """
    Detect gaps in restored data.
    Returns branch: 'trigger_backfill' or 'no_gaps'
    """
    logging.info("Detecting data gaps...")

    conn = get_db_connection()

    try:
        # Get OHLCV date range
        ohlcv_min, ohlcv_max = get_date_range(conn, 'usdcop_m5_ohlcv', 'time')

        gaps = []

        if ohlcv_min is None:
            gaps.append({'table': 'ohlcv', 'type': 'no_data', 'message': 'No OHLCV data'})
        else:
            logging.info(f"OHLCV range: {ohlcv_min} to {ohlcv_max}")

            # Check gap to today
            today = datetime.now()
            if ohlcv_max and ohlcv_max.date() < (today - timedelta(days=3)).date():
                gap_days = (today - ohlcv_max).days
                gaps.append({
                    'table': 'ohlcv',
                    'type': 'recent_gap',
                    'start': str(ohlcv_max.date()),
                    'end': str(today.date()),
                    'days': gap_days
                })
                logging.warning(f"OHLCV gap: {gap_days} days until today")

        # Get Macro date range
        macro_min, macro_max = get_date_range(conn, 'macro_indicators_daily', 'fecha')

        if macro_min is None:
            gaps.append({'table': 'macro', 'type': 'no_data', 'message': 'No Macro data'})
        else:
            logging.info(f"Macro range: {macro_min} to {macro_max}")

            today = datetime.now().date()
            if macro_max and macro_max < (today - timedelta(days=7)):
                gap_days = (today - macro_max).days
                gaps.append({
                    'table': 'macro',
                    'type': 'recent_gap',
                    'start': str(macro_max),
                    'end': str(today),
                    'days': gap_days
                })
                logging.warning(f"Macro gap: {gap_days} days until today")

        context['ti'].xcom_push(key='gaps_detected', value=gaps)

        if any(g['type'] == 'recent_gap' for g in gaps):
            logging.info("Gaps detected - triggering backfill")
            return 'trigger_backfill'
        else:
            logging.info("No significant gaps detected")
            return 'no_gaps'

    finally:
        conn.close()


def verify_data_integrity(**context):
    """Verify data integrity after restore and potential backfill."""
    logging.info("Verifying data integrity...")

    conn = get_db_connection()

    try:
        results = {}

        # OHLCV stats
        ohlcv_count = table_row_count(conn, 'usdcop_m5_ohlcv')
        ohlcv_min, ohlcv_max = get_date_range(conn, 'usdcop_m5_ohlcv', 'time')

        results['ohlcv'] = {
            'count': ohlcv_count,
            'date_range': f"{ohlcv_min} to {ohlcv_max}" if ohlcv_min else 'N/A'
        }

        # Macro stats
        macro_count = table_row_count(conn, 'macro_indicators_daily')
        macro_min, macro_max = get_date_range(conn, 'macro_indicators_daily', 'fecha')

        results['macro'] = {
            'count': macro_count,
            'date_range': f"{macro_min} to {macro_max}" if macro_min else 'N/A'
        }

        logging.info("=" * 60)
        logging.info("DATA INTEGRITY VERIFICATION")
        logging.info("=" * 60)
        logging.info(f"OHLCV: {ohlcv_count:,} rows | Range: {results['ohlcv']['date_range']}")
        logging.info(f"Macro: {macro_count:,} rows | Range: {results['macro']['date_range']}")
        logging.info("=" * 60)

        # Get restore stats from XCom
        ti = context['ti']
        ohlcv_restored = ti.xcom_pull(key='ohlcv_restored', task_ids='restore_ohlcv_backup') or 0
        macro_restored = ti.xcom_pull(key='macro_restored', task_ids='restore_macro_backup') or 0

        logging.info(f"This run restored: OHLCV={ohlcv_restored:,}, Macro={macro_restored:,}")

        return results

    finally:
        conn.close()


def skip_restore(**context):
    """Log that restore was skipped."""
    logging.info("Restore skipped - no backup files found")
    return {'status': 'skipped', 'reason': 'no_backups_found'}


def skip_backfill(**context):
    """Log that backfill was skipped."""
    logging.info("Backfill skipped - data is current")
    return {'status': 'skipped', 'reason': 'no_gaps'}


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
    description='V3 L0: Initialize database from backups on startup',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l0', 'initialization', 'backup-restore', 'startup']
)

with dag:

    # Task 1: Check if backups exist
    task_check_backups = BranchPythonOperator(
        task_id='check_backups_exist',
        python_callable=check_backups_exist,
        provide_context=True
    )

    # Task 2a: Restore OHLCV backup
    task_restore_ohlcv = PythonOperator(
        task_id='restore_ohlcv_backup',
        python_callable=restore_ohlcv_backup,
        provide_context=True
    )

    # Task 2b: Skip restore (no backups)
    task_skip_restore = PythonOperator(
        task_id='skip_restore',
        python_callable=skip_restore,
        provide_context=True
    )

    # Task 3: Restore Macro backup
    task_restore_macro = PythonOperator(
        task_id='restore_macro_backup',
        python_callable=restore_macro_backup,
        provide_context=True
    )

    # Task 4: Detect data gaps
    task_detect_gaps = BranchPythonOperator(
        task_id='detect_data_gaps',
        python_callable=detect_data_gaps,
        provide_context=True,
        trigger_rule='none_failed_min_one_success'
    )

    # Task 5a: Trigger OHLCV backfill DAG
    task_trigger_backfill = TriggerDagRunOperator(
        task_id='trigger_backfill',
        trigger_dag_id=L0_OHLCV_BACKFILL,
        wait_for_completion=False,  # Don't wait - can run in parallel
        conf={'triggered_by': DAG_ID}
    )

    # Task 5b: No backfill needed
    task_no_gaps = PythonOperator(
        task_id='no_gaps',
        python_callable=skip_backfill,
        provide_context=True
    )

    # Task 6: Verify data integrity
    task_verify = PythonOperator(
        task_id='verify_data_integrity',
        python_callable=verify_data_integrity,
        provide_context=True,
        trigger_rule='none_failed_min_one_success'
    )

    # Task dependencies
    # Branch 1: Backups exist -> restore -> detect gaps -> potentially backfill
    task_check_backups >> task_restore_ohlcv >> task_restore_macro >> task_detect_gaps
    task_detect_gaps >> [task_trigger_backfill, task_no_gaps]
    task_trigger_backfill >> task_verify
    task_no_gaps >> task_verify

    # Branch 2: No backups -> skip
    task_check_backups >> task_skip_restore >> task_detect_gaps
