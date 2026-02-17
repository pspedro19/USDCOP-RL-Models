#!/usr/bin/env python3
"""
USDCOP Trading System - Automatic Data Seeding Script
======================================================

Loads initial data into PostgreSQL/TimescaleDB on first startup:
1. OHLCV 5-min data from backup (usdcop_m5_ohlcv_*.csv.gz)
2. Macro daily data from MACRO_DAILY_CONSOLIDATED.csv
3. Features pre-calculated from backup (optional)
4. Triggers L0 backfill DAG to update data to current time

Features:
- Automatically finds the most recent backup file
- UPSERT mode (ON CONFLICT DO UPDATE) for idempotent seeding
- Supports 3 tables: OHLCV, Macro, Features
- Detailed logging with row counts
- Post-load validation
- Configurable via environment variables

Run this script after database initialization.
Uses UPSERT for idempotent re-runs (safe to run multiple times).

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
Updated: 2025-12-26 - Added UPSERT, features table, validation
"""

import os
import sys
import gzip
import logging
import psycopg2
from psycopg2 import sql
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration from environment
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'timescaledb'),
    'port': int(os.environ.get('POSTGRES_PORT', 5432)),
    'database': os.environ.get('POSTGRES_DB', 'usdcop'),
    'user': os.environ.get('POSTGRES_USER', 'admin'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'admin123')
}

# Data paths - Support both local development and Docker container
# In Docker: /app/backups and /app/data/pipeline are mounted volumes
# Local: use relative paths from project root

def get_data_paths():
    """Get data paths based on environment (Docker or local)."""
    # Check if running in Docker (container paths exist)
    docker_backup_dir = Path('/app/data/backups')
    docker_macro_path = Path('/app/data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet')

    if docker_backup_dir.exists():
        # Running in Docker container
        project_root = Path('/app')
        return docker_backup_dir, docker_macro_path, project_root
    else:
        # Running locally (development)
        project_root = Path(__file__).parent.parent
        return (
            project_root / 'data' / 'backups',
            project_root / 'data' / 'pipeline' / '04_cleaning' / 'output' / 'MACRO_DAILY_CLEAN.parquet',
            project_root,
        )

OHLCV_BACKUP_DIR, MACRO_DATA_PATH, PROJECT_ROOT = get_data_paths()

# Daily backup parquets (freshest, written by l0_seed_backup DAG)
OHLCV_BACKUP_PARQUET = PROJECT_ROOT / 'data' / 'backups' / 'seeds' / 'usdcop_m5_ohlcv_backup.parquet'
MACRO_BACKUP_PARQUET = PROJECT_ROOT / 'data' / 'backups' / 'seeds' / 'macro_indicators_daily_backup.parquet'

# Unified multi-pair seed (preferred) and single-pair fallback
UNIFIED_SEED_PATH = PROJECT_ROOT / 'seeds' / 'latest' / 'fx_multi_m5_ohlcv.parquet'
USDCOP_SEED_PATH = PROJECT_ROOT / 'seeds' / 'latest' / 'usdcop_m5_ohlcv.parquet'


def get_connection():
    """Get database connection with retry logic."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(5)
            else:
                raise


def table_has_data(conn, table_name: str) -> bool:
    """Check if table has any data."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT EXISTS(SELECT 1 FROM {} LIMIT 1)").format(
            sql.Identifier(table_name)
        ))
        return cur.fetchone()[0]


def find_latest_ohlcv_backup() -> Path:
    """Find the most recent OHLCV backup file."""
    if not OHLCV_BACKUP_DIR.exists():
        logger.warning(f"Backup directory not found: {OHLCV_BACKUP_DIR}")
        return None

    backups = list(OHLCV_BACKUP_DIR.glob('usdcop_m5_ohlcv_*.csv.gz'))
    if not backups:
        logger.warning("No OHLCV backup files found")
        return None

    # Sort by modification time, get latest
    return max(backups, key=lambda p: p.stat().st_mtime)


def seed_ohlcv_data(conn, force_upsert: bool = False) -> int:
    """
    Load OHLCV data from seed/backup into usdcop_m5_ohlcv table.

    Priority:
      1. Unified multi-pair parquet (fx_multi_m5_ohlcv.parquet) — all 3 pairs
      2. Single-pair parquet (usdcop_m5_ohlcv.parquet) — COP only
      3. Legacy CSV backup (csv.gz) — COP only

    Args:
        conn: Database connection
        force_upsert: If True, use UPSERT even if table has data

    Returns:
        Number of rows inserted/updated
    """
    from psycopg2.extras import execute_values

    has_data = table_has_data(conn, 'usdcop_m5_ohlcv')

    if has_data and not force_upsert:
        logger.info("usdcop_m5_ohlcv table already has data, skipping OHLCV seeding")
        logger.info("(Use FORCE_UPSERT=true to force update)")
        return 0

    # Priority: daily backup (freshest) -> unified seed -> single-pair seed -> legacy CSV
    df = None
    source_label = None

    if OHLCV_BACKUP_PARQUET.exists():
        df = pd.read_parquet(OHLCV_BACKUP_PARQUET)
        source_label = f"daily backup ({OHLCV_BACKUP_PARQUET.name})"
        logger.info(f"Loading from {source_label}: {len(df):,} rows")

    elif UNIFIED_SEED_PATH.exists():
        df = pd.read_parquet(UNIFIED_SEED_PATH)
        source_label = f"unified seed ({UNIFIED_SEED_PATH.name})"
        logger.info(f"Loading from {source_label}: {len(df):,} rows")

    elif USDCOP_SEED_PATH.exists():
        df = pd.read_parquet(USDCOP_SEED_PATH)
        source_label = f"single-pair seed ({USDCOP_SEED_PATH.name})"
        logger.info(f"Loading from {source_label}: {len(df):,} rows")

    else:
        backup_file = find_latest_ohlcv_backup()
        if not backup_file:
            logger.warning("No OHLCV seed or backup found, table will be empty")
            return 0
        with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f)
        source_label = f"CSV backup ({backup_file.name})"
        logger.info(f"Loading from {source_label}: {len(df):,} rows")

    # Log symbols if available
    if df is not None and 'symbol' in df.columns:
        symbols = df['symbol'].unique()
        logger.info(f"Symbols found: {list(symbols)}")

    # Ensure required columns
    required_cols = ['time', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Seed file missing required columns. Has: {df.columns.tolist()}")
        return 0

    mode = "UPSERT" if has_data else "INSERT"
    logger.info(f"Mode: {mode}")

    # Prepare data
    df['time'] = pd.to_datetime(df['time'])
    if 'symbol' not in df.columns:
        df['symbol'] = 'USD/COP'
    df['volume'] = df['volume'].fillna(0) if 'volume' in df.columns else 0
    df['source'] = df.get('source', 'seed_restore')
    if 'source' not in df.columns:
        df['source'] = 'seed_restore'
    df['source'] = df['source'].fillna('seed_restore')

    # UPSERT all rows
    inserted = 0
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

    with conn.cursor() as cur:
        batch_size = 10000
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            execute_values(cur, query, batch, page_size=batch_size)
            inserted += len(batch)

            if i % 50000 == 0 and i > 0:
                logger.info(f"Progress: {inserted:,}/{len(data):,} rows")

        conn.commit()

    logger.info(f"Inserted/Updated {inserted:,} OHLCV records from {source_label}")

    # Per-symbol summary
    if 'symbol' in df.columns:
        for sym in df['symbol'].unique():
            count = (df['symbol'] == sym).sum()
            logger.info(f"  {sym}: {count:,} rows")

    return inserted


def seed_macro_data(conn) -> int:
    """Load macro data into macro_indicators_daily table.

    Priority: daily backup parquet -> MACRO_DAILY_CLEAN parquet -> legacy CSV.
    """
    if table_has_data(conn, 'macro_indicators_daily'):
        logger.info("macro_indicators_daily table already has data, skipping macro seeding")
        return 0

    # Try daily backup first (freshest), then MACRO_DAILY_CLEAN parquet
    macro_source = None
    df = None

    if MACRO_BACKUP_PARQUET.exists():
        macro_source = MACRO_BACKUP_PARQUET
        logger.info(f"Loading macro from daily backup: {macro_source}")
        df = pd.read_parquet(macro_source)
    elif MACRO_DATA_PATH.exists():
        macro_source = MACRO_DATA_PATH
        logger.info(f"Loading macro from: {macro_source}")
        if str(macro_source).endswith('.parquet'):
            df = pd.read_parquet(macro_source)
        else:
            df = pd.read_csv(macro_source)
    else:
        logger.warning(f"No macro data found (checked {MACRO_BACKUP_PARQUET} and {MACRO_DATA_PATH})")
        return 0

    logger.info(f"Read {len(df)} macro records from {macro_source.name}")

    # Column mapping from CSV to database (lowercase, underscore format)
    # Note: parquet from daily backup already has lowercase columns — mapping is a no-op
    column_mapping = {
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
        'POLR_PRIME_RATE_USA_D_PRIME': 'polr_prime_rate_usa_d_prime',
        'PROD_INDUSTRIAL_USA_M_INDPRO': 'prod_industrial_usa_m_indpro',
        'RSBP_CURRENT_ACCOUNT_COL_Q_CACCT': 'rsbp_current_account_col_q_cacct',
        'RSBP_FDI_OUTFLOW_COL_A_IDCE': 'rsbp_fdi_outflow_col_a_idce',
        'RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT': 'rsbp_reserves_international_col_m_resint',
        'SENT_CONSUMER_USA_M_UMCSENT': 'sent_consumer_usa_m_umcsent',
        'VOLT_VIX_USA_D_VIX': 'volt_vix_usa_d_vix'
    }

    # Rename columns
    df = df.rename(columns=column_mapping)

    # Filter to only include mapped columns that exist
    available_cols = [col for col in column_mapping.values() if col in df.columns]
    df = df[available_cols]

    # Convert fecha to datetime
    df['fecha'] = pd.to_datetime(df['fecha']).dt.date

    # Insert data
    inserted = 0
    with conn.cursor() as cur:
        cols = df.columns.tolist()
        placeholders = ', '.join(['%s'] * len(cols))
        insert_query = f"""
            INSERT INTO macro_indicators_daily ({', '.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT (fecha) DO NOTHING
        """

        for _, row in df.iterrows():
            # Convert NaN to None for SQL NULL
            values = [None if pd.isna(v) else v for v in row.values]
            cur.execute(insert_query, values)
            inserted += cur.rowcount

        conn.commit()

    logger.info(f"Inserted {inserted} macro records into macro_indicators_daily")
    return inserted


def trigger_backfill_dag() -> bool:
    """
    Trigger the L0 OHLCV backfill DAG via Airflow REST API.
    This updates the backup data to the most recent market hours.

    Returns:
        True if triggered successfully, False otherwise
    """
    airflow_host = os.environ.get('AIRFLOW_HOST', 'airflow-webserver')
    airflow_port = os.environ.get('AIRFLOW_PORT', '8080')
    airflow_user = os.environ.get('AIRFLOW_USER', 'admin')
    airflow_password = os.environ.get('AIRFLOW_PASSWORD', 'admin')

    dag_id = 'v3.l0_ohlcv_backfill'
    url = f'http://{airflow_host}:{airflow_port}/api/v1/dags/{dag_id}/dagRuns'

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    payload = {
        'conf': {'triggered_by': 'data_seeding_script'},
        'note': 'Automatic backfill after data seeding'
    }

    try:
        logger.info(f"Triggering backfill DAG: {dag_id}")
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            auth=(airflow_user, airflow_password),
            timeout=30
        )

        if response.status_code in [200, 201]:
            run_id = response.json().get('dag_run_id', 'unknown')
            logger.info(f"Backfill DAG triggered successfully: {run_id}")
            return True
        else:
            logger.warning(f"Failed to trigger backfill DAG: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        logger.warning("Airflow not available yet - backfill DAG will need manual trigger")
        logger.info("To trigger manually: airflow dags trigger v3.l0_ohlcv_backfill")
        return False
    except Exception as e:
        logger.warning(f"Error triggering backfill DAG: {e}")
        return False


def validate_seeded_data(conn) -> dict:
    """
    Validate seeded data after insertion.

    Returns:
        Dictionary with validation results
    """
    results = {}

    with conn.cursor() as cur:
        # Check OHLCV (all symbols)
        cur.execute("""
            SELECT symbol, COUNT(*), MIN(time), MAX(time)
            FROM usdcop_m5_ohlcv
            GROUP BY symbol
            ORDER BY symbol
        """)
        ohlcv_rows = cur.fetchall()
        total_count = sum(r[1] for r in ohlcv_rows)
        results['ohlcv'] = {
            'count': total_count,
            'min_date': str(min(r[2] for r in ohlcv_rows)) if ohlcv_rows else None,
            'max_date': str(max(r[3] for r in ohlcv_rows)) if ohlcv_rows else None,
            'per_symbol': {r[0]: r[1] for r in ohlcv_rows},
        }

        # Check Macro
        try:
            cur.execute("SELECT COUNT(*), MIN(fecha), MAX(fecha) FROM macro_indicators_daily")
            macro_result = cur.fetchone()
            results['macro'] = {
                'count': macro_result[0],
                'min_date': str(macro_result[1]) if macro_result[1] else None,
                'max_date': str(macro_result[2]) if macro_result[2] else None
            }
        except Exception:
            results['macro'] = {'count': 0, 'min_date': None, 'max_date': None}

    return results


def main():
    """Main data seeding function."""
    logger.info("=" * 60)
    logger.info("USDCOP Trading System - Data Seeding")
    logger.info("=" * 60)

    # Check for force upsert mode
    force_upsert = os.environ.get('FORCE_UPSERT', 'false').lower() in ('true', '1', 'yes')
    if force_upsert:
        logger.info("FORCE_UPSERT=true: Will update existing data if present")

    try:
        conn = get_connection()
        logger.info(f"Connected to database: {DB_CONFIG['database']}@{DB_CONFIG['host']}")

        # Seed OHLCV data
        ohlcv_count = seed_ohlcv_data(conn, force_upsert=force_upsert)

        # Seed macro data
        macro_count = seed_macro_data(conn)

        # Validate seeded data
        logger.info("")
        logger.info("Validating seeded data...")
        validation = validate_seeded_data(conn)

        conn.close()

        logger.info("=" * 60)
        logger.info("DATA SEEDING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"  OHLCV records inserted/updated: {ohlcv_count:,}")
        logger.info(f"  Macro records inserted/updated: {macro_count:,}")
        logger.info("")
        logger.info("DATABASE STATE:")
        logger.info(f"  OHLCV total:  {validation['ohlcv']['count']:,} rows")
        if validation['ohlcv']['min_date']:
            logger.info(f"  OHLCV range:  {validation['ohlcv']['min_date']} to {validation['ohlcv']['max_date']}")
        logger.info(f"  Macro total:  {validation['macro']['count']:,} rows")
        if validation['macro']['min_date']:
            logger.info(f"  Macro range:  {validation['macro']['min_date']} to {validation['macro']['max_date']}")
        logger.info("=" * 60)

        # Trigger backfill DAG if data was seeded
        if ohlcv_count > 0:
            logger.info("")
            logger.info("Triggering automatic backfill to update data to current time...")
            backfill_triggered = trigger_backfill_dag()
            if not backfill_triggered:
                logger.info("")
                logger.info("NOTE: To manually trigger backfill when Airflow is ready:")
                logger.info("  airflow dags trigger v3.l0_ohlcv_backfill")
                logger.info("  or via UI: http://localhost:8080")

        return 0

    except Exception as e:
        logger.error(f"Data seeding failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
