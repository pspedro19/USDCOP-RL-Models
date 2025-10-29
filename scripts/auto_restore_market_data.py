#!/usr/bin/env python3
"""
Auto-Restore Market Data from Backup
======================================
Este script restaura automáticamente los datos de market_data desde el backup más reciente.
Uso: python auto_restore_market_data.py
"""

import os
import sys
import psycopg2
import pandas as pd
import gzip
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PostgreSQL configuration
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'usdcop-postgres-timescale'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'usdcop_trading'),
    'user': os.getenv('POSTGRES_USER', 'admin'),
    'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
}

# Backup configuration
BACKUP_BASE_DIR = Path(__file__).parent.parent / 'data' / 'backups'

def get_latest_backup():
    """Encontrar el backup más reciente"""
    if not BACKUP_BASE_DIR.exists():
        logger.error(f"❌ Backup directory not found: {BACKUP_BASE_DIR}")
        return None

    backup_files = list(BACKUP_BASE_DIR.glob('*/market_data.csv.gz'))
    if not backup_files:
        logger.error("❌ No backup files found")
        return None

    # Ordenar por fecha de modificación (más reciente primero)
    latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"📁 Latest backup: {latest_backup}")
    return latest_backup

def check_database_connection():
    """Verificar conexión a PostgreSQL"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        conn.close()
        logger.info("✅ PostgreSQL connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False

def check_data_exists():
    """Verificar si ya hay datos en market_data"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data;")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        logger.info(f"📊 Current records in market_data: {count}")
        return count
    except Exception as e:
        logger.error(f"❌ Error checking data: {e}")
        return 0

def restore_data_from_backup(backup_file):
    """Restaurar datos desde el archivo de backup"""
    logger.info(f"🔄 Starting data restore from: {backup_file}")

    try:
        # Read compressed CSV
        logger.info("📖 Reading backup file...")
        with gzip.open(backup_file, 'rt') as f:
            df = pd.read_csv(f)

        logger.info(f"📊 Loaded {len(df)} records from backup")

        # Connect to database
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()

        # Prepare data for insertion
        logger.info("💾 Inserting data into database...")
        insert_query = """
            INSERT INTO market_data (timestamp, symbol, price, bid, ask, volume, source, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp, symbol) DO NOTHING;
        """

        # Convert DataFrame to list of tuples
        data_tuples = [tuple(row) for row in df.values]

        # Batch insert
        from psycopg2.extras import execute_batch
        execute_batch(cursor, insert_query, data_tuples, page_size=1000)

        conn.commit()

        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM market_data;")
        final_count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        logger.info(f"✅ Successfully loaded {final_count} records into market_data")
        logger.info("📊 Running ANALYZE to update statistics...")

        # Update statistics
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute("ANALYZE market_data;")
        cursor.close()
        conn.close()

        return final_count

    except Exception as e:
        logger.error(f"❌ Error during restore: {e}")
        return 0

def main():
    """Main execution"""
    logger.info("=" * 60)
    logger.info("  USDCOP Trading System - Auto Data Restore")
    logger.info("=" * 60)

    # Check database connection
    if not check_database_connection():
        logger.error("❌ Cannot connect to database. Exiting.")
        sys.exit(1)

    # Check if data already exists
    existing_count = check_data_exists()
    if existing_count > 10000:
        logger.info(f"✅ Database already has {existing_count} records. No restore needed.")
        return

    logger.info("📦 Database is empty or has insufficient data. Starting restore...")

    # Find latest backup
    backup_file = get_latest_backup()
    if not backup_file:
        logger.error("❌ No backup found. Cannot restore data.")
        sys.exit(1)

    # Restore data
    final_count = restore_data_from_backup(backup_file)

    if final_count > 0:
        logger.info("=" * 60)
        logger.info(f"🎉 Data restore complete! {final_count} records loaded.")
        logger.info("=" * 60)
    else:
        logger.error("❌ Data restore failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
