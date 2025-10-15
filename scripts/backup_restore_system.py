#!/usr/bin/env python3
"""
Sistema de Backup y Restauración Automática para USDCOP Trading Data
=====================================================================

Funciones:
1. Crear backups automáticos de la tabla market_data
2. Restaurar automáticamente desde el último backup al inicializar
3. Mantener múltiples versiones de backup con rotación automática
4. Verificar integridad de datos
"""

import os
import sys
import psycopg2
import pandas as pd
import gzip
import json
from datetime import datetime, timedelta
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
POSTGRES_CONFIG = {
    'host': 'usdcop-postgres-timescale',
    'port': 5432,
    'database': 'usdcop_trading',
    'user': 'admin',
    'password': 'admin123'
}

# Backup configuration
BACKUP_DIR = '/home/GlobalForex/USDCOP-RL-Models/backups'
MAX_BACKUPS = 7  # Mantener últimos 7 backups
BACKUP_PREFIX = 'usdcop_market_data_backup'

def ensure_backup_directory():
    """Crear directorio de backup si no existe"""
    Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Backup directory: {BACKUP_DIR}")

def get_postgres_connection():
    """Obtener conexión a PostgreSQL"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return None

def check_table_exists():
    """Verificar si la tabla market_data existe"""
    conn = get_postgres_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'market_data'
                );
            """)
            exists = cursor.fetchone()[0]
            logger.info(f"📊 Table market_data exists: {exists}")
            return exists
    except Exception as e:
        logger.error(f"❌ Error checking table existence: {e}")
        return False
    finally:
        conn.close()

def get_table_stats():
    """Obtener estadísticas de la tabla"""
    conn = get_postgres_connection()
    if not conn:
        return None

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_records,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp,
                    COUNT(DISTINCT symbol) as symbols_count
                FROM market_data
                WHERE symbol = 'USDCOP';
            """)
            stats = cursor.fetchone()
            return {
                'total_records': stats[0],
                'first_timestamp': stats[1],
                'last_timestamp': stats[2],
                'symbols_count': stats[3]
            }
    except Exception as e:
        logger.error(f"❌ Error getting table stats: {e}")
        return None
    finally:
        conn.close()

def create_backup():
    """Crear backup completo de la tabla market_data"""
    logger.info("🔄 Creating backup of market_data table...")

    ensure_backup_directory()

    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"{BACKUP_PREFIX}_{timestamp}"
    backup_path = os.path.join(BACKUP_DIR, f"{backup_filename}.sql.gz")
    metadata_path = os.path.join(BACKUP_DIR, f"{backup_filename}_metadata.json")

    # Get table stats before backup
    stats = get_table_stats()
    if not stats:
        logger.error("❌ Cannot get table stats for backup")
        return False

    try:
        # Create SQL dump using pg_dump
        dump_command = f"""
        docker exec usdcop-postgres-timescale pg_dump -U admin -d usdcop_trading \
        --table=market_data --data-only --inserts \
        --where="symbol='USDCOP'" | gzip > {backup_path}
        """

        logger.info(f"📦 Creating backup: {backup_filename}")
        result = os.system(dump_command)

        if result == 0:
            # Create metadata file
            metadata = {
                'backup_timestamp': timestamp,
                'backup_date': datetime.now().isoformat(),
                'table_stats': stats,
                'backup_file': f"{backup_filename}.sql.gz",
                'backup_size_bytes': os.path.getsize(backup_path) if os.path.exists(backup_path) else 0
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"✅ Backup created successfully: {backup_filename}")
            logger.info(f"📊 Records backed up: {stats['total_records']:,}")
            logger.info(f"📅 Date range: {stats['first_timestamp']} → {stats['last_timestamp']}")

            # Clean old backups
            cleanup_old_backups()

            return True
        else:
            logger.error(f"❌ Backup creation failed with exit code: {result}")
            return False

    except Exception as e:
        logger.error(f"❌ Error creating backup: {e}")
        return False

def get_latest_backup():
    """Obtener información del backup más reciente"""
    ensure_backup_directory()

    try:
        # Find all metadata files
        metadata_files = []
        for file in os.listdir(BACKUP_DIR):
            if file.endswith('_metadata.json'):
                file_path = os.path.join(BACKUP_DIR, file)
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                    metadata['metadata_file'] = file
                    metadata_files.append(metadata)

        if not metadata_files:
            logger.warning("⚠️ No backup metadata files found")
            return None

        # Sort by backup timestamp (newest first)
        latest_backup = sorted(metadata_files, key=lambda x: x['backup_timestamp'], reverse=True)[0]

        logger.info(f"📋 Latest backup: {latest_backup['backup_file']}")
        logger.info(f"📅 Backup date: {latest_backup['backup_date']}")
        logger.info(f"📊 Records: {latest_backup['table_stats']['total_records']:,}")

        return latest_backup

    except Exception as e:
        logger.error(f"❌ Error finding latest backup: {e}")
        return None

def restore_from_backup(backup_info=None):
    """Restaurar desde backup"""
    if backup_info is None:
        backup_info = get_latest_backup()

    if not backup_info:
        logger.error("❌ No backup available for restoration")
        return False

    backup_file = os.path.join(BACKUP_DIR, backup_info['backup_file'])

    if not os.path.exists(backup_file):
        logger.error(f"❌ Backup file not found: {backup_file}")
        return False

    logger.info(f"🔄 Restoring from backup: {backup_info['backup_file']}")

    try:
        # First create the table structure if it doesn't exist
        create_table_if_not_exists()

        # Clear existing data
        conn = get_postgres_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM market_data WHERE symbol = 'USDCOP';")
                conn.commit()
            conn.close()
            logger.info("🗑️ Cleared existing USDCOP data")

        # Restore from backup
        restore_command = f"""
        zcat {backup_file} | docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading
        """

        result = os.system(restore_command)

        if result == 0:
            # Verify restoration
            current_stats = get_table_stats()
            if current_stats and current_stats['total_records'] > 0:
                logger.info("✅ Backup restoration completed successfully")
                logger.info(f"📊 Restored records: {current_stats['total_records']:,}")
                logger.info(f"📅 Date range: {current_stats['first_timestamp']} → {current_stats['last_timestamp']}")
                return True
            else:
                logger.error("❌ Restoration verification failed - no data found")
                return False
        else:
            logger.error(f"❌ Restoration failed with exit code: {result}")
            return False

    except Exception as e:
        logger.error(f"❌ Error during restoration: {e}")
        return False

def create_table_if_not_exists():
    """Crear tabla market_data si no existe"""
    conn = get_postgres_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    price NUMERIC NOT NULL,
                    bid NUMERIC,
                    ask NUMERIC,
                    volume BIGINT,
                    source TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (timestamp, symbol)
                );

                -- Create hypertable if not exists (TimescaleDB)
                SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

                -- Create indexes
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp
                ON market_data (symbol, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_market_data_timestamp
                ON market_data (timestamp DESC);
            """)
            conn.commit()
            logger.info("✅ Table market_data created/verified")
            return True
    except Exception as e:
        logger.error(f"❌ Error creating table: {e}")
        return False
    finally:
        conn.close()

def cleanup_old_backups():
    """Limpiar backups antiguos, mantener solo los últimos MAX_BACKUPS"""
    try:
        # Get all backup files
        backup_files = []
        for file in os.listdir(BACKUP_DIR):
            if file.startswith(BACKUP_PREFIX) and file.endswith('.sql.gz'):
                file_path = os.path.join(BACKUP_DIR, file)
                backup_files.append((file, os.path.getmtime(file_path)))

        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x[1], reverse=True)

        # Remove old backups
        if len(backup_files) > MAX_BACKUPS:
            for old_backup, _ in backup_files[MAX_BACKUPS:]:
                old_backup_path = os.path.join(BACKUP_DIR, old_backup)
                old_metadata_path = old_backup_path.replace('.sql.gz', '_metadata.json')

                if os.path.exists(old_backup_path):
                    os.remove(old_backup_path)
                if os.path.exists(old_metadata_path):
                    os.remove(old_metadata_path)

                logger.info(f"🗑️ Removed old backup: {old_backup}")

    except Exception as e:
        logger.error(f"❌ Error cleaning old backups: {e}")

def auto_initialize():
    """Inicialización automática del sistema"""
    logger.info("🚀 AUTO-INITIALIZE: USDCOP Trading Data System")
    logger.info("=" * 60)

    # Check if table exists and has data
    if check_table_exists():
        stats = get_table_stats()
        if stats and stats['total_records'] > 0:
            logger.info(f"✅ Table exists with {stats['total_records']:,} records")
            logger.info("📦 Creating backup of current data...")
            create_backup()
            return True

    # Table doesn't exist or is empty, try to restore from backup
    logger.info("📋 Table is empty or doesn't exist, attempting restore from backup...")
    latest_backup = get_latest_backup()

    if latest_backup:
        success = restore_from_backup(latest_backup)
        if success:
            logger.info("✅ Successfully restored from backup")
            return True

    logger.warning("⚠️ No backup available, system will start with empty table")
    create_table_if_not_exists()
    return False

def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print("""
Uso: python3 backup_restore_system.py <comando>

Comandos:
  auto-init    - Inicialización automática (restaurar desde backup si tabla vacía)
  backup       - Crear backup de la tabla actual
  restore      - Restaurar desde el último backup
  list         - Listar backups disponibles
  stats        - Mostrar estadísticas de la tabla actual
        """)
        return

    command = sys.argv[1]

    if command == 'auto-init':
        auto_initialize()
    elif command == 'backup':
        create_backup()
    elif command == 'restore':
        restore_from_backup()
    elif command == 'list':
        list_backups()
    elif command == 'stats':
        show_stats()
    else:
        print(f"❌ Comando desconocido: {command}")

def list_backups():
    """Listar todos los backups disponibles"""
    ensure_backup_directory()

    logger.info("📋 Available backups:")
    logger.info("=" * 50)

    try:
        metadata_files = []
        for file in os.listdir(BACKUP_DIR):
            if file.endswith('_metadata.json'):
                file_path = os.path.join(BACKUP_DIR, file)
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                    metadata_files.append(metadata)

        if not metadata_files:
            logger.info("⚠️ No backups found")
            return

        # Sort by backup timestamp (newest first)
        metadata_files.sort(key=lambda x: x['backup_timestamp'], reverse=True)

        for i, backup in enumerate(metadata_files):
            status = "📦 LATEST" if i == 0 else "📋"
            logger.info(f"{status} {backup['backup_file']}")
            logger.info(f"   📅 Date: {backup['backup_date']}")
            logger.info(f"   📊 Records: {backup['table_stats']['total_records']:,}")
            logger.info(f"   💾 Size: {backup['backup_size_bytes']:,} bytes")
            logger.info("")

    except Exception as e:
        logger.error(f"❌ Error listing backups: {e}")

def show_stats():
    """Mostrar estadísticas actuales de la tabla"""
    logger.info("📊 Current table statistics:")
    logger.info("=" * 40)

    if not check_table_exists():
        logger.warning("⚠️ Table market_data does not exist")
        return

    stats = get_table_stats()
    if stats:
        logger.info(f"📊 Total records: {stats['total_records']:,}")
        logger.info(f"📅 First record: {stats['first_timestamp']}")
        logger.info(f"📅 Last record: {stats['last_timestamp']}")
        logger.info(f"🔣 Symbols: {stats['symbols_count']}")
    else:
        logger.warning("⚠️ Could not retrieve table statistics")

if __name__ == "__main__":
    main()