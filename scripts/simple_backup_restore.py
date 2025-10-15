#!/usr/bin/env python3
"""
Sistema Simplificado de Backup y Restauración para USDCOP Trading Data
======================================================================

Maneja backups/restore de la tabla market_data usando containers Docker existentes
"""

import os
import sys
import json
import gzip
from datetime import datetime
import subprocess

# Configuration
BACKUP_DIR = '/home/GlobalForex/USDCOP-RL-Models/backups'
POSTGRES_CONTAINER = 'usdcop-postgres-timescale'
AIRFLOW_CONTAINER = 'usdcop-airflow-webserver'

def log(message):
    """Simple logging function"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def run_docker_command(container, command):
    """Execute command in Docker container"""
    full_command = f"docker exec {container} {command}"
    result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def ensure_backup_directory():
    """Create backup directory if it doesn't exist"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    log(f"📁 Backup directory: {BACKUP_DIR}")

def check_postgres_connection():
    """Check if PostgreSQL is accessible"""
    returncode, _, _ = run_docker_command(POSTGRES_CONTAINER,
        "psql -U admin -d usdcop_trading -c 'SELECT 1;'")
    return returncode == 0

def get_table_stats():
    """Get current table statistics"""
    if not check_postgres_connection():
        return None

    query = """
    SELECT
        COUNT(*) as total_records,
        MIN(timestamp) as first_timestamp,
        MAX(timestamp) as last_timestamp
    FROM market_data
    WHERE symbol = 'USDCOP';
    """

    returncode, stdout, stderr = run_docker_command(POSTGRES_CONTAINER,
        f"psql -U admin -d usdcop_trading -t -c \"{query}\"")

    if returncode == 0:
        lines = stdout.strip().split('\n')
        if lines:
            parts = lines[0].split('|')
            if len(parts) >= 3:
                return {
                    'total_records': int(parts[0].strip()),
                    'first_timestamp': parts[1].strip(),
                    'last_timestamp': parts[2].strip()
                }
    return None

def create_backup():
    """Create backup using Airflow container"""
    log("📦 Creating backup of USDCOP data...")

    ensure_backup_directory()

    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create backup script for Airflow container
    backup_script = f"""
import pandas as pd
import psycopg2
import gzip
import json
from datetime import datetime

POSTGRES_CONFIG = {{
    'host': 'usdcop-postgres-timescale',
    'port': 5432,
    'database': 'usdcop_trading',
    'user': 'admin',
    'password': 'admin123'
}}

try:
    conn = psycopg2.connect(**POSTGRES_CONFIG)

    query = '''
    SELECT
        timestamp,
        symbol,
        price,
        bid,
        ask,
        volume,
        source,
        created_at
    FROM market_data
    WHERE symbol = 'USDCOP'
    ORDER BY timestamp;
    '''

    df = pd.read_sql_query(query, conn)

    # Save as compressed CSV
    csv_backup = '/tmp/usdcop_backup_{timestamp}.csv.gz'
    with gzip.open(csv_backup, 'wt', encoding='utf-8') as f:
        df.to_csv(f, index=False)

    # Create metadata
    metadata = {{
        'backup_timestamp': '{timestamp}',
        'backup_date': datetime.now().isoformat(),
        'total_records': len(df),
        'first_timestamp': str(df['timestamp'].min()),
        'last_timestamp': str(df['timestamp'].max()),
        'file_format': 'CSV (gzipped)'
    }}

    metadata_file = '/tmp/usdcop_backup_{timestamp}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'SUCCESS:{timestamp}:{{len(df)}}')
    conn.close()

except Exception as e:
    print(f'ERROR:{{e}}')
"""

    # Execute backup script in Airflow container
    returncode, stdout, stderr = run_docker_command(AIRFLOW_CONTAINER,
        f"python3 -c \"{backup_script}\"")

    if returncode == 0 and 'SUCCESS:' in stdout:
        # Parse results
        result_line = [line for line in stdout.split('\n') if line.startswith('SUCCESS:')][0]
        parts = result_line.split(':')
        backup_timestamp = parts[1]
        record_count = parts[2]

        # Copy files from container to host
        csv_file = f'usdcop_backup_{backup_timestamp}.csv.gz'
        metadata_file = f'usdcop_backup_{backup_timestamp}_metadata.json'

        subprocess.run(f"docker cp {AIRFLOW_CONTAINER}:/tmp/{csv_file} {BACKUP_DIR}/", shell=True)
        subprocess.run(f"docker cp {AIRFLOW_CONTAINER}:/tmp/{metadata_file} {BACKUP_DIR}/", shell=True)

        log(f"✅ Backup created successfully: {csv_file}")
        log(f"📊 Records backed up: {record_count}")

        # Cleanup old backups (keep last 5)
        cleanup_old_backups()

        return True
    else:
        log(f"❌ Backup failed: {stderr}")
        return False

def restore_from_backup(backup_file=None):
    """Restore from backup"""
    ensure_backup_directory()

    if backup_file is None:
        # Find latest backup
        backup_files = [f for f in os.listdir(BACKUP_DIR) if f.endswith('.csv.gz')]
        if not backup_files:
            log("❌ No backup files found")
            return False

        backup_file = sorted(backup_files)[-1]

    backup_path = os.path.join(BACKUP_DIR, backup_file)

    if not os.path.exists(backup_path):
        log(f"❌ Backup file not found: {backup_path}")
        return False

    log(f"🔄 Restoring from backup: {backup_file}")

    # Copy backup to Airflow container
    subprocess.run(f"docker cp {backup_path} {AIRFLOW_CONTAINER}:/tmp/{backup_file}", shell=True)

    # Create restore script
    restore_script = f"""
import pandas as pd
import psycopg2
import gzip

POSTGRES_CONFIG = {{
    'host': 'usdcop-postgres-timescale',
    'port': 5432,
    'database': 'usdcop_trading',
    'user': 'admin',
    'password': 'admin123'
}}

try:
    # Load backup data
    with gzip.open('/tmp/{backup_file}', 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f)

    # Connect to database
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cursor = conn.cursor()

    # Clear existing USDCOP data
    cursor.execute("DELETE FROM market_data WHERE symbol = 'USDCOP';")

    # Insert backup data
    for _, row in df.iterrows():
        cursor.execute('''
            INSERT INTO market_data (timestamp, symbol, price, bid, ask, volume, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (row['timestamp'], row['symbol'], row['price'], row['bid'], row['ask'], row['volume'], row['source']))

    conn.commit()

    # Verify restoration
    cursor.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'USDCOP';")
    count = cursor.fetchone()[0]

    print(f'SUCCESS:{{count}}')

    conn.close()

except Exception as e:
    print(f'ERROR:{{e}}')
"""

    # Execute restore script
    returncode, stdout, stderr = run_docker_command(AIRFLOW_CONTAINER,
        f"python3 -c \"{restore_script}\"")

    if returncode == 0 and 'SUCCESS:' in stdout:
        result_line = [line for line in stdout.split('\n') if line.startswith('SUCCESS:')][0]
        record_count = result_line.split(':')[1]

        log(f"✅ Restore completed successfully")
        log(f"📊 Records restored: {record_count}")
        return True
    else:
        log(f"❌ Restore failed: {stderr}")
        return False

def list_backups():
    """List available backups"""
    ensure_backup_directory()

    log("📋 Available backups:")
    log("=" * 50)

    backup_files = []
    for file in os.listdir(BACKUP_DIR):
        if file.endswith('_metadata.json'):
            metadata_path = os.path.join(BACKUP_DIR, file)
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    backup_files.append(metadata)
            except:
                continue

    if not backup_files:
        log("⚠️ No backups found")
        return

    # Sort by timestamp
    backup_files.sort(key=lambda x: x['backup_timestamp'], reverse=True)

    for i, backup in enumerate(backup_files):
        status = "📦 LATEST" if i == 0 else "📋"
        csv_file = f"usdcop_backup_{backup['backup_timestamp']}.csv.gz"
        log(f"{status} {csv_file}")
        log(f"   📅 Date: {backup['backup_date']}")
        log(f"   📊 Records: {backup['total_records']:,}")
        log("")

def cleanup_old_backups():
    """Keep only the 5 most recent backups"""
    backup_files = []
    for file in os.listdir(BACKUP_DIR):
        if file.startswith('usdcop_backup_') and file.endswith('.csv.gz'):
            file_path = os.path.join(BACKUP_DIR, file)
            backup_files.append((file, os.path.getmtime(file_path)))

    # Sort by modification time (newest first)
    backup_files.sort(key=lambda x: x[1], reverse=True)

    # Remove old backups (keep 5 most recent)
    if len(backup_files) > 5:
        for old_backup, _ in backup_files[5:]:
            old_backup_path = os.path.join(BACKUP_DIR, old_backup)
            old_metadata_path = old_backup_path.replace('.csv.gz', '_metadata.json')

            if os.path.exists(old_backup_path):
                os.remove(old_backup_path)
            if os.path.exists(old_metadata_path):
                os.remove(old_metadata_path)

            log(f"🗑️ Removed old backup: {old_backup}")

def auto_init():
    """Auto-initialization: restore from backup if table is empty"""
    log("🚀 AUTO-INITIALIZATION: USDCOP Trading Data System")
    log("=" * 60)

    if not check_postgres_connection():
        log("❌ Cannot connect to PostgreSQL")
        return False

    stats = get_table_stats()

    if stats and stats['total_records'] > 0:
        log(f"✅ Table exists with {stats['total_records']:,} records")
        log("📦 Creating backup of current data...")
        create_backup()
        return True

    # Table is empty, try to restore from backup
    log("📋 Table is empty, attempting restore from backup...")

    backup_files = [f for f in os.listdir(BACKUP_DIR) if f.endswith('.csv.gz')]
    if backup_files:
        latest_backup = sorted(backup_files)[-1]
        success = restore_from_backup(latest_backup)
        if success:
            log("✅ Successfully restored from backup")
            return True

    log("⚠️ No backup available, system will start with empty table")
    return False

def show_stats():
    """Show current table statistics"""
    log("📊 Current table statistics:")
    log("=" * 40)

    if not check_postgres_connection():
        log("❌ Cannot connect to PostgreSQL")
        return

    stats = get_table_stats()
    if stats:
        log(f"📊 Total records: {stats['total_records']:,}")
        log(f"📅 First record: {stats['first_timestamp']}")
        log(f"📅 Last record: {stats['last_timestamp']}")
    else:
        log("⚠️ Could not retrieve table statistics")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("""
Uso: python3 simple_backup_restore.py <comando>

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
        auto_init()
    elif command == 'backup':
        create_backup()
    elif command == 'restore':
        restore_from_backup()
    elif command == 'list':
        list_backups()
    elif command == 'stats':
        show_stats()
    else:
        log(f"❌ Comando desconocido: {command}")

if __name__ == "__main__":
    main()