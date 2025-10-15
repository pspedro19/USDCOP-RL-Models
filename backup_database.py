#!/usr/bin/env python3
import os
import json
import psycopg2
from psycopg2 import sql
from datetime import datetime
import pandas as pd
import sys

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'usdcop_trading',
    'user': 'admin',
    'password': 'admin123'
}

def create_backup_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f"data/backups/{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir

def backup_table_to_csv(conn, table_name, backup_dir):
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)

        output_file = os.path.join(backup_dir, f"{table_name}.csv")
        df.to_csv(output_file, index=False)

        print(f"✓ Backed up {table_name}: {len(df)} rows to {output_file}")
        return len(df)
    except Exception as e:
        print(f"✗ Failed to backup {table_name}: {str(e)}")
        return 0

def backup_table_to_json(conn, table_name, backup_dir):
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)

        output_file = os.path.join(backup_dir, f"{table_name}.json")
        df.to_json(output_file, orient='records', date_format='iso', indent=2)

        print(f"✓ Backed up {table_name}: {len(df)} rows to {output_file}")
        return len(df)
    except Exception as e:
        print(f"✗ Failed to backup {table_name}: {str(e)}")
        return 0

def get_table_list(conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return tables

def main():
    print("Starting database backup...")

    backup_dir = create_backup_dir()
    print(f"Backup directory: {backup_dir}")

    try:
        # Connect to database
        print(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        conn = psycopg2.connect(**DB_CONFIG)
        print("✓ Connected to database")

        # Get all tables
        tables = get_table_list(conn)
        print(f"Found {len(tables)} tables")

        # Priority tables for trading system
        priority_tables = [
            'market_data',
            'l0_raw_data',
            'l1_standardized_data',
            'l2_prepared_data',
            'l3_feature_data',
            'l4_rlready_data',
            'l5_serving_data',
            'trades',
            'positions',
            'signals'
        ]

        # Backup priority tables first
        total_rows = 0
        for table in priority_tables:
            if table in tables:
                rows = backup_table_to_csv(conn, table, backup_dir)
                total_rows += rows

                # Also backup as JSON for important tables
                if table == 'market_data':
                    backup_table_to_json(conn, table, backup_dir)

        # Backup remaining tables
        for table in tables:
            if table not in priority_tables:
                rows = backup_table_to_csv(conn, table, backup_dir)
                total_rows += rows

        # Create metadata file
        metadata = {
            'backup_timestamp': datetime.now().isoformat(),
            'database': DB_CONFIG['database'],
            'tables_count': len(tables),
            'total_rows': total_rows,
            'tables': tables
        }

        metadata_file = os.path.join(backup_dir, 'backup_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Backup complete!")
        print(f"   Total tables: {len(tables)}")
        print(f"   Total rows: {total_rows:,}")
        print(f"   Location: {backup_dir}")

        conn.close()

    except psycopg2.OperationalError as e:
        print(f"\n❌ Could not connect to database: {str(e)}")
        print("\nTrying alternative connection methods...")

        # Try with different host
        alt_hosts = ['127.0.0.1', 'postgres', 'usdcop-postgres-timescale']
        for host in alt_hosts:
            try:
                print(f"Trying host: {host}")
                DB_CONFIG['host'] = host
                conn = psycopg2.connect(**DB_CONFIG)
                print(f"✓ Connected via {host}")
                main()  # Recursive call with new config
                return
            except:
                continue

        print("\n❌ All connection attempts failed")
        print("Please ensure PostgreSQL is running and accessible")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()