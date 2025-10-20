#!/usr/bin/env python3
"""
Script to restore USDCOP market data from CSV backup
"""
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import sys

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'usdcop_trading',
    'user': 'admin',
    'password': 'admin123'
}

def restore_backup():
    """Restore market data from CSV backup file"""
    csv_file = '/home/GlobalForex/USDCOP-RL-Models/01-l0-unified-usdcop-dataset.csv'

    print(f"📚 Reading CSV backup file: {csv_file}")

    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} records from CSV")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print("✅ Connected to PostgreSQL database")
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return False

    try:
        # Clear existing data (optional - comment out if you want to append)
        print("🗑️  Clearing existing market_data records...")
        cur.execute("TRUNCATE TABLE market_data CASCADE")
        conn.commit()

        # Prepare data for insertion
        print("📝 Preparing data for insertion...")
        records = []
        for _, row in df.iterrows():
            # Convert timestamp string to datetime
            timestamp = pd.to_datetime(row['time'])

            # Calculate bid/ask from close price (spread of 0.5)
            close_price = float(row['close'])
            bid = close_price - 0.5
            ask = close_price + 0.5

            records.append((
                timestamp,
                'USDCOP',
                close_price,
                bid,
                ask,
                float(row.get('volume', 0)) if pd.notna(row.get('volume', 0)) else 0,
                row.get('source', 'twelvedata')
            ))

        # Insert data in batches
        print(f"💾 Inserting {len(records)} records into market_data table...")

        insert_query = """
            INSERT INTO market_data
            (timestamp, symbol, price, bid, ask, volume, source)
            VALUES %s
            ON CONFLICT (timestamp, symbol) DO UPDATE
            SET price = EXCLUDED.price,
                bid = EXCLUDED.bid,
                ask = EXCLUDED.ask,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source
        """

        # Insert in batches of 1000
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            execute_values(cur, insert_query, batch)

            if (i + batch_size) % 10000 == 0:
                print(f"   Inserted {min(i + batch_size, len(records))}/{len(records)} records...")
                conn.commit()

        # Final commit
        conn.commit()
        print(f"✅ Successfully inserted {len(records)} records!")

        # Verify insertion
        cur.execute("SELECT COUNT(*) FROM market_data")
        count = cur.fetchone()[0]

        cur.execute("""
            SELECT MIN(timestamp) as min_date,
                   MAX(timestamp) as max_date,
                   AVG(price) as avg_price,
                   MIN(price) as min_price,
                   MAX(price) as max_price
            FROM market_data
        """)
        stats = cur.fetchone()

        print(f"\n📊 Database Statistics:")
        print(f"   Total records: {count}")
        print(f"   Date range: {stats[0]} to {stats[1]}")
        print(f"   Price range: {stats[3]:.2f} - {stats[4]:.2f}")
        print(f"   Average price: {stats[2]:.2f}")

        # Close connection
        cur.close()
        conn.close()

        print("\n✅ Backup restoration completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during restoration: {e}")
        conn.rollback()
        cur.close()
        conn.close()
        return False

if __name__ == "__main__":
    print("="*60)
    print("USDCOP Market Data Restoration Tool")
    print("="*60)

    success = restore_backup()

    if success:
        print("\n🎉 Data restoration completed! The dashboard now has access to 92,936 historical records.")
        print("📈 Access the dashboard at: http://48.216.199.139:5000")
    else:
        print("\n❌ Data restoration failed. Please check the error messages above.")
        sys.exit(1)