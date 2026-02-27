#!/usr/bin/env python3
"""
Upsert new M5 OHLCV candles from parquet into PostgreSQL.

Reads the parquet, filters rows newer than current DB max(time),
and inserts them using ON CONFLICT DO NOTHING (composite PK: time, symbol).
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# --- Config ---
PARQUET_PATH = "/home/globalforex/Documents/USDCOP-RL/USDCOP-RL-Models/seeds/latest/usdcop_m5_ohlcv.parquet"
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "admin",
    "password": "admin123",
    "dbname": "usdcop_trading",
}
TABLE = "public.usdcop_m5_ohlcv"
BATCH_SIZE = 2000


def main():
    # 1. Connect to DB and get current max time
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(f"SELECT COUNT(*), MIN(time), MAX(time) FROM {TABLE};")
    count_before, min_time, max_time = cur.fetchone()
    print(f"[DB BEFORE] count={count_before}, min={min_time}, max={max_time}")

    # 2. Read parquet
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[PARQUET]   rows={len(df)}, time range: {df.index.min()} -> {df.index.max()}")

    # 3. Convert index to UTC for comparison
    df.index = df.index.tz_convert("UTC")

    # 4. Filter only new rows (after current DB max)
    if max_time is not None:
        df_new = df[df.index > max_time].copy()
    else:
        df_new = df.copy()

    print(f"[NEW ROWS]  {len(df_new)} rows to upsert (after {max_time})")

    if df_new.empty:
        print("Nothing to insert. DB is already up to date.")
        cur.close()
        conn.close()
        return

    # 5. Prepare data tuples matching table columns:
    #    (time, symbol, open, high, low, close, volume, source)
    # Volume is bigint in DB, cast from float; source defaults to 'twelvedata'
    rows = []
    for ts, row in df_new.iterrows():
        rows.append((
            ts,                             # time (tz-aware)
            row["symbol"],                  # symbol
            round(float(row["open"]), 6),   # open  numeric(12,6)
            round(float(row["high"]), 6),   # high
            round(float(row["low"]), 6),    # low
            round(float(row["close"]), 6),  # close
            int(row["volume"]),             # volume bigint
            "twelvedata",                   # source
        ))

    # 6. Upsert using INSERT ... ON CONFLICT DO NOTHING
    insert_sql = f"""
        INSERT INTO {TABLE} (time, symbol, open, high, low, close, volume, source)
        VALUES %s
        ON CONFLICT (time, symbol) DO NOTHING
    """

    total_inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        execute_values(cur, insert_sql, batch, page_size=BATCH_SIZE)
        total_inserted += len(batch)
        print(f"  Batch {i // BATCH_SIZE + 1}: sent {len(batch)} rows (total: {total_inserted})")

    conn.commit()
    print(f"[COMMIT]    {total_inserted} rows sent to DB.")

    # 7. Verify
    cur.execute(f"SELECT COUNT(*), MIN(time), MAX(time) FROM {TABLE};")
    count_after, min_after, max_after = cur.fetchone()
    print(f"[DB AFTER]  count={count_after}, min={min_after}, max={max_after}")
    print(f"[DELTA]     +{count_after - count_before} rows")

    cur.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
