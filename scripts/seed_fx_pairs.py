"""
Seed BRL + MXN historical OHLCV data from parquet into PostgreSQL.
Skips COP (already has 93K+ rows from backup restore).

Usage:
    python scripts/seed_fx_pairs.py
    python scripts/seed_fx_pairs.py --symbols USD/BRL USD/MXN
    python scripts/seed_fx_pairs.py --include-cop   # Also re-seed COP
"""
import argparse
import os
import sys
import time

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


def get_connection():
    """Connect to PostgreSQL using env vars or defaults."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "admin123"),
        dbname=os.getenv("POSTGRES_DB", "usdcop_trading"),
    )


def seed_symbols(symbols: list[str], parquet_path: str, batch_size: int = 10_000):
    """Load OHLCV data from parquet into usdcop_m5_ohlcv for given symbols."""
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    if "time" not in df.columns and df.index.name == "time":
        df = df.reset_index()

    # Filter to requested symbols
    df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        print(f"No rows found for symbols {symbols}")
        return

    print(f"Loaded {len(df):,} rows for {df['symbol'].nunique()} symbols")
    for sym in symbols:
        count = len(df[df["symbol"] == sym])
        print(f"  {sym}: {count:,} rows")

    # Ensure volume column exists
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # Fix OHLC consistency (some bars have low > close or high < close)
    df["low"] = df[["low", "close", "open"]].min(axis=1)
    df["high"] = df[["high", "close", "open"]].max(axis=1)

    conn = get_connection()
    cur = conn.cursor()

    upsert_sql = """
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

    total_inserted = 0
    t0 = time.time()

    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_values("time")
        rows = [
            (row["time"], row["symbol"], row["open"], row["high"],
             row["low"], row["close"], row["volume"], "seed_restore")
            for _, row in sym_df.iterrows()
        ]

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            execute_values(cur, upsert_sql, batch, page_size=batch_size)
            conn.commit()
            total_inserted += len(batch)
            print(f"  {sym}: {total_inserted:,} / {len(df):,} rows...", end="\r")

        print(f"  {sym}: {len(rows):,} rows upserted")

    elapsed = time.time() - t0
    print(f"\nDone: {total_inserted:,} rows in {elapsed:.1f}s")

    # Verify
    for sym in symbols:
        cur.execute(
            "SELECT COUNT(*), MIN(time)::date, MAX(time)::date "
            "FROM usdcop_m5_ohlcv WHERE symbol = %s",
            (sym,),
        )
        count, min_date, max_date = cur.fetchone()
        print(f"  DB {sym}: {count:,} rows ({min_date} -> {max_date})")

    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Seed FX pairs from parquet")
    parser.add_argument(
        "--symbols", nargs="+", default=["USD/BRL", "USD/MXN"],
        help="Symbols to seed (default: BRL + MXN)"
    )
    parser.add_argument(
        "--include-cop", action="store_true",
        help="Also re-seed USD/COP"
    )
    parser.add_argument(
        "--parquet", default="seeds/latest/fx_multi_m5_ohlcv.parquet",
        help="Path to unified FX parquet"
    )
    args = parser.parse_args()

    symbols = list(args.symbols)
    if args.include_cop:
        symbols.append("USD/COP")

    seed_symbols(symbols, args.parquet)


if __name__ == "__main__":
    main()
