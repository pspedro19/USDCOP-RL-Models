#!/usr/bin/env python3
"""
Build features for 2025 backtest.
Populates inference_features_5m from OHLCV + Macro data.

Usage: python scripts/build_2025_features.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import psycopg2


def get_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "admin123"),
        database=os.getenv("POSTGRES_DB", "usdcop_trading")
    )


def calculate_rsi(prices: pd.Series, period: int = 9) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR as percentage of close."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()

    return (atr / close) * 100


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX indicator."""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.rolling(window=period, min_periods=period).mean()

    return adx


def main():
    print("=" * 60)
    print("BUILDING FEATURES FOR 2025 BACKTEST")
    print("=" * 60)

    conn = get_connection()

    # 1. Load OHLCV data for 2025
    print("\n1. Loading OHLCV data...")
    ohlcv_query = """
        SELECT time, open, high, low, close, volume
        FROM usdcop_m5_ohlcv
        WHERE time >= '2025-01-01' AND time < '2026-01-01'
        ORDER BY time
    """
    ohlcv = pd.read_sql(ohlcv_query, conn)
    ohlcv['time'] = pd.to_datetime(ohlcv['time'])
    ohlcv = ohlcv.set_index('time')
    print(f"   Loaded {len(ohlcv)} OHLCV rows")

    # 2. Load Macro data for 2025
    print("\n2. Loading Macro data...")
    macro_query = """
        SELECT
            fecha,
            fxrt_index_dxy_usa_d_dxy as dxy,
            volt_vix_usa_d_vix as vix,
            risk_embi_spread_col_d_embi as embi,
            comm_oil_brent_glb_d_brent as brent,
            fxrt_spot_usdmxn_mex_d_usdmxn as usdmxn,
            finc_bond_yield10y_usa_d_ust10y as ust10y,
            finc_bond_yield2y_usa_d_dgs2 as ust2y
        FROM macro_indicators_daily
        WHERE fecha >= '2024-12-01' AND fecha <= '2025-12-31'
        ORDER BY fecha
    """
    macro = pd.read_sql(macro_query, conn)
    macro['fecha'] = pd.to_datetime(macro['fecha'])
    macro = macro.set_index('fecha')
    macro = macro.ffill(limit=5)
    print(f"   Loaded {len(macro)} Macro rows")

    # 3. Calculate features
    print("\n3. Calculating features...")

    # Log returns
    ohlcv['log_ret_5m'] = np.log(ohlcv['close'] / ohlcv['close'].shift(1))
    ohlcv['log_ret_1h'] = np.log(ohlcv['close'] / ohlcv['close'].shift(12))
    ohlcv['log_ret_4h'] = np.log(ohlcv['close'] / ohlcv['close'].shift(48))

    # Technical indicators
    ohlcv['rsi_9'] = calculate_rsi(ohlcv['close'], period=9)
    ohlcv['atr_pct'] = calculate_atr_pct(ohlcv, period=14)
    ohlcv['adx_14'] = calculate_adx(ohlcv, period=14)

    # Z-score parameters (from training)
    zscore_params = {
        'dxy': {'mean': 104.5, 'std': 3.0},
        'vix': {'mean': 18.0, 'std': 5.0},
        'embi': {'mean': 350.0, 'std': 50.0},
    }

    # Macro z-scores
    macro['dxy_z'] = ((macro['dxy'] - zscore_params['dxy']['mean']) / zscore_params['dxy']['std']).clip(-3, 3)
    macro['vix_z'] = ((macro['vix'] - zscore_params['vix']['mean']) / zscore_params['vix']['std']).clip(-3, 3)
    macro['embi_z'] = ((macro['embi'] - zscore_params['embi']['mean']) / zscore_params['embi']['std']).clip(-3, 3)

    # Daily changes
    macro['dxy_change_1d'] = macro['dxy'].pct_change().clip(-0.05, 0.05)
    macro['brent_change_1d'] = macro['brent'].pct_change().clip(-0.1, 0.1)
    macro['usdmxn_change_1d'] = macro['usdmxn'].pct_change().clip(-0.05, 0.05)

    # Rate spread
    macro['rate_spread'] = macro['ust10y'] - macro['ust2y']

    print("   Features calculated")

    # 4. Merge OHLCV with Macro using backward merge (no look-ahead)
    print("\n4. Merging OHLCV with Macro (backward merge)...")

    ohlcv_reset = ohlcv.reset_index()
    macro_reset = macro.reset_index()

    # Convert fecha to datetime for merge
    ohlcv_reset['merge_date'] = pd.to_datetime(ohlcv_reset['time'].dt.date)
    macro_reset['merge_date'] = pd.to_datetime(macro_reset['fecha'])

    merged = pd.merge_asof(
        ohlcv_reset.sort_values('merge_date'),
        macro_reset[['merge_date', 'dxy_z', 'vix_z', 'embi_z', 'dxy_change_1d',
                     'brent_change_1d', 'rate_spread', 'usdmxn_change_1d']].sort_values('merge_date'),
        on='merge_date',
        direction='backward'
    )

    print(f"   Merged {len(merged)} rows")

    # 5. Prepare for insert
    print("\n5. Preparing data for insert...")

    features = merged[[
        'time', 'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
        'dxy_z', 'vix_z', 'embi_z',
        'dxy_change_1d', 'brent_change_1d', 'rate_spread',
        'rsi_9', 'atr_pct', 'adx_14', 'usdmxn_change_1d'
    ]].copy()

    features['updated_at'] = datetime.utcnow()

    # Debug: check NaN counts per column
    print("   NaN counts per column:")
    for col in features.columns:
        nan_count = features[col].isna().sum()
        if nan_count > 0:
            print(f"      {col}: {nan_count} NaNs")

    # Only drop rows where technical indicators are NaN (warmup period)
    # Keep rows even if some macro features are NaN
    features = features.dropna(subset=['log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'rsi_9', 'atr_pct', 'adx_14'])
    print(f"   {len(features)} rows ready for insert (after dropping warmup NaNs)")

    # 6. Clear existing 2025 data and insert
    print("\n6. Inserting into inference_features_5m...")

    cur = conn.cursor()
    cur.execute("DELETE FROM inference_features_5m WHERE time >= '2025-01-01' AND time < '2026-01-01'")
    conn.commit()
    print("   Cleared existing 2025 data")

    # Insert in batches using executemany
    batch_size = 1000
    total_inserted = 0

    columns = ['time', 'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
               'dxy_z', 'vix_z', 'embi_z', 'dxy_change_1d',
               'brent_change_1d', 'rate_spread', 'rsi_9', 'atr_pct',
               'adx_14', 'usdmxn_change_1d', 'updated_at']

    insert_sql = f"""
        INSERT INTO inference_features_5m ({', '.join(columns)})
        VALUES ({', '.join(['%s'] * len(columns))})
    """

    for i in range(0, len(features), batch_size):
        batch = features.iloc[i:i+batch_size]
        values = [tuple(row) for row in batch[columns].values]
        cur.executemany(insert_sql, values)
        conn.commit()
        total_inserted += len(batch)
        if (i // batch_size + 1) % 5 == 0:
            print(f"   Inserted batch {i//batch_size + 1}: {total_inserted}/{len(features)} rows")

    print(f"   Total inserted: {total_inserted} rows")

    # 7. Verify
    print("\n7. Verifying...")

    cur.execute("""
        SELECT
            COUNT(*) as total,
            MIN(time)::date as desde,
            MAX(time)::date as hasta
        FROM inference_features_5m
        WHERE time >= '2025-01-01' AND time < '2026-01-01'
    """)
    result = cur.fetchone()

    print(f"   Total rows: {result[0]}")
    print(f"   Date range: {result[1]} to {result[2]}")

    cur.close()
    conn.close()

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
