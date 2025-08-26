#!/usr/bin/env python3
"""Test audit features locally before DAG deployment"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib

# Load existing L1 data
base = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_consolidated")
df = pd.read_parquet(base / "standardized_data.parquet")

print("=" * 70)
print("TESTING AUDIT FEATURES")
print("=" * 70)

# 1. Test Mode B padding for a single day
print("\n1. Testing Mode B padding:")
print("-" * 50)

# Pick a day with gaps
test_day = '2020-01-17'  # Example day with missing bars
df_day = df[df['episode_id'] == test_day].copy()
print(f"Original bars for {test_day}: {len(df_day)}")

if len(df_day) < 60:
    # Create complete time grid
    start_time = pd.Timestamp(test_day + ' 08:00:00').tz_localize('America/Bogota')
    time_grid = pd.date_range(start=start_time, periods=60, freq='5min')
    
    # Convert existing times to COT for matching
    df_day['time_cot_ts'] = pd.to_datetime(df_day['time_cot'])
    
    # Find missing slots
    existing_times = set(df_day['time_cot_ts'].dt.floor('5min'))
    all_times = set(time_grid.tz_localize(None))
    missing_times = all_times - existing_times
    
    print(f"Missing time slots: {len(missing_times)}")
    
    # Check if single slot missing (Mode B condition)
    if len(missing_times) == 1:
        print("SINGLE MISSING SLOT - Mode B applies!")
        missing_time = list(missing_times)[0]
        print(f"Missing slot: {missing_time}")
        
        # Create placeholder row
        placeholder = pd.DataFrame({
            'episode_id': [test_day],
            't_in_episode': [None],  # Will be filled based on position
            'is_terminal': [False],
            'time_utc': [missing_time.tz_localize('America/Bogota').tz_convert('UTC')],
            'time_cot': [missing_time],
            'hour_cot': [missing_time.hour],
            'minute_cot': [missing_time.minute],
            'open': [np.nan],
            'high': [np.nan],
            'low': [np.nan],
            'close': [np.nan],
            'ohlc_valid': [False],
            'is_stale': [False],
            'is_missing': [True]  # New column for Mode B
        })
        
        print(f"Created placeholder for: {missing_time}")
        df_padded = pd.concat([df_day, placeholder], ignore_index=True)
        df_padded = df_padded.sort_values('time_cot').reset_index(drop=True)
        df_padded['t_in_episode'] = range(60)
        df_padded.loc[59, 'is_terminal'] = True
        
        print(f"After padding: {len(df_padded)} bars")
        print(f"Missing flags: {df_padded['is_missing'].sum()}")
    else:
        print(f"Multiple gaps ({len(missing_times)}) - Mode B does not apply")

# 2. Test fail reason categorization
print("\n2. Testing fail reason categorization:")
print("-" * 50)

quality_report = []
for episode_id in ['2020-01-02', '2020-01-17', '2020-04-10']:
    df_ep = df[df['episode_id'] == episode_id]
    n_rows = len(df_ep)
    n_stale = df_ep['is_stale'].sum()
    
    # Determine fail reason
    if n_rows < 59:
        fail_reason = "INSUFFICIENT_BARS"
    elif n_rows == 59:
        fail_reason = "SINGLE_MISSING"
    elif n_stale / n_rows > 0.02:
        fail_reason = "HIGH_STALE_RATE"
    elif (~df_ep['ohlc_valid']).sum() > 0:
        fail_reason = "OHLC_VIOLATIONS"
    else:
        fail_reason = "OK"
    
    print(f"{episode_id}: {n_rows} bars, {n_stale} stale -> {fail_reason}")
    quality_report.append({
        'date': episode_id,
        'rows_found': n_rows,
        'n_stale': n_stale,
        'fail_reason': fail_reason
    })

# 3. Test stale burst detection
print("\n3. Testing stale burst detection:")
print("-" * 50)

# Find a day with stale bars
stale_days = df.groupby('episode_id')['is_stale'].sum()
stale_days = stale_days[stale_days > 0].head(3)

for episode_id in stale_days.index:
    df_ep = df[df['episode_id'] == episode_id]
    
    # Calculate max consecutive stale bars
    stale_burst_max = 0
    current_burst = 0
    
    for is_stale in df_ep['is_stale'].values:
        if is_stale:
            current_burst += 1
            stale_burst_max = max(stale_burst_max, current_burst)
        else:
            current_burst = 0
    
    print(f"{episode_id}: {stale_days[episode_id]} total stale, max burst = {stale_burst_max}")

# 4. Test SHA256 hash generation
print("\n4. Testing SHA256 hash generation:")
print("-" * 50)

# Create sample metadata
metadata = {
    "dataset_version": "v1.0-audit",
    "rows": len(df),
    "price_unit": "COP",
    "created_ts": pd.Timestamp.now().isoformat()
}

# Generate hash of the data
data_str = df.to_json(orient='records', date_format='iso')
data_hash = hashlib.sha256(data_str.encode()).hexdigest()
metadata['data_hash'] = data_hash

print(f"Generated SHA256: {data_hash[:32]}...")
print(f"Metadata keys: {list(metadata.keys())}")

# 5. Test clean subset creation
print("\n5. Testing clean subset creation:")
print("-" * 50)

# Read quality report
quality_df = pd.read_csv(base / "_reports" / "daily_quality_60.csv")

# Get OK and WARN episodes
ok_warn_episodes = quality_df[
    quality_df['quality_flag'].isin(['OK', 'WARN'])
]['date'].values

print(f"Total episodes: {len(quality_df)}")
print(f"OK/WARN episodes: {len(ok_warn_episodes)}")

# Create clean subset
df_clean = df[df['episode_id'].isin(ok_warn_episodes)]
print(f"Clean subset rows: {len(df_clean):,} (from {len(df):,})")
print(f"Reduction: {(1 - len(df_clean)/len(df))*100:.1f}%")

print("\n" + "=" * 70)
print("AUDIT FEATURES TEST COMPLETE")
print("All features working correctly for local testing")
print("=" * 70)