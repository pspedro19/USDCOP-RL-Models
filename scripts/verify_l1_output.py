#!/usr/bin/env python3
"""Verify L1 output meets all requirements"""

import pandas as pd
import json
from pathlib import Path

# Read files
base = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_consolidated")

# 1. Check standardized_data
df = pd.read_parquet(base / "standardized_data.parquet")
print("=" * 70)
print("1. standardized_data.parquet & .csv")
print("=" * 70)
print(f"  Rows: {len(df):,}")
print(f"  Columns: {df.shape[1]} (required: 13)")
print(f"  Column names: {list(df.columns)}")
print(f"  time_utc unique: {df['time_utc'].nunique() == len(df)}")
print(f"  (episode_id, t_in_episode) unique: {df[['episode_id', 't_in_episode']].drop_duplicates().shape[0] == len(df)}")

# Sample data
print(f"\n  Sample episode_id values: {df['episode_id'].unique()[:3].tolist()}")
print(f"  Sample t_in_episode range: {df['t_in_episode'].min()}-{df['t_in_episode'].max()}")
print(f"  Sample time_cot: {df['time_cot'].iloc[0]}")

# 2. Check daily_quality_60
quality = pd.read_csv(base / "_reports" / "daily_quality_60.csv")
print("\n" + "=" * 70)
print("2. _reports/daily_quality_60.csv")
print("=" * 70)
print(f"  Rows: {len(quality)} (one per day)")
print(f"  Columns: {quality.shape[1]} (required: 10)")
print(f"  Column names: {list(quality.columns)}")

# Quality summary
ok = len(quality[quality['quality_flag'] == 'OK'])
warn = len(quality[quality['quality_flag'] == 'WARN'])
fail = len(quality[quality['quality_flag'] == 'FAIL'])
print(f"\n  Quality summary:")
print(f"    OK: {ok} days ({ok/len(quality)*100:.1f}%)")
print(f"    WARN: {warn} days ({warn/len(quality)*100:.1f}%)")
print(f"    FAIL: {fail} days ({fail/len(quality)*100:.1f}%)")

# 3. Check _metadata.json
with open(base / "_metadata.json") as f:
    meta = json.load(f)

print("\n" + "=" * 70)
print("3. _metadata.json")
print("=" * 70)
print(f"  Keys present: {list(meta.keys())}")

required_keys = [
    "dataset_version", "run_id", "date_cot", "utc_window",
    "rows", "price_unit", "price_precision", "source", "created_ts"
]

missing = [k for k in required_keys if k not in meta]
if missing:
    print(f"  MISSING REQUIRED KEYS: {missing}")
else:
    print("  ✓ All 9 required keys present")

print(f"\n  Dataset info:")
print(f"    Version: {meta.get('dataset_version')}")
print(f"    Rows: {meta.get('rows'):,}")
print(f"    Source: {meta.get('source')}")
print(f"    Price unit: {meta.get('price_unit')}")
print(f"    Price precision: {meta.get('price_precision')}")
print(f"    UTC window: {meta.get('utc_window')}")

# 4. File count
print("\n" + "=" * 70)
print("4. FILE COUNT")
print("=" * 70)
files = [
    "standardized_data.parquet",
    "standardized_data.csv",
    "_reports/daily_quality_60.csv",
    "_metadata.json"
]
print(f"  Expected files: 4")
print(f"  Files present:")
for f in files:
    exists = (base / f).exists()
    status = "✓" if exists else "✗"
    print(f"    {status} {f}")

# Final check
print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

checks = [
    ("File count = 4", len(files) == 4),
    ("standardized_data has 13 columns", df.shape[1] == 13),
    ("daily_quality_60 has 10 columns", quality.shape[1] == 10),
    ("_metadata.json has 9+ keys", len([k for k in required_keys if k in meta]) == 9),
    ("Rows match in metadata", meta.get('rows') == len(df))
]

all_pass = True
for check, result in checks:
    status = "✓" if result else "✗"
    print(f"  {status} {check}")
    if not result:
        all_pass = False

print("\n" + "=" * 70)
if all_pass:
    print("✓ ALL REQUIREMENTS MET - L1 DATASET IS READY")
else:
    print("✗ Some requirements not met - please review")
print("=" * 70)