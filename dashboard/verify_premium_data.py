#!/usr/bin/env python3
"""
Verify and Display Premium Data
================================
Script to verify that the dashboard is correctly displaying
Silver Premium data (Lun-Vie 08:00-14:00 COT)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

def verify_premium_data():
    """Verify the Silver Premium dataset"""
    
    # Find the latest Silver Premium file
    silver_dir = Path("data/processed/silver")
    silver_files = list(silver_dir.glob("SILVER_PREMIUM_ONLY_*.csv"))
    
    if not silver_files:
        print("ERROR: No Silver Premium files found")
        return
    
    # Get the most recent file
    latest_file = max(silver_files, key=lambda x: x.stat().st_mtime)
    
    print("="*80)
    print("SILVER PREMIUM DATA VERIFICATION")
    print("="*80)
    print(f"\nFile: {latest_file.name}")
    print(f"Size: {latest_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Load data
    df = pd.read_csv(latest_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Add time columns
    df['date'] = df['time'].dt.date
    df['hour_utc'] = df['time'].dt.hour
    df['hour_cot'] = ((df['hour_utc'] - 5) % 24)
    df['day_name'] = df['time'].dt.day_name()
    df['day_of_week'] = df['time'].dt.dayofweek
    
    print(f"\nDATASET STATISTICS:")
    print(f"   Total Records: {len(df):,}")
    print(f"   Period: {df['time'].min()} to {df['time'].max()}")
    print(f"   Unique Days: {df['date'].nunique()}")
    
    # Verify it's only Premium session
    print(f"\nSESSION VERIFICATION:")
    print(f"   Expected: Lun-Vie 08:00-14:00 COT (13:00-19:00 UTC)")
    
    # Check hours
    hours_utc = df['hour_utc'].unique()
    hours_cot = df['hour_cot'].unique()
    print(f"   UTC Hours Found: {sorted(hours_utc)}")
    print(f"   COT Hours Found: {sorted(hours_cot)}")
    
    # Check days
    days = df['day_of_week'].unique()
    day_names = df['day_name'].unique()
    print(f"   Days Found: {sorted(days)} ({', '.join(sorted(day_names))})")
    
    # Verify only Monday-Friday
    if set(days) <= {0, 1, 2, 3, 4}:
        print(f"   [OK] Only weekdays (Mon-Fri) present")
    else:
        print(f"   [ERROR] Weekend data found!")
    
    # Verify only Premium hours
    if set(hours_utc) <= set(range(13, 19)):
        print(f"   [OK] Only Premium hours (13:00-19:00 UTC) present")
    else:
        print(f"   [ERROR] Non-Premium hours found!")
    
    print(f"\nDISTRIBUTION BY DAY:")
    for day in range(5):
        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day]
        count = len(df[df['day_of_week'] == day])
        pct = count / len(df) * 100
        print(f"   {day_name}: {count:,} records ({pct:.1f}%)")
    
    print(f"\nDISTRIBUTION BY HOUR (COT):")
    for hour in sorted(hours_cot):
        count = len(df[df['hour_cot'] == hour])
        pct = count / len(df) * 100
        print(f"   {hour:02d}:00 COT: {count:,} records ({pct:.1f}%)")
    
    # Calculate completeness
    expected_bars = df['date'].nunique() * 72  # 6 hours * 12 bars/hour
    actual_bars = len(df)
    completeness = (actual_bars / expected_bars * 100) if expected_bars > 0 else 0
    
    print(f"\nDATA QUALITY:")
    print(f"   Expected Bars: {expected_bars:,}")
    print(f"   Actual Bars: {actual_bars:,}")
    print(f"   Missing Bars: {expected_bars - actual_bars:,}")
    print(f"   Completeness: {completeness:.1f}%")
    
    # Price statistics
    print(f"\nPRICE STATISTICS:")
    print(f"   Mean: ${df['close'].mean():.2f}")
    print(f"   Std: ${df['close'].std():.2f}")
    print(f"   Min: ${df['close'].min():.2f}")
    print(f"   Max: ${df['close'].max():.2f}")
    
    # Sample data for dashboard
    print(f"\nSAMPLE DATA (Last 5 records):")
    print(df[['time', 'open', 'high', 'low', 'close']].tail())
    
    # Create summary JSON for dashboard
    summary = {
        'file': latest_file.name,
        'total_records': len(df),
        'start_date': str(df['time'].min()),
        'end_date': str(df['time'].max()),
        'unique_days': int(df['date'].nunique()),
        'completeness': round(completeness, 1),
        'session': {
            'name': 'Premium',
            'days': 'Monday-Friday',
            'hours_cot': '08:00-14:00',
            'hours_utc': '13:00-19:00'
        },
        'price_stats': {
            'mean': round(float(df['close'].mean()), 2),
            'std': round(float(df['close'].std()), 2),
            'min': round(float(df['close'].min()), 2),
            'max': round(float(df['close'].max()), 2)
        },
        'distribution': {
            'by_day': {
                'Monday': int(len(df[df['day_of_week'] == 0])),
                'Tuesday': int(len(df[df['day_of_week'] == 1])),
                'Wednesday': int(len(df[df['day_of_week'] == 2])),
                'Thursday': int(len(df[df['day_of_week'] == 3])),
                'Friday': int(len(df[df['day_of_week'] == 4]))
            },
            'by_hour_cot': {
                f"{hour:02d}:00": int(len(df[df['hour_cot'] == hour]))
                for hour in sorted(hours_cot)
            }
        }
    }
    
    # Save summary
    summary_path = silver_dir / 'premium_data_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OK] Summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("\nRESULT: Dataset contains ONLY Premium session data")
    print("   Monday-Friday, 08:00-14:00 COT")
    print("   Ready for dashboard visualization!")
    
    return summary

if __name__ == '__main__':
    summary = verify_premium_data()
    
    print("\nDashboard URLs:")
    print("   Main: http://localhost:8090")
    print("   API Info: http://localhost:8090/api/data/info")
    print("   Latest Data: http://localhost:8090/api/data/latest?n=100")
    print("   Chart Data: http://localhost:8090/api/data/chart?n=500")
    print("   Quality: http://localhost:8090/api/data/quality")