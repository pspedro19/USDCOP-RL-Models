"""
L4 Comprehensive Audit Script
Verifies all aspects of L4 RL-Ready package
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import pytz
from minio import Minio
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# MinIO configuration
MINIO_CLIENT = Minio(
    'localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)

BUCKET_NAME = 'ds-usdcop-rlready'
PREFIX = 'usdcop_m5__05_l4_rlready/market=usdcop/timeframe=m5'

def get_latest_run():
    """Get the latest run from MinIO"""
    # List all objects to find dates
    objects = list(MINIO_CLIENT.list_objects(BUCKET_NAME, prefix=PREFIX, recursive=True))
    
    if not objects:
        raise Exception("No objects found in MinIO")
    
    # Extract unique date/run combinations
    runs = set()
    for obj in objects:
        parts = obj.object_name.split('/')
        date_part = None
        run_part = None
        for part in parts:
            if part.startswith('date='):
                date_part = part.replace('date=', '')
            elif part.startswith('run_id='):
                run_part = part.replace('run_id=', '')
        
        if date_part and run_part:
            runs.add((date_part, run_part))
    
    if not runs:
        raise Exception("No valid runs found")
    
    # Get the latest
    latest = sorted(runs)[-1]
    return latest[0], latest[1]

def load_file_from_minio(file_name, date, run_id):
    """Load a file from MinIO"""
    path = f"{PREFIX}/date={date}/run_id={run_id}/{file_name}"
    
    try:
        response = MINIO_CLIENT.get_object(BUCKET_NAME, path)
        data = response.read()
        response.close()
        response.release_conn()
        
        if file_name.endswith('.csv'):
            return pd.read_csv(BytesIO(data))
        elif file_name.endswith('.parquet'):
            return pd.read_parquet(BytesIO(data))
        elif file_name.endswith('.json'):
            return json.loads(data.decode('utf-8'))
        else:
            return data.decode('utf-8')
    except Exception as e:
        print(f"[WARNING] Could not load {file_name}: {e}")
        return None

def verify_premium_window(df):
    """Verify premium window 08:00-12:55 COT"""
    print("\n" + "="*80)
    print("1. PREMIUM WINDOW VERIFICATION (08:00-12:55 COT)")
    print("="*80)
    
    results = {
        'status': 'PASS',
        'episodes_checked': 0,
        'violations': []
    }
    
    # Convert timestamp to COT - handle both 'timestamp' and 'time_utc' column names
    time_col = 'timestamp' if 'timestamp' in df.columns else 'time_utc'
    df[time_col] = pd.to_datetime(df[time_col])
    utc_tz = pytz.UTC
    cot_tz = pytz.timezone('America/Bogota')
    
    for episode_id in df['episode_id'].unique()[:10]:  # Check first 10 episodes
        ep_data = df[df['episode_id'] == episode_id].copy()
        
        # Convert to COT
        ep_data['time_cot'] = ep_data[time_col].apply(
            lambda x: x.replace(tzinfo=utc_tz).astimezone(cot_tz) if pd.notna(x) else None
        )
        
        if len(ep_data) > 0:
            start_time_cot = ep_data.iloc[0]['time_cot']
            end_time_cot = ep_data.iloc[-1]['time_cot']
            
            start_hour = start_time_cot.hour
            start_minute = start_time_cot.minute
            end_hour = end_time_cot.hour
            end_minute = end_time_cot.minute
            
            print(f"  Episode {episode_id}: {start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d} COT", end='')
            
            if start_hour == 8 and start_minute == 0 and end_hour == 12 and end_minute == 55:
                print(" [PASS]")
            else:
                print(" [FAIL] - Expected 08:00-12:55")
                results['violations'].append(episode_id)
                results['status'] = 'FAIL'
            
            results['episodes_checked'] += 1
    
    print(f"\nSummary: {results['status']}")
    if results['violations']:
        print(f"Violations: {results['violations']}")
    
    return results

def verify_observation_dimensions(df, env_spec):
    """Verify observation dimensions match env_spec"""
    print("\n" + "="*80)
    print("2. OBSERVATION DIMENSIONS VERIFICATION")
    print("="*80)
    
    obs_cols = [c for c in df.columns if c.startswith('obs_')]
    expected_dim = env_spec.get('observation_dim', 17)
    
    print(f"  Expected dimension: {expected_dim}")
    print(f"  Actual obs columns: {len(obs_cols)}")
    print(f"  Columns: {obs_cols[:5]}..." if len(obs_cols) > 5 else f"  Columns: {obs_cols}")
    
    status = 'PASS' if len(obs_cols) == expected_dim else 'FAIL'
    print(f"\nStatus: {status}")
    
    return {'status': status, 'expected': expected_dim, 'actual': len(obs_cols)}

def verify_anti_leak(df):
    """Check for data leakage"""
    print("\n" + "="*80)
    print("3. ANTI-LEAK VERIFICATION")
    print("="*80)
    
    obs_cols = [c for c in df.columns if c.startswith('obs_')]
    
    # Calculate future returns
    df['future_return'] = df.groupby('episode_id')['mid_t2'].shift(-1) / df['mid_t'] - 1
    
    # Calculate correlations
    max_corr = 0
    problematic_features = []
    
    for col in obs_cols:
        if col in df.columns and 'future_return' in df.columns:
            corr = abs(df[col].corr(df['future_return']))
            if not pd.isna(corr):
                if corr > 0.10:
                    problematic_features.append((col, corr))
                max_corr = max(max_corr, corr)
    
    print(f"  Max correlation obs_* vs future return: {max_corr:.4f}")
    print(f"  Threshold: 0.10")
    
    if problematic_features:
        print(f"\n  [WARNING] High correlation features:")
        for feat, corr in problematic_features[:5]:
            print(f"    {feat}: {corr:.4f}")
    
    status = 'PASS' if max_corr < 0.10 else 'FAIL'
    print(f"\nStatus: {status}")
    
    return {'status': status, 'max_correlation': max_corr}

def verify_data_quality(df):
    """Verify data quality"""
    print("\n" + "="*80)
    print("4. DATA QUALITY VERIFICATION")
    print("="*80)
    
    results = {}
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['episode_id', 't_in_episode']).sum()
    print(f"  Duplicate (episode_id, t_in_episode): {duplicates}")
    results['duplicates'] = duplicates
    
    # Check NaNs in observations
    obs_cols = [c for c in df.columns if c.startswith('obs_')]
    nan_count = df[obs_cols].isna().sum().sum()
    print(f"  NaNs in obs_* columns: {nan_count}")
    results['nans'] = nan_count
    
    # Check blocked rate
    blocked_rate = df['is_blocked'].mean() * 100
    print(f"  Blocked rate: {blocked_rate:.2f}%")
    results['blocked_rate'] = blocked_rate
    
    # Check terminals
    terminals_correct = all(
        df[df['episode_id'] == ep_id]['is_terminal'].iloc[-1] == True
        for ep_id in df['episode_id'].unique()
    )
    print(f"  Terminals at t=59: {'PASS' if terminals_correct else 'FAIL'}")
    results['terminals_correct'] = terminals_correct
    
    status = 'PASS' if (duplicates == 0 and nan_count == 0 and terminals_correct) else 'FAIL'
    print(f"\nStatus: {status}")
    results['status'] = status
    
    return results

def verify_json_specs(env_spec, cost_model, checks_report):
    """Verify JSON specification files"""
    print("\n" + "="*80)
    print("5. JSON SPECIFICATIONS VERIFICATION")
    print("="*80)
    
    results = {}
    
    # env_spec checks
    print("\n  env_spec.json:")
    has_obs_list = 'obs_feature_list' in env_spec
    has_premium = 'premium_window_cot' in env_spec
    print(f"    Has obs_feature_list: {has_obs_list}")
    print(f"    Has premium_window_cot: {has_premium}")
    
    # cost_model checks  
    print("\n  cost_model.json:")
    if cost_model and 'statistics' in cost_model:
        spread_p95 = cost_model['statistics'].get('spread_p95_bps', 0)
        print(f"    Spread p95: {spread_p95:.2f} bps")
        spread_ok = 2 <= spread_p95 <= 15
        print(f"    Spread in range [2,15]: {spread_ok}")
    else:
        spread_ok = False
        print("    No statistics found")
    
    # checks_report
    print("\n  checks_report.json:")
    if checks_report:
        sample_mode = checks_report.get('sample_mode', False)
        has_gates = 'gates' in checks_report
        print(f"    Sample mode: {sample_mode}")
        print(f"    Has gates: {has_gates}")
    
    status = 'PASS' if (has_obs_list and has_premium and spread_ok) else 'FAIL'
    print(f"\nStatus: {status}")
    
    return {'status': status}

def main():
    print("="*80)
    print("L4 RL-READY COMPREHENSIVE AUDIT")
    print("="*80)
    
    # Get latest run
    try:
        date, run_id = get_latest_run()
        print(f"\nAnalyzing run: {run_id}")
        print(f"Date: {date}")
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Load files
    print("\nLoading files from MinIO...")
    replay_df = load_file_from_minio('replay_dataset.csv', date, run_id)
    episodes_df = load_file_from_minio('episodes_index.csv', date, run_id)
    env_spec = load_file_from_minio('env_spec.json', date, run_id)
    cost_model = load_file_from_minio('cost_model.json', date, run_id)
    checks_report = load_file_from_minio('checks_report.json', date, run_id)
    
    if replay_df is None or env_spec is None:
        print("[ERROR] Could not load required files")
        return
    
    # Run verifications
    results = {}
    
    # 1. Premium window
    results['premium_window'] = verify_premium_window(replay_df)
    
    # 2. Observation dimensions
    results['observation_dims'] = verify_observation_dimensions(replay_df, env_spec)
    
    # 3. Anti-leak
    results['anti_leak'] = verify_anti_leak(replay_df)
    
    # 4. Data quality
    results['data_quality'] = verify_data_quality(replay_df)
    
    # 5. JSON specs
    results['json_specs'] = verify_json_specs(env_spec, cost_model, checks_report)
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL AUDIT VERDICT")
    print("="*80)
    
    all_pass = all(r.get('status') == 'PASS' for r in results.values())
    
    for check, result in results.items():
        status = result.get('status', 'UNKNOWN')
        symbol = '✓' if status == 'PASS' else '✗'
        print(f"  {check}: [{status}]")
    
    print("\n" + "="*80)
    if all_pass:
        print("OVERALL: PASS - L4 package is ready for RL training")
    else:
        print("OVERALL: FAIL - Issues found, see details above")
    print("="*80)
    
    # Save results
    with open('l4_audit_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to l4_audit_results.json")

if __name__ == "__main__":
    main()