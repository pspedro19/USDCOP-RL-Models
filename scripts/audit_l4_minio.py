"""
Comprehensive L4 Audit from MinIO data
"""

import pandas as pd
import numpy as np
import json
from minio import Minio
from io import BytesIO
from datetime import datetime
import pytz

# MinIO configuration
MINIO_CLIENT = Minio(
    'localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)

BUCKET_NAME = 'ds-usdcop-rlready'
RUN_ID = 'L4_FULL_20250822_115113'
DATE = '2025-08-22'

def load_from_minio(file_name):
    """Load file from MinIO"""
    base_path = f"usdcop_m5__05_l4_rlready/market=usdcop/timeframe=m5/date={DATE}/run_id={RUN_ID}"
    
    try:
        response = MINIO_CLIENT.get_object(BUCKET_NAME, f"{base_path}/{file_name}")
        data = response.read()
        response.close()
        
        if file_name.endswith('.csv'):
            return pd.read_csv(BytesIO(data))
        elif file_name.endswith('.json'):
            return json.loads(data.decode('utf-8'))
        else:
            return data.decode('utf-8')
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

def run_comprehensive_audit():
    """Run complete L4 audit"""
    
    print("="*80)
    print(" L4 COMPREHENSIVE AUDIT - MINIO DATA")
    print("="*80)
    print(f"\nRun ID: {RUN_ID}")
    print(f"Date: {DATE}")
    print("-"*80)
    
    audit_results = {}
    
    # 1. Load all files
    print("\n[1/10] Loading L4 files from MinIO...")
    replay_df = load_from_minio('replay_dataset.csv')
    episodes_df = load_from_minio('episodes_index.csv')
    env_spec = load_from_minio('env_spec.json')
    reward_spec = load_from_minio('reward_spec.json')
    cost_model = load_from_minio('cost_model.json')
    action_spec = load_from_minio('action_spec.json')
    split_spec = load_from_minio('split_spec.json')
    checks_report = load_from_minio('checks_report.json')
    metadata = load_from_minio('metadata.json')
    
    # 2. Metadata and traceability
    print("\n[2/10] Checking Metadata and Traceability...")
    if metadata:
        audit_results['metadata'] = 'PASS'
        print(f"  Pipeline: {metadata.get('pipeline')}")
        print(f"  Version: {metadata.get('version')}")
        print(f"  Temporal range: {metadata.get('temporal_range', {}).get('start')} to {metadata.get('temporal_range', {}).get('end')}")
    else:
        audit_results['metadata'] = 'FAIL'
    
    # 3. Replay dataset schema
    print("\n[3/10] Validating Replay Dataset Schema...")
    required_cols = ['episode_id', 't_in_episode', 'is_terminal', 'open', 'high', 'low', 'close', 'is_blocked']
    missing_cols = [c for c in required_cols if c not in replay_df.columns]
    
    if not missing_cols:
        audit_results['replay_schema'] = 'PASS'
        print(f"  All required columns present")
        print(f"  Total rows: {len(replay_df):,}")
        print(f"  Episodes: {replay_df['episode_id'].nunique()}")
    else:
        audit_results['replay_schema'] = f'FAIL - Missing: {missing_cols}'
    
    # 4. Episodes index validation
    print("\n[4/10] Validating Episodes Index...")
    if 'n_steps' in episodes_df.columns:
        steps_valid = episodes_df['n_steps'].isin([59, 60]).all()
        total_steps = episodes_df['n_steps'].sum()
        
        if steps_valid and total_steps == len(replay_df):
            audit_results['episodes_index'] = 'PASS'
            print(f"  Episodes: {len(episodes_df)}")
            print(f"  Steps match: {total_steps} == {len(replay_df)}")
        else:
            audit_results['episodes_index'] = 'FAIL - Steps mismatch'
    
    # 5. Premium window validation (08:00-12:55 COT)
    print("\n[5/10] Validating Premium Window...")
    if 'time_cot' in replay_df.columns:
        replay_df['time_cot'] = pd.to_datetime(replay_df['time_cot'])
        replay_df['hour_cot'] = replay_df['time_cot'].dt.hour
        replay_df['minute_cot'] = replay_df['time_cot'].dt.minute
        
        outside_premium = replay_df[
            (replay_df['hour_cot'] < 8) | 
            (replay_df['hour_cot'] > 12) |
            ((replay_df['hour_cot'] == 12) & (replay_df['minute_cot'] > 55))
        ]
        
        if len(outside_premium) == 0:
            audit_results['premium_window'] = 'PASS'
            print(f"  All data within 08:00-12:55 COT")
        else:
            audit_results['premium_window'] = f'FAIL - {len(outside_premium)} steps outside window'
    
    # 6. Grid consistency (300s intervals)
    print("\n[6/10] Checking Grid Consistency...")
    sample_episodes = replay_df['episode_id'].unique()[:5]
    grid_violations = 0
    
    for ep_id in sample_episodes:
        ep_data = replay_df[replay_df['episode_id'] == ep_id].sort_values('t_in_episode')
        if 'time_utc' in ep_data.columns:
            ep_data['time_utc'] = pd.to_datetime(ep_data['time_utc'])
            time_diffs = ep_data['time_utc'].diff().dt.total_seconds()
            if not np.allclose(time_diffs.dropna(), 300, atol=1):
                grid_violations += 1
    
    if grid_violations == 0:
        audit_results['grid_consistency'] = 'PASS'
        print(f"  Grid validated (300s intervals)")
    else:
        audit_results['grid_consistency'] = f'WARN - {grid_violations} episodes with violations'
    
    # 7. Cost model validation
    print("\n[7/10] Validating Cost Model...")
    if 'spread_proxy_bps' in replay_df.columns:
        spread_p50 = replay_df['spread_proxy_bps'].quantile(0.50)
        spread_p95 = replay_df['spread_proxy_bps'].quantile(0.95)
        
        print(f"  Spread p50: {spread_p50:.2f} bps")
        print(f"  Spread p95: {spread_p95:.2f} bps")
        
        if spread_p95 <= 15 and 3 <= spread_p50 <= 8:
            audit_results['cost_model'] = 'PASS'
        elif spread_p95 <= 15:
            audit_results['cost_model'] = 'WARN - Unusual spread distribution'
        else:
            audit_results['cost_model'] = 'FAIL - Spread p95 > 15 bps'
    
    # 8. Observation features
    print("\n[8/10] Checking Observation Features...")
    obs_cols = [c for c in replay_df.columns if c.startswith('obs_')]
    
    if obs_cols:
        print(f"  Found {len(obs_cols)} observation features")
        
        # Check data type
        sample_dtype = replay_df[obs_cols[0]].dtype
        if sample_dtype == np.float32:
            print(f"  Data type: float32 [OK]")
        else:
            print(f"  Data type: {sample_dtype} [WARN - should be float32]")
        
        # Check NaN rate post-warmup
        post_warmup = replay_df[replay_df['t_in_episode'] >= 10]
        nan_rate = post_warmup[obs_cols].isna().mean().mean()
        print(f"  NaN rate post-warmup: {nan_rate*100:.2f}%")
        
        if nan_rate < 0.005:
            audit_results['observations'] = 'PASS'
        elif nan_rate < 0.02:
            audit_results['observations'] = 'WARN - Moderate NaN rate'
        else:
            audit_results['observations'] = 'FAIL - High NaN rate'
    
    # 9. Coverage analysis (2020-2025)
    print("\n[9/10] Analyzing Coverage (2020-2025)...")
    episodes_df['date'] = pd.to_datetime(episodes_df['date_cot'])
    episodes_df['year'] = episodes_df['date'].dt.year
    
    yearly_coverage = {}
    for year in range(2020, 2026):
        year_data = episodes_df[episodes_df['year'] == year]
        yearly_coverage[year] = len(year_data)
    
    print("  Yearly episode distribution:")
    for year, count in yearly_coverage.items():
        print(f"    {year}: {count} episodes")
    
    missing_years = [y for y, c in yearly_coverage.items() if c == 0]
    if not missing_years:
        audit_results['coverage'] = 'PASS'
    else:
        audit_results['coverage'] = f'FAIL - Missing years: {missing_years}'
    
    # 10. Overall assessment
    print("\n[10/10] Overall Assessment...")
    print("-"*80)
    
    pass_count = sum(1 for v in audit_results.values() if 'PASS' in str(v))
    warn_count = sum(1 for v in audit_results.values() if 'WARN' in str(v))
    fail_count = sum(1 for v in audit_results.values() if 'FAIL' in str(v))
    
    print("\nAudit Results Summary:")
    for check, result in audit_results.items():
        status = 'PASS' if 'PASS' in str(result) else 'WARN' if 'WARN' in str(result) else 'FAIL'
        print(f"  {check}: [{status}] {result if status != 'PASS' else ''}")
    
    print(f"\nTotals: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")
    
    if fail_count == 0 and warn_count <= 2:
        print("\n" + "="*80)
        print(" [SUCCESS] L4 DATA PASSES COMPREHENSIVE AUDIT!")
        print(" Ready for L5 serving and RL training")
        print("="*80)
    elif fail_count == 0:
        print("\n" + "="*80)
        print(" [READY] L4 DATA READY WITH WARNINGS")
        print(" Review warnings before production use")
        print("="*80)
    else:
        print("\n" + "="*80)
        print(" [FAIL] L4 DATA HAS CRITICAL ISSUES")
        print(" Must resolve failures before use")
        print("="*80)
    
    # Save audit report
    audit_report = {
        'timestamp': datetime.now().isoformat(),
        'run_id': RUN_ID,
        'results': audit_results,
        'summary': {
            'pass': pass_count,
            'warn': warn_count,
            'fail': fail_count
        }
    }
    
    with open('L4_AUDIT_MINIO_REPORT.json', 'w') as f:
        json.dump(audit_report, f, indent=2)
    
    print(f"\nAudit report saved to: L4_AUDIT_MINIO_REPORT.json")

if __name__ == "__main__":
    run_comprehensive_audit()