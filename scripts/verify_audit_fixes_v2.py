"""
Verification script for L4 Audit Fixes V2
Validates that all audit requirements are met
"""

from minio import Minio
import pandas as pd
import json
import io
import numpy as np
from datetime import datetime

# MinIO configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_SECURE = False

def get_minio_client():
    """Get MinIO client"""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )


def verify_audit_fixes():
    """Verify all audit fixes are properly implemented"""
    
    print("\n" + "="*80)
    print("L4 AUDIT FIXES V2 - VERIFICATION")
    print("="*80)
    
    client = get_minio_client()
    bucket = 'ds-usdcop-rlready'
    
    # Find latest run
    prefix = "usdcop_m5__05_l4_rlready/"
    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    
    # Get latest run
    runs = {}
    for obj in objects:
        if 'run_id=' in obj.object_name:
            parts = obj.object_name.split('run_id=')
            if len(parts) > 1:
                run_id = parts[1].split('/')[0]
                if run_id not in runs:
                    runs[run_id] = []
                runs[run_id].append(obj.object_name)
    
    if not runs:
        print("[ERROR] No runs found")
        return False
    
    # Use latest run
    latest_run = sorted(runs.keys())[-1]
    print(f"\n[INFO] Checking run: {latest_run}")
    
    # Build base path
    date_part = runs[latest_run][0].split('date=')[1].split('/')[0]
    base_path = f"usdcop_m5__05_l4_rlready/market=usdcop/timeframe=m5/date={date_part}/run_id={latest_run}"
    
    all_checks_passed = True
    
    # 1. CHECK PREMIUM WINDOW (CRITICAL)
    print("\n" + "-"*80)
    print("1. PREMIUM WINDOW VERIFICATION (08:00-12:55 COT)")
    print("-"*80)
    
    try:
        # Load replay dataset
        csv_key = f"{base_path}/replay_dataset.csv"
        obj = client.get_object(bucket, csv_key)
        csv_content = obj.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Check each episode
        episodes_check = []
        premium_window_pass = True
        
        for episode_id in df['episode_id'].unique():
            ep_df = df[df['episode_id'] == episode_id]
            
            # Check start time
            start_hour = ep_df.iloc[0]['hour_cot']
            start_minute = ep_df.iloc[0]['minute_cot']
            
            # Check end time
            end_hour = ep_df.iloc[-1]['hour_cot']
            end_minute = ep_df.iloc[-1]['minute_cot']
            
            # Validate
            is_valid = (start_hour == 8 and start_minute == 0 and 
                       end_hour == 12 and end_minute == 55)
            
            # Check episode_id format (should be YYYY-MM-DD)
            id_format_valid = len(episode_id) == 10 and episode_id[4] == '-' and episode_id[7] == '-'
            
            print(f"  Episode {episode_id}:")
            print(f"    Start: {start_hour:02d}:{start_minute:02d} COT")
            print(f"    End: {end_hour:02d}:{end_minute:02d} COT")
            print(f"    ID Format: {'YYYY-MM-DD' if id_format_valid else 'INVALID'}")
            print(f"    Status: [{'PASS' if is_valid and id_format_valid else 'FAIL'}]")
            
            if not (is_valid and id_format_valid):
                premium_window_pass = False
        
        print(f"\nPremium Window Check: [{'PASS' if premium_window_pass else 'FAIL'}]")
        if not premium_window_pass:
            all_checks_passed = False
            
    except Exception as e:
        print(f"[ERROR] Failed to verify premium window: {e}")
        all_checks_passed = False
    
    # 2. CHECK DATA TYPES
    print("\n" + "-"*80)
    print("2. DATA TYPES VERIFICATION")
    print("-"*80)
    
    try:
        dtype_checks = {
            'mid_t': str(df['mid_t'].dtype),
            'mid_t1': str(df['mid_t1'].dtype),
            'mid_t2': str(df['mid_t2'].dtype),
            'is_terminal': str(df['is_terminal'].dtype),
            'is_blocked': str(df['is_blocked'].dtype)
        }
        
        expected = {
            'mid_t': 'float64',
            'mid_t1': 'float64',
            'mid_t2': 'float64',
            'is_terminal': 'bool',
            'is_blocked': 'int64'
        }
        
        dtypes_pass = True
        for col, dtype in dtype_checks.items():
            expected_type = expected[col]
            is_correct = expected_type in dtype or dtype in expected_type
            
            print(f"  {col:<15} Expected: {expected_type:<10} Actual: {dtype:<10} [{'PASS' if is_correct else 'FAIL'}]")
            
            if not is_correct:
                dtypes_pass = False
        
        print(f"\nData Types Check: [{'PASS' if dtypes_pass else 'FAIL'}]")
        if not dtypes_pass:
            all_checks_passed = False
            
    except Exception as e:
        print(f"[ERROR] Failed to verify dtypes: {e}")
        all_checks_passed = False
    
    # 3. CHECK ENV_SPEC.JSON
    print("\n" + "-"*80)
    print("3. ENV_SPEC.JSON VERIFICATION")
    print("-"*80)
    
    try:
        json_key = f"{base_path}/env_spec.json"
        obj = client.get_object(bucket, json_key)
        env_spec = json.loads(obj.read())
        
        # Check for required fields
        has_premium_window = 'premium_window_cot' in env_spec
        has_obs_feature_list = 'obs_feature_list' in env_spec.get('observation', {})
        has_obs_name_map = 'obs_name_map' in env_spec.get('observation', {})
        
        print(f"  Has premium_window_cot: {has_premium_window}")
        if has_premium_window:
            pw = env_spec['premium_window_cot']
            print(f"    Start: {pw.get('start')}")
            print(f"    End: {pw.get('end')}")
            print(f"    Timezone: {pw.get('timezone')}")
        
        print(f"  Has obs_feature_list: {has_obs_feature_list}")
        if has_obs_feature_list:
            print(f"    Features: {len(env_spec['observation']['obs_feature_list'])}")
        
        print(f"  Has obs_name_map: {has_obs_name_map}")
        if has_obs_name_map:
            print(f"    Mappings: {len(env_spec['observation']['obs_name_map'])}")
        
        env_spec_pass = has_premium_window and has_obs_feature_list and has_obs_name_map
        print(f"\nEnv Spec Check: [{'PASS' if env_spec_pass else 'FAIL'}]")
        if not env_spec_pass:
            all_checks_passed = False
            
    except Exception as e:
        print(f"[ERROR] Failed to verify env_spec: {e}")
        all_checks_passed = False
    
    # 4. CHECK EPISODES INDEX
    print("\n" + "-"*80)
    print("4. EPISODES INDEX VERIFICATION")
    print("-"*80)
    
    try:
        csv_key = f"{base_path}/episodes_index.csv"
        obj = client.get_object(bucket, csv_key)
        csv_content = obj.read().decode('utf-8')
        episodes_df = pd.read_csv(io.StringIO(csv_content))
        
        # Check for required columns
        required_cols = [
            'start_time_utc', 'end_time_utc',
            'start_time_cot', 'end_time_cot',
            'quality_flag', 'blocked_rate',
            'premium_session'
        ]
        
        episodes_index_pass = True
        for col in required_cols:
            has_col = col in episodes_df.columns
            print(f"  Has {col}: {has_col}")
            if not has_col:
                episodes_index_pass = False
        
        # Check quality flags
        if 'quality_flag' in episodes_df.columns:
            quality_counts = episodes_df['quality_flag'].value_counts()
            print(f"\n  Quality Flags:")
            for flag, count in quality_counts.items():
                print(f"    {flag}: {count}")
        
        print(f"\nEpisodes Index Check: [{'PASS' if episodes_index_pass else 'FAIL'}]")
        if not episodes_index_pass:
            all_checks_passed = False
            
    except Exception as e:
        print(f"[ERROR] Failed to verify episodes_index: {e}")
        all_checks_passed = False
    
    # 5. CHECK VOLUME GATES
    print("\n" + "-"*80)
    print("5. VOLUME GATES VERIFICATION")
    print("-"*80)
    
    try:
        json_key = f"{base_path}/checks_report.json"
        obj = client.get_object(bucket, json_key)
        checks_report = json.loads(obj.read())
        
        num_episodes = checks_report['statistics']['num_episodes']
        status = checks_report['status']
        sample_mode = checks_report.get('sample_mode', False)
        
        print(f"  Episodes: {num_episodes}")
        print(f"  Status: {status}")
        print(f"  Sample Mode: {sample_mode}")
        
        if 'gates' in checks_report:
            print(f"\n  Gates:")
            for gate_name, gate_info in checks_report['gates'].items():
                if isinstance(gate_info, dict) and 'pass' in gate_info:
                    print(f"    {gate_name}: [{'PASS' if gate_info['pass'] else 'FAIL'}]")
        
        # Check logic
        volume_gate_pass = True
        if num_episodes < 10 and status != 'FAIL':
            print(f"  [WARNING] Less than 10 episodes but status is {status}")
            volume_gate_pass = False
        elif num_episodes < 500 and status not in ['FAIL', 'SAMPLE_ONLY']:
            print(f"  [WARNING] Less than 500 episodes but status is {status}")
            volume_gate_pass = False
        
        print(f"\nVolume Gates Check: [{'PASS' if volume_gate_pass else 'FAIL'}]")
        if not volume_gate_pass:
            all_checks_passed = False
            
    except Exception as e:
        print(f"[ERROR] Failed to verify volume gates: {e}")
        all_checks_passed = False
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("AUDIT COMPLIANCE SUMMARY")
    print("="*80)
    
    if all_checks_passed:
        print("\n[SUCCESS] ALL AUDIT REQUIREMENTS MET!")
        print("\nKey Achievements:")
        print("  [OK] Premium window: 08:00-12:55 COT exactly")
        print("  [OK] Episode IDs: YYYY-MM-DD format")
        print("  [OK] Data types: mid_t* as float64")
        print("  [OK] Env spec: includes premium_window_cot and obs_name_map")
        print("  [OK] Episodes index: has timezone columns and quality flags")
        print("  [OK] Volume gates: proper status based on episode count")
        print("\n[READY] Dataset ready for auditor review!")
    else:
        print("\n[WARNING] Some checks failed - review details above")
    
    return all_checks_passed


if __name__ == "__main__":
    verify_audit_fixes()