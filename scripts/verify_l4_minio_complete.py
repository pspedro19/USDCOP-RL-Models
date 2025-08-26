"""
Complete verification script for L4 RL-Ready outputs in MinIO
Checks all files, validates content, and confirms pipeline execution
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from minio import Minio
from minio.error import S3Error
import io
import pyarrow.parquet as pq

# MinIO configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_SECURE = False


def connect_minio():
    """Connect to MinIO"""
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        print("[OK] Connected to MinIO")
        return client
    except Exception as e:
        print(f"[ERROR] Failed to connect to MinIO: {e}")
        return None


def check_l4_outputs(client):
    """Check all L4 outputs in MinIO"""
    bucket = 'ds-usdcop-rlready'
    
    print(f"\n{'='*60}")
    print("CHECKING L4 RL-READY OUTPUTS IN MINIO")
    print(f"{'='*60}")
    
    # Check if bucket exists
    if not client.bucket_exists(bucket):
        print(f"[ERROR] Bucket {bucket} does not exist!")
        return False
    
    print(f"[OK] Bucket {bucket} exists")
    
    # List all L4 outputs
    print(f"\n[INFO] Searching for L4 outputs...")
    
    # Find the latest run
    prefix = "usdcop_m5__05_l4_rlready/"
    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    
    if not objects:
        print(f"[WARNING] No L4 outputs found in {bucket}/{prefix}")
        return False
    
    # Group by run_id
    runs = {}
    for obj in objects:
        parts = obj.object_name.split('/')
        # Find run_id in path
        for i, part in enumerate(parts):
            if part.startswith('run_id='):
                run_id = part.replace('run_id=', '')
                if run_id not in runs:
                    runs[run_id] = {
                        'files': [],
                        'path': '/'.join(parts[:i+1])
                    }
                runs[run_id]['files'].append(obj.object_name)
                break
    
    if not runs:
        print("[WARNING] No complete runs found")
        return False
    
    print(f"[OK] Found {len(runs)} run(s)")
    
    # Check the latest run
    latest_run = sorted(runs.keys())[-1]
    run_info = runs[latest_run]
    
    print(f"\n[INFO] Checking latest run: {latest_run}")
    print(f"[INFO] Path: {run_info['path']}")
    print(f"[INFO] Total files: {len(run_info['files'])}")
    
    # Required files
    required_files = [
        'replay_dataset.parquet',
        'episodes_index.parquet',
        'env_spec.json',
        'cost_model.json',
        'reward_spec.json',
        'action_spec.json',
        'split_spec.json',
        'checks_report.json',
        '_metadata/metadata.json'
    ]
    
    # Check each required file
    print(f"\n[INFO] Verifying required files:")
    missing_files = []
    found_files = {}
    
    for req_file in required_files:
        found = False
        for file_path in run_info['files']:
            if file_path.endswith(req_file):
                found = True
                found_files[req_file] = file_path
                
                # Get file info
                stat = client.stat_object(bucket, file_path)
                size_kb = stat.size / 1024
                
                print(f"  [OK] {req_file:<30} ({size_kb:.2f} KB)")
                break
        
        if not found:
            missing_files.append(req_file)
            print(f"  [MISSING] {req_file}")
    
    # Check for READY flag
    ready_found = False
    for file_path in run_info['files']:
        if file_path.endswith('_control/READY'):
            ready_found = True
            print(f"  [OK] {'_control/READY':<30} (Pipeline completed)")
            break
    
    if not ready_found:
        print(f"  [WARNING] _control/READY not found (Pipeline may not have completed)")
    
    if missing_files:
        print(f"\n[ERROR] Missing {len(missing_files)} required files")
        return False
    
    # Validate content of key files
    print(f"\n[INFO] Validating file contents:")
    
    # 1. Check replay_dataset.parquet
    if 'replay_dataset.parquet' in found_files:
        try:
            obj = client.get_object(bucket, found_files['replay_dataset.parquet'])
            content = obj.read()
            df = pd.read_parquet(io.BytesIO(content))
            
            print(f"\n  [REPLAY DATASET]")
            print(f"    - Rows: {len(df)}")
            print(f"    - Columns: {len(df.columns)}")
            
            # Check required columns
            required_cols = ['episode_id', 't_in_episode', 'is_terminal', 'time_utc', 
                           'open', 'high', 'low', 'close', 'is_blocked']
            missing_cols = set(required_cols) - set(df.columns)
            
            if missing_cols:
                print(f"    [WARNING] Missing columns: {missing_cols}")
            else:
                print(f"    [OK] All required columns present")
            
            # Check observation columns
            obs_cols = [col for col in df.columns if col.startswith('obs_')]
            print(f"    - Observation features: {len(obs_cols)}")
            
            # Check episodes
            n_episodes = df['episode_id'].nunique() if 'episode_id' in df.columns else 0
            print(f"    - Episodes: {n_episodes}")
            
        except Exception as e:
            print(f"    [ERROR] Failed to read replay_dataset: {e}")
    
    # 2. Check episodes_index.parquet
    if 'episodes_index.parquet' in found_files:
        try:
            obj = client.get_object(bucket, found_files['episodes_index.parquet'])
            content = obj.read()
            df_episodes = pd.read_parquet(io.BytesIO(content))
            
            print(f"\n  [EPISODES INDEX]")
            print(f"    - Total episodes: {len(df_episodes)}")
            
            if 'quality_flag' in df_episodes.columns:
                quality_dist = df_episodes['quality_flag'].value_counts().to_dict()
                print(f"    - Quality distribution:")
                for flag, count in quality_dist.items():
                    print(f"        {flag}: {count}")
            
            if 'blocked_rate' in df_episodes.columns:
                avg_blocked = df_episodes['blocked_rate'].mean() * 100
                print(f"    - Avg blocked rate: {avg_blocked:.2f}%")
            
        except Exception as e:
            print(f"    [ERROR] Failed to read episodes_index: {e}")
    
    # 3. Check env_spec.json
    if 'env_spec.json' in found_files:
        try:
            obj = client.get_object(bucket, found_files['env_spec.json'])
            content = obj.read()
            env_spec = json.loads(content)
            
            print(f"\n  [ENV SPEC]")
            print(f"    - Framework: {env_spec.get('framework', 'N/A')}")
            print(f"    - Observation dim: {env_spec.get('observation_dim', 'N/A')}")
            print(f"    - Action space: {env_spec.get('action_space', 'N/A')}")
            print(f"    - Seed: {env_spec.get('seed', 'N/A')}")
            
            if 'feature_list_order' in env_spec:
                print(f"    - Features: {len(env_spec['feature_list_order'])}")
            
        except Exception as e:
            print(f"    [ERROR] Failed to read env_spec: {e}")
    
    # 4. Check cost_model.json
    if 'cost_model.json' in found_files:
        try:
            obj = client.get_object(bucket, found_files['cost_model.json'])
            content = obj.read()
            cost_model = json.loads(content)
            
            print(f"\n  [COST MODEL]")
            print(f"    - Spread model: {cost_model.get('spread_model', 'N/A')}")
            print(f"    - Slippage model: {cost_model.get('slippage_model', 'N/A')}")
            print(f"    - Fee bps: {cost_model.get('fee_bps', 'N/A')}")
            
            if 'statistics' in cost_model:
                stats = cost_model['statistics']
                print(f"    - Spread p95: {stats.get('spread_p95_bps', 'N/A')} bps")
                print(f"    - Slippage p95: {stats.get('slippage_p95_bps', 'N/A')} bps")
            
        except Exception as e:
            print(f"    [ERROR] Failed to read cost_model: {e}")
    
    # 5. Check checks_report.json
    if 'checks_report.json' in found_files:
        try:
            obj = client.get_object(bucket, found_files['checks_report.json'])
            content = obj.read()
            checks = json.loads(content)
            
            print(f"\n  [CHECKS REPORT]")
            print(f"    - Overall status: {checks.get('status', 'N/A')}")
            
            if 'checks' in checks:
                for check_name, check_data in checks['checks'].items():
                    if isinstance(check_data, dict) and 'pass' in check_data:
                        status = "[PASS]" if check_data['pass'] else "[FAIL]"
                        print(f"    - {check_name}: {status}")
            
            if 'gates' in checks:
                for gate_name, gate_data in checks['gates'].items():
                    if isinstance(gate_data, dict) and 'pass' in gate_data:
                        status = "[PASS]" if gate_data['pass'] else "[FAIL]"
                        print(f"    - {gate_name}: {status}")
            
        except Exception as e:
            print(f"    [ERROR] Failed to read checks_report: {e}")
    
    # 6. Check metadata
    if '_metadata/metadata.json' in found_files:
        try:
            obj = client.get_object(bucket, found_files['_metadata/metadata.json'])
            content = obj.read()
            metadata = json.loads(content)
            
            print(f"\n  [METADATA]")
            print(f"    - DAG ID: {metadata.get('dag_id', 'N/A')}")
            print(f"    - Run ID: {metadata.get('run_id', 'N/A')}")
            print(f"    - Timestamp: {metadata.get('timestamp', 'N/A')}")
            
            if 'stats' in metadata:
                stats = metadata['stats']
                print(f"    - Total rows: {stats.get('total_rows', 'N/A')}")
                print(f"    - Total episodes: {stats.get('total_episodes', 'N/A')}")
                print(f"    - Features: {stats.get('features', 'N/A')}")
            
            print(f"    - Checks status: {metadata.get('checks_status', 'N/A')}")
            
        except Exception as e:
            print(f"    [ERROR] Failed to read metadata: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if ready_found and not missing_files:
        print("[SUCCESS] L4 Pipeline executed successfully!")
        print(f"[SUCCESS] All outputs saved correctly in MinIO")
        print(f"[SUCCESS] Run ID: {latest_run}")
        return True
    else:
        print("[WARNING] Pipeline may not have completed successfully")
        if missing_files:
            print(f"[WARNING] Missing files: {missing_files}")
        if not ready_found:
            print("[WARNING] READY flag not found")
        return False


def main():
    """Main verification function"""
    print("\n" + "="*60)
    print("L4 RL-READY COMPLETE VERIFICATION")
    print("="*60)
    
    # Connect to MinIO
    client = connect_minio()
    if not client:
        print("[ERROR] Cannot proceed without MinIO connection")
        return
    
    # Check L4 outputs
    success = check_l4_outputs(client)
    
    if success:
        print("\n" + "="*60)
        print("[CONFIRMED] L4 RL-READY PIPELINE EXECUTED SUCCESSFULLY")
        print("[CONFIRMED] ALL RESULTS SAVED IN MINIO")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("[STATUS] L4 Pipeline needs attention")
        print("[ACTION] Check Airflow UI for task status")
        print("[ACTION] URL: http://localhost:8081/dags/usdcop_m5__05_l4_rlready/grid")
        print("="*60)


if __name__ == "__main__":
    main()