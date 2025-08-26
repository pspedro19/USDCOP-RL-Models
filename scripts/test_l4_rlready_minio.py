"""
Test script for L4 RL-Ready MinIO operations
Tests bucket creation, connection, and basic file operations
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from minio import Minio
from minio.error import S3Error
import io
import pyarrow as pa
import pyarrow.parquet as pq

# MinIO configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_SECURE = False

def test_minio_connection():
    """Test MinIO connection and create buckets if needed"""
    try:
        # Create MinIO client
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        
        print("[OK] Connected to MinIO successfully")
        
        # List existing buckets
        buckets = client.list_buckets()
        print(f"\nExisting buckets: {[b.name for b in buckets]}")
        
        # Required buckets for L4
        required_buckets = [
            'ds-usdcop-feature',    # Input from L3
            'ds-usdcop-prepare',    # Input from L2 (normalization)
            'ds-usdcop-rlready'     # Output from L4
        ]
        
        # Create buckets if they don't exist
        for bucket_name in required_buckets:
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                print(f"[OK] Created bucket: {bucket_name}")
            else:
                print(f"[OK] Bucket exists: {bucket_name}")
        
        return client
        
    except Exception as e:
        print(f"[ERROR] MinIO connection failed: {e}")
        return None


def create_sample_l3_data(client):
    """Create sample L3 data for testing"""
    print("\n[INFO] Creating sample L3 data...")
    
    # Create sample features dataframe
    np.random.seed(42)
    n_episodes = 10
    steps_per_episode = 60
    n_rows = n_episodes * steps_per_episode
    
    # Generate episode data
    episodes = []
    for ep in range(n_episodes):
        episode_date = f"2024-01-{ep+1:02d}"
        for t in range(steps_per_episode):
            episodes.append({
                'episode_id': episode_date,
                't_in_episode': t,
                'is_terminal': t == 59
            })
    
    df = pd.DataFrame(episodes)
    
    # Add time columns
    base_time = pd.Timestamp('2024-01-01 14:00:00', tz='UTC')
    df['time_utc'] = [base_time + pd.Timedelta(minutes=5*i) for i in range(n_rows)]
    df['time_cot'] = df['time_utc'] - pd.Timedelta(hours=5)
    
    # Add OHLC data
    base_price = 4000
    df['close'] = base_price + np.cumsum(np.random.randn(n_rows) * 5)
    df['open'] = df['close'] + np.random.randn(n_rows) * 2
    df['high'] = np.maximum(df['open'], df['close']) + abs(np.random.randn(n_rows) * 3)
    df['low'] = np.minimum(df['open'], df['close']) - abs(np.random.randn(n_rows) * 3)
    df['volume'] = 1000 + np.random.randint(0, 500, n_rows)
    
    # Add trainable features
    trainable_features = [
        'return_1', 'return_3', 'rsi_14', 'bollinger_pct_b',
        'atr_14_norm', 'range_norm', 'clv', 'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos', 'ema_slope_6', 'macd_histogram',
        'parkinson_vol', 'body_ratio', 'stoch_k', 'williams_r'
    ]
    
    for feature in trainable_features:
        df[feature] = np.random.randn(n_rows)
    
    # Add some other features
    df['is_premium'] = 1
    df['is_valid_bar'] = 1
    df['ohlc_valid'] = 1
    df['atr_14'] = abs(np.random.randn(n_rows) * 10 + 20)
    
    # Save to MinIO
    execution_date = datetime.now().strftime('%Y-%m-%d')
    run_id = f"L3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Path structure
    base_path = f"usdcop_m5__04_l3_feature/date={execution_date}/run_id={run_id}"
    
    # 1. Save features.parquet
    table = pa.Table.from_pandas(df)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    
    features_key = f"{base_path}/features.parquet"
    client.put_object(
        'ds-usdcop-feature',
        features_key,
        buffer,
        length=buffer.getbuffer().nbytes
    )
    print(f"[OK] Saved: {features_key}")
    
    # 2. Save feature_spec.json
    feature_spec = {
        'features': list(df.columns),
        'dtypes': {col: str(df[col].dtype) for col in df.columns}
    }
    
    spec_key = f"{base_path}/_metadata/feature_spec.json"
    spec_data = json.dumps(feature_spec, indent=2).encode('utf-8')
    client.put_object(
        'ds-usdcop-feature',
        spec_key,
        io.BytesIO(spec_data),
        length=len(spec_data)
    )
    print(f"[OK] Saved: {spec_key}")
    
    # 3. Save leakage_gate.json
    leakage_gate = {
        'trainable_features': trainable_features,
        'excluded_features': ['is_premium', 'is_valid_bar', 'ohlc_valid'],
        'max_correlation': 0.08
    }
    
    gate_key = f"{base_path}/_reports/leakage_gate.json"
    gate_data = json.dumps(leakage_gate, indent=2).encode('utf-8')
    client.put_object(
        'ds-usdcop-feature',
        gate_key,
        io.BytesIO(gate_data),
        length=len(gate_data)
    )
    print(f"[OK] Saved: {gate_key}")
    
    # 4. Save metadata.json
    metadata = {
        'timestamp': datetime.utcnow().isoformat(),
        'dag_id': 'usdcop_m5__04_l3_feature',
        'run_id': run_id,
        'execution_date': execution_date,
        'rows': len(df),
        'columns': len(df.columns),
        'episodes': n_episodes
    }
    
    meta_key = f"{base_path}/_metadata/metadata.json"
    meta_data = json.dumps(metadata, indent=2).encode('utf-8')
    client.put_object(
        'ds-usdcop-feature',
        meta_key,
        io.BytesIO(meta_data),
        length=len(meta_data)
    )
    print(f"[OK] Saved: {meta_key}")
    
    # 5. Create READY signal
    ready_key = f"{base_path}/_control/READY"
    ready_data = json.dumps({
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'READY'
    }).encode('utf-8')
    client.put_object(
        'ds-usdcop-feature',
        ready_key,
        io.BytesIO(ready_data),
        length=len(ready_data)
    )
    print(f"[OK] Saved: {ready_key}")
    
    # 6. Create L2 normalization reference (optional)
    normalization_ref = {}
    for feature in trainable_features:
        # Global stats
        normalization_ref[feature] = {
            'median': float(df[feature].median()),
            'mad': float(np.median(np.abs(df[feature] - df[feature].median()))),
            'mean': float(df[feature].mean()),
            'std': float(df[feature].std())
        }
        
        # Hour-based stats (simplified)
        for hour in range(24):
            hour_data = df[df['time_utc'].dt.hour == hour][feature] if 'time_utc' in df else df[feature]
            if len(hour_data) > 0:
                normalization_ref[f"{feature}_h{hour:02d}"] = {
                    'median': float(hour_data.median()),
                    'mad': float(np.median(np.abs(hour_data - hour_data.median())) if len(hour_data) > 1 else 1.0)
                }
    
    norm_key = "usdcop_m5__03_l2_prepare/_statistics/normalization_ref.json"
    norm_data = json.dumps(normalization_ref, indent=2).encode('utf-8')
    
    # Create L2 bucket if needed
    if not client.bucket_exists('ds-usdcop-prepare'):
        client.make_bucket('ds-usdcop-prepare')
    
    client.put_object(
        'ds-usdcop-prepare',
        norm_key,
        io.BytesIO(norm_data),
        length=len(norm_data)
    )
    print(f"[OK] Saved: {norm_key}")
    
    return execution_date, base_path


def list_bucket_contents(client, bucket_name, prefix=""):
    """List contents of a bucket"""
    print(f"\n[DIR] Contents of {bucket_name}/{prefix}")
    try:
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
        count = 0
        for obj in objects:
            print(f"  - {obj.object_name} ({obj.size} bytes)")
            count += 1
        
        if count == 0:
            print("  (empty)")
        else:
            print(f"  Total: {count} objects")
            
    except Exception as e:
        print(f"  Error listing bucket: {e}")


def test_l4_output_structure(client):
    """Test L4 output structure"""
    print("\n[TEST] Testing L4 output structure...")
    
    execution_date = datetime.now().strftime('%Y-%m-%d')
    run_id = f"RLREADY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    base_path = f"usdcop_m5__05_l4_rlready/market=usdcop/timeframe=m5/date={execution_date}/run_id={run_id}"
    
    # Create sample output files
    outputs = {
        'replay_dataset.parquet': pd.DataFrame({'test': [1, 2, 3]}),
        'episodes_index.parquet': pd.DataFrame({'episode_id': ['2024-01-01']}),
        'env_spec.json': {'framework': 'gymnasium', 'seed': 42},
        'cost_model.json': {'spread_model': 'corwin_schultz_60m'},
        'reward_spec.json': {'formula': 'position * return - cost'},
        'action_spec.json': {'mapping': {'-1': 'short', '0': 'flat', '1': 'long'}},
        'split_spec.json': {'scheme': 'walk_forward'},
        'checks_report.json': {'status': 'PASS'},
        '_metadata/metadata.json': {'timestamp': datetime.utcnow().isoformat()},
        '_control/READY': {'status': 'READY', 'timestamp': datetime.utcnow().isoformat()}
    }
    
    bucket = 'ds-usdcop-rlready'
    
    for filename, data in outputs.items():
        key = f"{base_path}/{filename}"
        
        if filename.endswith('.parquet'):
            # Save as parquet
            table = pa.Table.from_pandas(data)
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            buffer.seek(0)
            client.put_object(bucket, key, buffer, length=buffer.getbuffer().nbytes)
        else:
            # Save as JSON
            json_data = json.dumps(data, indent=2).encode('utf-8')
            client.put_object(bucket, key, io.BytesIO(json_data), length=len(json_data))
        
        print(f"[OK] Created: {key}")
    
    return base_path


def verify_l4_outputs(client, base_path):
    """Verify L4 outputs are correctly saved"""
    print(f"\n[VERIFY] Verifying L4 outputs at: {base_path}")
    
    bucket = 'ds-usdcop-rlready'
    required_files = [
        'replay_dataset.parquet',
        'episodes_index.parquet',
        'env_spec.json',
        'cost_model.json',
        'reward_spec.json',
        'action_spec.json',
        'split_spec.json',
        'checks_report.json',
        '_metadata/metadata.json',
        '_control/READY'
    ]
    
    for filename in required_files:
        key = f"{base_path}/{filename}"
        try:
            stat = client.stat_object(bucket, key)
            print(f"[OK] Found: {filename} ({stat.size} bytes)")
        except S3Error:
            print(f"[ERROR] Missing: {filename}")


def main():
    """Main test function"""
    print("=" * 60)
    print("L4 RL-Ready MinIO Test")
    print("=" * 60)
    
    # Test connection
    client = test_minio_connection()
    if not client:
        print("[ERROR] Failed to connect to MinIO")
        return
    
    # Create sample L3 data
    execution_date, l3_path = create_sample_l3_data(client)
    
    # List L3 contents
    list_bucket_contents(client, 'ds-usdcop-feature', 'usdcop_m5__04_l3_feature')
    
    # Test L4 output structure
    l4_path = test_l4_output_structure(client)
    
    # List L4 contents
    list_bucket_contents(client, 'ds-usdcop-rlready', 'usdcop_m5__05_l4_rlready')
    
    # Verify outputs
    verify_l4_outputs(client, l4_path)
    
    print("\n" + "=" * 60)
    print("[OK] L4 RL-Ready MinIO test complete!")
    print("=" * 60)
    
    # Print Airflow trigger command
    print("\n[INFO] To trigger the L4 DAG in Airflow, use:")
    print(f"   airflow dags trigger usdcop_m5__05_l4_rlready --exec-date {execution_date}")


if __name__ == "__main__":
    main()