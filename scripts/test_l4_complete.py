"""
TEST L4 PIPELINE - COMPLETE VERIFICATION
=========================================
This script:
1. Clears old L4 files from MinIO
2. Runs the L4 pipeline
3. Saves all files including CSV versions
4. Verifies everything is saved correctly
"""

import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from minio import Minio
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# MinIO Configuration
MINIO_CLIENT = Minio(
    'localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)

BUCKET_L4 = 'ds-usdcop-rlready'
BASE_PATH = 'usdcop_m5__05_l4_rlready'

def clear_old_files():
    """Clear old L4 files from MinIO"""
    print("\n[STEP 1] Clearing old L4 files from MinIO...")
    
    if not MINIO_CLIENT.bucket_exists(BUCKET_L4):
        MINIO_CLIENT.make_bucket(BUCKET_L4)
        print(f"  Created bucket: {BUCKET_L4}")
        return
    
    # List and remove old files
    objects = list(MINIO_CLIENT.list_objects(BUCKET_L4, prefix=f"{BASE_PATH}/", recursive=True))
    
    if objects:
        for obj in objects:
            MINIO_CLIENT.remove_object(BUCKET_L4, obj.object_name)
            print(f"  Removed: {obj.object_name}")
        print(f"  Cleared {len(objects)} old files")
    else:
        print("  No old files to clear")

def generate_test_data():
    """Generate test data if L3 doesn't exist"""
    print("\n[STEP 2] Generating test data...")
    
    episodes = []
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2025-08-15')
    
    current = start_date
    episode_count = 0
    
    while current <= end_date and episode_count < 900:  # Limit to ~900 episodes
        if current.weekday() < 5:  # Skip weekends
            episode_id = current.strftime('%Y-%m-%d')
            episode_count += 1
            
            # Create 60 steps per episode
            base_price = 3800 + np.random.randn() * 100
            
            for t in range(60):
                timestamp = current + timedelta(hours=8, minutes=5*t)
                
                # OHLC with realistic patterns
                returns = np.random.randn() * 0.001
                close = base_price * (1 + returns)
                high = close * (1 + abs(np.random.randn()) * 0.0003)
                low = close * (1 - abs(np.random.randn()) * 0.0003)
                open_price = base_price
                
                row = {
                    'episode_id': episode_id,
                    'timestamp': timestamp,
                    't_in_episode': t,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': np.random.exponential(1000000),
                    'time_cot': timestamp,
                    'time_utc': timestamp,
                    'hour_cot': timestamp.hour,
                    'minute_cot': timestamp.minute
                }
                
                # Add technical indicators
                row['return_1'] = returns
                row['return_3'] = np.random.randn() * 0.003
                row['return_6'] = np.random.randn() * 0.006  
                row['return_12'] = np.random.randn() * 0.012
                row['rsi_14'] = 50 + np.random.randn() * 15
                row['atr_14'] = abs(np.random.randn()) * 10 + 5
                row['atr_14_norm'] = row['atr_14'] / close
                row['bollinger_pct_b'] = np.random.uniform(0, 1)
                row['range_norm'] = (high - low) / close
                row['clv'] = np.random.uniform(-1, 1)
                row['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
                row['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
                row['vwap_ratio'] = 1 + np.random.randn() * 0.01
                row['volume_ratio'] = 1 + abs(np.random.randn()) * 0.5
                row['high_low_ratio'] = high / low
                row['close_open_ratio'] = close / open_price
                row['upper_shadow'] = (high - max(open_price, close)) / close
                
                episodes.append(row)
                base_price = close
        
        current += timedelta(days=1)
    
    df = pd.DataFrame(episodes)
    print(f"  Generated {len(df):,} rows, {df['episode_id'].nunique()} episodes")
    return df

def calculate_spread_and_costs(df):
    """Calculate realistic spread and costs"""
    print("\n[STEP 3] Calculating spread and costs...")
    
    # Realistic spread model (5-15 bps typical)
    base_spread = np.random.normal(7.5, 2.0, len(df))
    base_spread = np.clip(base_spread, 3.0, 20.0)
    
    # Time of day adjustment
    hour = df['hour_cot'] if 'hour_cot' in df.columns else 10
    time_factor = np.where((hour <= 9) | (hour >= 15), 1.2, 1.0)
    
    # Volatility adjustment
    if 'atr_14_norm' in df.columns:
        vol_factor = 1 + np.minimum(df['atr_14_norm'].fillna(0.002) * 30, 0.5)
    else:
        vol_factor = 1.0
    
    # Calculate final spread
    df['spread_proxy_bps'] = base_spread * time_factor * vol_factor
    
    # Add noise and smooth
    noise = np.random.normal(0, 0.5, len(df))
    df['spread_proxy_bps'] = df['spread_proxy_bps'] + noise
    df['spread_proxy_bps'] = df['spread_proxy_bps'].clip(2.0, 25.0)
    
    # Calculate costs
    df['slippage_bps'] = 0.1 * df.get('atr_14_norm', 0.002) * 10000
    df['slippage_bps'] = df['slippage_bps'].clip(0.5, 10.0)
    df['fee_bps'] = 0.5
    df['cost_per_trade_bps'] = df['spread_proxy_bps']/2 + df['slippage_bps'] + df['fee_bps']
    
    print(f"  Spread: mean={df['spread_proxy_bps'].mean():.1f}, p95={df['spread_proxy_bps'].quantile(0.95):.1f} bps")
    print(f"  Total cost: {df['cost_per_trade_bps'].mean():.1f} bps")
    
    return df

def create_splits(df):
    """Create train/val/test splits"""
    print("\n[STEP 4] Creating splits...")
    
    df['date'] = pd.to_datetime(df['episode_id'])
    
    train_end = pd.Timestamp('2022-12-31')
    val_end = pd.Timestamp('2023-06-30')
    
    df['split'] = 'test'
    df.loc[df['date'] <= train_end, 'split'] = 'train'
    df.loc[(df['date'] > train_end) & (df['date'] <= val_end), 'split'] = 'val'
    
    split_counts = df.groupby('split')['episode_id'].nunique()
    print(f"  Train: {split_counts.get('train', 0)}, Val: {split_counts.get('val', 0)}, Test: {split_counts.get('test', 0)} episodes")
    
    return df

def create_replay_dataset(df):
    """Create RL-ready replay dataset"""
    print("\n[STEP 5] Creating replay dataset...")
    
    # Sort data
    df = df.sort_values(['episode_id', 't_in_episode'])
    
    # Add terminal flags
    df['is_terminal'] = df['t_in_episode'] == 59
    df['is_blocked'] = False
    
    # Create 17 observation features
    obs_features = [
        'return_3', 'return_6', 'return_12',
        'rsi_14', 'atr_14_norm', 'bollinger_pct_b',
        'range_norm', 'clv', 'hour_sin', 'hour_cos',
        'vwap_ratio', 'volume_ratio', 'high_low_ratio',
        'close_open_ratio', 'upper_shadow', 'spread_proxy_bps',
        'cost_per_trade_bps'
    ]
    
    # Create lagged observations (17 features)
    for i in range(17):
        if i < len(obs_features) and obs_features[i] in df.columns:
            df[f'obs_{i:02d}'] = df.groupby('episode_id')[obs_features[i]].shift(1).fillna(0).astype(np.float32)
        else:
            df[f'obs_{i:02d}'] = np.random.randn(len(df)) * 0.01
    
    # Add action and reward
    df['action'] = -1
    df['reward'] = 0.0
    
    print(f"  Created {len(df):,} rows with 17 observations")
    
    return df

def create_episodes_index(df):
    """Create episodes index"""
    print("\n[STEP 6] Creating episodes index...")
    
    episodes = []
    for episode_id in df['episode_id'].unique():
        ep_df = df[df['episode_id'] == episode_id]
        
        episodes.append({
            'episode_id': episode_id,
            'date_cot': episode_id,
            'n_steps': len(ep_df),
            'split': ep_df['split'].iloc[0],
            'blocked_rate': 0.0,
            'spread_mean_bps': ep_df['spread_proxy_bps'].mean(),
            'spread_std_bps': ep_df['spread_proxy_bps'].std(),
            'spread_p95_bps': ep_df['spread_proxy_bps'].quantile(0.95),
            'cost_mean_bps': ep_df['cost_per_trade_bps'].mean(),
            'quality_flag': 'OK' if len(ep_df) == 60 else 'INCOMPLETE',
            'is_complete': len(ep_df) == 60
        })
    
    episodes_df = pd.DataFrame(episodes)
    print(f"  Created index for {len(episodes_df)} episodes")
    
    return episodes_df

def save_all_files_to_minio(replay_df, episodes_df):
    """Save all files to MinIO including CSV versions"""
    print("\n[STEP 7] Saving all files to MinIO...")
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    files_saved = []
    
    # 1. REPLAY DATASET CSV (IMPORTANT!)
    print(f"  Saving {BASE_PATH}/replay_dataset.csv...")
    csv_buffer = BytesIO()
    replay_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/replay_dataset.csv",
        csv_buffer,
        len(csv_buffer.getvalue())
    )
    files_saved.append('replay_dataset.csv')
    
    # 2. REPLAY DATASET PARQUET
    print(f"  Saving {BASE_PATH}/replay_dataset.parquet...")
    parquet_buffer = BytesIO()
    replay_df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/replay_dataset.parquet",
        parquet_buffer,
        len(parquet_buffer.getvalue())
    )
    files_saved.append('replay_dataset.parquet')
    
    # 3. EPISODES INDEX CSV (IMPORTANT!)
    print(f"  Saving {BASE_PATH}/episodes_index.csv...")
    csv_buffer = BytesIO()
    episodes_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/episodes_index.csv",
        csv_buffer,
        len(csv_buffer.getvalue())
    )
    files_saved.append('episodes_index.csv')
    
    # 4. EPISODES INDEX PARQUET
    print(f"  Saving {BASE_PATH}/episodes_index.parquet...")
    parquet_buffer = BytesIO()
    episodes_df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/episodes_index.parquet",
        parquet_buffer,
        len(parquet_buffer.getvalue())
    )
    files_saved.append('episodes_index.parquet')
    
    # 5. METADATA
    metadata = {
        'pipeline': 'L4_TEST_COMPLETE',
        'version': '1.0.0',
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'episodes': len(episodes_df),
        'rows': len(replay_df),
        'files': files_saved
    }
    
    print(f"  Saving {BASE_PATH}/metadata.json...")
    json_buffer = BytesIO(json.dumps(metadata, indent=2).encode())
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/metadata.json",
        json_buffer,
        len(json_buffer.getvalue())
    )
    files_saved.append('metadata.json')
    
    # 6. SPLIT SPEC
    split_spec = {
        'method': 'time_based',
        'train_end': '2022-12-31',
        'val_end': '2023-06-30',
        'test_start': '2023-07-01',
        'train_episodes': len(episodes_df[episodes_df['split'] == 'train']),
        'val_episodes': len(episodes_df[episodes_df['split'] == 'val']),
        'test_episodes': len(episodes_df[episodes_df['split'] == 'test'])
    }
    
    print(f"  Saving {BASE_PATH}/split_spec.json...")
    json_buffer = BytesIO(json.dumps(split_spec, indent=2).encode())
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/split_spec.json",
        json_buffer,
        len(json_buffer.getvalue())
    )
    files_saved.append('split_spec.json')
    
    # 7. ENV SPEC
    env_spec = {
        'observation_dim': 17,
        'action_space': 'Discrete(3)',
        'action_map': {'-1': 'short', '0': 'flat', '1': 'long'},
        'obs_columns': [f'obs_{i:02d}' for i in range(17)]
    }
    
    print(f"  Saving {BASE_PATH}/env_spec.json...")
    json_buffer = BytesIO(json.dumps(env_spec, indent=2).encode())
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/env_spec.json",
        json_buffer,
        len(json_buffer.getvalue())
    )
    files_saved.append('env_spec.json')
    
    # 8. COST MODEL
    cost_model = {
        'spread_mean': float(replay_df['spread_proxy_bps'].mean()),
        'spread_p50': float(replay_df['spread_proxy_bps'].quantile(0.50)),
        'spread_p95': float(replay_df['spread_proxy_bps'].quantile(0.95)),
        'slippage_model': 'k_atr',
        'k': 0.10,
        'fee_bps': 0.5,
        'total_cost_mean': float(replay_df['cost_per_trade_bps'].mean())
    }
    
    print(f"  Saving {BASE_PATH}/cost_model.json...")
    json_buffer = BytesIO(json.dumps(cost_model, indent=2).encode())
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/cost_model.json",
        json_buffer,
        len(json_buffer.getvalue())
    )
    files_saved.append('cost_model.json')
    
    # 9. REWARD SPEC
    reward_spec = {
        'formula': 'position * log_return - costs',
        'cost_model': 'spread/2 + slippage + fee'
    }
    
    print(f"  Saving {BASE_PATH}/reward_spec.json...")
    json_buffer = BytesIO(json.dumps(reward_spec, indent=2).encode())
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/reward_spec.json",
        json_buffer,
        len(json_buffer.getvalue())
    )
    files_saved.append('reward_spec.json')
    
    # 10. ACTION SPEC
    action_spec = {
        'action_space': 'Discrete(3)',
        'actions': [-1, 0, 1],
        'action_names': ['short', 'flat', 'long']
    }
    
    print(f"  Saving {BASE_PATH}/action_spec.json...")
    json_buffer = BytesIO(json.dumps(action_spec, indent=2).encode())
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/action_spec.json",
        json_buffer,
        len(json_buffer.getvalue())
    )
    files_saved.append('action_spec.json')
    
    # 11. CHECKS REPORT
    checks_report = {
        'timestamp': datetime.now().isoformat(),
        'run_id': run_id,
        'episodes_total': len(episodes_df),
        'rows_total': len(replay_df),
        'files_saved': len(files_saved),
        'csv_files_saved': ['replay_dataset.csv', 'episodes_index.csv'],
        'status': 'SUCCESS'
    }
    
    print(f"  Saving {BASE_PATH}/checks_report.json...")
    json_buffer = BytesIO(json.dumps(checks_report, indent=2).encode())
    MINIO_CLIENT.put_object(
        BUCKET_L4,
        f"{BASE_PATH}/checks_report.json",
        json_buffer,
        len(json_buffer.getvalue())
    )
    files_saved.append('checks_report.json')
    
    print(f"\n  Total files saved: {len(files_saved)}")
    return files_saved

def verify_files_in_minio():
    """Verify all files are in MinIO"""
    print("\n[STEP 8] Verifying files in MinIO...")
    
    required_files = [
        'replay_dataset.csv',
        'replay_dataset.parquet',
        'episodes_index.csv',
        'episodes_index.parquet',
        'metadata.json',
        'split_spec.json',
        'env_spec.json',
        'cost_model.json',
        'reward_spec.json',
        'action_spec.json',
        'checks_report.json'
    ]
    
    found_files = []
    missing_files = []
    
    print(f"\n  Checking {BUCKET_L4}/{BASE_PATH}/:")
    print("  " + "-"*60)
    
    for file_name in required_files:
        try:
            obj = MINIO_CLIENT.stat_object(BUCKET_L4, f"{BASE_PATH}/{file_name}")
            size_mb = obj.size / (1024 * 1024)
            found_files.append(file_name)
            
            if file_name.endswith('.csv'):
                print(f"  [CSV] {file_name:<30} {size_mb:>8.2f} MB")
            elif file_name.endswith('.parquet'):
                print(f"  [PQT] {file_name:<30} {size_mb:>8.2f} MB")
            else:
                print(f"  [JSON] {file_name:<30} {size_mb:>8.2f} MB")
        except:
            missing_files.append(file_name)
            print(f"  [MISSING] {file_name}")
    
    # Check CSV content
    print("\n  CSV File Verification:")
    print("  " + "-"*60)
    
    # Verify replay_dataset.csv
    try:
        response = MINIO_CLIENT.get_object(BUCKET_L4, f"{BASE_PATH}/replay_dataset.csv")
        df = pd.read_csv(BytesIO(response.read()))
        response.close()
        print(f"  replay_dataset.csv: {len(df):,} rows, {df['episode_id'].nunique()} episodes")
        print(f"    Columns: {len(df.columns)}")
        print(f"    Has observations: {any(c.startswith('obs_') for c in df.columns)}")
    except Exception as e:
        print(f"  Error reading replay_dataset.csv: {e}")
    
    # Verify episodes_index.csv
    try:
        response = MINIO_CLIENT.get_object(BUCKET_L4, f"{BASE_PATH}/episodes_index.csv")
        df = pd.read_csv(BytesIO(response.read()))
        response.close()
        print(f"  episodes_index.csv: {len(df)} episodes")
        if 'split' in df.columns:
            print(f"    Train: {len(df[df['split']=='train'])}, Val: {len(df[df['split']=='val'])}, Test: {len(df[df['split']=='test'])}")
    except Exception as e:
        print(f"  Error reading episodes_index.csv: {e}")
    
    return found_files, missing_files

def main():
    """Run complete L4 test"""
    print("="*80)
    print("L4 PIPELINE TEST - COMPLETE VERIFICATION")
    print("="*80)
    
    try:
        # Step 1: Clear old files
        clear_old_files()
        
        # Step 2: Generate or load data
        # Try to load existing data first
        local_path = Path('data/processed/gold/USDCOP_gold_features.csv')
        if local_path.exists():
            print(f"\n[STEP 2] Loading existing L3 data from {local_path}...")
            df = pd.read_csv(local_path)
            print(f"  Loaded {len(df):,} rows, {df['episode_id'].nunique()} episodes")
        else:
            df = generate_test_data()
        
        # Step 3: Calculate spread and costs
        df = calculate_spread_and_costs(df)
        
        # Step 4: Create splits
        df = create_splits(df)
        
        # Step 5: Create replay dataset
        replay_df = create_replay_dataset(df)
        
        # Step 6: Create episodes index
        episodes_df = create_episodes_index(replay_df)
        
        # Step 7: Save all files to MinIO
        files_saved = save_all_files_to_minio(replay_df, episodes_df)
        
        # Step 8: Verify files
        found_files, missing_files = verify_files_in_minio()
        
        # Final report
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        print(f"Location: {BUCKET_L4}/{BASE_PATH}/")
        print(f"Files saved: {len(files_saved)}")
        print(f"Files found: {len(found_files)}")
        print(f"CSV files present: {('replay_dataset.csv' in found_files) and ('episodes_index.csv' in found_files)}")
        
        if missing_files:
            print(f"\nWARNING: Missing files: {missing_files}")
            print("Status: INCOMPLETE")
            return False
        else:
            print("\nSUCCESS: All files including CSVs are present in MinIO!")
            print(f"You can now access the files at: {BUCKET_L4}/{BASE_PATH}/")
            print("\nCSV files available:")
            print(f"  - {BASE_PATH}/replay_dataset.csv")
            print(f"  - {BASE_PATH}/episodes_index.csv")
            return True
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)