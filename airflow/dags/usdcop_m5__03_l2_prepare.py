"""
L2 PREPARE DAG - Complete Implementation
========================================
Takes L1 standardized data and produces ML-ready features with:
- Deseasonalization using HOD baselines
- Winsorization (returns only, not prices)
- Both STRICT (60/60 only) and FLEX (with placeholders) versions
- Complete audit trail and quality reports

Input from L1:
- standardized_data_accepted.parquet (55,705 rows, 929 episodes)
- hod_baseline.csv (HOD statistics for deseasonalization)
- _metadata.json (verification)
- daily_quality_60.csv (optional telemetry)

Output:
- data_premium_strict.parquet/csv (60/60 episodes only)
- data_premium_flexible.parquet/csv (59/60 padded to 60)
- Quality reports and statistics
- Complete metadata with SHA256 hashes
"""

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from airflow.models import Variable
import pandas as pd
import numpy as np
import json
import hashlib
import logging
import io
import uuid
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional
import pyarrow as pa
import pyarrow.parquet as pq

# Import critical outputs function
try:
    from l2_save_critical_outputs import save_critical_l2_outputs
    USE_CRITICAL_OUTPUTS = True
except ImportError:
    USE_CRITICAL_OUTPUTS = False
    logging.warning("Critical outputs module not available, using original save function")

# Import dynamic configuration
try:
    from utils.pipeline_config import get_bucket_config
    buckets = get_bucket_config("usdcop_m5__03_l2_prepare")
    BUCKET_INPUT = buckets.get('input', '01-l1-ds-usdcop-standardize')
    BUCKET_OUTPUT = buckets.get('output', '02-l2-ds-usdcop-prepare')
except ImportError:
    # Fallback if config loader is not available
    BUCKET_INPUT = "01-l1-ds-usdcop-standardize"
    BUCKET_OUTPUT = "02-l2-ds-usdcop-prepare"

# Configuration
DAG_ID = "usdcop_m5__03_l2_prepare"
MARKET = "usdcop"
TIMEFRAME = "m5"
DATASET_VERSION = "L2.v1.0"

# Winsorization parameters
WINSOR_SIGMA = 4.0  # Number of robust SDs for clipping
EPSILON = 1e-10     # Small value to avoid division by zero

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

def load_and_validate_l1_data(**context):
    """Step 0: Load L1 data and validate basic contracts"""
    logger.info("="*80)
    logger.info("L2 PREPARE - LOADING AND VALIDATING L1 DATA")
    logger.info("="*80)
    
    # Use dag_run.run_id for consistency
    dag_run_id = context['dag_run'].run_id
    # For UUID-based tracking in MinIO
    run_id = str(uuid.uuid4())
    execution_date = context['ds']
    logger.info(f"DAG Run ID: {dag_run_id}")
    logger.info(f"Storage Run ID: {run_id}")
    logger.info(f"Execution Date: {execution_date}")
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    
    # 1. Dynamically find and load L1 data
    # Look for the most recent L1 output
    input_key = None
    
    # Try consolidated path first
    consolidated_key = "usdcop_m5__02_l1_standardize/consolidated/standardized_data_accepted.parquet"
    try:
        if s3_hook.check_for_key(consolidated_key, bucket_name=BUCKET_INPUT):
            input_key = consolidated_key
            logger.info(f"Found consolidated L1 data")
    except:
        pass
    
    # If not found, look for date-specific data
    if not input_key:
        # Look for any L1 output with accepted data
        prefix = "usdcop_m5__02_l1_standardize/"
        try:
            keys = s3_hook.list_keys(bucket_name=BUCKET_INPUT, prefix=prefix)
            if keys:
                # Find accepted parquet files
                accepted_keys = [k for k in keys if 'accepted' in k and k.endswith('.parquet')]
                if accepted_keys:
                    # Get the most recent one
                    input_key = sorted(accepted_keys)[-1]
                    logger.info(f"Found L1 accepted data at: {input_key}")
        except Exception as e:
            logger.warning(f"Error searching for L1 data: {e}")
    
    # Fallback to default path
    if not input_key:
        input_key = consolidated_key
        logger.warning(f"Using fallback L1 path")
    
    logger.info(f"Loading L1 data from: s3://{BUCKET_INPUT}/{input_key}")
    
    try:
        file_obj = s3_hook.get_key(input_key, bucket_name=BUCKET_INPUT)
        content = file_obj.get()['Body'].read()
        df = pd.read_parquet(io.BytesIO(content))
        logger.info(f"✅ Loaded {len(df)} rows from L1 accepted data")
    except Exception as e:
        logger.error(f"❌ Failed to load L1 data: {e}")
        raise
    
    # 2. Load HOD baselines - search dynamically
    hod_key = None
    hod_search_paths = [
        "usdcop_m5__02_l1_standardize/consolidated/_statistics/hod_baseline.csv",
        "usdcop_m5__02_l1_standardize/_statistics/hod_baseline.csv",
    ]
    
    # Also search for any HOD baseline in L1 output
    try:
        keys = s3_hook.list_keys(bucket_name=BUCKET_INPUT, prefix="usdcop_m5__02_l1_standardize/")
        if keys:
            hod_keys = [k for k in keys if 'hod_baseline' in k and k.endswith('.csv')]
            if hod_keys:
                hod_search_paths.extend(hod_keys)
    except:
        pass
    
    for path in hod_search_paths:
        try:
            if s3_hook.check_for_key(path, bucket_name=BUCKET_INPUT):
                hod_key = path
                logger.info(f"Found HOD baseline at: {path}")
                break
        except:
            continue
    
    if not hod_key:
        # AUDIT FIX: Disallow HOD fallback in production - must have L1 baselines
        error_msg = (
            "CRITICAL ERROR: HOD baseline missing from L1! "
            "Aborting L2 to avoid neutral fallback which inflates winsorization. "
            "Please ensure L1 pipeline has generated HOD baselines before running L2."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    else:
        logger.info(f"Loading HOD baselines from: s3://{BUCKET_INPUT}/{hod_key}")
        try:
            file_obj = s3_hook.get_key(hod_key, bucket_name=BUCKET_INPUT)
            content = file_obj.get()['Body'].read()
            hod_df = pd.read_csv(io.BytesIO(content))
            logger.info(f"✅ Loaded HOD baselines for {len(hod_df)} hours")
        except Exception as e:
            logger.error(f"❌ Failed to load HOD baselines: {e}")
            raise
    
    # 3. Load L1 metadata for verification
    meta_key = "usdcop_m5__02_l1_standardize/consolidated/_metadata.json"
    try:
        file_obj = s3_hook.get_key(meta_key, bucket_name=BUCKET_INPUT)
        content = file_obj.get()['Body'].read()
        l1_metadata = json.loads(content)
        logger.info(f"L1 version: {l1_metadata.get('dataset_version')}")
        logger.info(f"L1 accepted rows: {l1_metadata.get('rows_accepted')}")
        logger.info(f"L1 accepted episodes: {l1_metadata.get('episodes_accepted')}")
    except Exception as e:
        logger.warning(f"Could not load L1 metadata: {e}")
        l1_metadata = {}
    
    # 4. Optional: Load quality report for reconciliation
    quality_key = "usdcop_m5__02_l1_standardize/consolidated/_reports/daily_quality_60.csv"
    try:
        file_obj = s3_hook.get_key(quality_key, bucket_name=BUCKET_INPUT)
        content = file_obj.get()['Body'].read()
        quality_df = pd.read_csv(io.BytesIO(content))
        logger.info(f"✅ Loaded quality report with {len(quality_df)} episodes")
    except Exception as e:
        logger.warning(f"Could not load quality report: {e}")
        quality_df = pd.DataFrame()
    
    # VALIDATION CHECKS
    logger.info("\n" + "="*40)
    logger.info("VALIDATION CHECKS")
    logger.info("="*40)
    
    # Check 1: Row count matches metadata
    if l1_metadata.get('rows_accepted'):
        expected_rows = l1_metadata.get('rows_accepted')
        if len(df) != expected_rows:
            logger.warning(f"⚠️ Row count mismatch! Loaded: {len(df)}, Expected: {expected_rows}")
        else:
            logger.info(f"✅ Row count matches: {len(df)}")
    
    # Check 2: Unique keys
    # time_utc should be unique
    if 'time_utc' in df.columns:
        df['time_utc'] = pd.to_datetime(df['time_utc'])
        duplicates_time = df['time_utc'].duplicated().sum()
        if duplicates_time > 0:
            logger.error(f"❌ Found {duplicates_time} duplicate time_utc values!")
            raise ValueError(f"time_utc not unique: {duplicates_time} duplicates")
        else:
            logger.info("✅ time_utc is unique")
    
    # (episode_id, t_in_episode) should be unique
    if 'episode_id' in df.columns and 't_in_episode' in df.columns:
        duplicates_episode = df[['episode_id', 't_in_episode']].duplicated().sum()
        if duplicates_episode > 0:
            logger.error(f"❌ Found {duplicates_episode} duplicate (episode_id, t_in_episode) pairs!")
            raise ValueError(f"Episode keys not unique: {duplicates_episode} duplicates")
        else:
            logger.info("✅ (episode_id, t_in_episode) is unique")
    
    # Check 3: Grid validation
    if 'minute_cot' in df.columns:
        valid_minutes = set(range(0, 60, 5))
        invalid_minutes = ~df['minute_cot'].isin(valid_minutes)
        if invalid_minutes.sum() > 0:
            logger.warning(f"⚠️ Found {invalid_minutes.sum()} bars with invalid minute_cot")
        else:
            logger.info("✅ All minute_cot values are valid M5 grid")
    
    # Check 4: Premium flag - IMPROVEMENT: Hard gate on 100% premium
    if 'is_premium' in df.columns:
        non_premium = (~df['is_premium']).sum()
        if non_premium > 0:
            error_msg = f"ERROR: Found {non_premium} non-premium bars! STRICT requires 100% premium hours."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.info("✅ All bars are premium (100% requirement met)")
    
    # Convert datetime columns
    if 'time_cot' in df.columns:
        df['time_cot'] = pd.to_datetime(df['time_cot'])
    
    # Sort by episode and time
    df = df.sort_values(['episode_id', 't_in_episode'])
    
    # Save debug files for manual inspection (optional)
    # DEBUG: Save to timestamped files to avoid conflicts
    debug_timestamp = int(datetime.now().timestamp())
    df.to_parquet(f'/tmp/debug_{debug_timestamp}_l2_input_data.parquet', index=False)
    hod_df.to_csv(f'/tmp/debug_{debug_timestamp}_l2_hod_baseline.csv', index=False)
    if not quality_df.empty:
        quality_df.to_csv(f'/tmp/debug_{debug_timestamp}_l1_quality_report.csv', index=False)
    
    # Create initial metadata
    l2_metadata = {
        'run_id': run_id,
        'execution_date': execution_date,
        'dataset_version': DATASET_VERSION,
        'source_l1_run_id': l1_metadata.get('run_id', 'unknown'),
        'source_l1_version': l1_metadata.get('dataset_version', 'unknown'),
        'input_rows': len(df),
        'input_episodes': df['episode_id'].nunique(),
        'l1_rows_accepted': l1_metadata.get('rows_accepted', 0),
        'l1_episodes_accepted': l1_metadata.get('episodes_accepted', 0)
    }
    
    # Save HOD baseline to MinIO for future use
    buffer = io.BytesIO()
    hod_df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    # Use the correct output bucket
    hod_key = f'{DAG_ID}/market={MARKET}/timeframe={TIMEFRAME}/date={execution_date}/run_id={run_id}/_statistics/hod_baseline.csv'
    s3_hook.load_bytes(
        buffer.getvalue(),
        key=hod_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    
    # Save L1 data to MinIO for next task
    buffer_l1 = io.BytesIO()
    df.to_parquet(buffer_l1, index=False)
    buffer_l1.seek(0)
    
    # AUDIT FIX: Calculate L1 input hash for lineage tracking
    l1_bytes = buffer_l1.getvalue()
    l1_input_hash = hashlib.sha256(l1_bytes).hexdigest()
    l2_metadata['l1_input_hash'] = l1_input_hash
    logger.info(f"L1 input hash: {l1_input_hash[:16]}...")
    
    l1_data_key = f'{DAG_ID}/market={MARKET}/timeframe={TIMEFRAME}/date={execution_date}/run_id={run_id}/l1_input_data.parquet'
    s3_hook.load_bytes(
        l1_bytes,
        key=l1_data_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    
    # Push to XCom
    context['task_instance'].xcom_push(key='l2_metadata', value=l2_metadata)
    context['task_instance'].xcom_push(key='validation_passed', value=True)
    context['task_instance'].xcom_push(key='hod_baseline_path', value=hod_key)
    context['task_instance'].xcom_push(key='l1_data_path', value=l1_data_key)
    context['task_instance'].xcom_push(key='run_id', value=run_id)
    context['task_instance'].xcom_push(key='bucket_output', value=BUCKET_OUTPUT)  # Add this for downstream tasks
    
    logger.info(f"\n✅ Validation complete. Loaded {len(df)} rows from {df['episode_id'].nunique()} episodes")
    
    return len(df)

def calculate_base_features(**context):
    """Step 1: Calculate base features (ret_log_1, range_bps, ohlc4)"""
    logger.info("="*80)
    logger.info("CALCULATING BASE FEATURES")
    logger.info("="*80)
    
    # Load L1 data from MinIO
    ti = context['task_instance']
    l1_data_path = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='l1_data_path')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    
    # FIX: Use BUCKET_OUTPUT directly since l1_data_path is saved to BUCKET_OUTPUT
    logger.info(f"Loading L1 data from: s3://{BUCKET_OUTPUT}/{l1_data_path}")
    obj = s3_hook.get_key(l1_data_path, bucket_name=BUCKET_OUTPUT)
    df = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    
    # Sort by episode and time
    df = df.sort_values(['episode_id', 't_in_episode'])
    
    # 1. Calculate OHLC4 (price typical)
    logger.info("Calculating OHLC4...")
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # 2. Calculate log returns (within episode only - respect boundaries)
    logger.info("Calculating log returns (respecting episode boundaries)...")
    df['ret_log_1'] = np.nan
    
    episodes_processed = 0
    for episode_id in df['episode_id'].unique():
        mask = df['episode_id'] == episode_id
        episode_df = df[mask].sort_values('t_in_episode')
        
        # Calculate log returns (skip first bar of episode)
        closes = episode_df['close'].values
        if len(closes) > 1:
            returns = np.concatenate([[np.nan], np.log(closes[1:] / closes[:-1])])
        else:
            returns = [np.nan]
        
        df.loc[mask, 'ret_log_1'] = returns
        episodes_processed += 1
        
        if episodes_processed % 100 == 0:
            logger.info(f"  Processed {episodes_processed} episodes...")
    
    # 3. Calculate range in basis points
    logger.info("Calculating range metrics...")
    # IMPROVEMENT: Document range formula - must match L1's p95_range_bps calculation
    # Formula: (high - low) / close * 10000 
    # Note: If L1 uses (high/low - 1) * 10000, this needs alignment
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    
    # 4. Calculate quality flags
    logger.info("Calculating quality flags...")
    
    # Check for OHLC violations (should be 0 in accepted data)
    if 'ohlc_valid' in df.columns:
        df['is_ohlc_violation'] = ~df['ohlc_valid']
    else:
        # Calculate if not present
        df['is_ohlc_violation'] = (
            (df['open'] > df['high']) | 
            (df['open'] < df['low']) |
            (df['close'] > df['high']) | 
            (df['close'] < df['low']) |
            (df['low'] > df['high'])
        )
        df['ohlc_valid'] = ~df['is_ohlc_violation']
    
    # TWEAK 1: Name clarity - use is_zero_range with is_stale as alias
    # Check for zero-range bars (all OHLC values equal)
    df['is_zero_range'] = (
        (df['open'] == df['high']) & 
        (df['high'] == df['low']) & 
        (df['low'] == df['close'])
    )
    
    # Keep is_stale as alias for backward compatibility
    if 'is_stale' not in df.columns:
        df['is_stale'] = df['is_zero_range']  # Alias for compatibility
    
    # Valid bar flag
    df['is_valid_bar'] = (~df['is_stale']) & (df['ohlc_valid'])
    
    # Mark missing bars (for now all False, will update in mask_and_strict step)
    df['is_missing_bar'] = False
    
    # 5. Basic statistics
    ohlc_violations = df['is_ohlc_violation'].sum()
    stale_bars = df['is_stale'].sum()
    stale_rate = stale_bars / len(df) * 100 if len(df) > 0 else 0
    
    logger.info(f"\nBase features statistics:")
    logger.info(f"  OHLC violations: {ohlc_violations} (should be 0)")
    logger.info(f"  Stale bars: {stale_bars} ({stale_rate:.2f}%)")
    logger.info(f"  Valid bars: {df['is_valid_bar'].sum()} / {len(df)}")
    logger.info(f"  Returns calculated: {(~df['ret_log_1'].isna()).sum()}")
    logger.info(f"  Range bps mean: {df['range_bps'].mean():.2f}")
    
    # Save to MinIO for next task
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    run_id = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='run_id')
    temp_key = f'temp/l2_pipeline/{run_id}/base_features.parquet'
    s3_hook.load_bytes(
        buffer.getvalue(),
        key=temp_key,
        bucket_name=bucket_output,
        replace=True
    )
    
    context['task_instance'].xcom_push(key='base_features_path', value=temp_key)
    
    # Update metadata
    l2_metadata = context['task_instance'].xcom_pull(key='l2_metadata')
    l2_metadata['base_features_stats'] = {
        'ohlc_violations': int(ohlc_violations),
        'stale_bars': int(stale_bars),
        'stale_rate_pct': round(stale_rate, 2),
        'returns_calculated': int((~df['ret_log_1'].isna()).sum())
    }
    context['task_instance'].xcom_push(key='l2_metadata', value=l2_metadata)
    
    return len(df)

def apply_winsorization(**context):
    """Step 3: Apply winsorization to returns only (not prices)"""
    logger.info("="*80)
    logger.info("APPLYING WINSORIZATION (RETURNS ONLY)")
    logger.info("="*80)
    
    # Load data from MinIO
    ti = context['task_instance']
    base_features_path = ti.xcom_pull(task_ids='calculate_base_features', key='base_features_path')
    hod_path = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='hod_baseline_path')
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    
    # Get run_id BEFORE using it
    run_id = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='run_id')
    
    # Read base features
    obj = s3_hook.get_key(base_features_path, bucket_name=bucket_output)
    df = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    
    # Read HOD baseline
    obj = s3_hook.get_key(hod_path, bucket_name=bucket_output)
    hod_df = pd.read_csv(io.BytesIO(obj.get()['Body'].read()))
    
    # Create HOD lookup
    hod_stats = {}
    for _, row in hod_df.iterrows():
        hour = int(row['hour_cot'])
        hod_stats[hour] = {
            'median': row['median_ret_log_5m'],
            'mad': row['mad_ret_log_5m']
        }
    
    # Apply winsorization per hour using robust thresholds
    logger.info(f"Applying winsorization with {WINSOR_SIGMA} sigma thresholds...")
    
    df['ret_log_1_winsor'] = df['ret_log_1'].copy()
    df['ret_winsor_flag'] = False
    
    winsor_counts = {}
    
    for hour in hod_stats.keys():
        mask = (df['hour_cot'] == hour) & (~df['ret_log_1'].isna())
        
        if mask.sum() > 0:
            median_h = hod_stats[hour]['median']
            mad_h = hod_stats[hour]['mad']
            
            # AUDIT FIX: Apply frozen recipe with MAD floor and IQR fallback
            mad_scaled = 1.4826 * mad_h
            
            # Estimate IQR scale from returns of this hour (robust and no future info)
            hour_ret = df.loc[mask & (~df['ret_log_1'].isna()), 'ret_log_1']
            if len(hour_ret) > 10:  # Need enough data for IQR
                iqr_scaled = (hour_ret.quantile(0.75) - hour_ret.quantile(0.25)) / 1.349
            else:
                iqr_scaled = 0
            
            # AUDIT FIX: Hour-aware floor - 10.0 bps for hour 8, 8.5 bps for others
            if hour == 8:
                floor = 10.0/10000  # 10.0 bps floor for hour 8
            else:
                floor = 8.5/10000  # 8.5 bps floor for other hours
            scale_h = max(float(mad_scaled), float(iqr_scaled), floor)
            
            # Calculate thresholds with the robust scale
            thr_lo = median_h - WINSOR_SIGMA * scale_h
            thr_hi = median_h + WINSOR_SIGMA * scale_h
            
            # Apply clipping
            original_values = df.loc[mask, 'ret_log_1'].values
            clipped_values = np.clip(original_values, thr_lo, thr_hi)
            
            # Update values and flags
            df.loc[mask, 'ret_log_1_winsor'] = clipped_values
            df.loc[mask, 'ret_winsor_flag'] = (original_values != clipped_values)
            
            # Count winsorized
            n_winsorized = (original_values != clipped_values).sum()
            winsor_counts[hour] = {
                'total': int(mask.sum()),  # Convert numpy int64 to native int
                'winsorized': int(n_winsorized),
                'pct': round(float(n_winsorized / mask.sum() * 100), 2) if mask.sum() > 0 else 0,
                'thr_lo': float(thr_lo),
                'thr_hi': float(thr_hi)
            }
            
            logger.info(f"  Hour {hour}: {n_winsorized}/{mask.sum()} winsorized ({winsor_counts[hour]['pct']}%)")
    
    # Mark outliers
    df['is_outlier_ret'] = df['ret_winsor_flag']
    
    # Overall statistics
    total_returns = (~df['ret_log_1'].isna()).sum()
    total_winsorized = df['ret_winsor_flag'].sum()
    winsor_rate = total_winsorized / total_returns * 100 if total_returns > 0 else 0
    
    logger.info(f"\nWinsorization summary:")
    logger.info(f"  Total returns: {total_returns}")
    logger.info(f"  Total winsorized: {total_winsorized}")
    logger.info(f"  Winsorization rate: {winsor_rate:.2f}%")
    
    # Save winsorization parameters - FIX: Include complete recipe
    winsor_params = {
        'recipe': {
            'lookback_days': 120,  # Frozen recipe
            'mad_floor_bps': 6.5,
            'sigma': WINSOR_SIGMA,
            'epsilon': EPSILON,
            'use_iqr_fallback': True
        },
        'metrics': {
            'total_returns': int(total_returns),
            'total_winsorized': int(total_winsorized),
            'winsor_rate_pct': round(winsor_rate, 4)  # More precision
        },
        'per_hour_stats': winsor_counts
    }
    
    # Save winsor params to MinIO
    winsor_buffer = io.BytesIO()
    winsor_buffer.write(json.dumps(winsor_params, indent=2).encode())
    winsor_buffer.seek(0)
    
    winsor_params_key = f'temp/l2_pipeline/{run_id}/winsor_params.json'
    s3_hook.load_bytes(
        winsor_buffer.getvalue(),
        key=winsor_params_key,
        bucket_name=bucket_output,
        replace=True
    )
    context['task_instance'].xcom_push(key='winsor_params_path', value=winsor_params_key)
    
    # Check range outliers using HOD p95
    logger.info("\nChecking range outliers...")
    
    df['is_outlier_range'] = False
    range_outlier_counts = {}
    
    for _, row in hod_df.iterrows():
        hour = int(row['hour_cot'])
        if 'p95_range_bps' in row and pd.notna(row['p95_range_bps']):
            p95_range = row['p95_range_bps']
            mask = df['hour_cot'] == hour
            
            if mask.sum() > 0:
                outlier_mask = mask & (df['range_bps'] > p95_range)
                df.loc[outlier_mask, 'is_outlier_range'] = True
                
                n_outliers = outlier_mask.sum()
                range_outlier_counts[hour] = {
                    'total': int(mask.sum()),  # Convert numpy int64 to native int
                    'outliers': int(n_outliers),
                    'pct': round(float(n_outliers / mask.sum() * 100), 2) if mask.sum() > 0 else 0,
                    'p95_threshold': float(p95_range)
                }
    
    total_range_outliers = df['is_outlier_range'].sum()
    range_outlier_rate = total_range_outliers / len(df) * 100
    
    logger.info(f"  Total range outliers: {total_range_outliers} ({range_outlier_rate:.2f}%)")
    
    # Save to MinIO for next task
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    # run_id already retrieved at the beginning of the function
    temp_key = f'temp/l2_pipeline/{run_id}/winsorized.parquet'
    s3_hook.load_bytes(
        buffer.getvalue(),
        key=temp_key,
        bucket_name=bucket_output,
        replace=True
    )
    
    context['task_instance'].xcom_push(key='winsorized_path', value=temp_key)
    
    # Update metadata
    l2_metadata = context['task_instance'].xcom_pull(key='l2_metadata')
    l2_metadata['winsorization'] = winsor_params
    l2_metadata['range_outliers'] = {
        'total': int(total_range_outliers),
        'rate_pct': round(range_outlier_rate, 2),
        'per_hour': range_outlier_counts
    }
    context['task_instance'].xcom_push(key='l2_metadata', value=l2_metadata)
    
    return total_winsorized

def calculate_deseasonalization(**context):
    """Step 4: Apply deseasonalization using HOD baselines
    
    UPDATED: Now supports both causal and non-causal deseasonalization.
    Use CAUSAL_DESEASONALIZATION flag to enable causal mode.
    """
    logger.info("="*80)
    logger.info("CALCULATING DESEASONALIZATION")
    logger.info("="*80)
    
    # Check if causal mode is enabled
    CAUSAL_DESEASONALIZATION = Variable.get("L2_CAUSAL_DESEASONALIZATION", default_var="true").lower() == "true"
    
    if CAUSAL_DESEASONALIZATION:
        logger.info("🔄 USING CAUSAL DESEASONALIZATION (expanding window)")
        return calculate_deseasonalization_causal(**context)
    else:
        logger.info("⚠️ USING NON-CAUSAL DESEASONALIZATION (legacy mode)")
        return calculate_deseasonalization_noncausal(**context)

def calculate_deseasonalization_causal(**context):
    """Apply CAUSAL deseasonalization using expanding window approach"""
    from l2_causal_deseasonalization import (
        calculate_expanding_hod_stats,
        apply_causal_deseasonalization
    )
    
    logger.info("Starting causal deseasonalization...")
    
    # Load data
    ti = context['task_instance']
    winsorized_path = ti.xcom_pull(task_ids='apply_winsorization', key='winsorized_path')
    run_id = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='run_id')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    
    # Load winsorized data
    obj = s3_hook.get_key(winsorized_path, bucket_name=bucket_output)
    df = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    logger.info(f"Loaded {len(df)} rows for causal deseasonalization")
    
    # Calculate causal HOD statistics
    causal_hod_df = calculate_expanding_hod_stats(df, min_samples=30)
    
    # Apply causal deseasonalization
    df = apply_causal_deseasonalization(df, causal_hod_df)
    
    # Save causal HOD stats for audit
    buffer_causal = io.BytesIO()
    causal_hod_df.to_parquet(buffer_causal, index=False)
    buffer_causal.seek(0)
    
    causal_hod_key = f'temp/l2_pipeline/{run_id}/causal_hod_stats.parquet'
    s3_hook.load_bytes(
        buffer_causal.getvalue(),
        key=causal_hod_key,
        bucket_name=bucket_output,
        replace=True
    )
    
    # Continue with rest of processing (ATR, save, etc.)
    return complete_deseasonalization_processing(df, context)

def complete_deseasonalization_processing(df, context):
    """Complete the deseasonalization processing (ATR calculation, saving, etc.)"""
    ti = context['task_instance']
    run_id = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='run_id')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    
    # Calculate ATR (14-period, within episode)
    logger.info("Calculating ATR...")
    
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Reset TR at episode boundaries
    for episode_id in df['episode_id'].unique():
        mask = df['episode_id'] == episode_id
        episode_df = df[mask].sort_values('t_in_episode')
        
        # Set first TR to high-low only (no previous close)
        first_idx = episode_df.index[0]
        df.loc[first_idx, 'tr'] = df.loc[first_idx, 'high'] - df.loc[first_idx, 'low']
    
    # Calculate ATR using EMA within episodes
    df['atr_14'] = np.nan
    
    for episode_id in df['episode_id'].unique():
        mask = df['episode_id'] == episode_id
        episode_df = df[mask].sort_values('t_in_episode')
        
        if len(episode_df) >= 14:
            tr_values = episode_df['tr'].values
            atr = np.zeros_like(tr_values)
            atr[:14] = np.nan  # Not enough data for first 14 bars
            atr[13] = np.mean(tr_values[:14])  # Initial ATR
            
            for i in range(14, len(tr_values)):
                atr[i] = (atr[i-1] * 13 + tr_values[i]) / 14
            
            df.loc[mask, 'atr_14'] = atr
    
    # Normalize ATR to basis points
    df['atr_14_norm'] = df['atr_14'] / df['close'] * 10000
    
    # Save data to MinIO
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    temp_key = f'temp/l2_pipeline/{run_id}/deseasonalized.parquet'
    s3_hook.load_bytes(
        buffer.getvalue(),
        key=temp_key,
        bucket_name=bucket_output,
        replace=True
    )
    
    context['task_instance'].xcom_push(key='deseasonalized_path', value=temp_key)
    
    # Calculate and return statistics
    total_bars = len(df)
    deseasonalized_bars = int((~df['ret_deseason'].isna()).sum())
    
    return {
        'total_bars': total_bars,
        'deseasonalized_bars': deseasonalized_bars,
        'coverage_pct': float(deseasonalized_bars / total_bars * 100) if total_bars > 0 else 0.0
    }

def calculate_deseasonalization_noncausal(**context):
    """Original non-causal deseasonalization (for backward compatibility)"""
    
    # Load data from MinIO with robust XCom handling
    ti = context['task_instance']
    
    # CRITICAL FIX: Add comprehensive XCom debugging and null checks
    logger.info("Retrieving XCom values...")
    
    winsorized_path = ti.xcom_pull(task_ids='apply_winsorization', key='winsorized_path')
    logger.info(f"winsorized_path from XCom: {winsorized_path}")
    
    hod_path = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='hod_baseline_path')
    logger.info(f"hod_baseline_path from XCom: {hod_path}")
    
    run_id = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='run_id')
    logger.info(f"run_id from XCom: {run_id}")
    
    # CRITICAL FIX: Explicit null checks with detailed error messages
    if winsorized_path is None:
        error_msg = (
            "CRITICAL ERROR: winsorized_path is None! "
            "The apply_winsorization task may have failed or not completed. "
            "Check task dependencies and ensure apply_winsorization runs before this task."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if hod_path is None:
        error_msg = (
            "CRITICAL ERROR: hod_baseline_path is None! "
            "The load_and_validate_l1_data task may have failed. "
            "Check L1 data loading and HOD baseline generation."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if run_id is None:
        error_msg = (
            "CRITICAL ERROR: run_id is None! "
            "The load_and_validate_l1_data task may have failed. "
            "Check task execution order and dependencies."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("✅ All XCom values retrieved successfully")
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    
    # CRITICAL FIX: Add S3 existence check before reading
    logger.info(f"Checking if winsorized data exists: {winsorized_path}")
    try:
        if not s3_hook.check_for_key(winsorized_path, bucket_name=bucket_output):
            raise FileNotFoundError(f"Winsorized data not found at: s3://{bucket_output}/{winsorized_path}")
        logger.info("✅ Winsorized data exists")
    except Exception as e:
        logger.error(f"Failed to check winsorized data existence: {e}")
        raise
    
    logger.info(f"Checking if HOD baseline exists: {hod_path}")
    try:
        if not s3_hook.check_for_key(hod_path, bucket_name=bucket_output):
            raise FileNotFoundError(f"HOD baseline not found at: s3://{bucket_output}/{hod_path}")
        logger.info("✅ HOD baseline exists")
    except Exception as e:
        logger.error(f"Failed to check HOD baseline existence: {e}")
        raise
    
    # Read winsorized data with error handling
    try:
        logger.info("Loading winsorized data...")
        obj = s3_hook.get_key(winsorized_path, bucket_name=bucket_output)
        df = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
        logger.info(f"✅ Loaded {len(df)} rows of winsorized data")
    except Exception as e:
        logger.error(f"Failed to load winsorized data: {e}")
        raise
    
    # Read HOD baseline with error handling
    try:
        logger.info("Loading HOD baseline...")
        obj = s3_hook.get_key(hod_path, bucket_name=bucket_output)
        hod_df = pd.read_csv(io.BytesIO(obj.get()['Body'].read()))
        logger.info(f"✅ Loaded HOD baseline with {len(hod_df)} hours")
    except Exception as e:
        logger.error(f"Failed to load HOD baseline: {e}")
        raise
    
    # Create HOD lookup
    hod_stats = {}
    for _, row in hod_df.iterrows():
        hour = int(row['hour_cot'])
        hod_stats[hour] = {
            'median': row['median_ret_log_5m'],
            'mad': row['mad_ret_log_5m'],
            'p95_range': row.get('p95_range_bps', np.nan)
        }
    
    # TWEAK 4: Range formula alignment check with L1
    # Verify L1's p95_range_bps matches our calculation within tolerance
    logger.info("Verifying range formula alignment with L1...")
    
    # Calculate our p95 for each hour and compare
    for hour in hod_stats.keys():
        hour_mask = df['hour_cot'] == hour
        if hour_mask.sum() > 10:  # Need enough data
            our_p95 = df.loc[hour_mask, 'range_bps'].quantile(0.95)
            l1_p95 = hod_stats[hour]['p95_range']
            
            if pd.notna(l1_p95) and pd.notna(our_p95):
                relative_diff = abs(our_p95 - l1_p95) / l1_p95 if l1_p95 != 0 else 0
                if relative_diff > 0.05:  # 5% tolerance
                    logger.warning(f"  Hour {hour}: L1 p95={l1_p95:.2f}, L2 p95={our_p95:.2f} (diff={relative_diff:.1%})")
                    # Don't fail, just warn - formulas might legitimately differ slightly
    
    logger.info("✓ Range formula alignment check complete")
    
    # Apply deseasonalization
    logger.info("Applying robust z-score deseasonalization...")
    
    df['ret_deseason'] = np.nan
    df['range_norm'] = np.nan
    
    deseason_coverage = {}
    
    for hour in hod_stats.keys():
        mask = df['hour_cot'] == hour
        
        if mask.sum() > 0:
            median_h = hod_stats[hour]['median']
            mad_h = hod_stats[hour]['mad']
            p95_range_h = hod_stats[hour]['p95_range']
            
            # Deseasonalize returns - AUDIT FIX: Use frozen recipe with MAD floor and IQR
            returns_mask = mask & (~df['ret_log_1_winsor'].isna())
            if returns_mask.sum() > 0:
                # Apply same frozen recipe as winsorization
                mad_scaled = 1.4826 * mad_h
                
                # Estimate IQR from winsorized returns of this hour
                hour_ret = df.loc[returns_mask, 'ret_log_1_winsor']
                if len(hour_ret) > 10:
                    iqr_scaled = (hour_ret.quantile(0.75) - hour_ret.quantile(0.25)) / 1.349
                else:
                    iqr_scaled = 0
                
                # AUDIT FIX: Hour-aware floor - 10.0 bps for hour 8, 8.5 bps for others
                if hour == 8:
                    floor = 10.0/10000  # 10.0 bps floor for hour 8
                else:
                    floor = 8.5/10000  # 8.5 bps floor for other hours
                scale_h = max(float(mad_scaled), float(iqr_scaled), floor)
                
                # Apply robust z-score with proper scale
                df.loc[returns_mask, 'ret_deseason'] = \
                    (df.loc[returns_mask, 'ret_log_1_winsor'] - median_h) / (scale_h + EPSILON)
            
            # Normalize range
            if pd.notna(p95_range_h) and p95_range_h > EPSILON:
                df.loc[mask, 'range_norm'] = df.loc[mask, 'range_bps'] / (p95_range_h + EPSILON)
            
            # Track coverage
            n_deseasonalized = returns_mask.sum()
            deseason_coverage[hour] = {
                'total_bars': mask.sum(),
                'deseasonalized': int(n_deseasonalized),
                'coverage_pct': round(n_deseasonalized / mask.sum() * 100, 2) if mask.sum() > 0 else 0
            }
            
            logger.info(f"  Hour {hour}: {n_deseasonalized}/{mask.sum()} deseasonalized ({deseason_coverage[hour]['coverage_pct']}%)")
    
    # Overall statistics
    total_bars = len(df)
    deseasonalized_bars = int((~df['ret_deseason'].isna()).sum())
    overall_coverage = float(deseasonalized_bars / total_bars * 100) if total_bars > 0 else 0.0
    
    logger.info(f"\nDeseasonalization summary:")
    logger.info(f"  Total bars: {total_bars}")
    logger.info(f"  Deseasonalized: {deseasonalized_bars}")
    logger.info(f"  Overall coverage: {overall_coverage:.2f}%")
    
    if overall_coverage < 99:
        logger.warning(f"⚠️ Deseasonalization coverage below 99%: {overall_coverage:.2f}%")
    else:
        logger.info(f"✅ Deseasonalization coverage adequate: {overall_coverage:.2f}%")
    
    # Optional: Calculate ATR (14-period, within episode)
    logger.info("\nCalculating ATR...")
    
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Reset TR at episode boundaries
    for episode_id in df['episode_id'].unique():
        mask = df['episode_id'] == episode_id
        episode_df = df[mask].sort_values('t_in_episode')
        
        # Set first TR to high-low only (no previous close)
        first_idx = episode_df.index[0]
        df.loc[first_idx, 'tr'] = df.loc[first_idx, 'high'] - df.loc[first_idx, 'low']
    
    # Calculate ATR using EMA within episodes
    df['atr_14'] = np.nan
    
    for episode_id in df['episode_id'].unique():
        mask = df['episode_id'] == episode_id
        episode_df = df[mask].sort_values('t_in_episode')
        
        if len(episode_df) >= 14:
            tr_values = episode_df['tr'].values
            atr = np.zeros_like(tr_values)
            atr[:14] = np.nan  # Not enough data for first 14 bars
            atr[13] = np.mean(tr_values[:14])  # Initial ATR
            
            for i in range(14, len(tr_values)):
                atr[i] = (atr[i-1] * 13 + tr_values[i]) / 14
            
            df.loc[mask, 'atr_14'] = atr
    
    # Normalize ATR to basis points
    df['atr_14_norm'] = df['atr_14'] / df['close'] * 10000
    
    # Save enhanced HOD stats
    hod_stats_enhanced = []
    for hour, stats in hod_stats.items():
        hour_data = df[df['hour_cot'] == hour]
        
        enhanced_stats = {
            'hour_cot': hour,
            'median_ret_log_5m': stats['median'],
            'mad_ret_log_5m': stats['mad'],
            'p95_range_bps': stats['p95_range'],
            'count': len(hour_data),
            'atr_14_median': float(hour_data['atr_14'].median()) if len(hour_data) > 0 else np.nan
        }
        hod_stats_enhanced.append(enhanced_stats)
    
    hod_stats_df = pd.DataFrame(hod_stats_enhanced)
    # Save HOD stats to MinIO
    buffer_hod = io.BytesIO()
    hod_stats_df.to_parquet(buffer_hod, index=False)
    buffer_hod.seek(0)
    
    # run_id already retrieved at the beginning
    hod_stats_key = f'temp/l2_pipeline/{run_id}/hod_stats.parquet'
    s3_hook.load_bytes(buffer_hod.getvalue(), key=hod_stats_key, bucket_name=bucket_output, replace=True)
    
    context['task_instance'].xcom_push(key='hod_stats_path', value=hod_stats_key)
    
    # Save data to MinIO
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    # run_id already retrieved at the beginning
    temp_key = f'temp/l2_pipeline/{run_id}/deseasonalized.parquet'
    s3_hook.load_bytes(
        buffer.getvalue(),
        key=temp_key,
        bucket_name=bucket_output,
        replace=True
    )
    
    context['task_instance'].xcom_push(key='deseasonalized_path', value=temp_key)
    
    # Save normalization reference - FIX: Include complete recipe and actual metrics
    # Calculate winsor rate on the data we have
    winsor_rate_actual = df['ret_winsor_flag'].mean() if 'ret_winsor_flag' in df.columns else 0.0
    ret_deseason_std = df['ret_deseason'].std() if 'ret_deseason' in df.columns else 1.0
    
    normalization_ref = {
        'version': 'L2.v2.0',
        'recipe': {
            'lookback_days': 120,  # Frozen recipe
            'mad_floor_bps_default': 8.5,  # Updated floor
            'mad_floor_bps_hour_8': 10.0,  # Special floor for hour 8
            'winsor_sigma': 4.0,
            'use_iqr_fallback': True,
            'method': 'robust_z_score'
        },
        'metrics_on_data': {
            'winsor_rate': round(float(winsor_rate_actual), 5),
            'ret_deseason_std': float(ret_deseason_std),
            'coverage_pct': round(overall_coverage, 2)
        },
        'deseasonalization': {
            'method': 'robust_z_score',
            'formula': '(x - median) / max(MAD×1.4826, IQR/1.349, floor_bps)',  # Updated formula
            'epsilon': EPSILON,
            'coverage_pct': round(overall_coverage, 2),
            'per_hour_coverage': dict(deseason_coverage)  # Ensure it's a dict
        },
        'range_normalization': {
            'method': 'p95_scaling',
            'formula': 'range_bps / p95_range_bps',
            'range_bps_formula': '(high - low) / close * 10000',  # IMPROVEMENT: Document formula
            'note': 'Ensure L1 p95_range_bps uses same formula'
        },
        'atr': {
            'period': 14,
            'method': 'EMA',
            'episodes_with_atr': int((~df.groupby('episode_id')['atr_14'].apply(lambda x: x.isna().all())).sum())
        }
    }
    
    # Save normalization reference to MinIO
    norm_buffer = io.BytesIO()
    try:
        norm_json = json.dumps(normalization_ref, indent=2, default=str)
        norm_buffer.write(norm_json.encode())
        norm_buffer.seek(0)
    except Exception as e:
        logger.error(f"Failed to JSON encode normalization_ref: {e}")
        # Create minimal normalization ref
        norm_json = json.dumps({'status': 'success', 'coverage': overall_coverage})
        norm_buffer.write(norm_json.encode())
        norm_buffer.seek(0)
    
    norm_ref_key = f'temp/l2_pipeline/{run_id}/normalization_ref.json'
    s3_hook.load_bytes(
        norm_buffer.getvalue(),
        key=norm_ref_key,
        bucket_name=bucket_output,
        replace=True
    )
    
    # Push path to XCom for save_all_outputs
    context['task_instance'].xcom_push(key='norm_ref_path', value=norm_ref_key)
    
    # Update metadata
    l2_metadata = context['task_instance'].xcom_pull(key='l2_metadata')
    l2_metadata['deseasonalization'] = normalization_ref
    context['task_instance'].xcom_push(key='l2_metadata', value=l2_metadata)
    
    # Return simple success message
    logger.info(f"✅ Task completed: {deseasonalized_bars} bars deseasonalized")
    return "Success"

def create_strict_and_flex_datasets(**context):
    """Step 5: Create STRICT (60/60 only) and FLEX (with placeholders) datasets"""
    logger.info("="*80)
    logger.info("CREATING STRICT AND FLEX DATASETS")
    logger.info("="*80)
    
    # Load data from MinIO
    ti = context['task_instance']
    deseasonalized_path = ti.xcom_pull(task_ids='calculate_deseasonalization', key='deseasonalized_path')
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    
    obj = s3_hook.get_key(deseasonalized_path, bucket_name=bucket_output)
    df = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    
    # Analyze episode completeness
    episode_counts = df.groupby('episode_id').size()
    complete_episodes = episode_counts[episode_counts == 60].index.tolist()
    incomplete_episodes = episode_counts[episode_counts < 60].index.tolist()
    episodes_59 = episode_counts[episode_counts == 59].index.tolist()
    
    logger.info(f"Episode analysis:")
    logger.info(f"  Complete (60/60): {len(complete_episodes)}")
    logger.info(f"  With 59/60: {len(episodes_59)}")
    logger.info(f"  Other incomplete: {len(incomplete_episodes) - len(episodes_59)}")
    
    # 1. Create STRICT dataset (60/60 only)
    logger.info("\nCreating STRICT dataset (60/60 episodes only)...")
    
    df_strict = df[df['episode_id'].isin(complete_episodes)].copy()
    df_strict['is_missing_bar'] = False  # All bars are present in strict
    
    strict_episodes = df_strict['episode_id'].nunique()
    strict_rows = len(df_strict)
    
    logger.info(f"  STRICT: {strict_rows} rows from {strict_episodes} episodes")
    
    # 2. Create FLEX dataset (pad 59/60 episodes to 60)
    logger.info("\nCreating FLEX dataset (with placeholders for missing bars)...")
    
    flex_dfs = []
    
    # Add all complete episodes as-is
    for episode_id in complete_episodes:
        episode_df = df[df['episode_id'] == episode_id].copy()
        episode_df['is_missing_bar'] = False
        flex_dfs.append(episode_df)
    
    # Process 59-bar episodes
    placeholders_added = 0
    
    for episode_id in episodes_59:
        episode_df = df[df['episode_id'] == episode_id].copy()
        episode_df['is_missing_bar'] = False
        
        # Find missing slot
        expected_steps = set(range(60))
        actual_steps = set(episode_df['t_in_episode'].values)
        missing_steps = list(expected_steps - actual_steps)
        
        if len(missing_steps) == 1:
            missing_step = missing_steps[0]
            logger.info(f"  Episode {episode_id}: Adding placeholder at t={missing_step}")
            
            # Create placeholder row with all necessary columns
            placeholder = pd.DataFrame([{
                'episode_id': episode_id,
                't_in_episode': missing_step,
                'is_terminal': (missing_step == 59),
                'time_utc': pd.NaT,
                'time_cot': pd.NaT,
                'hour_cot': 8 + missing_step // 12,
                'minute_cot': (missing_step % 12) * 5,
                'open': np.nan,
                'high': np.nan,
                'low': np.nan,
                'close': np.nan,
                'ohlc4': np.nan,
                'ret_log_1': np.nan,
                'ret_log_1_winsor': np.nan,
                'range_bps': np.nan,
                'range_norm': np.nan,
                'ret_deseason': np.nan,
                'atr_14': np.nan,
                'atr_14_norm': np.nan,
                'tr': np.nan,
                'ohlc_valid': True,  # No violation, just missing
                'is_ohlc_violation': False,
                'is_stale': False,
                'is_valid_bar': False,  # Not valid since it's missing
                'is_missing_bar': True,  # Mark as missing
                'ret_winsor_flag': False,
                'is_outlier_ret': False,
                'is_outlier_range': False,
                'quality_flag': episode_df['quality_flag'].iloc[0] if 'quality_flag' in episode_df else 'WARN',
                'warn_reason': episode_df['warn_reason'].iloc[0] if 'warn_reason' in episode_df else 'MISSING_BAR'
            }])
            
            # Interpolate timestamps if possible
            if missing_step > 0 and missing_step < 59:
                # Get surrounding times
                before_idx = episode_df[episode_df['t_in_episode'] == missing_step - 1].index
                after_idx = episode_df[episode_df['t_in_episode'] == missing_step + 1].index
                
                if len(before_idx) > 0 and len(after_idx) > 0:
                    time_before = episode_df.loc[before_idx[0], 'time_utc']
                    time_after = episode_df.loc[after_idx[0], 'time_utc']
                    
                    # Interpolate (should be exactly 5 minutes after previous)
                    placeholder.loc[0, 'time_utc'] = time_before + pd.Timedelta(minutes=5)
                    # IMPROVEMENT: Use proper timezone conversion instead of hardcoded offset
                    if pd.notna(placeholder.loc[0, 'time_utc']):
                        placeholder.loc[0, 'time_cot'] = placeholder.loc[0, 'time_utc'].tz_convert('America/Bogota') if hasattr(placeholder.loc[0, 'time_utc'], 'tz_convert') else placeholder.loc[0, 'time_utc'] - pd.Timedelta(hours=5)
            elif missing_step == 0:
                # Missing first bar - extrapolate from second
                second_idx = episode_df[episode_df['t_in_episode'] == 1].index
                if len(second_idx) > 0:
                    time_second = episode_df.loc[second_idx[0], 'time_utc']
                    placeholder.loc[0, 'time_utc'] = time_second - pd.Timedelta(minutes=5)
                    # IMPROVEMENT: Use proper timezone conversion
                    if pd.notna(placeholder.loc[0, 'time_utc']):
                        placeholder.loc[0, 'time_cot'] = placeholder.loc[0, 'time_utc'].tz_convert('America/Bogota') if hasattr(placeholder.loc[0, 'time_utc'], 'tz_convert') else placeholder.loc[0, 'time_utc'] - pd.Timedelta(hours=5)
            
            # Combine and sort
            episode_df = pd.concat([episode_df, placeholder], ignore_index=True)
            episode_df = episode_df.sort_values('t_in_episode')
            placeholders_added += 1
        
        flex_dfs.append(episode_df)
    
    # Combine FLEX dataset
    df_flex = pd.concat(flex_dfs, ignore_index=True)
    df_flex = df_flex.sort_values(['episode_id', 't_in_episode'])
    
    flex_episodes = df_flex['episode_id'].nunique()
    flex_rows = len(df_flex)
    
    logger.info(f"\n  FLEX: {flex_rows} rows from {flex_episodes} episodes")
    logger.info(f"  Placeholders added: {placeholders_added}")
    
    # Verify counts
    expected_flex_rows = (len(complete_episodes) + len(episodes_59)) * 60
    if flex_rows != expected_flex_rows:
        logger.warning(f"⚠️ FLEX row count mismatch! Expected: {expected_flex_rows}, Got: {flex_rows}")
    else:
        logger.info(f"✅ FLEX row count correct: {flex_rows}")
    
    # Add critical columns for L3 before saving
    # Ensure proper column naming for L3 compatibility
    if 'ret_log_1' in df_strict.columns and 'ret_log_5m' not in df_strict.columns:
        df_strict['ret_log_5m'] = df_strict['ret_log_1']
        df_flex['ret_log_5m'] = df_flex['ret_log_1']
    
    if 'ret_log_norm' in df_strict.columns and 'ret_deseason' not in df_strict.columns:
        df_strict['ret_deseason'] = df_strict['ret_log_norm']
        df_flex['ret_deseason'] = df_flex['ret_log_norm']
    
    if 'range_bps_norm' in df_strict.columns and 'range_norm' not in df_strict.columns:
        df_strict['range_norm'] = df_strict['range_bps_norm']
        df_flex['range_norm'] = df_flex['range_bps_norm']
    
    # Add winsor_flag if not present - FIX: Map from ret_winsor_flag (the actual flag)
    if 'winsor_flag' not in df_strict.columns:
        df_strict['winsor_flag'] = df_strict['ret_winsor_flag'] if 'ret_winsor_flag' in df_strict.columns else False
        df_flex['winsor_flag'] = df_flex['ret_winsor_flag'] if 'ret_winsor_flag' in df_flex.columns else False
    
    # Add is_missing and is_pad flags
    df_strict['is_missing'] = False  # STRICT never has missing data
    df_strict['is_pad'] = False      # STRICT never has padding
    
    df_flex['is_missing'] = False    # Can be updated if needed
    df_flex['is_pad'] = df_flex['t_in_episode'].apply(lambda x: False)  # Will be updated for actual pads
    
    # Mark padded rows in FLEX if we added placeholders
    if placeholders_added > 0:
        # Identify padded rows (those with null values in key columns after interpolation)
        pad_mask = df_flex['close'].isna() | (df_flex.get('volume', 0) == 0)
        df_flex.loc[pad_mask, 'is_pad'] = True
    
    # Save datasets to MinIO
    run_id = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='run_id')
    
    # Save STRICT
    buffer = io.BytesIO()
    df_strict.to_parquet(buffer, index=False)
    buffer.seek(0)
    strict_key = f'temp/l2_pipeline/{run_id}/strict.parquet'
    s3_hook.load_bytes(buffer.getvalue(), key=strict_key, bucket_name=bucket_output, replace=True)
    
    # Save FLEX
    buffer = io.BytesIO()
    df_flex.to_parquet(buffer, index=False)
    buffer.seek(0)
    flex_key = f'temp/l2_pipeline/{run_id}/flex.parquet'
    s3_hook.load_bytes(buffer.getvalue(), key=flex_key, bucket_name=bucket_output, replace=True)
    
    context['task_instance'].xcom_push(key='strict_path', value=strict_key)
    context['task_instance'].xcom_push(key='flex_path', value=flex_key)
    
    # Calculate dataset statistics
    dataset_stats = {
        'strict': {
            'episodes': strict_episodes,
            'rows': strict_rows,
            'missing_bars': 0,
            'stale_rate_pct': round(df_strict['is_stale'].sum() / strict_rows * 100, 2) if strict_rows > 0 else 0,
            'winsor_rate_pct': round(df_strict['ret_winsor_flag'].sum() / strict_rows * 100, 2) if strict_rows > 0 else 0
        },
        'flex': {
            'episodes': flex_episodes,
            'rows': flex_rows,
            'missing_bars': int(df_flex['is_missing_bar'].sum()),
            'placeholders_added': placeholders_added,
            'stale_rate_pct': round(df_flex[~df_flex['is_missing_bar']]['is_stale'].sum() / 
                                   len(df_flex[~df_flex['is_missing_bar']]) * 100, 2) if len(df_flex[~df_flex['is_missing_bar']]) > 0 else 0,
            'winsor_rate_pct': round(df_flex[~df_flex['is_missing_bar']]['ret_winsor_flag'].sum() / 
                                    len(df_flex[~df_flex['is_missing_bar']]) * 100, 2) if len(df_flex[~df_flex['is_missing_bar']]) > 0 else 0
        },
        'rejected_episodes': len(incomplete_episodes) - len(episodes_59)
    }
    
    # Update metadata
    l2_metadata = context['task_instance'].xcom_pull(key='l2_metadata')
    l2_metadata['dataset_stats'] = dataset_stats
    context['task_instance'].xcom_push(key='l2_metadata', value=l2_metadata)
    
    logger.info(f"\n✅ Datasets created successfully")
    logger.info(f"  STRICT: {strict_episodes} episodes (60/60 only)")
    logger.info(f"  FLEX: {flex_episodes} episodes (60/60 + 59/60 padded)")
    
    return {'strict_rows': strict_rows, 'flex_rows': flex_rows}

def apply_l2_quality_gating(**context):
    """Step 6: Apply L2 quality gating rules"""
    logger.info("="*80)
    logger.info("APPLYING L2 QUALITY GATING")
    logger.info("="*80)
    
    # Load datasets from MinIO
    ti = context['task_instance']
    strict_path = ti.xcom_pull(task_ids='create_strict_and_flex_datasets', key='strict_path')
    flex_path = ti.xcom_pull(task_ids='create_strict_and_flex_datasets', key='flex_path')
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    
    # Load STRICT
    obj = s3_hook.get_key(strict_path, bucket_name=bucket_output)
    df_strict = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    
    # Load FLEX
    obj = s3_hook.get_key(flex_path, bucket_name=bucket_output)
    df_flex = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    
    # Quality checks for STRICT
    logger.info("Quality checks for STRICT dataset:")
    
    # 1. OHLC invariants (should be 0)
    ohlc_violations_strict = df_strict['is_ohlc_violation'].sum() if 'is_ohlc_violation' in df_strict else 0
    logger.info(f"  OHLC violations: {ohlc_violations_strict}")
    
    # 2. Stale rate
    stale_rate_strict = df_strict['is_stale'].sum() / len(df_strict) * 100
    logger.info(f"  Stale rate: {stale_rate_strict:.2f}%")
    
    # 3. Winsorization rate
    winsor_rate_strict = df_strict['ret_winsor_flag'].sum() / len(df_strict) * 100
    logger.info(f"  Winsorization rate: {winsor_rate_strict:.2f}%")
    
    # 4. Range outlier rate
    range_outlier_rate_strict = df_strict['is_outlier_range'].sum() / len(df_strict) * 100
    logger.info(f"  Range outlier rate: {range_outlier_rate_strict:.2f}%")
    
    # 5. IMPROVEMENT: Deseason variance check
    ret_deseason_std = df_strict['ret_deseason'].std() if 'ret_deseason' in df_strict.columns else 1.0
    logger.info(f"  Ret deseason std: {ret_deseason_std:.3f}")
    
    # 6. IMPROVEMENT: Winsor rate on returns only (exclude first bar per episode)
    returns_mask = df_strict['t_in_episode'] > 0  # Exclude first bar
    winsor_rate_returns = df_strict.loc[returns_mask, 'ret_winsor_flag'].sum() / returns_mask.sum() * 100 if returns_mask.sum() > 0 else 0
    logger.info(f"  Winsorization rate (returns only): {winsor_rate_returns:.2f}%")
    
    # Per-episode quality
    episode_quality = []
    
    for episode_id in df_strict['episode_id'].unique():
        episode_df = df_strict[df_strict['episode_id'] == episode_id]
        
        n_bars = len(episode_df)
        n_valid = episode_df['is_valid_bar'].sum()
        n_stale = episode_df['is_stale'].sum()
        n_winsor = episode_df['ret_winsor_flag'].sum()
        n_range_outlier = episode_df['is_outlier_range'].sum()
        
        # L2 quality flag
        if n_bars == 60 and n_valid == 60 and n_stale == 0:
            quality_flag_l2 = 'STRICT_PERFECT'
        elif n_bars == 60 and n_stale / n_bars <= 0.01 and n_winsor / n_bars <= 0.005:
            quality_flag_l2 = 'STRICT_OK'
        elif n_bars == 60 and n_stale / n_bars <= 0.02 and n_winsor / n_bars <= 0.01:
            quality_flag_l2 = 'STRICT_WARN'
        else:
            quality_flag_l2 = 'STRICT_MARGINAL'
        
        episode_quality.append({
            'episode_id': episode_id,
            'dataset': 'STRICT',
            'n_bars': n_bars,
            'n_valid': int(n_valid),
            'n_stale': int(n_stale),
            'stale_rate_pct': round(n_stale / n_bars * 100, 2),
            'n_winsor': int(n_winsor),
            'winsor_rate_pct': round(n_winsor / n_bars * 100, 2),
            'n_range_outlier': int(n_range_outlier),
            'quality_flag_l2': quality_flag_l2
        })
    
    # Quality checks for FLEX
    logger.info("\nQuality checks for FLEX dataset:")
    
    # Exclude missing bars from quality calculations
    df_flex_real = df_flex[~df_flex['is_missing_bar']]
    
    # 1. OHLC invariants
    ohlc_violations_flex = df_flex_real['is_ohlc_violation'].sum() if 'is_ohlc_violation' in df_flex_real else 0
    logger.info(f"  OHLC violations (excluding missing): {ohlc_violations_flex}")
    
    # 2. Stale rate
    stale_rate_flex = df_flex_real['is_stale'].sum() / len(df_flex_real) * 100 if len(df_flex_real) > 0 else 0
    logger.info(f"  Stale rate (excluding missing): {stale_rate_flex:.2f}%")
    
    # 3. Missing bars
    missing_bars = df_flex['is_missing_bar'].sum()
    missing_rate = missing_bars / len(df_flex) * 100
    logger.info(f"  Missing bars: {missing_bars} ({missing_rate:.2f}%)")
    
    # Overall gating decision
    logger.info("\n" + "="*40)
    logger.info("L2 GATING DECISION")
    logger.info("="*40)
    
    # IMPROVEMENT: Enhanced gates with deseason variance and returns-based winsor
    strict_pass = (
        ohlc_violations_strict == 0 and
        stale_rate_strict <= 2.0 and
        winsor_rate_returns <= 1.0 and  # Use returns-based rate
        range_outlier_rate_strict <= 1.0 and
        0.8 <= ret_deseason_std <= 1.2  # Hard gate on deseason variance
    )
    
    flex_pass = (
        ohlc_violations_flex == 0 and
        stale_rate_flex <= 2.0 and
        missing_rate <= 5.0  # Allow up to 5% missing bars in FLEX
    )
    
    if strict_pass:
        logger.info("✅ STRICT dataset: PASS")
    else:
        logger.warning("⚠️ STRICT dataset: WARN (quality issues detected)")
    
    if flex_pass:
        logger.info("✅ FLEX dataset: PASS")
    else:
        logger.warning("⚠️ FLEX dataset: WARN (quality issues detected)")
    
    # Save episode quality report to MinIO
    episode_quality_df = pd.DataFrame(episode_quality)
    buffer = io.BytesIO()
    episode_quality_df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    run_id = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='run_id')
    quality_key = f'temp/l2_pipeline/{run_id}/episode_quality.csv'
    s3_hook.load_bytes(buffer.getvalue(), key=quality_key, bucket_name=bucket_output, replace=True)
    
    context['task_instance'].xcom_push(key='episode_quality_path', value=quality_key)
    
    # Save gated datasets (pass through)
    context['task_instance'].xcom_push(key='strict_gated_path', value=strict_path)
    context['task_instance'].xcom_push(key='flex_gated_path', value=flex_path)
    
    # Create gating report
    gating_report = {
        'timestamp': datetime.now(pytz.UTC).isoformat(),
        'strict': {
            'pass': bool(strict_pass),  # Convert numpy bool to Python bool
            'ohlc_violations': int(ohlc_violations_strict),
            'stale_rate_pct': round(float(stale_rate_strict), 2),
            'winsor_rate_pct': round(float(winsor_rate_strict), 2),
            'range_outlier_rate_pct': round(float(range_outlier_rate_strict), 2),
            'episodes': int(len(df_strict['episode_id'].unique())),
            'rows': int(len(df_strict))
        },
        'flex': {
            'pass': bool(flex_pass),  # Convert numpy bool to Python bool
            'ohlc_violations': int(ohlc_violations_flex),
            'stale_rate_pct': round(float(stale_rate_flex), 2),
            'missing_bars': int(missing_bars),
            'missing_rate_pct': round(float(missing_rate), 2),
            'episodes': int(len(df_flex['episode_id'].unique())),
            'rows': int(len(df_flex))
        },
        'overall_pass': bool(strict_pass)  # Use STRICT as primary gate
    }
    
    # Save gating report to MinIO with safe JSON encoding
    gating_buffer = io.BytesIO()
    try:
        gating_json = json.dumps(gating_report, indent=2, default=str)
        gating_buffer.write(gating_json.encode())
    except Exception as e:
        logger.error(f"Failed to encode gating report: {e}")
        # Create minimal report
        minimal_report = {'status': 'error', 'message': str(e)}
        gating_buffer.write(json.dumps(minimal_report).encode())
    gating_buffer.seek(0)
    
    # Get run_id from XCom
    ti = context['task_instance']
    run_id = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='run_id')
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    
    gating_key = f'temp/l2_pipeline/{run_id}/gating_report.json'
    s3_hook.load_bytes(
        gating_buffer.getvalue(),
        key=gating_key,
        bucket_name=bucket_output,
        replace=True
    )
    
    # Update metadata
    l2_metadata = context['task_instance'].xcom_pull(key='l2_metadata')
    l2_metadata['gating'] = gating_report
    context['task_instance'].xcom_push(key='l2_metadata', value=l2_metadata)
    
    return gating_report

def generate_l2_reports(**context):
    """Step 7: Generate all L2 reports"""
    logger.info("="*80)
    logger.info("GENERATING L2 REPORTS")
    logger.info("="*80)
    
    execution_date = context['ds']
    l2_metadata = context['task_instance'].xcom_pull(key='l2_metadata')
    
    # Load data from MinIO
    ti = context['task_instance']
    strict_gated_path = ti.xcom_pull(task_ids='apply_l2_quality_gating', key='strict_gated_path')
    flex_gated_path = ti.xcom_pull(task_ids='apply_l2_quality_gating', key='flex_gated_path')
    episode_quality_path = ti.xcom_pull(task_ids='apply_l2_quality_gating', key='episode_quality_path')
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    
    # Load gated STRICT
    obj = s3_hook.get_key(strict_gated_path, bucket_name=bucket_output)
    df_strict = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    
    # Load gated FLEX
    obj = s3_hook.get_key(flex_gated_path, bucket_name=bucket_output)
    df_flex = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    
    # Load episode quality
    obj = s3_hook.get_key(episode_quality_path, bucket_name=bucket_output)
    episode_quality_df = pd.read_csv(io.BytesIO(obj.get()['Body'].read()))
    
    # 1. Quality Metrics Report
    logger.info("Generating quality_metrics.json...")
    
    quality_metrics = {
        'timestamp': datetime.now(pytz.UTC).isoformat(),
        'execution_date': execution_date,
        'episodes': {
            'strict': int(df_strict['episode_id'].nunique()),
            'flex': int(df_flex['episode_id'].nunique()),
            'rejected': l2_metadata['dataset_stats'].get('rejected_episodes', 0)
        },
        'bar_level': {
            'stale_rate_pct': round(df_strict['is_stale'].sum() / len(df_strict) * 100, 2),
            'ohlc_violations': int(df_strict['is_ohlc_violation'].sum()) if 'is_ohlc_violation' in df_strict else 0,
            'winsor_ret_pct': round(df_strict['ret_winsor_flag'].sum() / len(df_strict) * 100, 2),
            'range_outlier_pct': round(df_strict['is_outlier_range'].sum() / len(df_strict) * 100, 2)
        },
        'gaps': {
            'episodes_59of60': int(df_flex['is_missing_bar'].sum() / 60) if 'is_missing_bar' in df_flex else 0,
            'max_gap_bars': 1  # We only handle single missing bars
        },
        'statistics': {
            'ret_log_1': {
                'mean': float(df_strict['ret_log_1'].mean()),
                'std': float(df_strict['ret_log_1'].std()),
                'min': float(df_strict['ret_log_1'].min()),
                'max': float(df_strict['ret_log_1'].max())
            },
            'ret_deseason': {
                'mean': float(df_strict['ret_deseason'].mean()),
                'std': float(df_strict['ret_deseason'].std()),
                'min': float(df_strict['ret_deseason'].min()),
                'max': float(df_strict['ret_deseason'].max())
            },
            'range_bps': {
                'mean': float(df_strict['range_bps'].mean()),
                'p50': float(df_strict['range_bps'].quantile(0.5)),
                'p95': float(df_strict['range_bps'].quantile(0.95))
            }
        }
    }
    
    # Save quality metrics to MinIO buffer with safe JSON encoding
    quality_metrics_buffer = io.BytesIO()
    quality_metrics_buffer.write(json.dumps(quality_metrics, indent=2, default=str).encode())
    quality_metrics_buffer.seek(0)
    
    run_id = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='run_id')
    quality_metrics_key = f'temp/l2_pipeline/{run_id}/reports/quality_metrics.json'
    s3_hook.load_bytes(quality_metrics_buffer.getvalue(), key=quality_metrics_key, bucket_name=bucket_output, replace=True)
    context['task_instance'].xcom_push(key='quality_metrics_path', value=quality_metrics_key)
    
    # 2. Daily Quality Counts Report
    logger.info("Generating dq_counts_daily.csv...")
    
    dq_counts = []
    for episode_id in df_strict['episode_id'].unique():
        episode_df = df_strict[df_strict['episode_id'] == episode_id]
        
        dq_counts.append({
            'date_cot': episode_id,
            'bars_day': len(episode_df),
            'expected_day': 60,
            'completeness_day': len(episode_df) / 60,
            'stale_count': int(episode_df['is_stale'].sum()),
            'winsor_count': int(episode_df['ret_winsor_flag'].sum()),
            'valid_bars': int(episode_df['is_valid_bar'].sum()),
            'status_day': 'PASS' if len(episode_df) == 60 else 'INCOMPLETE'
        })
    
    dq_counts_df = pd.DataFrame(dq_counts)
    # Save DQ counts to MinIO buffer
    dq_counts_buffer = io.BytesIO()
    dq_counts_df.to_csv(dq_counts_buffer, index=False)
    dq_counts_buffer.seek(0)
    
    dq_counts_key = f'temp/l2_pipeline/{run_id}/reports/dq_counts_daily.csv'
    s3_hook.load_bytes(dq_counts_buffer.getvalue(), key=dq_counts_key, bucket_name=bucket_output, replace=True)
    context['task_instance'].xcom_push(key='dq_counts_path', value=dq_counts_key)
    
    # 3. Outlier Report
    logger.info("Generating outlier_report.csv...")
    
    # Find all outliers
    outliers = df_strict[
        df_strict['ret_winsor_flag'] | 
        df_strict['is_outlier_range']
    ].copy()
    
    if len(outliers) > 0:
        outlier_report = outliers[['episode_id', 't_in_episode', 'time_utc', 
                                   'ret_log_1', 'ret_log_1_winsor', 'ret_winsor_flag',
                                   'range_bps', 'is_outlier_range']].copy()
        outlier_report['outlier_type'] = outlier_report.apply(
            lambda x: 'RETURN' if x['ret_winsor_flag'] else ('RANGE' if x['is_outlier_range'] else 'UNKNOWN'),
            axis=1
        )
        # Save outlier report to MinIO buffer
        outlier_buffer = io.BytesIO()
        outlier_report.to_csv(outlier_buffer, index=False)
        outlier_buffer.seek(0)
        
        outlier_key = f'temp/l2_pipeline/{run_id}/reports/outlier_report.csv'
        s3_hook.load_bytes(outlier_buffer.getvalue(), key=outlier_key, bucket_name=bucket_output, replace=True)
        context['task_instance'].xcom_push(key='outlier_path', value=outlier_key)
        logger.info(f"  Found {len(outlier_report)} outliers")
    else:
        # Create empty report
        outlier_report = pd.DataFrame(columns=['episode_id', 't_in_episode', 'outlier_type'])
        outlier_buffer = io.BytesIO()
        outlier_report.to_csv(outlier_buffer, index=False)
        outlier_buffer.seek(0)
        
        outlier_key = f'temp/l2_pipeline/{run_id}/reports/outlier_report.csv'
        s3_hook.load_bytes(outlier_buffer.getvalue(), key=outlier_key, bucket_name=bucket_output, replace=True)
        context['task_instance'].xcom_push(key='outlier_path', value=outlier_key)
        logger.info("  No outliers found")
    
    # 4. Coverage Report
    logger.info("Generating coverage_report.json...")
    
    coverage_report = {
        'timestamp': datetime.now(pytz.UTC).isoformat(),
        'execution_date': execution_date,
        'input_coverage': {
            'l1_episodes': l2_metadata.get('l1_episodes_accepted', 0),
            'l1_rows': l2_metadata.get('l1_rows_accepted', 0),
            'l2_episodes_processed': l2_metadata.get('input_episodes', 0),
            'l2_rows_processed': l2_metadata.get('input_rows', 0)
        },
        'output_coverage': {
            'strict_episodes': int(df_strict['episode_id'].nunique()),
            'strict_rows': len(df_strict),
            'strict_coverage_pct': round(df_strict['episode_id'].nunique() / l2_metadata.get('input_episodes', 1) * 100, 2),
            'flex_episodes': int(df_flex['episode_id'].nunique()),
            'flex_rows': len(df_flex),
            'flex_placeholders': int(df_flex['is_missing_bar'].sum()) if 'is_missing_bar' in df_flex else 0
        },
        'deseasonalization_coverage': {
            'returns_deseasonalized': int((~df_strict['ret_deseason'].isna()).sum()),
            'coverage_pct': round((~df_strict['ret_deseason'].isna()).sum() / len(df_strict) * 100, 2)
        },
        'winsorization_coverage': {
            'returns_winsorized': int(df_strict['ret_winsor_flag'].sum()),
            'coverage_pct': round(df_strict['ret_winsor_flag'].sum() / len(df_strict) * 100, 2)
        }
    }
    
    # Save coverage report to MinIO buffer
    coverage_buffer = io.BytesIO()
    coverage_buffer.write(json.dumps(coverage_report, indent=2, default=str).encode())
    coverage_buffer.seek(0)
    
    coverage_key = f'temp/l2_pipeline/{run_id}/reports/coverage_report.json'
    s3_hook.load_bytes(coverage_buffer.getvalue(), key=coverage_key, bucket_name=bucket_output, replace=True)
    context['task_instance'].xcom_push(key='coverage_path', value=coverage_key)
    
    logger.info("\n✅ All reports generated successfully")
    
    return {
        'reports_generated': [
            'quality_metrics.json',
            'dq_counts_daily.csv',
            'outlier_report.csv',
            'coverage_report.json',
            'l2_episode_quality.csv'
        ]
    }

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def save_csv_formatted(df):
    """Format DataFrame for CSV and return as bytes with hash"""
    df_csv = df.copy()
    
    # Format price columns to 6 decimals
    price_cols = ['open', 'high', 'low', 'close', 'ohlc4']
    for col in price_cols:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].apply(lambda x: f'{x:.6f}' if pd.notna(x) else '')
    
    # Format float columns to 4 decimals
    float_cols = ['ret_log_1', 'ret_log_1_winsor', 'ret_deseason', 'range_bps', 
                  'range_norm', 'atr_14', 'atr_14_norm', 'tr']
    for col in float_cols:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')
    
    # Format timestamps
    if 'time_utc' in df_csv.columns:
        df_csv['time_utc'] = pd.to_datetime(df_csv['time_utc']).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    if 'time_cot' in df_csv.columns:
        df_csv['time_cot'] = pd.to_datetime(df_csv['time_cot']).dt.strftime('%Y-%m-%dT%H:%M:%S-05:00')
    
    # Save to buffer
    csv_buffer = io.StringIO()
    df_csv.to_csv(csv_buffer, index=False, lineterminator='\n')
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    # Return bytes and hash
    return csv_bytes, hashlib.sha256(csv_bytes).hexdigest()

def save_all_outputs(**context):
    """Save only critical L2 outputs needed by L3"""
    logger.info("="*80)
    logger.info("SAVING CRITICAL L2 OUTPUTS FOR L3")
    logger.info("="*80)
    
    # Import numpy at function level to ensure we have it
    import numpy as np
    
    execution_date = context['ds']
    l2_metadata = context['task_instance'].xcom_pull(key='l2_metadata')
    run_id = l2_metadata['run_id']
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    
    # Load datasets from MinIO
    ti = context['task_instance']
    strict_path = ti.xcom_pull(task_ids='create_strict_and_flex_datasets', key='strict_path')
    flex_path = ti.xcom_pull(task_ids='create_strict_and_flex_datasets', key='flex_path')
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    bucket_output = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='bucket_output') or BUCKET_OUTPUT
    
    # Load STRICT
    obj = s3_hook.get_key(strict_path, bucket_name=bucket_output)
    df_strict = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    
    # Load FLEX
    obj = s3_hook.get_key(flex_path, bucket_name=bucket_output)
    df_flex = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    
    # File hashes dictionary
    file_hashes = {}
    
    # 1. Save main datasets
    logger.info("Saving main datasets...")
    
    # IMPROVEMENT: Freeze column order for L3 compatibility
    l3_columns = [
        "episode_id", "t_in_episode", "time_utc", "time_cot",
        "open", "high", "low", "close", "ret_log_5m", "range_bps",
        "ret_deseason", "range_norm", "winsor_flag", "is_missing", "is_pad"
    ]
    
    # Ensure all required columns exist and reorder
    for col in l3_columns:
        if col not in df_strict.columns:
            if col == 'ret_log_5m':
                df_strict[col] = df_strict.get('ret_log_1', 0)
            elif col == 'is_missing':
                df_strict[col] = False
            elif col == 'is_pad':
                df_strict[col] = False
    
    # Reindex to frozen column order
    df_strict = df_strict.reindex(columns=[c for c in l3_columns if c in df_strict.columns])
    df_flex = df_flex.reindex(columns=[c for c in l3_columns if c in df_flex.columns])
    
    # Assert no pads in STRICT
    assert df_strict['is_pad'].sum() == 0, "ERROR: Found pads in STRICT dataset!"
    logger.info("✓ Confirmed: No pads in STRICT dataset")
    
    # TWEAK 2: Grid belt-and-suspenders - assert exact 300s intervals
    dt = (df_strict.sort_values(['episode_id', 't_in_episode'])
          .groupby('episode_id')['time_utc'].diff().dropna().dt.total_seconds())
    assert (dt == 300).all(), f"ERROR: STRICT grid must be exact 300s! Found: {dt[dt != 300].unique()}"
    logger.info("✓ Confirmed: All STRICT intervals are exactly 300 seconds")
    
    # TWEAK 5: NaN discipline - assert ≤0.5% NaN on required L3 columns
    req_columns = ["time_utc", "time_cot", "open", "high", "low", "close",
                   "ret_log_5m", "range_bps", "ret_deseason", "range_norm", "winsor_flag"]
    nan_rates = pd.Series({c: df_strict[c].isna().mean() for c in req_columns if c in df_strict.columns})
    max_nan_rate = nan_rates.max()
    
    if max_nan_rate > 0.005:
        worst_cols = nan_rates[nan_rates > 0.005].to_dict()
        error_msg = f"ERROR: NaN rate exceeds 0.5% in STRICT! Worst columns: {worst_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"✓ Confirmed: All required columns have ≤0.5% NaN (max={max_nan_rate:.3%})")
    
    # STRICT dataset
    strict_parquet_key = f"{DAG_ID}/market={MARKET}/timeframe={TIMEFRAME}/date={execution_date}/run_id={run_id}/data_premium_strict.parquet"
    
    table = pa.Table.from_pandas(df_strict)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression='snappy')
    strict_parquet_bytes = buffer.getvalue().to_pybytes()
    
    s3_hook.load_bytes(
        bytes_data=strict_parquet_bytes,
        key=strict_parquet_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    file_hashes['data_premium_strict.parquet'] = hashlib.sha256(strict_parquet_bytes).hexdigest()
    logger.info(f"✅ Saved STRICT parquet: {len(df_strict)} rows")
    
    # STRICT CSV
    strict_csv_bytes, strict_csv_hash = save_csv_formatted(df_strict)
    strict_csv_key = f"{DAG_ID}/market={MARKET}/timeframe={TIMEFRAME}/date={execution_date}/run_id={run_id}/data_premium_strict.csv"
    
    s3_hook.load_bytes(
        bytes_data=strict_csv_bytes,
        key=strict_csv_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    file_hashes['data_premium_strict.csv'] = strict_csv_hash
    logger.info(f"✅ Saved STRICT CSV")
    
    # FLEX dataset
    flex_parquet_key = f"{DAG_ID}/market={MARKET}/timeframe={TIMEFRAME}/date={execution_date}/run_id={run_id}/data_premium_flexible.parquet"
    
    table = pa.Table.from_pandas(df_flex)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression='snappy')
    flex_parquet_bytes = buffer.getvalue().to_pybytes()
    
    s3_hook.load_bytes(
        bytes_data=flex_parquet_bytes,
        key=flex_parquet_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    file_hashes['data_premium_flexible.parquet'] = hashlib.sha256(flex_parquet_bytes).hexdigest()
    logger.info(f"✅ Saved FLEX parquet: {len(df_flex)} rows")
    
    # FLEX CSV
    flex_csv_bytes, flex_csv_hash = save_csv_formatted(df_flex)
    flex_csv_key = f"{DAG_ID}/market={MARKET}/timeframe={TIMEFRAME}/date={execution_date}/run_id={run_id}/data_premium_flexible.csv"
    
    s3_hook.load_bytes(
        bytes_data=flex_csv_bytes,
        key=flex_csv_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    file_hashes['data_premium_flexible.csv'] = flex_csv_hash
    logger.info(f"✅ Saved FLEX CSV")
    
    # 2. Save statistics
    logger.info("\nSaving statistics...")
    
    # HOD stats
    hod_stats_parquet_key = f"{DAG_ID}/_statistics/date={execution_date}/hod_stats.parquet"
    
    # Load HOD stats from MinIO
    ti = context['task_instance']
    hod_stats_path = ti.xcom_pull(task_ids='calculate_deseasonalization', key='hod_stats_path')
    
    obj = s3_hook.get_key(hod_stats_path, bucket_name=bucket_output)
    hod_stats_df = pd.read_parquet(io.BytesIO(obj.get()['Body'].read()))
    
    table = pa.Table.from_pandas(hod_stats_df)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression='snappy')
    
    s3_hook.load_bytes(
        bytes_data=buffer.getvalue().to_pybytes(),
        key=hod_stats_parquet_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    logger.info(f"✅ Saved hod_stats.parquet")
    
    # Normalization reference - read from MinIO and add STRICT metrics
    norm_ref_key = f"{DAG_ID}/_statistics/date={execution_date}/normalization_ref.json"
    
    # Get the path from XCom and read from MinIO
    norm_ref_temp_path = ti.xcom_pull(task_ids='calculate_deseasonalization', key='norm_ref_path')
    if norm_ref_temp_path:
        obj = s3_hook.get_key(norm_ref_temp_path, bucket_name=bucket_output)
        norm_ref_content = obj.get()['Body'].read().decode()
        
        # AUDIT FIX: Add metrics computed on STRICT dataset only
        norm_ref = json.loads(norm_ref_content)
        
        # Compute metrics on STRICT
        winsor_rate_strict = float(df_strict['ret_winsor_flag'].mean()) if 'ret_winsor_flag' in df_strict.columns else 0.0
        ret_deseason_std_strict = float(df_strict['ret_deseason'].std()) if 'ret_deseason' in df_strict.columns else 1.0
        
        # TWEAK 3: Compute winsor rate on returns only (exclude first bar per episode)
        returns_mask = df_strict['t_in_episode'] > 0
        winsor_rate_returns = float(df_strict.loc[returns_mask, 'ret_winsor_flag'].mean()) if returns_mask.sum() > 0 else 0.0
        
        norm_ref['metrics_on_strict'] = {
            'winsor_rate': round(winsor_rate_strict, 5),  # Bar-level rate
            'winsor_rate_returns': round(winsor_rate_returns, 5),  # Returns-only rate (for gating)
            'ret_deseason_std': ret_deseason_std_strict,
            'episodes': int(df_strict['episode_id'].nunique()),
            'rows': int(len(df_strict))
        }
        
        # AUDIT FIX: Fill lineage gaps
        l2_metadata_full = ti.xcom_pull(task_ids='load_and_validate_l1_data', key='l2_metadata')
        if l2_metadata_full:
            norm_ref.setdefault('lineage', {})
            norm_ref['lineage']['l1_input_hash'] = l2_metadata_full.get('l1_input_hash', 'unknown')
            norm_ref['lineage']['source_l1_run_id'] = l2_metadata_full.get('source_l1_run_id', 'unknown')
        
        # Update the content with STRICT metrics
        norm_ref_content = json.dumps(norm_ref, indent=2, default=str)
    else:
        # Fallback: create minimal content if not found
        norm_ref_content = json.dumps({'status': 'missing', 'timestamp': datetime.now(pytz.UTC).isoformat()}, default=str)
    
    s3_hook.load_string(
        string_data=norm_ref_content,
        key=norm_ref_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    logger.info(f"✅ Saved normalization_ref.json with STRICT metrics")
    
    # 3. Save reports
    logger.info("\nSaving reports...")
    
    # Load reports from MinIO using XCom paths
    report_paths = [
        ('quality_metrics.json', ti.xcom_pull(task_ids='generate_l2_reports', key='quality_metrics_path')),
        ('dq_counts_daily.csv', ti.xcom_pull(task_ids='generate_l2_reports', key='dq_counts_path')),
        ('outlier_report.csv', ti.xcom_pull(task_ids='generate_l2_reports', key='outlier_path')),
        ('coverage_report.json', ti.xcom_pull(task_ids='generate_l2_reports', key='coverage_path'))
    ]
    
    for report_name, temp_path in report_paths:
        if temp_path:
            report_key = f"{DAG_ID}/_reports/date={execution_date}/{report_name}"
            
            # Read from MinIO temp location
            obj = s3_hook.get_key(temp_path, bucket_name=bucket_output)
            content = obj.get()['Body'].read()
            
            # Save to final location
            s3_hook.load_bytes(
                bytes_data=content,
                key=report_key,
                bucket_name=BUCKET_OUTPUT,
                replace=True
            )
            logger.info(f"✅ Saved {report_name}")
        else:
            logger.warning(f"⚠️ Report path not found for {report_name}")
    
    # 4. Save audit files
    logger.info("\nSaving audit files...")
    
    # HOD coverage
    hod_coverage_key = f"{DAG_ID}/_audit/date={execution_date}/join_hod_coverage.csv"
    
    # Calculate HOD coverage
    hod_coverage = []
    for hour in range(8, 13):
        hour_data = df_strict[df_strict['hour_cot'] == hour]
        deseasonalized = hour_data[~hour_data['ret_deseason'].isna()]
        
        hod_coverage.append({
            'hour_cot': hour,
            'total_bars': len(hour_data),
            'bars_with_deseason': len(deseasonalized),
            'coverage_pct': len(deseasonalized) / len(hour_data) * 100 if len(hour_data) > 0 else 0
        })
    
    coverage_df = pd.DataFrame(hod_coverage)
    coverage_csv = coverage_df.to_csv(index=False)
    
    s3_hook.load_string(
        string_data=coverage_csv,
        key=hod_coverage_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    logger.info(f"✅ Saved join_hod_coverage.csv")
    
    # Transform log
    transform_log_key = f"{DAG_ID}/_audit/date={execution_date}/transform_log.jsonl"
    
    transform_log = [
        {"step": 1, "action": "load_l1_data", "rows": l2_metadata['input_rows']},
        {"step": 2, "action": "calculate_base_features", "returns_calculated": l2_metadata['base_features_stats']['returns_calculated']},
        {"step": 3, "action": "apply_winsorization", "winsorized": l2_metadata['winsorization']['total_winsorized']},
        {"step": 4, "action": "deseasonalization", "coverage_pct": l2_metadata['deseasonalization']['deseasonalization']['coverage_pct']},
        {"step": 5, "action": "create_datasets", "strict_rows": len(df_strict), "flex_rows": len(df_flex)},
        {"step": 6, "action": "quality_gating", "pass": l2_metadata['gating']['overall_pass']}
    ]
    
    transform_log_content = '\n'.join([json.dumps(entry, default=str) for entry in transform_log])
    
    s3_hook.load_string(
        string_data=transform_log_content,
        key=transform_log_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    logger.info(f"✅ Saved transform_log.jsonl")
    
    # 5. Create comprehensive metadata
    logger.info("\nCreating metadata...")
    
    # Ultra-safe conversion of l2_metadata
    try:
        l2_metadata_clean = convert_numpy_types(l2_metadata)
    except Exception as e:
        logger.warning(f"Could not clean l2_metadata: {e}")
        l2_metadata_clean = {}
    
    final_metadata = {
        'dataset_version': DATASET_VERSION,
        'run_id': run_id,
        'source_l1_run_id': l2_metadata_clean.get('source_l1_run_id', 'unknown'),
        'date_cot': execution_date,
        'rows_strict': int(len(df_strict)),
        'rows_flex': int(len(df_flex)),
        'episodes_strict': int(df_strict['episode_id'].nunique()),
        'episodes_flex': int(df_flex['episode_id'].nunique()),
        'price_unit': 'COP per USD',
        'price_precision': 6,
        'created_ts': datetime.now(pytz.UTC).isoformat(),
        'hash_data_premium_strict': file_hashes['data_premium_strict.parquet'],
        'hash_data_premium_flex': file_hashes['data_premium_flexible.parquet'],
        'inputs': {
            'standardized_data_accepted': f"s3://{BUCKET_INPUT}/usdcop_m5__02_l1_standardize/consolidated/standardized_data_accepted.parquet",
            'hod_baseline': f"s3://{BUCKET_INPUT}/usdcop_m5__02_l1_standardize/consolidated/_statistics/hod_baseline.csv"
        },
        'processing': {
            'winsorization': l2_metadata_clean.get('winsorization', {}),
            'deseasonalization': l2_metadata_clean.get('deseasonalization', {}),
            'dataset_stats': l2_metadata_clean.get('dataset_stats', {}),
            'gating': l2_metadata_clean.get('gating', {})
        },
        'file_hashes': file_hashes
    }
    
    metadata_key = f"{DAG_ID}/_metadata/date={execution_date}/metadata.json"
    
    # Aggressive conversion to ensure JSON serialization
    try:
        # First attempt with recursive conversion
        final_metadata_clean = convert_numpy_types(final_metadata)
        metadata_json = json.dumps(final_metadata_clean, indent=2, default=str)
    except Exception as e:
        logger.warning(f"First JSON serialization attempt failed: {e}")
        try:
            # Second attempt: stringify everything that might be problematic
            import numpy as np
            
            def safe_json_convert(obj):
                """Ultra-safe JSON conversion"""
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(k): safe_json_convert(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [safe_json_convert(item) for item in obj]
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                else:
                    return obj
            
            final_metadata_safe = safe_json_convert(final_metadata)
            metadata_json = json.dumps(final_metadata_safe, indent=2, default=str)
        except Exception as e2:
            logger.error(f"Second JSON serialization attempt failed: {e2}")
            # Final fallback: create minimal metadata
            metadata_json = json.dumps({
                'status': 'completed_with_warnings',
                'run_id': str(run_id),
                'date': str(execution_date),
                'rows_strict': int(len(df_strict)),
                'rows_flex': int(len(df_flex)),
                'message': 'Full metadata serialization failed, basic info saved'
            }, indent=2)
    
    s3_hook.load_string(
        string_data=metadata_json,
        key=metadata_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    logger.info(f"✅ Saved metadata.json")
    
    # 6. Create READY signal
    logger.info("\nCreating READY signal...")
    
    ready_key = f"{DAG_ID}/_control/date={execution_date}/run_id={run_id}/READY"
    # Ensure all values are JSON serializable
    ready_data = {
        'timestamp': datetime.now(pytz.UTC).isoformat(),
        'run_id': str(run_id),
        'execution_date': str(execution_date),
        'dataset_version': str(DATASET_VERSION),
        'gating_pass': bool(l2_metadata_clean.get('gating', {}).get('overall_pass', False)),
        'outputs': {
            'strict': {'episodes': int(df_strict['episode_id'].nunique()), 'rows': int(len(df_strict))},
            'flex': {'episodes': int(df_flex['episode_id'].nunique()), 'rows': int(len(df_flex))}
        }
    }
    
    s3_hook.load_string(
        string_data=json.dumps(ready_data, indent=2, default=str),
        key=ready_key,
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    logger.info(f"✅ Created READY signal")
    
    # 7. Update latest symlinks
    logger.info("\nUpdating latest symlinks...")
    
    latest_files = [
        (strict_parquet_bytes, 'prepared_premium.parquet'),
        (flex_parquet_bytes, 'prepared_premium_flex.parquet')
    ]
    
    for content, filename in latest_files:
        latest_key = f"{DAG_ID}/latest/{filename}"
        s3_hook.load_bytes(
            bytes_data=content,
            key=latest_key,
            bucket_name=BUCKET_OUTPUT,
            replace=True
        )
    
    # Also copy key reports to latest - read from MinIO, not /tmp/
    # Get report paths from XCom
    quality_metrics_path = ti.xcom_pull(task_ids='generate_l2_reports', key='quality_metrics_path')
    dq_counts_path = ti.xcom_pull(task_ids='generate_l2_reports', key='dq_counts_path')
    
    # Copy quality metrics report
    if quality_metrics_path:
        try:
            obj = s3_hook.get_key(quality_metrics_path, bucket_name=bucket_output)
            quality_metrics_data = obj.get()['Body'].read()
            s3_hook.load_bytes(
                bytes_data=quality_metrics_data,
                key=f"{DAG_ID}/latest/quality_metrics.json",
                bucket_name=BUCKET_OUTPUT,
                replace=True
            )
            logger.info(f"✅ Copied quality_metrics.json to latest")
        except Exception as e:
            logger.warning(f"Could not copy quality_metrics.json: {e}")
    
    # Copy daily quality counts report
    if dq_counts_path:
        try:
            obj = s3_hook.get_key(dq_counts_path, bucket_name=bucket_output)
            dq_counts_data = obj.get()['Body'].read()
            s3_hook.load_bytes(
                bytes_data=dq_counts_data,
                key=f"{DAG_ID}/latest/l2_quality_daily.csv",
                bucket_name=BUCKET_OUTPUT,
                replace=True
            )
            logger.info(f"✅ Copied l2_quality_daily.csv to latest")
        except Exception as e:
            logger.warning(f"Could not copy l2_quality_daily.csv: {e}")
    
    # Copy outlier report if it exists
    outlier_path = ti.xcom_pull(task_ids='generate_l2_reports', key='outlier_path')
    if outlier_path:
        try:
            obj = s3_hook.get_key(outlier_path, bucket_name=bucket_output)
            outlier_data = obj.get()['Body'].read()
            s3_hook.load_bytes(
                bytes_data=outlier_data,
                key=f"{DAG_ID}/latest/outlier_report.csv",
                bucket_name=BUCKET_OUTPUT,
                replace=True
            )
            logger.info(f"✅ Copied outlier_report.csv to latest")
        except Exception as e:
            logger.warning(f"Could not copy outlier_report.csv: {e}")
    
    # Copy coverage report if it exists
    coverage_path = ti.xcom_pull(task_ids='generate_l2_reports', key='coverage_path')
    if coverage_path:
        try:
            obj = s3_hook.get_key(coverage_path, bucket_name=bucket_output)
            coverage_data = obj.get()['Body'].read()
            s3_hook.load_bytes(
                bytes_data=coverage_data,
                key=f"{DAG_ID}/latest/coverage_report.json",
                bucket_name=BUCKET_OUTPUT,
                replace=True
            )
            logger.info(f"✅ Copied coverage_report.json to latest")
        except Exception as e:
            logger.warning(f"Could not copy coverage_report.json: {e}")
    
    # Copy metadata to latest
    # Use the same safe metadata that was already created
    s3_hook.load_string(
        string_data=metadata_json,  # Reuse the already safe JSON
        key=f"{DAG_ID}/latest/metadata.json",
        bucket_name=BUCKET_OUTPUT,
        replace=True
    )
    
    logger.info(f"✅ Updated latest symlinks")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("L2 PREPARE COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ STRICT dataset: {len(df_strict)} rows, {df_strict['episode_id'].nunique()} episodes")
    logger.info(f"✅ FLEX dataset: {len(df_flex)} rows, {df_flex['episode_id'].nunique()} episodes")
    logger.info(f"✅ Quality gate: {'PASS' if l2_metadata_clean.get('gating', {}).get('overall_pass', False) else 'WARN'}")
    logger.info(f"📁 Location: s3://{BUCKET_OUTPUT}/{DAG_ID}/")
    logger.info("="*80)
    
    return {
        'files_saved': int(len(file_hashes)),
        'strict_rows': int(len(df_strict)),
        'flex_rows': int(len(df_flex)),
        'gating_pass': bool(l2_metadata_clean.get('gating', {}).get('overall_pass', False))
    }

# Create DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='L2 data preparation with deseasonalization and quality gating',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['l2', 'prepare', 'features'],
)

# Define tasks
load_task = PythonOperator(
    task_id='load_and_validate_l1_data',
    python_callable=load_and_validate_l1_data,
    dag=dag,
)

features_task = PythonOperator(
    task_id='calculate_base_features',
    python_callable=calculate_base_features,
    dag=dag,
)

winsor_task = PythonOperator(
    task_id='apply_winsorization',
    python_callable=apply_winsorization,
    dag=dag,
)

deseason_task = PythonOperator(
    task_id='calculate_deseasonalization',
    python_callable=calculate_deseasonalization,
    dag=dag,
)

datasets_task = PythonOperator(
    task_id='create_strict_and_flex_datasets',
    python_callable=create_strict_and_flex_datasets,
    dag=dag,
)

gating_task = PythonOperator(
    task_id='apply_l2_quality_gating',
    python_callable=apply_l2_quality_gating,
    dag=dag,
)

reports_task = PythonOperator(
    task_id='generate_l2_reports',
    python_callable=generate_l2_reports,
    dag=dag,
)

save_task = PythonOperator(
    task_id='save_all_outputs',
    python_callable=save_critical_l2_outputs if USE_CRITICAL_OUTPUTS else save_all_outputs,
    dag=dag,
)

# Set dependencies
load_task >> features_task >> winsor_task >> deseason_task
deseason_task >> datasets_task >> gating_task >> reports_task >> save_task