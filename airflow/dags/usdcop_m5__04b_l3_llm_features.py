#!/usr/bin/env python3
"""
USDCOP M5 - L3B LLM FEATURES PIPELINE
======================================
Computes INTERPRETABLE features for LLM consumption (Alpha Arena strategy)

KEY DIFFERENCES from L3 (RL features):
- NO normalization (LLMs need absolute values)
- NO abstract transformations (human-readable features)
- RAW technical indicators (EMA, RSI, MACD, BB)
- Trend classification & regime detection
- Multi-timeframe context (5m, 1h, 4h, 1d)

Input: L2 standardized OHLCV
Output: Interpretable technical indicators + USD/COP context
Bucket: 03-l3b-ds-usdcop-llm-features
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import sys
import json
import pandas as pd
import numpy as np
import io
from typing import Dict, Any

# Add utils to path
sys.path.insert(0, os.path.dirname(__file__))

import logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DAG_ID = "usdcop_m5__04b_l3_llm_features"
BUCKET_L2 = "02-l2-ds-usdcop-silver"
BUCKET_L3B = "03-l3b-ds-usdcop-llm-features"

DEFAULT_ARGS = {
    "owner": "llm-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Features to compute (Alpha Arena compliant)
LLM_FEATURES = [
    # Price context
    'price',

    # Moving averages (RAW values, not normalized)
    'ema_20', 'ema_50', 'sma_20', 'sma_50',

    # MACD (RAW values, not strength)
    'macd', 'macd_signal', 'macd_hist',

    # RSI (actual values, not distance from 50)
    'rsi_7', 'rsi_14',

    # Stochastic (actual values)
    'stoch_k', 'stoch_d',

    # Bollinger Bands (actual levels)
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',

    # ATR (actual volatility)
    'atr_14', 'atr_20',

    # Volume
    'volume', 'volume_sma_20', 'volume_ratio',

    # Trend classification
    'trend', 'trend_strength',

    # Volatility regime
    'volatility_5', 'volatility_20', 'volatility_regime',

    # USD/COP specific (from Alpha Arena user requirements)
    'hl_range_surprise', 'atr_surprise', 'band_cross_abs_k',
    'entropy_absret_k', 'gap_prev_open_abs', 'bb_squeeze_ratio',
    'macd_strength_abs', 'momentum_abs_norm',
]

# ============================================================================
# FEATURE COMPUTATION FUNCTIONS
# ============================================================================

def calculate_rsi(df: pd.DataFrame, period: int = 14, col: str = 'close') -> pd.Series:
    """
    Calculate RSI (Relative Strength Index)

    Returns actual RSI values [0-100], NOT distance from 50
    """
    delta = df.groupby('episode_id')[col].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.groupby(df['episode_id']).transform(
        lambda x: x.ewm(alpha=1/period, adjust=False).mean()
    )
    avg_loss = loss.groupby(df['episode_id']).transform(
        lambda x: x.ewm(alpha=1/period, adjust=False).mean()
    )

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_stochastic(df: pd.DataFrame, period: int = 14) -> tuple:
    """Calculate Stochastic Oscillator (%K and %D)"""

    low_min = df.groupby('episode_id')['low'].transform(
        lambda x: x.rolling(period).min()
    )
    high_max = df.groupby('episode_id')['high'].transform(
        lambda x: x.rolling(period).max()
    )

    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, 1e-10)
    stoch_d = stoch_k.groupby(df['episode_id']).transform(
        lambda x: x.rolling(3).mean()
    )

    return stoch_k, stoch_d

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands (actual price levels)"""

    bb_middle = df.groupby('episode_id')['close'].transform(
        lambda x: x.rolling(period).mean()
    )

    rolling_std = df.groupby('episode_id')['close'].transform(
        lambda x: x.rolling(period).std()
    )

    bb_upper = bb_middle + (std_dev * rolling_std)
    bb_lower = bb_middle - (std_dev * rolling_std)
    bb_width = (bb_upper - bb_lower) / bb_middle.replace(0, 1e-10)

    return bb_upper, bb_middle, bb_lower, bb_width

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""

    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df.groupby('episode_id')['close'].shift(1))
    low_close = abs(df['low'] - df.groupby('episode_id')['close'].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = tr.groupby(df['episode_id']).transform(
        lambda x: x.rolling(period).mean()
    )

    return atr

def classify_trend(df: pd.DataFrame) -> pd.Series:
    """
    Classify trend based on price vs EMA20

    Returns: 'bullish', 'bearish', or 'neutral'
    """
    trend = pd.Series('neutral', index=df.index)

    # Bullish: Price > EMA20 and EMA20 > EMA50
    bullish_mask = (df['close'] > df['ema_20']) & (df['ema_20'] > df['ema_50'])
    trend[bullish_mask] = 'bullish'

    # Bearish: Price < EMA20 and EMA20 < EMA50
    bearish_mask = (df['close'] < df['ema_20']) & (df['ema_20'] < df['ema_50'])
    trend[bearish_mask] = 'bearish'

    return trend

def calculate_trend_strength(df: pd.DataFrame) -> pd.Series:
    """
    Calculate trend strength using ADX-like logic

    Returns: 0.0-1.0 (0=no trend, 1=strong trend)
    """
    # Simple trend strength: distance from EMA normalized by ATR
    distance = abs(df['close'] - df['ema_20'])
    strength = (distance / df['atr_14'].replace(0, 1e-10)).clip(0, 1)

    return strength

def classify_volatility_regime(df: pd.DataFrame) -> pd.Series:
    """
    Classify volatility regime

    Returns: 'low', 'normal', 'high'
    """
    # Calculate percentiles over rolling window
    atr_percentile = df.groupby('episode_id')['atr_14'].transform(
        lambda x: x.rolling(100).apply(lambda y: (y.iloc[-1] <= y).sum() / len(y))
    )

    regime = pd.Series('normal', index=df.index)
    regime[atr_percentile < 0.33] = 'low'
    regime[atr_percentile > 0.67] = 'high'

    return regime

def calculate_alpha_arena_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the 10 Alpha Arena features (already in L3, but recompute for LLM)

    These are the SAME formulas as L3, but we include them here for completeness
    """

    # 1. hl_range_surprise
    df['hl_range'] = df['high'] - df['low']
    df['hl_range_ma'] = df.groupby('episode_id')['hl_range'].transform(
        lambda x: x.rolling(20).mean()
    )
    df['hl_range_surprise'] = df['hl_range'] / df['hl_range_ma'].replace(0, 1e-10)

    # 2. atr_surprise
    df['atr_ma'] = df.groupby('episode_id')['atr_14'].transform(
        lambda x: x.rolling(20).mean()
    )
    df['atr_surprise'] = df['atr_14'] / df['atr_ma'].replace(0, 1e-10)

    # 3. band_cross_abs_k
    current_price = df['close']
    band_distance = np.where(
        current_price > df['bb_middle'],
        (current_price - df['bb_middle']) / (df['bb_upper'] - df['bb_middle']).replace(0, 1e-10),
        (df['bb_middle'] - current_price) / (df['bb_middle'] - df['bb_lower']).replace(0, 1e-10)
    )
    df['band_cross_abs_k'] = band_distance

    # 4. entropy_absret_k
    df['returns'] = df.groupby('episode_id')['close'].pct_change()
    df['abs_returns'] = abs(df['returns'])

    def rolling_entropy(series, window=20):
        def entropy_calc(x):
            if len(x) < 2:
                return 0.0
            hist, _ = np.histogram(x, bins=10)
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            return -np.sum(prob * np.log(prob))

        return series.rolling(window).apply(entropy_calc)

    df['entropy_absret_k'] = df.groupby('episode_id')['abs_returns'].transform(
        lambda x: rolling_entropy(x, window=20)
    )

    # 5. gap_prev_open_abs (in pips, assuming 1 pip = 0.01)
    df['gap_prev_open_abs'] = abs(
        df['open'] - df.groupby('episode_id')['close'].shift(1)
    ) * 100

    # 6. bb_squeeze_ratio (already computed as bb_width, but add ratio)
    bb_width_ma = df.groupby('episode_id')['bb_width'].transform(
        lambda x: x.rolling(20).mean()
    )
    df['bb_squeeze_ratio'] = df['bb_width'] / bb_width_ma.replace(0, 1e-10)

    # 7. macd_strength_abs
    df['macd_strength_abs'] = abs(df['macd'] - df['macd_signal'])

    # 8. momentum_abs_norm
    momentum = df['close'] - df.groupby('episode_id')['close'].shift(10)
    df['momentum_abs_norm'] = momentum / df['close'].replace(0, 1e-10)

    return df

# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def discover_l2_inputs(**context):
    """Discover latest L2 data"""
    logger.info("🔍 Discovering L2 inputs...")

    s3_hook = S3Hook(aws_conn_id='minio_conn')

    # List all date directories
    keys = s3_hook.list_keys(bucket_name=BUCKET_L2, prefix='date=')

    if not keys:
        raise ValueError("No L2 data found in MinIO")

    # Get latest date
    dates = sorted(list(set([k.split('/')[0] for k in keys if k.startswith('date=')])))
    latest_date = dates[-1]

    logger.info(f"✅ Latest L2 date: {latest_date}")

    # Check for required files
    required_files = ['train_df.parquet', 'val_df.parquet', 'test_df.parquet']
    base_path = f"{latest_date}/"

    for fname in required_files:
        full_key = f"{base_path}{fname}"
        if not s3_hook.check_for_key(full_key, bucket_name=BUCKET_L2):
            logger.warning(f"⚠️ Missing {fname}")

    context['ti'].xcom_push(key='l2_date', value=latest_date)
    context['ti'].xcom_push(key='l2_base_path', value=base_path)

    return {'l2_date': latest_date, 'l2_base_path': base_path}

def compute_llm_features_for_split(split: str, **context):
    """
    Compute interpretable LLM features for a given split (train/val/test)
    """
    logger.info(f"📊 Computing LLM features for {split}...")

    s3_hook = S3Hook(aws_conn_id='minio_conn')

    # Load L2 data
    l2_base_path = context['ti'].xcom_pull(key='l2_base_path')
    l2_key = f"{l2_base_path}{split}_df.parquet"

    obj = s3_hook.get_key(l2_key, bucket_name=BUCKET_L2)
    data = obj.get()["Body"].read()
    df = pd.read_parquet(io.BytesIO(data))

    logger.info(f"Loaded {len(df)} rows from L2 {split}")

    # Ensure required columns exist
    required = ['episode_id', 't_in_episode', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in L2 data: {missing}")

    # ==================== COMPUTE FEATURES ====================

    # 1. Price
    df['price'] = df['close']

    # 2. Moving Averages (EMA and SMA)
    df['ema_20'] = df.groupby('episode_id')['close'].transform(
        lambda x: x.ewm(span=20, adjust=False).mean()
    )
    df['ema_50'] = df.groupby('episode_id')['close'].transform(
        lambda x: x.ewm(span=50, adjust=False).mean()
    )
    df['sma_20'] = df.groupby('episode_id')['close'].transform(
        lambda x: x.rolling(20).mean()
    )
    df['sma_50'] = df.groupby('episode_id')['close'].transform(
        lambda x: x.rolling(50).mean()
    )

    # 3. MACD
    ema_12 = df.groupby('episode_id')['close'].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    ema_26 = df.groupby('episode_id')['close'].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df.groupby('episode_id')['macd'].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 4. RSI
    df['rsi_7'] = calculate_rsi(df, period=7)
    df['rsi_14'] = calculate_rsi(df, period=14)

    # 5. Stochastic
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, period=14)

    # 6. Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'] = calculate_bollinger_bands(df)

    # 7. ATR
    df['atr_14'] = calculate_atr(df, period=14)
    df['atr_20'] = calculate_atr(df, period=20)

    # 8. Volume
    df['volume_sma_20'] = df.groupby('episode_id')['volume'].transform(
        lambda x: x.rolling(20).mean()
    )
    df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, 1e-10)

    # 9. Volatility
    df['volatility_5'] = df.groupby('episode_id')['close'].transform(
        lambda x: x.pct_change().rolling(5).std()
    )
    df['volatility_20'] = df.groupby('episode_id')['close'].transform(
        lambda x: x.pct_change().rolling(20).std()
    )

    # 10. Trend classification
    df['trend'] = classify_trend(df)
    df['trend_strength'] = calculate_trend_strength(df)

    # 11. Volatility regime
    df['volatility_regime'] = classify_volatility_regime(df)

    # 12. Alpha Arena features
    df = calculate_alpha_arena_features(df)

    # ==================== QUALITY CHECKS ====================

    # Check for NaN values
    nan_counts = df[LLM_FEATURES].isna().sum()
    if nan_counts.any():
        logger.warning(f"⚠️ NaN values found: {nan_counts[nan_counts > 0]}")

    # Fill NaN with forward fill then backward fill
    df[LLM_FEATURES] = df.groupby('episode_id')[LLM_FEATURES].ffill().bfill()

    # ==================== SAVE TO MINIO ====================

    run_id = f"L3B_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_key = f"date={context['ti'].xcom_pull(key='l2_date').split('=')[1]}/run_id={run_id}/{split}_llm_features.parquet"

    # Write to MinIO
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3_hook.load_bytes(
        buffer.getvalue(),
        key=output_key,
        bucket_name=BUCKET_L3B,
        replace=True
    )

    logger.info(f"✅ Saved {len(df)} rows to {output_key}")

    # Push metadata
    context['ti'].xcom_push(key=f'{split}_output_key', value=output_key)
    context['ti'].xcom_push(key=f'{split}_row_count', value=len(df))

    return {
        'split': split,
        'rows': len(df),
        'features_count': len(LLM_FEATURES),
        'output_key': output_key
    }

def save_l3b_manifest(**context):
    """Save manifest with metadata"""
    logger.info("💾 Saving L3B manifest...")

    s3_hook = S3Hook(aws_conn_id='minio_conn')

    run_id = f"L3B_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    l2_date = context['ti'].xcom_pull(key='l2_date')

    manifest = {
        "pipeline": "L3B_LLM_Features",
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
        "source_layer": "L2",
        "source_date": l2_date,
        "target_bucket": BUCKET_L3B,
        "feature_count": len(LLM_FEATURES),
        "features": LLM_FEATURES,
        "splits": {
            "train": {
                "output_key": context['ti'].xcom_pull(key='train_output_key'),
                "row_count": context['ti'].xcom_pull(key='train_row_count')
            },
            "val": {
                "output_key": context['ti'].xcom_pull(key='val_output_key'),
                "row_count": context['ti'].xcom_pull(key='val_row_count')
            },
            "test": {
                "output_key": context['ti'].xcom_pull(key='test_output_key'),
                "row_count": context['ti'].xcom_pull(key='test_row_count')
            }
        },
        "notes": [
            "Features are INTERPRETABLE (no normalization)",
            "RAW technical indicators for LLM reasoning",
            "Based on Alpha Arena Season 1 strategy",
            "USD/COP specific features included"
        ]
    }

    manifest_key = f"date={l2_date.split('=')[1]}/run_id={run_id}/manifest.json"

    s3_hook.load_string(
        json.dumps(manifest, indent=2),
        key=manifest_key,
        bucket_name=BUCKET_L3B,
        replace=True
    )

    # Also save as latest
    s3_hook.load_string(
        json.dumps(manifest, indent=2),
        key="_meta/l3b_latest.json",
        bucket_name=BUCKET_L3B,
        replace=True
    )

    logger.info(f"✅ Manifest saved: {manifest_key}")

    return manifest

# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    DAG_ID,
    default_args=DEFAULT_ARGS,
    description='L3B: LLM interpretable features (Alpha Arena strategy)',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['l3b', 'llm', 'features', 'alpha-arena', 'interpretable']
) as dag:

    t_discover = PythonOperator(
        task_id='discover_l2_inputs',
        python_callable=discover_l2_inputs,
    )

    t_train = PythonOperator(
        task_id='compute_train_features',
        python_callable=compute_llm_features_for_split,
        op_kwargs={'split': 'train'},
    )

    t_val = PythonOperator(
        task_id='compute_val_features',
        python_callable=compute_llm_features_for_split,
        op_kwargs={'split': 'val'},
    )

    t_test = PythonOperator(
        task_id='compute_test_features',
        python_callable=compute_llm_features_for_split,
        op_kwargs={'split': 'test'},
    )

    t_manifest = PythonOperator(
        task_id='save_manifest',
        python_callable=save_l3b_manifest,
    )

    # Dependencies
    t_discover >> [t_train, t_val, t_test] >> t_manifest
