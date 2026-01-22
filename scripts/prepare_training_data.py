#!/usr/bin/env python3
"""
Data Preparation Script for RL Training.

Prepares raw OHLCV and macro data for RL model training:
- Loads raw data from database or files
- Computes technical and macro features using CanonicalFeatureBuilder (SSOT)
- Splits data into train/val/test sets (temporal split)
- Saves processed datasets in parquet format

Part of the DVC pipeline (stage: prepare_data).

SSOT: Feature calculations delegated to src.feature_store.builders.CanonicalFeatureBuilder
Contract: CTR-FEAT-001

Usage:
    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --config params.yaml

Author: Trading Team
Date: 2026-01-16
Updated: 2026-01-18 (P1 Remediation - SSOT integration)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# SSOT imports for feature building
try:
    from src.feature_store.builders import CanonicalFeatureBuilder, BuilderContext
    from src.core.contracts.feature_contract import FEATURE_ORDER
    SSOT_AVAILABLE = True
except ImportError:
    SSOT_AVAILABLE = False
    print("WARNING: SSOT imports unavailable, using legacy feature calculation")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "prepare": {
        "lookback_days": 365,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "random_seed": 42,
        "features": [
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d"
        ]
    }
}


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw OHLCV and macro data.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (ohlcv_df, macro_df)
    """
    # Try to load from database first
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            database=os.environ.get("POSTGRES_DB", "trading"),
            user=os.environ.get("POSTGRES_USER", "trading"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
        )

        lookback_days = config.get("prepare", {}).get("lookback_days", 365)
        start_date = datetime.now() - timedelta(days=lookback_days)

        # Load OHLCV data
        ohlcv_query = f"""
            SELECT time, open, high, low, close, volume
            FROM usdcop_m5_ohlcv
            WHERE time >= '{start_date.isoformat()}'
            ORDER BY time
        """
        ohlcv_df = pd.read_sql(ohlcv_query, conn)
        ohlcv_df['time'] = pd.to_datetime(ohlcv_df['time'])

        # Load macro data
        macro_query = f"""
            SELECT date, dxy, vix, embi, brent, treasury_10y, usdmxn
            FROM macro_indicators_daily
            WHERE date >= '{start_date.date().isoformat()}'
            ORDER BY date
        """
        macro_df = pd.read_sql(macro_query, conn)
        macro_df['date'] = pd.to_datetime(macro_df['date'])

        conn.close()

        print(f"Loaded {len(ohlcv_df)} OHLCV rows from database")
        print(f"Loaded {len(macro_df)} macro rows from database")

        return ohlcv_df, macro_df

    except Exception as e:
        print(f"Could not load from database: {e}")
        print("Attempting to load from files...")

    # Fallback: Load from CSV files
    raw_dir = PROJECT_ROOT / "data" / "raw"

    ohlcv_path = raw_dir / "ohlcv.csv"
    if ohlcv_path.exists():
        ohlcv_df = pd.read_csv(ohlcv_path, parse_dates=["time"])
    else:
        # Try parquet
        ohlcv_path = raw_dir / "ohlcv.parquet"
        if ohlcv_path.exists():
            ohlcv_df = pd.read_parquet(ohlcv_path)
        else:
            raise FileNotFoundError(f"No OHLCV data found in {raw_dir}")

    macro_path = raw_dir / "macro.csv"
    if macro_path.exists():
        macro_df = pd.read_csv(macro_path, parse_dates=["date"])
    else:
        macro_path = raw_dir / "macro.parquet"
        if macro_path.exists():
            macro_df = pd.read_parquet(macro_path)
        else:
            raise FileNotFoundError(f"No macro data found in {raw_dir}")

    print(f"Loaded {len(ohlcv_df)} OHLCV rows from files")
    print(f"Loaded {len(macro_df)} macro rows from files")

    return ohlcv_df, macro_df


# =============================================================================
# Feature Engineering (LEGACY - Used only when SSOT unavailable)
# =============================================================================
# DEPRECATION NOTICE: These functions are kept for backwards compatibility only.
# The canonical implementation is in CanonicalFeatureBuilder.build_batch().
# All new code should use the SSOT builder directly.
#
# Issue 2.3 Remediation (2026-01-18):
# - Added deprecation warnings to all legacy functions
# - Functions delegate to SSOT calculators when available
# - Legacy code only runs as fallback when SSOT imports fail
# =============================================================================

import warnings

# Try to import SSOT calculators for delegation
try:
    from src.feature_store.calculators import (
        RSICalculator,
        ATRPercentCalculator,
        ADXCalculator,
    )
    from src.feature_store.calculators.base import FeatureSpec
    CALCULATORS_AVAILABLE = True
except ImportError:
    CALCULATORS_AVAILABLE = False


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns at different horizons."""
    df = df.copy()

    # 5-minute return (1 bar)
    df['log_ret_5m'] = np.log(df['close'] / df['close'].shift(1))

    # 1-hour return (12 bars)
    df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(12))

    # 4-hour return (48 bars)
    df['log_ret_4h'] = np.log(df['close'] / df['close'].shift(48))

    return df


def compute_rsi(close: pd.Series, period: int = 9) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing.

    DEPRECATED: Use src.feature_store.calculators.RSICalculator instead.
    This function delegates to SSOT when available.
    """
    warnings.warn(
        "compute_rsi() is deprecated. Use src.feature_store.calculators.RSICalculator instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Delegate to SSOT when available
    if CALCULATORS_AVAILABLE:
        spec = FeatureSpec(name=f"rsi_{period}", period=period)
        calculator = RSICalculator(spec)
        df = pd.DataFrame({'close': close})
        return calculator.calculate(df)

    # Legacy fallback
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Compute Average True Range as percentage of close.

    DEPRECATED: Use src.feature_store.calculators.ATRPercentCalculator instead.
    This function delegates to SSOT when available.
    """
    warnings.warn(
        "compute_atr() is deprecated. Use src.feature_store.calculators.ATRPercentCalculator instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Delegate to SSOT when available
    if CALCULATORS_AVAILABLE:
        spec = FeatureSpec(name=f"atr_pct_{period}", period=period)
        calculator = ATRPercentCalculator(spec)
        return calculator.calculate(df)

    # Legacy fallback
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, min_periods=period).mean()
    return (atr / df['close']) * 100


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index.

    DEPRECATED: Use src.feature_store.calculators.ADXCalculator instead.
    This function delegates to SSOT when available.
    """
    warnings.warn(
        "compute_adx() is deprecated. Use src.feature_store.calculators.ADXCalculator instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Delegate to SSOT when available
    if CALCULATORS_AVAILABLE:
        spec = FeatureSpec(name=f"adx_{period}", period=period)
        calculator = ADXCalculator(spec)
        return calculator.calculate(df)

    # Legacy fallback
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    true_range = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.ewm(alpha=1/period, min_periods=period).mean()


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical features."""
    df = df.copy()

    # Returns
    df = compute_returns(df)

    # RSI
    df['rsi_9'] = compute_rsi(df['close'], period=9)

    # ATR as percentage
    df['atr_pct'] = compute_atr(df, period=10)

    # ADX
    df['adx_14'] = compute_adx(df, period=14)

    return df


def compute_macro_features(ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro data with OHLCV and compute macro features.
    Uses forward-fill to align daily macro data with 5-minute bars.
    """
    df = ohlcv_df.copy()
    df['date'] = df['time'].dt.date

    macro_df = macro_df.copy()
    macro_df['date'] = pd.to_datetime(macro_df['date']).dt.date

    # Merge on date
    df = df.merge(macro_df, on='date', how='left')

    # Forward fill macro values
    macro_cols = ['dxy', 'vix', 'embi', 'brent', 'treasury_10y', 'usdmxn']
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Z-score features (using rolling statistics to avoid look-ahead bias)
    window = 252  # ~1 year of trading days * 60 bars/day

    if 'dxy' in df.columns:
        df['dxy_z'] = (df['dxy'] - df['dxy'].rolling(window, min_periods=20).mean()) / \
                      df['dxy'].rolling(window, min_periods=20).std()
        df['dxy_change_1d'] = df['dxy'].pct_change(60)  # 60 bars = 1 day

    if 'vix' in df.columns:
        df['vix_z'] = (df['vix'] - df['vix'].rolling(window, min_periods=20).mean()) / \
                      df['vix'].rolling(window, min_periods=20).std()

    if 'embi' in df.columns:
        df['embi_z'] = (df['embi'] - df['embi'].rolling(window, min_periods=20).mean()) / \
                       df['embi'].rolling(window, min_periods=20).std()

    if 'brent' in df.columns:
        df['brent_change_1d'] = df['brent'].pct_change(60)

    if 'treasury_10y' in df.columns:
        # Rate spread: Colombia implied rate (10%) - US 10Y
        df['rate_spread'] = 10.0 - df['treasury_10y']
        df['rate_spread'] = (df['rate_spread'] - df['rate_spread'].rolling(window, min_periods=20).mean()) / \
                            df['rate_spread'].rolling(window, min_periods=20).std()

    if 'usdmxn' in df.columns:
        df['usdmxn_change_1d'] = df['usdmxn'].pct_change(60)

    # Clip extreme values
    clip_features = ['dxy_z', 'vix_z', 'embi_z', 'rate_spread']
    for feat in clip_features:
        if feat in df.columns:
            df[feat] = df[feat].clip(-4, 4)

    clip_changes = ['dxy_change_1d', 'brent_change_1d', 'usdmxn_change_1d']
    for feat in clip_changes:
        if feat in df.columns:
            df[feat] = df[feat].clip(-0.1, 0.1)

    return df


# =============================================================================
# Data Splitting
# =============================================================================

def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets using temporal ordering.

    Args:
        df: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Data split:")
    print(f"  Train: {len(train_df)} rows ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} rows ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} rows ({len(test_df)/n*100:.1f}%)")

    return train_df, val_df, test_df


# =============================================================================
# Main Pipeline
# =============================================================================

def prepare_data_ssot(
    ohlcv_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Prepare data using CanonicalFeatureBuilder (SSOT).

    Args:
        ohlcv_df: OHLCV DataFrame
        macro_df: Macro indicators DataFrame
        config: Configuration dictionary

    Returns:
        DataFrame with all features calculated and normalized
    """
    print("[SSOT] Using CanonicalFeatureBuilder for feature calculation...")

    # Initialize SSOT builder for training context
    try:
        builder = CanonicalFeatureBuilder.for_training(config=config)
        print(f"[SSOT] Builder initialized: {builder.get_observation_dim()} features")
        print(f"[SSOT] Norm stats hash: {builder.get_norm_stats_hash()[:16]}...")
    except Exception as e:
        print(f"[SSOT] Warning: Could not initialize builder: {e}")
        print("[SSOT] Falling back to legacy feature calculation...")
        raise

    # Merge OHLCV and macro data
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df['date'] = ohlcv_df['time'].dt.date
    macro_df = macro_df.copy()
    macro_df['date'] = pd.to_datetime(macro_df['date']).dt.date

    merged = ohlcv_df.merge(macro_df, on='date', how='left')

    # Forward fill macro values
    macro_cols = ['dxy', 'vix', 'embi', 'brent', 'treasury_10y', 'usdmxn']
    for col in macro_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    # Use SSOT builder for batch feature calculation
    df = builder.build_batch(merged, normalize=True)

    # Ensure time column is preserved
    if 'time' not in df.columns and 'time' in merged.columns:
        df['time'] = merged['time']

    # Add OHLCV columns if not present
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns and col in merged.columns:
            df[col] = merged[col]

    print(f"[SSOT] Features calculated: {len(builder.get_feature_order())} features")

    return df


def prepare_data_legacy(
    ohlcv_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare data using legacy feature calculation (fallback).

    Args:
        ohlcv_df: OHLCV DataFrame
        macro_df: Macro indicators DataFrame

    Returns:
        DataFrame with all features calculated
    """
    print("[LEGACY] Using legacy feature calculation...")

    # Compute technical features
    df = compute_technical_features(ohlcv_df)

    # Compute macro features
    df = compute_macro_features(df, macro_df)

    return df


def prepare_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main data preparation pipeline.

    Uses CanonicalFeatureBuilder (SSOT) when available, falls back to
    legacy feature calculation if SSOT imports are unavailable.

    Args:
        config: Configuration dictionary

    Returns:
        Manifest with dataset information
    """
    prepare_config = config.get("prepare", DEFAULT_CONFIG["prepare"])

    print("=" * 60)
    print("Data Preparation Pipeline")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n[1/4] Loading raw data...")
    ohlcv_df, macro_df = load_raw_data(config)

    # Step 2: Compute features using SSOT or legacy
    print("\n[2/4] Computing features...")
    if SSOT_AVAILABLE:
        try:
            df = prepare_data_ssot(ohlcv_df, macro_df, config)
            print("[SSOT] Feature calculation complete.")
        except Exception as e:
            print(f"[SSOT] Failed: {e}")
            print("[SSOT] Falling back to legacy...")
            df = prepare_data_legacy(ohlcv_df, macro_df)
    else:
        df = prepare_data_legacy(ohlcv_df, macro_df)

    # Step 3: Drop rows with NaN values
    print("\n[3/4] Cleaning data...")
    feature_cols = prepare_config.get("features", [])
    initial_rows = len(df)

    # Keep only required columns
    keep_cols = ['time', 'open', 'high', 'low', 'close', 'volume'] + feature_cols
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].dropna()

    print(f"Dropped {initial_rows - len(df)} rows with NaN values")
    print(f"Final dataset: {len(df)} rows")

    # Step 4: Split data
    print("\n[4/4] Splitting data...")
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=prepare_config.get("train_ratio", 0.7),
        val_ratio=prepare_config.get("val_ratio", 0.15)
    )

    # Save to processed directory
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train_features.parquet"
    val_path = processed_dir / "val_features.parquet"
    test_path = processed_dir / "test_features.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved datasets to {processed_dir}")

    # Create manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "config": prepare_config,
        "datasets": {
            "train": {
                "path": str(train_path),
                "rows": len(train_df),
                "columns": list(train_df.columns),
                "date_range": {
                    "start": train_df['time'].min().isoformat() if 'time' in train_df.columns else None,
                    "end": train_df['time'].max().isoformat() if 'time' in train_df.columns else None,
                }
            },
            "val": {
                "path": str(val_path),
                "rows": len(val_df),
                "date_range": {
                    "start": val_df['time'].min().isoformat() if 'time' in val_df.columns else None,
                    "end": val_df['time'].max().isoformat() if 'time' in val_df.columns else None,
                }
            },
            "test": {
                "path": str(test_path),
                "rows": len(test_df),
                "date_range": {
                    "start": test_df['time'].min().isoformat() if 'time' in test_df.columns else None,
                    "end": test_df['time'].max().isoformat() if 'time' in test_df.columns else None,
                }
            }
        },
        "features": feature_cols,
        "total_rows": len(df),
    }

    manifest_path = processed_dir / "data_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"Saved manifest to {manifest_path}")

    return manifest


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare training data for RL models"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to configuration file"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        config = DEFAULT_CONFIG

    try:
        manifest = prepare_data(config)
        print("\n" + "=" * 60)
        print("Data preparation complete!")
        print(f"Total rows: {manifest['total_rows']}")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"Error during data preparation: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
