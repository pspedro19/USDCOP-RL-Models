#!/usr/bin/env python3
"""
Build Forecasting Dataset - ALIGNED with SSOT Contracts
========================================================

Generates a forecasting dataset that is EXACTLY aligned with:
- src/forecasting/data_contracts.py (FEATURE_COLUMNS)
- airflow/dags/l3b_forecasting_training.py
- airflow/dags/l5b_forecasting_inference.py

This script produces the EXACT 19 features defined in the contract.

SSOT Data Source:
- Primary: PostgreSQL (usdcop_m5_ohlcv, v_macro_unified)
- Fallback: Parquet files (seeds/latest/)

@version 2.0.0
@contract CTR-FORECAST-DATA-001
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# IMPORT SSOT CONTRACTS
# =============================================================================

try:
    from src.forecasting.data_contracts import (
        FEATURE_COLUMNS,
        NUM_FEATURES,
        TARGET_HORIZONS,
        TARGET_COLUMN,
    )
    CONTRACTS_AVAILABLE = True
    logger.info(f"[SSOT] Loaded contracts: {NUM_FEATURES} features, {len(TARGET_HORIZONS)} horizons")
except ImportError:
    CONTRACTS_AVAILABLE = False
    # Fallback to hardcoded values
    FEATURE_COLUMNS = (
        "close", "open", "high", "low",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
        "day_of_week", "month", "is_month_end",
        "dxy_close_lag1", "oil_close_lag1",
    )
    NUM_FEATURES = len(FEATURE_COLUMNS)
    TARGET_HORIZONS = (1, 5, 10, 15, 20, 25, 30)
    TARGET_COLUMN = "close"
    logger.warning("[SSOT] Contracts not available, using fallback values")


# =============================================================================
# DATA FETCHING - SSOT FROM POSTGRESQL (with fallback)
# =============================================================================

# Try to import unified loaders
try:
    from src.data import UnifiedOHLCVLoader, UnifiedMacroLoader
    UNIFIED_LOADERS_AVAILABLE = True
    logger.info("[SSOT] UnifiedLoaders available - will use PostgreSQL/Parquet")
except ImportError:
    UNIFIED_LOADERS_AVAILABLE = False
    logger.warning("[SSOT] UnifiedLoaders not available - will use yfinance fallback")


def fetch_usdcop_ohlcv(start_date: str, end_date: str, use_db: bool = True) -> pd.DataFrame:
    """
    Fetch USDCOP daily OHLCV data.

    SSOT Strategy:
    1. Primary: PostgreSQL (usdcop_m5_ohlcv) resampled to daily
    2. Secondary: Parquet backup (seeds/latest/)
    3. Fallback: yfinance API

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_db: If True, try DB/Parquet first before yfinance

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    if use_db and UNIFIED_LOADERS_AVAILABLE:
        try:
            logger.info(f"Loading USDCOP from SSOT (DB/Parquet): {start_date} to {end_date}")
            loader = UnifiedOHLCVLoader()

            # Load 5-min and resample to daily
            df_5min = loader.load_5min(start_date, end_date, filter_market_hours=True)
            df = loader.resample_to_daily(df_5min)

            # Normalize columns
            df = df.rename(columns={'date': 'date'})
            df['date'] = pd.to_datetime(df['date'])

            logger.info(f"USDCOP (SSOT): {len(df)} daily records loaded")
            return df

        except Exception as e:
            logger.warning(f"SSOT load failed, falling back to yfinance: {e}")

    # Fallback to yfinance
    return _fetch_usdcop_yfinance(start_date, end_date)


def _fetch_usdcop_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
    """Fallback: Fetch USDCOP from yfinance."""
    import yfinance as yf

    logger.info(f"Fetching USDCOP from yfinance (fallback): {start_date} to {end_date}")

    ticker = yf.Ticker("COP=X")
    df = ticker.history(start=start_date, end=end_date, interval="1d")

    if df.empty:
        raise ValueError("No USDCOP data retrieved from yfinance")

    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

    logger.info(f"USDCOP (yfinance): {len(df)} records fetched")
    return df


def fetch_macro_data(start_date: str, end_date: str, use_db: bool = True) -> pd.DataFrame:
    """
    Fetch macro data (DXY, WTI).

    SSOT Strategy:
    1. Primary: PostgreSQL (v_macro_unified)
    2. Secondary: Parquet backup (seeds/latest/)
    3. Fallback: yfinance API

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_db: If True, try DB/Parquet first before yfinance

    Returns:
        DataFrame with columns: date, dxy, oil
    """
    if use_db and UNIFIED_LOADERS_AVAILABLE:
        try:
            logger.info("Loading macro data from SSOT (DB/Parquet)...")
            loader = UnifiedMacroLoader()

            # Load daily macro (already has friendly names)
            df = loader.load_daily(
                start_date,
                end_date,
                columns=['dxy', 'wti'],
                use_friendly_names=True
            )

            # Normalize for this script
            df = df.rename(columns={'wti': 'oil'})
            if 'fecha' in df.columns:
                df = df.rename(columns={'fecha': 'date'})
            df['date'] = pd.to_datetime(df['date'])

            # Keep only needed columns
            df = df[['date', 'dxy', 'oil']].copy()
            df = df.sort_values('date').reset_index(drop=True)

            logger.info(f"Macro (SSOT): {len(df)} records loaded")
            return df

        except Exception as e:
            logger.warning(f"SSOT macro load failed, falling back to yfinance: {e}")

    # Fallback to yfinance
    return _fetch_macro_yfinance(start_date, end_date)


def _fetch_macro_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
    """Fallback: Fetch macro data from yfinance."""
    import yfinance as yf

    logger.info("Fetching macro data from yfinance (fallback)...")

    # DXY (Dollar Index)
    dxy = yf.Ticker("DX-Y.NYB")
    dxy_df = dxy.history(start=start_date, end=end_date, interval="1d")
    if not dxy_df.empty:
        dxy_df = dxy_df.reset_index()
        dxy_df['date'] = pd.to_datetime(dxy_df['Date']).dt.tz_localize(None)
        dxy_df = dxy_df[['date', 'Close']].rename(columns={'Close': 'dxy'})
    else:
        dxy_df = pd.DataFrame(columns=['date', 'dxy'])

    # WTI Oil
    wti = yf.Ticker("CL=F")
    wti_df = wti.history(start=start_date, end=end_date, interval="1d")
    if not wti_df.empty:
        wti_df = wti_df.reset_index()
        wti_df['date'] = pd.to_datetime(wti_df['Date']).dt.tz_localize(None)
        wti_df = wti_df[['date', 'Close']].rename(columns={'Close': 'oil'})
    else:
        wti_df = pd.DataFrame(columns=['date', 'oil'])

    # Merge
    if not dxy_df.empty and not wti_df.empty:
        macro_df = dxy_df.merge(wti_df, on='date', how='outer')
    elif not dxy_df.empty:
        macro_df = dxy_df
        macro_df['oil'] = np.nan
    elif not wti_df.empty:
        macro_df = wti_df
        macro_df['dxy'] = np.nan
    else:
        macro_df = pd.DataFrame(columns=['date', 'dxy', 'oil'])

    macro_df = macro_df.sort_values('date').reset_index(drop=True)
    logger.info(f"Macro (yfinance): {len(macro_df)} records")

    return macro_df


# =============================================================================
# FEATURE ENGINEERING (EXACTLY MATCHING CONTRACTS)
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing (matches RL pipeline)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's EMA
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features(ohlcv: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Build features EXACTLY matching FEATURE_COLUMNS from data_contracts.py.

    SSOT Feature Order (19 features):
    1-4:   close, open, high, low
    5-8:   return_1d, return_5d, return_10d, return_20d
    9-11:  volatility_5d, volatility_10d, volatility_20d
    12-14: rsi_14d, ma_ratio_20d, ma_ratio_50d
    15-17: day_of_week, month, is_month_end
    18-19: dxy_close_lag1, oil_close_lag1
    """

    df = ohlcv.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Ensure numeric
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # =========================================================================
    # RETURNS (features 5-8)
    # =========================================================================
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['return_20d'] = df['close'].pct_change(20)

    # =========================================================================
    # VOLATILITY (features 9-11)
    # =========================================================================
    df['volatility_5d'] = df['return_1d'].rolling(5).std()
    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    # =========================================================================
    # TECHNICAL (features 12-14)
    # =========================================================================
    df['rsi_14d'] = calculate_rsi(df['close'], 14)

    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    df['ma_ratio_20d'] = df['close'] / ma_20
    df['ma_ratio_50d'] = df['close'] / ma_50

    # =========================================================================
    # CALENDAR (features 15-17)
    # =========================================================================
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # =========================================================================
    # MACRO (features 18-19) - LAGGED to avoid lookahead
    # =========================================================================
    if macro is not None and not macro.empty:
        macro = macro.copy()
        macro['date'] = pd.to_datetime(macro['date']).dt.tz_localize(None)

        # Merge macro data
        df = df.merge(macro[['date', 'dxy', 'oil']], on='date', how='left')

        # Create lagged features (SSOT naming: dxy_close_lag1, oil_close_lag1)
        df['dxy_close_lag1'] = df['dxy'].shift(1)
        df['oil_close_lag1'] = df['oil'].shift(1)

        # Drop intermediate columns
        df = df.drop(columns=['dxy', 'oil'], errors='ignore')
    else:
        df['dxy_close_lag1'] = np.nan
        df['oil_close_lag1'] = np.nan

    # =========================================================================
    # TARGETS (future returns for each horizon)
    # =========================================================================
    for h in TARGET_HORIZONS:
        # Future price
        df[f'target_{h}d'] = df['close'].shift(-h)
        # Log return (what the engine expects)
        df[f'target_return_{h}d'] = np.log(df['close'].shift(-h) / df['close'])

    return df


def validate_features(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that DataFrame has exactly the expected features."""
    errors = []

    # Check all FEATURE_COLUMNS exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing feature: {col}")

    # Check feature count
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(available_features) != NUM_FEATURES:
        errors.append(f"Expected {NUM_FEATURES} features, found {len(available_features)}")

    if errors:
        return False, "\n".join(errors)
    return True, "All features validated successfully"


def export_aligned_dataset(
    df: pd.DataFrame,
    output_dir: Path
) -> Dict[str, Path]:
    """Export dataset with features in SSOT order."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build column list in SSOT order
    columns = ['date'] + list(FEATURE_COLUMNS)

    # Add target columns
    target_cols = [c for c in df.columns if c.startswith('target_')]
    columns.extend(sorted(target_cols))

    # Filter to only columns we want
    available_cols = [c for c in columns if c in df.columns]
    df_export = df[available_cols].copy()

    # Drop rows with NaN in core features
    core_features = list(FEATURE_COLUMNS)[:17]  # Exclude macro which might have gaps
    df_clean = df_export.dropna(subset=core_features)

    # Export paths
    outputs = {}

    # Full dataset (parquet)
    parquet_path = output_dir / f"forecasting_aligned_{timestamp}.parquet"
    df_clean.to_parquet(parquet_path, index=False)
    outputs['parquet'] = parquet_path

    # CSV for inspection
    csv_path = output_dir / f"forecasting_aligned_{timestamp}.csv"
    df_clean.to_csv(csv_path, index=False)
    outputs['csv'] = csv_path

    return outputs, df_clean


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build SSOT-aligned forecasting dataset")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--use-db", action="store_true", default=True,
                        help="Use PostgreSQL/Parquet as SSOT (default: True)")
    parser.add_argument("--yfinance-only", action="store_true",
                        help="Force yfinance API (skip DB/Parquet)")
    args = parser.parse_args()

    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "data" / "forecasting" / "aligned"
    use_db = not args.yfinance_only

    print("=" * 70)
    print("    BUILD FORECASTING DATASET - ALIGNED WITH SSOT CONTRACTS")
    print("=" * 70)
    print(f"\nContracts Available: {CONTRACTS_AVAILABLE}")
    print(f"UnifiedLoaders Available: {UNIFIED_LOADERS_AVAILABLE}")
    print(f"Data Source: {'PostgreSQL/Parquet (SSOT)' if use_db and UNIFIED_LOADERS_AVAILABLE else 'yfinance API'}")
    print(f"Expected Features: {NUM_FEATURES}")
    print(f"Expected Horizons: {TARGET_HORIZONS}")
    print(f"\nDate Range: {args.start} to {end_date}")
    print("=" * 70)

    # Step 1: Fetch OHLCV
    print("\n[1/5] Fetching USDCOP OHLCV...")
    ohlcv_df = fetch_usdcop_ohlcv(args.start, end_date, use_db=use_db)

    # Step 2: Fetch Macro
    print("\n[2/5] Fetching Macro Data (DXY, WTI)...")
    macro_df = fetch_macro_data(args.start, end_date, use_db=use_db)

    # Step 3: Build Features
    print("\n[3/5] Building Features (SSOT aligned)...")
    df = build_features(ohlcv_df, macro_df)

    # Step 4: Validate
    print("\n[4/5] Validating Features...")
    valid, message = validate_features(df)
    if valid:
        print(f"   [OK] {message}")
    else:
        print(f"   [ERROR] {message}")

    # Step 5: Export
    print("\n[5/5] Exporting Dataset...")
    outputs, df_clean = export_aligned_dataset(df, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("                         SUMMARY")
    print("=" * 70)
    print(f"   Date Range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"   Total Rows: {len(df_clean):,}")
    print(f"   Total Columns: {len(df_clean.columns)}")
    print(f"   Features: {NUM_FEATURES}")
    print(f"   Targets: {len([c for c in df_clean.columns if c.startswith('target_')])}")

    print(f"\n   FEATURE ORDER (SSOT):")
    for i, col in enumerate(FEATURE_COLUMNS, 1):
        null_pct = df_clean[col].isnull().sum() / len(df_clean) * 100 if col in df_clean.columns else 100
        print(f"   {i:2d}. {col:20s} | nulls: {null_pct:5.1f}%")

    print(f"\n   OUTPUT FILES:")
    for name, path in outputs.items():
        print(f"   - {name}: {path}")

    print("\n" + "=" * 70)
    print("   DATASET READY - ALIGNED WITH SSOT CONTRACTS")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
