"""
USD/COP RL Trading System V17 - Dataset Generator
==================================================

Generates the complete V17 dataset with 28 market features following
the exact specification from README_V17_RAW_28F.md.

Features:
    - Loads most complete base dataset (RL_DS6_RAW_28F.csv or RL_DS5_MULTIFREQ_28F.csv)
    - Adds 4 Colombia-specific features:
        * vix_zscore
        * oil_above_60_flag
        * usdclp_ret_1d
        * banrep_intervention_proximity
    - Removes redundant features (momentum_5m, is_first_hour if needed)
    - Ensures exactly 28 market features (12 5min + 3 hourly + 13 daily)
    - Validates feature quality and correlations
    - Generates comprehensive validation report

Output:
    - data/pipeline/07_output/datasets_5min/RL_DS7_V17_28F.csv
    - data/pipeline/07_output/datasets_5min/VALIDATION_REPORT_V17_28F.txt

Author: Claude Code
Version: 1.0.0
Date: 2025-12-19
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Configuration for dataset generation."""

    # Input paths
    BASE_DATASET_1 = PROJECT_ROOT / "data/pipeline/07_output/datasets_5min/RL_DS6_RAW_28F.csv"
    BASE_DATASET_2 = PROJECT_ROOT / "data/pipeline/07_output/datasets_5min/RL_DS5_MULTIFREQ_28F.csv"

    USDCLP_DAILY = PROJECT_ROOT / "data/pipeline/01_sources/02_exchange_rates/fx_usdclp_CHL_d_USDCLP.csv"
    USDCOP_DAILY = PROJECT_ROOT / "data/pipeline/01_sources/16_usdcop_historical/USD_COP Historical Data.csv"

    # Output paths
    OUTPUT_DATASET = PROJECT_ROOT / "data/pipeline/07_output/datasets_5min/RL_DS7_V17_28F.csv"
    VALIDATION_REPORT = PROJECT_ROOT / "data/pipeline/07_output/datasets_5min/VALIDATION_REPORT_V17_28F.txt"

    # Feature specification (EXACTLY 28 market features)
    FEATURES_5MIN = [
        'log_ret_5m', 'log_ret_15m', 'log_ret_30m',
        'rsi_9', 'atr_pct', 'bb_position', 'adx_14', 'ema_cross',
        'high_low_range', 'session_progress', 'hour_sin', 'hour_cos'
    ]  # 12 features

    FEATURES_HOURLY = [
        'log_ret_1h', 'log_ret_4h', 'volatility_1h'
    ]  # 3 features

    FEATURES_DAILY = [
        'usdcop_ret_1d', 'usdcop_ret_5d', 'usdcop_volatility',
        'vix', 'vix_zscore',  # VIX + z-score
        'embi', 'dxy', 'brent', 'oil_above_60_flag',  # Brent + flag
        'rate_spread', 'usdmxn_ret_1d', 'usdclp_ret_1d',  # Spreads + LatAm
        'banrep_intervention_proximity'  # BanRep
    ]  # 13 features

    # Features to remove if present (redundant)
    FEATURES_TO_REMOVE = ['momentum_5m', 'is_first_hour']

    # Correlation threshold for redundancy check
    HIGH_CORRELATION_THRESHOLD = 0.95


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_base_dataset() -> pd.DataFrame:
    """
    Load the most complete base dataset available.

    Priority:
        1. RL_DS6_RAW_28F.csv (RAW values, no z-score - preferred)
        2. RL_DS5_MULTIFREQ_28F.csv (z-scored, fallback)

    Returns
    -------
    pd.DataFrame
        Base dataset with timestamp index
    """
    print("\n" + "="*70)
    print("LOADING BASE DATASET")
    print("="*70)

    # Try primary dataset
    if Config.BASE_DATASET_1.exists():
        print(f"\nLoading: {Config.BASE_DATASET_1.name}")
        df = pd.read_csv(Config.BASE_DATASET_1)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        print(f"✓ Loaded {len(df):,} rows from {Config.BASE_DATASET_1.name}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Features: {len(df.columns)}")
        return df

    # Try fallback dataset
    if Config.BASE_DATASET_2.exists():
        print(f"\nLoading: {Config.BASE_DATASET_2.name}")
        df = pd.read_csv(Config.BASE_DATASET_2)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        print(f"✓ Loaded {len(df):,} rows from {Config.BASE_DATASET_2.name}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Features: {len(df.columns)}")
        return df

    raise FileNotFoundError(
        f"Base dataset not found!\n"
        f"  Tried:\n"
        f"    - {Config.BASE_DATASET_1}\n"
        f"    - {Config.BASE_DATASET_2}"
    )


def load_usdclp_daily() -> pd.DataFrame:
    """
    Load USD/CLP daily data.

    Returns
    -------
    pd.DataFrame
        Daily USD/CLP with 'close' column, datetime index
    """
    print(f"\nLoading USD/CLP daily data...")

    if not Config.USDCLP_DAILY.exists():
        print(f"⚠ USD/CLP file not found: {Config.USDCLP_DAILY}")
        return pd.DataFrame()

    df = pd.read_csv(Config.USDCLP_DAILY, encoding='utf-8-sig')

    # Parse date and price
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

    # Parse price (handle format like "751.50")
    def parse_price(price_str):
        if pd.isna(price_str):
            return np.nan
        if isinstance(price_str, (int, float)):
            return float(price_str)
        return float(str(price_str).replace(',', ''))

    df['close'] = df['Price'].apply(parse_price)
    df = df.set_index('Date').sort_index()
    df = df[['close']].dropna()

    print(f"✓ Loaded {len(df)} daily USD/CLP prices")
    print(f"  Range: {df.index.min().date()} to {df.index.max().date()}")

    return df


def load_usdcop_daily() -> pd.DataFrame:
    """
    Load USD/COP daily data for BanRep calculation.

    Returns
    -------
    pd.DataFrame
        Daily USD/COP with 'close' column, datetime index
    """
    print(f"\nLoading USD/COP daily data...")

    if not Config.USDCOP_DAILY.exists():
        print(f"⚠ USD/COP file not found: {Config.USDCOP_DAILY}")
        return pd.DataFrame()

    df = pd.read_csv(Config.USDCOP_DAILY, encoding='utf-8-sig')

    # Parse date (format: DD/MM/YYYY in descending order)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

    # Parse price (format: "3,817.50")
    def parse_price(price_str):
        if pd.isna(price_str):
            return np.nan
        if isinstance(price_str, (int, float)):
            return float(price_str)
        return float(str(price_str).replace(',', ''))

    df['close'] = df['Price'].apply(parse_price)
    df = df.set_index('Date').sort_index()
    df = df[['close']].dropna()

    print(f"✓ Loaded {len(df)} daily USD/COP prices")
    print(f"  Range: {df.index.min().date()} to {df.index.max().date()}")

    return df


# ==============================================================================
# FEATURE CALCULATION
# ==============================================================================

def calculate_vix_zscore(df: pd.DataFrame) -> pd.Series:
    """
    Calculate VIX z-score using rolling window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'vix' column

    Returns
    -------
    pd.Series
        VIX z-score, clipped to [-4, 4]
    """
    if 'vix' not in df.columns:
        print("⚠ VIX column not found, returning zeros")
        return pd.Series(0.0, index=df.index)

    vix = df['vix']

    # Rolling z-score (252 trading days = 1 year at daily frequency)
    # For 5-min data, use forward-filled daily values
    daily_vix = vix.resample('D').last().ffill()

    # Calculate z-score on daily data
    window = 252
    mean = daily_vix.rolling(window, min_periods=20).mean()
    std = daily_vix.rolling(window, min_periods=20).std()
    z = ((daily_vix - mean) / (std + 1e-8)).clip(-4, 4)

    # Reindex to original frequency
    vix_zscore = z.reindex(df.index, method='ffill').fillna(0)

    print(f"✓ Calculated vix_zscore: mean={vix_zscore.mean():.4f}, std={vix_zscore.std():.4f}")

    return vix_zscore


def calculate_oil_above_60_flag(df: pd.DataFrame) -> pd.Series:
    """
    Calculate oil above $60 flag (correlation regime indicator).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'brent' column

    Returns
    -------
    pd.Series
        Binary flag: 1.0 if oil > 60, else 0.0
    """
    if 'brent' not in df.columns:
        print("⚠ Brent column not found, returning zeros")
        return pd.Series(0.0, index=df.index)

    flag = (df['brent'] > 60.0).astype(float)

    pct_above = (flag.sum() / len(flag)) * 100
    print(f"✓ Calculated oil_above_60_flag: {pct_above:.1f}% of time above $60")

    return flag


def calculate_usdclp_ret_1d(df: pd.DataFrame, usdclp_daily: pd.DataFrame) -> pd.Series:
    """
    Calculate USD/CLP 1-day return with proper daily lag.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataset with datetime index (5-min)
    usdclp_daily : pd.DataFrame
        Daily USD/CLP data with 'close' column

    Returns
    -------
    pd.Series
        Daily USD/CLP return, lagged 1 day, forward-filled to 5-min
    """
    if usdclp_daily.empty:
        print("⚠ USD/CLP daily data empty, returning zeros")
        return pd.Series(0.0, index=df.index)

    # Calculate daily return
    usdclp_daily['ret_1d'] = usdclp_daily['close'].pct_change(1)

    # Lag by 1 day (today uses yesterday's return)
    usdclp_daily_lag = usdclp_daily['ret_1d'].shift(1)

    # Reindex to 5-min frequency with forward-fill
    usdclp_ret = usdclp_daily_lag.reindex(df.index, method='ffill').fillna(0)

    print(f"✓ Calculated usdclp_ret_1d: mean={usdclp_ret.mean():.6f}, std={usdclp_ret.std():.6f}")

    return usdclp_ret


def calculate_banrep_intervention_proximity(
    df: pd.DataFrame,
    usdcop_daily: pd.DataFrame
) -> pd.Series:
    """
    Calculate BanRep intervention proximity using CORRECT method.

    Method (approved by expert):
        1. Use daily closes (external or resampled)
        2. Calculate MA20 on daily data
        3. Forward-fill to 5-min frequency
        4. Calculate deviation and normalize to [-1, +1]

    Parameters
    ----------
    df : pd.DataFrame
        Main dataset with 'close' column (5-min)
    usdcop_daily : pd.DataFrame
        Daily USD/COP closes (external source)

    Returns
    -------
    pd.Series
        BanRep intervention proximity [-1, +1]
    """
    if 'close' not in df.columns:
        print("⚠ Close column not found, returning zeros")
        return pd.Series(0.0, index=df.index)

    # STEP 1: Get daily closes
    if not usdcop_daily.empty:
        # Use external daily data (preferred)
        print("  Using external daily USD/COP closes for MA20 calculation")
        daily_close = usdcop_daily['close'].sort_index()
    else:
        # Fallback: resample from 5-min data
        print("  Resampling 5-min data to daily closes for MA20 calculation")
        daily_close = df['close'].resample('D').last().ffill()

    # STEP 2: Calculate MA20 on daily data (20 DAYS, not bars)
    ma20_daily = daily_close.rolling(20, min_periods=10).mean()

    # STEP 3: Forward-fill to 5-min frequency
    ma20_reindexed = ma20_daily.reindex(df.index, method='ffill')

    # STEP 4: Calculate deviation and normalize
    deviation = (df['close'] - ma20_reindexed) / (ma20_reindexed + 1e-8)
    proximity = (deviation / 0.05).clip(-1, 1).fillna(0)

    # Statistics
    pct_above_3 = (deviation.abs() > 0.03).sum() / len(deviation) * 100
    pct_trigger = (deviation.abs() > 0.05).sum() / len(deviation) * 100

    print(f"✓ Calculated banrep_intervention_proximity:")
    print(f"    Mean deviation: {deviation.mean():.4f} ({deviation.mean()*100:.2f}%)")
    print(f"    Std deviation: {deviation.std():.4f}")
    print(f"    Above 3% warning: {pct_above_3:.2f}% of time")
    print(f"    Above 5% trigger: {pct_trigger:.2f}% of time")

    return proximity


# ==============================================================================
# FEATURE VALIDATION
# ==============================================================================

def validate_feature_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    Validate that dataset has exactly 28 market features.

    Returns
    -------
    dict
        Feature counts by category
    """
    print("\n" + "="*70)
    print("VALIDATING FEATURE COUNTS")
    print("="*70)

    counts = {
        '5min': 0,
        'hourly': 0,
        'daily': 0,
        'total': 0
    }

    # Check 5-min features
    present_5min = [f for f in Config.FEATURES_5MIN if f in df.columns]
    counts['5min'] = len(present_5min)
    print(f"\n5-Minute Features: {counts['5min']}/{len(Config.FEATURES_5MIN)}")
    for f in Config.FEATURES_5MIN:
        status = "✓" if f in df.columns else "✗"
        print(f"  {status} {f}")

    # Check hourly features
    present_hourly = [f for f in Config.FEATURES_HOURLY if f in df.columns]
    counts['hourly'] = len(present_hourly)
    print(f"\nHourly Features: {counts['hourly']}/{len(Config.FEATURES_HOURLY)}")
    for f in Config.FEATURES_HOURLY:
        status = "✓" if f in df.columns else "✗"
        print(f"  {status} {f}")

    # Check daily features
    present_daily = [f for f in Config.FEATURES_DAILY if f in df.columns]
    counts['daily'] = len(present_daily)
    print(f"\nDaily Features: {counts['daily']}/{len(Config.FEATURES_DAILY)}")
    for f in Config.FEATURES_DAILY:
        status = "✓" if f in df.columns else "✗"
        print(f"  {status} {f}")

    counts['total'] = counts['5min'] + counts['hourly'] + counts['daily']

    print(f"\n{'='*70}")
    print(f"TOTAL MARKET FEATURES: {counts['total']}/28")
    print(f"  5-Minute:  {counts['5min']}/12")
    print(f"  Hourly:    {counts['hourly']}/3")
    print(f"  Daily:     {counts['daily']}/13")
    print(f"{'='*70}")

    if counts['total'] != 28:
        print(f"\n⚠ WARNING: Expected 28 features, got {counts['total']}")
    else:
        print(f"\n✓ SUCCESS: Exactly 28 market features present")

    return counts


def check_feature_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Check feature quality metrics.

    Returns
    -------
    dict
        Quality metrics
    """
    print("\n" + "="*70)
    print("CHECKING FEATURE QUALITY")
    print("="*70)

    all_features = Config.FEATURES_5MIN + Config.FEATURES_HOURLY + Config.FEATURES_DAILY
    present_features = [f for f in all_features if f in df.columns]

    quality = {
        'nan_counts': {},
        'inf_counts': {},
        'zero_variance': [],
        'ranges': {}
    }

    print(f"\nFeature Statistics:")
    print(f"{'Feature':<30} {'NaN %':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 88)

    for feature in present_features:
        series = df[feature]

        # NaN percentage
        nan_pct = (series.isna().sum() / len(series)) * 100
        quality['nan_counts'][feature] = nan_pct

        # Infinities
        inf_count = np.isinf(series).sum()
        quality['inf_counts'][feature] = inf_count

        # Zero variance
        if series.std() < 1e-10:
            quality['zero_variance'].append(feature)

        # Ranges
        quality['ranges'][feature] = {
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'std': series.std()
        }

        print(f"{feature:<30} {nan_pct:>8.2f}% {series.min():>12.6f} {series.max():>12.6f} "
              f"{series.mean():>12.6f} {series.std():>12.6f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Quality Summary:")
    total_nans = sum(quality['nan_counts'].values())
    total_infs = sum(quality['inf_counts'].values())
    print(f"  Total NaNs: {total_nans:.2f}%")
    print(f"  Total Infinities: {total_infs}")
    print(f"  Zero-variance features: {len(quality['zero_variance'])}")
    if quality['zero_variance']:
        print(f"    {quality['zero_variance']}")
    print(f"{'='*70}")

    return quality


def check_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for high correlations between features.

    Returns
    -------
    pd.DataFrame
        Pairs of highly correlated features (>0.95)
    """
    print("\n" + "="*70)
    print("CHECKING FEATURE CORRELATIONS")
    print("="*70)

    all_features = Config.FEATURES_5MIN + Config.FEATURES_HOURLY + Config.FEATURES_DAILY
    present_features = [f for f in all_features if f in df.columns]

    # Calculate correlation matrix
    corr_matrix = df[present_features].corr()

    # Find high correlations (exclude diagonal)
    high_corr = []
    for i in range(len(present_features)):
        for j in range(i+1, len(present_features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > Config.HIGH_CORRELATION_THRESHOLD:
                high_corr.append({
                    'feature_1': present_features[i],
                    'feature_2': present_features[j],
                    'correlation': corr
                })

    if high_corr:
        print(f"\nFound {len(high_corr)} high correlations (>{Config.HIGH_CORRELATION_THRESHOLD}):")
        print(f"{'Feature 1':<30} {'Feature 2':<30} {'Correlation':<12}")
        print("-" * 72)
        for item in high_corr:
            print(f"{item['feature_1']:<30} {item['feature_2']:<30} {item['correlation']:>12.4f}")
    else:
        print(f"\n✓ No high correlations found (threshold: {Config.HIGH_CORRELATION_THRESHOLD})")

    return pd.DataFrame(high_corr)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def generate_validation_report(
    df: pd.DataFrame,
    counts: Dict[str, int],
    quality: Dict[str, any],
    high_corr: pd.DataFrame,
    output_path: Path
):
    """Generate comprehensive validation report."""

    report = []
    report.append("="*70)
    report.append("USD/COP RL TRADING SYSTEM V17 - DATASET VALIDATION REPORT")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {Config.OUTPUT_DATASET.name}")
    report.append(f"\n{'='*70}")

    # Dataset info
    report.append("\n1. DATASET INFORMATION")
    report.append("-" * 70)
    report.append(f"Total rows: {len(df):,}")
    report.append(f"Date range: {df.index.min()} to {df.index.max()}")
    report.append(f"Trading days: {df.index.date.nunique():,}")
    report.append(f"File size: {Config.OUTPUT_DATASET.stat().st_size / 1024 / 1024:.2f} MB")

    # Feature counts
    report.append(f"\n2. FEATURE COUNTS")
    report.append("-" * 70)
    report.append(f"5-Minute features: {counts['5min']}/12")
    report.append(f"Hourly features: {counts['hourly']}/3")
    report.append(f"Daily features: {counts['daily']}/13")
    report.append(f"TOTAL MARKET FEATURES: {counts['total']}/28")

    if counts['total'] == 28:
        report.append("\n✓ SUCCESS: Exactly 28 market features present")
    else:
        report.append(f"\n⚠ WARNING: Expected 28 features, got {counts['total']}")

    # Feature quality
    report.append(f"\n3. FEATURE QUALITY")
    report.append("-" * 70)
    report.append(f"Features with NaNs: {sum(1 for v in quality['nan_counts'].values() if v > 0)}")
    report.append(f"Features with Infinities: {sum(1 for v in quality['inf_counts'].values() if v > 0)}")
    report.append(f"Zero-variance features: {len(quality['zero_variance'])}")

    # High correlations
    report.append(f"\n4. FEATURE CORRELATIONS")
    report.append("-" * 70)
    if len(high_corr) > 0:
        report.append(f"High correlations found: {len(high_corr)}")
        for _, row in high_corr.iterrows():
            report.append(f"  {row['feature_1']} <-> {row['feature_2']}: {row['correlation']:.4f}")
    else:
        report.append("✓ No high correlations (>0.95) detected")

    # Feature ranges
    report.append(f"\n5. FEATURE RANGES")
    report.append("-" * 70)
    report.append(f"{'Feature':<30} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    report.append("-" * 76)

    all_features = Config.FEATURES_5MIN + Config.FEATURES_HOURLY + Config.FEATURES_DAILY
    for feature in all_features:
        if feature in df.columns:
            r = quality['ranges'][feature]
            report.append(f"{feature:<30} {r['min']:>12.6f} {r['max']:>12.6f} "
                         f"{r['mean']:>12.6f} {r['std']:>12.6f}")

    report.append(f"\n{'='*70}")
    report.append("END OF VALIDATION REPORT")
    report.append("="*70)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\n✓ Validation report saved to: {output_path}")


def main():
    """Main dataset generation pipeline."""

    print("\n" + "="*70)
    print("USD/COP RL TRADING SYSTEM V17 - DATASET GENERATOR")
    print("="*70)
    print(f"\nScript: {Path(__file__).name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")

    try:
        # STEP 1: Load base dataset
        df = load_base_dataset()

        # STEP 2: Load additional data sources
        usdclp_daily = load_usdclp_daily()
        usdcop_daily = load_usdcop_daily()

        # STEP 3: Calculate Colombia-specific features
        print("\n" + "="*70)
        print("CALCULATING COLOMBIA-SPECIFIC FEATURES")
        print("="*70)

        df['vix_zscore'] = calculate_vix_zscore(df)
        df['oil_above_60_flag'] = calculate_oil_above_60_flag(df)
        df['usdclp_ret_1d'] = calculate_usdclp_ret_1d(df, usdclp_daily)
        df['banrep_intervention_proximity'] = calculate_banrep_intervention_proximity(df, usdcop_daily)

        # STEP 4: Remove redundant features
        print("\n" + "="*70)
        print("REMOVING REDUNDANT FEATURES")
        print("="*70)

        removed = []
        for feature in Config.FEATURES_TO_REMOVE:
            if feature in df.columns:
                df = df.drop(columns=[feature])
                removed.append(feature)
                print(f"✓ Removed: {feature}")

        if not removed:
            print("No redundant features found")

        # STEP 5: Select final feature set
        print("\n" + "="*70)
        print("SELECTING FINAL FEATURE SET")
        print("="*70)

        all_features = Config.FEATURES_5MIN + Config.FEATURES_HOURLY + Config.FEATURES_DAILY

        # Keep only specified features (+ timestamp will be added when saving)
        present_features = [f for f in all_features if f in df.columns]
        missing_features = [f for f in all_features if f not in df.columns]

        if missing_features:
            print(f"\n⚠ WARNING: Missing features:")
            for f in missing_features:
                print(f"  - {f}")

        df_final = df[present_features].copy()

        print(f"\n✓ Selected {len(present_features)} features")

        # STEP 6: Validate
        counts = validate_feature_counts(df_final)
        quality = check_feature_quality(df_final)
        high_corr = check_correlations(df_final)

        # STEP 7: Save dataset
        print("\n" + "="*70)
        print("SAVING DATASET")
        print("="*70)

        # Create output directory if needed
        Config.OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)

        # Reset index to have timestamp as column
        df_out = df_final.reset_index()

        # Save to CSV
        df_out.to_csv(Config.OUTPUT_DATASET, index=False)

        file_size_mb = Config.OUTPUT_DATASET.stat().st_size / 1024 / 1024
        print(f"\n✓ Dataset saved to: {Config.OUTPUT_DATASET}")
        print(f"  Rows: {len(df_out):,}")
        print(f"  Features: {len(df_final.columns)} market + 1 timestamp = {len(df_out.columns)} total")
        print(f"  Size: {file_size_mb:.2f} MB")

        # STEP 8: Generate validation report
        print("\n" + "="*70)
        print("GENERATING VALIDATION REPORT")
        print("="*70)

        generate_validation_report(df_final, counts, quality, high_corr, Config.VALIDATION_REPORT)

        # STEP 9: Final summary
        print("\n" + "="*70)
        print("DATASET GENERATION COMPLETE")
        print("="*70)
        print(f"\n✓ Output dataset: {Config.OUTPUT_DATASET}")
        print(f"✓ Validation report: {Config.VALIDATION_REPORT}")
        print(f"\nDataset summary:")
        print(f"  Rows: {len(df_out):,}")
        print(f"  Features: {counts['total']}/28 market features")
        print(f"  Date range: {df_final.index.min()} to {df_final.index.max()}")
        print(f"  Size: {file_size_mb:.2f} MB")

        if counts['total'] == 28:
            print(f"\n🎉 SUCCESS: Dataset V17 with exactly 28 market features is ready!")
        else:
            print(f"\n⚠ WARNING: Dataset has {counts['total']} features instead of 28")
            print(f"  Review validation report for details")

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
