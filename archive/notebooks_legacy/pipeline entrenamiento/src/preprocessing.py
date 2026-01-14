#!/usr/bin/env python
"""
Data Preprocessing Module for USD/COP RL Training
==================================================

Implementa correcciones de calidad del dataset:
1. Warmup period removal (3,600 filas = 60 dias)
2. Redundant features removal
3. Per-fold normalization
4. Winsorization of extreme returns

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-20
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class DataQualityConfig:
    """Configuration for data preprocessing."""

    # Warmup period
    warmup_days: int = 60
    bars_per_day: int = 60

    # Features to drop (redundant)
    features_to_drop: List[str] = None

    # Features to normalize per fold
    features_to_normalize: List[str] = None

    # Winsorization
    winsorize_percentile: float = 0.01
    return_columns: List[str] = None

    def __post_init__(self):
        if self.features_to_drop is None:
            self.features_to_drop = [
                'hour_sin',           # Redundante con session_progress (r=-0.975)
                'hour_cos',           # Redundante con session_progress (r=-0.943)
                'oil_above_60_flag',  # Redundante con rate_spread (r=-0.911)
                'vix_zscore',         # 68% zeros (bug en generacion por hora), usar vix_z
                'high_low_range',     # 93.5% zeros (solo datos desde 2023-06)
            ]

        if self.features_to_normalize is None:
            self.features_to_normalize = [
                'vix_z',              # CORREGIDO: usar vix_z (0.1% zeros) no vix_zscore
                'embi_z',
                'dxy_z',
                'usdcop_volatility',
                'rate_spread',
            ]

        if self.return_columns is None:
            self.return_columns = [
                'log_ret_5m',
                'log_ret_15m',
                'log_ret_30m',
                'log_ret_1h',
                'log_ret_4h',
                'usdcop_ret_1d',
                'usdcop_ret_5d',
                'brent_change_1d',
                'usdmxn_change_1d',
                'usdclp_ret_1d',
            ]


# ==============================================================================
# FEATURES DEFINITION (POST-CLEANUP)
# ==============================================================================

# Features originales (28)
FEATURES_ORIGINAL_5MIN = [
    'log_ret_5m', 'log_ret_15m', 'log_ret_30m', 'rsi_9', 'atr_pct',
    'bb_position', 'adx_14', 'ema_cross', 'high_low_range',
    'session_progress', 'hour_sin', 'hour_cos',
]

FEATURES_ORIGINAL_HOURLY = ['log_ret_1h', 'log_ret_4h', 'volatility_1h']

FEATURES_ORIGINAL_DAILY = [
    'usdcop_ret_1d', 'usdcop_ret_5d', 'usdcop_volatility',
    'vix_z', 'vix_zscore', 'embi_z', 'dxy_z', 'brent_change_1d',
    'oil_above_60_flag', 'rate_spread', 'usdmxn_change_1d',
    'usdclp_ret_1d', 'banrep_intervention_proximity',
]

# Features LIMPIAS (23) - sin redundantes ni problematicas
FEATURES_CLEAN_5MIN = [
    'log_ret_5m', 'log_ret_15m', 'log_ret_30m', 'rsi_9', 'atr_pct',
    'bb_position', 'adx_14', 'ema_cross',
    'session_progress',  # hour_sin, hour_cos, high_low_range REMOVIDAS
]  # 9 features

FEATURES_CLEAN_HOURLY = ['log_ret_1h', 'log_ret_4h', 'volatility_1h']  # 3

FEATURES_CLEAN_DAILY = [
    'usdcop_ret_1d', 'usdcop_ret_5d', 'usdcop_volatility',
    'vix_z', 'embi_z', 'dxy_z', 'brent_change_1d',  # CORREGIDO: vix_z (no vix_zscore)
    'rate_spread', 'usdmxn_change_1d',  # oil_above_60_flag REMOVIDO
    'usdclp_ret_1d', 'banrep_intervention_proximity',
]  # 11 features

ALL_FEATURES_CLEAN = FEATURES_CLEAN_5MIN + FEATURES_CLEAN_HOURLY + FEATURES_CLEAN_DAILY
# Total: 9 + 3 + 11 = 23 features


# ==============================================================================
# PREPROCESSING FUNCTIONS
# ==============================================================================

def remove_warmup_period(
    df: pd.DataFrame,
    warmup_days: int = 60,
    bars_per_day: int = 60,
) -> pd.DataFrame:
    """
    Remove warmup period from dataset.

    Razon: Features como vix_zscore tienen 78% ceros en el warm-up period
    porque requieren lookback de 20-60 dias.

    Args:
        df: DataFrame original
        warmup_days: Dias de warmup a descartar (default: 60)
        bars_per_day: Barras por dia (default: 60)

    Returns:
        DataFrame sin warmup period
    """
    warmup_bars = warmup_days * bars_per_day

    if len(df) <= warmup_bars:
        print(f"WARNING: Dataset ({len(df)} rows) smaller than warmup ({warmup_bars})")
        return df

    df_clean = df.iloc[warmup_bars:].copy()

    print(f"Warmup removal: {len(df):,} -> {len(df_clean):,} rows "
          f"(-{warmup_bars:,} = {warmup_days} days)")

    return df_clean


def remove_redundant_features(
    df: pd.DataFrame,
    features_to_drop: List[str],
) -> pd.DataFrame:
    """
    Remove redundant features.

    Features redundantes (correlacion > 80%):
    - hour_sin: r=-0.975 con session_progress
    - hour_cos: r=-0.943 con session_progress
    - oil_above_60_flag: r=-0.911 con rate_spread
    - vix_z: r=-0.859 con oil_above_60_flag

    Args:
        df: DataFrame
        features_to_drop: Lista de features a eliminar

    Returns:
        DataFrame sin features redundantes
    """
    cols_to_drop = [c for c in features_to_drop if c in df.columns]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Removed redundant features: {cols_to_drop}")
        print(f"Features remaining: {len([c for c in df.columns if c != 'timestamp'])}")

    return df


def winsorize_returns(
    df: pd.DataFrame,
    return_cols: List[str],
    percentile: float = 0.01,
) -> pd.DataFrame:
    """
    Winsorize extreme returns to reduce noise.

    Razon: log_ret_5m tiene skewness=11.38, kurtosis=301.84 (colas muy pesadas)

    Args:
        df: DataFrame
        return_cols: Columnas de returns a winsorizar
        percentile: Percentil de corte (default: 1%)

    Returns:
        DataFrame con returns winsorized
    """
    df = df.copy()

    for col in return_cols:
        if col not in df.columns:
            continue

        lower = df[col].quantile(percentile)
        upper = df[col].quantile(1 - percentile)

        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()

        if n_clipped > 0:
            df[col] = df[col].clip(lower, upper)
            print(f"Winsorized {col}: {n_clipped} outliers clipped to [{lower:.6f}, {upper:.6f}]")

    return df


def normalize_per_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features_to_normalize: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize features using train statistics only.

    Razon: vix_z, embi_z tienen mean != 0 (mal normalizados globalmente)

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        features_to_normalize: Features a normalizar

    Returns:
        (train_df_normalized, val_df_normalized, normalization_stats)
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    stats = {}

    for col in features_to_normalize:
        if col not in train_df.columns:
            continue

        mean = train_df[col].mean()
        std = train_df[col].std()

        if std < 1e-8:
            std = 1.0

        train_df[col] = (train_df[col] - mean) / std
        val_df[col] = (val_df[col] - mean) / std

        # Clip extremos post-normalizacion
        train_df[col] = train_df[col].clip(-5, 5)
        val_df[col] = val_df[col].clip(-5, 5)

        stats[col] = {'mean': float(mean), 'std': float(std)}

        print(f"Normalized {col}: mean={mean:.4f} -> 0, std={std:.4f} -> 1")

    return train_df, val_df, stats


def validate_warmup_complete(
    df: pd.DataFrame,
    check_cols: List[str] = None,
    zero_threshold: float = 0.05,
) -> Dict[str, Dict]:
    """
    Validate that warmup period has been properly removed.

    Args:
        df: DataFrame to validate
        check_cols: Columns to check for zeros
        zero_threshold: Max acceptable % of zeros

    Returns:
        Validation results
    """
    if check_cols is None:
        check_cols = ['vix_z', 'usdcop_volatility', 'usdcop_ret_5d']

    results = {}

    for col in check_cols:
        if col not in df.columns:
            continue

        zero_pct = (df[col] == 0).mean() * 100

        results[col] = {
            'zero_pct': zero_pct,
            'passed': zero_pct < zero_threshold * 100,
        }

        status = "OK" if results[col]['passed'] else "WARN"
        print(f"[{status}] {col}: {zero_pct:.1f}% zeros")

    return results


# ==============================================================================
# MAIN PREPROCESSING PIPELINE
# ==============================================================================

class DataPreprocessor:
    """
    Complete data preprocessing pipeline.

    Usage:
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.fit_transform(df)

        # For per-fold normalization:
        train_clean, val_clean = preprocessor.transform_fold(train_df, val_df)
    """

    def __init__(self, config: DataQualityConfig = None):
        self.config = config or DataQualityConfig()
        self.normalization_stats = {}
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps (except per-fold normalization).

        Steps:
        1. Remove warmup period
        2. Remove redundant features
        3. Winsorize extreme returns
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)

        original_rows = len(df)
        original_cols = len(df.columns)

        # Step 1: Remove warmup
        print("\n--- Step 1: Remove Warmup Period ---")
        df = remove_warmup_period(
            df,
            warmup_days=self.config.warmup_days,
            bars_per_day=self.config.bars_per_day,
        )

        # Step 2: Remove redundant features
        print("\n--- Step 2: Remove Redundant Features ---")
        df = remove_redundant_features(
            df,
            features_to_drop=self.config.features_to_drop,
        )

        # Step 3: Winsorize returns
        print("\n--- Step 3: Winsorize Extreme Returns ---")
        df = winsorize_returns(
            df,
            return_cols=self.config.return_columns,
            percentile=self.config.winsorize_percentile,
        )

        # Validate
        print("\n--- Validation ---")
        validate_warmup_complete(df)

        # Summary
        print("\n--- Summary ---")
        print(f"Rows: {original_rows:,} -> {len(df):,} ({len(df)/original_rows*100:.1f}%)")
        print(f"Columns: {original_cols} -> {len(df.columns)}")

        self.is_fitted = True

        return df

    def transform_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply per-fold normalization.

        IMPORTANTE: Llamar DESPUES de fit_transform()
        """
        print("\n--- Per-Fold Normalization ---")

        train_df, val_df, stats = normalize_per_fold(
            train_df,
            val_df,
            features_to_normalize=self.config.features_to_normalize,
        )

        self.normalization_stats = stats

        return train_df, val_df

    def get_clean_features(self) -> List[str]:
        """Get list of clean features (24 instead of 28)."""
        return ALL_FEATURES_CLEAN.copy()


# ==============================================================================
# QUICK DIAGNOSTIC
# ==============================================================================

def diagnose_dataset(df: pd.DataFrame) -> Dict:
    """
    Quick diagnostic of dataset quality.
    """
    print("\n" + "="*60)
    print("DATASET DIAGNOSTIC")
    print("="*60)

    results = {
        'rows': len(df),
        'columns': len(df.columns),
        'issues': [],
    }

    # Check zeros in problematic columns
    problem_cols = ['vix_zscore', 'vix_z', 'oil_above_60_flag', 'usdcop_volatility']
    print("\n--- Zero Check ---")
    for col in problem_cols:
        if col in df.columns:
            zero_pct = (df[col] == 0).mean() * 100
            print(f"{col}: {zero_pct:.1f}% zeros")
            if zero_pct > 10:
                results['issues'].append(f"{col} has {zero_pct:.1f}% zeros")

    # Check correlations
    print("\n--- Correlation Check ---")
    high_corr_pairs = [
        ('session_progress', 'hour_sin'),
        ('session_progress', 'hour_cos'),
        ('oil_above_60_flag', 'rate_spread'),
    ]
    for col1, col2 in high_corr_pairs:
        if col1 in df.columns and col2 in df.columns:
            corr = df[col1].corr(df[col2])
            print(f"{col1} <-> {col2}: {corr:.3f}")
            if abs(corr) > 0.8:
                results['issues'].append(f"High correlation {col1}<->{col2}: {corr:.3f}")

    # Check skewness
    print("\n--- Skewness Check ---")
    if 'log_ret_5m' in df.columns:
        skew = df['log_ret_5m'].skew()
        kurt = df['log_ret_5m'].kurtosis()
        print(f"log_ret_5m: skew={skew:.2f}, kurtosis={kurt:.2f}")
        if abs(skew) > 2:
            results['issues'].append(f"High skewness in log_ret_5m: {skew:.2f}")

    # Summary
    print("\n--- Summary ---")
    if results['issues']:
        print(f"Found {len(results['issues'])} issues:")
        for issue in results['issues']:
            print(f"  - {issue}")
    else:
        print("No issues found!")

    return results


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess USD/COP dataset')
    parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    parser.add_argument('--output', type=str, help='Output CSV path')
    parser.add_argument('--diagnose-only', action='store_true', help='Only run diagnostics')

    args = parser.parse_args()

    # Load
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, parse_dates=['timestamp'])

    if args.diagnose_only:
        diagnose_dataset(df)
    else:
        # Preprocess
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.fit_transform(df)

        # Save
        if args.output:
            df_clean.to_csv(args.output, index=False)
            print(f"\nSaved to {args.output}")
        else:
            print("\nNo output path specified, data not saved")
