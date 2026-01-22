# backend/src/features/common.py
"""
Common feature utilities for the USD/COP Forecasting Pipeline.

This module provides centralized functions for feature preparation
and target creation, eliminating duplication across pipeline scripts.

Usage:
    from src.features.common import prepare_features, create_targets

    df, feature_cols = prepare_features(df)
    targets = create_targets(df, horizons=[1, 5, 10, 15, 22, 30])
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Default horizons for the pipeline
DEFAULT_HORIZONS = [1, 5, 10, 15, 22, 30]


def prepare_features(
    df: pd.DataFrame,
    exclude_cols: Optional[set] = None,
    exclude_prefixes: Optional[tuple] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features from DataFrame by filtering out non-feature columns.

    Excludes:
    - OHLCV columns (date, close, open, high, low, volume)
    - Target columns (prefixed with 'target', 'y_', 'label')
    - Non-numeric columns

    Args:
        df: Input DataFrame with all columns
        exclude_cols: Additional columns to exclude (optional)
        exclude_prefixes: Additional prefixes to exclude (optional)

    Returns:
        Tuple of (original DataFrame, list of feature column names)

    Example:
        df, feature_cols = prepare_features(df)
        X = df[feature_cols].values
    """
    # Default exclusions
    default_exclude_cols = {
        'date', 'close', 'Close', 'open', 'Open',
        'high', 'High', 'low', 'Low', 'volume', 'Volume'
    }

    default_exclude_prefixes = ('target', 'y_', 'label')

    # Merge with custom exclusions
    if exclude_cols:
        default_exclude_cols = default_exclude_cols.union(exclude_cols)
    if exclude_prefixes:
        default_exclude_prefixes = default_exclude_prefixes + exclude_prefixes

    feature_cols = []
    for col in df.columns:
        # Skip excluded columns
        if col in default_exclude_cols:
            continue

        # Skip columns with excluded prefixes
        if any(col.lower().startswith(p) for p in default_exclude_prefixes):
            continue

        # Only include numeric columns
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            feature_cols.append(col)

    logger.debug(f"Selected {len(feature_cols)} feature columns")
    return df, feature_cols


def create_targets(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    price_col: str = 'close'
) -> Dict[int, pd.Series]:
    """
    Create log-return targets for multiple horizons.

    Calculates: ln(P_{t+h} / P_t) for each horizon h

    Args:
        df: DataFrame with price column
        horizons: List of horizons in trading days (default: [1, 5, 10, 15, 22, 30])
        price_col: Name of price column (default: 'close')

    Returns:
        Dictionary mapping horizon -> Series of log returns

    Example:
        targets = create_targets(df, horizons=[1, 5, 10, 15, 22, 30])
        y_5d = targets[5]  # 5-day ahead returns
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    # Handle case-insensitive price column
    if price_col not in df.columns:
        price_col_upper = price_col.capitalize()
        if price_col_upper in df.columns:
            price_col = price_col_upper
        else:
            raise ValueError(f"Price column '{price_col}' not found in DataFrame")

    close = df[price_col]
    targets = {}

    for h in horizons:
        future_price = close.shift(-h)
        targets[h] = np.log(future_price / close)
        logger.debug(f"Created target for H={h}: {(~targets[h].isna()).sum()} valid samples")

    return targets


def prepare_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    gap: Optional[int] = None,
    max_horizon: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test with temporal gap to prevent look-ahead bias.

    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        train_ratio: Proportion of data for training (default: 0.8)
        gap: Gap between train and test (default: max_horizon)
        max_horizon: Maximum horizon for default gap (default: 30)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Example:
        X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, gap=30)
    """
    if gap is None:
        gap = max_horizon

    n = len(X)
    train_end = int(n * train_ratio) - gap
    test_start = int(n * train_ratio)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test = X[test_start:]
    y_test = y[test_start:]

    logger.info(
        f"Train/Test split: train={len(X_train)}, test={len(X_test)}, "
        f"gap={gap} samples"
    )

    return X_train, X_test, y_train, y_test


def clean_features(
    X: np.ndarray,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Clean features by replacing inf/nan values.

    Args:
        X: Feature array
        fill_value: Value to replace nan with (default: 0.0)

    Returns:
        Cleaned feature array
    """
    X_clean = np.nan_to_num(X, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return X_clean


def prepare_features_for_training(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and target array for training.

    Handles:
    - Missing values in features (forward fill, backward fill, then 0)
    - Inf values replacement
    - Alignment with valid target values

    Args:
        df: DataFrame with feature columns
        feature_cols: List of feature column names
        target: Target Series

    Returns:
        Tuple of (X, y) arrays ready for training
    """
    # Prepare feature DataFrame
    X_df = df[feature_cols].copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.ffill().bfill().fillna(0)

    # Get valid indices (where target is not NaN)
    valid_idx = ~target.isna()

    # Extract arrays
    X = X_df.values[valid_idx]
    y = target.values[valid_idx]

    # Final cleanup
    X = clean_features(X)

    logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")

    return X, y
