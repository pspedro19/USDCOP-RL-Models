"""
USD/COP RL Trading System V11 - Utility Functions
==================================================

Data preprocessing and normalization utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def calculate_norm_stats(df_train: pd.DataFrame, features: List[str]) -> Dict:
    """
    Calculate normalization statistics from training data ONLY.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data (not test!)
    features : list
        List of feature column names

    Returns
    -------
    dict
        Dictionary with mean and std for each feature
    """
    norm_stats = {}

    for feat in features:
        if feat not in df_train.columns:
            continue

        col = df_train[feat].dropna()
        mean = col.mean()
        std = col.std() if col.std() > 1e-8 else 1.0

        norm_stats[feat] = {'mean': mean, 'std': std}

    return norm_stats


def normalize_df_v11(
    df_raw: pd.DataFrame,
    norm_stats: Dict,
    features: List[str],
    raw_return_col: str = '_raw_ret_5m'
) -> pd.DataFrame:
    """
    Normalize features for model input.

    V11 FIX: Saves RAW returns in separate column BEFORE normalizing!

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw (unnormalized) data
    norm_stats : dict
        Normalization statistics from training data
    features : list
        List of feature column names
    raw_return_col : str
        Column name for storing raw returns

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with raw returns preserved
    """
    df_norm = df_raw.copy()

    # V11 FIX: Save raw returns BEFORE normalizing
    if 'log_ret_5m' in df_norm.columns:
        df_norm[raw_return_col] = df_norm['log_ret_5m'].copy()
    else:
        df_norm[raw_return_col] = 0.0

    # Normalize features for model
    for feat in features:
        if feat not in df_norm.columns or feat not in norm_stats:
            continue

        col = df_norm[feat]
        stats = norm_stats[feat]

        df_norm[feat] = (col - stats['mean']) / stats['std']
        df_norm[feat] = df_norm[feat].clip(-5, 5).fillna(0)

    return df_norm


def analyze_regime_raw(
    df_raw: pd.DataFrame,
    name: str,
    logger=None
) -> Dict:
    """
    Analyze market regime from RAW data.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw (unnormalized) data
    name : str
        Name for logging (e.g., "Train", "Test")
    logger : optional
        Logger instance

    Returns
    -------
    dict
        Dictionary with cumulative return and regime classification
    """
    if 'log_ret_5m' not in df_raw.columns:
        return {'cumret': 0, 'regime': 'UNKNOWN'}

    ret = df_raw['log_ret_5m'].dropna()
    cumret = ret.sum() * 100

    if cumret > 5:
        regime = 'BULLISH'
    elif cumret < -5:
        regime = 'BEARISH'
    else:
        regime = 'SIDEWAYS'

    msg = f"{name}: cumret={cumret:+.2f}% [{regime}]"

    if logger:
        logger.log(f"  {msg}")
    else:
        print(msg)

    return {'cumret': cumret, 'regime': regime}


def load_and_prepare_data(
    data_path: str,
    features_for_model: List[str]
) -> tuple:
    """
    Load and prepare data for training.

    Parameters
    ----------
    data_path : str
        Path to CSV data file
    features_for_model : list
        List of feature column names

    Returns
    -------
    tuple
        (df, features) - DataFrame and validated feature list
    """
    df_raw = pd.read_csv(data_path)

    # Remove obs_ prefix if present
    df = df_raw.rename(
        columns={c: c.replace('obs_', '') for c in df_raw.columns}
    )

    # Set timestamp as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    # Validate features
    features = [f for f in features_for_model if f in df.columns]

    return df, features


def calculate_wfe(train_sharpe: float, test_sharpe: float) -> float:
    """
    Calculate Walk-Forward Efficiency.

    WFE = Test Sharpe / Train Sharpe

    Parameters
    ----------
    train_sharpe : float
        Sharpe ratio on training data
    test_sharpe : float
        Sharpe ratio on test data

    Returns
    -------
    float
        Walk-Forward Efficiency ratio
    """
    if abs(train_sharpe) < 0.01:
        return 0.0
    return test_sharpe / train_sharpe


def classify_wfe(wfe: float) -> str:
    """
    Classify Walk-Forward Efficiency.

    Parameters
    ----------
    wfe : float
        Walk-Forward Efficiency ratio

    Returns
    -------
    str
        Classification string
    """
    if wfe >= 0.65:
        return "BUENO - Modelo robusto"
    elif wfe >= 0.50:
        return "ACEPTABLE - Produccion con monitoreo"
    elif wfe >= 0.30:
        return "MODERADO - Necesita mejoras"
    else:
        return "POBRE - Overfitting severo"
