"""
Calculator para ADX - Backward Compatibility Layer
===================================================
This module uses Wilder's smoothing (alpha = 1/period) consistently.

DEPRECATED: Import directly from feature_store instead:
    from feature_store import ADXCalculator

Contrato: CTR-005d
Default period: 14
"""

import numpy as np
import pandas as pd

# Import from SSOT
from feature_store.core import ADXCalculator as _ADXCalculator, SmoothingMethod


def calculate(
    ohlcv: pd.DataFrame,
    period: int,
    bar_idx: int
) -> float:
    """
    Calcula ADX usando Wilder's smoothing.

    CRITICAL: All components use alpha = 1/period (Wilder's method).

    Args:
        ohlcv: DataFrame con columns [high, low, close]
        period: Periodo ADX (default: 14)
        bar_idx: Indice de la barra actual

    Returns:
        ADX en rango [0, 100]. Retorna 0.0 si no hay suficientes datos.
    """
    required_bars = period * 2 + 1
    if bar_idx < required_bars:
        return 0.0

    # Slice necesario
    start_idx = max(0, bar_idx - period * 3)
    df = ohlcv.iloc[start_idx:bar_idx + 1].copy()

    if len(df) < required_bars:
        return 0.0

    # +DM y -DM
    df["high_diff"] = df["high"].diff()
    df["low_diff"] = -df["low"].diff()

    df["+dm"] = np.where(
        (df["high_diff"] > df["low_diff"]) & (df["high_diff"] > 0),
        df["high_diff"],
        0.0
    )
    df["-dm"] = np.where(
        (df["low_diff"] > df["high_diff"]) & (df["low_diff"] > 0),
        df["low_diff"],
        0.0
    )

    # True Range
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["prev_close"]).abs(),
            (df["low"] - df["prev_close"]).abs()
        )
    )

    # CRITICAL: Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / period
    df["atr"] = df["tr"].ewm(alpha=alpha, adjust=False).mean()
    df["atr"] = df["atr"].replace(0, np.nan)

    df["+di"] = 100 * df["+dm"].ewm(alpha=alpha, adjust=False).mean() / df["atr"]
    df["-di"] = 100 * df["-dm"].ewm(alpha=alpha, adjust=False).mean() / df["atr"]

    # DX
    di_sum = df["+di"] + df["-di"]
    df["dx"] = np.where(
        di_sum > 0,
        100 * (df["+di"] - df["-di"]).abs() / di_sum,
        0.0
    )

    # ADX = smoothed DX
    adx = df["dx"].ewm(alpha=alpha, adjust=False).mean().iloc[-1]

    if np.isnan(adx):
        return 0.0

    return float(np.clip(adx, 0.0, 100.0))
