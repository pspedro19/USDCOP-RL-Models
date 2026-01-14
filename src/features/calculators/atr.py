"""
Calculator para ATR - Backward Compatibility Layer
===================================================
Delegates to feature_store.core for the actual implementation.

CRITICAL: Uses Wilder's smoothing (alpha = 1/period) for consistency
across training, backtest, and inference.

DEPRECATED: Import directly from feature_store instead:
    from feature_store import ATRPercentCalculator

Contrato: CTR-005c
Default period: 10
"""

import numpy as np
import pandas as pd

# Import from SSOT
from feature_store.core import ATRPercentCalculator as _ATRCalculator, SmoothingMethod


def calculate_pct(
    ohlcv: pd.DataFrame,
    period: int,
    bar_idx: int
) -> float:
    """
    Calcula ATR como porcentaje del precio actual.

    CRITICAL: Uses Wilder's smoothing (alpha = 1/period) for consistency.

    Args:
        ohlcv: DataFrame con columns [open, high, low, close]
        period: Periodo ATR (default: 10)
        bar_idx: Indice de la barra actual

    Returns:
        ATR % en rango [0, inf). Retorna 0.0 si no hay suficientes datos.

    Formula:
        TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        ATR = Wilder's EMA(TR, period)  # alpha = 1/period
        ATR_pct = ATR / close * 100
    """
    if bar_idx < period:
        return 0.0

    # Slice necesario
    start_idx = max(0, bar_idx - period * 2)
    df = ohlcv.iloc[start_idx:bar_idx + 1].copy()

    if len(df) < 2:
        return 0.0

    # True Range
    df["prev_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = (df["high"] - df["prev_close"]).abs()
    df["tr3"] = (df["low"] - df["prev_close"]).abs()
    df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

    # CRITICAL: Wilder's smoothing (alpha = 1/period)
    # This ensures consistency between training and inference
    alpha = 1.0 / period
    atr = df["tr"].ewm(alpha=alpha, adjust=False).mean().iloc[-1]

    if np.isnan(atr):
        return 0.0

    # Como porcentaje
    current_price = df["close"].iloc[-1]
    if current_price <= 0 or np.isnan(current_price):
        return 0.0

    atr_pct = (atr / current_price) * 100

    return float(max(0.0, atr_pct))
