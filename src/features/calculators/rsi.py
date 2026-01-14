"""
Calculator para RSI - Backward Compatibility Layer
===================================================
Delegates to feature_store.core for the actual implementation.

CRITICAL: Uses Wilder's smoothing (alpha = 1/period) for consistency
across training, backtest, and inference.

DEPRECATED: Import directly from feature_store instead:
    from feature_store import RSICalculator

Contrato: CTR-005b
Default period: 9
"""

import numpy as np
import pandas as pd

# Import from SSOT
from feature_store.core import RSICalculator as _RSICalculator, SmoothingMethod


def calculate(
    close: pd.Series,
    period: int,
    bar_idx: int
) -> float:
    """
    Calcula RSI usando Wilder's EMA para smoothing.

    CRITICAL: Uses alpha = 1/period (Wilder's method) for consistency.

    Args:
        close: Serie de precios de cierre
        period: Periodo RSI (default: 9)
        bar_idx: Indice de la barra actual

    Returns:
        RSI en rango [0, 100]. Retorna 50.0 si no hay suficientes datos.
    """
    if bar_idx < period:
        return 50.0

    # Slice hasta bar_idx inclusive
    start_idx = max(0, bar_idx - period * 2)
    prices = close.iloc[start_idx:bar_idx + 1]

    if len(prices) < 2:
        return 50.0

    # Calcular cambios
    delta = prices.diff()

    # Separar gains y losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # CRITICAL: Wilder's smoothing (alpha = 1/period)
    # This ensures consistency between training and inference
    alpha = 1.0 / period
    avg_gain = gains.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

    if np.isnan(avg_gain) or np.isnan(avg_loss):
        return 50.0

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return float(np.clip(rsi, 0.0, 100.0))
