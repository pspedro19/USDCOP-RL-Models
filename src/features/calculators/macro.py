"""
Calculator para indicadores macro - Backward Compatibility Layer
================================================================
Maintains the same interface but references feature_store for documentation.

DEPRECATED: Import directly from feature_store instead:
    from feature_store import MacroZScoreCalculator, MacroChangeCalculator

Contrato: CTR-005e

Calcula z-scores y cambios diarios para:
- DXY (Dollar Index)
- VIX (Volatility Index)
- EMBI (Emerging Market Bond Index)
- Brent (Oil)
- USDMXN
- Rate Spread
"""

import numpy as np
import pandas as pd
from typing import Union

# Import from SSOT for reference
from feature_store.core import MacroZScoreCalculator, MacroChangeCalculator


def z_score(
    series: pd.Series,
    bar_idx: int,
    lookback: int = 60
) -> float:
    """
    Calcula z-score usando ventana historica.

    Uses rolling window to prevent look-ahead bias.
    Consistent with feature_store.MacroZScoreCalculator.

    Args:
        series: Serie de valores macro
        bar_idx: Indice de la barra actual
        lookback: Numero de barras para calcular mean/std

    Returns:
        Z-score del valor actual. Retorna 0.0 si no hay suficientes datos.
    """
    if bar_idx < 1:
        return 0.0

    start_idx = max(0, bar_idx - lookback + 1)
    historical = series.iloc[start_idx:bar_idx + 1]

    if len(historical) < 2:
        return 0.0

    current = historical.iloc[-1]
    if np.isnan(current):
        return 0.0

    mean = historical.mean()
    std = historical.std()

    if std == 0 or np.isnan(std):
        return 0.0

    z = (current - mean) / std

    if np.isnan(z) or np.isinf(z):
        return 0.0

    return float(np.clip(z, -4.0, 4.0))


def change_1d(
    series: pd.Series,
    bar_idx: int,
    periods: int = 288  # 288 barras de 5min = 1 dia
) -> float:
    """
    Calcula cambio porcentual de 1 dia.

    Args:
        series: Serie de valores macro
        bar_idx: Indice de la barra actual
        periods: Numero de barras en 1 dia (288 para 5min)

    Returns:
        Cambio porcentual como decimal. Retorna 0.0 si no hay suficientes datos.
    """
    if bar_idx < periods:
        if bar_idx < 1:
            return 0.0
        periods = bar_idx

    current = series.iloc[bar_idx]
    previous = series.iloc[bar_idx - periods]

    if np.isnan(current) or np.isnan(previous):
        return 0.0

    if previous == 0:
        return 0.0

    change = (current - previous) / previous

    if np.isnan(change) or np.isinf(change):
        return 0.0

    return float(np.clip(change, -0.10, 0.10))


def get_value(
    series: pd.Series,
    bar_idx: int
) -> float:
    """
    Obtiene valor de la serie en el indice dado.

    Args:
        series: Serie de valores macro
        bar_idx: Indice de la barra actual

    Returns:
        Valor actual. Retorna 0.0 si es NaN.
    """
    if bar_idx < 0 or bar_idx >= len(series):
        return 0.0

    value = series.iloc[bar_idx]

    if np.isnan(value):
        return 0.0

    return float(value)


def volatility_5d(
    series: pd.Series,
    bar_idx: int,
    periods: int = 1440  # 1440 barras de 5min = 5 dias
) -> float:
    """
    Calcula volatilidad de 5 dias (std de retornos).

    Args:
        series: Serie de valores macro
        bar_idx: Indice de la barra actual
        periods: Numero de barras en 5 dias

    Returns:
        Volatilidad como decimal. Retorna 0.0 si no hay suficientes datos.
    """
    if bar_idx < 10:
        return 0.0

    start_idx = max(0, bar_idx - periods + 1)
    historical = series.iloc[start_idx:bar_idx + 1]

    if len(historical) < 10:
        return 0.0

    returns = historical.pct_change().dropna()

    if len(returns) < 5:
        return 0.0

    vol = returns.std()

    if np.isnan(vol):
        return 0.0

    return float(np.clip(vol, 0.0, 0.10))
