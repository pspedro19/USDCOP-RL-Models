"""
Regime Calculator - Anti-Look-ahead Implementation
Contrato: CTR-005 (parte de calculators)
CLAUDE-T5 | Plan Item: P0-9

IMPORTANTE: Este modulo implementa calculo de regimen SIN look-ahead bias.
Nunca usa rank(), percentile() o funciones similares que usen datos futuros.
"""

import numpy as np
import pandas as pd
from typing import Literal


def expanding_percentile(
    series: pd.Series,
    bar_idx: int,
    min_periods: int = 20
) -> float:
    """
    Calcula percentil usando SOLO datos historicos (sin look-ahead).

    Esta funcion es el reemplazo seguro para df['col'].rank(pct=True)
    que introduce look-ahead bias.

    Args:
        series: Serie de valores
        bar_idx: Indice actual
        min_periods: Minimo de datos para calculo valido

    Returns:
        Percentil en [0, 1]. Retorna 0.5 si no hay suficientes datos.

    Example:
        >>> series = pd.Series([1, 2, 3, 4, 5])
        >>> pct = expanding_percentile(series, bar_idx=4)
        >>> print(pct)  # 0.8 (4 de 5 valores son menores que 5)
    """
    if bar_idx < min_periods:
        return 0.5  # Neutral

    # Solo datos hasta bar_idx (inclusive)
    historical = series.iloc[:bar_idx + 1]
    current_value = historical.iloc[-1]

    # Manejar NaN
    if pd.isna(current_value):
        return 0.5

    historical_clean = historical.dropna()
    if len(historical_clean) < min_periods:
        return 0.5

    # Percentil manual sin look-ahead
    count_below = (historical_clean < current_value).sum()
    percentile = count_below / len(historical_clean)

    return float(percentile)


def expanding_zscore(
    series: pd.Series,
    bar_idx: int,
    min_periods: int = 20
) -> float:
    """
    Calcula z-score usando SOLO datos historicos (sin look-ahead).

    Args:
        series: Serie de valores
        bar_idx: Indice actual
        min_periods: Minimo de datos para calculo valido

    Returns:
        Z-score. Retorna 0.0 si no hay suficientes datos.
    """
    if bar_idx < min_periods:
        return 0.0

    historical = series.iloc[:bar_idx + 1]
    current_value = historical.iloc[-1]

    if pd.isna(current_value):
        return 0.0

    historical_clean = historical.dropna()
    if len(historical_clean) < min_periods:
        return 0.0

    mean = historical_clean.mean()
    std = historical_clean.std()

    if std == 0 or pd.isna(std):
        return 0.0

    z_score = (current_value - mean) / std
    return float(z_score)


def detect_regime(
    ohlcv: pd.DataFrame,
    bar_idx: int,
    volatility_period: int = 20,
    min_history: int = 20
) -> Literal['low_vol', 'medium_vol', 'high_vol']:
    """
    Detecta regimen de mercado sin look-ahead bias.

    IMPORTANTE: Esta funcion usa expanding_percentile en lugar de rank(pct=True)
    para evitar look-ahead bias.

    Args:
        ohlcv: DataFrame con columns [open, high, low, close, volume]
        bar_idx: Indice de la barra actual
        volatility_period: Periodo para calcular volatilidad
        min_history: Minimo de datos para clasificacion

    Returns:
        'low_vol': percentil volatilidad < 33%
        'medium_vol': percentil volatilidad entre 33% y 67%
        'high_vol': percentil volatilidad > 67%

    Example:
        >>> regime = detect_regime(ohlcv_df, bar_idx=50)
        >>> print(regime)  # 'medium_vol'
    """
    if bar_idx < max(volatility_period, min_history):
        return 'medium_vol'  # Neutral por defecto

    # Calcular volatilidad historica (log returns)
    close = ohlcv['close']
    returns = np.log(close / close.shift(1))
    volatility = returns.rolling(volatility_period).std()

    # Percentil sin look-ahead
    pct = expanding_percentile(volatility, bar_idx, min_periods=min_history)

    if pct < 0.33:
        return 'low_vol'
    elif pct < 0.67:
        return 'medium_vol'
    else:
        return 'high_vol'


def calculate_volatility_percentile_series(
    ohlcv: pd.DataFrame,
    volatility_period: int = 20,
    min_periods: int = 20
) -> pd.Series:
    """
    Calcula serie de percentiles de volatilidad SIN look-ahead.

    Esta funcion es el reemplazo seguro para:
        df['volatility'].rank(pct=True)  # MALO - look-ahead

    Args:
        ohlcv: DataFrame con OHLCV data
        volatility_period: Periodo para volatilidad
        min_periods: Minimo de datos para percentil

    Returns:
        Serie de percentiles [0, 1], NaN para primeros bar_idx
    """
    close = ohlcv['close']
    returns = np.log(close / close.shift(1))
    volatility = returns.rolling(volatility_period).std()

    # Calcular percentil para cada punto SIN look-ahead
    percentiles = pd.Series(index=ohlcv.index, dtype=float)

    for bar_idx in range(len(ohlcv)):
        if bar_idx < min_periods:
            percentiles.iloc[bar_idx] = np.nan
        else:
            percentiles.iloc[bar_idx] = expanding_percentile(
                volatility, bar_idx, min_periods
            )

    return percentiles


def validate_no_lookahead(
    calculation_func,
    series: pd.Series,
    bar_idx: int
) -> bool:
    """
    Valida que una funcion de calculo no tenga look-ahead bias.

    Modifica datos futuros y verifica que el resultado no cambie.

    Args:
        calculation_func: Funcion que toma (series, bar_idx)
        series: Serie de datos
        bar_idx: Indice a evaluar

    Returns:
        True si no hay look-ahead bias

    Example:
        >>> is_safe = validate_no_lookahead(expanding_percentile, series, 50)
        >>> assert is_safe, "Look-ahead bias detected!"
    """
    if bar_idx >= len(series) - 1:
        return True  # No hay datos futuros para verificar

    # Resultado original
    original_result = calculation_func(series, bar_idx)

    # Modificar datos futuros
    modified = series.copy()
    modified.iloc[bar_idx + 1:] = modified.iloc[bar_idx + 1:] * 1000  # Cambio drastico

    # Resultado con datos futuros modificados
    modified_result = calculation_func(modified, bar_idx)

    # Deben ser iguales si no hay look-ahead
    return np.isclose(original_result, modified_result, rtol=1e-10)
