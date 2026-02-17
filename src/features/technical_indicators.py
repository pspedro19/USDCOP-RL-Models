"""
SSOT: Technical Indicators Calculation Module
==============================================

Este módulo es la ÚNICA fuente de verdad para cálculos de indicadores técnicos.
Todos los componentes que necesiten calcular indicadores técnicos DEBEN usar
estas funciones.

IMPORTANTE: NO duplicar estas funciones en otros archivos.

Fecha: 2026-02-01
Versión: 2.0

Changelog:
- v2.0: CLOSE-ONLY indicators (volatility_pct, trend_z) para reemplazar ATR/ADX
- v1.0: ADX percentage-based fix para evitar saturación en USDCOP
- v1.0: RSI con Wilders smoothing consistente
- v1.0: Macro z-score con método rolling (evita look-ahead bias)

Contract: CTR-FEATURES-002 (CLOSE-ONLY)
"""

import pandas as pd
import numpy as np
from typing import Literal
import logging

logger = logging.getLogger(__name__)


def calculate_adx_wilders(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate ADX (Average Directional Index) using Wilders smoothing
    with percentage-based normalization.

    FIX CRÍTICO: Usa ATR como porcentaje del precio para evitar saturación
    en pares con valores altos como USDCOP (~4000 COP).

    El problema original era que con precios altos:
    - ATR absoluto es grande (~20-50 COP)
    - Directional movements son pequeños relativos al ATR
    - Esto causaba que +DI y -DI fueran muy pequeños
    - Y ADX saturaba cerca de 100

    La solución es normalizar todo como porcentaje del precio.

    Args:
        high: Serie de precios máximos
        low: Serie de precios mínimos
        close: Serie de precios de cierre
        period: Período para el cálculo (default: 14)

    Returns:
        pd.Series: ADX values entre 0 y 100

    Example:
        >>> df['adx_14'] = calculate_adx_wilders(df['high'], df['low'], df['close'])
        >>> assert df['adx_14'].mean() < 50  # Should not be saturated
    """
    alpha = 1.0 / period

    # Movimientos direccionales
    high_diff = high.diff()
    low_diff = low.diff()

    # FIX: Directional movement como PORCENTAJE del precio
    # En lugar de valores absolutos que escalan con el precio
    plus_dm_pct = (high_diff / close).where(
        (high_diff > low_diff.abs()) & (high_diff > 0), 0.0
    )
    minus_dm_pct = (low_diff.abs() / close).where(
        (low_diff.abs() > high_diff) & (low_diff < 0), 0.0
    )

    # True Range y ATR como porcentaje
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    atr_pct = (atr / close).clip(lower=1e-6)

    # Directional Indicators calculados sobre porcentajes
    plus_di = 100.0 * plus_dm_pct.ewm(alpha=alpha, adjust=False).mean() / atr_pct
    minus_di = 100.0 * minus_dm_pct.ewm(alpha=alpha, adjust=False).mean() / atr_pct

    # Clip a rango válido
    plus_di = plus_di.clip(0, 100)
    minus_di = minus_di.clip(0, 100)

    # DX y ADX
    di_sum = (plus_di + minus_di).clip(lower=1.0)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    # Validación
    result = adx.clip(0, 100)

    # Log warning si todavía está saturado
    if result.mean() > 60:
        logger.warning(
            f"ADX mean is {result.mean():.2f}, may still be saturated. "
            f"Expected mean < 40 for healthy distribution."
        )

    return result


def calculate_rsi_wilders(
    close: pd.Series,
    period: int = 9
) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) using Wilders smoothing.

    IMPORTANTE: Usa Wilders smoothing (EMA con alpha=1/period), NO SMA.
    Esto es consistente con la definición original de Wilder y con
    la mayoría de plataformas de trading.

    Wilders smoothing es equivalente a:
        smoothed = prev_smoothed * (period-1)/period + current/period

    Lo cual es un EMA con alpha = 1/period.

    Args:
        close: Serie de precios de cierre
        period: Período para el cálculo (default: 9)

    Returns:
        pd.Series: RSI values entre 0 y 100

    Example:
        >>> df['rsi_9'] = calculate_rsi_wilders(df['close'], period=9)
        >>> assert 0 <= df['rsi_9'].min() and df['rsi_9'].max() <= 100
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilders smoothing: alpha = 1/period
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    # Relative Strength
    rs = avg_gain / avg_loss.clip(lower=1e-10)

    # RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi.clip(0, 100)


def calculate_atr_percentage(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate ATR (Average True Range) as percentage of price.

    ATR como porcentaje permite comparaciones entre instrumentos
    con diferentes niveles de precio.

    NOTA: Esta función requiere H/L/C. Para datasets CLOSE-only,
    usar calculate_volatility_pct() en su lugar.

    Args:
        high: Serie de precios máximos
        low: Serie de precios mínimos
        close: Serie de precios de cierre
        period: Período para el cálculo (default: 14)

    Returns:
        pd.Series: ATR como porcentaje (0.001 = 0.1%)
    """
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()

    return atr / close


# =============================================================================
# CLOSE-ONLY Indicators (v2.0)
# =============================================================================
# Estos indicadores solo usan precio CLOSE, ideales para datasets donde
# O/H/L no son confiables o no están disponibles.
# =============================================================================

def calculate_volatility_pct(
    close: pd.Series,
    period: int = 14,
    annualize: bool = True,
    bars_per_day: int = 48
) -> pd.Series:
    """
    Calculate realized volatility from CLOSE prices only.

    REEMPLAZA ATR cuando solo tenemos CLOSE confiable.

    La volatilidad realizada se calcula como la desviación estándar
    de los log-returns, opcionalmente anualizada.

    Ventajas sobre ATR:
    1. Solo necesita CLOSE (más robusto a datos ruidosos de H/L)
    2. Estadísticamente más interpretable (es std de returns)
    3. Comparable entre instrumentos cuando se anualiza

    Args:
        close: Serie de precios de cierre
        period: Ventana para el cálculo rolling (default: 14)
        annualize: Si True, anualiza la volatilidad (default: True)
        bars_per_day: Barras por día para anualización (default: 48 para 5min)

    Returns:
        pd.Series: Volatilidad realizada (anualizada si annualize=True)

    Example:
        >>> df['volatility_pct'] = calculate_volatility_pct(df['close'], period=14)
        >>> # Típicamente entre 0.05 (5%) y 0.30 (30%) anualizado para FX
    """
    # Log returns
    log_returns = np.log(close / close.shift(1))

    # Rolling standard deviation
    rolling_std = log_returns.rolling(window=period, min_periods=period // 2).std()

    if annualize:
        # Anualizar: sqrt(trading_days * bars_per_day)
        # Para 5min bars: sqrt(252 * 48) ≈ 110
        annualization_factor = np.sqrt(252 * bars_per_day)
        return rolling_std * annualization_factor
    else:
        return rolling_std


def calculate_trend_z(
    close: pd.Series,
    sma_period: int = 50,
    clip_value: float = 3.0
) -> pd.Series:
    """
    Calculate trend strength as z-score of price position vs SMA.

    REEMPLAZA ADX cuando solo tenemos CLOSE confiable.

    Mide qué tan lejos está el precio de su media móvil en unidades
    de desviación estándar. Positivo = tendencia alcista, negativo = bajista.

    Interpretación:
    - trend_z > 2: Tendencia alcista fuerte
    - trend_z entre -1 y 1: Sin tendencia clara (rango)
    - trend_z < -2: Tendencia bajista fuerte

    Ventajas sobre ADX:
    1. Solo necesita CLOSE
    2. Incluye DIRECCIÓN de la tendencia (ADX solo magnitud)
    3. Ya es un z-score, no necesita normalización adicional
    4. Más interpretable estadísticamente

    Args:
        close: Serie de precios de cierre
        sma_period: Período para SMA y rolling std (default: 50)
        clip_value: Valor para clipear extremos (default: 3.0)

    Returns:
        pd.Series: Z-score de posición de precio, clipped a [-clip_value, clip_value]

    Example:
        >>> df['trend_z'] = calculate_trend_z(df['close'], sma_period=50)
        >>> assert df['trend_z'].mean() < 0.5  # Should be ~0 over long periods
    """
    # SMA del período
    sma = close.rolling(window=sma_period, min_periods=sma_period // 2).mean()

    # Rolling std del precio (no de returns, del precio mismo)
    rolling_std = close.rolling(window=sma_period, min_periods=sma_period // 2).std()

    # Z-score: (precio - media) / std
    trend_z = (close - sma) / rolling_std.clip(lower=1e-6)

    # Clip para evitar valores extremos
    return trend_z.clip(-clip_value, clip_value)


def calculate_momentum_z(
    close: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate momentum as z-score of returns over period.

    Alternativa a ROC (Rate of Change) con mejor comportamiento estadístico.

    Args:
        close: Serie de precios de cierre
        period: Período para el momentum (default: 20)

    Returns:
        pd.Series: Momentum z-score clipped a [-5, 5]

    Example:
        >>> df['momentum_z'] = calculate_momentum_z(df['close'], period=20)
    """
    # Return sobre el período
    returns = close.pct_change(periods=period)

    # Rolling mean y std para z-score
    window = period * 5  # Ventana más larga para estadísticas estables
    mean = returns.rolling(window=window, min_periods=window // 2).mean()
    std = returns.rolling(window=window, min_periods=window // 2).std()

    z = (returns - mean) / std.clip(lower=1e-8)

    return z.clip(-5, 5)


def calculate_macro_zscore(
    series: pd.Series,
    window: int = 252,
    method: Literal['rolling', 'fixed'] = 'rolling'
) -> pd.Series:
    """
    Calculate z-score for macro variables.

    IMPORTANTE: Usar 'rolling' para evitar look-ahead bias.
    'fixed' (expanding) solo para casos especiales documentados.

    El método 'rolling' calcula mean y std sobre una ventana móvil,
    lo cual es más realista para trading en tiempo real.

    El método 'fixed' usa expanding mean/std desde el inicio,
    lo cual puede causar look-ahead bias si no se tiene cuidado.

    Args:
        series: Serie temporal de valores macro
        window: Ventana para cálculo rolling (252 = ~1 año trading days)
        method: 'rolling' (recomendado) o 'fixed' (expanding)

    Returns:
        pd.Series: Z-scores clipped a [-10, 10]

    Example:
        >>> df['dxy_z'] = calculate_macro_zscore(df['dxy'], window=252, method='rolling')
    """
    if method == 'rolling':
        mean = series.rolling(window=window, min_periods=window // 2).mean()
        std = series.rolling(window=window, min_periods=window // 2).std()
    elif method == 'fixed':
        # CUIDADO: Esto puede causar look-ahead bias si se usa incorrectamente
        logger.warning(
            "Using 'fixed' method for z-score calculation. "
            "This may cause look-ahead bias if not used carefully."
        )
        mean = series.expanding().mean()
        std = series.expanding().std()
    else:
        raise ValueError(f"method must be 'rolling' or 'fixed', got {method}")

    z = (series - mean) / std.clip(lower=1e-8)

    return z.clip(-10, 10)


def calculate_log_returns(
    close: pd.Series,
    periods: int = 1
) -> pd.Series:
    """
    Calculate log returns over specified periods.

    Log returns son preferidos sobre simple returns porque:
    1. Son aditivos en el tiempo
    2. Tienen mejor comportamiento estadístico
    3. Son simétricos para gains/losses

    Args:
        close: Serie de precios de cierre
        periods: Número de períodos para el return (default: 1)

    Returns:
        pd.Series: Log returns
    """
    return np.log(close / close.shift(periods))


def calculate_rate_spread(
    local_rate: pd.Series,
    foreign_rate: pd.Series
) -> pd.Series:
    """
    Calculate interest rate spread (local - foreign).

    Para USDCOP: BanRep rate - Fed Funds rate

    Args:
        local_rate: Serie de tasa de interés local (ej: BanRep)
        foreign_rate: Serie de tasa de interés extranjera (ej: Fed)

    Returns:
        pd.Series: Rate spread en puntos porcentuales
    """
    return local_rate - foreign_rate


def calculate_spread_zscore(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 252,
    spread_formula: str = "series1 - series2"
) -> pd.Series:
    """
    Calculate z-score of spread between two series.

    Used for rate_spread_z (COL10Y - UST10Y) and yield_curve_z (UST10Y - UST2Y).

    Args:
        series1: First series (e.g., col10y, ust10y)
        series2: Second series (e.g., ust10y, ust2y)
        window: Rolling window for z-score calculation (default: 252 ~1 year)
        spread_formula: Description of spread calculation (for documentation)

    Returns:
        Z-score of the spread, clipped to [-5, 5]

    Example:
        >>> df['rate_spread_z'] = calculate_spread_zscore(df['col10y'], df['ust10y'])
        >>> df['yield_curve_z'] = calculate_spread_zscore(df['ust10y'], df['ust2y'])
    """
    spread = series1 - series2
    mean = spread.rolling(window=window, min_periods=window // 2).mean()
    std = spread.rolling(window=window, min_periods=window // 2).std()
    z = (spread - mean) / std.clip(lower=1e-8)
    return z.clip(-5, 5)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_adx(adx: pd.Series, expected_mean_max: float = 50.0) -> bool:
    """
    Validate that ADX is not saturated.

    Args:
        adx: ADX series to validate
        expected_mean_max: Maximum acceptable mean (default: 50)

    Returns:
        bool: True if valid, False if saturated
    """
    mean = adx.mean()
    std = adx.std()

    if mean > expected_mean_max:
        logger.error(f"ADX SATURATED: mean={mean:.2f} > {expected_mean_max}")
        return False

    if std < 10:
        logger.warning(f"ADX std low: {std:.2f}, may indicate issues")

    logger.info(f"ADX validation passed: mean={mean:.2f}, std={std:.2f}")
    return True


def validate_rsi(rsi: pd.Series) -> bool:
    """
    Validate RSI is in valid range.

    Args:
        rsi: RSI series to validate

    Returns:
        bool: True if valid
    """
    if rsi.min() < 0 or rsi.max() > 100:
        logger.error(f"RSI out of range: min={rsi.min():.2f}, max={rsi.max():.2f}")
        return False

    # Check for reasonable distribution
    pct_extreme = ((rsi < 10) | (rsi > 90)).sum() / len(rsi)
    if pct_extreme > 0.2:
        logger.warning(f"RSI has {pct_extreme:.1%} extreme values")

    return True


def validate_zscore(zscore: pd.Series, feature_name: str = "unknown") -> bool:
    """
    Validate z-score has reasonable distribution.

    A well-formed z-score should have:
    - Mean close to 0
    - Std close to 1
    - Few extreme values (>3 std)

    Args:
        zscore: Z-score series to validate
        feature_name: Name for logging

    Returns:
        bool: True if valid
    """
    mean = zscore.mean()
    std = zscore.std()

    if abs(mean) > 0.5:
        logger.warning(f"{feature_name} z-score mean={mean:.2f}, expected ~0")

    if std < 0.5 or std > 2.0:
        logger.warning(f"{feature_name} z-score std={std:.2f}, expected ~1")

    pct_extreme = ((zscore < -3) | (zscore > 3)).sum() / len(zscore)
    if pct_extreme > 0.05:
        logger.warning(f"{feature_name} has {pct_extreme:.1%} extreme z-scores (>3 std)")

    return True


def validate_volatility_pct(volatility: pd.Series, asset_type: str = "FX") -> bool:
    """
    Validate realized volatility is in reasonable range.

    Expected ranges (annualized):
    - FX majors: 5-15%
    - FX EM (like USDCOP): 8-25%
    - Equities: 15-40%

    Args:
        volatility: Volatility series (annualized)
        asset_type: "FX", "FX_EM", or "EQUITY"

    Returns:
        bool: True if valid
    """
    mean = volatility.mean()

    expected_ranges = {
        "FX": (0.03, 0.20),
        "FX_EM": (0.05, 0.35),
        "EQUITY": (0.10, 0.50),
    }

    min_exp, max_exp = expected_ranges.get(asset_type, (0.05, 0.40))

    if mean < min_exp or mean > max_exp:
        logger.warning(
            f"Volatility mean={mean:.2%} outside expected range "
            f"[{min_exp:.0%}, {max_exp:.0%}] for {asset_type}"
        )

    # Check for NaN values
    nan_pct = volatility.isna().sum() / len(volatility)
    if nan_pct > 0.05:
        logger.warning(f"Volatility has {nan_pct:.1%} NaN values")
        return False

    logger.info(f"Volatility validation passed: mean={mean:.2%}")
    return True


def validate_trend_z(trend_z: pd.Series) -> bool:
    """
    Validate trend z-score has reasonable distribution.

    A well-formed trend_z should have:
    - Mean close to 0 (no long-term bias)
    - Std close to 1
    - Reasonable autocorrelation (trends persist)

    Args:
        trend_z: Trend z-score series

    Returns:
        bool: True if valid
    """
    mean = trend_z.mean()
    std = trend_z.std()

    valid = True

    if abs(mean) > 0.3:
        logger.warning(f"Trend z-score mean={mean:.2f}, expected ~0")
        # This is a warning, not an error - could indicate long-term trend

    if std < 0.5 or std > 1.5:
        logger.warning(f"Trend z-score std={std:.2f}, expected ~1")

    # Check autocorrelation (trend should persist somewhat)
    if len(trend_z) > 100:
        autocorr = trend_z.autocorr(lag=1)
        if autocorr < 0.9:
            logger.warning(
                f"Trend z-score autocorr(1)={autocorr:.2f}, "
                f"expected >0.9 for trending indicator"
            )

    logger.info(f"Trend z-score validation: mean={mean:.2f}, std={std:.2f}")
    return valid


# =============================================================================
# Feature Engineering Helpers
# =============================================================================

def get_close_only_features(
    close: pd.Series,
    include_momentum: bool = False
) -> pd.DataFrame:
    """
    Calculate all CLOSE-only technical features in one call.

    Convenience function that returns a DataFrame with all standard
    CLOSE-only features for the v2.0 architecture.

    Args:
        close: Serie de precios de cierre
        include_momentum: Si True, incluye momentum_z adicional

    Returns:
        pd.DataFrame with columns:
        - log_ret_5m: Log return 1 bar
        - log_ret_1h: Log return 12 bars
        - log_ret_4h: Log return 48 bars
        - rsi_9: RSI período 9
        - volatility_pct: Realized volatility (annualized)
        - trend_z: Trend z-score
        - momentum_z: (optional) Momentum z-score

    Example:
        >>> features = get_close_only_features(df['close'])
        >>> df = pd.concat([df, features], axis=1)
    """
    result = pd.DataFrame(index=close.index)

    # Log returns at different horizons
    result['log_ret_5m'] = calculate_log_returns(close, periods=1)
    result['log_ret_1h'] = calculate_log_returns(close, periods=12)
    result['log_ret_4h'] = calculate_log_returns(close, periods=48)

    # RSI
    result['rsi_9'] = calculate_rsi_wilders(close, period=9)

    # CLOSE-only volatility and trend
    result['volatility_pct'] = calculate_volatility_pct(close, period=14)
    result['trend_z'] = calculate_trend_z(close, sma_period=50)

    if include_momentum:
        result['momentum_z'] = calculate_momentum_z(close, period=20)

    return result
