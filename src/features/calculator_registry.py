"""
Calculator Registry - Dynamic Feature Calculation from SSOT
============================================================

This module provides the bridge between pipeline_ssot.yaml feature definitions
and actual calculation functions. L2 uses this to dynamically calculate
features defined in the SSOT.

Usage:
    from src.features.calculator_registry import CalculatorRegistry, calculate_features_ssot

    # Option 1: Calculate all features from SSOT
    df_features = calculate_features_ssot(df_ohlcv, df_macro)

    # Option 2: Use registry directly
    registry = CalculatorRegistry()
    result = registry.calculate("log_return", df['close'], periods=1)

Contract: CTR-CALCULATOR-REGISTRY-001
Version: 1.0.0
Date: 2026-02-03
"""

import logging
import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# BUILT-IN CALCULATORS
# =============================================================================

def calculate_log_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns over specified periods."""
    return np.log(series / series.shift(periods))


def calculate_rsi_wilders(series: pd.Series, period: int = 9) -> pd.Series:
    """Calculate RSI using Wilders smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.clip(lower=1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi.clip(0, 100)


def calculate_volatility_pct(
    series: pd.Series,
    period: int = 14,
    annualize: bool = True,
    bars_per_day: int = 48
) -> pd.Series:
    """Calculate realized volatility from returns."""
    log_returns = np.log(series / series.shift(1))
    rolling_std = log_returns.rolling(window=period, min_periods=period // 2).std()

    if annualize:
        annualization_factor = np.sqrt(252 * bars_per_day)
        return rolling_std * annualization_factor
    return rolling_std


def calculate_volume_zscore(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    V22 P3: Rolling z-score of volume.

    Detects volume spikes/lulls for LSTM temporal patterns.

    Args:
        series: Volume series
        window: Rolling window size (default 20 bars)

    Returns:
        Z-score of volume
    """
    rolling_mean = series.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = series.rolling(window=window, min_periods=window // 2).std().clip(lower=1e-8)
    return (series - rolling_mean) / rolling_std


def calculate_trend_z(
    series: pd.Series,
    sma_period: int = 50,
    clip_value: float = 3.0
) -> pd.Series:
    """Calculate trend strength as z-score of price vs SMA."""
    sma = series.rolling(window=sma_period, min_periods=sma_period // 2).mean()
    rolling_std = series.rolling(window=sma_period, min_periods=sma_period // 2).std()
    trend_z = (series - sma) / rolling_std.clip(lower=1e-6)
    return trend_z.clip(-clip_value, clip_value)


def calculate_macro_zscore(
    series: pd.Series,
    window: int = 252,
    method: str = 'rolling'
) -> pd.Series:
    """Calculate z-score for macro variables using rolling window."""
    if method == 'rolling':
        mean = series.rolling(window=window, min_periods=window // 2).mean()
        std = series.rolling(window=window, min_periods=window // 2).std()
    else:  # expanding
        mean = series.expanding().mean()
        std = series.expanding().std()

    z = (series - mean) / std.clip(lower=1e-8)
    return z.clip(-10, 10)


def calculate_spread_zscore(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 252,
    spread_formula: str = "series1 - series2"
) -> pd.Series:
    """
    Calculate z-score of spread between two series.

    Args:
        series1: First series (e.g., col10y, ust10y)
        series2: Second series (e.g., ust10y, ust2y)
        window: Rolling window for z-score calculation
        spread_formula: Description of spread calculation (for documentation)

    Returns:
        Z-score of the spread
    """
    spread = series1 - series2
    mean = spread.rolling(window=window, min_periods=window // 2).mean()
    std = spread.rolling(window=window, min_periods=window // 2).std()
    z = (spread - mean) / std.clip(lower=1e-8)
    return z.clip(-5, 5)


def calculate_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage change over periods."""
    return series.pct_change(periods=periods)


def calculate_hurst_exponent(
    series: pd.Series,
    window: int = 120,
    min_window: int = 60,
) -> pd.Series:
    """Calculate rolling Hurst exponent via R/S analysis.

    H > 0.5: Trending (momentum)
    H = 0.5: Random walk
    H < 0.5: Mean-reverting

    Args:
        series: Price series (typically close)
        window: Rolling window size (default 120 bars = 10 hours)
        min_window: Minimum window for valid calculation

    Returns:
        Rolling Hurst exponent in [0, 1], centered around 0.5
    """
    log_prices = np.log(series.clip(lower=1e-8))
    result = pd.Series(np.nan, index=series.index, dtype=np.float64)

    for i in range(window, len(series)):
        segment = log_prices.iloc[i - window:i].values
        returns = np.diff(segment)

        if len(returns) < min_window or np.std(returns) < 1e-10:
            result.iloc[i] = 0.5
            continue

        # R/S analysis with sub-windows
        n = len(returns)
        max_k = min(n // 2, 32)
        if max_k < 4:
            result.iloc[i] = 0.5
            continue

        log_rs = []
        log_n = []
        for k in [4, 8, 16, max_k]:
            if k > n:
                continue
            n_chunks = n // k
            if n_chunks < 1:
                continue

            rs_values = []
            for chunk_idx in range(n_chunks):
                chunk = returns[chunk_idx * k:(chunk_idx + 1) * k]
                mean_r = np.mean(chunk)
                deviations = np.cumsum(chunk - mean_r)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(chunk, ddof=1) if len(chunk) > 1 else 1e-10
                if s > 1e-10:
                    rs_values.append(r / s)

            if rs_values:
                log_rs.append(np.log(np.mean(rs_values)))
                log_n.append(np.log(k))

        if len(log_rs) >= 2:
            # Linear regression: log(R/S) = H * log(n) + c
            coeffs = np.polyfit(log_n, log_rs, 1)
            h = float(np.clip(coeffs[0], 0.0, 1.0))
            result.iloc[i] = h
        else:
            result.iloc[i] = 0.5

    return result


def calculate_cross_pair_lead(
    target: pd.Series,
    leader: pd.Series,
    lookback: int = 15,
) -> pd.Series:
    """Calculate cross-pair lead indicator (log return of leader).

    Uses the lagged return of a leader currency pair as a predictive signal.

    Args:
        target: Target pair close prices (not used directly, for alignment)
        leader: Leader pair close prices (e.g., USDMXN)
        lookback: Period for log return calculation

    Returns:
        Log return of leader pair over lookback period
    """
    return np.log(leader / leader.shift(lookback))


def _atr_percentage_wrapper(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wrapper for calculate_atr_percentage from technical_indicators."""
    from src.features.technical_indicators import calculate_atr_percentage
    return calculate_atr_percentage(high, low, close, period=period)


def calculate_latam_basket_z(
    mxn_close: pd.Series,
    brl_close: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Z-score of equal-weighted MXN+BRL cumulative log returns.

    Captures regional LATAM FX sentiment. Positive = EM currencies weakening.

    Args:
        mxn_close: USD/MXN close prices
        brl_close: USD/BRL close prices
        window: Rolling window for z-score

    Returns:
        Z-scored LATAM basket cumulative return
    """
    mxn_ret = np.log(mxn_close / mxn_close.shift(1))
    brl_ret = np.log(brl_close / brl_close.shift(1))
    basket = (mxn_ret + brl_ret) / 2.0
    cum = basket.cumsum()
    mean = cum.rolling(window=window, min_periods=window // 2).mean()
    std = cum.rolling(window=window, min_periods=window // 2).std()
    return ((cum - mean) / std.clip(lower=1e-8)).clip(-5, 5)


def calculate_cop_vs_peers_z(
    cop_close: pd.Series,
    mxn_close: pd.Series,
    brl_close: pd.Series,
    lookback: int = 15,
    window: int = 240,
) -> pd.Series:
    """COP relative value vs MXN+BRL peers (z-scored).

    Positive = COP depreciated more than peers → mean-reversion probable.

    Args:
        cop_close: USD/COP close prices
        mxn_close: USD/MXN close prices
        brl_close: USD/BRL close prices
        lookback: Period for log return calculation
        window: Rolling window for z-score

    Returns:
        Z-scored relative value of COP vs peers
    """
    cop_ret = np.log(cop_close / cop_close.shift(lookback))
    mxn_ret = np.log(mxn_close / mxn_close.shift(lookback))
    brl_ret = np.log(brl_close / brl_close.shift(lookback))
    peers_ret = (mxn_ret + brl_ret) / 2.0
    relative = cop_ret - peers_ret
    mean = relative.rolling(window=window, min_periods=window // 2).mean()
    std = relative.rolling(window=window, min_periods=window // 2).std()
    return ((relative - mean) / std.clip(lower=1e-8)).clip(-5, 5)


def calculate_carry_to_vol(
    spread: pd.Series,
    volatility: pd.Series,
    clip_value: float = 5.0,
) -> pd.Series:
    """Calculate carry-to-volatility ratio.

    Higher ratio = more attractive carry relative to risk.

    Args:
        spread: Interest rate spread (e.g., COL10Y - UST10Y)
        volatility: Volatility measure (e.g., rolling std of returns)
        clip_value: Maximum absolute value for clipping

    Returns:
        Carry/vol ratio, clipped to [-clip_value, clip_value]
    """
    ratio = spread / volatility.clip(lower=1e-8)
    return ratio.clip(-clip_value, clip_value)


# =============================================================================
# CALCULATOR REGISTRY
# =============================================================================

@dataclass
class CalculatorInfo:
    """Information about a registered calculator."""
    name: str
    function: Callable
    description: str
    requires_multiple_series: bool = False


class CalculatorRegistry:
    """
    Registry of feature calculators.

    Maps calculator names from SSOT to actual Python functions.
    Supports both built-in calculators and dynamic module loading.
    """

    def __init__(self):
        self._calculators: Dict[str, CalculatorInfo] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register all built-in calculators."""
        builtins = [
            CalculatorInfo(
                name="log_return",
                function=calculate_log_returns,
                description="Log return: ln(price_t / price_{t-n})"
            ),
            CalculatorInfo(
                name="rsi_wilders",
                function=calculate_rsi_wilders,
                description="RSI with Wilder's exponential smoothing"
            ),
            CalculatorInfo(
                name="volatility_pct",
                function=calculate_volatility_pct,
                description="Realized volatility as annualized percentage"
            ),
            CalculatorInfo(
                name="trend_z",
                function=calculate_trend_z,
                description="Price position relative to SMA as z-score"
            ),
            CalculatorInfo(
                name="macro_zscore",
                function=calculate_macro_zscore,
                description="Rolling z-score for macro variables"
            ),
            CalculatorInfo(
                name="spread_zscore",
                function=calculate_spread_zscore,
                description="Spread between two series as z-score",
                requires_multiple_series=True
            ),
            CalculatorInfo(
                name="pct_change",
                function=calculate_pct_change,
                description="Simple percentage change"
            ),
            CalculatorInfo(
                name="volume_zscore",
                function=calculate_volume_zscore,
                description="V22 P3: Rolling z-score of volume (20-bar window)"
            ),
            CalculatorInfo(
                name="hurst_exponent",
                function=calculate_hurst_exponent,
                description="Rolling Hurst exponent via R/S analysis (mean-reversion vs trending)"
            ),
            CalculatorInfo(
                name="cross_pair_lead",
                function=calculate_cross_pair_lead,
                description="Cross-pair lead indicator (log return of leader pair)",
                requires_multiple_series=True,
            ),
            CalculatorInfo(
                name="carry_to_vol",
                function=calculate_carry_to_vol,
                description="Carry-to-volatility ratio (spread / vol)",
                requires_multiple_series=True,
            ),
            CalculatorInfo(
                name="atr_percentage",
                function=_atr_percentage_wrapper,
                description="ATR as percentage of price (H/L/C required)",
                requires_multiple_series=True,
            ),
            CalculatorInfo(
                name="latam_basket_z",
                function=calculate_latam_basket_z,
                description="Z-score of equal-weighted LATAM FX basket (MXN+BRL)",
                requires_multiple_series=True,
            ),
            CalculatorInfo(
                name="cop_vs_peers_z",
                function=calculate_cop_vs_peers_z,
                description="COP relative value vs MXN+BRL peers (z-scored)",
                requires_multiple_series=True,
            ),
        ]

        for calc in builtins:
            self._calculators[calc.name] = calc

        logger.debug(f"Registered {len(self._calculators)} built-in calculators")

    def register(
        self,
        name: str,
        function: Callable,
        description: str = "",
        requires_multiple_series: bool = False
    ) -> None:
        """Register a custom calculator."""
        self._calculators[name] = CalculatorInfo(
            name=name,
            function=function,
            description=description,
            requires_multiple_series=requires_multiple_series
        )
        logger.info(f"Registered calculator: {name}")

    def get(self, name: str) -> Optional[CalculatorInfo]:
        """Get calculator by name."""
        return self._calculators.get(name)

    def calculate(
        self,
        calculator_name: str,
        *series: pd.Series,
        **params: Any
    ) -> pd.Series:
        """
        Execute a calculator with given series and parameters.

        Args:
            calculator_name: Name of the calculator
            *series: Input series (1 or more depending on calculator)
            **params: Calculator parameters

        Returns:
            Calculated feature series
        """
        calc_info = self._calculators.get(calculator_name)
        if calc_info is None:
            raise ValueError(f"Unknown calculator: {calculator_name}")

        try:
            return calc_info.function(*series, **params)
        except Exception as e:
            logger.error(f"Calculator {calculator_name} failed: {e}")
            raise

    def list_calculators(self) -> List[str]:
        """List all registered calculator names."""
        return list(self._calculators.keys())


# =============================================================================
# GLOBAL REGISTRY INSTANCE
# =============================================================================

_registry: Optional[CalculatorRegistry] = None


def get_calculator_registry() -> CalculatorRegistry:
    """Get the global calculator registry instance."""
    global _registry
    if _registry is None:
        _registry = CalculatorRegistry()
    return _registry


# =============================================================================
# SSOT-DRIVEN FEATURE CALCULATION
# =============================================================================

def calculate_features_ssot(
    df_ohlcv: pd.DataFrame,
    df_macro: Optional[pd.DataFrame] = None,
    config: Optional["PipelineConfig"] = None,
    df_aux_ohlcv: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate all market features defined in pipeline_ssot.yaml.

    This is the MAIN entry point for L2 dataset builder to calculate features
    dynamically based on SSOT definitions.

    Args:
        df_ohlcv: DataFrame with OHLCV data (must have 'close' column and datetime index)
        df_macro: Optional DataFrame with macro data (must have datetime index)
        config: Optional PipelineConfig instance (loads from default if not provided)
        df_aux_ohlcv: Optional dict of auxiliary OHLCV DataFrames
                      (e.g., {"usdmxn": df_mxn}) for cross-pair features

    Returns:
        Tuple of:
        - DataFrame with all calculated features
        - Dictionary with normalization statistics (for train-only computation)

    Example:
        >>> from src.features.calculator_registry import calculate_features_ssot
        >>> df_features, norm_stats = calculate_features_ssot(df_ohlcv, df_macro)
    """
    # Load config if not provided
    if config is None:
        from src.config.pipeline_config import load_pipeline_config
        config = load_pipeline_config()

    registry = get_calculator_registry()
    result = pd.DataFrame(index=df_ohlcv.index)
    norm_stats = {}

    # Get market feature definitions (not state features)
    market_features = config.get_market_features()

    logger.info(f"Calculating {len(market_features)} market features from SSOT")

    for feature_def in market_features:
        try:
            # Get calculator
            calculator_name = feature_def.calculator
            if calculator_name is None:
                # Custom formula - handle specially
                if feature_def.custom_formula:
                    result[feature_def.name] = _eval_custom_formula(
                        feature_def.custom_formula,
                        df_ohlcv,
                        df_macro
                    )
                    logger.debug(f"Calculated {feature_def.name} via custom formula")
                    continue
                else:
                    logger.warning(f"Feature {feature_def.name} has no calculator, skipping")
                    continue

            # Get input series based on source
            if feature_def.source == "aux_ohlcv" and df_aux_ohlcv:
                # Handle "pairname.column" notation for auxiliary pairs
                input_series = _get_aux_input_series(
                    feature_def.input_columns,
                    df_ohlcv,
                    df_aux_ohlcv,
                    feature_def.preprocessing,
                )
            else:
                input_series = _get_input_series(
                    feature_def.source,
                    feature_def.input_columns,
                    df_ohlcv,
                    df_macro,
                    feature_def.preprocessing
                )

            # Execute calculator
            calc_info = registry.get(calculator_name)
            if calc_info is None:
                logger.warning(f"Calculator {calculator_name} not found for {feature_def.name}")
                continue

            if calc_info.requires_multiple_series:
                # Multi-series calculator (e.g., spread_zscore)
                calculated = registry.calculate(calculator_name, *input_series, **feature_def.params)
            else:
                # Single-series calculator
                calculated = registry.calculate(calculator_name, input_series[0], **feature_def.params)

            # Apply preprocessing clip if specified
            if feature_def.preprocessing.clip_outliers:
                low, high = feature_def.preprocessing.clip_outliers
                calculated = calculated.clip(low, high)

            result[feature_def.name] = calculated
            logger.debug(f"Calculated feature: {feature_def.name}")

        except Exception as e:
            logger.error(f"Failed to calculate {feature_def.name}: {e}")
            # Create NaN column to maintain structure
            result[feature_def.name] = np.nan

    # Add raw_log_ret_5m for PnL calculation (auxiliary feature)
    if 'close' in df_ohlcv.columns:
        result['raw_log_ret_5m'] = calculate_log_returns(df_ohlcv['close'], periods=1)

    # Reorder columns to match SSOT order
    feature_order = [f.name for f in market_features]
    available_cols = [c for c in feature_order if c in result.columns]
    extra_cols = [c for c in result.columns if c not in feature_order]
    result = result[available_cols + extra_cols]

    logger.info(f"Feature calculation complete: {len(result.columns)} columns, {len(result)} rows")

    return result, norm_stats


def _get_input_series(
    source: str,
    input_columns: List[str],
    df_ohlcv: pd.DataFrame,
    df_macro: Optional[pd.DataFrame],
    preprocessing: "PreprocessingConfig"
) -> List[pd.Series]:
    """
    Get input series for a calculator based on source and columns.

    Handles:
    - ohlcv source -> extract from df_ohlcv
    - macro_daily source -> extract from df_macro with T-1 shift
    - macro_monthly source -> extract from df_macro with T-1 shift
    """
    series_list = []

    if source == "ohlcv":
        source_df = df_ohlcv
    elif source in ("macro_daily", "macro_monthly"):
        if df_macro is None:
            raise ValueError(f"Source {source} requires df_macro but it's None")
        source_df = df_macro
    elif source == "aux_ohlcv":
        # Auxiliary OHLCV pairs: input_columns use "pairname.column" notation
        # e.g., "usdmxn.close" → look up "close" from df_aux_ohlcv["usdmxn"]
        # Handled by caller — we split pair.col and find the right DataFrame
        source_df = df_ohlcv  # Fallback; actual resolution below
    else:
        raise ValueError(f"Unknown source: {source}")

    for col in input_columns:
        if col not in source_df.columns:
            raise KeyError(f"Column {col} not found in {source} data")

        s = source_df[col].copy()

        # Apply preprocessing
        if preprocessing.ffill_limit:
            s = s.ffill(limit=preprocessing.ffill_limit)

        if preprocessing.shift > 0:
            s = s.shift(preprocessing.shift)  # T-1 anti-leakage

        if preprocessing.floor is not None:
            s = s.clip(lower=preprocessing.floor)

        series_list.append(s)

    return series_list


def _get_aux_input_series(
    input_columns: List[str],
    df_ohlcv: pd.DataFrame,
    df_aux_ohlcv: Dict[str, pd.DataFrame],
    preprocessing: "PreprocessingConfig",
) -> List[pd.Series]:
    """Get input series from auxiliary OHLCV DataFrames.

    Handles "pairname.column" notation: e.g., "usdmxn.close" looks up
    the "close" column from df_aux_ohlcv["usdmxn"].

    Plain column names (no dot) are resolved from df_ohlcv.
    """
    series_list = []

    for col_spec in input_columns:
        if "." in col_spec:
            pair_name, col_name = col_spec.split(".", 1)
            if pair_name not in df_aux_ohlcv:
                raise KeyError(
                    f"Auxiliary pair '{pair_name}' not found. "
                    f"Available: {list(df_aux_ohlcv.keys())}"
                )
            aux_df = df_aux_ohlcv[pair_name]
            if col_name not in aux_df.columns:
                raise KeyError(f"Column '{col_name}' not in {pair_name} data")

            s = aux_df[col_name].copy()
            # Reindex to match primary OHLCV index
            s = s.reindex(df_ohlcv.index, method="ffill")
        else:
            # Plain column from primary OHLCV
            if col_spec not in df_ohlcv.columns:
                raise KeyError(f"Column '{col_spec}' not found in OHLCV data")
            s = df_ohlcv[col_spec].copy()

        # Apply preprocessing
        if preprocessing.ffill_limit:
            s = s.ffill(limit=preprocessing.ffill_limit)
        if preprocessing.shift > 0:
            s = s.shift(preprocessing.shift)
        if preprocessing.floor is not None:
            s = s.clip(lower=preprocessing.floor)

        series_list.append(s)

    return series_list


def _eval_custom_formula(
    formula: str,
    df_ohlcv: pd.DataFrame,
    df_macro: Optional[pd.DataFrame]
) -> pd.Series:
    """
    Evaluate a custom formula string.

    Supports formulas like:
    - "(col10y - ust10y).diff(1)"
    - "col10y - ust10y"

    WARNING: Only use with trusted formulas from SSOT.
    """
    # Create evaluation context with available columns
    context = {}

    if df_ohlcv is not None:
        for col in df_ohlcv.columns:
            context[col] = df_ohlcv[col]

    if df_macro is not None:
        for col in df_macro.columns:
            context[col] = df_macro[col]

    # Add pandas/numpy for common operations
    context['pd'] = pd
    context['np'] = np

    try:
        result = eval(formula, {"__builtins__": {}}, context)
        if not isinstance(result, pd.Series):
            raise ValueError(f"Formula must return pd.Series, got {type(result)}")
        return result
    except Exception as e:
        logger.error(f"Failed to evaluate formula '{formula}': {e}")
        raise


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_features(
    df: pd.DataFrame,
    config: Optional["PipelineConfig"] = None,
    train_mask: Optional[pd.Series] = None,
    existing_stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize features according to SSOT specifications.

    Args:
        df: DataFrame with calculated features
        config: PipelineConfig instance
        train_mask: Boolean mask for training rows (stats computed only on True rows)
        existing_stats: Pre-computed normalization stats (for inference)

    Returns:
        Tuple of:
        - Normalized DataFrame
        - Dictionary of normalization stats per feature
    """
    if config is None:
        from src.config.pipeline_config import load_pipeline_config
        config = load_pipeline_config()

    result = df.copy()
    stats = existing_stats if existing_stats is not None else {}

    for feature_def in config.get_market_features():
        if feature_def.name not in df.columns:
            continue

        norm = feature_def.normalization
        col = df[feature_def.name]

        # Compute stats on training data only
        if norm.compute_on == "train_only" and train_mask is not None:
            train_col = col[train_mask]
        else:
            train_col = col

        if norm.method == "zscore":
            # Compute or use existing stats
            if feature_def.name not in stats:
                mean = float(train_col.mean())
                std = float(train_col.std())
                stats[feature_def.name] = {"mean": mean, "std": std}
            else:
                mean = stats[feature_def.name]["mean"]
                std = stats[feature_def.name]["std"]

            # Normalize
            normalized = (col - mean) / max(std, 1e-8)

        elif norm.method == "minmax":
            input_range = norm.input_range or (float(train_col.min()), float(train_col.max()))
            output_range = norm.output_range or (0, 1)

            in_min, in_max = input_range
            out_min, out_max = output_range

            normalized = (col - in_min) / max(in_max - in_min, 1e-8)
            normalized = normalized * (out_max - out_min) + out_min

            stats[feature_def.name] = {
                "input_min": in_min, "input_max": in_max,
                "output_min": out_min, "output_max": out_max
            }

        elif norm.method == "clip" or norm.method == "none":
            normalized = col
            stats[feature_def.name] = {"method": norm.method}
        else:
            logger.warning(f"Unknown normalization method {norm.method} for {feature_def.name}")
            normalized = col

        # Apply clipping
        if norm.clip:
            low, high = norm.clip
            normalized = normalized.clip(low, high)

        result[feature_def.name] = normalized

    return result, stats


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    'CalculatorRegistry',
    'get_calculator_registry',
    'calculate_features_ssot',
    'normalize_features',
    # Individual calculators (for direct use)
    'calculate_log_returns',
    'calculate_rsi_wilders',
    'calculate_volatility_pct',
    'calculate_trend_z',
    'calculate_macro_zscore',
    'calculate_spread_zscore',
    'calculate_pct_change',
    'calculate_hurst_exponent',
    'calculate_cross_pair_lead',
    'calculate_carry_to_vol',
    'calculate_latam_basket_z',
    'calculate_cop_vs_peers_z',
]
