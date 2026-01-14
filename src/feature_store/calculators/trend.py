"""
Trend Feature Calculators
==========================
Calculators for trend-based indicators (EMA, SMA, etc.)

These are the SINGLE SOURCE OF TRUTH for trend calculations.
All pipelines (training, backtest, inference) MUST use these.

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

import numpy as np
import pandas as pd

from .base import FeatureCalculator, CalculatorRegistry
from ..contracts import FeatureSpec, SmoothingMethod


@CalculatorRegistry.register("ema")
class EMACalculator(FeatureCalculator):
    """
    Exponential Moving Average Calculator

    EMA = Close * alpha + EMA[prev] * (1 - alpha)
    where alpha = 2 / (period + 1)
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate EMA.

        Args:
            data: DataFrame with 'close' column

        Returns:
            EMA values
        """
        self.validate_input(data)

        close = data["close"]
        window = self.spec.calculation.window

        ema = self._apply_smoothing(close, SmoothingMethod.EMA, window)

        return ema


@CalculatorRegistry.register("sma")
class SMACalculator(FeatureCalculator):
    """
    Simple Moving Average Calculator

    SMA = sum(Close[i]) / n for i in range(n)
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate SMA.

        Args:
            data: DataFrame with 'close' column

        Returns:
            SMA values
        """
        self.validate_input(data)

        close = data["close"]
        window = self.spec.calculation.window

        sma = self._apply_smoothing(close, SmoothingMethod.SMA, window)

        return sma


@CalculatorRegistry.register("ema_distance")
class EMADistanceCalculator(FeatureCalculator):
    """
    Distance from EMA as Percentage

    Distance = (Close - EMA) / EMA * 100

    Positive: Above EMA (bullish)
    Negative: Below EMA (bearish)
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate distance from EMA.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Distance percentage
        """
        self.validate_input(data)

        close = data["close"]
        window = self.spec.calculation.window

        ema = self._apply_smoothing(close, SmoothingMethod.EMA, window)
        distance = ((close - ema) / ema) * 100

        return distance.fillna(0.0)


@CalculatorRegistry.register("ema_crossover")
class EMACrossoverCalculator(FeatureCalculator):
    """
    EMA Crossover Signal

    Uses fast EMA (9) and slow EMA (21) by default.

    Returns:
    - +1: Fast above slow (bullish)
    - -1: Fast below slow (bearish)
    -  0: Cross happening
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate EMA crossover signal.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Crossover signal (-1, 0, 1)
        """
        self.validate_input(data)

        close = data["close"]
        fast_period = 9
        slow_period = 21

        fast_ema = self._apply_smoothing(close, SmoothingMethod.EMA, fast_period)
        slow_ema = self._apply_smoothing(close, SmoothingMethod.EMA, slow_period)

        # Calculate crossover signal
        crossover = np.sign(fast_ema - slow_ema)

        return crossover.fillna(0.0)


@CalculatorRegistry.register("price_position")
class PricePositionCalculator(FeatureCalculator):
    """
    Price Position in Range

    Position = (Close - Low[n]) / (High[n] - Low[n])

    Where High[n] and Low[n] are rolling highs/lows.

    Values:
    - 0.0: At recent low
    - 0.5: Middle of range
    - 1.0: At recent high
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price position in range.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Position normalized to 0-1
        """
        self.validate_input(data)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        window = self.spec.calculation.window

        rolling_high = high.rolling(window=window, min_periods=1).max()
        rolling_low = low.rolling(window=window, min_periods=1).min()

        price_range = rolling_high - rolling_low
        position = (close - rolling_low) / price_range.replace(0, np.nan)

        return position.fillna(0.5)


@CalculatorRegistry.register("trend_strength")
class TrendStrengthCalculator(FeatureCalculator):
    """
    Trend Strength Indicator

    Measures how far price has moved from its starting point
    in the lookback window, normalized by volatility.

    Strength = (Close - Close[n]) / ATR
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate trend strength.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Trend strength value
        """
        self.validate_input(data)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        window = self.spec.calculation.window

        # Calculate ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = self._apply_smoothing(tr, SmoothingMethod.WILDER, window)

        # Calculate price change
        price_change = close - close.shift(window)

        # Normalize by ATR
        strength = price_change / atr.replace(0, np.nan)

        return strength.fillna(0.0)


@CalculatorRegistry.register("linear_regression_slope")
class LinearRegressionSlopeCalculator(FeatureCalculator):
    """
    Linear Regression Slope

    Calculates the slope of a linear regression line
    fitted to the last n closes.

    Normalized by close price for comparability.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate linear regression slope.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Normalized slope value
        """
        self.validate_input(data)

        close = data["close"]
        window = self.spec.calculation.window

        def calc_slope(arr):
            if len(arr) < 2:
                return 0.0
            x = np.arange(len(arr))
            # Simple linear regression: slope = cov(x,y) / var(x)
            mean_x = x.mean()
            mean_y = arr.mean()
            numerator = ((x - mean_x) * (arr - mean_y)).sum()
            denominator = ((x - mean_x) ** 2).sum()
            if denominator == 0:
                return 0.0
            return numerator / denominator

        slope = close.rolling(window=window, min_periods=2).apply(
            calc_slope, raw=True
        )

        # Normalize by price level
        normalized_slope = slope / close * 100

        return normalized_slope.fillna(0.0)


@CalculatorRegistry.register("higher_highs_count")
class HigherHighsCountCalculator(FeatureCalculator):
    """
    Count of Higher Highs in Lookback Window

    Counts how many consecutive higher highs exist.
    Positive value indicates uptrend.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate higher highs count.

        Args:
            data: DataFrame with 'high' column

        Returns:
            Count of consecutive higher highs
        """
        self.validate_input(data)

        high = data["high"]
        window = self.spec.calculation.window

        # Is this high higher than previous?
        is_higher = (high > high.shift(1)).astype(int)

        # Rolling sum
        count = is_higher.rolling(window=window, min_periods=1).sum()

        return count.fillna(0.0)


@CalculatorRegistry.register("candle_direction")
class CandleDirectionCalculator(FeatureCalculator):
    """
    Candle Direction Indicator

    +1: Bullish candle (close > open)
    -1: Bearish candle (close < open)
     0: Doji (close == open)
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate candle direction.

        Args:
            data: DataFrame with 'open', 'close' columns

        Returns:
            Direction signal (-1, 0, 1)
        """
        self.validate_input(data)

        open_price = data["open"]
        close = data["close"]

        direction = np.sign(close - open_price)

        return direction.fillna(0.0)
