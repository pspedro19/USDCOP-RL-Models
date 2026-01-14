"""
Volatility Feature Calculators
===============================
Calculators for volatility-based indicators (ATR, Bollinger, etc.)

These are the SINGLE SOURCE OF TRUTH for volatility calculations.
All pipelines (training, backtest, inference) MUST use these.

CRITICAL: ATR uses Wilder's smoothing (EMA with alpha=1/n)
to match industry standards.

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

import numpy as np
import pandas as pd

from .base import FeatureCalculator, CalculatorRegistry
from ..contracts import FeatureSpec, SmoothingMethod


@CalculatorRegistry.register("atr")
class ATRCalculator(FeatureCalculator):
    """
    Average True Range Calculator

    True Range = max(H-L, |H-Prev Close|, |L-Prev Close|)
    ATR = Wilder's smoothed True Range

    IMPORTANT: Uses Wilder's smoothing (EMA with alpha=1/period)
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR using Wilder's smoothing.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            ATR values (absolute, not normalized)
        """
        self.validate_input(data)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        window = self.spec.calculation.window

        # Calculate True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Apply Wilder's smoothing
        atr = self._apply_smoothing(true_range, SmoothingMethod.WILDER, window)

        return atr.fillna(0.0)


@CalculatorRegistry.register("atr_pct")
class ATRPercentCalculator(FeatureCalculator):
    """
    ATR as Percentage of Close Price

    ATR% = ATR / Close * 100

    This normalizes ATR relative to price level,
    making it comparable across different price ranges.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR as percentage of close.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            ATR percentage values
        """
        self.validate_input(data)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        window = self.spec.calculation.window

        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Apply Wilder's smoothing
        atr = self._apply_smoothing(true_range, SmoothingMethod.WILDER, window)

        # Convert to percentage
        atr_pct = (atr / close) * 100

        return atr_pct.fillna(0.0)


@CalculatorRegistry.register("bollinger_width")
class BollingerWidthCalculator(FeatureCalculator):
    """
    Bollinger Band Width Calculator

    Width = (Upper - Lower) / Middle * 100

    Measures volatility relative to moving average.
    Higher values indicate higher volatility.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Bollinger Band width.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Bollinger width as percentage
        """
        self.validate_input(data)

        close = data["close"]
        window = self.spec.calculation.window
        num_std = 2.0  # Standard Bollinger uses 2 std devs

        # Calculate middle band (SMA)
        middle = close.rolling(window=window, min_periods=1).mean()

        # Calculate standard deviation
        std = close.rolling(window=window, min_periods=1).std()

        # Calculate bands
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        # Calculate width as percentage
        width = ((upper - lower) / middle) * 100

        return width.fillna(0.0)


@CalculatorRegistry.register("bollinger_position")
class BollingerPositionCalculator(FeatureCalculator):
    """
    Bollinger Band Position Calculator

    Position = (Close - Lower) / (Upper - Lower)

    Values:
    - 0.0: At lower band
    - 0.5: At middle band
    - 1.0: At upper band
    - >1.0: Above upper band
    - <0.0: Below lower band
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate position within Bollinger Bands.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Position normalized to ~0-1 range
        """
        self.validate_input(data)

        close = data["close"]
        window = self.spec.calculation.window
        num_std = 2.0

        # Calculate bands
        middle = close.rolling(window=window, min_periods=1).mean()
        std = close.rolling(window=window, min_periods=1).std()
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        # Calculate position
        band_range = upper - lower
        position = (close - lower) / band_range.replace(0, np.nan)

        return position.fillna(0.5)


@CalculatorRegistry.register("volatility_ratio")
class VolatilityRatioCalculator(FeatureCalculator):
    """
    Volatility Ratio Calculator

    Compares short-term to long-term volatility.
    VR = StdDev(short) / StdDev(long)

    Values > 1 indicate increasing volatility.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility ratio.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Volatility ratio
        """
        self.validate_input(data)

        close = data["close"]
        returns = close.pct_change()

        short_window = self.spec.calculation.window
        long_window = short_window * 4  # Long-term = 4x short-term

        short_vol = returns.rolling(window=short_window, min_periods=1).std()
        long_vol = returns.rolling(window=long_window, min_periods=1).std()

        vol_ratio = short_vol / long_vol.replace(0, np.nan)

        return vol_ratio.fillna(1.0)


@CalculatorRegistry.register("historical_volatility")
class HistoricalVolatilityCalculator(FeatureCalculator):
    """
    Historical Volatility Calculator

    HV = StdDev(log returns) * sqrt(252) * 100

    Annualized volatility as percentage.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate annualized historical volatility.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Historical volatility (annualized %)
        """
        self.validate_input(data)

        close = data["close"]
        window = self.spec.calculation.window

        # Calculate log returns
        log_returns = np.log(close / close.shift(1))

        # Calculate rolling std of log returns
        rolling_std = log_returns.rolling(window=window, min_periods=1).std()

        # Annualize (assuming 252 trading days)
        hv = rolling_std * np.sqrt(252) * 100

        return hv.fillna(0.0)


@CalculatorRegistry.register("parkinson_volatility")
class ParkinsonVolatilityCalculator(FeatureCalculator):
    """
    Parkinson Volatility Estimator

    More efficient estimator using high-low range:
    PV = sqrt(1/(4*ln(2)) * sum((ln(H/L))^2) / n)

    Better than close-to-close for intraday data.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Parkinson volatility.

        Args:
            data: DataFrame with 'high', 'low' columns

        Returns:
            Parkinson volatility estimate
        """
        self.validate_input(data)

        high = data["high"]
        low = data["low"]
        window = self.spec.calculation.window

        # Log high-low ratio squared
        log_hl_sq = (np.log(high / low)) ** 2

        # Parkinson constant
        constant = 1 / (4 * np.log(2))

        # Rolling mean and sqrt
        rolling_mean = log_hl_sq.rolling(window=window, min_periods=1).mean()
        parkinson = np.sqrt(constant * rolling_mean)

        # Annualize
        parkinson_annual = parkinson * np.sqrt(252) * 100

        return parkinson_annual.fillna(0.0)


@CalculatorRegistry.register("range_pct")
class RangePercentCalculator(FeatureCalculator):
    """
    High-Low Range as Percentage

    Range% = (High - Low) / Close * 100

    Simple measure of intraday volatility.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate high-low range as percentage.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Range percentage
        """
        self.validate_input(data)

        high = data["high"]
        low = data["low"]
        close = data["close"]

        range_pct = ((high - low) / close) * 100

        return range_pct.fillna(0.0)
