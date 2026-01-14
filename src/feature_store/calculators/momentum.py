"""
Momentum Feature Calculators
=============================
Calculators for momentum-based indicators (RSI, ADX, etc.)

These are the SINGLE SOURCE OF TRUTH for momentum calculations.
All pipelines (training, backtest, inference) MUST use these.

CRITICAL: RSI and ADX use Wilder's smoothing (EMA with alpha=1/n)
to match industry standards and ensure consistency.

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

import numpy as np
import pandas as pd

from .base import FeatureCalculator, CalculatorRegistry
from ..contracts import FeatureSpec, SmoothingMethod


@CalculatorRegistry.register("rsi")
class RSICalculator(FeatureCalculator):
    """
    Relative Strength Index Calculator

    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = EMA(gains) / EMA(losses)

    IMPORTANT: Uses Wilder's smoothing (EMA with alpha=1/period)
    This is the industry standard and matches TA-Lib.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI using Wilder's smoothing.

        Args:
            data: DataFrame with 'close' column

        Returns:
            RSI values (0-100 range, normalized to 0-1)
        """
        self.validate_input(data)

        close = data["close"]
        delta = close.diff()

        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)

        window = self.spec.calculation.window

        # Use Wilder's smoothing (alpha = 1/period)
        avg_gains = self._apply_smoothing(
            gains,
            SmoothingMethod.WILDER,
            window
        )
        avg_losses = self._apply_smoothing(
            losses,
            SmoothingMethod.WILDER,
            window
        )

        # Avoid division by zero
        rs = avg_gains / avg_losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Normalize to 0-1 range
        rsi_normalized = rsi / 100.0

        return rsi_normalized.fillna(0.5)


@CalculatorRegistry.register("adx")
class ADXCalculator(FeatureCalculator):
    """
    Average Directional Index Calculator

    Measures trend strength (0-100).
    Uses Wilder's smoothing for all components.

    Components:
    - +DI: Positive Directional Indicator
    - -DI: Negative Directional Indicator
    - DX: Directional Index = |+DI - -DI| / (+DI + -DI)
    - ADX: Smoothed DX
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ADX using Wilder's smoothing.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            ADX values (normalized to 0-1)
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
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Smooth with Wilder's method
        atr = self._apply_smoothing(tr, SmoothingMethod.WILDER, window)
        plus_di_smooth = self._apply_smoothing(plus_dm, SmoothingMethod.WILDER, window)
        minus_di_smooth = self._apply_smoothing(minus_dm, SmoothingMethod.WILDER, window)

        # Calculate +DI and -DI
        plus_di = 100 * plus_di_smooth / atr.replace(0, np.nan)
        minus_di = 100 * minus_di_smooth / atr.replace(0, np.nan)

        # Calculate DX
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100 * di_diff / di_sum.replace(0, np.nan)

        # Smooth DX to get ADX
        adx = self._apply_smoothing(dx, SmoothingMethod.WILDER, window)

        # Normalize to 0-1 range
        adx_normalized = adx / 100.0

        return adx_normalized.fillna(0.25)


@CalculatorRegistry.register("macd_signal")
class MACDSignalCalculator(FeatureCalculator):
    """
    MACD Signal Line Calculator

    Standard MACD uses:
    - Fast EMA: 12 periods
    - Slow EMA: 26 periods
    - Signal EMA: 9 periods

    Returns the signal line normalized relative to price.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate MACD signal line.

        Args:
            data: DataFrame with 'close' column

        Returns:
            MACD signal normalized to price scale
        """
        self.validate_input(data)

        close = data["close"]

        # Standard MACD parameters
        fast_period = 12
        slow_period = 26
        signal_period = 9

        # Calculate EMAs
        ema_fast = self._apply_smoothing(close, SmoothingMethod.EMA, fast_period)
        ema_slow = self._apply_smoothing(close, SmoothingMethod.EMA, slow_period)

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = self._apply_smoothing(macd_line, SmoothingMethod.EMA, signal_period)

        # Normalize relative to price (as percentage)
        signal_normalized = signal_line / close

        return signal_normalized.fillna(0.0)


@CalculatorRegistry.register("momentum")
class MomentumCalculator(FeatureCalculator):
    """
    Simple Momentum Calculator

    Momentum = (Close - Close[n]) / Close[n]
    where n is the lookback period.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum as percentage change.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Momentum values as percentage
        """
        self.validate_input(data)

        close = data["close"]
        window = self.spec.calculation.window

        momentum = (close - close.shift(window)) / close.shift(window)

        return momentum.fillna(0.0)


@CalculatorRegistry.register("roc")
class ROCCalculator(FeatureCalculator):
    """
    Rate of Change Calculator

    ROC = ((Close - Close[n]) / Close[n]) * 100
    Essentially momentum as percentage.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate rate of change.

        Args:
            data: DataFrame with 'close' column

        Returns:
            ROC values (percentage)
        """
        self.validate_input(data)

        close = data["close"]
        window = self.spec.calculation.window

        roc = ((close - close.shift(window)) / close.shift(window)) * 100

        return roc.fillna(0.0)


@CalculatorRegistry.register("stochastic_k")
class StochasticKCalculator(FeatureCalculator):
    """
    Stochastic %K Calculator

    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Stochastic %K.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            %K values normalized to 0-1
        """
        self.validate_input(data)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        window = self.spec.calculation.window

        lowest_low = low.rolling(window=window, min_periods=1).min()
        highest_high = high.rolling(window=window, min_periods=1).max()

        range_hl = highest_high - lowest_low
        stoch_k = (close - lowest_low) / range_hl.replace(0, np.nan) * 100

        # Normalize to 0-1
        stoch_k_normalized = stoch_k / 100.0

        return stoch_k_normalized.fillna(0.5)


@CalculatorRegistry.register("williams_r")
class WilliamsRCalculator(FeatureCalculator):
    """
    Williams %R Calculator

    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Returns values between -100 and 0, normalized to 0-1.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Williams %R.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Williams %R normalized to 0-1
        """
        self.validate_input(data)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        window = self.spec.calculation.window

        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()

        range_hl = highest_high - lowest_low
        williams_r = ((highest_high - close) / range_hl.replace(0, np.nan)) * -100

        # Normalize from [-100, 0] to [0, 1]
        williams_r_normalized = (williams_r + 100) / 100.0

        return williams_r_normalized.fillna(0.5)
