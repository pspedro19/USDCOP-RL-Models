"""
Macro Feature Calculators
==========================
Calculators for macro economic indicators (DXY, VIX, WTI, etc.)

These are the SINGLE SOURCE OF TRUTH for macro calculations.
All pipelines (training, backtest, inference) MUST use these.

CRITICAL: Z-score normalization uses rolling window to prevent look-ahead bias.

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

import numpy as np
import pandas as pd

from .base import FeatureCalculator, CalculatorRegistry
from ..contracts import FeatureSpec, SmoothingMethod, NormalizationMethod


@CalculatorRegistry.register("macro_zscore")
class MacroZScoreCalculator(FeatureCalculator):
    """
    Macro Indicator Z-Score Calculator

    Calculates rolling z-score for macro indicators like DXY, VIX, etc.

    CRITICAL: Uses rolling window for z-score to prevent look-ahead bias.
    The window size should match training configuration.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate rolling z-score for macro indicator.

        Args:
            data: DataFrame with the macro indicator column

        Returns:
            Z-score normalized values
        """
        self.validate_input(data)

        # Get the column name from requires
        column = self.spec.requires[0]
        values = data[column]
        window = self.spec.calculation.window

        # Calculate rolling z-score
        rolling_mean = values.rolling(window=window, min_periods=1).mean()
        rolling_std = values.rolling(window=window, min_periods=1).std()

        # Prevent division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (values - rolling_mean) / rolling_std

        # Clip to standard range
        zscore = zscore.clip(lower=-3.0, upper=3.0)

        return zscore.fillna(0.0)


@CalculatorRegistry.register("dxy_zscore")
class DXYZScoreCalculator(FeatureCalculator):
    """
    Dollar Index (DXY) Z-Score Calculator

    Uses 60-bar rolling window for z-score normalization.
    Consistent between training and inference.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate DXY z-score.

        Args:
            data: DataFrame with 'dxy' column

        Returns:
            DXY z-score
        """
        self.validate_input(data)

        dxy = data["dxy"]
        window = self.spec.calculation.window  # Default: 60

        # Rolling z-score
        rolling_mean = dxy.rolling(window=window, min_periods=1).mean()
        rolling_std = dxy.rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (dxy - rolling_mean) / rolling_std
        zscore = zscore.clip(lower=-3.0, upper=3.0)

        return zscore.fillna(0.0)


@CalculatorRegistry.register("vix_zscore")
class VIXZScoreCalculator(FeatureCalculator):
    """
    VIX Z-Score Calculator

    Higher VIX = higher fear/volatility.
    Normalized using rolling z-score.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate VIX z-score.

        Args:
            data: DataFrame with 'vix' column

        Returns:
            VIX z-score
        """
        self.validate_input(data)

        vix = data["vix"]
        window = self.spec.calculation.window

        # Rolling z-score
        rolling_mean = vix.rolling(window=window, min_periods=1).mean()
        rolling_std = vix.rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (vix - rolling_mean) / rolling_std
        zscore = zscore.clip(lower=-3.0, upper=3.0)

        return zscore.fillna(0.0)


@CalculatorRegistry.register("wti_zscore")
class WTIZScoreCalculator(FeatureCalculator):
    """
    WTI Oil Z-Score Calculator

    Oil price changes can affect USD/COP due to
    Colombia's oil exports.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate WTI z-score.

        Args:
            data: DataFrame with 'wti' column

        Returns:
            WTI z-score
        """
        self.validate_input(data)

        wti = data["wti"]
        window = self.spec.calculation.window

        # Rolling z-score
        rolling_mean = wti.rolling(window=window, min_periods=1).mean()
        rolling_std = wti.rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (wti - rolling_mean) / rolling_std
        zscore = zscore.clip(lower=-3.0, upper=3.0)

        return zscore.fillna(0.0)


@CalculatorRegistry.register("embi_zscore")
class EMBIZScoreCalculator(FeatureCalculator):
    """
    EMBI (Emerging Market Bond Index) Z-Score Calculator

    Higher EMBI spread = higher EM risk = weaker EM currencies.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate EMBI z-score.

        Args:
            data: DataFrame with 'embi' column

        Returns:
            EMBI z-score
        """
        self.validate_input(data)

        embi = data["embi"]
        window = self.spec.calculation.window

        # Rolling z-score
        rolling_mean = embi.rolling(window=window, min_periods=1).mean()
        rolling_std = embi.rolling(window=window, min_periods=1).std()
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (embi - rolling_mean) / rolling_std
        zscore = zscore.clip(lower=-3.0, upper=3.0)

        return zscore.fillna(0.0)


@CalculatorRegistry.register("macro_momentum")
class MacroMomentumCalculator(FeatureCalculator):
    """
    Macro Indicator Momentum Calculator

    Calculates rate of change for macro indicators.
    Momentum = (Value - Value[n]) / Value[n]
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate macro indicator momentum.

        Args:
            data: DataFrame with the macro indicator column

        Returns:
            Momentum as percentage change
        """
        self.validate_input(data)

        column = self.spec.requires[0]
        values = data[column]
        window = self.spec.calculation.window

        momentum = (values - values.shift(window)) / values.shift(window).replace(0, np.nan)

        return momentum.fillna(0.0)


@CalculatorRegistry.register("correlation_rolling")
class RollingCorrelationCalculator(FeatureCalculator):
    """
    Rolling Correlation Calculator

    Calculates rolling correlation between two series.
    Used for macro-price correlation features.
    """

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate rolling correlation.

        Args:
            data: DataFrame with two columns specified in requires

        Returns:
            Rolling correlation (-1 to 1)
        """
        self.validate_input(data)

        col1 = self.spec.requires[0]
        col2 = self.spec.requires[1]
        window = self.spec.calculation.window

        series1 = data[col1]
        series2 = data[col2]

        correlation = series1.rolling(window=window, min_periods=1).corr(series2)

        return correlation.fillna(0.0)
