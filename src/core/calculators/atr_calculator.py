"""
ATRCalculator - ATR Feature Calculator
=======================================

Calculates Average True Range (ATR) and ATR%.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_calculator import BaseFeatureCalculator
from ..interfaces.normalizer import INormalizer


class ATRCalculator(BaseFeatureCalculator):
    """
    Calculator for Average True Range (ATR) as percentage of close price.

    CRITICAL: Default period=10 per MAPEO Part 3, Line 361

    Formula:
        TR = max(high - low, |high - close_prev|, |low - close_prev|)
        ATR = EMA(TR, period)
        ATR% = (ATR / close) * 100

    Example:
        calc = ATRCalculator(period=10, as_percentage=True)
        atr_pct = calc.calculate(df)  # df must have high, low, close
    """

    def __init__(self,
                 period: int = 10,
                 as_percentage: bool = True,
                 normalizer: Optional[INormalizer] = None,
                 clip_bounds: Optional[tuple] = None):
        """
        Initialize ATR calculator.

        Args:
            period: ATR period (default: 10 per MAPEO)
            as_percentage: If True, return ATR as % of close (default: True)
            normalizer: Optional normalizer strategy
            clip_bounds: Optional clip bounds
        """
        name = 'atr_pct' if as_percentage else f'atr_{period}'
        super().__init__(
            name=name,
            dependencies=['high', 'low', 'close'],
            normalizer=normalizer,
            clip_bounds=clip_bounds
        )
        self.period = period
        self.as_percentage = as_percentage

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute ATR or ATR% values.

        Args:
            data: DataFrame with high, low, close columns
            **kwargs: Additional parameters (period, as_percentage can override)

        Returns:
            pd.Series with ATR or ATR% values
        """
        period = kwargs.get('period', self.period)
        as_percentage = kwargs.get('as_percentage', self.as_percentage)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR
        atr = tr.rolling(window=period).mean()

        # Return as percentage if requested
        if as_percentage:
            return (atr / close) * 100
        else:
            return atr

    def get_params(self) -> dict:
        """Get calculator parameters."""
        params = super().get_params()
        params['period'] = self.period
        params['as_percentage'] = self.as_percentage
        return params
