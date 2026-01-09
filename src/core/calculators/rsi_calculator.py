"""
RSICalculator - RSI Feature Calculator
=======================================

Calculates Relative Strength Index (RSI) indicator.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_calculator import BaseFeatureCalculator
from ..interfaces.normalizer import INormalizer


class RSICalculator(BaseFeatureCalculator):
    """
    Calculator for RSI (Relative Strength Index) - range 0-100.

    CRITICAL: Default period=9 per MAPEO Part 3, Line 360

    Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over period

    Example:
        calc = RSICalculator(period=9)
        rsi = calc.calculate(df)  # df must have 'close' column
    """

    def __init__(self,
                 period: int = 9,
                 normalizer: Optional[INormalizer] = None,
                 clip_bounds: Optional[tuple] = None):
        """
        Initialize RSI calculator.

        Args:
            period: RSI period (default: 9 per MAPEO)
            normalizer: Optional normalizer strategy
            clip_bounds: Optional clip bounds
        """
        super().__init__(
            name=f'rsi_{period}',
            dependencies=['close'],
            normalizer=normalizer,
            clip_bounds=clip_bounds
        )
        self.period = period

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute RSI values.

        Args:
            data: DataFrame with 'close' column
            **kwargs: Additional parameters (period can override)

        Returns:
            pd.Series with RSI values (0-100)
        """
        period = kwargs.get('period', self.period)
        close = data['close']

        # Calculate price changes
        delta = close.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_params(self) -> dict:
        """Get calculator parameters."""
        params = super().get_params()
        params['period'] = self.period
        return params
