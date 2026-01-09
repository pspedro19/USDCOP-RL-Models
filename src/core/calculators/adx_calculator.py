"""
ADXCalculator - ADX Feature Calculator
=======================================

Calculates Average Directional Index (ADX) indicator.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_calculator import BaseFeatureCalculator
from ..interfaces.normalizer import INormalizer


class ADXCalculator(BaseFeatureCalculator):
    """
    Calculator for ADX (Average Directional Index) - range 0-100.

    CRITICAL: Default period=14 per MAPEO Part 3, Line 362

    Formula:
        1. Calculate True Range (TR)
        2. Calculate Directional Movement (+DM, -DM)
        3. Calculate Directional Indicators (+DI, -DI)
        4. Calculate DX = 100 * |+DI - -DI| / (+DI + -DI)
        5. Calculate ADX = EMA(DX, period)

    Example:
        calc = ADXCalculator(period=14)
        adx = calc.calculate(df)  # df must have high, low, close
    """

    def __init__(self,
                 period: int = 14,
                 normalizer: Optional[INormalizer] = None,
                 clip_bounds: Optional[tuple] = None):
        """
        Initialize ADX calculator.

        Args:
            period: ADX period (default: 14 per MAPEO)
            normalizer: Optional normalizer strategy
            clip_bounds: Optional clip bounds
        """
        super().__init__(
            name=f'adx_{period}',
            dependencies=['high', 'low', 'close'],
            normalizer=normalizer,
            clip_bounds=clip_bounds
        )
        self.period = period

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute ADX values.

        Args:
            data: DataFrame with high, low, close columns
            **kwargs: Additional parameters (period can override)

        Returns:
            pd.Series with ADX values (0-100)
        """
        period = kwargs.get('period', self.period)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Calculate Directional Indicators
        plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(window=period).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(window=period).mean() / (atr + 1e-10)

        # Calculate DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()

        return adx

    def get_params(self) -> dict:
        """Get calculator parameters."""
        params = super().get_params()
        params['period'] = self.period
        return params
