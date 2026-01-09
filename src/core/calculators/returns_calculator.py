"""
ReturnsCalculator - Log Returns Feature Calculator
===================================================

Calculates logarithmic returns over specified periods.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_calculator import BaseFeatureCalculator
from ..interfaces.normalizer import INormalizer


class ReturnsCalculator(BaseFeatureCalculator):
    """
    Calculator for logarithmic returns.

    Formula: ln(close / close[-periods])

    Examples:
        # 5-minute returns
        calc = ReturnsCalculator(periods=1, name='log_ret_5m')
        ret_5m = calc.calculate(df)

        # 1-hour returns (12 bars)
        calc = ReturnsCalculator(periods=12, name='log_ret_1h')
        ret_1h = calc.calculate(df)

        # 4-hour returns (48 bars)
        calc = ReturnsCalculator(periods=48, name='log_ret_4h')
        ret_4h = calc.calculate(df)
    """

    def __init__(self,
                 periods: int = 1,
                 name: Optional[str] = None,
                 normalizer: Optional[INormalizer] = None,
                 clip_bounds: Optional[tuple] = None):
        """
        Initialize returns calculator.

        Args:
            periods: Number of periods for return calculation
            name: Custom name (default: 'log_ret_{periods}p')
            normalizer: Optional normalizer strategy
            clip_bounds: Optional clip bounds
        """
        if name is None:
            name = f'log_ret_{periods}p'

        super().__init__(
            name=name,
            dependencies=['close'],
            normalizer=normalizer,
            clip_bounds=clip_bounds
        )
        self.periods = periods

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute logarithmic returns.

        Args:
            data: DataFrame with 'close' column
            **kwargs: Additional parameters (periods can override)

        Returns:
            pd.Series with log returns
        """
        periods = kwargs.get('periods', self.periods)
        close = data['close']

        # Calculate log returns
        log_returns = np.log(close / close.shift(periods))

        return log_returns

    def get_params(self) -> dict:
        """Get calculator parameters."""
        params = super().get_params()
        params['periods'] = self.periods
        return params
