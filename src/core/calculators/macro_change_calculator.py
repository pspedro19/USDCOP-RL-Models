"""
MacroChangeCalculator - Macro Change Feature Calculator
========================================================

Calculates percentage changes for macro indicators with clipping.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_calculator import BaseFeatureCalculator
from ..interfaces.normalizer import INormalizer


class MacroChangeCalculator(BaseFeatureCalculator):
    """
    Calculator for macro indicator percentage changes with clipping.

    Calculates pct_change and applies clipping (no z-score normalization).

    Examples:
        # DXY daily change (288 bars = 1 day in 5-min data)
        calc = MacroChangeCalculator(
            indicator='dxy',
            periods=288,
            name='dxy_change_1d',
            clip_bounds=(-0.03, 0.03)
        )

        # Brent daily change
        calc = MacroChangeCalculator(
            indicator='brent',
            periods=288,
            name='brent_change_1d',
            clip_bounds=(-0.10, 0.10)
        )

        # USDMXN hourly return (12 bars = 1 hour)
        calc = MacroChangeCalculator(
            indicator='usdmxn',
            periods=12,
            name='usdmxn_ret_1h',
            clip_bounds=(-0.1, 0.1)
        )
    """

    def __init__(self,
                 indicator: str,
                 periods: int,
                 name: Optional[str] = None,
                 clip_bounds: Optional[tuple] = None):
        """
        Initialize macro change calculator.

        Args:
            indicator: Source column name (e.g., 'dxy', 'brent', 'usdmxn')
            periods: Number of periods for pct_change
            name: Output feature name (default: '{indicator}_change_{periods}p')
            clip_bounds: Clip bounds for percentage change
        """
        if name is None:
            name = f'{indicator}_change_{periods}p'

        super().__init__(
            name=name,
            dependencies=[indicator],
            normalizer=None,  # No z-score normalization for changes
            clip_bounds=clip_bounds
        )
        self.indicator = indicator
        self.periods = periods

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute percentage change for macro indicator.

        Args:
            data: DataFrame with indicator column
            **kwargs: Additional parameters (periods can override)

        Returns:
            pd.Series with percentage change values
        """
        periods = kwargs.get('periods', self.periods)
        values = data[self.indicator]

        # Calculate percentage change
        pct_change = values.pct_change(periods)

        return pct_change

    def get_params(self) -> dict:
        """Get calculator parameters."""
        params = super().get_params()
        params['indicator'] = self.indicator
        params['periods'] = self.periods
        return params
