"""
MacroZScoreCalculator - Macro Z-Score Feature Calculator
=========================================================

Calculates z-scores for macro indicators with fixed statistics.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_calculator import BaseFeatureCalculator
from ..interfaces.normalizer import INormalizer


class MacroZScoreCalculator(BaseFeatureCalculator):
    """
    Calculator for macro indicator z-scores with fixed statistics.

    Uses predefined mean and std for macro indicators:
    - DXY: mean=103, std=5
    - VIX: mean=20, std=10
    - EMBI: mean=300, std=100

    Formula: (value - mean) / std

    Example:
        calc = MacroZScoreCalculator(
            indicator='dxy',
            mean=103.0,
            std=5.0,
            name='dxy_z'
        )
        dxy_z = calc.calculate(df)  # df must have 'dxy' column
    """

    def __init__(self,
                 indicator: str,
                 mean: float,
                 std: float,
                 name: Optional[str] = None,
                 clip_bounds: Optional[tuple] = (-4.0, 4.0)):
        """
        Initialize macro z-score calculator.

        Args:
            indicator: Source column name (e.g., 'dxy', 'vix', 'embi')
            mean: Fixed mean for normalization
            std: Fixed standard deviation for normalization
            name: Output feature name (default: '{indicator}_z')
            clip_bounds: Clip bounds after z-score (default: [-4, 4])
        """
        if name is None:
            name = f'{indicator}_z'

        super().__init__(
            name=name,
            dependencies=[indicator],
            normalizer=None,  # We handle normalization in compute()
            clip_bounds=clip_bounds
        )
        self.indicator = indicator
        self.mean = mean
        self.std = std if std > 0 else 1.0

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute z-score for macro indicator.

        Args:
            data: DataFrame with indicator column
            **kwargs: Additional parameters

        Returns:
            pd.Series with z-score values
        """
        values = data[self.indicator]

        # Calculate z-score with fixed statistics
        z_score = (values - self.mean) / self.std

        return z_score

    def get_params(self) -> dict:
        """Get calculator parameters."""
        params = super().get_params()
        params['indicator'] = self.indicator
        params['mean'] = self.mean
        params['std'] = self.std
        return params
