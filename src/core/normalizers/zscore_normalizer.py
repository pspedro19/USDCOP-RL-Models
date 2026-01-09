"""
ZScoreNormalizer - Z-Score Normalization Strategy
==================================================

Implements Strategy Pattern for z-score normalization.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import pandas as pd
import numpy as np
from typing import Union
from ..interfaces.normalizer import INormalizer


class ZScoreNormalizer(INormalizer):
    """
    Z-Score normalization strategy.

    Formula: (x - mean) / std

    Example:
        normalizer = ZScoreNormalizer(mean=100.0, std=10.0)
        normalized = normalizer.normalize(105.0)  # Returns 0.5
        original = normalizer.denormalize(0.5)     # Returns 105.0
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Initialize z-score normalizer.

        Args:
            mean: Mean value for normalization
            std: Standard deviation for normalization
        """
        self.mean = mean
        self.std = std if std > 0 else 1.0  # Prevent division by zero

    def normalize(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Apply z-score normalization.

        Args:
            value: Raw value or series

        Returns:
            Normalized value or series
        """
        if isinstance(value, pd.Series):
            return (value - self.mean) / self.std
        else:
            if pd.isna(value) or np.isnan(value):
                return 0.0
            return (value - self.mean) / self.std

    def denormalize(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Reverse z-score normalization.

        Args:
            value: Normalized value or series

        Returns:
            Original-scale value or series
        """
        if isinstance(value, pd.Series):
            return (value * self.std) + self.mean
        else:
            if pd.isna(value) or np.isnan(value):
                return 0.0
            return (value * self.std) + self.mean

    def get_params(self) -> dict:
        """
        Get normalization parameters.

        Returns:
            Dictionary with mean and std
        """
        return {
            'type': 'zscore',
            'mean': self.mean,
            'std': self.std
        }

    def __repr__(self) -> str:
        return f"ZScoreNormalizer(mean={self.mean}, std={self.std})"
