"""
ClipNormalizer - Clipping Normalization Strategy
=================================================

Implements Strategy Pattern for value clipping.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import pandas as pd
import numpy as np
from typing import Union
from ..interfaces.normalizer import INormalizer


class ClipNormalizer(INormalizer):
    """
    Clipping normalization strategy.

    Clips values to specified bounds.

    Example:
        normalizer = ClipNormalizer(min_val=-4.0, max_val=4.0)
        clipped = normalizer.normalize(5.0)  # Returns 4.0
    """

    def __init__(self, min_val: float = -4.0, max_val: float = 4.0):
        """
        Initialize clip normalizer.

        Args:
            min_val: Minimum value for clipping
            max_val: Maximum value for clipping
        """
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Apply clipping.

        Args:
            value: Value or series to clip

        Returns:
            Clipped value or series
        """
        if isinstance(value, pd.Series):
            return value.clip(self.min_val, self.max_val)
        else:
            if pd.isna(value) or np.isnan(value):
                return 0.0
            return max(self.min_val, min(self.max_val, value))

    def denormalize(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Clipping is not reversible, so just return the value.

        Args:
            value: Clipped value or series

        Returns:
            Same value (clipping is not reversible)
        """
        return value

    def get_params(self) -> dict:
        """
        Get normalization parameters.

        Returns:
            Dictionary with min and max values
        """
        return {
            'type': 'clip',
            'min_val': self.min_val,
            'max_val': self.max_val
        }

    def __repr__(self) -> str:
        return f"ClipNormalizer(min_val={self.min_val}, max_val={self.max_val})"
