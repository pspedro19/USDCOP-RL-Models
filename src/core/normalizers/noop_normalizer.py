"""
NoOpNormalizer - Pass-Through Normalization Strategy
=====================================================

Implements Strategy Pattern for no normalization (pass-through).

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

import pandas as pd
import numpy as np
from typing import Union
from ..interfaces.normalizer import INormalizer


class NoOpNormalizer(INormalizer):
    """
    No-operation (pass-through) normalization strategy.

    Returns values unchanged. Useful for features that don't need normalization
    (e.g., macro_changes that are already clipped).

    Example:
        normalizer = NoOpNormalizer()
        value = normalizer.normalize(0.02)  # Returns 0.02 (unchanged)
    """

    def __init__(self):
        """Initialize no-op normalizer."""
        pass

    def normalize(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Return value unchanged.

        Args:
            value: Input value or series

        Returns:
            Same value (unchanged)
        """
        if isinstance(value, (float, int)):
            if pd.isna(value) or np.isnan(value):
                return 0.0
        return value

    def denormalize(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Return value unchanged.

        Args:
            value: Input value or series

        Returns:
            Same value (unchanged)
        """
        if isinstance(value, (float, int)):
            if pd.isna(value) or np.isnan(value):
                return 0.0
        return value

    def get_params(self) -> dict:
        """
        Get normalization parameters.

        Returns:
            Dictionary with type only
        """
        return {'type': 'noop'}

    def __repr__(self) -> str:
        return "NoOpNormalizer()"
