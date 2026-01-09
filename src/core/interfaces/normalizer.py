"""
INormalizer - Interface for feature normalizers
================================================

Abstract interface for Strategy Pattern implementation.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Union


class INormalizer(ABC):
    """
    Abstract interface for normalization strategies.

    Implements Strategy Pattern for different normalization methods:
    - Z-Score normalization
    - Clipping normalization
    - No-op (pass-through) normalization

    Example:
        normalizer = ZScoreNormalizer(mean=0.0, std=1.0)
        normalized = normalizer.normalize(values)
    """

    @abstractmethod
    def normalize(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Normalize a single value or series.

        Args:
            value: Raw value or series to normalize

        Returns:
            Normalized value or series
        """
        pass

    @abstractmethod
    def denormalize(self, value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Reverse normalization (optional, for some strategies).

        Args:
            value: Normalized value or series

        Returns:
            Original-scale value or series
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Get normalization parameters.

        Returns:
            Dictionary with normalization parameters
        """
        pass
