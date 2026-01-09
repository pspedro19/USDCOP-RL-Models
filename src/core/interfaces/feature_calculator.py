"""
IFeatureCalculator - Interface for feature calculators
======================================================

Abstract interface following Interface Segregation Principle (SOLID).

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


class IFeatureCalculator(ABC):
    """
    Abstract interface for feature calculators.

    Defines contract that all feature calculators must implement.
    Enables dependency injection and testability.

    Example:
        class RSICalculator(IFeatureCalculator):
            def calculate(self, data: pd.DataFrame) -> pd.Series:
                # Implementation here
                pass
    """

    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate the feature from input data.

        Args:
            data: Input DataFrame with OHLCV data
            **kwargs: Additional parameters specific to the calculator

        Returns:
            pd.Series with calculated feature values

        Raises:
            FeatureCalculationError: If calculation fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the feature name produced by this calculator.

        Returns:
            Feature name (e.g., 'rsi_9', 'log_ret_5m')
        """
        pass

    @abstractmethod
    def get_dependencies(self) -> list:
        """
        Get list of column names required for calculation.

        Returns:
            List of required column names (e.g., ['close'], ['high', 'low', 'close'])
        """
        pass

    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate that input data contains required columns.

        Args:
            data: Input DataFrame to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get calculator parameters (optional, can be overridden).

        Returns:
            Dictionary of parameter names and values
        """
        return {}
