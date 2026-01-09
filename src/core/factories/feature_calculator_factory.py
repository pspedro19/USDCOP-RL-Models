"""
FeatureCalculatorFactory - Factory Pattern for Feature Calculators
===================================================================

Creates appropriate calculator instances based on feature type.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

from typing import Dict, Type, Optional
from ..interfaces.feature_calculator import IFeatureCalculator
from ...shared.exceptions import ConfigurationError


class FeatureCalculatorFactory:
    """
    Factory for creating feature calculator instances.

    Implements Factory Pattern for clean object creation.

    Usage:
        factory = FeatureCalculatorFactory()
        rsi_calc = factory.create('rsi', period=9)
        atr_calc = factory.create('atr', period=10)
        ret_calc = factory.create('returns', periods=1)
    """

    _calculators: Dict[str, Type[IFeatureCalculator]] = {}

    @classmethod
    def register(cls, feature_type: str, calculator_class: Type[IFeatureCalculator]) -> None:
        """
        Register a calculator class for a feature type.

        Args:
            feature_type: Feature type identifier (e.g., 'rsi', 'atr', 'returns')
            calculator_class: Calculator class to register
        """
        cls._calculators[feature_type] = calculator_class

    @classmethod
    def create(cls, feature_type: str, **kwargs) -> IFeatureCalculator:
        """
        Create a calculator instance for the specified feature type.

        Args:
            feature_type: Type of feature calculator to create
            **kwargs: Parameters to pass to calculator constructor

        Returns:
            IFeatureCalculator instance

        Raises:
            ConfigurationError: If feature type is not registered

        Example:
            >>> factory = FeatureCalculatorFactory()
            >>> rsi_calc = factory.create('rsi', period=9)
            >>> atr_calc = factory.create('atr', period=10)
        """
        calculator_class = cls._calculators.get(feature_type)

        if calculator_class is None:
            raise ConfigurationError(
                f"Unknown feature type: '{feature_type}'",
                missing_key=feature_type
            )

        return calculator_class(**kwargs)

    @classmethod
    def get_registered_types(cls) -> list:
        """
        Get list of all registered feature types.

        Returns:
            List of registered feature type names
        """
        return list(cls._calculators.keys())

    @classmethod
    def is_registered(cls, feature_type: str) -> bool:
        """
        Check if a feature type is registered.

        Args:
            feature_type: Feature type to check

        Returns:
            True if registered, False otherwise
        """
        return feature_type in cls._calculators
