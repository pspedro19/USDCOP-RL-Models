"""
BaseFeatureCalculator - Template Method Pattern
================================================

Abstract base class implementing Template Method Pattern for feature calculation.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

from abc import abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from ..interfaces.feature_calculator import IFeatureCalculator
from ..interfaces.normalizer import INormalizer
from ...shared.exceptions import ValidationError, FeatureCalculationError


class BaseFeatureCalculator(IFeatureCalculator):
    """
    Base class for all feature calculators using Template Method Pattern.

    Defines the algorithm structure:
    1. validate_input() - Check input data has required columns
    2. compute() - Perform actual calculation (abstract, must override)
    3. normalize() - Apply normalization strategy (optional)
    4. clip() - Apply clipping (optional)

    The calculate() method orchestrates these steps.

    Example:
        class RSICalculator(BaseFeatureCalculator):
            def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                # RSI calculation logic here
                return rsi_series
    """

    def __init__(self,
                 name: str,
                 dependencies: list,
                 normalizer: Optional[INormalizer] = None,
                 clip_bounds: Optional[tuple] = None):
        """
        Initialize base calculator.

        Args:
            name: Feature name produced by this calculator
            dependencies: List of required column names
            normalizer: Optional normalizer strategy
            clip_bounds: Optional tuple (min, max) for clipping
        """
        self._name = name
        self._dependencies = dependencies
        self._normalizer = normalizer
        self._clip_bounds = clip_bounds

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Template Method: Orchestrates the calculation process.

        Steps:
        1. Validate input data
        2. Compute raw feature values
        3. Apply normalization (if configured)
        4. Apply clipping (if configured)

        Args:
            data: Input DataFrame
            **kwargs: Additional parameters

        Returns:
            pd.Series with calculated and normalized feature values

        Raises:
            ValidationError: If input validation fails
            FeatureCalculationError: If calculation fails
        """
        try:
            # Step 1: Validate
            self.validate_input(data)

            # Step 2: Compute raw values
            result = self.compute(data, **kwargs)

            # Step 3: Normalize (if normalizer configured)
            if self._normalizer is not None:
                result = self._normalizer.normalize(result)

            # Step 4: Clip (if bounds configured)
            if self._clip_bounds is not None:
                result = result.clip(self._clip_bounds[0], self._clip_bounds[1])

            return result

        except ValidationError:
            raise
        except Exception as e:
            raise FeatureCalculationError(
                f"Failed to calculate {self._name}: {str(e)}",
                feature_name=self._name,
                error_type=type(e).__name__
            )

    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute the raw feature values (must be overridden by subclasses).

        Args:
            data: Input DataFrame with required columns
            **kwargs: Additional calculation parameters

        Returns:
            pd.Series with raw feature values
        """
        pass

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
        missing = [col for col in self._dependencies if col not in data.columns]
        if missing:
            raise ValidationError(
                f"Missing required columns for {self._name}: {missing}"
            )
        return True

    def get_name(self) -> str:
        """Get feature name."""
        return self._name

    def get_dependencies(self) -> list:
        """Get required column names."""
        return self._dependencies.copy()

    def get_params(self) -> Dict[str, Any]:
        """
        Get calculator parameters.

        Returns:
            Dictionary with calculator configuration
        """
        params = {
            'name': self._name,
            'dependencies': self._dependencies,
        }

        if self._normalizer:
            params['normalizer'] = self._normalizer.get_params()

        if self._clip_bounds:
            params['clip_bounds'] = self._clip_bounds

        return params

    def set_normalizer(self, normalizer: INormalizer) -> None:
        """
        Set or update the normalizer strategy.

        Args:
            normalizer: Normalizer to use
        """
        self._normalizer = normalizer

    def set_clip_bounds(self, min_val: float, max_val: float) -> None:
        """
        Set or update clipping bounds.

        Args:
            min_val: Minimum value
            max_val: Maximum value
        """
        self._clip_bounds = (min_val, max_val)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}')"


class SimpleCalculator(BaseFeatureCalculator):
    """
    Simple calculator that accepts a computation function.

    Useful for quick lambda-based calculators without creating full classes.

    Example:
        calc = SimpleCalculator(
            name='log_ret_5m',
            dependencies=['close'],
            compute_fn=lambda df: np.log(df['close'] / df['close'].shift(1))
        )
    """

    def __init__(self,
                 name: str,
                 dependencies: list,
                 compute_fn: callable,
                 normalizer: Optional[INormalizer] = None,
                 clip_bounds: Optional[tuple] = None):
        """
        Initialize simple calculator.

        Args:
            name: Feature name
            dependencies: Required columns
            compute_fn: Function that takes DataFrame and returns Series
            normalizer: Optional normalizer
            clip_bounds: Optional clip bounds
        """
        super().__init__(name, dependencies, normalizer, clip_bounds)
        self._compute_fn = compute_fn

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Execute the compute function."""
        return self._compute_fn(data)
