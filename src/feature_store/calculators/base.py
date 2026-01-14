"""
Base Feature Calculator
========================
Abstract base class for all feature calculators.

SOLID Principles:
- Single Responsibility: Each calculator handles one feature type
- Open/Closed: Extend via new calculators, not modification
- Liskov Substitution: All calculators are interchangeable
- Interface Segregation: Minimal calculator interface
- Dependency Inversion: Depends on FeatureSpec abstraction

Design Pattern: Strategy Pattern
- Each calculator is a strategy for calculating a specific feature
- Calculators are registered in the registry for lookup

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import numpy as np
import pandas as pd

from ..contracts import (
    FeatureSpec,
    FeatureValue,
    SmoothingMethod,
    NormalizationMethod,
    NormalizationParams,
)


class FeatureCalculator(ABC):
    """
    Abstract base class for feature calculators.

    Each calculator implements a specific feature calculation
    as the SINGLE SOURCE OF TRUTH used by all pipelines.
    """

    def __init__(self, spec: FeatureSpec):
        """
        Initialize calculator with feature specification.

        Args:
            spec: Complete feature specification
        """
        self.spec = spec

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate raw feature values from input data.

        Args:
            data: DataFrame with required columns (OHLCV, etc.)

        Returns:
            Series with calculated feature values
        """
        pass

    def normalize(
        self,
        values: pd.Series,
        params: Optional[NormalizationParams] = None
    ) -> pd.Series:
        """
        Normalize feature values using spec or provided params.

        Args:
            values: Raw feature values
            params: Override normalization params (for inference)

        Returns:
            Normalized feature values
        """
        norm_params = params or self.spec.normalization

        if norm_params.method == NormalizationMethod.NONE:
            return values

        if norm_params.method == NormalizationMethod.ZSCORE:
            normalized = (values - norm_params.mean) / norm_params.std

        elif norm_params.method == NormalizationMethod.MINMAX:
            normalized = (values - norm_params.min_val) / (
                norm_params.max_val - norm_params.min_val
            )

        elif norm_params.method == NormalizationMethod.ROLLING_ZSCORE:
            rolling_mean = values.rolling(norm_params.rolling_window).mean()
            rolling_std = values.rolling(norm_params.rolling_window).std()
            normalized = (values - rolling_mean) / rolling_std

        elif norm_params.method == NormalizationMethod.ROBUST:
            median = values.median()
            q75, q25 = values.quantile(0.75), values.quantile(0.25)
            iqr = q75 - q25
            normalized = (values - median) / iqr

        else:
            raise ValueError(f"Unknown normalization method: {norm_params.method}")

        # Apply clipping
        if norm_params.clip_range:
            low, high = norm_params.clip_range
            normalized = normalized.clip(lower=low, upper=high)

        return normalized

    def calculate_and_normalize(
        self,
        data: pd.DataFrame,
        norm_params: Optional[NormalizationParams] = None
    ) -> pd.Series:
        """
        Calculate and normalize feature in one step.

        Args:
            data: Input DataFrame
            norm_params: Optional override for normalization

        Returns:
            Normalized feature values
        """
        raw_values = self.calculate(data)
        return self.normalize(raw_values, norm_params)

    @property
    def name(self) -> str:
        """Feature name"""
        return self.spec.name

    @property
    def requires(self) -> List[str]:
        """Required input columns"""
        return self.spec.requires

    def validate_input(self, data: pd.DataFrame) -> None:
        """Validate that required columns exist"""
        missing = set(self.requires) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _apply_smoothing(
        self,
        series: pd.Series,
        method: SmoothingMethod,
        window: int,
        alpha: Optional[float] = None
    ) -> pd.Series:
        """
        Apply smoothing to a series.

        This is the AUTHORITATIVE implementation for smoothing
        that MUST be used by all pipelines.

        Args:
            series: Input series
            method: Smoothing method
            window: Window size
            alpha: Optional alpha for EMA

        Returns:
            Smoothed series
        """
        if method == SmoothingMethod.NONE:
            return series

        if method == SmoothingMethod.SMA:
            return series.rolling(window=window, min_periods=1).mean()

        if method == SmoothingMethod.EMA:
            if alpha is None:
                alpha = 2 / (window + 1)
            return series.ewm(alpha=alpha, min_periods=1, adjust=False).mean()

        if method == SmoothingMethod.WILDER:
            # Wilder's smoothing: alpha = 1/n
            wilder_alpha = 1.0 / window
            return series.ewm(alpha=wilder_alpha, min_periods=1, adjust=False).mean()

        raise ValueError(f"Unknown smoothing method: {method}")


# =============================================================================
# CALCULATOR REGISTRY
# =============================================================================

class CalculatorRegistry:
    """
    Registry for feature calculators.

    Implements Factory Pattern for calculator creation.
    """
    _calculators: Dict[str, Type[FeatureCalculator]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a calculator"""
        def decorator(calculator_cls: Type[FeatureCalculator]):
            cls._calculators[name] = calculator_cls
            return calculator_cls
        return decorator

    @classmethod
    def create(cls, name: str, spec: FeatureSpec) -> FeatureCalculator:
        """Create calculator instance for a feature"""
        if name not in cls._calculators:
            raise KeyError(f"Unknown calculator: {name}")
        return cls._calculators[name](spec)

    @classmethod
    def list_calculators(cls) -> List[str]:
        """List registered calculators"""
        return list(cls._calculators.keys())

    @classmethod
    def get_calculator_class(cls, name: str) -> Type[FeatureCalculator]:
        """Get calculator class by name"""
        if name not in cls._calculators:
            raise KeyError(f"Unknown calculator: {name}")
        return cls._calculators[name]
