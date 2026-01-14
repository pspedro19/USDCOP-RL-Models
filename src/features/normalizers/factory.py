"""
Normalizer Factory with Strategy Pattern.

This module implements the Strategy Pattern for feature normalization,
allowing different normalization algorithms to be used interchangeably.

Design Patterns:
- Strategy Pattern: Normalizer is the strategy interface
- Factory Pattern: NormalizerFactory creates normalizer instances
- Template Method: Base validation in abstract methods

SOLID Principles:
- SRP: Each normalizer has single responsibility (one normalization method)
- OCP: Add new normalizers without modifying existing code
- LSP: All normalizers are substitutable
- ISP: Minimal Normalizer interface
- DIP: Factory depends on abstraction (Normalizer protocol)

Contrato: CTR-006
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple, Type, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# STRATEGY INTERFACE
# =============================================================================

class Normalizer(Protocol):
    """
    Protocol for feature normalizers.

    All normalization strategies must implement this interface.
    This enables the Strategy Pattern for swapping normalization methods.
    """

    def normalize(self, value: float) -> float:
        """
        Normalize a single value.

        Args:
            value: Raw feature value

        Returns:
            Normalized value
        """
        ...

    def denormalize(self, value: float) -> float:
        """
        Reverse normalization (for debugging/visualization).

        Args:
            value: Normalized value

        Returns:
            Original scale value
        """
        ...

    def normalize_batch(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize a batch of values (vectorized).

        Args:
            values: Array of raw feature values

        Returns:
            Array of normalized values
        """
        ...


# =============================================================================
# CONCRETE STRATEGIES
# =============================================================================

@dataclass
class ZScoreNormalizer:
    """
    Z-Score normalization: (x - mean) / std

    This is the primary normalization method used in the USDCOP system.
    Preserves relative distances and is robust to outliers with clipping.

    Args:
        mean: Population mean from training data
        std: Population std from training data
        clip: Tuple of (min, max) for clipping normalized values
    """

    mean: float = 0.0
    std: float = 1.0
    clip: Tuple[float, float] = field(default_factory=lambda: (-5.0, 5.0))

    def __post_init__(self):
        if self.std <= 0:
            logger.warning(f"ZScoreNormalizer: std={self.std} <= 0, defaulting to 1.0")
            object.__setattr__(self, 'std', 1.0)
        if isinstance(self.clip, list):
            object.__setattr__(self, 'clip', tuple(self.clip))

    def normalize(self, value: float) -> float:
        """Apply z-score normalization with clipping."""
        if np.isnan(value) or np.isinf(value):
            return 0.0

        z = (value - self.mean) / self.std
        return float(np.clip(z, self.clip[0], self.clip[1]))

    def denormalize(self, value: float) -> float:
        """Reverse z-score normalization."""
        return value * self.std + self.mean

    def normalize_batch(self, values: np.ndarray) -> np.ndarray:
        """Vectorized z-score normalization."""
        z = (values - self.mean) / self.std
        result = np.clip(z, self.clip[0], self.clip[1])
        # Handle NaN/Inf
        result = np.where(np.isfinite(result), result, 0.0)
        return result


@dataclass
class MinMaxNormalizer:
    """
    Min-Max normalization: (x - min) / (max - min)

    Scales values to a fixed range (default 0-1).
    Useful for bounded features like RSI.

    Args:
        min_val: Minimum value from training data
        max_val: Maximum value from training data
        output_range: Desired output range (default (0.0, 1.0))
    """

    min_val: float = 0.0
    max_val: float = 1.0
    output_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))

    def __post_init__(self):
        if self.max_val <= self.min_val:
            logger.warning(f"MinMaxNormalizer: max <= min, using defaults")
            object.__setattr__(self, 'min_val', 0.0)
            object.__setattr__(self, 'max_val', 1.0)
        if isinstance(self.output_range, list):
            object.__setattr__(self, 'output_range', tuple(self.output_range))

    def normalize(self, value: float) -> float:
        """Apply min-max normalization."""
        if np.isnan(value) or np.isinf(value):
            return self.output_range[0]

        range_in = self.max_val - self.min_val
        range_out = self.output_range[1] - self.output_range[0]

        scaled = (value - self.min_val) / range_in
        result = scaled * range_out + self.output_range[0]

        return float(np.clip(result, self.output_range[0], self.output_range[1]))

    def denormalize(self, value: float) -> float:
        """Reverse min-max normalization."""
        range_in = self.max_val - self.min_val
        range_out = self.output_range[1] - self.output_range[0]

        scaled = (value - self.output_range[0]) / range_out
        return scaled * range_in + self.min_val

    def normalize_batch(self, values: np.ndarray) -> np.ndarray:
        """Vectorized min-max normalization."""
        range_in = self.max_val - self.min_val
        range_out = self.output_range[1] - self.output_range[0]

        scaled = (values - self.min_val) / range_in
        result = scaled * range_out + self.output_range[0]
        result = np.clip(result, self.output_range[0], self.output_range[1])

        return np.where(np.isfinite(result), result, self.output_range[0])


@dataclass
class ClipNormalizer:
    """
    Clip-only normalization: clip values to a range without scaling.

    Useful for features that are already in a reasonable range
    but need bounds enforced (e.g., position, time_normalized).

    Args:
        clip: Tuple of (min, max) for clipping
    """

    clip: Tuple[float, float] = field(default_factory=lambda: (-1.0, 1.0))

    def __post_init__(self):
        if isinstance(self.clip, list):
            object.__setattr__(self, 'clip', tuple(self.clip))

    def normalize(self, value: float) -> float:
        """Apply clipping."""
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(np.clip(value, self.clip[0], self.clip[1]))

    def denormalize(self, value: float) -> float:
        """No-op for clip normalizer (returns same value)."""
        return value

    def normalize_batch(self, values: np.ndarray) -> np.ndarray:
        """Vectorized clipping."""
        result = np.clip(values, self.clip[0], self.clip[1])
        return np.where(np.isfinite(result), result, 0.0)


@dataclass
class NoOpNormalizer:
    """
    No-operation normalizer: passes values through unchanged.

    Useful for features that don't require normalization
    or for debugging/testing.
    """

    def normalize(self, value: float) -> float:
        """Pass through unchanged."""
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)

    def denormalize(self, value: float) -> float:
        """Pass through unchanged."""
        return value

    def normalize_batch(self, values: np.ndarray) -> np.ndarray:
        """Pass through unchanged."""
        return np.where(np.isfinite(values), values, 0.0)


# =============================================================================
# FACTORY
# =============================================================================

class NormalizerFactory:
    """
    Factory for creating normalizer instances.

    Implements Factory Pattern with registry of normalizer types.
    Supports configuration from dictionaries (YAML/JSON).

    Usage:
        # From method name and params
        normalizer = NormalizerFactory.create("zscore", mean=0.0, std=1.0)

        # From config dict
        config = {"method": "zscore", "mean": 0.0, "std": 1.0, "clip": [-5, 5]}
        normalizer = NormalizerFactory.from_config(config)
    """

    _registry: Dict[str, Type] = {
        "zscore": ZScoreNormalizer,
        "z-score": ZScoreNormalizer,
        "minmax": MinMaxNormalizer,
        "min-max": MinMaxNormalizer,
        "clip": ClipNormalizer,
        "none": NoOpNormalizer,
        "noop": NoOpNormalizer,
    }

    @classmethod
    def create(cls, method: str, **kwargs) -> Normalizer:
        """
        Create normalizer from method name and parameters.

        Args:
            method: Normalization method name
            **kwargs: Method-specific parameters

        Returns:
            Configured normalizer instance

        Raises:
            ValueError: If method is unknown
        """
        method_lower = method.lower()

        if method_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown normalization method: {method}. "
                f"Available: {available}"
            )

        normalizer_class = cls._registry[method_lower]

        # Filter kwargs to only include valid params for the class
        valid_kwargs = cls._filter_kwargs(normalizer_class, kwargs)

        return normalizer_class(**valid_kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Normalizer:
        """
        Create normalizer from configuration dictionary.

        Expected config format:
        {
            "method": "zscore",
            "mean": 0.0,
            "std": 1.0,
            "clip": [-5.0, 5.0]
        }

        Args:
            config: Configuration dictionary

        Returns:
            Configured normalizer instance
        """
        method = config.get("method", "zscore")

        # Extract params excluding 'method'
        params = {k: v for k, v in config.items() if k != "method"}

        return cls.create(method, **params)

    @classmethod
    def register(cls, name: str, normalizer_class: Type) -> None:
        """
        Register a new normalizer type.

        Args:
            name: Method name for lookup
            normalizer_class: Class implementing Normalizer protocol
        """
        cls._registry[name.lower()] = normalizer_class
        logger.info(f"Registered normalizer: {name}")

    @classmethod
    def list_methods(cls) -> list:
        """List available normalization methods."""
        return list(set(cls._registry.keys()))

    @staticmethod
    def _filter_kwargs(normalizer_class: Type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter kwargs to only include valid parameters for the class."""
        import inspect

        # Get valid parameters from class signature
        try:
            sig = inspect.signature(normalizer_class)
            valid_params = set(sig.parameters.keys())
        except (ValueError, TypeError):
            # Dataclass fallback
            if hasattr(normalizer_class, '__dataclass_fields__'):
                valid_params = set(normalizer_class.__dataclass_fields__.keys())
            else:
                return kwargs

        return {k: v for k, v in kwargs.items() if k in valid_params}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_normalizer(method: str, **kwargs) -> Normalizer:
    """
    Convenience function to create normalizer.

    Args:
        method: Normalization method name
        **kwargs: Method-specific parameters

    Returns:
        Configured normalizer instance
    """
    return NormalizerFactory.create(method, **kwargs)


def normalize_feature(
    value: float,
    config: Dict[str, Any]
) -> float:
    """
    Normalize a single feature value using config.

    Args:
        value: Raw feature value
        config: Normalization config with 'method' and params

    Returns:
        Normalized value
    """
    normalizer = NormalizerFactory.from_config(config)
    return normalizer.normalize(value)
