"""
Shared module - Configuration, exceptions, and utilities.
"""

from .config_loader import ConfigLoader, get_config, load_feature_config
from .config_loader_adapter import ConfigLoaderAdapter
from .exceptions import (
    USDCOPError,
    ConfigurationError,
    FeatureCalculationError,
    ValidationError,
    ObservationDimensionError,
    FeatureMissingError,
    # Legacy aliases
    FeatureBuilderError,
    NormalizationError
)

__all__ = [
    'ConfigLoader',
    'ConfigLoaderAdapter',
    'get_config',
    'load_feature_config',
    'USDCOPError',
    'ConfigurationError',
    'FeatureCalculationError',
    'ValidationError',
    'ObservationDimensionError',
    'FeatureMissingError',
    # Legacy
    'FeatureBuilderError',
    'NormalizationError',
]
