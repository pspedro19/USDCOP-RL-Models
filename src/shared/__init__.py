"""
Shared module - Configuration, exceptions, utilities, and schemas.

Submodules:
    - config_loader: Configuration loading utilities
    - exceptions: Custom exception classes
    - schemas: Shared Pydantic schemas for API contracts
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

# Schemas submodule (import on demand to avoid circular imports)
# Usage: from shared.schemas import TradeSchema, SignalType

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
    # Schemas submodule
    'schemas',
]
