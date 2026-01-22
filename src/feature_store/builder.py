"""
Feature Builder Compatibility Shim
===================================

This module provides backward compatibility for imports from
`feature_store.builder`. The canonical implementation is in
`feature_store.builders.canonical_feature_builder`.

DEPRECATION NOTICE:
    Direct imports from this module are deprecated.
    Use: from src.feature_store.builders import CanonicalFeatureBuilder

Migration Guide:
    # OLD (deprecated)
    from src.feature_store.builder import FeatureBuilder

    # NEW (recommended)
    from src.feature_store.builders import CanonicalFeatureBuilder as FeatureBuilder

Author: Trading Team
Version: 1.0.0
Date: 2026-01-18
"""

import warnings

# Import from canonical location
from .builders import (
    CanonicalFeatureBuilder,
    BuilderContext,
    NormStatsNotFoundError,
    ObservationDimensionError,
    FeatureCalculationError,
)

# Alias for backward compatibility
FeatureBuilder = CanonicalFeatureBuilder


def __getattr__(name):
    """Emit deprecation warning on attribute access."""
    if name == "FeatureBuilder":
        warnings.warn(
            "Importing FeatureBuilder from feature_store.builder is deprecated. "
            "Use: from src.feature_store.builders import CanonicalFeatureBuilder",
            DeprecationWarning,
            stacklevel=2,
        )
        return CanonicalFeatureBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeatureBuilder",
    "CanonicalFeatureBuilder",
    "BuilderContext",
    "NormStatsNotFoundError",
    "ObservationDimensionError",
    "FeatureCalculationError",
]
