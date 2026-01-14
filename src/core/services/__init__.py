"""
Services module - Feature building and calculation.

Two FeatureBuilder implementations available:
- FeatureBuilder: Original production version
- FeatureBuilderRefactored: SOLID-compliant version with design patterns

Usage:
    # Default (production)
    from src.core.services import FeatureBuilder

    # SOLID version (optional)
    from src.core.services import FeatureBuilderRefactored
"""

from .feature_builder import FeatureBuilder, create_feature_builder

# SOLID refactored version (optional, Phase 6)
from .feature_builder_refactored import FeatureBuilderRefactored

__all__ = [
    # Original (production)
    'FeatureBuilder',
    'create_feature_builder',
    # SOLID refactored (optional)
    'FeatureBuilderRefactored',
]
