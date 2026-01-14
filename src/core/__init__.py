"""
Core module - Business logic and services.

Refactored with SOLID principles and Design Patterns.

Version: 1.0.0
Date: 2025-01-07
"""

# Original implementation (backward compatibility)
from .services.feature_builder import FeatureBuilder, create_feature_builder

# Refactored implementation with SOLID & Design Patterns
from .services.feature_builder_refactored import (
    FeatureBuilderRefactored,
    create_feature_builder as create_feature_builder_refactored
)

# Interfaces
from .interfaces import (
    IFeatureCalculator,
    INormalizer,
    IObservationBuilder,
    IConfigLoader
)

# Factories
from .factories import (
    FeatureCalculatorFactory,
    NormalizerFactory
)

# Calculators
from .calculators import (
    BaseFeatureCalculator,
    RSICalculator,
    ATRCalculator,
    ADXCalculator,
    ReturnsCalculator,
    MacroZScoreCalculator,
    MacroChangeCalculator
)

# Normalizers
from .normalizers import (
    ZScoreNormalizer,
    create_zscore_normalizer,
    ClipNormalizer,
    NoOpNormalizer,
    CompositeNormalizer
)

# Builders
from .builders import (
    ObservationBuilder,
    create_observation_builder
)

# State Management
from .state import (
    ModelState,
    StateTracker,
    create_state_tracker
)

__all__ = [
    # Original (backward compatibility)
    'FeatureBuilder',
    'create_feature_builder',

    # Refactored
    'FeatureBuilderRefactored',
    'create_feature_builder_refactored',

    # Interfaces
    'IFeatureCalculator',
    'INormalizer',
    'IObservationBuilder',
    'IConfigLoader',

    # Factories
    'FeatureCalculatorFactory',
    'NormalizerFactory',

    # Calculators
    'BaseFeatureCalculator',
    'RSICalculator',
    'ATRCalculator',
    'ADXCalculator',
    'ReturnsCalculator',
    'MacroZScoreCalculator',
    'MacroChangeCalculator',

    # Normalizers
    'ZScoreNormalizer',
    'create_zscore_normalizer',
    'ClipNormalizer',
    'NoOpNormalizer',
    'CompositeNormalizer',

    # Builders
    'ObservationBuilder',
    'create_observation_builder',

    # State Management
    'ModelState',
    'StateTracker',
    'create_state_tracker',
]
