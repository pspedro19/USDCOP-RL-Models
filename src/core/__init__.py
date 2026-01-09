"""
Core module - Business logic and services.

Refactored with SOLID principles and Design Patterns.

Version: 19.0.0
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
    ZScoreNormalizerV19,
    create_zscore_normalizer_v19,
    ClipNormalizer,
    NoOpNormalizer,
    CompositeNormalizer
)

# Builders
from .builders import (
    ObservationBuilder,
    ObservationBuilderV19,
    create_observation_builder_v19
)

# State Management (V19)
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
    'ZScoreNormalizerV19',
    'create_zscore_normalizer_v19',
    'ClipNormalizer',
    'NoOpNormalizer',
    'CompositeNormalizer',

    # Builders
    'ObservationBuilder',
    'ObservationBuilderV19',
    'create_observation_builder_v19',

    # State Management (V19)
    'ModelState',
    'StateTracker',
    'create_state_tracker',
]
