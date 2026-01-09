"""
USD/COP Trading System - Source Code Package
=============================================

Main source code package containing core services and shared utilities.

Refactored with SOLID principles and Design Patterns:
- Single Responsibility Principle
- Open/Closed Principle
- Liskov Substitution Principle
- Interface Segregation Principle
- Dependency Inversion Principle

Design Patterns:
- Factory Pattern (FeatureCalculatorFactory, NormalizerFactory)
- Strategy Pattern (INormalizer implementations)
- Builder Pattern (ObservationBuilder)
- Template Method Pattern (BaseFeatureCalculator)
- Adapter Pattern (ConfigLoaderAdapter)

Author: Pedro @ Lean Tech Solutions
Version: 3.0.0
Date: 2025-12-17
"""

# Original implementation (backward compatibility)
from .core.services.feature_builder import FeatureBuilder, create_feature_builder

# Refactored implementation with SOLID & Design Patterns
from .core.services.feature_builder_refactored import (
    FeatureBuilderRefactored,
    create_feature_builder as create_feature_builder_refactored
)

# Configuration
from .shared.config_loader import ConfigLoader, get_config, load_feature_config
from .shared.config_loader_adapter import ConfigLoaderAdapter

# Exceptions
from .shared.exceptions import (
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

# Interfaces
from .core.interfaces import (
    IFeatureCalculator,
    INormalizer,
    IObservationBuilder,
    IConfigLoader
)

# Factories
from .core.factories import (
    FeatureCalculatorFactory,
    NormalizerFactory
)

# Calculators
from .core.calculators import (
    BaseFeatureCalculator,
    RSICalculator,
    ATRCalculator,
    ADXCalculator,
    ReturnsCalculator,
    MacroZScoreCalculator,
    MacroChangeCalculator
)

# Normalizers
from .core.normalizers import (
    ZScoreNormalizer,
    ClipNormalizer,
    NoOpNormalizer,
    CompositeNormalizer
)

# Builders
from .core.builders import ObservationBuilder

# Model Management
from .models import (
    ModelRegistry,
    ModelConfig,
    ModelLoader,
    InferenceEngine,
    InferenceResult,
    EnsembleResult,
)

# Trading (Paper Trading)
from .trading import (
    PaperTrader,
    PaperTrade,
    TradeDirection,
)

__version__ = "3.2.0"

__all__ = [
    # Original (backward compatibility)
    'FeatureBuilder',
    'create_feature_builder',

    # Refactored
    'FeatureBuilderRefactored',
    'create_feature_builder_refactored',

    # Configuration
    'ConfigLoader',
    'ConfigLoaderAdapter',
    'get_config',
    'load_feature_config',

    # Exceptions
    'USDCOPError',
    'ConfigurationError',
    'FeatureCalculationError',
    'ValidationError',
    'ObservationDimensionError',
    'FeatureMissingError',
    'FeatureBuilderError',
    'NormalizationError',

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
    'ClipNormalizer',
    'NoOpNormalizer',
    'CompositeNormalizer',

    # Builders
    'ObservationBuilder',

    # Model Management
    'ModelRegistry',
    'ModelConfig',
    'ModelLoader',
    'InferenceEngine',
    'InferenceResult',
    'EnsembleResult',

    # Trading (Paper Trading)
    'PaperTrader',
    'PaperTrade',
    'TradeDirection',
]
