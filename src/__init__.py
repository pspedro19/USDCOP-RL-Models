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
- Contract Pattern (FeatureContract)

CANONICAL IMPORTS:
    from src.features import FeatureBuilder, create_feature_builder
    from src import FeatureBuilder  # Also works (re-exported here)

Author: Pedro @ Lean Tech Solutions
Version: 4.0.0
Date: 2026-01-12
"""

# =============================================================================
# CANONICAL FEATURE BUILDER (Contract-based, SSOT)
# =============================================================================
from .features.builder import (
    FeatureBuilder,
    create_feature_builder,
)
from .features.contract import (
    FeatureContract,
    FEATURE_CONTRACT,
    get_contract,
    FEATURE_ORDER,
    OBSERVATION_DIM,
)
from .core.contracts.norm_stats_contract import load_norm_stats

# =============================================================================
# LEGACY IMPORTS (for backward compatibility - marked for deprecation)
# =============================================================================
from .core.services.feature_builder import (
    FeatureBuilder as FeatureBuilderLegacy,
    create_feature_builder as create_feature_builder_legacy,
)

# SOLID refactored version (optional, Phase 6)
# Note: FeatureBuilderRefactored is conditionally imported to avoid ImportError
# if the module is not yet implemented
try:
    from .core.services.feature_builder_refactored import (
        FeatureBuilderRefactored,
        create_feature_builder as create_feature_builder_refactored,
    )
except ImportError:
    # Module not implemented yet - provide stubs for backward compatibility
    FeatureBuilderRefactored = None
    create_feature_builder_refactored = None

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

# Calculators (SSOT: use feature_store.calculators)
try:
    from .feature_store.calculators import (
        BaseCalculator as BaseFeatureCalculator,
        RSICalculator,
        ATRPercentCalculator as ATRCalculator,
        ADXCalculator,
        LogReturnCalculator as ReturnsCalculator,
        MacroZScoreCalculator,
        MacroChangeCalculator,
    )
except ImportError:
    # Fallback stubs if calculators module not available
    BaseFeatureCalculator = None
    RSICalculator = None
    ATRCalculator = None
    ADXCalculator = None
    ReturnsCalculator = None
    MacroZScoreCalculator = None
    MacroChangeCalculator = None

# Normalizers
from .core.normalizers import (
    ZScoreNormalizer,
    ClipNormalizer,
    NoOpNormalizer,
    CompositeNormalizer
)

# Builders
from .core.builders import ObservationBuilder

# Model Management - Lazy imports to avoid ONNX loading on import
# These are imported on-demand when accessed
def __getattr__(name):
    """Lazy import for model-related classes to avoid ONNX loading issues."""
    _model_exports = {
        'ModelRegistry', 'ModelConfig', 'ModelLoader',
        'InferenceEngine', 'InferenceResult', 'EnsembleResult',
    }
    if name in _model_exports:
        from . import models
        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Trading (Paper Trading)
from .trading import (
    PaperTrader,
    PaperTrade,
    TradeDirection,
)

__version__ = "4.0.0"

__all__ = [
    # ==========================================================================
    # CANONICAL FEATURE BUILDER (Contract-based, SSOT)
    # ==========================================================================
    'FeatureBuilder',           # Contract-based (CANONICAL)
    'create_feature_builder',   # Factory for FeatureBuilder
    'load_norm_stats',          # Utility to load norm stats

    # Feature Contract
    'FeatureContract',
    'FEATURE_CONTRACT',
    'get_contract',
    'FEATURE_ORDER',
    'OBSERVATION_DIM',

    # ==========================================================================
    # LEGACY (backward compatibility - use canonical imports above)
    # ==========================================================================
    'FeatureBuilderLegacy',     # V2.1 legacy (use FeatureBuilder instead)
    'create_feature_builder_legacy',
    'FeatureBuilderRefactored', # V3.0 SOLID refactoring
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
