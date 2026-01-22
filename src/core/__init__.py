"""
Core module - Business logic and services.

Refactored with SOLID principles and Design Patterns.

Design Patterns included:
- Factory Pattern: FeatureCalculatorFactory, NormalizerFactory
- Builder Pattern: ObservationBuilder
- Event Bus Pattern: EventBus for pub/sub communication
- Decorator Pattern: Resilience decorators (retry, circuit breaker, timing)

Version: 1.1.0
Date: 2025-01-07
"""

# Original implementation (backward compatibility)
from .services.feature_builder import FeatureBuilder, create_feature_builder

# Refactored implementation with SOLID & Design Patterns
# Note: FeatureBuilderRefactored is conditionally imported to avoid ImportError
# if the module is not yet implemented
try:
    from .services.feature_builder_refactored import (
        FeatureBuilderRefactored,
        create_feature_builder as create_feature_builder_refactored
    )
except ImportError:
    # Module not implemented yet - provide stubs for backward compatibility
    FeatureBuilderRefactored = None
    create_feature_builder_refactored = None

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

# Calculators - REMOVED (DRY Clean Code)
# SSOT calculators are now in src/feature_store/calculators/
# Use: from src.feature_store.calculators import RSICalculator, ATRPercentCalculator, ADXCalculator

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

# Secrets Management
from .secrets import (
    SecretManager,
    get_secret
)

# Event Bus Pattern
from .events import (
    EventBus,
    TradeEvent,
    RiskEvent,
    SystemEvent,
    get_event_bus,
    subscribe_to,
    EventPriority,
)

# Decorator Pattern - Resilience
from .decorators import (
    with_retry,
    with_timing,
    with_circuit_breaker,
    with_timeout,
    with_fallback,
    CircuitBreaker,
    CircuitBreakerOpenError,
)

# Logging
from .logging import (
    LoggerFactory,
    get_logger
)

# Constants (P2 Clean Code - extracted magic numbers)
from .constants import (
    # Technical Indicators
    PERCENTAGE_SCALE,
    BARS_PER_HOUR,
    BARS_PER_DAY,
    DEFAULT_RSI_PERIOD,
    DEFAULT_ATR_PERIOD,
    DEFAULT_ADX_PERIOD,
    # Normalization
    DEFAULT_CLIP_MIN,
    DEFAULT_CLIP_MAX,
    DEFAULT_ZSCORE_LOOKBACK,
    # Risk Management
    MIN_CONFIDENCE_THRESHOLD,
    MAX_POSITION_SIZE,
    DEFAULT_STOP_LOSS_PCT,
    MAX_DAILY_LOSS_PCT,
    MAX_DRAWDOWN_PCT,
    # Market Hours
    MARKET_OPEN_HOUR,
    MARKET_CLOSE_HOUR,
    # Model Inference
    DEFAULT_OBSERVATION_DIM,
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
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

    # Secrets Management
    'SecretManager',
    'get_secret',

    # Event Bus
    'EventBus',
    'TradeEvent',
    'RiskEvent',
    'SystemEvent',
    'get_event_bus',
    'subscribe_to',
    'EventPriority',

    # Decorators - Resilience
    'with_retry',
    'with_timing',
    'with_circuit_breaker',
    'with_timeout',
    'with_fallback',
    'CircuitBreaker',
    'CircuitBreakerOpenError',

    # Logging
    'LoggerFactory',
    'get_logger',

    # Constants
    'PERCENTAGE_SCALE',
    'BARS_PER_HOUR',
    'BARS_PER_DAY',
    'DEFAULT_RSI_PERIOD',
    'DEFAULT_ATR_PERIOD',
    'DEFAULT_ADX_PERIOD',
    'DEFAULT_CLIP_MIN',
    'DEFAULT_CLIP_MAX',
    'DEFAULT_ZSCORE_LOOKBACK',
    'MIN_CONFIDENCE_THRESHOLD',
    'MAX_POSITION_SIZE',
    'DEFAULT_STOP_LOSS_PCT',
    'MAX_DAILY_LOSS_PCT',
    'MAX_DRAWDOWN_PCT',
    'MARKET_OPEN_HOUR',
    'MARKET_CLOSE_HOUR',
    'DEFAULT_OBSERVATION_DIM',
    'ACTION_HOLD',
    'ACTION_BUY',
    'ACTION_SELL',
]
