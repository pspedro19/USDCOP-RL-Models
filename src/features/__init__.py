"""
Features module for USD/COP RL Trading System.

This module provides feature building, normalization, and registry.
The current production version uses a 15-dimensional observation space (CTR-001, CTR-002).

Architecture:
    - contract.py: Feature contracts and specifications
    - registry.py: Feature registry (SSOT from YAML)
    - builder.py: Unified feature builder
    - calculators/: Individual feature calculators
    - normalizers/: Normalization strategies (Strategy Pattern)

Patterns:
    - Registry Pattern: Centralized feature definitions
    - Strategy Pattern: Interchangeable normalizers
    - Factory Pattern: Calculator and normalizer creation
"""

# Contract (CTR-002) - import first as it has no dependencies
from .contract import (
    FeatureContract,
    FEATURE_CONTRACT,
    get_contract,
    FEATURE_ORDER,
    OBSERVATION_DIM,
)

# Normalizers (CTR-006) - Strategy Pattern implementations
from . import normalizers
from .normalizers import (
    Normalizer,
    ZScoreNormalizer,
    MinMaxNormalizer,
    ClipNormalizer,
    NoOpNormalizer,
    NormalizerFactory,
)

# Registry (CTR-007) - SSOT from YAML
from .registry import (
    FeatureRegistry,
    FeatureDefinition,
    NormalizationConfig,
    get_registry,
    get_feature_order,
    get_feature_hash,
)

# FeatureBuilder (CTR-001) - depends on contract and calculators
from .builder import FeatureBuilder, create_feature_builder

# Circuit Breaker (Phase 13) - Feature quality monitoring
from .circuit_breaker import (
    FeatureCircuitBreaker,
    FeatureCircuitBreakerError,
    CircuitBreakerConfig,
    CircuitBreakerState,
    get_circuit_breaker,
)

# Gap Handler (Phase 14) - Centralized gap handling
from .gap_handler import (
    GapHandler,
    GapConfig,
    GapStatistics,
    get_gap_handler,
    handle_gaps,
    validate_ohlcv_data,
)

# Feature Reader (Week 1) - Feature store reading and validation
from .feature_reader import (
    FeatureReader,
    FeatureReadResult,
    FeatureValidationError,
    EXPECTED_FEATURE_ORDER,
)

# Temporal Joins (P0-06) - Point-in-time correct joins
from .temporal_joins import (
    merge_price_with_macro,
    validate_no_lookahead,
    join_multiple_sources,
    TemporalJoinConfig,
    JoinStatistics,
    get_join_statistics,
    fill_missing_macro,
)

# Trading Hours Filter (P0-07) - Colombian trading hours filtering
# Refactored to use SSOT from config/trading_calendar.json
from .trading_hours_filter import (
    # Protocol
    ITradingHoursFilter,
    # Config classes
    TradingCalendarConfig,
    TradingSession,
    # Main class
    TradingHoursFilter,
    # Factory
    TradingHoursFilterFactory,
    MarketType,
    HolidayType,
    # Config loading
    load_trading_calendar_config,
    ConfigLoadError,
    # Convenience functions
    get_trading_hours_filter,
    filter_to_trading_hours,
    is_trading_time,
    # Integration
    create_filter_from_trading_calendar,
    # Constants
    TRADING_WEEKDAYS,
    WEEKEND_DAYS,
    TRADING_CALENDAR_CONFIG_PATH,
)

__all__ = [
    # Feature Builder
    'FeatureBuilder',
    'create_feature_builder',
    # Contract
    'FeatureContract',
    'FEATURE_CONTRACT',
    'get_contract',
    'FEATURE_ORDER',
    'OBSERVATION_DIM',
    # Normalizers
    'normalizers',
    'Normalizer',
    'ZScoreNormalizer',
    'MinMaxNormalizer',
    'ClipNormalizer',
    'NoOpNormalizer',
    'NormalizerFactory',
    # Registry
    'FeatureRegistry',
    'FeatureDefinition',
    'NormalizationConfig',
    'get_registry',
    'get_feature_order',
    'get_feature_hash',
    # Circuit Breaker (Phase 13)
    'FeatureCircuitBreaker',
    'FeatureCircuitBreakerError',
    'CircuitBreakerConfig',
    'CircuitBreakerState',
    'get_circuit_breaker',
    # Gap Handler (Phase 14)
    'GapHandler',
    'GapConfig',
    'GapStatistics',
    'get_gap_handler',
    'handle_gaps',
    'validate_ohlcv_data',
    # Feature Reader (Week 1)
    'FeatureReader',
    'FeatureReadResult',
    'FeatureValidationError',
    'EXPECTED_FEATURE_ORDER',
    # Temporal Joins (P0-06)
    'merge_price_with_macro',
    'validate_no_lookahead',
    'join_multiple_sources',
    'TemporalJoinConfig',
    'JoinStatistics',
    'get_join_statistics',
    'fill_missing_macro',
    # Trading Hours Filter (P0-07) - SSOT Version
    'ITradingHoursFilter',
    'TradingCalendarConfig',
    'TradingSession',
    'TradingHoursFilter',
    'TradingHoursFilterFactory',
    'MarketType',
    'HolidayType',
    'load_trading_calendar_config',
    'ConfigLoadError',
    'get_trading_hours_filter',
    'filter_to_trading_hours',
    'is_trading_time',
    'create_filter_from_trading_calendar',
    'TRADING_WEEKDAYS',
    'WEEKEND_DAYS',
    'TRADING_CALENDAR_CONFIG_PATH',
]
