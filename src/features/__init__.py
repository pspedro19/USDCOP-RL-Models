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
# Normalizers (CTR-006) - Strategy Pattern implementations
from . import normalizers

# FeatureBuilder (CTR-001) - depends on contract and calculators
from .builder import FeatureBuilder, create_feature_builder

# Calculator Registry (SSOT v2.0) - Dynamic feature calculation
from .calculator_registry import (
    CalculatorRegistry,
    calculate_features_ssot,
    # Individual calculators
    calculate_log_returns,
    calculate_macro_zscore,
    calculate_pct_change,
    calculate_rsi_wilders,
    calculate_spread_zscore,
    calculate_trend_z,
    calculate_volatility_pct,
    get_calculator_registry,
    normalize_features,
)

# Circuit Breaker (Phase 13) - Feature quality monitoring
from .circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    FeatureCircuitBreaker,
    FeatureCircuitBreakerError,
    get_circuit_breaker,
)
from .contract import (
    FEATURE_CONTRACT,
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FeatureContract,
    get_contract,
)

# Feature Reader (Week 1) - Feature store reading and validation
from .feature_reader import (
    EXPECTED_FEATURE_ORDER,
    FeatureReader,
    FeatureReadResult,
    FeatureValidationError,
)

# Gap Handler (Phase 14) - Centralized gap handling
from .gap_handler import (
    GapConfig,
    GapHandler,
    GapStatistics,
    get_gap_handler,
    handle_gaps,
    validate_ohlcv_data,
)
from .normalizers import (
    ClipNormalizer,
    MinMaxNormalizer,
    NoOpNormalizer,
    Normalizer,
    NormalizerFactory,
    ZScoreNormalizer,
)

# Registry (CTR-007) - SSOT from YAML
from .registry import (
    FeatureDefinition,
    FeatureRegistry,
    NormalizationConfig,
    get_feature_hash,
    get_feature_order,
    get_registry,
)

# Temporal Joins (P0-06) - Point-in-time correct joins
from .temporal_joins import (
    JoinStatistics,
    TemporalJoinConfig,
    fill_missing_macro,
    get_join_statistics,
    join_multiple_sources,
    merge_price_with_macro,
    validate_no_lookahead,
)

# Trading Hours Filter (P0-07) - Colombian trading hours filtering
# Refactored to use SSOT from config/trading_calendar.json
from .trading_hours_filter import (
    TRADING_CALENDAR_CONFIG_PATH,
    # Constants
    TRADING_WEEKDAYS,
    WEEKEND_DAYS,
    ConfigLoadError,
    HolidayType,
    # Protocol
    ITradingHoursFilter,
    MarketType,
    # Config classes
    TradingCalendarConfig,
    # Main class
    TradingHoursFilter,
    # Factory
    TradingHoursFilterFactory,
    TradingSession,
    # Integration
    create_filter_from_trading_calendar,
    filter_to_trading_hours,
    # Convenience functions
    get_trading_hours_filter,
    is_trading_time,
    # Config loading
    load_trading_calendar_config,
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
    # Calculator Registry (SSOT v2.0)
    'CalculatorRegistry',
    'get_calculator_registry',
    'calculate_features_ssot',
    'normalize_features',
    'calculate_log_returns',
    'calculate_rsi_wilders',
    'calculate_volatility_pct',
    'calculate_trend_z',
    'calculate_macro_zscore',
    'calculate_spread_zscore',
    'calculate_pct_change',
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
