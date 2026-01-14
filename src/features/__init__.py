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

# Calculators (CTR-005) - import before builder
from . import calculators

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
    # Calculators
    'calculators',
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
]
