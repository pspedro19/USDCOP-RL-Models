"""
Feature Store - Unified Feature Engineering System
===================================================
Single Source of Truth for all feature calculations across pipelines.

This module consolidates:
- Feature contracts (immutable specification)
- Feature calculators (RSI, ATR, ADX with Wilder's smoothing)
- Feature builder (training, backtest, inference)
- Normalization stats management

Architecture:
    Training (L3) ──┐
    Backtest (L4) ──┼──▶ feature_store.core ──▶ Consistent Features
    Inference (L5) ─┘

IMPORTANT: All legacy modules (src/features/*, src/core/calculators/*)
delegate to this module to ensure feature parity.

Usage:
    from feature_store import (
        get_contract,
        get_feature_builder,
        FEATURE_CONTRACT,
        UnifiedFeatureBuilder,
    )

    # Get contract
    contract = get_contract("current")
    print(contract.observation_dim)  # 15

    # Build features
    builder = get_feature_builder("current")
    obs = builder.build_observation(ohlcv, macro, position, timestamp, bar_idx)

Author: Trading Team
Version: 2.1.0
Created: 2025-01-12
"""

# Core module - Single Source of Truth
# Adapters for backward compatibility
from .adapters import (
    AdapterFactory,
    BacktestFeatureAdapter,
    InferenceObservationAdapter,
    NormStatsNotFoundError,
    TrainingFeatureAdapter,
)

# Canonical Feature Builder - SINGLE SOURCE OF TRUTH
from .builders import (
    BuilderContext,
    CanonicalFeatureBuilder,
    FeatureCalculationError,
    IFeatureBuilder,
    ObservationDimensionError,
)

# Pydantic contracts (for advanced use cases)
from .contracts import (
    CalculationParams,
    CalculationRequest,
    CalculationResult,
    FeatureBatch,
    FeatureSetSpec,
    FeatureSpec,
    FeatureVector,
    MacroDataInput,
    NormalizationParams,
    NormalizationStats,
    RawDataInput,
)
from .core import (
    FEATURE_CONTRACT,
    FEATURE_ORDER,
    NORM_STATS_PATH,
    OBSERVATION_DIM,
    ADXCalculator,
    ATRPercentCalculator,
    BaseCalculator,
    CalculatorRegistry,
    FeatureCategory,
    # Contracts
    FeatureContract,
    # Enums
    FeatureVersion,
    # Calculators
    IFeatureCalculator,
    LogReturnCalculator,
    MacroChangeCalculator,
    MacroZScoreCalculator,
    NormalizationMethod,
    RSICalculator,
    SmoothingMethod,
    TechnicalPeriods,
    TradingHours,
    # Builder
    UnifiedFeatureBuilder,
    # Factories
    get_contract,
    get_feature_builder,
)

# Feast Inference Service - Feature retrieval with fallback
from .feast_service import (
    FeastConnectionError,
    FeastFeatureNotFoundError,
    FeastInferenceService,
    FeastMetrics,
    FeastServiceError,
    create_feast_service,
)

# Feature Readers - Read pre-computed features from L1 pipeline
from .readers import (
    FeatureNotFoundError,
    FeatureOrderMismatchError,
    FeatureReader,
    FeatureReaderError,
    FeatureResult,
    StaleFeatureError,
)

__all__ = [
    # Enums
    "FeatureVersion",
    "FeatureCategory",
    "SmoothingMethod",
    "NormalizationMethod",

    # Contracts (Core)
    "FeatureContract",
    "FEATURE_CONTRACT",
    "FEATURE_ORDER",
    "OBSERVATION_DIM",
    "NORM_STATS_PATH",
    "TechnicalPeriods",
    "TradingHours",

    # Contracts (Pydantic)
    "FeatureSpec",
    "FeatureSetSpec",
    "FeatureVector",
    "FeatureBatch",
    "NormalizationStats",
    "NormalizationParams",
    "CalculationParams",
    "RawDataInput",
    "MacroDataInput",
    "CalculationRequest",
    "CalculationResult",

    # Calculators
    "IFeatureCalculator",
    "BaseCalculator",
    "LogReturnCalculator",
    "RSICalculator",
    "ATRPercentCalculator",
    "ADXCalculator",
    "MacroZScoreCalculator",
    "MacroChangeCalculator",
    "CalculatorRegistry",

    # Builder
    "UnifiedFeatureBuilder",

    # Factories
    "get_contract",
    "get_feature_builder",

    # Adapters
    "InferenceObservationAdapter",
    "TrainingFeatureAdapter",
    "BacktestFeatureAdapter",
    "AdapterFactory",
    "NormStatsNotFoundError",

    # Canonical Builder (SSOT)
    "IFeatureBuilder",
    "CanonicalFeatureBuilder",
    "BuilderContext",
    "ObservationDimensionError",
    "FeatureCalculationError",

    # Feast Inference Service
    "FeastInferenceService",
    "FeastServiceError",
    "FeastConnectionError",
    "FeastFeatureNotFoundError",
    "FeastMetrics",
    "create_feast_service",

    # Feature Readers (L1 -> L5 integration)
    "FeatureResult",
    "FeatureReader",
    "FeatureReaderError",
    "FeatureNotFoundError",
    "StaleFeatureError",
    "FeatureOrderMismatchError",
]

__version__ = "2.3.0"
