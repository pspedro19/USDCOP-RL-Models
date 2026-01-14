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
from .core import (
    # Enums
    FeatureVersion,
    FeatureCategory,
    SmoothingMethod,
    NormalizationMethod,

    # Contracts
    FeatureContract,
    FEATURE_CONTRACT,
    FEATURE_ORDER,
    OBSERVATION_DIM,
    NORM_STATS_PATH,
    TechnicalPeriods,
    TradingHours,

    # Calculators
    IFeatureCalculator,
    BaseCalculator,
    LogReturnCalculator,
    RSICalculator,
    ATRPercentCalculator,
    ADXCalculator,
    MacroZScoreCalculator,
    MacroChangeCalculator,
    CalculatorRegistry,

    # Builder
    UnifiedFeatureBuilder,

    # Factories
    get_contract,
    get_feature_builder,
)

# Pydantic contracts (for advanced use cases)
from .contracts import (
    FeatureSpec,
    FeatureSetSpec,
    FeatureVector,
    FeatureBatch,
    NormalizationStats,
    NormalizationParams,
    CalculationParams,
    RawDataInput,
    MacroDataInput,
    CalculationRequest,
    CalculationResult,
)

# Adapters for backward compatibility
from .adapters import (
    InferenceObservationAdapter,
    TrainingFeatureAdapter,
    BacktestFeatureAdapter,
    AdapterFactory,
    NormStatsNotFoundError,
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
]

__version__ = "2.0.0"
