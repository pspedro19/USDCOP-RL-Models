"""
Feature Store Builders Package
===============================
Contains the canonical feature builders that serve as the Single Source of Truth (SSOT)
for feature calculation across all contexts: Training, Inference, and Backtest.

Design Principles:
- Single Source of Truth: All contexts use CanonicalFeatureBuilder
- Interface Segregation: IFeatureBuilder defines minimal contract
- Dependency Inversion: Depend on abstractions, not implementations
- Immutable Contracts: Feature order and dimensions are frozen

Usage:
    from feature_store.builders import CanonicalFeatureBuilder, IFeatureBuilder

    # For training
    builder = CanonicalFeatureBuilder.for_training(config)

    # For inference
    builder = CanonicalFeatureBuilder.for_inference(model_contract)

    # For backtesting
    builder = CanonicalFeatureBuilder.for_backtest(backtest_config)

Author: Trading Team
Version: 1.0.0
Created: 2025-01-16
"""

from .canonical_feature_builder import (
    IFeatureBuilder,
    CanonicalFeatureBuilder,
    NormStatsNotFoundError,
    ObservationDimensionError,
    FeatureCalculationError,
    BuilderContext,
)

__all__ = [
    "IFeatureBuilder",
    "CanonicalFeatureBuilder",
    "NormStatsNotFoundError",
    "ObservationDimensionError",
    "FeatureCalculationError",
    "BuilderContext",
]
