"""
Core Factories for USD/COP Trading System
==========================================

Factory Pattern implementations for creating calculators, normalizers, and storage.

Author: Pedro @ Lean Tech Solutions
Version: 2.0.0
Date: 2026-01-18
"""

from .feature_calculator_factory import FeatureCalculatorFactory
from .normalizer_factory import NormalizerFactory
from .storage_factory import (
    StorageBackend,
    StorageConfig,
    StorageFactory,
    get_storage_factory,
    get_dataset_repository,
    get_model_repository,
    get_backtest_repository,
    get_ab_comparison_repository,
)

__all__ = [
    # Feature and Normalizer Factories
    'FeatureCalculatorFactory',
    'NormalizerFactory',
    # Storage Factory
    'StorageBackend',
    'StorageConfig',
    'StorageFactory',
    'get_storage_factory',
    'get_dataset_repository',
    'get_model_repository',
    'get_backtest_repository',
    'get_ab_comparison_repository',
]
