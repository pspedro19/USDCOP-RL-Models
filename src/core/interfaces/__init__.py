"""
Core Interfaces for USD/COP Trading System
==========================================

Defines abstract interfaces for dependency injection and SOLID principles.

Interface Segregation Principle (ISP):
- IFeatureCalculator: Calculate single feature
- INormalizer: Normalize values
- IObservationBuilder: Build observation vectors
- IConfigLoader: Load configuration
- IModelLoader: Load models
- IPredictor: Run predictions
- IEnsembleStrategy: Combine ensemble results
- IRiskCheck: Individual risk checks
- IStateRepository: Generic state persistence
- IDailyStatsRepository: Daily stats persistence

Author: Trading Team
Version: 2.0.0
Date: 2025-01-14
"""

# Original interfaces
from .config_loader import IConfigLoader
from .feature_calculator import IFeatureCalculator

# Inference interfaces (Phase 2 - Strategy Pattern)
from .inference import (
    EnsembleResult,
    IEnsembleStrategy,
    IHealthChecker,
    IInferenceEngine,
    IModelLoader,
    InferenceResult,
    IPredictor,
    SignalType,
)
from .normalizer import INormalizer
from .observation_builder import IObservationBuilder

# Repository interfaces (Phase 4 - Repository Pattern)
from .repository import (
    ICacheRepository,
    IDailyStatsRepository,
    IHashRepository,
    IListRepository,
    IStateRepository,
    ITradeLogRepository,
)

# Risk interfaces (Phase 3 - Chain of Responsibility)
from .risk import (
    DailyStats,
    FullRiskCheckResult,
    ICircuitBreaker,
    ICooldownManager,
    IPositionSizer,
    IRiskCheck,
    IRiskManager,
    ITradingHoursChecker,
    RiskCheckResult,
    RiskContext,
    RiskStatus,
)

# Storage interfaces (Phase 5 - MinIO-First Architecture)
from .storage import (
    ArtifactMetadata,
    IABComparisonRepository,
    IBacktestRepository,
    IDatasetRepository,
    IModelRepository,
    IntegrityError,
    IObjectStorageRepository,
    ObjectNotFoundError,
    StorageError,
)

__all__ = [
    # Original interfaces
    'IFeatureCalculator',
    'INormalizer',
    'IObservationBuilder',
    'IConfigLoader',

    # Inference interfaces
    'SignalType',
    'InferenceResult',
    'EnsembleResult',
    'IModelLoader',
    'IPredictor',
    'IEnsembleStrategy',
    'IHealthChecker',
    'IInferenceEngine',

    # Risk interfaces
    'RiskStatus',
    'RiskContext',
    'RiskCheckResult',
    'DailyStats',
    'FullRiskCheckResult',
    'IRiskCheck',
    'ITradingHoursChecker',
    'ICircuitBreaker',
    'ICooldownManager',
    'IPositionSizer',
    'IRiskManager',

    # Repository interfaces
    'IStateRepository',
    'IHashRepository',
    'IListRepository',
    'IDailyStatsRepository',
    'ITradeLogRepository',
    'ICacheRepository',

    # Storage interfaces
    'ArtifactMetadata',
    'IObjectStorageRepository',
    'IDatasetRepository',
    'IModelRepository',
    'IBacktestRepository',
    'IABComparisonRepository',
    'ObjectNotFoundError',
    'StorageError',
    'IntegrityError',
]
