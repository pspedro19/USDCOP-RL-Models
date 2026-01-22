"""
Dependency Injection Container
==============================

Provides dependency injection infrastructure for the USDCOP trading system.

Components:
- ApplicationContext: Immutable context with explicit dependencies (preferred)
- ServiceContainer: Legacy service locator (deprecated, use ApplicationContext)
- Protocols: Interface definitions for all services

Migration Guide:
    # OLD (ServiceLocator anti-pattern - deprecated)
    container = ServiceContainer.get_instance()
    engine = container.resolve("inference_engine")

    # NEW (Dependency Injection - preferred)
    context = ApplicationContext.create_production(config)
    service = InferenceService(context)

Author: Trading Team
Version: 2.0.0
Date: 2026-01-16
"""

# New DI infrastructure (preferred)
from .application_context import (
    ApplicationContext,
    ContextHolder,
    # Default implementations
    NullEventBus,
    NullPredictor,
    NullModelLoader,
    NullFeatureBuilder,
    NullRiskManager,
    StandardLoggerAdapter,
    StandardLoggerFactory,
    InMemoryStateRepository,
    InMemoryTradeRepository,
    InMemoryCacheProvider,
    InMemoryDailyStatsRepository,
    DictConfigProvider,
)

# Protocol interfaces
from .protocols import (
    # Base types
    Event,
    SignalType,
    PredictionResult,
    EventListener,
    # Feature protocols
    IFeatureBuilder,
    IFeatureCalculator,
    INormalizer,
    # Model protocols
    IModelLoader,
    IPredictor,
    IEnsembleStrategy,
    # Repository protocols
    ITradeRepository,
    IStateRepository,
    IDailyStatsRepository,
    # Event protocols
    IEventBus,
    # Logging protocols
    ILogger,
    ILoggerFactory,
    # Health protocols
    IHealthChecker,
    # Risk protocols
    IRiskManager,
    # Config protocols
    IConfigProvider,
    # Cache protocols
    ICacheProvider,
)

# Legacy service container (deprecated - use ApplicationContext instead)
from .service_container import ServiceContainer

__all__ = [
    # Main DI infrastructure
    'ApplicationContext',
    'ContextHolder',

    # Protocol interfaces
    'Event',
    'SignalType',
    'PredictionResult',
    'EventListener',
    'IFeatureBuilder',
    'IFeatureCalculator',
    'INormalizer',
    'IModelLoader',
    'IPredictor',
    'IEnsembleStrategy',
    'ITradeRepository',
    'IStateRepository',
    'IDailyStatsRepository',
    'IEventBus',
    'ILogger',
    'ILoggerFactory',
    'IHealthChecker',
    'IRiskManager',
    'IConfigProvider',
    'ICacheProvider',

    # Default implementations
    'NullEventBus',
    'NullPredictor',
    'NullModelLoader',
    'NullFeatureBuilder',
    'NullRiskManager',
    'StandardLoggerAdapter',
    'StandardLoggerFactory',
    'InMemoryStateRepository',
    'InMemoryTradeRepository',
    'InMemoryCacheProvider',
    'InMemoryDailyStatsRepository',
    'DictConfigProvider',

    # Legacy (deprecated)
    'ServiceContainer',
]
