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
    DictConfigProvider,
    InMemoryCacheProvider,
    InMemoryDailyStatsRepository,
    InMemoryStateRepository,
    InMemoryTradeRepository,
    # Default implementations
    NullEventBus,
    NullFeatureBuilder,
    NullModelLoader,
    NullPredictor,
    NullRiskManager,
    StandardLoggerAdapter,
    StandardLoggerFactory,
)

# Protocol interfaces
from .protocols import (
    # Base types
    Event,
    EventListener,
    # Cache protocols
    ICacheProvider,
    # Config protocols
    IConfigProvider,
    IDailyStatsRepository,
    IEnsembleStrategy,
    # Event protocols
    IEventBus,
    # Feature protocols
    IFeatureBuilder,
    IFeatureCalculator,
    # Health protocols
    IHealthChecker,
    # Logging protocols
    ILogger,
    ILoggerFactory,
    # Model protocols
    IModelLoader,
    INormalizer,
    IPredictor,
    # Risk protocols
    IRiskManager,
    IStateRepository,
    # Repository protocols
    ITradeRepository,
    PredictionResult,
    SignalType,
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
