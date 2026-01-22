"""
Application Context - Dependency Injection Container
=====================================================

Replaces the ServiceLocator anti-pattern with proper Dependency Injection
using an immutable ApplicationContext that holds all dependencies.

Design Principles:
- Dependency Injection: Dependencies are passed through constructor
- Immutability: Context cannot be modified after creation
- Explicit Dependencies: All dependencies are visible in constructor
- Single Responsibility: Context only holds dependencies, doesn't create them

Benefits over ServiceLocator:
1. Testable: Easy to mock dependencies
2. Explicit: Dependencies are visible in function signatures
3. No Global State: Avoids hidden coupling through singletons
4. Type-Safe: Full IDE support and type checking

Usage:
    # Production
    context = ApplicationContext.create_production(config)
    service = InferenceService(context)

    # Testing
    context = ApplicationContext.create_for_testing(
        predictor=mock_predictor,
    )
    service = InferenceService(context)

Author: Trading Team
Version: 1.0.0
Date: 2026-01-16
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar

from .protocols import (
    IEventBus,
    IFeatureBuilder,
    ILoggerFactory,
    IModelLoader,
    IPredictor,
    IRiskManager,
    IStateRepository,
    ITradeRepository,
    IDailyStatsRepository,
    ICacheProvider,
    IConfigProvider,
)

if TYPE_CHECKING:
    from src.config import TradingConfig


T = TypeVar('T')

logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT IMPLEMENTATIONS (Null Objects and Adapters)
# =============================================================================

class NullEventBus:
    """Null Object pattern - does nothing but satisfies the interface."""

    def subscribe(self, event_type: str, listener) -> None:
        pass

    def unsubscribe(self, event_type: str, listener) -> None:
        pass

    def publish(self, event) -> None:
        pass

    def publish_async(self, event) -> None:
        pass


class StandardLoggerAdapter:
    """Adapts standard library logger to ILogger protocol."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._logger.critical(message, extra=kwargs)


class StandardLoggerFactory:
    """Creates standard library loggers wrapped in adapter."""

    def __init__(self, log_level: int = logging.INFO):
        self._log_level = log_level
        self._loggers: Dict[str, StandardLoggerAdapter] = {}

    def get_logger(self, name: str) -> StandardLoggerAdapter:
        if name not in self._loggers:
            # Configure logger
            std_logger = logging.getLogger(name)
            std_logger.setLevel(self._log_level)
            self._loggers[name] = StandardLoggerAdapter(name)
        return self._loggers[name]


class InMemoryStateRepository:
    """In-memory implementation for testing and local development."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._ttls: Dict[str, Optional[int]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._store.get(key)

    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        self._store[key] = value
        self._ttls[key] = ttl
        return True

    def delete(self, key: str) -> bool:
        self._store.pop(key, None)
        self._ttls.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        return key in self._store


class InMemoryTradeRepository:
    """In-memory trade repository for testing."""

    def __init__(self):
        self._trades: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def save_trade(self, trade: Dict[str, Any]) -> str:
        self._counter += 1
        trade_id = f"trade_{self._counter}"
        trade["trade_id"] = trade_id
        self._trades[trade_id] = trade
        return trade_id

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        return self._trades.get(trade_id)

    def get_trades(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> list:
        trades = list(self._trades.values())
        if filters:
            for key, value in filters.items():
                trades = [t for t in trades if t.get(key) == value]
        return trades[:limit]

    def get_recent_trades(self, limit: int = 10) -> list:
        trades = list(self._trades.values())
        return trades[-limit:]


class InMemoryCacheProvider:
    """In-memory cache for testing and local development."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        self._cache[key] = value
        return True

    def delete(self, key: str) -> bool:
        self._cache.pop(key, None)
        return True

    def get_or_set(self, key: str, factory, ttl: int = 300) -> Any:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]


class InMemoryDailyStatsRepository:
    """In-memory daily stats repository for testing."""

    def __init__(self):
        self._stats: Dict[str, Dict[str, Any]] = {}

    def get(self, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if date is None:
            from datetime import datetime
            date = datetime.now().strftime("%Y-%m-%d")
        return self._stats.get(date)

    def save(self, stats: Dict[str, Any]) -> bool:
        date = stats.get("date")
        if date:
            self._stats[date] = stats
            return True
        return False

    def increment_field(self, date: str, field: str, amount: float = 1.0) -> float:
        if date not in self._stats:
            self._stats[date] = {}
        current = self._stats[date].get(field, 0.0)
        new_value = current + amount
        self._stats[date][field] = new_value
        return new_value


class NullPredictor:
    """Null Object pattern - returns HOLD with low confidence."""

    @property
    def model_name(self) -> str:
        return "null_predictor"

    def predict(self, observation) -> tuple:
        return (0, 0.0)  # HOLD, 0 confidence

    def predict_with_probs(self, observation):
        from .protocols import PredictionResult
        return PredictionResult(
            action=0,
            confidence=0.0,
            action_probs={"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0},
            latency_ms=0.0,
        )


class NullModelLoader:
    """Null Object pattern for model loader."""

    @property
    def model_name(self) -> str:
        return "null_model"

    @property
    def input_shape(self) -> tuple:
        return (15,)  # Default observation dim

    def load(self, path: str) -> bool:
        return True

    def is_loaded(self) -> bool:
        return True

    def warmup(self, iterations: int = 5) -> None:
        pass


class NullFeatureBuilder:
    """Null Object pattern for feature builder."""

    @property
    def observation_dim(self) -> int:
        return 15

    @property
    def feature_order(self) -> list:
        return [
            'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
            'rsi_9', 'atr_pct', 'adx_14',
            'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
            'brent_change_1d', 'rate_spread', 'usdmxn_ret_1h',
        ]

    def build_observation(self, ohlcv, macro, position, bar_idx):
        import numpy as np
        return np.zeros(15, dtype=np.float32)


class NullRiskManager:
    """Null Object pattern - always allows trades."""

    def check_trade_allowed(self, signal, confidence, position) -> tuple:
        return (True, "Risk checks disabled")

    def get_position_size(self, signal, confidence) -> float:
        return 1.0


class DictConfigProvider:
    """Simple dict-based config provider."""

    def __init__(self, config: Dict[str, Any]):
        self._config = config

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        value = self._config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {})


# =============================================================================
# APPLICATION CONTEXT
# =============================================================================

@dataclass(frozen=True)
class ApplicationContext:
    """
    Immutable context with explicit dependencies.

    All services receive this context through constructor injection,
    making dependencies explicit and testing straightforward.

    This replaces the ServiceLocator pattern where services would
    reach into a global container to get dependencies.

    Attributes:
        config: Application configuration
        feature_builder: Builds observation vectors
        model_loader: Loads model artifacts
        predictor: Runs model predictions
        trade_repository: Persists trade records
        state_repository: Key-value state storage
        daily_stats_repository: Daily statistics storage
        cache_provider: Caching operations
        event_bus: Publish-subscribe events
        risk_manager: Risk management
        logger_factory: Creates loggers
    """
    config: Any  # TradingConfig or similar
    feature_builder: IFeatureBuilder
    model_loader: IModelLoader
    predictor: IPredictor
    trade_repository: ITradeRepository
    state_repository: IStateRepository
    daily_stats_repository: IDailyStatsRepository
    cache_provider: ICacheProvider
    event_bus: IEventBus
    risk_manager: IRiskManager
    logger_factory: ILoggerFactory

    def get_logger(self, name: str):
        """Convenience method to get a logger."""
        return self.logger_factory.get_logger(name)

    @classmethod
    def create_production(
        cls,
        config: "TradingConfig",
        redis_client: Optional[Any] = None,
        db_url: Optional[str] = None,
    ) -> "ApplicationContext":
        """
        Create production context with real implementations.

        Args:
            config: TradingConfig instance
            redis_client: Optional Redis client for state/cache
            db_url: Optional database URL for trade persistence

        Returns:
            Configured ApplicationContext for production use
        """
        logger.info("Creating production ApplicationContext")

        # Feature Builder - use CanonicalFeatureBuilder (SSOT)
        try:
            from src.feature_store.builders import CanonicalFeatureBuilder
            feature_builder = CanonicalFeatureBuilder.for_inference()
        except ImportError:
            logger.warning("CanonicalFeatureBuilder not available, using null implementation")
            feature_builder = NullFeatureBuilder()

        # Model Loader and Predictor
        try:
            from src.inference import InferenceEngine
            engine = InferenceEngine(config=config)
            model_loader = engine
            predictor = engine
        except ImportError:
            logger.warning("InferenceEngine not available, using null implementations")
            model_loader = NullModelLoader()
            predictor = NullPredictor()

        # Repositories - use Redis if available, otherwise in-memory
        if redis_client is not None:
            try:
                from src.repositories import RepositoryFactory
                RepositoryFactory.configure_redis(redis_client)
                state_repository = RepositoryFactory.get_state_repository()
                daily_stats_repository = RepositoryFactory.get_daily_stats_repository()
                trade_repository = RepositoryFactory.get_trade_log_repository()
            except ImportError:
                logger.warning("RepositoryFactory not available, using in-memory")
                state_repository = InMemoryStateRepository()
                daily_stats_repository = InMemoryDailyStatsRepository()
                trade_repository = InMemoryTradeRepository()
        else:
            state_repository = InMemoryStateRepository()
            daily_stats_repository = InMemoryDailyStatsRepository()
            trade_repository = InMemoryTradeRepository()

        # Cache - use Redis if available
        if redis_client is not None:
            try:
                from src.repositories import RepositoryFactory
                cache_provider = RepositoryFactory.get_cache_repository()
            except (ImportError, AttributeError):
                cache_provider = InMemoryCacheProvider()
        else:
            cache_provider = InMemoryCacheProvider()

        # Event Bus
        try:
            from src.core.events.event_bus import EventBus
            event_bus = EventBus()
        except ImportError:
            event_bus = NullEventBus()

        # Risk Manager
        try:
            from src.risk.checks import RiskCheckChain
            risk_config = {}
            if hasattr(config, 'risk'):
                risk_config = {
                    "max_consecutive_losses": config.risk.max_consecutive_losses,
                    "cooldown_bars": config.risk.cooldown_bars_after_losses,
                }
            risk_manager = RiskCheckChain.with_defaults(config=risk_config)
        except ImportError:
            risk_manager = NullRiskManager()

        # Logger Factory
        log_level = logging.INFO
        if hasattr(config, 'logging') and hasattr(config.logging, 'level'):
            log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
        logger_factory = StandardLoggerFactory(log_level)

        context = cls(
            config=config,
            feature_builder=feature_builder,
            model_loader=model_loader,
            predictor=predictor,
            trade_repository=trade_repository,
            state_repository=state_repository,
            daily_stats_repository=daily_stats_repository,
            cache_provider=cache_provider,
            event_bus=event_bus,
            risk_manager=risk_manager,
            logger_factory=logger_factory,
        )

        logger.info("Production ApplicationContext created successfully")
        return context

    @classmethod
    def create_minimal(cls, config: Optional[Any] = None) -> "ApplicationContext":
        """
        Create minimal context with null/in-memory implementations.

        Useful for quick operations, scripts, and local development.

        Args:
            config: Optional configuration (uses empty dict if not provided)

        Returns:
            Minimal ApplicationContext
        """
        if config is None:
            config = DictConfigProvider({})

        return cls(
            config=config,
            feature_builder=NullFeatureBuilder(),
            model_loader=NullModelLoader(),
            predictor=NullPredictor(),
            trade_repository=InMemoryTradeRepository(),
            state_repository=InMemoryStateRepository(),
            daily_stats_repository=InMemoryDailyStatsRepository(),
            cache_provider=InMemoryCacheProvider(),
            event_bus=NullEventBus(),
            risk_manager=NullRiskManager(),
            logger_factory=StandardLoggerFactory(),
        )

    @classmethod
    def create_for_testing(cls, **overrides) -> "ApplicationContext":
        """
        Create test context with mocks.

        All dependencies default to mock objects, but can be
        overridden with specific implementations for testing.

        Args:
            **overrides: Specific dependencies to override

        Returns:
            ApplicationContext configured for testing

        Example:
            # Test with custom predictor
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = (1, 0.85)
            context = ApplicationContext.create_for_testing(
                predictor=mock_predictor,
            )

            # Test with real feature builder
            context = ApplicationContext.create_for_testing(
                feature_builder=FeatureBuilder(),
            )
        """
        from unittest.mock import MagicMock

        # Default mocks
        defaults = {
            "config": MagicMock(),
            "feature_builder": MagicMock(spec=IFeatureBuilder),
            "model_loader": MagicMock(spec=IModelLoader),
            "predictor": MagicMock(spec=IPredictor),
            "trade_repository": InMemoryTradeRepository(),
            "state_repository": InMemoryStateRepository(),
            "daily_stats_repository": InMemoryDailyStatsRepository(),
            "cache_provider": InMemoryCacheProvider(),
            "event_bus": MagicMock(spec=IEventBus),
            "risk_manager": MagicMock(spec=IRiskManager),
            "logger_factory": StandardLoggerFactory(logging.DEBUG),
        }

        # Apply overrides
        defaults.update(overrides)

        # Configure mock defaults for common operations
        if isinstance(defaults["predictor"], MagicMock):
            defaults["predictor"].predict.return_value = (0, 0.5)
            defaults["predictor"].model_name = "test_model"

        if isinstance(defaults["feature_builder"], MagicMock):
            import numpy as np
            defaults["feature_builder"].observation_dim = 15
            defaults["feature_builder"].feature_order = [
                'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
                'rsi_9', 'atr_pct', 'adx_14',
                'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
                'brent_change_1d', 'rate_spread', 'usdmxn_ret_1h',
            ]
            defaults["feature_builder"].build_observation.return_value = np.zeros(15)

        if isinstance(defaults["risk_manager"], MagicMock):
            defaults["risk_manager"].check_trade_allowed.return_value = (True, "OK")
            defaults["risk_manager"].get_position_size.return_value = 1.0

        if isinstance(defaults["model_loader"], MagicMock):
            defaults["model_loader"].is_loaded.return_value = True
            defaults["model_loader"].load.return_value = True

        return cls(**defaults)

    def with_overrides(self, **overrides) -> "ApplicationContext":
        """
        Create a new context with specific dependencies overridden.

        Since ApplicationContext is immutable (frozen dataclass),
        this creates a new instance with the specified changes.

        Args:
            **overrides: Dependencies to override

        Returns:
            New ApplicationContext with overrides applied

        Example:
            # Override predictor for specific test
            new_context = context.with_overrides(
                predictor=custom_predictor
            )
        """
        current = {
            "config": self.config,
            "feature_builder": self.feature_builder,
            "model_loader": self.model_loader,
            "predictor": self.predictor,
            "trade_repository": self.trade_repository,
            "state_repository": self.state_repository,
            "daily_stats_repository": self.daily_stats_repository,
            "cache_provider": self.cache_provider,
            "event_bus": self.event_bus,
            "risk_manager": self.risk_manager,
            "logger_factory": self.logger_factory,
        }
        current.update(overrides)
        return ApplicationContext(**current)

    def health_check(self) -> Dict[str, Any]:
        """
        Run health checks on all components.

        Returns:
            Dictionary with health status for each component
        """
        health = {
            "status": "healthy",
            "components": {},
        }

        # Check predictor
        try:
            if hasattr(self.predictor, 'health_check'):
                health["components"]["predictor"] = self.predictor.health_check()
            elif hasattr(self.model_loader, 'is_loaded'):
                health["components"]["predictor"] = {
                    "status": "healthy" if self.model_loader.is_loaded() else "degraded",
                    "loaded": self.model_loader.is_loaded(),
                }
            else:
                health["components"]["predictor"] = {"status": "unknown"}
        except Exception as e:
            health["components"]["predictor"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"

        # Check repositories
        for name, repo in [
            ("state_repository", self.state_repository),
            ("trade_repository", self.trade_repository),
        ]:
            try:
                if hasattr(repo, 'health_check'):
                    health["components"][name] = repo.health_check()
                else:
                    health["components"][name] = {"status": "healthy"}
            except Exception as e:
                health["components"][name] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"

        return health


# =============================================================================
# CONTEXT HOLDER (Optional - for gradual migration)
# =============================================================================

class ContextHolder:
    """
    Holds the current application context.

    This is provided for gradual migration from ServiceLocator.
    New code should receive context through constructor injection.
    Legacy code can use ContextHolder.get() during migration.

    WARNING: This is a transitional pattern. Direct DI is preferred.
    """
    _context: Optional[ApplicationContext] = None

    @classmethod
    def set(cls, context: ApplicationContext) -> None:
        """Set the current application context."""
        cls._context = context
        logger.info("ApplicationContext set in ContextHolder")

    @classmethod
    def get(cls) -> ApplicationContext:
        """
        Get the current application context.

        Raises:
            RuntimeError: If context has not been set
        """
        if cls._context is None:
            raise RuntimeError(
                "ApplicationContext not set. "
                "Call ContextHolder.set(context) at application startup, "
                "or prefer constructor injection."
            )
        return cls._context

    @classmethod
    def get_optional(cls) -> Optional[ApplicationContext]:
        """Get context or None if not set."""
        return cls._context

    @classmethod
    def reset(cls) -> None:
        """Reset context (for testing)."""
        cls._context = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    'ApplicationContext',

    # Context holder (transitional)
    'ContextHolder',

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
]
