"""
Application Bootstrap
=====================

Configures all dependencies for production or testing.

Usage:
    # Production
    from src.bootstrap import bootstrap_production
    container = bootstrap_production(config)

    # Testing
    from src.bootstrap import bootstrap_testing
    container = bootstrap_testing()

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import logging
from typing import Optional, Any

from src.core.container import ServiceContainer
from src.repositories import RepositoryFactory

logger = logging.getLogger(__name__)


def bootstrap_production(
    config: Optional[Any] = None,
    redis_client: Optional[Any] = None,
) -> ServiceContainer:
    """
    Configure all dependencies for production.

    Args:
        config: MLOpsConfig or similar configuration object
        redis_client: Redis client instance (optional, will create if not provided)

    Returns:
        Configured ServiceContainer
    """
    container = ServiceContainer.get_instance()
    container.clear()

    logger.info("Bootstrapping production environment...")

    # Register configuration
    if config:
        container.register_singleton("config", config)

    # Configure Redis
    if redis_client:
        container.register_singleton("redis", redis_client)
        RepositoryFactory.configure_redis(redis_client)
    else:
        # Try to create Redis client from config
        try:
            import redis as redis_lib
            if config and hasattr(config, 'redis'):
                redis_client = redis_lib.Redis(
                    host=config.redis.host,
                    port=config.redis.port,
                    db=config.redis.db,
                    password=config.redis.password,
                    socket_timeout=config.redis.socket_timeout,
                    decode_responses=True,
                )
                container.register_singleton("redis", redis_client)
                RepositoryFactory.configure_redis(redis_client)
                logger.info("Redis client created from config")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory: {e}")
            RepositoryFactory.configure_memory()

    # Register repositories
    container.register_factory(
        "state_repository",
        lambda c: RepositoryFactory.get_state_repository()
    )
    container.register_factory(
        "daily_stats_repository",
        lambda c: RepositoryFactory.get_daily_stats_repository()
    )
    container.register_factory(
        "trade_log_repository",
        lambda c: RepositoryFactory.get_trade_log_repository()
    )

    # Register inference engine
    def create_inference_engine(c: ServiceContainer):
        from src.inference import InferenceEngine
        cfg = c.try_resolve("config")
        strategy = "weighted_average"
        if cfg and hasattr(cfg, 'ensemble_strategy'):
            strategy = cfg.ensemble_strategy
        return InferenceEngine(config=cfg, ensemble_strategy=strategy)

    container.register_factory("inference_engine", create_inference_engine)

    # Register risk check chain
    def create_risk_check_chain(c: ServiceContainer):
        from src.risk.checks import RiskCheckChain
        cfg = c.try_resolve("config")
        risk_config = {}
        if cfg and hasattr(cfg, 'risk_limits'):
            risk_config = cfg.risk_limits.to_dict() if hasattr(cfg.risk_limits, 'to_dict') else {}
        return RiskCheckChain.with_defaults(config=risk_config)

    container.register_factory("risk_check_chain", create_risk_check_chain)

    # Register ensemble strategy registry (already initialized)
    from src.core.strategies.ensemble_strategies import EnsembleStrategyRegistry
    container.register_singleton("ensemble_registry", EnsembleStrategyRegistry)

    logger.info(f"Production bootstrap complete. Services: {container.get_registered_services()}")
    return container


def bootstrap_testing() -> ServiceContainer:
    """
    Configure dependencies for testing.

    Uses in-memory implementations, no external dependencies.

    Returns:
        Configured ServiceContainer
    """
    container = ServiceContainer.get_instance()
    container.clear()

    logger.info("Bootstrapping test environment...")

    # Configure in-memory repositories
    RepositoryFactory.configure_memory()

    # Register repositories
    container.register_factory(
        "state_repository",
        lambda c: RepositoryFactory.get_state_repository()
    )
    container.register_factory(
        "daily_stats_repository",
        lambda c: RepositoryFactory.get_daily_stats_repository()
    )
    container.register_factory(
        "trade_log_repository",
        lambda c: RepositoryFactory.get_trade_log_repository()
    )

    # Register mock inference engine
    def create_mock_engine(c: ServiceContainer):
        from src.inference import InferenceEngine
        return InferenceEngine(ensemble_strategy="weighted_average")

    container.register_factory("inference_engine", create_mock_engine)

    # Register risk check chain with relaxed limits
    def create_test_chain(c: ServiceContainer):
        from src.risk.checks import RiskCheckChain
        return RiskCheckChain.with_defaults(config={
            "min_confidence": 0.0,  # No confidence requirement
            "max_daily_loss": -1.0,  # Very high tolerance
            "max_drawdown": -1.0,
            "max_consecutive_losses": 100,
            "max_trades_per_day": 1000,
        })

    container.register_factory("risk_check_chain", create_test_chain)

    logger.info(f"Test bootstrap complete. Services: {container.get_registered_services()}")
    return container


def bootstrap_minimal() -> ServiceContainer:
    """
    Minimal bootstrap for quick operations.

    Only registers core services without inference or risk.

    Returns:
        Configured ServiceContainer
    """
    container = ServiceContainer.get_instance()
    container.clear()

    logger.info("Minimal bootstrap...")

    # In-memory only
    RepositoryFactory.configure_memory()

    container.register_factory(
        "state_repository",
        lambda c: RepositoryFactory.get_state_repository()
    )

    logger.info("Minimal bootstrap complete")
    return container


def get_container() -> ServiceContainer:
    """
    Get the service container.

    If not bootstrapped, returns uninitialized container.
    """
    return ServiceContainer.get_instance()


def resolve(name: str) -> Any:
    """
    Convenience function to resolve service.

    Args:
        name: Service name

    Returns:
        Service instance
    """
    return ServiceContainer.get_instance().resolve(name)
