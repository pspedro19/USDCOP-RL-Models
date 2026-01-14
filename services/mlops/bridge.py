"""
MLOps Bridge Module
===================

Bridges the services/mlops layer with the refactored src/ architecture.
Provides backward-compatible APIs while using the new SOLID implementations.

This module initializes the service container and provides factory functions
for accessing refactored components.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import sys
import os
import logging
from typing import Optional, Any
from functools import lru_cache

# Add project root to path if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_bootstrap_initialized = False
_container = None


def _ensure_bootstrap():
    """Ensure bootstrap is initialized (lazy initialization)."""
    global _bootstrap_initialized, _container

    if _bootstrap_initialized:
        return _container

    try:
        from src.bootstrap import bootstrap_production, get_container
        from mlops.config import get_config

        config = get_config()
        _container = bootstrap_production(config)
        _bootstrap_initialized = True

        logger.info("MLOps bridge: Bootstrap initialized successfully")
        return _container

    except Exception as e:
        logger.warning(f"MLOps bridge: Bootstrap failed, using legacy mode: {e}")
        _bootstrap_initialized = True
        return None


def get_service_container():
    """Get the service container (initializes on first call)."""
    return _ensure_bootstrap()


@lru_cache(maxsize=1)
def get_inference_engine_v2():
    """
    Get the refactored InferenceEngine from src/inference.

    Returns the new SOLID-compliant inference engine with:
    - Strategy pattern for ensemble
    - Separated ModelLoader and Predictor
    - Full health checking
    """
    container = _ensure_bootstrap()

    if container:
        try:
            return container.resolve("inference_engine")
        except Exception as e:
            logger.warning(f"Failed to resolve from container: {e}")

    # Fallback to direct instantiation
    from src.inference import InferenceEngine
    return InferenceEngine()


@lru_cache(maxsize=1)
def get_risk_check_chain():
    """
    Get the refactored RiskCheckChain from src/risk/checks.

    Returns the Chain of Responsibility implementation with:
    - Individual risk checks
    - Configurable order
    - Easy extensibility
    """
    container = _ensure_bootstrap()

    if container:
        try:
            return container.resolve("risk_check_chain")
        except Exception as e:
            logger.warning(f"Failed to resolve from container: {e}")

    # Fallback to direct instantiation
    from src.risk.checks import RiskCheckChain
    return RiskCheckChain.with_defaults()


@lru_cache(maxsize=1)
def get_ensemble_strategy_registry():
    """
    Get the EnsembleStrategyRegistry for accessing ensemble strategies.
    """
    from src.core.strategies import EnsembleStrategyRegistry
    return EnsembleStrategyRegistry


@lru_cache(maxsize=1)
def get_repository_factory():
    """
    Get the RepositoryFactory for accessing state/stats repositories.
    """
    container = _ensure_bootstrap()

    from src.repositories import RepositoryFactory
    return RepositoryFactory


def get_daily_stats_repository():
    """Get the daily stats repository."""
    container = _ensure_bootstrap()

    if container:
        try:
            return container.resolve("daily_stats_repository")
        except Exception:
            pass

    from src.repositories import RepositoryFactory
    return RepositoryFactory.get_daily_stats_repository()


def get_trade_log_repository():
    """Get the trade log repository."""
    container = _ensure_bootstrap()

    if container:
        try:
            return container.resolve("trade_log_repository")
        except Exception:
            pass

    from src.repositories import RepositoryFactory
    return RepositoryFactory.get_trade_log_repository()


# Risk manager with chain integration
def get_risk_manager_with_chain():
    """
    Get a RiskManager that uses the new SOLID RiskCheckChain.

    This provides a RiskManager with the Chain of Responsibility
    pattern for risk checks, allowing for easier testing and extension.
    """
    from mlops.risk_manager import get_risk_manager_with_chain as _get_rm
    return _get_rm()


# Health check for the bridge
def health_check() -> dict:
    """Check health of all bridge components."""
    health = {
        "bridge_status": "healthy",
        "bootstrap_initialized": _bootstrap_initialized,
        "components": {}
    }

    try:
        container = get_service_container()
        if container:
            container_health = container.health_check()
            health["components"]["container"] = container_health
            if container_health.get("status") != "healthy":
                health["bridge_status"] = "degraded"
    except Exception as e:
        health["components"]["container"] = {"status": "error", "error": str(e)}
        health["bridge_status"] = "degraded"

    return health
