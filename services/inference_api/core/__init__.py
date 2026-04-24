"""
Core modules for inference service
==================================
Central exports for all core components.

SOLID Principles:
- Single Responsibility: Each module has one job
- Open/Closed: New builders via BuilderFactory registration
- Dependency Inversion: Use BuilderFactory, not direct imports

IMPORTANT: Use get_observation_builder() from builder_factory,
NOT direct instantiation of builder classes.
"""

# Import the new BuilderFactory - this is the PREFERRED way to get builders
from .builder_factory import (
    BuilderFactory,
    get_builder_for_type,
    get_observation_builder,  # PREFERRED - uses explicit registration
)

# Import CachedInferenceEngine - P1-3: Feast caching for low-latency inference
from .cached_inference import (
    CachedInferenceEngine,
    create_cached_inference_engine,
)
from .data_loader import DataLoader

# Import the SSOT Feature Adapter (Phase 10 - Feature Consistency)
from .feature_adapter import (
    FeatureCircuitBreakerConfig,
    FeatureCircuitBreakerError,
    InferenceFeatureAdapter,
    calculate_adx_wilder,
    calculate_atr_wilder,
    calculate_rsi_wilder,
    get_feature_adapter,
)
from .inference_engine import InferenceEngine
from .observation_builder import NormStatsNotFoundError, ObservationBuilder
from .trade_persister import TradePersister
from .trade_simulator import TradeSimulator

__all__ = [
    # Core services
    "DataLoader",
    "InferenceEngine",
    "TradeSimulator",
    "TradePersister",
    # Observation builders (prefer using BuilderFactory)
    "ObservationBuilder",
    # SSOT Feature Adapter (Phase 10 - Feature Consistency)
    "InferenceFeatureAdapter",
    "get_feature_adapter",
    "calculate_rsi_wilder",
    "calculate_atr_wilder",
    "calculate_adx_wilder",
    # Builder Factory (PREFERRED way to get builders)
    "BuilderFactory",
    "get_observation_builder",
    "get_builder_for_type",
    # Cached Inference Engine (P1-3: Feast caching)
    "CachedInferenceEngine",
    "create_cached_inference_engine",
    # Exceptions
    "NormStatsNotFoundError",
    "FeatureCircuitBreakerError",
    "FeatureCircuitBreakerConfig",
]


# =============================================================================
# DEPRECATED: Direct builder selection
# =============================================================================
# The old string-matching function is REMOVED.
# Use get_observation_builder(model_id) from builder_factory instead.
#
# OLD (FRAGILE - DO NOT USE):
#   if "v1" in model_id.lower():  # This is DANGEROUS string matching
#       return ObservationBuilderV1()
#
# NEW (EXPLICIT - USE THIS):
#   from .builder_factory import get_observation_builder
#   builder = get_observation_builder("ppo_primary")  # Uses ModelRegistry
# =============================================================================
