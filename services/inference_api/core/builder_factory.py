"""
Builder Factory - Explicit Registration Pattern
================================================
Replaces fragile string matching with explicit builder registration.

SOLID Principles:
- Single Responsibility: Only creates observation builders
- Open/Closed: New builders via registration, not code changes
- Dependency Inversion: Depends on BuilderType enum abstraction
- Liskov Substitution: All builders implement same interface

Design Patterns:
- Factory Pattern: Creates appropriate builder instances
- Registry Pattern: Explicit registration of builders
- Strategy Pattern: Different builders for different models

CRITICAL: This factory uses EXPLICIT registration.
No string matching like "v1" in model_id - that's FRAGILE and error-prone.
"""

import logging
from pathlib import Path
from typing import Dict, Type, Optional, Protocol
from functools import lru_cache

from .observation_builder import ObservationBuilder
from ..contracts.model_contract import (
    BuilderType,
    ModelRegistry,
    get_model_contract,
    BuilderNotRegisteredError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Builder Protocol (Interface)
# =============================================================================

class ObservationBuilderProtocol(Protocol):
    """
    Protocol defining the interface all builders must implement.

    This enables Liskov Substitution Principle - any builder
    can be used interchangeably where this protocol is expected.
    """
    OBSERVATION_DIM: int

    def build_observation(self, df, bar_idx: int, position: float, **kwargs):
        """Build observation vector from market data"""
        ...


# Type alias for builder instances
BuilderInstance = ObservationBuilder


# =============================================================================
# Builder Factory
# =============================================================================

class BuilderFactory:
    """
    Factory for creating observation builders.

    EXPLICIT REGISTRATION - No string matching!

    Usage:
        # Get builder for a registered model
        builder = BuilderFactory.get_builder("ppo_primary")

        # Or get by explicit builder type
        builder = BuilderFactory.get_builder_by_type(BuilderType.CURRENT_15DIM)
    """

    # Registry of builder types to builder classes
    _builder_classes: Dict[BuilderType, Type[BuilderInstance]] = {
        BuilderType.CURRENT_15DIM: ObservationBuilder,
    }

    # Cache of instantiated builders (to avoid re-loading norm_stats)
    _builder_instances: Dict[BuilderType, BuilderInstance] = {}

    @classmethod
    def register_builder(
        cls,
        builder_type: BuilderType,
        builder_class: Type[BuilderInstance]
    ) -> None:
        """
        Register a new builder type.

        Args:
            builder_type: BuilderType enum value
            builder_class: Class that implements ObservationBuilderProtocol
        """
        cls._builder_classes[builder_type] = builder_class
        logger.info(f"Registered builder: {builder_type.value} -> {builder_class.__name__}")

    @classmethod
    def get_builder(
        cls,
        model_id: str,
        norm_stats_path: Optional[Path] = None
    ) -> BuilderInstance:
        """
        Get observation builder for a model ID.

        This method uses EXPLICIT registration via ModelRegistry,
        NOT string matching on model_id.

        Args:
            model_id: Model identifier (e.g., "ppo_primary")
            norm_stats_path: Optional custom norm_stats path

        Returns:
            Appropriate ObservationBuilder instance

        Raises:
            BuilderNotRegisteredError: If model not registered
        """
        # Get model contract (explicit registration)
        contract = get_model_contract(model_id)

        # Get builder by type
        return cls.get_builder_by_type(contract.builder_type, norm_stats_path)

    @classmethod
    def get_builder_by_type(
        cls,
        builder_type: BuilderType,
        norm_stats_path: Optional[Path] = None
    ) -> BuilderInstance:
        """
        Get observation builder by explicit BuilderType.

        Args:
            builder_type: BuilderType enum value
            norm_stats_path: Optional custom norm_stats path

        Returns:
            Appropriate ObservationBuilder instance

        Raises:
            BuilderNotRegisteredError: If builder type not registered
        """
        if builder_type not in cls._builder_classes:
            raise BuilderNotRegisteredError(
                f"No builder registered for type '{builder_type.value}'. "
                f"Available types: {[t.value for t in cls._builder_classes.keys()]}"
            )

        # Return cached instance if no custom path and already created
        if norm_stats_path is None and builder_type in cls._builder_instances:
            return cls._builder_instances[builder_type]

        # Create new instance
        builder_class = cls._builder_classes[builder_type]

        if norm_stats_path:
            builder = builder_class(norm_stats_path=norm_stats_path)
        else:
            builder = builder_class()

        # Cache if using default path
        if norm_stats_path is None:
            cls._builder_instances[builder_type] = builder

        logger.info(
            f"Created builder: {builder_type.value} "
            f"(dim={builder.OBSERVATION_DIM if hasattr(builder, 'OBSERVATION_DIM') else '?'})"
        )

        return builder

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached builder instances"""
        cls._builder_instances.clear()
        logger.info("Builder cache cleared")

    @classmethod
    def list_builders(cls) -> Dict[str, str]:
        """List all registered builder types"""
        return {
            bt.value: bc.__name__
            for bt, bc in cls._builder_classes.items()
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def get_observation_builder(model_id: str) -> BuilderInstance:
    """
    Factory function to get the appropriate observation builder for a model.

    This is the main entry point for getting builders.
    Uses EXPLICIT registration via ModelContract, NOT string matching.

    Args:
        model_id: Model identifier (e.g., "ppo_primary", "ppo_secondary")

    Returns:
        Appropriate ObservationBuilder instance

    Raises:
        BuilderNotRegisteredError: If model not registered

    Example:
        builder = get_observation_builder("ppo_primary")
        obs = builder.build_observation(df, bar_idx=100, position=0.0)
    """
    return BuilderFactory.get_builder(model_id)


def get_builder_for_type(builder_type: BuilderType) -> BuilderInstance:
    """
    Get builder by explicit BuilderType enum.

    Args:
        builder_type: BuilderType enum value

    Returns:
        Appropriate ObservationBuilder instance
    """
    return BuilderFactory.get_builder_by_type(builder_type)
