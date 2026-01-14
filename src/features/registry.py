"""
Feature Registry - Single Source of Truth for Feature Definitions.

This module implements the Registry Pattern for feature definitions,
loading from the canonical YAML configuration.

Patterns Used:
- Registry Pattern: Centralized feature definitions
- Singleton Pattern: Single registry instance
- Data Class Pattern: Immutable feature definitions

Usage:
    registry = FeatureRegistry.load()
    feature = registry.get_feature("rsi_9")
    all_features = registry.get_all_features()
"""

import hashlib
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Path to the SSOT configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "feature_registry.yaml"


@dataclass(frozen=True)
class NormalizationConfig:
    """Immutable configuration for feature normalization."""

    method: str  # "zscore", "clip", "none"
    mean: float = 0.0
    std: float = 1.0
    clip: tuple = field(default_factory=lambda: (-5.0, 5.0))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationConfig":
        """Create from dictionary."""
        clip = data.get("clip", [-5.0, 5.0])
        if isinstance(clip, list):
            clip = tuple(clip)

        return cls(
            method=data.get("method", "zscore"),
            mean=data.get("mean", 0.0),
            std=data.get("std", 1.0),
            clip=clip,
        )


@dataclass(frozen=True)
class FeatureDefinition:
    """Immutable definition of a single feature.

    This class represents a single feature in the observation space,
    including its calculation method and normalization parameters.
    """

    name: str
    order: int
    category: str
    calculator: Optional[str]
    source: str
    params: Dict[str, Any]
    normalization: NormalizationConfig
    description: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            order=data["order"],
            category=data["category"],
            calculator=data.get("calculator"),
            source=data["source"],
            params=data.get("params", {}),
            normalization=NormalizationConfig.from_dict(
                data.get("normalization", {})
            ),
            description=data.get("description", ""),
        )

    @property
    def is_market_feature(self) -> bool:
        """Check if this is a market feature (not state)."""
        return self.category != "state"

    @property
    def is_state_feature(self) -> bool:
        """Check if this is a state feature."""
        return self.category == "state"


class FeatureRegistry:
    """Registry for all feature definitions.

    Implements Registry and Singleton patterns to provide a single
    source of truth for feature definitions.

    Example:
        registry = FeatureRegistry.load()
        feature = registry.get_feature("rsi_9")
        order = registry.get_feature_order()
    """

    _instance: Optional["FeatureRegistry"] = None

    def __init__(self, config: Dict[str, Any]):
        """Initialize registry from configuration dictionary."""
        self._config = config
        self._meta = config.get("_meta", {})
        self._observation_space = config.get("observation_space", {})
        self._features: Dict[str, FeatureDefinition] = {}
        self._features_by_order: Dict[int, FeatureDefinition] = {}

        # Parse feature definitions
        for feature_data in config.get("features", []):
            feature = FeatureDefinition.from_dict(feature_data)
            self._features[feature.name] = feature
            self._features_by_order[feature.order] = feature

        self._hash = self._compute_hash()

        logger.info(
            f"FeatureRegistry loaded: {len(self._features)} features, "
            f"version={self.version}, hash={self._hash[:8]}"
        )

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "FeatureRegistry":
        """Load registry from YAML configuration.

        Implements Singleton pattern - returns same instance on subsequent calls.

        Args:
            config_path: Path to YAML config. Uses default if not provided.

        Returns:
            FeatureRegistry instance
        """
        if cls._instance is not None:
            return cls._instance

        path = config_path or CONFIG_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"Feature registry config not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reload(cls, config_path: Optional[Path] = None) -> "FeatureRegistry":
        """Force reload of registry (clears singleton)."""
        cls._instance = None
        return cls.load(config_path)

    def _compute_hash(self) -> str:
        """Compute SHA256 hash of feature definitions for version tracking."""
        # Create deterministic string representation
        feature_str = ""
        for order in sorted(self._features_by_order.keys()):
            feature = self._features_by_order[order]
            feature_str += f"{feature.name}:{feature.order}:{feature.normalization.mean}:{feature.normalization.std};"

        return hashlib.sha256(feature_str.encode()).hexdigest()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def version(self) -> str:
        """Get registry version."""
        return self._meta.get("version", "unknown")

    @property
    def hash(self) -> str:
        """Get hash of feature definitions."""
        return self._hash

    @property
    def total_dimension(self) -> int:
        """Get total observation dimension."""
        return self._observation_space.get("total_dimension", 15)

    @property
    def market_features_count(self) -> int:
        """Get count of market features."""
        return self._observation_space.get("market_features_count", 13)

    @property
    def state_features_count(self) -> int:
        """Get count of state features."""
        return self._observation_space.get("state_features_count", 2)

    # =========================================================================
    # Feature Access
    # =========================================================================

    def get_feature(self, name: str) -> FeatureDefinition:
        """Get feature definition by name."""
        if name not in self._features:
            raise KeyError(f"Feature not found: {name}")
        return self._features[name]

    def get_feature_by_order(self, order: int) -> FeatureDefinition:
        """Get feature definition by order index."""
        if order not in self._features_by_order:
            raise KeyError(f"Feature at order {order} not found")
        return self._features_by_order[order]

    def get_all_features(self) -> List[FeatureDefinition]:
        """Get all features sorted by order."""
        return [
            self._features_by_order[i]
            for i in range(self.total_dimension)
        ]

    def get_market_features(self) -> List[FeatureDefinition]:
        """Get only market features (not state)."""
        return [f for f in self.get_all_features() if f.is_market_feature]

    def get_state_features(self) -> List[FeatureDefinition]:
        """Get only state features."""
        return [f for f in self.get_all_features() if f.is_state_feature]

    def get_feature_order(self) -> List[str]:
        """Get canonical feature order as list of names."""
        return self._observation_space.get("order", [])

    def get_features_by_category(self, category: str) -> List[FeatureDefinition]:
        """Get features filtered by category."""
        return [
            f for f in self._features.values()
            if f.category == category
        ]

    def get_features_by_source(self, source: str) -> List[FeatureDefinition]:
        """Get features filtered by data source."""
        return [
            f for f in self._features.values()
            if f.source == source
        ]

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_feature_vector(self, vector: List[float]) -> bool:
        """Validate that feature vector has correct dimension."""
        if len(vector) != self.total_dimension:
            raise ValueError(
                f"Feature vector dimension mismatch: "
                f"expected {self.total_dimension}, got {len(vector)}"
            )
        return True

    def get_validation_constraints(self) -> Dict[str, Any]:
        """Get validation constraints from config."""
        return self._config.get("validation", {})

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary for serialization."""
        return {
            "version": self.version,
            "hash": self.hash,
            "total_dimension": self.total_dimension,
            "features": [
                {
                    "name": f.name,
                    "order": f.order,
                    "category": f.category,
                    "normalization": {
                        "method": f.normalization.method,
                        "mean": f.normalization.mean,
                        "std": f.normalization.std,
                        "clip": list(f.normalization.clip),
                    }
                }
                for f in self.get_all_features()
            ]
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

@lru_cache(maxsize=1)
def get_registry() -> FeatureRegistry:
    """Get the singleton feature registry instance."""
    return FeatureRegistry.load()


def get_feature_order() -> List[str]:
    """Get the canonical feature order."""
    return get_registry().get_feature_order()


def get_feature_hash() -> str:
    """Get the hash of current feature definitions."""
    return get_registry().hash
