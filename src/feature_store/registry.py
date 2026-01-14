"""
Feature Registry
=================
Central registry for feature set versions and their specifications.

Provides:
- Version management for feature sets (V1, current, etc.)
- Feature lookup by name and version
- Normalization stats management
- Feature set validation

SOLID Principles:
- Single Responsibility: Registry only manages feature metadata
- Open/Closed: New versions via registration, not modification
- Dependency Inversion: Depends on FeatureSpec abstraction

Design Pattern: Registry Pattern + Singleton

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import (
    FeatureVersion,
    FeatureSpec,
    FeatureSetSpec,
    FeatureCategory,
    CalculationParams,
    NormalizationParams,
    NormalizationMethod,
    NormalizationStats,
    SmoothingMethod,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CURRENT FEATURE SET SPECIFICATION
# =============================================================================

def create_feature_set() -> FeatureSetSpec:
    """
    Create the current production feature set specification (15-dimensional).

    This is the SINGLE SOURCE OF TRUTH for production features.
    Used by training, backtest, and inference pipelines.

    Returns:
        FeatureSetSpec: The current production feature set specification.
    """
    features = [
        # Price features (3)
        FeatureSpec(
            name="return_5min",
            category=FeatureCategory.PRICE,
            description="5-minute log return",
            calculation=CalculationParams(window=1, smoothing=SmoothingMethod.NONE),
            normalization=NormalizationParams(
                method=NormalizationMethod.ZSCORE,
                mean=0.0, std=0.001, clip_range=(-3.0, 3.0)
            ),
            requires=["close"],
        ),
        FeatureSpec(
            name="return_15min",
            category=FeatureCategory.PRICE,
            description="15-minute cumulative return",
            calculation=CalculationParams(window=3, smoothing=SmoothingMethod.NONE),
            normalization=NormalizationParams(
                method=NormalizationMethod.ZSCORE,
                mean=0.0, std=0.002, clip_range=(-3.0, 3.0)
            ),
            requires=["close"],
        ),
        FeatureSpec(
            name="return_1h",
            category=FeatureCategory.PRICE,
            description="1-hour cumulative return",
            calculation=CalculationParams(window=12, smoothing=SmoothingMethod.NONE),
            normalization=NormalizationParams(
                method=NormalizationMethod.ZSCORE,
                mean=0.0, std=0.005, clip_range=(-3.0, 3.0)
            ),
            requires=["close"],
        ),

        # Momentum features (3)
        FeatureSpec(
            name="rsi_9",
            category=FeatureCategory.MOMENTUM,
            description="9-period RSI normalized to 0-1",
            calculation=CalculationParams(window=9, smoothing=SmoothingMethod.WILDER),
            normalization=NormalizationParams(
                method=NormalizationMethod.NONE,  # Already 0-1
                mean=0.5, std=0.2, clip_range=(0.0, 1.0)
            ),
            requires=["close"],
        ),
        FeatureSpec(
            name="adx_14",
            category=FeatureCategory.MOMENTUM,
            description="14-period ADX normalized to 0-1",
            calculation=CalculationParams(window=14, smoothing=SmoothingMethod.WILDER),
            normalization=NormalizationParams(
                method=NormalizationMethod.NONE,  # Already 0-1
                mean=0.25, std=0.15, clip_range=(0.0, 1.0)
            ),
            requires=["high", "low", "close"],
        ),
        FeatureSpec(
            name="macd_signal",
            category=FeatureCategory.MOMENTUM,
            description="MACD signal line normalized to price",
            calculation=CalculationParams(window=9, smoothing=SmoothingMethod.EMA),
            normalization=NormalizationParams(
                method=NormalizationMethod.ZSCORE,
                mean=0.0, std=0.001, clip_range=(-3.0, 3.0)
            ),
            requires=["close"],
        ),

        # Volatility features (3)
        FeatureSpec(
            name="atr_pct",
            category=FeatureCategory.VOLATILITY,
            description="ATR as percentage of price",
            calculation=CalculationParams(window=14, smoothing=SmoothingMethod.WILDER),
            normalization=NormalizationParams(
                method=NormalizationMethod.ZSCORE,
                mean=0.1, std=0.05, clip_range=(-3.0, 3.0)
            ),
            requires=["high", "low", "close"],
        ),
        FeatureSpec(
            name="bollinger_width",
            category=FeatureCategory.VOLATILITY,
            description="Bollinger band width as percentage",
            calculation=CalculationParams(window=20, smoothing=SmoothingMethod.SMA),
            normalization=NormalizationParams(
                method=NormalizationMethod.ZSCORE,
                mean=0.5, std=0.2, clip_range=(-3.0, 3.0)
            ),
            requires=["close"],
        ),
        FeatureSpec(
            name="volatility_ratio",
            category=FeatureCategory.VOLATILITY,
            description="Short-term to long-term volatility ratio",
            calculation=CalculationParams(window=5, smoothing=SmoothingMethod.NONE),
            normalization=NormalizationParams(
                method=NormalizationMethod.ZSCORE,
                mean=1.0, std=0.3, clip_range=(-3.0, 3.0)
            ),
            requires=["close"],
        ),

        # Trend features (2)
        FeatureSpec(
            name="ema_distance_20",
            category=FeatureCategory.TREND,
            description="Distance from 20-period EMA as percentage",
            calculation=CalculationParams(window=20, smoothing=SmoothingMethod.EMA),
            normalization=NormalizationParams(
                method=NormalizationMethod.ZSCORE,
                mean=0.0, std=0.5, clip_range=(-3.0, 3.0)
            ),
            requires=["close"],
        ),
        FeatureSpec(
            name="price_position",
            category=FeatureCategory.TREND,
            description="Price position in 20-period range (0-1)",
            calculation=CalculationParams(window=20, smoothing=SmoothingMethod.NONE),
            normalization=NormalizationParams(
                method=NormalizationMethod.NONE,  # Already 0-1
                mean=0.5, std=0.25, clip_range=(0.0, 1.0)
            ),
            requires=["high", "low", "close"],
        ),

        # Macro features (4)
        FeatureSpec(
            name="dxy_z",
            category=FeatureCategory.MACRO,
            description="Dollar index rolling z-score",
            calculation=CalculationParams(window=60, smoothing=SmoothingMethod.NONE),
            normalization=NormalizationParams(
                method=NormalizationMethod.ROLLING_ZSCORE,
                rolling_window=60, clip_range=(-3.0, 3.0)
            ),
            requires=["dxy"],
        ),
        FeatureSpec(
            name="vix_z",
            category=FeatureCategory.MACRO,
            description="VIX rolling z-score",
            calculation=CalculationParams(window=60, smoothing=SmoothingMethod.NONE),
            normalization=NormalizationParams(
                method=NormalizationMethod.ROLLING_ZSCORE,
                rolling_window=60, clip_range=(-3.0, 3.0)
            ),
            requires=["vix"],
        ),
        FeatureSpec(
            name="wti_z",
            category=FeatureCategory.MACRO,
            description="WTI oil rolling z-score",
            calculation=CalculationParams(window=60, smoothing=SmoothingMethod.NONE),
            normalization=NormalizationParams(
                method=NormalizationMethod.ROLLING_ZSCORE,
                rolling_window=60, clip_range=(-3.0, 3.0)
            ),
            requires=["wti"],
        ),
        FeatureSpec(
            name="embi_z",
            category=FeatureCategory.MACRO,
            description="EMBI spread rolling z-score",
            calculation=CalculationParams(window=60, smoothing=SmoothingMethod.NONE),
            normalization=NormalizationParams(
                method=NormalizationMethod.ROLLING_ZSCORE,
                rolling_window=60, clip_range=(-3.0, 3.0)
            ),
            requires=["embi"],
        ),
    ]

    return FeatureSetSpec(
        version=FeatureVersion.CURRENT,
        features=features,
        dimension=15,
        description="Current feature set with 15 dimensions optimized for USD/COP trading",
    )


# =============================================================================
# FEATURE REGISTRY
# =============================================================================

class FeatureRegistry:
    """
    Central registry for feature sets.

    Singleton pattern ensures consistent access across all pipelines.

    Usage:
        registry = FeatureRegistry.instance()
        current_spec = registry.get_feature_set(FeatureVersion.CURRENT)
        rsi_spec = registry.get_feature("rsi_9", FeatureVersion.CURRENT)
    """
    _instance: Optional["FeatureRegistry"] = None
    _feature_sets: Dict[FeatureVersion, FeatureSetSpec] = {}
    _normalization_stats: Dict[FeatureVersion, NormalizationStats] = {}

    def __new__(cls) -> "FeatureRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    @classmethod
    def instance(cls) -> "FeatureRegistry":
        """Get singleton instance"""
        return cls()

    def _initialize(self) -> None:
        """Initialize with default feature sets"""
        # Register current production feature set
        current_set = create_feature_set()
        self.register_feature_set(current_set)

        logger.info(f"Initialized FeatureRegistry with {len(self._feature_sets)} versions")

    def register_feature_set(self, spec: FeatureSetSpec) -> None:
        """Register a feature set version"""
        self._feature_sets[spec.version] = spec
        logger.info(f"Registered feature set: {spec.version.value} ({spec.dimension} dims)")

    def get_feature_set(self, version: FeatureVersion) -> FeatureSetSpec:
        """Get feature set specification by version"""
        if version not in self._feature_sets:
            raise KeyError(f"Unknown feature version: {version}")
        return self._feature_sets[version]

    def get_feature(self, name: str, version: FeatureVersion) -> FeatureSpec:
        """Get single feature specification"""
        feature_set = self.get_feature_set(version)
        feature = feature_set.get_feature(name)
        if feature is None:
            raise KeyError(f"Unknown feature: {name} in {version.value}")
        return feature

    def get_feature_names(self, version: FeatureVersion) -> List[str]:
        """Get ordered list of feature names"""
        return self.get_feature_set(version).get_feature_names()

    def get_dimension(self, version: FeatureVersion) -> int:
        """Get feature dimension for version"""
        return self.get_feature_set(version).dimension

    # =========================================================================
    # NORMALIZATION STATS MANAGEMENT
    # =========================================================================

    def load_normalization_stats(
        self,
        version: FeatureVersion,
        path: Optional[Path] = None
    ) -> NormalizationStats:
        """
        Load normalization stats from file or config.

        Args:
            version: Feature version
            path: Optional path to JSON stats file

        Returns:
            NormalizationStats object
        """
        if path and path.exists():
            with open(path) as f:
                data = json.load(f)
            stats = NormalizationStats.from_dict(data)
            self._normalization_stats[version] = stats
            logger.info(f"Loaded normalization stats for {version.value} from {path}")
            return stats

        # Fall back to default stats from feature specs
        feature_set = self.get_feature_set(version)
        stats_dict = {}
        for feature in feature_set.features:
            stats_dict[feature.name] = feature.normalization

        stats = NormalizationStats(
            version=version,
            stats=stats_dict,
            sample_size=0,  # Default stats
        )
        self._normalization_stats[version] = stats
        return stats

    def get_normalization_stats(self, version: FeatureVersion) -> NormalizationStats:
        """Get normalization stats for version"""
        if version not in self._normalization_stats:
            # Load default
            return self.load_normalization_stats(version)
        return self._normalization_stats[version]

    def save_normalization_stats(
        self,
        stats: NormalizationStats,
        path: Path
    ) -> None:
        """Save normalization stats to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        logger.info(f"Saved normalization stats for {stats.version.value} to {path}")

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_feature_vector(
        self,
        values: Dict[str, float],
        version: FeatureVersion
    ) -> List[str]:
        """
        Validate feature vector against specification.

        Returns list of validation errors (empty if valid).
        """
        errors = []
        feature_set = self.get_feature_set(version)
        expected_names = set(feature_set.get_feature_names())
        actual_names = set(values.keys())

        # Check for missing features
        missing = expected_names - actual_names
        if missing:
            errors.append(f"Missing features: {missing}")

        # Check for extra features
        extra = actual_names - expected_names
        if extra:
            errors.append(f"Unexpected features: {extra}")

        # Check dimension
        if len(values) != feature_set.dimension:
            errors.append(
                f"Wrong dimension: expected {feature_set.dimension}, got {len(values)}"
            )

        # Check individual values
        for name, value in values.items():
            if name in expected_names:
                feature = feature_set.get_feature(name)
                if feature.valid_range:
                    low, high = feature.valid_range
                    if value < low or value > high:
                        errors.append(
                            f"Feature {name} out of range: {value} not in [{low}, {high}]"
                        )

        return errors

    def list_versions(self) -> List[FeatureVersion]:
        """List all registered versions"""
        return list(self._feature_sets.keys())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_registry() -> FeatureRegistry:
    """Get the singleton feature registry"""
    return FeatureRegistry.instance()


def get_current_spec() -> FeatureSetSpec:
    """
    Get the current production feature set specification.

    Returns:
        FeatureSetSpec: The current production feature set.
    """
    return get_registry().get_feature_set(FeatureVersion.CURRENT)


def get_feature_names() -> List[str]:
    """
    Get ordered feature names for the current production feature set.

    Returns:
        List[str]: Ordered list of feature names (15 features).
    """
    return get_registry().get_feature_names(FeatureVersion.CURRENT)
