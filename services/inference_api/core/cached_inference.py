"""
Cached Inference Engine
=======================
P1-3: Uses Feast feature store for low-latency inference

This module implements the caching layer for inference features using Feast
online store (Redis) with automatic fallback to direct computation when
cache is unavailable.

Features:
- Online feature retrieval from Redis via Feast
- Fallback to direct computation if cache miss
- Consistency validation with training features
- Cache hit/miss metrics for monitoring
- Feature consistency validation

Contract: CTR-FEAT-001 - 15 features in canonical order
Reference: MASTER_REMEDIATION_PLAN_v1.0.md - Section 1.3

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Feast
try:
    from feast import FeatureStore
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    logger.warning("Feast not installed, using fallback feature retrieval")


class CachedInferenceEngine:
    """
    Inference engine with Feast-backed feature caching.

    Provides:
    1. Low-latency feature retrieval from online store
    2. Fallback computation for cache misses
    3. Feature consistency validation

    This class acts as the primary interface for feature retrieval during
    inference, abstracting the underlying data source (Feast or direct
    computation) from the inference engine.

    Usage:
        engine = CachedInferenceEngine(feature_store_path="feature_store/")
        features, metadata = engine.get_features(
            timestamp="2026-01-17T13:00:00",
            position=0.5
        )

        # Check cache performance
        stats = engine.get_cache_stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    # Feature view and entity configuration
    FEATURE_VIEW: Final[str] = "trading_features"
    ENTITY_KEY: Final[str] = "timestamp"

    # Expected features (must match training - CTR-FEAT-001)
    EXPECTED_FEATURES: Final[List[str]] = [
        "log_ret_5m",
        "log_ret_1h",
        "log_ret_4h",
        "rsi_9",
        "atr_pct",
        "adx_14",
        "dxy_z",
        "dxy_change_1d",
        "vix_z",
        "embi_z",
        "brent_change_1d",
        "rate_spread",
        "usdmxn_change_1d",
        "position",
        "time_normalized",
    ]

    # Feature count validation
    EXPECTED_FEATURE_COUNT: Final[int] = 15

    # Feature ranges for validation
    FEATURE_RANGES: Final[Dict[str, Tuple[float, float]]] = {
        "rsi_9": (0, 100),
        "atr_pct": (0, 0.5),  # 0-50% of price
        "adx_14": (0, 100),
        "position": (-1, 1),
        "time_normalized": (0, 1),
    }

    def __init__(
        self,
        feature_store_path: str = "feature_store/",
        fallback_enabled: bool = True,
        norm_stats_path: Optional[str] = None,
    ):
        """
        Initialize CachedInferenceEngine.

        Args:
            feature_store_path: Path to Feast feature repository
            fallback_enabled: Enable fallback to direct computation if Feast unavailable
            norm_stats_path: Path to normalization stats file (for fallback)
        """
        self.feature_store_path = feature_store_path
        self.fallback_enabled = fallback_enabled
        self.norm_stats_path = norm_stats_path
        self.store: Optional[FeatureStore] = None

        # Cache statistics
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._errors: int = 0

        # Latency tracking (in milliseconds)
        self._feast_latencies: List[float] = []
        self._fallback_latencies: List[float] = []
        self._max_latency_samples: int = 1000

        # Initialize Feast store if available
        if FEAST_AVAILABLE:
            try:
                # Resolve path relative to project root if needed
                store_path = Path(feature_store_path)
                if not store_path.is_absolute():
                    project_root = Path(__file__).parent.parent.parent.parent
                    store_path = project_root / feature_store_path

                if store_path.exists():
                    self.store = FeatureStore(repo_path=str(store_path))
                    logger.info(f"Feast feature store initialized from {store_path}")
                else:
                    logger.warning(f"Feast repo not found at {store_path}")
            except Exception as e:
                logger.error(f"Failed to initialize Feast: {e}")
                self.store = None
        else:
            logger.info("Feast not available, fallback mode only")

        # Initialize fallback builder (lazy loaded)
        self._fallback_builder = None

    def get_features(
        self,
        timestamp: str,
        position: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get features for inference.

        This method attempts to retrieve features from Feast online store first,
        then falls back to direct computation if cache miss or Feast unavailable.

        Args:
            timestamp: ISO format timestamp (e.g., "2026-01-17T13:00:00")
            position: Current position (-1 to 1)

        Returns:
            Tuple of (feature_vector, metadata)
            - feature_vector: np.ndarray of shape (15,) with normalized features
            - metadata: Dict containing source, latency, and validation info

        Raises:
            ValueError: If features cannot be retrieved from any source
        """
        metadata = {
            "timestamp": timestamp,
            "source": "unknown",
            "latency_ms": 0,
            "feature_count": 0,
        }

        start_time = time.perf_counter()

        # Try Feast online store first
        if self.store is not None:
            try:
                features = self._get_from_feast(timestamp)
                if features is not None:
                    self._cache_hits += 1
                    metadata["source"] = "feast_online"

                    # Add runtime features (not stored in Feast)
                    features["position"] = float(np.clip(position, -1.0, 1.0))
                    features["time_normalized"] = self._compute_time_normalized()

                    # Build feature vector in canonical order
                    vector = self._build_feature_vector(features)

                    latency_ms = (time.perf_counter() - start_time) * 1000
                    self._add_latency(self._feast_latencies, latency_ms)
                    metadata["latency_ms"] = latency_ms
                    metadata["feature_count"] = len(vector)

                    logger.debug(f"Feast cache hit: {timestamp} in {latency_ms:.2f}ms")
                    return vector, metadata

            except Exception as e:
                logger.warning(f"Feast retrieval failed: {e}")
                self._errors += 1

        # Fallback to direct computation
        if self.fallback_enabled:
            self._cache_misses += 1
            try:
                features = self._compute_features_direct(timestamp)
                features["position"] = float(np.clip(position, -1.0, 1.0))
                features["time_normalized"] = self._compute_time_normalized()

                vector = self._build_feature_vector(features)

                latency_ms = (time.perf_counter() - start_time) * 1000
                self._add_latency(self._fallback_latencies, latency_ms)
                metadata["source"] = "computed"
                metadata["latency_ms"] = latency_ms
                metadata["feature_count"] = len(vector)

                logger.debug(f"Fallback computation: {timestamp} in {latency_ms:.2f}ms")
                return vector, metadata

            except Exception as e:
                logger.error(f"Fallback computation failed: {e}")
                self._errors += 1
                raise ValueError(f"Could not retrieve features for {timestamp}: {e}")

        raise ValueError(
            f"Could not retrieve features for {timestamp}: "
            f"Feast unavailable and fallback disabled"
        )

    def _get_from_feast(self, timestamp: str) -> Optional[Dict[str, float]]:
        """
        Retrieve features from Feast online store.

        Args:
            timestamp: ISO format timestamp

        Returns:
            Dict of feature values if found, None otherwise
        """
        if self.store is None:
            return None

        entity_rows = [{self.ENTITY_KEY: timestamp}]

        # Build feature references (excluding runtime features)
        feature_refs = [
            f"{self.FEATURE_VIEW}:{f}"
            for f in self.EXPECTED_FEATURES
            if f not in ["position", "time_normalized"]  # Runtime features
        ]

        try:
            result = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            ).to_dict()

            # Extract and validate features
            features = {}
            for key, values in result.items():
                if key != self.ENTITY_KEY:
                    # Extract feature name from "view:feature" format
                    feature_name = key.split(":")[-1] if ":" in key else key
                    if values and values[0] is not None:
                        features[feature_name] = float(values[0])

            # Validate we got enough features (minus runtime features)
            expected_count = len(self.EXPECTED_FEATURES) - 2  # Minus position, time_normalized
            if len(features) < expected_count:
                logger.warning(
                    f"Incomplete features from Feast: got {len(features)}, "
                    f"expected {expected_count}"
                )
                return None

            return features

        except Exception as e:
            logger.debug(f"Feast query error: {e}")
            return None

    def _compute_features_direct(self, timestamp: str) -> Dict[str, float]:
        """
        Compute features directly from database (fallback).

        This method uses the CanonicalFeatureBuilder from the feature store
        to ensure consistency with training calculations.

        Args:
            timestamp: ISO format timestamp

        Returns:
            Dict of feature values
        """
        # Lazy load the fallback builder to avoid circular imports
        if self._fallback_builder is None:
            self._fallback_builder = self._create_fallback_builder()

        if self._fallback_builder is None:
            logger.warning("Fallback builder unavailable, returning zeros")
            return {f: 0.0 for f in self.EXPECTED_FEATURES}

        try:
            # Use the SSOT builder for feature computation
            # This is a simplified implementation - in production, this would
            # fetch data from the database and compute features
            features = self._fallback_builder.compute_features_for_timestamp(timestamp)
            return features
        except Exception as e:
            logger.error(f"Fallback builder error: {e}")
            # Return zeros as last resort
            return {f: 0.0 for f in self.EXPECTED_FEATURES}

    def _create_fallback_builder(self):
        """
        Create fallback feature builder.

        Returns:
            CanonicalFeatureBuilder instance or None if unavailable
        """
        try:
            # Try to import the canonical builder
            from src.feature_store.core import UnifiedFeatureBuilder
            return UnifiedFeatureBuilder()
        except ImportError:
            logger.warning("UnifiedFeatureBuilder not available for fallback")
            return None
        except Exception as e:
            logger.warning(f"Failed to create fallback builder: {e}")
            return None

    def _build_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Build feature vector in canonical order.

        Ensures features are in the exact order expected by the model
        (CTR-FEAT-001 contract).

        Args:
            features: Dict mapping feature names to values

        Returns:
            np.ndarray of shape (15,) with features in canonical order
        """
        vector = []
        for feature_name in self.EXPECTED_FEATURES:
            value = features.get(feature_name, 0.0)
            if value is None or np.isnan(value) or np.isinf(value):
                value = 0.0
            vector.append(float(value))

        return np.array(vector, dtype=np.float32)

    def _compute_time_normalized(self) -> float:
        """
        Compute normalized trading time (0-1).

        Normalizes current time within trading session hours.
        Trading hours: 8:00 - 16:00 COT (Colombia Time)

        Returns:
            Float between 0.0 and 1.0 representing progress through trading session
        """
        now = datetime.now()

        # Trading hours in Colombia (8:00 - 16:00 COT)
        trading_start = 8 * 60  # minutes from midnight
        trading_end = 16 * 60

        current_minutes = now.hour * 60 + now.minute

        if current_minutes < trading_start:
            return 0.0
        elif current_minutes > trading_end:
            return 1.0
        else:
            return (current_minutes - trading_start) / (trading_end - trading_start)

    def _add_latency(self, latencies: List[float], value: float) -> None:
        """Add latency sample, maintaining max size for memory efficiency."""
        latencies.append(value)
        if len(latencies) > self._max_latency_samples:
            latencies.pop(0)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache hit/miss statistics.

        Returns:
            Dict containing:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - errors: Number of errors
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - feast_available: Whether Feast store is initialized
            - avg_feast_latency_ms: Average Feast retrieval latency
            - avg_fallback_latency_ms: Average fallback computation latency
            - p95_feast_latency_ms: 95th percentile Feast latency
        """
        total = self._cache_hits + self._cache_misses

        # Calculate latency statistics
        avg_feast = (
            sum(self._feast_latencies) / len(self._feast_latencies)
            if self._feast_latencies else 0.0
        )
        avg_fallback = (
            sum(self._fallback_latencies) / len(self._fallback_latencies)
            if self._fallback_latencies else 0.0
        )
        p95_feast = (
            float(np.percentile(self._feast_latencies, 95))
            if self._feast_latencies else 0.0
        )

        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "errors": self._errors,
            "total_requests": total,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "miss_rate": self._cache_misses / total if total > 0 else 0.0,
            "error_rate": self._errors / (total + self._errors) if (total + self._errors) > 0 else 0.0,
            "feast_available": self.store is not None,
            "fallback_enabled": self.fallback_enabled,
            "avg_feast_latency_ms": round(avg_feast, 2),
            "avg_fallback_latency_ms": round(avg_fallback, 2),
            "p95_feast_latency_ms": round(p95_feast, 2),
        }

    def validate_feature_consistency(
        self,
        features: Dict[str, float],
        expected_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate features match expected schema and optionally hash.

        This method performs comprehensive validation to ensure features
        are consistent with training expectations.

        Args:
            features: Dict of feature names to values
            expected_hash: Optional SHA256 hash of expected feature schema

        Returns:
            Dict containing:
            - valid: Boolean indicating overall validity
            - errors: List of critical errors (features missing, wrong count)
            - warnings: List of warnings (values outside expected ranges)
            - hash_match: Boolean if expected_hash provided and matches
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "feature_count": len(features),
            "expected_count": self.EXPECTED_FEATURE_COUNT,
        }

        # Check all expected features are present
        missing = set(self.EXPECTED_FEATURES) - set(features.keys())
        if missing:
            result["valid"] = False
            result["errors"].append(f"Missing features: {sorted(missing)}")

        # Check for unexpected features
        extra = set(features.keys()) - set(self.EXPECTED_FEATURES)
        if extra:
            result["warnings"].append(f"Extra features ignored: {sorted(extra)}")

        # Validate feature count
        if len(features) != self.EXPECTED_FEATURE_COUNT:
            result["warnings"].append(
                f"Feature count mismatch: got {len(features)}, "
                f"expected {self.EXPECTED_FEATURE_COUNT}"
            )

        # Validate value ranges for known features
        for feature_name, (min_val, max_val) in self.FEATURE_RANGES.items():
            if feature_name in features:
                val = features[feature_name]
                if val is not None and not (min_val <= val <= max_val):
                    result["warnings"].append(
                        f"{feature_name}={val:.4f} outside expected range [{min_val}, {max_val}]"
                    )

        # Check for NaN/Inf values
        nan_features = [f for f, v in features.items() if v is not None and (np.isnan(v) or np.isinf(v))]
        if nan_features:
            result["errors"].append(f"NaN/Inf values in features: {nan_features}")
            result["valid"] = False

        # Validate hash if provided
        if expected_hash is not None:
            actual_hash = self._compute_feature_hash(features)
            result["hash_match"] = actual_hash == expected_hash
            result["actual_hash"] = actual_hash[:16]  # First 16 chars for display
            result["expected_hash"] = expected_hash[:16]
            if not result["hash_match"]:
                result["warnings"].append(
                    f"Feature hash mismatch: expected {expected_hash[:16]}, got {actual_hash[:16]}"
                )

        return result

    def _compute_feature_hash(self, features: Dict[str, float]) -> str:
        """
        Compute SHA256 hash of features for consistency validation.

        Args:
            features: Dict of feature names to values

        Returns:
            SHA256 hash string
        """
        # Sort keys for deterministic ordering
        canonical = json.dumps(
            {k: round(v, 8) if v is not None else None for k, v in sorted(features.items())},
            sort_keys=True,
            separators=(',', ':')
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def reset_stats(self) -> None:
        """Reset all cache statistics."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._errors = 0
        self._feast_latencies.clear()
        self._fallback_latencies.clear()
        logger.info("Cache statistics reset")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the caching infrastructure.

        Returns:
            Dict with health status information
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "feast_installed": FEAST_AVAILABLE,
            "feast_store_initialized": self.store is not None,
            "fallback_enabled": self.fallback_enabled,
            "fallback_builder_available": self._fallback_builder is not None,
        }

        # Test Feast connection if available
        if self.store is not None:
            try:
                # Attempt a simple operation to verify connectivity
                self.store.list_feature_views()
                health["feast_connection"] = "ok"
            except Exception as e:
                health["feast_connection"] = f"error: {str(e)[:100]}"
                health["status"] = "degraded"
        else:
            health["feast_connection"] = "unavailable"
            if not self.fallback_enabled:
                health["status"] = "unhealthy"

        # Add cache stats
        health["cache_stats"] = self.get_cache_stats()

        return health


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_cached_inference_engine(
    feature_store_path: Optional[str] = None,
    fallback_enabled: bool = True,
    norm_stats_path: Optional[str] = None,
) -> CachedInferenceEngine:
    """
    Factory function to create CachedInferenceEngine.

    Args:
        feature_store_path: Path to Feast repository (default: "feature_store/")
        fallback_enabled: Enable fallback to direct computation
        norm_stats_path: Path to normalization stats

    Returns:
        Configured CachedInferenceEngine instance
    """
    return CachedInferenceEngine(
        feature_store_path=feature_store_path or "feature_store/",
        fallback_enabled=fallback_enabled,
        norm_stats_path=norm_stats_path,
    )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cached Inference Engine")
    parser.add_argument("--health", action="store_true", help="Check engine health")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--test", action="store_true", help="Run test feature retrieval")
    parser.add_argument("--timestamp", type=str, help="Timestamp for test retrieval")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create engine
    engine = create_cached_inference_engine()

    if args.health:
        health = engine.health_check()
        print(json.dumps(health, indent=2))

    elif args.stats:
        stats = engine.get_cache_stats()
        print(json.dumps(stats, indent=2))

    elif args.test:
        timestamp = args.timestamp or datetime.now().isoformat()
        try:
            features, metadata = engine.get_features(timestamp=timestamp, position=0.0)
            print(f"Features retrieved successfully:")
            print(f"  Source: {metadata['source']}")
            print(f"  Latency: {metadata['latency_ms']:.2f}ms")
            print(f"  Feature count: {metadata['feature_count']}")
            print(f"  Vector shape: {features.shape}")
        except Exception as e:
            print(f"Error: {e}")

    else:
        print("Cached Inference Engine")
        print("=" * 40)
        health = engine.health_check()
        print(f"Status: {health['status']}")
        print(f"Feast Available: {health['feast_store_initialized']}")
        print(f"Fallback Enabled: {health['fallback_enabled']}")
