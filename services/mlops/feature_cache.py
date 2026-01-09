"""
Feature Cache Manager
=====================

High-performance feature caching using Redis for real-time inference.
Ensures consistency between training and serving (feature store lite).

Features:
- TTL-based caching
- Batch feature retrieval
- Feature versioning
- Cache statistics
"""

import os
import time
import logging
from typing import Optional, Dict, List, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

import numpy as np

try:
    import redis
except ImportError:
    redis = None

from .config import MLOpsConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class CachedFeatures:
    """Container for cached feature data."""
    timestamp: str
    features: Dict[str, float]
    feature_vector: List[float]
    source: str
    ttl_remaining: Optional[int] = None

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for inference."""
        return np.array(self.feature_vector, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "features": self.features,
            "feature_vector": self.feature_vector,
            "source": self.source,
            "ttl_remaining": self.ttl_remaining,
        }


class FeatureCache:
    """
    Redis-based feature cache for real-time inference.

    Provides a lightweight feature store implementation for
    caching computed features with TTL support.

    Usage:
        cache = FeatureCache()

        # Store features
        cache.set_features(
            key="2024-01-15T10:30:00",
            features={"rsi": 45.2, "macd": 0.003, ...},
            feature_order=["rsi", "macd", ...]
        )

        # Retrieve for inference
        cached = cache.get_features("2024-01-15T10:30:00")
        prediction = model.predict(cached.to_numpy())
    """

    # Key prefixes
    KEY_FEATURES = "features"
    KEY_LATEST = "features:latest"
    KEY_HISTORY = "features:history"
    KEY_STATS = "features:stats"

    # Default feature order for USDCOP model
    DEFAULT_FEATURE_ORDER = [
        # Price features
        "returns_1m", "returns_5m", "returns_15m", "returns_30m", "returns_1h",
        "log_returns_5m", "price_momentum", "price_acceleration",

        # Volatility
        "volatility_5m", "volatility_15m", "volatility_1h",
        "atr_14", "bollinger_width",

        # Technical indicators
        "rsi_14", "rsi_7", "macd", "macd_signal", "macd_histogram",
        "stochastic_k", "stochastic_d", "cci_20", "williams_r",

        # Moving averages
        "sma_ratio_20", "sma_ratio_50", "ema_ratio_12", "ema_ratio_26",
        "ma_crossover",

        # Volume
        "volume_ratio", "volume_momentum", "obv_change",

        # Market regime
        "trend_strength", "regime_indicator",

        # Macro indicators
        "dxy_change", "brent_change", "vix_level",
        "usdcop_spread", "rate_differential",

        # Time features
        "hour_sin", "hour_cos", "day_of_week", "is_morning_session",
    ]

    def __init__(
        self,
        config: Optional[MLOpsConfig] = None,
        redis_client: Optional[redis.Redis] = None,
        feature_order: Optional[List[str]] = None,
        default_ttl: int = 300,  # 5 minutes
    ):
        """
        Initialize feature cache.

        Args:
            config: MLOps configuration
            redis_client: Optional Redis client
            feature_order: Ordered list of feature names
            default_ttl: Default TTL in seconds
        """
        self.config = config or get_config()
        self.feature_order = feature_order or self.DEFAULT_FEATURE_ORDER
        self.default_ttl = default_ttl

        # Initialize Redis
        if redis_client:
            self.redis = redis_client
        elif redis:
            try:
                self.redis = redis.Redis(
                    host=self.config.redis.host,
                    port=self.config.redis.port,
                    db=self.config.redis.db,
                    password=self.config.redis.password,
                    decode_responses=True,
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis = None
        else:
            self.redis = None
            logger.warning("Redis not available. Feature cache disabled.")

        # In-memory fallback
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_key(self, timestamp: str) -> str:
        """Get Redis key for timestamp."""
        return f"{self.KEY_FEATURES}:{timestamp}"

    def set_features(
        self,
        timestamp: str,
        features: Dict[str, float],
        ttl: Optional[int] = None,
        source: str = "pipeline"
    ) -> bool:
        """
        Store features in cache.

        Args:
            timestamp: Feature timestamp (key)
            features: Dictionary of feature name -> value
            ttl: Time-to-live in seconds
            source: Source of features (pipeline, realtime, etc.)

        Returns:
            True if stored successfully
        """
        ttl = ttl or self.default_ttl

        # Build feature vector in correct order
        feature_vector = []
        for name in self.feature_order:
            if name in features:
                feature_vector.append(float(features[name]))
            else:
                feature_vector.append(0.0)  # Default for missing features
                logger.warning(f"Missing feature: {name}")

        data = {
            "timestamp": timestamp,
            "features": features,
            "feature_vector": feature_vector,
            "source": source,
            "cached_at": datetime.now().isoformat(),
        }

        if self.redis:
            try:
                key = self._get_key(timestamp)
                self.redis.setex(key, ttl, json.dumps(data))

                # Update latest pointer
                self.redis.set(self.KEY_LATEST, timestamp)

                # Add to history
                self.redis.lpush(self.KEY_HISTORY, timestamp)
                self.redis.ltrim(self.KEY_HISTORY, 0, 999)

                # Update stats
                self.redis.hincrby(self.KEY_STATS, "sets", 1)

                return True

            except Exception as e:
                logger.error(f"Failed to cache features: {e}")
                return False
        else:
            # Memory fallback
            self._memory_cache[timestamp] = data
            return True

    def get_features(self, timestamp: str) -> Optional[CachedFeatures]:
        """
        Retrieve features from cache.

        Args:
            timestamp: Feature timestamp (key)

        Returns:
            CachedFeatures if found, None otherwise
        """
        if self.redis:
            try:
                key = self._get_key(timestamp)
                data = self.redis.get(key)

                if data:
                    self._cache_hits += 1
                    self.redis.hincrby(self.KEY_STATS, "hits", 1)

                    parsed = json.loads(data)
                    ttl = self.redis.ttl(key)

                    return CachedFeatures(
                        timestamp=parsed["timestamp"],
                        features=parsed["features"],
                        feature_vector=parsed["feature_vector"],
                        source=parsed.get("source", "unknown"),
                        ttl_remaining=ttl if ttl > 0 else None,
                    )
                else:
                    self._cache_misses += 1
                    self.redis.hincrby(self.KEY_STATS, "misses", 1)
                    return None

            except Exception as e:
                logger.error(f"Failed to get features: {e}")
                return None
        else:
            # Memory fallback
            if timestamp in self._memory_cache:
                self._cache_hits += 1
                data = self._memory_cache[timestamp]
                return CachedFeatures(
                    timestamp=data["timestamp"],
                    features=data["features"],
                    feature_vector=data["feature_vector"],
                    source=data.get("source", "memory"),
                )
            self._cache_misses += 1
            return None

    def get_latest(self) -> Optional[CachedFeatures]:
        """Get the most recent cached features."""
        if self.redis:
            try:
                latest_ts = self.redis.get(self.KEY_LATEST)
                if latest_ts:
                    return self.get_features(latest_ts)
            except Exception as e:
                logger.error(f"Failed to get latest features: {e}")

        # Memory fallback
        if self._memory_cache:
            latest_ts = max(self._memory_cache.keys())
            return self.get_features(latest_ts)

        return None

    def get_batch(self, timestamps: List[str]) -> Dict[str, Optional[CachedFeatures]]:
        """
        Retrieve multiple feature sets.

        Args:
            timestamps: List of timestamps to retrieve

        Returns:
            Dictionary of timestamp -> CachedFeatures
        """
        results = {}

        if self.redis:
            try:
                pipe = self.redis.pipeline()
                keys = [self._get_key(ts) for ts in timestamps]

                for key in keys:
                    pipe.get(key)

                values = pipe.execute()

                for ts, value in zip(timestamps, values):
                    if value:
                        parsed = json.loads(value)
                        results[ts] = CachedFeatures(
                            timestamp=parsed["timestamp"],
                            features=parsed["features"],
                            feature_vector=parsed["feature_vector"],
                            source=parsed.get("source", "unknown"),
                        )
                    else:
                        results[ts] = None

            except Exception as e:
                logger.error(f"Failed to get batch: {e}")
                for ts in timestamps:
                    results[ts] = self.get_features(ts)
        else:
            for ts in timestamps:
                results[ts] = self.get_features(ts)

        return results

    def delete(self, timestamp: str) -> bool:
        """Delete cached features."""
        if self.redis:
            try:
                key = self._get_key(timestamp)
                return self.redis.delete(key) > 0
            except Exception as e:
                logger.error(f"Failed to delete: {e}")
                return False
        else:
            if timestamp in self._memory_cache:
                del self._memory_cache[timestamp]
                return True
            return False

    def clear(self) -> int:
        """Clear all cached features."""
        count = 0

        if self.redis:
            try:
                # Get all feature keys
                cursor = 0
                keys_to_delete = []

                while True:
                    cursor, keys = self.redis.scan(cursor, f"{self.KEY_FEATURES}:*")
                    keys_to_delete.extend(keys)
                    if cursor == 0:
                        break

                if keys_to_delete:
                    count = self.redis.delete(*keys_to_delete)

                # Clear related keys
                self.redis.delete(self.KEY_LATEST, self.KEY_HISTORY, self.KEY_STATS)

            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
        else:
            count = len(self._memory_cache)
            self._memory_cache.clear()

        return count

    def get_history(self, limit: int = 100) -> List[str]:
        """Get recent cached timestamps."""
        if self.redis:
            try:
                return self.redis.lrange(self.KEY_HISTORY, 0, limit - 1)
            except Exception as e:
                logger.error(f"Failed to get history: {e}")
                return []
        else:
            return sorted(self._memory_cache.keys(), reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0 else 0.0,
        }

        if self.redis:
            try:
                redis_stats = self.redis.hgetall(self.KEY_STATS)
                stats["redis"] = {
                    "sets": int(redis_stats.get("sets", 0)),
                    "hits": int(redis_stats.get("hits", 0)),
                    "misses": int(redis_stats.get("misses", 0)),
                }

                # Get memory usage
                info = self.redis.info("memory")
                stats["redis_memory_mb"] = info.get("used_memory", 0) / (1024 * 1024)

            except Exception as e:
                logger.error(f"Failed to get stats: {e}")

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Check cache health."""
        health = {
            "status": "healthy",
            "redis_connected": False,
            "latest_available": False,
        }

        if self.redis:
            try:
                self.redis.ping()
                health["redis_connected"] = True

                latest = self.redis.get(self.KEY_LATEST)
                health["latest_available"] = latest is not None
                health["latest_timestamp"] = latest

            except Exception as e:
                health["status"] = "unhealthy"
                health["error"] = str(e)
        else:
            health["status"] = "degraded"
            health["message"] = "Using in-memory cache"
            health["cache_size"] = len(self._memory_cache)

        return health


# Global instance
_feature_cache: Optional[FeatureCache] = None


def get_feature_cache() -> FeatureCache:
    """Get or create global feature cache."""
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = FeatureCache()
    return _feature_cache


def initialize_feature_cache(
    config: Optional[MLOpsConfig] = None,
    feature_order: Optional[List[str]] = None
) -> FeatureCache:
    """Initialize global feature cache."""
    global _feature_cache
    _feature_cache = FeatureCache(config=config, feature_order=feature_order)
    return _feature_cache
