"""
Redis State Repository
======================

Redis implementation of IStateRepository and IHashRepository.
Used for production state persistence.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import json
import logging
from typing import Optional, Dict, Any, List

from src.core.interfaces.repository import (
    IStateRepository,
    IHashRepository,
    IListRepository,
)

logger = logging.getLogger(__name__)


class RedisStateRepository(IStateRepository, IHashRepository, IListRepository):
    """
    Redis implementation of state persistence.

    Dependency Inversion: Business logic depends on IStateRepository,
    not directly on Redis.

    Thread-safe for concurrent access.
    """

    def __init__(self, redis_client):
        """
        Args:
            redis_client: Redis client instance (injected, not created)
        """
        self._redis = redis_client

    # =========================================================================
    # IStateRepository implementation
    # =========================================================================

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state by key."""
        try:
            data = self._redis.get(key)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.error(f"Redis GET error for key '{key}': {e}")
            return None

    def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set state with optional TTL."""
        try:
            data = json.dumps(value)
            if ttl:
                self._redis.setex(key, ttl, data)
            else:
                self._redis.set(key, data)
            return True
        except Exception as e:
            logger.error(f"Redis SET error for key '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete state by key."""
        try:
            self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error for key '{key}': {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(self._redis.exists(key))
        except Exception as e:
            logger.error(f"Redis EXISTS error for key '{key}': {e}")
            return False

    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key."""
        try:
            ttl = self._redis.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.error(f"Redis TTL error for key '{key}': {e}")
            return None

    # =========================================================================
    # IHashRepository implementation
    # =========================================================================

    def hget(self, key: str, field: str) -> Optional[str]:
        """Get single field from hash."""
        try:
            return self._redis.hget(key, field)
        except Exception as e:
            logger.error(f"Redis HGET error for key '{key}', field '{field}': {e}")
            return None

    def hset(self, key: str, field: str, value: str) -> bool:
        """Set single field in hash."""
        try:
            self._redis.hset(key, field, value)
            return True
        except Exception as e:
            logger.error(f"Redis HSET error for key '{key}', field '{field}': {e}")
            return False

    def hgetall(self, key: str) -> Dict[str, str]:
        """Get all fields from hash."""
        try:
            return self._redis.hgetall(key) or {}
        except Exception as e:
            logger.error(f"Redis HGETALL error for key '{key}': {e}")
            return {}

    def hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        """Set multiple fields in hash."""
        try:
            self._redis.hset(key, mapping=mapping)
            return True
        except Exception as e:
            logger.error(f"Redis HMSET error for key '{key}': {e}")
            return False

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        """Increment integer field by amount."""
        try:
            return self._redis.hincrby(key, field, amount)
        except Exception as e:
            logger.error(f"Redis HINCRBY error for key '{key}', field '{field}': {e}")
            return 0

    def hincrbyfloat(self, key: str, field: str, amount: float) -> float:
        """Increment float field by amount."""
        try:
            return float(self._redis.hincrbyfloat(key, field, amount))
        except Exception as e:
            logger.error(f"Redis HINCRBYFLOAT error for key '{key}', field '{field}': {e}")
            return 0.0

    # =========================================================================
    # IListRepository implementation
    # =========================================================================

    def lpush(self, key: str, *values: str) -> int:
        """Push values to front of list."""
        try:
            return self._redis.lpush(key, *values)
        except Exception as e:
            logger.error(f"Redis LPUSH error for key '{key}': {e}")
            return 0

    def rpush(self, key: str, *values: str) -> int:
        """Push values to back of list."""
        try:
            return self._redis.rpush(key, *values)
        except Exception as e:
            logger.error(f"Redis RPUSH error for key '{key}': {e}")
            return 0

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Get range of values from list."""
        try:
            return self._redis.lrange(key, start, end)
        except Exception as e:
            logger.error(f"Redis LRANGE error for key '{key}': {e}")
            return []

    def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim list to specified range."""
        try:
            self._redis.ltrim(key, start, end)
            return True
        except Exception as e:
            logger.error(f"Redis LTRIM error for key '{key}': {e}")
            return False

    def llen(self, key: str) -> int:
        """Get list length."""
        try:
            return self._redis.llen(key)
        except Exception as e:
            logger.error(f"Redis LLEN error for key '{key}': {e}")
            return 0

    # =========================================================================
    # Additional methods
    # =========================================================================

    def expire(self, key: str, seconds: int) -> bool:
        """Set TTL on key."""
        try:
            return bool(self._redis.expire(key, seconds))
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key '{key}': {e}")
            return False

    def pipeline(self):
        """Get pipeline for batch operations."""
        return self._redis.pipeline()

    def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self._redis.ping()
        except Exception:
            return False
