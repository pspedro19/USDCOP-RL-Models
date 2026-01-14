"""
In-Memory State Repository
==========================

In-memory implementation of IStateRepository for testing and fallback.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import time
import logging
from typing import Optional, Dict, Any, List
from threading import Lock
from dataclasses import dataclass

from src.core.interfaces.repository import (
    IStateRepository,
    IHashRepository,
    IListRepository,
)

logger = logging.getLogger(__name__)


@dataclass
class CachedValue:
    """Value with optional TTL."""
    value: Any
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class InMemoryStateRepository(IStateRepository, IHashRepository, IListRepository):
    """
    In-memory implementation of state persistence.

    Used for:
    - Unit testing (no external dependencies)
    - Development without Redis
    - Fallback when Redis is unavailable

    Thread-safe with locks.
    """

    def __init__(self):
        self._store: Dict[str, CachedValue] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._lists: Dict[str, List[str]] = {}
        self._lock = Lock()

    def _cleanup_expired(self):
        """Remove expired keys."""
        now = time.time()
        expired = [
            k for k, v in self._store.items()
            if v.expires_at and v.expires_at < now
        ]
        for k in expired:
            del self._store[k]

    # =========================================================================
    # IStateRepository implementation
    # =========================================================================

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state by key."""
        with self._lock:
            self._cleanup_expired()
            cached = self._store.get(key)
            if cached is None or cached.is_expired():
                return None
            return cached.value

    def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set state with optional TTL."""
        with self._lock:
            expires_at = time.time() + ttl if ttl else None
            self._store[key] = CachedValue(value=value, expires_at=expires_at)
            return True

    def delete(self, key: str) -> bool:
        """Delete state by key."""
        with self._lock:
            self._store.pop(key, None)
            self._hashes.pop(key, None)
            self._lists.pop(key, None)
            return True

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            self._cleanup_expired()
            cached = self._store.get(key)
            return cached is not None and not cached.is_expired()

    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key."""
        with self._lock:
            cached = self._store.get(key)
            if cached is None or cached.expires_at is None:
                return None
            remaining = int(cached.expires_at - time.time())
            return remaining if remaining > 0 else None

    # =========================================================================
    # IHashRepository implementation
    # =========================================================================

    def hget(self, key: str, field: str) -> Optional[str]:
        """Get single field from hash."""
        with self._lock:
            hash_data = self._hashes.get(key, {})
            return hash_data.get(field)

    def hset(self, key: str, field: str, value: str) -> bool:
        """Set single field in hash."""
        with self._lock:
            if key not in self._hashes:
                self._hashes[key] = {}
            self._hashes[key][field] = value
            return True

    def hgetall(self, key: str) -> Dict[str, str]:
        """Get all fields from hash."""
        with self._lock:
            return self._hashes.get(key, {}).copy()

    def hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        """Set multiple fields in hash."""
        with self._lock:
            if key not in self._hashes:
                self._hashes[key] = {}
            self._hashes[key].update(mapping)
            return True

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        """Increment integer field by amount."""
        with self._lock:
            if key not in self._hashes:
                self._hashes[key] = {}
            current = int(self._hashes[key].get(field, "0"))
            new_value = current + amount
            self._hashes[key][field] = str(new_value)
            return new_value

    def hincrbyfloat(self, key: str, field: str, amount: float) -> float:
        """Increment float field by amount."""
        with self._lock:
            if key not in self._hashes:
                self._hashes[key] = {}
            current = float(self._hashes[key].get(field, "0.0"))
            new_value = current + amount
            self._hashes[key][field] = str(new_value)
            return new_value

    # =========================================================================
    # IListRepository implementation
    # =========================================================================

    def lpush(self, key: str, *values: str) -> int:
        """Push values to front of list."""
        with self._lock:
            if key not in self._lists:
                self._lists[key] = []
            for v in reversed(values):
                self._lists[key].insert(0, v)
            return len(self._lists[key])

    def rpush(self, key: str, *values: str) -> int:
        """Push values to back of list."""
        with self._lock:
            if key not in self._lists:
                self._lists[key] = []
            self._lists[key].extend(values)
            return len(self._lists[key])

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Get range of values from list."""
        with self._lock:
            lst = self._lists.get(key, [])
            # Redis-style indexing (end is inclusive)
            if end == -1:
                return lst[start:]
            return lst[start:end + 1]

    def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim list to specified range."""
        with self._lock:
            if key not in self._lists:
                return True
            if end == -1:
                self._lists[key] = self._lists[key][start:]
            else:
                self._lists[key] = self._lists[key][start:end + 1]
            return True

    def llen(self, key: str) -> int:
        """Get list length."""
        with self._lock:
            return len(self._lists.get(key, []))

    # =========================================================================
    # Additional methods
    # =========================================================================

    def clear(self):
        """Clear all data (for testing)."""
        with self._lock:
            self._store.clear()
            self._hashes.clear()
            self._lists.clear()

    def health_check(self) -> bool:
        """Always healthy for in-memory."""
        return True
