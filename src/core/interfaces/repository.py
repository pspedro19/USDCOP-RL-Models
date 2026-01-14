"""
Repository Interfaces
=====================

Defines abstract interfaces for data persistence following DIP.
Allows swapping between Redis, PostgreSQL, in-memory, etc.

Author: Trading Team
Version: 2.0.0
Date: 2025-01-14
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, TypeVar, Generic
from datetime import datetime


T = TypeVar('T')


# =============================================================================
# Dependency Inversion: Abstract repositories
# =============================================================================

class IStateRepository(ABC):
    """
    Interface for generic key-value state persistence.

    Dependency Inversion: High-level modules depend on this interface,
    not on concrete Redis/PostgreSQL implementations.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get state by key.

        Args:
            key: State key

        Returns:
            State dict or None if not found
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set state with optional TTL.

        Args:
            key: State key
            value: State data
            ttl: Time to live in seconds (optional)

        Returns:
            True if set successfully
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete state by key.

        Args:
            key: State key

        Returns:
            True if deleted (or didn't exist)
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for key.

        Returns:
            Seconds remaining or None if no TTL/key doesn't exist
        """
        pass


class IHashRepository(ABC):
    """
    Interface for hash/map data structures.

    Used for complex state like daily stats where multiple fields
    need atomic updates.
    """

    @abstractmethod
    def hget(self, key: str, field: str) -> Optional[str]:
        """Get single field from hash."""
        pass

    @abstractmethod
    def hset(self, key: str, field: str, value: str) -> bool:
        """Set single field in hash."""
        pass

    @abstractmethod
    def hgetall(self, key: str) -> Dict[str, str]:
        """Get all fields from hash."""
        pass

    @abstractmethod
    def hmset(self, key: str, mapping: Dict[str, str]) -> bool:
        """Set multiple fields in hash."""
        pass

    @abstractmethod
    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        """Increment integer field by amount."""
        pass

    @abstractmethod
    def hincrbyfloat(self, key: str, field: str, amount: float) -> float:
        """Increment float field by amount."""
        pass


class IListRepository(ABC):
    """
    Interface for list/log data structures.

    Used for audit logs, trade history, etc.
    """

    @abstractmethod
    def lpush(self, key: str, *values: str) -> int:
        """Push values to front of list."""
        pass

    @abstractmethod
    def rpush(self, key: str, *values: str) -> int:
        """Push values to back of list."""
        pass

    @abstractmethod
    def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Get range of values from list."""
        pass

    @abstractmethod
    def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim list to specified range."""
        pass

    @abstractmethod
    def llen(self, key: str) -> int:
        """Get list length."""
        pass


class IDailyStatsRepository(ABC):
    """
    Interface for daily trading statistics persistence.

    Single Responsibility: Manage daily stats CRUD operations.
    """

    @abstractmethod
    def get(self, date: Optional[str] = None) -> Optional['DailyStats']:
        """
        Get daily stats for date.

        Args:
            date: Date string (YYYY-MM-DD) or None for today

        Returns:
            DailyStats or None if not found
        """
        pass

    @abstractmethod
    def save(self, stats: 'DailyStats') -> bool:
        """
        Save daily stats.

        Args:
            stats: DailyStats to save

        Returns:
            True if saved successfully
        """
        pass

    @abstractmethod
    def get_range(
        self,
        start_date: str,
        end_date: str
    ) -> List['DailyStats']:
        """
        Get stats for date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of DailyStats
        """
        pass

    @abstractmethod
    def update_field(
        self,
        date: str,
        field: str,
        value: Any
    ) -> bool:
        """Update single field atomically."""
        pass

    @abstractmethod
    def increment_field(
        self,
        date: str,
        field: str,
        amount: float = 1.0
    ) -> float:
        """Increment numeric field atomically."""
        pass


class ITradeLogRepository(ABC):
    """
    Interface for trade log persistence.

    Single Responsibility: Manage trade log entries.
    """

    @abstractmethod
    def log_trade(
        self,
        trade_id: str,
        signal: str,
        confidence: float,
        pnl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log a trade entry."""
        pass

    @abstractmethod
    def get_recent_trades(
        self,
        limit: int = 100,
        date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent trades."""
        pass

    @abstractmethod
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get specific trade by ID."""
        pass


class ICacheRepository(ABC):
    """
    Interface for cache operations.

    Used for feature caching, model output caching, etc.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        ttl: int = 300
    ) -> bool:
        """Set cached value with TTL."""
        pass

    @abstractmethod
    def invalidate(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Returns:
            Number of keys invalidated
        """
        pass

    @abstractmethod
    def get_or_set(
        self,
        key: str,
        factory: callable,
        ttl: int = 300
    ) -> Any:
        """
        Get from cache or compute and cache.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time to live in seconds

        Returns:
            Cached or computed value
        """
        pass


# =============================================================================
# Forward reference for DailyStats (defined in risk.py)
# =============================================================================

# Note: DailyStats is imported from risk.py when needed
# This avoids circular imports while maintaining type hints
