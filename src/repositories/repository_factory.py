"""
Repository Factory
==================

Factory for creating repository instances.
Centralizes dependency injection for repositories.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import logging
from typing import Optional

from src.core.interfaces.repository import (
    IStateRepository,
    IHashRepository,
    IListRepository,
    IDailyStatsRepository,
    ITradeLogRepository,
)

logger = logging.getLogger(__name__)


class RepositoryFactory:
    """
    Factory for repository instances.

    Factory Pattern: Centralizes creation logic.
    Singleton Pattern: Reuses configured instances.

    Usage:
        # Configure for Redis
        redis_client = redis.Redis(...)
        RepositoryFactory.configure_redis(redis_client)

        # Get repositories
        state_repo = RepositoryFactory.get_state_repository()
        stats_repo = RepositoryFactory.get_daily_stats_repository()

        # Or configure for testing
        RepositoryFactory.configure_memory()
    """

    _redis_client = None
    _state_repo: Optional[IStateRepository] = None
    _hash_repo: Optional[IHashRepository] = None
    _list_repo: Optional[IListRepository] = None
    _daily_stats_repo: Optional[IDailyStatsRepository] = None
    _trade_log_repo: Optional[ITradeLogRepository] = None
    _mode: str = "memory"  # "redis" or "memory"
    _timezone: str = "America/Bogota"

    @classmethod
    def configure_redis(
        cls,
        redis_client,
        timezone: str = "America/Bogota"
    ) -> None:
        """
        Configure factory for Redis storage.

        Args:
            redis_client: Redis client instance
            timezone: Timezone for date calculations
        """
        cls._redis_client = redis_client
        cls._mode = "redis"
        cls._timezone = timezone

        # Clear cached instances
        cls._state_repo = None
        cls._hash_repo = None
        cls._list_repo = None
        cls._daily_stats_repo = None
        cls._trade_log_repo = None

        logger.info("RepositoryFactory configured for Redis")

    @classmethod
    def configure_memory(cls, timezone: str = "America/Bogota") -> None:
        """
        Configure factory for in-memory storage.

        Used for testing and development.
        """
        cls._redis_client = None
        cls._mode = "memory"
        cls._timezone = timezone

        # Clear cached instances
        cls._state_repo = None
        cls._hash_repo = None
        cls._list_repo = None
        cls._daily_stats_repo = None
        cls._trade_log_repo = None

        logger.info("RepositoryFactory configured for in-memory storage")

    @classmethod
    def get_state_repository(cls) -> IStateRepository:
        """Get or create state repository."""
        if cls._state_repo is None:
            cls._state_repo = cls._create_state_repository()
        return cls._state_repo

    @classmethod
    def get_hash_repository(cls) -> IHashRepository:
        """Get or create hash repository."""
        if cls._hash_repo is None:
            cls._hash_repo = cls._create_hash_repository()
        return cls._hash_repo

    @classmethod
    def get_list_repository(cls) -> IListRepository:
        """Get or create list repository."""
        if cls._list_repo is None:
            cls._list_repo = cls._create_list_repository()
        return cls._list_repo

    @classmethod
    def get_daily_stats_repository(cls) -> IDailyStatsRepository:
        """Get or create daily stats repository."""
        if cls._daily_stats_repo is None:
            from .daily_stats_repository import DailyStatsRepository
            hash_repo = cls.get_hash_repository()
            cls._daily_stats_repo = DailyStatsRepository(
                hash_repo=hash_repo,
                timezone=cls._timezone
            )
        return cls._daily_stats_repo

    @classmethod
    def get_trade_log_repository(cls) -> ITradeLogRepository:
        """Get or create trade log repository."""
        if cls._trade_log_repo is None:
            from .trade_log_repository import TradeLogRepository
            list_repo = cls.get_list_repository()
            state_repo = cls.get_state_repository()
            cls._trade_log_repo = TradeLogRepository(
                list_repo=list_repo,
                state_repo=state_repo,
                timezone=cls._timezone
            )
        return cls._trade_log_repo

    @classmethod
    def _create_state_repository(cls) -> IStateRepository:
        """Create appropriate state repository."""
        if cls._mode == "redis" and cls._redis_client:
            from .redis_state_repository import RedisStateRepository
            return RedisStateRepository(cls._redis_client)
        else:
            from .memory_state_repository import InMemoryStateRepository
            return InMemoryStateRepository()

    @classmethod
    def _create_hash_repository(cls) -> IHashRepository:
        """Create appropriate hash repository."""
        # Our implementations combine interfaces
        return cls._create_state_repository()

    @classmethod
    def _create_list_repository(cls) -> IListRepository:
        """Create appropriate list repository."""
        # Our implementations combine interfaces
        return cls._create_state_repository()

    @classmethod
    def reset(cls) -> None:
        """Reset factory state (for testing)."""
        cls._redis_client = None
        cls._state_repo = None
        cls._hash_repo = None
        cls._list_repo = None
        cls._daily_stats_repo = None
        cls._trade_log_repo = None
        cls._mode = "memory"
        logger.info("RepositoryFactory reset")

    @classmethod
    def is_redis_mode(cls) -> bool:
        """Check if configured for Redis."""
        return cls._mode == "redis"

    @classmethod
    def health_check(cls) -> bool:
        """Check repository health."""
        try:
            repo = cls.get_state_repository()
            if hasattr(repo, 'health_check'):
                return repo.health_check()
            return True
        except Exception:
            return False
