"""
Repository Pattern Implementations
==================================

Concrete implementations of repository interfaces for DIP.
Allows swapping storage backends without changing business logic.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

from .daily_stats_repository import DailyStatsRepository
from .memory_state_repository import InMemoryStateRepository
from .redis_state_repository import RedisStateRepository
from .repository_factory import RepositoryFactory
from .trade_log_repository import TradeLogRepository

__all__ = [
    'DailyStatsRepository',
    'InMemoryStateRepository',
    'RedisStateRepository',
    'RepositoryFactory',
    'TradeLogRepository',
]
