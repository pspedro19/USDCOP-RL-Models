"""
Repository Pattern Implementations
==================================

Concrete implementations of repository interfaces for DIP.
Allows swapping storage backends without changing business logic.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

from .redis_state_repository import RedisStateRepository
from .memory_state_repository import InMemoryStateRepository
from .daily_stats_repository import DailyStatsRepository
from .trade_log_repository import TradeLogRepository
from .repository_factory import RepositoryFactory

__all__ = [
    'RedisStateRepository',
    'InMemoryStateRepository',
    'DailyStatsRepository',
    'TradeLogRepository',
    'RepositoryFactory',
]
