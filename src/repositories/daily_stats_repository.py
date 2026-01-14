"""
Daily Stats Repository
======================

Repository for daily trading statistics persistence.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import logging
from typing import Optional, List, Any
from datetime import datetime, timedelta

from src.core.interfaces.repository import (
    IDailyStatsRepository,
    IHashRepository,
)
from src.core.interfaces.risk import DailyStats

logger = logging.getLogger(__name__)


class DailyStatsRepository(IDailyStatsRepository):
    """
    Repository for daily trading statistics.

    Uses IHashRepository for atomic field updates.
    Composition over inheritance.
    """

    KEY_PREFIX = "risk:daily_stats"
    TTL_DAYS = 30  # Keep stats for 30 days

    def __init__(self, hash_repo: IHashRepository, timezone: str = "America/Bogota"):
        """
        Args:
            hash_repo: Hash repository for storage
            timezone: Timezone for date calculations
        """
        self._repo = hash_repo
        self._timezone = timezone
        try:
            import pytz
            self._tz = pytz.timezone(timezone)
        except ImportError:
            self._tz = None
            logger.warning("pytz not installed, using UTC")

    def _get_today(self) -> str:
        """Get today's date string."""
        if self._tz:
            return datetime.now(self._tz).strftime("%Y-%m-%d")
        return datetime.utcnow().strftime("%Y-%m-%d")

    def _get_key(self, date: str) -> str:
        """Get Redis key for date."""
        return f"{self.KEY_PREFIX}:{date}"

    def get(self, date: Optional[str] = None) -> Optional[DailyStats]:
        """Get daily stats for date."""
        date = date or self._get_today()
        key = self._get_key(date)

        data = self._repo.hgetall(key)
        if not data:
            return DailyStats(date=date)

        return DailyStats(
            date=date,
            pnl=float(data.get("pnl", 0)),
            pnl_percent=float(data.get("pnl_percent", 0)),
            peak_pnl=float(data.get("peak_pnl", 0)),
            drawdown=float(data.get("drawdown", 0)),
            trades_count=int(data.get("trades_count", 0)),
            winning_trades=int(data.get("winning_trades", 0)),
            losing_trades=int(data.get("losing_trades", 0)),
            consecutive_losses=int(data.get("consecutive_losses", 0)),
            consecutive_wins=int(data.get("consecutive_wins", 0)),
            last_trade_time=data.get("last_trade_time") or None,
            circuit_breaker_triggered=data.get("circuit_breaker_triggered", "false") == "true",
            circuit_breaker_reason=data.get("circuit_breaker_reason") or None,
        )

    def save(self, stats: DailyStats) -> bool:
        """Save daily stats."""
        key = self._get_key(stats.date)

        mapping = {
            "pnl": str(stats.pnl),
            "pnl_percent": str(stats.pnl_percent),
            "peak_pnl": str(stats.peak_pnl),
            "drawdown": str(stats.drawdown),
            "trades_count": str(stats.trades_count),
            "winning_trades": str(stats.winning_trades),
            "losing_trades": str(stats.losing_trades),
            "consecutive_losses": str(stats.consecutive_losses),
            "consecutive_wins": str(stats.consecutive_wins),
            "last_trade_time": stats.last_trade_time or "",
            "circuit_breaker_triggered": "true" if stats.circuit_breaker_triggered else "false",
            "circuit_breaker_reason": stats.circuit_breaker_reason or "",
        }

        result = self._repo.hmset(key, mapping)

        # Set TTL if supported
        if hasattr(self._repo, 'expire'):
            self._repo.expire(key, self.TTL_DAYS * 86400)

        return result

    def get_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[DailyStats]:
        """Get stats for date range."""
        stats_list = []

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            stats = self.get(date_str)
            if stats and stats.trades_count > 0:
                stats_list.append(stats)
            current += timedelta(days=1)

        return stats_list

    def update_field(
        self,
        date: str,
        field: str,
        value: Any
    ) -> bool:
        """Update single field atomically."""
        key = self._get_key(date)
        return self._repo.hset(key, field, str(value))

    def increment_field(
        self,
        date: str,
        field: str,
        amount: float = 1.0
    ) -> float:
        """Increment numeric field atomically."""
        key = self._get_key(date)

        if isinstance(amount, int):
            return float(self._repo.hincrby(key, field, amount))
        return self._repo.hincrbyfloat(key, field, amount)

    def get_or_create(self, date: Optional[str] = None) -> DailyStats:
        """Get stats or create empty record."""
        date = date or self._get_today()
        stats = self.get(date)

        if stats is None:
            stats = DailyStats(date=date)
            self.save(stats)

        return stats
