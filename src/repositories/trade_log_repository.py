"""
Trade Log Repository
====================

Repository for trade log persistence.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from src.core.interfaces.repository import (
    ITradeLogRepository,
    IListRepository,
    IStateRepository,
)

logger = logging.getLogger(__name__)


class TradeLogRepository(ITradeLogRepository):
    """
    Repository for trade logging.

    Uses list repository for chronological logs
    and state repository for trade lookup by ID.
    """

    KEY_LOG = "risk:trade_log"
    KEY_TRADE_PREFIX = "risk:trade"
    MAX_LOG_SIZE = 1000  # Keep last 1000 trades

    def __init__(
        self,
        list_repo: IListRepository,
        state_repo: Optional[IStateRepository] = None,
        timezone: str = "America/Bogota"
    ):
        """
        Args:
            list_repo: List repository for log storage
            state_repo: State repository for trade lookup (optional)
            timezone: Timezone for timestamps
        """
        self._list_repo = list_repo
        self._state_repo = state_repo
        self._timezone = timezone
        try:
            import pytz
            self._tz = pytz.timezone(timezone)
        except ImportError:
            self._tz = None

    def _get_now(self) -> str:
        """Get current timestamp."""
        if self._tz:
            return datetime.now(self._tz).isoformat()
        return datetime.utcnow().isoformat()

    def _get_today(self) -> str:
        """Get today's date string."""
        if self._tz:
            from datetime import datetime
            import pytz
            return datetime.now(self._tz).strftime("%Y-%m-%d")
        return datetime.utcnow().strftime("%Y-%m-%d")

    def log_trade(
        self,
        trade_id: str,
        signal: str,
        confidence: float,
        pnl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log a trade entry."""
        if not trade_id:
            trade_id = str(uuid.uuid4())

        entry = {
            "trade_id": trade_id,
            "signal": signal,
            "confidence": confidence,
            "pnl": pnl,
            "timestamp": self._get_now(),
            "date": self._get_today(),
            "metadata": metadata or {},
        }

        try:
            # Add to log list
            entry_json = json.dumps(entry)
            log_key = f"{self.KEY_LOG}:{self._get_today()}"
            self._list_repo.lpush(log_key, entry_json)

            # Trim to max size
            self._list_repo.ltrim(log_key, 0, self.MAX_LOG_SIZE - 1)

            # Store individual trade for lookup
            if self._state_repo:
                trade_key = f"{self.KEY_TRADE_PREFIX}:{trade_id}"
                self._state_repo.set(trade_key, entry, ttl=86400 * 7)  # 7 days

            return True
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            return False

    def get_recent_trades(
        self,
        limit: int = 100,
        date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent trades."""
        date = date or self._get_today()
        log_key = f"{self.KEY_LOG}:{date}"

        try:
            entries = self._list_repo.lrange(log_key, 0, limit - 1)
            return [json.loads(e) for e in entries]
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get specific trade by ID."""
        if not self._state_repo:
            return None

        try:
            trade_key = f"{self.KEY_TRADE_PREFIX}:{trade_id}"
            return self._state_repo.get(trade_key)
        except Exception as e:
            logger.error(f"Failed to get trade {trade_id}: {e}")
            return None

    def update_trade(
        self,
        trade_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update trade entry."""
        if not self._state_repo:
            return False

        try:
            trade = self.get_trade(trade_id)
            if not trade:
                return False

            trade.update(updates)
            trade["updated_at"] = self._get_now()

            trade_key = f"{self.KEY_TRADE_PREFIX}:{trade_id}"
            return self._state_repo.set(trade_key, trade, ttl=86400 * 7)
        except Exception as e:
            logger.error(f"Failed to update trade {trade_id}: {e}")
            return False
