"""
Max Trades Check
================

Validates daily trade count is within limit.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

from src.core.interfaces.risk import (
    IRiskCheck,
    RiskContext,
    RiskCheckResult,
    RiskStatus,
)


class MaxTradesCheck(IRiskCheck):
    """
    Check if daily trade limit has been reached.

    Prevents overtrading by limiting the number of
    trades per day.
    """

    def __init__(self, max_trades_per_day: int = 10):
        """
        Args:
            max_trades_per_day: Maximum trades allowed per day
        """
        self._max_trades = max_trades_per_day

    @property
    def name(self) -> str:
        return "max_trades"

    @property
    def order(self) -> int:
        return 70  # Last standard check

    def check(self, context: RiskContext) -> RiskCheckResult:
        """Check trade count limit."""
        if context.trades_today >= self._max_trades:
            return RiskCheckResult(
                approved=False,
                status=RiskStatus.MAX_TRADES_REACHED,
                message=f"Max trades ({self._max_trades}) reached for today",
                metadata={
                    "trades_today": context.trades_today,
                    "limit": self._max_trades,
                },
            )

        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message=f"Trade count ({context.trades_today}) within limit",
        )
