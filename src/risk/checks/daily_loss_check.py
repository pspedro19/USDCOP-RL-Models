"""
Daily Loss Limit Check
======================

Validates daily P&L is within acceptable limits.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

from typing import Optional, Callable

from src.core.interfaces.risk import (
    IRiskCheck,
    RiskContext,
    RiskCheckResult,
    RiskStatus,
)


class DailyLossLimitCheck(IRiskCheck):
    """
    Check if daily loss limit has been reached.

    When daily losses exceed the limit, circuit breaker
    should be triggered for the day.
    """

    def __init__(
        self,
        max_daily_loss: float = -0.02,  # -2%
        trigger_circuit_breaker_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            max_daily_loss: Maximum allowed daily loss (negative percentage)
            trigger_circuit_breaker_fn: Function to trigger circuit breaker
        """
        self._max_daily_loss = max_daily_loss
        self._trigger_circuit_breaker = trigger_circuit_breaker_fn

    @property
    def name(self) -> str:
        return "daily_loss_limit"

    @property
    def order(self) -> int:
        return 50  # Risk limit check

    def check(self, context: RiskContext) -> RiskCheckResult:
        """Check daily loss limit."""
        if context.daily_pnl_percent <= self._max_daily_loss:
            # Trigger circuit breaker
            reason = f"Daily loss limit reached: {context.daily_pnl_percent:.2%}"

            if self._trigger_circuit_breaker:
                self._trigger_circuit_breaker(reason)

            return RiskCheckResult(
                approved=False,
                status=RiskStatus.DAILY_LOSS_LIMIT,
                message=f"Daily loss {context.daily_pnl_percent:.2%} exceeds limit {self._max_daily_loss:.2%}",
                metadata={
                    "daily_pnl_percent": context.daily_pnl_percent,
                    "limit": self._max_daily_loss,
                },
            )

        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message=f"Daily P&L {context.daily_pnl_percent:.2%} within limit",
        )
