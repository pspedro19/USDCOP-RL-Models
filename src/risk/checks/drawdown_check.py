"""
Drawdown Check
==============

Validates current drawdown is within acceptable limits.

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


class DrawdownCheck(IRiskCheck):
    """
    Check if drawdown limit has been reached.

    Drawdown is measured from the daily peak P&L.
    When exceeded, circuit breaker should be triggered.
    """

    def __init__(
        self,
        max_drawdown: float = -0.01,  # -1%
        trigger_circuit_breaker_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            max_drawdown: Maximum allowed drawdown (negative percentage)
            trigger_circuit_breaker_fn: Function to trigger circuit breaker
        """
        self._max_drawdown = max_drawdown
        self._trigger_circuit_breaker = trigger_circuit_breaker_fn

    @property
    def name(self) -> str:
        return "drawdown"

    @property
    def order(self) -> int:
        return 55  # Risk limit check, after daily loss

    def check(self, context: RiskContext) -> RiskCheckResult:
        """Check drawdown limit."""
        if context.current_drawdown <= self._max_drawdown:
            # Trigger circuit breaker
            reason = f"Max drawdown reached: {context.current_drawdown:.2%}"

            if self._trigger_circuit_breaker:
                self._trigger_circuit_breaker(reason)

            return RiskCheckResult(
                approved=False,
                status=RiskStatus.MAX_DRAWDOWN,
                message=f"Drawdown {context.current_drawdown:.2%} exceeds limit {self._max_drawdown:.2%}",
                metadata={
                    "current_drawdown": context.current_drawdown,
                    "limit": self._max_drawdown,
                },
            )

        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message=f"Drawdown {context.current_drawdown:.2%} within limit",
        )
