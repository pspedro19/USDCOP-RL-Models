"""
Consecutive Losses Check
========================

Validates consecutive losses are within acceptable limits.

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


class ConsecutiveLossesCheck(IRiskCheck):
    """
    Check if consecutive losses limit has been reached.

    After too many consecutive losses, a cooldown period
    is enforced to prevent revenge trading.
    """

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        cooldown_seconds: int = 300,  # 5 minutes
        set_cooldown_fn: Optional[Callable[[int, str], None]] = None,
    ):
        """
        Args:
            max_consecutive_losses: Maximum allowed consecutive losses
            cooldown_seconds: Cooldown duration after limit reached
            set_cooldown_fn: Function to set cooldown
        """
        self._max_consecutive_losses = max_consecutive_losses
        self._cooldown_seconds = cooldown_seconds
        self._set_cooldown = set_cooldown_fn

    @property
    def name(self) -> str:
        return "consecutive_losses"

    @property
    def order(self) -> int:
        return 60  # Risk limit check

    def check(self, context: RiskContext) -> RiskCheckResult:
        """Check consecutive losses."""
        if context.consecutive_losses >= self._max_consecutive_losses:
            # Set cooldown
            reason = f"Consecutive losses limit ({context.consecutive_losses})"

            if self._set_cooldown:
                self._set_cooldown(self._cooldown_seconds, reason)

            return RiskCheckResult(
                approved=False,
                status=RiskStatus.CONSECUTIVE_LOSSES,
                message=f"{context.consecutive_losses} consecutive losses (limit: {self._max_consecutive_losses})",
                metadata={
                    "consecutive_losses": context.consecutive_losses,
                    "limit": self._max_consecutive_losses,
                    "cooldown_set": self._cooldown_seconds,
                },
            )

        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message=f"Consecutive losses ({context.consecutive_losses}) within limit",
        )
