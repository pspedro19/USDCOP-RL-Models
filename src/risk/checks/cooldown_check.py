"""
Cooldown Check
==============

Checks if cooldown period is active after losses.

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
    ICooldownManager,
)


class CooldownCheck(IRiskCheck):
    """
    Check if cooldown period is active.

    Cooldowns are set after consecutive losses or
    other risk events to prevent revenge trading.
    """

    def __init__(
        self,
        cooldown_manager: Optional[ICooldownManager] = None,
        is_active_fn: Optional[Callable[[], tuple[bool, Optional[int]]]] = None,
    ):
        """
        Args:
            cooldown_manager: ICooldownManager implementation
            is_active_fn: Alternative function to check cooldown status
        """
        self._cooldown_manager = cooldown_manager
        self._is_active_fn = is_active_fn

    @property
    def name(self) -> str:
        return "cooldown"

    @property
    def order(self) -> int:
        return 30  # After circuit breaker, before risk checks

    def check(self, context: RiskContext) -> RiskCheckResult:
        """Check cooldown status."""
        is_active = False
        remaining_seconds = None

        if self._cooldown_manager:
            is_active, remaining_seconds = self._cooldown_manager.is_active()
        elif self._is_active_fn:
            is_active, remaining_seconds = self._is_active_fn()
        else:
            # Check context for cooldown info
            is_active = context.extra.get("cooldown_active", False)
            remaining_seconds = context.extra.get("cooldown_remaining")

        if is_active:
            return RiskCheckResult(
                approved=False,
                status=RiskStatus.COOLDOWN_ACTIVE,
                message=f"Cooldown active: {remaining_seconds}s remaining",
                metadata={"remaining_seconds": remaining_seconds},
            )

        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message="No cooldown active",
        )
