"""
Hold Signal Check
=================

Passthrough check for HOLD signals - they always pass.

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


class HoldSignalCheck(IRiskCheck):
    """
    Check for HOLD signals.

    HOLD signals always pass since no action is taken.
    This check short-circuits the chain for efficiency.
    """

    @property
    def name(self) -> str:
        return "hold_signal"

    @property
    def order(self) -> int:
        return 0  # First check - immediately pass HOLD signals

    def check(self, context: RiskContext) -> RiskCheckResult:
        """HOLD signals always pass."""
        if context.signal == "HOLD":
            return RiskCheckResult(
                approved=True,
                status=RiskStatus.HOLD_SIGNAL,
                message="HOLD signal - no action needed",
            )

        # Not a HOLD signal, continue chain
        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message="Not a HOLD signal, continue checks",
        )
