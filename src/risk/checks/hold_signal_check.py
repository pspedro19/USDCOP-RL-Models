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
    RiskCheckResult,
    RiskContext,
    RiskStatus,
)


# Signals that must NEVER be blocked by the chain (audit A7-01):
#   HOLD  — no action taken, nothing to gate.
#   CLOSE/FLAT/EXIT — position-CLOSING signals; blocking one would trap an open
#   position behind an entry-oriented risk gate (the spec forbids it; RiskEnforcer
#   and RiskManager already treat exits as passthrough — the Chain was the one
#   path that didn't).
PASSTHROUGH_SIGNALS = frozenset({"HOLD", "CLOSE", "FLAT", "EXIT"})


class HoldSignalCheck(IRiskCheck):
    """
    Passthrough check for HOLD and position-closing signals.

    Both short-circuit the chain: HOLD because no action is taken, and
    CLOSE/FLAT/EXIT because a closing order must never be blocked by
    entry-oriented risk checks (audit A7-01).
    """

    @property
    def name(self) -> str:
        return "hold_signal"

    @property
    def order(self) -> int:
        return 0  # First check - immediately pass HOLD/exit signals

    def check(self, context: RiskContext) -> RiskCheckResult:
        """HOLD and CLOSE/FLAT/EXIT signals always pass (short-circuit)."""
        signal = str(context.signal or "").upper()
        if signal in PASSTHROUGH_SIGNALS:
            return RiskCheckResult(
                approved=True,
                status=RiskStatus.HOLD_SIGNAL,
                message=(
                    "HOLD signal - no action needed" if signal == "HOLD"
                    else f"{signal} signal - position-closing orders are never blocked"
                ),
            )

        # Not a passthrough signal, continue chain
        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message="Not a HOLD/exit signal, continue checks",
        )
