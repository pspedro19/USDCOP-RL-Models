"""
Circuit Breaker Check
=====================

Checks if circuit breaker has been triggered.

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
    ICircuitBreaker,
)


class CircuitBreakerCheck(IRiskCheck):
    """
    Check if circuit breaker is active.

    The circuit breaker is triggered when severe risk conditions
    are detected (e.g., daily loss limit, max drawdown).
    """

    def __init__(
        self,
        circuit_breaker: Optional[ICircuitBreaker] = None,
        is_active_fn: Optional[Callable[[], tuple[bool, Optional[str]]]] = None,
    ):
        """
        Args:
            circuit_breaker: ICircuitBreaker implementation
            is_active_fn: Alternative function to check circuit breaker status
        """
        self._circuit_breaker = circuit_breaker
        self._is_active_fn = is_active_fn

    @property
    def name(self) -> str:
        return "circuit_breaker"

    @property
    def order(self) -> int:
        return 20  # Early check - block all trading if triggered

    def check(self, context: RiskContext) -> RiskCheckResult:
        """Check circuit breaker status."""
        is_active = False
        reason = None

        if self._circuit_breaker:
            is_active, reason = self._circuit_breaker.is_active()
        elif self._is_active_fn:
            is_active, reason = self._is_active_fn()
        else:
            # Check context for circuit breaker info
            is_active = context.extra.get("circuit_breaker_active", False)
            reason = context.extra.get("circuit_breaker_reason")

        if is_active:
            return RiskCheckResult(
                approved=False,
                status=RiskStatus.CIRCUIT_BREAKER_ACTIVE,
                message=f"Circuit breaker active: {reason}",
                metadata={"reason": reason},
            )

        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message="Circuit breaker not active",
        )
