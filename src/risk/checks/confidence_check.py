"""
Confidence Check
================

Validates model confidence meets minimum threshold.

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


class ConfidenceCheck(IRiskCheck):
    """
    Check if model confidence meets minimum threshold.

    Low confidence predictions are rejected to avoid
    unreliable trading signals.
    """

    def __init__(self, min_confidence: float = 0.6):
        """
        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0)
        """
        self._min_confidence = min_confidence

    @property
    def name(self) -> str:
        return "confidence"

    @property
    def order(self) -> int:
        return 40  # Before risk limit checks

    def check(self, context: RiskContext) -> RiskCheckResult:
        """Check confidence threshold."""
        if context.confidence < self._min_confidence:
            return RiskCheckResult(
                approved=False,
                status=RiskStatus.LOW_CONFIDENCE,
                message=f"Confidence {context.confidence:.2%} below threshold {self._min_confidence:.2%}",
                metadata={
                    "confidence": context.confidence,
                    "threshold": self._min_confidence,
                },
            )

        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message=f"Confidence {context.confidence:.2%} meets threshold",
        )
