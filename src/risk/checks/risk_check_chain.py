"""
Risk Check Chain
================

Chain of Responsibility orchestrator for risk checks.
Executes checks in order and returns on first failure.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

from typing import List, Optional
import logging

from src.core.interfaces.risk import (
    IRiskCheck,
    RiskContext,
    RiskCheckResult,
    RiskStatus,
)


logger = logging.getLogger(__name__)


class RiskCheckChain:
    """
    Chain of Responsibility orchestrator.

    Executes risk checks in order by their `order` property.
    Stops on first failure unless check allows continuation.

    Usage:
        # Create chain with default checks
        chain = RiskCheckChain.with_defaults(config)

        # Or create custom chain
        chain = RiskCheckChain([
            HoldSignalCheck(),
            TradingHoursCheck(...),
            CircuitBreakerCheck(...),
            ...
        ])

        # Run chain
        context = RiskContext(signal="BUY", confidence=0.8, ...)
        result = chain.run(context)

        if result.approved:
            execute_trade()
    """

    def __init__(self, checks: Optional[List[IRiskCheck]] = None):
        """
        Args:
            checks: List of checks to run (will be sorted by order)
        """
        self._checks = sorted(checks or [], key=lambda c: c.order)
        self._check_names = [c.name for c in self._checks]

    def add_check(self, check: IRiskCheck) -> 'RiskCheckChain':
        """
        Add a check to the chain.

        Open/Closed: Add new checks without modifying existing code.

        Args:
            check: Risk check to add

        Returns:
            Self for chaining
        """
        self._checks.append(check)
        self._checks = sorted(self._checks, key=lambda c: c.order)
        self._check_names = [c.name for c in self._checks]
        return self

    def remove_check(self, name: str) -> 'RiskCheckChain':
        """
        Remove a check by name.

        Args:
            name: Check name to remove

        Returns:
            Self for chaining
        """
        self._checks = [c for c in self._checks if c.name != name]
        self._check_names = [c.name for c in self._checks]
        return self

    def run(self, context: RiskContext) -> RiskCheckResult:
        """
        Execute the check chain.

        Args:
            context: Risk context with signal and stats

        Returns:
            RiskCheckResult - first failure or final approval
        """
        checks_passed = []

        for check in self._checks:
            try:
                result = check.check(context)

                logger.debug(
                    f"Risk check '{check.name}' (order={check.order}): "
                    f"approved={result.approved}, status={result.status.value}"
                )

                if not result.approved:
                    # Add metadata about chain execution
                    result.metadata["checks_passed"] = checks_passed
                    result.metadata["check_that_failed"] = check.name
                    return result

                checks_passed.append(check.name)

                # Special case: HOLD signal short-circuits the chain
                if result.status == RiskStatus.HOLD_SIGNAL:
                    result.metadata["checks_passed"] = checks_passed
                    result.metadata["short_circuited"] = True
                    return result

            except Exception as e:
                logger.error(f"Risk check '{check.name}' raised exception: {e}")
                return RiskCheckResult(
                    approved=False,
                    status=RiskStatus.SYSTEM_ERROR,
                    message=f"Risk check '{check.name}' failed with error: {str(e)}",
                    metadata={
                        "check_name": check.name,
                        "error": str(e),
                        "checks_passed": checks_passed,
                    },
                )

        # All checks passed
        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message="All risk checks passed",
            metadata={"checks_passed": checks_passed},
        )

    @property
    def check_names(self) -> List[str]:
        """Get list of check names in execution order."""
        return self._check_names.copy()

    @property
    def check_count(self) -> int:
        """Get number of checks in chain."""
        return len(self._checks)

    @classmethod
    def with_defaults(
        cls,
        config: Optional[dict] = None,
        circuit_breaker=None,
        cooldown_manager=None,
        trigger_circuit_breaker_fn=None,
        set_cooldown_fn=None,
    ) -> 'RiskCheckChain':
        """
        Create chain with default checks.

        Args:
            config: Configuration dict with risk limits
            circuit_breaker: ICircuitBreaker implementation
            cooldown_manager: ICooldownManager implementation
            trigger_circuit_breaker_fn: Function to trigger circuit breaker
            set_cooldown_fn: Function to set cooldown

        Returns:
            Configured RiskCheckChain
        """
        from .hold_signal_check import HoldSignalCheck
        from .trading_hours_check import TradingHoursCheck
        from .circuit_breaker_check import CircuitBreakerCheck
        from .cooldown_check import CooldownCheck
        from .confidence_check import ConfidenceCheck
        from .daily_loss_check import DailyLossLimitCheck
        from .drawdown_check import DrawdownCheck
        from .consecutive_losses_check import ConsecutiveLossesCheck
        from .max_trades_check import MaxTradesCheck

        config = config or {}

        checks = [
            HoldSignalCheck(),
            TradingHoursCheck(
                start_time=config.get("start_time"),
                end_time=config.get("end_time"),
                trading_days=config.get("trading_days"),
                timezone_str=config.get("timezone", "America/Bogota"),
            ) if config.get("start_time") else TradingHoursCheck(),
            CircuitBreakerCheck(circuit_breaker=circuit_breaker),
            CooldownCheck(cooldown_manager=cooldown_manager),
            ConfidenceCheck(
                min_confidence=config.get("min_confidence", 0.6)
            ),
            DailyLossLimitCheck(
                max_daily_loss=config.get("max_daily_loss", -0.02),
                trigger_circuit_breaker_fn=trigger_circuit_breaker_fn,
            ),
            DrawdownCheck(
                max_drawdown=config.get("max_drawdown", -0.01),
                trigger_circuit_breaker_fn=trigger_circuit_breaker_fn,
            ),
            ConsecutiveLossesCheck(
                max_consecutive_losses=config.get("max_consecutive_losses", 3),
                cooldown_seconds=config.get("cooldown_after_loss", 300),
                set_cooldown_fn=set_cooldown_fn,
            ),
            MaxTradesCheck(
                max_trades_per_day=config.get("max_trades_per_day", 10)
            ),
        ]

        return cls(checks)

    def __repr__(self) -> str:
        return f"RiskCheckChain(checks={self._check_names})"
