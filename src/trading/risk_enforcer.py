"""
RiskEnforcer - Risk Enforcement Component
==========================================

Single Responsibility: Enforce risk rules before trade execution.
Split from the PaperTrader God Class to follow SOLID principles.

This component is responsible for:
- Pre-trade risk validation
- Position size limits
- Exposure limits
- Real-time risk monitoring
- Integrating with RiskManager for decision making

Design Patterns:
- Single Responsibility Principle: Only handles risk enforcement
- Decorator Pattern: Can wrap any executor with risk checks
- Chain of Responsibility: Risk rules evaluated in sequence
- Strategy Pattern: Different risk strategies can be plugged in

Author: Trading Team
Version: 1.0.0
Date: 2025-01-16
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)
import logging

from src.core.constants import (
    MAX_DAILY_LOSS_PCT,
    MAX_DRAWDOWN_PCT,
    MAX_POSITION_SIZE,
    MIN_CONFIDENCE_THRESHOLD,
    MAX_DAILY_TRADES,
    CONSECUTIVE_LOSS_LIMIT,
    COOLDOWN_MINUTES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class RiskDecision(str, Enum):
    """Risk enforcement decision."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REDUCE = "REDUCE"  # Allow but reduce size


class RiskReason(str, Enum):
    """Reasons for risk decisions."""
    APPROVED = "approved"
    KILL_SWITCH = "kill_switch_active"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    TRADE_LIMIT = "trade_limit_reached"
    COOLDOWN = "cooldown_active"
    MAX_POSITION = "max_position_exceeded"
    LOW_CONFIDENCE = "low_confidence"
    EXPOSURE_LIMIT = "exposure_limit"
    MARKET_CLOSED = "market_closed"
    SHORT_DISABLED = "short_disabled"


@dataclass
class RiskCheckResult:
    """
    Result of a risk check.

    Attributes:
        decision: ALLOW, BLOCK, or REDUCE
        reason: Reason for the decision
        message: Human-readable message
        adjusted_size: Adjusted position size (if REDUCE)
        metadata: Additional check details
    """
    decision: RiskDecision
    reason: RiskReason
    message: str
    adjusted_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        """Check if trade is allowed."""
        return self.decision in (RiskDecision.ALLOW, RiskDecision.REDUCE)

    @property
    def is_blocked(self) -> bool:
        """Check if trade is blocked."""
        return self.decision == RiskDecision.BLOCK

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "reason": self.reason.value,
            "message": self.message,
            "adjusted_size": self.adjusted_size,
            "metadata": self.metadata,
        }


@dataclass
class RiskLimits:
    """
    Configuration for risk limits.

    Attributes:
        max_drawdown_pct: Maximum portfolio drawdown before kill switch
        max_daily_loss_pct: Maximum daily loss before stopping
        max_trades_per_day: Maximum trades per day
        max_position_size: Maximum position size (units or percentage)
        max_position_pct: Maximum position as percentage of portfolio
        min_confidence: Minimum confidence for trade execution
        cooldown_after_losses: Consecutive losses before cooldown
        cooldown_minutes: Duration of cooldown period
        enable_short: Whether short positions are allowed
        max_exposure_pct: Maximum total exposure as percentage
    """
    max_drawdown_pct: float = MAX_DRAWDOWN_PCT * 100  # Convert to percentage
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT * 100
    max_trades_per_day: int = MAX_DAILY_TRADES
    max_position_size: float = MAX_POSITION_SIZE
    max_position_pct: float = 20.0  # 20% of portfolio per position
    min_confidence: float = MIN_CONFIDENCE_THRESHOLD
    cooldown_after_losses: int = CONSECUTIVE_LOSS_LIMIT
    cooldown_minutes: int = COOLDOWN_MINUTES
    enable_short: bool = True
    max_exposure_pct: float = 100.0  # Maximum total exposure


@dataclass
class RiskState:
    """
    Current risk state tracking.

    Attributes:
        kill_switch_active: Whether kill switch is triggered
        daily_blocked: Whether daily trading is blocked
        cooldown_until: When cooldown expires
        trade_count_today: Trades executed today
        daily_pnl_pct: Cumulative daily P&L percentage
        consecutive_losses: Current losing streak
        current_drawdown_pct: Current portfolio drawdown
        total_exposure: Total position exposure
    """
    kill_switch_active: bool = False
    daily_blocked: bool = False
    cooldown_until: Optional[datetime] = None
    trade_count_today: int = 0
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0
    current_drawdown_pct: float = 0.0
    total_exposure: float = 0.0
    current_day: datetime = field(default_factory=lambda: datetime.now().date())


# =============================================================================
# Risk Rule Protocol
# =============================================================================

class IRiskRule(Protocol):
    """Protocol for individual risk rules."""

    @property
    def name(self) -> str:
        """Rule name."""
        ...

    def check(
        self,
        signal: str,
        size: float,
        price: float,
        state: RiskState,
        limits: RiskLimits,
        **kwargs
    ) -> RiskCheckResult:
        """
        Check if trade passes this rule.

        Args:
            signal: Trading signal
            size: Proposed position size
            price: Current price
            state: Current risk state
            limits: Risk limits configuration
            **kwargs: Additional context

        Returns:
            RiskCheckResult
        """
        ...


# =============================================================================
# Risk Rules Implementation
# =============================================================================

class KillSwitchRule:
    """Check kill switch status."""

    @property
    def name(self) -> str:
        return "KillSwitch"

    def check(
        self,
        signal: str,
        size: float,
        price: float,
        state: RiskState,
        limits: RiskLimits,
        **kwargs
    ) -> RiskCheckResult:
        if state.kill_switch_active:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=RiskReason.KILL_SWITCH,
                message="Kill switch active - all trading halted",
                metadata={"drawdown": state.current_drawdown_pct}
            )

        # Check if current drawdown triggers kill switch
        if state.current_drawdown_pct >= limits.max_drawdown_pct:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=RiskReason.KILL_SWITCH,
                message=f"Drawdown {state.current_drawdown_pct:.1f}% >= limit {limits.max_drawdown_pct:.1f}%",
                metadata={
                    "current_drawdown": state.current_drawdown_pct,
                    "limit": limits.max_drawdown_pct,
                    "trigger_kill_switch": True,
                }
            )

        return RiskCheckResult(
            decision=RiskDecision.ALLOW,
            reason=RiskReason.APPROVED,
            message="Kill switch check passed"
        )


class DailyLossRule:
    """Check daily loss limit."""

    @property
    def name(self) -> str:
        return "DailyLoss"

    def check(
        self,
        signal: str,
        size: float,
        price: float,
        state: RiskState,
        limits: RiskLimits,
        **kwargs
    ) -> RiskCheckResult:
        if state.daily_blocked:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=RiskReason.DAILY_LOSS_LIMIT,
                message="Daily loss limit reached - trading blocked until tomorrow",
                metadata={"daily_pnl": state.daily_pnl_pct}
            )

        if state.daily_pnl_pct <= -limits.max_daily_loss_pct:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=RiskReason.DAILY_LOSS_LIMIT,
                message=f"Daily P&L {state.daily_pnl_pct:.2f}% <= limit -{limits.max_daily_loss_pct:.1f}%",
                metadata={
                    "daily_pnl": state.daily_pnl_pct,
                    "limit": -limits.max_daily_loss_pct,
                }
            )

        return RiskCheckResult(
            decision=RiskDecision.ALLOW,
            reason=RiskReason.APPROVED,
            message="Daily loss check passed"
        )


class TradeLimitRule:
    """Check daily trade count limit."""

    @property
    def name(self) -> str:
        return "TradeLimit"

    def check(
        self,
        signal: str,
        size: float,
        price: float,
        state: RiskState,
        limits: RiskLimits,
        **kwargs
    ) -> RiskCheckResult:
        if state.trade_count_today >= limits.max_trades_per_day:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=RiskReason.TRADE_LIMIT,
                message=f"Trade limit reached: {state.trade_count_today}/{limits.max_trades_per_day}",
                metadata={
                    "trades_today": state.trade_count_today,
                    "limit": limits.max_trades_per_day,
                }
            )

        return RiskCheckResult(
            decision=RiskDecision.ALLOW,
            reason=RiskReason.APPROVED,
            message="Trade limit check passed"
        )


class CooldownRule:
    """Check cooldown status."""

    @property
    def name(self) -> str:
        return "Cooldown"

    def check(
        self,
        signal: str,
        size: float,
        price: float,
        state: RiskState,
        limits: RiskLimits,
        **kwargs
    ) -> RiskCheckResult:
        if state.cooldown_until is not None:
            now = datetime.now()
            if now < state.cooldown_until:
                remaining = (state.cooldown_until - now).total_seconds() / 60
                return RiskCheckResult(
                    decision=RiskDecision.BLOCK,
                    reason=RiskReason.COOLDOWN,
                    message=f"Cooldown active: {remaining:.1f} minutes remaining",
                    metadata={
                        "cooldown_until": state.cooldown_until.isoformat(),
                        "remaining_minutes": remaining,
                    }
                )

        return RiskCheckResult(
            decision=RiskDecision.ALLOW,
            reason=RiskReason.APPROVED,
            message="Cooldown check passed"
        )


class PositionSizeRule:
    """Check position size limits."""

    @property
    def name(self) -> str:
        return "PositionSize"

    def check(
        self,
        signal: str,
        size: float,
        price: float,
        state: RiskState,
        limits: RiskLimits,
        **kwargs
    ) -> RiskCheckResult:
        if size > limits.max_position_size:
            # Reduce to max allowed
            return RiskCheckResult(
                decision=RiskDecision.REDUCE,
                reason=RiskReason.MAX_POSITION,
                message=f"Position size {size:.4f} reduced to max {limits.max_position_size:.4f}",
                adjusted_size=limits.max_position_size,
                metadata={
                    "requested_size": size,
                    "max_size": limits.max_position_size,
                }
            )

        return RiskCheckResult(
            decision=RiskDecision.ALLOW,
            reason=RiskReason.APPROVED,
            message="Position size check passed"
        )


class ConfidenceRule:
    """Check minimum confidence threshold."""

    @property
    def name(self) -> str:
        return "Confidence"

    def check(
        self,
        signal: str,
        size: float,
        price: float,
        state: RiskState,
        limits: RiskLimits,
        **kwargs
    ) -> RiskCheckResult:
        confidence = kwargs.get("confidence", 1.0)

        if confidence < limits.min_confidence:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=RiskReason.LOW_CONFIDENCE,
                message=f"Confidence {confidence:.2f} below threshold {limits.min_confidence:.2f}",
                metadata={
                    "confidence": confidence,
                    "threshold": limits.min_confidence,
                }
            )

        return RiskCheckResult(
            decision=RiskDecision.ALLOW,
            reason=RiskReason.APPROVED,
            message="Confidence check passed"
        )


class ShortRule:
    """Check if short trading is allowed."""

    @property
    def name(self) -> str:
        return "Short"

    def check(
        self,
        signal: str,
        size: float,
        price: float,
        state: RiskState,
        limits: RiskLimits,
        **kwargs
    ) -> RiskCheckResult:
        if signal.upper() == "SHORT" and not limits.enable_short:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=RiskReason.SHORT_DISABLED,
                message="Short trading is disabled"
            )

        return RiskCheckResult(
            decision=RiskDecision.ALLOW,
            reason=RiskReason.APPROVED,
            message="Short check passed"
        )


# =============================================================================
# RiskEnforcer Implementation
# =============================================================================

class RiskEnforcer:
    """
    Enforces risk rules before trade execution.

    Single Responsibility: Validate trades against risk limits.

    This class evaluates multiple risk rules in sequence (Chain of Responsibility)
    and returns a comprehensive risk decision. It can be used as a decorator
    around any trade executor.

    Features:
    - Configurable risk limits
    - Pluggable risk rules
    - State tracking for daily limits
    - Kill switch support
    - Cooldown management
    - Trade result recording

    Usage:
        enforcer = RiskEnforcer()

        # Check before executing trade
        result = enforcer.check_signal(
            signal="LONG",
            size=100,
            price=4250.50
        )

        if result.is_allowed:
            # Execute trade
            pass

        # Record trade result
        enforcer.record_trade(pnl_pct=-0.5, signal="LONG")
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        rules: Optional[List[IRiskRule]] = None,
    ) -> None:
        """
        Initialize the risk enforcer.

        Args:
            limits: Risk limits configuration
            rules: Custom risk rules (uses defaults if None)
        """
        self._limits = limits or RiskLimits()
        self._state = RiskState()

        # Default rules in evaluation order
        self._rules: List[IRiskRule] = rules or [
            KillSwitchRule(),
            DailyLossRule(),
            TradeLimitRule(),
            CooldownRule(),
            ShortRule(),
            PositionSizeRule(),
            ConfidenceRule(),
        ]

        logger.info(
            f"RiskEnforcer initialized: max_drawdown={self._limits.max_drawdown_pct}%, "
            f"max_daily_loss={self._limits.max_daily_loss_pct}%, "
            f"rules={[r.name for r in self._rules]}"
        )

    # =========================================================================
    # Signal Validation (Main Interface)
    # =========================================================================

    def check_signal(
        self,
        signal: str,
        size: float,
        price: float,
        confidence: float = 1.0,
        **kwargs
    ) -> RiskCheckResult:
        """
        Check if a trading signal passes all risk rules.

        This is the main entry point for risk validation.

        Args:
            signal: Trading signal (LONG, SHORT, CLOSE, HOLD)
            size: Proposed position size
            price: Current market price
            confidence: Model confidence (0-1)
            **kwargs: Additional context for rules

        Returns:
            RiskCheckResult with decision and details
        """
        # Check for day change
        self._check_daily_reset()

        # Exit signals always allowed
        if signal.upper() in ("CLOSE", "FLAT", "HOLD"):
            return RiskCheckResult(
                decision=RiskDecision.ALLOW,
                reason=RiskReason.APPROVED,
                message="Exit/hold signals always allowed"
            )

        # Evaluate all rules
        adjusted_size = size
        for rule in self._rules:
            result = rule.check(
                signal=signal,
                size=adjusted_size,
                price=price,
                state=self._state,
                limits=self._limits,
                confidence=confidence,
                **kwargs
            )

            # Block if any rule fails
            if result.is_blocked:
                logger.warning(f"Risk check failed [{rule.name}]: {result.message}")
                return result

            # Track size reductions
            if result.decision == RiskDecision.REDUCE and result.adjusted_size:
                adjusted_size = result.adjusted_size

        # All checks passed
        final_result = RiskCheckResult(
            decision=RiskDecision.ALLOW,
            reason=RiskReason.APPROVED,
            message="All risk checks passed",
            adjusted_size=adjusted_size if adjusted_size != size else None,
        )

        return final_result

    def validate_signal(
        self,
        signal: str,
        current_drawdown_pct: float
    ) -> Tuple[bool, str]:
        """
        Validate signal (compatibility with existing RiskManager interface).

        Args:
            signal: Trading signal
            current_drawdown_pct: Current drawdown percentage

        Returns:
            Tuple of (allowed, reason)
        """
        # Update state with current drawdown
        self._state.current_drawdown_pct = current_drawdown_pct

        result = self.check_signal(
            signal=signal,
            size=1.0,  # Dummy size for validation
            price=1.0,  # Dummy price
        )

        return result.is_allowed, result.message

    # =========================================================================
    # Trade Recording
    # =========================================================================

    def record_trade(self, pnl_pct: float, signal: str = "unknown") -> None:
        """
        Record the result of a completed trade.

        Updates internal state:
        - Daily P&L tracking
        - Consecutive losses
        - Trade counter
        - Cooldown trigger

        Args:
            pnl_pct: Trade P&L as percentage
            signal: Signal type that generated the trade
        """
        self._check_daily_reset()

        # Update counters
        self._state.trade_count_today += 1
        self._state.daily_pnl_pct += pnl_pct

        # Track consecutive losses
        if pnl_pct < 0:
            self._state.consecutive_losses += 1
            logger.info(
                f"Loss recorded: {pnl_pct:.2f}%, "
                f"consecutive losses: {self._state.consecutive_losses}"
            )

            # Check for cooldown trigger
            if self._state.consecutive_losses >= self._limits.cooldown_after_losses:
                from datetime import timedelta
                self._state.cooldown_until = datetime.now() + timedelta(
                    minutes=self._limits.cooldown_minutes
                )
                logger.warning(
                    f"COOLDOWN ACTIVATED: {self._state.consecutive_losses} consecutive losses"
                )
        else:
            # Reset streak on win
            if self._state.consecutive_losses > 0:
                logger.info(
                    f"Win recorded: {pnl_pct:.2f}%, "
                    f"consecutive losses reset from {self._state.consecutive_losses} to 0"
                )
            self._state.consecutive_losses = 0

        # Check daily loss limit
        if self._state.daily_pnl_pct <= -self._limits.max_daily_loss_pct:
            self._state.daily_blocked = True
            logger.critical(
                f"DAILY LOSS LIMIT REACHED: {self._state.daily_pnl_pct:.2f}%"
            )

    def record_trade_result(self, pnl_pct: float, signal: str = "unknown") -> None:
        """Alias for record_trade (compatibility interface)."""
        self.record_trade(pnl_pct, signal)

    # =========================================================================
    # State Management
    # =========================================================================

    def update_drawdown(self, drawdown_pct: float) -> None:
        """
        Update current drawdown and check kill switch.

        Args:
            drawdown_pct: Current portfolio drawdown percentage
        """
        self._state.current_drawdown_pct = drawdown_pct

        # Trigger kill switch if needed
        if drawdown_pct >= self._limits.max_drawdown_pct:
            self._state.kill_switch_active = True
            logger.critical(
                f"KILL SWITCH TRIGGERED: Drawdown {drawdown_pct:.1f}% "
                f">= limit {self._limits.max_drawdown_pct:.1f}%"
            )

    def update_exposure(self, exposure: float) -> None:
        """
        Update total position exposure.

        Args:
            exposure: Total position exposure
        """
        self._state.total_exposure = exposure

    def _check_daily_reset(self) -> None:
        """Check for day change and reset daily counters."""
        today = datetime.now().date()
        if today != self._state.current_day:
            logger.info(f"Day change detected: {self._state.current_day} -> {today}")
            self._reset_daily()

    def _reset_daily(self) -> None:
        """Reset daily counters."""
        self._state.trade_count_today = 0
        self._state.daily_pnl_pct = 0.0
        self._state.daily_blocked = False
        self._state.consecutive_losses = 0
        self._state.cooldown_until = None
        self._state.current_day = datetime.now().date()
        logger.info("Daily risk counters reset")

    def reset_kill_switch(self, confirm: bool = False) -> bool:
        """
        Manually reset kill switch.

        Args:
            confirm: Must be True to reset

        Returns:
            True if reset performed
        """
        if not confirm:
            logger.warning("Kill switch reset requires confirm=True")
            return False

        if self._state.kill_switch_active:
            self._state.kill_switch_active = False
            logger.critical("KILL SWITCH MANUALLY RESET")
            return True

        return False

    # =========================================================================
    # Status and Statistics
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        cooldown_remaining = 0.0
        if self._state.cooldown_until:
            remaining = (self._state.cooldown_until - datetime.now()).total_seconds()
            cooldown_remaining = max(0, remaining / 60)

        return {
            "kill_switch_active": self._state.kill_switch_active,
            "daily_blocked": self._state.daily_blocked,
            "cooldown_active": self._state.cooldown_until is not None,
            "cooldown_remaining_minutes": round(cooldown_remaining, 1),
            "trade_count_today": self._state.trade_count_today,
            "trades_remaining": max(0, self._limits.max_trades_per_day - self._state.trade_count_today),
            "daily_pnl_pct": round(self._state.daily_pnl_pct, 4),
            "consecutive_losses": self._state.consecutive_losses,
            "current_drawdown_pct": round(self._state.current_drawdown_pct, 4),
            "total_exposure": round(self._state.total_exposure, 2),
            "limits": {
                "max_drawdown_pct": self._limits.max_drawdown_pct,
                "max_daily_loss_pct": self._limits.max_daily_loss_pct,
                "max_trades_per_day": self._limits.max_trades_per_day,
                "cooldown_after_losses": self._limits.cooldown_after_losses,
                "cooldown_minutes": self._limits.cooldown_minutes,
                "enable_short": self._limits.enable_short,
            },
            "rules": [r.name for r in self._rules],
            "current_day": self._state.current_day.isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    @property
    def is_trading_allowed(self) -> bool:
        """Quick check if trading is allowed."""
        return (
            not self._state.kill_switch_active
            and not self._state.daily_blocked
            and (
                self._state.cooldown_until is None
                or datetime.now() >= self._state.cooldown_until
            )
            and self._state.trade_count_today < self._limits.max_trades_per_day
        )

    @property
    def limits(self) -> RiskLimits:
        """Get current risk limits."""
        return self._limits

    @property
    def state(self) -> RiskState:
        """Get current risk state."""
        return self._state

    # =========================================================================
    # Rule Management
    # =========================================================================

    def add_rule(self, rule: IRiskRule) -> None:
        """Add a risk rule."""
        self._rules.append(rule)
        logger.info(f"Added risk rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a risk rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                del self._rules[i]
                logger.info(f"Removed risk rule: {rule_name}")
                return True
        return False

    def get_rules(self) -> List[str]:
        """Get list of rule names."""
        return [r.name for r in self._rules]

    def __repr__(self) -> str:
        return (
            f"RiskEnforcer(kill_switch={self._state.kill_switch_active}, "
            f"daily_blocked={self._state.daily_blocked}, "
            f"trades={self._state.trade_count_today}/{self._limits.max_trades_per_day})"
        )
