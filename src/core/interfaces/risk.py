"""
Risk Management Interfaces
==========================

Defines abstract interfaces for the risk management subsystem.
Implements Chain of Responsibility pattern for risk checks.

Author: Trading Team
Version: 2.0.0
Date: 2025-01-14
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class RiskStatus(str, Enum):
    """Risk check status codes."""
    APPROVED = "APPROVED"
    HOLD_SIGNAL = "HOLD_SIGNAL"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MAX_TRADES_REACHED = "MAX_TRADES_REACHED"
    OUTSIDE_TRADING_HOURS = "OUTSIDE_TRADING_HOURS"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    CIRCUIT_BREAKER_ACTIVE = "CIRCUIT_BREAKER_ACTIVE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class RiskContext:
    """Context passed through risk check chain."""
    signal: str  # BUY, SELL, HOLD
    confidence: float
    daily_pnl_percent: float = 0.0
    current_drawdown: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    trades_today: int = 0
    win_rate: float = 0.0
    last_trade_time: Optional[str] = None
    enforce_trading_hours: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    approved: bool
    status: RiskStatus
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_continue_chain(self) -> bool:
        """Whether to continue to next check in chain."""
        return self.approved or self.status == RiskStatus.HOLD_SIGNAL


# =============================================================================
# Chain of Responsibility: Individual Risk Checks
# =============================================================================

class IRiskCheck(ABC):
    """
    Interface for individual risk checks.

    Chain of Responsibility Pattern: Each check can approve, reject, or pass.
    Single Responsibility: Each implementation handles ONE type of check.
    Open/Closed: Add new checks without modifying existing code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get check name for logging/debugging."""
        pass

    @property
    @abstractmethod
    def order(self) -> int:
        """
        Get execution order (lower = earlier).

        Suggested order ranges:
        - 0-9: Signal-level checks (HOLD passthrough)
        - 10-19: Time-based checks (trading hours)
        - 20-29: System-level checks (circuit breaker)
        - 30-39: Cooldown checks
        - 40-49: Confidence checks
        - 50-69: Risk limit checks (daily loss, drawdown, etc.)
        - 70-79: Trade count checks
        - 80+: Custom checks
        """
        pass

    @abstractmethod
    def check(self, context: RiskContext) -> RiskCheckResult:
        """
        Execute the risk check.

        Args:
            context: Current risk context with signal and stats

        Returns:
            RiskCheckResult indicating approval or rejection
        """
        pass


class ITradingHoursChecker(ABC):
    """
    Interface for trading hours validation.

    Single Responsibility: Validate trading time windows.
    """

    @abstractmethod
    def is_trading_hours(self) -> Tuple[bool, str]:
        """
        Check if current time is within trading hours.

        Returns:
            Tuple of (is_within_hours, message)
        """
        pass

    @property
    @abstractmethod
    def timezone(self) -> str:
        """Get configured timezone."""
        pass


class ICircuitBreaker(ABC):
    """
    Interface for circuit breaker operations.

    Single Responsibility: Manage circuit breaker state.
    """

    @abstractmethod
    def is_active(self) -> Tuple[bool, Optional[str]]:
        """
        Check if circuit breaker is active.

        Returns:
            Tuple of (is_active, reason)
        """
        pass

    @abstractmethod
    def trigger(self, reason: str) -> None:
        """
        Trigger the circuit breaker.

        Args:
            reason: Why the circuit breaker was triggered
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the circuit breaker."""
        pass


class ICooldownManager(ABC):
    """
    Interface for cooldown period management.

    Single Responsibility: Manage trading cooldowns.
    """

    @abstractmethod
    def is_active(self) -> Tuple[bool, Optional[int]]:
        """
        Check if cooldown is active.

        Returns:
            Tuple of (is_active, seconds_remaining)
        """
        pass

    @abstractmethod
    def set_cooldown(self, seconds: int, reason: str) -> None:
        """
        Set cooldown period.

        Args:
            seconds: Duration in seconds
            reason: Why cooldown was set
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear active cooldown."""
        pass


class IPositionSizer(ABC):
    """
    Interface for position sizing calculations.

    Single Responsibility: Calculate optimal position sizes.
    """

    @abstractmethod
    def calculate_size(
        self,
        signal: str,
        confidence: float,
        risk_context: RiskContext
    ) -> float:
        """
        Calculate recommended position size.

        Args:
            signal: Trading signal
            confidence: Model confidence
            risk_context: Current risk context

        Returns:
            Recommended position size (0.0 to 1.0)
        """
        pass

    @property
    @abstractmethod
    def max_position_size(self) -> float:
        """Get maximum allowed position size."""
        pass


class IRiskManager(ABC):
    """
    Combined interface for risk management.

    Facade Pattern: Unified interface for all risk operations.
    Uses IRiskCheck implementations through Chain of Responsibility.
    """

    @abstractmethod
    def check_signal(
        self,
        signal: str,
        confidence: float,
        enforce_trading_hours: bool = True
    ) -> 'FullRiskCheckResult':
        """
        Check if a trading signal should be executed.

        Args:
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Model confidence
            enforce_trading_hours: Whether to check trading hours

        Returns:
            Full risk check result with all details
        """
        pass

    @abstractmethod
    def update_trade_result(
        self,
        pnl: float,
        pnl_percent: float,
        is_win: bool,
        trade_id: Optional[str] = None
    ) -> None:
        """Update statistics after trade completion."""
        pass

    @abstractmethod
    def get_daily_stats(self) -> 'DailyStats':
        """Get current daily statistics."""
        pass


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: str
    pnl: float = 0.0
    pnl_percent: float = 0.0
    peak_pnl: float = 0.0
    drawdown: float = 0.0
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    last_trade_time: Optional[str] = None
    circuit_breaker_triggered: bool = False
    circuit_breaker_reason: Optional[str] = None

    @property
    def win_rate(self) -> float:
        if self.trades_count == 0:
            return 0.0
        return self.winning_trades / self.trades_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "peak_pnl": self.peak_pnl,
            "drawdown": self.drawdown,
            "trades_count": self.trades_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "win_rate": self.win_rate,
            "last_trade_time": self.last_trade_time,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "circuit_breaker_reason": self.circuit_breaker_reason,
        }


@dataclass
class FullRiskCheckResult:
    """Complete result of risk check including all context."""
    approved: bool
    status: RiskStatus
    original_signal: str
    adjusted_signal: str
    confidence: float
    daily_stats: DailyStats
    risk_metrics: Dict[str, Any]
    message: str
    timestamp: str
    checks_passed: list = field(default_factory=list)
    check_that_failed: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "status": self.status.value,
            "original_signal": self.original_signal,
            "adjusted_signal": self.adjusted_signal,
            "confidence": self.confidence,
            "daily_stats": self.daily_stats.to_dict(),
            "risk_metrics": self.risk_metrics,
            "message": self.message,
            "timestamp": self.timestamp,
            "checks_passed": self.checks_passed,
            "check_that_failed": self.check_that_failed,
        }
