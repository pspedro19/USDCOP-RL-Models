"""
Risk Manager - Safety Layer for USD/COP Trading
================================================

This module implements a critical safety layer that validates trading signals
before execution to prevent catastrophic losses. It provides:

- Kill switch when drawdown exceeds maximum threshold
- Daily loss limit that blocks trading for the day
- Trade count limit per day
- Cooldown period after consecutive losses
- Automatic daily reset

Usage:
    from src.risk import RiskManager, RiskLimits

    # Initialize with custom limits
    limits = RiskLimits(
        max_drawdown_pct=15.0,
        max_daily_loss_pct=5.0,
        max_trades_per_day=20
    )
    risk_manager = RiskManager(limits)

    # Validate signal before execution
    allowed, reason = risk_manager.validate_signal("long", current_drawdown_pct=8.5)
    if allowed:
        execute_trade()
    else:
        logger.warning(f"Trade blocked: {reason}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
import logging

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handler if not exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class RiskLimits:
    """
    Configuration for risk management thresholds.

    Attributes:
        max_drawdown_pct: Maximum portfolio drawdown before kill switch (default: 15%)
        max_daily_loss_pct: Maximum daily loss before blocking trading (default: 5%)
        max_trades_per_day: Maximum number of trades per day (default: 20)
        cooldown_after_losses: Number of consecutive losses before cooldown (default: 3)
        cooldown_minutes: Duration of cooldown period in minutes (default: 30)
    """
    max_drawdown_pct: float = 15.0      # Kill switch trigger
    max_daily_loss_pct: float = 5.0     # Stop trading today
    max_trades_per_day: int = 20        # Pause trading
    cooldown_after_losses: int = 3      # Consecutive losses before cooldown
    cooldown_minutes: int = 30          # Cooldown duration


@dataclass
class TradeRecord:
    """Record of a completed trade for tracking."""
    timestamp: datetime
    pnl_pct: float
    signal: str


class RiskManager:
    """
    Safety layer for validating signals before execution.

    This class implements multiple safety mechanisms to prevent catastrophic losses:

    1. Kill Switch: Triggered when portfolio drawdown exceeds max_drawdown_pct.
       Once triggered, ALL trading is halted until manual intervention.

    2. Daily Loss Limit: When cumulative daily losses exceed max_daily_loss_pct,
       trading is blocked for the remainder of the day.

    3. Trade Limit: Limits the number of trades per day to prevent overtrading.

    4. Cooldown: After consecutive losses, enforces a waiting period before
       allowing new trades.

    All blocking events are logged at CRITICAL level for alerting.
    """

    def __init__(self, limits: RiskLimits = None):
        """
        Initialize the RiskManager with specified limits.

        Args:
            limits: RiskLimits configuration. Uses defaults if None.
        """
        self.limits = limits or RiskLimits()

        # State tracking
        self._kill_switch_active: bool = False
        self._daily_blocked: bool = False
        self._cooldown_until: Optional[datetime] = None

        # Daily counters
        self._trade_count_today: int = 0
        self._daily_pnl_pct: float = 0.0
        self._consecutive_losses: int = 0

        # Trade history for analysis
        self._trade_history: List[TradeRecord] = []

        # Track the current trading day
        self._current_day: datetime = datetime.now().date()

        logger.info(
            f"RiskManager initialized with limits: "
            f"max_drawdown={self.limits.max_drawdown_pct}%, "
            f"max_daily_loss={self.limits.max_daily_loss_pct}%, "
            f"max_trades={self.limits.max_trades_per_day}, "
            f"cooldown_after={self.limits.cooldown_after_losses} losses, "
            f"cooldown_duration={self.limits.cooldown_minutes}min"
        )

    def validate_signal(
        self,
        signal: str,
        current_drawdown_pct: float
    ) -> Tuple[bool, str]:
        """
        Validate if a trading signal should be executed.

        This method checks all safety conditions before allowing a trade.
        Call this BEFORE executing any trade signal.

        Args:
            signal: Trading signal ("long", "short", "flat", "close")
            current_drawdown_pct: Current portfolio drawdown as percentage

        Returns:
            Tuple of (allowed: bool, reason: str)
            - allowed: True if trade is permitted, False otherwise
            - reason: Explanation for the decision

        Examples:
            >>> rm = RiskManager()
            >>> allowed, reason = rm.validate_signal("long", 5.0)
            >>> if allowed:
            ...     execute_trade()
        """
        # Check for day change and reset if needed
        self._check_daily_reset()

        # Always allow close/flat signals to exit positions
        if signal.lower() in ("close", "flat"):
            return True, "Exit signals always allowed for risk reduction"

        # 1. Kill Switch Check (most critical)
        if self._kill_switch_active:
            logger.critical(
                f"KILL SWITCH ACTIVE: Trade blocked. "
                f"Signal={signal}, Drawdown={current_drawdown_pct}%"
            )
            return False, "Kill switch active - trading halted"

        # Check if current drawdown triggers kill switch
        if current_drawdown_pct >= self.limits.max_drawdown_pct:
            self._kill_switch_active = True
            logger.critical(
                f"KILL SWITCH TRIGGERED: Drawdown {current_drawdown_pct}% "
                f">= limit {self.limits.max_drawdown_pct}%. "
                f"ALL TRADING HALTED!"
            )
            return False, f"Kill switch triggered - drawdown {current_drawdown_pct}% >= {self.limits.max_drawdown_pct}%"

        # 2. Daily Loss Limit Check
        if self._daily_blocked:
            logger.warning(
                f"Daily block active: Trade blocked. "
                f"Signal={signal}, Daily PnL={self._daily_pnl_pct}%"
            )
            return False, "Daily loss limit reached - trading blocked until tomorrow"

        # 3. Trade Count Limit Check
        if self._trade_count_today >= self.limits.max_trades_per_day:
            logger.warning(
                f"Trade limit reached: {self._trade_count_today} trades today. "
                f"Max allowed: {self.limits.max_trades_per_day}"
            )
            return False, f"Daily trade limit ({self.limits.max_trades_per_day}) reached"

        # 4. Cooldown Check
        if self._cooldown_until is not None:
            now = datetime.now()
            if now < self._cooldown_until:
                remaining = (self._cooldown_until - now).total_seconds() / 60
                logger.info(
                    f"Cooldown active: {remaining:.1f} minutes remaining. "
                    f"Trade blocked."
                )
                return False, f"Cooldown active - {remaining:.1f} minutes remaining"
            else:
                # Cooldown expired
                self._cooldown_until = None
                self._consecutive_losses = 0
                logger.info("Cooldown period ended. Trading resumed.")

        # All checks passed
        return True, "Trade allowed"

    def record_trade_result(self, pnl_pct: float, signal: str = "unknown"):
        """
        Record the result of a completed trade.

        This method updates internal state based on trade outcome:
        - Updates daily P&L tracking
        - Tracks consecutive losses for cooldown
        - Increments trade counter
        - Checks if daily loss limit is breached

        Call this AFTER each trade is closed with its P&L.

        Args:
            pnl_pct: Profit/Loss of the trade as percentage
                     Positive = profit, Negative = loss
            signal: The signal type that generated this trade

        Examples:
            >>> rm = RiskManager()
            >>> rm.record_trade_result(0.5)  # 0.5% profit
            >>> rm.record_trade_result(-0.3)  # 0.3% loss
        """
        # Check for day change
        self._check_daily_reset()

        # Record trade
        trade = TradeRecord(
            timestamp=datetime.now(),
            pnl_pct=pnl_pct,
            signal=signal
        )
        self._trade_history.append(trade)

        # Update counters
        self._trade_count_today += 1
        self._daily_pnl_pct += pnl_pct

        # Track consecutive losses
        if pnl_pct < 0:
            self._consecutive_losses += 1
            logger.info(
                f"Trade loss recorded: {pnl_pct:.2f}%. "
                f"Consecutive losses: {self._consecutive_losses}"
            )

            # Check for cooldown trigger
            if self._consecutive_losses >= self.limits.cooldown_after_losses:
                self._cooldown_until = datetime.now() + timedelta(
                    minutes=self.limits.cooldown_minutes
                )
                logger.warning(
                    f"COOLDOWN ACTIVATED: {self._consecutive_losses} consecutive losses. "
                    f"Trading paused until {self._cooldown_until.strftime('%H:%M:%S')}"
                )
        else:
            # Reset consecutive loss counter on profit
            if self._consecutive_losses > 0:
                logger.info(
                    f"Win recorded: {pnl_pct:.2f}%. "
                    f"Consecutive losses reset from {self._consecutive_losses} to 0"
                )
            self._consecutive_losses = 0

        # Check daily loss limit
        if self._daily_pnl_pct <= -self.limits.max_daily_loss_pct:
            self._daily_blocked = True
            logger.critical(
                f"DAILY LOSS LIMIT REACHED: Daily PnL {self._daily_pnl_pct:.2f}% "
                f"<= -{self.limits.max_daily_loss_pct}%. "
                f"Trading blocked for today!"
            )

        logger.debug(
            f"Trade recorded: PnL={pnl_pct:.2f}%, "
            f"Daily PnL={self._daily_pnl_pct:.2f}%, "
            f"Trades today={self._trade_count_today}"
        )

    def get_status(self) -> dict:
        """
        Get the current risk management status.

        Returns comprehensive status information for monitoring
        and dashboard display.

        Returns:
            dict with current risk status including:
            - kill_switch_active: Whether kill switch is triggered
            - daily_blocked: Whether daily trading is blocked
            - cooldown_active: Whether cooldown is in effect
            - cooldown_remaining_minutes: Time left in cooldown (if active)
            - trade_count_today: Number of trades executed today
            - trades_remaining: Trades allowed before hitting limit
            - daily_pnl_pct: Cumulative daily P&L percentage
            - consecutive_losses: Current streak of losing trades
            - limits: Current risk limit configuration
        """
        # Check for day change
        self._check_daily_reset()

        # Calculate cooldown remaining
        cooldown_remaining = 0.0
        if self._cooldown_until is not None:
            now = datetime.now()
            if now < self._cooldown_until:
                cooldown_remaining = (self._cooldown_until - now).total_seconds() / 60
            else:
                self._cooldown_until = None

        return {
            # Current state
            "kill_switch_active": self._kill_switch_active,
            "daily_blocked": self._daily_blocked,
            "cooldown_active": self._cooldown_until is not None,
            "cooldown_remaining_minutes": round(cooldown_remaining, 1),

            # Daily metrics
            "trade_count_today": self._trade_count_today,
            "trades_remaining": max(0, self.limits.max_trades_per_day - self._trade_count_today),
            "daily_pnl_pct": round(self._daily_pnl_pct, 4),
            "consecutive_losses": self._consecutive_losses,

            # Risk capacity
            "daily_loss_remaining_pct": round(
                self.limits.max_daily_loss_pct + self._daily_pnl_pct, 4
            ),

            # Configuration
            "limits": {
                "max_drawdown_pct": self.limits.max_drawdown_pct,
                "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                "max_trades_per_day": self.limits.max_trades_per_day,
                "cooldown_after_losses": self.limits.cooldown_after_losses,
                "cooldown_minutes": self.limits.cooldown_minutes
            },

            # Metadata
            "current_day": self._current_day.isoformat(),
            "last_updated": datetime.now().isoformat()
        }

    def reset_daily(self):
        """
        Reset daily counters and blocks.

        This is typically called automatically at day boundary,
        but can be called manually for testing or to force a reset.

        Note: This does NOT reset the kill switch, which requires
        manual intervention.
        """
        previous_pnl = self._daily_pnl_pct
        previous_trades = self._trade_count_today

        self._trade_count_today = 0
        self._daily_pnl_pct = 0.0
        self._daily_blocked = False
        self._consecutive_losses = 0
        self._cooldown_until = None
        self._current_day = datetime.now().date()

        logger.info(
            f"Daily reset completed. "
            f"Previous day: PnL={previous_pnl:.2f}%, Trades={previous_trades}. "
            f"New day: {self._current_day}"
        )

    def reset_kill_switch(self, confirm: bool = False) -> bool:
        """
        Manually reset the kill switch.

        This should only be called after careful review of the situation
        that triggered the kill switch. Requires explicit confirmation.

        Args:
            confirm: Must be True to actually reset

        Returns:
            bool: True if reset was performed, False otherwise
        """
        if not confirm:
            logger.warning(
                "Kill switch reset attempted without confirmation. "
                "Pass confirm=True to reset."
            )
            return False

        if self._kill_switch_active:
            self._kill_switch_active = False
            logger.critical(
                "KILL SWITCH MANUALLY RESET. "
                "Trading can resume. Review risk limits before continuing."
            )
            return True

        logger.info("Kill switch was not active. No reset needed.")
        return False

    def _check_daily_reset(self):
        """
        Check if the day has changed and reset if needed.

        Internal method called before any state checks to ensure
        daily counters are reset at day boundaries.
        """
        today = datetime.now().date()
        if today != self._current_day:
            logger.info(f"Day change detected: {self._current_day} -> {today}")
            self.reset_daily()

    def get_trade_history(self, limit: int = 100) -> List[dict]:
        """
        Get recent trade history.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade records as dictionaries
        """
        recent_trades = self._trade_history[-limit:] if self._trade_history else []
        return [
            {
                "timestamp": trade.timestamp.isoformat(),
                "pnl_pct": trade.pnl_pct,
                "signal": trade.signal
            }
            for trade in recent_trades
        ]
