"""
Risk Manager & Circuit Breaker
==============================

Production-grade risk management system for trading signals.
Implements multiple layers of protection:

1. Daily loss limits
2. Drawdown protection
3. Consecutive loss limits
4. Confidence thresholds
5. Trading hours enforcement
6. Cooldown periods
7. Position sizing

All state is persisted in Redis for reliability.
"""

import os
import time
import logging
from typing import Optional, Dict, Tuple, Any, List
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import json
import pytz

try:
    import redis
except ImportError:
    redis = None
    logging.warning("redis not installed. Install with: pip install redis")

from .config import MLOpsConfig, get_config, RiskLimits, TradingHours, SignalType

logger = logging.getLogger(__name__)


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
class RiskCheckResult:
    """Result of a risk check."""
    approved: bool
    status: RiskStatus
    original_signal: SignalType
    adjusted_signal: SignalType
    confidence: float
    daily_stats: DailyStats
    risk_metrics: Dict[str, Any]
    message: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "status": self.status.value,
            "original_signal": self.original_signal.value,
            "adjusted_signal": self.adjusted_signal.value,
            "confidence": self.confidence,
            "daily_stats": self.daily_stats.to_dict(),
            "risk_metrics": self.risk_metrics,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class RiskManager:
    """
    Production risk management system.

    Features:
    - Real-time risk monitoring
    - Circuit breaker implementation
    - Position sizing recommendations
    - Trading hours enforcement
    - Cooldown management

    Usage:
        risk_manager = RiskManager(config)

        # Check if signal should be executed
        result = risk_manager.check_signal(signal, confidence)

        if result.approved:
            execute_trade(result.adjusted_signal)
        else:
            logger.warning(f"Signal blocked: {result.status}")

        # Update after trade completion
        risk_manager.update_trade_result(pnl=100.0, is_win=True)
    """

    # Redis key prefixes
    KEY_DAILY_STATS = "risk:daily_stats"
    KEY_TRADE_LOG = "risk:trade_log"
    KEY_CIRCUIT_BREAKER = "risk:circuit_breaker"
    KEY_COOLDOWN = "risk:cooldown"

    def __init__(
        self,
        config: Optional[MLOpsConfig] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        self.config = config or get_config()
        self.limits = self.config.risk_limits
        self.trading_hours = self.config.trading_hours
        self.timezone = pytz.timezone(self.trading_hours.timezone)

        # Initialize Redis
        if redis_client:
            self.redis = redis_client
        elif redis:
            self.redis = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                db=self.config.redis.db,
                password=self.config.redis.password,
                socket_timeout=self.config.redis.socket_timeout,
                decode_responses=True,
            )
        else:
            self.redis = None
            logger.warning("Redis not available. Using in-memory state (not recommended for production)")

        self._memory_stats: Dict[str, DailyStats] = {}

    def _get_today(self) -> str:
        """Get today's date in Colombia timezone."""
        return datetime.now(self.timezone).strftime("%Y-%m-%d")

    def _get_now(self) -> datetime:
        """Get current time in Colombia timezone."""
        return datetime.now(self.timezone)

    def _get_stats_key(self, date_str: Optional[str] = None) -> str:
        """Get Redis key for daily stats."""
        date_str = date_str or self._get_today()
        return f"{self.KEY_DAILY_STATS}:{date_str}"

    def get_daily_stats(self, date_str: Optional[str] = None) -> DailyStats:
        """Get daily trading statistics."""
        date_str = date_str or self._get_today()

        if self.redis:
            key = self._get_stats_key(date_str)
            data = self.redis.hgetall(key)

            if data:
                return DailyStats(
                    date=date_str,
                    pnl=float(data.get("pnl", 0)),
                    pnl_percent=float(data.get("pnl_percent", 0)),
                    peak_pnl=float(data.get("peak_pnl", 0)),
                    drawdown=float(data.get("drawdown", 0)),
                    trades_count=int(data.get("trades_count", 0)),
                    winning_trades=int(data.get("winning_trades", 0)),
                    losing_trades=int(data.get("losing_trades", 0)),
                    consecutive_losses=int(data.get("consecutive_losses", 0)),
                    consecutive_wins=int(data.get("consecutive_wins", 0)),
                    last_trade_time=data.get("last_trade_time"),
                    circuit_breaker_triggered=data.get("circuit_breaker_triggered", "false") == "true",
                    circuit_breaker_reason=data.get("circuit_breaker_reason"),
                )

        # In-memory fallback
        if date_str in self._memory_stats:
            return self._memory_stats[date_str]

        return DailyStats(date=date_str)

    def _save_daily_stats(self, stats: DailyStats):
        """Save daily statistics."""
        if self.redis:
            key = self._get_stats_key(stats.date)
            pipe = self.redis.pipeline()

            pipe.hset(key, mapping={
                "pnl": stats.pnl,
                "pnl_percent": stats.pnl_percent,
                "peak_pnl": stats.peak_pnl,
                "drawdown": stats.drawdown,
                "trades_count": stats.trades_count,
                "winning_trades": stats.winning_trades,
                "losing_trades": stats.losing_trades,
                "consecutive_losses": stats.consecutive_losses,
                "consecutive_wins": stats.consecutive_wins,
                "last_trade_time": stats.last_trade_time or "",
                "circuit_breaker_triggered": "true" if stats.circuit_breaker_triggered else "false",
                "circuit_breaker_reason": stats.circuit_breaker_reason or "",
            })

            # Expire after 30 days
            pipe.expire(key, 86400 * 30)
            pipe.execute()
        else:
            self._memory_stats[stats.date] = stats

    def is_trading_hours(self) -> Tuple[bool, str]:
        """Check if current time is within trading hours."""
        now = self._get_now()
        current_time = now.time()
        current_weekday = now.weekday()

        if current_weekday not in self.trading_hours.trading_days:
            return False, f"Not a trading day (weekday={current_weekday})"

        if current_time < self.trading_hours.start_time:
            return False, f"Before market open ({self.trading_hours.start_time})"

        if current_time > self.trading_hours.end_time:
            return False, f"After market close ({self.trading_hours.end_time})"

        return True, "Within trading hours"

    def is_cooldown_active(self) -> Tuple[bool, Optional[int]]:
        """Check if cooldown period is active."""
        if not self.redis:
            return False, None

        cooldown_key = f"{self.KEY_COOLDOWN}:{self._get_today()}"
        ttl = self.redis.ttl(cooldown_key)

        if ttl > 0:
            return True, ttl

        return False, None

    def _set_cooldown(self, seconds: int, reason: str):
        """Set cooldown period."""
        if self.redis:
            cooldown_key = f"{self.KEY_COOLDOWN}:{self._get_today()}"
            self.redis.setex(cooldown_key, seconds, reason)
            logger.info(f"Cooldown set for {seconds}s: {reason}")

    def is_circuit_breaker_active(self) -> Tuple[bool, Optional[str]]:
        """Check if circuit breaker is triggered."""
        stats = self.get_daily_stats()
        return stats.circuit_breaker_triggered, stats.circuit_breaker_reason

    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker."""
        stats = self.get_daily_stats()
        stats.circuit_breaker_triggered = True
        stats.circuit_breaker_reason = reason
        self._save_daily_stats(stats)

        # Log circuit breaker event
        if self.redis:
            event = {
                "timestamp": self._get_now().isoformat(),
                "reason": reason,
                "stats": stats.to_dict(),
            }
            self.redis.lpush(f"{self.KEY_CIRCUIT_BREAKER}:log", json.dumps(event))
            self.redis.ltrim(f"{self.KEY_CIRCUIT_BREAKER}:log", 0, 999)

        logger.warning(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")

    def check_signal(
        self,
        signal: SignalType,
        confidence: float,
        enforce_trading_hours: bool = True
    ) -> RiskCheckResult:
        """
        Check if a trading signal should be executed.

        Args:
            signal: The trading signal (BUY/SELL/HOLD)
            confidence: Model confidence (0-1)
            enforce_trading_hours: Whether to enforce trading hours

        Returns:
            RiskCheckResult with approval status
        """
        timestamp = self._get_now().isoformat()
        stats = self.get_daily_stats()

        # Build risk metrics
        risk_metrics = {
            "daily_pnl_percent": stats.pnl_percent,
            "current_drawdown": stats.drawdown,
            "consecutive_losses": stats.consecutive_losses,
            "trades_today": stats.trades_count,
            "win_rate": stats.win_rate,
            "limits": self.limits.to_dict(),
        }

        # Helper to create result
        def make_result(
            approved: bool,
            status: RiskStatus,
            adjusted_signal: SignalType = None,
            message: str = ""
        ) -> RiskCheckResult:
            return RiskCheckResult(
                approved=approved,
                status=status,
                original_signal=signal,
                adjusted_signal=adjusted_signal or (signal if approved else SignalType.HOLD),
                confidence=confidence,
                daily_stats=stats,
                risk_metrics=risk_metrics,
                message=message,
                timestamp=timestamp,
            )

        # 1. HOLD signals always pass
        if signal == SignalType.HOLD:
            return make_result(True, RiskStatus.HOLD_SIGNAL, message="HOLD signal - no action needed")

        # 2. Check trading hours
        if enforce_trading_hours:
            is_trading, hours_msg = self.is_trading_hours()
            if not is_trading:
                return make_result(False, RiskStatus.OUTSIDE_TRADING_HOURS, message=hours_msg)

        # 3. Check circuit breaker
        cb_active, cb_reason = self.is_circuit_breaker_active()
        if cb_active:
            return make_result(
                False,
                RiskStatus.CIRCUIT_BREAKER_ACTIVE,
                message=f"Circuit breaker active: {cb_reason}"
            )

        # 4. Check cooldown
        cooldown_active, cooldown_ttl = self.is_cooldown_active()
        if cooldown_active:
            return make_result(
                False,
                RiskStatus.COOLDOWN_ACTIVE,
                message=f"Cooldown active: {cooldown_ttl}s remaining"
            )

        # 5. Check confidence threshold
        if confidence < self.limits.min_confidence:
            return make_result(
                False,
                RiskStatus.LOW_CONFIDENCE,
                message=f"Confidence {confidence:.2%} below threshold {self.limits.min_confidence:.2%}"
            )

        # 6. Check daily loss limit
        if stats.pnl_percent <= self.limits.max_daily_loss:
            self._trigger_circuit_breaker(f"Daily loss limit reached: {stats.pnl_percent:.2%}")
            return make_result(
                False,
                RiskStatus.DAILY_LOSS_LIMIT,
                message=f"Daily loss {stats.pnl_percent:.2%} exceeds limit {self.limits.max_daily_loss:.2%}"
            )

        # 7. Check drawdown limit
        if stats.drawdown <= self.limits.max_drawdown:
            self._trigger_circuit_breaker(f"Max drawdown reached: {stats.drawdown:.2%}")
            return make_result(
                False,
                RiskStatus.MAX_DRAWDOWN,
                message=f"Drawdown {stats.drawdown:.2%} exceeds limit {self.limits.max_drawdown:.2%}"
            )

        # 8. Check consecutive losses
        if stats.consecutive_losses >= self.limits.max_consecutive_losses:
            self._set_cooldown(self.limits.cooldown_after_loss, "Consecutive losses limit")
            return make_result(
                False,
                RiskStatus.CONSECUTIVE_LOSSES,
                message=f"{stats.consecutive_losses} consecutive losses (limit: {self.limits.max_consecutive_losses})"
            )

        # 9. Check max trades per day
        if stats.trades_count >= self.limits.max_trades_per_day:
            return make_result(
                False,
                RiskStatus.MAX_TRADES_REACHED,
                message=f"Max trades ({self.limits.max_trades_per_day}) reached for today"
            )

        # All checks passed
        return make_result(
            True,
            RiskStatus.APPROVED,
            adjusted_signal=signal,
            message="Signal approved for execution"
        )

    def update_trade_result(
        self,
        pnl: float,
        pnl_percent: float,
        is_win: bool,
        trade_id: Optional[str] = None
    ):
        """
        Update statistics after a trade completes.

        Args:
            pnl: Profit/loss in currency
            pnl_percent: Profit/loss as percentage
            is_win: Whether the trade was profitable
            trade_id: Optional trade identifier
        """
        stats = self.get_daily_stats()
        now = self._get_now()

        # Update P&L
        stats.pnl += pnl
        stats.pnl_percent += pnl_percent

        # Update peak and drawdown
        if stats.pnl_percent > stats.peak_pnl:
            stats.peak_pnl = stats.pnl_percent

        current_drawdown = stats.pnl_percent - stats.peak_pnl
        stats.drawdown = min(stats.drawdown, current_drawdown)

        # Update trade counts
        stats.trades_count += 1
        stats.last_trade_time = now.isoformat()

        if is_win:
            stats.winning_trades += 1
            stats.consecutive_wins += 1
            stats.consecutive_losses = 0
        else:
            stats.losing_trades += 1
            stats.consecutive_losses += 1
            stats.consecutive_wins = 0

            # Set cooldown after loss
            if stats.consecutive_losses >= 2:
                self._set_cooldown(
                    self.limits.cooldown_after_loss,
                    f"Loss cooldown ({stats.consecutive_losses} consecutive)"
                )

        self._save_daily_stats(stats)

        # Log trade
        if self.redis:
            trade_log = {
                "timestamp": now.isoformat(),
                "trade_id": trade_id,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "is_win": is_win,
                "cumulative_pnl": stats.pnl,
                "cumulative_pnl_percent": stats.pnl_percent,
            }
            self.redis.lpush(f"{self.KEY_TRADE_LOG}:{stats.date}", json.dumps(trade_log))
            self.redis.ltrim(f"{self.KEY_TRADE_LOG}:{stats.date}", 0, 999)

        logger.info(
            f"Trade result: {'WIN' if is_win else 'LOSS'} "
            f"P&L: {pnl_percent:+.2%}, Cumulative: {stats.pnl_percent:+.2%}"
        )

    def get_position_size_recommendation(
        self,
        confidence: float,
        current_capital: float
    ) -> Dict[str, Any]:
        """
        Get recommended position size based on risk parameters.

        Args:
            confidence: Model confidence
            current_capital: Current trading capital

        Returns:
            Position sizing recommendation
        """
        stats = self.get_daily_stats()

        # Base position size
        base_size = self.limits.max_position_size

        # Adjust for confidence
        if confidence >= self.limits.high_confidence_threshold:
            confidence_multiplier = 1.0
        else:
            confidence_multiplier = confidence / self.limits.high_confidence_threshold

        # Adjust for drawdown (reduce size as drawdown increases)
        drawdown_abs = abs(stats.drawdown)
        drawdown_multiplier = max(0.5, 1.0 - (drawdown_abs / abs(self.limits.max_drawdown)))

        # Adjust for consecutive losses
        loss_multiplier = max(0.5, 1.0 - (stats.consecutive_losses * 0.1))

        # Calculate final size
        recommended_size = base_size * confidence_multiplier * drawdown_multiplier * loss_multiplier
        recommended_amount = current_capital * recommended_size

        return {
            "recommended_size_percent": recommended_size,
            "recommended_amount": recommended_amount,
            "base_size": base_size,
            "adjustments": {
                "confidence_multiplier": confidence_multiplier,
                "drawdown_multiplier": drawdown_multiplier,
                "loss_multiplier": loss_multiplier,
            },
            "current_stats": {
                "drawdown": stats.drawdown,
                "consecutive_losses": stats.consecutive_losses,
            }
        }

    def reset_daily_stats(self):
        """Reset daily statistics (typically called at start of trading day)."""
        today = self._get_today()
        stats = DailyStats(date=today)
        self._save_daily_stats(stats)
        logger.info(f"Daily stats reset for {today}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        stats = self.get_daily_stats()
        is_trading, hours_msg = self.is_trading_hours()
        cooldown_active, cooldown_ttl = self.is_cooldown_active()
        cb_active, cb_reason = self.is_circuit_breaker_active()

        return {
            "timestamp": self._get_now().isoformat(),
            "trading_status": {
                "is_trading_hours": is_trading,
                "hours_message": hours_msg,
                "cooldown_active": cooldown_active,
                "cooldown_ttl": cooldown_ttl,
                "circuit_breaker_active": cb_active,
                "circuit_breaker_reason": cb_reason,
            },
            "daily_stats": stats.to_dict(),
            "limits": self.limits.to_dict(),
            "risk_utilization": {
                "daily_loss_utilized": abs(stats.pnl_percent / self.limits.max_daily_loss) if stats.pnl_percent < 0 else 0,
                "drawdown_utilized": abs(stats.drawdown / self.limits.max_drawdown) if stats.drawdown < 0 else 0,
                "trades_utilized": stats.trades_count / self.limits.max_trades_per_day,
            }
        }


# Global risk manager instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get or create global risk manager."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager


def initialize_risk_manager(
    config: Optional[MLOpsConfig] = None,
    redis_client: Optional[redis.Redis] = None
) -> RiskManager:
    """Initialize global risk manager."""
    global _risk_manager
    _risk_manager = RiskManager(config, redis_client)
    return _risk_manager
