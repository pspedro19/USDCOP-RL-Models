"""
Risk Check Implementations
==========================

Concrete implementations of IRiskCheck following Chain of Responsibility pattern.
Each check handles ONE specific risk condition.

Order ranges:
- 0-9: Signal-level checks (HOLD passthrough)
- 10-19: Time-based checks (trading hours)
- 20-29: System-level checks (circuit breaker)
- 30-39: Cooldown checks
- 40-49: Confidence checks
- 50-69: Risk limit checks (daily loss, drawdown, etc.)
- 70-79: Trade count checks
- 80+: Custom checks

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

from .circuit_breaker_check import CircuitBreakerCheck
from .confidence_check import ConfidenceCheck
from .consecutive_losses_check import ConsecutiveLossesCheck
from .cooldown_check import CooldownCheck
from .daily_loss_check import DailyLossLimitCheck
from .drawdown_check import DrawdownCheck
from .hold_signal_check import HoldSignalCheck
from .max_trades_check import MaxTradesCheck
from .risk_check_chain import RiskCheckChain
from .trading_hours_check import TradingHoursCheck

__all__ = [
    # Individual checks
    'HoldSignalCheck',
    'TradingHoursCheck',
    'CircuitBreakerCheck',
    'CooldownCheck',
    'ConfidenceCheck',
    'DailyLossLimitCheck',
    'DrawdownCheck',
    'ConsecutiveLossesCheck',
    'MaxTradesCheck',
    # Chain orchestrator
    'RiskCheckChain',
]
