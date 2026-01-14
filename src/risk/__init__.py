"""
Risk Management Module
======================

Safety layer for USD/COP trading system.
Provides kill switches, daily limits, and cooldown mechanisms
to prevent catastrophic losses.

Chain of Responsibility Pattern:
- Individual risk checks (HoldSignalCheck, TradingHoursCheck, etc.)
- RiskCheckChain orchestrator

Author: Trading Team
Version: 2.0.0
Date: 2025-01-14
"""

from .risk_manager import RiskManager, RiskLimits

# Chain of Responsibility pattern (Phase 3)
from .checks import (
    HoldSignalCheck,
    TradingHoursCheck,
    CircuitBreakerCheck,
    CooldownCheck,
    ConfidenceCheck,
    DailyLossLimitCheck,
    DrawdownCheck,
    ConsecutiveLossesCheck,
    MaxTradesCheck,
    RiskCheckChain,
)

__all__ = [
    # Legacy interface
    'RiskManager',
    'RiskLimits',
    # Chain of Responsibility
    'HoldSignalCheck',
    'TradingHoursCheck',
    'CircuitBreakerCheck',
    'CooldownCheck',
    'ConfidenceCheck',
    'DailyLossLimitCheck',
    'DrawdownCheck',
    'ConsecutiveLossesCheck',
    'MaxTradesCheck',
    'RiskCheckChain',
]
