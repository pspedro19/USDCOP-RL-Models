"""
Risk Management Module
======================

Safety layer for USD/COP trading system.
Provides kill switches, daily limits, and cooldown mechanisms
to prevent catastrophic losses.

Design Patterns:
- Chain of Responsibility: Individual risk checks
- Command Pattern: Encapsulated risk operations with undo support

Author: Trading Team
Version: 2.1.0
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

# Command Pattern for risk operations
from .commands import (
    Command,
    CommandResult,
    CommandInvoker,
    TriggerCircuitBreakerCommand,
    SetCooldownCommand,
    ClearCooldownCommand,
    ResetKillSwitchCommand,
    UpdateRiskLimitsCommand,
    BlockTradingCommand,
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
    # Command Pattern
    'Command',
    'CommandResult',
    'CommandInvoker',
    'TriggerCircuitBreakerCommand',
    'SetCooldownCommand',
    'ClearCooldownCommand',
    'ResetKillSwitchCommand',
    'UpdateRiskLimitsCommand',
    'BlockTradingCommand',
]
