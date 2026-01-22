"""
USD/COP Trading System - Trading Module
========================================

Trading components following SOLID principles with clean separation of concerns.

Components:
- PaperTrader: Legacy simulated trade execution (maintained for compatibility)
- PositionTracker: Single Responsibility - position management
- OrderExecutor: Single Responsibility - order execution
- RiskEnforcer: Single Responsibility - risk validation
- TradeExecutorFactory: Factory pattern for executor creation
- Trade States: State Pattern for trade lifecycle management

Usage:
    # Modern approach with separated components
    from src.trading import PositionTracker, OrderExecutor, RiskEnforcer

    position_tracker = PositionTracker()
    risk_enforcer = RiskEnforcer()
    executor = OrderExecutor(position_tracker, risk_validator=risk_enforcer)

    result = executor.execute_signal(
        model_id="ppo_v1",
        signal=SignalType.LONG,
        current_price=4250.50
    )

    # Legacy approach (still supported)
    from src.trading import PaperTrader

    trader = PaperTrader(initial_capital=10000.0)
    trade = trader.execute_signal("ppo_v1", "LONG", 4250.50)

    # Factory pattern
    from src.trading.factories import TradeExecutorFactory

    executor = TradeExecutorFactory.create("paper", config, context)

Author: USD/COP Trading System
Version: 2.0.0
"""

# Legacy Paper Trading (maintained for backward compatibility)
from .paper_trader import (
    PaperTrade,
    PaperTrader,
    TradeDirection,
)

# State Pattern for trade lifecycle
from .states import (
    TradeState,
    PendingState,
    OpenState,
    ClosingState,
    ClosedState,
    Trade,
)

# New SOLID Components (P2 Clean Code)
from .position_tracker import (
    Position,
    PositionDirection,
    PositionStatus,
    PositionTracker,
)

from .order_executor import (
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    SignalType,
    ExecutionResult,
    ExecutorConfig,
    OrderExecutor,
)

from .risk_enforcer import (
    RiskDecision,
    RiskReason,
    RiskCheckResult,
    RiskLimits,
    RiskState,
    RiskEnforcer,
)

# Trading Flags (Week 1) - Re-export from SSOT (src.config.trading_flags)
# NOTE: The canonical TradingFlags is in src.config.trading_flags
# This re-export maintains backward compatibility with existing imports
from src.config.trading_flags import (
    TradingFlags,
    TradingMode,
    get_trading_flags,
    reload_trading_flags,
    reset_trading_flags_cache,
    activate_kill_switch,
    deactivate_kill_switch,
    is_live_trading_enabled,
    is_paper_trading_enabled,
    is_kill_switch_active,
)

# Backward compatibility aliases
def load_trading_flags(force_reload: bool = False) -> TradingFlags:
    """Load trading flags (backward compat alias)."""
    if force_reload:
        return reload_trading_flags()
    return get_trading_flags()

def reload_flags() -> TradingFlags:
    """Reload flags (backward compat alias)."""
    return reload_trading_flags()

def clear_flags_cache() -> None:
    """Clear flags cache (backward compat alias)."""
    reset_trading_flags_cache()

def validate_for_production():
    """Validate for production (backward compat alias)."""
    return get_trading_flags().validate_for_production()

def get_current_flags() -> TradingFlags:
    """Get current flags (backward compat alias)."""
    return get_trading_flags()

def require_trading_enabled(func):
    """Decorator to check trading is enabled (backward compat)."""
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        flags = get_trading_flags()
        if not flags.can_execute_paper_trade():
            raise TradingDisabledError("Trading is disabled by flags")
        return func(*args, **kwargs)
    return wrapper

class TradingDisabledError(Exception):
    """Raised when trading is attempted while disabled."""
    pass

class TradingFlagsError(Exception):
    """Raised when there is an error with trading flags configuration."""
    pass

__version__ = "2.1.0"

__all__ = [
    # Legacy Paper Trading
    "PaperTrade",
    "PaperTrader",
    "TradeDirection",

    # State Pattern
    "TradeState",
    "PendingState",
    "OpenState",
    "ClosingState",
    "ClosedState",
    "Trade",

    # Position Tracking (SRP)
    "Position",
    "PositionDirection",
    "PositionStatus",
    "PositionTracker",

    # Order Execution (SRP)
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "SignalType",
    "ExecutionResult",
    "ExecutorConfig",
    "OrderExecutor",

    # Risk Enforcement (SRP)
    "RiskDecision",
    "RiskReason",
    "RiskCheckResult",
    "RiskLimits",
    "RiskState",
    "RiskEnforcer",

    # Trading Flags (Re-exported from src.config.trading_flags SSOT)
    "TradingFlags",
    "TradingMode",
    "get_trading_flags",
    "reload_trading_flags",
    "reset_trading_flags_cache",
    "activate_kill_switch",
    "deactivate_kill_switch",
    "is_live_trading_enabled",
    "is_paper_trading_enabled",
    "is_kill_switch_active",
    # Backward compatibility aliases
    "load_trading_flags",
    "reload_flags",
    "clear_flags_cache",
    "validate_for_production",
    "get_current_flags",
    "require_trading_enabled",
    "TradingDisabledError",
    "TradingFlagsError",
]
