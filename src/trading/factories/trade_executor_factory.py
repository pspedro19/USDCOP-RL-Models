"""
TradeExecutorFactory - Factory Pattern for Trade Executors
==========================================================

Creates appropriate trade executor instances based on execution mode.
Supports paper trading, live trading, and backtesting modes.

Design Patterns:
- Factory Pattern: Centralizes object creation
- Strategy Pattern: Executors are interchangeable strategies
- Protocol (Structural Subtyping): Defines executor interface

Author: Trading Team
Version: 1.0.0
Date: 2025-01-16
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Protocol,
    Type,
    runtime_checkable,
)

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TradeResult:
    """
    Result of a trade execution.

    Attributes:
        success: Whether the trade was executed successfully
        trade_id: Unique identifier for the trade
        signal: The signal that triggered the trade (LONG, SHORT, CLOSE)
        entry_price: Price at which the trade was opened
        exit_price: Price at which the trade was closed (if applicable)
        pnl: Profit/Loss in absolute terms
        pnl_pct: Profit/Loss as percentage
        timestamp: Time of execution
        message: Additional information about the execution
        metadata: Additional execution details
    """
    success: bool
    trade_id: Optional[int] = None
    signal: str = ""
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "trade_id": self.trade_id,
            "signal": self.signal,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass
class ExecutorConfig:
    """
    Configuration for trade executors.

    Attributes:
        initial_capital: Starting capital
        position_size_pct: Position size as percentage of capital
        enable_short: Whether short positions are allowed
        max_slippage_pct: Maximum allowed slippage percentage
        commission_pct: Commission percentage per trade
    """
    initial_capital: float = 10000.0
    position_size_pct: float = 0.1
    enable_short: bool = True
    max_slippage_pct: float = 0.001
    commission_pct: float = 0.0001


@dataclass
class ExecutorContext:
    """
    Runtime context for trade execution.

    Attributes:
        model_id: Identifier of the model generating signals
        db_connection: Optional database connection for persistence
        risk_manager: Optional risk manager for validation
        broker_client: Optional broker API client for live trading
    """
    model_id: str = "default"
    db_connection: Any = None
    risk_manager: Any = None
    broker_client: Any = None


# =============================================================================
# Protocol Definition (Structural Subtyping)
# =============================================================================

@runtime_checkable
class ITradeExecutor(Protocol):
    """
    Common interface for trade executors.

    Uses Python's Protocol for structural subtyping, allowing any class
    that implements these methods to be used as an executor without
    explicit inheritance.

    This follows the Interface Segregation Principle (ISP) by defining
    a minimal interface for trade execution.
    """

    def execute(self, signal: str, price: float, timestamp: Optional[datetime] = None) -> TradeResult:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal (LONG, SHORT, CLOSE, HOLD)
            price: Current market price
            timestamp: Optional execution timestamp

        Returns:
            TradeResult with execution details
        """
        ...

    def get_position(self) -> float:
        """
        Get current position size.

        Returns:
            Position size (positive for long, negative for short, 0 for flat)
        """
        ...

    def get_pnl(self) -> float:
        """
        Get current profit/loss.

        Returns:
            Total realized P&L
        """
        ...

    def get_unrealized_pnl(self, current_price: float) -> float:
        """
        Get unrealized profit/loss for open positions.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        ...

    def close_all(self, price: float, timestamp: Optional[datetime] = None) -> TradeResult:
        """
        Close all open positions.

        Args:
            price: Current market price
            timestamp: Optional close timestamp

        Returns:
            TradeResult with close details
        """
        ...

    def reset(self) -> None:
        """Reset executor to initial state."""
        ...

    @property
    def mode(self) -> str:
        """Get executor mode (paper, live, backtest)."""
        ...

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        ...


# =============================================================================
# Factory Implementation
# =============================================================================

class TradeExecutorFactory:
    """
    Factory for creating trade executors.

    Implements the Factory Pattern to centralize executor creation.
    Supports registration of custom executor types for extensibility.

    Usage:
        # Register executors (done at application startup)
        TradeExecutorFactory.register("paper", PaperTradeExecutor)
        TradeExecutorFactory.register("live", LiveTradeExecutor)
        TradeExecutorFactory.register("backtest", BacktestExecutor)

        # Create executor
        config = ExecutorConfig(initial_capital=50000)
        context = ExecutorContext(model_id="ppo_v1")
        executor = TradeExecutorFactory.create("paper", config, context)

        # Execute trades
        result = executor.execute("LONG", 4250.50)
    """

    _executors: Dict[str, Type[ITradeExecutor]] = {}
    _instances: Dict[str, ITradeExecutor] = {}

    @classmethod
    def register(cls, mode: str, executor_class: Type[ITradeExecutor]) -> None:
        """
        Register an executor class for a given mode.

        Args:
            mode: Execution mode identifier (paper, live, backtest)
            executor_class: Executor class implementing ITradeExecutor

        Raises:
            ValueError: If executor_class doesn't implement ITradeExecutor
        """
        # Note: We can't fully validate Protocol compliance at registration time
        # because Protocol uses structural subtyping, checked at runtime
        cls._executors[mode] = executor_class
        logger.info(f"Registered executor '{executor_class.__name__}' for mode '{mode}'")

    @classmethod
    def unregister(cls, mode: str) -> bool:
        """
        Unregister an executor for a given mode.

        Args:
            mode: Execution mode to unregister

        Returns:
            True if executor was unregistered, False if not found
        """
        if mode in cls._executors:
            del cls._executors[mode]
            logger.info(f"Unregistered executor for mode '{mode}'")
            return True
        return False

    @classmethod
    def create(
        cls,
        mode: Literal["paper", "live", "backtest"],
        config: Optional[ExecutorConfig] = None,
        context: Optional[ExecutorContext] = None,
    ) -> ITradeExecutor:
        """
        Create a trade executor for the specified mode.

        Args:
            mode: Execution mode (paper, live, backtest)
            config: Executor configuration
            context: Runtime context

        Returns:
            ITradeExecutor instance

        Raises:
            ValueError: If mode is not registered

        Example:
            >>> config = ExecutorConfig(initial_capital=50000)
            >>> context = ExecutorContext(model_id="ppo_v1")
            >>> executor = TradeExecutorFactory.create("paper", config, context)
        """
        if mode not in cls._executors:
            available = list(cls._executors.keys())
            raise ValueError(
                f"Unknown execution mode: '{mode}'. "
                f"Available modes: {available}"
            )

        config = config or ExecutorConfig()
        context = context or ExecutorContext()

        executor_class = cls._executors[mode]

        try:
            executor = executor_class(config, context)
            logger.info(f"Created {mode} executor for model '{context.model_id}'")
            return executor
        except Exception as e:
            logger.error(f"Failed to create {mode} executor: {e}")
            raise

    @classmethod
    def create_singleton(
        cls,
        mode: Literal["paper", "live", "backtest"],
        instance_key: str,
        config: Optional[ExecutorConfig] = None,
        context: Optional[ExecutorContext] = None,
    ) -> ITradeExecutor:
        """
        Create or retrieve a singleton executor instance.

        Useful for maintaining a single executor per model/mode combination.

        Args:
            mode: Execution mode
            instance_key: Unique key for this instance
            config: Executor configuration (ignored if instance exists)
            context: Runtime context (ignored if instance exists)

        Returns:
            ITradeExecutor instance (new or existing)
        """
        full_key = f"{mode}:{instance_key}"

        if full_key not in cls._instances:
            cls._instances[full_key] = cls.create(mode, config, context)

        return cls._instances[full_key]

    @classmethod
    def clear_singletons(cls) -> None:
        """Clear all singleton instances."""
        cls._instances.clear()
        logger.info("Cleared all singleton executor instances")

    @classmethod
    def get_registered_modes(cls) -> list:
        """
        Get list of all registered execution modes.

        Returns:
            List of registered mode names
        """
        return list(cls._executors.keys())

    @classmethod
    def is_registered(cls, mode: str) -> bool:
        """
        Check if an execution mode is registered.

        Args:
            mode: Execution mode to check

        Returns:
            True if registered, False otherwise
        """
        return mode in cls._executors


# =============================================================================
# Default Executor Implementations
# =============================================================================

class BaseTradeExecutor:
    """
    Base class for trade executors with common functionality.

    Provides shared implementation for position tracking, statistics,
    and basic trade management. Concrete implementations should extend
    this class and implement mode-specific behavior.
    """

    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        self._config = config
        self._context = context
        self._mode = "base"

        # Position tracking
        self._position: float = 0.0
        self._entry_price: Optional[float] = None
        self._entry_time: Optional[datetime] = None

        # P&L tracking
        self._realized_pnl: float = 0.0
        self._trade_count: int = 0
        self._winning_trades: int = 0
        self._losing_trades: int = 0

        # Capital tracking
        self._current_capital = config.initial_capital

    @property
    def mode(self) -> str:
        return self._mode

    def get_position(self) -> float:
        return self._position

    def get_pnl(self) -> float:
        return self._realized_pnl

    def get_unrealized_pnl(self, current_price: float) -> float:
        if self._position == 0 or self._entry_price is None:
            return 0.0

        if self._position > 0:  # Long
            return (current_price - self._entry_price) * abs(self._position)
        else:  # Short
            return (self._entry_price - current_price) * abs(self._position)

    def reset(self) -> None:
        self._position = 0.0
        self._entry_price = None
        self._entry_time = None
        self._realized_pnl = 0.0
        self._trade_count = 0
        self._winning_trades = 0
        self._losing_trades = 0
        self._current_capital = self._config.initial_capital
        logger.info(f"Reset {self._mode} executor for model '{self._context.model_id}'")

    @property
    def statistics(self) -> Dict[str, Any]:
        win_rate = 0.0
        if self._trade_count > 0:
            win_rate = self._winning_trades / self._trade_count * 100

        return {
            "mode": self._mode,
            "model_id": self._context.model_id,
            "position": self._position,
            "realized_pnl": round(self._realized_pnl, 2),
            "current_capital": round(self._current_capital, 2),
            "trade_count": self._trade_count,
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "win_rate": round(win_rate, 2),
            "return_pct": round(
                (self._current_capital - self._config.initial_capital)
                / self._config.initial_capital * 100, 2
            ),
        }

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on config."""
        capital_per_trade = self._current_capital * self._config.position_size_pct
        return capital_per_trade / price


class PaperTradeExecutor(BaseTradeExecutor):
    """
    Paper trading executor for simulated execution.

    Simulates trade execution without connecting to real markets.
    Useful for strategy validation and development.
    """

    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        super().__init__(config, context)
        self._mode = "paper"
        self._trade_id_counter = 0

    def execute(
        self,
        signal: str,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> TradeResult:
        timestamp = timestamp or datetime.now()
        signal_upper = signal.upper().strip()

        # HOLD - no action
        if signal_upper == "HOLD":
            return TradeResult(
                success=True,
                signal=signal_upper,
                message="No action taken for HOLD signal",
                timestamp=timestamp,
            )

        # CLOSE - close existing position
        if signal_upper == "CLOSE":
            return self.close_all(price, timestamp)

        # LONG signal
        if signal_upper == "LONG":
            if self._position > 0:
                return TradeResult(
                    success=True,
                    signal=signal_upper,
                    message="Already long, holding position",
                    timestamp=timestamp,
                )

            # Close short if exists
            if self._position < 0:
                self.close_all(price, timestamp)

            # Open long
            return self._open_position(signal_upper, price, timestamp, is_long=True)

        # SHORT signal
        if signal_upper == "SHORT":
            if not self._config.enable_short:
                return TradeResult(
                    success=False,
                    signal=signal_upper,
                    message="Short trading disabled",
                    timestamp=timestamp,
                )

            if self._position < 0:
                return TradeResult(
                    success=True,
                    signal=signal_upper,
                    message="Already short, holding position",
                    timestamp=timestamp,
                )

            # Close long if exists
            if self._position > 0:
                self.close_all(price, timestamp)

            # Open short
            return self._open_position(signal_upper, price, timestamp, is_long=False)

        return TradeResult(
            success=False,
            signal=signal_upper,
            message=f"Unknown signal: {signal_upper}",
            timestamp=timestamp,
        )

    def _open_position(
        self,
        signal: str,
        price: float,
        timestamp: datetime,
        is_long: bool
    ) -> TradeResult:
        """Open a new position."""
        self._trade_id_counter += 1
        size = self._calculate_position_size(price)

        self._position = size if is_long else -size
        self._entry_price = price
        self._entry_time = timestamp

        direction = "LONG" if is_long else "SHORT"

        logger.info(
            f"[PAPER] Opened {direction}: price={price:.4f}, size={size:.4f}, "
            f"model={self._context.model_id}"
        )

        return TradeResult(
            success=True,
            trade_id=self._trade_id_counter,
            signal=signal,
            entry_price=price,
            timestamp=timestamp,
            message=f"Opened {direction} position",
            metadata={"direction": direction, "size": size},
        )

    def close_all(self, price: float, timestamp: Optional[datetime] = None) -> TradeResult:
        timestamp = timestamp or datetime.now()

        if self._position == 0:
            return TradeResult(
                success=True,
                signal="CLOSE",
                message="No position to close",
                timestamp=timestamp,
            )

        # Calculate P&L
        if self._position > 0:  # Long
            pnl = (price - self._entry_price) * abs(self._position)
            direction = "LONG"
        else:  # Short
            pnl = (self._entry_price - price) * abs(self._position)
            direction = "SHORT"

        # Apply commission
        position_value = self._entry_price * abs(self._position)
        commission = position_value * self._config.commission_pct * 2  # Entry + exit
        pnl -= commission

        pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0.0

        # Update stats
        self._realized_pnl += pnl
        self._current_capital += pnl
        self._trade_count += 1

        if pnl > 0:
            self._winning_trades += 1
        elif pnl < 0:
            self._losing_trades += 1

        logger.info(
            f"[PAPER] Closed {direction}: entry={self._entry_price:.4f}, "
            f"exit={price:.4f}, pnl={pnl:.2f} ({pnl_pct:+.2f}%)"
        )

        # Reset position
        old_entry = self._entry_price
        self._position = 0.0
        self._entry_price = None
        self._entry_time = None

        return TradeResult(
            success=True,
            trade_id=self._trade_id_counter,
            signal="CLOSE",
            entry_price=old_entry,
            exit_price=price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            timestamp=timestamp,
            message=f"Closed {direction} position",
            metadata={"direction": direction, "commission": commission},
        )


class BacktestExecutor(BaseTradeExecutor):
    """
    Backtest executor for historical simulation.

    Similar to paper trading but optimized for backtesting
    with full trade history tracking.
    """

    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        super().__init__(config, context)
        self._mode = "backtest"
        self._trade_id_counter = 0
        self._trade_history: list = []

    def execute(
        self,
        signal: str,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> TradeResult:
        # Delegate to paper trading logic
        timestamp = timestamp or datetime.now()
        signal_upper = signal.upper().strip()

        if signal_upper == "HOLD":
            return TradeResult(
                success=True,
                signal=signal_upper,
                message="No action taken for HOLD signal",
                timestamp=timestamp,
            )

        if signal_upper == "CLOSE":
            return self.close_all(price, timestamp)

        if signal_upper == "LONG":
            if self._position > 0:
                return TradeResult(success=True, signal=signal_upper,
                                   message="Already long", timestamp=timestamp)
            if self._position < 0:
                self.close_all(price, timestamp)
            return self._open_position(signal_upper, price, timestamp, is_long=True)

        if signal_upper == "SHORT":
            if not self._config.enable_short:
                return TradeResult(success=False, signal=signal_upper,
                                   message="Short trading disabled", timestamp=timestamp)
            if self._position < 0:
                return TradeResult(success=True, signal=signal_upper,
                                   message="Already short", timestamp=timestamp)
            if self._position > 0:
                self.close_all(price, timestamp)
            return self._open_position(signal_upper, price, timestamp, is_long=False)

        return TradeResult(success=False, signal=signal_upper,
                          message=f"Unknown signal: {signal_upper}", timestamp=timestamp)

    def _open_position(
        self,
        signal: str,
        price: float,
        timestamp: datetime,
        is_long: bool
    ) -> TradeResult:
        self._trade_id_counter += 1
        size = self._calculate_position_size(price)

        self._position = size if is_long else -size
        self._entry_price = price
        self._entry_time = timestamp

        direction = "LONG" if is_long else "SHORT"

        return TradeResult(
            success=True,
            trade_id=self._trade_id_counter,
            signal=signal,
            entry_price=price,
            timestamp=timestamp,
            message=f"Opened {direction} position",
            metadata={"direction": direction, "size": size},
        )

    def close_all(self, price: float, timestamp: Optional[datetime] = None) -> TradeResult:
        timestamp = timestamp or datetime.now()

        if self._position == 0:
            return TradeResult(success=True, signal="CLOSE",
                              message="No position to close", timestamp=timestamp)

        if self._position > 0:
            pnl = (price - self._entry_price) * abs(self._position)
            direction = "LONG"
        else:
            pnl = (self._entry_price - price) * abs(self._position)
            direction = "SHORT"

        position_value = self._entry_price * abs(self._position)
        commission = position_value * self._config.commission_pct * 2
        pnl -= commission
        pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0.0

        # Store in history
        self._trade_history.append({
            "trade_id": self._trade_id_counter,
            "direction": direction,
            "entry_price": self._entry_price,
            "exit_price": price,
            "entry_time": self._entry_time,
            "exit_time": timestamp,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

        self._realized_pnl += pnl
        self._current_capital += pnl
        self._trade_count += 1

        if pnl > 0:
            self._winning_trades += 1
        elif pnl < 0:
            self._losing_trades += 1

        old_entry = self._entry_price
        self._position = 0.0
        self._entry_price = None
        self._entry_time = None

        return TradeResult(
            success=True,
            trade_id=self._trade_id_counter,
            signal="CLOSE",
            entry_price=old_entry,
            exit_price=price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            timestamp=timestamp,
            message=f"Closed {direction} position",
        )

    def get_trade_history(self) -> list:
        """Get full trade history for analysis."""
        return self._trade_history.copy()

    def reset(self) -> None:
        super().reset()
        self._trade_history.clear()


# =============================================================================
# Auto-register default executors
# =============================================================================

TradeExecutorFactory.register("paper", PaperTradeExecutor)
TradeExecutorFactory.register("backtest", BacktestExecutor)
