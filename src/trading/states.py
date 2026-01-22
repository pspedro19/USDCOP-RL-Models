"""
Trade State Pattern - State Machine for Trade Lifecycle
========================================================

This module implements the State Pattern for managing trade lifecycle states.
It provides a clean separation of behavior for different trade states:
- Pending: Trade awaiting execution
- Open: Active trade monitoring stop-loss and take-profit
- Closing: Transitional state while closing
- Closed: Final immutable state

Usage:
    from src.trading.states import (
        PendingState, OpenState, ClosingState, ClosedState
    )

    # Initialize trade with pending state
    trade = Trade(state=PendingState())

    # State transitions happen automatically based on price updates
    trade.state = trade.state.on_price_update(trade, current_price)

Author: USD/COP Trading System
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .paper_trader import PaperTrade

logger = logging.getLogger(__name__)


class TradeState(ABC):
    """
    Base state for trade lifecycle.

    Implements the State Pattern to encapsulate state-specific behavior
    for trade objects. Each concrete state handles price updates differently
    and determines what operations are allowed.
    """

    @abstractmethod
    def on_price_update(
        self,
        trade: "Trade",
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> "TradeState":
        """
        Handle price update and potentially transition to new state.

        Args:
            trade: The trade object being updated
            price: Current market price
            stop_loss: Optional stop-loss price level
            take_profit: Optional take-profit price level

        Returns:
            New TradeState (may be self if no transition)
        """
        pass

    @abstractmethod
    def can_close(self) -> bool:
        """
        Check if trade can be closed from current state.

        Returns:
            True if closing is allowed, False otherwise
        """
        pass

    @abstractmethod
    def can_modify(self) -> bool:
        """
        Check if trade parameters can be modified from current state.

        Returns:
            True if modifications are allowed, False otherwise
        """
        pass

    @abstractmethod
    def get_status(self) -> str:
        """
        Get human-readable status string.

        Returns:
            Status string describing current state
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PendingState(TradeState):
    """
    Trade pending execution.

    In this state, the trade is awaiting execution. Price updates
    do not affect the trade, but it can transition to OpenState
    once executed.
    """

    def on_price_update(
        self,
        trade: "Trade",
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeState:
        """
        Handle price update in pending state.

        Pending trades don't react to price updates until executed.
        Transition to OpenState must be triggered explicitly.

        Args:
            trade: The trade object being updated
            price: Current market price
            stop_loss: Optional stop-loss price level
            take_profit: Optional take-profit price level

        Returns:
            Self (no automatic transition from pending)
        """
        logger.debug(
            f"Trade {getattr(trade, 'trade_id', 'unknown')} pending, "
            f"price update ignored: {price}"
        )
        return self

    def can_close(self) -> bool:
        """
        Pending trades cannot be closed (not yet executed).

        Returns:
            False - pending trades must be cancelled, not closed
        """
        return False

    def can_modify(self) -> bool:
        """
        Pending trades can be modified before execution.

        Returns:
            True - parameters can be changed while pending
        """
        return True

    def get_status(self) -> str:
        """Get status string."""
        return "pending"

    def execute(self, trade: "Trade", execution_price: float) -> "OpenState":
        """
        Execute the pending trade and transition to open state.

        Args:
            trade: The trade to execute
            execution_price: Price at which trade is executed

        Returns:
            OpenState after execution
        """
        logger.info(
            f"Trade {getattr(trade, 'trade_id', 'unknown')} executed at {execution_price}"
        )
        return OpenState()


class OpenState(TradeState):
    """
    Trade open and active.

    In this state, the trade is actively monitoring the market.
    Price updates check for stop-loss and take-profit triggers.
    """

    def __init__(self):
        """Initialize open state with tracking for SL/TP triggers."""
        self._sl_triggered = False
        self._tp_triggered = False

    def on_price_update(
        self,
        trade: "Trade",
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeState:
        """
        Handle price update for open trade.

        Checks if price triggers stop-loss or take-profit levels.
        Transitions to ClosingState if either is triggered.

        Args:
            trade: The trade object being updated
            price: Current market price
            stop_loss: Stop-loss price level
            take_profit: Take-profit price level

        Returns:
            ClosingState if SL/TP triggered, self otherwise
        """
        trade_id = getattr(trade, 'trade_id', 'unknown')
        direction = getattr(trade, 'direction', 'LONG')

        # Check stop-loss
        if stop_loss is not None:
            if direction == "LONG" and price <= stop_loss:
                logger.warning(
                    f"Trade {trade_id} STOP LOSS triggered: "
                    f"price {price} <= SL {stop_loss}"
                )
                self._sl_triggered = True
                return ClosingState(reason="stop_loss", trigger_price=price)
            elif direction == "SHORT" and price >= stop_loss:
                logger.warning(
                    f"Trade {trade_id} STOP LOSS triggered: "
                    f"price {price} >= SL {stop_loss}"
                )
                self._sl_triggered = True
                return ClosingState(reason="stop_loss", trigger_price=price)

        # Check take-profit
        if take_profit is not None:
            if direction == "LONG" and price >= take_profit:
                logger.info(
                    f"Trade {trade_id} TAKE PROFIT triggered: "
                    f"price {price} >= TP {take_profit}"
                )
                self._tp_triggered = True
                return ClosingState(reason="take_profit", trigger_price=price)
            elif direction == "SHORT" and price <= take_profit:
                logger.info(
                    f"Trade {trade_id} TAKE PROFIT triggered: "
                    f"price {price} <= TP {take_profit}"
                )
                self._tp_triggered = True
                return ClosingState(reason="take_profit", trigger_price=price)

        # No trigger, remain open
        return self

    def can_close(self) -> bool:
        """
        Open trades can be closed manually.

        Returns:
            True - open trades can be closed at any time
        """
        return True

    def can_modify(self) -> bool:
        """
        Open trades can have SL/TP modified.

        Returns:
            True - stop-loss and take-profit can be adjusted
        """
        return True

    def get_status(self) -> str:
        """Get status string."""
        return "open"

    def request_close(self, reason: str = "manual") -> "ClosingState":
        """
        Request to close the trade manually.

        Args:
            reason: Reason for closing (default: "manual")

        Returns:
            ClosingState for transition
        """
        return ClosingState(reason=reason)


class ClosingState(TradeState):
    """
    Transitional state while closing.

    This state represents a trade in the process of being closed.
    It prevents modifications and further closes while the
    closing operation completes.
    """

    def __init__(
        self,
        reason: str = "unknown",
        trigger_price: Optional[float] = None
    ):
        """
        Initialize closing state.

        Args:
            reason: Reason for closing (stop_loss, take_profit, manual, etc.)
            trigger_price: Price that triggered the close
        """
        self.reason = reason
        self.trigger_price = trigger_price

    def on_price_update(
        self,
        trade: "Trade",
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeState:
        """
        Handle price update while closing.

        Closing trades ignore price updates - they are in
        a transitional state waiting for completion.

        Args:
            trade: The trade object being updated
            price: Current market price
            stop_loss: Ignored
            take_profit: Ignored

        Returns:
            Self (no transition, waiting for completion)
        """
        logger.debug(
            f"Trade {getattr(trade, 'trade_id', 'unknown')} closing, "
            f"price update ignored: {price}"
        )
        return self

    def can_close(self) -> bool:
        """
        Already closing, cannot initiate another close.

        Returns:
            False - close already in progress
        """
        return False

    def can_modify(self) -> bool:
        """
        Cannot modify trade while closing.

        Returns:
            False - no modifications allowed during close
        """
        return False

    def get_status(self) -> str:
        """Get status string with reason."""
        return f"closing:{self.reason}"

    def complete(self) -> "ClosedState":
        """
        Complete the closing process.

        Returns:
            ClosedState - final state after closing completes
        """
        return ClosedState(close_reason=self.reason)

    def __repr__(self) -> str:
        return f"ClosingState(reason='{self.reason}')"


class ClosedState(TradeState):
    """
    Trade closed - final state.

    This is a terminal state. Once a trade is closed, it cannot
    be modified or closed again. All operations are rejected.
    """

    def __init__(self, close_reason: str = "unknown"):
        """
        Initialize closed state.

        Args:
            close_reason: Reason the trade was closed
        """
        self.close_reason = close_reason

    def on_price_update(
        self,
        trade: "Trade",
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeState:
        """
        Handle price update for closed trade.

        Closed trades ignore all price updates.

        Args:
            trade: The trade object being updated
            price: Current market price
            stop_loss: Ignored
            take_profit: Ignored

        Returns:
            Self (terminal state, no transitions)
        """
        return self

    def can_close(self) -> bool:
        """
        Cannot close an already closed trade.

        Returns:
            False - already closed
        """
        return False

    def can_modify(self) -> bool:
        """
        Cannot modify a closed trade.

        Returns:
            False - closed trades are immutable
        """
        return False

    def get_status(self) -> str:
        """Get status string with close reason."""
        return f"closed:{self.close_reason}"

    def __repr__(self) -> str:
        return f"ClosedState(reason='{self.close_reason}')"


class Trade:
    """
    Trade entity with state-based behavior.

    This class represents a trade that uses the State Pattern
    to manage its lifecycle. Behavior changes based on current state.
    """

    def __init__(
        self,
        trade_id: str,
        direction: str = "LONG",
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        state: Optional[TradeState] = None
    ):
        """
        Initialize trade with initial state.

        Args:
            trade_id: Unique identifier for the trade
            direction: "LONG" or "SHORT"
            entry_price: Entry price (set when executed)
            stop_loss: Stop-loss price level
            take_profit: Take-profit price level
            state: Initial state (defaults to PendingState)
        """
        self.trade_id = trade_id
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self._state = state or PendingState()
        self.exit_price: Optional[float] = None
        self.close_reason: Optional[str] = None

    @property
    def state(self) -> TradeState:
        """Get current state."""
        return self._state

    @state.setter
    def state(self, new_state: TradeState) -> None:
        """
        Set new state with logging.

        Args:
            new_state: New TradeState to transition to
        """
        old_status = self._state.get_status()
        self._state = new_state
        new_status = self._state.get_status()
        logger.info(
            f"Trade {self.trade_id} state transition: "
            f"{old_status} -> {new_status}"
        )

    def update_price(self, price: float) -> None:
        """
        Update trade with new market price.

        Delegates to current state for behavior.

        Args:
            price: Current market price
        """
        new_state = self._state.on_price_update(
            self, price, self.stop_loss, self.take_profit
        )
        if new_state != self._state:
            self.state = new_state

    def can_close(self) -> bool:
        """Check if trade can be closed."""
        return self._state.can_close()

    def can_modify(self) -> bool:
        """Check if trade can be modified."""
        return self._state.can_modify()

    def get_status(self) -> str:
        """Get current status string."""
        return self._state.get_status()

    def execute(self, execution_price: float) -> bool:
        """
        Execute a pending trade.

        Args:
            execution_price: Price at which to execute

        Returns:
            True if executed, False if not in pending state
        """
        if not isinstance(self._state, PendingState):
            logger.warning(
                f"Trade {self.trade_id} cannot execute: "
                f"not in pending state ({self.get_status()})"
            )
            return False

        self.entry_price = execution_price
        self.state = self._state.execute(self, execution_price)
        return True

    def close(self, exit_price: float, reason: str = "manual") -> bool:
        """
        Close the trade.

        Args:
            exit_price: Price at which to close
            reason: Reason for closing

        Returns:
            True if closed, False if not allowed
        """
        if not self._state.can_close():
            logger.warning(
                f"Trade {self.trade_id} cannot close: "
                f"current state {self.get_status()}"
            )
            return False

        # Transition to closing
        if isinstance(self._state, OpenState):
            self.state = self._state.request_close(reason)

        # Complete the close
        if isinstance(self._state, ClosingState):
            self.exit_price = exit_price
            self.close_reason = self._state.reason
            self.state = self._state.complete()
            return True

        return False

    def modify_stop_loss(self, new_stop_loss: float) -> bool:
        """
        Modify stop-loss level.

        Args:
            new_stop_loss: New stop-loss price

        Returns:
            True if modified, False if not allowed
        """
        if not self._state.can_modify():
            logger.warning(
                f"Trade {self.trade_id} cannot modify SL: "
                f"current state {self.get_status()}"
            )
            return False

        old_sl = self.stop_loss
        self.stop_loss = new_stop_loss
        logger.info(
            f"Trade {self.trade_id} stop-loss modified: "
            f"{old_sl} -> {new_stop_loss}"
        )
        return True

    def modify_take_profit(self, new_take_profit: float) -> bool:
        """
        Modify take-profit level.

        Args:
            new_take_profit: New take-profit price

        Returns:
            True if modified, False if not allowed
        """
        if not self._state.can_modify():
            logger.warning(
                f"Trade {self.trade_id} cannot modify TP: "
                f"current state {self.get_status()}"
            )
            return False

        old_tp = self.take_profit
        self.take_profit = new_take_profit
        logger.info(
            f"Trade {self.trade_id} take-profit modified: "
            f"{old_tp} -> {new_take_profit}"
        )
        return True

    def __repr__(self) -> str:
        return (
            f"Trade(id={self.trade_id}, direction={self.direction}, "
            f"status={self.get_status()})"
        )
