"""
Position Manager Service
=========================
Tracks active positions and signal history.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from collections import deque

from ..models.signal_schema import TradingSignal, SignalAction
from ..config import get_config

logger = logging.getLogger(__name__)


class Position:
    """Represents an active trading position"""

    def __init__(
        self,
        signal: TradingSignal,
        entry_time: datetime,
        position_id: Optional[str] = None
    ):
        self.position_id = position_id or signal.signal_id
        self.signal = signal
        self.entry_time = entry_time
        self.entry_price = signal.entry_price
        self.stop_loss = signal.stop_loss
        self.take_profit = signal.take_profit
        self.position_size = signal.position_size
        self.current_pnl = 0.0
        self.current_pnl_pct = 0.0
        self.status = "OPEN"
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.exit_reason: Optional[str] = None

    def update(self, current_price: float):
        """Update position with current price"""
        if self.signal.action == SignalAction.BUY:
            self.current_pnl = current_price - self.entry_price
        elif self.signal.action == SignalAction.SELL:
            self.current_pnl = self.entry_price - current_price
        else:
            self.current_pnl = 0.0

        if self.entry_price > 0:
            self.current_pnl_pct = (self.current_pnl / self.entry_price) * 100
        else:
            self.current_pnl_pct = 0.0

    def check_exit_conditions(self, current_price: float) -> Optional[str]:
        """
        Check if position should be exited.

        Args:
            current_price: Current market price

        Returns:
            Exit reason if position should be closed, None otherwise
        """
        # Check stop loss
        if self.signal.action == SignalAction.BUY:
            if current_price <= self.stop_loss:
                return "STOP_LOSS"
            if current_price >= self.take_profit:
                return "TAKE_PROFIT"
        elif self.signal.action == SignalAction.SELL:
            if current_price >= self.stop_loss:
                return "STOP_LOSS"
            if current_price <= self.take_profit:
                return "TAKE_PROFIT"

        return None

    def close(self, exit_price: float, exit_reason: str):
        """Close the position"""
        self.exit_time = datetime.utcnow()
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.status = "CLOSED"
        self.update(exit_price)

        logger.info(
            f"Position {self.position_id} closed: "
            f"PnL={self.current_pnl:.2f} ({self.current_pnl_pct:.2f}%), "
            f"Reason={exit_reason}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'position_id': self.position_id,
            'signal_id': self.signal.signal_id,
            'action': self.signal.action.value,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'current_pnl': self.current_pnl,
            'current_pnl_pct': self.current_pnl_pct,
            'status': self.status,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason
        }


class PositionManager:
    """Manages trading positions and signal history"""

    def __init__(self):
        """Initialize position manager"""
        self.config = get_config()
        self.active_positions: Dict[str, Position] = {}
        self.signal_history: deque = deque(maxlen=self.config.signal_history_limit)
        self.closed_positions: List[Position] = []

        # Statistics
        self.total_signals = 0
        self.total_positions_opened = 0
        self.total_positions_closed = 0
        self.total_pnl = 0.0
        self.winning_positions = 0
        self.losing_positions = 0

    def add_signal(self, signal: TradingSignal) -> bool:
        """
        Add a signal to history and potentially open a position.

        Args:
            signal: Trading signal to add

        Returns:
            True if position was opened, False otherwise
        """
        # Add to signal history
        self.signal_history.append(signal)
        self.total_signals += 1

        # Open position if actionable signal
        if signal.action in [SignalAction.BUY, SignalAction.SELL]:
            position = Position(
                signal=signal,
                entry_time=signal.timestamp
            )
            self.active_positions[position.position_id] = position
            self.total_positions_opened += 1

            logger.info(
                f"Position opened: {position.position_id} "
                f"({signal.action.value} @ {signal.entry_price})"
            )
            return True

        return False

    def update_positions(self, current_price: float):
        """
        Update all active positions with current price.

        Args:
            current_price: Current market price
        """
        positions_to_close = []

        for position_id, position in self.active_positions.items():
            # Update position PnL
            position.update(current_price)

            # Check exit conditions
            exit_reason = position.check_exit_conditions(current_price)
            if exit_reason:
                positions_to_close.append((position_id, current_price, exit_reason))

        # Close positions that hit exit conditions
        for position_id, exit_price, exit_reason in positions_to_close:
            self.close_position(position_id, exit_price, exit_reason)

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str
    ) -> Optional[Position]:
        """
        Close a specific position.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_reason: Reason for exit

        Returns:
            Closed position if found, None otherwise
        """
        position = self.active_positions.pop(position_id, None)

        if position:
            position.close(exit_price, exit_reason)
            self.closed_positions.append(position)
            self.total_positions_closed += 1

            # Update statistics
            self.total_pnl += position.current_pnl
            if position.current_pnl > 0:
                self.winning_positions += 1
            else:
                self.losing_positions += 1

            return position

        logger.warning(f"Position {position_id} not found")
        return None

    def close_all_positions(self, current_price: float, reason: str = "MANUAL_CLOSE"):
        """
        Close all active positions.

        Args:
            current_price: Current market price
            reason: Reason for closing all positions
        """
        position_ids = list(self.active_positions.keys())

        for position_id in position_ids:
            self.close_position(position_id, current_price, reason)

        logger.info(f"Closed {len(position_ids)} positions: {reason}")

    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get list of active positions"""
        return [pos.to_dict() for pos in self.active_positions.values()]

    def get_closed_positions(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of closed positions.

        Args:
            limit: Maximum number of positions to return

        Returns:
            List of closed position dictionaries
        """
        positions = self.closed_positions[-limit:] if limit else self.closed_positions
        return [pos.to_dict() for pos in positions]

    def get_signal_history(
        self,
        limit: Optional[int] = None,
        action_filter: Optional[SignalAction] = None
    ) -> List[TradingSignal]:
        """
        Get signal history with optional filtering.

        Args:
            limit: Maximum number of signals to return
            action_filter: Filter by specific action

        Returns:
            List of trading signals
        """
        signals = list(self.signal_history)

        # Filter by action if specified
        if action_filter:
            signals = [s for s in signals if s.action == action_filter]

        # Apply limit
        if limit:
            signals = signals[-limit:]

        return signals

    def get_latest_signal(self) -> Optional[TradingSignal]:
        """Get the most recent signal"""
        if self.signal_history:
            return self.signal_history[-1]
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get position and signal statistics"""
        win_rate = (
            self.winning_positions / self.total_positions_closed * 100
            if self.total_positions_closed > 0 else 0.0
        )

        avg_pnl = (
            self.total_pnl / self.total_positions_closed
            if self.total_positions_closed > 0 else 0.0
        )

        return {
            'total_signals': self.total_signals,
            'total_positions_opened': self.total_positions_opened,
            'total_positions_closed': self.total_positions_closed,
            'active_positions': len(self.active_positions),
            'total_pnl': self.total_pnl,
            'avg_pnl': avg_pnl,
            'winning_positions': self.winning_positions,
            'losing_positions': self.losing_positions,
            'win_rate': win_rate
        }

    def cleanup_old_signals(self, days: int):
        """
        Remove signals older than specified days.

        Args:
            days: Number of days to retain
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        original_count = len(self.signal_history)

        # Filter out old signals
        self.signal_history = deque(
            [s for s in self.signal_history if s.timestamp >= cutoff_date],
            maxlen=self.config.signal_history_limit
        )

        removed = original_count - len(self.signal_history)
        if removed > 0:
            logger.info(f"Removed {removed} signals older than {days} days")

    def reset_statistics(self):
        """Reset all statistics (keeps history)"""
        self.total_signals = len(self.signal_history)
        self.total_positions_opened = len(self.active_positions) + len(self.closed_positions)
        self.total_positions_closed = len(self.closed_positions)
        self.total_pnl = sum(pos.current_pnl for pos in self.closed_positions)
        self.winning_positions = sum(1 for pos in self.closed_positions if pos.current_pnl > 0)
        self.losing_positions = sum(1 for pos in self.closed_positions if pos.current_pnl <= 0)

        logger.info("Statistics reset from history")
