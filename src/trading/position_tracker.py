"""
PositionTracker - Position Management Component
================================================

Single Responsibility: Track and manage trading positions.
Split from the PaperTrader God Class to follow SOLID principles.

This component is responsible for:
- Tracking open positions per model
- Managing position state (entry price, size, direction)
- Calculating unrealized P&L
- Maintaining position history

Design Patterns:
- Single Responsibility Principle: Only handles position tracking
- Observer Pattern: Can notify listeners of position changes
- Repository Pattern: Acts as a repository for position data

Author: Trading Team
Version: 1.0.0
Date: 2025-01-16
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
)
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class PositionDirection(str, Enum):
    """Direction of a trading position."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class PositionStatus(str, Enum):
    """Status of a trading position."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


@dataclass
class Position:
    """
    Represents a trading position.

    Immutable-style dataclass that captures all position information.
    Supports serialization for persistence and auditing.

    Attributes:
        position_id: Unique identifier
        model_id: Model that owns this position
        direction: LONG, SHORT, or FLAT
        size: Position size in units
        entry_price: Price at which position was opened
        entry_time: Timestamp of entry
        exit_price: Price at which position was closed (if closed)
        exit_time: Timestamp of exit (if closed)
        status: Current status (open, closed)
        pnl: Realized profit/loss
        pnl_pct: Realized P&L as percentage
        metadata: Additional position data
    """
    position_id: int
    model_id: str
    direction: PositionDirection
    size: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    status: PositionStatus = PositionStatus.OPEN
    pnl: float = 0.0
    pnl_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        result = asdict(self)
        result['direction'] = self.direction.value
        result['status'] = self.status.value
        if self.entry_time:
            result['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            result['exit_time'] = self.exit_time.isoformat()
        return result

    def to_json(self) -> str:
        """Convert position to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create Position from dictionary."""
        data = data.copy()
        if isinstance(data.get('direction'), str):
            data['direction'] = PositionDirection(data['direction'])
        if isinstance(data.get('status'), str):
            data['status'] = PositionStatus(data['status'])
        if isinstance(data.get('entry_time'), str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if isinstance(data.get('exit_time'), str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        return cls(**data)

    def is_open(self) -> bool:
        """Check if position is open."""
        return self.status == PositionStatus.OPEN

    def is_long(self) -> bool:
        """Check if position is long."""
        return self.direction == PositionDirection.LONG

    def is_short(self) -> bool:
        """Check if position is short."""
        return self.direction == PositionDirection.SHORT

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at given price.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        if self.status != PositionStatus.OPEN:
            return 0.0

        if self.direction == PositionDirection.LONG:
            return (current_price - self.entry_price) * self.size
        elif self.direction == PositionDirection.SHORT:
            return (self.entry_price - current_price) * self.size
        return 0.0


# =============================================================================
# Protocol for Position Change Listeners
# =============================================================================

class PositionChangeListener(Protocol):
    """Protocol for position change notification."""

    def on_position_opened(self, position: Position) -> None:
        """Called when a position is opened."""
        ...

    def on_position_closed(self, position: Position) -> None:
        """Called when a position is closed."""
        ...


# =============================================================================
# PositionTracker Implementation
# =============================================================================

class PositionTracker:
    """
    Tracks and manages trading positions.

    Single Responsibility: Manages position state and history.
    Does NOT handle order execution or risk validation.

    Features:
    - Multi-model position tracking
    - Position history maintenance
    - Unrealized P&L calculation
    - Observer pattern for position changes
    - Serialization support

    Usage:
        tracker = PositionTracker()

        # Open a position
        position = tracker.open_position(
            model_id="ppo_v1",
            direction=PositionDirection.LONG,
            size=100,
            price=4250.50
        )

        # Get unrealized P&L
        unrealized = tracker.get_unrealized_pnl("ppo_v1", current_price=4260.00)

        # Close the position
        closed = tracker.close_position("ppo_v1", price=4260.00)
    """

    def __init__(self) -> None:
        """Initialize the position tracker."""
        # Open positions by model_id
        self._positions: Dict[str, Position] = {}

        # Position history (closed positions)
        self._history: List[Position] = []

        # Position ID counter
        self._position_counter: int = 0

        # Change listeners
        self._listeners: List[PositionChangeListener] = []

        logger.info("PositionTracker initialized")

    def _generate_position_id(self) -> int:
        """Generate unique position ID."""
        self._position_counter += 1
        return self._position_counter

    # =========================================================================
    # Position Operations
    # =========================================================================

    def open_position(
        self,
        model_id: str,
        direction: PositionDirection,
        size: float,
        price: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Position:
        """
        Open a new position.

        Args:
            model_id: Model identifier
            direction: Position direction (LONG/SHORT)
            size: Position size in units
            price: Entry price
            timestamp: Entry timestamp (default: now)
            metadata: Additional position data

        Returns:
            Created Position

        Raises:
            ValueError: If model already has an open position
        """
        if model_id in self._positions:
            raise ValueError(
                f"Model '{model_id}' already has an open position. "
                "Close it first before opening a new one."
            )

        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        position = Position(
            position_id=self._generate_position_id(),
            model_id=model_id,
            direction=direction,
            size=size,
            entry_price=price,
            entry_time=timestamp,
            status=PositionStatus.OPEN,
            metadata=metadata,
        )

        self._positions[model_id] = position

        logger.info(
            f"Opened {direction.value} position: model={model_id}, "
            f"price={price:.4f}, size={size:.4f}"
        )

        # Notify listeners
        self._notify_position_opened(position)

        return position

    def close_position(
        self,
        model_id: str,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> Position:
        """
        Close an existing position.

        Args:
            model_id: Model identifier
            price: Exit price
            timestamp: Exit timestamp (default: now)

        Returns:
            Closed Position with calculated P&L

        Raises:
            ValueError: If model has no open position
        """
        if model_id not in self._positions:
            raise ValueError(f"No open position for model '{model_id}'")

        timestamp = timestamp or datetime.now()
        position = self._positions.pop(model_id)

        # Calculate P&L
        if position.direction == PositionDirection.LONG:
            pnl = (price - position.entry_price) * position.size
        else:  # SHORT
            pnl = (position.entry_price - price) * position.size

        # Calculate P&L percentage
        position_value = position.entry_price * position.size
        pnl_pct = (pnl / position_value * 100) if position_value > 0 else 0.0

        # Update position
        closed_position = Position(
            position_id=position.position_id,
            model_id=position.model_id,
            direction=position.direction,
            size=position.size,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=price,
            exit_time=timestamp,
            status=PositionStatus.CLOSED,
            pnl=pnl,
            pnl_pct=pnl_pct,
            metadata=position.metadata,
        )

        # Add to history
        self._history.append(closed_position)

        logger.info(
            f"Closed {position.direction.value} position: model={model_id}, "
            f"entry={position.entry_price:.4f}, exit={price:.4f}, "
            f"pnl={pnl:.2f} ({pnl_pct:+.2f}%)"
        )

        # Notify listeners
        self._notify_position_closed(closed_position)

        return closed_position

    def close_all_positions(
        self,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> List[Position]:
        """
        Close all open positions.

        Args:
            price: Exit price
            timestamp: Exit timestamp

        Returns:
            List of closed positions
        """
        closed = []
        model_ids = list(self._positions.keys())

        for model_id in model_ids:
            try:
                position = self.close_position(model_id, price, timestamp)
                closed.append(position)
            except Exception as e:
                logger.error(f"Failed to close position for {model_id}: {e}")

        logger.info(f"Closed {len(closed)} positions at price {price}")
        return closed

    # =========================================================================
    # Query Operations
    # =========================================================================

    def has_position(self, model_id: str) -> bool:
        """Check if model has an open position."""
        return model_id in self._positions

    def get_position(self, model_id: str) -> Optional[Position]:
        """Get open position for a model."""
        return self._positions.get(model_id)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    def get_position_direction(self, model_id: str) -> Optional[PositionDirection]:
        """Get direction of open position for a model."""
        position = self._positions.get(model_id)
        return position.direction if position else None

    def get_position_size(self, model_id: str) -> float:
        """Get size of open position for a model."""
        position = self._positions.get(model_id)
        if position is None:
            return 0.0
        if position.direction == PositionDirection.LONG:
            return position.size
        elif position.direction == PositionDirection.SHORT:
            return -position.size
        return 0.0

    def get_unrealized_pnl(
        self,
        model_id: str,
        current_price: float
    ) -> float:
        """
        Get unrealized P&L for a model's position.

        Args:
            model_id: Model identifier
            current_price: Current market price

        Returns:
            Unrealized P&L (0 if no position)
        """
        position = self._positions.get(model_id)
        if position is None:
            return 0.0
        return position.calculate_unrealized_pnl(current_price)

    def get_total_unrealized_pnl(self, current_price: float) -> float:
        """
        Get total unrealized P&L across all positions.

        Args:
            current_price: Current market price

        Returns:
            Total unrealized P&L
        """
        return sum(
            pos.calculate_unrealized_pnl(current_price)
            for pos in self._positions.values()
        )

    # =========================================================================
    # History Operations
    # =========================================================================

    def get_history(
        self,
        model_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Position]:
        """
        Get position history.

        Args:
            model_id: Optional filter by model
            limit: Optional limit on number of positions

        Returns:
            List of closed positions
        """
        history = self._history

        if model_id:
            history = [p for p in history if p.model_id == model_id]

        if limit:
            history = history[-limit:]

        return history

    def get_model_history(self, model_id: str) -> List[Position]:
        """Get all closed positions for a model."""
        return [p for p in self._history if p.model_id == model_id]

    def get_total_pnl(self, model_id: Optional[str] = None) -> float:
        """
        Get total realized P&L.

        Args:
            model_id: Optional filter by model

        Returns:
            Total realized P&L
        """
        if model_id:
            return sum(p.pnl for p in self._history if p.model_id == model_id)
        return sum(p.pnl for p in self._history)

    def get_trade_count(self, model_id: Optional[str] = None) -> int:
        """
        Get number of closed trades.

        Args:
            model_id: Optional filter by model

        Returns:
            Number of closed trades
        """
        if model_id:
            return sum(1 for p in self._history if p.model_id == model_id)
        return len(self._history)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive position statistics.

        Args:
            model_id: Optional filter by model

        Returns:
            Dictionary with position statistics
        """
        history = self.get_history(model_id=model_id)

        if not history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_pnl": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "profit_factor": 0.0,
                "open_positions": len(self._positions) if not model_id else (1 if model_id in self._positions else 0),
            }

        winners = [p for p in history if p.pnl > 0]
        losers = [p for p in history if p.pnl < 0]

        total_pnl = sum(p.pnl for p in history)
        gross_profit = sum(p.pnl for p in winners)
        gross_loss = abs(sum(p.pnl for p in losers))

        return {
            "total_trades": len(history),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": round(len(winners) / len(history) * 100, 2) if history else 0.0,
            "total_pnl": round(total_pnl, 2),
            "average_pnl": round(total_pnl / len(history), 2) if history else 0.0,
            "average_win": round(gross_profit / len(winners), 2) if winners else 0.0,
            "average_loss": round(gross_loss / len(losers), 2) if losers else 0.0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf'),
            "open_positions": len(self._positions) if not model_id else (1 if model_id in self._positions else 0),
            "long_trades": len([p for p in history if p.direction == PositionDirection.LONG]),
            "short_trades": len([p for p in history if p.direction == PositionDirection.SHORT]),
        }

    # =========================================================================
    # Observer Pattern
    # =========================================================================

    def add_listener(self, listener: PositionChangeListener) -> None:
        """Add a position change listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: PositionChangeListener) -> None:
        """Remove a position change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_position_opened(self, position: Position) -> None:
        """Notify listeners of position open."""
        for listener in self._listeners:
            try:
                listener.on_position_opened(position)
            except Exception as e:
                logger.error(f"Error notifying listener of position open: {e}")

    def _notify_position_closed(self, position: Position) -> None:
        """Notify listeners of position close."""
        for listener in self._listeners:
            try:
                listener.on_position_closed(position)
            except Exception as e:
                logger.error(f"Error notifying listener of position close: {e}")

    # =========================================================================
    # State Management
    # =========================================================================

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self._positions.clear()
        self._history.clear()
        self._position_counter = 0
        logger.info("PositionTracker reset")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state to dictionary."""
        return {
            "positions": {
                k: v.to_dict() for k, v in self._positions.items()
            },
            "history": [p.to_dict() for p in self._history],
            "position_counter": self._position_counter,
            "statistics": self.get_statistics(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionTracker':
        """Restore tracker from dictionary."""
        tracker = cls()
        tracker._position_counter = data.get("position_counter", 0)

        for model_id, pos_data in data.get("positions", {}).items():
            tracker._positions[model_id] = Position.from_dict(pos_data)

        for pos_data in data.get("history", []):
            tracker._history.append(Position.from_dict(pos_data))

        return tracker

    def __repr__(self) -> str:
        return (
            f"PositionTracker(open={len(self._positions)}, "
            f"history={len(self._history)})"
        )
