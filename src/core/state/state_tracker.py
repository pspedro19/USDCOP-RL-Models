"""
StateTracker - State Management for RL Trading Models
======================================================

Manages position state, P&L tracking, and state features for the
V19/V20 USDCOP trading system.

Author: Pedro @ Lean Tech Solutions / Claude Code
Version: 20.0.0
Date: 2026-01-09

Features:
    - ModelState dataclass for immutable state snapshots
    - StateTracker for mutable state management
    - V20 FIX: Redis + PostgreSQL persistence (state survives restarts)
    - Thread-safe operations

V20 Changes:
    - Implemented _persist_state() with Redis + PostgreSQL dual-write
    - Implemented _load_state() with Redis fast-read, PostgreSQL fallback
    - State now survives container restarts
"""

import math
import threading
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    """
    Immutable state snapshot for a trading model.

    Represents the complete state of a trading model at a point in time,
    including position, P&L metrics, and session statistics.

    Attributes:
        model_id: Unique identifier for the model
        position: Current position (-1=short, 0=flat, 1=long)
        entry_price: Price at which current position was entered
        unrealized_pnl: Unrealized profit/loss (current position)
        realized_pnl: Total realized profit/loss (closed trades)
        current_equity: Current account equity
        peak_equity: Maximum equity reached (for drawdown calculation)
        current_drawdown: Current drawdown from peak (as decimal, e.g., 0.05 = 5%)
        trade_count_session: Number of trades in current session
        bars_in_position: Number of bars since position was opened
        last_updated: Timestamp of last state update
        metadata: Optional additional metadata

    Example:
        >>> state = ModelState(model_id="ppo_v19_usdcop")
        >>> state.position
        0.0
        >>> state.current_equity
        10000.0
    """

    model_id: str
    position: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    current_equity: float = 10000.0
    peak_equity: float = 10000.0
    current_drawdown: float = 0.0
    trade_count_session: int = 0
    bars_in_position: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate state after initialization."""
        # Ensure position is in valid range
        self.position = max(-1.0, min(1.0, self.position))

        # Ensure non-negative values
        self.current_equity = max(0.0, self.current_equity)
        self.peak_equity = max(self.current_equity, self.peak_equity)
        self.trade_count_session = max(0, self.trade_count_session)
        self.bars_in_position = max(0, self.bars_in_position)

        # Calculate drawdown if not set
        if self.current_drawdown == 0.0 and self.peak_equity > 0:
            self.current_drawdown = max(
                0.0,
                (self.peak_equity - self.current_equity) / self.peak_equity
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary.

        Returns:
            Dictionary representation of state
        """
        d = asdict(self)
        d['last_updated'] = self.last_updated.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelState':
        """
        Create ModelState from dictionary.

        Args:
            data: Dictionary with state data

        Returns:
            New ModelState instance
        """
        data = data.copy()
        if isinstance(data.get('last_updated'), str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return abs(self.position) < 0.01

    def is_long(self) -> bool:
        """Check if position is long."""
        return self.position > 0.01

    def is_short(self) -> bool:
        """Check if position is short."""
        return self.position < -0.01


class StateTracker:
    """
    Manages mutable state for multiple trading models.

    Provides thread-safe state management with optional database persistence.
    Tracks positions, P&L, and generates state features for observations.

    Attributes:
        DEFAULT_EQUITY: Default starting equity (10000.0)
        DEFAULT_EPISODE_BARS: Default episode length in bars (60)

    Example:
        >>> tracker = StateTracker()
        >>> tracker.update_position("ppo_v19", 1.0, 4200.0)  # Go long
        >>> position, time_norm = tracker.get_state_features("ppo_v19", 30, 60)
        >>> position
        1.0
        >>> time_norm
        0.5
    """

    DEFAULT_EQUITY: float = 10000.0
    DEFAULT_EPISODE_BARS: int = 60

    def __init__(
        self,
        db_connection: Optional[Any] = None,
        initial_equity: float = 10000.0
    ):
        """
        Initialize the StateTracker.

        Args:
            db_connection: Optional database connection for persistence
            initial_equity: Starting equity for new models (default: 10000.0)
        """
        self._db = db_connection
        self._initial_equity = initial_equity
        self._states: Dict[str, ModelState] = {}
        self._lock = threading.RLock()
        self._history: Dict[str, List[Dict[str, Any]]] = {}

    def get_state_features(
        self,
        model_id: str,
        current_bar: int,
        total_bars: int = 60
    ) -> Tuple[float, float]:
        """
        Get state features for observation vector.

        Returns the position and time_normalized values needed for
        the V19 observation space (indices 13 and 14).

        Args:
            model_id: Unique identifier for the model
            current_bar: Current bar number in episode (1-based)
            total_bars: Total bars in episode (default: 60)

        Returns:
            Tuple of (position, time_normalized)
            - position: Current position (-1 to 1)
            - time_normalized: Normalized time (0 to 1)

        Formula:
            time_normalized = (current_bar - 1) / total_bars

        Note:
            - Creates new state if model_id not found
            - time_normalized is clipped to [0, 1]
        """
        state = self.get_or_create(model_id)

        # Ensure safe values
        position = self._safe_float(state.position)
        position = max(-1.0, min(1.0, position))

        # Calculate time normalized
        if total_bars <= 0:
            total_bars = self.DEFAULT_EPISODE_BARS
        time_normalized = (current_bar - 1) / total_bars
        time_normalized = max(0.0, min(1.0, time_normalized))

        return (position, time_normalized)

    def update_position(
        self,
        model_id: str,
        new_position: float,
        current_price: float
    ) -> ModelState:
        """
        Update position for a model.

        Handles position changes, P&L calculation, and state updates.

        Args:
            model_id: Unique identifier for the model
            new_position: New position value (-1 to 1)
            current_price: Current market price

        Returns:
            Updated ModelState

        Note:
            - Clips position to [-1, 1]
            - Calculates realized P&L on position close
            - Updates unrealized P&L for open positions
            - Increments trade count on position change
        """
        with self._lock:
            state = self.get_or_create(model_id)

            # Clip new position
            new_position = max(-1.0, min(1.0, new_position))

            # Get old values
            old_position = state.position
            entry_price = state.entry_price

            # Detect position change
            position_changed = abs(new_position - old_position) > 0.01

            # Calculate P&L
            realized_pnl = state.realized_pnl
            unrealized_pnl = 0.0

            if position_changed:
                # Closing existing position?
                if abs(old_position) > 0.01 and entry_price > 0:
                    # Calculate realized P&L for closed position
                    if old_position > 0:  # Was long
                        pnl = (current_price - entry_price) / entry_price
                    else:  # Was short
                        pnl = (entry_price - current_price) / entry_price

                    realized_pnl += pnl * abs(old_position) * state.current_equity

                # Opening new position?
                if abs(new_position) > 0.01:
                    entry_price = current_price
                else:
                    entry_price = 0.0

                # Increment trade count
                trade_count = state.trade_count_session + 1
                bars_in_position = 0
            else:
                # Position unchanged, update unrealized P&L
                if abs(old_position) > 0.01 and entry_price > 0:
                    if old_position > 0:  # Long
                        unrealized_pnl = ((current_price - entry_price) / entry_price) * \
                                         abs(old_position) * state.current_equity
                    else:  # Short
                        unrealized_pnl = ((entry_price - current_price) / entry_price) * \
                                         abs(old_position) * state.current_equity

                trade_count = state.trade_count_session
                bars_in_position = state.bars_in_position + 1

            # Update equity
            current_equity = state.current_equity + (realized_pnl - state.realized_pnl)
            peak_equity = max(state.peak_equity, current_equity)

            # Calculate drawdown
            if peak_equity > 0:
                current_drawdown = (peak_equity - current_equity) / peak_equity
            else:
                current_drawdown = 0.0

            # Create new state
            new_state = ModelState(
                model_id=model_id,
                position=new_position,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                current_equity=current_equity,
                peak_equity=peak_equity,
                current_drawdown=current_drawdown,
                trade_count_session=trade_count,
                bars_in_position=bars_in_position,
                last_updated=datetime.utcnow(),
                metadata=state.metadata.copy()
            )

            self._states[model_id] = new_state
            self._record_history(model_id, new_state)

            # Persist if database available
            if self._db is not None:
                self._persist_state(new_state)

            return new_state

    def get_or_create(self, model_id: str) -> ModelState:
        """
        Get existing state or create new one.

        Args:
            model_id: Unique identifier for the model

        Returns:
            ModelState for the model
        """
        with self._lock:
            if model_id not in self._states:
                # Try to load from database
                if self._db is not None:
                    state = self._load_state(model_id)
                    if state is not None:
                        self._states[model_id] = state
                        return state

                # Create new state
                self._states[model_id] = ModelState(
                    model_id=model_id,
                    current_equity=self._initial_equity,
                    peak_equity=self._initial_equity
                )

            return self._states[model_id]

    def get_state(self, model_id: str) -> Optional[ModelState]:
        """
        Get state for a model without creating new one.

        Args:
            model_id: Unique identifier for the model

        Returns:
            ModelState if exists, None otherwise
        """
        with self._lock:
            return self._states.get(model_id)

    def reset_state(self, model_id: str) -> ModelState:
        """
        Reset state for a model to initial values.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Fresh ModelState
        """
        with self._lock:
            new_state = ModelState(
                model_id=model_id,
                current_equity=self._initial_equity,
                peak_equity=self._initial_equity
            )
            self._states[model_id] = new_state
            return new_state

    def reset_all(self) -> None:
        """Reset all tracked states."""
        with self._lock:
            self._states.clear()
            self._history.clear()

    def get_all_models(self) -> List[str]:
        """
        Get list of all tracked model IDs.

        Returns:
            List of model IDs
        """
        with self._lock:
            return list(self._states.keys())

    def get_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get state history for a model.

        Args:
            model_id: Unique identifier for the model

        Returns:
            List of historical state dictionaries
        """
        with self._lock:
            return deepcopy(self._history.get(model_id, []))

    def increment_bar(self, model_id: str) -> None:
        """
        Increment bars_in_position counter.

        Args:
            model_id: Unique identifier for the model
        """
        with self._lock:
            if model_id in self._states:
                state = self._states[model_id]
                # Create new state with incremented bar count
                self._states[model_id] = ModelState(
                    model_id=state.model_id,
                    position=state.position,
                    entry_price=state.entry_price,
                    unrealized_pnl=state.unrealized_pnl,
                    realized_pnl=state.realized_pnl,
                    current_equity=state.current_equity,
                    peak_equity=state.peak_equity,
                    current_drawdown=state.current_drawdown,
                    trade_count_session=state.trade_count_session,
                    bars_in_position=state.bars_in_position + 1,
                    last_updated=datetime.utcnow(),
                    metadata=state.metadata.copy()
                )

    def _safe_float(self, value: Any) -> float:
        """
        Safely convert value to float.

        Args:
            value: Input value

        Returns:
            Float value, 0.0 if None/NaN
        """
        if value is None:
            return 0.0
        try:
            f = float(value)
            if math.isnan(f):
                return 0.0
            return f
        except (TypeError, ValueError):
            return 0.0

    def _record_history(self, model_id: str, state: ModelState) -> None:
        """
        Record state in history.

        Args:
            model_id: Model identifier
            state: State to record
        """
        if model_id not in self._history:
            self._history[model_id] = []

        # Limit history size
        max_history = 1000
        if len(self._history[model_id]) >= max_history:
            self._history[model_id] = self._history[model_id][-(max_history - 1):]

        self._history[model_id].append(state.to_dict())

    def _persist_state(self, state: ModelState) -> None:
        """
        V20 FIX: Persist state to Redis (fast) and PostgreSQL (durable).

        Dual-write strategy:
        1. Write to Redis for fast reads (TTL: 24h)
        2. Write to PostgreSQL for durability (survives Redis restart)

        Args:
            state: State to persist
        """
        state_dict = state.to_dict()
        model_id = state.model_id

        # 1. Write to Redis (fast, but may lose on restart)
        try:
            import redis
            redis_host = os.environ.get('REDIS_HOST', 'redis')
            redis_port = int(os.environ.get('REDIS_PORT', '6379'))
            r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

            redis_key = f"state:trading:{model_id}"
            r.setex(redis_key, 86400, json.dumps(state_dict))  # 24h TTL
            logger.debug(f"State persisted to Redis: {model_id}")
        except Exception as e:
            logger.warning(f"Redis persist failed for {model_id}: {e}")

        # 2. Write to PostgreSQL (durable, survives all restarts)
        try:
            if self._db is not None:
                cur = self._db.cursor()
                cur.execute("""
                    INSERT INTO trading.model_state
                    (model_id, position, entry_price, unrealized_pnl, realized_pnl,
                     current_equity, peak_equity, current_drawdown, trade_count_session,
                     bars_in_position, last_updated, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_id) DO UPDATE SET
                        position = EXCLUDED.position,
                        entry_price = EXCLUDED.entry_price,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        realized_pnl = EXCLUDED.realized_pnl,
                        current_equity = EXCLUDED.current_equity,
                        peak_equity = EXCLUDED.peak_equity,
                        current_drawdown = EXCLUDED.current_drawdown,
                        trade_count_session = EXCLUDED.trade_count_session,
                        bars_in_position = EXCLUDED.bars_in_position,
                        last_updated = EXCLUDED.last_updated,
                        metadata = EXCLUDED.metadata
                """, (
                    model_id,
                    state.position,
                    state.entry_price,
                    state.unrealized_pnl,
                    state.realized_pnl,
                    state.current_equity,
                    state.peak_equity,
                    state.current_drawdown,
                    state.trade_count_session,
                    state.bars_in_position,
                    state.last_updated,
                    json.dumps(state.metadata)
                ))
                self._db.commit()
                logger.debug(f"State persisted to PostgreSQL: {model_id}")
        except Exception as e:
            logger.warning(f"PostgreSQL persist failed for {model_id}: {e}")

    def _load_state(self, model_id: str) -> Optional[ModelState]:
        """
        V20 FIX: Load state from Redis (fast) or PostgreSQL (fallback).

        Read strategy:
        1. Try Redis first (fast)
        2. Fallback to PostgreSQL (if Redis miss or down)
        3. Return None if not found anywhere

        Args:
            model_id: Model identifier

        Returns:
            ModelState if found, None otherwise
        """
        # 1. Try Redis first (fast read)
        try:
            import redis
            redis_host = os.environ.get('REDIS_HOST', 'redis')
            redis_port = int(os.environ.get('REDIS_PORT', '6379'))
            r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

            redis_key = f"state:trading:{model_id}"
            state_json = r.get(redis_key)

            if state_json:
                state_dict = json.loads(state_json)
                logger.debug(f"State loaded from Redis: {model_id}")
                return ModelState.from_dict(state_dict)
        except Exception as e:
            logger.debug(f"Redis load failed for {model_id}: {e}")

        # 2. Fallback to PostgreSQL
        try:
            if self._db is not None:
                cur = self._db.cursor()
                cur.execute("""
                    SELECT model_id, position, entry_price, unrealized_pnl, realized_pnl,
                           current_equity, peak_equity, current_drawdown, trade_count_session,
                           bars_in_position, last_updated, metadata
                    FROM trading.model_state
                    WHERE model_id = %s
                """, (model_id,))

                row = cur.fetchone()
                if row:
                    state = ModelState(
                        model_id=row[0],
                        position=float(row[1]) if row[1] else 0.0,
                        entry_price=float(row[2]) if row[2] else 0.0,
                        unrealized_pnl=float(row[3]) if row[3] else 0.0,
                        realized_pnl=float(row[4]) if row[4] else 0.0,
                        current_equity=float(row[5]) if row[5] else self._initial_equity,
                        peak_equity=float(row[6]) if row[6] else self._initial_equity,
                        current_drawdown=float(row[7]) if row[7] else 0.0,
                        trade_count_session=int(row[8]) if row[8] else 0,
                        bars_in_position=int(row[9]) if row[9] else 0,
                        last_updated=row[10] if row[10] else datetime.utcnow(),
                        metadata=json.loads(row[11]) if row[11] else {}
                    )
                    logger.debug(f"State loaded from PostgreSQL: {model_id}")

                    # Re-populate Redis for next fast read
                    self._persist_to_redis_only(state)
                    return state
        except Exception as e:
            logger.warning(f"PostgreSQL load failed for {model_id}: {e}")

        return None

    def _persist_to_redis_only(self, state: ModelState) -> None:
        """Helper to write state to Redis only (for cache warming)."""
        try:
            import redis
            redis_host = os.environ.get('REDIS_HOST', 'redis')
            redis_port = int(os.environ.get('REDIS_PORT', '6379'))
            r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

            redis_key = f"state:trading:{state.model_id}"
            r.setex(redis_key, 86400, json.dumps(state.to_dict()))
        except Exception:
            pass  # Silent fail for cache warming

    def __repr__(self) -> str:
        return (
            f"StateTracker(models={len(self._states)}, "
            f"initial_equity={self._initial_equity})"
        )


# Factory function
def create_state_tracker(
    db_connection: Optional[Any] = None,
    initial_equity: float = 10000.0
) -> StateTracker:
    """
    Factory function to create a StateTracker instance.

    Args:
        db_connection: Optional database connection
        initial_equity: Starting equity for new models

    Returns:
        Configured StateTracker instance
    """
    return StateTracker(db_connection=db_connection, initial_equity=initial_equity)
