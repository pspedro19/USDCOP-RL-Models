"""
Smart Executor — Stateless state machine for intraday execution.
================================================================

Bridges L5c forecasting signals -> TrailingStopTracker -> BrokerAdapter.

Each method reads an ExecutionState, performs an action, and returns an
updated state. No DB I/O happens here — Airflow tasks handle persistence
to forecast_executions. This keeps the executor purely testable.

The TrailingStopTracker is reconstructed from persisted state (peak_price,
trailing_state) on each call, so no historical bars are needed.

PnL = direction * leverage * (exit_price - entry_price) / entry_price
(leveraged, consistent with vol-targeting paper trading).

@version 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.execution.broker_adapter import BrokerAdapter, OrderSide, OrderStatus
from src.execution.trailing_stop import (
    TrailingState,
    TrailingStopConfig,
    TrailingStopTracker,
)


class ExecutionStatus(Enum):
    IDLE = "idle"
    POSITIONED = "positioned"
    MONITORING = "monitoring"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class ExecutionState:
    """Full execution state — mirrors forecast_executions table columns."""
    signal_date: str
    status: ExecutionStatus = ExecutionStatus.IDLE
    direction: int = 0         # +1 long, -1 short
    leverage: float = 1.0
    entry_price: Optional[float] = None
    entry_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None
    peak_price: Optional[float] = None
    trailing_state: str = "waiting"
    bar_count: int = 0
    pnl_pct: Optional[float] = None
    pnl_unleveraged_pct: Optional[float] = None
    config_version: str = "smart_executor_v1"


@dataclass(frozen=True)
class SmartExecutorConfig:
    """Frozen config for the smart executor. Loaded from YAML at startup."""
    activation_pct: float = 0.002
    trail_pct: float = 0.003
    hard_stop_pct: float = 0.015
    slippage_bps: float = 1.0
    config_version: str = "smart_executor_v1"


class SmartExecutor:
    """
    Stateless executor: each method takes state in, returns state out.

    Usage:
        executor = SmartExecutor(config, broker)
        state = executor.enter_position(...)
        # persist state to DB
        state = executor.monitor_bar(state, ...)
        # persist again
    """

    def __init__(self, config: SmartExecutorConfig, broker: BrokerAdapter):
        self.config = config
        self.broker = broker
        self._trailing_config = TrailingStopConfig(
            activation_pct=config.activation_pct,
            trail_pct=config.trail_pct,
            hard_stop_pct=config.hard_stop_pct,
        )

    def enter_position(
        self,
        signal_date: str,
        direction: int,
        leverage: float,
        entry_price: float,
    ) -> ExecutionState:
        """
        Open a new position via the broker.

        Args:
            signal_date: Date of the forecasting signal (YYYY-MM-DD).
            direction: +1 for long, -1 for short.
            leverage: Position leverage from vol-targeting.
            entry_price: Price to enter at (daily close or next open).

        Returns:
            ExecutionState with status=POSITIONED or ERROR.
        """
        assert direction in (1, -1), f"direction must be 1 or -1, got {direction}"
        assert leverage > 0, f"leverage must be positive, got {leverage}"
        assert entry_price > 0, f"entry_price must be positive, got {entry_price}"

        side = OrderSide.BUY if direction == 1 else OrderSide.SELL
        result = self.broker.place_order(side, entry_price, quantity=leverage)

        if result.status != OrderStatus.FILLED:
            return ExecutionState(
                signal_date=signal_date,
                status=ExecutionStatus.ERROR,
                direction=direction,
                leverage=leverage,
                config_version=self.config.config_version,
            )

        return ExecutionState(
            signal_date=signal_date,
            status=ExecutionStatus.POSITIONED,
            direction=direction,
            leverage=leverage,
            entry_price=result.fill_price,
            entry_timestamp=result.timestamp,
            peak_price=result.fill_price,
            trailing_state="waiting",
            bar_count=0,
            config_version=self.config.config_version,
        )

    def monitor_bar(
        self,
        state: ExecutionState,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_idx: int,
    ) -> ExecutionState:
        """
        Process one 5-min bar through the trailing stop tracker.

        Reconstructs the tracker from persisted state (peak_price + trailing_state),
        feeds the bar, and returns updated state.

        Args:
            state: Current execution state (from DB).
            bar_high/low/close: OHLC of the current 5-min bar.
            bar_idx: Sequential bar index within the session.

        Returns:
            Updated ExecutionState. Status may transition to CLOSED if triggered.
        """
        if state.status not in (ExecutionStatus.POSITIONED, ExecutionStatus.MONITORING):
            return state

        tracker = self._reconstruct_tracker(state)
        tracker_state = tracker.update(bar_high, bar_low, bar_close, bar_idx)

        new_state = ExecutionState(
            signal_date=state.signal_date,
            status=state.status,
            direction=state.direction,
            leverage=state.leverage,
            entry_price=state.entry_price,
            entry_timestamp=state.entry_timestamp,
            peak_price=tracker.peak_price,
            trailing_state=tracker.state.value,
            bar_count=state.bar_count + 1,
            config_version=state.config_version,
        )

        if tracker_state == TrailingState.TRIGGERED:
            new_state.status = ExecutionStatus.CLOSED
            new_state.exit_price = tracker.exit_price
            new_state.exit_timestamp = datetime.now(timezone.utc)
            new_state.exit_reason = tracker.exit_reason
            self._compute_pnl(new_state)
            self.broker.close_position(bar_close)
        elif tracker.state == TrailingState.ACTIVE:
            new_state.status = ExecutionStatus.MONITORING
        else:
            new_state.status = ExecutionStatus.POSITIONED

        return new_state

    def expire_session(
        self, state: ExecutionState, last_close: float
    ) -> ExecutionState:
        """
        Close position at session end (12:55 COT) if still open.

        Args:
            state: Current execution state.
            last_close: Close price of the final session bar.

        Returns:
            ExecutionState with status=CLOSED, exit_reason='session_close'.
        """
        if state.status not in (ExecutionStatus.POSITIONED, ExecutionStatus.MONITORING):
            return state

        close_result = self.broker.close_position(last_close)

        new_state = ExecutionState(
            signal_date=state.signal_date,
            status=ExecutionStatus.CLOSED,
            direction=state.direction,
            leverage=state.leverage,
            entry_price=state.entry_price,
            entry_timestamp=state.entry_timestamp,
            peak_price=state.peak_price,
            trailing_state="expired",
            bar_count=state.bar_count,
            exit_price=close_result.fill_price if close_result.status == OrderStatus.FILLED else last_close,
            exit_timestamp=datetime.now(timezone.utc),
            exit_reason="session_close",
            config_version=state.config_version,
        )
        self._compute_pnl(new_state)
        return new_state

    @staticmethod
    def should_monitor(state: ExecutionState) -> bool:
        """Return True if the position is still open and needs monitoring."""
        return state.status in (ExecutionStatus.POSITIONED, ExecutionStatus.MONITORING)

    def _reconstruct_tracker(self, state: ExecutionState) -> TrailingStopTracker:
        """
        Reconstruct a TrailingStopTracker from persisted state.

        Only needs entry_price, direction, peak_price, and trailing_state
        to fully restore the tracker — no historical bars required.
        """
        tracker = TrailingStopTracker(
            entry_price=state.entry_price,
            direction=state.direction,
            config=self._trailing_config,
        )
        tracker.peak_price = state.peak_price
        persisted = state.trailing_state
        if persisted == "active":
            tracker.state = TrailingState.ACTIVE
        elif persisted == "triggered":
            tracker.state = TrailingState.TRIGGERED
        elif persisted == "expired":
            tracker.state = TrailingState.EXPIRED
        # else: stays WAITING (default)
        return tracker

    @staticmethod
    def _compute_pnl(state: ExecutionState) -> None:
        """Compute leveraged and unleveraged PnL % in-place."""
        if state.entry_price is None or state.exit_price is None:
            return
        raw_pnl = state.direction * (state.exit_price - state.entry_price) / state.entry_price
        state.pnl_unleveraged_pct = raw_pnl
        state.pnl_pct = raw_pnl * state.leverage
