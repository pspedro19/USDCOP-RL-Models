"""
Multi-Day Executor — Stateless state machine for weekly H=5 execution.
======================================================================

Manages a 5-day position with trailing stop + re-entry.
Same pattern as SmartExecutor: each method takes state in, returns state out.
No DB I/O — Airflow DAG handles persistence.

Key differences from SmartExecutor (intraday):
    - Position held across multiple days (Mon-Fri)
    - Tight trailing: activation 0.20%, trail 0.10%, hard_stop 3.50% (v2)
    - Re-entry enabled: after trailing exit, re-enter same direction
      with 20-minute cooldown to avoid churn in lateral markets
    - Week-end close: Friday 12:50 COT regardless of trailing state
    - Sub-trade tracking: each entry/re-entry is a separate subtrade

PnL = direction * leverage * (exit_price - entry_price) / entry_price
(leveraged, consistent with Track A).

@version 1.0.0
@contract FC-H5-EXEC-001
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional, Tuple

from src.execution.trailing_stop import (
    TrailingState,
    TrailingStopConfig,
    TrailingStopTracker,
)


class WeekStatus(Enum):
    """Status of the weekly execution."""
    PENDING = "pending"            # Signal generated, no entry yet
    POSITIONED = "positioned"      # Active subtrade (waiting for trailing activation)
    MONITORING = "monitoring"      # Active subtrade (trailing activated)
    COOLDOWN = "cooldown"          # After trailing exit, waiting to re-enter
    CLOSED = "closed"              # Week closed (all subtrades done)
    PAUSED = "paused"              # Circuit breaker triggered
    ERROR = "error"


@dataclass
class SubtradeState:
    """State of a single subtrade within a week."""
    subtrade_index: int = 0
    direction: int = 0               # +1 long, -1 short
    entry_price: float = 0.0
    entry_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None
    peak_price: float = 0.0
    trailing_state: str = "waiting"   # waiting/active/triggered/expired
    bar_count: int = 0
    pnl_pct: Optional[float] = None
    pnl_unleveraged_pct: Optional[float] = None
    cooldown_until: Optional[datetime] = None


@dataclass
class WeekExecutionState:
    """Full weekly execution state — mirrors forecast_h5_executions table."""
    signal_date: str                    # Monday signal date (YYYY-MM-DD)
    direction: int = 0                  # +1 long, -1 short
    leverage: float = 1.0
    status: WeekStatus = WeekStatus.PENDING
    entry_price: Optional[float] = None   # First subtrade entry
    entry_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None    # Last subtrade exit
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None
    subtrades: List[SubtradeState] = field(default_factory=list)
    week_pnl_pct: Optional[float] = None
    week_pnl_unleveraged_pct: Optional[float] = None
    config_version: str = "smart_executor_h5_v2"


@dataclass(frozen=True)
class MultiDayConfig:
    """Frozen config for the multi-day executor."""
    activation_pct: float = 0.002     # 0.20% (v2: tight trailing)
    trail_pct: float = 0.001          # 0.10% (v2: micro-profit capture)
    hard_stop_pct: float = 0.035      # 3.50% (v2: tighter hard stop)
    cooldown_minutes: int = 20        # 20 min = 4 bars at 5min
    slippage_bps: float = 1.0
    config_version: str = "smart_executor_h5_v2"


class MultiDayExecutor:
    """
    Stateless weekly executor with trailing stop + re-entry.

    Usage:
        executor = MultiDayExecutor(config)
        state = executor.enter(signal_date, direction, leverage, price, ts)
        # persist state to DB
        state, event = executor.update(state, bar_high, bar_low, bar_close, bar_ts)
        # if event == 'trailing_exit': persist subtrade, check cooldown
        # if event == 're_entry_ready': call enter_subtrade()
        state = executor.close_week(state, last_close, ts)
    """

    def __init__(self, config: MultiDayConfig):
        self.config = config
        self._trailing_config = TrailingStopConfig(
            activation_pct=config.activation_pct,
            trail_pct=config.trail_pct,
            hard_stop_pct=config.hard_stop_pct,
        )

    def enter(
        self,
        signal_date: str,
        direction: int,
        leverage: float,
        entry_price: float,
        entry_timestamp: datetime,
    ) -> WeekExecutionState:
        """
        Open initial position for the week.

        Args:
            signal_date: Monday signal date (YYYY-MM-DD).
            direction: +1 for long, -1 for short.
            leverage: Asymmetric leverage from vol-targeting.
            entry_price: Price at entry.
            entry_timestamp: Timestamp of entry bar.

        Returns:
            WeekExecutionState with first subtrade.
        """
        assert direction in (1, -1), f"direction must be 1 or -1, got {direction}"
        assert leverage > 0, f"leverage must be positive, got {leverage}"
        assert entry_price > 0, f"entry_price must be positive, got {entry_price}"

        subtrade = SubtradeState(
            subtrade_index=0,
            direction=direction,
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            peak_price=entry_price,
            trailing_state="waiting",
            bar_count=0,
        )

        return WeekExecutionState(
            signal_date=signal_date,
            direction=direction,
            leverage=leverage,
            status=WeekStatus.POSITIONED,
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            subtrades=[subtrade],
            config_version=self.config.config_version,
        )

    def enter_subtrade(
        self,
        state: WeekExecutionState,
        entry_price: float,
        entry_timestamp: datetime,
    ) -> WeekExecutionState:
        """
        Re-enter after trailing exit (same direction, new subtrade).

        Args:
            state: Current week execution state (must be in COOLDOWN with expired cooldown).
            entry_price: Price for re-entry.
            entry_timestamp: Timestamp of re-entry bar.

        Returns:
            Updated state with new subtrade appended.
        """
        new_index = len(state.subtrades)
        subtrade = SubtradeState(
            subtrade_index=new_index,
            direction=state.direction,
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            peak_price=entry_price,
            trailing_state="waiting",
            bar_count=0,
        )

        state.subtrades.append(subtrade)
        state.status = WeekStatus.POSITIONED
        return state

    def update(
        self,
        state: WeekExecutionState,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_timestamp: datetime,
    ) -> Tuple[WeekExecutionState, Optional[str]]:
        """
        Process one bar through the active subtrade's trailing stop.

        Args:
            state: Current week execution state.
            bar_high/bar_low/bar_close: OHLC of current bar.
            bar_timestamp: Timestamp of the bar.

        Returns:
            (updated_state, event) where event is:
              - None: no state change
              - 'trailing_exit': trailing stop fired, subtrade closed
              - 'hard_stop': hard stop fired, subtrade closed
              - 're_entry_ready': cooldown expired, ready for re-entry
        """
        if state.status == WeekStatus.COOLDOWN:
            active_sub = self._get_last_closed_subtrade(state)
            if active_sub and active_sub.cooldown_until:
                if bar_timestamp >= active_sub.cooldown_until:
                    return state, "re_entry_ready"
            return state, None

        if state.status not in (WeekStatus.POSITIONED, WeekStatus.MONITORING):
            return state, None

        active_sub = self._get_active_subtrade(state)
        if active_sub is None:
            return state, None

        tracker = self._reconstruct_tracker(active_sub)
        tracker_state = tracker.update(bar_high, bar_low, bar_close, active_sub.bar_count)

        # Update subtrade from tracker
        active_sub.peak_price = tracker.peak_price
        active_sub.trailing_state = tracker.state.value
        active_sub.bar_count += 1

        if tracker_state == TrailingState.TRIGGERED:
            # Subtrade closed by trailing/hard stop
            active_sub.exit_price = tracker.exit_price
            active_sub.exit_timestamp = bar_timestamp
            active_sub.exit_reason = tracker.exit_reason
            self._compute_subtrade_pnl(active_sub, state.leverage)

            # Set cooldown for potential re-entry
            active_sub.cooldown_until = bar_timestamp + timedelta(
                minutes=self.config.cooldown_minutes
            )
            state.status = WeekStatus.COOLDOWN

            event = "hard_stop" if tracker.exit_reason == "hard_stop" else "trailing_exit"
            return state, event

        elif tracker.state == TrailingState.ACTIVE:
            state.status = WeekStatus.MONITORING
        else:
            state.status = WeekStatus.POSITIONED

        return state, None

    def close_week(
        self,
        state: WeekExecutionState,
        last_close: float,
        close_timestamp: datetime,
    ) -> WeekExecutionState:
        """
        Close any remaining position at week end (Friday 12:50 COT).

        Args:
            state: Current week state.
            last_close: Close price for week-end exit.
            close_timestamp: Friday close timestamp.

        Returns:
            Updated state with status=CLOSED and aggregated PnL.
        """
        # Close active subtrade if any
        active_sub = self._get_active_subtrade(state)
        if active_sub is not None:
            active_sub.exit_price = last_close
            active_sub.exit_timestamp = close_timestamp
            active_sub.exit_reason = "week_end"
            active_sub.trailing_state = "expired"
            self._compute_subtrade_pnl(active_sub, state.leverage)

        # Aggregate week PnL from all subtrades
        state.status = WeekStatus.CLOSED
        state.exit_price = last_close
        state.exit_timestamp = close_timestamp
        state.exit_reason = "week_end"
        self._compute_week_pnl(state)

        return state

    def close_circuit_breaker(
        self,
        state: WeekExecutionState,
        close_price: float,
        close_timestamp: datetime,
    ) -> WeekExecutionState:
        """Close position due to circuit breaker trigger."""
        active_sub = self._get_active_subtrade(state)
        if active_sub is not None:
            active_sub.exit_price = close_price
            active_sub.exit_timestamp = close_timestamp
            active_sub.exit_reason = "circuit_breaker"
            active_sub.trailing_state = "triggered"
            self._compute_subtrade_pnl(active_sub, state.leverage)

        state.status = WeekStatus.PAUSED
        state.exit_price = close_price
        state.exit_timestamp = close_timestamp
        state.exit_reason = "circuit_breaker"
        self._compute_week_pnl(state)
        return state

    @staticmethod
    def should_monitor(state: WeekExecutionState) -> bool:
        """Return True if the week has an active position needing monitoring."""
        return state.status in (
            WeekStatus.POSITIONED,
            WeekStatus.MONITORING,
            WeekStatus.COOLDOWN,
        )

    @staticmethod
    def should_close_week(bar_timestamp: datetime, week_end_time: datetime) -> bool:
        """Return True if current bar is at or past the week-end close time."""
        return bar_timestamp >= week_end_time

    @staticmethod
    def compute_week_pnl(subtrades: List[SubtradeState], leverage: float) -> float:
        """
        Compute total leveraged PnL for the week from all subtrades.

        Args:
            subtrades: List of completed subtrades.
            leverage: Position leverage.

        Returns:
            Total leveraged PnL as a percentage.
        """
        total = 0.0
        for sub in subtrades:
            if sub.pnl_pct is not None:
                total += sub.pnl_pct
        return total

    def _get_active_subtrade(self, state: WeekExecutionState) -> Optional[SubtradeState]:
        """Get the currently open subtrade (no exit_price yet)."""
        for sub in reversed(state.subtrades):
            if sub.exit_price is None:
                return sub
        return None

    def _get_last_closed_subtrade(self, state: WeekExecutionState) -> Optional[SubtradeState]:
        """Get the most recently closed subtrade."""
        for sub in reversed(state.subtrades):
            if sub.exit_price is not None:
                return sub
        return None

    def _reconstruct_tracker(self, sub: SubtradeState) -> TrailingStopTracker:
        """Reconstruct a TrailingStopTracker from persisted subtrade state."""
        tracker = TrailingStopTracker(
            entry_price=sub.entry_price,
            direction=sub.direction,
            config=self._trailing_config,
        )
        tracker.peak_price = sub.peak_price
        state_map = {
            "active": TrailingState.ACTIVE,
            "triggered": TrailingState.TRIGGERED,
            "expired": TrailingState.EXPIRED,
        }
        tracker.state = state_map.get(sub.trailing_state, TrailingState.WAITING)
        return tracker

    @staticmethod
    def _compute_subtrade_pnl(sub: SubtradeState, leverage: float) -> None:
        """Compute leveraged and unleveraged PnL for a subtrade."""
        if sub.entry_price is None or sub.exit_price is None:
            return
        raw_pnl = sub.direction * (sub.exit_price - sub.entry_price) / sub.entry_price
        sub.pnl_unleveraged_pct = raw_pnl
        sub.pnl_pct = raw_pnl * leverage

    @staticmethod
    def _compute_week_pnl(state: WeekExecutionState) -> None:
        """Aggregate PnL from all subtrades into week totals."""
        total_lev = 0.0
        total_unlev = 0.0
        for sub in state.subtrades:
            if sub.pnl_pct is not None:
                total_lev += sub.pnl_pct
            if sub.pnl_unleveraged_pct is not None:
                total_unlev += sub.pnl_unleveraged_pct
        state.week_pnl_pct = total_lev
        state.week_pnl_unleveraged_pct = total_unlev
