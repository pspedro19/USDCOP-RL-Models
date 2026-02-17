"""
Trailing Stop Tracker — Pure logic, no I/O, no side effects.
=============================================================

Monitors intraday 5-min bars after a daily forecasting signal and exits
early when profits reverse, protecting intraday peaks.

States:
    WAITING   -> price hasn't moved enough to activate trailing
    ACTIVE    -> peak PnL exceeded activation_pct, trailing is armed
    TRIGGERED -> drawback from peak exceeded trail_pct (or hard stop hit)
    EXPIRED   -> session ended without trigger, exit at session close

Usage:
    tracker = TrailingStopTracker(entry_price=4200.0, direction=1, config=cfg)
    for bar in intraday_bars:
        state = tracker.update(bar.high, bar.low, bar.close, bar_idx)
        if state == TrailingState.TRIGGERED:
            break
    if tracker.state != TrailingState.TRIGGERED:
        tracker.expire(last_bar_close)

@version 1.0.0
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TrailingState(Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"


@dataclass
class TrailingStopConfig:
    """Trailing stop parameters."""
    activation_pct: float = 0.002   # 0.20% move to arm the trail
    trail_pct: float = 0.003        # 0.30% drawback from peak triggers exit
    hard_stop_pct: float = 0.015    # 1.50% adverse move = unconditional exit


class TrailingStopTracker:
    """Tracks intraday trailing stop for a single position."""

    def __init__(
        self,
        entry_price: float,
        direction: int,
        config: TrailingStopConfig,
    ):
        assert direction in (1, -1), f"direction must be 1 or -1, got {direction}"
        assert entry_price > 0, f"entry_price must be positive, got {entry_price}"

        self.entry_price = entry_price
        self.direction = direction
        self.config = config

        self.state = TrailingState.WAITING
        self.peak_price: float = entry_price  # best favorable price seen
        self.exit_price: Optional[float] = None
        self.exit_bar_idx: Optional[int] = None
        self.exit_reason: Optional[str] = None

    def update(self, bar_high: float, bar_low: float, bar_close: float, bar_idx: int) -> TrailingState:
        """
        Process one 5-min bar. Returns current state after update.

        For LONG: peak tracks bar_high, hard stop checks bar_low.
        For SHORT: peak tracks bar_low (inverted), hard stop checks bar_high.
        """
        if self.state in (TrailingState.TRIGGERED, TrailingState.EXPIRED):
            return self.state

        # --- Hard stop check (before anything else) ---
        if self.direction == 1:
            adverse_pct = (self.entry_price - bar_low) / self.entry_price
        else:
            adverse_pct = (bar_high - self.entry_price) / self.entry_price

        if adverse_pct >= self.config.hard_stop_pct:
            self._trigger(
                exit_price=self._hard_stop_price(),
                bar_idx=bar_idx,
                reason="hard_stop",
                bar_low=bar_low,
                bar_high=bar_high,
            )
            return self.state

        # --- Update peak price ---
        if self.direction == 1:
            self.peak_price = max(self.peak_price, bar_high)
        else:
            self.peak_price = min(self.peak_price, bar_low)

        # --- Compute peak PnL % ---
        peak_pnl = self.direction * (self.peak_price - self.entry_price) / self.entry_price

        # --- WAITING -> ACTIVE transition ---
        if self.state == TrailingState.WAITING:
            if peak_pnl >= self.config.activation_pct:
                self.state = TrailingState.ACTIVE

        # --- ACTIVE -> TRIGGERED check ---
        if self.state == TrailingState.ACTIVE:
            if self.direction == 1:
                drawback = (self.peak_price - bar_low) / self.peak_price
            else:
                drawback = (bar_high - self.peak_price) / self.peak_price

            if drawback >= self.config.trail_pct:
                trail_exit = self._trail_exit_price()
                self._trigger(
                    exit_price=trail_exit,
                    bar_idx=bar_idx,
                    reason="trailing_stop",
                    bar_low=bar_low,
                    bar_high=bar_high,
                )

        return self.state

    def expire(self, close_price: float) -> None:
        """Called at session end if not triggered. Exit at session close."""
        if self.state in (TrailingState.TRIGGERED, TrailingState.EXPIRED):
            return
        self.state = TrailingState.EXPIRED
        self.exit_price = close_price
        self.exit_reason = "session_close"

    @property
    def pnl_pct(self) -> Optional[float]:
        """Return PnL % of the trade, or None if still open."""
        if self.exit_price is None:
            return None
        return self.direction * (self.exit_price - self.entry_price) / self.entry_price

    def _hard_stop_price(self) -> float:
        """Theoretical hard stop level."""
        if self.direction == 1:
            return self.entry_price * (1 - self.config.hard_stop_pct)
        else:
            return self.entry_price * (1 + self.config.hard_stop_pct)

    def _trail_exit_price(self) -> float:
        """Theoretical trailing exit level based on peak."""
        if self.direction == 1:
            return self.peak_price * (1 - self.config.trail_pct)
        else:
            return self.peak_price * (1 + self.config.trail_pct)

    def _trigger(
        self,
        exit_price: float,
        bar_idx: int,
        reason: str,
        bar_low: float,
        bar_high: float,
    ) -> None:
        """Mark position as triggered, clamping exit to bar range."""
        if self.direction == 1:
            # LONG exit can't be below bar_low (worst fill)
            exit_price = max(exit_price, bar_low)
        else:
            # SHORT exit can't be above bar_high (worst fill)
            exit_price = min(exit_price, bar_high)

        self.state = TrailingState.TRIGGERED
        self.exit_price = exit_price
        self.exit_bar_idx = bar_idx
        self.exit_reason = reason
