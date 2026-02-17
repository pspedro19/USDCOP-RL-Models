"""
Execution Strategies — Pluggable signal-to-trade execution.
============================================================

Each execution strategy simulates how a UniversalSignalRecord becomes a
StrategyTrade against OHLCV bars.

Strategies:
    WeeklyTPHSExecution       — H5 Smart Simple (TP/HS/Friday close)
    DailyTrailingStopExecution — H1 Forecast+VT+Trail (trailing stop intraday)
    IntradaySLTPExecution      — RL PPO (SL/TP per 5-min bar)

Contract: CTR-EXEC-STRATEGY-001
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from src.contracts.signal_contract import UniversalSignalRecord
from src.contracts.strategy_schema import StrategyTrade
from src.forecasting.adaptive_stops import check_hard_stop, check_take_profit, get_exit_price


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class ExecutionStrategy(ABC):
    """ABC for signal-to-trade execution."""

    @abstractmethod
    def execute(
        self,
        signal: UniversalSignalRecord,
        ohlcv: pd.DataFrame,
        equity: float,
    ) -> Optional[StrategyTrade]:
        """
        Execute a signal against OHLCV bars, return a trade or None.

        Args:
            signal: The universal signal to execute.
            ohlcv: DataFrame with columns [date, open, high, low, close].
            equity: Current portfolio equity.

        Returns:
            StrategyTrade if a trade was executed, None otherwise.
        """


# ---------------------------------------------------------------------------
# H5 Weekly: TP/HS/Friday Close
# ---------------------------------------------------------------------------

@dataclass
class WeeklyTPHSConfig:
    """Configuration for weekly TP/HS execution."""
    maker_fee: float = 0.0      # 0% maker fee (MEXC)
    slippage_bps: float = 1.0   # 1 bps slippage on market orders


class WeeklyTPHSExecution(ExecutionStrategy):
    """
    H5 Smart Simple execution: TP/HS monitor Tue-Fri, Friday close.

    Entry: Monday close (limit order, 0% fee).
    Monitor: Tue-Fri check HS first, then TP each bar.
    Exit: TP (limit) / HS (limit) / Friday close (market, 1bps slippage).

    This replicates the exact logic from backtest_smart_simple_v1.py.
    """

    def __init__(self, config: Optional[WeeklyTPHSConfig] = None):
        self.config = config or WeeklyTPHSConfig()
        self._trade_counter = 0

    def execute(
        self,
        signal: UniversalSignalRecord,
        ohlcv: pd.DataFrame,
        equity: float,
    ) -> Optional[StrategyTrade]:
        """Execute a weekly signal against daily OHLCV bars."""
        if signal.skip_trade or signal.direction == 0:
            return None

        direction = signal.direction
        entry_price = signal.entry_price
        leverage = signal.leverage
        hard_stop_pct = signal.hard_stop_pct or 0.03
        take_profit_pct = signal.take_profit_pct or 0.015

        monday = pd.Timestamp(signal.signal_date)
        friday = monday + pd.offsets.BDay(4)

        # Get Tue-Fri bars
        week_dates = pd.bdate_range(monday + timedelta(days=1), friday)
        bars = []
        for day in week_dates:
            row = ohlcv[ohlcv["date"] == day]
            if not row.empty:
                bars.append({
                    "date": day,
                    "high": float(row["high"].iloc[0]),
                    "low": float(row["low"].iloc[0]),
                    "close": float(row["close"].iloc[0]),
                })

        if not bars:
            return None

        # Simulate week
        exit_price, exit_reason, exit_bar_idx = self._simulate_week(
            direction, entry_price, bars, hard_stop_pct, take_profit_pct,
        )

        # Compute PnL with costs
        pnl_pct = self._compute_pnl(
            direction, entry_price, exit_price, leverage, exit_reason,
        )

        new_equity = equity * (1 + pnl_pct)
        pnl_usd = new_equity - equity

        # Exit timestamp
        exit_date = bars[exit_bar_idx]["date"]

        self._trade_counter += 1

        # Metadata from signal
        meta = dict(signal.metadata) if signal.metadata else {}
        meta.update({
            "hard_stop_pct": round(hard_stop_pct * 100, 2),
            "take_profit_pct": round(take_profit_pct * 100, 2),
        })

        return StrategyTrade(
            trade_id=self._trade_counter,
            timestamp=str(monday),
            exit_timestamp=str(exit_date),
            side="LONG" if direction == 1 else "SHORT",
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_usd=round(pnl_usd, 2),
            pnl_pct=round(pnl_pct * 100, 4),
            exit_reason=exit_reason,
            equity_at_entry=round(equity, 2),
            equity_at_exit=round(new_equity, 2),
            leverage=round(leverage, 4),
            metadata=meta,
        )

    def _simulate_week(
        self,
        direction: int,
        entry: float,
        bars: List[dict],
        hard_stop_pct: float,
        take_profit_pct: float,
    ) -> tuple:
        """Simulate a 5-day hold with TP/HS/Friday-close."""
        for i, bar in enumerate(bars):
            h, l, c = bar["high"], bar["low"], bar["close"]

            # Check hard stop first (worst case)
            if check_hard_stop(direction, entry, h, l, hard_stop_pct):
                ep = get_exit_price(direction, entry, "hard_stop", hard_stop_pct, take_profit_pct, c)
                return ep, "hard_stop", i

            # Check take profit
            if check_take_profit(direction, entry, h, l, take_profit_pct):
                ep = get_exit_price(direction, entry, "take_profit", hard_stop_pct, take_profit_pct, c)
                return ep, "take_profit", i

        # Friday close (market order)
        return bars[-1]["close"], "week_end", len(bars) - 1

    def _compute_pnl(
        self,
        direction: int,
        entry: float,
        exit_price: float,
        leverage: float,
        exit_reason: str,
    ) -> float:
        """Compute PnL percentage with costs (matches backtest_smart_simple_v1.py)."""
        raw_pnl = direction * (exit_price - entry) / entry * leverage

        slippage = self.config.slippage_bps / 10000.0

        # Costs: entry always limit (0%), exit depends on reason
        entry_cost = self.config.maker_fee * leverage
        if exit_reason == "week_end":
            exit_cost = slippage * leverage  # Market order, slippage only
        else:
            exit_cost = self.config.maker_fee * leverage  # Limit order (TP/HS)

        return raw_pnl - entry_cost - exit_cost


# ---------------------------------------------------------------------------
# H1 Daily: Trailing Stop Intraday
# ---------------------------------------------------------------------------

@dataclass
class DailyTrailingStopConfig:
    """Configuration for daily trailing stop execution."""
    activation_pct: float = 0.002    # 0.20% move to arm the trail
    trail_pct: float = 0.003         # 0.30% drawback from peak
    hard_stop_pct: float = 0.015     # 1.50% adverse move
    slippage_bps: float = 1.0        # 1 bps slippage
    short_only: bool = True          # 2026 regime: SHORT-only


class DailyTrailingStopExecution(ExecutionStrategy):
    """
    H1 Forecast+VT+Trail execution: intraday trailing stop on 5-min bars.

    Entry: daily signal at close (limit order).
    Trail: activate at +0.2%, trail at 0.3%, hard_stop at 1.5%.
    Exit: trailing_stop / hard_stop / session_close (12:55 COT).
    Direction: SHORT-only (2026 regime change).

    Uses TrailingStopTracker from src/execution/trailing_stop.py.
    """

    def __init__(self, config: Optional[DailyTrailingStopConfig] = None):
        self.config = config or DailyTrailingStopConfig()
        self._trade_counter = 0

    def execute(
        self,
        signal: UniversalSignalRecord,
        ohlcv: pd.DataFrame,
        equity: float,
    ) -> Optional[StrategyTrade]:
        """Execute a daily signal against 5-min OHLCV bars."""
        from src.execution.trailing_stop import TrailingStopTracker, TrailingStopConfig, TrailingState

        if signal.skip_trade or signal.direction == 0:
            return None

        direction = signal.direction

        # Apply direction filter
        if self.config.short_only and direction == 1:
            return None

        entry_price = signal.entry_price
        leverage = signal.leverage

        # Use signal stop levels if provided, else use defaults
        ts_config = TrailingStopConfig(
            activation_pct=signal.trailing_activation_pct or self.config.activation_pct,
            trail_pct=signal.trailing_distance_pct or self.config.trail_pct,
            hard_stop_pct=signal.hard_stop_pct or self.config.hard_stop_pct,
        )

        signal_date = pd.Timestamp(signal.signal_date)

        # Get next trading day's 5-min bars
        next_day = signal_date + pd.offsets.BDay(1)
        day_bars = ohlcv[
            (ohlcv["date"].dt.normalize() == next_day.normalize())
        ].sort_values("date")

        if day_bars.empty:
            return None

        # Create tracker
        tracker = TrailingStopTracker(
            entry_price=entry_price,
            direction=direction,
            config=ts_config,
        )

        # Step through bars
        exit_date = None
        for idx, (_, bar) in enumerate(day_bars.iterrows()):
            state = tracker.update(
                bar_high=float(bar["high"]),
                bar_low=float(bar["low"]),
                bar_close=float(bar["close"]),
                bar_idx=idx,
            )
            if state == TrailingState.TRIGGERED:
                exit_date = bar["date"]
                break

        # If not triggered, expire at session close
        if tracker.state not in (TrailingState.TRIGGERED, TrailingState.EXPIRED):
            last_bar = day_bars.iloc[-1]
            tracker.expire(float(last_bar["close"]))
            exit_date = last_bar["date"]

        exit_price = tracker.exit_price
        exit_reason = tracker.exit_reason

        if exit_price is None:
            return None

        # Compute PnL
        slippage = self.config.slippage_bps / 10000.0
        raw_pnl = direction * (exit_price - entry_price) / entry_price * leverage
        total_cost = slippage * leverage * 2  # entry + exit slippage
        pnl_pct = raw_pnl - total_cost

        new_equity = equity * (1 + pnl_pct)
        pnl_usd = new_equity - equity

        self._trade_counter += 1

        return StrategyTrade(
            trade_id=self._trade_counter,
            timestamp=str(signal_date),
            exit_timestamp=str(exit_date) if exit_date is not None else None,
            side="LONG" if direction == 1 else "SHORT",
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_usd=round(pnl_usd, 2),
            pnl_pct=round(pnl_pct * 100, 4),
            exit_reason=exit_reason,
            equity_at_entry=round(equity, 2),
            equity_at_exit=round(new_equity, 2),
            leverage=round(leverage, 4),
            metadata=signal.metadata,
        )


# ---------------------------------------------------------------------------
# RL Intraday: SL/TP per 5-min bar
# ---------------------------------------------------------------------------

@dataclass
class IntradaySLTPConfig:
    """Configuration for intraday SL/TP execution."""
    stop_loss_pct: float = 0.025     # -2.5% SL
    take_profit_pct: float = 0.03    # +3.0% TP
    max_holding_bars: int = 576      # Max bars to hold (3 sessions)
    spread_bps: float = 2.5
    slippage_bps: float = 2.5


class IntradaySLTPExecution(ExecutionStrategy):
    """
    RL PPO execution: SL/TP per 5-min bar.

    Entry: open of next bar after signal.
    Monitor: SL/TP check per bar, max holding duration.
    Exit: stop_loss / take_profit / session_close / max_holding.

    This simulates the RL execution logic from BacktestEngine.
    """

    def __init__(self, config: Optional[IntradaySLTPConfig] = None):
        self.config = config or IntradaySLTPConfig()
        self._trade_counter = 0

    def execute(
        self,
        signal: UniversalSignalRecord,
        ohlcv: pd.DataFrame,
        equity: float,
    ) -> Optional[StrategyTrade]:
        """Execute an intraday signal against 5-min OHLCV bars."""
        if signal.skip_trade or signal.direction == 0:
            return None

        direction = signal.direction
        signal_ts = pd.Timestamp(signal.signal_date)

        # Entry at open of next bar
        future_bars = ohlcv[ohlcv["date"] > signal_ts].sort_values("date")
        if future_bars.empty:
            return None

        entry_bar = future_bars.iloc[0]
        entry_price = float(entry_bar["open"])
        entry_ts = entry_bar["date"]

        # Use signal stop levels if provided
        sl_pct = signal.hard_stop_pct or self.config.stop_loss_pct
        tp_pct = signal.take_profit_pct or self.config.take_profit_pct
        leverage = signal.leverage

        # Monitor subsequent bars
        subsequent = future_bars.iloc[1:]
        exit_price = None
        exit_reason = None
        exit_ts = None
        bars_held = 0

        for _, bar in subsequent.iterrows():
            bars_held += 1
            h = float(bar["high"])
            l = float(bar["low"])
            c = float(bar["close"])

            # Check SL
            if direction == 1 and l <= entry_price * (1 - sl_pct):
                exit_price = entry_price * (1 - sl_pct)
                exit_reason = "hard_stop"
                exit_ts = bar["date"]
                break
            elif direction == -1 and h >= entry_price * (1 + sl_pct):
                exit_price = entry_price * (1 + sl_pct)
                exit_reason = "hard_stop"
                exit_ts = bar["date"]
                break

            # Check TP
            if direction == 1 and h >= entry_price * (1 + tp_pct):
                exit_price = entry_price * (1 + tp_pct)
                exit_reason = "take_profit"
                exit_ts = bar["date"]
                break
            elif direction == -1 and l <= entry_price * (1 - tp_pct):
                exit_price = entry_price * (1 - tp_pct)
                exit_reason = "take_profit"
                exit_ts = bar["date"]
                break

            # Max holding
            if bars_held >= self.config.max_holding_bars:
                exit_price = c
                exit_reason = "session_close"
                exit_ts = bar["date"]
                break

        # If we ran out of bars, close at last bar
        if exit_price is None:
            if not subsequent.empty:
                last = subsequent.iloc[-1]
                exit_price = float(last["close"])
                exit_reason = "session_close"
                exit_ts = last["date"]
            else:
                exit_price = entry_price
                exit_reason = "no_bars"
                exit_ts = entry_ts

        # Compute PnL with costs
        cost_bps = (self.config.spread_bps + self.config.slippage_bps) / 10000.0
        raw_pnl = direction * (exit_price - entry_price) / entry_price * leverage
        total_cost = cost_bps * leverage * 2  # round-trip
        pnl_pct = raw_pnl - total_cost

        new_equity = equity * (1 + pnl_pct)
        pnl_usd = new_equity - equity

        self._trade_counter += 1

        return StrategyTrade(
            trade_id=self._trade_counter,
            timestamp=str(entry_ts),
            exit_timestamp=str(exit_ts) if exit_ts is not None else None,
            side="LONG" if direction == 1 else "SHORT",
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_usd=round(pnl_usd, 2),
            pnl_pct=round(pnl_pct * 100, 4),
            exit_reason=exit_reason,
            equity_at_entry=round(equity, 2),
            equity_at_exit=round(new_equity, 2),
            leverage=round(leverage, 4),
            metadata=signal.metadata,
        )
