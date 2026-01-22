"""
Trade Simulator
Simulates trades based on model signals and calculates P&L
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..config import get_settings
from .observation_builder import ObservationBuilder
from .inference_engine import InferenceEngine

settings = get_settings()


@dataclass
class Trade:
    """Represents a single trade"""
    trade_id: int
    model_id: str
    side: str  # "LONG" or "SHORT"
    entry_time: datetime
    entry_price: float
    entry_bar: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_bar: Optional[int] = None
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: Optional[str] = None
    equity_at_entry: float = 0.0
    equity_at_exit: Optional[float] = None
    entry_confidence: float = 0.0
    exit_confidence: Optional[float] = None
    duration_bars: int = 0


@dataclass
class SimulationState:
    """Tracks simulation state"""
    equity: float = 10000.0
    peak_equity: float = 10000.0
    position: str = "FLAT"  # "FLAT", "LONG", "SHORT"
    position_size: float = 0.0  # Fraction of equity in trade (0.5-1.0)
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_bar: int = 0
    entry_confidence: float = 0.0
    trade_count: int = 0
    stop_loss_price: float = 0.0  # Stop-loss trigger price
    take_profit_price: float = 0.0  # Take-profit trigger price
    last_trade_bar: int = 0  # Bar of last trade (for cooldown)


class TradeSimulator:
    """
    Simulates paper trading based on model signals.
    Based on logic from paper_trading.py
    """

    def __init__(
        self,
        initial_capital: float = None,
        transaction_cost_bps: float = None,
        slippage_bps: float = None,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
        min_position_size: float = None,
        max_position_size: float = None,
        max_position_duration_bars: int = None,
        min_bars_between_trades: int = None
    ):
        self.initial_capital = initial_capital or settings.initial_capital
        self.transaction_cost = (transaction_cost_bps or settings.transaction_cost_bps) / 10000
        self.slippage = (slippage_bps or settings.slippage_bps) / 10000
        # Risk management parameters
        self.stop_loss_pct = (stop_loss_pct or settings.stop_loss_pct) / 100
        self.take_profit_pct = (take_profit_pct or settings.take_profit_pct) / 100
        # Position sizing parameters
        self.min_position_size = min_position_size or settings.min_position_size
        self.max_position_size = max_position_size or settings.max_position_size
        # Position bias fix: force exits after max duration
        self.max_position_duration_bars = max_position_duration_bars or settings.max_position_duration_bars
        self.min_bars_between_trades = min_bars_between_trades or settings.min_bars_between_trades
        # Dynamic thresholds (easier to exit than enter)
        # These are defaults, will be overridden by model-specific thresholds
        self.threshold_long_entry = settings.threshold_long_entry
        self.threshold_short_entry = settings.threshold_short_entry
        self.threshold_exit = settings.threshold_exit

        # Model-specific thresholds (must match training thresholds!)
        # Uses 0.33/-0.33 (wider HOLD zone) - matches training
        self.MODEL_THRESHOLDS = {
            "ppo_primary": {"long_entry": 0.33, "short_entry": -0.33, "exit": 0.15},
            "ppo_secondary": {"long_entry": 0.33, "short_entry": -0.33, "exit": 0.15},
        }

    def _set_model_thresholds(self, model_id: str):
        """Set thresholds based on model_id"""
        if model_id in self.MODEL_THRESHOLDS:
            thresholds = self.MODEL_THRESHOLDS[model_id]
            self.threshold_long_entry = thresholds["long_entry"]
            self.threshold_short_entry = thresholds["short_entry"]
            self.threshold_exit = thresholds["exit"]

    def run_simulation(
        self,
        df: pd.DataFrame,
        inference_engine: InferenceEngine,
        observation_builder: ObservationBuilder,
        model_id: str = "ppo_primary",
        progress_callback: Optional[callable] = None,
        trade_callback: Optional[callable] = None
    ) -> List[Trade]:
        """
        Run full simulation on historical data.

        Args:
            df: DataFrame with OHLCV and macro data
            inference_engine: Loaded inference engine
            observation_builder: Observation builder instance
            model_id: Model ID to use
            progress_callback: Optional callback(progress, bar_idx, total_bars)
            trade_callback: Optional callback(trade, equity) called when trade closes

        Returns:
            List of executed trades
        """
        # Set model-specific thresholds
        self._set_model_thresholds(model_id)

        trades: List[Trade] = []
        state = SimulationState(equity=self.initial_capital, peak_equity=self.initial_capital)

        total_bars = len(df)

        # Skip first N bars to build up technical indicators
        start_bar = min(50, total_bars // 10)

        for bar_idx in range(start_bar, total_bars):
            row = df.iloc[bar_idx]

            # Calculate time_normalized (0-1 based on time of day) - SSOT feature name
            time_normalized = self._calculate_time_normalized(row["time"])

            # FIX: Always use position=0 to avoid position bias
            # The model learned to "stay in position" when it knows its current position
            # By always passing 0 (FLAT), the model makes unbiased decisions
            position_numeric = 0.0  # Always FLAT to avoid position bias

            # Build observation
            obs = observation_builder.build_observation(
                df=df,
                bar_idx=bar_idx,
                position=position_numeric,
                time_normalized=time_normalized  # SSOT feature name (index 14)
            )

            # Get model prediction
            signal, action, confidence = inference_engine.predict_signal(obs, model_id)

            # Execute trading logic
            current_price = float(row["close"])
            timestamp = row["time"]

            # === POSITION BIAS FIX: Check exits in priority order ===

            # 1. Check MAX DURATION first (force exit after N bars)
            duration_trade = self._check_max_duration(
                state=state,
                bar_idx=bar_idx,
                current_price=current_price,
                timestamp=timestamp,
                model_id=model_id
            )
            if duration_trade:
                trades.append(duration_trade)
                if trade_callback:
                    trade_callback(duration_trade, state.equity)
                state.last_trade_bar = bar_idx
                continue  # Skip signal processing this bar

            # 2. Check stop-loss / take-profit
            sl_tp_trade = self._check_stop_loss_take_profit(
                state=state,
                current_price=current_price,
                timestamp=timestamp,
                bar_idx=bar_idx,
                model_id=model_id
            )
            if sl_tp_trade:
                trades.append(sl_tp_trade)
                if trade_callback:
                    trade_callback(sl_tp_trade, state.equity)
                state.last_trade_bar = bar_idx

            # 3. Apply dynamic thresholds (easier to exit than enter)
            dynamic_signal = self._get_dynamic_signal(action, state.position)

            # 4. Check cooldown before allowing new trades
            bars_since_last_trade = bar_idx - state.last_trade_bar
            if bars_since_last_trade < self.min_bars_between_trades and state.position == "FLAT":
                dynamic_signal = "HOLD"  # Still in cooldown, don't enter

            trade = self._process_signal(
                state=state,
                signal=dynamic_signal,  # Use dynamic signal instead of raw signal
                confidence=confidence,
                price=current_price,
                timestamp=timestamp,
                bar_idx=bar_idx,
                model_id=model_id
            )

            if trade:
                trades.append(trade)
                if trade_callback:
                    trade_callback(trade, state.equity)
                state.last_trade_bar = bar_idx

            # Update peak equity
            state.peak_equity = max(state.peak_equity, state.equity)

            # Progress callback
            if progress_callback and bar_idx % 100 == 0:
                progress = (bar_idx - start_bar) / (total_bars - start_bar)
                progress_callback(progress, bar_idx, total_bars)

        # Close any open position at end
        if state.position != "FLAT":
            final_price = float(df.iloc[-1]["close"])
            final_time = df.iloc[-1]["time"]
            trade = self._close_position(
                state=state,
                price=final_price,
                timestamp=final_time,
                bar_idx=len(df) - 1,
                reason="END_OF_SIMULATION",
                model_id=model_id
            )
            if trade:
                trades.append(trade)
                if trade_callback:
                    trade_callback(trade, state.equity)

        return trades

    def _calculate_time_normalized(self, timestamp: datetime) -> float:
        """Calculate progress through trading session (0-1) - SSOT feature name: time_normalized"""
        if pd.isna(timestamp):
            return 0.5

        # Trading hours: 8:00 - 13:00 COT (13:00 - 18:00 UTC)
        hour = timestamp.hour
        minute = timestamp.minute

        # Normalize to 0-1 based on trading session
        session_start = 13  # 13:00 UTC = 8:00 COT
        session_end = 18    # 18:00 UTC = 13:00 COT
        session_length = session_end - session_start

        if hour < session_start:
            return 0.0
        elif hour >= session_end:
            return 1.0
        else:
            minutes_into_session = (hour - session_start) * 60 + minute
            total_session_minutes = session_length * 60
            return minutes_into_session / total_session_minutes

    def _position_to_numeric(self, position: str) -> float:
        """Convert position string to numeric value"""
        if position == "LONG":
            return 1.0
        elif position == "SHORT":
            return -1.0
        return 0.0

    def _check_max_duration(
        self,
        state: SimulationState,
        bar_idx: int,
        current_price: float,
        timestamp: datetime,
        model_id: str
    ) -> Optional[Trade]:
        """
        Force close position if held too long (Position Bias Fix).
        This prevents the model from holding a single position for months/years.
        """
        if state.position == "FLAT":
            return None

        bars_in_position = bar_idx - state.entry_bar
        if bars_in_position >= self.max_position_duration_bars:
            return self._close_position(
                state=state,
                price=current_price,
                timestamp=timestamp,
                bar_idx=bar_idx,
                reason="MAX_DURATION",
                model_id=model_id
            )
        return None

    def _get_dynamic_signal(
        self,
        action: float,
        current_position: str
    ) -> str:
        """
        Dynamic thresholds based on current position.
        Easier to exit than enter = encourages position changes.

        This helps combat position bias by making it easier to close
        existing positions than to open new ones.
        """
        if current_position == "LONG":
            # Already long: use lower threshold to exit
            if action < -self.threshold_exit:  # Small negative = exit to SHORT
                return "SHORT"
            elif action < self.threshold_exit:  # Near zero = exit to FLAT
                return "FLAT"
            else:
                return "HOLD"  # Still bullish, stay long
        elif current_position == "SHORT":
            # Already short: use lower threshold to exit
            if action > self.threshold_exit:  # Small positive = exit to LONG
                return "LONG"
            elif action > -self.threshold_exit:  # Near zero = exit to FLAT
                return "FLAT"
            else:
                return "HOLD"  # Still bearish, stay short
        else:  # FLAT
            # No position: use strict thresholds to enter
            if action > self.threshold_long_entry:
                return "LONG"
            elif action < self.threshold_short_entry:
                return "SHORT"
            else:
                return "HOLD"

    def _check_stop_loss_take_profit(
        self,
        state: SimulationState,
        current_price: float,
        timestamp: datetime,
        bar_idx: int,
        model_id: str
    ) -> Optional[Trade]:
        """Check if stop-loss or take-profit is triggered"""
        if state.position == "FLAT":
            return None

        triggered = False
        reason = ""

        if state.position == "LONG":
            # Stop-loss: price dropped below stop
            if current_price <= state.stop_loss_price:
                triggered = True
                reason = "STOP_LOSS"
            # Take-profit: price rose above target
            elif current_price >= state.take_profit_price:
                triggered = True
                reason = "TAKE_PROFIT"
        else:  # SHORT
            # Stop-loss: price rose above stop
            if current_price >= state.stop_loss_price:
                triggered = True
                reason = "STOP_LOSS"
            # Take-profit: price dropped below target
            elif current_price <= state.take_profit_price:
                triggered = True
                reason = "TAKE_PROFIT"

        if triggered:
            return self._close_position(
                state=state,
                price=current_price,
                timestamp=timestamp,
                bar_idx=bar_idx,
                reason=reason,
                model_id=model_id
            )
        return None

    def _calculate_position_size(self, confidence: float) -> float:
        """
        Scale position size based on model confidence.
        Higher confidence = larger position (up to max_position_size)
        """
        # Confidence typically ranges from 0.15 to 1.0
        # Normalize to 0-1 range, then scale between min and max position size
        normalized_conf = min(1.0, max(0.0, abs(confidence)))
        position_range = self.max_position_size - self.min_position_size
        return self.min_position_size + (normalized_conf * position_range)

    def _process_signal(
        self,
        state: SimulationState,
        signal: str,
        confidence: float,
        price: float,
        timestamp: datetime,
        bar_idx: int,
        model_id: str
    ) -> Optional[Trade]:
        """
        Process a trading signal and update state.

        Returns a Trade if a position was closed.
        """
        trade = None

        # Determine target position
        if signal == "LONG":
            target_position = "LONG"
        elif signal == "SHORT":
            target_position = "SHORT"
        else:
            target_position = state.position  # HOLD keeps current position

        # Check if we need to change position
        if target_position != state.position:
            # Close existing position first
            if state.position != "FLAT":
                trade = self._close_position(
                    state=state,
                    price=price,
                    timestamp=timestamp,
                    bar_idx=bar_idx,
                    reason="SIGNAL_CHANGE",
                    model_id=model_id,
                    exit_confidence=confidence
                )

            # Open new position
            if target_position != "FLAT":
                self._open_position(
                    state=state,
                    side=target_position,
                    price=price,
                    timestamp=timestamp,
                    bar_idx=bar_idx,
                    confidence=confidence
                )

        return trade

    def _open_position(
        self,
        state: SimulationState,
        side: str,
        price: float,
        timestamp: datetime,
        bar_idx: int,
        confidence: float
    ):
        """Open a new position with stop-loss, take-profit, and position sizing"""
        # Apply slippage (buy higher, sell lower)
        if side == "LONG":
            exec_price = price * (1 + self.slippage)
            # Stop-loss below entry, take-profit above
            state.stop_loss_price = exec_price * (1 - self.stop_loss_pct)
            state.take_profit_price = exec_price * (1 + self.take_profit_pct)
        else:
            exec_price = price * (1 - self.slippage)
            # Stop-loss above entry, take-profit below
            state.stop_loss_price = exec_price * (1 + self.stop_loss_pct)
            state.take_profit_price = exec_price * (1 - self.take_profit_pct)

        # Calculate position size based on confidence
        state.position_size = self._calculate_position_size(confidence)

        state.position = side
        state.entry_price = exec_price
        state.entry_time = timestamp
        state.entry_bar = bar_idx
        state.entry_confidence = confidence
        state.trade_count += 1

    def _close_position(
        self,
        state: SimulationState,
        price: float,
        timestamp: datetime,
        bar_idx: int,
        reason: str,
        model_id: str,
        exit_confidence: float = 0.0
    ) -> Trade:
        """Close current position and return Trade"""
        # Apply slippage (sell lower, cover higher)
        if state.position == "LONG":
            exec_price = price * (1 - self.slippage)
        else:
            exec_price = price * (1 + self.slippage)

        # Calculate P&L
        if state.position == "LONG":
            pnl_pct = (exec_price - state.entry_price) / state.entry_price
        else:  # SHORT
            pnl_pct = (state.entry_price - exec_price) / state.entry_price

        # Apply transaction costs
        pnl_pct -= self.transaction_cost * 2  # Entry + exit

        # Scale by position size (confidence-based sizing)
        position_size = state.position_size if state.position_size > 0 else 1.0
        pnl_usd = pnl_pct * state.equity * position_size

        # Update equity
        old_equity = state.equity
        state.equity += pnl_usd

        # Create trade record
        trade = Trade(
            trade_id=state.trade_count,
            model_id=model_id,
            side=state.position.lower(),
            entry_time=state.entry_time,
            entry_price=state.entry_price,
            entry_bar=state.entry_bar,
            exit_time=timestamp,
            exit_price=exec_price,
            exit_bar=bar_idx,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct * 100,  # As percentage
            exit_reason=reason,
            equity_at_entry=old_equity - pnl_usd,  # Retroactive
            equity_at_exit=state.equity,
            entry_confidence=state.entry_confidence,
            exit_confidence=exit_confidence,
            duration_bars=bar_idx - state.entry_bar
        )

        # Reset position
        state.position = "FLAT"
        state.entry_price = 0.0
        state.entry_time = None
        state.entry_bar = 0
        state.entry_confidence = 0.0
        state.position_size = 0.0
        state.stop_loss_price = 0.0
        state.take_profit_price = 0.0

        return trade

    def calculate_summary(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate summary statistics for trades"""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_trade_duration_minutes": 0.0,
            }

        pnls = [t.pnl_usd for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        total_return = (total_pnl / self.initial_capital) * 100

        # Calculate max drawdown from equity curve
        equities = [self.initial_capital]
        for trade in trades:
            equities.append(trade.equity_at_exit or equities[-1])

        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        # Average duration
        durations = [t.duration_bars * 5 for t in trades]  # 5-min bars
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": (len(winning) / len(trades) * 100) if trades else 0.0,
            "total_pnl": total_pnl,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_dd * 100,
            "avg_trade_duration_minutes": avg_duration,
        }
