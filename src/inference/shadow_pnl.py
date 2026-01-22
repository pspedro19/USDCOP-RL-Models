"""
Shadow PnL Tracker
==================

Virtual PnL tracking for shadow mode model validation.

This module provides paper trading simulation for challenger models,
allowing performance comparison without real capital risk.

Key Components:
- VirtualTrade: Record of a simulated trade
- ShadowMetrics: Performance metrics for shadow trading
- ShadowPnLTracker: Main tracking class

Design Principles:
- No side effects: Pure simulation, no real orders
- Full observability: Track all signals and positions
- Agreement tracking: Compare shadow vs champion signals

Use Case:
When promoting a new model from challenger to champion, we run it
in "shadow mode" first. This tracker simulates what the PnL would
have been if we had followed the challenger's signals.

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VirtualTrade:
    """
    Record of a virtual (simulated) trade.

    Attributes:
        signal: Original signal that triggered the trade (-1, 0, +1)
        entry_price: Price at position entry
        entry_time: Timestamp of entry
        exit_price: Price at position exit (None if still open)
        exit_time: Timestamp of exit (None if still open)
        pnl: Realized PnL (None if still open)
    """
    signal: int  # -1 (short), 0 (flat), +1 (long)
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None

    @property
    def is_closed(self) -> bool:
        """Check if trade has been closed."""
        return self.exit_price is not None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get trade duration in seconds."""
        if self.exit_time is None or self.entry_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds()

    def close(self, exit_price: float, exit_time: datetime) -> float:
        """
        Close the trade and compute PnL.

        Args:
            exit_price: Price at exit
            exit_time: Timestamp of exit

        Returns:
            Realized PnL
        """
        self.exit_price = exit_price
        self.exit_time = exit_time

        # PnL calculation: direction * (exit - entry)
        # Long (+1): profit if exit > entry
        # Short (-1): profit if exit < entry
        self.pnl = self.signal * (exit_price - self.entry_price)

        return self.pnl


@dataclass
class ShadowMetrics:
    """
    Performance metrics for shadow trading.

    Attributes:
        virtual_pnl: Total virtual PnL (sum of closed trades)
        virtual_sharpe: Sharpe ratio of virtual returns (annualized)
        trade_count: Number of completed trades
        win_rate: Percentage of profitable trades
        agreement_rate: Percentage of signals matching champion
    """
    virtual_pnl: float
    virtual_sharpe: float
    trade_count: int
    win_rate: float
    agreement_rate: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "virtual_pnl": self.virtual_pnl,
            "virtual_sharpe": self.virtual_sharpe,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "agreement_rate": self.agreement_rate,
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"ShadowMetrics("
            f"pnl={self.virtual_pnl:.4f}, "
            f"sharpe={self.virtual_sharpe:.2f}, "
            f"trades={self.trade_count}, "
            f"win_rate={self.win_rate:.1%}, "
            f"agreement={self.agreement_rate:.1%})"
        )


# =============================================================================
# Shadow PnL Tracker
# =============================================================================

class ShadowPnLTracker:
    """
    Virtual PnL tracker for shadow mode execution.

    Tracks the performance of a shadow (challenger) model by simulating
    trades based on its signals, without executing real orders.

    Key Features:
    - Virtual position tracking
    - PnL calculation on position changes
    - Signal agreement tracking with champion
    - Performance metrics computation

    Usage:
        tracker = ShadowPnLTracker(model_id="challenger_v2")

        # On each prediction
        tracker.on_prediction(
            shadow_signal=1,      # Challenger says LONG
            champion_signal=0,    # Champion says HOLD
            current_price=4250.50,
            timestamp=datetime.now(),
        )

        # Get performance metrics
        metrics = tracker.get_metrics()
        print(f"Virtual PnL: {metrics.virtual_pnl}")
        print(f"Agreement rate: {metrics.agreement_rate:.1%}")

    Attributes:
        model_id: Identifier for the shadow model
        virtual_position: Current virtual position (-1, 0, +1)
        trades: List of completed virtual trades
        champion_signals: History of champion signals
        shadow_signals: History of shadow signals
    """

    def __init__(self, model_id: str):
        """
        Initialize shadow PnL tracker.

        Args:
            model_id: Identifier for the shadow model being tracked
        """
        self.model_id = model_id

        # Position tracking
        self.virtual_position: int = 0  # -1 (short), 0 (flat), +1 (long)
        self._current_trade: Optional[VirtualTrade] = None

        # Trade history
        self.trades: List[VirtualTrade] = []

        # Signal history for agreement calculation
        self.champion_signals: List[int] = []
        self.shadow_signals: List[int] = []

        logger.info(f"ShadowPnLTracker initialized for model: {model_id}")

    def on_prediction(
        self,
        shadow_signal: int,
        champion_signal: int,
        current_price: float,
        timestamp: datetime,
    ) -> None:
        """
        Process a new prediction from both shadow and champion models.

        This method:
        1. Records both signals for agreement tracking
        2. Handles position changes based on shadow signal
        3. Opens/closes virtual trades as needed

        Args:
            shadow_signal: Signal from shadow model (-1, 0, +1)
            champion_signal: Signal from champion model (-1, 0, +1)
            current_price: Current market price
            timestamp: Timestamp of the prediction
        """
        # Record signals for agreement tracking
        self.champion_signals.append(champion_signal)
        self.shadow_signals.append(shadow_signal)

        # Check if position needs to change
        if shadow_signal != self.virtual_position:
            # Close existing position if any
            if self.virtual_position != 0:
                self._close_position(current_price, timestamp)

            # Open new position if signal is not flat
            if shadow_signal != 0:
                self._open_position(shadow_signal, current_price, timestamp)

            # Update position
            self.virtual_position = shadow_signal

            logger.debug(
                f"[{self.model_id}] Position changed to {shadow_signal} "
                f"at price {current_price:.4f}"
            )

    def _open_position(
        self,
        signal: int,
        price: float,
        timestamp: datetime,
    ) -> None:
        """
        Open a new virtual position.

        Args:
            signal: Direction (-1 for short, +1 for long)
            price: Entry price
            timestamp: Entry timestamp
        """
        self._current_trade = VirtualTrade(
            signal=signal,
            entry_price=price,
            entry_time=timestamp,
        )

        logger.debug(
            f"[{self.model_id}] Opened {'LONG' if signal > 0 else 'SHORT'} "
            f"at {price:.4f}"
        )

    def _close_position(
        self,
        price: float,
        timestamp: datetime,
    ) -> Optional[float]:
        """
        Close the current virtual position.

        Args:
            price: Exit price
            timestamp: Exit timestamp

        Returns:
            Realized PnL, or None if no position was open
        """
        if self._current_trade is None:
            logger.warning(f"[{self.model_id}] No position to close")
            return None

        # Close the trade
        pnl = self._current_trade.close(price, timestamp)

        # Add to completed trades
        self.trades.append(self._current_trade)

        logger.debug(
            f"[{self.model_id}] Closed position at {price:.4f}, "
            f"PnL: {pnl:.4f}"
        )

        # Clear current trade
        self._current_trade = None

        return pnl

    def get_metrics(self) -> ShadowMetrics:
        """
        Calculate shadow trading performance metrics.

        Returns:
            ShadowMetrics with:
            - virtual_pnl: Total PnL from closed trades
            - virtual_sharpe: Annualized Sharpe ratio
            - trade_count: Number of completed trades
            - win_rate: Percentage of profitable trades
            - agreement_rate: Signal agreement with champion
        """
        # Virtual PnL
        closed_pnls = [t.pnl for t in self.trades if t.pnl is not None]
        virtual_pnl = sum(closed_pnls) if closed_pnls else 0.0

        # Trade count
        trade_count = len(closed_pnls)

        # Win rate
        if trade_count > 0:
            winning_trades = sum(1 for pnl in closed_pnls if pnl > 0)
            win_rate = winning_trades / trade_count
        else:
            win_rate = 0.0

        # Sharpe ratio (annualized, assuming 5-min bars = 288 per day)
        if len(closed_pnls) >= 2:
            pnl_array = np.array(closed_pnls)
            mean_return = np.mean(pnl_array)
            std_return = np.std(pnl_array, ddof=1)

            if std_return > 0:
                # Annualize: sqrt(252 trading days * 288 bars/day)
                annualization_factor = np.sqrt(252 * 288)
                virtual_sharpe = (mean_return / std_return) * annualization_factor
            else:
                virtual_sharpe = 0.0
        else:
            virtual_sharpe = 0.0

        # Agreement rate
        if len(self.shadow_signals) > 0:
            agreements = sum(
                1 for s, c in zip(self.shadow_signals, self.champion_signals)
                if s == c
            )
            agreement_rate = agreements / len(self.shadow_signals)
        else:
            agreement_rate = 0.0

        metrics = ShadowMetrics(
            virtual_pnl=virtual_pnl,
            virtual_sharpe=virtual_sharpe,
            trade_count=trade_count,
            win_rate=win_rate,
            agreement_rate=agreement_rate,
        )

        logger.debug(f"[{self.model_id}] Computed metrics: {metrics}")

        return metrics

    def reset(self) -> None:
        """
        Reset tracker state.

        Clears all position and trade history.
        """
        self.virtual_position = 0
        self._current_trade = None
        self.trades = []
        self.champion_signals = []
        self.shadow_signals = []

        logger.info(f"[{self.model_id}] Tracker reset")

    @property
    def has_open_position(self) -> bool:
        """Check if there's an open virtual position."""
        return self.virtual_position != 0

    @property
    def unrealized_pnl(self) -> Optional[float]:
        """
        Get unrealized PnL for current open position.

        Note: Requires current price to be passed via mark_to_market().
        Returns None if no position or not marked.
        """
        if self._current_trade is None:
            return None
        # Cannot compute without current price
        return None

    def mark_to_market(self, current_price: float) -> float:
        """
        Calculate unrealized PnL at current price.

        Args:
            current_price: Current market price

        Returns:
            Unrealized PnL (0 if no position)
        """
        if self._current_trade is None:
            return 0.0

        return self._current_trade.signal * (
            current_price - self._current_trade.entry_price
        )

    def get_total_pnl(self, current_price: Optional[float] = None) -> float:
        """
        Get total PnL (realized + unrealized).

        Args:
            current_price: Current price for marking open position

        Returns:
            Total PnL
        """
        realized = sum(t.pnl for t in self.trades if t.pnl is not None)
        unrealized = self.mark_to_market(current_price) if current_price else 0.0

        return realized + unrealized
