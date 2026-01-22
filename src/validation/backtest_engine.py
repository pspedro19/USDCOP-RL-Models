"""
Unified Backtest Engine Module
==============================

Unified backtesting engine with realistic simulation including:
- Slippage modeling
- Transaction costs
- Position sizing
- Lookahead bias prevention
- Performance metrics calculation

Components:
- BacktestConfig: Configuration for backtest runs
- BacktestResult: Container for backtest results
- BacktestMetrics: Performance metrics
- UnifiedBacktestEngine: Main backtesting engine

Author: USD/COP Trading System
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd


class SignalType(Enum):
    """Trading signal types."""
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""

    # Capital and sizing
    initial_capital: float = 10000.0
    position_size_pct: float = 1.0  # Percentage of capital per trade

    # Costs and slippage
    slippage_pct: float = 0.001  # 0.1% slippage per trade
    commission_pct: float = 0.0005  # 0.05% commission per trade

    # Risk management
    max_drawdown_pct: float = 20.0
    max_position_hold_bars: int = 100

    # Data settings
    warmup_bars: int = 0  # Bars to skip at start for indicator warmup
    prevent_lookahead: bool = True  # Ensure no future data access

    # Logging
    verbose: bool = False
    log_trades: bool = True


@dataclass
class Trade:
    """Record of a single trade."""
    entry_bar: int
    entry_time: datetime
    entry_price: float
    direction: SignalType
    exit_bar: Optional[int] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    slippage_cost: float = 0.0
    commission_cost: float = 0.0
    bars_held: int = 0


@dataclass
class BacktestMetrics:
    """Performance metrics from a backtest."""

    # Returns
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Risk
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    volatility_pct: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_bars_held: float = 0.0

    # Costs
    total_slippage: float = 0.0
    total_commission: float = 0.0
    total_costs: float = 0.0


@dataclass
class BacktestResult:
    """Complete result from a backtest run."""

    # Metrics
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)

    # Series data
    equity_curve: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    drawdown: List[float] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)

    # Trade list
    trades: List[Trade] = field(default_factory=list)

    # Metadata
    config: Optional[BacktestConfig] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    bars_processed: int = 0

    # Validation
    lookahead_detected: bool = False
    validation_errors: List[str] = field(default_factory=list)


class UnifiedBacktestEngine:
    """Unified backtesting engine with realistic simulation.

    This engine provides:
    - Realistic slippage and commission modeling
    - Position management
    - Lookahead bias prevention
    - Comprehensive metrics calculation

    Example:
        config = BacktestConfig(
            initial_capital=10000,
            slippage_pct=0.001
        )
        engine = UnifiedBacktestEngine(config)
        result = engine.run(ohlcv_data, strategy_func)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize the backtest engine.

        Args:
            config: Backtest configuration (uses defaults if not provided)
        """
        self.config = config or BacktestConfig()

        # State
        self._position: int = 0  # -1, 0, 1
        self._entry_price: float = 0.0
        self._entry_bar: int = 0
        self._entry_time: Optional[datetime] = None
        self._equity: float = self.config.initial_capital
        self._cash: float = self.config.initial_capital
        self._current_bar: int = 0

        # Records
        self._trades: List[Trade] = []
        self._equity_curve: List[float] = []
        self._returns: List[float] = []
        self._positions: List[int] = []

    def reset(self) -> None:
        """Reset engine state for a new backtest."""
        self._position = 0
        self._entry_price = 0.0
        self._entry_bar = 0
        self._entry_time = None
        self._equity = self.config.initial_capital
        self._cash = self.config.initial_capital
        self._current_bar = 0
        self._trades = []
        self._equity_curve = []
        self._returns = []
        self._positions = []

    def run(
        self,
        data: pd.DataFrame,
        strategy: Callable[[pd.DataFrame, int], SignalType],
    ) -> BacktestResult:
        """Run a backtest with the given data and strategy.

        Args:
            data: OHLCV DataFrame with columns [open, high, low, close, volume]
            strategy: Function that takes (data, current_bar) and returns SignalType

        Returns:
            BacktestResult with metrics and trade history
        """
        self.reset()
        result = BacktestResult(config=self.config)
        result.start_time = datetime.now()

        n_bars = len(data)
        prev_equity = self.config.initial_capital

        # Main backtest loop
        for bar in range(self.config.warmup_bars, n_bars):
            self._current_bar = bar

            # Get current prices
            current_close = data['close'].iloc[bar]
            current_time = data.index[bar] if hasattr(data.index, '__getitem__') else None

            # Update mark-to-market equity
            if self._position != 0:
                price_change = current_close - self._entry_price
                if self._position == -1:  # Short
                    price_change = -price_change
                position_value = self._entry_price + price_change
                self._equity = self._cash + (position_value - self._entry_price) * self._position

            # Record state before signal
            self._equity_curve.append(self._equity)
            self._positions.append(self._position)

            # Get strategy signal - only pass data up to current bar (prevent lookahead)
            if self.config.prevent_lookahead:
                visible_data = data.iloc[:bar + 1].copy()
            else:
                visible_data = data

            signal = strategy(visible_data, bar)

            # Execute signal
            self._execute_signal(signal, current_close, bar, current_time)

            # Calculate returns
            ret = (self._equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            self._returns.append(ret)
            prev_equity = self._equity

        # Close any open position at end
        if self._position != 0:
            final_price = data['close'].iloc[-1]
            final_time = data.index[-1] if hasattr(data.index, '__getitem__') else None
            self._close_position(final_price, n_bars - 1, final_time)

        # Calculate metrics
        result.metrics = self._calculate_metrics()
        result.equity_curve = self._equity_curve.copy()
        result.returns = self._returns.copy()
        result.drawdown = self._calculate_drawdown()
        result.positions = self._positions.copy()
        result.trades = self._trades.copy()
        result.bars_processed = n_bars - self.config.warmup_bars

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    def _execute_signal(
        self,
        signal: SignalType,
        price: float,
        bar: int,
        time: Optional[datetime]
    ) -> None:
        """Execute a trading signal.

        Args:
            signal: The signal to execute
            price: Current price
            bar: Current bar index
            time: Current timestamp
        """
        if signal == SignalType.HOLD:
            return

        if signal == SignalType.CLOSE:
            if self._position != 0:
                self._close_position(price, bar, time)

        elif signal == SignalType.LONG:
            if self._position == -1:  # Close short first
                self._close_position(price, bar, time)
            if self._position == 0:
                self._open_position(1, price, bar, time)

        elif signal == SignalType.SHORT:
            if self._position == 1:  # Close long first
                self._close_position(price, bar, time)
            if self._position == 0:
                self._open_position(-1, price, bar, time)

    def _open_position(
        self,
        direction: int,
        price: float,
        bar: int,
        time: Optional[datetime]
    ) -> None:
        """Open a new position.

        Args:
            direction: 1 for long, -1 for short
            price: Entry price
            bar: Entry bar
            time: Entry timestamp
        """
        # Apply slippage
        slippage = price * self.config.slippage_pct
        if direction == 1:  # Long - worse entry
            adjusted_price = price + slippage
        else:  # Short - worse entry
            adjusted_price = price - slippage

        # Apply commission
        commission = adjusted_price * self.config.commission_pct

        self._position = direction
        self._entry_price = adjusted_price
        self._entry_bar = bar
        self._entry_time = time
        self._cash -= commission

    def _close_position(
        self,
        price: float,
        bar: int,
        time: Optional[datetime]
    ) -> None:
        """Close current position.

        Args:
            price: Exit price
            bar: Exit bar
            time: Exit timestamp
        """
        if self._position == 0:
            return

        # Apply slippage
        slippage = price * self.config.slippage_pct
        if self._position == 1:  # Closing long - worse exit
            adjusted_price = price - slippage
        else:  # Closing short - worse exit
            adjusted_price = price + slippage

        # Apply commission
        commission = adjusted_price * self.config.commission_pct

        # Calculate PnL
        if self._position == 1:  # Long
            pnl = adjusted_price - self._entry_price
        else:  # Short
            pnl = self._entry_price - adjusted_price

        pnl_pct = pnl / self._entry_price if self._entry_price > 0 else 0.0

        # Record trade
        trade = Trade(
            entry_bar=self._entry_bar,
            entry_time=self._entry_time,
            entry_price=self._entry_price,
            direction=SignalType.LONG if self._position == 1 else SignalType.SHORT,
            exit_bar=bar,
            exit_time=time,
            exit_price=adjusted_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            slippage_cost=slippage * 2,  # Entry + exit
            commission_cost=commission * 2,  # Entry + exit
            bars_held=bar - self._entry_bar,
        )
        self._trades.append(trade)

        # Update cash
        self._cash += pnl - commission
        self._equity = self._cash

        # Reset position
        self._position = 0
        self._entry_price = 0.0
        self._entry_bar = 0
        self._entry_time = None

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate performance metrics from backtest results."""
        metrics = BacktestMetrics()

        if not self._equity_curve:
            return metrics

        equity = np.array(self._equity_curve)
        returns = np.array(self._returns) if self._returns else np.array([0.0])

        # Basic returns
        metrics.total_return_pct = (
            (equity[-1] - self.config.initial_capital)
            / self.config.initial_capital * 100
        )

        # Annualized return (assuming 5-min bars, ~252 trading days, 78 bars/day)
        bars_per_year = 252 * 78
        n_bars = len(equity)
        if n_bars > 1:
            total_return = equity[-1] / self.config.initial_capital
            metrics.annualized_return_pct = (
                (total_return ** (bars_per_year / n_bars) - 1) * 100
            )

        # Volatility
        if len(returns) > 1:
            metrics.volatility_pct = np.std(returns) * np.sqrt(bars_per_year) * 100

        # Sharpe ratio (assuming 0% risk-free rate)
        if metrics.volatility_pct > 0:
            metrics.sharpe_ratio = metrics.annualized_return_pct / metrics.volatility_pct

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = np.std(downside_returns) * np.sqrt(bars_per_year) * 100
            if downside_vol > 0:
                metrics.sortino_ratio = metrics.annualized_return_pct / downside_vol

        # Drawdown
        drawdown = self._calculate_drawdown()
        if drawdown:
            metrics.max_drawdown_pct = max(drawdown) * 100
            metrics.avg_drawdown_pct = np.mean(drawdown) * 100

        # Calmar ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return_pct / metrics.max_drawdown_pct

        # Trade statistics
        metrics.total_trades = len(self._trades)
        if metrics.total_trades > 0:
            wins = [t for t in self._trades if t.pnl > 0]
            losses = [t for t in self._trades if t.pnl <= 0]

            metrics.winning_trades = len(wins)
            metrics.losing_trades = len(losses)
            metrics.win_rate = len(wins) / metrics.total_trades * 100

            if wins:
                metrics.avg_win_pct = np.mean([t.pnl_pct for t in wins]) * 100
            if losses:
                metrics.avg_loss_pct = np.mean([t.pnl_pct for t in losses]) * 100

            # Profit factor
            total_wins = sum(t.pnl for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl for t in losses)) if losses else 0
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses

            # Average hold time
            metrics.avg_bars_held = np.mean([t.bars_held for t in self._trades])

            # Costs
            metrics.total_slippage = sum(t.slippage_cost for t in self._trades)
            metrics.total_commission = sum(t.commission_cost for t in self._trades)
            metrics.total_costs = metrics.total_slippage + metrics.total_commission

        return metrics

    def _calculate_drawdown(self) -> List[float]:
        """Calculate drawdown series."""
        if not self._equity_curve:
            return []

        equity = np.array(self._equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return drawdown.tolist()


# Factory function for creating configured engine
def create_backtest_engine(
    initial_capital: float = 10000.0,
    slippage_pct: float = 0.001,
    commission_pct: float = 0.0005,
    prevent_lookahead: bool = True,
) -> UnifiedBacktestEngine:
    """Create a configured backtest engine.

    Args:
        initial_capital: Starting capital
        slippage_pct: Slippage percentage per trade
        commission_pct: Commission percentage per trade
        prevent_lookahead: Whether to prevent lookahead bias

    Returns:
        Configured UnifiedBacktestEngine
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        slippage_pct=slippage_pct,
        commission_pct=commission_pct,
        prevent_lookahead=prevent_lookahead,
    )
    return UnifiedBacktestEngine(config)


__all__ = [
    'SignalType',
    'BacktestConfig',
    'Trade',
    'BacktestMetrics',
    'BacktestResult',
    'UnifiedBacktestEngine',
    'create_backtest_engine',
]
