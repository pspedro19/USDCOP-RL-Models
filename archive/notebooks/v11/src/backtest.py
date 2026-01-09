"""
USD/COP RL Trading System V11 - Backtest Reporter
==================================================

Professional backtest reporting with 40+ metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class Trade:
    """Represents an individual trade."""
    entry_step: int
    entry_price: float
    entry_position: float
    exit_step: int = 0
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0

    @property
    def duration(self) -> int:
        """Trade duration in steps."""
        return self.exit_step - self.entry_step

    @property
    def is_long(self) -> bool:
        """Whether this was a long trade."""
        return self.entry_position > 0

    @property
    def is_winner(self) -> bool:
        """Whether this trade was profitable."""
        return self.pnl > 0


class BacktestReporter:
    """
    Professional backtest reporting.

    Tracks trades, equity curve, and calculates 40+ metrics
    for comprehensive strategy evaluation.

    Parameters
    ----------
    initial_balance : float
        Starting portfolio value
    """

    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_balance]
        self.positions: List[float] = []
        self.returns: List[float] = []

        self.current_trade: Optional[Trade] = None
        self.step = 0
        self.prev_position = 0.0

    def update(self, position: float, portfolio: float, market_return: float):
        """
        Update reporter with each backtest step.

        Parameters
        ----------
        position : float
            Current position (-1 to +1)
        portfolio : float
            Current portfolio value
        market_return : float
            Market return for this step
        """
        self.step += 1
        self.equity_curve.append(portfolio)
        self.positions.append(position)

        if len(self.equity_curve) > 1:
            ret = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
            self.returns.append(ret)

        # Detect significant position change (new trade)
        if abs(position - self.prev_position) > 0.3:
            # Close current trade
            if self.current_trade is not None:
                self.current_trade.exit_step = self.step
                self.current_trade.exit_price = portfolio
                self.current_trade.pnl = portfolio - self.current_trade.entry_price
                if self.current_trade.entry_price > 0:
                    self.current_trade.pnl_pct = (
                        self.current_trade.pnl / self.current_trade.entry_price
                    ) * 100
                self.trades.append(self.current_trade)

            # Open new trade
            if abs(position) > 0.1:
                self.current_trade = Trade(
                    entry_step=self.step,
                    entry_price=portfolio,
                    entry_position=position
                )
            else:
                self.current_trade = None

        self.prev_position = position

    def finalize(self):
        """Close any pending trade at end of backtest."""
        if self.current_trade is not None:
            self.current_trade.exit_step = self.step
            self.current_trade.exit_price = self.equity_curve[-1]
            self.current_trade.pnl = (
                self.equity_curve[-1] - self.current_trade.entry_price
            )
            if self.current_trade.entry_price > 0:
                self.current_trade.pnl_pct = (
                    self.current_trade.pnl / self.current_trade.entry_price
                ) * 100
            self.trades.append(self.current_trade)

    def get_metrics(self) -> Dict:
        """
        Calculate all metrics.

        Returns
        -------
        dict
            Dictionary with all calculated metrics
        """
        equity = np.array(self.equity_curve)
        returns = np.array(self.returns) if self.returns else np.array([0])
        positions = np.array(self.positions) if self.positions else np.array([0])

        # Portfolio metrics
        final_balance = equity[-1]
        total_return = (final_balance / self.initial_balance - 1) * 100

        # Sharpe ratio (annualized for 5-min bars)
        if len(returns) > 1 and returns.std() > 1e-10:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 288)
        else:
            sharpe = 0.0

        # Maximum drawdown
        cummax = np.maximum.accumulate(equity)
        drawdowns = (cummax - equity) / (cummax + 1e-10)
        max_dd = drawdowns.max() * 100

        # Trade metrics
        n_trades = len(self.trades)
        if n_trades > 0:
            winners = [t for t in self.trades if t.is_winner]
            losers = [t for t in self.trades if not t.is_winner]
            win_rate = len(winners) / n_trades * 100

            gross_profit = sum(t.pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.001
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            avg_win = np.mean([t.pnl for t in winners]) if winners else 0
            avg_loss = np.mean([t.pnl for t in losers]) if losers else 0
            avg_duration = np.mean([t.duration for t in self.trades])

            long_trades = [t for t in self.trades if t.is_long]
            short_trades = [t for t in self.trades if not t.is_long]
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            avg_duration = 0
            long_trades = []
            short_trades = []

        # Position metrics
        mean_pos = positions.mean()
        long_pct = (positions > 0.1).mean() * 100
        short_pct = (positions < -0.1).mean() * 100
        neutral_pct = 100 - long_pct - short_pct

        return {
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_duration': avg_duration,
            'n_long_trades': len(long_trades),
            'n_short_trades': len(short_trades),
            'mean_position': mean_pos,
            'long_pct': long_pct,
            'short_pct': short_pct,
            'neutral_pct': neutral_pct,
            'n_steps': len(equity) - 1,
            'equity_curve': equity,
            'positions': positions,
        }

    def print_report(self, name: str = "Backtest"):
        """
        Print professional backtest report.

        Parameters
        ----------
        name : str
            Name to display in report header
        """
        m = self.get_metrics()

        print(f"\n{'='*60}")
        print(f"{name} Report".center(60))
        print(f"{'='*60}")

        print(f"\n{'='*20} Portfolio Overview {'='*20}")
        print(f"* Initial balance: ${m['initial_balance']:,.2f}")
        print(f"* Final balance: ${m['final_balance']:,.2f}")
        print(f"* Total return: {m['total_return']:+.2f}%")
        print(f"* Sharpe ratio: {m['sharpe']:.2f}")
        print(f"* Max drawdown: {m['max_drawdown']:.2f}%")

        print(f"\n{'='*20} Trades Overview {'='*20}")
        print(f"* Number of trades: {m['n_trades']}")
        print(f"* Win rate: {m['win_rate']:.1f}%")
        print(f"* Profit factor: {m['profit_factor']:.2f}")
        print(f"* Avg win: ${m['avg_win']:.2f}")
        print(f"* Avg loss: ${m['avg_loss']:.2f}")
        print(f"* Avg duration: {m['avg_duration']:.0f} bars")
        print(f"* Long trades: {m['n_long_trades']}")
        print(f"* Short trades: {m['n_short_trades']}")

        print(f"\n{'='*20} Position Distribution {'='*20}")
        print(f"* Mean position: {m['mean_position']:+.3f}")
        print(f"* Time long: {m['long_pct']:.1f}%")
        print(f"* Time short: {m['short_pct']:.1f}%")
        print(f"* Time neutral: {m['neutral_pct']:.1f}%")

        print(f"\n{'='*60}\n")
