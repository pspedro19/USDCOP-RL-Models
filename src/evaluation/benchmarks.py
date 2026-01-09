"""
Benchmark Strategies for Model Comparison
=========================================

V20 Implementation: Benchmark strategies to compare RL model performance
against simple baselines.

From: 09_Documento Maestro Completo.md Section 6.9

Strategies:
- Buy & Hold: Simple long-only strategy
- MA Crossover: Moving average crossover (20/50)
- Random: Random signal generation (baseline)
- Always Flat: Never trade (risk-free baseline)

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result from a benchmark strategy backtest."""
    name: str
    equity_curve: np.ndarray
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    trade_count: int
    avg_trade_pnl_pct: float


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from an equity curve.

    Args:
        equity_curve: Array of equity values

    Returns:
        Maximum drawdown as decimal (e.g., 0.15 = 15%)
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return max_dd


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 60  # 5-min bars
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    std = np.std(excess_returns)

    if std == 0:
        return 0.0

    return np.mean(excess_returns) / std * np.sqrt(periods_per_year)


class BenchmarkStrategies:
    """
    Collection of benchmark trading strategies.

    Use these to compare RL model performance against simple baselines.
    If the model can't beat these benchmarks, it needs improvement.
    """

    def __init__(self, transaction_cost: float = 0.002):
        """
        Initialize benchmark strategies.

        Args:
            transaction_cost: Cost per trade as decimal (0.002 = 0.2%)
        """
        self.transaction_cost = transaction_cost

    def buy_and_hold(
        self,
        prices: np.ndarray,
        initial_equity: float = 10000.0
    ) -> BenchmarkResult:
        """
        Simple buy and hold strategy.

        Buy at start, hold until end. No trading costs after initial purchase.

        Args:
            prices: Array of close prices
            initial_equity: Starting capital

        Returns:
            BenchmarkResult with strategy performance
        """
        if len(prices) < 2:
            return self._empty_result("Buy & Hold")

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]

        # Build equity curve
        equity = [initial_equity * (1 - self.transaction_cost)]  # Initial buy cost
        for r in returns:
            equity.append(equity[-1] * (1 + r))

        equity = np.array(equity)
        total_returns = np.diff(equity) / equity[:-1]

        return BenchmarkResult(
            name="Buy & Hold",
            equity_curve=equity,
            total_return_pct=(equity[-1] / initial_equity - 1) * 100,
            sharpe_ratio=calculate_sharpe_ratio(total_returns),
            max_drawdown_pct=calculate_max_drawdown(equity) * 100,
            win_rate_pct=np.mean(returns > 0) * 100,
            trade_count=1,
            avg_trade_pnl_pct=(equity[-1] / initial_equity - 1) * 100
        )

    def ma_crossover(
        self,
        prices: np.ndarray,
        fast_period: int = 20,
        slow_period: int = 50,
        initial_equity: float = 10000.0
    ) -> BenchmarkResult:
        """
        Moving average crossover strategy.

        Go long when fast MA > slow MA, short when fast MA < slow MA.

        Args:
            prices: Array of close prices
            fast_period: Fast MA period
            slow_period: Slow MA period
            initial_equity: Starting capital

        Returns:
            BenchmarkResult with strategy performance
        """
        if len(prices) < slow_period + 1:
            return self._empty_result("MA Crossover")

        # Calculate MAs
        df = pd.DataFrame({'close': prices})
        df['ma_fast'] = df['close'].rolling(fast_period).mean()
        df['ma_slow'] = df['close'].rolling(slow_period).mean()

        # Generate signals
        df['signal'] = 0
        df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1   # Long
        df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1  # Short

        # Simulate trading
        equity = [initial_equity]
        position = 0
        trades = []
        trade_pnls = []

        for i in range(1, len(df)):
            # Calculate P&L if in position
            if position != 0:
                pnl_pct = position * (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
                new_equity = equity[-1] * (1 + pnl_pct)
            else:
                new_equity = equity[-1]

            # Check for position change
            new_signal = df['signal'].iloc[i]
            if not pd.isna(new_signal) and new_signal != position:
                # Apply transaction cost
                new_equity *= (1 - self.transaction_cost)
                trades.append({
                    'bar': i,
                    'old_pos': position,
                    'new_pos': new_signal,
                    'price': df['close'].iloc[i]
                })

                # Record trade P&L if closing position
                if position != 0 and len(trades) > 1:
                    trade_pnls.append(pnl_pct * 100)

                position = new_signal

            equity.append(new_equity)

        equity = np.array(equity)
        total_returns = np.diff(equity) / equity[:-1]

        win_rate = 0.0
        avg_pnl = 0.0
        if len(trade_pnls) > 0:
            win_rate = np.mean(np.array(trade_pnls) > 0) * 100
            avg_pnl = np.mean(trade_pnls)

        return BenchmarkResult(
            name=f"MA Crossover ({fast_period}/{slow_period})",
            equity_curve=equity,
            total_return_pct=(equity[-1] / initial_equity - 1) * 100,
            sharpe_ratio=calculate_sharpe_ratio(total_returns),
            max_drawdown_pct=calculate_max_drawdown(equity) * 100,
            win_rate_pct=win_rate,
            trade_count=len(trades),
            avg_trade_pnl_pct=avg_pnl
        )

    def random_signals(
        self,
        prices: np.ndarray,
        trade_probability: float = 0.05,
        initial_equity: float = 10000.0,
        seed: int = 42
    ) -> BenchmarkResult:
        """
        Random signal strategy (baseline).

        Randomly enter long/short positions with given probability.

        Args:
            prices: Array of close prices
            trade_probability: Probability of trade per bar
            initial_equity: Starting capital
            seed: Random seed for reproducibility

        Returns:
            BenchmarkResult with strategy performance
        """
        if len(prices) < 2:
            return self._empty_result("Random")

        np.random.seed(seed)

        equity = [initial_equity]
        position = 0
        trades = []
        trade_pnls = []

        for i in range(1, len(prices)):
            # Calculate P&L if in position
            if position != 0:
                pnl_pct = position * (prices[i] - prices[i-1]) / prices[i-1]
                new_equity = equity[-1] * (1 + pnl_pct)
            else:
                new_equity = equity[-1]

            # Random signal
            if np.random.random() < trade_probability:
                new_signal = np.random.choice([-1, 0, 1])

                if new_signal != position:
                    new_equity *= (1 - self.transaction_cost)
                    trades.append({'bar': i, 'signal': new_signal})

                    if position != 0:
                        trade_pnls.append(pnl_pct * 100)

                    position = new_signal

            equity.append(new_equity)

        equity = np.array(equity)
        total_returns = np.diff(equity) / equity[:-1]

        win_rate = 0.0
        avg_pnl = 0.0
        if len(trade_pnls) > 0:
            win_rate = np.mean(np.array(trade_pnls) > 0) * 100
            avg_pnl = np.mean(trade_pnls)

        return BenchmarkResult(
            name="Random Signals",
            equity_curve=equity,
            total_return_pct=(equity[-1] / initial_equity - 1) * 100,
            sharpe_ratio=calculate_sharpe_ratio(total_returns),
            max_drawdown_pct=calculate_max_drawdown(equity) * 100,
            win_rate_pct=win_rate,
            trade_count=len(trades),
            avg_trade_pnl_pct=avg_pnl
        )

    def always_flat(
        self,
        prices: np.ndarray,
        initial_equity: float = 10000.0
    ) -> BenchmarkResult:
        """
        Always flat strategy (risk-free baseline).

        Never trade, equity stays constant. Use to verify model adds value.

        Args:
            prices: Array of close prices
            initial_equity: Starting capital

        Returns:
            BenchmarkResult with flat equity curve
        """
        equity = np.full(len(prices), initial_equity)

        return BenchmarkResult(
            name="Always Flat (Risk-Free)",
            equity_curve=equity,
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            trade_count=0,
            avg_trade_pnl_pct=0.0
        )

    def _empty_result(self, name: str) -> BenchmarkResult:
        """Create empty result for insufficient data."""
        return BenchmarkResult(
            name=name,
            equity_curve=np.array([10000.0]),
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            trade_count=0,
            avg_trade_pnl_pct=0.0
        )


def compare_with_benchmarks(
    model_equity: np.ndarray,
    prices: np.ndarray,
    model_name: str = "RL Model",
    model_trades: int = 0,
    model_wins: int = 0,
    transaction_cost: float = 0.002
) -> Dict[str, Dict[str, float]]:
    """
    Compare model performance with all benchmarks.

    Args:
        model_equity: Model's equity curve
        prices: Price data used for trading
        model_name: Name for the model
        model_trades: Number of trades by model
        model_wins: Number of winning trades
        transaction_cost: Transaction cost for benchmarks

    Returns:
        Dict with performance metrics for each strategy
    """
    benchmarks = BenchmarkStrategies(transaction_cost=transaction_cost)

    # Run all benchmarks
    results = {
        "Buy & Hold": benchmarks.buy_and_hold(prices),
        "MA Crossover (20/50)": benchmarks.ma_crossover(prices, 20, 50),
        "Random": benchmarks.random_signals(prices),
        "Always Flat": benchmarks.always_flat(prices),
    }

    # Calculate model metrics
    initial_equity = model_equity[0]
    model_returns = np.diff(model_equity) / model_equity[:-1]
    model_win_rate = (model_wins / model_trades * 100) if model_trades > 0 else 0.0

    results[model_name] = BenchmarkResult(
        name=model_name,
        equity_curve=model_equity,
        total_return_pct=(model_equity[-1] / initial_equity - 1) * 100,
        sharpe_ratio=calculate_sharpe_ratio(model_returns),
        max_drawdown_pct=calculate_max_drawdown(model_equity) * 100,
        win_rate_pct=model_win_rate,
        trade_count=model_trades,
        avg_trade_pnl_pct=(model_equity[-1] / initial_equity - 1) / max(model_trades, 1) * 100
    )

    # Convert to dict format
    comparison = {}
    for name, result in results.items():
        comparison[name] = {
            "total_return_pct": round(result.total_return_pct, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 3),
            "max_drawdown_pct": round(result.max_drawdown_pct, 2),
            "win_rate_pct": round(result.win_rate_pct, 1),
            "trade_count": result.trade_count,
        }

    return comparison


def print_benchmark_comparison(comparison: Dict[str, Dict[str, float]]) -> None:
    """Print formatted benchmark comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    headers = ["Strategy", "Return %", "Sharpe", "Max DD %", "Win Rate %", "Trades"]
    row_format = "{:<25} {:>10} {:>10} {:>10} {:>12} {:>8}"

    print(row_format.format(*headers))
    print("-" * 80)

    for name, metrics in comparison.items():
        print(row_format.format(
            name[:24],
            f"{metrics['total_return_pct']:.2f}%",
            f"{metrics['sharpe_ratio']:.3f}",
            f"{metrics['max_drawdown_pct']:.2f}%",
            f"{metrics['win_rate_pct']:.1f}%",
            metrics['trade_count']
        ))

    print("=" * 80)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    # Generate synthetic price data (random walk with drift)
    n_bars = 1000
    returns = np.random.normal(0.0001, 0.002, n_bars)  # Small positive drift
    prices = 4200 * np.cumprod(1 + returns)

    # Generate synthetic model equity (slightly better than random)
    model_returns = returns + np.random.normal(0.00005, 0.001, n_bars)
    model_equity = 10000 * np.cumprod(1 + model_returns)

    # Compare
    comparison = compare_with_benchmarks(
        model_equity=model_equity,
        prices=prices,
        model_name="PPO V20",
        model_trades=50,
        model_wins=25
    )

    print_benchmark_comparison(comparison)

    print("\nBenchmark module test complete!")
