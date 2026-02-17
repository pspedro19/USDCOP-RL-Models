"""
Replay Backtest Engine — Thin orchestrator.
=============================================

Ties signals + execution strategies + metrics into a single replay loop.

Usage:
    engine = ReplayBacktestEngine(WeeklyTPHSExecution(), initial_capital=10000)
    result = engine.replay(signals, ohlcv_df)
    # result.trades -> List[StrategyTrade]
    # result.stats  -> StrategyStats
    # result.statistical_tests -> dict with p_value, sharpe, etc.

Contract: CTR-REPLAY-001
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.contracts.signal_contract import UniversalSignalRecord
from src.contracts.strategy_schema import StrategyTrade, StrategyStats
from src.contracts.execution_strategies import ExecutionStrategy

import pandas as pd


# ---------------------------------------------------------------------------
# ReplayResult
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    """Output of ReplayBacktestEngine.replay()."""
    trades: List[StrategyTrade]
    stats: StrategyStats
    statistical_tests: dict
    signals_total: int
    signals_skipped: int
    signals_traded: int

    @property
    def is_significant(self) -> bool:
        return self.statistical_tests.get("p_value", 1.0) < 0.05


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_strategy_stats(
    trades: List[StrategyTrade],
    initial_capital: float,
) -> StrategyStats:
    """Compute StrategyStats from a list of trades."""
    if not trades:
        return StrategyStats(
            final_equity=initial_capital,
            total_return_pct=0.0,
        )

    final_equity = trades[-1].equity_at_exit
    total_return_pct = (final_equity / initial_capital - 1) * 100

    # Win/loss counts
    pnl_pcts = [t.pnl_pct for t in trades]
    winning = sum(1 for p in pnl_pcts if p > 0)
    losing = sum(1 for p in pnl_pcts if p <= 0)
    total = len(trades)
    win_rate = (winning / total * 100) if total > 0 else None

    # Profit factor
    gross_profit = sum(p for p in pnl_pcts if p > 0)
    gross_loss = abs(sum(p for p in pnl_pcts if p < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = None  # NEVER Infinity

    # Sharpe (weekly for H5, daily for H1 — use trade PnL% as returns)
    returns = np.array([p / 100 for p in pnl_pcts])
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(52))
    else:
        sharpe = None

    # Max drawdown
    equity_curve = [initial_capital]
    for t in trades:
        equity_curve.append(t.equity_at_exit)
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = abs(float(np.min(dd))) * 100

    # Trading days
    dates = set()
    for t in trades:
        if t.timestamp:
            dates.add(t.timestamp[:10])
    trading_days = len(dates)

    # Exit reasons
    exit_reasons: Dict[str, int] = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Long/Short counts
    n_long = sum(1 for t in trades if t.side == "LONG")
    n_short = sum(1 for t in trades if t.side == "SHORT")

    return StrategyStats(
        final_equity=round(final_equity, 2),
        total_return_pct=round(total_return_pct, 2),
        sharpe=round(sharpe, 3) if sharpe is not None else None,
        max_dd_pct=round(max_dd, 2),
        win_rate_pct=round(win_rate, 1) if win_rate is not None else None,
        profit_factor=round(profit_factor, 2) if profit_factor is not None else None,
        trading_days=trading_days,
        exit_reasons=exit_reasons,
        n_long=n_long,
        n_short=n_short,
    )


def compute_statistical_tests(
    trades: List[StrategyTrade],
) -> dict:
    """Compute statistical tests from trades."""
    if not trades:
        return {"p_value": 1.0, "significant": False}

    returns = np.array([t.pnl_pct / 100 for t in trades])

    # Bootstrap p-value (one-sided: H0 = mean <= 0)
    np.random.seed(42)
    n_boot = 10000
    boot_means = [
        np.mean(np.random.choice(returns, size=len(returns), replace=True))
        for _ in range(n_boot)
    ]
    p_value = float(np.mean(np.array(boot_means) <= 0))

    # t-test
    if len(returns) > 1 and np.std(returns) > 0:
        t_stat = float(np.mean(returns) / (np.std(returns, ddof=1) / np.sqrt(len(returns))))
    else:
        t_stat = 0.0

    # Bootstrap 95% CI
    boot_arr = np.array(boot_means)
    ci_lo = float(np.percentile(boot_arr, 2.5)) * np.sqrt(52)  # Annualized
    ci_hi = float(np.percentile(boot_arr, 97.5)) * np.sqrt(52)

    return {
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "t_stat": round(t_stat, 3),
        "bootstrap_95ci_ann": [round(ci_lo, 4), round(ci_hi, 4)],
    }


def compute_direction_accuracy(
    trades: List[StrategyTrade],
    ohlcv: pd.DataFrame,
) -> Optional[float]:
    """Compute direction accuracy: how often did the trade direction match actual move."""
    if not trades:
        return None

    correct = 0
    total = 0
    for t in trades:
        # Get actual close at week/period end
        if t.exit_timestamp:
            exit_ts = pd.Timestamp(t.exit_timestamp)
            row = ohlcv[ohlcv["date"] <= exit_ts].tail(1)
            if row.empty:
                continue
            actual_close = float(row["close"].iloc[0])
            direction = 1 if t.side == "LONG" else -1
            actual_direction = 1 if actual_close > t.entry_price else -1
            if direction == actual_direction:
                correct += 1
            total += 1

    if total == 0:
        return None
    return round(correct / total * 100, 1)


# ---------------------------------------------------------------------------
# ReplayBacktestEngine
# ---------------------------------------------------------------------------

class ReplayBacktestEngine:
    """
    Thin orchestrator that replays universal signals through an execution strategy.

    Usage:
        engine = ReplayBacktestEngine(WeeklyTPHSExecution())
        result = engine.replay(signals, ohlcv)
    """

    def __init__(
        self,
        execution_strategy: ExecutionStrategy,
        initial_capital: float = 10000.0,
    ):
        self.executor = execution_strategy
        self.initial_capital = initial_capital

    def replay(
        self,
        signals: List[UniversalSignalRecord],
        ohlcv: pd.DataFrame,
    ) -> ReplayResult:
        """
        Replay signals through OHLCV data.

        Args:
            signals: Ordered list of universal signals.
            ohlcv: DataFrame with [date, open, high, low, close] columns.

        Returns:
            ReplayResult with trades, stats, and statistical tests.
        """
        equity = self.initial_capital
        trades: List[StrategyTrade] = []
        skipped = 0

        for signal in signals:
            if signal.skip_trade or signal.direction == 0:
                skipped += 1
                continue

            trade = self.executor.execute(signal, ohlcv, equity)
            if trade is not None:
                equity = trade.equity_at_exit
                trades.append(trade)
            else:
                skipped += 1

        stats = compute_strategy_stats(trades, self.initial_capital)
        tests = compute_statistical_tests(trades)

        return ReplayResult(
            trades=trades,
            stats=stats,
            statistical_tests=tests,
            signals_total=len(signals),
            signals_skipped=skipped,
            signals_traded=len(trades),
        )
