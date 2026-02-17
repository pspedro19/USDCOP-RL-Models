"""
SMA Baseline on Hourly USDCOP Data (PASO 0)
=============================================

Goal: Answer "does alpha exist at hourly frequency?"

Strategy:
    - SMA(20) crosses above SMA(50) -> LONG
    - SMA(20) crosses below SMA(50) -> SHORT
    - Trailing stop at 2%
    - SL = -4%, TP = +4% (match V21.5b)
    - Costs: 0 bps (maker) + 1 bps (slippage)

Evaluation: 2025 OOS data
Compare vs: B&H (-14.66%), random agent (-4.12%), V21.5b (+2.51%)

GO/NO-GO: If SMA baseline < -20%, alpha at hourly is questionable.

Usage:
    python scripts/sma_baseline_hourly.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_sma_baseline(
    df: pd.DataFrame,
    sma_fast: int = 20,
    sma_slow: int = 50,
    sl_pct: float = -0.04,
    tp_pct: float = 0.04,
    trailing_stop_pct: float = 0.02,
    cost_bps: float = 1.0,
    initial_capital: float = 10000.0,
) -> dict:
    """Run SMA crossover strategy on hourly bars."""

    cost_rate = cost_bps / 10000.0

    # Calculate SMAs
    df = df.copy()
    df['sma_fast'] = df['close'].rolling(sma_fast).mean()
    df['sma_slow'] = df['close'].rolling(sma_slow).mean()
    df = df.dropna(subset=['sma_fast', 'sma_slow'])

    # Generate signals: 1=LONG, -1=SHORT
    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1

    # Simulate trading
    capital = initial_capital
    position = 0  # -1, 0, 1
    entry_price = 0.0
    peak_pnl = 0.0
    trades = []
    equity_curve = [capital]

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        price = row['close']
        signal = row['signal']

        # Calculate PnL if in position
        step_pnl = 0.0
        if position != 0:
            log_ret = np.log(price / prev_row['close'])
            step_pnl = position * log_ret * capital

            # Check stops
            unrealized_pnl_pct = (price - entry_price) / entry_price * position
            peak_pnl = max(peak_pnl, unrealized_pnl_pct)

            close_reason = None

            # Stop loss
            if unrealized_pnl_pct <= sl_pct:
                close_reason = 'sl'
            # Take profit
            elif unrealized_pnl_pct >= tp_pct:
                close_reason = 'tp'
            # Trailing stop: activates after peak_pnl > trailing_stop_pct
            elif peak_pnl > trailing_stop_pct and unrealized_pnl_pct < peak_pnl - trailing_stop_pct:
                close_reason = 'trailing'

            if close_reason:
                # Close position
                cost = abs(position) * cost_rate * capital
                capital += step_pnl - cost
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': price,
                    'direction': position,
                    'pnl_pct': unrealized_pnl_pct,
                    'reason': close_reason,
                })
                position = 0
                entry_price = 0.0
                peak_pnl = 0.0
                equity_curve.append(capital)
                continue

        # Update capital with PnL
        capital += step_pnl

        # Signal-based entry/exit
        target = signal
        if target != position:
            # Close current position (if any)
            if position != 0:
                cost = abs(position) * cost_rate * capital
                capital -= cost
                unrealized_pnl_pct = (price - entry_price) / entry_price * position
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': price,
                    'direction': position,
                    'pnl_pct': unrealized_pnl_pct,
                    'reason': 'signal',
                })

            # Open new position
            if target != 0:
                cost = abs(target) * cost_rate * capital
                capital -= cost
                position = target
                entry_price = price
                peak_pnl = 0.0
            else:
                position = 0
                entry_price = 0.0
                peak_pnl = 0.0

        equity_curve.append(capital)

    # Close final position
    if position != 0:
        price = df.iloc[-1]['close']
        unrealized_pnl_pct = (price - entry_price) / entry_price * position
        trades.append({
            'entry_price': entry_price,
            'exit_price': price,
            'direction': position,
            'pnl_pct': unrealized_pnl_pct,
            'reason': 'end',
        })

    # Calculate metrics
    equity = np.array(equity_curve)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-8)
    total_return = (equity[-1] / equity[0] - 1) * 100

    # Annualization: 1260 bars/year for hourly
    bars_per_year = 1260
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year)) if np.std(returns) > 0 else 0.0

    # Max drawdown
    peak_eq = np.maximum.accumulate(equity)
    drawdown = (peak_eq - equity) / np.maximum(peak_eq, 1e-8)
    max_dd = float(np.max(drawdown) * 100)

    # Trade stats
    n_trades = len(trades)
    winning = sum(1 for t in trades if t['pnl_pct'] > 0)
    losing = sum(1 for t in trades if t['pnl_pct'] <= 0)
    win_rate = winning / max(n_trades, 1) * 100

    # Profit factor
    gross_profit = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    gross_loss = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    profit_factor = gross_profit / max(gross_loss, 1e-10)

    # Sortino
    downside = returns[returns < 0]
    sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(bars_per_year)) if len(downside) > 0 and np.std(downside) > 0 else 0.0

    # Close reason breakdown
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1

    return {
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown_pct': max_dd,
        'n_trades': n_trades,
        'winning_trades': winning,
        'losing_trades': losing,
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'close_reasons': reasons,
        'equity_curve': equity_curve,
        'trades': trades,
        'bars_evaluated': len(df),
    }


def main():
    # Load hourly seed
    hourly_path = PROJECT_ROOT / "seeds/latest/usdcop_1h_ohlcv.parquet"
    if not hourly_path.exists():
        print("Hourly seed not found. Run: python scripts/resample_5m_to_1h.py")
        return

    df = pd.read_parquet(hourly_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    if 'symbol' in df.columns:
        df = df.drop(columns=['symbol'])
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    # 2025 OOS data
    oos_start = '2025-01-01'
    oos_end = '2025-12-31'
    df_oos = df[(df.index >= oos_start) & (df.index <= oos_end)]
    print(f"OOS data: {len(df_oos)} bars ({df_oos.index.min()} to {df_oos.index.max()})")

    # Full dataset (including warmup for SMA calculation)
    warmup_start = '2024-09-01'  # ~4 months warmup for SMA(50)
    df_full = df[df.index >= warmup_start]
    print(f"Full data (with warmup): {len(df_full)} bars")

    # Run baseline with SMA warmup on full data
    print("\n" + "=" * 70)
    print("PASO 0: SMA BASELINE ON HOURLY USDCOP")
    print("=" * 70)
    print(f"Strategy: SMA({20}) / SMA({50}) crossover")
    print(f"Stops: SL={-4}%, TP={4}%, Trailing={2}%")
    print(f"Costs: 0 bps maker + 1 bps slippage")
    print(f"Period: {oos_start} to {oos_end}")

    # Compute SMAs on full dataset but evaluate only OOS
    df_full = df_full.copy()
    df_full['sma_fast'] = df_full['close'].rolling(20).mean()
    df_full['sma_slow'] = df_full['close'].rolling(50).mean()

    # Filter to OOS after SMA computation
    df_eval = df_full[(df_full.index >= oos_start) & (df_full.index <= oos_end)].copy()
    df_eval = df_eval.dropna(subset=['sma_fast', 'sma_slow'])
    print(f"OOS bars with valid SMAs: {len(df_eval)}")

    results = run_sma_baseline(df_eval)

    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total Return:    {results['total_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio:    {results['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:   {results['sortino_ratio']:.3f}")
    print(f"  Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
    print(f"  Trades:          {results['n_trades']} (W:{results['winning_trades']} L:{results['losing_trades']})")
    print(f"  Win Rate:        {results['win_rate_pct']:.1f}%")
    print(f"  Profit Factor:   {results['profit_factor']:.2f}")
    print(f"  Close Reasons:   {results['close_reasons']}")

    # Buy-and-hold comparison
    bh_ret = (df_eval.iloc[-1]['close'] / df_eval.iloc[0]['close'] - 1) * 100
    print(f"\n  Buy-and-Hold:    {bh_ret:+.2f}%")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  SMA Baseline:      {results['total_return_pct']:+.2f}%")
    print(f"  Buy-and-Hold:      {bh_ret:+.2f}%")
    print(f"  Random Agent:      -4.12%  (V21.5b reference)")
    print(f"  V21.5b (5min RL):  +2.51%  (best model)")

    # GO/NO-GO
    print("\n" + "=" * 70)
    if results['total_return_pct'] < -20:
        print("GO/NO-GO: NO-GO (< -20%, alpha at hourly is questionable)")
    elif results['total_return_pct'] > bh_ret:
        print(f"GO/NO-GO: STRONG GO (beats B&H by {results['total_return_pct'] - bh_ret:+.2f}pp)")
    elif results['total_return_pct'] > -4.12:
        print(f"GO/NO-GO: GO (beats random agent, margin = {results['total_return_pct'] + 4.12:+.2f}pp)")
    else:
        print(f"GO/NO-GO: MARGINAL (worse than random agent)")
    print("=" * 70)


if __name__ == "__main__":
    main()
