"""
Full 2025 Backtest - Complete evaluation over entire OOS period.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO


def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio."""
    excess_returns = returns - risk_free_rate
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    downside_std = np.std(downside_returns)
    if downside_std < 1e-10:
        return np.inf
    return np.mean(excess_returns) / downside_std


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    std = np.std(returns)
    if std < 1e-10:
        return 0.0
    return np.mean(excess_returns) / std


def calculate_max_drawdown(equity_curve: np.ndarray) -> tuple:
    """Calculate maximum drawdown and duration."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_dd = np.max(drawdown)
    max_dd_idx = np.argmax(drawdown)

    # Find drawdown duration
    peak_idx = np.argmax(equity_curve[:max_dd_idx+1])
    recovery_idx = max_dd_idx
    for i in range(max_dd_idx, len(equity_curve)):
        if equity_curve[i] >= peak[max_dd_idx]:
            recovery_idx = i
            break

    return max_dd, recovery_idx - peak_idx


def run_full_backtest():
    """Run backtest on entire 2025 dataset."""

    print("=" * 60)
    print("FULL 2025 BACKTEST")
    print("=" * 60)

    # Paths
    model_path = PROJECT_ROOT / "models" / "ppo_v20260202_104456_production" / "final_model.zip"
    dataset_path = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "datasets_5min" / "RL_DS3_MACRO_CORE.csv"
    norm_stats_path = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "datasets_5min" / "DS3_MACRO_CORE_norm_stats.json"

    # Load model
    print(f"\nLoading model: {model_path.name}")
    model = PPO.load(model_path)

    # Load dataset
    print(f"Loading dataset: {dataset_path.name}")
    df = pd.read_csv(dataset_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter to 2025 only
    test_start = pd.to_datetime('2025-01-01')
    df_test = df[df['timestamp'] >= test_start].reset_index(drop=True)

    print(f"Test period: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
    print(f"Total bars: {len(df_test)}")

    # Load norm stats
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    # Feature columns (13 market features + 2 state features)
    feature_cols = [
        'log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'rsi_9', 'volatility_pct', 'trend_z',
        'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z', 'brent_change_1d',
        'rate_spread_z', 'usdmxn_change_1d'
    ]

    # Configuration
    INITIAL_CAPITAL = 10_000.0
    TRANSACTION_COST_BPS = 75.0
    SLIPPAGE_BPS = 15.0
    THRESHOLD_LONG = 0.40
    THRESHOLD_SHORT = -0.40
    MAX_POSITION_HOLDING = 576

    total_cost_bps = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10_000

    # State variables
    capital = INITIAL_CAPITAL
    position = 0  # -1, 0, 1
    entry_bar = 0
    unrealized_pnl = 0.0

    # Tracking
    equity_curve = [capital]
    returns_list = []
    trades = []
    positions_held = []
    actions_taken = {'long': 0, 'hold': 0, 'short': 0}

    print("\nRunning backtest...")

    for i in range(1, len(df_test)):
        # Build observation
        obs = np.zeros(15, dtype=np.float32)

        # Market features (normalized)
        for j, col in enumerate(feature_cols):
            value = df_test.iloc[i][col]
            if col.endswith('_z'):
                obs[j] = np.clip(value, -5, 5)
            elif col in norm_stats:
                mean = norm_stats[col].get('mean', 0)
                std = norm_stats[col].get('std', 1)
                if std < 1e-8:
                    std = 1.0
                obs[j] = np.clip((value - mean) / std, -5, 5)
            else:
                obs[j] = np.clip(value, -5, 5)

        # State features
        obs[13] = position  # Current position
        obs[14] = unrealized_pnl  # Unrealized PnL (simplified)

        # Replace NaN
        obs = np.nan_to_num(obs, nan=0.0)

        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        action_value = float(action[0])

        # Map to discrete action
        if action_value > THRESHOLD_LONG:
            target_action = 1  # LONG
            actions_taken['long'] += 1
        elif action_value < THRESHOLD_SHORT:
            target_action = -1  # SHORT
            actions_taken['short'] += 1
        else:
            target_action = 0  # HOLD
            actions_taken['hold'] += 1

        # Check max position holding - force close
        if position != 0 and (i - entry_bar) >= MAX_POSITION_HOLDING:
            target_action = 0  # Force to flat

        # Calculate market return
        market_return = df_test.iloc[i]['log_ret_5m']
        if 'raw_log_ret_5m' in df_test.columns:
            market_return = df_test.iloc[i]['raw_log_ret_5m']

        # PnL from current position
        pnl = position * market_return * capital

        # Transaction cost if position changes
        trade_cost = 0.0
        if target_action != position:
            # Calculate cost
            change_magnitude = abs(target_action - position)
            trade_cost = change_magnitude * total_cost_bps * capital

            # Record trade
            if position != 0:  # Closing a position
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'bars_held': i - entry_bar,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'pnl': unrealized_pnl
                })

            # Update position
            position = target_action
            entry_bar = i
            unrealized_pnl = 0.0
        else:
            unrealized_pnl += pnl

        # Update capital
        net_pnl = pnl - trade_cost
        capital += net_pnl

        equity_curve.append(capital)
        returns_list.append(net_pnl / equity_curve[-2] if equity_curve[-2] > 0 else 0)
        positions_held.append(position)

    # Close final position if any
    if position != 0:
        trades.append({
            'entry_bar': entry_bar,
            'exit_bar': len(df_test) - 1,
            'bars_held': len(df_test) - 1 - entry_bar,
            'direction': 'LONG' if position == 1 else 'SHORT',
            'pnl': unrealized_pnl
        })

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    returns = np.array(returns_list)

    total_pnl = capital - INITIAL_CAPITAL
    total_return = (capital / INITIAL_CAPITAL - 1) * 100

    max_dd, dd_duration = calculate_max_drawdown(equity_curve)

    # Win rate
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

    # Profit factor
    gross_profit = sum(t['pnl'] for t in winning_trades)
    gross_loss = abs(sum(t['pnl'] for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Sharpe & Sortino (annualized)
    # 288 bars per day (24h), 252 trading days
    bars_per_year = 288 * 252
    sharpe_annual = calculate_sharpe(returns) * np.sqrt(bars_per_year)
    sortino_annual = calculate_sortino(returns) * np.sqrt(bars_per_year)

    # Daily returns for better ratio calculation
    daily_returns = []
    bars_per_day = 144  # 12h trading day
    for d in range(0, len(returns), bars_per_day):
        chunk = returns[d:d+bars_per_day]
        if len(chunk) > 0:
            daily_returns.append(np.sum(chunk))
    daily_returns = np.array(daily_returns)

    sharpe_daily = calculate_sharpe(daily_returns) * np.sqrt(252)
    sortino_daily = calculate_sortino(daily_returns) * np.sqrt(252)

    # APR calculation
    trading_days = len(df_test) / bars_per_day
    daily_return_avg = total_return / trading_days
    apr_simple = daily_return_avg * 252
    apr_compound = ((1 + total_return/100) ** (252/trading_days) - 1) * 100

    # Average trade stats
    avg_trade_pnl = np.mean([t['pnl'] for t in trades]) if trades else 0
    avg_bars_held = np.mean([t['bars_held'] for t in trades]) if trades else 0
    avg_trade_duration_hours = avg_bars_held * 5 / 60

    # Action distribution
    total_actions = sum(actions_taken.values())
    action_pct = {k: v/total_actions*100 for k, v in actions_taken.items()}

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS - FULL 2025")
    print("=" * 60)

    print(f"\n{'PERIOD':40} {df_test['timestamp'].min().date()} to {df_test['timestamp'].max().date()}")
    print(f"{'Total Bars':40} {len(df_test):,}")
    print(f"{'Trading Days':40} {trading_days:.1f}")

    print("\n" + "-" * 60)
    print("CAPITAL")
    print("-" * 60)
    print(f"{'Initial Capital':40} ${INITIAL_CAPITAL:,.2f}")
    print(f"{'Final Capital':40} ${capital:,.2f}")
    print(f"{'Total PnL':40} ${total_pnl:,.2f}")
    print(f"{'Total Return':40} {total_return:.2f}%")

    print("\n" + "-" * 60)
    print("ANNUALIZED RETURNS")
    print("-" * 60)
    print(f"{'APR (Simple)':40} {apr_simple:.1f}%")
    print(f"{'APR (Compound)':40} {apr_compound:.1f}%")

    print("\n" + "-" * 60)
    print("RISK METRICS")
    print("-" * 60)
    print(f"{'Max Drawdown':40} {max_dd*100:.2f}%")
    print(f"{'Drawdown Duration (bars)':40} {dd_duration}")
    print(f"{'Sharpe Ratio (annualized)':40} {sharpe_daily:.2f}")
    print(f"{'Sortino Ratio (annualized)':40} {sortino_daily:.2f}")

    print("\n" + "-" * 60)
    print("TRADE STATISTICS")
    print("-" * 60)
    print(f"{'Total Trades':40} {len(trades)}")
    print(f"{'Winning Trades':40} {len(winning_trades)}")
    print(f"{'Losing Trades':40} {len(losing_trades)}")
    print(f"{'Win Rate':40} {win_rate:.1f}%")
    print(f"{'Profit Factor':40} {profit_factor:.2f}")
    print(f"{'Avg Trade PnL':40} ${avg_trade_pnl:.2f}")
    print(f"{'Avg Trade Duration':40} {avg_trade_duration_hours:.1f} hours ({avg_bars_held:.0f} bars)")

    print("\n" + "-" * 60)
    print("ACTION DISTRIBUTION")
    print("-" * 60)
    print(f"{'LONG':40} {action_pct['long']:.1f}%")
    print(f"{'HOLD':40} {action_pct['hold']:.1f}%")
    print(f"{'SHORT':40} {action_pct['short']:.1f}%")

    print("\n" + "-" * 60)
    print("TRADE DETAILS")
    print("-" * 60)
    for i, t in enumerate(trades, 1):
        pnl_str = f"+${t['pnl']:.2f}" if t['pnl'] > 0 else f"-${abs(t['pnl']):.2f}"
        print(f"Trade {i:2d}: {t['direction']:5s} | Held {t['bars_held']:4d} bars ({t['bars_held']*5/60:.1f}h) | PnL: {pnl_str}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
    Capital: ${INITIAL_CAPITAL:,.0f} -> ${capital:,.2f} ({total_return:+.2f}%)
    APR: {apr_simple:.0f}% (simple) / {apr_compound:.0f}% (compound)
    Sharpe: {sharpe_daily:.2f} | Sortino: {sortino_daily:.2f}
    Max DD: {max_dd*100:.2f}% | Win Rate: {win_rate:.0f}%
    Trades: {len(trades)} | Profit Factor: {profit_factor:.2f}
    """)

    # Save results
    results = {
        'period': {
            'start': str(df_test['timestamp'].min()),
            'end': str(df_test['timestamp'].max()),
            'bars': len(df_test),
            'trading_days': trading_days
        },
        'capital': {
            'initial': INITIAL_CAPITAL,
            'final': capital,
            'pnl': total_pnl,
            'return_pct': total_return
        },
        'annualized': {
            'apr_simple': apr_simple,
            'apr_compound': apr_compound
        },
        'risk': {
            'max_drawdown_pct': max_dd * 100,
            'sharpe_annual': sharpe_daily,
            'sortino_annual': sortino_daily
        },
        'trades': {
            'total': len(trades),
            'winning': len(winning_trades),
            'losing': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_pnl': avg_trade_pnl,
            'avg_duration_hours': avg_trade_duration_hours
        },
        'actions': action_pct,
        'trade_details': trades
    }

    results_path = PROJECT_ROOT / "results" / f"full_2025_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    return results


if __name__ == "__main__":
    run_full_backtest()
