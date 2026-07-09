"""
Analyze when the PPO model loses money during 2025.
Identifies periods, months, and market conditions where losses occur.

Model: models/ppo_v20260202_104456_production/final_model.zip
Data: data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv (2025 subset)
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_v20260202_104456_production" / "final_model.zip"
DATA_PATH = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "datasets_5min" / "RL_DS3_MACRO_CORE.csv"
NORM_STATS_PATH = PROJECT_ROOT / "models" / "ppo_v20260202_104456_production" / "norm_stats.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "2025_loss_analysis.json"
OUTPUT_TXT_PATH = PROJECT_ROOT / "results" / "2025_loss_analysis.txt"

# Trading parameters (from trading_env.py)
TRANSACTION_COST_BPS = 75.0
SLIPPAGE_BPS = 15.0
THRESHOLD_LONG = 0.40
THRESHOLD_SHORT = -0.40
INITIAL_BALANCE = 10_000.0

# Feature list (from norm_stats)
CORE_FEATURES = [
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "volatility_pct", "trend_z",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
    "brent_change_1d", "rate_spread_z", "usdmxn_change_1d",
]


def load_data():
    """Load and filter 2025 data."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter for 2025
    df_2025 = df[df['timestamp'].dt.year == 2025].copy().reset_index(drop=True)
    print(f"Loaded {len(df_2025)} bars from 2025")
    print(f"Date range: {df_2025['timestamp'].min()} to {df_2025['timestamp'].max()}")

    return df_2025


def load_norm_stats():
    """Load normalization statistics."""
    with open(NORM_STATS_PATH, 'r') as f:
        return json.load(f)


def normalize_features(row, norm_stats):
    """Normalize features using z-score."""
    normalized = []
    for feature in CORE_FEATURES:
        value = row[feature]

        # Skip normalization for features already z-scored
        if feature.endswith('_z'):
            normalized.append(np.clip(value, -5.0, 5.0))
            continue

        if feature in norm_stats:
            mean = norm_stats[feature].get('mean', 0.0)
            std = norm_stats[feature].get('std', 1.0)

            # Skip if already normalized
            if abs(mean) < 0.1 and 0.8 < std < 1.2:
                normalized.append(np.clip(value, -5.0, 5.0))
                continue

            if std < 1e-8:
                std = 1.0
            z = (value - mean) / std
            normalized.append(np.clip(z, -5.0, 5.0))
        else:
            normalized.append(np.clip(value, -5.0, 5.0))

    return np.array(normalized, dtype=np.float32)


def run_simulation(model, df, norm_stats):
    """
    Run the model through 2025 data and track equity curve.

    Returns:
        List of dicts with timestamp, equity, pnl, position, action, market conditions
    """
    results = []

    balance = INITIAL_BALANCE
    position = 0  # -1 short, 0 flat, 1 long

    transaction_cost = TRANSACTION_COST_BPS / 10_000
    slippage = SLIPPAGE_BPS / 10_000

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # Build observation
        features = normalize_features(row, norm_stats)
        # Add state features: position (-1 to 1), time_normalized (0 to 1)
        time_normalized = i / len(df)
        obs = np.concatenate([features, [float(position), time_normalized]])
        obs = obs.astype(np.float32)

        # Get model action
        action, _ = model.predict(obs, deterministic=True)
        action_value = float(action[0])

        # Map to discrete action
        if action_value > THRESHOLD_LONG:
            target_position = 1
        elif action_value < THRESHOLD_SHORT:
            target_position = -1
        else:
            target_position = 0

        # Calculate trade cost if position changes
        trade_cost = 0.0
        position_changed = False
        if target_position != position:
            position_changed = True
            # Cost for changing position
            change_magnitude = abs(target_position - position)
            trade_cost = change_magnitude * (transaction_cost + slippage) * balance

        # Calculate PnL from market move
        market_return = next_row['log_ret_5m']
        gross_pnl = position * market_return * balance
        net_pnl = gross_pnl - trade_cost

        # Update balance
        balance += net_pnl

        # Record result
        results.append({
            'timestamp': row['timestamp'],
            'equity': balance,
            'pnl': net_pnl,
            'gross_pnl': gross_pnl,
            'trade_cost': trade_cost,
            'position': position,
            'action_value': action_value,
            'target_position': target_position,
            'position_changed': position_changed,
            'market_return': market_return,
            # Market conditions
            'close': row['close'],
            'volatility_pct': row['volatility_pct'],
            'trend_z': row['trend_z'],
            'vix_z': row['vix_z'],
            'dxy_z': row['dxy_z'],
            'rsi_9': row['rsi_9'],
        })

        # Update position
        position = target_position

    return results


def analyze_losses(results):
    """Analyze when and where losses occur."""
    df = pd.DataFrame(results)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.to_period('M')
    df['week'] = df['timestamp'].dt.isocalendar().week
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour

    # Calculate cumulative return
    df['cum_pnl'] = df['pnl'].cumsum()
    df['return_pct'] = (df['equity'] - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    # Calculate drawdown
    df['peak_equity'] = df['equity'].cummax()
    df['drawdown'] = (df['peak_equity'] - df['equity']) / df['peak_equity'] * 100

    analysis = {
        'summary': {},
        'monthly_analysis': {},
        'worst_periods': [],
        'loss_conditions': {},
        'win_vs_loss_conditions': {},
    }

    # Overall summary
    total_trades = df['position_changed'].sum()
    total_pnl = df['pnl'].sum()
    winning_bars = (df['pnl'] > 0).sum()
    losing_bars = (df['pnl'] < 0).sum()

    analysis['summary'] = {
        'total_bars': len(df),
        'total_trades': int(total_trades),
        'total_pnl': float(total_pnl),
        'final_equity': float(df['equity'].iloc[-1]),
        'final_return_pct': float((df['equity'].iloc[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100),
        'max_drawdown_pct': float(df['drawdown'].max()),
        'winning_bars': int(winning_bars),
        'losing_bars': int(losing_bars),
        'win_rate_bars': float(winning_bars / (winning_bars + losing_bars) * 100) if (winning_bars + losing_bars) > 0 else 0,
    }

    # Monthly analysis
    monthly_stats = df.groupby('month').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()],
        'equity': 'last',
        'drawdown': 'max',
        'position_changed': 'sum',
        'volatility_pct': 'mean',
        'vix_z': 'mean',
        'trend_z': 'mean',
    }).reset_index()

    monthly_stats.columns = ['month', 'total_pnl', 'bars', 'winning_bars', 'losing_bars',
                             'ending_equity', 'max_drawdown', 'trades', 'avg_volatility',
                             'avg_vix_z', 'avg_trend_z']

    for _, row in monthly_stats.iterrows():
        month_str = str(row['month'])
        analysis['monthly_analysis'][month_str] = {
            'total_pnl': float(row['total_pnl']),
            'bars': int(row['bars']),
            'winning_bars': int(row['winning_bars']),
            'losing_bars': int(row['losing_bars']),
            'ending_equity': float(row['ending_equity']),
            'max_drawdown_pct': float(row['max_drawdown']),
            'trades': int(row['trades']),
            'avg_volatility': float(row['avg_volatility']),
            'avg_vix_z': float(row['avg_vix_z']),
            'avg_trend_z': float(row['avg_trend_z']),
            'profitable': row['total_pnl'] > 0,
        }

    # Find worst losing periods (consecutive losses)
    df['is_losing'] = df['pnl'] < 0
    df['loss_streak_id'] = (df['is_losing'] != df['is_losing'].shift()).cumsum()

    loss_streaks = df[df['is_losing']].groupby('loss_streak_id').agg({
        'pnl': 'sum',
        'timestamp': ['first', 'last'],
        'equity': ['first', 'last'],
    }).reset_index()

    if len(loss_streaks) > 0:
        loss_streaks.columns = ['streak_id', 'total_loss', 'start', 'end', 'equity_start', 'equity_end']
        loss_streaks = loss_streaks.sort_values('total_loss').head(10)  # Top 10 worst streaks

        for _, streak in loss_streaks.iterrows():
            analysis['worst_periods'].append({
                'start': str(streak['start']),
                'end': str(streak['end']),
                'total_loss': float(streak['total_loss']),
                'equity_start': float(streak['equity_start']),
                'equity_end': float(streak['equity_end']),
                'loss_pct': float((streak['equity_start'] - streak['equity_end']) / streak['equity_start'] * 100),
            })

    # Find specific big single-bar losses
    big_losses = df[df['pnl'] < df['pnl'].quantile(0.01)].copy()  # Bottom 1%
    big_losses_list = []
    for _, row in big_losses.iterrows():
        big_losses_list.append({
            'timestamp': str(row['timestamp']),
            'pnl': float(row['pnl']),
            'position': int(row['position']),
            'market_return': float(row['market_return']),
            'volatility_pct': float(row['volatility_pct']),
            'trend_z': float(row['trend_z']),
            'vix_z': float(row['vix_z']),
        })
    analysis['biggest_single_losses'] = sorted(big_losses_list, key=lambda x: x['pnl'])[:20]

    # Analyze conditions when losing vs winning
    winning_df = df[df['pnl'] > 0]
    losing_df = df[df['pnl'] < 0]

    if len(winning_df) > 0 and len(losing_df) > 0:
        analysis['win_vs_loss_conditions'] = {
            'volatility': {
                'winning_avg': float(winning_df['volatility_pct'].mean()),
                'losing_avg': float(losing_df['volatility_pct'].mean()),
                'difference': float(losing_df['volatility_pct'].mean() - winning_df['volatility_pct'].mean()),
            },
            'trend_z': {
                'winning_avg': float(winning_df['trend_z'].mean()),
                'losing_avg': float(losing_df['trend_z'].mean()),
                'difference': float(losing_df['trend_z'].mean() - winning_df['trend_z'].mean()),
            },
            'vix_z': {
                'winning_avg': float(winning_df['vix_z'].mean()),
                'losing_avg': float(losing_df['vix_z'].mean()),
                'difference': float(losing_df['vix_z'].mean() - winning_df['vix_z'].mean()),
            },
            'rsi_9': {
                'winning_avg': float(winning_df['rsi_9'].mean()),
                'losing_avg': float(losing_df['rsi_9'].mean()),
                'difference': float(losing_df['rsi_9'].mean() - winning_df['rsi_9'].mean()),
            },
        }

    # Analyze by position type
    long_df = df[df['position'] == 1]
    short_df = df[df['position'] == -1]
    flat_df = df[df['position'] == 0]

    analysis['position_analysis'] = {
        'long': {
            'bars': int(len(long_df)),
            'total_pnl': float(long_df['pnl'].sum()),
            'avg_pnl': float(long_df['pnl'].mean()) if len(long_df) > 0 else 0,
            'win_rate': float((long_df['pnl'] > 0).mean() * 100) if len(long_df) > 0 else 0,
        },
        'short': {
            'bars': int(len(short_df)),
            'total_pnl': float(short_df['pnl'].sum()),
            'avg_pnl': float(short_df['pnl'].mean()) if len(short_df) > 0 else 0,
            'win_rate': float((short_df['pnl'] > 0).mean() * 100) if len(short_df) > 0 else 0,
        },
        'flat': {
            'bars': int(len(flat_df)),
            'total_pnl': float(flat_df['pnl'].sum()),
        },
    }

    # Find weeks with significant losses
    weekly_pnl = df.groupby(df['timestamp'].dt.to_period('W'))['pnl'].sum().reset_index()
    weekly_pnl.columns = ['week', 'pnl']
    worst_weeks = weekly_pnl.nsmallest(10, 'pnl')

    analysis['worst_weeks'] = [
        {'week': str(row['week']), 'pnl': float(row['pnl'])}
        for _, row in worst_weeks.iterrows()
    ]

    # Drawdown analysis
    df['in_drawdown'] = df['drawdown'] > 0
    df['drawdown_id'] = (df['in_drawdown'] != df['in_drawdown'].shift()).cumsum()

    drawdowns = df[df['in_drawdown']].groupby('drawdown_id').agg({
        'drawdown': 'max',
        'timestamp': ['first', 'last'],
        'pnl': 'sum',
    }).reset_index()

    if len(drawdowns) > 0:
        drawdowns.columns = ['dd_id', 'max_drawdown', 'start', 'end', 'pnl_during']
        drawdowns = drawdowns.nlargest(5, 'max_drawdown')

        analysis['major_drawdowns'] = [
            {
                'start': str(row['start']),
                'end': str(row['end']),
                'max_drawdown_pct': float(row['max_drawdown']),
                'pnl_during': float(row['pnl_during']),
            }
            for _, row in drawdowns.iterrows()
        ]

    return analysis, df


def generate_report(analysis):
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("2025 LOSS ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    s = analysis['summary']
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Bars Analyzed:     {s['total_bars']:,}")
    lines.append(f"Total Trades:            {s['total_trades']:,}")
    lines.append(f"Final Equity:            ${s['final_equity']:,.2f}")
    lines.append(f"Total Return:            {s['final_return_pct']:.2f}%")
    lines.append(f"Max Drawdown:            {s['max_drawdown_pct']:.2f}%")
    lines.append(f"Win Rate (bars):         {s['win_rate_bars']:.1f}%")
    lines.append("")

    # Monthly breakdown
    lines.append("MONTHLY BREAKDOWN")
    lines.append("-" * 40)
    lines.append(f"{'Month':<12} {'PnL':>12} {'Trades':>8} {'DD%':>8} {'Status':<12}")

    for month, data in sorted(analysis['monthly_analysis'].items()):
        status = "PROFIT" if data['profitable'] else "LOSS"
        lines.append(f"{month:<12} ${data['total_pnl']:>10,.2f} {data['trades']:>8} {data['max_drawdown_pct']:>7.1f}% {status:<12}")

    # Find losing months
    losing_months = [m for m, d in analysis['monthly_analysis'].items() if not d['profitable']]
    lines.append("")
    lines.append(f"Losing months: {', '.join(losing_months) if losing_months else 'None'}")
    lines.append("")

    # Worst periods
    lines.append("WORST LOSS PERIODS (Consecutive Losses)")
    lines.append("-" * 40)
    for i, period in enumerate(analysis['worst_periods'][:5], 1):
        lines.append(f"{i}. {period['start']} to {period['end']}")
        lines.append(f"   Loss: ${period['total_loss']:.2f} ({period['loss_pct']:.2f}%)")
    lines.append("")

    # Worst weeks
    lines.append("WORST WEEKS")
    lines.append("-" * 40)
    for i, week in enumerate(analysis['worst_weeks'][:5], 1):
        lines.append(f"{i}. {week['week']}: ${week['pnl']:.2f}")
    lines.append("")

    # Major drawdowns
    if 'major_drawdowns' in analysis:
        lines.append("MAJOR DRAWDOWNS")
        lines.append("-" * 40)
        for i, dd in enumerate(analysis['major_drawdowns'], 1):
            lines.append(f"{i}. {dd['start']} to {dd['end']}")
            lines.append(f"   Max Drawdown: {dd['max_drawdown_pct']:.2f}%, PnL: ${dd['pnl_during']:.2f}")
        lines.append("")

    # Position analysis
    lines.append("POSITION TYPE ANALYSIS")
    lines.append("-" * 40)
    pa = analysis['position_analysis']
    lines.append(f"LONG:  {pa['long']['bars']:>6} bars, PnL: ${pa['long']['total_pnl']:>10,.2f}, Win Rate: {pa['long']['win_rate']:.1f}%")
    lines.append(f"SHORT: {pa['short']['bars']:>6} bars, PnL: ${pa['short']['total_pnl']:>10,.2f}, Win Rate: {pa['short']['win_rate']:.1f}%")
    lines.append(f"FLAT:  {pa['flat']['bars']:>6} bars, PnL: ${pa['flat']['total_pnl']:>10,.2f}")
    lines.append("")

    # Market conditions analysis
    lines.append("MARKET CONDITIONS: WINNING vs LOSING BARS")
    lines.append("-" * 40)
    cond = analysis['win_vs_loss_conditions']
    for feature, data in cond.items():
        lines.append(f"{feature:>15}: Win={data['winning_avg']:>8.3f}, Loss={data['losing_avg']:>8.3f}, Diff={data['difference']:>8.3f}")
    lines.append("")

    # Biggest single losses
    lines.append("BIGGEST SINGLE-BAR LOSSES (Top 10)")
    lines.append("-" * 40)
    for i, loss in enumerate(analysis['biggest_single_losses'][:10], 1):
        lines.append(f"{i}. {loss['timestamp']}")
        lines.append(f"   PnL: ${loss['pnl']:.2f}, Position: {loss['position']}, Market Return: {loss['market_return']*100:.4f}%")
        lines.append(f"   Volatility: {loss['volatility_pct']:.2f}%, VIX_z: {loss['vix_z']:.2f}, Trend_z: {loss['trend_z']:.2f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("2025 LOSS ANALYSIS")
    print("=" * 60)

    # Create output directory
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully")

    # Load data
    df = load_data()
    norm_stats = load_norm_stats()

    # Run simulation
    print("\nRunning simulation through 2025 data...")
    results = run_simulation(model, df, norm_stats)
    print(f"Simulation complete: {len(results)} bars processed")

    # Analyze losses
    print("\nAnalyzing losses...")
    analysis, results_df = analyze_losses(results)

    # Generate report
    report = generate_report(analysis)
    print("\n" + report)

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nJSON analysis saved to: {OUTPUT_PATH}")

    with open(OUTPUT_TXT_PATH, 'w') as f:
        f.write(report)
    print(f"Text report saved to: {OUTPUT_TXT_PATH}")

    # Save equity curve CSV for further analysis
    equity_csv_path = PROJECT_ROOT / "results" / "2025_equity_curve.csv"
    results_df[['timestamp', 'equity', 'pnl', 'position', 'drawdown', 'volatility_pct', 'trend_z', 'vix_z']].to_csv(
        equity_csv_path, index=False
    )
    print(f"Equity curve saved to: {equity_csv_path}")

    return analysis


if __name__ == "__main__":
    main()
