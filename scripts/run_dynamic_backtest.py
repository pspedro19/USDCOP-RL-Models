#!/usr/bin/env python3
"""
Dynamic Backtest Runner - Professional Version

Runs a REAL backtest using the SAME BacktestEngine as L4 validation.
Loads model from lineage, filters data by date range, streams results.

Usage:
    python scripts/run_dynamic_backtest.py \
        --proposal-id prop_xxx \
        --start-date 2025-01-01 \
        --end-date 2025-12-31 \
        --output-format json

This ensures metrics match between L4 validation and dashboard backtest
when using the same date range.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="usdcop_trading",
        user="admin",
        password="admin123"
    )


def get_proposal_lineage(proposal_id: str) -> Optional[Dict]:
    """Get model lineage from promotion_proposals."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT model_id, lineage, metrics
            FROM promotion_proposals
            WHERE proposal_id = %s
        """, (proposal_id,))
        row = cur.fetchone()
        if not row:
            return None

        model_id, lineage, metrics = row
        if isinstance(lineage, str):
            lineage = json.loads(lineage)
        if isinstance(metrics, str):
            metrics = json.loads(metrics)

        return {
            "model_id": model_id,
            "lineage": lineage,
            "metrics": metrics
        }
    finally:
        cur.close()
        conn.close()


def load_model(model_path: Path):
    """Load PPO model."""
    from stable_baselines3 import PPO
    return PPO.load(str(model_path), device='cpu')


def load_norm_stats(norm_stats_path: Path) -> Dict:
    """Load normalization statistics."""
    with open(norm_stats_path) as f:
        return json.load(f)


def get_feature_columns() -> List[str]:
    """Get feature columns from SSOT."""
    try:
        from src.config.experiment_loader import load_experiment_ssot
        ssot = load_experiment_ssot()
        return ssot.get_feature_names()[:18]
    except Exception:
        return [
            'log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'log_ret_1d',
            'rsi_9', 'rsi_21', 'volatility_pct', 'trend_z',
            'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
            'brent_change_1d', 'gold_change_1d', 'rate_spread_z',
            'rate_spread_change', 'usdmxn_change_1d', 'yield_curve_z'
        ]


def load_test_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load test data filtered by date range."""
    # Try multiple data sources
    data_paths = [
        PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / "DS_production_test.parquet",
        PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / "DS_v3_close_only_test.parquet",
    ]

    df = None
    for path in data_paths:
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"Loaded data from {path}")
            break

    if df is None:
        raise FileNotFoundError("No test data found")

    # Filter by date range
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include end date

    df_filtered = df[(df.index >= start_dt) & (df.index < end_dt)]
    logger.info(f"Filtered to {len(df_filtered)} rows: {df_filtered.index.min()} to {df_filtered.index.max()}")

    return df_filtered


def run_backtest(model, df: pd.DataFrame, norm_stats: Dict) -> Dict[str, Any]:
    """Run backtest using the SAME BacktestEngine as L4."""
    from src.evaluation.backtest_engine import BacktestEngine, BacktestConfig

    feature_cols = get_feature_columns()
    config = BacktestConfig.from_ssot()
    engine = BacktestEngine(config, norm_stats, feature_cols)

    logger.info(f"Running backtest on {len(df)} bars...")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Config: threshold_long={config.threshold_long}, threshold_short={config.threshold_short}")
    logger.info(f"Config: stop_loss={config.stop_loss_pct}, take_profit={config.take_profit_pct}")

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        obs = engine.build_observation(row)
        action, _ = model.predict(obs, deterministic=True)
        action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
        engine.step(action_val, row, prev_row)

    result = engine.get_result(df)

    # Build trades list
    trades = []
    cumulative_pnl = 0.0

    for i, trade in enumerate(result.trades):
        cumulative_pnl += trade.pnl
        trades.append({
            'trade_id': i + 1,
            'entry_time': trade.entry_time.isoformat() if hasattr(trade.entry_time, 'isoformat') else str(trade.entry_time),
            'exit_time': trade.exit_time.isoformat() if hasattr(trade.exit_time, 'isoformat') else str(trade.exit_time),
            'side': trade.direction,
            'entry_price': float(trade.entry_price),
            'exit_price': float(trade.exit_price),
            'pnl': float(trade.pnl),
            'pnl_usd': float(trade.pnl),
            'pnl_percent': float(trade.pnl_pct),
            'status': 'closed',
            'duration_minutes': int(trade.bars_held * 5),
            'exit_reason': 'signal',
            'equity_at_entry': float(config.initial_capital + cumulative_pnl - trade.pnl),
            'equity_at_exit': float(config.initial_capital + cumulative_pnl),
        })

    return {
        'success': True,
        'source': 'real_backtest',
        'trade_count': result.total_trades,
        'trades': trades,
        'summary': {
            'total_trades': result.total_trades,
            'winning_trades': int(result.win_rate_pct * result.total_trades / 100) if result.total_trades > 0 else 0,
            'losing_trades': result.total_trades - int(result.win_rate_pct * result.total_trades / 100) if result.total_trades > 0 else 0,
            'win_rate': round(result.win_rate_pct, 2),
            'total_pnl': round(result.total_return_pct * config.initial_capital / 100, 2),
            'total_return_pct': round(result.total_return_pct, 2),
            'max_drawdown_pct': round(result.max_drawdown_pct, 2),
            'sharpe_ratio': round(result.sharpe_annual, 2),
        },
        'config': {
            'threshold_long': config.threshold_long,
            'threshold_short': config.threshold_short,
            'stop_loss_pct': config.stop_loss_pct,
            'take_profit_pct': config.take_profit_pct,
            'trailing_stop_enabled': config.trailing_stop_enabled,
            'spread_bps': config.spread_bps,
            'slippage_bps': config.slippage_bps,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Run dynamic backtest')
    parser.add_argument('--proposal-id', required=True, help='Proposal ID')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-format', default='json', choices=['json', 'summary'])
    args = parser.parse_args()

    # Get proposal lineage
    proposal = get_proposal_lineage(args.proposal_id)
    if not proposal:
        print(json.dumps({'error': f'Proposal {args.proposal_id} not found'}))
        sys.exit(1)

    lineage = proposal['lineage']
    model_id = proposal['model_id']

    # Resolve paths
    model_path = lineage.get('modelPath') or lineage.get('model_path')
    norm_stats_path = lineage.get('normStatsPath') or lineage.get('norm_stats_path')

    if not model_path:
        # Fallback to standard path
        model_path = f"models/{model_id}/final_model.zip"
    if not norm_stats_path:
        norm_stats_path = f"models/{model_id}/norm_stats.json"

    model_path = PROJECT_ROOT / model_path
    norm_stats_path = PROJECT_ROOT / norm_stats_path

    logger.info(f"Proposal: {args.proposal_id}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Norm stats: {norm_stats_path}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")

    # Load model and norm stats
    if not model_path.exists():
        print(json.dumps({'error': f'Model not found: {model_path}'}))
        sys.exit(1)

    if not norm_stats_path.exists():
        print(json.dumps({'error': f'Norm stats not found: {norm_stats_path}'}))
        sys.exit(1)

    model = load_model(model_path)
    norm_stats = load_norm_stats(norm_stats_path)

    # Load and filter data
    df = load_test_data(args.start_date, args.end_date)

    if len(df) == 0:
        print(json.dumps({'error': 'No data in specified date range'}))
        sys.exit(1)

    # Run backtest
    result = run_backtest(model, df, norm_stats)

    # Add metadata
    result['proposal_id'] = args.proposal_id
    result['model_id'] = model_id
    result['date_range'] = {
        'start': args.start_date,
        'end': args.end_date
    }

    # Output
    if args.output_format == 'summary':
        print(f"Trades: {result['summary']['total_trades']}")
        print(f"Win Rate: {result['summary']['win_rate']}%")
        print(f"Return: {result['summary']['total_return_pct']}%")
        print(f"Sharpe: {result['summary']['sharpe_ratio']}")
        print(f"Max DD: {result['summary']['max_drawdown_pct']}%")
    else:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
