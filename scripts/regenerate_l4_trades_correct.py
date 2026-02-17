#!/usr/bin/env python3
"""
Regenerate L4 Backtest Trades - CORRECT VERSION

Uses the proper BacktestEngine from src/evaluation/backtest_engine.py
which includes:
- Stop Loss (-2.5%)
- Take Profit (+3%)
- Trailing Stop
- Correct transaction costs (2.5 bps)
- Correct action thresholds (±0.50)

This should reproduce the EXACT results from L4:
- 31 trades
- 74% win rate
- 29.7% return
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
import numpy as np
import pandas as pd

# Import the CORRECT BacktestEngine
from src.evaluation.backtest_engine import BacktestEngine, BacktestConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="usdcop_trading",
        user="admin",
        password="admin123"
    )


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
        return ssot.get_feature_names()[:18]  # Market features only
    except Exception as e:
        logger.warning(f"Could not load SSOT features: {e}")
        # Fallback to known features
        return [
            'log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'log_ret_1d',
            'rsi_9', 'rsi_21', 'volatility_pct', 'trend_z',
            'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
            'brent_change_1d', 'gold_change_1d', 'rate_spread_z',
            'rate_spread_change', 'usdmxn_change_1d', 'yield_curve_z'
        ]


def run_backtest(model, df: pd.DataFrame, norm_stats: Dict, config: BacktestConfig) -> List[Dict]:
    """
    Run backtest using the correct BacktestEngine.

    Returns list of detailed trades for persistence.
    """
    feature_cols = get_feature_columns()
    engine = BacktestEngine(config, norm_stats, feature_cols)

    logger.info(f"Running backtest on {len(df)} bars...")
    logger.info(f"Config: threshold_long={config.threshold_long}, threshold_short={config.threshold_short}")
    logger.info(f"Config: stop_loss={config.stop_loss_pct}, take_profit={config.take_profit_pct}")
    logger.info(f"Config: trailing_stop={config.trailing_stop_enabled}")

    # Run through each bar
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Build observation
        obs = engine.build_observation(row)

        # Get model action
        action, _ = model.predict(obs, deterministic=True)
        action_val = float(action[0]) if hasattr(action, '__len__') else float(action)

        # Step the engine
        engine.step(action_val, row, prev_row)

    # Get results
    result = engine.get_result(df)

    logger.info(f"Backtest complete:")
    logger.info(f"  Total trades: {result.total_trades}")
    logger.info(f"  Win rate: {result.win_rate_pct:.1f}%")
    logger.info(f"  Total return: {result.total_return_pct:.2f}%")
    logger.info(f"  Sharpe: {result.sharpe_annual:.2f}")
    logger.info(f"  Max DD: {result.max_drawdown_pct:.2f}%")

    # Convert trades to detailed format for persistence
    detailed_trades = []
    cumulative_pnl = 0.0

    for i, trade in enumerate(result.trades):
        cumulative_pnl += trade.pnl

        detailed_trades.append({
            'trade_id': i + 1,
            'timestamp': trade.entry_time,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'side': trade.direction,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'pnl_usd': trade.pnl,
            'pnl_percent': trade.pnl_pct,
            'status': 'closed',
            'duration_minutes': trade.bars_held * 5,  # 5-min bars
            'exit_reason': 'signal',  # Could be more specific
            'equity_at_entry': config.initial_capital + cumulative_pnl - trade.pnl,
            'equity_at_exit': config.initial_capital + cumulative_pnl,
            'entry_confidence': 0.5,
            'exit_confidence': 0.5,
            'raw_action': 0.0,
        })

    return detailed_trades, result


def persist_trades(conn, proposal_id: str, model_id: str, trades: List[Dict]) -> int:
    """Persist trades to database."""
    if not trades:
        return 0

    cur = conn.cursor()
    inserted = 0

    try:
        for trade in trades:
            cur.execute("""
                INSERT INTO backtest_trades (
                    proposal_id, trade_id, model_id,
                    timestamp, entry_time, exit_time,
                    side, entry_price, exit_price,
                    pnl, pnl_usd, pnl_percent,
                    status, duration_minutes, exit_reason,
                    equity_at_entry, equity_at_exit,
                    entry_confidence, exit_confidence, raw_action
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT DO NOTHING
            """, (
                proposal_id,
                trade.get('trade_id'),
                model_id,
                trade.get('timestamp'),
                trade.get('entry_time'),
                trade.get('exit_time'),
                trade.get('side'),
                trade.get('entry_price'),
                trade.get('exit_price'),
                trade.get('pnl'),
                trade.get('pnl_usd'),
                trade.get('pnl_percent'),
                trade.get('status', 'closed'),
                trade.get('duration_minutes'),
                trade.get('exit_reason'),
                trade.get('equity_at_entry'),
                trade.get('equity_at_exit'),
                trade.get('entry_confidence'),
                trade.get('exit_confidence'),
                trade.get('raw_action'),
            ))
            inserted += 1

        conn.commit()
        logger.info(f"Persisted {inserted} trades for proposal {proposal_id}")
        return inserted

    except Exception as e:
        logger.error(f"Failed to persist trades: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()


def main():
    """Main function."""
    # Configuration
    proposal_id = "prop_ppo_ssot_20260203_152841"
    model_id = "ppo_ssot_20260203_152841"

    model_path = PROJECT_ROOT / "models" / model_id / "final_model.zip"
    norm_stats_path = PROJECT_ROOT / "models" / model_id / "norm_stats.json"

    # Find test data - try multiple paths
    test_data_paths = [
        PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / "DS_production_test.parquet",
        PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / "DS_v3_close_only_test.parquet",
        PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / "DS_default_test.parquet",
    ]

    test_data_path = None
    for path in test_data_paths:
        if path.exists():
            test_data_path = path
            break

    if not test_data_path:
        logger.error("No test data found!")
        return

    logger.info(f"Model: {model_path}")
    logger.info(f"Norm stats: {norm_stats_path}")
    logger.info(f"Test data: {test_data_path}")

    # Load model
    model = load_model(model_path)
    logger.info("Model loaded")

    # Load norm stats
    norm_stats = load_norm_stats(norm_stats_path)
    logger.info(f"Norm stats loaded: {len(norm_stats)} features")

    # Load test data
    df = pd.read_parquet(test_data_path)
    logger.info(f"Test data loaded: {len(df)} rows")

    # Ensure we have the time index
    if 'time' in df.columns:
        df = df.set_index('time')
    elif df.index.name != 'time':
        # Use existing index
        pass

    # Load config from SSOT
    config = BacktestConfig.from_ssot()
    logger.info(f"Config loaded from SSOT")

    # Run backtest
    detailed_trades, result = run_backtest(model, df, norm_stats, config)

    # Validate results match expected
    expected_trades = 31
    expected_win_rate = 74.2
    expected_return = 29.7

    logger.info("\n" + "="*60)
    logger.info("VALIDATION")
    logger.info("="*60)
    logger.info(f"Expected trades: {expected_trades}, Got: {result.total_trades}")
    logger.info(f"Expected win rate: {expected_win_rate}%, Got: {result.win_rate_pct:.1f}%")
    logger.info(f"Expected return: {expected_return}%, Got: {result.total_return_pct:.2f}%")

    if abs(result.total_trades - expected_trades) <= 5:
        logger.info("✓ Trade count CLOSE to expected")
    else:
        logger.warning(f"✗ Trade count differs significantly ({result.total_trades} vs {expected_trades})")

    # Persist trades
    conn = get_db_connection()
    persist_trades(conn, proposal_id, model_id, detailed_trades)
    conn.close()

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
