#!/usr/bin/env python3
"""
Regenerate L4 Backtest Trades - ONLY 2025 DATA

Filters test data to ONLY include 2025 (excludes 2026).
This ensures FloatingExperimentPanel metrics match a 2025-only backtest.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
import numpy as np
import pandas as pd

from src.evaluation.backtest_engine import BacktestEngine, BacktestConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="usdcop_trading",
        user="admin",
        password="admin123"
    )


def load_model(model_path: Path):
    from stable_baselines3 import PPO
    return PPO.load(str(model_path), device='cpu')


def load_norm_stats(norm_stats_path: Path) -> Dict:
    with open(norm_stats_path) as f:
        return json.load(f)


def get_feature_columns() -> List[str]:
    try:
        from src.config.experiment_loader import load_experiment_ssot
        ssot = load_experiment_ssot()
        return ssot.get_feature_names()[:18]
    except Exception as e:
        logger.warning(f"Could not load SSOT features: {e}")
        return [
            'log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'log_ret_1d',
            'rsi_9', 'rsi_21', 'volatility_pct', 'trend_z',
            'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
            'brent_change_1d', 'gold_change_1d', 'rate_spread_z',
            'rate_spread_change', 'usdmxn_change_1d', 'yield_curve_z'
        ]


def run_backtest(model, df: pd.DataFrame, norm_stats: Dict, config: BacktestConfig) -> tuple:
    feature_cols = get_feature_columns()
    engine = BacktestEngine(config, norm_stats, feature_cols)

    logger.info(f"Running backtest on {len(df)} bars (2025 ONLY)...")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        obs = engine.build_observation(row)
        action, _ = model.predict(obs, deterministic=True)
        action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
        engine.step(action_val, row, prev_row)

    result = engine.get_result(df)

    logger.info(f"Backtest complete (2025 ONLY):")
    logger.info(f"  Total trades: {result.total_trades}")
    logger.info(f"  Win rate: {result.win_rate_pct:.1f}%")
    logger.info(f"  Total return: {result.total_return_pct:.2f}%")
    logger.info(f"  Sharpe: {result.sharpe_annual:.2f}")
    logger.info(f"  Max DD: {result.max_drawdown_pct:.2f}%")

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
            'duration_minutes': trade.bars_held * 5,
            'exit_reason': 'signal',
            'equity_at_entry': config.initial_capital + cumulative_pnl - trade.pnl,
            'equity_at_exit': config.initial_capital + cumulative_pnl,
            'entry_confidence': 0.5,
            'exit_confidence': 0.5,
            'raw_action': 0.0,
        })

    return detailed_trades, result


def update_proposal_metrics(conn, proposal_id: str, result):
    """Update promotion_proposals with 2025-only metrics."""
    cur = conn.cursor()
    try:
        # Build new metrics JSON
        metrics = {
            "sharpe_ratio": round(result.sharpe_annual, 4),
            "max_drawdown": round(result.max_drawdown_pct / 100, 4),
            "win_rate": round(result.win_rate_pct / 100, 4),
            "profit_factor": round(result.profit_factor, 4) if hasattr(result, 'profit_factor') else 1.5,
            "total_trades": result.total_trades,
            "total_return": round(result.total_return_pct / 100, 4),
            "test_period": "2025-01-01 to 2025-12-31"
        }

        cur.execute("""
            UPDATE promotion_proposals
            SET metrics = %s::jsonb,
                updated_at = NOW()
            WHERE proposal_id = %s
        """, (json.dumps(metrics), proposal_id))

        conn.commit()
        logger.info(f"Updated proposal metrics: {metrics}")
    except Exception as e:
        logger.error(f"Failed to update proposal: {e}")
        conn.rollback()
    finally:
        cur.close()


def persist_trades(conn, proposal_id: str, model_id: str, trades: List[Dict]) -> int:
    if not trades:
        return 0

    cur = conn.cursor()

    try:
        # Delete existing trades first
        cur.execute("DELETE FROM backtest_trades WHERE proposal_id = %s", (proposal_id,))
        logger.info(f"Deleted existing trades for {proposal_id}")

        inserted = 0
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
        logger.info(f"Persisted {inserted} trades (2025 ONLY)")
        return inserted

    except Exception as e:
        logger.error(f"Failed to persist trades: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()


def main():
    proposal_id = "prop_ppo_ssot_20260203_152841"
    model_id = "ppo_ssot_20260203_152841"

    model_path = PROJECT_ROOT / "models" / model_id / "final_model.zip"
    norm_stats_path = PROJECT_ROOT / "models" / model_id / "norm_stats.json"

    # Load test data
    test_data_path = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min" / "DS_production_test.parquet"

    logger.info(f"Model: {model_path}")
    logger.info(f"Test data: {test_data_path}")

    model = load_model(model_path)
    logger.info("Model loaded")

    norm_stats = load_norm_stats(norm_stats_path)
    logger.info(f"Norm stats loaded: {len(norm_stats)} features")

    df = pd.read_parquet(test_data_path)
    logger.info(f"Full test data: {len(df)} rows, {df.index.min()} to {df.index.max()}")

    # FILTER TO 2025 ONLY
    df_2025 = df[(df.index >= '2025-01-01') & (df.index < '2026-01-01')]
    logger.info(f"2025 ONLY data: {len(df_2025)} rows, {df_2025.index.min()} to {df_2025.index.max()}")

    if len(df_2025) == 0:
        logger.error("No 2025 data found!")
        return

    config = BacktestConfig.from_ssot()

    # Run backtest on 2025 ONLY
    detailed_trades, result = run_backtest(model, df_2025, norm_stats, config)

    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS (2025 ONLY)")
    logger.info("="*60)
    logger.info(f"Trades: {result.total_trades}")
    logger.info(f"Win Rate: {result.win_rate_pct:.1f}%")
    logger.info(f"Total Return: {result.total_return_pct:.2f}%")
    logger.info(f"Sharpe: {result.sharpe_annual:.2f}")
    logger.info(f"Max DD: {result.max_drawdown_pct:.2f}%")

    # Update database
    conn = get_db_connection()

    # Update proposal metrics
    update_proposal_metrics(conn, proposal_id, result)

    # Persist new trades
    persist_trades(conn, proposal_id, model_id, detailed_trades)

    conn.close()
    logger.info("\nDone! Database updated with 2025-only results.")


if __name__ == "__main__":
    main()
