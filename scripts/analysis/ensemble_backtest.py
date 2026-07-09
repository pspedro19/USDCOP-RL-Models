#!/usr/bin/env python3
"""
V22 P1: Ensemble Backtest with Kelly Sizing
=============================================
Runs L4 backtest using ensemble voting across all 5 seed models
with Half-Kelly position sizing.

Usage:
    python scripts/ensemble_backtest.py --model-dir models/ppo_v22_production
    python scripts/ensemble_backtest.py --model-dir models/ppo_v22_production --use-lstm
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.pipeline_config import load_pipeline_config
from src.inference.ensemble_predictor import load_ensemble_from_multi_seed

logger = logging.getLogger(__name__)


def kelly_fraction(win_rate: float, rr_ratio: float = 1.0, fraction: float = 0.5) -> float:
    """
    Calculate Kelly fraction for position sizing.

    Args:
        win_rate: Historical win rate (0-1)
        rr_ratio: Reward-to-risk ratio (avg_win / avg_loss)
        fraction: Kelly fraction (0.5 = half-Kelly)

    Returns:
        Optimal fraction of capital to risk
    """
    q = 1 - win_rate
    kelly = (win_rate * rr_ratio - q) / rr_ratio
    return max(0, fraction * kelly)


def run_ensemble_backtest(
    model_dir: Path,
    use_lstm: bool = False,
    dataset_path: str = None,
    norm_stats_path: str = None,
):
    """Run ensemble backtest with Kelly sizing."""
    config = load_pipeline_config()

    # Load dataset
    if dataset_path is None:
        project_root = Path(__file__).parent.parent
        dataset_dir = project_root / config.paths.l2_output_dir
        test_file = list(dataset_dir.glob("*_test.csv"))
        if not test_file:
            test_file = list(dataset_dir.glob("*_test.parquet"))
        if not test_file:
            logger.error("No test dataset found")
            return None
        dataset_path = test_file[0]

    logger.info(f"Loading test dataset: {dataset_path}")
    if str(dataset_path).endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

    # Load norm stats
    if norm_stats_path is None:
        project_root = Path(__file__).parent.parent
        dataset_dir = project_root / config.paths.l2_output_dir
        stats_files = list(dataset_dir.glob("*norm_stats.json"))
        if stats_files:
            norm_stats_path = stats_files[0]

    norm_stats = {}
    if norm_stats_path and Path(norm_stats_path).exists():
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

    # Load ensemble
    ensemble = load_ensemble_from_multi_seed(
        base_dir=Path(model_dir),
        use_lstm=use_lstm,
        min_consensus=config.ensemble.min_consensus,
        action_type=config.action_type,
        threshold_long=config.backtest.threshold_long,
        threshold_short=config.backtest.threshold_short,
    )

    logger.info(f"Ensemble loaded: {ensemble.n_models} models")

    # Get feature columns
    market_features = [f.name for f in config.get_market_features()]
    feature_cols = [c for c in market_features if c in df.columns]

    # Position sizing config
    ps = config.position_sizing
    base_kelly = ps.base_fraction

    # Run backtest
    capital = config.backtest.initial_capital
    position = 0
    entry_price = 0.0
    equity_curve = [capital]
    trades = []
    trade_pnl = 0.0

    obs_dim = config.get_observation_dim()

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Build observation (simplified)
        obs = np.zeros(obs_dim, dtype=np.float32)
        for j, col in enumerate(feature_cols):
            if col in row.index:
                obs[j] = float(np.clip(row[col], -5, 5))

        # Get ensemble prediction
        action, confidence, vote_details = ensemble.predict(obs)

        # Kelly position sizing
        position_size = base_kelly
        if ps.consensus_scaling:
            position_size *= confidence  # Scale by consensus
        position_size = np.clip(position_size, ps.min_fraction, ps.max_fraction)

        # Map action to target position
        if config.action_type == "discrete":
            if action == 0:  # HOLD
                target = position
            elif action == 1:  # BUY
                target = 1
            elif action == 2:  # SELL
                target = -1
            elif action == 3:  # CLOSE
                target = 0
            else:
                target = 0
        else:
            target = action

        # Calculate PnL
        if 'raw_log_ret_5m' in row.index:
            market_return = row['raw_log_ret_5m']
        else:
            market_return = 0.0

        pnl = position * market_return * capital * position_size
        cost = 0.0

        if target != position:
            cost_rate = (config.backtest.transaction_cost_bps + config.backtest.slippage_bps) / 10000
            cost = abs(target - position) * cost_rate * capital * position_size

            if position != 0:
                trades.append({
                    'pnl': trade_pnl,
                    'bars': i,
                    'confidence': confidence,
                })
                trade_pnl = 0.0

            position = target

        trade_pnl += pnl
        capital += pnl - cost
        equity_curve.append(capital)

    # Calculate metrics
    equity = np.array(equity_curve)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-8)
    total_return = (equity[-1] / equity[0] - 1) * 100

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.maximum(peak, 1e-8)
    max_dd = float(np.max(drawdown) * 100)

    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 48)) if np.std(returns) > 0 else 0

    winning = [t for t in trades if t['pnl'] > 0]
    win_rate = len(winning) / max(len(trades), 1) * 100

    results = {
        'total_return_pct': round(total_return, 2),
        'sharpe_ratio': round(sharpe, 3),
        'max_drawdown_pct': round(max_dd, 2),
        'n_trades': len(trades),
        'win_rate_pct': round(win_rate, 1),
        'n_models': ensemble.n_models,
        'min_consensus': config.ensemble.min_consensus,
        'kelly_base': ps.base_fraction,
    }

    logger.info("=" * 60)
    logger.info("ENSEMBLE BACKTEST RESULTS")
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    # Save results
    output_path = Path(model_dir) / "ensemble_backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return results


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="V22 Ensemble Backtest")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with seed model subdirectories")
    parser.add_argument("--use-lstm", action="store_true", help="Models are RecurrentPPO")
    parser.add_argument("--dataset", type=str, help="Path to test dataset")
    parser.add_argument("--norm-stats", type=str, help="Path to norm stats JSON")

    args = parser.parse_args()

    run_ensemble_backtest(
        model_dir=Path(args.model_dir),
        use_lstm=args.use_lstm,
        dataset_path=args.dataset,
        norm_stats_path=args.norm_stats,
    )


if __name__ == "__main__":
    main()
