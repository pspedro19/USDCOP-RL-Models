#!/usr/bin/env python3
"""
Full Pipeline Runner: L2 → L3 → L4
===================================

Executes the complete RL training pipeline with SSOT configuration.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --timesteps 100000
    python scripts/run_full_pipeline.py --skip-l2  # Skip dataset generation
    python scripts/run_full_pipeline.py --skip-l3  # Skip training (eval only)

Author: Trading Team
Date: 2026-02-02
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure logs directory exists BEFORE configuring logging
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


def run_l2_dataset_builder() -> Optional[Path]:
    """
    L2: Build RL training dataset.

    Returns:
        Path to generated dataset or None on failure
    """
    logger.info("=" * 60)
    logger.info("L2: DATASET BUILDER")
    logger.info("=" * 60)

    try:
        # Import the dataset builder module
        sys.path.insert(0, str(PROJECT_ROOT / "data" / "pipeline" / "06_rl_dataset_builder"))

        # Run the 5min dataset builder
        dataset_script = PROJECT_ROOT / "data" / "pipeline" / "06_rl_dataset_builder" / "01_build_5min_datasets.py"

        if not dataset_script.exists():
            logger.error(f"Dataset builder script not found: {dataset_script}")
            return None

        logger.info(f"Running: {dataset_script}")

        # Execute the script
        import subprocess
        result = subprocess.run(
            [sys.executable, str(dataset_script)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=1800  # 30 minutes timeout
        )

        if result.returncode != 0:
            logger.error(f"L2 failed:\n{result.stderr}")
            return None

        logger.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

        # Find the generated dataset (DS3_MACRO_CORE is recommended)
        output_dir = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "datasets_5min"

        # Look for DS3 first, then any dataset
        for pattern in ["*DS3_MACRO_CORE*.csv", "*DS1*.csv", "*.csv"]:
            datasets = list(output_dir.glob(pattern))
            if datasets:
                dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)
                logger.info(f"L2 Complete: {dataset_path}")
                return dataset_path

        # Check for parquet files
        for pattern in ["*.parquet"]:
            datasets = list(output_dir.glob(pattern))
            if datasets:
                dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)
                logger.info(f"L2 Complete: {dataset_path}")
                return dataset_path

        logger.error(f"No datasets found in {output_dir}")
        return None

    except Exception as e:
        logger.error(f"L2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_l3_training(dataset_path: Path, total_timesteps: int = 100_000) -> Optional[Dict[str, Any]]:
    """
    L3: Train PPO model using TrainingEngine.

    Args:
        dataset_path: Path to training dataset
        total_timesteps: Training timesteps

    Returns:
        Training result dict or None on failure
    """
    logger.info("=" * 60)
    logger.info("L3: MODEL TRAINING")
    logger.info("=" * 60)

    try:
        from src.training.engine import TrainingEngine, TrainingRequest
        from src.config import load_experiment_config

        # Load SSOT config
        config = load_experiment_config(force_reload=True)

        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Timesteps: {total_timesteps:,}")
        logger.info(f"SSOT Config:")
        logger.info(f"  gamma: {config.training.gamma}")
        logger.info(f"  batch_size: {config.training.batch_size}")
        logger.info(f"  ent_coef: {config.training.ent_coef}")
        logger.info(f"  lr_decay: {config.training.lr_decay.enabled}")
        logger.info(f"  early_stopping: {config.training.early_stopping.enabled} (patience={config.training.early_stopping.patience})")

        # Create version string
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create training request
        request = TrainingRequest(
            version=version,
            dataset_path=dataset_path,
            total_timesteps=total_timesteps,
            experiment_name=config.logging.experiment_name,
            mlflow_enabled=False,  # Disable for local run
            auto_register=False,
            dvc_enabled=False,
            lineage_enabled=False,
        )

        # Run training
        engine = TrainingEngine(project_root=PROJECT_ROOT)
        result = engine.run(request)

        if result.success:
            logger.info(f"L3 Complete:")
            logger.info(f"  Model: {result.model_path}")
            logger.info(f"  Best Reward: {result.best_mean_reward:.2f}")
            logger.info(f"  Duration: {result.training_duration_seconds/60:.1f} min")
            return result.to_dict()
        else:
            logger.error(f"L3 failed: {result.errors}")
            return None

    except Exception as e:
        logger.error(f"L3 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _find_norm_stats(dataset_path: Path) -> Optional[Path]:
    """
    Find norm_stats.json file adjacent to a dataset.

    Search order:
    1. {dataset_dir}/{dataset_stem}_norm_stats.json (e.g., DS3_MACRO_CORE_norm_stats.json)
    2. {dataset_dir}/{dataset_stem.replace('RL_', '')}_norm_stats.json
    3. {dataset_dir}/norm_stats.json

    Args:
        dataset_path: Path to the dataset file

    Returns:
        Path to norm_stats.json if found, None otherwise
    """
    dataset_dir = dataset_path.parent
    dataset_stem = dataset_path.stem

    # Pattern 1: RL_DS3_MACRO_CORE.csv -> DS3_MACRO_CORE_norm_stats.json
    if dataset_stem.startswith('RL_'):
        pattern1 = dataset_dir / f"{dataset_stem[3:]}_norm_stats.json"
    else:
        pattern1 = dataset_dir / f"{dataset_stem}_norm_stats.json"

    # Pattern 2: Direct match with dataset name
    pattern2 = dataset_dir / f"{dataset_stem}_norm_stats.json"

    # Pattern 3: Generic norm_stats.json in same directory
    pattern3 = dataset_dir / "norm_stats.json"

    # Pattern 4: In config directory
    pattern4 = PROJECT_ROOT / "config" / "norm_stats.json"

    for p in [pattern1, pattern2, pattern3, pattern4]:
        if p.exists():
            logger.info(f"[L4] Found norm_stats: {p}")
            return p

    logger.warning(f"[L4] No norm_stats found for {dataset_path}")
    return None


def _calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio."""
    excess_returns = returns - risk_free_rate
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float('inf')
    downside_std = np.std(downside_returns)
    if downside_std < 1e-10:
        return float('inf')
    return float(np.mean(excess_returns) / downside_std)


def _calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    std = np.std(returns)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess_returns) / std)


def _calculate_max_drawdown(equity_curve: np.ndarray) -> tuple:
    """Calculate maximum drawdown percentage and duration in bars."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_dd = np.max(drawdown)
    max_dd_idx = np.argmax(drawdown)

    # Find drawdown duration
    peak_idx = np.argmax(equity_curve[:max_dd_idx+1]) if max_dd_idx > 0 else 0
    return float(max_dd), int(max_dd_idx - peak_idx)


def run_l4_evaluation(
    model_path: Path,
    dataset_path: Path,
    norm_stats_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    L4: CONTINUOUS BACKTEST over entire OOS period (2025).

    Runs a sequential walk-forward backtest over ALL test data,
    simulating what would have happened if the model was deployed
    from the test_start date through test_end.

    Calculates comprehensive metrics:
    - Capital: Initial, Final, Total PnL, Total Return
    - Annualized: APR (simple & compound)
    - Risk: Max Drawdown, Sharpe, Sortino
    - Trades: Count, Win Rate, Profit Factor, Avg Duration

    Args:
        model_path: Path to trained model
        dataset_path: Path to dataset for evaluation
        norm_stats_path: Path to norm_stats.json (auto-detected if None)

    Returns:
        Comprehensive evaluation metrics dict or None on failure
    """
    logger.info("=" * 60)
    logger.info("L4: CONTINUOUS BACKTEST (Full 2025 Period)")
    logger.info("=" * 60)

    try:
        from stable_baselines3 import PPO
        from src.config import load_experiment_config

        # Load SSOT configuration
        config = load_experiment_config()

        # Find norm_stats if not provided
        if norm_stats_path is None:
            norm_stats_path = _find_norm_stats(dataset_path)

        if norm_stats_path is None or not norm_stats_path.exists():
            logger.error("norm_stats.json not found! Run L2 first to generate normalization stats.")
            return None

        logger.info(f"Model: {model_path}")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Norm Stats: {norm_stats_path}")

        # Load model
        model = PPO.load(str(model_path), device='cpu')

        # Load norm_stats
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

        # Load dataset and filter to test period
        if str(dataset_path).endswith('.parquet'):
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path)

        # Ensure timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("Dataset must have 'timestamp' or 'datetime' column")

        logger.info(f"Full dataset: {len(df):,} rows")

        # Get test period from SSOT
        try:
            TEST_START = config.pipeline.date_ranges.test_start if hasattr(config.pipeline, 'date_ranges') else '2025-01-01'
            TEST_END = config.pipeline.date_ranges.test_end if hasattr(config.pipeline, 'date_ranges') else '2025-12-31'
        except Exception:
            TEST_START = '2025-01-01'
            TEST_END = '2025-12-31'

        test_df = df[(df['timestamp'] >= TEST_START) & (df['timestamp'] <= TEST_END)].reset_index(drop=True)

        if len(test_df) == 0:
            logger.warning(f"No test data in {TEST_START} to {TEST_END}, using last 15% of dataset")
            test_start_idx = int(len(df) * 0.85)
            test_df = df.iloc[test_start_idx:].reset_index(drop=True)

        logger.info(f"Test period: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")
        logger.info(f"Test bars: {len(test_df):,}")

        # Feature columns (must match training) - EXP-B-001: 18 market features
        # Read from SSOT to ensure consistency
        feature_cols = list(config.market_features) if hasattr(config, 'market_features') else [
            'log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'log_ret_1d',  # 4 returns
            'rsi_9', 'rsi_21',                                        # 2 RSI
            'volatility_pct', 'trend_z',                              # 2 technical
            'dxy_z', 'dxy_change_1d',                                 # 2 DXY
            'vix_z', 'embi_z',                                        # 2 risk
            'brent_change_1d',                                         # 1 oil
            'rate_spread_z', 'rate_spread_change',                    # 2 rates
            'usdmxn_change_1d',                                        # 1 peer
            'yield_curve_z', 'gold_change_1d',                        # 2 macro NEW
        ]
        logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

        # Configuration from SSOT - OPTIMIZED FOR MEXC/BINANCE USDT/COP
        # MEXC: 0% maker, 0.05% taker = 5bps round-trip total
        INITIAL_CAPITAL = 10_000.0
        TRANSACTION_COST_BPS = float(getattr(config.environment, 'transaction_cost_bps', 2.5) if hasattr(config, 'environment') else 2.5)
        SLIPPAGE_BPS = float(getattr(config.environment, 'slippage_bps', 2.5) if hasattr(config, 'environment') else 2.5)

        try:
            THRESHOLD_LONG = float(config.environment.thresholds.long) if hasattr(config.environment, 'thresholds') else 0.40
            THRESHOLD_SHORT = float(config.environment.thresholds.short) if hasattr(config.environment, 'thresholds') else -0.40
            MAX_POSITION_HOLDING = int(config.environment.max_position_holding) if hasattr(config.environment, 'max_position_holding') else 576
        except Exception:
            THRESHOLD_LONG = 0.40
            THRESHOLD_SHORT = -0.40
            MAX_POSITION_HOLDING = 576

        total_cost_rate = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10_000

        logger.info(f"Config: thresholds={THRESHOLD_LONG}/{THRESHOLD_SHORT}, max_hold={MAX_POSITION_HOLDING}, costs={TRANSACTION_COST_BPS}bps")

        # =====================================================================
        # CONTINUOUS BACKTEST LOOP
        # =====================================================================
        capital = INITIAL_CAPITAL
        position = 0  # -1=SHORT, 0=FLAT, 1=LONG
        entry_bar = 0
        entry_price = 0.0  # Track entry price for proper state feature

        # Tracking arrays
        equity_curve = [capital]
        returns_per_bar = []
        positions_held = []
        trades = []
        trade_pnl_accumulator = 0.0  # Track PnL for current trade
        actions_taken = {'long': 0, 'hold': 0, 'short': 0}

        logger.info("Running continuous backtest...")

        # Get observation dim from SSOT
        obs_dim = config.pipeline.observation_dim if hasattr(config.pipeline, 'observation_dim') else 20
        n_market_features = len(feature_cols)
        logger.info(f"Observation dim: {obs_dim} ({n_market_features} market + 2 state)")

        for i in range(1, len(test_df)):
            # Build observation vector (market features + 2 state features)
            obs = np.zeros(obs_dim, dtype=np.float32)

            # Market features (normalized)
            for j, col in enumerate(feature_cols):
                if col not in test_df.columns:
                    continue
                value = test_df.iloc[i][col]

                # Skip normalization for features already z-scored
                if col.endswith('_z'):
                    obs[j] = np.clip(value, -5, 5)
                elif col in norm_stats and '_meta' not in col:
                    stats = norm_stats[col]
                    mean = stats.get('mean', 0)
                    std = stats.get('std', 1)
                    if std < 1e-8:
                        std = 1.0
                    obs[j] = np.clip((value - mean) / std, -5, 5)
                else:
                    obs[j] = np.clip(value, -5, 5)

            # State features - SSOT compliant (matches TradingEnv)
            # Position at index n_market_features, unrealized_pnl at n_market_features+1
            pos_idx = n_market_features
            pnl_idx = n_market_features + 1
            obs[pos_idx] = float(position)  # Current position (-1, 0, 1)

            # SSOT: unrealized_pnl = (current_price - entry_price) / entry_price * position
            if position != 0 and entry_price > 0:
                current_price = test_df.iloc[i]['close']
                unrealized_pnl_pct = (current_price - entry_price) / entry_price * position
                obs[pnl_idx] = np.clip(unrealized_pnl_pct, -1.0, 1.0)
            else:
                obs[pnl_idx] = 0.0

            # Handle NaN
            obs = np.nan_to_num(obs, nan=0.0)

            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            action_value = float(action[0]) if hasattr(action, '__len__') else float(action)

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

            # Force close if max position holding exceeded
            if position != 0 and (i - entry_bar) >= MAX_POSITION_HOLDING:
                target_action = 0

            # P0.3 FIX: Calculate market return using RAW returns for accurate PnL
            # raw_log_ret_5m is the actual price movement, not z-scored
            if 'raw_log_ret_5m' in test_df.columns:
                market_return = test_df.iloc[i]['raw_log_ret_5m']
            else:
                # Fallback: calculate from close prices directly
                current_close = test_df.iloc[i]['close']
                prev_close = test_df.iloc[i-1]['close']
                market_return = np.log(current_close / prev_close) if prev_close > 0 else 0.0

            # PnL from current position
            position_pnl = position * market_return * capital

            # Transaction cost if position changes
            trade_cost = 0.0
            if target_action != position:
                change_magnitude = abs(target_action - position)
                trade_cost = change_magnitude * total_cost_rate * capital

                # Record completed trade with accumulated PnL
                if position != 0:
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'bars_held': i - entry_bar,
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'pnl': trade_pnl_accumulator,  # P0.3 FIX: Use accumulated PnL
                        'entry_time': str(test_df.iloc[entry_bar]['timestamp']),
                        'exit_time': str(test_df.iloc[i]['timestamp']),
                    })

                # Update position state
                position = target_action
                entry_bar = i
                entry_price = test_df.iloc[i]['close']
                trade_pnl_accumulator = 0.0  # Reset for new trade
            else:
                # Accumulate PnL while holding position
                trade_pnl_accumulator += position_pnl

            # Update capital
            net_pnl = position_pnl - trade_cost
            capital += net_pnl

            equity_curve.append(capital)
            bar_return = net_pnl / equity_curve[-2] if equity_curve[-2] > 0 else 0
            returns_per_bar.append(bar_return)
            positions_held.append(position)

        # Close final position if any
        if position != 0:
            trades.append({
                'entry_bar': entry_bar,
                'exit_bar': len(test_df) - 1,
                'bars_held': len(test_df) - 1 - entry_bar,
                'direction': 'LONG' if position == 1 else 'SHORT',
                'pnl': trade_pnl_accumulator,  # P0.3 FIX: Use accumulated PnL
                'entry_time': str(test_df.iloc[entry_bar]['timestamp']),
                'exit_time': str(test_df.iloc[-1]['timestamp']),
            })

        # =====================================================================
        # CALCULATE COMPREHENSIVE METRICS
        # =====================================================================
        equity_curve = np.array(equity_curve)
        returns = np.array(returns_per_bar)

        # Capital metrics
        total_pnl = capital - INITIAL_CAPITAL
        total_return_pct = (capital / INITIAL_CAPITAL - 1) * 100

        # Drawdown
        max_dd, dd_duration = _calculate_max_drawdown(equity_curve)

        # Trade statistics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_trade_pnl = np.mean([t['pnl'] for t in trades]) if trades else 0
        avg_bars_held = np.mean([t['bars_held'] for t in trades]) if trades else 0
        avg_trade_hours = avg_bars_held * 5 / 60  # 5 min bars

        # Daily returns for ratio calculation
        bars_per_day = 144  # 12h trading day
        daily_returns = []
        for d in range(0, len(returns), bars_per_day):
            chunk = returns[d:d+bars_per_day]
            if len(chunk) > 0:
                daily_returns.append(np.sum(chunk))
        daily_returns = np.array(daily_returns) if daily_returns else np.array([0])

        # Annualized ratios
        sharpe_annual = _calculate_sharpe(daily_returns) * np.sqrt(252)
        sortino_annual = _calculate_sortino(daily_returns) * np.sqrt(252)

        # APR calculation
        trading_days = len(test_df) / bars_per_day
        if trading_days > 0:
            daily_return_avg = total_return_pct / trading_days
            apr_simple = daily_return_avg * 252
            apr_compound = ((1 + total_return_pct/100) ** (252/trading_days) - 1) * 100 if trading_days > 0 else 0
        else:
            apr_simple = 0
            apr_compound = 0

        # Action distribution
        total_actions = sum(actions_taken.values())
        action_pct = {k: v/total_actions*100 if total_actions > 0 else 0 for k, v in actions_taken.items()}

        # =====================================================================
        # LOG RESULTS
        # =====================================================================
        logger.info("=" * 60)
        logger.info("L4 BACKTEST RESULTS - FULL 2025 PERIOD")
        logger.info("=" * 60)

        logger.info(f"\n{'PERIOD':<35} {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")
        logger.info(f"{'Total Bars':<35} {len(test_df):,}")
        logger.info(f"{'Trading Days':<35} {trading_days:.1f}")

        logger.info("\n" + "-" * 60)
        logger.info("CAPITAL")
        logger.info("-" * 60)
        logger.info(f"{'Initial Capital':<35} ${INITIAL_CAPITAL:,.2f}")
        logger.info(f"{'Final Capital':<35} ${capital:,.2f}")
        logger.info(f"{'Total PnL':<35} ${total_pnl:,.2f}")
        logger.info(f"{'Total Return':<35} {total_return_pct:.2f}%")

        logger.info("\n" + "-" * 60)
        logger.info("ANNUALIZED RETURNS")
        logger.info("-" * 60)
        logger.info(f"{'APR (Simple)':<35} {apr_simple:.1f}%")
        logger.info(f"{'APR (Compound)':<35} {apr_compound:.1f}%")

        logger.info("\n" + "-" * 60)
        logger.info("RISK METRICS")
        logger.info("-" * 60)
        logger.info(f"{'Max Drawdown':<35} {max_dd*100:.2f}%")
        logger.info(f"{'Drawdown Duration (bars)':<35} {dd_duration}")
        logger.info(f"{'Sharpe Ratio (annualized)':<35} {sharpe_annual:.2f}")
        logger.info(f"{'Sortino Ratio (annualized)':<35} {sortino_annual:.2f}")

        logger.info("\n" + "-" * 60)
        logger.info("TRADE STATISTICS")
        logger.info("-" * 60)
        logger.info(f"{'Total Trades':<35} {len(trades)}")
        logger.info(f"{'Winning Trades':<35} {len(winning_trades)}")
        logger.info(f"{'Losing Trades':<35} {len(losing_trades)}")
        logger.info(f"{'Win Rate':<35} {win_rate:.1f}%")
        logger.info(f"{'Profit Factor':<35} {profit_factor:.2f}")
        logger.info(f"{'Avg Trade PnL':<35} ${avg_trade_pnl:.2f}")
        logger.info(f"{'Avg Trade Duration':<35} {avg_trade_hours:.1f}h ({avg_bars_held:.0f} bars)")

        logger.info("\n" + "-" * 60)
        logger.info("ACTION DISTRIBUTION")
        logger.info("-" * 60)
        logger.info(f"{'LONG':<35} {action_pct['long']:.1f}%")
        logger.info(f"{'HOLD':<35} {action_pct['hold']:.1f}%")
        logger.info(f"{'SHORT':<35} {action_pct['short']:.1f}%")

        # Trade details (first 10)
        if trades:
            logger.info("\n" + "-" * 60)
            logger.info("TRADE DETAILS (first 10)")
            logger.info("-" * 60)
            for i, t in enumerate(trades[:10], 1):
                pnl_str = f"+${t['pnl']:.2f}" if t['pnl'] > 0 else f"-${abs(t['pnl']):.2f}"
                logger.info(f"Trade {i:2d}: {t['direction']:5s} | {t['bars_held']:4d} bars ({t['bars_held']*5/60:.1f}h) | {pnl_str}")

        logger.info("\n" + "=" * 60)

        # Check acceptance criteria
        passed = True
        warnings_list = []

        if total_return_pct < -20:
            warnings_list.append(f"Total return {total_return_pct:.1f}% < -20%")
            passed = False

        max_action_pct = max(action_pct.values())
        if max_action_pct > 95:
            warnings_list.append(f"Action imbalance: {max_action_pct:.1f}% > 95%")
            passed = False

        if len(trades) == 0:
            warnings_list.append("No trades executed")
            passed = False

        if max_dd > 0.50:
            warnings_list.append(f"Max DD {max_dd*100:.1f}% > 50%")
            passed = False

        if passed:
            logger.info("STATUS: PASSED")
        else:
            logger.warning("STATUS: FAILED")
            for w in warnings_list:
                logger.warning(f"  - {w}")

        logger.info("=" * 60)

        # Build comprehensive metrics dict
        metrics = {
            "period": {
                "start": str(test_df['timestamp'].min()),
                "end": str(test_df['timestamp'].max()),
                "bars": len(test_df),
                "trading_days": float(trading_days),
            },
            "capital": {
                "initial": INITIAL_CAPITAL,
                "final": float(capital),
                "total_pnl": float(total_pnl),
                "total_return_pct": float(total_return_pct),
            },
            "annualized": {
                "apr_simple": float(apr_simple),
                "apr_compound": float(apr_compound),
            },
            "risk": {
                "max_drawdown_pct": float(max_dd * 100),
                "drawdown_duration_bars": dd_duration,
                "sharpe_annual": float(sharpe_annual),
                "sortino_annual": float(sortino_annual),
            },
            "trades": {
                "total": len(trades),
                "winning": len(winning_trades),
                "losing": len(losing_trades),
                "win_rate_pct": float(win_rate),
                "profit_factor": float(profit_factor) if profit_factor != float('inf') else 999.99,
                "avg_pnl": float(avg_trade_pnl),
                "avg_duration_hours": float(avg_trade_hours),
                "avg_duration_bars": float(avg_bars_held),
            },
            "action_distribution": action_pct,
            "trade_details": trades[:20],  # First 20 trades
            "passed": passed,
            "warnings": warnings_list,
        }

        return metrics

    except Exception as e:
        logger.error(f"L4 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main pipeline entry point.

    Pipeline flow with SSOT norm_stats:
        L2 -> dataset.csv + norm_stats.json
        L3 -> uses L2 norm_stats for training
        L4 -> uses L2 norm_stats for evaluation (SAME as L3!)
    """
    parser = argparse.ArgumentParser(description="Run L2→L3→L4 Pipeline")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps")
    parser.add_argument("--skip-l2", action="store_true", help="Skip L2 (use existing dataset)")
    parser.add_argument("--skip-l3", action="store_true", help="Skip L3 (evaluate existing model)")
    parser.add_argument("--dataset", type=str, help="Path to existing dataset")
    parser.add_argument("--model", type=str, help="Path to existing model")
    parser.add_argument("--norm-stats", type=str, help="Path to norm_stats.json (auto-detected if not provided)")

    args = parser.parse_args()

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("FULL PIPELINE: L2 -> L3 -> L4 (with SSOT norm_stats)")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Timesteps: {args.timesteps:,}")
    logger.info("=" * 60)

    results = {
        "start_time": datetime.now().isoformat(),
        "timesteps": args.timesteps,
        "l2": None,
        "l3": None,
        "l4": None,
    }

    # =========================================================================
    # L2: Dataset Builder (generates dataset.csv + norm_stats.json)
    # =========================================================================
    dataset_path = None
    norm_stats_path = None

    if args.dataset:
        dataset_path = Path(args.dataset)
        logger.info(f"Using provided dataset: {dataset_path}")
    elif not args.skip_l2:
        dataset_path = run_l2_dataset_builder()
        if dataset_path:
            results["l2"] = {"dataset_path": str(dataset_path), "success": True}
        else:
            # Try to find existing dataset
            output_dir = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "datasets_5min"
            if output_dir.exists():
                # Prefer DS3_MACRO_CORE (recommended for production)
                ds3_datasets = list(output_dir.glob("*DS3_MACRO_CORE*.csv"))
                if ds3_datasets:
                    dataset_path = max(ds3_datasets, key=lambda p: p.stat().st_mtime)
                else:
                    datasets = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.parquet"))
                    if datasets:
                        dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)
                if dataset_path:
                    logger.info(f"Using existing dataset: {dataset_path}")
    else:
        # Find most recent dataset
        output_dir = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "datasets_5min"
        if output_dir.exists():
            # Prefer DS3_MACRO_CORE
            ds3_datasets = list(output_dir.glob("*DS3_MACRO_CORE*.csv"))
            if ds3_datasets:
                dataset_path = max(ds3_datasets, key=lambda p: p.stat().st_mtime)
            else:
                datasets = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.parquet"))
                if datasets:
                    dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)
            if dataset_path:
                logger.info(f"Using existing dataset: {dataset_path}")

    if not dataset_path or not dataset_path.exists():
        logger.error("No dataset available. Run L2 first or provide --dataset")
        return 1

    # =========================================================================
    # Find or validate norm_stats.json (CRITICAL for L3/L4 parity)
    # =========================================================================
    if args.norm_stats:
        norm_stats_path = Path(args.norm_stats)
    else:
        norm_stats_path = _find_norm_stats(dataset_path)

    if norm_stats_path is None or not norm_stats_path.exists():
        logger.error("=" * 60)
        logger.error("CRITICAL: norm_stats.json NOT FOUND!")
        logger.error("=" * 60)
        logger.error("This file is required for L3 and L4 to use the same normalization.")
        logger.error("Without it, L4 evaluation will fail (99% LONG prediction bug).")
        logger.error("")
        logger.error("Solutions:")
        logger.error("  1. Run L2 again (it now generates norm_stats.json)")
        logger.error("  2. Provide --norm-stats /path/to/norm_stats.json")
        logger.error("=" * 60)
        return 1

    logger.info(f"[SSOT] norm_stats.json: {norm_stats_path}")
    results["norm_stats_path"] = str(norm_stats_path)

    # =========================================================================
    # L3: Training (uses L2 norm_stats)
    # =========================================================================
    model_path = None
    if args.model:
        model_path = Path(args.model)
        logger.info(f"Using provided model: {model_path}")
    elif not args.skip_l3:
        l3_result = run_l3_training(dataset_path, args.timesteps)
        if l3_result:
            results["l3"] = l3_result
            model_path = Path(l3_result.get("model_path"))
        else:
            logger.error("L3 training failed")
            return 1
    else:
        # Find most recent model
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("**/final_model.zip"))
            if model_files:
                model_path = max(model_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using existing model: {model_path}")

    if not model_path or not model_path.exists():
        logger.error("No model available. Run L3 first or provide --model")
        return 1

    # =========================================================================
    # L4: Evaluation (uses SAME norm_stats as L3!)
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("L4 uses SAME norm_stats as L3 (fixing normalization mismatch bug)")
    logger.info("=" * 60)

    l4_result = run_l4_evaluation(model_path, dataset_path, norm_stats_path)
    if l4_result:
        results["l4"] = l4_result

    # =========================================================================
    # Save results
    # =========================================================================
    results["end_time"] = datetime.now().isoformat()
    results["duration_seconds"] = time.time() - start_time

    results_path = PROJECT_ROOT / "results" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved: {results_path}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {(time.time() - start_time) / 60:.1f} minutes")
    logger.info(f"Dataset: {dataset_path.name}")
    logger.info(f"Norm Stats: {norm_stats_path.name}")
    logger.info(f"Model: {model_path.name if model_path else 'N/A'}")

    if results.get("l4"):
        l4 = results["l4"]
        logger.info("")
        logger.info("L4 BACKTEST SUMMARY (Full 2025):")
        logger.info("-" * 40)

        # Capital
        capital = l4.get("capital", {})
        logger.info(f"  Capital: ${capital.get('initial', 10000):,.0f} -> ${capital.get('final', 0):,.2f}")
        logger.info(f"  Total PnL: ${capital.get('total_pnl', 0):,.2f}")
        logger.info(f"  Total Return: {capital.get('total_return_pct', 0):.2f}%")

        # Annualized
        annual = l4.get("annualized", {})
        logger.info(f"  APR: {annual.get('apr_simple', 0):.1f}% (simple) / {annual.get('apr_compound', 0):.1f}% (compound)")

        # Risk
        risk = l4.get("risk", {})
        logger.info(f"  Max Drawdown: {risk.get('max_drawdown_pct', 0):.2f}%")
        logger.info(f"  Sharpe: {risk.get('sharpe_annual', 0):.2f} | Sortino: {risk.get('sortino_annual', 0):.2f}")

        # Trades
        trades = l4.get("trades", {})
        logger.info(f"  Trades: {trades.get('total', 0)} ({trades.get('winning', 0)}W / {trades.get('losing', 0)}L)")
        logger.info(f"  Win Rate: {trades.get('win_rate_pct', 0):.1f}% | Profit Factor: {trades.get('profit_factor', 0):.2f}")

        # Actions
        action_dist = l4.get("action_distribution", {})
        logger.info(f"  Actions: LONG={action_dist.get('long', 0):.1f}%, "
                   f"HOLD={action_dist.get('hold', 0):.1f}%, "
                   f"SHORT={action_dist.get('short', 0):.1f}%")

        logger.info(f"  Status: {'PASSED' if l4.get('passed', False) else 'FAILED'}")

    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
