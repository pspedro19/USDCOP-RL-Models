#!/usr/bin/env python3
"""
Walk-Forward Validation Script
==============================

Prevents overfitting by training on rolling windows and validating on unseen data.

Walk-Forward Periods:
- Train 1: 2020-03 to 2022-12 (34 months) -> Test 1: 2023-01 to 2023-06 (6 months)
- Train 2: 2020-03 to 2023-06 (40 months) -> Test 2: 2023-07 to 2023-12 (6 months)
- Train 3: 2020-03 to 2023-12 (46 months) -> Test 3: 2024-01 to 2024-06 (6 months)
- Final:   2020-03 to 2024-12 (58 months) -> Test 4: 2025-01 to present (OOS)

Validation Gates (per fold):
- Total Return > -10%
- Sharpe Ratio > -1.0
- Max Drawdown < 30%
- Win Rate > 40%
- Profit Factor > 0.8
- Trades: 10-40
- Action Balance: No single action > 60%

Usage:
    python scripts/walk_forward_validation.py
    python scripts/walk_forward_validation.py --folds 3  # Only first 3 folds
    python scripts/walk_forward_validation.py --timesteps 200000

Author: Trading Team
Date: 2026-02-02
Contract: PHASE 3 Anti-Overfitting
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure logs directory exists
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "logs" / f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION GATES (from Plan Phase 5)
# =============================================================================
@dataclass
class ValidationGates:
    """Validation gates that each fold must pass."""

    min_return_pct: float = -10.0  # Fail if return < -10%
    min_sharpe: float = -1.0  # Fail if Sharpe < -1.0
    max_drawdown_pct: float = 30.0  # Fail if DD > 30%
    min_win_rate_pct: float = 40.0  # Fail if win rate < 40%
    min_profit_factor: float = 0.8  # Fail if PF < 0.8
    min_trades: int = 10  # Fail if fewer than 10 trades
    max_trades: int = 40  # Warn if more than 40 trades
    max_action_imbalance_pct: float = 60.0  # Fail if any action > 60%


# =============================================================================
# FOLD DEFINITIONS
# =============================================================================
@dataclass
class WalkForwardFold:
    """Definition of a single walk-forward fold."""

    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


# Default folds as specified in the plan
DEFAULT_FOLDS = [
    WalkForwardFold(
        name="Fold_1_2023H1",
        train_start="2020-03-01",
        train_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2023-06-30",
    ),
    WalkForwardFold(
        name="Fold_2_2023H2",
        train_start="2020-03-01",
        train_end="2023-06-30",
        test_start="2023-07-01",
        test_end="2023-12-31",
    ),
    WalkForwardFold(
        name="Fold_3_2024H1",
        train_start="2020-03-01",
        train_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2024-06-30",
    ),
    WalkForwardFold(
        name="Fold_4_2025_OOS",
        train_start="2020-03-01",
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-12-31",
    ),
]


# =============================================================================
# FOLD RESULT
# =============================================================================
@dataclass
class FoldResult:
    """Result of a single walk-forward fold."""

    fold: WalkForwardFold
    passed: bool
    metrics: Dict[str, Any]
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    model_path: Optional[str] = None
    training_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    std = np.std(returns)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess_returns) / std)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown percentage."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return float(np.max(drawdown))


def validate_fold_result(metrics: Dict[str, Any], gates: ValidationGates) -> Tuple[bool, List[str], List[str]]:
    """
    Validate fold metrics against gates.

    Returns:
        Tuple of (passed, violations, warnings)
    """
    violations = []
    warnings = []

    # Extract metrics
    total_return_pct = metrics.get("capital", {}).get("total_return_pct", -100)
    sharpe = metrics.get("risk", {}).get("sharpe_annual", -10)
    max_dd_pct = metrics.get("risk", {}).get("max_drawdown_pct", 100)
    win_rate = metrics.get("trades", {}).get("win_rate_pct", 0)
    profit_factor = metrics.get("trades", {}).get("profit_factor", 0)
    num_trades = metrics.get("trades", {}).get("total", 0)
    action_dist = metrics.get("action_distribution", {})

    # Check violations (hard failures)
    if total_return_pct < gates.min_return_pct:
        violations.append(f"Return {total_return_pct:.1f}% < {gates.min_return_pct}%")

    if sharpe < gates.min_sharpe:
        violations.append(f"Sharpe {sharpe:.2f} < {gates.min_sharpe}")

    if max_dd_pct > gates.max_drawdown_pct:
        violations.append(f"Max DD {max_dd_pct:.1f}% > {gates.max_drawdown_pct}%")

    if win_rate < gates.min_win_rate_pct:
        violations.append(f"Win Rate {win_rate:.1f}% < {gates.min_win_rate_pct}%")

    if profit_factor < gates.min_profit_factor:
        violations.append(f"Profit Factor {profit_factor:.2f} < {gates.min_profit_factor}")

    if num_trades < gates.min_trades:
        violations.append(f"Trades {num_trades} < {gates.min_trades}")

    # Check action imbalance
    max_action_pct = max(action_dist.values()) if action_dist else 100
    if max_action_pct > gates.max_action_imbalance_pct:
        violations.append(f"Action imbalance {max_action_pct:.1f}% > {gates.max_action_imbalance_pct}%")

    # Check warnings (soft failures)
    if num_trades > gates.max_trades:
        warnings.append(f"Trades {num_trades} > {gates.max_trades} (excessive trading)")

    passed = len(violations) == 0
    return passed, violations, warnings


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================
def train_fold(
    fold: WalkForwardFold,
    dataset_path: Path,
    norm_stats_path: Path,
    total_timesteps: int,
    output_dir: Path,
) -> Optional[Path]:
    """
    Train a model for a specific fold.

    Args:
        fold: Fold definition
        dataset_path: Path to full dataset
        norm_stats_path: Path to normalization stats
        total_timesteps: Training timesteps
        output_dir: Where to save the model

    Returns:
        Path to trained model or None on failure
    """
    try:
        from stable_baselines3 import PPO

        from src.config.pipeline_config import load_pipeline_config
        from src.training.config import RewardConfig
        from src.training.environments.env_factory import EnvironmentFactory, EnvFactoryConfig
        from src.training.environments.trading_env import TradingEnvConfig

        # Load dataset
        if str(dataset_path).endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path)

        # Ensure timestamp column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter to training period
        train_df = df[(df["timestamp"] >= fold.train_start) & (df["timestamp"] <= fold.train_end)].reset_index(
            drop=True
        )

        if len(train_df) < 50000:
            logger.warning(f"[{fold.name}] Training data too small: {len(train_df)} rows")
            return None

        logger.info(f"[{fold.name}] Training data: {len(train_df):,} rows")
        logger.info(f"[{fold.name}] Period: {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}")

        # Load norm_stats
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

        # Load SSOT config (V22: use pipeline_ssot instead of experiment_ssot)
        ssot_config = load_pipeline_config()

        # Create environment config from SSOT
        env_config = TradingEnvConfig(
            episode_length=ssot_config.environment.episode_length,
            initial_balance=ssot_config.environment.initial_balance,
            transaction_cost_bps=ssot_config.environment.transaction_cost_bps,
            slippage_bps=ssot_config.environment.slippage_bps,
            max_drawdown_pct=ssot_config.environment.max_drawdown_pct,
            max_position_duration=ssot_config.environment.max_position_duration,
            threshold_long=ssot_config.environment.threshold_long,
            threshold_short=ssot_config.environment.threshold_short,
            observation_dim=ssot_config.get_observation_dim(),
            action_type=ssot_config.action_type,
            n_actions=ssot_config.n_actions,
        )

        # Create environment
        factory = EnvironmentFactory()
        env = factory.create_training_env(
            df=train_df,
            norm_stats=norm_stats,
            config=env_config,
        )

        # PPO hyperparameters from SSOT
        net_arch = list(ssot_config._raw.get("training", {}).get("network", {}).get("policy_layers", [256, 256]))
        ppo_kwargs = dict(
            learning_rate=ssot_config.ppo.learning_rate,
            n_steps=ssot_config.ppo.n_steps,
            batch_size=ssot_config.ppo.batch_size,
            n_epochs=ssot_config.ppo.n_epochs,
            gamma=ssot_config.ppo.gamma,
            gae_lambda=ssot_config.ppo.gae_lambda,
            clip_range=ssot_config.ppo.clip_range,
            ent_coef=ssot_config.ppo.ent_coef,
            vf_coef=ssot_config.ppo.vf_coef,
            max_grad_norm=ssot_config.ppo.max_grad_norm,
            verbose=0,
            device="cpu",
        )

        # V22 P3: Conditional PPO vs RecurrentPPO
        use_lstm = ssot_config.lstm.enabled if ssot_config.lstm else False
        if use_lstm:
            from sb3_contrib import RecurrentPPO
            policy_kwargs = {
                "net_arch": net_arch,
                "lstm_hidden_size": ssot_config.lstm.hidden_size,
                "n_lstm_layers": ssot_config.lstm.n_layers,
            }
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env,
                policy_kwargs=policy_kwargs,
                **ppo_kwargs,
            )
            logger.info(f"[{fold.name}] RecurrentPPO (MlpLstmPolicy, lstm_hidden={ssot_config.lstm.hidden_size})")
        else:
            policy_kwargs = {"net_arch": net_arch}
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                **ppo_kwargs,
            )
            logger.info(f"[{fold.name}] PPO (MlpPolicy)")

        # Train
        logger.info(f"[{fold.name}] Training for {total_timesteps:,} timesteps...")
        model.learn(total_timesteps=total_timesteps)

        # Save model
        model_path = output_dir / f"{fold.name}_model.zip"
        model.save(str(model_path))
        logger.info(f"[{fold.name}] Model saved: {model_path}")

        env.close()
        return model_path

    except Exception as e:
        logger.error(f"[{fold.name}] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_fold(
    fold: WalkForwardFold,
    model_path: Path,
    dataset_path: Path,
    norm_stats_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a model on the test period of a fold.

    Args:
        fold: Fold definition
        model_path: Path to trained model
        dataset_path: Path to full dataset
        norm_stats_path: Path to normalization stats

    Returns:
        Metrics dictionary or None on failure
    """
    try:
        from stable_baselines3 import PPO

        from src.config.pipeline_config import load_pipeline_config

        # Load SSOT config (V22: pipeline_ssot)
        config = load_pipeline_config()

        # V22 P3: Conditional model loading (PPO vs RecurrentPPO)
        use_lstm = config.lstm.enabled if config.lstm else False
        if use_lstm:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(str(model_path), device="cpu")
            logger.info(f"[{fold.name}] Loaded RecurrentPPO model")
        else:
            model = PPO.load(str(model_path), device="cpu")
            logger.info(f"[{fold.name}] Loaded PPO model")

        # Load norm_stats
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

        # Load dataset
        if str(dataset_path).endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path)

        # Ensure timestamp column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter to test period
        test_df = df[(df["timestamp"] >= fold.test_start) & (df["timestamp"] <= fold.test_end)].reset_index(drop=True)

        if len(test_df) == 0:
            logger.error(f"[{fold.name}] No test data for period {fold.test_start} to {fold.test_end}")
            return None

        logger.info(f"[{fold.name}] Test data: {len(test_df):,} rows")
        logger.info(f"[{fold.name}] Test period: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")

        # Feature columns from SSOT (V22: 19 market features, not hardcoded 13)
        feature_cols = [f.name for f in config.get_market_features()]
        n_market_features = len(feature_cols)
        obs_dim = config.get_observation_dim()  # V22: 28 (dynamic, not hardcoded 15)

        # Config from SSOT
        INITIAL_CAPITAL = config.environment.initial_balance
        TRANSACTION_COST_BPS = config.environment.transaction_cost_bps
        SLIPPAGE_BPS = config.environment.slippage_bps
        THRESHOLD_LONG = config.environment.threshold_long
        THRESHOLD_SHORT = config.environment.threshold_short
        MAX_POSITION_HOLDING = config.environment.max_position_duration

        # V22 P2: Discrete action space config
        action_type = config.action_type  # "discrete" or "continuous"
        STOP_LOSS_PCT = config.environment.stop_loss_pct  # -0.04
        TAKE_PROFIT_PCT = config.environment.take_profit_pct  # 0.04

        total_cost_rate = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10_000

        # Backtest
        capital = INITIAL_CAPITAL
        position = 0
        entry_bar = 0
        entry_price = 0.0

        equity_curve = [capital]
        returns_per_bar = []
        trades = []
        trade_pnl_accumulator = 0.0
        actions_taken = {"long": 0, "hold": 0, "short": 0, "close": 0}

        for i in range(1, len(test_df)):
            # Build observation (V22: dynamic obs_dim from SSOT)
            obs = np.zeros(obs_dim, dtype=np.float32)

            # Market features (indices 0..n_market_features-1)
            for j, col in enumerate(feature_cols):
                if j >= obs_dim:
                    break
                if col not in test_df.columns:
                    continue
                value = test_df.iloc[i][col]
                if col.endswith("_z"):
                    obs[j] = np.clip(value, -5, 5)
                elif col in norm_stats:
                    stats = norm_stats[col]
                    mean = stats.get("mean", 0)
                    std = stats.get("std", 1)
                    if std < 1e-8:
                        std = 1.0
                    obs[j] = np.clip((value - mean) / std, -5, 5)
                else:
                    obs[j] = np.clip(value, -5, 5)

            # State features (V22: indices start at n_market_features)
            state_start = n_market_features
            obs[state_start] = float(position)  # position

            current_price = test_df.iloc[i]["close"]
            unrealized_pnl_pct = 0.0
            if position != 0 and entry_price > 0:
                unrealized_pnl_pct = (current_price - entry_price) / entry_price * position
            obs[state_start + 1] = np.clip(unrealized_pnl_pct, -1.0, 1.0)  # unrealized_pnl

            # sl_proximity
            if position != 0 and STOP_LOSS_PCT != 0:
                sl_prox = np.clip((unrealized_pnl_pct - STOP_LOSS_PCT) / abs(STOP_LOSS_PCT), 0, 2) / 2
            else:
                sl_prox = 1.0
            obs[state_start + 2] = sl_prox

            # tp_proximity
            if position != 0 and TAKE_PROFIT_PCT > 0:
                tp_prox = np.clip(unrealized_pnl_pct / TAKE_PROFIT_PCT, 0, 1)
            else:
                tp_prox = 0.0
            obs[state_start + 3] = tp_prox

            # bars_held
            if position != 0 and MAX_POSITION_HOLDING > 0:
                bars_held = np.clip((i - entry_bar) / MAX_POSITION_HOLDING, 0, 1)
            else:
                bars_held = 0.0
            obs[state_start + 4] = bars_held

            # V22 P1: Temporal features (hour_sin, hour_cos, dow_sin, dow_cos)
            if "timestamp" in test_df.columns:
                ts = test_df.iloc[i]["timestamp"]
                hour_frac = (ts.hour + ts.minute / 60.0 - 13.0) / 5.0
                hour_frac = max(0.0, min(1.0, hour_frac))
                dow_frac = ts.dayofweek / 5.0  # 5 trading days, NOT 7
                obs[state_start + 5] = float(np.sin(2 * np.pi * hour_frac))
                obs[state_start + 6] = float(np.cos(2 * np.pi * hour_frac))
                obs[state_start + 7] = float(np.sin(2 * np.pi * dow_frac))
                obs[state_start + 8] = float(np.cos(2 * np.pi * dow_frac))

            obs = np.nan_to_num(obs, nan=0.0)

            # Get action (V22: handle both discrete and continuous)
            action, _ = model.predict(obs, deterministic=True)

            if action_type == "discrete":
                # Discrete(4): 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
                action_int = int(action) if not hasattr(action, "__len__") else int(action[0])
                if action_int == 0:  # HOLD
                    target_action = position  # Maintain current position
                    actions_taken["hold"] += 1
                elif action_int == 1:  # BUY
                    target_action = 1
                    actions_taken["long"] += 1
                elif action_int == 2:  # SELL
                    target_action = -1
                    actions_taken["short"] += 1
                elif action_int == 3:  # CLOSE
                    target_action = 0
                    actions_taken["close"] += 1
                else:
                    target_action = position
                    actions_taken["hold"] += 1
            else:
                # Continuous Box(1): threshold-based mapping
                action_value = float(action[0]) if hasattr(action, "__len__") else float(action)
                if action_value > THRESHOLD_LONG:
                    target_action = 1
                    actions_taken["long"] += 1
                elif action_value < THRESHOLD_SHORT:
                    target_action = -1
                    actions_taken["short"] += 1
                else:
                    target_action = 0
                    actions_taken["hold"] += 1

            # Force close if max holding exceeded
            if position != 0 and (i - entry_bar) >= MAX_POSITION_HOLDING:
                target_action = 0

            # Stop-loss / take-profit checks
            if position != 0:
                if unrealized_pnl_pct <= STOP_LOSS_PCT:
                    target_action = 0  # Force close on SL
                elif unrealized_pnl_pct >= TAKE_PROFIT_PCT:
                    target_action = 0  # Force close on TP

            # Market return
            if "raw_log_ret_5m" in test_df.columns:
                market_return = test_df.iloc[i]["raw_log_ret_5m"]
            else:
                current_close = test_df.iloc[i]["close"]
                prev_close = test_df.iloc[i - 1]["close"]
                market_return = np.log(current_close / prev_close) if prev_close > 0 else 0.0

            # PnL
            position_pnl = position * market_return * capital

            trade_cost = 0.0
            if target_action != position:
                change_magnitude = abs(target_action - position)
                trade_cost = change_magnitude * total_cost_rate * capital

                if position != 0:
                    trades.append({
                        "entry_bar": entry_bar,
                        "exit_bar": i,
                        "bars_held": i - entry_bar,
                        "direction": "LONG" if position == 1 else "SHORT",
                        "pnl": trade_pnl_accumulator,
                    })

                position = target_action
                entry_bar = i
                entry_price = test_df.iloc[i]["close"]
                trade_pnl_accumulator = 0.0
            else:
                trade_pnl_accumulator += position_pnl

            net_pnl = position_pnl - trade_cost
            capital += net_pnl

            equity_curve.append(capital)
            bar_return = net_pnl / equity_curve[-2] if equity_curve[-2] > 0 else 0
            returns_per_bar.append(bar_return)

        # Close final position
        if position != 0:
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": len(test_df) - 1,
                "bars_held": len(test_df) - 1 - entry_bar,
                "direction": "LONG" if position == 1 else "SHORT",
                "pnl": trade_pnl_accumulator,
            })

        # Calculate metrics
        equity_curve = np.array(equity_curve)
        returns = np.array(returns_per_bar)

        total_pnl = capital - INITIAL_CAPITAL
        total_return_pct = (capital / INITIAL_CAPITAL - 1) * 100
        max_dd = calculate_max_drawdown(equity_curve)

        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Daily returns for Sharpe (use SSOT bars_per_day)
        trading_schedule = config.get_trading_schedule()
        bars_per_day = trading_schedule.get("bars_per_day", 78)  # SSOT: 78 for USDCOP
        daily_returns = []
        for d in range(0, len(returns), bars_per_day):
            chunk = returns[d : d + bars_per_day]
            if len(chunk) > 0:
                daily_returns.append(np.sum(chunk))
        daily_returns = np.array(daily_returns) if daily_returns else np.array([0])
        sharpe_annual = calculate_sharpe(daily_returns) * np.sqrt(252)

        # Action distribution
        total_actions = sum(actions_taken.values())
        action_pct = {k: v / total_actions * 100 if total_actions > 0 else 0 for k, v in actions_taken.items()}

        metrics = {
            "period": {
                "start": str(test_df["timestamp"].min()),
                "end": str(test_df["timestamp"].max()),
                "bars": len(test_df),
            },
            "capital": {
                "initial": INITIAL_CAPITAL,
                "final": float(capital),
                "total_pnl": float(total_pnl),
                "total_return_pct": float(total_return_pct),
            },
            "risk": {
                "max_drawdown_pct": float(max_dd * 100),
                "sharpe_annual": float(sharpe_annual),
            },
            "trades": {
                "total": len(trades),
                "winning": len(winning_trades),
                "losing": len(losing_trades),
                "win_rate_pct": float(win_rate),
                "profit_factor": float(profit_factor) if profit_factor != float("inf") else 999.99,
            },
            "action_distribution": action_pct,
        }

        return metrics

    except Exception as e:
        logger.error(f"[{fold.name}] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# MAIN
# =============================================================================
def run_walk_forward_validation(
    dataset_path: Path,
    norm_stats_path: Path,
    folds: List[WalkForwardFold],
    total_timesteps: int,
    output_dir: Path,
    gates: ValidationGates,
) -> Dict[str, Any]:
    """
    Run complete walk-forward validation.

    Args:
        dataset_path: Path to full dataset
        norm_stats_path: Path to normalization stats
        folds: List of fold definitions
        total_timesteps: Training timesteps per fold
        output_dir: Where to save models and results
        gates: Validation gates

    Returns:
        Summary dictionary with all results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "start_time": datetime.now().isoformat(),
        "timesteps_per_fold": total_timesteps,
        "folds": [],
        "summary": {},
    }

    fold_results: List[FoldResult] = []

    for fold in folds:
        logger.info("=" * 70)
        logger.info(f"FOLD: {fold.name}")
        logger.info(f"  Train: {fold.train_start} to {fold.train_end}")
        logger.info(f"  Test:  {fold.test_start} to {fold.test_end}")
        logger.info("=" * 70)

        # Train
        train_start = time.time()
        model_path = train_fold(
            fold=fold,
            dataset_path=dataset_path,
            norm_stats_path=norm_stats_path,
            total_timesteps=total_timesteps,
            output_dir=output_dir,
        )
        train_time = time.time() - train_start

        if model_path is None:
            fold_results.append(
                FoldResult(
                    fold=fold,
                    passed=False,
                    metrics={},
                    violations=["Training failed"],
                    training_time_seconds=train_time,
                )
            )
            continue

        # Evaluate
        eval_start = time.time()
        metrics = evaluate_fold(
            fold=fold,
            model_path=model_path,
            dataset_path=dataset_path,
            norm_stats_path=norm_stats_path,
        )
        eval_time = time.time() - eval_start

        if metrics is None:
            fold_results.append(
                FoldResult(
                    fold=fold,
                    passed=False,
                    metrics={},
                    violations=["Evaluation failed"],
                    model_path=str(model_path),
                    training_time_seconds=train_time,
                    evaluation_time_seconds=eval_time,
                )
            )
            continue

        # Validate
        passed, violations, warnings = validate_fold_result(metrics, gates)

        fold_result = FoldResult(
            fold=fold,
            passed=passed,
            metrics=metrics,
            violations=violations,
            warnings=warnings,
            model_path=str(model_path),
            training_time_seconds=train_time,
            evaluation_time_seconds=eval_time,
        )
        fold_results.append(fold_result)

        # Log result
        logger.info(f"\n[{fold.name}] RESULTS:")
        logger.info(f"  Return: {metrics['capital']['total_return_pct']:.2f}%")
        logger.info(f"  Sharpe: {metrics['risk']['sharpe_annual']:.2f}")
        logger.info(f"  Max DD: {metrics['risk']['max_drawdown_pct']:.2f}%")
        logger.info(f"  Trades: {metrics['trades']['total']} (Win: {metrics['trades']['win_rate_pct']:.1f}%)")
        logger.info(f"  Profit Factor: {metrics['trades']['profit_factor']:.2f}")
        logger.info(f"  Actions: L={metrics['action_distribution'].get('long', 0):.1f}%, "
                   f"H={metrics['action_distribution'].get('hold', 0):.1f}%, "
                   f"S={metrics['action_distribution'].get('short', 0):.1f}%")
        logger.info(f"  Status: {'PASSED' if passed else 'FAILED'}")

        if violations:
            for v in violations:
                logger.warning(f"    VIOLATION: {v}")
        if warnings:
            for w in warnings:
                logger.info(f"    WARNING: {w}")

    # Summary
    passed_folds = sum(1 for r in fold_results if r.passed)
    total_folds = len(fold_results)

    # Calculate cross-fold statistics
    returns = [r.metrics.get("capital", {}).get("total_return_pct", 0) for r in fold_results if r.metrics]
    sharpes = [r.metrics.get("risk", {}).get("sharpe_annual", 0) for r in fold_results if r.metrics]

    summary = {
        "total_folds": total_folds,
        "passed_folds": passed_folds,
        "pass_rate": passed_folds / total_folds * 100 if total_folds > 0 else 0,
        "mean_return_pct": float(np.mean(returns)) if returns else 0,
        "std_return_pct": float(np.std(returns)) if returns else 0,
        "mean_sharpe": float(np.mean(sharpes)) if sharpes else 0,
        "std_sharpe": float(np.std(sharpes)) if sharpes else 0,
        "overall_passed": passed_folds == total_folds,
    }

    results["folds"] = [
        {
            "name": r.fold.name,
            "train_period": f"{r.fold.train_start} to {r.fold.train_end}",
            "test_period": f"{r.fold.test_start} to {r.fold.test_end}",
            "passed": r.passed,
            "violations": r.violations,
            "warnings": r.warnings,
            "metrics": r.metrics,
            "model_path": r.model_path,
            "training_time_seconds": r.training_time_seconds,
            "evaluation_time_seconds": r.evaluation_time_seconds,
        }
        for r in fold_results
    ]
    results["summary"] = summary
    results["end_time"] = datetime.now().isoformat()

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Folds Passed: {passed_folds}/{total_folds} ({summary['pass_rate']:.1f}%)")
    logger.info(f"Mean Return: {summary['mean_return_pct']:.2f}% +/- {summary['std_return_pct']:.2f}%")
    logger.info(f"Mean Sharpe: {summary['mean_sharpe']:.2f} +/- {summary['std_sharpe']:.2f}")
    logger.info(f"Overall Status: {'PASSED' if summary['overall_passed'] else 'FAILED'}")
    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--dataset", type=str, help="Path to dataset (auto-detected if not provided)")
    parser.add_argument("--norm-stats", type=str, help="Path to norm_stats.json (auto-detected if not provided)")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps per fold")
    parser.add_argument("--folds", type=int, default=4, help="Number of folds to run (1-4)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for models and results")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info(f"Timesteps per fold: {args.timesteps:,}")
    logger.info(f"Folds to run: {args.folds}")
    logger.info("=" * 70)

    # Find dataset
    dataset_path = None
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        output_dir = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "datasets_5min"
        if output_dir.exists():
            ds3_datasets = list(output_dir.glob("*DS3_MACRO_CORE*.csv"))
            if ds3_datasets:
                dataset_path = max(ds3_datasets, key=lambda p: p.stat().st_mtime)
            else:
                datasets = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.parquet"))
                if datasets:
                    dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)

    if not dataset_path or not dataset_path.exists():
        logger.error("No dataset found. Provide --dataset or run L2 first.")
        return 1

    logger.info(f"Dataset: {dataset_path}")

    # Find norm_stats
    norm_stats_path = None
    if args.norm_stats:
        norm_stats_path = Path(args.norm_stats)
    else:
        # Try adjacent to dataset
        dataset_stem = dataset_path.stem
        if dataset_stem.startswith("RL_"):
            pattern1 = dataset_path.parent / f"{dataset_stem[3:]}_norm_stats.json"
        else:
            pattern1 = dataset_path.parent / f"{dataset_stem}_norm_stats.json"
        pattern2 = dataset_path.parent / "norm_stats.json"
        pattern3 = PROJECT_ROOT / "config" / "norm_stats.json"

        for p in [pattern1, pattern2, pattern3]:
            if p.exists():
                norm_stats_path = p
                break

    if not norm_stats_path or not norm_stats_path.exists():
        logger.error("No norm_stats.json found. Provide --norm-stats or run L2 first.")
        return 1

    logger.info(f"Norm Stats: {norm_stats_path}")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Run validation
    folds = DEFAULT_FOLDS[: args.folds]
    gates = ValidationGates()

    results = run_walk_forward_validation(
        dataset_path=dataset_path,
        norm_stats_path=norm_stats_path,
        folds=folds,
        total_timesteps=args.timesteps,
        output_dir=output_dir,
        gates=gates,
    )

    # Save results
    results_path = output_dir / "walk_forward_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved: {results_path}")

    return 0 if results["summary"]["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
