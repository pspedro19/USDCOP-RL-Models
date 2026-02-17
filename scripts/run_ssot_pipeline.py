"""
SSOT Pipeline Runner - L2 -> L3 -> L4 Seamless Integration
==========================================================
Runs the complete training pipeline using SSOT configuration.

Usage:
    python scripts/run_ssot_pipeline.py --stage all
    python scripts/run_ssot_pipeline.py --stage l2
    python scripts/run_ssot_pipeline.py --stage l3
    python scripts/run_ssot_pipeline.py --stage l4

Contract: CTR-PIPELINE-RUNNER-001
Version: 1.0.0
Date: 2026-02-03
"""

import argparse
import json
import logging
import random
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Run manifest (optional — pipeline works without it)
try:
    from src.ml_workflow.run_manifest import RunManifest
    MANIFEST_AVAILABLE = True
except ImportError:
    MANIFEST_AVAILABLE = False

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import EvalCallback

# =============================================================================
# REPRODUCIBILITY - Fixed seed for consistent results
# =============================================================================
SEED = 42

def set_reproducible_seeds(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============================================================================
# DETERMINISTIC EVAL CALLBACK - Reseeds eval env before each evaluation round
# =============================================================================

class DeterministicEvalCallback(EvalCallback):
    """
    EvalCallback that reseeds the eval environment before each evaluation round.

    This ensures that eval episodes always start from the same positions,
    making best_model selection reproducible across runs.
    """

    def __init__(self, *args, eval_seed: int = SEED, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_seed = eval_seed

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reseed eval env's np_random directly (avoid full reset)
            for env in self.eval_env.envs:
                env.np_random = np.random.Generator(np.random.PCG64(self.eval_seed))
        return super()._on_step()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# COLUMN MAPPING - Read from SSOT (pipeline_ssot.yaml → paths.sources.macro_column_map)
# =============================================================================

def get_macro_column_map() -> Dict[str, str]:
    """Load macro column mapping from SSOT config."""
    from src.config.pipeline_config import load_pipeline_config
    return load_pipeline_config().get_macro_column_map()


# =============================================================================
# SHARED UTILITIES (DRY: eliminates duplication across L3/L4 functions)
# =============================================================================

def _load_ssot_constants():
    """Load trading schedule and state features from SSOT config (called once)."""
    from src.config.pipeline_config import load_pipeline_config
    cfg = load_pipeline_config()

    # Trading schedule from SSOT (pipeline_ssot.yaml → trading_schedule)
    ts = cfg.get_trading_schedule()
    bars_per_day = ts.get("bars_per_day", 78)
    bars_per_year = ts.get("bars_per_year", 19656)
    warmup = ts.get("backtest_warmup_bars", 14)
    margin = ts.get("backtest_margin_bars", 50)

    # State features from SSOT (pipeline_ssot.yaml → features.state_features)
    state_features = cfg.get_state_feature_names()

    return bars_per_day, bars_per_year, warmup, margin, state_features

# Lazy-loaded from SSOT on first access
_SSOT_CONSTANTS = None

def _get_ssot_constants():
    """Get cached SSOT constants (loads once, caches for session)."""
    global _SSOT_CONSTANTS
    if _SSOT_CONSTANTS is None:
        _SSOT_CONSTANTS = _load_ssot_constants()
    return _SSOT_CONSTANTS

def get_bars_per_day() -> int:
    return _get_ssot_constants()[0]

def get_bars_per_year() -> int:
    return _get_ssot_constants()[1]

def get_backtest_warmup() -> int:
    return _get_ssot_constants()[2]

def get_backtest_margin() -> int:
    return _get_ssot_constants()[3]

def get_state_features() -> list:
    return _get_ssot_constants()[4]

def get_n_state_features() -> int:
    return len(get_state_features())


# =============================================================================
# FORECAST SIGNALS LOADER (EXP-RL-EXECUTOR)
# =============================================================================

# Module-level cache for forecast signals (loaded once per run)
_FORECAST_SIGNALS: Optional[Dict[str, tuple]] = None

def load_forecast_signals(path: str) -> Dict[str, tuple]:
    """
    Load forecast signals from parquet file.

    Returns:
        Dict mapping date string -> (direction: int, leverage: float)
    """
    global _FORECAST_SIGNALS
    if _FORECAST_SIGNALS is not None:
        return _FORECAST_SIGNALS

    logger.info(f"Loading forecast signals from: {path}")
    df = pd.read_parquet(path)

    signals = {}
    for _, row in df.iterrows():
        date_str = str(row["date"])
        direction = int(row["forecast_direction"])
        leverage = float(row.get("forecast_leverage", 1.0))
        signals[date_str] = (direction, leverage)

    logger.info(f"  Loaded {len(signals)} forecast signals")
    _FORECAST_SIGNALS = signals
    return signals


def create_env_config(
    config,
    feature_cols: List[str],
    episode_length: int,
    stage: str = "training",
) -> "TradingEnvConfig":
    """
    Factory for TradingEnvConfig. Single source for all L3/L4 env creation.

    Args:
        config: PipelineConfig from SSOT
        feature_cols: Market feature column names
        episode_length: Episode length in bars
        stage: "training" or "backtest" (controls reward_interval and config source)
    """
    from src.training.environments.trading_env import TradingEnvConfig

    # V22: Get action_type and n_actions from SSOT
    action_type = getattr(config, 'action_type', 'continuous')
    n_actions = getattr(config, 'n_actions', 4)

    # Stop mode and action interpretation from SSOT
    stop_mode = getattr(config, 'stop_mode', 'fixed_pct')
    atr_stop = getattr(config, 'atr_stop', {})
    action_interpretation = getattr(config, 'action_interpretation', 'threshold_3')
    zone_5_config = getattr(config, 'zone_5_config', {})

    # EXP-RL-EXECUTOR: Forecast constraint from SSOT
    forecast_constrained = config._raw.get("training", {}).get("environment", {}).get("forecast_constrained", False)

    # Training reads from config.environment, backtest from config.backtest
    if stage == "training":
        env = config.environment
        return TradingEnvConfig(
            episode_length=episode_length,
            warmup_bars=env.warmup_bars,
            initial_balance=env.initial_balance,
            transaction_cost_bps=env.transaction_cost_bps,
            slippage_bps=env.slippage_bps,
            max_drawdown_pct=env.max_drawdown_pct,
            max_position_duration=env.max_position_duration,
            stop_loss_pct=env.stop_loss_pct,
            stop_loss_penalty=env.stop_loss_penalty,
            take_profit_pct=env.take_profit_pct,
            take_profit_bonus=env.take_profit_bonus,
            exit_bonus=env.exit_bonus,
            threshold_long=env.threshold_long,
            threshold_short=env.threshold_short,
            trailing_stop_enabled=env.trailing_stop_enabled,
            trailing_stop_activation_pct=env.trailing_stop_activation_pct,
            trailing_stop_trail_factor=env.trailing_stop_trail_factor,
            trailing_stop_min_trail_pct=env.trailing_stop_min_trail_pct,
            trailing_stop_bonus=env.trailing_stop_bonus,
            reward_interval=env.reward_interval,
            min_hold_bars=env.min_hold_bars,  # V21.1
            observation_dim=len(feature_cols) + get_n_state_features(),
            core_features=feature_cols,
            state_features=get_state_features(),
            action_type=action_type,  # V22: discrete or continuous
            n_actions=n_actions,  # V22: 4 for Discrete(4)
            decision_interval=env.decision_interval,  # EXP-SWING-001
            stop_mode=stop_mode,
            atr_stop=atr_stop,
            action_interpretation=action_interpretation,
            zone_5_config=zone_5_config,
            forecast_constrained=forecast_constrained,  # EXP-RL-EXECUTOR
        )
    else:
        bt = config.backtest
        env = config.environment  # For fields not in BacktestConfig
        return TradingEnvConfig(
            episode_length=episode_length,
            warmup_bars=env.warmup_bars,  # SSOT: same warmup as training
            initial_balance=bt.initial_capital,
            transaction_cost_bps=bt.transaction_cost_bps,
            slippage_bps=bt.slippage_bps,
            max_drawdown_pct=99.0,  # V21.1: No drawdown termination during backtest (evaluate full period)
            max_position_duration=bt.max_position_duration,
            stop_loss_pct=bt.stop_loss_pct,
            stop_loss_penalty=env.stop_loss_penalty,
            take_profit_pct=bt.take_profit_pct,
            take_profit_bonus=env.take_profit_bonus,
            exit_bonus=env.exit_bonus,
            threshold_long=bt.threshold_long,
            threshold_short=bt.threshold_short,
            trailing_stop_enabled=bt.trailing_stop_enabled,
            trailing_stop_activation_pct=bt.trailing_stop_activation_pct,
            trailing_stop_trail_factor=bt.trailing_stop_trail_factor,
            trailing_stop_min_trail_pct=env.trailing_stop_min_trail_pct,
            trailing_stop_bonus=env.trailing_stop_bonus,
            reward_interval=1,  # Backtest: no reward buffering (architectural rule)
            min_hold_bars=env.min_hold_bars,  # V21.1: Same hold period as training
            decision_interval=bt.decision_interval,  # EXP-SWING-001
            observation_dim=len(feature_cols) + get_n_state_features(),
            core_features=feature_cols,
            state_features=get_state_features(),
            action_type=action_type,  # V22: discrete or continuous
            n_actions=n_actions,  # V22: 4 for Discrete(4)
            stop_mode=stop_mode,
            atr_stop=atr_stop,
            action_interpretation=action_interpretation,
            zone_5_config=zone_5_config,
            forecast_constrained=forecast_constrained,  # EXP-RL-EXECUTOR
        )


def load_best_model(l3_result: Dict[str, Any]):
    """Load best_model.zip with correct algorithm class via factory."""
    from src.training.algorithm_factory import create_algorithm

    model_dir = Path(l3_result["model_dir"])
    best_model_path = model_dir / "best_model.zip"
    final_model_path = Path(l3_result["model_path"])

    model_path = best_model_path if best_model_path.exists() else final_model_path
    if not best_model_path.exists():
        logger.warning("best_model.zip not found, using final_model.zip")

    # Resolve algorithm from l3_result (backward compat: fall back to use_lstm)
    algorithm_name = l3_result.get("algorithm", None)
    if algorithm_name is None:
        is_lstm = l3_result.get("use_lstm", False)
        algorithm_name = "recurrent_ppo" if is_lstm else "ppo"

    adapter = create_algorithm(algorithm_name)
    return adapter.load(str(model_path), device="cpu")


def run_backtest_loop(
    model,
    env,
    initial_capital: float,
    log_progress: bool = False,
    is_recurrent: Optional[bool] = None,
) -> List[float]:
    """Run backtest simulation loop, return equity curve.

    V22 FIX #2: Properly tracks LSTM hidden states for RecurrentPPO models.
    Without this, the LSTM resets to zeros every step, defeating temporal memory.

    Args:
        model: Loaded SB3 model
        env: Trading environment
        initial_capital: Starting capital
        log_progress: Log every 5000 steps
        is_recurrent: Override recurrent detection (from l3_result["is_recurrent"])
    """
    obs, _ = env.reset()
    done = False
    equity_curve = [initial_capital]
    step = 0

    # Detect recurrent model (explicit flag > attribute check for robustness)
    if is_recurrent is None:
        is_recurrent = hasattr(model.policy, 'lstm_actor')
    lstm_states = None
    episode_starts = np.array([True]) if is_recurrent else None

    while not done:
        if is_recurrent:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            episode_starts = np.array([False])  # Only True on first step
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        equity_curve.append(float(info.get("equity", info.get("balance", equity_curve[-1]))))
        step += 1

        if log_progress and step % 5000 == 0:
            logger.info(f"  Step {step}, Equity: ${equity_curve[-1]:.2f}")

    return equity_curve


def calculate_backtest_metrics(
    equity_curve: List[float],
    env,
    stage: str = "L4",
) -> Dict[str, Any]:
    """
    Calculate standardized backtest metrics from equity curve and environment.

    Uses SSOT trading_schedule for correct Sharpe annualization.
    Uses portfolio.winning_trades + losing_trades for correct trade counting.
    Uses PnL-based profit factor (not count-based).

    When decision_interval > 1, each equity entry represents multiple raw bars.
    Annualization factors are adjusted accordingly.
    """
    bars_per_year = get_bars_per_year()
    bars_per_day = get_bars_per_day()

    # decision_interval: each equity entry = this many raw bars
    decision_interval = getattr(env.config, 'decision_interval', 1)

    equity = np.array(equity_curve)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-8)

    total_return = float((equity[-1] / equity[0] - 1) * 100)

    # Annualization: entries per year = bars_per_year / decision_interval
    entries_per_year = bars_per_year / decision_interval
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(entries_per_year)) if np.std(returns) > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.maximum(peak, 1e-8)
    max_dd = float(np.max(drawdown) * 100)

    # Trade metrics from portfolio (V21: only closed trades)
    portfolio = env._portfolio
    winning = portfolio.winning_trades
    losing = portfolio.losing_trades
    total_closed = winning + losing
    n_trades = total_closed
    win_rate = float((winning / max(total_closed, 1)) * 100)

    # Profit factor: PnL-based (V21 fix)
    positive_equity_changes = np.sum(returns[returns > 0]) * equity[0]
    negative_equity_changes = abs(np.sum(returns[returns < 0]) * equity[0])
    profit_factor = float(positive_equity_changes / max(negative_equity_changes, 1e-10))

    # Time calculations: multiply entries by decision_interval to get raw bars
    raw_bars = len(equity_curve) * decision_interval
    days = raw_bars / bars_per_day
    years = days / 365
    apr = float((equity[-1] / equity[0]) ** (1 / max(years, 0.01)) - 1) * 100

    metrics = {
        "stage": stage,
        "total_return_pct": total_return,
        "apr_pct": apr,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "n_trades": n_trades,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "final_equity": float(equity[-1]),
        "test_days": days,
        "decision_interval": decision_interval,
    }

    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        metrics["sortino_ratio"] = float(
            np.mean(returns) / np.std(downside_returns) * np.sqrt(entries_per_year)
        )

    return metrics


def log_backtest_results(metrics: Dict[str, Any], stage: str) -> None:
    """Log backtest results in standardized format."""
    logger.info("=" * 60)
    logger.info(f"{stage} RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total Return: {metrics['total_return_pct']:+.2f}%")
    if "apr_pct" in metrics:
        logger.info(f"  APR: {metrics['apr_pct']:+.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    if "sortino_ratio" in metrics:
        logger.info(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    logger.info(f"  Trades: {metrics['n_trades']} (W:{metrics['winning_trades']} L:{metrics['losing_trades']})")
    logger.info(f"  Win Rate: {metrics['win_rate_pct']:.1f}%")
    logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info("=" * 60)


def validate_gates(metrics: Dict[str, Any], config) -> Tuple[bool, Dict[str, Any]]:
    """Validate backtest metrics against SSOT gates. Returns (passed, gate_results)."""
    gates = config.backtest
    checks = [
        ("min_return_pct", metrics["total_return_pct"] >= gates.min_return_pct,
         metrics["total_return_pct"], gates.min_return_pct),
        ("min_sharpe_ratio", metrics["sharpe_ratio"] >= gates.min_sharpe_ratio,
         metrics["sharpe_ratio"], gates.min_sharpe_ratio),
        ("max_drawdown_pct", metrics["max_drawdown_pct"] <= gates.max_drawdown_pct,
         metrics["max_drawdown_pct"], gates.max_drawdown_pct),
        ("min_trades", metrics["n_trades"] >= gates.min_trades,
         metrics["n_trades"], gates.min_trades),
        ("min_win_rate", metrics["win_rate_pct"] >= gates.min_win_rate * 100,
         metrics["win_rate_pct"], gates.min_win_rate * 100),
    ]

    passed = True
    gate_results = {}

    logger.info("VALIDATION GATES:")
    for name, check, actual, threshold in checks:
        status = "PASS" if check else "FAIL"
        logger.info(f"  {name}: {status} (actual={actual:.2f}, threshold={threshold:.2f})")
        gate_results[name] = {
            "passed": bool(check),
            "actual": float(actual),
            "threshold": float(threshold),
        }
        if not check:
            passed = False

    return passed, gate_results


# =============================================================================
# L2: DATASET BUILDING
# =============================================================================

def run_l2_dataset_build(output_dir: Path) -> Dict[str, Any]:
    """
    L2: Build dataset from SSOT configuration.

    Returns:
        Dictionary with dataset info and paths
    """
    logger.info("=" * 70)
    logger.info("L2: BUILDING DATASET FROM SSOT")
    logger.info("=" * 70)

    from src.config.pipeline_config import load_pipeline_config
    from src.data.ssot_dataset_builder import SSOTDatasetBuilder

    config = load_pipeline_config()

    # Load OHLCV data (path from SSOT config)
    logger.info("Loading OHLCV data...")
    ohlcv_seed = config._raw.get("paths", {}).get("sources", {}).get("ohlcv_seed", "seeds/latest/usdcop_m5_ohlcv.parquet")
    ohlcv_path = PROJECT_ROOT / ohlcv_seed
    df_ohlcv = pd.read_parquet(ohlcv_path)

    # Ensure datetime index
    if 'time' in df_ohlcv.columns:
        df_ohlcv['time'] = pd.to_datetime(df_ohlcv['time'])
        df_ohlcv = df_ohlcv.set_index('time')

    logger.info(f"  OHLCV: {len(df_ohlcv)} rows, {df_ohlcv.index.min()} to {df_ohlcv.index.max()}")

    # Load and map macro data
    logger.info("Loading macro data...")
    macro_path = PROJECT_ROOT / "data/pipeline/04_cleaning/output/macro_daily_clean.parquet"
    df_macro_raw = pd.read_parquet(macro_path)

    # Rename columns to SSOT names (mapping from pipeline_ssot.yaml)
    macro_col_map = get_macro_column_map()
    df_macro = df_macro_raw.rename(columns=macro_col_map)

    # Verify required columns (derived from SSOT feature input_columns)
    required_macro = list(set(macro_col_map.values()))
    missing = [c for c in required_macro if c not in df_macro.columns]
    if missing:
        logger.warning(f"Missing macro columns: {missing}")

    logger.info(f"  Macro: {len(df_macro)} rows, columns: {[c for c in df_macro.columns if c in required_macro]}")

    # Load auxiliary pair data if enabled in SSOT
    df_aux_ohlcv = {}
    aux_config = config._raw.get("aux_pairs", {})
    if aux_config.get("enabled", False):
        logger.info("Loading auxiliary pair data (aux_pairs enabled)...")
        for pair_key, pair_info in aux_config.get("pairs", {}).items():
            seed_path = PROJECT_ROOT / pair_info["seed_file"]
            if seed_path.exists():
                df_aux = pd.read_parquet(seed_path)
                if 'time' in df_aux.columns:
                    df_aux['time'] = pd.to_datetime(df_aux['time'])
                    df_aux = df_aux.rename(columns={'time': 'datetime'}).set_index('datetime')
                if df_aux.index.tz is not None:
                    df_aux.index = df_aux.index.tz_localize(None)
                df_aux = df_aux.sort_index()
                df_aux_ohlcv[pair_key] = df_aux
                logger.info(f"  Aux {pair_key}: {len(df_aux)} rows ({df_aux.index.min()} → {df_aux.index.max()})")
            else:
                logger.warning(f"  Aux {pair_key}: seed file not found at {seed_path}")

    # Build dataset using SSOT (builder handles macro reindexing efficiently)
    logger.info("Building dataset with SSOTDatasetBuilder...")
    builder = SSOTDatasetBuilder()

    output_dir.mkdir(parents=True, exist_ok=True)
    result = builder.build(
        df_ohlcv=df_ohlcv,
        df_macro=df_macro,  # Pass daily macro, builder handles reindex to 5min
        output_dir=output_dir,
        dataset_prefix="DS_production",
        df_aux_ohlcv=df_aux_ohlcv if df_aux_ohlcv else None,
    )

    # Summary
    logger.info("L2 COMPLETE:")
    logger.info(f"  Train: {len(result.train_df)} rows ({result.train_df.index.min()} → {result.train_df.index.max()})")
    logger.info(f"  Val: {len(result.val_df)} rows ({result.val_df.index.min()} → {result.val_df.index.max()})")
    logger.info(f"  Test: {len(result.test_df)} rows ({result.test_df.index.min()} → {result.test_df.index.max()})")
    logger.info(f"  Features: {len(result.feature_columns)} → {result.feature_columns}")
    logger.info(f"  Observation dim: {result.observation_dim} market + state features from SSOT")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Descriptive stats: {output_dir / 'DS_production_descriptive_stats.json'}")

    return {
        "train_rows": len(result.train_df),
        "val_rows": len(result.val_df),
        "test_rows": len(result.test_df),
        "feature_columns": result.feature_columns,
        "observation_dim": result.observation_dim,
        "norm_stats_path": str(output_dir / "DS_production_norm_stats.json"),
        "train_path": str(output_dir / "DS_production_train.parquet"),
        "val_path": str(output_dir / "DS_production_val.parquet"),
        "test_path": str(output_dir / "DS_production_test.parquet"),
        "lineage": result.lineage,
    }


# =============================================================================
# L3: MODEL TRAINING
# =============================================================================

def run_l3_training(
    l2_result: Dict[str, Any],
    models_dir: Path,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    L3: Train PPO model using SSOT configuration.

    Args:
        l2_result: Output from L2 dataset build
        models_dir: Directory to save trained model
        seed: Random seed for reproducibility

    Returns:
        Dictionary with training results and model path
    """
    logger.info("=" * 70)
    logger.info("L3: TRAINING MODEL WITH SSOT CONFIG")
    logger.info("=" * 70)

    from src.config.pipeline_config import load_pipeline_config
    from src.training.algorithm_factory import create_algorithm, get_algorithm_kwargs, resolve_algorithm_name
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    config = load_pipeline_config()

    # Load datasets
    logger.info("Loading L2 datasets...")
    train_df = pd.read_parquet(l2_result["train_path"])
    val_df = pd.read_parquet(l2_result["val_path"])

    with open(l2_result["norm_stats_path"], 'r') as f:
        norm_stats = json.load(f)

    logger.info(f"  Train: {len(train_df)} rows")
    logger.info(f"  Val: {len(val_df)} rows")

    # Create environment
    logger.info("Creating trading environment...")
    from src.training.environments.trading_env import TradingEnvironment

    feature_cols = l2_result["feature_columns"]
    env_cfg = create_env_config(config, feature_cols, config.environment.episode_length, stage="training")

    # EXP-RL-EXECUTOR: Load forecast signals if available
    fc_signals = _FORECAST_SIGNALS  # Module-level cache (loaded in main())

    def make_env():
        return TradingEnvironment(df=train_df, norm_stats=norm_stats, config=env_cfg,
                                  forecast_signals=fc_signals)

    train_env = DummyVecEnv([make_env])

    # Create eval environment with same config but different data
    def make_eval_env():
        return TradingEnvironment(df=val_df, norm_stats=norm_stats, config=env_cfg,
                                  forecast_signals=fc_signals)

    eval_env = DummyVecEnv([make_eval_env])

    # Resolve algorithm and get hyperparameters from SSOT
    algorithm_name = resolve_algorithm_name(config)
    algo_kwargs = get_algorithm_kwargs(config)

    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Hyperparameters: lr={algo_kwargs.get('learning_rate')}, "
                f"gamma={algo_kwargs.get('gamma')}, "
                f"batch_size={algo_kwargs.get('batch_size')}, "
                f"ent_coef={algo_kwargs.get('ent_coef')}")

    # Set reproducible seeds before model creation
    set_reproducible_seeds(seed)

    # Create algorithm adapter and model via factory
    adapter_kwargs = {}
    if algorithm_name == "recurrent_ppo":
        adapter_kwargs["lstm_hidden_size"] = config.lstm.hidden_size
        adapter_kwargs["n_lstm_layers"] = config.lstm.n_layers

    adapter = create_algorithm(algorithm_name, **adapter_kwargs)

    # Common kwargs for all algorithms
    algo_kwargs["seed"] = seed
    device_config = config._raw.get("training", {}).get("device", None)
    algo_kwargs["device"] = device_config if device_config else ('cuda' if torch.cuda.is_available() else 'cpu')
    algo_kwargs["verbose"] = 1
    algo_kwargs["tensorboard_log"] = str(PROJECT_ROOT / "logs/tensorboard")

    model = adapter.create(train_env, **algo_kwargs)

    # Create model directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = models_dir / f"{algorithm_name}_ssot_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    schedule = config.get_training_schedule()
    total_timesteps = schedule.get("total_timesteps", 500000)
    eval_freq = schedule.get("eval_freq", 25000)

    eval_callback = DeterministicEvalCallback(
        eval_env,
        eval_seed=seed,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "logs"),
        eval_freq=eval_freq,
        n_eval_episodes=schedule.get("n_eval_episodes", 10),
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=schedule.get("checkpoint_freq", 50000),
        save_path=str(model_dir / "checkpoints"),
        name_prefix=f"{algorithm_name}_ssot",
    )

    # Train
    logger.info(f"Training for {total_timesteps} timesteps...")
    logger.info(f"Model will be saved to: {model_dir}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_model_path = model_dir / "final_model.zip"
    model.save(str(final_model_path))

    # Save norm stats with model
    with open(model_dir / "norm_stats.json", 'w') as f:
        json.dump(norm_stats, f, indent=2)

    # Save training config
    training_config = {
        "ssot_version": config.version,
        "based_on_model": config.based_on_model,
        "feature_columns": feature_cols,
        "observation_dim": len(feature_cols) + get_n_state_features(),
        "ppo_config": {
            "learning_rate": config.ppo.learning_rate,
            "gamma": config.ppo.gamma,
            "n_steps": config.ppo.n_steps,
            "batch_size": config.ppo.batch_size,
            "ent_coef": config.ppo.ent_coef,
        },
        "environment_config": {
            "transaction_cost_bps": config.environment.transaction_cost_bps,
            "stop_loss_pct": config.environment.stop_loss_pct,
            "take_profit_pct": config.environment.take_profit_pct,
            "trailing_stop_enabled": config.environment.trailing_stop_enabled,
        },
        "total_timesteps": total_timesteps,
        "timestamp": timestamp,
    }

    with open(model_dir / "training_config.json", 'w') as f:
        json.dump(training_config, f, indent=2)

    logger.info("L3 COMPLETE:")
    logger.info(f"  Model saved: {final_model_path}")
    logger.info(f"  Norm stats: {model_dir / 'norm_stats.json'}")

    return {
        "model_path": str(final_model_path),
        "model_dir": str(model_dir),
        "norm_stats_path": str(model_dir / "norm_stats.json"),
        "training_config": training_config,
        "timestamp": timestamp,
        "algorithm": algorithm_name,
        "is_recurrent": adapter.is_recurrent(),
        "use_lstm": adapter.is_recurrent(),  # Backward compat
    }


# =============================================================================
# L3-MULTI: MULTI-SEED TRAINING
# =============================================================================

def run_l3_multi_seed(
    l2_result: Dict[str, Any],
    models_dir: Path,
    seeds: List[int] = None,
) -> Dict[str, Any]:
    """
    L3-MULTI: Train with multiple seeds and select best model.

    Args:
        l2_result: Output from L2 dataset build
        models_dir: Directory to save trained models
        seeds: List of seeds to train with

    Returns:
        Dictionary with best model and variance statistics
    """
    logger.info("=" * 70)
    logger.info("L3-MULTI: MULTI-SEED TRAINING")
    logger.info("=" * 70)

    seeds = seeds or [42, 123, 456, 789, 1337]
    logger.info(f"Training with seeds: {seeds}")

    results = {}
    for i, seed in enumerate(seeds):
        logger.info("-" * 50)
        logger.info(f"Training seed {seed} ({i+1}/{len(seeds)})")
        logger.info("-" * 50)

        try:
            result = run_l3_training(l2_result, models_dir, seed=seed)
            results[seed] = result
            logger.info(f"  Seed {seed}: Complete, model at {result['model_path']}")
        except Exception as e:
            logger.error(f"  Seed {seed}: FAILED - {e}")
            results[seed] = {"success": False, "error": str(e)}

    # Analyze results - load evaluations to get rewards
    successful_seeds = [s for s, r in results.items() if r.get("model_path")]

    if not successful_seeds:
        logger.error("All seeds failed!")
        return {"success": False, "error": "All seeds failed"}

    # Load eval results for each seed to get best_mean_reward
    seed_rewards = {}
    for seed in successful_seeds:
        model_dir = Path(results[seed]["model_dir"])
        eval_log = model_dir / "logs" / "evaluations.npz"
        if eval_log.exists():
            data = np.load(eval_log)
            seed_rewards[seed] = float(np.max(data["results"]))
        else:
            # Fallback: use final training metric if available
            seed_rewards[seed] = 0.0

    # Select best seed
    best_seed = max(seed_rewards, key=lambda s: seed_rewards[s])
    best_result = results[best_seed]

    # Calculate statistics
    rewards = list(seed_rewards.values())
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    cv_reward = std_reward / abs(mean_reward) if abs(mean_reward) > 1e-8 else float('inf')

    logger.info("=" * 70)
    logger.info("MULTI-SEED RESULTS")
    logger.info("=" * 70)
    logger.info(f"Best seed: {best_seed} (reward={seed_rewards[best_seed]:.4f})")
    logger.info(f"Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")
    logger.info(f"CV: {cv_reward:.2%}")

    if cv_reward > 0.30:
        logger.warning(f"HIGH VARIANCE WARNING: CV={cv_reward:.2%} > 30%")

    return {
        "best_seed": best_seed,
        "best_model_path": best_result["model_path"],
        "best_model_dir": best_result["model_dir"],
        "model_path": best_result["model_path"],
        "model_dir": best_result["model_dir"],
        "norm_stats_path": best_result["norm_stats_path"],
        "timestamp": best_result["timestamp"],
        "training_config": best_result["training_config"],
        "use_lstm": best_result.get("use_lstm", False),  # V22: propagate for L4 model loading
        "multi_seed_stats": {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "cv_reward": cv_reward,
            "seed_rewards": seed_rewards,
        },
        "all_results": {str(s): r for s, r in results.items()},
    }


# =============================================================================
# L4: BACKTESTING
# =============================================================================

def run_l4_backtest(
    l2_result: Dict[str, Any],
    l3_result: Dict[str, Any],
    results_dir: Path
) -> Dict[str, Any]:
    """
    L4: Backtest trained model on test set.

    Args:
        l2_result: Output from L2 dataset build
        l3_result: Output from L3 training
        results_dir: Directory to save backtest results

    Returns:
        Dictionary with backtest metrics
    """
    logger.info("=" * 70)
    logger.info("L4: BACKTESTING MODEL")
    logger.info("=" * 70)

    from src.config.pipeline_config import load_pipeline_config
    from src.training.environments.trading_env import TradingEnvironment

    config = load_pipeline_config()
    model = load_best_model(l3_result)

    # Load val + test data (combined for full 2025 backtest coverage)
    val_df = pd.read_parquet(l2_result["val_path"])
    test_df = pd.read_parquet(l2_result["test_path"])
    backtest_df = pd.concat([val_df, test_df]).sort_index()
    backtest_df = backtest_df[~backtest_df.index.duplicated(keep='first')]

    with open(l3_result["norm_stats_path"], 'r') as f:
        norm_stats = json.load(f)

    logger.info(f"  Combined: {len(backtest_df)} rows (V:{len(val_df)} + T:{len(test_df)})")

    # Verify parity
    errors = config.validate_training_backtest_parity()
    if errors:
        logger.warning(f"Parity errors: {errors}")
    else:
        logger.info("  Parity: VERIFIED")

    feature_cols = l2_result["feature_columns"]
    warmup = config.environment.warmup_bars
    bt_env_cfg = create_env_config(config, feature_cols, len(backtest_df) - warmup - get_backtest_margin(), stage="backtest")
    env = TradingEnvironment(df=backtest_df, norm_stats=norm_stats, config=bt_env_cfg,
                             forecast_signals=_FORECAST_SIGNALS)

    equity_curve = run_backtest_loop(model, env, config.backtest.initial_capital, log_progress=True)
    metrics = calculate_backtest_metrics(equity_curve, env, stage="L4-BACKTEST")
    log_backtest_results(metrics, "L4 BACKTEST")

    passed, gate_results = validate_gates(metrics, config)
    metrics["gates_passed"] = bool(passed)
    metrics["gate_results"] = gate_results

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = l3_result["timestamp"]
    with open(results_dir / f"backtest_{timestamp}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"equity": equity_curve}).to_parquet(results_dir / f"equity_curve_{timestamp}.parquet")

    return metrics


# =============================================================================
# L4-VAL: VALIDATION BACKTEST (Gate for promotion)
# =============================================================================

def run_l4_validation(
    l2_result: Dict[str, Any],
    l3_result: Dict[str, Any],
    results_dir: Path
) -> Dict[str, Any]:
    """
    L4-VAL: Backtest on VALIDATION set only.

    This is the GATE for model promotion. Must pass before test.

    Args:
        l2_result: Output from L2 dataset build
        l3_result: Output from L3 training
        results_dir: Directory to save backtest results

    Returns:
        Dictionary with validation metrics and pass/fail status
    """
    logger.info("=" * 70)
    logger.info("L4-VAL: VALIDATION BACKTEST")
    logger.info("=" * 70)

    from src.config.pipeline_config import load_pipeline_config
    from src.training.environments.trading_env import TradingEnvironment

    config = load_pipeline_config()
    model = load_best_model(l3_result)

    val_df = pd.read_parquet(l2_result["val_path"])
    with open(l3_result["norm_stats_path"], 'r') as f:
        norm_stats = json.load(f)
    logger.info(f"  Val: {len(val_df)} rows")

    feature_cols = l2_result["feature_columns"]
    warmup = config.environment.warmup_bars
    bt_env_cfg = create_env_config(config, feature_cols, len(val_df) - warmup - get_backtest_margin(), stage="backtest")
    env = TradingEnvironment(df=val_df, norm_stats=norm_stats, config=bt_env_cfg,
                             forecast_signals=_FORECAST_SIGNALS)

    equity_curve = run_backtest_loop(model, env, config.backtest.initial_capital)
    metrics = calculate_backtest_metrics(equity_curve, env, stage="L4-VAL")
    log_backtest_results(metrics, "L4-VAL")

    passed, gate_results = validate_gates(metrics, config)
    metrics["gates_passed"] = passed

    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = l3_result["timestamp"]
    with open(results_dir / f"l4_val_{timestamp}.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"L4-VAL: {'PASSED' if passed else 'FAILED'}")
    return metrics


def run_l4_test(
    l2_result: Dict[str, Any],
    l3_result: Dict[str, Any],
    results_dir: Path
) -> Dict[str, Any]:
    """
    L4-TEST: Combined backtest on VAL+TEST (full OOS period).

    Runs the model on the combined validation + test dataset to produce a single
    unified backtest covering the entire out-of-sample period (e.g. full year 2025).

    Args:
        l2_result: Output from L2 dataset build
        l3_result: Output from L3 training
        results_dir: Directory to save backtest results

    Returns:
        Dictionary with test metrics
    """
    logger.info("=" * 70)
    logger.info("L4-TEST: COMBINED OOS BACKTEST (VAL + TEST)")
    logger.info("=" * 70)

    from src.config.pipeline_config import load_pipeline_config
    from src.training.environments.trading_env import TradingEnvironment

    config = load_pipeline_config()
    model = load_best_model(l3_result)

    # Load and combine VAL + TEST data for unified OOS backtest
    logger.info("Loading VAL + TEST data (combined OOS)...")
    val_df = pd.read_parquet(l2_result["val_path"])
    test_df = pd.read_parquet(l2_result["test_path"])
    combined_df = pd.concat([val_df, test_df]).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    # Enforce test_end cutoff from SSOT config
    test_end = pd.Timestamp(config.date_ranges.test_end)
    if combined_df.index[-1] > test_end:
        pre_filter = len(combined_df)
        combined_df = combined_df[combined_df.index <= test_end]
        logger.info(f"  Filtered to test_end={config.date_ranges.test_end}: {pre_filter} -> {len(combined_df)} rows")

    with open(l3_result["norm_stats_path"], 'r') as f:
        norm_stats = json.load(f)

    logger.info(f"  Val: {len(val_df)} rows ({val_df.index[0]} -> {val_df.index[-1]})")
    logger.info(f"  Test: {len(test_df)} rows ({test_df.index[0]} -> {test_df.index[-1]})")
    logger.info(f"  Combined: {len(combined_df)} rows ({combined_df.index[0]} -> {combined_df.index[-1]})")

    # Create env using shared factory (DRY: eliminates hardcoded values)
    feature_cols = l2_result["feature_columns"]
    warmup = config.environment.warmup_bars
    bt_env_cfg = create_env_config(config, feature_cols, len(combined_df) - warmup - get_backtest_margin(), stage="backtest")
    env = TradingEnvironment(df=combined_df, norm_stats=norm_stats, config=bt_env_cfg,
                             forecast_signals=_FORECAST_SIGNALS)

    # Run backtest using shared loop
    equity_curve = run_backtest_loop(model, env, config.backtest.initial_capital, log_progress=True)

    # Calculate metrics using shared function
    metrics = calculate_backtest_metrics(equity_curve, env, stage="L4-TEST")
    metrics["period_start"] = str(combined_df.index[0])
    metrics["period_end"] = str(combined_df.index[-1])

    # Monthly returns (unique to L4-TEST for detailed reporting)
    idx = combined_df.index[:len(equity_curve)]
    equity_series = pd.Series(equity_curve, index=idx)
    monthly = equity_series.resample('ME').last()
    monthly_returns = monthly.pct_change().dropna() * 100
    monthly_dict = {d.strftime('%Y-%m'): round(float(r), 2) for d, r in monthly_returns.items()}
    metrics["monthly_returns"] = monthly_dict

    # Log results using shared logger + test-specific details
    log_backtest_results(metrics, "L4-TEST (COMBINED OOS)")

    logger.info("  Monthly Returns:")
    for month_str, ret in monthly_dict.items():
        logger.info(f"    {month_str}: {ret:+7.2f}%")

    logger.info("  Equity Curve:")
    equity = np.array(equity_curve)
    for month_key in sorted(set(idx.strftime('%Y-%m'))):
        mask = equity_series.index.strftime('%Y-%m') == month_key
        if mask.any():
            last_val = equity_series[mask].iloc[-1]
            ret_from_start = (last_val / equity[0] - 1) * 100
            logger.info(f"    {month_key}: ${last_val:>10,.0f}  ({ret_from_start:+.2f}%)")

    # Validate gates
    passed, gate_results = validate_gates(metrics, config)
    metrics["gates_passed"] = bool(passed)
    metrics["gate_results"] = gate_results

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = l3_result["timestamp"]

    results_file = results_dir / f"l4_test_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    equity_df = pd.DataFrame({"equity": equity_curve}, index=idx)
    equity_df.to_parquet(results_dir / f"equity_curve_test_{timestamp}.parquet")

    logger.info(f"Results saved to: {results_file}")
    logger.info(f"L4-TEST: {'PASSED' if passed else 'FAILED'}")

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run SSOT Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "l2", "l3", "l4", "l4-val", "l4-test"],
        default="all",
        help="Stage to run (default: all). Use l4-val for validation only, l4-test for test only."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pipeline/07_output/5min",
        help="Output directory for L2 datasets"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory for trained models"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/backtests",
        help="Directory for backtest results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for training (default: {SEED})"
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Enable multi-seed training (trains with 5 seeds, selects best)"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456,789,1337",
        help="Comma-separated list of seeds for multi-seed training"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID from EXPERIMENT_QUEUE.md (e.g., EXP-V215c-001)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment SSOT config (e.g., config/experiments/exp_hourly_ppo_001.yaml)"
    )
    parser.add_argument(
        "--forecast-signals",
        type=str,
        default=None,
        help="Path to historical forecast signals parquet (for EXP-RL-EXECUTOR)"
    )
    parser.add_argument(
        "--seed-db", action="store_true",
        help="Seed DB tables with results (RL: not yet implemented, accepted for compatibility)"
    )
    args = parser.parse_args()

    if args.seed_db:
        logger.warning("[RL] --seed-db flag received but RL DB seeding is not yet implemented. "
                        "Flag accepted without error for deploy API compatibility.")

    # Apply --config: set env var so load_pipeline_config() uses it
    if args.config:
        import os
        config_path = str(PROJECT_ROOT / args.config) if not Path(args.config).is_absolute() else args.config
        os.environ["PIPELINE_SSOT_PATH"] = config_path
        # Invalidate cached constants so they reload from new config
        global _SSOT_CONSTANTS
        _SSOT_CONSTANTS = None
        logger.info(f"Using experiment config: {config_path}")

    # EXP-RL-EXECUTOR: Load forecast signals if provided
    if args.forecast_signals:
        fc_path = str(PROJECT_ROOT / args.forecast_signals) if not Path(args.forecast_signals).is_absolute() else args.forecast_signals
        load_forecast_signals(fc_path)
        logger.info(f"Forecast signals loaded: {len(_FORECAST_SIGNALS)} days")

    output_dir = PROJECT_ROOT / args.output_dir
    models_dir = PROJECT_ROOT / args.models_dir
    results_dir = PROJECT_ROOT / args.results_dir

    # Parse multi-seed list
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]

    # Set reproducible seeds FIRST
    set_reproducible_seeds(args.seed)

    # Initialize run manifest (optional — pipeline works without it)
    manifest = None
    if MANIFEST_AVAILABLE:
        try:
            manifest = RunManifest(
                experiment_id=args.experiment_id,
                seed=args.seed,
                multi_seed=args.multi_seed,
                seeds=seed_list if args.multi_seed else [args.seed],
            )
            logger.info(f"Run ID: {manifest.run_id}")
        except Exception as e:
            logger.warning(f"RunManifest init failed: {e}")

    logger.info("=" * 70)
    logger.info("SSOT PIPELINE RUNNER")
    logger.info("=" * 70)
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Seed: {args.seed} (reproducible)")
    logger.info(f"Multi-seed: {args.multi_seed}")
    if args.multi_seed:
        logger.info(f"Seeds: {seed_list}")
    if _FORECAST_SIGNALS:
        logger.info(f"Forecast signals: {len(_FORECAST_SIGNALS)} days (EXP-RL-EXECUTOR)")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Models dir: {models_dir}")
    logger.info(f"Results dir: {results_dir}")

    results = {}

    # L2: Dataset Building
    if args.stage in ["all", "l2"]:
        results["l2"] = run_l2_dataset_build(output_dir)

        # Save L2 result for later stages
        with open(output_dir / "l2_result.json", 'w') as f:
            json.dump(results["l2"], f, indent=2)

        if manifest:
            try:
                manifest.record_l2(results["l2"])
            except Exception as e:
                logger.warning(f"Manifest L2 record failed: {e}")

    # Load L2 result if running L3 or L4 separately
    if args.stage in ["l3", "l4", "l4-val", "l4-test"] and "l2" not in results:
        l2_result_path = output_dir / "l2_result.json"
        if l2_result_path.exists():
            with open(l2_result_path, 'r') as f:
                results["l2"] = json.load(f)
        else:
            logger.error(f"L2 result not found at {l2_result_path}. Run L2 first.")
            return 1

    # L3: Training (with optional multi-seed)
    if args.stage in ["all", "l3"]:
        if args.multi_seed:
            results["l3"] = run_l3_multi_seed(results["l2"], models_dir, seeds=seed_list)
        else:
            results["l3"] = run_l3_training(results["l2"], models_dir, seed=args.seed)

        # Save L3 result for later stages
        with open(models_dir / "l3_result.json", 'w') as f:
            json.dump(results["l3"], f, indent=2)

        if manifest:
            try:
                manifest.record_l3(results["l3"])
            except Exception as e:
                logger.warning(f"Manifest L3 record failed: {e}")

    # Load L3 result if running L4 separately
    if args.stage in ["l4", "l4-val", "l4-test"] and "l3" not in results:
        l3_result_path = models_dir / "l3_result.json"
        if l3_result_path.exists():
            with open(l3_result_path, 'r') as f:
                results["l3"] = json.load(f)
        else:
            logger.error(f"L3 result not found at {l3_result_path}. Run L3 first.")
            return 1

    # L4: Backtesting (with val/test separation)
    if args.stage == "l4-val":
        # Validation only
        results["l4_val"] = run_l4_validation(results["l2"], results["l3"], results_dir)
        if manifest:
            try:
                manifest.record_l4(results["l4_val"], "l4_val")
            except Exception as e:
                logger.warning(f"Manifest L4-VAL record failed: {e}")

    elif args.stage == "l4-test":
        # Test only (should only run after validation passes)
        results["l4_test"] = run_l4_test(results["l2"], results["l3"], results_dir)
        if manifest:
            try:
                manifest.record_l4(results["l4_test"], "l4_test")
            except Exception as e:
                logger.warning(f"Manifest L4-TEST record failed: {e}")

    elif args.stage in ["all", "l4"]:
        # Single unified L4: Backtest over FULL 2025 (val + test combined)
        results["l4_test"] = run_l4_test(results["l2"], results["l3"], results_dir)
        if manifest:
            try:
                manifest.record_l4(results["l4_test"], "l4_test")
            except Exception as e:
                logger.warning(f"Manifest L4-TEST record failed: {e}")

        # Apply gates on full 2025 result
        from src.config.pipeline_config import load_pipeline_config
        bt_config = load_pipeline_config().backtest
        r = results["l4_test"]

        gates_passed = (
            r["total_return_pct"] >= bt_config.min_return_pct
            and r["sharpe_ratio"] >= bt_config.min_sharpe_ratio
            and r["max_drawdown_pct"] <= bt_config.max_drawdown_pct
            and r["n_trades"] >= bt_config.min_trades
        )

        logger.info("L4 GATES (Full 2025):")
        logger.info(f"  min_return:  {'PASS' if r['total_return_pct'] >= bt_config.min_return_pct else 'FAIL'} ({r['total_return_pct']:.2f}% >= {bt_config.min_return_pct}%)")
        logger.info(f"  min_sharpe:  {'PASS' if r['sharpe_ratio'] >= bt_config.min_sharpe_ratio else 'FAIL'} ({r['sharpe_ratio']:.3f} >= {bt_config.min_sharpe_ratio})")
        logger.info(f"  max_dd:      {'PASS' if r['max_drawdown_pct'] <= bt_config.max_drawdown_pct else 'FAIL'} ({r['max_drawdown_pct']:.2f}% <= {bt_config.max_drawdown_pct}%)")
        logger.info(f"  min_trades:  {'PASS' if r['n_trades'] >= bt_config.min_trades else 'FAIL'} ({r['n_trades']} >= {bt_config.min_trades})")
        logger.info(f"  L4 GATES: {'PASSED' if gates_passed else 'FAILED'}")

        results["l4"] = {
            "test": results["l4_test"],
            "gates_passed": gates_passed,
            "total_return_pct": r["total_return_pct"],
            "sharpe_ratio": r["sharpe_ratio"],
        }

    # Final summary
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    if "l2" in results:
        logger.info(f"L2: {results['l2']['train_rows']} train, {results['l2']['test_rows']} test")
    if "l3" in results:
        logger.info(f"L3: Model at {results['l3']['model_path']}")
        if "multi_seed_stats" in results["l3"]:
            stats = results["l3"]["multi_seed_stats"]
            logger.info(f"    Multi-seed CV: {stats['cv_reward']:.2%}")
            logger.info(f"    Best seed: {results['l3']['best_seed']}")
    if "l4_test" in results:
        r = results['l4_test']
        logger.info(f"L4 (Full 2025): Return={r['total_return_pct']:.2f}%, Sharpe={r['sharpe_ratio']:.3f}, WinRate={r.get('win_rate_pct', 0):.1f}%, MaxDD={r.get('max_drawdown_pct', 0):.1f}%")
    if "l4" in results:
        logger.info(f"L4: Return={results['l4']['total_return_pct']:.2f}%, Sharpe={results['l4']['sharpe_ratio']:.3f}")
        logger.info(f"    Gates: {'PASSED' if results['l4']['gates_passed'] else 'FAILED'}")

    # Save run manifest
    if manifest:
        try:
            manifest.compute_lineage()
            manifest_path = manifest.save(PROJECT_ROOT / "results" / "runs")
            logger.info(f"Run manifest: {manifest_path}")
        except Exception as e:
            logger.warning(f"Manifest save failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
