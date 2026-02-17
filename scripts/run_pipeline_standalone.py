"""
Standalone L2-L3-L4 Pipeline - Self-contained, no heavy src imports.
====================================================================
Bypasses src/__init__.py import chains that hang on Windows.
Directly loads YAML config and uses minimal dependencies.

Usage:
    python scripts/run_pipeline_standalone.py
    python scripts/run_pipeline_standalone.py --timesteps 300000
    python scripts/run_pipeline_standalone.py --seed 42 --verify-reproducibility

Author: Claude Code
Date: 2026-02-05
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# ============================================================================
# SETUP
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def set_seeds(seed: int) -> None:
    """Set ALL random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================================
# FEATURE CALCULATORS (standalone, no registry needed)
# ============================================================================

def calc_log_return(s: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(s / s.shift(periods))


def calc_rsi(s: pd.Series, period: int = 9) -> pd.Series:
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.clip(lower=1e-10)
    return (100.0 - (100.0 / (1.0 + rs))).clip(0, 100)


def calc_volatility(s: pd.Series, period: int = 14, bars_per_day: int = 48) -> pd.Series:
    log_ret = np.log(s / s.shift(1))
    rolling_std = log_ret.rolling(window=period, min_periods=period // 2).std()
    return rolling_std * np.sqrt(252 * bars_per_day)


def calc_trend_z(s: pd.Series, sma_period: int = 50) -> pd.Series:
    sma = s.rolling(window=sma_period, min_periods=sma_period // 2).mean()
    std = s.rolling(window=sma_period, min_periods=sma_period // 2).std()
    return ((s - sma) / std.clip(lower=1e-6)).clip(-3, 3)


def calc_macro_zscore(s: pd.Series, window: int = 252) -> pd.Series:
    mean = s.rolling(window=window, min_periods=window // 4).mean()
    std = s.rolling(window=window, min_periods=window // 4).std()
    return ((s - mean) / std.clip(lower=1e-8)).clip(-5, 5)


def calc_spread_zscore(s1: pd.Series, s2: pd.Series, window: int = 252) -> pd.Series:
    spread = s1 - s2
    mean = spread.rolling(window=window, min_periods=window // 4).mean()
    std = spread.rolling(window=window, min_periods=window // 4).std()
    return ((spread - mean) / std.clip(lower=1e-8)).clip(-5, 5)


# ============================================================================
# MACRO COLUMN MAPPING
# ============================================================================
MACRO_MAP = {
    "FXRT_INDEX_DXY_USA_D_DXY": "dxy",
    "VOLT_VIX_USA_D_VIX": "vix",
    "CRSK_SPREAD_EMBI_COL_D_EMBI": "embi",
    "COMM_OIL_BRENT_GLB_D_BRENT": "brent",
    "COMM_METAL_GOLD_GLB_D_GOLD": "gold",
    "FINC_BOND_YIELD10Y_COL_D_COL10Y": "col10y",
    "FINC_BOND_YIELD10Y_USA_D_UST10Y": "ust10y",
    "FINC_BOND_YIELD2Y_USA_D_DGS2": "ust2y",
    "FXRT_SPOT_USDMXN_MEX_D_USDMXN": "usdmxn",
}

FEATURE_ORDER = [
    "log_ret_5m", "log_ret_1h", "log_ret_4h", "log_ret_1d",
    "rsi_9", "rsi_21", "volatility_pct", "trend_z",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
    "brent_change_1d", "gold_change_1d",
    "rate_spread_z", "rate_spread_change",
    "usdmxn_change_1d", "yield_curve_z",
]


# ============================================================================
# L2: DATASET BUILD
# ============================================================================

def run_l2(cfg: dict, output_dir: Path) -> Dict[str, Any]:
    """Build train/val/test datasets from OHLCV + macro data."""
    logger.info("=" * 60)
    logger.info("L2: BUILDING DATASET")
    logger.info("=" * 60)

    # Load OHLCV
    ohlcv_path = PROJECT_ROOT / "seeds/latest/usdcop_m5_ohlcv.parquet"
    df_ohlcv = pd.read_parquet(ohlcv_path)
    if "time" in df_ohlcv.columns:
        df_ohlcv["time"] = pd.to_datetime(df_ohlcv["time"], utc=True).dt.tz_localize(None)
        df_ohlcv = df_ohlcv.set_index("time").sort_index()
    logger.info(f"  OHLCV: {len(df_ohlcv)} rows, {df_ohlcv.index.min()} to {df_ohlcv.index.max()}")

    close = df_ohlcv["close"]

    # Load macro
    macro_path = PROJECT_ROOT / "data/pipeline/04_cleaning/output/macro_daily_clean.parquet"
    df_macro = pd.read_parquet(macro_path)
    if df_macro.index.tz is not None:
        df_macro.index = df_macro.index.tz_localize(None)
    df_macro = df_macro.rename(columns=MACRO_MAP).sort_index()
    logger.info(f"  Macro: {len(df_macro)} rows")

    # --- Calculate OHLCV features ---
    features = pd.DataFrame(index=df_ohlcv.index)
    features["log_ret_5m"] = calc_log_return(close, 1)
    features["log_ret_1h"] = calc_log_return(close, 12)
    features["log_ret_4h"] = calc_log_return(close, 48)
    features["log_ret_1d"] = calc_log_return(close, 288)
    features["rsi_9"] = calc_rsi(close, 9)
    features["rsi_21"] = calc_rsi(close, 21)
    features["volatility_pct"] = calc_volatility(close, 14, 48)
    features["trend_z"] = calc_trend_z(close, 50)

    # --- Calculate macro features (T-1 shift for anti-leakage) ---
    macro_feats = pd.DataFrame(index=df_macro.index)

    for col_name, macro_col, calc_fn, params in [
        ("dxy_z", "dxy", lambda s: calc_macro_zscore(s, 252), {}),
        ("dxy_change_1d", "dxy", lambda s: s.pct_change(1), {}),
        ("vix_z", "vix", lambda s: calc_macro_zscore(s, 252), {}),
        ("embi_z", "embi", lambda s: calc_macro_zscore(s, 252), {}),
        ("brent_change_1d", "brent", lambda s: s.pct_change(1), {}),
        ("gold_change_1d", "gold", lambda s: s.pct_change(1), {}),
        ("usdmxn_change_1d", "usdmxn", lambda s: s.pct_change(1), {}),
    ]:
        if macro_col in df_macro.columns:
            s = df_macro[macro_col].ffill(limit=5).shift(1)  # T-1 anti-leakage
            macro_feats[col_name] = calc_fn(s)
        else:
            logger.warning(f"  Missing macro column: {macro_col}")
            macro_feats[col_name] = np.nan

    # Spread features
    if "col10y" in df_macro.columns and "ust10y" in df_macro.columns:
        col10y = df_macro["col10y"].ffill(limit=5).shift(1)
        ust10y = df_macro["ust10y"].ffill(limit=5).shift(1)
        macro_feats["rate_spread_z"] = calc_spread_zscore(col10y, ust10y, 252)
        macro_feats["rate_spread_change"] = (col10y - ust10y).diff(1)
    else:
        macro_feats["rate_spread_z"] = np.nan
        macro_feats["rate_spread_change"] = np.nan

    if "ust10y" in df_macro.columns and "ust2y" in df_macro.columns:
        ust10y = df_macro["ust10y"].ffill(limit=5).shift(1)
        ust2y = df_macro["ust2y"].ffill(limit=5).shift(1)
        macro_feats["yield_curve_z"] = calc_spread_zscore(ust10y, ust2y, 252)
    else:
        macro_feats["yield_curve_z"] = np.nan

    # --- Merge macro into OHLCV frequency using merge_asof ---
    features_reset = features.reset_index().rename(columns={features.index.name or "index": "datetime"})
    macro_reset = macro_feats.reset_index().rename(columns={macro_feats.index.name or "index": "datetime"})
    features_reset["datetime"] = pd.to_datetime(features_reset["datetime"])
    macro_reset["datetime"] = pd.to_datetime(macro_reset["datetime"])

    merged = pd.merge_asof(
        features_reset.sort_values("datetime"),
        macro_reset.sort_values("datetime"),
        on="datetime",
        direction="backward",
    ).set_index("datetime")

    # Add auxiliary columns
    merged["raw_log_ret_5m"] = calc_log_return(close, 1)
    merged["close"] = close

    logger.info(f"  Features calculated: {len(FEATURE_ORDER)} features, {len(merged)} rows")

    # --- Normalize: compute stats on train only ---
    dates = cfg["date_ranges"]
    train_end = pd.Timestamp(dates["train_end"])
    val_end = pd.Timestamp(dates["val_end"])

    train_mask = merged.index <= train_end
    norm_stats = {}

    for feat in FEATURE_ORDER:
        if feat not in merged.columns:
            continue
        train_vals = merged.loc[train_mask, feat].dropna()
        if len(train_vals) == 0:
            norm_stats[feat] = {"mean": 0.0, "std": 1.0, "method": "zscore"}
            continue

        # RSI: minmax [0,100] -> [0,1]
        if feat.startswith("rsi_"):
            norm_stats[feat] = {"method": "minmax", "input_min": 0, "input_max": 100,
                                "output_min": 0, "output_max": 1}
            merged[feat] = merged[feat] / 100.0
            continue

        # Z-score features (already z-scored): skip normalization
        if feat.endswith("_z"):
            norm_stats[feat] = {"method": "none"}
            merged[feat] = merged[feat].clip(-5, 5)
            continue

        # Default: z-score normalize
        mean, std = float(train_vals.mean()), float(train_vals.std())
        if std < 1e-8:
            std = 1.0
        norm_stats[feat] = {"mean": mean, "std": std, "method": "zscore"}
        merged[feat] = ((merged[feat] - mean) / std).clip(-5, 5)

    # --- Split ---
    train_df = merged[merged.index <= train_end].dropna(subset=FEATURE_ORDER).copy()
    val_df = merged[(merged.index > train_end) & (merged.index <= val_end)].dropna(subset=FEATURE_ORDER).copy()
    test_df = merged[merged.index > val_end].dropna(subset=FEATURE_ORDER).copy()

    logger.info(f"  Train: {len(train_df)} rows")
    logger.info(f"  Val:   {len(val_df)} rows")
    logger.info(f"  Test:  {len(test_df)} rows")

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "train.parquet")
    val_df.to_parquet(output_dir / "val.parquet")
    test_df.to_parquet(output_dir / "test.parquet")
    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    feature_hash = hashlib.md5(",".join(FEATURE_ORDER).encode()).hexdigest()[:8]
    logger.info(f"  Feature hash: {feature_hash}")
    logger.info("L2 COMPLETE")

    return {
        "train_path": str(output_dir / "train.parquet"),
        "val_path": str(output_dir / "val.parquet"),
        "test_path": str(output_dir / "test.parquet"),
        "norm_stats_path": str(output_dir / "norm_stats.json"),
        "feature_columns": FEATURE_ORDER,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }


# ============================================================================
# SIMPLE TRADING ENVIRONMENT (standalone, no src imports)
# ============================================================================

import gymnasium as gym
from gymnasium import spaces


class SimpleTradingEnv(gym.Env):
    """Minimal trading environment for PPO training. Fully self-contained."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        norm_stats: dict,
        episode_length: int = 1200,
        initial_balance: float = 10000.0,
        cost_bps: float = 5.0,  # round-trip cost
        stop_loss_pct: float = -0.025,
        take_profit_pct: float = 0.03,
        threshold: float = 0.50,
        max_hold_bars: int = 576,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n_bars = len(df)
        self.feature_cols = feature_cols
        self.norm_stats = norm_stats
        self.episode_length = episode_length
        self.initial_balance = initial_balance
        self.cost = cost_bps / 10_000
        self.stop_loss = stop_loss_pct
        self.take_profit = take_profit_pct
        self.threshold = threshold
        self.max_hold = max_hold_bars

        self.obs_dim = len(feature_cols) + 2  # + position + unrealized_pnl
        self.feature_matrix = df[feature_cols].values.astype(np.float32)

        if "raw_log_ret_5m" in df.columns:
            self.returns = df["raw_log_ret_5m"].values.astype(np.float32)
        else:
            self.returns = df["log_ret_5m"].values.astype(np.float32)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.obs_dim,), dtype=np.float32)

        self._reset_state()

    def _reset_state(self):
        self.position = 0  # -1, 0, 1
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.entry_bar = 0
        self.cumulative_return = 0.0
        self.trades = 0
        self.wins = 0
        self.step_count = 0
        self.idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        max_start = self.n_bars - self.episode_length - 20
        self.idx = self.np_random.integers(20, max(21, max_start))
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        act_val = float(action[0])
        target = 1 if act_val > self.threshold else (-1 if act_val < -self.threshold else 0)

        # Execute trade
        cost = 0.0
        if target != self.position:
            cost = self.cost * self.balance
            if self.position != 0:
                self.trades += 1
                pnl_pct = np.exp(self.cumulative_return) - 1.0
                if pnl_pct > 0:
                    self.wins += 1
            self.position = target
            self.entry_bar = self.idx
            self.cumulative_return = 0.0

        # Move forward
        self.idx += 1
        self.step_count += 1
        mkt_ret = float(self.returns[self.idx]) if self.idx < self.n_bars else 0.0

        # PnL
        gross_pnl = self.position * mkt_ret * self.balance
        net_pnl = gross_pnl - cost
        self.balance += net_pnl

        # Track cumulative return for stop/take
        if self.position != 0:
            self.cumulative_return += self.position * mkt_ret

        # Stop loss / take profit
        sl_hit = tp_hit = False
        unrealized = np.exp(self.cumulative_return) - 1.0 if self.position != 0 else 0.0

        if self.position != 0 and unrealized < self.stop_loss:
            sl_hit = True
            cost2 = self.cost * self.balance
            self.balance -= cost2
            self.trades += 1
            self.position = 0
            self.cumulative_return = 0.0

        if self.position != 0 and unrealized >= self.take_profit:
            tp_hit = True
            cost2 = self.cost * self.balance
            self.balance -= cost2
            self.trades += 1
            self.wins += 1
            self.position = 0
            self.cumulative_return = 0.0

        # Max hold duration
        if self.position != 0 and (self.idx - self.entry_bar) > self.max_hold:
            cost2 = self.cost * self.balance
            self.balance -= cost2
            self.trades += 1
            pnl_pct = np.exp(self.cumulative_return) - 1.0
            if pnl_pct > 0:
                self.wins += 1
            self.position = 0
            self.cumulative_return = 0.0

        # Peak tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Reward: scaled PnL with asymmetric loss penalty
        reward = (net_pnl / self.initial_balance) * 100.0
        if reward < 0:
            reward *= 1.5  # asymmetric loss
        if sl_hit:
            reward -= 0.3
        if tp_hit:
            reward += 0.5

        reward = float(np.clip(reward, -5.0, 5.0))

        # Termination
        dd = (self.peak_balance - self.balance) / max(self.peak_balance, 1e-8)
        terminated = (
            dd > 0.15
            or self.step_count >= self.episode_length
            or self.idx >= self.n_bars - 1
        )

        info = {
            "equity": self.balance,
            "balance": self.balance,
            "trades": self.trades,
            "win_rate": self.wins / max(self.trades, 1),
            "drawdown": dd,
        }

        return self._obs(), reward, terminated, False, info

    def _obs(self):
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if self.idx < len(self.feature_matrix):
            obs[:len(self.feature_cols)] = self.feature_matrix[self.idx]
        obs[-2] = float(self.position)
        unrealized = np.exp(self.cumulative_return) - 1.0 if self.position != 0 else 0.0
        obs[-1] = float(np.clip(unrealized, -1, 1))
        obs = np.nan_to_num(np.clip(obs, -5, 5), nan=0.0)
        return obs


# ============================================================================
# L3: TRAINING
# ============================================================================

def run_l3(
    l2: Dict[str, Any],
    cfg: dict,
    models_dir: Path,
    seed: int = SEED,
    total_timesteps: int = 500_000,
) -> Dict[str, Any]:
    """Train PPO model."""
    logger.info("=" * 60)
    logger.info("L3: TRAINING PPO MODEL")
    logger.info("=" * 60)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    set_seeds(seed)

    train_df = pd.read_parquet(l2["train_path"])
    val_df = pd.read_parquet(l2["val_path"])
    with open(l2["norm_stats_path"]) as f:
        norm_stats = json.load(f)

    feature_cols = l2["feature_columns"]
    ppo_cfg = cfg["training"]["ppo"]
    env_cfg = cfg["training"]["environment"]

    logger.info(f"  Train: {len(train_df)} rows, Val: {len(val_df)} rows")
    logger.info(f"  Features: {len(feature_cols)}, Obs dim: {len(feature_cols) + 2}")
    logger.info(f"  Timesteps: {total_timesteps}, Seed: {seed}")

    # Environment params
    env_params = dict(
        feature_cols=feature_cols,
        norm_stats=norm_stats,
        episode_length=env_cfg.get("episode_length", 1200),
        initial_balance=env_cfg.get("initial_balance", 10000),
        cost_bps=env_cfg.get("transaction_cost_bps", 2.5) + env_cfg.get("slippage_bps", 2.5),
        stop_loss_pct=env_cfg.get("stop_loss_pct", -0.025),
        take_profit_pct=env_cfg.get("take_profit_pct", 0.03),
        threshold=env_cfg.get("thresholds", {}).get("long", 0.50),
        max_hold_bars=env_cfg.get("max_position_duration", 576),
    )

    set_seeds(seed)
    train_env = DummyVecEnv([lambda: SimpleTradingEnv(df=train_df, **env_params)])
    eval_env = DummyVecEnv([lambda: SimpleTradingEnv(df=val_df, **env_params)])

    # Model dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = models_dir / f"ppo_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(seed)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=ppo_cfg.get("learning_rate", 0.0003),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 256),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.95),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.02),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        seed=seed,
        device="cpu",
        verbose=1,
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "logs"),
        eval_freq=max(total_timesteps // 20, 5000),
        n_eval_episodes=5,
        deterministic=True,
    )

    logger.info(f"  Training for {total_timesteps} timesteps...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback], progress_bar=True)
    elapsed = time.time() - t0
    logger.info(f"  Training complete in {elapsed:.0f}s")

    # Save
    model.save(str(model_dir / "final_model"))
    with open(model_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    with open(model_dir / "config.json", "w") as f:
        json.dump({"seed": seed, "timesteps": total_timesteps, "features": feature_cols,
                    "ppo": ppo_cfg, "env": env_cfg, "timestamp": timestamp}, f, indent=2)

    logger.info(f"  Model saved: {model_dir}")
    logger.info("L3 COMPLETE")

    return {
        "model_dir": str(model_dir),
        "model_path": str(model_dir / "final_model.zip"),
        "norm_stats_path": str(model_dir / "norm_stats.json"),
        "timestamp": timestamp,
    }


# ============================================================================
# L4: BACKTEST
# ============================================================================

def run_backtest_on_df(
    model_path: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    norm_stats: dict,
    cfg: dict,
    label: str = "BACKTEST",
) -> Dict[str, Any]:
    """Run deterministic backtest on a DataFrame."""
    from stable_baselines3 import PPO

    env_cfg = cfg["training"]["environment"]
    model = PPO.load(model_path)

    cost_bps = env_cfg.get("transaction_cost_bps", 2.5) + env_cfg.get("slippage_bps", 2.5)
    cost = cost_bps / 10_000
    threshold = env_cfg.get("thresholds", {}).get("long", 0.50)
    stop_loss = env_cfg.get("stop_loss_pct", -0.025)
    take_profit = env_cfg.get("take_profit_pct", 0.03)
    max_hold = env_cfg.get("max_position_duration", 576)
    initial_balance = 10000.0

    # Prepare data
    df = df.reset_index(drop=True)
    feature_matrix = df[feature_cols].values.astype(np.float32)

    if "raw_log_ret_5m" in df.columns:
        returns = df["raw_log_ret_5m"].values.astype(np.float32)
    else:
        returns = df["log_ret_5m"].values.astype(np.float32)

    obs_dim = len(feature_cols) + 2
    balance = initial_balance
    peak_balance = initial_balance
    position = 0
    entry_bar = 0
    cum_ret = 0.0
    trades = 0
    wins = 0
    equity_curve = [initial_balance]
    actions_taken = []

    for i in range(1, len(df) - 1):
        # Build observation
        obs = np.zeros(obs_dim, dtype=np.float32)
        obs[:len(feature_cols)] = feature_matrix[i]
        obs[-2] = float(position)
        unrealized = np.exp(cum_ret) - 1.0 if position != 0 else 0.0
        obs[-1] = float(np.clip(unrealized, -1, 1))
        obs = np.nan_to_num(np.clip(obs, -5, 5), nan=0.0)

        # Model prediction (deterministic)
        action, _ = model.predict(obs, deterministic=True)
        act_val = float(action[0])
        target = 1 if act_val > threshold else (-1 if act_val < -threshold else 0)
        actions_taken.append(target)

        # Trade
        trade_cost = 0.0
        if target != position:
            trade_cost = cost * balance
            if position != 0:
                trades += 1
                pnl_pct = np.exp(cum_ret) - 1.0
                if pnl_pct > 0:
                    wins += 1
            position = target
            entry_bar = i
            cum_ret = 0.0

        # Market return
        mkt_ret = float(returns[i + 1]) if i + 1 < len(returns) else 0.0
        gross_pnl = position * mkt_ret * balance
        net_pnl = gross_pnl - trade_cost
        balance += net_pnl

        # Track cumulative return
        if position != 0:
            cum_ret += position * mkt_ret

        # Stop loss
        unrealized = np.exp(cum_ret) - 1.0 if position != 0 else 0.0
        if position != 0 and unrealized < stop_loss:
            balance -= cost * balance
            trades += 1
            position = 0
            cum_ret = 0.0

        # Take profit
        if position != 0 and unrealized >= take_profit:
            balance -= cost * balance
            trades += 1
            wins += 1
            position = 0
            cum_ret = 0.0

        # Max hold
        if position != 0 and (i - entry_bar) > max_hold:
            balance -= cost * balance
            trades += 1
            pnl_pct = np.exp(cum_ret) - 1.0
            if pnl_pct > 0:
                wins += 1
            position = 0
            cum_ret = 0.0

        if balance > peak_balance:
            peak_balance = balance

        equity_curve.append(balance)

    # Metrics
    equity = np.array(equity_curve)
    total_return = (equity[-1] / equity[0] - 1) * 100
    days = len(equity) / 48.0  # 48 bars per trading day
    apr = total_return * (365 / max(days, 1))

    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-8)
    sharpe = float(np.mean(rets) / max(np.std(rets), 1e-10) * np.sqrt(252 * 48))

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-8)
    max_dd = float(np.max(dd) * 100)

    win_rate = wins / max(trades, 1) * 100

    # Action distribution
    actions_arr = np.array(actions_taken)
    n_long = int(np.sum(actions_arr == 1))
    n_short = int(np.sum(actions_arr == -1))
    n_hold = int(np.sum(actions_arr == 0))

    metrics = {
        "label": label,
        "total_return_pct": round(total_return, 2),
        "apr_pct": round(apr, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "trades": trades,
        "win_rate_pct": round(win_rate, 1),
        "final_equity": round(float(equity[-1]), 2),
        "days": round(days, 1),
        "actions": {"long": n_long, "short": n_short, "hold": n_hold},
    }

    return metrics, equity_curve


def run_l4(l2: Dict, l3: Dict, cfg: dict, results_dir: Path) -> Dict[str, Any]:
    """Run L4 backtest on val + test."""
    logger.info("=" * 60)
    logger.info("L4: BACKTESTING")
    logger.info("=" * 60)

    # Prefer best_model.zip
    model_dir = Path(l3["model_dir"])
    best_model = model_dir / "best_model.zip"
    final_model = Path(l3["model_path"])
    model_path = str(best_model) if best_model.exists() else str(final_model)
    logger.info(f"  Model: {Path(model_path).name}")

    with open(l3["norm_stats_path"]) as f:
        norm_stats = json.load(f)

    feature_cols = l2["feature_columns"]

    # Validation backtest
    val_df = pd.read_parquet(l2["val_path"])
    logger.info(f"  Running VAL backtest ({len(val_df)} rows)...")
    val_metrics, val_equity = run_backtest_on_df(
        model_path, val_df, feature_cols, norm_stats, cfg, "L4-VAL"
    )

    logger.info(f"  VAL: Return={val_metrics['total_return_pct']:.2f}%, "
                f"APR={val_metrics['apr_pct']:.2f}%, "
                f"Sharpe={val_metrics['sharpe_ratio']:.3f}, "
                f"DD={val_metrics['max_drawdown_pct']:.2f}%, "
                f"Trades={val_metrics['trades']}")

    # Test backtest
    test_df = pd.read_parquet(l2["test_path"])
    logger.info(f"  Running TEST backtest ({len(test_df)} rows)...")
    test_metrics, test_equity = run_backtest_on_df(
        model_path, test_df, feature_cols, norm_stats, cfg, "L4-TEST"
    )

    logger.info(f"  TEST: Return={test_metrics['total_return_pct']:.2f}%, "
                f"APR={test_metrics['apr_pct']:.2f}%, "
                f"Sharpe={test_metrics['sharpe_ratio']:.3f}, "
                f"DD={test_metrics['max_drawdown_pct']:.2f}%, "
                f"Trades={test_metrics['trades']}")

    # Combined (val+test)
    combined_df = pd.concat([val_df, test_df]).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
    comb_metrics, comb_equity = run_backtest_on_df(
        model_path, combined_df, feature_cols, norm_stats, cfg, "L4-COMBINED"
    )

    logger.info(f"  COMBINED: Return={comb_metrics['total_return_pct']:.2f}%, "
                f"APR={comb_metrics['apr_pct']:.2f}%, "
                f"Sharpe={comb_metrics['sharpe_ratio']:.3f}")

    # Gates check
    gates = cfg.get("backtest", {}).get("gates", {})
    gate_checks = {
        "min_sharpe": comb_metrics["sharpe_ratio"] >= gates.get("min_sharpe_ratio", 0.3),
        "max_drawdown": comb_metrics["max_drawdown_pct"] <= gates.get("max_drawdown_pct", 25),
        "min_trades": comb_metrics["trades"] >= gates.get("min_trades", 20),
    }
    all_passed = all(gate_checks.values())

    logger.info(f"  Gates: {'PASSED' if all_passed else 'FAILED'} {gate_checks}")

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = l3["timestamp"]
    results = {
        "val": val_metrics,
        "test": test_metrics,
        "combined": comb_metrics,
        "gates_passed": all_passed,
        "gate_checks": gate_checks,
    }
    with open(results_dir / f"backtest_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame({"equity": comb_equity}).to_parquet(results_dir / f"equity_{ts}.parquet")

    logger.info("L4 COMPLETE")
    return results


# ============================================================================
# REPRODUCIBILITY VERIFICATION
# ============================================================================

def verify_reproducibility(
    cfg: dict,
    output_base: Path,
    models_base: Path,
    total_timesteps: int,
    seed: int,
) -> bool:
    """Run pipeline twice with same seed and verify identical results."""
    logger.info("=" * 60)
    logger.info("REPRODUCIBILITY VERIFICATION")
    logger.info("=" * 60)

    results = []
    for run_id in [1, 2]:
        logger.info(f"\n--- Run {run_id}/2 (seed={seed}) ---")
        out_dir = output_base / f"repro_run{run_id}"
        mod_dir = models_base / f"repro_run{run_id}"

        set_seeds(seed)
        l2 = run_l2(cfg, out_dir)
        l3 = run_l3(l2, cfg, mod_dir, seed=seed, total_timesteps=total_timesteps)
        l4 = run_l4(l2, l3, cfg, out_dir / "results")
        results.append(l4)

    # Compare
    r1 = results[0]["combined"]
    r2 = results[1]["combined"]

    match = True
    for key in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "trades"]:
        v1, v2 = r1[key], r2[key]
        if abs(v1 - v2) > 0.01:
            logger.error(f"  MISMATCH: {key}: run1={v1}, run2={v2}")
            match = False
        else:
            logger.info(f"  MATCH: {key}: {v1} == {v2}")

    if match:
        logger.info("REPRODUCIBILITY: VERIFIED - Both runs produce identical results")
    else:
        logger.error("REPRODUCIBILITY: FAILED - Results differ between runs")

    return match


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone L2-L3-L4 Pipeline")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--verify-reproducibility", action="store_true",
                        help="Run twice to verify reproducibility (uses fewer timesteps)")
    parser.add_argument("--output-dir", type=str, default="data/pipeline/07_output/5min_standalone")
    parser.add_argument("--models-dir", type=str, default="models/standalone")
    parser.add_argument("--results-dir", type=str, default="results/standalone")
    args = parser.parse_args()

    # Load config
    with open(PROJECT_ROOT / "config" / "pipeline_ssot.yaml") as f:
        cfg = yaml.safe_load(f)

    output_dir = PROJECT_ROOT / args.output_dir
    models_dir = PROJECT_ROOT / args.models_dir
    results_dir = PROJECT_ROOT / args.results_dir

    if args.verify_reproducibility:
        # Quick reproducibility check with fewer timesteps
        ts = min(args.timesteps, 50_000)
        logger.info(f"Reproducibility check with {ts} timesteps...")
        ok = verify_reproducibility(cfg, output_dir, models_dir, ts, args.seed)
        return 0 if ok else 1

    # Full pipeline
    set_seeds(args.seed)

    logger.info("=" * 60)
    logger.info("STANDALONE L2-L3-L4 PIPELINE")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Timesteps: {args.timesteps}")
    logger.info("=" * 60)

    t0 = time.time()

    l2 = run_l2(cfg, output_dir)
    l3 = run_l3(l2, cfg, models_dir, seed=args.seed, total_timesteps=args.timesteps)
    l4 = run_l4(l2, l3, cfg, results_dir)

    elapsed = time.time() - t0

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"  L2: {l2['train_rows']} train, {l2['val_rows']} val, {l2['test_rows']} test")
    logger.info(f"  L3: Model at {l3['model_dir']}")

    comb = l4["combined"]
    logger.info(f"  L4 COMBINED:")
    logger.info(f"    Return: {comb['total_return_pct']:.2f}%")
    logger.info(f"    APR:    {comb['apr_pct']:.2f}%")
    logger.info(f"    Sharpe: {comb['sharpe_ratio']:.3f}")
    logger.info(f"    MaxDD:  {comb['max_drawdown_pct']:.2f}%")
    logger.info(f"    Trades: {comb['trades']}")
    logger.info(f"    WinRate: {comb['win_rate_pct']:.1f}%")
    logger.info(f"    Actions: {comb['actions']}")
    logger.info(f"  Gates: {'PASSED' if l4['gates_passed'] else 'FAILED'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
