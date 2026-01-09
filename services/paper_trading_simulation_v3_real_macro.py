"""
Paper Trading Simulation V3 - WITH REAL MACRO DATA
===================================================

Uses actual macro data from macro_indicators_daily table with forward-fill
for gaps (weekends/holidays).

Author: Claude Code
Version: 3.0.0
Date: 2026-01-07
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "model_id": "ppo_v1",
    "initial_capital": 10000.0,
    "slippage_bps": 2,
    "out_of_sample_start": "2025-12-27",
    "out_of_sample_end": "2026-01-06",
}

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB", "usdcop_trading"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin123"),
}

MODEL_PATH = PROJECT_ROOT / "models" / "ppo_v1_20251226_054154.zip"

# Normalization stats from training (v19_norm_stats.json)
NORM_STATS = {
    "dxy_z": {"mean": 100.21, "std": 5.60},
    "vix_z": {"mean": 21.16, "std": 7.89},
    "embi_z": {"mean": 322.01, "std": 62.68},
    "rate_spread": {"mean": 7.03, "std": 1.41},
}

# 20 Market features (after removing hour_sin, hour_cos)
MARKET_FEATURES = [
    "open", "high", "low", "close",
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "atr_pct", "adx_14", "bb_position",
    "dxy_z", "dxy_change_1d", "dxy_mom_5d",
    "vix_z", "embi_z",
    "brent_change_1d", "brent_vol_5d",
    "rate_spread", "usdmxn_change_1d"
]


# =============================================================================
# Database Functions
# =============================================================================

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_ohlcv_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load OHLCV data from PostgreSQL."""
    logger.info(f"Loading OHLCV data from {start_date} to {end_date}")
    conn = get_db_connection()
    query = """
        SELECT time, open, high, low, close, volume
        FROM usdcop_m5_ohlcv
        WHERE time >= %s AND time <= %s
        ORDER BY time
    """
    df = pd.read_sql(query, conn, params=[start_date + ' 00:00:00', end_date + ' 23:59:59'])
    conn.close()
    logger.info(f"Loaded {len(df)} OHLCV bars")
    return df


def load_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Load macro data from PostgreSQL with forward-fill for gaps."""
    logger.info(f"Loading macro data from {start_date} to {end_date}")
    conn = get_db_connection()

    # Load more history for forward-fill
    extended_start = (pd.to_datetime(start_date) - timedelta(days=30)).strftime('%Y-%m-%d')

    query = """
        SELECT
            fecha as date,
            fxrt_index_dxy_usa_d_dxy as dxy,
            volt_vix_usa_d_vix as vix,
            crsk_spread_embi_col_d_embi as embi,
            comm_oil_brent_glb_d_brent as brent,
            fxrt_spot_usdmxn_mex_d_usdmxn as usdmxn,
            finc_bond_yield10y_usa_d_ust10y as ust10y,
            finc_bond_yield2y_usa_d_dgs2 as ust2y
        FROM macro_indicators_daily
        WHERE fecha >= %s AND fecha <= %s
        ORDER BY fecha
    """
    df = pd.read_sql(query, conn, params=[extended_start, end_date])
    conn.close()

    logger.info(f"Loaded {len(df)} macro rows")

    # Forward-fill missing values
    df = df.ffill()

    # Calculate derived features
    df['dxy_z'] = (df['dxy'] - NORM_STATS['dxy_z']['mean']) / NORM_STATS['dxy_z']['std']
    df['vix_z'] = (df['vix'] - NORM_STATS['vix_z']['mean']) / NORM_STATS['vix_z']['std']

    # EMBI - use default if all NaN
    if df['embi'].isna().all():
        logger.warning("EMBI data missing - using default value 300")
        df['embi'] = 300.0
    df['embi_z'] = (df['embi'] - NORM_STATS['embi_z']['mean']) / NORM_STATS['embi_z']['std']

    # Rate spread = 10Y - 2Y
    df['rate_spread_raw'] = df['ust10y'] - df['ust2y']
    df['rate_spread'] = (df['rate_spread_raw'] - NORM_STATS['rate_spread']['mean']) / NORM_STATS['rate_spread']['std']

    # Daily changes
    df['dxy_change_1d'] = df['dxy'].pct_change().clip(-0.03, 0.03)
    df['dxy_mom_5d'] = df['dxy'].pct_change(5).clip(-0.05, 0.05)
    df['brent_change_1d'] = df['brent'].pct_change().clip(-0.10, 0.10)
    df['brent_vol_5d'] = df['brent'].pct_change().rolling(5).std().clip(0, 0.05)
    df['usdmxn_change_1d'] = df['usdmxn'].pct_change().clip(-0.10, 0.10)

    # Final forward-fill and fill remaining NaN with 0
    df = df.ffill().fillna(0)

    # Clip z-scores
    for col in ['dxy_z', 'vix_z', 'embi_z', 'rate_spread']:
        df[col] = df[col].clip(-4, 4)

    logger.info("Macro features calculated")
    return df


# =============================================================================
# Feature Calculators
# =============================================================================

class FeatureCalculator:
    """Calculate technical features from OHLCV data."""

    RSI_PERIOD = 9
    ATR_PERIOD = 10
    ADX_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2

    def calculate_log_return(self, series: pd.Series, periods: int) -> pd.Series:
        return np.log(series / series.shift(periods)).clip(-0.05, 0.05)

    def calculate_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/self.RSI_PERIOD, min_periods=self.RSI_PERIOD).mean()
        avg_loss = loss.ewm(alpha=1/self.RSI_PERIOD, min_periods=self.RSI_PERIOD).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    def calculate_atr_pct(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=self.ATR_PERIOD, min_periods=self.ATR_PERIOD).mean()
        return (atr / close) * 100

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean() / atr)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        return dx.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean().fillna(25)

    def calculate_bb_position(self, close: pd.Series) -> pd.Series:
        sma = close.rolling(self.BB_PERIOD).mean()
        std = close.rolling(self.BB_PERIOD).std()
        upper = sma + self.BB_STD * std
        lower = sma - self.BB_STD * std
        position = (close - lower) / (upper - lower)
        return position.clip(0, 1).fillna(0.5)

    def build_features(self, ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Build all 20 market features combining OHLCV and macro data."""
        df = ohlcv_df.copy()

        # Technical features from OHLCV
        df['log_ret_5m'] = self.calculate_log_return(df['close'], 1)
        df['log_ret_1h'] = self.calculate_log_return(df['close'], 12)
        df['log_ret_4h'] = self.calculate_log_return(df['close'], 48)
        df['rsi_9'] = self.calculate_rsi(df['close'])
        df['atr_pct'] = self.calculate_atr_pct(df['high'], df['low'], df['close'])
        df['adx_14'] = self.calculate_adx(df['high'], df['low'], df['close'])
        df['bb_position'] = self.calculate_bb_position(df['close'])

        # Add date column for macro merge
        df['date'] = pd.to_datetime(df['time']).dt.date

        # Merge macro features (forward-fill to 5min bars)
        macro_df['date'] = pd.to_datetime(macro_df['date']).dt.date
        macro_cols = ['date', 'dxy_z', 'dxy_change_1d', 'dxy_mom_5d',
                      'vix_z', 'embi_z', 'brent_change_1d', 'brent_vol_5d',
                      'rate_spread', 'usdmxn_change_1d']

        df = df.merge(macro_df[macro_cols], on='date', how='left')

        # Forward-fill any remaining gaps
        df = df.ffill().fillna(0)

        logger.info(f"Features built: {len(df)} rows with {len(MARKET_FEATURES)} features")
        return df


# =============================================================================
# State Tracker (12 state features)
# =============================================================================

class StateTracker:
    """Tracks the 12 state features for the environment."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.entry_price = None
        self.unrealized_pnl = 0.0
        self.cumulative_return = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown_episode = 0.0
        self.position_duration = 0
        self.trade_count = 0
        self.current_step = 0

    def update(self, price: float):
        if self.position != 0 and self.entry_price is not None:
            if self.position > 0:
                pnl = (price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - price) / self.entry_price
            self.unrealized_pnl = np.clip(pnl, -1, 1)
        else:
            self.unrealized_pnl = 0.0

        total_value = self.portfolio_value * (1 + self.unrealized_pnl * abs(self.position))
        if total_value > self.peak_value:
            self.peak_value = total_value
        self.current_drawdown = (self.peak_value - total_value) / self.peak_value
        self.max_drawdown_episode = max(self.max_drawdown_episode, self.current_drawdown)

        if self.position != 0:
            self.position_duration += 1

    def get_state_features(self, episode_length: int = 400) -> np.ndarray:
        state = np.zeros(12, dtype=np.float32)
        state[0] = self.position
        state[1] = self.unrealized_pnl
        state[2] = np.clip(self.cumulative_return, -1, 1)
        state[3] = -self.current_drawdown
        state[4] = -self.max_drawdown_episode
        state[5] = 0.5  # regime_encoded
        state[6] = 0.5  # session_phase
        state[7] = 0.5  # volatility_regime
        state[8] = 1.0  # cost_regime
        state[9] = min(self.position_duration / 50.0, 1.0)
        state[10] = min(self.trade_count / 50.0, 1.0)
        state[11] = max(0, 1.0 - self.current_step / episode_length)
        return state

    def open_position(self, side: str, price: float):
        self.position = 1.0 if side == "LONG" else -1.0
        self.entry_price = price
        self.position_duration = 0
        self.trade_count += 1

    def close_position(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        if self.position > 0:
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - price) / self.entry_price
        pnl_usd = pnl_pct * self.portfolio_value
        self.portfolio_value += pnl_usd
        self.cumulative_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        self.position = 0.0
        self.entry_price = None
        self.position_duration = 0
        self.unrealized_pnl = 0.0
        return pnl_usd


# =============================================================================
# Paper Trading Engine
# =============================================================================

class PaperTradingEngineV3:
    """Paper trading with real macro data."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_id = config["model_id"]
        self.initial_capital = config["initial_capital"]
        self.slippage_bps = config["slippage_bps"]
        self.feature_calc = FeatureCalculator()
        self.state_tracker = StateTracker(self.initial_capital)
        self.trades: List[Dict] = []
        self.equity_snapshots: List[Dict] = []
        self.model = None

    def load_model(self):
        from stable_baselines3 import PPO
        logger.info(f"Loading model from {MODEL_PATH}")
        self.model = PPO.load(str(MODEL_PATH))
        logger.info(f"Model loaded - obs_space: {self.model.observation_space.shape}")

    def build_observation(self, row: pd.Series, feature_means: Dict, feature_stds: Dict) -> np.ndarray:
        obs = np.zeros(32, dtype=np.float32)
        state = self.state_tracker.get_state_features()
        obs[:12] = state
        for i, feat in enumerate(MARKET_FEATURES):
            raw = float(row.get(feat, 0.0))
            mean = feature_means.get(feat, 0.0)
            std = feature_stds.get(feat, 1.0)
            if std < 1e-8:
                std = 1.0
            normalized = (raw - mean) / std
            obs[12 + i] = np.clip(normalized, -5, 5)
        return np.nan_to_num(obs, nan=0.0)

    def get_action(self, observation: np.ndarray) -> Tuple[float, str]:
        action, _ = self.model.predict(observation, deterministic=True)
        action_value = float(action[0])
        if action_value > 0.1:
            return action_value, "LONG"
        elif action_value < -0.1:
            return action_value, "SHORT"
        return action_value, "FLAT"

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        slip = price * (self.slippage_bps / 10000)
        return price + slip if is_buy else price - slip

    def run_simulation(self, ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> Dict:
        logger.info(f"Starting simulation with {len(ohlcv_df)} bars and real macro data")
        self.load_model()

        features_df = self.feature_calc.build_features(ohlcv_df, macro_df)

        # Log macro data summary
        logger.info("Macro data summary for simulation:")
        for col in ['dxy_z', 'vix_z', 'embi_z', 'rate_spread']:
            vals = features_df[col].dropna()
            if len(vals) > 0:
                logger.info(f"  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}, range=[{vals.min():.3f}, {vals.max():.3f}]")

        feature_means = {f: float(features_df[f].mean()) for f in MARKET_FEATURES if f in features_df.columns}
        feature_stds = {f: float(features_df[f].std()) + 1e-8 for f in MARKET_FEATURES if f in features_df.columns}

        warmup = 50
        current_position = "FLAT"
        entry_price = None
        entry_time = None
        entry_bar = None

        for i in range(warmup, len(features_df)):
            row = features_df.iloc[i]
            timestamp = row['time']
            price = float(row['close'])

            self.state_tracker.current_step = i - warmup
            self.state_tracker.update(price)

            obs = self.build_observation(row, feature_means, feature_stds)
            action_value, desired_position = self.get_action(obs)

            if desired_position != current_position:
                if current_position != "FLAT":
                    exec_price = self.apply_slippage(price, current_position == "SHORT")
                    pnl_usd = self.state_tracker.close_position(exec_price)
                    self.trades.append({
                        "model_id": self.model_id,
                        "side": current_position,
                        "entry_price": entry_price,
                        "exit_price": exec_price,
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "duration_bars": i - entry_bar,
                        "pnl_usd": pnl_usd,
                        "pnl_pct": (pnl_usd / self.state_tracker.portfolio_value) * 100,
                        "exit_reason": "SIGNAL_CHANGE",
                        "equity_at_entry": self.state_tracker.portfolio_value - pnl_usd,
                        "equity_at_exit": self.state_tracker.portfolio_value,
                        "drawdown_at_entry": self.state_tracker.current_drawdown * 100,
                        "bar_number": i
                    })
                    logger.info(f"Trade closed: {current_position} | PnL: ${pnl_usd:.2f}")

                if desired_position != "FLAT":
                    exec_price = self.apply_slippage(price, desired_position == "LONG")
                    self.state_tracker.open_position(desired_position, exec_price)
                    entry_price = exec_price
                    entry_time = timestamp
                    entry_bar = i

                current_position = desired_position

            if i % 5 == 0:
                unrealized = 0
                if current_position != "FLAT" and entry_price:
                    if current_position == "LONG":
                        unrealized = (price - entry_price) / entry_price * self.state_tracker.portfolio_value
                    else:
                        unrealized = (entry_price - price) / entry_price * self.state_tracker.portfolio_value
                self.equity_snapshots.append({
                    "model_id": self.model_id,
                    "timestamp": timestamp,
                    "equity": self.state_tracker.portfolio_value + unrealized,
                    "drawdown_pct": self.state_tracker.current_drawdown * 100,
                    "position": current_position,
                    "bar_close_price": price
                })

        if current_position != "FLAT":
            last_row = features_df.iloc[-1]
            price = float(last_row['close'])
            exec_price = self.apply_slippage(price, current_position == "SHORT")
            pnl_usd = self.state_tracker.close_position(exec_price)
            self.trades.append({
                "model_id": self.model_id,
                "side": current_position,
                "entry_price": entry_price,
                "exit_price": exec_price,
                "entry_time": entry_time,
                "exit_time": last_row['time'],
                "duration_bars": len(features_df) - 1 - entry_bar,
                "pnl_usd": pnl_usd,
                "pnl_pct": (pnl_usd / self.state_tracker.portfolio_value) * 100,
                "exit_reason": "END_OF_SAMPLE",
                "equity_at_entry": self.state_tracker.portfolio_value - pnl_usd,
                "equity_at_exit": self.state_tracker.portfolio_value,
                "drawdown_at_entry": self.state_tracker.current_drawdown * 100,
                "bar_number": len(features_df) - 1
            })

        metrics = self.calculate_metrics()

        logger.info("=" * 60)
        logger.info("SIMULATION COMPLETE (WITH REAL MACRO DATA)")
        logger.info(f"Final Equity: ${self.state_tracker.portfolio_value:,.2f}")
        logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']:.1f}%")
        logger.info("=" * 60)

        return {"config": self.config, "metrics": metrics, "trades": self.trades, "equity_snapshots": self.equity_snapshots}

    def calculate_metrics(self) -> Dict:
        total_return = (self.state_tracker.portfolio_value - self.initial_capital) / self.initial_capital * 100
        winning = sum(1 for t in self.trades if t["pnl_usd"] > 0)
        losing = len(self.trades) - winning
        win_rate = (winning / len(self.trades) * 100) if self.trades else 0

        if self.equity_snapshots:
            equities = [s["equity"] for s in self.equity_snapshots]
            if len(equities) > 1:
                returns = np.diff(equities) / equities[:-1]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 59)
            else:
                sharpe = 0
        else:
            sharpe = 0

        max_dd = max([s["drawdown_pct"] for s in self.equity_snapshots]) if self.equity_snapshots else 0
        gross_profit = sum(t["pnl_usd"] for t in self.trades if t["pnl_usd"] > 0)
        gross_loss = abs(sum(t["pnl_usd"] for t in self.trades if t["pnl_usd"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.99

        return {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "current_drawdown_pct": round(self.state_tracker.current_drawdown * 100, 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(min(profit_factor, 999.99), 2),
            "total_trades": len(self.trades),
            "winning_trades": winning,
            "losing_trades": losing,
            "final_equity": round(self.state_tracker.portfolio_value, 2),
            "realized_pnl": round(self.state_tracker.portfolio_value - self.initial_capital, 2)
        }


def save_results_to_db(results: Dict):
    """Save results to database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        model_id = results["config"]["model_id"]
        metrics = results["metrics"]

        cursor.execute("""
            UPDATE trading_state SET
                position = 'FLAT', entry_price = NULL, entry_time = NULL, bars_in_position = 0,
                unrealized_pnl = 0, realized_pnl = %s, equity = %s, peak_equity = %s,
                drawdown_pct = %s, trade_count = %s, winning_trades = %s, losing_trades = %s,
                last_signal = 'HOLD', last_updated = NOW()
            WHERE model_id = %s
        """, (metrics["realized_pnl"], metrics["final_equity"], metrics["final_equity"],
              metrics["current_drawdown_pct"], metrics["total_trades"],
              metrics["winning_trades"], metrics["losing_trades"], model_id))

        cursor.execute("DELETE FROM trades_history WHERE model_id = %s", (model_id,))
        cursor.execute("DELETE FROM equity_snapshots WHERE model_id = %s", (model_id,))

        if results["trades"]:
            trades_data = [(t["model_id"], t["side"], t["entry_price"], t["exit_price"],
                           t["entry_time"], t["exit_time"], t["duration_bars"], t["pnl_usd"],
                           t["pnl_pct"], t["exit_reason"], t["equity_at_entry"],
                           t["equity_at_exit"], t["drawdown_at_entry"], t["bar_number"])
                          for t in results["trades"]]
            execute_batch(cursor, """
                INSERT INTO trades_history (model_id, side, entry_price, exit_price, entry_time,
                    exit_time, duration_bars, pnl_usd, pnl_pct, exit_reason, equity_at_entry,
                    equity_at_exit, drawdown_at_entry, bar_number)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, trades_data)

        if results["equity_snapshots"]:
            snapshot_data = [(s["model_id"], s["timestamp"], s["equity"], s["drawdown_pct"],
                             s["position"], s["bar_close_price"]) for s in results["equity_snapshots"]]
            execute_batch(cursor, """
                INSERT INTO equity_snapshots (model_id, timestamp, equity, drawdown_pct, position, bar_close_price)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (model_id, timestamp) DO UPDATE SET equity = EXCLUDED.equity
            """, snapshot_data)

        conn.commit()
        logger.info("Results saved to database")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def main():
    print("=" * 70)
    print("PAPER TRADING SIMULATION V3 - WITH REAL MACRO DATA")
    print("=" * 70)

    ohlcv_df = load_ohlcv_data(CONFIG["out_of_sample_start"], CONFIG["out_of_sample_end"])
    macro_df = load_macro_data(CONFIG["out_of_sample_start"], CONFIG["out_of_sample_end"])

    if len(ohlcv_df) == 0:
        logger.error("No OHLCV data found")
        return

    engine = PaperTradingEngineV3(CONFIG)
    results = engine.run_simulation(ohlcv_df, macro_df)
    save_results_to_db(results)

    print("\n" + "=" * 70)
    print("RESULTS WITH REAL MACRO DATA")
    print("=" * 70)
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
