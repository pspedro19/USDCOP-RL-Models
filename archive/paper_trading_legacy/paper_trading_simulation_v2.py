"""
Paper Trading Simulation V2 - Correct 32-dim Observation
=========================================================

Runs PPO V1 model on out-of-sample data using the EXACT same observation
structure as training:
- 12 state features (position, unrealized_pnl, drawdown, etc.)
- 20 market features (from RL_DS3_MACRO_CORE.csv minus hour_sin/hour_cos)
- Total: 32 dimensions

Author: Claude Code
Version: 2.0.0
Date: 2026-01-07
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
# Feature Calculators
# =============================================================================

class FeatureCalculator:
    """Calculate the 20 market features from OHLCV + macro data."""

    RSI_PERIOD = 9
    ATR_PERIOD = 10
    ADX_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2

    def calculate_log_return(self, series: pd.Series, periods: int) -> pd.Series:
        """Calculate log returns."""
        return np.log(series / series.shift(periods)).clip(-0.05, 0.05)

    def calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI (0-100)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/self.RSI_PERIOD, min_periods=self.RSI_PERIOD).mean()
        avg_loss = loss.ewm(alpha=1/self.RSI_PERIOD, min_periods=self.RSI_PERIOD).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    def calculate_atr_pct(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR as percentage."""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=self.ATR_PERIOD, min_periods=self.ATR_PERIOD).mean()
        return (atr / close) * 100

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ADX (0-100)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean() / atr)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        return dx.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean().fillna(25)

    def calculate_bb_position(self, close: pd.Series) -> pd.Series:
        """Calculate Bollinger Band position (0-1)."""
        sma = close.rolling(self.BB_PERIOD).mean()
        std = close.rolling(self.BB_PERIOD).std()
        upper = sma + self.BB_STD * std
        lower = sma - self.BB_STD * std
        position = (close - lower) / (upper - lower)
        return position.clip(0, 1).fillna(0.5)

    def build_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Build all 20 market features."""
        df = ohlcv_df.copy()

        # Log returns
        df['log_ret_5m'] = self.calculate_log_return(df['close'], 1)
        df['log_ret_1h'] = self.calculate_log_return(df['close'], 12)
        df['log_ret_4h'] = self.calculate_log_return(df['close'], 48)

        # Technical indicators
        df['rsi_9'] = self.calculate_rsi(df['close'])
        df['atr_pct'] = self.calculate_atr_pct(df['high'], df['low'], df['close'])
        df['adx_14'] = self.calculate_adx(df['high'], df['low'], df['close'])
        df['bb_position'] = self.calculate_bb_position(df['close'])

        # Macro features - use neutral/default values
        # In production these come from macro_indicators_daily table
        df['dxy_z'] = 0.0
        df['dxy_change_1d'] = 0.0
        df['dxy_mom_5d'] = 0.0
        df['vix_z'] = 0.0
        df['embi_z'] = 0.0
        df['brent_change_1d'] = 0.0
        df['brent_vol_5d'] = 0.0
        df['rate_spread'] = 0.0
        df['usdmxn_change_1d'] = 0.0

        return df.fillna(method='ffill').fillna(0)


# =============================================================================
# State Tracker (12 state features)
# =============================================================================

class StateTracker:
    """
    Tracks the 12 state features used by TradingEnvironmentV19.

    State features:
    0. position: Current position [-1, 1]
    1. unrealized_pnl: Normalized PnL
    2. cumulative_return: Cumulative return
    3. current_drawdown: Current drawdown
    4. max_drawdown_episode: Max DD in episode
    5. regime_encoded: Market regime
    6. session_phase: Session phase [0-1]
    7. volatility_regime: Volatility bucket
    8. cost_regime: Cost multiplier (always 1.0)
    9. position_duration: Bars in position
    10. trade_count_normalized: Trades / 50
    11. time_remaining: Time left in episode
    """

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
        self.vol_percentiles = [0.5, 1.0, 1.5, 2.0, 2.5]

    def update(self, price: float, atr_pct: float, vix_z: float = 0.0, embi_z: float = 0.0):
        """Update state based on current market."""
        # Unrealized PnL
        if self.position != 0 and self.entry_price is not None:
            if self.position > 0:  # LONG
                pnl = (price - self.entry_price) / self.entry_price
            else:  # SHORT
                pnl = (self.entry_price - price) / self.entry_price
            self.unrealized_pnl = np.clip(pnl, -1, 1)
        else:
            self.unrealized_pnl = 0.0

        # Drawdown
        total_value = self.portfolio_value * (1 + self.unrealized_pnl * abs(self.position))
        if total_value > self.peak_value:
            self.peak_value = total_value
        self.current_drawdown = (self.peak_value - total_value) / self.peak_value
        self.max_drawdown_episode = max(self.max_drawdown_episode, self.current_drawdown)

        # Position duration
        if self.position != 0:
            self.position_duration += 1

    def get_state_features(self, episode_length: int = 400) -> np.ndarray:
        """Get 12 state features."""
        state = np.zeros(12, dtype=np.float32)

        state[0] = self.position
        state[1] = self.unrealized_pnl
        state[2] = np.clip(self.cumulative_return, -1, 1)
        state[3] = -self.current_drawdown  # Negative for loss
        state[4] = -self.max_drawdown_episode
        state[5] = 0.5  # Regime encoded (neutral)
        state[6] = 0.5  # Session phase (mid-session)
        state[7] = 0.5  # Volatility regime (medium)
        state[8] = 1.0  # Cost regime (production = 1.0)
        state[9] = min(self.position_duration / 50.0, 1.0)
        state[10] = min(self.trade_count / 50.0, 1.0)
        state[11] = max(0, 1.0 - self.current_step / episode_length)

        return state

    def open_position(self, side: str, price: float):
        """Open a new position."""
        self.position = 1.0 if side == "LONG" else -1.0
        self.entry_price = price
        self.position_duration = 0
        self.trade_count += 1

    def close_position(self, price: float) -> float:
        """Close position and return PnL."""
        if self.position == 0 or self.entry_price is None:
            return 0.0

        if self.position > 0:  # LONG
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:  # SHORT
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

class PaperTradingEngineV2:
    """Paper trading engine with correct 32-dim observations."""

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
        """Load the PPO model."""
        try:
            from stable_baselines3 import PPO
            logger.info(f"Loading model from {MODEL_PATH}")
            self.model = PPO.load(str(MODEL_PATH))
            logger.info(f"Model loaded - obs_space: {self.model.observation_space.shape}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def build_observation(self, row: pd.Series, feature_means: Dict, feature_stds: Dict) -> np.ndarray:
        """
        Build 32-dim observation vector.
        Order: [12 state features] + [20 market features]
        """
        obs = np.zeros(32, dtype=np.float32)

        # State features (0-11)
        state = self.state_tracker.get_state_features()
        obs[:12] = state

        # Market features (12-31) - normalized
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
        """Get action from model."""
        action, _ = self.model.predict(observation, deterministic=True)
        action_value = float(action[0])

        # Discretize: >0.1=LONG, <-0.1=SHORT, else=HOLD
        if action_value > 0.1:
            return action_value, "LONG"
        elif action_value < -0.1:
            return action_value, "SHORT"
        else:
            return action_value, "FLAT"

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to execution price."""
        slip = price * (self.slippage_bps / 10000)
        return price + slip if is_buy else price - slip

    def run_simulation(self, ohlcv_df: pd.DataFrame) -> Dict:
        """Run the paper trading simulation."""
        logger.info(f"Starting simulation with {len(ohlcv_df)} bars")

        self.load_model()

        # Calculate features
        features_df = self.feature_calc.build_features(ohlcv_df)

        # Compute normalization stats from data
        feature_means = {}
        feature_stds = {}
        for feat in MARKET_FEATURES:
            if feat in features_df.columns:
                feature_means[feat] = float(features_df[feat].mean())
                feature_stds[feat] = float(features_df[feat].std()) + 1e-8

        # Warmup period for indicators
        warmup = 50

        current_position = "FLAT"
        entry_price = None
        entry_time = None
        entry_bar = None

        for i in range(warmup, len(features_df)):
            row = features_df.iloc[i]
            timestamp = row['time']
            price = float(row['close'])
            atr = float(row.get('atr_pct', 0.05))

            self.state_tracker.current_step = i - warmup
            self.state_tracker.update(price, atr)

            # Build observation
            obs = self.build_observation(row, feature_means, feature_stds)

            # Get model action
            action_value, desired_position = self.get_action(obs)

            # Execute trade if position changes
            if desired_position != current_position:
                # Close existing position
                if current_position != "FLAT":
                    exec_price = self.apply_slippage(price, current_position == "SHORT")
                    pnl_usd = self.state_tracker.close_position(exec_price)

                    # Record trade
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

                # Open new position
                if desired_position != "FLAT":
                    exec_price = self.apply_slippage(price, desired_position == "LONG")
                    self.state_tracker.open_position(desired_position, exec_price)
                    entry_price = exec_price
                    entry_time = timestamp
                    entry_bar = i
                    logger.info(f"Position opened: {desired_position} @ {exec_price:.2f}")

                current_position = desired_position

            # Record equity snapshot (every 5 bars)
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

        # Close any remaining position
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

        # Calculate metrics
        metrics = self.calculate_metrics()

        logger.info("=" * 50)
        logger.info("SIMULATION COMPLETE")
        logger.info(f"Final Equity: ${self.state_tracker.portfolio_value:,.2f}")
        logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']:.1f}%")
        logger.info("=" * 50)

        return {
            "config": self.config,
            "metrics": metrics,
            "trades": self.trades,
            "equity_snapshots": self.equity_snapshots
        }

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        total_return = (self.state_tracker.portfolio_value - self.initial_capital) / self.initial_capital * 100

        winning = sum(1 for t in self.trades if t["pnl_usd"] > 0)
        losing = sum(1 for t in self.trades if t["pnl_usd"] <= 0)
        win_rate = (winning / len(self.trades) * 100) if self.trades else 0

        # Sharpe
        if self.equity_snapshots:
            equities = [s["equity"] for s in self.equity_snapshots]
            if len(equities) > 1:
                returns = np.diff(equities) / equities[:-1]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 59)
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Max DD
        max_dd = max([s["drawdown_pct"] for s in self.equity_snapshots]) if self.equity_snapshots else 0

        # Profit factor
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
    df = pd.read_sql(query, conn, params=[start_date, end_date])
    conn.close()

    logger.info(f"Loaded {len(df)} bars")
    return df


def save_results_to_db(results: Dict):
    """Save results to database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        model_id = results["config"]["model_id"]
        metrics = results["metrics"]

        # Update trading_state
        cursor.execute("""
            UPDATE trading_state SET
                position = 'FLAT',
                entry_price = NULL,
                entry_time = NULL,
                bars_in_position = 0,
                unrealized_pnl = 0,
                realized_pnl = %s,
                equity = %s,
                peak_equity = %s,
                drawdown_pct = %s,
                trade_count = %s,
                winning_trades = %s,
                losing_trades = %s,
                last_signal = 'HOLD',
                last_updated = NOW()
            WHERE model_id = %s
        """, (
            metrics["realized_pnl"],
            metrics["final_equity"],
            metrics["final_equity"],
            metrics["current_drawdown_pct"],
            metrics["total_trades"],
            metrics["winning_trades"],
            metrics["losing_trades"],
            model_id
        ))

        # Clear old data
        cursor.execute("DELETE FROM trades_history WHERE model_id = %s", (model_id,))
        cursor.execute("DELETE FROM equity_snapshots WHERE model_id = %s", (model_id,))

        # Insert trades
        if results["trades"]:
            trades_data = [
                (
                    t["model_id"], t["side"], t["entry_price"], t["exit_price"],
                    t["entry_time"], t["exit_time"], t["duration_bars"],
                    t["pnl_usd"], t["pnl_pct"], t["exit_reason"],
                    t["equity_at_entry"], t["equity_at_exit"],
                    t["drawdown_at_entry"], t["bar_number"]
                )
                for t in results["trades"]
            ]

            execute_batch(cursor, """
                INSERT INTO trades_history (
                    model_id, side, entry_price, exit_price,
                    entry_time, exit_time, duration_bars,
                    pnl_usd, pnl_pct, exit_reason,
                    equity_at_entry, equity_at_exit,
                    drawdown_at_entry, bar_number
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, trades_data)

        # Insert equity snapshots
        if results["equity_snapshots"]:
            snapshot_data = [
                (
                    s["model_id"], s["timestamp"], s["equity"],
                    s["drawdown_pct"], s["position"], s["bar_close_price"]
                )
                for s in results["equity_snapshots"]
            ]

            execute_batch(cursor, """
                INSERT INTO equity_snapshots (
                    model_id, timestamp, equity,
                    drawdown_pct, position, bar_close_price
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id, timestamp) DO UPDATE SET
                    equity = EXCLUDED.equity,
                    drawdown_pct = EXCLUDED.drawdown_pct,
                    position = EXCLUDED.position
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


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("PAPER TRADING SIMULATION V2 - 32-DIM OBSERVATION")
    print("=" * 60)

    ohlcv_df = load_ohlcv_data(
        CONFIG["out_of_sample_start"],
        CONFIG["out_of_sample_end"]
    )

    if len(ohlcv_df) == 0:
        logger.error("No OHLCV data found")
        return

    engine = PaperTradingEngineV2(CONFIG)
    results = engine.run_simulation(ohlcv_df)

    save_results_to_db(results)

    print("\n" + "=" * 60)
    print("RESULTS SAVED TO DATABASE")
    print("=" * 60)
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
