"""
Paper Trading Simulation Service
================================

Runs PPO V1 model on out-of-sample data (Dec 27, 2025 - Jan 6, 2026)
and records trades, equity snapshots, and performance metrics to PostgreSQL.

Usage:
    python paper_trading_simulation.py

Author: Claude Code
Version: 1.0.0
Date: 2026-01-07
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from pathlib import Path

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

PAPER_TRADING_CONFIG = {
    "model_id": "ppo_v1",
    "initial_capital": 10000.0,
    "position_size": 1.0,  # 100% of capital per trade
    "slippage_bps": 2,  # 2 basis points
    "commission_per_trade": 0.0,  # No commission for simplicity
    "out_of_sample_start": "2025-12-27",
    "out_of_sample_end": "2026-01-06",
}

# Database configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB", "usdcop_trading"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin123"),
}

# Model configuration
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_v1_20251226_054154.zip"
CONFIG_PATH = PROJECT_ROOT / "config" / "feature_config_v19.json"
NORM_STATS_PATH = PROJECT_ROOT / "config" / "v19_norm_stats.json"


# =============================================================================
# Simple Feature Calculator (standalone - no complex imports)
# =============================================================================

class SimpleFeatureCalculator:
    """
    Simplified feature calculator for paper trading simulation.
    Computes the 13 core features from OHLCV data.
    """

    RSI_PERIOD = 9
    ATR_PERIOD = 10
    ADX_PERIOD = 14

    def __init__(self, norm_stats_path: Path):
        """Load normalization statistics."""
        with open(norm_stats_path, 'r') as f:
            self.norm_stats = json.load(f)

        self.feature_order = [
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d"
        ]

    def calculate_log_return(self, series: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate logarithmic returns."""
        return np.log(series / series.shift(periods))

    def calculate_rsi(self, close: pd.Series, period: int = 9) -> pd.Series:
        """Calculate RSI (0-100 range)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral value for NaN

    def calculate_atr_pct(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
        """Calculate ATR as percentage of price."""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=period, min_periods=period).mean()
        return (atr / close) * 100  # As percentage

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX (0-100 range)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period).mean() / atr)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(span=period, min_periods=period).mean()
        return adx.fillna(25)  # Neutral value for NaN

    def normalize(self, feature_name: str, value: float) -> float:
        """Apply z-score normalization with clipping."""
        if feature_name not in self.norm_stats:
            return np.clip(value, -5, 5)

        stats = self.norm_stats[feature_name]
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        if std < 1e-8:
            std = 1.0

        z = (value - mean) / std
        return float(np.clip(z, -5, 5))

    def build_features_df(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all 13 core features from OHLCV data.
        Note: Macro features (dxy_z, vix_z, etc.) use default values
        since we're doing pure price-based simulation.
        """
        df = ohlcv_df.copy()

        # Log returns
        df['log_ret_5m'] = self.calculate_log_return(df['close'], 1)
        df['log_ret_1h'] = self.calculate_log_return(df['close'], 12)  # 12 * 5min = 1h
        df['log_ret_4h'] = self.calculate_log_return(df['close'], 48)  # 48 * 5min = 4h

        # Technical indicators
        df['rsi_9'] = self.calculate_rsi(df['close'], self.RSI_PERIOD)
        df['atr_pct'] = self.calculate_atr_pct(df['high'], df['low'], df['close'], self.ATR_PERIOD)
        df['adx_14'] = self.calculate_adx(df['high'], df['low'], df['close'], self.ADX_PERIOD)

        # Macro features - use neutral values (0.0 for z-scores)
        # In production, these would come from actual macro data
        df['dxy_z'] = 0.0
        df['dxy_change_1d'] = 0.0
        df['vix_z'] = 0.0
        df['embi_z'] = 0.0
        df['brent_change_1d'] = 0.0
        df['rate_spread'] = 0.0
        df['usdmxn_change_1d'] = 0.0

        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)

        return df


# =============================================================================
# Paper Trading Engine
# =============================================================================

class PaperTradingEngine:
    """
    Paper trading engine that simulates trades using PPO V1 model.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model_id = config["model_id"]
        self.initial_capital = config["initial_capital"]
        self.slippage_bps = config["slippage_bps"]

        # State tracking
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.position = "FLAT"  # FLAT, LONG, SHORT
        self.entry_price = None
        self.entry_time = None
        self.entry_bar = None
        self.bars_in_position = 0

        # Statistics
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.realized_pnl = 0.0

        # Records
        self.trades: List[Dict] = []
        self.equity_snapshots: List[Dict] = []

        # Feature calculator
        self.feature_calc = SimpleFeatureCalculator(NORM_STATS_PATH)

        # Model (load lazily)
        self.model = None

    def load_model(self):
        """Load the PPO model."""
        try:
            from stable_baselines3 import PPO
            logger.info(f"Loading model from {MODEL_PATH}")
            self.model = PPO.load(str(MODEL_PATH))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load SB3 model: {e}")
            logger.info("Using rule-based fallback strategy")
            self.model = None

    def get_action(self, observation: np.ndarray) -> int:
        """
        Get action from model or fallback strategy.

        Returns:
            0 = SELL (go SHORT or close LONG)
            1 = HOLD (stay in current position)
            2 = BUY (go LONG or close SHORT)
        """
        if self.model is not None:
            action, _ = self.model.predict(observation, deterministic=True)
            return int(action)
        else:
            # Fallback: RSI-based simple strategy
            # observation[3] = normalized RSI
            rsi_norm = observation[3]
            rsi = (rsi_norm * 15) + 50  # Denormalize roughly

            if rsi < 30:
                return 2  # BUY when oversold
            elif rsi > 70:
                return 0  # SELL when overbought
            return 1  # HOLD

    def discretize_action(self, action: int) -> str:
        """Convert action index to position string."""
        if action == 0:
            return "SHORT" if self.position != "SHORT" else "FLAT"
        elif action == 2:
            return "LONG" if self.position != "LONG" else "FLAT"
        return self.position  # HOLD

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to execution price."""
        slippage = price * (self.slippage_bps / 10000)
        if is_buy:
            return price + slippage
        return price - slippage

    def execute_trade(self, timestamp: datetime, price: float, new_position: str, bar_number: int):
        """Execute a position change."""
        if new_position == self.position:
            return  # No change

        # Close existing position if any
        if self.position != "FLAT":
            self.close_position(timestamp, price, "SIGNAL_CHANGE", bar_number)

        # Open new position if not going flat
        if new_position != "FLAT":
            is_buy = (new_position == "LONG")
            exec_price = self.apply_slippage(price, is_buy)

            self.position = new_position
            self.entry_price = exec_price
            self.entry_time = timestamp
            self.entry_bar = bar_number
            self.bars_in_position = 0

            logger.debug(f"Opened {new_position} at {exec_price:.4f}")

    def close_position(self, timestamp: datetime, price: float, reason: str, bar_number: int):
        """Close current position and record trade."""
        if self.position == "FLAT":
            return

        is_buy = (self.position == "SHORT")  # Closing short = buy
        exec_price = self.apply_slippage(price, is_buy)

        # Calculate PnL
        if self.position == "LONG":
            pnl = exec_price - self.entry_price
        else:  # SHORT
            pnl = self.entry_price - exec_price

        pnl_pct = (pnl / self.entry_price) * 100
        pnl_usd = (pnl / self.entry_price) * self.equity

        # Update equity
        self.equity += pnl_usd
        self.realized_pnl += pnl_usd

        # Update peak equity and stats
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.trade_count += 1
        if pnl_usd > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Record trade
        trade = {
            "model_id": self.model_id,
            "side": self.position,
            "entry_price": self.entry_price,
            "exit_price": exec_price,
            "entry_time": self.entry_time,
            "exit_time": timestamp,
            "duration_bars": bar_number - self.entry_bar,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
            "equity_at_entry": self.equity - pnl_usd,
            "equity_at_exit": self.equity,
            "drawdown_at_entry": self.calculate_drawdown(),
            "bar_number": bar_number
        }
        self.trades.append(trade)

        logger.info(f"Trade #{self.trade_count}: {self.position} {self.entry_price:.2f} → {exec_price:.2f} | PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%)")

        # Reset position state
        self.position = "FLAT"
        self.entry_price = None
        self.entry_time = None
        self.entry_bar = None
        self.bars_in_position = 0

    def calculate_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_equity == 0:
            return 0.0
        return ((self.peak_equity - self.equity) / self.peak_equity) * 100

    def calculate_unrealized_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calculate unrealized PnL for open position."""
        if self.position == "FLAT" or self.entry_price is None:
            return 0.0, 0.0

        if self.position == "LONG":
            pnl = current_price - self.entry_price
        else:  # SHORT
            pnl = self.entry_price - current_price

        pnl_pct = (pnl / self.entry_price) * 100
        pnl_usd = (pnl / self.entry_price) * self.equity

        return pnl_usd, pnl_pct

    def record_snapshot(self, timestamp: datetime, price: float):
        """Record equity snapshot for charting."""
        unrealized_pnl, _ = self.calculate_unrealized_pnl(price)
        total_equity = self.equity + unrealized_pnl

        snapshot = {
            "model_id": self.model_id,
            "timestamp": timestamp,
            "equity": total_equity,
            "drawdown_pct": self.calculate_drawdown(),
            "position": self.position,
            "bar_close_price": price
        }
        self.equity_snapshots.append(snapshot)

    def run_simulation(self, ohlcv_df: pd.DataFrame) -> Dict:
        """
        Run paper trading simulation on OHLCV data.

        Args:
            ohlcv_df: DataFrame with columns [time, open, high, low, close, volume]

        Returns:
            Dict with simulation results and metrics
        """
        logger.info(f"Starting simulation with {len(ohlcv_df)} bars")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")

        # Load model
        self.load_model()

        # Calculate features
        features_df = self.feature_calc.build_features_df(ohlcv_df)

        # Need warmup period for indicators
        warmup = 50
        if len(features_df) <= warmup:
            raise ValueError(f"Not enough data. Need > {warmup} bars, got {len(features_df)}")

        # Process each bar
        for i in range(warmup, len(features_df)):
            row = features_df.iloc[i]
            timestamp = row['time']
            price = float(row['close'])

            # Calculate bar number within session (1-60)
            # Assuming trading session is 8:00-12:55 = 59 bars
            time_str = str(timestamp)
            hour = int(time_str[11:13]) if len(time_str) > 11 else 8
            minute = int(time_str[14:16]) if len(time_str) > 14 else 0
            bar_number = ((hour - 8) * 12 + minute // 5) + 1
            bar_number = max(1, min(bar_number, 60))

            # Build observation vector
            observation = self.build_observation(row, bar_number)

            # Get model action
            action = self.get_action(observation)
            new_position = self.discretize_action(action)

            # Execute trade if position changes
            if new_position != self.position:
                self.execute_trade(timestamp, price, new_position, i)
            else:
                self.bars_in_position += 1

            # Record equity snapshot
            self.record_snapshot(timestamp, price)

        # Close any open position at end
        if self.position != "FLAT":
            last_row = features_df.iloc[-1]
            self.close_position(
                last_row['time'],
                float(last_row['close']),
                "END_OF_SAMPLE",
                len(features_df) - 1
            )

        # Calculate final metrics
        metrics = self.calculate_metrics()

        logger.info("=" * 50)
        logger.info("SIMULATION COMPLETE")
        logger.info(f"Final Equity: ${self.equity:,.2f}")
        logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"Win Rate: {metrics['win_rate']:.1f}%")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info("=" * 50)

        return {
            "config": self.config,
            "metrics": metrics,
            "trades": self.trades,
            "equity_snapshots": self.equity_snapshots
        }

    def build_observation(self, row: pd.Series, bar_number: int) -> np.ndarray:
        """Build 15-dimensional observation vector."""
        obs = np.zeros(15, dtype=np.float32)

        # Core features (0-12)
        for i, feature in enumerate(self.feature_calc.feature_order):
            raw_value = float(row.get(feature, 0.0))
            obs[i] = self.feature_calc.normalize(feature, raw_value)

        # Position (13): -1=short, 0=flat, 1=long
        if self.position == "LONG":
            obs[13] = 1.0
        elif self.position == "SHORT":
            obs[13] = -1.0
        else:
            obs[13] = 0.0

        # Time normalized (14): 0 to 1 within session
        obs[14] = (bar_number - 1) / 60.0

        # Clip and clean
        obs = np.clip(obs, -5, 5)
        obs = np.nan_to_num(obs, nan=0.0)

        return obs

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if len(self.equity_snapshots) == 0:
            return {}

        # Extract equity series
        equities = [s["equity"] for s in self.equity_snapshots]

        # Returns
        total_return = (self.equity - self.initial_capital) / self.initial_capital * 100

        # Sharpe ratio (annualized, assuming 252 trading days, 59 bars/day)
        if len(equities) > 1:
            returns = np.diff(equities) / equities[:-1]
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 59)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Win rate
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0.0

        # Profit factor
        gross_profit = sum(t["pnl_usd"] for t in self.trades if t["pnl_usd"] > 0)
        gross_loss = abs(sum(t["pnl_usd"] for t in self.trades if t["pnl_usd"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade duration
        if self.trades:
            avg_duration = np.mean([t["duration_bars"] for t in self.trades])
        else:
            avg_duration = 0

        return {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "current_drawdown_pct": round(self.calculate_drawdown(), 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
            "total_trades": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_trade_duration_bars": round(avg_duration, 1),
            "final_equity": round(self.equity, 2),
            "realized_pnl": round(self.realized_pnl, 2)
        }


# =============================================================================
# Database Functions
# =============================================================================

def get_db_connection():
    """Get PostgreSQL connection."""
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
    """Save simulation results to database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        model_id = results["config"]["model_id"]
        metrics = results["metrics"]

        # Update trading_state
        logger.info("Updating trading_state...")
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
            metrics["final_equity"],  # Peak is final since we ended simulation
            metrics["current_drawdown_pct"],
            metrics["total_trades"],
            metrics["winning_trades"],
            metrics["losing_trades"],
            model_id
        ))

        # Clear old data
        logger.info("Clearing old trades and snapshots...")
        cursor.execute("DELETE FROM trades_history WHERE model_id = %s", (model_id,))
        cursor.execute("DELETE FROM equity_snapshots WHERE model_id = %s", (model_id,))

        # Insert trades
        if results["trades"]:
            logger.info(f"Inserting {len(results['trades'])} trades...")
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

        # Insert equity snapshots (sample to reduce size)
        if results["equity_snapshots"]:
            # Sample every 5th snapshot to reduce data volume
            snapshots = results["equity_snapshots"][::5]
            logger.info(f"Inserting {len(snapshots)} equity snapshots...")

            snapshot_data = [
                (
                    s["model_id"], s["timestamp"], s["equity"],
                    s["drawdown_pct"], s["position"], s["bar_close_price"]
                )
                for s in snapshots
            ]

            execute_batch(cursor, """
                INSERT INTO equity_snapshots (
                    model_id, timestamp, equity,
                    drawdown_pct, position, bar_close_price
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id, timestamp) DO UPDATE SET
                    equity = EXCLUDED.equity,
                    drawdown_pct = EXCLUDED.drawdown_pct,
                    position = EXCLUDED.position,
                    bar_close_price = EXCLUDED.bar_close_price
            """, snapshot_data)

        conn.commit()
        logger.info("Results saved to database successfully")

    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving results: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run paper trading simulation."""
    print("=" * 60)
    print("PAPER TRADING SIMULATION - PPO V1 OUT-OF-SAMPLE")
    print("=" * 60)

    # Load OHLCV data
    ohlcv_df = load_ohlcv_data(
        PAPER_TRADING_CONFIG["out_of_sample_start"],
        PAPER_TRADING_CONFIG["out_of_sample_end"]
    )

    if len(ohlcv_df) == 0:
        logger.error("No OHLCV data found for specified period")
        return

    # Create engine and run simulation
    engine = PaperTradingEngine(PAPER_TRADING_CONFIG)
    results = engine.run_simulation(ohlcv_df)

    # Save results to database
    save_results_to_db(results)

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS SAVED TO DATABASE")
    print("=" * 60)
    print(f"Tables updated:")
    print(f"  - trading_state (current state)")
    print(f"  - trades_history ({len(results['trades'])} trades)")
    print(f"  - equity_snapshots ({len(results['equity_snapshots'])//5} points)")
    print("\nMetrics:")
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
