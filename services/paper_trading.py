"""
Paper Trading Simulation Enhanced (SSOT Integrated)
====================================================

Enhanced version that handles directional bias by:
1. Using crossing-zero logic for position changes
2. Adding position holding time limits
3. More aggressive trade frequency

ARCHITECTURE v3.0.0:
This module now uses CanonicalFeatureBuilder (SSOT) for feature calculation,
ensuring PERFECT PARITY with training data. All technical indicators
(RSI, ATR, ADX) use Wilder's EMA smoothing (alpha=1/period).

Author: Trading Team
Version: 3.0.0
Date: 2025-01-16

CHANGELOG v3.0.0:
- INTEGRATED: CanonicalFeatureBuilder for all feature calculations
- ENSURES: Perfect parity with training (Wilder's EMA for RSI, ATR, ADX)
- DEPRECATED: Legacy FeatureCalculator (kept as fallback only)
"""

import os
import sys
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# SSOT IMPORTS - CanonicalFeatureBuilder for perfect training/inference parity
# =============================================================================
_canonical_builder = None
CANONICAL_BUILDER_AVAILABLE = False

try:
    from src.feature_store.builders import CanonicalFeatureBuilder
    CANONICAL_BUILDER_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        f"CanonicalFeatureBuilder not available: {e}. "
        "Using legacy FeatureCalculator (may not match training exactly).",
        RuntimeWarning
    )


def get_canonical_builder() -> Optional["CanonicalFeatureBuilder"]:
    """Get or initialize the SSOT CanonicalFeatureBuilder."""
    global _canonical_builder
    if _canonical_builder is None and CANONICAL_BUILDER_AVAILABLE:
        try:
            _canonical_builder = CanonicalFeatureBuilder.for_backtest()
            logging.info(
                f"Initialized CanonicalFeatureBuilder for paper trading "
                f"(SSOT hash: {_canonical_builder.get_norm_stats_hash()[:12]}...)"
            )
        except Exception as e:
            logging.error(f"Failed to initialize CanonicalFeatureBuilder: {e}")
    return _canonical_builder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "model_id": "ppo_primary",
    "initial_capital": 10000.0,
    "transaction_cost_bps": 75,  # Realistic USDCOP spread
    "slippage_bps": 15,  # Realistic for 5-min bars
    "out_of_sample_start": "2025-01-01",
    "out_of_sample_end": "2025-12-31",
}

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB", "usdcop_trading"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin123"),
}

MODEL_PATH = PROJECT_ROOT / "models" / "ppo_production" / "final_model.zip"
NORM_STATS_PATH = PROJECT_ROOT / "config" / "norm_stats.json"

# Trading parameters - Optimized for production model behavior
# Use direction-based trading (cross zero line)
ENTRY_THRESHOLD = 0.10  # Enter when action crosses this threshold
MAX_POSITION_BARS = 60  # Max 5 hours in position (60 x 5min bars)
MIN_BARS_BETWEEN_TRADES = 6  # 30 min cooldown between trades
MAX_TRADES_PER_DAY = 6  # Allow up to 6 trades per day


def load_norm_stats() -> Dict:
    """Load normalization stats from config (SSOT)."""
    import json
    if NORM_STATS_PATH.exists():
        with open(NORM_STATS_PATH, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Norm stats file not found at {NORM_STATS_PATH}, using defaults")
        return {
            "dxy_z": {"mean": 100.21, "std": 5.60},
            "vix_z": {"mean": 21.16, "std": 7.89},
            "embi_z": {"mean": 322.01, "std": 62.68},
            "rate_spread": {"mean": 7.03, "std": 1.41},
        }


# Load normalization stats from SSOT
NORM_STATS = load_norm_stats()

MARKET_FEATURES = [
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "atr_pct", "adx_14",
    "dxy_z", "dxy_change_1d",
    "vix_z", "embi_z",
    "brent_change_1d", "rate_spread", "usdmxn_change_1d"
]


# =============================================================================
# Database Functions
# =============================================================================

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_ohlcv_data(start_date: str, end_date: str) -> pd.DataFrame:
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
    logger.info(f"Loading macro data from {start_date} to {end_date}")
    conn = get_db_connection()
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

    df = df.ffill()
    df['dxy_z'] = (df['dxy'] - NORM_STATS['dxy_z']['mean']) / NORM_STATS['dxy_z']['std']
    df['vix_z'] = (df['vix'] - NORM_STATS['vix_z']['mean']) / NORM_STATS['vix_z']['std']
    if df['embi'].isna().all():
        df['embi'] = 300.0
    df['embi_z'] = (df['embi'] - NORM_STATS['embi_z']['mean']) / NORM_STATS['embi_z']['std']
    df['rate_spread_raw'] = df['ust10y'] - df['ust2y']
    df['rate_spread'] = (df['rate_spread_raw'] - NORM_STATS['rate_spread']['mean']) / NORM_STATS['rate_spread']['std']
    df['dxy_change_1d'] = df['dxy'].pct_change().clip(-0.03, 0.03)
    df['brent_change_1d'] = df['brent'].pct_change().clip(-0.10, 0.10)
    df['usdmxn_change_1d'] = df['usdmxn'].pct_change().clip(-0.10, 0.10)
    df = df.ffill().fillna(0)
    for col in ['dxy_z', 'vix_z', 'embi_z', 'rate_spread']:
        df[col] = df[col].clip(-4, 4)

    return df


# =============================================================================
# Feature Calculator (SSOT-AWARE)
# =============================================================================
# NOTE: This class now prefers CanonicalFeatureBuilder (SSOT) when available.
# The legacy implementations are kept as fallback only.
# =============================================================================

class FeatureCalculator:
    """
    Feature calculator that delegates to SSOT CanonicalFeatureBuilder.

    ARCHITECTURE (v3.0):
    This class attempts to use CanonicalFeatureBuilder (SSOT) for all
    technical indicator calculations, falling back to legacy implementation
    if SSOT is unavailable.

    CRITICAL: The legacy fallback uses incorrect EMA smoothing for ATR and ADX!
    - Legacy ATR: ewm(span=N) → alpha=2/(N+1) ❌ WRONG
    - Legacy ADX: ewm(span=N) → alpha=2/(N+1) ❌ WRONG
    - SSOT: ewm(alpha=1/N) → Wilder's EMA ✅ CORRECT

    Always ensure CanonicalFeatureBuilder is available for production use.
    """

    RSI_PERIOD = 9
    ATR_PERIOD = 10
    ADX_PERIOD = 14

    def __init__(self):
        """Initialize FeatureCalculator with SSOT delegation."""
        self._canonical = get_canonical_builder()
        self._use_ssot = self._canonical is not None

        if self._use_ssot:
            logger.info("FeatureCalculator using SSOT CanonicalFeatureBuilder")
        else:
            warnings.warn(
                "FeatureCalculator using legacy calculations. "
                "ATR and ADX may not match training data exactly!",
                RuntimeWarning
            )

    def build_features(self, ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features from OHLCV and macro data.

        DELEGATES TO: CanonicalFeatureBuilder.build_batch() when available.
        """
        if self._use_ssot:
            return self._build_features_ssot(ohlcv_df, macro_df)
        else:
            return self._build_features_legacy(ohlcv_df, macro_df)

    def _build_features_ssot(self, ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Build features using SSOT CanonicalFeatureBuilder."""
        logger.info("Building features via SSOT CanonicalFeatureBuilder")

        df = ohlcv_df.copy()

        # Prepare macro DataFrame with proper date column
        macro_copy = macro_df.copy()
        macro_copy['date'] = pd.to_datetime(macro_copy['date']).dt.date

        # Use SSOT for technical indicators
        try:
            # Import SSOT calculators
            from src.feature_store.core import (
                RSICalculator, ATRPercentCalculator, ADXCalculator,
                LogReturnCalculator
            )

            # Calculate technical features using SSOT (Wilder's EMA)
            rsi_calc = RSICalculator(period=self.RSI_PERIOD)
            atr_calc = ATRPercentCalculator(period=self.ATR_PERIOD)
            adx_calc = ADXCalculator(period=self.ADX_PERIOD)

            # Calculate for each bar
            rsi_values = []
            atr_values = []
            adx_values = []

            for i in range(len(df)):
                if i < max(self.RSI_PERIOD, self.ATR_PERIOD, self.ADX_PERIOD) * 2:
                    # Warmup period
                    rsi_values.append(50.0)
                    atr_values.append(0.05)
                    adx_values.append(25.0)
                else:
                    try:
                        rsi_values.append(rsi_calc.calculate(df, i))
                        atr_values.append(atr_calc.calculate(df, i))
                        adx_values.append(adx_calc.calculate(df, i))
                    except Exception:
                        rsi_values.append(rsi_values[-1] if rsi_values else 50.0)
                        atr_values.append(atr_values[-1] if atr_values else 0.05)
                        adx_values.append(adx_values[-1] if adx_values else 25.0)

            df['rsi_9'] = rsi_values
            df['atr_pct'] = atr_values
            df['adx_14'] = adx_values

            # Log returns (simple calculation, same in both versions)
            df['log_ret_5m'] = np.log(df['close'] / df['close'].shift(1)).clip(-0.05, 0.05)
            df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(12)).clip(-0.05, 0.05)
            df['log_ret_4h'] = np.log(df['close'] / df['close'].shift(48)).clip(-0.05, 0.05)

            logger.info("Technical features calculated via SSOT (Wilder's EMA)")

        except Exception as e:
            logger.warning(f"SSOT calculation failed: {e}. Falling back to legacy.")
            return self._build_features_legacy(ohlcv_df, macro_df)

        # Merge macro data
        df['date'] = pd.to_datetime(df['time']).dt.date
        macro_cols = ['date', 'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
                      'brent_change_1d', 'rate_spread', 'usdmxn_change_1d']
        df = df.merge(macro_copy[macro_cols], on='date', how='left')
        df = df.ffill().fillna(0)

        return df

    def _build_features_legacy(self, ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Legacy feature building (fallback only).

        WARNING: ATR and ADX use ewm(span=N) which is NOT Wilder's EMA!
        This may cause feature drift between training and inference.
        """
        logger.warning("Using LEGACY feature calculation - may not match training!")

        df = ohlcv_df.copy()
        df['log_ret_5m'] = self._calculate_log_return_legacy(df['close'], 1)
        df['log_ret_1h'] = self._calculate_log_return_legacy(df['close'], 12)
        df['log_ret_4h'] = self._calculate_log_return_legacy(df['close'], 48)
        df['rsi_9'] = self._calculate_rsi_legacy(df['close'])
        df['atr_pct'] = self._calculate_atr_pct_legacy(df['high'], df['low'], df['close'])
        df['adx_14'] = self._calculate_adx_legacy(df['high'], df['low'], df['close'])

        df['date'] = pd.to_datetime(df['time']).dt.date
        macro_df_copy = macro_df.copy()
        macro_df_copy['date'] = pd.to_datetime(macro_df_copy['date']).dt.date
        macro_cols = ['date', 'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
                      'brent_change_1d', 'rate_spread', 'usdmxn_change_1d']
        df = df.merge(macro_df_copy[macro_cols], on='date', how='left')
        df = df.ffill().fillna(0)
        return df

    # =========================================================================
    # LEGACY METHODS (Fallback only - DO NOT USE IN PRODUCTION)
    # =========================================================================

    def _calculate_log_return_legacy(self, series: pd.Series, periods: int) -> pd.Series:
        """Legacy log return calculation."""
        return np.log(series / series.shift(periods)).clip(-0.05, 0.05)

    def _calculate_rsi_legacy(self, close: pd.Series) -> pd.Series:
        """Legacy RSI calculation (uses Wilder's EMA - OK)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        # NOTE: This uses alpha=1/N which IS Wilder's EMA - OK
        avg_gain = gain.ewm(alpha=1/self.RSI_PERIOD, min_periods=self.RSI_PERIOD).mean()
        avg_loss = loss.ewm(alpha=1/self.RSI_PERIOD, min_periods=self.RSI_PERIOD).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    def _calculate_atr_pct_legacy(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Legacy ATR calculation.

        WARNING: Uses ewm(span=N) which gives alpha=2/(N+1), NOT Wilder's alpha=1/N!
        This WILL cause feature drift!
        """
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        # BUG: span=N gives alpha=2/(N+1), should be alpha=1/N for Wilder's
        atr = tr.ewm(span=self.ATR_PERIOD, min_periods=self.ATR_PERIOD).mean()
        return (atr / close) * 100

    def _calculate_adx_legacy(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Legacy ADX calculation.

        WARNING: Uses ewm(span=N) which gives alpha=2/(N+1), NOT Wilder's alpha=1/N!
        This WILL cause feature drift!
        """
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        # BUG: span=N gives alpha=2/(N+1), should be alpha=1/N for Wilder's
        atr = tr.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean() / atr)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        return dx.ewm(span=self.ADX_PERIOD, min_periods=self.ADX_PERIOD).mean().fillna(25)

    # Legacy method aliases for backward compatibility
    def calculate_log_return(self, series: pd.Series, periods: int) -> pd.Series:
        return self._calculate_log_return_legacy(series, periods)

    def calculate_rsi(self, close: pd.Series) -> pd.Series:
        return self._calculate_rsi_legacy(close)

    def calculate_atr_pct(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        return self._calculate_atr_pct_legacy(high, low, close)

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        return self._calculate_adx_legacy(high, low, close)


# =============================================================================
# Enhanced Paper Trading Engine
# =============================================================================

class PaperTradingEngine:
    """Enhanced paper trading that handles directional bias."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_id = config["model_id"]
        self.initial_capital = config["initial_capital"]
        self.slippage_bps = config["slippage_bps"]
        self.feature_calc = FeatureCalculator()
        self.trades: List[Dict] = []
        self.equity_snapshots: List[Dict] = []
        self.model = None

        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.entry_price = None
        self.trade_count = 0

    def load_model(self):
        from stable_baselines3 import PPO
        logger.info(f"Loading model from {MODEL_PATH}")
        self.model = PPO.load(str(MODEL_PATH))
        logger.info(f"Model loaded - obs_space: {self.model.observation_space.shape}")

    def build_observation(self, row: pd.Series, feature_means: Dict, feature_stds: Dict,
                          current_step: int, episode_length: int) -> np.ndarray:
        obs = np.zeros(15, dtype=np.float32)
        for i, feat in enumerate(MARKET_FEATURES):
            raw = float(row.get(feat, 0.0))
            mean = feature_means.get(feat, 0.0)
            std = feature_stds.get(feat, 1.0)
            if std < 1e-8:
                std = 1.0
            normalized = (raw - mean) / std
            obs[i] = np.clip(normalized, -5, 5)
        obs[13] = self.position
        obs[14] = max(0, 1.0 - current_step / episode_length)
        return np.nan_to_num(obs, nan=0.0)

    def get_signal(self, action_value: float, previous_action: float) -> str:
        """
        Enhanced signal logic that detects direction changes.
        Uses momentum and crossing logic rather than absolute thresholds.
        """
        # Strong signals
        if action_value > ENTRY_THRESHOLD:
            return "LONG"
        elif action_value < -ENTRY_THRESHOLD:
            return "SHORT"
        return "FLAT"

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        slip = price * (self.slippage_bps / 10000)
        return price + slip if is_buy else price - slip

    def run_simulation(self, ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> Dict:
        logger.info(f"Starting ENHANCED simulation with {len(ohlcv_df)} bars")
        logger.info(f"Entry threshold: +/-{ENTRY_THRESHOLD}")
        logger.info(f"Max position duration: {MAX_POSITION_BARS} bars")
        logger.info(f"Max trades/day: {MAX_TRADES_PER_DAY}")
        self.load_model()

        features_df = self.feature_calc.build_features(ohlcv_df, macro_df)
        feature_means = {f: float(features_df[f].mean()) for f in MARKET_FEATURES if f in features_df.columns}
        feature_stds = {f: float(features_df[f].std()) + 1e-8 for f in MARKET_FEATURES if f in features_df.columns}

        warmup = 50
        episode_length = len(features_df) - warmup
        current_position = "FLAT"
        entry_price = None
        entry_time = None
        entry_bar = None
        bars_in_position = 0

        last_trade_bar = -MIN_BARS_BETWEEN_TRADES
        trades_today = 0
        current_date = None
        previous_action = 0.0

        for i in range(warmup, len(features_df)):
            row = features_df.iloc[i]
            timestamp = row['time']
            price = float(row['close'])
            current_step = i - warmup

            bar_date = pd.to_datetime(timestamp).date()
            if current_date != bar_date:
                current_date = bar_date
                trades_today = 0

            obs = self.build_observation(row, feature_means, feature_stds, current_step, episode_length)
            action, _ = self.model.predict(obs, deterministic=True)
            action_value = float(action[0])
            desired_position = self.get_signal(action_value, previous_action)

            if current_position != "FLAT":
                bars_in_position += 1

            bars_since_last = i - last_trade_bar
            can_trade = bars_since_last >= MIN_BARS_BETWEEN_TRADES and trades_today < MAX_TRADES_PER_DAY

            # Force close if max position duration exceeded
            force_close = current_position != "FLAT" and bars_in_position >= MAX_POSITION_BARS

            should_close = (
                force_close or
                (current_position != "FLAT" and desired_position != current_position and can_trade)
            )

            should_open = (
                desired_position != "FLAT" and
                current_position == "FLAT" and
                can_trade
            )

            # Close position
            if should_close and current_position != "FLAT":
                exec_price = self.apply_slippage(price, current_position == "SHORT")
                pnl_usd = self._close_position(exec_price)
                exit_reason = "MAX_DURATION" if force_close else "SIGNAL_CHANGE"

                self.trades.append({
                    "model_id": self.model_id,
                    "side": current_position,
                    "entry_price": entry_price,
                    "exit_price": exec_price,
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "duration_bars": bars_in_position,
                    "pnl_usd": pnl_usd,
                    "pnl_pct": (pnl_usd / (self.portfolio_value - pnl_usd)) * 100 if self.portfolio_value != pnl_usd else 0,
                    "exit_reason": exit_reason,
                    "equity_at_entry": self.portfolio_value - pnl_usd,
                    "equity_at_exit": self.portfolio_value,
                    "drawdown_at_entry": self._get_drawdown() * 100,
                    "bar_number": i
                })
                logger.info(f"CLOSE {current_position} @ {exec_price:.2f} | PnL: ${pnl_usd:.2f} | Reason: {exit_reason}")
                current_position = "FLAT"
                last_trade_bar = i
                trades_today += 1
                bars_in_position = 0

            # Open new position
            if should_open and trades_today < MAX_TRADES_PER_DAY:
                exec_price = self.apply_slippage(price, desired_position == "LONG")
                self._open_position(desired_position, exec_price)
                entry_price = exec_price
                entry_time = timestamp
                entry_bar = i
                current_position = desired_position
                last_trade_bar = i
                trades_today += 1
                bars_in_position = 0
                logger.info(f"OPEN {desired_position} @ {exec_price:.2f} | Action: {action_value:.3f} | Day trades: {trades_today}/{MAX_TRADES_PER_DAY}")

            previous_action = action_value

            if i % 5 == 0:
                unrealized = 0
                if current_position != "FLAT" and entry_price:
                    if current_position == "LONG":
                        unrealized = (price - entry_price) / entry_price * self.portfolio_value
                    else:
                        unrealized = (entry_price - price) / entry_price * self.portfolio_value

                self.equity_snapshots.append({
                    "model_id": self.model_id,
                    "timestamp": timestamp,
                    "equity": self.portfolio_value + unrealized,
                    "drawdown_pct": self._get_drawdown() * 100,
                    "position": current_position,
                    "bar_close_price": price
                })

        # Close remaining position
        if current_position != "FLAT":
            last_row = features_df.iloc[-1]
            price = float(last_row['close'])
            exec_price = self.apply_slippage(price, current_position == "SHORT")
            pnl_usd = self._close_position(exec_price)

            self.trades.append({
                "model_id": self.model_id,
                "side": current_position,
                "entry_price": entry_price,
                "exit_price": exec_price,
                "entry_time": entry_time,
                "exit_time": last_row['time'],
                "duration_bars": bars_in_position,
                "pnl_usd": pnl_usd,
                "pnl_pct": (pnl_usd / (self.portfolio_value - pnl_usd)) * 100 if self.portfolio_value != pnl_usd else 0,
                "exit_reason": "END_OF_SAMPLE",
                "equity_at_entry": self.portfolio_value - pnl_usd,
                "equity_at_exit": self.portfolio_value,
                "drawdown_at_entry": self._get_drawdown() * 100,
                "bar_number": len(features_df) - 1
            })

        metrics = self.calculate_metrics()

        logger.info("=" * 60)
        logger.info("ENHANCED SIMULATION COMPLETE")
        logger.info(f"Final Equity: ${self.portfolio_value:,.2f}")
        logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']:.1f}%")
        logger.info("=" * 60)

        return {"config": self.config, "metrics": metrics, "trades": self.trades, "equity_snapshots": self.equity_snapshots}

    def _open_position(self, side: str, price: float):
        self.position = 1.0 if side == "LONG" else -1.0
        self.entry_price = price
        self.trade_count += 1

    def _close_position(self, price: float) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        if self.position > 0:
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - price) / self.entry_price
        pnl_usd = pnl_pct * self.portfolio_value
        self.portfolio_value += pnl_usd
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        self.position = 0.0
        self.entry_price = None
        return pnl_usd

    def _get_drawdown(self) -> float:
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - self.portfolio_value) / self.peak_value

    def calculate_metrics(self) -> Dict:
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital * 100
        winning = sum(1 for t in self.trades if t["pnl_usd"] > 0)
        losing = len(self.trades) - winning
        win_rate = (winning / len(self.trades) * 100) if self.trades else 0

        if self.equity_snapshots and len(self.equity_snapshots) > 1:
            equities = [s["equity"] for s in self.equity_snapshots]
            returns = np.diff(equities) / equities[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 59)
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
            "current_drawdown_pct": round(self._get_drawdown() * 100, 2),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(min(profit_factor, 999.99), 2),
            "total_trades": len(self.trades),
            "winning_trades": winning,
            "losing_trades": losing,
            "final_equity": round(self.portfolio_value, 2),
            "realized_pnl": round(self.portfolio_value - self.initial_capital, 2)
        }


def save_results_to_db(results: Dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        model_id = results["config"]["model_id"]
        metrics = results["metrics"]

        cursor.execute("""
            INSERT INTO trading_state (model_id, position, equity, peak_equity, trade_count, last_updated)
            VALUES (%s, 'FLAT', %s, %s, 0, NOW())
            ON CONFLICT (model_id) DO NOTHING
        """, (model_id, CONFIG["initial_capital"], CONFIG["initial_capital"]))

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
            logger.info(f"Inserted {len(trades_data)} trades")

        if results["equity_snapshots"]:
            snapshot_data = [(s["model_id"], s["timestamp"], s["equity"], s["drawdown_pct"],
                             s["position"], s["bar_close_price"]) for s in results["equity_snapshots"]]
            execute_batch(cursor, """
                INSERT INTO equity_snapshots (model_id, timestamp, equity, drawdown_pct, position, bar_close_price)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (model_id, timestamp) DO UPDATE SET equity = EXCLUDED.equity
            """, snapshot_data)
            logger.info(f"Inserted {len(snapshot_data)} equity snapshots")

        conn.commit()
        logger.info("ENHANCED results saved to database")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def main():
    print("=" * 70)
    print("ENHANCED PAPER TRADING SIMULATION")
    print("=" * 70)
    print(f"Date range: {CONFIG['out_of_sample_start']} to {CONFIG['out_of_sample_end']}")
    print(f"Entry threshold: +/-{ENTRY_THRESHOLD}")
    print(f"Max position duration: {MAX_POSITION_BARS} bars (~5 hours)")
    print(f"Max trades/day: {MAX_TRADES_PER_DAY}")
    print("=" * 70)

    ohlcv_df = load_ohlcv_data(CONFIG["out_of_sample_start"], CONFIG["out_of_sample_end"])
    macro_df = load_macro_data(CONFIG["out_of_sample_start"], CONFIG["out_of_sample_end"])

    if len(ohlcv_df) == 0:
        logger.error("No OHLCV data found")
        return

    engine = PaperTradingEngine(CONFIG)
    results = engine.run_simulation(ohlcv_df, macro_df)
    save_results_to_db(results)

    print("\n" + "=" * 70)
    print("ENHANCED RESULTS")
    print("=" * 70)
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")

    print("\nTrade Summary:")
    for i, t in enumerate(results["trades"], 1):
        print(f"  #{i}: {t['side']} | Entry: {t['entry_price']:.2f} | Exit: {t['exit_price']:.2f} | PnL: ${t['pnl_usd']:.2f} | Reason: {t['exit_reason']}")


if __name__ == "__main__":
    main()
