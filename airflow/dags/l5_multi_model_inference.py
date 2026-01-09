"""
DAG: v3.l5_multi_model_inference
================================
Layer 5: Multi-Model Realtime Inference System (PRODUCTION V19)

Purpose:
    Execute inference across multiple RL models with 15-feature observation space
    matching TradingEnvironmentV19 from training.

Architecture:
    1. Model Registry Pattern - Load models from config table
    2. StateTracker (V19) - Track per-model state (position, time_normalized)
    3. ObservationBuilderV19 - 15-dim observation (13 core + 2 state)
    4. RiskManager - Safety layer for trade validation
    5. PaperTrader - Simulated trade execution
    6. ModelMonitor - Drift detection and health monitoring
    7. Multi-destination output - PostgreSQL, Redis Streams, Events table

SSOT:
    - config/feature_config_v19.json for 15-feature observation space
    - config/v19_norm_stats.json for normalization statistics
    - config.models table for enabled models

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-26
Version: 3.0.0 (V19 15-feature support)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import pandas as pd
import numpy as np
import psycopg2
import os
import json
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import pytz
import sys

# DRY: Using shared utilities
from utils.dag_common import get_db_connection, load_feature_config
from utils.trading_calendar import TradingCalendar, get_calendar, is_trading_day
from utils.datetime_handler import UnifiedDatetimeHandler

# =============================================================================
# V19 MODULE IMPORTS
# =============================================================================
# Add src to path for new modular components
sys.path.insert(0, '/opt/airflow')

from src.core.builders.observation_builder_v19 import ObservationBuilderV19
from src.core.state.state_tracker import StateTracker
from src.risk.risk_manager import RiskManager, RiskLimits
from src.trading.paper_trader import PaperTrader
from src.monitoring.model_monitor import ModelMonitor

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = load_feature_config(raise_on_error=False)
DAG_ID = "v3.l5_multi_model_inference"

# Observation space from SSOT (V19: 15 features = 13 core + 2 state)
OBS_DIM_EXPECTED = 15  # V19: 13 core market features + 2 state features
BARS_PER_SESSION = CONFIG.get("environment_config", {}).get("bars_per_session", 60)
MAX_DRAWDOWN_PCT = CONFIG.get("environment_config", {}).get("max_drawdown_pct", 15.0)

# V19 Feature order (for reference)
# 0-12: Core market features (log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14,
#       dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d, rate_spread, usdmxn_change_1d)
# 13-14: State features (position, time_normalized)

# Validation bounds
OBS_CLIP_MIN = -5.0
OBS_CLIP_MAX = 5.0

# Trading thresholds - V20 FIX: Changed from 0.3 to 0.10 to match training environment
# Training used ACTION_THRESHOLD = 0.10, production must match
LONG_THRESHOLD = CONFIG.get("trading", {}).get("signal_thresholds", {}).get("long", 0.10)
SHORT_THRESHOLD = CONFIG.get("trading", {}).get("signal_thresholds", {}).get("short", -0.10)

# Regime detection config
REGIME_CONFIG = CONFIG.get("regime_detection", {})
VOLATILITY_THRESHOLDS = REGIME_CONFIG.get("thresholds", {
    "crisis": 0.90,
    "high_volatility": 0.75,
    "normal": 0.50,
    "low_volatility": 0.0
})

# Cold Start Configuration
COLD_START_CONFIG = CONFIG.get("cold_start", {})
MIN_WARMUP_BARS = COLD_START_CONFIG.get("min_warmup_bars", 50)
MAX_OHLCV_STALENESS_MIN = COLD_START_CONFIG.get("max_ohlcv_staleness_minutes", 10)

# Redis configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))

# Colombian timezone
COT_TZ = pytz.timezone('America/Bogota')


# =============================================================================
# DATA CLASSES & ENUMS
# =============================================================================

class SignalAction(Enum):
    """Trading signal actions"""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    model_id: str
    model_name: str
    model_type: str
    model_path: str
    version: str
    enabled: bool
    is_production: bool
    priority: int
    # V20 FIX: Changed defaults from 0.3 to 0.10 to match training environment
    threshold_long: float = 0.10
    threshold_short: float = -0.10


@dataclass
class InferenceResult:
    """Result of a single model inference"""
    model_id: str
    model_name: str
    model_type: str
    raw_action: float
    signal: SignalAction
    confidence: float
    latency_ms: float
    observation_hash: str
    bar_number: int
    timestamp: datetime
    state_features: List[float] = field(default_factory=list)
    error: Optional[str] = None


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """Singleton registry for loaded models"""
    _instance = None
    _models: Dict[str, Any] = {}
    _configs: Dict[str, ModelConfig] = {}
    _load_times: Dict[str, datetime] = {}
    _cache_ttl_minutes: int = 60

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of enabled models from config table"""
        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                CREATE SCHEMA IF NOT EXISTS config;

                CREATE TABLE IF NOT EXISTS config.models (
                    model_id VARCHAR(100) PRIMARY KEY,
                    model_name VARCHAR(200) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    model_path VARCHAR(500) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    is_production BOOLEAN DEFAULT FALSE,
                    priority INT DEFAULT 100,
                    threshold_long DECIMAL(5,3) DEFAULT 0.10,  -- V20 FIX: Match training threshold
                    threshold_short DECIMAL(5,3) DEFAULT -0.10,  -- V20 FIX: Match training threshold
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );

                INSERT INTO config.models (model_id, model_name, model_type, model_path, version, enabled, is_production, priority)
                SELECT * FROM (VALUES
                    ('ppo_v1', 'PPO USDCOP V1 (Production)', 'PPO', '/opt/airflow/models/ppo_v1_20251226_054154.zip', 'v1.0.0', TRUE, TRUE, 1),
                    ('ppo_v15_legacy', 'PPO USDCOP V15 (Legacy)', 'PPO', '/opt/airflow/models/ppo_usdcop_v15_fold3.zip', 'v15.0.0', FALSE, FALSE, 5)
                ) AS v(model_id, model_name, model_type, model_path, version, enabled, is_production, priority)
                WHERE NOT EXISTS (SELECT 1 FROM config.models LIMIT 1);
            """)
            conn.commit()

            cur.execute("""
                SELECT model_id, model_name, model_type, model_path, version,
                       enabled, is_production, priority, threshold_long, threshold_short
                FROM config.models
                WHERE enabled = TRUE
                ORDER BY priority ASC
            """)

            models = []
            for row in cur.fetchall():
                models.append(ModelConfig(
                    model_id=row[0],
                    model_name=row[1],
                    model_type=row[2],
                    model_path=row[3],
                    version=row[4],
                    enabled=row[5],
                    is_production=row[6],
                    priority=row[7],
                    # V20 FIX: Fallback values match training thresholds
                    threshold_long=float(row[8]) if row[8] else 0.10,
                    threshold_short=float(row[9]) if row[9] else -0.10
                ))

            logging.info(f"Found {len(models)} enabled models in registry")
            return models

        finally:
            cur.close()
            conn.close()

    def load_model(self, config: ModelConfig) -> Optional[Any]:
        """Load a model from path with caching"""
        model_id = config.model_id

        if model_id in self._models:
            load_time = self._load_times.get(model_id)
            if load_time:
                age_minutes = (datetime.now() - load_time).total_seconds() / 60
                if age_minutes < self._cache_ttl_minutes:
                    return self._models[model_id]

        try:
            model_path = config.model_path
            # Try multiple fallback paths for model files
            fallback_paths = [
                config.model_path,  # Primary path from DB
                f"/opt/airflow/ml_models/{os.path.basename(config.model_path)}",  # ml_models directory
                f"/opt/airflow/models/{os.path.basename(config.model_path)}",  # models directory (volume mount)
                os.path.join(os.path.dirname(__file__), f"../../models/{os.path.basename(config.model_path)}")  # relative
            ]

            model_path = None
            for path in fallback_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path is None:
                logging.warning(f"Model file not found in any location: {fallback_paths}")
                return None

            if config.model_type == "PPO":
                from stable_baselines3 import PPO
                model = PPO.load(model_path)
            elif config.model_type == "SAC":
                from stable_baselines3 import SAC
                model = SAC.load(model_path)
            elif config.model_type == "TD3":
                from stable_baselines3 import TD3
                model = TD3.load(model_path)
            else:
                logging.error(f"Unknown model type: {config.model_type}")
                return None

            self._models[model_id] = model
            self._configs[model_id] = config
            self._load_times[model_id] = datetime.now()

            logging.info(f"Loaded model: {model_id} ({config.model_type}) from {model_path}")
            return model

        except Exception as e:
            logging.error(f"Error loading model {model_id}: {e}")
            return None

    def get_model(self, model_id: str) -> Optional[Any]:
        return self._models.get(model_id)


model_registry = ModelRegistry()


# =============================================================================
# V19 GLOBAL INSTANCES
# =============================================================================

# State tracker for managing model positions (V19: 2 state features)
state_tracker = StateTracker(initial_equity=10000.0)

# Observation builder for V19 15-dim observations (13 core + 2 state)
observation_builder = ObservationBuilderV19()

# Risk manager with conservative limits
risk_manager = RiskManager(RiskLimits(
    max_drawdown_pct=15.0,
    max_daily_loss_pct=5.0,
    max_trades_per_day=20
))

# Paper trader for simulated execution
paper_trader = PaperTrader(initial_capital=10000.0)

# Model monitors for drift detection (one per model)
model_monitors: Dict[str, ModelMonitor] = {}


# =============================================================================
# TECHNICAL INDICATOR HELPERS (V19)
# =============================================================================

def _calculate_rsi(cur, period: int = 9) -> float:
    """Calculate RSI from recent OHLCV data"""
    cur.execute(f"""
        SELECT close FROM usdcop_m5_ohlcv
        ORDER BY time DESC LIMIT {period + 1}
    """)
    closes = [r[0] for r in cur.fetchall()][::-1]
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff([float(c) for c in closes])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calculate_atr_pct(cur, period: int = 10) -> float:
    """Calculate ATR percentage"""
    cur.execute(f"""
        SELECT high, low, close FROM usdcop_m5_ohlcv
        ORDER BY time DESC LIMIT {period + 1}
    """)
    rows = cur.fetchall()[::-1]
    if len(rows) < period + 1:
        return 0.05

    tr_list = []
    for i in range(1, len(rows)):
        h, l, c = float(rows[i][0]), float(rows[i][1]), float(rows[i][2])
        prev_c = float(rows[i-1][2])
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr_list.append(tr)

    atr = np.mean(tr_list[-period:])
    return atr / float(rows[-1][2])


def _calculate_adx(cur, period: int = 14) -> float:
    """Calculate ADX (simplified placeholder)"""
    return 25.0  # Placeholder - full ADX requires more complex calculation


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_trading_hours() -> Tuple[bool, str]:
    """Check if current time is within Colombian trading hours"""
    calendar = get_calendar()
    now_cot = datetime.now(COT_TZ)

    if not calendar.is_trading_day(now_cot):
        reason = calendar.get_violation_reason(now_cot)
        return False, f"Non-trading day: {reason}"

    market_open = now_cot.replace(hour=8, minute=0, second=0, microsecond=0)
    market_close = now_cot.replace(hour=12, minute=55, second=0, microsecond=0)

    if now_cot < market_open:
        return False, f"Before market open (08:00 COT)"
    if now_cot > market_close:
        return False, f"After market close (12:55 COT)"

    return True, "Trading hours active"


def get_bar_number() -> int:
    """Calculate current bar number in trading session (1-60)"""
    now_cot = datetime.now(COT_TZ)
    market_start = now_cot.replace(hour=8, minute=0, second=0, microsecond=0)

    if now_cot < market_start:
        return 1

    elapsed = now_cot - market_start
    bar_num = int(elapsed.total_seconds() / 300) + 1
    return max(1, min(bar_num, BARS_PER_SESSION))


def discretize_action(raw_action: float, threshold_long: float, threshold_short: float) -> SignalAction:
    """Convert continuous action to discrete signal"""
    if raw_action > threshold_long:
        return SignalAction.LONG
    elif raw_action < threshold_short:
        return SignalAction.SHORT
    else:
        return SignalAction.HOLD


def generate_observation_hash(observation: np.ndarray) -> str:
    return hashlib.md5(observation.tobytes()).hexdigest()[:12]


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def create_output_tables():
    """Create all required output tables"""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            CREATE SCHEMA IF NOT EXISTS trading;
            CREATE SCHEMA IF NOT EXISTS events;
            CREATE SCHEMA IF NOT EXISTS dw;

            -- Model inferences table (V19: includes state_features)
            CREATE TABLE IF NOT EXISTS trading.model_inferences (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                model_id VARCHAR(100) NOT NULL,
                model_name VARCHAR(200),
                model_type VARCHAR(50),
                bar_number INT,
                raw_action DECIMAL(10,6),
                signal VARCHAR(20),
                confidence DECIMAL(5,4),
                latency_ms DECIMAL(10,3),
                observation_hash VARCHAR(20),
                observation_dim INT DEFAULT 30,
                current_price DECIMAL(12,4),
                state_features JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_model_inferences_timestamp
                ON trading.model_inferences (timestamp_utc DESC);
            CREATE INDEX IF NOT EXISTS idx_model_inferences_model
                ON trading.model_inferences (model_id, timestamp_utc DESC);

            -- Strategy signals table (for frontend API)
            CREATE TABLE IF NOT EXISTS dw.fact_strategy_signals (
                signal_id SERIAL PRIMARY KEY,
                strategy_id INT,
                timestamp_utc TIMESTAMPTZ DEFAULT NOW(),
                signal VARCHAR(20),
                side VARCHAR(10),
                confidence DECIMAL(5,4),
                size DECIMAL(5,4) DEFAULT 1.0,
                entry_price DECIMAL(12,4),
                stop_loss DECIMAL(12,4),
                take_profit DECIMAL(12,4),
                risk_usd DECIMAL(12,4) DEFAULT 0,
                reasoning TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_fact_strategy_signals_time
                ON dw.fact_strategy_signals (timestamp_utc DESC);

            -- Dim strategy table
            CREATE TABLE IF NOT EXISTS dw.dim_strategy (
                strategy_id SERIAL PRIMARY KEY,
                strategy_code VARCHAR(50) UNIQUE NOT NULL,
                strategy_name VARCHAR(200),
                strategy_type VARCHAR(50),
                model_id VARCHAR(100),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Seed strategies
            INSERT INTO dw.dim_strategy (strategy_code, strategy_name, strategy_type, model_id, is_active)
            VALUES
                ('RL_PPO', 'PPO USDCOP V1', 'RL', 'ppo_v1', TRUE),
                ('RL_PPO_LEGACY', 'PPO V15 Legacy', 'RL', 'ppo_v15_legacy', FALSE)
            ON CONFLICT (strategy_code) DO NOTHING;
        """)
        conn.commit()
        logging.info("Output tables created/verified")

    finally:
        cur.close()
        conn.close()


def insert_inference_result(result: InferenceResult, current_price: float):
    """Insert inference result to PostgreSQL"""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO trading.model_inferences
            (timestamp, model_id, model_name, model_type, bar_number,
             raw_action, signal, confidence, latency_ms, observation_hash,
             observation_dim, current_price, state_features)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            result.timestamp,
            result.model_id,
            result.model_name,
            result.model_type,
            result.bar_number,
            result.raw_action,
            result.signal.value,
            result.confidence,
            result.latency_ms,
            result.observation_hash,
            OBS_DIM_EXPECTED,
            current_price,
            json.dumps(result.state_features)
        ))

        # Also insert to fact_strategy_signals for frontend API
        cur.execute("""
            INSERT INTO dw.fact_strategy_signals
            (strategy_id, timestamp_utc, signal, side, confidence, size, entry_price, reasoning)
            SELECT
                ds.strategy_id,
                %s,
                %s,
                CASE WHEN %s = 'LONG' THEN 'buy' WHEN %s = 'SHORT' THEN 'sell' ELSE 'hold' END,
                %s,
                1.0,
                %s,
                %s
            FROM dw.dim_strategy ds
            WHERE ds.model_id = %s
        """, (
            result.timestamp,
            result.signal.value,
            result.signal.value, result.signal.value,
            result.confidence,
            current_price,
            f"Model {result.model_id}: raw_action={result.raw_action:.4f}",
            result.model_id
        ))

        conn.commit()

    finally:
        cur.close()
        conn.close()


def publish_to_redis(result: InferenceResult):
    """Publish inference result to Redis Streams"""
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        stream_key = f"signals:{result.model_id}:stream"
        message = {
            "model_id": result.model_id,
            "signal": result.signal.value,
            "raw_action": str(result.raw_action),
            "confidence": str(result.confidence),
            "latency_ms": str(result.latency_ms),
            "bar_number": str(result.bar_number),
            "timestamp": result.timestamp.isoformat()
        }

        r.xadd(stream_key, message, maxlen=1000)

    except Exception as e:
        logging.warning(f"Error publishing to Redis: {e}")


# =============================================================================
# MAIN DAG TASKS
# =============================================================================

def check_system_readiness(**ctx) -> Dict[str, Any]:
    """Task 1: Check trading hours and system readiness for V19 15-feature inference"""
    logging.info("Checking system readiness for V19 15-feature inference...")

    is_trading, reason = check_trading_hours()
    if not is_trading:
        logging.warning(f"Skipping inference: {reason}")
        return {"status": "SKIP", "reason": reason, "can_infer": False}

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT COUNT(*) as recent_bars,
                   EXTRACT(EPOCH FROM (NOW() - MAX(time))) / 60 as age_minutes
            FROM usdcop_m5_ohlcv
            WHERE time > NOW() - INTERVAL '1 hour'
        """)
        result = cur.fetchone()
        age_minutes = result[1] or 999

        if age_minutes > MAX_OHLCV_STALENESS_MIN:
            return {
                "status": "STALE",
                "reason": f"OHLCV data is {age_minutes:.1f} min old",
                "can_infer": False
            }

        return {
            "status": "READY",
            "reason": f"System ready. Data age: {age_minutes:.1f} min",
            "can_infer": True,
            "observation_dim": OBS_DIM_EXPECTED
        }

    finally:
        cur.close()
        conn.close()


def load_models(**ctx) -> Dict[str, Any]:
    """Task 2: Load enabled models from registry"""
    logging.info("Loading models from registry...")
    create_output_tables()

    model_configs = model_registry.get_enabled_models()
    loaded_models = []
    failed_models = []

    for config in model_configs:
        model = model_registry.load_model(config)
        if model is not None:
            loaded_models.append({
                "model_id": config.model_id,
                "model_name": config.model_name,
                "model_type": config.model_type,
                "is_production": config.is_production
            })
        else:
            failed_models.append(config.model_id)

    ctx["ti"].xcom_push(key="loaded_models", value=loaded_models)
    return {"loaded_count": len(loaded_models), "failed_count": len(failed_models)}


def build_observation(**ctx) -> Dict[str, Any]:
    """Task 3: Build V19 15-dim observation (13 core market + 2 state features)

    V20 FIX - Look-Ahead Bias Prevention:
    - Signal is generated using the PREVIOUS bar (bar N-1, which is fully closed)
    - Execution happens at the CURRENT bar's OPEN (bar N)
    - This prevents using future information (close price) for execution
    """
    logging.info(f"Building V19 15-dim observation (with look-ahead bias prevention)...")

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # V20 FIX: Query TWO latest bars
        # - Bar N-1 (prev_bar): Used for signal generation (CLOSED, no look-ahead)
        # - Bar N (current_bar): Use OPEN price for execution
        cur.execute("""
            WITH latest AS (
                SELECT time, open, close FROM usdcop_m5_ohlcv
                WHERE time > NOW() - INTERVAL '1 hour'
                ORDER BY time DESC LIMIT 50
            )
            SELECT
                -- Returns calculated from bar N-1 (signal bar)
                LN(close / LAG(close,1) OVER (ORDER BY time)) as log_ret_5m,
                LN(close / LAG(close,12) OVER (ORDER BY time)) as log_ret_1h,
                LN(close / LAG(close,48) OVER (ORDER BY time)) as log_ret_4h,
                close as signal_bar_close,
                open as current_bar_open,
                time as bar_time
            FROM latest ORDER BY time DESC LIMIT 2
        """)
        # V20 FIX: Fetch both bars
        ohlcv_rows = cur.fetchall()

        # Bar 0 = current bar (most recent), Bar 1 = previous bar (signal bar)
        current_bar = ohlcv_rows[0] if len(ohlcv_rows) > 0 else None
        signal_bar = ohlcv_rows[1] if len(ohlcv_rows) > 1 else current_bar

        # Get macro features
        cur.execute("""
            SELECT
                (fxrt_index_dxy_usa_d_dxy - 100.21) / 5.60 as dxy_z,
                (fxrt_index_dxy_usa_d_dxy - LAG(fxrt_index_dxy_usa_d_dxy,1) OVER (ORDER BY fecha)) / NULLIF(LAG(fxrt_index_dxy_usa_d_dxy,1) OVER (ORDER BY fecha), 0) as dxy_change_1d,
                (volt_vix_usa_d_vix - 21.16) / 7.89 as vix_z,
                (crsk_spread_embi_col_d_embi - 322.01) / 62.68 as embi_z,
                (comm_oil_brent_glb_d_brent - LAG(comm_oil_brent_glb_d_brent,1) OVER (ORDER BY fecha)) / NULLIF(LAG(comm_oil_brent_glb_d_brent,1) OVER (ORDER BY fecha), 0) as brent_change_1d,
                ((10.0 - finc_bond_yield10y_usa_d_ust10y) - 7.03) / 1.41 as rate_spread,
                (fxrt_spot_usdmxn_mex_d_usdmxn - LAG(fxrt_spot_usdmxn_mex_d_usdmxn,1) OVER (ORDER BY fecha)) / NULLIF(LAG(fxrt_spot_usdmxn_mex_d_usdmxn,1) OVER (ORDER BY fecha), 0) as usdmxn_change_1d
            FROM macro_indicators_daily
            WHERE fecha <= CURRENT_DATE
            ORDER BY fecha DESC LIMIT 1
        """)
        macro_result = cur.fetchone()

        # V20 FIX - Look-ahead bias prevention:
        # - signal_bar_close: Used for observation building (bar N-1, already closed)
        # - execution_price: OPEN of current bar (bar N) for trade execution
        signal_bar_close = float(signal_bar[3]) if signal_bar and signal_bar[3] else 4200.0
        execution_price = float(current_bar[4]) if current_bar and current_bar[4] else signal_bar_close
        signal_bar_time = signal_bar[5] if signal_bar and len(signal_bar) > 5 else None
        current_bar_time = current_bar[5] if current_bar and len(current_bar) > 5 else None

        logging.info(f"V20 Look-Ahead Fix: Signal from bar {signal_bar_time}, Execution at bar {current_bar_time}")
        logging.info(f"Signal bar close: {signal_bar_close:.2f}, Execution price (open): {execution_price:.2f}")

        bar_number = get_bar_number()

        # Build market features dict (13 core features for V19)
        # V20 FIX: Use signal_bar data (previous closed bar) for observation
        market_features = {
            "log_ret_5m": float(signal_bar[0]) if signal_bar and signal_bar[0] else 0.0,
            "log_ret_1h": float(signal_bar[1]) if signal_bar and signal_bar[1] else 0.0,
            "log_ret_4h": float(signal_bar[2]) if signal_bar and signal_bar[2] else 0.0,
            "rsi_9": _calculate_rsi(cur, 9),
            "atr_pct": _calculate_atr_pct(cur, 10),
            "adx_14": _calculate_adx(cur, 14),
            "dxy_z": float(macro_result[0]) if macro_result and macro_result[0] else 0.0,
            "dxy_change_1d": float(macro_result[1]) if macro_result and macro_result[1] else 0.0,
            "vix_z": float(macro_result[2]) if macro_result and macro_result[2] else 0.0,
            "embi_z": float(macro_result[3]) if macro_result and macro_result[3] else 0.0,
            "brent_change_1d": float(macro_result[4]) if macro_result and macro_result[4] else 0.0,
            "rate_spread": float(macro_result[5]) if macro_result and macro_result[5] else 0.0,
            "usdmxn_change_1d": float(macro_result[6]) if macro_result and macro_result[6] else 0.0
        }

        ctx["ti"].xcom_push(key="market_features", value=market_features)
        # V20 FIX: Push BOTH prices - signal bar close for reference, execution price for trades
        ctx["ti"].xcom_push(key="signal_bar_price", value=signal_bar_close)
        ctx["ti"].xcom_push(key="execution_price", value=execution_price)  # Current bar OPEN
        ctx["ti"].xcom_push(key="bar_number", value=bar_number)

        logging.info(f"V19 market features built: {len(market_features)} features")
        logging.info(f"V20 FIX: Signal bar price={signal_bar_close:.2f}, Execution price={execution_price:.2f}")

        return {"status": "success", "bar_number": bar_number, "signal_price": signal_bar_close, "execution_price": execution_price}
    finally:
        cur.close()
        conn.close()


def run_multi_model_inference(**ctx) -> Dict[str, Any]:
    """Task 4: Run V19 inference on all enabled models with 15-dim observation

    V20 FIX - Look-Ahead Bias Prevention:
    - Uses execution_price (current bar OPEN) for trade execution
    - Signal was generated from previous bar's closed data
    """
    ti = ctx["ti"]

    market_features = ti.xcom_pull(task_ids="build_observation", key="market_features")
    # V20 FIX: Use execution_price (current bar OPEN) instead of signal bar close
    execution_price = ti.xcom_pull(task_ids="build_observation", key="execution_price")
    signal_bar_price = ti.xcom_pull(task_ids="build_observation", key="signal_bar_price")
    bar_number = ti.xcom_pull(task_ids="build_observation", key="bar_number")
    loaded_models = ti.xcom_pull(task_ids="load_models", key="loaded_models")

    if not market_features:
        raise ValueError("No market features available")

    logging.info(f"V20 FIX: Executing trades at OPEN price {execution_price:.2f} (signal from bar close {signal_bar_price:.2f})")

    inference_results = []
    now = datetime.now(COT_TZ)

    for model_info in loaded_models:
        model_id = model_info["model_id"]
        model = model_registry.get_model(model_id)
        config = model_registry._configs.get(model_id)

        if model is None:
            continue

        try:
            # Get state features from StateTracker (V19: position, time_normalized)
            position, time_norm = state_tracker.get_state_features(
                model_id, bar_number, total_bars=BARS_PER_SESSION
            )

            # Build 15-dim observation using ObservationBuilderV19
            obs = observation_builder.build(
                market_features=market_features,
                position=position,
                time_normalized=time_norm
            )

            # V20 FIX: Detailed observation logging for debugging
            logging.info(f"[{model_id}] Observation shape: {obs.shape}, dtype: {obs.dtype}")
            logging.info(f"[{model_id}] Observation first 5 values: {obs[:5].tolist()}")
            logging.info(f"[{model_id}] Observation state features: position={position:.2f}, time_norm={time_norm:.3f}")

            # Validate dimension
            if len(obs) != OBS_DIM_EXPECTED:
                logging.error(f"[{model_id}] DIMENSION MISMATCH: {len(obs)} != {OBS_DIM_EXPECTED}")
                raise ValueError(f"Observation dim mismatch: {len(obs)} vs {OBS_DIM_EXPECTED}")

            # Run inference
            start_time = time.time()
            action, _ = model.predict(obs, deterministic=True)
            latency_ms = (time.time() - start_time) * 1000

            raw_action = float(action)
            signal = discretize_action(raw_action, config.threshold_long, config.threshold_short)
            confidence = min(abs(raw_action), 1.0)

            # Get or create model monitor
            if model_id not in model_monitors:
                model_monitors[model_id] = ModelMonitor(window_size=100)
            monitor = model_monitors[model_id]
            monitor.record_action(raw_action)

            # Validate with RiskManager before execution
            state = state_tracker.get_or_create(model_id)
            allowed, reason = risk_manager.validate_signal(
                signal.value, state.current_drawdown * 100
            )

            if allowed:
                # Execute paper trade at OPEN price (V20 look-ahead fix)
                trade = paper_trader.execute_signal(
                    model_id, signal.value, execution_price, now
                )

                # Update state tracker with new position
                new_position = 1.0 if signal == SignalAction.LONG else (
                    -1.0 if signal == SignalAction.SHORT else 0.0
                )
                state_tracker.update_position(model_id, new_position, execution_price)

                # Record PnL if trade closed
                if trade and trade.status == "closed":
                    monitor.record_pnl(trade.pnl_pct)
                    risk_manager.record_trade_result(trade.pnl_pct, signal.value)
            else:
                logging.warning(f"Trade blocked for {model_id}: {reason}")

            # Store result (V19: state_features are [position, time_norm])
            result = InferenceResult(
                model_id=model_id,
                model_name=model_info["model_name"],
                model_type=model_info["model_type"],
                raw_action=raw_action,
                signal=signal,
                confidence=confidence,
                latency_ms=latency_ms,
                observation_hash=generate_observation_hash(obs),
                bar_number=bar_number,
                timestamp=now,
                state_features=[position, time_norm]
            )

            insert_inference_result(result, execution_price)
            publish_to_redis(result)
            inference_results.append(asdict(result))

            logging.info(f"Model {model_id}: {signal.value} (raw={raw_action:.4f}, conf={confidence:.3f}, allowed={allowed})")

        except Exception as e:
            logging.error(f"Error in model {model_id}: {e}")
            inference_results.append({"model_id": model_id, "error": str(e)})

    ctx["ti"].xcom_push(key="inference_results", value=inference_results)
    return {"total_models": len(loaded_models), "successful": len([r for r in inference_results if "error" not in r])}


def validate_and_summarize(**ctx) -> Dict[str, Any]:
    """Task 5: Validate results and create summary"""
    ti = ctx["ti"]
    inference_results = ti.xcom_pull(task_ids="run_multi_model_inference", key="inference_results")

    if not inference_results:
        return {"status": "no_results"}

    signal_counts = {"LONG": 0, "SHORT": 0, "HOLD": 0}
    for result in inference_results:
        if "error" not in result:
            signal = result.get("signal", "HOLD")
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

    return {
        "total_models": len(inference_results),
        "signal_distribution": signal_counts,
        "observation_dim": OBS_DIM_EXPECTED,
        "timestamp": datetime.now(COT_TZ).isoformat()
    }


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    "owner": "trading-team",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "depends_on_past": False,
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description="V3 L5: Multi-model inference with 15-dim V19 observation space (13 core + 2 state)",
    schedule_interval="*/5 13-17 * * 1-5",
    catchup=False,
    max_active_runs=1,
    tags=["v3", "l5", "inference", "multi-model", "v19", "15-features"]
)

with dag:
    check_ready = PythonOperator(
        task_id="check_system_readiness",
        python_callable=check_system_readiness,
        provide_context=True
    )

    load = PythonOperator(
        task_id="load_models",
        python_callable=load_models,
        provide_context=True
    )

    build_obs = PythonOperator(
        task_id="build_observation",
        python_callable=build_observation,
        provide_context=True
    )

    infer = PythonOperator(
        task_id="run_multi_model_inference",
        python_callable=run_multi_model_inference,
        provide_context=True
    )

    validate = PythonOperator(
        task_id="validate_and_summarize",
        python_callable=validate_and_summarize,
        provide_context=True
    )

    check_ready >> load >> build_obs >> infer >> validate
