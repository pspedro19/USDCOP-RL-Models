"""
DAG: v3.l5_multi_model_inference
================================
Layer 5: Multi-Model Realtime Inference System (PRODUCTION)

Purpose:
    Execute inference across multiple RL models with 15-feature observation space
    matching TradingEnvironment from training.

Architecture:
    1. Model Registry Pattern - Load models from config table
    2. StateTracker - Track per-model state (position, time_normalized)
    3. ObservationBuilder - 15-dim observation (13 core + 2 state)
    4. RiskManager - Safety layer for trade validation
    5. PaperTrader - Simulated trade execution
    6. ModelMonitor - Drift detection and health monitoring
    7. Multi-destination output - PostgreSQL, Redis Streams, Events table

Schedule:
    Event-driven: Uses NewFeatureBarSensor to wait for new features from L1.
    Fallback schedule: */5 13-17 * * 1-5 (8:00-12:55 COT)

    Best Practice: Instead of running blindly every 5 minutes, the sensor
    waits for actual new feature data before running inference. This prevents:
    - Schedule drift
    - Overlapping jobs
    - Running inference on stale features

SSOT:
    - config/feature_config.json for 15-feature observation space
    - config/norm_stats.json for normalization statistics
    - config.models table for enabled models

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-26
Version: 5.0.0 (SSOT Integration + Canary/Rollback)
Updated: 2026-01-18

CHANGELOG v5.0.0:
- INTEGRATED: DeploymentManager for canary deployment and automatic rollback
- INTEGRATED: Traffic splitting for champion/challenger model selection
- INTEGRATED: Post-inference metrics reporting for promotion decisions
- INTEGRATED: Rollback trigger checking with cooldown management

CHANGELOG v4.0.0:
- INTEGRATED: CanonicalFeatureBuilder for RSI, ATR, ADX calculations
- ENSURES: Perfect parity with training (Wilder's EMA for all indicators)
- FALLBACK: Legacy calculators used only if SSOT unavailable
- REMOVES: ADX placeholder - now uses proper Wilder's smoothing
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.exceptions import AirflowSkipException
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

# Event-driven sensor (Best Practice: wait for actual data instead of fixed schedule)
from sensors.new_bar_sensor import NewFeatureBarSensor

# =============================================================================
# MODULE IMPORTS
# =============================================================================
# Add src to path for new modular components
sys.path.insert(0, '/opt/airflow')

# =============================================================================
# VAULT INTEGRATION (P1-4 remediation)
# =============================================================================
try:
    from src.shared.secrets.vault_client import get_vault_client, VaultSecretNotFoundError
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    get_vault_client = None
    VaultSecretNotFoundError = Exception


def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration from Vault with env fallback."""
    if VAULT_AVAILABLE:
        try:
            vault = get_vault_client()
            return vault.get_redis_config()
        except Exception as e:
            logging.debug(f"Vault Redis config failed: {e}")

    # Fallback to environment
    return {
        "host": os.environ.get("REDIS_HOST", "redis"),
        "port": int(os.environ.get("REDIS_PORT", "6379")),
        "password": os.environ.get("REDIS_PASSWORD"),
    }

from src.core.builders.observation_builder import ObservationBuilder
from src.core.state.state_tracker import StateTracker
from src.risk.risk_manager import RiskManager, RiskLimits
from src.trading.paper_trader import PaperTrader
from src.monitoring.model_monitor import ModelMonitor

# =============================================================================
# DEPLOYMENT MANAGER - Canary/Rollback Integration (v5.0.0)
# =============================================================================
try:
    from src.inference.deployment_manager import (
        DeploymentManager,
        ModelDeployment,
        DeploymentStage,
        get_active_model,
        check_rollback_needed,
    )
    DEPLOYMENT_MANAGER_AVAILABLE = True
    logging.info("[DEPLOY] DeploymentManager loaded for canary/rollback support")
except ImportError as e:
    DEPLOYMENT_MANAGER_AVAILABLE = False
    logging.warning(f"[DEPLOY] DeploymentManager not available: {e}. Using legacy model selection.")

    # Fallback stubs
    def get_active_model(*args, **kwargs):
        return None
    def check_rollback_needed(*args, **kwargs):
        return False, None

# =============================================================================
# SSOT IMPORTS - Feature Contract (CTR-FEATURE-001) - REQUIRED, NO FALLBACK
# =============================================================================
# CRITICAL: Feature order and observation dim must come from SSOT contract.
# This ensures inference uses the exact same features as training.
from src.core.contracts.feature_contract import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT_VERSION,
    FEATURE_ORDER_HASH,
    FEATURE_CONTRACT,
)
FEATURE_CONTRACT_AVAILABLE = True
logging.info(f"[CTR-FEATURE-001] Feature contract loaded: {OBSERVATION_DIM} features, hash={FEATURE_ORDER_HASH[:12]}...")

# Import CLIP values from SSOT constants
from src.core.constants import CLIP_MIN, CLIP_MAX

# =============================================================================
# CONTRACT VALIDATION - GAP 3, 10: L1→L3→L5 Validation
# =============================================================================
try:
    from src.core.services import (
        ContractValidator,
        ValidationResult,
        ValidationSeverity,
        PipelineStage,
        create_contract_validator,
    )
    CONTRACT_VALIDATOR_AVAILABLE = True
    logging.info("[CTR-VALIDATE] ContractValidator loaded for L5 validation")
except ImportError as e:
    CONTRACT_VALIDATOR_AVAILABLE = False
    logging.warning(f"[CTR-VALIDATE] ContractValidator not available: {e}")

# Trading Flags - Environment-controlled trading controls (SSOT: src.config.trading_flags)
from src.config.trading_flags import (
    TradingFlags,
    get_trading_flags,
    reload_trading_flags,
    TradingMode,
)

# Backward compatibility - map old function names
load_trading_flags = lambda force_reload=False: reload_trading_flags() if force_reload else get_trading_flags()

class TradingFlagsError(Exception):
    """Raised when there is an error with trading flags configuration."""
    pass

# =============================================================================
# SSOT IMPORTS - CanonicalFeatureBuilder for perfect training/inference parity
# =============================================================================
try:
    from src.feature_store.builders import CanonicalFeatureBuilder, BuilderContext
    CANONICAL_BUILDER_AVAILABLE = True
    logging.info("CanonicalFeatureBuilder (SSOT) loaded successfully")
except ImportError as e:
    CANONICAL_BUILDER_AVAILABLE = False
    logging.warning(f"CanonicalFeatureBuilder not available: {e}. Using legacy calculators.")

# Lazy-initialized SSOT builder
_canonical_builder = None

def get_canonical_builder():
    """Get or initialize the SSOT CanonicalFeatureBuilder for inference."""
    global _canonical_builder
    if _canonical_builder is None and CANONICAL_BUILDER_AVAILABLE:
        try:
            _canonical_builder = CanonicalFeatureBuilder.for_inference()
            logging.info(
                f"Initialized CanonicalFeatureBuilder for inference "
                f"(hash: {_canonical_builder.get_norm_stats_hash()[:12]}...)"
            )
        except Exception as e:
            logging.error(f"Failed to initialize CanonicalFeatureBuilder: {e}")
    return _canonical_builder

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = load_feature_config(raise_on_error=False)
from contracts.dag_registry import L5_PRODUCTION_INFERENCE, L0_MACRO_DAILY
DAG_ID = L5_PRODUCTION_INFERENCE

# =============================================================================
# OBSERVATION SPACE - CTR-FEATURE-001
# =============================================================================
# Observation dimension from SSOT contract (15 features = 13 core + 2 state)
# DO NOT hardcode - use OBSERVATION_DIM from contract import above
OBS_DIM_EXPECTED = OBSERVATION_DIM  # From SSOT contract (CTR-FEATURE-001)
BARS_PER_SESSION = CONFIG.get("environment_config", {}).get("bars_per_session", 60)
MAX_DRAWDOWN_PCT = CONFIG.get("environment_config", {}).get("max_drawdown_pct", 15.0)

# Feature order from SSOT contract - DO NOT hardcode
# CTR-FEATURE-001: FEATURE_ORDER defines the canonical order:
# 0-12: Core market features (log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14,
#       dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d, rate_spread, usdmxn_change_1d)
# 13-14: State features (position, time_normalized)

# Validation bounds from SSOT (src/core/constants.py)
OBS_CLIP_MIN = CLIP_MIN
OBS_CLIP_MAX = CLIP_MAX

# Trading thresholds - SSOT: config/trading_config.yaml (thresholds.long/short = 0.33/-0.33)
# Fallback to config/feature_config.json if available
LONG_THRESHOLD = CONFIG.get("trading", {}).get("signal_thresholds", {}).get("long", 0.33)
SHORT_THRESHOLD = CONFIG.get("trading", {}).get("signal_thresholds", {}).get("short", -0.33)

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

# Redis configuration - P1-4: Use Vault with env fallback
_redis_config = get_redis_config()
REDIS_HOST = _redis_config["host"]
REDIS_PORT = _redis_config["port"]
REDIS_PASSWORD = _redis_config.get("password")

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
    # Default thresholds - from config/trading_config.yaml SSOT
    threshold_long: float = 0.33  # From SSOT: thresholds.long
    threshold_short: float = -0.33  # From SSOT: thresholds.short
    # Traceability fields for contract validation (GAP 3, GAP 10)
    feature_service_name: str = "inference_features_5m"
    norm_stats_hash: Optional[str] = None
    # GAP 3: Hash of FEATURE_ORDER used during training (for L5 validation)
    feature_order_hash: Optional[str] = None


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
                    threshold_long DECIMAL(5,3) DEFAULT 0.33,  -- From SSOT: thresholds.long
                    threshold_short DECIMAL(5,3) DEFAULT -0.33,  -- From SSOT: thresholds.short
                    -- GAP 3: Traceability hashes for L5 contract validation
                    feature_order_hash VARCHAR(64),  -- Hash of FEATURE_ORDER from training
                    norm_stats_hash VARCHAR(64),     -- Hash of norm_stats.json from training
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );

                -- Add columns if they don't exist (migration for existing tables)
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                   WHERE table_schema = 'config' AND table_name = 'models'
                                   AND column_name = 'feature_order_hash') THEN
                        ALTER TABLE config.models ADD COLUMN feature_order_hash VARCHAR(64);
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                   WHERE table_schema = 'config' AND table_name = 'models'
                                   AND column_name = 'norm_stats_hash') THEN
                        ALTER TABLE config.models ADD COLUMN norm_stats_hash VARCHAR(64);
                    END IF;
                END $$;

                INSERT INTO config.models (model_id, model_name, model_type, model_path, version, enabled, is_production, priority)
                SELECT * FROM (VALUES
                    ('ppo_primary', 'PPO USDCOP Primary (Production)', 'PPO', '/opt/airflow/models/ppo_primary.zip', 'current', TRUE, TRUE, 1),
                    ('ppo_secondary', 'PPO USDCOP Secondary', 'PPO', '/opt/airflow/models/ppo_secondary.zip', 'current', FALSE, FALSE, 5)
                ) AS v(model_id, model_name, model_type, model_path, version, enabled, is_production, priority)
                WHERE NOT EXISTS (SELECT 1 FROM config.models LIMIT 1);
            """)
            conn.commit()

            cur.execute("""
                SELECT model_id, model_name, model_type, model_path, version,
                       enabled, is_production, priority, threshold_long, threshold_short,
                       feature_order_hash, norm_stats_hash
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
                    # Fallback values from config/trading_config.yaml SSOT
                    threshold_long=float(row[8]) if row[8] else 0.33,  # SSOT: thresholds.long
                    threshold_short=float(row[9]) if row[9] else -0.33,  # SSOT: thresholds.short
                    # GAP 3: Traceability hashes for L5 contract validation
                    feature_order_hash=row[10] if len(row) > 10 and row[10] else None,
                    norm_stats_hash=row[11] if len(row) > 11 and row[11] else None,
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
# GLOBAL INSTANCES
# =============================================================================

# State tracker for managing model positions (2 state features)
state_tracker = StateTracker(initial_equity=10000.0)

# Observation builder for 15-dim observations (13 core + 2 state)
observation_builder = ObservationBuilder()

# Risk manager with conservative limits (matches trading_config.yaml)
risk_manager = RiskManager(RiskLimits(
    max_drawdown_pct=15.0,
    max_daily_loss_pct=5.0,
    max_trades_per_day=20,
    cooldown_after_losses=5,   # Circuit breaker: 5 consecutive losses
    cooldown_minutes=60        # Cooldown: 12 bars × 5 min = 60 min
))

# Paper trader for simulated execution
paper_trader = PaperTrader(initial_capital=10000.0)

# Model monitors for drift detection (one per model)
model_monitors: Dict[str, ModelMonitor] = {}

# Deployment manager for canary/rollback (v5.0.0)
_deployment_manager: Optional[DeploymentManager] = None


def get_deployment_manager() -> Optional[DeploymentManager]:
    """Get or initialize the DeploymentManager singleton."""
    global _deployment_manager
    if _deployment_manager is None and DEPLOYMENT_MANAGER_AVAILABLE:
        try:
            _deployment_manager = DeploymentManager()
            logging.info("[DEPLOY] Initialized DeploymentManager for canary/rollback support")
        except Exception as e:
            logging.error(f"[DEPLOY] Failed to initialize DeploymentManager: {e}")
    return _deployment_manager


def select_model_for_inference(trade_id: Optional[str] = None) -> Tuple[str, bool]:
    """
    Select which model to use for inference based on canary deployment state.

    Returns:
        Tuple of (model_id, is_challenger) where:
        - model_id: The model to use for this inference
        - is_challenger: True if using challenger (canary), False if using champion
    """
    dm = get_deployment_manager()
    if dm is None:
        # Fallback to production model
        return "ppo_primary", False

    try:
        # Check if we should use challenger (canary traffic splitting)
        use_challenger = dm.should_use_challenger(trade_id)

        if use_challenger and dm.challenger:
            logging.debug(f"[CANARY] Using challenger model: {dm.challenger.model_id}")
            return dm.challenger.model_id, True
        elif dm.champion:
            logging.debug(f"[CANARY] Using champion model: {dm.champion.model_id}")
            return dm.champion.model_id, False
        else:
            # No models deployed - fallback
            return "ppo_primary", False

    except Exception as e:
        logging.error(f"[CANARY] Error selecting model: {e}. Using fallback.")
        return "ppo_primary", False


def report_inference_metrics(
    model_id: str,
    is_challenger: bool,
    pnl_pct: float,
    latency_ms: float,
    signal: str,
):
    """
    Report inference metrics to DeploymentManager for canary evaluation.

    This enables automatic promotion/rollback based on live performance.
    """
    dm = get_deployment_manager()
    if dm is None:
        return

    try:
        # Aggregate metrics for promotion/rollback decisions
        # These will be evaluated by L6 monitoring DAG
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO trading.canary_metrics
            (model_id, is_challenger, pnl_pct, latency_ms, signal, timestamp)
            VALUES (%s, %s, %s, %s, %s, NOW())
        """, (model_id, is_challenger, pnl_pct, latency_ms, signal))
        conn.commit()
        cur.close()
        conn.close()

        logging.debug(f"[CANARY] Reported metrics for {model_id}: pnl={pnl_pct:.4f}")

    except Exception as e:
        logging.warning(f"[CANARY] Failed to report metrics: {e}")


# =============================================================================
# TECHNICAL INDICATOR HELPERS (SSOT-AWARE)
# =============================================================================
# NOTE: These functions now prefer CanonicalFeatureBuilder (SSOT) when available.
# The legacy implementations are kept as fallback only.
# =============================================================================

def _calculate_technical_features_ssot(ohlcv_df: pd.DataFrame, bar_idx: int) -> Dict[str, float]:
    """
    Calculate technical features using SSOT CanonicalFeatureBuilder.

    This ensures PERFECT PARITY with training by using the exact same
    calculators (Wilder's EMA for RSI, ATR, ADX).

    Contract: CTR-FEATURE-001 - Uses FEATURE_ORDER for consistent feature names.

    Args:
        ohlcv_df: DataFrame with OHLCV data
        bar_idx: Current bar index

    Returns:
        Dict with technical features (rsi_9, atr_pct, adx_14, log returns)
    """
    builder = get_canonical_builder()
    if builder is None:
        raise RuntimeError("CanonicalFeatureBuilder not available")

    # Use SSOT calculators - CTR-FEATURE-001
    from src.feature_store.core import (
        RSICalculator, ATRPercentCalculator, ADXCalculator,
        LogReturnCalculator,
    )
    # Use FEATURE_CONTRACT from module-level SSOT import (not feature_store.core)
    # to ensure consistent feature periods

    # Default periods matching SSOT contract (rsi_9, atr_10, adx_14)
    periods = {"rsi": 9, "atr": 10, "adx": 14}

    # Initialize calculators
    rsi_calc = RSICalculator(period=periods["rsi"])
    atr_calc = ATRPercentCalculator(period=periods["atr"])
    adx_calc = ADXCalculator(period=periods["adx"])
    log_ret_5m = LogReturnCalculator("log_ret_5m", periods=1)
    log_ret_1h = LogReturnCalculator("log_ret_1h", periods=12)
    log_ret_4h = LogReturnCalculator("log_ret_4h", periods=48)

    return {
        "rsi_9": rsi_calc.calculate(ohlcv_df, bar_idx),
        "atr_pct": atr_calc.calculate(ohlcv_df, bar_idx),
        "adx_14": adx_calc.calculate(ohlcv_df, bar_idx),
        "log_ret_5m": log_ret_5m.calculate(ohlcv_df, bar_idx),
        "log_ret_1h": log_ret_1h.calculate(ohlcv_df, bar_idx),
        "log_ret_4h": log_ret_4h.calculate(ohlcv_df, bar_idx),
    }


def _load_ohlcv_for_ssot(cur, lookback: int = 100) -> pd.DataFrame:
    """Load OHLCV data from database into DataFrame for SSOT calculators."""
    cur.execute(f"""
        SELECT time, open, high, low, close, volume
        FROM usdcop_m5_ohlcv
        ORDER BY time DESC LIMIT {lookback}
    """)
    rows = cur.fetchall()[::-1]  # Reverse to chronological order

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])


def _calculate_rsi(cur, period: int = 9) -> float:
    """
    Calculate RSI from recent OHLCV data.

    ARCHITECTURE: Prefers SSOT CanonicalFeatureBuilder when available.
    Falls back to legacy SMA-based calculation if SSOT unavailable.

    WARNING: Legacy calculation uses SMA instead of Wilder's EMA.
    """
    # Try SSOT first
    if CANONICAL_BUILDER_AVAILABLE:
        try:
            ohlcv_df = _load_ohlcv_for_ssot(cur, lookback=period * 3)
            if len(ohlcv_df) >= period + 1:
                from src.feature_store.core import RSICalculator
                calc = RSICalculator(period=period)
                return calc.calculate(ohlcv_df, len(ohlcv_df) - 1)
        except Exception as e:
            logging.warning(f"SSOT RSI calculation failed: {e}. Using legacy.")

    # Legacy fallback (SMA-based - may differ from training!)
    logging.debug("Using legacy RSI calculation (SMA-based)")
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
    """
    Calculate ATR percentage.

    ARCHITECTURE: Prefers SSOT CanonicalFeatureBuilder when available.
    Falls back to legacy SMA-based calculation if SSOT unavailable.

    WARNING: Legacy calculation uses SMA instead of Wilder's EMA.
    """
    # Try SSOT first
    if CANONICAL_BUILDER_AVAILABLE:
        try:
            ohlcv_df = _load_ohlcv_for_ssot(cur, lookback=period * 3)
            if len(ohlcv_df) >= period + 1:
                from src.feature_store.core import ATRPercentCalculator
                calc = ATRPercentCalculator(period=period)
                return calc.calculate(ohlcv_df, len(ohlcv_df) - 1)
        except Exception as e:
            logging.warning(f"SSOT ATR calculation failed: {e}. Using legacy.")

    # Legacy fallback (SMA-based - may differ from training!)
    logging.debug("Using legacy ATR calculation (SMA-based)")
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


def _calculate_adx(cur, bar_idx: int = 0, period: int = 14) -> float:
    """
    Calculate ADX using Wilder's smoothing method.

    ARCHITECTURE: Prefers SSOT CanonicalFeatureBuilder when available.
    Falls back to inline Wilder's implementation if SSOT unavailable.

    NOTE: Both SSOT and legacy use Wilder's smoothing (alpha=1/period).
    """
    # Try SSOT first
    if CANONICAL_BUILDER_AVAILABLE:
        try:
            ohlcv_df = _load_ohlcv_for_ssot(cur, lookback=period * 3 + 1)
            if len(ohlcv_df) >= period * 2:
                from src.feature_store.core import ADXCalculator
                calc = ADXCalculator(period=period)
                return calc.calculate(ohlcv_df, len(ohlcv_df) - 1)
        except Exception as e:
            logging.warning(f"SSOT ADX calculation failed: {e}. Using legacy.")

    # Legacy fallback (Wilder's EMA - matches training)
    logging.debug("Using legacy ADX calculation (Wilder's EMA)")
    required_bars = period * 3 + 1

    cur.execute("""
        SELECT high, low, close FROM usdcop_m5_ohlcv
        ORDER BY time DESC LIMIT %s
    """, (required_bars,))

    rows = cur.fetchall()[::-1]  # Reverse to chronological
    if len(rows) < required_bars:
        return 25.0  # Neutral during warmup

    # Calculate TR, +DM, -DM
    tr_list, plus_dm_list, minus_dm_list = [], [], []
    for i in range(1, len(rows)):
        high, low, close = rows[i]
        prev_high, prev_low, prev_close = rows[i-1]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

        up_move, down_move = high - prev_high, prev_low - low
        plus_dm_list.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm_list.append(down_move if down_move > up_move and down_move > 0 else 0)

    # Wilder's smoothing
    alpha = 1.0 / period
    def wilder_smooth(values):
        result = [sum(values[:period]) / period]
        for v in values[period:]:
            result.append(result[-1] * (1 - alpha) + v * alpha)
        return result

    smoothed_tr = wilder_smooth(tr_list)
    smoothed_plus_dm = wilder_smooth(plus_dm_list)
    smoothed_minus_dm = wilder_smooth(minus_dm_list)

    # Calculate DX and ADX
    dx_list = []
    for i in range(len(smoothed_tr)):
        if smoothed_tr[i] == 0:
            dx_list.append(0)
            continue
        plus_di = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
        minus_di = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) else 0
        dx_list.append(dx)

    adx_values = wilder_smooth(dx_list) if len(dx_list) >= period else [25.0]
    return float(adx_values[-1])


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


def compute_norm_stats_hash(norm_stats_path: str = "/opt/airflow/config/norm_stats.json") -> Optional[str]:
    """Compute SHA256 hash of norm_stats.json for traceability.

    Args:
        norm_stats_path: Path to norm_stats.json file

    Returns:
        SHA256 hash string (first 16 chars) or None if file not found
    """
    try:
        with open(norm_stats_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()[:16]
    except FileNotFoundError:
        logging.warning(f"norm_stats.json not found at {norm_stats_path}")
        return None
    except Exception as e:
        logging.warning(f"Error computing norm_stats hash: {e}")
        return None


def validate_norm_stats_hash(expected_hash: Optional[str],
                             norm_stats_path: str = "/opt/airflow/config/norm_stats.json") -> Tuple[bool, str]:
    """Validate that norm_stats.json matches expected hash.

    Args:
        expected_hash: Expected hash from model config (None to skip validation)
        norm_stats_path: Path to norm_stats.json file

    Returns:
        Tuple of (is_valid, message)
    """
    if expected_hash is None:
        return True, "No expected hash configured (validation skipped)"

    current_hash = compute_norm_stats_hash(norm_stats_path)

    if current_hash is None:
        return False, f"Could not compute norm_stats hash (file missing or error)"

    if current_hash != expected_hash:
        logging.warning(
            f"NORM_STATS HASH MISMATCH: expected={expected_hash}, current={current_hash}. "
            f"This may indicate norm_stats.json was modified since model training."
        )
        return False, f"Hash mismatch: expected={expected_hash}, current={current_hash}"

    return True, f"norm_stats hash validated: {current_hash}"


def validate_l5_contract(
    observation: np.ndarray,
    model_feature_order_hash: Optional[str] = None,
    norm_stats_hash: Optional[str] = None,
    strict_mode: bool = False
) -> Tuple[bool, Optional[ValidationResult]]:
    """
    Validate L5 inference contract before running model prediction.

    GAP 3: Validates feature_order_hash matches training contract.
    GAP 10: Validates L1→L3→L5 pipeline transition contracts.

    Args:
        observation: The observation vector to validate
        model_feature_order_hash: Hash from model training (from model registry or metadata)
        norm_stats_hash: Hash of norm_stats.json from model training
        strict_mode: If True, block inference on validation failures

    Returns:
        Tuple of (is_valid, validation_result)

    Note:
        In strict_mode, validation failures will prevent inference.
        In non-strict mode (default), validation failures are logged as warnings.
    """
    if not CONTRACT_VALIDATOR_AVAILABLE:
        logging.debug("[CTR-VALIDATE] ContractValidator not available, skipping validation")
        return True, None

    try:
        validator = create_contract_validator()

        result = validator.validate_l5_inference(
            model_feature_order_hash=model_feature_order_hash,
            observation=observation,
            norm_stats_hash=norm_stats_hash,
        )

        # Log validation results
        if result.is_valid:
            logging.info(
                f"[CTR-VALIDATE] L5 contract validation PASSED: "
                f"{len(result.passed_checks)} checks passed"
            )
            for check in result.passed_checks:
                logging.debug(f"  ✓ {check}")
        else:
            error_count = len([e for e in result.errors if e.severity == ValidationSeverity.ERROR])
            warning_count = len([e for e in result.errors if e.severity == ValidationSeverity.WARNING])

            logging.warning(
                f"[CTR-VALIDATE] L5 contract validation FAILED: "
                f"{error_count} errors, {warning_count} warnings"
            )
            for error in result.errors:
                if error.severity == ValidationSeverity.ERROR:
                    logging.error(f"  ✗ [{error.code}] {error.message}")
                else:
                    logging.warning(f"  ⚠ [{error.code}] {error.message}")

            # In strict mode, validation failures block inference
            if strict_mode and error_count > 0:
                return False, result

        return True, result

    except Exception as e:
        logging.error(f"[CTR-VALIDATE] Contract validation error: {e}")
        # On error, allow inference to continue (fail-open) unless strict mode
        return not strict_mode, None


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

            -- Model inferences table (includes state_features)
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
                observation_dim INT DEFAULT 15,
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
                ('RL_PPO', 'PPO USDCOP Primary', 'RL', 'ppo_primary', TRUE),
                ('RL_PPO_SECONDARY', 'PPO USDCOP Secondary', 'RL', 'ppo_secondary', FALSE)
            ON CONFLICT (strategy_code) DO NOTHING;

            -- Canary metrics table (v5.0.0)
            CREATE TABLE IF NOT EXISTS trading.canary_metrics (
                id SERIAL PRIMARY KEY,
                model_id VARCHAR(100) NOT NULL,
                is_challenger BOOLEAN DEFAULT FALSE,
                pnl_pct DECIMAL(10,6),
                latency_ms DECIMAL(10,3),
                signal VARCHAR(20),
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_canary_metrics_model_time
                ON trading.canary_metrics (model_id, timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_canary_metrics_challenger
                ON trading.canary_metrics (is_challenger, timestamp DESC);
        """)
        conn.commit()
        logging.info("Output tables created/verified")

    finally:
        cur.close()
        conn.close()


def insert_inference_result(result: InferenceResult, current_price: float,
                           model_version: Optional[str] = None,
                           norm_stats_hash: Optional[str] = None):
    """Insert inference result to PostgreSQL with traceability columns.

    Args:
        result: InferenceResult from model inference
        current_price: Current execution price
        model_version: Model version string for traceability
        norm_stats_hash: Hash of norm_stats.json used for this inference

    Note: model_version and norm_stats_hash columns may need migration:
        ALTER TABLE trading.model_inferences ADD COLUMN model_version VARCHAR(50);
        ALTER TABLE trading.model_inferences ADD COLUMN norm_stats_hash VARCHAR(64);
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # NOTE: model_version and norm_stats_hash columns need to be added via migration
        # For now, store in state_features JSON as fallback until migration is applied
        state_features_extended = {
            "position": result.state_features[0] if result.state_features else 0.0,
            "time_normalized": result.state_features[1] if len(result.state_features) > 1 else 0.0,
            "model_version": model_version,
            "norm_stats_hash": norm_stats_hash
        }

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
            json.dumps(state_features_extended)
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
        # P1-4: Use Vault-sourced password if available
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True
        )

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

def validate_trading_flags(**ctx) -> Dict[str, Any]:
    """Task 0: Validate trading flags before any inference.

    This is the FIRST task in the DAG. It checks environment-controlled
    trading flags (TRADING_ENABLED, KILL_SWITCH_ACTIVE) and skips the
    entire DAG if trading is not allowed.

    Raises:
        AirflowSkipException: If trading flags indicate trading should not proceed.

    Returns:
        Dict with flag validation status.
    """
    logging.info("Validating trading flags...")

    try:
        # Load trading flags from environment (force reload to get fresh values)
        flags = load_trading_flags(force_reload=True)

        # Use the can_execute_trades() method for DAG-specific validation
        can_trade, reason = flags.can_execute_trades()

        logging.info(f"Trading flags check: TRADING_ENABLED={flags.trading_enabled}, "
                     f"KILL_SWITCH_ACTIVE={flags.kill_switch_active}, "
                     f"PAPER_TRADING={flags.paper_trading}")

        if not can_trade:
            logging.warning(f"Trading flags validation failed: {reason}")
            # Raise AirflowSkipException to skip all downstream tasks
            raise AirflowSkipException(f"Trading flags validation failed: {reason}")

        logging.info(f"Trading flags validation passed: {reason}")

        return {
            "status": "VALIDATED",
            "trading_enabled": flags.trading_enabled,
            "kill_switch_active": flags.kill_switch_active,
            "paper_trading": flags.paper_trading,
            "reason": reason
        }

    except AirflowSkipException:
        # Re-raise skip exceptions
        raise
    except Exception as e:
        logging.error(f"Error validating trading flags: {e}")
        # On error, be conservative and skip trading
        raise AirflowSkipException(f"Trading flags validation error: {e}")


def check_macro_data_readiness() -> Tuple[bool, str, float]:
    """Check if macro data is ready for inference.

    Queries the latest readiness report from l0_macro_unified DAG.
    Returns (is_ready, reason, score).
    """
    try:
        from airflow.models import DagRun, TaskInstance

        # Find latest successful run of macro DAG today
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        macro_runs = DagRun.find(
            dag_id=L0_MACRO_DAILY,
            state='success',
            execution_start_date=today_start,
        )

        if not macro_runs:
            logging.warning("No successful macro pipeline run found today")
            # Allow inference to continue but log warning
            return True, "No macro run today (continuing)", 0.0

        latest_run = sorted(macro_runs, key=lambda r: r.execution_date, reverse=True)[0]

        # Get readiness report from XCom
        ti = latest_run.get_task_instance('generate_readiness_report')
        if ti:
            is_ready = ti.xcom_pull(key='is_ready_for_inference')
            readiness = ti.xcom_pull(key='readiness_report') or {}

            if is_ready is False:
                return False, f"Macro data not ready. Score: {readiness.get('score', '0%')}", readiness.get('readiness_score', 0)

            score = readiness.get('score', '100%')
            return True, f"Macro data ready. Score: {score}", float(readiness.get('readiness_score', 1.0))

        return True, "Readiness task not found (continuing)", 0.0

    except Exception as e:
        logging.error(f"Error checking macro readiness: {e}")
        # Don't block inference on error - log and continue
        return True, f"Readiness check error: {e} (continuing)", 0.0


def check_system_readiness(**ctx) -> Dict[str, Any]:
    """Task 1: Check trading hours, data freshness, and system readiness for 15-feature inference"""
    logging.info("Checking system readiness for 15-feature inference...")

    is_trading, reason = check_trading_hours()
    if not is_trading:
        logging.warning(f"Skipping inference: {reason}")
        return {"status": "SKIP", "reason": reason, "can_infer": False}

    # Check macro data readiness from L0 pipeline
    macro_ready, macro_reason, macro_score = check_macro_data_readiness()
    logging.info(f"Macro data readiness: {macro_reason}")

    if not macro_ready:
        logging.warning(f"Skipping inference: {macro_reason}")
        return {
            "status": "MACRO_NOT_READY",
            "reason": macro_reason,
            "can_infer": False,
            "macro_score": macro_score
        }

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
            "reason": f"System ready. OHLCV age: {age_minutes:.1f} min, Macro: {macro_reason}",
            "can_infer": True,
            "observation_dim": OBS_DIM_EXPECTED,
            "macro_score": macro_score
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
    """Task 3: Build 15-dim observation (13 core market + 2 state features)

    Look-Ahead Bias Prevention:
    - Signal is generated using the PREVIOUS bar (bar N-1, which is fully closed)
    - Execution happens at the CURRENT bar's OPEN (bar N)
    - This prevents using future information (close price) for execution
    """
    logging.info(f"Building 15-dim observation (with look-ahead bias prevention)...")

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # FIX: Query TWO latest bars
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
        # FIX: Fetch both bars
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

        # Look-ahead bias prevention:
        # - signal_bar_close: Used for observation building (bar N-1, already closed)
        # - execution_price: OPEN of current bar (bar N) for trade execution
        signal_bar_close = float(signal_bar[3]) if signal_bar and signal_bar[3] else 4200.0
        execution_price = float(current_bar[4]) if current_bar and current_bar[4] else signal_bar_close
        signal_bar_time = signal_bar[5] if signal_bar and len(signal_bar) > 5 else None
        current_bar_time = current_bar[5] if current_bar and len(current_bar) > 5 else None

        logging.info(f"Look-Ahead Fix: Signal from bar {signal_bar_time}, Execution at bar {current_bar_time}")
        logging.info(f"Signal bar close: {signal_bar_close:.2f}, Execution price (open): {execution_price:.2f}")

        bar_number = get_bar_number()

        # Build market features dict (13 core features)
        # FIX: Use signal_bar data (previous closed bar) for observation
        market_features = {
            "log_ret_5m": float(signal_bar[0]) if signal_bar and signal_bar[0] else 0.0,
            "log_ret_1h": float(signal_bar[1]) if signal_bar and signal_bar[1] else 0.0,
            "log_ret_4h": float(signal_bar[2]) if signal_bar and signal_bar[2] else 0.0,
            "rsi_9": _calculate_rsi(cur, 9),
            "atr_pct": _calculate_atr_pct(cur, 10),
            "adx_14": _calculate_adx(cur, period=14),
            "dxy_z": float(macro_result[0]) if macro_result and macro_result[0] else 0.0,
            "dxy_change_1d": float(macro_result[1]) if macro_result and macro_result[1] else 0.0,
            "vix_z": float(macro_result[2]) if macro_result and macro_result[2] else 0.0,
            "embi_z": float(macro_result[3]) if macro_result and macro_result[3] else 0.0,
            "brent_change_1d": float(macro_result[4]) if macro_result and macro_result[4] else 0.0,
            "rate_spread": float(macro_result[5]) if macro_result and macro_result[5] else 0.0,
            "usdmxn_change_1d": float(macro_result[6]) if macro_result and macro_result[6] else 0.0
        }

        ctx["ti"].xcom_push(key="market_features", value=market_features)
        # FIX: Push BOTH prices - signal bar close for reference, execution price for trades
        ctx["ti"].xcom_push(key="signal_bar_price", value=signal_bar_close)
        ctx["ti"].xcom_push(key="execution_price", value=execution_price)  # Current bar OPEN
        ctx["ti"].xcom_push(key="bar_number", value=bar_number)

        logging.info(f"Market features built: {len(market_features)} features")
        logging.info(f"FIX: Signal bar price={signal_bar_close:.2f}, Execution price={execution_price:.2f}")

        return {"status": "success", "bar_number": bar_number, "signal_price": signal_bar_close, "execution_price": execution_price}
    finally:
        cur.close()
        conn.close()


def run_multi_model_inference(**ctx) -> Dict[str, Any]:
    """Task 4: Run inference on all enabled models with 15-dim observation

    Look-Ahead Bias Prevention:
    - Uses execution_price (current bar OPEN) for trade execution
    - Signal was generated from previous bar's closed data
    """
    ti = ctx["ti"]

    market_features = ti.xcom_pull(task_ids="build_observation", key="market_features")
    # FIX: Use execution_price (current bar OPEN) instead of signal bar close
    execution_price = ti.xcom_pull(task_ids="build_observation", key="execution_price")
    signal_bar_price = ti.xcom_pull(task_ids="build_observation", key="signal_bar_price")
    bar_number = ti.xcom_pull(task_ids="build_observation", key="bar_number")
    loaded_models = ti.xcom_pull(task_ids="load_models", key="loaded_models")

    if not market_features:
        raise ValueError("No market features available")

    logging.info(f"FIX: Executing trades at OPEN price {execution_price:.2f} (signal from bar close {signal_bar_price:.2f})")

    # =========================================================================
    # CONTRACT VALIDATION (CTR-FEATURE-001 + CTR-HASH-001)
    # =========================================================================
    # Log feature contract info for traceability
    logging.info(f"[CTR-FEATURE-001] Feature contract: version={FEATURE_CONTRACT_VERSION if FEATURE_CONTRACT_AVAILABLE else 'fallback'}, "
                 f"obs_dim={OBS_DIM_EXPECTED}, hash={FEATURE_ORDER_HASH[:12] if FEATURE_CONTRACT_AVAILABLE else 'N/A'}...")

    # Compute current norm_stats hash for traceability (CTR-HASH-001)
    current_norm_stats_hash = compute_norm_stats_hash()
    if current_norm_stats_hash:
        logging.info(f"[CTR-HASH-001] norm_stats.json hash: {current_norm_stats_hash}")
    else:
        logging.warning("[CTR-HASH-001] Could not compute norm_stats.json hash - file may be missing")

    inference_results = []
    now = datetime.now(COT_TZ)

    for model_info in loaded_models:
        model_id = model_info["model_id"]
        model = model_registry.get_model(model_id)
        config = model_registry._configs.get(model_id)

        if model is None:
            continue

        try:
            # Validate norm_stats hash if model has expected hash configured
            if config and config.norm_stats_hash:
                is_valid, validation_msg = validate_norm_stats_hash(
                    config.norm_stats_hash
                )
                if not is_valid:
                    logging.warning(
                        f"[{model_id}] NORM_STATS VALIDATION WARNING: {validation_msg}. "
                        f"Inference will continue but results may differ from training."
                    )
                else:
                    logging.debug(f"[{model_id}] {validation_msg}")

            # Get state features from StateTracker (position, time_normalized)
            position, time_norm = state_tracker.get_state_features(
                model_id, bar_number, total_bars=BARS_PER_SESSION
            )

            # Build 15-dim observation using ObservationBuilder
            obs = observation_builder.build(
                market_features=market_features,
                position=position,
                time_normalized=time_norm
            )

            # FIX: Detailed observation logging for debugging
            logging.info(f"[{model_id}] Observation shape: {obs.shape}, dtype: {obs.dtype}")
            logging.info(f"[{model_id}] Observation first 5 values: {obs[:5].tolist()}")
            logging.info(f"[{model_id}] Observation state features: position={position:.2f}, time_norm={time_norm:.3f}")

            # Validate dimension
            if len(obs) != OBS_DIM_EXPECTED:
                logging.error(f"[{model_id}] DIMENSION MISMATCH: {len(obs)} != {OBS_DIM_EXPECTED}")
                raise ValueError(f"Observation dim mismatch: {len(obs)} vs {OBS_DIM_EXPECTED}")

            # =========================================================================
            # GAP 3, 10: L5 CONTRACT VALIDATION
            # =========================================================================
            # Validate that this model was trained with compatible feature order.
            # This prevents inference using a model trained with different features.
            contract_valid, validation_result = validate_l5_contract(
                observation=obs,
                model_feature_order_hash=config.feature_order_hash if config else None,
                norm_stats_hash=config.norm_stats_hash if config else None,
                strict_mode=False,  # Set to True in production to block on mismatch
            )

            if not contract_valid:
                logging.error(
                    f"[{model_id}] CONTRACT VALIDATION FAILED - Skipping inference. "
                    f"Model may have been trained with incompatible feature order."
                )
                inference_results.append({
                    "model_id": model_id,
                    "error": "Contract validation failed - feature order mismatch"
                })
                continue

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
                # Execute paper trade at OPEN price (look-ahead fix)
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

            # Store result (state_features are [position, time_norm])
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

            insert_inference_result(
                result,
                execution_price,
                model_version=config.version if config else None,
                norm_stats_hash=current_norm_stats_hash
            )
            publish_to_redis(result)
            inference_results.append(asdict(result))

            # v5.0.0: Report metrics to canary system for rollback/promotion evaluation
            # Determine if this model is the challenger
            dm = get_deployment_manager()
            is_challenger = (dm and dm.challenger and dm.challenger.model_id == model_id)

            # Calculate PnL for this trade (if closed)
            trade_pnl = 0.0
            if allowed:
                state = state_tracker.get_or_create(model_id)
                trade_pnl = state.last_trade_pnl if hasattr(state, 'last_trade_pnl') else 0.0

            report_inference_metrics(
                model_id=model_id,
                is_challenger=is_challenger,
                pnl_pct=trade_pnl,
                latency_ms=latency_ms,
                signal=signal.value,
            )

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


def check_rollback_and_canary(**ctx) -> Dict[str, Any]:
    """Task 6: Check rollback triggers and canary promotion status (v5.0.0).

    This task runs after inference to:
    1. Check if any rollback triggers have been activated
    2. Check if canary model is ready for promotion
    3. Report deployment status

    Note: Actual rollback/promotion actions are handled by L6 monitoring DAG.
    This task only collects and reports status.
    """
    logging.info("[DEPLOY] Checking rollback triggers and canary status...")

    dm = get_deployment_manager()
    if dm is None:
        logging.info("[DEPLOY] DeploymentManager not available, skipping check")
        return {"status": "skipped", "reason": "DeploymentManager not available"}

    result = {
        "status": "checked",
        "champion": None,
        "challenger": None,
        "rollback_needed": False,
        "rollback_trigger": None,
        "promotion_ready": False,
    }

    try:
        # Get current deployment state
        if dm.champion:
            result["champion"] = {
                "model_id": dm.champion.model_id,
                "stage": dm.champion.stage.value,
                "deployed_at": dm.champion.deployed_at.isoformat() if dm.champion.deployed_at else None,
            }

        if dm.challenger:
            result["challenger"] = {
                "model_id": dm.challenger.model_id,
                "stage": dm.challenger.stage.value,
                "deployed_at": dm.challenger.deployed_at.isoformat() if dm.challenger.deployed_at else None,
            }

        # Compute recent metrics for rollback check
        conn = get_db_connection()
        cur = conn.cursor()

        # Get rolling metrics for production model
        cur.execute("""
            WITH recent_trades AS (
                SELECT pnl_pct
                FROM trading.canary_metrics
                WHERE model_id = %s
                  AND timestamp > NOW() - INTERVAL '7 days'
                  AND is_challenger = FALSE
            )
            SELECT
                COALESCE(AVG(pnl_pct), 0) as avg_pnl,
                COALESCE(COUNT(*), 0) as trade_count,
                COALESCE(SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0), 0.5) as win_rate
            FROM recent_trades
        """, (dm.champion.model_id if dm.champion else 'ppo_primary',))

        metrics_row = cur.fetchone()
        cur.close()
        conn.close()

        if metrics_row:
            avg_pnl = float(metrics_row[0]) if metrics_row[0] else 0
            trade_count = int(metrics_row[1]) if metrics_row[1] else 0
            win_rate = float(metrics_row[2]) if metrics_row[2] else 0.5

            # Estimate rolling Sharpe (simplified)
            rolling_sharpe = avg_pnl * 15.87  # Annualized approximation

            metrics = {
                "rolling_sharpe_7d": rolling_sharpe,
                "rolling_win_rate_7d": win_rate,
                "trade_count_7d": trade_count,
            }

            result["metrics"] = metrics

            # Check rollback triggers
            rollback_needed, trigger = dm.check_rollback_triggers(metrics)
            result["rollback_needed"] = rollback_needed

            if rollback_needed and trigger:
                result["rollback_trigger"] = trigger.name
                logging.warning(
                    f"[DEPLOY] ROLLBACK TRIGGER ACTIVATED: {trigger.name}. "
                    f"L6 monitoring DAG will handle rollback."
                )

            # Check canary promotion (if challenger exists)
            if dm.challenger:
                # Compute challenger metrics
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()

                    cur.execute("""
                        SELECT
                            AVG(pnl_pct) as avg_pnl,
                            COUNT(*) as trade_count
                        FROM trading.canary_metrics
                        WHERE model_id = %s
                          AND timestamp > NOW() - INTERVAL '3 days'
                          AND is_challenger = TRUE
                    """, (dm.challenger.model_id,))

                    challenger_row = cur.fetchone()
                    cur.close()
                    conn.close()

                    if challenger_row and challenger_row[1] >= 20:  # Minimum trades
                        challenger_sharpe = float(challenger_row[0]) * 15.87 if challenger_row[0] else 0
                        challenger_metrics = {
                            "rolling_sharpe_7d": challenger_sharpe,
                            "trade_count": int(challenger_row[1]),
                        }
                        result["challenger_metrics"] = challenger_metrics

                        # Check if ready for promotion
                        decision = dm.promote(challenger_metrics)
                        result["promotion_ready"] = decision.approved
                        result["promotion_reason"] = decision.reason

                except Exception as e:
                    logging.warning(f"[DEPLOY] Failed to check challenger metrics: {e}")

        logging.info(f"[DEPLOY] Rollback check complete: rollback_needed={result['rollback_needed']}")
        return result

    except Exception as e:
        logging.error(f"[DEPLOY] Error checking rollback/canary: {e}")
        return {"status": "error", "error": str(e)}


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    "owner": "trading-team",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "depends_on_past": False,
    "email_on_failure": True,  # P1 Remediation: Enable failure notifications
    "email": ["trading-alerts@example.com"],  # Configure in Airflow Variables
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description="V3 L5: Event-driven inference with NewFeatureBarSensor (15-dim)",
    schedule_interval="*/5 13-17 * * 1-5",  # Trigger schedule (sensor waits for actual data)
    catchup=False,
    max_active_runs=1,
    tags=["v3", "l5", "inference", "multi-model", "15-features", "sensor-driven"]
)

with dag:
    # ==========================================================================
    # TASK 0: VALIDATE TRADING FLAGS (FIRST TASK - Gate for entire DAG)
    # ==========================================================================
    # This task MUST run first to check if trading is allowed.
    # If TRADING_ENABLED=false or KILL_SWITCH_ACTIVE=true, the entire DAG is skipped.
    validate_flags = PythonOperator(
        task_id="validate_trading_flags",
        python_callable=validate_trading_flags,
        provide_context=True
    )

    # EVENT-DRIVEN SENSOR: Wait for new feature data instead of running blindly
    # This prevents schedule drift and ensures L1 has completed before L5 runs
    wait_features = NewFeatureBarSensor(
        task_id="wait_for_features",
        table_name="inference_features_5m",
        require_complete=True,          # Require all critical features present
        max_staleness_minutes=10,       # Features must be < 10 minutes old
        poke_interval=30,               # Check every 30 seconds
        timeout=300,                    # Max 5 minutes wait (matches schedule)
        mode="poke",                    # Keep worker while waiting
    )

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

    # v5.0.0: Check rollback triggers and canary promotion status
    check_deployment = PythonOperator(
        task_id="check_rollback_and_canary",
        python_callable=check_rollback_and_canary,
        provide_context=True
    )

    def mark_feature_processed(**context):
        """Mark feature time as processed for next sensor check."""
        ti = context['ti']
        detected_time = ti.xcom_pull(key='detected_feature_time', task_ids='wait_for_features')
        if detected_time:
            ti.xcom_push(key='last_processed_feature_time', value=detected_time)
            logging.info(f"Marked feature time as processed: {detected_time}")

    mark_processed = PythonOperator(
        task_id="mark_feature_processed",
        python_callable=mark_feature_processed,
        provide_context=True,
        trigger_rule="all_success"
    )

    # ==========================================================================
    # TASK CHAIN (v5.0.0)
    # ==========================================================================
    # 1. validate_flags: Check TRADING_ENABLED and KILL_SWITCH_ACTIVE first
    # 2. wait_features: Sensor waits for new features from L1
    # 3. check_ready: Verify trading hours and data freshness
    # 4. load: Load models from registry
    # 5. build_obs: Build 15-dim observation
    # 6. infer: Run multi-model inference
    # 7. validate: Validate and summarize results
    # 8. check_deployment: Check rollback triggers and canary promotion (v5.0.0)
    # 9. mark_processed: Mark feature as processed
    validate_flags >> wait_features >> check_ready >> load >> build_obs >> infer >> validate >> check_deployment >> mark_processed
