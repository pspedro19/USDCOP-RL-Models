"""
Trading Signals Service Configuration
======================================
Configuration settings for the trading signals backend service.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignalServiceConfig:
    """Trading Signals Service Configuration"""

    # Service Settings
    service_name: str = "trading-signals-service"
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8003
    debug: bool = False
    log_level: str = "INFO"

    # Database Configuration (inherited from common)
    postgres_host: str = "usdcop-postgres-timescale"
    postgres_port: int = 5432
    postgres_db: str = "usdcop_trading"
    postgres_user: str = "admin"
    postgres_password: str = "admin123"

    # Redis Configuration (for caching signals)
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    signal_cache_ttl: int = 300  # 5 minutes

    # Model Configuration
    model_path: str = "/app/models/ppo_lstm_v3.2.onnx"
    model_version: str = "ppo_lstm_v3.2"
    model_type: str = "PPO-LSTM"
    inference_timeout_ms: float = 100.0

    # Signal Generation Settings
    confidence_threshold: float = 0.65  # Minimum confidence to generate signal
    min_risk_reward_ratio: float = 1.5  # Minimum R:R for valid signals
    position_size_pct: float = 0.02  # 2% of capital per trade
    max_position_size_pct: float = 0.05  # 5% maximum

    # Risk Management (ATR-based)
    atr_period: int = 14
    atr_multiplier_sl: float = 2.0  # Stop loss: 2x ATR
    atr_multiplier_tp: float = 3.0  # Take profit: 3x ATR

    # Technical Indicators
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_short: int = 20
    ema_long: int = 50

    # WebSocket Settings
    ws_heartbeat_interval: int = 30  # seconds
    ws_max_connections: int = 100
    ws_message_queue_size: int = 1000

    # API Settings
    cors_origins: list = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "*"
    ])
    api_rate_limit: int = 100  # requests per minute

    # Trading Calendar
    trading_start_hour: int = 8
    trading_end_hour: int = 12
    trading_end_minute: int = 55
    timezone: str = "America/Bogota"
    trading_days: list = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    # Signal History
    signal_history_limit: int = 1000
    signal_retention_days: int = 90

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9003

    @classmethod
    def from_env(cls) -> 'SignalServiceConfig':
        """Load configuration from environment variables"""
        return cls(
            # Service
            host=os.environ.get('SIGNAL_SERVICE_HOST', '0.0.0.0'),
            port=int(os.environ.get('SIGNAL_SERVICE_PORT', '8003')),
            debug=os.environ.get('DEBUG', 'false').lower() == 'true',
            log_level=os.environ.get('LOG_LEVEL', 'INFO'),

            # Database
            postgres_host=os.environ.get('POSTGRES_HOST', 'usdcop-postgres-timescale'),
            postgres_port=int(os.environ.get('POSTGRES_PORT', '5432')),
            postgres_db=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
            postgres_user=os.environ.get('POSTGRES_USER', 'admin'),
            postgres_password=os.environ.get('POSTGRES_PASSWORD', 'admin123'),

            # Redis
            redis_host=os.environ.get('REDIS_HOST', 'redis'),
            redis_port=int(os.environ.get('REDIS_PORT', '6379')),
            redis_password=os.environ.get('REDIS_PASSWORD', ''),

            # Model
            model_path=os.environ.get('MODEL_PATH', '/app/models/ppo_lstm_v3.2.onnx'),
            model_version=os.environ.get('MODEL_VERSION', 'ppo_lstm_v3.2'),

            # Risk Management
            confidence_threshold=float(os.environ.get('CONFIDENCE_THRESHOLD', '0.65')),
            position_size_pct=float(os.environ.get('POSITION_SIZE_PCT', '0.02')),
            atr_multiplier_sl=float(os.environ.get('ATR_MULTIPLIER_SL', '2.0')),
            atr_multiplier_tp=float(os.environ.get('ATR_MULTIPLIER_TP', '3.0')),
        )

    def get_db_config(self) -> dict:
        """Get database configuration as dictionary"""
        return {
            'host': self.postgres_host,
            'port': self.postgres_port,
            'database': self.postgres_db,
            'user': self.postgres_user,
            'password': self.postgres_password
        }

    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Global configuration singleton
_config: Optional[SignalServiceConfig] = None


def get_config() -> SignalServiceConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = SignalServiceConfig.from_env()
        logger.info(f"Loaded configuration: {_config.service_name} v{_config.service_version}")
    return _config


# Export commonly used values
def get_model_config() -> dict:
    """Get model configuration"""
    config = get_config()
    return {
        'path': config.model_path,
        'version': config.model_version,
        'type': config.model_type,
        'timeout_ms': config.inference_timeout_ms
    }


def get_risk_config() -> dict:
    """Get risk management configuration"""
    config = get_config()
    return {
        'confidence_threshold': config.confidence_threshold,
        'min_risk_reward': config.min_risk_reward_ratio,
        'position_size_pct': config.position_size_pct,
        'max_position_size_pct': config.max_position_size_pct,
        'atr_period': config.atr_period,
        'atr_multiplier_sl': config.atr_multiplier_sl,
        'atr_multiplier_tp': config.atr_multiplier_tp
    }
