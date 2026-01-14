"""
MLOps Configuration Module
==========================

Centralized configuration for all MLOps components.
Supports environment variables and YAML configuration.

Environment Variables:
    MLOPS_MODEL_PATH: Path to ONNX model
    MLOPS_REDIS_URL: Redis connection URL
    MLOPS_LOG_LEVEL: Logging level
    MLOPS_ENV: Environment (development/staging/production)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import time
from enum import Enum
import yaml
import logging

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SignalType(str, Enum):
    HOLD = "HOLD"
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TradingHours:
    """Colombian trading hours configuration."""
    start_hour: int = 8
    start_minute: int = 0
    end_hour: int = 12
    end_minute: int = 55
    timezone: str = "America/Bogota"
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    @property
    def start_time(self) -> time:
        return time(self.start_hour, self.start_minute)

    @property
    def end_time(self) -> time:
        return time(self.end_hour, self.end_minute)

    def is_trading_time(self, current_time: time, current_weekday: int) -> bool:
        """Check if current time is within trading hours."""
        if current_weekday not in self.trading_days:
            return False
        return self.start_time <= current_time <= self.end_time


@dataclass
class RiskLimits:
    """Risk management thresholds."""
    # Daily limits
    max_daily_loss: float = -0.02  # -2% daily loss limit
    max_daily_profit_target: float = 0.05  # +5% daily profit target (optional stop)

    # Drawdown limits
    max_drawdown: float = -0.05  # -5% maximum drawdown
    max_intraday_drawdown: float = -0.03  # -3% intraday drawdown

    # Trade limits
    max_consecutive_losses: int = 5
    max_trades_per_day: int = 50
    max_position_size: float = 0.10  # 10% of capital per trade

    # Signal thresholds
    min_confidence: float = 0.60  # Minimum 60% confidence to execute
    high_confidence_threshold: float = 0.80  # High confidence threshold

    # Cooldown periods (seconds)
    cooldown_after_loss: int = 60  # 1 minute cooldown after loss
    cooldown_after_circuit_break: int = 300  # 5 minutes after circuit breaker

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_daily_loss": self.max_daily_loss,
            "max_daily_profit_target": self.max_daily_profit_target,
            "max_drawdown": self.max_drawdown,
            "max_intraday_drawdown": self.max_intraday_drawdown,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_trades_per_day": self.max_trades_per_day,
            "max_position_size": self.max_position_size,
            "min_confidence": self.min_confidence,
            "high_confidence_threshold": self.high_confidence_threshold,
        }


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    algorithm: str  # PPO, SAC, A2C
    onnx_path: str
    observation_dim: int = 45
    action_space_size: int = 3  # HOLD, BUY, SELL
    weight: float = 1.0  # Weight in ensemble
    enabled: bool = True


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "redis"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    retry_on_timeout: bool = True

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    enable_drift_detection: bool = True
    drift_check_interval_minutes: int = 60
    drift_threshold: float = 0.15  # 15% features drifted = alert

    enable_performance_tracking: bool = True
    performance_log_interval_seconds: int = 60

    enable_health_checks: bool = True
    health_check_interval_seconds: int = 30

    # Alert thresholds
    latency_alert_ms: float = 100.0
    error_rate_alert: float = 0.01  # 1% error rate


@dataclass
class MLOpsConfig:
    """Main MLOps configuration."""
    # Environment
    environment: Environment = Environment.DEVELOPMENT

    # Model settings
    models: List[ModelConfig] = field(default_factory=list)
    default_observation_dim: int = 45
    ensemble_strategy: str = "weighted_average"  # or "majority_vote"

    # Risk management
    risk_limits: RiskLimits = field(default_factory=RiskLimits)

    # Trading hours
    trading_hours: TradingHours = field(default_factory=TradingHours)

    # Redis
    redis: RedisConfig = field(default_factory=RedisConfig)

    # Monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Feature cache settings
    feature_cache_ttl_seconds: int = 300  # 5 minutes
    feature_history_size: int = 1000

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    @classmethod
    def from_env(cls) -> "MLOpsConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Environment
        env = os.getenv("MLOPS_ENV", "development").lower()
        config.environment = Environment(env)

        # Redis
        config.redis.host = os.getenv("REDIS_HOST", "redis")
        config.redis.port = int(os.getenv("REDIS_PORT", "6379"))
        config.redis.password = os.getenv("REDIS_PASSWORD")

        # Logging
        config.log_level = os.getenv("MLOPS_LOG_LEVEL", "INFO")

        # Risk limits from env
        if os.getenv("MLOPS_MAX_DAILY_LOSS"):
            config.risk_limits.max_daily_loss = float(os.getenv("MLOPS_MAX_DAILY_LOSS"))
        if os.getenv("MLOPS_MAX_DRAWDOWN"):
            config.risk_limits.max_drawdown = float(os.getenv("MLOPS_MAX_DRAWDOWN"))
        if os.getenv("MLOPS_MIN_CONFIDENCE"):
            config.risk_limits.min_confidence = float(os.getenv("MLOPS_MIN_CONFIDENCE"))

        return config

    @classmethod
    def from_yaml(cls, path: str) -> "MLOpsConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()

        if "environment" in data:
            config.environment = Environment(data["environment"])

        if "models" in data:
            config.models = [
                ModelConfig(**m) for m in data["models"]
            ]

        if "risk_limits" in data:
            config.risk_limits = RiskLimits(**data["risk_limits"])

        if "trading_hours" in data:
            config.trading_hours = TradingHours(**data["trading_hours"])

        if "redis" in data:
            config.redis = RedisConfig(**data["redis"])

        if "monitoring" in data:
            config.monitoring = MonitoringConfig(**data["monitoring"])

        return config

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = {
            "environment": self.environment.value,
            "models": [
                {
                    "name": m.name,
                    "algorithm": m.algorithm,
                    "onnx_path": m.onnx_path,
                    "observation_dim": m.observation_dim,
                    "weight": m.weight,
                    "enabled": m.enabled,
                }
                for m in self.models
            ],
            "risk_limits": self.risk_limits.to_dict(),
            "trading_hours": {
                "start_hour": self.trading_hours.start_hour,
                "start_minute": self.trading_hours.start_minute,
                "end_hour": self.trading_hours.end_hour,
                "end_minute": self.trading_hours.end_minute,
                "timezone": self.trading_hours.timezone,
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
            },
            "monitoring": {
                "enable_drift_detection": self.monitoring.enable_drift_detection,
                "drift_check_interval_minutes": self.monitoring.drift_check_interval_minutes,
                "drift_threshold": self.monitoring.drift_threshold,
            },
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def setup_logging(self):
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format=self.log_format,
        )


# Global configuration instance
_config: Optional[MLOpsConfig] = None


def get_config() -> MLOpsConfig:
    """Get or create global configuration."""
    global _config
    if _config is None:
        # Try to load from file first
        config_path = os.getenv("MLOPS_CONFIG_PATH", "config/mlops.yaml")
        if os.path.exists(config_path):
            _config = MLOpsConfig.from_yaml(config_path)
            logger.info(f"Loaded config from {config_path}")
        else:
            _config = MLOpsConfig.from_env()
            logger.info("Loaded config from environment")

        # Always apply env var overrides (especially for secrets)
        _config.redis.host = os.getenv("REDIS_HOST", _config.redis.host)
        _config.redis.port = int(os.getenv("REDIS_PORT", str(_config.redis.port)))
        _config.redis.password = os.getenv("REDIS_PASSWORD", _config.redis.password)
        _config.log_level = os.getenv("MLOPS_LOG_LEVEL", _config.log_level)

        _config.setup_logging()

    return _config


def set_config(config: MLOpsConfig):
    """Set global configuration."""
    global _config
    _config = config
    _config.setup_logging()
