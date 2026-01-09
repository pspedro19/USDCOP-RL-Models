"""
Service Configuration Utilities
================================
Shared configuration management for all services.

DRY: Centralizes hardcoded values like trading hours, thresholds, etc.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingHoursConfig:
    """Trading hours configuration"""
    start_hour: int = 8
    end_hour: int = 12
    end_minute: int = 55
    timezone: str = "America/Bogota"
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    def is_trading_time(self, hour: int, minute: int, weekday: int) -> bool:
        """Check if given time is within trading hours"""
        if weekday not in self.trading_days:
            return False

        if hour < self.start_hour:
            return False

        if hour > self.end_hour:
            return False

        if hour == self.end_hour and minute > self.end_minute:
            return False

        return True


@dataclass
class ServiceConfig:
    """Base service configuration"""
    # Database
    postgres_host: str = "usdcop-postgres-timescale"
    postgres_port: int = 5432
    postgres_db: str = "usdcop_trading"
    postgres_user: str = "admin"
    postgres_password: str = "admin123"

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""

    # Trading
    trading_hours: TradingHoursConfig = field(default_factory=TradingHoursConfig)

    # Benchmarks
    cagr_target_pct: float = 12.0
    sortino_min: float = 1.3
    max_drawdown_max: float = 15.0
    calmar_min: float = 0.8

    # Latency
    onnx_latency_max_ms: float = 20.0
    e2e_latency_max_ms: float = 100.0

    @classmethod
    def from_env(cls) -> 'ServiceConfig':
        """Create config from environment variables"""
        return cls(
            postgres_host=os.environ.get('POSTGRES_HOST', 'usdcop-postgres-timescale'),
            postgres_port=int(os.environ.get('POSTGRES_PORT', '5432')),
            postgres_db=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
            postgres_user=os.environ.get('POSTGRES_USER', 'admin'),
            postgres_password=os.environ.get('POSTGRES_PASSWORD', 'admin123'),
            redis_host=os.environ.get('REDIS_HOST', 'redis'),
            redis_port=int(os.environ.get('REDIS_PORT', '6379')),
            redis_password=os.environ.get('REDIS_PASSWORD', ''),
        )


# Singleton config instance
_service_config: Optional[ServiceConfig] = None


def get_service_config() -> ServiceConfig:
    """Get service configuration singleton"""
    global _service_config
    if _service_config is None:
        _service_config = ServiceConfig.from_env()
    return _service_config


def get_trading_hours() -> TradingHoursConfig:
    """Get trading hours configuration"""
    return get_service_config().trading_hours


def load_feature_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load feature_config.json.

    Args:
        config_path: Optional path to config file

    Returns:
        Config dictionary
    """
    if config_path is None:
        # Default: Look for config relative to services directory
        services_dir = Path(__file__).parent.parent
        config_path = services_dir.parent / "config" / "feature_config.json"

    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load feature config from {config_path}: {e}")
        return {}


# Pre-defined thresholds (from trading_analytics_api.py)
PRODUCTION_GATES = {
    'sortino_min': 1.3,
    'max_drawdown_max': 15.0,
    'calmar_min': 0.8,
    'onnx_latency_max_ms': 20.0,
    'e2e_latency_max_ms': 100.0,
}

BENCHMARK_TARGETS = {
    'cagr_target_pct': 12.0,
    'sharpe_target': 1.0,
    'win_rate_target': 0.55,
}

# Normalization constants (from feature_config.json - SSOT)
NORM_STATS = {
    'dxy_z': {'mean': 103.0, 'std': 5.0},
    'vix_z': {'mean': 20.0, 'std': 10.0},
    'embi_z': {'mean': 300.0, 'std': 100.0},
}
