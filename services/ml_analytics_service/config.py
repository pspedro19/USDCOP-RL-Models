"""
ML Analytics Service Configuration
===================================
Configuration management for ML Analytics backend service.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    """Service configuration"""
    host: str = "0.0.0.0"
    port: int = 8004
    debug: bool = False
    reload: bool = False
    service_name: str = "ML Analytics Service"
    version: str = "1.0.0"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    user: str
    password: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password
        }


@dataclass
class MetricsConfig:
    """Metrics calculation configuration"""
    # Rolling window options
    windows: Dict[str, str] = None

    # Drift detection thresholds
    drift_warning_threshold: float = 0.15
    drift_critical_threshold: float = 0.30

    # Performance calculation settings
    sharpe_ratio_risk_free_rate: float = 0.05  # 5% annual

    def __post_init__(self):
        if self.windows is None:
            self.windows = {
                '1h': '1 hour',
                '24h': '24 hours',
                '7d': '7 days',
                '30d': '30 days'
            }


def get_service_config() -> ServiceConfig:
    """Get service configuration from environment"""
    return ServiceConfig(
        host=os.environ.get('SERVICE_HOST', '0.0.0.0'),
        port=int(os.environ.get('SERVICE_PORT', '8004')),
        debug=os.environ.get('DEBUG', 'false').lower() == 'true',
        reload=os.environ.get('RELOAD', 'false').lower() == 'true'
    )


def get_database_config() -> DatabaseConfig:
    """Get database configuration from environment"""
    return DatabaseConfig(
        host=os.environ.get('POSTGRES_HOST', 'usdcop-postgres-timescale'),
        port=int(os.environ.get('POSTGRES_PORT', '5432')),
        database=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=os.environ.get('POSTGRES_PASSWORD', 'admin123')
    )


def get_metrics_config() -> MetricsConfig:
    """Get metrics configuration"""
    return MetricsConfig()


# Global configuration instances
SERVICE_CONFIG = get_service_config()
DATABASE_CONFIG = get_database_config()
METRICS_CONFIG = get_metrics_config()
