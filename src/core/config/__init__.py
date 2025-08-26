"""
Core Configuration Module
========================
Provides unified configuration management for the trading system.
"""

from .unified_config import (
    SystemConfig,
    UnifiedConfigLoader,
    config_loader,
    get_config,
    get_system_config
)

__all__ = [
    'SystemConfig',
    'UnifiedConfigLoader',
    'config_loader',
    'get_config',
    'get_system_config'
]