"""
USDCOP Trading System - Utils Package
=====================================
Consolidated utilities for V3 DAGs.

Available Modules:
- dag_common: Database connection and config loading
- datetime_handler: Timezone and datetime utilities
- dependency_handler: Dependency management
- enhanced_api_monitor: API monitoring

Author: Pedro @ Lean Tech Solutions
Version: 3.0.0
"""

# =============================================================================
# CORE EXPORTS - Only modules that exist
# =============================================================================

# DAG Common Utilities (database + config)
from .dag_common import (
    get_db_config,
    get_db_connection,
    load_feature_config,
    get_feature_config_path,
    DB_CONFIG,
    FEATURE_CONFIG_PATH
)

# Datetime Handling
from .datetime_handler import UnifiedDatetimeHandler

# Dependency Handling
from .dependency_handler import (
    MissingDependencyHandler as DependencyHandler,
    get_gymnasium_handler,
    get_stable_baselines3_handler,
    get_handler as get_dependency_handler
)

# API Monitoring
from .enhanced_api_monitor import EnhancedAPIMonitor as APIMonitor

# DAG Dependencies (GEMINI-T6)
from .dag_dependencies import (
    create_dag_dependency,
    DAG_DEPENDENCIES,
    get_upstream_sensors,
)

# Trading Calendar
from .trading_calendar import TradingCalendar, get_calendar, is_trading_day

# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = "3.0.0"
__author__ = "USDCOP Trading System"

# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # DAG Common
    'get_db_config',
    'get_db_connection',
    'load_feature_config',
    'get_feature_config_path',
    'DB_CONFIG',
    'FEATURE_CONFIG_PATH',

    # Datetime
    'UnifiedDatetimeHandler',

    # Dependencies
    'DependencyHandler',
    'get_gymnasium_handler',
    'get_stable_baselines3_handler',
    'get_dependency_handler',

    # API Monitor
    'APIMonitor',

    # Trading Calendar
    'TradingCalendar',
    'get_calendar',
    'is_trading_day',

    # DAG Dependencies
    'create_dag_dependency',
    'DAG_DEPENDENCIES',
    'get_upstream_sensors',
]
