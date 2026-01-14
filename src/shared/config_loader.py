"""
USD/COP Trading System - Configuration Loader
==============================================

Centralized configuration loading with caching for:
- feature_config.json (features, normalization, trading params)
- trading_calendar.json (market hours, holidays, schedules)
- database.yaml (database connections, table definitions)

Ensures Single Source of Truth (SSOT) across training and inference.

Author: Pedro @ Lean Tech Solutions
Version: 2.0.1
Date: 2025-12-17
Changes: Added thread-safe singleton with double-checked locking
"""

import json
import yaml
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import lru_cache

# Flexible import for exceptions
try:
    from .exceptions import ConfigurationError
except ImportError:
    # When loaded standalone, import from same directory
    _shared_path = Path(__file__).parent
    if str(_shared_path) not in sys.path:
        sys.path.insert(0, str(_shared_path))
    from exceptions import ConfigurationError


class ConfigLoader:
    """
    Loads and caches configuration from feature_config.json, trading_calendar.json,
    and database.yaml.

    Provides centralized access to all feature parameters, normalization stats,
    trading configuration, market hours, and database configuration.

    Usage:
        config = ConfigLoader()
        features = config.get_feature_order()
        norm_stats = config.get_norm_stats('rsi_9')
        market_hours = config.get_market_hours()
        db_config = config.get_database_config()
    """

    _instance = None
    _lock = threading.Lock()  # Thread safety for singleton
    _feature_config = None
    _trading_calendar = None
    _database_config = None

    def __new__(cls, config_dir: Optional[str] = None):
        """Thread-safe singleton pattern using double-checked locking"""
        if cls._instance is None:
            with cls._lock:  # Acquire lock before second check
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super(ConfigLoader, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_dir: Path to config directory. If None, uses default location.
        """
        if self._initialized:
            return

        if config_dir is None:
            # Default: config/ relative to project root
            # Assumes this file is in src/shared/
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self._load_all_configs()
        self._initialized = True

    def _load_all_configs(self):
        """Load all configuration files"""
        # Load feature_config.json
        feature_config_path = self.config_dir / "feature_config.json"
        if not feature_config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found",
                config_path=str(feature_config_path)
            )

        try:
            with open(feature_config_path, 'r', encoding='utf-8') as f:
                self._feature_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in feature_config.json: {e}",
                config_path=str(feature_config_path)
            )

        # Load trading_calendar.json
        calendar_path = self.config_dir / "trading_calendar.json"
        if calendar_path.exists():
            try:
                with open(calendar_path, 'r', encoding='utf-8') as f:
                    self._trading_calendar = json.load(f)
            except json.JSONDecodeError as e:
                raise ConfigurationError(
                    f"Invalid JSON in trading_calendar.json: {e}",
                    config_path=str(calendar_path)
                )
        else:
            # Use fallback from feature_config.json
            self._trading_calendar = {
                "market_hours": self._feature_config.get("trading", {}).get("market_hours", {}),
                "holidays_2025_colombia": self._feature_config.get("holidays_2025_colombia", [])
            }

        # Load database.yaml
        db_config_path = self.config_dir / "database.yaml"
        if db_config_path.exists():
            try:
                with open(db_config_path, 'r', encoding='utf-8') as f:
                    self._database_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigurationError(
                    f"Invalid YAML in database.yaml: {e}",
                    config_path=str(db_config_path)
                )
        else:
            # Minimal fallback
            self._database_config = {
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "usdcop_trading",
                    "user": "admin",
                    "password": "admin"
                }
            }

        self._validate_configs()

    def _validate_configs(self):
        """Validate required sections exist in configs"""
        # Validate feature_config
        required_sections = ['_meta', 'observation_space', 'features', 'normalization', 'trading']
        missing = [s for s in required_sections if s not in self._feature_config]

        if missing:
            raise ConfigurationError(
                f"feature_config.json missing required sections: {missing}",
                config_path=str(self.config_dir / "feature_config.json")
            )

        # Validate observation space
        obs = self._feature_config['observation_space']
        if 'order' not in obs or 'total_obs_dim' not in obs:
            raise ConfigurationError(
                "observation_space must contain 'order' and 'total_obs_dim'",
                config_path=str(self.config_dir / "feature_config.json")
            )

    # =========================================================================
    # FEATURE CONFIG ACCESSORS
    # =========================================================================

    @property
    def feature_config(self) -> Dict[str, Any]:
        """Get full feature configuration dictionary"""
        return self._feature_config

    @property
    def version(self) -> str:
        """Get configuration version"""
        return self._feature_config['_meta']['version']

    def get_feature_config(self) -> Dict[str, Any]:
        """Get full feature configuration (method version)"""
        return self._feature_config

    def get_feature_order(self) -> List[str]:
        """Get ordered list of 13 features for model input"""
        return self._feature_config['observation_space']['order']

    def get_obs_dim(self) -> int:
        """Get total observation dimension (13 features + 2 state = 15)"""
        return self._feature_config['observation_space']['total_obs_dim']

    def get_norm_stats(self, feature_name: str) -> Dict[str, float]:
        """
        Get normalization statistics for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dict with 'mean' and 'std' keys, or empty dict if not found
        """
        # Search in all feature categories
        for category in ['returns', 'technical', 'macro_zscore', 'macro_changes', 'macro_derived']:
            if category in self._feature_config['features']:
                items = self._feature_config['features'][category].get('items', [])
                for item in items:
                    if item.get('name') == feature_name:
                        return item.get('norm_stats', {})
        return {}

    def get_clip_bounds(self, feature_name: str) -> Optional[tuple]:
        """
        Get clipping bounds for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Tuple (min, max) or None if not defined
        """
        for category in ['returns', 'technical', 'macro_zscore', 'macro_changes']:
            if category in self._feature_config['features']:
                items = self._feature_config['features'][category].get('items', [])
                for item in items:
                    if item.get('name') == feature_name:
                        clip = item.get('clip')
                        if clip:
                            return tuple(clip)
        return None

    def get_technical_period(self, indicator: str) -> Optional[int]:
        """
        Get period for a technical indicator.

        Args:
            indicator: Indicator name (e.g., 'rsi_9', 'atr_pct', 'adx_14')

        Returns:
            Period value or None if not found
        """
        if 'technical' in self._feature_config['features']:
            items = self._feature_config['features']['technical']['items']
            for item in items:
                if item['name'] == indicator:
                    return item.get('period')
        return None

    def get_trading_params(self) -> Dict[str, Any]:
        """Get all trading parameters"""
        return self._feature_config.get('trading', {})

    def get_sql_features(self) -> List[str]:
        """Get list of features calculated in SQL"""
        if 'compute_strategy' in self._feature_config:
            sql_calc = self._feature_config['compute_strategy'].get('sql_calculated', {})
            return sql_calc.get('features', [])
        return []

    def get_python_features(self) -> List[str]:
        """Get list of features calculated in Python"""
        if 'compute_strategy' in self._feature_config:
            python_calc = self._feature_config['compute_strategy'].get('python_calculated', {})
            return python_calc.get('features', [])
        return []

    # =========================================================================
    # TRADING CALENDAR ACCESSORS
    # =========================================================================

    @property
    def trading_calendar(self) -> Dict[str, Any]:
        """Get full trading calendar configuration"""
        return self._trading_calendar

    def get_trading_calendar(self) -> Dict[str, Any]:
        """Get full trading calendar configuration (method version)"""
        return self._trading_calendar

    def get_market_hours(self) -> Dict[str, Any]:
        """
        Get market hours configuration.

        Returns:
            Dict with timezone, utc_offset, local start/end, utc start/end
        """
        # Try trading_calendar first, then fall back to feature_config
        if self._trading_calendar and 'market_hours' in self._trading_calendar:
            return self._trading_calendar['market_hours']
        return self._feature_config['trading']['market_hours']

    def get_holidays(self, year: int = 2025, country: str = 'colombia') -> List[str]:
        """
        Get list of holiday dates for the specified year and country.

        Args:
            year: Year (default: 2025)
            country: Country code ('colombia' or 'usa')

        Returns:
            List of holiday dates in 'YYYY-MM-DD' format
        """
        key = f"holidays_{year}_{country}"
        if self._trading_calendar and key in self._trading_calendar:
            holidays_data = self._trading_calendar[key]
            # Handle both dict format {'dates': [...]} and list format [...]
            if isinstance(holidays_data, dict):
                return holidays_data.get('dates', [])
            elif isinstance(holidays_data, list):
                return holidays_data
        # Fallback to feature_config
        if key in self._feature_config:
            holidays_data = self._feature_config[key]
            if isinstance(holidays_data, dict):
                return holidays_data.get('dates', [])
            elif isinstance(holidays_data, list):
                return holidays_data
        return []

    def get_schedules(self) -> Dict[str, Any]:
        """
        Get DAG scheduling configuration.

        Returns:
            Dict with cron expressions for ohlcv, macro, inference, monitoring
        """
        if self._trading_calendar and 'schedules' in self._trading_calendar:
            return self._trading_calendar['schedules']
        return {}

    # =========================================================================
    # DATABASE CONFIG ACCESSORS
    # =========================================================================

    @property
    def database_config(self) -> Dict[str, Any]:
        """Get full database configuration"""
        return self._database_config

    def get_database_config(self) -> Dict[str, Any]:
        """Get full database configuration (method version)"""
        return self._database_config

    def get_postgres_config(self) -> Dict[str, Any]:
        """Get PostgreSQL connection configuration"""
        return self._database_config.get('postgres', {})

    def get_connection_string(self, dialect: str = 'standard') -> str:
        """
        Get database connection string.

        Args:
            dialect: Connection string type ('standard', 'sqlalchemy', 'asyncpg')

        Returns:
            Connection string (may contain ${ENV_VAR} placeholders)
        """
        postgres = self._database_config.get('postgres', {})
        conn_strings = postgres.get('connection_strings', {})
        return conn_strings.get(dialect, '')

    def get_table_config(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific table.

        Args:
            table_name: Name of the table (e.g., 'ohlcv', 'macro_indicators')

        Returns:
            Table configuration dict or None if not found
        """
        tables = self._database_config.get('tables', {})
        return tables.get(table_name)


def get_config(config_dir: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration instance.

    This is a convenience function that returns the ConfigLoader singleton.
    The underlying ConfigLoader class handles singleton pattern with thread safety.

    Args:
        config_dir: Optional path to config directory (only used on first call)

    Returns:
        ConfigLoader singleton instance
    """
    return ConfigLoader(config_dir)


@lru_cache(maxsize=1)
def load_feature_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load feature configuration with caching.

    This is a convenience function that returns the feature config dict directly.

    Args:
        config_path: Optional path to feature_config.json

    Returns:
        Feature configuration dictionary
    """
    if config_path:
        config_dir = Path(config_path).parent
        config = ConfigLoader(str(config_dir))
    else:
        config = get_config()

    return config.get_feature_config()
