"""
ConfigLoaderAdapter - Adapter for IConfigLoader Interface
==========================================================

Adapts existing ConfigLoader to implement IConfigLoader interface.

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2025-12-17
"""

from typing import Dict, Any, List, Optional
from ..core.interfaces.config_loader import IConfigLoader
from .config_loader import ConfigLoader


class ConfigLoaderAdapter(IConfigLoader):
    """
    Adapter that makes ConfigLoader implement IConfigLoader interface.

    Implements Adapter Pattern for dependency injection compatibility.

    Example:
        config = ConfigLoaderAdapter()
        features = config.get_feature_order()
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize adapter.

        Args:
            config_dir: Path to config directory
        """
        self._config = ConfigLoader(config_dir)

    def get_feature_order(self) -> List[str]:
        """Get ordered list of features."""
        return self._config.get_feature_order()

    def get_obs_dim(self) -> int:
        """Get total observation dimension."""
        return self._config.get_obs_dim()

    def get_norm_stats(self, feature_name: str) -> Dict[str, float]:
        """Get normalization statistics for a feature."""
        return self._config.get_norm_stats(feature_name)

    def get_clip_bounds(self, feature_name: str) -> Optional[tuple]:
        """Get clipping bounds for a feature."""
        return self._config.get_clip_bounds(feature_name)

    def get_technical_period(self, indicator: str) -> Optional[int]:
        """Get period for technical indicator."""
        return self._config.get_technical_period(indicator)

    def get_trading_params(self) -> Dict[str, Any]:
        """Get trading parameters."""
        return self._config.get_trading_params()

    @property
    def version(self) -> str:
        """Get configuration version."""
        return self._config.version

    # Additional methods from original ConfigLoader
    def get_feature_config(self) -> Dict[str, Any]:
        """Get full feature configuration."""
        return self._config.get_feature_config()

    def get_market_hours(self) -> Dict[str, Any]:
        """Get market hours configuration."""
        return self._config.get_market_hours()

    def get_holidays(self, year: int = 2025, country: str = 'colombia') -> List[str]:
        """Get holiday dates."""
        return self._config.get_holidays(year, country)

    def get_postgres_config(self) -> Dict[str, Any]:
        """Get PostgreSQL configuration."""
        return self._config.get_postgres_config()

    def get_connection_string(self, dialect: str = 'standard') -> str:
        """Get database connection string."""
        return self._config.get_connection_string(dialect)

    def get_table_config(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table configuration."""
        return self._config.get_table_config(table_name)

    def __repr__(self) -> str:
        return f"ConfigLoaderAdapter(version={self.version})"
