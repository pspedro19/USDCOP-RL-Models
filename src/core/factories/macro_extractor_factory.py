"""
Macro Extractor Factory
=======================

Factory Pattern implementation for creating macro data extraction strategies.
Provides centralized creation of extractors based on configuration.

Contract: CTR-L0-FACTORY-001

Pattern: Factory Pattern
    - Centralized extractor creation
    - Configuration-driven instantiation
    - Lazy registration of strategies

Usage:
    factory = MacroExtractorFactory(config_path='/opt/airflow/config/l0_macro_sources.yaml')
    extractors = factory.create_all_extractors()

    # Or create individual extractor:
    fred_extractor = factory.create(MacroSource.FRED)

Version: 1.0.0
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

import yaml

from src.core.interfaces.macro_extractor import (
    BaseMacroExtractor,
    ExtractionResult,
    MacroExtractionStrategy,
)


logger = logging.getLogger(__name__)


class MacroSource(str, Enum):
    """
    Enumeration of all supported macro data sources.

    Maps to source configurations in l0_macro_sources.yaml.
    """
    FRED = "fred"
    TWELVEDATA = "twelvedata"
    INVESTING = "investing"
    BANREP = "banrep"
    BANREP_SDMX = "banrep_sdmx"  # BanRep SDMX REST API (alternative to Selenium)
    BANREP_BOP = "banrep_bop"   # BanRep Balance of Payments (quarterly data)
    BCRP = "bcrp"
    FEDESARROLLO = "fedesarrollo"
    DANE = "dane"


class MacroExtractorFactory:
    """
    Factory for creating macro data extraction strategies.

    Implements the Factory Pattern for centralized extractor creation.
    Supports lazy registration of strategy classes.

    Attributes:
        _strategies: Registered strategy classes by source
        _instances: Cached extractor instances
        config: Loaded configuration from YAML
    """

    # Class-level registry of strategy classes
    _strategies: Dict[MacroSource, Type[BaseMacroExtractor]] = {}

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize factory with configuration.

        Args:
            config_path: Path to l0_macro_sources.yaml
        """
        self.config_path = config_path or '/opt/airflow/config/l0_macro_sources.yaml'
        self.config: Dict[str, Any] = {}
        self._instances: Dict[MacroSource, BaseMacroExtractor] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"[Factory] Loaded config from {self.config_path}")
            else:
                logger.warning(f"[Factory] Config file not found: {self.config_path}")
                self.config = {}
        except Exception as e:
            logger.error(f"[Factory] Failed to load config: {e}")
            self.config = {}

    @classmethod
    def register(cls, source: MacroSource, strategy_class: Type[BaseMacroExtractor]) -> None:
        """
        Register a strategy class for a source.

        Args:
            source: The macro source enum
            strategy_class: The strategy class to register
        """
        cls._strategies[source] = strategy_class
        logger.debug(f"[Factory] Registered {strategy_class.__name__} for {source.value}")

    @classmethod
    def register_strategy(cls, source: MacroSource) -> Callable:
        """
        Decorator for registering strategy classes.

        Usage:
            @MacroExtractorFactory.register_strategy(MacroSource.FRED)
            class FREDExtractionStrategy(BaseMacroExtractor):
                ...

        Args:
            source: The macro source to register for

        Returns:
            Decorator function
        """
        def decorator(strategy_class: Type[BaseMacroExtractor]) -> Type[BaseMacroExtractor]:
            cls.register(source, strategy_class)
            return strategy_class
        return decorator

    def create(self, source: MacroSource, **kwargs) -> BaseMacroExtractor:
        """
        Create an extractor instance for a source.

        Args:
            source: The macro source to create extractor for
            **kwargs: Additional arguments to pass to constructor

        Returns:
            Extractor instance

        Raises:
            ValueError: If source not registered or not enabled
        """
        # Check if source is registered
        if source not in self._strategies:
            raise ValueError(f"Unknown source: {source.value}. "
                           f"Available: {list(self._strategies.keys())}")

        # Check if source is enabled
        enabled = self.config.get('enabled_sources', [])
        if source.value not in enabled:
            raise ValueError(f"Source {source.value} is not enabled in configuration")

        # Get source-specific config
        source_config = self.config.get('sources', {}).get(source.value, {})

        # Merge with global config
        global_config = self.config.get('global', {})
        merged_config = {**global_config, **source_config}

        # Create instance
        strategy_class = self._strategies[source]
        return strategy_class(config=merged_config, **kwargs)

    def get_or_create(self, source: MacroSource, **kwargs) -> BaseMacroExtractor:
        """
        Get cached instance or create new one.

        Args:
            source: The macro source
            **kwargs: Additional arguments for creation

        Returns:
            Extractor instance (cached if available)
        """
        if source not in self._instances:
            self._instances[source] = self.create(source, **kwargs)
        return self._instances[source]

    def create_all_extractors(self) -> Dict[MacroSource, BaseMacroExtractor]:
        """
        Create extractors for all enabled sources.

        Returns:
            Dictionary mapping source to extractor instance
        """
        extractors = {}
        enabled = self.config.get('enabled_sources', [])

        for source_name in enabled:
            try:
                source = MacroSource(source_name)
                if source in self._strategies:
                    extractors[source] = self.create(source)
                    logger.info(f"[Factory] Created extractor for {source.value}")
                else:
                    logger.warning(f"[Factory] No strategy registered for {source.value}")
            except ValueError as e:
                logger.warning(f"[Factory] Unknown source in config: {source_name}")
            except Exception as e:
                logger.error(f"[Factory] Failed to create extractor for {source_name}: {e}")

        return extractors

    def get_source_config(self, source: MacroSource) -> Dict[str, Any]:
        """
        Get configuration for a specific source.

        Args:
            source: The macro source

        Returns:
            Source configuration dictionary
        """
        return self.config.get('sources', {}).get(source.value, {})

    def get_indicators_for_source(self, source: MacroSource) -> Dict[str, str]:
        """
        Get indicator mappings for a source.

        Args:
            source: The macro source

        Returns:
            Dictionary of source_id -> column_name
        """
        source_config = self.get_source_config(source)
        indicators = source_config.get('indicators', {})

        # Handle different config formats
        if isinstance(indicators, dict):
            # Simple format: {source_id: column_name}
            result = {}
            for key, value in indicators.items():
                if isinstance(value, str):
                    result[key] = value
                elif isinstance(value, dict):
                    # Complex format: {source_id: {column: ..., name: ...}}
                    result[key] = value.get('column', '')
            return result

        return {}

    def get_lookback_days(self, source: MacroSource) -> int:
        """
        Get lookback days for a source.

        Args:
            source: The macro source

        Returns:
            Number of lookback days
        """
        source_config = self.get_source_config(source)
        return source_config.get(
            'lookback_days',
            self.config.get('global', {}).get('default_lookback_days', 90)
        )

    def is_source_enabled(self, source: MacroSource) -> bool:
        """
        Check if a source is enabled.

        Args:
            source: The macro source

        Returns:
            True if enabled
        """
        enabled = self.config.get('enabled_sources', [])
        return source.value in enabled

    @property
    def enabled_sources(self) -> list:
        """Get list of enabled source names."""
        return self.config.get('enabled_sources', [])

    @classmethod
    def get_registered_sources(cls) -> list:
        """Get list of registered source enums."""
        return list(cls._strategies.keys())


# Singleton factory instance (lazy initialization)
_factory_instance: Optional[MacroExtractorFactory] = None


def get_extractor_factory(config_path: Optional[str] = None) -> MacroExtractorFactory:
    """
    Get the singleton factory instance.

    Args:
        config_path: Optional config path (used only on first call)

    Returns:
        MacroExtractorFactory instance
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = MacroExtractorFactory(config_path)
    return _factory_instance


def reset_factory() -> None:
    """Reset the singleton factory instance (for testing)."""
    global _factory_instance
    _factory_instance = None
