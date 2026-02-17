# -*- coding: utf-8 -*-
"""
ExtractorRegistry - Single Source of Truth for all extractors.

Singleton pattern ensures consistent extractor instances across
all DAGs and services. Configuration loaded from SSOT (macro_variables_ssot.yaml).

Usage:
    registry = ExtractorRegistry()
    result = registry.extract_variable('FINC_RATE_IBR_OVERNIGHT_COL_D_IBR', last_n=5)
"""

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml

from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_EXTRACTOR_CLASSES: Dict[str, Type[BaseExtractor]] = {}


def _register_extractors():
    """Register all extractor classes (lazy loading)."""
    global _EXTRACTOR_CLASSES

    if _EXTRACTOR_CLASSES:
        return

    from .fred_extractor import FredExtractor
    from .suameca_extractor import SuamecaExtractor
    from .bcrp_extractor import BcrpExtractor
    from .investing_extractor import InvestingExtractor
    from .dane_extractor import DaneExtractor
    from .fedesarrollo_extractor import FedesarrolloExtractor

    _EXTRACTOR_CLASSES = {
        'fred': FredExtractor,
        'suameca': SuamecaExtractor,
        'bcrp': BcrpExtractor,
        'investing': InvestingExtractor,
        'dane': DaneExtractor,
        'fedesarrollo': FedesarrolloExtractor,
    }


def _get_ssot():
    """Try to load MacroSSOT."""
    try:
        # Add src to path if needed
        project_root = Path(__file__).parent.parent.parent.parent
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from data.macro_ssot import MacroSSOT
        return MacroSSOT()
    except ImportError:
        return None


class ExtractorRegistry:
    """
    Central registry of all data extractors.

    Now reads from SSOT (macro_variables_ssot.yaml) as the source of truth
    for variable definitions while maintaining backwards compatibility
    with the local config.yaml for extractor-specific settings.

    Provides:
    - Single point of access to all extractors
    - Variable-to-source mapping
    - Unified extraction interface
    """

    _instance: Optional['ExtractorRegistry'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._extractors: Dict[str, BaseExtractor] = {}
        self._variable_map: Dict[str, str] = {}  # variable -> source
        self._variable_config: Dict[str, dict] = {}  # variable -> config
        self._global_config: Dict[str, Any] = {}

        self._load_config()
        self._initialized = True

    def _load_config(self):
        """Load configuration from SSOT and local config.yaml."""
        _register_extractors()

        # First, try to load from SSOT
        ssot = _get_ssot()
        if ssot:
            self._load_from_ssot(ssot)
        else:
            # Fallback to local config.yaml
            self._load_from_local_config()

    def _load_from_ssot(self, ssot):
        """Load configuration from SSOT."""
        logger.info("Loading extractor config from SSOT")

        # Get global config from SSOT
        self._global_config = ssot.get_global_config()

        # Build variables list per source first
        source_variables: Dict[str, List[dict]] = {}
        for var_name in ssot.get_all_variables():
            var_def = ssot.get_variable(var_name)
            if var_def is None:
                continue

            source = var_def.extraction.primary_source
            source_config = var_def.extraction.source_configs.get(source, {})

            # Build variable config dict
            var_cfg = {
                'name': var_name,
                'frequency': 'D' if var_def.identity.frequency == 'daily' else (
                    'M' if var_def.identity.frequency == 'monthly' else 'Q'
                ),
                'intraday': var_def.extraction.intraday,
            }

            # Add source-specific config with field mapping
            if source == 'investing':
                # Map pair_id -> instrument_id
                if 'pair_id' in source_config and 'instrument_id' not in source_config:
                    source_config['instrument_id'] = source_config['pair_id']
                if 'domain_id' not in source_config:
                    source_config['domain_id'] = 'www'
                if 'url' in source_config and 'referer_url' not in source_config:
                    source_config['referer_url'] = source_config.get('url')

            var_cfg.update(source_config)

            if source not in source_variables:
                source_variables[source] = []
            source_variables[source].append(var_cfg)

        # Get source configs and create extractors
        sources = ['fred', 'suameca', 'bcrp', 'investing', 'dane', 'fedesarrollo']

        for source_name in sources:
            source_cfg = ssot.get_source_config(source_name)
            if not source_cfg.get('enabled', True):
                logger.info("Source %s is disabled", source_name)
                continue

            extractor_class = _EXTRACTOR_CLASSES.get(source_name)
            if extractor_class is None:
                logger.warning("No extractor class for source: %s", source_name)
                continue

            try:
                # Merge global config with source config and add variables list
                merged_cfg = {**self._global_config, **source_cfg}
                merged_cfg['variables'] = source_variables.get(source_name, [])
                extractor = extractor_class(merged_cfg)
                self._extractors[source_name] = extractor

                logger.info("Loaded extractor %s with %d variables from SSOT",
                           source_name, len(merged_cfg['variables']))
            except Exception as e:
                logger.error("Failed to load extractor %s: %s", source_name, e)

        # Map variables to sources and store config (already built above)
        for source_name, vars_list in source_variables.items():
            for var_cfg in vars_list:
                var_name = var_cfg['name']
                self._variable_map[var_name] = source_name
                self._variable_config[var_name] = var_cfg

        logger.info(
            "Loaded %d variables from SSOT across %d sources",
            len(self._variable_map), len(self._extractors)
        )

    def _load_from_local_config(self):
        """Fallback: Load configuration from local config.yaml."""
        config_path = Path(__file__).parent / 'config.yaml'

        if not config_path.exists():
            logger.warning("Config file not found: %s", config_path)
            return

        logger.info("Loading extractor config from local config.yaml")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self._global_config = config.get('global', {})

        for source_name, source_cfg in config.get('sources', {}).items():
            if not source_cfg.get('enabled', True):
                logger.info("Source %s is disabled", source_name)
                continue

            extractor_class = _EXTRACTOR_CLASSES.get(source_name)
            if extractor_class is None:
                logger.warning("No extractor class for source: %s", source_name)
                continue

            try:
                # Merge global config with source config
                merged_cfg = {**self._global_config, **source_cfg}
                extractor = extractor_class(merged_cfg)
                self._extractors[source_name] = extractor

                # Map variables to source
                for var_cfg in source_cfg.get('variables', []):
                    var_name = var_cfg.get('name')
                    if var_name:
                        self._variable_map[var_name] = source_name
                        self._variable_config[var_name] = var_cfg

                logger.info(
                    "Loaded extractor %s with %d variables",
                    source_name, len(source_cfg.get('variables', []))
                )
            except Exception as e:
                logger.error("Failed to load extractor %s: %s", source_name, e)

    def get_extractor(self, source: str) -> Optional[BaseExtractor]:
        """Get extractor by source name."""
        return self._extractors.get(source)

    def get_extractor_for_variable(self, variable: str) -> Optional[BaseExtractor]:
        """Get the extractor responsible for a variable."""
        source = self._variable_map.get(variable)
        return self._extractors.get(source) if source else None

    def get_variable_config(self, variable: str) -> dict:
        """Get configuration for a specific variable."""
        return self._variable_config.get(variable, {})

    def extract_variable(
        self,
        variable: str,
        start_date=None,
        end_date=None,
        last_n: int = None
    ) -> ExtractionResult:
        """
        Extract data for a variable using the appropriate extractor.

        Args:
            variable: Variable name
            start_date: Start date (for backfill)
            end_date: End date
            last_n: Extract only last N records (for realtime)

        Returns:
            ExtractionResult with data
        """
        from datetime import datetime

        extractor = self.get_extractor_for_variable(variable)

        if extractor is None:
            return ExtractionResult(
                source='unknown',
                variable=variable,
                data=None,
                success=False,
                error=f"No extractor found for variable: {variable}"
            )

        if last_n is not None:
            return extractor.extract_last_n(variable, n=last_n)

        if start_date is None:
            start_date = datetime(2020, 1, 1)
        if end_date is None:
            end_date = datetime.now()

        return extractor.extract(variable, start_date, end_date)

    @lru_cache(maxsize=1)
    def get_all_variables(self) -> List[str]:
        """Get list of all available variables."""
        return list(self._variable_map.keys())

    def get_all_sources(self) -> List[str]:
        """Get list of all loaded sources."""
        return list(self._extractors.keys())

    def get_variables_by_source(self, source: str) -> List[str]:
        """Get all variables for a specific source."""
        return [
            var for var, src in self._variable_map.items()
            if src == source
        ]

    def get_intraday_variables(self) -> List[str]:
        """Get variables that support intraday updates."""
        return [
            var for var, cfg in self._variable_config.items()
            if cfg.get('intraday', False)
        ]

    def get_non_intraday_variables(self) -> List[str]:
        """Get variables that do NOT support intraday updates (daily/monthly/quarterly)."""
        return [
            var for var, cfg in self._variable_config.items()
            if not cfg.get('intraday', False)
        ]

    def get_variables_by_frequency(self, frequency: str) -> List[str]:
        """Get variables by frequency (D, M, Q)."""
        return [
            var for var, cfg in self._variable_config.items()
            if cfg.get('frequency', 'D') == frequency
        ]
