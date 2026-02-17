"""
Macro Variables SSOT Loader
============================

Singleton loader for the Macro Variables Single Source of Truth (SSOT).
All systems that need macro variable metadata should read from this loader.

Contract: CTR-L0-SSOT-001

Usage:
    from src.data.macro_ssot import MacroSSOT

    ssot = MacroSSOT()

    # Get a specific variable
    var = ssot.get_variable('infl_cpi_all_usa_m_cpiaucsl')

    # Get all variables
    all_vars = ssot.get_all_variables()

    # Get variables by source
    fred_vars = ssot.get_variables_by_source('fred')

    # Get extraction config
    config = ssot.get_extraction_config('infl_cpi_all_usa_m_cpiaucsl', 'fred')

Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class VariableIdentity:
    """Identity information for a macro variable."""
    canonical_name: str
    display_name: str
    category: str
    country: str
    frequency: str  # 'daily', 'monthly', 'quarterly'


@dataclass
class ExtractionConfig:
    """Extraction configuration for a specific source."""
    primary_source: str
    fallback_source: Optional[str] = None
    intraday: bool = False
    source_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PublicationSchedule:
    """Publication schedule for anti-leakage."""
    delay_days: int = 0
    day_range: Optional[Tuple[int, int]] = None
    typical_day: Optional[int] = None
    time: Optional[str] = None
    timezone: str = "US/Eastern"
    month_lag: int = 1
    quarter_lag: Optional[int] = None
    days_after_quarter: Optional[int] = None
    simultaneous_with: Optional[str] = None


@dataclass
class ValidationConfig:
    """Validation configuration."""
    expected_range: Tuple[float, float]
    leakage_risk: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    priority: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'


@dataclass
class FFillConfig:
    """Forward-fill configuration."""
    max_days: int


@dataclass
class MacroVariableDef:
    """Complete definition of a macro variable."""
    identity: VariableIdentity
    extraction: ExtractionConfig
    schedule: PublicationSchedule
    validation: ValidationConfig
    ffill: FFillConfig

    @property
    def canonical_name(self) -> str:
        return self.identity.canonical_name

    @property
    def display_name(self) -> str:
        return self.identity.display_name

    @property
    def frequency(self) -> str:
        return self.identity.frequency

    @property
    def primary_source(self) -> str:
        return self.extraction.primary_source

    @property
    def is_intraday(self) -> bool:
        return self.extraction.intraday

    @property
    def max_ffill_days(self) -> int:
        return self.ffill.max_days

    @property
    def leakage_risk(self) -> str:
        return self.validation.leakage_risk


class MacroSSOT:
    """
    Singleton loader for Macro Variables SSOT.

    This is the SINGLE SOURCE OF TRUTH for all macro variable definitions.
    All other systems should read from this loader, not from separate configs.
    """

    _instance: Optional['MacroSSOT'] = None
    _initialized: bool = False

    def __new__(cls, config_path: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None):
        if self._initialized:
            return

        self._variables: Dict[str, MacroVariableDef] = {}
        self._raw_config: Dict[str, Any] = {}
        self._groups: Dict[str, List[str]] = {}

        self._load_config(config_path)
        self._initialized = True

    def _find_config_path(self, config_path: Optional[Path] = None) -> Path:
        """Find the SSOT config file."""
        if config_path and config_path.exists():
            return config_path

        possible_paths = [
            Path(__file__).parent.parent.parent / "config" / "macro_variables_ssot.yaml",
            Path("/opt/airflow/config/macro_variables_ssot.yaml"),
            Path("config/macro_variables_ssot.yaml"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"macro_variables_ssot.yaml not found in: {possible_paths}"
        )

    def _load_config(self, config_path: Optional[Path] = None):
        """Load and parse the SSOT YAML file."""
        path = self._find_config_path(config_path)
        logger.info(f"Loading Macro SSOT from: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)

        # Parse variables
        variables_raw = self._raw_config.get('variables', {})
        for var_name, var_config in variables_raw.items():
            self._variables[var_name] = self._parse_variable(var_name, var_config)

        # Parse groups
        self._groups = self._raw_config.get('variable_groups', {})

        logger.info(f"Loaded {len(self._variables)} variables from SSOT")

    def _parse_variable(self, name: str, config: Dict) -> MacroVariableDef:
        """Parse a variable configuration into a MacroVariableDef."""
        # Identity
        identity_raw = config.get('identity', {})
        identity = VariableIdentity(
            canonical_name=identity_raw.get('canonical_name', name),
            display_name=identity_raw.get('display_name', name),
            category=identity_raw.get('category', 'unknown'),
            country=identity_raw.get('country', 'unknown'),
            frequency=identity_raw.get('frequency', 'daily'),
        )

        # Extraction
        extract_raw = config.get('extraction', {})
        source_configs = {}
        for key, value in extract_raw.items():
            if isinstance(value, dict):
                source_configs[key] = value

        extraction = ExtractionConfig(
            primary_source=extract_raw.get('primary_source', 'unknown'),
            fallback_source=extract_raw.get('fallback_source'),
            intraday=extract_raw.get('intraday', False),
            source_configs=source_configs,
        )

        # Schedule
        sched_raw = config.get('schedule', {}).get('publication', {})
        day_range = sched_raw.get('day_range')
        if day_range:
            day_range = tuple(day_range)

        schedule = PublicationSchedule(
            delay_days=sched_raw.get('delay_days', 0),
            day_range=day_range,
            typical_day=sched_raw.get('typical_day'),
            time=sched_raw.get('time'),
            timezone=sched_raw.get('timezone', 'US/Eastern'),
            month_lag=sched_raw.get('month_lag', 1),
            quarter_lag=sched_raw.get('quarter_lag'),
            days_after_quarter=sched_raw.get('days_after_quarter'),
            simultaneous_with=sched_raw.get('simultaneous_with'),
        )

        # Validation
        valid_raw = config.get('validation', {})
        expected_range = valid_raw.get('expected_range', [0, 1000])
        if isinstance(expected_range, list):
            expected_range = tuple(expected_range)

        validation = ValidationConfig(
            expected_range=expected_range,
            leakage_risk=valid_raw.get('leakage_risk', 'UNKNOWN'),
            priority=valid_raw.get('priority', 'MEDIUM'),
        )

        # FFill
        ffill_raw = config.get('ffill', {})
        ffill = FFillConfig(
            max_days=ffill_raw.get('max_days', 5),
        )

        return MacroVariableDef(
            identity=identity,
            extraction=extraction,
            schedule=schedule,
            validation=validation,
            ffill=ffill,
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_variable(self, name: str) -> Optional[MacroVariableDef]:
        """Get a variable definition by canonical name."""
        return self._variables.get(name)

    @lru_cache(maxsize=1)
    def get_all_variables(self) -> List[str]:
        """Get list of all variable names."""
        return list(self._variables.keys())

    def get_all_variable_defs(self) -> Dict[str, MacroVariableDef]:
        """Get all variable definitions."""
        return self._variables.copy()

    def get_variables_by_source(self, source: str) -> List[str]:
        """Get all variables that use a specific primary source."""
        return [
            name for name, var in self._variables.items()
            if var.extraction.primary_source == source
        ]

    def get_variables_by_frequency(self, frequency: str) -> List[str]:
        """Get all variables by frequency (daily, monthly, quarterly)."""
        return [
            name for name, var in self._variables.items()
            if var.identity.frequency == frequency
        ]

    def get_intraday_variables(self) -> List[str]:
        """Get all variables that support intraday updates."""
        return [
            name for name, var in self._variables.items()
            if var.extraction.intraday
        ]

    def get_non_intraday_variables(self) -> List[str]:
        """Get all variables that do NOT support intraday updates."""
        return [
            name for name, var in self._variables.items()
            if not var.extraction.intraday
        ]

    def get_variables_by_group(self, group_name: str) -> List[str]:
        """Get variables in a named group."""
        return self._groups.get(group_name, [])

    def get_extraction_config(
        self,
        variable: str,
        source: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get extraction configuration for a variable.

        Args:
            variable: Variable canonical name
            source: Specific source (default: primary source)

        Returns:
            Extraction config dict or None
        """
        var = self._variables.get(variable)
        if not var:
            return None

        if source is None:
            source = var.extraction.primary_source

        return var.extraction.source_configs.get(source)

    def get_publication_config(self, variable: str) -> Optional[Dict[str, Any]]:
        """Get publication schedule config for a variable."""
        var = self._variables.get(variable)
        if not var:
            return None

        sched = var.schedule
        return {
            'delay_days': sched.delay_days,
            'day_range': list(sched.day_range) if sched.day_range else None,
            'typical_day': sched.typical_day,
            'time': sched.time,
            'timezone': sched.timezone,
            'month_lag': sched.month_lag,
            'quarter_lag': sched.quarter_lag,
            'days_after_quarter': sched.days_after_quarter,
        }

    def get_max_ffill_days(self, variable: str) -> int:
        """Get max forward-fill days for a variable."""
        var = self._variables.get(variable)
        if not var:
            # Default by frequency
            return 5
        return var.ffill.max_days

    def get_leakage_risk(self, variable: str) -> str:
        """Get leakage risk level for a variable."""
        var = self._variables.get(variable)
        if not var:
            return 'UNKNOWN'
        return var.validation.leakage_risk

    def get_expected_range(self, variable: str) -> Optional[Tuple[float, float]]:
        """Get expected value range for validation."""
        var = self._variables.get(variable)
        if not var:
            return None
        return var.validation.expected_range

    # =========================================================================
    # GLOBAL CONFIG ACCESS
    # =========================================================================

    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration settings."""
        return self._raw_config.get('global', {})

    def get_source_config(self, source: str) -> Dict[str, Any]:
        """Get source-level configuration."""
        sources = self._raw_config.get('sources', {})
        return sources.get(source, {})

    def get_ffill_limits(self, target_frequency: str = 'daily') -> Dict[str, int]:
        """Get FFILL limits for a target frequency."""
        global_config = self.get_global_config()
        ffill_limits = global_config.get('ffill_limits', {})
        return ffill_limits.get(target_frequency, {})

    # =========================================================================
    # CONVENIENCE METHODS FOR ADAPTERS
    # =========================================================================

    def get_fred_variables(self) -> Dict[str, str]:
        """Get FRED series_id -> column mapping for FRED extractor."""
        result = {}
        for name, var in self._variables.items():
            fred_config = var.extraction.source_configs.get('fred', {})
            series_id = fred_config.get('series_id')
            if series_id:
                result[series_id] = name
        return result

    def get_suameca_variables(self) -> Dict[int, str]:
        """Get SUAMECA serie_id -> column mapping."""
        result = {}
        for name, var in self._variables.items():
            suameca_config = var.extraction.source_configs.get('suameca', {})
            serie_id = suameca_config.get('serie_id')
            if serie_id:
                result[serie_id] = name
        return result

    def get_investing_variables(self) -> List[Dict[str, Any]]:
        """Get Investing.com variable configs."""
        result = []
        for name, var in self._variables.items():
            inv_config = var.extraction.source_configs.get('investing', {})
            if inv_config:
                result.append({
                    'column': name,
                    'name': var.display_name,
                    'method': inv_config.get('method', 'ajax'),
                    'pair_id': inv_config.get('pair_id'),
                    'instrument_id': inv_config.get('instrument_id'),
                    'url': inv_config.get('url'),
                    'domain_id': inv_config.get('domain_id', 'www'),
                    'expected_range': list(var.validation.expected_range),
                })
        return result

    def get_bcrp_variables(self) -> Dict[str, str]:
        """Get BCRP serie_code -> column mapping."""
        result = {}
        for name, var in self._variables.items():
            bcrp_config = var.extraction.source_configs.get('bcrp', {})
            serie_code = bcrp_config.get('serie_code')
            if serie_code:
                result[serie_code] = name
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_ssot(config_path: Optional[Path] = None) -> MacroSSOT:
    """Get the MacroSSOT singleton instance."""
    return MacroSSOT(config_path)


def get_variable(name: str) -> Optional[MacroVariableDef]:
    """Convenience function to get a variable definition."""
    return MacroSSOT().get_variable(name)


def get_max_ffill_days(variable: str) -> int:
    """Convenience function to get max ffill days."""
    return MacroSSOT().get_max_ffill_days(variable)


# =============================================================================
# MAIN - Test
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("MACRO SSOT TEST")
    print("=" * 70)

    ssot = MacroSSOT()

    print(f"\nTotal variables: {len(ssot.get_all_variables())}")

    print("\nVariables by frequency:")
    for freq in ['daily', 'monthly', 'quarterly']:
        vars_list = ssot.get_variables_by_frequency(freq)
        print(f"  {freq}: {len(vars_list)}")

    print("\nVariables by primary source:")
    for source in ['fred', 'investing', 'suameca', 'bcrp', 'fedesarrollo', 'dane']:
        vars_list = ssot.get_variables_by_source(source)
        print(f"  {source}: {len(vars_list)}")

    print("\nIntraday variables:", len(ssot.get_intraday_variables()))

    print("\nSample variable (infl_cpi_all_usa_m_cpiaucsl):")
    var = ssot.get_variable('infl_cpi_all_usa_m_cpiaucsl')
    if var:
        print(f"  Display name: {var.display_name}")
        print(f"  Frequency: {var.frequency}")
        print(f"  Primary source: {var.primary_source}")
        print(f"  Max ffill days: {var.max_ffill_days}")
        print(f"  Leakage risk: {var.leakage_risk}")
        print(f"  Typical publication day: {var.schedule.typical_day}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)
