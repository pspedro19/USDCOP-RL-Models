"""
Macro Data Extractor Interface
==============================

Defines the abstract interface for all macro data extraction strategies.
Implements the Strategy Pattern for interchangeable extraction implementations.

Contract: CTR-L0-INTERFACE-001

Pattern: Strategy Pattern
    - Each data source implements MacroExtractionStrategy
    - Extractors are created via MacroExtractorFactory
    - All extractors return data in the same format

Version: 1.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ExtractionResult:
    """
    Result of a single macro data extraction operation.

    Attributes:
        data: Dictionary mapping date_str -> {column_name -> value}
        errors: List of error messages encountered
        records_extracted: Number of unique dates extracted
        source_name: Name of the data source
        extraction_time: When extraction was performed
        duration_seconds: How long extraction took
    """
    data: Dict[str, Dict[str, float]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    records_extracted: int = 0
    source_name: str = ""
    extraction_time: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0

    @property
    def is_successful(self) -> bool:
        """True if some data was extracted."""
        return self.records_extracted > 0

    @property
    def has_errors(self) -> bool:
        """True if any errors occurred."""
        return len(self.errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XCom serialization."""
        return {
            'data': self.data,
            'errors': self.errors,
            'records_extracted': self.records_extracted,
            'source_name': self.source_name,
            'extraction_time': self.extraction_time.isoformat(),
            'duration_seconds': self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create from dictionary (XCom deserialization)."""
        return cls(
            data=data.get('data', {}),
            errors=data.get('errors', []),
            records_extracted=data.get('records_extracted', 0),
            source_name=data.get('source_name', ''),
            extraction_time=datetime.fromisoformat(data.get('extraction_time', datetime.utcnow().isoformat())),
            duration_seconds=data.get('duration_seconds', 0.0),
        )


@runtime_checkable
class MacroExtractionStrategy(Protocol):
    """
    Protocol for macro data extraction strategies.

    All data source extractors must implement this interface.
    This enables the Strategy Pattern for interchangeable extractors.
    """

    @property
    def source_name(self) -> str:
        """Unique identifier for this data source."""
        ...

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract macro data for given indicators.

        Args:
            indicators: Mapping of source_id -> column_name
                        (e.g., {"DTWEXBGS": "fxrt_index_dxy_usa_d_dxy"})
            lookback_days: Days of historical data to fetch
            **kwargs: Source-specific options

        Returns:
            ExtractionResult containing date -> column -> value mapping
        """
        ...


class BaseMacroExtractor(ABC):
    """
    Abstract base class for macro data extractors.

    Provides common functionality for all extraction strategies.
    Subclasses must implement extract() method.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize extractor with optional configuration.

        Args:
            config: Source-specific configuration dictionary
        """
        self.config = config or {}

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this data source."""
        pass

    @abstractmethod
    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract macro data for given indicators.

        Args:
            indicators: Mapping of source_id -> column_name
            lookback_days: Days of historical data to fetch
            **kwargs: Source-specific options

        Returns:
            ExtractionResult with extracted data
        """
        pass

    def _normalize_date(self, date_value: Any) -> Optional[str]:
        """
        Normalize a date value to ISO format string.

        Delegates to the unified DateParser.

        Args:
            date_value: Date in any supported format

        Returns:
            ISO date string (YYYY-MM-DD) or None
        """
        # Import here to avoid circular imports
        try:
            from utils.date_parser import DateParser
            return DateParser.parse(date_value)
        except ImportError:
            # Fallback: try pandas for date parsing
            import pandas as pd
            try:
                parsed = pd.to_datetime(date_value, errors='coerce')
                if pd.notna(parsed):
                    return parsed.strftime('%Y-%m-%d')
            except Exception:
                pass
            return None

    def _create_result(
        self,
        data: Dict[str, Dict[str, float]],
        errors: List[str],
        start_time: datetime
    ) -> ExtractionResult:
        """
        Create a standardized ExtractionResult.

        Args:
            data: Extracted data dictionary
            errors: List of error messages
            start_time: When extraction started

        Returns:
            ExtractionResult instance
        """
        end_time = datetime.utcnow()
        return ExtractionResult(
            data=data,
            errors=errors,
            records_extracted=len(data),
            source_name=self.source_name,
            extraction_time=start_time,
            duration_seconds=(end_time - start_time).total_seconds(),
        )

    def validate_indicators(self, indicators: Dict[str, str]) -> List[str]:
        """
        Validate indicator configuration.

        Args:
            indicators: Indicator mapping to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        if not indicators:
            errors.append(f"[{self.source_name}] No indicators configured")
        return errors


class ConfigurableExtractor(BaseMacroExtractor):
    """
    Base class for extractors that load config from YAML.

    Provides helper methods for accessing configuration values.
    """

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with optional default.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def get_api_key(self) -> Optional[str]:
        """
        Get API key from environment or Vault.

        Returns:
            API key string or None
        """
        import os

        # Try environment variable first
        env_key = self.config.get('api_key_env')
        if env_key:
            value = os.environ.get(env_key)
            if value:
                return value

        # Try Vault
        vault_path = self.config.get('api_key_vault_path')
        if vault_path:
            try:
                from src.shared.secrets.vault_client import get_vault_client
                vault = get_vault_client()
                return vault.get_secret(vault_path, 'api_key')
            except Exception:
                pass

        return None

    def get_timeout(self) -> int:
        """Get request timeout in seconds."""
        return self.config.get('request_timeout_seconds', 30)

    def get_retry_config(self) -> tuple:
        """Get retry configuration."""
        return (
            self.config.get('retry_attempts', 3),
            self.config.get('retry_delay_seconds', 60),
        )

    def validate_value(self, column: str, value: float) -> tuple:
        """
        Validate that a value is within expected range.

        Uses ranges defined in config.validation.ranges (from l0_macro_sources.yaml).

        Args:
            column: Column name to validate
            value: Value to check

        Returns:
            Tuple of (is_valid: bool, warning_message: str)
        """
        import logging
        logger = logging.getLogger(__name__)

        validation_config = self.config.get('validation', {})
        if not validation_config.get('enabled', True):
            return True, ""

        ranges = validation_config.get('ranges', {})

        if column in ranges:
            range_spec = ranges[column]
            if isinstance(range_spec, list) and len(range_spec) == 2:
                min_val, max_val = range_spec
                if not (min_val <= value <= max_val):
                    warning = f"[VALIDATION] {column}={value} outside range [{min_val}, {max_val}]"
                    logger.warning(warning)

                    on_invalid = validation_config.get('on_invalid', 'warn')
                    if on_invalid == 'skip':
                        return False, warning
                    elif on_invalid == 'error':
                        raise ValueError(warning)
                    # 'warn' - log but continue
                    return True, warning

        return True, ""

    def get_validation_ranges(self) -> Dict[str, List[float]]:
        """
        Get all validation ranges from config.

        Returns:
            Dictionary mapping column name to [min, max] range
        """
        return self.config.get('validation', {}).get('ranges', {})
