# -*- coding: utf-8 -*-
"""
BaseExtractor ABC - Common interface for all data extractors.

All extractors must implement:
- source_name: Unique identifier for the source
- variables: List of variables this extractor provides
- extract(): Fetch data for a variable/date range
- get_latest_date(): Get most recent available date
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Standardized extraction result (DRY)."""

    source: str
    variable: str
    data: pd.DataFrame
    last_date: Optional[datetime] = None
    records_count: int = 0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.data is not None and not self.data.empty:
            self.records_count = len(self.data)
            if 'fecha' in self.data.columns:
                self.last_date = pd.to_datetime(self.data['fecha']).max()
            elif self.data.index.name == 'fecha':
                self.last_date = self.data.index.max()


class BaseExtractor(ABC):
    """
    Abstract base class for all data extractors (DRY + SOLID).

    Each extractor is responsible for a single data source and
    provides a consistent interface for:
    - Full extraction (backfill)
    - Last N records extraction (realtime UPSERT)
    - Latest date check (gap detection)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize extractor with optional config.

        Args:
            config: Source-specific configuration from config.yaml
        """
        self.config = config or {}
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay_seconds', 5)
        self.timeout = self.config.get('timeout_seconds', 30)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique name for this data source."""
        pass

    @property
    @abstractmethod
    def variables(self) -> List[str]:
        """List of variable names this extractor provides."""
        pass

    @abstractmethod
    def extract(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        last_n: Optional[int] = None
    ) -> ExtractionResult:
        """
        Extract data for a variable within date range.

        Args:
            variable: Variable name to extract
            start_date: Start of date range
            end_date: End of date range
            last_n: If provided, return only last N records

        Returns:
            ExtractionResult with data and metadata
        """
        pass

    @abstractmethod
    def get_latest_date(self, variable: str) -> Optional[datetime]:
        """
        Get the most recent date available for a variable.

        Used for gap detection and incremental updates.
        """
        pass

    def extract_last_n(self, variable: str, n: int = 5) -> ExtractionResult:
        """
        Extract last N records for realtime UPSERT (DRY).

        This method provides a consistent way to get recent records
        for UPSERT operations without fetching full history.

        Args:
            variable: Variable name to extract
            n: Number of recent records to fetch

        Returns:
            ExtractionResult with last N records
        """
        end_date = datetime.now()

        # Determine lookback buffer based on variable frequency
        # Monthly variables need ~60 days buffer (2 months)
        # Quarterly variables need ~120 days buffer (4 months)
        # Daily variables need n * 3 days buffer
        freq = self._get_frequency(variable)

        if freq == 'Q':
            # Quarterly: go back n quarters + buffer
            start_date = end_date - timedelta(days=n * 120)
        elif freq == 'M':
            # Monthly: go back n months + buffer
            start_date = end_date - timedelta(days=n * 45)
        else:
            # Daily: use 2x buffer for business days
            start_date = end_date - timedelta(days=n * 3)

        result = self.extract(variable, start_date, end_date, last_n=n)

        if result.success and result.data is not None and len(result.data) > n:
            result.data = result.data.tail(n)
            result.records_count = len(result.data)

        return result

    def _get_frequency(self, variable: str) -> str:
        """
        Infer frequency from variable name or config.

        Returns 'D' (daily), 'M' (monthly), or 'Q' (quarterly).
        """
        # Try to get from config if available
        if hasattr(self, '_variables_config') and variable in self._variables_config:
            cfg = self._variables_config.get(variable, {})
            freq = cfg.get('frequency', 'D')
            if freq in ('D', 'M', 'Q'):
                return freq

        # Infer from variable name
        var_lower = variable.lower()
        if '_q_' in var_lower or var_lower.endswith('_q'):
            return 'Q'
        elif '_m_' in var_lower or var_lower.endswith('_m'):
            return 'M'
        return 'D'

    def validate_variable(self, variable: str) -> bool:
        """Check if this extractor handles the given variable."""
        return variable in self.variables

    def _with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic (DRY)."""
        import time

        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(
                    "[%s] Attempt %d/%d failed: %s",
                    self.source_name, attempt, self.retry_attempts, e
                )
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay * attempt)

        raise last_error
