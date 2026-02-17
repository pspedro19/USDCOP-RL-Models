# -*- coding: utf-8 -*-
"""
DANE Extractor - Colombian National Statistics Department.

Extracts:
- Exports (ftrd_exports_total_col_m_expusd)
- Imports (ftrd_imports_total_col_m_impusd)

Uses the existing scraper_dane_balanza.py module which downloads
Excel files from DANE's trade balance publications.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# Add scrapers path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCRAPERS_PATH = PROJECT_ROOT / 'data' / 'pipeline' / '02_scrapers' / '02_custom'


class DaneExtractor(BaseExtractor):
    """DANE Trade Balance Extractor."""

    # Variable mapping to scraper functions
    VARIABLE_FUNCTIONS = {
        'ftrd_exports_total_col_m_expusd': 'obtener_exportaciones',
        'ftrd_imports_total_col_m_impusd': 'obtener_importaciones',
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._variables_config = {
            v['name']: v for v in config.get('variables', [])
        }
        self._scraper_module = None

    @property
    def source_name(self) -> str:
        return "dane"

    @property
    def variables(self) -> List[str]:
        return list(self._variables_config.keys())

    def _load_scraper(self):
        """Lazy load the DANE scraper module."""
        if self._scraper_module is not None:
            return self._scraper_module

        try:
            import importlib.util
            scraper_path = SCRAPERS_PATH / 'scraper_dane_balanza.py'

            if not scraper_path.exists():
                logger.error("[DANE] Scraper module not found: %s", scraper_path)
                return None

            spec = importlib.util.spec_from_file_location("scraper_dane_balanza", scraper_path)
            self._scraper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._scraper_module)

            logger.info("[DANE] Scraper module loaded successfully")
            return self._scraper_module

        except Exception as e:
            logger.error("[DANE] Failed to load scraper module: %s", e)
            return None

    def extract(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        last_n: Optional[int] = None
    ) -> ExtractionResult:
        """Extract data from DANE via scraper."""

        func_name = self.VARIABLE_FUNCTIONS.get(variable)
        if not func_name:
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error=f"No function mapped for variable: {variable}"
            )

        scraper = self._load_scraper()
        if scraper is None:
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error="Could not load DANE scraper module"
            )

        try:
            # Get the function from the scraper module
            func = getattr(scraper, func_name, None)
            if func is None:
                return ExtractionResult(
                    source=self.source_name,
                    variable=variable,
                    data=pd.DataFrame(),
                    success=False,
                    error=f"Function {func_name} not found in scraper"
                )

            # Call the scraper function
            # For monthly data with publication delay, request more records
            request_n = last_n * 2 if last_n else 500
            df = func(n=request_n)

            if df is None or df.empty:
                return ExtractionResult(
                    source=self.source_name,
                    variable=variable,
                    data=pd.DataFrame(),
                    success=False,
                    error=f"Scraper returned empty data for {variable}"
                )

            # Standardize column names
            df['fecha'] = pd.to_datetime(df['fecha'])

            # Rename 'valor' to variable name
            if 'valor' in df.columns:
                df = df.rename(columns={'valor': variable})

            # Ensure we only have fecha and the variable column
            df = df[['fecha', variable]].copy()

            # For monthly data (DANE), don't filter by date range as it has
            # ~45 days publication delay. Just take the last N records.
            df = df.sort_values('fecha').reset_index(drop=True)

            if last_n and len(df) > last_n:
                df = df.tail(last_n).reset_index(drop=True)

            if df.empty:
                return ExtractionResult(
                    source=self.source_name,
                    variable=variable,
                    data=pd.DataFrame(),
                    success=False,
                    error=f"No data available for {variable}"
                )

            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=df,
                success=True,
                metadata={'function': func_name, 'latest_date': str(df['fecha'].max())[:10]}
            )

        except Exception as e:
            logger.error("[DANE] Failed to extract %s: %s", variable, e)
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error=str(e)
            )

    def get_latest_date(self, variable: str) -> Optional[datetime]:
        """Get latest available date for variable."""
        try:
            result = self.extract(
                variable,
                datetime(2020, 1, 1),
                datetime.now(),
                last_n=5
            )
            if result.success and result.data is not None and not result.data.empty:
                return result.data['fecha'].max()
        except Exception as e:
            logger.warning("[DANE] Failed to get latest date for %s: %s", variable, e)

        return None
