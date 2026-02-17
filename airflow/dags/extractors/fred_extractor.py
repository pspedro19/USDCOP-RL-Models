# -*- coding: utf-8 -*-
"""
FRED Extractor - Federal Reserve Economic Data API.

Extracts:
- DGS2 (2-Year Treasury Yield)
- DPRIME (Prime Rate)
- FEDFUNDS (Fed Funds Rate)
- CPIAUCSL (CPI)
- PCEPI (PCE)
- UNRATE (Unemployment)
- INDPRO (Industrial Production)
- M2SL (M2 Money Supply)
- UMCSENT (Consumer Sentiment)

API: https://api.stlouisfed.org/fred/series/observations
Note: FRED API has T+1/T+2 publication delay (normal behavior)
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import requests

# Load .env file from project root
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    pass  # dotenv not installed, rely on environment variables

from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class FredExtractor(BaseExtractor):
    """FRED API Extractor (DRY implementation)."""

    API_BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._api_key = os.environ.get(
            config.get('api_key_env', 'FRED_API_KEY'),
            config.get('api_key', '')
        )
        self._variables_config = {
            v['name']: v for v in config.get('variables', [])
        }

    @property
    def source_name(self) -> str:
        return "fred"

    @property
    def variables(self) -> List[str]:
        return list(self._variables_config.keys())

    def _get_series_id(self, variable: str) -> str:
        """Get FRED series ID for a variable."""
        cfg = self._variables_config.get(variable, {})
        return cfg.get('series_id', variable.split('_')[-1])

    def extract(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        last_n: Optional[int] = None
    ) -> ExtractionResult:
        """Extract data from FRED API."""

        if not self._api_key:
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error="FRED API key not configured"
            )

        series_id = self._get_series_id(variable)

        try:
            df = self._with_retry(
                self._fetch_series,
                series_id,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

            if df.empty:
                return ExtractionResult(
                    source=self.source_name,
                    variable=variable,
                    data=df,
                    success=False,
                    error=f"No data returned for {series_id}"
                )

            # Rename value column to variable name
            df = df.rename(columns={'value': variable})

            if last_n and len(df) > last_n:
                df = df.tail(last_n)

            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=df,
                success=True,
                metadata={'series_id': series_id}
            )

        except Exception as e:
            logger.error("[FRED] Failed to extract %s: %s", variable, e)
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error=str(e)
            )

    def _fetch_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch series data from FRED API."""
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        }

        resp = requests.get(self.API_BASE, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        if not observations:
            return pd.DataFrame()

        records = []
        for obs in observations:
            date_str = obs.get("date")
            value_str = obs.get("value", ".")

            if value_str == "." or value_str == "":
                continue

            try:
                records.append({
                    "fecha": pd.to_datetime(date_str),
                    "value": float(value_str)
                })
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("fecha").reset_index(drop=True)

        return df

    def get_latest_date(self, variable: str) -> Optional[datetime]:
        """Get latest available date for variable."""
        series_id = self._get_series_id(variable)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            df = self._fetch_series(series_id, start_date, end_date)
            if not df.empty:
                return df['fecha'].max()
        except Exception as e:
            logger.warning("[FRED] Failed to get latest date for %s: %s", variable, e)

        return None
