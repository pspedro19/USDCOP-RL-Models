# -*- coding: utf-8 -*-
"""
BCRP Extractor - Banco Central de Reserva del Peru (HTML Scraping).

Extracts:
- EMBI Colombia spread (PD04715XD)

URL: https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados/{series}/html
"""

import logging
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd
import requests

from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# Spanish month abbreviations mapping
SPANISH_MONTHS = {
    'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
}


class BcrpExtractor(BaseExtractor):
    """BCRP HTML Scraper for EMBI and other series."""

    HTML_BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._variables_config = {
            v['name']: v for v in config.get('variables', [])
        }

    @property
    def source_name(self) -> str:
        return "bcrp"

    @property
    def variables(self) -> List[str]:
        return list(self._variables_config.keys())

    def _get_serie_code(self, variable: str) -> str:
        """Get BCRP serie code for a variable."""
        cfg = self._variables_config.get(variable, {})
        return cfg.get('serie_code', '')

    def extract(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        last_n: Optional[int] = None
    ) -> ExtractionResult:
        """Extract data from BCRP HTML page."""

        serie_code = self._get_serie_code(variable)
        if not serie_code:
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error=f"No serie_code configured for {variable}"
            )

        try:
            df = self._with_retry(
                self._fetch_html_serie,
                serie_code
            )

            if df.empty:
                return ExtractionResult(
                    source=self.source_name,
                    variable=variable,
                    data=df,
                    success=False,
                    error=f"No data returned for serie {serie_code}"
                )

            # Filter by date range
            df = df[(df['fecha'] >= start_date) & (df['fecha'] <= end_date)]

            # Rename value column to variable name
            df = df.rename(columns={'value': variable})

            if last_n and len(df) > last_n:
                df = df.tail(last_n)

            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=df,
                success=True,
                metadata={'serie_code': serie_code, 'rows': len(df)}
            )

        except Exception as e:
            logger.error("[BCRP] Failed to extract %s: %s", variable, e)
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error=str(e)
            )

    def _parse_bcrp_date(self, date_str: str) -> Optional[datetime]:
        """Parse BCRP date format like '29Ene26' to datetime."""
        # Format: ddMmmyy (e.g., 29Ene26 = 29-Jan-2026)
        match = re.match(r'(\d{2})([A-Za-z]{3})(\d{2})', date_str.strip())
        if not match:
            return None

        day, month_abbr, year_short = match.groups()
        month = SPANISH_MONTHS.get(month_abbr.capitalize())
        if not month:
            return None

        # Convert 2-digit year to 4-digit
        year_int = int(year_short)
        year = 2000 + year_int if year_int < 50 else 1900 + year_int

        try:
            return datetime(year, int(month), int(day))
        except ValueError:
            return None

    def _fetch_html_serie(self, serie_code: str) -> pd.DataFrame:
        """Fetch serie data by scraping BCRP HTML page."""

        url = f"{self.HTML_BASE}/{serie_code}/html"
        logger.info("[BCRP] Fetching HTML: %s", url)

        resp = requests.get(url, headers=self.HEADERS, timeout=self.timeout)
        resp.raise_for_status()
        html = resp.text

        # Extract date-value pairs using regex
        # Pattern: <td class="periodo"><b>29Ene26</b></td> ... <td class="dato">263</td>
        pattern = r'<td class="periodo">\s*<b>(\d{2}[A-Za-z]{3}\d{2})</b>\s*</td>.*?<td class="dato">\s*(-?\d+(?:\.\d+)?)\s*</td>'

        matches = re.findall(pattern, html, re.DOTALL)

        if not matches:
            logger.warning("[BCRP] No data found in HTML for %s", serie_code)
            return pd.DataFrame()

        records = []
        for date_str, value_str in matches:
            dt = self._parse_bcrp_date(date_str)
            if dt is None:
                continue

            try:
                value = float(value_str)
                records.append({"fecha": dt, "value": value})
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.drop_duplicates(subset=['fecha'])
            df = df.sort_values("fecha").reset_index(drop=True)

        logger.info("[BCRP] Extracted %d records from HTML", len(df))
        return df

    def get_latest_date(self, variable: str) -> Optional[datetime]:
        """Get latest available date for variable."""
        serie_code = self._get_serie_code(variable)
        if not serie_code:
            return None

        try:
            df = self._fetch_html_serie(serie_code)
            if not df.empty:
                return df['fecha'].max()
        except Exception as e:
            logger.warning("[BCRP] Failed to get latest date for %s: %s", variable, e)

        return None
