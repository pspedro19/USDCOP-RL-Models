# -*- coding: utf-8 -*-
"""
SUAMECA Extractor - BanRep REST API.

Extracts via REST API:
- IBR Overnight (serie 241)
- TPM - Tasa de Politica Monetaria (serie 59)
- Prime Rate (serie 88) - Bloomberg source
- ITCR (serie 234) - Real Exchange Rate Index
- ITCR_USA (serie 219) - Bilateral Real Exchange Rate vs USA

API discovered via Angular SPA inspection:
    GET https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/
        estadisticaEconomicaRestService/consultaInformacionSerie?idSerie=XXX

Note: Graficador method (Selenium) is kept as fallback but REST API works
for all current series.
"""

import logging
import re
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd
import requests

from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# Spanish month abbreviations
SPANISH_MONTHS = {
    'Ene': 1, 'Feb': 2, 'Mar': 3, 'Abr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Ago': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dic': 12
}


class SuamecaExtractor(BaseExtractor):
    """SUAMECA/BanRep Extractor - REST API + Graficador (Selenium)."""

    API_BASE = (
        "https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/"
        "estadisticaEconomicaRestService"
    )

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
    }

    # Series that require Selenium/estadisticas-economicas method (REST API returns empty)
    # These series don't return data from the REST API and require browser scraping
    GRAFICADOR_SERIES = {
        4180,    # Terms of Trade (ftrd_terms_trade_col_m_tot)
        100002,  # CPI Colombia (infl_cpi_total_col_m_ipccol)
        414001,  # Current Account (rsbp_current_account_col_q_cacct)
    }

    # URL patterns for estadisticas-economicas pages
    ESTADISTICAS_URLS = {
        4180: 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4180/indice_terminos_intercambio_bienes',
        100002: 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/100002/ipc',
        414001: 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/414001/balanza_pagos_cuenta_corriente',
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._variables_config = {
            v['name']: v for v in config.get('variables', [])
        }

    @property
    def source_name(self) -> str:
        return "suameca"

    @property
    def variables(self) -> List[str]:
        return list(self._variables_config.keys())

    def _get_serie_id(self, variable: str) -> int:
        """Get SUAMECA serie ID for a variable."""
        cfg = self._variables_config.get(variable, {})
        return cfg.get('serie_id', 0)

    def _get_method(self, variable: str) -> str:
        """Get extraction method for a variable."""
        cfg = self._variables_config.get(variable, {})
        return cfg.get('method', 'rest_api')

    def _get_url(self, variable: str) -> str:
        """Get URL for graficador method."""
        cfg = self._variables_config.get(variable, {})
        return cfg.get('url', '')

    def extract(
        self,
        variable: str,
        start_date: datetime,
        end_date: datetime,
        last_n: Optional[int] = None
    ) -> ExtractionResult:
        """Extract data from SUAMECA (REST API or Graficador)."""

        serie_id = self._get_serie_id(variable)
        method = self._get_method(variable)

        if not serie_id:
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error=f"No serie_id configured for {variable}"
            )

        try:
            # Use Selenium method for specific series (graficador/selenium or in GRAFICADOR_SERIES)
            if method in ('graficador', 'selenium') or serie_id in self.GRAFICADOR_SERIES:
                url = self._get_url(variable)
                df = self._fetch_graficador(serie_id, url, start_date, end_date)
            else:
                df = self._with_retry(
                    self._fetch_serie,
                    serie_id,
                    start_date,
                    end_date
                )

            if df.empty:
                return ExtractionResult(
                    source=self.source_name,
                    variable=variable,
                    data=df,
                    success=False,
                    error=f"No data returned for serie {serie_id}"
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
                metadata={'serie_id': serie_id, 'method': method}
            )

        except Exception as e:
            logger.error("[SUAMECA] Failed to extract %s: %s", variable, e)
            return ExtractionResult(
                source=self.source_name,
                variable=variable,
                data=pd.DataFrame(),
                success=False,
                error=str(e)
            )

    def _fetch_graficador(
        self,
        serie_id: int,
        url: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch data using Selenium to scrape estadisticas-economicas pages.

        Uses undetected_chromedriver to bypass Radware Bot Manager protection.
        The page loads an Angular SPA that requires clicking "Vista tabla" button
        to display the data table, then parsing the HTML table structure.
        """
        # Use estadisticas-economicas URL if available (preferred)
        if serie_id in self.ESTADISTICAS_URLS:
            url = self.ESTADISTICAS_URLS[serie_id]
        elif not url:
            url = f"https://suameca.banrep.gov.co/graficador-interactivo/grafica/{serie_id}"

        logger.info("[SUAMECA] Fetching via Selenium: serie=%d, url=%s", serie_id, url)

        try:
            import undetected_chromedriver as uc
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
        except ImportError:
            logger.error("[SUAMECA] Selenium/undetected_chromedriver not installed")
            return pd.DataFrame()

        driver = None
        try:
            options = uc.ChromeOptions()
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--lang=es-CO')

            driver = uc.Chrome(options=options, version_main=None)
            driver.get(url)

            # Wait for initial page load and Radware bypass
            time.sleep(10)

            # Check for captcha/Radware
            page_source = driver.page_source.lower()
            if 'captcha' in page_source or 'radware' in page_source:
                logger.warning("[SUAMECA] Radware detected, waiting for bypass...")
                time.sleep(15)

            # Click "Vista tabla" button by ID
            try:
                vista_tabla_btn = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.ID, "vistaTabla"))
                )
                vista_tabla_btn.click()
                logger.info("[SUAMECA] Clicked 'Vista tabla' button")
                time.sleep(5)  # Wait for table to render
            except Exception as e:
                logger.warning("[SUAMECA] Could not click Vista tabla: %s", e)
                # Try alternative - look for table icon
                try:
                    icons = driver.find_elements(By.TAG_NAME, "mat-icon")
                    for icon in icons:
                        if any(x in icon.text.lower() for x in ['table', 'grid', 'list']):
                            icon.find_element(By.XPATH, '..').click()
                            time.sleep(3)
                            break
                except Exception:
                    pass

            # Find all tables and select the one with most rows
            tables = driver.find_elements(By.TAG_NAME, "table")
            logger.info("[SUAMECA] Found %d tables in page", len(tables))

            data_table = None
            max_rows = 0
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                if len(rows) > max_rows:
                    max_rows = len(rows)
                    data_table = table

            if not data_table or max_rows < 5:
                logger.warning("[SUAMECA] No data table found for serie %d", serie_id)
                return pd.DataFrame()

            logger.info("[SUAMECA] Processing table with %d rows", max_rows)

            # Parse table: SUAMECA uses <th> for dates and <td> for values
            # Structure: <tr><th>YYYY/MM/DD</th><td>value</td></tr>
            records = []
            rows = data_table.find_elements(By.TAG_NAME, "tr")

            for row in rows[1:]:  # Skip header row
                try:
                    th_elements = row.find_elements(By.TAG_NAME, "th")
                    td_elements = row.find_elements(By.TAG_NAME, "td")

                    if not th_elements or not td_elements:
                        continue

                    fecha_str = th_elements[0].text.strip()
                    valor_str = td_elements[0].text.strip()

                    # Clean value string
                    valor_str = (valor_str.replace(',', '.')
                                          .replace('%', '')
                                          .replace(' ', '')
                                          .replace('\n', ''))

                    if not fecha_str or not valor_str:
                        continue

                    # Parse value
                    try:
                        valor = float(valor_str)
                    except ValueError:
                        continue

                    # Parse date - try multiple formats
                    dt = None
                    for fmt in ['%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']:
                        try:
                            dt = datetime.strptime(fecha_str, fmt)
                            break
                        except ValueError:
                            continue

                    # Fallback to pandas parser
                    if dt is None:
                        try:
                            dt = pd.to_datetime(fecha_str)
                        except Exception:
                            continue

                    if dt is None:
                        continue

                    # Filter by date range
                    if dt.date() < start_date.date() or dt.date() > end_date.date():
                        continue

                    records.append({"fecha": dt, "value": valor})

                except Exception:
                    continue

            df = pd.DataFrame(records)
            if not df.empty:
                df['fecha'] = pd.to_datetime(df['fecha']).dt.normalize()
                df = df.drop_duplicates(subset=['fecha'])
                df = df.sort_values("fecha").reset_index(drop=True)

            logger.info("[SUAMECA] Selenium extracted %d records for serie %d", len(df), serie_id)
            return df

        except Exception as e:
            logger.error("[SUAMECA] Selenium failed for serie %d: %s", serie_id, e)
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def _parse_spanish_date(self, date_str: str) -> Optional[datetime]:
        """Parse Spanish date formats like '01-Ene-2020' or '01/Ene/20'."""
        date_str = date_str.strip()

        # Try patterns
        patterns = [
            r'(\d{1,2})[-/]([A-Za-z]{3})[-/](\d{4})',  # 01-Ene-2020
            r'(\d{1,2})[-/]([A-Za-z]{3})[-/](\d{2})',  # 01-Ene-20
        ]

        for pattern in patterns:
            match = re.match(pattern, date_str, re.IGNORECASE)
            if match:
                day, month_str, year = match.groups()
                month = SPANISH_MONTHS.get(month_str.capitalize())
                if not month:
                    continue

                year_int = int(year)
                if year_int < 100:
                    year_int = 2000 + year_int if year_int < 50 else 1900 + year_int

                try:
                    return datetime(year_int, month, int(day))
                except ValueError:
                    continue

        return None

    def _fetch_serie(
        self,
        serie_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch serie data from SUAMECA REST API."""

        url = f"{self.API_BASE}/consultaInformacionSerie"
        params = {"idSerie": serie_id}

        resp = requests.get(
            url,
            params=params,
            headers=self.HEADERS,
            timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()

        # API returns a list with one element containing 'data' array
        # Some series return empty list when not available
        if isinstance(data, list):
            if len(data) == 0:
                logger.warning("[SUAMECA] Serie %d returned empty list", serie_id)
                return pd.DataFrame()
            first_elem = data[0]
            if isinstance(first_elem, dict):
                serie_data = first_elem.get('data', [])
            else:
                logger.warning("[SUAMECA] Serie %d: unexpected structure", serie_id)
                return pd.DataFrame()
        elif isinstance(data, dict):
            serie_data = data.get('data', [])
        else:
            logger.warning("[SUAMECA] Serie %d: unexpected response type", serie_id)
            return pd.DataFrame()

        if not serie_data:
            return pd.DataFrame()

        records = []
        for item in serie_data:
            if isinstance(item, list) and len(item) >= 2:
                timestamp_ms = item[0]
                value = item[1]

                # Convert timestamp (ms) to datetime
                dt = pd.to_datetime(timestamp_ms, unit='ms')

                # Filter by date range
                if dt.date() < start_date.date() or dt.date() > end_date.date():
                    continue

                records.append({
                    "fecha": dt.normalize(),  # Remove time component
                    "value": float(value) if value is not None else None
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.dropna(subset=['value'])
            df = df.drop_duplicates(subset=['fecha'])
            df = df.sort_values("fecha").reset_index(drop=True)

        return df

    def get_latest_date(self, variable: str) -> Optional[datetime]:
        """Get latest available date for variable."""
        serie_id = self._get_serie_id(variable)
        if not serie_id:
            return None

        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = datetime(end_date.year - 1, 1, 1)
            df = self._fetch_serie(serie_id, start_date, end_date)
            if not df.empty:
                return df['fecha'].max()
        except Exception as e:
            logger.warning("[SUAMECA] Failed to get latest date for %s: %s", variable, e)

        return None
