"""
Macro Data Extraction Strategies
================================

Implementation of extraction strategies for all macro data sources.
Uses the Strategy Pattern for interchangeable extractors.

Contract: CTR-L0-STRATEGY-001

Strategies:
    - FREDExtractionStrategy: US economic indicators from FRED API
    - TwelveDataExtractionStrategy: FX and commodities from TwelveData
    - InvestingExtractionStrategy: Market data from Investing.com scraping
    - BanRepExtractionStrategy: Colombian data from BanRep SUAMECA (Selenium)
    - BCRPExtractionStrategy: EMBI spread from Peru central bank
    - FedesarrolloExtractionStrategy: Colombian confidence indices
    - DANEExtractionStrategy: Colombian trade balance data

Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# Use relative imports within Airflow DAGs
import sys
import os
# Add src to path for core imports
sys.path.insert(0, '/opt/airflow')
sys.path.insert(0, '/opt/airflow/dags')

from src.core.interfaces.macro_extractor import (
    BaseMacroExtractor,
    ConfigurableExtractor,
    ExtractionResult,
)
from src.core.factories.macro_extractor_factory import (
    MacroExtractorFactory,
    MacroSource,
)
from utils.date_parser import DateParser

# Prometheus metrics
try:
    from services.common.prometheus_metrics import (
        record_macro_ingestion_success,
        record_macro_ingestion_error,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    def record_macro_ingestion_success(source: str, indicator: str) -> None:
        pass
    def record_macro_ingestion_error(source: str, indicator: str, error_type: str) -> None:
        pass

logger = logging.getLogger(__name__)


# =============================================================================
# FRED Extraction Strategy
# =============================================================================

@MacroExtractorFactory.register_strategy(MacroSource.FRED)
class FREDExtractionStrategy(ConfigurableExtractor):
    """
    FRED API extraction strategy for US economic indicators.

    Extracts data from Federal Reserve Economic Data API.
    Supports daily, monthly, and quarterly indicators.
    """

    @property
    def source_name(self) -> str:
        return "fred"

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """Extract data from FRED API."""
        start_time = datetime.utcnow()
        results: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        # Get API key
        api_key = self.get_api_key()
        if not api_key:
            errors.append("FRED_API_KEY not found in Vault or environment")
            return self._create_result(results, errors, start_time)

        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
        except ImportError:
            errors.append("fredapi package not installed")
            return self._create_result(results, errors, start_time)

        today = datetime.now().date()
        start_date = today - timedelta(days=lookback_days)

        for series_id, column in indicators.items():
            try:
                logger.info(f"[FRED] Fetching {series_id} -> {column}")
                data = fred.get_series(series_id, observation_start=start_date)

                if data is not None and not data.empty:
                    for date_idx, value in data.items():
                        if pd.notna(value):
                            date_str = self._normalize_date(date_idx)
                            if date_str:
                                if date_str not in results:
                                    results[date_str] = {}
                                results[date_str][column] = float(value)

                    logger.info(f"  -> {len(data)} records")
                    record_macro_ingestion_success('fred', column)
                else:
                    errors.append(f"{series_id}: No data")
                    record_macro_ingestion_error('fred', column, 'no_data')

            except Exception as e:
                errors.append(f"{series_id}: {str(e)}")
                logger.error(f"[FRED] Error {series_id}: {e}")
                record_macro_ingestion_error('fred', column, 'api_error')

        return self._create_result(results, errors, start_time)


# =============================================================================
# TwelveData Extraction Strategy
# =============================================================================

@MacroExtractorFactory.register_strategy(MacroSource.TWELVEDATA)
class TwelveDataExtractionStrategy(ConfigurableExtractor):
    """
    TwelveData API extraction strategy for FX and commodities.

    Extracts daily OHLCV data for currency pairs and commodities.
    """

    @property
    def source_name(self) -> str:
        return "twelvedata"

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """Extract data from TwelveData API."""
        start_time = datetime.utcnow()
        results: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        api_key = self.get_api_key()
        if not api_key:
            errors.append("TWELVEDATA_API_KEY not found")
            return self._create_result(results, errors, start_time)

        interval = self.config.get('interval', '1day')
        outputsize = self.config.get('outputsize', 60)
        timeout = self.get_timeout()

        for symbol, column in indicators.items():
            try:
                logger.info(f"[TwelveData] Fetching {symbol} -> {column}")

                url = "https://api.twelvedata.com/time_series"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'outputsize': outputsize,
                    'apikey': api_key
                }

                resp = requests.get(url, params=params, timeout=timeout)
                data = resp.json()

                if 'values' in data:
                    for item in data['values']:
                        date_str = self._normalize_date(item['datetime'])
                        if date_str:
                            value = float(item['close'])
                            if date_str not in results:
                                results[date_str] = {}
                            results[date_str][column] = value

                    logger.info(f"  -> {len(data['values'])} records")
                    record_macro_ingestion_success('twelvedata', column)
                else:
                    errors.append(f"{symbol}: {data.get('message', 'No data')}")
                    record_macro_ingestion_error('twelvedata', column, 'no_data')

                # Rate limit
                time.sleep(self.config.get('rate_limit_delay_seconds', 1))

            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
                logger.error(f"[TwelveData] Error {symbol}: {e}")
                record_macro_ingestion_error('twelvedata', column, 'api_error')

        return self._create_result(results, errors, start_time)


# =============================================================================
# Investing.com Extraction Strategy
# =============================================================================

@MacroExtractorFactory.register_strategy(MacroSource.INVESTING)
class InvestingExtractionStrategy(ConfigurableExtractor):
    """
    Investing.com scraping strategy for market indices and commodities.

    Uses cloudscraper to bypass Cloudflare protection.
    """

    @property
    def source_name(self) -> str:
        return "investing"

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """Extract data from Investing.com via scraping."""
        start_time = datetime.utcnow()
        results: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        try:
            import cloudscraper
            from bs4 import BeautifulSoup
        except ImportError:
            errors.append("cloudscraper/beautifulsoup4 not installed")
            return self._create_result(results, errors, start_time)

        scraper = cloudscraper.create_scraper()
        user_agent = self.config.get(
            'user_agent',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        headers = {'User-Agent': user_agent}
        max_rows = self.config.get('max_rows', 65)
        delay = self.config.get('request_delay_seconds', 2)

        # Get URLs config
        urls_config = self.config.get('urls', [])
        for url_entry in urls_config:
            url = url_entry.get('url')
            column = url_entry.get('column')
            name = url_entry.get('name', column)

            if not url or not column:
                continue

            try:
                logger.info(f"[Investing] Fetching {name} ({column})")

                resp = scraper.get(url, headers=headers, timeout=25)

                if resp.status_code != 200:
                    errors.append(f"{column}: HTTP {resp.status_code}")
                    record_macro_ingestion_error('investing', column, 'http_error')
                    continue

                soup = BeautifulSoup(resp.text, 'html.parser')

                # Find data table
                table = soup.find('table', class_='freeze-column-w-1')
                if not table:
                    tables = soup.find_all('table')
                    if tables:
                        table = max(tables, key=lambda t: len(str(t)))

                if not table:
                    errors.append(f"{column}: No table found")
                    record_macro_ingestion_error('investing', column, 'parse_error')
                    continue

                rows = table.find_all('tr')[1:max_rows]
                extracted = 0
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        try:
                            date_str = self._normalize_date(cols[0].get_text(strip=True))
                            value_text = cols[1].get_text(strip=True).replace(',', '')
                            value = float(value_text)

                            if date_str:
                                if date_str not in results:
                                    results[date_str] = {}
                                results[date_str][column] = value
                                extracted += 1
                        except (ValueError, AttributeError):
                            continue

                logger.info(f"  -> {extracted} records extracted")
                record_macro_ingestion_success('investing', column)
                time.sleep(delay)

            except Exception as e:
                errors.append(f"{column}: {str(e)}")
                logger.error(f"[Investing] Error {column}: {e}")
                record_macro_ingestion_error('investing', column, 'scrape_error')

        return self._create_result(results, errors, start_time)


# =============================================================================
# BanRep SUAMECA Extraction Strategy (Selenium)
# =============================================================================

@MacroExtractorFactory.register_strategy(MacroSource.BANREP)
class BanRepExtractionStrategy(ConfigurableExtractor):
    """
    BanRep SUAMECA extraction strategy using Selenium.

    Extracts Colombian monetary and economic indicators from
    Banco de la República's statistical system.
    """

    @property
    def source_name(self) -> str:
        return "banrep"

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """Extract data from BanRep via Selenium."""
        start_time = datetime.utcnow()
        results: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        try:
            import undetected_chromedriver as uc
            from selenium.webdriver.common.by import By
        except ImportError:
            errors.append("undetected-chromedriver not installed")
            return self._create_result(results, errors, start_time)

        # Get indicator config (includes URLs)
        indicators_config = self.config.get('indicators', {})

        # Setup Chrome
        driver = None
        try:
            options = uc.ChromeOptions()
            chrome_opts = self.config.get('chrome_options', [
                '--headless=new',
                '--no-sandbox',
                '--disable-dev-shm-usage'
            ])
            for opt in chrome_opts:
                options.add_argument(opt)

            driver = uc.Chrome(options=options)
            logger.info("[BanRep] Chrome initialized")
        except Exception as e:
            errors.append(f"Chrome init failed: {str(e)}")
            logger.error(f"[BanRep] Chrome init failed: {e}")
            return self._create_result(results, errors, start_time)

        page_wait = self.config.get('page_load_wait_seconds', 5)

        try:
            for serie_id, config in indicators_config.items():
                column = config.get('column')
                url = config.get('url')
                name = config.get('name', serie_id)

                if not column or not url:
                    continue

                try:
                    logger.info(f"[BanRep] Scraping {name} (Serie {serie_id})")

                    driver.get(url)
                    time.sleep(page_wait)

                    # Click "Vista tabla" button
                    buttons = driver.find_elements(By.TAG_NAME, "button")
                    for btn in buttons:
                        if 'tabla' in btn.text.lower():
                            btn.click()
                            time.sleep(page_wait)
                            break

                    # Find data table
                    tables = driver.find_elements(By.TAG_NAME, "table")
                    data_table = None
                    max_rows = 0
                    for table in tables:
                        rows = table.find_elements(By.TAG_NAME, "tr")
                        if len(rows) > max_rows:
                            max_rows = len(rows)
                            data_table = table

                    if not data_table or max_rows < 10:
                        errors.append(f"{name}: No table found")
                        record_macro_ingestion_error('banrep', column, 'parse_error')
                        continue

                    # Extract data
                    rows = data_table.find_elements(By.TAG_NAME, "tr")
                    extracted = 0

                    for row in rows[1:]:
                        try:
                            th = row.find_elements(By.TAG_NAME, "th")
                            td = row.find_elements(By.TAG_NAME, "td")

                            if not th or not td:
                                continue

                            fecha_str = th[0].text.strip()
                            valor_str = td[0].text.strip().replace(',', '.').replace('%', '')

                            if not fecha_str or not valor_str:
                                continue

                            # Parse BanRep date format (YYYY/MM/DD)
                            date_str = DateParser.parse_banrep_date(fecha_str)
                            if not date_str:
                                date_str = self._normalize_date(fecha_str)

                            if date_str:
                                valor = float(valor_str)
                                if date_str not in results:
                                    results[date_str] = {}
                                results[date_str][column] = valor
                                extracted += 1

                        except (ValueError, AttributeError):
                            continue

                    logger.info(f"  -> {extracted} records")
                    record_macro_ingestion_success('banrep', column)

                except Exception as e:
                    errors.append(f"{name}: {str(e)}")
                    logger.error(f"[BanRep] Error {name}: {e}")
                    record_macro_ingestion_error('banrep', column, 'selenium_error')

        finally:
            if driver:
                driver.quit()
                logger.info("[BanRep] Chrome closed")

        return self._create_result(results, errors, start_time)


# =============================================================================
# BCRP (EMBI) Extraction Strategy
# =============================================================================

@MacroExtractorFactory.register_strategy(MacroSource.BCRP)
class BCRPExtractionStrategy(ConfigurableExtractor):
    """
    BCRP Peru extraction strategy for EMBI Colombia spread.

    Extracts sovereign risk spread from Peru central bank statistics.
    """

    # Spanish month mapping for EMBI date format
    MESES = {
        'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
        'Set': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
    }

    @property
    def source_name(self) -> str:
        return "bcrp"

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """Extract EMBI data from BCRP Peru."""
        start_time = datetime.utcnow()
        results: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        try:
            import cloudscraper
            from bs4 import BeautifulSoup
        except ImportError:
            errors.append("cloudscraper/beautifulsoup4 not installed")
            return self._create_result(results, errors, start_time)

        url = self.config.get('url')
        column = self.config.get('column', 'crsk_spread_embi_col_d_embi')

        if not url:
            errors.append("BCRP URL not configured")
            return self._create_result(results, errors, start_time)

        user_agent = self.config.get(
            'user_agent',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        timeout = self.config.get('request_timeout_seconds', 20)
        scraper = cloudscraper.create_scraper()

        try:
            logger.info(f"[EMBI] Fetching from BCRP Peru: {url}")
            response = scraper.get(url, headers=headers, timeout=timeout)

            if response.status_code != 200:
                errors.append(f"EMBI: HTTP {response.status_code}")
                record_macro_ingestion_error('embi', column, 'http_error')
                return self._create_result(results, errors, start_time)

            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table')

            # Find largest table
            main_table = max(tables, key=lambda t: len(str(t)), default=None)

            if not main_table:
                errors.append("EMBI: No table found")
                record_macro_ingestion_error('embi', column, 'parse_error')
                return self._create_result(results, errors, start_time)

            rows = main_table.find_all('tr')
            extracted = 0

            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    fecha_str = cells[0].get_text(strip=True)
                    valor_str = cells[1].get_text(strip=True)

                    # Parse EMBI date format (e.g., '06Ene26')
                    date_str = self._parse_embi_date(fecha_str)

                    if date_str and valor_str.isdigit():
                        if date_str not in results:
                            results[date_str] = {}
                        results[date_str][column] = int(valor_str)
                        extracted += 1

            logger.info(f"  -> {extracted} EMBI records extracted")
            record_macro_ingestion_success('embi', column)

        except Exception as e:
            errors.append(f"EMBI: {str(e)}")
            logger.error(f"[EMBI] Error: {e}")
            record_macro_ingestion_error('embi', column, 'scrape_error')

        return self._create_result(results, errors, start_time)

    def _parse_embi_date(self, value: str) -> Optional[str]:
        """Parse EMBI date format like '06Ene26'."""
        match = re.match(r'(\d{2})([A-Za-z]{3})(\d{2})', value)
        if match:
            day, month_txt, year_2d = match.groups()
            month = self.MESES.get(month_txt.capitalize())
            if month:
                year_int = int(year_2d)
                year = f"20{year_2d}" if year_int <= 49 else f"19{year_2d}"
                return f"{year}-{month}-{day}"
        return None


# =============================================================================
# Fedesarrollo Extraction Strategy
# =============================================================================

@MacroExtractorFactory.register_strategy(MacroSource.FEDESARROLLO)
class FedesarrolloExtractionStrategy(ConfigurableExtractor):
    """
    Fedesarrollo extraction strategy for Colombian confidence indices.

    Extracts CCI (Consumer Confidence) and ICI (Industrial Confidence).
    """

    @property
    def source_name(self) -> str:
        return "fedesarrollo"

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """Extract data from Fedesarrollo scrapers."""
        start_time = datetime.utcnow()
        results: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        # Add scraper path to sys.path
        scraper_path = self.config.get('scraper_path', '/opt/airflow/data/pipeline/02_scrapers/02_custom')
        if scraper_path not in sys.path:
            sys.path.insert(0, scraper_path)

        scraper_module = self.config.get('scraper_module', 'scraper_fedesarrollo')
        lookback_months = self.config.get('lookback_months', 12)

        try:
            module = __import__(scraper_module)
        except ImportError as e:
            errors.append(f"Fedesarrollo scraper not available: {str(e)}")
            logger.error(f"[Fedesarrollo] Import error: {e}")
            return self._create_result(results, errors, start_time)

        indicators_config = self.config.get('indicators', {})

        for indicator_id, config in indicators_config.items():
            column = config.get('column')
            func_name = config.get('function')
            name = config.get('name', indicator_id)

            if not column or not func_name:
                continue

            try:
                logger.info(f"[Fedesarrollo] Fetching {name}...")

                # Get the scraper function
                scraper_func = getattr(module, func_name, None)
                if not scraper_func:
                    errors.append(f"{name}: Function {func_name} not found")
                    continue

                df = scraper_func(n=lookback_months)

                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        date_str = self._normalize_date(row.get('fecha'))
                        if date_str:
                            if date_str not in results:
                                results[date_str] = {}
                            results[date_str][column] = float(row['valor'])

                    logger.info(f"  -> {len(df)} {name} records extracted")
                    record_macro_ingestion_success('fedesarrollo', column)
                else:
                    errors.append(f"{name}: No data returned")
                    record_macro_ingestion_error('fedesarrollo', column, 'no_data')

            except Exception as e:
                errors.append(f"{name}: {str(e)}")
                logger.error(f"[Fedesarrollo] Error {name}: {e}")
                record_macro_ingestion_error('fedesarrollo', column, 'scrape_error')

        return self._create_result(results, errors, start_time)


# =============================================================================
# DANE Extraction Strategy
# =============================================================================

@MacroExtractorFactory.register_strategy(MacroSource.DANE)
class DANEExtractionStrategy(ConfigurableExtractor):
    """
    DANE extraction strategy for Colombian trade balance data.

    Extracts exports and imports values from Colombia's statistics agency.
    """

    @property
    def source_name(self) -> str:
        return "dane"

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """Extract data from DANE scrapers."""
        start_time = datetime.utcnow()
        results: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        # Add scraper path to sys.path
        scraper_path = self.config.get('scraper_path', '/opt/airflow/data/pipeline/02_scrapers/02_custom')
        if scraper_path not in sys.path:
            sys.path.insert(0, scraper_path)

        scraper_module = self.config.get('scraper_module', 'scraper_dane_balanza')
        lookback_months = self.config.get('lookback_months', 15)

        try:
            module = __import__(scraper_module)
        except ImportError as e:
            errors.append(f"DANE scraper not available: {str(e)}")
            logger.error(f"[DANE] Import error: {e}")
            return self._create_result(results, errors, start_time)

        indicators_config = self.config.get('indicators', {})

        for indicator_id, config in indicators_config.items():
            column = config.get('column')
            func_name = config.get('function')
            name = config.get('name', indicator_id)

            if not column or not func_name:
                continue

            try:
                logger.info(f"[DANE] Fetching {name}...")

                # Get the scraper function
                scraper_func = getattr(module, func_name, None)
                if not scraper_func:
                    errors.append(f"{name}: Function {func_name} not found")
                    continue

                df = scraper_func(n=lookback_months)

                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        fecha = row.get('fecha')
                        if isinstance(fecha, str):
                            date_str = fecha
                        else:
                            date_str = self._normalize_date(fecha)

                        if date_str:
                            if date_str not in results:
                                results[date_str] = {}
                            results[date_str][column] = float(row['valor'])

                    logger.info(f"  -> {len(df)} {name} records extracted")
                    record_macro_ingestion_success('dane', column)
                else:
                    errors.append(f"{name}: No data returned")
                    record_macro_ingestion_error('dane', column, 'no_data')

            except Exception as e:
                errors.append(f"{name}: {str(e)}")
                logger.error(f"[DANE] Error {name}: {e}")
                record_macro_ingestion_error('dane', column, 'scrape_error')

        return self._create_result(results, errors, start_time)


# =============================================================================
# BanRep BOP (Balance of Payments) Extraction Strategy
# =============================================================================

@MacroExtractorFactory.register_strategy(MacroSource.BANREP_BOP)
class BanRepBOPExtractionStrategy(ConfigurableExtractor):
    """
    BanRep Balance of Payments extraction strategy using Selenium.

    Extracts quarterly Balance of Payments data from BanRep SUAMECA catalog
    via the Highcharts data table. Available series:
    - Cuenta Corriente Trimestral (Current Account)
    - Cuenta Financiera Trimestral (Financial Account)

    NOTE: IED (Foreign Direct Investment INTO Colombia) is NOT available
    as a separate quarterly series. It's aggregated within Cuenta Financiera.

    Contract: CTR-L0-STRATEGY-BOP-001
    """

    # Catalog URL for Balance of Payments
    CATALOG_URL = "https://suameca.banrep.gov.co/estadisticas-economicas/catalogo?tema=balanza_de_pagos"

    # Column mapping: indicator_id -> table column index (1-based after date column)
    # Table: Period | Cuenta Corriente | Cuenta Financiera | Ingreso Primario | ...
    COLUMN_MAP = {
        'cuenta_corriente': 1,   # Column 1 = Cuenta Corriente trimestral
        'cuenta_financiera': 2,  # Column 2 = Cuenta Financiera trimestral
    }

    @property
    def source_name(self) -> str:
        return "banrep_bop"

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """Extract Balance of Payments data from BanRep via Selenium."""
        start_time = datetime.utcnow()
        results: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.chrome.options import Options
        except ImportError:
            errors.append("selenium not installed")
            return self._create_result(results, errors, start_time)

        # Get indicator config
        indicators_config = self.config.get('indicators', {})

        # Setup Chrome
        driver = None
        try:
            chrome_options = Options()
            chrome_opts = self.config.get('chrome_options', [
                '--headless=new',
                '--no-sandbox',
                '--disable-dev-shm-usage'
            ])
            for opt in chrome_opts:
                chrome_options.add_argument(opt)

            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(60)
            logger.info("[BanRep-BOP] Chrome initialized")
        except Exception as e:
            errors.append(f"Chrome init failed: {str(e)}")
            logger.error(f"[BanRep-BOP] Chrome init failed: {e}")
            return self._create_result(results, errors, start_time)

        page_wait = self.config.get('page_load_wait_seconds', 8)

        try:
            # Step 1: Navigate to catalog
            logger.info(f"[BanRep-BOP] Loading catalog: {self.CATALOG_URL}")
            driver.get(self.CATALOG_URL)
            time.sleep(page_wait)

            # Step 2: Expand "Cuenta corriente y cuenta financiera" category
            self._click_element_by_text(driver, "Cuenta corriente y cuenta financiera")
            time.sleep(5)

            # Step 3: Click "Vista tabla" to show data table
            self._click_vista_tabla(driver)
            time.sleep(5)

            # Step 4: Extract data from Highcharts table
            extracted = self._extract_from_highcharts_table(
                driver, indicators_config, results
            )

            if extracted > 0:
                logger.info(f"[BanRep-BOP] Total {extracted} records extracted")
                for config in indicators_config.values():
                    col = config.get('column', '')
                    if col:
                        record_macro_ingestion_success('banrep_bop', col)
            else:
                errors.append("No data extracted from BOP table")
                for config in indicators_config.values():
                    col = config.get('column', '')
                    if col:
                        record_macro_ingestion_error('banrep_bop', col, 'no_data')

        except Exception as e:
            errors.append(f"BOP extraction error: {str(e)}")
            logger.error(f"[BanRep-BOP] Error: {e}")

        finally:
            if driver:
                driver.quit()
                logger.info("[BanRep-BOP] Chrome closed")

        return self._create_result(results, errors, start_time)

    def _click_element_by_text(self, driver, text: str) -> bool:
        """Click element containing specified text."""
        from selenium.webdriver.common.by import By
        try:
            elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
            for el in elements:
                try:
                    driver.execute_script('arguments[0].click();', el)
                    logger.info(f"[BanRep-BOP] Clicked: {text[:40]}")
                    return True
                except:
                    pass
        except Exception as e:
            logger.warning(f"[BanRep-BOP] Could not click '{text}': {e}")
        return False

    def _click_vista_tabla(self, driver) -> bool:
        """Click Vista tabla button."""
        from selenium.webdriver.common.by import By
        try:
            btn = driver.find_element(By.ID, 'vistaTabla')
            driver.execute_script('arguments[0].click();', btn)
            logger.info("[BanRep-BOP] Clicked Vista tabla")
            return True
        except Exception as e:
            logger.warning(f"[BanRep-BOP] Vista tabla not found: {e}")
            return False

    def _extract_from_highcharts_table(
        self,
        driver,
        indicators_config: Dict[str, Dict],
        results: Dict[str, Dict[str, float]]
    ) -> int:
        """Extract data from the Highcharts data table (id=highcharts-data-table-0)."""
        from selenium.webdriver.common.by import By

        extracted = 0

        try:
            table = driver.find_element(By.ID, 'highcharts-data-table-0')
            rows = table.find_elements(By.TAG_NAME, 'tr')

            if len(rows) < 2:
                logger.warning("[BanRep-BOP] No data rows found")
                return 0

            # Build column index -> db column mapping
            col_mapping = {}
            for ind_id, config in indicators_config.items():
                col_name = config.get('column', '')
                col_idx = self.COLUMN_MAP.get(ind_id, 1)
                if col_name:
                    col_mapping[col_idx] = col_name

            # Process data rows (skip header)
            data_rows = rows[1:]
            logger.info(f"[BanRep-BOP] Processing {len(data_rows)} rows")

            for row in data_rows:
                try:
                    th_cells = row.find_elements(By.TAG_NAME, 'th')
                    td_cells = row.find_elements(By.TAG_NAME, 'td')

                    if not th_cells or not td_cells:
                        continue

                    # Parse date from th (format: YYYY/MM/DD)
                    date_text = th_cells[0].text.strip()
                    date_str = self._parse_date(date_text)

                    if not date_str:
                        continue

                    if date_str not in results:
                        results[date_str] = {}

                    # Extract values for each configured column
                    for col_idx, db_col in col_mapping.items():
                        td_idx = col_idx - 1  # td_cells is 0-indexed
                        if 0 <= td_idx < len(td_cells):
                            val_text = td_cells[td_idx].text.strip()
                            try:
                                value = float(val_text.replace(',', '.'))
                                results[date_str][db_col] = value
                                extracted += 1
                            except ValueError:
                                pass

                except Exception as e:
                    logger.debug(f"[BanRep-BOP] Row error: {e}")
                    continue

        except Exception as e:
            logger.error(f"[BanRep-BOP] Table extraction error: {e}")

        return extracted

    def _parse_date(self, text: str) -> Optional[str]:
        """Parse date from YYYY/MM/DD format to ISO YYYY-MM-DD."""
        if not text:
            return None

        match = re.match(r'(\d{4})/(\d{2})/(\d{2})', text)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"

        return self._normalize_date(text)


# =============================================================================
# BanRep SDMX REST API Extraction Strategy (Alternative to Selenium)
# =============================================================================

@MacroExtractorFactory.register_strategy(MacroSource.BANREP_SDMX)
class BanRepSDMXExtractionStrategy(ConfigurableExtractor):
    """
    BanRep SDMX REST API extraction strategy.

    Alternative to Selenium-based scraping for Colombian monetary indicators.
    Uses BanRep's SDMX REST API for more reliable data extraction.

    API Documentation:
        https://suameca.banrep.gov.co/estadisticas-economicas/api

    Supports indicators:
        - IBR Overnight Rate
        - TPM Policy Rate
        - ITCR (Real Exchange Rate Index)
        - Terms of Trade
        - International Reserves
        - Colombia CPI
    """

    SDMX_BASE_URL = "https://suameca.banrep.gov.co/estadisticas-economicas/api/v1"

    @property
    def source_name(self) -> str:
        return "banrep_sdmx"

    def extract(
        self,
        indicators: Dict[str, str],
        lookback_days: int,
        **kwargs
    ) -> ExtractionResult:
        """Extract data from BanRep SDMX REST API."""
        start_time = datetime.utcnow()
        results: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        timeout = self.get_timeout()
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'USDCOP-Trading-Pipeline/1.0',
        }

        # Get indicator config
        indicators_config = self.config.get('indicators', {})

        # Calculate date range
        today = datetime.now().date()
        start_date = today - timedelta(days=lookback_days)

        for serie_id, config in indicators_config.items():
            column = config.get('column')
            name = config.get('name', serie_id)
            frequency = config.get('frequency', 'D')  # D=daily, M=monthly

            if not column:
                continue

            try:
                logger.info(f"[BanRep-SDMX] Fetching {name} (Serie {serie_id})")

                # SDMX REST API endpoint for time series data
                url = f"{self.SDMX_BASE_URL}/series/{serie_id}/data"
                params = {
                    'startPeriod': start_date.strftime('%Y-%m-%d'),
                    'endPeriod': today.strftime('%Y-%m-%d'),
                    'format': 'json',
                }

                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    extracted = self._parse_sdmx_response(data, column, results)
                    logger.info(f"  -> {extracted} records from SDMX API")
                    record_macro_ingestion_success('banrep_sdmx', column)

                elif response.status_code == 404:
                    # Fallback: Try alternative JSON endpoint
                    alt_url = f"https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/{serie_id}"
                    alt_response = requests.get(alt_url, headers=headers, timeout=timeout)

                    if alt_response.status_code == 200:
                        data = alt_response.json()
                        extracted = self._parse_alt_response(data, column, results)
                        logger.info(f"  -> {extracted} records from alt JSON endpoint")
                        record_macro_ingestion_success('banrep_sdmx', column)
                    else:
                        errors.append(f"{name}: HTTP {alt_response.status_code}")
                        record_macro_ingestion_error('banrep_sdmx', column, 'http_error')
                else:
                    errors.append(f"{name}: HTTP {response.status_code}")
                    record_macro_ingestion_error('banrep_sdmx', column, 'http_error')

                # Rate limiting
                time.sleep(self.config.get('rate_limit_delay_seconds', 1))

            except requests.exceptions.Timeout:
                errors.append(f"{name}: Timeout")
                logger.warning(f"[BanRep-SDMX] Timeout for {name}")
                record_macro_ingestion_error('banrep_sdmx', column, 'timeout')

            except requests.exceptions.RequestException as e:
                errors.append(f"{name}: {str(e)}")
                logger.error(f"[BanRep-SDMX] Request error {name}: {e}")
                record_macro_ingestion_error('banrep_sdmx', column, 'request_error')

            except Exception as e:
                errors.append(f"{name}: {str(e)}")
                logger.error(f"[BanRep-SDMX] Error {name}: {e}")
                record_macro_ingestion_error('banrep_sdmx', column, 'api_error')

        return self._create_result(results, errors, start_time)

    def _parse_sdmx_response(
        self,
        data: Dict[str, Any],
        column: str,
        results: Dict[str, Dict[str, float]]
    ) -> int:
        """Parse SDMX JSON response format."""
        extracted = 0

        # SDMX-JSON structure: data.dataSets[0].series.0:0:0.observations
        try:
            datasets = data.get('data', {}).get('dataSets', [])
            if not datasets:
                return 0

            observations = datasets[0].get('series', {})
            time_periods = data.get('data', {}).get('structure', {}).get('dimensions', {}).get('observation', [])

            # Find time dimension
            time_dim = None
            for dim in time_periods:
                if dim.get('id') == 'TIME_PERIOD':
                    time_dim = dim.get('values', [])
                    break

            if not time_dim:
                return 0

            # Extract values
            for series_key, series_data in observations.items():
                obs = series_data.get('observations', {})
                for idx, values in obs.items():
                    if int(idx) < len(time_dim):
                        period = time_dim[int(idx)].get('id')
                        date_str = self._normalize_date(period)
                        if date_str and values:
                            value = values[0] if isinstance(values, list) else values
                            if value is not None:
                                if date_str not in results:
                                    results[date_str] = {}
                                results[date_str][column] = float(value)
                                extracted += 1

        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"[BanRep-SDMX] Parse error: {e}")

        return extracted

    def _parse_alt_response(
        self,
        data: Dict[str, Any],
        column: str,
        results: Dict[str, Dict[str, float]]
    ) -> int:
        """Parse alternative JSON response format from BanRep."""
        extracted = 0

        try:
            # Alternative format: {data: [{fecha: ..., valor: ...}, ...]}
            records = data.get('data', [])
            if isinstance(records, list):
                for record in records:
                    fecha = record.get('fecha') or record.get('date') or record.get('period')
                    valor = record.get('valor') or record.get('value') or record.get('obs_value')

                    if fecha and valor is not None:
                        date_str = self._normalize_date(fecha)
                        if date_str:
                            if date_str not in results:
                                results[date_str] = {}
                            results[date_str][column] = float(valor)
                            extracted += 1

        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"[BanRep-SDMX] Alt parse error: {e}")

        return extracted
