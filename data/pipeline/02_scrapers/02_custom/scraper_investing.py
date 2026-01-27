# -*- coding: utf-8 -*-
"""
Investing.com Scraper - Dual Strategy (AJAX + REST API)
========================================================

Scraper robusto para Investing.com con dos estrategias:

1. AJAX Endpoint (/instruments/HistoricalDataAjax)
   - Usa pair_id
   - Funciona para: DXY, VIX, US10Y, US2Y, Gold, Coffee, Brent, WTI
   - NO funciona correctamente para: COLCAP, USDMXN, USDCLP (valores invertidos/incorrectos)

2. REST API (/api/financialdata/historical/{id})
   - Usa instrument_id (diferente del pair_id)
   - Funciona para: COLCAP, USDMXN, USDCLP, y todos los demás
   - Requiere header 'domain-id' y 'Accept-Encoding: gzip, deflate' (no brotli)

El scraper elige automáticamente el mejor método según el indicador.

Contract: CTR-L0-SCRAPER-INVESTING-002
Version: 3.0.0
"""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import pandas as pd

try:
    import cloudscraper
    from bs4 import BeautifulSoup
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class FetchMethod(Enum):
    """Método de extracción preferido."""
    AJAX = "ajax"           # POST /instruments/HistoricalDataAjax
    API = "api"             # GET /api/financialdata/historical/{id}
    HTML = "html"           # GET page + parse table (fallback)


@dataclass
class IndicatorConfig:
    """Configuración de un indicador de Investing.com."""
    column: str                     # Nombre de columna en DB
    url: str                        # URL de la página histórica
    name: str                       # Nombre legible
    method: FetchMethod             # Método preferido
    pair_id: Optional[int] = None   # ID para AJAX (puede ser incorrecto para algunos)
    instrument_id: Optional[int] = None  # ID para REST API (siempre correcto)
    domain_id: str = 'www'          # Domain para API (es, www, etc.)
    expected_range: Optional[Tuple[float, float]] = None  # Rango de validación


# =============================================================================
# INDICATOR CONFIGURATION - Single Source of Truth
# =============================================================================
# Esta configuración define qué método usar para cada indicador.
# Verificado el 2026-01-24.

INDICATOR_CONFIG: Dict[str, IndicatorConfig] = {
    # =========================================================================
    # INDICADORES QUE FUNCIONAN CON AJAX (pair_id correcto)
    # =========================================================================
    'fxrt_index_dxy_usa_d_dxy': IndicatorConfig(
        column='fxrt_index_dxy_usa_d_dxy',
        url='https://www.investing.com/indices/usdollar-historical-data',
        name='DXY (Dollar Index)',
        method=FetchMethod.AJAX,
        pair_id=8827,
        instrument_id=8827,
        expected_range=(80, 130),
    ),
    'volt_vix_usa_d_vix': IndicatorConfig(
        column='volt_vix_usa_d_vix',
        url='https://www.investing.com/indices/volatility-s-p-500-historical-data',
        name='VIX',
        method=FetchMethod.AJAX,
        pair_id=44336,
        instrument_id=44336,
        expected_range=(9, 90),
    ),
    'finc_bond_yield10y_usa_d_ust10y': IndicatorConfig(
        column='finc_bond_yield10y_usa_d_ust10y',
        url='https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data',
        name='US 10Y Treasury',
        method=FetchMethod.AJAX,
        pair_id=23705,
        instrument_id=23705,
        expected_range=(0, 12),
    ),
    'finc_bond_yield2y_usa_d_dgs2': IndicatorConfig(
        column='finc_bond_yield2y_usa_d_dgs2',
        url='https://www.investing.com/rates-bonds/u.s.-2-year-bond-yield-historical-data',
        name='US 2Y Treasury',
        method=FetchMethod.AJAX,
        pair_id=23701,
        instrument_id=23701,
        expected_range=(0, 12),
    ),
    'comm_metal_gold_glb_d_gold': IndicatorConfig(
        column='comm_metal_gold_glb_d_gold',
        url='https://www.investing.com/commodities/gold-historical-data',
        name='Gold',
        method=FetchMethod.AJAX,
        pair_id=8830,
        instrument_id=8830,
        expected_range=(1000, 6000),  # Updated: gold has risen significantly
    ),
    'comm_agri_coffee_glb_d_coffee': IndicatorConfig(
        column='comm_agri_coffee_glb_d_coffee',
        url='https://www.investing.com/commodities/us-coffee-c-historical-data',
        name='Coffee',
        method=FetchMethod.AJAX,
        pair_id=8832,
        instrument_id=8832,
        expected_range=(50, 500),
    ),
    'comm_oil_brent_glb_d_brent': IndicatorConfig(
        column='comm_oil_brent_glb_d_brent',
        url='https://www.investing.com/commodities/brent-oil-historical-data',
        name='Brent Oil',
        method=FetchMethod.AJAX,
        pair_id=8833,
        instrument_id=8833,
        expected_range=(30, 150),
    ),
    'comm_oil_wti_glb_d_wti': IndicatorConfig(
        column='comm_oil_wti_glb_d_wti',
        url='https://www.investing.com/commodities/crude-oil-historical-data',
        name='WTI Oil',
        method=FetchMethod.AJAX,
        pair_id=8849,
        instrument_id=8849,
        expected_range=(25, 150),
    ),

    # =========================================================================
    # INDICADORES QUE REQUIEREN REST API (AJAX devuelve valores incorrectos)
    # =========================================================================
    'eqty_index_colcap_col_d_colcap': IndicatorConfig(
        column='eqty_index_colcap_col_d_colcap',
        url='https://www.investing.com/indices/colcap-historical-data',
        name='COLCAP',
        method=FetchMethod.API,  # AJAX pair_id 40830 returns wrong value (53846)
        pair_id=40830,           # WRONG - returns points * 100 or similar
        instrument_id=49642,     # CORRECT - returns actual COLCAP value
        expected_range=(800, 3000),
    ),
    'fxrt_spot_usdmxn_mex_d_usdmxn': IndicatorConfig(
        column='fxrt_spot_usdmxn_mex_d_usdmxn',
        url='https://es.investing.com/currencies/usd-mxn-historical-data',
        name='USD/MXN',
        method=FetchMethod.API,  # AJAX pair_id 2124 returns MXN/USD (inverted)
        pair_id=2124,            # WRONG - returns ~0.85 (inverted)
        instrument_id=39,        # CORRECT - returns ~17-20
        domain_id='es',
        expected_range=(12, 30),
    ),
    'fxrt_spot_usdclp_chl_d_usdclp': IndicatorConfig(
        column='fxrt_spot_usdclp_chl_d_usdclp',
        url='https://es.investing.com/currencies/usd-clp-historical-data',
        name='USD/CLP',
        method=FetchMethod.API,  # AJAX pair_id 2126 returns wrong value
        pair_id=2126,            # WRONG - returns ~0.73 (inverted/scaled)
        instrument_id=2110,      # CORRECT - returns ~700-1000
        domain_id='es',
        expected_range=(600, 1200),
    ),

    # =========================================================================
    # INDICADORES COLOMBIANOS (REST API verificado 2026-01-24)
    # =========================================================================
    'finc_bond_yield10y_col_d_col10y': IndicatorConfig(
        column='finc_bond_yield10y_col_d_col10y',
        url='https://www.investing.com/rates-bonds/colombia-10-year-bond-yield-historical-data',
        name='Colombia 10Y',
        method=FetchMethod.API,
        instrument_id=29236,
        expected_range=(4, 20),
    ),
    'finc_bond_yield5y_col_d_col5y': IndicatorConfig(
        column='finc_bond_yield5y_col_d_col5y',
        url='https://www.investing.com/rates-bonds/colombia-5-year-bond-yield-historical-data',
        name='Colombia 5Y',
        method=FetchMethod.API,
        instrument_id=29240,
        expected_range=(4, 20),
    ),

    # =========================================================================
    # USD/COP - Primary FX pair for this project
    # =========================================================================
    'fxrt_spot_usdcop_col_d_usdcop': IndicatorConfig(
        column='fxrt_spot_usdcop_col_d_usdcop',
        url='https://www.investing.com/currencies/usd-cop-historical-data',
        name='USD/COP',
        method=FetchMethod.API,
        instrument_id=2112,
        expected_range=(3000, 5500),
    ),
}


# =============================================================================
# LEGACY MAPPINGS (for backwards compatibility)
# =============================================================================

INVESTING_PAIR_IDS: Dict[str, int] = {
    config.column: config.pair_id
    for config in INDICATOR_CONFIG.values()
    if config.pair_id is not None
}

INVESTING_INSTRUMENT_IDS: Dict[str, int] = {
    config.column: config.instrument_id
    for config in INDICATOR_CONFIG.values()
    if config.instrument_id is not None
}


# =============================================================================
# URL Mappings
# =============================================================================

INVESTING_URLS: Dict[str, str] = {
    # Commodities
    'WTI': 'https://www.investing.com/commodities/crude-oil-historical-data',
    'BRENT': 'https://www.investing.com/commodities/brent-oil-historical-data',
    'COAL': 'https://www.investing.com/commodities/newcastle-coal-futures-historical-data',
    'GOLD': 'https://www.investing.com/commodities/gold-historical-data',
    'COFFEE': 'https://www.investing.com/commodities/us-coffee-c-historical-data',

    # Forex - USDCOP is PRIMARY for Forecasting pipeline
    'USDCOP': 'https://www.investing.com/currencies/usd-cop-historical-data',
    'USDCLP': 'https://es.investing.com/currencies/usd-clp-historical-data',
    'USDMXN': 'https://es.investing.com/currencies/usd-mxn-historical-data',

    # Indices
    'DXY': 'https://www.investing.com/indices/usdollar-historical-data',
    'VIX': 'https://www.investing.com/indices/volatility-s-p-500-historical-data',
    'COLCAP': 'https://www.investing.com/indices/colcap-historical-data',

    # Bonds
    'UST10Y': 'https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data',
    'UST2Y': 'https://www.investing.com/rates-bonds/u.s.-2-year-bond-yield-historical-data',
}


# =============================================================================
# Spanish Month Mapping (for date parsing)
# =============================================================================

ES_MONTHS: Dict[str, int] = {
    'ene': 1, 'enero': 1,
    'feb': 2, 'febrero': 2,
    'mar': 3, 'marzo': 3,
    'abr': 4, 'abril': 4,
    'may': 5, 'mayo': 5,
    'jun': 6, 'junio': 6,
    'jul': 7, 'julio': 7,
    'ago': 8, 'agosto': 8,
    'sep': 9, 'sept': 9, 'septiembre': 9,
    'oct': 10, 'octubre': 10,
    'nov': 11, 'noviembre': 11,
    'dic': 12, 'diciembre': 12,
}

EN_MONTHS: Dict[str, int] = {
    'jan': 1, 'january': 1,
    'feb': 2, 'february': 2,
    'mar': 3, 'march': 3,
    'apr': 4, 'april': 4,
    'may': 5,
    'jun': 6, 'june': 6,
    'jul': 7, 'july': 7,
    'aug': 8, 'august': 8,
    'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10,
    'nov': 11, 'november': 11,
    'dec': 12, 'december': 12,
}


# =============================================================================
# Session Management
# =============================================================================

def create_session() -> 'cloudscraper.CloudScraper':
    """
    Create a cloudscraper session with realistic browser headers.

    Returns:
        CloudScraper session configured for Investing.com
    """
    if not CLOUDSCRAPER_AVAILABLE:
        raise ImportError(
            "cloudscraper and beautifulsoup4 required. "
            "Install with: pip install cloudscraper beautifulsoup4"
        )

    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True,
        }
    )

    scraper.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
    })

    return scraper


# =============================================================================
# Date Parsing
# =============================================================================

def parse_investing_date(date_str: str) -> Optional[str]:
    """
    Parse date from Investing.com to ISO format (YYYY-MM-DD).

    Supports multiple formats:
        - English: "Jan 22, 2026", "January 22, 2026"
        - Spanish: "22 ene. 2026", "22 enero 2026"
        - US format: "01/22/2026" (MM/DD/YYYY)
        - ISO format: "2026-01-22"

    Args:
        date_str: Date string from Investing.com

    Returns:
        ISO date string (YYYY-MM-DD) or None if parsing fails
    """
    if not date_str:
        return None

    date_str = date_str.strip().lower()

    # Remove periods and extra whitespace
    date_str = date_str.replace('.', '').strip()

    # Try ISO format first (already correct)
    if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            pass

    # Try English format: "jan 22, 2026" or "january 22, 2026"
    try:
        parts = date_str.replace(',', '').split()
        if len(parts) == 3:
            month_str = parts[0]
            day = int(parts[1])
            year = int(parts[2])

            month = EN_MONTHS.get(month_str[:3])
            if month:
                return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        pass

    # Try Spanish format: "22 ene 2026" or "22 enero 2026"
    try:
        parts = date_str.split()
        if len(parts) == 3:
            day = int(parts[0])
            month_str = parts[1][:3]  # First 3 chars
            year = int(parts[2])

            month = ES_MONTHS.get(month_str)
            if month:
                return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        pass

    # Try US format: MM/DD/YYYY
    try:
        dt = datetime.strptime(date_str, '%m/%d/%Y')
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        pass

    # Try pandas as fallback
    try:
        parsed = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(parsed):
            return parsed.strftime('%Y-%m-%d')
    except Exception:
        pass

    logger.debug(f"Could not parse date: {date_str}")
    return None


# =============================================================================
# AJAX Historical Data Fetching
# =============================================================================

def fetch_historical_ajax(
    pair_id: int,
    start_date: str,
    end_date: str,
    session: Optional['cloudscraper.CloudScraper'] = None,
    delay: float = 3.0,
    referer_url: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical data using Investing.com AJAX endpoint.

    This endpoint allows fetching large date ranges with custom filters.
    Requires pair_id for the instrument.

    Args:
        pair_id: Instrument ID in Investing.com
        start_date: Start date in 'MM/DD/YYYY' format
        end_date: End date in 'MM/DD/YYYY' format
        session: CloudScraper session (creates new if None)
        delay: Base delay between requests in seconds
        referer_url: Optional referer URL for the request

    Returns:
        DataFrame with columns ['fecha', 'valor']
    """
    if session is None:
        session = create_session()

    ajax_url = 'https://www.investing.com/instruments/HistoricalDataAjax'

    # Default referer
    if referer_url is None:
        referer_url = 'https://www.investing.com/'

    headers = {
        'X-Requested-With': 'XMLHttpRequest',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': referer_url,
        'Accept': '*/*',
    }

    payload = {
        'curr_id': pair_id,
        'smlID': str(random.randint(1000000, 99999999)),
        'header': 'Historical Data',
        'st_date': start_date,
        'end_date': end_date,
        'interval_sec': 'Daily',
        'sort_col': 'date',
        'sort_ord': 'DESC',
        'action': 'historical_data',
    }

    # Add jitter to delay
    actual_delay = delay + random.uniform(0, 1)
    time.sleep(actual_delay)

    try:
        response = session.post(
            ajax_url,
            data=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find table with id='curr_table' first, then any table
        table = soup.find('table', {'id': 'curr_table'})
        if not table:
            table = soup.find('table')

        if not table:
            logger.warning(f"No table found in AJAX response for pair_id {pair_id}")
            return pd.DataFrame(columns=['fecha', 'valor'])

        rows = []
        for tr in table.find_all('tr')[1:]:  # Skip header
            cells = tr.find_all('td')
            if len(cells) >= 2:
                fecha_str = cells[0].get_text(strip=True)
                valor_str = cells[1].get_text(strip=True).replace(',', '')

                try:
                    fecha = parse_investing_date(fecha_str)
                    valor = float(valor_str)
                    if fecha:
                        rows.append({'fecha': fecha, 'valor': valor})
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skip row: {fecha_str}, {valor_str} - {e}")
                    continue

        df = pd.DataFrame(rows)
        logger.debug(f"AJAX fetch returned {len(df)} rows for pair_id {pair_id}")
        return df

    except Exception as e:
        logger.error(f"AJAX fetch failed for pair_id {pair_id}: {e}")
        return pd.DataFrame(columns=['fecha', 'valor'])


def fetch_historical_ajax_chunked(
    pair_id: int,
    start_date: str,
    end_date: str,
    session: Optional['cloudscraper.CloudScraper'] = None,
    chunk_days: int = 365,
    delay: float = 3.0,
    referer_url: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical data in yearly chunks to avoid timeout issues.

    Args:
        pair_id: Instrument ID in Investing.com
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        session: CloudScraper session (creates new if None)
        chunk_days: Days per chunk (default 365)
        delay: Base delay between requests
        referer_url: Optional page URL to visit first for cookies

    Returns:
        DataFrame with columns ['fecha', 'valor']
    """
    if session is None:
        session = create_session()

    # Visit a page first to get cookies if referer provided
    if referer_url:
        try:
            logger.debug(f"  Visiting {referer_url} to get cookies...")
            session.get(referer_url, timeout=30)
            time.sleep(2)
        except Exception as e:
            logger.warning(f"  Could not visit referer page: {e}")

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    all_data = []
    current_start = start_dt

    while current_start < end_dt:
        chunk_end = min(current_start + timedelta(days=chunk_days), end_dt)

        # Convert to MM/DD/YYYY for AJAX endpoint
        start_str = current_start.strftime('%m/%d/%Y')
        end_str = chunk_end.strftime('%m/%d/%Y')

        logger.info(f"  Fetching {current_start.date()} to {chunk_end.date()}")

        df = fetch_historical_ajax(pair_id, start_str, end_str, session, delay, referer_url)

        if not df.empty:
            all_data.append(df)
            logger.info(f"    -> {len(df)} rows")

        current_start = chunk_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame(columns=['fecha', 'valor'])

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['fecha']).sort_values('fecha')
    return combined.reset_index(drop=True)


# =============================================================================
# HTML Page Scraping (for recent data)
# =============================================================================

def obtener_investing_com(
    url: str,
    n: int = 100,
    session: Optional['cloudscraper.CloudScraper'] = None,
    delay: float = 3.0,
    max_retries: int = 3
) -> Optional[pd.DataFrame]:
    """
    Scrape historical data from Investing.com HTML page.

    This function extracts recent data (typically last 20-100 days)
    from the visible HTML table on the historical data page.

    Args:
        url: URL of the historical data page
        n: Maximum number of records to extract
        session: CloudScraper session (creates new if None)
        delay: Base delay before request
        max_retries: Number of retry attempts

    Returns:
        DataFrame with columns [fecha, valor] or None if failed
    """
    if session is None:
        session = create_session()

    for attempt in range(max_retries):
        try:
            # Rate limiting with jitter
            actual_delay = delay + random.uniform(0, 1)
            time.sleep(actual_delay)

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }

            response = session.get(url, headers=headers, timeout=25)

            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {url}")
                if attempt < max_retries - 1:
                    backoff = delay * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find data table using multiple selectors
            table = soup.find('table', class_='freeze-column-w-1')
            if not table:
                table = soup.find('table', {'data-test': 'historical-data-table'})
            if not table:
                # Find largest table
                tables = soup.find_all('table')
                if tables:
                    table = max(tables, key=lambda t: len(str(t)))

            if not table:
                logger.warning(f"No table found at {url}")
                return None

            # Extract rows
            rows = table.find_all('tr')[1:]  # Skip header

            data = []
            for row in rows[:n*2]:  # Request extra rows in case of parse failures
                cols = row.find_all('td')
                if len(cols) >= 2:
                    fecha_str = cols[0].get_text(strip=True)
                    valor_str = cols[1].get_text(strip=True).replace(',', '')

                    try:
                        fecha = parse_investing_date(fecha_str)
                        valor = float(valor_str)
                        if fecha:
                            data.append({'fecha': fecha, 'valor': valor})
                    except (ValueError, TypeError):
                        continue

                if len(data) >= n:
                    break

            if not data:
                logger.warning(f"No data extracted from table at {url}")
                return None

            df = pd.DataFrame(data)
            df = df.sort_values('fecha', ascending=False).reset_index(drop=True)
            logger.info(f"Extracted {len(df)} rows from {url}")
            return df.head(n)

        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                backoff = delay * (2 ** attempt)
                time.sleep(backoff)
            else:
                logger.error(f"All retries failed for {url}")
                return None

    return None


def obtener_investing_com_enhanced(
    url: str,
    column_name: str,
    n: int = 100,
    session: Optional['cloudscraper.CloudScraper'] = None,
    delay: float = 3.0
) -> pd.DataFrame:
    """
    Scrape historical data with column name for database integration.

    Args:
        url: URL of the historical data page
        column_name: Database column name for the value
        n: Maximum number of records
        session: CloudScraper session
        delay: Base delay in seconds

    Returns:
        DataFrame with columns ['fecha', column_name]
    """
    df = obtener_investing_com(url, n, session, delay)

    if df is None or df.empty:
        return pd.DataFrame(columns=['fecha', column_name])

    df = df.rename(columns={'valor': column_name})
    return df


# =============================================================================
# Specific Variable Functions (for backwards compatibility)
# =============================================================================

def obtener_wti(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener precio WTI Oil"""
    return obtener_investing_com(INVESTING_URLS['WTI'], n)


def obtener_brent(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener precio Brent Oil"""
    return obtener_investing_com(INVESTING_URLS['BRENT'], n)


def obtener_coal(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener precio Coal"""
    return obtener_investing_com(INVESTING_URLS['COAL'], n)


def obtener_gold(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener precio Gold"""
    return obtener_investing_com(INVESTING_URLS['GOLD'], n)


def obtener_coffee(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener precio Coffee"""
    return obtener_investing_com(INVESTING_URLS['COFFEE'], n)


def obtener_usdclp(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener tipo de cambio USD/CLP"""
    return obtener_investing_com(INVESTING_URLS['USDCLP'], n)


def obtener_usdmxn(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener tipo de cambio USD/MXN"""
    return obtener_investing_com(INVESTING_URLS['USDMXN'], n)


def obtener_dxy(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener índice DXY (Dollar Index)"""
    return obtener_investing_com(INVESTING_URLS['DXY'], n)


def obtener_vix(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener VIX Volatility Index"""
    return obtener_investing_com(INVESTING_URLS['VIX'], n)


def obtener_ust10y(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener rendimiento bono USA 10Y"""
    return obtener_investing_com(INVESTING_URLS['UST10Y'], n)


def obtener_ust2y(n: int = 20) -> Optional[pd.DataFrame]:
    """Obtener rendimiento bono USA 2Y"""
    return obtener_investing_com(INVESTING_URLS['UST2Y'], n)


def obtener_usdcop(n: int = 20) -> Optional[pd.DataFrame]:
    """
    Obtener tipo de cambio USD/COP (OFICIAL para Forecasting).

    IMPORTANTE: Para datos OHLCV completos, usar USDCOPInvestingScraper
    de scraper_usdcop_investing.py que retorna open, high, low, close.

    Esta función solo retorna fecha y precio de cierre.
    """
    return obtener_investing_com(INVESTING_URLS['USDCOP'], n)


# =============================================================================
# REST API Historical Data Fetching
# =============================================================================

def fetch_historical_api(
    instrument_id: int,
    start_date: str,
    end_date: str,
    session: Optional['cloudscraper.CloudScraper'] = None,
    delay: float = 3.0,
    referer_url: Optional[str] = None,
    domain_id: str = 'es'
) -> pd.DataFrame:
    """
    Fetch historical data using Investing.com's modern API endpoint.

    This is the preferred method for FX pairs and other instruments.
    Uses the /api/financialdata/historical/{id} endpoint.

    Args:
        instrument_id: Instrument ID (e.g., 39 for USD/MXN, 2110 for USD/CLP)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        session: CloudScraper session (creates new if None)
        delay: Base delay between requests in seconds
        referer_url: URL to use as referer (helps avoid blocking)
        domain_id: Domain ID for API ('es', 'www', etc.)

    Returns:
        DataFrame with columns ['fecha', 'valor']
    """
    if session is None:
        session = create_session()

    # Visit the referer page first to establish cookies
    if referer_url:
        try:
            logger.info(f"  Visiting {referer_url} to establish session...")
            session.get(referer_url, timeout=30)
            time.sleep(delay)
        except Exception as e:
            logger.warning(f"  Could not visit referer page: {e}")

    api_url = f'https://api.investing.com/api/financialdata/historical/{instrument_id}'

    params = {
        'start-date': start_date,
        'end-date': end_date,
        'time-frame': 'Daily',
        'add-missing-rows': 'false'
    }

    headers = {
        'Accept': 'application/json',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',  # Exclude 'br' (brotli) as it may not decompress correctly
        'Referer': referer_url or 'https://es.investing.com/',
        'domain-id': domain_id,
    }

    # Add jitter to delay
    actual_delay = delay + random.uniform(0, 1)
    time.sleep(actual_delay)

    try:
        response = session.get(api_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()

        if 'data' not in data or not data['data']:
            logger.warning(f"No data in API response for instrument {instrument_id}")
            return pd.DataFrame(columns=['fecha', 'valor'])

        rows = []
        for item in data['data']:
            try:
                # Parse date from API format (DD.MM.YYYY or timestamp)
                date_str = item.get('rowDateTimestamp', item.get('rowDate', ''))
                if 'T' in date_str:
                    # ISO format: 2026-01-23T00:00:00Z
                    fecha = date_str.split('T')[0]
                else:
                    # DD.MM.YYYY format
                    parts = date_str.split('.')
                    if len(parts) == 3:
                        fecha = f"{parts[2]}-{parts[1]}-{parts[0]}"
                    else:
                        continue

                # Get the raw close value (more precise)
                valor = float(item.get('last_closeRaw', item.get('last_close', '0').replace(',', '')))

                rows.append({'fecha': fecha, 'valor': valor})
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Skipping row: {e}")
                continue

        df = pd.DataFrame(rows)

        if df.empty:
            return pd.DataFrame(columns=['fecha', 'valor'])

        df = df.drop_duplicates(subset=['fecha']).sort_values('fecha')
        df = df.reset_index(drop=True)

        logger.info(f"API fetch returned {len(df)} rows for instrument {instrument_id}")
        return df

    except Exception as e:
        logger.error(f"API fetch failed for instrument {instrument_id}: {e}")
        return pd.DataFrame(columns=['fecha', 'valor'])


def fetch_historical_from_url(
    url: str,
    start_date: str,
    end_date: str,
    session: Optional['cloudscraper.CloudScraper'] = None,
    delay: float = 3.0,
    instrument_id: Optional[int] = None
) -> pd.DataFrame:
    """
    High-level function to fetch historical data from Investing.com URL.

    Automatically determines the best method to use based on the URL
    and instrument type.

    Args:
        url: URL of the historical data page
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        session: CloudScraper session (creates new if None)
        delay: Base delay between requests
        instrument_id: Optional instrument ID for API-based fetching

    Returns:
        DataFrame with columns ['fecha', 'valor']
    """
    if instrument_id:
        return fetch_historical_api(
            instrument_id=instrument_id,
            start_date=start_date,
            end_date=end_date,
            session=session,
            delay=delay,
            referer_url=url
        )

    # Try to determine instrument ID from URL
    if 'usd-mxn' in url.lower():
        return fetch_historical_api(
            instrument_id=39,  # USD/MXN
            start_date=start_date,
            end_date=end_date,
            session=session,
            delay=delay,
            referer_url=url
        )
    elif 'usd-clp' in url.lower():
        return fetch_historical_api(
            instrument_id=2110,  # USD/CLP
            start_date=start_date,
            end_date=end_date,
            session=session,
            delay=delay,
            referer_url=url
        )
    elif 'usd-cop' in url.lower():
        return fetch_historical_api(
            instrument_id=2112,  # USD/COP (verified: 1583 rows, values 3287-3639)
            start_date=start_date,
            end_date=end_date,
            session=session,
            delay=delay,
            referer_url=url
        )

    # Fallback to HTML scraping for other URLs
    logger.warning(f"No instrument ID found for {url}, falling back to HTML scraping")
    return _fetch_historical_html_fallback(url, start_date, end_date, session, delay)


def _fetch_historical_html_fallback(
    url: str,
    start_date: str,
    end_date: str,
    session: Optional['cloudscraper.CloudScraper'] = None,
    delay: float = 4.0
) -> pd.DataFrame:
    """
    Fallback HTML scraping for pages without known instrument IDs.

    Note: This only gets recent data visible on the page (~20-30 days).
    For full historical data, use fetch_historical_api with the correct instrument_id.
    """
    if session is None:
        session = create_session()

    logger.info(f"Fetching visible data from {url} (limited to page content)")

    try:
        time.sleep(delay + random.uniform(0, 1))
        response = session.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the data table
        table = soup.find('table', class_='freeze-column-w-1')
        if not table:
            tables = soup.find_all('table')
            if tables:
                table = max(tables, key=lambda t: len(t.find_all('tr')))

        if not table:
            logger.warning(f"No table found at {url}")
            return pd.DataFrame(columns=['fecha', 'valor'])

        rows = []
        for tr in table.find_all('tr')[1:]:
            cells = tr.find_all('td')
            if len(cells) >= 2:
                fecha_str = cells[0].get_text(strip=True)
                valor_str = cells[1].get_text(strip=True)

                try:
                    # Parse DD.MM.YYYY format
                    parts = fecha_str.split('.')
                    if len(parts) == 3:
                        fecha = f"{parts[2]}-{parts[1]}-{parts[0]}"
                    else:
                        fecha = parse_investing_date(fecha_str)

                    # Convert European number format
                    valor = float(valor_str.replace('.', '').replace(',', '.'))

                    if fecha:
                        rows.append({'fecha': fecha, 'valor': valor})
                except (ValueError, TypeError):
                    continue

        if not rows:
            return pd.DataFrame(columns=['fecha', 'valor'])

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=['fecha']).sort_values('fecha')
        return df.reset_index(drop=True)

    except Exception as e:
        logger.error(f"HTML fallback failed for {url}: {e}")
        return pd.DataFrame(columns=['fecha', 'valor'])


# =============================================================================
# Utility Functions
# =============================================================================

def get_pair_id(column_name: str) -> Optional[int]:
    """
    Get the Investing.com pair ID for a database column name.

    Args:
        column_name: Database column name

    Returns:
        Pair ID or None if not found
    """
    return INVESTING_PAIR_IDS.get(column_name)


def list_supported_indicators() -> List[Tuple[str, int, str]]:
    """
    List all supported indicators with their pair IDs and URLs.

    Returns:
        List of (column_name, pair_id, url) tuples
    """
    result = []
    for column, pair_id in INVESTING_PAIR_IDS.items():
        # Find URL for this column (if mapped)
        url = None
        for name, u in INVESTING_URLS.items():
            if name.lower() in column.lower():
                url = u
                break
        result.append((column, pair_id, url or 'N/A'))
    return result


# =============================================================================
# ROBUST EXTRACTION FUNCTION (uses INDICATOR_CONFIG)
# =============================================================================

def fetch_historical_robust(
    column: str,
    start_date: str,
    end_date: str,
    session: Optional['cloudscraper.CloudScraper'] = None,
    delay: float = 3.0,
    validate: bool = True
) -> pd.DataFrame:
    """
    Fetch historical data using the optimal method for each indicator.

    This is the main entry point for extracting data from Investing.com.
    It automatically selects AJAX, REST API, or HTML scraping based on
    the indicator configuration, with automatic fallback if the primary
    method fails.

    Args:
        column: Database column name (e.g., 'fxrt_index_dxy_usa_d_dxy')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        session: CloudScraper session (creates new if None)
        delay: Base delay between requests in seconds
        validate: If True, validate values against expected range

    Returns:
        DataFrame with columns ['fecha', 'valor']

    Example:
        >>> df = fetch_historical_robust(
        ...     'fxrt_spot_usdmxn_mex_d_usdmxn',
        ...     '2020-01-01',
        ...     '2026-01-23'
        ... )
        >>> print(f"Got {len(df)} rows")
    """
    # Get indicator configuration
    config = INDICATOR_CONFIG.get(column)
    if not config:
        logger.warning(f"No configuration found for {column}, using legacy methods")
        # Try to use legacy pair_id
        pair_id = INVESTING_PAIR_IDS.get(column)
        if pair_id:
            return fetch_historical_ajax_chunked(
                pair_id=pair_id,
                start_date=start_date,
                end_date=end_date,
                session=session,
                delay=delay
            )
        return pd.DataFrame(columns=['fecha', 'valor'])

    if session is None:
        session = create_session()

    logger.info(f"Fetching {config.name} ({column})")
    logger.info(f"  Method: {config.method.value}")
    logger.info(f"  Date range: {start_date} to {end_date}")

    df = pd.DataFrame(columns=['fecha', 'valor'])

    # Try primary method
    try:
        if config.method == FetchMethod.API:
            if config.instrument_id:
                logger.info(f"  Using REST API (instrument_id={config.instrument_id})")
                df = fetch_historical_api(
                    instrument_id=config.instrument_id,
                    start_date=start_date,
                    end_date=end_date,
                    session=session,
                    delay=delay,
                    referer_url=config.url,
                    domain_id=config.domain_id
                )
        elif config.method == FetchMethod.AJAX:
            if config.pair_id:
                logger.info(f"  Using AJAX (pair_id={config.pair_id})")
                df = fetch_historical_ajax_chunked(
                    pair_id=config.pair_id,
                    start_date=start_date,
                    end_date=end_date,
                    session=session,
                    delay=delay,
                    referer_url=config.url
                )
        elif config.method == FetchMethod.HTML:
            logger.info(f"  Using HTML scraping")
            df = _fetch_historical_html_fallback(
                url=config.url,
                start_date=start_date,
                end_date=end_date,
                session=session,
                delay=delay
            )
    except Exception as e:
        logger.warning(f"  Primary method failed: {e}")

    # If primary failed and we have alternative IDs, try fallback
    if df.empty:
        logger.info(f"  Trying fallback method...")
        try:
            if config.method == FetchMethod.AJAX and config.instrument_id:
                # AJAX failed, try API
                logger.info(f"  Fallback: REST API (instrument_id={config.instrument_id})")
                df = fetch_historical_api(
                    instrument_id=config.instrument_id,
                    start_date=start_date,
                    end_date=end_date,
                    session=session,
                    delay=delay,
                    referer_url=config.url,
                    domain_id=config.domain_id
                )
            elif config.method == FetchMethod.API and config.pair_id:
                # API failed, try AJAX
                logger.info(f"  Fallback: AJAX (pair_id={config.pair_id})")
                df = fetch_historical_ajax_chunked(
                    pair_id=config.pair_id,
                    start_date=start_date,
                    end_date=end_date,
                    session=session,
                    delay=delay,
                    referer_url=config.url
                )
        except Exception as e:
            logger.warning(f"  Fallback method also failed: {e}")

    # Last resort: HTML scraping
    if df.empty and config.method != FetchMethod.HTML:
        logger.info(f"  Last resort: HTML scraping")
        try:
            df = _fetch_historical_html_fallback(
                url=config.url,
                start_date=start_date,
                end_date=end_date,
                session=session,
                delay=delay
            )
        except Exception as e:
            logger.error(f"  HTML scraping also failed: {e}")

    # Validate values if requested
    if validate and not df.empty and config.expected_range:
        min_val, max_val = config.expected_range
        invalid_mask = (df['valor'] < min_val) | (df['valor'] > max_val)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.warning(
                f"  {invalid_count} values outside expected range "
                f"[{min_val}, {max_val}] for {config.name}"
            )
            # Log some examples
            invalid_samples = df[invalid_mask].head(3)
            for _, row in invalid_samples.iterrows():
                logger.warning(f"    {row['fecha']}: {row['valor']}")

    logger.info(f"  Result: {len(df)} rows")
    return df


def get_indicator_config(column: str) -> Optional[IndicatorConfig]:
    """
    Get the configuration for an indicator.

    Args:
        column: Database column name

    Returns:
        IndicatorConfig or None if not found
    """
    return INDICATOR_CONFIG.get(column)


def list_configured_indicators() -> List[Dict[str, Any]]:
    """
    List all configured indicators with their settings.

    Returns:
        List of dicts with indicator info
    """
    result = []
    for column, config in INDICATOR_CONFIG.items():
        result.append({
            'column': column,
            'name': config.name,
            'method': config.method.value,
            'pair_id': config.pair_id,
            'instrument_id': config.instrument_id,
            'url': config.url,
            'expected_range': config.expected_range,
        })
    return result


# =============================================================================
# Main - Test Execution
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("INVESTING.COM SCRAPER - TEST SUITE")
    print("="*60)

    # Test HTML scraping
    print("\n=== TEST: HTML Scraping (DXY) ===")
    df = obtener_dxy(5)
    if df is not None:
        print(df)
    else:
        print("FAILED")

    # Test date parsing
    print("\n=== TEST: Date Parsing ===")
    test_dates = [
        "Jan 22, 2026",
        "22 ene. 2026",
        "22 enero 2026",
        "01/22/2026",
        "2026-01-22",
    ]
    for d in test_dates:
        result = parse_investing_date(d)
        print(f"  '{d}' -> {result}")

    # Test AJAX fetch (small range)
    print("\n=== TEST: AJAX Fetch (DXY, 1 week) ===")
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=7)
    df = fetch_historical_ajax(
        pair_id=8827,  # DXY
        start_date=start.strftime('%m/%d/%Y'),
        end_date=end.strftime('%m/%d/%Y')
    )
    if not df.empty:
        print(df)
    else:
        print("FAILED or no data")

    print("\n" + "="*60)
    print("TESTS COMPLETED")
    print("="*60)
