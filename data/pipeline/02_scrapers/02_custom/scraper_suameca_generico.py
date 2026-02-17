# -*- coding: utf-8 -*-
"""
SUAMECA Generic Scraper - BanRep
================================

Provides functions for extracting data from SUAMECA/BanRep:
- IBR Overnight (REST API - serie 241)
- TPM - Tasa de Politica Monetaria (REST API - serie 59)
- ITCR - Indice Tasa Cambio Real (Selenium - serie 4170)
- TOT - Terminos de Intercambio (Selenium - serie 4180)
- IPCCOL - IPC Colombia (Selenium - serie 100002)
- RESINT - Reservas Internacionales (REST API - serie 1060)
- PRIME - Tasa Prima (REST API returns empty, use FRED DPRIME)

API discovered via Angular SPA inspection:
    GET https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/
        estadisticaEconomicaRestService/consultaInformacionSerie?idSerie=XXX

For series that don't work via REST API (return empty), uses Selenium.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# REST API Config
API_BASE = (
    "https://suameca.banrep.gov.co/estadisticas-economicas-back/rest/"
    "estadisticaEconomicaRestService"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "es-CO,es;q=0.9,en;q=0.8",
}

# Series configuration
SERIES_CONFIG = {
    'ibr': {'serie_id': 241, 'method': 'rest', 'name': 'IBR Overnight'},
    'tpm': {'serie_id': 59, 'method': 'rest', 'name': 'TPM - Tasa Politica Monetaria'},
    'resint': {
        'serie_id': 15051,  # Serie actualizada
        'method': 'selenium',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/15051/reservas_internacionales',
        'name': 'Reservas Internacionales',
        'use_vista_tabla': True
    },
    'prime': {'serie_id': 220001, 'method': 'rest', 'name': 'Prime Rate Colombia'},  # Usually empty
    'itcr': {
        'serie_id': 4170,
        'method': 'selenium',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4170/indice_tasa_cambio_real_itcr',
        'name': 'ITCR (Tasa Cambio Real)',
        'use_vista_tabla': True
    },
    'tot': {
        'serie_id': 4180,
        'method': 'selenium',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4180/indice_terminos_intercambio_bienes',
        'name': 'TOT (Terminos Intercambio)',
        'use_vista_tabla': True
    },
    'ipccol': {
        'serie_id': 100002,
        'method': 'selenium',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/100002/ipc',
        'name': 'IPC Colombia',
        'use_vista_tabla': True
    },
    'cacct': {
        'serie_id': 414001,
        'method': 'selenium',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/414001/cuentacorriente',
        'name': 'Cuenta Corriente',
        'use_vista_tabla': True
    },
}


def _fetch_rest_api(serie_id: int, n: int = 20) -> Optional[pd.DataFrame]:
    """Fetch data from SUAMECA REST API."""
    try:
        url = f"{API_BASE}/consultaInformacionSerie"
        params = {"idSerie": serie_id}

        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Parse response
        if isinstance(data, list):
            if len(data) == 0:
                logger.warning(f"[SUAMECA REST] Serie {serie_id} returned empty list")
                return None
            first_elem = data[0]
            if isinstance(first_elem, dict):
                serie_data = first_elem.get('data', [])
            else:
                return None
        elif isinstance(data, dict):
            serie_data = data.get('data', [])
        else:
            return None

        if not serie_data:
            return None

        records = []
        for item in serie_data:
            if isinstance(item, list) and len(item) >= 2:
                timestamp_ms = item[0]
                value = item[1]
                dt = pd.to_datetime(timestamp_ms, unit='ms')
                records.append({
                    "fecha": dt.normalize(),
                    "valor": float(value) if value is not None else None
                })

        if not records:
            return None

        df = pd.DataFrame(records)
        df = df.dropna(subset=['valor'])
        df = df.drop_duplicates(subset=['fecha'])
        df = df.sort_values("fecha", ascending=False).reset_index(drop=True)

        # Return last n records
        return df.head(n)

    except Exception as e:
        logger.error(f"[SUAMECA REST] Error fetching serie {serie_id}: {e}")
        return None


def _fetch_selenium(url: str, serie_name: str, n: int = 20, headless: bool = True, use_vista_tabla: bool = False) -> Optional[pd.DataFrame]:
    """Fetch data from SUAMECA using Selenium (for series that don't work via REST).

    Args:
        url: URL de la serie SUAMECA
        serie_name: Nombre para logging
        n: Número de registros a retornar (0 = todos)
        headless: Ejecutar Chrome sin ventana
        use_vista_tabla: Si True, hace click en botón "Vista tabla" (id=vistaTabla)
    """
    try:
        import undetected_chromedriver as uc
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.action_chains import ActionChains
    except ImportError:
        logger.error("[SUAMECA] Selenium/undetected_chromedriver not installed")
        return None

    driver = None
    try:
        # Setup Chrome
        options = uc.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--lang=es-CO')

        driver = uc.Chrome(options=options, version_main=None)
        logger.info(f"[SUAMECA SELENIUM] Fetching {serie_name} from {url}")

        driver.get(url)
        time.sleep(8)

        # Check for captcha
        page_source = driver.page_source.lower()
        if 'captcha' in page_source or 'radware' in page_source or 'challenge' in page_source:
            logger.warning(f"[SUAMECA] Captcha detected, waiting...")
            time.sleep(20)

        # Wait for Angular
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "app-root"))
        )
        time.sleep(3)

        # Click "Vista tabla" button if required (IPCCOL, ITCR, TOT, CACCT, RESINT)
        table_clicked = False
        if use_vista_tabla:
            try:
                # Método 1: Buscar por ID exacto "vistaTabla"
                vista_tabla_btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "vistaTabla"))
                )
                ActionChains(driver).move_to_element(vista_tabla_btn).click().perform()
                table_clicked = True
                logger.info(f"[SUAMECA SELENIUM] Clicked 'Vista tabla' button for {serie_name}")
                time.sleep(5)
            except Exception as e:
                logger.warning(f"[SUAMECA SELENIUM] Could not find vistaTabla button by ID: {e}")
                # Método 2: Buscar por texto del botón
                try:
                    buttons = driver.find_elements(By.TAG_NAME, "button")
                    for btn in buttons:
                        btn_text = btn.text.lower().strip()
                        if 'vista tabla' in btn_text:
                            ActionChains(driver).move_to_element(btn).click().perform()
                            table_clicked = True
                            logger.info(f"[SUAMECA SELENIUM] Clicked 'Vista tabla' by text")
                            time.sleep(5)
                            break
                except Exception as e2:
                    logger.warning(f"[SUAMECA SELENIUM] Fallback search failed: {e2}")

        # Fallback: buscar iconos mat-icon (método original)
        if not table_clicked:
            try:
                buttons = driver.find_elements(By.TAG_NAME, "button")
                for btn in buttons:
                    icons = btn.find_elements(By.TAG_NAME, "mat-icon")
                    for icon in icons:
                        icon_text = icon.text.lower()
                        if any(x in icon_text for x in ['table', 'list', 'grid', 'view_list', 'table_chart']):
                            ActionChains(driver).move_to_element(btn).click().perform()
                            time.sleep(3)
                            break
            except:
                pass

        time.sleep(5)

        # Find data table
        tables = driver.find_elements(By.TAG_NAME, "table")
        data_table = None
        max_rows = 0
        for table in tables:
            rows = table.find_elements(By.TAG_NAME, "tr")
            if len(rows) > max_rows:
                max_rows = len(rows)
                data_table = table

        if not data_table or max_rows < 5:
            logger.warning(f"[SUAMECA SELENIUM] No table found for {serie_name}")
            return None

        # Extract data
        data = []
        rows = data_table.find_elements(By.TAG_NAME, "tr")

        for row in rows[1:]:  # Skip header
            try:
                th_elements = row.find_elements(By.TAG_NAME, "th")
                td_elements = row.find_elements(By.TAG_NAME, "td")

                if not th_elements or not td_elements:
                    continue

                fecha_str = th_elements[0].text.strip()
                valor_str = td_elements[0].text.strip()
                valor_str = valor_str.replace(',', '.').replace('%', '').replace(' ', '').replace('\n', '')

                if not fecha_str or not valor_str:
                    continue

                try:
                    valor = float(valor_str)
                except:
                    continue

                # Parse date
                fecha = None
                for fmt in ['%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y']:
                    try:
                        fecha = datetime.strptime(fecha_str, fmt).date()
                        break
                    except:
                        continue

                if fecha is None:
                    try:
                        fecha = pd.to_datetime(fecha_str).date()
                    except:
                        continue

                if fecha:
                    data.append({'fecha': pd.Timestamp(fecha), 'valor': valor})

            except:
                continue

        if not data:
            return None

        df = pd.DataFrame(data)
        df = df.sort_values('fecha', ascending=False).reset_index(drop=True)
        logger.info(f"[SUAMECA SELENIUM] Extracted {len(df)} records for {serie_name}")

        return df.head(n)

    except Exception as e:
        logger.error(f"[SUAMECA SELENIUM] Error fetching {serie_name}: {e}")
        return None

    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def _get_data(serie_key: str, n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get data for a SUAMECA series."""
    config = SERIES_CONFIG.get(serie_key.lower())
    if not config:
        logger.error(f"[SUAMECA] Unknown series: {serie_key}")
        return None

    use_vista_tabla = config.get('use_vista_tabla', False)

    if config['method'] == 'rest':
        df = _fetch_rest_api(config['serie_id'], n)
        if df is not None and not df.empty:
            return df

        # If REST fails and there's a selenium fallback URL, try that
        if 'url' in config:
            return _fetch_selenium(config['url'], config['name'], n, headless, use_vista_tabla)
        return None

    elif config['method'] == 'selenium':
        return _fetch_selenium(config['url'], config['name'], n, headless, use_vista_tabla)

    return None


# ============================================================================
# Public API Functions (called by HPC V3)
# ============================================================================

def obtener_ibr(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get IBR Overnight rate."""
    return _get_data('ibr', n, headless)


def obtener_tpm(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get TPM - Tasa de Politica Monetaria."""
    return _get_data('tpm', n, headless)


def obtener_prime(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get Prime Rate Colombia (usually returns empty from REST API)."""
    return _get_data('prime', n, headless)


def obtener_itcr(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get ITCR - Indice Tasa Cambio Real (uses Selenium)."""
    return _get_data('itcr', n, headless)


def obtener_reservas(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get Reservas Internacionales."""
    return _get_data('resint', n, headless)


def obtener_terminos_intercambio(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get TOT - Terminos de Intercambio (uses Selenium)."""
    return _get_data('tot', n, headless)


def obtener_cuenta_corriente(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get Cuenta Corriente (uses Selenium)."""
    return _get_data('cacct', n, headless)


def obtener_ipc_colombia(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get IPC Colombia (uses Selenium)."""
    return _get_data('ipccol', n, headless)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    print("=" * 70)
    print("SUAMECA SCRAPER TEST")
    print("=" * 70)

    # Test REST API series
    for serie in ['ibr', 'tpm', 'resint']:
        print(f"\n[TEST] {serie.upper()}")
        df = _get_data(serie, n=5)
        if df is not None and not df.empty:
            print(f"  OK - {len(df)} records")
            print(f"  Latest: {df.iloc[0]['fecha']} = {df.iloc[0]['valor']}")
        else:
            print(f"  FAILED - No data")

    # Test Selenium series (slower)
    print("\n[INFO] Selenium series (ITCR, TOT, IPCCOL) require Chrome...")
    for serie in ['itcr', 'tot', 'ipccol']:
        print(f"\n[TEST] {serie.upper()}")
        df = _get_data(serie, n=5, headless=True)
        if df is not None and not df.empty:
            print(f"  OK - {len(df)} records")
            print(f"  Latest: {df.iloc[0]['fecha']} = {df.iloc[0]['valor']}")
        else:
            print(f"  FAILED - No data")
