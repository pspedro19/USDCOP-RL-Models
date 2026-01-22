#!/usr/bin/env python3
"""
Scraper Multi-Fuente para Datos Macro de Colombia
==================================================

Fuentes soportadas (en orden de prioridad):
1. datos.gov.co - API pública de datos abiertos Colombia (TRM, DTF)
2. Investing.com - Scraping con cloudscraper (COLCAP, TES, Gold, Coffee)
3. SUAMECA BanRep - Selenium bypass para captcha (IBR, TPM, ITCR, etc.)

Variables soportadas:
- IBR (Serie 241) - Indicador Bancario de Referencia
- TPM (Serie 59) - Tasa de Política Monetaria
- ITCR (Serie 4170) - Índice Tasa de Cambio Real
- TOT (Serie 4180) - Términos de Intercambio
- RESINT (Serie 15051) - Reservas Internacionales
- IPCCOL (Serie 100002) - IPC Colombia

Estado de fuentes (2025-12-18):
- datos.gov.co TRM: ✅ Funcionando
- Investing.com: ✅ Funcionando (COLCAP, TES, Gold, Coffee)
- SUAMECA BanRep: ⚠️ URLs retornan 404 - necesita Selenium

Instalación requerida:
    pip install cloudscraper beautifulsoup4 pandas psycopg2-binary
    # Opcional para Selenium bypass:
    pip install undetected-chromedriver selenium seleniumbase
"""
import os
import sys
import time
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import logging
import re

try:
    import cloudscraper
    from bs4 import BeautifulSoup
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de series SUAMECA - URLs completas verificadas 2025-12-18
BANREP_SERIES = {
    'IBR': {
        'serie_id': '241',
        'db_column': 'finc_rate_ibr_overnight_col_d_ibr',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/241/tasas_interes_indicador_bancario_referencia_ibr',
        'frecuencia': 'D',
        'descripcion': 'Indicador Bancario de Referencia Overnight'
    },
    'TPM': {
        'serie_id': '59',
        'db_column': 'polr_policy_rate_col_d_tpm',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/59/tasas_interes_politica_monetaria',
        'frecuencia': 'D',
        'descripcion': 'Tasa de Política Monetaria'
    },
    'ITCR': {
        'serie_id': '4170',
        'db_column': 'fxrt_reer_bilateral_col_m_itcr',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4170/indice_tasa_cambio_real_itcr',
        'frecuencia': 'M',
        'descripcion': 'Índice de Tasa de Cambio Real'
    },
    'TOT': {
        'serie_id': '4180',
        'db_column': 'ftrd_terms_trade_col_m_tot',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4180/indice_terminos_intercambio_bienes',
        'frecuencia': 'M',
        'descripcion': 'Términos de Intercambio'
    },
    'RESINT': {
        'serie_id': '15051',
        'db_column': 'rsbp_reserves_international_col_m_resint',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/15051/reservas_internacionales',
        'frecuencia': 'M',
        'descripcion': 'Reservas Internacionales'
    },
    'IPCCOL': {
        'serie_id': '100002',
        'db_column': 'infl_cpi_total_col_m_ipccol',
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/100002/ipc',
        'frecuencia': 'M',
        'descripcion': 'IPC Colombia'
    },
}


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'usdcop-postgres-timescale'),
        port=int(os.environ.get('POSTGRES_PORT', 5432)),
        database=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=os.environ.get('POSTGRES_PASSWORD', 'admin123')
    )


def setup_undetected_chrome():
    """Setup undetected Chrome driver to bypass Radware captcha."""
    try:
        import undetected_chromedriver as uc

        options = uc.ChromeOptions()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--lang=es-CO')

        driver = uc.Chrome(options=options, version_main=None)
        logger.info("Undetected Chrome driver initialized")
        return driver

    except ImportError:
        logger.error("undetected-chromedriver not installed. Run: pip install undetected-chromedriver")
        return None
    except Exception as e:
        logger.error(f"Error setting up Chrome: {e}")
        return None


def setup_seleniumbase():
    """Alternative: Setup SeleniumBase UC mode."""
    try:
        from seleniumbase import Driver

        driver = Driver(uc=True, headless=True)
        logger.info("SeleniumBase UC driver initialized")
        return driver

    except ImportError:
        logger.warning("SeleniumBase not installed. Run: pip install seleniumbase")
        return None
    except Exception as e:
        logger.error(f"Error setting up SeleniumBase: {e}")
        return None


def scrape_banrep_serie(driver, serie_config: dict, days: int = 60, download_dir: str = '/tmp') -> list:
    """
    Scrape a BanRep series using Selenium.

    Proceso:
    1. Navegar a la URL de la serie
    2. Esperar que cargue la página (bypass de Radware si es necesario)
    3. Buscar y hacer clic en el botón de descarga/tabla encima de la gráfica
    4. Extraer datos de la tabla o del archivo descargado

    Args:
        driver: Selenium WebDriver
        serie_config: Configuration dict for the series
        days: Number of days of data to fetch
        download_dir: Directory for downloaded files

    Returns:
        List of dicts with 'fecha' and 'valor'
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains

    url = serie_config['url']
    serie_id = serie_config['serie_id']

    logger.info(f"Scraping {serie_config['descripcion']} (Serie {serie_id})...")
    logger.info(f"URL: {url}")

    try:
        driver.get(url)
        time.sleep(5)  # Wait for initial page load

        # Check if we hit captcha
        page_source = driver.page_source.lower()
        if 'captcha' in page_source or 'radware' in page_source or 'challenge' in page_source:
            logger.warning("Captcha/Radware detected! Waiting for bypass...")
            time.sleep(15)  # Give undetected-chromedriver time to bypass
            page_source = driver.page_source.lower()

            if 'captcha' in page_source or 'radware' in page_source:
                logger.error("Could not bypass captcha - may need manual intervention")
                return []

        # Wait for page content to load
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except:
            logger.error("Page did not load properly")
            return []

        logger.info("Page loaded, looking for data controls...")

        # Strategy 1: Look for "Vista tabla" or table view button
        data = []

        try:
            # Find buttons above the graph
            buttons = driver.find_elements(By.TAG_NAME, "button")
            icons = driver.find_elements(By.TAG_NAME, "mat-icon")

            logger.info(f"Found {len(buttons)} buttons and {len(icons)} icons")

            # Look for table/grid icon or download button
            for btn in buttons:
                btn_text = btn.text.lower()
                btn_class = btn.get_attribute('class') or ''
                btn_title = btn.get_attribute('title') or ''
                aria_label = btn.get_attribute('aria-label') or ''

                # Check for table view button
                if any(x in btn_text + btn_class + btn_title + aria_label for x in
                       ['tabla', 'table', 'grid', 'list', 'datos', 'ver datos']):
                    logger.info(f"Clicking table view button: {btn_text or btn_title or 'icon'}")
                    try:
                        btn.click()
                        time.sleep(3)
                        break
                    except:
                        ActionChains(driver).move_to_element(btn).click().perform()
                        time.sleep(3)
                        break

            # Look for mat-icon buttons (Angular Material icons)
            for icon in icons:
                icon_text = icon.text.lower()
                if any(x in icon_text for x in ['table', 'grid', 'list', 'view_list', 'table_chart']):
                    logger.info(f"Clicking icon: {icon_text}")
                    try:
                        parent = icon.find_element(By.XPATH, '..')
                        parent.click()
                        time.sleep(3)
                        break
                    except:
                        pass

        except Exception as e:
            logger.warning(f"Error looking for table button: {e}")

        # Strategy 2: Try to find and extract data from table
        time.sleep(2)  # Wait for any table to render

        tables = driver.find_elements(By.TAG_NAME, "table")
        logger.info(f"Found {len(tables)} tables")

        # Find the largest table (data table)
        data_table = None
        max_rows = 0
        for table in tables:
            rows = table.find_elements(By.TAG_NAME, "tr")
            if len(rows) > max_rows:
                max_rows = len(rows)
                data_table = table

        if data_table and max_rows > 10:
            logger.info(f"Processing data table with {max_rows} rows")
            rows = data_table.find_elements(By.TAG_NAME, "tr")

            # SUAMECA table structure:
            # Row 0: Header row (<th> elements)
            # Row 1+: Data rows where:
            #   - Date is in <th> element (first column)
            #   - Value is in first <td> element (second column)

            for row in rows[1:]:  # Skip header row
                try:
                    # Get date from <th> element
                    th_elements = row.find_elements(By.TAG_NAME, "th")
                    td_elements = row.find_elements(By.TAG_NAME, "td")

                    if not th_elements or not td_elements:
                        continue

                    fecha_str = th_elements[0].text.strip()
                    valor_str = td_elements[0].text.strip()

                    # Clean value string - handle different formats
                    valor_str = valor_str.replace(',', '.').replace('%', '').replace(' ', '').replace('\n', '')

                    # Skip if empty
                    if not fecha_str or not valor_str:
                        continue

                    # Skip if value is not numeric
                    try:
                        valor = float(valor_str)
                    except:
                        continue

                    # Parse date - BanRep uses YYYY/MM/DD format
                    fecha = None

                    # Try exact formats first (most common in BanRep)
                    for fmt in ['%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']:
                        try:
                            fecha = datetime.strptime(fecha_str, fmt).date()
                            break
                        except:
                            continue

                    # Try pandas for more flexible parsing
                    if fecha is None:
                        try:
                            fecha = pd.to_datetime(fecha_str).date()
                        except:
                            continue

                    if fecha is None:
                        continue

                    # Validate value is reasonable for this type of data
                    if -100000 < valor < 10000000:
                        data.append({'fecha': fecha, 'valor': valor})

                except (ValueError, IndexError) as e:
                    continue

            logger.info(f"Extracted {len(data)} records from table")

        # Strategy 3: Look for download button and parse CSV/Excel
        if not data:
            logger.info("No table data found, looking for download option...")
            try:
                # Look for download buttons
                download_btns = driver.find_elements(By.XPATH,
                    "//button[contains(@class, 'download') or contains(@title, 'descargar') or contains(@aria-label, 'download')]")

                for btn in download_btns:
                    logger.info(f"Found download button: {btn.get_attribute('title') or btn.text}")

            except Exception as e:
                logger.warning(f"Error looking for download: {e}")

        # Process results
        if data:
            data = sorted(data, key=lambda x: x['fecha'], reverse=True)[:days]
            logger.info(f"Found {len(data)} records")
            if data:
                logger.info(f"Latest: {data[0]['fecha']} = {data[0]['valor']}")
                logger.info(f"Oldest: {data[-1]['fecha']} = {data[-1]['valor']}")
        else:
            logger.warning("No data extracted from page")
            # Save screenshot for debugging
            try:
                screenshot_path = f"/tmp/banrep_{serie_id}_debug.png"
                driver.save_screenshot(screenshot_path)
                logger.info(f"Debug screenshot saved: {screenshot_path}")
            except:
                pass

        return data

    except Exception as e:
        logger.error(f"Error scraping {serie_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def fetch_from_datos_gov(serie_name: str) -> list:
    """
    Fetch from datos.gov.co API (no captcha).
    Currently supports: TRM (DTF was removed from datos.gov.co)
    """
    import requests

    # datos.gov.co datasets - verified 2025-12-18
    DATOS_GOV_ENDPOINTS = {
        'TRM': 'https://www.datos.gov.co/resource/mcec-87by.json',
    }

    if serie_name not in DATOS_GOV_ENDPOINTS:
        return []

    url = DATOS_GOV_ENDPOINTS[serie_name]

    try:
        params = {
            '$limit': 100,
            '$order': 'vigenciadesde DESC'
        }

        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            result = []

            for item in data:
                try:
                    fecha = pd.to_datetime(item.get('vigenciadesde', item.get('fecha'))).date()
                    valor = float(item.get('valor', 0))
                    result.append({'fecha': fecha, 'valor': valor})
                except:
                    continue

            logger.info(f"datos.gov.co: Found {len(result)} records for {serie_name}")
            return result
        else:
            logger.warning(f"datos.gov.co returned {resp.status_code}")
            return []

    except Exception as e:
        logger.error(f"Error fetching from datos.gov.co: {e}")
        return []


def fetch_from_investing(serie_name: str, days: int = 60) -> list:
    """
    Fetch historical data from Investing.com using cloudscraper.

    Supports: COLCAP, TES_10Y, TES_5Y, Gold, Coffee
    """
    if not CLOUDSCRAPER_AVAILABLE:
        logger.warning("cloudscraper not installed - Investing.com source unavailable")
        return []

    # Investing.com URLs verified working 2025-12-18
    INVESTING_URLS = {
        'COLCAP': 'https://www.investing.com/indices/colcap-historical-data',
        'TES_10Y': 'https://www.investing.com/rates-bonds/colombia-10-year-bond-yield-historical-data',
        'TES_5Y': 'https://www.investing.com/rates-bonds/colombia-5-year-bond-yield-historical-data',
        'Gold': 'https://www.investing.com/commodities/gold-historical-data',
        'Coffee': 'https://www.investing.com/commodities/us-coffee-c-historical-data',
    }

    if serie_name not in INVESTING_URLS:
        return []

    url = INVESTING_URLS[serie_name]

    try:
        scraper = cloudscraper.create_scraper()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
        }

        resp = scraper.get(url, headers=headers, timeout=25)

        if resp.status_code != 200:
            logger.warning(f"Investing.com returned {resp.status_code} for {serie_name}")
            return []

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Find the historical data table
        table = soup.find('table', class_='freeze-column-w-1')
        if not table:
            tables = soup.find_all('table')
            if tables:
                table = max(tables, key=lambda t: len(str(t)))

        if not table:
            logger.warning(f"No table found for {serie_name}")
            return []

        rows = table.find_all('tr')[1:days+10]
        data = []

        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                try:
                    fecha_str = cols[0].get_text(strip=True)
                    valor_str = cols[1].get_text(strip=True).replace(',', '')

                    fecha = pd.to_datetime(fecha_str).date()
                    valor = float(valor_str)

                    data.append({'fecha': fecha, 'valor': valor})
                except (ValueError, AttributeError):
                    continue

        if data:
            logger.info(f"Investing.com: Found {len(data)} records for {serie_name}")
            if data:
                logger.info(f"Latest: {data[0]['fecha']} = {data[0]['valor']}")

        return data

    except Exception as e:
        logger.error(f"Error fetching {serie_name} from Investing.com: {e}")
        return []


def apply_forward_fill(column: str, days: int = 30) -> int:
    """
    Apply forward fill for missing values in a column.
    Useful for IBR and other variables when source is unavailable.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Get the last known value
        cur.execute(f"""
            SELECT fecha, {column}
            FROM macro_indicators_daily
            WHERE {column} IS NOT NULL
            ORDER BY fecha DESC
            LIMIT 1
        """)

        result = cur.fetchone()
        if not result:
            logger.warning(f"No data found for {column} to forward fill")
            return 0

        last_date, last_value = result
        logger.info(f"Last known value for {column}: {last_date} = {last_value}")

        # Fill missing dates with the last known value
        cur.execute(f"""
            WITH dates_to_fill AS (
                SELECT fecha
                FROM macro_indicators_daily
                WHERE fecha > %s
                  AND fecha <= CURRENT_DATE
                  AND {column} IS NULL
            )
            UPDATE macro_indicators_daily m
            SET {column} = %s, updated_at = NOW()
            FROM dates_to_fill d
            WHERE m.fecha = d.fecha
        """, [last_date, last_value])

        updated = cur.rowcount
        conn.commit()

        if updated > 0:
            logger.info(f"Forward-filled {updated} rows for {column} with value {last_value}")

        return updated

    except Exception as e:
        logger.error(f"Error applying forward fill for {column}: {e}")
        conn.rollback()
        return 0

    finally:
        cur.close()
        conn.close()


def update_database(column: str, data: list) -> int:
    """Update database with fetched data."""
    if not data:
        return 0

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        updated = 0

        for item in data:
            cur.execute(f"""
                INSERT INTO macro_indicators_daily (fecha, {column})
                VALUES (%s, %s)
                ON CONFLICT (fecha) DO UPDATE SET
                    {column} = EXCLUDED.{column},
                    updated_at = NOW()
                WHERE macro_indicators_daily.{column} IS NULL
                   OR macro_indicators_daily.{column} != EXCLUDED.{column}
            """, [item['fecha'], item['valor']])

            if cur.rowcount > 0:
                updated += 1

        conn.commit()
        return updated

    except Exception as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
        return 0

    finally:
        cur.close()
        conn.close()


def main():
    """
    Main function - BanRep scraper usando Selenium.

    Este scraper usa Selenium con undetected-chromedriver para:
    1. Navegar a las páginas de SUAMECA BanRep
    2. Bypass de Radware Bot Manager (captcha)
    3. Hacer clic en el botón de tabla/descarga
    4. Extraer los datos

    Variables BanRep soportadas:
    - IBR (Serie 241): Indicador Bancario de Referencia
    - TPM (Serie 59): Tasa de Política Monetaria
    - ITCR (Serie 4170): Índice Tasa de Cambio Real
    - TOT (Serie 4180): Términos de Intercambio
    - RESINT (Serie 15051): Reservas Internacionales
    - IPCCOL (Serie 100002): IPC Colombia
    """
    print("=" * 70)
    print("BANCO DE LA REPÚBLICA - SELENIUM SCRAPER")
    print("Con bypass de Radware Bot Manager (undetected-chromedriver)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_updated = 0

    # =========================================================================
    # PART 1: BANREP SUAMECA via Selenium
    # =========================================================================

    print("\n[PART 1] SUAMECA BANREP (Selenium)")
    print("-" * 50)

    # Setup Selenium driver
    driver = setup_undetected_chrome()

    if driver is None:
        print("  ⚠️ undetected-chromedriver failed, trying SeleniumBase...")
        driver = setup_seleniumbase()

    if driver is None:
        print("\n[ERROR] No se pudo inicializar Selenium")
        print("Instala uno de estos paquetes:")
        print("  pip install undetected-chromedriver")
        print("  pip install seleniumbase")
        print("\nAplicando forward-fill como fallback...")

        # Fallback to forward-fill
        FFILL_COLUMNS = [
            ('IBR', 'finc_rate_ibr_overnight_col_d_ibr'),
            ('TPM', 'polr_policy_rate_col_d_tpm'),
        ]
        for name, column in FFILL_COLUMNS:
            print(f"\n[{name}] Forward-fill")
            updated = apply_forward_fill(column, days=30)
            total_updated += updated
            if updated > 0:
                print(f"  ✅ Forward-filled {updated} rows")
    else:
        try:
            # Scrape each BanRep series
            for serie_name, config in BANREP_SERIES.items():
                print(f"\n[{serie_name}] {config['descripcion']}")
                print(f"  Serie SUAMECA: {config['serie_id']}")
                print(f"  URL: {config['url']}")

                data = scrape_banrep_serie(driver, config, days=60)

                if data:
                    updated = update_database(config['db_column'], data)
                    total_updated += updated
                    print(f"  ✅ Updated {updated} rows")
                else:
                    print(f"  ⚠️ No data extracted, applying forward-fill...")
                    updated = apply_forward_fill(config['db_column'], days=30)
                    total_updated += updated
                    if updated > 0:
                        print(f"  ✅ Forward-filled {updated} rows")

        except Exception as e:
            print(f"\n[ERROR] Selenium error: {e}")
        finally:
            print("\nCerrando navegador...")
            driver.quit()

    # =========================================================================
    # PART 2: Optional - Investing.com for additional data
    # =========================================================================

    include_investing = os.environ.get('INCLUDE_INVESTING', 'false').lower() == 'true'

    if include_investing:
        print("\n[PART 2] INVESTING.COM (Opcional)")
        print("-" * 50)

        INVESTING_TO_DB = {
            'COLCAP': 'eqty_index_colcap_col_d_colcap',
            'TES_10Y': 'finc_bond_yield10y_col_d_col10y',
            'TES_5Y': 'finc_bond_yield5y_col_d_col5y',
            'Gold': 'comm_metal_gold_glb_d_gold',
            'Coffee': 'comm_agri_coffee_glb_d_coffee',
        }

        for source_name, db_column in INVESTING_TO_DB.items():
            print(f"\n[{source_name}] via Investing.com")
            data = fetch_from_investing(source_name, days=60)
            if data:
                updated = update_database(db_column, data)
                total_updated += updated
                print(f"  ✅ Updated {updated} rows")
            else:
                print(f"  ⚠️ No data fetched")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print(f"TOTAL ROWS UPDATED: {total_updated}")
    print("=" * 70)
    print("\nVariables BanRep (SUAMECA):")
    for name, config in BANREP_SERIES.items():
        print(f"  - {name}: {config['descripcion']}")
    print("\nUso:")
    print("  python scraper_banrep_selenium.py")
    print("  INCLUDE_INVESTING=true python scraper_banrep_selenium.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
