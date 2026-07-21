# -*- coding: utf-8 -*-
"""
SUAMECA V4 Scraper - Balance of Payments / FDI Variables
=========================================================

Extracts quarterly Balance of Payments data from SUAMECA/BanRep:
- IED - Inversion Extranjera Directa (FDI Inflows) - Serie 414012
- IDCE - Inversion Directa Colombia al Exterior (FDI Outflows) - Serie 414013
- CACCT - Cuenta Corriente - Serie 414001
- RESINT - Reservas Internacionales (also via REST API)
- ITCR - Indice Tasa Cambio Real
- TOT - Terminos de Intercambio
- IPCCOL - IPC Colombia

Uses Selenium for web scraping since REST API returns empty for these series.

Balance of Payments URL pattern:
    https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/{serie_id}
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# URLs del graficador-interactivo (descarga CSV - más rápido que Vista Tabla)
GRAFICADOR_URLS = {
    'fdiin': "https://suameca.banrep.gov.co/graficador-interactivo/grafica/15133",
    'fdiout': "https://suameca.banrep.gov.co/graficador-interactivo/grafica/15134",
    'itcr': "https://suameca.banrep.gov.co/graficador-interactivo/grafica/234",      # ITCR ponderaciones totales
    'itcr_usa': "https://suameca.banrep.gov.co/graficador-interactivo/grafica/219",  # ITCR bilateral con USA
}

# Series configuration
BOP_SERIES = {
    'ied': {
        'serie_id': 15133,
        'url': GRAFICADOR_URLS['fdiin'],
        'name': 'IED - Inversion Extranjera Directa en Colombia',
        'frequency': 'Q',
        'use_graficador': True  # Estrategia: graficador-interactivo + descarga CSV
    },
    'idce': {
        'serie_id': 15134,
        'url': GRAFICADOR_URLS['fdiout'],
        'name': 'IDCE - Inversion Directa Colombia al Exterior',
        'frequency': 'Q',
        'use_graficador': True  # Estrategia: graficador-interactivo + descarga CSV
    },
    'cacct': {
        'serie_id': 414001,
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/414001/cuentacorriente',
        'name': 'Cuenta Corriente',
        'frequency': 'Q',
        'use_vista_tabla': True
    },
    'resint': {
        'serie_id': 15051,  # Serie actualizada
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/15051/reservas_internacionales',
        'name': 'Reservas Internacionales',
        'frequency': 'M',
        'use_vista_tabla': True
    },
    'itcr': {
        'serie_id': 234,
        'url': GRAFICADOR_URLS['itcr'],
        'url_fallback': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4170/indice_tasa_cambio_real_itcr',
        'name': 'ITCR - Ponderaciones Totales',
        'frequency': 'M',
        'use_graficador': True,  # Primario: Graficador (más rápido)
        'use_vista_tabla': True   # Fallback: Vista Tabla
    },
    'itcr_usa': {
        'serie_id': 219,
        'url': GRAFICADOR_URLS['itcr_usa'],
        'name': 'ITCR - Bilateral USA',
        'frequency': 'M',
        'use_graficador': True
    },
    'tot': {
        'serie_id': 4180,
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4180/indice_terminos_intercambio_bienes',
        'name': 'TOT',
        'frequency': 'M',
        'use_vista_tabla': True
    },
    'ipccol': {
        'serie_id': 100002,
        'url': 'https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/100002/ipc',
        'name': 'IPC Colombia',
        'frequency': 'M',
        'use_vista_tabla': True
    },
}


def _setup_driver(headless: bool = True):
    """Setup Chrome driver with undetected-chromedriver."""
    try:
        import undetected_chromedriver as uc

        options = uc.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--lang=es-CO')

        driver = uc.Chrome(options=options, version_main=None)
        return driver
    except ImportError:
        logger.error("[SUAMECA V4] undetected_chromedriver not installed")
        return None
    except Exception as e:
        logger.error(f"[SUAMECA V4] Chrome setup failed: {e}")
        return None


def _scrape_serie(url: str, serie_name: str, n: int = 20, headless: bool = True, use_vista_tabla: bool = False) -> Optional[pd.DataFrame]:
    """Scrape a single SUAMECA series using Selenium.

    Args:
        url: URL de la serie SUAMECA
        serie_name: Nombre para logging
        n: Número de registros a retornar (0 = todos)
        headless: Ejecutar Chrome sin ventana
        use_vista_tabla: Si True, hace click en botón "Vista tabla" (id=vistaTabla)
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains

    driver = _setup_driver(headless)
    if driver is None:
        return None

    try:
        logger.info(f"[SUAMECA V4] Fetching {serie_name}...")
        driver.get(url)
        time.sleep(8)

        # Check for captcha
        page_source = driver.page_source.lower()
        if 'captcha' in page_source or 'radware' in page_source:
            logger.warning(f"[SUAMECA V4] Captcha detected for {serie_name}, waiting...")
            time.sleep(25)

        # Wait for Angular app
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
                logger.info(f"[SUAMECA V4] Clicked 'Vista tabla' button for {serie_name}")
                # Algunas series (RESINT, ITCR) necesitan más tiempo para cargar
                time.sleep(20)  # Esperar a que cargue la tabla completa
            except Exception as e:
                logger.warning(f"[SUAMECA V4] Could not find vistaTabla button by ID: {e}")
                # Método 2: Buscar por texto del botón
                try:
                    buttons = driver.find_elements(By.TAG_NAME, "button")
                    for btn in buttons:
                        btn_text = btn.text.lower().strip()
                        btn_class = btn.get_attribute('class') or ''
                        if 'vista tabla' in btn_text or 'vistaTabla' in btn_class:
                            ActionChains(driver).move_to_element(btn).click().perform()
                            table_clicked = True
                            logger.info(f"[SUAMECA V4] Clicked 'Vista tabla' by text for {serie_name}")
                            time.sleep(5)
                            break
                except Exception as e2:
                    logger.warning(f"[SUAMECA V4] Fallback button search failed: {e2}")

        # Fallback: buscar iconos de tabla (método original para FDI)
        if not table_clicked and not use_vista_tabla:
            try:
                buttons = driver.find_elements(By.TAG_NAME, "button")
                for btn in buttons:
                    try:
                        icons = btn.find_elements(By.TAG_NAME, "mat-icon")
                        for icon in icons:
                            icon_text = icon.text.lower()
                            if any(x in icon_text for x in ['table', 'list', 'grid', 'view_list', 'table_chart', 'format_list']):
                                ActionChains(driver).move_to_element(btn).click().perform()
                                table_clicked = True
                                time.sleep(3)
                                break
                    except:
                        pass
                    if table_clicked:
                        break
            except Exception as e:
                logger.debug(f"[SUAMECA V4] Table button click failed: {e}")

        time.sleep(5)

        # Find data table (buscar tabla con más filas, puede haber varias)
        # Intentar varias veces por si la tabla aún está cargando
        data_table = None
        max_rows = 0

        for attempt in range(5):
            tables = driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                if len(rows) > max_rows:
                    max_rows = len(rows)
                    data_table = table

            if max_rows >= 10:  # Esperar hasta tener al menos 10 filas
                break

            # Esperar más si la tabla tiene pocas filas
            logger.info(f"[SUAMECA V4] Table has only {max_rows} rows, waiting... (attempt {attempt+1}/5)")
            time.sleep(8)

        if not data_table or max_rows < 5:
            logger.warning(f"[SUAMECA V4] No suitable table found for {serie_name} (max_rows={max_rows})")
            return None

        logger.info(f"[SUAMECA V4] Found table with {max_rows} rows")

        # Extract data
        data = []
        rows = data_table.find_elements(By.TAG_NAME, "tr")

        for row in rows[1:]:  # Skip header
            try:
                # SUAMECA structure: <th>date</th><td>value</td>
                th_elements = row.find_elements(By.TAG_NAME, "th")
                td_elements = row.find_elements(By.TAG_NAME, "td")

                if not th_elements or not td_elements:
                    continue

                fecha_str = th_elements[0].text.strip()
                valor_str = td_elements[0].text.strip()

                # Clean value
                valor_str = valor_str.replace(',', '.').replace('%', '').replace(' ', '').replace('\n', '')

                if not fecha_str or not valor_str:
                    continue

                try:
                    valor = float(valor_str)
                except:
                    continue

                # Parse date
                fecha = None
                for fmt in ['%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y', '%Y/%m', '%Y-%m']:
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
            logger.warning(f"[SUAMECA V4] No data extracted for {serie_name}")
            return None

        df = pd.DataFrame(data)
        df = df.sort_values('fecha', ascending=False).reset_index(drop=True)
        logger.info(f"[SUAMECA V4] Extracted {len(df)} records for {serie_name}")
        logger.info(f"[SUAMECA V4] Latest: {df.iloc[0]['fecha']} = {df.iloc[0]['valor']}")

        return df.head(n)

    except Exception as e:
        logger.error(f"[SUAMECA V4] Error scraping {serie_name}: {e}")
        return None

    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def _scrape_fdi_via_graficador(url: str, serie_name: str, n: int = 0, headless: bool = True) -> Optional[pd.DataFrame]:
    """
    Scrape FDI series (FDIIN/FDIOUT) usando el graficador-interactivo de SUAMECA.

    Estrategia:
    1. Ir al graficador-interactivo (ej: .../grafica/15133)
    2. Click en botón de menú Highcharts (icono hamburguesa/descargar)
    3. Seleccionar "Download CSV" o "Descargar CSV"
    4. Leer el CSV descargado
    5. Retornar DataFrame

    URLs:
    - FDIIN: https://suameca.banrep.gov.co/graficador-interactivo/grafica/15133
    - FDIOUT: https://suameca.banrep.gov.co/graficador-interactivo/grafica/15134

    Args:
        url: URL del graficador-interactivo
        serie_name: Nombre corto para logging
        n: Número de registros (0 = todos)
        headless: Ejecutar Chrome sin ventana
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains
    import tempfile
    import glob
    import os

    # Crear directorio temporal para descargas
    download_dir = tempfile.mkdtemp()

    driver = None
    try:
        import undetected_chromedriver as uc

        options = uc.ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--lang=es-CO')

        # Configurar directorio de descarga
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        options.add_experimental_option("prefs", prefs)

        driver = uc.Chrome(options=options, version_main=None)

        logger.info(f"[SUAMECA GRAFICADOR] Fetching {serie_name}...")
        logger.info(f"[SUAMECA GRAFICADOR] URL: {url}")

        # Paso 1: Ir al graficador
        driver.get(url)
        time.sleep(10)

        # Check for captcha
        page_source = driver.page_source.lower()
        if 'captcha' in page_source or 'radware' in page_source:
            logger.warning(f"[SUAMECA GRAFICADOR] Captcha detected, waiting...")
            time.sleep(25)

        # Esperar a que cargue el gráfico Highcharts
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".highcharts-container, .highcharts-root"))
            )
            logger.info("[SUAMECA GRAFICADOR] Highcharts loaded")
        except:
            logger.warning("[SUAMECA GRAFICADOR] Highcharts container not found, continuing...")

        time.sleep(5)

        # Paso 2: Buscar y hacer click en el botón de menú/descarga de Highcharts
        menu_clicked = False

        # Selectores para el botón de menú de Highcharts
        menu_selectors = [
            (By.CSS_SELECTOR, "button.highcharts-a11y-proxy-element[title='Descargar']"),
            (By.CSS_SELECTOR, "button[title='Descargar']"),
            (By.CSS_SELECTOR, "button[aria-label*='menu']"),
            (By.CSS_SELECTOR, ".highcharts-button"),
            (By.CSS_SELECTOR, ".highcharts-contextbutton"),
            (By.CSS_SELECTOR, "g.highcharts-button"),
            (By.CSS_SELECTOR, ".highcharts-exporting-group"),
        ]

        for by, selector in menu_selectors:
            try:
                elements = driver.find_elements(by, selector)
                for elem in elements:
                    if elem.is_displayed():
                        ActionChains(driver).move_to_element(elem).click().perform()
                        menu_clicked = True
                        logger.info(f"[SUAMECA GRAFICADOR] Clicked menu button: {selector}")
                        time.sleep(2)
                        break
                if menu_clicked:
                    break
            except Exception as e:
                continue

        if not menu_clicked:
            # Intentar con JavaScript
            try:
                driver.execute_script("""
                    var buttons = document.querySelectorAll('.highcharts-button, .highcharts-contextbutton, button[title="Descargar"]');
                    if (buttons.length > 0) {
                        buttons[0].dispatchEvent(new MouseEvent('click', {bubbles: true}));
                    }
                """)
                menu_clicked = True
                logger.info("[SUAMECA GRAFICADOR] Clicked menu via JavaScript")
                time.sleep(2)
            except:
                pass

        # Paso 3: Seleccionar "Download CSV" del menú
        if menu_clicked:
            time.sleep(2)
            csv_clicked = False

            csv_selectors = [
                (By.XPATH, "//*[contains(text(), 'Download CSV')]"),
                (By.XPATH, "//*[contains(text(), 'Descargar CSV')]"),
                (By.XPATH, "//*[contains(text(), 'CSV')]"),
                (By.CSS_SELECTOR, ".highcharts-menu-item"),
            ]

            for by, selector in csv_selectors:
                try:
                    elements = driver.find_elements(by, selector)
                    for elem in elements:
                        elem_text = elem.text.lower()
                        if 'csv' in elem_text:
                            ActionChains(driver).move_to_element(elem).click().perform()
                            csv_clicked = True
                            logger.info(f"[SUAMECA GRAFICADOR] Clicked CSV download: {elem.text}")
                            time.sleep(3)
                            break
                    if csv_clicked:
                        break
                except:
                    continue

            if not csv_clicked:
                # Intentar click en cualquier item del menú que contenga CSV
                try:
                    driver.execute_script("""
                        var items = document.querySelectorAll('.highcharts-menu-item, li');
                        for (var i = 0; i < items.length; i++) {
                            if (items[i].textContent.toLowerCase().includes('csv')) {
                                items[i].click();
                                break;
                            }
                        }
                    """)
                    csv_clicked = True
                    logger.info("[SUAMECA GRAFICADOR] Clicked CSV via JavaScript")
                    time.sleep(3)
                except:
                    pass

        # Paso 4: Esperar y leer el archivo CSV descargado
        time.sleep(5)

        # Buscar archivo CSV en el directorio de descarga
        csv_files = glob.glob(os.path.join(download_dir, "*.csv"))

        if csv_files:
            csv_file = csv_files[0]
            logger.info(f"[SUAMECA GRAFICADOR] Found CSV: {csv_file}")

            # Leer el CSV (Highcharts usa punto y coma como separador)
            try:
                # Intentar diferentes separadores
                df = None
                for sep in [';', ',', '\t']:
                    try:
                        df_temp = pd.read_csv(csv_file, sep=sep)
                        if len(df_temp.columns) >= 2:
                            df = df_temp
                            logger.info(f"[SUAMECA GRAFICADOR] CSV parsed with separator: '{sep}'")
                            break
                        # Si solo hay 1 columna, el separador es incorrecto
                        elif len(df_temp.columns) == 1 and sep != '\t':
                            continue
                    except:
                        continue

                if df is None or len(df.columns) < 2:
                    # Leer el archivo raw y parsear manualmente
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    logger.info(f"[SUAMECA GRAFICADOR] Raw CSV first lines: {lines[:3]}")

                    # Parsear manualmente
                    data = []
                    for line in lines[1:]:  # Skip header
                        line = line.strip()
                        if not line:
                            continue

                        # Intentar separar por ; o ,
                        parts = line.split(';') if ';' in line else line.split(',')
                        if len(parts) >= 2:
                            fecha_str = parts[0].strip().strip('"')
                            valor_str = parts[1].strip().strip('"')

                            # Limpiar valor (formato europeo: 1.234,56 -> 1234.56)
                            if ',' in valor_str and '.' in valor_str:
                                valor_str = valor_str.replace('.', '').replace(',', '.')
                            elif ',' in valor_str:
                                valor_str = valor_str.replace(',', '.')

                            try:
                                valor = float(valor_str)
                                fecha = pd.to_datetime(fecha_str, errors='coerce')
                                if pd.notna(fecha):
                                    data.append({'fecha': fecha, 'valor': valor})
                            except:
                                continue

                    if data:
                        df = pd.DataFrame(data)

                if df is not None and len(df.columns) >= 2:
                    logger.info(f"[SUAMECA GRAFICADOR] CSV columns: {list(df.columns)}")

                    # Renombrar columnas
                    df.columns = ['fecha', 'valor'] + list(df.columns[2:]) if len(df.columns) > 2 else ['fecha', 'valor']

                    # Convertir fecha
                    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
                    df = df.dropna(subset=['fecha'])

                    # Convertir valor (manejar formato europeo)
                    if df['valor'].dtype == 'object':
                        df['valor'] = df['valor'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
                    df = df.dropna(subset=['valor'])

                    df = df.sort_values('fecha', ascending=False).reset_index(drop=True)
                    logger.info(f"[SUAMECA GRAFICADOR] Extracted {len(df)} records for {serie_name}")

                    if not df.empty:
                        logger.info(f"[SUAMECA GRAFICADOR] Latest: {df.iloc[0]['fecha']} = {df.iloc[0]['valor']}")

                    if n > 0:
                        return df.head(n)
                    return df

            except Exception as e:
                logger.error(f"[SUAMECA GRAFICADOR] Error reading CSV: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f"[SUAMECA GRAFICADOR] No CSV file found in {download_dir}")

            # Alternativa: extraer datos directamente del gráfico Highcharts
            logger.info("[SUAMECA GRAFICADOR] Trying to extract data from Highcharts directly...")

            try:
                # Extraer datos del objeto Highcharts via JavaScript
                data_json = driver.execute_script("""
                    var chart = Highcharts.charts.find(c => c !== undefined);
                    if (chart && chart.series && chart.series[0]) {
                        return chart.series[0].data.map(function(point) {
                            return {x: point.x, y: point.y};
                        });
                    }
                    return null;
                """)

                if data_json:
                    data = []
                    for point in data_json:
                        if point['x'] and point['y'] is not None:
                            # x es timestamp en milliseconds
                            fecha = pd.to_datetime(point['x'], unit='ms')
                            data.append({'fecha': fecha, 'valor': point['y']})

                    if data:
                        df = pd.DataFrame(data)
                        df = df.sort_values('fecha', ascending=False).reset_index(drop=True)
                        logger.info(f"[SUAMECA GRAFICADOR] Extracted {len(df)} records from Highcharts")

                        if not df.empty:
                            logger.info(f"[SUAMECA GRAFICADOR] Latest: {df.iloc[0]['fecha']} = {df.iloc[0]['valor']}")

                        if n > 0:
                            return df.head(n)
                        return df

            except Exception as e:
                logger.warning(f"[SUAMECA GRAFICADOR] Could not extract from Highcharts: {e}")

        return None

    except Exception as e:
        logger.error(f"[SUAMECA GRAFICADOR] Error scraping {serie_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

        # Limpiar directorio temporal
        try:
            import shutil
            shutil.rmtree(download_dir, ignore_errors=True)
        except:
            pass


# ============================================================================
# Public API Functions (called by HPC V3)
# ============================================================================

def obtener_ied_suameca(n: int = 0, headless: bool = True) -> Optional[pd.DataFrame]:
    """
    Get IED/FDIIN - Inversión Extranjera Directa en Colombia (FDI Inflows).

    Usa estrategia de graficador-interactivo:
    1. Va a https://suameca.banrep.gov.co/graficador-interactivo/grafica/15133
    2. Click en botón de descarga Highcharts
    3. Selecciona "Download CSV"
    4. Lee el CSV descargado

    Args:
        n: Número de registros (0 = todos)
        headless: Ejecutar Chrome sin ventana
    """
    config = BOP_SERIES['ied']
    return _scrape_fdi_via_graficador(
        url=config['url'],
        serie_name="FDIIN (IED)",
        n=n,
        headless=headless
    )


def obtener_idce_suameca(n: int = 0, headless: bool = True) -> Optional[pd.DataFrame]:
    """
    Get IDCE/FDIOUT - Inversión Directa de Colombia en el Exterior (FDI Outflows).

    Usa estrategia de graficador-interactivo:
    1. Va a https://suameca.banrep.gov.co/graficador-interactivo/grafica/15134
    2. Click en botón de descarga Highcharts
    3. Selecciona "Download CSV"
    4. Lee el CSV descargado

    Args:
        n: Número de registros (0 = todos)
        headless: Ejecutar Chrome sin ventana
    """
    config = BOP_SERIES['idce']
    return _scrape_fdi_via_graficador(
        url=config['url'],
        serie_name="FDIOUT (IDCE)",
        n=n,
        headless=headless
    )


def obtener_cuenta_corriente_suameca(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get Cuenta Corriente (Current Account) - uses Vista tabla button."""
    config = BOP_SERIES['cacct']
    return _scrape_serie(config['url'], config['name'], n, headless, use_vista_tabla=config.get('use_vista_tabla', True))


def obtener_reservas_suameca_v4(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get Reservas Internacionales via Selenium - uses Vista tabla button."""
    config = BOP_SERIES['resint']
    return _scrape_serie(config['url'], config['name'], n, headless, use_vista_tabla=config.get('use_vista_tabla', True))


def obtener_itcr_suameca_v4(n: int = 0, headless: bool = True) -> Optional[pd.DataFrame]:
    """
    Get ITCR - Índice Tasa de Cambio Real (Ponderaciones Totales).

    Estrategia:
    1. Primario: Graficador CSV (más rápido ~35s)
       URL: https://suameca.banrep.gov.co/graficador-interactivo/grafica/234
    2. Fallback: Vista Tabla (más lento ~63s)
       URL: https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4170/indice_tasa_cambio_real_itcr

    Args:
        n: Número de registros (0 = todos)
        headless: Ejecutar Chrome sin ventana
    """
    config = BOP_SERIES['itcr']

    # Intento 1: Graficador (más rápido)
    logger.info("[ITCR] Trying Graficador (primary)...")
    df = _scrape_fdi_via_graficador(
        url=config['url'],
        serie_name=config['name'],
        n=n,
        headless=headless
    )

    if df is not None and not df.empty:
        return df

    # Intento 2: Vista Tabla (fallback)
    logger.info("[ITCR] Graficador failed, trying Vista Tabla (fallback)...")
    return _scrape_serie(
        url=config.get('url_fallback', config['url']),
        serie_name=config['name'],
        n=n,
        headless=headless,
        use_vista_tabla=True
    )


def obtener_itcr_usa_suameca(n: int = 0, headless: bool = True) -> Optional[pd.DataFrame]:
    """
    Get ITCR_USA - Índice Tasa de Cambio Real Bilateral con Estados Unidos.

    URL: https://suameca.banrep.gov.co/graficador-interactivo/grafica/219

    Args:
        n: Número de registros (0 = todos)
        headless: Ejecutar Chrome sin ventana
    """
    config = BOP_SERIES['itcr_usa']
    return _scrape_fdi_via_graficador(
        url=config['url'],
        serie_name=config['name'],
        n=n,
        headless=headless
    )


def obtener_tot_suameca_v4(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get TOT - Terms of Trade Index - uses Vista tabla button."""
    config = BOP_SERIES['tot']
    return _scrape_serie(config['url'], config['name'], n, headless, use_vista_tabla=config.get('use_vista_tabla', True))


def obtener_ipc_suameca_v4(n: int = 20, headless: bool = True) -> Optional[pd.DataFrame]:
    """Get IPC Colombia - uses Vista tabla button."""
    config = BOP_SERIES['ipccol']
    return _scrape_serie(config['url'], config['name'], n, headless, use_vista_tabla=config.get('use_vista_tabla', True))


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    print("=" * 70)
    print("SUAMECA V4 SCRAPER TEST - BanRep Variables")
    print("=" * 70)

    # Test Vista tabla variables (IPCCOL, ITCR, TOT, CACCT, RESINT)
    vista_tabla_funcs = [
        ('IPCCOL (IPC Colombia)', obtener_ipc_suameca_v4),
        ('ITCR (Tasa Cambio Real)', obtener_itcr_suameca_v4),
        ('TOT (Terminos Intercambio)', obtener_tot_suameca_v4),
        ('CACCT (Cuenta Corriente)', obtener_cuenta_corriente_suameca),
        ('RESINT (Reservas Intl)', obtener_reservas_suameca_v4),
    ]

    print("\n--- VISTA TABLA VARIABLES ---")
    for name, func in vista_tabla_funcs:
        print(f"\n[TEST] {name}")
        df = func(n=5, headless=False)  # headless=False para debug
        if df is not None and not df.empty:
            print(f"  OK - {len(df)} records")
            print(f"  Latest: {df.iloc[0]['fecha']} = {df.iloc[0]['valor']}")
        else:
            print(f"  FAILED - No data")

    # Test FDI variables (estrategia GRAFICADOR-INTERACTIVO)
    print("\n--- FDI VARIABLES (estrategia GRAFICADOR + CSV) ---")
    fdi_tests = [
        ('FDIIN', 'https://suameca.banrep.gov.co/graficador-interactivo/grafica/15133', obtener_ied_suameca),
        ('FDIOUT', 'https://suameca.banrep.gov.co/graficador-interactivo/grafica/15134', obtener_idce_suameca),
    ]

    for name, url, func in fdi_tests:
        print(f"\n[TEST] {name}")
        print(f"  URL: {url}")
        df = func(n=10, headless=False)  # headless=False para debug
        if df is not None and not df.empty:
            print(f"  OK - {len(df)} records")
            print(f"  Date range: {df['fecha'].min()} to {df['fecha'].max()}")
            print(f"  Latest: {df.iloc[0]['fecha']} = {df.iloc[0]['valor']} Millones USD")
        else:
            print(f"  FAILED - No data")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
