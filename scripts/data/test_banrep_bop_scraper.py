#!/usr/bin/env python3
"""
================================================================================
SCRAPER CORRECTO - PORTAL SUAMECA BANCO DE LA REPÚBLICA
================================================================================

FLUJO DEL PORTAL (según interfaz real):

1. Ir a: https://suameca.banrep.gov.co/estadisticas-economicas/catalogo
2. Navegar por el catálogo y expandir categorías
3. Click en "Añadir serie" (ícono carrito) para cada serie deseada
4. Click en "Ver datos de series" (ícono gráfico)
5. Configurar fechas: "Desde" y "Hasta"
6. Seleccionar periodicidad (si aplica)
7. Click en "Exportar a Excel" (botón verde)

VARIABLES A DESCARGAR:
- Cuenta Corriente Trimestral
- FDI Inflows (IED) Trimestral
- FDI Outflows (IDCE) Trimestral

RUTA EN CATÁLOGO:
Sector externo, tasas de cambio y derivados
  └── Balanza de pagos
       └── Cuenta corriente y cuenta financiera

================================================================================
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import time
import os
from datetime import datetime
from pathlib import Path

try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False
    print("webdriver-manager no instalado. Instala con: pip install webdriver-manager")


class BanRepSUAMECAScraper:
    """
    Scraper para el portal SUAMECA del Banco de la República

    Flujo de navegación:
    1. Catálogo → Expandir categorías → Añadir al carrito
    2. Ver datos de series → Configurar fechas → Exportar Excel
    """

    # URLs del portal
    BASE_URL = "https://suameca.banrep.gov.co"
    CATALOGO_URL = f"{BASE_URL}/estadisticas-economicas/catalogo"
    DESCARGA_URL = f"{BASE_URL}/descarga-multiple-de-datos/"

    # Configuración de las series a descargar
    SERIES_BALANZA_PAGOS = {
        'CUENTA_CORRIENTE': {
            'nombre_busqueda': 'Cuenta corriente',
            'nombre_completo': 'Balanza de pagos - Cuenta corriente - trimestral',
            'ruta_catalogo': [
                'Sector externo, tasas de cambio y derivados',
                'Balanza de pagos',
                'Cuenta corriente y cuenta financiera'
            ],
            'descripcion': 'Balance de cuenta corriente de Colombia',
            'unidad': 'Millones USD',
            'frecuencia': 'Trimestral'
        },
        'FDI_INFLOWS': {
            'nombre_busqueda': 'Inversión Extranjera Directa en Colombia',
            'nombre_completo': 'Balanza de pagos - Flujo de Inversión Extranjera Directa en Colombia - trimestral',
            'ruta_catalogo': [
                'Sector externo, tasas de cambio y derivados',
                'Balanza de pagos',
                'Cuenta corriente y cuenta financiera'
            ],
            'descripcion': 'IED - Inversión extranjera entrando a Colombia',
            'unidad': 'Millones USD',
            'frecuencia': 'Trimestral'
        },
        'FDI_OUTFLOWS': {
            'nombre_busqueda': 'Inversión Directa de Colombia en el Exterior',
            'nombre_completo': 'Balanza de pagos - Flujo de Inversión Directa de Colombia en el Exterior - trimestral',
            'ruta_catalogo': [
                'Sector externo, tasas de cambio y derivados',
                'Balanza de pagos',
                'Cuenta corriente y cuenta financiera'
            ],
            'descripcion': 'IDCE - Inversión colombiana hacia el exterior',
            'unidad': 'Millones USD',
            'frecuencia': 'Trimestral'
        }
    }

    def __init__(self, download_dir: str = None, headless: bool = True):
        """
        Inicializar scraper

        Args:
            download_dir: Directorio para guardar archivos descargados
            headless: Si True, ejecuta navegador sin interfaz gráfica
        """
        self.download_dir = Path(download_dir) if download_dir else Path.cwd() / "data_banrep"
        self.download_dir.mkdir(exist_ok=True)
        self.headless = headless
        self.driver = None
        self.wait = None

    def setup_driver(self) -> bool:
        """
        Configurar Selenium WebDriver con Chrome

        Returns:
            True si se configuró correctamente
        """
        print("Configurando navegador Chrome...")

        chrome_options = Options()

        # Modo headless (sin interfaz gráfica)
        if self.headless:
            chrome_options.add_argument("--headless=new")

        # Opciones para estabilidad
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-popup-blocking")

        # User agent realista
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Configurar directorio de descargas
        prefs = {
            "download.default_directory": str(self.download_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # Evitar detección de automatización
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        try:
            if WEBDRIVER_MANAGER_AVAILABLE:
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                # Intentar usar ChromeDriver del sistema
                self.driver = webdriver.Chrome(options=chrome_options)

            self.driver.implicitly_wait(5)
            self.wait = WebDriverWait(self.driver, 30)

            # Evitar detección
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            print("Navegador configurado correctamente")
            print(f"Directorio de descargas: {self.download_dir}")
            return True

        except Exception as e:
            print(f"Error configurando navegador: {e}")
            return False

    def close_driver(self):
        """Cerrar navegador"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            print("Navegador cerrado")

    def _wait_and_click(self, by, value, timeout=30, scroll=True):
        """
        Esperar a que elemento sea clickeable y hacer click

        Args:
            by: Tipo de selector (By.XPATH, By.CSS_SELECTOR, etc.)
            value: Valor del selector
            timeout: Tiempo máximo de espera
            scroll: Si hacer scroll al elemento antes de click
        """
        element = WebDriverWait(self.driver, timeout).until(
            EC.element_to_be_clickable((by, value))
        )

        if scroll:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                element
            )
            time.sleep(0.5)

        try:
            element.click()
        except:
            # Si click normal falla, usar JavaScript
            self.driver.execute_script("arguments[0].click();", element)

        return element

    def _expand_category(self, category_name: str) -> bool:
        """
        Expandir una categoría en el catálogo

        Args:
            category_name: Nombre de la categoría a expandir
        """
        try:
            # Buscar el elemento de la categoría
            xpath_options = [
                f"//h3[contains(text(), '{category_name}')]",
                f"//span[contains(text(), '{category_name}')]",
                f"//div[contains(text(), '{category_name}')]",
                f"//*[contains(text(), '{category_name}')]"
            ]

            for xpath in xpath_options:
                try:
                    element = self.driver.find_element(By.XPATH, xpath)
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                        element
                    )
                    time.sleep(0.5)

                    # Click para expandir
                    try:
                        element.click()
                    except:
                        self.driver.execute_script("arguments[0].click();", element)

                    time.sleep(1)
                    print(f"  Expandido: {category_name}")
                    return True
                except NoSuchElementException:
                    continue

            print(f"  No se encontró categoría: {category_name}")
            return False

        except Exception as e:
            print(f"  Error expandiendo {category_name}: {e}")
            return False

    def _add_series_to_cart(self, serie_nombre: str) -> bool:
        """
        Añadir una serie al carrito de descarga

        Args:
            serie_nombre: Nombre o parte del nombre de la serie
        """
        try:
            # Buscar la fila de la serie
            xpath_serie = f"//*[contains(text(), '{serie_nombre}')]"

            serie_element = self.wait.until(
                EC.presence_of_element_located((By.XPATH, xpath_serie))
            )

            # Buscar botón de añadir al carrito (puede ser ícono de carrito o +)
            parent = serie_element.find_element(By.XPATH, "./ancestor::tr | ./ancestor::li | ./parent::*")

            # Intentar diferentes selectores para el botón de añadir
            add_button_xpaths = [
                ".//button[contains(@class, 'cart') or contains(@class, 'add')]",
                ".//button[contains(@title, 'Añadir') or contains(@title, 'agregar')]",
                ".//i[contains(@class, 'cart') or contains(@class, 'shopping')]/..",
                ".//span[contains(@class, 'cart')]/..",
                ".//button[contains(@aria-label, 'añadir')]",
                ".//input[@type='checkbox']"
            ]

            for xpath in add_button_xpaths:
                try:
                    add_btn = parent.find_element(By.XPATH, xpath)
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                        add_btn
                    )
                    time.sleep(0.3)
                    add_btn.click()
                    print(f"  Añadida al carrito: {serie_nombre[:50]}...")
                    return True
                except NoSuchElementException:
                    continue

            # Si no encontró botón, intentar click directo en la serie
            serie_element.click()
            time.sleep(0.5)
            print(f"  Seleccionada: {serie_nombre[:50]}...")
            return True

        except Exception as e:
            print(f"  Error añadiendo serie {serie_nombre[:30]}...: {e}")
            return False

    def _configure_dates(self, fecha_inicio: str, fecha_fin: str):
        """
        Configurar rango de fechas para la descarga

        Args:
            fecha_inicio: Fecha inicial en formato DD/MM/YYYY
            fecha_fin: Fecha final en formato DD/MM/YYYY
        """
        try:
            # Buscar campos de fecha
            fecha_inputs = self.driver.find_elements(
                By.XPATH,
                "//input[@type='text' and (contains(@placeholder, 'dd') or contains(@placeholder, 'fecha') or contains(@class, 'date'))]"
            )

            if len(fecha_inputs) >= 2:
                # Campo fecha inicio
                fecha_inputs[0].clear()
                fecha_inputs[0].send_keys(fecha_inicio)
                time.sleep(0.3)

                # Campo fecha fin
                fecha_inputs[1].clear()
                fecha_inputs[1].send_keys(fecha_fin)
                time.sleep(0.3)

                print(f"  Fechas configuradas: {fecha_inicio} - {fecha_fin}")
                return True
            else:
                print("  No se encontraron campos de fecha")
                return False

        except Exception as e:
            print(f"  Error configurando fechas: {e}")
            return False

    def _click_export_excel(self) -> bool:
        """
        Hacer click en botón de Exportar a Excel
        """
        try:
            export_xpaths = [
                "//button[contains(text(), 'Excel')]",
                "//button[contains(text(), 'Exportar')]",
                "//a[contains(text(), 'Excel')]",
                "//button[contains(@class, 'excel')]",
                "//i[contains(@class, 'excel')]/..",
                "//span[contains(text(), 'Excel')]/..",
                "//*[contains(@title, 'Excel')]"
            ]

            for xpath in export_xpaths:
                try:
                    export_btn = self._wait_and_click(By.XPATH, xpath, timeout=10)
                    print("  Click en Exportar a Excel")
                    time.sleep(5)  # Esperar descarga
                    return True
                except:
                    continue

            print("  No se encontró botón de exportar")
            return False

        except Exception as e:
            print(f"  Error exportando: {e}")
            return False

    def _find_downloaded_file(self, timeout=30) -> Path:
        """
        Buscar archivo descargado recientemente

        Args:
            timeout: Tiempo máximo de espera en segundos
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Buscar archivos Excel o CSV
            for pattern in ['*.xlsx', '*.xls', '*.csv']:
                files = list(self.download_dir.glob(pattern))
                if files:
                    # Ordenar por fecha de modificación
                    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    newest = files[0]

                    # Verificar que no esté en descarga
                    if not str(newest).endswith('.crdownload'):
                        # Verificar que sea reciente (últimos 60 segundos)
                        if time.time() - newest.stat().st_mtime < 60:
                            return newest

            time.sleep(1)

        return None

    def descargar_balanza_pagos_metodo_catalogo(
        self,
        series_keys: list = None,
        fecha_inicio: str = "01/01/2000",
        fecha_fin: str = None
    ) -> dict:
        """
        MÉTODO 1: NAVEGACIÓN POR CATÁLOGO

        Flujo:
        1. Ir al catálogo
        2. Expandir: Sector externo → Balanza de pagos → Cuenta corriente
        3. Añadir series al carrito
        4. Ver datos → Configurar fechas → Exportar

        Args:
            series_keys: Lista de claves de series a descargar
            fecha_inicio: Fecha inicial
            fecha_fin: Fecha final (None = hoy)
        """
        if series_keys is None:
            series_keys = list(self.SERIES_BALANZA_PAGOS.keys())

        if fecha_fin is None:
            fecha_fin = datetime.now().strftime("%d/%m/%Y")

        results = {}

        print("\n" + "="*60)
        print("MÉTODO 1: NAVEGACIÓN POR CATÁLOGO")
        print("="*60)

        if not self.setup_driver():
            return results

        try:
            # 1. Ir al catálogo
            print(f"\nNavegando a: {self.CATALOGO_URL}")
            self.driver.get(self.CATALOGO_URL)
            time.sleep(3)

            # 2. Expandir categorías de Balanza de Pagos
            print("\nExpandiendo categorías...")

            # Expandir "Sector externo, tasas de cambio y derivados"
            self._expand_category("Sector externo")
            time.sleep(1)

            # Expandir "Balanza de pagos"
            self._expand_category("Balanza de pagos")
            time.sleep(1)

            # Expandir "Cuenta corriente y cuenta financiera"
            self._expand_category("Cuenta corriente")
            time.sleep(2)

            # 3. Añadir series al carrito
            print("\nAñadiendo series al carrito...")

            for serie_key in series_keys:
                if serie_key not in self.SERIES_BALANZA_PAGOS:
                    print(f"  Serie desconocida: {serie_key}")
                    continue

                config = self.SERIES_BALANZA_PAGOS[serie_key]
                self._add_series_to_cart(config['nombre_busqueda'])
                time.sleep(1)

            # 4. Click en "Ver datos de series"
            print("\nAbriendo vista de datos...")

            ver_datos_xpaths = [
                "//button[contains(text(), 'Ver datos')]",
                "//button[contains(text(), 'ver datos')]",
                "//a[contains(text(), 'Ver datos')]",
                "//*[contains(@class, 'view-data')]",
                "//button[contains(@title, 'Ver')]"
            ]

            for xpath in ver_datos_xpaths:
                try:
                    self._wait_and_click(By.XPATH, xpath, timeout=10)
                    print("  Click en Ver datos de series")
                    break
                except:
                    continue

            time.sleep(3)

            # 5. Configurar fechas
            print("\nConfigurando fechas...")
            self._configure_dates(fecha_inicio, fecha_fin)

            # 6. Exportar a Excel
            print("\nExportando a Excel...")
            if self._click_export_excel():
                # Buscar archivo descargado
                downloaded = self._find_downloaded_file()
                if downloaded:
                    print(f"  Archivo descargado: {downloaded.name}")

                    # Procesar archivo
                    df = self._process_excel(downloaded)
                    results['BALANZA_PAGOS'] = df

        except Exception as e:
            print(f"\nError durante scraping: {e}")

            # Guardar screenshot para debug
            screenshot_path = self.download_dir / f"error_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            try:
                self.driver.save_screenshot(str(screenshot_path))
                print(f"Screenshot guardado: {screenshot_path}")
            except:
                pass

        finally:
            self.close_driver()

        return results

    def descargar_balanza_pagos_metodo_descarga_multiple(
        self,
        series_keys: list = None,
        fecha_inicio: str = "01/01/2000",
        fecha_fin: str = None
    ) -> dict:
        """
        MÉTODO 2: DESCARGA MÚLTIPLE DE DATOS

        Flujo:
        1. Ir a página de Descarga múltiple
        2. Buscar series por nombre
        3. Seleccionar con checkbox
        4. Configurar fechas
        5. Descargar

        Args:
            series_keys: Lista de claves de series
            fecha_inicio: Fecha inicial
            fecha_fin: Fecha final
        """
        if series_keys is None:
            series_keys = list(self.SERIES_BALANZA_PAGOS.keys())

        if fecha_fin is None:
            fecha_fin = datetime.now().strftime("%d/%m/%Y")

        results = {}

        print("\n" + "="*60)
        print("MÉTODO 2: DESCARGA MÚLTIPLE DE DATOS")
        print("="*60)

        if not self.setup_driver():
            return results

        try:
            # 1. Ir a página de descarga múltiple
            print(f"\nNavegando a: {self.DESCARGA_URL}")
            self.driver.get(self.DESCARGA_URL)
            time.sleep(3)

            # 2. Para cada serie, buscar y seleccionar
            for serie_key in series_keys:
                if serie_key not in self.SERIES_BALANZA_PAGOS:
                    continue

                config = self.SERIES_BALANZA_PAGOS[serie_key]
                nombre = config['nombre_busqueda']

                print(f"\nBuscando: {nombre}")

                # Buscar campo de búsqueda
                try:
                    search_input = self.driver.find_element(
                        By.XPATH,
                        "//input[@type='search' or @type='text' and contains(@placeholder, 'buscar')]"
                    )
                    search_input.clear()
                    search_input.send_keys(nombre)
                    time.sleep(2)

                    # Seleccionar checkbox de la serie
                    checkbox = self.driver.find_element(
                        By.XPATH,
                        f"//tr[contains(., '{nombre}')]//input[@type='checkbox']"
                    )
                    if not checkbox.is_selected():
                        checkbox.click()

                    print(f"  Seleccionada: {nombre[:40]}...")

                except Exception as e:
                    print(f"  No se pudo seleccionar: {e}")

            # 3. Configurar fechas
            print("\nConfigurando fechas...")
            self._configure_dates(fecha_inicio, fecha_fin)
            time.sleep(1)

            # 4. Click en descargar/exportar
            print("\nDescargando...")

            download_xpaths = [
                "//button[contains(text(), 'Descargar')]",
                "//button[contains(text(), 'Exportar')]",
                "//button[contains(text(), 'Excel')]",
                "//a[contains(@href, 'download')]"
            ]

            for xpath in download_xpaths:
                try:
                    self._wait_and_click(By.XPATH, xpath, timeout=10)
                    print("  Click en Descargar")
                    break
                except:
                    continue

            time.sleep(5)

            # 5. Buscar archivo descargado
            downloaded = self._find_downloaded_file()
            if downloaded:
                print(f"  Archivo: {downloaded.name}")
                df = self._process_excel(downloaded)
                results['DESCARGA_MULTIPLE'] = df

        except Exception as e:
            print(f"\nError: {e}")

        finally:
            self.close_driver()

        return results

    def _process_excel(self, filepath: Path) -> pd.DataFrame:
        """Procesar archivo Excel descargado"""
        try:
            if filepath.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)

            print(f"  Procesado: {len(df)} filas, {len(df.columns)} columnas")

            # Guardar copia procesada
            output_path = self.download_dir / f"balanza_pagos_procesado_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(output_path, index=False)
            print(f"  Guardado: {output_path.name}")

            return df

        except Exception as e:
            print(f"  Error procesando: {e}")
            return pd.DataFrame()


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def descargar_variables_balanza_pagos(
    output_dir: str = "data_banrep",
    headless: bool = True,
    metodo: str = "catalogo"
) -> dict:
    """
    Función principal para descargar las 3 variables de Balanza de Pagos

    Args:
        output_dir: Directorio de salida
        headless: Sin interfaz gráfica
        metodo: "catalogo" o "descarga_multiple"

    Returns:
        Diccionario con DataFrames
    """
    print("\n" + "="*70)
    print("DESCARGA DE VARIABLES DE BALANZA DE PAGOS - BANCO DE LA REPÚBLICA")
    print("="*70)
    print(f"\nVariables a descargar:")
    print("  - Cuenta Corriente Trimestral")
    print("  - FDI Inflows (IED) Trimestral")
    print("  - FDI Outflows (IDCE) Trimestral")
    print(f"\nMétodo: {metodo}")
    print(f"Headless: {headless}")

    scraper = BanRepSUAMECAScraper(
        download_dir=output_dir,
        headless=headless
    )

    if metodo == "catalogo":
        results = scraper.descargar_balanza_pagos_metodo_catalogo()
    else:
        results = scraper.descargar_balanza_pagos_metodo_descarga_multiple()

    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)

    for nombre, df in results.items():
        if df is not None and not df.empty:
            print(f"  {nombre}: {len(df)} registros")
        else:
            print(f"  {nombre}: Sin datos")

    return results


if __name__ == "__main__":
    # Ejecutar scraping
    results = descargar_variables_balanza_pagos(
        output_dir="data_banrep",
        headless=True,  # Cambiar a False para ver el navegador
        metodo="catalogo"
    )
