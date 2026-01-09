#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACTUALIZADOR UNIFICADO USD/COP - V2
===================================
Script standalone para actualizar todos los datasets del pipeline.

Funcionalidades:
1. Actualiza datos OHLCV 5min desde TwelveData API (32 API keys disponibles)
2. Actualiza datos OHLCV diarios via scraping de Investing.com
3. Actualiza datos macro via scrapers HPC
4. Regenera todos los datasets intermedios
5. Regenera los 10 datasets RL (5min y diarios)

Uso:
    python update_all_datasets.py              # Actualiza todo
    python update_all_datasets.py --check      # Solo verifica estado
    python update_all_datasets.py --daily-only # Solo datasets diarios
    python update_all_datasets.py --5min-only  # Solo datasets 5min

Autor: Sistema Automatizado
Fecha: 2025-12-05
"""

import os
import sys
import argparse
import subprocess
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import time
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# RUTAS
# =============================================================================
PIPELINE_DIR = Path(__file__).parent  # pipeline/
PROJECT_ROOT = PIPELINE_DIR.parent.parent  # USDCOP-RL-Models/
DATA_DIR = PIPELINE_DIR.parent  # data/

# Cargar .env desde raiz del proyecto
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
    logger.info(f"Cargado .env desde: {ENV_FILE}")

# =============================================================================
# COLORES PARA CONSOLA
# =============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")

def print_step(step_num, text):
    print(f"\n{Colors.BLUE}[PASO {step_num}]{Colors.END} {text}")

def print_success(text):
    print(f"  {Colors.GREEN}[OK]{Colors.END} {text}")

def print_warning(text):
    print(f"  {Colors.YELLOW}[!]{Colors.END} {text}")

def print_error(text):
    print(f"  {Colors.RED}[X]{Colors.END} {text}")


# =============================================================================
# 1. CLIENTE TWELVEDATA STANDALONE
# =============================================================================
class TwelveDataUpdater:
    """Cliente TwelveData standalone para actualizar datos 5min"""

    def __init__(self):
        self.base_url = "https://api.twelvedata.com"
        self.symbol = "USD/COP"
        self.timezone = "America/Bogota"
        self.api_keys = self._load_api_keys()
        self.current_key_idx = 0

    def _load_api_keys(self) -> List[str]:
        """Carga todas las API keys desde variables de entorno"""
        keys = []

        # Legacy keys (TWELVEDATA_API_KEY_1 to _8)
        for i in range(1, 9):
            key = os.environ.get(f'TWELVEDATA_API_KEY_{i}')
            if key:
                keys.append(key)

        # Group 1 keys
        for i in range(1, 9):
            key = os.environ.get(f'API_KEY_G1_{i}')
            if key and key not in keys:
                keys.append(key)

        # Group 2 keys
        for i in range(1, 9):
            key = os.environ.get(f'API_KEY_G2_{i}')
            if key and key not in keys:
                keys.append(key)

        # Group 3 keys
        for i in range(1, 9):
            key = os.environ.get(f'API_KEY_G3_{i}')
            if key and key not in keys:
                keys.append(key)

        logger.info(f"Cargadas {len(keys)} API keys de TwelveData")
        return keys

    def _get_next_key(self) -> str:
        """Obtiene la siguiente API key (rotacion round-robin)"""
        if not self.api_keys:
            raise ValueError("No hay API keys configuradas")
        key = self.api_keys[self.current_key_idx]
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        return key

    def fetch_data(self, start_date: datetime, end_date: datetime, interval: str = "5min") -> 'pd.DataFrame':
        """Fetch datos OHLCV desde TwelveData API"""
        import pandas as pd
        import requests

        all_data = []
        current_start = start_date
        chunk_days = 7  # 7 dias por request

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)

            api_key = self._get_next_key()

            params = {
                'symbol': self.symbol,
                'interval': interval,
                'start_date': current_start.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': current_end.strftime('%Y-%m-%d %H:%M:%S'),
                'timezone': self.timezone,
                'apikey': api_key,
                'format': 'JSON',
                'outputsize': 5000
            }

            logger.info(f"Fetching {current_start.date()} to {current_end.date()} (key: ...{api_key[-4:]})")

            try:
                response = requests.get(f"{self.base_url}/time_series", params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if 'status' in data and data['status'] == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        logger.warning(f"API Error: {error_msg}")

                        if 'API credits' in error_msg or 'limit' in error_msg:
                            logger.info("Rotando a siguiente API key...")
                            time.sleep(2)
                            continue

                    if 'values' in data:
                        df_chunk = pd.DataFrame(data['values'])
                        df_chunk = df_chunk.rename(columns={'datetime': 'timestamp'})
                        df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'])

                        # Convertir a numerico
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in df_chunk.columns:
                                df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')

                        all_data.append(df_chunk)
                        logger.info(f"  Obtenidos {len(df_chunk)} registros")
                else:
                    logger.error(f"HTTP Error {response.status_code}")

            except Exception as e:
                logger.error(f"Error: {e}")

            current_start = current_end
            time.sleep(1)  # Rate limiting

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            return df

        return pd.DataFrame()

    def update_backup(self) -> Tuple[bool, str]:
        """Actualiza el backup de datos 5min"""
        import pandas as pd

        backup_dir = PROJECT_ROOT / "backups" / "database"
        backup_files = list(backup_dir.glob("usdcop_m5_ohlcv_*.csv.gz"))

        if not backup_files:
            print_error("No hay backup existente de datos 5min")
            return False, "No backup found"

        # Encontrar backup mas reciente
        latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)

        # Leer ultimo timestamp del backup
        with gzip.open(latest_backup, 'rt') as f:
            df_backup = pd.read_csv(f)

        # El backup usa 'time' como columna de timestamp
        time_col = 'time' if 'time' in df_backup.columns else 'timestamp'
        df_backup[time_col] = pd.to_datetime(df_backup[time_col])
        last_timestamp = df_backup[time_col].max()

        logger.info(f"Ultimo dato en backup: {last_timestamp}")

        # Calcular rango a actualizar
        # Convertir a naive datetime para comparacion
        if last_timestamp.tzinfo is not None:
            last_timestamp_naive = last_timestamp.replace(tzinfo=None)
        else:
            last_timestamp_naive = last_timestamp

        now = datetime.now()
        start_date = last_timestamp_naive + timedelta(minutes=5)

        if start_date >= now:
            print_warning("Backup ya esta actualizado")
            return True, "Already up to date"

        # Fetch nuevos datos
        logger.info(f"Descargando datos desde {start_date} hasta {now}")
        df_new = self.fetch_data(start_date, now, interval="5min")

        if df_new.empty:
            print_warning("No se obtuvieron datos nuevos")
            return True, "No new data available"

        # Combinar con backup existente
        # Renombrar 'timestamp' a 'time' para compatibilidad con backup
        if 'timestamp' in df_new.columns and time_col == 'time':
            df_new = df_new.rename(columns={'timestamp': 'time'})

        # Asegurar columnas compatibles (usar las columnas del backup)
        backup_cols = df_backup.columns.tolist()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df_new.columns:
                df_new[col] = 0

        # Agregar columnas faltantes del backup
        if 'symbol' not in df_new.columns:
            df_new['symbol'] = 'USD/COP'
        if 'source' not in df_new.columns:
            df_new['source'] = 'twelvedata'
        if 'created_at' not in df_new.columns:
            df_new['created_at'] = datetime.now().isoformat()
        if 'updated_at' not in df_new.columns:
            df_new['updated_at'] = datetime.now().isoformat()

        # Seleccionar solo columnas del backup
        common_cols = [c for c in backup_cols if c in df_new.columns]
        df_new_filtered = df_new[common_cols].copy()

        # Asegurar que timestamps tengan el mismo timezone
        # Convertir ambos a UTC naive para comparacion consistente
        if df_backup[time_col].dt.tz is not None:
            df_backup[time_col] = df_backup[time_col].dt.tz_localize(None)
        if time_col in df_new_filtered.columns:
            if df_new_filtered[time_col].dt.tz is not None:
                df_new_filtered[time_col] = df_new_filtered[time_col].dt.tz_localize(None)

        df_combined = pd.concat([df_backup, df_new_filtered], ignore_index=True)
        df_combined = df_combined.sort_values(time_col).drop_duplicates(subset=[time_col])

        # Guardar nuevo backup
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_backup_path = backup_dir / f"usdcop_m5_ohlcv_{timestamp_str}.csv.gz"

        with gzip.open(new_backup_path, 'wt') as f:
            df_combined.to_csv(f, index=False)

        new_records = len(df_combined) - len(df_backup)
        print_success(f"Backup actualizado: +{new_records} registros")
        print_success(f"Nuevo backup: {new_backup_path.name}")

        return True, f"Added {new_records} records"


# =============================================================================
# 2. SCRAPER DATOS DIARIOS (INVESTING.COM)
# =============================================================================
class DailyOHLCUpdater:
    """Scraper para datos diarios de USD/COP desde Investing.com"""

    def __init__(self):
        self.data_file = PIPELINE_DIR / "00_raw_sources" / "usdcop_daily_historical" / "USD_COP Historical Data.csv"

    def update(self) -> Tuple[bool, str]:
        """Actualiza datos diarios via scraping"""
        import pandas as pd

        try:
            import cloudscraper
            from bs4 import BeautifulSoup
        except ImportError:
            print_error("Falta cloudscraper o beautifulsoup4. Instalar con: pip install cloudscraper beautifulsoup4")
            return False, "Missing dependencies"

        if not self.data_file.exists():
            print_error(f"Archivo no encontrado: {self.data_file}")
            return False, "File not found"

        # Leer datos actuales
        df_current = pd.read_csv(self.data_file)
        df_current['Date'] = pd.to_datetime(df_current['Date'], format='%m/%d/%Y')
        last_date = df_current['Date'].max()

        logger.info(f"Ultimo dato diario: {last_date.date()}")

        today = datetime.now().date()
        if last_date.date() >= today:
            print_warning("Datos diarios ya actualizados")
            return True, "Already up to date"

        # Scrape nuevos datos
        logger.info("Scrapeando Investing.com...")

        try:
            scraper = cloudscraper.create_scraper()
            url = "https://www.investing.com/currencies/usd-cop-historical-data"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.9',
            }

            response = scraper.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                print_error(f"HTTP Error: {response.status_code}")
                return False, f"HTTP {response.status_code}"

            soup = BeautifulSoup(response.content, 'html.parser')

            # Buscar tabla de datos historicos
            table = soup.find('table', {'class': 'freeze-column-w-1'})
            if not table:
                table = soup.find('table', {'id': 'curr_table'})

            if not table:
                print_warning("No se encontro tabla de datos (posible cambio en estructura)")
                return False, "Table not found"

            # Parsear filas
            new_rows = []
            rows = table.find_all('tr')[1:]  # Skip header

            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    try:
                        date_str = cols[0].text.strip()
                        date = pd.to_datetime(date_str, format='%b %d, %Y')

                        if date.date() > last_date.date():
                            new_rows.append({
                                'Date': date.strftime('%m/%d/%Y'),
                                'Price': cols[1].text.strip().replace(',', ''),
                                'Open': cols[2].text.strip().replace(',', ''),
                                'High': cols[3].text.strip().replace(',', ''),
                                'Low': cols[4].text.strip().replace(',', ''),
                                'Vol.': cols[5].text.strip() if len(cols) > 5 else '-',
                                'Change %': cols[6].text.strip() if len(cols) > 6 else '-'
                            })
                    except Exception as e:
                        continue

            if not new_rows:
                print_warning("No hay datos nuevos disponibles")
                return True, "No new data"

            # Agregar nuevos datos
            df_new = pd.DataFrame(new_rows)
            df_combined = pd.concat([df_new, df_current], ignore_index=True)
            df_combined.to_csv(self.data_file, index=False)

            print_success(f"Agregados {len(new_rows)} dias nuevos")
            return True, f"Added {len(new_rows)} days"

        except Exception as e:
            print_error(f"Error scraping: {e}")
            return False, str(e)


# =============================================================================
# 3. ACTUALIZADOR DATOS MACRO
# =============================================================================
def update_macro_data() -> Tuple[bool, str]:
    """Ejecuta el actualizador HPC de datos macro"""

    scraper_script = PIPELINE_DIR / "02_update_scrapers" / "orchestrators" / "actualizador_hpc_v3.py"

    if not scraper_script.exists():
        print_warning(f"Script macro no encontrado: {scraper_script}")
        return False, "Script not found"

    logger.info("Ejecutando actualizador macro HPC...")

    try:
        # Cambiar al directorio del script para imports relativos
        original_dir = os.getcwd()
        os.chdir(scraper_script.parent)

        result = subprocess.run(
            [sys.executable, str(scraper_script)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutos max
            cwd=str(scraper_script.parent)
        )

        os.chdir(original_dir)

        if result.returncode == 0:
            print_success("Datos macro actualizados")
            return True, "Macro data updated"
        else:
            print_warning(f"Scraper termino con codigo {result.returncode}")
            if result.stderr:
                logger.warning(result.stderr[-500:])  # Ultimos 500 chars
            return True, "Completed with warnings"

    except subprocess.TimeoutExpired:
        print_warning("Timeout en actualizador macro")
        return False, "Timeout"
    except Exception as e:
        print_error(f"Error: {e}")
        return False, str(e)


# =============================================================================
# 4. REGENERAR DATASETS INTERMEDIOS
# =============================================================================
def regenerate_intermediate() -> bool:
    """Ejecuta scripts 01 y 02 para regenerar datos intermedios"""

    scripts_dir = PIPELINE_DIR / "03_processing" / "scripts"

    scripts = [
        ("01_clean_macro_data.py", "Limpieza de datos macro"),
        ("02_resample_consolidated.py", "Resampleo y consolidacion"),
    ]

    for script_name, description in scripts:
        script_path = scripts_dir / script_name

        if not script_path.exists():
            print_warning(f"Script no encontrado: {script_name}")
            continue

        print(f"  Ejecutando: {description}")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(scripts_dir)
            )

            if result.returncode == 0:
                print_success(f"{script_name} completado")
            else:
                print_warning(f"{script_name} con warnings")

        except Exception as e:
            print_error(f"Error en {script_name}: {e}")
            return False

    return True


# =============================================================================
# 5. REGENERAR DATASETS RL
# =============================================================================
def regenerate_rl_datasets(mode: str = "all") -> bool:
    """
    Regenera datasets RL
    mode: 'all', '5min', 'daily'
    """
    import pandas as pd

    scripts_dir = PIPELINE_DIR / "03_processing" / "scripts"

    if mode in ['all', '5min']:
        script_5min = scripts_dir / "03_create_rl_datasets.py"
        if script_5min.exists():
            print(f"  Generando: Datasets 5 minutos")
            try:
                result = subprocess.run(
                    [sys.executable, str(script_5min)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=str(scripts_dir)
                )
                if result.returncode == 0:
                    print_success("Datasets 5 minutos completados")
                else:
                    print_warning("Datasets 5min con warnings")
            except Exception as e:
                print_error(f"Error: {e}")
                return False

    if mode in ['all', 'daily']:
        script_daily = scripts_dir / "03b_create_rl_datasets_daily.py"
        if script_daily.exists():
            print(f"  Generando: Datasets diarios")
            try:
                result = subprocess.run(
                    [sys.executable, str(script_daily)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(scripts_dir)
                )
                if result.returncode == 0:
                    # Mostrar tamano
                    ds_daily = PIPELINE_DIR / "04_rl_datasets" / "datasets_daily"
                    if ds_daily.exists():
                        files = list(ds_daily.glob("RL_DS*.csv"))
                        total_size = sum(f.stat().st_size for f in files) / (1024*1024)
                        print_success(f"Datasets diarios completados")
                        print(f"       TOTAL: {total_size:.1f} MB")
                        print(f"       Ubicacion: {ds_daily}")
                else:
                    print_warning("Datasets diarios con warnings")
            except Exception as e:
                print_error(f"Error: {e}")
                return False

    return True


# =============================================================================
# 6. CHECK STATUS
# =============================================================================
def check_status():
    """Verifica el estado actual de todos los datos"""
    import pandas as pd

    print_header("ESTADO ACTUAL DE LOS DATASETS")

    # Datos 5min
    print("\n[*] DATOS 5 MINUTOS:")
    backup_dir = PROJECT_ROOT / "backups" / "database"
    backup_files = list(backup_dir.glob("usdcop_m5_ohlcv_*.csv.gz"))

    if backup_files:
        latest = max(backup_files, key=lambda x: x.stat().st_mtime)
        try:
            with gzip.open(latest, 'rt') as f:
                # Leer solo primeras y ultimas lineas
                lines = f.readlines()
                last_line = lines[-1] if lines else ""
                num_records = len(lines) - 1  # -1 por header

            if last_line:
                last_date = last_line.split(',')[0]
                print(f"   Backup: {latest.name}")
                print(f"   Ultima fecha: {last_date}")
                print(f"   Registros: ~{num_records:,}")
        except Exception as e:
            print(f"   [!] Error leyendo backup: {e}")
    else:
        print(f"   [!] No hay backup de datos 5min")

    # API Keys status
    print("\n[*] API KEYS TWELVEDATA:")
    updater = TwelveDataUpdater()
    print(f"   Keys disponibles: {len(updater.api_keys)}")
    if updater.api_keys:
        print(f"   Primera key: ...{updater.api_keys[0][-4:]}")

    # Datos diarios
    print("\n[*] DATOS DIARIOS:")
    daily_file = PIPELINE_DIR / "00_raw_sources" / "usdcop_daily_historical" / "USD_COP Historical Data.csv"

    if daily_file.exists():
        df = pd.read_csv(daily_file)
        print(f"   Archivo: {daily_file.name}")
        print(f"   Ultima fecha: {df['Date'].iloc[0]}")
        print(f"   Registros: {len(df):,}")
    else:
        print(f"   [!] Archivo no encontrado")

    # Datasets RL 5min
    print("\n[*] DATASETS RL (5 MINUTOS):")
    ds_5min = PIPELINE_DIR / "04_rl_datasets" / "datasets"
    if ds_5min.exists():
        files = list(ds_5min.glob("RL_DS*.csv"))
        total_size = sum(f.stat().st_size for f in files) / (1024*1024)
        print(f"   Archivos: {len(files)}")
        print(f"   Tamano total: {total_size:.1f} MB")
    else:
        print(f"   [!] Carpeta no existe")

    # Datasets RL diarios
    print("\n[*] DATASETS RL (DIARIOS):")
    ds_daily = PIPELINE_DIR / "04_rl_datasets" / "datasets_daily"
    if ds_daily.exists():
        files = list(ds_daily.glob("RL_DS*.csv"))
        total_size = sum(f.stat().st_size for f in files) / (1024*1024)
        print(f"   Archivos: {len(files)}")
        print(f"   Tamano total: {total_size:.1f} MB")
    else:
        print(f"   [!] Carpeta no existe")

    print()


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Actualizador unificado de datasets USD/COP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python update_all_datasets.py              # Actualiza todo
  python update_all_datasets.py --daily-only # Solo datasets diarios
  python update_all_datasets.py --5min-only  # Solo datasets 5min
  python update_all_datasets.py --check      # Solo verifica estado
  python update_all_datasets.py --skip-macro # Omite actualizacion macro
        """
    )

    parser.add_argument('--daily-only', action='store_true',
                        help='Solo actualizar y generar datasets diarios')
    parser.add_argument('--5min-only', action='store_true',
                        help='Solo actualizar y generar datasets 5min')
    parser.add_argument('--check', action='store_true',
                        help='Solo verificar estado actual')
    parser.add_argument('--skip-macro', action='store_true',
                        help='Omitir actualizacion de datos macro')
    parser.add_argument('--skip-ohlc', action='store_true',
                        help='Omitir actualizacion de datos OHLC')

    args = parser.parse_args()

    # Solo check
    if args.check:
        check_status()
        return

    start_time = datetime.now()

    # Header
    print_header("ACTUALIZADOR UNIFICADO USD/COP")
    print(f"Fecha: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.daily_only:
        print("Modo: Solo diarios")
    elif getattr(args, '5min_only', False):
        print("Modo: Solo 5min")
    else:
        print("Modo: Completo")

    step = 1

    # =========================
    # PASO 1: Actualizar OHLC 5min
    # =========================
    if not args.daily_only and not args.skip_ohlc:
        print_step(step, "Actualizando datos OHLC 5min (TwelveData)")
        step += 1

        try:
            updater = TwelveDataUpdater()
            if updater.api_keys:
                success, msg = updater.update_backup()
                if not success:
                    print_warning(f"5min update: {msg}")
            else:
                print_warning("No hay API keys configuradas para TwelveData")
                print("  Configure las keys en el archivo .env del proyecto")
        except Exception as e:
            print_error(f"Error actualizando 5min: {e}")

    # =========================
    # PASO 2: Actualizar OHLC diario
    # =========================
    if not getattr(args, '5min_only', False) and not args.skip_ohlc:
        print_step(step, "Actualizando datos OHLC diarios (Investing.com)")
        step += 1

        try:
            daily_updater = DailyOHLCUpdater()
            success, msg = daily_updater.update()
            if not success:
                print_warning(f"Daily update: {msg}")
        except Exception as e:
            print_error(f"Error actualizando diario: {e}")

    # =========================
    # PASO 3: Actualizar datos macro
    # =========================
    if not args.skip_macro:
        print_step(step, "Actualizando datos macro (scrapers HPC)")
        step += 1

        success, msg = update_macro_data()
        if not success:
            print_warning(f"Macro update: {msg}")

    # =========================
    # PASO 4: Regenerar intermedios
    # =========================
    print_step(step, "Regenerando datasets intermedios")
    step += 1
    regenerate_intermediate()

    # =========================
    # PASO 5: Regenerar RL datasets
    # =========================
    print_step(step, "Regenerando datasets RL")

    if args.daily_only:
        regenerate_rl_datasets(mode='daily')
    elif getattr(args, '5min_only', False):
        regenerate_rl_datasets(mode='5min')
    else:
        regenerate_rl_datasets(mode='all')

    # =========================
    # RESUMEN FINAL
    # =========================
    elapsed = (datetime.now() - start_time).total_seconds() / 60

    print_header("ACTUALIZACION COMPLETADA")
    print(f"Tiempo total: {elapsed:.1f} minutos")

    # Mostrar estado final
    check_status()


if __name__ == "__main__":
    main()
