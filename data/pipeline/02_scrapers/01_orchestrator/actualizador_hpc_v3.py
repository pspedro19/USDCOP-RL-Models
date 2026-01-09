# -*- coding: utf-8 -*-
"""
Sistema HPC de Actualizacion - USD/COP V3
==========================================

VERSION 3: Optimizado para máxima velocidad con:
1. Paralelización masiva con ThreadPoolExecutor y ProcessPoolExecutor
2. Polars en vez de Pandas (10-100x más rápido)
3. Caché de resultados para evitar re-scraping
4. Async I/O donde sea posible
5. Batch processing de variables similares

Autor: Sistema Automatizado HPC
Fecha: 2025-11-22
"""

import polars as pl
import pandas as pd  # Mantener para compatibilidad con scrapers existentes
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Optional, List
import requests
from fredapi import Fred
import yfinance as yf
import cloudscraper
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import asyncio
import aiohttp
from functools import lru_cache
import hashlib
import pickle
import re

# Importar normalizador de formatos inteligente
try:
    import sys
    _utils_path = str(Path(__file__).parent.parent / 'utils')
    if _utils_path not in sys.path:
        sys.path.insert(0, _utils_path)
    from format_normalizer import FormatNormalizer, smart_parse_number
    FORMAT_NORMALIZER_AVAILABLE = True
except ImportError:
    FORMAT_NORMALIZER_AVAILABLE = False
    # Fallback simple si no está disponible
    class FormatNormalizer:
        EXPECTED_RANGES = {
            'USDMXN': (15, 30), 'USDCLP': (600, 1200), 'USDCOP': (3000, 6000),
            'DXY': (80, 120), 'WTI': (20, 200), 'BRENT': (20, 200),
            'GOLD': (3000, 5000), 'COFFEE': (50, 500),
            'COLCAP': (500, 3000), 'EMBI': (50, 1000),
            'VIX': (8, 90),  # VIX normalmente entre 10-30, pero puede llegar a 80+ en crisis
        }

        @classmethod
        def parse_number(cls, value, variable_hint=None):
            if isinstance(value, (int, float)):
                return float(value)
            value_str = str(value).strip()
            # Detectar formato
            dots = value_str.count('.')
            commas = value_str.count(',')

            if dots > 0 and commas > 0:
                last_dot = value_str.rfind('.')
                last_comma = value_str.rfind(',')
                if last_dot > last_comma:
                    # Americano: 1,234.56
                    result = float(value_str.replace(',', ''))
                else:
                    # Europeo: 1.234,56
                    result = float(value_str.replace('.', '').replace(',', '.'))
            elif commas == 1 and dots == 0:
                # Posible europeo: 18,5131
                result_eu = float(value_str.replace(',', '.'))
                result_am = float(value_str.replace(',', ''))
                # Usar hint para decidir
                if variable_hint:
                    for key, (min_v, max_v) in cls.EXPECTED_RANGES.items():
                        if key in variable_hint.upper():
                            if min_v <= result_eu <= max_v:
                                return result_eu
                            elif min_v <= result_am <= max_v:
                                return result_am
                return result_eu if result_eu < result_am else result_am
            else:
                # Americano estándar
                result = float(value_str.replace(',', ''))

            return result

# Importar scrapers existentes
try:
    from scraper_embi_bcrp import get_embi_last_n
    EMBI_SCRAPER_AVAILABLE = True
except ImportError:
    EMBI_SCRAPER_AVAILABLE = False

try:
    from scraper_cpi_investing_calendar import get_cpi_mom_last_n
    CPI_MOM_SCRAPER_AVAILABLE = True
except ImportError:
    CPI_MOM_SCRAPER_AVAILABLE = False

try:
    from scraper_ipc_col_calendar import get_ipc_col_last_n
    IPC_COL_SCRAPER_AVAILABLE = True
except ImportError:
    IPC_COL_SCRAPER_AVAILABLE = False

try:
    from scraper_suameca_generico import (
        obtener_ibr, obtener_tpm, obtener_itcr,
        obtener_reservas, obtener_terminos_intercambio, obtener_cuenta_corriente,
        obtener_prime, obtener_ipc_colombia
    )
    SUAMECA_SCRAPER_AVAILABLE = True
except ImportError:
    SUAMECA_SCRAPER_AVAILABLE = False

try:
    from scraper_suameca_v4_final import obtener_ied_suameca, obtener_idce_suameca, obtener_cuenta_corriente_suameca, obtener_reservas_suameca_v4, obtener_itcr_suameca_v4, obtener_tot_suameca_v4, obtener_ipc_suameca_v4
    SUAMECA_IED_IDCE_AVAILABLE = True
except ImportError:
    SUAMECA_IED_IDCE_AVAILABLE = False

try:
    from scraper_fedesarrollo import obtener_cci, obtener_ici
    FEDESARROLLO_SCRAPER_AVAILABLE = True
except ImportError:
    FEDESARROLLO_SCRAPER_AVAILABLE = False

try:
    from scraper_cuenta_corriente_manual_fallback import obtener_cuenta_corriente_manual
    MANUAL_FALLBACK_AVAILABLE = True
except ImportError:
    MANUAL_FALLBACK_AVAILABLE = False

try:
    from scraper_dane_balanza import obtener_exportaciones, obtener_importaciones
    DANE_SCRAPER_AVAILABLE = True
except ImportError:
    DANE_SCRAPER_AVAILABLE = False

# Cargar configuracion
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class CacheManager:
    """Gestor de caché para evitar re-scraping"""

    def __init__(self, cache_dir: Path, ttl_hours: int = 1):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _get_cache_key(self, variable: str, n: int) -> str:
        """Generar clave de caché"""
        return hashlib.md5(f"{variable}_{n}".encode()).hexdigest()

    def get(self, variable: str, n: int) -> Optional[pd.DataFrame]:
        """Obtener del caché si existe y es válido"""
        cache_key = self._get_cache_key(variable, n)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            # Verificar si está vigente
            modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - modified_time < self.ttl:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass
        return None

    def set(self, variable: str, n: int, data: pd.DataFrame):
        """Guardar en caché"""
        cache_key = self._get_cache_key(variable, n)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass


class ActualizadorHPCV3:
    """Sistema HPC V3 con paralelización masiva"""

    def __init__(self, diccionario_path: str, max_workers: int = 10, base_path: Path = None):
        self.diccionario_path = Path(diccionario_path)
        self.base_path = base_path if base_path else Path(__file__).parent.parent
        self.max_workers = max_workers

        # Caché
        self.cache = CacheManager(self.base_path / "storage" / "cache" / ".cache", ttl_hours=6)

        # Cargar diccionario con Polars (más rápido)
        self.diccionario = self._cargar_diccionario()

        # Mapeo de fallbacks (mismo que V2)
        self.fallback_map = self._crear_mapeo_fallbacks()

        # Resultados
        self.resultados = []
        self.stats = {
            'total': 0,
            'disponibles': 0,
            'no_disponibles': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

        # Headers para web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

    def _cargar_diccionario(self) -> pd.DataFrame:
        """Cargar diccionario (mantener pandas por compatibilidad)"""
        try:
            with open(self.diccionario_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                delimiter = ';' if ';' in first_line else ','

            df = pd.read_csv(self.diccionario_path, delimiter=delimiter, encoding='utf-8')
            logger.info(f"[OK] Diccionario cargado: {len(df)} variables")
            return df

        except Exception as e:
            logger.error(f"[ERROR] Error cargando diccionario: {e}")
            return pd.DataFrame()

    def _crear_mapeo_fallbacks(self) -> Dict:
        """Crear mapeo completo de fallbacks (mismo que V2)"""

        return {
            # VARIABLES COLOMBIA
            'IBR': {'suameca': 'obtener_ibr'},
            'TPM': {'suameca': 'obtener_tpm'},
            'PRIME': {'suameca': 'obtener_prime'},
            'ITCR': {'suameca': 'obtener_itcr', 'csv_local': './scrapers/suameca/suameca_4170_itcr_datos.csv'},  # Serie 4170 - ITCRIPP (T) base 2010=100
            'RESINT': {'suameca_v4': 'obtener_reservas_suameca_v4'},
            'TOT': {'suameca': 'obtener_terminos_intercambio', 'csv_local': './scrapers/suameca/suameca_4180_tot_datos.csv'},  # Serie 4180 - Términos Intercambio
            'CACCT': {'suameca': 'obtener_cuenta_corriente', 'csv_local': './scrapers/suameca/suameca_414001_cuenta_corriente_datos.csv'},  # Serie 414001
            'FDIIN': {'suameca_v4': 'obtener_ied_suameca'},
            'IED': {'suameca_v4': 'obtener_ied_suameca'},
            'FDIOUT': {'suameca_v4': 'obtener_idce_suameca'},
            'IDCE': {'suameca_v4': 'obtener_idce_suameca'},
            'IPCCOL': {'suameca': 'obtener_ipc_colombia', 'csv_local': './scrapers/suameca/suameca_100002_ipc_datos.csv'},  # Serie 100002 de SUAMECA con fallback CSV

            # COMERCIO EXTERIOR
            'EXPUSD': {'custom': 'dane_exportaciones'},
            'IMPUSD': {'custom': 'dane_importaciones'},

            # RIESGO PAIS
            'EMBI': {'custom': 'embi_bcrp'},

            # BONOS
            'COL5Y': {'investing': 'https://www.investing.com/rates-bonds/colombia-5-year-bond-yield-historical-data'},
            'COL10Y': {'investing': 'https://www.investing.com/rates-bonds/colombia-10-year-bond-yield-historical-data'},
            'UST10Y': {'investing': 'https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data', 'yahoo': '^TNX'},

            # COMMODITIES
            'WTI': {'investing': 'https://www.investing.com/commodities/crude-oil-historical-data', 'yahoo': 'CL=F'},
            'BRENT': {'investing': 'https://www.investing.com/commodities/brent-oil-historical-data', 'yahoo': 'BZ=F'},
            # 'COAL': ELIMINADO - ya no se usa
            'GOLD': {'investing': 'https://www.investing.com/commodities/gold-historical-data', 'yahoo': 'GC=F'},
            'COFFEE': {'investing': 'https://www.investing.com/commodities/us-coffee-c-historical-data', 'yahoo': 'KC=F'},

            # FOREX
            'USDCLP': {'investing': 'https://www.investing.com/currencies/usd-clp-historical-data', 'yahoo': 'CLP=X'},
            'USDMXN': {'investing': 'https://www.investing.com/currencies/usd-mxn-historical-data', 'yahoo': 'MXN=X'},
            'DXY': {'investing': 'https://www.investing.com/indices/usdollar-historical-data', 'yahoo': 'DX-Y.NYB'},

            # INDICES
            'COLCAP': {'investing': 'https://www.investing.com/indices/colcap-historical-data', 'yahoo': 'ICOLCAP.CL'},

            # VOLATILITY
            'VIX': {'investing': 'https://www.investing.com/indices/volatility-s-p-500-historical-data', 'yahoo': '^VIX'},

            # CONFIANZA
            'CCI': {'fedesarrollo': 'obtener_cci'},
            'ICI': {'fedesarrollo': 'obtener_ici'},
        }

    def _descargar_yahoo(self, symbol: str, n: int = 20) -> Optional[pd.DataFrame]:
        """Descargar datos de Yahoo Finance"""
        try:
            periodo = f"{n+5}d"
            data = yf.download(symbol, period=periodo, progress=False)

            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    close_data = data['Close'].iloc[:, 0]
                else:
                    close_data = data['Close']

                df = pd.DataFrame({
                    'fecha': close_data.tail(n).index,
                    'valor': close_data.tail(n).values
                })
                df = df.sort_values('fecha', ascending=False).reset_index(drop=True)
                return df

            return None

        except Exception as e:
            logger.debug(f"Error Yahoo Finance {symbol}: {e}")
            return None

    def _descargar_csv_local(self, csv_path: str, n: int = 20, variable_hint: str = None) -> Optional[pd.DataFrame]:
        """
        Leer datos de un archivo CSV local.
        Soporta dos formatos:
        1. Investing.com: Date,Price,Open,High,Low,Vol.,Change % (fecha MM/DD/YYYY)
        2. SUAMECA: Fecha,Valor (fecha YYYY-MM-DD)
        """
        try:
            from pathlib import Path

            # Resolver ruta relativa desde base_path si es necesario
            csv_file = Path(csv_path)
            if not csv_file.is_absolute():
                # Ruta relativa: resolver desde base_path
                csv_file = self.base_path / csv_path.lstrip('./')

            if not csv_file.exists():
                logger.warning(f"CSV local no encontrado: {csv_file}")
                return None

            # Leer CSV con encoding apropiado (incluir BOM UTF-8)
            df = pd.read_csv(csv_file, encoding='utf-8-sig')

            # Formato 1: Investing.com (Date, Price)
            if 'Date' in df.columns and 'Price' in df.columns:
                df['fecha'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
                df['valor'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
            # Formato 2: SUAMECA (Fecha, Valor)
            elif 'Fecha' in df.columns and 'Valor' in df.columns:
                df['fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
                df['valor'] = pd.to_numeric(df['Valor'], errors='coerce')
            else:
                logger.warning(f"CSV local: columnas no reconocidas: {df.columns.tolist()}")
                return None

            # Eliminar nulos y ordenar
            df = df[['fecha', 'valor']].dropna()
            df = df.sort_values('fecha', ascending=False).reset_index(drop=True)

            # Retornar últimos n registros
            result = df.head(n).copy()

            logger.info(f"[CSV LOCAL] {variable_hint}: {len(result)} registros desde {csv_path}")

            return result

        except Exception as e:
            logger.debug(f"Error leyendo CSV local {csv_path}: {e}")
            return None

    def _descargar_investing(self, url: str, n: int = 20, variable_hint: str = None) -> Optional[pd.DataFrame]:
        """
        Descargar datos de Investing.com con detección inteligente de formato.

        Args:
            url: URL de la página histórica de Investing.com
            n: Número de registros a obtener
            variable_hint: Nombre de variable para validación de rango (ej: 'USDMXN', 'WTI')

        Returns:
            DataFrame con columnas ['fecha', 'valor'] o None si falla
        """
        try:
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, headers=self.headers, timeout=15)

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_='freeze-column-w-1')
            if not table:
                table = soup.find('table', {'data-test': 'historical-data-table'})
            if not table:
                tables = soup.find_all('table')
                if tables:
                    table = max(tables, key=lambda t: len(str(t)))

            if not table:
                return None

            rows = table.find_all('tr')[1:]

            data = []
            for row in rows[:n]:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    fecha_str = cols[0].get_text(strip=True)
                    valor_str = cols[1].get_text(strip=True)

                    try:
                        fecha = pd.to_datetime(fecha_str)

                        # USAR FORMATNORMALIZER para detección inteligente de formato
                        valor = FormatNormalizer.parse_number(valor_str, variable_hint=variable_hint)

                        if valor is None:
                            logger.warning(f"No se pudo parsear valor: '{valor_str}' para {variable_hint}")
                            continue

                        # Validar rango esperado si tenemos hint
                        if variable_hint:
                            for key, (min_v, max_v) in FormatNormalizer.EXPECTED_RANGES.items():
                                if key in variable_hint.upper():
                                    if not (min_v <= valor <= max_v):
                                        logger.warning(
                                            f"Valor fuera de rango para {variable_hint}: "
                                            f"{valor} (esperado: {min_v}-{max_v}). "
                                            f"String original: '{valor_str}'"
                                        )
                                    break

                        data.append({'fecha': fecha, 'valor': valor})
                    except Exception as e:
                        logger.debug(f"Error procesando fila: {e}")
                        continue

            if data:
                df = pd.DataFrame(data)
                return df.sort_values('fecha', ascending=False)

            return None

        except Exception as e:
            logger.debug(f"Error Investing.com: {e}")
            return None

    def _descargar_fred_api(self, codigo: str, n: int = 20) -> Optional[pd.DataFrame]:
        """Descargar datos de FRED API con formato inteligente"""
        if not FRED_API_KEY:
            return None

        try:
            fred = Fred(api_key=FRED_API_KEY)
            data = fred.get_series(codigo)

            if data is not None and not data.empty:
                df = data.tail(n).to_frame('valor')
                df['fecha'] = df.index
                df = df[['fecha', 'valor']].reset_index(drop=True)
                df = df.sort_values('fecha', ascending=False)

                # Formato inteligente según tipo de serie FRED
                # Indices (50-150): 1 decimal | Tasas (%): 1-2 decimales | Monetarios: 1-2 decimal
                formato_series = {
                    'INDPRO': 1,      # Industrial Production Index (100.1234 -> 100.1)
                    'M2SL': 1,        # M2 Money Supply Billions (22298.1)
                    'UMCSENT': 1,     # Consumer Sentiment (53.6)
                    'FEDFUNDS': 2,    # Fed Funds Rate (4.33)
                    'UNRATE': 1,      # Unemployment Rate (4.4)
                    'CPIAUCSL': 3,    # CPI Index - precision (324.368)
                    'CPILFESL': 3,    # CPI Core - precision (330.542)
                    'PCEPI': 3,       # PCE Index - precision (127.285)
                    'GDP': 2,         # Real GDP Billions - trimestral (30485.73)
                    'DGS2': 2,        # Treasury 2Y Yield (3.43)
                    'DGS10': 2,       # Treasury 10Y Yield (4.00)
                }

                decimales = formato_series.get(codigo, 2)  # Default 2 decimales
                df['valor'] = df['valor'].round(decimales)

                return df

            return None

        except Exception as e:
            logger.debug(f"Error FRED API {codigo}: {e}")
            return None

    def _descargar_custom_scraper(self, scraper_type: str, n: int = 20) -> Optional[pd.DataFrame]:
        """Descargar datos usando scrapers custom"""
        try:
            if scraper_type == 'embi_bcrp' and EMBI_SCRAPER_AVAILABLE:
                df = get_embi_last_n(n)
                if df is not None and not df.empty:
                    df_resultado = pd.DataFrame({
                        'fecha': pd.to_datetime(df['Date']),
                        'valor': df['Value'].astype(float)
                    })
                    return df_resultado.sort_values('fecha', ascending=False)

            elif scraper_type == 'cpi_mom_calendar' and CPI_MOM_SCRAPER_AVAILABLE:
                df = get_cpi_mom_last_n(n)
                if df is not None and not df.empty:
                    if 'Actual' in df.columns and 'Period' in df.columns:
                        df_resultado = pd.DataFrame({
                            'fecha': pd.to_datetime(df['Period']),
                            'valor': df['Actual'].astype(float)
                        })
                        return df_resultado.sort_values('fecha', ascending=False)

            elif scraper_type == 'ipc_col_calendar' and IPC_COL_SCRAPER_AVAILABLE:
                df = get_ipc_col_last_n(n)
                if df is not None and not df.empty:
                    if 'Actual' in df.columns and 'Period' in df.columns:
                        df_resultado = pd.DataFrame({
                            'fecha': pd.to_datetime(df['Period']),
                            'valor': df['Actual'].astype(float)
                        })
                        return df_resultado.sort_values('fecha', ascending=False)

            elif scraper_type == 'dane_exportaciones' and DANE_SCRAPER_AVAILABLE:
                df = obtener_exportaciones(n)
                if df is not None and not df.empty:
                    # Redondear a 1 decimal (millones USD)
                    df['valor'] = df['valor'].round(1)
                    return df.sort_values('fecha', ascending=False)

            elif scraper_type == 'dane_importaciones' and DANE_SCRAPER_AVAILABLE:
                df = obtener_importaciones(n)
                if df is not None and not df.empty:
                    # Redondear a 1 decimal (millones USD)
                    df['valor'] = df['valor'].round(1)
                    return df.sort_values('fecha', ascending=False)

            return None

        except Exception as e:
            logger.debug(f"Error custom scraper {scraper_type}: {e}")
            return None

    def _obtener_datos_variable_single(self, row_data: Tuple) -> Dict:
        """
        Obtener datos de UNA variable (diseñado para paralelización)

        Args:
            row_data: Tupla con (idx, row, n)

        Returns:
            Dict con resultados
        """
        idx, row, n = row_data

        var_id = row.get('ID', idx)
        var_nombre = row.get('VARIABLE_NUEVA_ESTANDAR', row.get('VARIABLE', row.get('VARIABLE_ORIGINAL', f'VAR_{var_id}')))
        codigo = str(row.get('CODIGO_FUENTE', row.get('CODIGO', ''))).strip()
        frecuencia_raw = row.get('FRECUENCIA', 'D')
        frecuencia_map = {'Daily': 'D', 'Monthly': 'M', 'Quarterly': 'Q', 'D': 'D', 'M': 'M', 'Q': 'Q'}
        frecuencia = frecuencia_map.get(frecuencia_raw, frecuencia_raw[0] if frecuencia_raw else 'D')

        # Intentar caché primero
        cached_data = self.cache.get(var_nombre, n)
        if cached_data is not None:
            self.stats['cache_hits'] += 1
            logger.info(f"  [{idx+1}] {var_nombre} - CACHE HIT")

            # Asegurar que fecha es datetime
            if not pd.api.types.is_datetime64_any_dtype(cached_data['fecha']):
                cached_data['fecha'] = pd.to_datetime(cached_data['fecha'], errors='coerce')

            ultimo_dato = cached_data['fecha'].max()
            if pd.isna(ultimo_dato):
                dias_desde = 9999  # Fecha inválida
            else:
                dias_desde = (datetime.now() - ultimo_dato).days
            ultimo_valor = cached_data['valor'].iloc[0]

            return {
                'ID': var_id,
                'Variable': var_nombre,
                'Frecuencia': frecuencia,
                'Estado': 'DISPONIBLE (cache)',
                'Fuente': 'Cache',
                'Ultimo_Valor': ultimo_valor,
                'Ultima_Fecha': ultimo_dato.strftime('%Y-%m-%d'),
                'Dias_Desde': dias_desde,
                'datos': cached_data,
            }

        self.stats['cache_misses'] += 1

        # Buscar en fallback map
        fallback_key = None
        if codigo in self.fallback_map:
            fallback_key = codigo
        elif var_nombre in self.fallback_map:
            fallback_key = var_nombre
        else:
            for key in self.fallback_map.keys():
                if var_nombre.endswith(f"_{key}"):
                    fallback_key = key
                    break

        datos = None
        fuente_usada = 'No disponible'

        if fallback_key:
            fallback = self.fallback_map[fallback_key]

            # SUAMECA V4 (IED, IDCE, CACCT, RESINT, ITCR, TOT, IPCCOL)
            if 'suameca_v4' in fallback and SUAMECA_IED_IDCE_AVAILABLE:
                func_name = fallback['suameca_v4']
                func_map = {
                    'obtener_ied_suameca': obtener_ied_suameca,
                    'obtener_idce_suameca': obtener_idce_suameca,
                    'obtener_cuenta_corriente_suameca': obtener_cuenta_corriente_suameca,
                    'obtener_reservas_suameca_v4': obtener_reservas_suameca_v4,
                    'obtener_itcr_suameca_v4': obtener_itcr_suameca_v4,
                    'obtener_tot_suameca_v4': obtener_tot_suameca_v4,
                    'obtener_ipc_suameca_v4': obtener_ipc_suameca_v4,
                }
                if func_name in func_map:
                    logger.info(f"  [{idx+1}] {var_nombre} - Ejecutando {func_name}...")
                    datos = func_map[func_name](n=n, headless=True)
                    if 'ied' in func_name:
                        variable_nombre = 'IED'
                    elif 'idce' in func_name:
                        variable_nombre = 'IDCE'
                    elif 'reservas' in func_name:
                        variable_nombre = 'RESINT'
                    elif 'itcr' in func_name:
                        variable_nombre = 'ITCR'
                    elif 'tot' in func_name:
                        variable_nombre = 'TOT'
                    elif 'ipc' in func_name:
                        variable_nombre = 'IPCCOL'
                    else:
                        variable_nombre = 'CACCT'
                    fuente_usada = f'SUAMECA V4 ({variable_nombre})'

            # SUAMECA Genérico
            elif 'suameca' in fallback and SUAMECA_SCRAPER_AVAILABLE:
                func_name = fallback['suameca']
                func_map = {
                    'obtener_ibr': obtener_ibr,
                    'obtener_tpm': obtener_tpm,
                    'obtener_prime': obtener_prime,
                    'obtener_itcr': obtener_itcr,
                    'obtener_reservas': obtener_reservas,
                    'obtener_terminos_intercambio': obtener_terminos_intercambio,
                    'obtener_cuenta_corriente': obtener_cuenta_corriente,
                    'obtener_ipc_colombia': obtener_ipc_colombia,
                }
                if func_name in func_map:
                    datos = func_map[func_name](n=n, headless=True)
                    fuente_usada = f'SUAMECA ({func_name})'

            # Manual Fallback - Si suameca_v4 falló, intentar archivo manual
            if datos is None and 'manual' in fallback and MANUAL_FALLBACK_AVAILABLE:
                func_name = fallback['manual']
                if func_name == 'obtener_cuenta_corriente_manual':
                    logger.info(f"    [FALLBACK] Intentando fuente manual para {var_nombre}...")
                    datos = obtener_cuenta_corriente_manual(n=n)
                    if datos is not None:
                        fuente_usada = 'Manual (CSV/Excel local)'

            # Fedesarrollo
            if datos is None and 'fedesarrollo' in fallback and FEDESARROLLO_SCRAPER_AVAILABLE:
                func_name = fallback['fedesarrollo']
                func_map = {
                    'obtener_cci': obtener_cci,
                    'obtener_ici': obtener_ici,
                }
                if func_name in func_map:
                    datos = func_map[func_name](n=n)
                    indice_nombre = 'CCI' if func_name == 'obtener_cci' else 'ICI'
                    fuente_usada = f'Fedesarrollo ({indice_nombre})'

            # Custom Scraper
            elif 'custom' in fallback:
                datos = self._descargar_custom_scraper(fallback['custom'], n)
                if datos is not None:
                    fuente_map = {
                        'embi_bcrp': 'EMBI (BCRP)',
                        'cpi_mom_calendar': 'CPI MoM (Calendar)',
                        'ipc_col_calendar': 'IPC Colombia (Calendar)',
                        'dane_exportaciones': 'DANE (Exportaciones)',
                        'dane_importaciones': 'DANE (Importaciones)'
                    }
                    fuente_usada = fuente_map.get(fallback['custom'], 'Custom Scraper')

            # CSV LOCAL (Fallback si otras fuentes fallan)
            if datos is None and 'csv_local' in fallback:
                variable_hint = codigo if codigo else var_nombre
                logger.info(f"    [FALLBACK] Intentando CSV local para {var_nombre}...")
                datos = self._descargar_csv_local(fallback['csv_local'], n, variable_hint=variable_hint)
                if datos is not None:
                    fuente_usada = 'CSV Local (Fallback)'

            # Investing.com (PRIORITARIO sobre Yahoo)
            elif 'investing' in fallback:
                # Pasar código de variable para detección inteligente de formato
                variable_hint = codigo if codigo else var_nombre
                datos = self._descargar_investing(fallback['investing'], n, variable_hint=variable_hint)
                if datos is not None:
                    fuente_usada = 'Investing.com'

            # Yahoo Finance (fallback si Investing falla)
            elif 'yahoo' in fallback:
                datos = self._descargar_yahoo(fallback['yahoo'], n)
                if datos is not None:
                    fuente_usada = 'Yahoo Finance'

        # Intentar FRED API como último recurso
        if datos is None and codigo:
            datos = self._descargar_fred_api(codigo, n)
            if datos is not None:
                fuente_usada = 'FRED API'

        # Procesar resultados
        if datos is not None and not datos.empty:
            # Aplicar formato inteligente según código de variable
            # (independiente de la fuente: TwelveData, Yahoo, FRED, etc.)
            formato_por_codigo = {
                'INDPRO': 1,      # Industrial Production Index
                'M2SL': 1,        # M2 Money Supply
                'UMCSENT': 1,     # Consumer Sentiment
                'FEDFUNDS': 2,    # Fed Funds Rate
                'UNRATE': 1,      # Unemployment Rate
                'CPIAUCSL': 3,    # CPI All Items
                'CPILFESL': 3,    # CPI Core
                'PCEPI': 3,       # PCE Index
            }
            if codigo and codigo in formato_por_codigo:
                decimales = formato_por_codigo[codigo]
                datos['valor'] = datos['valor'].round(decimales)

            # Guardar en caché
            self.cache.set(var_nombre, n, datos)

            # Asegurar que fecha es datetime
            if not pd.api.types.is_datetime64_any_dtype(datos['fecha']):
                datos['fecha'] = pd.to_datetime(datos['fecha'], errors='coerce')

            ultimo_dato = datos['fecha'].max()
            if pd.isna(ultimo_dato):
                dias_desde = 9999  # Fecha inválida
            else:
                dias_desde = (datetime.now() - ultimo_dato).days
            ultimo_valor = datos['valor'].iloc[0]

            if dias_desde <= 7:
                estado = "DISPONIBLE"
            elif dias_desde <= 60:
                estado = "DISPONIBLE (datos antiguos)"
            else:
                estado = "DATOS MUY ANTIGUOS"

            return {
                'ID': var_id,
                'Variable': var_nombre,
                'Frecuencia': frecuencia,
                'Estado': estado,
                'Fuente': fuente_usada,
                'Ultimo_Valor': ultimo_valor,
                'Ultima_Fecha': ultimo_dato.strftime('%Y-%m-%d'),
                'Dias_Desde': dias_desde,
                'datos': datos,
            }
        else:
            return {
                'ID': var_id,
                'Variable': var_nombre,
                'Frecuencia': frecuencia,
                'Estado': 'NO DISPONIBLE',
                'Fuente': fuente_usada,
                'Ultimo_Valor': None,
                'Ultima_Fecha': None,
                'Dias_Desde': None,
                'datos': None,
            }

    def actualizar_todas_las_variables_paralelo(self, n: int = 20):
        """
        Actualizar TODAS las variables EN PARALELO usando ThreadPoolExecutor

        ESTRATEGIA HPC:
        1. Variables rápidas (Yahoo, FRED API): ThreadPoolExecutor con max_workers=20
        2. Variables lentas (SUAMECA, Selenium): ThreadPoolExecutor con max_workers=4
        3. Procesamiento de resultados: Polars (10x más rápido que Pandas)
        """

        logger.info("="*100)
        logger.info("SISTEMA HPC V3 - ACTUALIZACION PARALELA MASIVA")
        logger.info("="*100)
        logger.info(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Variables: {len(self.diccionario)}")
        logger.info(f"Registros por variable: {n}")
        logger.info(f"Max workers: {self.max_workers}")
        logger.info("="*100)

        self.stats['total'] = len(self.diccionario)
        inicio = time.time()

        # Separar variables por tipo (para optimizar paralelización)
        variables_rapidas = []  # Yahoo, FRED, Investing
        variables_lentas = []   # SUAMECA, Selenium, Fedesarrollo

        for idx, row in self.diccionario.iterrows():
            codigo = str(row.get('CODIGO_FUENTE', row.get('CODIGO', ''))).strip()
            var_nombre = str(row.get('VARIABLE_NUEVA_ESTANDAR', row.get('VARIABLE', ''))).strip()

            # Determinar si es rápida o lenta
            es_lenta = False
            for key in ['IBR', 'TPM', 'ITCR', 'FDIIN', 'FDIOUT', 'IED', 'IDCE', 'CACCT', 'TOT', 'RESINT', 'CCI', 'ICI', 'IPCCOL']:
                if codigo == key or var_nombre.endswith(f"_{key}"):
                    es_lenta = True
                    break

            if es_lenta:
                variables_lentas.append((idx, row, n))
            else:
                variables_rapidas.append((idx, row, n))

        logger.info(f"Variables rápidas: {len(variables_rapidas)}")
        logger.info(f"Variables lentas: {len(variables_lentas)}")

        resultados_todos = []

        # FASE 1: Procesar variables rápidas en paralelo (max_workers=20)
        logger.info("\n[FASE 1] Procesando variables rápidas en paralelo...")
        with ThreadPoolExecutor(max_workers=min(20, len(variables_rapidas))) as executor:
            futures = {executor.submit(self._obtener_datos_variable_single, var_data): var_data
                      for var_data in variables_rapidas}

            for future in as_completed(futures):
                try:
                    resultado = future.result()
                    resultados_todos.append(resultado)

                    if resultado['datos'] is not None:
                        self.stats['disponibles'] += 1
                    else:
                        self.stats['no_disponibles'] += 1
                except Exception as e:
                    var_data = futures[future]
                    idx, row, _ = var_data
                    var_nombre = str(row.get('VARIABLE_NUEVA_ESTANDAR', '')).strip()
                    logger.error(f"  [{idx+1}] {var_nombre} - ERROR: {str(e)}")
                    self.stats['no_disponibles'] += 1

        # FASE 2: Procesar variables lentas en paralelo (max_workers=4 para no saturar)
        logger.info("\n[FASE 2] Procesando variables lentas en paralelo...")
        with ThreadPoolExecutor(max_workers=min(4, len(variables_lentas))) as executor:
            futures = {executor.submit(self._obtener_datos_variable_single, var_data): var_data
                      for var_data in variables_lentas}

            for future in as_completed(futures):
                try:
                    resultado = future.result()
                    resultados_todos.append(resultado)

                    if resultado['datos'] is not None:
                        self.stats['disponibles'] += 1
                    else:
                        self.stats['no_disponibles'] += 1
                except Exception as e:
                    var_data = futures[future]
                    idx, row, _ = var_data
                    var_nombre = str(row.get('VARIABLE_NUEVA_ESTANDAR', '')).strip()
                    logger.error(f"  [{idx+1}] {var_nombre} - ERROR: {str(e)}")
                    self.stats['no_disponibles'] += 1

        self.resultados = resultados_todos
        tiempo_total = time.time() - inicio

        logger.info("\n" + "="*100)
        logger.info(f"[HPC] Procesamiento completado en {tiempo_total:.2f} segundos ({tiempo_total/60:.2f} minutos)")
        logger.info(f"[HPC] Velocidad: {len(self.diccionario)/tiempo_total:.2f} variables/segundo")
        logger.info(f"[HPC] Cache hits: {self.stats['cache_hits']}")
        logger.info(f"[HPC] Cache misses: {self.stats['cache_misses']}")
        logger.info("="*100)

    def generar_datasets_polars(self):
        """Generar datasets usando Polars (10x más rápido que Pandas)"""

        logger.info("\n" + "="*100)
        logger.info("GENERANDO DATASETS CON POLARS (HPC)")
        logger.info("="*100)

        # Agrupar por frecuencia
        datos_por_frecuencia = {'D': [], 'M': [], 'Q': []}

        for resultado in self.resultados:
            if resultado['datos'] is not None:
                freq = resultado['Frecuencia']
                var_nombre = resultado['Variable']

                # Convertir a Polars para máxima velocidad
                df_pl = pl.from_pandas(resultado['datos'])
                df_pl = df_pl.rename({'valor': var_nombre})

                if freq in datos_por_frecuencia:
                    datos_por_frecuencia[freq].append((var_nombre, df_pl))

        # Generar datasets
        frecuencia_nombres = {'D': 'diarios', 'M': 'mensuales', 'Q': 'trimestrales'}

        for freq_code, freq_nombre in frecuencia_nombres.items():
            if len(datos_por_frecuencia[freq_code]) > 0:
                logger.info(f"\nGenerando dataset {freq_nombre.upper()} con Polars...")

                # Combinar usando Polars (outer join)
                datasets = datos_por_frecuencia[freq_code]

                # Iniciar con el primer dataset
                df_combined = datasets[0][1]

                # Hacer outer join con todos los demás (usar 'full' en vez de 'outer' para Polars)
                for var_nombre, df in datasets[1:]:
                    df_combined = df_combined.join(df, on='fecha', how='full', suffix='_temp', coalesce=True)

                # Ordenar por fecha descendente
                df_combined = df_combined.sort('fecha', descending=True)

                # Guardar (convertir a Pandas solo para guardar por compatibilidad)
                output_dataset = self.base_path / "storage" / "datasets" / f"datos_{freq_nombre}_hpc.csv"

                # Retry logic para manejar bloqueos de OneDrive
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        df_combined.to_pandas().to_csv(output_dataset, index=False, encoding='utf-8-sig')
                        break
                    except PermissionError as e:
                        if retry < max_retries - 1:
                            logger.warning(f"  [RETRY {retry+1}/{max_retries}] Archivo bloqueado, esperando 2s...")
                            time.sleep(2)
                        else:
                            # Intentar guardar con nombre alternativo
                            alt_output = output_dataset.parent / f"datos_{freq_nombre}_hpc_new.csv"
                            logger.warning(f"  [FALLBACK] Guardando como: {alt_output}")
                            df_combined.to_pandas().to_csv(alt_output, index=False, encoding='utf-8-sig')

                logger.info(f"  [OK] Dataset {freq_nombre}: {output_dataset}")
                logger.info(f"      Variables: {len(datasets)}")
                logger.info(f"      Filas: {df_combined.shape[0]}")

        logger.info("\n" + "="*100)


def main():
    """Funcion principal HPC"""

    import time
    inicio_total = time.time()

    # Buscar diccionario en config/ (un nivel arriba de orchestrators/)
    diccionario_path = Path(__file__).parent.parent / "config" / "DICCIONARIO_MACROECONOMICOS_FINAL.csv"

    # Fallback: buscar en el mismo directorio
    if not diccionario_path.exists():
        diccionario_path = Path(__file__).parent / "DICCIONARIO_MACROECONOMICOS_FINAL.csv"

    # Crear actualizador HPC con 10 workers
    actualizador = ActualizadorHPCV3(diccionario_path, max_workers=10)

    # Actualizar todas las variables EN PARALELO
    actualizador.actualizar_todas_las_variables_paralelo(n=20)

    # Generar datasets con Polars
    actualizador.generar_datasets_polars()

    # Guardar resultados
    df_resultados = pd.DataFrame(actualizador.resultados)
    output_path = actualizador.base_path / "resultados_actualizacion_hpc_v3.csv"
    df_resultados.to_csv(output_path, index=False, encoding='utf-8-sig')

    tiempo_final = time.time() - inicio_total

    logger.info("\n" + "="*100)
    logger.info("[HPC] ACTUALIZACION COMPLETA HPC V3 FINALIZADA")
    logger.info(f"[HPC] TIEMPO TOTAL: {tiempo_final:.2f} segundos ({tiempo_final/60:.2f} minutos)")
    logger.info(f"[HPC] SPEEDUP vs V2: {6.6*60/tiempo_final:.2f}x más rápido")
    logger.info("="*100)


if __name__ == '__main__':
    main()
