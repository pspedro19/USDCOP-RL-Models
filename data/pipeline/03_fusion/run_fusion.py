"""
================================================================================
MACRO DATA FUSION - Consolidacion de Datos Macroeconomicos por Frecuencia
================================================================================

Lee el diccionario de variables y genera 3 datasets consolidados:
    1. DATASET_MACRO_DAILY.csv      - Variables diarias (D) - outer join por fecha
    2. DATASET_MACRO_MONTHLY.csv    - Variables mensuales (M) - outer join por fecha
    3. DATASET_MACRO_QUARTERLY.csv  - Variables trimestrales (Q) - outer join por fecha

Caracteristicas v4.1:
    - Sistema inteligente de deteccion de formatos de fecha
    - Sistema inteligente de deteccion de formatos numericos (MEJORADO)
      * Detecta posicion del separador decimal (ultimo separador)
      * Valida patron de miles (grupos de 3 digitos)
      * Override automatico para fuentes conocidas (Investing.com, FRED, Yahoo)
    - Soporte para tablas pivot (años en columnas, meses en filas)
    - Soporte para fechas con periodo de referencia: "Oct 24, 2025 (Sep)"

Estructura del proyecto:
    Fuse Macro/
    ├── config/
    │   └── DICCIONARIO_MACROECONOMICOS_FINAL.csv
    ├── output/
    │   ├── DATASET_MACRO_DAILY.csv
    │   ├── DATASET_MACRO_MONTHLY.csv
    │   └── DATASET_MACRO_QUARTERLY.csv
    ├── logs/
    │   └── REPORTE_PROCESAMIENTO.csv
    └── scripts/
        └── macro_data_fusion.py

Uso:
    cd "Fuse Macro/scripts"
    python macro_data_fusion.py

Version: 4.1 - Con deteccion inteligente mejorada de formatos numericos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import sys
import re
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACION PARAMETRIZADA
# ==============================================================================

class Config:
    """
    Configuracion del sistema - MODIFICAR AQUI SEGUN NECESIDADES
    """
    # Rutas base (relativas al directorio del script)
    BASE_DIR = Path(__file__).parent  # 03_fusion/
    PIPELINE_DIR = BASE_DIR.parent  # pipeline/

    # Carpetas de entrada
    CONFIG_DIR = PIPELINE_DIR / "00_config"
    MACROS_RAW_DIR = PIPELINE_DIR / "01_sources"

    # Carpetas de salida
    OUTPUT_DIR = BASE_DIR / "output"
    LOGS_DIR = BASE_DIR / "logs"

    # Archivos de entrada
    DICCIONARIO_FILE = "DICCIONARIO_MACROECONOMICOS_FINAL.csv"
    DELIMITADOR_DICCIONARIO = ";"

    # Archivos de salida
    OUTPUT_DAILY = "DATASET_MACRO_DAILY.csv"
    OUTPUT_MONTHLY = "DATASET_MACRO_MONTHLY.csv"
    OUTPUT_QUARTERLY = "DATASET_MACRO_QUARTERLY.csv"
    OUTPUT_REPORTE = "REPORTE_PROCESAMIENTO.csv"

    # Parametros de procesamiento
    FECHA_INICIO = "2020-01-01"      # Filtrar desde esta fecha
    FECHA_FIN = "2025-12-31"         # Filtrar hasta esta fecha
    SOLO_DIAS_LABORABLES = True      # True = solo Lun-Vie para datos diarios

    @classmethod
    def get_diccionario_path(cls) -> Path:
        return cls.CONFIG_DIR / cls.DICCIONARIO_FILE

    # Mapeo de categorías del diccionario a nombres de carpetas en pipeline
    # Nueva estructura con prefijos numéricos en 01_sources/
    CATEGORIA_TO_FOLDER = {
        'COMMODITIES': '01_commodities',
        'COUNTRY_RISK': '06_country_risk',
        'ECONOMIC_GROWTH': '07_economic_growth',
        'EQUITIES': '14_equities',
        'EXCHANGE_RATES': '02_exchange_rates',
        'EXCHANGE RATES': '02_exchange_rates',
        'FIXED_INCOME': '03_fixed_income',
        'FOREIGN_TRADE': '08_foreign_trade',
        'INFLATION': '04_inflation',
        'LABOR_MARKET': '10_labor_market',
        'MONEY_SUPPLY': '11_money_supply',
        'POLICY_RATES': '05_policy_rates',
        'POLICY RATES': '05_policy_rates',
        'PRODUCTION': '15_production',
        'RESERVES_BALANCE_OF_PAYMENTS': '09_reserves_bop',
        'RESERVES_BALANCE_PAYMENTS': '09_reserves_bop',
        'SENTIMENT': '12_sentiment',
        'VOLATILITY': '13_volatility',
    }

    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        return cls.OUTPUT_DIR / filename

    @classmethod
    def get_log_path(cls, filename: str) -> Path:
        return cls.LOGS_DIR / filename

    @classmethod
    def get_raw_file_path(cls, categoria: str, nombre_archivo: str) -> Path:
        """Construye la ruta al archivo raw basándose en la categoría y nombre"""
        folder = cls.CATEGORIA_TO_FOLDER.get(categoria, categoria.lower().replace(' ', '_'))
        return cls.MACROS_RAW_DIR / folder / nombre_archivo


# ==============================================================================
# CONFIGURACION ENCODING WINDOWS
# ==============================================================================

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ==============================================================================
# MESES EN ESPANOL E INGLES
# ==============================================================================

MESES_ESP = {'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12,
             'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
             'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12}

MESES_ENG = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
             'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
             'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}


# ==============================================================================
# SISTEMA INTELIGENTE DE DETECCION DE FORMATOS
# ==============================================================================

class SmartDateDetector:
    """
    Detecta automaticamente el formato de fecha analizando patrones en los datos.
    """

    # Patrones de fecha comunes
    PATTERNS = {
        'YYYY-MM-DD': r'^\d{4}-\d{2}-\d{2}$',
        'DD/MM/YYYY': r'^\d{1,2}/\d{1,2}/\d{4}$',
        'MM/DD/YYYY': r'^\d{1,2}/\d{1,2}/\d{4}$',  # Ambiguo con DD/MM/YYYY
        'YYYY/MM/DD': r'^\d{4}/\d{1,2}/\d{1,2}$',
        'ddMMMyy': r'^\d{2}[a-zA-Z]{3}\d{2}$',  # 01Ene20
        'mmm-yy': r'^[a-zA-Z]{3}-\d{2}$',  # nov-24
        'Mmm DD, YYYY': r'^[A-Za-z]{3}\s+\d{1,2},\s*\d{4}',  # Oct 24, 2025
        'Mmm DD, YYYY (Mmm)': r'^[A-Za-z]{3}\s+\d{1,2},\s*\d{4}\s*\([A-Za-z]+\)$',  # Oct 24, 2025 (Sep)
    }

    @classmethod
    def detect_format(cls, dates_sample: List[str], fuente: str = '') -> Dict:
        """
        Analiza una muestra de fechas y detecta el formato mas probable.

        Returns:
            Dict con 'format', 'confidence', 'ambiguous', 'recommendation'
        """
        if not dates_sample:
            return {'format': 'unknown', 'confidence': 0, 'ambiguous': True}

        # Limpiar muestra
        clean_sample = []
        for d in dates_sample[:50]:  # Analizar max 50 muestras
            if pd.notna(d):
                s = str(d).strip()
                if s and not any(x in s.lower() for x in ['fecha', 'date', 'periodo', 'unnamed']):
                    clean_sample.append(s)

        if not clean_sample:
            return {'format': 'unknown', 'confidence': 0, 'ambiguous': True}

        # Detectar formato
        result = cls._analyze_patterns(clean_sample, fuente)
        return result

    @classmethod
    def _analyze_patterns(cls, samples: List[str], fuente: str) -> Dict:
        """Analiza patrones en las muestras"""

        # Contador de patrones detectados
        pattern_counts = Counter()
        ambiguous_count = 0

        for sample in samples:
            # Patron con fecha y periodo de referencia: "Oct 24, 2025 (Sep)"
            if re.match(cls.PATTERNS['Mmm DD, YYYY (Mmm)'], sample):
                pattern_counts['Mmm DD, YYYY (ref)'] += 1
                continue

            # Patron: "Oct 24, 2025"
            if re.match(cls.PATTERNS['Mmm DD, YYYY'], sample):
                pattern_counts['Mmm DD, YYYY'] += 1
                continue

            # YYYY-MM-DD
            if re.match(cls.PATTERNS['YYYY-MM-DD'], sample):
                pattern_counts['YYYY-MM-DD'] += 1
                continue

            # YYYY/MM/DD
            if re.match(cls.PATTERNS['YYYY/MM/DD'], sample):
                pattern_counts['YYYY/MM/DD'] += 1
                continue

            # ddMMMyy (01Ene20)
            if re.match(cls.PATTERNS['ddMMMyy'], sample):
                pattern_counts['ddMMMyy'] += 1
                continue

            # mmm-yy (nov-24)
            if re.match(cls.PATTERNS['mmm-yy'], sample):
                pattern_counts['mmm-yy'] += 1
                continue

            # DD/MM/YYYY o MM/DD/YYYY
            if '/' in sample:
                parts = sample.split('/')
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    p0, p1 = int(parts[0]), int(parts[1])

                    # Si alguno > 12, no es ambiguo
                    if p0 > 12:
                        pattern_counts['DD/MM/YYYY'] += 1
                    elif p1 > 12:
                        pattern_counts['MM/DD/YYYY'] += 1
                    else:
                        # Ambiguo: ambos <= 12
                        ambiguous_count += 1
                        # Usar fuente para decidir
                        if 'investing' in fuente.lower():
                            pattern_counts['MM/DD/YYYY'] += 1
                        else:
                            pattern_counts['DD/MM/YYYY'] += 1

        if not pattern_counts:
            return {'format': 'unknown', 'confidence': 0, 'ambiguous': True}

        # Formato mas comun
        most_common = pattern_counts.most_common(1)[0]
        total = sum(pattern_counts.values())
        confidence = most_common[1] / total if total > 0 else 0

        return {
            'format': most_common[0],
            'confidence': confidence,
            'ambiguous': ambiguous_count > len(samples) * 0.3,  # >30% ambiguo
            'pattern_counts': dict(pattern_counts),
            'samples_analyzed': len(samples)
        }


class SmartNumericDetector:
    """
    Detecta automaticamente el formato numerico (decimal con coma o punto).

    Estrategia mejorada v4.1:
    1. Analiza la POSICION del ultimo separador (ese es el decimal)
    2. Verifica consistencia de separadores de miles (grupos de 3)
    3. Considera el contexto (fuente de datos)

    Formatos soportados:
    - Americano: 1,234.56 o 1234.56 (punto decimal, coma miles)
    - Europeo: 1.234,56 o 1234,56 (coma decimal, punto miles)
    """

    @classmethod
    def detect_format(cls, values_sample: List, fuente: str = '') -> Dict:
        """
        Analiza una muestra de valores y detecta el formato numerico.

        Returns:
            Dict con 'decimal_sep', 'thousands_sep', 'has_percentage', 'confidence'
        """
        if not values_sample:
            return {'decimal_sep': '.', 'thousands_sep': ',', 'has_percentage': False, 'confidence': 0}

        # Convertir a strings y limpiar
        clean_sample = []
        for v in values_sample[:100]:
            if pd.notna(v):
                s = str(v).strip()
                if s and s not in ['nan', 'NaN', '', '-']:
                    clean_sample.append(s)

        if not clean_sample:
            return {'decimal_sep': '.', 'thousands_sep': ',', 'has_percentage': False, 'confidence': 0}

        return cls._analyze_numeric_patterns(clean_sample, fuente)

    @classmethod
    def _analyze_numeric_patterns(cls, samples: List[str], fuente: str = '') -> Dict:
        """
        Analiza patrones numericos con estrategia mejorada.

        Reglas clave:
        1. Si hay ambos , y . en el numero, el ULTIMO es el decimal
        2. Si solo hay uno, verificar patron de miles (grupos de 3)
        3. Investing.com SIEMPRE usa formato americano
        """

        has_percentage = any('%' in s for s in samples)

        # Fuentes conocidas con formato fijo
        fuente_lower = str(fuente).lower()
        if any(x in fuente_lower for x in ['investing', 'fred', 'yahoo']):
            # Estas fuentes SIEMPRE usan formato americano
            return {
                'decimal_sep': '.',
                'thousands_sep': ',',
                'has_percentage': has_percentage,
                'confidence': 1.0,
                'source_override': True
            }

        # Analizar patrones
        comma_decimal_score = 0  # 1.234,56 (europeo)
        point_decimal_score = 0  # 1,234.56 (americano)
        total_analyzed = 0

        for sample in samples:
            s = sample.replace('%', '').replace(' ', '').strip()
            if not s or s in ['-', 'nan', 'NaN']:
                continue

            has_comma = ',' in s
            has_point = '.' in s

            # Caso 1: Tiene ambos separadores
            if has_comma and has_point:
                total_analyzed += 1
                # El ULTIMO separador es el decimal
                last_comma = s.rfind(',')
                last_point = s.rfind('.')

                if last_comma > last_point:
                    # Coma es decimal: 1.234,56 (europeo)
                    # Verificar que punto sea separador de miles valido
                    if cls._is_valid_thousands_separator(s, '.'):
                        comma_decimal_score += 2
                    else:
                        comma_decimal_score += 1
                else:
                    # Punto es decimal: 1,234.56 (americano)
                    # Verificar que coma sea separador de miles valido
                    if cls._is_valid_thousands_separator(s, ','):
                        point_decimal_score += 2
                    else:
                        point_decimal_score += 1

            # Caso 2: Solo tiene coma
            elif has_comma and not has_point:
                total_analyzed += 1
                # Verificar si es decimal o miles
                parts = s.split(',')
                if len(parts) == 2:
                    # Si la parte decimal tiene 1-2 digitos, es decimal europeo
                    if len(parts[1]) <= 2 and parts[1].isdigit():
                        comma_decimal_score += 1
                    # Si tiene 3 digitos exactos y parte izquierda es corta, podria ser miles
                    elif len(parts[1]) == 3 and parts[1].isdigit() and len(parts[0]) <= 3:
                        # Ambiguo, podria ser 1,234 (miles) o 1,234 (decimal con 3 decimales)
                        # Por defecto americano (coma = miles)
                        point_decimal_score += 0.5
                    else:
                        comma_decimal_score += 0.5

            # Caso 3: Solo tiene punto
            elif has_point and not has_comma:
                total_analyzed += 1
                # Verificar si es decimal o miles
                parts = s.split('.')
                if len(parts) == 2:
                    # Si la parte decimal tiene 1-4 digitos, es decimal americano
                    if len(parts[1]) <= 4 and parts[1].isdigit():
                        point_decimal_score += 1
                    # Si tiene 3 digitos exactos y parte izquierda es corta, podria ser miles europeo
                    elif len(parts[1]) == 3 and parts[1].isdigit() and len(parts[0]) <= 3:
                        # Ambiguo pero menos comun, por defecto americano
                        point_decimal_score += 0.5
                    else:
                        point_decimal_score += 0.5

        # Determinar resultado
        if total_analyzed == 0:
            return {
                'decimal_sep': '.',
                'thousands_sep': ',',
                'has_percentage': has_percentage,
                'confidence': 0.5
            }

        total_score = comma_decimal_score + point_decimal_score

        if comma_decimal_score > point_decimal_score:
            confidence = comma_decimal_score / total_score if total_score > 0 else 0.5
            return {
                'decimal_sep': ',',
                'thousands_sep': '.',
                'has_percentage': has_percentage,
                'confidence': confidence
            }
        else:
            confidence = point_decimal_score / total_score if total_score > 0 else 0.5
            return {
                'decimal_sep': '.',
                'thousands_sep': ',',
                'has_percentage': has_percentage,
                'confidence': confidence
            }

    @classmethod
    def _is_valid_thousands_separator(cls, s: str, sep: str) -> bool:
        """
        Verifica si un separador es valido como separador de miles.
        Los miles deben estar en grupos de exactamente 3 digitos.
        """
        # Encontrar todos los grupos separados
        parts = s.replace(',', '|').replace('.', '|').split('|')

        # El primer grupo puede tener 1-3 digitos
        # Los siguientes deben tener exactamente 3 digitos (excepto el ultimo que es decimal)

        if len(parts) < 2:
            return True

        # Excluir el ultimo (que es la parte decimal)
        thousands_parts = parts[:-1]

        if len(thousands_parts) == 0:
            return True

        # Primer grupo: 1-3 digitos
        if not (1 <= len(thousands_parts[0]) <= 3):
            return False

        # Grupos intermedios: exactamente 3 digitos
        for part in thousands_parts[1:]:
            if len(part) != 3:
                return False

        return True


class SmartFileStructureDetector:
    """
    Detecta si un archivo tiene estructura estandar (fecha, valor) o es tabla pivot.
    """

    MESES_NOMBRES = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                     'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']

    @classmethod
    def detect_structure(cls, df: pd.DataFrame) -> Dict:
        """
        Detecta la estructura del DataFrame.

        Returns:
            Dict con 'structure_type', 'date_column', 'value_column', 'pivot_info'
        """
        if df is None or df.empty:
            return {'structure_type': 'unknown', 'error': 'DataFrame vacio'}

        # Verificar si es tabla pivot (años en columnas)
        pivot_info = cls._check_pivot_table(df)
        if pivot_info['is_pivot']:
            return {
                'structure_type': 'pivot',
                'pivot_info': pivot_info
            }

        # Estructura estandar: buscar columnas de fecha y valor
        date_col = cls._find_date_column(df)
        value_col = cls._find_value_column(df, date_col)

        return {
            'structure_type': 'standard',
            'date_column': date_col,
            'value_column': value_col
        }

    @classmethod
    def _check_pivot_table(cls, df: pd.DataFrame) -> Dict:
        """Verifica si es tabla pivot con años en columnas y meses en filas"""

        # Buscar columnas que sean años (2015-2030)
        year_cols = []
        for col in df.columns:
            col_str = str(col).strip()
            # Remover asteriscos y espacios
            col_clean = re.sub(r'[\*\s]', '', col_str)
            try:
                year = int(float(col_clean))
                if 2000 <= year <= 2030:
                    year_cols.append((col, year))
            except (ValueError, TypeError):
                pass

        # Si hay al menos 3 columnas de años, probablemente es pivot
        if len(year_cols) >= 3:
            # Buscar columna de meses
            mes_col = None
            for col in df.columns:
                # Verificar si contiene nombres de meses
                col_values = df[col].dropna().astype(str).str.lower().tolist()
                month_matches = sum(1 for v in col_values if any(m in v for m in cls.MESES_NOMBRES))
                if month_matches >= 6:  # Al menos 6 meses
                    mes_col = col
                    break

            if mes_col:
                return {
                    'is_pivot': True,
                    'year_columns': year_cols,
                    'month_column': mes_col,
                    'years_found': [y for _, y in year_cols]
                }

        return {'is_pivot': False}

    @classmethod
    def _find_date_column(cls, df: pd.DataFrame) -> Optional[str]:
        """Busca la columna de fecha"""
        for col in df.columns:
            col_lower = str(col).lower()
            if any(x in col_lower for x in ['fecha', 'date', 'periodo', 'observation', 'release']):
                return col
        return df.columns[0] if len(df.columns) > 0 else None

    @classmethod
    def _find_value_column(cls, df: pd.DataFrame, exclude_col: str = None) -> Optional[str]:
        """Busca la columna de valor"""
        for col in df.columns:
            if col == exclude_col:
                continue
            # Preferir columnas numericas
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                if df[col].notna().sum() > 0:
                    return col

        # Buscar por nombre
        for col in df.columns:
            if col == exclude_col:
                continue
            col_lower = str(col).lower()
            if any(x in col_lower for x in ['price', 'valor', 'actual', 'value', 'close', 'adj']):
                return col

        # Retornar segunda columna si existe
        cols = [c for c in df.columns if c != exclude_col]
        return cols[0] if cols else None


def unpivot_table(df: pd.DataFrame, pivot_info: Dict, var_name: str) -> Optional[pd.DataFrame]:
    """
    Convierte tabla pivot (años en columnas, meses en filas) a formato estandar.
    """
    try:
        mes_col = pivot_info['month_column']
        year_cols = pivot_info['year_columns']

        results = []

        for idx, row in df.iterrows():
            mes_str = str(row[mes_col]).strip().lower() if pd.notna(row[mes_col]) else ''

            # Encontrar numero del mes
            mes_num = None
            for nombre, num in MESES_ESP.items():
                if nombre in mes_str:
                    mes_num = num
                    break

            if mes_num is None:
                continue

            # Extraer valor para cada año
            for col_name, year in year_cols:
                if year < 2020:  # Filtrar años anteriores a 2020
                    continue

                valor = row[col_name]
                if pd.notna(valor):
                    try:
                        valor_num = float(valor)
                        # Crear fecha (ultimo dia del mes)
                        if mes_num == 12:
                            fecha = pd.Timestamp(year=year, month=12, day=31)
                        else:
                            fecha = pd.Timestamp(year=year, month=mes_num + 1, day=1) - pd.Timedelta(days=1)

                        results.append({'fecha': fecha, var_name: valor_num})
                    except (ValueError, TypeError):
                        pass

        if results:
            return pd.DataFrame(results).sort_values('fecha').drop_duplicates(subset=['fecha'], keep='last')

        return None
    except Exception as e:
        print(f"    Error en unpivot: {e}")
        return None


def process_dane_balanza_comercial(path: str, var_name: str, fecha_inicio: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Procesa archivo DANE de Balanza Comercial mensual (anex-BCOM-Mensual-*.xlsx)

    Estructura:
    - Header en fila 8: Mes, Exportaciones, Importaciones, Balanza comercial
    - Datos desde fila 9
    - Valores en MILLONES de dolares FOB

    Args:
        path: Ruta al archivo
        var_name: Nombre de la variable (EXPUSD o IMPUSD)
        fecha_inicio: Fecha minima a incluir

    Returns:
        DataFrame con columnas ['fecha', var_name]
    """
    try:
        # Leer archivo con header en fila 8
        df = pd.read_excel(path, header=8)

        if df.empty:
            return None

        # Renombrar columnas
        df.columns = ['Mes', 'Exportaciones', 'Importaciones', 'Balanza']

        # Determinar que columna usar segun el nombre de variable
        if 'expusd' in var_name.lower() or 'export' in var_name.lower():
            value_col = 'Exportaciones'
        elif 'impusd' in var_name.lower() or 'import' in var_name.lower():
            value_col = 'Importaciones'
        else:
            value_col = 'Balanza'

        print(f"    [DANE BCOM] Extrayendo columna: {value_col}")

        # =====================================================================
        # DETECCION INTELIGENTE DE FORMATO NUMERICO
        # =====================================================================
        value_samples = df[value_col].dropna().tolist()[:100]
        numeric_format = SmartNumericDetector.detect_format(value_samples, fuente='DANE')
        decimal_sep = numeric_format.get('decimal_sep', '.')

        print(f"    [DANE BCOM] Formato detectado: decimal='{decimal_sep}' (confidence={numeric_format.get('confidence', 0):.2f})")

        # Funcion para convertir valor segun formato detectado
        def convert_value(val):
            if pd.isna(val):
                return None

            # Si ya es numerico, retornar directamente
            if isinstance(val, (int, float)):
                return float(val)

            # Convertir a string y limpiar
            s = str(val).strip().replace('%', '').replace(' ', '')
            if not s or s in ['-', 'nan', 'NaN']:
                return None

            try:
                if decimal_sep == ',':
                    # Formato europeo: 1.234,56 -> quitar puntos de miles, coma a punto
                    s = s.replace('.', '').replace(',', '.')
                else:
                    # Formato americano: 1,234.56 -> quitar comas de miles
                    s = s.replace(',', '')

                return float(s)
            except (ValueError, TypeError):
                return None

        # Filtrar filas validas
        results = []
        for idx, row in df.iterrows():
            fecha = row['Mes']
            valor = row[value_col]

            # Verificar si es fecha valida
            if pd.isna(fecha):
                continue

            try:
                if isinstance(fecha, pd.Timestamp):
                    fecha_ts = fecha
                elif isinstance(fecha, datetime):
                    fecha_ts = pd.Timestamp(fecha)
                elif isinstance(fecha, str):
                    # Saltar filas que no son fechas
                    if any(x in fecha.lower() for x in ['mes', 'fuente', 'cifras', 'actualizado', 'nan']):
                        continue
                    fecha_ts = pd.to_datetime(fecha)
                else:
                    continue

                # Convertir valor usando detector inteligente
                valor_num = convert_value(valor)

                # Verificar si valor es valido y positivo
                if valor_num is not None and valor_num > 0 and fecha_ts >= fecha_inicio:
                    # Redondear a 1 decimal (valores en millones USD)
                    valor_num = round(valor_num, 1)
                    results.append({'fecha': fecha_ts, var_name: valor_num})

            except:
                continue

        if results:
            df_result = pd.DataFrame(results)
            df_result = df_result.sort_values('fecha').drop_duplicates(subset=['fecha'], keep='last')
            print(f"    [DANE BCOM] Extraidos {len(df_result)} registros desde {df_result['fecha'].min().date()} hasta {df_result['fecha'].max().date()}")
            return df_result

        return None

    except Exception as e:
        print(f"    Error procesando DANE BCOM: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_fedesarrollo_confianza(path: str, var_name: str, fecha_inicio: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Procesa archivos de Fedesarrollo de Índices de Confianza (ICC e ICI)

    ICC (Índice de Confianza del Consumidor):
    - Header en fila 0 con nombres largos
    - Columna 0: fechas (mmm-yy)
    - Columna 1: ICC values

    ICI (Índice de Confianza Industrial):
    - Columna 0: fechas (mmm-yy)
    - Columna 1: ICI values

    Args:
        path: Ruta al archivo
        var_name: Nombre de la variable (CCI o ICI)
        fecha_inicio: Fecha minima a incluir

    Returns:
        DataFrame con columnas ['fecha', var_name]
    """
    try:
        # Determinar tipo de archivo
        is_icc = 'cci' in var_name.lower() or 'consumidor' in path.lower()
        is_ici = 'ici' in var_name.lower() and not is_icc

        if is_icc:
            # ICC: Header especial, skiprows=1 para saltar el header largo
            df = pd.read_excel(path, header=None, skiprows=1)
            fecha_col = 0
            valor_col = 1
            print(f"    [FEDESARROLLO ICC] Procesando archivo ICC...")
        else:
            # ICI: Header normal
            df = pd.read_excel(path)
            fecha_col = df.columns[0]
            valor_col = df.columns[1]
            print(f"    [FEDESARROLLO ICI] Procesando archivo ICI...")

        if df.empty:
            return None

        # Obtener columnas correctas
        if is_icc:
            df_work = df[[0, 1]].copy()
            df_work.columns = ['fecha_raw', 'valor_raw']
        else:
            df_work = df[[fecha_col, valor_col]].copy()
            df_work.columns = ['fecha_raw', 'valor_raw']

        # =====================================================================
        # DETECCION INTELIGENTE DE FORMATO NUMERICO
        # =====================================================================
        value_samples = df_work['valor_raw'].dropna().astype(str).tolist()[:100]
        numeric_format = SmartNumericDetector.detect_format(value_samples, fuente='Fedesarrollo')
        decimal_sep = numeric_format.get('decimal_sep', '.')

        print(f"    [FEDESARROLLO] Formato detectado: decimal='{decimal_sep}' (confidence={numeric_format.get('confidence', 0):.2f})")

        # Funcion para convertir valor
        def convert_value(val):
            if pd.isna(val):
                return None

            # Si ya es numerico
            if isinstance(val, (int, float)):
                return float(val)

            # Convertir string
            s = str(val).strip().replace('%', '').replace(' ', '')
            if not s or s in ['-', 'nan', 'NaN']:
                return None

            try:
                if decimal_sep == ',':
                    s = s.replace('.', '').replace(',', '.')
                else:
                    s = s.replace(',', '')
                return float(s)
            except (ValueError, TypeError):
                return None

        # Procesar filas
        results = []
        for idx, row in df_work.iterrows():
            fecha_raw = row['fecha_raw']
            valor_raw = row['valor_raw']

            if pd.isna(fecha_raw):
                continue

            try:
                # Parsear fecha (formato mmm-yy)
                fecha_str = str(fecha_raw).strip().lower()

                # Saltar filas no válidas
                if any(x in fecha_str for x in ['indice', 'índice', 'fuente', 'nan', 'fecha']):
                    continue

                # Convertir mmm-yy a fecha
                fecha_ts = None
                for mes_esp, mes_num in MESES_ESP.items():
                    if fecha_str.startswith(mes_esp[:3]):
                        try:
                            year_str = fecha_str.split('-')[-1]
                            year = int(year_str)
                            if year < 100:
                                year = 2000 + year if year < 50 else 1900 + year
                            fecha_ts = pd.Timestamp(year=year, month=mes_num, day=1)
                            break
                        except:
                            continue

                if fecha_ts is None:
                    continue

                # Convertir valor
                valor_num = convert_value(valor_raw)

                if valor_num is not None and fecha_ts >= fecha_inicio:
                    # Redondear a 1 decimal
                    valor_num = round(valor_num, 1)
                    results.append({'fecha': fecha_ts, var_name: valor_num})

            except Exception as e:
                continue

        if results:
            df_result = pd.DataFrame(results)
            df_result = df_result.sort_values('fecha').drop_duplicates(subset=['fecha'], keep='last')
            print(f"    [FEDESARROLLO] Extraidos {len(df_result)} registros desde {df_result['fecha'].min().date()} hasta {df_result['fecha'].max().date()}")
            return df_result

        return None

    except Exception as e:
        print(f"    Error procesando Fedesarrollo: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_dane_imports_exports(path: str, var_name: str, fecha_inicio: pd.Timestamp) -> Optional[pd.DataFrame]:
    """
    Procesa archivos DANE de importaciones/exportaciones con formato especial:
    - Multiples paises con breakdown mensual
    - Fila "Total" por cada pais
    - Años en columnas

    Estrategia: Buscar todas las filas "Total" de cada pais y sumarlas por mes/año.
    """
    try:
        # Leer archivo saltando las primeras filas de header
        df = pd.read_excel(path, header=7)

        if df.empty:
            return None

        # Identificar columna de meses
        mes_col = None
        for col in df.columns:
            col_values = df[col].dropna().astype(str).str.lower().tolist()
            month_matches = sum(1 for v in col_values if v in ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                                                                'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre', 'total'])
            if month_matches > 10:
                mes_col = col
                break

        if mes_col is None:
            return None

        # Identificar columnas de años (buscar en fila 0 que tiene los años)
        year_cols = []
        first_row = df.iloc[0]
        for col in df.columns:
            val = first_row[col]
            if pd.notna(val):
                val_str = str(val).replace('*', '').strip()
                try:
                    year = int(float(val_str))
                    if 2020 <= year <= 2030:
                        year_cols.append((col, year))
                except (ValueError, TypeError):
                    pass

        if len(year_cols) < 1:
            return None

        print(f"    [DANE] Detectado: {len(year_cols)} años, columna meses: {mes_col}")

        # Sumar valores de todos los paises por mes
        # Filtrar solo las filas con meses válidos (excluyendo "Total")
        results = {}  # {(year, month): sum}

        for idx, row in df.iterrows():
            mes_str = str(row[mes_col]).strip().lower() if pd.notna(row[mes_col]) else ''

            # Solo procesar filas con nombres de meses (no "Total")
            mes_num = None
            meses_validos = {'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                           'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12}
            if mes_str in meses_validos:
                mes_num = meses_validos[mes_str]
            else:
                continue  # Saltar filas que no son meses

            for col_name, year in year_cols:
                valor = row[col_name]
                if pd.notna(valor):
                    try:
                        valor_num = float(valor)
                        if valor_num > 0:  # Solo sumar valores positivos
                            key = (year, mes_num)
                            if key not in results:
                                results[key] = 0
                            results[key] += valor_num
                    except (ValueError, TypeError):
                        pass

        if not results:
            return None

        # Convertir a DataFrame
        data = []
        for (year, month), total in results.items():
            if month == 12:
                fecha = pd.Timestamp(year=year, month=12, day=31)
            else:
                fecha = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(days=1)
            data.append({'fecha': fecha, var_name: total})

        df_result = pd.DataFrame(data)
        df_result = df_result[df_result['fecha'] >= fecha_inicio]
        df_result = df_result.sort_values('fecha').drop_duplicates(subset=['fecha'], keep='last')

        return df_result

    except Exception as e:
        print(f"    Error procesando DANE: {e}")
        return None


# ==============================================================================
# FUNCIONES DE PARSEO DE FECHAS
# ==============================================================================

def parsear_fecha(fecha_str: str, fuente: str = '', use_reference_period: bool = False) -> Optional[pd.Timestamp]:
    """
    Parsea una fecha desde string probando multiples formatos.

    Args:
        fecha_str: String con la fecha
        fuente: Fuente de datos (Investing.com, SUAMECA, FRED, etc.)
        use_reference_period: Si True, extrae el periodo de referencia de formatos como "Oct 24, 2025 (Sep)"
    """
    if pd.isna(fecha_str):
        return None

    fecha_str = str(fecha_str).strip()

    # Saltear headers y valores invalidos
    if any(x in fecha_str.lower() for x in ['fecha', 'date', 'periodo', 'unnamed', 'dd/mm', 'descargado']):
        return None

    # 0. Formato "Mmm DD, YYYY (Mmm)" - fecha con periodo de referencia
    # Ejemplo: "Oct 24, 2025 (Sep)" -> usar Sep 2025 si use_reference_period=True
    ref_match = re.match(r'^([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\s*\(([A-Za-z]+)\)$', fecha_str)
    if ref_match:
        mes_release = ref_match.group(1).lower()[:3]
        dia_release = int(ref_match.group(2))
        anio = int(ref_match.group(3))
        mes_ref = ref_match.group(4).lower()[:3]

        mes_num = MESES_ENG.get(mes_ref) or MESES_ESP.get(mes_ref)
        if mes_num and use_reference_period:
            # Usar periodo de referencia (mes del dato)
            # Fin del mes de referencia
            if mes_num == 12:
                return pd.Timestamp(year=anio, month=12, day=31)
            else:
                # El periodo de referencia puede ser del año anterior si mes_ref > mes_release
                anio_ref = anio
                mes_release_num = MESES_ENG.get(mes_release) or MESES_ESP.get(mes_release)
                if mes_release_num and mes_num > mes_release_num:
                    anio_ref = anio - 1
                return pd.Timestamp(year=anio_ref, month=mes_num + 1, day=1) - pd.Timedelta(days=1)
        elif mes_num:
            # Usar fecha de publicacion
            mes_pub_num = MESES_ENG.get(mes_release) or MESES_ESP.get(mes_release)
            if mes_pub_num:
                try:
                    return pd.Timestamp(year=anio, month=mes_pub_num, day=dia_release)
                except:
                    pass

    # 0b. Formato "Mmm DD, YYYY" sin periodo de referencia
    simple_match = re.match(r'^([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})$', fecha_str)
    if simple_match:
        mes_str = simple_match.group(1).lower()[:3]
        dia = int(simple_match.group(2))
        anio = int(simple_match.group(3))
        mes = MESES_ENG.get(mes_str) or MESES_ESP.get(mes_str)
        if mes:
            try:
                return pd.Timestamp(year=anio, month=mes, day=dia)
            except:
                pass

    # 1. Formato con barras (/, puede ser DD/MM/YYYY, MM/DD/YYYY, o YYYY/MM/DD)
    if '/' in fecha_str:
        parts = fecha_str.split('/')
        if len(parts) == 3:
            p0, p1, p2 = parts[0], parts[1], parts[2]

            # Si primer elemento tiene 4 digitos -> YYYY/MM/DD
            if len(p0) == 4:
                try:
                    return pd.to_datetime(fecha_str, format='%Y/%m/%d')
                except:
                    pass

            # Si tercer elemento tiene 4 digitos (año al final)
            if len(p2) == 4 and p0.isdigit() and p1.isdigit():
                n0, n1 = int(p0), int(p1)

                # Si segundo numero > 12, es MM/DD/YYYY (dia en segunda posicion)
                if n1 > 12:
                    try:
                        return pd.to_datetime(fecha_str, format='%m/%d/%Y')
                    except:
                        pass

                # Si primer numero > 12, es DD/MM/YYYY (dia en primera posicion)
                if n0 > 12:
                    try:
                        return pd.to_datetime(fecha_str, format='%d/%m/%Y')
                    except:
                        pass

                # Ambos <= 12: AMBIGUO - usar fuente para decidir
                # Investing.com usa MM/DD/YYYY (formato USA)
                # BanRep/SUAMECA usa DD/MM/YYYY (formato latinoamericano)
                if 'investing' in fuente.lower():
                    try:
                        return pd.to_datetime(fecha_str, format='%m/%d/%Y')
                    except:
                        pass
                else:
                    # Default: DD/MM/YYYY para fuentes latinoamericanas
                    try:
                        return pd.to_datetime(fecha_str, format='%d/%m/%Y')
                    except:
                        pass

                # Fallback al otro formato
                try:
                    return pd.to_datetime(fecha_str, format='%m/%d/%Y')
                except:
                    pass
                try:
                    return pd.to_datetime(fecha_str, format='%d/%m/%Y')
                except:
                    pass

    # 2. Formato YYYY-MM-DD (FRED)
    if '-' in fecha_str and len(fecha_str) >= 10:
        try:
            return pd.to_datetime(fecha_str, format='%Y-%m-%d')
        except:
            pass

    # 3. Formato ddMMMyy (01Ene20 - BCRP EMBI)
    if len(fecha_str) >= 7 and fecha_str[:2].isdigit():
        try:
            dia = int(fecha_str[:2])
            mes_str = fecha_str[2:5].lower()
            anio_str = fecha_str[5:]
            mes = MESES_ESP.get(mes_str) or MESES_ENG.get(mes_str)
            if mes:
                anio = int(anio_str)
                anio = 2000 + anio if anio < 100 else anio
                return pd.Timestamp(year=anio, month=mes, day=dia)
        except:
            pass

    # 4. Formato mmm-aa (nov-24 - Fedesarrollo)
    if '-' in fecha_str and len(fecha_str) <= 7:
        try:
            parts = fecha_str.lower().split('-')
            if len(parts) == 2:
                mes = MESES_ESP.get(parts[0]) or MESES_ENG.get(parts[0])
                if mes:
                    anio = int(parts[1])
                    anio = 2000 + anio if anio < 100 else anio
                    return pd.Timestamp(year=anio, month=mes, day=1)
        except:
            pass

    # 5. Intento automatico
    try:
        result = pd.to_datetime(fecha_str, errors='coerce')
        if pd.notna(result):
            return result
    except:
        pass

    return None


# ==============================================================================
# FUNCIONES DE LECTURA DE ARCHIVOS
# ==============================================================================

def leer_archivo(path: str) -> Optional[pd.DataFrame]:
    """Lee archivo CSV o Excel con manejo de encodings"""
    path_obj = Path(path)

    # Verificar si existe, intentar con acentos
    if not path_obj.exists():
        alternativas = [
            path.replace('politica', 'política'),
            path.replace('Indice', 'Índice'),
        ]
        for alt in alternativas:
            if Path(alt).exists():
                path = alt
                break

    if not Path(path).exists():
        return None

    extension = Path(path).suffix.lower()

    if extension == '.csv':
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                return pd.read_csv(path, encoding=encoding, thousands=',')
            except:
                continue
    elif extension in ['.xlsx', '.xls']:
        try:
            return pd.read_excel(path, engine='openpyxl')
        except:
            try:
                return pd.read_excel(path)
            except:
                pass

    return None


def convertir_a_numerico(series: pd.Series, formato_numero: str = '') -> pd.Series:
    """Convierte serie a numerico segun formato"""
    if series.dtype in ['float64', 'int64', 'float32', 'int32']:
        return series

    series = series.astype(str)
    series = series.str.replace('%', '', regex=False).str.strip()

    if 'coma' in str(formato_numero).lower():
        series = series.str.replace('.', '', regex=False)
        series = series.str.replace(',', '.', regex=False)
    else:
        series = series.str.replace(',', '', regex=False)

    return pd.to_numeric(series, errors='coerce')


# ==============================================================================
# PROCESADOR DE VARIABLES
# ==============================================================================

def procesar_variable(row: pd.Series, config: Config) -> Tuple[Optional[pd.DataFrame], dict]:
    """
    Procesa una variable del diccionario y retorna DataFrame con fecha y valor.
    Usa deteccion inteligente de formato de fecha, numeros, y estructura de archivo.
    """
    var_name = row['VARIABLE_NUEVA_ESTANDAR']
    nombre_archivo = row['NOMBRE_ARCHIVO']
    categoria = row.get('CATEGORIA_NIVEL_1', '')
    # Construir ruta dinámica usando Config (ignorando PATH_FILE_COMPLETO del diccionario)
    path = str(Config.get_raw_file_path(categoria, nombre_archivo))
    formato_numero = row.get('FORMATO_NUMERO_ORIGINAL', 'Punto decimal')
    fuente = row.get('FUENTE_SECUNDARIA', '')

    info = {
        'variable': var_name,
        'archivo': nombre_archivo,
        'success': False,
        'rows': 0,
        'error': '',
        'detection_info': {}
    }

    # Leer archivo
    df = leer_archivo(path)
    if df is None:
        info['error'] = 'Archivo no encontrado'
        return None, info

    # =========================================================================
    # CASO ESPECIAL: ARCHIVOS FEDESARROLLO (ICC e ICI)
    # =========================================================================
    fecha_inicio = pd.Timestamp(config.FECHA_INICIO)

    # INDICES DE CONFIANZA FEDESARROLLO (ICC - Consumidor, ICI - Industrial)
    if ('cci' in var_name.lower() or 'ici' in var_name.lower()) and \
       ('confianza' in nombre_archivo.lower() or 'fedesarrollo' in str(fuente).lower()):
        print(f"    [FEDESARROLLO] Archivo de Índice de Confianza detectado: {nombre_archivo}")
        resultado = process_fedesarrollo_confianza(path, var_name, fecha_inicio)
        if resultado is not None and len(resultado) > 0:
            info['success'] = True
            info['rows'] = len(resultado)
            info['detection_info']['structure'] = 'fedesarrollo_confianza'
            info['rango'] = f"{resultado['fecha'].min().date()} a {resultado['fecha'].max().date()}"
            return resultado, info

    # =========================================================================
    # CASO ESPECIAL: ARCHIVOS DANE (imports/exports con formatos especiales)
    # =========================================================================

    # BALANZA COMERCIAL DANE (archivo anex-BCOM-Mensual-*.xlsx)
    # Este archivo contiene AMBAS variables: Exportaciones e Importaciones
    if ('expusd' in var_name.lower() or 'impusd' in var_name.lower()) and \
       ('bcom' in nombre_archivo.lower() or 'balanza' in nombre_archivo.lower()):
        print(f"    [DANE BCOM] Archivo de Balanza Comercial detectado: {nombre_archivo}")
        resultado = process_dane_balanza_comercial(path, var_name, fecha_inicio)
        if resultado is not None and len(resultado) > 0:
            info['success'] = True
            info['rows'] = len(resultado)
            info['detection_info']['structure'] = 'dane_balanza_comercial'
            info['rango'] = f"{resultado['fecha'].min().date()} a {resultado['fecha'].max().date()}"
            return resultado, info

    # EXPORTACIONES DANE: formato series de tiempo con header en fila 9
    if 'expusd' in var_name.lower() and any(x in str(df.columns).lower() for x in ['unnamed']):
        print(f"    [DANE EXPORTS] Archivo DANE exportaciones detectado...")
        resultado = process_dane_exports(path, var_name, fecha_inicio)
        if resultado is not None and len(resultado) > 0:
            info['success'] = True
            info['rows'] = len(resultado)
            info['detection_info']['structure'] = 'dane_exports'
            info['rango'] = f"{resultado['fecha'].min().date()} a {resultado['fecha'].max().date()}"
            return resultado, info

    # IMPORTACIONES DANE: formato pivot con paises y años en columnas
    if 'impusd' in var_name.lower() and any(x in str(df.columns).lower() for x in ['unnamed']):
        # Verificar si tiene estructura DANE pivot (multiples paises, años en columnas)
        for col in df.columns:
            col_values = df[col].dropna().astype(str).str.lower().tolist()
            month_count = sum(1 for v in col_values if v in ['enero', 'febrero', 'marzo'])
            if month_count > 5:  # Mas de 5 repeticiones de meses = formato DANE pivot
                print(f"    [DANE] Archivo DANE importaciones detectado, usando procesador especial...")
                resultado = process_dane_imports_exports(path, var_name, fecha_inicio)
                if resultado is not None and len(resultado) > 0:
                    info['success'] = True
                    info['rows'] = len(resultado)
                    info['detection_info']['structure'] = 'dane_pivot'
                    info['rango'] = f"{resultado['fecha'].min().date()} a {resultado['fecha'].max().date()}"
                    return resultado, info
                break

    # =========================================================================
    # DETECCION INTELIGENTE DE ESTRUCTURA
    # =========================================================================
    structure = SmartFileStructureDetector.detect_structure(df)
    info['detection_info']['structure'] = structure['structure_type']

    # Si es tabla pivot, convertir a formato estandar
    if structure['structure_type'] == 'pivot':
        print(f"    [PIVOT] Tabla pivot detectada, convirtiendo...")
        resultado = unpivot_table(df, structure['pivot_info'], var_name)
        if resultado is None or len(resultado) == 0:
            info['error'] = 'Error convirtiendo tabla pivot'
            return None, info

        # Filtrar por rango de fechas
        fecha_inicio = pd.Timestamp(config.FECHA_INICIO)
        fecha_fin = pd.Timestamp(config.FECHA_FIN)
        resultado = resultado[(resultado['fecha'] >= fecha_inicio) & (resultado['fecha'] <= fecha_fin)]

        info['success'] = True
        info['rows'] = len(resultado)
        info['rango'] = f"{resultado['fecha'].min().date()} a {resultado['fecha'].max().date()}" if len(resultado) > 0 else "N/A"
        return resultado, info

    # =========================================================================
    # PROCESAMIENTO ESTANDAR
    # =========================================================================
    col_fecha = structure.get('date_column')
    col_valor = structure.get('value_column')

    # Fallback: buscar columnas manualmente
    if col_fecha is None:
        for col in df.columns:
            if any(x in str(col).lower() for x in ['fecha', 'date', 'periodo', 'observation', 'release']):
                col_fecha = col
                break
        if col_fecha is None:
            col_fecha = df.columns[0]

    if col_valor is None:
        for col in df.columns:
            if col != col_fecha:
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    if df[col].notna().sum() > 0:
                        col_valor = col
                        break

        if col_valor is None:
            for col in df.columns:
                if col != col_fecha and any(x in str(col).lower() for x in ['price', 'valor', 'actual', 'value', 'close']):
                    col_valor = col
                    break

        if col_valor is None and len(df.columns) > 1:
            col_valor = [c for c in df.columns if c != col_fecha][0]

    if col_valor is None:
        info['error'] = 'No se encontro columna de valor'
        return None, info

    # =========================================================================
    # DETECCION INTELIGENTE DE FORMATO DE FECHA
    # =========================================================================
    date_samples = df[col_fecha].dropna().astype(str).tolist()[:50]
    date_format_info = SmartDateDetector.detect_format(date_samples, fuente)
    info['detection_info']['date_format'] = date_format_info.get('format', 'unknown')
    info['detection_info']['date_confidence'] = date_format_info.get('confidence', 0)

    # Determinar si usar periodo de referencia (para datos como CPI con fechas tipo "Oct 24, 2025 (Sep)")
    use_ref_period = date_format_info.get('format') == 'Mmm DD, YYYY (ref)'

    # =========================================================================
    # DETECCION INTELIGENTE DE FORMATO NUMERICO
    # =========================================================================
    value_samples = df[col_valor].dropna().tolist()[:100]
    numeric_format_info = SmartNumericDetector.detect_format(value_samples, fuente)
    info['detection_info']['numeric_format'] = f"decimal={numeric_format_info.get('decimal_sep', '.')}"
    info['detection_info']['has_percentage'] = numeric_format_info.get('has_percentage', False)
    info['detection_info']['source_override'] = numeric_format_info.get('source_override', False)

    # Crear DataFrame limpio
    df_clean = df[[col_fecha, col_valor]].copy()
    df_clean.columns = ['fecha_raw', var_name]

    # Parsear fechas (pasando la fuente y si usar periodo de referencia)
    df_clean['fecha'] = df_clean['fecha_raw'].apply(
        lambda x: parsear_fecha(x, fuente, use_reference_period=use_ref_period)
    )

    # Filtrar filas validas
    df_clean = df_clean[df_clean['fecha'].notna()].copy()

    if len(df_clean) == 0:
        info['error'] = 'No se pudieron parsear fechas'
        return None, info

    # Convertir valores usando formato detectado
    if numeric_format_info.get('decimal_sep') == ',':
        # Formato europeo: coma como decimal
        df_clean[var_name] = convertir_a_numerico(df_clean[var_name], 'coma decimal')
    else:
        df_clean[var_name] = convertir_a_numerico(df_clean[var_name], formato_numero)

    # Filtrar por rango de fechas
    fecha_inicio = pd.Timestamp(config.FECHA_INICIO)
    fecha_fin = pd.Timestamp(config.FECHA_FIN)
    df_clean = df_clean[(df_clean['fecha'] >= fecha_inicio) & (df_clean['fecha'] <= fecha_fin)]

    # Ordenar y eliminar duplicados
    df_clean = df_clean.sort_values('fecha')
    df_clean = df_clean.drop_duplicates(subset=['fecha'], keep='last')

    # Resultado final
    resultado = df_clean[['fecha', var_name]].copy()

    info['success'] = True
    info['rows'] = len(resultado)
    info['rango'] = f"{resultado['fecha'].min().date()} a {resultado['fecha'].max().date()}" if len(resultado) > 0 else "N/A"

    return resultado, info


# ==============================================================================
# CONSOLIDADOR POR FRECUENCIA
# ==============================================================================

def consolidar_frecuencia(diccionario: pd.DataFrame, frecuencia: str, config: Config) -> Tuple[Optional[pd.DataFrame], List[dict]]:
    """
    Consolida todas las variables de una frecuencia usando OUTER JOIN por fecha.
    """
    print(f"\n{'='*70}")
    print(f"FRECUENCIA: {frecuencia}")
    print(f"{'='*70}")

    vars_freq = diccionario[diccionario['FRECUENCIA'] == frecuencia].copy()

    if len(vars_freq) == 0:
        print(f"  No hay variables con frecuencia {frecuencia}")
        return None, []

    print(f"Variables a procesar: {len(vars_freq)}")

    datasets = []
    log = []

    for idx, row in vars_freq.iterrows():
        df_var, info = procesar_variable(row, config)
        log.append(info)

        if info['success']:
            print(f"  [OK] {info['variable']}: {info['rows']} filas")
            datasets.append(df_var)
        else:
            print(f"  [ERROR] {info['variable']}: {info['error']}")

    if len(datasets) == 0:
        return None, log

    # OUTER JOIN
    print(f"\nRealizando OUTER JOIN de {len(datasets)} variables...")

    resultado = datasets[0]
    for i, df in enumerate(datasets[1:], 2):
        resultado = pd.merge(resultado, df, on='fecha', how='outer')

    resultado = resultado.sort_values('fecha').reset_index(drop=True)

    # Filtrar dias laborables para frecuencia diaria
    if frecuencia == 'D' and config.SOLO_DIAS_LABORABLES:
        filas_antes = len(resultado)
        resultado = resultado[resultado['fecha'].dt.dayofweek <= 4].reset_index(drop=True)
        print(f"  Filtro dias laborables: {filas_antes} -> {len(resultado)} filas")

    print(f"\nResultado {frecuencia}:")
    print(f"  Filas: {len(resultado)}")
    print(f"  Variables: {len(resultado.columns) - 1}")
    if len(resultado) > 0:
        print(f"  Rango: {resultado['fecha'].min().date()} a {resultado['fecha'].max().date()}")

    return resultado, log


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Funcion principal"""
    print("="*70)
    print("MACRO DATA FUSION v4.1 - Deteccion Inteligente Mejorada")
    print("="*70)
    print(f"Ejecucion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = Config()

    # Mostrar configuracion
    print(f"\nConfiguracion:")
    print(f"  Base dir: {config.BASE_DIR}")
    print(f"  Config dir: {config.CONFIG_DIR}")
    print(f"  Output dir: {config.OUTPUT_DIR}")
    print(f"  Logs dir: {config.LOGS_DIR}")

    # Verificar que existen las carpetas
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Verificar diccionario
    diccionario_path = config.get_diccionario_path()
    if not diccionario_path.exists():
        print(f"\nERROR: No se encuentra {diccionario_path}")
        sys.exit(1)

    # Leer diccionario
    print(f"\n[1/4] Leyendo diccionario...")
    diccionario = pd.read_csv(diccionario_path, sep=config.DELIMITADOR_DICCIONARIO, encoding='utf-8')
    print(f"  Total variables: {len(diccionario)}")

    freq_counts = diccionario['FRECUENCIA'].value_counts()
    for freq, count in freq_counts.items():
        print(f"    {freq}: {count} variables")

    # Procesar cada frecuencia
    print(f"\n[2/4] Procesando datasets...")

    all_logs = []

    # DAILY
    df_daily, log_d = consolidar_frecuencia(diccionario, 'D', config)
    all_logs.extend(log_d)
    if df_daily is not None:
        output_path = config.get_output_path(config.OUTPUT_DAILY)
        df_daily.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  Guardado: {output_path}")

    # MONTHLY
    df_monthly, log_m = consolidar_frecuencia(diccionario, 'M', config)
    all_logs.extend(log_m)
    if df_monthly is not None:
        output_path = config.get_output_path(config.OUTPUT_MONTHLY)
        df_monthly.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  Guardado: {output_path}")

    # QUARTERLY
    df_quarterly, log_q = consolidar_frecuencia(diccionario, 'Q', config)
    all_logs.extend(log_q)
    if df_quarterly is not None:
        output_path = config.get_output_path(config.OUTPUT_QUARTERLY)
        df_quarterly.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  Guardado: {output_path}")

    # Reporte de procesamiento
    print(f"\n[3/4] Generando reporte...")
    df_log = pd.DataFrame(all_logs)
    log_path = config.get_log_path(config.OUTPUT_REPORTE)
    df_log.to_csv(log_path, index=False, encoding='utf-8')
    print(f"  Guardado: {log_path}")

    # Resumen final
    print(f"\n[4/4] RESUMEN FINAL")
    print("="*70)

    exitosos = sum(1 for l in all_logs if l['success'])
    fallidos = sum(1 for l in all_logs if not l['success'])

    print(f"\n  Procesamiento:")
    print(f"    Exitosos: {exitosos}")
    print(f"    Fallidos: {fallidos}")

    if fallidos > 0:
        print(f"\n  Variables con error:")
        for l in all_logs:
            if not l['success']:
                print(f"    - {l['variable']}: {l['error']}")

    print(f"\n  Archivos generados:")
    if df_daily is not None:
        print(f"    {config.OUTPUT_DAILY}: {len(df_daily)} filas x {len(df_daily.columns)-1} vars")
    if df_monthly is not None:
        print(f"    {config.OUTPUT_MONTHLY}: {len(df_monthly)} filas x {len(df_monthly.columns)-1} vars")
    if df_quarterly is not None:
        print(f"    {config.OUTPUT_QUARTERLY}: {len(df_quarterly)} filas x {len(df_quarterly.columns)-1} vars")

    print("\n" + "="*70)
    print("COMPLETADO!")
    print("="*70)

    return df_daily, df_monthly, df_quarterly


if __name__ == "__main__":
    df_d, df_m, df_q = main()
