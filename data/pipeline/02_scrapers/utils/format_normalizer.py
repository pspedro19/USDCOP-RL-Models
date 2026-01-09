# -*- coding: utf-8 -*-
"""
Normalizador Inteligente de Formatos
=====================================

Sistema para detectar y normalizar automáticamente:
1. Formatos numéricos (europeo vs americano)
2. Formatos de fecha
3. Separadores de miles y decimales

Autor: Sistema Automatizado
Fecha: 2025-11-25
"""

import re
import pandas as pd
from typing import Union, Optional, Tuple
from datetime import datetime


class FormatNormalizer:
    """
    Normalizador inteligente de formatos numéricos y de fecha.

    Detecta automáticamente el formato y convierte a formato estándar:
    - Números: float con punto decimal
    - Fechas: datetime o string ISO (YYYY-MM-DD)
    """

    # Patrones de detección de formato numérico
    PATTERNS = {
        # Formato americano: 1,234.56 o 1234.56
        'american': re.compile(r'^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$'),
        # Formato europeo: 1.234,56 o 1234,56
        'european': re.compile(r'^-?\d{1,3}(\.\d{3})*(,\d+)?$|^-?\d+(,\d+)?$'),
        # Formato sin separador de miles: 1234.56 o 1234,56
        'simple_american': re.compile(r'^-?\d+\.\d+$'),
        'simple_european': re.compile(r'^-?\d+,\d+$'),
    }

    # Rangos esperados por tipo de variable (para validación)
    EXPECTED_RANGES = {
        # Tipos de cambio
        'USDMXN': (15, 30),
        'USDCLP': (600, 1200),
        'USDCOP': (3000, 6000),
        'DXY': (80, 120),

        # Tasas de interés (%)
        'YIELD': (0, 20),
        'RATE': (0, 20),
        'IBR': (0, 20),
        'TPM': (0, 20),
        'PRIME': (0, 15),
        'FEDFUNDS': (0, 10),

        # Commodities
        'WTI': (20, 200),
        'BRENT': (20, 200),
        # 'COAL': ELIMINADO - variable ya no se usa
        'GOLD': (3000, 5000),   # Oro - rango actualizado Nov 2025
        'COFFEE': (50, 500),

        # Índices
        'COLCAP': (500, 3000),
        'EMBI': (50, 1000),

        # Inflación (%)
        'CPI': (-5, 20),
        'PCE': (50, 200),  # Es índice, no %

        # Otros
        'DEFAULT': (-1e12, 1e12),
    }

    @classmethod
    def detect_number_format(cls, value: str) -> str:
        """
        Detecta el formato de un número string.

        Args:
            value: String con el número a analizar

        Returns:
            'american', 'european', o 'unknown'
        """
        value = value.strip()

        # Quitar signos de moneda y espacios
        value = re.sub(r'[$€£¥\s]', '', value)

        # Contar puntos y comas
        dots = value.count('.')
        commas = value.count(',')

        # Caso 1: Solo tiene punto - probablemente americano
        if dots == 1 and commas == 0:
            # Verificar que el punto esté cerca del final (decimales)
            dot_pos = value.rfind('.')
            decimals = len(value) - dot_pos - 1
            if decimals <= 6:  # Hasta 6 decimales es razonable
                return 'american'

        # Caso 2: Solo tiene coma - probablemente europeo
        if commas == 1 and dots == 0:
            comma_pos = value.rfind(',')
            decimals = len(value) - comma_pos - 1
            if decimals <= 6:
                return 'european'

        # Caso 3: Tiene ambos - analizar posición
        if dots > 0 and commas > 0:
            last_dot = value.rfind('.')
            last_comma = value.rfind(',')

            # El último separador es el decimal
            if last_dot > last_comma:
                return 'american'  # 1,234.56
            else:
                return 'european'  # 1.234,56

        # Caso 4: Múltiples puntos sin comas - europeo (separador miles)
        if dots > 1 and commas == 0:
            return 'european'

        # Caso 5: Múltiples comas sin puntos - americano (separador miles)
        if commas > 1 and dots == 0:
            return 'american'

        # Caso 6: Sin separadores - formato simple
        if dots == 0 and commas == 0:
            return 'integer'

        return 'unknown'

    @classmethod
    def parse_number(cls, value: Union[str, float, int],
                     expected_format: str = 'auto',
                     variable_hint: str = None) -> Optional[float]:
        """
        Convierte un valor a float, detectando automáticamente el formato.

        Args:
            value: Valor a convertir (string, float o int)
            expected_format: 'american', 'european', o 'auto' para detección automática
            variable_hint: Nombre de variable para validar rango esperado

        Returns:
            float o None si no se puede parsear
        """
        # Si ya es numérico, retornar directamente
        if isinstance(value, (int, float)):
            return float(value)

        if pd.isna(value) or value is None:
            return None

        # Convertir a string y limpiar
        value_str = str(value).strip()

        # Quitar caracteres no numéricos excepto separadores y signo
        value_clean = re.sub(r'[^\d.,\-]', '', value_str)

        if not value_clean or value_clean in ['.', ',', '-']:
            return None

        # Detectar formato si es auto
        if expected_format == 'auto':
            detected_format = cls.detect_number_format(value_clean)
        else:
            detected_format = expected_format

        try:
            # Convertir según formato detectado
            if detected_format == 'european':
                # Formato europeo: 1.234,56 -> 1234.56
                # Quitar puntos (separador miles) y cambiar coma por punto
                value_normalized = value_clean.replace('.', '').replace(',', '.')
            elif detected_format == 'american':
                # Formato americano: 1,234.56 -> 1234.56
                # Solo quitar comas (separador miles)
                value_normalized = value_clean.replace(',', '')
            elif detected_format == 'integer':
                value_normalized = value_clean
            else:
                # Intentar americano por defecto
                value_normalized = value_clean.replace(',', '')

            result = float(value_normalized)

            # Validar rango si hay hint de variable
            if variable_hint:
                expected_range = cls._get_expected_range(variable_hint)
                if expected_range:
                    min_val, max_val = expected_range
                    if not (min_val <= result <= max_val):
                        # Intentar formato alternativo
                        if detected_format == 'american':
                            alt_value = value_clean.replace('.', '').replace(',', '.')
                        else:
                            alt_value = value_clean.replace(',', '')

                        try:
                            alt_result = float(alt_value)
                            if min_val <= alt_result <= max_val:
                                return alt_result
                        except:
                            pass

            return result

        except (ValueError, TypeError):
            return None

    @classmethod
    def _get_expected_range(cls, variable_name: str) -> Optional[Tuple[float, float]]:
        """Obtiene el rango esperado para una variable."""
        variable_upper = variable_name.upper()

        for key, range_val in cls.EXPECTED_RANGES.items():
            if key in variable_upper:
                return range_val

        return cls.EXPECTED_RANGES.get('DEFAULT')

    @classmethod
    def parse_date(cls, value: Union[str, datetime, pd.Timestamp],
                   formats: list = None) -> Optional[pd.Timestamp]:
        """
        Convierte un valor a fecha, probando múltiples formatos.

        Args:
            value: Valor a convertir
            formats: Lista de formatos a probar (None para usar defaults)

        Returns:
            pd.Timestamp o None
        """
        if pd.isna(value) or value is None:
            return None

        # Si ya es datetime/Timestamp
        if isinstance(value, (datetime, pd.Timestamp)):
            return pd.Timestamp(value)

        value_str = str(value).strip()

        # Formatos comunes a probar
        if formats is None:
            formats = [
                '%Y-%m-%d',           # ISO: 2025-11-24
                '%Y/%m/%d',           # 2025/11/24
                '%d/%m/%Y',           # Europeo: 24/11/2025
                '%m/%d/%Y',           # Americano: 11/24/2025
                '%d-%m-%Y',           # 24-11-2025
                '%b %d, %Y',          # Nov 24, 2025
                '%B %d, %Y',          # November 24, 2025
                '%d %b %Y',           # 24 Nov 2025
                '%d %B %Y',           # 24 November 2025
                '%Y-%m-%dT%H:%M:%S',  # ISO con tiempo
                '%Y%m%d',             # Compacto: 20251124
            ]

        # Intentar cada formato
        for fmt in formats:
            try:
                return pd.to_datetime(value_str, format=fmt)
            except:
                continue

        # Último intento: dejar que pandas intente
        try:
            return pd.to_datetime(value_str)
        except:
            return None

    @classmethod
    def normalize_dataframe(cls, df: pd.DataFrame,
                           date_col: str = 'fecha',
                           value_col: str = 'valor',
                           variable_name: str = None) -> pd.DataFrame:
        """
        Normaliza un DataFrame con columnas de fecha y valor.

        Args:
            df: DataFrame a normalizar
            date_col: Nombre de columna de fecha
            value_col: Nombre de columna de valor
            variable_name: Nombre de variable para validación de rango

        Returns:
            DataFrame normalizado
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        # Normalizar fechas
        if date_col in df.columns:
            df[date_col] = df[date_col].apply(cls.parse_date)

        # Normalizar valores
        if value_col in df.columns:
            df[value_col] = df[value_col].apply(
                lambda x: cls.parse_number(x, variable_hint=variable_name)
            )

        # Eliminar filas con valores nulos
        df = df.dropna(subset=[date_col, value_col])

        return df


def smart_parse_number(value: str, variable_hint: str = None) -> Optional[float]:
    """Función de conveniencia para parsear números."""
    return FormatNormalizer.parse_number(value, variable_hint=variable_hint)


def smart_parse_date(value: str) -> Optional[pd.Timestamp]:
    """Función de conveniencia para parsear fechas."""
    return FormatNormalizer.parse_date(value)


def normalize_dataframe(df: pd.DataFrame, variable_name: str = None) -> pd.DataFrame:
    """Función de conveniencia para normalizar DataFrames."""
    return FormatNormalizer.normalize_dataframe(df, variable_name=variable_name)


# Test del módulo
if __name__ == '__main__':
    print("="*80)
    print("TEST DEL NORMALIZADOR DE FORMATOS")
    print("="*80)

    # Test números
    test_numbers = [
        ("1,234.56", "american"),
        ("1.234,56", "european"),
        ("18.5131", "USDMXN"),
        ("18,5131", "USDMXN"),
        ("1234", "integer"),
        ("3,990.50", "USDCOP"),
        ("3.990,50", "USDCOP"),
    ]

    print("\nTest de números:")
    for value, hint in test_numbers:
        result = smart_parse_number(value, hint)
        format_detected = FormatNormalizer.detect_number_format(value)
        print(f"  '{value}' ({hint}) -> {result} (formato: {format_detected})")

    # Test fechas
    test_dates = [
        "2025-11-24",
        "24/11/2025",
        "Nov 24, 2025",
        "2025/11/24",
    ]

    print("\nTest de fechas:")
    for date_str in test_dates:
        result = smart_parse_date(date_str)
        print(f"  '{date_str}' -> {result}")
