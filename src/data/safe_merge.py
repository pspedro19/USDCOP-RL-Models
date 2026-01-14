"""
Safe Data Operations - Anti-Leakage Implementation
Contrato: CTR-006
CLAUDE-T6, CLAUDE-T7 | Plan Items: P0-10, P0-11

IMPORTANTE: Este modulo implementa operaciones de datos SIN data leakage.
- ffill siempre tiene limite
- merge_asof nunca tiene tolerance
"""

import pandas as pd
import numpy as np
from typing import Optional, List


# P0-10: Limite maximo para ffill (12 horas = 144 barras de 5min)
FFILL_LIMIT_5MIN = 144  # 12 hours
FFILL_LIMIT_1H = 12     # 12 hours
FFILL_LIMIT_DAILY = 5   # 5 business days


def safe_ffill(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    limit: int = FFILL_LIMIT_5MIN,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Forward fill con limite obligatorio.

    IMPORTANTE: Nunca usar df.ffill() sin limite - puede propagar
    datos obsoletos indefinidamente.

    Args:
        df: DataFrame a procesar
        columns: Columnas especificas a llenar (None = todas)
        limit: Numero maximo de filas a propagar (default: 144 = 12 horas)
        inplace: Si True, modifica df in-place

    Returns:
        DataFrame con valores llenados

    Example:
        >>> df = safe_ffill(df, columns=['dxy', 'vix'], limit=144)

    Raises:
        ValueError: Si limit <= 0
    """
    if limit <= 0:
        raise ValueError(f"limit debe ser > 0, got {limit}")

    if not inplace:
        df = df.copy()

    if columns is None:
        df = df.ffill(limit=limit)
    else:
        for col in columns:
            if col in df.columns:
                df[col] = df[col].ffill(limit=limit)

    return df


def safe_merge_macro(
    df_ohlcv: pd.DataFrame,
    df_macro: pd.DataFrame,
    datetime_col: str = 'datetime',
    track_source: bool = True
) -> pd.DataFrame:
    """
    Merge macro data SIN data leakage.

    IMPORTANTE: No usar tolerance en merge_asof - permite data leakage.

    Args:
        df_ohlcv: OHLCV data con datetime column
        df_macro: Macro data con datetime column
        datetime_col: Nombre de la columna datetime
        track_source: Si True, agrega columna macro_source_date para auditoria

    Returns:
        DataFrame merged sin data leakage

    Raises:
        ValueError: Si se detecta data leakage

    Example:
        >>> df = safe_merge_macro(df_ohlcv, df_macro)
        >>> assert 'macro_source_date' in df.columns  # Para auditoria
    """
    # Copiar para no modificar originales
    df_ohlcv = df_ohlcv.copy()
    df_macro = df_macro.copy()

    # Asegurar datetime column existe
    if datetime_col not in df_ohlcv.columns:
        raise ValueError(f"'{datetime_col}' no encontrado en df_ohlcv")
    if datetime_col not in df_macro.columns:
        raise ValueError(f"'{datetime_col}' no encontrado en df_macro")

    # Para datos macro diarios, usar inicio del dia
    df_macro_daily = df_macro.copy()
    df_macro_daily[datetime_col] = pd.to_datetime(
        df_macro_daily[datetime_col]
    )

    # Normalizar a inicio del dia para macro data
    if df_macro_daily[datetime_col].dt.hour.nunique() > 1:
        # Tiene diferentes horas - es intraday, no modificar
        pass
    else:
        # Es daily - normalizar a inicio del dia
        df_macro_daily[datetime_col] = df_macro_daily[datetime_col].dt.normalize()

    if track_source:
        df_macro_daily['macro_source_date'] = df_macro_daily[datetime_col].copy()

    # Merge SIN tolerance - CRITICO para evitar data leakage
    df = pd.merge_asof(
        df_ohlcv.sort_values(datetime_col),
        df_macro_daily.sort_values(datetime_col),
        on=datetime_col,
        direction='backward'
        # NO tolerance - strict temporal ordering
    )

    # Validar no hay future data
    if track_source:
        validate_no_future_data(df, datetime_col, 'macro_source_date')

    return df


def validate_no_future_data(
    df: pd.DataFrame,
    target_col: str = 'datetime',
    source_col: str = 'macro_source_date'
) -> bool:
    """
    Valida que no hay data del futuro en el merge.

    Args:
        df: DataFrame merged
        target_col: Columna con timestamp objetivo
        source_col: Columna con timestamp de origen de datos

    Returns:
        True si no hay data leakage

    Raises:
        ValueError: Si se detecta data leakage
    """
    if source_col not in df.columns:
        return True  # No tracking column - skip validation

    # Convertir a datetime si es necesario
    target = pd.to_datetime(df[target_col])
    source = pd.to_datetime(df[source_col])

    # Detectar rows donde source > target (data del futuro)
    leakage = df[source > target]

    if len(leakage) > 0:
        first_leak = leakage.iloc[0]
        raise ValueError(
            f"DATA LEAKAGE DETECTADO: {len(leakage)} rows con datos del futuro.\n"
            f"Ejemplo - Target: {first_leak[target_col]}, Source: {first_leak[source_col]}"
        )

    return True


def check_ffill_in_source(filepath: str) -> List[dict]:
    """
    Analiza un archivo Python para encontrar ffill() sin limite.

    Args:
        filepath: Ruta al archivo Python

    Returns:
        Lista de diccionarios con info sobre ffill sin limite

    Example:
        >>> issues = check_ffill_in_source("scripts/build_data.py")
        >>> for issue in issues:
        ...     print(f"Line {issue['line']}: ffill without limit")
    """
    import ast

    with open(filepath, encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source)
    issues = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Buscar .ffill()
            if hasattr(func, 'attr') and func.attr == 'ffill':
                # Verificar si tiene limit keyword
                has_limit = any(
                    kw.arg == 'limit' for kw in node.keywords
                )
                if not has_limit:
                    issues.append({
                        'line': node.lineno,
                        'col': node.col_offset,
                        'type': 'ffill_no_limit'
                    })

    return issues


def check_merge_asof_tolerance(filepath: str) -> List[dict]:
    """
    Analiza un archivo Python para encontrar merge_asof con tolerance.

    Args:
        filepath: Ruta al archivo Python

    Returns:
        Lista de diccionarios con info sobre merge_asof con tolerance

    Example:
        >>> issues = check_merge_asof_tolerance("scripts/build_data.py")
        >>> for issue in issues:
        ...     print(f"Line {issue['line']}: merge_asof has tolerance - DATA LEAKAGE RISK")
    """
    import ast

    with open(filepath, encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source)
    issues = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Buscar merge_asof
            if hasattr(func, 'attr') and func.attr == 'merge_asof':
                # Verificar si tiene tolerance keyword
                has_tolerance = any(
                    kw.arg == 'tolerance' for kw in node.keywords
                )
                if has_tolerance:
                    issues.append({
                        'line': node.lineno,
                        'col': node.col_offset,
                        'type': 'merge_asof_tolerance'
                    })

    return issues
