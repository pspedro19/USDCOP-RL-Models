"""
Scraper para Balanza Comercial de Colombia desde DANE
Obtiene datos mensuales de exportaciones e importaciones
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import io
import re

# URL base y patron dinamico del archivo Excel de Balanza Comercial Mensual
# Formato: anex-BCOM-Mensual-{mes}{año}.xlsx (ej: sep2025, oct2025)
DANE_BASE_URL = 'https://www.dane.gov.co/files/operaciones/BCOM/'
BALANZA_PATTERN = 'anex-BCOM-Mensual-{mes}{anio}.xlsx'

# Meses en español para el patron
MESES_ES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
}


def buscar_archivo_mas_reciente():
    """
    Busca dinamicamente el archivo Excel más reciente de Balanza Comercial

    Intenta desde el mes actual hacia atrás hasta encontrar un archivo válido

    Returns:
        URL del archivo encontrado o None
    """
    from datetime import datetime

    now = datetime.now()
    anio_actual = now.year
    mes_actual = now.month

    # Intentar desde el mes actual hasta 6 meses atrás
    for offset in range(6):
        # Calcular mes y año
        mes_idx = mes_actual - 1 - offset  # -1 porque MESES_ES es 0-indexed
        anio = anio_actual

        while mes_idx < 0:
            mes_idx += 12
            anio -= 1

        mes_str = MESES_ES[mes_idx]
        archivo = BALANZA_PATTERN.format(mes=mes_str, anio=anio)
        url = f"{DANE_BASE_URL}{archivo}"

        try:
            # Hacer HEAD request para verificar si existe
            response = requests.head(url, headers=HEADERS, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                print(f"  [OK] Archivo encontrado: {archivo}")
                return url
            else:
                print(f"  [--] {archivo} no disponible (status: {response.status_code})")
        except Exception as e:
            print(f"  [--] Error verificando {archivo}: {str(e)[:50]}")

    return None


def descargar_archivo_excel():
    """
    Descarga el archivo Excel de Balanza Comercial del DANE
    Busca dinámicamente el archivo más reciente

    Returns:
        BytesIO con el contenido del archivo o None si falla
    """
    print("\n[DANE] Descargando archivo Excel de Balanza Comercial...")
    print(f"  Buscando archivo más reciente...")

    # Buscar archivo dinamicamente
    url = buscar_archivo_mas_reciente()

    if url is None:
        print(f"  [ERROR] No se encontró ningún archivo disponible")
        return None

    print(f"  URL: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)

        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            print(f"  [OK] Archivo descargado ({len(response.content)} bytes)")
            return io.BytesIO(response.content)
        else:
            print(f"  [ERROR] No se pudo descargar el archivo")
            return None

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {str(e)[:80]}")
        return None


def encontrar_hoja_datos(excel_file):
    """
    Encuentra la hoja correcta con datos de serie mensual

    Args:
        excel_file: BytesIO con el archivo Excel

    Returns:
        Nombre de la hoja con los datos o None
    """
    print("\n[DANE] Analizando hojas del archivo Excel...")

    try:
        # Leer nombres de todas las hojas
        excel = pd.ExcelFile(excel_file)
        hojas = excel.sheet_names

        print(f"  Hojas encontradas: {len(hojas)}")
        for i, hoja in enumerate(hojas, 1):
            print(f"    {i}. {hoja}")

        # Buscar hoja que contenga "mensual" o "serie" en el nombre
        for hoja in hojas:
            hoja_lower = hoja.lower()
            if 'mensual' in hoja_lower or 'serie' in hoja_lower or 'total' in hoja_lower:
                print(f"\n  [OK] Hoja seleccionada: '{hoja}'")
                return hoja

        # Si no encuentra por nombre, usar la primera hoja
        print(f"\n  [WARN] No se encontró hoja con nombre esperado, usando primera hoja: '{hojas[0]}'")
        return hojas[0]

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {str(e)[:80]}")
        return None


def parsear_fecha_columnas(row, col_indices):
    """
    Parsea la fecha desde columnas separadas (Año, Mes) o columna única

    Args:
        row: Fila del DataFrame
        col_indices: Dict con indices de columnas {'año': idx, 'mes': idx} o {'fecha': idx}

    Returns:
        Fecha en formato YYYY-MM-DD o None
    """
    try:
        if 'fecha' in col_indices:
            # Columna única de fecha
            fecha_val = row.iloc[col_indices['fecha']]

            # Si es Timestamp de pandas
            if isinstance(fecha_val, pd.Timestamp):
                return fecha_val.strftime('%Y-%m-%d')

            # Si es string, intentar parsear
            if isinstance(fecha_val, str):
                # Intentar varios formatos
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y']:
                    try:
                        return datetime.strptime(fecha_val, fmt).strftime('%Y-%m-%d')
                    except:
                        continue

        elif 'año' in col_indices and 'mes' in col_indices:
            # Columnas separadas de año y mes
            año = row.iloc[col_indices['año']]
            mes = row.iloc[col_indices['mes']]

            # Convertir mes a número si es texto
            if isinstance(mes, str):
                meses = {
                    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
                    'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
                    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
                    'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
                }
                mes_num = meses.get(mes.lower().strip())
                if mes_num:
                    mes = mes_num

            # Convertir a enteros
            año = int(año)
            mes = int(mes)

            # Crear fecha (primer día del mes)
            return f"{año:04d}-{mes:02d}-01"

        return None

    except Exception as e:
        return None


def limpiar_valor_numerico(valor):
    """
    Limpia y convierte un valor a numérico

    Args:
        valor: Valor a limpiar (puede tener comas, puntos, etc.)

    Returns:
        Float con el valor limpio o None
    """
    try:
        if pd.isna(valor):
            return None

        # Si ya es numérico
        if isinstance(valor, (int, float)):
            return float(valor)

        # Si es string, limpiar
        if isinstance(valor, str):
            # Remover espacios
            valor = valor.strip()

            # Remover separadores de miles
            valor = valor.replace(',', '')
            valor = valor.replace(' ', '')

            # Convertir
            return float(valor)

        return None

    except:
        return None


def extraer_datos_balanza(excel_file, hoja_nombre, n=15):
    """
    Extrae los últimos N registros de exportaciones e importaciones

    Args:
        excel_file: BytesIO con el archivo Excel
        hoja_nombre: Nombre de la hoja a leer
        n: Número de registros a extraer (default: 15)

    Returns:
        DataFrame con columnas: fecha, exportaciones_usd_millones, importaciones_usd_millones
    """
    print(f"\n[DANE] Extrayendo datos de la hoja '{hoja_nombre}'...")

    try:
        # Leer la hoja completa (sin header para explorar)
        df_raw = pd.read_excel(excel_file, sheet_name=hoja_nombre, header=None)

        print(f"  Dimensiones: {df_raw.shape[0]} filas x {df_raw.shape[1]} columnas")

        # Buscar la fila de encabezados
        header_row = None
        for idx in range(min(20, len(df_raw))):  # Buscar en las primeras 20 filas
            row_text = ' '.join([str(cell).lower() for cell in df_raw.iloc[idx] if pd.notna(cell)])

            # Buscar keywords que indiquen encabezados
            if ('mes' in row_text or 'fecha' in row_text) and \
               ('export' in row_text or 'import' in row_text):
                header_row = idx
                print(f"  [OK] Fila de encabezados encontrada en índice: {header_row}")
                break

        if header_row is None:
            print(f"  [WARN] No se encontró fila de encabezados, asumiendo fila 0")
            header_row = 0

        # Leer con el encabezado correcto
        excel_file.seek(0)
        df = pd.read_excel(excel_file, sheet_name=hoja_nombre, header=header_row)

        print(f"\n  Columnas encontradas:")
        for i, col in enumerate(df.columns):
            print(f"    {i}: {col}")

        # Identificar columnas por nombre
        col_mes = None
        col_exportaciones = None
        col_importaciones = None

        for i, col in enumerate(df.columns):
            col_lower = str(col).lower().strip()

            # Mes/Fecha
            if col_lower == 'mes' or 'fecha' in col_lower:
                col_mes = col
                print(f"  [OK] Columna de fecha: '{col}'")

            # Exportaciones
            if 'export' in col_lower:
                col_exportaciones = col
                print(f"  [OK] Columna de exportaciones: '{col}'")

            # Importaciones
            if 'import' in col_lower:
                col_importaciones = col
                print(f"  [OK] Columna de importaciones: '{col}'")

        if col_mes is None or col_exportaciones is None or col_importaciones is None:
            print(f"  [ERROR] No se encontraron todas las columnas necesarias")
            print(f"    Mes: {col_mes}")
            print(f"    Exportaciones: {col_exportaciones}")
            print(f"    Importaciones: {col_importaciones}")
            return None

        # Filtrar filas con datos válidos
        df_clean = df[[col_mes, col_exportaciones, col_importaciones]].copy()

        # Remover filas con valores NaN en cualquier columna
        df_clean = df_clean.dropna()

        # Convertir fechas a string formato YYYY-MM-DD
        def convertir_fecha(fecha_val):
            try:
                # Si es Timestamp de pandas
                if isinstance(fecha_val, pd.Timestamp):
                    return fecha_val.strftime('%Y-%m-%d')

                # Si es datetime
                if isinstance(fecha_val, datetime):
                    return fecha_val.strftime('%Y-%m-%d')

                # Si es string, intentar parsear
                if isinstance(fecha_val, str):
                    # Intentar varios formatos
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y']:
                        try:
                            return datetime.strptime(fecha_val, fmt).strftime('%Y-%m-%d')
                        except:
                            continue

                # Si todo falla, intentar convertir a Timestamp
                ts = pd.to_datetime(fecha_val)
                return ts.strftime('%Y-%m-%d')

            except:
                return None

        df_clean['fecha'] = df_clean[col_mes].apply(convertir_fecha)

        # Remover filas donde no se pudo convertir la fecha
        df_clean = df_clean[df_clean['fecha'].notna()]

        # Renombrar columnas
        df_result = pd.DataFrame({
            'fecha': df_clean['fecha'],
            'exportaciones_usd_millones': df_clean[col_exportaciones],
            'importaciones_usd_millones': df_clean[col_importaciones]
        })

        # Ordenar por fecha
        df_result = df_result.sort_values('fecha', ascending=True)

        # Tomar los últimos N
        df_result = df_result.tail(n).reset_index(drop=True)

        print(f"\n  [OK] {len(df_result)} registros extraídos (últimos {n})")

        # Mostrar info de unidades
        valor_promedio_exp = df_result['exportaciones_usd_millones'].mean()
        valor_promedio_imp = df_result['importaciones_usd_millones'].mean()
        print(f"  Valores promedio:")
        print(f"    Exportaciones: {valor_promedio_exp:.2f} millones USD")
        print(f"    Importaciones: {valor_promedio_imp:.2f} millones USD")

        return df_result

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def obtener_exportaciones(n=15):
    """
    Obtiene los últimos N valores de exportaciones de Colombia

    Args:
        n: Número de registros a obtener (default: 15)

    Returns:
        DataFrame con columnas: fecha, valor
    """
    print(f"\n{'='*80}")
    print(f"OBTENER EXPORTACIONES - Últimos {n} registros")
    print('='*80)

    # Descargar archivo
    excel_file = descargar_archivo_excel()
    if excel_file is None:
        return None

    # Encontrar hoja
    hoja = encontrar_hoja_datos(excel_file)
    if hoja is None:
        return None

    # Reiniciar el BytesIO
    excel_file.seek(0)

    # Extraer datos
    df = extraer_datos_balanza(excel_file, hoja, n)
    if df is None:
        return None

    # Retornar solo fecha y exportaciones
    df_exp = df[['fecha', 'exportaciones_usd_millones']].copy()
    df_exp.columns = ['fecha', 'valor']

    print(f"\n  Resumen de exportaciones:")
    print(f"    Registros: {len(df_exp)}")
    print(f"    Rango: {df_exp['fecha'].min()} a {df_exp['fecha'].max()}")
    print(f"    Valor promedio: {df_exp['valor'].mean():.2f} millones USD")

    return df_exp


def obtener_importaciones(n=15):
    """
    Obtiene los últimos N valores de importaciones de Colombia

    Args:
        n: Número de registros a obtener (default: 15)

    Returns:
        DataFrame con columnas: fecha, valor
    """
    print(f"\n{'='*80}")
    print(f"OBTENER IMPORTACIONES - Últimos {n} registros")
    print('='*80)

    # Descargar archivo
    excel_file = descargar_archivo_excel()
    if excel_file is None:
        return None

    # Encontrar hoja
    hoja = encontrar_hoja_datos(excel_file)
    if hoja is None:
        return None

    # Reiniciar el BytesIO
    excel_file.seek(0)

    # Extraer datos
    df = extraer_datos_balanza(excel_file, hoja, n)
    if df is None:
        return None

    # Retornar solo fecha e importaciones
    df_imp = df[['fecha', 'importaciones_usd_millones']].copy()
    df_imp.columns = ['fecha', 'valor']

    print(f"\n  Resumen de importaciones:")
    print(f"    Registros: {len(df_imp)}")
    print(f"    Rango: {df_imp['fecha'].min()} a {df_imp['fecha'].max()}")
    print(f"    Valor promedio: {df_imp['valor'].mean():.2f} millones USD")

    return df_imp


def obtener_balanza_completa(n=15):
    """
    Obtiene los últimos N valores de exportaciones e importaciones

    Args:
        n: Número de registros a obtener (default: 15)

    Returns:
        DataFrame con columnas: fecha, exportaciones_usd_millones, importaciones_usd_millones
    """
    print(f"\n{'='*80}")
    print(f"OBTENER BALANZA COMERCIAL COMPLETA - Últimos {n} registros")
    print('='*80)

    # Descargar archivo
    excel_file = descargar_archivo_excel()
    if excel_file is None:
        return None

    # Encontrar hoja
    hoja = encontrar_hoja_datos(excel_file)
    if hoja is None:
        return None

    # Reiniciar el BytesIO
    excel_file.seek(0)

    # Extraer datos
    df = extraer_datos_balanza(excel_file, hoja, n)

    return df


def guardar_balanza_csv(df, filepath=None):
    """
    Guarda los datos de balanza comercial en CSV

    Args:
        df: DataFrame con los datos
        filepath: Ruta del archivo (opcional)

    Returns:
        True si se guardó correctamente, False en caso contrario
    """
    if df is None or df.empty:
        print(f"  [ERROR] DataFrame vacío, no se puede guardar")
        return False

    try:
        if filepath is None:
            # Ruta por defecto
            base_path = Path(__file__).parent / 'data' / 'Macros' / 'TRADE_BALANCE'
            base_path.mkdir(parents=True, exist_ok=True)
            filepath = base_path / 'balanza_comercial_last15.csv'

        df.to_csv(filepath, index=False)
        print(f"\n  [SAVED] Archivo guardado: {filepath}")
        return True

    except Exception as e:
        print(f"  [ERROR] No se pudo guardar archivo: {e}")
        return False


# ============================================================================
# MAIN - TEST STANDALONE
# ============================================================================

def main():
    """
    Test standalone del scraper de Balanza Comercial DANE
    """
    print("=" * 80)
    print("TEST: SCRAPER BALANZA COMERCIAL - DANE COLOMBIA")
    print("=" * 80)

    # Test 1: Obtener balanza completa
    print("\n[TEST 1] Obteniendo balanza comercial completa...")
    df_completa = obtener_balanza_completa(n=15)

    if df_completa is not None and not df_completa.empty:
        print("\n[TEST 1] EXITO")
        print(f"\nPrimeros 5 registros:")
        print(df_completa.head())
        print(f"\nÚltimos 5 registros:")
        print(df_completa.tail())

        # Guardar
        guardar_balanza_csv(df_completa)
    else:
        print("\n[TEST 1] FALLO - No se pudieron obtener datos")

    # Test 2: Obtener solo exportaciones
    print("\n" + "-" * 80)
    print("\n[TEST 2] Obteniendo solo exportaciones...")
    df_exp = obtener_exportaciones(n=15)

    if df_exp is not None and not df_exp.empty:
        print("\n[TEST 2] EXITO")
        print(f"\nÚltimos 5 registros de exportaciones:")
        print(df_exp.tail())
    else:
        print("\n[TEST 2] FALLO")

    # Test 3: Obtener solo importaciones
    print("\n" + "-" * 80)
    print("\n[TEST 3] Obteniendo solo importaciones...")
    df_imp = obtener_importaciones(n=15)

    if df_imp is not None and not df_imp.empty:
        print("\n[TEST 3] EXITO")
        print(f"\nÚltimos 5 registros de importaciones:")
        print(df_imp.tail())
    else:
        print("\n[TEST 3] FALLO")

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    if df_completa is not None and df_exp is not None and df_imp is not None:
        print("\n[EXITO] Scraper de Balanza Comercial DANE funciona correctamente")
        print(f"\nÚltimo dato disponible:")
        ultimo = df_completa.iloc[-1]
        print(f"  Fecha: {ultimo['fecha']}")
        print(f"  Exportaciones: ${ultimo['exportaciones_usd_millones']:.2f} millones USD")
        print(f"  Importaciones: ${ultimo['importaciones_usd_millones']:.2f} millones USD")
        print(f"  Balanza: ${ultimo['exportaciones_usd_millones'] - ultimo['importaciones_usd_millones']:.2f} millones USD")

        print("\n\nPróximo paso:")
        print("  - Integrar en actualizar_diario_v2.py")
        print("  - Agregar a pipeline de datos macro")
    else:
        print("\n[FALLO] Scraper tiene problemas")
        print("\nPosibles causas:")
        print("  - DANE cambió formato del archivo Excel")
        print("  - URL del archivo cambió")
        print("  - Estructura de columnas diferente")
        print("  - Problema de conectividad")


if __name__ == "__main__":
    main()
