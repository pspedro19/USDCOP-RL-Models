# -*- coding: utf-8 -*-
"""
Fallback Manual para Cuenta Corriente
======================================

Si los scrapers automáticos fallan, este módulo lee un archivo CSV descargado manualmente.

USO:
1. Ir a: https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/414001
2. Click en "Vista tabla"
3. Click en botón de descarga (ícono de Excel)
4. Seleccionar "Balanza de pagos - Cuenta corriente - bienes y servicios - trimestral"
5. Guardar el archivo como: cuenta_corriente_manual.csv o cuenta_corriente_manual.xlsx

El sistema buscará automáticamente este archivo y lo usará si existe.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional


def obtener_cuenta_corriente_manual(n: int = 20) -> Optional[pd.DataFrame]:
    """
    Lee Cuenta Corriente desde archivo manual si existe.

    Busca archivos en este orden:
    1. suameca_414001_cuenta_corriente_datos.csv (generado por scrapers)
    2. cuenta_corriente_manual.csv
    3. cuenta_corriente_manual.xlsx
    4. graficador_series.xlsx (descarga directa de SUAMECA)

    Args:
        n: Número de registros más recientes

    Returns:
        DataFrame con columnas ['Fecha', 'Valor'] o None si no encuentra archivo
    """

    base_path = Path(__file__).parent

    # Lista de archivos posibles (en orden de preferencia)
    archivos_posibles = [
        base_path / "suameca_414001_cuenta_corriente_datos.csv",
        base_path / "cuenta_corriente_manual.csv",
        base_path / "cuenta_corriente_manual.xlsx",
        base_path / "graficador_series.xlsx",
        base_path / "descargas_cuenta_corriente" / "graficador_series.xlsx",
    ]

    for archivo in archivos_posibles:
        if archivo.exists():
            try:
                print(f"[INFO] Leyendo archivo manual: {archivo.name}")

                # Leer según extensión
                if archivo.suffix.lower() == '.csv':
                    df = pd.read_csv(archivo)
                else:  # .xlsx o .xls
                    df = pd.read_excel(archivo, sheet_name=0)

                # Buscar columnas
                fecha_col = None
                valor_col = None

                # Buscar columna de fecha
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'fecha' in col_lower or 'periodo' in col_lower or 'date' in col_lower:
                        fecha_col = col
                        break

                # Si no hay columna de fecha explícita, usar la primera
                if not fecha_col:
                    fecha_col = df.columns[0]

                # Buscar columna de valor (la que contiene "cuenta corriente" y "bienes")
                for col in df.columns:
                    col_lower = str(col).lower()
                    if ('cuenta corriente' in col_lower and
                        'bienes' in col_lower and
                        'servicios' in col_lower):
                        # Excluir otras series
                        if ('financiera' not in col_lower and
                            'ingreso' not in col_lower and
                            'transferencias' not in col_lower):
                            valor_col = col
                            break

                # Si no encontró columna específica, usar la segunda columna
                if not valor_col:
                    if len(df.columns) >= 2:
                        valor_col = df.columns[1]
                    else:
                        print(f"[ERROR] No se pudo identificar columna de valores")
                        continue

                # Crear DataFrame limpio
                df_clean = pd.DataFrame()
                df_clean['Fecha'] = pd.to_datetime(df[fecha_col], errors='coerce')
                df_clean['Valor'] = pd.to_numeric(df[valor_col], errors='coerce')

                # Eliminar nulos
                df_clean = df_clean.dropna()

                # Ordenar por fecha descendente
                df_clean = df_clean.sort_values('Fecha', ascending=False).reset_index(drop=True)

                # Tomar últimos n registros
                df_result = df_clean.head(n)

                if len(df_result) > 0:
                    print(f"[OK] {len(df_result)} registros cargados desde archivo manual")
                    print(f"[OK] Último valor: {df_result['Valor'].iloc[0]:.2f} ({df_result['Fecha'].iloc[0].date()})")
                    return df_result

            except Exception as e:
                print(f"[WARN] Error leyendo {archivo.name}: {str(e)}")
                continue

    print("[INFO] No se encontró archivo manual de Cuenta Corriente")
    return None


def main():
    """Test del lector manual"""
    print("="*80)
    print("TEST - LECTOR MANUAL DE CUENTA CORRIENTE")
    print("="*80)

    df = obtener_cuenta_corriente_manual(n=10)

    if df is not None:
        print("\n✅ ÉXITO - Datos cargados:\n")
        print(df.to_string(index=False))
    else:
        print("\n❌ No se encontraron archivos manuales")
        print("\nPara usar esta función:")
        print("1. Descarga el archivo desde SUAMECA")
        print("2. Guárdalo como 'cuenta_corriente_manual.csv' o 'cuenta_corriente_manual.xlsx'")
        print("3. Colócalo en la carpeta actualizador_produccion/")


if __name__ == '__main__':
    main()
