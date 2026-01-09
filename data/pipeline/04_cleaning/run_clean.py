"""
SCRIPT FINAL - Procesamiento de Datos Macro
============================================
Genera 3 datasets limpios listos para BD:
  1. MACRO_DAILY_CLEAN.csv      - Diario sin fines de semana
  2. MACRO_MONTHLY_CLEAN.csv    - Mensual normalizado a inicio de mes
  3. MACRO_QUARTERLY_CLEAN.csv  - Trimestral desde 2019-Q4

Combina historicos + recientes, priorizando datos recientes en overlapping.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURACION
# =============================================================================
BASE_PATH = Path(__file__).parent  # 04_cleaning/
PIPELINE_DIR = BASE_PATH.parent  # pipeline/

# Input: datos de fusion + datos de scrapers
INPUT_HIST = PIPELINE_DIR / "03_fusion" / "output"
INPUT_RECENT = PIPELINE_DIR / "02_scrapers" / "01_orchestrator" / "output"

# Fallback a ubicacion anterior si no existe
if not INPUT_RECENT.exists():
    INPUT_RECENT = PIPELINE_DIR / "03_processing" / "input" / "recent"

OUTPUT_PATH = BASE_PATH / "output"
OUTPUT_PATH.mkdir(exist_ok=True)

# Columnas mensuales que reportan a FIN DE MES (se shiftean al mes siguiente)
FIN_MES_COLS = [
    'INFL_CPI_TOTAL_COL_M_IPCCOL',
    'RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT',
    'FTRD_TERMS_TRADE_COL_M_TOT',
    'FXRT_REER_BILATERAL_COL_M_ITCR'
]

def parse_date_flex(d):
    """Parsea fechas en multiples formatos"""
    if pd.isna(d):
        return pd.NaT
    for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
        try:
            return pd.to_datetime(str(d).strip(), format=fmt)
        except:
            continue
    return pd.to_datetime(d, errors='coerce')

def clean_numeric(x):
    """Limpia formato europeo de numeros (1.010.338 -> 1010.338)"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x.count('.') > 1:
            parts = x.split('.')
            return float(''.join(parts[:-1]) + '.' + parts[-1])
        try:
            return float(x)
        except:
            return np.nan
    return x

print("=" * 70)
print("PROCESAMIENTO DE DATOS MACRO")
print("=" * 70)
print(f"Fecha ejecucion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output: {OUTPUT_PATH}")


# =============================================================================
# 1. DATOS DIARIOS
# =============================================================================
print("\n" + "-" * 70)
print("1. PROCESANDO DATOS DIARIOS")
print("-" * 70)

# Leer historico
df_daily_h = pd.read_csv(INPUT_HIST / "DATASET_MACRO_DAILY.csv")
df_daily_h['fecha'] = pd.to_datetime(df_daily_h['fecha'])
print(f"   Historico: {len(df_daily_h)} filas | {df_daily_h['fecha'].min().date()} a {df_daily_h['fecha'].max().date()}")

# Leer reciente
df_daily_r = pd.read_csv(INPUT_RECENT / "datos_diarios_hpc.csv")
df_daily_r['fecha'] = pd.to_datetime(df_daily_r['fecha'])
print(f"   Reciente:  {len(df_daily_r)} filas | {df_daily_r['fecha'].min().date()} a {df_daily_r['fecha'].max().date()}")

# Identificar todas las columnas
all_cols = sorted(set(df_daily_h.columns) | set(df_daily_r.columns) - {'fecha'})

# Crear DataFrame con todas las fechas
all_dates = sorted(set(df_daily_h['fecha']) | set(df_daily_r['fecha']))
df_daily = pd.DataFrame({'fecha': all_dates})

# Para cada columna, combinar datos (reciente tiene prioridad)
for col in all_cols:
    if col == 'fecha':
        continue
    df_daily[col] = np.nan

    # Primero poner historico
    if col in df_daily_h.columns:
        hist_map = df_daily_h.set_index('fecha')[col].to_dict()
        for idx, row in df_daily.iterrows():
            if row['fecha'] in hist_map and pd.notna(hist_map[row['fecha']]):
                df_daily.loc[idx, col] = hist_map[row['fecha']]

    # Luego sobrescribir con reciente
    if col in df_daily_r.columns:
        recent_map = df_daily_r.set_index('fecha')[col].to_dict()
        for idx, row in df_daily.iterrows():
            if row['fecha'] in recent_map and pd.notna(recent_map[row['fecha']]):
                df_daily.loc[idx, col] = recent_map[row['fecha']]

# Eliminar fines de semana
df_daily['dow'] = df_daily['fecha'].dt.dayofweek
df_daily = df_daily[df_daily['dow'] < 5].drop(columns=['dow'])
df_daily = df_daily.sort_values('fecha').reset_index(drop=True)

# Ordenar columnas
cols_order = ['fecha'] + sorted([c for c in df_daily.columns if c != 'fecha'])
df_daily = df_daily[cols_order]

# Guardar
df_daily.to_csv(OUTPUT_PATH / "MACRO_DAILY_CLEAN.csv", index=False)
print(f"   [OK] MACRO_DAILY_CLEAN.csv: {len(df_daily)} filas, {len(df_daily.columns)} cols")
print(f"        Rango: {df_daily['fecha'].min().date()} a {df_daily['fecha'].max().date()}")


# =============================================================================
# 2. DATOS MENSUALES (Normalizados a inicio de mes)
# =============================================================================
print("\n" + "-" * 70)
print("2. PROCESANDO DATOS MENSUALES")
print("-" * 70)

# Leer historico
df_monthly_h = pd.read_csv(INPUT_HIST / "DATASET_MACRO_MONTHLY.csv", sep=';')
df_monthly_h['fecha'] = df_monthly_h['fecha'].apply(parse_date_flex)
df_monthly_h['year_month'] = df_monthly_h['fecha'].dt.to_period('M')
df_monthly_h['day'] = df_monthly_h['fecha'].dt.day
print(f"   Historico: {len(df_monthly_h)} filas | {df_monthly_h['fecha'].min().date()} a {df_monthly_h['fecha'].max().date()}")

# Leer reciente
df_monthly_r = pd.read_csv(INPUT_RECENT / "datos_mensuales_hpc.csv")
df_monthly_r['fecha'] = pd.to_datetime(df_monthly_r['fecha'])
df_monthly_r['year_month'] = df_monthly_r['fecha'].dt.to_period('M')
df_monthly_r['day'] = df_monthly_r['fecha'].dt.day
print(f"   Reciente:  {len(df_monthly_r)} filas | {df_monthly_r['fecha'].min().date()} a {df_monthly_r['fecha'].max().date()}")

# Todas las columnas (excepto metadata)
meta_cols = {'fecha', 'year_month', 'day'}
all_cols_m = sorted((set(df_monthly_h.columns) | set(df_monthly_r.columns)) - meta_cols)

# Todos los meses
all_months = sorted(set(df_monthly_h['year_month']) | set(df_monthly_r['year_month']))
df_monthly = pd.DataFrame({'year_month': all_months})

print(f"   Meses a procesar: {len(all_months)} ({min(all_months)} a {max(all_months)})")

# Separar columnas por tipo de reporte
inicio_cols = [c for c in all_cols_m if c not in FIN_MES_COLS]
fin_cols = [c for c in all_cols_m if c in FIN_MES_COLS]

print(f"   Columnas inicio de mes: {len(inicio_cols)}")
print(f"   Columnas fin de mes (shift): {len(fin_cols)}")

# Procesar columnas de INICIO DE MES
for col in inicio_cols:
    df_monthly[col] = np.nan
    for idx, row in df_monthly.iterrows():
        ym = row['year_month']
        val = np.nan

        # Buscar en historico (dia 1-15)
        if col in df_monthly_h.columns:
            data = df_monthly_h[(df_monthly_h['year_month'] == ym) & (df_monthly_h['day'] <= 15)]
            vals = data[col].dropna()
            if len(vals) > 0:
                val = clean_numeric(vals.iloc[-1])

        # Sobrescribir con reciente
        if col in df_monthly_r.columns:
            data = df_monthly_r[(df_monthly_r['year_month'] == ym) & (df_monthly_r['day'] <= 15)]
            vals = data[col].dropna()
            if len(vals) > 0:
                val = clean_numeric(vals.iloc[-1])

        df_monthly.loc[idx, col] = val

# Procesar columnas de FIN DE MES (shift al mes siguiente)
for col in fin_cols:
    df_monthly[col] = np.nan
    for idx, row in df_monthly.iterrows():
        ym = row['year_month']
        prev_month = ym - 1  # Buscar en el mes anterior
        val = np.nan

        # Buscar en historico (dia 15-31 del mes anterior)
        if col in df_monthly_h.columns:
            data = df_monthly_h[(df_monthly_h['year_month'] == prev_month) & (df_monthly_h['day'] >= 15)]
            vals = data[col].dropna()
            if len(vals) > 0:
                val = clean_numeric(vals.iloc[-1])

        # Sobrescribir con reciente
        if col in df_monthly_r.columns:
            data = df_monthly_r[(df_monthly_r['year_month'] == prev_month) & (df_monthly_r['day'] >= 15)]
            vals = data[col].dropna()
            if len(vals) > 0:
                val = clean_numeric(vals.iloc[-1])

        df_monthly.loc[idx, col] = val

# Convertir year_month a fecha (inicio de mes)
df_monthly['fecha'] = df_monthly['year_month'].apply(lambda x: x.to_timestamp())
df_monthly = df_monthly.drop(columns=['year_month'])

# Ordenar columnas
cols_order = ['fecha'] + sorted([c for c in df_monthly.columns if c != 'fecha'])
df_monthly = df_monthly[cols_order]
df_monthly = df_monthly.sort_values('fecha').reset_index(drop=True)

# Guardar
df_monthly.to_csv(OUTPUT_PATH / "MACRO_MONTHLY_CLEAN.csv", index=False)
print(f"   [OK] MACRO_MONTHLY_CLEAN.csv: {len(df_monthly)} filas, {len(df_monthly.columns)} cols")
print(f"        Rango: {df_monthly['fecha'].min().date()} a {df_monthly['fecha'].max().date()}")
print(f"        Todas las fechas son dia 1 del mes")


# =============================================================================
# 3. DATOS TRIMESTRALES (desde 2019-Q4)
# =============================================================================
print("\n" + "-" * 70)
print("3. PROCESANDO DATOS TRIMESTRALES")
print("-" * 70)

# Leer historico
df_qtr_h = pd.read_csv(INPUT_HIST / "DATASET_MACRO_QUARTERLY.csv")
df_qtr_h['fecha'] = pd.to_datetime(df_qtr_h['fecha'])
df_qtr_h['quarter'] = df_qtr_h['fecha'].dt.to_period('Q')
print(f"   Historico: {len(df_qtr_h)} filas | {df_qtr_h['fecha'].min().date()} a {df_qtr_h['fecha'].max().date()}")

# Leer reciente
df_qtr_r = pd.read_csv(INPUT_RECENT / "datos_trimestrales_hpc.csv")
df_qtr_r['fecha'] = pd.to_datetime(df_qtr_r['fecha'])
df_qtr_r['quarter'] = df_qtr_r['fecha'].dt.to_period('Q')
print(f"   Reciente:  {len(df_qtr_r)} filas | {df_qtr_r['fecha'].min().date()} a {df_qtr_r['fecha'].max().date()}")

# Todas las columnas
meta_cols = {'fecha', 'quarter'}
all_cols_q = sorted((set(df_qtr_h.columns) | set(df_qtr_r.columns)) - meta_cols)

# Todos los trimestres desde 2019Q4
all_quarters = sorted(set(df_qtr_h['quarter']) | set(df_qtr_r['quarter']))
all_quarters = [q for q in all_quarters if q >= pd.Period('2019Q4')]
df_qtr = pd.DataFrame({'quarter': all_quarters})

print(f"   Trimestres a procesar: {len(all_quarters)} ({min(all_quarters)} a {max(all_quarters)})")

# Procesar cada columna
for col in all_cols_q:
    df_qtr[col] = np.nan
    for idx, row in df_qtr.iterrows():
        q = row['quarter']
        val = np.nan

        # Buscar en historico
        if col in df_qtr_h.columns:
            data = df_qtr_h[df_qtr_h['quarter'] == q]
            vals = data[col].dropna()
            if len(vals) > 0:
                val = vals.iloc[-1]

        # Sobrescribir con reciente
        if col in df_qtr_r.columns:
            data = df_qtr_r[df_qtr_r['quarter'] == q]
            vals = data[col].dropna()
            if len(vals) > 0:
                val = vals.iloc[-1]

        df_qtr.loc[idx, col] = val

# Convertir quarter a fecha (fin de trimestre)
df_qtr['fecha'] = df_qtr['quarter'].apply(lambda x: x.to_timestamp('Q'))
df_qtr = df_qtr.drop(columns=['quarter'])

# Ordenar columnas
cols_order = ['fecha'] + sorted([c for c in df_qtr.columns if c != 'fecha'])
df_qtr = df_qtr[cols_order]
df_qtr = df_qtr.sort_values('fecha').reset_index(drop=True)

# Guardar
df_qtr.to_csv(OUTPUT_PATH / "MACRO_QUARTERLY_CLEAN.csv", index=False)
print(f"   [OK] MACRO_QUARTERLY_CLEAN.csv: {len(df_qtr)} filas, {len(df_qtr.columns)} cols")
print(f"        Rango: {df_qtr['fecha'].min().date()} a {df_qtr['fecha'].max().date()}")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)

import os
for f in sorted(os.listdir(OUTPUT_PATH)):
    if f.endswith('.csv'):
        size = os.path.getsize(OUTPUT_PATH / f) / 1024
        print(f"   {f}: {size:.1f} KB")

print(f"""
DATASETS LISTOS PARA INSERTAR EN BASE DE DATOS:

1. MACRO_DAILY_CLEAN.csv
   - Frecuencia: Diaria (sin fines de semana)
   - Filas: {len(df_daily)}
   - Columnas: {len(df_daily.columns)}
   - Rango: {df_daily['fecha'].min().date()} a {df_daily['fecha'].max().date()}

2. MACRO_MONTHLY_CLEAN.csv
   - Frecuencia: Mensual (normalizado a inicio de mes, dia 1)
   - Filas: {len(df_monthly)}
   - Columnas: {len(df_monthly.columns)}
   - Rango: {df_monthly['fecha'].min().date()} a {df_monthly['fecha'].max().date()}
   - Columnas shifteadas de fin a inicio: {fin_cols}

3. MACRO_QUARTERLY_CLEAN.csv
   - Frecuencia: Trimestral (desde 2019-Q4)
   - Filas: {len(df_qtr)}
   - Columnas: {len(df_qtr.columns)}
   - Rango: {df_qtr['fecha'].min().date()} a {df_qtr['fecha'].max().date()}
""")

# Mostrar sample de cada uno
print("-" * 70)
print("SAMPLE - Ultimas 5 filas de cada dataset:")
print("-" * 70)

print("\nDIARIO:")
print(df_daily.tail().to_string(index=False))

print("\nMENSUAL:")
print(df_monthly.tail().to_string(index=False))

print("\nTRIMESTRAL:")
print(df_qtr.tail().to_string(index=False))
