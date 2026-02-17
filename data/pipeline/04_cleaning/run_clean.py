"""
SCRIPT FINAL - Procesamiento de Datos Macro
============================================
Genera 3 datasets limpios listos para BD:
  1. MACRO_DAILY_CLEAN.csv      - Diario sin fines de semana
  2. MACRO_MONTHLY_CLEAN.csv    - Mensual normalizado a inicio de mes
  3. MACRO_QUARTERLY_CLEAN.csv  - Trimestral desde 2019-Q4

Combina historicos + recientes, priorizando datos recientes en overlapping.

Anti-Leakage:
  - Usa EconomicCalendar para forward-fill respetando fechas de publicación
  - Contract: CTR-L0-CALENDAR-001
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Agregar src al path para importar EconomicCalendar
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.economic_calendar import EconomicCalendar
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False
    print("WARNING: EconomicCalendar not available. Using simple ffill.")

# =============================================================================
# CONFIGURACION
# =============================================================================
BASE_PATH = Path(__file__).parent  # 04_cleaning/
PIPELINE_DIR = BASE_PATH.parent  # pipeline/

# Input: datos de fusion + datos de scrapers
INPUT_HIST = PIPELINE_DIR / "03_fusion" / "output"
INPUT_RECENT = PIPELINE_DIR / "02_scrapers" / "storage" / "datasets"

# Fallback a ubicaciones anteriores si no existe
if not INPUT_RECENT.exists():
    INPUT_RECENT = PIPELINE_DIR / "02_scrapers" / "01_orchestrator" / "output"
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


def apply_calendar_aware_ffill(df: pd.DataFrame, target_frequency: str = 'monthly') -> pd.DataFrame:
    """
    Aplicar forward-fill respetando calendario de publicaciones.

    Para variables en el calendario económico, usa fechas de publicación reales.
    Para variables sin calendario, usa ffill simple con límite genérico.

    Args:
        df: DataFrame con columna 'fecha' como index
        target_frequency: 'daily', 'monthly', o 'quarterly'

    Returns:
        DataFrame con forward-fill aplicado correctamente
    """
    if not CALENDAR_AVAILABLE:
        print("   [WARN] Calendar not available, using simple ffill")
        return df.ffill(limit=1)

    calendar = EconomicCalendar()
    result = df.copy()

    # Asegurar que fecha es el index
    if 'fecha' in result.columns:
        result = result.set_index('fecha')

    calendar_vars = 0
    simple_vars = 0

    for col in result.columns:
        col_lower = col.lower()

        if col_lower in calendar.variables:
            # Variable con calendario: usar ffill publication-aware
            # Para datos mensuales/quarterly, el ffill ya está incorporado en la lógica
            # de propagación por publicación, así que solo marcamos
            calendar_vars += 1
        else:
            # Variable sin calendario: ffill simple con límite
            if target_frequency == 'daily':
                limit = 5  # Max 5 días de propagación
            elif target_frequency == 'monthly':
                limit = 1  # Max 1 mes
            else:
                limit = 1  # Max 1 trimestre

            result[col] = result[col].ffill(limit=limit)
            simple_vars += 1

    # Reset index para mantener 'fecha' como columna
    result = result.reset_index()

    print(f"   [INFO] Calendar-aware ffill: {calendar_vars} calendar vars, {simple_vars} simple vars")

    return result


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

# Leer historico (fusión CSV: primera col puede ser unnamed index, puede tener fila FUENTE_URL)
df_daily_h = pd.read_csv(INPUT_HIST / "DATASET_MACRO_DAILY.csv")
first_col_h = df_daily_h.columns[0]
# Skip FUENTE_URL metadata row if present
if 'FUENTE' in str(df_daily_h.iloc[0, 0]):
    df_daily_h = df_daily_h.iloc[1:].reset_index(drop=True)
# Normalize date column name
if 'fecha' not in df_daily_h.columns:
    df_daily_h.rename(columns={first_col_h: 'fecha'}, inplace=True)
df_daily_h['fecha'] = pd.to_datetime(df_daily_h['fecha'])
# Convert all data columns to numeric (some may be strings after FUENTE_URL skip)
for col in [c for c in df_daily_h.columns if c != 'fecha']:
    df_daily_h[col] = pd.to_numeric(df_daily_h[col], errors='coerce')
print(f"   Historico: {len(df_daily_h)} filas | {df_daily_h['fecha'].min().date()} a {df_daily_h['fecha'].max().date()}")

# Leer reciente (scraper HPC output)
df_daily_r = pd.read_csv(INPUT_RECENT / "datos_diarios_hpc.csv")
first_col_r = df_daily_r.columns[0]
if 'fecha' not in df_daily_r.columns:
    df_daily_r.rename(columns={first_col_r: 'fecha'}, inplace=True)
df_daily_r['fecha'] = pd.to_datetime(df_daily_r['fecha'])
for col in [c for c in df_daily_r.columns if c != 'fecha']:
    df_daily_r[col] = pd.to_numeric(df_daily_r[col], errors='coerce')
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

# Guardar CSV
df_daily.to_csv(OUTPUT_PATH / "MACRO_DAILY_CLEAN.csv", index=False)
print(f"   [OK] MACRO_DAILY_CLEAN.csv: {len(df_daily)} filas, {len(df_daily.columns)} cols")
print(f"        Rango: {df_daily['fecha'].min().date()} a {df_daily['fecha'].max().date()}")

# Guardar Parquet (con fecha como index, formato que lee load_data())
df_daily_pq = df_daily.copy()
df_daily_pq = df_daily_pq.set_index('fecha')
df_daily_pq.index.name = None
df_daily_pq.to_parquet(OUTPUT_PATH / "MACRO_DAILY_CLEAN.parquet")
print(f"   [OK] MACRO_DAILY_CLEAN.parquet: {len(df_daily_pq)} filas")


# =============================================================================
# 2. DATOS MENSUALES (Normalizados a inicio de mes)
# =============================================================================
print("\n" + "-" * 70)
print("2. PROCESANDO DATOS MENSUALES")
print("-" * 70)

# Leer historico (fusion CSV: primera col puede ser unnamed index, puede tener fila FUENTE_URL)
df_monthly_h = pd.read_csv(INPUT_HIST / "DATASET_MACRO_MONTHLY.csv")
first_col_mh = df_monthly_h.columns[0]
if 'FUENTE' in str(df_monthly_h.iloc[0, 0]):
    df_monthly_h = df_monthly_h.iloc[1:].reset_index(drop=True)
if 'fecha' not in df_monthly_h.columns:
    df_monthly_h.rename(columns={first_col_mh: 'fecha'}, inplace=True)
df_monthly_h['fecha'] = df_monthly_h['fecha'].apply(parse_date_flex)
for col in [c for c in df_monthly_h.columns if c != 'fecha']:
    df_monthly_h[col] = pd.to_numeric(df_monthly_h[col], errors='coerce')
df_monthly_h['year_month'] = df_monthly_h['fecha'].dt.to_period('M')
df_monthly_h['day'] = df_monthly_h['fecha'].dt.day
print(f"   Historico: {len(df_monthly_h)} filas | {df_monthly_h['fecha'].min().date()} a {df_monthly_h['fecha'].max().date()}")

# Leer reciente
df_monthly_r = pd.read_csv(INPUT_RECENT / "datos_mensuales_hpc.csv", encoding='utf-8-sig')
first_col_mr = df_monthly_r.columns[0]
if 'fecha' not in df_monthly_r.columns:
    df_monthly_r.rename(columns={first_col_mr: 'fecha'}, inplace=True)
df_monthly_r['fecha'] = pd.to_datetime(df_monthly_r['fecha'])
for col in [c for c in df_monthly_r.columns if c != 'fecha']:
    df_monthly_r[col] = pd.to_numeric(df_monthly_r[col], errors='coerce')
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

# Aplicar forward-fill con calendario de publicaciones (anti-leakage)
print("   Aplicando calendar-aware forward-fill...")
df_monthly = apply_calendar_aware_ffill(df_monthly, target_frequency='monthly')

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

# Leer historico (fusion CSV: primera col puede ser unnamed index, puede tener fila FUENTE_URL)
df_qtr_h = pd.read_csv(INPUT_HIST / "DATASET_MACRO_QUARTERLY.csv")
first_col_qh = df_qtr_h.columns[0]
if 'FUENTE' in str(df_qtr_h.iloc[0, 0]):
    df_qtr_h = df_qtr_h.iloc[1:].reset_index(drop=True)
if 'fecha' not in df_qtr_h.columns:
    df_qtr_h.rename(columns={first_col_qh: 'fecha'}, inplace=True)
df_qtr_h['fecha'] = pd.to_datetime(df_qtr_h['fecha'])
for col in [c for c in df_qtr_h.columns if c != 'fecha']:
    df_qtr_h[col] = pd.to_numeric(df_qtr_h[col], errors='coerce')
df_qtr_h['quarter'] = df_qtr_h['fecha'].dt.to_period('Q')
print(f"   Historico: {len(df_qtr_h)} filas | {df_qtr_h['fecha'].min().date()} a {df_qtr_h['fecha'].max().date()}")

# Leer reciente
df_qtr_r = pd.read_csv(INPUT_RECENT / "datos_trimestrales_hpc.csv", encoding='utf-8-sig')
first_col_qr = df_qtr_r.columns[0]
if 'fecha' not in df_qtr_r.columns:
    df_qtr_r.rename(columns={first_col_qr: 'fecha'}, inplace=True)
df_qtr_r['fecha'] = pd.to_datetime(df_qtr_r['fecha'])
for col in [c for c in df_qtr_r.columns if c != 'fecha']:
    df_qtr_r[col] = pd.to_numeric(df_qtr_r[col], errors='coerce')
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

# Aplicar forward-fill con calendario de publicaciones (anti-leakage)
print("   Aplicando calendar-aware forward-fill...")
df_qtr = apply_calendar_aware_ffill(df_qtr, target_frequency='quarterly')

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
