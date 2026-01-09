"""
SCRIPT DE RESAMPLEO CONSOLIDADO DE DATOS MACRO
===============================================
Genera 2 datasets finales:
  1. MACRO_DAILY_CONSOLIDATED.csv  - Diario + Mensual + Trimestral fusionados
  2. MACRO_5MIN_CONSOLIDATED.csv   - Resampleado a 5min (L-V, 8:00am-12:55pm)

Estrategia:
- merge_asof con direction='backward' para evitar look-ahead bias
- SIN ffill para gaps (datos crudos de la fuente)
- Solo expande datos diarios a 5min (sin rellenar valores faltantes)
- Filtra festivos de Colombia usando colombian-holidays
- Grid de 5min: 60 intervalos por dia

VERSION 2.0 - Actualizado para mantener datos crudos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Importar libreria de festivos colombianos
try:
    from colombian_holidays import is_holiday
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    print("[WARN] colombian-holidays no instalado. Ejecutar: pip install colombian-holidays")

# =============================================================================
# CONFIGURACION
# =============================================================================
BASE_PATH = Path(__file__).parent  # 05_resampling/
PIPELINE_DIR = BASE_PATH.parent  # pipeline/

# Input: datos limpios de 04_cleaning
INPUT_PATH = PIPELINE_DIR / "04_cleaning" / "output"

# Fallback a ubicacion anterior
if not INPUT_PATH.exists():
    INPUT_PATH = PIPELINE_DIR / "03_processing" / "intermediate"

OUTPUT_PATH = BASE_PATH / "output"
OUTPUT_PATH.mkdir(exist_ok=True)

# Horario de mercado Colombia (COT)
MARKET_START_HOUR = 8
MARKET_START_MIN = 0
MARKET_END_HOUR = 12
MARKET_END_MIN = 55
TIMEZONE = 'America/Bogota'

print("=" * 70)
print("RESAMPLEO CONSOLIDADO DE DATOS MACRO v2.0")
print("=" * 70)
print(f"Fecha ejecucion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output: {OUTPUT_PATH}")
print(f"Festivos Colombia: {'Habilitado' if HOLIDAYS_AVAILABLE else 'Deshabilitado'}")


# =============================================================================
# 1. CARGAR DATASETS LIMPIOS
# =============================================================================
print("\n" + "-" * 70)
print("1. CARGANDO DATASETS")
print("-" * 70)

# Diario
df_daily = pd.read_csv(INPUT_PATH / "MACRO_DAILY_CLEAN.csv")
df_daily['fecha'] = pd.to_datetime(df_daily['fecha'])
print(f"   DIARIO: {len(df_daily)} filas, {len(df_daily.columns)} cols")
print(f"           {df_daily['fecha'].min().date()} a {df_daily['fecha'].max().date()}")

# Mensual
df_monthly = pd.read_csv(INPUT_PATH / "MACRO_MONTHLY_CLEAN.csv")
df_monthly['fecha'] = pd.to_datetime(df_monthly['fecha'])
print(f"   MENSUAL: {len(df_monthly)} filas, {len(df_monthly.columns)} cols")
print(f"            {df_monthly['fecha'].min().date()} a {df_monthly['fecha'].max().date()}")

# Trimestral
df_quarterly = pd.read_csv(INPUT_PATH / "MACRO_QUARTERLY_CLEAN.csv")
df_quarterly['fecha'] = pd.to_datetime(df_quarterly['fecha'])
print(f"   TRIMESTRAL: {len(df_quarterly)} filas, {len(df_quarterly.columns)} cols")
print(f"               {df_quarterly['fecha'].min().date()} a {df_quarterly['fecha'].max().date()}")


# =============================================================================
# 2. CREAR CALENDARIO BASE (DIAS HABILES SIN FESTIVOS)
# =============================================================================
print("\n" + "-" * 70)
print("2. CREANDO CALENDARIO BASE (L-V sin festivos)")
print("-" * 70)

# Usar el rango del dataset diario como base
min_date = df_daily['fecha'].min()
max_date = df_daily['fecha'].max()

# Crear calendario de dias habiles (sin fines de semana)
all_bdays = pd.bdate_range(start=min_date, end=max_date, freq='B')
df_calendar = pd.DataFrame({'fecha': all_bdays})

print(f"   Dias habiles (L-V): {len(df_calendar)}")

# Filtrar festivos de Colombia
if HOLIDAYS_AVAILABLE:
    def is_colombian_holiday(date):
        """Verifica si una fecha es festivo en Colombia"""
        try:
            return is_holiday(date.to_pydatetime())
        except:
            return False

    # Aplicar filtro
    mask_holidays = df_calendar['fecha'].apply(is_colombian_holiday)
    holidays_removed = mask_holidays.sum()
    df_calendar = df_calendar[~mask_holidays].copy()

    print(f"   Festivos removidos: {holidays_removed}")
    print(f"   Dias finales: {len(df_calendar)}")
else:
    print(f"   [WARN] Sin filtro de festivos")

print(f"   Rango: {min_date.date()} a {max_date.date()}")


# =============================================================================
# 3. MERGE DIARIO (DIRECTO - SIN FFILL)
# =============================================================================
print("\n" + "-" * 70)
print("3. MERGE DATASET DIARIO (sin ffill)")
print("-" * 70)

# Merge directo con calendario - los dias sin datos quedaran con NaN
df_merged = df_calendar.merge(df_daily, on='fecha', how='left')
print(f"   Despues de merge diario: {len(df_merged)} filas")

# Contar NaN
daily_cols = [c for c in df_daily.columns if c != 'fecha']
nan_daily = df_merged[daily_cols].isna().sum().sum()
print(f"   NaN en columnas diarias: {nan_daily}")


# =============================================================================
# 4. MERGE MENSUAL (MERGE_ASOF - SIN LOOK-AHEAD, SIN FFILL)
# =============================================================================
print("\n" + "-" * 70)
print("4. MERGE DATASET MENSUAL (merge_asof backward, sin ffill)")
print("-" * 70)

# Preparar mensual
df_monthly_sorted = df_monthly.sort_values('fecha').copy()
monthly_cols = [c for c in df_monthly.columns if c != 'fecha']

# Ordenar merged
df_merged = df_merged.sort_values('fecha')

# MERGE_ASOF: Para cada fecha, toma el valor mensual MAS RECIENTE disponible
# direction='backward' garantiza que NO hay look-ahead bias
# SIN tolerance - si no hay dato reciente, queda NaN
df_merged = pd.merge_asof(
    df_merged,
    df_monthly_sorted,
    on='fecha',
    direction='backward',  # Solo usa datos del pasado
    tolerance=pd.Timedelta('35 days')  # Max 35 dias (1 mes + margen)
)

print(f"   Despues de merge mensual: {len(df_merged)} filas")
print(f"   Columnas mensuales agregadas: {len(monthly_cols)}")


# =============================================================================
# 5. MERGE TRIMESTRAL (MERGE_ASOF - SIN LOOK-AHEAD, SIN FFILL)
# =============================================================================
print("\n" + "-" * 70)
print("5. MERGE DATASET TRIMESTRAL (merge_asof backward, sin ffill)")
print("-" * 70)

# Preparar trimestral
df_quarterly_sorted = df_quarterly.sort_values('fecha').copy()
quarterly_cols = [c for c in df_quarterly.columns if c != 'fecha']

# Renombrar (agregar sufijo _Q para identificar)
rename_map_q = {c: f"{c}_Q" for c in quarterly_cols}
df_quarterly_renamed = df_quarterly_sorted.rename(columns=rename_map_q)

# MERGE_ASOF trimestral - SIN ffill
df_merged = pd.merge_asof(
    df_merged,
    df_quarterly_renamed,
    on='fecha',
    direction='backward',
    tolerance=pd.Timedelta('95 days')  # Max 95 dias (1 trimestre + margen)
)

print(f"   Despues de merge trimestral: {len(df_merged)} filas")
print(f"   Columnas trimestrales agregadas: {len(quarterly_cols)}")


# =============================================================================
# 6. ESTADISTICAS DE COMPLETITUD (SIN FFILL)
# =============================================================================
print("\n" + "-" * 70)
print("6. ESTADISTICAS DE COMPLETITUD (DATOS CRUDOS)")
print("-" * 70)

df_result = df_merged.copy()

# Identificar tipos de columnas
all_data_cols = [c for c in df_result.columns if c != 'fecha']

# Columnas originalmente diarias
daily_original_cols = [c for c in daily_cols if c in df_result.columns]

# Columnas mensuales
monthly_integrated_cols = [c for c in monthly_cols if c in df_result.columns]

# Columnas trimestrales (con sufijo _Q)
quarterly_integrated_cols = [c for c in df_result.columns if c.endswith('_Q')]

print(f"   Columnas diarias: {len(daily_original_cols)}")
print(f"   Columnas mensuales: {len(monthly_integrated_cols)}")
print(f"   Columnas trimestrales: {len(quarterly_integrated_cols)}")

# Calcular completitud por tipo
def calc_completitud(df, cols):
    if not cols:
        return 0.0
    total = len(df) * len(cols)
    nans = df[cols].isna().sum().sum()
    return (1 - nans / total) * 100 if total > 0 else 0.0

print(f"\n   Completitud por tipo:")
print(f"   - Diarias: {calc_completitud(df_result, daily_original_cols):.1f}%")
print(f"   - Mensuales: {calc_completitud(df_result, monthly_integrated_cols):.1f}%")
print(f"   - Trimestrales: {calc_completitud(df_result, quarterly_integrated_cols):.1f}%")

total_completitud = calc_completitud(df_result, all_data_cols)
print(f"   - TOTAL: {total_completitud:.1f}%")


# =============================================================================
# 7. ORDENAR COLUMNAS Y GUARDAR DATASET DIARIO
# =============================================================================
print("\n" + "-" * 70)
print("7. GUARDANDO DATASET DIARIO CONSOLIDADO")
print("-" * 70)

# Ordenar columnas: fecha primero, luego alfabetico
cols_order = ['fecha'] + sorted([c for c in df_result.columns if c != 'fecha'])
df_daily_consolidated = df_result[cols_order].copy()
df_daily_consolidated = df_daily_consolidated.sort_values('fecha').reset_index(drop=True)

# Guardar
output_daily = OUTPUT_PATH / "MACRO_DAILY_CONSOLIDATED.csv"
df_daily_consolidated.to_csv(output_daily, index=False)

print(f"   Archivo: {output_daily.name}")
print(f"   Filas: {len(df_daily_consolidated)}")
print(f"   Columnas: {len(df_daily_consolidated.columns)}")
print(f"   Rango: {df_daily_consolidated['fecha'].min().date()} a {df_daily_consolidated['fecha'].max().date()}")


# =============================================================================
# 8. GENERAR GRID DE 5 MINUTOS (EXPANSION DE DATOS DIARIOS)
# =============================================================================
print("\n" + "-" * 70)
print("8. GENERANDO GRID DE 5 MINUTOS")
print("-" * 70)

# Crear grid de tiempos (8:00 a 12:55 cada 5min)
time_slots = []
h, m = MARKET_START_HOUR, MARKET_START_MIN
while (h < MARKET_END_HOUR) or (h == MARKET_END_HOUR and m <= MARKET_END_MIN):
    time_slots.append((h, m))
    m += 5
    if m >= 60:
        m = 0
        h += 1

print(f"   Horario: {MARKET_START_HOUR:02d}:{MARKET_START_MIN:02d} - {MARKET_END_HOUR:02d}:{MARKET_END_MIN:02d}")
print(f"   Intervalos por dia: {len(time_slots)}")

# Expandir cada dia a sus intervalos de 5min
# NOTA: Aqui SI se repite el valor diario para cada barra de 5min del mismo dia
#       Esto NO es ffill de gaps, es la expansion natural de dato diario a 5min
print("   Expandiendo datos diarios a 5min (1 valor por dia -> 60 barras)...")

expanded_rows = []
total_days = len(df_daily_consolidated)

for idx, row in df_daily_consolidated.iterrows():
    base_date = row['fecha']

    # Ya filtramos festivos arriba, pero verificar weekday por seguridad
    if base_date.weekday() >= 5:
        continue

    for hour, minute in time_slots:
        # Crear nueva fila con los datos del dia
        new_row = row.to_dict()

        # Crear timestamp
        try:
            ts_cot = pd.Timestamp(
                year=base_date.year,
                month=base_date.month,
                day=base_date.day,
                hour=hour,
                minute=minute,
                tz=TIMEZONE
            )
            ts_utc = ts_cot.tz_convert('UTC')
        except:
            # Si hay error de timezone, usar naive
            ts_cot = pd.Timestamp(
                year=base_date.year,
                month=base_date.month,
                day=base_date.day,
                hour=hour,
                minute=minute
            )
            ts_utc = ts_cot

        new_row['datetime_utc'] = ts_utc
        new_row['datetime_cot'] = ts_cot
        new_row['hour'] = hour
        new_row['minute'] = minute
        new_row['t_in_day'] = (hour - MARKET_START_HOUR) * 12 + minute // 5  # 0-59

        expanded_rows.append(new_row)

    # Progreso
    if (idx + 1) % 500 == 0:
        print(f"      Procesados {idx + 1}/{total_days} dias...")

df_5min = pd.DataFrame(expanded_rows)
print(f"   Expansion completada: {len(df_5min)} filas")


# =============================================================================
# 9. ORDENAR Y GUARDAR DATASET 5MIN
# =============================================================================
print("\n" + "-" * 70)
print("9. GUARDANDO DATASET 5MIN CONSOLIDADO")
print("-" * 70)

# Ordenar columnas: datetime primero, luego el resto
cols_5min = ['datetime_utc', 'datetime_cot', 'fecha', 'hour', 'minute', 't_in_day']
other_cols = sorted([c for c in df_5min.columns if c not in cols_5min])
cols_order_5min = cols_5min + other_cols

df_5min_consolidated = df_5min[cols_order_5min].copy()
df_5min_consolidated = df_5min_consolidated.sort_values('datetime_utc').reset_index(drop=True)

# Guardar
output_5min = OUTPUT_PATH / "MACRO_5MIN_CONSOLIDATED.csv"
df_5min_consolidated.to_csv(output_5min, index=False)

print(f"   Archivo: {output_5min.name}")
print(f"   Filas: {len(df_5min_consolidated)}")
print(f"   Columnas: {len(df_5min_consolidated.columns)}")


# =============================================================================
# 10. VALIDACIONES
# =============================================================================
print("\n" + "-" * 70)
print("10. VALIDACIONES")
print("-" * 70)

# Validacion 1: No hay fines de semana
df_5min_consolidated['dow'] = pd.to_datetime(df_5min_consolidated['fecha']).dt.dayofweek
weekends = df_5min_consolidated[df_5min_consolidated['dow'] >= 5]
check_no_weekends = len(weekends) == 0
print(f"   [{'OK' if check_no_weekends else 'FAIL'}] Sin fines de semana: {len(weekends)} filas")

# Validacion 2: No hay festivos colombianos
if HOLIDAYS_AVAILABLE:
    holidays_in_data = df_5min_consolidated['fecha'].apply(
        lambda x: is_holiday(x.to_pydatetime()) if hasattr(x, 'to_pydatetime') else False
    ).sum()
    check_no_holidays = holidays_in_data == 0
    print(f"   [{'OK' if check_no_holidays else 'FAIL'}] Sin festivos Colombia: {holidays_in_data} filas")
else:
    print(f"   [SKIP] Validacion festivos (libreria no disponible)")

# Validacion 3: Grid completo por dia
intervals_per_day = df_5min_consolidated.groupby('fecha').size()
expected_intervals = len(time_slots)
incomplete_days = intervals_per_day[intervals_per_day != expected_intervals]
check_complete_grid = len(incomplete_days) == 0
print(f"   [{'OK' if check_complete_grid else 'FAIL'}] Grid completo: {len(incomplete_days)} dias incompletos")

# Validacion 4: Ordenamiento cronologico
is_sorted = df_5min_consolidated['datetime_utc'].is_monotonic_increasing
print(f"   [{'OK' if is_sorted else 'FAIL'}] Ordenamiento cronologico")

# Validacion 5: Completitud de datos
data_cols = [c for c in df_5min_consolidated.columns if c not in ['datetime_utc', 'datetime_cot', 'fecha', 'hour', 'minute', 't_in_day', 'dow']]
total_data_cells = len(df_5min_consolidated) * len(data_cols)
nan_count = df_5min_consolidated[data_cols].isna().sum().sum()
completitud_5min = (1 - nan_count / total_data_cells) * 100
print(f"   [INFO] Completitud datos: {completitud_5min:.2f}%")
print(f"   [INFO] Celdas con NaN: {nan_count:,} de {total_data_cells:,}")

# Limpiar columna temporal
df_5min_consolidated = df_5min_consolidated.drop(columns=['dow'])
df_5min_consolidated.to_csv(output_5min, index=False)


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)

import os
print(f"\nArchivos generados en: {OUTPUT_PATH}\n")

for f in sorted(os.listdir(OUTPUT_PATH)):
    if f.endswith('.csv'):
        fpath = OUTPUT_PATH / f
        size_kb = os.path.getsize(fpath) / 1024
        size_mb = size_kb / 1024
        if size_mb > 1:
            print(f"   {f}: {size_mb:.1f} MB")
        else:
            print(f"   {f}: {size_kb:.1f} KB")

print(f"""
DATASETS CONSOLIDADOS (DATOS CRUDOS - SIN FFILL):

1. MACRO_DAILY_CONSOLIDATED.csv
   - Frecuencia: DIARIA (L-V sin festivos Colombia)
   - Filas: {len(df_daily_consolidated)}
   - Columnas: {len(df_daily_consolidated.columns)}
   - Rango: {df_daily_consolidated['fecha'].min().date()} a {df_daily_consolidated['fecha'].max().date()}
   - Incluye: Daily + Monthly + Quarterly fusionados via merge_asof
   - SIN ffill aplicado (datos crudos de la fuente)

2. MACRO_5MIN_CONSOLIDATED.csv
   - Frecuencia: 5 MINUTOS (L-V, 8:00-12:55 COT, sin festivos)
   - Filas: {len(df_5min_consolidated)}
   - Columnas: {len(df_5min_consolidated.columns)}
   - Intervalos/dia: {len(time_slots)}
   - Dias totales: {df_5min_consolidated['fecha'].nunique()}
   - Completitud: {completitud_5min:.2f}%
   - NaN presentes: SI (datos crudos sin rellenar)

ESTRATEGIA UTILIZADA:
- merge_asof con direction='backward' (sin look-ahead bias)
- SIN ffill para gaps (mantiene NaN donde no hay dato)
- Expansion diario->5min: 1 valor diario repetido en 60 barras (correcto)
- Filtro: L-V sin festivos Colombia (colombian-holidays)
- Horario: {MARKET_START_HOUR:02d}:{MARKET_START_MIN:02d} - {MARKET_END_HOUR:02d}:{MARKET_END_MIN:02d} COT
""")

# Mostrar muestra de NaN por columna
print("-" * 70)
print("COLUMNAS CON MAS NaN:")
print("-" * 70)
nan_by_col = df_5min_consolidated[data_cols].isna().sum().sort_values(ascending=False)
for col, cnt in nan_by_col.head(10).items():
    pct = cnt / len(df_5min_consolidated) * 100
    print(f"   {col}: {cnt:,} NaN ({pct:.1f}%)")
