"""
SCRIPT 05 - Generación de Datasets RL en 15 minutos para USD/COP
=================================================================
Version 1.0 - Resampleo de 5min a 15min

Genera datasets de 15 minutos a partir de los datasets de 5 minutos existentes.

JUSTIFICACIÓN (basado en investigación 167 papers 2018-2025):
- 5min tiene SNR ~1:100 (señal muy débil vs ruido)
- 15min mejora SNR a ~1:33 (3x mejor)
- Proyecto D3F4LT4ST logró Sharpe 3.26 val / 1.49 OOS con 15min
- Menos overtrading, costos reducidos

HORARIO DE MERCADO USDCOP:
- 8:00am - 12:55pm COT (Colombia Time, UTC-5)
- 13:00 - 17:55 UTC
- Lunes a Viernes
- Sin festivos colombianos

BARRAS POR DÍA:
- 5min:  60 barras/día (8:00-12:55, cada 5min)
- 15min: 20 barras/día (8:00-12:45, cada 15min)

Author: Claude Code
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# Intentar importar festivos colombianos
try:
    from colombian_holidays import is_holiday_date
    HAS_COLOMBIAN_HOLIDAYS = True
except ImportError:
    HAS_COLOMBIAN_HOLIDAYS = False
    print("WARNING: colombian_holidays no instalado. Usando lista manual de festivos.")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_PATH = Path(__file__).parent  # 06_rl_dataset_builder/
PIPELINE_PATH = BASE_PATH.parent   # pipeline/

INPUT_5MIN = PIPELINE_PATH / "07_output" / "datasets_5min"
OUTPUT_15MIN = PIPELINE_PATH / "07_output" / "datasets_15min"
OUTPUT_15MIN.mkdir(parents=True, exist_ok=True)

# Horario de mercado (en UTC)
MARKET_OPEN_UTC = time(13, 0)   # 8:00 COT
MARKET_CLOSE_UTC = time(17, 55) # 12:55 COT

# Festivos colombianos 2020-2025 (si no hay librería)
COLOMBIAN_HOLIDAYS_MANUAL = [
    # 2020
    '2020-01-01', '2020-01-06', '2020-03-23', '2020-04-09', '2020-04-10',
    '2020-05-01', '2020-05-25', '2020-06-15', '2020-06-22', '2020-06-29',
    '2020-07-20', '2020-08-07', '2020-08-17', '2020-10-12', '2020-11-02',
    '2020-11-16', '2020-12-08', '2020-12-25',
    # 2021
    '2021-01-01', '2021-01-11', '2021-03-22', '2021-04-01', '2021-04-02',
    '2021-05-01', '2021-05-17', '2021-06-07', '2021-06-14', '2021-07-05',
    '2021-07-20', '2021-08-07', '2021-08-16', '2021-10-18', '2021-11-01',
    '2021-11-15', '2021-12-08', '2021-12-25',
    # 2022
    '2022-01-01', '2022-01-10', '2022-03-21', '2022-04-14', '2022-04-15',
    '2022-05-01', '2022-05-30', '2022-06-20', '2022-06-27', '2022-07-04',
    '2022-07-20', '2022-08-07', '2022-08-15', '2022-10-17', '2022-11-07',
    '2022-11-14', '2022-12-08', '2022-12-25',
    # 2023
    '2023-01-01', '2023-01-09', '2023-03-20', '2023-04-06', '2023-04-07',
    '2023-05-01', '2023-05-22', '2023-06-12', '2023-06-19', '2023-07-03',
    '2023-07-20', '2023-08-07', '2023-08-21', '2023-10-16', '2023-11-06',
    '2023-11-13', '2023-12-08', '2023-12-25',
    # 2024
    '2024-01-01', '2024-01-08', '2024-03-25', '2024-03-28', '2024-03-29',
    '2024-05-01', '2024-05-13', '2024-06-03', '2024-06-10', '2024-07-01',
    '2024-07-20', '2024-08-07', '2024-08-19', '2024-10-14', '2024-11-04',
    '2024-11-11', '2024-12-08', '2024-12-25',
    # 2025
    '2025-01-01', '2025-01-06', '2025-03-24', '2025-04-17', '2025-04-18',
    '2025-05-01', '2025-06-02', '2025-06-23', '2025-06-30', '2025-07-20',
    '2025-08-07', '2025-08-18', '2025-10-13', '2025-11-03', '2025-11-17',
    '2025-12-08', '2025-12-25',
]
COLOMBIAN_HOLIDAYS_SET = set(pd.to_datetime(COLOMBIAN_HOLIDAYS_MANUAL).date)


def is_colombian_holiday(dt):
    """Verificar si una fecha es festivo colombiano."""
    if HAS_COLOMBIAN_HOLIDAYS:
        return is_holiday_date(dt.date() if hasattr(dt, 'date') else dt)
    else:
        date_only = dt.date() if hasattr(dt, 'date') else dt
        return date_only in COLOMBIAN_HOLIDAYS_SET


def is_market_hours(dt):
    """
    Verificar si un timestamp está en horario de mercado.

    Horario: L-V, 8:00-12:55 COT (13:00-17:55 UTC)
    """
    # Verificar día de la semana (0=Lunes, 6=Domingo)
    if dt.weekday() >= 5:  # Sábado o Domingo
        return False

    # Verificar festivos
    if is_colombian_holiday(dt):
        return False

    # Verificar hora (asumiendo UTC)
    t = dt.time()
    return MARKET_OPEN_UTC <= t <= MARKET_CLOSE_UTC


def resample_ohlc(df_5min, freq='15min'):
    """
    Resamplear OHLC de 5min a frecuencia mayor.

    Reglas de agregación:
    - open: primer valor del período
    - high: máximo del período
    - low: mínimo del período
    - close: último valor del período
    """
    df = df_5min.copy()

    # Asegurar que timestamp es índice datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    # Definir reglas de agregación
    ohlc_cols = ['open', 'high', 'low', 'close']
    agg_rules = {}

    for col in df.columns:
        if col == 'open':
            agg_rules[col] = 'first'
        elif col == 'high':
            agg_rules[col] = 'max'
        elif col == 'low':
            agg_rules[col] = 'min'
        elif col == 'close':
            agg_rules[col] = 'last'
        elif col == 'volume':
            agg_rules[col] = 'sum'
        elif 'log_ret' in col or 'return' in col.lower():
            # Los retornos se suman (log returns son aditivos)
            agg_rules[col] = 'sum'
        else:
            # Para features: usar último valor (más reciente)
            agg_rules[col] = 'last'

    # Resamplear
    df_resampled = df.resample(freq).agg(agg_rules)

    # Eliminar filas con NaN en close (períodos sin datos)
    df_resampled = df_resampled.dropna(subset=['close'])

    # Reset index para tener timestamp como columna
    df_resampled = df_resampled.reset_index()
    df_resampled = df_resampled.rename(columns={'index': 'timestamp'})

    return df_resampled


def recalculate_technical_indicators(df, period_multiplier=3):
    """
    Recalcular indicadores técnicos para la nueva frecuencia.

    Como pasamos de 5min a 15min (3x), los períodos de ventana
    se dividen por 3 para mantener el mismo horizonte temporal.

    Ejemplo: RSI_9 en 5min = 45min lookback
             RSI_3 en 15min = 45min lookback (mismo horizonte)
    """
    df = df.copy()

    # RSI ajustado (período / 3, mínimo 3)
    if 'close' in df.columns:
        rsi_period = max(3, 9 // period_multiplier)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_9'] = 100 - (100 / (1 + rs))

    # ATR ajustado
    if all(c in df.columns for c in ['high', 'low', 'close']):
        atr_period = max(3, 14 // period_multiplier)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        df['atr_pct'] = atr / df['close'] * 100

    # Bollinger Bands position
    if 'close' in df.columns:
        bb_period = max(5, 20 // period_multiplier)
        sma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        df['bb_position'] = (df['close'] - lower) / (upper - lower + 1e-10)
        df['bb_position'] = df['bb_position'].clip(0, 1)

    # ADX simplificado
    if all(c in df.columns for c in ['high', 'low', 'close']):
        adx_period = max(3, 14 // period_multiplier)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_smooth = tr.rolling(adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(adx_period).mean() / (atr_smooth + 1e-10))
        minus_di = 100 * (minus_dm.rolling(adx_period).mean() / (atr_smooth + 1e-10))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx_14'] = dx.rolling(adx_period).mean()

    # Recalcular log returns para 15min
    if 'close' in df.columns:
        df['log_ret_15m'] = np.log(df['close'] / df['close'].shift(1))
        df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(4))   # 4 x 15min = 1h
        df['log_ret_4h'] = np.log(df['close'] / df['close'].shift(16))  # 16 x 15min = 4h

    # Hour sin/cos para 15min
    if 'timestamp' in df.columns:
        hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    return df


def filter_market_hours(df):
    """Filtrar solo horas de mercado."""
    df = df.copy()

    if 'timestamp' not in df.columns:
        print("WARNING: No timestamp column found")
        return df

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filtrar por horario de mercado
    mask = df['timestamp'].apply(is_market_hours)
    df_filtered = df[mask].copy()

    print(f"  Filtrado: {len(df)} -> {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}%)")

    return df_filtered


def convert_5min_to_15min(input_file, output_file, recalc_indicators=True):
    """
    Convertir un dataset de 5min a 15min.

    Args:
        input_file: Path al CSV de 5min
        output_file: Path de salida para 15min
        recalc_indicators: Si recalcular indicadores técnicos
    """
    print(f"\nProcesando: {input_file.name}")

    # Cargar datos
    df = pd.read_csv(input_file)
    print(f"  Input: {len(df)} rows, {len(df.columns)} columns")

    # Resamplear a 15min
    df_15min = resample_ohlc(df, freq='15min')
    print(f"  Después resample: {len(df_15min)} rows")

    # Filtrar horario de mercado
    df_15min = filter_market_hours(df_15min)

    # Recalcular indicadores si es necesario
    if recalc_indicators:
        df_15min = recalculate_technical_indicators(df_15min)

    # Limpiar NaN del warmup
    initial_len = len(df_15min)
    df_15min = df_15min.dropna()
    print(f"  Después dropna: {len(df_15min)} rows (eliminados {initial_len - len(df_15min)} warmup)")

    # Estadísticas
    if 'timestamp' in df_15min.columns:
        dates = pd.to_datetime(df_15min['timestamp'])
        n_days = dates.dt.date.nunique()
        bars_per_day = len(df_15min) / n_days if n_days > 0 else 0
        print(f"  Días de trading: {n_days}")
        print(f"  Barras por día: {bars_per_day:.1f}")
        print(f"  Rango: {dates.min()} a {dates.max()}")

    # Guardar
    df_15min.to_csv(output_file, index=False)
    print(f"  Output: {output_file}")

    return df_15min


def main():
    """Convertir todos los datasets de 5min a 15min."""
    print("=" * 70)
    print("GENERACIÓN DE DATASETS 15 MINUTOS PARA USD/COP RL")
    print("=" * 70)
    print(f"\nInput:  {INPUT_5MIN}")
    print(f"Output: {OUTPUT_15MIN}")

    # Datasets a convertir (los más importantes)
    datasets_to_convert = [
        'RL_DS1_MINIMAL.csv',
        'RL_DS3_MACRO_CORE.csv',
        'RL_DS5_MULTIFREQ_28F.csv',
        'RL_DS12_V17_FINAL.csv',
    ]

    # Buscar todos los datasets disponibles
    available = list(INPUT_5MIN.glob('RL_DS*.csv'))
    print(f"\nDatasets disponibles: {len(available)}")

    converted = []

    for ds_name in datasets_to_convert:
        input_file = INPUT_5MIN / ds_name

        if not input_file.exists():
            print(f"\nWARNING: {ds_name} no encontrado, saltando...")
            continue

        # Nombre de salida: agregar _15MIN
        output_name = ds_name.replace('.csv', '_15MIN.csv')
        output_file = OUTPUT_15MIN / output_name

        try:
            df = convert_5min_to_15min(input_file, output_file)
            converted.append({
                'name': output_name,
                'rows': len(df),
                'cols': len(df.columns),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE CONVERSIÓN")
    print("=" * 70)

    for ds in converted:
        print(f"  {ds['name']}: {ds['rows']:,} rows, {ds['cols']} cols")

    print(f"\nTotal datasets convertidos: {len(converted)}")
    print(f"Ubicación: {OUTPUT_15MIN}")

    # Crear archivo de metadatos
    metadata = {
        'conversion_date': datetime.now().isoformat(),
        'source_frequency': '5min',
        'target_frequency': '15min',
        'market_hours': '08:00-12:55 COT (13:00-17:55 UTC)',
        'bars_per_day': 20,
        'datasets': converted,
    }

    import json
    with open(OUTPUT_15MIN / 'METADATA.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n¡Conversión completada!")
    print("\nPara usar en training V19:")
    print(f"  python run_training.py --data {OUTPUT_15MIN / 'RL_DS3_MACRO_CORE_15MIN.csv'}")


if __name__ == '__main__':
    main()
