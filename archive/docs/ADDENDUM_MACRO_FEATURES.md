# 📊 ADDENDUM: MACRO FEATURES IMPLEMENTATION
## Pipeline Completo L0 → L3 → L4

**Versión:** 1.0
**Fecha:** 2025-11-05
**Complemento a:** PLAN_ESTRATEGICO_MEJORAS_RL.md

---

## 🎯 OBJETIVO

Implementar integración completa de features macro (WTI Crude Oil, DXY Dollar Index) desde la capa L0 hasta L4, con fallback manual si TwelveData API no tiene los datos disponibles.

---

## 📋 TABLA DE CONTENIDOS

1. [Verificación TwelveData API](#verificacion-twelvedata)
2. [L0: Creación de Pipeline Macro](#l0-pipeline-macro)
3. [PostgreSQL: Schema Macro](#postgresql-schema)
4. [Fallback Manual: Investing.com](#fallback-manual)
5. [L3: Integration en Feature Engineering](#l3-integration)
6. [L4: Expansion de Observation Space](#l4-expansion)
7. [Testing y Validation](#testing)

---

## 🔍 1. VERIFICACIÓN TWELVEDATA API {#verificacion-twelvedata}

### **1.1 Endpoints Disponibles**

TwelveData API soporta commodities y forex indexes:

```bash
# Test WTI Crude Oil
curl "https://api.twelvedata.com/time_series?symbol=CL&interval=5min&apikey=YOUR_KEY&outputsize=100"

# Test DXY Dollar Index
curl "https://api.twelvedata.com/time_series?symbol=DXY&interval=5min&apikey=YOUR_KEY&outputsize=100"
```

**Symbols correctos:**
- **WTI Crude Oil:** `CL` (NYMEX Crude Oil Futures)
- **DXY Dollar Index:** `DXY` (US Dollar Index)

**Verificación manual:**
```python
import requests

API_KEY = os.getenv('API_KEY_G1_1')  # Usar una de tus keys existentes

# Test WTI
response = requests.get(
    "https://api.twelvedata.com/time_series",
    params={
        'symbol': 'CL',
        'interval': '5min',
        'apikey': API_KEY,
        'outputsize': 10
    }
)

if response.status_code == 200:
    data = response.json()
    if 'values' in data:
        print("✅ WTI available on TwelveData")
    elif 'code' in data and data['code'] == 400:
        print("❌ WTI NOT available - use fallback")
else:
    print(f"⚠️ API error: {response.status_code}")

# Repeat for DXY
```

**Resultado esperado:**
- ✅ Si `values` in response: TwelveData tiene el dato → Usar OPCIÓN A
- ❌ Si `code: 400` o error: TwelveData NO tiene el dato → Usar OPCIÓN B (fallback manual)

---

## 🏗️ 2. L0: CREACIÓN DE PIPELINE MACRO {#l0-pipeline-macro}

### **2.1 Nuevo DAG: `usdcop_m5__01b_l0_macro_acquire.py`**

**Ubicación:** `airflow/dags/usdcop_m5__01b_l0_macro_acquire.py`

**Descripción:**
Pipeline L0 dedicado para adquisición de datos macro (WTI, DXY) desde TwelveData API, con almacenamiento en PostgreSQL y MinIO.

#### **Estructura del DAG**

```python
"""
DAG: usdcop_m5__01b_l0_macro_acquire
====================================
Layer: L0 - MACRO DATA ACQUISITION

🎯 PROPÓSITO:
- Adquirir datos de WTI Crude Oil y DXY Dollar Index
- Resample a 5min para alinear con USD/COP
- Insertar en PostgreSQL tabla macro_ohlcv
- Exportar a MinIO para consumo por L3

📊 FUENTES:
- Primaria: TwelveData API (symbols: CL, DXY)
- Fallback: CSV manual desde Investing.com

⏱️ FRECUENCIA:
- Diaria a las 23:00 UTC (datos intraday completos)
- Backfill automático si gaps detectados

🔄 OUTPUT:
- PostgreSQL: tabla macro_ohlcv
- MinIO: bucket 00-raw-macro-marketdata
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import pandas as pd
import numpy as np
import requests
import logging
import psycopg2
from psycopg2.extras import execute_values
import pytz
import io

# Configuration
DAG_ID = 'usdcop_m5__01b_l0_macro_acquire'
BUCKET_OUTPUT = '00-raw-macro-marketdata'

POSTGRES_CONFIG = {
    'host': 'usdcop-postgres-timescale',
    'port': 5432,
    'database': 'usdcop_trading',
    'user': 'admin',
    'password': 'admin123'
}

# Symbols to fetch
MACRO_SYMBOLS = {
    'WTI': 'CL',      # Crude Oil WTI
    'DXY': 'DXY'      # US Dollar Index
}

def get_postgres_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(**POSTGRES_CONFIG)

def check_existing_macro_data(**context):
    """Check what macro data exists and determine gaps"""

    logging.info("🔍 Checking existing macro data...")

    gap_info = {
        'WTI': {'has_data': False, 'last_date': None},
        'DXY': {'has_data': False, 'last_date': None},
        'fetch_ranges': []
    }

    conn = get_postgres_connection()

    for macro_name, symbol in MACRO_SYMBOLS.items():
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_records,
                    MAX(time) as last_date
                FROM macro_ohlcv
                WHERE symbol = %s
            """, (symbol,))

            result = cursor.fetchone()
            total_records, last_date = result

            gap_info[macro_name]['has_data'] = (total_records or 0) > 0
            gap_info[macro_name]['last_date'] = last_date

            logging.info(f"📊 {macro_name}: {total_records:,} records, last: {last_date}")

    conn.close()

    # Determine fetch strategy
    current_time = datetime.now(pytz.UTC)

    for macro_name, info in gap_info.items():
        if not info['has_data']:
            # No data - fetch last 30 days
            start_date = current_time - timedelta(days=30)
            gap_info['fetch_ranges'].append({
                'symbol': macro_name,
                'start': start_date.isoformat(),
                'end': current_time.isoformat(),
                'mode': 'initial'
            })
        else:
            # Has data - incremental from last_date
            last_date = info['last_date']
            if hasattr(last_date, 'astimezone'):
                last_date_aware = last_date.astimezone(pytz.UTC)
            else:
                last_date_aware = last_date.replace(tzinfo=pytz.UTC)

            hours_gap = (current_time - last_date_aware).total_seconds() / 3600

            if hours_gap > 6:  # More than 6 hours gap
                gap_info['fetch_ranges'].append({
                    'symbol': macro_name,
                    'start': last_date_aware.isoformat(),
                    'end': current_time.isoformat(),
                    'mode': 'incremental'
                })

    context['ti'].xcom_push(key='gap_info', value=gap_info)

    logging.info(f"📦 Will fetch {len(gap_info['fetch_ranges'])} symbol ranges")
    return gap_info

def fetch_macro_data_twelvedata(**context):
    """Fetch macro data from TwelveData API"""

    gap_info = context['ti'].xcom_pull(key='gap_info')

    if not gap_info['fetch_ranges']:
        logging.info("✅ No gaps to fill")
        return {'status': 'up_to_date', 'records': 0}

    # Load API keys
    api_keys = []
    for group in ['G1', 'G2', 'G3']:
        for i in range(1, 9):
            key = os.environ.get(f'API_KEY_{group}_{i}')
            if key and key.strip():
                api_keys.append(key.strip())

    if not api_keys:
        raise ValueError("No API keys available")

    logging.info(f"✅ Loaded {len(api_keys)} API keys")

    all_data = []
    current_key_idx = 0

    for fetch_range in gap_info['fetch_ranges']:
        symbol_name = fetch_range['symbol']
        symbol_code = MACRO_SYMBOLS[symbol_name]
        start_dt = pd.to_datetime(fetch_range['start'])
        end_dt = pd.to_datetime(fetch_range['end'])

        logging.info(f"📥 Fetching {symbol_name} ({symbol_code}): {start_dt.date()} → {end_dt.date()}")

        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol_code,
                'interval': '5min',  # Native 5min data
                'apikey': api_keys[current_key_idx],
                'start_date': start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timezone': 'UTC',
                'outputsize': 5000,
                'format': 'JSON'
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if 'values' in data and data['values']:
                    df = pd.DataFrame(data['values'])
                    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                    df = df.rename(columns={'datetime': 'time'})

                    # Convert OHLC
                    for col in ['open', 'high', 'low', 'close']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
                    df['symbol'] = symbol_code
                    df['source'] = 'twelvedata'

                    all_data.append(df)

                    logging.info(f"✅ Downloaded {len(df)} bars for {symbol_name}")

                elif 'code' in data:
                    if data['code'] == 400:
                        logging.error(f"❌ Symbol {symbol_code} not available on TwelveData")
                        logging.error(f"   Use FALLBACK: Manual CSV upload required")
                    elif data['code'] == 429:
                        logging.warning(f"⚠️ Rate limit - rotating API key")
                        current_key_idx = (current_key_idx + 1) % len(api_keys)

            else:
                logging.error(f"❌ API error: {response.status_code}")

        except Exception as e:
            logging.error(f"❌ Error fetching {symbol_name}: {e}")
            continue

    # Save to PostgreSQL
    if all_data:
        df_combined = pd.concat(all_data, ignore_index=True)
        df_final = df_combined.sort_values('time').drop_duplicates(subset=['time', 'symbol']).reset_index(drop=True)

        logging.info(f"💾 Inserting {len(df_final)} rows to PostgreSQL...")

        insert_to_postgres_macro(df_final)

        # Export to MinIO
        export_to_minio_macro(df_final, context)

        return {
            'status': 'success',
            'records': len(df_final),
            'symbols_fetched': df_final['symbol'].unique().tolist()
        }

    return {'status': 'no_data', 'records': 0}

def insert_to_postgres_macro(df):
    """Insert macro data to PostgreSQL"""

    conn = get_postgres_connection()

    data_tuples = []
    for _, row in df.iterrows():
        data_tuples.append((
            row['time'],
            row['symbol'],
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            int(row.get('volume', 0)),
            row.get('source', 'twelvedata')
        ))

    insert_sql = """
    INSERT INTO macro_ohlcv (time, symbol, open, high, low, close, volume, source)
    VALUES %s
    ON CONFLICT (time, symbol) DO UPDATE SET
        open = EXCLUDED.open,
        high = GREATEST(macro_ohlcv.high, EXCLUDED.high),
        low = LEAST(macro_ohlcv.low, EXCLUDED.low),
        close = EXCLUDED.close,
        volume = macro_ohlcv.volume + EXCLUDED.volume,
        updated_at = NOW()
    """

    with conn.cursor() as cursor:
        execute_values(cursor, insert_sql, data_tuples, page_size=500)
        conn.commit()

    conn.close()

    logging.info(f"✅ Inserted {len(data_tuples)} macro records")

def export_to_minio_macro(df, context):
    """Export macro data to MinIO"""

    s3_hook = S3Hook(aws_conn_id='minio_conn')

    latest_date = df['time'].max().strftime('%Y-%m-%d')
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save parquet
    output_key = f"{DAG_ID}/date={latest_date}/run_id={run_id}/macro_data.parquet"

    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
    parquet_buffer.seek(0)

    s3_hook.load_bytes(
        bytes_data=parquet_buffer.read(),
        bucket_name=BUCKET_OUTPUT,
        key=output_key,
        replace=True
    )

    logging.info(f"✅ Exported to MinIO: s3://{BUCKET_OUTPUT}/{output_key}")

# DAG Definition
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='L0 Macro Acquire - Fetch WTI and DXY from TwelveData',
    schedule_interval='0 23 * * *',  # Daily at 23:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=['l0', 'macro', 'twelvedata']
)

with dag:
    check_gaps_task = PythonOperator(
        task_id='check_existing_macro_data',
        python_callable=check_existing_macro_data,
        provide_context=True
    )

    fetch_task = PythonOperator(
        task_id='fetch_macro_data',
        python_callable=fetch_macro_data_twelvedata,
        provide_context=True
    )

    check_gaps_task >> fetch_task
```

---

## 🗄️ 3. POSTGRESQL: SCHEMA MACRO {#postgresql-schema}

### **3.1 Nueva Tabla: `macro_ohlcv`**

**Archivo:** `init-scripts/03-create-macro-schema.sql`

```sql
-- =====================================================
-- MACRO DATA SCHEMA
-- Table for WTI, DXY, and other macro indicators
-- =====================================================

-- Create macro_ohlcv table
CREATE TABLE IF NOT EXISTS macro_ohlcv (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    symbol TEXT NOT NULL,  -- 'CL' (WTI), 'DXY' (Dollar Index)
    open DECIMAL(12,6) NOT NULL,
    high DECIMAL(12,6) NOT NULL,
    low DECIMAL(12,6) NOT NULL,
    close DECIMAL(12,6) NOT NULL,
    volume BIGINT DEFAULT 0,
    source TEXT DEFAULT 'twelvedata',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (time, symbol),
    -- Data quality constraints
    CONSTRAINT chk_macro_prices_positive CHECK (open > 0 AND high > 0 AND low > 0 AND close > 0),
    CONSTRAINT chk_macro_high_gte_low CHECK (high >= low),
    CONSTRAINT chk_macro_high_gte_open CHECK (high >= open),
    CONSTRAINT chk_macro_high_gte_close CHECK (high >= close),
    CONSTRAINT chk_macro_low_lte_open CHECK (low <= open),
    CONSTRAINT chk_macro_low_lte_close CHECK (low <= close),
    CONSTRAINT chk_macro_volume_non_negative CHECK (volume >= 0)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('macro_ohlcv', 'time', if_not_exists => TRUE, migrate_data => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_macro_ohlcv_symbol_time ON macro_ohlcv (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_macro_ohlcv_time ON macro_ohlcv (time DESC);
CREATE INDEX IF NOT EXISTS idx_macro_ohlcv_source ON macro_ohlcv (source);

-- Create view for latest macro data
CREATE OR REPLACE VIEW latest_macro AS
SELECT DISTINCT ON (symbol)
    symbol,
    time as timestamp,
    open,
    high,
    low,
    close,
    volume,
    source
FROM macro_ohlcv
ORDER BY symbol, time DESC;

-- Permissions
GRANT ALL PRIVILEGES ON macro_ohlcv TO admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON macro_ohlcv TO airflow;
```

### **3.2 Migration Script para Actualización**

**Archivo:** `postgres/migrations/004_add_macro_table.sql`

```sql
-- Migration 004: Add Macro OHLCV Table
-- Run this if upgrading existing database

-- Check if table exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'macro_ohlcv') THEN
        RAISE NOTICE 'Creating macro_ohlcv table...';

        CREATE TABLE macro_ohlcv (
            time TIMESTAMP WITH TIME ZONE NOT NULL,
            symbol TEXT NOT NULL,
            open DECIMAL(12,6) NOT NULL,
            high DECIMAL(12,6) NOT NULL,
            low DECIMAL(12,6) NOT NULL,
            close DECIMAL(12,6) NOT NULL,
            volume BIGINT DEFAULT 0,
            source TEXT DEFAULT 'twelvedata',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            PRIMARY KEY (time, symbol),
            CONSTRAINT chk_macro_prices_positive CHECK (open > 0 AND high > 0 AND low > 0 AND close > 0),
            CONSTRAINT chk_macro_high_gte_low CHECK (high >= low),
            CONSTRAINT chk_macro_high_gte_open CHECK (high >= open),
            CONSTRAINT chk_macro_high_gte_close CHECK (high >= close),
            CONSTRAINT chk_macro_low_lte_open CHECK (low <= open),
            CONSTRAINT chk_macro_low_lte_close CHECK (low <= close),
            CONSTRAINT chk_macro_volume_non_negative CHECK (volume >= 0)
        );

        -- Create hypertable
        PERFORM create_hypertable('macro_ohlcv', 'time', if_not_exists => TRUE);

        -- Create indexes
        CREATE INDEX idx_macro_ohlcv_symbol_time ON macro_ohlcv (symbol, time DESC);
        CREATE INDEX idx_macro_ohlcv_time ON macro_ohlcv (time DESC);
        CREATE INDEX idx_macro_ohlcv_source ON macro_ohlcv (source);

        RAISE NOTICE 'macro_ohlcv table created successfully';
    ELSE
        RAISE NOTICE 'macro_ohlcv table already exists, skipping creation';
    END IF;
END
$$;
```

---

## 📥 4. FALLBACK MANUAL: INVESTING.COM {#fallback-manual}

### **4.1 Escenario: TwelveData No Tiene Los Datos**

Si TwelveData API retorna error 400 (symbol not available), usar proceso manual:

#### **Paso 1: Descargar CSV desde Investing.com**

1. Ir a https://www.investing.com/commodities/crude-oil-historical-data
2. Seleccionar rango de fechas
3. Descargar CSV
4. Repetir para DXY: https://www.investing.com/indices/usdollar-historical-data

#### **Paso 2: Formato del CSV**

**Estructura esperada:**
```csv
Date,Price,Open,High,Low,Vol.,Change %
01/01/2025,75.23,74.89,75.45,74.67,125.43K,+0.45%
```

#### **Paso 3: Script de Carga Manual**

**Archivo:** `scripts/upload_macro_manual.py`

```python
"""
Manual Macro Data Upload Script
================================
Use when TwelveData API doesn't have WTI/DXY data

Usage:
    python scripts/upload_macro_manual.py --symbol WTI --file data/wti_2025.csv
    python scripts/upload_macro_manual.py --symbol DXY --file data/dxy_2025.csv
"""

import argparse
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import pytz

POSTGRES_CONFIG = {
    'host': 'localhost',  # or usdcop-postgres-timescale if inside Docker
    'port': 5432,
    'database': 'usdcop_trading',
    'user': 'admin',
    'password': 'admin123'
}

SYMBOL_MAP = {
    'WTI': 'CL',
    'DXY': 'DXY'
}

def parse_investing_csv(file_path, symbol_name):
    """Parse CSV from Investing.com"""

    df = pd.read_csv(file_path)

    # Rename columns to standard format
    df = df.rename(columns={
        'Date': 'time',
        'Price': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Vol.': 'volume'
    })

    # Parse date (Investing.com format: MM/DD/YYYY)
    df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y')
    df['time'] = df['time'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')

    # Clean volume (remove 'K', 'M' suffixes)
    if 'volume' in df.columns:
        df['volume'] = df['volume'].str.replace('K', '000').str.replace('M', '000000')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
    else:
        df['volume'] = 0

    # Add metadata
    df['symbol'] = SYMBOL_MAP[symbol_name]
    df['source'] = 'manual_csv'

    # Keep only OHLCV columns
    df = df[['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source']]

    # Resample to 5min (forward fill daily data across trading hours)
    df = df.set_index('time')
    df_resampled = df.resample('5T').ffill()  # Forward fill
    df_resampled = df_resampled.reset_index()

    return df_resampled

def upload_to_postgres(df):
    """Upload data to PostgreSQL"""

    conn = psycopg2.connect(**POSTGRES_CONFIG)

    data_tuples = []
    for _, row in df.iterrows():
        data_tuples.append((
            row['time'],
            row['symbol'],
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            int(row['volume']),
            row['source']
        ))

    insert_sql = """
    INSERT INTO macro_ohlcv (time, symbol, open, high, low, close, volume, source)
    VALUES %s
    ON CONFLICT (time, symbol) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = macro_ohlcv.volume + EXCLUDED.volume,
        updated_at = NOW()
    """

    with conn.cursor() as cursor:
        execute_values(cursor, insert_sql, data_tuples, page_size=500)
        conn.commit()

    conn.close()

    print(f"✅ Uploaded {len(data_tuples)} rows to PostgreSQL")

def main():
    parser = argparse.ArgumentParser(description='Upload manual macro data from Investing.com CSV')
    parser.add_argument('--symbol', required=True, choices=['WTI', 'DXY'], help='Symbol name')
    parser.add_argument('--file', required=True, help='Path to CSV file from Investing.com')
    args = parser.parse_args()

    print(f"📥 Loading {args.symbol} data from {args.file}...")

    df = parse_investing_csv(args.file, args.symbol)

    print(f"📊 Parsed {len(df)} rows")
    print(f"📅 Date range: {df['time'].min()} → {df['time'].max()}")

    upload_to_postgres(df)

    print("✅ Manual upload complete!")

if __name__ == '__main__':
    main()
```

#### **Paso 4: Formulario Excel para Actualización Diaria**

**Archivo:** `data/macro_daily_update_template.xlsx`

**Estructura:**

| Date       | WTI_Close | DXY_Close |
|------------|-----------|-----------|
| 2025-01-15 | 75.23     | 102.45    |
| 2025-01-16 | 75.67     | 102.89    |

**Script de procesamiento:**

```python
# scripts/process_daily_excel.py
import pandas as pd

def process_daily_excel(excel_path):
    """Process daily update Excel"""

    df = pd.read_excel(excel_path)

    # Convert to long format
    df_wti = df[['Date', 'WTI_Close']].copy()
    df_wti.columns = ['time', 'close']
    df_wti['symbol'] = 'CL'
    df_wti['open'] = df_wti['close']  # Assuming daily data
    df_wti['high'] = df_wti['close']
    df_wti['low'] = df_wti['close']
    df_wti['volume'] = 0
    df_wti['source'] = 'manual_excel'

    df_dxy = df[['Date', 'DXY_Close']].copy()
    df_dxy.columns = ['time', 'close']
    df_dxy['symbol'] = 'DXY'
    df_dxy['open'] = df_dxy['close']
    df_dxy['high'] = df_dxy['close']
    df_dxy['low'] = df_dxy['close']
    df_dxy['volume'] = 0
    df_dxy['source'] = 'manual_excel'

    df_combined = pd.concat([df_wti, df_dxy], ignore_index=True)

    # Upload to PostgreSQL
    upload_to_postgres(df_combined)

    print("✅ Daily Excel processed and uploaded")

if __name__ == '__main__':
    process_daily_excel('data/macro_daily_update_template.xlsx')
```

---

## 🔧 5. L3: INTEGRATION EN FEATURE ENGINEERING {#l3-integration}

### **5.1 Modificar `usdcop_m5__04_l3_feature.py`**

**Añadir al inicio del archivo:**

```python
# Macro data configuration
MACRO_ENABLED = True  # Toggle to enable/disable macro features
MACRO_SYMBOLS = {
    'WTI': 'CL',
    'DXY': 'DXY'
}
```

**Nueva función: `fetch_macro_data()`**

```python
def fetch_macro_data(start_date, end_date):
    """
    Fetch macro data from PostgreSQL for the given date range

    Args:
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)

    Returns:
        DataFrame with columns: time_utc, wti, dxy
    """

    if not MACRO_ENABLED:
        logging.info("⏭️ Macro features disabled, skipping")
        return None

    logging.info("📥 Fetching macro data from PostgreSQL...")

    conn = get_postgres_connection()

    query = """
    SELECT
        time as time_utc,
        symbol,
        close
    FROM macro_ohlcv
    WHERE time >= %s AND time <= %s
    AND symbol IN ('CL', 'DXY')
    ORDER BY time, symbol
    """

    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()

    if df.empty:
        logging.warning("⚠️ No macro data found for date range")
        return None

    # Pivot to wide format
    df_pivot = df.pivot(index='time_utc', columns='symbol', values='close')
    df_pivot = df_pivot.rename(columns={'CL': 'wti', 'DXY': 'dxy'})
    df_pivot = df_pivot.reset_index()

    # Ensure timezone-aware
    df_pivot['time_utc'] = pd.to_datetime(df_pivot['time_utc'], utc=True)

    logging.info(f"✅ Loaded {len(df_pivot)} macro bars (WTI: {'wti' in df_pivot.columns}, DXY: {'dxy' in df_pivot.columns})")

    return df_pivot
```

**Nueva función: `calculate_macro_features()`**

```python
def calculate_macro_features(df_usdcop, df_macro):
    """
    Calculate macro-derived features and merge with USD/COP data

    Args:
        df_usdcop: USD/COP OHLCV dataframe with time_utc
        df_macro: Macro dataframe with time_utc, wti, dxy

    Returns:
        DataFrame with added macro features
    """

    if df_macro is None or df_macro.empty:
        logging.warning("⚠️ No macro data available, skipping macro features")
        return df_usdcop

    logging.info("🧮 Calculating macro features...")

    # Merge with USDCOP data (left join, forward fill for missing values)
    df = df_usdcop.merge(df_macro, on='time_utc', how='left')

    # Forward fill macro data (commodities update less frequently than FX)
    df['wti'] = df['wti'].fillna(method='ffill')
    df['dxy'] = df['dxy'].fillna(method='ffill')

    # Calculate macro features

    # 1. WTI returns (5-bar and 20-bar)
    df['wti_return_5'] = df['wti'].pct_change(5)
    df['wti_return_20'] = df['wti'].pct_change(20)

    # 2. WTI z-score (60-bar rolling window)
    df['wti_zscore_60'] = (df['wti'] - df['wti'].rolling(60).mean()) / df['wti'].rolling(60).std()

    # 3. DXY returns (5-bar and 20-bar)
    df['dxy_return_5'] = df['dxy'].pct_change(5)
    df['dxy_return_20'] = df['dxy'].pct_change(20)

    # 4. DXY z-score (60-bar rolling window)
    df['dxy_zscore_60'] = (df['dxy'] - df['dxy'].rolling(60).mean()) / df['dxy'].rolling(60).std()

    # 5. COP-WTI correlation (60-bar rolling)
    df['cop_wti_corr_60'] = df['close'].rolling(60).corr(df['wti'])

    # Apply causality shift (5 bars = 25 minutes)
    macro_feature_cols = [
        'wti_return_5', 'wti_return_20', 'wti_zscore_60',
        'dxy_return_5', 'dxy_return_20', 'dxy_zscore_60',
        'cop_wti_corr_60'
    ]

    for col in macro_feature_cols:
        df[col] = df[col].shift(5)  # Causality: use 5-bar lag

    # Fill NaN with 0 (conservative - no macro signal)
    df[macro_feature_cols] = df[macro_feature_cols].fillna(0)

    logging.info(f"✅ Added {len(macro_feature_cols)} macro features")

    return df
```

**Integrar en `main_l3_task()`:**

```python
def main_l3_task(**context):
    """Main L3 feature engineering task"""

    # ... existing code to load L2 data ...

    # Fetch macro data
    start_date = df['time_utc'].min()
    end_date = df['time_utc'].max()

    df_macro = fetch_macro_data(start_date, end_date)

    # Calculate original features
    df = calculate_tier1_features(df)
    df = calculate_tier2_features(df)

    # NEW: Calculate macro features
    if df_macro is not None:
        df = calculate_macro_features(df, df_macro)

    # ... rest of the pipeline ...

    # Save to MinIO
    save_to_minio(df, context)
```

---

## 📦 6. L4: EXPANSION DE OBSERVATION SPACE {#l4-expansion}

### **6.1 Modificar `usdcop_m5__05_l4_rlready.py`**

**Actualizar OBS_MAPPING:**

```python
OBS_MAPPING = {
    # ORIGINALES (obs_00 a obs_16) - NO CAMBIAR
    'obs_00': 'hl_range_surprise',
    'obs_01': 'atr_surprise',
    'obs_02': 'body_ratio_abs',
    'obs_03': 'wick_asym_abs',
    'obs_04': 'macd_strength_abs',
    'obs_05': 'compression_ratio',
    'obs_06': 'band_cross_abs_k',
    'obs_07': 'entropy_absret_k',
    'obs_08': 'momentum_abs_norm',
    'obs_09': 'doji_freq_k',
    'obs_10': 'gap_prev_open_abs',
    'obs_11': 'rsi_dist_50',
    'obs_12': 'stoch_dist_mid',
    'obs_13': 'bb_squeeze_ratio',
    'obs_14': 'hour_sin',
    'obs_15': 'hour_cos',
    'obs_16': 'spread_proxy_bps_norm',

    # MACRO FEATURES (NUEVO) - obs_17 a obs_23
    'obs_17': 'wti_return_5',
    'obs_18': 'wti_return_20',
    'obs_19': 'wti_zscore_60',
    'obs_20': 'dxy_return_5',
    'obs_21': 'dxy_return_20',
    'obs_22': 'dxy_zscore_60',
    'obs_23': 'cop_wti_corr_60'
}
```

**Actualizar normalización:**

```python
def normalize_observations(df):
    """Normalize observations with method per group"""

    # Original features (obs_00 to obs_16): Z-score
    original_obs = [f'obs_{i:02d}' for i in range(17)]

    for obs in original_obs:
        if obs in df.columns:
            rolling_mean = df[obs].rolling(60, min_periods=1).mean()
            rolling_std = df[obs].rolling(60, min_periods=1).std()
            df[f'{obs}_norm'] = (df[obs] - rolling_mean) / (rolling_std + 1e-8)
            df[f'{obs}_norm'] = df[f'{obs}_norm'].clip(-5, 5)

    # Macro features (obs_17 to obs_23): RobustScaler
    macro_obs = [f'obs_{i:02d}' for i in range(17, 24)]

    for obs in macro_obs:
        if obs in df.columns:
            rolling_median = df[obs].rolling(60, min_periods=1).median()
            rolling_q75 = df[obs].rolling(60, min_periods=1).quantile(0.75)
            rolling_q25 = df[obs].rolling(60, min_periods=1).quantile(0.25)
            rolling_iqr = rolling_q75 - rolling_q25

            df[f'{obs}_norm'] = (df[obs] - rolling_median) / (rolling_iqr + 1e-8)
            df[f'{obs}_norm'] = df[f'{obs}_norm'].clip(-5, 5)

    return df
```

---

## ✅ 7. TESTING Y VALIDATION {#testing}

### **7.1 Test PostgreSQL Connection**

```bash
# Inside Docker container
docker exec -it usdcop-postgres-timescale psql -U admin -d usdcop_trading

# Check macro table
SELECT COUNT(*), MIN(time), MAX(time) FROM macro_ohlcv WHERE symbol = 'CL';
SELECT COUNT(*), MIN(time), MAX(time) FROM macro_ohlcv WHERE symbol = 'DXY';
```

### **7.2 Test TwelveData API**

```python
# Test script: scripts/test_macro_api.py
import requests
import os

API_KEY = os.getenv('API_KEY_G1_1')

for symbol in ['CL', 'DXY']:
    response = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            'symbol': symbol,
            'interval': '5min',
            'apikey': API_KEY,
            'outputsize': 10
        }
    )

    if response.status_code == 200:
        data = response.json()
        if 'values' in data:
            print(f"✅ {symbol}: {len(data['values'])} bars available")
        else:
            print(f"❌ {symbol}: {data.get('message', 'No data')}")
    else:
        print(f"⚠️ {symbol}: HTTP {response.status_code}")
```

### **7.3 Test L3 Integration**

```python
# Run L3 DAG with macro features enabled
# Verify output has 24 observations instead of 17

df_l4 = pd.read_parquet('04-l4-ds-usdcop-rlready/latest.parquet')

print(f"Observations: {[col for col in df_l4.columns if col.startswith('obs_')]}")
assert 'obs_17' in df_l4.columns, "Macro features not found!"
assert 'obs_23' in df_l4.columns, "Macro features incomplete!"

print("✅ Macro features successfully integrated!")
```

---

## 📊 CHECKLIST DE IMPLEMENTACIÓN

### **L0 - Macro Pipeline**
- [ ] Crear `usdcop_m5__01b_l0_macro_acquire.py`
- [ ] Crear `init-scripts/03-create-macro-schema.sql`
- [ ] Ejecutar migration `004_add_macro_table.sql`
- [ ] Test TwelveData API para WTI y DXY
- [ ] Si fallback necesario: crear `scripts/upload_macro_manual.py`
- [ ] Ejecutar L0 macro DAG y verificar PostgreSQL

### **L3 - Feature Engineering**
- [ ] Añadir `fetch_macro_data()` en `usdcop_m5__04_l3_feature.py`
- [ ] Añadir `calculate_macro_features()`
- [ ] Integrar en `main_l3_task()`
- [ ] Test pipeline completo L0→L3
- [ ] Verificar output en MinIO tiene 7 columnas macro

### **L4 - Observation Mapping**
- [ ] Actualizar `OBS_MAPPING` en `usdcop_m5__05_l4_rlready.py`
- [ ] Actualizar normalización (RobustScaler para macro)
- [ ] Test pipeline completo L0→L4
- [ ] Verificar dataset final tiene obs_00 a obs_23

### **Environment Update**
- [ ] Modificar `environments.py` para detectar 24 features
- [ ] Modificar `config.py`: obs_dim = 24
- [ ] Test environment con nuevo observation space

### **Validation**
- [ ] Feature importance con 24 features
- [ ] Verificar IC < 0.30 para macro features
- [ ] Verificar no-NaN en features macro
- [ ] Run training con 24 features y comparar vs 17

---

**FIN ADDENDUM_MACRO_FEATURES.md**
