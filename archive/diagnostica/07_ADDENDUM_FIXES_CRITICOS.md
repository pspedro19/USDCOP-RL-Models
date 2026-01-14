# ADDENDUM: FIXES CRÍTICOS FALTANTES
## Complemento al Plan Maestro - 10 Expertos
## Fecha: 2026-01-08

---

## RESUMEN DE GAPS IDENTIFICADOS

| # | Issue | Severidad | Estado en Plan Original |
|---|-------|-----------|------------------------|
| 1 | Look-Ahead Bias | 🔴 CRÍTICO | NO CUBIERTO |
| 2 | Macro Scraper Fix | 🔴 CRÍTICO | Solo mencionado |
| 3 | Timezone Consistency | 🟡 ALTO | NO CUBIERTO |
| 4 | US Holidays Flag | 🟡 ALTO | NO CUBIERTO |
| 5 | Dataset Split V20 | 🟡 ALTO | No especificado |
| 6 | Entropy Regularization | 🟡 ALTO | Incompleto |
| 7 | Benchmarks | 🟢 MEDIO | NO DEFINIDO |
| 8 | Early Stopping | 🟢 MEDIO | No especificado |
| 9 | Reward V20 Math Bug | 🔴 CRÍTICO | Bug en código |

---

## FIX 1: LOOK-AHEAD BIAS (CRÍTICO)

### El Problema

```python
# CÓDIGO ACTUAL (l5_multi_model_inference.py):
def execute_inference(row):
    observation = build_observation(row)
    action = model.predict(observation)

    # ⚠️ PROBLEMA: Ejecuta al CLOSE de la barra actual
    execution_price = row['close']  # ← LOOK-AHEAD BIAS
    execute_trade(action, execution_price)
```

**¿Por qué es un problema?**
- Cuando ves el `close` de una barra, esa barra YA TERMINÓ
- En la vida real, no puedes ejecutar en un precio del pasado
- El backtest usa información que no estaría disponible en tiempo real
- **Infla artificialmente los resultados del backtest**

### La Solución

```python
# CÓDIGO CORREGIDO:
class InferenceEngine:
    def __init__(self):
        self.pending_signal = None
        self.pending_price = None

    def on_bar_close(self, current_bar):
        """Called when a bar closes - generate signal for NEXT bar."""
        observation = build_observation(current_bar)
        action = model.predict(observation)

        # Guardar señal para ejecutar en la siguiente barra
        self.pending_signal = discretize_action(action)
        self.pending_price = current_bar['close']  # Para logging

        logger.info(f"Signal generated: {self.pending_signal} at bar close {current_bar['time']}")

    def on_bar_open(self, new_bar):
        """Called when a new bar opens - execute pending signal."""
        if self.pending_signal is not None:
            # ✅ CORRECTO: Ejecutar al OPEN de la nueva barra
            execution_price = new_bar['open']

            logger.info(f"Executing {self.pending_signal} at {execution_price} (bar open {new_bar['time']})")

            execute_trade(self.pending_signal, execution_price)
            self.pending_signal = None
```

### Implementación en DAG

```python
# airflow/dags/l5_multi_model_inference.py

def run_inference_task(**context):
    """Modified inference task with look-ahead bias fix."""

    # Get last 2 bars
    bars = get_recent_bars(limit=2)

    if len(bars) < 2:
        logger.warning("Not enough bars for inference")
        return

    prev_bar = bars.iloc[0]  # Barra anterior (cerrada)
    current_bar = bars.iloc[1]  # Barra actual (abriendo)

    # Generar señal basada en barra CERRADA
    observation = observation_builder.build(prev_bar)
    action = model.predict(observation)
    signal = discretize_action(action, threshold=0.10)

    # Ejecutar al OPEN de barra ACTUAL
    execution_price = current_bar['open']

    logger.info(f"""
    Look-Ahead Bias Fix Applied:
    - Signal generated from bar: {prev_bar['time']}
    - Execution at bar open: {current_bar['time']}
    - Execution price: {execution_price}
    - Signal: {signal}
    """)

    if signal != 'HOLD':
        paper_trader.execute(signal, execution_price)
```

### Query de Validación

```sql
-- Verificar que trades se ejecutan al open, no al close
SELECT
    t.entry_time,
    t.entry_price,
    o.open as bar_open,
    o.close as bar_close,
    CASE
        WHEN ABS(t.entry_price - o.open) < ABS(t.entry_price - o.close)
        THEN 'CORRECT (used open)'
        ELSE 'INCORRECT (used close)'
    END as execution_check
FROM trades_history t
JOIN usdcop_m5_ohlcv o ON t.entry_time = o.time
ORDER BY t.entry_time DESC
LIMIT 20;
```

---

## FIX 2: MACRO SCRAPER (CRÍTICO)

### El Problema

```sql
-- Últimos 14 días de macro data
SELECT fecha, fxrt_index_dxy_usa_d_dxy as dxy
FROM macro_indicators_daily
WHERE fecha > CURRENT_DATE - 14
ORDER BY fecha DESC;

-- Resultado actual:
-- 2026-01-08: NULL ← PROBLEMA
-- 2026-01-07: NULL ← PROBLEMA
-- 2026-01-06: 104.23
-- 2026-01-05: NULL (weekend)
-- ...
```

### Diagnóstico del Scraper

```python
# scripts/diagnose_macro_scraper.py

import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_banrep_scraper():
    """Diagnose why BanRep scraper is failing."""

    issues = []

    # 1. Check Selenium/Chrome availability
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(options=options)
        logger.info("✅ Chrome/Selenium working")
        driver.quit()
    except WebDriverException as e:
        issues.append(f"Chrome/Selenium error: {e}")
        logger.error(f"❌ Chrome/Selenium failed: {e}")

    # 2. Check BanRep website accessibility
    try:
        import requests
        response = requests.get("https://www.banrep.gov.co/es/estadisticas", timeout=30)
        if response.status_code == 200:
            logger.info("✅ BanRep website accessible")
        else:
            issues.append(f"BanRep returned status {response.status_code}")
    except Exception as e:
        issues.append(f"BanRep connection error: {e}")

    # 3. Check alternative APIs
    try:
        # FRED API for DXY
        fred_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "DTWEXBGS",
            "api_key": "YOUR_FRED_API_KEY",
            "file_type": "json",
            "limit": 5
        }
        # response = requests.get(fred_url, params=params)
        logger.info("📝 FRED API available as fallback for DXY")
    except:
        pass

    return issues


if __name__ == "__main__":
    issues = diagnose_banrep_scraper()
    if issues:
        print("\n🔴 ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ All checks passed - investigate scraper logic")
```

### Scraper con Retry y Fallback

```python
# airflow/dags/utils/macro_scraper_robust.py

import time
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

class RobustMacroScraper:
    """Macro data scraper with retry logic and fallbacks."""

    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 60  # seconds
        self.fred_api_key = os.getenv('FRED_API_KEY')

    def fetch_with_retry(self, fetch_func, *args, **kwargs):
        """Execute fetch function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                result = fetch_func(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)

        logger.error(f"All {self.max_retries} attempts failed")
        return None

    def fetch_dxy(self, date: datetime) -> Optional[float]:
        """Fetch DXY with fallback to FRED API."""

        # Primary: BanRep/custom source
        dxy = self.fetch_with_retry(self._fetch_dxy_primary, date)
        if dxy is not None:
            return dxy

        # Fallback: FRED API
        logger.info("Using FRED API fallback for DXY")
        return self._fetch_dxy_fred(date)

    def _fetch_dxy_primary(self, date: datetime) -> Optional[float]:
        """Primary DXY source."""
        # Original scraper logic here
        pass

    def _fetch_dxy_fred(self, date: datetime) -> Optional[float]:
        """Fallback: FRED API for DXY."""
        if not self.fred_api_key:
            logger.warning("FRED API key not configured")
            return None

        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "DTWEXBGS",  # Trade Weighted U.S. Dollar Index
                "api_key": self.fred_api_key,
                "file_type": "json",
                "observation_start": date.strftime("%Y-%m-%d"),
                "observation_end": date.strftime("%Y-%m-%d"),
            }
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if data.get("observations"):
                value = data["observations"][0].get("value")
                if value and value != ".":
                    return float(value)
        except Exception as e:
            logger.error(f"FRED API error: {e}")

        return None

    def fetch_vix(self, date: datetime) -> Optional[float]:
        """Fetch VIX with fallback."""
        # Primary source
        vix = self.fetch_with_retry(self._fetch_vix_primary, date)
        if vix is not None:
            return vix

        # Fallback: Yahoo Finance
        return self._fetch_vix_yahoo(date)

    def _fetch_vix_yahoo(self, date: datetime) -> Optional[float]:
        """Fallback: Yahoo Finance for VIX."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(start=date, end=date + timedelta(days=1))
            if not hist.empty:
                return float(hist['Close'].iloc[0])
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
        return None

    def fill_missing_macro(self, conn, days_back: int = 7):
        """Fill missing macro data for recent days."""
        import psycopg2

        query = """
        SELECT fecha
        FROM macro_indicators_daily
        WHERE fecha > CURRENT_DATE - %s
          AND (fxrt_index_dxy_usa_d_dxy IS NULL
               OR volt_vix_usa_d_vix IS NULL)
        ORDER BY fecha
        """

        with conn.cursor() as cur:
            cur.execute(query, (days_back,))
            missing_dates = [row[0] for row in cur.fetchall()]

        logger.info(f"Found {len(missing_dates)} dates with missing macro data")

        for date in missing_dates:
            dxy = self.fetch_dxy(date)
            vix = self.fetch_vix(date)

            if dxy is not None or vix is not None:
                update_query = """
                UPDATE macro_indicators_daily
                SET
                    fxrt_index_dxy_usa_d_dxy = COALESCE(%s, fxrt_index_dxy_usa_d_dxy),
                    volt_vix_usa_d_vix = COALESCE(%s, volt_vix_usa_d_vix),
                    updated_at = NOW()
                WHERE fecha = %s
                """
                with conn.cursor() as cur:
                    cur.execute(update_query, (dxy, vix, date))
                    conn.commit()

                logger.info(f"Updated macro for {date}: DXY={dxy}, VIX={vix}")
```

### DAG para Macro con Alertas

```python
# airflow/dags/l0_macro_with_alerts.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def check_macro_nulls(**context):
    """Check for NULL macro data and alert if found."""
    import psycopg2

    conn = psycopg2.connect(os.getenv('DATABASE_URL'))

    query = """
    SELECT COUNT(*) as null_count
    FROM macro_indicators_daily
    WHERE fecha > CURRENT_DATE - 3
      AND EXTRACT(DOW FROM fecha) BETWEEN 1 AND 5  -- Weekdays only
      AND (fxrt_index_dxy_usa_d_dxy IS NULL
           OR volt_vix_usa_d_vix IS NULL)
    """

    with conn.cursor() as cur:
        cur.execute(query)
        null_count = cur.fetchone()[0]

    if null_count > 0:
        # Send alert
        alert_msg = f"""
        🚨 MACRO DATA ALERT

        {null_count} recent trading days have NULL macro data.
        This affects inference quality.

        Action: Check scraper logs and run fill_missing_macro()
        """
        logger.warning(alert_msg)
        # TODO: Send to Slack/email

        # Try to fill missing data
        from utils.macro_scraper_robust import RobustMacroScraper
        scraper = RobustMacroScraper()
        scraper.fill_missing_macro(conn, days_back=7)

    conn.close()
    return null_count
```

---

## FIX 3: TIMEZONE CONSISTENCY (ALTO)

### El Problema

```
DATOS HISTÓRICOS (antes de Dec 17, 2025):
- Timestamps en UTC real
- Horario: 13:00-17:55 UTC (= 8:00-12:55 COT)

DATOS RECIENTES (después de Dec 17, 2025):
- Timestamps en COT pero almacenados como UTC
- Horario: 08:00-12:55 "UTC" (en realidad COT)

IMPACTO:
- Features temporales (hour_sin, hour_cos) calculados incorrectamente
- El modelo recibe observaciones inconsistentes
```

### Query de Diagnóstico

```sql
-- diagnostica/08_timezone_audit.sql

-- Verificar distribución de horas por período
SELECT
    CASE
        WHEN DATE(time) < '2025-12-17' THEN 'OLD_FORMAT'
        ELSE 'NEW_FORMAT'
    END as data_period,
    EXTRACT(HOUR FROM time) as hour,
    COUNT(*) as bar_count
FROM usdcop_m5_ohlcv
WHERE time > '2025-01-01'
GROUP BY 1, 2
ORDER BY 1, 2;

-- Resultado esperado si hay inconsistencia:
-- OLD_FORMAT: horas 13, 14, 15, 16, 17
-- NEW_FORMAT: horas 8, 9, 10, 11, 12

-- Verificar primer y último bar de cada día
SELECT
    DATE(time) as fecha,
    MIN(time::time) as primer_bar,
    MAX(time::time) as ultimo_bar,
    CASE
        WHEN MIN(time::time) >= '13:00' THEN 'UTC_FORMAT'
        WHEN MIN(time::time) >= '08:00' AND MIN(time::time) < '09:00' THEN 'COT_FORMAT'
        ELSE 'UNKNOWN'
    END as format_detected
FROM usdcop_m5_ohlcv
WHERE time > '2025-11-01'
GROUP BY DATE(time)
ORDER BY fecha DESC
LIMIT 60;
```

### Solución: Normalizar a UTC

```python
# scripts/normalize_timezones.py

import psycopg2
from datetime import datetime, timedelta
import pytz

def normalize_timestamps_to_utc(conn, start_date='2025-12-17'):
    """
    Normalize timestamps stored as COT-as-UTC back to true UTC.

    Data after 2025-12-17 was stored with COT times but UTC offset.
    We need to add 5 hours to convert to true UTC.
    """

    # Verify we're dealing with COT data
    check_query = """
    SELECT
        MIN(time::time) as first_bar,
        MAX(time::time) as last_bar
    FROM usdcop_m5_ohlcv
    WHERE DATE(time) = %s
    """

    with conn.cursor() as cur:
        cur.execute(check_query, (start_date,))
        first, last = cur.fetchone()

        # If first bar is around 8:00, it's COT stored as UTC
        if first and first.hour < 10:
            print(f"Confirmed: Data from {start_date} is in COT format (first bar: {first})")
        else:
            print(f"Data appears to already be in UTC (first bar: {first})")
            return

    # Update timestamps to true UTC (add 5 hours)
    update_query = """
    UPDATE usdcop_m5_ohlcv
    SET time = time + INTERVAL '5 hours'
    WHERE DATE(time) >= %s
      AND time::time < '13:00'  -- Only COT-format data
    """

    print(f"This will modify timestamps from {start_date} onwards.")
    confirm = input("Type 'YES' to proceed: ")

    if confirm == 'YES':
        with conn.cursor() as cur:
            cur.execute(update_query, (start_date,))
            rows_updated = cur.rowcount
            conn.commit()
            print(f"Updated {rows_updated} rows to UTC")
    else:
        print("Aborted")


def add_timezone_column(conn):
    """Add explicit timezone column for future data."""

    alter_query = """
    ALTER TABLE usdcop_m5_ohlcv
    ADD COLUMN IF NOT EXISTS tz_info VARCHAR(10) DEFAULT 'UTC';

    COMMENT ON COLUMN usdcop_m5_ohlcv.tz_info IS
    'Timezone of the timestamp. All data should be in UTC.';
    """

    with conn.cursor() as cur:
        cur.execute(alter_query)
        conn.commit()
        print("Added tz_info column")


if __name__ == "__main__":
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    normalize_timestamps_to_utc(conn, start_date='2025-12-17')
    add_timezone_column(conn)
    conn.close()
```

### Feature Builder Timezone-Aware

```python
# src/features/observation_builder_v20.py

class ObservationBuilderV20:
    """Timezone-aware observation builder."""

    def __init__(self):
        self.cot_tz = pytz.timezone('America/Bogota')
        self.utc_tz = pytz.UTC

    def build_temporal_features(self, timestamp):
        """Build temporal features always in COT (trading timezone)."""

        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            # Assume UTC if no timezone
            timestamp = self.utc_tz.localize(timestamp)

        # Convert to COT for temporal features
        cot_time = timestamp.astimezone(self.cot_tz)

        hour = cot_time.hour + cot_time.minute / 60.0
        day_of_week = cot_time.weekday()

        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 5)
        dow_cos = np.cos(2 * np.pi * day_of_week / 5)

        return {
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'dow_sin': dow_sin,
            'dow_cos': dow_cos,
        }
```

---

## FIX 4: US HOLIDAYS FLAG (ALTO)

### El Problema

```python
# trading_calendar.py actual:
class TradingCalendar:
    def __init__(self, include_us_holidays: bool = False):  # ← INCORRECTO
        self.include_us_holidays = include_us_holidays
```

USD/COP se ve afectado por festivos de USA porque:
1. El dólar es la moneda base
2. Los mercados de USD tienen menos liquidez en festivos USA
3. Noticias y flujos de USA afectan el par

### La Solución

```python
# airflow/dags/utils/trading_calendar.py

US_HOLIDAYS_2025_2026 = [
    # 2025
    '2025-01-01',  # New Year's Day
    '2025-01-20',  # MLK Day
    '2025-02-17',  # Presidents Day
    '2025-05-26',  # Memorial Day
    '2025-06-19',  # Juneteenth
    '2025-07-04',  # Independence Day
    '2025-09-01',  # Labor Day
    '2025-10-13',  # Columbus Day
    '2025-11-11',  # Veterans Day
    '2025-11-27',  # Thanksgiving
    '2025-12-25',  # Christmas

    # 2026
    '2026-01-01',  # New Year's Day
    '2026-01-19',  # MLK Day
    '2026-02-16',  # Presidents Day
    '2026-05-25',  # Memorial Day
    '2026-06-19',  # Juneteenth
    '2026-07-03',  # Independence Day (observed)
    '2026-09-07',  # Labor Day
    '2026-10-12',  # Columbus Day
    '2026-11-11',  # Veterans Day
    '2026-11-26',  # Thanksgiving
    '2026-12-25',  # Christmas
]

class TradingCalendarV2:
    def __init__(self):
        # SIEMPRE incluir festivos USA para USD/COP
        self.colombia_holidays = set(COLOMBIA_HOLIDAYS)
        self.us_holidays = set(US_HOLIDAYS_2025_2026)
        self.all_holidays = self.colombia_holidays | self.us_holidays

    def is_trading_day(self, date) -> bool:
        """Check if date is a valid trading day."""
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)

        # Weekend
        if date.weekday() >= 5:
            return False

        # Colombian holiday
        if date_str in self.colombia_holidays:
            return False

        # US holiday (affects USD liquidity)
        if date_str in self.us_holidays:
            return False

        return True

    def is_reduced_liquidity(self, date) -> bool:
        """Check if day has reduced liquidity (US holiday without COL holiday)."""
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        return date_str in self.us_holidays and date_str not in self.colombia_holidays
```

### Actualizar Inferencia

```python
# En l5_multi_model_inference.py:

from utils.trading_calendar import TradingCalendarV2

def should_run_inference(**context):
    """Check if we should run inference today."""
    calendar = TradingCalendarV2()
    today = datetime.now().date()

    if not calendar.is_trading_day(today):
        logger.info(f"Skipping inference - {today} is not a trading day")
        return False

    if calendar.is_reduced_liquidity(today):
        logger.warning(f"Reduced liquidity day (US holiday) - consider reduced position size")

    return True
```

---

## FIX 5: DATASET SPLIT V20 (ALTO)

### Especificación del Dataset

```python
# config/dataset_config_v20.py

DATASET_CONFIG_V20 = {
    "source_table": "usdcop_m5_ohlcv",

    "date_ranges": {
        "train": {
            "start": "2020-01-01",
            "end": "2024-12-31",
            "purpose": "Training - 5 años de datos históricos",
            "expected_bars": 150_000,  # ~30K bars/año × 5
        },
        "validation": {
            "start": "2025-01-01",
            "end": "2025-06-30",
            "purpose": "Hyperparameter tuning y early stopping",
            "expected_bars": 15_000,
        },
        "test": {
            "start": "2025-07-01",
            "end": "2026-01-08",
            "purpose": "Out-of-sample evaluation final",
            "expected_bars": 15_000,
        },
    },

    "features": {
        "technical": [
            "log_ret_5m", "log_ret_1h",
            "rsi_9", "macd_hist", "bb_width",
            "vol_ratio", "atr_pct",
        ],
        "temporal": [
            "hour_sin", "hour_cos",
            "dow_sin", "dow_cos",
        ],
        "macro": [
            "dxy_z", "vix_z",
        ],
        "state": [
            "position", "time_normalized",
        ],
    },

    "normalization": {
        "method": "z_score",
        "clip_range": (-5.0, 5.0),
        "use_train_stats": True,  # CRÍTICO: calcular stats solo en train
        "stats_file": "config/v20_norm_stats.json",
    },

    "filters": {
        "market_hours_only": True,
        "market_open": "08:00",
        "market_close": "12:55",
        "timezone": "America/Bogota",
        "exclude_holidays": True,
        "holiday_calendar": "COLOMBIA_AND_US",
    },

    "quality_checks": {
        "max_null_pct": 0.01,  # Max 1% nulls
        "min_bars_per_day": 50,  # At least 50 bars per trading day
        "max_gap_minutes": 30,  # Max 30 min gap within market hours
    },
}
```

### Script de Generación

```python
# scripts/generate_dataset_v20.py

import pandas as pd
import numpy as np
import json
from config.dataset_config_v20 import DATASET_CONFIG_V20

def generate_dataset_v20():
    """Generate train/val/test datasets for V20."""

    conn = get_db_connection()
    config = DATASET_CONFIG_V20

    datasets = {}

    for split_name, split_config in config['date_ranges'].items():
        print(f"\nGenerating {split_name} dataset...")

        # Query data
        query = f"""
        SELECT time, open, high, low, close, volume
        FROM {config['source_table']}
        WHERE time >= '{split_config['start']}'
          AND time < '{split_config['end']}'
          AND EXTRACT(DOW FROM time) BETWEEN 1 AND 5
          AND time::time >= '{config['filters']['market_open']}'
          AND time::time <= '{config['filters']['market_close']}'
        ORDER BY time
        """

        df = pd.read_sql(query, conn)
        print(f"  Raw bars: {len(df)}")

        # Build features
        df = build_features_v20(df)

        # Quality checks
        null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if null_pct > config['quality_checks']['max_null_pct']:
            print(f"  ⚠️ Warning: {null_pct:.2%} null values")

        datasets[split_name] = df
        print(f"  Final bars: {len(df)}")

    # Calculate normalization stats from TRAIN only
    train_df = datasets['train']
    norm_stats = {}

    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col in ['time', 'open', 'high', 'low', 'close', 'volume']:
            continue
        norm_stats[col] = {
            'mean': float(train_df[col].mean()),
            'std': float(train_df[col].std()),
        }

    # Save norm stats
    with open(config['normalization']['stats_file'], 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"\nSaved normalization stats to {config['normalization']['stats_file']}")

    # Apply normalization to all splits
    for split_name, df in datasets.items():
        for col, stats in norm_stats.items():
            if col in df.columns:
                df[col] = (df[col] - stats['mean']) / (stats['std'] + 1e-8)
                df[col] = df[col].clip(-5, 5)

        # Save dataset
        output_path = f"data/processed/{split_name}_v20.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved {split_name} to {output_path}")

    return datasets


def build_features_v20(df):
    """Build V20 features from OHLCV data."""

    # Log returns
    df['log_ret_5m'] = np.log(df['close'] / df['close'].shift(1))
    df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(12))

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(9).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_9'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    df['macd_hist'] = macd_line - signal_line

    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_width'] = (2 * std20) / (sma20 + 1e-8)

    # Volume ratio
    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / (df['close'] + 1e-8)

    # Temporal features
    df['hour'] = pd.to_datetime(df['time']).dt.hour + pd.to_datetime(df['time']).dt.minute / 60
    df['dow'] = pd.to_datetime(df['time']).dt.weekday

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 5)

    # Drop intermediate columns
    df = df.drop(columns=['hour', 'dow'], errors='ignore')

    # Drop NaN rows (from rolling calculations)
    df = df.dropna()

    return df


if __name__ == "__main__":
    generate_dataset_v20()
```

---

## FIX 6: ENTROPY REGULARIZATION (ALTO)

### El Problema

Acciones extremas (-0.8 a +0.8) indican que el modelo está muy seguro de sus predicciones, lo cual es peligroso porque:
1. No deja espacio para incertidumbre
2. Causa overtrading (0% HOLD)
3. Posible overfitting

### La Solución

```python
# notebooks/train_ppo_v20.py

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# PPO Config V20 con entropy regularization
ppo_config_v20 = {
    # Learning
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,

    # GAE
    "gamma": 0.99,
    "gae_lambda": 0.95,

    # Policy
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,

    # CRÍTICO: Entropy coefficient
    # Valor alto = más exploración = acciones menos extremas
    "ent_coef": 0.01,  # Probar 0.01, 0.02, 0.05

    # Value function
    "vf_coef": 0.5,

    # Gradient clipping
    "max_grad_norm": 0.5,

    # KL divergence target (previene updates muy agresivos)
    "target_kl": 0.015,
}

# Network architecture
policy_kwargs = {
    "net_arch": {
        "pi": [256, 256],  # Policy network
        "vf": [256, 256],  # Value network
    },
    "activation_fn": torch.nn.Tanh,
}

# Create model
model = PPO(
    "MlpPolicy",
    env,
    **ppo_config_v20,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./logs/ppo_v20/",
)

# Train with callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/ppo_v20/",
    log_path="./logs/ppo_v20/",
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True,
)

model.learn(
    total_timesteps=5_000_000,
    callback=eval_callback,
    progress_bar=True,
)
```

### Monitorear Distribución de Acciones

```python
# Durante training, monitorear:
def log_action_distribution(model, env, n_episodes=10):
    """Log action distribution to detect extreme actions."""
    actions = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            actions.append(action[0])
            obs, _, done, _ = env.step(action)

    actions = np.array(actions)

    stats = {
        "mean": np.mean(actions),
        "std": np.std(actions),
        "min": np.min(actions),
        "max": np.max(actions),
        "pct_extreme": np.mean(np.abs(actions) > 0.8) * 100,
        "pct_hold_zone": np.mean(np.abs(actions) < 0.15) * 100,
    }

    print(f"""
    Action Distribution:
    - Mean: {stats['mean']:.4f}
    - Std: {stats['std']:.4f}
    - Range: [{stats['min']:.4f}, {stats['max']:.4f}]
    - Extreme (>0.8): {stats['pct_extreme']:.1f}%
    - Hold zone (<0.15): {stats['pct_hold_zone']:.1f}%
    """)

    # Alert if too extreme
    if stats['pct_extreme'] > 50:
        print("⚠️ WARNING: >50% of actions are extreme. Increase ent_coef.")

    if stats['pct_hold_zone'] < 10:
        print("⚠️ WARNING: <10% in hold zone. Model may be overtrading.")

    return stats
```

---

## FIX 7: BENCHMARKS (MEDIO)

### Definición de Benchmarks

```python
# src/evaluation/benchmarks.py

import numpy as np
import pandas as pd

class BenchmarkStrategies:
    """Benchmark strategies for comparison."""

    @staticmethod
    def buy_and_hold(prices):
        """Simple buy and hold strategy."""
        returns = np.diff(prices) / prices[:-1]
        equity = [1.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        return np.array(equity)

    @staticmethod
    def random_signals(prices, seed=42):
        """Random signal strategy with same frequency."""
        np.random.seed(seed)
        n = len(prices)
        signals = np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])

        equity = [1.0]
        position = 0

        for i in range(1, n):
            if position != 0:
                pnl = position * (prices[i] - prices[i-1]) / prices[i-1]
                equity.append(equity[-1] * (1 + pnl))
            else:
                equity.append(equity[-1])

            position = signals[i]

        return np.array(equity)

    @staticmethod
    def ma_crossover(prices, fast=20, slow=50):
        """Moving average crossover strategy."""
        df = pd.DataFrame({'close': prices})
        df['ma_fast'] = df['close'].rolling(fast).mean()
        df['ma_slow'] = df['close'].rolling(slow).mean()

        df['signal'] = 0
        df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
        df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1

        equity = [1.0]
        position = 0

        for i in range(1, len(df)):
            if position != 0:
                pnl = position * (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
                equity.append(equity[-1] * (1 + pnl - 0.002))  # Include transaction cost
            else:
                equity.append(equity[-1])

            if df['signal'].iloc[i] != position:
                position = df['signal'].iloc[i]

        return np.array(equity)


def compare_with_benchmarks(model_equity, prices, model_name="Model"):
    """Compare model performance with benchmarks."""

    benchmarks = {
        "Buy & Hold": BenchmarkStrategies.buy_and_hold(prices),
        "Random": BenchmarkStrategies.random_signals(prices),
        "MA Crossover": BenchmarkStrategies.ma_crossover(prices),
        model_name: model_equity,
    }

    results = {}
    for name, equity in benchmarks.items():
        returns = np.diff(equity) / equity[:-1]
        results[name] = {
            "total_return": (equity[-1] / equity[0] - 1) * 100,
            "sharpe": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 60),  # Annualized
            "max_drawdown": calculate_max_drawdown(equity) * 100,
            "volatility": np.std(returns) * np.sqrt(252 * 60) * 100,
        }

    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    print(f"{'Strategy':<20} {'Return %':<12} {'Sharpe':<10} {'MaxDD %':<10}")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['total_return']:>10.2f}% {metrics['sharpe']:>10.2f} {metrics['max_drawdown']:>10.2f}%")

    return results
```

---

## FIX 8: EARLY STOPPING (MEDIO)

### Implementación

```python
# src/training/callbacks.py

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np

class StopTrainingOnNoImprovement(BaseCallback):
    """Stop training if no improvement for N evaluations."""

    def __init__(self, max_no_improvement_evals: int = 5, min_evals: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.n_evals = 0

    def _on_step(self) -> bool:
        # This is called after each evaluation
        if self.parent is not None:
            self.n_evals += 1

            # Get current mean reward from parent EvalCallback
            if hasattr(self.parent, 'best_mean_reward'):
                current_reward = self.parent.last_mean_reward

                if current_reward > self.best_mean_reward:
                    self.best_mean_reward = current_reward
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        print(f"New best reward: {current_reward:.4f}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"No improvement for {self.no_improvement_count} evals")

                # Check if we should stop
                if self.n_evals >= self.min_evals:
                    if self.no_improvement_count >= self.max_no_improvement_evals:
                        print(f"\n🛑 Early stopping: No improvement for {self.max_no_improvement_evals} evaluations")
                        return False

        return True


class LogActionDistributionCallback(BaseCallback):
    """Log action distribution periodically."""

    def __init__(self, check_freq: int = 50000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Sample actions
            actions = []
            obs = self.training_env.reset()
            for _ in range(1000):
                action, _ = self.model.predict(obs, deterministic=False)
                actions.append(action[0])
                obs, _, done, _ = self.training_env.step(action)
                if done:
                    obs = self.training_env.reset()

            actions = np.array(actions)
            pct_extreme = np.mean(np.abs(actions) > 0.8) * 100
            pct_hold = np.mean(np.abs(actions) < 0.15) * 100

            self.logger.record("actions/pct_extreme", pct_extreme)
            self.logger.record("actions/pct_hold_zone", pct_hold)
            self.logger.record("actions/std", np.std(actions))

            if self.verbose > 0:
                print(f"Actions: {pct_extreme:.1f}% extreme, {pct_hold:.1f}% hold zone")

        return True


# Usage in training:
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=StopTrainingOnNoImprovement(
        max_no_improvement_evals=5,
        min_evals=10,
        verbose=1
    ),
    best_model_save_path="./models/ppo_v20/",
    eval_freq=10000,
    n_eval_episodes=10,
)

action_callback = LogActionDistributionCallback(check_freq=50000, verbose=1)

model.learn(
    total_timesteps=5_000_000,
    callback=[eval_callback, action_callback],
)
```

---

## FIX 9: REWARD V20 MATH BUG (CRÍTICO)

### El Problema

```python
# CÓDIGO ORIGINAL (BUGGY):
def calculate(self, pnl, action, prev_action, ...):
    reward = pnl

    # Costo por cambiar posición
    if action != prev_action:
        reward -= self.transaction_cost  # reward = pnl - 0.002

    # Penalización asimétrica
    if pnl < 0:
        reward *= self.drawdown_penalty  # reward = (pnl - 0.002) * 2.0 ← BUG!

# El bug: Si pnl = -0.01 y cambió posición:
# reward = -0.01 - 0.002 = -0.012
# reward = -0.012 * 2.0 = -0.024
#
# Pero queríamos:
# Penalización por pérdida: -0.01 * 2 = -0.02
# Costo de transacción: -0.002
# Total: -0.022 (NO -0.024)
```

### La Solución

```python
# src/training/reward_calculator_v20.py

class RewardCalculatorV20:
    """
    Corrected reward calculator for V20.

    Order of operations:
    1. Base PnL
    2. Asymmetric penalty on PnL (NOT on costs)
    3. Transaction costs (additive, not multiplied)
    4. Hold bonus
    5. Consistency bonus
    6. Drawdown penalty
    """

    def __init__(self):
        self.transaction_cost = 0.002    # 0.2% per trade
        self.hold_bonus = 0.0001         # Per bar in hold
        self.loss_multiplier = 2.0       # Losses hurt 2x
        self.consistency_bonus = 0.0005  # Per consecutive win
        self.drawdown_threshold = 0.05   # 5% DD threshold
        self.drawdown_penalty = 0.001    # Penalty per % DD

    def calculate(
        self,
        pnl: float,
        action: int,
        prev_action: int,
        position_time: int,
        consecutive_wins: int,
        equity_peak: float,
        equity_current: float
    ) -> float:
        """
        Calculate reward with correct order of operations.

        Args:
            pnl: Raw P&L from trade (as percentage of equity)
            action: Current action (-1, 0, 1)
            prev_action: Previous action
            position_time: Bars in current position
            consecutive_wins: Number of consecutive winning trades
            equity_peak: Peak equity value
            equity_current: Current equity value

        Returns:
            Shaped reward value
        """

        # =============================================
        # 1. BASE REWARD = PnL
        # =============================================
        base_reward = pnl

        # =============================================
        # 2. ASYMMETRIC PENALTY (on PnL only, not costs)
        # =============================================
        if pnl < 0:
            base_reward = pnl * self.loss_multiplier

        # =============================================
        # 3. TRANSACTION COST (additive, separate)
        # =============================================
        transaction_penalty = 0.0
        if action != prev_action:
            transaction_penalty = -self.transaction_cost

        # =============================================
        # 4. HOLD BONUS (for patience)
        # =============================================
        hold_bonus = 0.0
        if action == 0:  # HOLD
            hold_bonus = self.hold_bonus * min(position_time, 10)

        # =============================================
        # 5. CONSISTENCY BONUS (for winning streaks)
        # =============================================
        consistency_bonus = 0.0
        if pnl > 0:
            consistency_bonus = self.consistency_bonus * min(consecutive_wins, 5)

        # =============================================
        # 6. DRAWDOWN PENALTY (for large drawdowns)
        # =============================================
        drawdown_penalty = 0.0
        if equity_peak > 0:
            current_dd = (equity_peak - equity_current) / equity_peak
            if current_dd > self.drawdown_threshold:
                drawdown_penalty = -self.drawdown_penalty * (current_dd - self.drawdown_threshold) * 100

        # =============================================
        # TOTAL REWARD
        # =============================================
        total_reward = (
            base_reward
            + transaction_penalty
            + hold_bonus
            + consistency_bonus
            + drawdown_penalty
        )

        return total_reward


# Unit tests
def test_reward_calculator():
    calc = RewardCalculatorV20()

    # Test 1: Winning trade, no position change
    reward = calc.calculate(
        pnl=0.01, action=1, prev_action=1,
        position_time=5, consecutive_wins=3,
        equity_peak=10000, equity_current=10100
    )
    expected = 0.01 + 0 + 0 + 0.0015 + 0  # pnl + no cost + no hold + consistency
    assert abs(reward - expected) < 1e-6, f"Test 1 failed: {reward} != {expected}"

    # Test 2: Losing trade with position change
    reward = calc.calculate(
        pnl=-0.01, action=-1, prev_action=1,
        position_time=0, consecutive_wins=0,
        equity_peak=10000, equity_current=9900
    )
    expected = -0.01 * 2.0 - 0.002 + 0 + 0 + 0  # pnl*2 + cost
    assert abs(reward - expected) < 1e-6, f"Test 2 failed: {reward} != {expected}"

    # Test 3: HOLD action
    reward = calc.calculate(
        pnl=0, action=0, prev_action=0,
        position_time=5, consecutive_wins=0,
        equity_peak=10000, equity_current=10000
    )
    expected = 0 + 0 + 0.0005 + 0 + 0  # pnl + no cost + hold bonus
    assert abs(reward - expected) < 1e-6, f"Test 3 failed: {reward} != {expected}"

    print("✅ All reward calculator tests passed")


if __name__ == "__main__":
    test_reward_calculator()
```

---

## CHECKLIST ACTUALIZADO

```
FIXES CRÍTICOS (P0) - HOY:
├── [x] Threshold 0.30 → 0.10 (en plan original)
├── [x] StateTracker persistence (en plan original)
├── [ ] Look-ahead bias fix (NUEVO)
├── [ ] Reward V20 math bug fix (NUEVO)

FIXES ALTOS (P1) - ESTA SEMANA:
├── [ ] Macro scraper con retry + fallback (NUEVO)
├── [ ] Timezone normalization (NUEVO)
├── [ ] US holidays flag (NUEVO)
├── [ ] Dataset split V20 specification (NUEVO)
├── [ ] Entropy regularization config (NUEVO)

FIXES MEDIOS (P2) - PRÓXIMA SEMANA:
├── [ ] Benchmarks implementation (NUEVO)
├── [ ] Early stopping callbacks (NUEVO)
```

---

## TIMELINE ACTUALIZADO

```
DÍA 1 (HOY):
├── Threshold fix (SQL)
├── Look-ahead bias fix (código)
├── Reward V20 math bug fix (código)
└── Timezone audit query

DÍA 2:
├── StateTracker persistence
├── Macro scraper diagnosis
└── US holidays flag

DÍA 3:
├── Timezone normalization script
├── Macro scraper con fallback
└── Dataset V20 generation

DÍA 4-5:
├── Environment V20 con reward corregido
├── PPO config con entropy
└── Training setup

SEMANA 2:
├── Training V20
├── Benchmarks comparison
└── Early stopping tuning

SEMANA 3:
├── A/B testing
└── Decisión final
```

---

## RESUMEN FINAL

El plan original cubría **85%** de lo necesario. Este addendum agrega los **15% críticos** que faltaban:

| Fix | Impacto si no se corrige |
|-----|--------------------------|
| Look-ahead bias | Backtest inflado, modelo no funciona en real |
| Macro scraper | Observaciones corruptas, inferencia degradada |
| Timezone | Features temporales incorrectos |
| US holidays | Tradear en días de baja liquidez |
| Reward math bug | Training no converge correctamente |
| Entropy | Modelo sigue dando acciones extremas |

**Sin estos fixes, el modelo V20 tendría los mismos problemas que V19.**

---

*Addendum generado para complementar Plan Maestro - 10 Expertos*
*Claude Code Audit System - 2026-01-08*
