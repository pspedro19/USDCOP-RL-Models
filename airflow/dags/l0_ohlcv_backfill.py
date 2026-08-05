"""
DAG: core_l0_01_ohlcv_backfill
==============================
USD/COP Trading System - L0 Consolidated OHLCV Backfill

Purpose:
    Unified OHLCV backfill for all 3 FX pairs (USD/COP, USD/MXN, USD/BRL).
    Handles seed restore (empty DB), gap detection, API backfill, and seed export.

    Consolidates:
    - l0_ohlcv_backfill.py (multi-symbol gap detection + API fill)
    - l0_ohlcv_historical_backfill.py (local file loading + seed export)
    - l0_backup_restore.py (OHLCV restore portion)

Schedule:
    Manual trigger only (no cron). Run after DB rebuild or to fill gaps.

Usage:
    # Backfill all 3 pairs (default)
    airflow dags trigger core_l0_01_ohlcv_backfill

    # Backfill a specific pair
    airflow dags trigger core_l0_01_ohlcv_backfill --conf '{"symbols": ["USD/MXN"]}'

    # Force gap detection even with seeds available
    airflow dags trigger core_l0_01_ohlcv_backfill --conf '{"force_backfill": true}'

Flow:
    start → health_check → [for each symbol]:
        → check_db → [empty? → restore_from_seed | has data → detect_gaps]
        → [gaps? → backfill_via_api | no gaps → skip]
    → export_seeds → validate → report

Timezone rules:
    - ALL timestamps stored in America/Bogota (COT)
    - USD/COP, USD/MXN: TwelveData timezone=America/Bogota (native COT)
    - USD/BRL: TwelveData timezone=UTC → convert to COT before insert
      (BRL returns incomplete data with timezone=America/Bogota)

Author: Pedro @ Lean Tech Solutions
Version: 4.0.0
Created: 2026-02-12
Contract: CTR-L0-BACKFILL-003
"""

from datetime import datetime, timedelta, date
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import pandas as pd
import requests
import pytz
from psycopg2.extras import execute_values
import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

from utils.dag_common import get_db_connection, load_feature_config
from contracts.dag_registry import CORE_L0_OHLCV_BACKFILL

CONFIG = load_feature_config(raise_on_error=False)
OHLCV_CONFIG = CONFIG.get('sources', {}).get('ohlcv', {})
TRADING_CONFIG = CONFIG.get('trading', {})
MARKET_HOURS = TRADING_CONFIG.get('market_hours', {})

DAG_ID = CORE_L0_OHLCV_BACKFILL

# Timezone
TIMEZONE_STR = MARKET_HOURS.get('timezone', 'America/Bogota')
COT_TZ = pytz.timezone(TIMEZONE_STR)
UTC_TZ = pytz.UTC

# Trading hours
_local_start = MARKET_HOURS.get('local_start', '08:00')
_local_end = MARKET_HOURS.get('local_end', '12:55')
MARKET_START_HOUR = int(_local_start.split(':')[0])
MARKET_START_MINUTE = int(_local_start.split(':')[1])
MARKET_END_HOUR = int(_local_end.split(':')[0])
MARKET_END_MINUTE = int(_local_end.split(':')[1])
MARKET_DAYS = TRADING_CONFIG.get('trading_days', [0, 1, 2, 3, 4])
BARS_PER_SESSION = TRADING_CONFIG.get('bars_per_session', 60)

# All 3 FX pairs
ALL_SYMBOLS = ['USD/COP', 'USD/MXN', 'USD/BRL']

# Per-symbol config: API timezone + seed file path
PROJECT_ROOT = Path(__file__).parent.parent.parent
SYMBOL_CONFIG = {
    'USD/COP': {
        'api_tz': TIMEZONE_STR,
        'needs_tz_convert': False,
        'seed_path': PROJECT_ROOT / 'seeds' / 'latest' / 'usdcop_m5_ohlcv.parquet',
    },
    'USD/MXN': {
        'api_tz': TIMEZONE_STR,
        'needs_tz_convert': False,
        'seed_path': PROJECT_ROOT / 'seeds' / 'latest' / 'usdmxn_m5_ohlcv.parquet',
    },
    'USD/BRL': {
        'api_tz': 'UTC',
        'needs_tz_convert': True,  # BRL returns incomplete data with America/Bogota
        'seed_path': PROJECT_ROOT / 'seeds' / 'latest' / 'usdbrl_m5_ohlcv.parquet',
    },
}

# Holidays (Colombia + US market holidays — gap detection skips these)
HOLIDAYS_STR = CONFIG.get('holidays_2025_colombia', [])
HOLIDAYS = {datetime.strptime(d, '%Y-%m-%d').date() for d in HOLIDAYS_STR}

ADDITIONAL_HOLIDAYS = [
    # 2020
    '2020-01-01', '2020-01-20', '2020-02-17', '2020-04-10', '2020-05-25',
    '2020-07-03', '2020-09-07', '2020-11-26', '2020-12-25',
    # 2021
    '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31',
    '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24', '2021-12-31',
    # 2022
    '2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30', '2022-06-20',
    '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-26',
    # 2023
    '2023-01-02', '2023-01-16', '2023-02-20', '2023-04-07', '2023-05-29',
    '2023-06-19', '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25',
    # 2024
    '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27',
    '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-24',
    '2024-12-25', '2024-12-31',
    # 2025
    '2025-01-01', '2025-01-06', '2025-03-24', '2025-04-17', '2025-04-18',
    '2025-05-01', '2025-06-02', '2025-06-23', '2025-06-30', '2025-07-20',
    '2025-08-07', '2025-08-18', '2025-10-13', '2025-11-03', '2025-11-17',
    '2025-12-08', '2025-12-24', '2025-12-25', '2025-12-31',
    # 2026
    '2026-01-01', '2026-01-12', '2026-01-19', '2026-02-16', '2026-03-23',
    '2026-04-02', '2026-04-03', '2026-05-01', '2026-05-18', '2026-05-25',
    '2026-06-08', '2026-06-15', '2026-06-19', '2026-06-29', '2026-07-03',
    '2026-07-20', '2026-08-07', '2026-08-17', '2026-09-07', '2026-10-12',
    '2026-11-02', '2026-11-16', '2026-11-26', '2026-12-08', '2026-12-24',
    '2026-12-25',
]
for d in ADDITIONAL_HOLIDAYS:
    try:
        HOLIDAYS.add(datetime.strptime(d, '%Y-%m-%d').date())
    except Exception:
        pass

# API Configuration
TWELVEDATA_API_KEYS = [
    os.environ.get(f'TWELVEDATA_API_KEY_{i}') for i in range(1, 9)
] + [os.environ.get('TWELVEDATA_API_KEY')]
TWELVEDATA_API_KEYS = [k for k in TWELVEDATA_API_KEYS if k]

_api_key_index = 0

TWELVEDATA_INTERVAL = OHLCV_CONFIG.get('granularity', '5min')
MAX_BARS_PER_REQUEST = 5000
API_RATE_DELAY_SECONDS = 1
MAX_BACKFILL_DAYS = 365
MIN_BARS_THRESHOLD = 10

if not TWELVEDATA_API_KEYS:
    logging.warning("No TWELVEDATA_API_KEY environment variables set - backfill will fail")


# =============================================================================
# HELPERS
# =============================================================================

def get_next_api_key() -> Optional[str]:
    """Rotate through API keys."""
    global _api_key_index
    if not TWELVEDATA_API_KEYS:
        return None
    key = TWELVEDATA_API_KEYS[_api_key_index % len(TWELVEDATA_API_KEYS)]
    _api_key_index += 1
    return key


def is_trading_day(d: date) -> bool:
    """Check if a date is a trading day (Mon-Fri, not a holiday)."""
    return d.weekday() in MARKET_DAYS and d not in HOLIDAYS


def get_all_trading_days(start_date: date, end_date: date) -> List[date]:
    """Get all trading days between two dates (inclusive)."""
    trading_days = []
    current = start_date
    while current <= end_date:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)
    return trading_days


def _validated_scope(raw, origen: str) -> List[str]:
    """Valida la forma del alcance pedido en el `conf`. Fail-closed (CXD-518).

    Sin esto, `symbols` aceptaba cualquier cosa y cada forma rota fallaba de un modo distinto
    y silencioso:

    - `"USD/MXN"` (string) es iterable, así que `symbol not in scope` hacía comparación de
      SUBCADENAS. Con un solo par coincide por accidente; con `"USD/MXN,USD/COP"` el
      aislamiento se vuelve impredecible en vez de romperse;
    - `[]` dejaba a `health_check` devolviendo `{'status': 'healthy', 'symbols': []}` —un run
      que se declara sano y no puede procesar nada— y el fallo aparecía tarde, en cada tarea;
    - `['FOO']` producía un run donde las tres tareas quedan `skipped` y el export consulta un
      alcance que no existe. Verde por vacuidad, otra vez;
    - `['USD/MXN', 'USD/MXN']` exportaba dos veces el mismo símbolo.

    Un alcance mal escrito tiene que romper en `health_check`, que es donde se declara, no
    dispersarse en síntomas aguas abajo.
    """
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"{origen} debe ser una lista de símbolos, no {type(raw).__name__} ({raw!r}); "
            f"un string es iterable y convertiría el filtro de alcance en comparación de "
            f"subcadenas"
        )
    scope = list(raw)
    if not scope:
        raise ValueError(f"{origen} vacío: un run sin alcance no se declara sano, se rechaza")
    no_str = [s for s in scope if not isinstance(s, str)]
    if no_str:
        raise ValueError(f"{origen} contiene valores no-string: {no_str!r}")
    duplicados = sorted({s for s in scope if scope.count(s) > 1})
    if duplicados:
        raise ValueError(f"{origen} tiene duplicados {duplicados}: el export los escribiría dos veces")
    desconocidos = [s for s in scope if s not in ALL_SYMBOLS]
    if desconocidos:
        raise ValueError(
            f"{origen} pide símbolos que este DAG no gobierna: {desconocidos} "
            f"(gobernados: {list(ALL_SYMBOLS)})"
        )
    return scope


def get_target_symbols(context) -> List[str]:
    """Alcance pedido en `dag_run.conf`, validado; sin `conf` son los 3 pares."""
    conf = context.get('dag_run').conf or {}
    if 'symbols' in conf:
        symbols = _validated_scope(conf['symbols'], "conf['symbols']")
    elif 'symbol' in conf:
        # Override legacy de un solo símbolo: misma validación, envuelto en lista.
        symbols = _validated_scope([conf['symbol']], "conf['symbol']")
    else:
        symbols = list(ALL_SYMBOLS)
    logging.info(f"[BACKFILL] Target symbols: {symbols}")
    return symbols


def resolve_scope(context) -> List[str]:
    """El alcance REAL de este run, leído del XCom que publicó `health_check`.

    Existe porque el `conf` no gobernaba nada: las tres tareas `process_*` se crean al
    parsear el DAG y cada una leía su propio `params['symbol']`, así que
    `symbols=['USD/MXN']` disparó igualmente gap detection de COP y BRL (run
    `codex_bl40_usdmxn_20260805T0015`, CXD-515). `get_target_symbols` calculaba la lista
    correcta y nadie la consultaba.

    **Fail-closed a propósito**: si el XCom no está, se levanta. Caer a `ALL_SYMBOLS` es
    exactamente cómo se perdió el aislamiento — un default permisivo convierte "no sé cuál
    es mi alcance" en "todos", que es la respuesta más peligrosa de las posibles.
    """
    scope = context['ti'].xcom_pull(key='target_symbols', task_ids='health_check')
    if not scope:
        raise ValueError(
            "target_symbols ausente en el XCom de health_check: sin alcance declarado no se "
            "procesa nada (no se asume ALL_SYMBOLS)"
        )
    return list(scope)


def get_data_date_range(conn, symbol: str) -> Tuple[Optional[date], Optional[date]]:
    """Get MIN and MAX dates from OHLCV table for a symbol."""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT DATE(MIN(time)), DATE(MAX(time))
            FROM usdcop_m5_ohlcv WHERE symbol = %s
        """, (symbol,))
        result = cur.fetchone()
        if result and result[0] and result[1]:
            return result[0], result[1]
        return None, None
    finally:
        cur.close()


def get_bars_per_day(conn, symbol: str) -> Dict[date, int]:
    """Get bar count per trading day for a symbol."""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT DATE(time) as trading_date, COUNT(*) as bar_count
            FROM usdcop_m5_ohlcv
            WHERE symbol = %s
              AND EXTRACT(DOW FROM time) BETWEEN 1 AND 5
              AND (
                  (time::time >= '13:00:00'::time AND time::time <= '17:55:00'::time)
                  OR
                  (time::time >= '08:00:00'::time AND time::time <= '12:55:00'::time)
              )
            GROUP BY DATE(time)
            ORDER BY DATE(time)
        """, (symbol,))
        return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        cur.close()


def detect_all_gaps(trading_days: List[date], bars_per_day: Dict[date, int]) -> List[Dict]:
    """Detect all gap days (missing or partial)."""
    gaps = []
    for day in trading_days:
        actual_bars = bars_per_day.get(day, 0)
        if actual_bars == 0:
            gaps.append({
                'date': day, 'expected_bars': BARS_PER_SESSION,
                'actual_bars': 0, 'missing_bars': BARS_PER_SESSION,
                'gap_type': 'FULL_DAY_MISSING',
            })
        elif actual_bars < MIN_BARS_THRESHOLD:
            gaps.append({
                'date': day, 'expected_bars': BARS_PER_SESSION,
                'actual_bars': actual_bars,
                'missing_bars': BARS_PER_SESSION - actual_bars,
                'gap_type': 'PARTIAL_DAY',
            })
    return gaps


def group_consecutive_gaps(gaps: List[Dict]) -> List[Dict]:
    """Group consecutive gap days into ranges."""
    if not gaps:
        return []
    gaps = sorted(gaps, key=lambda x: x['date'])
    ranges = []
    current_range = {
        'start_date': gaps[0]['date'], 'end_date': gaps[0]['date'],
        'days_missing': 1, 'total_bars_missing': gaps[0]['missing_bars'],
    }
    for gap in gaps[1:]:
        prev_date = current_range['end_date']
        current_date = gap['date']
        days_diff = (current_date - prev_date).days
        trading_days_between = len(get_all_trading_days(
            prev_date + timedelta(days=1), current_date - timedelta(days=1)
        ))
        if trading_days_between == 0 or days_diff <= 3:
            current_range['end_date'] = current_date
            current_range['days_missing'] += 1
            current_range['total_bars_missing'] += gap['missing_bars']
        else:
            ranges.append(current_range)
            current_range = {
                'start_date': gap['date'], 'end_date': gap['date'],
                'days_missing': 1, 'total_bars_missing': gap['missing_bars'],
            }
    ranges.append(current_range)
    return ranges


def fetch_ohlcv_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from TwelveData API for a date range.

    Uses per-symbol timezone: BRL requires UTC, others use COT.
    For BRL, timestamps are converted from UTC to COT after fetch.
    """
    api_key = get_next_api_key()
    if not api_key:
        logging.error("No TWELVEDATA_API_KEY configured")
        return pd.DataFrame()

    cfg = SYMBOL_CONFIG[symbol]

    # TwelveData quirk: end_date must be > start_date
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    if end_dt <= start_dt:
        end_dt = start_dt + timedelta(days=1)
        end_date = end_dt.strftime('%Y-%m-%d')

    url = 'https://api.twelvedata.com/time_series'
    params = {
        'symbol': symbol,
        'interval': TWELVEDATA_INTERVAL,
        'format': 'JSON',
        'timezone': cfg['api_tz'],
        'apikey': api_key,
        'start_date': start_date,
        'end_date': end_date,
    }

    try:
        logging.info(f"[{symbol}] Fetching {start_date}→{end_date} (tz={cfg['api_tz']})")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'code' in data and data['code'] != 200:
            logging.warning(f"[{symbol}] API warning: {data.get('message', 'Unknown')}")
            return pd.DataFrame()
        if 'values' not in data:
            logging.warning(f"[{symbol}] No data returned for {start_date}→{end_date}")
            return pd.DataFrame()

        records = []
        for bar in data['values']:
            bar_time = datetime.fromisoformat(bar['datetime'])
            if cfg['needs_tz_convert']:
                if bar_time.tzinfo is None:
                    bar_time = UTC_TZ.localize(bar_time)
                bar_time = bar_time.astimezone(COT_TZ)
            elif bar_time.tzinfo is None:
                # ROOT CAUSE of the tri-convention table (fixed 2026-07-21, CTR-DQ-TZ-001).
                # TwelveData returns WALL-CLOCK strings in the requested timezone. For COP/MXN
                # (api_tz=America/Bogota) this datetime is naive COT wall time; inserting it
                # naive lets postgres read it as UTC — a 12:55 COT close became the instant
                # 12:55 UTC (07:55 COT, pre-session). 15,865 rows were mislabeled this way and
                # corrected by scripts/ops/fix_tz_wall_cot_rows.py. BRL never had the bug only
                # because its UTC-fetch branch above localizes. Localizing to COT here makes
                # the value a true tz-aware instant; postgres stores instants, not wall times.
                bar_time = COT_TZ.localize(bar_time)
            records.append({
                'time': bar_time,
                'open': float(bar['open']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'close': float(bar['close']),
                'volume': float(bar.get('volume', 0)),
                'symbol': symbol,
                'source': 'twelvedata_backfill',
            })

        df = pd.DataFrame(records)
        logging.info(f"[{symbol}] Fetched {len(df)} bars")
        return df

    except Exception as e:
        logging.error(f"[{symbol}] API fetch error: {e}")
        raise


def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only market hours + trading days, removing holidays."""
    if df.empty:
        return df
    df = df.copy()
    df['_time_dt'] = pd.to_datetime(df['time'])
    mask = (
        (df['_time_dt'].dt.hour >= MARKET_START_HOUR) &
        ((df['_time_dt'].dt.hour < MARKET_END_HOUR) |
         ((df['_time_dt'].dt.hour == MARKET_END_HOUR) &
          (df['_time_dt'].dt.minute <= MARKET_END_MINUTE))) &
        (df['_time_dt'].dt.dayofweek.isin(MARKET_DAYS)) &
        (~df['_time_dt'].dt.date.isin(HOLIDAYS))
    )
    result = df[mask].drop(columns=['_time_dt'])
    n_dropped = len(df) - len(result)
    if n_dropped > 0:
        logging.info(f"  Filtered {n_dropped} off-session/holiday bars")
    return result


def _screen_backfill_frame(conn, df: pd.DataFrame) -> pd.DataFrame:
    """Filtra el lote dejando solo las barras aceptadas por el gate de calidad.

    Se agrupa **por símbolo** porque el backfill puede traer varios en un lote y la
    identidad canónica es por par `(proveedor, símbolo)`: publicar el lote entero bajo
    un solo símbolo le pondría a unas barras la identidad de otras.

    Un símbolo sin alias canónico (hoy `USD/BRL`, que no tiene `AssetProfile`) pasa sin
    filtrar **con aviso**: filtrarlo apagaría su backfill, y escribirlo callado haría
    creer que superó un gate que nunca corrió.
    """
    from src.data_quality.ingest_guard import BAR_FIELDS, publish_or_declare_gap

    trozos = []
    for symbol, grupo in df.groupby('symbol', sort=False):
        filas = [
            {campo: fila[campo] for campo in BAR_FIELDS if campo in fila}
            for _, fila in grupo.iterrows()
        ]
        publicacion = publish_or_declare_gap(
            conn,
            symbol=str(symbol),
            provider_id='twelvedata_backfill',
            rows=filas,
            interval_id='PT5M',
            source_uri=f'dag://l0_ohlcv_backfill/{symbol}',
        )
        if publicacion is None:
            logging.warning(
                "[%s] fuera de la cobertura Fabric: sin alias canonico en "
                "reference.provider_symbol, la barra NO pasa por el gate de calidad",
                symbol,
            )
            trozos.append(grupo)
            continue
        if publicacion.quarantine_count:
            logging.warning(
                "[%s] %d barra(s) en cuarentena, NO se escriben en la tabla legada",
                symbol, publicacion.quarantine_count,
            )
        instantes_ok = {f['time'] for f in publicacion.accepted}
        trozos.append(grupo[grupo['time'].isin(instantes_ok)])

    return pd.concat(trozos) if trozos else df.iloc[0:0]


def insert_ohlcv_batch(conn, df: pd.DataFrame) -> int:
    """Insert OHLCV batch with UPSERT on (time, symbol)."""
    if df.empty:
        return 0
    cur = conn.cursor()
    try:
        # C025/C026 — publicacion Fabric ANTES del UPSERT legado y en la MISMA
        # transaccion (el commit de abajo cierra ambos caminos). Solo entran las barras
        # `accepted`: una barra en cuarentena que aterriza igual anula la cuarentena.
        aceptadas = _screen_backfill_frame(conn, df)
        if aceptadas.empty:
            conn.commit()
            logging.info("0 barras aceptadas tras el gate de calidad")
            return 0

        values = [
            (row['time'], row['symbol'], row['open'], row['high'],
             row['low'], row['close'], row['volume'], row['source'])
            for _, row in aceptadas.iterrows()
        ]
        execute_values(
            cur,
            """
            INSERT INTO usdcop_m5_ohlcv
                (time, symbol, open, high, low, close, volume, source)
            VALUES %s
            ON CONFLICT (time, symbol) DO UPDATE SET
                volume = EXCLUDED.volume,
                source = EXCLUDED.source,
                updated_at = NOW()
            """,
            values,
            page_size=1000,
        )
        conn.commit()
        return len(values)
    except Exception as e:
        conn.rollback()
        logging.error(f"Insert error: {e}")
        raise
    finally:
        cur.close()


# =============================================================================
# DAG TASKS
# =============================================================================

def health_check(**context):
    """Verify DB connectivity and table existence."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'usdcop_m5_ohlcv'
            )
        """)
        exists = cur.fetchone()[0]
        if not exists:
            raise ValueError("Table usdcop_m5_ohlcv does not exist")
        logging.info("[HEALTH] usdcop_m5_ohlcv table exists ✓")
    finally:
        cur.close()
        conn.close()

    symbols = get_target_symbols(context)
    context['ti'].xcom_push(key='target_symbols', value=symbols)

    force = (context.get('dag_run').conf or {}).get('force_backfill', False)
    context['ti'].xcom_push(key='force_backfill', value=force)

    return {'status': 'healthy', 'symbols': symbols}


def process_symbol(**context):
    """
    Process a single symbol: restore from seed or detect gaps + backfill.

    This task is called once per symbol via op_kwargs.
    """
    symbol = context['params']['symbol']

    # El alcance manda ANTES de tocar la DB: un símbolo fuera de `symbols` no se procesa, y
    # su estado en Airflow es `skipped`, no `success` con `bars_backfilled: 0`. La diferencia
    # importa: un SUCCESS vacío se lee como "corrió y no había nada", que es una afirmación
    # sobre los datos; `skipped` dice la verdad — no se miró.
    scope = resolve_scope(context)
    if symbol not in scope:
        raise AirflowSkipException(
            f"{symbol} fuera del alcance declarado {scope}: no se ejecuta gap detection"
        )

    force = context['ti'].xcom_pull(key='force_backfill', task_ids='health_check') or False
    cfg = SYMBOL_CONFIG.get(symbol)
    if not cfg:
        logging.error(f"[{symbol}] Unknown symbol, skipping")
        return {'symbol': symbol, 'status': 'error', 'reason': 'unknown_symbol'}

    seed_path = cfg['seed_path']
    result = {
        'symbol': symbol,
        'restored': 0,
        'gaps_found': 0,
        'bars_backfilled': 0,
        'status': 'ok',
    }

    conn = get_db_connection()
    try:
        # --- Check DB state ---
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE symbol = %s", (symbol,))
        row_count = cur.fetchone()[0]
        cur.close()
        logging.info(f"[{symbol}] DB rows: {row_count}")

        # --- Restore from seed if DB empty ---
        if row_count == 0 and seed_path.exists() and not force:
            logging.info(f"[{symbol}] DB empty, restoring from seed: {seed_path}")
            df = pd.read_parquet(seed_path)
            logging.info(f"[{symbol}] Loaded {len(df)} rows from seed")

            # Ensure required columns
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            # Filter to this symbol only (unified parquet may have multiple)
            df = df[df['symbol'] == symbol].copy()
            if 'source' not in df.columns:
                df['source'] = 'seed_restore'

            if not df.empty:
                inserted = insert_ohlcv_batch(conn, df)
                result['restored'] = inserted
                logging.info(f"[{symbol}] Restored {inserted} rows from seed")

            # After restore, re-check count for gap detection
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE symbol = %s", (symbol,))
            row_count = cur.fetchone()[0]
            cur.close()

        # --- Gap detection ---
        min_date, max_date = get_data_date_range(conn, symbol)
        if min_date is None:
            logging.info(f"[{symbol}] No data after restore, skipping gap detection")
            result['status'] = 'no_data'
            return result

        # Extend to today
        today = datetime.now(COT_TZ).date()
        if max_date < today:
            max_date = today

        all_trading_days = get_all_trading_days(min_date, max_date)
        bars = get_bars_per_day(conn, symbol)
        gaps = detect_all_gaps(all_trading_days, bars)
        gap_ranges = group_consecutive_gaps(gaps)
        result['gaps_found'] = len(gap_ranges)

        logging.info(f"[{symbol}] Range: {min_date}→{max_date}, "
                     f"trading days: {len(all_trading_days)}, "
                     f"days with data: {len(bars)}, "
                     f"gap ranges: {len(gap_ranges)}")

        if not gap_ranges:
            logging.info(f"[{symbol}] Data complete, no gaps")
            return result

        # --- Backfill gaps via API ---
        total_inserted = 0
        fetch_errors = []
        for i, gr in enumerate(gap_ranges, 1):
            start_date = gr['start_date'].isoformat()
            end_date = gr['end_date'].isoformat()
            logging.info(f"[{symbol}] Gap #{i}/{len(gap_ranges)}: {start_date}→{end_date} "
                         f"({gr['days_missing']} days)")
            try:
                df = fetch_ohlcv_data(symbol, start_date, end_date)
                if not df.empty:
                    df = filter_market_hours(df)
                    if not df.empty:
                        inserted = insert_ohlcv_batch(conn, df)
                        total_inserted += inserted
                        logging.info(f"[{symbol}] Gap #{i}: inserted {inserted} bars")
                time.sleep(API_RATE_DELAY_SECONDS)
            except Exception as e:
                logging.error(f"[{symbol}] Gap #{i} error: {e}")
                fetch_errors.append(f"gap#{i} {start_date}→{end_date}: {e}")
                continue

        result['bars_backfilled'] = total_inserted
        result['fetch_errors'] = len(fetch_errors)
        result['first_fetch_error'] = fetch_errors[0] if fetch_errors else None
        logging.info(f"[{symbol}] Backfill complete: {total_inserted} bars inserted across "
                     f"{len(gap_ranges)} gaps ({len(fetch_errors)} fetch errors)")

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logging.error(f"[{symbol}] Processing failed: {e}")
    finally:
        conn.close()

    # ── Fail-closed FUERA del try: aquí sí escapa y la tarea queda `failed` ──────────
    #
    # Estas dos comprobaciones tienen que vivir fuera del bloque anterior: su `except
    # Exception` captura todo, pone `status: 'error'` en el diccionario y **retorna
    # normalmente**, asi que Airflow marcaba SUCCESS. Un `raise` dentro del try lo habria
    # tragado ese mismo `except`.
    #
    # Caso 1 — TODA peticion rechazada no es "backfill sin novedades", es un backfill que NO
    # OCURRIO. Medido en el run `codex_bl40_usdmxn_20260805T0110`: los dos fetch murieron con
    # `401 Unauthorized` porque las 8 claves de TwelveData son los placeholders literales
    # `YOUR_REAL_TWELVEDATA_API_KEY_*` de `.env.example`, y la tarea devolvio
    # `{'status': 'ok', 'bars_backfilled': 0}` y Airflow la marco SUCCESS. Ese SUCCESS afirma
    # "mire y no habia nada que traer" cuando el hecho es "no me dejaron mirar", y aguas abajo
    # se lee como que L0 esta al dia.
    #
    # El discriminador es el ERROR, no el cero: un hueco que la API sirve vacio (festivo, fuera
    # de su historia) es un cero legitimo y sigue siendo `ok`.
    if result.get('fetch_errors') and not result['bars_backfilled']:
        raise RuntimeError(
            f"{symbol}: los {result['fetch_errors']} fetch intentados fallaron y no entro "
            f"ninguna barra — el backfill no ocurrio. Primero: {result['first_fetch_error']}"
        )

    # Caso 2 — un `status: 'error'` que se reporta como SUCCESS es la misma mentira con otra
    # forma: si el procesamiento se rompio, la tarea esta roja.
    if result['status'] == 'error':
        raise RuntimeError(f"{symbol}: procesamiento fallido — {result.get('error')}")

    return result


def export_seeds(**context):
    """
    Export updated seed parquet files from DB.

    This replaces the formal backup system — seed files committed to Git
    ARE the backup. Generates per-symbol + unified parquets.
    """
    logging.info("=" * 60)
    logging.info("EXPORTING UPDATED SEED FILES")
    logging.info("=" * 60)

    # In the container the dags mount at /opt/airflow/dags, so PROJECT_ROOT
    # (3 parents up) resolves to /opt (read-only), not the repo root. The seed
    # volume is mounted at /opt/airflow/seeds (rw). Anchor on AIRFLOW_HOME so the
    # export lands on the mounted, writable seeds dir (matches l0_seed_backup.py).
    import os
    seeds_dir = Path(os.environ.get('AIRFLOW_HOME', '/opt/airflow')) / 'seeds' / 'latest'
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # Mismo alcance que `process_*`. Antes iteraba `ALL_SYMBOLS` y además reescribía el
    # parquet unificado desde la tabla COMPLETA: un run acotado a USD/MXN sobrescribió los
    # cuatro parquets trackeados con cero datos nuevos y hubo que restaurarlos a HEAD
    # (CXD-515). Un export fuera de alcance no es inocuo: los seeds son el backup de restore.
    scope = resolve_scope(context)
    logging.info(f"[EXPORT] Alcance declarado: {scope}")

    conn = get_db_connection()
    cur = conn.cursor()
    exported = []

    try:
        for symbol in scope:
            safe_name = symbol.replace('/', '').lower()
            seed_file = seeds_dir / f'{safe_name}_m5_ohlcv.parquet'

            cur.execute("""
                SELECT time, symbol, open, high, low, close, volume
                FROM usdcop_m5_ohlcv
                WHERE symbol = %s
                ORDER BY time
            """, (symbol,))
            rows = cur.fetchall()

            if not rows:
                logging.info(f"[{symbol}] No data to export")
                continue

            df = pd.DataFrame(rows, columns=[
                'time', 'symbol', 'open', 'high', 'low', 'close', 'volume'
            ])
            df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('America/Bogota')
            df.to_parquet(seed_file, index=False)
            logging.info(f"[{symbol}] Exported {len(df)} rows → {seed_file.name}")
            exported.append({'symbol': symbol, 'rows': len(df), 'file': str(seed_file)})

        # Unified seed (all pairs) — SOLO si el run cubrió los tres pares. Reescribirlo desde
        # un run parcial publicaría un "unificado" cuya mitad no se verificó en esta corrida.
        rows = []
        if set(scope) != set(ALL_SYMBOLS):
            logging.info(
                f"[UNIFIED] Omitido: alcance parcial {scope}; el unificado sólo se reescribe "
                f"cuando el run cubre {ALL_SYMBOLS}"
            )
            exported.append({'symbol': 'ALL', 'rows': 0, 'file': None, 'skipped': 'partial_scope'})
        else:
            cur.execute("""
                SELECT time, symbol, open, high, low, close, volume
                FROM usdcop_m5_ohlcv
                ORDER BY symbol, time
            """)
            rows = cur.fetchall()

        if rows:
            df_all = pd.DataFrame(rows, columns=[
                'time', 'symbol', 'open', 'high', 'low', 'close', 'volume'
            ])
            df_all['time'] = pd.to_datetime(df_all['time'], utc=True).dt.tz_convert('America/Bogota')
            unified_file = seeds_dir / 'fx_multi_m5_ohlcv.parquet'
            df_all.to_parquet(unified_file, index=False)
            logging.info(f"[UNIFIED] Exported {len(df_all)} rows → {unified_file.name}")
            exported.append({'symbol': 'ALL', 'rows': len(df_all), 'file': str(unified_file)})

    finally:
        cur.close()
        conn.close()

    logging.info(f"Seed export complete: {len(exported)} files")
    return {'exported': exported}


def validate_results(**context):
    """Valida el estado final de los símbolos DENTRO DEL ALCANCE del run.

    Reportar los tres pares tras un run de uno solo produce un informe que parece cubrir lo
    que no se tocó. El alcance viaja en el propio reporte (`_scope`) para que la evidencia
    diga de qué es evidencia.
    """
    scope = resolve_scope(context)
    conn = get_db_connection()
    cur = conn.cursor()
    report = {'_scope': scope}

    try:
        for symbol in scope:
            cur.execute("""
                SELECT COUNT(*), MIN(time), MAX(time)
                FROM usdcop_m5_ohlcv WHERE symbol = %s
            """, (symbol,))
            row = cur.fetchone()
            report[symbol] = {
                'total_bars': row[0],
                'earliest': str(row[1]) if row[1] else None,
                'latest': str(row[2]) if row[2] else None,
            }
            logging.info(f"[{symbol}] {row[0]} bars, {row[1]} → {row[2]}")

    finally:
        cur.close()
        conn.close()

    logging.info("=" * 60)
    logging.info(f"BACKFILL VALIDATION COMPLETE — alcance {scope}")
    for sym in scope:
        info = report[sym]
        logging.info(f"  {sym}: {info['total_bars']} bars ({info['earliest']} → {info['latest']})")
    logging.info("=" * 60)

    return report


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='L0: Unified OHLCV backfill — seed restore, gap detection, API fill, seed export (3 pairs)',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['core', 'l0', 'ohlcv', 'backfill', 'multi-pair'],
)

with dag:

    task_health = PythonOperator(
        task_id='health_check',
        python_callable=health_check,
    )

    # One task per symbol — they run sequentially to respect API rate limits.
    #
    # `trigger_rule='none_failed'` es OBLIGATORIO aquí y no un detalle de estilo (CXD-521).
    # La cadena es secuencial por el pool de API, así que con el `all_success` por defecto un
    # `skipped` INTENCIONAL aguas arriba hace inalcanzable todo lo que va detrás: el run
    # `codex_bl40_usdmxn_20260805T0059` con `symbols=['USD/MXN']` termino SUCCESS en 11s
    # habiendo ejecutado SOLO el skip de COP — MXN, BRL, export y validate quedaron `skipped`
    # sin lanzarse, y el aislamiento que este DAG acababa de ganar se convirtio en un run que
    # no hace nada y lo llama exito.
    #
    # `none_failed` distingue las dos cosas que `all_success` confunde: continua ante un
    # upstream `skipped` (decision de alcance, no incidente) y para ante `failed`/
    # `upstream_failed` (incidente real). Mismo criterio que ya usan export/validate.
    symbol_tasks = []
    for sym in ALL_SYMBOLS:
        safe_id = sym.replace('/', '_').lower()
        task = PythonOperator(
            task_id=f'process_{safe_id}',
            python_callable=process_symbol,
            params={'symbol': sym},
            pool='api_requests',
            trigger_rule='none_failed',
        )
        symbol_tasks.append(task)

    task_export = PythonOperator(
        task_id='export_seeds',
        python_callable=export_seeds,
        trigger_rule='none_failed_min_one_success',
    )

    task_validate = PythonOperator(
        task_id='validate_results',
        python_callable=validate_results,
        trigger_rule='none_failed_min_one_success',
    )

    # Chain: health → symbols (secuencial, por el pool de API) → export → validate
    #
    # `task_export` cuelga de LOS TRES símbolos, no sólo del último. Con
    # `none_failed_min_one_success` y un único upstream, un run acotado a USD/MXN deja a BRL
    # `skipped` — cero éxitos entre los upstream directos — y export/validate se saltarían
    # aunque MXN hubiera corrido bien. El fan-in es lo que hace que la regla signifique lo que
    # dice: "al menos un símbolo se proceso de verdad". La secuencia COP→MXN→BRL se conserva
    # para no golpear la API en paralelo; el fan-in sólo añade dependencias, no concurrencia.
    task_health >> symbol_tasks[0]
    for i in range(len(symbol_tasks) - 1):
        symbol_tasks[i] >> symbol_tasks[i + 1]
    for task in symbol_tasks:
        task >> task_export
    task_export >> task_validate
