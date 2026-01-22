"""
DAG: bootstrap_system
=====================
Pipeline de Inicializacion del Sistema - USD/COP Forecasting

ARQUITECTURA:
- Carga SOLO datos base (OHLCV + Macro consolidado)
- NO carga features_ml.csv (se CALCULA desde datos base)
- Trigger l1_feature_engineering para generar features
- Trigger monthly_training para entrenar modelos

TABLAS BASE (Semilla):
- core.usdcop_historical  <- usdcop_historical.csv
- core.macro_indicators   <- macro_consolidated.csv

TABLA CALCULADA:
- core.features_ml        <- l1_feature_engineering (calcula desde base)

Trigger: Manual (no tiene schedule)

Prerequisitos:
- data/raw/usdcop_historical.csv
- data/processed/macro_consolidated.csv
- Servicios Docker levantados
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.exceptions import AirflowException
from airflow.models import Variable
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import logging
import os
import json
from pathlib import Path

DAG_ID = "00_bootstrap_system"

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(os.getenv('DATA_DIR', '/opt/airflow/data'))

# Archivos semilla - SOLO DATOS BASE
USDCOP_CSV = DATA_DIR / 'raw' / 'usdcop_historical.csv'
MACRO_CSV = DATA_DIR / 'processed' / 'macro_consolidated.csv'

# Alternativas
USDCOP_CSV_ALT = DATA_DIR / 'usdcop_historical.csv'
MACRO_CSV_ALT = DATA_DIR / 'macro_consolidated.csv'

# =============================================================================
# DEFAULT ARGS
# =============================================================================

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(hours=2),
}


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_db_connection():
    """Obtener conexion a PostgreSQL."""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'pipeline_db'),
        user=os.getenv('POSTGRES_USER', 'pipeline'),
        password=os.getenv('POSTGRES_PASSWORD', 'pipeline_secret'),
    )


def safe_float(val):
    """Convertir valor a float de forma segura."""
    if pd.isna(val) or val == '' or val == '-':
        return None
    try:
        result = float(val)
        if np.isinf(result) or np.isnan(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def check_prerequisites(**context):
    """
    Verificar que existen los archivos CSV base necesarios.
    """
    logging.info("=" * 60)
    logging.info("BOOTSTRAP SYSTEM - Verificando prerequisitos")
    logging.info("=" * 60)

    # Buscar archivo USDCOP
    usdcop_path = None
    for path in [USDCOP_CSV, USDCOP_CSV_ALT]:
        if path.exists():
            usdcop_path = path
            logging.info(f"[OK] Encontrado USDCOP: {path}")
            break

    if not usdcop_path:
        raise AirflowException(
            f"No se encontro archivo USDCOP. "
            f"Buscado en: {USDCOP_CSV}, {USDCOP_CSV_ALT}"
        )

    # Buscar archivo Macro
    macro_path = None
    for path in [MACRO_CSV, MACRO_CSV_ALT]:
        if path.exists():
            macro_path = path
            logging.info(f"[OK] Encontrado Macro: {path}")
            break

    if not macro_path:
        logging.warning(f"[WARN] Archivo Macro no encontrado - continuando sin macro")

    # Verificar conexion a PostgreSQL
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        logging.info("[OK] Conexion a PostgreSQL")
    except Exception as e:
        raise AirflowException(f"No se puede conectar a PostgreSQL: {e}")

    # Guardar paths en XCom
    context['ti'].xcom_push(key='usdcop_path', value=str(usdcop_path))
    context['ti'].xcom_push(key='macro_path', value=str(macro_path) if macro_path else None)

    return {
        'usdcop_path': str(usdcop_path),
        'macro_path': str(macro_path) if macro_path else None,
        'status': 'ready'
    }


def check_if_already_initialized(**context):
    """
    Verificar si el sistema ya fue inicializado.
    """
    logging.info("Verificando si el sistema ya esta inicializado...")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Verificar si hay datos en usdcop_historical
        cursor.execute("SELECT COUNT(*) FROM core.usdcop_historical")
        usdcop_count = cursor.fetchone()[0]

        # Verificar si hay modelos registrados
        cursor.execute("SELECT COUNT(*) FROM ml.model_registry")
        models_count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        logging.info(f"Datos actuales: usdcop={usdcop_count}, models={models_count}")

        if usdcop_count > 1000 and models_count > 0:
            logging.info("Sistema ya inicializado, saltando bootstrap")
            return 'skip_bootstrap'
        else:
            logging.info("Sistema requiere inicializacion")
            return 'load_usdcop_historical'

    except Exception as e:
        logging.warning(f"Error verificando estado: {e}")
        return 'load_usdcop_historical'


def load_usdcop_historical(**context):
    """
    Cargar datos OHLCV historicos a core.usdcop_historical.
    Formato: Investing.com (MM/DD/YYYY)
    """
    logging.info("=" * 60)
    logging.info("CARGANDO USDCOP HISTORICAL (Datos Base)")
    logging.info("=" * 60)

    usdcop_path = context['ti'].xcom_pull(key='usdcop_path')

    # Leer CSV
    logging.info(f"Leyendo: {usdcop_path}")
    df = pd.read_csv(usdcop_path, encoding='utf-8-sig')

    # Detectar formato de columnas
    if len(df.columns) >= 6:
        # Formato Investing.com: Date, Price, Open, High, Low, Vol., Change%
        df.columns = ['date', 'close', 'open', 'high', 'low', 'volume', 'change_pct'][:len(df.columns)]

    # Parsear fecha (varios formatos)
    for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']:
        try:
            df['date'] = pd.to_datetime(df['date'], format=fmt)
            logging.info(f"Formato de fecha detectado: {fmt}")
            break
        except:
            continue

    # Limpiar columnas numericas
    for col in ['close', 'open', 'high', 'low']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    # Parsear volumen (K, M, B suffixes)
    def parse_volume(val):
        if pd.isna(val) or val == '' or val == '-':
            return None
        val = str(val).strip()
        try:
            if val.endswith('K'):
                return int(float(val[:-1]) * 1000)
            elif val.endswith('M'):
                return int(float(val[:-1]) * 1000000)
            elif val.endswith('B'):
                return int(float(val[:-1]) * 1000000000)
            return int(float(val.replace(',', '')))
        except:
            return None

    if 'volume' in df.columns:
        df['volume'] = df['volume'].apply(parse_volume)

    # Ordenar por fecha
    df = df.sort_values('date').reset_index(drop=True)

    # Calcular retornos y volatilidad
    df['return_1d'] = np.log(df['close'] / df['close'].shift(1))
    df['return_5d'] = np.log(df['close'] / df['close'].shift(5))
    df['return_10d'] = np.log(df['close'] / df['close'].shift(10))
    df['return_22d'] = np.log(df['close'] / df['close'].shift(22))
    df['volatility_5d'] = df['return_1d'].rolling(5).std() * np.sqrt(252)
    df['volatility_21d'] = df['return_1d'].rolling(21).std() * np.sqrt(252)

    logging.info(f"Registros a cargar: {len(df)}")
    logging.info(f"Rango: {df['date'].min().date()} a {df['date'].max().date()}")

    # Insertar en BD
    conn = get_db_connection()
    cursor = conn.cursor()

    # Truncar tabla (carga fresca)
    cursor.execute("TRUNCATE TABLE core.usdcop_historical RESTART IDENTITY CASCADE")

    insert_sql = """
        INSERT INTO core.usdcop_historical
        (date, open_price, high_price, low_price, close_price, volume,
         return_1d, return_5d, return_10d, return_22d, volatility_5d, volatility_21d, source)
        VALUES %s
        ON CONFLICT (date) DO UPDATE SET
            close_price = EXCLUDED.close_price,
            updated_at = CURRENT_TIMESTAMP
    """

    records = []
    for _, row in df.iterrows():
        vol = row.get('volume')
        if pd.isna(vol):
            vol = None

        records.append((
            row['date'].date(),
            safe_float(row.get('open')),
            safe_float(row.get('high')),
            safe_float(row.get('low')),
            safe_float(row['close']),
            vol,
            safe_float(row['return_1d']),
            safe_float(row['return_5d']),
            safe_float(row['return_10d']),
            safe_float(row['return_22d']),
            safe_float(row['volatility_5d']),
            safe_float(row['volatility_21d']),
            'bootstrap_csv'
        ))

    execute_values(cursor, insert_sql, records, page_size=500)
    conn.commit()

    cursor.close()
    conn.close()

    logging.info(f"[OK] Cargados {len(records)} registros a core.usdcop_historical")

    context['ti'].xcom_push(key='usdcop_loaded', value=len(records))

    return {'records_loaded': len(records)}


def load_macro_consolidated(**context):
    """
    Cargar datos macro consolidados a core.macro_indicators.
    """
    logging.info("=" * 60)
    logging.info("CARGANDO MACRO CONSOLIDATED (Datos Base)")
    logging.info("=" * 60)

    macro_path = context['ti'].xcom_pull(key='macro_path')

    if not macro_path:
        logging.warning("No hay archivo macro, saltando...")
        return {'status': 'skipped'}

    # Leer CSV
    logging.info(f"Leyendo: {macro_path}")
    df = pd.read_csv(macro_path)

    # Detectar columna de fecha
    date_col = None
    for col in ['date', 'Date', 'DATE', 'fecha', 'Fecha']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        date_col = df.columns[0]

    df['date'] = pd.to_datetime(df[date_col])

    # Obtener columnas de indicadores (excluyendo fecha)
    indicator_cols = [c for c in df.columns if c not in ['date', date_col, 'Unnamed: 0']]

    # Ordenar por fecha
    df = df.sort_values('date').reset_index(drop=True)

    logging.info(f"Indicadores encontrados: {len(indicator_cols)}")
    logging.info(f"Registros: {len(df)}")
    logging.info(f"Rango: {df['date'].min().date()} a {df['date'].max().date()}")

    # Insertar en BD
    conn = get_db_connection()
    cursor = conn.cursor()

    # Truncar tabla (carga fresca)
    cursor.execute("TRUNCATE TABLE core.macro_indicators RESTART IDENTITY CASCADE")

    insert_sql = """
        INSERT INTO core.macro_indicators (date, indicator_code, value, source, frequency)
        VALUES %s
        ON CONFLICT (date, indicator_code) DO UPDATE SET
            value = EXCLUDED.value
    """

    records = []
    for _, row in df.iterrows():
        for col in indicator_cols:
            val = safe_float(row[col])
            if val is not None:
                records.append((
                    row['date'].date(),
                    col,
                    val,
                    'bootstrap_csv',
                    'daily'
                ))

    if records:
        execute_values(cursor, insert_sql, records, page_size=1000)
        conn.commit()

    cursor.close()
    conn.close()

    logging.info(f"[OK] Cargados {len(records)} registros a core.macro_indicators")

    context['ti'].xcom_push(key='macro_loaded', value=len(records))

    return {'records_loaded': len(records), 'indicators': len(indicator_cols)}


def verify_base_data(**context):
    """
    Verificar que los datos base fueron cargados correctamente.
    """
    logging.info("=" * 60)
    logging.info("VERIFICANDO DATOS BASE")
    logging.info("=" * 60)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Verificar usdcop_historical
    cursor.execute("""
        SELECT COUNT(*), MIN(date), MAX(date)
        FROM core.usdcop_historical
    """)
    usdcop_count, usdcop_min, usdcop_max = cursor.fetchone()

    logging.info(f"core.usdcop_historical: {usdcop_count} registros")
    logging.info(f"  Rango: {usdcop_min} a {usdcop_max}")

    # Verificar macro_indicators
    cursor.execute("""
        SELECT COUNT(*), COUNT(DISTINCT indicator_code), MIN(date), MAX(date)
        FROM core.macro_indicators
    """)
    macro_count, n_indicators, macro_min, macro_max = cursor.fetchone()

    logging.info(f"core.macro_indicators: {macro_count} registros, {n_indicators} indicadores")
    if macro_min:
        logging.info(f"  Rango: {macro_min} a {macro_max}")

    cursor.close()
    conn.close()

    # Validar minimos
    if usdcop_count < 500:
        raise AirflowException(f"Datos USDCOP insuficientes: {usdcop_count} < 500")

    logging.info("[OK] Datos base verificados")

    return {
        'usdcop_count': usdcop_count,
        'macro_count': macro_count,
        'n_indicators': n_indicators
    }


def mark_system_initialized(**context):
    """
    Marcar el sistema como inicializado.
    """
    logging.info("=" * 60)
    logging.info("MARCANDO SISTEMA COMO INICIALIZADO")
    logging.info("=" * 60)

    usdcop_loaded = context['ti'].xcom_pull(key='usdcop_loaded') or 0
    macro_loaded = context['ti'].xcom_pull(key='macro_loaded') or 0

    try:
        Variable.set(
            'system_bootstrap_completed',
            json.dumps({
                'timestamp': datetime.now().isoformat(),
                'usdcop_loaded': usdcop_loaded,
                'macro_loaded': macro_loaded,
                'architecture': 'base_data_only_v2',
                'version': '2.0'
            })
        )
        logging.info("[OK] Sistema marcado como inicializado")
    except Exception as e:
        logging.warning(f"No se pudo guardar variable de Airflow: {e}")

    return {'status': 'initialized'}


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Bootstrap: Carga datos base, calcula features, entrena modelos',
    schedule_interval=None,  # Solo se ejecuta manualmente
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['bootstrap', 'initialization', 'setup'],
    doc_md=__doc__,
) as dag:

    # Task: Verificar prerequisitos
    check_prereqs = PythonOperator(
        task_id='check_prerequisites',
        python_callable=check_prerequisites,
    )

    # Task: Verificar si ya esta inicializado
    check_initialized = BranchPythonOperator(
        task_id='check_if_initialized',
        python_callable=check_if_already_initialized,
    )

    # Task: Cargar USDCOP historical
    load_usdcop = PythonOperator(
        task_id='load_usdcop_historical',
        python_callable=load_usdcop_historical,
    )

    # Task: Cargar Macro consolidated
    load_macro = PythonOperator(
        task_id='load_macro_consolidated',
        python_callable=load_macro_consolidated,
    )

    # Task: Verificar datos base
    verify_data = PythonOperator(
        task_id='verify_base_data',
        python_callable=verify_base_data,
    )

    # Task: Trigger Feature Engineering (CALCULA features_ml)
    trigger_features = TriggerDagRunOperator(
        task_id='trigger_feature_engineering',
        trigger_dag_id='03_l1_feature_engineering',
        wait_for_completion=True,
        poke_interval=30,
        execution_timeout=timedelta(hours=1),
    )

    # Task: Trigger Training
    trigger_training = TriggerDagRunOperator(
        task_id='trigger_monthly_training',
        trigger_dag_id='04_l2_monthly_training',
        wait_for_completion=True,
        poke_interval=60,
        execution_timeout=timedelta(hours=4),
    )

    # Task: Trigger Inferencia inicial
    trigger_inference = TriggerDagRunOperator(
        task_id='trigger_weekly_inference',
        trigger_dag_id='05_l2_weekly_inference',
        wait_for_completion=True,
        poke_interval=30,
        execution_timeout=timedelta(hours=1),
    )

    # Task: Marcar como inicializado
    mark_initialized = PythonOperator(
        task_id='mark_system_initialized',
        python_callable=mark_system_initialized,
        trigger_rule='none_failed_min_one_success',
    )

    # Task: Skip (si ya esta inicializado)
    skip_bootstrap = EmptyOperator(
        task_id='skip_bootstrap',
    )

    # Task: Fin
    end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success',
    )

    # ==========================================================================
    # DEPENDENCIES
    # ==========================================================================
    #
    # FLUJO:
    # 1. check_prerequisites
    # 2. check_if_initialized
    #    ├── skip_bootstrap → end
    #    └── load_usdcop → load_macro → verify_data
    #        → trigger_features (CALCULA core.features_ml)
    #        → trigger_training
    #        → trigger_inference
    #        → mark_initialized → end

    check_prereqs >> check_initialized

    # Branch: Si ya inicializado, saltar
    check_initialized >> skip_bootstrap >> end

    # Branch: Si no inicializado, cargar datos base y procesar
    check_initialized >> load_usdcop >> load_macro >> verify_data
    verify_data >> trigger_features >> trigger_training >> trigger_inference
    trigger_inference >> mark_initialized >> end
