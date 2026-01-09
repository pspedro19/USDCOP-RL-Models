"""
DAG: usdcop_m5__00b_l0_macro_scraping
================================================================================
Actualiza datos macroeconómicos 3 veces al día desde múltiples fuentes:
- TwelveData API: DXY, VIX, commodities, FX pairs
- FRED API: Fed Funds, Treasury yields
- Web Scraping: EMBI (BCRP), calendarios económicos

Horarios de ejecución (COT):
- 07:55 AM: Pre-apertura (captura valores overnight)
- 10:30 AM: Mid-morning (actualización pre-pico)
- 12:00 PM: Cierre sesión (valores finales del día)

Tablas actualizadas:
- dw.fact_macro_realtime
- macro_ohlcv (legacy)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import numpy as np
import json
import sys
import os
import requests

sys.path.insert(0, os.path.dirname(__file__))

DAG_ID = 'usdcop_m5__00b_l0_macro_scraping'

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════════

TWELVEDATA_SYMBOLS = {
    'DXY': 'DXY',
    'VIX': 'VIX',
    'USDMXN': 'USD/MXN',
    'USDCLP': 'USD/CLP',
    'BRENT': 'BRENT',
    'WTI': 'WTI',
    'GOLD': 'XAU/USD',
}

FRED_SERIES = {
    'FEDFUNDS': 'FEDFUNDS',
    'DGS2': 'DGS2',
    'DGS10': 'DGS10',
}

# Rangos esperados para validación
EXPECTED_RANGES = {
    'DXY': (80, 130),
    'VIX': (8, 90),
    'USDMXN': (15, 30),
    'USDCLP': (600, 1200),
    'BRENT': (20, 200),
    'WTI': (20, 200),
    'GOLD': (1000, 3000),
    'EMBI': (50, 1500),
    'FEDFUNDS': (0, 10),
    'DGS2': (0, 10),
    'DGS10': (0, 10),
}

# Parámetros de normalización para Z-scores
NORM_PARAMS = {
    'DXY': {'mean': 103, 'std': 5},
    'VIX': {'mean': 20, 'std': 10},
    'EMBI': {'mean': 300, 'std': 100},
}

default_args = {
    'owner': 'data_engineering',
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
}


# ═══════════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE UTILIDAD
# ═══════════════════════════════════════════════════════════════════════════════════

def get_db_connection():
    """Obtiene conexión a PostgreSQL"""
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'usdcop-postgres-timescale'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        database=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=os.environ.get('POSTGRES_PASSWORD', 'admin123')
    )


def get_twelvedata_api_keys():
    """Obtiene lista de API keys de TwelveData"""
    keys = []
    for i in range(1, 33):
        key = os.environ.get(f'TWELVEDATA_API_KEY_{i}')
        if key:
            keys.append(key)

    # Keys por grupos
    for g in range(1, 4):
        for i in range(1, 9):
            key = os.environ.get(f'API_KEY_G{g}_{i}')
            if key:
                keys.append(key)

    return keys if keys else [os.environ.get('TWELVEDATA_API_KEY', '')]


def validate_value(symbol, value):
    """Valida que el valor esté en rango esperado"""
    if symbol not in EXPECTED_RANGES:
        return True
    min_val, max_val = EXPECTED_RANGES[symbol]
    return min_val <= value <= max_val


def calculate_z_score(symbol, value):
    """Calcula Z-score para el símbolo"""
    if symbol not in NORM_PARAMS:
        return None
    params = NORM_PARAMS[symbol]
    return (value - params['mean']) / params['std']


# ═══════════════════════════════════════════════════════════════════════════════════
# SCRAPERS
# ═══════════════════════════════════════════════════════════════════════════════════

def fetch_twelvedata(**context):
    """Obtiene datos de TwelveData API"""
    api_keys = get_twelvedata_api_keys()
    results = {}
    errors = []

    key_index = 0

    for local_symbol, td_symbol in TWELVEDATA_SYMBOLS.items():
        try:
            api_key = api_keys[key_index % len(api_keys)]
            key_index += 1

            url = f"https://api.twelvedata.com/price?symbol={td_symbol}&apikey={api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()

            if 'price' in data:
                value = float(data['price'])

                if validate_value(local_symbol, value):
                    results[local_symbol] = {
                        'value': value,
                        'source': 'twelvedata',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    print(f"TwelveData {local_symbol}: {value}")
                else:
                    errors.append(f"{local_symbol}: valor fuera de rango ({value})")
            else:
                errors.append(f"{local_symbol}: {data.get('message', 'error desconocido')}")

        except Exception as e:
            errors.append(f"{local_symbol}: {str(e)}")

    context['ti'].xcom_push(key='twelvedata_results', value=results)
    context['ti'].xcom_push(key='twelvedata_errors', value=errors)

    return {'fetched': len(results), 'errors': len(errors)}


def fetch_fred(**context):
    """Obtiene datos de FRED API"""
    results = {}
    errors = []

    fred_api_key = os.environ.get('FRED_API_KEY', '')

    for local_symbol, fred_id in FRED_SERIES.items():
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': fred_id,
                'api_key': fred_api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 1
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if 'observations' in data and len(data['observations']) > 0:
                value = float(data['observations'][0]['value'])

                if validate_value(local_symbol, value):
                    results[local_symbol] = {
                        'value': value,
                        'source': 'fred',
                        'date': data['observations'][0]['date'],
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    print(f"FRED {local_symbol}: {value}")
                else:
                    errors.append(f"{local_symbol}: valor fuera de rango ({value})")
            else:
                errors.append(f"{local_symbol}: sin datos")

        except Exception as e:
            errors.append(f"{local_symbol}: {str(e)}")

    context['ti'].xcom_push(key='fred_results', value=results)
    context['ti'].xcom_push(key='fred_errors', value=errors)

    return {'fetched': len(results), 'errors': len(errors)}


def scrape_embi(**context):
    """Obtiene EMBI Colombia desde BCRP Perú"""
    import cloudscraper

    result = {}
    errors = []

    try:
        url = "https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados/PD04715XD/html/"

        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, timeout=15)

        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Buscar tabla de datos
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in reversed(rows):  # Último dato primero
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        try:
                            value_text = cells[1].text.strip().replace(',', '.')
                            value = float(value_text)

                            if validate_value('EMBI', value):
                                result = {
                                    'value': value,
                                    'source': 'bcrp',
                                    'timestamp': datetime.utcnow().isoformat()
                                }
                                print(f"EMBI Colombia: {value}")
                                break
                        except ValueError:
                            continue
                if result:
                    break
        else:
            errors.append(f"EMBI: HTTP {response.status_code}")

    except ImportError:
        # cloudscraper no disponible - usar fallback
        errors.append("EMBI: cloudscraper no disponible")

    except Exception as e:
        errors.append(f"EMBI: {str(e)}")

    context['ti'].xcom_push(key='embi_result', value=result)
    context['ti'].xcom_push(key='embi_errors', value=errors)

    return {'fetched': 1 if result else 0, 'errors': len(errors)}


def get_previous_values(**context):
    """Obtiene valores anteriores para calcular cambios"""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT
                dxy, vix, embi, brent,
                timestamp_utc
            FROM dw.fact_macro_realtime
            WHERE timestamp_utc > NOW() - INTERVAL '2 days'
            ORDER BY timestamp_utc DESC
            LIMIT 1
        """)

        row = cur.fetchone()

        if row:
            prev_values = {
                'dxy': float(row[0]) if row[0] else None,
                'vix': float(row[1]) if row[1] else None,
                'embi': float(row[2]) if row[2] else None,
                'brent': float(row[3]) if row[3] else None,
            }
        else:
            prev_values = {}

        context['ti'].xcom_push(key='previous_values', value=prev_values)

    finally:
        cur.close()
        conn.close()


def consolidate_and_calculate(**context):
    """Consolida datos y calcula métricas derivadas"""
    twelvedata = context['ti'].xcom_pull(key='twelvedata_results') or {}
    fred = context['ti'].xcom_pull(key='fred_results') or {}
    embi = context['ti'].xcom_pull(key='embi_result') or {}
    prev_values = context['ti'].xcom_pull(key='previous_values') or {}

    consolidated = {
        'timestamp_utc': datetime.utcnow().isoformat(),
        'scrape_run_id': context['dag_run'].run_id,

        # Valores principales
        'dxy': twelvedata.get('DXY', {}).get('value'),
        'vix': twelvedata.get('VIX', {}).get('value'),
        'embi': embi.get('value'),
        'brent': twelvedata.get('BRENT', {}).get('value'),
        'wti': twelvedata.get('WTI', {}).get('value'),
        'gold': twelvedata.get('GOLD', {}).get('value'),
        'usdmxn': twelvedata.get('USDMXN', {}).get('value'),
        'usdclp': twelvedata.get('USDCLP', {}).get('value'),

        # Tasas
        'fed_funds': fred.get('FEDFUNDS', {}).get('value'),
        'ust_2y': fred.get('DGS2', {}).get('value'),
        'ust_10y': fred.get('DGS10', {}).get('value'),
    }

    # Calcular cambios vs día anterior
    if consolidated['dxy'] and prev_values.get('dxy'):
        consolidated['dxy_change_1d'] = (consolidated['dxy'] - prev_values['dxy']) / prev_values['dxy']
    else:
        consolidated['dxy_change_1d'] = 0

    if consolidated['vix'] and prev_values.get('vix'):
        consolidated['vix_change_1d'] = (consolidated['vix'] - prev_values['vix']) / prev_values['vix']
    else:
        consolidated['vix_change_1d'] = 0

    if consolidated['embi'] and prev_values.get('embi'):
        consolidated['embi_change_1d'] = (consolidated['embi'] - prev_values['embi']) / prev_values['embi']
    else:
        consolidated['embi_change_1d'] = 0

    if consolidated['brent'] and prev_values.get('brent'):
        consolidated['brent_change_1d'] = (consolidated['brent'] - prev_values['brent']) / prev_values['brent']
    else:
        consolidated['brent_change_1d'] = 0

    # Calcular Z-scores
    if consolidated['dxy']:
        consolidated['dxy_z'] = calculate_z_score('DXY', consolidated['dxy'])

    if consolidated['vix']:
        consolidated['vix_z'] = calculate_z_score('VIX', consolidated['vix'])
        # Régimen de volatilidad
        if consolidated['vix'] < 15:
            consolidated['vix_regime'] = 0  # Low vol
        elif consolidated['vix'] < 25:
            consolidated['vix_regime'] = 1  # Normal
        elif consolidated['vix'] < 35:
            consolidated['vix_regime'] = 2  # Elevated
        else:
            consolidated['vix_regime'] = 3  # High vol

    if consolidated['embi']:
        consolidated['embi_z'] = calculate_z_score('EMBI', consolidated['embi'])

    # Rate spread (Fed Funds - Treasury 2Y)
    if consolidated['fed_funds'] and consolidated['ust_2y']:
        consolidated['rate_spread'] = consolidated['fed_funds'] - consolidated['ust_2y']

    # Term spread (10Y - 2Y)
    if consolidated['ust_10y'] and consolidated['ust_2y']:
        consolidated['term_spread'] = consolidated['ust_10y'] - consolidated['ust_2y']

    # Contar valores obtenidos vs intentados
    values_obtained = sum(1 for v in consolidated.values() if v is not None and v != 0)
    consolidated['values_changed'] = values_obtained

    print(f"Datos consolidados: {values_obtained} valores obtenidos")
    context['ti'].xcom_push(key='consolidated_data', value=consolidated)

    return {'values_obtained': values_obtained}


def store_macro_data(**context):
    """Almacena datos macro en la base de datos"""
    data = context['ti'].xcom_pull(key='consolidated_data')

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Insertar en fact_macro_realtime
        cur.execute("""
            INSERT INTO dw.fact_macro_realtime (
                timestamp_utc, scrape_run_id,
                dxy, dxy_change_1d, dxy_z,
                vix, vix_change_1d, vix_z, vix_regime,
                embi, embi_change_1d, embi_z,
                brent, brent_change_1d,
                wti, gold, usdmxn, usdclp,
                fed_funds, ust_2y, ust_10y,
                rate_spread, term_spread,
                values_changed, source
            ) VALUES (
                %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s
            ) RETURNING macro_id
        """, (
            data['timestamp_utc'], data['scrape_run_id'],
            data.get('dxy'), data.get('dxy_change_1d'), data.get('dxy_z'),
            data.get('vix'), data.get('vix_change_1d'), data.get('vix_z'), data.get('vix_regime'),
            data.get('embi'), data.get('embi_change_1d'), data.get('embi_z'),
            data.get('brent'), data.get('brent_change_1d'),
            data.get('wti'), data.get('gold'), data.get('usdmxn'), data.get('usdclp'),
            data.get('fed_funds'), data.get('ust_2y'), data.get('ust_10y'),
            data.get('rate_spread'), data.get('term_spread'),
            data.get('values_changed'), 'macro_scraping_dag'
        ))

        macro_id = cur.fetchone()[0]

        # También insertar en macro_ohlcv para compatibilidad
        symbols_to_insert = [
            ('DXY', data.get('dxy')),
            ('VIX', data.get('vix')),
            ('EMBI', data.get('embi')),
            ('BRENT', data.get('brent')),
            ('WTI', data.get('wti')),
            ('GOLD', data.get('gold')),
            ('USDMXN', data.get('usdmxn')),
            ('USDCLP', data.get('usdclp')),
        ]

        for symbol, value in symbols_to_insert:
            if value is not None:
                cur.execute("""
                    INSERT INTO macro_ohlcv (time, symbol, close, source)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (time, symbol) DO UPDATE
                    SET close = EXCLUDED.close, updated_at = NOW()
                """, (datetime.utcnow(), symbol, value, 'macro_scraping_dag'))

        conn.commit()
        print(f"Datos almacenados: macro_id={macro_id}")

        context['ti'].xcom_push(key='macro_id', value=macro_id)

    finally:
        cur.close()
        conn.close()


def check_data_quality(**context):
    """Verifica calidad de datos y genera alertas si es necesario"""
    data = context['ti'].xcom_pull(key='consolidated_data')
    twelvedata_errors = context['ti'].xcom_pull(key='twelvedata_errors') or []
    fred_errors = context['ti'].xcom_pull(key='fred_errors') or []
    embi_errors = context['ti'].xcom_pull(key='embi_errors') or []

    all_errors = twelvedata_errors + fred_errors + embi_errors

    # Verificar datos críticos
    critical_missing = []
    if data.get('dxy') is None:
        critical_missing.append('DXY')
    if data.get('vix') is None:
        critical_missing.append('VIX')
    if data.get('embi') is None:
        critical_missing.append('EMBI')

    # Generar alertas si hay problemas
    if all_errors or critical_missing:
        conn = get_db_connection()
        cur = conn.cursor()

        try:
            if critical_missing:
                cur.execute("""
                    INSERT INTO dw.fact_inference_alerts (
                        alert_type, severity, message, details
                    ) VALUES (%s, %s, %s, %s)
                """, (
                    'DATA_MISSING',
                    'WARNING' if len(critical_missing) < 2 else 'ERROR',
                    f"Datos macro faltantes: {', '.join(critical_missing)}",
                    json.dumps({'missing': critical_missing, 'errors': all_errors})
                ))

            if len(all_errors) > 3:
                cur.execute("""
                    INSERT INTO dw.fact_inference_alerts (
                        alert_type, severity, message, details
                    ) VALUES (%s, %s, %s, %s)
                """, (
                    'SCRAPER_ERROR',
                    'WARNING',
                    f"Múltiples errores en scraping macro: {len(all_errors)} errores",
                    json.dumps({'errors': all_errors})
                ))

            conn.commit()

        finally:
            cur.close()
            conn.close()

    quality_report = {
        'total_errors': len(all_errors),
        'critical_missing': critical_missing,
        'values_obtained': data.get('values_changed', 0)
    }

    print(f"Reporte de calidad: {quality_report}")
    return quality_report


def generate_report(**context):
    """Genera reporte final de la ejecución"""
    data = context['ti'].xcom_pull(key='consolidated_data')
    macro_id = context['ti'].xcom_pull(key='macro_id')

    report = {
        'macro_id': macro_id,
        'timestamp': data.get('timestamp_utc'),
        'values': {
            'DXY': data.get('dxy'),
            'VIX': data.get('vix'),
            'EMBI': data.get('embi'),
            'Brent': data.get('brent'),
        },
        'z_scores': {
            'DXY_z': data.get('dxy_z'),
            'VIX_z': data.get('vix_z'),
            'EMBI_z': data.get('embi_z'),
        },
        'regime': {
            'VIX_regime': data.get('vix_regime'),
        }
    }

    print("=" * 60)
    print("REPORTE DE SCRAPING MACRO")
    print("=" * 60)
    print(f"ID: {macro_id}")
    print(f"Timestamp: {data.get('timestamp_utc')}")
    print(f"\nValores:")
    for k, v in report['values'].items():
        print(f"  {k}: {v}")
    print(f"\nZ-Scores:")
    for k, v in report['z_scores'].items():
        print(f"  {k}: {v:.2f}" if v else f"  {k}: N/A")
    print("=" * 60)

    return report


# ═══════════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════════

def create_macro_scraping_dag(dag_id, schedule, description_suffix):
    """Factory function para crear DAGs de scraping macro en diferentes horarios"""

    dag = DAG(
        dag_id=dag_id,
        default_args=default_args,
        description=f'Scraping de datos macroeconómicos - {description_suffix}',
        schedule_interval=schedule,
        start_date=days_ago(1),
        catchup=False,
        max_active_runs=1,
        tags=['scraping', 'macro', 'l0', 'data_engineering']
    )

    with dag:
        start = EmptyOperator(task_id='start')

        fetch_td = PythonOperator(
            task_id='fetch_twelvedata',
            python_callable=fetch_twelvedata
        )

        fetch_fr = PythonOperator(
            task_id='fetch_fred',
            python_callable=fetch_fred
        )

        fetch_embi = PythonOperator(
            task_id='scrape_embi',
            python_callable=scrape_embi
        )

        get_prev = PythonOperator(
            task_id='get_previous_values',
            python_callable=get_previous_values
        )

        consolidate = PythonOperator(
            task_id='consolidate_and_calculate',
            python_callable=consolidate_and_calculate
        )

        store = PythonOperator(
            task_id='store_macro_data',
            python_callable=store_macro_data
        )

        quality = PythonOperator(
            task_id='check_data_quality',
            python_callable=check_data_quality
        )

        report = PythonOperator(
            task_id='generate_report',
            python_callable=generate_report
        )

        end = EmptyOperator(task_id='end')

        start >> [fetch_td, fetch_fr, fetch_embi, get_prev]
        [fetch_td, fetch_fr, fetch_embi, get_prev] >> consolidate
        consolidate >> store >> quality >> report >> end

    return dag


# ═══════════════════════════════════════════════════════════════════════════════════
# CREAR 3 DAGs PARA LOS 3 HORARIOS
# ═══════════════════════════════════════════════════════════════════════════════════
# Horarios COT → UTC:
#   07:55 COT = 12:55 UTC
#   10:30 COT = 15:30 UTC
#   12:00 COT = 17:00 UTC

# DAG 1: Pre-apertura (7:55 COT)
dag_0755 = create_macro_scraping_dag(
    dag_id='usdcop_m5__00b_l0_macro_scraping_0755',
    schedule='55 12 * * 1-5',
    description_suffix='7:55 COT Pre-apertura'
)

# DAG 2: Mid-morning (10:30 COT)
dag_1030 = create_macro_scraping_dag(
    dag_id='usdcop_m5__00b_l0_macro_scraping_1030',
    schedule='30 15 * * 1-5',
    description_suffix='10:30 COT Mid-morning'
)

# DAG 3: Cierre sesión (12:00 COT)
dag_1200 = create_macro_scraping_dag(
    dag_id='usdcop_m5__00b_l0_macro_scraping_1200',
    schedule='0 17 * * 1-5',
    description_suffix='12:00 COT Cierre sesión'
)

# También crear el DAG principal con el ID original para compatibilidad
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Scraping de datos macroeconómicos - Principal (7:55 COT)',
    schedule_interval='55 12 * * 1-5',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['scraping', 'macro', 'l0', 'data_engineering']
) as dag:

    start = EmptyOperator(task_id='start')

    # Fetch data en paralelo
    fetch_td = PythonOperator(
        task_id='fetch_twelvedata',
        python_callable=fetch_twelvedata
    )

    fetch_fr = PythonOperator(
        task_id='fetch_fred',
        python_callable=fetch_fred
    )

    fetch_embi = PythonOperator(
        task_id='scrape_embi',
        python_callable=scrape_embi
    )

    get_prev = PythonOperator(
        task_id='get_previous_values',
        python_callable=get_previous_values
    )

    # Consolidar y calcular
    consolidate = PythonOperator(
        task_id='consolidate_and_calculate',
        python_callable=consolidate_and_calculate
    )

    # Almacenar
    store = PythonOperator(
        task_id='store_macro_data',
        python_callable=store_macro_data
    )

    # Quality check
    quality = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality
    )

    # Reporte
    report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report
    )

    end = EmptyOperator(task_id='end')

    # Flujo
    start >> [fetch_td, fetch_fr, fetch_embi, get_prev]
    [fetch_td, fetch_fr, fetch_embi, get_prev] >> consolidate
    consolidate >> store >> quality >> report >> end
