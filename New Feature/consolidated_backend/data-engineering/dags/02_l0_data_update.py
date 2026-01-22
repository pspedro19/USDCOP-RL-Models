"""
L0 - Data Update DAG (INCREMENTAL)
===================================
Layer 0: Actualización INCREMENTAL de datos USD/COP.

FLUJO INCREMENTAL:
1. Consulta la última fecha en core.usdcop_historical
2. Scrape solo datos nuevos desde esa fecha
3. UPSERT incremental (acumula semana a semana)
4. Aplica forward fill basado en frecuencias de publicación
5. Valida cobertura de datos

Source: USD/COP OHLCV from Investing.com

Schedule: Lunes 7:30 AM COT (12:30 UTC) - Semanal antes de inferencia
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import requests
import os
import time
import logging

# Import business days filter utilities
from utils.dag_common import is_colombian_business_day, clean_non_business_days_from_table

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'pipeline_db'),
        user=os.getenv('POSTGRES_USER', 'pipeline'),
        password=os.getenv('POSTGRES_PASSWORD', 'pipeline_secret')
    )


# Forward fill rules based on publication frequency
FFILL_RULES = {
    # Daily data - max 3 days ffill (weekends + holiday)
    'daily': {
        'max_ffill_days': 3,
        'columns': [
            'dxy', 'vix', 'wti', 'brent', 'gold', 'coffee',
            'ust10y', 'ust2y', 'usdmxn', 'usdclp', 'colcap',
            'col10y', 'col5y', 'ibr'
        ]
    },
    # Weekly data - max 7 days ffill
    'weekly': {
        'max_ffill_days': 7,
        'columns': ['reserves']
    },
    # Monthly data - max 35 days ffill
    'monthly': {
        'max_ffill_days': 35,
        'columns': [
            'fedfunds', 'cpi_usa', 'pce', 'unrate', 'indpro', 'm2',
            'consumer_sentiment', 'itcr', 'ipc_col', 'tpm'
        ]
    },
    # Delayed monthly data - max 60 days ffill (45 day publication lag + 15 days)
    'delayed_monthly': {
        'max_ffill_days': 60,
        'columns': ['exports', 'imports', 'terms_trade']
    },
    # Quarterly data - max 100 days ffill
    'quarterly': {
        'max_ffill_days': 100,
        'columns': ['gdp_usa', 'fdi_inflow', 'current_account']
    }
}

# All columns that should receive ffill
ALL_FFILL_COLUMNS = []
for freq_config in FFILL_RULES.values():
    ALL_FFILL_COLUMNS.extend(freq_config['columns'])


# =============================================================================
# USD/COP UPDATE FUNCTIONS
# =============================================================================

def get_last_usdcop_date():
    """Obtiene la última fecha registrada en core.usdcop_historical."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT MAX(date) FROM core.usdcop_historical
        """)
        result = cursor.fetchone()
        if result and result[0]:
            return result[0]
        return None
    except Exception as e:
        logging.warning(f"Error getting last date: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def fetch_usdcop_investing(**context):
    """
    Fetch USD/COP OHLCV from Investing.com - INCREMENTAL.

    Solo obtiene datos NUEVOS desde la última fecha en la base de datos.
    Acumula semana a semana sin recargar todo el histórico.
    """
    try:
        import cloudscraper
        from bs4 import BeautifulSoup
    except ImportError:
        logging.error("cloudscraper/beautifulsoup4 not installed")
        return {'status': 'error', 'message': 'Missing packages'}

    # Obtener última fecha existente
    last_date = get_last_usdcop_date()
    if last_date:
        logging.info(f"[USD/COP] Última fecha en DB: {last_date}")
    else:
        logging.info("[USD/COP] No hay datos previos, carga inicial")
        last_date = datetime.now().date() - timedelta(days=365)  # 1 año atrás para carga inicial

    url = "https://www.investing.com/currencies/usd-cop-historical-data"

    try:
        scraper = cloudscraper.create_scraper()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        logging.info(f"[USD/COP] Fetching from {url}")
        resp = scraper.get(url, headers=headers, timeout=30)

        if resp.status_code != 200:
            logging.error(f"HTTP {resp.status_code}")
            return {'status': 'error', 'http_status': resp.status_code}

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Find the data table
        table = soup.find('table', {'class': 'freeze-column-w-1'})
        if not table:
            tables = soup.find_all('table')
            for t in tables:
                if 'Date' in t.get_text() and 'Close' in t.get_text():
                    table = t
                    break

        if not table:
            logging.error("No data table found")
            return {'status': 'error', 'message': 'Table not found'}

        rows = table.find_all('tr')[1:35]  # Últimos ~35 días disponibles
        records = []
        skipped = 0

        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 6:
                try:
                    date_str = cols[0].get_text(strip=True)
                    date = pd.to_datetime(date_str).date()

                    # FILTRO INCREMENTAL: Solo fechas nuevas
                    if date <= last_date:
                        skipped += 1
                        continue

                    close = float(cols[1].get_text(strip=True).replace(',', ''))
                    open_price = float(cols[2].get_text(strip=True).replace(',', ''))
                    high = float(cols[3].get_text(strip=True).replace(',', ''))
                    low = float(cols[4].get_text(strip=True).replace(',', ''))

                    vol_text = cols[5].get_text(strip=True) if len(cols) > 5 else None
                    volume = None
                    if vol_text and vol_text != '-':
                        vol_text = vol_text.replace(',', '')
                        if vol_text.endswith('K'):
                            volume = int(float(vol_text[:-1]) * 1000)
                        elif vol_text.endswith('M'):
                            volume = int(float(vol_text[:-1]) * 1000000)
                        else:
                            try:
                                volume = int(float(vol_text))
                            except:
                                volume = None

                    records.append({
                        'date': date,
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume
                    })

                except Exception as e:
                    logging.warning(f"Error parsing row: {e}")
                    continue

        logging.info(f"[USD/COP] INCREMENTAL: {len(records)} nuevos, {skipped} ya existentes")

        if records:
            context['ti'].xcom_push(key='usdcop_data', value=records)
            return {'status': 'success', 'new_records': len(records), 'skipped': skipped}
        else:
            logging.info("[USD/COP] Sin datos nuevos (todo actualizado)")
            return {'status': 'up_to_date', 'skipped': skipped}

    except Exception as e:
        logging.error(f"[USD/COP] Error: {e}")
        return {'status': 'error', 'message': str(e)}


def upsert_usdcop(**context):
    """
    Upsert USD/COP data to core.usdcop_historical.
    Calculates returns and volatility metrics.
    """
    ti = context['ti']
    records = ti.xcom_pull(key='usdcop_data')

    if not records:
        logging.warning("No USD/COP data to upsert")
        return {'status': 'no_data'}

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get existing data for return calculations
        cursor.execute("""
            SELECT date, close_price
            FROM core.usdcop_historical
            ORDER BY date DESC
            LIMIT 30
        """)
        existing = {row[0]: row[1] for row in cursor.fetchall()}

        # Prepare records with returns
        upsert_sql = """
            INSERT INTO core.usdcop_historical
            (date, open_price, high_price, low_price, close_price, volume, source)
            VALUES %s
            ON CONFLICT (date) DO UPDATE SET
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                volume = EXCLUDED.volume,
                updated_at = CURRENT_TIMESTAMP
        """

        values = []
        for r in records:
            values.append((
                r['date'],
                r['open'],
                r['high'],
                r['low'],
                r['close'],
                r['volume'],
                'investing_com_daily'
            ))

        execute_values(cursor, upsert_sql, values, page_size=100)

        # Recalculate returns for recent records
        cursor.execute("""
            UPDATE core.usdcop_historical h
            SET
                return_1d = ln(h.close_price / NULLIF(prev.close_price, 0)),
                updated_at = CURRENT_TIMESTAMP
            FROM (
                SELECT date, close_price,
                       LAG(close_price, 1) OVER (ORDER BY date) as prev_close
                FROM core.usdcop_historical
            ) prev
            WHERE h.date = prev.date
              AND prev.prev_close IS NOT NULL
              AND h.date >= CURRENT_DATE - 30
        """)

        conn.commit()
        logging.info(f"[USD/COP] Upserted {len(values)} records")

        return {'status': 'success', 'upserted': len(values)}

    except Exception as e:
        conn.rollback()
        logging.error(f"[USD/COP] Upsert error: {e}")
        return {'status': 'error', 'message': str(e)}
    finally:
        cursor.close()
        conn.close()


# =============================================================================
# FORWARD FILL FUNCTIONS
# =============================================================================

def apply_smart_ffill(**context):
    """
    Apply forward fill to macro_indicators based on publication frequencies.

    Rules:
    - Daily data: max 3 days (weekends/holidays)
    - Weekly data: max 7 days
    - Monthly data: max 35 days
    - Delayed monthly: max 60 days
    - Quarterly data: max 100 days

    This prevents stale data from propagating too far.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    total_filled = 0
    fill_details = []

    try:
        today = datetime.now().date()

        for frequency, config in FFILL_RULES.items():
            max_days = config['max_ffill_days']
            columns = config['columns']

            for column in columns:
                try:
                    # Get the last known value and date
                    cursor.execute("""
                        SELECT date, value
                        FROM core.macro_indicators
                        WHERE indicator_code = %s
                          AND value IS NOT NULL
                        ORDER BY date DESC
                        LIMIT 1
                    """, [column])

                    result = cursor.fetchone()
                    if not result:
                        continue

                    last_date, last_value = result
                    days_since = (today - last_date).days

                    # Only fill if within the allowed window
                    if days_since > 0 and days_since <= max_days:
                        # Find dates that need filling
                        cursor.execute("""
                            SELECT DISTINCT h.date
                            FROM core.usdcop_historical h
                            LEFT JOIN core.macro_indicators m
                              ON m.date = h.date AND m.indicator_code = %s
                            WHERE h.date > %s
                              AND h.date <= CURRENT_DATE
                              AND m.value IS NULL
                            ORDER BY h.date
                        """, [column, last_date])

                        dates_to_fill = [row[0] for row in cursor.fetchall()]

                        if dates_to_fill:
                            # Insert ffill records
                            for fill_date in dates_to_fill:
                                cursor.execute("""
                                    INSERT INTO core.macro_indicators
                                    (date, indicator_code, value, source, frequency)
                                    VALUES (%s, %s, %s, 'ffill', %s)
                                    ON CONFLICT (date, indicator_code) DO UPDATE SET
                                        value = EXCLUDED.value,
                                        source = 'ffill',
                                        updated_at = CURRENT_TIMESTAMP
                                """, [fill_date, column, last_value, frequency])

                            total_filled += len(dates_to_fill)
                            fill_details.append({
                                'column': column,
                                'frequency': frequency,
                                'filled_count': len(dates_to_fill),
                                'last_value': float(last_value),
                                'last_date': str(last_date)
                            })

                            logging.info(f"[FFILL] {column} ({frequency}): {len(dates_to_fill)} days filled with {last_value}")

                    elif days_since > max_days:
                        logging.warning(f"[STALE] {column}: {days_since} days old (max={max_days})")

                except Exception as e:
                    logging.warning(f"[FFILL] Error on {column}: {e}")
                    continue

        conn.commit()
        logging.info(f"[FFILL] Total filled: {total_filled} records")

        context['ti'].xcom_push(key='ffill_details', value=fill_details)
        return {'status': 'success', 'total_filled': total_filled, 'details': fill_details}

    except Exception as e:
        conn.rollback()
        logging.error(f"[FFILL] Error: {e}")
        return {'status': 'error', 'message': str(e)}
    finally:
        cursor.close()
        conn.close()


# =============================================================================
# BUSINESS DAYS CLEANUP
# =============================================================================

def cleanup_non_business_days(**context):
    """
    Remove any non-business day records (weekends and Colombian holidays)
    from the USD/COP historical data.

    This ensures the pipeline only contains valid trading days.
    """
    logging.info("[CLEANUP] Removing non-business day records...")

    total_cleaned = 0

    try:
        deleted = clean_non_business_days_from_table('core.usdcop_historical', 'date')
        total_cleaned += deleted

        logging.info(f"[CLEANUP] Total records removed: {total_cleaned}")
        return {'status': 'success', 'deleted': total_cleaned}

    except Exception as e:
        logging.error(f"[CLEANUP] Error: {e}")
        return {'status': 'error', 'message': str(e)}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_data_coverage(**context):
    """
    Validate data coverage for recent dates.

    Checks:
    1. USD/COP price exists for today/yesterday
    2. Key macro variables are not too stale
    3. Overall coverage percentage
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    issues = []

    try:
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        # Check USD/COP
        cursor.execute("""
            SELECT MAX(date) as last_date
            FROM core.usdcop_historical
        """)
        usdcop_last = cursor.fetchone()[0]

        if usdcop_last:
            days_old = (today - usdcop_last).days
            if days_old > 3:  # Allow for weekends
                issues.append(f"USD/COP is {days_old} days old (last: {usdcop_last})")
                logging.warning(f"[VALIDATE] {issues[-1]}")
            else:
                logging.info(f"[VALIDATE] USD/COP: OK (last: {usdcop_last})")
        else:
            issues.append("USD/COP has no data")

        # Check key macro indicators
        key_indicators = ['dxy', 'vix', 'wti', 'brent', 'embi']

        for indicator in key_indicators:
            cursor.execute("""
                SELECT MAX(date) as last_date
                FROM core.macro_indicators
                WHERE indicator_code = %s
            """, [indicator])
            result = cursor.fetchone()

            if result and result[0]:
                days_old = (today - result[0]).days
                if days_old > 5:
                    issues.append(f"{indicator} is {days_old} days old")
                    logging.warning(f"[VALIDATE] {issues[-1]}")
                else:
                    logging.info(f"[VALIDATE] {indicator}: OK (last: {result[0]})")
            else:
                issues.append(f"{indicator} has no data")

        # Check overall coverage for last week
        cursor.execute("""
            SELECT
                h.date,
                COUNT(DISTINCT m.indicator_code) as indicator_count
            FROM core.usdcop_historical h
            LEFT JOIN core.macro_indicators m ON m.date = h.date
            WHERE h.date >= CURRENT_DATE - 7
            GROUP BY h.date
            ORDER BY h.date DESC
        """)

        coverage_data = cursor.fetchall()

        logging.info("[VALIDATE] Recent coverage:")
        for row in coverage_data:
            logging.info(f"  {row[0]}: {row[1]} indicators")

        context['ti'].xcom_push(key='validation_issues', value=issues)

        status = 'warning' if issues else 'success'
        return {
            'status': status,
            'issues': issues,
            'issue_count': len(issues)
        }

    except Exception as e:
        logging.error(f"[VALIDATE] Error: {e}")
        return {'status': 'error', 'message': str(e)}
    finally:
        cursor.close()
        conn.close()


def generate_data_report(**context):
    """Generate summary report of data status."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        report = []
        report.append("=" * 60)
        report.append("L0 DATA UPDATE REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 60)

        # USD/COP status
        cursor.execute("""
            SELECT COUNT(*), MIN(date), MAX(date)
            FROM core.usdcop_historical
        """)
        count, min_date, max_date = cursor.fetchone()
        report.append(f"\nUSD/COP Historical:")
        report.append(f"  Total records: {count}")
        report.append(f"  Date range: {min_date} to {max_date}")

        # Latest prices
        cursor.execute("""
            SELECT date, close_price
            FROM core.usdcop_historical
            ORDER BY date DESC
            LIMIT 5
        """)
        report.append("\n  Latest prices:")
        for row in cursor.fetchall():
            report.append(f"    {row[0]}: {row[1]:,.2f}")

        # Macro indicators status
        cursor.execute("""
            SELECT COUNT(DISTINCT indicator_code), COUNT(*), MIN(date), MAX(date)
            FROM core.macro_indicators
        """)
        indicators, total, min_date, max_date = cursor.fetchone()
        report.append(f"\nMacro Indicators:")
        report.append(f"  Distinct indicators: {indicators}")
        report.append(f"  Total records: {total}")
        report.append(f"  Date range: {min_date} to {max_date}")

        # FFILL details
        ffill_details = context['ti'].xcom_pull(key='ffill_details') or []
        if ffill_details:
            report.append(f"\nForward Fill Applied:")
            for detail in ffill_details:
                report.append(f"  {detail['column']}: {detail['filled_count']} days")

        # Validation issues
        issues = context['ti'].xcom_pull(key='validation_issues') or []
        if issues:
            report.append(f"\nValidation Issues:")
            for issue in issues:
                report.append(f"  - {issue}")
        else:
            report.append("\nValidation: All checks passed")

        report.append("\n" + "=" * 60)

        report_text = "\n".join(report)
        logging.info(report_text)

        return {'status': 'success', 'report': report_text}

    except Exception as e:
        logging.error(f"[REPORT] Error: {e}")
        return {'status': 'error', 'message': str(e)}
    finally:
        cursor.close()
        conn.close()


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

DAG_ID = "02_l0_data_update"

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Layer 0: Raw data update with smart forward fill',
    schedule_interval='30 12 * * 1',  # Lunes 7:30 AM COT (12:30 UTC) - Semanal antes de inferencia
    catchup=False,
    max_active_runs=1,
    tags=['layer0', 'data', 'update', 'weekly'],
)

# Tasks
start = EmptyOperator(task_id='start', dag=dag)

# USD/COP update
t_fetch_usdcop = PythonOperator(
    task_id='fetch_usdcop',
    python_callable=fetch_usdcop_investing,
    provide_context=True,
    dag=dag,
)

t_upsert_usdcop = PythonOperator(
    task_id='upsert_usdcop',
    python_callable=upsert_usdcop,
    provide_context=True,
    dag=dag,
)

# Clean non-business days
t_cleanup_holidays = PythonOperator(
    task_id='cleanup_non_business_days',
    python_callable=cleanup_non_business_days,
    provide_context=True,
    dag=dag,
)

# Forward fill
t_smart_ffill = PythonOperator(
    task_id='apply_smart_ffill',
    python_callable=apply_smart_ffill,
    provide_context=True,
    dag=dag,
)

# Validation
t_validate = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data_coverage,
    provide_context=True,
    dag=dag,
)

# Report
t_report = PythonOperator(
    task_id='generate_report',
    python_callable=generate_data_report,
    provide_context=True,
    dag=dag,
)

end = EmptyOperator(task_id='end', dag=dag)

# Dependencies
#
# USD/COP fetch -> upsert -> cleanup_holidays -> ffill -> validate -> report
#
start >> t_fetch_usdcop >> t_upsert_usdcop >> t_cleanup_holidays >> t_smart_ffill >> t_validate >> t_report >> end
