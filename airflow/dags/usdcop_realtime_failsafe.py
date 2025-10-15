"""
USDCOP Real-time Failsafe DAG
Validates data integrity and fills gaps during market hours
Runs every hour to ensure data quality and completeness
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable
import requests
import json
import logging
import pytz
from decimal import Decimal

# Colombia timezone
COT_TZ = pytz.timezone('America/Bogota')

# Default arguments
default_args = {
    'owner': 'realtime-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1, tzinfo=COT_TZ),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    'catchup': False,
    'max_active_runs': 1
}

# Create DAG
dag = DAG(
    'usdcop_realtime_failsafe',
    default_args=default_args,
    description='Gap detection and filling for USDCOP real-time data',
    schedule_interval='0 * * * *',  # Every hour
    tags=['usdcop', 'realtime', 'failsafe', 'quality'],
    is_paused_upon_creation=False
)

def is_market_hours(**context):
    """Check if market is currently open"""
    now = datetime.now(COT_TZ)
    current_time = now.time()
    weekday = now.weekday()  # 0=Monday, 6=Sunday

    # Check if it's a weekday
    if weekday >= 5:  # Saturday or Sunday
        logging.info("Market closed - Weekend")
        return False

    # Check time range (8:00 - 12:55 COT)
    market_start = datetime.strptime("08:00", "%H:%M").time()
    market_end = datetime.strptime("12:55", "%H:%M").time()

    is_open = market_start <= current_time <= market_end

    if not is_open:
        logging.info(f"Market closed - Current time {current_time} outside market hours")
        return "Market closed - no action needed"

    return True

def detect_data_gaps(**context):
    """Detect gaps in real-time data for the last hour"""
    try:
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

        # Check for gaps in the last hour
        now = datetime.now(COT_TZ)
        one_hour_ago = now - timedelta(hours=1)

        gap_detection_query = """
        WITH expected_times AS (
            SELECT generate_series(
                date_trunc('minute', %s),
                date_trunc('minute', %s),
                interval '1 minute'
            ) AS expected_time
        ),
        actual_data AS (
            SELECT DISTINCT date_trunc('minute', datetime) AS actual_time
            FROM market_data
            WHERE symbol = 'USDCOP'
            AND datetime >= %s
            AND datetime <= %s
            AND trading_session = true
        )
        SELECT et.expected_time
        FROM expected_times et
        LEFT JOIN actual_data ad ON et.expected_time = ad.actual_time
        WHERE ad.actual_time IS NULL
        AND EXTRACT(dow FROM et.expected_time) BETWEEN 1 AND 5  -- Monday to Friday
        AND EXTRACT(hour FROM et.expected_time) BETWEEN 8 AND 12  -- Market hours
        ORDER BY et.expected_time;
        """

        gaps = postgres_hook.get_records(
            gap_detection_query,
            parameters=[one_hour_ago, now, one_hour_ago, now]
        )

        gap_periods = []
        if gaps:
            for gap in gaps:
                gap_time = gap[0]
                gap_periods.append({
                    'start_time': gap_time.isoformat(),
                    'end_time': (gap_time + timedelta(minutes=1)).isoformat()
                })

        logging.info(f"Found {len(gap_periods)} data gaps in the last hour")

        # Store gaps in XCom for next task
        context['task_instance'].xcom_push(key='data_gaps', value=gap_periods)

        return len(gap_periods)

    except Exception as e:
        logging.error(f"Error detecting data gaps: {e}")
        raise

def fetch_twelvedata_backup(**context):
    """Fetch backup data from TwelveData API for detected gaps"""
    try:
        gaps = context['task_instance'].xcom_pull(key='data_gaps')

        if not gaps:
            logging.info("No gaps detected, no backup data needed")
            return 0

        # Get TwelveData API key
        api_keys = [
            Variable.get("TWELVEDATA_API_KEY_1", default_var=None),
            Variable.get("TWELVEDATA_API_KEY_2", default_var=None),
            Variable.get("TWELVEDATA_API_KEY_3", default_var=None),
        ]

        api_key = next((key for key in api_keys if key), None)
        if not api_key:
            logging.error("No TwelveData API key available")
            raise ValueError("No API key available")

        backup_data = []

        for gap in gaps:
            try:
                # Convert gap times
                start_time = datetime.fromisoformat(gap['start_time'])
                end_time = datetime.fromisoformat(gap['end_time'])

                # TwelveData time series API call
                url = "https://api.twelvedata.com/time_series"
                params = {
                    'symbol': 'USD/COP',
                    'interval': '1min',
                    'start_date': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'apikey': api_key,
                    'format': 'JSON'
                }

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if 'values' in data and data['values']:
                    for value in data['values']:
                        backup_record = {
                            'symbol': 'USDCOP',
                            'datetime': datetime.strptime(value['datetime'], '%Y-%m-%d %H:%M:%S'),
                            'open': float(value['open']),
                            'high': float(value['high']),
                            'low': float(value['low']),
                            'close': float(value['close']),
                            'volume': int(value.get('volume', 0)),
                            'source': 'twelvedata_backup',
                            'trading_session': True,
                            'timeframe': '1min'
                        }
                        backup_data.append(backup_record)

                # Rate limiting - wait between requests
                import time
                time.sleep(1)

            except Exception as e:
                logging.error(f"Error fetching backup data for gap {gap}: {e}")

        logging.info(f"Fetched {len(backup_data)} backup records")

        # Store backup data
        context['task_instance'].xcom_push(key='backup_data', value=backup_data)

        return len(backup_data)

    except Exception as e:
        logging.error(f"Error fetching backup data: {e}")
        raise

def fill_gaps_in_postgresql(**context):
    """Fill detected gaps with backup data"""
    try:
        backup_data = context['task_instance'].xcom_pull(key='backup_data')

        if not backup_data:
            logging.info("No backup data to insert")
            return 0

        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

        # Insert backup data with gap_filled flag
        insert_query = """
        INSERT INTO market_data (symbol, datetime, open, high, low, close, volume, source, trading_session, timeframe, gap_filled)
        VALUES (%(symbol)s, %(datetime)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(source)s, %(trading_session)s, %(timeframe)s, true)
        ON CONFLICT (symbol, datetime, timeframe, source)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            gap_filled = true,
            updated_at = NOW()
        """

        inserted_count = 0
        for record in backup_data:
            try:
                postgres_hook.run(insert_query, parameters=record)
                inserted_count += 1
            except Exception as e:
                logging.error(f"Error inserting backup record {record}: {e}")

        logging.info(f"Successfully filled {inserted_count} gaps")
        return inserted_count

    except Exception as e:
        logging.error(f"Error filling gaps: {e}")
        raise

def validate_data_quality(**context):
    """Validate data quality for the last hour"""
    try:
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

        now = datetime.now(COT_TZ)
        one_hour_ago = now - timedelta(hours=1)

        # Data quality checks
        quality_checks = {
            'total_records': """
                SELECT COUNT(*) FROM market_data
                WHERE symbol = 'USDCOP'
                AND datetime >= %s AND datetime <= %s
                AND trading_session = true
            """,
            'price_anomalies': """
                SELECT COUNT(*) FROM market_data
                WHERE symbol = 'USDCOP'
                AND datetime >= %s AND datetime <= %s
                AND (close <= 0 OR close > 10000 OR high < low)
            """,
            'missing_volume': """
                SELECT COUNT(*) FROM market_data
                WHERE symbol = 'USDCOP'
                AND datetime >= %s AND datetime <= %s
                AND volume IS NULL
            """,
            'gap_filled_count': """
                SELECT COUNT(*) FROM market_data
                WHERE symbol = 'USDCOP'
                AND datetime >= %s AND datetime <= %s
                AND gap_filled = true
            """
        }

        quality_results = {}
        for check_name, query in quality_checks.items():
            result = postgres_hook.get_first(query, parameters=[one_hour_ago, now])
            quality_results[check_name] = result[0] if result else 0

        # Calculate quality percentage
        total_expected = 60  # 60 minutes per hour
        total_actual = quality_results['total_records']
        quality_percentage = (total_actual / total_expected * 100) if total_expected > 0 else 0

        quality_report = {
            'timestamp': now.isoformat(),
            'period': f"{one_hour_ago.isoformat()} to {now.isoformat()}",
            'quality_percentage': round(quality_percentage, 2),
            'total_records': total_actual,
            'expected_records': total_expected,
            'price_anomalies': quality_results['price_anomalies'],
            'missing_volume': quality_results['missing_volume'],
            'gaps_filled': quality_results['gap_filled_count']
        }

        logging.info(f"Data quality report: {json.dumps(quality_report, indent=2)}")

        # Store quality report
        context['task_instance'].xcom_push(key='quality_report', value=quality_report)

        return quality_percentage

    except Exception as e:
        logging.error(f"Error validating data quality: {e}")
        raise

def update_health_metrics(**context):
    """Update system health metrics based on quality checks"""
    try:
        quality_report = context['task_instance'].xcom_pull(key='quality_report')

        if not quality_report:
            logging.warning("No quality report available")
            return

        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

        # Determine health status
        quality_percentage = quality_report['quality_percentage']
        if quality_percentage >= 95:
            status = 'healthy'
        elif quality_percentage >= 85:
            status = 'warning'
        else:
            status = 'critical'

        # Insert health metric
        health_query = """
        INSERT INTO system_health (time, service_name, status, response_time_ms, details)
        VALUES (NOW(), 'realtime_failsafe', %s, NULL, %s)
        """

        postgres_hook.run(
            health_query,
            parameters=[status, json.dumps(quality_report)]
        )

        logging.info(f"Updated health metrics with status: {status}")

    except Exception as e:
        logging.error(f"Error updating health metrics: {e}")

# Task definitions
check_market_hours_task = PythonOperator(
    task_id='check_market_hours',
    python_callable=is_market_hours,
    dag=dag
)

detect_gaps_task = PythonOperator(
    task_id='detect_data_gaps',
    python_callable=detect_data_gaps,
    dag=dag
)

fetch_backup_task = PythonOperator(
    task_id='fetch_backup_data',
    python_callable=fetch_twelvedata_backup,
    dag=dag
)

fill_gaps_task = PythonOperator(
    task_id='fill_gaps',
    python_callable=fill_gaps_in_postgresql,
    dag=dag
)

validate_quality_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag
)

update_health_task = PythonOperator(
    task_id='update_health_metrics',
    python_callable=update_health_metrics,
    dag=dag
)

# Task dependencies
check_market_hours_task >> detect_gaps_task >> fetch_backup_task >> fill_gaps_task >> validate_quality_task >> update_health_task