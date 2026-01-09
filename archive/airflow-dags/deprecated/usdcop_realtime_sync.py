"""
USDCOP Real-time Sync DAG
Syncs data from Redis buffer to PostgreSQL every 5 minutes during market hours
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable
import redis
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
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'catchup': False,
    'max_active_runs': 1
}

# Create DAG
dag = DAG(
    'usdcop_realtime_sync',
    default_args=default_args,
    description='Sync USDCOP real-time data from Redis to PostgreSQL',
    schedule_interval='*/5 8-12 * * 1-5',  # Every 5 min, 8AM-12PM, Mon-Fri COT
    tags=['usdcop', 'realtime', 'sync'],
    is_paused_upon_creation=False
)

def get_redis_connection():
    """Get Redis connection"""
    redis_host = Variable.get("REDIS_HOST", default_var="redis")
    redis_port = int(Variable.get("REDIS_PORT", default_var=6379))
    redis_password = Variable.get("REDIS_PASSWORD", default_var="redis123")

    return redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True,
        db=0
    )

def is_market_hours(**context):
    """Check if current time is within market hours"""
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

    return is_open

def get_redis_buffer_data(**context):
    """Retrieve and clear data from Redis buffer"""
    try:
        redis_client = get_redis_connection()

        # Test connection
        redis_client.ping()

        # Get all buffered data
        buffer_key = "usdcop:5min_buffer"
        buffered_data = redis_client.lrange(buffer_key, 0, -1)

        # Clear the buffer
        if buffered_data:
            redis_client.delete(buffer_key)
            logging.info(f"Retrieved {len(buffered_data)} records from Redis buffer")
        else:
            logging.info("No data in Redis buffer")

        # Parse JSON data
        parsed_data = []
        for item in buffered_data:
            try:
                parsed_item = json.loads(item)
                parsed_data.append(parsed_item)
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON data: {e}")

        # Store in XCom for next task
        context['task_instance'].xcom_push(key='buffer_data', value=parsed_data)

        return len(parsed_data)

    except Exception as e:
        logging.error(f"Error getting Redis buffer data: {e}")
        raise

def validate_realtime_data(**context):
    """Validate and clean real-time data"""
    try:
        # Get data from previous task
        buffer_data = context['task_instance'].xcom_pull(key='buffer_data')

        if not buffer_data:
            logging.info("No data to validate")
            return 0

        cleaned_data = []

        for record in buffer_data:
            try:
                # Validate required fields
                if not all(key in record for key in ['symbol', 'timestamp', 'last']):
                    logging.warning(f"Invalid record missing required fields: {record}")
                    continue

                # Validate data types and ranges
                price = float(record['last'])
                if price <= 0 or price > 10000:  # Reasonable range for USDCOP
                    logging.warning(f"Invalid price: {price}")
                    continue

                # Normalize timestamp
                timestamp_str = record['timestamp']
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'

                timestamp = datetime.fromisoformat(timestamp_str)

                # Clean record
                clean_record = {
                    'symbol': record['symbol'],
                    'datetime': timestamp,
                    'open': float(record.get('open', price)),
                    'high': float(record.get('high', price)),
                    'low': float(record.get('low', price)),
                    'close': price,
                    'volume': int(record.get('volume', 0)),
                    'source': record.get('source', 'websocket'),
                    'trading_session': True
                }

                cleaned_data.append(clean_record)

            except Exception as e:
                logging.error(f"Error validating record {record}: {e}")
                continue

        # Store cleaned data
        context['task_instance'].xcom_push(key='cleaned_data', value=cleaned_data)

        logging.info(f"Validated {len(cleaned_data)} records out of {len(buffer_data)}")
        return len(cleaned_data)

    except Exception as e:
        logging.error(f"Error validating data: {e}")
        raise

def sync_to_postgresql(**context):
    """Sync cleaned data to PostgreSQL with UPSERT"""
    try:
        # Get cleaned data
        cleaned_data = context['task_instance'].xcom_pull(key='cleaned_data')

        if not cleaned_data:
            logging.info("No cleaned data to sync")
            return 0

        # Get PostgreSQL connection
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

        # UPSERT query to avoid duplicates
        upsert_query = """
        INSERT INTO market_data (symbol, datetime, open, high, low, close, volume, source, trading_session, timeframe)
        VALUES (%(symbol)s, %(datetime)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(source)s, %(trading_session)s, '5min')
        ON CONFLICT (symbol, datetime, timeframe, source)
        DO UPDATE SET
            close = EXCLUDED.close,
            high = GREATEST(market_data.high, EXCLUDED.high),
            low = LEAST(market_data.low, EXCLUDED.low),
            volume = market_data.volume + EXCLUDED.volume,
            updated_at = NOW()
        """

        # Execute batch insert
        inserted_count = 0
        for record in cleaned_data:
            try:
                postgres_hook.run(upsert_query, parameters=record)
                inserted_count += 1
            except Exception as e:
                logging.error(f"Error inserting record {record}: {e}")

        logging.info(f"Successfully synced {inserted_count} records to PostgreSQL")

        # Update sync stats in Redis
        try:
            redis_client = get_redis_connection()
            sync_stats = {
                'last_sync': datetime.now(COT_TZ).isoformat(),
                'records_synced': inserted_count,
                'total_records': len(cleaned_data)
            }
            redis_client.setex(
                'usdcop:sync_stats',
                3600,  # 1 hour TTL
                json.dumps(sync_stats)
            )
        except Exception as e:
            logging.error(f"Error updating sync stats: {e}")

        return inserted_count

    except Exception as e:
        logging.error(f"Error syncing to PostgreSQL: {e}")
        raise

def cleanup_old_realtime_data(**context):
    """Clean up old real-time data (older than 24 hours)"""
    try:
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

        cleanup_query = """
        DELETE FROM realtime_market_data
        WHERE time < NOW() - INTERVAL '24 hours'
        """

        result = postgres_hook.run(cleanup_query)
        logging.info("Cleaned up old real-time data")

    except Exception as e:
        logging.error(f"Error cleaning up old data: {e}")

# Task definitions
check_market_hours_task = PythonOperator(
    task_id='check_market_hours',
    python_callable=is_market_hours,
    dag=dag
)

get_buffer_data_task = PythonOperator(
    task_id='get_redis_buffer_data',
    python_callable=get_redis_buffer_data,
    dag=dag
)

validate_data_task = PythonOperator(
    task_id='validate_realtime_data',
    python_callable=validate_realtime_data,
    dag=dag
)

sync_to_db_task = PythonOperator(
    task_id='sync_to_postgresql',
    python_callable=sync_to_postgresql,
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup_old_data',
    python_callable=cleanup_old_realtime_data,
    dag=dag
)

# Health check query
health_check_task = PostgresOperator(
    task_id='health_check',
    postgres_conn_id='postgres_default',
    sql="""
    INSERT INTO system_health (time, service_name, status, response_time_ms)
    VALUES (NOW(), 'realtime_sync_dag', 'healthy',
        EXTRACT(EPOCH FROM (NOW() - '{{ ds }}'))::int * 1000);
    """,
    dag=dag
)

# Task dependencies
check_market_hours_task >> get_buffer_data_task >> validate_data_task >> sync_to_db_task >> [cleanup_task, health_check_task]