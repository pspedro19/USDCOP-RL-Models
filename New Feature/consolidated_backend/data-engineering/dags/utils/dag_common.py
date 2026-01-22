"""
Common utilities for Airflow DAGs
"""
# =============================================================================
# CENTRAL UTILITIES FOR AIRFLOW DAGS
# =============================================================================
# This module is the authoritative source for database and MinIO connections
# in the data-engineering context. All DAGs and scrapers should import from here.
#
# For backend/API code, use:
#   - backend/src/database.py
#   - api/src/config.py
# =============================================================================

import os
import logging
import psycopg2
from contextlib import contextmanager
from datetime import datetime, date
from typing import Union, List
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# COLOMBIAN BUSINESS DAYS FILTER
# =============================================================================

def is_colombian_business_day(check_date: Union[datetime, date]) -> bool:
    """
    Check if a date is a Colombian business day (not weekend, not holiday).

    Args:
        check_date: Date to check

    Returns:
        True if it's a business day, False otherwise
    """
    try:
        from colombian_holidays import is_holiday
    except ImportError:
        logger.warning("colombian-holidays not installed, only checking weekends")
        is_holiday = lambda x: False

    # Convert to datetime if needed
    if isinstance(check_date, date) and not isinstance(check_date, datetime):
        check_date = datetime.combine(check_date, datetime.min.time())

    # Check weekend (Saturday=5, Sunday=6)
    if check_date.weekday() >= 5:
        return False

    # Check Colombian holiday
    if is_holiday(check_date):
        return False

    return True


def filter_business_days(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Filter DataFrame to only include Colombian business days.

    Args:
        df: DataFrame with a date column
        date_column: Name of the date column

    Returns:
        Filtered DataFrame with only business days
    """
    if df.empty:
        return df

    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Apply filter
    mask = df[date_column].apply(is_colombian_business_day)
    filtered_df = df[mask].copy()

    removed_count = len(df) - len(filtered_df)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} non-business day records from {len(df)} total")

    return filtered_df


def get_business_days_range(start_date: date, end_date: date) -> List[date]:
    """
    Get list of Colombian business days between two dates.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of business days
    """
    business_days = []
    current = start_date

    while current <= end_date:
        if is_colombian_business_day(current):
            business_days.append(current)
        current = current + pd.Timedelta(days=1)

    return business_days


def clean_non_business_days_from_table(table_name: str, date_column: str = 'date') -> int:
    """
    Remove non-business day records from a PostgreSQL table.

    Args:
        table_name: Full table name (schema.table)
        date_column: Name of the date column

    Returns:
        Number of records deleted
    """
    try:
        from colombian_holidays import list_holidays
    except ImportError:
        logger.error("colombian-holidays not installed")
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get date range from table
        cursor.execute(f"SELECT MIN({date_column}), MAX({date_column}) FROM {table_name}")
        min_date, max_date = cursor.fetchone()

        if not min_date or not max_date:
            logger.info(f"Table {table_name} is empty")
            return 0

        # Get all holidays for the years in range
        holidays_set = set()
        for year in range(min_date.year, max_date.year + 1):
            year_holidays = list_holidays(year)
            for (month, day), name in year_holidays.items():
                holidays_set.add(date(year, month, day))

        # Convert to list of strings for SQL
        holiday_dates = [h.strftime('%Y-%m-%d') for h in holidays_set]

        # Delete weekends
        cursor.execute(f"""
            DELETE FROM {table_name}
            WHERE EXTRACT(DOW FROM {date_column}) IN (0, 6)
        """)
        weekend_deleted = cursor.rowcount

        # Delete holidays
        if holiday_dates:
            cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {date_column}::date = ANY(%s::date[])
            """, (holiday_dates,))
            holiday_deleted = cursor.rowcount
        else:
            holiday_deleted = 0

        conn.commit()
        total_deleted = weekend_deleted + holiday_deleted
        logger.info(f"Cleaned {table_name}: {weekend_deleted} weekend + {holiday_deleted} holiday = {total_deleted} records removed")
        return total_deleted

    except Exception as e:
        conn.rollback()
        logger.error(f"Error cleaning {table_name}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_db_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'pipeline_db'),
        user=os.getenv('POSTGRES_USER', 'pipeline'),
        password=os.getenv('POSTGRES_PASSWORD', 'pipeline_secret')
    )


@contextmanager
def db_connection():
    """Context manager for database connection."""
    conn = None
    try:
        conn = get_db_connection()
        yield conn
    finally:
        if conn:
            conn.close()


def get_minio_client():
    """Get MinIO client."""
    try:
        from minio import Minio

        return Minio(
            endpoint=os.getenv('MINIO_ENDPOINT', 'minio:9000'),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'minio_secret'),
            secure=False
        )
    except ImportError:
        logger.warning("minio package not installed")
        return None


def upsert_to_postgres(df, table_name, conflict_columns, connection=None):
    """
    Upsert DataFrame to PostgreSQL table.

    Args:
        df: pandas DataFrame
        table_name: Target table name
        conflict_columns: List of columns for ON CONFLICT
        connection: Optional existing connection
    """
    from io import StringIO

    close_conn = False
    if connection is None:
        connection = get_db_connection()
        close_conn = True

    try:
        cursor = connection.cursor()

        # Create temp table
        temp_table = f"temp_{table_name}_{os.getpid()}"
        columns = ', '.join(df.columns)

        # Copy data to temp table
        cursor.execute(f"CREATE TEMP TABLE {temp_table} (LIKE {table_name} INCLUDING ALL)")

        # Use COPY for fast insert
        output = StringIO()
        df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
        output.seek(0)

        cursor.copy_from(output, temp_table, columns=tuple(df.columns), null='\\N')

        # Upsert from temp to main table
        conflict_cols = ', '.join(conflict_columns)
        update_cols = ', '.join([f"{c} = EXCLUDED.{c}" for c in df.columns if c not in conflict_columns])

        cursor.execute(f"""
            INSERT INTO {table_name} ({columns})
            SELECT {columns} FROM {temp_table}
            ON CONFLICT ({conflict_cols})
            DO UPDATE SET {update_cols}
        """)

        # Cleanup
        cursor.execute(f"DROP TABLE {temp_table}")
        connection.commit()

        logger.info(f"Upserted {len(df)} rows to {table_name}")

    except Exception as e:
        connection.rollback()
        logger.error(f"Error upserting to {table_name}: {e}")
        raise
    finally:
        cursor.close()
        if close_conn:
            connection.close()


def save_to_minio(local_path, bucket, object_name):
    """Save file to MinIO."""
    client = get_minio_client()
    if client is None:
        logger.warning("MinIO client not available")
        return False

    try:
        # Ensure bucket exists
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

        client.fput_object(bucket, object_name, local_path)
        logger.info(f"Saved {object_name} to MinIO bucket {bucket}")
        return True

    except Exception as e:
        logger.error(f"Error saving to MinIO: {e}")
        return False


def load_from_minio(bucket, object_name, local_path):
    """Load file from MinIO."""
    client = get_minio_client()
    if client is None:
        logger.warning("MinIO client not available")
        return False

    try:
        client.fget_object(bucket, object_name, local_path)
        logger.info(f"Loaded {object_name} from MinIO bucket {bucket}")
        return True

    except Exception as e:
        logger.error(f"Error loading from MinIO: {e}")
        return False
