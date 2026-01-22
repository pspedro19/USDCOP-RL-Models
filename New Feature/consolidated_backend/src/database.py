# backend/src/database.py
"""
Centralized database utilities for the USD/COP Forecasting Pipeline.

This module provides a single source of truth for database connections,
eliminating duplication across the project.

Usage:
    from src.database import get_db_connection, db_connection, get_minio_client

    # Direct connection
    conn = get_db_connection()

    # Context manager
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Environment Variables
# =============================================================================

def get_postgres_config() -> Dict[str, str]:
    """
    Get PostgreSQL configuration from environment variables.

    Returns:
        Dictionary with host, port, database, user, password
    """
    return {
        'host': os.getenv('POSTGRES_HOST', 'postgres'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'pipeline_db'),
        'user': os.getenv('POSTGRES_USER', 'pipeline'),
        'password': os.getenv('POSTGRES_PASSWORD', 'pipeline_secret')
    }


def parse_database_url(url: str) -> Dict[str, str]:
    """
    Parse PostgreSQL connection URL into components.

    Args:
        url: PostgreSQL connection URL (format: postgresql://user:pass@host:port/db)

    Returns:
        Dictionary with host, port, database, user, password keys
    """
    url = url.replace("postgresql://", "")
    user_pass, host_db = url.split("@")
    user, password = user_pass.split(":")
    host_port, db = host_db.split("/")
    host, port = host_port.split(":") if ":" in host_port else (host_port, "5432")

    return {
        "host": host,
        "port": port,
        "database": db,
        "user": user,
        "password": password
    }


# =============================================================================
# Database Connection
# =============================================================================

def get_db_connection(database_url: Optional[str] = None):
    """
    Get PostgreSQL database connection.

    Args:
        database_url: Optional DATABASE_URL. If not provided, uses env vars.

    Returns:
        psycopg2 connection object

    Raises:
        ImportError: If psycopg2 is not installed
        Exception: If connection fails
    """
    import psycopg2

    if database_url:
        config = parse_database_url(database_url)
    else:
        config = get_postgres_config()

    return psycopg2.connect(
        host=config['host'],
        port=config['port'],
        database=config['database'],
        user=config['user'],
        password=config['password']
    )


@contextmanager
def db_connection(database_url: Optional[str] = None):
    """
    Context manager for database connection.

    Automatically closes connection when done.

    Args:
        database_url: Optional DATABASE_URL

    Yields:
        psycopg2 connection object

    Example:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
    """
    conn = None
    try:
        conn = get_db_connection(database_url)
        yield conn
    finally:
        if conn:
            conn.close()


# =============================================================================
# MinIO Client
# =============================================================================

def get_minio_config() -> Dict[str, str]:
    """
    Get MinIO configuration from environment variables.

    Returns:
        Dictionary with endpoint, access_key, secret_key
    """
    return {
        'endpoint': os.getenv('MINIO_ENDPOINT', 'minio:9000'),
        'access_key': os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
        'secret_key': os.getenv('MINIO_SECRET_KEY', 'minio_secret')
    }


def get_minio_client():
    """
    Get MinIO client.

    Returns:
        Minio client object or None if minio package not installed
    """
    try:
        from minio import Minio

        config = get_minio_config()
        return Minio(
            endpoint=config['endpoint'],
            access_key=config['access_key'],
            secret_key=config['secret_key'],
            secure=False
        )
    except ImportError:
        logger.warning("minio package not installed")
        return None


# =============================================================================
# Utility Functions
# =============================================================================

def upsert_to_postgres(df, table_name: str, conflict_columns: list, connection=None):
    """
    Upsert DataFrame to PostgreSQL table.

    Args:
        df: pandas DataFrame
        table_name: Target table name (with schema if needed)
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
        temp_table = f"temp_{table_name.replace('.', '_')}_{os.getpid()}"
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
        update_cols = ', '.join([
            f"{c} = EXCLUDED.{c}"
            for c in df.columns
            if c not in conflict_columns
        ])

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


def save_to_minio(local_path: str, bucket: str, object_name: str) -> bool:
    """
    Save file to MinIO.

    Args:
        local_path: Path to local file
        bucket: MinIO bucket name
        object_name: Object name in bucket

    Returns:
        True if successful, False otherwise
    """
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


def load_from_minio(bucket: str, object_name: str, local_path: str) -> bool:
    """
    Load file from MinIO.

    Args:
        bucket: MinIO bucket name
        object_name: Object name in bucket
        local_path: Path to save locally

    Returns:
        True if successful, False otherwise
    """
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
