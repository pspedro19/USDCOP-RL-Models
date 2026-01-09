"""
Database Connection Utilities
=============================
Centralized database connection management for all services.

DRY: Replaces 6x duplicated POSTGRES_CONFIG across services.

Features:
    - Environment-based configuration
    - Connection pooling support
    - Query execution helpers
    - Pandas DataFrame integration

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration container"""
    host: str
    port: int
    database: str
    user: str
    password: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for psycopg2.connect()"""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password
        }

    def to_url(self) -> str:
        """Convert to connection URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


def get_db_config() -> DatabaseConfig:
    """
    Get database configuration from environment variables.

    Environment Variables:
        POSTGRES_HOST: Database host (default: usdcop-postgres-timescale)
        POSTGRES_PORT: Database port (default: 5432)
        POSTGRES_DB: Database name (default: usdcop_trading)
        POSTGRES_USER: Database user (default: admin)
        POSTGRES_PASSWORD: Database password (default: admin123)

    Returns:
        DatabaseConfig instance
    """
    return DatabaseConfig(
        host=os.environ.get('POSTGRES_HOST', 'usdcop-postgres-timescale'),
        port=int(os.environ.get('POSTGRES_PORT', '5432')),
        database=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=os.environ.get('POSTGRES_PASSWORD', 'admin123')
    )


# Legacy compatibility: Export as dict
def get_postgres_config() -> Dict[str, Any]:
    """Get PostgreSQL config as dictionary (legacy compatibility)"""
    return get_db_config().to_dict()


# Alias for backward compatibility
POSTGRES_CONFIG = None  # Lazily initialized


def _get_postgres_config_dict() -> Dict[str, Any]:
    """Lazy initialization of POSTGRES_CONFIG"""
    global POSTGRES_CONFIG
    if POSTGRES_CONFIG is None:
        POSTGRES_CONFIG = get_db_config().to_dict()
    return POSTGRES_CONFIG


def get_db_connection(config: Optional[DatabaseConfig] = None) -> psycopg2.extensions.connection:
    """
    Get a database connection.

    Args:
        config: Optional DatabaseConfig. If None, uses environment config.

    Returns:
        psycopg2 connection object

    Usage:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM table")
            results = cur.fetchall()
        finally:
            conn.close()
    """
    if config is None:
        config = get_db_config()

    try:
        return psycopg2.connect(**config.to_dict())
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


# Connection pool singleton
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def get_connection_pool(
    min_connections: int = 1,
    max_connections: int = 10,
    config: Optional[DatabaseConfig] = None
) -> pool.ThreadedConnectionPool:
    """
    Get or create a thread-safe connection pool.

    Args:
        min_connections: Minimum pool size
        max_connections: Maximum pool size
        config: Optional DatabaseConfig

    Returns:
        ThreadedConnectionPool instance
    """
    global _connection_pool

    if _connection_pool is None:
        if config is None:
            config = get_db_config()

        _connection_pool = pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            **config.to_dict()
        )
        logger.info(f"Created connection pool: min={min_connections}, max={max_connections}")

    return _connection_pool


@contextmanager
def get_pooled_connection():
    """
    Context manager for pooled connections.

    Usage:
        with get_pooled_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM table")
    """
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def execute_query(
    query: str,
    params: Optional[tuple] = None,
    config: Optional[DatabaseConfig] = None
) -> List[tuple]:
    """
    Execute a query and return results as list of tuples.

    Args:
        query: SQL query string
        params: Query parameters (optional)
        config: Database config (optional)

    Returns:
        List of result tuples
    """
    conn = get_db_connection(config)
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        if cur.description:  # SELECT query
            return cur.fetchall()
        else:  # INSERT/UPDATE/DELETE
            conn.commit()
            return []
    finally:
        conn.close()


def execute_query_df(
    query: str,
    params: Optional[tuple] = None,
    config: Optional[DatabaseConfig] = None
):
    """
    Execute a query and return results as pandas DataFrame.

    Args:
        query: SQL query string
        params: Query parameters (optional)
        config: Database config (optional)

    Returns:
        pandas DataFrame with results

    Raises:
        ImportError: If pandas is not installed
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for execute_query_df")

    conn = get_db_connection(config)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def close_pool():
    """Close the connection pool (call on shutdown)"""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Connection pool closed")
