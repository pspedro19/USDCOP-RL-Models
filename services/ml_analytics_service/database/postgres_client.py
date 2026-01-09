"""
PostgreSQL Database Client
==========================
Database connection and query execution for ML Analytics Service.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from config import DATABASE_CONFIG

logger = logging.getLogger(__name__)


class PostgresClient:
    """PostgreSQL database client with connection pooling"""

    def __init__(self, min_connections: int = 1, max_connections: int = 10):
        """
        Initialize PostgreSQL client with connection pool.

        Args:
            min_connections: Minimum pool size
            max_connections: Maximum pool size
        """
        self.config = DATABASE_CONFIG.to_dict()
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self._pool = pool.ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                **self.config
            )
            logger.info(
                f"PostgreSQL connection pool initialized: "
                f"min={self.min_connections}, max={self.max_connections}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            psycopg2 connection object
        """
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self._pool.putconn(conn)

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as list of dictionaries.

        Args:
            query: SQL query string
            params: Query parameters (optional)
            fetch: Whether to fetch results (False for INSERT/UPDATE/DELETE)

        Returns:
            List of result dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                if fetch and cur.description:
                    return [dict(row) for row in cur.fetchall()]
                return []

    def execute_single(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a query and return single result as dictionary.

        Args:
            query: SQL query string
            params: Query parameters (optional)

        Returns:
            Result dictionary or None
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                if cur.description:
                    row = cur.fetchone()
                    return dict(row) if row else None
                return None

    def execute_scalar(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> Any:
        """
        Execute a query and return single scalar value.

        Args:
            query: SQL query string
            params: Query parameters (optional)

        Returns:
            Single value
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                return result[0] if result else None

    def get_inference_data(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get inference data from dw.fact_rl_inference table.

        Args:
            model_id: Filter by model ID (optional)
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            limit: Maximum rows to return

        Returns:
            List of inference records
        """
        query = """
            SELECT
                inference_id,
                timestamp_utc,
                timestamp_cot,
                model_id,
                model_version,
                fold_id,
                action_raw,
                action_discretized,
                confidence,
                close_price,
                position_before,
                position_after,
                position_change,
                reward,
                cumulative_reward,
                latency_ms,
                log_ret_5m,
                rsi_9,
                atr_pct,
                adx_14,
                dxy_z,
                vix_z,
                embi_z
            FROM dw.fact_rl_inference
            WHERE 1=1
        """
        params = []

        if model_id:
            query += " AND model_id = %s"
            params.append(model_id)

        if start_time:
            query += " AND timestamp_utc >= %s"
            params.append(start_time)

        if end_time:
            query += " AND timestamp_utc <= %s"
            params.append(end_time)

        query += " ORDER BY timestamp_utc DESC LIMIT %s"
        params.append(limit)

        return self.execute_query(query, tuple(params))

    def get_ohlcv_data(
        self,
        symbol: str = 'USD/COP',
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get OHLCV data from usdcop_m5_ohlcv table.

        Args:
            symbol: Trading symbol
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            limit: Maximum rows to return

        Returns:
            List of OHLCV records
        """
        query = """
            SELECT
                time,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                source
            FROM usdcop_m5_ohlcv
            WHERE symbol = %s
        """
        params = [symbol]

        if start_time:
            query += " AND time >= %s"
            params.append(start_time)

        if end_time:
            query += " AND time <= %s"
            params.append(end_time)

        query += " ORDER BY time DESC LIMIT %s"
        params.append(limit)

        return self.execute_query(query, tuple(params))

    def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = self.execute_scalar("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def close(self):
        """Close connection pool"""
        if self._pool:
            self._pool.closeall()
            logger.info("PostgreSQL connection pool closed")
