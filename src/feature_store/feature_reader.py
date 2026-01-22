"""
Feature Reader - Read L1 Features from Database
================================================

Provides a clean interface for reading features computed by the L1 pipeline
from the inference_features_5m table.

This class is used by:
- L1FeaturesSensor: To check if features are available
- L5InferenceTask: To read features for inference

Usage:
    from src.feature_store.feature_reader import FeatureReader

    reader = FeatureReader()

    # Check if features exist for a timestamp
    if reader.has_features("USD/COP", timestamp):
        features = reader.get_features("USD/COP", timestamp)

    # Get latest features
    features = reader.get_latest_features("USD/COP")

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import logging
import os
import psycopg2
import numpy as np
import json

logger = logging.getLogger(__name__)


def _get_db_connection():
    """Get database connection using environment variables."""
    password = os.environ.get('POSTGRES_PASSWORD')
    if not password:
        raise ValueError("POSTGRES_PASSWORD environment variable is required")

    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'timescaledb'),
        port=int(os.environ.get('POSTGRES_PORT', '5432')),
        database=os.environ.get('POSTGRES_DB', 'usdcop'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=password
    )


# Feature order matching SSOT (CTR-FEAT-001)
FEATURE_ORDER = (
    "log_ret_5m",
    "log_ret_1h",
    "log_ret_4h",
    "rsi_9",
    "atr_pct",
    "adx_14",
    "dxy_z",
    "dxy_change_1d",
    "vix_z",
    "embi_z",
    "brent_change_1d",
    "rate_spread",
    "usdmxn_change_1d",
    "position",
    "time_normalized",
)


@dataclass
class FeatureRecord:
    """A single feature record from the database."""
    timestamp: datetime
    features: np.ndarray
    norm_stats_hash: Optional[str]
    builder_version: Optional[str]
    age_seconds: float

    def is_fresh(self, max_age_minutes: int = 10) -> bool:
        """Check if features are fresh enough."""
        return self.age_seconds <= (max_age_minutes * 60)


class FeatureReader:
    """
    Read L1 features from the inference_features_5m table.

    This class provides a clean interface for reading pre-computed features
    from the L1 pipeline. It handles:
    - Reading features for a specific timestamp
    - Getting the latest available features
    - Checking feature freshness
    - Validating norm_stats_hash consistency

    Thread Safety:
        Creates new database connections for each operation.
        Safe for concurrent use.

    Example:
        reader = FeatureReader()

        # Get latest features for inference
        record = reader.get_latest_features("USD/COP")
        if record and record.is_fresh():
            observation = record.features
    """

    def __init__(
        self,
        table_name: str = "inference_features_5m",
        cache_table: str = "feature_cache",
    ):
        """
        Initialize FeatureReader.

        Args:
            table_name: Table to read features from
            cache_table: Table with cached feature vectors and metadata
        """
        self.table_name = table_name
        self.cache_table = cache_table

    def _get_connection(self):
        """Get a database connection."""
        return _get_db_connection()

    def has_features(
        self,
        symbol: str,
        timestamp: datetime,
        max_age_minutes: int = 10,
    ) -> bool:
        """
        Check if features exist for a given timestamp.

        Args:
            symbol: Trading symbol (e.g., "USD/COP")
            timestamp: The timestamp to check
            max_age_minutes: Maximum age in minutes to consider valid

        Returns:
            bool: True if valid features exist
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Check for features within time window
            cur.execute(f"""
                SELECT COUNT(*)
                FROM {self.table_name}
                WHERE time >= %s - INTERVAL '{max_age_minutes} minutes'
                  AND time <= %s
            """, (timestamp, timestamp))

            count = cur.fetchone()[0]
            cur.close()
            conn.close()

            return count > 0

        except Exception as e:
            logger.error(f"Error checking feature availability: {e}")
            if conn:
                conn.close()
            return False

    def get_features(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Optional[FeatureRecord]:
        """
        Get features for a specific timestamp.

        Args:
            symbol: Trading symbol (e.g., "USD/COP")
            timestamp: The timestamp to get features for

        Returns:
            FeatureRecord if found, None otherwise
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Get the closest feature record to the timestamp
            cur.execute(f"""
                SELECT
                    time,
                    log_ret_5m, log_ret_1h, log_ret_4h,
                    rsi_9, atr_pct, adx_14,
                    dxy_z, dxy_change_1d, vix_z, embi_z,
                    brent_change_1d, rate_spread, usdmxn_change_1d,
                    position, time_normalized,
                    EXTRACT(EPOCH FROM (NOW() - time)) as age_seconds
                FROM {self.table_name}
                WHERE time <= %s
                ORDER BY time DESC
                LIMIT 1
            """, (timestamp,))

            row = cur.fetchone()

            if not row:
                cur.close()
                conn.close()
                return None

            # Build feature array in canonical order
            features = np.array([
                row[1] or 0.0,   # log_ret_5m
                row[2] or 0.0,   # log_ret_1h
                row[3] or 0.0,   # log_ret_4h
                row[4] or 50.0,  # rsi_9
                row[5] or 0.0,   # atr_pct
                row[6] or 25.0,  # adx_14
                row[7] or 0.0,   # dxy_z
                row[8] or 0.0,   # dxy_change_1d
                row[9] or 0.0,   # vix_z
                row[10] or 0.0,  # embi_z
                row[11] or 0.0,  # brent_change_1d
                row[12] or 0.0,  # rate_spread
                row[13] or 0.0,  # usdmxn_change_1d
                row[14] or 0.0,  # position
                row[15] or 0.5,  # time_normalized
            ], dtype=np.float32)

            # Get metadata from cache if available
            norm_stats_hash = None
            builder_version = None

            try:
                cur.execute(f"""
                    SELECT builder_version
                    FROM {self.cache_table}
                    WHERE timestamp <= %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (timestamp,))
                cache_row = cur.fetchone()
                if cache_row:
                    builder_version = cache_row[0]
            except Exception:
                pass  # Cache table might not exist

            cur.close()
            conn.close()

            return FeatureRecord(
                timestamp=row[0],
                features=features,
                norm_stats_hash=norm_stats_hash,
                builder_version=builder_version,
                age_seconds=row[16] or 0.0,
            )

        except Exception as e:
            logger.error(f"Error getting features: {e}")
            if conn:
                conn.close()
            return None

    def get_latest_features(
        self,
        symbol: str,
        max_age_minutes: int = 10,
    ) -> Optional[FeatureRecord]:
        """
        Get the latest available features.

        Args:
            symbol: Trading symbol (e.g., "USD/COP")
            max_age_minutes: Maximum age in minutes to consider valid

        Returns:
            FeatureRecord if found and fresh, None otherwise
        """
        return self.get_features(symbol, datetime.now())

    def get_feature_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the latest features.

        Returns:
            Dict with builder_version, norm_stats_hash, latest_timestamp
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Get latest timestamp
            cur.execute(f"""
                SELECT MAX(time), COUNT(*)
                FROM {self.table_name}
            """)
            row = cur.fetchone()
            latest_time = row[0]
            total_count = row[1]

            # Get builder version from cache
            builder_version = None
            try:
                cur.execute(f"""
                    SELECT builder_version, timestamp
                    FROM {self.cache_table}
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                cache_row = cur.fetchone()
                if cache_row:
                    builder_version = cache_row[0]
            except Exception:
                pass

            cur.close()
            conn.close()

            return {
                "latest_timestamp": latest_time.isoformat() if latest_time else None,
                "total_records": total_count,
                "builder_version": builder_version,
                "table_name": self.table_name,
            }

        except Exception as e:
            logger.error(f"Error getting feature metadata: {e}")
            if conn:
                conn.close()
            return {}

    def check_norm_stats_hash(
        self,
        expected_hash: str,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if the norm_stats_hash matches expected.

        This is used to validate that inference is using the same
        normalization statistics as training.

        Args:
            expected_hash: The expected hash from training
            timestamp: Optional timestamp to check (defaults to latest)

        Returns:
            Tuple of (matches: bool, actual_hash: Optional[str])
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            if timestamp:
                query = f"""
                    SELECT features_json
                    FROM {self.cache_table}
                    WHERE timestamp <= %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                cur.execute(query, (timestamp,))
            else:
                query = f"""
                    SELECT features_json
                    FROM {self.cache_table}
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                cur.execute(query)

            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                return False, None

            # Extract hash from features_json if present
            try:
                features = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                actual_hash = features.get("norm_stats_hash")
                return actual_hash == expected_hash, actual_hash
            except Exception:
                return False, None

        except Exception as e:
            logger.error(f"Error checking norm_stats_hash: {e}")
            if conn:
                conn.close()
            return False, None


# Singleton instance for convenience
_feature_reader: Optional[FeatureReader] = None


def get_feature_reader() -> FeatureReader:
    """Get the singleton FeatureReader instance."""
    global _feature_reader
    if _feature_reader is None:
        _feature_reader = FeatureReader()
    return _feature_reader


def reset_feature_reader():
    """Reset the singleton instance (for testing)."""
    global _feature_reader
    _feature_reader = None
